"""Single-GPU evaluation between optimizer steps, reusing the training model."""

import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule


class DecoderAblationEvalCallback(Callback):
    def __init__(self, cfg_dict, output_dir: Path):
        super().__init__()
        self.cfg_dict = deepcopy(cfg_dict)
        self.output_dir = output_dir
        self.last_eval_step = -1
        self.rows = []

    def on_fit_start(self, trainer, pl_module):
        cfg = self.cfg_dict
        if trainer.world_size != 1 or not cfg.model.encoder.freeze_scene_encoder:
            raise ValueError("Decoder ablation evaluation requires one GPU and frozen scene encoders")
        if not cfg.trainer.auto_eval or not cfg.test.compute_scores or not cfg.test.measure_fps:
            raise ValueError("Decoder ablation requires auto_eval, compute_scores and measure_fps")
        if cfg.train.distiller:
            raise ValueError("Decoder ablation does not use scene distillation")
        self.interval = cfg.trainer.eval_every_n_steps
        if not self.interval or self.interval < 1:
            raise ValueError("eval_every_n_steps must be positive")
        if cfg.trainer.max_steps % self.interval:
            raise ValueError("max_steps must be divisible by eval_every_n_steps")
        # Only the Gaussian prediction decoder may enter the optimizer. The
        # renderer has no trainable parameters; feature heads belong to the scene.
        trainable = [name for name, p in pl_module.named_parameters() if p.requires_grad]
        if not trainable or any(not name.startswith("encoder.gaussian_decoder.")
                                for name in trainable):
            raise ValueError(f"Unexpected trainable parameters: {trainable}")
        eval_cfg = deepcopy(cfg)
        eval_cfg.mode = "test"
        self.indices = {}
        for name, dataset in eval_cfg.dataset.items():
            index_path = Path(cfg.trainer.auto_eval_index_path
                              or f"assets/evaluation_index_{name}.json")
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if not any(value is not None for value in index.values()):
                raise ValueError(f"Empty evaluation index: {index_path}")
            self.indices[name] = str(index_path)
            dataset.view_sampler = {
                "name": "evaluation", "index_path": str(index_path),
                "num_context_views": dataset.view_sampler.num_context_views,
            }
        for flag in ("save_image", "save_video", "save_compare"):
            eval_cfg.test[flag] = cfg.trainer.get(f"auto_eval_{flag}", False)
        typed = load_typed_root_config(eval_cfg)
        self.eval_test_cfg = typed.test
        self.eval_data = DataModule(typed.dataset, typed.data_loader, step_tracker=None)
        # One set of loaders with dedicated RNGs; no training sampler state is
        # changed. Each evaluation iterates the complete fixed test index.
        self.eval_loaders = self.eval_data.test_dataloader()
        if not isinstance(self.eval_loaders, list):
            self.eval_loaders = [self.eval_loaders]
        manifest = {
            "scene_checkpoint": cfg.model.encoder.pretrained_weights,
            "decoder_initialization": "random",
            "seed": cfg.seed,
            "k": cfg.model.encoder.moment_decoder.num_neighbors,
            "knn_query_backend": cfg.model.encoder.moment_decoder.knn_query_backend,
            "evaluation_indices": self.indices,
            "gpu": torch.cuda.get_device_name(pl_module.device),
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        (self.output_dir / "ablation.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"[KNN Ablation] Full evaluation every {self.interval} optimizer steps", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        # Accumulation can trigger this hook more than once at the same step.
        if step == 0 or step % self.interval or step == self.last_eval_step:
            return
        self.last_eval_step = step
        checkpoint = self.output_dir / "checkpoints" / f"step_{step:08d}.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint, weights_only=False)
        modes = {module: module.training for module in pl_module.modules()}
        gradients = {parameter: parameter.requires_grad for parameter in pl_module.parameters()}
        training_test_cfg = pl_module.test_cfg
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        # The evaluation must not change the training RNG stream or gradient
        # flags, including on an exception during camera pose optimization.
        try:
            with torch.random.fork_rng(devices=[pl_module.device.index]):
                pl_module.eval()
                pl_module.requires_grad_(False)
                pl_module.test_cfg = deepcopy(self.eval_test_cfg)
                pl_module.test_cfg.output_path = self.output_dir / "evaluation" / f"step_{step:08d}"
                pl_module._auto_eval_step = step
                pl_module._auto_eval_log_prefix = "auto_eval"
                pl_module.on_test_start()
                print(f"[KNN Ablation] Evaluating step {step}: {checkpoint}", flush=True)
                batch_index = 0
                with torch.no_grad():
                    for loader in self.eval_loaders:
                        for eval_batch in loader:
                            eval_batch = pl_module.transfer_batch_to_device(
                                eval_batch, pl_module.device, 0
                            )
                            pl_module.test_step(eval_batch, batch_index)
                            batch_index += 1
                pl_module.on_test_end()
                metrics = pl_module._test_result_payload
                self.rows.append({
                    "k": self.cfg_dict.model.encoder.moment_decoder.num_neighbors,
                    "step": step,
                    **{name: metrics[f"auto_eval/overall/{name}"]
                       for name in ("psnr", "ssim", "lpips", "fps", "render_fps", "num_samples")},
                    "checkpoint": str(checkpoint),
                })
                result_path = self.output_dir / "metrics.csv"
                temporary = result_path.with_suffix(".csv.tmp")
                with temporary.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                    writer.writeheader()
                    writer.writerows(self.rows)
                temporary.replace(result_path)
        finally:
            for parameter, requires_grad in gradients.items():
                parameter.requires_grad_(requires_grad)
            for module, training in modes.items():
                module.training = training
            pl_module.test_cfg = training_test_cfg
            for attr in ("_auto_eval_step", "_auto_eval_log_prefix"):
                if hasattr(pl_module, attr):
                    delattr(pl_module, attr)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        print(f"[KNN Ablation] Step {step} complete; results: {result_path}", flush=True)
