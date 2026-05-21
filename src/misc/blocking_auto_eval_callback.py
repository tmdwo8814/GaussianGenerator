import json
import zlib
import gc
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import get_cfg
from src.loss import get_losses
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


class BlockingAutoEvalCallback(Callback):
    """
    Training 중 특정 step마다:
      1. checkpoint 저장
      2. fresh eval model 생성
      3. checkpoint load
      4. freeze/eval 상태로 exact multi-GPU evaluation
      5. wandb logging
      6. eval model 삭제
      7. training 재개
    """

    def __init__(self) -> None:
        super().__init__()
        self.eval_every_n_steps = None
        self.eval_cfg = None
        self.eval_data_module = None
        self.eval_total_samples = None
        self.eval_local_total_samples = None

    def _infer_dataset_name(self) -> str:
        dataset_cfg = get_cfg()["dataset"]
        dataset_names = list(dataset_cfg.keys())
        if len(dataset_names) != 1:
            raise ValueError(
                f"BlockingAutoEvalCallback expects exactly one dataset, got: {dataset_names}"
            )
        return dataset_names[0]

    def _infer_eval_index_path(self) -> Path:
        trainer_cfg = get_cfg()["trainer"]

        if trainer_cfg.get("auto_eval_index_path", None) is not None:
            index_path = Path(trainer_cfg["auto_eval_index_path"])
        else:
            dataset_name = self._infer_dataset_name()
            index_path = Path(f"assets/evaluation_index_{dataset_name}.json")

        if not index_path.exists():
            raise FileNotFoundError(
                f"Expected evaluation index at {index_path}, but it does not exist."
            )

        return index_path

    def _build_eval_root_cfg(self):
        cfg_dict = OmegaConf.create(OmegaConf.to_container(get_cfg(), resolve=True))
        trainer_cfg = cfg_dict["trainer"]

        cfg_dict["mode"] = "test"

        index_path = str(self._infer_eval_index_path())

        for _, ds_cfg in cfg_dict["dataset"].items():
            current_vs = ds_cfg.get("view_sampler", {})
            num_context_views = current_vs.get("num_context_views", 2)

            ds_cfg["view_sampler"] = {
                "name": "evaluation",
                "index_path": index_path,
                "num_context_views": num_context_views,
            }

        cfg_dict["test"]["save_image"] = bool(trainer_cfg.get("auto_eval_save_image", False))
        cfg_dict["test"]["save_video"] = bool(trainer_cfg.get("auto_eval_save_video", False))
        cfg_dict["test"]["save_compare"] = bool(trainer_cfg.get("auto_eval_save_compare", False))

        return load_typed_root_config(cfg_dict)

    def _scene_owner_rank(self, scene: str, world_size: int) -> int:
        return zlib.crc32(scene.encode("utf-8")) % max(world_size, 1)

    def _infer_counts(self, index_path: Path, world_size: int, global_rank: int):
        with index_path.open("r") as f:
            index = json.load(f)

        scenes = [scene for scene, entry in index.items() if entry is not None]
        total = len(scenes)

        if world_size <= 1:
            return total, total

        local = sum(
            self._scene_owner_rank(scene, world_size) == global_rank
            for scene in scenes
        )
        return total, local

    def _move_batch_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)

        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}

        if isinstance(batch, list):
            return [self._move_batch_to_device(v, device) for v in batch]

        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(v, device) for v in batch)

        return batch

    def _find_checkpoint_callback(self, trainer):
        for callback in trainer.callbacks:
            if callback.__class__.__name__ == "ModelCheckpoint":
                return callback
        return None

    def _save_eval_checkpoint(self, trainer, pl_module, step: int) -> Path:
        ckpt_callback = self._find_checkpoint_callback(trainer)
        if ckpt_callback is None or ckpt_callback.dirpath is None:
            raise RuntimeError("Could not find ModelCheckpoint callback or dirpath.")

        ckpt_dir = Path(ckpt_callback.dirpath)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"auto_eval_step_{step:0>8}.ckpt"

        if trainer.global_rank == 0:
            state_dict = {
                k: v.detach().cpu()
                for k, v in pl_module.state_dict().items()
            }
            torch.save({"state_dict": state_dict}, ckpt_path)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        return ckpt_path

    def _build_fresh_eval_model(self, ckpt_path: Path, device, step: int) -> ModelWrapper:
        encoder, encoder_visualizer = get_encoder(self.eval_cfg.model.encoder)

        eval_model = ModelWrapper(
            self.eval_cfg.optimizer,
            self.eval_cfg.test,
            self.eval_cfg.train,
            encoder,
            encoder_visualizer,
            get_decoder(self.eval_cfg.model.decoder),
            get_losses(self.eval_cfg.loss),
            step_tracker=None,
            distiller=None,
        )

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        eval_model.load_state_dict(state_dict, strict=True)

        eval_model.to(device)
        eval_model.eval()

        for p in eval_model.parameters():
            p.requires_grad_(False)

        # fresh eval model은 Lightning Trainer에 attach되어 있지 않으므로 rank/step을 직접 저장
        eval_model._auto_eval_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        eval_model._auto_eval_world_size = (
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        )
        eval_model._auto_eval_step = step

        return eval_model

    def on_fit_start(self, trainer, pl_module) -> None:
        trainer_cfg = get_cfg()["trainer"]

        if not trainer_cfg.get("auto_eval", False):
            if trainer.global_rank == 0:
                print("[AutoEval] disabled")
            return

        self.eval_every_n_steps = int(
            trainer_cfg.get(
                "eval_every_n_steps",
                get_cfg()["checkpointing"]["every_n_train_steps"],
            )
        )

        self.eval_cfg = self._build_eval_root_cfg()

        self.eval_data_module = DataModule(
            self.eval_cfg.dataset,
            self.eval_cfg.data_loader,
            step_tracker=None,
            global_rank=trainer.global_rank,
        )

        index_path = self._infer_eval_index_path()
        self.eval_total_samples, self.eval_local_total_samples = self._infer_counts(
            index_path=index_path,
            world_size=max(int(trainer.world_size), 1),
            global_rank=trainer.global_rank,
        )

        if trainer.global_rank == 0:
            print(
                f"[AutoEval] enabled"
                f" | eval_every_n_steps={self.eval_every_n_steps}"
                f" | total_samples={self.eval_total_samples}"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step

        if self.eval_every_n_steps is None or step == 0:
            return

        if step % self.eval_every_n_steps != 0:
            return

        trainer.strategy.barrier("before_auto_eval")

        if trainer.global_rank == 0:
            print(f"\n[AutoEval] step {step}: saving checkpoint and starting fresh-model evaluation")

        ckpt_path = self._save_eval_checkpoint(trainer, pl_module, step)

        was_training = pl_module.training
        eval_model = None

        try:
            # training model은 건드리지 않고, fresh eval model만 사용
            eval_model = self._build_fresh_eval_model(
                ckpt_path=ckpt_path,
                device=pl_module.device,
                step=step,
            )

            eval_model.begin_blocking_eval(
                total_samples=self.eval_total_samples,
                local_total=self.eval_local_total_samples,
            )

            eval_loaders = self.eval_data_module.test_dataloader()
            if not isinstance(eval_loaders, list):
                eval_loaders = [eval_loaders]

            eval_batch_idx = 0
            with torch.no_grad():
                # align_pose 내부에서는 torch.set_grad_enabled(True)를 사용하므로
                # 여기 no_grad는 encoder 등 일반 forward 보호용
                pass

            for loader in eval_loaders:
                for eval_batch in loader:
                    eval_batch = self._move_batch_to_device(eval_batch, pl_module.device)
                    eval_model.test_step(eval_batch, eval_batch_idx)
                    eval_batch_idx += 1

            eval_model.end_blocking_eval(
                log_prefix="auto_eval",
                log_step=step,
                dump_benchmarks=False,
            )

        finally:
            if eval_model is not None:
                del eval_model

            gc.collect()
            torch.cuda.empty_cache()

            if was_training:
                pl_module.train()

        trainer.strategy.barrier("after_auto_eval")

        if trainer.global_rank == 0:
            print(f"[AutoEval] step {step}: done\n")