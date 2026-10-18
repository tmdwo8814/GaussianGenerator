"""Evaluate saved checkpoints and append auto_eval results to an existing run."""

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.distiller import get_distiller
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def select_checkpoints(checkpoint_dir: Path, steps: list[int]):
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not steps or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("resume_eval.steps must contain positive integers")
    if len(steps) != len(set(steps)):
        raise ValueError("resume_eval.steps contains duplicate steps")
    requested, found = set(steps), {}
    for checkpoint in sorted(checkpoint_dir.glob("*.ckpt")):
        match = re.search(r"step[_=](\d+)", checkpoint.name)
        if match is None or int(match.group(1)) not in requested:
            continue
        step = int(match.group(1))
        if step in found:
            raise ValueError(f"Multiple checkpoints for step {step}: {found[step]}, {checkpoint}")
        found[step] = checkpoint
    missing = requested - found.keys()
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for steps {sorted(missing)}")
    return [(step, found[step]) for step in sorted(requested)]


def prepare_eval_config(cfg_dict: DictConfig) -> None:
    """Apply the same sampler and image options as post-training auto_eval."""
    cfg_dict.mode = "test"
    for dataset_name, dataset_cfg in cfg_dict.dataset.items():
        index_path = Path(
            cfg_dict.trainer.get("auto_eval_index_path")
            or f"assets/evaluation_index_{dataset_name}.json"
        )
        with index_path.open(encoding="utf-8") as handle:
            valid_scenes = {
                scene for scene, entry in json.load(handle).items() if entry is not None
            }
        if not valid_scenes:
            raise ValueError(f"Evaluation index has no valid scenes: {index_path}")
        dataset_scenes = set()
        all_roots_indexed = True
        for root in dataset_cfg.roots:
            test_dir = Path(root) / "test"
            if not test_dir.is_dir() or not any(test_dir.glob("*.torch")):
                raise FileNotFoundError(f"No test .torch chunks found in {test_dir}")
            scene_index = test_dir / "index.json"
            if scene_index.is_file():
                with scene_index.open(encoding="utf-8") as handle:
                    dataset_scenes.update(json.load(handle))
            else:
                all_roots_indexed = False
        if all_roots_indexed and not valid_scenes.intersection(dataset_scenes):
            raise ValueError(
                f"Evaluation index {index_path} shares no scenes with dataset test indices "
                f"under {list(dataset_cfg.roots)}. Check trainer.auto_eval_index_path."
            )
        dataset_cfg.view_sampler = {
            "name": "evaluation",
            "index_path": str(index_path),
            "num_context_views": dataset_cfg.view_sampler.get("num_context_views", 2),
        }
    for flag in ("save_image", "save_video", "save_compare"):
        cfg_dict.test[flag] = cfg_dict.trainer.get(f"auto_eval_{flag}", False)
    if not cfg_dict.test.compute_scores:
        raise ValueError("Resuming evaluation requires test.compute_scores=true")


def create_resume_logger(wandb_cfg: DictConfig, output_dir: Path):
    if wandb_cfg.mode != "online":
        raise ValueError("Resuming evaluation requires wandb.mode=online")
    for key in ("entity", "project", "id"):
        if not wandb_cfg.get(key):
            raise ValueError(f"Resuming evaluation requires wandb.{key}")
    # Preserve the original run's name, config, tags, and notes.
    return WandbLogger(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        id=wandb_cfg.id,
        resume="must",
        mode="online",
        log_model=False,
        save_dir=output_dir,
    )


class ResumeEvalProgress(Callback):
    """Keep progress on disk and send separate, committed W&B progress metrics."""

    def __init__(self, output_dir: Path, expected_run: str):
        self.output_dir = output_dir
        self.expected_run = expected_run
        self.session_batches = 0

    def record(self, rank, event, step=None, **fields):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "rank": rank, "pid": os.getpid(), "event": event, "ckpt_step": step,
            **fields,
        }
        with (self.output_dir / f"progress-rank{rank}.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(record) + "\n")

    def _report(self, trainer, pl_module, event):
        elapsed = time.monotonic() - self.started_at
        step = pl_module._auto_eval_step
        self.record(
            trainer.global_rank, event, step,
            completed_batches=self.completed_batches, elapsed_seconds=elapsed,
        )
        if trainer.global_rank == 0:
            print(
                f"[ResumeEval] {event} | ckpt_step={step}"
                f" | rank0_completed_batches={self.completed_batches}"
                f" | elapsed={elapsed:.1f}s", flush=True,
            )
            # No explicit W&B history step: append after the training history.
            # Final quality metrics remain under auto_eval/* in ModelWrapper.
            trainer.logger.experiment.log({
                "resume_eval/ckpt_step": step,
                "resume_eval/completed_batches": self.completed_batches,
                "resume_eval/session_batches": self.session_batches,
                "resume_eval/elapsed_seconds": elapsed,
            }, commit=True)

    def on_test_start(self, trainer, pl_module):
        self.completed_batches = 0
        self.started_at = time.monotonic()
        if trainer.global_rank == 0:
            run = trainer.logger.experiment
            if run.path != self.expected_run:
                raise RuntimeError(f"Unexpected W&B run: {run.path}; expected {self.expected_run}")
            run.define_metric("resume_eval/session_batches")
            run.define_metric("resume_eval/*", step_metric="resume_eval/session_batches")
            print(f"[ResumeEval] Connected to existing run: {run.url}", flush=True)
        self._report(trainer, pl_module, "started")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.completed_batches += 1
        self.session_batches += 1
        if self.completed_batches == 1 or self.completed_batches % 10 == 0:
            self._report(trainer, pl_module, "progress")

    def on_test_end(self, trainer, pl_module):
        self._report(trainer, pl_module, "local_end")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def evaluate(cfg_dict: DictConfig) -> None:
    options = cfg_dict.get("resume_eval")
    if options is None or not options.get("checkpoint_dir") or not options.get("steps"):
        raise ValueError("Set +resume_eval.checkpoint_dir and +resume_eval.steps")
    if cfg_dict.checkpointing.load is not None:
        raise ValueError("Use resume_eval.checkpoint_dir instead of checkpointing.load")
    checkpoints = select_checkpoints(
        Path(options.checkpoint_dir).expanduser().resolve(), list(options.steps)
    )
    prepare_eval_config(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger = create_resume_logger(cfg_dict.wandb, output_dir)
    expected_run = "/".join(str(cfg_dict.wandb[key]) for key in ("entity", "project", "id"))
    progress = ResumeEvalProgress(output_dir, expected_run)
    trainer = Trainer(
        accelerator="gpu", devices="auto", num_nodes=cfg.trainer.num_nodes,
        strategy=("ddp_find_unused_parameters_true"
                  if torch.cuda.device_count() > 1 else "auto"),
        logger=logger, callbacks=[progress], enable_checkpointing=False,
        enable_progress_bar=False, inference_mode=not cfg.test.align_pose,
    )
    step, exit_code = None, 1
    try:
        progress.record(trainer.global_rank, "initializing_model")
        torch.manual_seed(cfg.seed + trainer.global_rank)
        encoder, visualizer = get_encoder(cfg.model.encoder)
        distiller = get_distiller(cfg.train.distiller).eval() if cfg.train.distiller else None
        model = ModelWrapper(
            cfg.optimizer, cfg.test, cfg.train, encoder, visualizer,
            get_decoder(cfg.model.decoder), get_losses(cfg.loss),
            step_tracker=None, distiller=distiller,
        )
        data_module = DataModule(
            cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=trainer.global_rank
        )
        test_output_path = cfg.test.output_path
        for step, checkpoint in checkpoints:
            progress.record(trainer.global_rank, "loading_checkpoint", step, path=str(checkpoint))
            if trainer.global_rank == 0:
                print(f"[ResumeEval] Evaluating step {step}: {checkpoint}", flush=True)
            model._auto_eval_step = step
            model._auto_eval_log_prefix = "auto_eval"
            model.test_cfg.output_path = test_output_path / f"step_{step:08d}"
            trainer.test(model, datamodule=data_module, ckpt_path=str(checkpoint))
            # ModelWrapper.on_test_end has globally reduced this count on every rank.
            num_samples = int(model._test_metric_count.item())
            if num_samples == 0:
                raise RuntimeError(
                    f"Checkpoint {step}: no test metrics accumulated. Check the dataset roots, "
                    "evaluation index, and skipped examples. Remaining checkpoints were stopped."
                )
            progress.record(trainer.global_rank, "checkpoint_complete", step, num_samples=num_samples)
        exit_code = 0
    except BaseException:
        # Save the original exception before W&B finalization can block or fail.
        progress.record(trainer.global_rank, "failed", step, traceback=traceback.format_exc())
        raise
    finally:
        if wandb.run is not None:
            progress.record(trainer.global_rank, "finishing_wandb", step, exit_code=exit_code)
            wandb.finish(exit_code=exit_code)
            progress.record(trainer.global_rank, "wandb_finished", step, exit_code=exit_code)


if __name__ == "__main__":
    evaluate()
