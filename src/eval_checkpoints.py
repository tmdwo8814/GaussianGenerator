"""Evaluate saved checkpoints and append auto_eval metrics to an existing run.

Use scripts/eval_re10k.sh for the cluster defaults. This entry point shares the
training configuration and ModelWrapper test hooks, but never calls Trainer.fit.
"""

import json
import os
import re
import shlex
from pathlib import Path

import hydra
import torch
import wandb
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
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


def select_checkpoints(
    checkpoint_dir: Path, steps: list[int]
) -> list[tuple[int, Path]]:
    """Resolve every requested step before opening the W&B run."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not steps or any(type(step) is not int or step <= 0 for step in steps):
        raise ValueError("resume_eval.steps must be a nonempty list of positive integers")
    if len(set(steps)) != len(steps):
        raise ValueError("resume_eval.steps contains duplicate steps")

    requested = set(steps)
    found = {}
    for checkpoint in sorted(checkpoint_dir.glob("*.ckpt")):
        match = re.search(r"step[_=](\d+)", checkpoint.name)
        if match is None or int(match.group(1)) not in requested:
            continue
        step = int(match.group(1))
        if step in found:
            raise ValueError(
                f"Multiple checkpoints for step {step}: {found[step]}, {checkpoint}"
            )
        found[step] = checkpoint

    missing = requested - found.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints for steps {sorted(missing)} in {checkpoint_dir}"
        )
    return [(step, found[step]) for step in sorted(requested)]


def inherit_training_auth(script: Path) -> str:
    """Reuse a literal training-script key without sourcing/executing the script."""
    if os.environ.get("WANDB_API_KEY"):
        return "environment"
    if script.is_file():
        for line in script.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(?:export\s+)?WANDB_API_KEY\s*=\s*(.+)$", line)
            if match is None:
                continue
            try:
                words = shlex.split(match.group(1), comments=True)
            except ValueError:
                continue
            if len(words) != 1:
                continue
            value = words[0]
            fallback = re.fullmatch(r"\$\{WANDB_API_KEY:-([^{}]*)\}", value)
            if fallback:
                value = fallback.group(1)
            if not value or any(c.isspace() or c in "$`\\{}" for c in value):
                continue
            os.environ["WANDB_API_KEY"] = value
            return "training script"
    return "SDK saved login"


def save_evaluation_result(model, step: int, checkpoint: Path, output_dir: Path) -> None:
    """Persist the globally reduced scores, independently of W&B upload."""
    count = int(model._test_metric_count.item())
    if count == 0:
        raise RuntimeError(f"Step {step}: evaluation produced no scored samples")
    metrics = {"auto_eval/ckpt_step": step}
    groups = [("overall", model._test_metric_sums, model._test_metric_count)]
    groups += [(tag, model._test_overlap_sums[tag], model._test_overlap_counts[tag])
               for tag in model._test_overlap_tags]
    for tag, sums, samples in groups:
        if samples.item() == 0:
            continue
        metrics[f"auto_eval/{tag}/num_samples"] = int(samples.item())
        for name, total in sums.items():
            metrics[f"auto_eval/{tag}/{name.removesuffix('_ours')}"] = (total / samples).item()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / "metrics.json"
    temporary = result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"checkpoint": str(checkpoint), "metrics": metrics},
                                    indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(result)
    print(f"[ResumeEval] Completed step {step}; local scores: {result}", flush=True)


def prepare_eval_config(cfg_dict: DictConfig) -> None:
    """Match the sampler and output flags used by src.main's post-training eval."""
    cfg_dict.mode = "test"
    for dataset_name, dataset_cfg in cfg_dict.dataset.items():
        num_context_views = dataset_cfg.view_sampler.get("num_context_views", 2)
        dataset_cfg.view_sampler = {
            "name": "evaluation",
            "index_path": str(
                cfg_dict.trainer.get("auto_eval_index_path")
                or f"assets/evaluation_index_{dataset_name}.json"
            ),
            "num_context_views": num_context_views,
        }
        if not Path(dataset_cfg.view_sampler["index_path"]).is_file():
            raise FileNotFoundError(dataset_cfg.view_sampler["index_path"])
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
    # Do not pass name/config/tags/notes: preserve the original training run.
    return WandbLogger(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        id=wandb_cfg.id,
        resume="must",
        mode="online",
        log_model=False,
        save_dir=output_dir,
    )


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
    print(f"[ResumeEval] Output: {output_dir}; align_pose={cfg.test.align_pose}", flush=True)
    for step, checkpoint in checkpoints:
        print(f"[ResumeEval] Selected {step}: {checkpoint}", flush=True)
    if options.get("dry_run", False):
        print("[ResumeEval] Dry run OK; no Trainer or W&B run was opened.", flush=True)
        return
    auth_source = inherit_training_auth(Path(__file__).resolve().parents[1] / "scripts/train_re10k.sh")
    print(f"[ResumeEval] W&B authentication source: {auth_source}", flush=True)
    if wandb.run is not None:
        raise RuntimeError("Start resumed evaluation in a fresh process, without an active W&B run")
    logger = create_resume_logger(cfg_dict.wandb, output_dir)
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        inference_mode=not cfg.test.align_pose,
    )
    torch.manual_seed(cfg.seed + trainer.global_rank)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    distiller = get_distiller(cfg.train.distiller).eval() if cfg.train.distiller else None
    model = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker=None,
        distiller=distiller,
    )
    model.strict_loading = True

    class EvaluationDataModule(DataModule):
        def setup(self, stage=None):
            # DDP rank is assigned after construction in standalone evaluation.
            self.global_rank = self.trainer.global_rank
            super().setup(stage)

    data_module = EvaluationDataModule(
        cfg.dataset, cfg.data_loader, step_tracker=None, global_rank=trainer.global_rank
    )
    test_output_path = cfg.test.output_path
    exit_code = 1
    try:
        for step, checkpoint in checkpoints:
            if trainer.global_rank == 0:
                print(f"[ResumeEval] Evaluating step {step}: {checkpoint}", flush=True)
            model._auto_eval_step = step
            model._auto_eval_log_prefix = "auto_eval"
            model.test_cfg.output_path = test_output_path / f"step_{step:08d}"
            trainer.test(model, datamodule=data_module, ckpt_path=str(checkpoint))
            if trainer.global_rank == 0:
                save_evaluation_result(model, step, checkpoint, model.test_cfg.output_path)
        exit_code = 0
    finally:
        # Flush the last checkpoint's metrics before the batch job exits.
        # Only rank zero has an active run (initialized by on_test_start).
        if trainer.global_rank == 0 and wandb.run is not None:
            wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    evaluate()
