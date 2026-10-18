"""Evaluate saved checkpoints and append auto_eval metrics to an existing run.

Use scripts/eval_re10k.sh for the cluster defaults. This entry point shares the
training configuration and ModelWrapper test hooks, but never calls Trainer.fit.
"""

import re
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
    data_module = DataModule(
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
        exit_code = 0
    finally:
        # Flush the last checkpoint's metrics before the batch job exits.
        # Only rank zero has an active run (initialized by on_test_start).
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    evaluate()
