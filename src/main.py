import os
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

import re


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"

def get_ckpt_step(path: Path) -> int:
    match = re.search(r"step[_=](\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not parse step from checkpoint name: {path.name}")
    return int(match.group(1))


def get_eval_checkpoints(checkpoint_dir: Path, eval_every_n_steps: int) -> list[Path]:
    ckpts = sorted(checkpoint_dir.glob("*.ckpt"), key=get_ckpt_step)
    return [
        ckpt for ckpt in ckpts
        if get_ckpt_step(ckpt) % eval_every_n_steps == 0
    ]


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []

    # Get the Slurm job ID from enviroment variable
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    tags = cfg_dict.wandb.get("tags", [])
    tags += [f"job_id={slurm_job_id}"] if slurm_job_id != "unknown" else []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.name})",
            tags=tags,
            log_model=False,
            save_dir=output_dir,
            notes=f"outputs/{output_dir.parent.name}/{output_dir.name}",
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        if wandb.run is not None:
            wandb.run.log_code("src")

            # Post-training checkpoint evaluation용 x-axis
            wandb.define_metric("auto_eval/ckpt_step")
            wandb.define_metric("auto_eval/*", step_metric="auto_eval/ckpt_step")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=0 if cfg.trainer.val_check_interval is None else 1.0,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        inference_mode=False if cfg.test.align_pose else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu', weights_only=False)
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)

        if cfg.trainer.auto_eval:
            eval_every_n_steps = (
                cfg.trainer.eval_every_n_steps
                if cfg.trainer.eval_every_n_steps is not None
                else cfg.checkpointing.every_n_train_steps
            )

            checkpoint_dir = output_dir / "checkpoints"
            eval_ckpts = get_eval_checkpoints(checkpoint_dir, eval_every_n_steps)

            if trainer.global_rank == 0:
                print(
                    f"[PostEval] Found {len(eval_ckpts)} checkpoints "
                    f"for eval_every_n_steps={eval_every_n_steps}"
                )

            # evaluation용 view sampler를 command-line override처럼 강제로 설정
            cfg_dict.mode = "test"
            for dataset_name in cfg_dict.dataset.keys():
                num_context_views = cfg_dict.dataset[dataset_name].view_sampler.get(
                    "num_context_views", 2
                )
                cfg_dict.dataset[dataset_name].view_sampler = {
                    "name": "evaluation",
                    "index_path": str(
                        cfg.trainer.auto_eval_index_path
                        if cfg.trainer.auto_eval_index_path is not None
                        else f"assets/evaluation_index_{dataset_name}.json"
                    ),
                    "num_context_views": num_context_views,
                }

            cfg_dict.test.save_image = cfg.trainer.auto_eval_save_image
            cfg_dict.test.save_video = cfg.trainer.auto_eval_save_video
            cfg_dict.test.save_compare = cfg.trainer.auto_eval_save_compare

            # global config 갱신
            set_cfg(cfg_dict)
            eval_cfg = load_typed_root_config(cfg_dict)

            eval_data_module = DataModule(
                eval_cfg.dataset,
                eval_cfg.data_loader,
                step_tracker=None,
                global_rank=trainer.global_rank,
            )

            for ckpt in eval_ckpts:
                step = get_ckpt_step(ckpt)

                if trainer.global_rank == 0:
                    print(f"[PostEval] Evaluating step {step}: {ckpt}")

                # model_wrapper는 같은 객체를 재사용하되, checkpoint weight로 다시 load됨
                model_wrapper._auto_eval_step = step
                model_wrapper._auto_eval_log_prefix = "auto_eval"

                trainer.test(
                    model_wrapper,
                    datamodule=eval_data_module,
                    ckpt_path=str(ckpt),
                )

            if hasattr(model_wrapper, "_auto_eval_step"):
                delattr(model_wrapper, "_auto_eval_step")
            if hasattr(model_wrapper, "_auto_eval_log_prefix"):
                delattr(model_wrapper, "_auto_eval_log_prefix")

    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()