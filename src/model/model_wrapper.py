from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any
import json
import time
import zlib

import moviepy.editor as mpy
import torch
import torch.distributed as dist
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self._test_start_time = None
        self._test_total_samples = None
        self._test_local_total_samples = None
        self._test_local_seen = 0

        self._test_metric_names = ("psnr_ours", "ssim_ours", "lpips_ours")
        self._test_overlap_tags = ("small", "medium", "large")


    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat(
                                    [batch_combined[k][kk], batch_per_dl[k][kk]], dim=0
                                )
                        else:
                            raise NotImplementedError
            batch = batch_combined
    
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
    
        # >>> NEW 1: visualization_dump을 항상 생성 (stats_pred 수집용)
        visualization_dump = {}
        if self.distiller is not None:
            visualization_dump["collect_distiller"] = True
    
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
    
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]
    
        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())
    
        # >>> NEW 2: stats_GT 계산 (T_GT + sigma_D2_GT)
        stats_gt = self._compute_stats_gt(
            gaussians=gaussians,
            extrinsics=batch["target"]["extrinsics"],
            intrinsics=batch["target"]["intrinsics"],
            near=batch["target"]["near"],
            far=batch["target"]["far"],
            image_shape=(h, w),
            output=output,
        )
        # Attach to gaussians so LossAlphaStats can access it
        gaussians.stats_gt = stats_gt   # (B, V_target, 2, H, W) — detached
    
        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians, self._eval_step())
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
    
        # distillation (unchanged)
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            distillation_loss = self.distiller_loss(
                pseudo_gt1["pts3d"], pseudo_gt2["pts3d"],
                visualization_dump["means"][:, 0].squeeze(-2),
                visualization_dump["means"][:, 1].squeeze(-2),
                pseudo_gt1["conf"], pseudo_gt2["conf"],
                disable_view1=False,
            ) * 0.1
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss
    
        self.log("loss/total", total_loss)
    
        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)
    
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
    
        return total_loss
    # def training_step(self, batch, batch_idx):
    #     # combine batch from different dataloaders
    #     if isinstance(batch, list):
    #         batch_combined = None
    #         for batch_per_dl in batch:
    #             if batch_combined is None:
    #                 batch_combined = batch_per_dl
    #             else:
    #                 for k in batch_combined.keys():
    #                     if isinstance(batch_combined[k], list):
    #                         batch_combined[k] += batch_per_dl[k]
    #                     elif isinstance(batch_combined[k], dict):
    #                         for kk in batch_combined[k].keys():
    #                             batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
    #                     else:
    #                         raise NotImplementedError
    #         batch = batch_combined
    #     batch: BatchedExample = self.data_shim(batch)
    #     _, _, _, h, w = batch["target"]["image"].shape

    #     # Run the model.
    #     visualization_dump = None
    #     if self.distiller is not None:
    #         visualization_dump = {}
    #     gaussians = self.encoder(batch["context"], self.global_step, visualization_dump=visualization_dump)
   
    #     output = self.decoder.forward(
    #         gaussians,
    #         batch["target"]["extrinsics"],
    #         batch["target"]["intrinsics"],
    #         batch["target"]["near"],
    #         batch["target"]["far"],
    #         (h, w),
    #         depth_mode=self.train_cfg.depth_mode,
    #     )
    #     target_gt = batch["target"]["image"]

    #     # Compute metrics.
    #     psnr_probabilistic = compute_psnr(
    #         rearrange(target_gt, "b v c h w -> (b v) c h w"),
    #         rearrange(output.color, "b v c h w -> (b v) c h w"),
    #     )
    #     self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

    #     # Compute and log loss.
    #     total_loss = 0
    #     for loss_fn in self.losses:
    #         loss = loss_fn.forward(output, batch, gaussians, self._eval_step())
    #         self.log(f"loss/{loss_fn.name}", loss)
    #         total_loss = total_loss + loss

    #     # distillation
    #     if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
    #         with torch.no_grad():
    #             pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
    #         distillation_loss = self.distiller_loss(pseudo_gt1['pts3d'], pseudo_gt2['pts3d'],
    #                                                 visualization_dump['means'][:, 0].squeeze(-2),
    #                                                 visualization_dump['means'][:, 1].squeeze(-2),
    #                                                 pseudo_gt1['conf'], pseudo_gt2['conf'], disable_view1=False) * 0.1
    #         self.log("loss/distillation_loss", distillation_loss)
    #         total_loss = total_loss + distillation_loss

    #     self.log("loss/total", total_loss)

    #     if (
    #         self.global_rank == 0
    #         and self.global_step % self.train_cfg.print_log_every_n_steps == 0
    #     ):
    #         print(
    #             f"train step {self.global_step}; "
    #             f"scene = {[x[:20] for x in batch['scene']]}; "
    #             f"context = {batch['context']['index'].tolist()}; "
    #             f"loss = {total_loss:.6f}"
    #         )
    #     self.log("info/global_step", self.global_step)  # hack for ckpt monitor

    #     # Tell the data loader processes about the current step.
    #     if self.step_tracker is not None:
    #         self.step_tracker.set_step(self.global_step)

    #     return total_loss

    @torch.no_grad()
    def _compute_stats_gt(
        self,
        gaussians,
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        output,
    ):
        """
        Compute GT alpha-blending statistics from Gaussian_2nd.
    
        Returns:
            stats_gt: (B, V, 2, H, W) — detached, on same device
                ch0: T_GT        = 1 - accumulated_alpha
                ch1: sigma_D2_GT = depth variance (normalised to [0,1])
        """
        from math import isqrt
        from einops import rearrange, repeat
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        from ..geometry.projection import get_fov
        from .decoder.cuda_splatting import get_projection_matrix
    
        device = gaussians.means.device
        b, v_tgt, _, _ = extrinsics.shape
        h, w = image_shape

        # Flatten B, V → B*V
        extr_flat    = rearrange(extrinsics, "b v i j -> (b v) i j")   # (B*V, 4, 4)
        intr_flat    = rearrange(intrinsics, "b v i j -> (b v) i j")   # (B*V, 3, 3)
        near_flat    = rearrange(near,       "b v -> (b v)")            # (B*V,)
        far_flat     = rearrange(far,        "b v -> (b v)")            # (B*V,)

        scale_flat   = 1.0 / near_flat                                  # (B*V,)

        # scale_invariant: means, covariances를 per-(b,v) scale로 scaling
        # gaussians.means: (B, G, 3) → repeat for each target view
        means_rep = repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v_tgt)
        covs_rep  = repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v_tgt)
        opa_rep   = repeat(gaussians.opacities, "b g -> (b v) g", v=v_tgt)

        means_scaled = means_rep * scale_flat[:, None, None]
        covs_scaled  = covs_rep  * (scale_flat[:, None, None, None] ** 2)
        near_s       = near_flat * scale_flat
        far_s        = far_flat  * scale_flat

        fov_x, fov_y  = get_fov(intr_flat).unbind(dim=-1)              # (B*V,)
        tan_fov_x     = (0.5 * fov_x).tan()
        tan_fov_y     = (0.5 * fov_y).tan()

        proj_mats = get_projection_matrix(near_s, far_s, fov_x, fov_y) # (B*V, 4, 4)
        proj_mats = rearrange(proj_mats, "bv i j -> bv j i")
        view_mats = rearrange(torch.inverse(extr_flat), "bv i j -> bv j i")
        full_proj  = view_mats @ proj_mats

        row, col = torch.triu_indices(3, 3)
        G = gaussians.means.shape[1]
    
        T_gt_list      = []
        sigma_D2_list  = []
    
        for bv_idx in range(b * v_tgt):
            means_bv = means_scaled[bv_idx]      # (G, 3)
            covs_bv  = covs_scaled[bv_idx]       # (G, 3, 3)
            opa_bv   = opa_rep[bv_idx]           # (G,)
            cov_pack = covs_bv[:, row, col]      # (G, 6)

            mean2D = torch.zeros_like(means_bv, requires_grad=False)

            settings = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=tan_fov_x[bv_idx].item(),   # flat index
                tanfovy=tan_fov_y[bv_idx].item(),
                bg=torch.zeros(3, device=device),
                scale_modifier=1.0,
                viewmatrix=view_mats[bv_idx],
                projmatrix=full_proj[bv_idx],
                projmatrix_raw=proj_mats[bv_idx],
                sh_degree=0,
                campos=extr_flat[bv_idx, :3, 3],
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(settings)

            w2c       = torch.inverse(extr_flat[bv_idx])
            R_w2c     = w2c[:3, :3]
            t_w2c     = w2c[:3, 3]
            z_cam     = (means_bv @ R_w2c.T + t_w2c)[:, 2] * scale_flat[bv_idx]
    
            # Pass 0: get T_GT via accumulated alpha output
            ones_color = torch.ones(G, 3, device=device)
            _, _, _, alpha_pix, _ = rasterizer(
                means3D=means_bv,
                means2D=mean2D,
                shs=None,
                colors_precomp=ones_color,
                opacities=opa_bv.unsqueeze(-1),
                cov3D_precomp=cov_pack,
            )
            T_gt = 1.0 - alpha_pix.squeeze(0)                        # (H, W)
    
            # Pass 1: E[z]
            z_color = z_cam.unsqueeze(-1).expand(-1, 3)              # (G, 3)
            img_z, _, _, _, _ = rasterizer(
                means3D=means_bv,
                means2D=mean2D,
                shs=None,
                colors_precomp=z_color,
                opacities=opa_bv.unsqueeze(-1),
                cov3D_precomp=cov_pack,
            )
            Ez = img_z.mean(dim=0)                                   # (H, W)
    
            # Pass 2: E[z²]
            z2_color = (z_cam ** 2).unsqueeze(-1).expand(-1, 3)
            img_z2, _, _, _, _ = rasterizer(
                means3D=means_bv,
                means2D=mean2D,
                shs=None,
                colors_precomp=z2_color,
                opacities=opa_bv.unsqueeze(-1),
                cov3D_precomp=cov_pack,
            )
            Ez2 = img_z2.mean(dim=0)                                 # (H, W)
    
            sigma_D2 = (Ez2 - Ez ** 2).clamp(min=0.0)               # (H, W)
    
            T_gt_list.append(T_gt)
            sigma_D2_list.append(sigma_D2)
    
        # Stack: (B*V, H, W) → (B, V, H, W)
        T_gt_all      = torch.stack(T_gt_list,     dim=0).reshape(b, v_tgt, h, w)
        sigma_D2_all  = torch.stack(sigma_D2_list, dim=0).reshape(b, v_tgt, h, w)
    
        # Normalise sigma_D2 to [0, 1] per sample
        sigma_max = sigma_D2_all.flatten(2).max(dim=-1).values       # (B, V)
        sigma_D2_norm = sigma_D2_all / (sigma_max.unsqueeze(-1).unsqueeze(-1) + 1e-6)
    
        # Concatenate channels: (B, V, 2, H, W)
        stats_gt = torch.stack([T_gt_all, sigma_D2_norm], dim=2).detach()
    
        # stats_pred from encoder is (B, V_context=2, 2, H, W)
        # stats_gt is computed for target views (V_target).
        # For L_stat we need them aligned:
        # Use only the first target view to match context view count,
        # OR average over target views.
        # Simple approach: use first target view only → (B, 1, 2, H, W)
        # then expand to match stats_pred shape (B, 2, 2, H, W) via repeat.
        # Better: compute mean over target views → (B, 1, 2, H, W) broadcast.
        # For now: return (B, V_target, 2, H, W) and let loss handle alignment.
        return stats_gt   # (B, V_target, 2, H, W), detached

    def _format_seconds(self, seconds: float) -> str:
        seconds = int(max(seconds, 0))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


    def _find_eval_index_path(self) -> Optional[Path]:
        cfg = get_cfg()
        if "dataset" not in cfg:
            return None

        for dataset_name in cfg["dataset"].keys():
            ds_cfg = cfg["dataset"][dataset_name]
            if "view_sampler" in ds_cfg and "index_path" in ds_cfg["view_sampler"]:
                return Path(ds_cfg["view_sampler"]["index_path"])
        return None


    def _scene_owner_rank(self, scene: str, world_size: int) -> int:
        return zlib.crc32(scene.encode("utf-8")) % max(world_size, 1)


    def _infer_test_sample_counts(self) -> tuple[Optional[int], Optional[int]]:
        index_path = self._find_eval_index_path()
        if index_path is None or not index_path.exists():
            return None, None

        with index_path.open("r") as f:
            index = json.load(f)

        scenes = [scene for scene, entry in index.items() if entry is not None]
        total = len(scenes)

        world_size = max(int(getattr(self.trainer, "world_size", 1)), 1)
        if world_size == 1:
            return total, total

        local_total = sum(
            self._scene_owner_rank(scene, world_size) == self.global_rank
            for scene in scenes
        )
        return total, local_total


    def _init_test_metric_buffers(self) -> None:
        self._test_metric_sums = {
            k: torch.zeros((), device=self.device) for k in self._test_metric_names
        }
        self._test_metric_count = torch.zeros((), device=self.device)

        self._test_overlap_sums = {
            tag: {k: torch.zeros((), device=self.device) for k in self._test_metric_names}
            for tag in self._test_overlap_tags
        }
        self._test_overlap_counts = {
            tag: torch.zeros((), device=self.device) for tag in self._test_overlap_tags
        }


    def _update_test_metric_buffers(
        self,
        all_metrics: Optional[dict[str, Tensor]],
        overlap_tag: Optional[str],
    ) -> None:
        self._test_local_seen += 1

        if all_metrics is None:
            return

        self._test_metric_count += 1
        for k, v in all_metrics.items():
            self._test_metric_sums[k] += v.detach()

        if overlap_tag in self._test_overlap_sums:
            self._test_overlap_counts[overlap_tag] += 1
            for k, v in all_metrics.items():
                self._test_overlap_sums[overlap_tag][k] += v.detach()


    def _maybe_print_test_progress(self) -> None:
        # rank 0 shard 기준 ETA / preview metric
        if self._rank() != 0:
            return

        seen = self._test_local_seen
        if seen == 0:
            return

        should_print = (seen % 50 == 0)
        if self._test_local_total_samples is not None and seen == self._test_local_total_samples:
            should_print = True
        if not should_print:
            return

        elapsed = time.time() - self._test_start_time
        msg = f"[Eval][rank0 shard] {seen}"

        if self._test_local_total_samples is not None:
            msg += f"/{self._test_local_total_samples}"
            eta = (elapsed / seen) * max(self._test_local_total_samples - seen, 0)
            msg += f" | eta={self._format_seconds(eta)}"

        msg += f" | elapsed={self._format_seconds(elapsed)}"

        if self.test_cfg.compute_scores and self._test_metric_count.item() > 0:
            preview = {
                k: (self._test_metric_sums[k] / self._test_metric_count).item()
                for k in self._test_metric_names
            }
            msg += (
                f" | PSNR={preview['psnr_ours']:.3f}"
                f" | SSIM={preview['ssim_ours']:.3f}"
                f" | LPIPS={preview['lpips_ours']:.3f}"
            )

        print(msg)


    def _reduce_test_metric_buffers(self) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return

        dist.all_reduce(self._test_metric_count, op=dist.ReduceOp.SUM)
        for k in self._test_metric_names:
            dist.all_reduce(self._test_metric_sums[k], op=dist.ReduceOp.SUM)

        for tag in self._test_overlap_tags:
            dist.all_reduce(self._test_overlap_counts[tag], op=dist.ReduceOp.SUM)
            for k in self._test_metric_names:
                dist.all_reduce(self._test_overlap_sums[tag][k], op=dist.ReduceOp.SUM)


    def _print_final_test_metrics(
        self,
        log_prefix: str = "test",
        log_step: Optional[int] = None,
    ) -> None:
        if self._rank() != 0 or not self.test_cfg.compute_scores:
            return

        if self._test_metric_count.item() == 0:
            print(f"[{log_prefix}] No test metrics were accumulated.")
            return

        overall = {
            k: (self._test_metric_sums[k] / self._test_metric_count).item()
            for k in self._test_metric_names
        }

        print(f"\n[{log_prefix} | Final Evaluation | global | all ranks]")
        print(
            tabulate(
                [
                    [
                        "ours",
                        f"{overall['psnr_ours']:.3f}",
                        f"{overall['ssim_ours']:.3f}",
                        f"{overall['lpips_ours']:.3f}",
                        int(self._test_metric_count.item()),
                    ]
                ],
                headers=["Method", "PSNR", "SSIM", "LPIPS", "Num Samples"],
            )
        )

        overlap_rows = []
        for tag in self._test_overlap_tags:
            count = int(self._test_overlap_counts[tag].item())
            if count == 0:
                continue

            overlap_psnr = (
                self._test_overlap_sums[tag]["psnr_ours"]
                / self._test_overlap_counts[tag]
            ).item()
            overlap_ssim = (
                self._test_overlap_sums[tag]["ssim_ours"]
                / self._test_overlap_counts[tag]
            ).item()
            overlap_lpips = (
                self._test_overlap_sums[tag]["lpips_ours"]
                / self._test_overlap_counts[tag]
            ).item()

            overlap_rows.append(
                [
                    tag,
                    f"{overlap_psnr:.3f}",
                    f"{overlap_ssim:.3f}",
                    f"{overlap_lpips:.3f}",
                    count,
                ]
            )

        if overlap_rows:
            print(f"\n[{log_prefix} | Final Evaluation by overlap | global | all ranks]")
            print(
                tabulate(
                    overlap_rows,
                    headers=["Overlap", "PSNR", "SSIM", "LPIPS", "Num Samples"],
                )
            )

        # wandb time-series logging + summary logging
        log_payload = {
            f"{log_prefix}/overall/psnr": overall["psnr_ours"],
            f"{log_prefix}/overall/ssim": overall["ssim_ours"],
            f"{log_prefix}/overall/lpips": overall["lpips_ours"],
            f"{log_prefix}/overall/num_samples": int(self._test_metric_count.item()),
        }

        for tag in self._test_overlap_tags:
            count = int(self._test_overlap_counts[tag].item())
            if count == 0:
                continue

            log_payload[f"{log_prefix}/{tag}/psnr"] = (
                self._test_overlap_sums[tag]["psnr_ours"]
                / self._test_overlap_counts[tag]
            ).item()
            log_payload[f"{log_prefix}/{tag}/ssim"] = (
                self._test_overlap_sums[tag]["ssim_ours"]
                / self._test_overlap_counts[tag]
            ).item()
            log_payload[f"{log_prefix}/{tag}/lpips"] = (
                self._test_overlap_sums[tag]["lpips_ours"]
                / self._test_overlap_counts[tag]
            ).item()
            log_payload[f"{log_prefix}/{tag}/num_samples"] = count

        if wandb.run is not None:
            ckpt_step = self.global_step if log_step is None else log_step
            log_payload[f"{log_prefix}/ckpt_step"] = ckpt_step

            wandb.log(log_payload)

            for k, v in log_payload.items():
                wandb.run.summary[k] = v

    def on_test_start(self) -> None:
        self._test_start_time = time.time()
        self._test_local_seen = 0
        self._init_test_metric_buffers()
        self._test_total_samples, self._test_local_total_samples = self._infer_test_sample_counts()

        if self.global_rank == 0:
            msg = (
                f"[Eval] world_size={self.trainer.world_size}"
                f" | align_pose={self.test_cfg.align_pose}"
            )
            if self._test_total_samples is not None:
                msg += f" | total_samples={self._test_total_samples}"
            if self._test_local_total_samples is not None:
                msg += f" | rank0_local_samples={self._test_local_total_samples}"
            print(msg)

    def _detach_tree(self, x):
        if torch.is_tensor(x):
            return x.detach()

        if isinstance(x, dict):
            return {k: self._detach_tree(v) for k, v in x.items()}

        if isinstance(x, list):
            return [self._detach_tree(v) for v in x]

        if isinstance(x, tuple) and hasattr(x, "_fields"):  # namedtuple
            return type(x)(*(self._detach_tree(v) for v in x))

        if isinstance(x, tuple):
            return tuple(self._detach_tree(v) for v in x)

        if hasattr(x, "__dataclass_fields__"):
            from dataclasses import replace
            return replace(
                x,
                **{
                    field: self._detach_tree(getattr(x, field))
                    for field in x.__dataclass_fields__
                },
            )

        return x

    def _rank(self) -> int:
        if hasattr(self, "_auto_eval_rank"):
            return self._auto_eval_rank
        return self.global_rank


    def _eval_step(self) -> int:
        if hasattr(self, "_auto_eval_step"):
            return self._auto_eval_step
        return self.global_step

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        eval_step = self._eval_step()

        # Render Gaussians.
        # Evaluation에서는 Gaussian을 고정해야 하므로 encoder graph를 만들지 않음.
        with self.benchmarker.time("encoder"):
            with torch.no_grad():
                gaussians = self.encoder(
                    batch["context"],
                    eval_step,
                )

        gaussians = self._detach_tree(gaussians)

        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )

        all_metrics = None
        overlap_tag = None

        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                "psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
                "ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                "lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
            }

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            context_img = inverse_normalize(batch["context"]["image"][0])
            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

        self._update_test_metric_buffers(all_metrics, overlap_tag)
        self._maybe_print_test_progress()

    def begin_blocking_eval(
        self,
        total_samples: Optional[int] = None,
        local_total: Optional[int] = None,
    ) -> None:
        """
        Evaluation 시작 전에 필요한 초기화
        - 평가 샘플 수, local 샘플 수 초기화
        - benchmark 파일 준비
        """
        self._test_start_time = time.time()  # 평가 시작 시간
        self._test_local_seen = 0  # 로컬 샘플 수 초기화
        self._test_total_samples = total_samples  # 총 샘플 수
        self._test_local_total_samples = local_total  # 로컬 샘플 수
        self._init_test_metric_buffers()  # 성능 지표 초기화

        # auto-eval마다 benchmark를 fresh하게 시작
        self.benchmarker = Benchmarker()  # benchmark 초기화


    def end_blocking_eval(
        self,
        log_prefix: str = "test",
        log_step: Optional[int] = None,
        dump_benchmarks: bool = False,
    ) -> None:
        """
        Evaluation이 끝난 후 결과 집계 및 로그 기록
        - 최종 성능 메트릭 계산 및 로그 기록
        - benchmark 파일 저장
        """
        self._reduce_test_metric_buffers()  # 모든 rank의 metric을 합산
        self._print_final_test_metrics(log_prefix=log_prefix, log_step=log_step)  # 최종 성능 출력

        # benchmark 파일 저장
        if dump_benchmarks:
            name = get_cfg()["wandb"]["name"]
            out_dir = self.test_cfg.output_path / name
            out_dir.mkdir(parents=True, exist_ok=True)

            if getattr(self.trainer, "world_size", 1) > 1:
                self.benchmarker.dump(out_dir / f"benchmark_rank{self.global_rank}.json")
                self.benchmarker.dump_memory(out_dir / f"peak_memory_rank{self.global_rank}.json")
            else:
                self.benchmarker.dump(out_dir / "benchmark.json")
                self.benchmarker.dump_memory(out_dir / "peak_memory.json")
                self.benchmarker.summarize()

    def test_step_align(self, batch, gaussians):
        # self.encoder.eval()
        # # freeze all parameters
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        out_dir.mkdir(parents=True, exist_ok=True)

        self._reduce_test_metric_buffers()

        log_prefix = getattr(self, "_auto_eval_log_prefix", "test")
        log_step = getattr(self, "_auto_eval_step", None)

        self._print_final_test_metrics(
            log_prefix=log_prefix,
            log_step=log_step,
        )

        if getattr(self.trainer, "world_size", 1) > 1:
            self.benchmarker.dump(out_dir / f"benchmark_rank{self.global_rank}.json")
            self.benchmarker.dump_memory(out_dir / f"peak_memory_rank{self.global_rank}.json")

            if self.global_rank == 0:
                print(f"[Eval] per-rank benchmark files were saved to {out_dir}")
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )

        if self.distiller is not None:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            depth1, depth2 = pseudo_gt1['pts3d'][..., -1], pseudo_gt2['pts3d'][..., -1]
            conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            depth_dust = torch.cat([depth1, depth2], dim=0)
            depth_dust = vis_depth_map(depth_dust)
            conf_dust = torch.cat([conf1, conf2], dim=0)
            conf_dust = confidence_map(conf_dust)
            dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            comparison = hcat(add_label(vcat(*dust_vis), "Context"), comparison)

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.
        projections = hcat(
                *render_projections(
                    gaussians,
                    256,
                    extra_label="",
                )[0]
            )
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=20)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )


    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }