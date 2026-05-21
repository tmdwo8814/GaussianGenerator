from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .common.alpha_blending_stats import (
    GaussianRefinementDecoder,
    build_gauss_feat,
    apply_delta,
)
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["noposplat", "noposplat_multi"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_center_head(
            output_mode='pts3d',
            head_type='dpt',
            landscape_only=True,
            depth_mode=('exp', -inf, inf),
            conf_mode=None,
        )
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)

        # Stage 3: Gaussian refinement decoder
        # (AlphaBlendingPredictor removed — stats_pred now comes from DPT head)
        self.gaussian_refinement_decoder = GaussianRefinementDecoder()

    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone,
                                             has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone,
                                             has_conf=bool(conf_mode))
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )
            self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(
                head_type, 'gs_params', self.backbone,
                has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(
                head_type, 'gs_params', self.backbone,
                has_conf=False, out_nchan=self.raw_gs_dim)
        elif head_type == 'dpt_gs':
            # dpt_gs head returns (gs_out, stats_pred) tuple
            self.gaussian_param_head = head_factory(
                head_type, 'gs_params', self.backbone,
                has_conf=False, out_nchan=self.raw_gs_dim)
            self.gaussian_param_head2 = head_factory(
                head_type, 'gs_params', self.backbone,
                has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2 ** x
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def _run_gs_params_head(self, head, dec_tokens, pts3d, img, shape):
        """
        Run the gs_params head.
        dpt_gs head returns (gs_out, stats_pred).
        Other head types return gs_out only (stats_pred = None).
        """
        if self.gs_params_head_type == 'linear':
            gs_out = rearrange_head(
                head(dec_tokens[-1]),
                self.patch_size,
                *shape[0].cpu().tolist(),
            )
            return gs_out, None

        elif self.gs_params_head_type == 'dpt':
            gs_out = head([tok.float() for tok in dec_tokens],
                          shape[0].cpu().tolist())
            gs_out = rearrange(gs_out, "b d h w -> b (h w) d")
            return gs_out, None

        elif self.gs_params_head_type == 'dpt_gs':
            result = head(
                [tok.float() for tok in dec_tokens],
                pts3d.permute(0, 3, 1, 2),
                img[:, :3],
                shape[0].cpu().tolist(),
            )
            # dpt_gs returns (gs_out, stats_pred)
            gs_out, stats_pred = result
            gs_out = rearrange(gs_out, "b d h w -> b (h w) d")
            return gs_out, stats_pred

        else:
            raise NotImplementedError(f"unexpected {self.gs_params_head_type=}")

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # ── Stage 1: 기존 NoPoSplat encoding ─────────────────────────────
        dec1, dec2, shape1, shape2, view1, view2 = self.backbone(
            context, return_views=True)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

            # gs_params head — dpt_gs returns (gs_out, stats_pred) per view
            GS_res1, stats_pred_1 = self._run_gs_params_head(
                self.gaussian_param_head, dec1, res1['pts3d'], view1['img'], shape1)
            GS_res2, stats_pred_2 = self._run_gs_params_head(
                self.gaussian_param_head2, dec2, res2['pts3d'], view2['img'], shape2)

        pts3d1 = rearrange(res1['pts3d'], "b h w d -> b (h w) d")
        pts3d2 = rearrange(res2['pts3d'], "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1).unsqueeze(-2)
        depths  = pts_all[..., -1].unsqueeze(-1)

        gaussians_raw = torch.stack([GS_res1, GS_res2], dim=1)
        gaussians_raw = rearrange(gaussians_raw, "... (srf c) -> ... srf c",
                                  srf=self.cfg.num_surfaces)
        densities = gaussians_raw[..., 0].sigmoid().unsqueeze(-1)

        if self.pose_free:
            gaussians_1st = self.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians_raw[..., 1:], "b v r srf c -> b v r srf () c"),
            )
        else:
            xy_ray, _ = sample_image_grid((h, w), device)
            xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
            xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)
            gaussians_1st = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                self.map_pdf_to_opacity(densities, global_step),
                rearrange(gaussians_raw[..., 1:], "b v r srf c -> b v r srf () c"),
                (h, w),
            )

        # Flatten to (B, V*G, ...) for per-view processing
        means_1st = rearrange(gaussians_1st.means,       "b v r srf spp xyz -> b (v r srf spp) xyz")
        covs_1st  = rearrange(gaussians_1st.covariances, "b v r srf spp i j -> b (v r srf spp) i j")
        harm_1st  = rearrange(gaussians_1st.harmonics,   "b v r srf spp c d  -> b (v r srf spp) c d")
        opa_1st   = rearrange(gaussians_1st.opacities,   "b v r srf spp     -> b (v r srf spp)")

        # ── Stage 2/3: per-view refinement ───────────────────────────────
        G_per_view = h * w
        stats_pred_views = [stats_pred_1, stats_pred_2]   # (B,2,H,W) or None
        rgb_context = context["image"]                     # (B, V, 3, H, W)

        means_list, covs_list, harm_list, opa_list = [], [], [], []
        stats_pred_list = []

        for vi in range(v):
            sl = slice(vi * G_per_view, (vi + 1) * G_per_view)

            m_v   = means_1st[:, sl]
            o_v   = opa_1st[:, sl]
            c_v   = covs_1st[:, sl]
            hr_v  = harm_1st[:, sl]
            rgb_v = rgb_context[:, vi]
            stats_v = stats_pred_views[vi]    # (B, 2, H, W) from DPT aux head

            # Build Gaussian feature map (B, 88, H, W)
            gauss_feat = build_gauss_feat(m_v, o_v, c_v, hr_v, h, w)

            if stats_v is not None:
                # Stage 3: refinement conditioned on stats from DPT head
                delta_v = self.gaussian_refinement_decoder(gauss_feat, stats_v, rgb_v)
                m_v2, o_v2, c_v2, hr_v2 = apply_delta(m_v, o_v, c_v, hr_v, delta_v, h, w)
                stats_pred_list.append(stats_v)
            else:
                # Fallback for non-dpt_gs head types (no refinement)
                m_v2, o_v2, c_v2, hr_v2 = m_v, o_v, c_v, hr_v

            means_list.append(m_v2)
            opa_list.append(o_v2)
            covs_list.append(c_v2)
            harm_list.append(hr_v2)

        means_2nd = torch.cat(means_list, dim=1)
        covs_2nd  = torch.cat(covs_list,  dim=1)
        harm_2nd  = torch.cat(harm_list,  dim=1)
        opa_2nd   = torch.cat(opa_list,   dim=1)

        # stats_pred: (B, V, 2, H, W) — stored for L_stat in training_step
        if stats_pred_list:
            stats_pred = torch.stack(stats_pred_list, dim=1)
        else:
            stats_pred = None

        # ── Visualization dump ───────────────────────────────────────────
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
            visualization_dump["scales"] = rearrange(
                gaussians_1st.scales, "b v r srf spp xyz -> b (v r srf spp) xyz")
            visualization_dump["rotations"] = rearrange(
                gaussians_1st.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw")
            visualization_dump["means"] = rearrange(
                gaussians_1st.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz",
                h=h, w=w)
            visualization_dump["opacities"] = rearrange(
                gaussians_1st.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w)
            visualization_dump["stats_pred"] = stats_pred   # (B, V, 2, H, W)

        gaussians_out = Gaussians(
            means=means_2nd,
            covariances=covs_2nd,
            harmonics=harm_2nd,
            opacities=opa_2nd,
        )
        # Attach stats_pred for LossAlphaStats
        gaussians_out.stats_pred = stats_pred

        return gaussians_out

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )
            return batch
        return data_shim

# from copy import deepcopy
# from dataclasses import dataclass
# from typing import Literal, Optional

# import torch
# import torch.nn.functional as F
# from einops import rearrange
# from jaxtyping import Float
# from torch import Tensor, nn

# from .backbone.croco.misc import transpose_to_landscape
# from .heads import head_factory
# from ...dataset.shims.bounds_shim import apply_bounds_shim
# from ...dataset.shims.normalize_shim import apply_normalize_shim
# from ...dataset.shims.patch_shim import apply_patch_shim
# from ...dataset.types import BatchedExample, DataShim
# from ...geometry.projection import sample_image_grid
# from ..types import Gaussians
# from .backbone import Backbone, BackboneCfg, get_backbone
# from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
# from .common.alpha_blending_stats import (
#     AlphaBlendingPredictor,
#     GaussianRefinementDecoder,
#     build_gauss_feat,
#     apply_delta,
# )
# from .encoder import Encoder
# from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

# inf = float('inf')


# @dataclass
# class OpacityMappingCfg:
#     initial: float
#     final: float
#     warm_up: int


# @dataclass
# class EncoderNoPoSplatCfg:
#     name: Literal["noposplat", "noposplat_multi"]
#     d_feature: int
#     num_monocular_samples: int
#     backbone: BackboneCfg
#     visualizer: EncoderVisualizerEpipolarCfg
#     gaussian_adapter: GaussianAdapterCfg
#     apply_bounds_shim: bool
#     opacity_mapping: OpacityMappingCfg
#     gaussians_per_pixel: int
#     num_surfaces: int
#     gs_params_head_type: str
#     input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
#     input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
#     pretrained_weights: str = ""
#     pose_free: bool = True


# def rearrange_head(feat, patch_size, H, W):
#     B = feat.shape[0]
#     feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
#     feat = F.pixel_shuffle(feat, patch_size)
#     feat = rearrange(feat, "b d h w -> b (h w) d")
#     return feat


# class EncoderNoPoSplat(Encoder[EncoderNoPoSplatCfg]):
#     backbone: nn.Module
#     gaussian_adapter: GaussianAdapter

#     def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
#         super().__init__(cfg)

#         self.backbone = get_backbone(cfg.backbone, 3)

#         self.pose_free = cfg.pose_free
#         if self.pose_free:
#             self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
#         else:
#             self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

#         self.patch_size = self.backbone.patch_embed.patch_size[0]
#         self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

#         self.gs_params_head_type = cfg.gs_params_head_type

#         self.set_center_head(
#             output_mode='pts3d',
#             head_type='dpt',
#             landscape_only=True,
#             depth_mode=('exp', -inf, inf),
#             conf_mode=None,
#         )
#         self.set_gs_params_head(cfg, cfg.gs_params_head_type)

#         # >>> STAGE 2/3 START
#         self.alpha_blending_predictor = AlphaBlendingPredictor()
#         self.gaussian_refinement_decoder = GaussianRefinementDecoder()
#         # <<< STAGE 2/3 END

#     def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
#         self.backbone.depth_mode = depth_mode
#         self.backbone.conf_mode = conf_mode
#         self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
#         self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
#         self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
#         self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

#     def set_gs_params_head(self, cfg, head_type):
#         if head_type == 'linear':
#             self.gaussian_param_head = nn.Sequential(
#                 nn.ReLU(),
#                 nn.Linear(
#                     self.backbone.dec_embed_dim,
#                     cfg.num_surfaces * self.patch_size ** 2 * self.raw_gs_dim,
#                 ),
#             )
#             self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
#         elif head_type == 'dpt':
#             self.gaussian_param_head = head_factory(
#                 head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
#             self.gaussian_param_head2 = head_factory(
#                 head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
#         elif head_type == 'dpt_gs':
#             self.gaussian_param_head = head_factory(
#                 head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
#             self.gaussian_param_head2 = head_factory(
#                 head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
#         else:
#             raise NotImplementedError(f"unexpected {head_type=}")

#     def map_pdf_to_opacity(
#         self,
#         pdf: Float[Tensor, " *batch"],
#         global_step: int,
#     ) -> Float[Tensor, " *batch"]:
#         cfg = self.cfg.opacity_mapping
#         x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
#         exponent = 2**x
#         return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

#     def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
#         head = getattr(self, f'head{head_num}')
#         return head(decout, img_shape, ray_embedding=ray_embedding)

#     def forward(
#         self,
#         context: dict,
#         global_step: int = 0,
#         visualization_dump: Optional[dict] = None,
#     ) -> Gaussians:
#         device = context["image"].device
#         b, v, _, h, w = context["image"].shape

#         # ── Stage 1: 기존 NoPoSplat decoding ──────────────────────────────
#         dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
#         with torch.cuda.amp.autocast(enabled=False):
#             res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
#             res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

#             if self.gs_params_head_type == 'linear':
#                 GS_res1 = rearrange_head(self.gaussian_param_head(dec1[-1]), self.patch_size, h, w)
#                 GS_res2 = rearrange_head(self.gaussian_param_head2(dec2[-1]), self.patch_size, h, w)
#             elif self.gs_params_head_type == 'dpt':
#                 GS_res1 = self.gaussian_param_head([tok.float() for tok in dec1], shape1[0].cpu().tolist())
#                 GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
#                 GS_res2 = self.gaussian_param_head2([tok.float() for tok in dec2], shape2[0].cpu().tolist())
#                 GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
#             elif self.gs_params_head_type == 'dpt_gs':
#                 GS_res1 = self.gaussian_param_head(
#                     [tok.float() for tok in dec1],
#                     res1['pts3d'].permute(0, 3, 1, 2),
#                     view1['img'][:, :3],
#                     shape1[0].cpu().tolist(),
#                 )
#                 GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
#                 GS_res2 = self.gaussian_param_head2(
#                     [tok.float() for tok in dec2],
#                     res2['pts3d'].permute(0, 3, 1, 2),
#                     view2['img'][:, :3],
#                     shape2[0].cpu().tolist(),
#                 )
#                 GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")

#         pts3d1 = rearrange(res1['pts3d'], "b h w d -> b (h w) d")
#         pts3d2 = rearrange(res2['pts3d'], "b h w d -> b (h w) d")
#         pts_all = torch.stack((pts3d1, pts3d2), dim=1).unsqueeze(-2)

#         depths = pts_all[..., -1].unsqueeze(-1)

#         gaussians_raw = torch.stack([GS_res1, GS_res2], dim=1)
#         gaussians_raw = rearrange(gaussians_raw, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
#         densities = gaussians_raw[..., 0].sigmoid().unsqueeze(-1)

#         if self.pose_free:
#             gaussians_1st = self.gaussian_adapter.forward(
#                 pts_all.unsqueeze(-2),
#                 depths,
#                 self.map_pdf_to_opacity(densities, global_step),
#                 rearrange(gaussians_raw[..., 1:], "b v r srf c -> b v r srf () c"),
#             )
#         else:
#             xy_ray, _ = sample_image_grid((h, w), device)
#             xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
#             xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)
#             gaussians_1st = self.gaussian_adapter.forward(
#                 rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
#                 rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
#                 rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
#                 depths,
#                 self.map_pdf_to_opacity(densities, global_step),
#                 rearrange(gaussians_raw[..., 1:], "b v r srf c -> b v r srf () c"),
#                 (h, w),
#             )

#         # Flatten to (B, V*H*W, ...) for standard Gaussians type
#         means_1st = rearrange(gaussians_1st.means,       "b v r srf spp xyz -> b (v r srf spp) xyz")
#         covs_1st  = rearrange(gaussians_1st.covariances, "b v r srf spp i j -> b (v r srf spp) i j")
#         harm_1st  = rearrange(gaussians_1st.harmonics,   "b v r srf spp c d  -> b (v r srf spp) c d")
#         opa_1st   = rearrange(gaussians_1st.opacities,   "b v r srf spp     -> b (v r srf spp)")

#         # ── Stage 2 & 3: per-view alpha blending prediction + refinement ──
#         # G_per_view = H * W (num_surfaces=1, gaussians_per_pixel=1)
#         G_per_view = h * w

#         stats_pred_list = []   # will hold (B, 2, H, W) per view

#         # Accumulate refined attributes per view then re-concatenate
#         means_list, covs_list, harm_list, opa_list = [], [], [], []

#         # RGB context images: (B, V, 3, H, W)
#         rgb_context = context["image"]   # normalised, shape (B, V, 3, H, W)

#         for vi in range(v):
#             # Slice per-view Gaussians: (B, G_per_view, ...)
#             sl = slice(vi * G_per_view, (vi + 1) * G_per_view)

#             m_v   = means_1st[:, sl]   # (B, G, 3)
#             o_v   = opa_1st[:, sl]     # (B, G)
#             c_v   = covs_1st[:, sl]    # (B, G, 3, 3)
#             hr_v  = harm_1st[:, sl]    # (B, G, 3, d_sh)
#             rgb_v = rgb_context[:, vi] # (B, 3, H, W)

#             # Build pixel-aligned feature map (B, 88, H, W)
#             gauss_feat = build_gauss_feat(m_v, o_v, c_v, hr_v, h, w)

#             # Stage 2: predict alpha blending statistics
#             stats_v = self.alpha_blending_predictor(gauss_feat)   # (B, 2, H, W)
#             stats_pred_list.append(stats_v)

#             # Stage 3: predict residual Gaussian corrections
#             delta_v = self.gaussian_refinement_decoder(gauss_feat, stats_v, rgb_v)  # (B, 88, H, W)

#             # Apply delta with constraints
#             m_v2, o_v2, c_v2, hr_v2 = apply_delta(m_v, o_v, c_v, hr_v, delta_v, h, w)

#             means_list.append(m_v2)
#             opa_list.append(o_v2)
#             covs_list.append(c_v2)
#             harm_list.append(hr_v2)

#         # Concatenate views back
#         means_2nd = torch.cat(means_list, dim=1)   # (B, 2*G, 3)
#         covs_2nd  = torch.cat(covs_list,  dim=1)   # (B, 2*G, 3, 3)
#         harm_2nd  = torch.cat(harm_list,  dim=1)   # (B, 2*G, 3, d_sh)
#         opa_2nd   = torch.cat(opa_list,   dim=1)   # (B, 2*G)

#         # stats_pred: (B, V, 2, H, W)
#         stats_pred = torch.stack(stats_pred_list, dim=1)

#         # ── Visualization dump ───────────────────────────────────────────
#         if visualization_dump is not None:
#             visualization_dump["depth"] = rearrange(
#                 depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
#             )
#             visualization_dump["scales"] = rearrange(
#                 gaussians_1st.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
#             )
#             visualization_dump["rotations"] = rearrange(
#                 gaussians_1st.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
#             )
#             visualization_dump["means"] = rearrange(
#                 gaussians_1st.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
#             )
#             visualization_dump["opacities"] = rearrange(
#                 gaussians_1st.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
#             )
#             # Store stats_pred and world-space means_2nd for training_step
#             visualization_dump["stats_pred"] = stats_pred        # (B, V, 2, H, W)
#             visualization_dump["means_2nd"]  = means_2nd         # (B, 2G, 3)

#         gaussians_out = Gaussians(
#             means=means_2nd,
#             covariances=covs_2nd,
#             harmonics=harm_2nd,
#             opacities=opa_2nd,
#         )

#         # Attach stats_pred for loss computation (accessed in LossAlphaStats)
#         gaussians_out.stats_pred = stats_pred   # (B, V, 2, H, W)

#         return gaussians_out

#     def get_data_shim(self) -> DataShim:
#         def data_shim(batch: BatchedExample) -> BatchedExample:
#             batch = apply_normalize_shim(
#                 batch,
#                 self.cfg.input_mean,
#                 self.cfg.input_std,
#             )
#             return batch
#         return data_shim

