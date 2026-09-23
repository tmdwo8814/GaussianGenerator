"""NoPoSplat's existing DPT + RGB features, projected to a common dimension."""

from .dpt_gs_head import PixelwiseTaskWithDPT


def create_dpt_feature_head(backbone, feature_dim: int = 64):
    """Create one independent feature head per input-view branch.

    Reuses the existing DPT refinement and RGB merger. Only its final output
    head changes: a 1x1 projection replaces direct Gaussian-attribute regression.
    """
    depth = backbone.dec_depth
    if depth <= 9:
        raise ValueError("The existing DPT feature extractor requires dec_depth > 9")
    return PixelwiseTaskWithDPT(
        num_channels=feature_dim,
        feature_dim=256,
        hooks_idx=[0, depth * 2 // 4, depth * 3 // 4, depth],
        dim_tokens=[backbone.enc_embed_dim] + [backbone.dec_embed_dim] * 3,
        postprocess=None,
        head_type="features",
    )
