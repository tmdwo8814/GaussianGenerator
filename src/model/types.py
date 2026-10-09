from dataclasses import dataclass

from jaxtyping import Bool, Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    # Optional mask of output slots, independent of opacity or target visibility.
    # Dense tensors retain source order; compact_scene selects actual primitives.
    active_mask: Bool[Tensor, "batch gaussian"] | None = None

    def compact_scene(self, index: int) -> "Gaussians":
        selection = slice(None) if self.active_mask is None else self.active_mask[index]
        return Gaussians(**{
            name: getattr(self, name)[index, selection].unsqueeze(0)
            for name in ('means', 'covariances', 'harmonics', 'opacities')
        })
