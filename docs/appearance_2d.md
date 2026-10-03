# SH decoding with additional image neighbors

On `exp/appearance-capacity`, the default experiment is the expanded version
documented in [appearance_capacity.md](appearance_capacity.md). This document
describes the original single-reader control (`appearance_capacity=false`,
`separate_appearance=false`, `appearance_2d=true`).

This experiment keeps the original moment decoder's geometry and adds an
appearance-only neighborhood. A support j still distributes its geometry
budget to its 16 nearest 3D slots. For SH, its candidates additionally include
the eight image neighbors at radius 1 and eight at radius 4 in its own source
view. This tests whether source-image context helps recover appearance when
the 3D neighborhood omits useful texture/edge observations.

## Candidate graph

Inputs keep the original view-major raster order:
`index = view * H * W + y * W + x`.

```text
geometry_neighbors[j] = exact_3d_knn(points[j], k=16), self first
image_neighbors[j]    = same-view pixels at the 8 offsets for radius 1 and 4
appearance_neighbors = geometry_neighbors UNION image_neighbors
```

There are at most 32 distinct outgoing candidates per support. Incoming degree
per slot can be higher, since multiple source supports can send to one slot.
Image borders are masked, never wrapped or replicated. Image candidates that
already occur in the same row's 3D candidates are masked to avoid double
counting. Original geometry neighbor indices and order are unchanged.

Image candidates are constructed once for the whole local batch. The existing
3D kNN is still called once per scene. No matching model, camera, reprojection,
bilinear interpolation or extra CNN is needed. Radius values are pixel units
at the current input resolution; the experiment uses 256x256 images.

## Appearance readout

1. Run the original geometry allocation and moment aggregation, yielding mass,
   center mu[i], covariance and pooled DPT256 feature fG[i].
2. Project each source's `[DPT feature, context RGB]` into compact key/value
   vectors, and fG[i] into a slot query (width 32).
3. Score every valid appearance edge j -> i, normalize **over all incoming
   supports for slot i**, and pool position-aware appearance values:

```text
xi[j,i] = [x[j] - mu[i], ||x[j] - mu[i]||^2,
           same_view * (uv[j] - uv[i]), same_view, new_image_edge]
key[j], value[j] = source_projection([feature[j], RGB[j]])
query[i]        = query_projection(fG[i])
e[j,i]          = score(SiLU(key[j] + query[i] + score_position(xi[j,i])))
beta[j,i]       = exp(e[j,i]) / sum_{t: t -> i} exp(e[t,i])
c[i]            = sum_{j: j -> i} beta[j,i] * SiLU(value[j] + value_position(xi[j,i]))
SH[i]           = mask * (old_SH_head(fG[i]) + extra_SH_head(c[i]))
```

XYZ uses the existing detached scene-scale normalization. UV is normalized by
image width/height, and refers to each slot's original anchor pixel, not the
projection of its moved Gaussian center. Cross-view UV deltas are zero; their
same_view flag is also zero. The new_image_edge flag is 1 only for the appended
image candidate columns. Duplicate/invalid columns have exactly zero beta.

The SH expression is equivalent to a linear head on `[fG, c]`. The new columns
(`extra_SH_head`) are initialized to zero; all existing head parameters and RNG
initialization remain identical for a fixed seed. Initial outputs and gradients
of original parameters match the original decoder. The extra SH head learns
first; gradients into its reader become nonzero once those columns change.

The image edges carry feature information, not geometry mass. The appearance
softmax does not multiply by geometry allocation (which is zero on new edges).
Geometry budgets, covariance/coverage formulas, opacity and Gaussian count are
unchanged. At fixed inputs/base parameters only SH changes; during end-to-end
training the shared features, points and allocation can still receive gradients
through the appearance pathway.

The encoder restores source RGB to [0,1] using its configured input mean/std.
It passes only context RGB and image_shape=(V,H,W), never target data. The
decoder can process any V given that layout; the existing NoPoSplat front end
still supports two views. Unstructured token/voxel inputs would need a source
image association adapter; they must not be silently treated as a pixel grid.

## Training and controls

`config/experiment/re10k_moment.yaml` enables:

```yaml
feature_dim: 256
num_neighbors: 16
appearance_2d: true
appearance_dim: 32
appearance_2d_radii: [1, 4]
log_every_n_steps: 50
```

Use the existing entry point from the training host:

```bash
sbatch scripts/train_re10k.sh
```

The script defaults to `re10k_moment`. Losses (MSE/LPIPS), pretrained
initialization, training/evaluation schedule, optimizer and dataset settings
are unchanged by this implementation. Start a fresh comparison run from the
same pretrained initialization, not a trained separate-weight/Sim3 checkpoint.

`appearance_2d=false` restores the original moment path, including its parameter
keys. `appearance_2d_radii=[]` keeps the new reader with the same 3D candidates
only, for the head-capacity control. This does NOT disable the reader.
The independent appearance-3D32 control is not implemented here: changing
num_neighbors to 32 also changes geometry, and is not that controlled ablation.

## W&B diagnostics

All metrics use the `appearance_2d/` prefix. They are detached, collected every
50 steps by default (rounded up to a multiple of Lightning's logging interval)
and averaged across DDP ranks. Set log_every_n_steps=0 to disable their cost.

| Metric | Meaning |
| --- | --- |
| added_candidates | Mean number of valid, new image candidates per source, after deduplication. |
| added_candidate_fraction | Added image edges / all valid appearance edges. |
| new_2d_fraction | Added image edges / valid in-image proposals, before 3D deduplication. |
| incoming_candidates | Mean number of incoming valid candidates per slot. |
| image_weight | Mean fraction of incoming appearance weight assigned to NEW image edges. |
| image_weight_scale_0 / scale_1 | Contributions from radii 1 / 4 with the default radius list. |
| entropy | Mean incoming attention entropy. |
| context_sh_rms | RMS of the masked SH contribution from the new context, initially zero. |
| edge_mse / smooth_mse | Rendered RGB MSE on / outside the image-edge mask described below. |
| edge_fraction | Fraction of target pixels in the edge mask; 0 means edge_mse is undefined and reported as 0. |

The edge mask uses a mean absolute RGB difference >0.05 to a right or bottom
neighbor in the [0,1] target image. It is a cheap texture/edge diagnostic, not
a geometry boundary label, new supervision or additional loss. These target
measurements are made only AFTER rendering and never affect the decoder input.
For comparing candidate strategies, use the same reader and loss settings;
interpret edge_mse alongside edge_fraction and final PSNR/SSIM/LPIPS.

Nonzero image_weight shows that new candidates participate in the readout,
not that they improve rendering. At initialization context_sh_rms is zero even
though image_weight is positive. Measure both plus held-out reconstruction.

## Cost and verification

Candidate scoring and positional messages run at width 32, not DPT width 256.
The original DPT256 path remains intact. Chunk checkpointing avoids retaining
full edge-value activations across all scenes; incoming softmax has a direct
sparse backward. Image candidates use integer indexing, and no second kNN,
renderer pass, matching network or pretrained dependency is introduced.
The appearance reader still adds compute and memory; GPU step time needs to be
measured on the training machine.

Run `python -m unittest discover -s tests -v`. Tests compare sparse attention and
context against independent dense references (values and gradients), check
candidate boundaries/deduplication, zero-initialized equivalence, unchanged
geometry with an active SH head, influence from new image edges, checkpointing,
RGB normalization/order, configuration/optimizer integration, and W&B logging.
CUDA tests skip on hosts without CUDA.
