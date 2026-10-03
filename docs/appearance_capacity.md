# Appearance capacity, experiment 1

Branch: `exp/appearance-capacity`. Version: `moment-appearance-capacity-v1`.
The user's W&B name in `re10k_moment.yaml` is preserved.

## Flow

1. Keep the DPT256 features, exact 3D kNN (K=16, including self), support
   budgets, geometry allocation, moment centers/covariances and opacity mapping.
2. Reuse the allocation MLP's hidden features to predict one additional score
   `a[j,i]` on each existing geometry edge. The new output weight starts at zero.
3. Compute `wA[j,i] = incoming_softmax(log(wG[j,i]) + a[j,i])`, and
   `fA[i] = sum_j wA[j,i] * f[j]`. Zero geometry weights stay zero. `wG`
   includes the original tiny self prior; it still governs every geometry
   attribute, including the pooled feature used by the coverage scale head.
4. Build the same deduplicated union of 3D edges and radius-1/radius-4 image
   edges once per scene evaluation. Two independent width-32 readers use this graph.
   Each has its own source key/value, geometry-pooled query, positional
   projections and incoming softmax. Concatenate their contexts into `c64`.
5. Produce raw SH as `Linear(fA) + MLP(concat(fA, c64))`. The MLP is
   `320 -> 128 -> SiLU -> 75` at SH degree 4 (SiLU follows the first Linear).
   Its final weight and bias start at zero. Apply the original SH mask once.

The extra image edges do NOT carry geometry mass or multiply by `wG`.
They can influence SH even when they are absent from the geometry graph.
Only context RGB and its view-major raster layout enter the decoder; target
images/poses are used by the unchanged renderer and MSE/LPIPS losses.

The current reduced decoder has **154,649 trainable parameters**, excluding DPT
and point heads: moment 52,942 + separate score 64 + two readers 50,880 + MLP 50,763.
Heads are independent but the complete decoder is shared across input views.

## Configuration and launch

`re10k_moment.yaml` enables the experiment by default on this branch:

```yaml
feature_dim: 256
num_neighbors: 16
appearance_2d: true
separate_appearance: true
appearance_capacity: true
appearance_heads: 2
appearance_dim: 32       # Per head, not total context width.
appearance_mlp_dim: 128
checkpoint_appearance: true
appearance_2d_radii: [1, 4]
log_every_n_steps: 50
```

Use the existing training entry point: `sbatch scripts/train_re10k.sh` on the
Slurm host. The script still selects `re10k_moment`. Start the comparison from
the same pretrained initialization as earlier runs. An old trained 2DKNN
decoder checkpoint has a different reader/SH-head architecture; it is not a
strictly compatible resume checkpoint for this experiment.

For the original 2DKNN control, set BOTH `appearance_capacity=false` and
`separate_appearance=false`; keep `appearance_2d=true`. To restore the original
moment decoder, also set `appearance_2d=false`. Setting `appearance_2d_radii=[]`
only removes the extra image candidates and retains the configured readers.

## Logging

Metrics appear under `appearance_capacity/` at the existing DDP-synchronized
logging interval. They are detached and do not add a loss. Set the interval
to zero to disable their cost.

| Metric | Interpretation |
| --- | --- |
| weight_tv | Mean incoming total variation between wG and wA. Initially approximately zero. |
| geometry/appearance_entropy | Incoming entropy of the corresponding original-graph weights. |
| geometry/appearance_concentration | Mean sum of squared incoming weights. |
| geometry/appearance_effective_supports | Mean per-slot inverse concentration; not a candidate count. |
| head_0..1/entropy | Each head's incoming attention entropy on the expanded graph. |
| head_0..1/image_weight | Each head's weight on NEW image edges after deduplication. |
| head_0..1/context_rms | Magnitude of each head's 32-D context. |
| head_attention_tv | Mean incoming TV across head pairs; tests whether selections differ. |
| image_weight, entropy | Mean across the configured readers. |
| image_weight_scale_0 / scale_1 | Mean reader weights on new radius-1 / radius-4 edges. |
| added_candidates, incoming_candidates, added_candidate_fraction, new_2d_fraction | Same candidate definitions as the original 2DKNN experiment. |
| base_sh_rms | RMS of the masked Linear(fA) output. |
| extra_sh_rms | RMS of the masked additional MLP output, initially zero. |
| edge_mse, smooth_mse, edge_fraction | Existing post-render image diagnostics; not an SSIM loss. |

The selection statistics average all slots (including almost-empty ones),
not only visible Gaussians. `extra_sh_rms` describes an MLP of BOTH fA and
context; it does not isolate the context's causal contribution. Nonzero head
TV or image attention alone also does not prove a rendering-quality gain.

## Cost and verification

The optimized forward completes SH and Gaussian attributes **one scene at a
time**, retaining final attributes rather than stacking full-batch DPT256
appearance features, geometry-pooled features, and multi-head contexts. This
removes the list-plus-stack copies from the old capacity path.

`checkpoint_appearance: true` adds a non-reentrant checkpoint around incoming
appearance selection, feature pooling, the reader, and SH generation. Its
large intermediate features, key/value/query projections and weights are
recomputed scene by scene during backward. Inputs shared with the geometry
path remain available; final attributes and backbone/renderer memory are
still required. Inner chunk checkpoints continue to bound edge activations.
`chunk_size`, head count, widths, candidates, precision, batch size and loss
are unchanged by this optimization. Unused Python references to chunk logits
and the last message are also released earlier.

kNN is **outside** the new checkpoint: exactly one search per scene, including
backward. The image-edge merge and appearance computations can be recomputed.
This trades extra computation for lower retention; it is not a promise of
equal step time or of fitting every batch into GPU memory. Inference performs
one pass with no backward recomputation. `checkpoint_appearance: false` skips
the new checkpoint but keeps scene-by-scene output generation for diagnosis.

All parameter names, shapes and optimizer parameter order remain unchanged
from the 2-head/MLP-128 version. That version's weights remain strictly loadable;
this does not make old 4-head or baseline checkpoints shape-compatible. The
optimization preserves the mathematical output/gradients, with possible small
floating-point differences from per-scene rather than batched GEMMs. W&B metric
names and full-batch SH RMS aggregation remain unchanged.

No new package is required. `test_capacity_memory.py` compares the optimized
path to the previous stack-then-SH flow, including nonzero heads, all parameter
and input gradients, detached metrics and strict state loading. It also checks
unique tensor storage saved by autograd at DPT256/2-head/MLP-128 widths. That CPU
retention measurement excludes Python-only copies, temporary workspace,
backbone and renderer; it is **not** a CUDA peak-VRAM benchmark. Measure actual
step time and peak VRAM on the training host.

Run `python -m unittest discover -s tests -v`. New checks cover incoming
normalization and gradients, independent heads sharing one graph, parameter
count, zero-initialized equivalence, nonzero gradients through every active
head, unchanged geometry, image-only candidates, empty/border graphs,
checkpoint parity, optimizer grouping, Hydra configuration and W&B logging.
CUDA parity tests run only on a CUDA-capable host.
