# Experiment 2: Moment Shape v1

Branch: `exp/moment-shape`. Experiment: `re10k_moment`. W&B: `moment-shape-v1`.

This version keeps the original DPT256, 3D kNN16, support budgets, allocation,
Gaussian centers and opacity mapping. It combines separate appearance weights
with the original single-reader 2DKNN, then adds learned covariance shaping.
It does not include experiment 1's multi-head reader or nonlinear SH head.

## Forward path

1. Geometry allocation produces incoming weights `wG`, mass `m`, mean `mu`,
   central covariance `C`, and geometry-pooled feature `fG`, as before.
2. A zero-initialized `Linear(64,1,bias=False)` reads the existing allocation
   hidden state to score appearance edges. At each receiving slot,
   `wA = normalize_incoming(wG * exp(score))`. Zero geometry weights stay zero.
3. Base SH uses `wA`; the original 32-channel 2DKNN reader adds context SH.
   Pooling the affine SH outputs instead of DPT256 is mathematically equivalent
   and avoids an extra full 256-channel pooled feature tensor.
4. Existing scalar coverage `s` gives `S = s^2 C`. A `262 -> 128 -> 6` SiLU MLP
   reads `fG` and six unique entries of `S / max(trace(S), covariance_floor^2)`.
5. Its outputs form a lower-triangular `L`. Diagonals use
   `exp(log(4) * tanh(raw))`; off-diagonals use `0.5 * tanh(raw)`.
   The final MLP layer starts at zero, so `L = I` initially.
6. Final covariance is `(L S L^T + epsilon I) * scene_scale^2`.
   The existing fixed floor plus detached FP32 roundoff guard is applied
   **after** the transform. Means, mass-to-opacity and SH do not use `L`.

The shape transform cannot restore rank to a zero covariance moment; such slots
retain the original isotropic floor. It offers bounded shape freedom without
replacing moment-based geometry. Lower-triangular factors are expressed in the
existing common coordinate frame; this is not a rotation-equivariance claim.

## Cost and training

Decoder parameters (excluding backbone, DPT and point heads): **115,284**.
The shape MLP adds **34,438** parameters; separate selection adds **64**.
No extra kNN, matching, eigendecomposition or matrix inverse is used. Shape MLP
activations use the existing chunk size and activation checkpointing.

MSE, LPIPS, pretrained initialization, e2e optimization and evaluation settings
are unchanged. No SSIM loss is added. Use a fresh run for this experiment:

```bash
sbatch scripts/train_re10k.sh
```

The script defaults to `re10k_moment` unless `EXPERIMENT` is set. A strict resume
of an old architecture checkpoint is not supported: the new heads have missing
parameters and the optimizer state differs. Baseline pretrained initialization
through the existing initialization path is retained.

## W&B checks

Detached diagnostics are logged every 50 steps (rounded to the trainer flush
interval), under `moment_shape/`:

- `shape/identity_distance`, `shape/diagonal_mean`, `shape/shear_abs_mean`:
  whether the transform learns; initial values are 0, 1 and 0.
- `shape/radius_before`, `shape/radius_after`, `shape/radius_ratio`:
  3D RMS radius in scene-normalized coordinates, before the isotropic floor.
  These are **not projected pixel footprints**. Zero-moment slots report ratio
  1; `shape/nonzero_moment_fraction` makes their prevalence visible.
- `shape/transform_volume_ratio`: mean determinant of `L` (initially 1).
- `weight_tv`, geometry/appearance entropy, concentration and effective support
  counts: whether appearance uses different incoming weights.
- Original 2DKNN selection, context SH, edge MSE and smooth-region MSE metrics.

Set `moment_shape=false` to disable covariance shaping; additionally set
`separate_appearance=false` to recover the original 2DKNN architecture. Shape
and separate flags default to false outside this experiment config.

CPU checks: `python -m unittest discover -s tests -v`. Tests cover initialization
and original gradients, shape bounds, positive definite covariance including
degenerate moments, checkpoint parity, scene-scale equivariance, one kNN per
scene, optimizer inclusion, and logging without adding a loss. CUDA parity is
checked only when a CUDA device is available.
