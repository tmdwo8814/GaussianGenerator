# Experiment 3: local CNN appearance before allocation

Implemented on the current `exp/local-cnn` branch. Experiment name:
`moment-local-cnn-v1`. Base: separate_weight, DPT256, 3D kNN16.

For support j in the ORIGINAL view raster:

```
local_j = Conv1x1(ReLU(Conv3x3(F_view)))_j
h_j = Linear_SH(f_j) + local_j
SH_i = sum_j wA[j,i] * h_j
```

The CNN is shared across views and uses **256 -> 256 -> 75** channels at SH
degree 4. The 3x3 convolution has padding 1, stride 1 and no bias, following
the baseline head's first convolution. The 1x1 convolution has a bias; its
weight AND bias start at zero. The original SH degree mask is applied once,
after incoming pooling. No dropout, BatchNorm, extra loss or pretrained CNN
is added. This is the requested compact CNN path, not an exact copy of every
layer in the original head.

The module receives DPT features, which already include RGB information. It
does not receive raw RGB, target images or target cameras. `image_shape=(V,H,W)`
only restores the original view-major raster layout. Each view is convolved
independently; Gaussian centers, including their movement, never define the
CNN's spatial arrangement. The code does not convolve pooled slot features.

Geometry allocation, support budgets, moment means/covariances, scalar coverage
and opacity mapping remain unchanged. Existing appearance weights select the
combined SH on the original kNN graph. There is no 2DKNN reader or new neighbor
search. With fixed input features and original parameters, enabling the CNN
changes only SH. End-to-end training can of course also change geometry via
the shared backbone and photometric gradients.

## Files

- `src/model/encoder/common/local_cnn_appearance.py`: convolution and reshape.
- `src/model/encoder/heads/moment_gaussian_decoder.py`:
  `predict_support_harmonics`, then original incoming appearance pooling.
- `src/model/encoder/encoder_noposplat.py`: passes source `(V,H,W)` metadata.
- `config/experiment/re10k_moment.yaml`: enables `local_cnn`, width 256.

## Initialization and resource use

New CNN: **609,099** parameters. Decoder total: **662,105**, excluding the
backbone, DPT and point heads. Both existing head parameters and subsequent
initialization RNG state are preserved. The zero final Conv initially
preserves separate_weight outputs and its original gradients. The final Conv
learns on the first step; the first Conv receives nonzero task gradients once
the final Conv weights become nonzero.

The CNN runs once per complete view inside the existing scene loop. With
`checkpoint_chunks: true`, its dense hidden activation is recomputed during
backward. This uses **no spatial tiling**, and the full 3x3 neighborhood remains
intact. Only the 75-channel combined source SH is pooled; no additional pooled
DPT256 map is needed. CNN compute and CUDA convolution workspace still add cost;
this is not a guarantee against OOM. Measure step time and VRAM on the host.

## Training and diagnostics

Use a fresh run from the existing pretrained initialization:

```bash
sbatch scripts/train_re10k.sh
```

The script defaults to `re10k_moment` unless `EXPERIMENT` is set. MSE/LPIPS,
e2e optimization, data and evaluation settings are inherited unchanged.
Existing trained separate_weight checkpoints cannot be strictly resumed with
the new CNN/optimizer state. No new dependencies or downloads are needed.
Set `model.encoder.moment_decoder.local_cnn=false` for the original control.

At the existing logging interval, W&B receives the original `moment/*` metrics
plus `moment/local_cnn/source_base_sh_rms`, `source_extra_sh_rms`, and
`source_extra_base_ratio`. RMS uses the SH degree mask and is measured on source
supports **before pooling/rendering**. It indicates whether the new path is
used, not its causal contribution to rendered image quality. Extra RMS starts
at zero. Disabled diagnostics do not run these reductions.

Validation: `python -m unittest discover -s tests -v`. Tests cover parameter
counts, initialization/RNG and original gradients, independent dense SH pooling,
directional 2D neighborhoods without view/border leakage, geometry invariance,
checkpoint parity, original DPT map forwarding, optimizer inclusion and tiny
views. CUDA parity runs only when a CUDA device is available. The pre-existing
configuration test's stale kNN expectation (32) is corrected to the actual 16.
