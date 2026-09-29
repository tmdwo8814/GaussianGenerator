# Separate appearance weights (experiment 1)

`exp/separate_weight` keeps DPT256, the current k=32 graph, geometry allocation,
support budget, moments, coverage head, opacity mapping, renderer, MSE/LPIPS,
pretrained initialization, and evaluation settings. There is no Sim3, matcher,
teacher, extra rendering pass, or experiment-2 allocation refinement.

## Computation

For support `j`, `neighbors[j,k]=i` is a destination slot. Geometry is unchanged:

```
q[j,k] = budget[j] * softmax(outgoing geometry logits)[j,k]
mass[i] = sum of incoming q
wG[j,k] = (q[j,k] + epsilon * self_edge) / (mass[i] + epsilon)
```

The existing pair MLP hidden activation feeds one additional bias-free linear
head producing an appearance score **per edge**. Its weights start at zero.

```
wA[j,k] = wG[j,k] * exp(score[j,k])
          / sum of (wG * exp(score)) arriving at slot i
```

Normalization is over incoming edges, not the outgoing k candidates. The
implementation uses an incoming log-sum-exp for stability and preserves exactly
zero edges. The existing self epsilon handles empty slots but never adds opacity.
At initialization wA equals wG within floating-point tolerance. A scalar bias
shared by all scores would cancel in normalization, so the new head has no bias.

Centers, covariance, and coverage still use wG. Appearance uses wA. Because the
existing SH head H is affine, `H(sum(wA*f)) = sum(wA*H(f))`; the implementation
pools 75 SH channels (degree 4), avoiding a second 256-channel feature pool.
Both weights retain gradients into geometry allocation, features, and points.
These are separate selection weights, not fully isolated gradient paths.

Read `common/separate_appearance.py` for normalization,
`heads/moment_gaussian_decoder.py` for integration, and
`common/moment_diagnostics.py` for measurements.

## Training and evaluation

Run the existing entry point:

```bash
sbatch scripts/train_re10k.sh
```

Its default experiment is `re10k_moment`. `checkpointing.load` stays null, so a
fresh run uses the same MASt3R pretrained initialization instead of continuing a
Sim3 checkpoint. Do not pass a resume checkpoint for this comparison. E2E,
optimizer, MSE/LPIPS, dataset selection, training length, and automatic evaluation
remain inherited from `re10k`.

The current branch uses k=32; this change preserves it. Compare against a k=32
control. To reproduce the earlier k=16 setting, override `num_neighbors=16` in
both experiments. To use the original shared weights, set
`model.encoder.moment_decoder.separate_appearance=false` in a fresh run.

## W&B diagnostics

`log_every_n_steps: 50` requests detached diagnostics on training steps. The
wrapper rounds this interval up to a multiple of Lightning's logging interval so
the measurements are emitted on flush steps. Set 0 to disable. No diagnostics
are retained/computed during ordinary inference. DDP averages local-rank values.

All statistics describe **incoming** weights. Scalar means equally weight slots
with `mass > mass_epsilon` within each local batch; epsilon-only empty slots are
excluded. Incoming support count is not bounded by k.

| W&B key (`moment/` prefix) | Definition / interpretation |
| --- | --- |
| `weight_tv` | Mean `0.5 * sum_incoming(abs(wG-wA))`; 0 initially, in [0,1]. |
| `geometry_entropy`, `appearance_entropy` | Mean incoming `-sum(w*log(w))`, natural logarithms. Lower is more concentrated. |
| `geometry_concentration`, `appearance_concentration` | Mean incoming `sum(w*w)`. Higher is more concentrated. |
| `geometry_effective_supports`, `appearance_effective_supports` | Mean `1/sum(w*w)`, not renderer contribution or an actual Gaussian count. |
| `covariance_radius_normalized` | Mean `sqrt(trace(final covariance)/3) / scene_scale`. Includes learned coverage and covariance floor. |
| `concentration_covariance_corr` | Pearson correlation between geometry concentration and log(normalized covariance radius), over active local-batch slots. |
| `concentration_covariance_corr_valid` | 1 when both quantities have variance; undefined correlation is reported as 0 with this flag 0. DDP average is a valid-rank fraction. |
| `active_slot_fraction` | Fraction of slots passing the mass threshold above. |

An appearance score can change SH without changing geometry for fixed shared
parameters. During training the shared MLP/backbone still receive gradients from
both paths, so their learned geometry may evolve differently. A change in these
diagnostics alone does not demonstrate better reconstruction; compare NVS scores.

## Checks

```bash
python -m unittest discover -s tests -p 'test_separate_appearance.py' -v
python -m unittest discover -s tests -v
```

Tests cover an independent dense reference, finite-difference gradients,
zero-score equivalence to the original decoder, unchanged geometry at fixed
parameters, SH pooling equivalence, checkpointing, diagnostic definitions, and
the training-to-logger path. CUDA tests run only where CUDA is available.
