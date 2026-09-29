# Experiment 2: group-aware allocation refinement

The original allocation score describes a support–slot pair before the slot
has received its group. This experiment lets the same pair take the gathered
group into account, then redistributes each support's existing budget once.
It tests whether this extra context improves Gaussian reconstruction; reduced
appearance disagreement alone is not evidence of better rendering.

## Flow and equations

1. Keep the original trainable backbone, point maps, DPT 256-dimensional
   features, and one 3D kNN graph (currently k=16, self first).
2. Compute original pair logits and support budgets:
   `q0[j,i] = b[j] * softmax_i(l0[j,i])`.
3. Gather the **incoming** group of each slot i using q0. Its incoming degree
   can exceed k, since k limits the candidates of each source support j.
4. Predict a new score from the gathered group's statistics and redistribute
   **once on the same graph**:
   `l1[j,i] = l0[j,i] + MLP(group_inputs[j,i])`,
   `q1[j,i] = b[j] * softmax_i(l1[j,i])`.
5. Use q1 for the original moment decoder's geometry, coverage scale,
   appearance and opacity. Render once and train with the original MSE + LPIPS.

The statistics in step 3 are:

```text
a[j]       = masked_SH_head(feature[j])
m0[i]      = sum_j q0[j,i]
w0[j,i]    = (q0[j,i] + epsilon * [j=i]) / (m0[i] + epsilon)
mu0[i]     = sum_j w0[j,i] * point[j]
abar0[i]   = sum_j w0[j,i] * a[j]
D0[i]      = sum_j w0[j,i] * ||a[j] - abar0[i]||^2

group_inputs[j,i] = [l0[j,i], point[j] - mu0[i],
                    ||a[j] - abar0[i]||^2, D0[i], log(1 + m0[i])]
```

Positions use the existing detached scene-scale normalization. D0 is
implemented as `max(E[||a||^2] - ||E[a]||^2, 0)` to pool all statistics in one
pass. The existing SH mask is applied before comparing coefficients, so high
bands already suppressed by the renderer's input parameterization do not
dominate this descriptor. This is a view-independent appearance proxy;
it does not evaluate target-view color or visibility.

The MLP has 7 inputs, hidden size 32, SiLU, and a scalar output: **288 new
parameters**. Its last weight is zero-initialized, so q1=q0 initially. Existing
head initialization and the subsequent random initialization sequence are
preserved. After training begins, gradients pass through the group statistics
into the original allocation, SH head, features and point map; no detach is
inserted in this path.

Final attributes follow the original formulas, replacing q0 with q1:

```text
m1[i]      = sum_j q1[j,i]
w1[j,i]    = (q1[j,i] + epsilon * [j=i]) / (m1[i] + epsilon)
mu1[i]     = sum_j w1[j,i] * point[j]
C1[i]      = sum_j w1[j,i] * (point[j] - mu1[i]) (point[j] - mu1[i])^T
f1[i]      = sum_j w1[j,i] * feature[j]
Sigma[i]   = coverage_head(f1[i])^2 * C1[i] + existing covariance floor
SH[i]      = masked_SH_head(f1[i])
opacity[i] = 1 - exp(-m1[i])
```

Centers/covariances are rescaled to scene coordinates as before. The epsilon
self contribution stabilizes moments only; it adds no opacity. Each support's
total outgoing mass remains b[j]. There is one shared final allocation, not
the separate geometry/appearance weighting of experiment 1. The graph and
Gaussian count remain fixed, and there is no second rendering pass, alignment
module, teacher, or auxiliary loss.

## Files and training

- `common/group_aware_refinement.py`: group pooling and one score refinement.
- `heads/moment_gaussian_decoder.py`: invokes it before final moment pooling.
- `common/refinement_diagnostics.py`: detached logging summaries.
- `encoder_noposplat.py` / `model_wrapper.py`: pass and log optional diagnostics.
- `config/experiment/re10k_moment.yaml`: enables this experiment.

The existing `scripts/train_re10k.sh` entry point still selects re10k_moment.
Start from the same pretrained initialization as the previous experiments;
do not resume an already trained separate-weight/Sim3 checkpoint for this
comparison. Losses, optimizer, train/evaluation settings and script are unchanged.

```yaml
group_aware_refine: true   # false restores the original moment decoder path
refine_hidden_dim: 32
log_every_n_steps: 50     # 0 disables diagnostics
```

The current branch keeps k=16, matching the current `exp/separate_weight`
configuration. Keep k identical across the runs used for comparison.

## W&B diagnostics

The `refine/` metrics are detached and collected only at the configured
interval, rounded up to a multiple of Lightning's log flush interval.

| Metric suffix | Meaning |
| --- | --- |
| `allocation_tv` | Mean outgoing total variation between softmax(l0) and softmax(l1); 0 initially. |
| `logit_update_centered_rms` | RMS score change after subtracting each source's candidate mean (softmax-invariant offsets removed). |
| `concentration_before/after` | Incoming sum of squared weights; larger values mean more concentrated support selection. |
| `entropy_before/after` | Incoming allocation entropy. |
| `appearance_variance_before/after` | SH disagreement using the same current SH head, before/after redistribution. |
| `covariance_radius_normalized` | Final sqrt(trace(Sigma)/3), divided by scene scale. |
| `concentration_covariance_corr` | Pearson correlation of final concentration with log(normalized radius). |
| `concentration_covariance_corr_valid` | 0 when that correlation is undefined; the reported correlation is then 0. |
| `active_slot_fraction` | Fraction with final mass above epsilon. This is diagnostic, not pruning. |

Incoming means use final active slots in the local batch. Outgoing metrics use
all supports. DDP averages rank summaries. SH disagreement is a learned proxy,
so judge the experiment using PSNR/SSIM/LPIPS together with these diagnostics.

## Resource cost and verification

One extra sparse pooling pass over XYZ, masked SH and squared SH norm, a small
edge MLP, and one extra per-support SH projection are added. No second kNN
search or full intermediate covariance is computed. The existing custom sparse
pool backward and chunk checkpointing bound edge activation memory. Logging
steps additionally measure the final group's statistics under no_grad.

Run `python -m unittest discover -s tests -v`. Tests cover dense-reference values
and gradients, group influence from another incoming support, zero-initialized
equivalence, budget conservation, shared final weights, checkpointing, one kNN
per scene, encoder/optimizer integration, and logging. CUDA tests skip when no
GPU is available; GPU throughput and training quality require the training host.
