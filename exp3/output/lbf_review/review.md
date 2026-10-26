# LBF result review

Source: existing full-population CSVs; no inference or thresholds changed.

| Space | Original LBF | Ours LBF | Relative reduction | Improved scenes | Delta 95% CI (pp) |
|---|---:|---:|---:|---:|---:|
| Projected | 7.388% | 2.195% | 70.29% | 466/500 | [-5.688, -4.708] |
| World | 5.398% | 1.231% | 77.19% | 451/500 | [-4.758, -3.611] |

Within-scene rank correlations reverse sign in the mean:

- Projected: -0.411 -> 0.327; positive rho scenes 20 -> 455; rho increases in 499 scenes.
- World: -0.357 -> 0.210; positive rho scenes 27 -> 410; rho increases in 493 scenes.

The original-LBF transition plot does not include incidence at other source slots:

- 707,813 source slots are LBF in both models.
- 711,412 additional source slots are LBF only in ours.
- 3,873,810 original LBF slots cross the size/opacity boundary with valid active outputs.
- Source-slot correspondence is not tracking an identical physical Gaussian.

Extreme-tail limitation:

- All 18 projected count grid point estimates decrease.
- 15 pooled paired-bootstrap difference intervals are below zero.
- All three 16 px difference intervals include zero; do not claim extreme LBF elimination.

Area limitation:

- The pooled area fraction has an enormous CI, and the scene-mean area-fraction difference includes zero.
- Absolute summed potential LBF area: 1.7886e+16 -> 1.79995e+16.
- Total potential area: 2.23828e+16 -> 3.34645e+17.
- The apparent pooled-fraction gain largely reflects denominator growth, not a decrease in the numerator.
- Per-scene area fraction improves in 230/500 scenes.
- A few scenes dominate each sum. Near-zero positive camera depth is a plausible cause because J contains 1/z and x/z^2; scalar CSVs cannot confirm individual-Gaussian causes.
- Use the full cache to inspect depth, anisotropy and off-screen support, then use renderer-consistent culling/projection and screen clipping for a visible-footprint analysis.
- No outliers were excluded in this review.

Supported wording: substantial suppression of LBF prevalence and reversal of within-scene scale-opacity association.
Unsupported wording: complete removal, resolved extreme tails, or established reduction of visible LBF coverage/causal compensation.

The other compensation CSVs in the input folder reference a separate 900-scene, step-80000 baseline experiment; they are not paired evidence for these 500-scene checkpoints.
