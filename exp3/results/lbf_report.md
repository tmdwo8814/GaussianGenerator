# LBF measurements

Scenes: 500; seed: 0; bootstrap: 10000.

Main cutoff: baseline full-population P90, opacity < 0.25, active opacity >= 1/255.
Frozen size cutoffs: projected 3.97829 px; world 0.0854643.
Projected reference: origin context view and dataset camera poses; no pose optimization.

| Space / metric | Original | Ours | Ours - original (95% CI) | Paired Wilcoxon p |
|---|---:|---:|---:|---:|
| projected / lbf_count | 7.388% | 2.195% | -5.193 pp [-5.688, -4.708] | 6.041e-79 |
| projected / faint_given_large | 73.876% | 39.818% | -34.058 pp [-35.926, -32.097] | 2.209e-81 |
| projected / lbf_area | 79.910% | 5.379% | -74.531 pp [-96.283, 82.982] | 0.36094 |
| world / lbf_count | 5.398% | 1.231% | -4.166 pp [-4.758, -3.611] | 2.4077e-73 |
| world / faint_given_large | 53.977% | 20.655% | -33.322 pp [-36.509, -30.015] | 2.1937e-70 |

Projected grid cells/metrics with non-decreasing point estimates: 0.
All grid rows and exclusions are retained in per_scene.csv and summary.csv.
Wilcoxon p-values are unadjusted; the P90 / alpha=0.25 comparison is the main setting.
CIs condition on the frozen baseline size thresholds.
Area is potential unfiltered 1-sigma ellipse area, without occlusion/overlap/screen-boundary correction.
Transitions match source-pixel slots, not identical physical Gaussians.

| Decoder | Total slots | Active | Nonpositive origin depth (active) | Behind all contexts | Origin only behind |
|---|---:|---:|---:|---:|---:|
| original | 65536000 | 65176850 | 53305 (51736) | 18477 | 34828 |
| ours | 65536000 | 64966686 | 356296 (307129) | 77284 | 279012 |
