# Final++rel

This experiment keeps Final++'s weighted appearance std and SH MLP
`[mean256, std256, context64] -> 256 -> 256 -> SH75`.
It adds one geometry-conditioned interaction to the shared allocation hidden
feature. The candidate graph, budgets, outgoing softmax, moments, opacity,
appearance selection and 2D reader retain their existing definitions.

For source support j and destination slot i, let
`u_i = W_i f_i`, `v_j = W_j f_j` (64 channels each).
Coordinates and delta use the existing detached per-scene scale.

```text
delta_ji = x_j - x_i
g_ji = [delta_ji, ||delta_ji||^2]                         # 4 channels
rho_j^2 = max(mean_i ||delta_ji||^2, scene_epsilon^2)     # includes self
local_g_ji = [delta_ji/rho_j, ||delta_ji||^2/rho_j^2]

r_ji = LayerNorm(u_i) * LayerNorm(v_j)                   # no affine params
gamma_ji = tanh(Linear_4_to_64(local_g_ji))               # zero weight + bias
h_ji = SiLU(u_i + v_j + W_g g_ji + b + gamma_ji * r_ji)

geometry_logit_ji = allocation_head[2](h_ji)
appearance_score_ji = appearance_head(h_ji)
```

The original geometry term retains its scene-normalized distance information.
Only the new gate uses neighborhood-relative geometry. The radius is
differentiable; clamping its square before rsqrt avoids infinite derivatives
for coincident points. LayerNorm uses epsilon 1e-5.

The gate adds `4*64 + 64 = 320` parameters. Post-DPT decoder counts:

| Model | Parameters |
| --- | ---: |
| NoPoSplat baseline (two view heads) | 1,222,310 |
| Final++ | 336,665 |
| Final++rel | 336,985 |

With identical original weights, zero initialization reproduces Final++'s
outputs and original-parameter gradients. The new gate receives gradients
immediately. Its initialization is isolated from the existing RNG sequence.

Both `re10k_moment.yaml` and `acid_moment.yaml` enable
`model.encoder.moment_decoder.geometry_conditioned_relation: true`.
Training length, losses, precision and batch size are unchanged. The existing
training launcher still works:

```bash
sbatch scripts/train_re10k.sh
# Control experiment: exact Final++ architecture.
sbatch scripts/train_re10k.sh \
  model.encoder.moment_decoder.geometry_conditioned_relation=false \
  wandb.name=final_plus
```

Full Final++ checkpoints have no `geometry_relation.gate.*` keys. Do not use
them as an exact optimizer resume for Final++rel. For evaluating an old
Final++ checkpoint, disable this flag. Ordinary backbone initialization is
unchanged.

W&B metrics use the existing `log_every_n_steps` schedule and
`appearance_capacity/` prefix:

- `relation_gate_rms`: magnitude of the tanh gate.
- `relation_update_rms`: magnitude of the extra pre-SiLU term.
- `relation_relative_update`: update RMS divided by original pre-SiLU RMS
  (denominator squared sum clamped at 1e-12).

Each value is computed over all candidate edges/channels per scene, then
averaged over scenes and ranks, like the other decoder diagnostics. All three
are zero initially. Increasing values show that the branch is being used;
they alone do not establish a quality improvement.

Feature projections and LayerNorm run once per support. The edge geometry,
gate and product use existing allocation chunks/checkpoints; no extra kNN or
full-scene edge-feature tensor is added. Two normalized 64-channel support
arrays are retained; chunk temporaries and backward work still increase.
Actual training speed and peak CUDA memory need measurement on the GPU.

```bash
python -m unittest discover -s tests -p test_geometry_relation.py -v
```

Tests cover zero-start Final++ parity, original/new gradients, nonzero gates
against a dense independent formula, budget conservation, chunk/scene
checkpoint equivalence, inference, coincident points and both composed
experiment configs. The CUDA/Inductor check runs only on a CUDA host.
