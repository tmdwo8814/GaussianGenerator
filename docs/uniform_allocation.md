# Uniform allocation ablation

The uniform ablation changes only the outgoing allocation rule of Final++:

```text
learned: q[j,k] = b[j] * softmax(logits[j,:])[k]
uniform: q[j,k] = b[j] / K_actual
b[j] = budget_max * sigmoid(budget_head(feature[j]))
```

`K_actual` is the candidate count returned by the existing 3D kNN, including
self. Each support still sends total mass `b[j]`; incoming slot masses and
normalized moment weights are not uniform. Budget is trainable, not detached.

The shared relation MLP still produces the separate appearance scores. The 2D
appearance reader, capacity heads, feature moments, Gaussian construction and
losses stay intact. Only the unused final scalar allocation scorer is frozen;
its checkpoint keys, tensors and random initialization are retained. The shared
relation layer remains trainable when separate appearance is enabled.

The current branch defaults to [scene-shared budgeting](scene_shared_budget.md).
For Uniform, set `allocation_mode: uniform` and `budget_mode: support`. Existing
budget caps are preserved: RE10K uses 1.0, ACID uses 2.0, both initialized to
`log(2)`. Match the cap and training schedule to the Ours run used in the table.
No changes are made to checkpoint saving or automatic evaluation schedules.

```bash
sbatch scripts/train_re10k.sh model.encoder.moment_decoder.allocation_mode=uniform model.encoder.moment_decoder.budget_mode=support wandb.name=final_plus_uniform
python -m src.main +experiment=acid_moment wandb.mode=online model.encoder.moment_decoder.allocation_mode=uniform model.encoder.moment_decoder.budget_mode=support wandb.name=final_plus_uniform_acid
```

For the original learned allocation, override
`model.encoder.moment_decoder.allocation_mode=learned` and
`model.encoder.moment_decoder.budget_mode=support` (and give the run a
different W&B name). Start each ablation from the same original pretrained
initialization; do not resume an Ours checkpoint for a full-training comparison.

```bash
python -m unittest discover -s tests -p test_uniform_allocation.py -v
```
