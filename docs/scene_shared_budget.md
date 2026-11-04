# Scene-shared budget ablation

Keep the Final++ learned allocation and appearance paths. Replace support-wise
budgets with a single differentiable mean per scene:

```text
b[j] = budget_max * sigmoid(budget_head(feature[j]))
b_mean = mean over ALL supports j of the current scene
a[j,k] = softmax(allocation_logits[j,:])[k]
q[j,k] = b_mean * a[j,k]
```

The mean is taken after sigmoid, over all context views together. It is computed
before chunking and separately for each batch item, with no cross-scene or DDP
averaging. Gradients pass through the mean into the budget head and features.
Each support sends `b_mean`; the total mass is `M * b_mean = sum_j b[j]`.
Incoming slot masses and opacities can still differ because allocation is learned.

There are no new parameters. Both the allocation scorer and budget head are
trainable; the original checkpoint keys, initialization and RNG sequence are
retained. The candidate graph, moment construction, separate appearance weights,
2D reader, SH capacity, losses and train/eval schedule are unchanged.

Both experiment configs use:

```yaml
allocation_mode: learned
budget_mode: scene_shared
```

Existing budget caps are preserved (RE10K: 1.0, ACID: 2.0), with `budget_init`
equal to `log(2)`. Match these settings to the Ours run used for comparison.
Start from the original pretrained initialization, not a trained Ours checkpoint.

```bash
sbatch scripts/train_re10k.sh
python -m src.main +experiment=acid_moment wandb.mode=online
```

W&B names are `final_plus_scene-shared` and `final_plus_scene-shared_acid`.
To restore Ours, override `model.encoder.moment_decoder.budget_mode=support`
and use a different run name. Uniform also requires `allocation_mode=uniform`;
see [Uniform allocation](uniform_allocation.md).

```bash
python -m unittest discover -s tests -p test_scene_shared_budget.py -v
```
