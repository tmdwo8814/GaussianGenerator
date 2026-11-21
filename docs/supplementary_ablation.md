# Supplementary B.1/B.2: decoder-only ablations

This mode is separate from the main-paper frozen-BEAM readout experiment.
The common scene encoder (backbone, point heads, DPT/RGB feature extractors)
is frozen and stays in eval mode. BEAM, budget, appearance selection/readers,
SH prediction and any required regression outputs train jointly.

| Variant | Assignment | Center | Covariance | Opacity |
| --- | --- | --- | --- | --- |
| M1 | uniform | R | R | R |
| M2 | uniform | F | F | F |
| M3 | learned | R | R | R |
| M4 | learned | F | F | F |
| M5 | learned | R | F | F |
| M6 | learned | F | R | F |
| M7 | learned | F | F | R |

M3/M4 results are reused in B.2. Existing main-paper metrics are not reference
rows for this newly trained decoder-only protocol.

## Decoder options

- `frozen_encoder_decoder_only: true` selects this training protocol.
- `geometry_readout_only: false` keeps BEAM and appearance trainable.
- `geometry_mode: egf` is the common decoder path; selection is controlled by
  `attribute_modes: RRR/FFF/RFF/FRF/FFR` in center/covariance/opacity order.
- `assignment_mode: learned/uniform` changes only outgoing proportions.
  Uniform uses the actual candidate count, preserves learned support budgets,
  and preserves the shared assignment-MLP hidden representation for appearance.
- `covariance_coverage: false` gives the paper's weighted covariance with the
  existing numeric floor/roundoff guard; `true` retains the legacy learned
  coverage factor on fitted covariances. The experiment configs select false
  consistently for the paper's fitting formula.

Regression reuses the existing single affine pooled-feature readout. Only
required outputs are allocated: 10 for RRR, 3 for RFF, 6 for FRF and 1 for FFR.
At feature width 256 these have 2570, 771, 1542 and 257 parameters, respectively.
Selected rows use the same canonical random initialization across variants.

Fitted covariance always uses the original weighted mean, including in M5.
The appearance reader also uses that fitted mean in all variants; the selected
rendered center never changes the definition of fitted covariance or the
appearance query's construction.

## Checkpoint initialization

`initialize_supplementary_ablation` loads and validates every frozen frontend
weight from a common Lightning checkpoint. Original NoPoSplat attribute-output
convolutions under the two DPT output heads are intentionally discarded.
No pretrained BEAM or appearance/Gaussian-decoder weight is loaded.
Missing or mismatched frozen weights fail explicitly rather than leaving a
randomly initialized feature extractor frozen. The common checkpoint is
/workspace/research/GaussianGenerator/pretrained_weights/re10k.ckpt.

## SSH batch command

Activate the server's training environment, then run in tmux:

```bash
cd /workspace/research/GaussianGenerator
wandb login
export RUN_ROOT="$PWD/outputs/supplementary/$(date +%Y-%m-%d_%H-%M-%S)"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_re10k_supplementary.sh
```

Each model uses ONE GPU and batch size 16 (global batch 16 per model).
Initially GPU 0 runs M1, GPU 1 runs M2, GPU 2 runs M3 and GPU 3 runs M4.
Each model trains for 10,000 optimizer steps, saves checkpoints at 5,000/10,000,
and then evaluates BOTH checkpoints. As soon as that process finishes, its
GPU takes the next waiting model (M5, then M6, then M7). There is no barrier
between groups of four and no DDP between models. Shared frozen feature weights are loaded from
the common checkpoint; the Gaussian decoder is freshly initialized each time.
Full checkpoints include optimizer state. Each model processes 160,000 training
examples over 10k updates (including repeated scenes). AdamW LR is 1e-4,
LR warm-up is 500 steps, K=16, with 256x256 inputs and the
existing RE10K train/test splits and evaluation index. Color is trainable in
this supplementary protocol. Main-paper readout experiments are unchanged.

The default checkpoint path can be changed with ENCODER_CHECKPOINT. To run
only selected variants, set MODELS; each batch uses a fresh RUN_ROOT:

```bash
ENCODER_CHECKPOINT=/different/path/re10k.ckpt MODELS="M5 M6 M7" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_re10k_supplementary.sh
```

Additional arguments are passed to src.main for every model. W&B uses the
existing login/environment credentials, project noposplat, with one run per
variant (supp_m1_uniform_rrr, etc.). Metrics are logged under
auto_eval/overall/{psnr,ssim,lpips} at checkpoint steps 5000 and 10000.
The launcher pins trainer.devices=1 and data_loader.train.batch_size=16.
Each child sees only its assigned physical GPU through CUDA_VISIBLE_DEVICES;
inside that process it is named cuda:0, even when the assigned GPU is 1/2/3.
A failed run is marked FAILED; other models keep running and the queue continues.
The launcher returns a nonzero exit code if any model fails. Ctrl+C or SIGTERM
stops the active child process groups and leaves queued models unstarted.

An already-running DDP job does not change when these files are uploaded.
Stop the old launcher and its four training processes before starting this
new queue on the same GPUs. Start in a fresh RUN_ROOT: the previous global-batch
64 checkpoint is not resumed into these global-batch 16 experiments.

## Progress and saved results

In another SSH terminal, set RUN_ROOT to the printed batch directory:

```bash
python scripts/supplementary_status.py "$RUN_ROOT"
watch -n 10 "python scripts/supplementary_status.py '$RUN_ROOT'"
tail -f "$RUN_ROOT/scheduler.log"
tail -f "$RUN_ROOT/M1/console.log"
```

The status table includes the physical GPU assigned to each model. Pending
models show GPU '-'. Status is inferred from run markers, recent train-step logs, checkpoint names
and measured score files. States are pending, training, evaluating, complete
and failed. Use nvidia-smi to check actual GPU activity.

```text
$RUN_ROOT/
  variants.txt
  encoder_checkpoint.txt
  gpus.txt                            # GPU pool in initial assignment order
  scheduler.log                       # model starts, GPU assignments and completions
  results.csv                         # one row per variant/checkpoint
  M1/                                 # also M2 ... M7
    .hydra/config.yaml                # exact experiment configuration
    console.log                       # training + evaluation output
    gpu.txt                           # assigned physical GPU ID
    launch.json                       # PID, exact command and start time
    RUNNING / COMPLETED / FAILED      # run markers
    checkpoints/*.ckpt                # 5k and 10k full checkpoints
    evaluation/
      step_00005000/metrics.json       # globally reduced metrics
      step_00010000/metrics.json
    test/                             # benchmark artifacts
```

The scheduler refreshes results.csv after each model. Refresh it manually while
evaluation is running with:

```bash
python scripts/supplementary_status.py "$RUN_ROOT" --write-csv
```

The completed suite has 14 CSV rows: 7 models x 2 checkpoints. Use the step=10000
rows for the final B.1/B.2 tables and reuse M3/M4 instead of training them twice.
Completion requires both valid metric files, rather than only a clean process
exit or the existence of checkpoints. Existing metrics such as 25.840 dB must
not be substituted for the newly trained M4 result.

Scene-evidence caching is not enabled because the current sampler and image
augmentation can change the inputs. Geometry aggregation remains in FP32.

## CPU verification

```bash
python -m unittest discover -s tests -p test_supplementary_ablation.py -v
python -m unittest discover -s tests -p test_supplementary_scheduler.py -v
```

These checks cover all seven combinations, learned budget and color gradients,
frozen encoder state, uniform normalization, independent attribute selection,
minimal partial heads and strict frontend-only checkpoint loading.
Scheduler checks use short CPU subprocesses to verify initial GPU assignment,
dynamic reuse after evaluation, GPU isolation, 14 result rows, failure handling
and interruption cleanup. No CUDA training is performed by those tests.
