# Frozen-BEAM geometry readout on RE10K

두 실험은 Final++ checkpoint에서 같은 encoder, point/DPT feature heads,
BEAM assignment/budget, appearance score/readers/SH heads를 로드하고 고정한다.
학습 가능한 부분은 다음뿐이다.

| Experiment | Geometry | Trainable modules |
| --- | --- | --- |
| `re10k_regression` | Center + full covariance regression | `regression_head`, `scale_head` |
| `re10k_egf_readout` | Existing weighted mean + central covariance | `scale_head` |

EGF 수식에는 초기화할 파라미터가 없다. 양쪽의 coverage `scale_head`는
같은 seed로 재초기화한다. Regression head도 무작위 초기화하며,
작은 isotropic Cholesky bias를 사용해 초기 covariance를 안정화한다.
Checkpoint의 optimizer, scheduler, global step은 가져오지 않는다.

## Run on the server

SSH로 접속하여 기존 학습 환경을 활성화한 뒤 실행한다.
`CUDA_VISIBLE_DEVICES`에는 실제 사용할 GPU 번호를 지정한다.

```bash
cd /workspace/research/GaussianGenerator
wandb login

# Regression (default)
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_re10k_readout.sh

# Ours: regression 실행이 끝난 뒤 같은 GPU에서 실행
CUDA_VISIBLE_DEVICES=0,1,2 EXPERIMENT=re10k_egf_readout bash scripts/train_re10k_readout.sh
```

두 command는 foreground에서 실행되므로 차례로 실행하면 된다.
SSH 연결이 끊겨도 유지하려면 tmux 세션 안에서 실행한다.
새 launcher는 저장된 `wandb login` 또는 외부에서 설정한 `WANDB_API_KEY`를 사용하며,
기존 launcher의 API key를 읽거나 덮어쓰지 않는다.
Checkpoint 기본 경로는 다음과 같다.

```text
/workspace/research/GaussianGenerator/pretrained_weights/Final++_re10k.ckpt
```

경로가 다르면 launcher에 override를 전달한다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train_re10k_readout.sh \
  model.encoder.pretrained_weights='/different/path/Final++_re10k.ckpt'
```

Launcher 없이 Python을 직접 실행해도 동일하다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=1 python -u -m src.main \
  +experiment=re10k_regression wandb.mode=online
```

시작 로그의 `[GeometryReadout] trainable:` 목록에 `scale_head`와
`regression_head` 이외의 encoder 모듈이 없어야 한다. 모든 shared checkpoint key를
검증하므로 잘못된 Final++ 구조나 불완전한 checkpoint는 학습 전에 실패한다.

## Training and evaluation

- 10,000 optimizer steps; checkpoints at 5,000 and 10,000.
- 현재 `src.main` 흐름을 유지한다. 학습 종료 후 두 checkpoint를 순서대로 평가한다.
- 기존 `re10k_moment`의 dataset roots, train/test splits, augmentation,
  batch size, GPU 설정, dataloader seeds, evaluation index를 사용한다.
- 학습 context 2개, target 1개; 평가 view는 기존
  `assets/evaluation_index_re10k.json`의 고정 index를 사용한다.
- 75k-step view-sampling warm-up만 비활성화하여 최종 45–90 frame 간격으로 시작한다.
- AdamW LR `1e-4`, LR warm-up 500 steps, 기존 cosine scheduler.
- 기존 MSE + 0.05 LPIPS, test pose alignment 100 steps 및 metric 집계를 유지한다.
- 이미지/영상 저장은 기존 기본값처럼 비활성화한다.

`outputs/exp_regression_readout_re10k/<timestamp>/`와
`outputs/exp_egf_readout_re10k/<timestamp>/`에 config/checkpoints/logs가 저장된다.
W&B의 `auto_eval/ckpt_step` 5000/10000에서 Average 및 overlap별
PSNR/SSIM/LPIPS를 비교한다. 주 비교 시점은 10k로 고정한다.

## W&B logging

SSH launcher는 `wandb.mode=online`으로 `WandbLogger`를 사용한다.
기본 project는 `noposplat`이고 run 이름은 `regression_readout_re10k (<timestamp>)`,
`egf_readout_re10k (<timestamp>)`이다. 실행 초기에 출력되는 W&B run URL에서 확인한다.
다른 project를 쓰려면 launcher 뒤에 `wandb.project=<project>`를 전달한다.
팀을 지정하려면 해당 팀에 쓰기 권한이 있는 계정으로 로그인한 후
`WANDB_ENTITY=<team>` 환경변수를 사용한다. 현재 학습 entry point는
`wandb.entity` 설정을 logger에 전달하지 않으므로 환경변수로 지정한다.

- 학습: `loss/mse`, `loss/lpips`, `train/psnr_probabilistic`, learning rate.
- 평가: `auto_eval/overall/psnr`, `auto_eval/overall/ssim`, `auto_eval/overall/lpips`.
- Overlap별 평가: `auto_eval/{small,medium,large}/{psnr,ssim,lpips}`.
- 평가 그래프 x축: `auto_eval/ckpt_step` (5000, 10000).

평가 결과는 10k 학습 종료 후 해당 학습 run에 기록된다. 이전 checkpoint의
5000을 W&B의 전역 history step으로 사용하지 않고 별도 metric 축으로 기록하므로,
이미 10k까지 기록된 학습 로그 때문에 5k 평가값이 누락되지 않는다.
온라인 업로드에는 서버의 W&B 인증과 네트워크 연결이 필요하다.

## Regression and shared appearance

기존 sparse graph의 outgoing edge `neighbors[j,k]=i`를 그대로 사용한다.
각 slot은 모든 incoming supports를 읽으며, 새 kNN이나 assignment는 만들지 않는다.
Scene-normalized relative position 3개와 squared distance를 feature와 함께
`(D+4) -> 128 -> 128` evidence MLP로 변환한 뒤 기존 geometry weights로 pooling한다.
`[slot feature, geometry-pooled feature, pooled evidence, log(mass+epsilon)]`
를 `-> 256 -> 256 -> 9` MLP에 입력한다.

3개 출력은 support point 기준 center offset, 나머지 6개는 Cholesky 성분이다.
대각은 softplus로 양수화하고 `L L^T`를 만든다. 양쪽 모두 기존 coverage scale,
covariance floor/roundoff guard 및 scene scale 복원을 동일하게 적용한다.
초기 Cholesky std는 scene scale 대비 0.01이며
`model.encoder.moment_decoder.regression_init_scale`로 조정할 수 있다.

Appearance reader는 두 실험에서 모두 기존 fitted mean을 기준 위치로 사용한다.
Regression head는 fitted mean/covariance를 입력으로 받지 않는다.
따라서 frozen BEAM evidence, SH, opacity는 같고 renderer에 넘기는
center/covariance만 다르다. 공통 경로는 no-grad로 계산하고, frozen 모듈은
train/eval 전환 뒤에도 eval 상태를 유지한다.

이 실험은 EGF와 함께 학습된 Final++ evidence를 고정한 조건부 비교다.

## CPU verification

기존 프로젝트 의존성이 설치된 환경에서 실행한다. CUDA renderer는 import하지 않는다.

```bash
python -m unittest discover -s tests -p test_geometry_readout.py -v
```

실제 renderer를 포함한 GPU 학습 및 서버 checkpoint/data 경로 검증은 서버에서 수행한다.
