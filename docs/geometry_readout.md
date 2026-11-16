# Minimal geometry and opacity regression on frozen BEAM

현재 실행 대상은 re10k_regression 하나다. Final++의 encoder, point/DPT
feature heads, BEAM assignment/budget와 color/SH 경로를 모두 로드하고 고정한다.
Center, covariance, opacity 출력 부분만 새 단일 Linear head로 교체한다.

## Single regression head

기존 BEAM geometry weights로 이미 계산하던 incoming pooled feature를 사용한다.

```text
h_i = sum_j w_ji f_j                 # 256 channels, frozen
Linear(256, 10)                     # 2,570 trainable parameters
  - 3: center offset
  - 6: lower-triangular covariance factor
  - 1: opacity logit
```

추가 evidence MLP, feature projection, hidden layer, slot-feature concatenation은
모두 제거했다. 새 이웃 탐색이나 assignment도 추가하지 않는다.
학습 가능한 파라미터는 gaussian_decoder.regression_head.readout.weight와
bias뿐이다. 기존 scale_head는 Final++ 값으로 고정하고 regression 출력에는
적용하지 않는다. Covariance의 크기는 6개 출력이 직접 결정한다.

- Center: 정규화된 원래 support point에 예측한 3D offset을 더한다.
- Covariance: Cholesky 대각을 softplus로 양수화하고 L @ L.T를 구성한다.
  기존 covariance floor/roundoff guard와 scene-scale 복원을 적용한다.
- Opacity: 예측한 logit의 sigmoid를 사용한다. 1-exp(-mass)를 사용하지 않는다.
- Color: 원래 EGF mean을 기준 위치로 사용하는 frozen appearance/SH 경로를 유지한다.
  회귀한 center나 opacity를 color branch에 다시 입력하지 않는다.

Head는 무작위 초기화하고, covariance 대각 bias는 scene scale 대비 std 0.01로,
opacity bias는 0으로 설정한다. 초기 opacity는 약 0.5다. EGF mean/covariance나
opacity를 회귀 head의 입력/정답으로 제공하지 않는다.
Checkpoint에서 optimizer, scheduler, global step은 가져오지 않는다.
기존 9-output MLP regression checkpoint와는 구조가 다르므로 Final++에서 새로 시작한다.

## SSH: four GPUs, regression only

기존 학습 환경을 활성화하고 repository root에서 실행한다.

```bash
cd /workspace/research/GaussianGenerator
wandb login
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_re10k_readout.sh
```

Launcher는 CUDA_VISIBLE_DEVICES 미설정 시 0,1,2,3을 사용한다.
다른 GPU 네 개를 쓰려면 해당 번호를 지정한다. Experiment의 trainer.devices=4로
학습/평가 GPU 수도 명시한다. SSH 연결 종료 후에도 유지하려면 tmux 안에서 실행한다.
새 launcher는 저장된 W&B 로그인 또는 환경변수 인증을 그대로 사용한다.
Ours 실험을 자동 실행하지 않는다.

Checkpoint 기본 경로:

```text
/workspace/research/GaussianGenerator/pretrained_weights/Final++_re10k.ckpt
```

경로가 다르면 launcher에 override를 전달한다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_re10k_readout.sh \
  model.encoder.pretrained_weights='/different/path/Final++_re10k.ckpt'
```

시작 로그의 [GeometryReadout] trainable_parameters=2570과 trainable 목록을 확인한다.
공통 checkpoint key와 shape가 맞지 않으면 학습 전에 실패한다.

## Training and evaluation

- Regression만 10,000 optimizer steps 학습한다.
- Checkpoint는 5,000과 10,000에서 저장하며, 학습 종료 후 두 checkpoint를 각각 한 번 평가한다.
- 기존 RE10K roots, train/test splits, augmentation, dataloader seed와 evaluation index를 유지한다.
- 입력 해상도 256x256, 학습 context 2개/target 1개, 평가 view는
  assets/evaluation_index_re10k.json을 사용한다.
- 기존 batch size 16/GPU를 유지하므로 4 GPU의 전체 batch는 64다.
- View-sampling warm-up은 0으로 두어 최종 45–90 frame 간격으로 시작한다.
- AdamW LR 1e-4, LR warm-up 500 steps, cosine scheduler를 사용한다.
- MSE + 0.05 LPIPS, test pose alignment 100 steps, metric 집계를 유지한다.
- 이미지/영상 저장은 비활성화한다.

결과는 outputs/exp_regression_minimal_re10k/<timestamp>/에 저장한다.
기존 re10k_egf_readout control 설정은 보존되며, 이번 실행에는 사용하지 않는다.

## W&B logging

wandb.mode=online, 기본 project noposplat, run 이름
regression_minimal_re10k (<timestamp>)를 사용한다.

- 학습: loss/mse, loss/lpips, train/psnr_probabilistic, learning rate.
- 평가: auto_eval/overall/{psnr,ssim,lpips} 및 overlap별 지표.
- 평가 x축: auto_eval/ckpt_step의 5000과 10000, 총 두 지점.

평가 결과는 해당 학습 run에 기록한다. 5k를 W&B의 전역 history step으로
설정하지 않으므로, 10k 학습 뒤 평가해도 과거 step으로 처리되어 누락되지 않는다.
온라인 업로드에는 서버의 W&B 인증과 네트워크 연결이 필요하다.
다른 project는 wandb.project=<project>, 팀은 WANDB_ENTITY=<team>으로 지정한다.

## Verification

```bash
python -m unittest discover -s tests -p test_geometry_readout.py -v
python -m unittest discover -s tests -p test_auto_eval_logging.py -v
```

CPU 테스트는 단일 Linear와 파라미터 수, opacity 출력/gradient, color 유지,
공통 weight 고정, covariance 유효성, checkpoint replay와 평가 설정을 검증한다.
실제 GPU 학습 및 서버의 checkpoint/data 경로는 서버에서 확인한다.
