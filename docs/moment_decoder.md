# Moment Gaussian decoder — 첫 번째 실험

보조 학습 없이 기존 RE10K MSE + 0.05 × LPIPS로 end-to-end 학습한다.
기존 MASt3R 초기화, point heads, renderer, 데이터 및 평가 조건을 유지한다.

## 읽는 순서

1. `src/model/encoder/encoder_noposplat.py`: `moment` 분기의 전체 연결.
2. `src/model/encoder/heads/dpt_feature_head.py`: view별 DPT + RGB feature를 64차원으로 projection.
3. `src/model/encoder/heads/moment_gaussian_decoder.py`: 배분 → moment → Gaussian attributes.
4. `src/model/encoder/common/sparse_knn.py`: 정확한 3D 이웃 탐색.
5. `config/experiment/re10k_moment.yaml`: decoder의 초기 설정.

```text
context RGB → 기존 backbone
                ├─ 기존 point heads → X1, X2
                └─ view별 DPT + RGB merger + 1×1 projection → F1, F2
                           ↓ view 순서대로 연결
              points [B,M,3], features [B,M,64]
                           ↓ MomentGaussianDecoder
                  means / covariances / SH / opacities
                           ↓ 기존 renderer + target camera
                        MSE + LPIPS
```

## 핵심 수식과 코드의 대응

지지점 `j`는 자신의 RGB 픽셀에서 나온 point와 feature를 가진다.
모든 view의 point는 이미 동일한 NoPoSplat canonical 좌표계에 있다.
첫 실험은 2-view, 픽셀당 슬롯 하나로, `M = 2 × H × W`이다.
코어 decoder는 view 수에 의존하지 않는 `[B,M,3]`, `[B,M,D]` 인터페이스를 가진다.
다른 모델에서는 해당 모델의 공통 좌표계 지지점과 대응 feature를 이 인터페이스에 연결한다.

아래 식은 원래 좌표계로 표기한다. 코드에서는 detached scene scale
`L = max(median(||x_j||), scene_epsilon)`로 정규화하여 계산한 후,
최종 중심에 `L`, covariance에 `L²`를 곱한다.

- `build_knn`: `neighbors[j,k] = i`. 지지점 `j`가 질량을 보낼 후보 슬롯 `i`.
  self를 첫 열에 두고 최대 16개를 선택한다. 후보는 모든 입력 view에 걸쳐 검색한다.
- `predict_allocation`: `b_j = budget_max × sigmoid(budget_head(f_j))`.
  `allocation_head([f_i, f_j, (x_j-x_i)/L, ||(x_j-x_i)/L||²])`의 logit을
  **각 지지점의 후보 방향**으로 softmax하여 `A_ij`를 얻고, `q_ij = A_ij b_j`로 배분한다.
- `aggregate_moments`: **목적지 슬롯 방향으로** 모아 `m_i = Σ_j q_ij`를 계산한다.
  `w_ij = (q_ij + ε·1[i=j]) / (m_i + ε)`로 moment만 안정화한다.
  `μ_i = Σ_j w_ij x_j`, `C_i = Σ_j w_ij (x_j-μ_i)(x_j-μ_i)ᵀ`,
  `h_i = Σ_j w_ij f_j`를 계산한다.
- `build_gaussians`: `s_i`는 `h_i`에서 예측하며 범위는 `[1,3]`, 초기값은 `1.75`이다.
  `Σ_i = s_i² C_i + (covariance_floor × L)² I`,
  `o_i = 1-exp(-m_i)`, `SH_i = SH_head(h_i) × 기존 SH mask`로 생성한다.

FP32에서 작은 covariance floor가 반올림으로 사라지지 않도록, 코드에는
covariance trace에 비례하는 작은 detached 수치 보정도 포함되어 있다.
mean 계산에 등장하는 `points + mean_offsets`는 가중 평균을 안정적으로 계산한 것이다.
별도로 위치 residual을 예측하는 head는 없다.

한 슬롯이 받는 지지점 수는 16개보다 많을 수 있다. epsilon은 opacity 질량에
더하지 않는다. 기존 opacity mapping과 raw Gaussian adapter는 새 출력에 적용하지 않는다.
Gaussian 출력 순서는 원래 지지점 순서를 유지한다.

## 메모리와 gradient

정확한 kNN은 SciPy `cKDTree`로 CPU에서 계산하고 index를 원래 device로 돌려준다.
전체 `M×M` 거리 행렬은 생성하지 않는다. `scipy>=1.9`를 의존성에 명시했다.
이 구현에는 CPU 탐색 및 GPU↔CPU 전송 비용이 있으며, 서버에서 측정해야 한다.

배분과 moment 계산은 입력 device에서 FP32로 수행한다. 기본 chunk 크기는 8,192이고,
학습 시 non-reentrant activation checkpointing을 적용한다. 전체 `[B,M,K,D]`
이웃 feature를 유지하지 않는다. kNN index와 scene scale만 gradient에서 분리하고,
point·DPT·backbone으로 향하는 gradient는 유지한다.

## 실행

기존 실행 파일의 기본 실험을 `re10k_moment`로 연결했다.

```bash
sbatch scripts/train_re10k.sh
```

학습 서버에서 짧은 연결 확인:

```bash
sbatch scripts/train_re10k.sh trainer.max_steps=2 trainer.auto_eval=false data_loader.train.batch_size=1 wandb.mode=disabled
```

기존 baseline 선택:

```bash
EXPERIMENT=re10k sbatch scripts/train_re10k.sh
```

기본 batch size와 학습 길이는 baseline과 같다. 동일 batch size가 GPU 메모리에 맞는지는
실제 backbone·renderer를 포함한 서버 실행으로 확인한다. 자동 평가는 기존과 같이
전체 학습 후 checkpoint별로 수행하며, 기존 target pose alignment 설정도 유지한다.

## 검증

```bash
python -m unittest discover -s tests -p 'test_moment_gaussian_decoder.py' -v
```

테스트는 실제 decoder/DPT 구현을 사용한다. CPU에서 CUDA renderer import를 피하기 위해
테스트 안에서만 독립적인 package 이름으로 해당 모듈을 로드한다. 검사 대상은 kNN,
배분 질량 보존, incoming moment의 dense 기준값, 빈 슬롯, covariance,
checkpointing 전후 gradient, DPT 역전파, Hydra 설정 및 optimizer 그룹이다.
이 테스트는 실제 GPU renderer와 데이터셋을 포함한 전체 학습 검증을 대신하지 않는다.
