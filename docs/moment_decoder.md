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

## 실행 구현

- `common/sparse_knn.py`: CPU는 SciPy, CUDA는 GPU KD-tree로 정확한 이웃을 찾는다.
  전체 `M×M` 거리 행렬은 만들지 않는다. invalid point는 batch 단위로 검사한다.
- `common/cuda_knn.py`: CuPy 트리 생성, query 선택, self를 첫 열에 배치한다.
  DLPack과 PyTorch의 현재 CUDA stream을 공유하며, 학습 중 CPU 비교 검사는 하지 않는다.
- `common/knn_query16.py`, `knn_query16.cu`: FP64·3D·K=16 전용 검색이다.
  `knn_query_backend: specialized`가 기본값이고, 다른 K에는 CuPy 일반 검색을 사용한다.
  `knn_query_backend: cupy`로 기존 정확한 검색도 선택할 수 있다.
  동일 거리 후보의 선택은 backend에 따라 달라질 수 있다.
- `common/sparse_feature_pool.py`: `h_i = Σ_j w_ij f_j`의 feature와 weight gradient를
  직접 계산한다. 입력 feature, weight, index만 저장하며 `[M,K,D]` 전체 activation은
  유지하지 않는다. 배분과 point로 향하는 gradient도 그대로 연결된다.

배분의 첫 선형층은 `W_i f_i + W_j f_j + W_g geometry + bias`로 계산한다.
각 feature를 한 번만 변환하며 source/destination 변환은 하나의 행렬곱으로 묶는다.
비선형 활성화는 합산 후 적용한다. 기존 파라미터와 checkpoint key는 유지한다.

배분과 moment 계산은 FP32, 기본 chunk는 32,768이다. 배분 MLP·mean offset·covariance에는
non-reentrant checkpointing을 유지하고 feature pooling은 직접 backward를 사용한다.
scene scale과 최종 attribute 변환은 batch 단위로 처리한다. 새 SH 출력에 mask를
in-place 적용하고, point/feature head는 FP32 token 변환을 공유한다.
kNN index와 scene scale만 gradient에서 분리한다.

## 학습과 평가

CUDA 12.8 환경에서 추가 의존성을 설치한 뒤 실행한다. 이미 설치했다면 재설치할 필요 없다.

```bash
python -m pip install -r requirements-fast.txt
sbatch scripts/train_re10k.sh
```

`requirements-fast.txt`는 실행에 필요한 CuPy 14.2를 지정한다. NumPy 2 이상이 필요하며,
최초 실행 시 NVRTC가 전용 CUDA kernel을 컴파일하고 캐시한다.
학습 실행 파일은 `src.main`을 직접 호출한다. 기존 baseline 선택과 설정 override도 지원한다.

```bash
EXPERIMENT=re10k sbatch scripts/train_re10k.sh
sbatch scripts/train_re10k.sh trainer.max_steps=1000
```

기본 batch size와 학습 길이는 baseline과 같다. 기존 자동 평가 및 target pose alignment
설정을 유지한다. GPU 전체 메모리에는 PyTorch 외에 CuPy/NCCL 사용량도 포함된다.

## 검증

```bash
python -m unittest discover -s tests -v
```

테스트는 실제 decoder/DPT 구현을 사용한다. CPU에서는 CUDA renderer import를 피하기
위해 테스트 안에서만 독립적인 package 이름으로 모듈을 로드한다.

- `test_moment_gaussian_decoder.py`: 질량 보존, incoming moment의 독립 dense 기준값,
  빈 슬롯, covariance, checkpoint gradient, DPT 역전파와 encoder/config 연결.
- `test_moment_fast.py`: 기존 배분 식과 출력·gradient 일치, checkpoint 호환성,
  정확한 kNN과 self 처리. CUDA에서는 전용 검색을 CuPy/SciPy와 대조하고
  중복 좌표·작은 scene·non-default stream·여러 device도 검사한다.
- `test_sparse_feature_pool.py`: pooling의 dense 기준값과 1차·2차 수치 미분,
  batch attribute 변환의 출력·gradient 일치.

정확성 비교는 학습 시작 경로가 아닌 테스트에서 수행한다. CUDA가 없으면 GPU 검사는
건너뛴다. 부동소수점 및 atomic accumulation 순서 때문에 bitwise 동일한 학습 궤적을
보장하지는 않는다.
