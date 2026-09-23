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

`knn_backend: auto`는 CUDA에서 CuPy 14의 GPU `KDTree`로 트리 구성과 검색을
모두 수행한다. `eps=0, p=2`인 정확한 Euclidean kNN이며, 전체 `M×M` 거리 행렬은
생성하지 않는다. 모든 view의 후보, self 포함 K=16, 출력 슬롯 수를 유지한다.
DLPack과 PyTorch의 현재 CUDA stream을 공유하여 매 scene의 CPU 검색과 왕복 전송을
없앴다. 최초 scene에서만 32개 query의 거리를 SciPy 결과와 대조한다.
동일 거리에 있는 후보의 선택 순서는 두 라이브러리에서 다를 수 있다.
CPU 입력은 기존 SciPy 경로를 사용한다. CUDA에서 CuPy가 없으면 설치 안내와 함께
중단하며, 느린 CPU 경로로 자동 전환하지 않는다.

현재 기본값 `knn_query_backend: specialized`에서는 CuPy가 만든 동일한 트리를
3D/K=16 전용 CUDA kernel로 검색한다. FP64 정밀도와 정확한 검색 조건은 유지하고,
범용 거리 연산 및 반복적인 global 후보 버퍼 갱신을 줄였다.
K가 16이 아닌 경우는 기존 CuPy query를 사용한다. 기존 검색을 명시적으로 선택하려면
`knn_query_backend: cupy`로 설정한다. 시작 시 실제 CUDA kernel을 작은 검증 데이터의
SciPy 결과와 대조한 후, 기존과 같이 첫 실제 scene에서도 표본 검사를 한다.
새 kernel의 속도 및 전체 GPU 검증은 서버 실행이 필요하다.

배분 MLP의 첫 선형층은 `W_i f_i + W_j f_j + W_g geometry + bias`로 계산한다.
각 지지점의 feature 변환을 한 번만 수행하고 이웃마다 결과를 모으므로, 기존 수식과
파라미터 이름을 유지하면서 중복 연산을 줄인다. 비선형 활성화는 합산 후 그대로 적용한다.
기존 moment decoder의 가중치를 그대로 불러올 수 있다. 부동소수점 연산 순서가 달라져
bitwise 동일한 결과나 학습 궤적을 보장하지는 않는다.

배분과 moment 계산은 입력 device에서 FP32로 수행한다. 기본 chunk 크기는 32,768이고,
학습 시 non-reentrant activation checkpointing을 적용한다. 전체 `[B,M,K,D]`
이웃 feature를 유지하지 않는다. kNN index와 scene scale만 gradient에서 분리하고,
point·DPT·backbone으로 향하는 gradient는 유지한다.

## 추가 최적화 묶음

K=16 전용 검색과 함께 다음 구현 최적화를 적용했다. 학습 파라미터, 이웃 정의,
배분 softmax, mass 정규화, central covariance, opacity/SH 식은 유지한다.

- `common/sparse_feature_pool.py`: incoming feature pooling의 직접 backward.
  `h_i = Σ_(j,k: neighbor(j,k)=i) w_jk f_j`에 대해
  `∂L/∂f_j = Σ_k w_jk g_neighbor(j,k)`,
  `∂L/∂w_jk = f_j · g_neighbor(j,k)`를 chunk 단위로 계산한다.
  forward 메시지를 checkpoint로 재생성하지 않고, 전체 `[M,K,D]` activation도
  저장하지 않는다. weight를 통한 배분/geometry gradient는 그대로 전달한다.
  mean offset과 covariance, 배분 MLP에는 기존 checkpoint 설정을 유지한다.
- 배분의 source/destination feature projection을 하나의 행렬곱으로 묶는다.
  기존 weight를 이어 붙여 사용하므로 파라미터 및 checkpoint key는 동일하다.
- detached scene scale의 median을 batch 단위로 계산한다. 짝수 지지점 수에서도
  기존과 같은 lower median을 사용한다.
- 각 scene의 moment를 모은 후 scale/SH head와 최종 attribute 변환을 한 번 수행한다.
  새로 생성된 SH linear 출력에 mask를 in-place 적용하여 추가 SH 텐서를 만들지 않는다.
  `attributes_calls`는 local batch마다 1이며, kNN/배분/aggregation은 여전히 scene별이다.
- moment 분기의 point head와 feature head가 FP32 token 변환 결과를 공유한다.
  dtype 변환이 필요한 경우의 중복 복사와 backward를 줄인다. 기존 baseline 분기는 유지한다.

pooling은 독립 dense 식 및 수치 미분으로 1차·2차 gradient를 검사했다.
batch attribute 변환도 기존 scene별 식과 출력·gradient를 비교했다.
별도 변경 전 코드와 전체 decoder 출력 및 모든 입력/파라미터 gradient를 비교했다.
GPU의 atomic accumulation과 행렬곱 연산 순서가 달라질 수 있으므로 bitwise 동일한
학습 궤적을 보장하지는 않는다. GPU 속도·peak memory의 실제 변화는 측정이 필요하다.

검토했지만 이 묶음에 적용하지 않은 변경은 checkpoint 전체 해제, batch 전체의
이웃 feature 동시 전개, blanket `torch.compile`, 낮은 정밀도와 approximate kNN이다.
메모리 사용이나 수치적 동작이 달라질 수 있어 현 측정만으로 일괄 활성화하지 않았다.

## 실행

기존 실행 파일의 기본 실험을 `re10k_moment`로 연결했다. CUDA 12.8 학습 환경에서
추가 의존성을 한 번 설치한 다음 기존 명령으로 실행한다.

```bash
python -m pip install -r requirements-fast.txt
sbatch scripts/train_re10k.sh
```

CuPy 14.2는 **NumPy 2 이상**을 요구한다. NumPy 1 기반 환경에서는 설치 시 버전이
올라가므로 NumPy C ABI를 사용하는 기존 바이너리 패키지의 호환성도 필요하다.
PyTorch를 다시 설치할 필요는 없다. CUDA 12.x wheel과 GPU KDTree API의 근거는
[CuPy 설치 문서](https://docs.cupy.dev/en/stable/install.html)와
[v14.2.0 KDTree 구현](https://github.com/cupy/cupy/blob/v14.2.0/cupyx/scipy/spatial/_kdtree.py)이다.
NVRTC나 CUDA header 탐색 오류가 발생하면 서버의 CUDA 12.8 toolkit 경로가
올바른지 확인한다. 최초 실행의 CUDA kernel 컴파일과 정확성 검사는 정상 학습 step
시간과 구분해야 한다.

변경 후 속도를 한 번에 측정하려면 기존 profiler를 사용한다.

```bash
TRAIN_MODULE=scripts.profile_training EXPERIMENT=re10k_moment \
  sbatch scripts/train_re10k.sh --output outputs/profile_fast.json
```

기존 서버 측정에서 moment forward 약 5.36초 중 CPU kNN이 약 5.07초였다.
위 변경은 그 병목과 배분의 중복 연산을 대상으로 한다. GPU KD-tree의 실제 속도와
전체 학습 시간이 baseline 수준에 도달했는지는 서버에서 측정해야 한다.
PyTorch profiler의 peak memory에는 CuPy 자체 memory pool이 포함되지 않으므로
GPU 전체 사용량과 구분한다. 큰 chunk의 실제 peak memory도 GPU 실행으로 확인해야 한다.

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
python -m unittest discover -s tests -p 'test_moment_fast.py' -v
```

테스트는 실제 decoder/DPT 구현을 사용한다. CPU에서 CUDA renderer import를 피하기 위해
테스트 안에서만 독립적인 package 이름으로 해당 모듈을 로드한다. 검사 대상은 kNN,
배분 질량 보존, incoming moment의 dense 기준값, 빈 슬롯, covariance,
checkpointing 전후 gradient, DPT 역전파, Hydra 설정 및 optimizer 그룹이다.
속도 최적화 테스트는 기존 concat MLP와의 출력·gradient 및 checkpoint key 호환성을
검사한다. CUDA 환경에서는 실제 GPU kNN의 거리, 중복 좌표, 작은 scene,
non-default stream과 여러 device를 추가 검사한다. CUDA가 없으면 GPU 검사는 건너뛴다.
이 테스트는 실제 GPU renderer와 데이터셋을 포함한 전체 학습 검증을 대신하지 않는다.
