# DPT256 + student matching + Sim(3)

`re10k_moment`는 DPT의 **256채널 출력을 그대로** moment decoder에 전달한다.
DPT 출력 head는 `Identity`이며, 32차원 축소는 matching 전용 descriptor에서만 일어난다.
기존 최적화된 kNN/배분/moment/SH 계산은 변경하지 않았다.

## 실행

기존 `scripts/train_re10k.sh`는 기본적으로 `+experiment=re10k_moment`를 선택한다.
따라서 기존 실행 방식으로 새 실험이 실행된다. 이 브랜치의 새 학습부터 사용한다.
이전 decoder-only checkpoint는 matcher 파라미터가 없으므로 완전한 resume checkpoint가 아니다.
학습을 재개할 때는 이 버전으로 저장한 checkpoint를 사용한다.

RoMa가 현재 환경에 설치되어 있지 않으면 저장소 루트에서:

```bash
python -m pip install -e ./RoMaV2
python -c "from romav2 import RoMaV2; print('RoMaV2 import OK')"
```

RoMa의 가중치와 backbone 가중치는 최초 초기화 때 필요하다.
서버에서 기존 RoMa 확인을 이미 마쳤다면 같은 환경/cache를 사용할 수 있다.
RoMa 추론이나 GPU 학습 속도는 CPU 단위 테스트로 검증할 수 없다.

## 실제 흐름

1. NoPoSplat이 view A/B의 point map과 DPT feature를 생성한다.
2. B를 16×16 cell로 나누어 중심 정수 픽셀 256개를 선택한다.
3. A의 32×32 후보를 coarse matching하고, 선택 위치 주변 16×16 픽셀에서 fine matching한다.
4. 기본 query confidence가 높은 64개 cell에서 네 개의 고정 quarter 위치를 추가한다.
   기본 결과와 A descriptor map을 재사용하며, 추가 query 256개만 matching한다.
5. 같은 A 픽셀에 대응하는 query는 confidence가 가장 높은 하나를 fitting에 사용한다.
   동률이면 query 순서로 결정한다. 최종 정합 대응은 최대 512개다.
6. confidence로 weighted Sim(3)를 fitting하고, 잔차로 Cauchy 재가중한 뒤 한 번 더 fitting한다.
7. A는 유지하고, B의 **모든** 지지점에 하나의 `sR+t`를 적용한다.
8. 전체 점군에서 기존 global 3D kNN(k=16, self 포함)과 moment decoder를 실행한다.

Matching confidence는 정합용이다. Gaussian opacity나 지지점 제거에 사용하지 않는다.
Non-overlap 지지점도 보존되고, 해당 view의 같은 변환을 받는다.
단일 view kNN으로 나누는 분기는 없다. 수치적 퇴화/대응점 부족 때만 identity 변환을 사용한다.
정렬은 전체 scale/rotation/translation을 교정하며, 비선형 깊이 왜곡을 보장해서 고치지는 않는다.

첫 구현의 grid matcher는 두 view의 pixel support adapter다. `weighted_sim3.py`는 임의의
3D 대응점과 가중치를 받는다. Token/voxel/N-view 포팅에는 후보 탐색 및 view 연결 adapter가 추가로 필요하다.

## 학습과 gradient

- RoMa `fast`는 context RGB B→A를 **GPU별 step당 한 장면**에서만 계산한다.
  Teacher는 encoder 정규화 전 RGB를 받고, target pose/RGB를 사용하지 않는다.
- 기본/추가 query 모두 같은 dense teacher field에서 label을 읽는다.
  RoMa의 별도 random/balanced `sample()`과 KDE는 호출하지 않는다.
- Teacher loss: coarse CE + 유효 window 안의 fine CE + 선택된 대응의 correctness confidence BCE.
- 정합 loss: 학생 대응으로 구한 변환이 RoMa 대응의 3D 점들을 맞추도록 지도한다.
  원본 point/feature는 detach하므로 이 보조 loss가 앞단 geometry를 붕괴시키는 경로를 차단한다.
- 메인 배치의 matching/fitting은 no-grad. 구한 변환을 **live point tensor**에 적용하므로
  rendering loss는 backbone/point head/DPT/moment decoder까지 전달된다.
- Teacher 장면 하나의 작은 matcher/fitting만 다시 gradient와 함께 실행한다.
  Rendering loss가 matcher까지 역전파되지는 않는다.
- 처음 `warmup_steps: 1000`에는 원래 좌표로 photometric 학습을 계속하면서 matcher를 준비한다.
  1000 step부터 학생 변환을 적용한다. 별도의 학습 step을 추가하지 않고 kNN은 항상 global이다.
  이 고정 warm-up 길이는 성공 보장 기준이 아니므로 아래 audit 지표로 확인한다.
- 추론에는 학생 matcher/Sim(3)만 사용한다. RoMa는 import/초기화하지 않으며 checkpoint에도 포함되지 않는다.

## Bilinear와 비용

Feature, query, fine window, XYZ는 모두 정수 index gather다. XYZ 보간은 하지 않는다.
추가한 bilinear는 **RoMa의 3채널 field(대응 xy + confidence)**를 query 위치에서 읽는
`grid_sample(..., align_corners=False)`뿐이다. Teacher loss용 최대 512개와
로깅 주기의 독립 audit 256개 위치에 사용하며 no-grad다.
이는 RoMa/DPT가 자체적으로 수행하는 기존 resize 연산과 별개다.

A descriptor map은 재사용하고, full-image `unfold` 없이 선택한 window만 모은다.
큰 feature map에는 64차원 projection 뒤 LayerNorm을 적용해 256채널 LN 임시 버퍼를 피한다.
보조 rendering, CPU pose 추정, 64회 RANSAC 반복은 없다.
RoMa forward 자체의 비용은 남는다. 실제 step time은 아래 wall-time 지표와 GPU 측정으로 비교해야 한다.

## W&B에서 우선 볼 지표

기본 50 step마다 수집한다. 여러 GPU의 값은 한 번의 packed all-reduce로 평균한다.
**유효 측정이 없으면 NaN(차트 공백)이며, 0 residual로 성공처럼 표시하지 않는다.**

| 지표 | 해석 |
|---|---|
| `align/warmup` | 1이면 변환 적용 전 matcher 준비 구간. Gaussian 학습은 진행 중 |
| `align/applied_fraction` | 실제로 유효한 학생 변환을 적용한 장면 비율 |
| `align/fit_valid_fraction` | 수치적으로 Sim(3)를 계산할 수 있는 비율. **정답 정합률이 아님** |
| `align/identity_fallback_fraction` | 대응점 부족/퇴화로 identity를 사용한 비율 |
| `align/unique_pairs`, `align/effective_pairs` | 중복 제거 후 대응 수, confidence 편중을 반영한 유효 대응 수 |
| `align/scale`, `align/rotation_deg`, `align/translation_relative` | 변환 크기. Translation은 A 점군 RMS 반경으로 정규화 |
| `align/student_residual_before`, `align/student_residual_after` | 학생 자신의 대응에서 계산한 잔차. 이것만 낮아져서는 충분하지 않음 |
| `align_teacher/valid_pairs` | RoMa confidence/좌표 검사를 통과한 학습 대응 수 |
| `align_teacher/coarse_accuracy` | Coarse 후보 분류 정확도 |
| `align_teacher/fine_window_hit_fraction` | RoMa 대응 픽셀이 학생 fine window 안에 있는 비율 |
| `align_teacher/pixel_accuracy`, `align_teacher/pixel_error` | RoMa 기준 최종 matching 정확도/픽셀 오차. 정확도 허용치는 `correctness_pixels`(기본 2px) |
| `align_teacher/confidence_correct`, `align_teacher/confidence_wrong` | 올바른/틀린 대응의 학생 confidence. 둘의 분리가 필요 |
| `align_teacher/audit_pairs`, `align_teacher/audit_measured_pairs` | 독립 audit에서 유효한 teacher 대응 수와 실제 측정 가능한 수 |
| `align_teacher/audit_residual_before`, `align_teacher/audit_residual_after` | **Fitting/CE에 쓰지 않은 고정 query**에서 RoMa 대응 3D 오차의 전후 비교 |
| `align_teacher/audit_after_before_ratio` | 1보다 작으면 해당 teacher 기준 정합 오차 감소. 전후 기준은 같은 유효 점 집합 |
| `align_teacher/audit_improved_fraction` | Audit 대응 중 3D 오차가 줄어든 점의 비율 |
| `align_teacher/loss_*`, `loss/alignment_teacher` | 보조 loss 구성과 가중 합 |
| `align_perf/step_wall_ms_mean` | Batch 시작 사이의 평균 실제 시간. Data/DDP/backward 등을 포함 |
| `align_perf/teacher_ms` | RoMa 준비의 CUDA event 시간. 첫 초기화 측정은 steady-state 비교에서 제외 |
| `align_perf/student_forward_ms` | 메인 배치 matching/fitting/좌표 적용 시간 |
| `align_perf/supervision_forward_ms` | 한 장면의 보조 forward 시간. 보조 backward는 포함하지 않음 |
| `align_perf/torch_peak_allocated_gib` | 프로세스의 PyTorch peak allocated. CuPy/NCCL 제외, step별 reset하지 않음 |

Audit query는 각 cell의 1/8 위치로, 기본 중심 및 추가 quarter 위치와 다르다.
Audit은 정답 geometry 평가가 아니라 **RoMa prior에 대한 독립 확인**이다.
PSNR/SSIM/LPIPS 개선 여부도 함께 봐야 한다. Warm-up 동안 audit은 적용 전 학생 변환을 평가한다.

Timing은 완료된 CUDA event를 다음 batch에서 읽으며 로깅 때문에 `cuda.synchronize()`를 호출하지 않는다.
Stage 시간은 직전 측정 구간의 값일 수 있다. Wall-time의 최초 모델 초기화 구간도 제외하여 비교한다.

## 조절할 설정

- `train.descriptor_teacher.warmup_steps`: 변환 적용 전 준비 구간. 초기값 1000.
- `every_n_steps`: teacher 실행 간격. 기본 1. 진단 로깅 step에는 teacher를 실행해 지표 공백을 방지한다.
- `log_every_n_steps`: 기본 50. Trainer의 W&B flush 주기와 맞추는 것을 권장한다.
- `match_weight`, `alignment_weight`: 초기값 각각 0.05.
- `model.encoder.support_alignment.enabled=false`와 `train.descriptor_teacher.enabled=false`를
  함께 지정하면 DPT256 decoder-only 대조 실험이 된다.
- `train.descriptor_teacher.enabled=false`만 지정하면 저장된 학생 matcher를 쓰되 teacher 학습을 생략한다.

## 검증

```bash
python -m unittest discover -s tests -p 'test*moment*.py' -v
python -m unittest discover -s tests -p test_support_alignment.py -v
```

CPU 테스트는 정합 복원/반사 방지/퇴화/gradient, sampling, teacher 좌표 변환,
학습 경로와 로깅 연결을 확인한다. CUDA kNN 테스트와 실제 RoMa/GPU throughput은 서버에서 확인한다.
