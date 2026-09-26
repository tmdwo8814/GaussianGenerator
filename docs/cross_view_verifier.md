# RoMaV2 teacher + cross-view descriptor 실험

현재 `re10k_moment`에 활성화되어 있습니다. Gaussian decoder는 기존의
feature 64차원, k=16, 지지점 배분 → moment 생성 구조를 유지합니다.
이 실험에서 바꾸는 부분은 **kNN 이전의 점군 정합과 검증**입니다.

## 실행 흐름

```text
context RGB → 기존 backbone / DPT → view별 points, features
                                        │
                대표 지지점 → 작은 공유 descriptor MLP
                                        │
                   양방향 descriptor 매칭 → robust SE(3) 정합
                                        │
                          별도로 남겨둔 대응점으로 검증
                              ┌─────────┴─────────┐
                            성공                 실패
                   두 번째 점군 정합        원래 점군 유지
                   전체 3D kNN              view별 3D kNN
                              └─────────┬─────────┘
                             기존 moment decoder
                                        │
                             기존 renderer / loss
```

- 대표점은 view당 최대 1,024개입니다. 현재 pixel adapter는 균일한 2D 격자로
  선택합니다. 전체 131,072개 지지점끼리 descriptor 행렬을 만들지 않습니다.
- descriptor는 `LayerNorm → Linear → GELU → Linear → L2 normalization`이고,
  출력은 32차원입니다. Gaussian 생성에 쓰는 feature 64차원과는 별개입니다.
- cosine similarity, 1·2순위 차이, mutual matching으로 대응점을 선택합니다.
  대응점을 찾을 때 아직 정합되지 않은 3D 거리를 사용하지 않습니다.
- `x_b @ R.T + t ≈ x_a`인 강체 변환을 구합니다. 두 점군은 이미 NoPoSplat의
  canonical 좌표계로 예측되므로, **R,t는 예측 오차의 보정 변환**입니다.
  카메라 relative pose가 아니며 scale은 변경하지 않습니다.
- 대응점 3/4로 RANSAC·weighted Kabsch 정합, 나머지 1/4로 검증합니다.
  대응점 수, 검증 inlier 비율, 점들의 3D 분포를 검사합니다.
  평면은 허용하지만 한 선이나 좁은 영역에 몰린 정합은 거절합니다.
- 거리 허용치는 두 점군의 **대표점 최근접 거리 중앙값 중 작은 값 × 0.5**입니다.
  좌표 원점과 전역 scale에 의존하지 않도록 한 초기 기준입니다. 원본 지지점의
  kNN 반경과 같지 않으며, 모든 이웃의 정확성을 보장하는 기준도 아닙니다.
- 실패하면 두 view의 Gaussian을 모두 생성하되, 배분할 이웃만 view 내부로 제한합니다.
  성공하면 정합한 점군에서 기존 cross-view kNN을 사용합니다.

정합은 점군 단위입니다. 성공한 장면의 비중첩 영역까지 개별적으로 검증하거나,
잘못된 모든 국소 변형을 고치지는 않습니다. 영역별 필터와 adaptive slot 수는
이번 버전에 포함하지 않았습니다. NoPoSplat 연결부는 기존처럼 2-view이며,
새 verifier 본체는 pixel 좌표 대신 대표 지지점 index를 받아 token/voxel에 연결할 수 있습니다.

## 학습 경로

```text
실제 augmented context RGB [0,1] → frozen RoMaV2 → 대응 pixel 쌍
                                                         │
DPT feature (detach) → 대응 위치에서 sampling → descriptor → 양방향 contrastive CE
```

한 rank에서 step마다 batch의 **한 scene**만 순환 선택합니다. RoMaV2를 encoder
forward 전에 실행하여 teacher의 중간 activation과 학습 그래프가 겹치지 않게 했습니다.
teacher weight 자체는 GPU에 상주하므로 추가 메모리와 학습 시간이 필요합니다.

RoMaV2의 balanced sampling 결과 중 최대 512쌍을 지도에 사용합니다. 같은 위치나
가까운 위치의 쌍은 서로 negative로 취급하지 않습니다. 유효한 쌍이 부족하면
descriptor loss는 0이며, 모든 descriptor parameter에 0-gradient를 연결해 DDP를 지원합니다.

학습 손실은 `기존 loss + 0.05 × descriptor loss`입니다. Descriptor loss의 입력
feature를 detach하므로 추가 손실은 descriptor MLP만 학습합니다. Backbone,
point head, DPT, moment decoder는 기존 렌더링 손실로 e2e 학습합니다.
Hard matching·RANSAC·검증 결정도 detach하지만, 채택한 R,t를 적용한 점들은
렌더링 손실의 gradient를 point head로 전달합니다.

예전 aux 버전의 추가 RGB rendering loss, pose 추정, 중간 view 합성은 없습니다.
GT target pose/RGB는 verifier·teacher에 전달되지 않습니다.

추론에서는 checkpoint의 descriptor만 사용합니다. RoMaV2 import, weight 다운로드,
실행은 학습 시작 시에만 발생하며 teacher weight는 checkpoint에 저장하지 않습니다.
학습 초기의 descriptor는 미학습 상태라 대부분 view 내부 kNN으로 시작할 수 있습니다.

## 설치와 실행

기존 aux 환경에서 RoMaV2 실행을 이미 확인했다면 추가 설치 없이 시작할 수 있습니다.
새 환경에서는 working NoPoSplat 환경과 저장소 안의 `RoMaV2/` clone을 준비한 뒤:

```bash
python -m pip install -r requirements-cvverify.txt
```

이는 사용 중인 torch 2.11.0 / torchvision 0.26.0 조합을 명시합니다. GPU용 torch는
기존 CUDA 12.8 설치를 유지하세요. 현재 clone의 pyproject는 Python >=3.10이며,
Linux에서는 `fused-local-corr`를 dependency로 요청합니다.

GPU 할당 안에서 최초 weight 다운로드와 실행 확인을 할 수 있습니다:

```bash
python -m scripts.check_romav2 --setting fast
```

예제 이미지 없이 생성한 self-pair로 실행 여부만 확인합니다. 실제 대응 품질을
보려면 `--image-a a.png --image-b b.png`로 겹치는 이미지 두 장을 지정하세요.
RoMaV2/DINOv3 최초 다운로드에는 네트워크와 쓰기 가능한 torch.hub cache가 필요합니다.

학습은 기존 실행 파일을 사용합니다. 첫 실행에만 preflight를 함께 실행하려면:

```bash
ROMA_PREFLIGHT=1 sbatch scripts/train_re10k.sh
# 준비 완료 후
sbatch scripts/train_re10k.sh
```

짧은 실행 확인(학습 전체 및 20k 간격 평가는 변경하지 않고 CLI로만 제한):

```bash
sbatch scripts/train_re10k.sh trainer.max_steps=100 trainer.auto_eval=false wandb.mode=disabled
```

원래 moment 버전 대조 실험은 두 경로를 함께 끕니다:

```bash
sbatch scripts/train_re10k.sh \
  model.encoder.cross_view_verifier.enabled=false \
  train.descriptor_teacher.enabled=false wandb.name=gaussian_decoder_control
```

기존 model checkpoint에는 descriptor weight가 없으므로 이 버전의 학습을
`checkpointing.load`로 그대로 resume하지 마세요. 기본 backbone 초기화 경로는 유지했습니다.
CVverify checkpoint 평가에서는 verifier를 켜두면 됩니다. Teacher는 test 경로에서 실행되지 않습니다.

## 먼저 볼 로그

- `cv/teacher_pairs`, `cv/teacher_accuracy`: 실제 지도 쌍 수와 양방향 검색 정확도.
- `loss/descriptor`: weight 적용 전의 descriptor loss.
- `cv/matches`: 학생 descriptor가 찾은 대응점 수.
- `cv/accepted`: batch 안에서 정합 검증을 통과한 scene 비율.
- `cv/validation_inliers`: 정합에 쓰지 않은 대응점의 검증 inlier 비율.
- `cv/relative_residual`: 검증 거리 중앙값 / 허용치. 정합 후보조차 만들지 못한
  scene은 0으로 집계하므로 `accepted`·`matches`와 함께 봐야 합니다.

학습 로그에도 `CV verify: accepted=...; matches=...; validation_inliers=...`를 출력합니다.
후반까지 `accepted=0`이면 새 정합이 실제로 활용되지 않는 상태입니다.
반대로 통과율만 높다고 PSNR 개선이 입증되는 것은 아니므로 기존 20k 간격 평가와 함께 판단하세요.

CPU 확인:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

GPU에서의 RoMa 실행, DDP 학습 속도, 실제 정합 통과율과 재구성 성능은 별도 실험으로 확인해야 합니다.
