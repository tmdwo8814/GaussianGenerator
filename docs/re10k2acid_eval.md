# RE10K → ACID cross-domain evaluation

RE10K에서 학습한 Final++와 NoPoSplat 체크포인트를 추가 학습 없이 ACID `test`에서 평가한다.
모델/loss는 각각 `re10k_moment`, `re10k` 설정을 유지하고, 데이터 경로와 평가 뷰만 ACID로 바꾼다.
ACID에서 학습한 체크포인트는 이 실험에 사용하지 않는다.

서버 저장소 루트에서 기존 Python 환경으로 실행한다.

```bash
bash scripts/eval_re10k2acid.sh --gpus 0 1
```

기본 경로는 다음과 같다.

```text
ours:    pretrained_weights/Final++_re10k.ckpt
baseline: pretrained_weights/re10k.ckpt
dataset: /workspace/datasets/acid/test/*.torch
         /workspace/datasets/acid/test/index.json
views:   assets/evaluation_index_acid.json
```

다른 위치를 사용한다면 명시적으로 지정한다.

```bash
bash scripts/eval_re10k2acid.sh --gpus 0 1 \
  --ours-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/Final++_re10k.ckpt' \
  --baseline-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/re10k.ckpt' \
  --dataset-root /workspace/datasets/acid
```

`--dry-run`을 추가하면 경로/인덱스/모델 설정만 검증한다.
Python 직접 실행은 `python -m scripts.eval_re10k2acid --gpus 0 1`이다.

## 평가 프로토콜

- GPU 0에서 Final++, GPU 1에서 NoPoSplat을 동시에 실행한다. 모델마다 GPU 한 대만 사용한다.
- 기본 평가는 저장소의 `assets/evaluation_index_acid.json`을 사용한다. DL3DV용 간격 22 인덱스를 새로 생성하지 않는다.
- 이 인덱스는 총 1,848개 장면 중 1,341개 평가 장면과 507개 `null` 항목을 포함한다. `null`은 평가하지 않는다.
- 기존 ACID 평가와 비교할 수 있도록 저장된 context/target을 그대로 사용한다. 76개 장면에는 context와 겹치는 target이 포함되어 있다.
- 입력/출력 256×256, batch size 1, seed 111123, 모델당 데이터 로더 worker 4개를 사용한다.
- ACID의 기존 평가와 같이 RE10K 데이터 로더와 원본 크기 `[360, 640]`, crop/카메라 정규화를 사용한다. 내부 설정 이름은 `dataset.re10k`이지만 실제 경로는 ACID이다.
- 기존과 동일한 PSNR·SSIM·LPIPS 및 target pose alignment 100 steps를 사용한다. 모델 가중치는 학습하지 않는다.
- 각 모델 체크포인트를 strict loading하고, 두 모델이 동일한 요청 장면/뷰를 모두 평가했는지 검증한다.

다른 ACID 평가 인덱스를 사용하려면 `--index /path/to/evaluation_index_acid.json`을 지정한다.
데이터셋의 `test/index.json`은 장면별 chunk 파일 매핑이므로 평가 뷰 인덱스로 사용할 수 없다.

## 진행률과 결과

모델별 완료 장면 수, 진행률, 경과 시간과 ETA를 실행 터미널에 출력한다.
장면 완료 시 이전 출력으로부터 10초 이상 지났으면 갱신한다.
초기 컴파일과 장면별 처리 시간 차이 때문에 초반 ETA는 변동될 수 있다.

결과는 `outputs/cross_domain/re10k2acid/<실행시각>/`에 저장된다.

```text
evaluation_index_acid.json       # 두 모델이 공유하는 인덱스 사본
ours/{config.yaml,console.log,progress.json,evaluated_scenes.json,coverage.json,metrics.json}
baseline/{config.yaml,console.log,progress.json,evaluated_scenes.json,coverage.json,metrics.json}
renders/{ours,baseline}/         # benchmark / peak memory, 선택적으로 이미지
summary.json
summary.csv
```

`metrics.json`에는 학습/평가 데이터셋, 실험 설정 이름, 체크포인트 경로, 인덱스 SHA-256,
장면/target 수와 점수를 기록한다. 데이터 필터로 요청 장면이 누락되면 `coverage.json`과
로그를 남기고 오류로 종료한다.

기존 옵션도 그대로 사용할 수 있다.

```bash
bash scripts/eval_re10k2acid.sh --models ours --gpus 0
bash scripts/eval_re10k2acid.sh --models baseline --gpus 1
bash scripts/eval_re10k2acid.sh --gpus 0 1 --save-image
bash scripts/eval_re10k2acid.sh --gpus 0 1 --no-align-pose
```
