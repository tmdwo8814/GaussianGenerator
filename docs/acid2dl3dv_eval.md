# ACID → DL3DV cross-domain evaluation

ACID에서 학습한 Final++와 NoPoSplat 체크포인트를 DL3DV `test`에서 평가한다.
Final++는 `acid_moment`, baseline은 `acid` 실험 설정을 사용하고 체크포인트를 strict loading한다.
ACID의 학습/모델/loss 설정을 유지한 채 데이터 로더만 DL3DV로 교체한다.
ACID 설정의 데이터셋 이름도 내부적으로 `dataset.re10k`이므로 이 노드를 제거하고
`dataset.dl3dv`만 평가한다. 서버에 ACID 데이터셋이 따로 없어도 된다.

서버 저장소 루트에서 기존 `noposplat` Python 환경으로 실행한다.
아래 체크포인트 경로와 `/workspace/datasets/dl3dv`가 기본값이다.

```bash
bash scripts/eval_acid2dl3dv.sh --gpus 0 1 \
  --ours-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/Final++_acid.ckpt' \
  --baseline-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/noposplat_acid.ckpt'
```

같은 경로에 파일이 있다면 `bash scripts/eval_acid2dl3dv.sh --gpus 0 1`만 실행해도 된다.
경로 및 설정만 점검하려면 `--dry-run`을 추가한다.
Python 직접 실행도 동일하다: `python -m scripts.eval_acid2dl3dv --gpus 0 1`.

- GPU 0: Final++, GPU 1: NoPoSplat을 동시에 실행한다. GPU 2는 사용하지 않는다.
- 입력/출력 256×256, batch size 1, seed 111123, 모델당 데이터 로더 worker 4개를 사용한다.
- 기존 RE10K → DL3DV와 같은 PSNR·SSIM·LPIPS 및 target pose alignment 100 steps를 사용한다.
- 모델별 완료 장면 수, 진행률, 경과 시간과 ETA를 실행 터미널에 출력한다.
- W&B 연결이나 추가 모델 학습은 하지 않는다.

## 이전 실험과 같은 평가 뷰 사용

기존 RE10K → DL3DV 실행에서 저장한 인덱스를 지정하면 평가 뷰를 그대로 재사용하고,
`.torch` 파일을 다시 읽어 인덱스를 생성하는 과정도 생략한다.

```bash
bash scripts/eval_acid2dl3dv.sh --gpus 0 1 \
  --index 'outputs/cross_domain/re10k2dl3dv/<이전실행시각>/evaluation_index_dl3dv.json'
```

`--index`를 생략하면 기존과 동일한 알고리즘으로 공유 인덱스를 생성한다.
기본 context는 `[0, 22]`, target은 `[5, 11, 16]`이며, 23프레임 미만 장면은 제외 목록에 기록한다.
이는 이번 실험용 고정 프로토콜이다. 이전 실행에서 뷰 설정을 바꿨다면 저장한 인덱스를 지정한다.
데이터셋의 `test/index.json`은 chunk 매핑이므로 `--index`에 지정하는 뷰 인덱스와 다르다.

## 결과

기본 출력은 `outputs/cross_domain/acid2dl3dv/<실행시각>/`이다.
각 모델의 `config.yaml`, `console.log`, `progress.json`, `evaluated_scenes.json`,
`coverage.json`, `metrics.json`과 전체 비교표 `summary.json`, `summary.csv`를 저장한다.
기존 RE10K 평가 코드와 같은 실행/검증/ETA 구현을 재사용한다.
두 모델이 요청한 동일 장면과 뷰를 모두 평가한 경우에만 비교표를 만든다.

공통 옵션도 그대로 사용할 수 있다.

```bash
# 단일 모델 평가
bash scripts/eval_acid2dl3dv.sh --models ours --gpus 0
bash scripts/eval_acid2dl3dv.sh --models baseline --gpus 1

# 예측 이미지 및 비교 패널 저장
bash scripts/eval_acid2dl3dv.sh --gpus 0 1 --save-image

# 두 모델 모두 target pose alignment 비활성화
bash scripts/eval_acid2dl3dv.sh --gpus 0 1 --no-align-pose
```

ETA 계산, 평가 뷰와 누락 검증, 결과 파일에 대한 설명은
[공통 평가 안내](re10k2dl3dv_eval.md)를 참고한다.
