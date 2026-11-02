# RE10K → DL3DV cross-domain evaluation

RE10K full-train 체크포인트를 **추가 학습 없이** DL3DV `test` split에서 평가한다.
`re10k_moment`의 현재 Final++ 구조와 `re10k`의 NoPoSplat 구조를 그대로 사용한다.
학습 설정 및 기존 평가 스크립트는 변경하지 않는다.

서버에서 해당 브랜치와 학습에 사용한 Python 환경을 준비하고, 저장소 루트에서 실행한다.
기본 체크포인트 위치는 다음과 같다. 실제 위치가 다르면 아래 옵션으로 지정한다.

```text
pretrained_weights/Final++_re10k.ckpt
pretrained_weights/re10k.ckpt
/workspace/datasets/dl3dv/test/*.torch
/workspace/datasets/dl3dv/test/index.json
```

```bash
# 경로 및 두 모델의 Hydra 설정만 검증한다. GPU 사용/평가/출력 저장은 하지 않는다.
bash scripts/eval_re10k2dl3dv.sh --gpus 0 1 --dry-run

# Final++: GPU 0, baseline: GPU 1. 두 평가를 동시에 시작하고 완료까지 기다린다.
bash scripts/eval_re10k2dl3dv.sh --gpus 0 1

# 체크포인트 위치를 직접 지정하는 경우
bash scripts/eval_re10k2dl3dv.sh --gpus 0 1 \
  --ours-checkpoint '/path/to/Final++_re10k.ckpt' \
  --baseline-checkpoint /path/to/re10k.ckpt
```

Python 직접 실행도 동일하다: `python -m scripts.eval_re10k2dl3dv --gpus 0 1`.
각 프로세스에는 `CUDA_VISIBLE_DEVICES`로 GPU 한 대만 노출하고 Lightning도 `devices=1`로
고정한다. GPU 2는 사용하지 않는다. W&B 연결이나 기존 학습 run 갱신은 하지 않는다.
모든 체크포인트는 Lightning으로 **strict loading**하므로 구조가 맞지 않는 가중치는 오류로 종료한다.
전체 체크포인트로 복원하므로 MASt3R 초기화용 `.pth` 파일은 따로 필요하지 않다.

## 평가 뷰와 설정

저장소에는 DL3DV용 context/target 평가 인덱스가 없다. 데이터셋의 `test/index.json`은
`scene → chunk filename` 매핑이므로 평가 뷰 인덱스로 사용할 수 없다.

`--index`를 지정하지 않으면 CPU에서 `.torch` 파일을 한 번씩 읽어 **두 모델이 공유하는**
평가 인덱스를 만든다. 기본 프로토콜은 다음과 같다.

- 장면마다 context 2장: `[0, 22]`. 기존 DL3DV bounded sampler의 테스트 context 간격을 따른다.
- 장면마다 target 3장: `[5, 11, 16]`. 입력과 겹치지 않는 중간 프레임만 평가한다.
- 숫자는 `.torch`에 저장된 프레임 배열의 0-based 위치이다. 원본 영상의 timestamp가 아니다.
- 23프레임 미만 장면은 `null`로 기록하고 제외한다. 제외 목록은 `evaluation_index_dl3dv.protocol.json`에 남긴다.
- `--context-gap 22 --num-targets 21`을 주면 context 사이의 모든 프레임을 평가한다.

이는 **이번 실험용 고정 프로토콜이며 공식 DL3DV benchmark 프로토콜을 의미하지 않는다.**
기존 논문/실험과 직접 비교하려면 해당 실험의 뷰 인덱스를 `--index`로 지정한다.
인덱스는 실행 결과 폴더에 복사해 고정한다.

```bash
bash scripts/eval_re10k2dl3dv.sh --gpus 0 1 \
  --index /path/to/evaluation_index_dl3dv.json
```

```json
{
  "scene-key": {"context": [0, 22], "target": [5, 11, 16]}
}
```

공통 설정은 입력/출력 **256×256**, batch size 1, seed 111123, 데이터 로더 worker 4개/모델,
PSNR·SSIM·LPIPS이다. DL3DV의 원본 크기 설정 `[270, 480]`과 기존 crop/카메라 정규화를 사용한다.
기존 RE10K 모델의 `make_scale_invariant=true`도 유지한다.

기본값은 기존 `src.main`과 같은 **target pose alignment 100 steps**이다. 이 과정은 target
정답 이미지와 기존 MSE/LPIPS loss로 카메라 pose를 최적화하며, 모델 가중치는 학습하지 않는다.
pose alignment 없이 평가하려면 두 모델에 공통으로 `--no-align-pose`를 지정한다.
생성한 인덱스에는 실제 overlap 측정값이 없으므로 overlap별 점수는 출력하지 않는다.
지표는 기존 `ModelWrapper.test_step`과 동일하게 target 평균 후 장면 평균을 낸다.

## 결과와 로그

기본 출력 위치는 `outputs/cross_domain/re10k2dl3dv/<실행시각>/`이다.
`--output-dir /path/to/new_run`으로 직접 지정할 수 있다. 기존 결과가 있는 폴더는 덮어쓰지 않는다.

```text
<run>/
  evaluation_index_dl3dv.json
  evaluation_index_dl3dv.protocol.json  # 자동 생성한 경우
  ours/
    config.yaml
    console.log
    progress.json
    evaluated_scenes.json
    coverage.json
    metrics.json
  baseline/
    config.yaml
    console.log
    progress.json
    evaluated_scenes.json
    coverage.json
    metrics.json
  renders/{ours,baseline}/             # benchmark / peak memory, 선택적으로 이미지
  summary.json
  summary.csv
```

실행 터미널에 모델별 완료 장면 수, 진행률, 경과 시간과 ETA를 출력한다.
평가 장면이 완료될 때 이전 출력으로부터 10초 이상 지났으면 갱신하며,
시작/종료 상태는 즉시 기록한다.
같은 정보는 각 모델의 `console.log`와 `progress.json`에도 저장한다.

```text
[Progress][ours] evaluating | 35/140 scenes (25.0%) | elapsed=00:05:00 | ETA=00:15:00
[Progress][baseline] evaluating | 70/140 scenes (50.0%) | elapsed=00:04:00 | ETA=00:04:00
```

ETA는 평가 시작 이후 장면당 평균 시간으로 계산한 추정치이다. 첫 장면이 완료되기 전에는
`--:--:--`로 표시하며, 초기 CUDA/컴파일 및 장면별 처리 시간 차이로 초반 추정은 변동될 수 있다.
모델/체크포인트 초기화 시간은 평가 경과 시간에 포함하지 않는다.

이미 실행 중인 이전 버전 프로세스에는 새 ETA 표시가 적용되지 않는다.
해당 실행의 기존 로그는 별도 터미널에서 다음과 같이 확인한다.

```bash
tail -f <run>/ours/console.log <run>/baseline/console.log
```

`metrics.json`에는 체크포인트 경로,
평가 인덱스 SHA-256, pose alignment 설정, 장면/target 수, 최종 점수가 저장된다.
두 모델이 동일한 모든 요청 장면과 뷰를 평가했을 때만 비교표를 저장한다.
shape/FOV/baseline/손상 이미지 필터 때문에 요청한 장면이 누락되면 `coverage.json`과 로그를
남기고 오류로 종료하므로, 빈 평가나 서로 다른 장면의 평균을 정상 결과로 취급하지 않는다.

이미지와 비교 패널도 저장하려면 `--save-image`를 추가한다.
단일 모델만 실행할 수도 있다.

```bash
bash scripts/eval_re10k2dl3dv.sh --models ours --gpus 0
bash scripts/eval_re10k2dl3dv.sh --models baseline --gpus 1
```

검증: `python -m unittest discover -s tests -p 'test_cross_domain_eval.py'`.
이 검증에는 Hydra/OmegaConf만 필요하며 서버의 CUDA나 실제 데이터/체크포인트는 필요하지 않다.
