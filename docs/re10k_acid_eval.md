# RE10K ↔ ACID: four cross-domain evaluations

한 명령으로 네 실험을 동시에 실행하며, 실험마다 GPU 한 대를 사용한다.

```bash
bash scripts/eval_re10k_acid.sh --gpus 0 1 2 3
```

| GPU | 학습 → 평가 | 모델 | 기본 체크포인트 (`pretrained_weights/`) |
| --- | --- | --- | --- |
| 0 | RE10K → ACID | Final++ | `Final++_re10k.ckpt` |
| 1 | RE10K → ACID | NoPoSplat | `re10k.ckpt` |
| 2 | ACID → RE10K | Final++ | `Final++_acid.ckpt` |
| 3 | ACID → RE10K | NoPoSplat | `noposplat_acid.ckpt` |

서버 저장소 루트에서 기존 `noposplat` Python 환경으로 실행한다.
`--dry-run`을 추가하면 네 체크포인트와 두 데이터셋의 평가 인덱스, 네 모델 설정을
모두 검증하고 종료한다. 출력 파일이나 GPU 프로세스는 만들지 않는다.

데이터 경로는 `/workspace/datasets/re10k`, `/workspace/datasets/acid`이며,
각각 `test/*.torch`와 `test/index.json`이 필요하다. 평가 뷰는 저장소의
`assets/evaluation_index_re10k.json`, `assets/evaluation_index_acid.json`을 사용한다.
기존 벤치마크의 `null` 장면과 context/target 구성을 그대로 보존한다.
각 방향의 두 모델은 같은 인덱스 사본을 공유하고 같은 장면/뷰를 평가했는지 검증한다.

모델 설정은 학습 데이터에 맞춰 `re10k_moment`, `re10k`, `acid_moment`, `acid`를 사용한다.
가중치를 strict loading하고 모델 가중치를 추가 학습하지 않는다.
입력/출력 256×256, batch size 1, seed 111123, target pose alignment 100 steps,
PSNR·SSIM·LPIPS, 기존 crop/카메라 정규화를 사용한다.
`--no-align-pose`는 네 실험 모두에 적용된다.

네 GPU 프로세스를 모두 시작한 다음 완료까지 기다린다. 단일 GPU 프로세스의 ETA와
장면 완료 수를 터미널에 방향/모델 이름과 함께 출력한다.

```text
[Progress][re10k2acid/ours] evaluating | ... | elapsed=... | ETA=...
[Progress][re10k2acid/baseline] evaluating | ... | elapsed=... | ETA=...
[Progress][acid2re10k/ours] evaluating | ... | elapsed=... | ETA=...
[Progress][acid2re10k/baseline] evaluating | ... | elapsed=... | ETA=...
```

기본 결과 위치는 `outputs/cross_domain/re10k_acid/<실행시각>/`이다.

```text
<run>/
  re10k2acid/
    evaluation_index_acid.json
    ours/{config.yaml,console.log,progress.json,evaluated_scenes.json,coverage.json,metrics.json}
    baseline/{config.yaml,console.log,progress.json,evaluated_scenes.json,coverage.json,metrics.json}
    renders/{ours,baseline}/
    summary.json
    summary.csv
  acid2re10k/
    evaluation_index_re10k.json
    ours/{config.yaml,console.log,progress.json,evaluated_scenes.json,coverage.json,metrics.json}
    baseline/{config.yaml,console.log,progress.json,evaluated_scenes.json,coverage.json,metrics.json}
    renders/{ours,baseline}/
    summary.json
    summary.csv
  summary.json
  summary.csv
```

루트의 `summary.csv`에는 네 실험을 각각 한 행으로 저장한다.
장면 일치 여부는 같은 평가 데이터셋을 사용하는 두 모델 사이에서 확인한다.
실험이 실패하면 다른 실험의 완료까지 기다리고 오류로 종료하며,
완료된 실험의 개별 결과와 로그는 남긴다.

경로를 바꾸려면 다음 옵션을 사용한다.

```bash
bash scripts/eval_re10k_acid.sh --gpus 0 1 2 3 \
  --re10k-root /workspace/datasets/re10k \
  --acid-root /workspace/datasets/acid \
  --re10k-ours-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/Final++_re10k.ckpt' \
  --re10k-baseline-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/re10k.ckpt' \
  --acid-ours-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/Final++_acid.ckpt' \
  --acid-baseline-checkpoint '/workspace/research/GaussianGenerator/pretrained_weights/noposplat_acid.ckpt'
```

`--re10k-index`, `--acid-index`로 평가 뷰 인덱스를 지정할 수 있다.
데이터셋의 `test/index.json`은 chunk 매핑이므로 이 옵션에 넣지 않는다.
`--num-workers`는 GPU당 worker 수이며 기본 4개로 총 16개이다.
`--save-image`는 네 실험 모두의 예측 이미지와 비교 패널을 저장한다.
Python 직접 실행: `python -m scripts.eval_re10k_acid --gpus 0 1 2 3`.
