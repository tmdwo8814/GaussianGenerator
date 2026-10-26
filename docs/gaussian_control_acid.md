# RE10K-trained Final++ → ACID Gaussian control

RE10K에서 학습한 `Final++_RE10k.ckpt`를 고정하고 ACID에서 uniform pruning, LightGaussian-style pruning, NanoGS-style merging, Redistribute를 평가한다. 추가 학습이나 ACID weight 로딩은 하지 않는다.

기존 `src/eval_gaussian_control.py`, `src/eval_lightgaussian_control.py`, 제어 알고리즘과 encoder 파일을 수정하지 않고 같은 entry point를 호출한다. 따라서 실행 중인 RE10K 실험의 source/cache signature를 바꾸지 않는다. ACID가 기존 코드에서 `DatasetRE10k`와 `dataset.re10k` 키를 사용하는 것은 로더 형식의 공통점 때문이며, 실제 root와 evaluation index는 ACID다. `gaussian_control_acid.yaml`은 RE10K 실험의 모델·weight·loss·target camera 정렬 설정을 상속한다.

## 실행

기존 conda 환경에서 프로젝트 루트로 이동한 후, 먼저 경로/index/디스크 공간을 확인한다. 이 점검은 GPU를 사용하거나 inference를 실행하지 않는다.

```bash
bash scripts/eval_gaussian_control_acid.sh --check-only
```

기본값은 다음과 같다.

- Weight: `/workspace/research/GaussianGenerator/pretrained_weights/Final++_RE10k.ckpt`
- ACID root: `/workspace/datasets/acid`
- Evaluation index: `assets/evaluation_index_acid.json`의 **1,341개** 장면
- 결과 root: `outputs/gaussian_control/re10k_to_acid`
- 유지 비율: `100, 75, 50, 25, 10, 5, 1%`

서버에서 먼저 두 장면을 모든 방법으로 실행한다. `GPU_ID`에는 현재 사용할 수 있는 GPU 번호를 지정한다.

```bash
GPU_ID=1 OUTPUT_ROOT=outputs/gaussian_control/re10k_to_acid_smoke bash scripts/eval_gaussian_control_acid.sh --max-scenes 2
```

전체 실행:

```bash
GPU_ID=1 bash scripts/eval_gaussian_control_acid.sh
```

Slurm에서는 GPU를 scheduler가 지정하므로 `GPU_ID` 없이 제출한다. 기존 ARIEL partition/node 형식을 따르며 GPU 1개, CPU 8개, RAM 60G를 요청한다.

```bash
sbatch scripts/eval_gaussian_control_acid.sh
```

경로는 `CHECKPOINT`, `DATASET_ROOT`, `EVALUATION_INDEX`, `OUTPUT_ROOT`, `CACHE_DIR`로 바꿀 수 있다. 일반 옵션은 마지막에 추가한다. 예를 들어 저장 공간이 큰 별도 디스크에 cache만 저장할 수 있다.

```bash
GPU_ID=1 CACHE_DIR=/path/to/large_disk/acid_control_cache bash scripts/eval_gaussian_control_acid.sh
```

## 처리 순서와 재개

1. **cache:** 장면당 frozen Final++ inference를 한 번 수행하여 Gaussian·support·feature·graph를 저장한다. full 100%로 target camera를 기존 RE10K와 동일하게 한 번 정렬하여 모든 조건에서 공유한다.
2. **controls:** 동일 cache에서 기존 Pruning·NanoGS·Redistribute 코드를 실행한다. full 100% NVS 성능은 한 번 계산하고 세 방법이 공유한다.
3. **lightgaussian:** ACID context intrinsics와 cached support로 PnP 카메라를 복원하고 기존 importance pruning을 실행한다. 모든 장면의 **controls 100% 결과를 그대로 재사용**하며, baseline이 없으면 실행을 중단한다. recovery·SH distillation·quantization은 하지 않는다.
4. **통합 보고서:** 네 방법의 모든 비율이 완료된 **같은 장면 집합**으로 합산한다. 100%는 `final_100.txt` 한 개로 정리한다.

같은 명령을 다시 실행하면 저장된 scene과 조건을 재사용한다. `all` 재개 시 checkpoint/config identity는 다시 검증하지만 완료 scene을 다시 추론하지 않는다. cache가 완료된 뒤 평가만 재개하려면 다음을 사용한다.

```bash
GPU_ID=1 bash scripts/eval_gaussian_control_acid.sh --stage evaluate
```

`--stage cache`, `--stage controls`, `--stage lightgaussian`로 각각 실행할 수도 있다. `controls`는 cache와 저장된 작은 decoder만 사용하므로 backbone checkpoint/dataset이 필요하지 않다. LightGaussian의 context camera 복원에는 ACID metadata가 필요하다. subset을 실행한 경우 재개 시에도 같은 `--max-scenes`를 지정한다. 설정/비율/시간 측정 조건이 바뀌면 새 output root를 사용한다.

전체 cache는 RE10K에서 관측한 장면당 약 210.3 MiB를 기준으로 **약 275 GiB**다. cache 단계 실행 전, 미저장 장면에 대한 추정치에 5% 여유와 2 GiB를 추가하여 free disk를 확인한다. 실제 크기는 달라질 수 있다. cache를 자동으로 지우지 않는다. NanoGS는 RE10K와 같은 CPU 구현이므로 이 단계가 오래 걸릴 수 있다.

## 결과와 진행 상황

기본 output root 아래 구조:

```text
cache/                 # inference 결과 및 공유 target cameras
controls/              # 기존 Pruning·NanoGS·Redistribute 조건별 txt, scene JSON, 누적 summary
lightgaussian/          # 기존 LightGaussian 조건별 txt, context cameras, CUDA build, scene JSON
logs/cache.log
logs/controls.log
logs/lightgaussian.log
experiment.json        # RE10K -> ACID, 경로, 비율, 시간 설정, 실행 명령
run_status.txt         # running / failed / complete 및 현재 phase
summary_status.json    # 네 방법 공통 완료 장면 수, 최종 완료 여부
summary.txt            # 동일 장면으로 집계한 통합 표
summary.json
final_100.txt          # 공유 기준값
prune_075.txt          # 이하 각 방법/각 축소 비율별 통합 보고서
lightgaussian_075.txt
merge_075.txt
redistribute_075.txt
...
```

phase별 `controls/summary.txt`와 `lightgaussian/summary.txt`는 각각 기존 갱신 주기를 따른다. root의 통합 보고서는 phase 종료/예외 시 갱신하며 네 방법의 모든 조건이 완료된 장면만 포함한다. LightGaussian이 시작되기 전에는 통합 완료 장면 수가 0이므로, 진행 중인 단계의 보고서를 확인한다.

전체 실행 중에도 통합 보고서만 갱신할 수 있다. GPU나 새 추론은 필요하지 않다.

```bash
bash scripts/eval_gaussian_control_acid.sh --stage report
```

기존 읽기 전용 진행률 도구도 ACID 경로를 지정하여 그대로 사용한다.

```bash
python scripts/gaussian_control_status.py --cache-dir outputs/gaussian_control/re10k_to_acid/cache --output-dir outputs/gaussian_control/re10k_to_acid/controls
```

LightGaussian 단계에서는 `--output-dir`을 `outputs/gaussian_control/re10k_to_acid/lightgaussian`으로 바꾼다. 이 도구의 ETA는 선택한 단계만의 추정치이며 후속 단계 시간은 포함하지 않는다.

## 동일한 평가 조건

모든 방법은 원본 `N`에 대해 `max(1,floor(N*p/100))`개를 만든다. 방법마다 Gaussian 선택·병합·재분배 방식은 RE10K 코드와 동일하다. LightGaussian은 두 context view만 사용하고, Redistribute는 cached support/feature/allocation graph와 frozen decoder를 사용한다. NVS는 동일한 cached/aligned target camera와 기존 PSNR·SSIM·LPIPS 함수를 사용한다.

각 방법의 postprocess 시간에는 최종 Gaussian 생성 연산을 포함한다. LightGaussian scoring renders와 Redistribute 재분배·decoding도 포함한다. inference/cache I/O/camera preparation/NVS rendering/metrics는 제외한다. warm-up 1회와 측정 1회가 기본이며 RE10K와 같다.

기존 실험과 CPU·디스크를 공유할 수 있어 시간 표기의 기본값은 `concurrent`다. 독립 실행으로 속도를 재측정할 때는 다른 output root에서 `--timing-mode isolated`를 지정한다. 이 옵션은 실행 환경을 자동으로 독점시키지 않는다.

검증: ACID와 RE10K Hydra 설정의 모델·checkpoint·loss·정렬 설정 일치, 기존 evaluator 호출 인자, stage 순서 및 실패 후 재개, 데이터/캐시 identity와 저장 공간 guard, 네 방법의 동일 장면 집계와 공유 100% 검증을 CPU에서 테스트한다. 실제 ACID/CUDA inference는 서버의 smoke 실행에서 확인해야 한다.
