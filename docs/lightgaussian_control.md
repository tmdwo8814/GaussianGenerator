# LightGaussian pruning-only control

추가 학습 없이 기존 Final++ cache를 읽어 LightGaussian의 significance 기반 pruning만 평가한다. 표에서는 **LightGaussian (pruning only)**로 표기한다. Full LightGaussian의 recovery, SH distillation, quantization은 포함하지 않는다.

## 동시 실행과 자원

2026-10-26에 제공된 서버 출력은 CPU 96개, 기존 프로세스 약 1.44 CPU 코어 사용, RAM `MemAvailable` 약 369 GiB, 디스크 433 GiB 여유를 보였다. GPU 0은 기존 실험이 사용하고 GPU 1·2는 비어 있었다. `free` 4.4 GiB와 cgroup `memory.current` 약 369 GiB에는 회수 가능한 파일 캐시가 반영되므로 RAM 부족으로 해석하지 않는다.

새 실행은 GPU 1을 명시적으로 선택하고 CPU/BLAS/OpenCV/PyTorch 스레드를 1개로 제한한다. CUDA 빌드도 `MAX_JOBS=1`이며, 가능한 경우 `nice -n 10`, `ionice -c 3`로 실행한다. 기존 cache는 읽기만 하고 기존 `final_plus_results`에는 쓰지 않는다. 새 Gaussian 파일이나 1 TiB 이상인 원본 cache의 복사본을 만들지 않는다. 카메라와 결과만 별도 폴더에 저장한다.

별도 GPU여도 CPU·메모리 대역폭·디스크는 공유하므로 **기존/추가 실험 모두 처리 시간에 영향을 받을 수 있다**. 기본 `timing_mode=concurrent`를 metadata와 txt에 남긴다. 논문용 최종 속도는 다른 실험이 끝난 뒤 별도 출력 폴더에서 `--timing-mode isolated`로 측정한다. 이 옵션은 경합 상태를 자동으로 보장하는 기능이 아니라 기록용이다.

## 실행

기존 conda 환경에서 프로젝트 루트로 이동한다. 아래 명령은 각각 한 줄이다.

```bash
GPU_ID=1 bash scripts/eval_lightgaussian_control.sh --check-only
```

`--check-only`는 cache 완결성, 데이터셋 메타데이터, 기존 렌더러/평가 라이브러리, CUDA/nvcc/ninja, 가용 RAM·VRAM·디스크를 확인한다. 모델 추론이나 새 CUDA 빌드는 하지 않는다. 현재 실행 중인 환경의 `diff_gaussian_rasterization`은 교체하지 않는다.

먼저 두 장면으로 CUDA 빌드·카메라 복원·렌더링을 확인한다.

```bash
GPU_ID=1 LIGHTGAUSSIAN_OUTPUT_DIR="$PWD/outputs/gaussian_control/lightgaussian_smoke" bash scripts/eval_lightgaussian_control.sh --max-scenes 2
```

전체 실행:

```bash
GPU_ID=1 bash scripts/eval_lightgaussian_control.sh
```

기본 cache는 `outputs/gaussian_control/final_plus_cache`, 결과는 `outputs/gaussian_control/final_plus_lightgaussian_results`, RE10K root는 `/workspace/datasets/re10k`다. 각각 `CACHE_DIR`, `LIGHTGAUSSIAN_OUTPUT_DIR`, `DATASET_ROOT`로 변경할 수 있다. 중단 후 같은 명령을 실행하면 완료 조건을 건너뛴다. smoke와 전체 실행은 선택 장면 수가 달라 서로 다른 결과 폴더를 사용해야 한다. 스크립트는 서버로 파일을 동기화한 뒤 실행한다.

Slurm에서는 GPU를 scheduler가 배정하므로 `GPU_ID`를 지정하지 않는다. 기존 `train_re10k.sh`의 ARIEL partition/node/GPU 형식을 따른다.

```bash
sbatch scripts/eval_lightgaussian_control.sh
```

동시 실행이 끝난 뒤 속도 재측정:

```bash
GPU_ID=1 LIGHTGAUSSIAN_OUTPUT_DIR="$PWD/outputs/gaussian_control/lightgaussian_isolated" bash scripts/eval_lightgaussian_control.sh --timing-mode isolated
```

## 평가와 시간 범위

- 남길 비율은 **100, 75, 50, 25, 10, 5, 1%**, 개수는 기존과 같은 `max(1, floor(N*p/100))`이다. 경계 동점도 정확한 개수를 유지한다.
- 선택에는 **두 context view만** 사용한다. target RGB나 target camera는 significance 입력에 들어가지 않는다.
- 기존 cache에 context 카메라가 없어서, RE10K chunk에서 context intrinsics와 baseline 메타데이터만 읽는다. 저장된 context 3D support map과 픽셀 대응으로 PnP를 수행하여 cached Gaussian 좌표계의 카메라를 복원한다. context 이미지는 decode하지 않으며 encoder를 호출하지 않는다. 카메라는 별도 `context_cameras/`에 저장하고 재개 시 재사용한다. 복원 실패 시 해당 장면을 몰래 제외하지 않고 중단한다.
- 공식 released implementation에 맞춰, ray에 기여하는 Gaussian의 `opacity * T_before_splat`를 합산하고 `(volume / V_ref)^0.1`을 곱한다. `V_ref`는 내림차순 volume의 `int(0.9*N)` 원소이며 release code처럼 clipping하지 않는다. 픽셀 alpha는 기여 유효성 및 transmittance 갱신에 사용한다. 단순 visibility count나 opacity-only 근사 점수는 사용하지 않는다.
- 정확한 예산만큼 점수 상위 Gaussian을 남긴다. 위치·공분산·opacity·SH degree 4는 그대로 유지한다.
- **각 비율의 postprocess 시간에 context significance 렌더링, volume 계산, ranking, gathering을 모두 포함**한다. 입력은 GPU에 올라온 Gaussian과 준비된 카메라이며, CUDA synchronize 전후 wall time을 측정한다. 비율 간 significance를 재사용해서 scoring 시간을 누락하지 않는다. CUDA 빌드·첫 warm-up·I/O·카메라 준비·NVS 렌더링·지표 계산은 제외한다. 카메라 준비 시간은 `camera_prepare_ms`로 별도 기록한다.
- NVS는 기존 cached target cameras를 고정하고 기존 PSNR/SSIM/LPIPS 함수를 그대로 사용한다. 장면별 target 평균 후 장면 평균을 계산한다.
- 100%는 identity이고 postprocess 0 ms다. 기존 `final_plus_results/scenes/*.json`에 같은 cache signature의 100% 결과가 있으면 재사용한다. 없을 때만 cached Gaussian을 다시 렌더링하며 모델 inference는 수행하지 않는다. 재사용한 100% render timing은 원래 실행의 값이다. 다른 폴더를 지정하려면 `--baseline-results /path/to/results`를 사용한다.

기존 방법들은 사전 계산된 mass 또는 입력 grid만으로 선택할 수 있지만 LightGaussian은 준비된 카메라와 scoring render가 필요하다. 이 입력/시간 범위 차이를 함께 명시한다. 논문 본문 예시: “LightGaussian [1] is evaluated in its pruning-only form using the two context views, without recovery.”

## 결과와 진행 확인

새 결과 폴더에 아래 파일을 저장한다.

- `lightgaussian_100.txt`, `lightgaussian_075.txt`, …, `lightgaussian_001.txt`: 조건별 PSNR/SSIM/LPIPS, Gaussian 수, postprocess 평균/p50/p95, view당 렌더링 시간.
- `summary.txt`, `summary.json`, `per_scene.jsonl`: 전체/overlap별 누적 결과. **첫 장면, 이후 25장면마다, 종료/예외 시** 갱신한다. `--report-every 1`로 매 장면 갱신할 수 있다.
- `scenes/*.json`: 조건이 끝날 때마다 atomic 저장하는 재개 기준. 진행 중인 현재 장면의 완료 조건도 보존한다.
- `context_cameras/*.pt`: 복원된 context cameras와 PnP reprojection 통계.
- `metadata.json`, `run_status.txt`: 코드/캐시 identity, 설정, 하드웨어·우선순위·동시실행 여부, 상태.
- `cuda_build/`: 전용 이름으로 JIT 빌드한 CUDA extension. 기존 렌더러에 설치하거나 덮어쓰지 않는다. 최초 빌드는 시간이 필요하다.

전체 실행의 진행률/ETA는 기존 읽기 전용 도구로 확인한다.

```bash
python scripts/gaussian_control_status.py --output-dir outputs/gaussian_control/final_plus_lightgaussian_results
```

이 status 도구는 원본 manifest 전체를 기준으로 하므로 `--max-scenes 2` smoke의 전체 완료율 계산에는 사용하지 않는다.

## 구현 출처와 검증

[1] [LightGaussian official implementation](https://github.com/VITA-Group/LightGaussian). 중요도 count renderer는 공식 submodule [compress-diff-gaussian-rasterization](https://github.com/Kevin-2017/compress-diff-gaussian-rasterization/tree/9893d8c607684bc1a432e179cc97035d72d84520)의 commit `9893d8c607684bc1a432e179cc97035d72d84520`을 사용한다. GLM도 submodule commit으로 고정했고 provenance는 `src/third_party/lightgaussian_rasterizer/UPSTREAM.json`에 기록했다. upstream 라이선스를 함께 보존한다.

원본 count kernel의 병렬 hit/importance 누적 두 줄만 `atomicAdd`로 수정하여 합산 시 race를 제거했다. 부동소수 atomic 합산 순서로 작은 수치 차이는 발생할 수 있다. 이식 중 별도 RGB를 0으로 전달하는 것은 significance가 RGB에 의존하지 않기 때문이며, 기존 degree-4 SH는 평가 렌더링에 그대로 사용된다.

CPU 테스트는 volume weighting, 정확한 수량/동점 처리와 속성 보존, 두 context의 importance 합산, 합성 카메라의 PnP 복원, crop/baseline, 기존 100% 재사용, 중단/재개 및 원본 파일 보존을 확인한다.

```bash
python -m unittest discover -s tests -p test_lightgaussian_control.py -v
```

실제 native kernel의 1-pixel/2-Gaussian 결과를 `.4`, `.2*(1-.4)=.12`의 직접 계산과 비교하는 GPU 테스트도 제공한다.

```bash
CUDA_VISIBLE_DEVICES=1 MAX_JOBS=1 LIGHTGAUSSIAN_CUDA_TEST=1 python -m unittest discover -s tests -p test_lightgaussian_control.py -v
```

Windows 편집 환경에는 CUDA GPU/toolkit과 실제 cache가 없어 native build 및 실제 RE10K 결과는 여기에서 검증할 수 없다. 서버의 두 장면 smoke가 그 검증 단계다.
