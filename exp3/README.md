# NoPoSplat / Ours scale–opacity 분석

현재 학습 코드의 `re10k`와 **Final++ `re10k_moment` encoder 전체**로 Gaussian을 생성하고 두 패널의 분포 그림을 저장한다. 사용자가 확정한 범위대로 재현 점수·PSNR/gap·LBF 검정은 실행하지 않는다.

- 가로축: `log10(max_k s_k)`, `s_k = sqrt(eig_k(최종 covariance))`.
- 세로축: rasterizer wrapper로 그대로 전달되는 최종 opacity.
- Scale 단위: encoder의 canonical world 좌표. `f/z` 투영이나 renderer의 near-plane 재조정 전 값이다.
- 기본 선택: `opacity >= 1/255`인 active Gaussian.
- 장면: 공통 500개, overlap별 층화 선택, seed 0. `NUM_SCENES=0`은 non-null eval index 전체.
- 표시: 장면 통합 population에서 균일하게 최대 60,000개를 추출. 두 패널이 축 범위와 log-count 색 범위를 공유한다.

## 서버 실행

기존 `ragaussian` 환경을 활성화하고 저장소 root에서 실행한다. 제공된 checkpoint 두 개와 dataset 서버 경로가 기본값이다.

```bash
mkdir -p logs
sbatch exp3/run_exp3.sh

# 작은 공통 subset으로 실행 확인
NUM_SCENES=3 sbatch exp3/run_exp3.sh

# 전체 eval index
NUM_SCENES=0 sbatch exp3/run_exp3.sh

# Slurm을 사용하지 않을 때
bash exp3/run_exp3.sh
```

Original 60k / ours 140k를 각각의 수렴 checkpoint로 사용한다. 모델을 순서대로 GPU 한 개에 올린다. `eval()` / `no_grad()` / `requires_grad_(False)`를 적용하며 학습 코드와 checkpoint는 수정하지 않는다.

**Checkpoint의 모든 `encoder.*` weight를 `strict=True`로 복원한다.** Ours의 `appearance_2d`, `separate_appearance`, `appearance_capacity`, `appearance_moments`를 현재 설정대로 유지한다. SH MLP는 `[mean256, std256, context64] → 256 → 256 → SH75`이며, appearance key를 제외하거나 head를 교체하는 처리는 없다. 추론 시 compilation startup만 끄며 최종 covariance·opacity·SH는 기존 encoder가 반환한 값을 사용한다. MASt3R pretrained 초기값으로 checkpoint weight를 대체하지 않는다.

Overlap 층화는 small `[0,0.3]`, medium `(0.3,0.55]`, large `(0.55,1]`이다. 이전 오류를 일으켰던 `0.04031`도 유효하다. NaN/무한대나 `[0,1]` 밖의 값은 오류로 보고한다. Dataset이 선택 장면을 건너뛰면 누락을 `failures.txt`에 기록하고 그림 생성을 중단한다.

## 옵션과 재개

```bash
# 경로·장면 수·표본 수 설정
ORIGINAL_CHECKPOINT=/path/original.ckpt \
OURS_CHECKPOINT=/path/ours.ckpt \
DATASET_ROOT=/path/re10k \
OUTPUT_DIR=exp3/output/my_run \
NUM_SCENES=100 SAMPLE_SIZE=60000 bash exp3/run_exp3.sh

# 완료된 original 결과를 재사용하여 ours부터 재개
OUTPUT_DIR=exp3/output/my_run sbatch exp3/run_exp3.sh --resume

# 저장된 표본으로 추론 없이 그림만 재생성
python -m exp3.plot_scale_opacity --plot-only --output-dir exp3/output/my_run

# 전체 Gaussian 파라미터도 저장
bash exp3/run_exp3.sh --save-gaussians

# inactive slot도 분포에 포함
bash exp3/run_exp3.sh --include-inactive
```

`--resume`은 기존 output 폴더를 지정해야 한다. 선택 장면/view·index hash·seed·표본 수·checkpoint 경로 등 분석 설정을 확인한 뒤 완료된 decoder의 표본과 장면 기록을 재사용한다. 예전 버전에서 original만 완료한 결과도 사용할 수 있다. 미완료 decoder는 처음부터 다시 실행하고 이전 부분 CSV 행을 정리한다. 예전 geometry-only 경로에서 생성한 **ours** 표본은 재사용하지 않으므로 새 output 폴더를 사용해야 한다. 과거 실패 이력은 보존하며 최종 상태는 `status.txt`에서 확인한다.

Checkpoint 학습 당시 override가 있었다면 `--override KEY=VALUE`(공통), `--original-override KEY=VALUE`, `--ours-override KEY=VALUE`를 여러 번 전달할 수 있다. `--num-workers`와 `--device`도 직접 지정할 수 있다. 기본 해상도는 두 모델 모두 256×256이다.

## 출력

기본 폴더는 `exp3/output/<timestamp>/`다. 새 실행은 비어 있는 폴더만 사용한다.

- `figs/figC_scale_opacity.{png,pdf}`: 7 in 두 패널 그림, PDF fonttype 42.
- `samples/{original,ours}.npz`: `log_s_max`, `opacity`, `scene_index`, `gaussian_index`. 실제 추론 결과에서 추출한 표시용 표본이다.
- `scenes.json`: 두 모델이 공유하는 장면/view 목록. `scene_index`는 JSON key 순서다.
- `index_exclusions.json`: null index 및 subset 미선정 장면.
- `per_scene.csv`: 전체/active/선택 Gaussian 개수 및 전체 선택 Gaussian의 median log scale.
- `config_{original,ours}.yaml`, `checkpoint_load_{original,ours}.json`: 실제 설정 및 전체 encoder의 strict 복원 기록.
- `metadata.json`, `env.txt`: 정의·seed·checkpoint step·index hash·git commit·패키지 버전. 재개 환경은 `env_resume.txt`.
- `failures.txt`, `status.txt`: 실패 이력과 RUNNING/FAILED/COMPLETE 상태.
- `dumps/<decoder>/<scene_index:06d>.npz`: `--save-gaussians` 선택 시 최종 `means`, `cov`, `opacity`, `harmonics` 전체를 float32로 저장한다. 카메라/렌더링 기반 분석은 이번 범위에 포함하지 않는다.

## 검증

```bash
python -m unittest discover -s exp3/tests -v
```

현재 설정으로 실제 moment decoder를 구성하고, 전체 appearance/SH weight를 포함한 checkpoint fixture를 저장·strict 복원해 mean/covariance/opacity/SH가 일치하는지 검증한다. 누락되거나 크기가 다른 appearance weight는 오류로 처리하는지도 확인한다. 추가로 실제 eval index 선택, 고정 표본 추출, PNG/PDF 생성, original 완료 후 재개를 검사한다. 테스트의 합성 데이터는 실험 결과로 보존하지 않는다. 실제 제공 checkpoint를 이용한 GPU 추론은 서버에서 실행한다.
