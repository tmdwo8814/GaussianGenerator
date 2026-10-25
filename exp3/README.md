# NoPoSplat / Ours scale–opacity 그림

사용자 후속 지시에 따라 **Gaussian 생성과 두 패널 그림**으로 범위를 줄였다. 제공된 original 60k / ours 140k checkpoint를 각각의 수렴 결과로 사용한다. 모델별 재현 점수 검증, PSNR/gap 분석, LBF 통계 검정은 실행하지 않는다. 기존 모델·학습 코드도 변경하지 않는다.

가로축은 요청한 **`log10(max_k s_k)`**, `s_k = sqrt(eig_k(final covariance))`다. Raw scale head 출력이나 `f/z`를 사용하지 않는다. 세로축은 rasterizer로 그대로 전달되는 최종 opacity다. 두 패널은 같은 축 범위와 log-count 색 범위를 공유한다. 그림의 scale 단위는 encoder의 canonical world 좌표이며 renderer의 near-plane 재조정 전이다.

## 서버 실행

기존 학습 환경을 활성화한 뒤 저장소 root에서 실행한다. Checkpoint와 dataset은 제공된 서버 경로가 기본값이다. 추가 패키지는 없으며 기존 `requirements.txt`의 numpy/matplotlib 및 모델 의존성을 사용한다.

```bash
# 먼저 작은 공통 subset으로 확인
NUM_SCENES=3 bash exp3/run_exp3.sh

# 기본: overlap별로 층화한 500개 공통 장면, seed 0
bash exp3/run_exp3.sh

# eval index 전체: non-null 5,601개 장면
NUM_SCENES=0 bash exp3/run_exp3.sh

# Slurm
mkdir -p logs
sbatch exp3/run_exp3.sh
```

두 decoder를 순서대로 GPU에 올리므로 GPU 하나로 실행한다. Checkpoint의 모든 `encoder.*` weight를 **strict=True**로 복원하고, `eval()` / `no_grad()` / `requires_grad_(False)`를 적용한다. MASt3R pretrained 초기값으로 대체하지 않는다. 최종 covariance와 opacity는 기존 encoder의 최종 반환 tensor에서 읽는다.

기본적으로 `opacity >= 1/255`인 active Gaussian을 사용한다. 전체 선택 장면의 active Gaussian에 무작위 priority를 부여하여 패널당 최대 60,000개를 균일 표본 추출한다. Scene별 동일 quota가 아니므로 장면마다 active 개수가 달라도 pooled 분포를 반영한다. 전체 개수와 장면별 median은 표본이 아닌 전체 active Gaussian에서 계산한다. 다른 장면이 dataset 내부에서 누락되면 `failures.txt`에 기록하고 그림 생성을 중단한다.

```bash
# 경로 및 표본 수 변경
ORIGINAL_CHECKPOINT=/path/original.ckpt \
OURS_CHECKPOINT=/path/ours.ckpt \
DATASET_ROOT=/path/re10k \
OUTPUT_DIR=exp3/output/my_run \
NUM_SCENES=100 SAMPLE_SIZE=60000 bash exp3/run_exp3.sh

# inactive slot도 표시하거나 Gaussian 전체를 따로 저장
bash exp3/run_exp3.sh --include-inactive --save-gaussians

# 추론 없이 저장된 표본으로 그림만 다시 생성
python -m exp3.plot_scale_opacity --plot-only --output-dir exp3/output/my_run

# checkpoint 생성 당시 추가 Hydra override가 있었을 경우
bash exp3/run_exp3.sh --ours-override model.encoder.moment_decoder.scale_max=3.0
```

`--override`는 두 모델에 공통 적용하고 `--original-override` / `--ours-override`는 해당 모델에만 적용한다. 각 옵션은 여러 번 쓸 수 있다. 동일 장면·256×256 해상도·실험별 head 설정은 실행 시 검증한다. 모든 slot을 포함할 때 scale이 정확히 0이면 log가 정의되지 않으므로 오류로 보고한다.

이번 그림은 camera-dependent 투영이나 렌더링을 사용하지 않아 pose alignment가 필요하지 않다. 원래 요청의 PSNR/gap 실험으로 확장할 때는 사용자가 선택한 target/context 각각의 pose alignment를 적용한다.

## 저장 파일

기본 출력 위치는 `exp3/output/<timestamp>/`다. 기존 결과와 섞이지 않도록 비어 있는 directory만 사용한다.

- `figs/figC_scale_opacity.png`, `.pdf`: NoPoSplat / Ours 두 패널, 가로 7 in, PDF fonttype 42.
- `samples/original.npz`, `ours.npz`: 그래프에 사용한 실제 표본. `log_s_max`, `opacity`, `scene_index`, `gaussian_index` 포함.
- `scenes.json`: 두 모델이 공유하는 고정 장면/view 목록. `scene_index`는 이 JSON의 key 순서다.
- `index_exclusions.json`: null index 및 subset 미선정 장면을 명시.
- `per_scene.csv`: 전체/active/선택 Gaussian 개수와 전체 선택 Gaussian의 median log scale.
- `metadata.json`, `config_{original,ours}.yaml`, `env.txt`: 정의·설정·seed·checkpoint step·패키지 버전·git commit.
- `failures.txt`, `status.txt`: 실패 정보와 `RUNNING` / `FAILED` / `COMPLETE` 상태.
- `dumps/<decoder>/<scene_index:06d>.npz`: `--save-gaussians` 선택 시 전체 slot의 최종 `means`, `cov`, `opacity`, float32. 이 파일만으로 RGB 재렌더링하는 기능은 범위에 포함하지 않는다.

## 로컬 검증

```bash
python -m unittest discover -s exp3/tests -v
```

테스트는 합성 covariance의 알려진 scale·회전/배율 불변 관계, pooled 표본 추출, 공통 장면 선택, strict checkpoint 복원 경계, 실제 PNG/PDF 출력을 검증한다. 서버 checkpoint를 이용한 GPU 추론은 서버에서 실행해야 한다. 테스트의 합성 값은 실제 분석 결과로 저장하지 않는다.
