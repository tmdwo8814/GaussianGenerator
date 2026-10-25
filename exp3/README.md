# NoPoSplat / Ours scale–opacity 분석

현재 학습 코드의 `re10k`와 **Final++ `re10k_moment` encoder 전체**로 Gaussian을 생성하고 분포 그림을 저장한다. `--lbf`를 추가하면 Large but Faint(LBF) 비율, 투영 면적 비율, 장면별 신뢰구간, 동일 source-pixel slot의 변화까지 계산한다. 재현 점수·PSNR/gap 평가와 학습은 실행하지 않는다.

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
# LBF 분석 및 기존 raw scale–opacity 그림
sbatch exp3/run_exp3.sh --lbf

# 기존 raw scale–opacity 그림만 생성
sbatch exp3/run_exp3.sh

# 작은 공통 subset으로 실행 확인
NUM_SCENES=3 sbatch exp3/run_exp3.sh --lbf

# 전체 eval index
NUM_SCENES=0 sbatch exp3/run_exp3.sh --lbf

# Slurm을 사용하지 않을 때
bash exp3/run_exp3.sh --lbf
```

Original 60k / ours 140k를 각각의 수렴 checkpoint로 사용한다. 모델을 순서대로 GPU 한 개에 올린다. `eval()` / `no_grad()` / `requires_grad_(False)`를 적용하며 학습 코드와 checkpoint는 수정하지 않는다.

**Checkpoint의 모든 `encoder.*` weight를 `strict=True`로 복원한다.** Ours의 `appearance_2d`, `separate_appearance`, `appearance_capacity`, `appearance_moments`를 현재 설정대로 유지한다. SH MLP는 `[mean256, std256, context64] → 256 → 256 → SH75`이며, appearance key를 제외하거나 head를 교체하는 처리는 없다. 추론 시 compilation startup만 끄며 최종 covariance·opacity·SH는 기존 encoder가 반환한 값을 사용한다. MASt3R pretrained 초기값으로 checkpoint weight를 대체하지 않는다.

Overlap 층화는 small `[0,0.3]`, medium `(0.3,0.55]`, large `(0.55,1]`이다. 이전 오류를 일으켰던 `0.04031`도 유효하다. NaN/무한대나 `[0,1]` 밖의 값은 오류로 보고한다. Dataset이 선택 장면을 건너뛰면 누락을 `failures.txt`에 기록하고 그림 생성을 중단한다.

## LBF 계산

`--lbf`의 주 지표는 아래와 같다. 그림 모양이나 결과가 유리한지에 따라 기준을 바꾸지 않는다.

1. `s_world = sqrt(lambda_max(final covariance))`.
2. `s_px = s_world * fx / z`. Gaussian의 source-pixel이 속한 **원래 context view**를 reference로 사용한다. `fx`는 저장소의 `get_fov`에서 유도한 가로 pixel focal, `z`는 해당 카메라 좌표에서의 평균 깊이다. Dataset의 canonical camera pose를 두 모델에 동일하게 사용하며 pose optimization은 하지 않는다. 따라서 투영 수치는 이 reference camera 정의를 전제로 한다.
3. Original의 전체 active Gaussian 중 reference 깊이가 양수인 population에서 `s_px`의 **P90을 정확히 계산**한다. Ours 추론 전에 `results/lbf_thresholds.json`에 고정하고 두 모델에 같은 값을 적용한다. World scale P90도 original의 전체 active population에서 별도로 계산한다.
4. LBF는 `size > original P90` 및 `opacity < 0.25`. Active는 `opacity >= 1/255`다. P90과 같은 값은 large로 세지 않으며, ties가 있으면 original의 large 비율이 10%보다 작을 수 있다.

Projected 분석에서 원래 카메라 깊이가 0 이하인 slot은 제외하고 `N_excl`, `N_excl_active`, `N_excl_all_behind`, `N_excl_origin_only`로 보고한다. 다른 context 카메라가 앞쪽이라고 reference를 바꾸지 않는다. World 분석은 모든 active slot을 포함하므로 projected 분석과 분모가 다를 수 있다.

전체 population에 대해 세 가지 비율을 계산한다.

- `lbf_count = N_LBF / N_analysis`: 전체 분석 대상에서 LBF가 차지하는 비율.
- `faint_given_large = N_LBF / N_large`: 큰 Gaussian 중 희미한 Gaussian의 비율. 큰 Gaussian 자체가 줄었는지 구분할 수 있다.
- `lbf_area = sum(A_LBF) / sum(A_analysis)`: projected 분석에서 면적으로 가중한 LBF 비율. `A = pi * sqrt(det(J R Sigma R^T J^T))`이고 `J`는 perspective projection의 Jacobian이다. **필터 적용 전 1-sigma 타원의 잠재 면적**이며, opacity를 곱하지 않는다. 가려짐, 타원 간 중첩, 화면 경계 clipping, rasterizer의 low-pass/slope clamp는 반영하지 않으므로 실제 보이는 pixel 점유율과 구분해야 한다.

주 추정치는 장면을 통합한 분자/분모의 비율이다. 두 모델에서 **동일한 장면을 함께 재추출하는 scene-cluster bootstrap 10,000회**로 각 비율과 `ours-original` 차이의 95% CI를 계산한다. 장면별 비율의 단순 평균과 paired Wilcoxon도 별도로 저장한다. 분모 0인 비율은 CSV 빈칸/JSON null로 남긴다. Bootstrap은 고정한 size threshold를 조건으로 한다.

민감도 분석은 original의 P80/P90/P95와 projected 4/8/16 px, opacity 0.05/0.10/0.25의 모든 조합을 저장한다. World 분석은 P80/P90/P95 조합이다. 전체 및 overlap별 결과, scale–opacity의 장면별 Spearman rho도 포함한다. Wilcoxon p는 다중검정 보정 전 값이며, 주 설정은 P90/0.25다.

Original의 projected LBF였던 source-pixel slot을 ours의 **같은 view/row/column slot**과 대조하여 `inactive`, `invalid_reference`, `smaller_only`, `more_opaque_only`, `smaller_and_more_opaque`, `remains_lbf`로 분류한다. 앞의 두 항목을 먼저 분리하고 나머지는 같은 P90/0.25 경계로 나눈다. 이는 출력 slot 비교이며 동일한 물리 Gaussian의 추적을 의미하지 않는다.

Hexbin·KDE contour·주변 histogram은 균일 표본으로 그리지만, 그림의 LBF 비율과 CI는 **전체 population**에서 가져온다. 두 패널의 축, bin, log-count normalization과 LBF 경계는 동일하다. KDE는 표시된 구간에서 정규화한 50/80/95% density contour이며 통계 계산에는 사용하지 않는다. 불리한 결과와 비감소하는 민감도 분석 항목도 보고서에 그대로 남긴다.

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

# LBF 실행 재개 (처음 실행 때의 --lbf도 유지)
OUTPUT_DIR=exp3/output/my_lbf_run sbatch exp3/run_exp3.sh --lbf --resume

# 추론 없이 재생성. LBF 실행 폴더이면 전체 통계와 그림도 재계산
python -m exp3.plot_scale_opacity --plot-only --output-dir exp3/output/my_run

# LBF bootstrap 횟수만 변경하여 재계산
python -m exp3.plot_scale_opacity --plot-only \
    --output-dir exp3/output/my_lbf_run --bootstrap-samples 10000

# 전체 Gaussian 파라미터도 저장
bash exp3/run_exp3.sh --save-gaussians

# inactive slot도 분포에 포함
bash exp3/run_exp3.sh --include-inactive
```

`--resume`은 기존 output 폴더를 지정해야 한다. 선택 장면/view·index hash·seed·표본 수·checkpoint 경로 등 분석 설정을 확인한 뒤 완료된 decoder의 표본과 장면 기록을 재사용한다. 예전 버전에서 original만 완료한 결과도 사용할 수 있다. 미완료 decoder는 처음부터 다시 실행하고 이전 부분 CSV 행을 정리한다. 예전 geometry-only 경로에서 생성한 **ours** 표본은 재사용하지 않으므로 새 output 폴더를 사용해야 한다. 과거 실패 이력은 보존하며 최종 상태는 `status.txt`에서 확인한다.

기존 raw-only 실행의 60,000개 표시 표본만으로는 전체 통계나 투영 크기를 복원할 수 없다. **LBF 분석은 새 output 폴더에서 `--lbf`로 한 번 재추론해야 한다.** 그 뒤에는 `lbf_cache`를 이용하므로 `--plot-only`에 checkpoint·dataset 접근이 필요 없다. LBF는 inactive 포함 옵션과 함께 사용할 수 없다. 장면별 전체 scalar cache를 저장하므로 500개 장면에서 수 GB의 추가 디스크가 필요할 수 있다. Exact percentile 계산은 임시 disk memmap을 사용하고 완료 후 삭제한다.

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
- `dumps/<decoder>/<scene_index:06d>.npz`: `--save-gaussians` 선택 시 최종 `means`, `cov`, `opacity`, `harmonics` 전체를 float32로 저장한다.

`--lbf` 추가 출력:

- `figs/figC_lbf_projected.{png,pdf}`: projected size–opacity hexbin + KDE + marginals, 공통 LBF box와 전체 비율/CI.
- `figs/figC_lbf_world.{png,pdf}`: 같은 형식의 world scale 그림.
- `figs/figC_lbf_transitions.{png,pdf}`: original LBF slot의 6가지 변화 비율.
- `results/lbf_report.{md,json}`: 주 설정의 count/area 비율, 차이의 CI와 paired p, 비감소하는 grid 결과.
- `results/lbf_thresholds.json`: original에서만 산출한 고정 threshold와 정의.
- `results/per_scene.csv`: 모든 threshold/opacity 조합의 전체 개수·면적·분포 요약 및 depth 제외 진단.
- `results/summary.csv`: 전체 및 overlap별 pooled 비율, 장면 평균, CI, paired p. `estimate`는 0–1 비율이며 차이도 같은 단위다. 보고서의 `%`/`pp` 표시는 여기에 100을 곱한다. Spearman rho는 상관계수다.
- `results/transitions{,_per_scene}.csv`: 변화별 전체 개수/비율/CI 및 장면별 개수.
- `results/table_robustness.tex`: projected count 비율의 모든 18개 조합.
- `lbf_cache/<decoder>/<scene_index:06d>.npz`: 전체 slot의 world/projected scale, opacity, area, reference depth/focal/view, 카메라·layout. `complete.json`으로 장면 일치 여부와 완결성을 확인한다.
- `lbf_samples/<decoder>.npz`: projected 그림용 균일 표본.
- `results/lbf_status.txt`: 통계/그림 생성 상태.

## 검증

### 저장된 수치 결과 검토

전체 scalar cache나 checkpoint 없이 `results/`의 CSV/JSON만으로 장면별 상관·개선율, 크기별 LBF tail, 기존 LBF 유지/신규 분류, 면적 합계의 장면 집중도를 검토할 수 있다.

```bash
python -m exp3.review_lbf_results --results-dir exp3/results \
    --output-dir exp3/output/lbf_review
```

출력은 `review.{md,json}` 및 네 종류의 PNG/PDF다. 입력 파일 hash와 CSV/보고서 population 일치를 확인하며 입력 통계·임계값은 수정하지 않는다. `fig_lbf_evidence`는 장면별 Spearman rho, 비율 분해, paired scene scatter를 보여준다. `fig_lbf_tail_robustness`는 크기 임계값에 따른 효과를, `fig_lbf_coupling_and_incidence`는 상관 분포와 원본/ours LBF slot의 중첩을 보여준다. `fig_lbf_area_diagnostics`는 면적 비율·장면 집중도·절대 잠재 면적을 함께 보여준다. 이 검토 스크립트의 해석 문구는 현재 500장면 결과를 검토하기 위한 것이며, 새 실험의 통계적 결론은 생성된 수치를 다시 확인해야 한다.

Raw Jacobian area 합계가 소수 장면에 집중되면 pooled area fraction이 크게 줄어도 실제 LBF 면적 감소의 증거가 되지 않는다. 분자/분모의 절대 합계, 장면별 평균/paired CI 및 depth/화면 밖 support를 함께 확인해야 한다. 화면에서 보이는 footprint를 평가하려면 renderer와 동일한 culling/projection과 화면 경계를 반영하는 별도 분석이 필요하다.

### 로컬 테스트

```bash
python -m unittest discover -s exp3/tests -v
```

현재 설정으로 실제 moment decoder를 구성하고, 전체 appearance/SH weight를 포함한 checkpoint fixture를 저장·strict 복원해 mean/covariance/opacity/SH가 일치하는지 검증한다. 누락되거나 크기가 다른 appearance weight는 오류로 처리하는지도 확인한다. 추가로 실제 eval index 선택, 고정 표본 추출, PNG/PDF 생성, original 완료 후 재개, 투영/면적 수식, 전역 scale 불변성, baseline-only exact percentile, 전체 통계와 표본의 분리, paired bootstrap, 슬롯 분류, LBF 보고서 생성을 검사한다. 테스트의 합성 데이터는 실험 결과로 보존하지 않는다. 실제 제공 checkpoint를 이용한 GPU 추론은 서버에서 실행한다.
