# 실험 3 — Step 1 코드 조사: NoPoSplat / CroCo

조사일: 2026-10-25. 코드 commit: `5e3b5511ed995f198458a3b195701829f74f17ea`.
범위: 소스·설정·eval index 정적 조사. 요청서의 Step 1 중단 지시에 따라 분석 코드 작성과 inference는 수행하지 않았다.

**후속 지시 반영:** 사용자가 제공한 서로 다른 step을 각각의 수렴 checkpoint로 사용하고 서버 경로를 유지하도록 승인했다. 이후 재현 검증을 생략하고 Gaussian의 `log10(max_k s_k)`–opacity 그래프만 생성하도록 범위를 축소했다. 해당 구현과 실행법은 [exp3/README.md](../README.md)에 있다. 아래 내용은 구현 전 조사 기록이며, decoder-only 학습 이력·PSNR 기준값은 이번 그림 실행의 선행 조건이 아니다. 이번 그림의 가로축에는 조사 당시 제안한 `f/z` 투영이나 camera alignment를 적용하지 않는다.

## 비교 대상과 확인 범위

| 항목 | 코드에서 확인한 값 |
|---|---|
| original | `+experiment=re10k`, `gs_params_head_type=dpt_gs` |
| ours | `+experiment=re10k_moment`, `gs_params_head_type=moment` |
| 공통 encoder / backbone | `noposplat` / `croco`, `ViTLarge_BaseDecoder`, `pose_free=true` |
| 해상도 / context 수 | 256×256 / 2 |
| original checkpoint — 사용자 제공 | `/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_noposplat/2026-10-12_16-46-59/checkpoints/epoch_0-step_60000.ckpt` |
| ours checkpoint — 사용자 제공 | `/nas2/data/tmdwo8814/research/GaussianGenerator/outputs/exp_final_plus/2026-10-16_18-35-06/checkpoints/epoch_2-step_140000.ckpt` |
| eval index | `assets/evaluation_index_re10k.json` |
| 전체 index를 사용할 때의 장면 수 | non-null 5,601개: small 1,403 / medium 2,568 / large 1,630 |
| index의 나머지 항목 | null 1,593개, 전체 key 7,194개. null은 평가 인덱스 부재로 명시적으로 기록할 대상 |
| index의 view 수 | 모든 non-null 항목이 context 2개, target 3개 |
| 후속 실험의 장면 선정안 | subset 대신 non-null index 전체 5,601개, seed 0. 실제 데이터 로딩 가능 여부는 미확인 |

설정 근거: [re10k.yaml](../../config/experiment/re10k.yaml), [re10k_moment.yaml](../../config/experiment/re10k_moment.yaml), [encoder 설정](../../config/model/encoder/noposplat.yaml), [dataset 설정](../../config/dataset/re10k.yaml). Overlap 집계에는 [get_overlap_tag](../../src/misc/utils.py#L38)의 구간을 사용했다.

현재 두 실험과 제공된 checkpoint가 지정하는 비교 대상은 NoPoSplat 한 종류다. [backbone registry](../../src/model/encoder/backbone/__init__.py)에는 `resnet`, `dino`, `croco`, `croco_multi`가 있으나, 등록만으로 각각의 original/ours 실험이 구현되어 있다는 뜻은 아니다. DINO/ResNet은 feature-map 반환 인터페이스이고 현재 NoPoSplat이 요구하는 CroCo의 `patch_embed`, token decoder, `return_views=True` 경로와 다르다. 다른 backbone의 비교 checkpoint·설정은 제공되지 않았다.

`noposplat_multi` / `croco_multi` 경로도 존재하지만 [EncoderNoPoSplatMulti.forward](../../src/model/encoder/encoder_noposplat_multi.py#L147)는 `dpt_gs`만 처리한다. 현재 ours의 [2-view 검사](../../src/model/encoder/encoder_noposplat.py#L158) 때문에 6/12/24-view 확장은 그대로 실행할 수 없다. 다른 backbone이나 multiview 결과를 확인한 것으로 표시하지 않는다.

## 1. 최종 Gaussian 생성 위치

- 공통 반환 타입은 [src/model/types.py::Gaussians](../../src/model/types.py#L8)의 `means`, `covariances`, `harmonics`, `opacities`다. 분석은 raw head 출력 대신 이 최종 tensor를 대상으로 해야 한다.
- **Original:** [EncoderNoPoSplat.forward](../../src/model/encoder/encoder_noposplat.py#L220)에서 `res1['pts3d']`, `res2['pts3d']`를 쌓아 `pts_all`을 만들고, `GS_res1`, `GS_res2`를 opacity / scale / quaternion / SH로 분리한다. `pose_free=true`이므로 [UnifiedGaussianAdapter.forward](../../src/model/encoder/common/gaussian_adapter.py#L122)를 호출한다. 최종 `Gaussians`는 encoder의 274–290행에서 펼쳐 반환한다.
- **Ours:** [EncoderNoPoSplat.forward](../../src/model/encoder/encoder_noposplat.py#L190)에서 두 point map을 `points (B,2HW,3)`, 두 DPT256 feature map을 `features (B,2HW,256)`으로 연결한다. `self.gaussian_decoder(points, features, ...)`가 최종 Gaussian을 반환한다. [MomentGaussianDecoder.forward](../../src/model/encoder/heads/moment_gaussian_decoder.py#L391) → `aggregate_moments` → [build_gaussians](../../src/model/encoder/heads/moment_gaussian_decoder.py#L340) 경로다. Baseline adapter를 다시 적용하지 않는다.
- 두 point map은 공통으로 DPT point head를 거친다. [reg_dense_depth](../../src/model/encoder/heads/postprocess.py#L22)의 `exp` 모드는 raw 3-vector의 길이 `d`에 대해 `xyz / max(d,1e-8) * expm1(d)`다. 이것은 scale activation과 별개이며, z 양수를 강제하지 않는다.

## 2. [CHECK] 최종 activation과 rasterizer 입력

### Original

- scale: **`min(0.001 * softplus(raw_scale), 0.3)`**. 현재 경로는 `UnifiedGaussianAdapter`이므로 다른 `GaussianAdapter`에 있는 sigmoid `[0.5,15]` 및 depth/focal multiplier를 적용하지 않는다.
- quaternion: `q / (||q|| + 1e-8)`, 성분 순서는 **xyzw**다.
- covariance: [build_covariance](../../src/model/encoder/common/gaussians.py#L33)의 `R(q) diag(s²) R(q)ᵀ`. Quaternion-to-matrix에도 epsilon이 있어 raw scale을 고유값 대신 사용하는 것은 피한다. Pose-free 경로에서는 별도의 context-camera rotation을 곱하지 않는다.
- opacity: `densities = sigmoid(raw_opacity)` 후 `map_pdf_to_opacity`. 현재 `initial=final=0`이라 exponent가 1이고 mapping은 항등이다. 최종 opacity는 sigmoid 값이다. [encoder 231–239행](../../src/model/encoder/encoder_noposplat.py#L231).

### Ours

- budget: `b_j = 2 * sigmoid(budget_head(f_j))`; allocation은 outgoing 이웃 방향 softmax, `q_ji = b_j * a_ji`다. `budget_max=2`는 incoming mass `m_i`의 상한이 아니다.
- opacity: **`-expm1(-mass)` = `1-exp(-m_i)`**, 추가 sigmoid나 clamp 없이 반환한다.
- `coverage`가 요청서의 **γ_i**다. **`γ_i = 1 + 2 * sigmoid(scale_head(geometry_pooled_i))`**로 제한된다. 수학적으로 `1 < γ_i < 3`; 부동소수점 saturation으로 경계값도 가능하다. 설정상 초기값은 1.75다. 자유로운 scale 변수가 아니다.
- 직접 quaternion이나 축별 scale을 예측하지 않는다. 정규화 좌표에서 얻은 대칭 central covariance `C_i`에 대해 최종 값은 다음과 같다.

  `Σ_i = r² [γ_i² C_i + (1e-8 + 8 ε_fp32 trace(γ_i² C_i)) I]`

  여기서 `r`은 scene scale이고 trace 항은 detach된 roundoff guard다. 이 보정까지 포함한 **반환된 최종 covariance**에서 고유값을 구해야 한다. [build_gaussians 354–388행](../../src/model/encoder/heads/moment_gaussian_decoder.py#L354).

### Rasterization 직전의 추가 변환

[DecoderSplattingCUDA.forward](../../src/model/decoder/decoder_splatting_cuda.py#L37)는 최종 Gaussian을 view별로 반복하여 [render_cuda](../../src/model/decoder/cuda_splatting.py#L46)에 전달한다. 두 실험 모두 `make_scale_invariant=true`를 상속한다.

- 각 렌더링 view에서 `t=1/near`: `means → t·means`, `covariances → t²·covariances`, C2W translation도 `t`배, near/far도 같은 비율로 변경한다.
- CUDA rasterizer는 `means3D=t·means`, `cov3D_precomp=t²·Σ`의 상삼각 **(00,01,02,11,12,22)**, 그리고 **변경 없는 최종 opacity**를 받는다. `scale_modifier=1.0`이며 raw scale/rotation을 전달하지 않는다. [render_cuda 64–72, 101–128행](../../src/model/decoder/cuda_splatting.py#L64).
- 분석용 covariance와 depth는 **같은 좌표계**를 사용해야 한다. view별 near rescale 전의 최종 `Gaussians`와 C2W camera를 함께 저장하면 `s_max·f/z`는 이 추가 전역 rescale에 불변이다. 저장 위치를 명시하고 round-trip은 동일한 renderer 설정으로 수행해야 한다.
- 외부 `diff_gaussian_rasterization`의 CUDA 구현은 이 checkout에 없다. Python 경계의 실제 입력까지는 확인했으나, 해당 설치본의 내부 alpha cutoff / EWA filter / compositing 규칙은 직접 확인하지 못했다. 요청서의 `1/255`는 분석 정의로 유지하되 설치본 검증과 혼동하지 않는다.

## 3. [CHECK] Pixel origin 복원

현재 설정은 **두 decoder 모두 pixel-indexed**이며 한 pixel당 slot 하나다. `V=2`, `H=W=256`에서 장면당 총 `N=131,072`개를 출력한다. Flatten 순서는 view-major, 그 안에서 row-major다.

slot index `i`에 대해 `origin_view = floor(i/(H·W))`, `p=i mod (H·W)`, `origin_pix=(x=p mod W, y=floor(p/W))`로 복원한다. 실제 frame 번호는 `context['index'][origin_view]`로 별도 복원한다. Pixel 좌표는 crop/resize 후 256×256 좌표다. [original flatten](../../src/model/encoder/encoder_noposplat.py#L274), [ours concat](../../src/model/encoder/encoder_noposplat.py#L194), [slot 순서 보존 계약](../../src/model/encoder/heads/moment_gaussian_decoder.py#L395).

여기서 pixel origin은 **slot의 출처**다. Original의 pose-free mean도 source ray 위에 강제되지 않으며 ours의 mean은 이웃 support의 가중 중심으로 이동한다. Ours는 양쪽 view를 아우르는 3D kNN을 사용해 여러 source pixel의 기여를 받지만 destination slot의 origin label은 유지된다. 최종 mean을 재투영해서 origin을 다시 정하지 않는다.

요청서에 따라 reference camera는 이 origin context view다. 모든 context에서 z≤0인 경우의 제외와 별개로, origin의 z≤0이고 다른 context의 z>0인 경우는 원문 정의에 처리 규칙이 없다. 이 경우를 조용히 다른 camera로 교체해서는 안 되며 후속 구현 전에 별도 처리 정책을 명시해야 한다.

## 4. [CHECK] Intrinsics와 pixel focal

[DatasetRE10k.convert_poses](../../src/dataset/dataset_re10k.py#L224)는 **정규화된 K**를 만들고 W2C를 inverse하여 C2W를 반환한다. [crop_shim](../../src/dataset/shims/crop_shim.py#L31)은 crop 전후 비율로 `K00`, `K11`을 수정한다. 따라서 원본 카메라 데이터가 아니라 **renderer에 전달되는 crop 후 K**를 사용해야 한다.

- 일반적인 중심 principal point에서 pixel focal은 `fx_px = K00·W`, `fy_px = K11·H`다. 이 실험의 W,H는 256이다.
- 단, 현재 renderer는 K를 직접 CUDA로 보내지 않고 [get_fov](../../src/geometry/projection.py#L247)로 FOV를 계산한 뒤 symmetric projection을 만든다. 실제 rasterizer에 대응하는 focal은 **`fx_render = W/(2 tan(FOV_x/2))`, `fy_render = H/(2 tan(FOV_y/2))`**다. Principal point가 중심이 아니면 `K00·W`, `K11·H`와 달라질 수 있다. 원본 intrinsics를 조사하지 못했으므로 중심이라고 가정하지 않는다.
- 요청서의 scalar `f`가 x/y 중 어느 축인지 정하지 않았으므로 후속 실행 전에 고정해야 한다. 수치 결과를 보기 전의 제안은 `f=fx_render`로 통일하고 두 축 focal을 모두 metadata에 보존하는 것이다. 아직 분석을 실행하거나 이 선택을 결과에 적용하지 않았다.
- Depth는 `inverse(C2W)·[mean,1]`의 z 성분이다. Camera 간 거리나 Euclidean range로 대체하지 않는다.

## 5. [CHECK] World / scene scale

- [DatasetRE10k.__iter__](../../src/dataset/dataset_re10k.py#L173): `make_baseline_1=true`이면 첫/마지막 context camera translation 사이의 거리로 모든 camera translation과 near/far를 나눈다. 두 context의 baseline은 1이 된다.
- `relative_pose=true`이며 [camera_normalization](../../src/misc/cam_utils.py#L27)은 첫 context C2W의 inverse를 왼쪽에서 곱한다. 첫 context camera가 identity인 canonical frame이다. CroCo도 [두 point map을 view1 frame으로 출력](../../src/model/encoder/backbone/backbone_croco.py#L61)하도록 정의된다. 예측점이 실제로 dataset camera와 완벽히 정합한다는 뜻은 아니다.
- Ours 내부에서 `r=max(median_j ||x_j||,1e-6)`를 detach하여 사용한다. 정규화 support `x/r`에서 moment를 계산하고 최종 mean은 r배, covariance는 r²배로 복원한다. **장면 중심을 빼는 normalization은 없다.** [forward 424–438행](../../src/model/encoder/heads/moment_gaussian_decoder.py#L424).
- 마지막으로 renderer가 view별 `1/near` rescale을 적용한다. Dataset baseline normalization, moment 내부 normalization, renderer normalization은 서로 다른 단계다.

## 6. [CHECK] m_i, |J_i|, γ_i와 빈 slot

현재 ours 설정에서는 `encoder.eval()`과 `torch.no_grad()`, batch size 1에서 `encoder(..., analysis_cache=cache)`로 조사 정보를 받을 수 있다. [cache 제약](../../src/model/encoder/heads/moment_gaussian_decoder.py#L409), [cache 생성 위치](../../src/model/encoder/heads/moment_gaussian_decoder.py#L453). 이 cache는 `separate_appearance` 분기에서 채워지며 현재 ours는 해당 설정이 true다. 모든 moment 변형에서 같은 cache가 생긴다고 가정해서는 안 된다.

| 분석량 | 얻는 위치 / 의미 |
|---|---|
| `m_i` | `cache['mass']`; `aggregate_moments`에서 `q`를 destination index로 `index_add`한 **raw incoming mass** |
| `|J_i|` | `cache['neighbors']`의 destination별 등장 횟수. Shape `(N,min(16,N))`의 각 row는 **source j**다. Self 포함, incoming degree는 16으로 제한되지 않는다. q>0인 edge 수로 재정의하지 않는다 |
| `γ_i` | `build_gaussians`의 local `coverage`. 최종 반환 타입과 cache에는 직접 저장되지 않는다. 해당 실행의 `scale_head` 출력을 hook하여 sigmoid/range 변환하거나, cache의 `points/features/neighbors/q`로 같은 geometry pooling을 재구성하여 얻을 수 있다 |
| `geometry_pooled_i` | `aggregate_moments(...)[3]`. Appearance가 별도 weight를 사용해도 gamma는 **geometry pooled feature**를 사용한다 |

[aggregate_moments 236–287행](../../src/model/encoder/heads/moment_gaussian_decoder.py#L236)의 실제 epsilon 처리는 단순한 `mass.clamp_min`이 아니다. `ε=1e-6`인 self prior를 넣어 edge weight를 다음처럼 만든다.

`w_ji = (q_ji + ε·1[j=i]) / (m_i + ε)`

이 weight로 mean, pooled feature, central covariance를 계산한다. Mean은 원래 support에 대한 weighted offset으로 누적하며 covariance는 마지막에 대칭화한다. **ε prior는 opacity용 mass에는 더하지 않는다.** 따라서 구현의 moment는 요청서의 이상적 `q/m` 식에 epsilon 안정화가 추가된 형태다.

- m=0이면 mean은 자기 support, covariance는 central moment 0에 최종 diagonal floor를 더한 값, opacity는 0이다. Slot은 출력에서 삭제되지 않는다.
- m≈0에서도 분모 `m+ε`와 self prior로 안정화된다. 분석은 요청서대로 최종 `α≥1/255`만 사용해야 하며 빈 slot을 faint primitive로 포함하지 않는다.
- 현재 설정의 gamma 범위는 **1–3으로 제한됨**이 최종 답이다. Checkpoint 당시 override까지 동일했는지는 checkpoint metadata 또는 당시 `.hydra/config.yaml`로 별도 확인해야 한다.

## 후속 실행 전에 남은 확인 사항

1. **Decoder-only 비교의 학습 이력이 확인되지 않았다.** 현재 [re10k 설정](../../config/experiment/re10k.yaml)은 `backbone_lr_multiplier=0.1`이며 [configure_optimizers](../../src/model/model_wrapper.py#L788)는 학습 가능한 backbone/point-head parameter를 optimizer에 포함한다. [CroCo 초기화](../../src/model/encoder/backbone/backbone_croco.py#L90)의 freeze 호출은 주석이다. 제공된 이름의 step도 original 60k / ours 140k로 다르다. 현재 기본 설정은 backbone을 학습하는 경로이며, 이 두 파일을 동일 backbone 동결·동일 schedule의 Tab. 2 pair라고 입증할 근거는 없다. 당시 config, 공통 frozen backbone의 일치 여부, 동일 schedule의 checkpoint pair가 필요하다. Inference 시 freeze하는 것만으로 decoder-only 학습 이력이 성립하지는 않는다.
2. **Checkpoint와 데이터에 현재 접근할 수 없다.** Windows workspace에서 사용자 제공 `/nas2/...` 두 경로와 `/local_datasets/3dgs/re10k/re10k`가 존재하지 않으며 local checkpoint / 당시 Hydra config도 찾지 못했다. Checkpoint의 실제 weight·override·학습 provenance는 검사하지 않았다. Tab. 1/2 기준 PSNR/SSIM/LPIPS 수치도 제공되지 않아 ±0.1 dB 재현 판정을 아직 할 수 없다.
3. **기본 평가는 camera pose 최적화를 포함한다.** [config/main.yaml](../../config/main.yaml)의 `test.align_pose=true`, `pose_align_steps=100`; [test_step_align](../../src/model/model_wrapper.py#L388)은 고정 Gaussian과 target GT를 사용해 pose delta만 Adam으로 최적화한다. 이는 모델 재학습과 다르지만 요청서의 inference-only가 pose 최적화도 금지하는 의미라면 기존 점수 재현과 충돌한다. Target/context에 적용할 camera protocol을 고정해야 gap 해석이 가능하다. 현재는 context camera를 정합하는 동일 경로가 없으며 `get_pnp_pose`는 helper로만 존재한다. `test_step_align`의 `camera_dump`로 최종 target C2W를 보존할 수 있다.
4. **요청서의 공통 dump 필드만으로 RGB round-trip은 불가능하다.** 최종 Gaussian의 `harmonics`도 필요하며 camera C2W/K, near/far, background, image shape, scale-invariant 설정과 정합된 pose를 함께 보존해야 한다. Baseline/ours 모두 SH degree 4, `(N,3,25)`를 사용한다. 작은 covariance가 float16 저장으로 손상될 수 있으므로 실제 round-trip 오차로 저장 dtype을 결정해야 한다.
5. **Dataset은 일부 장면을 자체적으로 건너뛴다.** [DatasetRE10k.__iter__](../../src/dataset/dataset_re10k.py#L119)의 index/FOV/image/baseline 검사에서 `continue`한다. 후속 실행에서는 index manifest와 완료 장면을 대조하여 누락을 `failures.txt`에 남겨야 한다. non-null 5,601개는 성공 장면 수가 아니라 예정 목록이다.
6. **`make_plots.py::fig_c`는 이 checkout에서 찾지 못했다.** 기존 [analysis_scale_opacity.py](../../src/scripts/analysis_scale_opacity.py)는 covariance/opacity perturbation 실험이며 이번 LBF 분포 비교와 다른 프로토콜이다. 해당 파일을 이번 실험의 구현이나 결과로 간주하지 않았다.

이 문서에는 실험 결과, 재현 점수, round-trip 오차가 없다. 요청서의 중단 지점까지 조사만 완료했으며, 분석 스크립트·dump·CSV·그림·표는 생성하지 않았다.
