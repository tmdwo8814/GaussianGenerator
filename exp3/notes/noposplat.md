# 현재 버전 조사 메모

코드 기준 commit: `ba4c6314696b59d013e7275a22ca9d18b599287b`.

현재 실험은 사용자 후속 지시에 따른 Gaussian 분포 그림이다. Original은 `re10k`, ours는 Final++ `re10k_moment`를 사용하며 제공된 서로 다른 수렴 step을 그대로 사용한다.

- 최종 출력: `src/model/encoder/encoder_noposplat.py::EncoderNoPoSplat.forward`가 반환하는 `Gaussians.means/covariances/opacities/harmonics`.
- Original: `UnifiedGaussianAdapter`에서 scale=`min(0.001*softplus(raw),0.3)`, quaternion xyzw 정규화, `R diag(s²) Rᵀ`로 covariance 생성. 현재 opacity mapping은 sigmoid opacity에 대한 항등 mapping이다.
- Ours: `MomentGaussianDecoder.build_gaussians`의 최종 covariance와 `-expm1(-mass)` opacity를 사용한다. Coverage γ는 `1+2*sigmoid(scale_head(pooled))`로 1–3 범위이며 covariance에는 floor와 FP32 roundoff guard가 추가된다.
- Final++ appearance: feature256, context `2×32`, weighted feature std256을 사용한다. SH MLP weight shape는 `(256,576)`, `(256,256)`, `(75,256)`이다. 이 경로를 포함한 encoder 전체를 strict 복원한다.
- Ours mean/covariance의 안정화는 `mass_epsilon=1e-6` self prior를 사용한다. Prior는 opacity mass에 추가되지 않는다. 분포에서는 기본적으로 `α≥1/255`인 active slot만 선택한다.
- 두 모델 모두 2×256×256 pixel 순서로 총 131,072개 slot을 출력한다. Flatten은 view-major / row-major다.
- Dataset은 정규화 intrinsics, baseline=1, 첫 context camera 기준 canonical frame을 사용한다. Moment 내부 scene scale은 support norm의 median이며 최종 출력에서 복원된다.
- Renderer는 near 기반 전역 rescale을 추가하지만, 이번 가로축은 **그 전 최종 covariance의 `log10(max_k sqrt(eig_k(Σ)))`**다. 투영 크기 `f/z`나 pose alignment는 사용하지 않는다.
- 이전 코드 버전을 위한 appearance 생략 처리는 이번 구현에 없다. 전체 encoder weight 누락·shape mismatch는 오류로 보고한다.

실행 방법과 출력은 [README](../README.md)에 정리했다. 서버 데이터·실제 checkpoint를 이용한 결과는 로컬 테스트와 구분한다.
