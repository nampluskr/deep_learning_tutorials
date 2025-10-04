# 2. Normalizing Flow 방식 상세 분석

## 2.1 패러다임 개요

Normalizing Flow는 생성 모델(generative model)의 일종으로, 가역적인(invertible) 변환을 통해 복잡한 데이터 분포를 단순한 분포(예: 표준 정규분포)로 매핑한다.

**핵심 수식**:

$$\mathbf{x} \xleftrightarrow{f} \mathbf{z}$$

여기서:
- $\mathbf{x}$: 복잡한 데이터 (이미지 특징)
- $\mathbf{z}$: 단순한 latent variable, $\mathbf{z} \sim \mathcal{N}(0, I)$
- $f$: 가역 변환 (invertible transformation)

**Change of Variables 공식**:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log\left|\det\frac{\partial f}{\partial \mathbf{x}}\right|$$

$$= \log p(f(\mathbf{x})) + \log|\det J_f(\mathbf{x})|$$

여기서 $J_f$는 Jacobian 행렬

**이상 탐지 원리**:
- 높은 $\log p(\mathbf{x})$ → 정상 (학습된 분포 내)
- 낮은 $\log p(\mathbf{x})$ → 이상 (분포 밖)

**Anomaly Score**:

$$\text{Score}(\mathbf{x}) = -\log p(\mathbf{x})$$

---

## 2.2 CFLOW (2021)

### 2.2.1 기본 정보

- **논문**: Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows
- **발표**: WACV 2022
- **저자**: Denis Gudovskiy et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/cflow

### 2.2.2 핵심 원리

CFLOW는 **Conditional Normalizing Flow**를 사용하여 이미지 특징의 위치별 조건부 분포를 모델링한다.

**Conditional Distribution**:

각 공간 위치 $(i,j)$에서:

$$p(\mathbf{x}_{i,j} | \text{pos}_{i,j}) = p(f(\mathbf{x}_{i,j} | \text{pos}_{i,j})) \cdot |\det J_f|$$

여기서 $\text{pos}_{i,j}$는 위치 정보 (position encoding)

**Multi-scale Architecture**:

$$\log p(\mathbf{x}) = \sum_{s=1}^{S} \log p_s(\mathbf{x}_s)$$

여기서 $s$는 스케일 인덱스 (layer1, layer2, layer3)

### 2.2.3 기술적 세부사항

**Position Encoding**:

2D positional embedding:

$$\text{PE}(i,j) = [\sin(\omega_1 i), \cos(\omega_1 i), \sin(\omega_2 j), \cos(\omega_2 j), ...]$$

**Affine Coupling Layer**:

입력 $\mathbf{x}$를 두 부분으로 분할: $\mathbf{x} = [\mathbf{x}_a, \mathbf{x}_b]$

변환:
$$\mathbf{y}_a = \mathbf{x}_a$$
$$\mathbf{y}_b = \mathbf{x}_b \odot \exp(s(\mathbf{x}_a, c)) + t(\mathbf{x}_a, c)$$

여기서:
- $s, t$: 신경망 (scale, translation)
- $c$: condition (position encoding)
- $\odot$: element-wise 곱셈

**Jacobian determinant**:

$$\log|\det J| = \sum s(\mathbf{x}_a, c)$$

(대각 행렬이므로 계산이 효율적)

### 2.2.4 성능

**MVTec AD 벤치마크**:
- Image AUROC: 98.2%
- Pixel AUROC: 97.6%
- 추론 속도: 100-150ms per image
- 메모리: 500MB-1GB

### 2.2.5 장점

1. **확률적 해석**: Log-likelihood로 명확한 이상 점수
2. **Pixel-level Localization**: 각 위치별 NLL 계산 가능
3. **Multi-scale**: 다양한 크기의 이상 탐지
4. **조건부 모델링**: 위치별 다른 정상 패턴 학습 가능

### 2.2.6 단점

1. **학습 복잡도**: Flow network 학습 시간 오래 걸림 (2-3시간)
2. **메모리 사용**: Multi-scale flow networks 저장
3. **추론 시간**: Forward flow 계산 비용 (100-150ms)
4. **하이퍼파라미터**: Flow depth, coupling layers 수 등 튜닝 필요

---

## 2.3 FastFlow (2021)

### 2.3.1 기본 정보

- **논문**: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows
- **발표**: CVPR 2021 (Oral)
- **저자**: Jiawei Yu et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/fastflow

### 2.3.2 핵심 원리

FastFlow는 CFLOW의 **3D flow를 2D flow로 단순화**하여 추론 속도를 대폭 향상시켰다.

**핵심 차이점**:
- CFLOW: 3D tensor $(C \times H \times W)$에 flow 적용
- FastFlow: 2D spatial locations $(H \times W)$에 flow 적용, 채널 차원 분리

**수학적 표현**:

특징 맵 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$를 reshape:

$$\mathbf{F} \rightarrow \{f_{h,w}\}_{h=1,w=1}^{H,W}, \quad f_{h,w} \in \mathbb{R}^C$$

각 위치에서 독립적으로 flow 적용:

$$p(\mathbf{F}) = \prod_{h=1}^{H} \prod_{w=1}^{W} p(f_{h,w})$$

### 2.3.3 CFLOW 대비 핵심 차이점

| 측면 | CFLOW | FastFlow | 개선 효과 |
|------|-------|----------|----------|
| **Flow 차원** | 3D $(C \times H \times W)$ | 2D $(H \times W)$ | 계산량 대폭 감소 |
| **채널 처리** | 통합 처리 | 독립 처리 후 aggregation | 병렬화 가능 |
| **Coupling Layers** | 8-12개 | 4-8개 | 학습/추론 속도 향상 |
| **Jacobian 계산** | 고차원 | 저차원 | 빠른 log-det 계산 |
| **추론 속도** | 100-150ms | 20-50ms | 2-3배 향상 |
| **Image AUROC** | 98.2% | 98.5% | 유지/향상 |

### 2.3.4 기술적 세부사항

**2D Normalizing Flow**:

각 채널 $c$에 대해:

$$\log p(f_{h,w}^{(c)}) = \log p(z_{h,w}^{(c)}) + \log|\det J_{f_c}|$$

전체 이상 점수:

$$\text{Score} = -\sum_{c=1}^{C} \sum_{h=1}^{H} \sum_{w=1}^{W} \log p(f_{h,w}^{(c)})$$

**Simplified Coupling Layer**:

$$\mathbf{y} = \mathbf{x} \odot \exp(s) + t$$

여기서 $s, t \in \mathbb{R}^{H \times W}$ (채널별로 독립)

### 2.3.5 CFLOW 대비 개선사항

**1) 추론 속도**:
- CFLOW: 100-150ms
- FastFlow: 20-50ms
- **개선율**: 2-3배 향상

**2) 메모리 효율**:
- CFLOW: 각 스케일에서 3D flow
- FastFlow: 각 스케일에서 2D flow
- **개선율**: 30-50% 메모리 감소

**3) 성능 유지/향상**:
- CFLOW: 98.2% AUROC
- FastFlow: 98.5% AUROC
- **결과**: 속도 향상하면서 성능도 미세 향상

**4) 학습 시간**:
- CFLOW: 2-3시간 (MVTec AD 1개 카테고리)
- FastFlow: 30-60분
- **개선율**: 3-4배 향상

**5) 병렬화**:
- 채널 독립 처리로 병렬화 용이

### 2.3.6 Trade-off 분석

**FastFlow가 포기한 것**:
- 채널 간 상관관계 모델링
- 복잡한 3D 분포 표현력

**FastFlow가 얻은 것**:
- 대폭 빠른 속도
- 간단한 구조로 학습 안정성
- 실시간 처리 가능성

**왜 성능이 유지/향상되는가?**:
- 채널 간 상관관계가 이상 탐지에 크게 중요하지 않음
- 2D 공간 구조가 더 중요
- 간단한 모델이 오히려 과적합 방지

### 2.3.7 성능

**MVTec AD 벤치마크**:
- Image AUROC: 98.5%
- Pixel AUROC: 97.8%
- 추론 속도: 20-50ms per image
- 메모리: 500MB-1GB
- 학습 시간: 30-60분

### 2.3.8 장점

1. **빠른 속도**: 실시간 처리 가능 수준 (20-50ms)
2. **높은 정확도**: CFLOW와 동등 이상 (98.5%)
3. **학습 효율**: 빠른 학습으로 빠른 iteration
4. **구현 간단**: 2D flow로 복잡도 감소

### 2.3.9 단점

1. **채널 정보 손실**: 채널 간 관계 무시
2. **이론적 완전성**: CFLOW보다 단순한 가정
3. 여전히 flow 계산 오버헤드 존재

---

## 2.4 CS-Flow (2021)

### 2.4.1 기본 정보

- **논문**: Fully Convolutional Cross-Scale-Flows for Image-based Defect Detection
- **발표**: WACV 2022
- **저자**: Marco Rudolph et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/csflow

### 2.4.2 핵심 원리

CS-Flow는 서로 다른 스케일 간의 정보를 명시적으로 교환하는 **Cross-Scale Flow**를 제안한다.

**기존 방법의 한계**:
- CFLOW, FastFlow: 각 스케일 독립적으로 처리 후 합산
- 스케일 간 상호작용 없음

**CS-Flow의 해결책**:

$$p(\mathbf{x}) = p(\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \mathbf{x}^{(3)})$$

스케일 간 정보 흐름:
- 큰 스케일 → 작은 스케일: context 정보 전달
- 작은 스케일 → 큰 스케일: detail 정보 전달

### 2.4.3 CFLOW/FastFlow와의 차이점

| 측면 | CFLOW | FastFlow | CS-Flow |
|------|-------|----------|---------|
| **스케일 처리** | 독립적 | 독립적 | 상호 연결 |
| **정보 흐름** | 단방향 | 단방향 | 양방향 (cross-scale) |
| **Architecture** | 3개 독립 flow | 3개 독립 flow | 통합 hierarchical flow |
| **다양한 크기 결함** | 중간 | 중간 | 우수 ★★★★★ |
| **Image AUROC** | 98.2% | 98.5% | 97.9% |

### 2.4.4 기술적 세부사항

**Cross-Scale Connection**:

Top-down path (context):
$$\mathbf{z}^{(3)} = f_3(\mathbf{x}^{(3)})$$
$$\mathbf{x}^{(2)'} = \mathbf{x}^{(2)} + \text{Upsample}(\mathbf{z}^{(3)})$$
$$\mathbf{z}^{(2)} = f_2(\mathbf{x}^{(2)'})$$

**Fully Convolutional Design**:
- 입력 이미지 크기에 무관하게 동작
- 다양한 해상도 지원

### 2.4.5 성능

**MVTec AD 벤치마크**:
- Image AUROC: 97.9%
- Pixel AUROC: 97.5%
- 추론 속도: 80-120ms per image
- 특정 카테고리(Grid, Tile)에서 우수

### 2.4.6 장점

1. **Multi-scale robustness**: 다양한 크기 결함에 강건
2. **Context integration**: 전역-지역 정보 통합
3. **Flexible resolution**: 다양한 입력 크기 지원

### 2.4.7 단점

1. **복잡한 구조**: 구현 및 디버깅 어려움
2. **학습 시간**: Cross-scale connection으로 더 오래 걸림 (3-4시간)
3. **하이퍼파라미터**: 더 많은 튜닝 필요
4. **속도**: FastFlow보다 느림 (80-120ms)

---

## 2.5 U-Flow (2022)

### 2.5.1 기본 정보

- **논문**: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold
- **발표**: arXiv 2022
- **저자**: Matej Bergant et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/uflow

### 2.5.2 핵심 원리

U-Flow는 **U-Net 구조**를 normalizing flow에 적용하고, 비지도 방식으로 임계값을 자동 설정한다.

**기존 방법의 한계**:
- 이상 점수 계산 후 임계값을 수동으로 설정해야 함
- 데이터셋마다 다른 임계값 필요

**U-Flow의 해결책**:

자동 임계값 추정:

$$\tau = \mu_{\text{train}} + k \cdot \sigma_{\text{train}}$$

여기서:
- $\mu_{\text{train}}$: 학습 데이터 이상 점수 평균
- $\sigma_{\text{train}}$: 표준편차
- $k$: 상수 (예: 3)

### 2.5.3 기술적 세부사항

**U-shaped Flow Network**:

Encoder path (downsampling):
$$\mathbf{z}_1, \mathbf{z}_2, \mathbf{z}_3, \mathbf{z}_4$$

Decoder path (upsampling):
$$\mathbf{z}_4', \mathbf{z}_3', \mathbf{z}_2', \mathbf{z}_1'$$

Skip connections (U-Net style):
$$\mathbf{z}_i' = \text{Upsample}(\mathbf{z}_{i+1}') + \mathbf{z}_i$$

**Unsupervised Threshold Estimation**:

방법 1: Percentile-based
$$\tau = \text{percentile}(S_{\text{train}}, 95)$$

방법 2: Statistical (mean + k·std)
$$\tau = \mu + 3\sigma$$

### 2.5.4 성능

**MVTec AD 벤치마크**:
- Image AUROC: 97.6%
- Pixel AUROC: 96.8%
- 추론 속도: 90-140ms per image
- 자동 임계값: 수동 설정 대비 1-2%p 이내 차이

### 2.5.5 장점

1. **자동 임계값**: 수동 튜닝 불필요
2. **적응형**: 운영 환경 변화에 대응 가능
3. **U-Net 구조**: 효과적인 정보 융합

### 2.5.6 단점

1. **복잡한 구조**: U-Net + Flow 결합
2. **학습 시간**: 더 깊은 네트워크로 느림 (4-5시간)
3. **임계값 신뢰성**: 학습 데이터 품질에 의존

---

## 2.6 Normalizing Flow 방식 종합 비교

### 2.6.1 기술적 진화 과정

```
CFLOW (2021)
├─ 혁신: Conditional flow로 위치별 조건부 분포
├─ 특징: Multi-scale, 3D flow
├─ 문제: 느린 속도 (100-150ms)
└─ 성능: 98.2% AUROC

        ↓ 속도 최적화

FastFlow (2021)
├─ 개선: 3D→2D flow로 계산량 대폭 감소
├─ 결과: 2-3배 속도 향상 (20-50ms)
├─ 성능: 98.5% AUROC (유지/향상)
└─ Trade-off: 채널 간 상관관계 무시

        ↓ 기능 강화 (분기)

CS-Flow (2021)           U-Flow (2022)
├─ 개선: Cross-scale 융합   ├─ 개선: U-Net + 자동 임계값
├─ 장점: 다양한 크기 결함    ├─ 장점: 운영 자동화
├─ 성능: 97.9% AUROC       ├─ 성능: 97.6% AUROC
└─ 단점: 복잡도 증가        └─ 단점: 학습 시간 증가
```

### 2.6.2 상세 비교표

| 비교 항목 | CFLOW | FastFlow | CS-Flow | U-Flow |
|----------|-------|----------|---------|--------|
| **발표 연도** | 2021 | 2021 | 2021 | 2022 |
| **Flow 차원** | 3D $(C \times H \times W)$ | 2D $(H \times W)$ | 2D + Cross-scale | 2D + U-Net |
| **스케일 처리** | 독립적 3개 flow | 독립적 3개 flow | 상호 연결 flow | Hierarchical U-Net |
| **채널 처리** | 통합 | 독립 | 독립 | 통합 |
| **Skip Connection** | 없음 | 없음 | Cross-scale | U-Net style |
| **Coupling Layers** | 8-12개 | 4-8개 | 6-10개 | 10-16개 |
| **임계값 설정** | 수동 | 수동 | 수동 | 자동 ★★★★★ |
| **Image AUROC** | 98.2% | 98.5% ★★★★★ | 97.9% | 97.6% |
| **Pixel AUROC** | 97.6% | 97.8% ★★★★★ | 97.5% | 96.8% |
| **추론 속도** | 100-150ms | 20-50ms ★★★★★ | 80-120ms | 90-140ms |
| **학습 시간** | 2-3시간 | 30-60분 ★★★★★ | 3-4시간 | 4-5시간 |
| **메모리 사용** | 500MB-1GB | 500MB-1GB | 600MB-1.2GB | 700MB-1.5GB |
| **구현 난이도** | 중간 | 낮음 ★★★★★ | 높음 | 높음 |
| **다양한 크기 결함** | 중간 | 중간 | 우수 ★★★★★ | 중간 |
| **운영 편의성** | 낮음 | 낮음 | 낮음 | 높음 ★★★★★ |
| **주요 혁신** | Conditional flow | 2D flow (속도) | Cross-scale | Auto threshold |
| **종합 평가** | ★★★☆☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |

### 2.6.3 핵심 Trade-off 분석

**1) 속도 vs 표현력**:
```
CFLOW: 느림 (100-150ms) + 강력한 3D 표현력
  ↓
FastFlow: 빠름 (20-50ms) + 단순한 2D 표현력
결과: 속도 3배↑, 성능 유지/향상
      (채널 상관관계가 덜 중요함을 입증)
```

**2) 성능 vs 복잡도**:
```
FastFlow: 간단 + 높은 성능 (98.5%)
  ↓
CS-Flow: 복잡 + 비슷한 성능 (97.9%)
결과: 복잡도 증가가 성능 향상으로 이어지지 않음
      → FastFlow가 더 실용적
```

**3) 자동화 vs 성능**:
```
FastFlow: 수동 임계값 + 98.5%
  ↓
U-Flow: 자동 임계값 + 97.6%
결과: 0.9%p 성능 희생, 운영 편의성 획득
      → 상황에 따라 선택
```

### 2.6.4 실무 적용 가이드

**CFLOW 선택 시나리오**:
- 최고 정확도와 해석 가능성 필요
- 속도 제약 없음
- 연구 목적 또는 baseline
- **추천도**: ★★☆☆☆ (FastFlow로 대체 가능)

**FastFlow 선택 시나리오**:
- 높은 정확도 + 빠른 속도 필요
- 실시간 처리는 아니지만 응답 시간 중요
- 대부분의 실무 적용
- 균형잡힌 성능 원함
- **추천도**: ★★★★★ (최고 추천)

**CS-Flow 선택 시나리오**:
- 크기가 매우 다양한 결함 존재
- 복잡한 multi-scale 패턴
- 특정 카테고리(Grid, Tile)에서 최고 성능 필요
- **추천도**: ★★★☆☆ (특수 상황)

**U-Flow 선택 시나리오**:
- 자동화된 운영 시스템 구축
- 임계값 수동 조정 불가능
- 환경 변화가 잦은 상황
- **추천도**: ★★★☆☆ (자동화 필요 시)

---

## 부록: 관련 테이블

### A.1 Normalizing Flow vs 다른 패러다임

| 패러다임 | 대표 모델 | Image AUROC | 추론 속도 | 주요 장점 | 주요 단점 |
|---------|----------|-------------|-----------|----------|----------|
| **Normalizing Flow** | FastFlow | 98.5% | 20-50ms | 확률적 해석, 빠름 | 학습 복잡 |
| Memory-Based | PatchCore | 99.1% | 50-100ms | 최고 정확도 | 메모리 사용 |
| Knowledge Distillation | Reverse Distillation | 98.6% | 100-200ms | SOTA급 | 느림 |
| Knowledge Distillation | EfficientAd | 97.8% | 1-5ms | 극한 속도 | 중간 정확도 |
| Reconstruction | DRAEM | 97.5% | 50-100ms | Few-shot | Simulation 의존 |

### A.2 응용 시나리오별 Normalizing Flow 모델 선택

| 시나리오 | 권장 모델 | 이유 | 예상 성능 |
|---------|----------|------|----------|
| **균형잡힌 일반 검사** | FastFlow | 속도+정확도 | 98.5% AUROC, 20-50ms |
| **다양한 크기 결함** | CS-Flow | Multi-scale 강건 | 97.9% AUROC |
| **자동화 운영** | U-Flow | 자동 임계값 | 97.6% AUROC |
| **연구/Baseline** | CFLOW | 확률적 해석 | 98.2% AUROC |
| **빠른 프로토타입** | FastFlow | 빠른 학습 (30-60분) | 98.5% AUROC |

### A.3 성능 벤치마크 요약

**정확도 순위**:
1. FastFlow (98.5%) ★★★★★
2. CFLOW (98.2%)
3. CS-Flow (97.9%)
4. U-Flow (97.6%)

**속도 순위**:
1. FastFlow (20-50ms) ★★★★★
2. CS-Flow (80-120ms)
3. U-Flow (90-140ms)
4. CFLOW (100-150ms)

**실용성 순위**:
1. FastFlow (속도+성능+간단함) ★★★★★
2. U-Flow (자동화)
3. CFLOW (baseline)
4. CS-Flow (특수 케이스)

### A.4 하드웨어 요구사항

| 모델 | GPU 메모리 | CPU 추론 | 학습 시간 | 권장 환경 |
|------|-----------|----------|----------|----------|
| **CFLOW** | 4-8GB | 불가능 | 2-3시간 | GPU 필수 |
| **FastFlow** | 4-8GB | 매우 느림 | 30-60분 | GPU 필수 |
| **CS-Flow** | 6-8GB | 불가능 | 3-4시간 | GPU 필수 |
| **U-Flow** | 6-10GB | 불가능 | 4-5시간 | 고성능 GPU |

### A.5 개발-배포 체크리스트 (Normalizing Flow)

**Phase 1: 모델 선택**
- [ ] 성능 목표 설정 (98%+ 필요?)
- [ ] 속도 요구사항 확인 (실시간? 준실시간?)
- [ ] FastFlow vs 다른 모델 결정

**Phase 2: 데이터 준비**
- [ ] 정상 샘플 수집 (100-500장)
- [ ] 이미지 전처리 및 augmentation
- [ ] Train/validation split

**Phase 3: 학습**
- [ ] Pre-trained backbone 선택
- [ ] Flow depth 설정 (4-8 coupling layers)
- [ ] Multi-scale 설정 (보통 3 scales)
- [ ] 학습 모니터링 (NLL convergence)

**Phase 4: 평가 및 최적화**
- [ ] Validation set 성능 확인
- [ ] 임계값 설정 (수동 or U-Flow 자동)
- [ ] Pixel-level localization 품질 검증
- [ ] 속도 벤치마크

**Phase 5: 배포**
- [ ] 추론 파이프라인 최적화
- [ ] Batch inference 구현 (가능 시)
- [ ] 메모리 사용량 모니터링
- [ ] A/B 테스트

### A.6 Normalizing Flow 선택 의사결정 트리

```
Normalizing Flow 사용 고려?
│
├─ 확률적 해석 필요? → YES
│   └─ 속도 중요? 
│       ├─ YES → FastFlow
│       └─ NO → CFLOW
│
├─ 다양한 크기 결함? → YES
│   └─ CS-Flow
│
├─ 자동 임계값 필요? → YES
│   └─ U-Flow
│
└─ 일반적 상황
    └─ FastFlow (기본 추천)
```

---

## 결론

Normalizing Flow 방식은 **FastFlow를 중심으로 확률론적 해석과 실용적 성능의 균형**을 이루고 있다. 특히:

1. **정확도**: 98.5% AUROC (SOTA급)
2. **속도**: 20-50ms (실용적 수준)
3. **이론적 견고성**: 명확한 확률론적 기반
4. **실무 적용**: 균형잡힌 성능으로 널리 채택

**핵심 발견**:
- CFLOW → FastFlow 진화: 3D→2D 단순화로 속도 3배 향상, 성능 유지
- 채널 간 상관관계가 이상 탐지에 덜 중요함을 입증
- CS-Flow, U-Flow: 특수 상황에서 유용하나 일반적으로는 FastFlow 우세

**최종 권장사항**:
- **대부분의 경우**: FastFlow 사용 (98.5%, 20-50ms)
- **다양한 크기 결함**: CS-Flow 고려
- **자동화 필요**: U-Flow 고려
- **연구/Baseline**: CFLOW 사용

Normalizing Flow는 Memory-based와 함께 이상 탐지의 핵심 패러다임으로 자리잡았으며, 특히 FastFlow는 실무에서 가장 널리 사용되는 모델 중 하나이다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: CFLOW, FastFlow, CS-Flow, U-Flow

**주요 내용**:
1. Normalizing Flow 패러다임 개요 (Change of Variables 공식 포함)
2. CFLOW 상세 분석 (Conditional Flow 수식)
3. FastFlow 상세 분석 (2D vs 3D Flow 비교)
4. CS-Flow 상세 분석 (Cross-Scale Connection)
5. U-Flow 상세 분석 (자동 임계값)
6. 종합 비교 및 실무 가이드
7. **부록**: overall_report.md의 관련 테이블 포함