# 4. Reconstruction-Based 방식 상세 분석

## 4.1 패러다임 개요

Reconstruction-based 방식은 정상 데이터로 학습된 재구성 모델(Auto-encoder, GAN 등)이 정상 샘플은 잘 재구성하지만 이상 샘플은 제대로 재구성하지 못하는 원리를 이용한다.

**핵심 수식**:

$$\text{Anomaly Score} = \|\mathbf{x} - \hat{\mathbf{x}}\|$$

여기서:
- $\mathbf{x}$: 입력 이미지
- $\hat{\mathbf{x}} = \text{Decoder}(\text{Encoder}(\mathbf{x}))$: 재구성된 이미지

**재구성 모델**:

$$\mathbf{z} = \text{Encoder}(\mathbf{x}), \quad \hat{\mathbf{x}} = \text{Decoder}(\mathbf{z})$$

**핵심 가정**: "정상 데이터만으로 학습된 모델은 정상 패턴의 manifold를 학습하며, 이상 데이터는 이 manifold 밖에 위치하여 재구성 시 큰 오류 발생"

**학습 목표**:

$$\min_{\theta} \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{normal}}} \|\mathbf{x} - \text{Decoder}(\text{Encoder}(\mathbf{x}))\|^2$$

---

## 4.2 GANomaly (2018)

### 4.2.1 기본 정보

- **논문**: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training
- **발표**: ACCV 2018
- **저자**: Samet Akcay et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/ganomaly

### 4.2.2 핵심 원리

GANomaly는 Generative Adversarial Network를 이용한 초기 이상 탐지 모델로, **Encoder-Decoder-Encoder (E-D-E)** 구조를 사용한다.

**독특한 E-D-E 구조**:

$$\mathbf{x} \xrightarrow{E_1} \mathbf{z}_1 \xrightarrow{D} \hat{\mathbf{x}} \xrightarrow{E_2} \mathbf{z}_2$$

**이중 Encoder의 원리**:
- 정상: $\mathbf{x} \approx \hat{\mathbf{x}} \Rightarrow \mathbf{z}_1 \approx \mathbf{z}_2$ (재구성 성공)
- 이상: $\mathbf{x} \neq \hat{\mathbf{x}} \Rightarrow \mathbf{z}_1 \neq \mathbf{z}_2$ (재구성 실패)

**Anomaly Score**:

$$\text{Score} = \|\mathbf{z}_1 - \mathbf{z}_2\|_2$$

### 4.2.3 기술적 세부사항

**Generator Loss** (E-D-E):

$$\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}} + \lambda_{\text{enc}}\mathcal{L}_{\text{enc}}$$

**Adversarial Loss**:
$$\mathcal{L}_{\text{adv}} = \mathbb{E}[\log D(\mathbf{x})] + \mathbb{E}[\log(1 - D(G(\mathbf{x})))]$$

**Contextual Loss** (재구성):
$$\mathcal{L}_{\text{con}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_1$$

**Encoder Loss** (latent 일관성):
$$\mathcal{L}_{\text{enc}} = \|\mathbf{z}_1 - \mathbf{z}_2\|_2$$

**Discriminator Loss**:

$$\mathcal{L}_D = -\mathbb{E}[\log D(\mathbf{x})] - \mathbb{E}[\log(1 - D(G(\mathbf{x})))]$$

### 4.2.4 성능

**MVTec AD 벤치마크**:
- Image AUROC: 93-95%
- 추론 속도: 50-80ms
- 메모리: 500MB-1GB
- 학습 시간: 6-10시간 (불안정한 수렴)

### 4.2.5 장점

1. **Semi-supervised 가능**: 소량의 이상 샘플 활용 가능
2. **생성 모델**: Realistic한 재구성 가능
3. **이중 검증**: Latent + Reconstruction 두 가지 신호
4. **초기 연구**: GAN 기반 이상 탐지의 선구자

### 4.2.6 단점

1. **학습 불안정**: GAN 특유의 mode collapse, oscillation
2. **하이퍼파라미터 민감**: $\lambda_{\text{adv}}, \lambda_{\text{con}}, \lambda_{\text{enc}}$ 튜닝 어려움
3. **느린 수렴**: Adversarial training으로 6-10시간 소요
4. **낮은 성능**: 최신 모델 대비 정확도 낮음 (93-95%)
5. **복잡한 구조**: E-D-E + Discriminator (4개 네트워크)

---

## 4.3 DRAEM (2021)

### 4.3.1 기본 정보

- **논문**: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection
- **발표**: ICCV 2021
- **저자**: Vitjan Zavrtanik et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/draem

### 4.3.2 핵심 원리

DRAEM은 reconstruction 패러다임을 혁신적으로 변화시켰다. **Simulated Anomaly**를 사용하여 supervised 학습 효과를 얻는다.

**패러다임 전환**:

기존 (GANomaly):
- 정상 데이터만 사용
- Unsupervised learning
- 이상 샘플을 본 적 없음

DRAEM:
- 정상 + Simulated anomaly 사용
- Supervised learning (discriminative)
- 이상 패턴을 명시적으로 학습

**Anomaly Simulation**:

$$\mathbf{x}_{\text{aug}} = (1 - \mathbf{m}) \odot \mathbf{x}_{\text{normal}} + \mathbf{m} \odot \mathbf{t}_{\text{source}}$$

여기서:
- $\mathbf{m} \in [0,1]^{H \times W}$: Binary mask (결함 영역)
- $\mathbf{t}_{\text{source}}$: 다른 이미지/텍스처의 패치
- $\odot$: Element-wise 곱셈

### 4.3.3 GANomaly 대비 핵심 차이점

| 측면 | GANomaly | DRAEM | 개선 효과 |
|------|----------|-------|----------|
| **학습 방식** | Unsupervised (정상만) | Supervised (정상+시뮬레이션) | 명확한 학습 신호 |
| **이상 샘플** | 없음 | Simulated anomaly | 이상 패턴 학습 |
| **네트워크 구조** | E-D-E + Discriminator | Reconstructive + Discriminative | 간단하면서 효과적 |
| **학습 안정성** | 불안정 (GAN) | 안정 (Supervised) | 빠른 수렴 |
| **Loss 함수** | Adversarial + L1 + L2 | SSIM + Focal + L2 | 더 강건한 학습 |
| **Image AUROC** | 93-95% | 97.5% | +2.5~4.5%p |
| **학습 시간** | 6-10시간 | 2-4시간 | 2-3배 단축 |
| **Few-shot** | 어려움 | 가능 (10-50장) | 적은 데이터 학습 |

### 4.3.4 기술적 세부사항

**Dual Network Architecture**:

1. **Reconstructive Subnetwork**: 시뮬레이션된 이상 제거
   $$\hat{\mathbf{x}} = R(\mathbf{x}_{\text{aug}})$$

2. **Discriminative Subnetwork**: 이상 영역 segmentation
   $$\hat{\mathbf{m}} = S(R(\mathbf{x}_{\text{aug}}))$$

**Loss Function**:

$$\mathcal{L} = \mathcal{L}_{\text{SSIM}}(\hat{\mathbf{x}}, \mathbf{x}_{\text{orig}}) + \mathcal{L}_{\text{focal}}(\hat{\mathbf{m}}, \mathbf{m})$$

**SSIM Loss** (구조적 유사도):

$$\text{SSIM}(\mathbf{x}, \mathbf{y}) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

$$\mathcal{L}_{\text{SSIM}} = 1 - \text{SSIM}(\mathbf{x}, \mathbf{y})$$

**Focal Loss** (class imbalance 해결):

$$\mathcal{L}_{\text{focal}} = -\alpha_t(1-p_t)^\gamma \log(p_t)$$

여기서:
- $p_t$: 예측 확률
- $\gamma$: Focusing parameter (보통 2)
- $\alpha_t$: Class balancing weight

### 4.3.5 Anomaly Simulation 상세

**Mask 생성 방법**:

1. **Random polygons**: 불규칙한 다각형
2. **Perlin noise**: 자연스러운 패턴
3. **Random brush strokes**: 붓 터치

**Source 선택**:
- DTD (Describable Textures Dataset)
- 다른 MVTec 카테고리
- Random noise

**Blending**:

$$\mathbf{x}_{\text{aug}} = \text{GaussianBlur}(\mathbf{x}_{\text{aug}}, \sigma)$$

부드러운 전환을 위한 후처리

### 4.3.6 성능

**MVTec AD 벤치마크**:
- Image AUROC: 97.5%
- Pixel AUROC: 96.8%
- 추론 속도: 50-100ms
- 메모리: 300-500MB
- 학습 시간: 2-4시간

### 4.3.7 장점

1. **높은 정확도**: 97.5% AUROC (reconstruction 중 최고)
2. **학습 안정**: GAN 없이 supervised learning
3. **Few-shot 가능**: 10-50장 정상 샘플로 학습
4. **빠른 학습**: 2-4시간
5. **강건성**: 다양한 이상 유형에 효과적
6. **Interpretable**: 명확한 segmentation map

### 4.3.8 단점

1. **Simulation 품질 의존**: 가짜 결함의 realistic 정도 중요
2. **Domain gap**: 시뮬레이션과 실제 결함 차이
3. **추가 데이터**: 텍스처 소스 데이터셋 필요
4. **하이퍼파라미터**: Simulation 파라미터 튜닝

---

## 4.4 DSR (2022)

### 4.4.1 기본 정보

- **논문**: A Dual Subspace Re-Projection Network for Surface Anomaly Detection
- **발표**: ECCV 2022
- **저자**: Vitjan Zavrtanik et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/dsr

### 4.4.2 핵심 원리

DSR은 두 개의 독립적인 **부분공간(subspace)**을 학습하고, 이미지를 이 부분공간에 재투영(re-projection)하여 재구성한다.

**Dual Subspace**:

1. **Quantization Subspace**: 이미지의 구조적 정보
   - VQ-VAE (Vector Quantized VAE) 기반
   - Discrete codebook 사용

2. **Target Subspace**: 세부적인 텍스처 정보
   - Continuous representation
   - 일반적인 VAE 기반

**재구성**:

$$\hat{\mathbf{x}} = f(\mathbf{z}_q, \mathbf{z}_t)$$

여기서:
- $\mathbf{z}_q$: Quantization subspace embedding (discrete)
- $\mathbf{z}_t$: Target subspace embedding (continuous)

### 4.4.3 DRAEM 대비 핵심 차이점

| 측면 | DRAEM | DSR | 개선 효과 |
|------|-------|-----|----------|
| **학습 방식** | Supervised (simulated anomaly) | Unsupervised (정상만) | 이상 샘플 불필요 |
| **재구성 방법** | 단일 Auto-encoder | Dual subspace projection | 구조+텍스처 분리 |
| **특징 표현** | Continuous latent | Discrete + Continuous | 더 풍부한 표현 |
| **적용 분야** | 일반 결함 | 복잡한 텍스처 표면 | 텍스처 결함 우수 |
| **Image AUROC** | 97.5% | 96.5-98.0% (카테고리별) | 텍스처에서 우수 |

### 4.4.4 기술적 세부사항

**Quantization Subspace (VQ-VAE)**:

Encoder:
$$\mathbf{z}_e = E_q(\mathbf{x})$$

Vector Quantization:
$$\mathbf{z}_q = \text{VQ}(\mathbf{z}_e) = \underset{\mathbf{e}_k \in \mathcal{C}}{\arg\min} \|\mathbf{z}_e - \mathbf{e}_k\|_2$$

여기서 $\mathcal{C} = \{\mathbf{e}_1, ..., \mathbf{e}_K\}$는 learnable codebook

Decoder:
$$\mathbf{f}_q = D_q(\mathbf{z}_q)$$

**VQ-VAE Loss**:

$$\mathcal{L}_{\text{VQ}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \|\text{sg}[\mathbf{z}_e] - \mathbf{z}_q\|^2 + \beta\|\mathbf{z}_e - \text{sg}[\mathbf{z}_q]\|^2$$

여기서 $\text{sg}[\cdot]$는 stop gradient

**Target Subspace (VAE)**:

Encoder:
$$\mu, \log\sigma^2 = E_t(\mathbf{x})$$

Reparameterization:
$$\mathbf{z}_t = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Decoder:
$$\mathbf{f}_t = D_t(\mathbf{z}_t)$$

**VAE Loss**:

$$\mathcal{L}_{\text{VAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \text{KL}(q(\mathbf{z}_t|\mathbf{x}) \| p(\mathbf{z}_t))$$

**Final Reconstruction**:

$$\hat{\mathbf{x}} = \text{Fusion}(\mathbf{f}_q, \mathbf{f}_t)$$

### 4.4.5 성능

**MVTec AD 벤치마크**:
- Image AUROC: 96.5-98.0% (카테고리별 차이 큼)
- Carpet, Leather, Tile 등 텍스처 카테고리에서 우수
- 추론 속도: 80-120ms
- 메모리: 500-800MB

### 4.4.6 장점

1. **복잡한 텍스처**: 직물, 카펫 등에서 우수
2. **풍부한 표현**: Discrete + Continuous
3. **Unsupervised**: 정상 데이터만 필요
4. **구조-텍스처 분리**: 명확한 표현

### 4.4.7 단점

1. **복잡한 구조**: 두 개의 subspace 학습
2. **학습 시간**: 각 subspace 학습으로 오래 걸림 (4-6시간)
3. **하이퍼파라미터**: Codebook 크기, latent 차원 등
4. **일반 결함**: 단순 결함에서는 DRAEM보다 낮을 수 있음

---

## 4.5 Reconstruction-Based 방식 종합 비교

### 4.5.1 기술적 진화 과정

```
GANomaly (2018)
├─ 시작: GAN 기반 E-D-E 구조
├─ 방식: Unsupervised, adversarial
├─ 성능: 93-95% AUROC
└─ 문제: 학습 불안정, 낮은 성능

        ↓ 패러다임 혁신

DRAEM (2021)
├─ 혁신: Simulated anomaly 사용
├─ 방식: Supervised, discriminative
├─ 성능: 97.5% AUROC (+2.5~4.5%p)
├─ 개선: 안정적 학습, Few-shot 가능
└─ 영향: Reconstruction 방식의 새 기준

        ↓ 특화 발전

DSR (2022)
├─ 특화: 복잡한 텍스처 표면
├─ 방식: Dual subspace (VQ-VAE + VAE)
├─ 성능: 96.5-98.0% (텍스처에서 우수)
└─ 특징: 구조-텍스처 분리 표현
```

### 4.5.2 상세 비교표

| 비교 항목 | GANomaly | DRAEM | DSR |
|----------|----------|-------|-----|
| **발표 연도** | 2018 | 2021 | 2022 |
| **학습 방식** | Unsupervised (GAN) | Supervised (Simulated) | Unsupervised (Dual VAE) |
| **네트워크 구조** | E-D-E + Discriminator | Reconstructive + Discriminative | Quantization + Target Subspace |
| **네트워크 수** | 4개 (E-D-E + D) | 2개 (Recon + Disc) | 2개 (VQ + Target) |
| **이상 샘플 사용** | 없음 | Simulated anomaly | 없음 |
| **학습 안정성** | 낮음 (mode collapse) ★☆☆☆☆ | 높음 (supervised) ★★★★★ | 중간 ★★★☆☆ |
| **주요 Loss** | Adversarial + L1 + L2 | SSIM + Focal + L2 | VQ + VAE + KL |
| **특징 표현** | Continuous latent | Continuous latent | Discrete + Continuous |
| **Image AUROC** | 93-95% | 97.5% ★★★★★ | 96.5-98.0% |
| **Pixel AUROC** | 91-93% | 96.8% ★★★★★ | 95.5-97.5% |
| **추론 속도** | 50-80ms | 50-100ms | 80-120ms |
| **학습 시간** | 6-10시간 | 2-4시간 ★★★★★ | 4-6시간 |
| **메모리 사용** | 500MB-1GB | 300-500MB ★★★★★ | 500-800MB |
| **Few-shot 능력** | 없음 | 우수 (10-50장) ★★★★★ | 중간 (50-100장) |
| **복잡한 텍스처** | 중간 | 중간 | 우수 ★★★★★ |
| **단순 결함** | 낮음 | 우수 ★★★★★ | 중간 |
| **구현 난이도** | 높음 (GAN 불안정) | 중간 | 높음 (Dual subspace) |
| **Interpretability** | 낮음 (latent distance) | 높음 (segmentation map) ★★★★★ | 중간 |
| **주요 혁신** | GAN 기반 이상 탐지 | Simulated anomaly | Dual subspace 분리 |
| **적합 환경** | Deprecated | 일반 결함 탐지 | 텍스처 표면 검사 |
| **종합 평가** | ★☆☆☆☆ | ★★★★★ | ★★★★☆ |

---

## 부록: 관련 테이블

### A.1 Reconstruction-Based vs 다른 패러다임

| 패러다임 | 대표 모델 | Image AUROC | 추론 속도 | 주요 장점 | 주요 단점 |
|---------|----------|-------------|-----------|----------|----------|
| **Reconstruction** | DRAEM | 97.5% | 50-100ms | Few-shot, 안정적 | Simulation 의존 |
| Memory-Based | PatchCore | 99.1% | 50-100ms | 최고 정확도 | 메모리 |
| Normalizing Flow | FastFlow | 98.5% | 20-50ms | 확률적 해석 | 학습 복잡 |
| Knowledge Distillation | Reverse Distillation | 98.6% | 100-200ms | SOTA급 | 느림 |
| Knowledge Distillation | EfficientAd | 97.8% | 1-5ms | 극한 속도 | 중간 정확도 |

### A.2 응용 시나리오별 Reconstruction 모델 선택

| 시나리오 | 권장 모델 | 이유 | 예상 성능 |
|---------|----------|------|----------|
| **Few-shot (10-50장)** | DRAEM | Simulated anomaly | 97.5% AUROC |
| **복잡한 텍스처 (직물, 카펫)** | DSR | Dual subspace | 98.0% AUROC |
| **일반 결함 탐지** | DRAEM | 안정적 학습 | 97.5% AUROC |
| **학습 데이터 많음** | DRAEM 또는 다른 패러다임 | - | - |
| **Unsupervised 필수** | DSR | 정상 데이터만 | 96.5-98.0% |

### A.3 핵심 Trade-off 분석

**학습 방식 Trade-off**:
```
Unsupervised (GANomaly, DSR):
+ 정상 데이터만 필요
- 암묵적 학습, 성능 제한
- 학습 불안정 (GANomaly)

Supervised (DRAEM):
+ 명확한 학습 신호
+ 높은 성능 (97.5%)
- Simulation 품질 의존
```

**구조 복잡도 vs 성능**:
```
GANomaly (복잡): 4개 네트워크, 93-95%
DRAEM (중간): 2개 네트워크, 97.5%
DSR (복잡): Dual subspace, 96.5-98.0%

결과: 복잡도가 반드시 성능으로 이어지지 않음
      DRAEM이 가장 효율적
```

**적용 도메인 Trade-off**:
```
DRAEM: 일반 결함 (97.5%)
DSR:   텍스처 표면 (98.0%)
       단순 결함 (96.5%)

→ 도메인 특화 vs 범용성
```

### A.4 하드웨어 요구사항

| 모델 | GPU 메모리 | CPU 추론 | 학습 시간 | 권장 환경 |
|------|-----------|----------|----------|----------|
| **GANomaly** | 4-8GB | 느림 (200ms+) | 6-10시간 | GPU 필수 |
| **DRAEM** | 4GB | 느림 (200ms+) | 2-4시간 | GPU 권장 |
| **DSR** | 4-8GB | 느림 (300ms+) | 4-6시간 | GPU 필수 |

### A.5 개발-배포 체크리스트 (Reconstruction-Based)

**Phase 1: 모델 선택**
- [ ] Few-shot 필요 여부 확인 (10-50장?)
- [ ] 텍스처 복잡도 평가
- [ ] DRAEM vs DSR 결정

**Phase 2: 데이터 준비 (DRAEM)**
- [ ] 정상 샘플 수집 (10-50장 이상)
- [ ] 텍스처 소스 준비 (DTD 등)
- [ ] Anomaly simulation 전략 수립

**Phase 3: Simulation 설계 (DRAEM)**
- [ ] Mask 생성 방법 선택 (polygon, perlin, brush)
- [ ] Source 선택 전략
- [ ] Blending 파라미터 조정

**Phase 4: 학습**
- [ ] Reconstructive network 학습
- [ ] Discriminative network 학습
- [ ] Loss convergence 모니터링

**Phase 5: 평가**
- [ ] Validation set 성능 확인
- [ ] Simulation 품질 검증
- [ ] 실제 결함과 비교

**Phase 6: 배포**
- [ ] 추론 파이프라인 구축
- [ ] Segmentation map 활용 방안
- [ ] 모니터링 시스템

### A.6 Reconstruction 선택 의사결정 트리

```
데이터 상황?
│
├─ 정상 샘플 적음 (10-50장)
│   └─ DRAEM (Few-shot)
│       - Simulated anomaly 활용
│
├─ 복잡한 텍스처 표면
│   └─ DSR
│       - 직물, 카펫, 가죽
│
├─ 일반적 상황
│   └─ DRAEM
│       - 안정적 학습
│
└─ Unsupervised 필수
    ├─ 복잡한 텍스처 → DSR
    └─ 단순 결함 → 다른 패러다임 고려
```

### A.7 GANomaly의 역사적 의의

**초기 연구로서의 가치**:
- GAN 기반 이상 탐지의 선구자 (2018)
- E-D-E 구조의 독창성
- 이후 연구의 기초

**현재 상태**:
- Deprecated (사용 비추천)
- DRAEM으로 완전히 대체됨
- 학습 불안정성과 낮은 성능

**교훈**:
- GAN의 학습 불안정성이 실무 적용의 장벽
- Supervised 접근(DRAEM)이 더 효과적
- 복잡한 구조가 반드시 좋은 것은 아님

### A.8 DRAEM Simulation 파라미터 가이드

**Mask 생성**:
- Polygon vertices: 4-12개
- Perlin noise scale: 0.1-1.0
- Brush stroke width: 5-20 pixels

**Source 선택**:
- DTD (Describable Textures): 47개 카테고리
- 다른 MVTec 카테고리
- Gaussian noise (sigma: 0.1-0.3)

**Blending**:
- Gaussian blur kernel: 3-7
- Alpha blending: 0.7-1.0

### A.9 성능 벤치마크 요약

**정확도 순위**:
1. DRAEM (97.5%) ★★★★★
2. DSR (96.5-98.0%, 텍스처)
3. GANomaly (93-95%)

**학습 안정성 순위**:
1. DRAEM (supervised) ★★★★★
2. DSR (dual VAE)
3. GANomaly (GAN, 불안정)

**실용성 순위**:
1. DRAEM (Few-shot + 안정) ★★★★★
2. DSR (텍스처 특화)
3. GANomaly (deprecated)

---

## 결론

Reconstruction-based 방식은 **DRAEM을 정점으로 Few-shot 학습과 안정적 성능**을 보여주고 있다. 특히:

1. **패러다임 혁신**: Simulated anomaly로 supervised 학습 효과
2. **Few-shot 능력**: 10-50장으로 97.5% AUROC 달성
3. **학습 안정성**: GAN 없이 안정적 수렴
4. **실용성**: 간단한 구조, 빠른 학습 (2-4시간)

**핵심 발견**:
- GANomaly → DRAEM: Simulated anomaly로 +2.5~4.5%p 향상
- Unsupervised (GAN) → Supervised (Simulation): 안정성과 성능 모두 개선
- DSR: 텍스처 특화 도메인에서 추가 가치

**최종 권장사항**:
- **Few-shot 환경**: DRAEM 사용 (10-50장, 97.5%)
- **복잡한 텍스처**: DSR 사용 (직물, 카펫)
- **일반 결함**: DRAEM 또는 다른 패러다임 (PatchCore, FastFlow) 고려
- **GANomaly**: 사용 비추천 (deprecated)

Reconstruction-based 방식은 특히 **학습 데이터가 부족한 상황**에서 강력한 대안을 제공하며, DRAEM의 simulated anomaly 접근은 이상 탐지 연구에 중요한 기여를 했다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: GANomaly, DRAEM, DSR

**주요 내용**:
1. Reconstruction-Based 패러다임 개요
2. GANomaly 상세 분석 (E-D-E 구조, GAN loss)
3. DRAEM 상세 분석 (Simulated Anomaly 혁신, SSIM/Focal loss)
4. DSR 상세 분석 (Dual Subspace: VQ-VAE + VAE)
5. 종합 비교 및 진화 과정
6. **부록**: overall_report.md의 관련 테이블 포함
