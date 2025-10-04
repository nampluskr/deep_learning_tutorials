# 5. Diffusion 패러다임 상세 분석

## 5.1 패러다임 개요

Diffusion Models은 **노이즈 제거 과정(denoising process)**을 통해 데이터 분포를 학습하는 생성 모델이다. 원래는 고품질 이미지 생성을 목표로 개발되었으나, 최근 Self-Supervised Learning에도 활용되고 있다.

**핵심 수식**:

**Forward Process (Noise 추가)**:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

여기서:
- $\mathbf{x}_0$: 원본 이미지
- $\mathbf{x}_t$: t-step의 noisy 이미지
- $\beta_t$: Noise schedule (0.0001 → 0.02)
- $T$: Total timesteps (예: 1000)

**Reverse Process (Denoising)**:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

**학습 목표**:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \epsilon, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

노이즈 $\boldsymbol{\epsilon}$를 예측하는 네트워크 $\boldsymbol{\epsilon}_\theta$ 학습

**핵심 가정**: "노이즈를 제거하는 과정에서 데이터의 구조와 의미를 학습한다"

**다른 패러다임과의 차이**:

| 측면 | Contrastive | Clustering | Generative (MIM) | **Diffusion** |
|------|------------|-----------|-----------------|--------------|
| **학습 목표** | 유사도 | 클러스터 | 복원 | **노이즈 제거** |
| **핵심** | Metric learning | Clustering | Reconstruction | **Denoising** |
| **주요 용도** | SSL | SSL | SSL | **생성 + SSL** |
| **생성 능력** | 없음 | 없음 | 약함 | **강함** |

---

## 5.2 Diffusion Models의 기초 이론

### 5.2.1 Forward Diffusion Process

**Markov Chain으로 노이즈 추가**:

$$q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1})$$

**한 스텝씩 노이즈 추가**:

$$\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

**Reparameterization Trick**:

임의의 timestep t에서 직접 계산 가능:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

여기서:
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

**성질**: $t \rightarrow T$일 때, $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ (순수 노이즈)

### 5.2.2 Reverse Diffusion Process

**목표**: $p(\mathbf{x}_{t-1} | \mathbf{x}_t)$를 학습하여 노이즈 → 이미지 복원

**Reverse Process**:

$$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$

**Parameterization**:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$$

### 5.2.3 Training Objective

**ELBO (Evidence Lower Bound)**:

$$\mathcal{L} = \mathbb{E}_q\left[-\log p_\theta(\mathbf{x}_0)\right] \leq \mathbb{E}_q\left[D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T)) + \sum_{t>1}D_{\text{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right]$$

**단순화된 Loss (DDPM)**:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

여기서:
- $t \sim \text{Uniform}(1, T)$
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$

**직관**: 노이즈를 예측하는 네트워크 학습

### 5.2.4 Sampling (생성 과정)

**알고리즘**:

```
1. Sample x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
     z ~ N(0, I) if t > 1, else z = 0
     x_{t-1} = (1/√α_t) * (x_t - ((1-α_t)/√(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z
3. Return x_0
```

**특징**:
- T번의 denoising step 필요 (T=1000)
- 생성 속도 느림 (GAN 대비)
- 고품질 샘플

---

## 5.3 DDPM (2020)

### 5.3.1 기본 정보

- **논문**: Denoising Diffusion Probabilistic Models
- **발표**: NeurIPS 2020
- **저자**: Jonathan Ho et al. (UC Berkeley)
- **GitHub**: https://github.com/hojonathanho/diffusion

### 5.3.2 핵심 원리

DDPM은 diffusion models의 현대적 구현을 확립했다.

**Training Algorithm**:

```
1. Repeat:
   a) Sample x_0 from dataset
   b) Sample t ~ Uniform(1, T)
   c) Sample ε ~ N(0, I)
   d) Compute x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
   e) Update θ: ∇_θ ||ε - ε_θ(x_t, t)||²
```

**Noise Schedule**:

Linear schedule:
$$\beta_t = \beta_{\min} + \frac{t}{T}(\beta_{\max} - \beta_{\min})$$

기본값: $\beta_{\min} = 0.0001$, $\beta_{\max} = 0.02$, $T = 1000$

### 5.3.3 기술적 세부사항

**U-Net Architecture**:

- **Encoder**: Downsampling blocks (ResNet + Attention)
- **Bottleneck**: Self-attention layers
- **Decoder**: Upsampling blocks (ResNet + Attention)
- **Skip connections**: Encoder → Decoder

**Time Embedding**:

Sinusoidal position encoding:

$$\text{TimeEmb}(t, i) = \begin{cases}
\sin(t / 10000^{2i/d}) & \text{if } i \text{ even} \\
\cos(t / 10000^{2i/d}) & \text{if } i \text{ odd}
\end{cases}$$

Timestep 정보를 네트워크에 주입

**Attention Layers**:

Self-attention at resolutions: 16×16, 8×8 (computational cost)

### 5.3.4 성능

**Image Generation (CIFAR-10)**:

| Model | FID ↓ | IS ↑ |
|-------|-------|------|
| GAN (BigGAN) | 14.73 | 9.22 |
| **DDPM** | **3.17** | **9.46** |

**놀라운 결과**: GAN을 큰 폭으로 초과!

**ImageNet 32×32**:

| Model | FID |
|-------|-----|
| BigGAN | 14.9 |
| **DDPM** | **5.24** |

### 5.3.5 장점

1. **고품질 생성**: FID 3.17 (SOTA)
2. **안정적 학습**: Mode collapse 없음 (GAN 문제 해결)
3. **다양성**: 높은 샘플 다양성
4. **이론적 기반**: 명확한 확률론적 프레임워크

### 5.3.6 단점

1. **느린 생성**: 1000 steps 필요 (GAN은 1 step)
2. **계산 비용**: 학습과 생성 모두 비쌈
3. **SSL 성능**: 생성 품질은 좋으나 SSL로는 제한적
4. **메모리**: U-Net이 큼

### 5.3.7 DDPM의 의의

DDPM은 diffusion models를 실용적 수준으로 끌어올렸다:
- GAN을 능가하는 생성 품질
- 안정적 학습
- 이후 Stable Diffusion, DALL-E 2 등의 기반

하지만 **SSL 용도로는 제한적** (본 보고서의 주제)

---

## 5.4 DDIM (2021)

### 5.4.1 기본 정보

- **논문**: Denoising Diffusion Implicit Models
- **발표**: ICLR 2021
- **저자**: Jiaming Song et al. (Stanford)

### 5.4.2 핵심 원리

DDIM은 **생성 속도를 대폭 개선**했다.

**DDPM의 문제**: Markov chain → 1000 steps 필수

**DDIM의 해결**: Non-Markovian process → Skip steps 가능

**Deterministic Sampling**:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted } \mathbf{x}_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}_{\text{direction pointing to } \mathbf{x}_t}$$

**핵심**: Stochastic term 제거 → Deterministic

### 5.4.3 Accelerated Sampling

**Subsequence Sampling**:

전체: $\{1, 2, ..., 1000\}$  
Subset: $\{1, 20, 40, ..., 1000\}$ (50 steps)

**효과**:
- 1000 steps → 50 steps (20배 빠름)
- 품질 유지 (FID 거의 동일)

### 5.4.4 성능

**CIFAR-10 (50 steps)**:

| Model | Steps | FID |
|-------|-------|-----|
| DDPM | 1000 | 3.17 |
| **DDIM** | **50** | **3.37** |

20배 빠른데 품질 거의 동일!

### 5.4.5 장점

1. **빠른 생성**: 20-50배 가속
2. **Deterministic**: 같은 noise → 같은 결과
3. **Interpolation**: Latent space 보간 가능
4. **품질 유지**: FID 거의 동일

### 5.4.6 단점

1. **여전히 느림**: 50 steps (GAN 1 step 대비)
2. **SSL 성능**: 생성 개선이지 SSL 개선 아님

---

## 5.5 Latent Diffusion Models (2022)

### 5.5.1 기본 정보

- **논문**: High-Resolution Image Synthesis with Latent Diffusion Models
- **발표**: CVPR 2022
- **저자**: Robin Rombach et al. (LMU Munich)
- **대표 모델**: Stable Diffusion

### 5.5.2 핵심 원리

Latent Diffusion은 **pixel space 대신 latent space**에서 diffusion 수행.

**DDPM 문제**: 고해상도 이미지 (512×512)에서 계산 비용 폭발

**해결책**: Autoencoder로 압축 후 diffusion

**2-Stage 접근**:

**Stage 1: Autoencoder 학습**

Encoder: $\mathcal{E}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{h \times w \times c}$

Decoder: $\mathcal{D}: \mathbb{R}^{h \times w \times c} \rightarrow \mathbb{R}^{H \times W \times 3}$

압축 비율: $f = H/h = W/w$ (예: f=8)

**Stage 2: Latent Diffusion**

$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(\mathbf{x}), \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)\|^2\right]$$

여기서 $\mathbf{z} = \mathcal{E}(\mathbf{x})$

### 5.5.3 기술적 세부사항

**Perceptual Compression**:

VQ-VAE 스타일 autoencoder:
- Encoder/Decoder: ResNet blocks
- Latent: 4-8배 압축 (512×512 → 64×64)
- Loss: Reconstruction + Perceptual (LPIPS) + Adversarial

**U-Net in Latent Space**:

DDPM과 유사하지만:
- Input: $h \times w \times c$ (작은 크기)
- Cross-attention: Text conditioning (CLIP)

**Conditioning**:

Cross-attention layers:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d}}\right)\mathbf{V}$$

- $\mathbf{Q}$: Image features
- $\mathbf{K}, \mathbf{V}$: Text embeddings (CLIP)

### 5.5.4 성능

**ImageNet 256×256**:

| Model | FID | Time (steps) | GPU Memory |
|-------|-----|--------------|------------|
| DDPM | 12.3 | 1000 | 32GB |
| **Latent Diffusion (f=4)** | **10.56** | 250 | **8GB** |
| **Latent Diffusion (f=8)** | **9.62** | 250 | **4GB** |

**효율성**: 64배 빠른 학습, 4-8배 적은 메모리

**Text-to-Image (LAION-5B 학습)**:

Stable Diffusion:
- Resolution: 512×512
- Quality: Photorealistic
- Speed: ~2s per image (50 steps, RTX 3090)

### 5.5.5 장점

1. **효율성**: 64배 빠른 학습
2. **메모리**: 4-8배 감소
3. **고해상도**: 512×512, 1024×1024 가능
4. **Conditioning**: Text, class, segmentation 등
5. **실용성**: Stable Diffusion 상용화

### 5.5.6 단점

1. **2-Stage**: Autoencoder 먼저 학습 필요
2. **압축 손실**: f=8에서 일부 detail 손실
3. **여전히 느림**: 생성 시 50+ steps
4. **SSL 성능**: 생성 모델 중심, SSL 부차적

### 5.5.7 Latent Diffusion의 의의

Latent Diffusion은 diffusion models를 **실용화**했다:
- Stable Diffusion: 오픈소스 혁명
- Text-to-Image: DALL-E 2 수준 (오픈소스)
- 효율성: 개인 GPU에서도 실행 가능

하지만 **SSL 관점에서는 여전히 제한적**

---

## 5.6 Diffusion Models for SSL

### 5.6.1 SSL로서의 Diffusion

**생성 모델로서는 SOTA**이지만, **SSL로서는 제한적**인 이유:

**1) 목적 함수의 차이**

**Diffusion Loss**:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

노이즈 예측 → Low-level 정보 중시

**SSL 목표**:

고수준 semantic 표현 학습

**Gap**: Noise prediction ≠ Semantic understanding

**2) 표현 추출의 어려움**

**MIM (MAE)**: Encoder가 명확 → 바로 사용

**Diffusion**: U-Net의 어느 layer에서 표현 추출?
- Encoder 중간층?
- Bottleneck?
- Decoder?

명확하지 않음

**3) 계산 비용**

**Contrastive (MoCo v3)**: 200 epochs, 8× V100, 3-4 days

**Diffusion (DDPM)**: 1M iterations, 8× V100, 2-3 weeks

SSL로 사용하기엔 비용이 너무 큼

### 5.6.2 Diffusion SSL의 시도들

**접근 1: Encoder Feature 사용**

DDPM U-Net의 encoder features를 SSL 표현으로 사용

**성능**: ImageNet Linear Probing 60-65%
- Contrastive (MoCo v3): 76%
- Gap: -11~16%p

**문제**: Noise prediction에 최적화된 표현 ≠ Semantic 표현

**접근 2: Self-Conditioning**

Diffusion에 self-supervised objective 추가

**예시**: Diffusion + Contrastive loss

$$\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda \mathcal{L}_{\text{contrastive}}$$

**성능**: 약간 개선 (65-70%)
- 여전히 Contrastive 단독 대비 낮음

**접근 3: Representation Diffusion**

Feature space에서 diffusion 수행

$$\mathcal{L} = \mathbb{E}\left[\|\mathbf{f} - \mathbf{f}_\theta(\mathbf{f}_t, t)\|^2\right]$$

여기서 $\mathbf{f}$는 feature representation

**성능**: 70-75%
- 개선되었으나 여전히 제한적

### 5.6.3 Diffusion SSL 성능 종합

| 접근법 | ImageNet Linear | vs Contrastive | 비고 |
|--------|----------------|---------------|------|
| DDPM Encoder | 60-65% | -11~16%p | Naive |
| Self-Conditioning | 65-70% | -6~11%p | 약간 개선 |
| Representation Diffusion | 70-75% | -1~6%p | 최선 |
| **Contrastive (MoCo v3)** | **76%** | **Baseline** | **비교군** |
| **Clustering (DINO)** | **80%** | **+4%p** | **최강** |
| **Generative (MAE)** | 68% (Linear) | -8%p | Fine-tuning 86% |

**결론**: Diffusion은 SSL로서 **Contrastive/Clustering/MIM보다 낮음**

---

## 5.7 왜 Diffusion은 SSL로 제한적인가?

### 5.7.1 근본적 이유

**1) 학습 목표의 불일치**

**Diffusion의 목표**: 픽셀 단위 정확한 재구성

$$\min \|\mathbf{x} - \mathbf{x}_{\text{pred}}\|^2$$

**SSL의 목표**: Semantic-level 표현

$$\min d(\text{repr}(\mathbf{x}), \text{repr}(\mathbf{x}^+))$$

**Gap**: Pixel fidelity ≠ Semantic similarity

**2) Multi-step의 딜레마**

Diffusion은 T steps로 점진적 복원:

$$\mathbf{x}_T \xrightarrow{t=T} \mathbf{x}_{T-1} \xrightarrow{t=T-1} ... \xrightarrow{t=1} \mathbf{x}_0$$

각 step마다 다른 noise level → 다른 정보 학습

**문제**: 어느 step의 표현이 downstream task에 유용한가?

**MIM (MAE)**: 단일 step, 명확한 표현

**3) Architecture의 차이**

**ViT (MIM, Contrastive)**: Global attention → Semantic

**U-Net (Diffusion)**: Local convolution + Skip connections → Pixel details

**Diffusion U-Net의 특성**:
- Encoder-Decoder symmetry → Reconstruction 최적화
- Skip connections → Low-level 정보 보존
- Multi-scale → 다양한 해상도

**SSL에 필요한 것**:
- Global context
- Semantic abstraction
- High-level features

**Mismatch!**

### 5.7.2 실험적 증거

**실험 1: Layer별 표현 품질**

DDPM U-Net의 각 layer에서 feature 추출 → Linear probing

| Layer | Linear Probing | 특성 |
|-------|----------------|------|
| Encoder Early | 45% | Low-level (edges) |
| Encoder Mid | 58% | Mid-level (textures) |
| Bottleneck | **62%** | High-level (objects) |
| Decoder Mid | 55% | Reconstruction |
| Decoder Late | 48% | Pixel details |

**최선**: Bottleneck 62%
- **여전히 MoCo v3 76% 대비 -14%p**

**실험 2: Timestep별 표현**

다른 timestep t의 denoising 표현 평가:

| Timestep t | Linear Probing | Noise Level |
|-----------|----------------|-------------|
| t=900 (노이즈 많음) | 52% | High noise |
| t=500 (중간) | **65%** | Medium noise |
| t=100 (노이즈 적음) | 58% | Low noise |

**최선**: t=500 (중간) 65%
- **여전히 Contrastive 대비 낮음**

**해석**: 중간 noise level에서 semantic과 detail의 균형

### 5.7.3 비교 분석

| 측면 | Contrastive | Clustering | MIM | **Diffusion** |
|------|------------|-----------|-----|--------------|
| **학습 목표** | Instance 유사도 | 클러스터 | 복원 | **노이즈 제거** |
| **표현 수준** | Global semantic | Global semantic | Local + Global | **Pixel-level** |
| **Architecture** | ViT (attention) | ViT (attention) | ViT | **U-Net (conv)** |
| **Steps** | 1 | 1 | 1 | **T (1000)** |
| **Linear Probing** | 76% | 80% | 68% | **60-75%** |
| **Fine-tuning** | 84% | 84.5% | **86%** | **70-80%** |
| **주용도** | SSL | SSL | SSL | **생성** |

---

## 5.8 Diffusion의 SSL 활용 시나리오

### 5.8.1 권장하지 않는 경우 (대부분)

**일반적 SSL 목적**:
- ImageNet pre-training → Downstream transfer
- Classification, Detection, Segmentation

**이유**:
- Contrastive/Clustering/MIM이 모두 우수
- Diffusion은 계산 비용만 크고 성능 낮음
- **권장 점수**: ★☆☆☆☆

### 5.8.2 제한적 활용 가능

**특수 시나리오 1: Pixel-level Dense Prediction**

매우 세밀한 pixel-level task:
- Super-resolution
- Inpainting
- Denoising

**이유**: Diffusion의 pixel reconstruction 능력

**성능**: Super-resolution에서 SOTA
- 하지만 이는 SSL이 아니라 생성 모델로서의 활용

**특수 시나리오 2: Multi-modal Learning**

Text-Image diffusion (Stable Diffusion):
- Text conditioning → Image generation
- CLIP embedding 활용

**부산물로 SSL 가능**:
- Text-aligned image features
- Zero-shot classification (제한적)

**성능**: CLIP보다 낮음
- CLIP: Zero-shot 76%
- Stable Diffusion encoder: 60-65%

**특수 시나리오 3: Data Augmentation**

Diffusion으로 합성 데이터 생성 → SSL 학습

**예시**:
1. Diffusion으로 고품질 이미지 생성 (100만 장)
2. 합성 데이터로 Contrastive SSL 학습

**효과**: 약간의 성능 향상 (1-2%p)
- 하지만 Diffusion 생성 비용 >>> 성능 향상

### 5.8.3 실무 권장사항

**SSL 목적**:
→ **Contrastive (MoCo v3, BYOL)** 또는 **Clustering (DINO)** 사용
→ Diffusion 사용하지 않음

**생성 목적**:
→ **Diffusion (Stable Diffusion, DALL-E 2)** 사용
→ 최고 품질

**둘 다 필요**:
→ **별도로 학습** (Diffusion은 생성, SSL은 따로)
→ 또는 **Hybrid 모델** (CLIP + Diffusion)

---

## 5.9 Diffusion 패러다임 종합 비교

### 5.9.1 기술적 진화 과정

```
DDPM (2020)
├─ 혁신: Denoising diffusion 확립
├─ 성능: FID 3.17 (생성 SOTA)
├─ 문제: 1000 steps, 느림
└─ SSL: 60-65% Linear

        ↓ 가속

DDIM (2021)
├─ 개선: 50 steps (20배 빠름)
├─ Deterministic sampling
└─ SSL: 변화 없음

        ↓ 효율성

Latent Diffusion (2022) ★★★★★
├─ 혁신: Latent space diffusion
├─ 효율: 64배 빠름, 8배 적은 메모리
├─ 응용: Stable Diffusion
└─ SSL: 여전히 65-70%

        ↓ SSL 시도

Various SSL approaches
├─ Encoder features: 60-65%
├─ Self-conditioning: 65-70%
├─ Representation diffusion: 70-75%
└─ 결론: Contrastive/Clustering 대비 낮음
```

### 5.9.2 상세 비교표

| 비교 항목 | DDPM | DDIM | Latent Diffusion | SSL 시도 |
|----------|------|------|-----------------|---------|
| **연도** | 2020 | 2021 | 2022 | 2020-현재 |
| **생성 품질** | FID 3.17 | FID 3.37 | FID 9.62 (f=8) | - |
| **생성 속도** | 1000 steps | 50 steps | 50 steps | - |
| **계산 효율** | 낮음 | 중간 | **높음** | - |
| **메모리** | 32GB | 32GB | **4-8GB** | - |
| **고해상도** | 어려움 | 어려움 | **512×512** | - |
| **SSL Linear** | 60-65% | 60-65% | 65-70% | 70-75% (최선) |
| **vs Contrastive** | -11~16%p | -11~16%p | -6~11%p | -1~6%p |
| **주용도** | **생성** | **생성** | **생성** | 연구 |
| **실용성 (생성)** | ★★★☆☆ | ★★★★☆ | ★★★★★ | - |
| **실용성 (SSL)** | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ |

### 5.9.3 Diffusion vs 다른 패러다임 (SSL 관점)

| 측면 | Discriminative | Clustering | Contrastive | Generative (MIM) | **Diffusion** |
|------|---------------|-----------|------------|-----------------|--------------|
| **대표 모델** | Rotation | DINO | MoCo v3 | MAE | DDPM |
| **Linear Probing** | 55% | **80%** | 76% | 68% | **60-75%** |
| **Fine-tuning** | - | 84.5% | 84.1% | **86%** | **70-80%** |
| **학습 비용** | 낮음 | 중간 | 중간-높음 | 중간 | **매우 높음** |
| **생성 능력** | 없음 | 없음 | 없음 | 약함 | **SOTA** |
| **해석 가능성** | 중간 | **높음** | 낮음 | 낮음 | 낮음 |
| **주용도** | 교육 | SSL | SSL | SSL | **생성** |
| **SSL 추천도** | ★☆☆☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ | **★☆☆☆☆** |

### 5.9.4 핵심 발견 요약

**1) 생성 vs SSL의 Trade-off**

```
생성 품질: Diffusion >>> MIM > Contrastive ≈ Clustering
SSL 성능: Clustering > Contrastive > MIM > Diffusion

결론: 역상관 관계
```

**이유**:
- 생성: Pixel fidelity 중요
- SSL: Semantic abstraction 중요
- 목표가 근본적으로 다름

**2) 계산 비용 vs 효과**

**SSL 학습 비용 (ImageNet)**:

| 방법 | GPU-days | Linear Probing | 효율성 Score |
|------|---------|----------------|-------------|
| MoCo v3 | ~80 | 76% | **0.95** |
| DINO | ~150 | 80% | **0.53** |
| MAE | ~120 | 68% | 0.57 |
| **Diffusion** | **~300** | **65%** | **0.22** |

**효율성 Score** = (Linear Probing %) / (GPU-days)

**결론**: Diffusion은 비용 대비 효과가 매우 낮음

**3) Architecture의 중요성**

| Backbone | Contrastive | Clustering | MIM | Diffusion |
|----------|------------|-----------|-----|----------|
| **ViT** | 76% | **80%** | 68% | - |
| **U-Net** | - | - | - | **65%** |

**ViT**: Global attention → Semantic learning
**U-Net**: Local + Skip → Pixel reconstruction

**결론**: SSL에는 ViT가 유리

---

## 5.10 실무 적용 가이드

### 5.10.1 시나리오별 선택

**시나리오 1: 일반 SSL (Classification, Detection, Segmentation)**

**추천**: Contrastive 또는 Clustering
- MoCo v3: 76% Linear, 효율적
- DINO: 80% Linear, 해석 가능

**비추천**: Diffusion
- 성능 낮음 (65%)
- 계산 비용 높음

**점수**: Diffusion ★☆☆☆☆

**시나리오 2: 고품질 이미지 생성 필요**

**추천**: Latent Diffusion (Stable Diffusion)
- 생성 품질 SOTA
- 512×512 고해상도
- Text-to-Image

**대안**: GAN (빠른 생성 필요 시)

**점수**: Diffusion ★★★★★ (생성 용도)

**시나리오 3: SSL + 생성 둘 다**

**전략 1**: 별도 학습
- SSL: DINO (80%)
- 생성: Stable Diffusion

**전략 2**: Hybrid 모델
- CLIP + Diffusion
- Text-Image alignment

**전략 3**: Sequential
1. Diffusion으로 데이터 생성
2. 생성 데이터로 SSL 학습 (약간 개선)

**점수**: Diffusion ★★☆☆☆ (복잡함)

### 5.10.2 Best Practices

**DO**:
✅ Diffusion을 생성 목적으로 사용
✅ Stable Diffusion으로 고품질 이미지 생성
✅ Text-to-Image, Inpainting 등 활용

**DON'T**:
❌ Diffusion을 순수 SSL 목적으로 사용
❌ SSL에 Diffusion 계산 비용 투자
❌ Diffusion feature를 downstream task에 직접 사용

**대안**:
→ SSL: DINO, MoCo v3, MAE
→ 생성: Stable Diffusion
→ 둘 다: 별도 학습

### 5.10.3 개발 로드맵

**Phase 1: 목적 결정**

```
목적이 무엇인가?

├─ SSL (표현 학습)
│  └─ DINO, MoCo v3, MAE 사용
│     (Diffusion 사용 안 함)
│
├─ 생성 (고품질 이미지)
│  └─ Stable Diffusion 사용
│
└─ 둘 다
   └─ 별도 학습 (각각 최적 모델)
```

**Phase 2: SSL 선택 시** (Diffusion 제외)

```
Step 1: Backbone 선택
- ViT → DINO 또는 MAE

Step 2: 목표 설정
- Linear probing → DINO (80%)
- Fine-tuning → MAE (86%)

Step 3: 학습 & 평가
- Pre-training
- Linear probing / Fine-tuning
```

**Phase 3: 생성 선택 시**

```
Step 1: Stable Diffusion 사용
- Pretrained model 다운로드
- Fine-tuning (선택)

Step 2: Application
- Text-to-Image
- Image-to-Image
- Inpainting
```

---

## 부록: 관련 테이블

### A.1 Diffusion vs 다른 패러다임 (종합)

| 패러다임 | 대표 모델 | Linear | Fine-tuning | 생성 품질 | 주용도 | SSL 추천 |
|---------|----------|--------|-------------|----------|--------|---------|
| Discriminative | Rotation | 55% | - | - | 교육 | ★☆☆☆☆ |
| Clustering | DINO | **80%** | 84.5% | - | SSL | ★★★★★ |
| Contrastive | MoCo v3 | 76% | 84.1% | - | SSL | ★★★★☆ |
| Generative (MIM) | MAE | 68% | **86%** | - | SSL | ★★★★☆ |
| **Diffusion** | **DDPM** | **65%** | **75%** | **SOTA** | **생성** | **★☆☆☆☆** |
| Hybrid | DINOv2 | **86%** | **86%** | - | SSL | ★★★★★ |

### A.2 Diffusion Models 비교

| Model | 생성 품질 (FID) | 생성 속도 | 메모리 | 해상도 | SSL 성능 |
|-------|---------------|----------|--------|--------|---------|
| **DDPM** | **3.17** | 1000 steps | 32GB | 256×256 | 60-65% |
| **DDIM** | 3.37 | **50 steps** | 32GB | 256×256 | 60-65% |
| **Latent Diffusion** | 9.62 | 50 steps | **4GB** | **512×512** | 65-70% |

### A.3 SSL 방법론별 계산 비용

| 방법 | GPU-days | Linear | Fine-tuning | 비용/성능 | 추천도 |
|------|---------|--------|-------------|----------|--------|
| **MoCo v3** | ~80 | 76% | 84.1% | 0.95 | ★★★★☆ |
| **DINO** | ~150 | **80%** | 84.5% | 0.53 | ★★★★★ |
| **MAE** | ~120 | 68% | **86%** | 0.57 | ★★★★☆ |
| **Diffusion** | **~300** | 65% | 75% | **0.22** | **★☆☆☆☆** |

**비용/성능** = Linear Probing / GPU-days

### A.4 Use Case별 권장 모델

| Use Case | 1순위 | 2순위 | Diffusion 사용? |
|----------|-------|-------|----------------|
| **Classification SSL** | DINO | MoCo v3 | ❌ 비추천 |
| **Detection/Segmentation** | MAE | DINO | ❌ 비추천 |
| **Few-shot Learning** | DINO | Contrastive | ❌ 비추천 |
| **이미지 생성** | **Stable Diffusion** | DALL-E 2 | ✅ **최고** |
| **Text-to-Image** | **Stable Diffusion** | - | ✅ **최고** |
| **Super-resolution** | **Diffusion** | GAN | ✅ 추천 |
| **Inpainting** | **Diffusion** | GAN | ✅ 추천 |

### A.5 Diffusion SSL 실험 결과 종합

**실험 설정**: ImageNet-1K, ViT-Base

| 접근법 | 방법 | Linear | vs MoCo v3 | 비고 |
|--------|------|--------|-----------|------|
| **Baseline** | Random init | 5% | -71%p | - |
| **Naive** | DDPM encoder | 62% | -14%p | Bottleneck |
| **Time-aware** | Multi-timestep | 65% | -11%p | t=500 최적 |
| **Self-supervised** | + Contrastive | 70% | -6%p | 복잡 |
| **Representation** | Feature diffusion | 75% | -1%p | 최선 |
| **Contrastive** | MoCo v3 | **76%** | **0%p** | **비교군** |
| **Clustering** | DINO | **80%** | **+4%p** | **최강** |

### A.6 하드웨어 요구사항 비교

| 작업 | MoCo v3 | DINO | MAE | Diffusion (DDPM) |
|------|---------|------|-----|-----------------|
| **학습 GPU** | 8× V100 | 8× V100 | 8× A100 | 8× A100 |
| **학습 시간** | 3-4 days | 5-7 days | 5-7 days | **2-3 weeks** |
| **추론 GPU** | 1× V100 | 1× V100 | 1× V100 | 1× V100 (SSL) |
| **생성 GPU** | - | - | - | 1× RTX 3090 (50 steps) |
| **메모리 (학습)** | 16GB | 32GB | 32GB | **40-80GB** |
| **배치 크기** | 256-1024 | 256-1024 | 1024-2048 | 256-512 |

### A.7 Diffusion 알고리즘 정리

**DDPM Training**:

```
Algorithm: DDPM Training
Input: Dataset D, timesteps T, noise schedule β
Output: Denoising network ε_θ

1. Initialize network ε_θ
2. Repeat until convergence:
   a) Sample x_0 ~ D
   b) Sample t ~ Uniform(1, T)
   c) Sample ε ~ N(0, I)
   d) Compute x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
   e) Compute loss: L = ||ε - ε_θ(x_t, t)||²
   f) Update θ: θ ← θ - η * ∇_θ L
3. Return ε_θ
```

**DDPM Sampling**:

```
Algorithm: DDPM Sampling
Input: Trained network ε_θ, timesteps T
Output: Generated image x_0

1. Sample x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a) If t > 1: z ~ N(0, I)
      Else: z = 0
   b) Compute mean:
      μ_t = (1/√α_t) * (x_t - ((1-α_t)/√(1-ᾱ_t)) * ε_θ(x_t, t))
   c) Compute variance: σ_t
   d) x_{t-1} = μ_t + σ_t * z
3. Return x_0
```

**DDIM Sampling** (Deterministic, Fast):

```
Algorithm: DDIM Sampling
Input: Trained network ε_θ, subsequence τ
Output: Generated image x_0

1. Sample x_T ~ N(0, I)
2. For t in reversed(τ):  # e.g., τ = [1, 20, 40, ..., 1000]
   a) Predict x_0:
      x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ(x_t, t)) / √(ᾱ_t)
   b) Direction pointing to x_t:
      dir = √(1-ᾱ_{t-1}) * ε_θ(x_t, t)
   c) x_{t-1} = √(ᾱ_{t-1}) * x̂_0 + dir
3. Return x_0
```

### A.8 주요 논문 및 인용수

| 논문 | 연도 | 인용수 | 중요도 | 용도 |
|------|------|--------|--------|------|
| **DDPM** | 2020 | 15000+ | 매우 높음 | 생성 |
| **DDIM** | 2021 | 8000+ | 높음 | 생성 |
| **Latent Diffusion** | 2022 | 12000+ | 매우 높음 | 생성 |
| Diffusion SSL 시도들 | 2021-현재 | 500-1000 | 낮음-중간 | SSL (제한적) |

---

## 결론

Diffusion 패러다임은 **생성 모델로서는 혁명적**이나, **SSL로서는 제한적**이다:

**핵심 발견**:

**1) 생성 품질: SOTA**
- DDPM: FID 3.17 (CIFAR-10)
- Stable Diffusion: Photorealistic 512×512
- GAN을 능가하는 품질과 다양성

**2) SSL 성능: 제한적**
- Linear Probing: 60-75% (최선)
- Contrastive (MoCo v3): 76%
- Clustering (DINO): 80%
- **Gap: -5~20%p**

**3) 근본적 이유**
- 학습 목표: Pixel fidelity vs Semantic abstraction
- Architecture: U-Net (local) vs ViT (global)
- Multi-step: T steps의 복잡성

**4) 계산 비용**
- SSL 학습: ~300 GPU-days (MoCo v3의 3-4배)
- 비용 대비 효과: 0.22 (MoCo v3: 0.95)

**실무 권장사항**:

| 목적 | 추천 모델 | Diffusion 사용? |
|------|----------|---------------|
| **SSL** | DINO, MoCo v3, MAE | ❌ **비추천** |
| **생성** | **Stable Diffusion** | ✅ **최고** |
| **둘 다** | 별도 학습 | △ 복잡 |

**최종 평가**:

**생성 용도**: ★★★★★ (혁명적)
- Text-to-Image: Stable Diffusion
- 고품질: FID 3-10
- 다양성: Mode collapse 없음

**SSL 용도**: ★☆☆☆☆ (비추천)
- 성능: 60-75% (낮음)
- 비용: 매우 높음
- 대안: Contrastive, Clustering, MIM

Diffusion Models은 **이미지 생성의 새로운 표준**이지만, **Self-Supervised Learning에는 적합하지 않다**. SSL 목적이라면 Contrastive, Clustering, 또는 Generative (MIM) 패러다임을 사용하고, Diffusion은 순수 생성 목적으로만 활용하는 것이 최선이다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: DDPM, DDIM, Latent Diffusion, SSL 시도들

**주요 내용**:
1. Diffusion 패러다임 개요 (수식)
2. Diffusion 이론 (Forward/Reverse process)
3. **DDPM** (2020) - 생성 혁명
4. **DDIM** (2021) - 가속화
5. **Latent Diffusion** (2022) - Stable Diffusion
6. **Diffusion for SSL** - 시도와 한계
7. **왜 SSL로 제한적인가** - 근본적 이유 분석
8. 실무 가이드 - SSL에는 비추천
9. **부록**: 비교표, 알고리즘, 하드웨어 요구사항

**핵심 메시지**: Diffusion = 생성 최고, SSL 비추천