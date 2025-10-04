# 3. Contrastive Learning 패러다임 상세 분석

## 3.1 패러다임 개요

Contrastive Learning은 **"유사한 샘플은 가깝게, 다른 샘플은 멀게"** 배치하는 원리로 표현을 학습한다. Instance discrimination을 핵심으로, 같은 이미지의 augmented view는 positive pair, 다른 이미지는 negative pair로 취급한다.

**핵심 수식**:

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau)}{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau) + \sum_{j=1}^{N}\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j^-)/\tau)}$$

여기서:
- $\mathbf{z}_i$: Anchor embedding
- $\mathbf{z}_i^+$: Positive (같은 이미지의 다른 view)
- $\mathbf{z}_j^-$: Negative (다른 이미지)
- $\tau$: Temperature parameter
- $\text{sim}(\cdot, \cdot)$: Cosine similarity

**핵심 가정**: "같은 이미지의 augmentation은 의미가 같고, 다른 이미지는 의미가 다르다"

**Discriminative/Clustering과의 차이**:

| 측면 | Discriminative | Clustering | Contrastive |
|------|---------------|-----------|------------|
| **Pseudo-label** | 수작업 task | 클러스터 ID | Positive/Negative |
| **Learning** | Classification | Clustering + Classification | **Metric learning** |
| **핵심** | Task 예측 | 클러스터 형성 | **거리 학습** |
| **Negative** | 불필요 | 불필요 | **필요** (대부분) |

---

## 3.2 InstDisc (Instance Discrimination, 2018)

### 3.2.1 기본 정보

- **논문**: Unsupervised Feature Learning via Non-Parametric Instance Discrimination
- **발표**: CVPR 2018
- **저자**: Zhirong Wu et al. (Chinese University of Hong Kong)
- **인용수**: 2000+회

### 3.2.2 핵심 원리

InstDisc는 **각 이미지를 독립적인 클래스**로 취급한다.

**문제 설정**:

N개 이미지 → N-way classification:

$$p(i | \mathbf{v}) = \frac{\exp(\mathbf{v}^T \mathbf{f}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{v}^T \mathbf{f}_j / \tau)}$$

여기서:
- $\mathbf{v}$: Query image의 embedding
- $\mathbf{f}_i$: i번째 이미지의 memory bank feature
- $N$: 전체 이미지 수 (ImageNet: 1.28M)

**NCE Loss (Noise Contrastive Estimation)**:

전체 N개 대신 K개만 sampling:

$$\mathcal{L} = -\mathbb{E}_{\mathbf{x}} \left[\log \frac{\exp(f(\mathbf{x})^T f(\mathbf{x}^+)/\tau)}{\exp(f(\mathbf{x})^T f(\mathbf{x}^+)/\tau) + \sum_{i=1}^{K}\exp(f(\mathbf{x})^T f(\mathbf{x}_i^-)/\tau)}\right]$$

### 3.2.3 기술적 세부사항

**Memory Bank**:

모든 이미지의 feature를 저장:

$$\mathbf{M} = [\mathbf{f}_1, \mathbf{f}_2, ..., \mathbf{f}_N] \in \mathbb{R}^{D \times N}$$

**업데이트 (Momentum)**:

$$\mathbf{f}_i \leftarrow m \mathbf{f}_i + (1-m) \mathbf{v}_i$$

여기서 $m = 0.5$ (momentum coefficient)

**Proximal Regularization**:

Feature가 너무 빠르게 변하는 것 방지:

$$\mathcal{L}_{\text{prox}} = \|\mathbf{v} - \mathbf{f}\|^2$$

**최종 Loss**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NCE}} + \lambda \mathcal{L}_{\text{prox}}$$

### 3.2.4 성능

**ImageNet Linear Probing**:
- ResNet-50: **54.0%**
- AlexNet: 50.8%

**전통적 방법 대비**:
- Random init: ~5%
- **개선**: +45~49%p

**Discriminative 대비**:
- Jigsaw (60%), Rotation (55%)
- InstDisc (54%): 비슷하거나 약간 낮음

### 3.2.5 장점

1. **Instance discrimination 개념**: Contrastive의 시작
2. **Memory bank**: 효율적 negative sampling
3. **확장 가능**: N-way classification without parameters
4. **간단한 구조**: NCE loss만

### 3.2.6 단점

1. **Memory bank 관리**: N개 feature 저장 (메모리)
2. **Momentum 업데이트**: Feature 불일치 문제
3. **성능 제한**: 54% (Jigsaw 60% 대비 낮음)
4. **Augmentation 약함**: 단순 crop, color jitter

### 3.2.7 InstDisc의 의의

InstDisc는 **Contrastive Learning의 문을 열었다**. "각 이미지를 독립적 클래스로"라는 아이디어는 이후 SimCLR, MoCo 등의 기반이 되었다. Memory bank는 MoCo의 queue로 발전했다.

---

## 3.3 CPC (Contrastive Predictive Coding, 2018)

### 3.3.1 기본 정보

- **논문**: Representation Learning with Contrastive Predictive Coding
- **발표**: arXiv 2018
- **저자**: Aaron van den Oord et al. (DeepMind)
- **인용수**: 5000+회

### 3.3.2 핵심 원리

CPC는 **미래 표현을 예측**하는 방식으로 contrastive learning을 수행한다.

**문제 설정** (이미지):

이미지를 패치로 분할 → 상단 패치로 하단 패치 예측:

$$\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T]$$

**Context**: 상단 패치들의 aggregated representation

$$\mathbf{c}_t = g_{\text{enc}}([\mathbf{x}_1, ..., \mathbf{x}_t])$$

**Prediction**: 미래 패치 $\mathbf{x}_{t+k}$ 예측

$$\hat{\mathbf{z}}_{t+k} = W_k \mathbf{c}_t$$

**InfoNCE Loss**:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(\hat{\mathbf{z}}_{t+k}^T \mathbf{z}_{t+k})}{\sum_{j}\exp(\hat{\mathbf{z}}_{t+k}^T \mathbf{z}_j)}\right]$$

### 3.3.3 기술적 세부사항

**Architecture** (Vision):

```
Input Image (256×256)
    ↓ Split into 8×8 = 64 patches
[Patch Encoder] (ResNet blocks)
    ↓
Context features: c_1, c_2, ..., c_t
    ↓ [Autoregressive model] (PixelCNN-style)
Context vector: c_t
    ↓ [Prediction] (Linear)
Predict: z_{t+1}, z_{t+2}, ...
```

**Multi-step Prediction**:

여러 미래 step 동시 예측:

$$\mathcal{L} = \sum_{k=1}^{K} \mathcal{L}_{\text{InfoNCE}}(t, t+k)$$

### 3.3.4 성능

**ImageNet Linear Probing**:
- ResNet-101: **48.7%**
- 당시 기준으로는 준수

**다른 도메인**:
- **Audio**: 73% (LibriSpeech)
- **Text**: 높은 성능

### 3.3.5 장점

1. **범용성**: Vision, Audio, Text 모두 적용
2. **Autoregressive**: 시계열 데이터에 강함
3. **Multi-step**: 여러 미래 예측
4. **이론적 기반**: Mutual information maximization

### 3.3.6 단점

1. **Vision 성능**: 48.7% (InstDisc 54% 대비 낮음)
2. **복잡한 구조**: Autoregressive model 필요
3. **패치 분할**: 이미지 특화 설계 필요
4. **Audio 편향**: Vision보다 Audio에서 더 효과적

### 3.3.7 CPC의 의의

CPC는 **InfoNCE loss를 제안**하여 Contrastive Learning의 이론적 기반을 확립했다. SimCLR, MoCo 등이 사용하는 loss의 원형이다. Vision보다는 Audio/Speech에서 더 성공적이었다.

---

## 3.4 MoCo (Momentum Contrast, 2020)

### 3.4.1 기본 정보

- **논문**: Momentum Contrast for Unsupervised Visual Representation Learning
- **발표**: CVPR 2020
- **저자**: Kaiming He et al. (Facebook AI Research)
- **GitHub**: https://github.com/facebookresearch/moco
- **인용수**: 10000+회

### 3.4.2 핵심 원리

MoCo는 **Queue + Momentum Encoder**로 InstDisc의 memory bank 문제를 해결했다.

**InstDisc의 문제**:
- Memory bank feature가 오래됨 (momentum 0.5)
- 전체 N개 저장 (메모리 큼)

**MoCo의 해결**:
- **Queue**: 최근 K개만 유지 (K=65536)
- **Momentum encoder**: 천천히 업데이트 (m=0.999)

**수학적 정식화**:

**Query Encoder** (학습):

$$\mathbf{q} = f_q(\mathbf{x}^q)$$

**Key Encoder** (momentum):

$$\mathbf{k} = f_k(\mathbf{x}^k)$$

여기서 $\theta_k$는 EMA:

$$\theta_k \leftarrow m \theta_k + (1-m) \theta_q$$

**InfoNCE Loss**:

$$\mathcal{L}_q = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau)}{\exp(\mathbf{q} \cdot \mathbf{k}_+ / \tau) + \sum_{i=0}^{K}\exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}$$

여기서:
- $\mathbf{k}_+$: Positive (같은 이미지)
- $\{\mathbf{k}_i\}_{i=0}^{K}$: Queue의 negative samples

### 3.4.3 기술적 세부사항

**Queue 관리**:

```python
# Queue: (D, K) - D=128 (feature dim), K=65536
queue = torch.randn(D, K)
queue_ptr = 0

# Training step
for batch in dataloader:
    # Encode
    q = query_encoder(x_q)  # (B, D)
    k = key_encoder(x_k)    # (B, D)
    
    # Loss
    logits = torch.mm(q, queue)  # (B, K)
    loss = cross_entropy(logits, labels)
    
    # Update queue (dequeue + enqueue)
    queue[:, queue_ptr:queue_ptr+B] = k.T
    queue_ptr = (queue_ptr + B) % K
    
    # Update key encoder (momentum)
    for param_q, param_k in zip(query_encoder.parameters(), 
                                 key_encoder.parameters()):
        param_k.data = m * param_k.data + (1 - m) * param_q.data
```

**Momentum Coefficient**:

$$m = 0.999 \quad (\text{매우 천천히 업데이트})$$

**효과**: Query와 Key encoder의 일관성 유지

**Shuffle BN**:

Batch Normalization의 "cheating" 방지:

```python
# Key는 다른 GPU에서 BN 계산
k = key_encoder(shuffle_batch(x_k))
k = unshuffle_batch(k)
```

**효과**: BN statistics로 positive 추측 불가

### 3.4.4 성능

**ImageNet Linear Probing**:

| Model | Backbone | Epochs | ImageNet |
|-------|----------|--------|----------|
| InstDisc | ResNet-50 | 200 | 54.0% |
| **MoCo v1** | **ResNet-50** | **200** | **60.6%** |

**MoCo v2 (2020)**:

추가 개선 (SimCLR tricks 적용):
- MLP projection head
- Stronger augmentation
- Cosine learning rate

**성능**: **67.5%** → **71.1%** (200 epochs)

### 3.4.5 장점

1. **효율성**: Queue (65K) << Memory bank (1.28M)
2. **일관성**: Momentum encoder로 안정적
3. **확장 가능**: Batch size 독립적
4. **성능**: 60.6% (InstDisc 54% 대비 +6.6%p)
5. **단순함**: Queue + Momentum만

### 3.4.6 단점

1. **하이퍼파라미터**: Queue size K, momentum m
2. **Shuffle BN**: 구현 복잡도 증가
3. **Negative 필요**: Queue가 negative samples
4. **여전히 gap**: Supervised 76.5% 대비 -9~16%p

### 3.4.7 MoCo의 혁명

MoCo는 **Contrastive Learning을 실용화**했다:
- InstDisc 54% → MoCo 60.6% (+6.6%p)
- Memory bank → Queue (메모리 효율)
- Momentum encoder (일관성)

특히 **dictionary as a queue** 개념은 매우 우아한 해결책이다.

---

## 3.5 SimCLR (2020)

### 3.5.1 기본 정보

- **논문**: A Simple Framework for Contrastive Learning of Visual Representations
- **발표**: ICML 2020
- **저자**: Ting Chen et al. (Google Research)
- **GitHub**: https://github.com/google-research/simclr
- **인용수**: 15000+회

### 3.5.2 핵심 원리

SimCLR은 **"Simple"**을 강조하며, large batch + strong augmentation으로 SOTA를 달성했다.

**핵심 아이디어**: "Queue 없이, batch 내 negative만으로 충분하다"

**수학적 정식화**:

**Batch 내 Contrastive**:

Batch size $N$, 각 이미지마다 2개 view → $2N$ samples:

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

여기서:
- $(i, j)$: Positive pair (같은 이미지의 다른 view)
- $2N-2$: Negative samples (다른 이미지들)

**최종 Loss** (symmetric):

$$\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} [\mathcal{L}_{2k-1, 2k} + \mathcal{L}_{2k, 2k-1}]$$

### 3.5.3 기술적 세부사항

**4가지 핵심 요소**:

**1) Composition of Data Augmentation**

매우 강한 augmentation:

```python
augmentation = [
    RandomResizedCrop(224, scale=(0.08, 1.0)),
    RandomHorizontalFlip(),
    ColorJitter(0.8, 0.8, 0.8, 0.2),  # 강함!
    RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=23)       # 중요!
]
```

**효과**: Color jitter + Blur가 핵심 (+10%p)

**2) MLP Projection Head**

$$\mathbf{h} = f(\mathbf{x}) \quad (\text{Encoder})$$
$$\mathbf{z} = g(\mathbf{h}) = W^{(2)} \sigma(W^{(1)} \mathbf{h}) \quad (\text{MLP})$$

**Architecture**: 2-layer MLP (2048-d hidden, 128-d output)

**효과**: Representation $\mathbf{h}$ 보호, $\mathbf{z}$로 contrastive

**3) Large Batch Size**

$$N = 4096 \quad (\text{또는 } 8192)$$

**효과**: 더 많은 negative samples → 성능 향상

**Batch size 영향**:

| Batch Size | ImageNet |
|-----------|----------|
| 256 | 61% |
| 1024 | 66% |
| 4096 | **70%** |
| 8192 | 70.1% |

**4) Temperature**

$$\tau = 0.5 \quad (\text{또는 } 0.1)$$

낮은 temperature → Sharp distribution

### 3.5.4 성능

**ImageNet Linear Probing**:

| Model | Backbone | Batch | Epochs | ImageNet |
|-------|----------|-------|--------|----------|
| MoCo v1 | ResNet-50 | 256 | 200 | 60.6% |
| **SimCLR** | **ResNet-50** | **4096** | **1000** | **69.3%** |
| SimCLR | ResNet-50×2 | 4096 | 1000 | **74.2%** |

**Semi-supervised** (1% labels):

SimCLR pre-training + 1% ImageNet labels: **73.9%**
- Supervised (100% labels): 76.5%
- **놀라운 결과**: 1%로 Supervised의 96.6%

### 3.5.5 장점

1. **단순함**: Queue, momentum 불필요
2. **SOTA**: 69.3% (MoCo 60.6% 대비 +8.7%p)
3. **강한 augmentation**: Color jitter + Blur
4. **MLP head**: 표현 보호
5. **Semi-supervised**: 1% labels로 73.9%

### 3.5.6 단점

1. **Large batch 필수**: 4096 (128× V100 TPU)
2. **긴 학습**: 1000 epochs
3. **계산 비용**: 매우 큼
4. **Batch 내 negative**: 같은 batch에 유사 이미지 있으면 문제
5. **여전히 gap**: Supervised 대비 -7.2%p

### 3.5.7 SimCLR의 혁명

SimCLR은 **"Simple is powerful"**을 입증했다:
- Queue 없이 batch만으로
- Strong augmentation의 중요성
- MLP projection head

특히 **Gaussian Blur**와 **Color Jitter**의 조합은 이후 모든 Contrastive 방법의 표준이 되었다.

---

## 3.6 BYOL (Bootstrap Your Own Latent, 2020)

### 3.6.1 기본 정보

- **논문**: Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
- **발표**: NeurIPS 2020
- **저자**: Jean-Bastien Grill et al. (DeepMind)
- **GitHub**: https://github.com/deepmind/deepmind-research/tree/master/byol

### 3.6.2 핵심 원리

BYOL의 혁명: **Negative samples 없이도 학습 가능!**

**핵심 아이디어**: "Online network가 Target network를 예측하면, collapse 없이 학습된다"

**수학적 정식화**:

**Online Network**:

$$\mathbf{y}_\theta = q_\theta(z_\theta(f_\theta(\mathbf{x})))$$

여기서:
- $f_\theta$: Encoder
- $z_\theta$: Projector (MLP)
- $q_\theta$: **Predictor (MLP, BYOL만의 특징)**

**Target Network**:

$$\mathbf{y}'_\xi = z_\xi(f_\xi(\mathbf{x}'))$$

여기서 $\xi$는 EMA:

$$\xi \leftarrow \tau \xi + (1-\tau) \theta$$

**Loss** (MSE, symmetric):

$$\mathcal{L}_\theta = \|\bar{\mathbf{y}}_\theta - \bar{\mathbf{y}}'_\xi\|_2^2$$

여기서 $\bar{\mathbf{y}} = \mathbf{y} / \|\mathbf{y}\|_2$ (L2 normalization)

**Symmetric Loss**:

$$\mathcal{L} = \|\bar{\mathbf{y}}_\theta - \bar{\mathbf{y}}'_\xi\|_2^2 + \|\bar{\mathbf{y}}'_\theta - \bar{\mathbf{y}}_\xi\|_2^2$$

### 3.6.3 기술적 세부사항

**Architecture**:

```
Online Network:
Encoder f_θ (ResNet-50)
    ↓
Projector z_θ (MLP: 4096-256)
    ↓
Predictor q_θ (MLP: 256-256) ← BYOL만 있음!
    ↓
y_θ

Target Network (EMA):
Encoder f_ξ (ResNet-50)
    ↓
Projector z_ξ (MLP: 4096-256)
    ↓
y'_ξ
```

**Predictor가 핵심**:

Negative 없이도 collapse 방지:

$$q_\theta: \mathbb{R}^{256} \rightarrow \mathbb{R}^{256}$$

**Stop Gradient**:

Target에는 gradient 전파 안 함:

$$\mathcal{L} = \|\bar{\mathbf{y}}_\theta - \text{sg}(\bar{\mathbf{y}}'_\xi)\|_2^2$$

### 3.6.4 왜 Collapse하지 않는가?

**직관적 설명**:

1. **Predictor**: Online이 Target을 모방하도록 학습
2. **Stop gradient**: Target은 천천히 변함 (EMA)
3. **Asymmetry**: Online ≠ Target (predictor 때문)
4. **Bootstrap**: Target이 더 나은 표현 제공

**이론적 설명** (논란):

BYOL 논문에서는 명확한 이론 없음. 후속 연구:
- Batch Normalization이 중요 (implicit negative)
- Augmentation diversity
- EMA의 regularization 효과

### 3.6.5 성능

**ImageNet Linear Probing**:

| Model | Backbone | Negative | ImageNet |
|-------|----------|----------|----------|
| MoCo v2 | ResNet-50 | Yes (Queue) | 71.1% |
| SimCLR | ResNet-50 | Yes (Batch) | 69.3% |
| **BYOL** | **ResNet-50** | **No** | **74.3%** |

**Negative 없이 최고 성능!**

**Transfer Learning**:

- **ImageNet Semi-supervised** (1%): 75.8%
- **COCO Detection**: 49.8 AP
- **Segmentation**: 76.2 mIoU (PASCAL VOC)

### 3.6.6 장점

1. **No negative**: Queue, large batch 불필요
2. **SOTA**: 74.3% (당시 최고)
3. **안정적**: Collapse 방지
4. **간단한 구현**: Predictor 추가만
5. **Transfer 우수**: Downstream task 강함

### 3.6.7 단점

1. **Predictor 필요**: 추가 network
2. **이론 부족**: 왜 작동하는지 불명확
3. **BN 의존**: Batch Normalization 필수 (논란)
4. **EMA 튜닝**: Target momentum 중요
5. **여전히 gap**: Supervised 76.5% 대비 -2.2%p

### 3.6.8 BYOL의 혁명

BYOL은 **"Negative가 필수가 아니다"**를 증명했다:
- Contrastive의 패러다임 전환
- Predictor의 중요성 발견
- 74.3% (SimCLR 69.3% 대비 +5%p)

하지만 "왜 작동하는가?"는 여전히 연구 주제이다.

---

## 3.7 SwAV vs MoCo vs SimCLR vs BYOL 비교

### 3.7.1 4대 방법론 비교

| 측면 | MoCo | SimCLR | BYOL | SwAV |
|------|------|--------|------|------|
| **발표** | CVPR 2020 | ICML 2020 | NeurIPS 2020 | NeurIPS 2020 |
| **저자** | Facebook | Google | DeepMind | Facebook |
| **핵심** | Queue | Large batch | No negative | Clustering |
| **Negative** | Queue (65K) | Batch (8K) | **없음** | 없음 |
| **특수 구조** | Momentum encoder | MLP head | **Predictor** | Prototypes |
| **Batch size** | 256 | **4096-8192** | 256-512 | 256 |
| **ImageNet** | 71.1% | 69.3% | **74.3%** | 75.3% |
| **계산 비용** | 중간 | **매우 높음** | 중간 | 중간 |
| **구현 난이도** | 중간 | 낮음 | 낮음 | 높음 |

### 3.7.2 Negative Sampling 전략

**MoCo**: Queue (dictionary)
- 장점: 메모리 효율, batch 독립
- 단점: Queue 관리, momentum tuning

**SimCLR**: Batch 내
- 장점: 단순, queue 불필요
- 단점: Large batch 필수

**BYOL**: 없음
- 장점: Negative 불필요
- 단점: Predictor 필요, 이론 불명확

**SwAV**: 없음 (Clustering)
- 장점: Semantic clustering
- 단점: Prototypes 관리

### 3.7.3 성능 vs 비용

**효율성** (같은 epochs 기준):

```
SimCLR: 69.3%, 128× TPU, 1000 epochs
MoCo v2: 71.1%, 8× V100, 200 epochs ← 가장 효율적
BYOL: 74.3%, 32× V100, 1000 epochs
SwAV: 75.3%, 64× V100, 800 epochs
```

**Cost-Performance**:

| Model | ImageNet | GPU-days | Score (성능/비용) |
|-------|----------|----------|------------------|
| **MoCo v2** | 71.1% | ~80 | **0.89** ← 최고 |
| SimCLR | 69.3% | ~500 | 0.14 |
| BYOL | 74.3% | ~320 | 0.23 |
| SwAV | 75.3% | ~512 | 0.15 |

### 3.7.4 실무 선택 가이드

**시나리오별 추천**:

**제한된 GPU (4-8× V100)**:
→ **MoCo v2** (71.1%, 효율적)

**TPU 사용 가능**:
→ **SimCLR** (69.3%, 단순)

**최고 성능**:
→ **BYOL** (74.3%) 또는 **SwAV** (75.3%)

**해석 가능성**:
→ **SwAV** (Clustering)

---

## 3.8 MoCo v3 (2021)

### 3.8.1 기본 정보

- **논문**: An Empirical Study of Training Self-Supervised Vision Transformers
- **발표**: ICCV 2021
- **저자**: Xinlei Chen et al. (Facebook AI Research)

### 3.8.2 핵심 원리

MoCo v3는 **ViT (Vision Transformer)에 MoCo 적용** + instability 해결.

**MoCo v2 → v3 변화**:
- Backbone: ResNet → **ViT**
- Projection head: 2-layer → **3-layer MLP**
- Batch size: 256 → **4096**
- Predictor: 없음 → **추가** (BYOL-style)

**수학적 정식화**:

MoCo v2와 유사하지만:

$$\mathbf{q} = \text{pred}(g_\theta(f_\theta(\mathbf{x}^q)))$$
$$\mathbf{k} = g_\xi(f_\xi(\mathbf{x}^k))$$

**Predictor 추가** (BYOL에서 차용)

### 3.8.3 ViT 학습의 불안정성

**문제**: ViT로 MoCo 학습 시 **학습 붕괴** (training collapse)

**현상**:
- Loss가 갑자기 spike
- Accuracy 급락
- Gradient exploding

**원인**: ViT의 self-attention이 불안정

**해결책**:

**1) Patch Projection Freezing**:

첫 layer (patch embedding) 고정:

```python
for param in vit.patch_embed.parameters():
    param.requires_grad = False
```

**2) Random Patch Projection**:

Patch projection을 random으로 초기화 후 고정

**효과**: 학습 안정화

### 3.8.4 성능

**ImageNet Linear Probing**:

| Model | Backbone | ImageNet | vs Supervised |
|-------|----------|----------|---------------|
| MoCo v2 | ResNet-50 | 71.1% | -5.4%p |
| **MoCo v3** | **ViT-B/16** | **76.7%** | **+0.2%p** |

**Supervised 초과!**

**Fine-tuning**:

| Model | ImageNet Fine-tuning |
|-------|---------------------|
| Supervised (ViT-B) | 84.5% |
| **MoCo v3 (ViT-B)** | **84.1%** (-0.4%p) |

거의 동등!

### 3.8.5 장점

1. **ViT 성공**: Transformer에 Contrastive 적용
2. **Supervised 동등**: 76.7% vs 76.5%
3. **안정적 학습**: Patch freezing
4. **MoCo 장점 유지**: 효율성

### 3.8.6 단점

1. **Patch freezing 필요**: 추가 trick
2. **Large batch**: 4096 (MoCo v2 대비 증가)
3. **여전히 BYOL 대비 낮음**: 74.3% (ResNet) vs 76.7% (ViT)

### 3.8.7 MoCo v3의 의의

MoCo v3는 **Contrastive Learning을 Transformer 시대로** 이끌었다:
- ViT + Contrastive 성공
- Supervised와 동등
- ViT 학습 불안정성 해결

---

## 3.9 Contrastive 패러다임 종합 비교

### 3.9.1 기술적 진화 과정

```
InstDisc (2018)
├─ 혁신: Instance discrimination
├─ Memory bank (1.28M features)
└─ 성능: 54.0%

        ↓ Queue

MoCo (2020)
├─ Queue (65K) + Momentum encoder
├─ 효율성 대폭 향상
└─ 성능: 60.6% → 71.1% (v2)

        ↓ Large batch

SimCLR (2020)
├─ Batch 내 negative (4096)
├─ Strong augmentation (Color + Blur)
└─ 성능: 69.3%

        ↓ No negative

BYOL (2020) ★★★★★
├─ Negative 불필요!
├─ Predictor의 발견
└─ 성능: 74.3%

        ↓ ViT

MoCo v3 (2021)
├─ Vision Transformer 적용
├─ Patch freezing
└─ 성능: 76.7% (Supervised 초과!)
```

### 3.9.2 상세 비교표

| 비교 항목 | InstDisc | MoCo v2 | SimCLR | BYOL | MoCo v3 |
|----------|---------|---------|--------|------|---------|
| **연도** | 2018 | 2020 | 2020 | 2020 | 2021 |
| **Negative** | Memory bank | Queue | Batch | **없음** | Queue |
| **특수 구조** | - | Momentum | MLP head | **Predictor** | Predictor |
| **Batch size** | 256 | 256 | **4096** | 512 | 4096 |
| **Backbone** | ResNet | ResNet | ResNet | ResNet | **ViT** |
| **ImageNet** | 54.0% | 71.1% | 69.3% | **74.3%** | **76.7%** |
| **vs Supervised** | -22.5%p | -5.4%p | -7.2%p | -2.2%p | **+0.2%p** |
| **GPU 요구** | 낮음 | 중간 | **매우 높음** | 중간 | 높음 |
| **효율성** | 중간 | **높음** | 낮음 | 중간 | 중간 |
| **종합 평가** | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |

### 3.9.3 핵심 기술 발전

**1) Negative Sampling 진화**:

```
InstDisc (2018): Memory bank (전체 N개)
    ↓ 메모리 문제
MoCo (2020): Queue (최근 K개)
    ↓ Batch 크기 제약
SimCLR (2020): Batch 내 (2N-2개)
    ↓ 패러다임 전환
BYOL (2020): 없음!
```

**2) 성능 향상 요인**:

**InstDisc → MoCo (+17.1%p)**:
- Queue로 최신 feature 유지
- Momentum encoder 일관성

**MoCo → SimCLR (-1.8%p, 다른 장점)**:
- Strong augmentation
- MLP projection head
- 단순한 구조

**SimCLR → BYOL (+5.0%p)**:
- Negative 제거
- Predictor 도입
- 안정적 학습

**BYOL → MoCo v3 (+2.4%p)**:
- ViT backbone
- Supervised 초과

### 3.9.4 Contrastive vs Clustering vs Generative

| 측면 | Contrastive (MoCo v3) | Clustering (DINO) | Generative (MAE) |
|------|---------------------|------------------|-----------------|
| **원리** | Instance discrimination | 클러스터 형성 | MIM |
| **Linear** | **76.7%** | 80.1% | 68% |
| **Fine-tuning** | 84.1% | 84.5% | **85.9%** |
| **해석성** | 낮음 | **높음** (attention) | 낮음 |
| **Few-shot** | 72% | **75%** | 낮음 |
| **Negative** | 필요 (일부) | **불필요** | **불필요** |
| **계산 비용** | 중간-높음 | 중간 | **낮음** |

**언제 Contrastive를 선택하는가?**

✅ **Instance recognition 중요**
✅ **ResNet backbone** (ViT는 DINO 추천)
✅ **Supervised와 동등** 필요
✅ **확립된 방법론** (MoCo, SimCLR)

**언제 다른 패러다임을 선택하는가?**

→ **Clustering (DINO)**: 해석성, Few-shot, Linear probing 최강
→ **Generative (MAE)**: Fine-tuning 최강, Dense prediction

### 3.9.5 실무 적용 가이드

**InstDisc**: ★☆☆☆☆ (사용 비추천)
- 역사적 의의만
- MoCo로 완전 대체

**MoCo v2**: ★★★★☆ (ResNet 사용 시)
- **효율성**: 8× V100, 200 epochs
- **성능**: 71.1%
- **안정적**: 많은 코드베이스

**SimCLR**: ★★★☆☆ (TPU 사용 시)
- 단순한 구조
- Large batch 필수
- 교육 목적으로 좋음

**BYOL**: ★★★★☆ (No negative 원할 때)
- Negative 불필요
- 74.3% 성능
- Predictor 추가만

**MoCo v3**: ★★★★★ (ViT 사용 시)
- **Supervised와 동등** (76.7%)
- ViT backbone
- Contrastive 최고

**추천 워크플로우**:

```
Step 1: Backbone 선택
- ResNet → MoCo v2
- ViT → MoCo v3 또는 DINO

Step 2: GPU 고려
- 제한적 (4-8× V100) → MoCo v2
- 충분 (32+× V100) → BYOL 또는 MoCo v3

Step 3: 목표 설정
- Linear probing → DINO (80%)
- Supervised 동등 → MoCo v3 (76.7%)
- 효율성 → MoCo v2

Step 4: 학습 & 평가
- Pre-training (200-1000 epochs)
- Linear probing
- Fine-tuning (선택)
```

---

## 부록: 관련 테이블 및 코드

### A.1 Contrastive vs 다른 패러다임

| 패러다임 | 대표 모델 | Linear | Fine-tuning | 주요 장점 | 주요 단점 |
|---------|----------|--------|-------------|----------|----------|
| Discriminative | Rotation | 55% | - | 교육적 | 낮은 성능 |
| Clustering | DINO | **80.1%** | 84.5% | **해석성, Few-shot** | ViT 의존 |
| **Contrastive** | **MoCo v3** | **76.7%** | 84.1% | **Supervised 동등** | Large batch |
| Generative | MAE | 68% | **85.9%** | **Fine-tuning 최강** | Linear 약함 |
| Hybrid | DINOv2 | **84.5%** | - | **SOTA** | 계산 비용 |

### A.2 성능 벤치마크 상세

**ImageNet Linear Probing 진화**:

| 연도 | 모델 | Linear | Gap vs Supervised |
|------|------|--------|-------------------|
| 2018 | InstDisc | 54.0% | -22.5%p |
| 2020 | MoCo v2 | 71.1% | -5.4%p |
| 2020 | SimCLR | 69.3% | -7.2%p |
| 2020 | BYOL | 74.3% | -2.2%p |
| 2021 | **MoCo v3** | **76.7%** | **+0.2%p** |

Supervised baseline: 76.5% (ResNet-50)

**Transfer Learning** (COCO Detection):

| Pre-training | AP^bbox | AP^mask |
|--------------|---------|---------|
| Random | 31.0 | 28.5 |
| Supervised | 40.6 | 36.8 |
| MoCo v2 | 40.9 | 37.0 |
| BYOL | **41.2** | **37.3** |
| DINO | 42.1 | 38.2 |

### A.3 하드웨어 요구사항

| 모델 | 학습 GPU | 학습 시간 | 메모리 | 추론 GPU |
|------|----------|----------|--------|----------|
| **MoCo v2** | 8× V100 | 3-4 days | 16GB | 1× V100 |
| SimCLR | 128× TPU | 7-10 days | - | 1× V100 |
| BYOL | 32× V100 | 10-14 days | 32GB | 1× V100 |
| **MoCo v3** | 32× V100 | 5-7 days | 32GB | 1× V100 |

### A.4 MoCo v2 구현 (PyTorch)

```python
import torch
import torch.nn as nn

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size (default: 65536)
        m: momentum coefficient (default: 0.999)
        T: temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Create encoders
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        
        # Initialize key encoder with query encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Create queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        
        # Replace oldest batch in queue
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Shuffle for Shuffle BN"""
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        
        num_gpus = batch_size_all // batch_size_this
        
        # Random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        
        # Broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)
        
        # Index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)
        
        # Shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo shuffle"""
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        
        num_gpus = batch_size_all // batch_size_this
        
        # Restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        
        return x_gather[idx_this]
    
    def forward(self, im_q, im_k):
        """
        Input:
            im_q: query images
            im_k: key images
        Output:
            logits, targets
        """
        
        # Query embeddings
        q = self.encoder_q(im_q)  # (B, dim)
        q = nn.functional.normalize(q, dim=1)
        
        # Key embeddings
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            # Shuffle for Shuffle BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            
            # Undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        
        # Positive logits: (B, 1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: (B, K)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: (B, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.T
        
        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels


# Helper function
@torch.no_grad()
def concat_all_gather(tensor):
    """Gather tensors from all gpus"""
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


# Training loop
def train_moco(train_loader, model, criterion, optimizer, epoch):
    model.train()
    
    for i, (images, _) in enumerate(train_loader):
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)
        
        # Compute output
        logits, labels = model(im_q=images[0], im_k=images[1])
        loss = criterion(logits, labels)
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### A.5 실험 체크리스트

**MoCo v2 학습**:

**Phase 1: 환경 준비**
- [ ] ResNet-50 backbone
- [ ] ImageNet-1K 데이터
- [ ] 8× V100 GPU (최소)

**Phase 2: 하이퍼파라미터**
- [ ] Queue size: 65536
- [ ] Momentum: 0.999
- [ ] Temperature: 0.07
- [ ] Batch size: 256
- [ ] Epochs: 200
- [ ] Learning rate: 0.03 (cosine decay)

**Phase 3: 학습**
- [ ] Shuffle BN 구현
- [ ] Queue 관리
- [ ] Momentum 업데이트

**Phase 4: 평가**
- [ ] Linear probing
  - Expected: 71.1%
- [ ] Transfer learning (COCO)
  - Expected: 40.9 AP

### A.6 주요 논문 및 인용수

| 논문 | 연도 | 인용수 | 중요도 |
|------|------|--------|--------|
| **InstDisc** | 2018 | 2000+ | 역사적 |
| **CPC** | 2018 | 5000+ | 이론적 |
| **MoCo** | 2020 | 10000+ | 매우 높음 |
| **SimCLR** | 2020 | 15000+ | 매우 높음 |
| **BYOL** | 2020 | 8000+ | 혁명적 |
| **MoCo v3** | 2021 | 3000+ | 높음 |

---

## 결론

Contrastive Learning 패러다임은 **InstDisc에서 MoCo v3까지 급격한 발전**을 이루었다:

**성능 진화**:
```
2018: InstDisc 54.0%
2020: MoCo v2 71.1% (+17.1%p)
2020: SimCLR 69.3%
2020: BYOL 74.3% (+3.2%p, No negative!)
2021: MoCo v3 76.7% (+2.4%p, Supervised 초과!)
```

**핵심 기여**:
1. **Instance discrimination**: 각 이미지를 독립 클래스로
2. **효율적 negative sampling**: Queue (MoCo), Batch (SimCLR)
3. **No negative 발견**: BYOL의 혁명
4. **Supervised 동등/초과**: MoCo v3 76.7%
5. **ViT 적응**: Transformer 시대로

**독특한 가치**:

| 측면 | Contrastive의 강점 |
|------|------------------|
| **Linear Probing** | 76.7% (Supervised 동등) |
| **Instance Recognition** | 최강 |
| **효율성** | MoCo v2 (8× V100, 200 epochs) |
| **확립된 방법론** | 많은 코드베이스, 안정적 |
| **ResNet 최적화** | CNN에서 검증됨 |

**최종 권장사항**:

**ResNet 사용**:
- **MoCo v2**: 71.1%, 효율적, 안정적
- 8× V100, 200 epochs, 3-4 days

**ViT 사용**:
- **MoCo v3**: 76.7%, Supervised 동등
- **또는 DINO**: 80.1%, 해석성, Few-shot 최강

**No negative 선호**:
- **BYOL**: 74.3%, Predictor만 추가

**교육/연구**:
- **SimCLR**: 가장 단순, 이해하기 쉬움

Contrastive Learning은 **Instance discrimination을 통한 고성능 SSL**을 실현했으며, 특히 **MoCo의 효율성**과 **BYOL의 "No negative" 발견**은 Self-Supervised Learning의 중요한 이정표이다. MoCo v3로 **Supervised와 동등**한 수준에 도달하여, SSL이 실용적 대안임을 증명했다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: InstDisc, CPC, MoCo (v1/v2/v3), SimCLR, BYOL

**주요 내용**:
1. Contrastive Learning 패러다임 개요
2. InstDisc (2018) - Instance discrimination 시작
3. CPC (2018) - InfoNCE loss 이론
4. **MoCo (2020)** - Queue + Momentum encoder
5. **SimCLR (2020)** - Large batch + Strong augmentation
6. **BYOL (2020)** - No negative 혁명
7. **MoCo v3 (2021)** - ViT + Supervised 초과
8. 종합 비교 및 실무 가이드
9. **부록**: MoCo v2 구현, 하드웨어 요구사항