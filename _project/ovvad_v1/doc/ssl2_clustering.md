# 2. Clustering 패러다임 상세 분석

## 2.1 패러다임 개요

Clustering 방식은 특징 공간에서 **자동으로 클러스터를 형성**하고, 이를 pseudo-label로 사용하여 학습한다. Discriminative의 수작업 task 설계를 벗어나, 데이터 자체의 구조를 활용한다.

**핵심 수식**:

$$\text{Step 1 (Clustering): } \mathbf{c}_i = \text{Cluster}(\{f_\theta(\mathbf{x}_j)\}_{j=1}^N)$$

$$\text{Step 2 (Classification): } \min_\theta \mathbb{E}_{\mathbf{x}} \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}), \mathbf{c}_{\mathbf{x}})$$

여기서:
- $f_\theta$: Feature encoder
- $\mathbf{c}_i$: 클러스터 assignment (pseudo-label)
- $N$: 학습 샘플 수

**핵심 가정**: "비슷한 이미지는 특징 공간에서 클러스터를 형성하며, 이 클러스터는 의미적 카테고리에 대응한다"

**Discriminative와의 차이**:

| 측면 | Discriminative | Clustering |
|------|---------------|-----------|
| **Pseudo-label** | 수작업 설계 (회전, 퍼즐) | 자동 생성 (클러스터) |
| **Task** | 고정 (4-way, 8-way) | 데이터 적응적 (K개 클러스터) |
| **설계 필요성** | 높음 | 낮음 (K만 선택) |
| **의미성** | 간접적 | 직접적 (semantic 클러스터) |

---

## 2.2 DeepCluster (2018)

### 2.2.1 기본 정보

- **논문**: Deep Clustering for Unsupervised Learning of Visual Features
- **발표**: ECCV 2018
- **저자**: Mathilde Caron et al. (Facebook AI Research)
- **GitHub**: https://github.com/facebookresearch/deepcluster

### 2.2.2 핵심 원리

DeepCluster는 **K-means clustering과 CNN 학습을 번갈아 수행**한다.

**알고리즘**:

```
Repeat until convergence:
  1. Feature 추출: f_i = f_θ(x_i) for all images
  2. K-means clustering: c_i = argmin_k ||f_i - μ_k||²
  3. Cluster를 label로 사용하여 CNN 학습
  4. θ 업데이트
```

**수학적 정식화**:

**Clustering step**:

$$\mathbf{c}^* = \underset{\mathbf{c}}{\arg\min} \sum_{i=1}^{N} \min_{k=1}^{K} \|f_\theta(\mathbf{x}_i) - \boldsymbol{\mu}_k\|^2$$

여기서 $\boldsymbol{\mu}_k$는 k번째 centroid

**Learning step**:

$$\min_\theta \sum_{i=1}^{N} \mathcal{L}_{\text{CE}}(g_\theta(\mathbf{x}_i), c_i^*)$$

### 2.2.3 기술적 세부사항

**Architecture**:
- **Backbone**: AlexNet 또는 VGG
- **Feature dim**: 마지막 conv layer output (예: 4096-dim)
- **Clustering**: Standard K-means (scikit-learn)

**하이퍼파라미터**:
- **K (cluster 수)**: 10,000 (ImageNet)
- **Reassignment interval**: 매 epoch

**Empty Cluster 문제**:

일부 클러스터가 비어있을 수 있음:

$$|\{\mathbf{x}_i : c_i = k\}| = 0$$

**해결책**:
1. 비어있는 클러스터를 가장 큰 클러스터에서 랜덤 샘플로 초기화
2. 또는 클러스터 수 동적 조정

**Trivial Solution 문제**:

모든 이미지가 하나의 클러스터로:

$$c_i = c_j = 1, \quad \forall i, j$$

**해결책**: Reassignment (empty cluster handling)

### 2.2.4 성능

**ImageNet Linear Probing**:
- AlexNet: 48.4%
- VGG-16: 53.8%

**전통적 방법 대비**:
- Random init: ~12%
- **개선**: +36~42%p

**Supervised 대비**:
- AlexNet Supervised: ~60%
- Gap: -7~12%p

### 2.2.5 장점

1. **자동 pseudo-label**: Task 설계 불필요
2. **Semantic clustering**: 의미적으로 유사한 이미지 그룹화
3. **확장 가능**: 대규모 데이터셋 적용 가능
4. **시각화 가능**: 클러스터를 직접 볼 수 있음

### 2.2.6 단점

1. **Trivial solution**: 모든 샘플이 하나의 클러스터
2. **Empty clusters**: 일부 클러스터 사용 안 됨
3. **K-means 비용**: 매 epoch마다 전체 데이터 클러스터링
4. **K 선택**: 최적 클러스터 수 찾기 어려움
5. **성능 제한**: 53.8% (Discriminative 60-65% 대비 낮음)

### 2.2.7 DeepCluster의 의의

DeepCluster는 **"데이터 자체가 supervision을 만든다"**는 아이디어를 실현했다. Discriminative의 수작업 task에서 벗어나 자동화된 pseudo-label 생성을 보여주었다. 하지만 trivial solution과 K-means 비용이 큰 문제였다.

---

## 2.3 SwAV (2020)

### 2.3.1 기본 정보

- **논문**: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
- **발표**: NeurIPS 2020
- **저자**: Mathilde Caron et al. (Facebook AI Research)
- **GitHub**: https://github.com/facebookresearch/swav

### 2.3.2 핵심 원리

SwAV는 **online clustering**과 **swapped prediction**을 결합했다.

**DeepCluster의 문제**:
- Offline K-means (매 epoch 전체 데이터)
- 계산 비용 큼
- Batch 단위 학습 어려움

**SwAV의 해결**:
- **Online clustering**: Batch 단위로 cluster assignment
- **Sinkhorn-Knopp**: Optimal transport로 균등 분포 강제
- **Swapped prediction**: 다른 view의 cluster 예측

**수학적 정식화**:

**Cluster Assignment (Sinkhorn-Knopp)**:

주어진 feature $\mathbf{Z} = [z_1, ..., z_B]$에 대해:

$$\max_{\mathbf{Q}} \text{Tr}(\mathbf{Q}^T \mathbf{C}^T \mathbf{Z}) + \epsilon H(\mathbf{Q})$$

subject to: $\mathbf{Q} \mathbf{1} = \frac{1}{K}\mathbf{1}, \quad \mathbf{Q}^T \mathbf{1} = \frac{1}{B}\mathbf{1}$

여기서:
- $\mathbf{Q}$: Assignment matrix (B × K)
- $\mathbf{C}$: Prototypes (K × D)
- $H(\mathbf{Q})$: Entropy regularization
- 제약 조건: 균등 분포 (equipartition)

**Swapped Prediction Loss**:

같은 이미지의 두 view $\mathbf{x}_1, \mathbf{x}_2$에 대해:

$$\mathcal{L}(\mathbf{x}_1, \mathbf{x}_2) = \ell(\mathbf{z}_1, \mathbf{q}_2) + \ell(\mathbf{z}_2, \mathbf{q}_1)$$

여기서:
- $\mathbf{z}_1 = f_\theta(\mathbf{x}_1)$: View 1의 embedding
- $\mathbf{q}_2$: View 2의 cluster assignment (stop gradient)
- $\ell(\mathbf{z}, \mathbf{q}) = -\sum_{k} q_k \log p_k$, $p_k = \frac{\exp(\mathbf{z}^T \mathbf{c}_k / \tau)}{\sum_j \exp(\mathbf{z}^T \mathbf{c}_j / \tau)}$

### 2.3.3 DeepCluster 대비 핵심 차이점

| 측면 | DeepCluster | SwAV | 개선 효과 |
|------|------------|------|----------|
| **Clustering** | Offline K-means | Online Sinkhorn-Knopp | 속도 대폭 향상 |
| **Frequency** | Epoch 단위 | Batch 단위 | 실시간 업데이트 |
| **Equipartition** | 수동 처리 | 자동 제약 (optimal transport) | Trivial 방지 |
| **학습 방식** | Standard classification | Swapped prediction | 일관성 학습 |
| **Multi-crop** | 없음 | 있음 (2 global + 6 local) | 성능 향상 |
| **ImageNet** | 53.8% | **75.3%** | **+21.5%p** |

### 2.3.4 기술적 세부사항

**Sinkhorn-Knopp 알고리즘**:

Iterative algorithm for optimal transport:

```python
def sinkhorn(Q, num_iters=3):
    """
    Q: (B, K) similarity scores
    return: (B, K) assignment matrix
    """
    Q = torch.exp(Q / epsilon)
    
    for _ in range(num_iters):
        # Row normalization
        Q /= Q.sum(dim=1, keepdim=True)
        # Column normalization
        Q /= Q.sum(dim=0, keepdim=True) * (B / K)
    
    return Q
```

**수렴 보장**: 3 iteration으로 충분 (빠름!)

**Multi-Crop Strategy**:

- **2 global crops**: 224×224 (standard)
- **6 local crops**: 96×96 (smaller)

**효과**:
- Global: 전체 context
- Local: 세부 정보
- 다양한 스케일 학습

**총 Loss**:

$$\mathcal{L} = \sum_{\substack{i,j=1 \\ i \neq j}}^{8} \ell(\mathbf{z}_i, \mathbf{q}_j)$$

8개 view 간 모든 쌍 (8×7 = 56 pairs)

### 2.3.5 성능

**ImageNet Linear Probing**:

| Model | Backbone | Epochs | ImageNet |
|-------|----------|--------|----------|
| DeepCluster | ResNet-50 | 800 | 53.8% |
| **SwAV** | **ResNet-50** | **800** | **75.3%** |
| SwAV + multi-crop | ResNet-50 | 800 | **75.3%** |

**Supervised 대비**:
- Supervised (ResNet-50): 76.5%
- SwAV: 75.3%
- **Gap: -1.2%p** (거의 동등!)

**주요 발견**:
- Multi-crop이 핵심 (+3~5%p)
- Sinkhorn-Knopp로 안정적 학습
- Online clustering으로 빠른 수렴

### 2.3.6 장점

1. **SOTA급 성능**: 75.3% (Supervised 76.5%에 근접)
2. **Online clustering**: 빠른 학습, batch 단위
3. **Equipartition 보장**: Trivial solution 자동 방지
4. **Multi-crop**: 다양한 스케일 학습
5. **확장 가능**: 대규모 데이터셋

### 2.3.7 단점

1. **복잡한 구조**: Sinkhorn-Knopp, multi-crop
2. **하이퍼파라미터**: Prototypes 수 K, epsilon 등
3. **메모리 사용**: Multi-crop으로 증가
4. **여전히 gap**: Supervised 대비 -1.2%p

### 2.3.8 SwAV의 혁명

SwAV는 **Clustering 패러다임을 실용적 수준으로 끌어올렸다**:
- DeepCluster 53.8% → SwAV 75.3% (+21.5%p)
- Supervised에 근접 (-1.2%p)
- Contrastive (MoCo v2 71%) 초과

특히 **Sinkhorn-Knopp의 도입**으로 online clustering을 실현하고, trivial solution을 우아하게 해결했다.

---

## 2.4 DINO (2021)

### 2.4.1 기본 정보

- **논문**: Emerging Properties in Self-Supervised Vision Transformers
- **발표**: ICCV 2021
- **저자**: Mathilde Caron et al. (Facebook AI Research)
- **GitHub**: https://github.com/facebookresearch/dino

### 2.4.2 핵심 원리

DINO는 **Self-Distillation with NO labels**의 약자로, **Teacher-Student self-distillation**과 **implicit clustering**을 결합했다.

**핵심 아이디어**: "Teacher와 Student의 출력 분포를 일치시키면, 자연스럽게 클러스터가 형성된다"

**수학적 정식화**:

**Teacher-Student Setup**:

$$\mathbf{p}_t = \text{softmax}\left(\frac{g_{\theta_t}(\mathbf{x})}{\tau_t}\right)$$

$$\mathbf{p}_s = \text{softmax}\left(\frac{g_{\theta_s}(\mathbf{x})}{\tau_s}\right)$$

여기서:
- $g_\theta$: Projection head (K-dim output)
- $\tau_t, \tau_s$: Temperature (teacher > student)
- $\theta_t$: EMA of $\theta_s$

**Loss (Cross-Entropy)**:

$$\mathcal{L} = -\sum_{k=1}^{K} p_t^{(k)} \log p_s^{(k)}$$

**Teacher Update (EMA)**:

$$\theta_t \leftarrow m \theta_t + (1-m) \theta_s, \quad m \in [0.996, 1)$$

### 2.4.3 SwAV 대비 핵심 차이점

| 측면 | SwAV | DINO | 차이점 |
|------|------|------|--------|
| **Clustering** | Explicit (Sinkhorn-Knopp) | Implicit (softmax) | 더 단순 |
| **Prototypes** | 명시적 $\mathbf{C}$ 업데이트 | 없음 | 단순화 |
| **Teacher** | 없음 | EMA teacher | Self-distillation |
| **Temperature** | Single | Dual ($\tau_t \neq \tau_s$) | Sharpening |
| **Centering** | 없음 | 있음 (collapse 방지) | 안정성 |
| **ImageNet (ViT-S)** | - | **78.3%** | - |
| **ImageNet (ViT-B)** | 75.3% | **80.1%** | **+4.8%p** |

### 2.4.4 기술적 세부사항

**Centering Mechanism**:

Collapse 방지를 위해 teacher output을 center:

$$g_t(\mathbf{x}) \leftarrow g_t(\mathbf{x}) - \mathbf{c}$$

여기서 $\mathbf{c}$는 EMA center:

$$\mathbf{c} \leftarrow m_c \mathbf{c} + (1-m_c) \frac{1}{B}\sum_{i=1}^{B} g_{\theta_t}(\mathbf{x}_i)$$

**왜 효과적인가?**
- 모든 차원이 같은 평균값으로 쏠리는 것 방지
- Uniform distribution collapse 방지

**Temperature Sharpening**:

$$\tau_t = 0.04 \quad (\text{sharp}), \quad \tau_s = 0.1 \quad (\text{smooth})$$

**효과**:
- Teacher: Sharp distribution (confident)
- Student: Smooth distribution (학습 중)
- Student가 Teacher의 confident prediction 모방

**Multi-Crop**:

- 2 global crops (224×224)
- 10 local crops (96×96)

**Loss**:

$$\mathcal{L} = \sum_{x \in \{g_1, g_2\}} \sum_{x' \in \{g_1, g_2, l_1, ..., l_{10}\}, x' \neq x} H(p_t(x), p_s(x'))$$

Global을 teacher, 모든 view를 student로

### 2.4.5 Self-Attention Visualization의 혁명

**DINO의 놀라운 발견**:

ViT의 **self-attention map이 물체 경계를 자동으로 탐지**한다!

**Visualization**:

CLS token의 attention:

$$\text{Attn}_{\text{CLS}} = \text{softmax}\left(\frac{\mathbf{q}_{\text{CLS}} \mathbf{K}^T}{\sqrt{d}}\right)$$

**결과**:
- 사람 이미지 → 사람 영역만 highlight
- 개 이미지 → 개 영역만 highlight
- 배경은 무시

**Zero-shot Segmentation**:

Attention map을 threshold하면 segmentation mask:

$$\text{Mask} = \mathbb{1}[\text{Attn}_{\text{CLS}} > \tau]$$

**성능** (PASCAL VOC):
- Zero-shot (no segmentation label): 65% mIoU
- Supervised baseline: 70% mIoU
- **놀라운 결과**: 레이블 없이 65%!

### 2.4.6 성능

**ImageNet Linear Probing**:

| Model | Backbone | ImageNet | Supervised Gap |
|-------|----------|----------|---------------|
| SwAV | ResNet-50 | 75.3% | -1.2%p |
| **DINO** | **ViT-S/16** | **78.3%** | **+1.8%p** |
| **DINO** | **ViT-B/16** | **80.1%** | **+3.6%p** |

**ViT-B로 Supervised 초과!**

**k-NN Classification** (no linear layer):

| Model | ImageNet k-NN |
|-------|---------------|
| Supervised (ViT-B) | 74.5% |
| **DINO (ViT-B)** | **78.3%** (+3.8%p) |

**Downstream Tasks**:

- **Object Detection** (COCO): 48.2 AP
- **Semantic Segmentation** (ADE20K): 49.2 mIoU
- **Copy Detection**: 70.8% (retrieval)

### 2.4.7 장점

1. **Supervised 초과**: 80.1% (ViT-B)
2. **Self-Attention 해석**: 물체 경계 자동 탐지
3. **Zero-shot Segmentation**: 65% mIoU
4. **단순한 구조**: Explicit clustering 불필요
5. **ViT 최적화**: Transformer와 궁합
6. **k-NN 우수**: 78.3% (Supervised 74.5%)

### 2.4.8 단점

1. **ViT 의존**: CNN에서는 성능 낮음
2. **학습 시간**: 800 epochs 필요
3. **하이퍼파라미터**: Centering momentum, temperature 등
4. **메모리**: Multi-crop (12개 view)

### 2.4.9 DINO의 혁명적 영향

DINO는 **Clustering 패러다임을 새로운 수준으로** 끌어올렸다:

**1) Supervised 초과**:
- ViT-B: 80.1% vs Supervised 76.5% (+3.6%p)
- **SSL > Supervised 입증**

**2) Self-Attention의 해석 가능성**:
- 물체 경계 자동 탐지
- Zero-shot segmentation 65%
- **Explainable AI 실현**

**3) Implicit Clustering의 우아함**:
- Explicit clustering (Sinkhorn-Knopp) 불필요
- Teacher-Student로 자연스러운 클러스터 형성

**4) k-NN의 강력함**:
- Linear layer 없이 78.3%
- Feature 품질 입증

DINO는 "Self-Supervised Learning이 Supervised보다 나을 수 있다"는 것을 처음으로 명확히 보여준 모델이다.

---

## 2.5 DINOv2 (2023)

### 2.5.1 기본 정보

- **논문**: DINOv2: Learning Robust Visual Features without Supervision
- **발표**: arXiv 2023
- **저자**: Maxime Oquab et al. (Meta AI)
- **GitHub**: https://github.com/facebookresearch/dinov2

### 2.5.2 핵심 원리

DINOv2는 **DINO + Large-scale curated data**로 Foundation Model을 구축했다.

**DINO 대비 변화**:

| 측면 | DINO | DINOv2 | 개선 |
|------|------|--------|------|
| **데이터** | ImageNet-1K (1.3M) | **LVD-142M (142M)** | 100배 증가 |
| **큐레이션** | 없음 | 있음 (중복 제거, 품질 필터) | 품질 향상 |
| **모델 크기** | ViT-B (86M) | ViT-g (1.1B) | 12배 증가 |
| **학습 시간** | 1주 | 2주 (scale up) | - |
| **ImageNet** | 80.1% | **84.5%** (linear) | +4.4%p |
| **k-NN** | 78.3% | **86.3%** | +8.0%p |

### 2.5.3 Data Curation Pipeline

**LVD-142M 구축**:

1. **초기 수집**: Web crawling, 수십억 장
2. **중복 제거**: Self-similarity 기반 deduplication
3. **품질 필터링**: 
   - NSFW 제거
   - 저품질 이미지 제거
   - Concept coverage 확인
4. **최종**: 142M curated images

**큐레이션 효과**:

| 데이터 | 크기 | ImageNet |
|--------|------|----------|
| ImageNet-1K | 1.3M | 80.1% |
| Web (raw) 142M | 142M | 82.5% |
| **LVD-142M (curated)** | **142M** | **84.5%** |

**품질 > 양**: 큐레이션으로 +2%p 향상

### 2.5.4 성능

**ImageNet 벤치마크**:

| Model | Linear | k-NN | Fine-tuning |
|-------|--------|------|-------------|
| Supervised (ViT-g) | 76.5% | - | 84.5% |
| DINO (ViT-B) | 80.1% | 78.3% | 84.5% |
| **DINOv2 (ViT-L)** | **84.5%** | **85.0%** | - |
| **DINOv2 (ViT-g)** | **83.5%** | **86.3%** | - |

**k-NN 86.3%**: Fine-tuning 없이 역대 최고!

**Downstream Tasks**:

| Task | Dataset | DINOv2 | Supervised |
|------|---------|--------|------------|
| **Segmentation** | ADE20K | **51.1 mIoU** | 48.1 mIoU |
| **Depth** | NYUv2 | **0.048 error** | 0.055 error |
| **Detection** | COCO | **52.3 AP** | 51.3 AP |

**모든 downstream에서 Supervised 초과!**

### 2.5.5 장점

1. **역대 최고 SSL**: Linear 84.5%, k-NN 86.3%
2. **Dense prediction 우수**: Segmentation, Depth
3. **Zero-shot transfer**: 다양한 도메인
4. **Foundation Model**: 범용 backbone

### 2.5.6 단점

1. **계산 비용**: 142M 이미지, 2주 학습
2. **데이터 큐레이션**: 복잡한 파이프라인
3. **모델 크기**: ViT-g 1.1B parameters
4. **재현 어려움**: 일반 연구실에서 불가능

### 2.5.7 DINOv2의 의의

DINOv2는 **"Scale + Curation = Foundation Model"**을 입증했다:

**데이터의 힘**:
- 1.3M → 142M: +4.4%p
- 큐레이션: +2%p

**SSL의 최종 형태**:
- Linear 84.5% (Supervised 76.5% 대비 +8%p)
- k-NN 86.3% (Supervised fine-tuning 84.5% 초과)

DINOv2는 **Vision Foundation Model의 표준**이 되었다.

---

## 2.6 Clustering 패러다임 종합 비교

### 2.6.1 기술적 진화 과정

```
DeepCluster (2018)
├─ 혁신: K-means clustering → pseudo-label
├─ 문제: Trivial solution, offline clustering
└─ 성능: 53.8% (AlexNet)

        ↓ Online clustering

SwAV (2020)
├─ 혁신: Sinkhorn-Knopp, swapped prediction
├─ 해결: Online, equipartition 자동
├─ Multi-crop: 다양한 스케일
└─ 성능: 75.3% (Supervised 근접)

        ↓ Self-distillation

DINO (2021) ★★★★★
├─ 혁신: Teacher-Student, implicit clustering
├─ 특징: Self-attention 해석 가능
├─ 돌파: Supervised 초과 (80.1%)
└─ Zero-shot segmentation (65%)

        ↓ Large-scale

DINOv2 (2023) ★★★★★
├─ 확장: 142M curated images
├─ 모델: ViT-g (1.1B params)
├─ 성능: 84.5% linear, 86.3% k-NN
└─ Foundation Model 표준
```

### 2.6.2 상세 비교표

| 비교 항목 | DeepCluster | SwAV | DINO | DINOv2 |
|----------|------------|------|------|--------|
| **연도** | 2018 | 2020 | 2021 | 2023 |
| **Clustering** | Offline K-means | Online Sinkhorn | Implicit (softmax) | Implicit |
| **Teacher** | 없음 | 없음 | EMA | EMA |
| **Prototypes** | K-means centroids | 명시적 업데이트 | 없음 (implicit) | 없음 |
| **Equipartition** | 수동 | Sinkhorn 제약 | Centering | Centering |
| **Multi-crop** | 없음 | 2+6 | 2+10 | 2+10 |
| **데이터** | ImageNet-1K | ImageNet-1K | ImageNet-1K | **LVD-142M** |
| **Backbone** | AlexNet/VGG | ResNet-50 | ViT-S/B | ViT-L/g |
| **ImageNet Linear** | 53.8% | 75.3% | 80.1% | **84.5%** |
| **k-NN** | - | - | 78.3% | **86.3%** |
| **해석 가능성** | 낮음 | 낮음 | **높음** (attention) | **높음** |
| **Zero-shot Seg** | 없음 | 없음 | **65%** | **70%+** |
| **학습 복잡도** | 중간 | 높음 | 중간 | 매우 높음 |
| **종합 평가** | ★★☆☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ |

### 2.6.3 핵심 기술 발전

**1) Trivial Solution 해결**:

| 모델 | 방법 | 효과 |
|------|------|------|
| DeepCluster | Empty cluster reassignment | 부분적 |
| **SwAV** | **Sinkhorn-Knopp (equipartition)** | **완전 해결** |
| DINO | Centering | 우아한 해결 |

**2) Clustering 방식**:

| 모델 | 방식 | 특징 | 비용 |
|------|------|------|------|
| DeepCluster | Offline K-means | 전체 데이터 | 높음 |
| **SwAV** | **Online Sinkhorn** | **Batch 단위** | **낮음** |
| DINO | Implicit (softmax) | 자연스러움 | 매우 낮음 |

**3) 성능 향상 요인**:

**DeepCluster → SwAV (+21.5%p)**:
- Online clustering
- Equipartition 자동화
- Multi-crop

**SwAV → DINO (+4.8%p)**:
- Teacher-Student
- ViT backbone
- Implicit clustering

**DINO → DINOv2 (+4.4%p)**:
- 142M curated data
- Larger model (ViT-g)
- Better training recipe

### 2.6.4 Clustering vs Contrastive

**패러다임 비교**:

| 측면 | Clustering (DINO) | Contrastive (MoCo v3) |
|------|------------------|---------------------|
| **원리** | 클러스터 형성 + 일관성 | Instance discrimination |
| **Negative** | 불필요 | 필요 (MoCo) / 불필요 (BYOL) |
| **Linear** | **80%** | 76% |
| **해석성** | **높음** (attention map) | 낮음 |
| **Zero-shot Seg** | **65%** | 없음 |
| **Few-shot** | **75%** | 72% |
| **Backbone** | **ViT 최적** | ViT/ResNet 모두 |

**언제 Clustering을 선택하는가?**

✅ **Linear probing 중요**: 80% (Contrastive 76%)
✅ **해석 가능성 필요**: Attention map 시각화
✅ **Zero-shot segmentation**: DINO만 가능
✅ **Few-shot learning**: 75% (최고)
✅ **ViT backbone**: Transformer와 궁합

**언제 Contrastive를 선택하는가?**

✅ **ResNet backbone**: CNN에서 안정적
✅ **간단한 구현**: SimCLR, MoCo
✅ **Fine-tuning 중심**: 최종 성능 비슷

### 2.6.5 실무 적용 가이드

**DeepCluster**: ★☆☆☆☆ (사용 비추천)
- 역사적 의의만
- SwAV로 완전히 대체됨

**SwAV**: ★★★☆☆ (특수 상황)
- ResNet backbone 사용 시
- Explicit clustering 필요 시
- 일반적으로는 DINO 추천

**DINO**: ★★★★★ (최고 추천)
- **대부분의 경우 최적 선택**
- ViT backbone 사용
- Linear probing 80%
- 해석 가능성 필요
- Few-shot learning

**DINOv2**: ★★★★★ (Foundation Model)
- **최고 성능** (84.5% linear)
- Pretrained model 사용 (학습 불필요)
- 모든 downstream task
- Dense prediction 우수

**추천 워크플로우**:

```
Step 1: Pretrained DINOv2 다운로드
- ViT-L 또는 ViT-g
- 즉시 사용 가능

Step 2: Linear probing
- Frozen backbone
- 빠른 평가 (84-86%)

Step 3: Fine-tuning (선택)
- Downstream task
- 최종 성능 향상

Step 4: Visualization (선택)
- Attention map 추출
- Zero-shot segmentation
```

---

## 부록: 관련 테이블

### A.1 Clustering vs 다른 패러다임

| 패러다임 | 대표 모델 | Linear | Fine-tuning | 주요 장점 |
|---------|----------|--------|-------------|----------|
| Discriminative | Rotation | 55% | - | 교육적 |
| **Clustering** | **DINO** | **80%** | 84.5% | **해석성, Few-shot** |
| Contrastive | MoCo v3 | 76% | 84.1% | Instance |
| **Generative** | MAE | 68% | **85.9%** | Dense pred |
| Hybrid | **DINOv2** | **84.5%** | - | **SOTA** |

### A.2 성능 벤치마크 상세

**ImageNet Linear Probing 진화**:

| 연도 | 모델 | Linear | Gap vs Supervised |
|------|------|--------|-------------------|
| 2018 | DeepCluster | 53.8% | -22.7%p |
| 2020 | SwAV | 75.3% | -1.2%p |
| 2021 | **DINO (ViT-B)** | **80.1%** | **+3.6%p** |
| 2023 | **DINOv2 (ViT-L)** | **84.5%** | **+8.0%p** |

Supervised baseline: 76.5% (ResNet-50)

**k-NN Classification** (no linear layer):

| 모델 | k-NN | Linear |
|------|------|--------|
| Supervised (ViT-B) | 74.5% | 76.5% |
| DINO (ViT-B) | **78.3%** | 80.1% |
| DINOv2 (ViT-g) | **86.3%** | 83.5% |

k-NN이 Linear보다 높다!

### A.3 Zero-shot Segmentation

**PASCAL VOC (no segmentation label)**:

| 모델 | Method | mIoU |
|------|--------|------|
| Supervised | Fine-tuning | 70% |
| **DINO** | **Attention map** | **65%** |
| **DINOv2** | **Attention map** | **70%+** |

레이블 없이 Supervised 수준!

### A.4 Few-shot Learning

**Mini-ImageNet 5-way 5-shot**:

| Pre-training | Accuracy |
|--------------|----------|
| Random | 40% |
| Supervised | 65% |
| Contrastive (SimCLR) | 72% |
| **Clustering (DINO)** | **75%** |

Few-shot에서 Clustering 최강!

### A.5 개발-배포 체크리스트

**DINO 학습**:

**Phase 1: 환경 준비**
- [ ] ViT backbone 선택 (ViT-S/B)
- [ ] Pre-training 데이터 (ImageNet-1K)
- [ ] GPU 리소스 (8× V100 이상)

**Phase 2: 학습**
- [ ] Teacher temperature: 0.04
- [ ] Student temperature: 0.1
- [ ] Momentum: 0.996
- [ ] Multi-crop: 2 global + 10 local
- [ ] Epochs: 800
- [ ] Centering momentum: 0.9

**Phase 3: 평가**
- [ ] Linear probing
  - Expected: 78-80% (ViT-B)
- [ ] k-NN evaluation
  - Expected: 76-78%
- [ ] Attention visualization
  - Zero-shot segmentation

**Phase 4: Downstream**
- [ ] Task 선택
- [ ] Feature extraction (frozen)
- [ ] Linear/Fine-tuning

**DINOv2 활용**:

**빠른 시작** (추천):
- [ ] Pretrained DINOv2 다운로드
  - ViT-L: https://github.com/facebookresearch/dinov2
- [ ] Feature extraction
- [ ] Linear probing
  - Expected: 84-86%

### A.6 Attention Map 추출 코드

```python
import torch
from torchvision import transforms
from PIL import Image

# Load pretrained DINO
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), 
                        (0.229, 0.224, 0.225)),
])

img = Image.open('image.jpg')
img_tensor = transform(img).unsqueeze(0)

# Extract attention
with torch.no_grad():
    # Get attention weights
    attentions = model.get_last_selfattention(img_tensor)
    
    # CLS token attention (first token)
    # attentions shape: (1, num_heads, num_patches+1, num_patches+1)
    cls_attn = attentions[0, :, 0, 1:]  # (num_heads, num_patches)
    
    # Average over heads
    cls_attn = cls_attn.mean(dim=0)  # (num_patches,)
    
    # Reshape to spatial
    # For 224x224 image with patch_size=16: 14x14 patches
    w_featmap = img_tensor.shape[-2] // 16
    h_featmap = img_tensor.shape[-1] // 16
    
    attn_map = cls_attn.reshape(h_featmap, w_featmap)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(attn_map.numpy(), cmap='viridis')
plt.title('DINO Attention')
plt.show()

# Zero-shot segmentation
threshold = attn_map.mean() + attn_map.std()
mask = (attn_map > threshold).float()
```

### A.7 하드웨어 요구사항

| 모델 | 학습 GPU | 학습 시간 | 추론 GPU |
|------|----------|----------|----------|
| **DINO (ViT-S)** | 4× V100 | 3-4 days | 1× V100 |
| **DINO (ViT-B)** | 8× V100 | 5-7 days | 1× V100 |
| **DINOv2 (ViT-L)** | 64× A100 | 10-14 days | 1× V100 |
| **DINOv2 (ViT-g)** | 128× A100 | 14-20 days | 2× V100 |

**추론 전용 (Pretrained DINOv2)**:
- Linear probing: 1× V100 (충분)
- Feature extraction: 1× V100 또는 CPU

---

## 결론

Clustering 패러다임은 **DeepCluster에서 DINOv2까지 급격한 발전**을 이루었다:

**성능 진화**:
```
2018: DeepCluster 53.8%
2020: SwAV 75.3% (+21.5%p)
2021: DINO 80.1% (+4.8%p) - Supervised 초과
2023: DINOv2 84.5% (+4.4%p) - SOTA
```

**핵심 기여**:
1. **자동 pseudo-label**: Task 설계 불필요
2. **Trivial solution 해결**: Sinkhorn-Knopp, Centering
3. **Supervised 초과**: 80.1% → 84.5%
4. **해석 가능성**: Self-attention map
5. **Zero-shot segmentation**: 65-70%

**독특한 가치**:

| 측면 | Clustering의 강점 |
|------|------------------|
| **Linear Probing** | **80-84.5%** (최고) |
| **해석 가능성** | **Attention map** (유일) |
| **Zero-shot Seg** | **65-70%** (혁명적) |
| **Few-shot** | **75%** (최고) |
| **k-NN** | **86.3%** (놀라움) |

**최종 권장사항**:

**학습하는 경우**:
- **DINO (ViT-B)**: 80.1%, 해석성, Few-shot
- 800 epochs, 8× V100, 5-7 days

**Pretrained 사용** (추천):
- **DINOv2 (ViT-L/g)**: 84.5%, SOTA
- 즉시 사용, Linear probing만
- 모든 downstream task

Clustering은 **해석 가능한 고성능 SSL**을 실현했으며, 특히 **DINO의 attention map**과 **DINOv2의 foundation model**은 Vision AI의 새로운 표준이 되었다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: DeepCluster, SwAV, DINO, DINOv2

**주요 내용**:
1. Clustering 패러다임 개요
2. DeepCluster (2018) - K-means의 시작
3. SwAV (2020) - Online clustering 혁명
4. **DINO (2021)** - Self-distillation, Attention map
5. **DINOv2 (2023)** - Foundation Model (84.5%)
6. 종합 비교 및 실무 가이드
7. **부록**: Attention 추출, Zero-shot segmentation