# 4. Generative (Masked Image Modeling) 패러다임 상세 분석

## 4.1 패러다임 개요

Generative 방식, 특히 **Masked Image Modeling (MIM)**은 이미지의 일부를 마스킹하고 나머지 부분으로부터 마스킹된 영역을 복원하는 과정에서 표현을 학습한다. NLP의 BERT에서 영감을 받아 Vision Transformer 시대에 가장 강력한 SSL 방법 중 하나로 자리잡았다.

**핵심 수식**:

$$\min_\theta \mathbb{E}_{\mathbf{x}} \mathcal{L}(\mathbf{x}_{\text{masked}}, f_\theta(\mathbf{x}_{\text{visible}}))$$

여기서:
- $\mathbf{x}$: 원본 이미지
- $\mathbf{x}_{\text{visible}}$: 보이는 패치들 (예: 25%)
- $\mathbf{x}_{\text{masked}}$: 마스킹된 패치들 (예: 75%)
- $f_\theta$: Encoder-Decoder 네트워크
- $\mathcal{L}$: Reconstruction loss (MSE, Cross-Entropy 등)

**핵심 가정**: "이미지의 일부에서 전체를 복원하려면 의미적 이해와 구조적 지식이 필요하다"

**Contrastive/Clustering과의 차이**:

| 측면 | Contrastive | Clustering | Generative |
|------|------------|-----------|-----------|
| **학습 목표** | 유사도 학습 | 클러스터 형성 | **복원** |
| **Supervision** | Positive/Negative | 클러스터 ID | **자기 자신** |
| **핵심** | Metric learning | Clustering | **Reconstruction** |
| **학습 신호** | Instance pairs | Pseudo-labels | **Pixels/Tokens** |

---

## 4.2 Auto-encoder / VAE (전통적 접근)

### 4.2.1 기본 Auto-encoder

**구조**:

$$\mathbf{z} = \text{Encoder}(\mathbf{x})$$
$$\hat{\mathbf{x}} = \text{Decoder}(\mathbf{z})$$

**Loss**:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

**한계**:
- **단순 픽셀 복사**: 의미적 표현 학습 부족
- **Low-level 정보에 집중**: 텍스처, 색상만 복원
- **Shortcut learning**: Encoder가 압축 없이 정보 전달

**성능**: ImageNet Linear Probing 50-60%

### 4.2.2 VAE (Variational Auto-encoder)

**확률적 정식화**:

$$q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{x}), \boldsymbol{\sigma}_\theta^2(\mathbf{x}))$$

**ELBO (Evidence Lower BOund)**:

$$\mathcal{L}_{\text{VAE}} = \underbrace{\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization}}$$

**개선점**:
- Latent space 정규화
- 연속적이고 smooth한 표현
- 생성 능력

**한계**:
- 여전히 픽셀 복원에 집중
- Blurry reconstruction (MSE loss)

**성능**: ImageNet Linear Probing 55-65%

### 4.2.3 전통적 방법의 근본적 한계

**문제 1: Pixel-level 복원의 함정**

픽셀 복원 loss:
$$\mathcal{L} = \sum_{i,j} (\mathbf{x}_{i,j} - \hat{\mathbf{x}}_{i,j})^2$$

이는 high-frequency details (텍스처, 노이즈)를 중시하고, semantic 정보는 경시한다.

**문제 2: Encoder-Decoder 정보 흐름**

전통적 AE는 모든 정보를 latent $\mathbf{z}$에 압축해야 하므로:
- Decoder가 너무 강력하면: Encoder 무시
- Decoder가 너무 약하면: 복원 실패

**문제 3: Pre-training과 Downstream의 괴리**

- Pre-training: 픽셀 복원
- Downstream (Classification): Semantic 이해
- Gap이 크다

---

## 4.3 BEiT (2021)

### 4.3.1 기본 정보

- **논문**: BERT Pre-Training of Image Transformers
- **발표**: ICLR 2022
- **저자**: Hangbo Bao et al. (Microsoft Research)
- **GitHub**: https://github.com/microsoft/unilm/tree/master/beit

### 4.3.2 핵심 원리

BEiT는 **BERT의 Masked Language Modeling을 Vision에 적용**하되, 연속적인 픽셀 대신 **discrete visual tokens**을 예측한다.

**2-Stage 접근**:

**Stage 1: Tokenizer 학습 (dVAE)**

이미지를 discrete tokens으로 변환:

$$\mathbf{x} \xrightarrow{\text{dVAE}} \mathbf{v} = \{v_1, v_2, ..., v_N\}, \quad v_i \in \{1, 2, ..., K\}$$

여기서:
- $N = (H/P) \times (W/P)$: 패치 개수
- $K = 8192$: Vocabulary 크기

**dVAE 구조**:

Encoder: $\mathbf{x} \rightarrow \mathbf{z}_{\text{discrete}}$ (VQ-VAE)

Decoder: $\mathbf{z}_{\text{discrete}} \rightarrow \hat{\mathbf{x}}$

**Vector Quantization**:

$$\mathbf{z}_q = \underset{\mathbf{e}_k \in \mathcal{E}}{\arg\min} \|\mathbf{z}_e - \mathbf{e}_k\|_2$$

여기서 $\mathcal{E} = \{\mathbf{e}_1, ..., \mathbf{e}_K\}$는 codebook (K=8192)

**Stage 2: BEiT Pre-training**

1. **Random Masking**: 40% 패치 마스킹
   $$\mathbf{v}_{\text{masked}} = \text{Mask}(\mathbf{v}, p=0.4)$$

2. **ViT Encoding**: 
   $$\mathbf{h} = \text{ViT}(\mathbf{v}_{\text{masked}})$$

3. **Token Prediction**: Classification head
   $$\hat{v}_i = \text{softmax}(\mathbf{W}\mathbf{h}_i), \quad i \in \text{masked positions}$$

4. **Loss**: Cross-Entropy
   $$\mathcal{L} = -\sum_{i \in M} \log p(v_i | \mathbf{v}_{\backslash M})$$

### 4.3.3 기술적 세부사항

**Tokenizer (dVAE) 학습**:

목적: 이미지를 8192개 discrete codes로 압축

Loss:
$$\mathcal{L}_{\text{dVAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \|\text{sg}[\mathbf{z}_e] - \mathbf{e}_k\|^2 + \beta\|\mathbf{z}_e - \text{sg}[\mathbf{e}_k]\|^2$$

여기서 $\text{sg}[\cdot]$는 stop-gradient

**BEiT Architecture**:

- **Backbone**: ViT-Base (12 layers, 768 dim)
- **Patch size**: 16×16
- **Input**: 224×224 → 14×14 = 196 patches
- **Masking ratio**: 40%
- **Vocabulary**: 8192 tokens

**Masking Strategy**:

BEiT는 **Block-wise masking**을 사용:
- 연속된 패치 블록을 마스킹
- Random masking보다 어려운 task
- 더 강력한 context 이해 필요

**Prediction Head**:

간단한 Linear layer:
$$\hat{v}_i = \text{softmax}(\mathbf{W} \mathbf{h}_i + \mathbf{b})$$

$\mathbf{W} \in \mathbb{R}^{8192 \times 768}$

### 4.3.4 BERT와의 비교

| 측면 | BERT (NLP) | BEiT (Vision) |
|------|-----------|---------------|
| **입력** | Word tokens (discrete) | Image patches (continuous) |
| **Tokenization** | WordPiece | dVAE (8192 visual tokens) |
| **Masking** | 15% | 40% |
| **Prediction** | Masked words | Masked visual tokens |
| **Loss** | Cross-Entropy | Cross-Entropy |
| **Pre-training** | Large text corpus | ImageNet-1K (1.3M) |

**핵심 차이**:
- BERT: 자연적으로 discrete (단어)
- BEiT: 연속적 이미지를 discrete tokens으로 변환 (dVAE)

### 4.3.5 성능

**ImageNet-1K 벤치마크**:

| Model | Linear Probing | Fine-tuning |
|-------|----------------|-------------|
| Supervised (ViT-B) | 76.5% | 84.5% |
| BEiT-Base | 56.7% | 83.2% |
| **BEiT-Large** | **82.1%** | **85.2%** |

**Downstream Tasks**:

- **Semantic Segmentation (ADE20K)**: 47.1 mIoU
- **Object Detection (COCO)**: 51.3 AP^box

### 4.3.6 장점

1. **Discrete Tokens**: Classification loss로 명확한 학습 신호
2. **BERT 성공 재현**: NLP의 검증된 방법론 적용
3. **Fine-tuning 우수**: 85.2% (Supervised 84.5% 초과)
4. **Semantic 학습**: Token prediction이 의미적 이해 유도

### 4.3.7 단점

1. **2-Stage 복잡도**: dVAE 먼저 학습 필요
2. **Tokenizer 품질 의존**: dVAE가 나쁘면 BEiT도 나쁨
3. **낮은 Linear Probing**: 56.7% (Supervised 76.5% 대비 -19.8%p)
4. **메모리 사용**: Codebook 8192개 저장
5. **계산 비용**: dVAE inference 추가 오버헤드

### 4.3.8 BEiT의 의의

BEiT는 "Vision에도 BERT 방식이 작동한다"는 것을 입증했다. 특히:
- Discrete tokens의 효과성
- Masked prediction의 강력함
- Fine-tuning에서 Supervised 초과 가능성

하지만 2-stage 복잡도는 이후 MAE가 해결하게 된다.

---

## 4.4 MAE (2022)

### 4.4.1 기본 정보

- **논문**: Masked Autoencoders Are Scalable Vision Learners
- **발표**: CVPR 2022
- **저자**: Kaiming He et al. (Meta/Facebook AI)
- **GitHub**: https://github.com/facebookresearch/mae

### 4.4.2 핵심 원리

MAE는 **극단적으로 단순한** 접근법으로 MIM을 재정의했다:
- Discrete tokens 불필요 → **Continuous pixels** 직접 예측
- 40% 마스킹 → **75% 마스킹**
- 복잡한 구조 → **Asymmetric Encoder-Decoder**

**핵심 아이디어**: "이미지는 중복이 많으므로, 75%를 마스킹해도 복원 가능하며, 이는 더 어려운 task이므로 더 나은 표현을 학습한다"

### 4.4.3 BEiT 대비 핵심 차이점

| 측면 | BEiT | MAE | 개선 효과 |
|------|------|-----|----------|
| **Tokenization** | dVAE (8192 tokens) | 없음 (raw pixels) | 단순화 |
| **Prediction Target** | Discrete tokens | Continuous pixels | 단순화 |
| **Loss** | Cross-Entropy | MSE (pixel space) | 단순화 |
| **Masking Ratio** | 40% | 75% | 어려운 task |
| **Encoder Input** | All patches (masked) | Visible patches만 (25%) | 3-4배 빠름 |
| **Decoder** | 없음 (shared encoder) | 경량 (8 layers) | 효율적 |
| **Linear Probing** | 56.7% | 68.0% | +11.3%p |
| **Fine-tuning** | 83.2% | **85.9%** | +2.7%p |
| **학습 시간** | 길다 | **3배 빠름** | 효율성 |

### 4.4.4 기술적 세부사항

**Patchify 및 Masking**:

1. **이미지를 패치로 분할**:
   $$\mathbf{x} \in \mathbb{R}^{224 \times 224 \times 3} \rightarrow \{\mathbf{p}_1, ..., \mathbf{p}_N\}, \quad N = 196, \mathbf{p}_i \in \mathbb{R}^{16 \times 16 \times 3}$$

2. **Random Masking (75%)**:
   $$M \subset \{1, ..., N\}, \quad |M| = 0.75N = 147$$
   $$\mathbf{p}_{\text{visible}} = \{\mathbf{p}_i : i \notin M\}, \quad |\mathbf{p}_{\text{visible}}| = 49$$

**Asymmetric Encoder-Decoder**:

**Encoder (Heavy, ViT-Large)**:
- **입력**: Visible patches만 (25%)
- **구조**: 24 layers, 1024 dim
- **출력**: $\mathbf{h}_{\text{visible}} \in \mathbb{R}^{49 \times 1024}$

$$\mathbf{h}_{\text{visible}} = \text{ViT-Encoder}(\mathbf{p}_{\text{visible}})$$

**핵심**: Encoder는 마스킹된 패치를 보지 않음 → 3-4배 빠른 학습

**Decoder (Light)**:
- **입력**: Encoder output + Mask tokens
  $$\mathbf{h}_{\text{all}} = [\mathbf{h}_{\text{visible}}, \mathbf{m}_{\text{masked}}]$$
  여기서 $\mathbf{m}_{\text{masked}} \in \mathbb{R}^{147 \times 512}$는 학습 가능한 mask tokens

- **구조**: 8 layers, 512 dim (Encoder보다 경량)
- **출력**: Predicted patches
  $$\hat{\mathbf{p}}_{\text{masked}} = \text{Decoder}(\mathbf{h}_{\text{all}})$$

**Reconstruction Loss**:

Masked patches만 복원 (MSE in pixel space):

$$\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \|\mathbf{p}_i - \hat{\mathbf{p}}_i\|^2$$

**중요**: Visible patches는 loss에 포함 안 됨

### 4.4.5 MAE의 핵심 설계 선택

**1) High Masking Ratio (75%)**

**BERT**: 15% masking
- 이유: 단어는 discrete, 중복 적음
- 85% 보이면 context 충분

**MAE**: 75% masking
- 이유: 이미지는 중복 많음, 75%도 복원 가능
- 25%만으로도 semantic 정보 파악 가능
- **효과**: 어려운 task → 더 나은 표현

**실험 결과**:

| Masking Ratio | Linear Probing | Fine-tuning |
|--------------|----------------|-------------|
| 15% | 48% | 82% |
| 50% | 62% | 84% |
| **75%** | **68%** | **86%** |

75%가 최적!

**2) Asymmetric Encoder-Decoder**

**대칭 구조 (전통적 AE)**:
- Encoder: Heavy, Decoder: Heavy
- 문제: 학습 느림, 메모리 많이 사용

**MAE의 비대칭**:
- Encoder: Very heavy (24 layers), visible만 처리
- Decoder: Light (8 layers), 복원만 담당

**이유**:
- **Encoder**: Representation 학습이 목적 → 강력해야 함
- **Decoder**: Reconstruction은 pretext task일 뿐 → 경량 가능

**효과**:
- Encoder는 25% 패치만 처리 → 3-4배 빠른 학습
- Decoder는 경량 → 메모리 절감
- Downstream에서는 Decoder 버림 → 추론 효율

**3) Pixel Prediction (vs Tokens)**

**BEiT**: Discrete tokens 예측 (Cross-Entropy)
**MAE**: Continuous pixels 예측 (MSE)

**MAE 선택 이유**:
- **단순함**: Tokenizer 불필요
- **End-to-End**: 단일 stage 학습
- **효과적**: Pixel reconstruction도 충분히 강력

**픽셀 복원이 왜 작동하는가?**

전통적 AE는 실패했지만, MAE는 성공한 이유:
1. **75% masking**: 단순 복사 불가능, semantic 이해 필요
2. **ViT**: Global attention으로 long-range dependency
3. **Large scale**: ViT-Large로 충분한 capacity

### 4.4.6 성능

**ImageNet-1K 벤치마크**:

| Model | Params | Linear Probing | Fine-tuning |
|-------|--------|----------------|-------------|
| Supervised (ViT-L) | 304M | 76.5% | 84.5% |
| BEiT-Large | 304M | 82.1% | 85.2% |
| **MAE-Large** | **304M** | **68.0%** | **85.9%** |
| **MAE-Huge** | **632M** | **75.0%** | **87.8%** |

**Downstream Tasks**:

| Task | Dataset | MAE | Supervised |
|------|---------|-----|------------|
| **Object Detection** | COCO | 53.3 AP^box | 51.3 AP^box |
| **Semantic Segmentation** | ADE20K | 48.1 mIoU | 47.4 mIoU |
| **Instance Segmentation** | COCO | 47.2 AP^mask | 45.8 AP^mask |

**모든 downstream에서 Supervised 초과!**

**학습 효율**:

| 측면 | BEiT | MAE | 개선 |
|------|------|-----|------|
| **Pre-training Epochs** | 800 | 1600 | 2배 |
| **GPU-hours** | ~2000 | ~1200 | **40% 감소** |
| **Reason** | - | 75% masking → 25%만 처리 | 3-4배 빠른 iteration |

**Wall-clock time**: MAE가 빠름 (epoch 수는 많지만 iteration이 빠름)

### 4.4.7 장점

1. **최고 Fine-tuning**: 85.9% (ViT-L), 87.8% (ViT-H)
2. **단순함**: 1-stage, pixel prediction, 간단한 구조
3. **학습 효율**: 3-4배 빠른 iteration
4. **Downstream 우수**: 모든 task에서 Supervised 초과
5. **확장성**: ViT-Huge까지 잘 확장
6. **구현 용이**: 100줄 미만 PyTorch 코드

### 4.4.8 단점

1. **낮은 Linear Probing**: 68% (Supervised 76.5% 대비 -8.5%p)
   - **이유**: Pixel reconstruction은 low-level 정보 중시
   - **해결**: Fine-tuning으로 해결 (85.9%)

2. **Fine-tuning 필수**: Linear probing만으로는 부족
   - Contrastive (MoCo v3): Linear 76%, Fine-tuning 84%
   - MAE: Linear 68%, Fine-tuning 86%

3. **많은 Epochs**: 1600 epochs 필요 (Contrastive는 300)

### 4.4.9 MAE의 게임 체인저적 영향

MAE는 MIM 패러다임을 **완전히 재정의**했다:

**Before MAE (BEiT 시대)**:
- MIM은 복잡하다 (2-stage, discrete tokens)
- Supervised보다 낮다고 생각됨
- Contrastive가 더 나은 선택

**After MAE**:
- MIM은 간단하다 (1-stage, pixels)
- **Supervised를 초과한다** (85.9% vs 84.5%)
- Fine-tuning 중심 task에서 최고 선택

**영향**:
- 수많은 후속 연구 (SimMIM, MaskFeat, iBOT, ...)
- Vision Transformer SSL의 표준
- Dense prediction (Segmentation, Detection)에서 우세

---

## 4.5 SimMIM (2022)

### 4.5.1 기본 정보

- **논문**: SimMIM: A Simple Framework for Masked Image Modeling
- **발표**: CVPR 2022
- **저자**: Zhenda Xie et al. (Microsoft Research)
- **GitHub**: https://github.com/microsoft/SimMIM

### 4.5.2 핵심 원리

SimMIM은 MAE를 **더 단순화**한 모델이다. MAE의 asymmetric encoder-decoder 대신, **단일 encoder + linear prediction head**만 사용한다.

**MAE와의 차이**:

| 측면 | MAE | SimMIM |
|------|-----|--------|
| **Encoder** | ViT (visible만) | ViT/Swin (all patches) |
| **Decoder** | 8-layer Transformer | 1-layer Linear |
| **Masking** | 75% | 50% |
| **Prediction** | Pixels (16×16) | Pixels (16×16 or 32×32) |

### 4.5.3 기술적 세부사항

**Architecture**:

$$\mathbf{h} = \text{Encoder}(\mathbf{x}_{\text{masked}})$$
$$\hat{\mathbf{x}}_{\text{masked}} = \text{Linear}(\mathbf{h}_{\text{masked}})$$

**Masking Strategy**:
- **Random masking**: 50%
- **MAE보다 낮은 이유**: Encoder가 모든 패치 처리하므로 너무 높으면 정보 부족

**Prediction Head**:

간단한 Linear layer:
$$\hat{\mathbf{p}}_i = \mathbf{W}\mathbf{h}_i + \mathbf{b}$$

$\mathbf{W} \in \mathbb{R}^{768 \times d}$, d = patch dimension (예: 768)

### 4.5.4 성능

**ImageNet-1K**:

| Model | Backbone | Linear Probing | Fine-tuning |
|-------|----------|----------------|-------------|
| MAE | ViT-B | 68.0% | 83.6% |
| **SimMIM** | **Swin-B** | **56.7%** | **83.8%** |

**Downstream (ADE20K Segmentation)**:

| Model | mIoU |
|-------|------|
| Supervised | 48.1 |
| MAE | 48.1 |
| **SimMIM** | **52.7** (+4.6%p) |

**Swin Transformer에서 특히 강력!**

### 4.5.5 장점

1. **극단적 단순함**: Linear prediction head만
2. **Swin Transformer**: Hierarchical ViT에 잘 맞음
3. **Dense prediction 우수**: Segmentation에서 탁월
4. **구현 간단**: MAE보다 더 간단

### 4.5.6 단점

1. **낮은 Linear Probing**: 56.7% (MAE 68.0% 대비도 낮음)
2. **Encoder 비효율**: 모든 패치 처리 → MAE보다 느림
3. **Swin 의존**: ViT에서는 MAE만 못함

### 4.5.7 SimMIM의 의의

SimMIM은 "더 단순해도 작동한다"는 것을 보여주었다. 특히:
- Decoder 필요 없음 (Linear만으로 충분)
- Swin Transformer와의 궁합
- Dense prediction에서 우수

하지만 일반적으로는 MAE가 더 널리 사용됨.

---

## 4.6 MaskFeat (2022)

### 4.6.1 기본 정보

- **논문**: Masked Feature Prediction for Self-Supervised Visual Pre-Training
- **발표**: CVPR 2022
- **저자**: Chen Wei et al. (Meta/Facebook AI)
- **GitHub**: https://github.com/facebookresearch/SlowFast

### 4.6.2 핵심 원리

MaskFeat는 **픽셀 대신 hand-crafted features를 예측**한다:
- MAE: Raw pixels 예측
- MaskFeat: HOG (Histogram of Oriented Gradients) 예측

**Motivation**: "Pixel은 low-level 정보가 많고, mid-level features (edges, textures)가 semantic 학습에 더 유용하다"

### 4.6.3 기술적 세부사항

**HOG Features**:

각 패치에서:
1. Gradient 계산: $\nabla I = (\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y})$
2. Magnitude와 Orientation: 
   $$m = \sqrt{(\frac{\partial I}{\partial x})^2 + (\frac{\partial I}{\partial y})^2}$$
   $$\theta = \arctan(\frac{\partial I / \partial y}{\partial I / \partial x})$$
3. Histogram of orientations (9 bins)

**Prediction Target**:

$$\mathcal{L} = \frac{1}{|M|}\sum_{i \in M} \|\text{HOG}(\mathbf{p}_i) - \hat{\mathbf{f}}_i\|^2$$

**Architecture**: MAE와 동일 (Encoder-Decoder)

### 4.6.4 성능

**ImageNet-1K**:

| Model | Target | Linear Probing | Fine-tuning |
|-------|--------|----------------|-------------|
| MAE | Pixels | 68.0% | 83.6% |
| **MaskFeat** | **HOG** | **71.4%** | **84.0%** |

**Improvement**: Linear +3.4%p, Fine-tuning +0.4%p

### 4.6.5 장점

1. **더 나은 Linear Probing**: 71.4% (MAE 68.0% 대비 +3.4%p)
2. **Mid-level 학습**: HOG가 semantic 정보 더 많음
3. **Video에도 효과**: MaskFeat는 video SSL에도 사용됨

### 4.6.6 단점

1. **Hand-crafted Features**: HOG는 수작업 설계
2. **추가 계산**: HOG 추출 오버헤드
3. **미세한 개선**: Fine-tuning에서는 0.4%p만 향상

### 4.6.7 MaskFeat의 의의

MaskFeat는 "Prediction target 선택의 중요성"을 보여주었다:
- Raw pixels보다 HOG가 더 나은 표현 학습
- Mid-level features의 유용성
- 하지만 실무에서는 MAE의 단순함이 선호됨

---

## 4.7 Generative 패러다임 종합 비교

### 4.7.1 기술적 진화 과정

```
Auto-encoder / VAE (전통)
├─ 접근: 전체 이미지 복원
├─ 문제: 단순 픽셀 복사, low-level 정보
└─ 성능: 50-65% Linear Probing

        ↓ Discrete Tokens

BEiT (2021)
├─ 혁신: BERT 방식 적용 (discrete tokens)
├─ 특징: dVAE tokenizer + 40% masking
├─ 문제: 2-stage 복잡도
└─ 성능: 56.7% Linear, 83.2% Fine-tuning

        ↓ 단순화 + High Masking

MAE (2022) ★★★★★
├─ 혁신: 1-stage, pixels, 75% masking
├─ 특징: Asymmetric encoder-decoder
├─ 효과: 3-4배 빠른 학습
└─ 성능: 68.0% Linear, 85.9% Fine-tuning (SOTA)

        ↓ 분기

SimMIM (2022)              MaskFeat (2022)
├─ 단순화: Linear head        ├─ 개선: HOG features
├─ Swin Transformer          ├─ Mid-level 학습
└─ 성능: 83.8% (Swin)        └─ 성능: 71.4% Linear
```

### 4.7.2 상세 비교표

| 비교 항목 | AE/VAE | BEiT | MAE | SimMIM | MaskFeat |
|----------|--------|------|-----|--------|----------|
| **연도** | ~2015 | 2021 | 2022 | 2022 | 2022 |
| **Tokenization** | 없음 | dVAE (8192) | 없음 | 없음 | 없음 |
| **Prediction Target** | Pixels | Discrete tokens | Pixels | Pixels | HOG features |
| **Loss** | MSE | Cross-Entropy | MSE | MSE | MSE |
| **Masking Ratio** | 0% (전체) | 40% | **75%** | 50% | 75% |
| **Encoder Input** | All | Masked all | **Visible만** | Masked all | Visible만 |
| **Decoder** | Heavy | Shared | **Light (8L)** | **Linear** | Light (8L) |
| **Backbone** | CNN | ViT | ViT | ViT/Swin | ViT |
| **Linear Probing** | 50-65% | 56.7% | 68.0% | 56.7% | **71.4%** |
| **Fine-tuning** | - | 83.2% | **85.9%** | 83.8% | 84.0% |
| **학습 효율** | 낮음 | 중간 | **높음** (3-4배) | 낮음 | 중간 |
| **구현 복잡도** | 낮음 | 높음 (2-stage) | 중간 | **매우 낮음** | 중간 |
| **Dense Pred** | 약함 | 중간 | **강함** | **매우 강함** | 강함 |
| **종합 평가** | ★☆☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ |

### 4.7.3 핵심 Trade-off

**1) Linear Probing vs Fine-tuning**

```
Contrastive (MoCo v3):
- Linear: 76% (높음)
- Fine-tuning: 84% (중간)
- 특징: Instance discrimination

Generative (MAE):
- Linear: 68% (낮음)
- Fine-tuning: 86% (높음)
- 특징: Reconstruction → Local patterns
```

**왜 이런 차이가 나는가?**

**Contrastive**:
- Global similarity 학습 → Instance classification에 직접 유용
- Linear layer만 추가해도 성능 좋음

**Generative (MIM)**:
- Local reconstruction 학습 → Pixel-level 정보 풍부
- Dense prediction에 유리하지만, global classification에는 부족
- Fine-tuning으로 task-specific 정보 추가 필요

**2) Masking Ratio의 Trade-off**

| Masking | Task 난이도 | 학습 효율 | 표현 품질 | 최적 모델 |
|---------|-----------|----------|----------|----------|
| 15% | 쉬움 | 낮음 | 낮음 | - |
| 40% | 중간 | 중간 | 중간 | BEiT |
| 50% | 중간 | 중간 | 중간 | SimMIM |
| **75%** | **어려움** | **높음** | **높음** | **MAE** |

75%가 최적인 이유:
- 충분히 어려워서 semantic 이해 필요
- 25%만 처리 → 빠른 학습
- 이미지의 중복성으로 여전히 복원 가능

**3) Prediction Target의 Trade-off**

| Target | 장점 | 단점 | 대표 모델 |
|--------|------|------|----------|
| **Pixels** | 단순, End-to-End | Low-level 정보 많음 | MAE, SimMIM |
| **Tokens (discrete)** | 명확한 학습 신호 | 2-stage, 복잡 | BEiT |
| **HOG features** | Mid-level 학습 | Hand-crafted | MaskFeat |

**결론**: Pixels이 단순함과 성능의 균형

### 4.7.4 실무 적용 가이드

**MAE 선택 시나리오** (★★★★★ 최고 추천):
- **Fine-tuning 가능**: 최고 성능 (85.9%)
- **Dense prediction**: Segmentation, Detection
- **ViT 사용**: Transformer backbone
- **대규모 데이터**: ImageNet-1K 이상
- **충분한 학습 시간**: 1600 epochs

**BEiT 선택 시나리오** (★★☆☆☆ 제한적):
- **연구 목적**: Discrete tokens 탐색
- **BERT 방식 선호**: NLP 배경
- **현실적으로**: MAE로 대체 가능

**SimMIM 선택 시나리오** (★★★☆☆):
- **Swin Transformer**: Hierarchical ViT
- **극단적 단순함**: Linear head만
- **Dense prediction**: Segmentation 중심

**MaskFeat 선택 시나리오** (★★☆☆☆):
- **Linear probing 중요**: 71.4%
- **Video SSL**: MaskFeat가 video에도 유용
- **현실적으로**: MAE와 큰 차이 없음

### 4.7.5 Generative vs Contrastive vs Clustering

**언제 Generative (MAE)를 선택하는가?**

✅ **Fine-tuning 가능**: 85.9% 최고 성능
✅ **Dense prediction**: Segmentation, Detection 우수
✅ **Pixel-level 정보**: Local details 중요
✅ **Large model**: ViT-Large, ViT-Huge 사용

**언제 Contrastive (MoCo v3)를 선택하는가?**

✅ **Linear probing만**: 76% 높은 성능
✅ **Instance discrimination**: Classification 중심
✅ **ResNet backbone**: CNN 사용 시

**언제 Clustering (DINO)를 선택하는가?**

✅ **Linear probing 최강**: 80%
✅ **해석 가능성**: Attention map
✅ **Few-shot**: 75% 최고

**Best Practice**:
- **Classification + Linear**: Clustering (DINO 80%)
- **Classification + Fine-tuning**: Generative (MAE 86%)
- **Segmentation/Detection**: Generative (MAE, SimMIM)
- **둘 다 중요**: Hybrid (iBOT - MIM + Contrastive)

---

## 부록: 관련 테이블

### A.1 Generative vs 다른 패러다임

| 패러다임 | 대표 모델 | Linear | Fine-tuning | 주요 장점 | 주요 단점 |
|---------|----------|--------|-------------|----------|----------|
| Discriminative | Rotation | 55% | - | 교육적 | 낮은 성능 |
| Clustering | DINO | **80%** | 84.5% | 해석성, Few-shot | ViT 의존 |
| Contrastive | MoCo v3 | 76% | 84.1% | Instance | Large batch |
| **Generative** | **MAE** | **68%** | **85.9%** | **Dense pred, Fine-tuning** | **Linear 낮음** |
| Hybrid | DINOv2 | **86%** | **86%** | SOTA | 계산 비용 |

### A.2 Masking Ratio 실험 (MAE 기준)

| Masking | Encoder Speed | Linear | Fine-tuning | 비고 |
|---------|--------------|--------|-------------|------|
| 15% | 1× (baseline) | 48% | 82% | 너무 쉬움 |
| 50% | 2× | 62% | 84% | 중간 |
| **75%** | **4×** | **68%** | **86%** | **최적** |
| 90% | 10× | 52% | 83% | 너무 어려움 |

### A.3 Downstream Task 성능 (ViT-Large)

| Task | Dataset | Supervised | MAE | 개선 |
|------|---------|-----------|-----|------|
| **Classification** | ImageNet | 84.5% | 85.9% | +1.4%p |
| **Object Detection** | COCO | 51.3 AP | 53.3 AP | +2.0 AP |
| **Segmentation** | ADE20K | 47.4 mIoU | 48.1 mIoU | +0.7 mIoU |
| **Instance Seg** | COCO | 45.8 AP | 47.2 AP | +1.4 AP |

### A.4 개발-배포 체크리스트 (MAE)

**Phase 1: 환경 준비**
- [ ] ViT backbone 선택 (Base/Large/Huge)
- [ ] Pre-training 데이터 확보 (ImageNet-1K 권장)
- [ ] GPU 리소스 (8× V100/A100 권장)

**Phase 2: Pre-training**
- [ ] Masking ratio 설정 (기본 75%)
- [ ] Encoder: ViT-Base/Large (24 layers)
- [ ] Decoder: 8 layers (경량)
- [ ] Epochs: 800-1600 (데이터셋 크기에 따라)
- [ ] Optimizer: AdamW, lr=1.5e-4
- [ ] 학습 모니터링 (reconstruction loss)

**Phase 3: Evaluation**
- [ ] Linear probing (빠른 평가)
  - Expected: 60-70% (ViT-Base)
- [ ] Fine-tuning (최종 성능)
  - Expected: 83-86% (ViT-Base/Large)

**Phase 4: Downstream Application**
- [ ] Task 선택 (Classification/Detection/Segmentation)
- [ ] MAE encoder를 backbone으로 사용
- [ ] Task-specific head 추가
- [ ] Fine-tuning

### A.5 하드웨어 요구사항

| Model | Pre-training | Fine-tuning | 추론 |
|-------|-------------|-------------|------|
| **MAE-Base** | 8× V100 (32GB) | 4× V100 | 1× V100 |
| **MAE-Large** | 8× A100 (40-80GB) | 8× V100 | 1× V100 |
| **MAE-Huge** | 16× A100 (80GB) | 16× A100 | 2× V100 |

**학습 시간**:
- MAE-Base: ~3 days (800 epochs, ImageNet-1K)
- MAE-Large: ~5 days (1600 epochs)

### A.6 MAE 구현 (PyTorch)

```python
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking
        x: [N, L, D]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep visible
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # Embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking: 75% masked
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove cls token
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # Mean loss on removed patches
        return loss
    
    def patchify(self, imgs):
        """
        imgs: [N, 3, H, W]
        x: [N, L, patch_size**2 * 3]
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# Training
model = MAE()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

for epoch in range(1600):
    for imgs, _ in dataloader:
        loss, _, _ = model(imgs, mask_ratio=0.75)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### A.7 MAE 최적화 Tips

**1) 학습 가속**:
```python
# Mixed Precision Training
scaler = torch.cuda.amp.GradScaler()

# Gradient Checkpointing (메모리 절감)
model.gradient_checkpointing = True

# Larger Batch Size
# MAE는 Contrastive와 달리 batch size에 덜 민감
batch_size = 2048  # 가능한 크게
```

**2) 메모리 최적화**:
- Encoder는 visible patches만 처리 (이미 효율적)
- Decoder는 경량 (8 layers)
- Gradient checkpointing 사용

**3) 하이퍼파라미터**:
```python
# 검증된 기본 설정 (MAE paper)
masking_ratio = 0.75
encoder_depth = 24  # ViT-Large
decoder_depth = 8
embed_dim = 1024
decoder_embed_dim = 512
warmup_epochs = 40
epochs = 1600
base_lr = 1.5e-4
weight_decay = 0.05
```

### A.8 주요 논문 및 인용수

| 논문 | 연도 | 인용수 | 중요도 |
|------|------|--------|--------|
| **BEiT** | 2021 | 3000+ | 높음 |
| **MAE** | 2022 | 8000+ | 매우 높음 |
| **SimMIM** | 2022 | 1500+ | 중간 |
| **MaskFeat** | 2022 | 800+ | 중간 |

---

## 결론

Generative (MIM) 패러다임은 **MAE를 정점으로 Fine-tuning 중심 환경에서 최고 성능**을 보여주고 있다:

**성능 진화**:
```
2015: AE/VAE 50-65%
2021: BEiT 56.7% (Linear), 83.2% (Fine-tuning)
2022: MAE 68.0% (Linear), 85.9% (Fine-tuning) ← SOTA
```

**핵심 기여**:
1. **Fine-tuning SOTA**: 85.9% (ViT-L), 87.8% (ViT-H)
2. **Dense prediction 우수**: Segmentation, Detection에서 탁월
3. **단순함**: 1-stage, pixel prediction
4. **학습 효율**: 75% masking으로 3-4배 빠른 iteration

**핵심 발견**:
- **BEiT → MAE**: 2-stage → 1-stage 단순화로 성능 향상
- **75% masking**: 높은 masking이 오히려 더 나은 표현 학습
- **Asymmetric design**: Heavy encoder + Light decoder = 효율성
- **Pixel prediction**: Discrete tokens 불필요, raw pixels도 충분

**Trade-off**:
- Linear probing: 낮음 (68%)
- Fine-tuning: 최고 (85.9%)
- → **Fine-tuning 가능한 환경에서 최적**

**최종 권장사항**:

| 시나리오 | 추천 모델 | 이유 |
|---------|----------|------|
| **Classification + Fine-tuning** | MAE | 85.9% SOTA |
| **Dense prediction** | MAE, SimMIM | Segmentation 우수 |
| **Linear probing만** | DINO, MoCo v3 | 80%, 76% |
| **균형** | iBOT | MIM + Contrastive |

Generative (MIM) 방식은 Contrastive, Clustering과 함께 Vision SSL의 3대 핵심 패러다임으로 자리잡았으며, 특히 **Transformer + Fine-tuning 환경**에서 최고 선택이다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: AE/VAE, BEiT, MAE, SimMIM, MaskFeat

**주요 내용**:
1. Generative 패러다임 개요 (MIM 수식)
2. 전통적 AE/VAE 분석 및 한계
3. BEiT 상세 분석 (BERT for Vision, dVAE)
4. **MAE 상세 분석** (75% masking, Asymmetric design, 85.9%)
5. SimMIM, MaskFeat 비교
6. 종합 비교 및 실무 가이드
7. Trade-off 분석 (Linear vs Fine-tuning)
8. **부록**: 성능 벤치마크, 구현 코드, 하드웨어 요구사항