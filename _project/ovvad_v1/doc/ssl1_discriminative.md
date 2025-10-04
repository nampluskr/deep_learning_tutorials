# 1. Discriminative (Pretext Tasks) 패러다임 상세 분석

## 1.1 패러다임 개요

Discriminative 방식은 **수작업으로 설계한 Pretext Task**를 푸는 과정에서 이미지의 구조와 의미를 학습한다. "이미지를 어떻게 변형했는가?"를 맞추는 문제를 통해 이미지에 대한 표현을 획득한다.

**핵심 수식**:

$$\min_\theta \mathbb{E}_{\mathbf{x}} \mathcal{L}_{\text{CE}}(g_\theta(T(\mathbf{x})), t)$$

여기서:
- $\mathbf{x}$: 입력 이미지
- $T(\mathbf{x})$: Transformation (회전, 순서 섞기 등)
- $t$: Transformation type (pseudo-label)
- $g_\theta$: Predictor 네트워크
- $\mathcal{L}_{\text{CE}}$: Cross-Entropy Loss

**핵심 가정**: "변환을 인식하려면 이미지의 구조와 의미를 이해해야 한다"

**학습 과정**:
1. 이미지에 사전 정의된 변환 적용
2. 변환 유형을 네트워크가 예측
3. 예측과 실제 변환 간 오차 최소화
4. 부산물로 좋은 표현 학습

---

## 1.2 Context Prediction (2015)

### 1.2.1 기본 정보

- **논문**: Unsupervised Visual Representation Learning by Context Prediction
- **발표**: ICCV 2015
- **저자**: Carl Doersch et al. (CMU)
- **인용수**: 2000+회

### 1.2.2 핵심 원리

Context Prediction은 **패치 간의 상대적 위치 관계**를 예측하는 task이다.

**문제 설정**:

이미지를 9개 영역으로 나누고, 중앙 패치와 주변 8개 패치 중 하나를 선택:

```
[1] [2] [3]
[4] [5] [6]
[7] [8] [9]
```

**Task**: 중앙 패치(5번)와 다른 패치(예: 1번)가 주어졌을 때, 상대적 위치를 예측

$$p(\text{pos} | \mathbf{p}_5, \mathbf{p}_i), \quad i \in \{1,2,3,4,6,7,8,9\}$$

**8-way Classification**: 8개 위치 중 하나

### 1.2.3 기술적 세부사항

**Architecture**:

두 개의 AlexNet-style 네트워크 (weight sharing):

$$\mathbf{f}_5 = f_\theta(\mathbf{p}_5)$$
$$\mathbf{f}_i = f_\theta(\mathbf{p}_i)$$

**Concatenation + FC**:

$$\mathbf{h} = [\mathbf{f}_5, \mathbf{f}_i]$$
$$\hat{y} = \text{softmax}(\mathbf{W}\mathbf{h} + \mathbf{b})$$

**Loss**:

$$\mathcal{L} = -\sum_{k=1}^{8} y_k \log \hat{y}_k$$

### 1.2.4 성능

**ImageNet Linear Probing**: 30-35%
- Supervised (AlexNet): ~60%
- Gap: -25~30%p

**주요 발견**:
- Chromatic aberration 문제: 패치 경계의 색상 차이로 위치 추측 가능
- 해결: 패치 사이에 gap 추가, color jittering

### 1.2.5 장점

1. **최초의 본격적 Pretext Task**: SSL의 가능성 제시
2. **직관적**: 공간 관계 이해는 시각 인식의 기본
3. **간단한 구현**: AlexNet 2개 + FC layer

### 1.2.6 단점

1. **낮은 성능**: 30-35% (실용성 부족)
2. **Shortcut learning**: 색상 차이 등 low-level cue 사용
3. **제한적 표현**: 공간 관계만 학습

### 1.2.7 Context Prediction의 의의

Context Prediction은 **Self-Supervised Learning의 시작**을 알렸다. "레이블 없이도 이미지의 구조를 학습할 수 있다"는 가능성을 보여주었다. 하지만 성능이 낮고 shortcut learning 문제로 인해 실용성은 제한적이었다.

---

## 1.3 Colorization (2016)

### 1.3.1 기본 정보

- **논문**: Colorful Image Colorization
- **발표**: ECCV 2016
- **저자**: Richard Zhang et al. (UC Berkeley)
- **GitHub**: https://github.com/richzhang/colorization

### 1.3.2 핵심 원리

Colorization은 **흑백 이미지를 컬러로 복원**하는 task를 Pretext로 사용한다.

**문제 설정**:

$$\mathbf{x}_{\text{gray}} = \text{RGB2Gray}(\mathbf{x})$$
$$\min_\theta \mathcal{L}(\text{Colorize}_\theta(\mathbf{x}_{\text{gray}}), \mathbf{x})$$

**왜 효과적인가?**
- 색을 맞추려면 **의미적 이해** 필요
- 예: 하늘은 파랑, 잔디는 초록, 바나나는 노랑
- Low-level 텍스처만으로는 불가능

### 1.3.3 기술적 세부사항

**Color Space**: Lab color space 사용
- L channel: Lightness (입력)
- ab channels: Color (예측)

**Quantized ab space**:
- ab를 313개 bin으로 quantization
- Classification 문제로 전환

**Architecture**:

CNN Encoder-Decoder:

$$\mathbf{z} = \text{Encoder}(\mathbf{L})$$
$$\hat{\mathbf{ab}} = \text{Decoder}(\mathbf{z})$$

**Loss**: Multinomial cross-entropy (313-way)

$$\mathcal{L} = -\sum_{h,w} \sum_{q=1}^{313} \mathbf{Z}_{h,w,q} \log \hat{\mathbf{Z}}_{h,w,q}$$

여기서 $\mathbf{Z}$는 ground-truth distribution

**Class Rebalancing**:
- 회색, 갈색이 지배적 → imbalanced
- Rare color에 높은 weight 부여

### 1.3.4 성능

**ImageNet Linear Probing**: 38-40%
- Context Prediction 대비 +5~8%p
- 여전히 Supervised 대비 낮음

**Downstream Task**:
- Object Detection (PASCAL VOC): 40% mAP
- Semantic Segmentation: 35% mIoU

### 1.3.5 장점

1. **의미적 이해**: 색상은 객체 종류와 연관
2. **자연스러운 task**: 실제로 유용한 응용
3. **시각적 검증**: 결과를 눈으로 확인 가능
4. **풍부한 supervision**: 픽셀마다 signal

### 1.3.6 단점

1. **여전히 낮은 성능**: 40% (Supervised 대비 -20~25%p)
2. **색상 모호성**: 여러 색이 가능한 경우 (옷 등)
3. **Gray objects**: 회색 객체는 학습 안 됨
4. **Computational cost**: Decoder 필요

### 1.3.7 Colorization의 의의

Colorization은 **의미적 pretext task의 효과**를 보여주었다. Context Prediction보다 성능이 향상된 이유는 색상이 객체 카테고리와 강하게 연관되어 있기 때문이다. 하지만 여전히 성능 gap이 크고, 색상 모호성 문제가 존재했다.

---

## 1.4 Jigsaw Puzzle (2016)

### 1.4.1 기본 정보

- **논문**: Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles
- **발표**: ECCV 2016
- **저자**: Mehdi Noroozi et al.

### 1.4.2 핵심 원리

Jigsaw Puzzle은 **섞인 패치의 순서를 맞추는** task이다.

**문제 설정**:

1. 이미지를 3×3 = 9개 패치로 분할
2. 패치 순서를 무작위로 섞음
3. 원래 순서(permutation) 예측

$$\mathbf{x} \rightarrow \{\mathbf{p}_1, ..., \mathbf{p}_9\}$$
$$\text{Permute: } \pi \in S_9$$
$$\text{Predict: } \hat{\pi}$$

**Permutation Set 크기**: 9! = 362,880 (너무 큼)
- 실제로는 100개 또는 1000개 permutation만 사용

### 1.4.3 기술적 세부사항

**Architecture**:

Siamese Network (9개 branch):

$$\mathbf{f}_i = f_\theta(\mathbf{p}_i), \quad i = 1, ..., 9$$

**Concatenation**:

$$\mathbf{h} = [\mathbf{f}_1, \mathbf{f}_2, ..., \mathbf{f}_9]$$

**Permutation Classification**:

$$\hat{\pi} = \text{softmax}(\mathbf{W}\mathbf{h} + \mathbf{b})$$

K-way classification (K = 100 or 1000)

**Permutation 선택**:
- Hamming distance 기반으로 diverse한 permutation 선택
- 너무 쉽거나 어려운 것 제외

### 1.4.4 성능

**ImageNet Linear Probing**: 45-60%
- Permutation 수에 따라 차이
- 1000 permutation: ~60%

**PASCAL VOC Transfer**:
- Detection: 45% mAP
- Classification: 70% accuracy

### 1.4.5 왜 효과적인가?

**공간 관계 학습**:
- 패치 순서를 맞추려면 **공간 구조** 이해 필요
- 예: 자동차는 바퀴가 아래, 지붕이 위
- 예: 얼굴은 눈이 위, 입이 아래

**객체 부분 인식**:
- 각 패치가 객체의 어느 부분인지 파악 필요
- 부분-전체 관계 학습

### 1.4.6 장점

1. **향상된 성능**: 60% (Colorization 40% 대비 +20%p)
2. **강력한 pretext**: 공간 구조가 핵심 정보
3. **확장 가능**: Permutation 수 조절로 난이도 조절

### 1.4.7 단점

1. **Permutation 선택**: 어떤 permutation을 쓸지 설계 필요
2. **계산 비용**: 9개 패치 개별 처리
3. **여전히 gap**: Supervised 대비 -15~20%p
4. **Chromatic aberration**: Context Prediction과 유사한 shortcut

### 1.4.8 Jigsaw의 의의

Jigsaw Puzzle은 **Pretext Tasks의 성능을 크게 향상**시켰다. 60%는 당시로서는 획기적이었으며, 공간 구조 학습의 중요성을 입증했다. 많은 후속 연구의 baseline이 되었다.

---

## 1.5 Rotation Prediction (RotNet, 2018)

### 1.5.1 기본 정보

- **논문**: Unsupervised Representation Learning by Predicting Image Rotations
- **발표**: ICLR 2018
- **저자**: Spyros Gidaris et al.

### 1.5.2 핵심 원리

RotNet은 **이미지 회전 각도를 예측**하는 가장 단순한 pretext task 중 하나이다.

**문제 설정**:

이미지를 0°, 90°, 180°, 270° 중 하나로 회전:

$$\mathbf{x}_r = \text{Rotate}(\mathbf{x}, r), \quad r \in \{0°, 90°, 180°, 270°\}$$

**4-way Classification**:

$$\min_\theta \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}_r), r)$$

### 1.5.3 기술적 세부사항

**Architecture**:

단순한 CNN (ResNet, AlexNet 등):

$$\mathbf{z} = \text{Encoder}(\mathbf{x}_r)$$
$$\hat{r} = \text{softmax}(\mathbf{W}\mathbf{z} + \mathbf{b})$$

**Loss**: Cross-Entropy (4-way)

$$\mathcal{L} = -\sum_{k=0}^{3} \mathbb{1}[r=k\cdot90°] \log p(k | \mathbf{x}_r)$$

**Training**:
- 각 이미지마다 4개 회전 버전 생성
- Batch에서 uniform sampling

### 1.5.4 왜 효과적인가?

**방향 인식**:
- 회전 각도를 맞추려면 **물체의 방향** 이해 필요
- 예: 사람은 머리가 위, 자동차는 바퀴가 아래

**구조 학습**:
- 객체의 **상하좌우 구조** 파악
- 대칭성, 비대칭성 이해

**간단함의 힘**:
- 4-way classification (매우 단순)
- 추가 설계 불필요 (permutation 선택 등)

### 1.5.5 성능

**ImageNet Linear Probing**: 54-55%
- AlexNet backbone 기준
- ResNet-50: 48-50% (당시 연구)

**CIFAR-10/100**:
- Linear probing: 85% (CIFAR-10)
- Fine-tuning: 95% (CIFAR-10)

### 1.5.6 장점

1. **극단적 단순함**: 회전 4개만
2. **구현 용이**: 몇 줄 코드
3. **빠른 학습**: Augmentation overhead 적음
4. **해석 가능**: 방향 이해가 명확

### 1.5.7 단점

1. **제한적 성능**: 55% (Jigsaw 60% 대비 -5%p)
2. **Task가 너무 단순**: 4개 클래스만
3. **Rotation-invariant objects**: 대칭 객체는 학습 안 됨
4. **Low-level cue**: 텍스처 방향으로 추측 가능

### 1.5.8 RotNet의 의의

RotNet은 **"단순함도 효과적이다"**를 보여주었다. Jigsaw의 복잡한 permutation 설계 없이도 55%를 달성했다. 많은 연구에서 빠른 baseline으로 사용되었으며, 교육 목적으로도 유용하다.

---

## 1.6 Multi-Task Pretext (2017-2018)

### 1.6.1 핵심 아이디어

여러 Pretext Task를 **동시에 학습**하면 더 나은가?

**Multi-Task Learning**:

$$\mathcal{L}_{\text{total}} = \sum_{i=1}^{K} \lambda_i \mathcal{L}_i$$

여기서:
- $\mathcal{L}_i$: i번째 pretext task loss
- $\lambda_i$: Task weight

**예시 조합**:
- Rotation + Jigsaw
- Colorization + Context
- Rotation + Exemplar

### 1.6.2 주요 연구

**Doersch et al. (2017)**:
- Context + Colorization
- 성능: ~43% (각각보다 높음)

**Noroozi et al. (2018)**:
- Jigsaw + Counting (object counting)
- 성능: ~65%

### 1.6.3 성능

**Multi-task의 효과**:

| Task | Single | Multi-task | 개선 |
|------|--------|-----------|------|
| Rotation | 55% | 58-60% | +3-5%p |
| Jigsaw | 60% | 63-65% | +3-5%p |
| Context + Color | 40% | 43% | +3%p |

**일관된 패턴**: Multi-task가 3-5%p 향상

### 1.6.4 왜 효과적인가?

**Complementary Information**:
- 각 task가 다른 측면의 정보 학습
- Rotation: 방향
- Jigsaw: 공간 구조
- Colorization: 의미

**Regularization 효과**:
- 한 task에 과적합 방지
- 더 범용적인 표현

### 1.6.5 단점

1. **학습 복잡도**: Multiple loss balancing
2. **하이퍼파라미터**: Task weight $\lambda_i$ 튜닝
3. **계산 비용**: 여러 head 동시 학습
4. **여전히 gap**: 65% (Supervised 76.5% 대비 -11.5%p)

---

## 1.7 Discriminative 패러다임 종합 비교

### 1.7.1 기술적 진화 과정

```
Context Prediction (2015)
├─ 혁신: 최초의 본격적 pretext task
├─ 문제: Chromatic aberration, 낮은 성능
└─ 성능: 30-35%

        ↓ 의미적 task

Colorization (2016)
├─ 개선: 색상 = 의미적 정보
├─ 문제: 모호성, 여전히 낮음
└─ 성능: 38-40%

        ↓ 공간 구조

Jigsaw Puzzle (2016)
├─ 혁신: 공간 관계 학습
├─ 돌파: 60% 달성
└─ 성능: 45-60%

        ↓ 단순화

Rotation (2018)
├─ 단순화: 4-way classification
├─ 장점: 구현 쉬움
└─ 성능: 54-55%

        ↓ 조합

Multi-Task (2017-2018)
├─ 조합: 여러 task 동시 학습
├─ 개선: 3-5%p 향상
└─ 성능: 63-65% (최고)

        ↓ 패러다임 전환

Contrastive Learning (2018→)
└─ SimCLR, MoCo: 70-76% (Discriminative 대체)
```

### 1.7.2 상세 비교표

| 비교 항목 | Context | Colorization | Jigsaw | Rotation | Multi-Task |
|----------|---------|-------------|--------|----------|------------|
| **연도** | 2015 | 2016 | 2016 | 2018 | 2017-18 |
| **Pretext Task** | 패치 위치 | 색상 복원 | 패치 순서 | 회전 각도 | 조합 |
| **분류 수** | 8-way | 313-way | 100-1000 | 4-way | Multiple |
| **ImageNet Linear** | 30-35% | 38-40% | 45-60% | 54-55% | 63-65% |
| **구현 복잡도** | 중간 | 높음 | 높음 | **매우 낮음** | 높음 |
| **학습 시간** | 중간 | 느림 | 중간 | **빠름** | 느림 |
| **핵심 학습** | 공간 관계 | 의미 | 공간 구조 | 방향 | 종합 |
| **Shortcut 문제** | 심각 | 중간 | 중간 | 적음 | 적음 |
| **확장성** | 낮음 | 중간 | 중간 | 낮음 | 중간 |
| **교육 가치** | 높음 | 높음 | 중간 | **매우 높음** | 낮음 |
| **실용성** | 낮음 | 낮음 | 중간 | 낮음 | 중간 |
| **종합 평가** | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |

### 1.7.3 핵심 한계 분석

**1) Task 설계 의존성**

모든 Discriminative 방법의 근본적 한계:
- "좋은" pretext task 찾기 어려움
- 수작업 설계 필요
- Task마다 다른 성능

**2) Low-level Shortcut Learning**

네트워크가 의도와 다르게 학습:
- Context: 패치 경계 색상 차이
- Jigsaw: Chromatic aberration
- Rotation: 텍스처 방향

**3) Task와 Downstream의 Gap**

Pretext task ≠ Downstream task:
- Rotation 잘 맞춰도 Classification 잘한다는 보장 없음
- 간접적인 학습

**4) 성능 정체**

Multi-task로도 65%:
- Supervised 76.5% 대비 -11.5%p
- 실용성 부족

### 1.7.4 Discriminative vs Contrastive

**패러다임 전환 (2018-2020)**:

| 측면 | Discriminative | Contrastive |
|------|---------------|------------|
| **원리** | Pretext task 예측 | Instance 유사도 |
| **설계** | 수작업 task | 자동 (augmentation) |
| **성능** | 55-65% | 70-76% |
| **일반성** | Task-specific | Task-agnostic |
| **확장성** | 제한적 | 높음 |

**왜 Contrastive가 이겼는가?**

1. **일반성**: Task 설계 불필요, augmentation만
2. **성능**: 10-15%p 향상
3. **이론**: Instance discrimination이 더 직접적
4. **확장**: Large batch, momentum encoder 등 발전 가능

### 1.7.5 실무 적용 (현재 시점)

**Discriminative 방식의 현재 위치**:

**사용하지 않음** (★☆☆☆☆):
- Contrastive, MIM이 훨씬 우수
- 역사적 의의만 존재

**예외: 교육 목적** (★★★☆☆):
- **Rotation**: SSL 개념 이해에 최적
  - 구현 간단 (10-20줄 코드)
  - 빠른 학습 (1-2시간)
  - 직관적 결과

**교육 커리큘럼**:
```
Step 1: RotNet 구현 (1일)
- 4-way rotation classification
- ResNet-18 backbone
- CIFAR-10에서 빠른 실험

Step 2: 한계 체험 (1일)
- Linear probing: ~55%
- Supervised와 비교: -20%p gap
- Shortcut learning 관찰

Step 3: Contrastive로 전환 (1주)
- SimCLR 구현
- 70%+ 달성
- 패러다임 전환 이해
```

---

## 부록: 관련 테이블

### A.1 Discriminative vs 다른 패러다임

| 패러다임 | 대표 모델 | Linear | Fine-tuning | 주요 장점 | 주요 단점 |
|---------|----------|--------|-------------|----------|----------|
| **Discriminative** | Jigsaw/Rotation | 55-65% | - | 직관적, 교육적 | 낮은 성능 |
| **Clustering** | DINO | **80%** | 84.5% | 해석성, Few-shot | 복잡한 학습 |
| Contrastive | MoCo v3 | 76% | 84.1% | Instance | Large batch |
| **Generative** | MAE | 68% | **85.9%** | Dense pred | Linear 낮음 |
| Hybrid | DINOv2 | **86%** | **86%** | SOTA | 계산 비용 |

### A.2 Pretext Task별 세부 비교

| Task | 입력 | 출력 | 학습 내용 | 성능 | 난이도 |
|------|------|------|----------|------|--------|
| **Context** | 2 patches | 8-way | 공간 관계 | 30-35% | 중간 |
| **Colorization** | Gray | 313-way | 의미 | 38-40% | 높음 |
| **Jigsaw** | 9 patches | 100-1000 | 구조 | 45-60% | 높음 |
| **Rotation** | Image | 4-way | 방향 | 54-55% | **낮음** |
| **Multi-Task** | Various | Multiple | 종합 | 63-65% | 매우 높음 |

### A.3 Shortcut Learning 분석

| Task | Shortcut | 해결책 | 효과 |
|------|----------|--------|------|
| **Context** | Chromatic aberration | Gap between patches | 부분 해결 |
| **Colorization** | Gray objects | Class rebalancing | 부분 해결 |
| **Jigsaw** | Edge artifacts | Color jittering | 부분 해결 |
| **Rotation** | Texture orientation | - | 여전히 존재 |

### A.4 교육용 구현 가이드 (RotNet)

**최소 구현 (PyTorch)**:

```python
# 1. Rotation Augmentation
def rotate_batch(images):
    """
    images: (B, C, H, W)
    return: rotated images, labels
    """
    B = images.size(0)
    rotations = torch.randint(0, 4, (B,))
    
    rotated = []
    for img, rot in zip(images, rotations):
        angle = rot * 90
        rotated.append(TF.rotate(img, angle))
    
    return torch.stack(rotated), rotations

# 2. Model
class RotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(pretrained=False)
        self.encoder.fc = nn.Linear(512, 4)  # 4-way
    
    def forward(self, x):
        return self.encoder(x)

# 3. Training Loop
model = RotNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for images, _ in dataloader:
    rotated, labels = rotate_batch(images)
    outputs = model(rotated)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. Linear Probing
# Freeze encoder, train only FC for classification
model.encoder.fc = nn.Linear(512, num_classes)
for param in model.encoder.parameters():
    param.requires_grad = False
model.encoder.fc.weight.requires_grad = True
model.encoder.fc.bias.requires_grad = True
```

**실험 체크리스트**:
- [ ] CIFAR-10에서 rotation 학습 (1-2시간)
- [ ] Rotation accuracy 확인 (>95% 달성 가능)
- [ ] Linear probing (CIFAR-10 분류)
- [ ] 성능 측정 (~85% 예상)
- [ ] Supervised와 비교 (~95% vs 85%)

### A.5 역사적 타임라인

```
2015: Context Prediction (ICCV)
      └─ SSL 가능성 제시, 30-35%

2016: Colorization (ECCV), Jigsaw (ECCV)
      └─ 의미적 학습, 공간 구조, 40-60%

2017: Multi-Task 시도
      └─ 조합의 효과, 63-65%

2018: Rotation (ICLR)
      └─ 단순화, 55%
      
      ↓ 패러다임 전환
      
2018: InstDisc (CVPR)
      └─ Contrastive 시작, 60%

2020: SimCLR (ICML), MoCo (CVPR)
      └─ Contrastive 혁명, 70-71%
      └─ Discriminative 완전 대체
```

### A.6 주요 논문 및 인용수

| 논문 | 연도 | 인용수 | 중요도 |
|------|------|--------|--------|
| **Context Prediction** | 2015 | 2000+ | 역사적 |
| **Colorization** | 2016 | 3000+ | 높음 |
| **Jigsaw Puzzle** | 2016 | 2500+ | 높음 |
| **Rotation** | 2018 | 1500+ | 교육적 |
| **SimCLR** (비교) | 2020 | 10000+ | 매우 높음 |

---

## 결론

Discriminative (Pretext Tasks) 패러다임은 **Self-Supervised Learning의 개척자**였다. 2015-2018년 동안 다양한 pretext task가 제안되었고, 성능은 30%에서 65%까지 향상되었다.

**핵심 기여**:
1. **가능성 제시**: 레이블 없이도 학습 가능
2. **다양한 시도**: Context, Color, Jigsaw, Rotation 등
3. **한계 발견**: Task 설계 의존, Shortcut learning

**근본적 한계**:
1. **낮은 성능**: 최대 65% (Supervised 76.5% 대비 -11.5%p)
2. **Task 설계**: 수작업 필요, 일반성 부족
3. **Task gap**: Pretext ≠ Downstream

**패러다임 전환 (2018-2020)**:

Discriminative → Contrastive:
- Task 설계 불필요 (자동 augmentation)
- 성능 향상 (65% → 70-76%)
- 일반성 확보 (Instance discrimination)

**현재의 역할**:
- **실무**: 사용하지 않음 (Contrastive/MIM으로 대체)
- **교육**: Rotation 등으로 SSL 개념 학습
- **역사**: SSL의 초석, 많은 아이디어 영감

**최종 평가**:
- 역사적 의의: ★★★★★
- 실용적 가치: ★☆☆☆☆
- 교육적 가치: ★★★★☆

Discriminative 방식은 "좋은 실패"였다. 성능 면에서는 Contrastive에 밀렸지만, Self-Supervised Learning의 길을 열었고, 많은 후속 연구에 영감을 주었다. 특히 **Rotation Prediction**은 교육 목적으로 여전히 유용하다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: Context Prediction, Colorization, Jigsaw Puzzle, Rotation Prediction, Multi-Task

**주요 내용**:
1. Discriminative 패러다임 개요
2. Context Prediction (2015) - SSL의 시작
3. Colorization (2016) - 의미적 학습
4. Jigsaw Puzzle (2016) - 공간 구조
5. Rotation Prediction (2018) - 단순화
6. Multi-Task (2017-18) - 조합
7. 종합 비교 및 패러다임 전환 분석
8. 교육적 활용 가이드 (RotNet)
9. **부록**: 역사적 타임라인, 구현 가이드

