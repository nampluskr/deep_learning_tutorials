# 3. Knowledge Distillation 방식 상세 분석

## 3.1 패러다임 개요

Knowledge Distillation 기반 이상 탐지는 Teacher-Student 프레임워크를 사용한다. Pre-trained teacher 네트워크가 정상 데이터의 특징을 추출하고, student 네트워크는 정상 데이터에서 teacher의 지식을 모방하도록 학습된다.

**핵심 원리**:

$$\text{Anomaly Score} = \|f_T(\mathbf{x}) - f_S(\mathbf{x})\|$$

여기서:
- $f_T(\mathbf{x})$: Teacher 네트워크의 출력 (frozen)
- $f_S(\mathbf{x})$: Student 네트워크의 출력 (learnable)

**학습 목표**:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{normal}}} \|f_T(\mathbf{x}) - f_S(\mathbf{x})\|^2$$

**이상 탐지 가정**:
- 정상 샘플: $f_T(\mathbf{x}) \approx f_S(\mathbf{x})$ (모방 성공)
- 이상 샘플: $f_T(\mathbf{x}) \neq f_S(\mathbf{x})$ (모방 실패, 학습하지 못한 패턴)

---

## 3.2 STFPM (2021)

### 3.2.1 기본 정보

- **논문**: Student-Teacher Feature Pyramid Matching for Anomaly Detection
- **발표**: BMVC 2021
- **저자**: Hanqiu Wang et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/stfpm

### 3.2.2 핵심 원리

STFPM은 Feature Pyramid Network(FPN) 구조에서 teacher와 student의 multi-scale 특징을 매칭한다.

**Multi-scale Feature Matching**:

$$\mathcal{L} = \sum_{l=1}^{L} \|f_T^{(l)}(\mathbf{x}) - f_S^{(l)}(\mathbf{x})\|_2^2$$

여기서:
- $l$: 레이어 인덱스 (layer1, layer2, layer3)
- $L$: 총 레이어 수 (보통 3)

**Pyramid의 의미**:
- Layer 1 (낮은 레벨): 세밀한 텍스처, 엣지
- Layer 2 (중간 레벨): 중간 수준 패턴
- Layer 3 (높은 레벨): 의미적(semantic) 정보

### 3.2.3 기술적 세부사항

**Network Architecture**:
- Teacher: Pre-trained ResNet18 (frozen, $\nabla_\theta f_T = 0$)
- Student: ResNet18 (learnable, random initialization)

**Multi-scale Anomaly Map**:

각 레이어에서:
$$A^{(l)}_{i,j} = \|f_T^{(l)}_{i,j} - f_S^{(l)}_{i,j}\|_2$$

최종 anomaly map:
$$A = \frac{1}{L}\sum_{l=1}^{L} \text{Upsample}(A^{(l)})$$

**Image-level Score**:
$$\text{Score} = \max_{i,j} A_{i,j}$$

### 3.2.4 성능

**MVTec AD 벤치마크**:
- Image AUROC: 96.8%
- Pixel AUROC: 96.2%
- 추론 속도: 20-40ms per image
- 메모리: 500MB-1GB

### 3.2.5 장점

1. **간단한 구조**: Teacher-student만으로 구성
2. **빠른 추론**: Forward pass만 필요 (20-40ms)
3. **End-to-end 학습**: 단일 loss로 학습
4. **Multi-scale**: 다양한 크기 이상 탐지
5. **Pre-trained 활용**: ImageNet 지식 전이

### 3.2.6 단점

1. **Teacher 품질 의존**: Pre-trained 모델의 품질에 성능 의존
2. **중간 성능**: SOTA 대비 낮은 정확도 (96.8%)
3. **Domain gap**: ImageNet과 산업 이미지 간 차이
4. **단순 매칭**: 복잡한 패턴 학습 제한

---

## 3.3 FRE (2023)

### 3.3.1 기본 정보

- **논문**: A Fast Method For Anomaly Detection And Segmentation
- **발표**: BMVC 2023
- **저자**: Anonymous
- **역할**: Knowledge Distillation 패러다임 내 속도 최적화 시도

### 3.3.2 핵심 원리

FRE는 STFPM의 Feature Pyramid Matching 구조를 유지하되, **추론 속도 최적화**에 집중한 경량화 시도이다.

**최적화 방향**:
1. 경량화된 feature extractor
2. 간소화된 feature matching pipeline
3. Efficient anomaly score calculation

### 3.3.3 STFPM 대비 개선 시도

| 측면 | STFPM | FRE | 개선 시도 |
|------|-------|-----|----------|
| **추론 속도** | 20-40ms | 10-30ms | 약 2배 향상 |
| **구조** | ResNet18 | 경량화된 backbone | 간소화 |
| **메모리** | 500MB-1GB | 300-500MB | 감소 |
| **Image AUROC** | 96.8% | 95-96% | -0.8~1.8%p 하락 |

### 3.3.4 한계 및 현재 상태

**한계**:
1. **성능 저하**: 96.8% → 95-96% (-0.8~1.8%p)
2. **제한적 속도 개선**: 2배 미만의 향상
3. **근본적 돌파구 부재**: 점진적 개선에 그침
4. **EfficientAd 등장**: 2024년 EfficientAd가 1-5ms로 혁명적 속도 달성

**역사적 의의**:
- STFPM에서 EfficientAd로 가는 **과도기적 모델**
- Knowledge Distillation 패러다임 내에서 속도 최적화의 "징검다리" 역할
- 점진적 개선만으로는 실무 임팩트가 제한적임을 보여줌

**현재 상태**: 
- **Deprecated** (사용 비추천)
- EfficientAd로 완전히 대체됨
- 학술적 참고용으로만 가치

**교훈**:
- 점진적 개선(2배)만으로는 실무 채택 어려움
- 혁명적 발전(20-40배, EfficientAd)이 패러다임을 바꿈
- 속도와 성능은 trade-off가 아닐 수 있음

### 3.3.5 성능

**MVTec AD 벤치마크**:
- Image AUROC: 95-96%
- Pixel AUROC: 94-95%
- 추론 속도: 10-30ms
- **추천도**: ★☆☆☆☆ (사용 비추천)

---

## 3.4 Reverse Distillation (2022)

### 3.4.1 기본 정보

- **논문**: Anomaly Detection via Reverse Distillation from One-Class Embedding
- **발표**: CVPR 2022
- **저자**: Hanqiu Deng et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/reverse_distillation

### 3.4.2 핵심 원리

Reverse Distillation은 **전통적인 knowledge distillation을 역전**시킨다.

**패러다임 역전**:

전통적 KD:
$$\text{Teacher (복잡)} \rightarrow \text{Student (단순)}$$

Reverse KD:
$$\text{Teacher (단순)} \leftarrow \text{Student (복잡)}$$

**One-class Embedding**:

Teacher가 정상 데이터의 압축된 표현 생성:
$$\mathbf{z}_T = E_T(\mathbf{x}), \quad \mathbf{z}_T \in \mathbb{R}^d$$

Student가 embedding을 역으로 재구성:
$$\hat{\mathbf{z}}_T = D_S(E_S(\mathbf{x}))$$

**Loss Function**:

$$\mathcal{L} = \mathcal{L}_{\text{cos}}(\mathbf{z}_T, \hat{\mathbf{z}}_T) + \lambda \mathcal{L}_{\text{L2}}(\mathbf{z}_T, \hat{\mathbf{z}}_T)$$

Cosine similarity loss:
$$\mathcal{L}_{\text{cos}} = 1 - \frac{\mathbf{z}_T \cdot \hat{\mathbf{z}}_T}{\|\mathbf{z}_T\| \|\hat{\mathbf{z}}_T\|}$$

### 3.4.3 STFPM 대비 핵심 차이점

| 측면 | STFPM | Reverse Distillation | 개선 효과 |
|------|-------|---------------------|----------|
| **Teacher 구조** | 복잡 (ResNet18) | 단순 (Encoder only) | One-class embedding 생성 |
| **Student 구조** | 단순 (동일 ResNet) | 복잡 (Encoder-Decoder) | 강력한 재구성 능력 |
| **학습 방향** | Teacher → Student (모방) | Student → Teacher (역재구성) | 패러다임 전환 |
| **특징 표현** | Multi-scale features | One-class embedding | 정상 패턴 압축 |
| **Loss 함수** | L2 | Cosine + L2 | 더 강건한 학습 |
| **Image AUROC** | 96.8% | 98.6% | +1.8%p |
| **Pixel AUROC** | 96.2% | 98.5% | +2.3%p |

### 3.4.4 기술적 세부사항

**One-class Encoder (Teacher)**:

정상 데이터만을 표현하는 압축된 embedding:
$$\mathbf{z}_T = \text{Projection}(\text{Backbone}(\mathbf{x}))$$

L2 normalization (hypersphere):
$$\mathbf{z}_T \leftarrow \frac{\mathbf{z}_T}{\|\mathbf{z}_T\|_2}$$

**Multi-scale Decoder (Student)**:

서로 다른 receptive field를 가진 decoder branches:
- Branch 1: Small RF (세밀한 결함)
- Branch 2: Medium RF
- Branch 3: Large RF (큰 결함)

**Anomaly Score**:

각 스케일에서:
$$S_s = 1 - \text{sim}(\mathbf{z}_T, \hat{\mathbf{z}}_T^{(s)}) + \alpha \|\mathbf{z}_T - \hat{\mathbf{z}}_T^{(s)}\|_2$$

최종 점수:
$$\text{Score} = \max_s S_s$$

### 3.4.5 왜 역전이 효과적인가?

**이론적 우수성**:

STFPM:
- Teacher가 ImageNet의 일반 특징 학습
- 일반 특징이 이상 탐지에 최적이 아닐 수 있음

Reverse Distillation:
- Teacher가 **타겟 도메인의 정상 패턴만 압축**
- Student가 이 압축된 표현을 복원
- 타겟 도메인에 특화된 표현 학습

**실증적 효과**:
- 정상: 재구성 성공 (학습됨)
- 이상: 재구성 실패 (학습 안됨)
- False Positive 감소

### 3.4.6 성능

**MVTec AD 벤치마크**:
- Image AUROC: 98.6% (SOTA급)
- Pixel AUROC: 98.5% (최고 수준)
- 추론 속도: 100-200ms
- 메모리: 500MB-1GB
- 학습 시간: 3-5시간

### 3.4.7 장점

1. **높은 정확도**: SOTA급 성능 (98.6%)
2. **우수한 Localization**: 정밀한 pixel-level 탐지 (98.5%)
3. **강건성**: False Positive 낮음
4. **타겟 특화**: 도메인에 최적화된 표현
5. **이론적 근거**: One-class learning의 명확한 원리

### 3.4.8 단점

1. **복잡한 구조**: Encoder-Decoder + Multi-scale
2. **느린 추론**: STFPM 대비 2-3배 느림 (100-200ms)
3. **메모리 사용**: 더 큰 student network
4. **학습 시간**: 3-5시간 소요

---

## 3.5 EfficientAd (2024)

### 3.5.1 기본 정보

- **논문**: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
- **발표**: WACV 2024
- **저자**: Kilian Batzner et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/efficient_ad

### 3.5.2 핵심 원리

EfficientAd는 Knowledge Distillation과 Auto-encoder를 결합하고, **극단적인 최적화**를 통해 millisecond 레벨의 추론 속도를 달성한다.

**핵심 설계 철학**:
1. Student-Teacher 구조 유지 (효과성)
2. 경량 네트워크 설계 (속도)
3. Patch Description Network 추가 (정확도)
4. Auto-encoder 통합 (재구성 기반 탐지 추가)

**Patch Description Network (PDN)**:

극도로 경량화된 네트워크:
$$\text{PDN}: \mathbb{R}^{3 \times H \times W} \rightarrow \mathbb{R}^{384 \times H/16 \times W/16}$$

파라미터 수: ~50K (vs ResNet18: ~11M)

### 3.5.3 이전 모델 대비 핵심 차이점

| 측면 | STFPM | FRE | Reverse Distillation | EfficientAd | 개선 효과 |
|------|-------|-----|---------------------|-------------|----------|
| **아키텍처** | ResNet18 | 경량 ResNet | WideResNet + Decoder | PDN (경량) | 10배+ 경량화 |
| **Teacher** | Pre-trained | Pre-trained | One-class Encoder | Pre-trained + Fine-tuned | 균형 |
| **Student** | ResNet18 | 경량 ResNet | 복잡한 Decoder | 초경량 PDN | 속도↑↑ |
| **추가 모듈** | 없음 | 없음 | 없음 | Auto-encoder | 정확도 보완 |
| **추론 속도** | 20-40ms | 10-30ms | 100-200ms | 1-5ms | 20-200배 향상 |
| **Image AUROC** | 96.8% | 95-96% | 98.6% | 97.8% | 균형 |
| **하드웨어** | GPU | GPU | GPU | GPU/CPU | CPU 가능 |

### 3.5.4 기술적 세부사항

**PDN Architecture**:

극도로 경량화된 4-layer CNN:
```
Input (3, 256, 256)
  ↓ Conv 4×4, stride=2
(32, 128, 128)
  ↓ Conv 4×4, stride=2
(64, 64, 64)
  ↓ Conv 4×4, stride=2
(128, 32, 32)
  ↓ Conv 4×4, stride=2
(384, 16, 16)
```

총 파라미터: ~50K

**Hybrid Loss**:

$$\mathcal{L} = \mathcal{L}_{\text{KD}} + \lambda \mathcal{L}_{\text{AE}}$$

Knowledge Distillation:
$$\mathcal{L}_{\text{KD}} = \|f_T(\mathbf{x}) - f_S(\mathbf{x})\|_2^2$$

Auto-encoder:
$$\mathcal{L}_{\text{AE}} = \|\mathbf{x} - \text{Dec}(\text{Enc}(\mathbf{x}))\|_2^2$$

**Anomaly Score**:

$$\text{Score} = \alpha \cdot \text{Score}_{\text{KD}} + (1-\alpha) \cdot \text{Score}_{\text{AE}}$$

보통 $\alpha = 0.7$

### 3.5.5 속도 최적화 기법

**1) Model Quantization (INT8)**:
- FP32 → INT8: 4배 메모리 감소, 2배 속도 향상

**2) ONNX Export**:
- PyTorch → ONNX: 최적화된 추론

**3) TensorRT (GPU)**:
- NVIDIA TensorRT로 추가 가속화

**4) Half Precision (FP16)**:
- 2배 빠름, 메모리 절반

### 3.5.6 이전 모델 대비 개선사항

**1) 추론 속도 혁명**:
```
STFPM: 20-40ms
FRE: 10-30ms (2배 개선)
Reverse Distillation: 100-200ms
EfficientAd: 1-5ms (4-200배 개선)
```

**2) CPU 추론 가능**:
- STFPM/FRE/RD: GPU 필수
- EfficientAd: CPU에서도 10-20ms 수준
- **효과**: 엣지 디바이스 배포 가능

**3) 메모리 효율**:
```
STFPM: ~500MB
FRE: ~400MB
Reverse Distillation: ~1GB
EfficientAd: <200MB (60-80% 감소)
```

**4) 성능 trade-off**:
```
정확도: 97.8%
- STFPM(96.8%)보다 높음
- FRE(95-96%)보다 높음
- Reverse Distillation(98.6%)보다 약간 낮음 (-0.8%p)
- 하지만 속도 20-200배 향상으로 충분히 보상
```

**5) 실용성**:
- **실시간 처리**: 200-1000 FPS 가능
- **엣지 배포**: Raspberry Pi, 모바일에서도 동작
- **비용 절감**: 저렴한 하드웨어 사용 가능

### 3.5.7 성능

**MVTec AD 벤치마크**:
- Image AUROC: 97.8%
- Pixel AUROC: 97.2%
- 추론 속도: 1-5ms (GPU), 10-20ms (CPU)
- 메모리: <200MB
- FPS: 200-1000 (GPU), 50-100 (CPU)

### 3.5.8 장점

1. **극한의 속도**: 1-5ms, 실시간 처리
2. **CPU 가능**: GPU 없이도 동작
3. **낮은 메모리**: <200MB
4. **좋은 정확도**: 97.8% AUROC (실용 충분)
5. **경량 구조**: 엣지 디바이스 배포
6. **하이브리드**: KD + AE로 강건성

### 3.5.9 단점

1. **최고 정확도 아님**: Reverse Distillation 대비 낮음
2. **복잡한 최적화**: Quantization, ONNX 등 추가 작업
3. **Fine-tuning 필요**: Teacher fine-tuning 단계

---

## 3.6 Knowledge Distillation 내 속도 최적화 발전 과정

### 3.6.1 단계적 발전

```
STFPM (2021)
├─ 시작: Feature Pyramid Matching
├─ 속도: 20-40ms
├─ 성능: 96.8% AUROC
└─ 기여: KD 패러다임 확립

        ↓ 과도기적 시도 (2023)

FRE (2023)
├─ 목표: 속도 2배 향상
├─ 속도: 10-30ms (약 2배)
├─ 성능: 95-96% (-0.8~1.8%p 하락)
├─ 한계: 제한적 개선, 성능 저하
└─ 결과: Deprecated

        ↓ 혁명적 발전 (2024)

EfficientAd (2024)
├─ 혁신: PDN + AE 하이브리드
├─ 속도: 1-5ms (20-200배)
├─ 성능: 97.8% (FRE보다 높음)
└─ 영향: 실시간 처리 현실화
```

### 3.6.2 속도 최적화 경로 분석

**점진적 개선 vs 혁명적 발전**:

| 단계 | 모델 | 속도 | 성능 | 접근 방식 | 결과 |
|------|------|------|------|----------|------|
| **1단계** | STFPM | 20-40ms | 96.8% | FPN matching | Baseline |
| **2단계** | FRE | 10-30ms | 95-96% | 경량화 시도 | 실패 (deprecated) |
| **3단계** | EfficientAd | 1-5ms | 97.8% | 근본적 재설계 | 성공 (혁명) |

**교훈**:
1. 점진적 개선(2배)만으로는 패러다임 전환 불가
2. 혁명적 발전(20-40배)이 실무 채택 이끌어냄
3. 속도와 성능은 trade-off가 아닐 수 있음 (EfficientAd가 증명)

---

## 3.7 Knowledge Distillation 방식 종합 비교

### 3.7.1 기술적 진화 과정

```
STFPM (2021)
├─ 시작: Teacher-Student 패러다임 도입
├─ 구조: Teacher(복잡) → Student(동일)
├─ 성능: 96.8% AUROC, 20-40ms
└─ 한계: 중간 수준 정확도

        ↓ 패러다임 역전 (2022)

Reverse Distillation (2022)
├─ 혁신: Teacher(단순) ← Student(복잡)
├─ One-class embedding 재구성
├─ 성능: 98.6% AUROC (SOTA급)
├─ 속도: 100-200ms
└─ 한계: 느린 속도, 높은 리소스

        ↓ 속도 최적화 시도 (2023)

FRE (2023) [과도기]
├─ 목표: STFPM 경량화
├─ 성능: 95-96% (-0.8~1.8%p)
├─ 속도: 10-30ms (2배 개선)
└─ 결과: Deprecated

        ↓ 실용화 혁명 (2024)

EfficientAd (2024)
├─ 목표: 속도 + 정확도 균형
├─ 구조: 경량 PDN + Auto-encoder
├─ 성능: 97.8% AUROC
├─ 속도: 1-5ms (20-200배)
└─ 혁신: 실시간 처리, 엣지 배포
```

### 3.7.2 상세 비교표

| 비교 항목 | STFPM | FRE | Reverse Distillation | EfficientAd |
|----------|-------|-----|---------------------|-------------|
| **발표 연도** | 2021 | 2023 | 2022 | 2024 |
| **Teacher 구조** | ResNet18 (pre-trained) | 경량 backbone | 단순 Encoder (one-class) | PDN (fine-tuned) |
| **Student 구조** | ResNet18 (학습) | 경량 backbone | 복잡한 Encoder-Decoder | 경량 PDN |
| **Teacher 크기** | 11M params | ~5M params | 5-10M params | 50K params |
| **Student 크기** | 11M params | ~5M params | 15-20M params | 50K params |
| **학습 방향** | T → S (모방) | T → S (모방) | S → T (역재구성) | T ← → S + AE |
| **특징 표현** | Multi-scale features | Multi-scale features | One-class embedding | Patch descriptors |
| **추가 모듈** | 없음 | 없음 | 없음 | Auto-encoder |
| **Loss 함수** | MSE (L2) | MSE (L2) | Cosine + L2 | MSE + Recon |
| **Image AUROC** | 96.8% | 95-96% | 98.6% ★★★★★ | 97.8% ★★★★☆ |
| **Pixel AUROC** | 96.2% | 94-95% | 98.5% ★★★★★ | 97.2% ★★★★☆ |
| **추론 속도 (GPU)** | 20-40ms | 10-30ms | 100-200ms | 1-5ms ★★★★★ |
| **추론 속도 (CPU)** | 불가능 | 불가능 | 불가능 | 10-20ms ★★★★★ |
| **메모리 사용** | 500MB-1GB | 300-500MB | 500MB-1GB | <200MB ★★★★★ |
| **학습 시간** | 1-2시간 | 1-2시간 | 3-5시간 | 2-3시간 |
| **FPS (GPU)** | 25-50 | 33-100 | 5-10 | 200-1000 ★★★★★ |
| **하드웨어 요구** | GPU 권장 | GPU 권장 | GPU 필수 | GPU/CPU ★★★★★ |
| **엣지 배포** | 어려움 | 어려움 | 불가능 | 가능 ★★★★★ |
| **구현 난이도** | 낮음 | 낮음 | 높음 | 중간 |
| **주요 혁신** | FPN matching | 경량화 시도 | 역방향 증류 | 극한 최적화 |
| **현재 상태** | Baseline | **Deprecated** | 정밀 검사용 | **표준** |
| **적합 환경** | 일반 검사 | **사용 안함** | 정밀 검사 | 실시간/엣지 |
| **종합 평가** | ★★★☆☆ | ★☆☆☆☆ | ★★★★★ | ★★★★★ |

---

## 부록: 관련 테이블

### A.1 Knowledge Distillation vs 다른 패러다임

| 패러다임 | 대표 모델 | Image AUROC | 추론 속도 | 주요 장점 | 주요 단점 |
|---------|----------|-------------|-----------|----------|----------|
| **KD (정밀)** | Reverse Distillation | 98.6% | 100-200ms | SOTA급 정확도 | 느림 |
| **KD (실시간)** | EfficientAd | 97.8% | 1-5ms | 극한 속도, 엣지 | 중간 정확도 |
| **KD (Deprecated)** | FRE | 95-96% | 10-30ms | - | 성능 저하 |
| Memory-Based | PatchCore | 99.1% | 50-100ms | 최고 정확도 | 메모리 |
| Normalizing Flow | FastFlow | 98.5% | 20-50ms | 확률적 해석 | 학습 복잡 |
| Reconstruction | DRAEM | 97.5% | 50-100ms | Few-shot | Simulation |

### A.2 응용 시나리오별 KD 모델 선택

| 시나리오 | 권장 모델 | 이유 | 예상 성능 |
|---------|----------|------|----------|
| **정밀 검사 (반도체, 의료)** | Reverse Distillation | SOTA급 정확도 | 98.6% AUROC |
| **고속 생산 라인** | EfficientAd | 실시간 처리 (1-5ms) | 97.8% AUROC |
| **엣지 디바이스 (IoT)** | EfficientAd | CPU 가능, 경량 | 97.8% AUROC |
| **일반 검사** | STFPM 또는 FastFlow | 균형 잡힌 성능 | 96.8-98.5% |
| **빠른 프로토타입** | STFPM | 간단한 구현 | 96.8% AUROC |
| **속도 최적화 필요** | EfficientAd | 압도적 속도 | 200-1000 FPS |

### A.3 성능-속도 Trade-off 분석

**정확도 vs 속도**:
```
Reverse Distillation: 98.6% @ 100-200ms
    ↓ (정확도 0.8%p 희생)
EfficientAd: 97.8% @ 1-5ms
    → 속도 20-200배 향상

ROI: 0.8%p 정확도 감소로 20-200배 속도 획득
     대부분의 실무 환경에서 EfficientAd가 더 가치 있음
```

### A.4 하드웨어 요구사항 및 배포

| 모델 | GPU 메모리 | CPU 추론 | 엣지 배포 | 권장 환경 |
|------|-----------|----------|----------|----------|
| **STFPM** | 4GB+ | 느림 (100ms+) | 어려움 | GPU 서버 |
| **FRE** | 4GB | 느림 (80ms+) | 어려움 | **사용 안함** |
| **Reverse Distillation** | 4-8GB | 불가능 | 불가능 | 고성능 GPU 서버 |
| **EfficientAd** | 2GB+ | 가능 (10-20ms) | 가능 ★★★★★ | GPU/CPU/엣지 |

### A.5 개발-배포 체크리스트 (Knowledge Distillation)

**Phase 1: 요구사항 분석**
- [ ] 정확도 목표 (97%? 98%+?)
- [ ] 속도 제약 (실시간? 준실시간?)
- [ ] 하드웨어 환경 (GPU? CPU? 엣지?)
- [ ] EfficientAd vs Reverse Distillation 결정

**Phase 2: 데이터 준비**
- [ ] 정상 샘플 수집 (100-500장)
- [ ] 이미지 전처리 및 augmentation
- [ ] Train/validation split

**Phase 3: 모델 학습**
- [ ] Pre-trained backbone 선택
- [ ] Teacher network 준비 (fine-tuning 여부)
- [ ] Student network 학습
- [ ] Loss convergence 모니터링

**Phase 4: 최적화 (EfficientAd)**
- [ ] Model quantization (INT8)
- [ ] ONNX export
- [ ] TensorRT optimization (선택)
- [ ] Half precision (FP16)

**Phase 5: 평가 및 배포**
- [ ] Validation set 성능 확인
- [ ] 추론 속도 벤치마크
- [ ] 메모리 사용량 모니터링
- [ ] 엣지 디바이스 테스트 (해당 시)

### A.6 Knowledge Distillation 선택 의사결정 트리

```
정확도 vs 속도 우선순위?
│
├─ 최고 정확도 필수 (>98.5%)
│   └─ Reverse Distillation
│       - 정밀 검사
│       - GPU 서버
│
├─ 실시간 처리 필수 (<10ms)
│   └─ EfficientAd (유일한 선택)
│       - 고속 라인
│       - 엣지 디바이스
│
├─ 균형 필요
│   ├─ 속도 중시 → EfficientAd
│   └─ 정확도 중시 → Reverse Distillation
│
└─ 빠른 프로토타입
    └─ STFPM (간단한 구현)
```

### A.7 FRE의 교훈

**과도기적 모델의 역할**:
- STFPM → FRE: 점진적 개선 시도 (2배 속도)
- FRE → EfficientAd: 혁명적 발전 (20-40배 속도)

**핵심 교훈**:
1. 점진적 개선만으로는 실무 임팩트 제한적
2. 혁명적 발전이 패러다임을 바꿈
3. 속도 최적화는 성능 저하를 동반하지 않을 수 있음 (EfficientAd 증명)

**현재 상태**:
- FRE: Deprecated, 사용 비추천
- 대안: EfficientAd 사용

---

## 결론

Knowledge Distillation 방식은 **두 가지 방향**으로 발전했다:

1. **정확도 극대화**: Reverse Distillation (98.6%)
   - One-class embedding 역재구성
   - SOTA급 pixel-level localization
   - 정밀 검사에 최적

2. **속도 극대화**: EfficientAd (1-5ms)
   - PDN + AE 하이브리드
   - 실시간 처리, 엣지 배포
   - 대부분의 실무 환경에 최적

**FRE의 역사적 의의**:
- STFPM과 EfficientAd 사이의 "징검다리"
- 점진적 개선의 한계를 보여줌
- 현재는 deprecated, 학술적 참고용

**최종 권장사항**:
- **정밀 검사**: Reverse Distillation (98.6%, 100-200ms)
- **실시간/엣지**: EfficientAd (97.8%, 1-5ms)
- **일반 검사**: FastFlow 또는 PatchCore 고려
- **FRE**: 사용 비추천 (deprecated)

Knowledge Distillation은 정확도(Reverse Distillation)와 속도(EfficientAd) 양극단에서 모두 강력한 솔루션을 제공하며, 이상 탐지의 핵심 패러다임으로 자리잡았다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: STFPM, FRE, Reverse Distillation, EfficientAd

**주요 내용**:
1. Knowledge Distillation 패러다임 개요
2. STFPM 상세 분석
3. **FRE 상세 분석** (과도기적 모델로서의 역할 및 deprecated 상태 명시)
4. Reverse Distillation 상세 분석 (패러다임 역전)
5. EfficientAd 상세 분석 (실시간 혁명)
6. 속도 최적화 발전 과정 (STFPM → FRE → EfficientAd)
7. 종합 비교
8. **부록**: overall_report.md의 관련 테이블 포함