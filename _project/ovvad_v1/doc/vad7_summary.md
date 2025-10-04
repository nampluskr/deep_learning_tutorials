# Anomalib 라이브러리 이상 탐지 모델 종합 비교 분석

## 1. 개요

본 보고서는 Anomalib v2.1.0 및 최신 버전에 포함된 이상 탐지(Anomaly Detection) 모델들의 기술적 특징, 주요 원리, 발전 과정, 그리고 성능을 체계적으로 분석한다. 총 21개 모델을 6개 패러다임으로 분류하여 각 모델의 기술적 차별점과 개선사항을 제시한다.

**분석 대상 모델**: PaDiM, PatchCore, DFKDE, CFLOW, FastFlow, CS-Flow, U-Flow, STFPM, FRE, Reverse Distillation, EfficientAd, GANomaly, DRAEM, DSR, CFA, DFM, WinCLIP, Dinomaly, VLM-AD, SuperSimpleNet, UniNet

## 2. 패러다임별 분류 및 핵심 원리

### 2.1 Memory-Based / Feature Matching 방식

정상 샘플의 특징 벡터를 메모리에 저장하고, 테스트 샘플과의 유사도를 비교하여 이상을 탐지한다.

**핵심 수식**:
$$\text{Anomaly Score} = d(f_{\text{test}}, \mathcal{M}_{\text{normal}})$$

여기서 $\mathcal{M}_{\text{normal}}$은 정상 샘플들의 메모리 뱅크, $d(\cdot, \cdot)$는 거리 함수

#### 2.1.1 PaDiM (2020)

**Patch Distribution Modeling Framework**

각 패치 위치에서 정상 샘플들의 특징 분포를 다변량 가우시안으로 모델링:

$$p(\mathbf{x}_{i,j}) = \mathcal{N}(\boldsymbol{\mu}_{i,j}, \boldsymbol{\Sigma}_{i,j})$$

**이상 점수**: Mahalanobis distance
$$M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

**성능**: Image AUROC 96.5%, Pixel AUROC 95.8%  
**장점**: 직관적, 빠른 추론 (30-50ms)  
**단점**: 높은 메모리 사용 (2-5GB)

#### 2.1.2 PatchCore (2022)

**Towards Total Recall in Industrial Anomaly Detection**

PaDiM의 메모리 문제를 Coreset Selection으로 해결. Greedy 알고리즘으로 대표 패치만 선택:

$$\mathcal{C} = \underset{|\mathcal{C}|=M}{\arg\min} \max_{\mathbf{x} \in \mathcal{X}} \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{x} - \mathbf{c}\|_2$$

**개선사항**:
- 메모리 사용량 90% 감소 (2-5GB → 100-500MB)
- 성능 향상: 96.5% → 99.1% AUROC
- k-NN 기반 이상 점수 계산

**성능**: Image AUROC 99.1% (현재까지 single-class 최고)  
**추론 속도**: 50-100ms

#### 2.1.3 DFKDE (2022)

**Deep Feature Kernel Density Estimation**

딥러닝 특징에 Kernel Density Estimation 적용:

$$\hat{p}(\mathbf{x}) = \frac{1}{Nh} \sum_{i=1}^{N} K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$

**성능**: Image AUROC 95.5-96.8%  
**한계**: 고차원에서 curse of dimensionality

---

### 2.2 Normalizing Flow 방식

가역적 변환을 통해 복잡한 데이터 분포를 단순 분포로 매핑:

$$\log p(\mathbf{x}) = \log p(f(\mathbf{x})) + \log\left|\det\frac{\partial f}{\partial \mathbf{x}}\right|$$

높은 $\log p(\mathbf{x})$ = 정상, 낮은 $\log p(\mathbf{x})$ = 이상

#### 2.2.1 CFLOW (2021)

**Conditional Normalizing Flows**

위치 조건부 분포 모델링. Multi-scale에서 독립적인 flow network 학습.

**성능**: Image AUROC 98.2%  
**속도**: 100-150ms (느림)

#### 2.2.2 FastFlow (2021)

**2D Normalizing Flows for Speed**

CFLOW의 3D flow를 2D로 단순화하여 속도 대폭 향상:

**개선사항**:
- 추론 속도 2-3배 향상 (100-150ms → 20-50ms)
- 성능 유지/향상: 98.5% AUROC
- 메모리 효율 개선

#### 2.2.3 CS-Flow (2021)

**Cross-Scale Flow**

스케일 간 정보 교환으로 다양한 크기 결함 탐지.

**성능**: Image AUROC 97.9%  
**적용**: 크기 다양한 결함 (Grid, Tile 등)

#### 2.2.4 U-Flow (2022)

**U-shaped Flow with Unsupervised Threshold**

U-Net 구조 + 자동 임계값 설정.

**성능**: Image AUROC 97.6%  
**장점**: 운영 자동화

---

### 2.3 Knowledge Distillation 방식

Teacher 네트워크의 지식을 Student가 모방:

$$\text{Anomaly Score} = \|f_T(\mathbf{x}) - f_S(\mathbf{x})\|$$

정상 샘플: $f_T \approx f_S$, 이상 샘플: $f_T \neq f_S$

#### 2.3.1 STFPM (2021)

**Student-Teacher Feature Pyramid Matching**

Multi-scale feature matching:

$$\mathcal{L} = \sum_{l=1}^{L} \|f_T^{(l)}(\mathbf{x}) - f_S^{(l)}(\mathbf{x})\|_2^2$$

**성능**: Image AUROC 96.8%  
**속도**: 20-40ms (빠름)  
**장점**: 간단한 구조, end-to-end 학습

#### 2.3.2 FRE (2023)

**Feature Reconstruction Error (Fast Method)**

STFPM의 경량화 시도. 속도 최적화에 집중.

**개선 시도**:
- 추론 속도 약 2배 향상 (20-40ms → 10-30ms)
- 구조 경량화

**한계**:
- 성능 저하 (96.8% → 95-96%, -0.8~1.8%p)
- 속도 개선 폭 제한적 (2배 미만)
- EfficientAd 등장으로 실용적 가치 상실

**역사적 의의**: STFPM에서 EfficientAd로 가는 과도기적 모델  
**현재 상태**: Deprecated (EfficientAd로 대체)

#### 2.3.3 Reverse Distillation (2022)

**패러다임 역전**: Teacher(단순) ← Student(복잡)

Teacher가 one-class embedding 생성, Student가 이를 역으로 재구성:

$$\mathcal{L} = \mathcal{L}_{\text{cos}}(z_T, z_S) + \lambda \mathcal{L}_{\text{L2}}(z_T, z_S)$$

**성능**: Image AUROC 98.6%, Pixel AUROC 98.5%  
**장점**: SOTA급 정확도, 우수한 localization  
**속도**: 100-200ms (느림)

#### 2.3.4 EfficientAd (2024)

**Millisecond-Level Latency**

Patch Description Network (PDN) + Auto-encoder 하이브리드:

**개선사항**:
- 극한의 속도: 1-5ms (20-200배 향상)
- 경량 구조: <200MB
- CPU에서도 실시간 가능 (10-20ms)

**성능**: Image AUROC 97.8%  
**적용**: 실시간 라인, 엣지 디바이스

---

### 2.4 Reconstruction-Based 방식

재구성 오류를 이상 점수로 사용:

$$\text{Anomaly Score} = \|\mathbf{x} - \text{Decoder}(\text{Encoder}(\mathbf{x}))\|$$

정상: 재구성 성공, 이상: 재구성 실패

#### 2.4.1 GANomaly (2018)

**GAN-based E-D-E 구조**

Encoder-Decoder-Encoder 구조. 두 encoder의 latent code 차이:

$$\mathcal{L}_G = \mathcal{L}_{\text{adv}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}} + \lambda_{\text{enc}}\mathcal{L}_{\text{enc}}$$

**성능**: Image AUROC 93-95%  
**한계**: GAN 학습 불안정성, 낮은 성능

#### 2.4.2 DRAEM (2021)

**Discriminatively Trained Reconstruction with Simulated Anomaly**

**패러다임 혁신**: Simulated anomaly로 supervised 학습

정상 이미지에 인위적 결함 추가 → 제거하도록 학습:

$$\mathcal{L} = \mathcal{L}_{\text{SSIM}}(\mathbf{x}_{\text{rec}}, \mathbf{x}_{\text{orig}}) + \mathcal{L}_{\text{focal}}(\mathbf{m}_{\text{pred}}, \mathbf{m}_{\text{gt}})$$

**성능**: Image AUROC 97.5%  
**장점**: 안정적 학습, Few-shot 가능 (10-50장)  
**개선**: GANomaly 대비 +2.5~4.5%p

#### 2.4.3 DSR (2022)

**Dual Subspace Re-Projection**

Quantization subspace (구조) + Target subspace (텍스처):

**성능**: Image AUROC 96.5-98.0% (카테고리별)  
**적용**: 복잡한 텍스처 표면 (직물, 카펫)

---

### 2.5 Feature Adaptation 방식

Pre-trained 특징을 타겟 도메인에 적응:

$$\mathbf{f}_{\text{adapted}} = \mathcal{A}(\mathbf{f}_{\text{pretrained}}, \mathcal{D}_{\text{target}})$$

#### 2.5.1 DFM (2019)

**Deep Feature Modeling with PCA**

딥러닝 특징에 PCA 적용:

$$\text{Anomaly Score} = \|\mathbf{x} - \mathbf{x}_{\text{reconstructed}}\|_{\boldsymbol{\Sigma}^{-1}}$$

**성능**: Image AUROC 94.5-95.5%  
**장점**: 극도로 간단 (학습 5-15분), 빠름 (10-20ms)  
**한계**: 선형 가정, 낮은 성능

#### 2.5.2 CFA (2022)

**Coupled-Hypersphere Feature Adaptation**

Hypersphere embedding으로 domain adaptation:

단위 구 표면에 특징 projection → Angular distance로 이상 점수

**성능**: Image AUROC 96.5-97.5%  
**적용**: Domain shift가 큰 환경

---

### 2.6 Foundation Model 기반 방식

대규모 pre-trained 모델 활용 (CLIP, DINOv2, GPT-4V).

#### 2.6.1 WinCLIP (2023)

**Zero-shot with CLIP**

텍스트 프롬프트만으로 이상 탐지:

$$\text{Score} = \text{sim}(\mathbf{I}, \text{"defective"}) - \text{sim}(\mathbf{I}, \text{"normal"})$$

**성능**: Image AUROC 91-95% (zero-shot)  
**장점**: 학습 데이터 불필요, 즉시 배포  
**적용**: 신제품, 다품종 소량 생산

#### 2.6.2 Dinomaly (2025)

**DINOv2-based Multi-class SOTA**

"Less is More" 철학. 간단한 구조 + 강력한 DINOv2 특징:

**성능**: 
- Multi-class: Image AUROC 98.8%
- Single-class: Image AUROC 99.2%

**장점**: 단일 모델로 multi-class 처리, 메모리 80% 절감

#### 2.6.3 VLM-AD (2024)

**Explainable AI with GPT-4V**

Vision-Language Model로 자연어 설명 생성.

**성능**: Image AUROC 96-97%  
**장점**: 자연어 설명, 보고서 자동 생성  
**한계**: API 비용 ($0.01-0.05/img), 느림 (2-5초)

#### 2.6.4 SuperSimpleNet (2024)

**Unified Unsupervised + Supervised**

**성능**: Image AUROC 97.2%  
**장점**: 실용적 통합 접근

#### 2.6.5 UniNet (2025)

**Unified Contrastive Learning**

**성능**: Image AUROC 98.3%  
**장점**: 강건한 decision boundary

---

## 3. 시간순 발전 과정

### 3.1 태동기 (2018-2019)

- **GANomaly (2018)**: GAN 기반 초기 시도, 학습 불안정성
- **DFM (2019)**: PCA 기반 간단한 접근, 성능 제한적

### 3.2 성장기 (2020-2021)

- **PaDiM (2020)**: Memory-based 기초 확립
- **2021년 기술적 다양화**:
  - Normalizing Flow 전성기: CFLOW, FastFlow, CS-Flow
  - Knowledge Distillation 등장: STFPM
  - Reconstruction 혁신: DRAEM (simulated anomaly)

### 3.3 성숙기 (2022)

- **PatchCore**: Memory-based 완성, SOTA (99.1%)
- **Reverse Distillation**: KD 패러다임 발전
- **U-Flow, CFA, DSR**: 각 분야 개선

### 3.4 과도기적 시도 (2023)

- **FRE**: KD 내 속도 최적화 시도
  - STFPM 대비 2배 속도 향상 목표
  - 성능 저하 (-0.8~1.8%p)와 제한적 개선으로 실용성 제한
  - 이후 EfficientAd 등장으로 대체됨

### 3.5 Foundation Model 시대 (2023-2025)

- **WinCLIP (2023)**: Zero-shot 이상 탐지 시작
- **EfficientAd (2024)**: Millisecond 레벨 혁명
- **VLM-AD (2024)**: Explainable AI
- **Dinomaly (2025)**: Multi-class SOTA
- **UniNet (2025)**: Contrastive 통합

### 3.6 주요 기술적 전환점

1. **PaDiM → PatchCore**: Coreset으로 메모리 90% 절감 + 성능 향상
2. **CFLOW → FastFlow**: 2D flow로 2-3배 속도 향상
3. **STFPM → Reverse Distillation**: 패러다임 역전으로 성능 대폭 향상
4. **전통적 방법 → Foundation Models**: Zero-shot, Multi-class 가능
5. **단계적 속도 개선**: STFPM (20-40ms) → FRE (10-30ms) → EfficientAd (1-5ms)

---

## 4. 성능 비교

### 4.1 MVTec AD 벤치마크 기준

| 모델 | Image AUROC | Pixel AUROC | 추론 속도 | 메모리 | 발표연도 |
|------|-------------|-------------|-----------|--------|----------|
| **PatchCore** | **99.1%** | 98.2% | 50-100ms | 100-500MB | 2022 |
| **Dinomaly** | 98.8% (multi) | 97.5% | 80-120ms | 1.5-2GB | 2025 |
| **Reverse Distillation** | 98.6% | **98.5%** | 100-200ms | 500MB-1GB | 2022 |
| **FastFlow** | 98.5% | 97.8% | 20-50ms | 500MB-1GB | 2021 |
| **UniNet** | 98.3% | 97.0% | 50-80ms | 400-600MB | 2025 |
| **CFLOW** | 98.2% | 97.6% | 100-150ms | 500MB-1GB | 2021 |
| **EfficientAd** | 97.8% | 97.2% | **1-5ms** | **<200MB** | 2024 |
| **DRAEM** | 97.5% | 96.8% | 50-100ms | 300-500MB | 2021 |
| **SuperSimpleNet** | 97.2% | 95.8% | 40-60ms | 300-500MB | 2024 |
| **STFPM** | 96.8% | 96.2% | 20-40ms | 500MB-1GB | 2021 |
| **VLM-AD** | 96-97% | 94-96% | 2-5초 | API | 2024 |
| **PaDiM** | 96.5% | 95.8% | 30-50ms | 2-5GB | 2020 |
| **FRE** | 95-96% | 94-95% | 10-30ms | 300-500MB | 2023 |
| **WinCLIP** | 91-95% | 89-93% | 50-100ms | 500MB-1.5GB | 2023 |

### 4.2 카테고리별 최고 성능

#### 정확도 최우선
1. **PatchCore**: 99.1% (single-class 최고)
2. **Dinomaly**: 98.8% (multi-class 최고)
3. **Reverse Distillation**: 98.5% (pixel-level 최고)

#### 속도 최우선
1. **EfficientAd**: 1-5ms (압도적)
2. **FRE**: 10-30ms (deprecated)
3. **STFPM**: 20-40ms

#### 메모리 효율
1. **EfficientAd**: <200MB
2. **DRAEM**: 300-500MB
3. **SuperSimpleNet**: 300-500MB

#### 균형잡힌 성능
1. **FastFlow**: 98.5% + 20-50ms
2. **Dinomaly**: 98.8% (multi) + 합리적 리소스
3. **Reverse Distillation**: 98.6% + 중간 속도

---

## 5. 패러다임별 장단점 요약

### 5.1 Memory-Based

**장점**: 최고 정확도 (99.1%), 직관적, 수학적으로 명확  
**단점**: 메모리 사용량 (PaDiM), 학습 데이터 증가 시 메모리 증가  
**대표**: PatchCore ★★★★★

### 5.2 Normalizing Flow

**장점**: 확률론적 해석 가능, pixel-level 우수, 빠른 속도 (FastFlow)  
**단점**: 학습 복잡, 하이퍼파라미터 튜닝  
**대표**: FastFlow ★★★★★

### 5.3 Knowledge Distillation

**장점**: End-to-end 학습, 속도-정확도 균형, 극한 최적화 가능 (EfficientAd)  
**단점**: 구조 설계 필요, teacher 품질 의존  
**대표**: 
- 정밀 검사: Reverse Distillation ★★★★★
- 실시간: EfficientAd ★★★★★
- Deprecated: FRE ★☆☆☆☆

### 5.4 Reconstruction-Based

**장점**: 직관적, Few-shot 가능 (DRAEM), 복잡한 텍스처 처리 (DSR)  
**단점**: GAN 불안정 (GANomaly), Simulation 품질 의존 (DRAEM)  
**대표**: DRAEM ★★★★☆

### 5.5 Feature Adaptation

**장점**: Domain shift 해결, 간단한 구현 (DFM)  
**단점**: 낮은 성능, 복잡한 학습 (CFA)  
**대표**: 실무 활용 제한적 ★★☆☆☆

### 5.6 Foundation Model

**장점**: Zero-shot 가능, Multi-class 우수, Explainable (VLM-AD)  
**단점**: 모델 크기, API 비용 (VLM-AD)  
**대표**: 
- Multi-class: Dinomaly ★★★★★
- Zero-shot: WinCLIP ★★★★☆
- Explainable: VLM-AD ★★★★☆

---

## 6. 실무 적용 가이드

### 6.1 시나리오별 최적 모델

| 시나리오 | 1순위 | 2순위 | 이유 |
|---------|-------|-------|------|
| **최고 정확도** | PatchCore | Dinomaly | 99%+ 필수 |
| **Multi-class** | Dinomaly | - | 단일 모델 압도적 |
| **실시간 처리** | EfficientAd | FastFlow | 1-5ms 초고속 |
| **엣지 디바이스** | EfficientAd | - | CPU 가능 |
| **신제품 (데이터 없음)** | WinCLIP | VLM-AD | Zero-shot |
| **Few-shot (10-50장)** | DRAEM | - | Simulated anomaly |
| **품질 보고서** | VLM-AD | - | 자연어 설명 |
| **빠른 프로토타입** | WinCLIP | DFM | 즉시 사용/15분 학습 |
| **균형 잡힌 검사** | FastFlow | Dinomaly | 속도+정확도 |
| **복잡한 텍스처** | DSR | Dinomaly | VQ-VAE |

### 6.2 하드웨어 요구사항

**GPU 메모리**:
- 4GB 미만: STFPM, DRAEM, EfficientAd
- 4-8GB: FastFlow, CFLOW, Reverse Distillation
- 8GB+: PatchCore, Dinomaly

**추론 환경**:
- CPU Only: EfficientAd
- GPU 권장: 대부분 모델
- 고성능 GPU: VLM-AD, Dinomaly

### 6.3 개발-배포 로드맵

**Phase 1: 프로토타이핑** (1-2주)
- WinCLIP (zero-shot) 또는 DFM (15분 학습)으로 빠른 검증
- 성능 목표 설정 및 데이터 수집 계획

**Phase 2: 성능 최적화** (2-4주)
- PatchCore 또는 Reverse Distillation으로 정확도 극대화
- 학습 데이터 수집 (100-500장)
- 벤치마크 수행

**Phase 3: 배포 준비** (2-3주)
- 속도 요구사항에 따라:
  - 실시간: EfficientAd로 전환
  - 일반: FastFlow 또는 그대로 유지
- 최적화 (양자화, 프루닝)

**Phase 4: 운영** (지속적)
- 모니터링 및 재학습
- 최신 Foundation Model 추적 (Dinomaly, UniNet 등)

---

## 7. 결론 및 향후 전망

### 7.1 핵심 발견

1. **성능-속도-메모리 Trade-off**: 세 가지를 모두 만족하는 모델은 없음
2. **Foundation Model의 부상**: Zero-shot, Multi-class 능력으로 패러다임 전환
3. **실용화 가속**: EfficientAd로 실시간 처리 현실화
4. **Explainability**: VLM-AD로 설명 가능한 AI 구현
5. **속도 최적화 경로**: STFPM → FRE (과도기) → EfficientAd (혁명)

### 7.2 2025년 Best Practices

**Top 3 권장 모델**:
1. **Dinomaly**: Multi-class 환경의 새로운 표준 (98.8%)
2. **EfficientAd**: 실시간 처리의 유일한 해답 (1-5ms)
3. **PatchCore**: Single-class 최고 정확도 (99.1%)

**상황별 Best Pick**:
- 대부분의 경우: **Dinomaly** (multi-class 가능)
- 실시간 라인: **EfficientAd** (극한 속도)
- 신제품/프로토타입: **WinCLIP** (zero-shot)
- 품질 보고서: **VLM-AD** (explainable)

### 7.3 향후 연구 방향

1. **Multi-modal Fusion**: 이미지 + 센서 데이터
2. **Continual Learning**: 지속적 정상 패턴 학습
3. **Edge AI**: 더 경량화된 모델
4. **Domain-specific Foundation Models**: 산업 특화 대규모 모델
5. **Uncertainty Estimation**: 신뢰도 있는 이상 점수

### 7.4 최종 의사결정 플로우

```
정확도 vs 속도 우선순위?
├─ 최고 정확도 (>99%)
│   ├─ Single-class → PatchCore
│   └─ Multi-class → Dinomaly
│
├─ 실시간 (<10ms)
│   └─ EfficientAd (유일)
│
└─ 균형 필요
    ├─ 데이터 없음 → WinCLIP
    ├─ Few-shot → DRAEM
    ├─ Multi-class → Dinomaly
    ├─ 설명 필요 → VLM-AD
    └─ 일반 → FastFlow
```

### 7.5 최종 요약

Anomalib 라이브러리는 2018-2025년의 이상 탐지 연구 발전을 포괄적으로 담고 있다. 6개 패러다임이 공존하며, 각각 고유한 강점을 가진다:

- **Memory-based (PatchCore)**: 최고 정확도 99.1%
- **Normalizing Flow (FastFlow)**: 속도-정확도 균형 98.5%
- **Knowledge Distillation**: 
  - 정밀 검사 (Reverse Distillation 98.6%)
  - 실시간 (EfficientAd 1-5ms)
  - Deprecated (FRE, EfficientAd로 대체)
- **Reconstruction (DRAEM)**: Few-shot 97.5%
- **Feature Adaptation**: 특수 domain shift 환경
- **Foundation Model (Dinomaly)**: Multi-class 혁명 98.8%

실무 적용 시 정확도, 속도, 메모리, 학습 데이터, 설명 가능성 등을 종합 고려하여 최적 모델을 선택해야 한다. 특히 2024-2025년의 Foundation Model들은 zero-shot/few-shot 방향으로 패러다임을 전환하고 있으며, 이는 향후 산업 적용의 중요한 트렌드가 될 것으로 전망된다.
