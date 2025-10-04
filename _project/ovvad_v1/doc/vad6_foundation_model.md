# 6. Foundation Model 기반 방식 상세 분석

## 6.1 패러다임 개요

Foundation Model 기반 방식은 대규모 데이터셋(수억~수십억 개 샘플)으로 사전 학습된 범용 모델을 이상 탐지에 활용한다. CLIP, DINOv2, GPT-4V 등의 foundation model은 강력한 일반화 능력과 zero-shot/few-shot 성능을 제공한다.

**핵심 원리**:

$$\text{Anomaly Score} = f_{\text{foundation}}(\mathbf{x}, \text{context})$$

여기서:
- $f_{\text{foundation}}$: 대규모 사전 학습된 모델
- $\text{context}$: 텍스트 프롬프트, 참조 이미지 등

**패러다임 전환**:

전통적 방법:
- 타겟 도메인 데이터로만 학습
- 수백 장의 학습 데이터 필요
- Single-class 모델

Foundation Model:
- 대규모 범용 데이터로 사전 학습
- Zero-shot 가능 (학습 데이터 0장)
- Multi-class 단일 모델

**주요 Foundation Models**:
1. **CLIP** (Contrastive Language-Image Pre-training): 이미지-텍스트 연결
2. **DINOv2** (Self-Distillation with No Labels): 강력한 시각 표현
3. **GPT-4V** (Vision): 멀티모달 이해 및 설명 생성

---

## 6.2 WinCLIP (2023)

### 6.2.1 기본 정보

- **논문**: CLIP-based Zero-shot Anomaly Detection
- **발표**: arXiv 2023
- **저자**: Anomalib team
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/winclip

### 6.2.2 핵심 원리

WinCLIP은 OpenAI의 **CLIP 모델**을 활용하여 **텍스트 프롬프트만으로** 이상 탐지를 수행한다. 학습 데이터 없이 즉시 사용 가능한 zero-shot 이상 탐지다.

**CLIP 기본 구조**:

$$\text{sim}(\mathbf{I}, \mathbf{T}) = \frac{\text{CLIP}_{\text{image}}(\mathbf{I}) \cdot \text{CLIP}_{\text{text}}(\mathbf{T})}{\|\text{CLIP}_{\text{image}}(\mathbf{I})\| \|\text{CLIP}_{\text{text}}(\mathbf{T})\|}$$

**WinCLIP의 이상 점수**:

$$\text{Score} = \text{sim}(\mathbf{I}, \text{"defective"}) - \text{sim}(\mathbf{I}, \text{"normal"})$$

또는:

$$\text{Score} = 1 - \frac{\text{sim}(\mathbf{I}, \text{"normal"})}{\text{sim}(\mathbf{I}, \text{"normal"}) + \text{sim}(\mathbf{I}, \text{"defective"})}$$

### 6.2.3 기술적 세부사항

**Windowing Strategy**:

이미지를 여러 패치로 분할:
$$\mathbf{I} = \{\mathbf{p}_1, \mathbf{p}_2, ..., \mathbf{p}_N\}$$

각 패치에서 이상 점수 계산:
$$s_i = \text{Score}(\mathbf{p}_i)$$

**Multi-scale Windows**:
- 큰 윈도우: 전역적 이상
- 중간 윈도우: 영역 이상
- 작은 윈도우: 국소적 결함

**Prompt Engineering**:

효과적인 프롬프트 예시:
```
Normal prompts:
- "a photo of a perfect {class}"
- "a photo of a flawless {class}"
- "a high-quality {class} without defects"

Defective prompts:
- "a photo of a defective {class}"
- "a photo of a {class} with scratches"
- "a photo of a broken {class}"
```

**Ensemble Prompts**:

여러 프롬프트 평균:
$$\text{Score} = \frac{1}{M}\sum_{j=1}^{M} \text{Score}_j$$

### 6.2.4 Zero-shot의 의미

**전통적 방법과의 차이**:

전통적 (PatchCore):
1. 정상 샘플 100-500장 수집
2. 특징 추출 및 메모리 뱅크 구축
3. 학습 (1-2시간)
4. 배포

WinCLIP:
1. 제품 이름만 입력 (예: "transistor")
2. 프롬프트 작성
3. **즉시 배포** (학습 0분)

**Zero-shot의 가치**:
- 신제품 즉시 검사 가능
- 학습 데이터 수집 불필요
- 다품종 소량 생산에 이상적

### 6.2.5 성능

**MVTec AD 벤치마크**:
- Image AUROC: 91-95% (zero-shot)
- Pixel AUROC: 89-93% (zero-shot)
- 추론 속도: 50-100ms (CLIP 모델 크기에 따라)
- 메모리: 500MB-1.5GB (CLIP 모델)
- 학습 시간: 0분 (zero-shot)

### 6.2.6 장점

1. **Zero-shot**: 학습 데이터 불필요
2. **즉시 배포**: 프롬프트만으로 즉시 사용
3. **유연성**: 텍스트로 쉽게 조정
4. **신제품 대응**: 새로운 제품에 즉시 적용
5. **다품종**: 단일 모델로 여러 제품 검사
6. **Interpretable**: 텍스트 설명 가능

### 6.2.7 단점

1. **낮은 정확도**: 91-95% (SOTA 99.1% 대비 -4~8%p)
2. **프롬프트 의존**: 프롬프트 품질에 성능 좌우
3. **CLIP 한계**: 산업 이미지 학습 부족
4. **모델 크기**: 500MB-1.5GB
5. **세밀한 결함**: 작은 결함 탐지 어려움

---

## 6.3 Dinomaly (2025)

### 6.3.1 기본 정보

- **논문**: Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection
- **발표**: 2025
- **저자**: Jia et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib (예정)

### 6.3.2 핵심 원리

Dinomaly는 **DINOv2** foundation model을 활용하여 **단일 모델로 multi-class 이상 탐지**를 수행한다. "Less is More" 철학으로 간단한 구조로 SOTA급 성능을 달성한다.

**DINOv2의 강점**:
- Self-supervised learning으로 학습
- 강력한 시각적 표현 (semantic + low-level)
- 별도의 텍스트 없이 이미지만으로 학습

**Multi-class 이상 탐지**:

$$p(\text{anomaly} | \mathbf{x}, c) = f_{\text{DINOv2}}(\mathbf{x}, c)$$

여기서 $c$는 클래스 레이블

**단일 모델의 의미**:

전통적 방법 (PatchCore):
- 카테고리 1 모델: 100-500MB
- 카테고리 2 모델: 100-500MB
- ...
- 카테고리 15 모델: 100-500MB
- **총 메모리**: 1.5-7.5GB

Dinomaly:
- **단일 통합 모델**: 300-500MB
- 모든 카테고리 처리
- **메모리 절감**: 80-90%

### 6.3.3 기술적 세부사항

**DINOv2 Feature Extraction**:

Multi-scale features:
$$\mathbf{f} = [\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4]$$

Layer 선택:
- Layer 1-2: Low-level (texture, edges)
- Layer 3-4: High-level (semantic)

**Class-conditional Memory Bank**:

각 클래스별 대표 특징 저장:
$$\mathcal{M} = \{\mathcal{M}_{c_1}, \mathcal{M}_{c_2}, ..., \mathcal{M}_{c_K}\}$$

**Anomaly Score**:

테스트 샘플 $\mathbf{x}$, 클래스 $c$:
$$\text{Score}(\mathbf{x}, c) = \min_{\mathbf{f}_i \in \mathcal{M}_c} \|\mathbf{f}(\mathbf{x}) - \mathbf{f}_i\|_2$$

**Less is More 철학**:

복잡한 구조 대신:
- 강력한 DINOv2 특징
- 간단한 k-NN 매칭
- Minimal 하이퍼파라미터

### 6.3.4 PatchCore 대비 핵심 차이점

| 측면 | PatchCore | Dinomaly | 개선 효과 |
|------|-----------|----------|----------|
| **모델 수** | 클래스당 1개 (15개 클래스 = 15개 모델) | 1개 통합 모델 | 메모리 80-90% 절감 |
| **Feature Backbone** | ImageNet ResNet/WideResNet | DINOv2 (self-supervised) | 더 강력한 표현 |
| **학습 방식** | 클래스별 독립 학습 | Multi-class 통합 학습 | 관리 간소화 |
| **메모리 (15 클래스)** | 1.5-7.5GB | 300-500MB | 80-90% 절감 |
| **Image AUROC (multi)** | 99.1% (single) | 98.8% (multi) | -0.3%p (미미) |
| **Image AUROC (single)** | 99.1% | 99.2% | +0.1%p |
| **추론 속도** | 50-100ms | 80-120ms | 약간 느림 |
| **배포 복잡도** | 높음 (다중 모델) | 낮음 (단일 모델) | 운영 간소화 |

### 6.3.5 왜 Multi-class가 중요한가?

**실무 시나리오**:

제조 라인에서 15개 제품 검사:

전통적 방법 (PatchCore):
- 15개 모델 각각 학습
- 15개 모델 메모리 관리
- 제품 추가 시 재학습
- 모델 버전 관리 복잡

Dinomaly:
- 1개 모델로 모든 제품 처리
- 메모리 효율적
- 새 제품 추가 용이
- 단순한 관리

**비용 절감**:
- GPU 메모리 절감 → 저렴한 하드웨어
- 관리 비용 절감 → 인력 절감
- 빠른 배포 → Time-to-market 단축

### 6.3.6 성능

**MVTec AD 벤치마크**:
- Image AUROC (multi-class): 98.8%
- Image AUROC (single-class): 99.2%
- Pixel AUROC: 97.5%
- 추론 속도: 80-120ms
- 메모리: 300-500MB (단일 모델, 모든 클래스)
- 학습 시간: 2-4시간 (모든 클래스 통합)

### 6.3.7 장점

1. **Multi-class SOTA**: 98.8% (단일 모델)
2. **메모리 효율**: 80-90% 절감
3. **운영 간소화**: 단일 모델 관리
4. **강력한 표현**: DINOv2 특징
5. **간단한 구조**: Less is More
6. **확장성**: 새 클래스 추가 용이

### 6.3.7 단점

1. **DINOv2 의존**: 대형 모델 (300MB+)
2. **약간 느림**: 80-120ms (PatchCore 50-100ms)
3. **GPU 메모리**: 학습 시 4-8GB 필요
4. **Single-class**: PatchCore 대비 동등 (99.2% vs 99.1%)

---

## 6.4 VLM-AD (2024)

### 6.4.1 기본 정보

- **논문**: Vision-Language Models for Explainable Anomaly Detection
- **발표**: 2024
- **저자**: Various
- **핵심**: GPT-4V 등 VLM 활용

### 6.4.2 핵심 원리

VLM-AD는 **Vision-Language Model (GPT-4V, Claude, Gemini 등)**을 활용하여 이상을 탐지할 뿐만 아니라 **자연어로 설명**을 생성한다.

**입력**:
$$\text{Input} = (\mathbf{I}, \text{Prompt})$$

**출력**:
$$\text{Output} = \{\text{is\_anomaly}, \text{confidence}, \text{explanation}, \text{location}\}$$

**예시 출력**:
```json
{
  "is_anomaly": true,
  "confidence": 0.92,
  "defects": [
    {
      "type": "scratch",
      "location": "upper left corner",
      "severity": "moderate",
      "size": "approximately 5mm",
      "possible_cause": "handling damage during assembly",
      "recommendation": "inspect handling process"
    }
  ],
  "overall_quality": "defective",
  "explanation": "The image shows a clear scratch on the 
                  surface in the upper left quadrant..."
}
```

### 6.4.3 기술적 세부사항

**Prompt Template**:

```
You are an expert quality inspector. Analyze this image 
of a {product_name} and determine if there are any defects.

Common defects include:
- Scratches
- Dents
- Color variations
- Missing parts
- Contamination

Provide:
1. Is this product defective? (Yes/No)
2. Confidence level (0-1)
3. List of defects found (if any)
4. Location of each defect
5. Severity (minor/moderate/severe)
6. Possible cause
7. Recommendation

Be specific and detailed in your analysis.
```

**Few-shot Learning**:

정상/이상 예시 제공:
```
Here are examples:

Normal example:
[Image] - This is a perfect product with no defects.

Defective example 1:
[Image] - This has a scratch on the left side.

Defective example 2:
[Image] - This has discoloration in the center.

Now analyze this image:
[Test Image]
```

### 6.4.4 WinCLIP/Dinomaly 대비 핵심 차이점

| 측면 | WinCLIP | Dinomaly | VLM-AD |
|------|---------|----------|--------|
| **Foundation Model** | CLIP | DINOv2 | GPT-4V / Claude / Gemini |
| **Zero-shot** | 예 | 아니오 (학습 필요) | 예 |
| **Multi-class** | 가능 | 우수 ★★★★★ | 가능 |
| **설명 생성** | 없음 | 없음 | 자연어 설명 ★★★★★ |
| **정확도** | 91-95% | 98.8% | 96-97% |
| **추론 속도** | 50-100ms | 80-120ms | 2-5초 |
| **비용** | 무료 (오픈소스) | 무료 (오픈소스) | API 비용 ($0.01-0.05/img) |
| **Interpretability** | 낮음 | 낮음 | 매우 높음 ★★★★★ |

### 6.4.5 Explainable AI의 가치

**전통적 모델의 한계**:
- PatchCore: "이상 점수 0.87" → 무슨 의미?
- 어디에 결함? 무슨 종류? 얼마나 심각?

**VLM-AD의 장점**:
- "상단 좌측에 5mm 크기의 중간 정도 스크래치 발견"
- "조립 과정의 핸들링 손상으로 추정"
- "핸들링 프로세스 점검 권장"

**비즈니스 가치**:
1. **근본 원인 분석**: "possible_cause" 제공
2. **개선 방향**: "recommendation" 제시
3. **품질 보고서**: 자동 생성
4. **의사소통**: 비기술자도 이해 가능
5. **감사 추적**: 명확한 근거

### 6.4.6 성능

**MVTec AD 벤치마크**:
- Image AUROC: 96-97%
- Pixel AUROC: 94-96%
- 추론 속도: 2-5초 per image
- 비용: $0.01-0.05 per image (API 사용 시)
- 메모리: API 사용 (로컬 메모리 불필요)

### 6.4.7 장점

1. **Explainable**: 자연어 설명 생성
2. **Zero-shot**: 학습 데이터 불필요
3. **Multi-modal**: 이미지 + 텍스트
4. **보고서 자동화**: 품질 보고서 생성
5. **유연성**: 프롬프트로 조정
6. **근본 원인 분석**: 가능한 원인 제시

### 6.4.8 단점

1. **API 비용**: $0.01-0.05 per image (대량 처리 시 고비용)
2. **느린 속도**: 2-5초 (실시간 불가)
3. **인터넷 필요**: 온프레미스 어려움
4. **일관성**: 같은 이미지에 다른 답변 가능
5. **정확도**: 96-97% (SOTA 대비 낮음)
6. **환각 가능성**: 존재하지 않는 결함 보고

---

## 6.5 SuperSimpleNet (2024)

### 6.5.1 기본 정보

- **논문**: A Unified Framework for Unsupervised and Supervised Anomaly Detection
- **발표**: 2024
- **GitHub**: https://github.com/open-edge-platform/anomalib

### 6.5.2 핵심 원리

SuperSimpleNet은 **Unsupervised와 Supervised 방법을 통합**한 프레임워크다. Foundation model의 특징을 활용하되, 실무에서 흔한 "일부 이상 샘플 확보" 시나리오에 최적화했다.

**Hybrid Learning**:

$$\mathcal{L} = \mathcal{L}_{\text{unsupervised}} + \lambda \mathcal{L}_{\text{supervised}}$$

- $\mathcal{L}_{\text{unsupervised}}$: 정상 데이터 분포 학습
- $\mathcal{L}_{\text{supervised}}$: 레이블된 이상 샘플 활용 (있는 경우)

### 6.5.3 성능

**MVTec AD 벤치마크**:
- Image AUROC: 97.2%
- Pixel AUROC: 95.8%
- 추론 속도: 40-60ms
- 메모리: 300-500MB

### 6.5.4 장점

1. **실용적**: Unsupervised + Supervised 통합
2. **유연성**: 이상 샘플 유무에 관계없이 작동
3. **중간 성능**: 97.2% AUROC

### 6.5.5 단점

1. **복잡한 프레임워크**: 두 방법론 통합의 복잡도
2. **중간 성능**: SOTA 대비 낮음

---

## 6.6 UniNet (2025)

### 6.6.1 기본 정보

- **논문**: Unified Contrastive Learning for Anomaly Detection
- **발표**: 2025
- **GitHub**: https://github.com/open-edge-platform/anomalib (예정)

### 6.6.2 핵심 원리

UniNet은 **Contrastive Learning**을 활용하여 강건한 decision boundary를 학습한다.

**Contrastive Loss**:

$$\mathcal{L}_{\text{contrastive}} = -\log\frac{\exp(\text{sim}(\mathbf{f}_i, \mathbf{f}_i^+)/\tau)}{\sum_{j}\exp(\text{sim}(\mathbf{f}_i, \mathbf{f}_j)/\tau)}$$

여기서:
- $\mathbf{f}_i^+$: Positive pair (같은 정상 샘플의 augmentation)
- $\mathbf{f}_j$: Negative pairs (다른 샘플들)
- $\tau$: Temperature

### 6.6.3 성능

**MVTec AD 벤치마크**:
- Image AUROC: 98.3%
- Pixel AUROC: 97.0%
- 추론 속도: 50-80ms
- 메모리: 400-600MB

### 6.6.4 장점

1. **강건한 표현**: Contrastive learning
2. **높은 성능**: 98.3% AUROC
3. **Decision boundary**: 명확한 경계

### 6.6.5 단점

1. **복잡한 학습**: Contrastive learning의 난이도
2. **하이퍼파라미터**: Temperature, batch size 등 민감

---

## 6.7 Foundation Model 방식 종합 비교

### 6.7.1 기술적 진화 과정

```
WinCLIP (2023)
├─ 시작: CLIP 기반 zero-shot
├─ 특징: 텍스트 프롬프트만으로 탐지
├─ 성능: 91-95% AUROC
└─ 가치: 즉시 배포, 신제품 대응

        ↓ 성능 향상

SuperSimpleNet (2024)      VLM-AD (2024)
├─ 통합: Unsup + Sup      ├─ 혁신: Explainable AI
├─ 성능: 97.2%             ├─ 성능: 96-97%
└─ 실용: 유연한 학습       └─ 가치: 자연어 설명

        ↓ Multi-class 혁명

Dinomaly (2025)            UniNet (2025)
├─ 혁신: 단일 모델 multi   ├─ 방법: Contrastive
├─ Backbone: DINOv2        ├─ 성능: 98.3%
├─ 성능: 98.8% (multi)     └─ 특징: 강건한 경계
├─ 메모리: 80-90% 절감
└─ 영향: Multi-class 표준
```

### 6.7.2 상세 비교표

| 비교 항목 | WinCLIP | Dinomaly | VLM-AD | SuperSimpleNet | UniNet |
|----------|---------|----------|--------|----------------|--------|
| **발표 연도** | 2023 | 2025 | 2024 | 2024 | 2025 |
| **Foundation Model** | CLIP | DINOv2 | GPT-4V/Claude | Custom | Custom Contrastive |
| **학습 방식** | Zero-shot | Few-shot | Zero-shot | Hybrid | Supervised |
| **Multi-class** | 가능 (프롬프트) | 우수 (통합 모델) ★★★★★ | 가능 | 중간 | 중간 |
| **모델 수 (15 클래스)** | 1개 | 1개 ★★★★★ | 1개 (API) | 15개 | 15개 |
| **메모리 (15 클래스)** | 500MB-1.5GB | 300-500MB ★★★★★ | API | 4.5-7.5GB | 6-9GB |
| **Image AUROC** | 91-95% | 98.8% (multi) ★★★★★ | 96-97% | 97.2% | 98.3% |
| **Pixel AUROC** | 89-93% | 97.5% ★★★★★ | 94-96% | 95.8% | 97.0% |
| **추론 속도** | 50-100ms | 80-120ms | 2-5초 | 40-60ms | 50-80ms |
| **학습 시간** | 0분 ★★★★★ | 2-4시간 | 0분 ★★★★★ | 1-3시간 | 2-4시간 |
| **비용** | 무료 | 무료 | $0.01-0.05/img | 무료 | 무료 |
| **설명 생성** | 없음 | 없음 | 자연어 ★★★★★ | 없음 | 없음 |
| **Zero-shot** | 예 ★★★★★ | 아니오 | 예 ★★★★★ | 아니오 | 아니오 |
| **신제품 대응** | 즉시 ★★★★★ | 빠름 | 즉시 ★★★★★ | 중간 | 중간 |
| **Interpretability** | 낮음 | 낮음 | 매우 높음 ★★★★★ | 낮음 | 낮음 |
| **주요 혁신** | CLIP zero-shot | Multi-class 통합 | Explainable AI | Hybrid 학습 | Contrastive |
| **적합 환경** | 신제품, 즉시 배포 | Multi-class 생산 | 품질 보고서 | 유연한 학습 | 강건성 필요 |
| **종합 평가** | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |

---

## 부록: 관련 테이블

### A.1 Foundation Model vs 전통적 패러다임

| 패러다임 | 대표 모델 | Image AUROC | Multi-class | Zero-shot | 주요 장점 |
|---------|----------|-------------|-------------|-----------|----------|
| **Foundation (Multi)** | Dinomaly | 98.8% | 우수 ★★★★★ | 아니오 | 단일 모델, 메모리 절감 |
| **Foundation (Zero)** | WinCLIP | 91-95% | 가능 | 예 ★★★★★ | 즉시 배포 |
| **Foundation (Explain)** | VLM-AD | 96-97% | 가능 | 예 | 자연어 설명 |
| Memory-Based | PatchCore | 99.1% | 불가능 | 아니오 | 최고 정확도 (single) |
| Normalizing Flow | FastFlow | 98.5% | 불가능 | 아니오 | 확률적 해석 |
| Knowledge Distillation | EfficientAd | 97.8% | 불가능 | 아니오 | 극한 속도 |

### A.2 응용 시나리오별 Foundation Model 선택

| 시나리오 | 권장 모델 | 이유 | 예상 성능 |
|---------|----------|------|----------|
| **Multi-class 생산 라인** | Dinomaly | 단일 모델로 모든 제품 | 98.8% AUROC |
| **신제품 즉시 검사** | WinCLIP | Zero-shot, 학습 불필요 | 91-95% AUROC |
| **품질 보고서 자동화** | VLM-AD | 자연어 설명 생성 | 96-97% AUROC |
| **다품종 소량 생산** | WinCLIP | 제품별 프롬프트만 | 91-95% AUROC |
| **일부 이상 샘플 있음** | SuperSimpleNet | Hybrid 학습 | 97.2% AUROC |
| **강건성 중시** | UniNet | Contrastive learning | 98.3% AUROC |

### A.3 Multi-class 비용 분석

**시나리오**: 15개 제품 카테고리 검사

**전통적 방법 (PatchCore)**:
```
모델 1: 500MB (Transistor)
모델 2: 500MB (Capacitor)
...
모델 15: 500MB (PCB)
────────────────────────
총 메모리: 7.5GB
관리 복잡도: 15개 모델 각각 버전 관리
배포 시간: 15 × 1시간 = 15시간
```

**Foundation Model (Dinomaly)**:
```
단일 모델: 500MB (모든 제품)
────────────────────────
총 메모리: 500MB (93% 절감)
관리 복잡도: 1개 모델만 관리
배포 시간: 3시간 (80% 단축)
```

**비용 절감**:
- GPU 메모리: 8GB → 2GB (저렴한 하드웨어)
- 개발 시간: 15시간 → 3시간
- 유지보수 비용: 대폭 감소

### A.4 Zero-shot vs Few-shot vs Fully Trained

| 방법 | 학습 데이터 | 학습 시간 | 정확도 | 적용 시나리오 |
|------|------------|----------|--------|--------------|
| **Zero-shot** (WinCLIP) | 0장 | 0분 | 91-95% | 신제품, 즉시 배포 |
| **Few-shot** (DRAEM) | 10-50장 | 2-4시간 | 97.5% | 데이터 부족 |
| **Fully Trained** (PatchCore) | 100-500장 | 1-2시간 | 99.1% | 충분한 데이터 |
| **Multi-class** (Dinomaly) | 100-500장 (각 클래스) | 2-4시간 (전체) | 98.8% | Multi-class 환경 |

### A.5 하드웨어 요구사항

| 모델 | GPU 메모리 (학습) | GPU 메모리 (추론) | CPU 추론 | 권장 환경 |
|------|-----------------|-----------------|----------|----------|
| **WinCLIP** | 불필요 | 2-4GB | 가능 (느림) | GPU 권장 |
| **Dinomaly** | 4-8GB | 2-4GB | 불가능 | GPU 필수 |
| **VLM-AD** | API | API | API | 인터넷 필요 |
| **SuperSimpleNet** | 4-8GB | 2-4GB | 느림 | GPU 권장 |
| **UniNet** | 4-8GB | 2-4GB | 불가능 | GPU 필수 |

### A.6 개발-배포 체크리스트 (Foundation Model)

**Phase 1: 시나리오 분석**
- [ ] Multi-class 환경인가?
- [ ] 학습 데이터 확보 가능한가?
- [ ] 즉시 배포가 필요한가?
- [ ] 설명 가능성이 중요한가?

**Phase 2: 모델 선택**
- [ ] Multi-class + 데이터 있음 → Dinomaly
- [ ] Zero-shot 필요 → WinCLIP
- [ ] 설명 필요 → VLM-AD
- [ ] 유연한 학습 → SuperSimpleNet

**Phase 3: WinCLIP 사용 시**
- [ ] 제품 이름 정의
- [ ] 효과적인 프롬프트 작성
- [ ] Normal/Defective 프롬프트 테스트
- [ ] Ensemble prompts 실험

**Phase 4: Dinomaly 사용 시**
- [ ] 모든 클래스 데이터 수집
- [ ] DINOv2 backbone 준비
- [ ] Class-conditional memory bank 구축
- [ ] Multi-class 통합 학습

**Phase 5: VLM-AD 사용 시**
- [ ] API 키 발급 (GPT-4V, Claude 등)
- [ ] 프롬프트 템플릿 설계
- [ ] Few-shot 예시 준비
- [ ] 비용 계산 ($0.01-0.05/img)

**Phase 6: 평가 및 배포**
- [ ] 각 클래스별 성능 확인
- [ ] Multi-class 전체 성능 측정
- [ ] 추론 속도 벤치마크
- [ ] 메모리 사용량 모니터링

### A.7 Foundation Model 선택 의사결정 트리

```
이상 탐지 필요
│
├─ Multi-class 환경?
│   ├─ YES → Dinomaly
│   │   - 단일 모델로 모든 클래스
│   │   - 98.8% AUROC
│   │   - 메모리 80-90% 절감
│   │
│   └─ NO → 다른 패러다임 고려
│       - Single-class: PatchCore (99.1%)
│
├─ 학습 데이터 없음?
│   ├─ YES → WinCLIP
│   │   - Zero-shot
│   │   - 즉시 배포
│   │   - 91-95% AUROC
│   │
│   └─ NO → 데이터 활용 모델
│
├─ 설명 가능성 필수?
│   └─ YES → VLM-AD
│       - 자연어 설명
│       - 품질 보고서 자동화
│       - API 비용 고려
│
└─ 일반적 상황
    └─ 정확도 우선 → PatchCore
        속도 우선 → EfficientAd
        균형 → FastFlow
```

### A.8 성능 벤치마크 요약

**Multi-class 성능**:
1. Dinomaly (98.8%) ★★★★★
2. UniNet (98.3%)
3. SuperSimpleNet (97.2%)
4. VLM-AD (96-97%)
5. WinCLIP (91-95%)

**Zero-shot 성능**:
1. VLM-AD (96-97%) ★★★★☆
2. WinCLIP (91-95%) ★★★★☆

**설명 가능성**:
1. VLM-AD (자연어 설명) ★★★★★
2. 나머지 (숫자 점수만)

**메모리 효율 (Multi-class, 15개 클래스)**:
1. Dinomaly (300-500MB) ★★★★★
2. WinCLIP (500MB-1.5GB)
3. VLM-AD (API, 로컬 메모리 불필요)
4. 전통적 방법 (1.5-7.5GB)

### A.9 Foundation Model의 미래 전망

**현재 트렌드**:
1. **Multi-class 통합**: Dinomaly → 산업 표준화
2. **Zero-shot 확대**: WinCLIP → 더 많은 도메인
3. **Explainable AI**: VLM-AD → 필수 요구사항
4. **Domain-specific FM**: 산업 특화 foundation model 등장 예정

**향후 발전 방향**:
1. **더 강력한 Backbone**: DINOv2 → DINOv3, SAM2 등
2. **Edge Foundation Models**: 경량화된 FM for 엣지
3. **Multi-modal Fusion**: 이미지 + 센서 데이터
4. **Continual Learning**: 지속적 업데이트
5. **Domain Adaptation**: 산업용 특화 fine-tuning

**산업 적용 전망**:
- 2025-2026: Multi-class 모델 보편화 (Dinomaly 방식)
- 2026-2027: Zero-shot 모델 확산 (신제품 대응)
- 2027+: Explainable AI 필수화 (규제 대응)

---

## 결론

Foundation Model 기반 방식은 **Multi-class, Zero-shot, Explainable AI**로 이상 탐지의 패러다임을 전환하고 있다.

**핵심 발견**:

1. **Dinomaly (2025)**: 
   - Multi-class 혁명: 단일 모델로 98.8% AUROC
   - 메모리 80-90% 절감
   - 운영 간소화
   - **가장 추천되는 Foundation Model**

2. **WinCLIP (2023)**:
   - Zero-shot: 학습 데이터 불필요
   - 즉시 배포: 프롬프트만으로 91-95%
   - 신제품, 다품종 소량 생산에 이상적

3. **VLM-AD (2024)**:
   - Explainable: 자연어 설명 생성
   - 품질 보고서 자동화
   - 근본 원인 분석
   - **API 비용과 속도 trade-off**

4. **SuperSimpleNet, UniNet**:
   - 실용적 접근 (Hybrid, Contrastive)
   - 중간 수준 성능 (97-98%)

**패러다임 전환**:

전통적 방법:
- Single-class, 클래스당 모델 필요
- 100-500장 학습 데이터 필수
- 숫자 점수만 제공

Foundation Model:
- Multi-class, 단일 통합 모델
- Zero-shot 가능 (또는 적은 데이터)
- 설명 가능 (VLM-AD)

**최종 권장사항**:

**Foundation Model 사용 시나리오**:
- ✅ **Multi-class 환경**: Dinomaly (98.8%, 메모리 절감)
- ✅ **신제품/즉시 배포**: WinCLIP (zero-shot, 91-95%)
- ✅ **품질 보고서**: VLM-AD (자연어 설명)
- ✅ **다품종 소량**: WinCLIP (제품별 프롬프트)

**전통적 패러다임 사용 권장**:
- ✅ **Single-class 최고 정확도**: PatchCore (99.1%)
- ✅ **실시간 처리**: EfficientAd (1-5ms)
- ✅ **균형 성능**: FastFlow (98.5%)

Foundation Model은 특히 **Multi-class 환경과 Zero-shot 요구사항**에서 강력하며, Dinomaly는 2025년 Multi-class 이상 탐지의 새로운 표준이 될 것으로 전망된다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: WinCLIP, Dinomaly, VLM-AD, SuperSimpleNet, UniNet

**주요 내용**:
1. Foundation Model 패러다임 개요 (CLIP, DINOv2, GPT-4V)
2. WinCLIP 상세 분석 (Zero-shot, 프롬프트 기반)
3. Dinomaly 상세 분석 (Multi-class 혁명, DINOv2)
4. VLM-AD 상세 분석 (Explainable AI, GPT-4V)
5. SuperSimpleNet, UniNet 개요
6. 종합 비교 및 미래 전망
7. **부록**: overall_report.md의 관련 테이블 포함
