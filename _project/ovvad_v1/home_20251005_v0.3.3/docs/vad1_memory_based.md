# 1. Memory-Based / Feature Matching 방식 상세 분석

## 1.1 패러다임 개요

Memory-based 방식은 정상 샘플의 특징 벡터를 메모리에 저장하고, 테스트 시점에 저장된 특징과의 거리(distance) 또는 유사도(similarity)를 계산하여 이상을 탐지하는 접근법이다.

**핵심 수식**:
$$\text{Anomaly Score} = d(f_{\text{test}}, \mathcal{M}_{\text{normal}})$$

여기서:
- $f_{\text{test}}$: 테스트 샘플의 특징 벡터
- $\mathcal{M}_{\text{normal}}$: 정상 샘플들의 메모리 뱅크
- $d(\cdot, \cdot)$: 거리 함수 (Mahalanobis, Euclidean 등)

**핵심 가정**: "정상 샘플들은 특징 공간(feature space)에서 밀집된 분포를 형성하며, 이상 샘플은 이 분포에서 멀리 떨어져 있다"

---

## 1.2 PaDiM (2020)

### 1.2.1 기본 정보

- **논문**: Patch Distribution Modeling Framework for Anomaly Detection and Localization
- **발표**: ICPR 2020
- **저자**: Thomas Defard et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/padim

### 1.2.2 핵심 원리

PaDiM은 이미지를 패치(patch) 단위로 분할하고, 각 공간적 위치에서 정상 패턴의 확률 분포를 모델링한다.

**수학적 정식화**:

각 패치 위치 $(i,j)$에서:

$$p(\mathbf{x}_{i,j}) = \mathcal{N}(\boldsymbol{\mu}_{i,j}, \boldsymbol{\Sigma}_{i,j})$$

여기서:
- $\mathbf{x}_{i,j} \in \mathbb{R}^d$: 위치 $(i,j)$의 특징 벡터
- $\boldsymbol{\mu}_{i,j}$: 평균 벡터
- $\boldsymbol{\Sigma}_{i,j}$: 공분산 행렬

**이상 점수 계산**: Mahalanobis distance

$$M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

### 1.2.3 기술적 세부사항

**특징 추출**:
- Pre-trained CNN (ResNet, WideResNet) 사용
- 여러 레이어에서 특징 추출 (layer1, layer2, layer3)
- Multi-scale representation: $\mathbf{f} = [\mathbf{f}^{(1)}, \mathbf{f}^{(2)}, \mathbf{f}^{(3)}]$
- 특징 차원: 일반적으로 100-550 차원

**확률 분포 모델링**:
1. 모든 학습 이미지에서 각 위치 $(i,j)$의 특징 수집
2. 평균 벡터 $\boldsymbol{\mu}_{i,j}$ 계산
3. 공분산 행렬 $\boldsymbol{\Sigma}_{i,j}$ 계산
4. 차원 축소: Random projection으로 $d \rightarrow d'$ (계산 효율)

**추론 과정**:
1. 테스트 이미지의 각 패치에서 Mahalanobis distance 계산
2. Distance map 생성 (pixel-level anomaly localization)
3. 최대값 또는 평균값을 image-level anomaly score로 사용

### 1.2.4 성능

**MVTec AD 벤치마크**:
- Image AUROC: 96.5%
- Pixel AUROC: 95.8%
- 추론 속도: 30-50ms per image (GPU)
- 메모리: 2-5GB (데이터셋 크기에 따라)

### 1.2.5 장점

1. **직관적 해석**: 통계적 거리 기반으로 명확한 의미
2. **빠른 추론**: Forward pass만 필요, 추가 네트워크 학습 불필요
3. **안정적 성능**: 하이퍼파라미터에 덜 민감
4. **Pixel-level localization**: 명확한 anomaly map 제공

### 1.2.6 단점

1. **높은 메모리 사용**:
   - 모든 패치 위치에서 공분산 행렬 저장
   - 예: 224×224 이미지, 28×28 feature map → 784개의 공분산 행렬
   - 각 공분산: $d \times d$ 크기 (d=550일 경우 302,500개 파라미터)
   - 총 메모리: 수 GB 수준

2. **학습 데이터 증가 시 메모리 선형 증가**: 공분산 계산에 모든 샘플 필요

3. **고차원 문제**: 차원이 높을수록 공분산 추정의 신뢰도 감소

---

## 1.3 PatchCore (2022)

### 1.3.1 기본 정보

- **논문**: Towards Total Recall in Industrial Anomaly Detection
- **발표**: CVPR 2022
- **저자**: Karsten Roth et al.
- **GitHub**: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib/models/image/patchcore

### 1.3.2 핵심 원리

PatchCore는 PaDiM의 메모리 문제를 해결하기 위해 **Coreset Selection** 알고리즘을 도입했다. 모든 학습 패치를 저장하는 대신, 전체 분포를 대표할 수 있는 소수의 핵심 패치만 선택한다.

**핵심 아이디어**: "모든 정상 패턴을 저장할 필요 없이, 패턴 공간을 충분히 커버하는 대표 샘플만 있으면 된다"

**수학적 정식화**:

$$\mathcal{C} = \underset{|\mathcal{C}|=M}{\arg\min} \max_{\mathbf{x} \in \mathcal{X}} \min_{\mathbf{c} \in \mathcal{C}} \|\mathbf{x} - \mathbf{c}\|_2$$

여기서:
- $\mathcal{X}$: 모든 학습 패치 집합
- $\mathcal{C}$: Coreset (크기 $M$)
- $M \ll |\mathcal{X}|$ (보통 $M = 0.01 \sim 0.1 \times |\mathcal{X}|$)

### 1.3.3 PaDiM 대비 핵심 차이점

| 측면 | PaDiM | PatchCore | 개선 효과 |
|------|-------|-----------|----------|
| **저장 방식** | 각 위치별 분포 $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | 대표 패치 집합 (coreset) | 메모리 90%+ 감소 |
| **거리 측정** | Mahalanobis distance | Euclidean distance (k-NN) | 계산 간소화 |
| **특징 선택** | Random projection | Locally aware patch features | 품질 향상 |
| **커버리지** | 전체 위치 | Coreset으로 전체 커버 | 성능 유지/향상 |
| **메모리** | 2-5GB | 100-500MB | 90-95% 감소 |
| **Image AUROC** | 96.5% | 99.1% | +2.6%p |

### 1.3.4 기술적 세부사항

**Locally Aware Patch Features**:
- 인접 픽셀 정보를 포함한 local context 반영
- Adaptive average pooling으로 neighborhood 정보 통합
- 더 discriminative한 특징 생성

**Greedy Coreset Subsampling 알고리즘**:

```
입력: N개의 패치 특징 {f_1, f_2, ..., f_N}
출력: M개의 coreset C (M << N)

1. 초기화: C = {}, remaining = {f_1, ..., f_N}
2. 첫 번째 샘플 무작위 선택: C = {f_random}
3. For i = 2 to M:
   a. For each f in remaining:
      distance = min_{c ∈ C} ||f - c||_2
   b. f_max = argmax_f distance  # 가장 먼 샘플 선택
   c. C = C ∪ {f_max}
4. Return C
```

**Greedy Coreset의 원리**:
- **Maximum Distance Strategy**: 각 단계에서 기존 coreset과 가장 먼 샘플 선택
- **Coverage 보장**: 모든 학습 샘플이 coreset의 어떤 샘플과는 가까이 위치
- **이론적 근거**: ε-cover 이론 - 작은 coreset으로 전체 공간을 ε 반경으로 커버

**k-NN 기반 이상 탐지**:

테스트 패치 $\mathbf{f}_{\text{test}}$에 대해:

$$\text{Anomaly Score} = \frac{1}{k}\sum_{i=1}^{k} \|\mathbf{f}_{\text{test}} - \mathbf{c}_i\|_2$$

여기서 $\mathbf{c}_i$는 $k$개의 nearest neighbors in coreset

### 1.3.5 PaDiM 대비 개선사항

**1) 메모리 효율성**:
- PaDiM: 28×28 feature map = 784개 공분산 행렬 = ~2-5GB
- PatchCore: 전체 패치의 1% coreset = ~100-200MB
- **개선율**: 90-95% 메모리 감소

**2) 성능 향상**:
- PaDiM: 96.5% AUROC
- PatchCore: 99.1% AUROC
- **개선율**: +2.6%p

**3) 계산 복잡도**:
- PaDiM: $O(d^2)$ - Mahalanobis distance (공분산 역행렬)
- PatchCore: $O(d)$ - Euclidean distance
- k-NN search는 FAISS 등으로 최적화 가능

**4) 확장성**:
- PaDiM: 학습 데이터 증가 시 메모리 선형 증가
- PatchCore: Coreset 크기 고정 가능, 메모리 제어 가능

### 1.3.6 성능

**MVTec AD 벤치마크**:
- Image AUROC: 99.1% (현재까지 single-class 최고)
- Pixel AUROC: 98.2%
- 추론 속도: 50-100ms per image (GPU)
- 메모리: 100-500MB (coreset 크기에 따라)

### 1.3.7 장점

1. **SOTA 정확도**: MVTec AD에서 최고 수준 성능
2. **메모리 효율**: PaDiM 대비 90% 이상 감소
3. **확장 가능**: 대규모 데이터셋에도 적용 가능
4. **이론적 보장**: Coreset의 coverage 수학적으로 보장

### 1.3.8 단점

1. **Coreset 선택 시간**: 
   - 학습 단계에서 greedy selection 시간 소요
   - N개 패치에서 M개 선택: $O(NM)$ 복잡도
   - 대규모 데이터셋에서 수십 분 소요 가능

2. **k-NN search 오버헤드**: 
   - 추론 시 coreset 전체와 거리 계산 필요
   - Approximate NN (FAISS 등) 사용으로 완화 가능

3. **하이퍼파라미터 민감성**: Coreset 크기 선택이 성능에 영향

---

## 1.4 DFKDE (2022)

### 1.4.1 기본 정보

- **논문**: Deep Feature Kernel Density Estimation
- **저자**: Anomalib team
- **GitHub**: https://github.com/openvinotoolkit/anomalib

### 1.4.2 핵심 원리

DFKDE는 딥러닝 특징에 전통적인 통계학의 **Kernel Density Estimation**을 적용한다.

**KDE 기본 수식**:

$$\hat{p}(\mathbf{x}) = \frac{1}{Nh} \sum_{i=1}^{N} K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$

여기서:
- $K(\cdot)$: Kernel 함수 (일반적으로 Gaussian)
- $h$: Bandwidth (smoothing parameter)
- $N$: 학습 샘플 수

**이상 점수**:

$$\text{Anomaly Score} = -\log \hat{p}(\mathbf{x})$$

낮은 확률 밀도 = 높은 이상 점수

### 1.4.3 PaDiM/PatchCore와의 차이점

| 측면 | PaDiM | PatchCore | DFKDE |
|------|-------|-----------|-------|
| **분포 가정** | Parametric (Gaussian) | Non-parametric (Exemplar) | Non-parametric (KDE) |
| **저장 내용** | 평균, 공분산 | Coreset 패치 | 모든 특징 (또는 샘플링) |
| **거리 측정** | Mahalanobis | Euclidean (k-NN) | Kernel-based density |
| **유연성** | 단일 가우시안 | 복잡한 분포 가능 | 임의 분포 가능 |

### 1.4.4 기술적 세부사항

**Gaussian Kernel**:

$$K(\mathbf{u}) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{\|\mathbf{u}\|^2}{2}\right)$$

**Bandwidth 선택**:
- Scott's rule: $h = n^{-1/(d+4)} \cdot \sigma$
- Silverman's rule: $h = \left(\frac{4\sigma^5}{3n}\right)^{1/5}$
- Cross-validation으로 최적화

**차원 축소**: PCA로 고차원 curse 완화

### 1.4.5 성능

**MVTec AD 벤치마크**:
- Image AUROC: 95.5-96.8% (구현에 따라)
- 추론 속도: 50-100ms per image
- 메모리: 1-3GB (샘플 저장 방식에 따라)

### 1.4.6 장점

1. **분포 유연성**: 가우시안 가정 불필요, 복잡한 분포 모델링 가능
2. **수학적 해석**: 확률 밀도 기반으로 명확한 의미
3. **신뢰구간 제공**: 통계적 유의성 검정 가능

### 1.4.7 단점

1. **Curse of Dimensionality**: 
   - 고차원에서 KDE 성능 저하
   - 차원이 높을수록 샘플 간 거리 차이 감소
   - 필요 샘플 수가 지수적으로 증가

2. **메모리 사용**: 모든 (또는 많은) 학습 샘플 저장 필요

3. **계산 비용**: 추론 시 모든 저장 샘플과 거리 계산

4. **Bandwidth 민감성**: $h$ 선택이 성능에 큰 영향

---

## 1.5 Memory-Based 방식 종합 비교

### 1.5.1 기술적 진화 과정

```
PaDiM (2020)
├─ 기여: 패치별 다변량 가우시안 모델링
├─ 문제: 높은 메모리 사용량 (2-5GB)
├─ 원인: 모든 패치 위치에서 공분산 행렬 저장
└─ 성능: 96.5% AUROC

        ↓ 메모리 효율화

PatchCore (2022)
├─ 혁신: Coreset Selection
├─ 해결: 메모리 90% 감소 (100-500MB)
├─ 추가: Locally aware features
├─ 성능: 99.1% AUROC (+2.6%p)
└─ 영향: Memory-based 방식의 SOTA, 산업 적용성 크게 향상

        ↓ 통계적 변형

DFKDE (2022)
├─ 접근: KDE로 유연한 분포 모델링
├─ 장점: 비모수적 방법, 복잡한 분포 가능
└─ 한계: 고차원 curse, PatchCore보다 낮은 성능
```

### 1.5.2 상세 비교표

| 비교 항목 | PaDiM | PatchCore | DFKDE |
|----------|-------|-----------|-------|
| **발표 연도** | 2020 | 2022 | 2022 |
| **분포 모델** | Parametric Gaussian | Non-parametric Exemplar | Non-parametric KDE |
| **저장 구조** | $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ per location | Coreset patches | Feature samples |
| **메모리 사용** | 2-5GB | 100-500MB | 1-3GB |
| **메모리 효율** | 낮음 | 매우 높음 ★★★★★ | 중간 |
| **거리 측정** | Mahalanobis | Euclidean (k-NN) | Kernel density |
| **계산 복잡도 (추론)** | $O(d^2)$ | $O(Md) + k$-NN | $O(Nd)$ |
| **Image AUROC** | 96.5% | 99.1% ★★★★★ | 95.5-96.8% |
| **Pixel AUROC** | 95.8% | 98.2% ★★★★★ | 94.5-96.0% |
| **추론 속도** | 30-50ms | 50-100ms | 50-100ms |
| **분포 유연성** | 단일 가우시안만 | 임의 분포 | 임의 분포 |
| **고차원 강건성** | 중간 | 높음 ★★★★☆ | 낮음 (curse) |
| **확장성** | 낮음 | 높음 ★★★★★ | 낮음 |
| **하이퍼파라미터** | 적음 | 중간 (coreset 크기, k) | 많음 (bandwidth, PCA) |
| **구현 난이도** | 낮음 | 중간 | 중간 |
| **이론적 근거** | Mahalanobis 거리 | ε-cover 이론 | KDE 이론 |
| **종합 평가** | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |

### 1.5.3 핵심 기여 및 영향

**PaDiM의 기여**:
- Memory-based 접근의 효과성 입증
- Patch 단위 분포 모델링 패러다임 확립
- 간단하면서도 강력한 baseline 제공
- 이후 연구의 기초가 됨

**PatchCore의 기여**:
- Coreset selection을 통한 메모리 문제 해결
- Memory-based 방법의 SOTA 달성 (99.1%)
- 산업 적용 가능성 크게 향상 (메모리 효율)
- 이후 연구의 주요 baseline이 됨
- 현재까지 single-class에서 최고 성능

**DFKDE의 기여**:
- 통계적 방법론의 적용 가능성 탐색
- 비모수적 접근의 유연성 제시
- 하지만 실용적으로는 PatchCore에 밀림

### 1.5.4 실무 적용 가이드

**PaDiM 선택 시나리오**:
- 빠른 프로토타이핑 필요
- 구현 복잡도 최소화 원함
- 메모리 제약이 크지 않음
- 중간 수준 성능으로 충분
- **추천도**: ★★★☆☆

**PatchCore 선택 시나리오**:
- 최고 정확도 필요 (99%+)
- 메모리 효율 중요
- 대규모 데이터셋
- 장기 운영 예정 (안정성)
- Single-class 환경
- **추천도**: ★★★★★ (최고 추천)

**DFKDE 선택 시나리오**:
- 통계적 해석 필요
- 비정규 분포 의심
- 연구 목적
- **추천도**: ★☆☆☆☆ (실무에서는 비추천)

---

## 부록: 관련 테이블

### A.1 전체 패러다임 성능 비교

| 패러다임 | 대표 모델 | Image AUROC | 추론 속도 | 메모리 | 주요 장점 |
|---------|----------|-------------|-----------|--------|----------|
| **Memory-Based** | PatchCore | **99.1%** | 50-100ms | 100-500MB | 최고 정확도 |
| Normalizing Flow | FastFlow | 98.5% | 20-50ms | 500MB-1GB | 속도-정확도 균형 |
| Knowledge Distillation | Reverse Distillation | 98.6% | 100-200ms | 500MB-1GB | SOTA급 정확도 |
| Knowledge Distillation | EfficientAd | 97.8% | **1-5ms** | **<200MB** | 극한 속도 |
| Reconstruction | DRAEM | 97.5% | 50-100ms | 300-500MB | Few-shot |
| Foundation Model | Dinomaly | 98.8% (multi) | 80-120ms | 1.5-2GB | Multi-class |

### A.2 응용 시나리오별 Memory-Based 모델 선택

| 시나리오 | 권장 모델 | 이유 | 예상 성능 |
|---------|----------|------|----------|
| **최고 정확도 필수** | PatchCore | Single-class SOTA | 99.1% AUROC |
| **빠른 프로토타입** | PaDiM | 간단한 구현 | 96.5% AUROC |
| **메모리 제약 심함** | PatchCore | 100-500MB | 99.1% AUROC |
| **대규모 데이터셋** | PatchCore | Coreset 확장성 | 99.1% AUROC |
| **통계적 해석 필요** | DFKDE | KDE 이론 | 95.5-96.8% AUROC |

### A.3 Memory-Based vs 다른 패러다임

| 측면 | Memory-Based (PatchCore) | Normalizing Flow (FastFlow) | KD (EfficientAd) |
|------|------------------------|---------------------------|-----------------|
| **정확도** | 99.1% ★★★★★ | 98.5% ★★★★☆ | 97.8% ★★★★☆ |
| **속도** | 50-100ms ★★★☆☆ | 20-50ms ★★★★☆ | 1-5ms ★★★★★ |
| **메모리** | 100-500MB ★★★★☆ | 500MB-1GB ★★★☆☆ | <200MB ★★★★★ |
| **학습 복잡도** | 중간 ★★★☆☆ | 높음 ★★☆☆☆ | 중간 ★★★☆☆ |
| **해석 가능성** | 높음 ★★★★★ | 중간 ★★★☆☆ | 낮음 ★★☆☆☆ |
| **적용 분야** | 정밀 검사 | 균형잡힌 검사 | 실시간 검사 |

### A.4 하드웨어 요구사항

| 모델 | GPU 메모리 | CPU 추론 | 권장 환경 |
|------|-----------|----------|----------|
| **PaDiM** | 4GB+ | 느림 (100-200ms) | GPU 권장 |
| **PatchCore** | 4-8GB | 매우 느림 (500ms+) | GPU 필수 |
| **DFKDE** | 4GB+ | 느림 (150-300ms) | GPU 권장 |

### A.5 개발-배포 체크리스트 (Memory-Based)

**Phase 1: 모델 선택**
- [ ] 정확도 목표 설정 (95%? 99%+?)
- [ ] 메모리 제약 확인
- [ ] PatchCore vs PaDiM 결정

**Phase 2: 데이터 준비**
- [ ] 정상 샘플 수집 (100-500장 권장)
- [ ] 이미지 전처리 (resize, normalize)
- [ ] Train/validation split

**Phase 3: 학습 및 평가**
- [ ] Pre-trained backbone 선택 (ResNet18/WideResNet50)
- [ ] PatchCore: Coreset 크기 설정 (1-10%)
- [ ] Validation set에서 성능 확인
- [ ] Pixel-level localization 품질 검증

**Phase 4: 최적화 (선택사항)**
- [ ] FAISS를 이용한 k-NN 가속화
- [ ] Coreset 크기 조정 (성능 vs 메모리 trade-off)
- [ ] Batch inference 구현

**Phase 5: 배포**
- [ ] 추론 시간 벤치마크
- [ ] 메모리 사용량 모니터링
- [ ] 임계값 설정 및 검증

---

## 결론

Memory-based 방식은 **PatchCore를 정점으로 현재까지 single-class 이상 탐지에서 최고 성능(99.1% AUROC)**을 보여주고 있다. 특히:

1. **정확도**: 모든 패러다임 중 최고 (99.1%)
2. **메모리 효율**: Coreset selection으로 실용적 수준 (100-500MB)
3. **이론적 견고성**: 명확한 수학적 기반 (거리 측정, ε-cover)
4. **산업 적용**: 제조업 품질 검사에서 널리 채택

**최종 권장사항**:
- **대부분의 single-class 환경**: PatchCore 사용 (99.1% 정확도)
- **빠른 프로토타입**: PaDiM 사용 (96.5% 정확도, 간단)
- **Multi-class 환경**: Dinomaly 등 Foundation Model 고려

Memory-based 방식은 당분간 이상 탐지의 핵심 패러다임으로 남을 것으로 전망된다.

---

**문서 버전**: 1.0  
**작성일**: 2025  
**분석 모델**: PaDiM, PatchCore, DFKDE

**주요 내용**:
1. 패러다임 개요 (핵심 수식 포함)
2. PaDiM 상세 분석 (LaTeX 수식 중심)
3. PatchCore 상세 분석 (Coreset 알고리즘 설명)
4. DFKDE 상세 분석
5. 종합 비교 및 실무 가이드
6. **부록**: overall_report.md의 관련 테이블 포함
