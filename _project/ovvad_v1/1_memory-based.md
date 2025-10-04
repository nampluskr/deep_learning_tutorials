## 1. Memory-Based 방식 상세 분석

### 1.1 패러다임 개요

Memory-based 방식은 정상 샘플의 특징 벡터를 메모리에 저장하고, 테스트 시점에 저장된 특징과의 거리(distance) 또는 유사도(similarity)를 계산하여 이상을 탐지하는 접근법이다. 이 방식의 핵심 가정은 "정상 샘플들은 특징 공간(feature space)에서 밀집된 분포를 형성하며, 이상 샘플은 이 분포에서 멀리 떨어져 있다"는 것이다.

### 1.2 PaDiM (2020) - 패러다임의 기초 확립

#### 1.2.1 핵심 원리
PaDiM(Patch Distribution Modeling)은 이미지를 패치(patch) 단위로 분할하고, 각 공간적 위치에서 정상 패턴의 확률 분포를 모델링한다.

**수학적 정식화**:
- 각 패치 위치 (i,j)에서 특징 벡터 추출: f_{i,j} ∈ R^d
- 정상 샘플들의 특징으로 다변량 가우시안 분포 추정: N(μ_{i,j}, Σ_{i,j})
- 이상 점수 = Mahalanobis distance: M(x) = √((x - μ)^T Σ^(-1) (x - μ))

#### 1.2.2 기술적 세부사항

**특징 추출**:
- Pre-trained ResNet 또는 WideResNet 사용
- 여러 레이어(layer1, layer2, layer3)에서 특징 추출
- 각 레이어의 특징을 concatenate하여 multi-scale representation 구성
- 특징 차원: 일반적으로 100-550 차원

**확률 분포 모델링**:
```
For each spatial location (i,j):
1. 모든 학습 이미지에서 해당 위치의 특징 수집
2. 평균 벡터 μ_{i,j} 계산
3. 공분산 행렬 Σ_{i,j} 계산
4. 차원 축소: Random projection으로 d 차원을 d' 차원으로 축소 (계산 효율)
```

**이상 탐지**:
- 테스트 이미지의 각 패치에서 Mahalanobis distance 계산
- Distance map 생성 (pixel-level anomaly localization)
- 최대값 또는 평균값을 image-level anomaly score로 사용

#### 1.2.3 장점
- **직관적 해석**: 통계적 거리 기반으로 명확한 의미
- **빠른 추론**: Forward pass만 필요, 추가 네트워크 학습 불필요
- **안정적 성능**: 하이퍼파라미터에 덜 민감
- **Localization 능력**: Pixel-level anomaly map 제공

#### 1.2.4 단점
- **높은 메모리 사용**: 모든 패치 위치에서 공분산 행렬 저장
  - 예: 224×224 이미지, 28×28 feature map → 784개의 공분산 행렬
  - 각 공분산 행렬: d×d 크기 (d=550일 경우 302,500개 파라미터)
  - 총 메모리: 수 GB 수준
- **학습 데이터 증가 시 메모리 선형 증가**: 공분산 계산에 모든 샘플 필요
- **고차원 문제**: 차원이 높을수록 공분산 추정의 신뢰도 감소

#### 1.2.5 성능
- MVTec AD: Image AUROC 96.5%, Pixel AUROC 95.8%
- 추론 속도: 30-50ms per image (GPU)
- 메모리: 2-5GB (데이터셋 크기에 따라)

---

### 1.3 PatchCore (2022) - Memory 효율성의 혁신

#### 1.3.1 핵심 원리
PatchCore는 PaDiM의 메모리 문제를 해결하기 위해 Coreset Selection 알고리즘을 도입했다. 모든 학습 패치를 저장하는 대신, 전체 분포를 대표할 수 있는 소수의 핵심 패치만 선택한다.

**핵심 아이디어**:
- "모든 정상 패턴을 저장할 필요 없이, 패턴 공간을 충분히 커버하는 대표 샘플만 있으면 된다"

#### 1.3.2 PaDiM 대비 핵심 차이점

| 측면 | PaDiM | PatchCore | 개선 효과 |
|------|-------|-----------|----------|
| **저장 방식** | 각 위치별 분포 (μ, Σ) | 대표 패치 집합 (coreset) | 메모리 90%+ 감소 |
| **거리 측정** | Mahalanobis distance | Euclidean distance (k-NN) | 계산 간소화 |
| **특징 선택** | Random projection | Locally aware patch features | 품질 향상 |
| **커버리지** | 전체 위치 | Coreset으로 전체 커버 | 성능 유지/향상 |

#### 1.3.3 기술적 세부사항

**Locally Aware Patch Features**:
```python
# PaDiM: 여러 레이어 concatenation
features_padim = concat([layer1, layer2, layer3])

# PatchCore: Adaptive average pooling으로 local 정보 보존
features_patchcore = adaptive_avg_pool(layer2_layer3, neighborhood=3×3)
```
- 인접 픽셀 정보를 포함한 local context 반영
- 더 discriminative한 특징 생성

**Coreset Selection 알고리즘**:
```
입력: N개의 패치 특징 {f_1, f_2, ..., f_N}
출력: M개의 coreset C (M << N, 보통 M = 0.01~0.1 × N)

1. 초기화: C = {}, remaining = {f_1, ..., f_N}
2. 첫 번째 샘플 무작위 선택: C = {f_random}
3. For i = 2 to M:
   a. For each f in remaining:
      - distance = min_{c ∈ C} ||f - c||_2
   b. f_max = argmax_f distance  # 가장 먼 샘플 선택
   c. C = C ∪ {f_max}
   d. remaining = remaining \ {f_max}
4. Return C
```

**Greedy Coreset Subsampling의 원리**:
- **Maximum Distance Strategy**: 각 단계에서 기존 coreset과 가장 먼 샘플 선택
- **Coverage 보장**: 모든 학습 샘플이 coreset의 어떤 샘플과는 가까이 위치하도록 보장
- **이론적 근거**: ε-cover 이론 - 작은 coreset으로 전체 공간을 ε 반경으로 커버

**이상 탐지 (k-NN based)**:
```
For test patch f_test:
1. Coreset에서 k개의 nearest neighbors 찾기
2. 거리 계산: distances = [||f_test - c_i||_2 for i in top_k]
3. Anomaly score = max(distances) or mean(distances)
```

#### 1.3.4 PaDiM 대비 개선사항

**1) 메모리 효율성**:
- PaDiM: 28×28 feature map = 784개 공분산 행렬 = ~2-5GB
- PatchCore: 전체 패치의 1% coreset = ~100-200MB
- **개선율**: 90-95% 메모리 감소

**2) 성능 향상**:
- PaDiM: 96.5% AUROC
- PatchCore: 99.1% AUROC
- **개선율**: +2.6%p

**3) 계산 복잡도**:
- PaDiM: O(d²) - Mahalanobis distance 계산 (공분산 역행렬)
- PatchCore: O(d) - Euclidean distance 계산
- k-NN search 오버헤드 있으나 전체적으로 더 효율적

**4) 확장성**:
- PaDiM: 학습 데이터 증가 시 메모리 선형 증가
- PatchCore: Coreset 크기 고정 가능, 메모리 제어 가능

#### 1.3.5 장점
- **SOTA 정확도**: MVTec AD에서 최고 수준 성능
- **메모리 효율**: PaDiM 대비 90% 이상 감소
- **확장 가능**: 대규모 데이터셋에도 적용 가능
- **이론적 보장**: Coreset의 coverage 수학적으로 보장

#### 1.3.6 단점
- **Coreset 선택 시간**: 학습 단계에서 greedy selection 시간 소요
  - N개 패치에서 M개 선택: O(NM) 복잡도
  - 대규모 데이터셋에서 수십 분 소요 가능
- **k-NN search 오버헤드**: 추론 시 coreset 전체와 거리 계산 필요
  - Approximate NN (FAISS 등) 사용으로 완화 가능
- **하이퍼파라미터 민감성**: Coreset 크기 선택이 성능에 영향

#### 1.3.7 성능
- MVTec AD: Image AUROC 99.1%, Pixel AUROC 98.2%
- 추론 속도: 50-100ms per image (GPU)
- 메모리: 100-500MB (coreset 크기에 따라)

---

### 1.4 DFKDE (2022) - 통계적 접근의 변형

#### 1.4.1 핵심 원리
DFKDE(Deep Feature Kernel Density Estimation)는 딥러닝 특징에 전통적인 통계학의 Kernel Density Estimation을 적용한다.

**KDE 기본 원리**:
- 확률 밀도 함수(PDF) 추정: p(x) = (1/Nh) Σ K((x - x_i)/h)
- K: Kernel 함수 (일반적으로 Gaussian)
- h: Bandwidth (smoothing parameter)

#### 1.4.2 PaDiM/PatchCore와의 차이점

| 측면 | PaDiM | PatchCore | DFKDE |
|------|-------|-----------|-------|
| **분포 가정** | Parametric (Gaussian) | Non-parametric (Exemplar) | Non-parametric (KDE) |
| **저장 내용** | 평균, 공분산 | Coreset 패치 | 모든 특징 (또는 샘플링) |
| **거리 측정** | Mahalanobis | Euclidean (k-NN) | Kernel-based density |
| **유연성** | 단일 가우시안 | 복잡한 분포 가능 | 임의 분포 가능 |

#### 1.4.3 기술적 세부사항

**특징 추출 및 차원 축소**:
- Pre-trained CNN에서 특징 추출 (PaDiM과 유사)
- PCA로 차원 축소 (고차원 curse 완화)
- 축소된 특징에 KDE 적용

**Kernel Density Estimation**:
```python
# 학습: 모든 정상 샘플의 특징 저장
train_features = [f_1, f_2, ..., f_N]

# 추론: 테스트 샘플의 밀도 추정
def anomaly_score(f_test):
    density = 0
    for f_train in train_features:
        distance = ||f_test - f_train||_2
        density += gaussian_kernel(distance, bandwidth=h)
    density /= N
    return -log(density)  # 낮은 밀도 = 높은 이상 점수
```

**Bandwidth 선택**:
- Scott's rule: h = n^(-1/(d+4)) × σ
- Silverman's rule: h = (4σ^5 / 3n)^(1/5)
- Cross-validation으로 최적화

#### 1.4.4 장점
- **분포 유연성**: 가우시안 가정 불필요, 복잡한 분포 모델링 가능
- **수학적 해석**: 확률 밀도 기반으로 명확한 의미
- **신뢰구간 제공**: 통계적 유의성 검정 가능

#### 1.4.5 단점
- **Curse of Dimensionality**: 고차원에서 KDE 성능 저하
  - 차원이 높을수록 샘플 간 거리 차이 감소
  - 필요 샘플 수가 지수적으로 증가
- **메모리 사용**: 모든 (또는 많은) 학습 샘플 저장 필요
- **계산 비용**: 추론 시 모든 저장 샘플과 거리 계산
- **Bandwidth 민감성**: h 선택이 성능에 큰 영향

#### 1.4.6 성능
- MVTec AD: Image AUROC 95.5-96.8% (구현에 따라)
- 추론 속도: 50-100ms per image
- 메모리: 1-3GB (샘플 저장 방식에 따라)

---

### 1.5 Memory-Based 방식 종합 비교

#### 1.5.1 기술적 진화 과정

```
PaDiM (2020)
├─ 문제: 높은 메모리 사용량
├─ 원인: 모든 패치 위치에서 공분산 행렬 저장
└─ 성능: 96.5% AUROC

        ↓ 개선

PatchCore (2022)
├─ 해결: Coreset Selection으로 대표 패치만 저장
├─ 추가: Locally aware features
├─ 결과: 메모리 90% 감소, 성능 99.1% 향상
└─ Trade-off: Coreset 선택 시간 증가

        ↓ 변형

DFKDE (2022)
├─ 접근: KDE로 유연한 분포 모델링
├─ 장점: 비모수적 방법, 복잡한 분포 가능
└─ 한계: 고차원 curse, PatchCore보다 낮은 성능
```

#### 1.5.2 상세 비교표

| 비교 항목 | PaDiM | PatchCore | DFKDE |
|----------|-------|-----------|-------|
| **발표 연도** | 2020 | 2022 | 2022 |
| **분포 모델** | Parametric Gaussian | Non-parametric Exemplar | Non-parametric KDE |
| **저장 구조** | (μ, Σ) per location | Coreset patches | Feature samples |
| **메모리 사용** | 2-5GB | 100-500MB | 1-3GB |
| **메모리 효율** | 낮음 | 매우 높음 | 중간 |
| **거리 측정** | Mahalanobis | Euclidean (k-NN) | Kernel density |
| **계산 복잡도 (추론)** | O(d²) | O(Md) + k-NN | O(Nd) |
| **Image AUROC** | 96.5% | 99.1% | 95.5-96.8% |
| **Pixel AUROC** | 95.8% | 98.2% | 94.5-96.0% |
| **추론 속도** | 30-50ms | 50-100ms | 50-100ms |
| **분포 유연성** | 단일 가우시안만 | 임의 분포 | 임의 분포 |
| **고차원 강건성** | 중간 (공분산 추정) | 높음 | 낮음 (curse) |
| **확장성** | 낮음 (데이터↑→메모리↑) | 높음 (coreset 크기 고정) | 낮음 (샘플 필요) |
| **하이퍼파라미터** | 적음 (차원 축소 정도) | 중간 (coreset 크기, k) | 많음 (bandwidth, PCA 차원) |
| **구현 난이도** | 낮음 | 중간 | 중간 |
| **이론적 근거** | Mahalanobis 거리 | ε-cover 이론 | KDE 이론 |

#### 1.5.3 핵심 기여 및 영향

**PaDiM의 기여**:
- Memory-based 접근의 효과성 입증
- Patch 단위 분포 모델링 패러다임 확립
- 간단하면서도 강력한 baseline 제공

**PatchCore의 기여**:
- Coreset selection을 통한 메모리 문제 해결
- Memory-based 방법의 SOTA 달성
- 산업 적용 가능성 크게 향상 (메모리 효율)
- 이후 연구의 주요 baseline이 됨

**DFKDE의 기여**:
- 통계적 방법론의 적용 가능성 탐색
- 비모수적 접근의 유연성 제시
- 하지만 실용적으로는 PatchCore에 밀림

#### 1.5.4 실무 적용 가이드

**PaDiM 선택 시나리오**:
- 빠른 프로토타이핑 필요
- 구현 복잡도 최소화 원함
- 메모리 제약이 크지 않음
- 중간 수준 성능으로 충분

**PatchCore 선택 시나리오**:
- 최고 정확도 필요
- 메모리 효율 중요
- 대규모 데이터셋
- 장기 운영 예정 (안정성)

**DFKDE 선택 시나리오**:
- 통계적 해석 필요
- 비정규 분포 의심
- 연구 목적
- 실무에서는 비추천 (PatchCore 우세)

