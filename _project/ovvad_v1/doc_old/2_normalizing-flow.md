## 2. Normalizing Flow 방식 상세 분석

### 2.1 패러다임 개요

Normalizing Flow는 생성 모델(generative model)의 일종으로, 가역적인(invertible) 변환을 통해 복잡한 데이터 분포를 단순한 분포(예: 표준 정규분포)로 매핑한다. 이상 탐지에서는 정상 데이터의 분포를 학습하고, 테스트 샘플의 log-likelihood를 계산하여 이상 점수로 사용한다.

**핵심 수학적 원리**:
```
x (복잡한 데이터) ←→ z (단순한 latent, N(0, I))
              f (가역 변환)

log p(x) = log p(z) + log|det(∂f/∂x)|
         = log p(f(x)) + log|det(Jacobian)|
```

- 높은 log p(x) = 정상 (분포 내)
- 낮은 log p(x) = 이상 (분포 밖)

### 2.2 CFLOW (2021) - Conditional Flow의 도입

#### 2.2.1 핵심 원리
CFLOW는 조건부 정규화 흐름(Conditional Normalizing Flow)을 사용하여 이미지 특징의 위치별 조건부 분포를 모델링한다.

**Conditioning의 의미**:
- 각 공간적 위치 (i,j)에서 조건부 분포: p(f_{i,j} | position)
- Position embedding을 condition으로 사용
- 같은 위치는 유사한 패턴, 다른 위치는 다른 패턴 허용

#### 2.2.2 기술적 세부사항

**Multi-scale Architecture**:
```
Input Image
    ↓ Pre-trained Encoder (e.g., WideResNet)
    ├─ Layer 1 (high resolution, low semantics)
    ├─ Layer 2 (medium)
    └─ Layer 3 (low resolution, high semantics)
         ↓
    각 레이어마다 독립적인 Flow Network
         ↓
    Scale-wise log-likelihood 계산
         ↓
    Multi-scale aggregation
```

**Flow Network 구조**:
```python
class ConditionalFlowBlock:
    def __init__(self):
        self.coupling_layers = [
            AffineCouplingLayer(condition_dim),
            ActNorm(),
            InvertibleConv1x1(),
            AffineCouplingLayer(condition_dim),
            ...
        ]
    
    def forward(self, x, condition):
        log_det = 0
        for layer in self.coupling_layers:
            x, log_det_i = layer(x, condition)
            log_det += log_det_i
        return x, log_det
```

**Affine Coupling Layer**:
```
Input: x, condition c
Split: x = [x_a, x_b]
Transform: 
    s, t = NN(x_a, c)  # scale, translation
    y_a = x_a
    y_b = x_b ⊙ exp(s) + t
Output: y = [y_a, y_b]

Jacobian: 대각 행렬 → log|det| = sum(s)
```

**Position Encoding**:
- 2D positional embedding: PE(i,j) = [sin/cos 함수의 조합]
- Feature와 concatenate: [f_{i,j}, PE(i,j)]
- Flow network의 condition으로 입력

#### 2.2.3 학습 과정

```python
# 학습 (정상 데이터만 사용)
for image in train_set:
    # 1. 특징 추출
    features = encoder(image)  # Multi-scale
    
    # 2. 각 스케일에서 Flow 적용
    total_nll = 0
    for scale_features, flow_net in zip(features, flow_networks):
        # Position encoding
        positions = get_position_encoding(scale_features.shape)
        
        # Forward flow: x → z
        z, log_det = flow_net(scale_features, positions)
        
        # Negative log-likelihood
        nll = -log_prob_gaussian(z) - log_det
        total_nll += nll
    
    # 3. 최적화
    loss = total_nll
    loss.backward()
```

**추론 과정**:
```python
# 추론
def anomaly_score(test_image):
    features = encoder(test_image)
    total_nll = 0
    
    for scale_features, flow_net in zip(features, flow_networks):
        positions = get_position_encoding(scale_features.shape)
        z, log_det = flow_net(scale_features, positions)
        nll = -log_prob_gaussian(z) - log_det
        total_nll += nll
    
    return total_nll  # 높을수록 이상
```

#### 2.2.4 장점
- **확률적 해석**: Log-likelihood로 명확한 이상 점수
- **Pixel-level Localization**: 각 위치별 NLL 계산 가능
- **Multi-scale**: 다양한 크기의 이상 탐지
- **조건부 모델링**: 위치별 다른 정상 패턴 학습

#### 2.2.5 단점
- **학습 복잡도**: Flow network 학습 시간 오래 걸림
- **메모리 사용**: Multi-scale flow networks 저장
- **추론 시간**: Forward flow 계산 비용 (100-150ms)
- **하이퍼파라미터**: Flow depth, coupling layers 수 등 튜닝 필요

#### 2.2.6 성능
- MVTec AD: Image AUROC 98.2%, Pixel AUROC 97.6%
- 추론 속도: 100-150ms per image
- 메모리: 500MB-1GB

---

### 2.3 FastFlow (2021) - 속도 최적화의 혁신

#### 2.3.1 핵심 원리
FastFlow는 CFLOW의 복잡한 3D flow를 2D flow로 단순화하여 추론 속도를 대폭 향상시켰다.

**핵심 차이점**:
- CFLOW: 3D tensor (C×H×W)에 flow 적용
- FastFlow: 2D spatial locations (H×W)에 flow 적용, 채널 차원 분리

#### 2.3.2 CFLOW 대비 핵심 차이점

| 측면 | CFLOW | FastFlow | 개선 효과 |
|------|-------|----------|----------|
| **Flow 차원** | 3D (C×H×W) | 2D (H×W) | 계산량 대폭 감소 |
| **채널 처리** | 통합 처리 | 독립 처리 후 aggregation | 병렬화 가능 |
| **Coupling Layers** | 8-12개 | 4-8개 | 학습/추론 속도 향상 |
| **Position Encoding** | 복잡한 2D encoding | 간소화된 encoding | 계산 간소화 |
| **Jacobian 계산** | 고차원 | 저차원 | 빠른 log-det 계산 |

#### 2.3.3 기술적 세부사항

**2D Normalizing Flow Architecture**:
```python
class FastFlow2D:
    """
    Input: feature map (B, C, H, W)
    Process: 
    1. Reshape: (B, C, H, W) → (B*C, H*W, 1)
    2. Apply 2D flow on (H, W) for each channel independently
    3. Aggregate: channel-wise NLL summation
    """
    
    def forward(self, features):
        B, C, H, W = features.shape
        
        # Reshape for 2D flow
        x = features.permute(0, 1, 2, 3).reshape(B*C, H*W)
        
        # 2D normalizing flow
        z, log_det = self.flow_2d(x)
        
        # Calculate NLL per channel
        nll = -log_prob_gaussian(z) - log_det
        nll = nll.reshape(B, C, H, W)
        
        # Channel-wise aggregation
        total_nll = nll.sum(dim=1)  # (B, H, W)
        
        return total_nll
```

**Simplified Coupling Layer**:
```python
# CFLOW: 복잡한 3D coupling
def cflow_coupling(x):  # x: (C, H, W)
    # 전체 3D tensor 변환
    s, t = network_3d(x)  # 큰 네트워크 필요
    return x * exp(s) + t

# FastFlow: 간단한 2D coupling
def fastflow_coupling(x):  # x: (H, W)
    # 2D 공간만 변환
    s, t = network_2d(x)  # 작은 네트워크
    return x * exp(s) + t
```

**Multi-scale Integration**:
```python
def multi_scale_inference(image):
    features = encoder(image)  # [feat1, feat2, feat3]
    anomaly_maps = []
    
    for feat, flow_2d in zip(features, flows):
        # 각 스케일에서 2D flow 적용 (병렬화 가능)
        nll_map = flow_2d(feat)  # (H_i, W_i)
        
        # 원본 해상도로 upsampling
        nll_map = F.interpolate(nll_map, size=(H, W))
        anomaly_maps.append(nll_map)
    
    # Multi-scale fusion
    final_map = sum(anomaly_maps) / len(anomaly_maps)
    return final_map
```

#### 2.3.4 CFLOW 대비 개선사항

**1) 추론 속도**:
- CFLOW: 100-150ms per image
- FastFlow: 20-50ms per image
- **개선율**: 2-3배 속도 향상

**2) 메모리 효율**:
- CFLOW: 각 스케일에서 3D flow (C×H×W 차원)
- FastFlow: 각 스케일에서 2D flow (H×W 차원)
- **개선율**: 30-50% 메모리 감소

**3) 성능 유지/향상**:
- CFLOW: 98.2% AUROC
- FastFlow: 98.5% AUROC
- **결과**: 속도 향상하면서 성능도 미세 향상

**4) 학습 시간**:
- CFLOW: 2-3시간 (MVTec AD 1개 카테고리)
- FastFlow: 30-60분
- **개선율**: 3-4배 학습 속도 향상

**5) 병렬화**:
- CFLOW: 채널 간 의존성으로 병렬화 제한
- FastFlow: 채널 독립 처리로 병렬화 용이

#### 2.3.5 Trade-off 분석

**FastFlow가 포기한 것**:
- 채널 간 상관관계 모델링
- 복잡한 3D 분포 표현력

**FastFlow가 얻은 것**:
- 대폭 빠른 속도
- 간단한 구조로 학습 안정성
- 실시간 처리 가능성

**왜 성능이 유지/향상되는가?**:
- 채널 간 상관관계가 이상 탐지에 크게 중요하지 않음
- 2D 공간 구조가 더 중요
- 간단한 모델이 오히려 과적합 방지

#### 2.3.6 장점
- **빠른 속도**: 실시간 처리 가능 수준
- **높은 정확도**: CFLOW와 동등 이상
- **학습 효율**: 빠른 학습으로 빠른 iteration
- **구현 간단**: 2D flow로 복잡도 감소

#### 2.3.7 단점
- **채널 정보 손실**: 채널 간 관계 무시
- **이론적 완전성**: CFLOW보다 단순한 가정
- 여전히 flow 계산 오버헤드 존재

#### 2.3.8 성능
- MVTec AD: Image AUROC 98.5%, Pixel AUROC 97.8%
- 추론 속도: 20-50ms per image
- 메모리: 500MB-1GB

---

### 2.4 CS-Flow (2021) - Cross-Scale 정보 융합

#### 2.4.1 핵심 원리
CS-Flow는 서로 다른 스케일 간의 정보를 명시적으로 교환하는 Cross-Scale Flow를 제안한다.

**기존 방법의 한계**:
- CFLOW, FastFlow: 각 스케일 독립적으로 처리 후 합산
- 스케일 간 상호작용 없음

**CS-Flow의 해결책**:
- Cross-Scale Connection: 스케일 간 정보 흐름
- 큰 스케일 → 작은 스케일로 context 정보 전달
- 작은 스케일 → 큰 스케일로 detail 정보 전달

#### 2.4.2 CFLOW/FastFlow와의 차이점

| 측면 | CFLOW | FastFlow | CS-Flow |
|------|-------|----------|---------|
| **스케일 처리** | 독립적 | 독립적 | 상호 연결 |
| **정보 흐름** | 단방향 (feature→flow) | 단방향 | 양방향 (cross-scale) |
| **Architecture** | 3개 독립 flow | 3개 독립 flow | 통합 hierarchical flow |
| **다양한 크기 결함** | 중간 | 중간 | 우수 |

#### 2.4.3 기술적 세부사항

**Cross-Scale Flow Architecture**:
```python
class CrossScaleFlow:
    def __init__(self):
        self.flow_high = FlowBlock()    # High resolution
        self.flow_mid = FlowBlock()     # Mid resolution
        self.flow_low = FlowBlock()     # Low resolution
        
        # Cross-scale connections
        self.downsample_h2m = nn.Conv2d(...)  # High to Mid
        self.downsample_m2l = nn.Conv2d(...)  # Mid to Low
        self.upsample_l2m = nn.ConvTranspose2d(...)  # Low to Mid
        self.upsample_m2h = nn.ConvTranspose2d(...)  # Mid to High
    
    def forward(self, feat_high, feat_mid, feat_low):
        # Top-down path (context)
        z_low, log_det_low = self.flow_low(feat_low)
        
        context_m = self.upsample_l2m(z_low)
        feat_mid_aug = feat_mid + context_m
        z_mid, log_det_mid = self.flow_mid(feat_mid_aug)
        
        context_h = self.upsample_m2h(z_mid)
        feat_high_aug = feat_high + context_h
        z_high, log_det_high = self.flow_high(feat_high_aug)
        
        # Bottom-up path (details) - 역방향도 가능
        
        return [z_high, z_mid, z_low], [log_det_high, log_det_mid, log_det_low]
```

**Fully Convolutional Design**:
- 입력 이미지 크기에 무관하게 동작
- 다양한 해상도 지원
- 학습 시와 추론 시 다른 크기 가능

#### 2.4.4 CFLOW/FastFlow 대비 개선사항

**1) 다양한 크기의 결함 탐지**:
- 작은 결함: High-resolution flow가 탐지, Low-resolution context 활용
- 큰 결함: Low-resolution flow가 탐지, High-resolution detail 활용
- **개선**: 크기 robust성 향상

**2) Context-aware Detection**:
- 큰 context 정보가 작은 detail 해석에 도움
- 예: 전체적인 텍스처 패턴 내에서 국소적 이상 판단

**3) 성능**:
- 특정 카테고리(크기 다양한 결함)에서 CFLOW/FastFlow 대비 1-2%p 향상
- 평균적으로는 비슷하거나 약간 우수

#### 2.4.5 장점
- **Multi-scale robustness**: 다양한 크기 결함에 강건
- **Context integration**: 전역-지역 정보 통합
- **Flexible resolution**: 다양한 입력 크기 지원

#### 2.4.6 단점
- **복잡한 구조**: 구현 및 디버깅 어려움
- **학습 시간**: Cross-scale connection으로 학습 더 오래 걸림
- **하이퍼파라미터**: 더 많은 튜닝 필요
- **속도**: FastFlow보다 느림 (80-120ms)

#### 2.4.7 성능
- MVTec AD: Image AUROC 97.9%, Pixel AUROC 97.5%
- 특정 카테고리(Grid, Tile 등)에서 우수
- 추론 속도: 80-120ms per image

---

### 2.5 U-Flow (2022) - 자동 임계값 설정

#### 2.5.1 핵심 원리
U-Flow는 U-Net 구조를 normalizing flow에 적용하고, 비지도 방식으로 임계값을 자동 설정한다.

**기존 방법의 한계**:
- 이상 점수 계산 후 임계값을 수동으로 설정해야 함
- 데이터셋마다 다른 임계값 필요
- 운영 환경에서 재조정 필요

**U-Flow의 해결책**:
- Unsupervised threshold estimation
- 정상 데이터의 이상 점수 분포에서 자동 계산
- 운영 중 자동 적응

#### 2.5.2 기술적 세부사항

**U-shaped Flow Network**:
```python
class UFlow:
    def __init__(self):
        # Encoder (downsampling flows)
        self.enc1 = FlowBlock()  # 256x256
        self.enc2 = FlowBlock()  # 128x128
        self.enc3 = FlowBlock()  # 64x64
        self.enc4 = FlowBlock()  # 32x32
        
        # Decoder (upsampling flows)
        self.dec4 = FlowBlock()  # 32x32
        self.dec3 = FlowBlock()  # 64x64
        self.dec2 = FlowBlock()  # 128x128
        self.dec1 = FlowBlock()  # 256x256
        
        # Skip connections (U-Net style)
        self.skip_connections = True
    
    def forward(self, x):
        # Encoder
        e1, ld1 = self.enc1(x)
        e2, ld2 = self.enc2(downsample(e1))
        e3, ld3 = self.enc3(downsample(e2))
        e4, ld4 = self.enc4(downsample(e3))
        
        # Decoder with skip connections
        d4, ld5 = self.dec4(e4)
        d3, ld6 = self.dec3(upsample(d4) + e3)  # Skip
        d2, ld7 = self.dec2(upsample(d3) + e2)  # Skip
        d1, ld8 = self.dec1(upsample(d2) + e1)  # Skip
        
        log_det = sum([ld1, ld2, ld3, ld4, ld5, ld6, ld7, ld8])
        return d1, log_det
```

**Unsupervised Threshold Estimation**:
```python
def estimate_threshold(train_scores):
    """
    정상 데이터의 이상 점수 분포에서 임계값 자동 추정
    """
    # 방법 1: Percentile-based
    threshold = np.percentile(train_scores, 95)  # 상위 5%
    
    # 방법 2: Statistical (mean + k*std)
    mu = np.mean(train_scores)
    sigma = np.std(train_scores)
    threshold = mu + 3 * sigma
    
    # 방법 3: Otsu's method (bimodal distribution)
    threshold = otsu_threshold(train_scores)
    
    return threshold

# 학습 후 임계값 추정
train_scores = []
for image in train_set:
    score = model(image)
    train_scores.append(score)

threshold = estimate_threshold(train_scores)

# 추론 시 사용
test_score = model(test_image)
is_anomaly = test_score > threshold
```

**Adaptive Threshold**:
```python
class AdaptiveThreshold:
    def __init__(self, initial_threshold):
        self.threshold = initial_threshold
        self.recent_scores = deque(maxlen=100)
    
    def update(self, score, is_normal_confirmed):
        """운영 중 임계값 업데이트"""
        self.recent_scores.append(score)
        
        if is_normal_confirmed and len(self.recent_scores) > 50:
            # 정상으로 확인된 샘플들의 분포로 재추정
            normal_scores = [s for s in self.recent_scores if s < self.threshold]
            self.threshold = estimate_threshold(normal_scores)
    
    def predict(self, score):
        return score > self.threshold
```

#### 2.5.3 다른 Flow 모델 대비 개선사항

**1) 자동화**:
- 기존: 수동 임계값 설정 필요
- U-Flow: 자동 추정
- **효과**: 운영 편의성 대폭 향상

**2) 적응성**:
- 기존: 고정 임계값
- U-Flow: 적응형 임계값
- **효과**: 환경 변화에 robust

**3) U-Net 구조의 장점**:
- Skip connection으로 multi-scale 정보 효과적 융합
- Encoder-decoder로 계층적 특징 학습

#### 2.5.4 장점
- **자동 임계값**: 수동 튜닝 불필요
- **적응형**: 운영 환경 변화에 대응
- **U-Net 구조**: 효과적인 정보 융합

#### 2.5.5 단점
- **복잡한 구조**: U-Net + Flow 결합
- **학습 시간**: 더 깊은 네트워크로 느림
- **임계값 신뢰성**: 학습 데이터의 품질에 의존

#### 2.5.6 성능
- MVTec AD: Image AUROC 97.6%, Pixel AUROC 96.8%
- 추론 속도: 90-140ms per image
- 자동 임계값 성능: 수동 설정 대비 1-2%p 이내 차이

---

### 2.6 Normalizing Flow 방식 종합 비교

#### 2.6.1 기술적 진화 과정

```
CFLOW (2021)
├─ 혁신: Conditional flow로 위치별 조건부 분포 학습
├─ 특징: Multi-scale, 3D flow
├─ 문제: 느린 속도 (100-150ms)
└─ 성능: 98.2% AUROC

        ↓ 속도 최적화

FastFlow (2021)
├─ 개선: 3D→2D flow로 계산량 대폭 감소
├─ 결과: 2-3배 속도 향상 (20-50ms)
├─ 성능: 98.5% AUROC (유지/향상)
└─ Trade-off: 채널 간 상관관계 무시

        ↓ 기능 강화 (분기)

CS-Flow (2021)                U-Flow (2022)
├─ 개선: Cross-scale 정보 융합     ├─ 개선: U-Net 구조 + 자동 임계값
├─ 장점: 다양한 크기 결함         ├─ 장점: 운영 자동화
├─ 성능: 97.9% AUROC            ├─ 성능: 97.6% AUROC
└─ 단점: 복잡도 증가             └─ 단점: 학습 시간 증가
```

#### 2.6.2 상세 비교표

| 비교 항목 | CFLOW | FastFlow | CS-Flow | U-Flow |
|----------|-------|----------|---------|--------|
| **발표 연도** | 2021 | 2021 | 2021 | 2022 |
| **Flow 차원** | 3D (C×H×W) | 2D (H×W) | 2D + Cross-scale | 2D + U-Net |
| **스케일 처리** | 독립적 3개 flow | 독립적 3개 flow | 상호 연결 flow | Hierarchical U-Net |
| **채널 처리** | 통합 | 독립 | 독립 | 통합 |
| **Skip Connection** | 없음 | 없음 | Cross-scale | U-Net style |
| **Coupling Layers** | 8-12개 | 4-8개 | 6-10개 | 10-16개 |
| **임계값 설정** | 수동 | 수동 | 수동 | 자동 |
| **Image AUROC** | 98.2% | 98.5% | 97.9% | 97.6% |
| **Pixel AUROC** | 97.6% | 97.8% | 97.5% | 96.8% |
| **추론 속도** | 100-150ms | 20-50ms | 80-120ms | 90-140ms |
| **학습 시간** | 2-3시간 | 30-60분 | 3-4시간 | 4-5시간 |
| **메모리 사용** | 500MB-1GB | 500MB-1GB | 600MB-1.2GB | 700MB-1.5GB |
| **구현 난이도** | 중간 | 낮음 | 높음 | 높음 |
| **다양한 크기 결함** | 중간 | 중간 | 우수 | 중간 |
| **운영 편의성** | 낮음 | 낮음 | 낮음 | 높음 |
| **주요 혁신** | Conditional flow | 2D flow (속도) | Cross-scale | Auto threshold |

#### 2.6.3 핵심 Trade-off 분석

**1) 속도 vs 표현력**:
```
CFLOW: 느림 + 강력한 3D 표현력
  ↓
FastFlow: 빠름 + 단순한 2D 표현력
결과: 속도 3배↑, 성능 유지/향상 (채널 상관관계가 덜 중요)
```

**2) 성능 vs 복잡도**:
```
FastFlow: 간단 + 높은 성능 (98.5%)
  ↓
CS-Flow: 복잡 + 비슷한 성능 (97.9%)
결과: 복잡도 증가가 성능 향상으로 이어지지 않음
→ FastFlow가 더 실용적
```

**3) 자동화 vs 성능**:
```
FastFlow: 수동 임계값 + 98.5%
  ↓
U-Flow: 자동 임계값 + 97.6%
결과: 0.9%p 성능 희생, 운영 편의성 획득
→ 상황에 따라 선택
```

#### 2.6.4 실무 적용 가이드

**CFLOW 선택 시나리오**:
- 최고 정확도와 해석 가능성 필요
- 속도 제약 없음
- 연구 목적 또는 baseline
- **추천도**: ★★☆☆☆ (FastFlow로 대체 가능)

**FastFlow 선택 시나리오**:
- 높은 정확도 + 빠른 속도 필요
- 실시간 처리는 아니지만 응답 시간 중요
- 대부분의 실무 적용
- **추천도**: ★★★★★ (최고 추천)

**CS-Flow 선택 시나리오**:
- 크기가 매우 다양한 결함 존재
- 복잡한 multi-scale 패턴
- 특정 카테고리(Grid, Tile)에서 최고 성능 필요
- **추천도**: ★★★☆☆ (특수 상황)

**U-Flow 선택 시나리오**:
- 자동화된 운영 시스템 구축
- 임계값 수동 조정 불가능
- 환경 변화가 잦은 상황
- **추천도**: ★★★☆☆ (자동화 필요 시)

#### 2.6.5 성능 벤치마크 요약

**정확도 순위**:
1. FastFlow (98.5%)
2. CFLOW (98.2%)
3. CS-Flow (97.9%)
4. U-Flow (97.6%)

**속도 순위**:
1. FastFlow (20-50ms) ⭐
2. CS-Flow (80-120ms)
3. U-Flow (90-140ms)
4. CFLOW (100-150ms)

**실용성 순위**:
1. FastFlow (속도+성능+간단함) ⭐
2. U-Flow (자동화)
3. CFLOW (baseline)
4. CS-Flow (특수 케이스)

**연구 가치**:
1. CFLOW (패러다임 제시) ⭐
2. FastFlow (효율성 혁신) ⭐
3. CS-Flow (multi-scale fusion)
4. U-Flow (자동화)

