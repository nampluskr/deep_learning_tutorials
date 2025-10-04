## 5. Feature Adaptation 방식 상세 분석

### 5.1 패러다임 개요

Feature Adaptation 방식은 Pre-trained 모델의 특징을 타겟 도메인(검사 대상)에 맞게 적응(adaptation)시켜 이상 탐지 성능을 향상시킨다. 핵심 문제는 **Domain Shift**이다.

**Domain Shift 문제**:
```
Source Domain (ImageNet 등):
- 일반 객체 (고양이, 자동차 등)
- 자연광, 다양한 배경
- RGB 컬러 이미지

Target Domain (산업 검사):
- 특정 제품 (PCB, 금속 표면 등)
- 인공 조명, 단순 배경
- 특수한 텍스처/패턴

문제: Source에서 학습한 특징이 Target에 최적이 아님
```

**Feature Adaptation의 목적**:
- Source domain의 일반 지식 유지
- Target domain의 특수성 학습
- Domain gap 최소화

### 5.2 CFA (2022) - Coupled Hypersphere Feature Adaptation

#### 5.2.1 핵심 원리
CFA(Coupled-hypersphere-based Feature Adaptation)는 Hypersphere embedding을 사용하여 pre-trained 특징을 타겟 도메인의 정상 분포에 적응시킨다.

**Hypersphere Embedding**:
```
일반 Euclidean space: 무한대로 확장 가능
Hypersphere: 단위 구(sphere) 표면에 특징 projection

장점:
1. Bounded space: 특징이 구 표면에 제한됨
2. Angular distance: 방향성 중시
3. Compact representation: 정상 패턴이 밀집된 cluster 형성
```

**Coupled Hypersphere**:
```
Source Hypersphere: ImageNet 특징의 구
Target Hypersphere: 타겟 도메인 특징의 구

Coupling: 두 구 사이의 매핑 학습
- Source의 일반 지식 활용
- Target의 특수성 반영
```

#### 5.2.2 기술적 세부사항

**Architecture**:
```python
class CFA(nn.Module):
    def __init__(self, backbone='resnet18', num_scales=3):
        super().__init__()
        
        # Pre-trained backbone (frozen initially)
        self.backbone = resnet18(pretrained=True)
        
        # Multi-scale feature extractors
        self.feature_extractors = nn.ModuleList([
            nn.Identity(),  # layer1
            nn.Identity(),  # layer2
            nn.Identity()   # layer3
        ])
        
        # Adaptation layers (learnable)
        self.adaptation_layers = nn.ModuleList([
            FeatureAdaptation(in_dim=64, out_dim=64),
            FeatureAdaptation(in_dim=128, out_dim=128),
            FeatureAdaptation(in_dim=256, out_dim=256)
        ])
        
        # Hypersphere projection heads
        self.projection_heads = nn.ModuleList([
            ProjectionHead(64, hypersphere_dim=128),
            ProjectionHead(128, hypersphere_dim=256),
            ProjectionHead(256, hypersphere_dim=512)
        ])
```

**Feature Adaptation Layer**:
```python
class FeatureAdaptation(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        # Adaptation network
        self.adapt = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        
        # Residual connection
        self.residual = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        adapted = self.adapt(x)
        residual = self.residual(x)
        return adapted + residual
```

**Hypersphere Projection**:
```python
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hypersphere_dim):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, hypersphere_dim, 1),
            nn.BatchNorm2d(hypersphere_dim),
            nn.ReLU(),
            nn.Conv2d(hypersphere_dim, hypersphere_dim, 1)
        )
    
    def forward(self, x):
        # Project to hypersphere
        projected = self.projection(x)
        
        # L2 normalization (unit hypersphere)
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized
```

**학습 과정 (Two-stage)**:
```python
def train_cfa(train_loader):
    # Stage 1: Feature adaptation
    print("Stage 1: Adapting features to target domain")
    for epoch in range(adaptation_epochs):
        for images in train_loader:
            # Extract pre-trained features
            with torch.no_grad():
                source_features = backbone.extract_features(images)
            
            # Adapt features
            adapted_features = []
            for feat, adapter in zip(source_features, adaptation_layers):
                adapted = adapter(feat)
                adapted_features.append(adapted)
            
            # Self-supervised adaptation loss
            loss = self_supervised_loss(adapted_features, images)
            
            optimizer_adapt.zero_grad()
            loss.backward()
            optimizer_adapt.step()
    
    # Stage 2: Hypersphere embedding
    print("Stage 2: Learning hypersphere embeddings")
    for epoch in range(embedding_epochs):
        for images in train_loader:
            # Extract adapted features
            adapted_features = [adapter(feat) 
                              for feat, adapter in zip(backbone.extract_features(images), 
                                                      adaptation_layers)]
            
            # Project to hypersphere
            embeddings = [proj(feat) 
                         for feat, proj in zip(adapted_features, projection_heads)]
            
            # Contrastive loss on hypersphere
            loss = hypersphere_contrastive_loss(embeddings)
            
            optimizer_proj.zero_grad()
            loss.backward()
            optimizer_proj.step()
```

**Self-supervised Adaptation Loss**:
```python
def self_supervised_loss(adapted_features, images):
    """
    타겟 도메인의 구조적 정보 학습
    """
    # Augmentation: Random crop, rotate
    aug1 = augment(images)
    aug2 = augment(images)
    
    feat1 = [adapter(backbone.extract(aug1)) for adapter in adaptation_layers]
    feat2 = [adapter(backbone.extract(aug2)) for adapter in adaptation_layers]
    
    # Contrastive loss: 같은 이미지의 augmentation은 유사해야 함
    loss = 0
    for f1, f2 in zip(feat1, feat2):
        loss += F.cosine_embedding_loss(f1.flatten(1), f2.flatten(1), 
                                        torch.ones(f1.size(0)))
    
    return loss / len(feat1)
```

**Hypersphere Contrastive Loss**:
```python
def hypersphere_contrastive_loss(embeddings, temperature=0.1):
    """
    Hypersphere 상에서 정상 패턴들을 밀집시킴
    """
    loss = 0
    
    for emb in embeddings:  # Multi-scale
        B, C, H, W = emb.shape
        
        # Flatten spatial dimensions
        emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # Cosine similarity matrix (angular distance on sphere)
        sim_matrix = torch.mm(emb_flat, emb_flat.t()) / temperature
        
        # Positive pairs: 같은 이미지의 인접 패치
        positive_mask = create_spatial_positive_mask(B, H, W)
        
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)
        
        loss += -torch.log(pos_sim / all_sim).mean()
    
    return loss / len(embeddings)
```

**추론 과정**:
```python
def inference(test_image):
    with torch.no_grad():
        # Extract and adapt features
        source_features = backbone.extract_features(test_image)
        adapted_features = [adapter(feat) 
                           for feat, adapter in zip(source_features, adaptation_layers)]
        
        # Project to hypersphere
        embeddings = [proj(feat) 
                     for feat, proj in zip(adapted_features, projection_heads)]
        
        # Calculate anomaly scores
        anomaly_maps = []
        
        for emb in embeddings:
            # Distance to normal cluster on hypersphere
            # (정상 샘플들의 centroid와의 angular distance)
            centroid = normal_centroids[scale]  # Pre-computed during training
            
            # Cosine distance (1 - cosine similarity)
            distance = 1 - F.cosine_similarity(emb, centroid.unsqueeze(0), dim=1)
            
            # Upsample to original resolution
            distance = F.interpolate(distance.unsqueeze(1), size=(H, W), mode='bilinear')
            anomaly_maps.append(distance)
        
        # Multi-scale fusion
        final_map = torch.stack(anomaly_maps).mean(dim=0).squeeze()
        
        # Image score
        image_score = final_map.max()
    
    return image_score, final_map
```

**Normal Centroid Computation**:
```python
def compute_normal_centroids(train_loader):
    """
    학습 후 정상 샘플들의 hypersphere centroid 계산
    """
    centroids = []
    
    for scale in range(num_scales):
        embeddings_list = []
        
        for images in train_loader:
            with torch.no_grad():
                # Extract embeddings at this scale
                emb = model.forward_scale(images, scale)
                embeddings_list.append(emb)
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)  # (N, C, H, W)
        
        # Compute centroid (mean on hypersphere)
        centroid = all_embeddings.mean(dim=[0, 2, 3])  # (C,)
        centroid = F.normalize(centroid, p=2, dim=0)  # Project back to sphere
        
        centroids.append(centroid)
    
    return centroids
```

#### 5.2.3 장점
- **Domain shift 해결**: 타겟 도메인에 특화
- **Pre-trained 활용**: ImageNet 지식 전이
- **Hypersphere**: Compact하고 discriminative한 표현
- **Multi-scale**: 다양한 크기 이상 탐지
- **Unsupervised**: 정상 데이터만 필요

#### 5.2.4 단점
- **Two-stage 학습**: Adaptation + Embedding (복잡)
- **학습 시간**: 각 stage마다 수 시간
- **하이퍼파라미터**: Temperature, hypersphere dim 등
- **성능**: SOTA 대비 낮음 (96.5-97.5%)

#### 5.2.5 성능
- MVTec AD: Image AUROC 96.5-97.5%
- 추론 속도: 40-70ms per image
- 메모리: 500MB-1GB
- 학습 시간: 5-8시간 (two-stage)

---

### 5.3 DFM (2019) - Deep Feature Modeling

#### 5.3.1 핵심 원리
DFM(Deep Feature Modeling)은 딥러닝 특징에 PCA(Principal Component Analysis)를 적용하여 정상 데이터의 주요 변동 방향을 학습한다. 가장 간단한 feature adaptation 방법이다.

**PCA 기반 Adaptation**:
```
1. Pre-trained CNN에서 특징 추출
2. PCA로 주성분 분석
3. 정상 변동: 주요 주성분 방향
4. 이상: 주성분으로 설명 안되는 변동
```

#### 5.3.2 CFA 대비 핵심 차이점

| 측면 | DFM | CFA | 개선 효과 (CFA) |
|------|-----|-----|----------------|
| **Adaptation 방법** | PCA (선형) | Neural network (비선형) | 복잡한 패턴 학습 |
| **특징 공간** | Euclidean | Hypersphere | Bounded, angular |
| **학습 방식** | 단일 PCA | Two-stage (adapt+embed) | 더 정교한 학습 |
| **계산 복잡도** | 매우 낮음 | 높음 | Trade-off |
| **Image AUROC** | 94.5-95.5% | 96.5-97.5% | +2%p |
| **구현 난이도** | 매우 낮음 | 높음 | - |

#### 5.3.3 기술적 세부사항

**Complete Pipeline**:
```python
class DFM:
    def __init__(self, backbone='resnet18', n_components=100):
        self.backbone = resnet18(pretrained=True)
        self.backbone.eval()
        
        self.n_components = n_components
        self.pca = None
        self.mean = None
        self.components = None
    
    def extract_features(self, images):
        """Pre-trained CNN features"""
        with torch.no_grad():
            # Extract from multiple layers
            features = []
            
            x = self.backbone.conv1(images)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            features.append(adaptive_avg_pool2d(x, (1, 1)))  # Global pool
            
            x = self.backbone.layer2(x)
            features.append(adaptive_avg_pool2d(x, (1, 1)))
            
            x = self.backbone.layer3(x)
            features.append(adaptive_avg_pool2d(x, (1, 1)))
            
            # Concatenate multi-scale features
            features = torch.cat([f.squeeze() for f in features], dim=1)
        
        return features  # (B, 64+128+256 = 448)
    
    def fit(self, train_loader):
        """PCA fitting on normal data"""
        print("Extracting features from training data...")
        all_features = []
        
        for images in train_loader:
            features = self.extract_features(images)
            all_features.append(features.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)  # (N, 448)
        
        print(f"Fitting PCA with {self.n_components} components...")
        self.mean = np.mean(all_features, axis=0)
        
        # Center data
        centered = all_features - self.mean
        
        # Compute covariance matrix
        cov = np.cov(centered, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_components
        self.components = eigenvectors[:, :self.n_components]
        self.eigenvalues = eigenvalues[:self.n_components]
        
        print(f"Explained variance: {eigenvalues[:self.n_components].sum() / eigenvalues.sum():.2%}")
    
    def predict(self, test_image):
        """Anomaly detection"""
        # Extract features
        features = self.extract_features(test_image.unsqueeze(0))
        features = features.cpu().numpy().squeeze()
        
        # Center
        centered = features - self.mean
        
        # Project to principal subspace
        projected = np.dot(centered, self.components)
        
        # Reconstruct
        reconstructed = np.dot(projected, self.components.T)
        
        # Reconstruction error (distance to subspace)
        error = np.linalg.norm(centered - reconstructed)
        
        # Mahalanobis distance (in subspace)
        mahalanobis = np.sqrt(np.sum((projected ** 2) / (self.eigenvalues + 1e-6)))
        
        # Combined score
        anomaly_score = error + mahalanobis
        
        return anomaly_score
```

**Visualization of PCA Subspace**:

```python
def visualize_pca_subspace(dfm, normal_samples, anomalous_samples):
    """
    PCA 부분공간 시각화 (2D projection)
    """
    # Extract features
    normal_features = [dfm.extract_features(img) for img in normal_samples]
    anomaly_features = [dfm.extract_features(img) for img in anomalous_samples]
    
    normal_features = torch.cat(normal_features, dim=0).cpu().numpy()
    anomaly_features = torch.cat(anomaly_features, dim=0).cpu().numpy()
    
    # Center
    normal_centered = normal_features - dfm.mean
    anomaly_centered = anomaly_features - dfm.mean
    
    # Project to first 2 principal components
    normal_proj = np.dot(normal_centered, dfm.components[:, :2])
    anomaly_proj = np.dot(anomaly_centered, dfm.components[:, :2])
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(normal_proj[:, 0], normal_proj[:, 1], 
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(anomaly_proj[:, 0], anomaly_proj[:, 1], 
                c='red', label='Anomaly', alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA Subspace Visualization')
    plt.show()
```

**Why PCA Works for Anomaly Detection**:
```python
def explain_pca_anomaly_detection():
    """
    PCA가 이상 탐지에 효과적인 이유
    """
    explanation = """
    1. 정상 데이터의 변동 (Variance):
       - 주요 주성분: 정상적인 변화 (조명, 각도 등)
       - 높은 eigenvalue: 큰 변동성
    
    2. 이상 데이터의 특성:
       - 주성분으로 설명 안됨
       - Reconstruction error 큼
       - Subspace에서 벗어남
    
    3. 수학적 원리:
       정상: x ≈ μ + Σ(a_i * v_i)  (주성분의 선형결합)
       이상: x ≠ μ + Σ(a_i * v_i)  (residual 큼)
       
       Anomaly Score = ||x - x_reconstructed||
    """
    return explanation
```

#### 5.3.4 CFA 대비 개선사항 (CFA의 관점)

**1) 선형 vs 비선형**:
```
DFM: PCA (선형 변환)
- 정상 manifold가 선형이라 가정
- 복잡한 비선형 패턴 학습 불가

CFA: Neural network adaptation (비선형)
- 복잡한 비선형 관계 학습
- 더 풍부한 표현력
```

**2) Euclidean vs Hypersphere**:
```
DFM: Euclidean space
- 무한대로 확장 가능
- 거리 기반 (L2 distance)

CFA: Hypersphere
- Bounded space (단위 구)
- Angular distance (cosine similarity)
- 더 compact한 정상 cluster
```

**3) 성능 향상**:
```
DFM: 94.5-95.5% AUROC
CFA: 96.5-97.5% AUROC
개선율: +2%p
```

**4) Adaptation 품질**:
```
DFM:
- PCA는 전역적(global) 변환
- 타겟 도메인 특성 제한적 반영

CFA:
- Self-supervised learning으로 타겟 특성 학습
- 두 단계 adaptation으로 더 정교
```

#### 5.3.5 장점
- **극도로 간단**: PCA만으로 구현
- **빠른 학습**: 수 분 내 완료
- **해석 가능**: 주성분의 의미 명확
- **낮은 메모리**: PCA matrix만 저장
- **빠른 추론**: 행렬 곱셈만

#### 5.3.6 단점
- **낮은 성능**: 최신 모델 대비 3-5%p 낮음
- **선형 가정**: 복잡한 비선형 패턴 학습 불가
- **고차원 문제**: 특징 차원이 높으면 PCA 효과 감소
- **적응 제한**: 타겟 도메인 특성 제한적 반영

#### 5.3.7 성능
- MVTec AD: Image AUROC 94.5-95.5%
- 추론 속도: 10-20ms per image (매우 빠름)
- 메모리: <100MB (매우 낮음)
- 학습 시간: 5-15분 (매우 빠름)

---

### 5.4 Feature Adaptation 방식 종합 비교

#### 5.4.1 기술적 진화 과정

```
DFM (2019)
├─ 시작: PCA 기반 간단한 adaptation
├─ 방식: 선형 변환, Euclidean space
├─ 성능: 94.5-95.5% AUROC
└─ 한계: 선형 가정, 낮은 성능

        ↓ 비선형 + Bounded space

CFA (2022)
├─ 혁신: Neural adaptation + Hypersphere
├─ 방식: Two-stage (adapt + embed)
├─ 성능: 96.5-97.5% AUROC (+2%p)
├─ 개선: 비선형 학습, angular distance
└─ 한계: 복잡한 학습, 여전히 SOTA 대비 낮음
```

#### 5.4.2 상세 비교표

| 비교 항목 | DFM | CFA |
|----------|-----|-----|
| **발표 연도** | 2019 | 2022 |
| **Adaptation 방법** | PCA (선형 변환) | Neural network (비선형) |
| **특징 공간** | Euclidean | Hypersphere (unit sphere) |
| **학습 방식** | 단일 단계 (PCA fitting) | Two-stage (adaptation + embedding) |
| **학습 목표** | Variance maximization | Self-supervised + Contrastive |
| **거리 측정** | L2 + Mahalanobis | Cosine distance (angular) |
| **Multi-scale** | 간단한 concatenation | 독립적인 scale-wise adaptation |
| **Image AUROC** | 94.5-95.5% | 96.5-97.5% |
| **Pixel AUROC** | 92.5-94.0% | 95.0-96.5% |
| **추론 속도** | 10-20ms (매우 빠름) | 40-70ms |
| **학습 시간** | 5-15분 (매우 빠름) | 5-8시간 |
| **메모리 사용** | <100MB (매우 낮음) | 500MB-1GB |
| **구현 난이도** | 매우 낮음 (PCA) | 높음 (Two-stage training) |
| **하이퍼파라미터** | n_components | Temperature, hypersphere_dim, learning rates |
| **Domain shift 해결** | 제한적 (선형) | 우수 (비선형) |
| **해석 가능성** | 높음 (주성분 의미) | 중간 (embedding space) |
| **확장성** | 높음 (간단) | 중간 (복잡한 학습) |
| **주요 혁신** | PCA를 이상 탐지에 적용 | Hypersphere embedding |
| **적합 환경** | 빠른 프로토타입, 간단한 도메인 | Domain shift가 큰 환경 |
| **추천도** | ★★☆☆☆ (baseline) | ★★★☆☆ (특수 케이스) |

#### 5.4.3 핵심 Trade-off 분석

**성능 vs 복잡도**:
```
DFM: 
- 간단함: PCA만
- 빠름: 15분 학습, 10ms 추론
- 성능: 94.5% (-5.5%p from SOTA)

CFA:
- 복잡함: Two-stage neural training
- 느림: 8시간 학습, 70ms 추론
- 성능: 96.5% (-3.5%p from SOTA)

결과: 복잡도 증가로 2%p 개선
      하지만 여전히 PatchCore(99.1%) 대비 낮음
```

**선형 vs 비선형**:
```
DFM (선형):
+ 수학적으로 명확
+ 해석 가능
- 복잡한 패턴 학습 불가

CFA (비선형):
+ 복잡한 관계 학습
+ 더 나은 adaptation
- Black box
```

**Euclidean vs Hypersphere**:
```
Euclidean (DFM):
- Unbounded space
- L2 distance
- 정상 cluster가 흩어질 수 있음

Hypersphere (CFA):
- Bounded space (unit sphere)
- Angular distance
- 정상 cluster가 compact
```

#### 5.4.4 실무 적용 가이드

**DFM 선택 시나리오**:
- 빠른 baseline 필요
- 극도로 제한된 리소스 (시간/계산)
- 간단한 해석 필요
- Domain shift가 크지 않음
- **추천도**: ★★☆☆☆

**사용 예시**:
```python
# DFM: 15분이면 학습 완료
dfm = DFM(n_components=100)
dfm.fit(train_loader)  # 15분
score = dfm.predict(test_image)  # 10ms

# 빠른 iteration 가능
for n in [50, 100, 200]:
    dfm = DFM(n_components=n)
    dfm.fit(train_loader)
    evaluate(dfm)
```

**CFA 선택 시나리오**:
- Domain shift가 큰 환경
  - ImageNet → 산업 이미지
  - RGB → Grayscale/특수 스펙트럼
  - 자연광 → 인공 조명
- Pre-trained 모델의 한계 느낄 때
- 충분한 학습 시간/리소스
- **추천도**: ★★★☆☆

**사용 예시**:
```python
# CFA: Domain shift 해결
# 예: ImageNet(자연 이미지) → PCB 검사
cfa = CFA(backbone='resnet18')

# Stage 1: Feature adaptation (4시간)
cfa.adapt_to_target(train_loader)

# Stage 2: Hypersphere embedding (4시간)
cfa.learn_hypersphere(train_loader)

# 추론
score = cfa.predict(test_image)  # 70ms
```

#### 5.4.5 다른 패러다임과의 비교

**Feature Adaptation vs Memory-Based**:
```
Memory-Based (PatchCore):
- Pre-trained 특징을 그대로 사용
- Coreset으로 메모리 관리
- 99.1% AUROC

Feature Adaptation (CFA):
- Pre-trained 특징을 adaptation
- Domain shift 해결 시도
- 96.5% AUROC

결과: Adaptation이 항상 도움되는 것은 아님
      PatchCore가 더 효과적
      (ImageNet 특징이 이미 충분히 일반적)
```

**Feature Adaptation vs Knowledge Distillation**:
```
Knowledge Distillation (Reverse Distillation):
- Teacher-student로 타겟 도메인 학습
- One-class embedding
- 98.6% AUROC

Feature Adaptation (CFA):
- Pre-trained 특징 adaptation
- Hypersphere embedding
- 96.5% AUROC

결과: KD가 더 효과적
      End-to-end 학습이 adaptation보다 강력
```

#### 5.4.6 Feature Adaptation의 한계

**1) 성능 한계**:
```
최고 성능 (CFA): 96.5-97.5%
SOTA (PatchCore): 99.1%
차이: 1.6-2.6%p

원인:
- Pre-trained 특징의 근본적 한계
- Adaptation만으로 극복 어려움
```

**2) 복잡도 대비 효과**:
```
DFM → CFA: 2%p 성능 향상
학습 시간: 15분 → 8시간 (32배)
추론 시간: 10ms → 70ms (7배)

ROI: 매우 낮음
```

**3) 대체 방법의 우세**:
```
PatchCore (Memory-based): 99.1%, 간단
Reverse Distillation (KD): 98.6%, 효과적
FastFlow (Flow): 98.5%, 빠름

Feature Adaptation: 96.5%, 복잡

결과: 실무에서 선택 이유가 적음
```

#### 5.4.7 Feature Adaptation의 의의

**연구적 가치**:
- Domain adaptation 연구의 일환
- Hypersphere embedding 탐색
- Pre-trained 모델의 한계 파악

**실무적 가치**:
- 제한적 (다른 방법이 더 효과적)
- DFM: 빠른 baseline으로만 유용
- CFA: 특수한 domain shift 상황에서만

**역사적 의의**:
- DFM (2019): 초기 feature-based 접근
- CFA (2022): 발전했지만 패러다임 한계 드러남
- 이후 연구는 다른 방향 (KD, Flow 등)으로 이동

