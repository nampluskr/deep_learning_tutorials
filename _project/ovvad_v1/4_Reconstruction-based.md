# Anomalib 핵심 패러다임 상세 분석 보고서 (계속)

## 4. Reconstruction-Based 방식 상세 분석

### 4.1 패러다임 개요

Reconstruction-based 방식은 정상 데이터로 학습된 재구성 모델(Auto-encoder, GAN 등)이 정상 샘플은 잘 재구성하지만 이상 샘플은 제대로 재구성하지 못하는 원리를 이용한다. 재구성 오류(reconstruction error)의 크기를 이상 점수로 사용한다.

**핵심 가정**:
```
정상 데이터 학습: Encoder → Latent (압축) → Decoder → Reconstruction
정상 샘플: ||Input - Reconstruction|| ≈ 0
이상 샘플: ||Input - Reconstruction|| >> 0 (학습하지 못한 패턴)

Anomaly Score = Reconstruction Error
```

**기본 원리**:
- 정상 데이터만으로 학습된 모델은 정상 패턴의 manifold를 학습
- 이상 데이터는 이 manifold 밖에 위치
- Manifold로 projection(재구성) 시 큰 오류 발생

### 4.2 GANomaly (2018) - GAN 기반 초기 접근

#### 4.2.1 핵심 원리
GANomaly는 Generative Adversarial Network를 이용한 초기 이상 탐지 모델로, Encoder-Decoder-Encoder (E-D-E) 구조를 사용한다.

**독특한 E-D-E 구조**:
- 일반 Auto-encoder: E-D (단일 encoder)
- GANomaly: E-D-E (이중 encoder)
- 첫 번째 encoder와 두 번째 encoder의 latent code 차이를 이상 점수로 사용

**왜 이중 Encoder인가?**:
```
Input x → E1 → z1 → D → x' → E2 → z2

정상: x ≈ x' → z1 ≈ z2 (재구성 성공)
이상: x ≠ x' → z1 ≠ z2 (재구성 실패)

Anomaly Score = ||z1 - z2||
```

#### 4.2.2 기술적 세부사항

**Network Architecture**:
```python
class GANomaly(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        
        # Generator: E-D-E 구조
        self.encoder1 = nn.Sequential(
            # Input: (3, 256, 256)
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # Output: (256, 32, 32)
        )
        
        # Bottleneck: Feature → Latent
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(256, latent_dim, 32),  # (100, 1, 1)
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
            # Output: (3, 256, 256)
        )
        
        # Second Encoder
        self.encoder2 = nn.Sequential(
            # 동일한 구조
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # ... (encoder1과 동일)
        )
        
        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(256, latent_dim, 32),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 32),  # Real/Fake score
            nn.Sigmoid()
        )
```

**학습 과정 (Adversarial Training)**:
```python
def train_step(real_images):
    batch_size = real_images.size(0)
    
    # ===== Generator (E-D-E) Training =====
    # Forward pass
    feat1 = encoder1(real_images)
    z1 = bottleneck1(feat1)  # First latent code
    
    fake_images = decoder(z1)
    
    feat2 = encoder2(fake_images)
    z2 = bottleneck2(feat2)  # Second latent code
    
    # Loss 1: Adversarial loss (fool discriminator)
    pred_fake = discriminator(fake_images)
    loss_adv = criterion_bce(pred_fake, torch.ones(batch_size, 1))
    
    # Loss 2: Contextual loss (재구성 품질)
    loss_context = criterion_l1(fake_images, real_images)
    
    # Loss 3: Encoder loss (latent code consistency)
    loss_encoder = criterion_l2(z1, z2)
    
    # Total generator loss
    loss_g = w_adv * loss_adv + w_con * loss_context + w_enc * loss_encoder
    
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()
    
    # ===== Discriminator Training =====
    # Real images
    pred_real = discriminator(real_images)
    loss_d_real = criterion_bce(pred_real, torch.ones(batch_size, 1))
    
    # Fake images
    pred_fake = discriminator(fake_images.detach())
    loss_d_fake = criterion_bce(pred_fake, torch.zeros(batch_size, 1))
    
    # Total discriminator loss
    loss_d = (loss_d_real + loss_d_fake) / 2
    
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
    
    return loss_g, loss_d
```

**추론 과정**:
```python
def inference(test_image):
    with torch.no_grad():
        # First encoding
        feat1 = encoder1(test_image)
        z1 = bottleneck1(feat1)
        
        # Reconstruction
        fake_image = decoder(z1)
        
        # Second encoding
        feat2 = encoder2(fake_image)
        z2 = bottleneck2(feat2)
        
        # Anomaly score 1: Latent space distance
        score_latent = torch.norm(z1 - z2, p=2, dim=1).mean()
        
        # Anomaly score 2: Reconstruction error
        score_recon = torch.norm(test_image - fake_image, p=1, dim=1).mean()
        
        # Combined score
        anomaly_score = score_latent + 0.1 * score_recon
        
        # Pixel-level anomaly map
        anomaly_map = torch.norm(test_image - fake_image, p=1, dim=1)
    
    return anomaly_score, anomaly_map
```

**Loss Function 상세**:
```python
# 1. Adversarial Loss (GAN objective)
L_adv = E[log D(x)] + E[log(1 - D(G(x)))]

# 2. Contextual Loss (L1 reconstruction)
L_con = ||x - G(x)||_1

# 3. Encoder Loss (latent consistency)
L_enc = ||E1(x) - E2(G(x))||_2

# Total Generator Loss
L_G = λ_adv * L_adv + λ_con * L_con + λ_enc * L_enc
```

#### 4.2.3 장점
- **Semi-supervised 가능**: 소량의 이상 샘플 활용 가능
- **생성 모델**: Realistic한 재구성 가능
- **이중 검증**: Latent + Reconstruction 두 가지 신호
- **초기 연구**: GAN 기반 이상 탐지의 선구자

#### 4.2.4 단점
- **학습 불안정**: GAN 특유의 mode collapse, oscillation
- **하이퍼파라미터 민감**: λ_adv, λ_con, λ_enc 튜닝 어려움
- **느린 수렴**: Adversarial training으로 학습 시간 오래 걸림
- **낮은 성능**: 최신 모델 대비 정확도 낮음 (93-95%)
- **복잡한 구조**: E-D-E + Discriminator

#### 4.2.5 성능
- MVTec AD: Image AUROC 93-95% (구현에 따라)
- 추론 속도: 50-80ms per image
- 메모리: 500MB-1GB
- 학습 시간: 6-10시간 (불안정한 수렴)

---

### 4.3 DRAEM (2021) - Simulated Anomaly의 혁신

#### 4.3.1 핵심 원리
DRAEM(Discriminatively trained Reconstruction Embedding)은 reconstruction 패러다임을 혁신적으로 변화시켰다. 정상 이미지에 인위적으로 이상 패턴을 추가(simulate)하여 학습 데이터를 생성하고, 이를 제거하도록 학습한다.

**패러다임 전환**:
```
기존 (GANomaly):
- 정상 데이터만 사용
- Unsupervised learning
- 이상 샘플을 본 적 없음

DRAEM:
- 정상 + Simulated anomaly 사용
- Supervised learning (discriminative)
- 이상 패턴을 명시적으로 학습
```

**핵심 아이디어**:
```
1. 정상 이미지에서 패치를 잘라내기
2. 다른 이미지/텍스처의 패치로 붙이기
3. → Simulated anomalous image
4. 이를 원본으로 복원하도록 학습
```

#### 4.3.2 GANomaly 대비 핵심 차이점

| 측면 | GANomaly | DRAEM | 개선 효과 |
|------|----------|-------|----------|
| **학습 방식** | Unsupervised (정상만) | Supervised (정상+시뮬레이션) | 명확한 학습 신호 |
| **이상 샘플** | 없음 | Simulated anomaly | 이상 패턴 학습 |
| **네트워크 구조** | E-D-E + Discriminator | Reconstructive + Discriminative | 간단하면서 효과적 |
| **학습 안정성** | 불안정 (GAN) | 안정 (Supervised) | 빠른 수렴 |
| **Loss 함수** | Adversarial + L1 + L2 | SSIM + Focal + L2 | 더 강건한 학습 |
| **Image AUROC** | 93-95% | 97.5% | +2.5-4.5%p |
| **학습 시간** | 6-10시간 | 2-4시간 | 2-3배 단축 |
| **Few-shot** | 어려움 | 가능 | 적은 데이터로 학습 |

#### 4.3.3 기술적 세부사항

**Anomaly Simulation Process**:
```python
class AnomalySimulator:
    def __init__(self):
        # 다양한 텍스처 소스 (DTD, Places 등)
        self.texture_source = load_texture_dataset()
    
    def simulate_anomaly(self, normal_image):
        """정상 이미지에 가짜 결함 추가"""
        H, W = normal_image.shape[1:]
        
        # 1. Anomaly mask 생성
        mask = self.generate_irregular_mask(H, W)
        
        # 2. 소스 이미지에서 텍스처 추출
        source = random.choice(self.texture_source)
        source_patch = self.extract_patch(source, mask.shape)
        
        # 3. Perlin noise로 자연스러운 패턴
        perlin = generate_perlin_noise(H, W, scale=random.uniform(0.1, 1.0))
        
        # 4. 결함 합성
        anomalous_image = normal_image.clone()
        anomalous_image = (1 - mask) * normal_image + mask * source_patch
        
        # 5. Blending for smoothness
        anomalous_image = gaussian_blur(anomalous_image, kernel_size=5)
        
        return anomalous_image, mask
    
    def generate_irregular_mask(self, H, W):
        """불규칙한 결함 영역 생성"""
        # 방법 1: Random polygons
        num_vertices = random.randint(4, 12)
        vertices = [(random.randint(0, W), random.randint(0, H)) 
                   for _ in range(num_vertices)]
        mask = draw_polygon(vertices, (H, W))
        
        # 방법 2: Random brush strokes
        mask = draw_random_strokes(H, W, num_strokes=random.randint(1, 5))
        
        # 방법 3: Perlin noise threshold
        perlin = generate_perlin_noise(H, W)
        mask = (perlin > random.uniform(0.3, 0.7)).float()
        
        return mask
```

**Network Architecture**:
```python
class DRAEM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Reconstructive Subnetwork
        self.reconstructive = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Decoder
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
        
        # Discriminative Subnetwork (Segmentation)
        self.discriminative = nn.Sequential(
            # Input: Reconstructed image (3 channels)
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            
            # Decoder (U-Net style with skip connections)
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # Binary segmentation
            nn.Sigmoid()
        )
```

**학습 과정**:
```python
def train_step(normal_images):
    # 1. Anomaly simulation
    anomalous_images, gt_masks = [], []
    for img in normal_images:
        aug_img, mask = anomaly_simulator.simulate_anomaly(img)
        anomalous_images.append(aug_img)
        gt_masks.append(mask)
    
    anomalous_images = torch.stack(anomalous_images)
    gt_masks = torch.stack(gt_masks)
    
    # 2. Reconstructive network: 이상 제거
    reconstructed = reconstructive_net(anomalous_images)
    
    # Loss 1: SSIM (Structural Similarity)
    loss_ssim = 1 - ssim(reconstructed, normal_images)
    
    # Loss 2: L2 reconstruction
    loss_l2 = F.mse_loss(reconstructed, normal_images)
    
    # Total reconstruction loss
    loss_recon = loss_ssim + 0.1 * loss_l2
    
    # 3. Discriminative network: 이상 영역 segmentation
    anomaly_map = discriminative_net(reconstructed)
    
    # Loss 3: Focal loss (class imbalance 해결)
    loss_focal = focal_loss(anomaly_map, gt_masks)
    
    # Total loss
    total_loss = loss_recon + loss_focal
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss
```

**SSIM Loss 상세**:
```python
def ssim_loss(x, y, window_size=11):
    """
    Structural Similarity Index
    - 구조적 유사도 측정 (L2보다 perceptual)
    """
    mu_x = gaussian_filter(x, window_size)
    mu_y = gaussian_filter(y, window_size)
    
    sigma_x = gaussian_filter(x**2, window_size) - mu_x**2
    sigma_y = gaussian_filter(y**2, window_size) - mu_y**2
    sigma_xy = gaussian_filter(x*y, window_size) - mu_x*mu_y
    
    C1, C2 = 0.01**2, 0.03**2
    
    ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    
    return ssim.mean()
```

**Focal Loss 상세**:
```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    이상 영역이 작을 때 class imbalance 해결
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    
    # Modulating factor
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    
    # Alpha balancing
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    
    loss = alpha_t * focal_weight * bce
    return loss.mean()
```

**추론 과정**:
```python
def inference(test_image):
    with torch.no_grad():
        # 1. Reconstruction
        reconstructed = reconstructive_net(test_image)
        
        # 2. Discriminative segmentation
        anomaly_map = discriminative_net(reconstructed)  # (B, 1, H, W)
        
        # 3. Combine reconstruction error + segmentation
        recon_error = torch.abs(test_image - reconstructed).mean(dim=1, keepdim=True)
        
        # Weighted combination
        final_map = 0.5 * anomaly_map + 0.5 * recon_error
        final_map = final_map.squeeze(1)  # (B, H, W)
        
        # Image-level score
        image_score = final_map.max()
    
    return image_score, final_map
```

#### 4.3.4 GANomaly 대비 개선사항

**1) 학습 안정성**:
```
GANomaly:
- Adversarial training으로 불안정
- Mode collapse, oscillation 빈번
- 수렴까지 6-10시간

DRAEM:
- Supervised learning으로 안정
- 명확한 학습 신호 (GT mask)
- 수렴까지 2-4시간 (2-3배 빠름)
```

**2) 성능 향상**:
```
GANomaly: 93-95% AUROC
DRAEM: 97.5% AUROC
개선율: +2.5-4.5%p
```

**3) Few-shot 능력**:
```
GANomaly:
- 정상 샘플 수백 장 필요
- 적은 데이터로 학습 어려움

DRAEM:
- 정상 샘플 10-50장으로도 학습 가능
- Anomaly simulation으로 데이터 증강
```

**4) Interpretability**:
```
GANomaly:
- Latent space distance (해석 어려움)

DRAEM:
- Anomaly segmentation map (명확한 위치)
- Reconstruction error (직관적)
```

**5) 구조 간소화**:
```
GANomaly: E-D-E + Discriminator (4개 네트워크)
DRAEM: Reconstructive + Discriminative (2개 네트워크)
결과: 더 간단하면서 효과적
```

#### 4.3.5 장점
- **높은 정확도**: 97.5% AUROC (reconstruction 중 최고)
- **학습 안정**: GAN 없이 supervised learning
- **Few-shot 가능**: 10-50장 정상 샘플로 학습
- **빠른 학습**: 2-4시간
- **강건성**: 다양한 이상 유형에 효과적
- **Interpretable**: 명확한 segmentation map

#### 4.3.6 단점
- **Simulation 품질 의존**: 가짜 결함의 realistic 정도 중요
- **Domain gap**: 시뮬레이션과 실제 결함 차이
- **추가 데이터**: 텍스처 소스 데이터셋 필요
- **하이퍼파라미터**: Simulation 파라미터 튜닝

#### 4.3.7 성능
- MVTec AD: Image AUROC 97.5%, Pixel AUROC 96.8%
- 추론 속도: 50-100ms per image
- 메모리: 300-500MB
- 학습 시간: 2-4시간

---

### 4.4 DSR (2022) - Dual Subspace Re-Projection

#### 4.4.1 핵심 원리
DSR(Dual Subspace Re-Projection)은 두 개의 독립적인 부분공간(subspace)을 학습하고, 이미지를 이 부분공간에 재투영(re-projection)하여 재구성한다.

**Dual Subspace의 개념**:
```
Subspace 1: Quantization Subspace
- 이미지의 구조적 정보 표현
- Discrete codebook 사용
- VQ-VAE (Vector Quantized VAE) 기반

Subspace 2: Target Subspace
- 세부적인 텍스처 정보 표현
- Continuous representation
- 일반적인 VAE 기반

재구성 = Subspace 1 + Subspace 2
```

#### 4.4.2 DRAEM 대비 핵심 차이점

| 측면 | DRAEM | DSR | 개선 효과 |
|------|-------|-----|----------|
| **학습 방식** | Supervised (simulated anomaly) | Unsupervised (정상만) | 이상 샘플 불필요 |
| **재구성 방법** | 단일 Auto-encoder | Dual subspace projection | 구조+텍스처 분리 |
| **특징 표현** | Continuous latent | Discrete + Continuous | 더 풍부한 표현 |
| **적용 분야** | 일반 결함 | 복잡한 텍스처 표면 | 텍스처 결함 우수 |
| **Image AUROC** | 97.5% | 96.5-98.0% (카테고리별) | 텍스처에서 우수 |

#### 4.4.3 기술적 세부사항

**Quantization Subspace (VQ-VAE)**:
```python
class QuantizationSubspace(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        
        # Encoder: Image → Discrete codes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, embedding_dim, 3, 1, 1)
        )
        
        # Codebook: Learnable discrete embeddings
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1/num_embeddings, 1/num_embeddings)
        
        # Decoder: Discrete codes → Image features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )
    
    def vector_quantization(self, z_e):
        """
        Continuous embedding → Nearest discrete code
        """
        # z_e: (B, D, H, W)
        B, D, H, W = z_e.shape
        
        # Flatten spatial dimensions
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        
        # Find nearest codebook entry
        distances = torch.cdist(z_e_flat, self.codebook.weight)  # (B*H*W, K)
        indices = distances.argmin(dim=1)  # (B*H*W,)
        
        # Quantized embedding
        z_q_flat = self.codebook(indices)  # (B*H*W, D)
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Straight-through estimator (gradient)
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, indices
    
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)  # Continuous
        
        # Vector quantization
        z_q, indices = self.vector_quantization(z_e)  # Discrete
        
        # Decode
        recon_features = self.decoder(z_q)
        
        return recon_features, z_q, z_e
```

**Target Subspace (Standard VAE)**:
```python
class TargetSubspace(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 32, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 32 * 32)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        h_flat = h.flatten(1)
        
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decode
        h_decode = self.fc_decode(z).reshape(-1, 256, 32, 32)
        recon_features = self.decoder(h_decode)
        
        return recon_features, mu, logvar
```

**DSR Complete Model**:
```python
class DSR(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Two subspaces
        self.quantization_subspace = QuantizationSubspace()
        self.target_subspace = TargetSubspace()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),  # 64 from each subspace
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Quantization subspace (structure)
        feat_q, z_q, z_e = self.quantization_subspace(x)
        
        # Target subspace (texture)
        feat_t, mu, logvar = self.target_subspace(x)
        
        # Fuse features
        feat_combined = torch.cat([feat_q, feat_t], dim=1)
        reconstruction = self.fusion(feat_combined)
        
        return reconstruction, (z_q, z_e, mu, logvar)
```

**학습 과정**:
```python
def train_step(normal_images):
    # Forward pass
    reconstructed, (z_q, z_e, mu, logvar) = model(normal_images)
    
    # Loss 1: Reconstruction loss
    loss_recon = F.mse_loss(reconstructed, normal_images)
    
    # Loss 2: VQ commitment loss
    loss_vq = F.mse_loss(z_q.detach(), z_e)
    
    # Loss 3: Codebook loss
    loss_codebook = F.mse_loss(z_q, z_e.detach())
    
    # Loss 4: KL divergence (VAE regularization)
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = loss_recon + 0.25 * loss_vq + loss_codebook + 0.0001 * loss_kl
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss
```

**추론 과정 (Anomaly Detection)**:
```python
def inference(test_image):
    with torch.no_grad():
        # Dual subspace reconstruction
        reconstructed, _ = model(test_image)
        
        # Anomaly map: Reconstruction error
        anomaly_map = torch.abs(test_image - reconstructed).mean(dim=1)
        
        # Feature-level error (from both subspaces)
        feat_q, _, _ = model.quantization_subspace(test_image)
        feat_q_recon, _, _ = model.quantization_subspace(reconstructed)
        error_q = torch.norm(feat_q - feat_q_recon, p=2, dim=1)
        
        feat_t, _, _ = model.target_subspace(test_image)
        feat_t_recon, _, _ = model.target_subspace(reconstructed)
        error_t = torch.norm(feat_t - feat_t_recon, p=2, dim=1)
        
        # Combined error
        final_map = anomaly_map + 0.5 * (error_q + error_t)
        
        # Image score
        image_score = final_map.max()
    
    return image_score, final_map
```

#### 4.4.4 DRAEM 대비 특징

**1) Unsupervised vs Supervised**:
```
DRAEM: Simulated anomaly 필요
DSR: 정상 데이터만으로 학습
→ 이상 샘플 생성 불필요
```

**2) 표현력**:
```
DRAEM: 단일 continuous latent
DSR: Discrete + Continuous dual representation
→ 더 풍부한 특징 표현
```

**3) 복잡한 텍스처 처리**:
```
DRAEM: 일반적인 결함에 강점
DSR: 복잡한 텍스처 표면에 특화
→ 직물, 카펫, 나무 등에서 우수
```

#### 4.4.5 장점
- **복잡한 텍스처**: 직물, 카펫 등에서 우수
- **풍부한 표현**: Discrete + Continuous
- **Unsupervised**: 정상 데이터만 필요
- **구조-텍스처 분리**: 명확한 표현

#### 4.4.6 단점
- **복잡한 구조**: 두 개의 subspace 학습
- **학습 시간**: 각 subspace 학습으로 오래 걸림
- **하이퍼파라미터**: Codebook 크기, latent 차원 등
- **일반 결함**: 단순 결함에서는 DRAEM보다 낮을 수 있음

#### 4.4.7 성능
- MVTec AD: Image AUROC 96.5-98.0% (카테고리별 차이 큼)
- Carpet, Leather, Tile 등 텍스처 카테고리에서 우수
- 추론 속도: 80-120ms per image
- 메모리: 500MB-800MB

---

### 4.5 Reconstruction-Based 방식 종합 비교

#### 4.5.1 기술적 진화 과정

```
GANomaly (2018)
├─ 시작: GAN 기반 E-D-E 구조
├─ 방식: Unsupervised, adversarial
├─ 성능: 93-95% AUROC
└─ 문제: 학습 불안정, 낮은 성능

        ↓ 패러다임 혁신

DRAEM (2021)
├─ 혁신: Simulated anomaly 사용
├─ 방식: Supervised, discriminative
├─ 성능: 97.5% AUROC (+2.5-4.5%p)
├─ 개선: 안정적 학습, Few-shot 가능
└─ 영향: Reconstruction 방식의 새 기준

        ↓ 특화 발전

DSR (2022)
├─ 특화: 복잡한 텍스처 표면
├─ 방식: Dual subspace (VQ-VAE + VAE)
├─ 성능: 96.5-98.0% (텍스처에서 우수)
└─ 특징: 구조-텍스처 분리 표현
```

#### 4.5.2 상세 비교표

| 비교 항목 | GANomaly | DRAEM | DSR |
|----------|----------|-------|-----|
| **발표 연도** | 2018 | 2021 | 2022 |
| **학습 방식** | Unsupervised (GAN) | Supervised (Simulated) | Unsupervised (Dual VAE) |
| **네트워크 구조** | E-D-E + Discriminator | Reconstructive + Discriminative | Quantization + Target Subspace |
| **이상 샘플 사용** | 없음 | Simulated anomaly | 없음 |
| **학습 안정성** | 낮음 (mode collapse) | 높음 (supervised) | 중간 (복잡한 구조) |
| **주요 Loss** | Adversarial + L1 + L2 | SSIM + Focal + L2 | MSE + VQ + KL |
| **특징 표현** | Continuous latent | Continuous latent | Discrete + Continuous |
| **Image AUROC** | 93-95% | 97.5% | 96.5-98.0% |
| **Pixel AUROC** | 91-93% | 96.8% | 95.5-97.5% |
| **추론 속도** | 50-80ms | 50-100ms | 80-120ms |
| **학습 시간** | 6-10시간 | 2-4시간 | 4-6시간 |
| **메모리 사용** | 500MB-1GB | 300-500MB | 500-800MB |
| **Few-shot 능력** | 없음 | 우수 (10-50장) | 중간 |
| **복잡한 텍스처** | 중간 | 중간 | 우수 |
| **단순 결함** | 낮음 | 우수 | 중간 |
| **구현 난이도** | 높음 (GAN) | 중간 | 높음 (Dual subspace) |
| **Interpretability** | 낮음 (latent distance) | 높음 (segmentation map) | 중간 |
| **주요 혁신** | GAN 기반 이상 탐지 | Simulated anomaly | Dual subspace 분리 |
| **적합 환경** | 연구 목적 (deprecated) | 일반 결함 탐지 | 텍스처 표면 검사 |

#### 4.5.3 핵심 Trade-off 분석

**학습 방식 Trade-off**:
```
Unsupervised (GANomaly, DSR):
+ 정상 데이터만 필요
- 암묵적 학습, 성능 제한

Supervised (DRAEM):
+ 명확한 학습 신호
+ 높은 성능
- Simulation 품질 의존
```

**구조 복잡도 vs 성능**:
```
GANomaly (복잡): 4개 네트워크, 93-95%
DRAEM (중간): 2개 네트워크, 97.5%
DSR (복잡): Dual subspace, 96.5-98.0%

결과: 복잡도가 반드시 성능으로 이어지지 않음
DRAEM이 가장 효율적
```

**적용 도메인 Trade-off**:
```
DRAEM: 일반 결함 (97.5%)
DSR: 텍스처 표면 (98.0%)
      단순 결함 (96.5%)

→ 도메인 특화 vs 범용성
```

#### 4.5.4 실무 적용 가이드

**GANomaly 선택 시나리오**:
- 현재는 비추천 (deprecated)
- DRAEM이나 DSR로 대체
- 연구 목적의 baseline으로만 사용
- **추천도**: ★☆☆☆☆

**DRAEM 선택 시나리오**:
- 일반적인 결함 탐지
- Few-shot 학습 필요 (10-50장)
- 안정적인 학습 원함
- 빠른 프로토타이핑
- 명확한 결함 위치 파악 필요
- **추천도**: ★★★★★

**DSR 선택 시나리오**:
- 복잡한 텍스처 표면 (직물, 카펫, 가죽, 나무)
- 구조와 텍스처 모두 중요
- 미세한 텍스처 변화 탐지
- Unsupervised 학습 선호
- **추천도**: ★★★★☆ (텍스처 표면), ★★☆☆☆ (일반)

