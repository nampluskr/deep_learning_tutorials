## 추가 내용

- `config.py`
  - `save_config(config, output_dir, filename)`
  - `load_config(output_dir, filename)`

- `modeler.py`
  - `save_weights(model, output_dir, filename)`
  - `load_weights(output_dir, filename)`

- `evaluator.py`
  - `compute_threshold(scores, labels, method="percentile", percentile=95)`

## 평가 가능 모델 (`img_size = 256 / 512 /1024`)

#### 1. VanillaAE (vanilla_ae)
- 구조: 단순 CNN 기반 Autoencoder
- 특징: 입력 이미지를 인코더 → 디코더로 복원, reconstruction error 로 anomaly 판단
- 장점: 구조가 단순하여 baseline 비교용으로 적합

#### 2. UNetAE (unet_ae)
- 구조: UNet-style Autoencoder (skip connection 포함)
- 특징: 인코더의 중간 feature map 을 디코더와 연결, 세부 정보 보존
- 장점: 미세 구조 anomaly 검출에 더 효과적

#### 3. VAE (vae)
- 구조: Variational Autoencoder (확률적 latent space)
- 특징: 입력 이미지를 잠재 확률 분포로 매핑, 샘플링 후 복원
- 손실: reconstruction loss + KL divergence
- 장점: 불확실성 추정 및 잠재 공간 regularization

#### 4. BetaVAE (beta_vae)
- 구조: VAE 변형 모델
- 특징: KL 항에 β 가중치 적용 (정보 압축 강화)
- 장점: disentangled representation 학습 가능 → anomaly 특성 분리 용이

#### 5. WAE (wae)
- 구조: Wasserstein Autoencoder
- 특징: 잠재 공간과 prior 분포 간 차이를 KL 대신 MMD penalty로 측정
- 장점: 잠재 공간 분포를 더 유연하게 학습, VAE 대비 mode collapse 완화

#### 6. BackboneVanillaAE (backbone_vanilla_ae)
- 구조: 사전학습(Pretrained) CNN backbone + VanillaAE 디코더
- 지원 백본: ResNet34 / ResNet50 / VGG16 / VGG19
- 특징: backbone feature → global latent → decoder 복원
- 장점: 대규모 사전학습 네트워크 활용으로 일반화 성능 향상

#### 7. BackboneUNetAE (backbone_unet_ae)
- 구조: 사전학습(Pretrained) CNN backbone + UNet-style decoder
- 지원 백본: ResNet34 / ResNet50 / VGG16 / VGG19
- 특징: backbone feature pyramid + skip connections → 고해상도 anomaly 복원
- 장점: 고해상도 OLED anomaly 와 같은 미세 패턴 검출에 적합
