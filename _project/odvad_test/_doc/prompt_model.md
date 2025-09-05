# OLED 화질 이상 탐지 프레임워크 - SOTA 모델 구현 프롬프트

## 프로젝트 현황 및 구조 개요

### 목표
- OLED 디스플레이 화질 이상 탐지를 위한 모델 평가 프레임워크 구축
- anomalib SOTA 모델들을 순수 PyTorch 환경에서 구현
- 5단계 Level 불량 수준 정량화, 유형 분류, 위치 감지

### 핵심 제약사항
- **인터넷 연결 불가** 로컬 환경에서 동작 / backbone 모델의 가중치를 미리 backbones 폴더에 저장해 놓음
- **anomalib 코드 그대로 사용** - 원본 수정 없이 활용 / InferenceBatch 대신 딕셔너리 사용
- **anomalib component 코드 그대로 사용** - 예. TimmFeatureExtractor 등
- **순수 PyTorch 구현** - Lightning 의존성 제거 -> Modeler 래퍼 클래스로 작성
- **모듈화된 아키텍처** - 팩토리 패턴으로 확장성 확보

## 현재 구현 완료된 시스템 아키텍처

### 1. 현재 Model 구조 - 필수 메서드 구현 패턴

#### **핵심: 모델별 특화된 compute_anomaly_map & compute_anomaly_score**
```python
# STFPM 모델 구조 (참고 모델)
class STFPMModel(nn.Module):
    def __init__(self, layers, backbone="resnet18"):
        super().__init__()
        self.backbone = backbone
        self.teacher_model = TimmFeatureExtractor(backbone=backbone, layers=layers).eval()
        self.student_model = TimmFeatureExtractor(backbone=backbone, layers=layers, requires_grad=True)
        self.anomaly_map_generator = AnomalyMapGenerator()

    def compute_anomaly_map(self, teacher_features: dict, student_features: dict, 
                           image_size: tuple) -> torch.Tensor:
        """STFPM-specific: Feature difference-based anomaly map."""
        return self.anomaly_map_generator(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=image_size
        )
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """STFPM-specific: Max pooling for image-level score."""
        return torch.amax(anomaly_map, dim=(-2, -1))

    def forward(self, images):
        teacher_features = self.teacher_model(images)
        student_features = self.student_model(images)
        
        if self.training:
            return teacher_features, student_features  # Training mode
        
        # Inference mode - 표준 출력 형태로 변환
        anomaly_map = self.compute_anomaly_map(teacher_features, student_features, images.shape[-2:])
        pred_score = self.compute_anomaly_score(anomaly_map)
        return {'pred_score': pred_score, 'anomaly_map': anomaly_map}

# 기존 Reconstruction 모델 예시
class VanillaAE(nn.Module):
    def compute_anomaly_map(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """AE-specific: Pixel-level reconstruction error."""
        return torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """AE-specific: Max anomaly value per image."""
        return torch.amax(anomaly_map, dim=(-2, -1))

# 기존 Flow 모델 예시 (구현 예정)
class FastFlowModel(nn.Module):
    def compute_anomaly_map(self, log_prob: torch.Tensor, output_size: tuple) -> torch.Tensor:
        """Flow-specific: Negative log-likelihood map."""
        # log_prob을 spatial map으로 변환
        return -log_prob.view(-1, 1, *output_size)
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Flow-specific: Mean likelihood per image."""
        return torch.mean(anomaly_map, dim=(-2, -1)).squeeze(1)
```

#### **모델별 compute_anomaly_map 시그니처 패턴**
```python
# Reconstruction 기반 모델들
def compute_anomaly_map(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """픽셀별 재구성 오차 계산"""

# Distillation 기반 모델들  
def compute_anomaly_map(self, teacher_features: dict, student_features: dict, image_size: tuple) -> torch.Tensor:
    """특징 차이 기반 이상 맵 계산"""

# Flow 기반 모델들
def compute_anomaly_map(self, log_prob: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """확률 밀도 기반 이상 맵 계산"""

# GAN/Discriminator 기반 모델들 (DRAEM 등)
def compute_anomaly_map(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                       discriminator_features: torch.Tensor) -> torch.Tensor:
    """재구성 오차 + 판별기 특징 조합"""

# Memory 기반 모델들 (PaDiM, PatchCore 등)
def compute_anomaly_map(self, features: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
    """메모리 뱅크와의 거리 기반 이상 맵"""
```

#### **모델별 compute_anomaly_score 특화 구현**
```python
# 기본 패턴 (대부분 모델)
def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
    """Max pooling - 가장 이상한 픽셀 값"""
    return torch.amax(anomaly_map, dim=(-2, -1))

# Flow 모델 특화
def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
    """Mean pooling - 전체 likelihood 평균"""
    return torch.mean(anomaly_map, dim=(-2, -1)).squeeze(1)

# OLED 특화 (구현 예정)
def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
    """Weighted pooling - 중심부 가중치 적용"""
    # OLED 특성상 중심부 불량이 더 중요
    weight_map = self.create_center_weight_map(anomaly_map.shape[-2:])
    weighted_map = anomaly_map * weight_map
    return torch.sum(weighted_map, dim=(-2, -1)) / torch.sum(weight_map)
```

### 2. 현재 Modeler 구조 (modeler.py)
```python
class BaseModeler(ABC):
    def predict_step(self, inputs) -> dict:
        """표준화된 inference 프로세스"""
        self.model.eval()
        inputs = self.to_device(inputs)
        
        with torch.no_grad():
            # 모델의 forward() 호출 - inference mode
            predictions = self.model(inputs['image'])
            
            # 표준 형태로 반환 (모든 모델 공통)
            return {
                'pred_scores': predictions['pred_score'],
                'anomaly_maps': predictions['anomaly_map']
            }

# STFPM Modeler 구조 (참고)
class STFPMModeler(BaseModeler):
    def train_step(self, inputs, optimizer):
        # Training mode: (teacher_features, student_features) 반환
        teacher_features, student_features = self.model(inputs['image'])
        loss = self.loss_fn(teacher_features, student_features)
        return {'loss': loss.item(), 'feature_sim': similarity_score}

    # predict_step은 BaseModeler에서 자동 처리
    # model.forward() → inference mode → {'pred_score': ..., 'anomaly_map': ...}
```

### 3. 신규 모델 구현 필수 패턴

#### **1단계: 모델별 특화 메서드 구현**
```python
class NewModel(nn.Module):
    def __init__(self, ...):
        # anomalib torch_model.py 내용 그대로 복사
        super().__init__()
        # 모델 아키텍처 초기화
        
    def compute_anomaly_map(self, ...모델별_특화_파라미터...) -> torch.Tensor:
        """모델별 특화된 anomaly map 계산 로직 구현 (필수)
        
        Returns:
            torch.Tensor: [B, 1, H, W] 형태의 anomaly map
        """
        # 모델별 고유 계산 로직
        # - Reconstruction: MSE, SSIM 등
        # - Distillation: Feature difference, cosine similarity 등  
        # - Flow: Negative log-likelihood
        # - GAN: Discriminator score + reconstruction error
        return anomaly_map
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """모델별 특화된 image-level score 계산 (필수)
        
        Args:
            anomaly_map: [B, 1, H, W] pixel-level anomaly map
            
        Returns:
            torch.Tensor: [B,] 형태의 image-level anomaly score
        """
        # 모델별 고유 집계 방식
        # - Max pooling (대부분): torch.amax(anomaly_map, dim=(-2, -1))
        # - Mean pooling (Flow): torch.mean(anomaly_map, dim=(-2, -1))
        # - Weighted pooling (OLED 특화): 중심부 가중치 적용
        # - Percentile pooling: 상위 N% 평균
        return pred_score

    def forward(self, images):
        if self.training:
            # anomalib 원본 training 출력 그대로 유지
            return model_specific_training_outputs
        else:
            # Inference mode: 표준 dict 형태로 변환
            # 1. 모델별 연산 수행
            model_outputs = self.some_model_specific_computation(images)
            
            # 2. compute_anomaly_map 호출 (모델별 특화 파라미터)
            anomaly_map = self.compute_anomaly_map(...model_specific_params...)
            
            # 3. compute_anomaly_score 호출
            pred_score = self.compute_anomaly_score(anomaly_map)
            
            # 4. 표준 형태 반환
            return {'pred_score': pred_score, 'anomaly_map': anomaly_map}
```

#### **2단계: 모델별 특화 Modeler 구현**
```python
class NewModeler(BaseModeler):
    def train_step(self, inputs, optimizer):
        # 모델별 training outputs 처리
        model_outputs = self.model(inputs['image'])  # training mode
        
        # 모델별 loss 계산 (파라미터 매핑)
        if model_type == "reconstruction":
            loss = self.loss_fn(model_outputs[0], inputs['image'])  # (reconstructed, original)
        elif model_type == "distillation":
            loss = self.loss_fn(model_outputs[0], model_outputs[1])  # (teacher, student)
        elif model_type == "flow":
            loss = self.loss_fn(model_outputs[0])  # (log_prob,)
        elif model_type == "draem":
            loss = self.loss_fn(model_outputs[0], inputs['image'], model_outputs[1])  # (recon, orig, disc)
            
        return {'loss': loss.item(), 'model_specific_metric': metric_value}

    # predict_step은 BaseModeler에서 표준 처리
    # self.model(inputs['image']) → inference mode → {'pred_score': ..., 'anomaly_map': ...}
```

## 우선 구현 대상 모델 (우선순위별)

### **1순위: DRAEM** (Reconstruction + Discriminator)
```python
# 예상 구현 패턴
class DRAEMModel(nn.Module):
    def compute_anomaly_map(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                           discriminator_features: torch.Tensor) -> torch.Tensor:
        """DRAEM-specific: Reconstruction error + Discriminator score combination."""
        recon_error = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        disc_score = torch.sigmoid(discriminator_features)  # Discriminator confidence
        # 가중 조합
        return 0.7 * recon_error + 0.3 * disc_score
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """DRAEM-specific: 95th percentile for robust scoring."""
        return torch.quantile(anomaly_map.view(anomaly_map.size(0), -1), 0.95, dim=1)
```

### **2순위: FastFlow** (Normalizing Flow)
```python
# 예상 구현 패턴  
class FastFlowModel(nn.Module):
    def compute_anomaly_map(self, log_prob: torch.Tensor, output_size: tuple) -> torch.Tensor:
        """FastFlow-specific: Negative log-likelihood spatial map."""
        # log_prob: [B, C*H*W] → [B, 1, H, W]
        neg_log_prob = -log_prob.view(-1, *output_size)
        return neg_log_prob.unsqueeze(1)  # Add channel dimension
    
    def compute_anomaly_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """FastFlow-specific: Mean likelihood per image."""
        return torch.mean(anomaly_map, dim=(-2, -1)).squeeze(1)
```

### **3순위: EfficientAD** (Hybrid Approach)
```python
# 예상 구현 패턴
class EfficientADModel(nn.Module):
    def compute_anomaly_map(self, reconstructed: torch.Tensor, original: torch.Tensor,
                           teacher_features: dict, student_features: dict) -> torch.Tensor:
        """EfficientAD-specific: Multi-task anomaly combination."""
        # Reconstruction component
        recon_map = torch.mean((original - reconstructed) ** 2, dim=1, keepdim=True)
        
        # Distillation component  
        feature_map = self.compute_feature_difference(teacher_features, student_features)
        
        # Learned combination weights
        combined_map = self.fusion_network(torch.cat([recon_map, feature_map], dim=1))
        return combined_map
```

## 필요한 anomalib 파일 요청

**EfficientADModel 구현을 위해** 다음 파일들을 제공해주세요:

### 1. **핵심 파일 (필수)**
- **torch_model.py**: DRAEM 메인 모델 클래스
- **loss.py**: DRAEM 손실함수 (reconstruction + discriminator)
- **anomaly_map.py**: DRAEM 이상 맵 생성기 (있는 경우)

### 2. **구현 확인사항**
다음 정보를 함께 제공해주세요:
- **Training outputs**: DRAEM의 forward() 메서드가 training 모드에서 반환하는 값들
- **Anomaly computation**: DRAEM에서 anomaly map을 계산하는 방식
- **Discriminator 역할**: Discriminator가 어떤 특징을 추출하고 어떻게 활용되는지
- **Synthetic data**: Synthetic anomaly 생성 방식 및 학습 과정

### 3. **특화 구현 요구사항**
- `compute_anomaly_map(original, reconstructed, discriminator_features)` 구현 방향
- `compute_anomaly_score(anomaly_map)` 최적 집계 방식
- Reconstruction loss + Adversarial loss 조합 비율

---

**중요** anomalib 라이브러리가 설치되지 않은 환경에서 FastFlow를 실행하기 위해 필요한 컴포넌트들을 코드를 요청하고, 사용자가 제공한 코드를 model_xxx.py에 통합해야 합니다.

**EfficientADModel 의 anomalib 파일들을 제공해주시면, 위의 패턴에 맞춰 모델별 특화된 `compute_anomaly_map`과 `compute_anomaly_score` 메서드를 포함하여 완전한 EfficientADModel 모델을 구현하겠습니다.**


### EfficientAD 파일
```
backbones/
├── resnet18-f37072fd.pth
├── resnet50-0676ba61.pth
├── wide_resnet50_2-95faca4d.pth
├── efficientnet_b0_ra-3dd342df.pth
├── pretrained_teacher_small.pth      # <- 새로 추가
├── pretrained_teacher_medium.pth     # <- 새로 추가
└── imagenette2/                      # <- 선택적
    ├── train/
    └── val/
```

- Pre-trained Teacher 가중치: https://github.com/openvinotoolkit/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip
- ImageNette 데이터셋 (선택적): https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz