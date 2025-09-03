```
anomaly_framework/
│
├─ main.py                       # run_experiment(config)
├─ config.py                     # BASE_CONFIGS / DATA_CONFIGS / MODEL_CONFIGS / TRAIN_CONFIGS
├─ utils.py                      # show_xxx_info / show_results
├─ registry.py                   # XXX_REGISTRY (레지스터리) / build_xxx (팩토리 함수)
│
├─ backbones/                    # pretrained bacbones weights
│
├─ dataloaders/                     # 데이터셋 모듈
│   ├─ __init__.py
│   ├─ dataloader_base.py           # BaseDataloader
│   ├─ dataloader_mvtec.py          # MVTecDataset + MVTecDataloader
│   ├─ dataloader_visa.py           # VisADataset + VisADataloader
│   └─ dataloader_btad.py           # BTADDataset + BTADDataloader
│
├─ models/                       # 모델 정의
│   ├─ __init__.py
│   ├─ model_base.py             # 공통 블록 / Feature Extractor / Utility 함수
│   ├─ flow_components.py        # FastFlow 모델 구현시 필요
│   ├─ model_draem.py            # DRAEMModel (원본 구조) +
│   ├─ model_fastflow.py         # FastFlowModel (원본 구조) + anomaly_map
│   ├─ model_padim.py            # PaDiMModel (원본 구조) + anomaly_map
│   ├─ model_patchcore.py        # PatchCoreModel (원본 구조) + anomaly_map
│   ├─ model_stfpm.py            # STFPMModel (원본 구조) + anomaly_map
│   ├─ model_ae.py               # AE models (VanillaAE, UNetAE) + AELoss
│   └─ model_vae.py              # VAE models (VanillaVAE, UNetVAE) + VAELoss
│
├─ modelers/                     # 모델 래퍼 (모델 / 손실함수)
│   ├─ __init__.py
│   ├─ modeler_base.py           # BaseModeler
│   ├─ modeler_draem.py          # DRAEMModel (원본 구조)
│   ├─ modeler_fastflow.py       # FastflowModeler
│   ├─ modeler_padim.py          # PadimModeler
│   ├─ modeler_patchcore.py      # PatchCoreModeler
│   ├─ modeler_stfpm.py          # STFPMModeler
│   ├─ modeler_autoencoder.py    # AEModeler
│   └─ modeler_vae.py            # VAEModeler
│
├─ trainers/                     # Trainer (학습 / 평가)
│   ├─ __init__.py
│   ├─ trainer_base.py           # BaseTrainer
│   ├─ trainer_memory.py         # MemoryTrainer
│   ├─ trainer_gradient.py       # GradientTrainer
│   ├─ trainer_flow.py           # FlowTrainer
│   └─ trainer_classification.py # ClassificationTrainer
│
├─ metrics/                      # Metricvs (검증 / 평가)
│   ├─ __init__.py
│   ├─ metrics_base.py           # AUROC, AUPR, Accuracy, Precision, Recall, F1
│   ├─ metrics_memory.py         # metrics for Memory-based models
│   ├─ metrics_gradient.py       # metrics for Gradient-based models (PSNRMetric, SSIMMetric)
│   ├─ metrics_flow.py           # metrics for Flow-based models
│   ├─ ssim.py                   # pytorch_msssim 원본 코드 사용
│   └─ lpips.py                  # lpips 원본 코드 사용
│
└─ experiments/                # 학습 및 평가 결과 저장 (모델 가중치, anomaly map 시각화)
```

- Modeler 코드 스타일은 lighting 스타일을 참고하여 작성한 것인데, 중요한 차이점이 있습니다.
- validation step은 training step과 동일한 연산을 하되 backpropagation만 생략한 것 입니다.

#### Lightning 표준 패턴 적용
training_step(batch, optimizer)     # 역전파 포함
validation_step(batch)              # 역전파 없음 - 학습 중 검증 (training mode 유지)
test_step(batch)                    # 완전한 추론 모드 - 종합 평가 (inference mode)
predict_step(batch)                 # 단순 예측

### 이상감지 표준 용어
anomaly_maps - 픽셀 레벨 이상 맵
anomaly_scores - 이미지 레벨 이상 점수
normal_mask / anomaly_mask - 정상/이상 마스크
separation - 정상/이상 점수 분리도

### 메트릭 계산 분리
- def compute_train_metrics(self, outputs, batch):      # 학습용 (PSNR, SSIM)
- def compute_valid_metrics(self, outputs, batch):      # 검증용 (조기 종료)
- def compute_test_metrics(self, scores, maps, batch):  # 평가용 (AUROC, AUPR)

#### BaseModeler 구조 - 핵심 메서드 시그너처
```pythons
class BaseModeler(ABC):
    def __init__(self, model, loss_fn=None, metrics=None, device=None):
        """Initialize modeler with model, loss function, and metrics."""
        
    @abstractmethod  
    def forward(self, inputs):
        # 모델별 순전파
        """Core forward pass - model specific implementation."""
        # Returns: model-specific outputs (features, reconstructions, etc.)

    def compute_loss(self, outputs, batch):  # 모델별 손실 계산
        
    def training_step(self, inputs, optimizer):
        """Training step with backpropagation."""
        # Uses: self.forward() + loss computation + backprop
        # Returns: {'loss': float, 'train_metrics': dict}

    @torch.no_grad()
    def validate_step(self, inputs):
        # training step과 동일한 연산을 하되 backpropagation만 생략 (학습중 검증 - 과적합 방지 early-stopping)
        """Validation step during training (no backpropagation, but still in training context)"""
        """Validation step for early stopping (no backprop).""" 
        # Uses: self.forward() + loss + basic metrics (AUROC, etc.)
        # Returns: {'val_loss': float, 'val_auroc': float, 'val_metrics': dict}
        """Validation step during training (no backpropagation, but still in training context)"""
        self.model.train()  # Keep in training mode for teacher-student feature extraction


    @torch.no_grad()
    def evaluate_step(self, inputs):
        """Evaluation step for inference (complete evaluation mode)"""
        self.model.eval()
        
    def compute_anomaly_maps(self, inputs):
        """Compute pixel-level anomaly heatmaps.""" 
        # Uses: self.forward() + model-specific anomaly map generation
        # Returns: torch.Tensor [B, 1, H, W] - pixel-level anomaly intensity
        
    def compute_anomaly_scores(self, inputs):
        """Compute image-level anomaly scores."""
        # Uses: self.compute_anomaly_maps() or direct scoring
        # Returns: torch.Tensor [B] - image-level anomaly scores
```

#### BaseTrainer 구조 - 핵심 메서드 시그너처
```pythons
class BaseTrainer(ABC):
    def __init__(self, modeler, optimizer, scheduler=None, stopper=None, logger=None):
        """Initialize trainer with modeler and training components."""
        
    def fit(self, train_loader, num_epochs, valid_loader=None):
        """Main training loop using training_step + validation_step."""
        # Returns: training history dict
        
    def evaluate(self, test_loader):    
        # 전체 테스트셋 종합 평가
        """Comprehensive evaluation using compute_anomaly_maps/scores."""
        # Returns: {'anomaly_maps': Tensor, 'anomaly_scores': Tensor, 'labels': Tensor, 'metrics': dict}
        
    def predict(self, test_loader): 
        # 단순 점수 반환 (utils.show_results용)
        """Simple prediction returning scores and labels for evaluation."""
        # Returns: (scores_tensor, labels_tensor) - for utils.show_results()
        
    def run_epoch(self, data_loader, epoch, num_epochs, mode):
        """Internal method for running single epoch."""
        # mode: 'train' uses training_step(), 'valid' uses validation_step()
```