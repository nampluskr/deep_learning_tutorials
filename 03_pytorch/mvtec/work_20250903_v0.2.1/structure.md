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