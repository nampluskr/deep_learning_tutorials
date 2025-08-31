```
anomaly_framework/
│
├─ main.py                     # run_dataset_model
├─ config.py                   # Configuration / Hyperparameters (SimpleNamespace)
├─ utils.py                    # show_xxx_info / show_results
│
├─ datasets/                   # 데이터셋 모듈
│   ├─ __init__.py
│   ├─ dataset_mvtec.py        # MVTecDataset + MVTecDataloader
│   ├─ dataset_visa.py         # VisADataset + VisADataloader
│   ├─ dataset_btad.py         # BTADDataset + BTADDataloader
│   └─ dataset_oled.py         # OLEDDataset + OLEDDataloader
│
├─ models/                     # 모델 정의
│   ├─ __init__.py
│   ├─ model_base.py           # 공통 블록 / Feature Extractor / Utility 함수
│   ├─ model_fastflow.py       # FastFlowModel (원본 구조) + anomaly_map
│   ├─ model_padim.py          # PaDiMModel (원본 구조) + anomaly_map
│   ├─ model_patchcore.py      # PatchCoreModel (원본 구조) + anomaly_map
│   ├─ model_stfpm.py          # STFPMModel (원본 구조) + anomaly_map
│   ├─ model_autoencoder.py    # AE models (VanillaAE, UNetAE, BackboneAE 등)
│   └─ model_vae.py            # VAE models (VanillaVAE, UNetVAE, BackboneVAE 등)
│
├─ modelers/                   # 모델 래퍼 (모델 / 손실함수)
│   ├─ __init__.py
│   ├─ modeler_base.py         # BaseModeler
│   ├─ modeler_fastflow.py     # FastflowModeler
│   ├─ modeler_padim.py        # PadimModeler
│   ├─ modeler_patchcore.py    # PatchCoreModeler
│   ├─ modeler_stfpm.py        # STFPMModeler
│   ├─ modeler_autoencoder.py  # AEModeler
│   └─ modeler_vae.py          # VAEModeler
│
├─ trainers/                   # Trainer (학습 / 평가)
│   ├─ __init__.py
│   ├─ trainer_base.py         # BaseTrainer
│   ├─ trainer_memory.py       # MemoryTrainer
│   ├─ trainer_gradient.py     # GradientTrainer
│   ├─ trainer_flow.py         # FlowTrainer
│   └─ trainer_oled.py         # OLEDTrainer
│
├─ metrics/                    # Metricvs (검증 / 평가)
│   ├─ __init__.py
│   ├─ metrics_base.py         # AUROC, AUPR, Accuracy, Precision, Recall, F1
│   ├─ metrics_memory.py       # metrics for Memory-based models
│   ├─ metrics_gradient.py     # metrics for Gradient-based models (PSNRMetric, SSIMMetric)
│   ├─ metrics_flow.py         # metrics for Flow-based models
│   ├─ metrics_oled.py         # metrics for OLED
│   ├─ ssim.py                 # pytorch_msssim 원본 코드 사용
│   └─ lpips.py                # lpips 원본 코드 사용

└─ experiments/                # 학습 및 평가 결과 저장 (모델 가중치, anomaly map 시각화)
    ├─ experiments_name_01/
    │   ├─ name_01_weights.pth
    │   ├─ name_01_confing.json
    │   ├─ name_01_history.xxx
    │   ├─ name_01_results.xxx
    │   ├─ name_01_anmaly_map.xxx
    │   └─ name_01_xxx.xxx
    ├─ experiments_name_02/
    │   └─ ...
    └─ experiments_name_03/
        └─ ...
```

```
backbones/
├── resnet18-f37072fd.pth                           # 기존
├── resnet50-0676ba61.pth                           # 기존
├── wide_resnet50_2-95faca4d.pth                   # 기존
├── efficientnet_b0_ra-3dd342df.pth                # 기존
├── lpips_alex.pth                                 # 기존
├── lpips_vgg.pth                                  # 기존
├── lpips_squeeze.pth                              # 기존
├── cait_M48_448-c8ad5d8d.pth                     # 새로 다운로드
└── deit_base_distilled_patch16_384-d5dc9272.pth  # 새로 다운로드
```

파일명: cait_M48_448-c8ad5d8d.pth
다운로드 URL: https://dl.fbaipublicfiles.com/deit/cait_M48_448-c8ad5d8d.pth

파일명: deit_base_distilled_patch16_384-d5dc9272.pth
다운로드 URL: https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d5dc9272.pth