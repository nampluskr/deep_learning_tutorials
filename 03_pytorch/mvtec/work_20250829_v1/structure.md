```
anomaly_framework/
│
├─ main.py                     # 실행 entry point (훈련/평가/시각화)
├─ trainer.py                  # Trainer 클래스 (fit/evaluate)
├─ metrics.py                  # 평가 메트릭 (PSNR, SSIM, ΔE2000, FFT, Wavelet, LPIPS 등)
│
├─ datasets/                   # 데이터셋 모듈
│   ├─ dataset_mvtec.py        # MVTec AD Dataset + get_dataloaders
│   ├─ dataset_visa.py         # VisA Dataset + get_dataloaders
│   ├─ dataset_btad.py         # BTAD Dataset + get_dataloaders
│   └─ dataset_oled.py         # OLED Custom Dataset + get_dataloaders
│
├─ models/                     # 모델 정의
│   ├─ model_base.py           # 공통 블록 (ResNetFeatureExtractor, EncoderBlock, DecoderBlock)
│   ├─ model_fastflow.py       # FastFlow 원본 구조 + anomaly_map
│   ├─ model_padim.py          # PaDiM 원본 구조 + anomaly_map
│   ├─ model_patchcore.py      # PatchCore 원본 구조 + anomaly_map
│   ├─ model_stfpm.py          # STFPM 원본 구조 + anomaly_map
│   ├─ model_autoencoder.py    # Autoencoder (VanillaAE, UNetAE, BackboneAE 등)
│   └─ model_vae.py            # Variational Autoencoder (VanillaVAE, UNetVAE, BackboneVAE 등)
│
├─ modelers/                   # 모델 래퍼 (훈련/평가 통합 인터페이스)
│   ├─ modeler_base.py         # BaseModeler
│   ├─ modeler_fastflow.py     # FastflowModeler (train/validate/predict)
│   ├─ modeler_padim.py        # PadimModeler (dummy train, predict anomaly map)
│   ├─ modeler_patchcore.py    # PatchCoreModeler (dummy train, predict anomaly map)
│   ├─ modeler_stfpm.py        # STFPMModeler (teacher-student feature matching)
│   ├─ modeler_autoencoder.py  # AutoencoderModeler (reconstruction 기반)
│   └─ modeler_vae.py          # VAEModeler (ELBO 기반)
│
└─ results/                    # 학습 및 평가 결과 저장 (모델 가중치, anomaly map 시각화)
    └─ sample_0.png
       sample_1.png
       ...
```