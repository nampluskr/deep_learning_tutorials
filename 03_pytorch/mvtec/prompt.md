## 프로젝트 개요 및 목표
#### 핵심 목표
- OLED 디스플레이 화질 이상 탐지 프레임워크 구축
- MVTec/VisA/BTAD 데이터셋으로 pretraining → OLED 데이터로 파인튜닝
- 5개 모델 (Autoencoder, FastFlow, PaDiM, PatchCore, STFPM) 평가 후 우수한 3-4개로 Hybrid/Ensemble 구축
- 5단계 Level로 불량 수준 정량화, 유형 분류, 위치 감지

#### 핵심 특징

- 로컬 환경 최적화
  - 인터넷 연결 없는 환경에서 동작
  - pretrained weight를 backbones/ 폴더에 사전 저장
  - 최소한의 외부 라이브러리 의존성
- anomalib 활용 전략
  - 검증된 anomalib 라이브러리의 SOTA 모델들을 수정 없이 그대로 사용
  - PyTorch Lightning 의존성 제거하여 순수 PyTorch로 구현 --> modeler_XXX.py 파일내 XXXModeler 래퍼로 구현
  - 기존 torch_model.py 코드 그대로 활용 --> model_XXX.py 내 XXXModel 로 구현
  - 기존 torch_model.py 파일과 같은 폴더에 있는 loss.py ---> model_XXX.py 내 XXXLoss 로 구현
- 모듈화된 아키텍처
  - 팩토리 패턴으로 모델/데이터셋 확장성 확보
  - Modeler 래퍼로 통일된 train/validate/predict 인터페이스
  - 독립적인 모듈 구성
- OLED 특화 고려사항
  - 고해상도 이미지 처리 대비
  - ΔE2000, FFT 기반 무라 검출 등 전용 메트릭 준비
  - 측정 데이터와 이미지 데이터 융합 가능성

#### 제약사항 및 한계

- 인터넷 연결 제한
  - 실시간 모델/데이터 다운로드 불가
  - 모든 pretrained weight 사전 준비 필요
  - 라이브러리 업데이트/설치 제한

- 라이브러리 제약
  - 설치 완료: timm, sklearn
  - 설치 불가: anomalib, lightning, opencv-python, albumentations, pytorch_msssim, lpips

- 코드 수정 제한
  - anomalib 모델 / 손실함수 / anomaly_map / metric 코드는 원본 그대로 사용
  - 기존 검증된 코드의 안정성 유지 필요

### 이전 대화 프롬프트
- mvtec / visa / btad 데이터셋으로 pretraining 한수 OLED 데이터로 파인튜닝 할려고합니다.
학습할 모델은 Autoencoder (baseline 모델) / fastflow / padim / patchcore / stfpm 모델이고 우수한 3개 ~ 4개의 모델을 Hybrid 또는 Ensemble 모델을 구축할려고 합니다.
- 중요한 점은 구축되는 환경은 인터넷 사용이 안되는 로컬서버이므로, 평가한 모델을 수동으로 업로드해서 사용해야 합니다. 이를 위해 필요한 backbone 파일은 미리 다운로드 받아서 backbones 폴더 아래에 저장해 놓습니다.
- 평가할 sota 모델은 anomalib 라이브러리의 models/image/torch_model.py 를 수정하지 않고 그대로 사용하려고 합니다. 
- 현재 프로젝트 Files 에 첨부된 파일은 초창기 개발중인 코드들입니다. 최종적으로 위의 디렉토리 구조를 가지는 형태로 개발할려고 합니다. 현재의 프로젝트 상황을 분석하고, 할일을 단계별로 분석해 주세요. 코드 구현은 지금은 하지 않고 하나씩 단계별로 요청할 예정입니다.
- 전체 프레임워크는 다음과 같은 구조를 가집니다.
```
anomaly_framework/
│
├─ main.py                     # 실행 entry point (훈련/평가/시각화)
├─ trainer.py                  # Trainer 클래스 (fit/evaluate)
├─ metrics.py                  # 평가 메트릭 (PSNR, SSIM, ΔE2000, FFT, Wavelet, LPIPS 등)
├─ evaluator.py                # (필요시 생성 아직 미구현) 평가 함수들 / 시각화 / 리포트
├─ utils.py                    # (필요시 생성 아직 미구현) 유틸 함수들 / 타일링, 전처리, 후처리
│
├─ datasets/                   # 데이터셋 모듈
│   ├─ __init__.py
│   ├─ dataset_factory.py      # transform / dataloader 팩토리 함수
│   ├─ dataset_mvtec.py        # MVTec AD Dataset + get_dataloaders
│   ├─ dataset_visa.py         # VisA Dataset + get_dataloaders
│   ├─ dataset_btad.py         # BTAD Dataset + get_dataloaders
│   └─ dataset_oled.py         # OLED Custom Dataset + get_dataloaders
│
├─ models/                     # 모델 정의
│   ├─ __init__.py
│   ├─ model_factory.py        # 모델 팩토리 함수 
│   ├─ model_base.py           # 공통 블록 (ResNetFeatureExtractor, EncoderBlock, DecoderBlock)
│   ├─ model_fastflow.py       # FastFlow 원본 구조 + anomaly_map
│   ├─ model_padim.py          # PaDiM 원본 구조 + anomaly_map
│   ├─ model_patchcore.py      # PatchCore 원본 구조 + anomaly_map
│   ├─ model_stfpm.py          # STFPM 원본 구조 + anomaly_map
│   ├─ model_autoencoder.py    # Autoencoder (VanillaAE, UNetAE, BackboneAE 등)
│   └─ model_vae.py            # Variational Autoencoder (VanillaVAE, UNetVAE, BackboneVAE 등)
│
├─ modelers/                   # 모델 래퍼 (훈련/평가 통합 인터페이스)
│   ├─ __init__.py
│   ├─ modeler_base.py         # BaseModeler
│   ├─ modeler_factory.py      # Modeler 팩토리 함수
│   ├─ modeler_fastflow.py     # FastflowModeler (train/validate/predict)
│   ├─ modeler_padim.py        # PadimModeler (dummy train, predict anomaly map)
│   ├─ modeler_patchcore.py    # PatchCoreModeler (dummy train, predict anomaly map)
│   ├─ modeler_stfpm.py        # STFPMModeler (teacher-student feature matching)
│   ├─ modeler_autoencoder.py  # AutoencoderModeler (reconstruction 기반)
│   └─ modeler_vae.py          # VAEModeler (ELBO 기반)
│
└─ experiments/                # 학습 및 평가 결과 저장 (config.output_dir)
    └─ exeriments_name_01
        └─ xxx_weights.pth         # 학습된 모델의 패러미터
        └─ xxx_config.json         # 학습에 사용된 조건 
        └─ xxx_log.txt             # 로그 기록 (학습/평가)
        └─ xxx_history             # 학습 결과 (training/validation)
        └─ xxx_result              # 평가 결과 (test/evaluation) 
    └─ exeriments_name_02
    └─ exeriments_name_03

```
- 현재 첨부 코드의 버그는 다 수정된 상태입니다. 제일 먼저 데이터셋 부터 구현하고자 합니다.
data_dir = ./data/mvtec, ./data/visa, ./data/btad, 에 저장되어 있다.
- BaseDataset 추상 클래스 구현하지 않고, torch.utils.data.Dataset 를 추가 클래스로 사용하겠습니다. 대신 공통 생성자 형식을 반영하고 싶습니다.
```
def init(self, data_dir, categories, split="train", transform=None, normal_only=False, **kwargs):
```
- 4가지 서로 다른 데이터셋에 적용가능한 최적 생성자 형태를 제안해 주세요.
- 현재는 실행되는 Toy 또는 기본 프레임워크 작성이 목적이므로 최소한의 기능과 robust 한 코드가 필요합니다. 필요한 기능은 앞으로 추가 하겠습니다.
- 4개의 데이터셋별로 다음과 같이 4개의 파일을 생성해 주세요.
```
dataset_mvtec.py -> MVTecDataset, get_dataloaders
dataset_btad.py -> BTADDataset, get_dataloaders
dataset_visa.py -> VisADataset, get_dataloaders
dataset_oled.py -> OLEDDataset, get_dataloaders
```
- dataset_mvtec.py 파일을 다음과 같이 작성하였습니다. 위 내용이 잘 반영되었는데, 최소한만 수정해서
간결하고  robust 한 코드가 되게 해 주세요. 최소한의 수정으로 다른 데이터셋에도 적용가능한 형태인지 분석해 주세요. 현 단계에서 다른 메서드를 추가하지 말아 주세요.
- 데이터셋의 categories 는 반드시 리스트 형식으로 주어집니다. 체크하는 코드를 삭제해 주세요.
- datast_XXX.py 파일에는 개별 XXXDataset / XXXDataloader 를 포함하도록 다시 수정하였습니다. 그리고 dataset_factory.py 파일에는 다음과 같은 get_dataloader(name, **params) 팩토리 함수를 추가하였습니다. 각각의 dataset_XXX.py 는 다른 파일과 의존하지 않고 독립적이어야 합니다.
- main.py 는 datasets 폴더에 있는 dataset_factory.py 에 있는 get_dataloader 함수를 임포트해야 합니다. 임포트 하는 코드를 알려주세요.
- 이제 model_XXX 와 modeler_XXX 를 작성해야 합니다. model / loss / anomaly_map 계산함수는 직접 구현하지 않고, anomalib 라이브러리에 구현된 소스 코드를 수정없이 그대로 가져와서 사용하고자 합니다.
```
경로: https://github.com/open-edge-platform/anomalib/tree/main/src/anomalib

model_stfpm.py 파일에는 STFPM 모델의 STFPMModel / STFPMLoss / AnomalyMapGenerator 포함하여 1개의 파일로 만든다.
* https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/stfpm/torch_model.py
* https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/stfpm/loss.py
* https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/stfpm/anomaly_map.py

- 단 외부에서 작성된 공통된 코드는 model_base.py 에 추가한다. 예를 들면 stfpm 모델의 경우에는 TimmFeatureExtractor 가 포함되어야 한다. 필요한 backbone 파일은 backbones 폴더 밑에 미리 저장되어 있다고 가정합니다. 
- 앞에서도 말했듯이, 본 프레임워크는 로컬 장비로 인터넷 연결이 제한적이어서 필요한 파일은 별도로 업로드 해야 한다. 최대한 pytorch 기반으로 from scracth 방식으로 작성되어야 합니다.
- 이렇게 작성된 model_stfpm.py (anomalib 에서 코드 수정없이 복사해서 사용) 을 이용해서
modeler_stfpm.py 에 STFPMModeler 래퍼 클래스를 구현한다.
- 필요한 train_step / validate_step / predict_step / compute_anomaly_scores 함수는
anomallib 파일에서 lightning_model 에 구현된 코드를 최대한 활용해서 작성해야 한다.

* https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/stfpm/lightning_model.py

model_stfpm.py 파일과 modeler_stfpm.py 파일을 생성해 주세요.
```
- model_stfpm.py 에 있는 코드는 anomalib 에 있는 코드를 수정없이 그대로 사용해야 됩니다. 필요하다면 사용자가 해당 코드를 직접 대화창에 붙여 넣어 줄 수 있습니다. 원본 코드가 그대로 사용되었는지 다시 확인해 주세요.
- 먼저 STFPM 에 대한 torch_model.py / loss.py / anomaly_map.py 그리고 lightning_model.py 입니다. modeler_stfpm.py 에 있는 공통 부분은 modeler_base.py 파일로 만들어 주세요.
- 첨부는 timm.py 파일입니다. 이부분도 최대한 그대로 사용되도록 model_base.py 에 반영해 주세요.
- dryrun_find_featuremap_dims 파일 내용입니다. 그대로 추가 해주세요. model_base.py 에 추가해 주세요.
인터넷 연결 의존성은 로컬 시스템으로 연결이 불가합니다. 
- 4개의 pretrained 모델의  weight를 다운 받아야 합니다. 경로를 알려주세요.  코드 생성없이 본문에 4개 파일에 대한 인터넷 경로를 알려주세요.
- bacbones 폴더에 저장된 weights 파일은 해시값을 포함해 원래 파일 이름 그대로 사용하려고 합니다. 
이를 위해 weight 경로가 포함된 코드의 weight path 를 수정해 주세요.
- 현재 프레임워크는 로컬에서 실행되기 때문에 설치된 torch / torchvision 외에는 최대한 설치 라이브러리를 제한해야 합니다. 
- 현재까지 작성된 model_xxx / modeler_xxx 파일에서 추가로 필요한 anomalib 파일이 있는 확인해 주세요. 필요하면 utils.py / metrics.py 등으로 미리 원본 코드를 복사해서 생성해야 합니다.
- 코드 생성을 하지말고 결과를 본문에 보여주세요.
- sklearn 은 기본적으로 설치되어 있어 의존성을 제거할 필요가 없습니다. 간단하고 robust 한 코드가 필요합니다.
- 기존 다른 팩토리 함수와 동일한 형식으로 사용하기 위해, XXXMetric 형식으로 클래스 형태로 다시 작성해 주세요.
- psnr, ssim, lpips metric 도 추가해 주세요. ssim 은 pytorch_msssim 라이브러리를 설치해서 사용합니다.
- lpips 메트릭 적용을 위해 필요한 pretrained pth 파일은  backones 밑에 저장되어 있어야 합니다. 다운 받아야 하는 파일의 경로와, 저장된 weight path 를 포함해서 lpips 메트릭을 수정해 주세요.
- pytorch_msssim / lpips / timee 라이브러리는 이미 설치가 되어 있습니다. 체그 코드를 제거해 주세요. 그리고 코드 간결성을 위해 타입 힌트도 제거해 주세요.
- backbones/ lpips_alex.pth, lpips_vgg.pth, lpips_squeeze.pth 다운받아서 이름 변경, 에러 체크없이 파일 경로를 정의해서 수정되도록 해주세요.  로컬 시스템이라 인터넷으로 파일을 다운로드 받을 수 없습니다.
- get_metric 팩트리 함수를 파일 맨처음에 위치하고 싶습니다. 어떻게 하는게 좋은지 검코해 주세요. available_metrics -> available_classes 로 수정해 주세요.
- `from __future__ import annotations`
- 별도의 모델 weight 를 다운 받을수가 없는데도, timm 라이브러리 설치가 반드시 필요한가요?
- 필요시 인터넷이 되는 다른 시스템에서 필요한 weight 파일을 모두 다운로드 받아서 backbones 파일에 저장합니다. anomalib 모델을 그대로 사용하는데 필요시, pretrained 모델의 패러미터를 다운 받아 사용할 수 있어야 합니다. 이경우 anomailib 및 timm 라이브러리의 의존성이 없어야 합니다. 이 부분이 제대로 구현되었는지 검토해 주세요.

```
backbones/
├── resnet18-f37072fd.pth           # 원래 파일명 그대로
├── resnet50-0676ba61.pth           # 원래 파일명 그대로  
├── wide_resnet50_2-95faca4d.pth    # 원래 파일명 그대로
├── efficientnet_b0_ra-3dd342df.pth # 원래 파일명 그대로
├── lpips_alex.pth                  # 이름 변경
├── lpips_vgg.pth                   # 이름 변경
└── lpips_squeeze.pth               # 이름 변경
```

- 설치된 라이브러리 (winpython 또는 기본 anaconda 파일 모두 설치되어 있음)
```
timm.py
sklearn
```

- 설치하면 절대로 안되는 라이브러리
```
anomalib        # torch_model.py / loss.py / anomaly_map.py 원본 코드 수정없이 그대로 사용
lightning       # pytorch 로 직접 구현 학습/평가 루프
opencv-python   # torchvision 만 사용
albumentation   # torchvision 만 사용
pytorch_msssim  # 원본 코드 수정없이 그대로 사용
lpips           # 원본 코드 수정없이 그대로 사용
```
