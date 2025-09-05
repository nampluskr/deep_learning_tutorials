## 최종 전체 모델 구현 아키텍처 테이블

### 베이스라인 모델 (직접 구현)

| 파일명 | 모델명 | 패러다임 | Modeler | Trainer | 예상 성능(AUROC) |
|--------|--------|----------|---------|---------|-----------------|
| **model_base.py** | - | Common Components | - | - | - |
| **model_ae.py** | VanillaAE | Reconstruction-based | AEModeler | GradientTrainer | 0.65-0.75 |
| | UNetAE | Reconstruction-based | AEModeler | GradientTrainer | 0.70-0.80 |
| **model_vae.py** | VanillaVAE | Probabilistic | VAEModeler | GradientTrainer | 0.68-0.78 |
| | UNetVAE | Probabilistic | VAEModeler | GradientTrainer | 0.75-0.85 |
| | BetaVAE | Probabilistic | BetaVAEModeler | GradientTrainer | 0.70-0.80 |
| **model_wae.py** | MMDWAE | Probabilistic | MMDWAEModeler | GradientTrainer | 0.72-0.82 |
| | WAEGAN | Probabilistic | WAEGANModeler | GANTrainer | 0.75-0.85 |
| **model_clustering.py** | DeepSVDD | Clustering-based | DeepSVDDModeler | ClusteringTrainer | 0.75-0.85 |
| | DAGMM | Clustering-based | DAGMMModeler | GradientTrainer | 0.72-0.82 |
| **model_generative.py** | PixelCNN | Autoregressive | PixelCNNModeler | AutoregressiveTrainer | 0.72-0.82 |
| | MAE | Masking-based | MAEModeler | MaskingTrainer | 0.75-0.85 |
`
### SOTA 모델 (anomalib 기반)

| 파일명 | 모델명 | 패러다임 | Modeler | Trainer | 예상 성능(AUROC) |
|--------|--------|----------|---------|---------|-----------------|
| **model_stfpm.py** | STFPM | Feature-based | STFPMModeler | GradientTrainer | 0.85-0.92 |
| **model_fastflow.py** | FastFlow | Distribution-based | FastFlowModeler | GradientTrainer | 0.82-0.90 |
| **model_draem.py** | DRAEM | Reconstruction-based | DRAEMModeler | GradientTrainer | 0.80-0.88 |
| **model_efficientad.py** | EfficientAD | Feature-based | EfficientADModeler | GradientTrainer | 0.88-0.94 |
| **model_cflow.py** | CFlow | Distribution-based | CFlowModeler | GradientTrainer | 0.82-0.90 |
| **model_ganomaly.py** | GANomaly | Reconstruction-based | GANomalyModeler | GANTrainer | 0.75-0.85 |
| **model_reverse_distillation.py** | ReverseDistillation | Feature-based | ReverseDistillationModeler | GradientTrainer | 0.80-0.88 |
| **model_dfm.py** | DFM | Feature-based | DFMModeler | MemoryTrainer | 0.75-0.85 |
| **model_uflow.py** | UFlow | Distribution-based | UFlowModeler | GradientTrainer | 0.82-0.90 |
| **model_winclip.py** | WinCLIP | Feature-based | WinCLIPModeler | ZeroShotTrainer | 0.70-0.82 |
| **model_cutpaste.py** | CutPaste | Self-supervised | CutPasteModeler | GradientTrainer | 0.85-0.92 |
| **model_simplenet.py** | SimpleNet | Self-supervised | SimpleNetModeler | GradientTrainer | 0.90-0.96 |
| **model_uninet.py** | UniNet | Self-supervised | UniNetModeler | GradientTrainer | 0.88-0.94 |
| **model_padim.py** | PaDiM | Memory-based | PaDiMModeler | MemoryTrainer | 0.78-0.88 |
| **model_patchcore.py** | PatchCore | Memory-based | PatchCoreModeler | MemoryTrainer | 0.85-0.92 |

## 성능 기준 분석

### 베이스라인 성능 범위 (0.65-0.85)
- **최고 베이스라인**: UNetVAE, WAEGAN (0.75-0.85)
- **중위 베이스라인**: DeepSVDD, MAE (0.75-0.85)
- **기본 베이스라인**: VanillaAE (0.65-0.75)

### SOTA 성능 범위 (0.70-0.96)
- **최고 성능**: SimpleNet (0.90-0.96), EfficientAD (0.88-0.94)
- **고성능**: STFPM, CutPaste, UniNet, PatchCore (0.85-0.94)
- **중상위 성능**: FastFlow, CFlow, UFlow (0.82-0.90)
- **중위 성능**: DRAEM, ReverseDistillation (0.80-0.88)

## 패러다임별 성능 요약

### 높은 성능 패러다임 (AUROC > 0.85)
1. **Self-supervised**: SimpleNet (0.90-0.96), UniNet (0.88-0.94), CutPaste (0.85-0.92)
2. **Feature-based**: EfficientAD (0.88-0.94), STFPM (0.85-0.92)
3. **Memory-based**: PatchCore (0.85-0.92)

### 중간 성능 패러다임 (AUROC 0.80-0.85)
- **Distribution-based**: FastFlow, CFlow, UFlow (0.82-0.90)
- **Reconstruction-based**: DRAEM (0.80-0.88)

### 낮은 성능 패러다임 (AUROC < 0.80)
- **Reconstruction-based**: VanillaAE, UNetAE (0.65-0.80)
- **Probabilistic**: 대부분 베이스라인 (0.68-0.85)

## OLED 화질 검사 성능 예상

### Tier 1: 최고 성능 (AUROC > 0.88)
- **SimpleNet** (0.90-0.96): 실시간 + 최고 성능
- **EfficientAD** (0.88-0.94): 밀리초 지연시간
- **UniNet** (0.88-0.94): 최신 contrastive learning

### Tier 2: 고성능 (AUROC 0.85-0.88)
- **STFPM** (0.85-0.92): 다중 스케일 특징
- **CutPaste** (0.85-0.92): Self-supervised 증강
- **PatchCore** (0.85-0.92): 메모리 기반

### Tier 3: 중상위 성능 (AUROC 0.80-0.85)
- **FastFlow** (0.82-0.90): 정규화 플로우
- **CFlow** (0.82-0.90): 조건부 플로우


이 성능 예상치를 통해 **OLED 화질 이상탐지에 가장 적합한 모델들을 우선적으로 선별**할 수 있습니다.


### 베이스라인 모델 (직접 구현)

| 파일명 | 모델명 | 패러다임 | Modeler | Trainer | 순위 |
|--------|--------|----------|---------|---------|------|
| **model_base.py** | - | Common Components | - | - | - |
| **model_ae.py** | VanillaAE | Reconstruction-based | AEModeler | GradientTrainer | 1 |
| | UNetAE | Reconstruction-based | AEModeler | GradientTrainer | 2 |
| **model_vae.py** | VanillaVAE | Probabilistic | VAEModeler | GradientTrainer | 3 |
| | UNetVAE | Probabilistic | VAEModeler | GradientTrainer | 4 |
| | BetaVAE | Probabilistic | BetaVAEModeler | GradientTrainer | 5 |
| **model_wae.py** | MMDWAE | Probabilistic | MMDWAEModeler | GradientTrainer | 6 |
| | WAEGAN | Probabilistic | WAEGANModeler | GANTrainer | 7 |
| **model_clustering.py** | DeepSVDD | Clustering-based | DeepSVDDModeler | ClusteringTrainer | 8 |
| | DAGMM | Clustering-based | DAGMMModeler | GradientTrainer | 9 |
| **model_generative.py** | PixelCNN | Autoregressive | PixelCNNModeler | AutoregressiveTrainer | 선택적 |
| | MAE | Masking-based | MAEModeler | MaskingTrainer | 선택적 |

### SOTA 모델 (anomalib 기반)

| 파일명 | 모델명 | 패러다임 | Modeler | Trainer | 순위 |
|--------|--------|----------|---------|---------|------|
| **model_stfpm.py** | STFPM | Feature-based | STFPMModeler | GradientTrainer | 1 |
| **model_fastflow.py** | FastFlow | Distribution-based | FastFlowModeler | GradientTrainer | 2 |
| **model_draem.py** | DRAEM | Reconstruction-based | DRAEMModeler | GradientTrainer | 3 |
| **model_efficientad.py** | EfficientAD | Feature-based | EfficientADModeler | GradientTrainer | 4 |
| **model_cflow.py** | CFlow | Distribution-based | CFlowModeler | GradientTrainer | 5 |
| **model_ganomaly.py** | GANomaly | Reconstruction-based | GANomalyModeler | GANTrainer | 6 |
| **model_reverse_distillation.py** | ReverseDistillation | Feature-based | ReverseDistillationModeler | GradientTrainer | 7 |
| **model_dfm.py** | DFM | Feature-based | DFMModeler | MemoryTrainer | 8 |
| **model_uflow.py** | UFlow | Distribution-based | UFlowModeler | GradientTrainer | 9 |
| **model_winclip.py** | WinCLIP | Feature-based | WinCLIPModeler | ZeroShotTrainer | 10 |
| **model_cutpaste.py** | CutPaste | **Self-supervised** | CutPasteModeler | GradientTrainer | 11 |
| **model_simplenet.py** | SimpleNet | **Self-supervised** | SimpleNetModeler | GradientTrainer | 12 |
| **model_uninet.py** | UniNet | **Self-supervised** | UniNetModeler | GradientTrainer | 13 |
| **model_padim.py** | PaDiM | Memory-based | PaDiMModeler | MemoryTrainer | 14 |
| **model_patchcore.py** | PatchCore | Memory-based | PatchCoreModeler | MemoryTrainer | 15 |


### 전체 패러다임 (9개)
1. **Reconstruction-based** (재구성 기반)
2. **Probabilistic** (확률적 모델)
3. **Feature-based** (특징 기반) 
4. **Distribution-based** (분포 기반)
5. **Memory-based** (메모리 기반)
6. **Self-supervised** (자기지도학습)
7. **Clustering-based** (클러스터링 기반)
8. **Autoregressive** (자기회귀)
9. **Masking-based** (마스킹 기반)
