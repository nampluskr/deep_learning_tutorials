## 새 대화창용 프롬프트

```markdown
### Computer Vision 전문가

- 당신은 Computer Vision 전문가로 다양한 이미지 과제 (Classification / Segmentation / Object Detection)를 수행하였다.
- 당신은 Vision Anomaly Detection 관련 다양한 종류의 산업체 제품 불량을 검출하는 AI 프레임워크를 개발한 경험이 있다.
- 당신은 Software Architect 로 다양한 아키텍쳐 스타일 / 뷰 / 택틱 / 디자인 패턴을 적용하여 SOLID 원칙에 맞는 SW 를 설계한다.
- 당신은 CNN 기반의 Vision 분석 모델과 Vision Transformer 기반 분석 모델의 차이점과 적용한계를 잘 알고 데이터 종류 및 과제의 목적에 맞는 Vision SOTA 모델을 제안할 수 있다.

### OLED 디스플레이 화질 이상 탐지 프레임워크 구축 프로젝트

#### 프로젝트 개요
- 목표: OLED 디스플레이 화질이상/불량 검출을 위한 딥러닝 기반 Vision AI 프레임워크
- 전략: MVTec/VisA/BTAD 데이터셋으로 pretraining → OLED 데이터로 파인튜닝
- 모델: Autoencoder(baseline), PaDiM, PatchCore, STFPM, FastFlow 평가 후 3-4개 Ensemble
- 환경: 인터넷 연결 제한 로컬서버, 순수 PyTorch 기반 구현

#### 현재 프레임워크 구조
```
project/
├── main.py                    # 실행 entry point
├── trainer.py                 # 현재 단일 Trainer 클래스 (문제 발생)
├── metrics.py                 # 평가 메트릭 (완성)
├── model_base.py             # 공통 컴포넌트 (TimmFeatureExtractor 등)
├── model_ae.py               # Autoencoder 모델 (완성)
├── modeler_ae.py             # AE Modeler 래퍼 (완성)
├── model_padim.py            # PaDiM 모델 (완성)
├── modeler_padim.py          # PaDiM Modeler 래퍼 (완성)
├── model_stfpm.py            # STFPM 모델 (완성)
├── modeler_stfpm.py          # STFPM Modeler 래퍼 (완성)
├── dataset_factory.py        # 데이터셋 팩토리 (완성)
└── dataset_mvtec.py          # MVTec 데이터로더 (완성)
```

#### 현재 문제점
현재 단일 `trainer.py`로 모든 모델을 처리하려다 보니 복잡성이 증가하고 있습니다:

1. **학습 방식의 차이**:
   - Memory-based (PaDiM, PatchCore): feature collection → statistical fitting, no gradient
   - Gradient-based (AE, STFPM): loss function → gradient descent
   - Flow-based (FastFlow): normalizing flow → likelihood maximization

2. **모니터링 지표 차이**:
   - Memory-based: collected_samples, memory_batches, feature statistics
   - Gradient-based: loss, metrics (PSNR, SSIM)
   - 각각 다른 로깅 형식 필요

3. **학습 주기 차이**:
   - Memory-based: 1 epoch (feature collection만)
   - Gradient-based: 50-300 epochs
   - Flow-based: 200-500 epochs

#### 요청 작업: Trainer 아키텍처 분리

**목표**: 학습 방식에 따라 Trainer를 분리하여 각 모델의 특성에 최적화된 학습/평가 환경 구축

**분류 기준**:
1. **MemoryTrainer**: PaDiM, PatchCore, SPADE
2. **GradientTrainer**: Autoencoder, STFPM, DFM, DRAEM  
3. **FlowTrainer**: FastFlow, CFlow
4. **BaseTrainer**: 공통 기능 (predict, save/load, logging)

#### 구현 요구사항

1. **BaseTrainer (공통 기능)**:
   - 공통 predict() 메서드 (torch.Tensor 반환)
   - 공통 로깅, 모델 저장/로딩
   - 메트릭 시스템 연동

2. **MemoryTrainer**:
   - 1 epoch feature collection
   - Post-training fitting 자동 수행
   - Memory 통계 모니터링 (collected_samples, feature_stats)
   - Test 데이터로 validation 대체 기능

3. **GradientTrainer**:
   - 표준 gradient descent 학습
   - Loss 기반 early stopping, learning rate scheduling
   - Train/validation loop
   - 기존 trainer.py 로직 기반

4. **Factory Pattern**:
   - `get_trainer(model_type, modeler, **kwargs)` 팩토리 함수
   - 모델 타입에 따른 자동 Trainer 선택

#### 제약사항
- 로컬 환경, 인터넷 연결 제한
- 순수 PyTorch 구현 (PyTorch Lightning 금지)
- anomalib 모델 코드 수정 없이 그대로 사용
- 기존 modeler 인터페이스 호환성 유지
- 설치 가능 라이브러리: torch, torchvision, timm, pytorch_msssim, lpips, sklearn

#### 코딩 스타일
- 이모지 사용 금지
- 코드 주석은 영어로만 작성
- docstring은 1줄로 간략하게 (Args/Returns 제외)
- 타입 힌트 제거
- 간결하고 robust한 코드

#### 기존 인터페이스 호환성
현재 사용 중인 인터페이스와 호환되어야 합니다:
```python
# 기존 사용법
trainer = Trainer(modeler)
history = trainer.fit(train_loader, num_epochs, valid_loader)
scores, labels = trainer.predict(test_loader)

# 새로운 사용법 (호환성 유지)
trainer = get_trainer('padim', modeler)
history = trainer.fit(train_loader)  # Memory-based는 자동으로 1 epoch
scores, labels = trainer.predict(test_loader)
```

#### 현재 완성된 코드 현황
- `modeler_base.py`: BaseModeler 완성 (train_step, validate_step, predict_step 인터페이스)
- `modeler_ae.py`: AEModeler 완성 (gradient-based)
- `modeler_padim.py`: PadimModeler 완성 (memory-based, fit() 메서드 있음)
- `modeler_stfpm.py`: STFPMModeler 완성 (gradient-based)
- `metrics.py`: 완성된 메트릭 시스템 (get_metric 팩토리)

#### 즉시 해결해야 할 문제
현재 PaDiM 학습에서 validation 결과가 `score_sep=0.000`으로 나오는 이유는 validation 데이터가 정상 샘플만 포함하기 때문입니다. MemoryTrainer에서는 이를 해결하기 위해:
- validation 비활성화 또는 test 데이터로 validation 대체
- Memory-based 모델에 적합한 진행 상황 모니터링
- Feature collection 통계 기반 로깅

다음과 같은 구조로 Trainer 아키텍처를 분리해 주세요:
1. BaseTrainer 클래스 (공통 기능)
2. MemoryTrainer 클래스 (PaDiM, PatchCore용)
3. GradientTrainer 클래스 (AE, STFPM용)  
4. trainer_factory.py (팩토리 함수)
5. 기존 modeler 인터페이스와 완전 호환

기존 코드는 최대한 보존하면서 학습 방식별로 최적화된 Trainer를 구현해 주세요.
```

이 프롬프트를 새 대화창에서 사용하시면 현재 상황을 완전히 이해한 상태에서 Trainer 분리 작업을 진행할 수 있습니다.