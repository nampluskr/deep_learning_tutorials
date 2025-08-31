## OLED 화질 이상 탐지 프레임워크 - 추가 SOTA 모델 구현 요청

현재 OLED 화질 이상 탐지를 위한 PyTorch 기반 프레임워크를 개발 중이며, AutoEncoder, PaDiM, STFPM 모델 구현이 완료되었습니다. 추가 SOTA 모델을 구현하여 성능을 비교 평가하고자 합니다.

### 요청 사항

**[모델명] 모델을 위한 다음 2개 파일을 생성해주세요:**

1. **model_[모델명].py**: anomalib의 torch_model.py + loss.py + anomaly_map.py 통합
2. **modeler_[모델명].py**: lightning_model.py를 BaseModeler 형식으로 변환

### 제공할 원본 파일들

다음 anomalib 원본 파일들을 첨부하겠습니다:
- `torch_model.py` (모델 아키텍처)
- `loss.py` (손실 함수)
- `anomaly_map.py` (이상 맵 생성, 있는 경우)
- `lightning_model.py` (Lightning 기반 학습 로직)

### 구현 요구사항

**model_[모델명].py 파일:**
- anomalib 원본 코드를 **수정 없이** 그대로 합쳐서 구성
- InferenceBatch(pred_score, anomaly_map) NamedTuple 사용
- 기존 model_padim.py, model_stfpm.py와 동일한 구조 유지
- 필요한 공통 함수가 model_base.py에 없으면 추가 요청

**modeler_[모델명].py 파일:**
- BaseModeler 클래스 상속 (modeler_base.py 기준)
- train_step(), validate_step(), predict_step() 메서드 구현
- compute_anomaly_scores() 메서드 구현  
- configure_optimizers() 메서드 구현
- learning_type 속성 정의 ("one_class" 또는 "supervised")
- 기존 modeler_padim.py, modeler_stfpm.py와 일관성 유지

### 기존 프레임워크 호환성

생성된 파일들은 다음과 같이 사용 가능해야 합니다:
```python
def run_[모델명](verbose=False):
    print("\n" + "="*50 + "\nRUNNING EXPERIMENT: [모델명]\n" + "="*50)
    
    categories=["carpet", "leather", "tile"]
    data = get_data_for_[타입](categories)  # gradient/memory/flow
    
    modeler = [모델명]Modeler(
        model = [모델명]Model(...),
        loss_fn = [모델명]Loss(...),
        metrics = {...},
    )
    trainer = [타입]Trainer(modeler, scheduler=None, stopper=None, logger=None)
    
    if verbose:
        show_data_info(data)
        show_modeler_info(modeler)
        show_trainer_info(trainer)
    
    trainer.fit(data.train_loader(), num_epochs=[적절한 값], valid_loader=data.valid_loader())
    scores, labels = trainer.predict(data.test_loader())
    show_results(scores, labels)
```

### 참고 정보

- **로컬 환경**: 인터넷 연결 없음, 모든 weights는 backbones/ 폴더에 저장
- **사용 불가 라이브러리**: anomalib, lightning, opencv, albumentations
- **Trainer 타입**: GradientTrainer, MemoryTrainer, FlowTrainer 중 모델 특성에 맞게 선택
- **주석**: 영어로만 작성, docstring은 1줄로 간략하게

구현하고자 하는 모델명: **[FastFlow/PatchCore/CFlow/DRAEM/DFM 등]**