# ROADMAP

## Stage 0. Starting Point

### Phase 0.1. NumPy baseline 확인

- [x] 작업 폴더를 생성한다.
- [x] 기본 파일과 폴더를 생성한다.
- [ ] `src/mnist_numpy.py`가 실행 가능한지 확인한다.
- [ ] NumPy 구현의 data loading, model, loss, backward, update, evaluate 흐름을 파악한다.
- [ ] PyTorch 전환 대상 코드를 목록화한다.

## Stage 1. Data Pipeline 전환

### Phase 1.1. Dataset 구조 전환

- [ ] NumPy data loading 코드를 PyTorch `Dataset` 구조로 변환한다.
- [ ] `DataLoader`를 정의한다.
- [ ] batch tensor shape를 확인한다.
- [ ] CUDA device로 batch를 이동하는 흐름을 확인한다.

## Stage 2. Model 전환

### Phase 2.1. PyTorch model 정의

- [ ] NumPy parameter initialization 구조를 확인한다.
- [ ] PyTorch `nn.Module` 기반 CNN classifier를 정의한다.
- [ ] 단일 batch forward pass를 확인한다.

## Stage 3. Train Loop 전환

### Phase 3.1. Loss / optimizer / train function 정의

- [ ] NumPy loss 계산 흐름을 PyTorch loss function으로 전환한다.
- [ ] NumPy backward / update 흐름을 PyTorch autograd와 optimizer로 전환한다.
- [ ] `train` function을 정의한다.
- [ ] 짧은 epoch 실행으로 loss 감소 여부를 확인한다.

## Stage 4. Evaluate / Inference 전환

### Phase 4.1. 평가 및 추론 코드 작성

- [ ] `evaluate` function을 정의한다.
- [ ] test accuracy를 출력한다.
- [ ] 단일 image inference 예제를 작성한다.

## Stage 5. 비교 및 정리

### Phase 5.1. NumPy와 PyTorch 구현 비교

- [ ] NumPy 구현과 PyTorch 구현의 대응 관계를 정리한다.
- [ ] 실행 결과를 `README.md`에 정리한다.
- [ ] 필요한 문서를 `docs/contents/`에 정리한다.
