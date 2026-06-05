# SPEC

## 1. 목적

이미 작성된 NumPy from scratch MNIST classifier를 기준 구현으로 삼아, 동일한 학습 흐름을 PyTorch 기반 코드로 단계적으로 전환한다.

이 작업의 목적은 NumPy 구현과 PyTorch 구현의 대응 관계를 확인하면서 Dataset / DataLoader, model, loss, backward, optimizer, train, evaluate, inference 흐름을 이해하는 것이다.

## 2. 범위

- NumPy from scratch baseline 분석
- NumPy data loading 흐름을 PyTorch Dataset / DataLoader로 전환
- NumPy parameter / forward 흐름을 PyTorch model로 전환
- NumPy loss / backward / update 흐름을 PyTorch loss / autograd / optimizer로 전환
- CUDA 기반 training loop 구현
- evaluation loop 구현
- inference 예제 작성
- smoke test 작성
- 실행 결과를 `outputs/`에 저장

## 3. 데이터셋

MNIST dataset을 사용한다.
dataset root는 코드 상단의 `DATASET_DIR`로 정의한다.

```python
import os

DATASET_DIR = os.getenv("DATASET_DIR", "/mnt/d/datasets/mnist")
BACKBONE_DIR = os.getenv("BACKBONE_DIR", "/mnt/d/backbones")
```

현재 단계에서는 `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` 파일을 직접 읽는다.
경로 결합은 `os.path.join`을 사용한다.

## 4. 모델 및 Backbone

NumPy baseline은 from scratch 방식으로 작성된 MNIST classifier를 사용한다.
PyTorch 전환 단계에서는 동일한 학습 흐름을 `nn.Module`, autograd, optimizer 기반 코드로 재구성한다.
pretrained backbone은 기본 사용하지 않지만, 경로 기준은 `BACKBONE_DIR`를 유지한다.

## 5. 실행 환경

- OS 환경: WSL
- Conda 환경: `pytorch_env`
- Framework: PyTorch
- Device: CUDA

CUDA가 사용할 수 없는 경우 실행하지 않는다.

## 6. 작업별 규칙

- type hint는 사용하지 않는다.
- 경로 처리는 `pathlib.Path`를 사용하지 않고 `os.path`를 사용한다.
- Python 코드 내 주석은 필요한 경우에만 영어로 작성한다.
- tutorial 흐름을 숨기는 과도한 추상화는 사용하지 않는다.
- 대용량 dataset, checkpoint, pretrained weight는 Git 추적 대상으로 추가하지 않는다.

## 7. 구현 단계

1. `src/mnist_numpy.py` NumPy baseline을 확인한다.
2. NumPy 구현의 data loading, model, loss, backward, update, evaluate 흐름을 분석한다.
3. PyTorch `Dataset` / `DataLoader`를 정의한다.
4. PyTorch model을 정의한다.
5. PyTorch training function을 정의한다.
6. PyTorch evaluation function을 정의한다.
7. PyTorch inference 예제를 작성한다.
8. NumPy 구현과 PyTorch 구현의 대응 관계를 정리한다.

## 8. 평가 기준

- CUDA 사용 가능 여부를 확인한다.
- MNIST train/test image와 label shape를 확인한다.
- 단일 batch forward pass가 성공한다.
- 짧은 학습 실행이 성공한다.
- evaluation 결과가 출력된다.
- NumPy 구현과 PyTorch 구현의 주요 단계 대응 관계가 정리되어 있다.

## 9. 산출물

| 산출물 | 위치 |
|------|------|
| 학습 로그 | `outputs/logs/` |
| 결과 이미지 | `outputs/figures/` |
| checkpoint | `outputs/checkpoints/` |
