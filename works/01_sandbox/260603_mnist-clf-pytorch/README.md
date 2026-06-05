# MNIST NumPy to PyTorch Classifier

## 개요

이 작업은 이미 작성된 NumPy from scratch MNIST classifier를 기준 구현으로 삼아, 동일한 학습 흐름을 PyTorch 기반 코드로 단계적으로 전환하는 sandbox 작업이다.

NumPy 구현은 학습 흐름을 직접 확인하기 위한 baseline이며, 최종 목표는 Dataset / DataLoader, model, train, evaluate, inference 흐름을 PyTorch 방식으로 재구성하는 것이다.

## 실행 환경

- Conda 환경: `pytorch_env`
- Framework: PyTorch
- Device: CUDA
- Dataset path: `DATASET_DIR`
- Backbone path: `BACKBONE_DIR`

## 기본 실행 흐름

1. `pytorch_env` 환경을 활성화한다.
2. `src/mnist_numpy.py` NumPy baseline을 실행하고 구조를 확인한다.
3. NumPy data loading 흐름을 PyTorch Dataset / DataLoader로 전환한다.
4. NumPy model, loss, backward, update 흐름을 PyTorch model, loss, optimizer, train loop로 전환한다.
5. evaluate와 inference 코드를 작성하고 두 구현의 흐름을 비교한다.

## 주요 경로

| 경로 | 용도 |
|------|------|
| `configs/` | 실행 설정 |
| `src/` | Python source code |
| `scripts/` | 실행 script |
| `tests/` | smoke test |
| `docs/contents/` | 작업 문서 |
| `outputs/` | 실행 결과 |
