### MNIST o4-mini-high

```python
import numpy as np
import os
import gzip

# ────────────────────────────────────────────────────────────────
# 1) MNIST 데이터 로딩 함수
# ────────────────────────────────────────────────────────────────
def load_mnist_images(data_dir, filename):
    path = os.path.join(data_dir, filename)
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    path = os.path.join(data_dir, filename)
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data

# ────────────────────────────────────────────────────────────────
# 2) 기본 Module 클래스
# ────────────────────────────────────────────────────────────────
class Module:
    def __call__(self, *args):
        return self.forward(*args)
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, *args):
        raise NotImplementedError
    def parameters(self):
        return []

# ────────────────────────────────────────────────────────────────
# 3) 레이어 구현
# ────────────────────────────────────────────────────────────────
class Linear(Module):
    def __init__(self, in_features, out_features):
        # He 초기화
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        # 기울기 저장용
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        # x: (batch, in_features)
        self.x = x
        return x.dot(self.W) + self.b  # (batch, out_features)

    def backward(self, grad_output):
        # grad_output: (batch, out_features)
        # 기울기 계산
        self.dW = self.x.T.dot(grad_output)            # (in_features, out_features)
        self.db = np.sum(grad_output, axis=0)          # (out_features,)
        # 하류로 전파할 기울기
        grad_input = grad_output.dot(self.W.T)         # (batch, in_features)
        return grad_input

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

class ReLU(Module):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, grad_output):
        return grad_output * self.mask

class Sigmoid(Module):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)

# ────────────────────────────────────────────────────────────────
# 4) 손실 함수: Softmax + CrossEntropy
# ────────────────────────────────────────────────────────────────
class CrossEntropyLoss(Module):
    def forward(self, logits, y_true):
        # logits: (batch, num_classes), y_true: (batch,)
        # 수치 안정화를 위해 max 빼기
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)  # (batch, classes)
        N = logits.shape[0]
        # -log p_true 의 평균
        self.y_true = y_true
        log_likelihood = -np.log(self.probs[np.arange(N), y_true] + 1e-15)
        return np.mean(log_likelihood)

    def backward(self):
        # dL/dlogits
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y_true] -= 1
        return grad / N

# ────────────────────────────────────────────────────────────────
# 5) MLP 네트워크 정의
# ────────────────────────────────────────────────────────────────
class MLP(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.act = ReLU()
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def backward(self, grad_output):
        grad = self.fc2.backward(grad_output)
        grad = self.act.backward(grad)
        grad = self.fc1.backward(grad)
        return grad

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

# ────────────────────────────────────────────────────────────────
# 6) Simple SGD Optimizer
# ────────────────────────────────────────────────────────────────
class SGD:
    def __init__(self, parameters, lr=0.1):
        # parameters: list of (param_array, grad_array)
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for _, grad in self.parameters:
            grad[...] = 0

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad

# ────────────────────────────────────────────────────────────────
# 7) 정확도 계산 함수
# ────────────────────────────────────────────────────────────────
def accuracy(logits, y):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

# ────────────────────────────────────────────────────────────────
# 8) 학습 및 평가 루프
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 데이터 경로 설정
    data_dir = r"D:\Non_Documents\2025\datasets\fashion_mnist"  # 실제 경로로 수정하세요

    # 1) 데이터 로딩
    x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    x_test  = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
    y_test  = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

    # 2) 전처리: 정규화 & 평탄화
    x_train = x_train.astype(np.float32).reshape(-1, 28*28) / 255.0  # (60000, 784)
    x_test  = x_test.astype(np.float32).reshape(-1, 28*28)  / 255.0  # (10000, 784)
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    # 3) 모델·손실·옵티마이저 초기화
    model    = MLP(input_dim=28*28, hidden_dim=128, output_dim=10)
    loss_fn  = CrossEntropyLoss()
    optimizer= SGD(model.parameters(), lr=0.1)

    # 4) 학습 설정
    num_epochs = 10
    batch_size = 64
    n_train = x_train.shape[0]

    for epoch in range(1, num_epochs+1):
        # (1) 미니배치 학습
        perm = np.random.permutation(n_train)
        x_shuf = x_train[perm]
        y_shuf = y_train[perm]

        for i in range(0, n_train, batch_size):
            xb = x_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]

            # 순전파
            logits = model(xb)               # (batch, 10)
            loss   = loss_fn(logits, yb)     # 스칼라

            # 역전파
            optimizer.zero_grad()
            grad_logits = loss_fn.backward()   # (batch, 10)
            model.backward(grad_logits)
            optimizer.step()

        # (2) 에포크마다 평가
        train_logits = model(x_train[:1000])  # 속도 고려해 1k 샘플로
        train_acc    = accuracy(train_logits, y_train[:1000])
        test_logits  = model(x_test)
        test_acc     = accuracy(test_logits, y_test)

        print(f"[Epoch {epoch:2d}/{num_epochs}] "
              f"Loss: {loss:.4f}  "
              f"TrainAcc: {train_acc*100:5.2f}%  "
              f"TestAcc: {test_acc*100:5.2f}%")
```