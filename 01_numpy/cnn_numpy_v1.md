### CNN from scratch using numpy

```python
import numpy as np
import os
import gzip

# ────────────────────────────────────────────────────────────────
# 1) MNIST 데이터 로딩
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
# 2) Module 기본 클래스
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
# 3) Conv2d 레이어 (naive loop 버전)
# ────────────────────────────────────────────────────────────────
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0):
        self.in_c = in_channels
        self.out_c = out_channels
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            self.kh, self.kw = kernel_size
        self.stride = stride
        self.pad = padding

        # He 초기화
        scale = np.sqrt(2.0 / (in_channels * self.kh * self.kw))
        self.W = np.random.randn(out_channels, in_channels,
                                 self.kh, self.kw) * scale
        self.b = np.zeros(out_channels)

        # gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        assert C == self.in_c

        # padding
        pad = self.pad
        x_p = np.pad(x,
                     ((0,0),(0,0),(pad,pad),(pad,pad)),
                     mode='constant')
        self.x_p = x_p  # backward 용
        # 출력 크기
        out_h = (H + 2*pad - self.kh) // self.stride + 1
        out_w = (W + 2*pad - self.kw) // self.stride + 1

        out = np.zeros((N, self.out_c, out_h, out_w))

        # convolution
        for n in range(N):
            for oc in range(self.out_c):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = x_p[n, :, h_start:h_start+self.kh,
                                        w_start:w_start+self.kw]
                        out[n, oc, i, j] = np.sum(
                            window * self.W[oc]) + self.b[oc]
        return out

    def backward(self, grad_out):
        # grad_out: (N, out_c, out_h, out_w)
        N, _, out_h, out_w = grad_out.shape
        _, _, H_p, W_p = self.x_p.shape

        # zero gradients
        self.dW.fill(0)
        self.db.fill(0)
        dx_p = np.zeros_like(self.x_p)

        # bias gradient
        self.db[:] = np.sum(grad_out, axis=(0,2,3))

        # weight & input gradient
        for n in range(N):
            for oc in range(self.out_c):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = self.x_p[n, :, h_start:h_start+self.kh,
                                                w_start:w_start+self.kw]
                        grad = grad_out[n, oc, i, j]
                        self.dW[oc] += window * grad
                        dx_p[n, :, h_start:h_start+self.kh,
                                  w_start:w_start+self.kw] += self.W[oc] * grad

        # remove padding
        if self.pad > 0:
            dx = dx_p[:, :,
                      self.pad:-self.pad,
                      self.pad:-self.pad]
        else:
            dx = dx_p
        return dx

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

# ────────────────────────────────────────────────────────────────
# 4) 나머지 레이어들: Flatten, Linear, ReLU
# ────────────────────────────────────────────────────────────────
class Flatten(Module):
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)
    def backward(self, grad):
        return grad.reshape(self.orig_shape)

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0/in_features)
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
    def forward(self, x):
        self.x = x
        return x.dot(self.W) + self.b
    def backward(self, grad_out):
        self.dW = self.x.T.dot(grad_out)
        self.db = np.sum(grad_out, axis=0)
        return grad_out.dot(self.W.T)
    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]

class ReLU(Module):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    def backward(self, grad):
        return grad * self.mask

# ────────────────────────────────────────────────────────────────
# 5) 손실 함수: Softmax + CrossEntropy
# ────────────────────────────────────────────────────────────────
class CrossEntropyLoss(Module):
    def forward(self, logits, y_true):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true
        N = logits.shape[0]
        loss = -np.log(self.probs[np.arange(N), y_true] + 1e-15)
        return np.mean(loss)
    def backward(self):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.y_true] -= 1
        return grad / N

# ────────────────────────────────────────────────────────────────
# 6) 간단한 CNN 모델 정의
# ────────────────────────────────────────────────────────────────
class SimpleCNN(Module):
    def __init__(self):
        # 입력 채널 1 → 출력 채널 8, 3×3 커널, 패딩 1
        self.conv1   = Conv2d(1,  8, 3, padding=1)
        self.relu1   = ReLU()
        self.flatten = Flatten()
        # 28×28 으로 유지되므로 8*28*28 → 128
        self.fc1     = Linear(8*28*28, 128)
        self.relu2   = ReLU()
        self.fc2     = Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)      # → (N,8,28,28)
        x = self.relu1(x)
        x = self.flatten(x)    # → (N,8*28*28)
        x = self.fc1(x)        # → (N,128)
        x = self.relu2(x)
        x = self.fc2(x)        # → (N,10)
        return x

    def backward(self, grad):
        grad = self.fc2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        return grad

    def parameters(self):
        params = []
        for layer in [self.conv1, self.fc1, self.fc2]:
            params += layer.parameters()
        return params

# ────────────────────────────────────────────────────────────────
# 7) 옵티마이저, 정확도 함수
# ────────────────────────────────────────────────────────────────
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr
    def zero_grad(self):
        for _, g in self.params:
            g.fill(0)
    def step(self):
        for p, g in self.params:
            p -= self.lr * g

def accuracy(logits, y):
    return np.mean(np.argmax(logits, axis=1) == y)

# ────────────────────────────────────────────────────────────────
# 8) 학습/평가 루프
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = r"D:\Non_Documents\2025\datasets\mnist"  # 실제 경로로 수정

    # 1) 데이터 로딩
    x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    x_test  = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
    y_test  = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

    # 2) 전처리: normalize & 채널 추가
    x_train = x_train.astype(np.float32)/255.0
    x_test  = x_test.astype(np.float32)/255.0
    # (N,28,28) → (N,1,28,28)
    x_train = x_train[:, None, :, :]
    x_test  = x_test[:, None, :, :]

    # 3) 모델·손실·옵티마이저
    model     = SimpleCNN()
    loss_fn   = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    # 4) 학습 설정
    epochs     = 5
    batch_size = 64
    n_train    = x_train.shape[0]

    for ep in range(1, epochs+1):
        # shuffle
        idx = np.random.permutation(n_train)
        x_s, y_s = x_train[idx], y_train[idx]

        # 학습
        for i in range(0, n_train, batch_size):
            xb = x_s[i:i+batch_size]
            yb = y_s[i:i+batch_size]

            logits = model(xb)             # (B,10)
            loss   = loss_fn(logits, yb)   # 스칼라

            optimizer.zero_grad()
            grad = loss_fn.backward()      # (B,10)
            model.backward(grad)
            optimizer.step()

        # 평가
        train_acc = accuracy(model(x_train[:1000]), y_train[:1000])
        test_acc  = accuracy(model(x_test), y_test)
        print(f"[Epoch {ep}/{epochs}] Loss: {loss:.4f}  "
              f"TrainAcc: {train_acc*100:5.2f}%  "
              f" TestAcc: {test_acc*100:5.2f}%")
```