## MNIST Classification

### Load Data

```python
import numpy as np
import os
import gzip

def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

## Load Data
data_dir = r"D:\Non_Documents\2025\datasets\mnist"

x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
x_test = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
y_test = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")

## Data Preprocessing
x_train_np = x_train.astype(np.float32).reshape(-1, 28*28) / 255
x_test_np = x_test.astype(np.float32).reshape(-1, 28*28) / 255
y_train_np = y_train.astype(np.int64)
y_test_np = y_test.astype(np.int64)
```

### Modeling

```python
class Module:
    def __call__(self, *args): return self.forward(*args)
    def forward(self, *args): raise NotImplementedError
    def backward(self, *args): raise NotImplementedError
    def parameters(self): return []
    def zero_grad(self): pass

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        ## He Initialization
        self.w *= np.sqrt(2 / in_features)

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.w) + self.b

    def backward(self, dout):
        batch_size = self.x.shape[0]
        self.grad_w[...] = np.matmul(self.x.T, dout) / batch_size
        self.grad_b[...] = np.sum(dout, axis=0) / batch_size
        return np.matmul(dout, self.w.T)

    def parameters(self):
        return [(self.w, self.grad_w), (self.b, self.grad_b)]

    def zero_grad(self):
        self.grad_w[...] = 0
        self.grad_b[...] = 0

class Sigmoid(Module):
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-np.clip(x, -30, 30)))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1 - self.y)

class ReLU(Module):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

class Softmax(Module):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.y

    def backward(self, dout):
        return dout
    
class CrossEntropyLoss(Module):
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        if y_true.ndim == 1:
            self.y_true_onehot = np.zeros_like
            self.y_true_onehot[np.arange(batch_size), y_true] = 1
        else:
            self.y_true_onehot = y_true
            
        self.y_pred = y_pred
        loss = -np.sum(self.y_true_onehot*np.log(y_pred + 1e-8)) / batch_size
        return loss
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        return (self.y_pred - self.y_true_onehot) / batch_size

class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = [
            Linear(input_size, hidden_size),
            Sigmoid(),  ## or ReLU()
            Linear(hidden_size, output_size),
            Softmax(),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
```

### Training

```python
def accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(dim=1)
    return torch.eq(y_pred, y_true).float().mean()

class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr
        
    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad
            
    def zero_grad(self):
        for _, grad in self.parameters:
            grad[...] = 0

model = MLP(28*28, 256, 10)
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)

n_epochs = 20
batch_size = 64
for epoch in range(1, n_epochs + 1):

    batch_loss = batch_acc = 0
    indices = np.random.permutation(len(x_train))
    for i in range(len(x_train) // batch_size):
        x = x_train_np[indices[i*batch_size: (i+1)*batch_size]]
        y = y_train_np[indices[i*batch_size: (i+1)*batch_size]]
        
        print(x.shape, y.shape)
        
        # Forward propagation
        pred = model(x)
        loss = criterion(pred, y)
        # acc = accuracy(pred, y)

        # Backward propagation
        dout = criterion.backward()
        model.backward(dout)
        
        # Update weights and biases
        optimizer.step()
        optimizer.zero_grad()
        
        batch_loss += loss
        # batch_acc += acc.item()

    if epoch % (n_epochs // 10) == 0:
        print(f"[{epoch:2d}/{n_epochs}] loss: {batch_loss/(i+1):.3f} ")
            #   f" acc: {batch_acc/(i+1):.3f}")
```

### claude

```python
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

# 기본 모듈 구현
class Module:
    def __call__(self, *args): 
        return self.forward(*args)
    
    def forward(self, *args): 
        raise NotImplementedError
    
    def backward(self, *args): 
        raise NotImplementedError
    
    def parameters(self): 
        return []
    
    def zero_grad(self): 
        pass

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # He 초기화 사용
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.w) + self.b

    def backward(self, dout):
        batch_size = self.x.shape[0]
        # 그래디언트 누적 (PyTorch 스타일)
        self.grad_w += np.matmul(self.x.T, dout) / batch_size
        self.grad_b += np.sum(dout, axis=0) / batch_size
        return np.matmul(dout, self.w.T)

    def parameters(self):
        return [(self.w, self.grad_w), (self.b, self.grad_b)]
    
    def zero_grad(self):
        self.grad_w[...] = 0
        self.grad_b[...] = 0

class Sigmoid(Module):
    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))  # 안정성을 위한 클리핑
        return self.y
    
    def backward(self, dout):
        return dout * self.y * (1 - self.y)

class ReLU(Module):
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    
    def backward(self, dout):
        return dout * self.mask

class Softmax(Module):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.y
    
    def backward(self, dout):
        return dout

class CrossEntropyLoss(Module):
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        
        # one-hot 인코딩으로 변환
        if y_true.ndim == 1:
            self.y_true_one_hot = np.zeros_like(y_pred)
            self.y_true_one_hot[np.arange(batch_size), y_true] = 1
        else:
            self.y_true_one_hot = y_true
            
        self.y_pred = y_pred
        # 수치 안정성을 위한 작은 값 추가
        loss = -np.sum(self.y_true_one_hot * np.log(y_pred + 1e-7)) / batch_size
        return loss
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        # softmax의 그래디언트와 cross entropy의 그래디언트를 결합
        return (self.y_pred - self.y_true_one_hot) / batch_size

class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        super().__init__()
        self.layers = []
        
        # 첫 번째 레이어
        self.layers.append(Linear(input_size, hidden_size))
        
        # 활성화 함수 선택
        if activation.lower() == 'relu':
            self.layers.append(ReLU())
        else:
            self.layers.append(Sigmoid())
            
        # 출력 레이어
        self.layers.append(Linear(hidden_size, output_size))
        self.layers.append(Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

# SGD 옵티마이저 구현
class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr
        
    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad
            
    def zero_grad(self):
        for _, grad in self.parameters:
            grad[...] = 0

# 정확도 계산 함수
def accuracy(y_pred, y_true):
    y_pred_indices = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_indices == y_true)

# 데이터 로딩 함수
def load_mnist_images(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(data_dir, filename):
    data_path = os.path.join(data_dir, filename)
    with gzip.open(data_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 메인 실행 코드
if __name__ == "__main__":
    # 데이터 경로 설정 (경로는 필요에 따라 변경)
    data_dir = r"D:\Non_Documents\2025\datasets\mnist"
    
    # 데이터 로드
    x_train = load_mnist_images(data_dir, "train-images-idx3-ubyte.gz")
    y_train = load_mnist_labels(data_dir, "train-labels-idx1-ubyte.gz")
    x_test = load_mnist_images(data_dir, "t10k-images-idx3-ubyte.gz")
    y_test = load_mnist_labels(data_dir, "t10k-labels-idx1-ubyte.gz")
    
    # 데이터 전처리
    x_train_np = x_train.astype(np.float32).reshape(-1, 28*28) / 255
    x_test_np = x_test.astype(np.float32).reshape(-1, 28*28) / 255
    y_train_np = y_train.astype(np.int64)
    y_test_np = y_test.astype(np.int64)
    
    # 하이퍼파라미터 설정
    input_size = 28*28
    hidden_size = 256
    output_size = 10
    lr = 0.001
    n_epochs = 20
    batch_size = 64
    
    # 모델, 손실 함수, 옵티마이저 초기화
    model = MLP(input_size, hidden_size, output_size)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    
    # 학습 및 손실, 정확도 기록
    train_losses = []
    train_accs = []
    
    # 학습 루프
    for epoch in range(1, n_epochs + 1):
        batch_loss = batch_acc = 0
        # 데이터 셔플
        indices = np.random.permutation(len(x_train_np))
        n_batches = len(x_train_np) // batch_size
        
        for i in range(n_batches):
            # 미니배치 추출
            x = x_train_np[indices[i*batch_size: (i+1)*batch_size]]
            y = y_train_np[indices[i*batch_size: (i+1)*batch_size]]
            
            # 순전파
            pred = model(x)
            loss = criterion(pred, y)
            acc = accuracy(pred, y)
            
            # 역전파
            dout = criterion.backward()
            model.backward(dout)
            
            # 가중치 업데이트
            optimizer.step()
            optimizer.zero_grad()
            
            # 손실 및 정확도 누적
            batch_loss += loss
            batch_acc += acc
        
        # 에폭당 평균 손실 및 정확도 계산
        avg_loss = batch_loss / n_batches
        avg_acc = batch_acc / n_batches
        
        # 기록
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        
        # 진행 상황 출력
        if epoch % (n_epochs // 10) == 0 or epoch == 1:
            print(f"[{epoch:2d}/{n_epochs}] loss: {avg_loss:.4f}, acc: {avg_acc:.4f}")
    
    # 테스트 세트에서 정확도 평가
    test_batch_size = 100
    n_test_batches = len(x_test_np) // test_batch_size
    test_acc = 0
    
    for i in range(n_test_batches):
        x_test_batch = x_test_np[i*test_batch_size: (i+1)*test_batch_size]
        y_test_batch = y_test_np[i*test_batch_size: (i+1)*test_batch_size]
        
        # 순전파 (평가 모드)
        pred = model(x_test_batch)
        acc = accuracy(pred, y_test_batch)
        test_acc += acc
    
    test_acc /= n_test_batches
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 학습 곡선 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, n_epochs + 1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_epochs + 1), train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    # 예측 결과 시각화
    def visualize_predictions(n_samples=10):
        # 테스트 세트에서 무작위 샘플 선택
        indices = np.random.choice(len(x_test_np), n_samples, replace=False)
        x_samples = x_test_np[indices]
        y_samples = y_test_np[indices]
        
        # 예측
        y_pred = model(x_samples)
        predicted = np.argmax(y_pred, axis=1)
        
        # 시각화
        plt.figure(figsize=(15, 3))
        for i in range(n_samples):
            plt.subplot(1, n_samples, i+1)
            plt.imshow(x_samples[i].reshape(28, 28), cmap='gray')
            plt.title(f"True: {y_samples[i]}\nPred: {predicted[i]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 예측 결과 시각화 함수 호출
    visualize_predictions(10)
```