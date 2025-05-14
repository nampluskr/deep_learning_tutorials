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
