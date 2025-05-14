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

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.matmul(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.matmul(dout, self.w.T)

    def parameters(self):
        return [(self.w, self.grad_w), (self.b, self.grad_b)]

class Sigmoid(Module):
    pass

class ReLU(Module):
    pass

class MLP(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = [
            Linear(input_size, hidden_size),
            Sigmoid(),  ## or ReLU()
            Linear(hidden_size, output_size)
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
        for layer in self.parameters:
            params.extend(layer.parameters())
        return params
```

### Training

```python

```
