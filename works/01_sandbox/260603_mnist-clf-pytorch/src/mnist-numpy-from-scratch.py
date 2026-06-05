import gzip
import os
import numpy as np


def load_mnist_image(data_dir, split="train"):
    file_name = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    file_path = os.path.join(data_dir, file_name)

    with gzip.open(file_path, "rb") as f:
        f.read(16)
        images = np.frombuffer(f.read(), dtype=np.uint8)

    return images.reshape(-1, 28, 28)


def load_mnist_labels(data_dir, split="train"):
    file_name = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    file_path = os.path.join(data_dir, file_name)

    with gzip.open(file_path, "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def onehot(labels, num_classes):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def normalize(images):
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(dout, x):
    return dout * (x > 0)

def softmax(logits):
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy(probs, labels_onehot):
    batch_size = labels_onehot.shape[0]
    clipped_probs = np.clip(probs, 1e-7, 1.0)
    return -np.sum(labels_onehot * np.log(clipped_probs)) / batch_size


def init_params(input_dim, hidden_dim, output_dim):
    params = {
        "w1": np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim),
        "b1": np.zeros((1, hidden_dim), dtype=np.float32),
        "w2": np.random.randn(hidden_dim, output_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim),
        "b2": np.zeros((1, output_dim), dtype=np.float32),
    }
    return params


def forward(x, params):
    z1 = np.matmul(x, params["w1"]) + params["b1"]
    a1 = relu(z1)
    logits = np.matmul(a1, params["w2"]) + params["b2"]
    probs = softmax(logits)

    cache = {"x": x, "z1": z1, "a1": a1, "probs": probs}
    return probs, cache


def backward(labels_onehot, params, cache):
    batch_size = labels_onehot.shape[0]
    grad_logits = (cache["probs"] - labels_onehot) / batch_size

    grads = {}
    grads["w2"] = np.matmul(cache["a1"].T, grad_logits)
    grads["b2"] = np.sum(grad_logits, axis=0, keepdims=True)

    grad_a1 = np.matmul(grad_logits, params["w2"].T)
    grad_z1 = relu_grad(grad_a1, cache["z1"])
    grads["w1"] = np.matmul(cache["x"].T, grad_z1)
    grads["b1"] = np.sum(grad_z1, axis=0, keepdims=True)
    return grads


def update_params(params, grads, learning_rate):
    for key in params:
        params[key] -= learning_rate * grads[key]


def accuracy(x, labels, params, batch_size):
    preds = []

    for start_idx in range(0, x.shape[0], batch_size):
        end_idx = start_idx + batch_size
        batch_x = x[start_idx:end_idx]
        probs, _ = forward(batch_x, params)
        preds.append(np.argmax(probs, axis=1))

    preds = np.concatenate(preds)
    return np.mean(preds == labels)


def train(x_train, y_train, x_test, y_test, max_epochs, batch_size, learning_rate, hidden_dim):
    input_dim = x_train.shape[1]
    output_dim = 10
    params = init_params(input_dim, hidden_dim, output_dim)
    y_train_onehot = onehot(y_train, output_dim)

    for epoch in range(1, max_epochs + 1):
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]
        y_train_onehot = y_train_onehot[indices]
        epoch_loss = 0.0
        batch_count = 0

        for start_idx in range(0, x_train.shape[0], batch_size):
            end_idx = start_idx + batch_size
            batch_x = x_train[start_idx:end_idx]
            batch_y =
            
             y_train_onehot[start_idx:end_idx]

            probs, cache = forward(batch_x, params)
            loss = cross_entropy(probs, batch_y)
            grads = backward(batch_y, params, cache)
            update_params(params, grads, learning_rate)

            epoch_loss += loss
            batch_count += 1

        train_acc = accuracy(x_train, y_train, params, batch_size)
        test_acc = accuracy(x_test, y_test, params, batch_size)
        avg_loss = epoch_loss / batch_count

        print(
            f"epoch: {epoch + 1}/{max_epochs}, "
            f"loss: {avg_loss:.3f}, "
            f"train_acc: {train_acc:.3f}, "
            f"test_acc: {test_acc:.3f}"
        )
    return params


if __name__ == "__main__":
    DATASET_DIR = os.getenv("DATASET_DIR", "/mnt/d/datasets/mnist")
    SEED = 42
    MAX_EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    HIDDEN_DIM = 128

    np.random.seed(SEED)

    train_images = load_mnist_image(DATASET_DIR, split="train")
    train_labels = load_mnist_labels(DATASET_DIR, split="train")
    test_images = load_mnist_image(DATASET_DIR, split="test")
    test_labels = load_mnist_labels(DATASET_DIR, split="test")

    x_train = normalize(train_images)
    x_test = normalize(test_images)

    print(f"train_images: {train_images.shape}, {train_images.dtype}")
    print(f"train_labels: {train_labels.shape}, {train_labels.dtype}")
    print(f"test_images: {test_images.shape}, {test_images.dtype}")
    print(f"test_labels: {test_labels.shape}, {test_labels.dtype}")

    model_params = train(
        x_train,
        train_labels,
        x_test,
        test_labels,
        MAX_EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        HIDDEN_DIM,
    )
