from tensorflow import keras
from keras import layers

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.my_dataset import MyDataset

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CUSTOM_ROOT = os.path.join(PROJECT_ROOT, "data", "custom_digits")

# -------------------------
# Constants (match your CNN preprocessing)
# -------------------------
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# -------------------------
# Model: EXACT same as your current MLP
# -------------------------
def build_mlp():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.10),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.10),
        layers.Dense(10, activation='softmax')
    ])
    return model


# -------------------------
# Preprocess (CUSTOM only) aligned with CNN
# -------------------------
def preprocess_custom(x: np.ndarray) -> np.ndarray:
    """
    - scale to [0,1]
    - invert (white bg -> black bg like MNIST)
    - normalize with MNIST mean/std
    - ensure shape (N, 28, 28)
    """
    x = np.array(x)
    if x.size == 0:
        return x

    x = x.astype("float32")
    if x.max() > 1.0:
        x /= 255.0

    x = 1.0 - x
    x = (x - MNIST_MEAN) / MNIST_STD

    if x.ndim == 2 and x.shape[1] == 784:
        x = x.reshape(-1, 28, 28)

    return x


# -------------------------
# Light augmentation for CUSTOM (train only)
# -------------------------
def augment_custom_train(x_train: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Small affine-like augmentation using Keras preprocessing layers via a temporary model.
    This keeps it simple and avoids rewriting image ops by hand.
    """
    # Keras augmentation layers work on rank-4 tensors: (N, H, W, C)
    x4 = x_train[..., None]  # add channel dim

    aug = keras.Sequential([
        layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomRotation(0.10, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0, seed=seed),
    ])

    # Run augmentation once to generate a “jittered” version.
    x_aug = aug(x4, training=True).numpy()
    return x_aug[..., 0]  # drop channel dim


# -------------------------
# Split helper (same spirit as torch manual_seed(42))
# -------------------------
def split_train_test(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    idx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(train_ratio * len(idx))
    tr, te = idx[:split], idx[split:]
    return x[tr], y[tr], x[te], y[te]


# -------------------------
# 1) Train MNIST (DO NOT CHANGE TRAINING PARAMS)
# -------------------------
def train_mnist(augment: bool = False):
    # NOTE: "augment" only affects the tag/log order to match the CNN script.
    # We keep MNIST training exactly as in your current mlp.py.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = build_mlp()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=25,
        validation_split=0.1,
        batch_size=128,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    tag = "AUG" if augment else "NOAUG"
    print(f"[MNIST-{tag}] Test loss: {test_loss:.4f}")
    print(f"[MNIST-{tag}] Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc


# -------------------------
# 2) Train CUSTOM (BMP 28x28) then test on same CUSTOM (split)
# -------------------------
def train_personnal(augment: bool = False):
    if not os.path.isdir(CUSTOM_ROOT):
        raise FileNotFoundError(f"Dossier introuvable: {CUSTOM_ROOT}")

    ds = MyDataset(CUSTOM_ROOT)
    x, y = ds.get_training_data(custom_ratio=1.0)

    x = preprocess_custom(x)
    y = np.array(y, dtype=np.int64)

    if x.size == 0:
        raise RuntimeError(f"Aucune donnée custom chargée depuis: {CUSTOM_ROOT}")

    x_tr, y_tr, x_te, y_te = split_train_test(x, y, train_ratio=0.8, seed=42)

    # Augment train only (test stays clean)
    if augment:
        x_tr = augment_custom_train(x_tr, seed=42)

    model = build_mlp()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # We keep the same "feel" as your MLP training (epochs/bs/valsplit),
    # without touching the MNIST part. If you want, you can set different
    # params for custom later—but for now we keep it consistent/simple.
    model.fit(
        x_tr, y_tr,
        epochs=25,
        validation_split=0.1,
        batch_size=128,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)

    tag = "AUG" if augment else "NOAUG"
    print(f"[CUSTOM-{tag}] Test loss: {test_loss:.4f}")
    print(f"[CUSTOM-{tag}] Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc


# -------------------------
# 3) Compare (same order / same style as CNN)
# -------------------------
if __name__ == "__main__":
    mnist_no = train_mnist(augment=False)
    mnist_aug = train_mnist(augment=True)

    custom_no = train_personnal(augment=False)
    custom_aug = train_personnal(augment=True)

    print("\n=== COMPARAISON ===")
    print(f"MNIST  NOAUG: loss={mnist_no[0]:.4f} | acc={mnist_no[1]:.4f}")
    print(f"MNIST AUGMENT: loss={mnist_aug[0]:.4f} | acc={mnist_aug[1]:.4f}")
    print(f"CUSTOM NOAUG: loss={custom_no[0]:.4f} | acc={custom_no[1]:.4f}")
    print(f"CUSTOM AUGMENT: loss={custom_aug[0]:.4f} | acc={custom_aug[1]:.4f}")

    print("\nDelta (CUSTOM - MNIST):")
    print(f"NO AUG : loss: {custom_no[0] - mnist_no[0]:+.4f}   ;   acc: {custom_no[1] - mnist_no[1]:+.4f}")
    print(f"AUG    : loss: {custom_aug[0] - mnist_aug[0]:+.4f}   ;   acc: {custom_aug[1] - mnist_aug[1]:+.4f}")
