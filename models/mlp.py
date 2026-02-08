from tensorflow import keras
from keras import layers

import os
import sys
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.my_dataset import MyDataset

# Optional: save PNG confusion matrices (works without GUI)
try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CUSTOM_ROOT = os.path.join(PROJECT_ROOT, "data", "custom_digits")
CM_DIR = os.path.join(PROJECT_ROOT, "confusion_matrices_mlp")


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
    """
    x4 = x_train[..., None]  # (N, 28, 28, 1)

    aug = keras.Sequential([
        layers.RandomTranslation(0.10, 0.10, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomRotation(0.10, fill_mode="constant", fill_value=0.0, seed=seed),
        layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0, seed=seed),
    ])

    x_aug = aug(x4, training=True).numpy()
    return x_aug[..., 0]


# -------------------------
# Split helper
# -------------------------
def split_train_test(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    idx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(train_ratio * len(idx))
    tr, te = idx[:split], idx[split:]
    return x[tr], y[tr], x[te], y[te]


# -------------------------
# Confusion matrix utils (no sklearn required)
# -------------------------
def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def print_confusion_matrix(cm: np.ndarray, title: str):
    print(f"\n--- {title} ---")
    print(cm)
    print("row=true label, col=pred label")

def save_confusion_matrix_png(cm: np.ndarray, out_path: str, title: str):
    if not MPL_OK:
        print(f"[WARN] matplotlib not available -> cannot save {out_path}")
        return

    import matplotlib.pyplot as plt  # safe re-import

    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()

    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved confusion matrix to {out_path}")

def compute_and_log_cm(model, x_test, y_test, tag: str, save_png: bool = True, n_classes: int = 10):
    # Predict labels
    y_proba = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    cm = confusion_matrix_np(y_test, y_pred, n_classes=n_classes)

    print_confusion_matrix(cm, title=f"{tag} Confusion Matrix")

    if save_png:
        out_path = os.path.join(CM_DIR, f"cm_{tag.lower().replace('[','').replace(']','').replace(' ','_')}.png")
        save_confusion_matrix_png(cm, out_path, title=f"{tag} Confusion Matrix")

    return cm


# -------------------------
# 1) Train MNIST (DO NOT CHANGE TRAINING PARAMS)
# -------------------------
def train_mnist(augment: bool = False, save_cm: bool = True):
    # NOTE: "augment" only affects the tag/log order to match the CNN script.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    model = build_mlp()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    t0 = time.perf_counter()

    model.fit(
        x_train, y_train,
        epochs=25,
        validation_split=0.1,
        batch_size=128,
        verbose=1
    )

    train_time = time.perf_counter() - t0

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    tag = "AUG" if augment else "NOAUG"
    print(f"[MNIST-{tag}] Test loss: {test_loss:.4f}")
    print(f"[MNIST-{tag}] Test accuracy: {test_acc:.4f}")

    # Confusion Matrix (MNIST test set)
    compute_and_log_cm(model, x_test, y_test, tag=f"[MNIST-{tag}]", save_png=save_cm)

    return test_loss, test_acc, train_time


# -------------------------
# 2) Train CUSTOM then test on same CUSTOM (split)
# -------------------------
def train_personnal(augment: bool = False, save_cm: bool = True):
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

    t0 = time.perf_counter()

    model.fit(
        x_tr, y_tr,
        epochs=25,
        validation_split=0.1,
        batch_size=128,
        verbose=1
    )

    train_time = time.perf_counter() - t0
    test_loss, test_acc = model.evaluate(x_te, y_te, verbose=0)

    tag = "AUG" if augment else "NOAUG"
    print(f"[CUSTOM-{tag}] Test loss: {test_loss:.4f}")
    print(f"[CUSTOM-{tag}] Test accuracy: {test_acc:.4f}")

    # Confusion Matrix (CUSTOM test split)
    compute_and_log_cm(model, x_te, y_te, tag=f"[CUSTOM-{tag}]", save_png=save_cm)

    return test_loss, test_acc, train_time


# -------------------------
# 3) Compare (same order / same style as CNN)
# -------------------------
if __name__ == "__main__":
    mnist_no = train_mnist(augment=False, save_cm=True)
    mnist_aug = train_mnist(augment=True, save_cm=True)

    custom_no = train_personnal(augment=False, save_cm=True)
    custom_aug = train_personnal(augment=True, save_cm=True)

    print("\n=== COMPARAISON ===")
    print(f"MNIST  NOAUG: loss={mnist_no[0]:.4f} | acc={mnist_no[1]:.4f} | time={mnist_no[2]:.2f}s ")
    print(f"MNIST AUGMENT: loss={mnist_aug[0]:.4f} | acc={mnist_aug[1]:.4f} | time={mnist_aug[2]:.2f}s")
    print(f"CUSTOM NOAUG: loss={custom_no[0]:.4f} | acc={custom_no[1]:.4f} | time={custom_no[2]:.2f}s")
    print(f"CUSTOM AUGMENT: loss={custom_aug[0]:.4f} | acc={custom_aug[1]:.4f} | time={custom_aug[2]:.2f}s")

    print("\nDelta (CUSTOM - MNIST):")
    print(f"NO AUG : loss: {custom_no[0] - mnist_no[0]:+.4f}   ;   acc: {custom_no[1] - mnist_no[1]:+.4f}")
    print(f"AUG    : loss: {custom_aug[0] - mnist_aug[0]:+.4f}   ;   acc: {custom_aug[1] - mnist_aug[1]:+.4f}")
