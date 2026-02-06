from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Chargement des données
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Visualisation - chiffres
#plt.figure(figsize=(10,5))

#for i in range(10):
#	plt.subplot(2, 5, i + 1)
#	plt.imshow(x_train[i], cmap="gray")
#	plt.title(f"Label: {y_train[i]}")
#	plt.axis("off")

#plt.tight_layout()
#plt.savefig("mnist_samples.png")


# Visualisation de la distribution des classes
#labels, counts = np.unique(y_train, return_counts=True)

#plt.bar(labels, counts)
#plt.xlabel("Digit")
#plt.ylabel("Number of samples")
#plt.title("Distribution des chiffres dans MNIST")

#plt.tight_layout()
#plt.savefig("mnist_distrib.png")

# Statistiques descriptives
# Flatten des images (60000, 784)
x_flat = x_train.reshape(x_train.shape[0], -1)

# --- Satistiques globales ---
global_stats = {
    "mean_pixel": np.mean(x_flat),
    "std_pixel": np.std(x_flat),
    "min_pixel": np.min(x_flat),
    "q1_pixel": np.percentile(x_flat, 25),
    "median_pixel": np.median(x_flat),
    "q3_pixel": np.percentile(x_flat, 75),
    "max_pixel": np.max(x_flat)
}

global_df = pd.DataFrame(global_stats, index=["MNIST Global"])

# ---------- Statistiques par chiffre ----------
stats_per_class = []

for digit in range(10):
    pixels_digit = x_flat[y_train == digit]

    stats_per_class.append({
        "digit": digit,
        "mean_pixel": np.mean(pixels_digit),
        "std_pixel": np.std(pixels_digit),
        "min_pixel": np.min(pixels_digit),
        "q1_pixel": np.percentile(pixels_digit, 25),
        "median_pixel": np.median(pixels_digit),
        "q3_pixel": np.percentile(pixels_digit, 75),
        "max_pixel": np.max(pixels_digit)
    })

class_df = pd.DataFrame(stats_per_class)

# ---------- Table finale ----------
final_df = pd.concat(
    [global_df.reset_index().rename(columns={"index": "digit"}), class_df],
    ignore_index=True
)

print(final_df.round(2))
