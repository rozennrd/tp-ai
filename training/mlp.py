import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from tensorflow import keras
from keras import layers
from keras . datasets import mnist
from utils.combined_datasets import CombinedDataset
from data.my_dataset import MyDataset
import numpy as np

dataset_path = "images_processed"  # Change this to your actual path
    
# Create dataset
# dataset = CombinedDataset(dataset_path, augment_custom=True)
new_dataset = MyDataset(dataset_path)
dataset = mnist

# Visualize samples
# dataset.visualize_samples()

# Get training data
x_train_new_ds, y_train_new_ds = new_dataset.get_training_data(custom_ratio=0.1)
# x_test, y_test = dataset.get_test_data()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"\nTraining data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Unique labels: {np.unique(y_train)}")


# Definir le modele avec l'api sequential

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.10),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.10),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=25,
    validation_split=0.1,
    batch_size=128
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

new_dataset_test_loss, new_dataset_test_acc = model.evaluate(x_train_new_ds, y_train_new_ds, verbose=1) 

print(f"Test accuracy: {test_acc:.4f}")

print (f"Test accuracy on new dataset: {new_dataset_test_acc:.4f}")


