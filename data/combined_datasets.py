import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class CombinedDataset:
    def __init__(self, custom_images_path, augment_custom=False):
        print(f"Loading datasets from: {custom_images_path}")
        
        # Load MNIST
        print("Loading MNIST dataset...")
        (self.mnist_train_images, self.mnist_train_labels), \
        (self.mnist_test_images, self.mnist_test_labels) = tf.keras.datasets.mnist.load_data()
        
        # Load custom dataset
        print("Loading custom dataset...")
        self.custom_images, self.custom_labels = self.load_custom_dataset(custom_images_path)
        
        print(f"MNIST training samples: {len(self.mnist_train_images)}")
        print(f"Custom dataset samples: {len(self.custom_images)}")
        
        if len(self.custom_images) == 0:
            print("WARNING: Custom dataset is empty!")
            print("Will use only MNIST dataset for training.")
        
        # Normalize MNIST (0-255 to 0-1)
        self.mnist_train_images = self.mnist_train_images / 255.0
        self.mnist_test_images = self.mnist_test_images / 255.0
        
        # Normalize custom dataset if not empty
        if len(self.custom_images) > 0:
            self.custom_images = self.custom_images / 255.0
        
        # Augment custom dataset if needed
        if augment_custom and len(self.custom_images) > 0:
            print("Augmenting custom dataset...")
            self.custom_images, self.custom_labels = self.augment_dataset(self.custom_images, self.custom_labels)
            print(f"Custom dataset after augmentation: {len(self.custom_images)} samples")
    
    def load_custom_dataset(self, path):
        """Load your custom BMP images."""
        images = []
        labels = []
        
        # Check if path exists
        if not os.path.exists(path):
            print(f"ERROR: Custom dataset path does not exist: {path}")
            return np.array([]), np.array([])
        
        print(f"Scanning folder: {path}")
        files = os.listdir(path)
        print(f"Found {len(files)} files in folder")
        
        for filename in files:
            if filename.endswith('.bmp'):
                try:
                    # Parse label from filename (e.g., "1-0.bmp" -> label=1)
                    label_part = filename.split('-')[0]
                    label = int(label_part)
                    
                    # Load image
                    img_path = os.path.join(path, filename)
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Ensure it's 28x28
                    if img_array.shape == (28, 28):
                        images.append(img_array)
                        labels.append(label)
                    else:
                        print(f"Warning: Image {filename} has shape {img_array.shape}, expected (28, 28)")
                        
                except ValueError as e:
                    print(f"Warning: Could not parse label from filename {filename}: {e}")
                except Exception as e:
                    print(f"Warning: Could not load image {filename}: {e}")
        
        print(f"Successfully loaded {len(images)} images from custom dataset")
        return np.array(images), np.array(labels)
    
    def augment_dataset(self, images, labels):
        """Apply data augmentation to custom dataset."""
        if len(images) == 0:
            return images, labels
            
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Add flipped versions
            augmented_images.append(np.fliplr(img))
            augmented_labels.append(label)
            
            # Add rotated versions (if scipy is available)
            try:
                from scipy.ndimage import rotate
                for angle in [-10, 10]:
                    rotated = rotate(img, angle, reshape=False, mode='reflect')
                    augmented_images.append(rotated)
                    augmented_labels.append(label)
            except ImportError:
                # If scipy not available, just use flips
                pass
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def get_training_data(self, custom_ratio=0.1):
        """Get combined training data."""
        print(f"\nCreating combined training dataset (custom ratio: {custom_ratio})")
        
        # If custom dataset is empty, return only MNIST
        if len(self.custom_images) == 0:
            print("Custom dataset is empty, using only MNIST")
            return self.mnist_train_images, self.mnist_train_labels
        
        # Determine how many custom samples to use
        mnist_train_size = len(self.mnist_train_images)
        custom_size = int(mnist_train_size * custom_ratio)
        
        print(f"MNIST training samples: {mnist_train_size}")
        print(f"Target custom samples: {custom_size}")
        print(f"Available custom samples: {len(self.custom_images)}")
        
        # If we have fewer custom samples than needed, repeat them
        if custom_size > len(self.custom_images):
            repeat_factor = max(1, custom_size // len(self.custom_images))
            custom_images_repeated = np.tile(self.custom_images, (repeat_factor, 1, 1))
            custom_labels_repeated = np.tile(self.custom_labels, repeat_factor)
            
            # Trim to exact size
            custom_images_repeated = custom_images_repeated[:custom_size]
            custom_labels_repeated = custom_labels_repeated[:custom_size]
            print(f"Repeated custom dataset {repeat_factor} times")
        else:
            custom_images_repeated = self.custom_images[:custom_size]
            custom_labels_repeated = self.custom_labels[:custom_size]
        
        # Combine datasets
        combined_images = np.concatenate([self.mnist_train_images, custom_images_repeated])
        combined_labels = np.concatenate([self.mnist_train_labels, custom_labels_repeated])
        
        # Shuffle
        indices = np.arange(len(combined_images))
        np.random.shuffle(indices)
        combined_images = combined_images[indices]
        combined_labels = combined_labels[indices]
        
        print(f"Final combined dataset size: {len(combined_images)}")
        print(f"MNIST: {len(self.mnist_train_images)}, Custom: {len(custom_images_repeated)}")
        
        return combined_images, combined_labels
    
    def get_test_data(self):
        """Get test data (MNIST test + some custom)."""
        # If no custom data, return only MNIST test
        if len(self.custom_images) == 0:
            return self.mnist_test_images, self.mnist_test_labels
        
        # Use all MNIST test + some custom for validation
        custom_test_size = min(1000, len(self.custom_images) // 5)  # 20% of custom data for testing
        
        combined_test_images = np.concatenate([
            self.mnist_test_images,
            self.custom_images[-custom_test_size:]  # Last portion for testing
        ])
        combined_test_labels = np.concatenate([
            self.mnist_test_labels,
            self.custom_labels[-custom_test_size:]
        ])
        
        return combined_test_images, combined_test_labels
    
    def visualize_samples(self, num_samples=10):
        """Visualize samples from both datasets."""
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
        
        # Show MNIST samples
        for i in range(num_samples):
            axes[0, i].imshow(self.mnist_train_images[i], cmap='gray')
            axes[0, i].set_title(f"MNIST: {self.mnist_train_labels[i]}")
            axes[0, i].axis('off')
        
        # Show custom samples (if available)
        if len(self.custom_images) > 0:
            for i in range(min(num_samples, len(self.custom_images))):
                axes[1, i].imshow(self.custom_images[i], cmap='gray')
                axes[1, i].set_title(f"Custom: {self.custom_labels[i]}")
                axes[1, i].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, "No custom images", 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.show()

# Simple test script
if __name__ == "__main__":
    # Test with your dataset path
    dataset_path = "images_processed"  # Change this to your actual path
    
    # Create dataset
    dataset = CombinedDataset(dataset_path, augment_custom=True)
    
    # Visualize samples
    dataset.visualize_samples()
    
    # Get training data
    x_train, y_train = dataset.get_training_data(custom_ratio=0.1)
    
 