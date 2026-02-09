import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class MyDataset:
    def __init__(self, custom_images_path):
        print(f"Loading datasets from: {custom_images_path}")
        
        # Load custom dataset
        print("Loading custom dataset...")
        self.custom_images, self.custom_labels = self.load_custom_dataset(custom_images_path)
        
        
        print(f"Custom dataset samples: {len(self.custom_images)}")
        
        if len(self.custom_images) == 0:
            print("WARNING: Custom dataset is empty!")
        
        # Normalize custom dataset if not empty
        if len(self.custom_images) > 0:
            self.custom_images = self.custom_images / 255.0

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

    def get_training_data(self):
        """Get training data from custom dataset."""
        print(f"\nCreating training dataset")
        
        if len(self.custom_images) == 0:
            print("ERROR: No custom images available!")
            return np.array([]), np.array([])
        
        print(f"Available custom samples: {len(self.custom_images)}")
       
        # Shuffle the data
        indices = np.arange(len(self.custom_images))
        np.random.shuffle(indices)
        custom_images = self.custom_images[indices]
        custom_labels = self.custom_labels[indices]
        
        print(f"Training dataset size: {len(custom_images)}")
        
        return custom_images, custom_labels
    
    def get_test_data(self):
        """Get test data from custom dataset."""
        if len(self.custom_images) == 0:
            print("ERROR: No custom images available!")
            return np.array([]), np.array([])
        
        # Use 20% of custom data for testing
        custom_test_size = min(1000, len(self.custom_images) // 5)
        
        # Take last portion for testing
        test_images = self.custom_images[-custom_test_size:]
        test_labels = self.custom_labels[-custom_test_size:]
        
        print(f"Test dataset size: {len(test_images)}")
        
        return test_images, test_labels
    
    def visualize_samples(self, num_samples=10):
        """Visualize samples from the custom dataset."""
        if len(self.custom_images) == 0:
            print("No custom images to visualize!")
            return
        
        # Limit to available images
        num_samples = min(num_samples, len(self.custom_images))
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            axes[i].imshow(self.custom_images[i], cmap='gray')
            axes[i].set_title(f"Label: {self.custom_labels[i]}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
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
