#!/usr/bin/env python3
"""
Test script to verify the TensorFlow/Keras environment is working correctly.
Run this inside the Docker container to test the setup.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("=== Testing Package Imports ===")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU(s) available: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
        else:
            print("⚠ No GPUs detected (CPU only)")
            
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✓ PIL/Pillow {Image.__version__}")
    except ImportError as e:
        print(f"✗ PIL/Pillow import failed: {e}")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    return True

def test_project_files():
    """Test that project files are accessible."""
    print("\n=== Testing Project File Access ===")
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check for project files
    project_files = [
        'models/mlp.py',
        'data/combined_datasets.py', 
        'data/my_dataset.py',
        'data/custom_digits/'
    ]
    
    all_found = True
    for file_path in project_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_found = False
    
    return all_found

def test_tensorflow_functionality():
    """Test basic TensorFlow functionality."""
    print("\n=== Testing TensorFlow Functionality ===")
    
    try:
        import tensorflow as tf
        
        # Test basic operations
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = tf.add(a, b)
        
        print(f"✓ Basic tensor operations work: {c.numpy()}")
        
        # Test Keras
        from tensorflow import keras
        from keras import layers
        
        # Create a simple model
        model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(5,)),
            layers.Dense(1, activation='sigmoid')
        ])
        
        print("✓ Keras model creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TP-AI Docker Environment Test")
    print("=" * 40)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test file access
    if not test_project_files():
        success = False
    
    # Test TensorFlow functionality
    if not test_tensorflow_functionality():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! Environment is ready.")
        print("\nYou can now run your training script:")
        print("  python models/mlp.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())