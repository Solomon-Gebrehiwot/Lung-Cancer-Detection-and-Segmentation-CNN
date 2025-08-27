"""
Test script for the segmentation model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from segmentation_model import create_unet_model, compile_segmentation_model

def test_unet_model():
    """Test that the U-Net model can be created and compiled."""
    print("Testing U-Net model creation...")
    
    # Create model
    model = create_unet_model(input_shape=(256, 256, 3), num_classes=1)
    
    # Compile model
    model = compile_segmentation_model(model, learning_rate=0.001)
    
    # Print model summary
    model.summary()
    
    # Test with dummy data
    print("\nTesting model with dummy data...")
    dummy_input = np.random.random((1, 256, 256, 3))
    dummy_output = model.predict(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_unet_model()