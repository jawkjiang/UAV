"""
Model Interface Validation Test

Tests that all models conform to the required interface:
- Input: [batch_size, 50, 13]
- Output: [batch_size, 1] with values in [0, 1]
"""

import torch
import sys

from model import create_model


def test_model_interface():
    """Test that all models conform to interface."""
    
    model_types = ['cnn', 'lstm', 'bilstm', 'gru', 'cnn_lstm', 'tcn', 'transformer']
    n_features = 13
    window_size = 50
    batch_size = 4
    
    print("="*80)
    print("Model Interface Validation Test")
    print("="*80)
    print(f"\nTest configuration:")
    print(f"  Input shape: [{batch_size}, {window_size}, {n_features}]")
    print(f"  Expected output shape: [{batch_size}, 1]")
    print(f"  Expected output range: [0, 1]")
    print()
    
    all_passed = True
    
    for model_type in model_types:
        print(f"Testing {model_type.upper()}...", end=' ')
        
        try:
            # Create model
            model = create_model(model_type, n_features, window_size, dropout=0.3)
            model.eval()
            
            # Test input
            x = torch.randn(batch_size, window_size, n_features)
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Validate output shape
            assert output.shape == (batch_size, 1), \
                f"Expected shape ({batch_size}, 1), got {output.shape}"
            
            # Validate output range
            assert (output >= 0).all() and (output <= 1).all(), \
                f"Output not in [0, 1] range. Min: {output.min():.4f}, Max: {output.max():.4f}"
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            
            print(f"✓ PASSED (params: {num_params:,})")
            
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            all_passed = False
    
    print()
    print("="*80)
    if all_passed:
        print("✓ All models passed interface validation!")
    else:
        print("✗ Some models failed validation. Please fix the errors above.")
    print("="*80)
    
    return all_passed


if __name__ == '__main__':
    success = test_model_interface()
    sys.exit(0 if success else 1)
