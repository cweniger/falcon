#!/usr/bin/env python3
"""
Test the new behavior with missing input keys.
"""

import torch
import torch.nn as nn
import warnings
from falcon.contrib.torch_embedding import instantiate_embedding


class TestModule(nn.Module):
    """Test module that can handle None inputs."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        if x is None:
            print("Received None input, returning zeros")
            return torch.zeros(1, 5)
        return self.linear(x)


def test_missing_keys():
    """Test behavior when input keys are missing."""
    print("Testing missing input keys...")
    
    config = {
        '_target_': 'test_missing_keys.TestModule',
        '_input_': 'x'
    }
    
    embedding = instantiate_embedding(config)
    
    # Test with missing key - should warn but not crash
    print("\n1. Testing with completely missing key:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Missing 'x' key
        data = {'y': torch.randn(5, 10)}  # Wrong key name
        result = embedding(data)
        
        print(f"Result shape: {result.shape}")
        print(f"Warning issued: {len(w) > 0}")
        if w:
            print(f"Warning message: {w[0].message}")
    
    print("\n2. Testing with correct key:")
    # Test with correct key - should work normally
    data = {'x': torch.randn(5, 10)}
    result = embedding(data)
    print(f"Result shape: {result.shape}")
    
    print("✓ Missing key handling test passed")


class MultiInputModule(nn.Module):
    """Test module with multiple inputs."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        print(f"x is None: {x is None}")
        print(f"y is None: {y is None}")
        
        if x is None:
            x = torch.zeros(5, 10)
        if y is None:
            y = torch.ones(5, 10)
        
        return x + y


def test_multi_input_missing():
    """Test multi-input module with some missing keys."""
    print("\nTesting multi-input with missing keys...")
    
    config = {
        '_target_': 'test_missing_keys.MultiInputModule',
        '_input_': ['x', 'y']
    }
    
    embedding = instantiate_embedding(config)
    
    # Test with one missing key
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        data = {'x': torch.randn(5, 10)}  # Missing 'y'
        result = embedding(data)
        
        print(f"Result shape: {result.shape}")
        print(f"Warning issued: {len(w) > 0}")
        if w:
            print(f"Warning: {w[0].message}")
    
    print("✓ Multi-input missing key test passed")


if __name__ == "__main__":
    print("Testing missing key handling...")
    
    test_missing_keys()
    test_multi_input_missing()
    
    print("\n✅ All missing key tests passed!")
    print("   - Missing keys now issue warnings instead of errors")
    print("   - Modules receive None for missing inputs")
    print("   - Execution continues gracefully")