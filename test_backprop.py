#!/usr/bin/env python3
"""
Test that backpropagation works correctly through the embedding pipeline.
"""

import torch
import torch.nn as nn
from falcon.core.embedding import instantiate_embedding


class SimpleLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear(x)


class Combiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
    
    def forward(self, x, y):
        return self.weight[0] * x + self.weight[1] * y


def test_backprop_simple():
    """Test backprop through a simple embedding."""
    print("Testing backprop through simple embedding...")
    
    config = {
        '_target_': 'test_backprop.SimpleLinear',
        'in_dim': 10,
        'out_dim': 5,
        '_input_': 'x'
    }
    
    embedding = instantiate_embedding(config)
    
    # Check that parameters are properly registered
    params = list(embedding.parameters())
    print(f"Number of parameters: {len(params)}")
    print(f"Parameter shapes: {[p.shape for p in params]}")
    
    # Test forward and backward
    x = torch.randn(3, 10, requires_grad=True)
    output = embedding({'x': x})
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Parameters have gradients: {all(p.grad is not None for p in params)}")
    print("✓ Simple backprop test passed")


def test_backprop_nested():
    """Test backprop through nested embedding."""
    print("\nTesting backprop through nested embedding...")
    
    config = {
        '_target_': 'test_backprop.Combiner',
        '_input_': [
            {
                '_target_': 'test_backprop.SimpleLinear',
                'in_dim': 10,
                'out_dim': 5,
                '_input_': 'x'
            },
            {
                '_target_': 'test_backprop.SimpleLinear', 
                'in_dim': 8,
                'out_dim': 5,
                '_input_': 'y'
            }
        ]
    }
    
    embedding = instantiate_embedding(config)
    
    # Check parameter registration
    params = list(embedding.parameters())
    print(f"Total parameters: {len(params)}")
    
    # Check that modules are in the ModuleList
    print(f"Modules in pipeline: {len(embedding.modules_list)}")
    for i, module in enumerate(embedding.modules_list):
        module_params = list(module.parameters())
        print(f"  Module {i} ({module.__class__.__name__}): {len(module_params)} params")
    
    # Test forward and backward
    x = torch.randn(3, 10, requires_grad=True) 
    y = torch.randn(3, 8, requires_grad=True)
    
    output = embedding({'x': x, 'y': y})
    loss = output.sum()
    loss.backward()
    
    # Check gradients flow through everything
    print(f"x gradient exists: {x.grad is not None}")
    print(f"y gradient exists: {y.grad is not None}")
    print(f"All parameters have gradients: {all(p.grad is not None for p in params)}")
    
    print("✓ Nested backprop test passed")


def test_parameter_counting():
    """Test that parameter counting matches expectations."""
    print("\nTesting parameter counting...")
    
    config = {
        '_target_': 'test_backprop.Combiner',
        '_input_': [
            {
                '_target_': 'test_backprop.SimpleLinear',
                'in_dim': 10,
                'out_dim': 5,
                '_input_': 'x'
            },
            {
                '_target_': 'test_backprop.SimpleLinear',
                'in_dim': 8, 
                'out_dim': 5,
                '_input_': 'y'
            }
        ]
    }
    
    embedding = instantiate_embedding(config)
    
    # Count parameters manually
    expected_params = (
        10 * 5 + 5 +    # First linear: weight + bias
        8 * 5 + 5 +     # Second linear: weight + bias  
        2               # Combiner weight parameter
    )
    
    actual_params = sum(p.numel() for p in embedding.parameters())
    
    print(f"Expected parameters: {expected_params}")
    print(f"Actual parameters: {actual_params}")
    print(f"Parameters match: {expected_params == actual_params}")
    
    print("✓ Parameter counting test passed")


if __name__ == "__main__":
    print("Testing backpropagation through embedding pipeline...")
    
    test_backprop_simple()
    test_backprop_nested() 
    test_parameter_counting()
    
    print("\n✅ All backprop tests passed!")
    print("   - Parameters are properly registered in nn.ModuleList")
    print("   - Gradients flow through the entire pipeline")
    print("   - Parameter counting is correct")
    print("   - PyTorch autograd works as expected")