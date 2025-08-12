#!/usr/bin/env python3
"""
Test script for the new embedding infrastructure.
"""

import torch
import torch.nn as nn
from falcon.core.embedding import instantiate_embedding


class TestLinear(nn.Module):
    """Simple test module for linear transformation."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class TestCombiner(nn.Module):
    """Simple test module that combines inputs."""
    def __init__(self, mode: str = "add"):
        super().__init__()
        self.mode = mode
    
    def forward(self, x, y):
        if self.mode == "add":
            return x + y
        elif self.mode == "concatenate":
            return torch.cat([x, y], dim=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def test_simple_embedding():
    """Test simple single-input embedding."""
    print("Testing simple embedding...")
    
    config = {
        '_target_': 'test_embedding.TestLinear',
        'input_dim': 10,
        'output_dim': 5,
        '_input_': 'x'
    }
    
    embedding = instantiate_embedding(config)
    
    # Test input keys
    assert embedding.input_keys == ['x']
    
    # Test forward pass
    data = {'x': torch.randn(32, 10)}
    result = embedding(data)
    
    assert result.shape == (32, 5)
    print("✓ Simple embedding test passed")


def test_multi_input_embedding():
    """Test multi-input embedding."""
    print("Testing multi-input embedding...")
    
    config = {
        '_target_': 'test_embedding.TestCombiner',
        'mode': 'add',
        '_input_': ['x', 'y']
    }
    
    embedding = instantiate_embedding(config)
    
    # Test input keys
    assert set(embedding.input_keys) == {'x', 'y'}
    
    # Test forward pass
    data = {
        'x': torch.randn(32, 10),
        'y': torch.randn(32, 10)
    }
    result = embedding(data)
    
    assert result.shape == (32, 10)
    print("✓ Multi-input embedding test passed")


def test_nested_embedding():
    """Test nested embedding configuration."""
    print("Testing nested embedding...")
    
    config = {
        '_target_': 'test_embedding.TestCombiner',
        'mode': 'add',
        '_input_': [
            'x',
            {
                '_target_': 'test_embedding.TestLinear',
                'input_dim': 10,
                'output_dim': 10,
                '_input_': 'y'
            }
        ]
    }
    
    embedding = instantiate_embedding(config)
    
    # Test input keys
    assert set(embedding.input_keys) == {'x', 'y'}
    
    # Test forward pass
    data = {
        'x': torch.randn(32, 10),
        'y': torch.randn(32, 10)
    }
    result = embedding(data)
    
    assert result.shape == (32, 10)
    print("✓ Nested embedding test passed")


def test_complex_nested_embedding():
    """Test complex hierarchical embedding."""
    print("Testing complex nested embedding...")
    
    config = {
        '_target_': 'test_embedding.TestCombiner',
        'mode': 'add',
        '_input_': [
            {
                '_target_': 'test_embedding.TestLinear',
                'input_dim': 10,
                'output_dim': 5,
                '_input_': 'x'
            },
            {
                '_target_': 'test_embedding.TestLinear',
                'input_dim': 10,
                'output_dim': 5,
                '_input_': {
                    '_target_': 'test_embedding.TestLinear',
                    'input_dim': 15,
                    'output_dim': 10,
                    '_input_': 'y'
                }
            }
        ]
    }
    
    embedding = instantiate_embedding(config)
    
    # Test input keys
    assert set(embedding.input_keys) == {'x', 'y'}
    
    # Test forward pass
    data = {
        'x': torch.randn(32, 10),
        'y': torch.randn(32, 15)
    }
    result = embedding(data)
    
    assert result.shape == (32, 5)
    print("✓ Complex nested embedding test passed")


if __name__ == "__main__":
    print("Running embedding infrastructure tests...")
    
    test_simple_embedding()
    test_multi_input_embedding()
    test_nested_embedding()
    test_complex_nested_embedding()
    
    print("\n✅ All tests passed! The embedding infrastructure is working correctly.")