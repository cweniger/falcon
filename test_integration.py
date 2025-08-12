#!/usr/bin/env python3
"""
Integration test for the new embedding infrastructure with FALCON.
"""

import torch
import torch.nn as nn
from falcon.core.embedding import instantiate_embedding


def test_simple_integration():
    """Test that the new embedding system creates standard torch modules."""
    print("Testing simple integration with torch.nn.Linear...")
    
    config = {
        '_target_': 'torch.nn.Linear',
        'in_features': 10,
        'out_features': 5,
        '_input_': 'x'
    }
    
    embedding = instantiate_embedding(config)
    
    # Verify it's a proper torch module
    assert isinstance(embedding, nn.Module)
    assert hasattr(embedding, 'parameters')
    assert hasattr(embedding, 'state_dict')
    
    # Test forward pass
    data = {'x': torch.randn(32, 10)}
    result = embedding(data)
    
    assert result.shape == (32, 5)
    print("✓ Simple integration test passed")


class SimpleE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x


def test_falcon_style_embedding():
    """Test creating an embedding similar to FALCON's existing E class."""
    print("Testing FALCON-style embedding...")
    
    config = {
        '_target_': 'test_integration.SimpleE',
        'input_dim': 100,
        'output_dim': 8,
        '_input_': 'observations'
    }
    
    embedding = instantiate_embedding(config)
    
    # Test with data
    data = {'observations': torch.randn(16, 100)}
    result = embedding(data)
    
    assert result.shape == (16, 8)
    print("✓ FALCON-style embedding test passed")


class CombineEmbedding(nn.Module):
    def __init__(self, dim1: int, dim2: int, output_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim1, output_dim // 2)
        self.linear2 = nn.Linear(dim2, output_dim // 2)
    
    def forward(self, theta, x_obs):
        # Process parameters and observations separately, then combine
        theta_embed = self.linear1(theta)
        obs_embed = self.linear2(x_obs)
        return torch.cat([theta_embed, obs_embed], dim=-1)


def test_multi_input_like_snpe():
    """Test multi-input embedding similar to what SNPE_A might use."""
    print("Testing multi-input embedding like SNPE...")
    
    config = {
        '_target_': 'test_integration.CombineEmbedding',
        'dim1': 4,  # parameter dimension
        'dim2': 10, # observation dimension  
        'output_dim': 8,
        '_input_': ['theta', 'x_obs']
    }
    
    embedding = instantiate_embedding(config)
    
    # Test input keys
    assert set(embedding.input_keys) == {'theta', 'x_obs'}
    
    # Test forward pass
    data = {
        'theta': torch.randn(32, 4),
        'x_obs': torch.randn(32, 10)
    }
    result = embedding(data)
    
    assert result.shape == (32, 8)
    print("✓ Multi-input embedding test passed")


class Preprocessor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(self.norm(x))


class MainNet(nn.Module):
    def __init__(self, param_dim: int, obs_dim: int, output_dim: int):
        super().__init__()
        self.combine = nn.Linear(param_dim + obs_dim, output_dim)
    
    def forward(self, params, processed_obs):
        combined = torch.cat([params, processed_obs], dim=-1)
        return self.combine(combined)


def test_nested_preprocessing():
    """Test nested preprocessing similar to complex FALCON pipelines."""
    print("Testing nested preprocessing...")
    
    config = {
        '_target_': 'test_integration.MainNet',
        'param_dim': 4,
        'obs_dim': 6,
        'output_dim': 8,
        '_input_': [
            'params',
            {
                '_target_': 'test_integration.Preprocessor',
                'input_dim': 20,
                'output_dim': 6,
                '_input_': 'raw_observations'
            }
        ]
    }
    
    embedding = instantiate_embedding(config)
    
    # Test input keys
    assert set(embedding.input_keys) == {'params', 'raw_observations'}
    
    # Test forward pass
    data = {
        'params': torch.randn(16, 4),
        'raw_observations': torch.randn(16, 20)
    }
    result = embedding(data)
    
    assert result.shape == (16, 8)
    print("✓ Nested preprocessing test passed")


if __name__ == "__main__":
    print("Running FALCON integration tests...")
    
    test_simple_integration()
    test_falcon_style_embedding()
    test_multi_input_like_snpe()
    test_nested_preprocessing()
    
    print("\n✅ All integration tests passed!")
    print("🎯 The new embedding system integrates well with FALCON patterns")