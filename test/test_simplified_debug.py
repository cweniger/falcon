#!/usr/bin/env python3
"""
Demonstrate the new simplified embedding architecture and its debugging capabilities.
"""

import torch
import torch.nn as nn
from falcon.contrib.torch_embedding import instantiate_embedding


class Linear1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x):
        print(f"Linear1 input shape: {x.shape}")
        result = self.linear(x)
        print(f"Linear1 output shape: {result.shape}")
        return result


class Linear2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x):
        print(f"Linear2 input shape: {x.shape}")
        result = self.linear(x)
        print(f"Linear2 output shape: {result.shape}")
        return result


class Combiner(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        print(f"Combiner input shapes: x={x.shape}, y={y.shape}")
        result = x + y
        print(f"Combiner output shape: {result.shape}")
        return result


def test_debug_complex_pipeline():
    """Test a complex pipeline and show how easy it is to debug."""
    print("=== Testing Complex Pipeline with Debug Output ===")
    
    config = {
        '_target_': 'test_simplified_debug.Combiner',
        '_input_': [
            {
                '_target_': 'test_simplified_debug.Linear1',
                'dim': 10,
                '_input_': 'x'
            },
            {
                '_target_': 'test_simplified_debug.Linear2', 
                'dim': 10,
                '_input_': 'y'
            }
        ]
    }
    
    embedding = instantiate_embedding(config)
    
    # Show the internal structure (easy to debug!)
    print(f"\nPipeline structure:")
    print(f"  Number of modules: {len(embedding.modules_list)}")
    print(f"  Input keys for each module: {embedding.input_keys_list}")
    print(f"  Output keys: {embedding.output_keys}")
    print(f"  Required input keys: {embedding.input_keys}")
    
    # Run the pipeline
    print(f"\nExecuting pipeline:")
    data = {
        'x': torch.randn(5, 10),
        'y': torch.randn(5, 10)
    }
    
    result = embedding(data)
    print(f"Final result shape: {result.shape}")
    
    return embedding


def inspect_embedding_internals(embedding):
    """Show how easy it is to inspect and debug the embedding structure."""
    print("\n=== Embedding Internal Structure ===")
    
    for i, (module, input_keys, output_key) in enumerate(
        zip(embedding.modules_list, embedding.input_keys_list, embedding.output_keys)
    ):
        print(f"Step {i+1}:")
        print(f"  Module: {module.__class__.__name__}")
        print(f"  Inputs needed: {input_keys}")
        print(f"  Output stored as: '{output_key}'")
        print(f"  Module parameters: {sum(p.numel() for p in module.parameters())}")
        print()


if __name__ == "__main__":
    print("Demonstrating the simplified embedding architecture...")
    
    # Test the complex pipeline
    embedding = test_debug_complex_pipeline()
    
    # Show how easy it is to inspect
    inspect_embedding_internals(embedding)
    
    print("✅ The new simplified architecture makes debugging much easier!")
    print("   - Linear execution through a module list")
    print("   - Clear input/output specifications for each step") 
    print("   - Easy to trace data flow through the pipeline")
    print("   - Simple to inspect module parameters and structure")