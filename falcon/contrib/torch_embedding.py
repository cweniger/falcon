"""
FALCON Embedding Infrastructure - Simplified Sequential Execution

This module provides a declarative way to build complex neural network pipelines using
configuration dictionaries. The system flattens nested configurations into a linear
sequence of modules that execute sequentially with a shared data dictionary.

## Key Features

1. **Sequential Execution**: Modules run in a flat list, easy to debug and understand
2. **Shared Data Dictionary**: One dictionary stores all intermediate results
3. **Declarative Configuration**: Define complex pipelines using simple YAML/dict configs
4. **Automatic Flattening**: Nested configurations are automatically linearized
5. **PyTorch Native**: Full backpropagation support, standard nn.Module behavior
6. **Fault Tolerant**: Missing inputs default to None with warnings

## Basic Usage

```python
from falcon.core.embedding import instantiate_embedding
import torch

# Simple single-input embedding
config = {
    '_target_': 'torch.nn.Linear',
    'in_features': 10,
    'out_features': 5,
    '_input_': 'x'
}

embedding = instantiate_embedding(config)
result = embedding({'x': torch.randn(32, 10)})  # -> torch.Size([32, 5])
```

## Configuration Keywords

- **_target_**: Module class path (e.g., "torch.nn.Linear", "mymodule.MyClass")
- **_input_**: Input specification (string, list, or nested config)
- All other keys become constructor arguments for the target class

## Multi-Input Examples

```python
# Multiple direct inputs
config = {
    '_target_': 'mymodule.Combiner',
    'mode': 'add',
    '_input_': ['x', 'y']  # Module receives: module(x_tensor, y_tensor)
}

embedding = instantiate_embedding(config)
result = embedding({
    'x': torch.randn(32, 10),
    'y': torch.randn(32, 10)
})
```

## Nested Pipeline Examples

```python
# Preprocessing pipeline
config = {
    '_target_': 'mymodule.Classifier',
    'num_classes': 10,
    '_input_': {
        '_target_': 'torch.nn.Linear',
        'in_features': 100,
        'out_features': 50,
        '_input_': 'raw_features'
    }
}

# This creates a linear pipeline:
# raw_features -> Linear(100->50) -> temp_1 -> Classifier(temp_1) -> output
```

## Complex Hierarchical Examples

```python
# Multi-branch processing with combination
config = {
    '_target_': 'mymodule.AttentionCombiner',
    'hidden_dim': 64,
    '_input_': [
        # Branch 1: Direct input
        'global_context',

        # Branch 2: CNN processing
        {
            '_target_': 'mymodule.CNNEncoder',
            'channels': [3, 32, 64],
            '_input_': 'image_data'
        },

        # Branch 3: Multi-stage text processing
        {
            '_target_': 'mymodule.TextClassifier',
            'num_classes': 100,
            '_input_': {
                '_target_': 'transformers.BertModel.from_pretrained',
                'pretrained_model_name_or_path': 'bert-base-uncased',
                '_input_': 'text_tokens'
            }
        }
    ]
}

# This flattens to a 4-module pipeline:
# 1. text_tokens -> BertModel -> temp_1
# 2. image_data -> CNNEncoder -> temp_2
# 3. temp_1 -> TextClassifier -> temp_3
# 4. [global_context, temp_2, temp_3] -> AttentionCombiner -> final_output

embedding = instantiate_embedding(config)
result = embedding({
    'global_context': torch.randn(32, 128),
    'image_data': torch.randn(32, 3, 224, 224),
    'text_tokens': torch.randint(0, 1000, (32, 50))
})
```

## Real-World FALCON Example

```python
# Simulation-based inference embedding for parameter estimation
config = {
    '_target_': 'falcon.contrib.PosteriorNet',
    'hidden_dims': [128, 64],
    'output_dim': 4,  # Parameter dimension
    '_input_': [
        # Raw parameters (for training)
        'theta',

        # Processed observations
        {
            '_target_': 'falcon.contrib.ObservationEncoder',
            'embedding_dim': 64,
            '_input_': {
                '_target_': 'falcon.contrib.PCAProjector',
                'n_components': 32,
                '_input_': {
                    '_target_': 'falcon.contrib.DiagonalWhitener',
                    'normalize': True,
                    '_input_': 'x_obs'
                }
            }
        }
    ]
}

# Creates pipeline: x_obs -> Whitener -> PCA -> Encoder -> [theta, encoded] -> PosteriorNet
embedding = instantiate_embedding(config)

# Training mode: provide both theta and x_obs
train_result = embedding({
    'theta': torch.randn(64, 4),      # True parameters
    'x_obs': torch.randn(64, 1000)   # Observations
})

# Inference mode: provide only x_obs (theta will be None with warning)
inference_result = embedding({
    'x_obs': torch.randn(1, 1000)
})
```

## Debugging and Inspection

The sequential architecture makes debugging trivial:

```python
embedding = instantiate_embedding(config)

# Inspect the pipeline structure
print(f"Pipeline has {len(embedding.modules_list)} modules:")
for i, (module, inputs, output) in enumerate(zip(
    embedding.modules_list,
    embedding.input_keys_list,
    embedding.output_keys
)):
    print(f"  Step {i+1}: {module.__class__.__name__}")
    print(f"    Inputs: {inputs}")
    print(f"    Output: {output}")
    print(f"    Parameters: {sum(p.numel() for p in module.parameters())}")

# Required inputs for the entire pipeline
print(f"Required inputs: {embedding.input_keys}")
```

## Error Handling

Missing inputs are handled gracefully:

```python
# If 'y' is missing, module receives None and a warning is issued
result = embedding({'x': tensor})  # Warning: Missing input keys: {'y'}, will use None values
```

## PyTorch Integration

The system is fully compatible with PyTorch training:

```python
# Standard PyTorch training loop
embedding = instantiate_embedding(config)
optimizer = torch.optim.Adam(embedding.parameters())

for batch in dataloader:
    optimizer.zero_grad()
    output = embedding(batch)
    loss = criterion(output, targets)
    loss.backward()  # Gradients flow through entire pipeline
    optimizer.step()

# Save/load like any PyTorch model
torch.save(embedding.state_dict(), 'embedding.pth')
embedding.load_state_dict(torch.load('embedding.pth'))
```

## Implementation Notes

- Modules are stored in `nn.ModuleList` for proper parameter registration
- Temporary variables use names like `temp_1`, `temp_2`, etc.
- The final output is always the result of the last module in the sequence
- All intermediate results are kept in the working dictionary during execution
- The system preserves the computational graph for backpropagation

This architecture provides the perfect balance of flexibility, simplicity, and debuggability
for building complex neural network pipelines declaratively.
"""

import importlib
import warnings
from typing import Dict, Any, List, Union, Tuple
import torch.nn as nn


class EmbeddingWrapper(nn.Module):
    """Sequential execution of modules with shared data dictionary."""

    def __init__(
        self,
        modules: List[nn.Module],
        input_keys_list: List[List[str]],
        output_keys: List[str],
        required_input_keys: List[str],
    ):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.input_keys_list = input_keys_list
        self.output_keys = output_keys
        self.input_keys = required_input_keys

    def forward(self, data_dict: Dict[str, Any]) -> Any:
        missing_keys = set(self.input_keys) - set(data_dict.keys())
        if missing_keys:
            warnings.warn(
                f"Missing input keys: {missing_keys}, will use None values", UserWarning
            )

        work_dict = data_dict.copy()

        for module, input_keys, output_key in zip(
            self.modules_list, self.input_keys_list, self.output_keys
        ):
            module_inputs = [work_dict.get(key) for key in input_keys]
            output = module(*module_inputs)
            work_dict[output_key] = output

        return work_dict[self.output_keys[-1]]


def _collect_input_keys(config: Union[Dict, str, List]) -> List[str]:
    keys = []
    if isinstance(config, str):
        keys.append(config)
    elif isinstance(config, list):
        for item in config:
            keys.extend(_collect_input_keys(item))
    elif isinstance(config, dict) and "_input_" in config:
        keys.extend(_collect_input_keys(config["_input_"]))
    return list(set(keys))


def _flatten_config_to_modules(
    config: Dict[str, Any], temp_counter: int = 0
) -> Tuple[List[nn.Module], List[List[str]], List[str], int]:
    modules = []
    input_keys_list = []
    output_keys = []

    target = config["_target_"]
    cls = target
    if isinstance(target, str):
        module_name, class_name = target.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)

    input_config = config["_input_"]
    kwargs = {k: v for k, v in config.items() if k not in ["_target_", "_input_"]}

    if isinstance(input_config, str):
        input_keys_for_module = [input_config]
    elif isinstance(input_config, list):
        input_keys_for_module = []
        for item in input_config:
            if isinstance(item, str):
                input_keys_for_module.append(item)
            elif isinstance(item, dict) and "_target_" in item:
                (
                    nested_modules,
                    nested_input_keys_list,
                    nested_output_keys,
                    temp_counter,
                ) = _flatten_config_to_modules(item, temp_counter)
                modules.extend(nested_modules)
                input_keys_list.extend(nested_input_keys_list)
                output_keys.extend(nested_output_keys)
                input_keys_for_module.append(nested_output_keys[-1])
    elif isinstance(input_config, dict):
        nested_modules, nested_input_keys_list, nested_output_keys, temp_counter = (
            _flatten_config_to_modules(input_config, temp_counter)
        )
        modules.extend(nested_modules)
        input_keys_list.extend(nested_input_keys_list)
        output_keys.extend(nested_output_keys)
        input_keys_for_module = [nested_output_keys[-1]]

    instance = cls(**kwargs)
    temp_counter += 1
    output_key = f"temp_{temp_counter}"
    modules.append(instance)
    input_keys_list.append(input_keys_for_module)
    output_keys.append(output_key)
    return modules, input_keys_list, output_keys, temp_counter


def instantiate_embedding(embedding_config: Dict[str, Any]) -> EmbeddingWrapper:
    """Instantiate embedding pipeline from config."""
    required_input_keys = _collect_input_keys(embedding_config)
    modules, input_keys_list, output_keys, _ = _flatten_config_to_modules(
        embedding_config
    )
    return EmbeddingWrapper(modules, input_keys_list, output_keys, required_input_keys)
