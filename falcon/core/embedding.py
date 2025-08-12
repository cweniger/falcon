"""
Embedding instantiation for nested computational graphs.
Handles dynamic instantiation of embedding pipelines from configuration.
"""

import importlib
from typing import Dict, Any, List, Union, Tuple, Optional
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class EmbeddingWrapper(nn.Module):
    """
    A torch.nn.Module that orchestrates a hierarchical embedding pipeline.
    
    Attributes:
        input_keys: List of input keys required by the embedding pipeline
    """
    
    def __init__(self, root_module: nn.Module, input_keys: List[str]):
        """
        Initialize the embedding wrapper.
        
        Args:
            root_module: The root module of the embedding pipeline
            input_keys: List of input keys required by the pipeline
        """
        super().__init__()
        
        self.root_module = root_module
        self.input_keys = input_keys
    
    
    def forward(self, data_dict: Dict[str, Any]) -> Any:
        """
        Forward pass through the embedding pipeline.
        
        Args:
            data_dict: Dictionary with input keys mapped to their values
            
        Returns:
            Output of the embedding pipeline
        """
        # Validate that all required keys are present
        missing_keys = set(self.input_keys) - set(data_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required input keys: {missing_keys}")
        
        # Execute the root module
        return self._execute_module(self.root_module, data_dict)
    
    def _execute_module(self, module: nn.Module, data_dict: Dict[str, Any]) -> Any:
        """
        Execute a module with proper input handling.
        
        Args:
            module: The module to execute
            data_dict: Available data dictionary
            
        Returns:
            Module output
        """
        if hasattr(module, '_embedding_input_spec'):
            # This module has special input handling
            input_spec = module._embedding_input_spec
            
            if isinstance(input_spec, str):
                # Single input key
                return module(data_dict[input_spec])
            elif isinstance(input_spec, list):
                # Multiple inputs
                inputs = []
                for spec in input_spec:
                    if isinstance(spec, str):
                        inputs.append(data_dict[spec])
                    elif hasattr(spec, '__call__'):
                        # It's a nested module
                        inputs.append(self._execute_module(spec, data_dict))
                    else:
                        raise ValueError(f"Invalid input spec: {spec}")
                
                return module(*inputs)
            elif hasattr(input_spec, '__call__'):
                # Single nested module
                nested_output = self._execute_module(input_spec, data_dict)
                return module(nested_output)
            else:
                raise ValueError(f"Invalid input specification: {input_spec}")
        else:
            raise ValueError(f"Module {module} has no _embedding_input_spec. All modules must have input specifications.")


def _collect_input_keys(config: Union[Dict, str, List]) -> List[str]:
    """
    Recursively collect all input keys from a nested configuration.
    
    Args:
        config: Configuration dictionary, string, or list
        
    Returns:
        List of unique input keys found in the configuration
    """
    keys = []
    
    if isinstance(config, str):
        # If it's a string, treat it as a direct input key
        keys.append(config)
    elif isinstance(config, list):
        # Handle list of inputs
        for item in config:
            keys.extend(_collect_input_keys(item))
    elif isinstance(config, dict):
        # Look for '_input_' field
        if '_input_' in config:
            keys.extend(_collect_input_keys(config['_input_']))
        
        # Recursively check for nested _target_ configurations
        for key, value in config.items():
            if key not in ['_target_', '_input_'] and isinstance(value, (dict, list)):
                if isinstance(value, dict) and '_target_' in value:
                    keys.extend(_collect_input_keys(value))
    
    return list(set(keys))  # Remove duplicates


def _create_module_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a module instance from configuration with proper input handling.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Module instance with embedded input specification
    """
    target = config.get('_target_')
    if not target:
        raise ValueError(f"Config must have '_target_': {config}")
    
    # Get the class
    if isinstance(target, str):
        module_name_parts, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_name_parts)
        cls = getattr(module, class_name)
    else:
        cls = target
    
    # Get input configuration
    input_config = config.get('_input_')
    
    # Prepare kwargs (excluding reserved keywords)
    kwargs = {}
    reserved_keys = ['_target_', '_input_']
    for key, value in config.items():
        if key not in reserved_keys:
            kwargs[key] = value
    
    # Instantiate the module
    instance = cls(**kwargs)
    
    # Attach input specification to the module
    if input_config is not None:
        if isinstance(input_config, str):
            # Single direct input key
            instance._embedding_input_spec = input_config
        elif isinstance(input_config, dict):
            # Nested configuration
            nested_module = _create_module_from_config(input_config)
            instance._embedding_input_spec = nested_module
        elif isinstance(input_config, list):
            # List of mixed inputs (strings and nested configs)
            processed_inputs = []
            for item in input_config:
                if isinstance(item, str):
                    processed_inputs.append(item)
                elif isinstance(item, dict) and '_target_' in item:
                    processed_inputs.append(_create_module_from_config(item))
                else:
                    raise ValueError(f"Invalid input item: {item}")
            instance._embedding_input_spec = processed_inputs
    
    return instance


def instantiate_embedding(embedding_config: Dict[str, Any]) -> EmbeddingWrapper:
    """
    Instantiate an embedding pipeline from configuration.
    
    Args:
        embedding_config: Configuration dictionary for the embedding
        
    Returns:
        EmbeddingWrapper instance with hierarchical execution
        
    Example:
        >>> config = {
        ...     '_target_': 'torch.nn.Linear',
        ...     'in_features': 10,
        ...     'out_features': 5,
        ...     '_input_': 'x'
        ... }
        >>> embedding = instantiate_embedding(config)
        >>> result = embedding({'x': torch.randn(32, 10)})
        
        >>> # Complex nested example
        >>> config = {
        ...     '_target_': 'module.combine',
        ...     'mode': 'add',
        ...     '_input_': [
        ...         'x',
        ...         {
        ...             '_target_': 'module.CNN',
        ...             'layers': 3,
        ...             '_input_': 'y'
        ...         }
        ...     ]
        ... }
        >>> embedding = instantiate_embedding(config)
        >>> result = embedding({'x': tensor1, 'y': tensor2})
    """
    # Convert OmegaConf to regular dict if needed
    if hasattr(embedding_config, '_content'):
        embedding_config = OmegaConf.to_container(embedding_config, resolve=True)
    
    # Collect all input keys from the configuration
    input_keys = _collect_input_keys(embedding_config)
    
    # Create the root module with embedded input specifications
    root_module = _create_module_from_config(embedding_config)
    
    # Create the wrapper
    return EmbeddingWrapper(root_module, input_keys)


