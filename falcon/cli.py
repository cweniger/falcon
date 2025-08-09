#!/usr/bin/env python3
"""
Falcon Adaptive Training - Standalone CLI Tool
Usage: python adaptive.py --config-path ./model-repo --config-name config
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

import torch
import hydra
from omegaconf import DictConfig

import sbi.analysis
import falcon


def parse_operational_flags():
    """Extract all operational flags (--*) from sys.argv, leaving Hydra args."""
    import sys
    
    # Extract all operational flags (anything starting with "--")
    operational_flags = [arg for arg in sys.argv if arg.startswith('--')]
    hydra_args = [arg for arg in sys.argv if not arg.startswith('--')]
    
    # Update sys.argv to only contain hydra arguments
    sys.argv[:] = hydra_args
    
    # Helper to parse flag values
    def get_flag_value(flag_name):
        for flag in operational_flags:
            if flag.startswith(f'{flag_name}='):
                return flag.split('=', 1)[1]
        return None
    
    # Helper to check if flag exists
    def has_flag(flag_name):
        return flag_name in operational_flags
    
    return operational_flags, get_flag_value, has_flag






def launch_mode(cfg: DictConfig) -> None:
    """Launch mode: Full training and inference pipeline."""
    # Add model path to Python path for imports
    if cfg.model_path:
        model_path = Path(cfg.model_path).resolve()
        if model_path not in sys.path:
            sys.path.insert(0, str(model_path))
    
    # Initialise logger (should be done before any other falcon code)
    wandb_dir = cfg.wandb.get('dir', None)
    falcon.start_wandb_logger(wandb_project=cfg.wandb.project, wandb_group=cfg.wandb.group, wandb_dir=wandb_dir)


    ########################
    ### Model definition ###
    ########################

    # Instantiate model components directly from graph
    from falcon.core.graph import create_graph_from_config
    graph = create_graph_from_config(cfg.graph, _cfg=cfg)
    
    # Load observations from file path
    from falcon.core.utils import load_observations
    observations = load_observations(cfg.observations)

    print(graph)
    print("Observation shapes:", {k: v.shape for k, v in observations.items()})

    ####################
    ### Run analysis ###
    ####################

    # 0) Deploy graph
    deployed_graph = falcon.DeployedGraph(graph, model_path=cfg.get('model_path'))

    # 1) Prepare dataset manager for deployed graph and store initial samples

    #shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    #dataset_manager = falcon.get_zarr_dataset_manager(shapes_and_dtypes, Path(cfg.directories.sim_dir),
    #        num_min_sims = cfg.training.n_train, num_max_sims = cfg.training.n_train,
    #        num_val_sims=cfg.training.num_val_samples, num_resims = cfg.training.num_resims)

    dataset_manager = falcon.get_ray_dataset_manager(
            min_training_samples = cfg.training.min_training_samples,
            max_training_samples = cfg.training.max_training_samples,
            validation_window_size=cfg.training.validation_window_size,
            resample_batch_size=cfg.training.resample_batch_size
            )

    # 2) Generate initial samples
    dataset_manager.generate_samples(deployed_graph, num_sims = cfg.training.min_training_samples)

    # 3) Train the graph
    if cfg.runtime.reload:
        deployed_graph.load(Path(cfg.directories.graph_dir))
    else:
        deployed_graph.launch(dataset_manager, observations)
        deployed_graph.save(Path(cfg.directories.graph_dir))

    ##################
    ### Evaluation ###
    ##################

    samples = deployed_graph.conditioned_sample(cfg.training.num_conditional_samples, observations)
    plot_samples = samples['z']
    sbi.analysis.pairplot(plot_samples, figsize=(15, 15))


    ##########################
    ### Clean up resources ###
    ##########################

    deployed_graph.shutdown()
    falcon.finish_wandb_logger()


def sample_mode(cfg: DictConfig, sample_type: str, num_samples: int) -> None:
    """Sample mode: Generate samples using different sampling strategies."""
    # Add model path to Python path for imports
    if cfg.model_path:
        model_path = Path(cfg.model_path).resolve()
        if model_path not in sys.path:
            sys.path.insert(0, str(model_path))
    
    # Instantiate model components directly from graph
    from falcon.core.graph import create_graph_from_config
    graph = create_graph_from_config(cfg.graph, _cfg=cfg)
    
    print(f"Generating {num_samples} samples using {sample_type} sampling...")
    print(graph)
    
    # Deploy graph for sampling
    deployed_graph = falcon.DeployedGraph(graph, model_path=cfg.get('model_path'))
    
    # Load saved models if reload is enabled
    if cfg.runtime.reload:
        deployed_graph.load(Path(cfg.directories.graph_dir))
    
    if sample_type == 'prior':
        # Generate forward samples from prior
        samples = deployed_graph.sample(num_samples)
        
    elif sample_type == 'posterior':
        # TODO: Implement posterior sampling (requires trained model and observations)
        from falcon.core.utils import load_observations
        observations = load_observations(cfg.observations)
        samples = deployed_graph.conditioned_sample(num_samples, observations)
        
    elif sample_type == 'proposal':
        # Proposal sampling requires observations for conditioning
        # Load observations from config
        from falcon.core.utils import load_observations
        observations = load_observations(cfg.observations)
        samples = deployed_graph.proposal_sample(num_samples, observations)
        
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")
    
    # Apply smart key selection based on mode and user overrides
    if sample_type in ['prior', 'proposal']:
        # Default: save everything
        default_keys = set(samples.keys())
    elif sample_type == 'posterior':
        # Default: save only posterior nodes (nodes with evidence)
        default_keys = {k for k, node in graph.node_dict.items() if node.evidence and k in samples}
    
    # Apply user overrides
    exclude_keys = cfg.sample.exclude_keys
    add_keys = cfg.sample.add_keys
    
    if exclude_keys:
        exclude_set = set(exclude_keys.split(','))
        default_keys -= exclude_set
    
    if add_keys:
        add_set = set(add_keys.split(','))
        default_keys |= add_set
    
    # Filter samples to selected keys
    save_data = {k: samples[k] for k in default_keys if k in samples}
    
    print(f"Generated samples with shapes:")
    for key, value in save_data.items():
        print(f"  {key}: {value.shape}")
    
    # Save to NPZ file
    output_path = cfg.sample.output_path
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if it's not empty
        os.makedirs(output_dir, exist_ok=True)
    
    import numpy as np
    np.savez(output_path, **save_data)
    
    print(f"Saved {sample_type} samples to: {output_path}")
    
    # Clean up
    deployed_graph.shutdown()


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def launch_main(cfg: DictConfig) -> None:
    """Launch mode entry point."""
    launch_mode(cfg)




def main():
    """Main CLI entry point with explicit mode dispatch."""
    import sys
    
    if len(sys.argv) < 2 or sys.argv[1] not in ['launch', 'sample']:
        print("Error: Must specify mode. Usage:")
        print("  falcon launch [hydra_options...]")
        print("  falcon sample --prior|--posterior|--proposal --num-samples=N --output=path [hydra_options...]")
        sys.exit(1)
    
    mode = sys.argv.pop(1)  # Remove mode from sys.argv
    
    if mode == 'sample':
        # Parse operational flags
        operational_flags, get_flag_value, has_flag = parse_operational_flags()
        
        # Extract sample-specific parameters
        sample_type = None
        if has_flag('--prior'):
            sample_type = 'prior'
        elif has_flag('--posterior'):
            sample_type = 'posterior'
        elif has_flag('--proposal'):
            sample_type = 'proposal'
            
        num_samples_str = get_flag_value('--num-samples')
        num_samples = int(num_samples_str) if num_samples_str else None
        output_path = get_flag_value('--output')
        exclude_keys = get_flag_value('--exclude-keys')
        add_keys = get_flag_value('--add-keys')
        
        # Validate required parameters
        if sample_type is None:
            print("Error: Must specify sample type: --prior, --posterior, or --proposal")
            sys.exit(1)
        if num_samples is None:
            print("Error: Must specify --num-samples=N")
            sys.exit(1)
        if output_path is None:
            print("Error: Must specify --output=path")
            sys.exit(1)
        
        # Create a modified main function that injects operational parameters
        def make_sample_main(sample_type, num_samples, output_path, exclude_keys, add_keys):
            @hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
            def sample_main_with_params(cfg: DictConfig) -> None:
                # Add operational parameters to cfg under 'sample' namespace
                from omegaconf import OmegaConf
                OmegaConf.set_struct(cfg, False)  # Allow new keys
                cfg.sample = {
                    'type': sample_type,
                    'num_samples': num_samples, 
                    'output_path': output_path,
                    'exclude_keys': exclude_keys,
                    'add_keys': add_keys
                }
                sample_mode(cfg, sample_type, num_samples)
            return sample_main_with_params
        
        # Create and call the sample function
        sample_main_func = make_sample_main(sample_type, num_samples, output_path, exclude_keys, add_keys)
        sample_main_func()
    else:  # mode == 'launch'
        launch_main()


if __name__ == "__main__":
    main()
