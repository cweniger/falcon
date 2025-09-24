#!/usr/bin/env python3
"""
Falcon Adaptive Training - Standalone CLI Tool
Usage: python adaptive.py --config-path ./model-repo --config-name config
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import ray

import hydra
from omegaconf import DictConfig, OmegaConf
import sbi.analysis

import falcon
from falcon.core.utils import load_observations
from falcon.core.graph import create_graph_from_config


def parse_operational_flags():
    """Extract all operational flags (--*) from sys.argv, leaving Hydra args."""

    # Extract all operational flags (anything starting with "--")
    operational_flags = [arg for arg in sys.argv if arg.startswith("--")]
    hydra_args = [arg for arg in sys.argv if not arg.startswith("--")]

    # Update sys.argv to only contain hydra arguments
    sys.argv[:] = hydra_args

    # Helper to parse flag values
    def get_flag_value(flag_name):
        for flag in operational_flags:
            if flag.startswith(f"{flag_name}="):
                return flag.split("=", 1)[1]
        return None

    # Helper to check if flag exists
    def has_flag(flag_name):
        return flag_name in operational_flags

    return operational_flags, get_flag_value, has_flag


def launch_mode(cfg: DictConfig) -> None:
    """Launch mode: Full training and inference pipeline."""
    ray_init_args = cfg.get("ray", {}).get("init", {})
    ray.init(**ray_init_args)

    # Add model path to Python path for imports
    if cfg.model_path:
        model_path = Path(cfg.model_path).resolve()
        if model_path not in sys.path:
            sys.path.insert(0, str(model_path))

    # Initialise logger (should be done before any other falcon code)
    wandb_dir = cfg.logging.get("dir", None)
    falcon.start_wandb_logger(
        wandb_project=cfg.logging.project,
        wandb_group=cfg.logging.group,
        wandb_dir=wandb_dir,
    )

    ########################
    ### Model definition ###
    ########################

    # Instantiate model components directly from graph
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # TODO: This is some hack right now
    observations = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()
    }

    print(graph)
    print("Observation shapes:", {k: v.shape for k, v in observations.items()})

    ####################
    ### Run analysis ###
    ####################

    # 1) Deploy graph
    deployed_graph = falcon.DeployedGraph(graph, model_path=cfg.get("model_path"))

    # 2) Prepare dataset manager for deployed graph and store initial samples
    dataset_manager = falcon.get_ray_dataset_manager(
        min_training_samples=cfg.buffer.min_training_samples,
        max_training_samples=cfg.buffer.max_training_samples,
        validation_window_size=cfg.buffer.validation_window_size,
        resample_batch_size=cfg.buffer.resample_batch_size,
        resample_interval=cfg.buffer.resample_interval,
        keep_resampling=cfg.buffer.keep_resampling,
        initial_samples_path=cfg.buffer.get("initial_samples_path", None),
    )

    # 3) Launch training & simulations
    graph_path = Path(cfg.paths.graph)
    deployed_graph.launch(dataset_manager, observations, graph_path=graph_path)

    ##########################
    ### Clean up resources ###
    ##########################

    deployed_graph.shutdown()
    falcon.finish_wandb_logger()


def sample_mode(cfg: DictConfig, sample_type: str) -> None:
    """Sample mode: Generate samples using different sampling strategies."""
    ray_init_args = cfg.get("ray", {}).get("init", {})
    ray.init(**ray_init_args)

    # Add model path to Python path for imports
    if cfg.model_path:
        model_path = Path(cfg.model_path).resolve()
        if model_path not in sys.path:
            sys.path.insert(0, str(model_path))

    # Instantiate model components directly from graph
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # TODO: This is some hack right now
    observations = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()
    }

    if sample_type == "prior":
        sample_cfg = cfg.sample.prior
    elif sample_type == "posterior":
        sample_cfg = cfg.sample.posterior
    elif sample_type == "proposal":
        sample_cfg = cfg.sample.proposal
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    num_samples = sample_cfg.get("n", 42)
    print(f"Generating {num_samples} samples using {sample_type} sampling...")
    print(graph)

    # Deploy graph for sampling
    deployed_graph = falcon.DeployedGraph(graph, model_path=cfg.get("model_path"))

    if sample_type == "prior":
        # Generate forward samples from prior
        samples = deployed_graph.sample(num_samples)

    elif sample_type == "posterior":
        # TODO: Implement posterior sampling (requires trained model and observations)
        deployed_graph.load(Path(cfg.paths.graph))
        samples = deployed_graph.conditioned_sample(num_samples, observations)

    elif sample_type == "proposal":
        # Proposal sampling requires observations for conditioning
        # Load observations from config
        deployed_graph.load(Path(cfg.paths.graph))
        samples = deployed_graph.proposal_sample(num_samples, observations)

    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    # Apply smart key selection based on mode and user overrides
    if sample_type in ["prior", "proposal"]:
        # Default: save everything
        default_keys = set(samples.keys())
    elif sample_type == "posterior":
        # Default: save only posterior nodes (nodes with evidence)
        default_keys = {
            k for k, node in graph.node_dict.items() if node.evidence and k in samples
        }

    # Apply user overrides
    exclude_keys = sample_cfg.get("exclude_keys", None)
    add_keys = sample_cfg.get("add_keys", None)

    if exclude_keys:
        exclude_set = set(exclude_keys.split(","))
        default_keys -= exclude_set

    if add_keys:
        add_set = set(add_keys.split(","))
        default_keys |= add_set

    # Filter samples to selected keys
    save_data = {k: samples[k] for k in default_keys if k in samples}

    print(f"Generated samples with shapes:")
    for key, value in save_data.items():
        print(f"  {key}: {value.shape}")

    # Save to NPZ file
    output_path = sample_cfg.path
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if it's not empty
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving samples to: {output_path}")
    save_data_reversed = []
    num_samples = len(next(iter(save_data.values())))
    for i in range(num_samples):
        save_data_reversed.append({k: v[i] for k, v in save_data.items()})
    joblib.dump(save_data_reversed, output_path)

    print(f"Saved {sample_type} samples to: {output_path}")

    # Clean up
    deployed_graph.shutdown()


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def launch_main(cfg: DictConfig) -> None:
    """Launch mode entry point."""
    launch_mode(cfg)


def main():
    """Main CLI entry point with explicit mode dispatch."""

    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch"]:
        print("Error: Must specify mode. Usage:")
        print("  falcon launch [hydra_options...]")
        print("  falcon sample prior [hydra_options...]")
        print("  falcon sample proposal [hydra_options...]")
        print("  falcon sample posterior [hydra_options...]")
        sys.exit(1)

    mode = sys.argv.pop(1)  # Remove mode from sys.argv

    if mode == "sample":
        sample_type = sys.argv.pop(1)
        if sample_type not in ["prior", "proposal", "posterior"]:
            print(f"Error: Unknown sample type: {sample_type}")
            sys.exit(1)

        # Create a modified main function that injects operational parameters
        def make_sample_main(sample_type):
            @hydra.main(
                version_base=None, config_path=os.getcwd(), config_name="config"
            )
            def sample_main_with_params(cfg: DictConfig) -> None:
                sample_mode(cfg, sample_type)

            return sample_main_with_params

        # Create and call the sample function
        sample_main_func = make_sample_main(sample_type)
        sample_main_func()
    else:  # mode == 'launch'
        launch_main()


if __name__ == "__main__":
    main()
