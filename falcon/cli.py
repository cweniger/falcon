#!/usr/bin/env python3
"""
Falcon Adaptive Training - Standalone CLI Tool
Usage: falcon launch [--run-dir DIR] [--config-name FILE] [key=value ...]
       falcon sample prior|posterior|proposal [--run-dir DIR] [--config-name FILE] [key=value ...]

Run directory behavior:
  - If --run-dir not specified, generates: outputs/adj-noun-YYMMDD-HHMM
  - If --run-dir exists with config.yaml, resumes from saved config
  - Otherwise, loads ./config.yaml and saves resolved config to run_dir
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import joblib
import torch
import ray

from omegaconf import OmegaConf, DictConfig

import falcon
from falcon.core.utils import load_observations
from falcon.core.graph import create_graph_from_config
from falcon.core.logger import init_logging
from falcon.core.run_name import generate_run_dir


# Register custom OmegaConf resolvers
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)


def load_config(config_name: str = "config.yaml", run_dir: str = None, overrides: list = None) -> DictConfig:
    """Load config with run_dir injection and resume support.

    Args:
        config_name: Config file name (e.g., config.yaml)
        run_dir: Run directory path. If None, auto-generates one.
        overrides: List of key=value CLI overrides

    Returns:
        Resolved config with run_dir injected
    """
    # 1. Default run_dir if not specified
    if run_dir is None:
        run_dir = generate_run_dir()

    run_dir_path = Path(run_dir)
    saved_config = run_dir_path / "config.yaml"

    # 2. Load config (from run_dir if resuming, else from cwd)
    if saved_config.exists():
        print(f"Resuming from: {saved_config}")
        cfg = OmegaConf.load(saved_config)
    else:
        config_path = Path.cwd() / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        cfg = OmegaConf.load(config_path)

    # 3. Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # 4. Inject run_dir for ${run_dir} interpolations
    cfg.run_dir = run_dir

    # 5. Resolve all interpolations
    OmegaConf.resolve(cfg)

    # 6. Create run_dir and save config if new run
    run_dir_path.mkdir(parents=True, exist_ok=True)
    if not saved_config.exists():
        OmegaConf.save(cfg, saved_config)
        print(f"Saved config to: {saved_config}")

    return cfg


def launch_mode(cfg: DictConfig) -> None:
    """Launch mode: Full training and inference pipeline."""
    ray_init_args = cfg.get("ray", {}).get("init", {})
    ray.init(**ray_init_args)

#    # Add model path to Python path for imports
#    if cfg.model_path:
#        model_path = Path(cfg.model_path).resolve()
#        if model_path not in sys.path:
#            sys.path.insert(0, str(model_path))

    # Initialise logger (should be done before any other falcon code)
    init_logging(cfg)

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
    deployed_graph = falcon.DeployedGraph(graph, model_path=cfg.paths.get("import"))

    # 2) Prepare dataset manager for deployed graph and store initial samples
    dataset_manager = falcon.get_ray_dataset_manager(
        min_training_samples=cfg.buffer.min_training_samples,
        max_training_samples=cfg.buffer.max_training_samples,
        validation_window_size=cfg.buffer.validation_window_size,
        resample_batch_size=cfg.buffer.resample_batch_size,
        resample_interval=cfg.buffer.resample_interval,
        keep_resampling=cfg.buffer.keep_resampling,
        initial_samples_path=cfg.buffer.get("initial_samples_path", None),
        dump_config=cfg.buffer.get("dump", None),
    )

    # 3) Launch training & simulations
    graph_path = Path(cfg.paths.graph)
    deployed_graph.launch(dataset_manager, observations, graph_path=graph_path)

    ##########################
    ### Clean up resources ###
    ##########################

    deployed_graph.shutdown()
    falcon.finish_logging()


def sample_mode(cfg: DictConfig, sample_type: str) -> None:
    """Sample mode: Generate samples using different sampling strategies."""
    ray_init_args = cfg.get("ray", {}).get("init", {})
    ray.init(**ray_init_args)

#    # Add model path to Python path for imports
#    if cfg.model_path:
#        model_path = Path(cfg.model_path).resolve()
#        if model_path not in sys.path:
#            sys.path.insert(0, str(model_path))

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
    deployed_graph = falcon.DeployedGraph(graph, model_path=cfg.paths.get("import"))

    if sample_type == "prior":
        # Generate forward samples from prior
        samples = deployed_graph.sample(num_samples)

    elif sample_type == "posterior":
        # TODO: Implement posterior sampling (requires trained model and observations)
        deployed_graph.load(Path(cfg.paths.graph))
        samples = deployed_graph.sample_posterior(num_samples, observations)

    elif sample_type == "proposal":
        # Proposal sampling requires observations for conditioning
        # Load observations from config
        deployed_graph.load(Path(cfg.paths.graph))
        samples = deployed_graph.sample_proposal(num_samples, observations)

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


def parse_args():
    """Parse falcon CLI arguments."""
    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch"]:
        print("Usage:")
        print("  falcon launch [--run-dir DIR] [--config-name FILE] [key=value ...]")
        print("  falcon sample prior|posterior|proposal [--run-dir DIR] [--config-name FILE] [key=value ...]")
        print()
        print("Options:")
        print("  --run-dir DIR        Run directory (default: auto-generated)")
        print("  --config-name FILE   Config file (default: config.yaml)")
        sys.exit(1)

    mode = sys.argv[1]
    args = sys.argv[2:]

    sample_type = None
    if mode == "sample":
        if not args or args[0] not in ["prior", "posterior", "proposal"]:
            print("Error: sample requires type: prior, posterior, or proposal")
            sys.exit(1)
        sample_type = args.pop(0)

    # Extract --run-dir, --config-name and collect overrides
    run_dir = None
    config_name = "config.yaml"
    overrides = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--run-dir" and i + 1 < len(args):
            run_dir = args[i + 1]
            i += 1
        elif arg.startswith("--run-dir="):
            run_dir = arg.split("=", 1)[1]
        elif arg == "--config-name" and i + 1 < len(args):
            config_name = args[i + 1]
            i += 1
        elif arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        elif "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        i += 1

    return mode, sample_type, config_name, run_dir, overrides


def main():
    """Main CLI entry point."""
    mode, sample_type, config_name, run_dir, overrides = parse_args()
    cfg = load_config(config_name, run_dir, overrides)

    if mode == "launch":
        launch_mode(cfg)
    else:
        sample_mode(cfg, sample_type)


if __name__ == "__main__":
    main()
