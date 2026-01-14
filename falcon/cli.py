#!/usr/bin/env python3
"""
Falcon Adaptive Training - Standalone CLI Tool
Usage: falcon launch [--run-dir DIR] [--config-name FILE] [key=value ...]
       falcon sample prior|posterior|proposal [--run-dir DIR] [--config-name FILE] [key=value ...]
       falcon graph [--config-name FILE]

Run directory behavior:
  - If --run-dir not specified, generates: outputs/adj-noun-YYMMDD-HHMM
  - If --run-dir exists with config.yaml, resumes from saved config
  - Otherwise, loads ./config.yaml and saves resolved config to run_dir
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import ray

from omegaconf import OmegaConf, DictConfig

import falcon
from falcon.core.utils import load_observations
from falcon.core.graph import create_graph_from_config
from falcon.core.logger import init_logging
from falcon.core.logging import initialize_logging_for, set_falcon_log
from falcon.core.run_name import generate_run_dir


# Register custom OmegaConf resolvers
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)


def render_git_graph_simple(graph):
    """Render a simplified git-log style ASCII graph visualization.

    Shows DAG structure with cleaner visualization.
    """
    sorted_names = graph.sorted_node_names
    parents_dict = graph.parents_dict

    # Build children dict to know which nodes have children
    children_dict = {name: [] for name in sorted_names}
    for name in sorted_names:
        for parent in parents_dict[name]:
            if parent in children_dict:
                children_dict[parent].append(name)

    # Track active vertical lines by column
    # Each entry is the node name that "owns" that column
    columns = []  # List of node names, index = column

    lines = []

    for idx, name in enumerate(sorted_names):
        node = graph.node_dict[name]
        is_last = (idx == len(sorted_names) - 1)

        # Find parent columns
        parent_cols = []
        for p in parents_dict[name]:
            if p in columns:
                parent_cols.append(columns.index(p))
        parent_cols.sort()

        # Determine this node's column
        if parent_cols:
            my_col = parent_cols[0]
            # Remove other parent columns (they merge here)
            for pc in reversed(parent_cols[1:]):
                columns[pc] = None
        else:
            # New root - use first empty column or append
            if None in columns:
                my_col = columns.index(None)
            else:
                my_col = len(columns)
                columns.append(None)

        # Ensure columns list is long enough
        while len(columns) <= my_col:
            columns.append(None)

        # Draw merge lines if multiple parents
        if len(parent_cols) > 1:
            merge_line = []
            max_parent_col = max(parent_cols)
            for c in range(max_parent_col + 1):
                if c == my_col:
                    merge_line.append('|')
                elif c in parent_cols:
                    merge_line.append('/')
                elif columns[c] is not None:
                    merge_line.append('|')
                else:
                    merge_line.append(' ')
                # Add space after, except for the merge slash
                if c < max_parent_col:
                    if c + 1 in parent_cols and c + 1 != my_col:
                        merge_line.append('')  # No space before /
                    else:
                        merge_line.append(' ')
            lines.append(''.join(merge_line))

        # Draw the node line
        line = []
        for c in range(len(columns)):
            if c == my_col:
                line.append('*')
            elif columns[c] is not None:
                line.append('|')
            else:
                line.append(' ')
            if c < len(columns) - 1 or columns[c] is not None:
                line.append(' ')

        # Get node info
        simulator_cls = graph.get_simulator_cls(name)
        if hasattr(simulator_cls, "display_name"):
            class_name = simulator_cls.display_name
        else:
            class_name = str(simulator_cls)
            if '.' in class_name:
                class_name = class_name.split('.')[-1]

        # Add evidence info (inference direction)
        evidence = graph.evidence_dict.get(name, [])
        evidence_str = f"  ← {', '.join(evidence)}" if evidence else ""

        observed = " (observed)" if graph.observed_dict.get(name) else ""

        node_line = ''.join(line).rstrip() + f" {name}{evidence_str}{observed}"
        lines.append(node_line)

        # Update column ownership
        columns[my_col] = name

        # Remove parent columns (connection completed)
        for p in parents_dict[name]:
            if p in columns:
                col_idx = columns.index(p)
                if col_idx != my_col:
                    columns[col_idx] = None

        # Draw continuation line only if there are more nodes and active columns
        if not is_last:
            # Check if this node has children (needs continuation)
            has_children = len(children_dict[name]) > 0
            # Check if there are other active columns
            other_active = any(c is not None and c != name for c in columns)

            if has_children or other_active:
                cont = []
                for c in range(len(columns)):
                    if columns[c] is not None:
                        cont.append('|')
                    else:
                        cont.append(' ')
                    cont.append(' ')  # Space between columns
                cont_str = ''.join(cont).rstrip()
                if cont_str:
                    lines.append(cont_str)

    return '\n'.join(l for l in lines if l.strip())


def graph_mode(cfg: DictConfig) -> None:
    """Graph mode: Display the graph structure."""
    # Create graph from config (no Ray needed)
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # Collect info
    observed = [k for k, v in graph.observed_dict.items() if v]
    with_estimator = [n.name for n in graph.node_list if n.estimator_cls]

    print()
    print(render_git_graph_simple(graph))
    print()
    print(f"Nodes: {len(graph.node_list)} | Observed: {', '.join(observed)} | Estimators: {', '.join(with_estimator)}")


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

    return cfg


class TeeOutput:
    """Write to both terminal and log file."""
    def __init__(self, log_file, terminal):
        self.terminal = terminal
        self.log = open(log_file, "a")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


def launch_mode(cfg: DictConfig) -> None:
    """Launch mode: Full training and inference pipeline."""
    ray_init_args = cfg.get("ray", {}).get("init", {})
    # Suppress worker stdout/stderr forwarding to driver (use output.log instead)
    ray_init_args.setdefault("log_to_driver", False)
    # Use a fixed namespace so falcon monitor can discover actors
    ray_init_args.setdefault("namespace", "falcon")
    # Suppress Ray startup banner
    ray_init_args.setdefault("logging_level", "ERROR")
    ray.init(**ray_init_args)

    # Get output directory from config
    output_dir = Path(cfg.run_dir)

    # Create falcon.log and tee output to both terminal and file
    falcon_log = output_dir / "falcon.log"
    tee = TeeOutput(falcon_log, sys.stdout)

    # Print header (non-timestamped section)
    tee.write(f"falcon  ▁▁▁▃▆█▆▃▁▁▁▁  v{falcon.__version__}\n\n")
    tee.write("Output:\n")
    tee.write(f"  {output_dir}\n\n")

    # Show Ray cluster info with resources
    ctx = ray.get_runtime_context()
    gcs_address = ctx.gcs_address
    is_local = ray_init_args.get("address") is None
    ray_status = "new local instance" if is_local else "existing cluster"
    resources = ray.cluster_resources()
    cpu = int(resources.get("CPU", 0))
    gpu = int(resources.get("GPU", 0))
    mem_gb = resources.get("memory", 0) / (1024**3)
    tee.write("Ray:\n")
    tee.write(f"  {gcs_address} ({ray_status})\n")
    tee.write(f"  Resources: {cpu} CPU, {gpu} GPU, {mem_gb:.1f} GB\n\n")

    # Initialise logger (should be done before any other falcon code)
    init_logging(cfg)
    # Capture driver stdout/stderr to driver/output.log (keep terminal output too)
    initialize_logging_for("driver", keep_original=True)

    ########################
    ### Model definition ###
    ########################

    # Instantiate model components directly from graph
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # TODO: This is some hack right now
    observations = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()
    }

    tee.write(str(graph) + "\n\n")
    tee.write("Observed:\n")
    for name, shape in observations.items():
        tee.write(f"  {name} {list(shape.shape)}\n")
    tee.write("\n")

    # Pass falcon.log file handle to logging module for timestamped output
    set_falcon_log(tee.log)

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
    """Sample mode: Generate samples using different sampling strategies.

    Samples are saved as individual NPZ files in:
        {paths.samples}/{sample_type}/{batch_timestamp}/000000.npz, ...
    """
    ray_init_args = cfg.get("ray", {}).get("init", {})
    # Suppress worker stdout/stderr forwarding to driver (use output.log instead)
    ray_init_args.setdefault("log_to_driver", False)
    # Use a fixed namespace for consistency
    ray_init_args.setdefault("namespace", "falcon")
    ray.init(**ray_init_args)

    # Initialise logger and capture driver output
    init_logging(cfg)
    initialize_logging_for("driver", keep_original=True)

    # Instantiate model components directly from graph
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # TODO: This is some hack right now
    observations = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()
    }

    if sample_type == "prior":
        sample_cfg = cfg.sample.get("prior", {})
    elif sample_type == "posterior":
        sample_cfg = cfg.sample.get("posterior", {})
    elif sample_type == "proposal":
        sample_cfg = cfg.sample.get("proposal", {})
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

    # Determine output directory
    samples_dir = cfg.paths.get("samples", f"{cfg.run_dir}/samples_dir")
    batch_timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = Path(samples_dir) / sample_type / batch_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving samples to: {output_dir}/")

    # Save each sample as individual NPZ file
    num_samples = len(next(iter(save_data.values())))
    for i in range(num_samples):
        sample_data = {k: v[i] for k, v in save_data.items()}
        sample_data["_batch"] = batch_timestamp
        sample_path = output_dir / f"{i:06d}.npz"
        np.savez(sample_path, **sample_data)

    print(f"Saved {num_samples} {sample_type} samples to: {output_dir}/")

    # Clean up
    deployed_graph.shutdown()


def monitor_mode(address: str = "auto", refresh: float = 1.0):
    """Monitor mode: Launch the TUI monitor for training runs."""
    import subprocess
    subprocess.run([
        sys.executable, "-m", "falcon.monitor",
        "--address", address,
        "--refresh", str(refresh),
    ])


def parse_args():
    """Parse falcon CLI arguments."""
    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch", "graph", "monitor"]:
        print("Usage:")
        print("  falcon launch [--run-dir DIR] [--config-name FILE] [key=value ...]")
        print("  falcon sample prior|posterior|proposal [--run-dir DIR] [--config-name FILE] [key=value ...]")
        print("  falcon graph [--config-name FILE]")
        print("  falcon monitor [--address ADDR] [--refresh SECS]")
        print()
        print("Options:")
        print("  --run-dir DIR        Run directory (default: auto-generated)")
        print("  --config-name FILE   Config file (default: config.yaml)")
        print("  --address ADDR       Ray cluster address (default: auto)")
        print("  --refresh SECS       Monitor refresh interval (default: 1.0)")
        sys.exit(1)

    mode = sys.argv[1]
    args = sys.argv[2:]

    # Handle monitor mode separately (doesn't need config)
    if mode == "monitor":
        address = "auto"
        refresh = 1.0
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--address" and i + 1 < len(args):
                address = args[i + 1]
                i += 1
            elif arg.startswith("--address="):
                address = arg.split("=", 1)[1]
            elif arg == "--refresh" and i + 1 < len(args):
                refresh = float(args[i + 1])
                i += 1
            elif arg.startswith("--refresh="):
                refresh = float(arg.split("=", 1)[1])
            i += 1
        return mode, None, None, None, None, address, refresh

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

    return mode, sample_type, config_name, run_dir, overrides, None, None


def main():
    """Main CLI entry point."""
    mode, sample_type, config_name, run_dir, overrides, address, refresh = parse_args()

    # Monitor mode doesn't need config loading
    if mode == "monitor":
        monitor_mode(address=address, refresh=refresh)
        return

    cfg = load_config(config_name, run_dir, overrides)

    if mode == "launch":
        launch_mode(cfg)
    elif mode == "graph":
        graph_mode(cfg)
    else:
        sample_mode(cfg, sample_type)


if __name__ == "__main__":
    main()
