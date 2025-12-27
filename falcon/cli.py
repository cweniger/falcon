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
from falcon.core.logger import init_logging


def render_git_graph(graph):
    """Render a git-log style ASCII graph visualization.

    Args:
        graph: A Graph object with sorted_node_names and parents_dict

    Returns:
        String with git-style graph visualization
    """
    sorted_names = graph.sorted_node_names
    parents_dict = graph.parents_dict

    # Build children dict (reverse of parents)
    children_dict = {name: [] for name in sorted_names}
    for name in sorted_names:
        for parent in parents_dict[name]:
            if parent in children_dict:
                children_dict[parent].append(name)

    # Assign columns to nodes
    # Strategy: each node gets a column, trying to reuse columns when possible
    node_col = {}
    active_cols = []  # List of (col, node_name) for active vertical lines
    max_col = -1

    lines = []

    for i, name in enumerate(sorted_names):
        node = graph.node_dict[name]
        parents = [p for p in parents_dict[name] if p in node_col]
        children = children_dict[name]

        # Find which columns have lines coming into this node
        parent_cols = sorted([node_col[p] for p in parents]) if parents else []

        # Determine this node's column
        if parent_cols:
            # Place on leftmost parent column
            my_col = parent_cols[0]
        else:
            # New root node - find first available column
            used_cols = set(c for c, _ in active_cols)
            my_col = 0
            while my_col in used_cols:
                my_col += 1

        node_col[name] = my_col
        max_col = max(max_col, my_col)

        # Remove parent connections that terminate here
        active_cols = [(c, n) for c, n in active_cols if n not in parents or c != node_col.get(n)]

        # Build the line for this node
        # First, handle merge lines if multiple parents
        if len(parent_cols) > 1:
            # Draw merge line
            merge_line = []
            for c in range(max(parent_cols) + 1):
                if c in parent_cols:
                    if c == my_col:
                        merge_line.append('|')
                    elif c < my_col:
                        merge_line.append('|')
                    else:
                        merge_line.append('|')
                elif c == my_col:
                    merge_line.append('|')
                elif any(c == col for col, _ in active_cols):
                    merge_line.append('|')
                else:
                    merge_line.append(' ')

            # Draw the merge connections
            merge_str = ""
            for c in range(max(parent_cols) + 1):
                if c in parent_cols and c != my_col:
                    if c < my_col:
                        merge_str += '|'
                    else:
                        merge_str += '|'
                elif c == my_col:
                    merge_str += '|'
                elif any(c == col for col, _ in active_cols):
                    merge_str += '|'
                else:
                    merge_str += ' '
                merge_str += ' '

            # Simplified merge line with backslashes
            if parent_cols[-1] > my_col:
                pre_line = []
                for c in range(parent_cols[-1] + 1):
                    if c < my_col:
                        if any(c == col for col, _ in active_cols) or c in parent_cols:
                            pre_line.append('| ')
                        else:
                            pre_line.append('  ')
                    elif c == my_col:
                        pre_line.append('|')
                        # Add merge line
                        pre_line.append('\\' * (parent_cols[-1] - my_col))
                        break
                lines.append(''.join(pre_line))

        # Build the node line
        line_parts = []
        for c in range(max_col + 2):
            if c < my_col:
                if any(c == col for col, _ in active_cols):
                    line_parts.append('| ')
                else:
                    line_parts.append('  ')
            elif c == my_col:
                line_parts.append('* ')
                break

        # Get node info
        simulator_cls = graph.get_simulator_cls(name)
        if hasattr(simulator_cls, "display_name"):
            class_name = simulator_cls.display_name
        else:
            class_name = str(simulator_cls).split('.')[-1]

        parents_str = ', '.join(parents_dict[name]) if parents_dict[name] else '(root)'
        evidence = graph.evidence_dict.get(name, [])
        evidence_str = f" [evidence: {', '.join(evidence)}]" if evidence else ""

        node_line = ''.join(line_parts) + f"{name} <- {parents_str}{evidence_str}"
        lines.append(node_line)

        # Add this node to active columns if it has children
        if children:
            active_cols.append((my_col, name))

        # Draw continuation lines after node (if not last)
        if i < len(sorted_names) - 1:
            cont_line = []
            # Update active cols - remove this node's parents, add this node if it has children
            new_active = [(c, n) for c, n in active_cols if n != name or n in [name] and children]
            for c in range(max_col + 2):
                if any(c == col for col, _ in new_active):
                    cont_line.append('| ')
                elif c <= my_col:
                    cont_line.append('  ')
                else:
                    break
            if cont_line and any(p != ' ' for p in ''.join(cont_line)):
                lines.append(''.join(cont_line).rstrip())

    return '\n'.join(lines)


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
                idx = columns.index(p)
                if idx != my_col:
                    columns[idx] = None

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


@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def launch_main(cfg: DictConfig) -> None:
    """Launch mode entry point."""
    launch_mode(cfg)


def main():
    """Main CLI entry point with explicit mode dispatch."""

    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch", "graph"]:
        print("Error: Must specify mode. Usage:")
        print("  falcon launch [hydra_options...]")
        print("  falcon sample prior|proposal|posterior [hydra_options...]")
        print("  falcon graph [hydra_options...]")
        sys.exit(1)

    mode = sys.argv.pop(1)  # Remove mode from sys.argv

    if mode == "graph":
        @hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
        def graph_main(cfg: DictConfig) -> None:
            graph_mode(cfg)
        graph_main()
    elif mode == "sample":
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
