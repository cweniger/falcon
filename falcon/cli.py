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

BANNER = "falcon \x1b[2m\u2581\u2582\u2585\u2587\u2588\u2586\u2583\u2582\u2581\u2581\x1b[0m"


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


def graph_mode(cfg) -> None:
    """Graph mode: Display the graph structure."""
    from falcon.core.graph import create_graph_from_config

    # Create graph from config (no Ray needed)
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # Collect info
    observed = [k for k, v in graph.observed_dict.items() if v]
    with_estimator = [n.name for n in graph.node_list if n.estimator_cls]

    print()
    print(render_git_graph_simple(graph))
    print()
    print(f"Nodes: {len(graph.node_list)} | Observed: {', '.join(observed)} | Estimators: {', '.join(with_estimator)}")


def load_config(config_name: str = "config.yaml", run_dir: str = None, overrides: list = None):
    """Load config with run_dir injection and resume support.

    Args:
        config_name: Config file name (e.g., config.yaml)
        run_dir: Run directory path. If None, auto-generates one.
        overrides: List of key=value CLI overrides

    Returns:
        Resolved config with run_dir injected
    """
    from datetime import datetime
    from omegaconf import OmegaConf
    from falcon.core.run_name import generate_run_dir

    # Register custom OmegaConf resolvers
    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

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


class _InteractiveStream:
    """Stream adapter that routes writes to InteractiveDisplay.log()."""
    def __init__(self, display):
        self.display = display
    def write(self, msg):
        msg = msg.rstrip()
        if msg:
            self.display.log(msg)
    def flush(self):
        pass


class _GracefulShutdown:
    """Handle double Ctrl+C pattern for non-interactive mode."""
    def __init__(self):
        self._interrupt_count = 0
        self._stopping = False
        self._original_handler = None

    def install(self):
        """Install signal handler."""
        import signal
        self._original_handler = signal.signal(signal.SIGINT, self._handler)

    def uninstall(self):
        """Restore original signal handler."""
        import signal
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)

    def _handler(self, signum, frame):
        """Handle SIGINT with double Ctrl+C pattern."""
        self._interrupt_count += 1
        if self._interrupt_count == 1:
            self._stopping = True
            print("\n\x1b[33m⚠ Stopping gracefully... (Ctrl+C again to force quit)\x1b[0m", flush=True)
            return
        # Second interrupt: force exit
        self.uninstall()
        raise KeyboardInterrupt

    @property
    def stop_requested(self) -> bool:
        return self._stopping


def launch_mode(cfg, interactive: bool = False) -> None:
    """Launch mode: Full training and inference pipeline."""
    import logging
    import threading
    import torch
    import ray
    from omegaconf import OmegaConf
    import falcon
    from falcon.core.graph import create_graph_from_config
    from falcon.core.logger import Logger, set_logger, info

    # Initialize interactive display or graceful shutdown handler
    display = None
    shutdown_handler = None
    if interactive:
        from falcon.interactive import InteractiveDisplay
        display = InteractiveDisplay()
        display.start()
    else:
        # Non-interactive mode: install double Ctrl+C handler
        shutdown_handler = _GracefulShutdown()
        shutdown_handler.install()

    # Get output directory from config
    output_dir = Path(cfg.run_dir)

    # Generate wandb group if not set - use run-dir folder name
    logging_cfg = OmegaConf.to_container(cfg.get("logging", {}), resolve=True)
    if logging_cfg.get("wandb", {}).get("enabled", False):
        if not logging_cfg.get("wandb", {}).get("group"):
            # Use the run-dir folder name as the group name
            logging_cfg.setdefault("wandb", {})["group"] = output_dir.name

    # Ensure local dir is set to graph path
    logging_cfg.setdefault("local", {})["dir"] = str(cfg.paths.graph)

    # Create driver logger and set as module-level logger
    # This enables falcon.info(), falcon.log() etc. for DeployedGraph and other components
    driver_logger = Logger("driver", logging_cfg, capture_exceptions=True)
    set_logger(driver_logger)

    # If interactive mode, replace console handler with one that routes to display
    if display:
        for handler in driver_logger._logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                driver_logger._logger.removeHandler(handler)
                # Add custom handler that routes to display
                interactive_handler = logging.StreamHandler(_InteractiveStream(display))
                interactive_handler.setFormatter(logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S'
                ))
                interactive_handler.setLevel(logging.INFO)
                driver_logger._logger.addHandler(interactive_handler)
                break

    # Log startup info
    info(f"falcon v{falcon.__version__}")
    info(f"Output: {output_dir}")

    # Initialize Ray
    ray_init_args = cfg.get("ray", {}).get("init", {})
    # Forward actor stdout/stderr to driver when console.level is set,
    # so node log messages and crash output reach the terminal.
    console_level = logging_cfg.get("console", {}).get("level", None)
    ray_init_args.setdefault("log_to_driver", console_level is not None)
    # Use a fixed namespace so falcon monitor can discover actors
    ray_init_args.setdefault("namespace", "falcon")
    # Suppress Ray startup banner
    ray_init_args.setdefault("logging_level", "ERROR")
    ray.init(**ray_init_args)

    # Show Ray cluster info with resources
    ctx = ray.get_runtime_context()
    gcs_address = ctx.gcs_address
    is_local = ray_init_args.get("address") is None
    ray_status = "new local instance" if is_local else "existing cluster"
    resources = ray.cluster_resources()
    cpu = int(resources.get("CPU", 0))
    gpu = int(resources.get("GPU", 0))
    mem_gb = resources.get("memory", 0) / (1024**3)

    info(f"Ray: {gcs_address} ({ray_status})")
    info(f"Resources: {cpu} CPU, {gpu} GPU, {mem_gb:.1f} GB")

    ########################
    ### Model definition ###
    ########################

    # Instantiate model components directly from graph
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # Convert observations to tensors, adding batch dimension
    observations = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()
    }

    # Log graph info
    info(str(graph))
    for name, shape in observations.items():
        info(f"Observed: {name} {list(shape.shape)}")

    ####################
    ### Run analysis ###
    ####################

    # 1) Deploy graph (pass logging config)
    deployed_graph = falcon.DeployedGraph(
        graph,
        model_path=cfg.paths.get("import"),
        log_config=logging_cfg,
    )

    # 2) Prepare dataset manager for deployed graph and store initial samples
    dataset_manager = falcon.get_ray_dataset_manager(
        min_training_samples=cfg.buffer.min_training_samples,
        max_training_samples=cfg.buffer.max_training_samples,
        validation_window_size=cfg.buffer.validation_window_size,
        resample_batch_size=cfg.buffer.resample_batch_size,
        resample_interval=cfg.buffer.resample_interval,
        simulate_chunk_size=cfg.buffer.get("simulate_chunk_size", 0),
        keep_resampling=cfg.buffer.keep_resampling,
        initial_samples_path=cfg.buffer.get("initial_samples_path", None),
        buffer_path=cfg.paths.buffer,
        store_fraction=cfg.buffer.get("store_fraction", 0.0),
        log_config=logging_cfg,
    )

    # 3) Start status polling thread for interactive mode
    status_thread = None
    graph_path = Path(cfg.paths.graph)
    if display:
        # Set log directory so display can read node output.log files
        display.set_log_dir(str(graph_path))

        def poll_status():
            """Background thread to poll MonitorBridge and update display."""
            import time
            while display.is_running:
                try:
                    bridge = ray.get_actor("falcon:monitor_bridge")
                    status = ray.get(bridge.get_status.remote())

                    # Update nodes
                    for name, node_status in status.get("nodes", {}).items():
                        display.update_node(
                            name=name,
                            status=node_status.get("status", "unknown"),
                            current_epoch=node_status.get("current_epoch", 0),
                            total_epochs=node_status.get("total_epochs", 0),
                            loss=node_status.get("loss"),
                            samples=node_status.get("samples", 0),
                        )

                    # Update buffer stats
                    buffer = status.get("buffer", {})
                    display.update_buffer(
                        training=buffer.get("training", 0),
                        validation=buffer.get("validation", 0),
                    )
                except Exception:
                    pass  # MonitorBridge may not be ready yet

                # Redraw footer to refresh log tail
                with display._lock:
                    display._draw_footer()

                time.sleep(1.0)

        status_thread = threading.Thread(target=poll_status, daemon=True)
        status_thread.start()

    # 4) Launch training & simulations
    # Create stop check callback for graceful shutdown
    if display:
        stop_check = lambda: display.stop_requested
    elif shutdown_handler:
        stop_check = lambda: shutdown_handler.stop_requested
    else:
        stop_check = None

    try:
        deployed_graph.launch(dataset_manager, observations, graph_path=graph_path, stop_check=stop_check)
    finally:
        ##########################
        ### Clean up resources ###
        ##########################

        # Stop interactive display first (restores terminal)
        if display:
            display.stop()

        # Uninstall shutdown handler
        if shutdown_handler:
            shutdown_handler.uninstall()

        deployed_graph.shutdown()
        driver_logger.shutdown()


def sample_mode(cfg, sample_type: str) -> None:
    """Sample mode: Generate samples using different sampling strategies.

    Samples are saved as individual NPZ files in:
        {paths.samples}/{sample_type}/{batch_timestamp}/000000.npz, ...
    """
    from datetime import datetime
    import numpy as np
    import torch
    import ray
    from omegaconf import OmegaConf
    import falcon
    from falcon.core.graph import create_graph_from_config
    from falcon.core.logger import Logger, set_logger, info

    # Setup logging config
    logging_cfg = OmegaConf.to_container(cfg.get("logging", {}), resolve=True)
    logging_cfg.setdefault("local", {})["dir"] = str(cfg.paths.graph)

    # Create driver logger and set as module-level logger
    driver_logger = Logger("driver", logging_cfg, capture_exceptions=True)
    set_logger(driver_logger)
    ray_init_args = cfg.get("ray", {}).get("init", {})
    # Forward actor stdout/stderr to driver when console.level is set
    console_level = logging_cfg.get("console", {}).get("level", None)
    ray_init_args.setdefault("log_to_driver", console_level is not None)
    # Use a fixed namespace for consistency
    ray_init_args.setdefault("namespace", "falcon")
    ray.init(**ray_init_args)

    # Instantiate model components directly from graph
    graph, observations = create_graph_from_config(cfg.graph, _cfg=cfg)

    # Convert observations to tensors, adding batch dimension
    observations = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in observations.items()
    }

    if sample_type == "prior":
        sample_cfg = cfg.sample.get("prior", None)
    elif sample_type == "posterior":
        sample_cfg = cfg.sample.get("posterior", None)
    elif sample_type == "proposal":
        sample_cfg = cfg.sample.get("proposal", None)
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    if sample_cfg is None or "n" not in sample_cfg:
        raise ValueError(f"Missing sample.{sample_type}.n in config. Add it to your config.yaml.")

    num_samples = sample_cfg.n
    info(f"Generating {num_samples} samples using {sample_type} sampling...")
    info(str(graph))

    # Deploy graph for sampling
    deployed_graph = falcon.DeployedGraph(
        graph,
        model_path=cfg.paths.get("import"),
        log_config=logging_cfg,
    )

    if sample_type == "prior":
        # Generate forward samples from prior
        sample_refs = deployed_graph.sample(num_samples)

    elif sample_type == "posterior":
        deployed_graph.load(Path(cfg.paths.graph))
        sample_refs = deployed_graph.sample_posterior(num_samples, observations)

    elif sample_type == "proposal":
        deployed_graph.load(Path(cfg.paths.graph))
        sample_refs = deployed_graph.sample_proposal(num_samples, observations)

    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    # Resolve refs to arrays (keys are flat: 'theta.value', 'theta.log_prob', 'x.value', ...)
    samples = deployed_graph._refs_to_arrays(sample_refs)

    # Build key selection based on node names (strip .value/.log_prob suffixes)
    node_keys = {k for k in samples.keys() if k.endswith('.value')}

    if sample_type in ["prior", "proposal"]:
        # Default: save all .value keys
        default_keys = set(node_keys)
    elif sample_type == "posterior":
        # Default: save only posterior nodes (nodes with evidence)
        default_keys = {
            f"{k}.value" for k, node in graph.node_dict.items()
            if node.evidence and f"{k}.value" in samples
        }

    # Apply user overrides (user specifies node names, we match .value keys)
    exclude_keys = sample_cfg.get("exclude_keys", None)
    add_keys = sample_cfg.get("add_keys", None)

    if exclude_keys:
        exclude_set = {f"{k}.value" for k in exclude_keys.split(",")}
        default_keys -= exclude_set

    if add_keys:
        add_set = {f"{k}.value" for k in add_keys.split(",")}
        default_keys |= add_set

    # Filter samples to selected keys, strip .value suffix for user-facing output
    save_data = {}
    for k in default_keys:
        if k in samples:
            # Strip '.value' suffix for cleaner output key names
            user_key = k[:-6] if k.endswith('.value') else k
            save_data[user_key] = samples[k]

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
    driver_logger.shutdown()


def monitor_mode(address: str = "auto", refresh: float = 1.0):
    """Monitor mode: Launch the TUI monitor directly (no subprocess)."""
    from falcon.monitor import init_ray_for_monitor, FalconMonitor
    if not init_ray_for_monitor(address):
        sys.exit(1)
    app = FalconMonitor(ray_address=address, refresh_interval=refresh)
    app.run()


def _get_version():
    """Get package version string."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("falcon")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def parse_args():
    """Parse falcon CLI arguments."""
    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch", "graph", "monitor"]:
        print(f"{BANNER} v{_get_version()}")
        print()
        print("Usage:")
        print("  falcon launch [--run-dir DIR] [--config-name FILE] [--interactive] [key=value ...]")
        print("  falcon sample prior|posterior|proposal [--run-dir DIR] [--config-name FILE] [key=value ...]")
        print("  falcon graph [--config-name FILE]")
        print("  falcon monitor [--address ADDR] [--refresh SECS]")
        print()
        print("Options:")
        print("  --run-dir DIR        Run directory (default: auto-generated)")
        print("  --config-name FILE   Config file (default: config.yaml)")
        print("  --interactive, -i    Interactive TUI with live status footer")
        print("  --address ADDR       Ray cluster address (default: auto)")
        print("  --refresh SECS       Monitor refresh interval (default: 1.0)")
        sys.exit(0)

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
        return mode, None, None, None, None, False, address, refresh

    sample_type = None
    if mode == "sample":
        if not args or args[0] not in ["prior", "posterior", "proposal"]:
            print("Error: sample requires type: prior, posterior, or proposal")
            sys.exit(1)
        sample_type = args.pop(0)

    # Extract --run-dir, --config-name, --interactive and collect overrides
    run_dir = None
    config_name = "config.yaml"
    interactive = False
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
        elif arg in ("--interactive", "-i"):
            interactive = True
        elif "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        i += 1

    return mode, sample_type, config_name, run_dir, overrides, interactive, None, None


def main():
    """Main CLI entry point."""
    mode, sample_type, config_name, run_dir, overrides, interactive, address, refresh = parse_args()

    # Print startup banner (skip for interactive mode - it will draw its own)
    if not interactive:
        print(f"{BANNER} v{_get_version()}", flush=True)

    # Monitor mode doesn't need config loading
    if mode == "monitor":
        monitor_mode(address=address, refresh=refresh)
        return

    cfg = load_config(config_name, run_dir, overrides)

    if mode == "launch":
        launch_mode(cfg, interactive=interactive)
    elif mode == "graph":
        graph_mode(cfg)
    else:
        sample_mode(cfg, sample_type)


if __name__ == "__main__":
    main()
