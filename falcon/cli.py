#!/usr/bin/env python3
"""
Falcon Adaptive Training - Standalone CLI Tool
Usage: falcon launch [--output DIR] [--config FILE] [key=value ...]
       falcon sample prior|posterior|proposal|ppd [--output DIR] [--config FILE] [key=value ...]
       falcon graph [--config FILE]

Run directory behavior:
  - If --output not specified, generates: outputs/adj-noun-YYMMDD-HHMM
  - If --output exists with config.yml, resumes from saved config
  - Otherwise, loads ./config.yml and saves resolved config to output dir
"""

import sys
import os
import warnings
from pathlib import Path

BANNER = "falcon \x1b[2m\u2581\u2582\u2585\u2587\u2588\u2586\u2583\u2582\u2581\u2581\x1b[0m"


def render_git_graph_simple(graph):
    """Render a simplified git-log style ASCII graph visualization.

    Shows DAG structure with cleaner visualization.
    """
    sorted_names = graph.forward_order
    parents_dict = graph.forward_deps

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


def load_config(config_name: str = "config.yml", run_dir: str = None, overrides: list = None):
    """Load config with run_dir injection and resume support.

    Args:
        config_name: Config file name (e.g., config.yml)
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
    auto_generated = run_dir is None
    if auto_generated:
        run_dir = generate_run_dir()

    run_dir_path = Path(run_dir)
    saved_config = run_dir_path / "config.yml"

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

    # 7. Update 'latest' symlink in parent directory (only for auto-generated run dirs)
    if auto_generated:
        latest_link = run_dir_path.parent / "latest"
        try:
            latest_link.unlink(missing_ok=True)
            latest_link.symlink_to(run_dir_path.name)
        except OSError:
            pass  # Ignore on platforms/filesystems that don't support symlinks

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


def _fmt_duration(seconds):
    """Format a duration in seconds as e.g. '4m 23s' or '1h 02m 05s'."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _build_run_summary(status, output_dir, cfg, deployed_graph, start_time=None, end_time=None):
    """Build a compact end-of-run summary as a list of text lines."""
    from datetime import datetime

    lines = []
    lines.append("=" * 60)
    lines.append(f"falcon launch {status}")
    lines.append(f"Output:  {output_dir}")
    samples_path = cfg.paths.get("samples", f"{cfg.run_dir}/samples")
    lines.append(f"Samples: {samples_path}")
    graph_path = Path(cfg.paths.graph)
    lines.append(f"Logs:    {graph_path / 'driver' / 'output.log'}  (driver)")
    try:
        node_names = list(cfg.graph.keys())
    except Exception:
        node_names = []
    if node_names:
        n = len(node_names)
        if n <= 6:
            suffix = f"(per-node: {', '.join(node_names)})"
        else:
            suffix = f"(per-node, {n} nodes)"
        lines.append(f"         {graph_path / '<node>' / 'output.log'}  {suffix}")
    if start_time is not None:
        lines.append(f"Started: {datetime.fromtimestamp(start_time):%Y-%m-%d %H:%M:%S}")
    if end_time is not None:
        lines.append(f"Ended:   {datetime.fromtimestamp(end_time):%Y-%m-%d %H:%M:%S}")
    if start_time is not None and end_time is not None:
        lines.append(f"Runtime: {_fmt_duration(end_time - start_time)}")
    try:
        import ray
        if ray.is_initialized():
            res = ray.cluster_resources()
            cpu = int(res.get("CPU", 0))
            gpu = int(res.get("GPU", 0))
            mem_gb = res.get("memory", 0) / (1024 ** 3)
            n_alive = sum(1 for n in ray.nodes() if n.get("Alive"))
            lines.append(
                f"Ray:     {n_alive} node(s) | {cpu} CPU, {gpu} GPU, {mem_gb:.1f} GB"
            )
    except Exception:
        pass
    lines.append("=" * 60)
    return lines


def _save_samples(samples, sample_cfg, sample_type, graph, cfg, info_fn=print):
    """Save generated samples to disk.

    Args:
        samples: Dict of sample arrays (keys like 'theta.value', 'x.value')
        sample_cfg: Config for this sample type (with exclude_keys, add_keys)
        sample_type: 'prior', 'posterior', 'proposal', or 'ppd'
        graph: The Graph object (for determining default keys)
        cfg: Full config (for paths)
        info_fn: Function to use for info logging
    """
    import numpy as np
    from pathlib import Path

    # Build key selection based on node names (strip .value/.log_prob suffixes)
    node_keys = {k for k in samples.keys() if k.endswith('.value')}

    if sample_type in ["prior", "proposal", "ppd"]:
        # Default: save all .value keys
        default_keys = set(node_keys)
    elif sample_type == "posterior":
        # Default: save only posterior nodes (nodes with evidence)
        default_keys = {
            f"{k}.value" for k, node in graph.node_dict.items()
            if node.evidence and f"{k}.value" in samples
        }
    else:
        default_keys = set(node_keys)

    # Apply user overrides (user specifies node names, we match .value keys)
    exclude_keys = sample_cfg.get("exclude_keys", None)
    add_keys = sample_cfg.get("add_keys", None)

    if exclude_keys:
        if isinstance(exclude_keys, str):
            exclude_set = {f"{k}.value" for k in exclude_keys.split(",")}
        else:
            exclude_set = {f"{k}.value" for k in exclude_keys}
        default_keys -= exclude_set

    if add_keys:
        if isinstance(add_keys, str):
            add_set = {f"{k}.value" for k in add_keys.split(",")}
        else:
            add_set = {f"{k}.value" for k in add_keys}
        default_keys |= add_set

    # Filter samples to selected keys, strip .value suffix for user-facing output
    save_data = {}
    for k in default_keys:
        if k in samples:
            # Strip '.value' suffix for cleaner output key names
            user_key = k[:-6] if k.endswith('.value') else k
            save_data[user_key] = samples[k]

    info_fn(f"Generated samples with shapes:")
    for key, value in save_data.items():
        info_fn(f"  {key}: {value.shape}")

    # Determine output directory (flat structure)
    samples_dir = cfg.paths.get("samples", f"{cfg.run_dir}/samples")
    output_dir = Path(samples_dir) / sample_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find next index from existing NPZ files
    existing = sorted(output_dir.glob("*.npz"))
    start_idx = len(existing)

    info_fn(f"Saving samples to: {output_dir}/")

    # Save each sample as individual NPZ file
    num_samples = len(next(iter(save_data.values())))
    for i in range(num_samples):
        sample_data = {k: v[i] for k, v in save_data.items()}
        sample_path = output_dir / f"{start_idx + i:06d}.npz"
        np.savez(sample_path, **sample_data)

    info_fn(f"Saved {num_samples} {sample_type} samples to: {output_dir}/")


def launch_mode(cfg, auto_sample: bool = True, timeout: float = None) -> None:
    """Launch mode: Full training and inference pipeline."""
    import time as _time
    import torch
    import ray
    from omegaconf import OmegaConf
    import falcon
    from falcon.core.graph import create_graph_from_config
    from falcon.core.logger import Logger, set_logger, info

    launch_start = _time.time()

    # Install graceful shutdown handler (double Ctrl+C to force quit)
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

    # Log startup info
    info(f"falcon v{falcon.__version__}")
    info(f"Output: {output_dir}")

    # Initialize Ray
    ray_init_args = cfg.get("ray", {}).get("init", {})
    # Forward actor stdout/stderr to driver when console.level is set,
    # so node log messages and crash output reach the terminal.
    console_level = logging_cfg.get("console", {}).get("level", None)
    ray_init_args.setdefault("log_to_driver", console_level is not None)
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

    graph_path = Path(cfg.paths.graph)

    # Create stop check callback for graceful shutdown (handles Ctrl+C and timeout)
    _start_time = _time.time()
    _timeout_logged = False

    def stop_check():
        nonlocal _timeout_logged
        # Check user interrupt (Ctrl+C)
        if shutdown_handler.stop_requested:
            return True
        # Check timeout
        if timeout is not None:
            elapsed = _time.time() - _start_time
            if elapsed >= timeout:
                if not _timeout_logged:
                    info(f"Timeout reached ({timeout}s), stopping gracefully...")
                    _timeout_logged = True
                return True
        return False

    run_status = "completed"
    deployed_graph = None
    try:
        # 1) Deploy graph (pass logging config)
        deployed_graph = falcon.DeployedGraph(
            graph,
            model_path=cfg.paths.get("import"),
            log_config=logging_cfg,
        )

        # 2) Prepare dataset manager for deployed graph and store initial samples
        from omegaconf import OmegaConf as _OmegaConf
        from falcon.core.raystore import BufferConfig as _BufferConfig
        buffer_cfg = _OmegaConf.merge(_OmegaConf.structured(_BufferConfig), cfg.buffer)
        dataset_manager = falcon.get_ray_dataset_manager(
            buffer_cfg,
            snapshots_path=str(Path(cfg.run_dir) / "buffer" / "snapshots"),
            log_config=logging_cfg,
        )

        deployed_graph.launch(dataset_manager, observations, graph_path=graph_path, stop_check=stop_check)

        #############################
        ### Posterior sampling    ###
        #############################

        # Check if posterior sampling is configured and enabled
        sample_cfg = cfg.get("sample", {}).get("posterior", {})
        num_posterior_samples = sample_cfg.get("n", 0)

        if auto_sample and num_posterior_samples > 0:
            info(f"Generating {num_posterior_samples} posterior samples...")

            sample_refs = deployed_graph.sample_posterior(num_posterior_samples, observations)
            samples = deployed_graph._refs_to_arrays(sample_refs)

            # Save posterior samples
            _save_samples(
                samples=samples,
                sample_cfg=sample_cfg,
                sample_type="posterior",
                graph=graph,
                cfg=cfg,
                info_fn=info,
            )

        # Check if PPD sampling is configured and enabled
        ppd_cfg = cfg.get("sample", {}).get("ppd", {})
        num_ppd_samples = ppd_cfg.get("n", 0)

        if auto_sample and num_ppd_samples > 0:
            info(f"Generating {num_ppd_samples} PPD samples...")

            sample_refs = deployed_graph.sample_ppd(num_ppd_samples, observations)
            samples = deployed_graph._refs_to_arrays(sample_refs)

            _save_samples(
                samples=samples,
                sample_cfg=ppd_cfg,
                sample_type="ppd",
                graph=graph,
                cfg=cfg,
                info_fn=info,
            )

    except KeyboardInterrupt:
        run_status = "interrupted"
        raise
    except Exception as e:
        run_status = f"failed ({type(e).__name__}: {e})"
        raise
    finally:
        ##########################
        ### Clean up resources ###
        ##########################

        # A graceful Ctrl+C / timeout exits the launch normally, not via an
        # exception, so reflect that here.
        if run_status == "completed" and shutdown_handler.stop_requested:
            run_status = "interrupted"

        # Build the end-of-run summary and route it through the driver logger
        # so it lands in driver/output.log (and, in plain mode, on stdout).
        summary_lines = _build_run_summary(
            run_status, output_dir, cfg, deployed_graph,
            start_time=launch_start, end_time=_time.time(),
        )
        for line in summary_lines:
            info(line)

        # Uninstall shutdown handler
        shutdown_handler.uninstall()

        if deployed_graph is not None:
            deployed_graph.shutdown()
        driver_logger.shutdown()


def sample_mode(cfg, sample_type: str) -> None:
    """Sample mode: Generate samples using different sampling strategies.

    Samples are saved as individual NPZ files in:
        {paths.samples}/{sample_type}/000000.npz, ...
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
    elif sample_type == "ppd":
        sample_cfg = cfg.sample.get("ppd", None)
    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    if sample_cfg is None or "n" not in sample_cfg:
        raise ValueError(f"Missing sample.{sample_type}.n in config. Add it to your config.yml.")

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

    elif sample_type == "ppd":
        deployed_graph.load(Path(cfg.paths.graph))
        sample_refs = deployed_graph.sample_ppd(num_samples, observations)

    else:
        raise ValueError(f"Unknown sample type: {sample_type}")

    # Resolve refs to arrays and save
    samples = deployed_graph._refs_to_arrays(sample_refs)
    _save_samples(
        samples=samples,
        sample_cfg=sample_cfg,
        sample_type=sample_type,
        graph=graph,
        cfg=cfg,
        info_fn=print,
    )

    # Clean up
    deployed_graph.shutdown()
    driver_logger.shutdown()


def _get_version():
    """Get package version string."""
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version("falcon-sbi")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def parse_args():
    """Parse falcon CLI arguments."""
    if len(sys.argv) < 2 or sys.argv[1] not in ["sample", "launch", "graph"]:
        print(f"{BANNER} v{_get_version()}")
        print()
        print("Usage:")
        print("  falcon launch [--output DIR] [--config FILE] [key=value ...]")
        print("  falcon sample prior|posterior|proposal|ppd [--output DIR] [--config FILE] [key=value ...]")
        print("  falcon graph [--config FILE]")
        print()
        print("Options:")
        print("  -o, --output DIR       Output directory (default: auto-generated)")
        print("  -c, --config FILE      Config file (default: config.yml)")
        print("  --no-auto-sample       Skip automatic sampling after training")
        print("  --timeout SECONDS      Stop training after SECONDS (graceful stop)")
        sys.exit(0)

    mode = sys.argv[1]
    args = sys.argv[2:]

    sample_type = None
    if mode == "sample":
        if not args or args[0] not in ["prior", "posterior", "proposal", "ppd"]:
            print("Error: sample requires type: prior, posterior, proposal, or ppd")
            sys.exit(1)
        sample_type = args.pop(0)

    # Extract --output, --config and collect overrides
    run_dir = None
    config_name = "config.yml"
    auto_sample = True  # Run configured sample types after training
    timeout = None  # Training timeout in seconds
    overrides = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--output", "-o") and i + 1 < len(args):
            run_dir = args[i + 1]
            i += 1
        elif arg.startswith("--output="):
            run_dir = arg.split("=", 1)[1]
        elif arg == "--run-dir" and i + 1 < len(args):
            warnings.warn("--run-dir is deprecated, use --output or -o instead", FutureWarning, stacklevel=2)
            run_dir = args[i + 1]
            i += 1
        elif arg.startswith("--run-dir="):
            warnings.warn("--run-dir is deprecated, use --output or -o instead", FutureWarning, stacklevel=2)
            run_dir = arg.split("=", 1)[1]
        elif arg in ("--config", "-c") and i + 1 < len(args):
            config_name = args[i + 1]
            i += 1
        elif arg.startswith("--config="):
            config_name = arg.split("=", 1)[1]
        elif arg == "--config-name" and i + 1 < len(args):
            warnings.warn("--config-name is deprecated, use --config or -c instead", FutureWarning, stacklevel=2)
            config_name = args[i + 1]
            i += 1
        elif arg.startswith("--config-name="):
            warnings.warn("--config-name is deprecated, use --config or -c instead", FutureWarning, stacklevel=2)
            config_name = arg.split("=", 1)[1]
        elif arg == "--no-auto-sample":
            auto_sample = False
        elif arg == "--timeout" and i + 1 < len(args):
            timeout = float(args[i + 1])
            i += 1
        elif arg.startswith("--timeout="):
            timeout = float(arg.split("=", 1)[1])
        elif "=" in arg and not arg.startswith("-"):
            overrides.append(arg)
        i += 1

    return mode, sample_type, config_name, run_dir, overrides, auto_sample, timeout


def main():
    """Main CLI entry point."""
    mode, sample_type, config_name, run_dir, overrides, auto_sample, timeout = parse_args()

    print(f"{BANNER} v{_get_version()}", flush=True)

    cfg = load_config(config_name, run_dir, overrides)

    if mode == "launch":
        launch_mode(cfg, auto_sample=auto_sample, timeout=timeout)
    elif mode == "graph":
        graph_mode(cfg)
    else:
        sample_mode(cfg, sample_type)


if __name__ == "__main__":
    main()
