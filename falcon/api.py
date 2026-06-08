"""Python/notebook API entry points for Falcon."""

from pathlib import Path
from typing import List, Optional, Union

from omegaconf import DictConfig, OmegaConf


class Config:
    """Thin wrapper over OmegaConf DictConfig with fluent override and display helpers."""

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg

    def override(self, *dotted_strings: str) -> "Config":
        """Return a new Config with the given dotted overrides applied.

        Example::

            cfg = falcon.config("config.yml").override(
                "buffer.max_samples=500",
                "graph.theta.estimator.loop.max_epochs=200",
            )
        """
        overrides = OmegaConf.from_dotlist(list(dotted_strings))
        return Config(OmegaConf.merge(self._cfg, overrides))

    def to_yaml(self) -> str:
        """Return the config as a YAML string."""
        return OmegaConf.to_yaml(self._cfg)

    def _repr_markdown_(self) -> str:
        return f"```yaml\n{self.to_yaml()}\n```"

    def __repr__(self) -> str:
        return f"Config(\n{self.to_yaml()})"

    @property
    def _dict_config(self) -> DictConfig:
        return self._cfg


def config(source) -> Config:
    """Load or wrap a Falcon configuration.

    Args:
        source: Path to a YAML file (str or Path), a plain dict, or a DictConfig.

    Returns:
        :class:`Config` with ``.override()`` and ``.to_yaml()`` methods.
    """
    if isinstance(source, (str, Path)):
        cfg = OmegaConf.load(source)
    elif isinstance(source, dict):
        cfg = OmegaConf.create(source)
    elif isinstance(source, DictConfig):
        cfg = source
    else:
        raise TypeError(
            f"config() expected a path, dict, or DictConfig; got {type(source).__name__}"
        )
    return Config(cfg)


# ---------------------------------------------------------------------------
# Ray lifecycle
# ---------------------------------------------------------------------------


def init(**ray_init_kwargs) -> None:
    """Connect to or start a Ray cluster.

    Idempotent: a second call is a no-op if Ray is already initialised.
    Pass any ``ray.init()`` keyword argument directly.

    Examples::

        falcon.init()                    # local cluster, auto-detect resources
        falcon.init(address="auto")      # connect to existing local cluster
        falcon.init(address="ray://...")  # connect to remote cluster
    """
    import ray
    if ray.is_initialized():
        return
    ray_init_kwargs.setdefault("namespace", "falcon")
    ray_init_kwargs.setdefault("logging_level", "ERROR")
    ray_init_kwargs.setdefault("log_to_driver", True)
    ray.init(**ray_init_kwargs)


def shutdown() -> None:
    """Shut down the Ray cluster started by :func:`falcon.init`."""
    import ray
    ray.shutdown()


# ---------------------------------------------------------------------------
# Config preparation (shared between launch() and the CLI)
# ---------------------------------------------------------------------------


def _prepare_config(
    target: Union[Config, DictConfig, dict, str, Path],
    output: Optional[Union[str, Path]],
    overrides: Optional[List[str]],
):
    """Resolve *target* into a runnable OmegaConf config with ``run_dir`` set.

    Returns:
        (cfg, output_dir_path)
    """
    from datetime import datetime
    from falcon.core.run_name import generate_run_dir

    OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

    # Determine output directory first
    if output is None:
        run_dir = generate_run_dir()
    else:
        run_dir = str(output)

    run_dir_path = Path(run_dir)
    saved_config = run_dir_path / "config.yml"

    # If resuming an existing run, use the saved config
    if saved_config.exists():
        cfg = OmegaConf.load(saved_config)
    else:
        if isinstance(target, Config):
            cfg = target._dict_config
        elif isinstance(target, DictConfig):
            cfg = target
        elif isinstance(target, dict):
            cfg = OmegaConf.create(target)
        elif isinstance(target, (str, Path)):
            cfg = OmegaConf.load(target)
        else:
            raise TypeError(
                f"launch() target must be a Config, path, or dict; got {type(target).__name__}"
            )

    # Apply overrides
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # Inject run_dir and resolve interpolations
    cfg.run_dir = run_dir
    OmegaConf.resolve(cfg)

    # Persist config so the run is reproducible
    run_dir_path.mkdir(parents=True, exist_ok=True)
    if not saved_config.exists():
        OmegaConf.save(cfg, saved_config)

    return cfg, run_dir_path


# ---------------------------------------------------------------------------
# launch()
# ---------------------------------------------------------------------------


def launch(
    target,
    output=None,
    *,
    overrides=None,
    auto_sample: bool = True,
    timeout: float = None,
    wait: bool = True,
):
    """Run a Falcon training pipeline from a notebook or script.

    Blocks by default (``wait=True``) and returns a finished :class:`Run`.
    Lazily calls :func:`falcon.init` if Ray is not yet initialised.

    Args:
        target: A :class:`Config` object, path to a YAML config file, or plain
            dict. Passing a ``Graph`` is supported in Step 5.
        output: Output directory. Auto-generated (``output/<adj-noun-date>``)
            if *None*. An existing directory with a ``config.yml`` is resumed.
        overrides: Iterable of dotted override strings applied on top of
            *target* (e.g. ``["buffer.max_samples=500"]``).
        auto_sample: Generate posterior samples after training (default True).
        timeout: Stop training gracefully after this many seconds.
        wait: Block until training completes (default True). ``wait=False`` is
            not yet implemented.

    Returns:
        :class:`falcon.core.run_loader.Run` with config, metrics, and samples.
    """
    if not wait:
        raise NotImplementedError(
            "wait=False (non-blocking launch) is not yet implemented. "
            "Use wait=True (the default) or run falcon.launch() in a thread manually."
        )

    from falcon.cli import _run_pipeline
    from falcon.core.run_loader import load_run

    cfg, output_dir = _prepare_config(target, output, overrides)

    # Lazy Ray init
    import ray
    if not ray.is_initialized():
        init()

    _run_pipeline(cfg, auto_sample=auto_sample, timeout=timeout)

    return load_run(output_dir)
