"""Python/notebook API entry points for Falcon."""

from pathlib import Path

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
