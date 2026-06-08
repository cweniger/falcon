"""Utilities for flat ↔ nested config transforms and synthesized signatures.

Used by estimator config builders (Gaussian, Flow) to provide a flat
keyword-argument surface (loop_max_epochs=300) over nested dataclass configs.
"""

import dataclasses
import inspect


def flat_to_nested(flat_kwargs: dict, sections: dict) -> dict:
    """Convert {section_field: value} to {section: {field: value}}.

    Unknown keys (e.g. 'embedding', 'device') are kept at the top level.

    Args:
        flat_kwargs: Flat keyword arguments from the user (e.g. loop_max_epochs=300).
        sections: Mapping of section_name -> dataclass type (defines valid prefixes).
    """
    nested = {}
    known_prefixes = set(sections.keys())
    for key, value in flat_kwargs.items():
        matched = False
        for section in known_prefixes:
            prefix = f"{section}_"
            if key.startswith(prefix):
                field = key[len(prefix):]
                nested.setdefault(section, {})[field] = value
                matched = True
                break
        if not matched:
            nested[key] = value
    return nested


def make_flat_signature(sections: dict, extra_params=None) -> inspect.Signature:
    """Build a flat keyword-only inspect.Signature from section_name → dataclass.

    The resulting signature has one ``prefix_fieldname`` parameter per field
    in each section dataclass. IPython and Jedi honour an explicit
    ``__signature__`` attribute, so assigning this to ``Cls.__init__.__signature__``
    gives Colab/Jupyter full autocomplete with defaults.

    Args:
        sections: Ordered dict of section_name -> dataclass type.
        extra_params: Optional list of additional inspect.Parameter objects
            appended after the section params (e.g. embedding, device).
    """
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for section, dc in sections.items():
        for f in dataclasses.fields(dc):
            name = f"{section}_{f.name}"
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = inspect.Parameter.empty  # factory defaults not shown inline
            else:
                default = inspect.Parameter.empty
            annotation = f.type if isinstance(f.type, type) else inspect.Parameter.empty
            params.append(inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            ))
    for p in (extra_params or []):
        params.append(p)
    return inspect.Signature(params)


def apply_flat_signature(cls, sections: dict, extra_params=None) -> None:
    """Assign a synthesized flat __signature__ to cls.__init__ in place."""
    sig = make_flat_signature(sections, extra_params)
    cls.__init__.__signature__ = sig
