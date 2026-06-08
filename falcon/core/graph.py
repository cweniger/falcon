import os
import re
import warnings

import numpy as np
from omegaconf import OmegaConf


class Node:
    def __init__(
        self,
        name,
        simulator_cls,
        estimator_cls=None,
        parents=None,
        evidence=None,
        scaffolds=None,
        observed=False,
        resample=False,
        simulator_config=None,
        estimator_config=None,
        actor_config=None,
        num_actors=1,
        sample_chunk_size=0,
    ):
        """Node definition for a graphical model.

        Args:
            name (str): Name of the node.
            create_distr (class): Distribution class to create the node.
            config (dict): Configuration for the distribution.
            parents (list): List of parent node names (forward model).
            evidence (list): List of evidence node names (inference model).
            observed (bool): Whether the node is observed (act as root nodes for inference model).
            actor_name (str): Optional name of the actor to deploy the node.
            resample (bool): Whether to resample the node
        """
        self.name = name

        self.simulator_cls = simulator_cls
        self.estimator_cls = estimator_cls

        self.parents = parents or []
        self.evidence = evidence or []
        self.scaffolds = scaffolds or []
        self.observed = observed
        self.resample = resample
        self.train = self.estimator_cls is not None

        self.simulator_config = simulator_config or {}
        self.estimator_config = estimator_config or {}
        self.actor_config = actor_config or {}
        self.num_actors = num_actors
        self.sample_chunk_size = sample_chunk_size


class Graph:
    def __init__(self, node_list=None):
        # Observations supplied directly as arrays via add_node(observed=array)
        self._api_observations = {}
        self._build(list(node_list) if node_list is not None else [])

    def _build(self, node_list):
        """(Re)compute all derived topology from *node_list*."""
        # Storing the node list
        self.node_list = node_list
        self.node_dict = {node.name: node for node in node_list}
        self.simulator_cls_dict = {node.name: node.simulator_cls for node in node_list}

        # Raw config data
        self.name_list = [node.name for node in node_list]
        self.evidence_dict = {node.name: node.evidence for node in node_list}
        self.scaffolds_dict = {node.name: node.scaffolds for node in node_list}
        self.observed_dict = {node.name: node.observed for node in node_list}

        # Forward graph (simulation): node -> [parent dependencies]
        self.forward_deps = {node.name: node.parents for node in node_list}
        self.forward_order = self._topological_sort(
            self.name_list, self.forward_deps
        )

        # Backward graph (inference): node -> [dependencies for inference]
        # Start with nodes that are observed or have evidence
        backward_set = {
            node.name for node in node_list if node.observed or len(node.evidence) > 0
        }
        # Expand: include deterministic ancestors reachable via evidence references
        queue = []
        for name in list(backward_set):
            for ev in self.evidence_dict[name]:
                if ev not in backward_set:
                    queue.append(ev)
        while queue:
            name = queue.pop()
            if name in backward_set:
                continue
            backward_set.add(name)
            # Guard: node may reference a not-yet-added peer
            for parent in self.forward_deps.get(name, []):
                if parent not in backward_set:
                    queue.append(parent)

        backward_names = [n.name for n in node_list if n.name in backward_set]

        # Build merged dependency dict:
        # - Observed nodes: [] (leaf nodes, values provided externally)
        # - Nodes with evidence: evidence_dict (inference direction)
        # - Deterministic intermediates: forward_deps (simulation direction)
        self.backward_deps = {}
        for name in backward_names:
            if self.observed_dict[name]:
                self.backward_deps[name] = []
            elif self.evidence_dict[name]:
                self.backward_deps[name] = self.evidence_dict[name]
            else:
                self.backward_deps[name] = self.forward_deps[name]

        self.backward_order = self._topological_sort(
            backward_names, self.backward_deps
        )

    def add_node(
        self,
        name: str,
        simulator,
        estimator=None,
        *,
        parents=None,
        evidence=None,
        scaffolds=None,
        observed=None,
        num_actors: int = 1,
        sample_chunk_size: int = 0,
        **ray_kwargs,
    ) -> "Graph":
        """Add a node to the graph and return *self* for chaining.

        Args:
            name: Node name (must be unique in this graph).
            simulator: Simulator class, string ``_target_``, or a live instance.
                Live instances (e.g. ``Product([...])``) are shipped to Ray
                actors via cloudpickle.
            estimator: Estimator class, string ``_target_``, config builder
                (e.g. ``Flow(loop_max_epochs=300)``), or ``None`` for
                observation-only nodes.
            parents: List of parent node names (forward / simulation direction).
            evidence: List of evidence node names (inference direction).
            scaffolds: List of scaffold node names.
            observed: Observed value — a numpy/torch array (used directly), or
                ``True`` / a file-path string (YAML semantics).
            num_actors: Number of Ray actors to spawn for this node.
            sample_chunk_size: Chunk size for sampling (0 = no chunking).
            **ray_kwargs: Node-level Ray actor options, prefixed with ``ray_``
                (e.g. ``ray_num_gpus=0.5``, ``ray_num_cpus=2``,
                ``ray_runtime_env={...}``).  The ``ray_`` prefix is stripped
                before passing to Ray.

        Returns:
            *self*, so calls can be chained.
        """
        import numpy as np

        # Collect actor config from ray_* kwargs
        actor_config = {}
        for key, val in ray_kwargs.items():
            if not key.startswith("ray_"):
                raise TypeError(
                    f"add_node() got unexpected keyword argument '{key}'. "
                    f"Ray actor options must be prefixed with 'ray_' "
                    f"(e.g. ray_num_gpus=0.5)."
                )
            actor_config[key[4:]] = val  # strip "ray_" prefix

        # Handle observed= as a live array
        if observed is not None and not isinstance(observed, (str, bool)):
            self._api_observations[name] = np.asarray(observed)
            observed = True  # mark as observed for the Node

        node = Node(
            name=name,
            simulator_cls=simulator,
            estimator_cls=estimator,
            parents=list(parents or []),
            evidence=list(evidence or []),
            scaffolds=list(scaffolds or []),
            observed=bool(observed) if observed is not None else False,
            actor_config=actor_config,
            num_actors=num_actors,
            sample_chunk_size=sample_chunk_size,
        )

        self._build(self.node_list + [node])
        return self

    def get_parents(self, node_name):
        return self.forward_deps[node_name]

    def get_evidence(self, node_name):
        return self.evidence_dict[node_name]

    def get_simulator_cls(self, node_name):
        return self.simulator_cls_dict[node_name]

    @staticmethod
    def _topological_sort(name_list, deps):
        """Topological sort based on dependency structure. Raises ValueError on cycles."""
        # Create a dictionary to track the number of dependencies (incoming edges) for each node
        num_parents = {node: 0 for node in name_list}

        # Count the number of dependencies for each node (incoming edges)
        for node in name_list:
            for parent in deps[node]:
                if parent in num_parents:
                    num_parents[node] += 1

        # Create a list of nodes with no parents (no incoming edges)
        no_parents = [node for node in name_list if num_parents[node] == 0]

        # List to hold the sorted nodes
        sorted_node_names = []

        while no_parents:
            node = no_parents.pop()
            sorted_node_names.append(node)

            # For each node, look at its dependents
            for child in name_list:
                if node in deps[child]:
                    num_parents[child] -= 1
                    if num_parents[child] == 0:
                        no_parents.append(child)

        # If the sorted list doesn't include all nodes, there must be a cycle
        if len(sorted_node_names) != len(name_list):
            # Print informative error message about what is going wrong exactly
            print("Sorted nodes:", sorted_node_names)
            raise ValueError("Graph has a cycle")

        return sorted_node_names

    def __add__(self, other):
        """Merge two graphs."""
        new_node_list = self.node_list + other.node_list
        return Graph(new_node_list)

    def __str__(self):
        # Return graph structure
        # - Based on topological sort
        # - Include node names and their parents in the form NAME <- PARENT1, PARENT2, ... [MODULE]
        graph_str = "Falcon graph structure:\n"
        graph_str += f"  Node name          List of parents                                 Class name\n"
        for node in self.forward_order:
            parents = self.get_parents(node)
            simulator_cls = self.get_simulator_cls(node)
            if hasattr(simulator_cls, "display_name"):
                class_name = simulator_cls.display_name
            else:
                class_name = str(simulator_cls)
            graph_str += (
                f"* {node:<15} <- {', '.join(parents):<45} | {class_name:<20}\n"
            )
        return graph_str


class Extractor:
    def __init__(self, index):
        self.index = index

    def sample(self, batch_dim, parent_conditions=[]):
        (composite,) = parent_conditions
        x = composite[self.index]
        return x


def CompositeNode(names, module, **kwargs):
    """Auxiliary function to create a composite node with multiple child nodes."""

    # Generate name of composite node from names of child nodes
    joined_names = "comp_" + "_".join(names)

    # Instantiate composite node
    node_comp = Node(joined_names, module, **kwargs)

    # Instantiate child nodes, which extract the individual components
    nodes = []
    for i, name in enumerate(names):
        node = Node(
            name, Extractor, parents=[joined_names], simulator_config=dict(index=i)
        )
        nodes.append(node)

    # Return composite node and child nodes, which both must be added to the graph
    return node_comp, *nodes


def _parse_observation_path(path: str) -> tuple:
    """Parse observation path with optional NPZ key extraction.

    Supports syntax like "file.npz['key']" to extract a specific key from NPZ.

    Args:
        path: Path string, optionally with ['key'] suffix for NPZ files

    Returns:
        Tuple of (file_path, key) where key is None for regular files
    """
    match = re.match(r"^(.+\.npz)\['([^']+)'\]$", path)
    if match:
        return match.group(1), match.group(2)
    return path, None


# Valid keys for node configuration
_VALID_NODE_KEYS = frozenset({
    "parents", "evidence", "scaffolds", "observed", "resample",
    "simulator", "estimator", "ray", "num_actors", "sample_chunk_size",
})


def _validate_node_config(node_name: str, node_config: dict) -> None:
    """Validate a node configuration, raising errors or warnings as appropriate.

    Args:
        node_name: Name of the node being validated
        node_config: Configuration dictionary for the node

    Raises:
        ValueError: If required fields are missing
    """
    # Check for unknown keys (likely typos)
    unknown_keys = set(node_config.keys()) - _VALID_NODE_KEYS
    if unknown_keys:
        warnings.warn(
            f"Node '{node_name}' has unknown config keys: {unknown_keys}. "
            f"Valid keys are: {sorted(_VALID_NODE_KEYS)}",
            UserWarning,
            stacklevel=3,
        )

    # Require simulator
    if "simulator" not in node_config:
        raise ValueError(
            f"Node '{node_name}' is missing required 'simulator' field."
        )


def _validate_node_references(nodes: list, node_names: set) -> None:
    """Validate that all node references (parents, evidence, scaffolds) exist.

    Args:
        nodes: List of Node objects
        node_names: Set of all node names in the graph

    Raises:
        ValueError: If a referenced node does not exist
    """
    for node in nodes:
        for ref_type, refs in [
            ("parents", node.parents),
            ("evidence", node.evidence),
            ("scaffolds", node.scaffolds),
        ]:
            for ref in refs:
                if ref not in node_names:
                    raise ValueError(
                        f"Node '{node.name}' references unknown {ref_type[:-1]} '{ref}'. "
                        f"Available nodes: {sorted(node_names)}"
                    )


def create_graph_from_config(graph_config, _cfg=None):
    """Create a computational graph from YAML configuration.

    Args:
        graph_config: Dictionary containing graph node definitions
        _cfg: Full Hydra configuration object (optional)

    Returns:
        Graph: The computational graph

    Raises:
        ValueError: If configuration is invalid (missing required fields, unknown references)
    """
    nodes = []
    observations = {}

    for node_name, node_config in graph_config.items():
        # Validate configuration
        _validate_node_config(node_name, node_config)

        # Extract node parameters
        parents = node_config.get("parents", [])
        evidence = node_config.get("evidence", [])
        scaffolds = node_config.get("scaffolds", [])
        observed = node_config.get(
            "observed", False
        )  # TODO: Remove from internal logic
        data_path = node_config.get("observed", None)
        resample = node_config.get("resample", False)
        actor_config = node_config.get("ray", {})
        num_actors = node_config.get("num_actors", 1)
        sample_chunk_size = node_config.get("sample_chunk_size", 0)

        if actor_config != {}:
            actor_config = OmegaConf.to_container(actor_config, resolve=True)

        if data_path is not None:
            # Parse path for NPZ key extraction syntax: "file.npz['key']"
            file_path, key = _parse_observation_path(data_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Observation file not found: {file_path}")
            data = np.load(file_path)
            if key is not None:
                # Extract specific key from NPZ
                data = data[key]
            elif hasattr(data, 'files') and len(data.files) == 1:
                # Auto-extract single-key NPZ files
                data = data[data.files[0]]
            observations[node_name] = data

        # Extract target from simulator
        simulator = node_config.get("simulator")
        if isinstance(simulator, str):
            simulator_cls = simulator
            simulator_config = {}
        else:
            simulator_cls = simulator.get("_target_")
            simulator_config = simulator
            simulator_config = OmegaConf.to_container(simulator_config, resolve=True)
            simulator_config.pop("_target_", None)

        # Extract target from infer
        if "estimator" in node_config:
            estimator = node_config.get("estimator")
            if isinstance(estimator, str):
                estimator_cls = estimator
                estimator_config = {}
            else:
                estimator_cls = estimator.get("_target_")
                estimator_config = estimator
                estimator_config = OmegaConf.to_container(
                    estimator_config, resolve=True
                )
                estimator_config.pop("_target_", None)
        else:
            estimator_cls = None
            estimator_config = {}

        # Create the node
        node = Node(
            name=node_name,
            simulator_cls=simulator_cls,
            estimator_cls=estimator_cls,
            parents=parents,
            evidence=evidence,
            scaffolds=scaffolds,
            observed=observed,
            resample=resample,
            simulator_config=simulator_config,
            estimator_config=estimator_config,
            actor_config=actor_config,
            num_actors=num_actors,
            sample_chunk_size=sample_chunk_size,
        )

        nodes.append(node)

    # Validate node references
    node_names = {node.name for node in nodes}
    _validate_node_references(nodes, node_names)

    # Create and return the graph
    return Graph(nodes), observations
