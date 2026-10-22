import json
import os
import time
import ray
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import numpy as np

from falcon.core.logger import Logger, set_logger, info, warning

from falcon.core.model_sampler import ModelSampler
from falcon.core.raystore import BufferView
from falcon.core.round_trainer import RoundTrainer
from falcon.core.state_io import has_checkpoint
from .utils import LazyLoader

_RAY_ACTOR_KEYS = {
    "accelerator_type", "memory", "name", "num_cpus", "num_gpus",
    "object_store_memory", "placement_group", "placement_group_bundle_index",
    "placement_group_capture_child_tasks", "resources", "runtime_env",
    "scheduling_strategy", "_metadata", "enable_task_events", "_labels",
    "lifetime", "max_restarts", "max_task_retries", "max_pending_calls",
    "namespace", "get_if_exists",
}  # concurrency is set by the actor classes
TRAIN_SUFFIX = "/train"


def train_stream_name(node_name: str) -> str:
    """Name of a node's train actor, its log stream and its directory under graph/."""
    return node_name + TRAIN_SUFFIX


def resolve_actor_options(actor_config: dict, has_estimator: bool) -> Tuple[dict, Optional[dict], List[str]]:
    """Split a node's ``ray:`` config into sample-actor and train-actor options.

    ``num_sample_gpus`` / ``num_train_gpus`` set the GPUs of each actor, and
    ``num_sample_cpus`` / ``num_train_cpus`` their CPUs (default: ``num_cpus``);
    every other Ray option applies to both. The deprecated ``num_gpus`` is
    split evenly between the two actors, so the node keeps its GPU budget.

    Returns:
        (sample_options, train_options or None, warnings)
    """
    cfg = dict(actor_config or {})
    sample_gpus = cfg.pop("num_sample_gpus", None)
    train_gpus = cfg.pop("num_train_gpus", None)
    legacy_gpus = cfg.pop("num_gpus", None)
    sample_cpus = cfg.pop("num_sample_cpus", None)
    train_cpus = cfg.pop("num_train_cpus", None)
    notes = []

    if legacy_gpus is not None:
        if sample_gpus is not None or train_gpus is not None:
            raise ValueError(
                "ray.num_gpus cannot be combined with ray.num_sample_gpus / ray.num_train_gpus; "
                "use only the latter"
            )
        if has_estimator:
            sample_gpus = train_gpus = legacy_gpus / 2
            notes.append(
                f"ray.num_gpus is deprecated; using num_train_gpus={train_gpus:g} and "
                f"num_sample_gpus={sample_gpus:g}"
            )
        else:
            sample_gpus = legacy_gpus
            notes.append(f"ray.num_gpus is deprecated; using num_sample_gpus={sample_gpus:g}")

    if not has_estimator:
        for key, value in (("num_train_gpus", train_gpus), ("num_train_cpus", train_cpus)):
            if value is not None:
                notes.append(f"ray.{key} is ignored: the node has no estimator")

    unknown = sorted(set(cfg) - _RAY_ACTOR_KEYS)
    if unknown:
        notes.append(f"ignoring unsupported ray options {unknown}")
    common = {k: v for k, v in cfg.items() if k in _RAY_ACTOR_KEYS}

    sample = dict(common)
    if sample_gpus is not None:
        sample["num_gpus"] = sample_gpus
    if sample_cpus is not None:
        sample["num_cpus"] = sample_cpus
    if not has_estimator:
        return sample, None, notes

    train = dict(common)
    if train_gpus is not None:
        train["num_gpus"] = train_gpus
    if train_cpus is not None:
        train["num_cpus"] = train_cpus
    if "name" in common:
        train["name"] = train_stream_name(common["name"])
    return sample, train, notes


_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _local_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS
        return os.cpu_count() or 1


def _init_actor(import_dirs, log_config, log_name, num_threads=None):
    """Import paths, thread limits and module-level logger of an actor process."""
    if num_threads is not None:
        # Takes effect for libraries that are initialized after this point
        for var in _THREAD_ENV_VARS:
            os.environ[var] = str(num_threads)
    for p in (import_dirs or []):
        resolved = str(Path(p).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    # Enables falcon.log(), falcon.info() etc. for simulators and estimators
    logger = Logger(log_name, log_config or {"local": {"enabled": True, "dir": "."}},
                    capture_exceptions=True)
    set_logger(logger)
    return logger


def _make_simulator(node):
    # Live instances (from Python API) are used directly; string / class
    # paths go through LazyLoader for deferred import + instantiation.
    if isinstance(node.simulator_cls, (str, type)):
        return LazyLoader(node.simulator_cls)(**node.simulator_config)
    return node.simulator_cls


def _make_model(node, simulator_instance, num_threads=None):
    """Set-up estimator model of the node, or None for simulator-only nodes."""
    if node.estimator_cls is None:
        return None
    from falcon.core.base_estimator import BaseEstimator
    if isinstance(node.estimator_cls, BaseEstimator):
        # Notebook path: already a configured instance (e.g. Flow(max_epochs=200));
        # each actor process receives its own copy
        model = node.estimator_cls
    elif isinstance(node.estimator_cls, (str, type)):
        # YAML path: pass flat config dict as kwargs to __init__
        model = LazyLoader(node.estimator_cls)(**node.estimator_config)
    else:
        raise TypeError(
            f"estimator_cls must be a BaseEstimator instance, class, or "
            f"string; got {type(node.estimator_cls).__name__}"
        )
    if num_threads is not None:
        model.set_num_threads(num_threads)
    model.setup(simulator_instance, theta_key=node.name,
                condition_keys=node.evidence + node.scaffolds)
    return model


def _artifact(node) -> dict:
    """Targets and config needed to rebuild the node's posterior from its checkpoint.

    Empty when the node uses live objects or config values JSON cannot store.
    """
    if not (isinstance(node.estimator_cls, str) and isinstance(node.simulator_cls, str)):
        return {}
    artifact = {
        "estimator": {"_target_": node.estimator_cls, **node.estimator_config},
        "simulator": {"_target_": node.simulator_cls, **node.simulator_config},
    }
    try:
        json.dumps(artifact)
    except TypeError:
        return {}
    return artifact


@ray.remote(concurrency_groups={"control": 1})
class SampleActor:
    """Serves prior, proposal and posterior samples of one node.

    For estimator nodes it holds the best networks, which the train actor
    publishes through ``set_state``. Status and log calls run on their own
    thread, so they are answered while a sampling call runs.
    """

    def __init__(self, node, import_dirs=None, log_config=None, log_name=None, num_threads=None):
        self.node = node
        self.name = node.name
        self._logger = _init_actor(import_dirs, log_config, log_name or node.name, num_threads)
        self._status = "initializing"

        self.simulator_instance = _make_simulator(node)
        self.model = _make_model(node, self.simulator_instance, num_threads)
        self.sampler = ModelSampler(self.model) if self.model is not None else None
        self.rng = np.random.default_rng()
        self.parents = node.parents
        self._status = "idle"

    def set_state(self, tree):
        """Install a state published by the train actor."""
        self.sampler.apply(tree)

    # ==================== Ref/Array Boundary ====================

    def _resolve_refs(self, condition_refs):
        """Convert refs to arrays. Single batched ray.get.

        Detects broadcast refs (all identical) and resolves once,
        returning shape (1, ...) so consumers can expand efficiently
        (e.g. torch .expand() on GPU, or np.broadcast_to on CPU).

        Args:
            condition_refs: Dict[str, List[ObjectRef]] or None

        Returns:
            Dict[str, ndarray] — broadcast entries have shape (1, ...),
            non-broadcast have shape (N, ...). Empty dict if no refs.
        """
        if not condition_refs:
            return {}
        # Flatten refs, detecting broadcast (all-same-ref) lists
        all_refs = []
        slices = {}
        for name, refs in condition_refs.items():
            if len(refs) == 0:
                raise ValueError(
                    f"_resolve_refs: empty ref list for '{name}' "
                    f"(zero-sample dispatch? n_samples < num_actors?)")
            if len(set(refs)) == 1:  # all same ref → broadcast
                slices[name] = ('broadcast', len(all_refs))
                all_refs.append(refs[0])
            else:
                slices[name] = ('full', len(all_refs), len(all_refs) + len(refs))
                all_refs.extend(refs)
        all_values = ray.get(all_refs)  # ONE call resolves everything
        result = {}
        for name, info in slices.items():
            if info[0] == 'broadcast':
                val = all_values[info[1]]
                result[name] = np.asarray(val)[np.newaxis]  # (1, ...) — compact, consumer expands
            else:
                result[name] = np.stack(all_values[info[1]:info[2]])
        return result

    def _batch_to_refs(self, output):
        """Convert output dict to list of per-sample ref dicts.

        Uses ThreadPoolExecutor to parallelize ray.put calls (GIL released
        during serialization).

        Args:
            output: {'value': arr[N,...], 'log_prob': arr[N], ...}

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with flat keys
                e.g. [{'theta.value': ref, 'theta.log_prob': ref}, ...]
        """
        value = output['value']
        n = value.shape[0] if isinstance(value, np.ndarray) else len(next(iter(value.values())))
        # Phase 1: fire all ray.put calls in parallel via thread pool
        ref_columns = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for key, arr in output.items():
                full_key = f"{self.name}.{key}"
                if isinstance(arr, np.ndarray):
                    ref_columns[full_key] = list(pool.map(ray.put, [arr[i] for i in range(n)]))
                else:  # dict-valued (CompositeNode)
                    ref_columns[full_key] = list(pool.map(
                        ray.put, [{k: v[i] for k, v in arr.items()} for i in range(n)]
                    ))
        # Phase 2: assemble per-sample dicts
        return [{k: refs[i] for k, refs in ref_columns.items()} for i in range(n)]

    def _chunked_sample(self, n_samples, condition_refs, mode):
        """Resolve refs, chunk, sample, return refs.

        Broadcast conditions (shape[0]==1) pass through without slicing,
        letting consumers expand efficiently (GPU expand or np.broadcast_to).

        Args:
            n_samples: Number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] or None
            mode: "prior", "proposal" or "posterior"

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample
        """
        conditions = self._resolve_refs(condition_refs)
        chunk_size = getattr(self.node, 'sample_chunk_size', 0) or n_samples
        result = []
        self._status = "sampling"
        try:
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                chunk = {
                    k: v if v.shape[0] == 1 else v[start:end]
                    for k, v in conditions.items()
                } if conditions else None
                output = self._sample_chunk(end - start, chunk, mode)
                result.extend(self._batch_to_refs(output))
        finally:
            self._status = "idle"
        return result

    def _sample_chunk(self, n_samples, conditions, mode):
        if self.sampler is None:
            return self._simulate(n_samples, conditions)
        return self.sampler.sample(mode, n_samples, conditions or None, self.rng)

    def _simulate(self, n_samples, conditions=None):
        """Call the simulator with resolved arrays.

        Expands broadcast conditions (shape[0]==1) via np.broadcast_to
        so simulators see per-sample arrays.

        Returns:
            dict: {'value': ndarray}
        """
        if conditions:
            conditions = {
                k: np.broadcast_to(v, (n_samples,) + v.shape[1:]) if v.shape[0] == 1 else v
                for k, v in conditions.items()
            }
        # Simulator: extract parent arrays as positional args
        incoming = [conditions[p] for p in self.parents] if conditions else []
        if hasattr(self.simulator_instance, "simulate_batch"):
            value = self.simulator_instance.simulate_batch(n_samples, *incoming)
        else:
            value = np.stack([self.simulator_instance.simulate(*[v[i] for v in incoming])
                              for i in range(n_samples)])
        return {'value': value}

    # ==================== Public Sampling Methods ====================

    def sample(self, n_samples, condition_refs=None):
        """Forward (prior) samples as one dict of ObjectRefs per sample."""
        return self._chunked_sample(n_samples, condition_refs, "prior")

    def sample_posterior(self, n_samples, condition_refs=None):
        """Posterior samples as one dict of ObjectRefs per sample."""
        return self._chunked_sample(n_samples, condition_refs, "posterior")

    def sample_proposal(self, n_samples, condition_refs=None):
        """Proposal samples as one dict of ObjectRefs per sample."""
        return self._chunked_sample(n_samples, condition_refs, "proposal")

    def load(self, node_dir):
        if self.sampler is None:
            return False
        return self.sampler.load(Path(node_dir))

    # ==================== Control ====================

    @ray.method(concurrency_group="control")
    def get_status(self) -> dict:
        status = {"name": self.name, "status": self._status, "samples": 0}
        if self.sampler is not None:
            if self._status == "idle":
                status["status"] = self.sampler.status
            status["round"] = self.sampler.best_round
            status["samples"] = self.sampler.samples_served
        return status

    @ray.method(concurrency_group="control")
    def get_output_log_tail(self, num_lines: int = 50) -> list:
        return self._logger.get_output_log_tail(num_lines)

    @ray.method(concurrency_group="control")
    def shutdown(self):
        self._logger.shutdown()


@ray.remote(concurrency_groups={"control": 1})
class TrainActor:
    """Trains the networks of one estimator node (named ``<node>/train``).

    After every promotion it publishes the best state to the node's sample
    actors and waits until they have installed it. Stop, status and log calls
    run on their own thread, so they are answered while training runs.
    """

    def __init__(self, node, import_dirs=None, log_config=None, num_threads=None):
        self.node = node
        self.name = train_stream_name(node.name)
        self._logger = _init_actor(import_dirs, log_config, self.name, num_threads)
        self.model = _make_model(node, _make_simulator(node), num_threads)
        meta = {
            "theta_key": node.name,
            "condition_keys": list(node.evidence + node.scaffolds),
            "artifact": _artifact(node),
        }
        self.trainer = RoundTrainer(self.model, publish=self._publish, meta=meta)
        self._samplers = []

    def _publish(self, tree):
        # One copy in the object store, shared by all sample actors
        ref = ray.put(tree)
        ray.get([s.set_state.remote(ref) for s in self._samplers])

    def train(self, dataset_manager, samplers):
        self._samplers = list(samplers)
        info(f"[{self.name}] Training started")
        buffer = BufferView(dataset_manager, cache_device=self.model.cache_device)
        self.trainer.run(buffer)
        loss = self.trainer.best_val_loss
        if loss is not None:
            info(f"[{self.name}] Training completed (loss: {loss:.4f})")
        else:
            info(f"[{self.name}] Training completed")

    def save(self, node_dir):
        return self.trainer.save(Path(node_dir))

    def load(self, node_dir):
        return self.trainer.load(Path(node_dir))

    # ==================== Control ====================

    @ray.method(concurrency_group="control")
    def request_stop(self):
        """Stop after the current training step; the acceptance test still runs."""
        self.trainer.request_stop()

    @ray.method(concurrency_group="control")
    def get_status(self) -> dict:
        trainer = self.trainer
        # val_loss is NaN for epochs without a validation
        val_losses = [v for v in trainer.history["val_loss"][-40:] if v == v][-20:]
        n_samples = trainer.history["n_samples"]
        return {
            "name": self.name,
            "status": trainer.status,
            "samples": n_samples[-1] if n_samples else 0,
            "round": trainer.round,
            "rounds_accepted": trainer.rounds_accepted,
            "current_epoch": trainer.round_epoch,
            "total_epochs": trainer.config.max_epochs,
            "loss": val_losses[-1] if val_losses else None,
            "best_loss": trainer.best_val_loss,
            "loss_history": val_losses,
        }

    @ray.method(concurrency_group="control")
    def get_output_log_tail(self, num_lines: int = 50) -> list:
        return self._logger.get_output_log_tail(num_lines)

    @ray.method(concurrency_group="control")
    def shutdown(self):
        self._logger.shutdown()


class SamplePool:
    """The sample actors of one node; splits sampling calls across them."""

    def __init__(self, actors):
        self.actors = list(actors)

    def call(self, method_name, n_samples, condition_refs=None):
        """Distribute a sampling call across the actors, slicing ref lists.

        Args:
            method_name: Name of the method to call on each actor
            n_samples: Total number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] or None

        Returns:
            List[Dict[str, ObjectRef]]: Concatenated results from all actors
        """
        num_actors = len(self.actors)
        per_actor = n_samples / num_actors
        ranges = [(int(i * per_actor), int((i + 1) * per_actor)) for i in range(num_actors)]
        ranges[-1] = (ranges[-1][0], n_samples)

        futures = []
        for actor, (start, end) in zip(self.actors, ranges):
            if end - start <= 0:
                # n_samples < num_actors leaves some actors with an empty
                # slice; a zero-sample dispatch is rejected by _resolve_refs
                # (explicit ValueError). Skip idle actors instead.
                continue
            chunk_refs = {k: v[start:end] for k, v in condition_refs.items()} if condition_refs else None
            futures.append(getattr(actor, method_name).remote(end - start, condition_refs=chunk_refs))
        result = []
        for sample_list in ray.get(futures):
            result.extend(sample_list or [])
        return result

    def status_ref(self):
        return self.actors[0].get_status.remote()

    def load(self, node_dir):
        return ray.get([a.load.remote(node_dir) for a in self.actors])

    def shutdown_refs(self):
        return [a.shutdown.remote() for a in self.actors]


class DeployedGraph:
    def __init__(self, graph, import_dirs=None, log_config=None, train=True):
        """Deploy a graph as Ray actors.

        Every node gets ``num_actors`` sample actors; estimator nodes also get
        one train actor, unless ``train`` is False (sampling only).

        Note: This class uses falcon.info(), falcon.warning() etc. for logging.
        These functions use the module-level logger set by cli.py via set_logger().
        """
        self.graph = graph
        self.import_dirs = import_dirs or []
        self.log_config = log_config or {}
        self.train = train
        self.samplers: Dict[str, SamplePool] = {}
        self.trainers: Dict[str, object] = {}  # node name -> TrainActor handle
        self._dataset_manager_actor = None

        self.deploy_nodes()

    def _actor_options(self):
        """(sample_options, train_options) per node; train options only if training."""
        options = {}
        for node in self.graph.node_list:
            sample, train, notes = resolve_actor_options(node.actor_config, node.estimator_cls is not None)
            for note in notes:
                warning(f"[{node.name}] {note}")
            options[node.name] = (sample, train if self.train else None)
        return options

    def _thread_counts(self, options):
        """CPU threads of each estimator actor: its num_cpus, or an even share of this machine.

        The train and sample actors of a node compute at the same time, so
        each one gets its own part of the cores instead of all of them.
        Simulator-only actors are limited only when they set num_cpus.
        """
        heavy = 0
        for node in self.graph.node_list:
            if node.estimator_cls is not None:
                heavy += node.num_actors + (options[node.name][1] is not None)
        share = max(1, _local_cpu_count() // max(1, heavy))

        def threads(opts, is_estimator):
            if "num_cpus" in opts:
                return max(1, int(opts["num_cpus"]))
            return share if is_estimator else None

        counts = {}
        for node in self.graph.node_list:
            sample, train = options[node.name]
            is_estimator = node.estimator_cls is not None
            counts[node.name] = (threads(sample, is_estimator),
                                 threads(train, True) if train is not None else None)
        return counts

    def _check_resource_budget(self, options):
        """Warn if actor GPU/CPU requests exceed cluster capacity."""
        cluster = ray.cluster_resources()
        available_gpus = cluster.get("GPU", 0)
        available_cpus = cluster.get("CPU", 0)

        requests = []  # (actor name, gpus, cpus)
        for node in self.graph.node_list:
            sample, train = options[node.name]
            for _ in range(node.num_actors):
                requests.append((node.name, sample.get("num_gpus", 0), sample.get("num_cpus", 1)))
            if train is not None:
                requests.append((train_stream_name(node.name), train.get("num_gpus", 0), train.get("num_cpus", 1)))

        total_gpus = sum(r[1] for r in requests)
        total_cpus = sum(r[2] for r in requests)
        if total_gpus > available_gpus:
            summary = ", ".join(f"{name}: {gpus} GPU" for name, gpus, _ in requests if gpus > 0)
            warning(
                f"GPU over-subscription: actors request {total_gpus:.1f} GPUs "
                f"but only {available_gpus:.1f} available. "
                f"Actors may hang — reduce ray.num_train_gpus / ray.num_sample_gpus in your "
                f"config or increase available GPUs. ({summary})"
            )
        if total_cpus > available_cpus:
            warning(
                f"CPU over-subscription: actors request {total_cpus} CPUs "
                f"but only {available_cpus:.0f} available — actors may queue."
            )

    def deploy_nodes(self):
        """Deploy all nodes in the graph as Ray actors."""
        info("Spinning up graph...")
        options = self._actor_options()
        self._check_resource_budget(options)
        threads = self._thread_counts(options)

        # Create all actors (non-blocking)
        ready = {}  # display name -> list of ready refs
        labels = {}  # display name -> thread note
        for node in self.graph.node_list:
            sample_opts, train_opts = options[node.name]
            sample_threads, train_threads = threads[node.name]
            actors = []
            for i in range(node.num_actors):
                opts = dict(sample_opts)
                log_name = node.name if i == 0 else f"{node.name}/{i}"
                if i > 0 and "name" in opts:
                    opts["name"] = f"{opts['name']}/{i}"
                actors.append(SampleActor.options(**opts).remote(
                    node, self.import_dirs, self.log_config, log_name, sample_threads
                ))
            self.samplers[node.name] = SamplePool(actors)
            ready[node.name] = [a.__ray_ready__.remote() for a in actors]
            if sample_threads is not None:
                labels[node.name] = f" ({sample_threads} CPU threads)"
            if train_opts is not None:
                trainer = TrainActor.options(**train_opts).remote(
                    node, self.import_dirs, self.log_config, train_threads
                )
                self.trainers[node.name] = trainer
                ready[train_stream_name(node.name)] = [trainer.__ray_ready__.remote()]
                labels[train_stream_name(node.name)] = f" ({train_threads} CPU threads)"

        # Wait for all actors to initialize
        for name, refs in ready.items():
            try:
                done, _ = ray.wait(refs, num_returns=len(refs), timeout=60.0)
                if len(done) < len(refs):
                    raise RuntimeError(
                        f"Actor '{name}' did not initialize within 60 s. "
                        "This usually means Ray cannot schedule the actor — "
                        "check that ray.num_train_gpus / num_sample_gpus / num_cpus in your "
                        "config do not exceed available cluster resources."
                    )
                ray.get(done)  # re-raise any actor-side exception
                info(f"  ✓ {name}{labels.get(name, '')}")
            except ray.exceptions.RayActorError as e:
                raise RuntimeError(f"Failed to initialize actor '{name}': {e}") from e

    def _merge_refs(self, sample_refs, node_refs):
        """Merge node refs into sample refs list.

        Args:
            sample_refs: List[Dict[str, ObjectRef]] - accumulator
            node_refs: List[Dict[str, ObjectRef]] - new refs from one node
        """
        if not sample_refs:
            # First node: initialize with node_refs
            sample_refs.extend(node_refs)
        else:
            # Merge keys from node_refs into existing sample dicts
            for i, ref_dict in enumerate(node_refs):
                sample_refs[i].update(ref_dict)

    def _arrays_to_condition_refs(self, conditions, num_samples):
        """Convert arrays/tensors to per-sample ObjectRefs.

        Handles broadcast: arrays with shape[0]==1 use a single ref repeated.

        Args:
            conditions: Dict[str, ndarray/Tensor]
            num_samples: Number of samples

        Returns:
            Dict[str, List[ObjectRef]]
        """
        if not conditions:
            return {}
        result = {}
        for name, arr in conditions.items():
            # Only numpy crosses actor boundaries
            arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
            if arr.shape[0] == 1:
                ref = ray.put(arr[0])
                result[name] = [ref] * num_samples
            else:
                result[name] = [ray.put(arr[i]) for i in range(arr.shape[0])]
        return result

    def _extract_value_refs(self, sample_refs):
        """Extract .value refs grouped by node name from sample_refs.

        Args:
            sample_refs: List[Dict[str, ObjectRef]] with flat keys like 'theta.value'

        Returns:
            Dict[str, List[ObjectRef]] keyed by node name
        """
        if not sample_refs:
            return {}
        result = {}
        for key in sample_refs[0].keys():
            if key.endswith('.value'):
                node_name = key[:-6]
                result[node_name] = [d[key] for d in sample_refs]
        return result

    def _execute_graph(self, num_samples, node_order, condition_refs, sample_method):
        """Execute graph traversal with specified sampling method.

        All data flows as ObjectRefs - no array resolution at this layer.

        Args:
            num_samples: Number of samples to generate
            node_order: Node names in execution order
            condition_refs: Dict[str, List[ObjectRef]] for pre-set nodes
            sample_method: One of "sample", "sample_posterior", "sample_proposal"

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        ref_trace = dict(condition_refs)  # {name: [ObjectRef, ...]}
        sample_refs = []

        for name in node_order:
            if name in ref_trace:
                # FIXME: conditioned nodes are skipped, so their refs are missing
                # from sample_refs. Callers must manually merge them back (see
                # training loop). _execute_graph should insert them directly.
                continue

            # Build condition refs for this node (parents always, evidence for inference)
            node_condition_refs = {}
            for parent in self.graph.get_parents(name):
                node_condition_refs[parent] = ref_trace[parent]
            if sample_method != "sample":
                for evidence in self.graph.get_evidence(name):
                    node_condition_refs[evidence] = ref_trace[evidence]

            node_refs = self.samplers[name].call(sample_method, num_samples, node_condition_refs)

            # Update trace with value refs for downstream nodes
            ref_trace[name] = [d[f'{name}.value'] for d in node_refs]

            self._merge_refs(sample_refs, node_refs)

        return sample_refs

    def sample(self, num_samples, conditions=None):
        """Run forward sampling through the graph.

        Args:
            num_samples: Number of samples to generate
            conditions: Optional dict of pre-set conditions (arrays/tensors)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        condition_refs = self._arrays_to_condition_refs(conditions, num_samples) if conditions else {}
        return self._execute_graph(
            num_samples, self.graph.forward_order, condition_refs, "sample",
        )

    def sample_posterior(self, num_samples, conditions=None):
        """Run posterior sampling through the inference graph.

        Args:
            num_samples: Number of samples to generate
            conditions: Optional dict of pre-set conditions (arrays/tensors)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        condition_refs = self._arrays_to_condition_refs(conditions, num_samples) if conditions else {}
        return self._execute_graph(
            num_samples, self.graph.backward_order, condition_refs, "sample_posterior",
        )

    def sample_proposal(self, num_samples, conditions=None):
        """Run proposal sampling through the inference graph.

        Args:
            num_samples: Number of samples to generate
            conditions: Optional dict of pre-set conditions (arrays/tensors)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        condition_refs = self._arrays_to_condition_refs(conditions, num_samples) if conditions else {}
        return self._execute_graph(
            num_samples, self.graph.backward_order, condition_refs, "sample_proposal",
        )

    def sample_ppd(self, num_samples, conditions=None):
        """Run posterior predictive distribution (PPD) sampling.

        Two-phase: sample latent variables from the posterior, then forward-simulate
        observables from those posterior samples.

        Args:
            num_samples: Number of samples to generate
            conditions: Observations dict (same as passed to sample_posterior)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
                (both posterior latents and forward-simulated observables)
        """
        observation_refs = self._arrays_to_condition_refs(conditions, num_samples) if conditions else {}

        # Phase 1: theta ~ p(theta | x_obs)
        posterior_refs = self._execute_graph(
            num_samples, self.graph.backward_order, observation_refs, "sample_posterior",
        )

        # Phase 2: x_ppd ~ p(x | theta)  — forward-simulate fresh observables
        # Condition on posterior theta; observed nodes are NOT pre-set here so they
        # get re-simulated rather than returning the original observations.
        posterior_condition_refs = self._extract_value_refs(posterior_refs)
        forward_refs = self._execute_graph(
            num_samples, self.graph.forward_order, posterior_condition_refs, "sample",
        )

        # Merge: posterior_refs holds theta.value; forward_refs holds x_ppd.value.
        # Conditioned (theta) nodes are absent from forward_refs due to the skip in
        # _execute_graph, so update is safe — no key collisions.
        merged = [dict(p) for p in posterior_refs]
        for i, fwd_dict in enumerate(forward_refs):
            merged[i].update(fwd_dict)
        return merged

    def _refs_to_arrays(self, sample_refs):
        """Convert List[Dict[str, ObjectRef]] to Dict[str, ndarray].

        Uses a single batched ray.get for efficiency.
        """
        if not sample_refs:
            return {}
        keys = list(sample_refs[0].keys())
        # Flatten all refs across all keys into one list
        all_refs = []
        key_slices = {}
        for key in keys:
            refs = [d[key] for d in sample_refs if key in d]
            key_slices[key] = (len(all_refs), len(all_refs) + len(refs))
            all_refs.extend(refs)
        all_values = ray.get(all_refs)  # ONE call
        return {key: np.stack(all_values[start:end]) for key, (start, end) in key_slices.items()}

    def get_status(self, timeout: float = 2.0) -> dict:
        """Status of all actors, keyed by actor name (``z``, ``z/train``), and of the buffer."""
        refs = {}
        for node in self.graph.node_list:
            refs[node.name] = self.samplers[node.name].status_ref()
            if node.name in self.trainers:
                refs[train_stream_name(node.name)] = self.trainers[node.name].get_status.remote()
        nodes = {}
        for name, ref in refs.items():
            try:
                nodes[name] = ray.get(ref, timeout=timeout)
            except Exception as e:
                nodes[name] = {"status": "error", "error": str(e)}
        buffer = {}
        if self._dataset_manager_actor is not None:
            try:
                buffer = ray.get(self._dataset_manager_actor.get_store_stats.remote(), timeout=timeout)
            except Exception:
                pass
        return {"nodes": nodes, "buffer": buffer}

    def shutdown(self):
        """Shut down the deployed graph and release resources."""
        refs = [t.shutdown.remote() for t in self.trainers.values()]
        for pool in self.samplers.values():
            refs.extend(pool.shutdown_refs())
        ray.get(refs)

    def launch(self, dataset_manager, observations, graph_path=None, stop_check=None):
        """Launch training.

        Args:
            dataset_manager: Dataset manager for samples
            observations: Observation data
            graph_path: Path to save/load graph
            stop_check: Optional callable that returns True when graceful stop is requested
        """
        self._launch(dataset_manager, observations, graph_path=graph_path, stop_check=stop_check)

    def _launch(self, dataset_manager, observations, graph_path=None, stop_check=None):
        # Resume if saved checkpoints exist (not just logging directories)
        if graph_path is not None and any(
            has_checkpoint(Path(graph_path) / name) for name in self.graph.node_dict
        ):
            self.load(graph_path)

        dataset_manager = dataset_manager.dataset_manager_actor
        self._dataset_manager_actor = dataset_manager

        # Initial data generation: load from disk first, then generate remaining
        # Nodes handle chunking internally based on their sample_chunk_size config
        num_initial = ray.get(dataset_manager.num_initial_samples.remote())
        num_loaded = ray.get(dataset_manager.load_initial_samples.remote())
        num_to_generate = num_initial - num_loaded
        if num_to_generate > 0:
            info(f"Generating {num_to_generate} initial samples...")
            # sample() returns List[Dict[str, ObjectRef]], nodes handle chunking
            sample_refs = self.sample(num_to_generate)
            ray.get(dataset_manager.append_refs.remote(sample_refs))
        info(f"Initial samples ready ({num_loaded} loaded, {num_to_generate} generated)")

        info("")
        info("Starting analysis.")

        # Training - start all train actors; each publishes to its node's sample actors
        train_futures = {}  # Map future -> node_name for completion tracking
        for name, trainer in self.trainers.items():
            train_future = trainer.train.remote(dataset_manager, self.samplers[name].actors)
            train_futures[train_future] = name
            info(f"[{train_stream_name(name)}] Training started")
            time.sleep(1)

        simulate_interval = ray.get(dataset_manager.get_simulate_interval.remote())

        # Track last status log time for periodic updates
        last_status_log = time.time()
        STATUS_LOG_INTERVAL = 60  # seconds

        latent_nodes = {n.name for n in self.graph.node_list if n.estimator_cls is not None}
        train_future_list = list(train_futures.keys())
        stop_requested = False
        pending_append = None
        while train_future_list:
            # Check for graceful stop request
            if not stop_requested and stop_check is not None and stop_check():
                info("Graceful stop requested, finishing the current round's acceptance test...")
                stop_requested = True
                for trainer in self.trainers.values():
                    try:
                        ray.get(trainer.request_stop.remote(), timeout=10)
                    except Exception:
                        pass  # the trainer may have finished already

            # Short poll: the loop's own pacing comes from the time.sleep()
            # below, so a long timeout here just adds to simulate_interval.
            ready, train_future_list = ray.wait(
                train_future_list, num_returns=len(train_future_list), timeout=0.05
            )

            # Skip simulation if stopping
            if not stop_requested:
                time.sleep(simulate_interval)
                num_new_samples = ray.get(dataset_manager.num_resims.remote())
                if num_new_samples > 0:
                    # Proposals come from the sample actors, which run
                    # independently of training.
                    proposal_refs = self.sample_proposal(num_new_samples, observations)
                    condition_refs = self._extract_value_refs(proposal_refs)
                    # Only keep latent nodes (with estimators) from proposal.
                    # Deterministic intermediates and observed nodes must be
                    # re-simulated to maintain data consistency.
                    condition_refs = {k: v for k, v in condition_refs.items()
                                      if k in latent_nodes}
                    sample_refs = self._execute_graph(
                        num_new_samples, self.graph.forward_order, condition_refs, "sample"
                    )
                    # Only merge latent node values from proposal into
                    # sample_refs.  Deterministic intermediates (e.g. tokens)
                    # were correctly re-simulated in _execute_graph above and
                    # must not be overwritten with observation-based values.
                    for i, prop_ref in enumerate(proposal_refs):
                        sample_refs[i].update(
                            {k: v for k, v in prop_ref.items()
                             if k.split(".")[0] in latent_nodes}
                        )
                    # Pipeline the append: block on the *previous* one, then
                    # fire this one and let it overlap the next simulation.
                    if pending_append is not None:
                        ray.get(pending_append)
                    pending_append = dataset_manager.append_refs.remote(sample_refs)

            # Periodic status update (every ~60 seconds)
            now = time.time()
            if now - last_status_log >= STATUS_LOG_INTERVAL:
                last_status_log = now
                self._log_status(dataset_manager)

            # Log completed training nodes
            for completed_task in ready:
                ray.get(completed_task)  # Retrieve result or raise exception
                node_name = train_futures.get(completed_task)
                if node_name:
                    status = ray.get(self.trainers[node_name].get_status.remote())
                    loss = status.get("best_loss")
                    name = train_stream_name(node_name)
                    if loss is not None:
                        info(f"[{name}] Training completed (loss: {loss:.4f})")
                    else:
                        info(f"[{name}] Training completed")

        # Flush the last deferred buffer append before shutdown.
        if pending_append is not None:
            ray.get(pending_append)

        # Save graph if path is provided
        if graph_path is not None:
            self.save(graph_path)

        info("")
        info("Analysis completed.")

    def _log_status(self, dataset_manager):
        """Log periodic status of active train actors and buffer (separate lines)."""
        for name, trainer in self.trainers.items():
            status = ray.get(trainer.get_status.remote())
            if status["status"] == "training":
                rnd = status.get("round", 0)
                epoch = status.get("current_epoch", 0)
                total = status.get("total_epochs", 0)
                loss = status.get("loss")
                loss_str = f"{loss:.2f}" if loss is not None else "?"
                info(f"[{train_stream_name(name)}] round {rnd}, epoch {epoch}/{total}, loss {loss_str}")

        # Log buffer stats (including total ever simulated)
        stats = ray.get(dataset_manager.get_store_stats.remote())
        info(f"Buffer: {stats['training']} train, {stats['validation']} val ({stats['total_length']} total)")

    def save(self, graph_dir):
        """Save the checkpoints of all trained nodes."""
        graph_dir = Path(graph_dir).expanduser().resolve()
        graph_dir.mkdir(parents=True, exist_ok=True)
        ray.get([trainer.save.remote(graph_dir / name) for name, trainer in self.trainers.items()])

    def load(self, graph_dir):
        """Load the checkpoints of all estimator nodes into their actors."""
        info(f"Loading deployed graph from: {graph_dir}")
        refs = []
        for node in self.graph.node_list:
            if node.estimator_cls is None:
                continue
            node_dir = Path(graph_dir) / node.name
            refs.extend(a.load.remote(node_dir) for a in self.samplers[node.name].actors)
            if node.name in self.trainers:
                refs.append(self.trainers[node.name].load.remote(node_dir))
        ray.get(refs)
