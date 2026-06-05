import time
import ray
import asyncio
import torch
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from omegaconf import ListConfig

from falcon.core.logger import Logger, set_logger, debug, info, warning, error, log

from falcon.core.raystore import BufferView
from .utils import LazyLoader

_RAY_ACTOR_KEYS = {
    "accelerator_type", "memory", "name", "num_cpus", "num_gpus",
    "object_store_memory", "placement_group", "placement_group_bundle_index",
    "placement_group_capture_child_tasks", "resources", "runtime_env",
    "scheduling_strategy", "_metadata", "enable_task_events", "_labels",
    "concurrency_groups", "lifetime", "max_concurrency", "max_restarts",
    "max_task_retries", "max_pending_calls", "namespace", "get_if_exists",
}


def _ray_options(actor_config):
    """Filter actor_config to only valid Ray actor options."""
    return {k: v for k, v in actor_config.items() if k in _RAY_ACTOR_KEYS}


@ray.remote
class MultiplexNodeWrapper:
    def __init__(self, actor_config, node, graph, num_actors, model_path=None, log_config=None):
        self.num_actors = num_actors
        self.wrapped_node_list = [
            NodeWrapper.options(**_ray_options(actor_config)).remote(node, graph, model_path, log_config)
            for _ in range(self.num_actors)
        ]

    def _multiplexed_call(self, method_name, n_samples, condition_refs=None):
        """Distribute work across actors, slicing ref lists.

        Args:
            method_name: Name of the method to call on each actor
            n_samples: Total number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] or None

        Returns:
            List[Dict[str, ObjectRef]]: Concatenated results from all actors
        """
        num_samples_per_node = n_samples / self.num_actors
        index_range_list = [
            (int(i * num_samples_per_node), int((i + 1) * num_samples_per_node))
            for i in range(self.num_actors)
        ]
        index_range_list[-1] = (index_range_list[-1][0], n_samples)

        futures = []
        for i, (start, end) in enumerate(index_range_list):
            chunk_refs = {k: v[start:end] for k, v in condition_refs.items()} if condition_refs else None
            method = getattr(self.wrapped_node_list[i], method_name)
            futures.append(method.remote(end - start, condition_refs=chunk_refs))
        sample_lists = ray.get(futures)
        result = []
        for sample_list in sample_lists:
            if sample_list:
                result.extend(sample_list)
        return result

    def sample(self, n_samples, condition_refs=None):
        return self._multiplexed_call('sample', n_samples, condition_refs)

    def sample_posterior(self, n_samples, condition_refs=None):
        return self._multiplexed_call('sample_posterior', n_samples, condition_refs)

    def sample_proposal(self, n_samples, condition_refs=None):
        return self._multiplexed_call('sample_proposal', n_samples, condition_refs)

    def shutdown(self):
        for node in self.wrapped_node_list:
            node.shutdown.remote()

    def save(self, node_dir):
        pass  # Silently ignore, multiplexed nodes are never saved

    def load(self, node_dir):
        pass  # Silently ignore, multiplexed nodes are never saved

    def wait_ready(self):
        """Wait for all child actors to initialize."""
        ray.get([actor.__ray_ready__.remote() for actor in self.wrapped_node_list])

    def get_status(self) -> dict:
        """Return aggregated status from all child actors."""
        # Get status from first child (they should all be similar)
        if self.wrapped_node_list:
            try:
                return ray.get(self.wrapped_node_list[0].get_status.remote(), timeout=2.0)
            except Exception:
                return {"status": "error", "error": "timeout"}
        return {"status": "unknown"}

    def get_output_log_tail(self, num_lines: int = 50) -> list:
        """Return log tail from first child actor."""
        if self.wrapped_node_list:
            try:
                return ray.get(
                    self.wrapped_node_list[0].get_output_log_tail.remote(num_lines),
                    timeout=2.0
                )
            except Exception:
                return ["[Error fetching logs]"]
        return []


# TODO: NodeWrapper is async solely because train() uses asyncio.sleep(0) to yield.
# This makes every ray.get inside the actor (e.g. CachedDataLoader) block the event
# loop and trigger warnings. Consider splitting into separate training and sampling
# actors — sampling reads best_model which is independent of training state.
@ray.remote
class NodeWrapper:
    def __init__(self, node, graph, model_path=None, log_config=None):
        # Suppress Ray warning about blocking ray.get in async actor.
        # Ray emits this once per actor via a global flag. We set the flag
        # to True before any ray.get calls to prevent the warning.
        # NodeWrapper is async (for train's pause/resume), but sampling methods
        # are synchronous and need blocking ray.get. This is unavoidable without
        # splitting into separate training/sampling actors.
        # TODO: Consider actor split to fully separate async training from sync sampling.
        try:
            import ray._private.worker as _ray_worker
            _ray_worker.blocking_get_inside_async_warned = True
        except (ImportError, AttributeError):
            pass  # Ray internals changed, warning will appear

        # Add model_path to sys.path if provided
        if model_path:
            model_path = Path(model_path).resolve()
            if str(model_path) not in sys.path:
                sys.path.insert(0, str(model_path))

        self.node = node
        self.name = node.name

        # Create logger and set as module-level logger
        # This enables falcon.log(), falcon.info() etc. for simulators and estimators
        if log_config:
            self._logger = Logger(self.name, log_config, capture_exceptions=True)
        else:
            # Fallback: create a minimal logger config
            self._logger = Logger(self.name, {"local": {"enabled": True, "dir": "."}}, capture_exceptions=True)
        set_logger(self._logger)

        # Status tracking for monitoring
        self._status = "initializing"

        simulator_cls = LazyLoader(node.simulator_cls)
        self.simulator_instance = simulator_cls(**node.simulator_config)

        # Condition keys for embedding (evidence + scaffolds)
        self.condition_keys = self.node.evidence + self.node.scaffolds
        debug(f"Condition keys: {self.condition_keys}")

        if node.estimator_cls is not None:
            estimator_cls = LazyLoader(node.estimator_cls)
            self.estimator_instance = estimator_cls(
                self.simulator_instance,
                theta_key=node.name,
                condition_keys=self.condition_keys,
                config=node.estimator_config,
            )
        else:
            self.estimator_instance = None

        self.parents = node.parents
        self.evidence = node.evidence
        self.scaffolds = node.scaffolds
        self.graph = graph

        # Mark initialization complete
        self._status = "idle"
        self._stop_requested = False

    def request_stop(self):
        """Request graceful stop after current epoch."""
        self._stop_requested = True
        # Signal to estimator to terminate after current epoch
        if self.estimator_instance is not None and hasattr(self.estimator_instance, 'interrupt'):
            self.estimator_instance.interrupt()

    async def train(self, dataset_manager, observations={}, num_trailing_samples=None):
        self._status = "training"
        info(f"[{self.name}] Training started")
        debug(f"Condition keys: {self.evidence + self.scaffolds}")

        # Create BufferView - estimator controls what keys it needs
        # Use estimator's device for cache if cache_on_device is enabled
        cache_device = None
        if hasattr(self.estimator_instance, 'cache_on_device') and self.estimator_instance.cache_on_device:
            cache_device = str(getattr(self.estimator_instance, 'device', 'cpu'))
        buffer = BufferView(dataset_manager, cache_device=cache_device)

        await self.estimator_instance.train(buffer)
        self._status = "done"

        # Get final loss for completion message
        final_loss = None
        if hasattr(self.estimator_instance, 'best_conditional_flow_val_loss'):
            final_loss = self.estimator_instance.best_conditional_flow_val_loss
        elif hasattr(self.estimator_instance, 'history'):
            losses = self.estimator_instance.history.get('val_loss', [])
            if losses:
                final_loss = min(losses)

        if final_loss is not None:
            info(f"[{self.name}] Training completed (loss: {final_loss:.4f})")
        else:
            info(f"[{self.name}] Training completed")

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
                result[name] = val[np.newaxis]  # (1, ...) — compact, consumer expands
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

    def _chunked_sample(self, n_samples, condition_refs, method):
        """Resolve refs, chunk, call method, return refs.

        Broadcast conditions (shape[0]==1) pass through without slicing,
        letting consumers expand efficiently (GPU expand or np.broadcast_to).

        Args:
            n_samples: Number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] or None
            method: Internal method (_simulate, _sample_posterior, _sample_proposal)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample
        """
        conditions = self._resolve_refs(condition_refs)
        chunk_size = getattr(self.node, 'sample_chunk_size', 0) or n_samples
        result = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = {
                k: v if v.shape[0] == 1 else v[start:end]
                for k, v in conditions.items()
            } if conditions else None
            output = method(end - start, chunk)
            result.extend(self._batch_to_refs(output))
        return result

    # ==================== Public Sampling Methods ====================

    def sample(self, n_samples, condition_refs=None):
        """Sample and return ObjectRefs. Handles chunking internally.

        Args:
            n_samples: Number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] from parent nodes

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample
        """
        return self._chunked_sample(n_samples, condition_refs, self._simulate)

    def sample_posterior(self, n_samples, condition_refs=None):
        """Sample from posterior and return ObjectRefs.

        Args:
            n_samples: Number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] from condition nodes

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample
        """
        return self._chunked_sample(n_samples, condition_refs, self._sample_posterior)

    def sample_proposal(self, n_samples, condition_refs=None):
        """Sample from proposal and return ObjectRefs.

        Args:
            n_samples: Number of samples to generate
            condition_refs: Dict[str, List[ObjectRef]] from condition nodes

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample
        """
        return self._chunked_sample(n_samples, condition_refs, self._sample_proposal)

    # ==================== Internal Sampling Methods ====================

    def _simulate(self, n_samples, conditions=None):
        """Call simulator/estimator with resolved arrays.

        Expands broadcast conditions (shape[0]==1) via np.broadcast_to
        so simulators see per-sample arrays.

        Returns:
            dict: {'value': ndarray} or {'value': ndarray, 'log_prob': ndarray}
        """
        if self.estimator_instance is not None:
            return self.estimator_instance.sample_prior(n_samples, conditions=conditions or None)
        # Expand broadcast conditions for simulators
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

    def _sample_posterior(self, n_samples, conditions=None):
        """Call estimator posterior sampling with resolved arrays.

        Returns:
            dict: {'value': ndarray, 'log_prob': ndarray}
        """
        if self.estimator_instance is None:
            return self._simulate(n_samples, conditions)
        return self.estimator_instance.sample_posterior(n_samples, conditions=conditions)

    def _sample_proposal(self, n_samples, conditions=None):
        """Call estimator proposal sampling with resolved arrays.

        Returns:
            dict: {'value': ndarray, 'log_prob': ndarray}
        """
        if self.estimator_instance is None:
            return self._simulate(n_samples, conditions)
        return self.estimator_instance.sample_proposal(n_samples, conditions=conditions)

    def save(self, node_dir):
        if self.estimator_instance is not None:
            node_dir.mkdir(parents=True, exist_ok=True)
            return self.estimator_instance.save(node_dir)

    def load(self, node_dir):
        if self.estimator_instance is not None:
            node_dir.mkdir(parents=True, exist_ok=True)
            return self.estimator_instance.load(node_dir)

    def get_status(self) -> dict:
        """Return current status for monitoring."""
        status = {
            "name": self.name,
            "status": self._status,
            "samples": 0,
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "loss_history": [],
        }

        # Get estimator state if available
        if self.estimator_instance is not None:
            est = self.estimator_instance
            if hasattr(est, "history"):
                status["loss_history"] = est.history.get("val_loss", [])[-20:]
                if status["loss_history"]:
                    status["loss"] = status["loss_history"][-1]
                status["current_epoch"] = len(est.history.get("epochs", []))
            if hasattr(est, "loop_config"):
                status["total_epochs"] = est.loop_config.max_epochs
            if hasattr(est, "history") and est.history.get("n_samples"):
                status["samples"] = est.history["n_samples"][-1]

        return status

    def get_output_log_tail(self, num_lines: int = 50) -> list:
        """Return recent log lines from output.log."""
        return self._logger.get_output_log_tail(num_lines)

    def shutdown(self):
        """Shutdown the node and its logger."""
        if hasattr(self, '_logger'):
            self._logger.shutdown()


class DeployedGraph:
    def __init__(self, graph, model_path=None, log_config=None):
        """Initialize a DeployedGraph with the given conceptual graph of nodes.

        Note: This class uses falcon.info(), falcon.warning() etc. for logging.
        These functions use the module-level logger set by cli.py via set_logger().
        """
        self.graph = graph
        self.model_path = model_path
        self.log_config = log_config or {}
        self.wrapped_nodes_dict = {}

        self.deploy_nodes()

    def _check_resource_budget(self):
        """Raise if node GPU/CPU requests exceed cluster capacity."""
        cluster = ray.cluster_resources()
        available_gpus = cluster.get("GPU", 0)
        available_cpus = cluster.get("CPU", 0)

        total_gpus = sum(
            node.actor_config.get("num_gpus", 0) * node.num_actors
            for node in self.graph.node_list
        )
        total_cpus = sum(
            node.actor_config.get("num_cpus", 1) * node.num_actors
            for node in self.graph.node_list
        )

        if total_gpus > available_gpus:
            node_summary = ", ".join(
                f"{n.name}: {n.actor_config.get('num_gpus', 0)} GPU"
                for n in self.graph.node_list
                if n.actor_config.get("num_gpus", 0) > 0
            )
            warning(
                f"GPU over-subscription: nodes request {total_gpus:.1f} GPUs "
                f"but only {available_gpus:.1f} available. "
                f"Actors may hang — reduce ray.num_gpus in your config or increase "
                f"available GPUs. ({node_summary})"
            )
        if total_cpus > available_cpus:
            warning(
                f"CPU over-subscription: nodes request {total_cpus} CPUs "
                f"but only {available_cpus:.0f} available — actors may queue."
            )

    def deploy_nodes(self):
        """Deploy all nodes in the graph as Ray actors."""
        info("Spinning up graph...")
        self._check_resource_budget()

        # Create all actors (non-blocking)
        for node in self.graph.node_list:
            if node.num_actors > 1:
                self.wrapped_nodes_dict[node.name] = MultiplexNodeWrapper.remote(
                    node.actor_config,
                    node,
                    self.graph,
                    node.num_actors,
                    self.model_path,
                    self.log_config,
                )
            else:
                self.wrapped_nodes_dict[node.name] = NodeWrapper.options(
                    **_ray_options(node.actor_config)
                ).remote(node, self.graph, self.model_path, self.log_config)

        # Wait for all actors to initialize
        for name, actor in self.wrapped_nodes_dict.items():
            try:
                # MultiplexNodeWrapper has wait_ready(), NodeWrapper uses __ray_ready__
                if hasattr(actor, 'wait_ready'):
                    ready_ref = actor.wait_ready.remote()
                else:
                    ready_ref = actor.__ray_ready__.remote()
                done, _ = ray.wait([ready_ref], timeout=60.0)
                if not done:
                    raise RuntimeError(
                        f"Node '{name}' did not initialize within 60 s. "
                        "This usually means Ray cannot schedule the actor — "
                        "check that ray.num_gpus/num_cpus in your config do not "
                        "exceed available cluster resources."
                    )
                ray.get(done[0])  # re-raise any actor-side exception
                info(f"  ✓ {name}")

            except ray.exceptions.RayActorError as e:
                raise RuntimeError(f"Failed to initialize node '{name}': {e}") from e

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

            remote_method = getattr(self.wrapped_nodes_dict[name], sample_method)
            node_refs = ray.get(
                remote_method.remote(num_samples, condition_refs=node_condition_refs)
            )

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

    def shutdown(self):
        """Shut down the deployed graph and release resources."""
        ray.get([node.shutdown.remote() for node in self.wrapped_nodes_dict.values()])

    def launch(self, dataset_manager, observations, graph_path=None, stop_check=None):
        """Launch training.

        Args:
            dataset_manager: Dataset manager for samples
            observations: Observation data
            graph_path: Path to save/load graph
            stop_check: Optional callable that returns True when graceful stop is requested
        """
        asyncio.run(self._launch(dataset_manager, observations, graph_path=graph_path, stop_check=stop_check))

    async def _launch(self, dataset_manager, observations, graph_path=None, stop_check=None):
        # Load graph if saved model files exist (not just logging directories)
        if graph_path is not None and any(graph_path.glob("*/*.pth")):
            self.load(graph_path)

        # TODO: Make distrinction clearer between dataset_manager and dataset_manager_actor
        dataset_manager = dataset_manager.dataset_manager_actor

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

        # Training - start all training nodes
        train_futures = {}  # Map future -> node_name for completion tracking
        for name, node in self.graph.node_dict.items():
            if node.train:
                wrapped_node = self.wrapped_nodes_dict[name]
                train_future = wrapped_node.train.remote(
                    dataset_manager, observations=observations
                )
                train_futures[train_future] = name
                info(f"[{name}] Training started")
                time.sleep(1)

        simulate_interval = ray.get(dataset_manager.get_simulate_interval.remote())

        # Track last status log time for periodic updates
        last_status_log = time.time()
        STATUS_LOG_INTERVAL = 60  # seconds

        train_future_list = list(train_futures.keys())
        stop_requested = False
        while train_future_list:
            # Check for graceful stop request
            if stop_check is not None and stop_check():
                info("Graceful stop requested, finishing current epoch...")
                stop_requested = True
                # Signal all training nodes to stop after current epoch
                for name, node in self.wrapped_nodes_dict.items():
                    try:
                        ray.get(node.request_stop.remote(), timeout=1)
                    except Exception:
                        pass  # Node may not support request_stop

            ready, train_future_list = ray.wait(
                train_future_list, num_returns=len(train_future_list), timeout=1
            )

            # Skip simulation if stopping
            if not stop_requested:
                time.sleep(simulate_interval)
                num_new_samples = ray.get(dataset_manager.num_resims.remote())
                if num_new_samples > 0:
                    # sample_proposal interleaves with training via async yield points —
                    # no pause needed. Forward simulation runs on separate actors concurrently.
                    proposal_refs = self.sample_proposal(num_new_samples, observations)
                    condition_refs = self._extract_value_refs(proposal_refs)
                    # Only keep latent nodes (with estimators) from proposal.
                    # Deterministic intermediates and observed nodes must be
                    # re-simulated to maintain data consistency.
                    latent_nodes = {n.name for n in self.graph.node_list
                                    if n.estimator_cls is not None}
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
                    ray.get(dataset_manager.append_refs.remote(sample_refs))

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
                    # Get final loss from node status
                    status = ray.get(self.wrapped_nodes_dict[node_name].get_status.remote())
                    loss = status.get("loss")
                    if loss is not None:
                        info(f"[{node_name}] Training completed (loss: {loss:.4f})")
                    else:
                        info(f"[{node_name}] Training completed")

        # Save graph if path is provided
        if graph_path is not None:
            self.save(graph_path)

        info("")
        info("Analysis completed.")

    def _log_status(self, dataset_manager):
        """Log periodic status of active training nodes and buffer (separate lines)."""
        # Log progress for each active training node
        for name, node in self.wrapped_nodes_dict.items():
            status = ray.get(node.get_status.remote())
            if status["status"] == "training":
                epoch = status.get("current_epoch", 0)
                total = status.get("total_epochs", 0)
                loss = status.get("loss")
                loss_str = f"{loss:.2f}" if loss is not None else "?"
                info(f"[{name}] epoch {epoch}/{total}, loss {loss_str}")

        # Log buffer stats (including total ever simulated)
        stats = ray.get(dataset_manager.get_store_stats.remote())
        info(f"Buffer: {stats['training']} train, {stats['validation']} val ({stats['total_length']} total)")

    def save(self, graph_dir):
        """Save the deployed graph node status."""
        graph_dir = graph_dir.expanduser().resolve()
        graph_dir.mkdir(parents=True, exist_ok=True)
        save_futures = []
        for name, node in self.wrapped_nodes_dict.items():
            node_dir = graph_dir / name
            save_future = node.save.remote(node_dir)
            save_futures.append(save_future)
        ray.get(save_futures)

    def load(self, graph_dir):
        """Load the deployed graph nodes status."""
        info(f"Loading deployed graph from: {graph_dir}")
        load_futures = []
        for name, node in self.wrapped_nodes_dict.items():
            node_dir = Path(graph_dir) / name
            load_future = node.load.remote(node_dir)
            load_futures.append(load_future)
        ray.get(load_futures)

