import time
import ray
import asyncio
import torch
import os
import sys
from pathlib import Path
import numpy as np
from omegaconf import ListConfig

from falcon.core.logger import Logger, set_logger, debug, info, warning, error, log
from falcon.core.raystore import BufferView
from .utils import LazyLoader, as_rvbatch


@ray.remote
class MultiplexNodeWrapper:
    def __init__(self, actor_config, node, graph, num_actors, model_path=None, log_config=None):
        self.num_actors = num_actors
        self.wrapped_node_list = [
            NodeWrapper.options(**actor_config).remote(node, graph, model_path, log_config)
            for _ in range(self.num_actors)
        ]

    def sample(self, n_samples, incoming=None):
        """Multiplexed sampling returning refs.

        Distributes work across actors and gathers refs.

        Args:
            n_samples: Number of samples to generate
            incoming: List of arrays from parent nodes

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to values
        """
        num_samples_per_node = n_samples / self.num_actors
        index_range_list = [
            (int(i * num_samples_per_node), int((i + 1) * num_samples_per_node))
            for i in range(self.num_actors)
        ]
        index_range_list[-1] = (index_range_list[-1][0], n_samples)

        futures = []
        for i, (start, end) in enumerate(index_range_list):
            my_incoming = [v[start:end] for v in incoming] if incoming else None
            futures.append(
                self.wrapped_node_list[i].sample.remote(
                    end - start, incoming=my_incoming
                )
            )
        # Each actor returns List[Dict[str, ObjectRef]]
        sample_lists = ray.get(futures)
        # Concatenate lists
        result = []
        for sample_list in sample_lists:
            if sample_list:
                result.extend(sample_list)
        return result

    def sample_posterior(self, n_samples, conditions=None):
        """Multiplexed posterior sampling returning refs."""
        num_samples_per_node = n_samples / self.num_actors
        index_range_list = [
            (int(i * num_samples_per_node), int((i + 1) * num_samples_per_node))
            for i in range(self.num_actors)
        ]
        index_range_list[-1] = (index_range_list[-1][0], n_samples)

        futures = []
        for i, (start, end) in enumerate(index_range_list):
            chunk_conditions = None
            if conditions:
                chunk_conditions = {k: v[start:end] for k, v in conditions.items()}
            futures.append(
                self.wrapped_node_list[i].sample_posterior.remote(
                    end - start, conditions=chunk_conditions
                )
            )
        sample_lists = ray.get(futures)
        result = []
        for sample_list in sample_lists:
            if sample_list:
                result.extend(sample_list)
        return result

    def sample_proposal(self, n_samples, conditions=None):
        """Multiplexed proposal sampling returning refs."""
        num_samples_per_node = n_samples / self.num_actors
        index_range_list = [
            (int(i * num_samples_per_node), int((i + 1) * num_samples_per_node))
            for i in range(self.num_actors)
        ]
        index_range_list[-1] = (index_range_list[-1][0], n_samples)

        futures = []
        for i, (start, end) in enumerate(index_range_list):
            chunk_conditions = None
            if conditions:
                chunk_conditions = {k: v[start:end] for k, v in conditions.items()}
            futures.append(
                self.wrapped_node_list[i].sample_proposal.remote(
                    end - start, conditions=chunk_conditions
                )
            )
        sample_lists = ray.get(futures)
        result = []
        for sample_list in sample_lists:
            if sample_list:
                result.extend(sample_list)
        return result

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


# TODO: NodeWrapper is async solely because train() needs to yield for pause/resume.
# This makes every ray.get inside the actor (e.g. CachedDataLoader) block the event
# loop and trigger warnings. Consider splitting into separate training and sampling
# actors — sampling reads best_model which is independent of training state, so it
# doesn't actually need to pause training. Weight sync on validation improvement only.
@ray.remote
class NodeWrapper:
    def __init__(self, node, graph, model_path=None, log_config=None):
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
                final_loss = losses[-1]

        if final_loss is not None:
            info(f"[{self.name}] Training completed (loss: {final_loss:.4f})")
        else:
            info(f"[{self.name}] Training completed")

    def _simulate(self, n_samples, incoming=None):
        """Internal method that calls the actual simulator/estimator.

        Returns batched samples (RVBatch or ndarray).
        """
        # Default to empty list for simulator paths
        if incoming is None:
            incoming = []

        if self.estimator_instance is not None:
            # Convert list to dict for new interface (prior typically has no conditions)
            conditions = None
            if incoming:
                conditions = {k: v for k, v in zip(self.parents, incoming)}
            samples = self.estimator_instance.sample_prior(n_samples, conditions=conditions)
            samples = as_rvbatch(samples)
            return samples
        if hasattr(self.simulator_instance, "simulate_batch"):
            return self.simulator_instance.simulate_batch(n_samples, *incoming)
        else:
            samples = []
            for i in range(n_samples):
                params = [v[i] for v in incoming]
                samples.append(self.simulator_instance.simulate(*params))
            return np.stack(samples)

    def sample(self, n_samples, incoming=None):
        """Sample and return ObjectRefs. Handles chunking internally.

        Args:
            n_samples: Number of samples to generate
            incoming: List of arrays from parent nodes (one per parent)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to values
        """
        chunk_size = self.node.actor_config.get('chunk_size', 0)
        if chunk_size <= 0:
            chunk_size = n_samples  # No chunking

        result = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk_incoming = [v[start:end] for v in incoming] if incoming is not None else None

            # Call simulator (uses simulate_batch if available)
            samples_batched = self._simulate(end - start, chunk_incoming)

            # Handle RVBatch wrapper
            if hasattr(samples_batched, 'value'):
                value, logprob = samples_batched.value, samples_batched.logprob
            else:
                value, logprob = samples_batched, None

            # Convert chunk to refs
            n = value.shape[0] if isinstance(value, np.ndarray) else len(next(iter(value.values())))
            for i in range(n):
                sample_dict = {}
                if isinstance(value, np.ndarray):
                    sample_dict[self.name] = ray.put(value[i])
                else:
                    # value is a dict, store the i-th slice as a dict under self.name
                    sample_i = {k: v[i] for k, v in value.items()}
                    sample_dict[self.name] = ray.put(sample_i)
                if logprob is not None:
                    sample_dict[f"{self.name}.logprob"] = ray.put(logprob[i])
                result.append(sample_dict)

        return result

    def _sample_posterior(self, n_samples, conditions=None):
        """Internal method for posterior sampling. Returns RVBatch."""
        samples = self.estimator_instance.sample_posterior(n_samples, conditions=conditions)
        samples = as_rvbatch(samples)
        return samples

    def _sample_proposal(self, n_samples, conditions=None):
        """Internal method for proposal sampling. Returns RVBatch."""
        samples = self.estimator_instance.sample_proposal(n_samples, conditions=conditions)
        samples = as_rvbatch(samples)
        return samples

    def _convert_rvbatch_to_refs(self, rvbatch, chunk_start, chunk_end):
        """Convert RVBatch to list of dicts with ObjectRefs."""
        result = []
        value, logprob = rvbatch.value, rvbatch.logprob

        n = value.shape[0] if isinstance(value, np.ndarray) else len(next(iter(value.values())))
        for i in range(n):
            sample_dict = {}
            if isinstance(value, np.ndarray):
                sample_dict[self.name] = ray.put(value[i])
            else:
                # value is a dict, store the i-th slice as a dict under self.name
                sample_i = {k: v[i] for k, v in value.items()}
                sample_dict[self.name] = ray.put(sample_i)
            if logprob is not None:
                sample_dict[f"{self.name}.logprob"] = ray.put(logprob[i])
            result.append(sample_dict)
        return result

    def sample_posterior(self, n_samples, conditions=None):
        """Sample from posterior and return ObjectRefs. Handles chunking internally.

        Args:
            n_samples: Number of samples to generate
            conditions: Dict mapping condition keys to arrays

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to values
        """
        chunk_size = self.node.actor_config.get('chunk_size', 0)
        if chunk_size <= 0:
            chunk_size = n_samples

        result = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk_conditions = None
            if conditions:
                chunk_conditions = {k: v[start:end] for k, v in conditions.items()}

            rvbatch = self._sample_posterior(end - start, conditions=chunk_conditions)
            result.extend(self._convert_rvbatch_to_refs(rvbatch, start, end))

        return result

    def sample_proposal(self, n_samples, conditions=None):
        """Sample from proposal and return ObjectRefs. Handles chunking internally.

        Args:
            n_samples: Number of samples to generate
            conditions: Dict mapping condition keys to arrays

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to values
        """
        chunk_size = self.node.actor_config.get('chunk_size', 0)
        if chunk_size <= 0:
            chunk_size = n_samples

        result = []
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk_conditions = None
            if conditions:
                chunk_conditions = {k: v[start:end] for k, v in conditions.items()}

            rvbatch = self._sample_proposal(end - start, conditions=chunk_conditions)
            result.extend(self._convert_rvbatch_to_refs(rvbatch, start, end))

        return result

    # TODO: Currently not used anywhere, add tests?
    def call_simulator_method(self, method_name, *args, **kwargs):
        method = getattr(self.simulator_instance, method_name)
        return method(*args, **kwargs)

    # TODO: Currently not used anywhere, add tests?
    def call_estimator_method(self, method_name, *args, **kwargs):
        method = getattr(self.estimator_instance, method_name)
        return method(*args, **kwargs)

    # TODO: Currently not used anywhere, add tests?
    def shutdown(self):
        pass

    def save(self, node_dir):
        if self.estimator_instance is not None:
            node_dir.mkdir(parents=True, exist_ok=True)
            return self.estimator_instance.save(node_dir)

    def load(self, node_dir):
        if self.estimator_instance is not None:
            node_dir.mkdir(parents=True, exist_ok=True)
            return self.estimator_instance.load(node_dir)

    def pause(self):
        if self.estimator_instance is not None:
            return self.estimator_instance.pause()

    def resume(self):
        if self.estimator_instance is not None:
            return self.estimator_instance.resume()

    def interrupt(self):
        if self.estimator_instance is not None:
            return self.estimator_instance.interrupt()

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
                status["total_epochs"] = est.loop_config.num_epochs
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
        self.monitor_bridge = None

        self._create_monitor_bridge()
        self.deploy_nodes()

    def _create_monitor_bridge(self):
        """Create the MonitorBridge actor for falcon monitor TUI."""
        from falcon.core.monitor_bridge import MonitorBridge
        run_dir = str(self.model_path) if self.model_path else "unknown"
        try:
            # Name the actor so falcon monitor can discover it
            self.monitor_bridge = MonitorBridge.options(
                name="falcon:monitor_bridge",
                lifetime="detached",  # Keep alive even if creator dies
            ).remote(run_dir=run_dir)
        except ValueError as e:
            # Actor with this name might already exist from a previous run
            if "already exists" in str(e):
                try:
                    self.monitor_bridge = ray.get_actor("falcon:monitor_bridge")
                except Exception:
                    warning("Could not connect to existing monitor bridge")
                    self.monitor_bridge = None
            else:
                warning(f"Could not create monitor bridge: {e}")
                self.monitor_bridge = None
        except Exception as e:
            warning(f"Could not create monitor bridge: {e}")
            self.monitor_bridge = None

    def deploy_nodes(self):
        """Deploy all nodes in the graph as Ray actors."""
        info("Spinning up graph...")

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
                    **node.actor_config
                ).remote(node, self.graph, self.model_path, self.log_config)

        # Wait for all actors to initialize and register with monitor bridge
        for name, actor in self.wrapped_nodes_dict.items():
            try:
                # MultiplexNodeWrapper has wait_ready(), NodeWrapper uses __ray_ready__
                if hasattr(actor, 'wait_ready'):
                    ray.get(actor.wait_ready.remote())
                else:
                    ray.get(actor.__ray_ready__.remote())
                info(f"  ✓ {name}")

                # Register node with monitor bridge
                if self.monitor_bridge:
                    ray.get(self.monitor_bridge.register_node.remote(name, actor))
            except ray.exceptions.RayActorError as e:
                raise RuntimeError(f"Failed to initialize node '{name}': {e}") from e

    def _resolve_refs_to_trace(self, sample_refs, trace):
        """Resolve ObjectRefs and update trace with stacked arrays.

        Args:
            sample_refs: List[Dict[str, ObjectRef]] - one dict per sample
            trace: Dict to update with stacked arrays
        """
        if not sample_refs:
            return

        # Collect all keys from the refs
        keys = set()
        for ref_dict in sample_refs:
            keys.update(ref_dict.keys())

        # Resolve and stack for each key
        for key in keys:
            if key not in trace:
                values = [ray.get(ref_dict[key]) for ref_dict in sample_refs if key in ref_dict]
                if values:
                    trace[key] = np.stack(values)

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
        # Debug: print keys in first sample
        if sample_refs:
            debug(f"_merge_refs: sample_refs[0] keys = {list(sample_refs[0].keys())}")

    def _execute_graph(self, num_samples, sorted_node_names, conditions, sample_method):
        """Execute graph traversal with specified sampling method.

        Args:
            num_samples: Number of samples to generate
            sorted_node_names: Node names in execution order
            conditions: Initial trace conditions (dict of arrays for observed nodes)
            sample_method: One of "sample", "sample_posterior", "sample_proposal"

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        # trace holds stacked arrays for passing between nodes during graph traversal
        trace = conditions.copy()

        # sample_refs accumulates ObjectRefs from all nodes
        sample_refs = []

        for name in sorted_node_names:
            if name in trace:
                continue

            # Get conditions based on sampling method
            if sample_method == "sample":
                incoming = [trace[parent] for parent in self.graph.get_parents(name)]
                node_refs = ray.get(
                    self.wrapped_nodes_dict[name].sample.remote(num_samples, incoming=incoming)
                )
            else:
                # Build conditions dict from parents and evidence
                node_conditions = {}
                for parent in self.graph.get_parents(name):
                    node_conditions[parent] = trace[parent]
                for evidence in self.graph.get_evidence(name):
                    node_conditions[evidence] = trace[evidence]

                remote_method = getattr(self.wrapped_nodes_dict[name], sample_method)
                node_refs = ray.get(
                    remote_method.remote(num_samples, conditions=node_conditions)
                )

            # Resolve refs to arrays for the trace (needed for child nodes)
            self._resolve_refs_to_trace(node_refs, trace)

            # Merge refs into accumulator
            self._merge_refs(sample_refs, node_refs)

        return sample_refs

    def sample(self, num_samples, conditions=None):
        """Run forward sampling through the graph.

        Args:
            num_samples: Number of samples to generate
            conditions: Optional dict of pre-set conditions (arrays)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        return self._execute_graph(
            num_samples,
            self.graph.sorted_node_names,
            conditions or {},
            "sample",
        )

    def sample_posterior(self, num_samples, conditions=None):
        """Run posterior sampling through the inference graph.

        Args:
            num_samples: Number of samples to generate
            conditions: Optional dict of pre-set conditions (arrays)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        return self._execute_graph(
            num_samples,
            self.graph.sorted_inference_node_names,
            conditions or {},
            "sample_posterior",
        )

    def sample_proposal(self, num_samples, conditions=None):
        """Run proposal sampling through the inference graph.

        Args:
            num_samples: Number of samples to generate
            conditions: Optional dict of pre-set conditions (arrays)

        Returns:
            List[Dict[str, ObjectRef]]: One dict per sample with refs to all node values
        """
        return self._execute_graph(
            num_samples,
            self.graph.sorted_inference_node_names,
            conditions or {},
            "sample_proposal",
        )

    def _refs_to_arrays(self, sample_refs):
        """Convert List[Dict[str, ObjectRef]] to Dict[str, ndarray].

        Resolves refs and stacks into batched arrays.
        """
        if not sample_refs:
            return {}

        # Collect all keys
        keys = set()
        for ref_dict in sample_refs:
            keys.update(ref_dict.keys())

        # Resolve and stack
        result = {}
        for key in keys:
            values = [ray.get(ref_dict[key]) for ref_dict in sample_refs if key in ref_dict]
            if values:
                result[key] = np.stack(values)
        return result

    def shutdown(self):
        """Shut down the deployed graph and release resources."""
        ray.get([node.shutdown.remote() for node in self.wrapped_nodes_dict.values()])

    def launch(self, dataset_manager, observations, graph_path=None):
        asyncio.run(self._launch(dataset_manager, observations, graph_path=graph_path))

    async def _launch(self, dataset_manager, observations, graph_path=None):
        # Load graph if saved model files exist (not just logging directories)
        if graph_path is not None and any(graph_path.glob("*/*.pth")):
            self.load(graph_path)

        # TODO: Make distrinction clearer between dataset_manager and dataset_manager_actor
        dataset_manager = dataset_manager.dataset_manager_actor

        # Register dataset manager with monitor bridge for monitoring
        if self.monitor_bridge:
            ray.get(self.monitor_bridge.register_dataset_manager.remote(dataset_manager))
            # Set log directory so monitor bridge can read log files
            if graph_path is not None:
                ray.get(self.monitor_bridge.set_log_dir.remote(str(graph_path)))

        # Initial data generation: load from disk first, then generate remaining
        # Nodes handle chunking internally based on their ray.chunk_size config
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
        info("Starting analysis. Monitor with: falcon monitor")

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

        resample_interval = ray.get(dataset_manager.get_resample_interval.remote())
        # time.sleep(60) # Wait sixty seconds before starting resampling

        # Track last status log time for periodic updates
        last_status_log = time.time()
        STATUS_LOG_INTERVAL = 60  # seconds

        train_future_list = list(train_futures.keys())
        while train_future_list:
            ready, train_future_list = ray.wait(
                train_future_list, num_returns=len(train_future_list), timeout=1
            )
            time.sleep(resample_interval)
            num_new_samples = ray.get(dataset_manager.num_resims.remote())
            self.pause()
            if num_new_samples > 0:
                # Sample from proposal (returns refs)
                proposal_refs = self.sample_proposal(num_new_samples, observations)
                # Convert refs to arrays for use as conditions
                proposal_arrays = self._refs_to_arrays(proposal_refs)
                # Remove observations from conditions
                for key in observations.keys():
                    proposal_arrays.pop(key, None)
                # Forward sample with proposal as conditions (returns refs)
                sample_refs = self.sample(num_new_samples, conditions=proposal_arrays)
                # Merge proposal refs (z1, z2, logprobs) into sample refs (img1, img2, x)
                # so the complete sample is stored in the buffer
                for i, prop_ref in enumerate(proposal_refs):
                    sample_refs[i].update(prop_ref)
                # Append refs directly to buffer
                ray.get(dataset_manager.append_refs.remote(sample_refs))
            self.resume()

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

    def pause(self):
        """Pause all nodes in the deployed graph."""
        pause_futures = []
        for _, node in self.wrapped_nodes_dict.items():
            pause_future = node.pause.remote()
            pause_futures.append(pause_future)
        ray.get(pause_futures)

    def resume(self):
        """Resume all nodes in the deployed graph."""
        resume_futures = []
        for _, node in self.wrapped_nodes_dict.items():
            resume_future = node.resume.remote()
            resume_futures.append(resume_future)
        ray.get(resume_futures)

    def interrupt(self):
        """Interrupt all nodes in the deployed graph."""
        interrupt_futures = []
        for _, node in self.wrapped_nodes_dict.items():
            interrupt_future = node.interrupt.remote()
            interrupt_futures.append(interrupt_future)
        ray.get(interrupt_futures)
