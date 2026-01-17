import time
import ray
import asyncio
import torch
import os
import sys
from pathlib import Path
import numpy as np
from omegaconf import ListConfig

from falcon.core.logger import Logger
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
        num_samples_per_node = n_samples / self.num_actors
        index_range_list = [
            (int(i * num_samples_per_node), int((i + 1) * num_samples_per_node))
            for i in range(self.num_actors)
        ]
        index_range_list[-1] = (index_range_list[-1][0], n_samples)

        futures = []
        for i, (start, end) in enumerate(index_range_list):
            my_incoming = [v[start:end] for v in incoming]
            futures.append(
                self.wrapped_node_list[i].sample.remote(
                    end - start, incoming=my_incoming
                )
            )
        samples = ray.get(futures)
        samples = [s for s in samples if len(s) > 0]  # Only include non-empty samples
        samples = np.concatenate(samples, axis=0)
        return samples

    def sample_posterior(self, *args, **kwargs):
        raise NotImplementedError

    def sample_proposal(self, *args, **kwargs):
        raise NotImplementedError

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

    def get_log_tail(self, num_lines: int = 50) -> list:
        """Return log tail from first child actor."""
        if self.wrapped_node_list:
            try:
                return ray.get(
                    self.wrapped_node_list[0].get_log_tail.remote(num_lines),
                    timeout=2.0
                )
            except Exception:
                return ["[Error fetching logs]"]
        return []


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

        # Create logger directly (no Ray actor)
        if log_config:
            self.logger = Logger(self.name, log_config, capture_exceptions=True)
        else:
            # Fallback: create a minimal logger config
            self.logger = Logger(self.name, {"local": {"enabled": True, "dir": "."}}, capture_exceptions=True)

        # Set module-level logger so falcon.log() works in simulators
        from falcon.core.logger import set_logger
        set_logger(self.logger)

        # Status tracking for monitoring
        self._status = "initializing"

        simulator_cls = LazyLoader(node.simulator_cls)
        self.simulator_instance = simulator_cls(**node.simulator_config)

        # Condition keys for embedding (evidence + scaffolds)
        self.condition_keys = self.node.evidence + self.node.scaffolds
        self.logger.debug(f"Condition keys: {self.condition_keys}")

        if node.estimator_cls is not None:
            estimator_cls = LazyLoader(node.estimator_cls)
            self.estimator_instance = estimator_cls(
                self.simulator_instance,
                theta_key=node.name,
                condition_keys=self.condition_keys,
                config=node.estimator_config,
                logger=self.logger,
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
        self.logger.info(f"[{self.name}] Training started")
        self.logger.debug(f"Condition keys: {self.evidence + self.scaffolds}")

        # Create BufferView - estimator controls what keys it needs
        buffer = BufferView(dataset_manager)

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
            self.logger.info(f"[{self.name}] Training completed (loss: {final_loss:.4f})")
        else:
            self.logger.info(f"[{self.name}] Training completed")

    def sample(self, n_samples, incoming=None):
        if self.estimator_instance is not None:
            samples = self.estimator_instance.sample_prior(
                n_samples, parent_conditions=incoming
            )
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

    def sample_posterior(
        self, n_samples, parent_conditions=[], evidence_conditions=[]
    ):
        samples = self.estimator_instance.sample_posterior(
            n_samples,
            parent_conditions=parent_conditions,
            evidence_conditions=evidence_conditions,
        )
        samples = as_rvbatch(samples)
        return samples

    def sample_proposal(self, n_samples, parent_conditions=[], evidence_conditions=[]):
        samples = self.estimator_instance.sample_proposal(
            n_samples,
            parent_conditions=parent_conditions,
            evidence_conditions=evidence_conditions,
        )
        samples = as_rvbatch(samples)
        return samples

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

    def get_log_tail(self, num_lines: int = 50) -> list:
        """Return recent log lines from output.log."""
        return self.logger.get_log_tail(num_lines)

    def shutdown(self):
        """Shutdown the node and its logger."""
        if hasattr(self, 'logger'):
            self.logger.shutdown()


class DeployedGraph:
    def __init__(self, graph, model_path=None, log_config=None):
        """Initialize a DeployedGraph with the given conceptual graph of nodes."""
        self.graph = graph
        self.model_path = model_path
        self.log_config = log_config or {}
        self.wrapped_nodes_dict = {}
        self.coordinator = None

        # Create a driver-side logger for DeployedGraph (but don't capture exceptions,
        # as the main driver logger already does that)
        self._logger = None
        self._init_logger()

        self._create_coordinator()
        self.deploy_nodes()

    def _init_logger(self):
        """Initialize or reinitialize the logger."""
        if self._logger is None:
            self._logger = Logger("deployed_graph", self.log_config, capture_exceptions=False)

    @property
    def logger(self):
        """Get the logger, initializing if needed (e.g., after deserialization)."""
        if self._logger is None:
            self._init_logger()
        return self._logger

    def __getstate__(self):
        """Exclude logger from pickling (for Ray serialization)."""
        state = self.__dict__.copy()
        state['_logger'] = None  # Logger contains locks that can't be pickled
        return state

    def __setstate__(self, state):
        """Restore state and reinitialize logger."""
        self.__dict__.update(state)
        # Logger will be reinitialized on first access via the property

    def _create_coordinator(self):
        """Create the FalconCoordinator actor for monitoring."""
        from falcon.core.coordinator import FalconCoordinator
        run_dir = str(self.model_path) if self.model_path else "unknown"
        try:
            # Name the actor so falcon monitor can discover it
            self.coordinator = FalconCoordinator.options(
                name="falcon:coordinator",
                lifetime="detached",  # Keep alive even if creator dies
            ).remote(run_dir=run_dir)
        except ValueError as e:
            # Actor with this name might already exist from a previous run
            if "already exists" in str(e):
                try:
                    self.coordinator = ray.get_actor("falcon:coordinator")
                except Exception:
                    self.logger.warning("Could not connect to existing coordinator")
                    self.coordinator = None
            else:
                self.logger.warning(f"Could not create coordinator: {e}")
                self.coordinator = None
        except Exception as e:
            self.logger.warning(f"Could not create coordinator: {e}")
            self.coordinator = None

    def deploy_nodes(self):
        """Deploy all nodes in the graph as Ray actors."""
        self.logger.info("Spinning up graph...")

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

        # Wait for all actors to initialize and register with coordinator
        for name, actor in self.wrapped_nodes_dict.items():
            try:
                # MultiplexNodeWrapper has wait_ready(), NodeWrapper uses __ray_ready__
                if hasattr(actor, 'wait_ready'):
                    ray.get(actor.wait_ready.remote())
                else:
                    ray.get(actor.__ray_ready__.remote())
                self.logger.info(f"  ✓ {name}")

                # Register node with coordinator
                if self.coordinator:
                    ray.get(self.coordinator.register_node.remote(name, actor))
            except ray.exceptions.RayActorError as e:
                raise RuntimeError(f"Failed to initialize node '{name}': {e}") from e

    def _execute_graph(self, num_samples, sorted_node_names, conditions, sample_method):
        """Execute graph traversal with specified sampling method.

        Args:
            num_samples: Number of samples to generate
            sorted_node_names: Node names in execution order
            conditions: Initial trace conditions
            sample_method: One of "sample", "sample_posterior", "sample_proposal"

        Returns:
            Trace dictionary with sampled values and logprobs
        """
        trace = conditions.copy()

        for name in sorted_node_names:
            if name in trace:
                continue

            # Get conditions based on sampling method
            if sample_method == "sample":
                incoming = [trace[parent] for parent in self.graph.get_parents(name)]
                rvbatch = ray.get(
                    self.wrapped_nodes_dict[name].sample.remote(num_samples, incoming=incoming)
                )
            else:
                parent_conditions = [trace[parent] for parent in self.graph.get_parents(name)]
                evidence_conditions = [trace[parent] for parent in self.graph.get_evidence(name)]
                remote_method = getattr(self.wrapped_nodes_dict[name], sample_method)
                rvbatch = ray.get(
                    remote_method.remote(
                        num_samples,
                        parent_conditions=parent_conditions,
                        evidence_conditions=evidence_conditions,
                    )
                )

            rvbatch = as_rvbatch(rvbatch)
            trace[name] = rvbatch.value
            if rvbatch.logprob is not None:
                trace[f"{name}.logprob"] = rvbatch.logprob

        return trace

    def sample(self, num_samples, conditions=None):
        """Run forward sampling through the graph."""
        return self._execute_graph(
            num_samples,
            self.graph.sorted_node_names,
            conditions or {},
            "sample",
        )

    def sample_posterior(self, num_samples, conditions=None):
        """Run posterior sampling through the inference graph."""
        return self._execute_graph(
            num_samples,
            self.graph.sorted_inference_node_names,
            conditions or {},
            "sample_posterior",
        )

    def sample_proposal(self, num_samples, conditions=None):
        """Run proposal sampling through the inference graph."""
        return self._execute_graph(
            num_samples,
            self.graph.sorted_inference_node_names,
            conditions or {},
            "sample_proposal",
        )

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

        # Register dataset manager with coordinator for monitoring
        if self.coordinator:
            ray.get(self.coordinator.register_dataset_manager.remote(dataset_manager))
            # Set log directory so coordinator can read log files
            if graph_path is not None:
                ray.get(self.coordinator.set_log_dir.remote(str(graph_path)))

        # Initial data generation
        ray.get(dataset_manager.initialize_samples.remote(self))

        self.logger.info("")
        self.logger.info("Starting analysis. Monitor with: falcon monitor")

        # Training - start all training nodes
        train_futures = {}  # Map future -> node_name for completion tracking
        for name, node in self.graph.node_dict.items():
            if node.train:
                wrapped_node = self.wrapped_nodes_dict[name]
                train_future = wrapped_node.train.remote(
                    dataset_manager, observations=observations
                )
                train_futures[train_future] = name
                self.logger.info(f"[{name}] Training started")
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
            while num_new_samples > 0:
                this_n = min(num_new_samples, 512)
                new_samples = self.sample_proposal(this_n, observations)
                for key in observations.keys():  # Remove observations from new samples
                    del new_samples[key]
                new_samples_batched = self.sample(this_n, conditions=new_samples)
                # Convert dict-of-arrays to list-of-dicts for append
                new_samples = [
                    {k: v[i] for k, v in new_samples_batched.items()} for i in range(this_n)
                ]
                ray.get(dataset_manager.append.remote(new_samples))
                num_new_samples -= this_n
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
                        self.logger.info(f"[{node_name}] Training completed (loss: {loss:.4f})")
                    else:
                        self.logger.info(f"[{node_name}] Training completed")

        # Save graph if path is provided
        if graph_path is not None:
            self.save(graph_path)

        self.logger.info("")
        self.logger.info("Analysis completed.")

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
                self.logger.info(f"[{name}] epoch {epoch}/{total}, loss {loss_str}")

        # Log buffer stats (including total ever simulated)
        stats = ray.get(dataset_manager.get_store_stats.remote())
        self.logger.info(f"Buffer: {stats['training']} train, {stats['validation']} val ({stats['total_length']} total)")

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
        self.logger.info(f"Loading deployed graph from: {graph_dir}")
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
