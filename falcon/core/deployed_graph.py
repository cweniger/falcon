import time
import ray
from torch.utils.data import DataLoader
import asyncio
import torch
import os
import sys
from pathlib import Path
import numpy as np
from omegaconf import ListConfig

from falcon.core.logging import initialize_logging_for
from .utils import LazyLoader, as_rvbatch


@ray.remote
class MultiplexNodeWrapper:
    def __init__(self, actor_config, node, graph, num_actors, model_path=None):
        self.num_actors = num_actors
        self.wrapped_node_list = [
            NodeWrapper.options(**actor_config).remote(node, graph, model_path)
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

    def conditioned_sample(self, *args, **kwargs):
        raise NotImplementedError

    def proposal_sample(self, *args, **kwargs):
        raise NotImplementedError

    def shutdown(self):
        for node in self.wrapped_node_list:
            node.shutdown.remote()

    def save(self, node_dir):
        pass  # Silently ignore, multiplexed nodes are never saved

    def load(self, node_dir):
        pass  # Silently ignore, multiplexed nodes are never saved


@ray.remote
class NodeWrapper:
    def __init__(self, node, graph, model_path=None):
        # Add model_path to sys.path if provided
        if model_path:
            model_path = Path(model_path).resolve()
            if str(model_path) not in sys.path:
                sys.path.insert(0, str(model_path))

        self.node = node

        simulator_cls = LazyLoader(node.simulator_cls)
        self.simulator_instance = simulator_cls(**node.simulator_config)

        _embedding_keywords = self.node.evidence + self.node.scaffolds
        print("Embedding keywords:", _embedding_keywords)

        if node.estimator_cls is not None:
            estimator_cls = LazyLoader(node.estimator_cls)
            self.estimator_instance = estimator_cls(
                self.simulator_instance,
                _embedding_keywords=_embedding_keywords,
                **node.estimator_config,
            )
        else:
            self.estimator_instance = None

        self.parents = node.parents
        self.evidence = node.evidence
        self.scaffolds = node.scaffolds
        self.name = node.name
        self.graph = graph

        # Initialize and set global logger
        initialize_logging_for(self.name)  # Set logger for global scope of this actor

    async def train(self, dataset_manager, observations={}, num_trailing_samples=None):
        print("Training started for:", self.name)
        keys_train = (
            [self.name, self.name + ".logprob"] + self.evidence + self.scaffolds
        )
        keys_val = [self.name, self.name + ".logprob"] + self.evidence + self.scaffolds

        batch_size = self.node.estimator_config.get("batch_size", 128)

        dataset_train = ray.get(
            dataset_manager.get_train_dataset_view.remote(keys_train, filter=None)
        )
        dataset_val = ray.get(
            dataset_manager.get_val_dataset_view.remote(keys_val, filter=None)
        )
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size)

        def hook_fn(module, batch):
            ids, theta, theta_logprob, conditions = (
                batch[0],
                batch[1],
                batch[2],
                batch[3:],
            )
            for i, k in enumerate(self.evidence):
                if k in observations.keys():
                    conditions[i] = observations[k]
            # Corresponding id
            mask = module.discardable(theta, theta_logprob, conditions)
            ids = ids[mask]
            ids = list(ids.numpy())

            # Deactivate samples
            dataset_manager.deactivate.remote(ids)

        #        hook_fn = None

        await self.estimator_instance.train(
            dataloader_train, dataloader_val, hook_fn=hook_fn, dataset_manager=dataset_manager
        )
        print("...training complete for:", self.name)

    def sample(self, n_samples, incoming=None):
        if self.estimator_instance is not None:
            samples = self.estimator_instance.prior_sample(
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

    def conditioned_sample(
        self, n_samples, parent_conditions=[], evidence_conditions=[]
    ):
        samples = self.estimator_instance.conditioned_sample(
            n_samples,
            parent_conditions=parent_conditions,
            evidence_conditions=evidence_conditions,
        )
        samples = as_rvbatch(samples)
        return samples

    def proposal_sample(self, n_samples, parent_conditions=[], evidence_conditions=[]):
        samples = self.estimator_instance.proposal_sample(
            n_samples,
            parent_conditions=parent_conditions,
            evidence_conditions=evidence_conditions,
        )
        samples = as_rvbatch(samples)
        return samples

    def call_simulator_method(self, method_name, *args, **kwargs):
        method = getattr(self.simulator_instance, method_name)
        return method(*args, **kwargs)

    def call_estimator_method(self, method_name, *args, **kwargs):
        method = getattr(self.estimator_instance, method_name)
        return method(*args, **kwargs)

    def shutdown(self):
        pass

    def save(self, node_dir):
        # Silently ignore if the module does not have a save method
        if hasattr(self.estimator_instance, "save"):
            node_dir.mkdir(parents=True, exist_ok=True)
            return self.estimator_instance.save(node_dir)

    def load(self, node_dir):
        # Silently ignore if the module does not have a load method
        if hasattr(self.estimator_instance, "load"):
            node_dir.mkdir(parents=True, exist_ok=True)
            return self.estimator_instance.load(node_dir)

    def pause(self):
        if hasattr(self.estimator_instance, "pause"):
            return self.estimator_instance.pause()

    def resume(self):
        if hasattr(self.estimator_instance, "resume"):
            return self.estimator_instance.resume()

    def interrupt(self):
        if hasattr(self.estimator_instance, "interrupt"):
            return self.estimator_instance.interrupt()


class DeployedGraph:
    def __init__(self, graph, model_path=None):
        """Initialize a DeployedGraph with the given conceptual graph of nodes."""
        self.graph = graph
        self.model_path = model_path
        self.wrapped_nodes_dict = {}
        self.deploy_nodes()

    def deploy_nodes(self):
        """Deploy all nodes in the graph as Ray actors."""
        ray.init(ignore_reinit_error=True)  # Initialize Ray if not already done
        for node in self.graph.node_list:
            if node.num_actors > 1:
                self.wrapped_nodes_dict[node.name] = MultiplexNodeWrapper.remote(
                    node.actor_config,
                    node,
                    self.graph,
                    node.num_actors,
                    self.model_path,
                )
            else:
                self.wrapped_nodes_dict[node.name] = NodeWrapper.options(
                    **node.actor_config
                ).remote(node, self.graph, self.model_path)

    def sample(self, num_samples, conditions={}):
        """Run the graph using deployed nodes and return results."""
        sorted_node_names = self.graph.sorted_node_names
        trace = conditions.copy()

        # Process nodes in topological order
        for name in sorted_node_names:
            if name in trace.keys():
                continue
            incoming = [trace[parent] for parent in self.graph.get_parents(name)]
            rvbatch = ray.get(
                self.wrapped_nodes_dict[name].sample.remote(
                    num_samples, incoming=incoming
                )
            )
            rvbatch = as_rvbatch(rvbatch)
            trace[name] = rvbatch.value
            if rvbatch.logprob is not None:
                trace[f"{name}.logprob"] = rvbatch.logprob
        return trace

    def conditioned_sample(self, num_samples, conditions={}):
        """Run the graph using deployed nodes and return results."""
        sorted_node_names = self.graph.sorted_inference_node_names
        trace = conditions.copy()

        # Process nodes in topological order
        for name in sorted_node_names:
            if name in trace.keys():
                continue
            evidence_conditions = [
                trace[parent] for parent in self.graph.get_evidence(name)
            ]
            parent_conditions = [
                trace[parent] for parent in self.graph.get_parents(name)
            ]
            rvbatch = ray.get(
                self.wrapped_nodes_dict[name].conditioned_sample.remote(
                    num_samples,
                    parent_conditions=parent_conditions,
                    evidence_conditions=evidence_conditions,
                )
            )
            trace[name] = rvbatch.value
            if rvbatch.logprob is not None:
                trace[f"{name}.logprob"] = rvbatch.logprob

        return trace

    def proposal_sample(self, num_samples, conditions={}):
        """Run the graph using deployed nodes and return results."""
        sorted_node_names = self.graph.sorted_inference_node_names
        trace = conditions.copy()

        # Process nodes in topological order
        for name in sorted_node_names:
            if name in trace.keys():
                continue
            parent_conditions = [
                trace[parent] for parent in self.graph.get_parents(name)
            ]
            evidence_conditions = [
                trace[parent] for parent in self.graph.get_evidence(name)
            ]
            rvbatch = ray.get(
                self.wrapped_nodes_dict[name].proposal_sample.remote(
                    num_samples,
                    parent_conditions=parent_conditions,
                    evidence_conditions=evidence_conditions,
                )
            )
            trace[name] = rvbatch.value
            if rvbatch.logprob is not None:
                trace[f"{name}.logprob"] = rvbatch.logprob

        return trace

    def shutdown(self):
        """Shut down the deployed graph and release resources."""
        ray.get([node.shutdown.remote() for node in self.wrapped_nodes_dict.values()])

    def launch(self, dataset_manager, observations, graph_path=None):
        asyncio.run(self._launch(dataset_manager, observations, graph_path=graph_path))

    async def _launch(self, dataset_manager, observations, graph_path=None):
        # Load graph if path is provided
        if graph_path is not None and graph_path.exists():
            self.load(graph_path)

        # TODO: Make distrinction clearer between dataset_manager and dataset_manager_actor
        dataset_manager = dataset_manager.dataset_manager_actor

        # Initial data generation
        ray.get(dataset_manager.initialize_samples.remote(self))

        # Training
        train_future_list = []
        for name, node in self.graph.node_dict.items():
            if node.train:
                wrapped_node = self.wrapped_nodes_dict[name]
                train_future = wrapped_node.train.remote(
                    dataset_manager, observations=observations
                )
                train_future_list.append(train_future)
                time.sleep(1)

        resample_interval = ray.get(dataset_manager.get_resample_interval.remote())
        # time.sleep(60) # Wait sixty seconds before starting resampling

        while train_future_list:
            ready, train_future_list = ray.wait(
                train_future_list, num_returns=len(train_future_list), timeout=1
            )
            time.sleep(resample_interval)
            num_new_samples = ray.get(dataset_manager.num_resims.remote())
            self.pause()
            while num_new_samples > 0:
                this_n = min(num_new_samples, 512)
                new_samples = self.proposal_sample(this_n, observations)
                for key in observations.keys():  # Remove observations from new samples
                    del new_samples[key]
                new_samples = self.sample(this_n, conditions=new_samples)
                ray.get(dataset_manager.append.remote(new_samples))
                num_new_samples -= this_n
            self.resume()

            for completed_task in ready:
                result = ray.get(
                    completed_task
                )  # Retrieve the result or raise an exception

            # FIXME: Not optimal, should happen after specific time steps
            if graph_path is not None:
                self.save(graph_path)

        # Save graph if path is provided
        if graph_path is not None:
            self.save(graph_path)

    def save(self, graph_dir):
        """Save the deployed graph node status."""
        print("💾 Saving deployed graph to:", str(graph_dir))
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
        print("💾 Loading deployed graph from:", str(graph_dir))
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
