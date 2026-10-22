"""Ray options of the sample and train actors, and node names (no Ray cluster)."""

import pytest

from falcon.core.deployed_graph import resolve_actor_options
from falcon.core.graph import Graph, create_graph_from_config


def test_split_gpu_keys_and_shared_options():
    sample, train, notes = resolve_actor_options(
        {"num_train_gpus": 1, "num_sample_gpus": 0.25, "num_cpus": 2, "runtime_env": {"env_vars": {"A": "1"}}},
        has_estimator=True,
    )
    assert sample == {"num_gpus": 0.25, "num_cpus": 2, "runtime_env": {"env_vars": {"A": "1"}}}
    assert train == {"num_gpus": 1, "num_cpus": 2, "runtime_env": {"env_vars": {"A": "1"}}}
    assert notes == []


def test_deprecated_num_gpus_is_split_evenly():
    sample, train, notes = resolve_actor_options({"num_gpus": 1}, has_estimator=True)
    assert sample["num_gpus"] == train["num_gpus"] == 0.5
    assert "deprecated" in notes[0]


def test_deprecated_num_gpus_on_simulator_nodes():
    sample, train, notes = resolve_actor_options({"num_gpus": 1}, has_estimator=False)
    assert sample == {"num_gpus": 1}
    assert train is None
    assert "num_sample_gpus=1" in notes[0]


def test_mixing_old_and_new_gpu_keys_is_an_error():
    with pytest.raises(ValueError, match="num_gpus"):
        resolve_actor_options({"num_gpus": 1, "num_train_gpus": 1}, has_estimator=True)


def test_train_gpus_without_estimator_are_ignored():
    sample, train, notes = resolve_actor_options({"num_train_gpus": 1}, has_estimator=False)
    assert sample == {} and train is None
    assert "ignored" in notes[0]


def test_names_and_unsupported_options():
    sample, train, notes = resolve_actor_options({"name": "z", "max_concurrency": 4}, has_estimator=True)
    assert sample == {"name": "z"}
    assert train == {"name": "z/train"}
    assert "max_concurrency" in notes[0]


def test_split_cpu_keys_override_num_cpus():
    sample, train, notes = resolve_actor_options(
        {"num_cpus": 2, "num_train_cpus": 6}, has_estimator=True,
    )
    assert sample == {"num_cpus": 2}
    assert train == {"num_cpus": 6}
    assert notes == []

    sample, train, notes = resolve_actor_options({"num_train_cpus": 6}, has_estimator=False)
    assert sample == {} and train is None
    assert "num_train_cpus is ignored" in notes[0]


def test_thread_counts_share_the_cores_between_estimator_actors(monkeypatch):
    from falcon.core import deployed_graph
    from falcon.core.deployed_graph import DeployedGraph

    monkeypatch.setattr(deployed_graph, "_local_cpu_count", lambda: 18)
    graph = (
        Graph()
        .add_node("z", simulator="p.Prior", estimator="e.Est", evidence=["x"], num_actors=2)
        .add_node("x", simulator="m.Sim", parents=["z"], observed=True)
        .add_node("w", simulator="p.Prior", estimator="e.Est", evidence=["x"])
    )
    dg = DeployedGraph.__new__(DeployedGraph)
    dg.graph, dg.train = graph, True
    options = {
        "z": ({}, {}),
        "x": ({}, None),
        "w": ({}, {"num_cpus": 4}),
    }
    counts = dg._thread_counts(options)

    # Five estimator actors (z: 2 sample + 1 train, w: 1 sample + 1 train) share 18 cores
    assert counts == {"z": (3, 3), "x": (None, None), "w": (3, 4)}


def test_node_names_must_not_contain_a_slash():
    with pytest.raises(ValueError, match="z/train"):
        Graph().add_node("z/train", simulator="model.Sim")
    with pytest.raises(ValueError):
        create_graph_from_config({"a/b": {"simulator": "model.Sim"}})
