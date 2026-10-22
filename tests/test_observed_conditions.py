"""Which nodes must be simulated to condition an estimator node on the observation."""

from falcon.core.deployed_graph import observation_nodes
from falcon.core.graph import Graph


def _graph():
    graph = Graph()
    graph.add_node("z", "model.Prior", "falcon.estimators.FlowMatching", evidence=["tokens"])
    graph.add_node("x", "model.Signal", parents=["z"], observed=True)
    graph.add_node("clean", "model.Clean", parents=["x"])
    graph.add_node("tokens", "model.Tokenize", parents=["clean"])
    graph.add_node("w", "model.Prior", "falcon.estimators.FlowMatching", evidence=["x"])
    graph.add_node("v", "model.Prior", "falcon.estimators.FlowMatching", evidence=["x", "w"])
    return graph


def test_observed_evidence_needs_no_simulation():
    assert observation_nodes(_graph(), "w", {"x"}) == []


def test_derived_evidence_is_simulated_from_the_observation_in_order():
    assert observation_nodes(_graph(), "z", {"x"}) == ["clean", "tokens"]


def test_evidence_on_a_latent_node_is_not_fixed_by_the_observation():
    assert observation_nodes(_graph(), "v", {"x"}) is None


def test_missing_observation_fixes_nothing():
    assert observation_nodes(_graph(), "z", set()) is None
