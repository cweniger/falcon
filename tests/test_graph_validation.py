"""Tests for evidence/embedding _input_ mismatch validation in create_graph_from_config."""

import pytest
import warnings

from omegaconf import OmegaConf

from falcon.core.graph import (
    _collect_input_keys,
    _validate_evidence_vs_embedding,
    create_graph_from_config,
)


class TestCollectInputKeys:
    def test_string(self):
        assert _collect_input_keys("x") == ["x"]

    def test_list(self):
        assert sorted(_collect_input_keys(["x", "y"])) == ["x", "y"]

    def test_nested_dict(self):
        config = {"_target_": "model.E", "_input_": ["x", "y"]}
        assert sorted(_collect_input_keys(config)) == ["x", "y"]

    def test_deeply_nested(self):
        config = {
            "_target_": "model.Outer",
            "_input_": [
                {"_target_": "model.A", "_input_": "x"},
                {"_target_": "model.B", "_input_": "y"},
            ],
        }
        assert sorted(_collect_input_keys(config)) == ["x", "y"]

    def test_deduplicates(self):
        config = {"_target_": "model.E", "_input_": ["x", "x"]}
        assert _collect_input_keys(config) == ["x"]

    def test_no_input_key(self):
        assert _collect_input_keys({"_target_": "model.E"}) == []


class TestValidateEvidenceVsEmbedding:
    def _embedding(self, inputs):
        return {"_target_": "model.E", "_input_": inputs}

    def test_match_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_evidence_vs_embedding(
                "theta",
                evidence=["x"],
                estimator_config={"embedding": self._embedding("x")},
                scaffolds=[],
            )

    def test_mismatch_missing_emits_warning(self):
        with pytest.warns(UserWarning, match="missing from 'evidence'"):
            _validate_evidence_vs_embedding(
                "theta",
                evidence=[],
                estimator_config={"embedding": self._embedding("x")},
                scaffolds=[],
            )

    def test_mismatch_extra_emits_warning(self):
        with pytest.warns(UserWarning, match="in 'evidence' but not in embedding"):
            _validate_evidence_vs_embedding(
                "theta",
                evidence=["x", "z"],
                estimator_config={"embedding": self._embedding("x")},
                scaffolds=[],
            )

    def test_scaffold_excluded_from_evidence(self):
        # x is in _input_ but also a scaffold → should not appear in expected evidence
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_evidence_vs_embedding(
                "theta",
                evidence=[],
                estimator_config={"embedding": self._embedding(["x", "y"])},
                scaffolds=["x", "y"],
            )

    def test_no_embedding_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_evidence_vs_embedding(
                "theta",
                evidence=["x"],
                estimator_config={"network": {}},
                scaffolds=[],
            )

    def test_empty_estimator_config_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _validate_evidence_vs_embedding("theta", evidence=[], estimator_config={}, scaffolds=[])


class TestCreateGraphFromConfigValidation:
    """Integration: validate that create_graph_from_config raises the warning end-to-end."""

    def _minimal_config(self, evidence, embedding_inputs):
        # x has no parents here to avoid a backward-graph cycle in the topology sort
        return {
            "theta": {
                "simulator": {"_target_": "falcon.priors.Hypercube", "priors": [["uniform", -1.0, 1.0]]},
                "estimator": {
                    "_target_": "falcon.estimators.Flow",
                    "embedding": {"_target_": "model.E", "_input_": embedding_inputs},
                },
                "evidence": evidence,
            },
            "x": {
                "simulator": {"_target_": "model.Simulator"},
            },
        }

    def test_matching_evidence_no_warning(self):
        config = OmegaConf.create(self._minimal_config(evidence=["x"], embedding_inputs="x"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            create_graph_from_config(config)

    def test_mismatched_evidence_warns(self):
        config = OmegaConf.create(self._minimal_config(evidence=[], embedding_inputs="x"))
        with pytest.warns(UserWarning, match="evidence/embedding mismatch"):
            create_graph_from_config(config)
