"""Tests for gate module — CapabilityGate."""

import pytest

from tibet_context.gate import CapabilityGate
from tibet_context.container import ContextContainer
from tibet_context.layers import Layer, CapabilityProfile


def _make_container():
    layers = {
        0: Layer(level=0, content="summary", token_count=2, min_capability=3),
        1: Layer(level=1, content="conversation detail", token_count=5, min_capability=14),
        2: Layer(level=2, content="deep context with full analysis", token_count=8, min_capability=32),
    }
    return ContextContainer(id="gate-test", layers=layers, tibet_chain_id="chain-gate")


class TestCapabilityGate:
    def test_check_capability_known(self):
        gate = CapabilityGate()
        assert gate.check_capability("qwen2.5:3b") == 3
        assert gate.check_capability("qwen2.5:32b") == 32

    def test_check_capability_unknown(self):
        gate = CapabilityGate()
        with pytest.raises(ValueError, match="Unknown model"):
            gate.check_capability("unknown-model")

    def test_register_model(self):
        gate = CapabilityGate()
        gate.register_model("my-custom:4b", 4)
        assert gate.check_capability("my-custom:4b") == 4

    def test_unlock_layers_3b(self):
        gate = CapabilityGate()
        container = _make_container()
        layers = gate.unlock_layers(container, "qwen2.5:3b")
        assert len(layers) == 1
        assert layers[0].level == 0

    def test_unlock_layers_14b(self):
        gate = CapabilityGate()
        container = _make_container()
        layers = gate.unlock_layers(container, "qwen2.5:14b")
        assert len(layers) == 2
        levels = [l.level for l in layers]
        assert 0 in levels and 1 in levels

    def test_unlock_layers_32b(self):
        gate = CapabilityGate()
        container = _make_container()
        layers = gate.unlock_layers(container, "qwen2.5:32b")
        assert len(layers) == 3

    def test_max_accessible_level(self):
        gate = CapabilityGate()
        assert gate.max_accessible_level("qwen2.5:3b") == 0
        assert gate.max_accessible_level("qwen2.5:14b") == 1
        assert gate.max_accessible_level("qwen2.5:32b") == 2

    def test_carbonara_test_fail(self):
        gate = CapabilityGate()
        assert gate.carbonara_test("qwen2.5:3b") is False
        assert gate.carbonara_test("qwen2.5:7b") is False
        assert gate.carbonara_test("qwen2.5:14b") is False

    def test_carbonara_test_pass(self):
        gate = CapabilityGate()
        assert gate.carbonara_test("qwen2.5:32b") is True
        assert gate.carbonara_test("qwen2.5:72b") is True

    def test_carbonara_test_custom_level(self):
        gate = CapabilityGate()
        assert gate.carbonara_test("qwen2.5:14b", required_level=1) is True
        assert gate.carbonara_test("qwen2.5:3b", required_level=0) is True

    def test_gate_report(self):
        gate = CapabilityGate()
        container = _make_container()
        report = gate.gate_report(container, "qwen2.5:7b")
        assert report["model_id"] == "qwen2.5:7b"
        assert report["capability"] == 7
        assert 0 in report["accessible_levels"]
        assert 2 in report["blocked_levels"]
        assert report["carbonara_pass"] is False

    def test_custom_profile(self):
        profile = CapabilityProfile.from_dict({
            "name": "llama",
            "layers": {
                "0": {"min_capability": 1, "max_tokens": 512},
                "1": {"min_capability": 8, "max_tokens": 4096},
                "2": {"min_capability": 70, "max_tokens": 16384},
            },
        })
        gate = CapabilityGate(profile=profile)
        gate.register_model("llama3.2:1b", 1)
        assert gate.carbonara_test("llama3.2:1b") is False
        assert gate.carbonara_test("llama3.1:8b") is False
        assert gate.carbonara_test("llama3.1:70b") is True
