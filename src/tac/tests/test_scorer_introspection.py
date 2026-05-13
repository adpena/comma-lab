"""Tests for tac.diagnostics.scorer_introspection and cuda_cpu_drift.

Per CLAUDE.md "MPS auth eval is NOISE" and "Submission auth eval — BOTH CPU
AND CUDA": these tests run CPU-only and never make a score claim. Outputs are
tagged ``[diagnostic-not-score]`` in the records under test.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tac.diagnostics import (
    DriftMetrics,
    IntrospectionRecord,
    LayerStats,
    ScorerIntrospector,
    compounding_factor,
    compute_layer_drift,
    fingerprint_tensor,
)
from tac.diagnostics.cuda_cpu_drift import (
    drift_to_dict,
    estimate_compounding_for_path,
)
from tac.diagnostics.scorer_introspection import (
    AttentionFingerprint,
    list_attention_like_layers,
    hash_record,
)


class TinyMLP(nn.Module):
    """Small CPU model for unit testing — deterministic, no randomness."""

    def __init__(self, in_dim: int = 4, hidden: int = 8, out_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _seeded_model(seed: int = 42) -> TinyMLP:
    torch.manual_seed(seed)
    m = TinyMLP().eval()
    return m


# ---------------------------------------------------------------------------
# LayerStats fingerprint tests
# ---------------------------------------------------------------------------


def test_layer_stats_basic_shape():
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stats = LayerStats.from_tensor(t)
    assert stats.shape == (2, 3)
    assert stats.numel == 6
    assert stats.dtype.endswith("float32")
    assert math.isclose(stats.mean, 3.5, rel_tol=1e-6)
    assert math.isclose(stats.min, 1.0, rel_tol=1e-6)
    assert math.isclose(stats.max, 6.0, rel_tol=1e-6)
    # L2 norm of [1..6] is sqrt(91)
    assert math.isclose(stats.l2_norm, math.sqrt(91.0), rel_tol=1e-5)


def test_layer_stats_empty_tensor():
    t = torch.empty(0)
    stats = LayerStats.from_tensor(t)
    assert stats.numel == 0
    assert stats.l2_norm == 0.0
    assert stats.sparsity_frac == 0.0


def test_fingerprint_tensor_alias_matches():
    t = torch.randn(7, 11)
    a = LayerStats.from_tensor(t)
    b = fingerprint_tensor(t)
    assert a.shape == b.shape and a.numel == b.numel
    assert math.isclose(a.l2_norm, b.l2_norm, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Attention-like layer enumeration
# ---------------------------------------------------------------------------


def test_list_attention_like_layers_handles_no_attention_models():
    m = _seeded_model()
    assert list_attention_like_layers(m) == []


def test_list_attention_like_layers_picks_up_multihead_attention():
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            y, _ = self.mha(x, x, x, need_weights=False)
            return self.lin(y)

    m = Wrap().eval()
    found = list_attention_like_layers(m)
    assert any(name == "mha" for name, _ in found)


# ---------------------------------------------------------------------------
# ScorerIntrospector smoke + determinism
# ---------------------------------------------------------------------------


def test_introspector_attaches_and_removes_hooks_via_session():
    m = _seeded_model()
    insp = ScorerIntrospector(m, capture_mode="fingerprint")
    assert insp._hook_attached is False
    with insp.session():
        assert insp._hook_attached is True
    assert insp._hook_attached is False
    # all hooks released:
    assert len(insp._hooks) == 0


def test_introspector_records_each_module_once():
    m = _seeded_model()
    x = torch.randn(2, 4)
    insp = ScorerIntrospector(m, capture_mode="full", full_threshold_elements=1024)
    with insp.session():
        record = insp.capture(x)
    # Expect fc1, act, fc2 in the record (skip Identity/Dropout/...)
    names = record.layer_names
    for expected in ("fc1", "act", "fc2"):
        assert expected in names, f"missing {expected!r} in {names}"
    # Every record entry should have at least one input + one output stats.
    for layer in record.layers:
        assert len(layer.input_stats) >= 1
        assert len(layer.output_stats) >= 1


def test_introspector_full_mode_keeps_tensors_within_threshold():
    m = _seeded_model()
    x = torch.randn(2, 4)
    insp = ScorerIntrospector(m, capture_mode="full", full_threshold_elements=1024)
    with insp.session():
        record = insp.capture(x)
    fc2 = record.get("fc2")
    assert fc2 is not None
    # fc2 output is (2, 3) -> 6 elements, well under threshold; should retain.
    assert fc2.full_output is not None
    assert fc2.full_output[0].shape == (2, 3)


def test_introspector_fingerprint_mode_drops_full_tensors():
    m = _seeded_model()
    x = torch.randn(2, 4)
    insp = ScorerIntrospector(m, capture_mode="fingerprint")
    with insp.session():
        record = insp.capture(x)
    for layer in record.layers:
        assert layer.full_input is None
        assert layer.full_output is None


def test_introspector_determinism_same_input_same_record():
    m1 = _seeded_model(42)
    m2 = _seeded_model(42)
    x = torch.randn(2, 4)
    insp1 = ScorerIntrospector(m1, capture_mode="full")
    insp2 = ScorerIntrospector(m2, capture_mode="full")
    with insp1.session():
        r1 = insp1.capture(x)
    with insp2.session():
        r2 = insp2.capture(x)
    assert r1.layer_names == r2.layer_names
    # Stats round to identical for deterministic tiny network.
    for a, b in zip(r1.layers, r2.layers):
        if a.full_output is None or b.full_output is None:
            continue
        for ta, tb in zip(a.full_output, b.full_output):
            assert torch.allclose(ta, tb)


def test_introspector_capture_requires_attached_hooks():
    m = _seeded_model()
    insp = ScorerIntrospector(m)
    with pytest.raises(RuntimeError):
        insp.capture(torch.randn(1, 4))


def test_introspector_invalid_capture_mode():
    m = _seeded_model()
    with pytest.raises(ValueError):
        ScorerIntrospector(m, capture_mode="bogus")


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


def test_round_trip_to_disk_preserves_record(tmp_path: Path):
    m = _seeded_model()
    x = torch.randn(2, 4)
    insp = ScorerIntrospector(m, capture_mode="full", full_threshold_elements=1024)
    with insp.session():
        record = insp.capture(x)
    out = record.to_disk(tmp_path / "rec.pt")
    assert out.exists()

    loaded = IntrospectionRecord.from_disk(out)
    assert loaded.model_kind == record.model_kind
    assert loaded.layer_names == record.layer_names
    # full tensors round-trip equal:
    for a, b in zip(record.layers, loaded.layers):
        if a.full_output is None:
            continue
        assert b.full_output is not None
        for ta, tb in zip(a.full_output, b.full_output):
            assert torch.allclose(ta, tb)


def test_to_json_emits_diagnostic_not_score_tag(tmp_path: Path):
    m = _seeded_model()
    insp = ScorerIntrospector(m, capture_mode="fingerprint")
    with insp.session():
        record = insp.capture(torch.randn(2, 4))
    out = record.to_json(tmp_path / "rec.json")
    text = out.read_text()
    assert "[diagnostic-not-score]" in text


# ---------------------------------------------------------------------------
# Drift comparison tests
# ---------------------------------------------------------------------------


def _capture_full(model: nn.Module, x: torch.Tensor) -> IntrospectionRecord:
    insp = ScorerIntrospector(model, capture_mode="full", full_threshold_elements=1 << 16)
    with insp.session():
        return insp.capture(x)


def test_drift_zero_when_models_identical():
    m1 = _seeded_model(42)
    m2 = _seeded_model(42)
    x = torch.randn(2, 4)
    r1 = _capture_full(m1, x)
    r2 = _capture_full(m2, x)
    drift = compute_layer_drift(r1, r2)
    for name, entries in drift.items():
        for entry in entries:
            if entry.has_full_tensors:
                assert entry.l2_relative_error == pytest.approx(0.0, abs=1e-6)
                assert entry.max_abs_error == pytest.approx(0.0, abs=1e-6)


def test_drift_nonzero_after_small_weight_perturbation():
    m1 = _seeded_model(42)
    m2 = _seeded_model(42)
    # Perturb m2's fc2 weight by 1%
    with torch.no_grad():
        m2.fc2.weight.add_(0.01 * torch.randn_like(m2.fc2.weight))
    x = torch.randn(2, 4)
    r1 = _capture_full(m1, x)
    r2 = _capture_full(m2, x)
    drift = compute_layer_drift(r1, r2)
    fc2_drift = drift["fc2"][0]
    assert fc2_drift.has_full_tensors
    # Drift should be measurable (>0) on fc2.
    assert fc2_drift.l2_relative_error > 1e-6
    # Upstream layers (fc1) untouched -> drift remains zero.
    fc1_drift = drift["fc1"][0]
    assert fc1_drift.l2_relative_error == pytest.approx(0.0, abs=1e-6)


def test_drift_kl_for_logit_like_layer_present():
    m1 = _seeded_model(42)
    m2 = _seeded_model(42)
    with torch.no_grad():
        m2.fc2.bias.add_(0.5)
    x = torch.randn(8, 4)
    r1 = _capture_full(m1, x)
    r2 = _capture_full(m2, x)
    drift = compute_layer_drift(r1, r2)
    # fc2 is a Linear -> module_type contains "Linear" -> KL & rank computed.
    fc2 = drift["fc2"][0]
    assert fc2.kl_divergence is not None
    assert fc2.rank_top1_disagreement is not None


def test_drift_handles_missing_layer_in_b():
    m1 = _seeded_model(42)
    x = torch.randn(2, 4)
    r1 = _capture_full(m1, x)
    # Surgically drop the last layer's record from r2:
    r2 = _capture_full(m1, x)
    r2.layers = [layer for layer in r2.layers if layer.name != "fc2"]
    drift = compute_layer_drift(r1, r2)
    fc2 = drift["fc2"][0]
    assert "missing" in fc2.note


def test_drift_to_dict_is_json_serializable():
    import json

    m = _seeded_model()
    x = torch.randn(2, 4)
    r1 = _capture_full(m, x)
    drift = compute_layer_drift(r1, r1)
    flat = drift_to_dict(drift)
    json.dumps(flat)  # no exception


def test_drift_fingerprint_only_proxies_when_full_absent():
    m = _seeded_model()
    x = torch.randn(2, 4)
    insp = ScorerIntrospector(m, capture_mode="fingerprint")
    with insp.session():
        r1 = insp.capture(x)
        r2 = insp.capture(x)
    drift = compute_layer_drift(r1, r2)
    fc2 = drift["fc2"][0]
    assert fc2.has_full_tensors is False
    assert fc2.fingerprint_only_l2_proxy is not None
    assert fc2.fingerprint_only_max_proxy is not None


# ---------------------------------------------------------------------------
# Compounding-factor closed-form tests
# ---------------------------------------------------------------------------


def test_compounding_factor_zero_eps_is_identity():
    assert compounding_factor([0.0] * 12) == pytest.approx(1.0, abs=1e-12)


def test_compounding_factor_classic_pr102_hypothesis():
    # 12 layers of 14% per-layer drift -> 1.14^12 ~ 4.82, near the empirical
    # PR102 CUDA-vs-CPU 5x pose multiplier.
    factor = compounding_factor([0.14] * 12)
    assert factor == pytest.approx(1.14**12, rel=1e-9)
    assert 4.5 < factor < 5.0


def test_compounding_factor_handles_empty_list():
    assert compounding_factor([]) == 1.0


def test_estimate_compounding_for_path_returns_expected_keys():
    # Build a synthetic drift dict with 3 RepMixerBlock-style entries.
    drift = {
        "vision.stages.0.blocks.0": [
            DriftMetrics(
                layer_name="vision.stages.0.blocks.0",
                module_type="RepMixerBlock",
                output_index=0,
                l2_relative_error=0.10,
                max_abs_error=0.0,
                mean_abs_error=0.0,
                kl_divergence=None,
                rank_top1_disagreement=None,
                has_full_tensors=True,
                fingerprint_only_l2_proxy=None,
                fingerprint_only_max_proxy=None,
            )
        ],
        "vision.stages.0.blocks.1": [
            DriftMetrics(
                layer_name="vision.stages.0.blocks.1",
                module_type="RepMixerBlock",
                output_index=0,
                l2_relative_error=0.20,
                max_abs_error=0.0,
                mean_abs_error=0.0,
                kl_divergence=None,
                rank_top1_disagreement=None,
                has_full_tensors=True,
                fingerprint_only_l2_proxy=None,
                fingerprint_only_max_proxy=None,
            )
        ],
        # non-block leaf — should be excluded.
        "vision.stages.0.blocks.0.token_mixer.norm": [
            DriftMetrics(
                layer_name="vision.stages.0.blocks.0.token_mixer.norm",
                module_type="MobileOneBlock",
                output_index=0,
                l2_relative_error=0.99,
                max_abs_error=0.0,
                mean_abs_error=0.0,
                kl_divergence=None,
                rank_top1_disagreement=None,
                has_full_tensors=True,
                fingerprint_only_l2_proxy=None,
                fingerprint_only_max_proxy=None,
            )
        ],
    }
    summary = estimate_compounding_for_path(drift)
    assert summary["num_layers_in_path"] == 2
    assert summary["compound_factor_l2_rel"] == pytest.approx(1.10 * 1.20, rel=1e-9)


def test_hash_record_changes_when_layers_change():
    m = _seeded_model()
    insp = ScorerIntrospector(m, capture_mode="fingerprint")
    with insp.session():
        record = insp.capture(torch.randn(2, 4))
    h0 = hash_record(record)
    record.layers = record.layers[:-1]
    h1 = hash_record(record)
    assert h0 != h1


def test_drift_records_carry_module_type():
    m = _seeded_model()
    x = torch.randn(2, 4)
    r = _capture_full(m, x)
    drift = compute_layer_drift(r, r)
    fc2 = drift["fc2"][0]
    assert "Linear" in fc2.module_type


def test_attention_fingerprint_dataclass_constructs_with_minimal_args():
    af = AttentionFingerprint(layer_name="x", layer_type="RepMixerBlock")
    assert af.num_heads is None
    assert af.softmax_entropy is None


def test_introspector_attach_attention_hooks_is_idempotent():
    m = _seeded_model()
    insp = ScorerIntrospector(m)
    insp.attach_attention_hooks()
    insp.attach_attention_hooks()
    assert insp._hook_attached
    insp.remove_hooks()
    assert not insp._hook_attached


def test_drift_raises_when_models_differ():
    class A(nn.Module):
        def forward(self, x): return x

    class B(nn.Module):
        def forward(self, x): return x

    insp_a = ScorerIntrospector(A(), capture_mode="fingerprint")
    insp_b = ScorerIntrospector(B(), capture_mode="fingerprint")
    with insp_a.session(), insp_b.session():
        r_a = insp_a.capture(torch.randn(1, 2))
        r_b = insp_b.capture(torch.randn(1, 2))
    with pytest.raises(ValueError):
        compute_layer_drift(r_a, r_b)
