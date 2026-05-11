"""Tests for the four Phase 2 pre-design probe scaffolds.

Per CLAUDE.md "Subagent coherence-by-default": these are SCAFFOLDS that
emit typed-atom rows for the cathedral autopilot. They never dispatch GPU
jobs themselves. The tests cover the pure helpers + the typed-atom row
emission + the CLI entry points.

Coverage:
  - probe_t17_a_codebook_perplexity_smoke: 12 tests
  - probe_t17_b_codebook_sparsity:          12 tests
  - probe_t18_a_invertibility_smoke:        13 tests
  - probe_t18_b_hard_gate_byte_savings:     14 tests
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def _load_module(name: str):
    path = EXPERIMENTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t17a = _load_module("probe_t17_a_codebook_perplexity_smoke")
t17b = _load_module("probe_t17_b_codebook_sparsity")
t18a = _load_module("probe_t18_a_invertibility_smoke")
t18b = _load_module("probe_t18_b_hard_gate_byte_savings")


# ── T17-A codebook perplexity smoke ────────────────────────────────────────


def test_t17a_default_config_caps():
    cfg = t17a.T17ASmokeConfig()
    assert cfg.smoke_iters == 200
    assert cfg.codebook_size == 512
    assert cfg.estimated_cost_usd == 2.0
    assert cfg.operator_authorized is False


def test_t17a_perplexity_uniform_equals_K():
    counts = [10] * 8
    p = t17a.codebook_perplexity_from_counts(counts)
    assert math.isclose(p, 8.0, rel_tol=1e-9)


def test_t17a_perplexity_collapsed_to_one_code_equals_1():
    counts = [100, 0, 0, 0, 0]
    p = t17a.codebook_perplexity_from_counts(counts)
    assert math.isclose(p, 1.0, rel_tol=1e-9)


def test_t17a_perplexity_zero_total_returns_zero():
    assert t17a.codebook_perplexity_from_counts([0, 0, 0]) == 0.0


def test_t17a_gate_threshold_is_ratio_times_K():
    cfg = t17a.T17ASmokeConfig(codebook_size=512, perplexity_gate_ratio=0.25)
    assert t17a.perplexity_gate_threshold(cfg) == 128.0


def test_t17a_assess_collapse_below_threshold():
    cfg = t17a.T17ASmokeConfig(codebook_size=512)
    # 50 perplexity points below threshold = 128
    trace = [50.0] * 20
    ok, reason = t17a.assess_perplexity_trajectory(trace, cfg)
    assert ok is False
    assert "collapse" in reason


def test_t17a_assess_above_threshold_passes():
    cfg = t17a.T17ASmokeConfig(codebook_size=512)
    trace = [400.0] * 20
    ok, reason = t17a.assess_perplexity_trajectory(trace, cfg)
    assert ok is True
    assert "above threshold" in reason


def test_t17a_assess_no_samples_fails():
    cfg = t17a.T17ASmokeConfig()
    ok, reason = t17a.assess_perplexity_trajectory([], cfg)
    assert ok is False
    assert "no perplexity samples" in reason


def test_t17a_typed_atom_row_default_blockers_include_operator_auth():
    cfg = t17a.T17ASmokeConfig()
    row = t17a.emit_typed_atom_row(cfg)
    assert "operator_authorization_required_for_dispatch" in row.blockers


def test_t17a_typed_atom_row_clears_blocker_when_authorized():
    cfg = t17a.T17ASmokeConfig(operator_authorized=True)
    row = t17a.emit_typed_atom_row(cfg)
    assert "operator_authorization_required_for_dispatch" not in row.blockers


def test_t17a_main_writes_output(tmp_path, capsys):
    out = tmp_path / "row.json"
    rc = t17a.main(["--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["probe_name"] == "t17_a_codebook_perplexity_smoke"
    assert payload["smoke_only"] is True
    assert payload["estimated_dispatch_cost_usd"] == 2.0


def test_t17a_main_rejects_invalid_smoke_iters(capsys):
    rc = t17a.main(["--smoke-iters", "0"])
    assert rc == 2
    assert "smoke-iters" in capsys.readouterr().err


# ── T17-B codebook sparsity ────────────────────────────────────────────────


def test_t17b_default_config():
    cfg = t17b.T17BSmokeConfig()
    assert cfg.codebook_size == 512
    assert cfg.sparsity_floor_bits == 0.50
    assert cfg.estimated_cost_usd == 2.0


def test_t17b_usage_entropy_uniform_equals_log2K():
    counts = [10] * 8
    h = t17b.usage_entropy_bits(counts)
    assert math.isclose(h, 3.0, rel_tol=1e-9)


def test_t17b_usage_entropy_concentrated_is_low():
    counts = [100, 1, 1, 1]
    h = t17b.usage_entropy_bits(counts)
    assert h < 1.0


def test_t17b_usage_entropy_zero_total_returns_zero():
    assert t17b.usage_entropy_bits([0, 0]) == 0.0


def test_t17b_ceiling_is_log2_K():
    assert t17b.usage_entropy_ceiling_bits(512) == 9.0
    assert t17b.usage_entropy_ceiling_bits(1) == 0.0


def test_t17b_assess_uniform_fails_sparsity_gate():
    cfg = t17b.T17BSmokeConfig(codebook_size=8, sparsity_floor_bits=0.5)
    counts = [10] * 8  # entropy = 3.0; ceiling = 3.0; target = 2.5
    ok, reason = t17b.assess_sparsity(counts, cfg)
    assert ok is False
    assert "no sparsity emerged" in reason


def test_t17b_assess_sparse_passes_gate():
    cfg = t17b.T17BSmokeConfig(codebook_size=8, sparsity_floor_bits=0.5)
    counts = [100, 1, 1, 1, 1, 1, 1, 1]
    ok, reason = t17b.assess_sparsity(counts, cfg)
    assert ok is True


def test_t17b_assess_no_counts_fails():
    cfg = t17b.T17BSmokeConfig()
    ok, reason = t17b.assess_sparsity([], cfg)
    assert ok is False


def test_t17b_typed_atom_row_default_blockers():
    cfg = t17b.T17BSmokeConfig()
    row = t17b.emit_typed_atom_row(cfg)
    assert "operator_authorization_required_for_dispatch" in row.blockers


def test_t17b_typed_atom_row_when_authorized():
    cfg = t17b.T17BSmokeConfig(operator_authorized=True)
    row = t17b.emit_typed_atom_row(cfg)
    assert row.blockers == []


def test_t17b_main_writes_output(tmp_path):
    out = tmp_path / "row.json"
    rc = t17b.main(["--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["probe_name"] == "t17_b_codebook_sparsity"


def test_t17b_main_rejects_invalid_floor(capsys):
    rc = t17b.main(["--sparsity-floor-bits", "0"])
    assert rc == 2


# ── T18-A invertibility smoke ──────────────────────────────────────────────


def test_t18a_default_config():
    cfg = t18a.T18ASmokeConfig()
    assert cfg.invertibility_floor == 0.50
    assert cfg.sustain_window == 50
    assert cfg.estimated_cost_usd == 2.0


def test_t18a_assess_clean_trace_passes():
    cfg = t18a.T18ASmokeConfig()
    trace = [0.01] * 200
    ok, reason = t18a.assess_invertibility(trace, cfg)
    assert ok is True
    assert "invertibility gate passed" in reason


def test_t18a_assess_sustained_breach_fails():
    cfg = t18a.T18ASmokeConfig(sustain_window=10)
    # 10+ consecutive over-floor samples = sustained breach
    trace = [0.01] * 20 + [1.5] * 15 + [0.01] * 20
    ok, reason = t18a.assess_invertibility(trace, cfg)
    assert ok is False
    assert "NN-3 invertibility breach" in reason


def test_t18a_assess_single_spike_does_not_fail():
    cfg = t18a.T18ASmokeConfig(sustain_window=50)
    trace = [0.01] * 100 + [10.0] + [0.01] * 100
    ok, _ = t18a.assess_invertibility(trace, cfg)
    assert ok is True


def test_t18a_assess_empty_trace_fails():
    cfg = t18a.T18ASmokeConfig()
    ok, reason = t18a.assess_invertibility([], cfg)
    assert ok is False


def test_t18a_assess_short_trace_below_window_passes():
    # If trace is shorter than sustain_window, can't sustain
    cfg = t18a.T18ASmokeConfig(sustain_window=50)
    trace = [10.0] * 20  # all over-floor but only 20 samples
    ok, _ = t18a.assess_invertibility(trace, cfg)
    assert ok is True  # not sustained over 50


def test_t18a_typed_atom_row_blockers_default():
    cfg = t18a.T18ASmokeConfig()
    row = t18a.emit_typed_atom_row(cfg)
    assert "operator_authorization_required_for_dispatch" in row.blockers


def test_t18a_typed_atom_row_when_authorized():
    cfg = t18a.T18ASmokeConfig(operator_authorized=True)
    row = t18a.emit_typed_atom_row(cfg)
    assert row.blockers == []


def test_t18a_typed_atom_row_carries_extra_blockers():
    cfg = t18a.T18ASmokeConfig(operator_authorized=True)
    row = t18a.emit_typed_atom_row(
        cfg, extra_blockers=["modal_gpu_unavailable"],
    )
    assert "modal_gpu_unavailable" in row.blockers


def test_t18a_main_writes_output(tmp_path):
    out = tmp_path / "row.json"
    rc = t18a.main(["--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["probe_name"] == "t18_a_invertibility_smoke"


def test_t18a_main_rejects_invalid_floor(capsys):
    rc = t18a.main(["--invertibility-floor", "0"])
    assert rc == 2


def test_t18a_main_rejects_zero_iters(capsys):
    rc = t18a.main(["--smoke-iters", "0"])
    assert rc == 2


def test_t18a_typed_atom_row_predicted_score_delta_is_zero():
    # Smoke probe never claims a score delta
    cfg = t18a.T18ASmokeConfig()
    row = t18a.emit_typed_atom_row(cfg)
    assert row.predicted_score_delta == 0.0


# ── T18-B HARD GATE byte savings ───────────────────────────────────────────


def test_t18b_default_config():
    cfg = t18b.T18BHardGateConfig()
    assert cfg.minimum_savings_ratio == 0.01
    assert cfg.slice_frames == 64
    assert cfg.estimated_cost_usd == 2.0


def test_t18b_measurement_savings_ratio():
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=10000, t18_bytes=8000, slice_frames=64,
    )
    assert m.absolute_savings_bytes == 2000
    assert m.savings_ratio == 0.20


def test_t18b_measurement_zero_baseline_returns_zero_ratio():
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=0, t18_bytes=0, slice_frames=64,
    )
    assert m.savings_ratio == 0.0


def test_t18b_hard_gate_refuses_no_savings():
    cfg = t18b.T18BHardGateConfig()
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=10000, t18_bytes=10000, slice_frames=64,
    )
    ok, reason = t18b.assess_hard_gate(m, cfg)
    assert ok is False
    assert "HARD GATE refused" in reason
    assert "no net savings" in reason


def test_t18b_hard_gate_refuses_negative_savings():
    cfg = t18b.T18BHardGateConfig()
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=10000, t18_bytes=11000, slice_frames=64,
    )
    ok, reason = t18b.assess_hard_gate(m, cfg)
    assert ok is False


def test_t18b_hard_gate_refuses_below_minimum_ratio():
    cfg = t18b.T18BHardGateConfig(minimum_savings_ratio=0.05)
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=10000, t18_bytes=9700, slice_frames=64,
    )
    # 3% savings < 5% required
    ok, reason = t18b.assess_hard_gate(m, cfg)
    assert ok is False
    assert "minimum" in reason


def test_t18b_hard_gate_passes_above_minimum_ratio():
    cfg = t18b.T18BHardGateConfig(minimum_savings_ratio=0.01)
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=10000, t18_bytes=9800, slice_frames=64,
    )
    # 2% savings > 1% required
    ok, reason = t18b.assess_hard_gate(m, cfg)
    assert ok is True
    assert "HARD GATE passed" in reason


def test_t18b_hard_gate_zero_baseline_fails():
    cfg = t18b.T18BHardGateConfig()
    m = t18b.ByteSavingsMeasurement(
        baseline_bytes=0, t18_bytes=100, slice_frames=64,
    )
    ok, reason = t18b.assess_hard_gate(m, cfg)
    assert ok is False
    assert "baseline_bytes" in reason


def test_t18b_typed_atom_row_default_blockers():
    cfg = t18b.T18BHardGateConfig()
    row = t18b.emit_typed_atom_row(cfg)
    assert "operator_authorization_required_for_dispatch" in row.blockers


def test_t18b_typed_atom_row_when_authorized():
    cfg = t18b.T18BHardGateConfig(operator_authorized=True)
    row = t18b.emit_typed_atom_row(cfg)
    assert row.blockers == []


def test_t18b_typed_atom_row_includes_unanimous_council_tag():
    cfg = t18b.T18BHardGateConfig()
    row = t18b.emit_typed_atom_row(cfg)
    assert "hard_gate_byte_savings_unanimous_council" in row.claude_md_compliance_tags


def test_t18b_main_writes_output(tmp_path):
    out = tmp_path / "row.json"
    rc = t18b.main(["--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["probe_name"] == "t18_b_hard_gate_byte_savings"


def test_t18b_main_rejects_invalid_savings_ratio(capsys):
    rc = t18b.main(["--minimum-savings-ratio", "1.5"])
    assert rc == 2


def test_t18b_main_rejects_zero_slice_frames(capsys):
    rc = t18b.main(["--slice-frames", "0"])
    assert rc == 2


# ── Cross-probe consistency ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module",
    [t17a, t17b, t18a, t18b],
)
def test_all_probes_share_lane_id(module):
    # All four probes belong to the same Phase 2 probe lane.
    if hasattr(module, "PROBE_LANE_ID"):
        assert module.PROBE_LANE_ID == "lane_phase2_probes_t17ab_t18ab"


@pytest.mark.parametrize(
    "module",
    [t17a, t17b, t18a, t18b],
)
def test_all_probes_smoke_only_default(module):
    # Every probe defaults to smoke-only; full dispatch is gated on operator.
    cfg_cls_name = next(
        n for n in dir(module) if n.endswith("Config") and not n.startswith("_")
    )
    cfg = getattr(module, cfg_cls_name)()
    assert cfg.operator_authorized is False
    assert cfg.estimated_cost_usd == 2.00


@pytest.mark.parametrize(
    "module",
    [t17a, t17b, t18a, t18b],
)
def test_all_probes_emit_typed_atom_with_evidence_tag(module):
    cfg_cls_name = next(
        n for n in dir(module) if n.endswith("Config") and not n.startswith("_")
    )
    cfg = getattr(module, cfg_cls_name)()
    row = module.emit_typed_atom_row(cfg)
    assert "[predicted;" in row.evidence_grade
    assert "operator_gate_non_negotiable_at_every_dispatch" in row.claude_md_compliance_tags
