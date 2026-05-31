# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the generalized substrate pixel-consumption probe.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" + Slot EEE Class 2
(tests-verify-constants-not-behavior): the headline tests below MUST FAIL if the
probe's verdict were replaced by a constant. Each behavioral test perturbs a
*real* synthetic decoder, runs a *real* render function, and asserts the verdict
flips based on the *actual* pixel delta and base variance — NOT on a canonical
constant. The archive-integration tests (guarded by skipif) run the *real*
substrate inflate reconstruct on a *real* built archive and assert the verdict is
derived from the actual rendered pixels.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_PROBE_PATH = REPO_ROOT / "tools" / "probe_substrate_archive_pixel_consumption.py"
_PROBE_MOD_NAME = "probe_substrate_archive_pixel_consumption"


def _load_probe():
    spec = importlib.util.spec_from_file_location(_PROBE_MOD_NAME, _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can resolve cls.__module__.
    sys.modules[_PROBE_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


# --------------------------------------------------------------------------- #
# Pure-logic NO-FAKE tests: the classifier reacts to the ACTUAL delta + variance.
# --------------------------------------------------------------------------- #
def test_classify_consumed_when_delta_above_threshold_and_variance_high():
    v = probe._classify_from_delta(
        max_delta=0.02,  # >> MIN_PIXEL_DELTA
        mean_delta=0.003,
        base_count=100,
        n_tensors=10,
        perturbed_key="blocks.6.weight",
        base_variance=0.11,  # >> MIN_BASE_SPATIAL_VARIANCE -> trained
    )
    assert v.verdict == probe.PIXEL_CONSUMED


def test_classify_placeholder_when_delta_below_threshold():
    # The SAME high base_variance but a SUB-THRESHOLD delta must NOT be consumed.
    # If the classifier returned a constant PIXEL_CONSUMED, this would fail.
    v = probe._classify_from_delta(
        max_delta=5e-8,  # < MIN_PIXEL_DELTA
        mean_delta=1e-9,
        base_count=100,
        n_tensors=10,
        perturbed_key="x",
        base_variance=0.11,
    )
    assert v.verdict == probe.PLACEHOLDER_OR_PARSE_GUARD


def test_classify_near_untrained_when_consumed_but_flat_base():
    # Delta IS above threshold (weights wired) but base variance is degenerate
    # (flat-gray) -> NEAR_UNTRAINED, NOT PIXEL_CONSUMED. This distinguishes a
    # mock-teacher archive (wired but no trained signal) from a real one.
    v = probe._classify_from_delta(
        max_delta=1.9e-4,  # > MIN_PIXEL_DELTA
        mean_delta=1e-6,
        base_count=100,
        n_tensors=12,
        perturbed_key="blocks.12.weight",
        base_variance=2.8e-5,  # < MIN_BASE_SPATIAL_VARIANCE -> degenerate
    )
    assert v.verdict == probe.PIXEL_CONSUMED_BUT_NEAR_UNTRAINED


def test_classify_threshold_boundary_is_strict_greater_than():
    # Exactly AT the threshold must NOT count as consumed (strict >).
    at = probe._classify_from_delta(
        probe.MIN_PIXEL_DELTA, 0.0, 1, 1, "k", base_variance=1.0
    )
    above = probe._classify_from_delta(
        probe.MIN_PIXEL_DELTA * 2, 0.0, 1, 1, "k", base_variance=1.0
    )
    assert at.verdict == probe.PLACEHOLDER_OR_PARSE_GUARD
    assert above.verdict == probe.PIXEL_CONSUMED


def test_classify_nan_delta_is_adapter_error():
    v = probe._classify_from_delta(
        float("nan"), float("nan"), 1, 1, "k", base_variance=1.0
    )
    assert v.verdict == probe.ADAPTER_ERROR


def test_base_spatial_variance_reflects_real_structure():
    flat = np.full((1000,), 0.5, dtype=np.float32)
    structured = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    assert probe._base_spatial_variance(flat) == pytest.approx(0.0, abs=1e-9)
    assert probe._base_spatial_variance(structured) > 0.2
    assert probe._base_spatial_variance(np.zeros((0,), np.float32)) == 0.0


def test_sweep_finds_the_consumed_tensor_not_the_largest_norm():
    """NO-FAKE: a synthetic render forward that ONLY reads one (small-norm) tensor
    must be detected by the sweep, even though a different tensor has the largest
    norm. This is the exact methodology subtlety the largest-norm-only approach
    missed on Z7.
    """
    import torch

    # Two tensors: "big" has huge norm but is IGNORED by render; "live" is read.
    sd = {
        "big": torch.ones(100) * 50.0,  # largest norm, ignored
        "live": torch.ones(4),  # small norm, drives the output
    }
    base = np.array([float(sd["live"].sum())], dtype=np.float32)

    def render_fn(sd2):
        # Render reads ONLY "live" — perturbing "big" must produce zero delta.
        return np.array([float(sd2["live"].sum())], dtype=np.float32)

    key, mx, _mn, _pert = probe._sweep_perturb_all_tensors(sd, render_fn, base)
    assert key == "live", "sweep must find the consumed tensor, not the big-norm one"
    assert mx > probe.MIN_PIXEL_DELTA


def test_sweep_returns_zero_when_no_tensor_consumed():
    """A render that ignores ALL tensors -> sweep finds no pixel change."""
    import torch

    sd = {"a": torch.ones(10), "b": torch.ones(10)}
    base = np.array([42.0], dtype=np.float32)

    def render_fn(_sd2):
        return np.array([42.0], dtype=np.float32)  # constant, reads nothing

    key, mx, _mn, _pert = probe._sweep_perturb_all_tensors(sd, render_fn, base)
    assert mx == 0.0
    # key is whichever tensor was probed; delta is what classifies, and it is 0.


# --------------------------------------------------------------------------- #
# Registry + axis-discipline tests.
# --------------------------------------------------------------------------- #
def test_registry_covers_renderer_and_codec_families():
    adapters = probe._adapters()
    families = {a.architecture_family for a in adapters.values()}
    assert "renderer" in families
    assert "codec" in families
    # The two empirical anchors of the audit must be registered.
    assert "z6_v2_cargo_cult_unwind" in adapters
    assert adapters["z6_v2_cargo_cult_unwind"].architecture_family == "renderer"
    assert "z8_hierarchical_predictive_coding" in adapters
    assert adapters["z8_hierarchical_predictive_coding"].architecture_family == "codec"


def test_not_built_when_archive_missing():
    m = probe.probe_substrate_archive_pixel_consumption(
        "z6_v2_cargo_cult_unwind", REPO_ROOT / "does" / "not" / "exist.bin"
    )
    assert m["trained_weight_verdict"] == probe.NOT_BUILT
    assert m["score_claim"] is False
    assert m["promotable"] is False
    assert m["axis_tag"] == "[macOS-CPU advisory]"


def test_unknown_substrate_returns_adapter_not_implemented(tmp_path):
    arc = tmp_path / "0.bin"
    arc.write_bytes(b"FAKE" + b"\x00" * 100)
    m = probe.probe_substrate_archive_pixel_consumption(
        "totally_unregistered_substrate", arc
    )
    assert m["trained_weight_verdict"] == probe.ADAPTER_NOT_IMPLEMENTED


def test_ratification_semantics_distinguish_training_from_codec():
    s_consumed = probe._ratification_semantics("renderer", probe.PIXEL_CONSUMED)
    s_codec = probe._ratification_semantics("codec", probe.CODEC_DRIVEN)
    s_placeholder = probe._ratification_semantics(
        "renderer", probe.PLACEHOLDER_OR_PARSE_GUARD
    )
    assert "TRAINING" in s_consumed
    assert "CODEC" in s_codec
    assert "NOTHING" in s_placeholder
    # The three semantics MUST be distinct (no constant collapse).
    assert len({s_consumed, s_codec, s_placeholder}) == 3


def test_manifest_never_carries_score_claim():
    """Axis discipline (Catalog #127/#192/#323/#341): this probe is NEVER a score."""
    m = probe.probe_substrate_archive_pixel_consumption(
        "nscs06_v8_chroma_lut", None
    )
    for forbidden in (
        "score_claim",
        "promotable",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
    ):
        assert m[forbidden] is False


# --------------------------------------------------------------------------- #
# Archive-integration NO-FAKE tests: run the REAL substrate inflate on a REAL
# built archive and assert the verdict is derived from actual rendered pixels.
# Guarded by skipif so a fresh checkout without the archives does not fail.
# --------------------------------------------------------------------------- #
_Z6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/z6_v2_canonical_29650ep_mlx_local_full_run_20260530T204724Z/0.bin"
)
_Z8_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/z8_m10_inflate_consumes_real_trained_weights_macos_cpu_advisory_smoke_20260530T155420Z/0.bin"
)
_Z7_ARCHIVE = (
    REPO_ROOT
    / ".omx/research/z7_mamba2_wave_n11_stabilizer_600pair_50ep_20260530T233603Z/0.bin"
)


@pytest.mark.skipif(not _Z6_ARCHIVE.is_file(), reason="Z6-v2 canonical archive absent")
def test_z6_v2_real_archive_is_pixel_consumed():
    m = probe.probe_substrate_archive_pixel_consumption(
        "z6_v2_cargo_cult_unwind", _Z6_ARCHIVE
    )
    # The canonical 29650ep MLX-LOCAL run trained a genuine renderer: perturbing
    # its decoder weights MUST move real pixels, and the base recon MUST have
    # spatial structure (NOT flat-gray). If load_state_dict(strict=False) had
    # silently dropped the trained weights, this would be PLACEHOLDER and FAIL.
    assert m["trained_weight_verdict"] == probe.PIXEL_CONSUMED
    wp = m["weight_perturbation"]
    assert wp["max_abs_pixel_delta"] > probe.MIN_PIXEL_DELTA
    assert wp["base_spatial_variance"] > probe.MIN_BASE_SPATIAL_VARIANCE
    assert m["ratification_ready_for_pc_training"] is True


@pytest.mark.skipif(not _Z8_ARCHIVE.is_file(), reason="Z8 archive absent")
def test_z8_real_archive_is_codec_driven_not_trained_renderer():
    # Re-confirms B's finding (commit 182b88406): Z8's trained categorical-posterior
    # HNeRV renderer weights are NOT pixel-consumed; pixels come from the classical
    # Mallat wavelet codec. Ratifying Z8 scores the CODEC, not the PC-training.
    m = probe.probe_substrate_archive_pixel_consumption(
        "z8_hierarchical_predictive_coding", _Z8_ARCHIVE
    )
    assert m["trained_weight_verdict"] == probe.CODEC_DRIVEN
    assert m["ratification_ready_for_pc_training"] is False
    assert "CODEC" in m["ratification_validates"]


@pytest.mark.skipif(not _Z7_ARCHIVE.is_file(), reason="Z7 stabilizer archive absent")
def test_z7_stabilizer_archive_is_near_untrained():
    # The wave_n11 stabilizer used a mock-teacher: decoder is WIRED (perturbation
    # moves pixels) but base recon is flat-gray (no trained signal). Verdict must
    # be NEAR_UNTRAINED, NOT PIXEL_CONSUMED, and NOT ratification-ready.
    m = probe.probe_substrate_archive_pixel_consumption(
        "time_traveler_l5_z7_mamba2", _Z7_ARCHIVE
    )
    assert m["trained_weight_verdict"] == probe.PIXEL_CONSUMED_BUT_NEAR_UNTRAINED
    wp = m["weight_perturbation"]
    # Wired but degenerate: delta above floor, variance below floor.
    assert wp["base_spatial_variance"] < probe.MIN_BASE_SPATIAL_VARIANCE
    assert m["ratification_ready_for_pc_training"] is False


def test_min_thresholds_are_sane():
    # Sanity: the thresholds bracket the empirical regime (fp32 jitter << floor
    # << real perturbation delta; degenerate variance << floor << trained variance).
    assert 0 < probe.MIN_PIXEL_DELTA < 1e-3
    assert 0 < probe.MIN_BASE_SPATIAL_VARIANCE < 1e-1
    assert not math.isnan(probe.MIN_PIXEL_DELTA)
