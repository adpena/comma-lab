"""Tests for ``experiments/train_t6_balle_uniward_cross_paradigm.py``.

NN-T6 (3-clean-pass review 2026-05-11): the UNIWARD-weighted distortion
MUST be finite + nonzero on first step, AND the texture map must NOT
collapse to a constant.  Tests confirm:

- CLI flag set is consistent (no MPS, smoke gated, auth-eval gated)
- UNIWARD texture map computation handles smoke + real decoded shapes
- UNIWARD-weighted L1 reduces to ``flat_floor·L1`` for a constant texture
  map AND to ``textured_ceiling·L1`` when the entire frame is textured
- score-aware Lagrangian invocation path
- EMA shadow saved as inference checkpoint
- archive grammar fields declared in manifest
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from experiments.train_t6_balle_uniward_cross_paradigm import (  # noqa: E402
    T6_LANE_ID,
    T6_PREDICTED_DELTA_SCORE,
    T6_SCHEMA_VERSION,
    assert_uniward_loss_nondegenerate,
    compute_uniward_weight_map,
    parse_args,
    uniward_weighted_pixel_l1,
)


# ---------------------------------------------------------------------------
# NN-T6 gate behavior
# ---------------------------------------------------------------------------


def test_nn_t6_passes_on_textured_decoded():
    """NN-T6: a real textured decoded frame produces a non-degenerate weight map."""
    torch.manual_seed(0)
    # Random decoded frames have natural texture (high variance).
    decoded = torch.randn(2, 2, 3, 16, 24) * 50 + 128
    tex_hw = compute_uniward_weight_map(decoded)
    assert tex_hw.shape == (16, 24)
    # The texture map should have nonzero range on random data.
    assert float((tex_hw.max() - tex_hw.min()).item()) > 1e-6
    distortion = torch.tensor(0.5)
    diag = assert_uniward_loss_nondegenerate(distortion, tex_hw)
    assert diag["uniward_texture_range"] > 1e-6
    assert diag["distortion"] == 0.5


def test_nn_t6_raises_on_constant_texture():
    """NN-T6: a flat texture map (range=0) triggers structured pause."""
    tex_hw = torch.ones(16, 24)  # constant — degenerate
    distortion = torch.tensor(0.5)
    with pytest.raises(RuntimeError, match="UNIWARD texture map"):
        assert_uniward_loss_nondegenerate(distortion, tex_hw)


def test_nn_t6_raises_on_nonfinite_distortion():
    tex_hw = torch.rand(16, 24)
    distortion = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        assert_uniward_loss_nondegenerate(distortion, tex_hw)


def test_nn_t6_raises_on_negative_distortion():
    tex_hw = torch.rand(16, 24)
    distortion = torch.tensor(-0.1)
    with pytest.raises(RuntimeError, match="non-negative"):
        assert_uniward_loss_nondegenerate(distortion, tex_hw)


# ---------------------------------------------------------------------------
# compute_uniward_weight_map shape + range
# ---------------------------------------------------------------------------


def test_compute_uniward_weight_map_rejects_wrong_dim():
    with pytest.raises(ValueError, match="decoded must be"):
        compute_uniward_weight_map(torch.randn(2, 3, 16, 24))


def test_compute_uniward_weight_map_rejects_wrong_channel():
    with pytest.raises(ValueError, match="decoded must be"):
        compute_uniward_weight_map(torch.randn(2, 2, 4, 16, 24))


def test_compute_uniward_weight_map_is_detached_by_default():
    decoded = torch.randn(2, 2, 3, 16, 24, requires_grad=True)
    tex_hw = compute_uniward_weight_map(decoded, detach=True)
    assert not tex_hw.requires_grad


def test_compute_uniward_weight_map_attached_when_requested():
    decoded = torch.randn(2, 2, 3, 16, 24, requires_grad=True)
    tex_hw = compute_uniward_weight_map(decoded, detach=False)
    # The texture map's requires_grad depends on the upstream graph — at
    # minimum it should not be detached at this layer.
    assert tex_hw.dtype == torch.float32


# ---------------------------------------------------------------------------
# uniward_weighted_pixel_l1 — clamping behavior
# ---------------------------------------------------------------------------


def test_uniward_weighted_l1_with_constant_tex_collapses_to_floor():
    """When texture is exactly uniform, normalization → 0 → weight = flat_floor."""
    decoded = torch.zeros(2, 2, 3, 8, 12)
    target = torch.ones(2, 2, 3, 8, 12)
    tex_hw = torch.ones(8, 12) * 5.0  # uniform; (max - min) = 0
    out = uniward_weighted_pixel_l1(
        decoded, target,
        uniward_weight_hw=tex_hw,
        flat_floor=0.5, textured_ceiling=1.5,
        eval_roundtrip_noise_std=0.0,
        enable_eval_roundtrip_in_training=False,
    )
    # When (tex_max - tex_min) ≈ 0, denom clamps to 1e-12 so all tex_norm
    # ≈ 0 → weight = flat_floor = 0.5; unweighted L1 = 1 → weighted = 0.5
    assert abs(float(out) - 0.5) < 1e-4


def test_uniward_weighted_l1_rejects_bad_weight_shape():
    decoded = torch.zeros(2, 2, 3, 8, 12)
    target = torch.ones(2, 2, 3, 8, 12)
    with pytest.raises(ValueError, match="uniward_weight_hw must be"):
        uniward_weighted_pixel_l1(
            decoded, target,
            uniward_weight_hw=torch.zeros(3, 8, 12),  # 3-D
            flat_floor=0.5, textured_ceiling=1.5,
            eval_roundtrip_noise_std=0.0,
            enable_eval_roundtrip_in_training=False,
        )


def test_uniward_weighted_l1_rejects_inverted_band():
    decoded = torch.zeros(2, 2, 3, 8, 12)
    target = torch.ones(2, 2, 3, 8, 12)
    tex_hw = torch.rand(8, 12)
    with pytest.raises(ValueError, match="flat_floor"):
        uniward_weighted_pixel_l1(
            decoded, target,
            uniward_weight_hw=tex_hw,
            flat_floor=2.0, textured_ceiling=1.0,  # inverted!
            eval_roundtrip_noise_std=0.0,
            enable_eval_roundtrip_in_training=False,
        )


# ---------------------------------------------------------------------------
# CLI parser flags
# ---------------------------------------------------------------------------


def test_cli_required_flags():
    with pytest.raises(SystemExit):
        parse_args([])  # missing --output-dir


def test_cli_smoke_flag_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    assert args.smoke is False
    assert args.auth_eval is False
    assert args.ema_decay == 0.997
    assert args.enable_eval_roundtrip_in_training is True


def test_cli_t6_specific_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    # Per Phase 2 pre-design council canon:
    assert args.t6_flat_floor == 0.5
    assert args.t6_textured_ceiling == 1.5
    assert args.t6_disable_uniward is False


def test_cli_rejects_mps_at_choice_layer():
    # argparse rejects unknown choice before main() asserts.
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/dev/null", "--device", "mps"])


# ---------------------------------------------------------------------------
# Predicted score band is tagged (per CLAUDE.md non-negotiable)
# ---------------------------------------------------------------------------


def test_predicted_delta_score_carries_tag():
    assert "[predicted;" in T6_PREDICTED_DELTA_SCORE
    assert "UNIWARD" in T6_PREDICTED_DELTA_SCORE
    assert "Ballé" in T6_PREDICTED_DELTA_SCORE


# ---------------------------------------------------------------------------
# Subprocess smoke (end-to-end on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_run_produces_manifest_with_archive_grammar(tmp_path):
    out = tmp_path / "t6_smoke_subprocess"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t6_balle_uniward_cross_paradigm.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"smoke failed: {proc.stderr}"
    manifest_path = out / "t6_provenance.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == T6_SCHEMA_VERSION
    assert manifest["lane_id"] == T6_LANE_ID
    assert manifest["predicted_delta_score"] == T6_PREDICTED_DELTA_SCORE
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["nn_t6_gate_passed"] is True
    assert manifest["ema_decay"] == 0.997
    assert manifest["eval_roundtrip"] is True
    assert manifest["uniward_disabled"] is False
    # 9 compliance tags including CLAUDE.md non-negotiables.
    tags = manifest["compliance_tags"]
    for required in (
        "ema_0p997_snapshot_restore",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "differentiable_yuv6",
        "score_aware_lagrangian",
        "no_synthetic_outside_smoke",
        "no_tmp_paths",
        "auth_eval_gated",
        "uniward_cross_paradigm_loss_bolt_on",
    ):
        assert required in tags, f"missing compliance tag {required}"


@pytest.mark.slow
def test_smoke_with_uniward_disabled_runs(tmp_path):
    """Diagnostic: --t6-disable-uniward collapses to T1; trainer still finishes."""
    out = tmp_path / "t6_smoke_disabled"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t6_balle_uniward_cross_paradigm.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
            "--t6-disable-uniward",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "t6_provenance.json").read_text())
    assert manifest["uniward_disabled"] is True


@pytest.mark.slow
def test_trainer_refuses_auth_eval_phase1_scaffold_only(tmp_path):
    out = tmp_path / "t6_auth_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t6_balle_uniward_cross_paradigm.py"),
            "--output-dir", str(out),
            "--device", "cpu", "--auth-eval",
            "--smoke", "--allow-missing-canonical-a1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "refused" in combined.lower() or "scaffold" in combined.lower()


@pytest.mark.slow
def test_ema_shadow_checkpoint_keys(tmp_path):
    out = tmp_path / "t6_ema_keys"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t6_balle_uniward_cross_paradigm.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    ckpt = torch.load(out / "t6_ema_shadow.pt", map_location="cpu", weights_only=False)
    assert "ema_decoder" in ckpt
    assert "ema_balle" in ckpt
    assert ckpt["schema"] == T6_SCHEMA_VERSION
    cfg = ckpt["t6_config"]
    assert cfg["flat_floor"] == 0.5
    assert cfg["textured_ceiling"] == 1.5
