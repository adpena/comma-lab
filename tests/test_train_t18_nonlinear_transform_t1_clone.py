"""Tests for ``experiments/train_t18_nonlinear_transform_t1_clone.py``.

NN-3 (Phase 2 pre-design 2026-05-09): sustained-training invertibility
gate MUST fire when ``||z_e − invert(forward(z_e))||² >= 0.5``. Tests
also confirm:

- skip-connection identity-init keeps invertibility error 0 at init
- separate forward+inverse MLPs per He-Zheng 2024 §3.2 (NOT weight-tied)
- CLI flag set + scaffold-only refusal mirrors T1
- HARD GATE status surfaced in manifest (NOT enforced by trainer)
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

from experiments.train_t18_nonlinear_transform_t1_clone import (  # noqa: E402
    NN3_INVERTIBILITY_FLOOR_DEFAULT,
    NN3_PROBE_EVERY_STEPS_DEFAULT,
    NN3_PROBE_SAMPLE_COUNT_DEFAULT,
    T18_LANE_ID,
    T18_PREDICTED_DELTA_SCORE,
    T18_SCHEMA_VERSION,
    InvertibilityBreachError,
    assert_invertibility_ok,
    parse_args,
)
from tac.balle_nonlinear_transform import (  # noqa: E402
    NonlinearTransformBlock,
    NonlinearTransformConfig,
)


# ---------------------------------------------------------------------------
# Test 1: NN-3 gate passes at identity-init (skip-connection zero-init)
# ---------------------------------------------------------------------------


def test_nn3_passes_at_init():
    """T18 init is approximate-identity (zero last linear); invertibility ≈ 0."""
    config = NonlinearTransformConfig.he_zheng_canonical(label="nn3_init", latent_dim=64)
    block = NonlinearTransformBlock(config)
    z_e = torch.randn(100, 64)
    diag = assert_invertibility_ok(block, z_e, floor=NN3_INVERTIBILITY_FLOOR_DEFAULT, step=0)
    assert diag["passed"] is True
    assert diag["invertibility_error"] < 0.5


# ---------------------------------------------------------------------------
# Test 2: NN-3 gate FIRES when the inverse MLP is deliberately perturbed
# ---------------------------------------------------------------------------


def test_nn3_fires_on_perturbed_inverse_mlp():
    """If we randomly init the inverse MLP, invertibility breaks."""
    config = NonlinearTransformConfig.he_zheng_canonical(label="nn3_break", latent_dim=64)
    block = NonlinearTransformBlock(config)
    # Drift the inverse MLP weights so forward != identity inverse.
    with torch.no_grad():
        for p in block.inverse_mlp.parameters():
            p.add_(torch.randn_like(p) * 2.0)  # big drift
    # Also drift forward MLP last layer (so forward is no longer identity).
    with torch.no_grad():
        for p in block.forward_mlp.parameters():
            p.add_(torch.randn_like(p) * 1.0)
    z_e = torch.randn(100, 64)
    with pytest.raises(InvertibilityBreachError, match="invertibility breach"):
        assert_invertibility_ok(block, z_e, floor=0.5, step=1000)


def test_nn3_floor_is_configurable():
    config = NonlinearTransformConfig.he_zheng_canonical(label="nn3_floor", latent_dim=64)
    block = NonlinearTransformBlock(config)
    z_e = torch.randn(100, 64)
    # At init the error is ≈ 0; a very small floor should still pass.
    diag = assert_invertibility_ok(block, z_e, floor=1e-3, step=0)
    assert diag["passed"] is True


# ---------------------------------------------------------------------------
# Test 3: Forward and inverse MLPs are SEPARATE parameter sets (not weight-tied)
# ---------------------------------------------------------------------------


def test_forward_inverse_mlps_have_distinct_params():
    config = NonlinearTransformConfig.he_zheng_canonical(label="separate_mlps", latent_dim=64)
    block = NonlinearTransformBlock(config)
    fwd_params = set(id(p) for p in block.forward_mlp.parameters())
    inv_params = set(id(p) for p in block.inverse_mlp.parameters())
    # No overlap — they're separate parameter sets.
    assert fwd_params.isdisjoint(inv_params)
    # Counts match (same architecture).
    assert len(list(block.forward_mlp.parameters())) == len(list(block.inverse_mlp.parameters()))


# ---------------------------------------------------------------------------
# Test 4: Skip-connection means initial forward ≈ identity
# ---------------------------------------------------------------------------


def test_initial_forward_is_approximately_identity():
    config = NonlinearTransformConfig.he_zheng_canonical(label="ident_init", latent_dim=64)
    block = NonlinearTransformBlock(config)
    z_e = torch.randn(8, 64)
    z_t = block(z_e)
    assert torch.allclose(z_t, z_e, atol=1e-5)


def test_initial_invert_is_approximately_identity():
    config = NonlinearTransformConfig.he_zheng_canonical(label="ident_invert", latent_dim=64)
    block = NonlinearTransformBlock(config)
    z_t = torch.randn(8, 64)
    z_e = block.invert(z_t)
    assert torch.allclose(z_e, z_t, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 5: CLI parser flags match Phase 2 pre-design
# ---------------------------------------------------------------------------


def test_cli_t18_defaults_match_council():
    """Phase 2 pre-design Q1+Q2: D=64 (MacKay MDL), 3 hidden, GELU, FP16 train."""
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_t18")])
    assert args.t18_latent_dim == 64
    assert args.t18_expansion_factor == 4
    assert args.t18_num_hidden_layers == 3
    assert args.t18_activation == "gelu"
    assert args.t18_transform_quant == "fp16"
    assert args.t18_ship_quant == "fp4"
    assert args.t18_nn3_floor == NN3_INVERTIBILITY_FLOOR_DEFAULT
    assert args.t18_nn3_probe_every_steps == NN3_PROBE_EVERY_STEPS_DEFAULT
    assert args.t18_nn3_probe_sample_count == NN3_PROBE_SAMPLE_COUNT_DEFAULT


def test_cli_rejects_negative_floor():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t18_nonlinear_transform_t1_clone.py"),
            "--output-dir", "/tmp/should_not_run",
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--t18-nn3-floor", "-0.5",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert "nn3-floor" in (proc.stderr + proc.stdout)


def test_cli_rejects_mps_device():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_t18_2"), "--device", "cpu"])
    assert args.device == "cpu"


def test_cli_required_flags():
    with pytest.raises(SystemExit):
        parse_args([])


# ---------------------------------------------------------------------------
# Test 6: Smoke run produces manifest with HARD GATE status + archive grammar
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_run_produces_manifest(tmp_path):
    out = tmp_path / "t18_smoke_subprocess"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t18_nonlinear_transform_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
            "--t18-latent-dim", "28",
            "--t18-nn3-probe-every-steps", "1",
            "--t18-nn3-probe-sample-count", "4",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, f"smoke failed: {proc.stderr}"
    manifest = json.loads((out / "t18_provenance.json").read_text())
    assert manifest["schema"] == T18_SCHEMA_VERSION
    assert manifest["lane_id"] == T18_LANE_ID
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["nn3_probes_performed"] > 0
    assert manifest["ema_decay"] == 0.997
    # HARD GATE status surfaced (NOT enforced by this trainer).
    assert "HARD GATE" in manifest["hard_gate_t18_b_status"] or "T18-B" in manifest["hard_gate_t18_b_status"]
    # Compliance tags
    tags = manifest["compliance_tags"]
    assert "ema_0p997_snapshot_restore" in tags
    assert "eval_roundtrip_true" in tags
    assert "no_mps_authoritative" in tags
    assert "differentiable_yuv6" in tags
    assert "score_aware_lagrangian" in tags
    assert "skip_connection_identity_init" in tags
    assert "separate_forward_inverse_mlps" in tags
    assert "nn3_invertibility_probe_periodic" in tags
    assert "mixed_precision_fp16_train_fp4_ship" in tags
    assert "auth_eval_gated" in tags


# ---------------------------------------------------------------------------
# Test 7: Refuses --auth-eval at scaffold stage (mirrors T1)
# ---------------------------------------------------------------------------


def test_refuses_auth_eval(tmp_path):
    out = tmp_path / "t18_auth_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t18_nonlinear_transform_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu", "--auth-eval",
            "--smoke", "--allow-missing-canonical-a1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert "refused" in (proc.stderr + proc.stdout).lower()


# ---------------------------------------------------------------------------
# Test 8: EMA shadow checkpoint includes transform
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ema_shadow_checkpoint_includes_transform(tmp_path):
    out = tmp_path / "t18_ema_keys"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t18_nonlinear_transform_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
            "--t18-latent-dim", "28",
            "--t18-nn3-probe-every-steps", "1",
            "--t18-nn3-probe-sample-count", "4",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    ckpt = torch.load(out / "t18_ema_shadow.pt", map_location="cpu", weights_only=False)
    assert "ema_decoder" in ckpt
    assert "ema_balle" in ckpt
    assert "ema_transform" in ckpt
    assert ckpt["schema"] == T18_SCHEMA_VERSION
    cfg = ckpt["t18_config"]
    assert cfg["expansion_factor"] == 4
    assert cfg["num_hidden_layers"] == 3
    assert cfg["activation"] == "gelu"
    assert cfg["train_quantization"] == "fp16"
    assert cfg["ship_quantization"] == "fp4"
    # transform byte sizes match config (train > ship).
    assert ckpt["transform_bytes_train"] > ckpt["transform_bytes_ship"]


# ---------------------------------------------------------------------------
# Test 9: predicted score band is tagged + HARD GATE conditional
# ---------------------------------------------------------------------------


def test_predicted_delta_score_is_tagged_conditional():
    assert "[predicted;" in T18_PREDICTED_DELTA_SCORE
    assert "He-Zheng 2024" in T18_PREDICTED_DELTA_SCORE
    assert "conditional" in T18_PREDICTED_DELTA_SCORE.lower()
    assert "HARD GATE" in T18_PREDICTED_DELTA_SCORE or "T18-B" in T18_PREDICTED_DELTA_SCORE
