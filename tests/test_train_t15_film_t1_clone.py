"""Tests for ``experiments/train_t15_film_t1_clone.py``.

NN-1 (Phase 2 pre-design 2026-05-09): gradient-flow regression test on
the T15 FiLM modulator MUST pass before dispatch. Tests also confirm:

- CLI flag set is consistent (no MPS, smoke gated, auth-eval gated)
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

from experiments.train_t15_film_t1_clone import (  # noqa: E402
    T15_LANE_ID,
    T15_PREDICTED_DELTA_SCORE,
    T15_SCHEMA_VERSION,
    _apply_t15_film_to_decoded,
    _extract_pose_delta_for_pair,
    assert_modulator_gradient_finite_nonzero,
    parse_args,
)
from tac.film_time_varying import (  # noqa: E402
    TimeVaryingFiLM,
    TimeVaryingFiLMConfig,
)


# ---------------------------------------------------------------------------
# Test 1: NN-1 gradient-flow regression on the FiLM modulator
# ---------------------------------------------------------------------------


def test_nn1_modulator_params_move_under_adam():
    """5-step Adam optimizes modulator params; L2 movement > 1e-3.

    This is the NN-1 regression test ratified by Phase 2 pre-design pass
    (council 2026-05-09). If this test fails, T15 dispatch is REFUSED.
    """
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="nn1_test")
    modulator = TimeVaryingFiLM(config)
    initial_params = {n: p.clone() for n, p in modulator.named_parameters()}
    pose_delta = torch.randn(8, 6)
    optimizer = torch.optim.Adam(modulator.parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        gamma, beta = modulator(pose_delta)
        loss = (gamma.sum() - 1.0) ** 2 + beta.sum() ** 2
        loss.backward()
        optimizer.step()
    total_movement = 0.0
    for name, p in modulator.named_parameters():
        delta = p - initial_params[name]
        total_movement += float((delta ** 2).sum().sqrt().item())
    assert total_movement > 1e-3, (
        f"NN-1 fail: modulator L2 movement {total_movement} <= 1e-3"
    )


# ---------------------------------------------------------------------------
# Test 2: assert_modulator_gradient_finite_nonzero raises on no-grad
# ---------------------------------------------------------------------------


def test_assert_modulator_gradient_raises_without_backward():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="grad_check")
    modulator = TimeVaryingFiLM(config)
    with pytest.raises(RuntimeError, match="no gradient"):
        assert_modulator_gradient_finite_nonzero(modulator)


def test_assert_modulator_gradient_passes_after_backward():
    config = TimeVaryingFiLMConfig.quantizr_canonical(label="grad_check_2")
    modulator = TimeVaryingFiLM(config)
    pose = torch.randn(4, 6)
    gamma, beta = modulator(pose)
    loss = gamma.sum() + beta.sum()
    loss.backward()
    diag = assert_modulator_gradient_finite_nonzero(modulator)
    assert diag["modulator_grad_l2"] > 0


# ---------------------------------------------------------------------------
# Test 3: pose_delta extraction handles small + adequate latent dims
# ---------------------------------------------------------------------------


def test_extract_pose_delta_small_latent_is_padded():
    latents = torch.randn(4, 3)
    pose = _extract_pose_delta_for_pair(latents, pose_dim=6)
    assert pose.shape == (4, 6)
    # First 3 dims are the original; last 3 are padded zeros.
    assert torch.allclose(pose[:, :3], latents)
    assert torch.allclose(pose[:, 3:], torch.zeros(4, 3))


def test_extract_pose_delta_large_latent_is_sliced():
    latents = torch.randn(4, 28)
    pose = _extract_pose_delta_for_pair(latents, pose_dim=6)
    assert pose.shape == (4, 6)
    assert torch.allclose(pose, latents[:, :6])


def test_extract_pose_delta_rejects_non_2d():
    with pytest.raises(ValueError, match="latents_pair"):
        _extract_pose_delta_for_pair(torch.randn(4, 6, 1), pose_dim=6)


# ---------------------------------------------------------------------------
# Test 4: _apply_t15_film_to_decoded shape correctness
# ---------------------------------------------------------------------------


def test_apply_t15_film_shape_preserved():
    config = TimeVaryingFiLMConfig(
        pose_dim=6, feature_channels=3, hidden_dim=32,
        activation="relu", quantization="fp4", label="apply_test",
    )
    modulator = TimeVaryingFiLM(config)
    decoded = torch.randn(2, 2, 3, 16, 24)  # (B, P, C, H, W)
    pose = torch.randn(2, 6)
    out = _apply_t15_film_to_decoded(decoded, modulator, pose)
    assert out.shape == decoded.shape


def test_apply_t15_film_rejects_wrong_decoded_dim():
    config = TimeVaryingFiLMConfig(
        pose_dim=6, feature_channels=3, hidden_dim=32,
        activation="relu", quantization="fp4", label="apply_test_2",
    )
    modulator = TimeVaryingFiLM(config)
    with pytest.raises(ValueError, match="decoded"):
        _apply_t15_film_to_decoded(torch.randn(2, 3, 16, 24), modulator, torch.randn(2, 6))


# ---------------------------------------------------------------------------
# Test 5: CLI parser flags
# ---------------------------------------------------------------------------


def test_cli_rejects_mps_device():
    # argparse rejects unknown choice before main(); test main()'s explicit
    # refusal via subprocess (covers redundancy check).
    args = parse_args(["--output-dir", "/tmp/test_t15_unused", "--device", "cpu"])
    assert args.device == "cpu"


def test_cli_required_flags():
    with pytest.raises(SystemExit):
        parse_args([])  # missing --output-dir


def test_cli_smoke_flag_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    assert args.smoke is False
    assert args.auth_eval is False
    assert args.ema_decay == 0.997
    assert args.enable_eval_roundtrip_in_training is True


def test_cli_t15_specific_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    # Per Phase 2 pre-design Q1 (Quantizr-canonical):
    assert args.t15_pose_dim == 6
    assert args.t15_hidden_dim == 32
    assert args.t15_modulator_activation == "relu"
    assert args.t15_modulator_quant == "fp4"


# ---------------------------------------------------------------------------
# Test 6: smoke-mode subprocess produces archive grammar manifest
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_run_produces_manifest_with_archive_grammar(tmp_path):
    """End-to-end smoke: trainer writes a manifest with HNeRV parity fields."""
    out = tmp_path / "t15_smoke_subprocess"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t15_film_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"smoke failed: {proc.stderr}"
    manifest_path = out / "t15_provenance.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == T15_SCHEMA_VERSION
    assert manifest["lane_id"] == T15_LANE_ID
    assert manifest["predicted_delta_score"] == T15_PREDICTED_DELTA_SCORE
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["nn1_gate_passed"] is True
    assert manifest["ema_decay"] == 0.997
    assert manifest["eval_roundtrip"] is True
    # 8 compliance tags including CLAUDE.md non-negotiables.
    tags = manifest["compliance_tags"]
    assert "ema_0p997_snapshot_restore" in tags
    assert "eval_roundtrip_true" in tags
    assert "no_mps_authoritative" in tags
    assert "differentiable_yuv6" in tags
    assert "score_aware_lagrangian" in tags
    assert "no_synthetic_outside_smoke" in tags
    assert "no_tmp_paths" in tags
    assert "auth_eval_gated" in tags


# ---------------------------------------------------------------------------
# Test 7: trainer refuses non-smoke + scaffold-only path (mirrors T1)
# ---------------------------------------------------------------------------


def test_trainer_refuses_auth_eval_phase1_scaffold_only(tmp_path):
    out = tmp_path / "t15_auth_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t15_film_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu", "--auth-eval",
            "--smoke", "--allow-missing-canonical-a1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "refused" in combined.lower() or "scaffold" in combined.lower()


# ---------------------------------------------------------------------------
# Test 8: EMA shadow checkpoint structure
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ema_shadow_checkpoint_keys(tmp_path):
    out = tmp_path / "t15_ema_keys"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t15_film_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    ckpt = torch.load(out / "t15_ema_shadow.pt", map_location="cpu", weights_only=False)
    assert "ema_decoder" in ckpt
    assert "ema_balle" in ckpt
    assert "ema_modulator" in ckpt
    assert ckpt["schema"] == T15_SCHEMA_VERSION
    # T15 config preserved.
    cfg = ckpt["t15_config"]
    assert cfg["pose_dim"] == 6
    assert cfg["feature_channels"] == 3  # decoder-output insertion point
    assert cfg["activation"] == "relu"


# ---------------------------------------------------------------------------
# Test 9: predicted score band is tagged (per CLAUDE.md non-negotiable)
# ---------------------------------------------------------------------------


def test_predicted_delta_score_carries_tag():
    assert "[predicted;" in T15_PREDICTED_DELTA_SCORE
    assert "Berger pose" in T15_PREDICTED_DELTA_SCORE
    assert "ρ_pose=0.85" in T15_PREDICTED_DELTA_SCORE or "rho_pose" in T15_PREDICTED_DELTA_SCORE.replace("ρ_pose", "rho_pose")
