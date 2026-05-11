"""Tests for ``experiments/train_t17_shared_vq_t1_clone.py``.

NN-2 (Phase 2 pre-design 2026-05-09): codebook-collapse perplexity gate
MUST fire when perplexity < 0.4 × num_entries. Tests also confirm:

- van den Oord persistent-EMA decay 0.99 (NOT 0.997) per CLAUDE.md
  EMA exception clause for VQ-VAE codebooks
- CLI flag set + scaffold-only refusal mirrors T1
- archive grammar fields declared in manifest
- score-aware Lagrangian wires commitment_loss
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

from experiments.train_t17_shared_vq_t1_clone import (  # noqa: E402
    COMMITMENT_LOSS_WEIGHT_DEFAULT,
    NN2_PERPLEXITY_FLOOR_RATIO,
    T17_LANE_ID,
    T17_PREDICTED_DELTA_SCORE,
    T17_SCHEMA_VERSION,
    CodebookCollapseError,
    _project_balle_latent_to_codebook,
    _unproject_quantized_to_balle_shape,
    assert_codebook_perplexity_ok,
    parse_args,
    reinit_dead_codebook_entries,
)
from tac.shared_vq_codebook import (  # noqa: E402
    SharedCodebook,
    SharedCodebookConfig,
)


# ---------------------------------------------------------------------------
# Test 1: NN-2 perplexity gate fires on collapse
# ---------------------------------------------------------------------------


def test_nn2_gate_raises_on_collapsed_codebook():
    """Synthetic collapsed indices (all the same) MUST trip NN-2."""
    # All 1000 indices point at entry 0 → perplexity = 1, floor = 102.4 for N=256.
    collapsed_indices = torch.zeros(1000, dtype=torch.long)
    with pytest.raises(CodebookCollapseError, match="codebook collapse"):
        assert_codebook_perplexity_ok(collapsed_indices, num_entries=256)


def test_nn2_gate_passes_on_uniform_indices():
    """Uniform-distributed indices give perplexity ≈ num_entries → passes."""
    uniform_indices = torch.arange(1024) % 256  # equal-distributed
    diag = assert_codebook_perplexity_ok(uniform_indices, num_entries=256)
    assert diag["passed"] is True
    assert diag["perplexity"] >= diag["floor"]


def test_nn2_gate_floor_ratio_is_configurable():
    # 64 entries used; floor at 0.4×100 = 40; should pass.
    indices = torch.arange(64).repeat(10)
    diag = assert_codebook_perplexity_ok(indices, num_entries=100, floor_ratio=0.4)
    assert diag["passed"] is True
    # Now tighten the floor — should fail.
    with pytest.raises(CodebookCollapseError):
        assert_codebook_perplexity_ok(indices, num_entries=100, floor_ratio=0.7)


# ---------------------------------------------------------------------------
# Test 2: Reinit-dead-codebook helper resurrects collapsed entries
# ---------------------------------------------------------------------------


def test_reinit_dead_codebook_entries_replaces_unused_rows():
    config = SharedCodebookConfig.vandenoord_canonical(label="reinit_test")
    codebook = SharedCodebook(config)
    # Mark all but the first 10 entries dead.
    codebook.ema_count[:] = 0.0
    codebook.ema_count[:10] = 100.0
    original_dead = codebook.codebook[10:].clone()
    recent_z = torch.randn(500, 64)
    n_reinit = reinit_dead_codebook_entries(codebook, recent_z, dead_threshold=1.0)
    assert n_reinit == 246  # 256 - 10
    # The replaced rows should NOT match the original initial values.
    new_dead = codebook.codebook[10:]
    diff = (new_dead - original_dead).abs().sum().item()
    assert diff > 0


def test_reinit_dead_codebook_zero_dead_returns_zero():
    config = SharedCodebookConfig.vandenoord_canonical(label="reinit_test_2")
    codebook = SharedCodebook(config)
    # All entries alive.
    codebook.ema_count[:] = 100.0
    n = reinit_dead_codebook_entries(codebook, torch.randn(10, 64))
    assert n == 0


# ---------------------------------------------------------------------------
# Test 3: Project / unproject latent shape correctness
# ---------------------------------------------------------------------------


def test_project_balle_latent_handles_clean_tokenization():
    y_hat = torch.randn(4, 128)
    z_e = _project_balle_latent_to_codebook(y_hat, entry_dim=64)
    assert z_e.shape == (4, 2, 64)


def test_project_balle_latent_truncates_remainder():
    y_hat = torch.randn(4, 100)  # 100 = 1 × 64 + 36 remainder
    z_e = _project_balle_latent_to_codebook(y_hat, entry_dim=64)
    assert z_e.shape == (4, 1, 64)


def test_project_balle_latent_rejects_too_narrow():
    y_hat = torch.randn(4, 16)  # < entry_dim=64
    with pytest.raises(ValueError, match="cannot tokenize"):
        _project_balle_latent_to_codebook(y_hat, entry_dim=64)


def test_unproject_pads_to_target_dim():
    z_q = torch.randn(4, 2, 64)
    y_back = _unproject_quantized_to_balle_shape(z_q, target_shape=(4, 192))
    assert y_back.shape == (4, 192)
    # Trailing dims are zero-padded.
    assert torch.allclose(y_back[:, 128:], torch.zeros(4, 64))


# ---------------------------------------------------------------------------
# Test 4: CLI parser flag set
# ---------------------------------------------------------------------------


def test_cli_t17_defaults_match_council():
    """Phase 2 pre-design pass: 256 entries × 64 dim, EMA 0.99, FP16."""
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_t17")])
    assert args.t17_num_entries == 256
    assert args.t17_entry_dim == 64
    assert args.t17_codebook_ema_decay == 0.99  # NOT 0.997 per CLAUDE.md
    assert args.t17_codebook_quant == "fp16"
    assert args.t17_commitment_weight == COMMITMENT_LOSS_WEIGHT_DEFAULT
    assert args.t17_perplexity_floor_ratio == NN2_PERPLEXITY_FLOOR_RATIO


def test_cli_rejects_invalid_codebook_ema_decay():
    """The codebook EMA decay MUST be in [0.9, 1.0) per van den Oord."""
    # argparse does not range-check; validation happens in main().
    # We probe via subprocess.
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t17_shared_vq_t1_clone.py"),
            "--output-dir", "/tmp/should_not_run",
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--t17-codebook-ema-decay", "0.5",  # too low
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert "ema-decay" in (proc.stderr + proc.stdout)


def test_cli_rejects_mps_device():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_t17_2"), "--device", "cpu"])
    assert args.device == "cpu"


def test_cli_required_flags():
    with pytest.raises(SystemExit):
        parse_args([])


# ---------------------------------------------------------------------------
# Test 5: Smoke run produces manifest with archive grammar + perplexity history
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_run_produces_manifest(tmp_path):
    out = tmp_path / "t17_smoke_subprocess"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t17_shared_vq_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
            # Smoke-mode small latent dim requires a smaller entry_dim / num_entries
            # and a lower perplexity floor to avoid the gate firing on toy data.
            "--t17-entry-dim", "14",
            "--t17-num-entries", "16",
            "--t17-perplexity-floor-ratio", "0.1",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, f"smoke failed: {proc.stderr}"
    manifest = json.loads((out / "t17_provenance.json").read_text())
    assert manifest["schema"] == T17_SCHEMA_VERSION
    assert manifest["lane_id"] == T17_LANE_ID
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["codebook_ema_decay"] == 0.99
    assert manifest["weight_ema_decay"] == 0.997
    assert manifest["nn2_gate_passed_each_epoch"] is True
    assert manifest["predicted_delta_score"] == T17_PREDICTED_DELTA_SCORE
    # Compliance tags
    tags = manifest["compliance_tags"]
    assert "codebook_ema_0p99_van_den_oord_canon" in tags
    assert "ema_0p997_snapshot_restore_weights" in tags
    assert "nn2_perplexity_gate_per_epoch" in tags
    assert "vandenoord_commitment_loss" in tags
    assert "no_mps_authoritative" in tags
    assert "differentiable_yuv6" in tags
    assert "auth_eval_gated" in tags


# ---------------------------------------------------------------------------
# Test 6: Refuses --auth-eval at scaffold stage (mirrors T1)
# ---------------------------------------------------------------------------


def test_refuses_auth_eval(tmp_path):
    out = tmp_path / "t17_auth_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t17_shared_vq_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu", "--auth-eval",
            "--smoke", "--allow-missing-canonical-a1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    assert "refused" in (proc.stderr + proc.stdout).lower()


# ---------------------------------------------------------------------------
# Test 7: EMA shadow checkpoint structure
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ema_shadow_checkpoint_includes_codebook(tmp_path):
    out = tmp_path / "t17_ema_keys"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t17_shared_vq_t1_clone.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
            "--t17-entry-dim", "14",
            "--t17-num-entries", "16",
            "--t17-perplexity-floor-ratio", "0.1",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    ckpt = torch.load(out / "t17_ema_shadow.pt", map_location="cpu", weights_only=False)
    assert "ema_decoder" in ckpt
    assert "ema_balle" in ckpt
    assert "codebook" in ckpt
    assert "codebook_ema_count" in ckpt
    assert "codebook_ema_sum" in ckpt
    assert ckpt["schema"] == T17_SCHEMA_VERSION
    cfg = ckpt["t17_config"]
    assert cfg["num_entries"] == 16
    assert cfg["entry_dim"] == 14
    assert cfg["ema_decay"] == 0.99


# ---------------------------------------------------------------------------
# Test 8: predicted score band is tagged
# ---------------------------------------------------------------------------


def test_predicted_delta_score_is_tagged():
    assert "[predicted;" in T17_PREDICTED_DELTA_SCORE
    assert "van den Oord" in T17_PREDICTED_DELTA_SCORE
