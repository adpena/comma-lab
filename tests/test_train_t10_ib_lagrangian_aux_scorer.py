"""Tests for ``experiments/train_t10_ib_lagrangian_aux_scorer.py``.

NN-T10 (3-clean-pass review 2026-05-11): the trainer MUST produce a
``distillation_gap_estimate.json`` artifact that Phase 3 reads to gate
``Phase3DispatchGate`` (Catalog #134).  Tests confirm:

- CLI flag set is consistent (no MPS, auth-eval refused, smoke gated)
- Smoke-mode dataloader produces shape-correct batches
- Smoke-mode contest-scorer mimic returns shape-correct outputs
- ``AuxiliaryScorerConfig`` is built with council-canonical defaults
- Smoke-run produces ``distillation_gap_estimate.json`` with the right keys
- EMA shadow checkpoint structure
- Phase 3 prerequisite tag set on manifest
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

from experiments.train_t10_ib_lagrangian_aux_scorer import (  # noqa: E402
    PHASE3_DISTILL_GAP_THRESHOLD,
    T10_LANE_ID,
    T10_PREDICTED_DELTA_SCORE,
    T10_SCHEMA_VERSION,
    _SmokeDataloader,
    _make_smoke_contest_scorer,
    parse_args,
)
from tac.ib_lagrangian_aux_scorer import (  # noqa: E402
    AuxiliaryScorerConfig,
    AuxiliaryScorerError,
)


# ---------------------------------------------------------------------------
# _SmokeDataloader
# ---------------------------------------------------------------------------


def test_smoke_dataloader_yields_correct_shapes():
    dl = _SmokeDataloader(
        n_batches=2, batch_size=4, seg_class_count=5, pose_dim=6, seed=0,
        height=16, width=24,
    )
    batches = list(dl)
    assert len(batches) == 2
    for frames, gt_seg, gt_pose in batches:
        assert frames.shape == (4, 2, 3, 16, 24)
        assert gt_seg.shape == (4, 16, 24)
        assert gt_seg.dtype == torch.long
        assert gt_pose.shape == (4, 6)


def test_smoke_dataloader_seg_labels_in_range():
    dl = _SmokeDataloader(
        n_batches=1, batch_size=3, seg_class_count=5, pose_dim=6, seed=42,
        height=8, width=12,
    )
    _, gt_seg, _ = next(iter(dl))
    assert int(gt_seg.min().item()) >= 0
    assert int(gt_seg.max().item()) < 5


def test_smoke_dataloader_rejects_bad_args():
    with pytest.raises(ValueError, match="n_batches"):
        _SmokeDataloader(
            n_batches=0, batch_size=1, seg_class_count=2, pose_dim=1, seed=0,
        )
    with pytest.raises(ValueError, match="batch_size"):
        _SmokeDataloader(
            n_batches=1, batch_size=0, seg_class_count=2, pose_dim=1, seed=0,
        )


# ---------------------------------------------------------------------------
# Smoke contest scorer mimic
# ---------------------------------------------------------------------------


def test_smoke_contest_scorer_returns_correct_shapes():
    fwd = _make_smoke_contest_scorer(seg_class_count=5, pose_dim=6)
    frames = torch.rand(2, 2, 3, 16, 24) * 255.0
    seg, pose = fwd(frames)
    assert seg.shape == (2, 5, 16, 24)
    assert pose.shape == (2, 6)


def test_smoke_contest_scorer_rejects_wrong_dim():
    fwd = _make_smoke_contest_scorer(seg_class_count=5, pose_dim=6)
    with pytest.raises(ValueError, match="expected"):
        fwd(torch.rand(2, 3, 16, 24))  # missing T


# ---------------------------------------------------------------------------
# AuxiliaryScorerConfig — council canonical values
# ---------------------------------------------------------------------------


def test_aux_config_council_canonical_smoke_passes():
    cfg = AuxiliaryScorerConfig.council_canonical(
        distill_label="t10_test", smoke_mode=True, cuda_required=False,
    )
    assert cfg.distill_temperature == 2.0
    assert cfg.lambda_gt == 0.5
    assert cfg.ema_decay == 0.997
    assert cfg.seg_class_count == 5
    assert cfg.pose_dim == 6
    assert cfg.smoke_mode is True
    assert cfg.cuda_required is False


def test_aux_config_rejects_low_ema_decay():
    with pytest.raises(AuxiliaryScorerError, match="ema_decay"):
        AuxiliaryScorerConfig(
            distill_temperature=2.0, lambda_gt=0.5, ema_decay=0.5,
            seg_class_count=5, pose_dim=6, cuda_required=False,
            smoke_mode=True, distill_label="bad",
        )


def test_aux_config_rejects_bad_lambda_gt():
    with pytest.raises(AuxiliaryScorerError, match="lambda_gt"):
        AuxiliaryScorerConfig(
            distill_temperature=2.0, lambda_gt=10.0, ema_decay=0.997,
            seg_class_count=5, pose_dim=6, cuda_required=False,
            smoke_mode=True, distill_label="bad2",
        )


# ---------------------------------------------------------------------------
# CLI parser flags
# ---------------------------------------------------------------------------


def test_cli_required_flags():
    with pytest.raises(SystemExit):
        parse_args([])  # missing --output-dir


def test_cli_t10_canonical_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    assert args.distill_temperature == 2.0
    assert args.lambda_gt == 0.5
    assert args.ema_decay == 0.997
    assert args.seg_class_count == 5
    assert args.pose_dim == 6
    assert args.smoke is False
    assert args.auth_eval is False


def test_cli_rejects_mps():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/dev/null", "--device", "mps"])


# ---------------------------------------------------------------------------
# Constants tagged per CLAUDE.md
# ---------------------------------------------------------------------------


def test_predicted_delta_score_is_na_with_phase3_pointer():
    # Per pre-design memo: T10 has NO direct score Δ; it's a Phase 3
    # prerequisite.  The string must explicitly say so.
    assert "Phase 3 prerequisite" in T10_PREDICTED_DELTA_SCORE
    assert "Catalog #134" in T10_PREDICTED_DELTA_SCORE


def test_phase3_threshold_is_hinton_canonical():
    assert PHASE3_DISTILL_GAP_THRESHOLD == 0.03


# ---------------------------------------------------------------------------
# Subprocess smoke (end-to-end on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_run_produces_distillation_gap_estimate_json(tmp_path):
    out = tmp_path / "t10_smoke_subprocess"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t10_ib_lagrangian_aux_scorer.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke",
            "--epochs", "1",
            "--n-batches", "2",
            "--batch-size", "2",
            "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"smoke failed: {proc.stderr}"
    gap_path = out / "distillation_gap_estimate.json"
    assert gap_path.is_file()
    gap = json.loads(gap_path.read_text())
    assert gap["schema"] == T10_SCHEMA_VERSION
    assert gap["lane_id"] == T10_LANE_ID
    assert gap["phase3_threshold"] == PHASE3_DISTILL_GAP_THRESHOLD
    # smoke gap may exceed threshold — that's expected; it's a build check
    assert isinstance(gap["distillation_gap_estimate"], float)
    assert isinstance(gap["passes_phase3_threshold"], bool)
    assert gap["smoke_mode"] is True
    assert "[predicted;" in gap["evidence_grade"]


@pytest.mark.slow
def test_smoke_run_produces_provenance_json(tmp_path):
    out = tmp_path / "t10_smoke_prov"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t10_ib_lagrangian_aux_scorer.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke",
            "--epochs", "1",
            "--n-batches", "2",
            "--batch-size", "2",
            "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    manifest_path = out / "t10_provenance.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == T10_SCHEMA_VERSION
    assert manifest["lane_id"] == T10_LANE_ID
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["research_only"] is True
    assert manifest["phase3_prereq_artifact"] == "distillation_gap_estimate.json"
    assert manifest["ema_decay"] == 0.997
    assert manifest["distill_temperature"] == 2.0
    # Compliance tags include the canon set.
    tags = manifest["compliance_tags"]
    for required in (
        "ema_0p997",
        "hinton_distill_T_2p0",
        "lambda_gt_0p5",
        "no_mps_authoritative",
        "no_synthetic_outside_smoke",
        "no_tmp_paths",
        "auth_eval_refused_T10_has_no_archive",
        "phase3_prereq_distillation_gap_artifact",
    ):
        assert required in tags, f"missing compliance tag {required}"


@pytest.mark.slow
def test_trainer_refuses_auth_eval(tmp_path):
    out = tmp_path / "t10_auth_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t10_ib_lagrangian_aux_scorer.py"),
            "--output-dir", str(out),
            "--device", "cpu", "--auth-eval", "--smoke",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "refused" in combined.lower() or "Phase3DispatchGate" in combined


@pytest.mark.slow
def test_non_smoke_path_is_operator_gated(tmp_path):
    """Non-smoke path must refuse with a structured operator-gated message."""
    out = tmp_path / "t10_nonsmoke_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t10_ib_lagrangian_aux_scorer.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--epochs", "1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    # Either it refuses because cuda not available OR because non-smoke
    # path is operator-gated — both are correct.
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert (
        "operator-gated" in combined.lower()
        or "cuda" in combined.lower()
    ), combined


@pytest.mark.slow
def test_ema_shadow_checkpoint_keys(tmp_path):
    out = tmp_path / "t10_ema_keys"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_t10_ib_lagrangian_aux_scorer.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke",
            "--epochs", "1",
            "--n-batches", "2",
            "--batch-size", "2",
            "--seed", "42",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    ckpt = torch.load(out / "t10_aux_scorer_ema_shadow.pt", map_location="cpu", weights_only=False)
    assert "ema_state_dict" in ckpt
    assert ckpt["schema"] == T10_SCHEMA_VERSION
    cfg = ckpt["t10_config"]
    assert cfg["distill_temperature"] == 2.0
    assert cfg["lambda_gt"] == 0.5
    assert cfg["ema_decay"] == 0.997
    assert cfg["seg_class_count"] == 5
    assert cfg["pose_dim"] == 6
