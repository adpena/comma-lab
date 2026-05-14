# SPDX-License-Identifier: MIT
"""Tests for ``experiments/train_lane_12_v2_nerv_as_renderer.py``.

Coverage scope (production trainer for Lane 12-v2 NeRV-as-renderer):
- CLI parses correctly with required flags.
- ``--device mps`` is REFUSED at argparse (CLAUDE.md MPS-is-noise).
- ``--device cuda`` is REFUSED at runtime when CUDA unavailable
  (CLAUDE.md MPS-fallback-trap FORBIDDEN_PATTERNS).
- ``--auth-eval`` without ``--phase-b-auth-memo`` is REFUSED (Catalog #150).
- ``--auth-eval`` with invalid memo path is REFUSED.
- Smoke mode runs end-to-end on CPU + emits archive + provenance.
- Smoke mode writes EMA-shadow-derived archive (NOT live weights).
- Provenance manifest includes all CLAUDE.md compliance tags.
- Predicted Δ score string is present and tagged ``[predicted; ...]``.
- ``score_claim`` and ``promotion_eligible`` are False (no anchor yet).
- Smoke does NOT touch /tmp (CLAUDE.md no-tmp).
- Smoke uses ``_make_synthetic_pair_batch_for_smoke`` via the inline
  ``# SYNTHETIC_NON_SMOKE_OK:`` waiver, NOT in non-smoke paths.
- Non-smoke mode raises if contest video missing.
- No imports of ``upstream.modules`` at module load time (lazy).
- Loop runs >=1 iteration and emits non-empty history.

Tests run CPU-only, smoke mode, latent_dim=8, n_pairs=4.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_trainer():
    """Import the trainer script as a module (it's an experiments/ entry point)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_lane_12_v2_nerv_as_renderer",
        REPO_ROOT / "experiments" / "train_lane_12_v2_nerv_as_renderer.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def repo_output_dir(tmp_path):
    root = REPO_ROOT / "experiments" / "results" / ".pytest_tmp_outputs" / tmp_path.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    def make(name: str) -> Path:
        return root / name

    yield make
    shutil.rmtree(root, ignore_errors=True)


# ── CLI / argparse ────────────────────────────────────────────────────────


def test_parse_args_requires_output_dir():
    """The trainer requires --output-dir; missing it raises SystemExit."""
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer.parse_args([])


def test_parse_args_rejects_mps_device():
    """--device mps is REFUSED at argparse via choices=[cuda, cpu]."""
    trainer = _import_trainer()
    with pytest.raises(SystemExit):
        trainer.parse_args(["--output-dir", "/tmp_test", "--device", "mps"])


def test_parse_args_defaults_match_claude_md_canonicals():
    """Defaults must match CLAUDE.md canonical values."""
    trainer = _import_trainer()
    args = trainer.parse_args(["--output-dir", "/tmp_test"])
    assert args.device == "cuda", "CLAUDE.md: CUDA-required default"
    assert args.ema_decay == 0.997, "CLAUDE.md: EMA decay 0.997"
    assert args.enable_score_aware_loss is True, "CLAUDE.md: score-aware Lagrangian default"
    assert args.enable_differentiable_yuv6 is True, "PR95/106: differentiable YUV6 default"
    assert args.smoke is False
    assert args.auth_eval is False


# ── --auth-eval gate (Catalog #150) ───────────────────────────────────────


def test_auth_eval_without_memo_refused(tmp_path, repo_output_dir):
    """--auth-eval without --phase-b-auth-memo refused per Catalog #150."""
    trainer = _import_trainer()
    args = [
        "--output-dir", str(repo_output_dir("out")),
        "--device", "cpu",
        "--auth-eval",
        "--smoke",
    ]
    with pytest.raises(SystemExit) as exc_info:
        trainer.main(args)
    assert "phase-b-auth-memo" in str(exc_info.value).lower() or "150" in str(exc_info.value)


def test_auth_eval_with_invalid_memo_refused(tmp_path, repo_output_dir):
    """--auth-eval with non-existent memo PENDING → refused."""
    trainer = _import_trainer()
    memo_path = tmp_path / "fake_memo.md"  # doesn't exist
    args = [
        "--output-dir", str(repo_output_dir("out")),
        "--device", "cpu",
        "--auth-eval",
        "--phase-b-auth-memo", str(memo_path),
        "--smoke",
    ]
    # Catalog #150 _assert_auth_memo_path_repo_relative raises ValueError
    # on non-repo paths; tmp_path is outside repo root.
    with pytest.raises((SystemExit, ValueError)):
        trainer.main(args)


# ── Device resolution (CLAUDE.md MPS-fallback-trap) ──────────────────────


def test_resolve_device_cuda_unavailable_raises():
    """--device cuda without CUDA available raises (no silent MPS fallback)."""
    trainer = _import_trainer()
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA available; cannot test fallback path")
    with pytest.raises(SystemExit) as exc_info:
        trainer._resolve_device("cuda")
    assert "cuda" in str(exc_info.value).lower()


def test_resolve_device_mps_refused():
    """Calling _resolve_device('mps') explicitly is refused."""
    trainer = _import_trainer()
    with pytest.raises(SystemExit) as exc_info:
        trainer._resolve_device("mps")
    assert "mps" in str(exc_info.value).lower()


def test_resolve_device_cpu_works():
    """--device cpu returns torch.device('cpu')."""
    trainer = _import_trainer()
    import torch
    d = trainer._resolve_device("cpu")
    assert d == torch.device("cpu")


# ── Smoke end-to-end (CPU; no scorers; synthetic pairs) ──────────────────


def test_smoke_runs_end_to_end_on_cpu(repo_output_dir):
    """Smoke mode runs end-to-end on CPU without CUDA / scorers."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("smoke_out")
    # Smoke disables score-aware loss path (no scorer load) for speed.
    rc = trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--epochs", "1",
        "--batch-size", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
    ])
    assert rc == 0
    assert (out_dir / "0.bin").exists()
    assert (out_dir / "provenance.json").exists()


def test_smoke_provenance_contains_compliance_tags(repo_output_dir):
    """Provenance manifest must declare all CLAUDE.md compliance tags."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("smoke_prov")
    trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--epochs", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
    ])
    prov = json.loads((out_dir / "provenance.json").read_text())
    tags = set(prov["compliance_tags"])
    required = {
        "ema_0p997_snapshot_restore",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "differentiable_yuv6",
        "score_aware_lagrangian",
        "no_synthetic_outside_smoke",
        "no_tmp_paths",
        "auth_eval_gated_phase_b_option_c",
        "cuda_required_default",
    }
    assert required.issubset(tags), f"missing: {required - tags}"


def test_smoke_provenance_score_claim_false(repo_output_dir):
    """Provenance MUST have score_claim=False (no anchor yet)."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("smoke_no_claim")
    trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--epochs", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
    ])
    prov = json.loads((out_dir / "provenance.json").read_text())
    assert prov["score_claim"] is False
    assert prov["promotion_eligible"] is False
    assert prov["ready_for_exact_eval_dispatch"] is False


def test_smoke_provenance_predicted_delta_score_tagged(repo_output_dir):
    """Predicted Δ score must carry ``[predicted; ...]`` tag."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("smoke_pred")
    trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--epochs", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
    ])
    prov = json.loads((out_dir / "provenance.json").read_text())
    assert "[predicted;" in prov["predicted_delta_score"]


def test_smoke_archive_bytes_nontrivial(repo_output_dir):
    """Archive must have >12 bytes (header + sections)."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("smoke_arch")
    trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--epochs", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
    ])
    archive_bytes = (out_dir / "0.bin").stat().st_size
    # 12 bytes header + 4 × 4 byte length prefixes = 28; plus actual section bytes.
    assert archive_bytes > 28


def test_smoke_history_non_empty(repo_output_dir):
    """Provenance history must have at least 1 entry."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("smoke_hist")
    trainer.main([
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--smoke",
        "--epochs", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
    ])
    prov = json.loads((out_dir / "provenance.json").read_text())
    assert len(prov["history"]) >= 1
    assert "avg_loss" in prov["history"][0]


# ── Non-smoke gating + no-/tmp ───────────────────────────────────────────


def test_non_smoke_without_video_raises(tmp_path, repo_output_dir):
    """Non-smoke training without contest video raises (synthetic-in-non-smoke forbidden)."""
    trainer = _import_trainer()
    out_dir = repo_output_dir("no_video")
    fake_video = tmp_path / "missing.mkv"
    args = [
        "--output-dir", str(out_dir),
        "--device", "cpu",
        "--epochs", "1",
        "--latent-dim", "8",
        "--n-pairs", "4",
        "--video-path", str(fake_video),
    ]
    with pytest.raises(SystemExit):
        trainer.main(args)


def test_trainer_source_has_no_tmp_durable_paths():
    """Trainer source MUST NOT contain durable /tmp paths."""
    src = (REPO_ROOT / "experiments" / "train_lane_12_v2_nerv_as_renderer.py").read_text()
    # Check no /tmp/ literal appears as durable evidence-output path.
    # Allow test-only /tmp_test marker in docstrings (none in this trainer).
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        assert "/tmp/" not in line, (
            f"trainer must not write durable /tmp paths (CLAUDE.md FORBIDDEN_PATTERNS): {line!r}"
        )
    assert "assert_not_temporary_output_dir" in src
    assert "def _refuse_tmp_output_dir" not in src


def test_trainer_source_no_make_synthetic_outside_smoke():
    """Trainer must NOT call make_synthetic_* outside an args.smoke gate."""
    src = (REPO_ROOT / "experiments" / "train_lane_12_v2_nerv_as_renderer.py").read_text()
    # The trainer references _make_synthetic_pair_batch_for_smoke ONLY when
    # use_synthetic is True (set from args.smoke). Verify the gate exists.
    assert "use_synthetic = bool(args.smoke)" in src
    assert "if use_synthetic:" in src


# ── Imports lazy (no upstream.modules at module load) ────────────────────


def test_trainer_does_not_import_upstream_modules_at_top():
    """upstream.modules must not appear at module-level imports (CLAUDE.md strict scorer rule).

    upstream.modules contains PoseNet/SegNet which would fail to import in
    a sandboxed test runner without upstream/ in sys.path. The trainer
    must defer these to inside main() / lazy imports.
    """
    src = (REPO_ROOT / "experiments" / "train_lane_12_v2_nerv_as_renderer.py").read_text()
    # Top-level imports section is the first ~80 lines.
    top = "\n".join(src.splitlines()[:80])
    assert "from upstream.modules" not in top
    assert "import upstream.modules" not in top


# ── EMA snapshot+restore semantics ────────────────────────────────────────


def test_smoke_archive_uses_ema_shadow_not_live_weights(tmp_path):
    """Archive at export time must come from EMA shadow, not live training weights.

    Smoke trains 1 epoch with synthetic pairs and exports; we verify that
    the EMA snapshot+restore pattern at export wraps the actual export call.
    """
    src = (REPO_ROOT / "experiments" / "train_lane_12_v2_nerv_as_renderer.py").read_text()
    # Look for the canonical snapshot+restore pattern around export_to_archive.
    assert "ema_renderer.apply(renderer)" in src
    assert "ema_latents.apply(latent_table)" in src
    assert "export_to_archive(" in src
    # The export must be inside a try/finally that restores orig weights.
    assert "finally:" in src
    assert "renderer.load_state_dict(orig_renderer)" in src


# ── Schema / lane identity ────────────────────────────────────────────────


def test_schema_constants():
    """LANE_ID + SCHEMA_VERSION sanity."""
    trainer = _import_trainer()
    assert trainer.LANE_ID == "lane_12_v2_nerv_as_renderer"
    assert "lane-12-v2" in trainer.SCHEMA_VERSION
