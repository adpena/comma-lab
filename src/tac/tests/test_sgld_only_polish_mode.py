# SPDX-License-Identifier: MIT
"""Tests for the E.8 SGLD-only polish convergence-diagnostic entrypoint.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + the
operator's OP-2 directive (lane
``lane_b1_e8_sgld_trainer_scope_fix_op2_20260519``): the SGLD paradigm is
intact; only the prior single-arm A1 passthrough implementation that the
empirical E.8 failures (fc-01KRZCHVY6C1TSFNNS6KN13G70 +
fc-01KRZCSQ7FPVMSAXZQDSZJCTN4 2026-05-19) ran was falsified. This test file
proves the new ``--sgld-only-polish-mode`` flag routes through a real
Welling-Teh SGLD polish loop with canonical Provenance.

Cross-references:
- Trainer: ``experiments/train_substrate_stack_of_stacks.py``
- Driver: ``scripts/remote_lane_substrate_stack_of_stacks.sh``
- Recipe: ``.omx/operator_authorize_recipes/
  substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml``
- Canonical SGLD optimizer: ``src/tac/optimization/langevin_optimizer.py``
  (Welling & Teh 2011 SGLD; Catalog #344 canonical equation registry pending)
- CC predecessor: ``.omx/research/b1_e7_e8_modal_dispatch_harvest_landed_20260519.md``
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRAINER = REPO / "experiments/train_substrate_stack_of_stacks.py"
DRIVER = REPO / "scripts/remote_lane_substrate_stack_of_stacks.sh"
RECIPE = (
    REPO
    / ".omx/operator_authorize_recipes"
    / "substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml"
)


# ---------------------------------------------------------------------------
# Argparse + helper-API tests (no GPU required; verify flag is wired)
# ---------------------------------------------------------------------------


def test_sgld_only_polish_mode_flag_is_recognized() -> None:
    """The --sgld-only-polish-mode flag MUST be present in the argparse surface."""
    sys.path.insert(0, str(REPO / "src"))
    from experiments.train_substrate_stack_of_stacks import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            "/tmp/_test_sgld_arg_only",
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--sgld-only-polish-mode",
        ]
    )
    assert args.sgld_only_polish_mode is True
    # sister diagnostic flags
    assert args.sgld_polish_quantization_bits == 8
    assert abs(args.sgld_polish_log_every - 0.1) < 1e-9


def test_sgld_only_polish_mode_off_by_default() -> None:
    """The flag must default to False; passthrough behavior preserved."""
    sys.path.insert(0, str(REPO / "src"))
    from experiments.train_substrate_stack_of_stacks import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        [
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            "/tmp/_test_no_sgld",
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert args.sgld_only_polish_mode is False


def test_sgld_only_polish_mode_helper_classifier() -> None:
    """The _is_sgld_only_polish_mode_build helper must match the flag state."""
    sys.path.insert(0, str(REPO / "src"))
    from experiments.train_substrate_stack_of_stacks import (
        _build_parser,
        _is_sgld_only_polish_mode_build,
    )

    parser = _build_parser()
    args_on = parser.parse_args(
        [
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            "/tmp/_t_on",
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--sgld-only-polish-mode",
        ]
    )
    args_off = parser.parse_args(
        [
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            "/tmp/_t_off",
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
        ]
    )
    assert _is_sgld_only_polish_mode_build(args_on) is True
    assert _is_sgld_only_polish_mode_build(args_off) is False


# ---------------------------------------------------------------------------
# End-to-end trainer subprocess tests (real composition + SGLD polish loop)
# ---------------------------------------------------------------------------


def test_sgld_polish_runs_end_to_end_and_emits_plateau_log(tmp_path: Path) -> None:
    """SGLD-only polish mode end-to-end: archive built + plateau log written."""
    output_dir = tmp_path / "sgld_e2e"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--sgld-only-polish-mode",
            "--langevin-polish-epochs",
            "20",
            "--langevin-t-init",
            "1.0",
            "--max-pairs",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "stage=sgld_only_polish_begin" in result.stderr
    assert "stage=sgld_only_polish_done" in result.stderr
    # Plateau log MUST exist with SGLD-specific schema.
    plateau_path = output_dir / "sgld_polish_log.json"
    assert plateau_path.is_file()
    plateau = json.loads(plateau_path.read_text(encoding="utf-8"))
    assert plateau["schema"] == "sgld_polish_log_v1_e8_convergence_diagnostic"
    assert plateau["t_init"] == 1.0
    assert plateau["polish_epochs"] == 20
    assert plateau["literature_anchor"].startswith("Welling & Teh")
    # NON-PROMOTABLE per Catalog #324
    assert plateau["score_claim"] is False
    assert plateau["promotion_eligible"] is False
    assert plateau["evidence_grade"] == "predicted"
    assert plateau["axis_tag"] == "[predicted]"
    # Plateau log entries MUST contain canonical fields (step, loss, temperature)
    assert len(plateau["plateau_log"]) >= 2
    first = plateau["plateau_log"][0]
    assert {"step", "loss", "temperature", "noise_scale", "wall_clock_seconds"}.issubset(
        first.keys()
    )
    # Temperature must monotone-decrease (Welling-Teh cosine schedule canonical)
    temps = [e["temperature"] for e in plateau["plateau_log"]]
    assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))


def test_sgld_polish_emits_provenance_with_canonical_markers(tmp_path: Path) -> None:
    """SGLD polish provenance MUST carry canonical Provenance markers per Catalog #323."""
    output_dir = tmp_path / "sgld_prov"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--sgld-only-polish-mode",
            "--langevin-polish-epochs",
            "5",
            "--max-pairs",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    prov = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    # Catalog #323 canonical Provenance discipline.
    assert prov["sgld_only_polish_mode"] is True
    assert prov["score_claim"] is False
    assert prov["promotion_eligible"] is False
    assert prov["axis_tag"] == "[predicted]"
    assert prov["evidence_grade"] == "predicted"
    assert prov["runtime_contract"] == (
        "sgld_only_polish_convergence_diagnostic_research_only"
    )
    # Catalog #324 predicted_band_validation_status pending_post_training.
    assert prov["predicted_band_validation_status"] == "pending_post_training"
    # NON-PROMOTABLE: ready_for_exact_eval MUST be False even though the archive
    # was successfully built.
    assert prov["ready_for_exact_eval_dispatch"] is False
    assert prov["canary_exact_eval_ready"] is False


def test_sgld_polish_summary_marks_research_only_with_blockers(tmp_path: Path) -> None:
    """SGLD polish summary MUST mark research_only=True + explicit blockers."""
    output_dir = tmp_path / "sgld_summary"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--sgld-only-polish-mode",
            "--langevin-polish-epochs",
            "5",
            "--max-pairs",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    summary = json.loads(
        (output_dir / "stack_of_stacks_compose_summary.json").read_text(encoding="utf-8")
    )
    assert summary["sgld_only_polish_mode"] is True
    assert summary["research_only"] is True
    assert summary["score_claim"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert summary["axis_tag"] == "[predicted]"
    assert summary["evidence_grade"] == "predicted"
    # Explicit blocker per Catalog #324 + #287
    assert any(
        "parameter_drift_proxy_not_contest_score_aware" in b
        for b in summary["dispatch_blockers"]
    )


def test_sgld_polish_archive_is_valid_zip_with_x_member(tmp_path: Path) -> None:
    """SGLD-polished archive MUST still be a valid single-member ZIP per HNeRV parity L4."""
    import zipfile

    output_dir = tmp_path / "sgld_archive"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--sgld-only-polish-mode",
            "--langevin-polish-epochs",
            "5",
            "--max-pairs",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    archive = output_dir / "submission_dir/archive.zip"
    assert archive.is_file()
    with zipfile.ZipFile(archive, "r") as zf:
        assert zf.namelist() == ["x"]
        # Per HNeRV parity discipline + Catalog #146: archive must be non-empty.
        member_bytes = zf.read("x")
        assert len(member_bytes) > 0
    # Submission dir must include canonical inflate.sh + inflate.py + base_runtime.
    assert (output_dir / "submission_dir/inflate.sh").is_file()
    assert (output_dir / "submission_dir/inflate.py").is_file()
    assert (output_dir / "submission_dir/base_runtime/inflate.sh").is_file()


# ---------------------------------------------------------------------------
# Passthrough regression: default mode must NOT trigger SGLD
# ---------------------------------------------------------------------------


def test_passthrough_mode_does_not_emit_sgld_log(tmp_path: Path) -> None:
    """Default (non-SGLD) mode preserves prior passthrough behavior."""
    output_dir = tmp_path / "passthrough_regression"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--max-pairs",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # No SGLD-specific stages in passthrough mode
    assert "stage=sgld_only_polish_begin" not in result.stderr
    assert "stage=sgld_only_polish_done" not in result.stderr
    assert not (output_dir / "sgld_polish_log.json").exists()

    summary = json.loads(
        (output_dir / "stack_of_stacks_compose_summary.json").read_text(encoding="utf-8")
    )
    assert summary["sgld_only_polish_mode"] is False
    # Passthrough is the canonical canary path and remains ready_for_exact_eval
    assert summary["ready_for_exact_eval_dispatch"] is True
    assert summary["research_only"] is False
    assert summary["runtime_contract"] == "single_arm_a1_passthrough_exact_eval_canary"


# ---------------------------------------------------------------------------
# Recipe + driver wiring tests (Catalog #326 substrate driver mode hardcode)
# ---------------------------------------------------------------------------


def test_recipe_declares_sgld_only_polish_mode_opt_in() -> None:
    """E.8 recipe MUST set STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE='1' per Catalog #326."""
    text = RECIPE.read_text(encoding="utf-8")
    assert 'STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE: "1"' in text
    # And recipe MUST still cite the Welling-Teh anchor in its literature_anchor
    assert "Welling & Teh" in text


def test_driver_threads_sgld_only_polish_mode_env_var_to_flag() -> None:
    """Driver MUST honor STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE='1' -> --sgld-only-polish-mode."""
    driver_text = DRIVER.read_text(encoding="utf-8")
    # Default off (Catalog #326: explicit recipe-side opt-in required)
    assert (
        'STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE="${STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE:-0}"'
        in driver_text
    )
    # When env is "1", trainer receives --sgld-only-polish-mode flag
    assert "--sgld-only-polish-mode" in driver_text
    assert 'if [ "$STACK_OF_STACKS_SGLD_ONLY_POLISH_MODE" = "1" ]; then' in driver_text


# ---------------------------------------------------------------------------
# Welling-Teh canonical math invariants (LangevinOptimizer integration)
# ---------------------------------------------------------------------------


def test_sgld_polish_helper_routes_through_langevin_optimizer() -> None:
    """The _run_sgld_only_polish helper MUST use the canonical LangevinOptimizer."""
    sys.path.insert(0, str(REPO / "src"))
    import inspect

    from experiments import train_substrate_stack_of_stacks as mod

    source = inspect.getsource(mod._run_sgld_only_polish)
    # Canonical Welling-Teh integration: LangevinOptimizer + EMA + temperature schedule
    assert "LangevinOptimizer(" in source
    assert "EMA(" in source
    assert "T_init=args.langevin_t_init" in source
    assert "T_final=args.langevin_t_final" in source
    assert "schedule=args.langevin_schedule" in source
    # NON-PROMOTABLE evidence per Catalog #324
    assert "DIAGNOSTIC" in source.upper() or "diagnostic" in source.lower()


def test_sgld_polish_helper_preserves_archive_byte_length() -> None:
    """SGLD polish MUST preserve archive grammar (byte-length invariant)."""
    sys.path.insert(0, str(REPO / "src"))
    import argparse as _argparse

    import torch

    from experiments.train_substrate_stack_of_stacks import _run_sgld_only_polish

    args = _argparse.Namespace(
        langevin_polish_epochs=10,
        langevin_t_init=1.0,
        langevin_t_final=1e-4,
        langevin_schedule="cosine",
        ema_decay=0.997,
        lr=1e-4,
        weight_decay=1e-5,
        seed=42,
        sgld_polish_log_every=0.1,
        sgld_polish_quantization_bits=8,
    )
    # Deterministic int8 composed bytes
    rng = torch.Generator()
    rng.manual_seed(42)
    composed = bytes((torch.randint(-128, 127, (1024,), generator=rng).to(torch.int8)).numpy().tobytes())
    polished, plateau = _run_sgld_only_polish(
        composed_bytes=composed,
        args=args,
        device=torch.device("cpu"),
    )
    # Byte-length invariant (Catalog #146 archive grammar)
    assert len(polished) == len(composed)
    # Plateau log produced
    assert len(plateau) >= 2
    # Welling-Teh canonical: temperature monotone-decreases (cosine schedule)
    temps = [e["temperature"] for e in plateau]
    assert all(temps[i] >= temps[i + 1] for i in range(len(temps) - 1))
    # Per-step loss is finite (no NaN/Inf escape)
    losses = [e["loss"] for e in plateau]
    assert all(l == l for l in losses)  # no NaN
    assert all(abs(l) < 1e6 for l in losses)  # bounded
