# SPDX-License-Identifier: MIT
"""Tests for Phase A Pareto summary solver/planner annotations."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "phase_a_pareto_summary.py"
    spec = importlib.util.spec_from_file_location("phase_a_pareto_summary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(mod, *, lane: str, archive_bytes: int | None):
    delta = (
        archive_bytes - mod.PR101_BROTLI_BYTES
        if archive_bytes is not None and lane != "A4_alt_filler_stc_pose"
        else None
    )
    return mod.PhaseAEntry(
        lane=lane,
        manifest_path=mod.REPO_ROOT / "experiments" / "results" / lane / "build_manifest.json",
        archive_bytes=archive_bytes,
        rel_err=None,
        rel_err_form="unknown",
        evidence_grade="[CPU-prep]",
        score_claim=False,
        ready_for_exact_eval_dispatch=False,
        dispatch_blockers=(),
        dispatch_status=None,
        timestamp="2026-05-08T00:00:00Z",
        delta_vs_brotli_bytes=delta,
    )


def test_annotate_planning_targets_marks_byte_comparable_gap() -> None:
    mod = _load_tool_module()
    entry = _entry(mod, lane="ADMM_lossy_coarsening_baseline", archive_bytes=147_285)
    budget = mod.build_subtarget_budget()

    mod.annotate_planning_targets([entry], budget=budget)

    assert budget.max_archive_bytes == 137_103
    assert entry.byte_budget_comparable is True
    assert entry.subtarget_gap_bytes == 147_285 - 137_103
    assert entry.meets_subtarget_byte_budget is False


def test_annotate_planning_targets_keeps_pose_only_anchor_noncomparable() -> None:
    mod = _load_tool_module()
    entry = _entry(mod, lane="A4_alt_filler_stc_pose", archive_bytes=3_960)

    mod.annotate_planning_targets([entry])

    assert entry.byte_budget_comparable is False
    assert entry.subtarget_gap_bytes is None
    assert entry.meets_subtarget_byte_budget is None


def test_render_markdown_exposes_solver_budget_and_axis_separation() -> None:
    mod = _load_tool_module()
    entry = _entry(mod, lane="ADMM_lossy_coarsening_baseline", archive_bytes=147_285)

    md = mod.render_markdown([entry])

    assert "## Solver planning targets" in md
    assert "Max archive bytes: **137,103 B**" in md
    assert "[prediction; closed-form target byte budget]" in md
    assert "`target_axis=cuda_internal` priority `pose`" in md
    assert "`target_axis=cpu_leaderboard` priority `seg`" in md
    assert "| ADMM_lossy_coarsening_baseline | 147,285 | -30,859 | +10,182 |" in md


def test_render_markdown_references_existing_a6_ledger() -> None:
    mod = _load_tool_module()

    md = mod.render_markdown([])

    ledger = (
        mod.REPO_ROOT
        / ".omx/research/pr101_a6_selfcomp_blockfp_hyperprior_measured_negative_20260508_codex.md"
    )
    assert str(ledger.relative_to(mod.REPO_ROOT)) in md
    assert ledger.is_file()
    assert "PR106 substrate ranked first" in md


def test_render_markdown_keeps_a1_dual_axis_and_deltaepszeta_scaffold() -> None:
    mod = _load_tool_module()

    md = mod.render_markdown([])

    assert "0.192847577437" in md
    assert "0.226352023478" in md
    assert "paired CPU/CUDA exact anchors" in md
    assert "Delta-epsilon-zeta Phase 1" in md
    assert "78,580 B" in md
    assert "no runtime-consumed archive, no score claim, and no dispatch" in md


def test_render_json_carries_machine_readable_planning_targets() -> None:
    mod = _load_tool_module()
    entry = _entry(mod, lane="ADMM_lossy_coarsening_baseline", archive_bytes=147_285)

    payload = json.loads(mod.render_json([entry]))

    budget = payload["planning_targets"]["subtarget_byte_budget"]
    assert budget["max_archive_bytes"] == 137_103
    assert budget["evidence_grade"] == "[prediction; closed-form target byte budget]"
    deltaepszeta = payload["planning_targets"]["delta_epsilon_zeta_phase1"]
    assert deltaepszeta["score_claim"] is False
    assert deltaepszeta["dispatch_attempted"] is False
    assert deltaepszeta["runtime_consumed_archive"] is False
    assert deltaepszeta["target_total_h0_h2_gap_bytes"] == 78_580
    assert deltaepszeta["targets_sha256"] == (
        "bf9b68cd7cb3c0067128c458cd43adffcbf8492b98c657223fb701fc94c116a4"
    )
    axes = payload["planning_targets"]["axis_advisors"]
    assert axes["cuda_internal"]["priority_axis"] == "pose"
    assert axes["cpu_leaderboard"]["priority_axis"] == "seg"
    lane = payload["lanes"][0]
    assert lane["byte_budget_comparable"] is True
    assert lane["subtarget_gap_bytes"] == 10_182
    assert lane["meets_subtarget_byte_budget"] is False
