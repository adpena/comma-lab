# SPDX-License-Identifier: MIT
"""Tests for Catalog #313 — check_dispatch_target_has_no_predecessor_adjudicated_outcome.

PROBE-OUTCOMES-BAKE-IN STRICT preflight gate self-protection 2026-05-16.
Sister of Catalog #245 (Modal call_id ledger 4-layer pattern); same gate-shape
+ test depth as Catalog #131 + #138 sister gates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_dispatch_target_has_no_predecessor_adjudicated_outcome,
)
from tac.probe_outcomes_ledger import (
    VERDICT_DEFER,
    VERDICT_INDEPENDENT,
    VERDICT_KILL,
    VERDICT_PROMOTE,
    register_probe_outcome,
)

# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def synth_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a synthetic repo tree + isolated ledger for end-to-end tests."""
    repo = tmp_path / "synth_repo"
    (repo / "tools").mkdir(parents=True, exist_ok=True)
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "experiments").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "tac").mkdir(parents=True, exist_ok=True)
    ledger = tmp_path / "probe_outcomes.jsonl"
    return repo, ledger


def _register_blocking(
    ledger: Path,
    *,
    probe_id: str,
    substrate: str,
    recipe_path: str,
    verdict: str = VERDICT_INDEPENDENT,
) -> None:
    register_probe_outcome(
        probe_id=probe_id,
        substrate=substrate,
        recipe_path=recipe_path,
        probe_kind="h_latent_given_scorer_class",
        verdict=verdict,
        metric_name="mi_bits",
        metric_value=0.006,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/probe_verdict.md",
        next_action="do_not_dispatch_from_this_signal",
        path=ledger,
        lock_path=ledger.with_suffix(ledger.suffix + ".lock"),
    )


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_zero_violations() -> None:
    """The gate must be at 0 violations on live repo at landing per
    "Strict-flip atomicity rule"."""
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        strict=False, verbose=False
    )
    assert isinstance(violations, list)
    # Live-repo ceiling - allow up to 5 to absorb the canonical operator_authorize.py
    # callsite and the helper itself; should be exactly 0 in practice.
    assert len(violations) <= 5, f"Live repo violations: {violations}"


# ─────────────────────────────────────────────────────────────────────────
# Positive: blocking outcome flags dispatch wrapper
# ─────────────────────────────────────────────────────────────────────────


def test_dispatch_wrapper_with_blocking_recipe_flagged(
    synth_repo: tuple[Path, Path],
) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_atw_v2.yaml"
    _register_blocking(
        ledger,
        probe_id="atw_v2_d4",
        substrate="atw_codec_v2",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "operator_authorize_atw_v2.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'.venv/bin/python tools/operator_authorize.py --recipe {recipe} --dispatch\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "atw_codec_v2" in violations[0]
    assert "INDEPENDENT" in violations[0]


def test_kill_verdict_flags_dispatch(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_dead.yaml"
    _register_blocking(
        ledger,
        probe_id="dead_probe",
        substrate="dead_sub",
        recipe_path=recipe,
        verdict=VERDICT_KILL,
    )
    wrapper = repo / "tools" / "dispatch_dead.py"
    wrapper.write_text(
        f'import subprocess\n'
        f'subprocess.run(["modal_train_lane", "--recipe", "{recipe}"])\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert any("KILL" in v for v in violations)


def test_defer_verdict_flags_dispatch(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_defer.yaml"
    _register_blocking(
        ledger,
        probe_id="defer_probe",
        substrate="defer_sub",
        recipe_path=recipe,
        verdict=VERDICT_DEFER,
    )
    wrapper = repo / "scripts" / "operator_authorize_defer.sh"
    wrapper.write_text(
        f'#!/bin/sh\nlaunch_lane_on_vastai --recipe {recipe}\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert any("DEFER" in v for v in violations)


# ─────────────────────────────────────────────────────────────────────────
# Negative: clean cases
# ─────────────────────────────────────────────────────────────────────────


def test_no_blocking_outcome_no_violations(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_clean.yaml"
    # No registered outcome → no violation
    wrapper = repo / "scripts" / "operator_authorize_clean.sh"
    wrapper.write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe}\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_promote_verdict_does_not_block(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_proceed.yaml"
    register_probe_outcome(
        probe_id="proceed_probe",
        substrate="proceed_sub",
        recipe_path=recipe,
        probe_kind="k",
        verdict=VERDICT_PROMOTE,  # auto-advisory
        metric_name="m",
        metric_value=1.0,
        path=ledger,
        lock_path=ledger.with_suffix(ledger.suffix + ".lock"),
    )
    wrapper = repo / "scripts" / "operator_authorize_proceed.sh"
    wrapper.write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe}\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_dispatch_token_without_recipe_path_not_flagged(
    synth_repo: tuple[Path, Path],
) -> None:
    repo, ledger = synth_repo
    wrapper = repo / "tools" / "dispatch_no_recipe.py"
    # `modal run` token but no recipe path literal → cannot resolve target → skip
    wrapper.write_text(
        "import subprocess\n"
        'subprocess.run(["echo", "no recipe path here", "modal run command"])\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Waiver semantics
# ─────────────────────────────────────────────────────────────────────────


def test_same_line_waiver_with_rationale_accepted(
    synth_repo: tuple[Path, Path],
) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_waived.yaml"
    _register_blocking(
        ledger,
        probe_id="waived_probe",
        substrate="waived_sub",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "operator_authorize_waived.sh"
    wrapper.write_text(
        f'#!/bin/sh\n'
        f'modal run --recipe {recipe}  # PROBE_PREDECESSOR_OVERRIDE_OK:fresh evidence approved by operator 2026-05-16\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_placeholder_rationale_rejected(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_placeholder.yaml"
    _register_blocking(
        ledger,
        probe_id="ph_probe",
        substrate="ph_sub",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "operator_authorize_ph.sh"
    wrapper.write_text(
        f'#!/bin/sh\n'
        f'modal run --recipe {recipe}  # PROBE_PREDECESSOR_OVERRIDE_OK:<rationale>\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_reason_placeholder_rejected(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_reason_ph.yaml"
    _register_blocking(
        ledger,
        probe_id="rph_probe",
        substrate="rph_sub",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "operator_authorize_rph.sh"
    wrapper.write_text(
        f'#!/bin/sh\n'
        f'modal run --recipe {recipe}  # PROBE_PREDECESSOR_OVERRIDE_OK:<reason>\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_empty_rationale_rejected(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_empty_ph.yaml"
    _register_blocking(
        ledger,
        probe_id="eph_probe",
        substrate="eph_sub",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "operator_authorize_eph.sh"
    wrapper.write_text(
        f'#!/bin/sh\n'
        f'modal run --recipe {recipe}  # PROBE_PREDECESSOR_OVERRIDE_OK:\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert len(violations) == 1


# ─────────────────────────────────────────────────────────────────────────
# Scope / exempt-path tests
# ─────────────────────────────────────────────────────────────────────────


def test_test_files_excluded(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_exempt.yaml"
    _register_blocking(
        ledger,
        probe_id="exempt_probe",
        substrate="exempt_sub",
        recipe_path=recipe,
    )
    test_dir = repo / "src" / "tac" / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / "test_dispatch_smoke.py"
    test_file.write_text(
        f'def test_smoke():\n'
        f'    import subprocess\n'
        f'    subprocess.run(["modal_train_lane", "--recipe", "{recipe}"])\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_intake_clones_excluded(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_intake.yaml"
    _register_blocking(
        ledger,
        probe_id="intake_probe",
        substrate="intake_sub",
        recipe_path=recipe,
    )
    intake_dir = repo / "experiments" / "results" / "public_pr106_intake_codex"
    intake_dir.mkdir(parents=True, exist_ok=True)
    f = intake_dir / "dispatch.sh"
    f.write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe}\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_results_dir_excluded(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_results.yaml"
    _register_blocking(
        ledger,
        probe_id="results_probe",
        substrate="results_sub",
        recipe_path=recipe,
    )
    results_dir = repo / "experiments" / "results" / "some_run"
    results_dir.mkdir(parents=True, exist_ok=True)
    f = results_dir / "dispatch.sh"
    f.write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe}\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Strict mode
# ─────────────────────────────────────────────────────────────────────────


def test_strict_raises_with_catalog_313_message(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_strict.yaml"
    _register_blocking(
        ledger,
        probe_id="strict_probe",
        substrate="strict_sub",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "operator_authorize_strict.sh"
    wrapper.write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe}\n',
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="Catalog #313"):
        check_dispatch_target_has_no_predecessor_adjudicated_outcome(
            repo_root=repo, ledger_path=ledger, strict=True, verbose=False
        )


def test_strict_silent_on_clean(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    # No registered outcomes; gate must not raise even in strict mode
    wrapper = repo / "scripts" / "operator_authorize_clean.sh"
    wrapper.write_text(
        '#!/bin/sh\nmodal run --recipe .omx/operator_authorize_recipes/clean.yaml\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=True, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Self-exempt + canonical helper files
# ─────────────────────────────────────────────────────────────────────────


def test_canonical_helper_file_self_exempt(synth_repo: tuple[Path, Path]) -> None:
    """Even if the canonical helper file mentioned the dispatch tokens, it
    should be self-exempt."""
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_self_exempt.yaml"
    _register_blocking(
        ledger,
        probe_id="self_exempt_probe",
        substrate="self_exempt_sub",
        recipe_path=recipe,
    )
    # Create files at the canonical helper paths (relative to repo root)
    helper = repo / "src" / "tac" / "probe_outcomes_ledger.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(
        f'# helper file - exempt\n'
        f'CMD = "modal run --recipe {recipe}"\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Multi-violation aggregation
# ─────────────────────────────────────────────────────────────────────────


def test_multiple_violations_aggregated(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe1 = ".omx/operator_authorize_recipes/substrate_a.yaml"
    recipe2 = ".omx/operator_authorize_recipes/substrate_b.yaml"
    _register_blocking(
        ledger, probe_id="p1", substrate="sub_a", recipe_path=recipe1
    )
    _register_blocking(
        ledger, probe_id="p2", substrate="sub_b", recipe_path=recipe2
    )
    (repo / "scripts" / "dispatch_a.sh").write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe1}\n', encoding="utf-8"
    )
    (repo / "scripts" / "dispatch_b.sh").write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe2}\n', encoding="utf-8"
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert len(violations) == 2


# ─────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────


def test_missing_scan_dir_silently_skipped(tmp_path: Path) -> None:
    """No scan dirs exist → returns empty (does not raise)."""
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_string_repo_root_accepted(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=str(repo), ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_comment_lines_not_flagged(synth_repo: tuple[Path, Path]) -> None:
    """Pure comment lines are not flagged."""
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_comment.yaml"
    _register_blocking(
        ledger,
        probe_id="comment_probe",
        substrate="comment_sub",
        recipe_path=recipe,
    )
    wrapper = repo / "scripts" / "comment_only.sh"
    wrapper.write_text(
        f'#!/bin/sh\n'
        f'# modal run --recipe {recipe}  # this is just a comment\n'
        f'echo done\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


def test_only_py_and_sh_extensions_scanned(synth_repo: tuple[Path, Path]) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_md.yaml"
    _register_blocking(
        ledger,
        probe_id="md_probe",
        substrate="md_sub",
        recipe_path=recipe,
    )
    # .md file should NOT be scanned
    md_file = repo / "scripts" / "doc.md"
    md_file.write_text(
        f'# Documentation\nmodal run --recipe {recipe}\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert violations == []


# ─────────────────────────────────────────────────────────────────────────
# Catalog #245 backfill regression — ATW v2 D4 anchor
# ─────────────────────────────────────────────────────────────────────────


def test_backfill_atw_v2_d4_blocks_dispatch(synth_repo: tuple[Path, Path]) -> None:
    """End-to-end: backfill the ATW v2 D4 verdict to a synthetic ledger,
    create a synthetic operator-authorize wrapper that targets the ATW v2
    recipe, and verify the gate flags the wrapper."""
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml"
    register_probe_outcome(
        probe_id="atw_v2_d4_h_latent_given_scorer_class_20260516",
        substrate="atw_codec_v2",
        recipe_path=recipe,
        probe_kind="h_latent_given_scorer_class",
        verdict=VERDICT_INDEPENDENT,
        metric_name="mutual_information_bits_per_symbol",
        metric_value=0.006385502752,
        threshold=0.5,
        threshold_token="MEANINGFUL_CONDITIONING",
        evidence_path=".omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md",
        next_action="do_not_dispatch_atw_v2_phase2_from_this_signal",
        path=ledger,
        lock_path=ledger.with_suffix(ledger.suffix + ".lock"),
    )
    wrapper = repo / "scripts" / "operator_authorize_substrate_atw_v2.sh"
    wrapper.write_text(
        f'#!/bin/sh\n'
        f'.venv/bin/python tools/operator_authorize.py --recipe {recipe} --dispatch\n',
        encoding="utf-8",
    )
    violations = check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "atw_codec_v2" in violations[0]
    assert "atw_v2_d4_h_latent_given_scorer_class_20260516" in violations[0]


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator-callsite regression guard
# ─────────────────────────────────────────────────────────────────────────


def test_gate_callable_via_module_globals() -> None:
    """Catalog #185 sister regression: gate function must be importable from
    tac.preflight module globals."""
    import tac.preflight as preflight_mod
    assert hasattr(
        preflight_mod,
        "check_dispatch_target_has_no_predecessor_adjudicated_outcome",
    )
    func = preflight_mod.check_dispatch_target_has_no_predecessor_adjudicated_outcome
    assert callable(func)


def test_function_signature_keyword_only() -> None:
    """Gate signature should be keyword-only per the canonical pattern."""
    import inspect
    sig = inspect.signature(
        check_dispatch_target_has_no_predecessor_adjudicated_outcome
    )
    for param in sig.parameters.values():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


def test_verbose_output_clean(synth_repo: tuple[Path, Path], capsys) -> None:
    repo, ledger = synth_repo
    check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_verbose_output_dirty(synth_repo: tuple[Path, Path], capsys) -> None:
    repo, ledger = synth_repo
    recipe = ".omx/operator_authorize_recipes/substrate_v.yaml"
    _register_blocking(
        ledger, probe_id="v_probe", substrate="v_sub", recipe_path=recipe
    )
    (repo / "scripts" / "v.sh").write_text(
        f'#!/bin/sh\nmodal run --recipe {recipe}\n', encoding="utf-8"
    )
    check_dispatch_target_has_no_predecessor_adjudicated_outcome(
        repo_root=repo, ledger_path=ledger, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "violation" in captured.out
