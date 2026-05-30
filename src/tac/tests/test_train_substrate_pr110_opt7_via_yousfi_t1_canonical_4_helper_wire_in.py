# SPDX-License-Identifier: MIT
"""Behavioral tests for PR110-OPT-7 via Yousfi-T1 trainer 4-helper canonical wire-in.

Per CLAUDE.md NO FAKE IMPLEMENTATIONS Class 2 (tests-verify-constants-not-
behavior): these tests verify the trainer ACTUALLY invokes each of the 4
canonical helpers per the predecessor DEFER reactivation criteria, NOT just
that the trainer imports them.

Closes the predecessor DEFER per
``feedback_pr110_opt7_l1_paired_cuda_ratification_DEFER_pending_trainer_wire_in_landed_20260530``.

The 4 canonical wire-in blockers structurally verified by these tests:

1. Tier 1 ``score_pair_components`` (Catalog #270 + Catalog #164) — canonical
   scorer-loss helper invoked from ``_full_main``.
2. Tier 3 ``gate_auth_eval_call`` (Catalog #226) — canonical auth-eval helper
   invoked from ``_full_main`` per canonical kwarg signature.
3. Tier 3 ``select_inflate_device`` (Catalog #205) — canonical inflate device
   helper invoked from ``_full_main`` per ``PACT_INFLATE_DEVICE`` env-var
   routing.
4. Tier 3 scorer-loader canonical assignment order (Catalog #222) —
   ``pose_scorer, seg_scorer = load_differentiable_scorers(...)`` (NOT
   reversed) in ``_full_main``.

Test categories:

- **Module-level canonical helper imports**: verify the 4 canonical helpers
  are imported at module top so the canonical_dispatch_optimization_protocol
  token check passes.
- **Function definitions**: verify ``main`` / ``_smoke_main`` / ``_full_main``
  exist as the canonical dispatch structure.
- **Trainer-mode dispatcher**: verify CLI ``--trainer-mode`` + env
  ``PR110_OPT7_TRAINER_MODE`` precedence per Catalog #326 driver mode
  hardcode discipline.
- **Phase C smoke path preservation**: verify ``_smoke_main`` still works
  end-to-end with 5/5 helpers invoked + substantive=PASS + archive emitted.
- **AST-level canonical helper invocation evidence**: verify each of the 4
  canonical helpers is INVOKED (not just imported) inside ``_full_main``
  body via AST scan — closes the NO FAKE IMPLEMENTATIONS Class 1 surface.
- **Canonical scorer-loader assignment ORDER**: verify the LHS tuple of
  ``load_differentiable_scorers`` assignment is ``(pose_scorer, seg_scorer)``
  NOT reversed per Catalog #222.
- **Catalog #243 + #270 pre-flight harness PASS**: integration test that
  invokes the canonical helpers + verifies overall_pass=true.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER_PATH = (
    REPO_ROOT
    / "experiments"
    / "train_substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1.py"
)


def _trainer_text() -> str:
    return TRAINER_PATH.read_text(encoding="utf-8")


def _trainer_ast() -> ast.Module:
    return ast.parse(_trainer_text(), filename=str(TRAINER_PATH))


def _function_by_name(name: str) -> ast.FunctionDef:
    tree = _trainer_ast()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in trainer AST")


def _calls_in_function(fn: ast.FunctionDef) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            calls.append(node)
    return calls


def _call_callee_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


# ---------------------------------------------------------------------------
# Category 1: module-level canonical helper imports
# ---------------------------------------------------------------------------

def test_canonical_helper_imports_present_at_module_top() -> None:
    """All 4 canonical helpers imported at module top.

    Required for canonical_dispatch_optimization_protocol token presence
    check + sister Catalog #243 auth_eval_reachability check.
    """
    text = _trainer_text()
    assert "from tac.substrates._shared.inflate_runtime import" in text
    assert "select_inflate_device" in text
    assert "from tac.substrates._shared.smoke_auth_eval_gate import" in text
    assert "gate_auth_eval_call" in text
    assert "from tac.substrates.score_aware_common import" in text
    assert "score_pair_components" in text


def test_canonical_helper_imports_alias_canonical_pattern() -> None:
    """Canonical aliases match sister-trainer canonical pattern."""
    text = _trainer_text()
    # Sister trainers (a1_plus_lapose) use _canon_ prefix
    assert "_canon_select_inflate_device" in text
    assert "_canon_gate_auth_eval_call" in text
    assert "_canon_score_pair_components" in text


# ---------------------------------------------------------------------------
# Category 2: function definitions
# ---------------------------------------------------------------------------

def test_main_function_exists() -> None:
    fn = _function_by_name("main")
    assert isinstance(fn, ast.FunctionDef)


def test_smoke_main_function_exists() -> None:
    fn = _function_by_name("_smoke_main")
    assert isinstance(fn, ast.FunctionDef)


def test_full_main_function_exists() -> None:
    fn = _function_by_name("_full_main")
    assert isinstance(fn, ast.FunctionDef)


# ---------------------------------------------------------------------------
# Category 3: trainer-mode dispatcher (Catalog #326)
# ---------------------------------------------------------------------------

def test_main_resolves_trainer_mode_from_env_var() -> None:
    """Catalog #326 driver mode hardcode discipline: env var routing."""
    text = _trainer_text()
    assert "PR110_OPT7_TRAINER_MODE" in text
    assert "SMOKE_ONLY" in text
    assert "--trainer-mode" in text


def test_main_dispatches_to_smoke_or_full_based_on_mode() -> None:
    """Verify main() dispatches to _smoke_main or _full_main."""
    main_fn = _function_by_name("main")
    callees = [_call_callee_name(c) for c in _calls_in_function(main_fn)]
    assert "_smoke_main" in callees, "main() must call _smoke_main"
    assert "_full_main" in callees, "main() must call _full_main"


# ---------------------------------------------------------------------------
# Category 4: AST-level canonical helper invocation evidence in _full_main
# (closes CLAUDE.md NO FAKE IMPLEMENTATIONS Class 1)
# ---------------------------------------------------------------------------

def test_full_main_invokes_select_inflate_device() -> None:
    """Canonical select_inflate_device IS invoked in _full_main (not just imported)."""
    full_fn = _function_by_name("_full_main")
    callees = [_call_callee_name(c) for c in _calls_in_function(full_fn)]
    assert "_canon_select_inflate_device" in callees, (
        "_full_main must INVOKE canonical select_inflate_device (Catalog #205)"
    )


def test_full_main_invokes_load_differentiable_scorers() -> None:
    """Canonical load_differentiable_scorers IS invoked in _full_main."""
    full_fn = _function_by_name("_full_main")
    callees = [_call_callee_name(c) for c in _calls_in_function(full_fn)]
    assert "load_differentiable_scorers" in callees, (
        "_full_main must INVOKE canonical load_differentiable_scorers (Catalog #222)"
    )


def test_full_main_invokes_score_pair_components() -> None:
    """Canonical score_pair_components IS invoked in _full_main."""
    full_fn = _function_by_name("_full_main")
    callees = [_call_callee_name(c) for c in _calls_in_function(full_fn)]
    assert "_canon_score_pair_components" in callees, (
        "_full_main must INVOKE canonical score_pair_components (Catalog #164 + #270)"
    )


def test_full_main_invokes_gate_auth_eval_call() -> None:
    """Canonical gate_auth_eval_call IS invoked in _full_main."""
    full_fn = _function_by_name("_full_main")
    callees = [_call_callee_name(c) for c in _calls_in_function(full_fn)]
    assert "_canon_gate_auth_eval_call" in callees, (
        "_full_main must INVOKE canonical gate_auth_eval_call (Catalog #226)"
    )


# ---------------------------------------------------------------------------
# Category 5: scorer-loader canonical assignment ORDER (Catalog #222)
# ---------------------------------------------------------------------------

def test_full_main_scorer_loader_assignment_canonical_order() -> None:
    """pose_scorer, seg_scorer = load_differentiable_scorers (NOT reversed).

    Per Catalog #222 canonical contract: load_differentiable_scorers returns
    (posenet, segnet). The LHS tuple MUST be (pose_scorer, seg_scorer) in
    this canonical order. Reversed assignment (seg_scorer, pose_scorer =
    load_differentiable_scorers(...)) crashes in scorer_loss_terms_btchw.
    """
    text = _trainer_text()
    # Canonical pattern present
    assert "pose_scorer, seg_scorer = load_differentiable_scorers" in text, (
        "Trainer must use canonical scorer-loader assignment order per Catalog #222"
    )
    # Reversed pattern MUST NOT be present
    assert "seg_scorer, pose_scorer = load_differentiable_scorers" not in text, (
        "Trainer must NOT use REVERSED scorer-loader assignment per Catalog #222"
    )


def test_full_main_gate_auth_eval_call_uses_canonical_kwargs() -> None:
    """gate_auth_eval_call uses canonical kwargs (Catalog #365 signature drift discipline).

    Per Catalog #365: canonical kwargs are archive_zip / output_json /
    contest_auth_eval_script / substrate_tag / args (NOT deprecated
    archive=, json_out=, lane_id=, substrate_id=).
    """
    text = _trainer_text()
    # Canonical kwargs MUST be present near gate_auth_eval_call invocation
    assert "archive_zip=" in text
    assert "output_json=" in text
    assert "contest_auth_eval_script=" in text
    assert "substrate_tag=" in text
    # Deprecated kwargs MUST NOT be present
    assert "archive=archive_zip" not in text
    assert "json_out=" not in text


# ---------------------------------------------------------------------------
# Category 6: Phase C MLX-LOCAL smoke path preserved
# ---------------------------------------------------------------------------

def test_smoke_main_path_end_to_end(tmp_path: Path) -> None:
    """Phase C MLX-LOCAL N=24 smoke runs end-to-end with 5/5 helpers + substantive=PASS.

    Sister of the predecessor commit `1230b3b9c` MLX-LOCAL Phase C 7/7 GREEN
    validation. Verifies the wire-in does NOT regress the canonical smoke
    path that produced the L1 PROMOTION empirical anchor.
    """
    env = {**os.environ, "SMOKE_ONLY": "1"}
    # Force SMOKE_ONLY in case PR110_OPT7_TRAINER_MODE was set in the test env.
    env.pop("PR110_OPT7_TRAINER_MODE", None)
    out_dir = tmp_path / "smoke"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER_PATH),
            "--output-dir",
            str(out_dir),
            "--n-pairs",
            "24",
            "--vulnerable-pair-budget",
            "4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"smoke trainer failed rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    # Verify canonical artifacts emitted
    stats_path = out_dir / "training_stats.json"
    assert stats_path.is_file(), f"training_stats.json missing at {stats_path}"
    stats = json.loads(stats_path.read_text())
    assert stats["trainer_mode"] == "smoke"
    assert stats["slot_eee_verification"]["substantive_distinctness_verdict"] == "PASS"
    assert stats["slot_eee_verification"]["invocation_count"] == 5
    assert stats["evidence_grade"] == "[macOS-MLX research-signal]"
    assert stats["score_claim"] is False
    assert stats["promotion_eligible"] is False
    # Verify archive emitted
    archive_path = out_dir / "submission" / "submission_dir" / "0.bin"
    assert archive_path.is_file()
    assert archive_path.stat().st_size > 0


def test_trainer_mode_env_var_routes_to_full(tmp_path: Path) -> None:
    """PR110_OPT7_TRAINER_MODE=full env var dispatches to _full_main.

    On macOS (no CUDA) the _full_main path fail-closes rc=4 per canonical
    contract per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
    non-negotiable. This test verifies the DISPATCHER routes correctly,
    not the full path's CUDA-required behavior.
    """
    env = {**os.environ, "PR110_OPT7_TRAINER_MODE": "full"}
    env.pop("SMOKE_ONLY", None)
    out_dir = tmp_path / "full_dispatch_test"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER_PATH),
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    # On macOS (no CUDA): rc=4 with FATAL message about CUDA requirement
    # On Linux+CUDA: rc would be 0 (but we still test the dispatcher route)
    import torch
    if not torch.cuda.is_available():
        assert result.returncode == 4, (
            f"full mode on non-CUDA host should rc=4; got {result.returncode}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "full mode requires CUDA" in result.stderr
    # Verify dispatcher resolved to full mode (visible in stdout)
    assert "resolved trainer_mode=full" in result.stdout


def test_trainer_mode_cli_flag_overrides_env(tmp_path: Path) -> None:
    """--trainer-mode CLI flag overrides PR110_OPT7_TRAINER_MODE env var."""
    env = {**os.environ, "PR110_OPT7_TRAINER_MODE": "full"}
    env.pop("SMOKE_ONLY", None)
    out_dir = tmp_path / "cli_override_test"
    # CLI flag = smoke; env var = full; CLI should win
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER_PATH),
            "--output-dir",
            str(out_dir),
            "--trainer-mode",
            "smoke",
            "--n-pairs",
            "24",
            "--vulnerable-pair-budget",
            "4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"smoke via CLI override failed rc={result.returncode}\n{result.stderr}"
    )
    assert "resolved trainer_mode=smoke" in result.stdout


# ---------------------------------------------------------------------------
# Category 7: Catalog #243 + #270 pre-flight harness PASS (integration)
# ---------------------------------------------------------------------------

def test_canonical_dispatch_optimization_protocol_passes() -> None:
    """Integration test: canonical dispatch optimization protocol PASSES.

    Per the predecessor DEFER memo, the canonical protocol must report
    overall_pass=true with zero Tier 1/2/3 blockers after the wire-in.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "canonical_dispatch_optimization_protocol.py"),
            "--trainer",
            str(TRAINER_PATH),
            "--recipe",
            "substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_modal_t4_dispatch",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode in (0, 1), (
        f"protocol tool failed rc={result.returncode}\nstderr={result.stderr}"
    )
    verdict = json.loads(result.stdout)
    assert verdict["overall_pass"] is True, (
        f"canonical dispatch protocol must PASS; blockers={verdict['blockers']}"
    )
    # All 5 Tier 1 signals
    assert all(verdict["tier1"]["pass_signals"].values()), (
        f"Tier 1 must all PASS; got {verdict['tier1']['pass_signals']}"
    )
    # All Tier 2 recipe + driver signals
    assert all(verdict["tier2"]["pass_signals"].values()), (
        f"Tier 2 must all PASS; got {verdict['tier2']['pass_signals']}"
    )
    # All 5 Tier 3 signals
    assert all(verdict["tier3"]["pass_signals"].values()), (
        f"Tier 3 must all PASS; got {verdict['tier3']['pass_signals']}"
    )


def test_canonical_dispatch_optimization_protocol_4_specific_signals_pass() -> None:
    """The 4 specific canonical wire-in signals named in the DEFER report PASS.

    Per ``feedback_pr110_opt7_l1_paired_cuda_ratification_DEFER_pending_trainer_wire_in_landed_20260530``:
    - tier1: canonical_scorer_loss (was FAIL)
    - tier3: canonical_auth_eval_helper (was FAIL)
    - tier3: canonical_inflate_device (was FAIL)
    - tier3: scorer_loader_order_correct (was FAIL)
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "canonical_dispatch_optimization_protocol.py"),
            "--trainer",
            str(TRAINER_PATH),
            "--recipe",
            "substrate_pr110_opt7_fridrich_uniward_inverse_scorer_basis_via_yousfi_t1_modal_t4_dispatch",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    verdict = json.loads(result.stdout)
    # The 4 wire-in blockers from the DEFER
    assert verdict["tier1"]["pass_signals"]["canonical_scorer_loss"] is True
    assert verdict["tier3"]["pass_signals"]["canonical_auth_eval_helper"] is True
    assert verdict["tier3"]["pass_signals"]["canonical_inflate_device"] is True
    assert verdict["tier3"]["pass_signals"]["scorer_loader_order_correct"] is True


# ---------------------------------------------------------------------------
# Category 8: regression — wire-in does NOT introduce reversed scorer loader
# ---------------------------------------------------------------------------

def test_canonical_dispatch_protocol_no_reversed_scorer_loader() -> None:
    """No reversed scorer-loader assignment per Catalog #222 STRICT pattern.

    The canonical_dispatch_optimization_protocol scans for the reversed
    pattern and refuses any source containing it.
    """
    text = _trainer_text()
    import re
    reversed_patterns = (
        r"seg_scorer\s*,\s*pose_scorer\s*=\s*load_differentiable_scorers",
        r"seg_scorer\s*,\s*pose_scorer\s*=\s*load_default_scorers",
        r"segnet\s*,\s*posenet\s*=\s*load_default_scorers",
        r"segnet\s*,\s*posenet\s*=\s*load_differentiable_scorers",
    )
    for pat in reversed_patterns:
        assert not re.search(pat, text), (
            f"Trainer contains REVERSED scorer-loader pattern: {pat!r} per Catalog #222"
        )
