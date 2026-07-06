"""Tests for the triality drift-detector Stop hook (tools/triality_drift_detector.py).

Covers the pure classify() decision surface (drift / clean / opted-out / non-
substantive / recorded) plus an integration smoke that proves the real hook
exits 0 (fail-open) on the live repo and never wedges a session.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "triality_drift_detector.py"


def _load():
    spec = importlib.util.spec_from_file_location("triality_drift_detector", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()


# ------------------------------- classify() ------------------------------------
def test_drift_substantive_commit_without_triality_touch():
    subjects = ["witness: byte-close n600 exact row d_seg 0.0031 measured"]
    files = ["tools/levelset_byte_close_and_eval.py", "src/tac/boundary_math/foo.py"]
    assert D.classify(subjects, files) == "drift"


def test_clean_when_dag_touched_same_window():
    # The general fallback net: a substantive commit that does NOT require a
    # specific leg (no lever/measure/verdict/island/seed signature — just
    # n600/frontier trajectory) is cleared by a DAG touch.
    subjects = ["witness: n600 frontier trajectory point", "triality DAG: FEED"]
    files = ["tools/foo.py",
             ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"]
    assert D.classify(subjects, files) == "clean"


def test_drift_measured_finding_touched_only_dag():
    # PER-LEG teeth (2026-07-06): a MEASURED byte-close / exact-row commit MUST touch
    # the canonical equations — touching only the DAG is NOT enough anymore.
    subjects = ["witness: byte-close n600 exact row measured d_seg 0.0031"]
    files = [".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"]
    assert D.classify(subjects, files) == "drift"
    assert "equations" in D.missing_legs(subjects, files)


def test_drift_lever_touched_only_dag():
    # PER-LEG teeth: a LEVER / wire-in commit MUST touch the DSL — the exact loophole
    # that let the DSL silently drift while only the DAG was recorded.
    subjects = ["witness: SeedIslandEased lever wired into trainer"]
    files = [".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md",
             "experiments/train_levelset_witness_realized_through_R_mlx.py"]
    assert D.classify(subjects, files) == "drift"
    assert "DSL" in D.missing_legs(subjects, files)


def test_clean_lever_touches_dsl_even_without_dag():
    # A lever commit that DOES touch the DSL is clean (the required leg was updated).
    subjects = ["witness: SeedIslandEased lever wired"]
    files = ["src/tac/witness_dsl/curriculum_dsl.py",
             "experiments/train_levelset_witness_realized_through_R_mlx.py"]
    assert D.classify(subjects, files) == "clean"


def test_clean_when_dsl_touched():
    subjects = ["witness: new lever wired"]
    files = ["src/tac/witness_dsl/curriculum_dsl.py"]
    assert D.classify(subjects, files) == "clean"


def test_clean_when_equations_touched():
    subjects = ["measured: register canonical equation for erasure"]
    files = ["src/tac/canonical_equations.py"]
    assert D.classify(subjects, files) == "clean"


def test_clean_non_substantive_commit():
    subjects = ["chore: fix typo in comment", "docs: reword paragraph"]
    files = ["README.md"]
    assert D.classify(subjects, files) == "clean"


def test_clean_opted_out_even_if_substantive():
    # "kernel" is substantive, but the [no-triality] token forces clean.
    subjects = ["kernel: refactor fused-R helper [no-triality]"]
    files = ["src/tac/local_acceleration/metal_fused_r_operator.py"]
    assert D.is_substantive(subjects)  # would be drift without the opt-out
    assert D.classify(subjects, files) == "clean"


def test_clean_skip_drift_token():
    subjects = ["witness: measured probe [skip-drift]"]
    files = ["tools/probe.py"]
    assert D.classify(subjects, files) == "clean"


def test_clean_no_commits():
    assert D.classify([], []) == "clean"


def test_opted_out_wins_over_substantive_in_mixed_window():
    # one opts out → whole window treated as chore (conservative: never nag)
    subjects = ["witness: byte-close measured", "apparatus tweak [no-triality]"]
    files = ["tools/x.py"]
    assert D.classify(subjects, files) == "clean"


def test_substantive_regex_hits_expected_tokens():
    for kw in ["measured", "byte-close", "d_seg", "d_pose", "pointer", "launch",
               "lever", "verdict", "witness", "n600", "erasure", "islands"]:
        assert D.is_substantive([f"something {kw} something"]), kw


def test_substantive_regex_misses_chore():
    assert not D.is_substantive(["chore: bump version"])
    assert not D.is_substantive(["docs: fix link"])
    assert not D.is_substantive(["reformat imports"])


def test_triality_touch_detects_all_prefixes():
    assert D.has_triality_touch([".omx/research/sub015_DAG_x.md"])
    assert D.has_triality_touch(["src/tac/witness_dsl/gauge.py"])
    assert D.has_triality_touch(["src/tac/canonical_equations.py"])
    assert D.has_triality_touch(["docs/triality_dag_dsl_equations_deepmath.md"])
    assert D.has_triality_touch([".omx/research/CANONICAL_RESEARCH_INDEX_20260629.md"])
    assert not D.has_triality_touch(["src/tac/boundary_math/foo.py", "tools/bar.py"])


def test_build_reason_is_concise_and_actionable():
    r = D.build_reason(["witness: byte-close measured d_seg 0.003"])
    assert "DAG FEED" in r
    assert "[no-triality]" in r
    assert len(r) < 900  # one firm nudge, not an essay


# ------------------------------- integration -----------------------------------
def test_hook_exits_zero_on_real_repo_empty_stdin():
    """Fail-open contract: empty stdin on the live repo → exit 0, no crash."""
    proc = subprocess.run(
        [sys.executable, str(_TOOL)],
        input="", capture_output=True, text=True, timeout=30, cwd=str(_REPO),
    )
    assert proc.returncode == 0


def test_hook_exits_zero_on_garbage_stdin():
    """Malformed input → still fail-open (never wedge)."""
    proc = subprocess.run(
        [sys.executable, str(_TOOL)],
        input="not json {{{", capture_output=True, text=True, timeout=30, cwd=str(_REPO),
    )
    assert proc.returncode == 0


def test_hook_stop_hook_active_short_circuits_to_allow():
    """stop_hook_active=true → allow (loop-safe), no block JSON emitted."""
    proc = subprocess.run(
        [sys.executable, str(_TOOL)],
        input=json.dumps({"stop_hook_active": True, "cwd": str(_REPO)}),
        capture_output=True, text=True, timeout=30, cwd=str(_REPO),
    )
    assert proc.returncode == 0
    # when allowing, no decision:block payload on stdout
    assert '"decision"' not in proc.stdout


# --- calibration + per-commit (adversarial review 2026-07-06) ---------------------
def test_both_legs_required_when_neither_touched():
    subj = ["witness: island-birth lever measured d_seg 0.0031 verdict"]
    files = [".omx/research/sub015_DAG_x.md"]
    miss = D.missing_legs(subj, files)
    assert "DSL" in miss and "equations" in miss
    assert D.classify(subj, files) == "drift"


def test_measured_numeric_row_requires_equations():
    # MEDIUM-1 fix: a numeric d_seg/d_pose row (no measur/verdict stem) still needs equations.
    subj = ["witness: d_seg 0.0031 n600 best at ep50"]
    files = [".omx/research/sub015_DAG_x.md"]
    assert "equations" in D.missing_legs(subj, files)
    assert D.classify(subj, files) == "drift"


def test_added_lever_family_stems_close_the_loophole():
    # MEDIUM-1 fix: the reviewer's slip-through subject now requires the DSL.
    subj = ["witness: SeedIslandEased seg-birth term added"]
    files = [".omx/research/sub015_DAG_x.md", "experiments/train_levelset_witness_realized_through_R_mlx.py"]
    assert "DSL" in D.missing_legs(subj, files)


def test_noisy_stems_do_not_overfire():
    # LOW-1 fix: dropped launch/floor/law/erasure must not force a leg on unrelated chores.
    for subj in ("launcher: retry flaky ssh",
                 "fix floor division bug in rate calc",
                 "erasure coding: bump zfec dep",
                 "outlaw the old flag"):
        assert D.missing_legs([subj], ["tools/x.py"]) == [], f"over-fire on {subj!r}"


def test_per_commit_window_drifts_lever_masked_by_unrelated_dsl_touch():
    # LOW-2 fix: a lever commit whose OWN files miss the DSL drifts even if a DIFFERENT
    # commit in the same window touched the DSL.
    commits = [
        ("witness: new lever wired", ["experiments/train_levelset_witness_realized_through_R_mlx.py"]),
        ("dsl: unrelated docstring tweak", ["src/tac/witness_dsl/gauge.py"]),
    ]
    assert D.window_drifts(commits) is True
    # and if the lever commit DOES touch the DSL, the window is clean
    commits2 = [("witness: new lever wired", ["src/tac/witness_dsl/curriculum_dsl.py"])]
    assert D.window_drifts(commits2) is False


def test_per_commit_opt_out_is_per_commit():
    commits = [("witness: new lever wired [no-triality]", ["trainer.py"])]
    assert D.window_drifts(commits) is False


def test_build_reason_names_the_missing_leg():
    # LOW-3: the per-leg branch must actually name the leg, not just generic substrings.
    r = D.build_reason(["measured d_seg 0.0031 verdict"], [".omx/research/sub015_DAG_x.md"])
    assert "canonical equations" in r and "src/tac/canonical_equations" in r
    r2 = D.build_reason(["new lever wired"], ["trainer.py"])
    assert "src/tac/witness_dsl" in r2
