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


# --- per-leg calibration (window granularity; adversarial review r1+r2 2026-07-06) ----
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


def test_lever_stem_requires_dsl_touch():
    # A lever/wire-in commit that touched only the DAG still requires the DSL leg (the teeth).
    subj = ["witness: island-birth lever wired into trainer"]
    files = [".omx/research/sub015_DAG_x.md", "experiments/train_levelset_witness_realized_through_R_mlx.py"]
    assert "DSL" in D.missing_legs(subj, files)


def test_noisy_stems_do_not_overfire():
    # dropped launch/floor/law/erasure must not force a leg on unrelated chores.
    for subj in ("launcher: retry flaky ssh",
                 "fix floor division bug in rate calc",
                 "erasure coding: bump zfec dep",
                 "outlaw the old flag"):
        assert D.missing_legs([subj], ["tools/x.py"]) == [], f"over-fire on {subj!r}"


def test_r2_dropped_broad_stems_do_not_overfire_on_dag_feed():
    # r2 MEDIUM-3 REGRESSION: the r1-added seed/island/activation/birth stems over-fired on
    # DAG-FEED / chore commits that merely MENTION a seed/island. Dropped → a DAG-FEED touching
    # only the DAG (the RECORDING mechanism) must be clean, not "DSL drift".
    # (subjects deliberately AVOID measure/verdict/lever verbs — this isolates the dropped
    #  seed/island/activation/birth stems; a subject that ALSO says "measured" SHOULD require
    #  the equations leg, which is correct and tested elsewhere.)
    for subj in ("DAG: FEED-06u paint-seed killed, island born early",
                 "bump random seed to 42",
                 "activation function refactor"):
        assert D.missing_legs([subj], [".omx/research/sub015_DAG_x.md"]) == [], f"over-fire: {subj!r}"


def test_r2_separate_dag_feed_commit_workflow_is_clean():
    # r2 MEDIUM-1 REGRESSION: the project MANDATES one-change-per-commit — a work commit and a
    # SEPARATE DAG-FEED commit. At window granularity that turn must be CLEAN (the work subject
    # requires no specific leg; the DAG touch in the same window satisfies the fallback). The
    # per-commit rewrite wrongly drifted on the work commit alone; window granularity is correct.
    subjects = ["witness probe n600 frontier: base_ch=32 run",
                "DAG FEED: record witness n600 probe trajectory point"]
    files = ["experiments/train_levelset_witness_realized_through_R_mlx.py",
             ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"]
    assert D.classify(subjects, files) == "clean"


def test_r4_dropped_measur_stem_does_not_overfire_on_dag_feed():
    # r4 REGRESSION: the everyday word "measured/measurement" over-fired on descriptive
    # commits AND DAG-FEED commits recording a routine measurement. Dropped from
    # EQUATION_REQUIRING; a DAG-FEED that RECORDS a measurement (only DAG touched) is clean.
    for subj in ("DAG FEED-06p: n600 measurement recorded",
                 "viz: measurement overlay for the descent tab",
                 "chore: re-measure the memory envelope"):
        assert D.missing_legs([subj], [".omx/research/sub015_DAG_x.md"]) == [], f"over-fire: {subj!r}"


def test_r4_real_finding_still_requires_equations():
    # The precise finding signals SURVIVE the measur-drop: a numeric d_seg row (_MEASURED_ROW),
    # a byte-close, a verdict, and the hyphenated "exact-row" spelling all still need equations.
    dag = [".omx/research/sub015_DAG_x.md"]
    assert "equations" in D.missing_legs(["measured d_seg 0.0031 n600"], dag)      # numeric row
    assert "equations" in D.missing_legs(["byte-close of the witness archive"], dag)
    assert "equations" in D.missing_legs(["verdict: lever-D NO-GO on the flip axis"], dag)
    assert "equations" in D.missing_legs(["first exact-row for the vehicle"], dag)  # hyphenated (r4)
    assert "equations" in D.missing_legs(["first exact row for the vehicle"], dag)  # space form


def test_r4_leverage_does_not_require_dsl():
    # r4+r6 cosmetic: "leverage"/"leveraged"/"leveraging" must NOT trip the DSL requirement.
    for benign in ("docs: leverage the existing cache better",
                   "perf: leveraged the fused kernel",
                   "wip: leveraging the atlas for speed"):
        assert D.missing_legs([benign], ["tools/x.py"]) == [], f"over-fire: {benign!r}"
    # but real lever words still do:
    assert "DSL" in D.missing_legs(["new lever wired"], ["tools/x.py"])
    assert "DSL" in D.missing_legs(["levers refactor in trainer"], ["tools/x.py"])
    assert "DSL" in D.missing_legs(["lever-D reactivation"], ["tools/x.py"])


def test_r6_measured_row_catches_short_connector_but_not_version():
    # r6: a SHORT connector (of/=/→/at, ≤6 chars) between the metric and a DECIMAL value still
    # counts as a finding — the intended recall widening over the old tight-adjacency regex.
    dag = [".omx/research/sub015_DAG_x.md"]
    for row in ("measured d_seg of 0.0031 n600", "d_pose = 3.4e-5", "d_seg→0.0047 best",
                "d_seg 0.0047"):
        assert "equations" in D.missing_legs([row], dag), f"missed finding: {row!r}"
    # ...but a version/epoch INTEGER near the metric (NO decimal value) is NOT a measured row.
    # (The window is deliberately ≤6 chars + requires a decimal — widening to catch long
    #  connectors like "dropped to" would start false-matching "d_seg v2.1"-style versions,
    #  a worse trade; long-connector phrasings are an accepted, documented miss.)
    for nonrow in ("d_pose head wired at ep50", "rename d_seg logging field",
                   "d_seg dashboard tab polish",
                   "witness d_seg v2.0 rewrite",       # r7: a VERSION token, not a measurement
                   "d_pose module v3.1 refactor"):
        assert D.missing_legs([nonrow], dag) == [], f"over-fire: {nonrow!r}"


def test_r5_non_string_subject_does_not_raise():
    # r5 robustness: the pure functions coerce with str() so a non-string subject can never
    # raise (unreachable from main(), but defensive).
    assert D.is_substantive([123]) is False
    assert D.is_opted_out([123]) is False
    assert D.missing_legs([123], [".omx/research/sub015_DAG_x.md"]) == []
    assert isinstance(D.build_reason([123], ["tools/x.py"]), str)
    assert D.classify([123, "witness lever wired"], ["tools/x.py"]) == "drift"  # the real subj still bites


def test_dogfood_pointer_footer_does_not_require_equations():
    # DOGFOOD 2026-07-06: the ubiquitous provenance FOOTER "pointer 0.19110 UNMOVED (apparatus)"
    # is boilerplate in ~every commit, NOT a measured finding — dropping "pointer\w*" from
    # EQUATION_REQUIRING stops it forcing the equations leg on apparatus commits.
    dag = [".omx/research/sub015_DAG_x.md"]
    for footer in ("#332 DSL de-orphaning ... pointer 0.19110 UNMOVED (apparatus)",
                   "kernel: fuse R+stem; pointer 0.19110 unmoved",
                   "docs: reword; pointer arithmetic cleanup"):
        assert D.missing_legs([footer], dag) == [], f"pointer-footer over-fire: {footer!r}"
    # ...but a genuine pointer MOVE states its mechanism (byte-close / exact-row / numeric row):
    assert "equations" in D.missing_legs(["frontier lowered via byte-close, pointer 0.185"], dag)
    assert "equations" in D.missing_legs(["first exact-row: pointer 0.185"], dag)


def test_build_reason_names_the_missing_leg():
    # LOW-3: the per-leg branch must actually name the leg, not just generic substrings.
    r = D.build_reason(["measured d_seg 0.0031 verdict"], [".omx/research/sub015_DAG_x.md"])
    assert "canonical equations" in r and "src/tac/canonical_equations" in r
    r2 = D.build_reason(["new lever wired"], ["trainer.py"])
    assert "src/tac/witness_dsl" in r2


# --- CONSUMER LEG (2026-07-07: DSL public-surface growth must reach the consumers) ---
_DSL_DIFF_PUBLIC_DEF = """\
diff --git a/src/tac/witness_dsl/curriculum_dsl.py b/src/tac/witness_dsl/curriculum_dsl.py
--- a/src/tac/witness_dsl/curriculum_dsl.py
+++ b/src/tac/witness_dsl/curriculum_dsl.py
@@ -10,0 +11,3 @@
+def EikonalAnnealLever(strength: float) -> "Lever":
+    return Lever(name="eikonal_anneal", strength=strength)
+
"""

_DSL_DIFF_DOCSTRING_AND_PRIVATE = """\
diff --git a/src/tac/witness_dsl/curriculum_dsl.py b/src/tac/witness_dsl/curriculum_dsl.py
--- a/src/tac/witness_dsl/curriculum_dsl.py
+++ b/src/tac/witness_dsl/curriculum_dsl.py
@@ -1,2 +1,4 @@
+    Reworded docstring line about levers and def conventions.
+def _private_helper(x):
+class _PrivateThing:
+    tau = 0.5  # tuned comment
"""

_DSL_DIFF_INIT_EXPORT = """\
diff --git a/src/tac/witness_dsl/__init__.py b/src/tac/witness_dsl/__init__.py
--- a/src/tac/witness_dsl/__init__.py
+++ b/src/tac/witness_dsl/__init__.py
@@ -5,0 +6,1 @@
+from .schedule_readback import ScheduleReadback
"""

_DSL_FILES = ["src/tac/witness_dsl/curriculum_dsl.py"]
_DSL_SUBJ = ["witness_dsl: add EikonalAnnealLever factory"]


def test_consumer_leg_fires_on_new_public_def_without_consumer_touch():
    # (1) new public DSL factory + no consumer surface touched → nudge fires.
    assert D.consumer_leg_missing(_DSL_SUBJ, _DSL_FILES, _DSL_DIFF_PUBLIC_DEF) is True
    assert D.consumer_leg_missing_safe(_DSL_SUBJ, _DSL_FILES, _DSL_DIFF_PUBLIC_DEF) is True


def test_consumer_leg_silent_when_consumer_touched():
    # (2) same growth, but a consumer surface was updated in the window → silent.
    for consumer in ("src/tac/witness_dsl/schedule_readback.py",
                     "tools/dashboard_server.py",
                     "tools/costate_digest.py",
                     "src/tac/witness_control/producer_bridge.py"):
        files = [*_DSL_FILES, consumer]
        assert D.consumer_leg_missing(_DSL_SUBJ, files, _DSL_DIFF_PUBLIC_DEF) is False, consumer


def test_consumer_leg_silent_with_consumers_generic_token():
    # (3) [consumers-generic] = the author's assertion that describe()/registry
    # introspection (generic rendering) covers the change → silent.
    subj = ["witness_dsl: add EikonalAnnealLever factory [consumers-generic]"]
    assert D.is_consumers_generic(subj)
    assert D.consumer_leg_missing(subj, _DSL_FILES, _DSL_DIFF_PUBLIC_DEF) is False


def test_consumer_leg_silent_on_non_public_change():
    # (4) docstring / private def / private class / field tweak → NOT public surface → silent.
    assert D.dsl_public_surface_added(_DSL_DIFF_DOCSTRING_AND_PRIVATE) is False
    assert D.consumer_leg_missing(
        _DSL_SUBJ, _DSL_FILES, _DSL_DIFF_DOCSTRING_AND_PRIVATE
    ) is False


def test_consumer_leg_fails_open_on_exception():
    # (5) an exception inside the new leg (here: a non-string diff object whose
    # .splitlines() access raises past the str-coercion via a hostile __str__) must
    # fail open — the safe wrapper returns False, never raises.
    class Hostile:
        def __str__(self):
            raise RuntimeError("boom")
    assert D.consumer_leg_missing_safe(_DSL_SUBJ, _DSL_FILES, Hostile()) is False


def test_consumer_leg_init_export_counts_as_public_surface():
    # __init__ export growth is public surface (a rename/re-export changes the API).
    files = ["src/tac/witness_dsl/__init__.py"]
    assert D.dsl_public_surface_added(_DSL_DIFF_INIT_EXPORT) is True
    assert D.consumer_leg_missing(["witness_dsl: re-export ScheduleReadback"],
                                  files, _DSL_DIFF_INIT_EXPORT) is True


def test_consumer_leg_respects_window_wide_opt_out():
    # [no-triality]/[skip-drift] (the existing escape valve) also silences the new leg.
    subj = ["witness_dsl: add factory scaffolding [no-triality]"]
    assert D.consumer_leg_missing(subj, _DSL_FILES, _DSL_DIFF_PUBLIC_DEF) is False


def test_consumer_leg_silent_when_dsl_not_touched():
    # No witness_dsl file in the window → the leg never evaluates the diff.
    assert D.consumer_leg_missing(_DSL_SUBJ, ["tools/x.py"], _DSL_DIFF_PUBLIC_DEF) is False


def test_consumer_nudge_text_is_advisory_and_documents_token():
    r = D.CONSUMER_NUDGE
    assert "[consumers-generic]" in r
    assert "describe()" in r
    assert "costate" in r and "dashboard" in r
    assert "Advisory" in r
    assert len(r) < 1000  # one firm nudge, not an essay


def test_existing_legs_unchanged_by_consumer_leg():
    # The consumer leg is ADDITIVE: classify() (the existing legs) is untouched — a
    # DSL-touching lever commit is still "clean" for classify() even when the new
    # leg would nudge (main() ORs them; the pure surfaces stay independent).
    assert D.classify(_DSL_SUBJ, _DSL_FILES) == "clean"


# --- VERDICT-SCOPE LEG (requirement R 2026-07-08: INSTANCE < FORMULATION < FAMILY <
# PARADIGM; one failed formulation is not a dead family) --------------------------------
_DOC = ".omx/research/witness_lever_verdict_20260708.md"


def test_verdict_scope_missing_declaration_blocks_with_exact_fix():
    added = ["## Lever-Q probe result",
             "Verdict: NO-GO on the flip axis at n600 (0.0083 vs 0.0082 baseline)."]
    v = D.verdict_scope_violations(_DOC, added)
    assert len(v) == 1
    assert "verdict_scope: formulation" in v[0]  # the exact one-line fix is shown
    assert _DOC in v[0]


def test_verdict_scope_declared_instance_passes():
    added = ["Verdict: NO-GO at n96 smoke.",
             "verdict_scope: instance — K=64 max-plus fit, default hyperparameters"]
    assert D.verdict_scope_violations(_DOC, added) == []


def test_verdict_scope_declared_formulation_kill_needs_reformulation_queue():
    added = ["Verdict: KILLED — pooled-unsigned UniWARD cost field at n600.",
             "verdict_scope: formulation — pooled-unsigned form"]
    v = D.verdict_scope_violations(_DOC, added)
    assert len(v) == 1 and "reformulation" in v[0]
    # ...and enumerating the untested alternatives clears it:
    added2 = [*added, "untested formulations / alternatives: signed field, hinge cost, per-range"]
    assert D.verdict_scope_violations(_DOC, added2) == []


def test_verdict_scope_family_requires_citation_or_two_formulations():
    base = ["Verdict: FALSIFIED across the basis family.",
            "verdict_scope: family"]
    v = D.verdict_scope_violations(_DOC, base)
    assert any("family" in m and ("arXiv" in m or "formulations" in m) for m in v)
    # citation clears it:
    ok1 = [*base, "Per Candes-Donoho theorem (arXiv:math/0011037), no isotropic basis attains the rate."]
    assert not any("citation" in m for m in D.verdict_scope_violations(_DOC, ok1))
    # explicit ≥2-distinct-formulations evidence clears it:
    ok2 = [*base, "Killed across two structurally distinct formulations at n600."]
    assert not any("citation" in m for m in D.verdict_scope_violations(_DOC, ok2))


def test_verdict_scope_family_falsified_with_citation_needs_no_reformulation_queue():
    # FALSIFIED is not KILL-class → the reformulation-queue rule does not apply.
    added = ["Verdict: FALSIFIED across the family.",
             "verdict_scope: family",
             "Theorem (impossibility bound): see arXiv:2401.00001."]
    assert D.verdict_scope_violations(_DOC, added) == []


def test_verdict_scope_quoted_and_negated_lines_exempt():
    added = ["> prior run said KILLED — quoting for context",          # md quote
             'the earlier "NO-GO" was over-scoped and is reopened',    # quoted + cue
             "this is not a KILL of the family, merely a config miss",  # negation cue
             "the `FALSIFIED` token itself is what this rule scans for",  # code span
             "discussion of verdict_scope: family semantics and KILL handling"]  # rule line
    assert D.negative_verdict_tokens(added) == ([], False)
    assert D.verdict_scope_violations(_DOC, added) == []


def test_verdict_scope_lowercase_prose_not_tokens():
    # single-word tokens are CASE-SENSITIVE uppercase: everyday prose stays silent.
    added = ["killed the stale process; dead code removed; no-go areas in the parser",
             "the deadline is Friday; inert gas analogy; falsified nothing here"]
    assert D.negative_verdict_tokens(added) == ([], False)


def test_verdict_scope_multiword_phrases_case_insensitive():
    tokens, kill = D.negative_verdict_tokens(
        ["the lever performed At Chance on the reachability axis"])
    assert tokens and kill is False
    tokens2, _ = D.negative_verdict_tokens(["conclusion: family dead per this one run"])
    assert tokens2


def test_verdict_scope_waiver_exempts_and_placeholder_rejected():
    line = "Verdict: NO-GO  # VERDICT_SCOPE_OK: legacy 2026-06 memo migration, scoped upstream"
    assert D.verdict_scope_violations(_DOC, [line]) == []
    bad = "Verdict: NO-GO  # VERDICT_SCOPE_OK:<rationale>"
    assert len(D.verdict_scope_violations(_DOC, [bad])) == 1
    bad2 = "Verdict: NO-GO  # VERDICT_SCOPE_OK: tbd"
    assert len(D.verdict_scope_violations(_DOC, [bad2])) == 1


def test_verdict_doc_in_scope_patterns_and_exemptions():
    assert D.verdict_doc_in_scope(".omx/research/lever_q_verdict_20260708.md")
    assert D.verdict_doc_in_scope(".omx/research/probe_eikonal_n600_20260708.md")
    assert D.verdict_doc_in_scope(".omx/research/council_x_review_20260708.md")
    assert D.verdict_doc_in_scope(".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md")
    # exempt: rule/ledger/memory/index surfaces (they QUOTE verdicts + state the rule)
    assert not D.verdict_doc_in_scope(".omx/research/t5_crucible/ORCHESTRATION_LEDGER.md")
    assert not D.verdict_doc_in_scope(".omx/research/MEMORY_CLUSTER_council_2026Q3.md")
    assert not D.verdict_doc_in_scope(".omx/research/CANONICAL_RESEARCH_INDEX_20260629.md")
    assert not D.verdict_doc_in_scope(".omx/research/verdict_scope_rule_design_20260708.md")
    # out of scope: non-md and non-decision paths
    assert not D.verdict_doc_in_scope("tools/foo.py")
    assert not D.verdict_doc_in_scope(".omx/research/random_notes_20260708.md")


def test_added_lines_from_diff_extracts_only_additions():
    diff = ("diff --git a/x.md b/x.md\n--- a/x.md\n+++ b/x.md\n@@ -1,2 +1,3 @@\n"
            " context line\n-removed KILLED line\n+added line one\n+added line two\n")
    assert D.added_lines_from_diff(diff) == ["added line one", "added line two"]
    assert D.added_lines_from_diff(None) == []


def test_verdict_scope_evidence_text_strips_declaration_lines():
    # measured 2026-07-08: including the declaration makes the FM echo it back.
    added = ["Verdict: NO-GO at n600.", "verdict_scope: family", "evidence body"]
    ev = D.verdict_scope_evidence_text(added)
    assert "verdict_scope" not in ev and "evidence body" in ev


def test_fm_scope_advisories_fail_open_when_fm_absent(monkeypatch):
    monkeypatch.setenv("VERDICT_SCOPE_FM_PYTHON", "/nonexistent/python")
    monkeypatch.setenv("DASH_FM_PYTHON", "/nonexistent/python")
    monkeypatch.setattr(D.os.path, "expanduser", lambda p: "/nonexistent/python")
    out = D.fm_scope_advisories([{"path": "x.md", "declared": ["family"], "evidence": "e"}])
    assert out == []
    assert D.fm_scope_advisories([]) == []


def test_verdict_scope_violations_fail_open_on_hostile_input():
    class Hostile:
        def __str__(self):
            raise RuntimeError("boom")
    # non-string added lines coerce defensively in the token scan / joins; a hostile
    # __str__ must not escape the leg's fail-open posture in main() — the pure fn may
    # raise, but main() wraps the whole leg in try/except. Verify the common non-string
    # cases do NOT raise:
    assert D.negative_verdict_tokens([None, 123]) == ([], False)
    assert D.verdict_scope_violations(_DOC, [None, 123]) == []


# --------------------- RECALL-DEPTH LEG (#713; ddm_hw1 task #785) ---------------------
def test_recall_depth_ledger_append_without_recall_fires_advisory():
    subs = ["ledger QA88: flip row to BUILT"]
    files = [".omx/research/ddm_deferral_queue_ledger_20260729.md"]
    appended = {files[0]: "QA88 BUILT hot-state manifest shipped"}
    advs = D.recall_depth_advisories(subs, files, appended)
    assert len(advs) == 1
    assert "RECALL-DEPTH" in advs[0] and "ADVISORY (never blocks)" in advs[0]


def test_recall_depth_dag_append_without_recall_fires():
    subs = ["routing FEED: new landing"]
    files = [".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"]
    appended = {files[0]: "FEED: shipped a thing"}
    assert len(D.recall_depth_advisories(subs, files, appended)) == 1


def test_recall_depth_body_recall_token_silences():
    subs = ["ledger QA88: BUILT"]
    files = [".omx/research/ddm_deferral_queue_ledger_20260729.md"]
    appended = {files[0]: "BUILT; STORES CONSULTED: memory: all_arms; per [[foo]]"}
    assert D.recall_depth_advisories(subs, files, appended) == []


def test_recall_depth_subject_recall_token_silences():
    # A recall token in the commit SUBJECT covers all appends in the window.
    subs = ["ledger QA88 BUILT (grepped .omx/research + memory: prior work)"]
    files = [".omx/research/ddm_deferral_queue_ledger_20260729.md"]
    appended = {files[0]: "QA88 BUILT"}
    assert D.recall_depth_advisories(subs, files, appended) == []


def test_recall_depth_non_ledger_file_silent():
    assert D.recall_depth_advisories(["x"], ["src/tac/foo.py"], {}) == []


def test_recall_depth_is_ledger_or_dag_append_matcher():
    assert D.is_ledger_or_dag_append(".omx/research/sub015_DAG_topaiml_x.md")
    assert D.is_ledger_or_dag_append(".omx/research/ddm_deferral_queue_ledger_20260729.md")
    assert not D.is_ledger_or_dag_append("src/tac/witness_dsl/foo.py")
    assert not D.is_ledger_or_dag_append(".omx/research/some_memo_20260730.md")
    assert not D.is_ledger_or_dag_append(None)


def test_recall_depth_safe_wrapper_fail_open():
    # A hostile appended_text mapping value must not escape the fail-open wrapper.
    class Hostile:
        def __str__(self):
            raise RuntimeError("boom")

    files = [".omx/research/ddm_deferral_queue_ledger_20260729.md"]
    out = D.recall_depth_advisories_safe(["s"], files, {files[0]: Hostile()})
    assert out == []


# ------------------ SHIFT-LEFT LEG CLASSIFIER (ddm_hw1 task #785) --------------------
def test_owed_legs_line_dsl_lever_change():
    line = D.owed_legs_line("witness: new lever wired", ["experiments/train.py"])
    assert "DSL" in line and "shift-left" in line


def test_owed_legs_line_equations_measured_row():
    line = D.owed_legs_line("witness verdict d_seg 0.0031 measured", ["experiments/x.py"])
    assert "equations" in line


def test_owed_legs_line_opt_out_silent():
    assert D.owed_legs_line("witness: new lever wired [no-triality]", ["experiments/x.py"]) == ""


def test_owed_legs_line_ordinary_chore_silent():
    assert D.owed_legs_line("docs: fix a typo", ["docs/x.md"]) == ""


def test_owed_legs_line_dsl_touch_satisfies_dsl_leg():
    # A lever change that DID touch the DSL leg owes no DSL suggestion.
    line = D.owed_legs_line("witness: new lever", ["src/tac/witness_dsl/foo.py"])
    assert "DSL (" not in line


def test_owed_legs_line_consumer_leg_on_public_surface_growth():
    diff = (
        "+++ b/src/tac/witness_dsl/new_mod.py\n"
        "+def NewLeverFactory(x):\n"
    )
    line = D.owed_legs_line(
        "witness: new lever factory", ["src/tac/witness_dsl/new_mod.py"], diff
    )
    assert "consumer" in line


def test_owed_legs_line_fail_open_on_hostile_files():
    class Hostile:
        def __str__(self):
            raise RuntimeError("boom")

    # A hostile file entry must not break a commit — the classifier returns "".
    assert D.owed_legs_line("witness: lever", [Hostile()]) == ""
