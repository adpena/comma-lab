"""#377 build-wave DSL leg: the FusedRKernel gap + 3 flagship #332 designed levers folded.

Charter (build-wave #377, operator GO 2026-07-09 "Build all unbuilt"): close the ONE
completeness gap P7 flagged — the score-neutral always-on ``--fused-r-kernel`` compute lever
(SPEC_v75 open-items / memory L70 / #348) — plus the unambiguous BOOLEAN-activation designed
levers from SPEC_v75 §10's "26 genuine designed levers" (the #332 signal-loss surface): the
costate ``--closed-loop-*`` controller, ``--curriculum-reanchor-levers`` (#302 M1), and LEVER-4
``--margin-saliency-reachability`` (#268). Value-configured clusters (eikonal magnitudes, pose
carrier, structured-init, ...) are documented as FOLD-OWED-to-#332 in the landing memo rather
than folded with invented magnitudes (fake-lever risk) — see the disposition table.

Same pattern as test_feed07_dsl_wirein: (C) composable + validate()==[] + real-argparse parse;
(D) activation-ledger known/never-fired/duty-to-measure visibility. Triality: DSL leg only
(equations N/A — these are transcription/compute levers, no new S_tau law). means != ends:
composition plumbing, NOT a score; the pointer moves only via a byte-closed exact row.
"""
from __future__ import annotations

import dataclasses as dc

from tac import witness_autoconfig as wac
from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as LR

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# The 4 factories this build-wave landed.
_NEW_LEVERS = (
    "FusedRKernel", "ClosedLoopEikonalControl", "CurriculumReanchorLevers",
    "MarginSaliencyReachability",
)

# The trainer flags each now MAPS (must all leave completeness().unmapped).
_NEWLY_MAPPED = frozenset({
    "--fused-r-kernel",
    "--closed-loop-control", "--closed-loop-eikonal-bump", "--closed-loop-eikonal-max",
    "--closed-loop-max-bumps", "--closed-loop-stop-after-windows",
    "--closed-loop-min-sustained-windows",
    "--curriculum-reanchor-levers",
    "--margin-saliency-reachability",
})


def test_new_levers_composable_and_return_single_lever():
    """(C) all 4 are zero-required-arg single-``Lever`` factories composable via --dsl-lever
    (the sibling class-fix predicate) — FusedRKernel/CurriculumReanchor/MarginSaliency are nilary;
    ClosedLoopEikonalControl has all-defaulted schedule params (trainer defaults, not invented)."""
    comp = LR.name_composable_levers()
    for name in _NEW_LEVERS:
        assert name in comp, f"{name} must be composable via --dsl-lever"
        assert isinstance(LR.resolve_composable_lever(name), cd.Lever)


def test_new_levers_validate_clean_over_baseline():
    """(C) DSL-program leg: each new lever over BASELINE validates clean (never-invent-flags,
    store_true-C2 for --margin-saliency-reachability, type-compat for the closed-loop schedule)."""
    for name in _NEW_LEVERS:
        prog = cd.BASELINE.with_lever(LR.resolve_composable_lever(name))
        assert prog.validate() == [], f"{name} must reference only real, type-compatible flags"


def test_fused_r_kernel_holds_the_gap_flag():
    """(C) the named gap: FusedRKernel emits exactly ``--fused-r-kernel`` True (score-neutral)."""
    lv = cd.FusedRKernel()
    assert lv.overrides == {"--fused-r-kernel": True}
    assert lv.epochs_delta == 0  # compute-config change, no epoch budget


def test_closed_loop_maps_all_six_flags_at_trainer_defaults():
    """(C) ClosedLoopEikonalControl holds the whole --closed-loop-* cluster at the TRAINER's
    designed defaults (faithful, not invented magnitudes)."""
    lv = cd.ClosedLoopEikonalControl()
    assert lv.overrides["--closed-loop-control"] is True
    assert lv.overrides["--closed-loop-eikonal-bump"] == 0.05
    assert lv.overrides["--closed-loop-eikonal-max"] == 0.20
    assert lv.overrides["--closed-loop-max-bumps"] == 2
    assert lv.overrides["--closed-loop-stop-after-windows"] == 3
    assert lv.overrides["--closed-loop-min-sustained-windows"] == 3
    # explicit override is faithfully typed (int stays int, float stays float)
    lv2 = cd.ClosedLoopEikonalControl(eikonal_bump=0.1, max_bumps=4)
    assert lv2.overrides["--closed-loop-eikonal-bump"] == 0.1
    assert lv2.overrides["--closed-loop-max-bumps"] == 4


def test_composition_parses_through_real_trainer_argparse():
    """(C) base sealed config + all 4 new levers -> the REAL trainer argparse accepts it."""
    ap = cd.build_real_trainer_parser()
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    argv = _render_argv(dc.replace(cfg, dsl_levers=_NEW_LEVERS).to_trainer_flags("OUT"))
    try:
        ap.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"build-wave composition argv rejected by the REAL trainer argparse (rc={exc.code})"
        ) from exc


def test_newly_mapped_flags_leave_unmapped():
    """(C) completeness() no longer flags any of the 9 folded flags as unmapped, and no DSL-emitted
    flag drifted stale (the folded overrides are all real trainer flags)."""
    c = LR.completeness()
    still = _NEWLY_MAPPED & set(c.unmapped)
    assert not still, f"these folded flags are still unmapped: {sorted(still)}"
    assert c.stale == [], f"folding introduced stale (dead/typo) flags: {c.stale}"


def test_new_levers_land_in_activation_ledger_duty_to_measure(tmp_path):
    """(D) ledger visibility: known_levers() auto-derives from lever_factories(), so the 4 new
    levers appear and — never having fired — sit in never_fired()/duty_to_measure() for the #247
    costate SENSE layer. Asserted against an ISOLATED empty ledger so a live event file cannot
    mask the auto-derivation."""
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers, never_fired
    known = known_levers()
    empty = tmp_path / "activation_ledger_empty.jsonl"  # nonexistent => zero events
    nf = never_fired(path=empty)
    duty = duty_to_measure(path=empty)
    for name in _NEW_LEVERS:
        assert name in known, f"{name} missing from the activation ledger's known set"
        assert name in nf, f"{name} must surface as never-fired on an empty ledger"
        assert name in duty, f"{name} must land in the SENSE duty_to_measure queue"


def test_margin_saliency_reachability_distinct_from_margin_saliency():
    """(C) guard against name/flag conflation: the new LEVER-4 reachability lever holds the
    store_true ``--margin-saliency-reachability``, NOT the pre-existing KKT-waterfill
    ``MarginSaliency`` weight lever."""
    lv = cd.MarginSaliencyReachability()
    assert lv.overrides == {"--margin-saliency-reachability": True}
    assert "--margin-saliency-reachability" not in cd.MarginSaliency().overrides


def _render_argv(pairs) -> list[str]:
    argv: list[str] = []
    for flag, val in pairs:
        argv.append(flag)
        if val is not None:
            argv.append(str(val))
    return argv
