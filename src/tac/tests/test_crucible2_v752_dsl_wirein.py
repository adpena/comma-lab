"""T5 CRUCIBLE-2 v7.5.2 launch-1 DSL wire-in (P7 DELIVERABLE, task #379).

The committed, reproducible standing form of the P6-seal compile smoke: the SEALED v7.5.2 launch-1
config (`SYNTHESIS_v3_v752_20260709.md`, sealed at P6-R13) authored AS a typed
`TypedWitnessConfig` (via `witness_autoconfig.derive_crucible_v752_config`) VALIDATES clean AND its
compiled argv PARSES through the trainer's REAL argparse (0 unknown flags), and its
`dsl_program_manifest` VERIFIES — proving the launch-1 ON-set is DSL-provenance-typed, not
hand-assembled.

This is the FEED-07 pattern (compile-through-real-argparse + validate()==[] + coverage) applied to
v7.5.2. It ALSO pins the P7 transcription discipline: the EXCLUDED launch-1 items (amber realization
OI-5, freq_along-26/lane_carried OI-4, --pose-finish-engage-on owed-1) are asserted to be NOT
silently included — they are P8-wall decisions the sealed doc defers, guarded here so a future edit
cannot silently close them.

means != ends: config plumbing, NOT a score. Pointer contest-CPU 0.19110 UNMOVED; only a byte-closed
upstream/evaluate.py n600 row < 0.19110 moves it. The launch ships banked-R1 pose (d_pose 0.001610 →
0.127) as its floor; d_seg is the open axis.
"""
from __future__ import annotations

from tac import witness_autoconfig as wac
from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as LR
from tac.witness_dsl.typed_config import verify_launch_manifest

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _argv_flag_value(argv, flag):
    """Return the value emitted for ``flag`` in a compiled argv, or a sentinel.

    ``'__BARE__'`` for a bare store_true flag, ``'__ABSENT__'`` if not emitted."""
    toks = list(argv)
    if flag not in toks:
        return "__ABSENT__"
    i = toks.index(flag)
    nxt = toks[i + 1] if i + 1 < len(toks) else None
    if nxt is None or str(nxt).startswith("--"):
        return "__BARE__"
    return str(nxt)


def test_crucible_v752_launch1_validates_and_parses_end_to_end():
    """The v7.5.2 launch-1 typed program validates clean AND compiles through the REAL argparse."""
    typed, argv, manifest = wac.compile_crucible_v752_config(_GT, num_pairs=600, epochs=3000)
    assert typed.name == "crucible_v752"
    # (1) DSL validate: 0 violations (never-invent-flags + store_true-C2 + type-compat + ordering).
    assert typed.validate_program() == [], "v7.5.2 launch-1 must be DSL-valid"
    # (2) compile-through-real-argparse: the P6 compile smoke, now standing. 0 unknown flags.
    ap = cd.build_real_trainer_parser()
    try:
        ap.parse_args(list(argv)[2:])  # skip [python, trainer_path]
    except SystemExit as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"crucible_v752 launch-1 argv rejected by the REAL trainer argparse (rc={exc.code})"
        ) from exc
    # (3) DSL-provenance manifest verifies against the emitted flag-name fingerprint (rc=7 gate input).
    emitted = sorted({f for f, _ in wac._crucible_v7_argv_pairs(argv)})
    ok, detail = verify_launch_manifest(manifest, emitted)
    assert ok, f"v7.5.2 dsl_program_manifest must verify: {detail}"
    assert manifest["program_name"] == "crucible_v752"
    assert manifest["schema"] == "dsl_program_manifest.v1"


def test_crucible_v752_inherits_sealed_crucible_v7_base():
    """v7.5.2 = crucible_v7 (the sealed `base:`) + the launch-1 delta — the inheritance is real.

    Asserts the sealed crucible_v7 trunk is carried through (directional basis + AA-ipe + counter-
    force + temporal-screw + terminal pose-finish backstop + safe-compile + muon-event)."""
    v7 = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    typed = wac.derive_crucible_v752_config(_GT, num_pairs=600, epochs=3000)
    argv = typed.to_program().compile_trainer_argv()
    # the crucible_v7 sealed trunk survives into v7.5.2 (spot-check the load-bearing flags).
    assert _argv_flag_value(argv, "--self-orient") == "__BARE__"          # directional basis
    assert _argv_flag_value(argv, "--render-aa") == "ipe"                 # AA-ipe (NOT supersample)
    assert _argv_flag_value(argv, "--muon-start-event") == "powerlaw_meat"  # v3 AMENDMENT-6
    assert _argv_flag_value(argv, "--safe-compile-regions") == "hosc_activation"
    assert _argv_flag_value(argv, "--pose-finish-start-epoch") != "__ABSENT__"  # backstop (owed-1 uses it)
    # wall-clock budget preserved (same epochs => same DERIVED budget; no stale re-derive).
    assert typed.budget_days_value() == v7.budget_days_value()


def test_crucible_v752_delta_is_only_the_genuine_new_taper():
    """HONEST provenance (§8.5 no-op-claim guard): the ONLY genuinely-new base flag vs crucible_v7 is
    #121 d_seg-aware taper; fused-R/async-verdict are INHERITED (idempotent re-assert)."""
    v7 = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    typed = wac.derive_crucible_v752_config(_GT, num_pairs=600, epochs=3000)
    # #121 taper is GENUINELY NEW (absent from crucible_v7, present in v752).
    for f in ("--dseg-aware-taper", "--dseg-aware-taper-strength",
              "--dseg-aware-taper-scale", "--dseg-aware-taper-floor"):
        assert f not in v7.base, f"{f} should NOT be in crucible_v7 (v752 delta)"
        assert f in typed.base, f"{f} must be in v752 base"
    # fused-R + async-verdict are INHERITED from crucible_v7 (re-assert is a no-op, documented).
    for f in ("--fused-r-kernel", "--async-verdict"):
        assert f in v7.base and f in typed.base, f"{f} is inherited from crucible_v7, not new"
    # W-1 REMOVAL: σ_cc′ (--length-sigma-matrix) was ON in crucible_v7 (inherited from v6), REMOVED
    # from v7.5.2 launch-1 (→ trainer default σ≡1). A genuine sealed delta, not just additions.
    assert v7.base.get("--length-sigma-matrix") == "fitted-20260707", (
        "crucible_v7 inherits σ_cc′ ON — the premise of the W-1 removal")
    assert "--length-sigma-matrix" not in typed.base, (
        "v7.5.2 must REMOVE σ_cc′ from launch-1 (W-1: ladder rung 1b own increment)")


def test_crucible_v752_excluded_items_are_not_silently_included():
    """P7 transcription discipline: the sealed-DEFERRED / owed launch-1 items are NOT silently
    decided by this typed program. Guard so a future edit cannot quietly close an OPEN-ITEM."""
    typed = wac.derive_crucible_v752_config(_GT, num_pairs=600, epochs=3000)
    argv = typed.to_program().compile_trainer_argv()
    # OI-5 amber: NOT included (would be silently defeated by the inherited explicit --grad-clip 1.0;
    #   trainer L9989 "explicit ALWAYS wins over --stability-preset"). The realization is a P8-wall item.
    assert _argv_flag_value(argv, "--stability-preset") in ("__ABSENT__", "none"), (
        "amber must NOT be naively composed (silently defeated by inherited --grad-clip 1.0; OI-5)")
    # OI-4 freq_along: crucible_v7's lane_offloaded regime law emits 6; the sealed §B's 26 is the
    #   lane_carried regime — a launch-decision regime choice, NOT a silent P7 override to 26.
    assert _argv_flag_value(argv, "--freq-along") == "6.0", (
        "freq_along must stay crucible_v7's lane_offloaded 6 (sealed 26=lane_carried is OI-4, a P8 choice)")
    # owed-1 pose-gate: --pose-finish-engage-on does not exist yet (built by #383); must NOT be stubbed.
    # (COMMENT UPDATED 2026-07-10 — #383 LANDED; fresh-eyes advisory P0-1) the engage-on flag EXISTS
    # now. The SEALED derive artifact stays a TRANSCRIPTION (the seal predates the build); the
    # LAUNCHER-facing program COMPOSES it — PRESENCE + the expected-active-lever manifest are asserted
    # in test_crucible_v752_launch_config_wiring.
    assert _argv_flag_value(argv, "--pose-finish-engage-on") == "__ABSENT__", (
        "the SEALED derive transcription must not silently compose the #383 gate; the composition "
        "rides compile_crucible_v752_launch_config (asserted PRESENT there)")
    # ladder-only levers must NOT be in launch-1 (σ_cc′ own-increment rung 1b).
    assert _argv_flag_value(argv, "--length-sigma-matrix") == "__ABSENT__", (
        "σ_cc′ is ladder rung 1b (own increment), NOT launch-1")


def test_crucible_v752_self_orient_off_drops_directional_frontend():
    """owed-16 P9 RESOLVED-REFUTING amendment (operator GO 2026-07-10): self_orient=False DROPS the
    self-orient directional basis (flags + the FEED_07a lever), keeping the always-on curvelet bank
    (--max-bank-freq) and everything else. This is the GO'd pointer-moving launch config."""
    on = wac.derive_crucible_v752_config(_GT, num_pairs=600, epochs=3000)                    # sealed ON
    off = wac.derive_crucible_v752_config(_GT, num_pairs=600, epochs=3000, self_orient=False)  # GO'd OFF
    argv_off = off.to_program().compile_trainer_argv()
    # OFF still validates + parses through the REAL argparse.
    assert off.validate_program() == [], "self-orient-OFF v752 must be DSL-valid"
    ap = cd.build_real_trainer_parser()
    ns = ap.parse_args(list(argv_off)[2:])
    assert ns.self_orient is False, "OFF config: --self-orient absent ⇒ trainer default False"
    # the 5 self-orient-only flags are DROPPED; --max-bank-freq (always-on curvelet bank) is KEPT.
    for f in ("--self-orient", "--n-dir-freqs", "--freq-across", "--freq-along", "--reorient-every"):
        assert _argv_flag_value(argv_off, f) == "__ABSENT__", f"{f} must be dropped when self-orient OFF"
    assert _argv_flag_value(argv_off, "--max-bank-freq") != "__ABSENT__", (
        "--max-bank-freq feeds the ALWAYS-ON curvelet bank (not self-orient) — must be KEPT")
    # the FEED_07a directional lever is removed from the composition (else it re-emits the flags).
    assert "FEED_07a_directional_basis_rebalance" in {lv.name for lv in on.levers}
    assert "FEED_07a_directional_basis_rebalance" not in {lv.name for lv in off.levers}
    # the rest of the trunk is UNCHANGED (spot-check the load-bearing non-directional flags survive).
    assert _argv_flag_value(argv_off, "--render-aa") == "ipe"
    assert _argv_flag_value(argv_off, "--lane-render-band") == "__BARE__"        # lane_offloaded band kept
    assert _argv_flag_value(argv_off, "--safe-compile-regions") == "hosc_activation"
    assert _argv_flag_value(argv_off, "--dseg-aware-taper") == "__BARE__"        # #121 taper kept
    assert _argv_flag_value(argv_off, "--pose-finish-start-epoch") != "__ABSENT__"
    # ON and OFF are DISTINCT typed configs (different hash) — the amendment is real, not a no-op.
    assert on.typed_config_hash() != off.typed_config_hash()


def test_crucible_v752_launch_config_wiring():
    """The P8-wall launcher wire-in (operator GO 2026-07-10): compile_crucible_v752_launch_config
    returns the full launcher-facing cfg protocol with self-orient OFF by default, reusing v7's
    constants + schedule-governance manifests and carrying its own DSL-provenance fingerprint.
    P0-1 (fresh-eyes advisory, 2026-07-10): the #383 PoseFinishConditioningGate is COMPOSED —
    --pose-finish-engage-on sigma_min_plateau is PRESENT (the prior absence assertion was the stale
    dangerous-direction regression the advisory flagged)."""
    lc = wac.compile_crucible_v752_launch_config(_GT, num_pairs=600, epochs=3000)  # default self_orient=False
    assert lc.name == "crucible_v752"
    # launch config is self-orient-OFF by construction (the GO'd amendment).
    cmd = lc.to_command("experiments/results/__probe__", perf_env=True)
    assert "--self-orient" not in cmd, "launch config must be self-orient-OFF (owed-16 GO'd amendment)"
    assert "FEED_07a_directional_basis_rebalance" not in lc.dsl_levers
    # P0-1: the #383 conditioning gate is composed — PRESENCE asserted (was stale-absent).
    argv = list(lc.typed.to_program().compile_trainer_argv())
    assert _argv_flag_value(argv, "--pose-finish-engage-on") == "sigma_min_plateau", (
        "P0-1: the launch program must compose PoseFinishConditioningGate "
        "(--pose-finish-engage-on sigma_min_plateau) — #383 landed; built-but-not-composed is the bug")
    assert "pose_finish_conditioning_gate" in lc.dsl_levers
    # the backstop + finish weight come from the inherited pose config (the lever adds ONLY the mode).
    assert _argv_flag_value(argv, "--pose-finish-start-epoch") == "726"
    assert _argv_flag_value(argv, "--w-pose") == "1.0"
    # P0-1 required gate: the expected-active-lever manifest is pinned + carried + matches.
    exp = lc.dsl_program_manifest.get("expected_active_levers")
    assert exp is not None, "dsl_program_manifest must carry expected_active_levers (advisory P0-1)"
    assert sorted(exp) == sorted(lc.dsl_levers) == sorted(wac.CRUCIBLE_V752_LAUNCH_EXPECTED_LEVERS)
    # the three provenance manifests the launcher gate chain consumes are present + coherent.
    assert lc.dsl_program_manifest["program_name"] == "crucible_v752"
    assert lc.constants_manifest, "constants_manifest reused from crucible_v7 (identical LawRefs)"
    assert lc.schedule_governance, "schedule_governance reused from crucible_v7 (identical WHEN tokens)"
    # the DSL-provenance manifest verifies against the emitted argv flag fingerprint (rc=7 gate input).
    emitted = sorted({f for f, _ in wac._crucible_v7_argv_pairs(argv)})
    ok, detail = verify_launch_manifest(lc.dsl_program_manifest, emitted)
    assert ok, f"launch-config dsl_program_manifest must verify: {detail}"
    # wall-clock budget inherited from crucible_v7 (conservative ceiling for the smaller OFF footprint).
    assert lc.epochs == 3000


def test_crucible_v752_pilot_is_two_token_diff_with_gate_composed():
    """The pilot (epochs=300) stays exactly a 2-token diff from the launch config WITH the P0-1 gate
    composed on both sides (--epochs + the Polyak degenerate clamp, both inert <= ep300) — the
    resume-from-pilot contract of the pre-registered decision record."""
    import itertools
    a300 = list(wac.compile_crucible_v752_launch_config(
        _GT, num_pairs=600, epochs=300).typed.to_program().compile_trainer_argv())
    a3000 = list(wac.compile_crucible_v752_launch_config(
        _GT, num_pairs=600, epochs=3000).typed.to_program().compile_trainer_argv())
    diffs = [(a, b) for a, b in itertools.zip_longest(a300, a3000) if a != b]
    assert diffs == [("300", "3000"), ("301", "2546")], (
        f"pilot must differ from the launch in exactly --epochs + the polyak clamp; got {diffs}")


def test_resolve_pose_finish_engage_backstop_semantics():
    """P0-2 minimum-viable guard (fresh-eyes advisory): the epoch backstop cannot override a
    DEGENERATE / should_ship_banked_r1 classification; a real conditioning fire always engages;
    healthy-but-never-fired at the backstop = the fail-safe engage (unchanged)."""
    from tac.witness_control.sigma_min_plateau import (
        backstop_banked_r1_row,
        resolve_pose_finish_engage,
    )

    # a real fire engages regardless of everything else.
    assert resolve_pose_finish_engage(cond_fired=True, backstop_hit=False, degenerate=False) == (True, False)
    assert resolve_pose_finish_engage(cond_fired=True, backstop_hit=True, degenerate=False) == (True, False)
    # pre-backstop, not fired: pose-blind, no banked decision.
    assert resolve_pose_finish_engage(cond_fired=False, backstop_hit=False, degenerate=False) == (False, False)
    assert resolve_pose_finish_engage(cond_fired=False, backstop_hit=False, degenerate=True) == (False, False)
    # THE FIX: degenerate at the backstop epoch => NO engage + banked-R1 selected (was: engage anyway).
    assert resolve_pose_finish_engage(cond_fired=False, backstop_hit=True, degenerate=True) == (False, True)
    # healthy-but-never-fired at the backstop => the fail-safe engage (unchanged incumbent semantics).
    assert resolve_pose_finish_engage(cond_fired=False, backstop_hit=True, degenerate=False) == (True, False)
    # the loud typed row carries the custody fields.
    row = backstop_banked_r1_row(726, backstop_epoch=726, classification="DEGENERATE_GUARD_TRIPPED",
                                 n_points=9)
    assert row["stage"] == "confound_alarm"
    assert row["alarm"] == "pose_finish_backstop_overridden_banked_r1"
    assert row["should_ship_banked_r1"] is True and row["epoch"] == 726


def test_crucible_v752_completeness_fused_r_folded_and_taper_held():
    """completeness() disposition, POST-#377 fold (commit 3925001ec): --fused-r-kernel is now
    DSL-MAPPED — the score-neutral compute gap the prior '--fused-r-kernel is the only gap'
    disposition named is CLOSED (FusedRKernel folded as a Lever factory; SPEC §9 completeness note
    predates the fold). The launch-1 taper/async delta flags remain DSL-held. The v752 base still
    carries designed levers being folded incrementally under the DSL registry; this test pins the two
    STABLE facts (fused-r closed + taper held), not the in-flight unmapped count."""
    c = LR.completeness()
    unmapped = set(c.unmapped)
    # #377 (3925001ec) folded FusedRKernel into the DSL: the prior completeness gap is CLOSED.
    assert "--fused-r-kernel" not in unmapped, (
        "--fused-r-kernel was folded into the DSL by #377/3925001ec; it must be completeness().mapped now")
    # the launch-1 delta levers remain DSL-held (completeness().mapped).
    for f in ("--dseg-aware-taper", "--dseg-aware-taper-strength",
              "--dseg-aware-taper-scale", "--dseg-aware-taper-floor", "--async-verdict"):
        assert f not in unmapped, f"{f} should be DSL-held (completeness().mapped)"
