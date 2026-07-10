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
    assert _argv_flag_value(argv, "--pose-finish-engage-on") == "__ABSENT__", (
        "--pose-finish-engage-on is owed-1 (unbuilt); the program rides --pose-finish-start-epoch backstop")
    # ladder-only levers must NOT be in launch-1 (σ_cc′ own-increment rung 1b).
    assert _argv_flag_value(argv, "--length-sigma-matrix") == "__ABSENT__", (
        "σ_cc′ is ladder rung 1b (own increment), NOT launch-1")


def test_crucible_v752_completeness_gap_is_only_fused_r_and_documented():
    """The completeness() gap disposition: of the launch-1 base surface, ONLY --fused-r-kernel is
    DSL-unmapped (a score-neutral perf lever, documented pre-DSL); the taper/async flags ARE held."""
    c = LR.completeness()
    unmapped = set(c.unmapped)
    assert "--fused-r-kernel" in unmapped, "--fused-r-kernel is the documented completeness() gap"
    for f in ("--dseg-aware-taper", "--dseg-aware-taper-strength",
              "--dseg-aware-taper-scale", "--dseg-aware-taper-floor", "--async-verdict"):
        assert f not in unmapped, f"{f} should be DSL-held (completeness().mapped)"
