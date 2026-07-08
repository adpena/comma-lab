"""crucible_v7 — the FIRST requirement-V-native launch config (authored AS a
TypedWitnessConfig; DSL-emitted argv) VERIFICATION HARNESS.

Asserts (the council's structural pre-checks + the schedule-provenance gate):
  * the typed config validates (0 WitnessProgram.validate violations);
  * the schedule-provenance gate classifies the emitted argv with 0 NAKED;
  * the mutually-excluded --tau-softplus-start-epoch is ABSENT (+ l7 + tau-hold-frac);
  * the pose block is VERBATIM vs v6;
  * the DSL-provenance manifest fingerprint matches the emitted argv (+ fails on drift);
  * the diff-vs-v6 table is exactly the designed set of deltas (stability).

means != ends: gates a MEANS. Only a byte-closed n600 exact row < 0.19110 moves the pointer.
"""
from __future__ import annotations

import pytest

import tac.witness_autoconfig as wac
import tools.schedule_provenance_gate as gate
from tac.witness_dsl.curriculum_dsl import TRAINER_REL
from tac.witness_dsl.typed_config import (
    PROGRAM_MANIFEST_SCHEMA,
    TypedWitnessConfig,
    verify_launch_manifest,
)

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

# The DESIGNED diff-vs-v6 deltas (DRAFT §1 + §2) — the stability fixture. Any drift here is a
# real config change the council must see, not a silent edit.
_EXPECTED_ADDED = frozenset({
    "--seg-form-unify-tau",
    "--tail-cycles-max", "--tail-start-epoch", "--tail-cycle-floor-epochs", "--tail-dwell-min",
    "--tail-tau-halving", "--tail-lr-prop-tau", "--tail-stop-marginal-s",
    "--ladder-island-homotopy", "--ladder-movable-r0", "--ladder-movable-birth-epochs",
    "--ladder-movable-hold-epochs", "--ladder-movable-anneal-epochs", "--ladder-movable-lambda-gate",
    "--ladder-lane-r0", "--ladder-lane-birth-epochs", "--ladder-lane-hold-epochs",
    "--ladder-lane-anneal-epochs", "--ladder-lane-lambda-gate", "--ladder-gate-softness",
    "--ladder-release-coeff", "--ladder-sigma-eff", "--ladder-lane-dash-gate",
    "--ladder-max-step-px", "--ladder-refresh-every",
    # (operator override 2026-07-08) the three co-emitted SENSOR->START WIRING flags (each transition
    # now FIRES on its sensor; the paired --*-start-epoch is a fail-safe backstop cap).
    "--muon-start-event", "--lane-band-start-event", "--seg-chroma-boundary-start-event",
    # the DSL WitnessProgram's VerdictCadence defaults (v6 emits via its dataclass, which lacks them;
    # v7 IS the DSL-emitted argv, which carries them) — a v6-vs-v7 EMITTER delta, not a semantic knob.
    "--verdict-device", "--verdict-anchor-every",
    # (S6-R4 self-paced τ-advance, operator 2026-07-08) the LAST clock-hardcoding (the anneal-epochs τ
    # denominator) converted to the EVENT-driven geometric octave ladder. Only the mode flag is emitted;
    # the octave count / dwell caps DERIVE in the trainer (no bare literals in the config).
    "--tau-advance-mode",
})
_EXPECTED_REMOVED = frozenset({
    "--tau-softplus-start-epoch", "--l7-start-epoch", "--tau-hold-frac",
})
_EXPECTED_CHANGED = frozenset({
    "--tau-anneal-shape", "--lane-band-start-epoch", "--seg-chroma-boundary-start-epoch",
    # (seal v7 r1 R-1; operator APPROVED 2026-07-08) the Arm-A basis lever
    # DirectionalBasisRebalance(lane_offloaded) overrides v6's starved basis: n_dir_freqs 2->4,
    # freq_along 4->6 (freq_across 32 re-emitted as float). --self-orient is already True in v6 =>
    # NOT changed. These are CHANGES (existing v6 flags), not additions.
    "--n-dir-freqs", "--freq-along", "--freq-across",
})


@pytest.fixture(scope="module")
def compiled():
    return wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000)


@pytest.fixture(scope="module")
def trainer_text():
    from pathlib import Path
    return Path(TRAINER_REL).read_text()


# ── (a) config construction: it IS a requirement-V-native TypedWitnessConfig ─────────
def test_derive_returns_typed_witness_config():
    typed = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    assert isinstance(typed, TypedWitnessConfig)
    assert typed.name == "crucible_v7"
    assert typed.epochs == 3000 and typed.num_pairs == 600


def test_typed_config_validates_clean():
    typed = wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    assert typed.validate_program() == [], typed.validate_program()


def test_compile_is_deterministic(compiled):
    again = wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000)
    assert compiled.argv == again.argv


def test_argv_is_dsl_emitted_via_witness_program(compiled):
    """The emitter is the DSL WitnessProgram (requirement V), not a hand argv."""
    argv2 = compiled.typed.to_program().compile_trainer_argv()
    assert list(compiled.argv) == argv2


def test_num_pairs_and_epochs_emitted(compiled):
    fd = {f: v for f, v in compiled.emitted_pairs}
    assert fd.get("--num-pairs") == "600"
    assert fd.get("--epochs") == "3000"


# ── (b) the schedule-provenance gate: 0 NAKED (the whole point of the restart) ───────
def test_schedule_provenance_gate_zero_naked(compiled, trainer_text):
    registry = gate.schedule_when_flags(trainer_text)
    ereg = gate.event_start_flags(trainer_text)
    manifest_keys = set(compiled.constants_manifest.keys())
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=registry,
        manifest_keys=manifest_keys, governance=compiled.schedule_governance,
        event_registry=ereg)
    ok, violations, table = gate.gate_report(verdicts)
    assert ok, f"NAKED triggers remain:\n{table}"
    assert violations == []


def test_gate_classifies_all_three_starts_as_cap(compiled, trainer_text):
    registry = gate.schedule_when_flags(trainer_text)
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=registry,
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=compiled.schedule_governance,
        event_registry=gate.event_start_flags(trainer_text))
    by_flag = {v.flag: v for v in verdicts}
    for flag in ("--muon-start-epoch", "--lane-band-start-epoch",
                 "--seg-chroma-boundary-start-epoch"):
        assert flag in by_flag, f"{flag} should be a gated positive-epoch trigger"
        assert by_flag[flag].cls == gate.CLASS_CAP, by_flag[flag]


def test_gate_classifies_all_three_events_as_event_triggered(compiled, trainer_text):
    """operator override 2026-07-08: the three co-emitted --*-start-event wirings classify
    EVENT_TRIGGERED (the transition FIRES on its sensor; the paired epoch is a FAIL_SAFE_CAP backstop)."""
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=gate.schedule_when_flags(trainer_text),
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=compiled.schedule_governance,
        event_registry=gate.event_start_flags(trainer_text))
    by_flag = {v.flag: v for v in verdicts}
    for flag in ("--muon-start-event", "--lane-band-start-event",
                 "--seg-chroma-boundary-start-event"):
        assert flag in by_flag, f"{flag} should be a classified event wiring"
        assert by_flag[flag].cls == gate.CLASS_EVENT, by_flag[flag]


def test_governance_caps_declare_role_backstops_events_role_fires(compiled):
    """S4 R1: every CAP declares role=backstops (un-misreadable as a firing claim); every EVENT
    declares role=fires. The paired cap's `sensor` names the event it backs up."""
    gov = compiled.schedule_governance
    for cap, ev in (("--muon-start-epoch", "--muon-start-event"),
                    ("--lane-band-start-epoch", "--lane-band-start-event"),
                    ("--seg-chroma-boundary-start-epoch", "--seg-chroma-boundary-start-event")):
        assert gov[cap]["class"] == "cap" and gov[cap]["role"] == "backstops", gov[cap]
        assert gov[cap]["sensor"] == ev, gov[cap]
        assert gov[ev]["class"] == "event" and gov[ev]["role"] == "fires", gov[ev]


def test_cap_epoch_values_are_the_designed_caps(compiled):
    fd = {f: v for f, v in compiled.emitted_pairs}
    assert fd["--muon-start-epoch"] == str(wac._CRUCIBLE_V7_MUON_CAP)
    assert fd["--lane-band-start-epoch"] == str(wac._CRUCIBLE_V7_LANE_BAND_CAP)
    assert fd["--seg-chroma-boundary-start-epoch"] == str(wac._CRUCIBLE_V7_CHROMA_CAP)


def test_governance_sensors_are_recognised_and_co_emitted(compiled):
    emitted = {f for f, _ in compiled.emitted_pairs}
    for flag, entry in compiled.schedule_governance.items():
        sensor = entry["sensor"]
        assert sensor in gate.RECOGNISED_EVENT_SENSORS, (flag, sensor)
        assert sensor in emitted, f"declared sensor {sensor} for {flag} must be co-emitted"


# ── (c) the deletions: mutual exclusion + the removed flags ──────────────────────────
def test_tau_softplus_start_epoch_absent(compiled):
    assert "--tau-softplus-start-epoch" not in {f for f, _ in compiled.emitted_pairs}


def test_l7_and_tau_hold_frac_absent(compiled):
    emitted = {f for f, _ in compiled.emitted_pairs}
    assert "--l7-start-epoch" not in emitted
    assert "--tau-hold-frac" not in emitted


def test_seg_form_unify_tau_present(compiled):
    assert "--seg-form-unify-tau" in {f for f, _ in compiled.emitted_pairs}


def test_tau_anneal_shape_is_geometric(compiled):
    fd = {f: v for f, v in compiled.emitted_pairs}
    assert fd["--tau-anneal-shape"] == "geometric"


def test_mutual_exclusion_holds_at_trainer_validation(compiled):
    """The trainer's own mutual-exclusion guard passes for the v7 argv (no explicit
    --tau-softplus-start-epoch riding --seg-form-unify-tau)."""
    from experiments.train_levelset_witness_realized_through_R_mlx import (
        validate_seg_form_unify_tau_config,
    )
    # argv WITHOUT the excluded flag => OK
    validate_seg_form_unify_tau_config(seg_form_unify_tau=True, argv=list(compiled.argv))
    # sanity: the guard DOES fire when the excluded flag is present (default-OFF sibling contract)
    with pytest.raises(ValueError, match="tau-softplus-start-epoch"):
        validate_seg_form_unify_tau_config(
            seg_form_unify_tau=True, argv=["--tau-softplus-start-epoch", "300"])


# ── (d) pose block VERBATIM vs v6 ────────────────────────────────────────────────────
def test_pose_block_verbatim_vs_v6(compiled):
    v6 = {f: v for f, v in compiled.v6_flags}
    v7 = {f: v for f, v in compiled.emitted_pairs}

    def _norm(x):
        return True if x is None else str(x)

    for pf in ("--w-pose", "--pose-carrier", "--pose-carrier-source",
               "--pose-carrier-residual-mode"):
        assert _norm(v6.get(pf)) == _norm(v7.get(pf)), pf
    assert v7["--pose-carrier-source"] == "generated"
    assert v7["--pose-carrier-residual-mode"] == "table"
    assert v7["--w-pose"] == "1.0"


# ── (e) the DSL-provenance manifest ──────────────────────────────────────────────────
def test_dsl_program_manifest_shape(compiled):
    man = compiled.dsl_program_manifest
    assert man["schema"] == PROGRAM_MANIFEST_SCHEMA
    assert man["program_name"] == "crucible_v7"
    assert man["typed_validated"] is True
    assert man["flag_names"] and man["flag_fingerprint"]


def test_manifest_fingerprint_matches_emitted_argv(compiled):
    emitted_names = sorted({f for f, _ in compiled.emitted_pairs})
    ok, detail = verify_launch_manifest(compiled.dsl_program_manifest, emitted_names)
    assert ok, detail


def test_manifest_fails_on_flag_drift(compiled):
    emitted_names = sorted({f for f, _ in compiled.emitted_pairs}) + ["--totally-new-flag"]
    ok, detail = verify_launch_manifest(compiled.dsl_program_manifest, emitted_names)
    assert not ok and "disagrees" in detail


def test_manifest_absent_refused(compiled):
    emitted_names = sorted({f for f, _ in compiled.emitted_pairs})
    ok, detail = verify_launch_manifest({}, emitted_names)
    assert not ok and "no DSL program manifest" in detail


# ── (f) the diff-vs-v6 table STABILITY (the council review surface) ──────────────────
def test_diff_table_added_removed_changed_exact(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    added = {f for f, _ in diff["added"]}
    removed = {f for f, _ in diff["removed"]}
    changed = {f for f, _, _ in diff["changed"]}
    assert added == _EXPECTED_ADDED, added ^ _EXPECTED_ADDED
    assert removed == _EXPECTED_REMOVED, removed ^ _EXPECTED_REMOVED
    assert changed == _EXPECTED_CHANGED, changed ^ _EXPECTED_CHANGED


def test_diff_changed_values_are_the_designed_deltas(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    changed = {f: (a, b) for f, a, b in diff["changed"]}
    assert changed["--tau-anneal-shape"] == ("cosine_hold", "geometric")
    assert str(changed["--lane-band-start-epoch"][0]) == "350"
    assert str(changed["--lane-band-start-epoch"][1]) == "500"
    assert str(changed["--seg-chroma-boundary-start-epoch"][0]) == "300"
    assert str(changed["--seg-chroma-boundary-start-epoch"][1]) == "450"
    # (seal v7 r1 R-1) the Arm-A basis lever's designed deltas: n_dir_freqs 2->4, freq_along 4->6
    # (freq_across value-identical 32, re-emitted as float by the lever).
    assert str(changed["--n-dir-freqs"][0]) == "2"
    assert str(changed["--n-dir-freqs"][1]) == "4"
    assert str(changed["--freq-along"][0]) == "4"
    assert str(changed["--freq-along"][1]) == "6.0"
    assert float(changed["--freq-across"][0]) == float(changed["--freq-across"][1]) == 32.0


def test_diff_excludes_run_dir_placeholder(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    changed = {f for f, _, _ in diff["changed"]}
    assert "--out-dir" not in changed and "--gt-cache" not in changed


def test_diff_flag_counts(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    # v7 = v6 - 3 removed + |added| (out-dir/gt-cache excluded from both sides symmetrically). The
    # added set = the 3 spine/lever families + the 3 SENSOR->START WIRING flags + the DSL VerdictCadence
    # emitter delta = exactly len(_EXPECTED_ADDED).
    assert diff["v7_flag_count"] == diff["v6_flag_count"] - 3 + len(_EXPECTED_ADDED)


# ── (g) unchanged-flag byte-identity (v6 sealed values carry over) ───────────────────
def test_unchanged_flags_are_byte_identical_to_v6(compiled):
    """Every flag NOT in the designed delta set emits the SAME token as v6 (the 'v6 sealed
    values carry over unchanged' law)."""
    delta = _EXPECTED_ADDED | _EXPECTED_REMOVED | _EXPECTED_CHANGED | {"--out-dir", "--gt-cache"}
    v6 = {f: (True if v is None else str(v)) for f, v in compiled.v6_flags}
    v7 = {f: (True if v is None else str(v)) for f, v in compiled.emitted_pairs}
    for flag in set(v6) & set(v7):
        if flag in delta:
            continue
        assert v6[flag] == v7[flag], f"{flag}: v6={v6[flag]!r} v7={v7[flag]!r}"


# ── (h) wiring-gap honesty list (council input, not a failure) ───────────────────────
def test_wiring_status_all_three_wired():
    """operator override 2026-07-08: the three OWED sensor->start wirings are now BUILT — each entry
    reports WIRED with its sensor + the fail-safe backstop it demotes the fixed epoch to."""
    gaps = wac.crucible_v7_wiring_gaps()
    assert len(gaps) == 3
    joined = " ".join(gaps).lower()
    assert "powerlaw_meat" in joined and "annulus" in joined and "nucleus" in joined
    assert joined.count("wired") == 3 and "backstop" in joined


# ── (i) Arm-A basis lever (seal v7 r1 R-1; operator APPROVED 2026-07-08) ──────────────
def _argv_val(argv, flag):
    """The value token after ``flag`` in a compiled argv (None => bare flag / absent)."""
    for i, t in enumerate(argv):
        if t == flag:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            return None if (nxt is None or str(nxt).startswith("--")) else nxt
    return "<absent>"


def test_basis_lever_activation_in_compiled_argv(compiled):
    """The DirectionalBasisRebalance(lane_offloaded) lever fires into the compiled argv: the derived
    rebalance (n_dir_freqs 4, freq_along 6, freq_across 32) single-emitted (no double-emit vs base)."""
    argv = list(compiled.argv)
    assert _argv_val(argv, "--n-dir-freqs") == "4"
    assert _argv_val(argv, "--freq-along") == "6.0"
    assert float(_argv_val(argv, "--freq-across")) == 32.0
    assert _argv_val(argv, "--self-orient") is None  # bare (already True in base; lever no-op override)
    # single-emit (the DSL base+lever dict merge dedupes; later lever wins)
    for flag in ("--n-dir-freqs", "--freq-along", "--freq-across", "--self-orient"):
        assert argv.count(flag) == 1, f"{flag} double-emitted"


def test_basis_lever_in_dsl_levers_activation_surface(compiled):
    """The lever is on the activation-ledger surface (CrucibleV7LaunchConfig.dsl_levers) — so a real
    launch records its FIRST 'fired' event (never-fired Arm-A finally fires). Names are Lever.name,
    matching _CRUCIBLE_V7_DSL_LEVERS."""
    levers = compiled.to_launch_config().dsl_levers
    assert "FEED_07a_directional_basis_rebalance" in levers
    assert len(levers) == 4  # the 3 v6-inherited spine levers + the Arm-A basis lever
    assert tuple(levers) == wac._CRUCIBLE_V7_DSL_LEVERS


def test_basis_lever_matches_derived_equation_law():
    """The lever's freq_along is the equations-leg derivation (lane_offloaded = Candes-Donoho
    parabolic sqrt(across)), NOT a hand number — the triality equations<->DSL agreement."""
    from tac.canonical_equations.anisotropic_basis_two_regime_allocation_20260707 import (
        freq_along_for_regime,
    )
    assert freq_along_for_regime(32, "lane_offloaded") == 6  # max(4, round(sqrt(32)))


def test_basis_lever_freq_across_value_identical():
    """freq_across is VALUE-unchanged (32); only re-emitted as float by the lever — a type
    normalization, not a semantic knob change."""
    diff = wac.diff_crucible_v6_to_v7(
        wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000).v6_flags,
        wac.compile_crucible_v7_config(_GT, num_pairs=600, epochs=3000).emitted_pairs)
    changed = {f: (a, b) for f, a, b in diff["changed"]}
    assert float(changed["--freq-across"][0]) == float(changed["--freq-across"][1])


# ── (j) memory WATERFILL: the basis lever fits the envelope (the gating math) ─────────
def test_basis_allocation_provenance_shape():
    """The waterfill-provenance accessor exposes the derivation the memo records (no bare numbers;
    every value re-derived from the REAL preflight projection)."""
    prov = wac.crucible_v7_basis_allocation_provenance()
    assert prov["regime"] == "lane_offloaded"
    assert prov["chosen_lever"] == "FEED_07a_directional_basis_rebalance"
    assert prov["freq_along"] == 6 and prov["n_dir_freqs"] == 4
    assert prov["in_feat_delta"] == 8  # dir_w = 4*(4-2)


def test_basis_waterfill_projection_matches_preflight():
    """The provenance peak/cf deltas equal the REAL preflight projection at in_feat 88 vs 96 (the
    gating math is re-derived from tools/witness_memory_preflight, never asserted)."""
    import sys
    from pathlib import Path
    tp = Path(wac.__file__).resolve().parents[2] / "tools"
    if str(tp) not in sys.path:
        sys.path.insert(0, str(tp))
    import witness_memory_preflight as wmp
    base = wmp.project_peak_rss_gib(num_pairs=600, in_feat=88, self_orient=True,
                                    verdict_batch=32, render_aa="ipe", total_ram_gib=128.0,
                                    safe_frac=0.70)
    lever = wmp.project_peak_rss_gib(num_pairs=600, in_feat=96, self_orient=True,
                                     verdict_batch=32, render_aa="ipe", total_ram_gib=128.0,
                                     safe_frac=0.70)
    prov = wac.crucible_v7_basis_allocation_provenance()
    assert prov["peak_delta_gib"] == round(lever.projected_peak_gib - base.projected_peak_gib, 2)
    assert prov["cf_cache_delta_gib"] == round(lever.cf_cache_gib - base.cf_cache_gib, 2)
    # the lever grows cf_mx_cache (feat_ratio 96/88) => a real, positive, bounded memory delta
    assert 3.0 < prov["peak_delta_gib"] < 5.0


def test_basis_lever_admitted_by_both_envelopes():
    """ENVELOPE ADMISSION: the lane_offloaded allocation's projected peak fits BOTH the conservative
    0.70 concurrent envelope AND the 0.85 sole-workload envelope with positive margin — so the
    DERIVED DSL lever is preferred as-designed (no fall-through to a minimal along-only rebalance)."""
    prov = wac.crucible_v7_basis_allocation_provenance()
    assert prov["admitted_0p70"] is True and prov["admitted_0p85"] is True
    assert prov["margin_0p70_gib"] > 0 and prov["margin_0p85_gib"] > 0
    # the peak sits well under even the conservative ceiling (never-crash physics leg 1 holds)
    lever_peak = prov["candidates"]["dsl_lever_lane_offloaded"]["peak_gib"]
    assert lever_peak < prov["envelope_0p70_gib"]


def test_minimal_along_rebalance_is_memory_neutral():
    """A minimal along-only rebalance (freq_along 4->6/8 with n_dir_freqs held at 2) is
    MEMORY-NEUTRAL — freq_along VALUE does not enter in_feat (trainer dir_w = 4*n_dir_freqs). This is
    why the waterfill's only memory-relevant candidate is the n_dir_freqs bump."""
    import sys
    from pathlib import Path
    tp = Path(wac.__file__).resolve().parents[2] / "tools"
    if str(tp) not in sys.path:
        sys.path.insert(0, str(tp))
    import witness_memory_preflight as wmp
    # n_dir_freqs held at 2 => in_feat identical regardless of freq_along value.
    f_base = {"self_orient": True, "n_dir_freqs": 2, "max_bank_freq": 64.0}
    assert wmp.derive_in_feat_from_flags(f_base) == 88
    f_dsl = dict(f_base); f_dsl["n_dir_freqs"] = 4
    assert wmp.derive_in_feat_from_flags(f_dsl) == 96


def test_zero_naked_preserved_after_basis_lever(compiled, trainer_text):
    """The basis lever changes only basis flags (none schedule-triggered), so the schedule-provenance
    gate STILL reports 0 NAKED after the lever lands (regression guard for the R-1 config change)."""
    registry = gate.schedule_when_flags(trainer_text)
    ereg = gate.event_start_flags(trainer_text)
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=registry,
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=compiled.schedule_governance, event_registry=ereg)
    ok, violations, _ = gate.gate_report(verdicts)
    assert ok and violations == []
