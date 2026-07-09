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
    # (v7.3 delta 2, synthesis item 8 IN-v7) the R-7 Polyak/Ruppert tail averager (extra ckpt candidate
    # alongside EMA); start-epoch DERIVED-AT-CONFIG from the TAIL turnpike (constants_manifest LawRef).
    "--polyak-finisher-arm", "--polyak-finisher-start-epoch",
    # (v7.3 delta 3, synthesis item 4 ELEVATED by GPU cert) arm the safe-compile hosc region; the
    # launcher b2 gate fingerprint-verifies at admission (device-conditional; GPU ADMIT, CPU REFUSE).
    "--safe-compile-regions",
    # (v7.3 round-2 R3 fix, seal_v73_r2_structure) per-group grad-clip ON — bounds the ep1 gnorm_hijack
    # (island_amplify ~20% of ep1 loss) so it can't starve the seg gradient during the Road-forming window.
    "--per-group-grad-clip",
    # (v7.5 birth-counter-force, road_anomaly_probe_20260708.md) Lever-1 CHAN-VESE AREA CONSTRAINT (the
    # precision counter-force vs the recall-only birth stack's Road over-paint) + Lever-2 MORSE-SMALE
    # BIRTH-COMPLETION EVENT (the birth->boundary regime hand-off). All new flags (absent from v6).
    "--area-constraint-birth", "--area-constraint-birth-force", "--area-constraint-tolerance",
    "--area-constraint-classes",
    "--birth-completion-event", "--birth-completion-tau-persist", "--birth-completion-area-band",
    "--birth-completion-ramp-epochs", "--birth-completion-post-level", "--birth-completion-classes",
    # (v7.5 RAMP-LANDED, memo §RAMP-LANDED) the Lever-2 ramp is now APPLIED to the birth-loss surfaces
    # (per-class island-amplify / persistence-recall / logit-adjust offset); post_level is DERIVED from
    # tau_persist (= 1 - 0.8 = 0.2). --birth-completion-ramp is the new (absent-from-v6) apply switch.
    "--birth-completion-ramp",
    # (v7.5 Lever-3 regime coherence) restrict the v6-inherited --logit-adjust-loss-tau boost to the
    # non-offloaded birthed class(es); DERIVED from the basis regime (lane_offloaded => "3", movable only).
    "--logit-adjust-classes",
    # (v7.5 ACTUATION item A.2, v75_optimal_form_actuation_spec_20260708.md §A.2) #287 ego-phase DASH COMB
    # DSL lever — the along-tangent 3.2x-deficit corrector; supplies the ~25-cyc dash structure the
    # lane_offloaded cartoon band cannot represent, rule-118 FREE at decode. Composes with --lane-render-band.
    "--lane-band-dash-comb", "--lane-band-comb-softness-m",
    # (v7.5 ACTUATION item B.4, spec §B.4 / P0 FORCE 1 #360) TEMPORAL SCREW-CONSISTENCY DSL lever — the
    # ~50x Undriv-jitter lever (GROUND-class annulus prob-warp MSE; kills the 44% lane-dominated flicker
    # residual). All 6 flags absent from v6. Its --seg-temporal-screw-start-event annulus_plateau
    # EVENT-governs the start (unify-τ replacement for the dissolved-l7 formed-partition gate);
    # --seg-temporal-screw-start-epoch is the fail-safe backstop cap.
    "--seg-temporal-screw-weight", "--seg-temporal-screw-start-epoch", "--seg-temporal-screw-start-event",
    "--seg-temporal-screw-xi-source", "--seg-temporal-screw-classes", "--seg-temporal-screw-band",
    # (v7.5 ACTUATION item D.9, spec §D.9 / FEED-238resolved) TERMINAL POSE-FINISH TypedStage — the R1
    # two-phase (pose-blind until d_seg converges at the muon switch, then terminal joint pose-descent;
    # SUPERSEDES co-train-pose-from-ep0). --pose-finish-start-epoch (absent from v6) is CAP-governed
    # backstopping --muon-start-event (the pose-finish co-fires with the d_seg-converged muon switch).
    "--pose-finish-start-epoch",
})
_EXPECTED_REMOVED = frozenset({
    "--tau-softplus-start-epoch", "--l7-start-epoch", "--tau-hold-frac",
    # (v7.5 ACTUATION item A.1, spec §A.1) drop the DEAD --structured-init-include-lane (lane_px=0 => inert
    # no-op). The lane nucleation now rides paint-then-SDF (--lane-prior-phi1-mode paint, a CHANGED delta).
    "--structured-init-include-lane",
})
_EXPECTED_CHANGED = frozenset({
    "--tau-anneal-shape", "--lane-band-start-epoch", "--seg-chroma-boundary-start-epoch",
    # (seal v7 r1 R-1; operator APPROVED 2026-07-08) the Arm-A basis lever
    # DirectionalBasisRebalance(lane_offloaded) overrides v6's starved basis: n_dir_freqs 2->4,
    # freq_along 4->6 (freq_across 32 re-emitted as float). --self-orient is already True in v6 =>
    # NOT changed. These are CHANGES (existing v6 flags), not additions.
    "--n-dir-freqs", "--freq-along", "--freq-across",
    # (v7.3 round-2 BLOCKER fix, seal_v73_r2_deepmath) event-mode hosc_beta_end 10.0 -> 3.177 (the
    # control's frozen β(726); the clock-endpoint 10.0 would FREEZE β≈10 under the event octave driver).
    "--hosc-beta-end",
    # (v7.3 round-2 M1 fix, seal_v73_r2_structure) lane-regime coherence: persistence-recall classes
    # 'auto' (lane+movable) -> '3' (movable only; lane rides the analytic band under lane_offloaded).
    "--persistence-classes",
    # (v7.5 ACTUATION item A.1, spec §A.1) lane nucleation fix: --lane-prior-phi1-mode replace (MEASURED
    # NO-OP, #291) -> paint (paint-then-SDF; MEASURED lane FN 0.00713->0.00211 ~3x on real GT).
    "--lane-prior-phi1-mode",
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


# ── (b.4) v7.5 ACTUATION item B.4 — temporal-screw EVENT-governed on annulus_plateau ──────────
def test_b4_temporal_screw_composed_and_active(compiled):
    """v7.5 B.4: the P0 FORCE 1 temporal-screw lever is composed with a POSITIVE cold-start weight (0.1)
    and the confound-SAFE ground_gt xi source over the GROUND classes (0,1,2)."""
    fd = {f: v for f, v in compiled.emitted_pairs}
    assert "temporal_screw_consistency" in compiled.to_launch_config().dsl_levers
    assert float(fd["--seg-temporal-screw-weight"]) == 0.1
    assert fd["--seg-temporal-screw-xi-source"] == "ground_gt"    # ZERO pose coupling (L68)
    assert fd["--seg-temporal-screw-classes"] == "0,1,2"          # GROUND only (homography wrong off-ground)


def test_b4_temporal_screw_start_is_event_governed_not_naked(compiled, trainer_text):
    """THE DESIGN DECISION (operator B.4): the start is EVENT-governed on the annulus_plateau formed-
    boundary sensor (the unify-τ replacement for the dissolved-l7 'formed partition' gate), and the
    --seg-temporal-screw-start-epoch is the fail-safe BACKSTOP CAP — NOT a naked positive epoch."""
    gov = compiled.schedule_governance
    fd = {f: v for f, v in compiled.emitted_pairs}
    # the EVENT wiring is co-emitted + declared role=fires on the shared annulus_plateau sensor
    assert fd["--seg-temporal-screw-start-event"] == "annulus_plateau"
    assert gov["--seg-temporal-screw-start-event"]["class"] == "event"
    assert gov["--seg-temporal-screw-start-event"]["role"] == "fires"
    # the cap is the SAME formed-boundary value as chroma (both are formed-boundary annulus levers)
    assert gov["--seg-temporal-screw-start-epoch"]["class"] == "cap"
    assert gov["--seg-temporal-screw-start-epoch"]["role"] == "backstops"
    assert gov["--seg-temporal-screw-start-epoch"]["sensor"] == "--seg-temporal-screw-start-event"
    assert fd["--seg-temporal-screw-start-epoch"] == str(wac._CRUCIBLE_V7_TEMPORAL_SCREW_CAP)
    assert wac._CRUCIBLE_V7_TEMPORAL_SCREW_CAP == wac._CRUCIBLE_V7_CHROMA_CAP
    # the gate classifies the event EVENT_TRIGGERED + the epoch FAIL_SAFE_CAP (not NAKED)
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=gate.schedule_when_flags(trainer_text),
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=gov, event_registry=gate.event_start_flags(trainer_text))
    by_flag = {v.flag: v for v in verdicts}
    assert by_flag["--seg-temporal-screw-start-event"].cls == gate.CLASS_EVENT
    assert by_flag["--seg-temporal-screw-start-epoch"].cls == gate.CLASS_CAP


# ── (d.9) v7.5 ACTUATION item D.9 — terminal pose-finish TypedStage (R1 two-phase) ──────────────
def test_d9_pose_finish_stage_gates_pose_terminal(compiled):
    """v7.5 D.9: the terminal POSE-FINISH TypedStage gates the pose term to AFTER d_seg converges (the R1
    two-phase), superseding co-train-pose-from-ep0. --w-pose stays the finish weight (1.0); the new
    --pose-finish-start-epoch (== the muon cap) makes it terminal, co-firing with the muon switch."""
    fd = {f: v for f, v in compiled.emitted_pairs}
    assert fd["--pose-finish-start-epoch"] == str(wac._CRUCIBLE_V7_MUON_CAP)  # co-fires with the muon switch
    assert float(fd["--w-pose"]) == 1.0                                       # the finish-phase weight (carrier ON)
    # the pose_finish stage is a real TypedStage on the config (sister of the muon stage)
    stage_names = {s.name for s in compiled.typed.stages}
    assert "pose_finish" in stage_names and "muon" in stage_names


def test_d9_pose_finish_start_epoch_cap_governed_by_muon_event(compiled, trainer_text):
    """THE governance: --pose-finish-start-epoch is a FAIL_SAFE_CAP backstopping the muon event (the
    pose-finish co-fires with the d_seg-converged muon switch — NOT a naked positive epoch)."""
    gov = compiled.schedule_governance
    assert gov["--pose-finish-start-epoch"]["class"] == "cap"
    assert gov["--pose-finish-start-epoch"]["role"] == "backstops"
    assert gov["--pose-finish-start-epoch"]["sensor"] == "--muon-start-event"
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=gate.schedule_when_flags(trainer_text),
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=gov, event_registry=gate.event_start_flags(trainer_text))
    by_flag = {v.flag: v for v in verdicts}
    assert by_flag["--pose-finish-start-epoch"].cls == gate.CLASS_CAP


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
    # (v7.3 round-2 BLOCKER fix) event-mode hosc_beta_end: 10.0 (clock endpoint) -> 3.177 (control β(726))
    assert float(changed["--hosc-beta-end"][0]) == 10.0
    assert float(changed["--hosc-beta-end"][1]) == 3.177
    # (v7.3 round-2 M1 fix) lane-regime coherence: persistence-recall 'auto' -> '3' (movable only)
    assert str(changed["--persistence-classes"][0]) == "auto"
    assert str(changed["--persistence-classes"][1]) == "3"


def test_diff_excludes_run_dir_placeholder(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    changed = {f for f, _, _ in diff["changed"]}
    assert "--out-dir" not in changed and "--gt-cache" not in changed


def test_diff_flag_counts(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    # v7 = v6 - |removed| + |added| (out-dir/gt-cache excluded from both sides symmetrically). The
    # added set = the 3 spine/lever families + the 3 SENSOR->START WIRING flags + the DSL VerdictCadence
    # emitter delta + the v7.5 ACTUATION dash-comb pair; removed = the 3 deleted schedule flags + the
    # v7.5 ACTUATION dead --structured-init-include-lane. Derive both from the designed fixtures.
    assert diff["v7_flag_count"] == diff["v6_flag_count"] - len(_EXPECTED_REMOVED) + len(_EXPECTED_ADDED)


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
    assert "R7_polyak_finisher" in levers
    assert "v75_area_constraint_birth" in levers and "v75_birth_completion_event" in levers
    assert "n287_dash_comb" in levers  # v7.5 ACTUATION item A.2 (along-tangent dash-comb corrector)
    assert "temporal_screw_consistency" in levers  # v7.5 ACTUATION item B.4 (P0 FORCE 1 temporal-screw)
    assert len(levers) == 9  # v7.3 five + two v7.5 birth-counter-force + A.2 dash-comb + B.4 temporal-screw
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


# ── (i2) SEAL v7.3 round-2 A1: event-mode hosc β endpoint (BLOCKER fix) ────────────────
def test_hosc_beta_end_is_event_frozen_value_not_clock_endpoint(compiled):
    """The v7 (EVENT mode) hosc_beta_end is the control's FROZEN β(726)≈3.177, NOT the inherited
    clock-frame endpoint 10.0 (which the octave-fraction driver would FREEZE β≈10 = forbidden
    saturation). ≤ 4.0 honors the anneal-β divergence bound."""
    argv = list(compiled.argv)
    hbe = _argv_val(argv, "--hosc-beta-end")
    assert hbe is not None and float(hbe) == wac._CRUCIBLE_V7_HOSC_BETA_END_EVENT == 3.177
    assert float(hbe) <= 4.0  # inside the anneal-β divergence bound (never a fixed high β)
    # the config is event-mode (the driver that makes the clock endpoint a frozen-β hazard)
    assert dict(compiled.emitted_pairs).get("--tau-advance-mode") == "event"


# ── (i3) SEAL v7.3 round-2 M1: lane-regime coherence (persistence-recall class gate) ───
def test_persistence_classes_derived_from_basis_regime():
    """The persistence-recall class targeting is DERIVED from the basis regime (the M1 coherence law),
    not a hand literal: lane_offloaded -> movable only ('3'); lane_carried -> 'auto' (keep lane)."""
    from tac.witness_dsl.curriculum_dsl import persistence_classes_for_basis_regime
    assert persistence_classes_for_basis_regime("lane_offloaded") == "3"
    assert persistence_classes_for_basis_regime("lane_carried") == "auto"
    with pytest.raises(ValueError):
        persistence_classes_for_basis_regime("nonsense_regime")


def test_persistence_classes_coherent_with_basis_regime_in_argv(compiled):
    """v7 commits to lane_offloaded (the emitted basis freq_along=6), so the emitted persistence-recall
    classes must EXCLUDE lane ('3' = movable only) — the M1 basis/loss coherence, verified in the argv."""
    d = dict(compiled.emitted_pairs)
    assert d.get("--persistence-classes") == "3"        # movable only (lane rides the analytic band)
    assert str(d.get("--freq-along")) == "6.0"          # the lane_offloaded (cartoon-scale) basis
    assert wac._CRUCIBLE_V7_BASIS_REGIME == "lane_offloaded"  # single-source regime


def test_f3_lane_offloaded_structurally_co_emits_the_analytic_band(compiled):
    """F-3 structural coupling (positive): under lane_offloaded, lane is dropped from the learned
    persistence recall ('3' = movable only), so the FREE analytic lane band MUST carry lane — assert
    the band flag is present in the EMITTED argv, not merely assumed at byte-close."""
    assert "--lane-render-band" in list(compiled.argv)
    assert dict(compiled.emitted_pairs).get("--persistence-classes") == "3"  # lane excluded from recall


def test_f3_guard_fails_loud_at_compile_if_band_absent(monkeypatch):
    """F-3 structural coupling (negative): if a future proven base dropped --lane-render-band while the
    regime stays lane_offloaded, the compile MUST fail LOUD (not silently starve lane at byte-close).
    Simulate by wrapping derive_crucible_v6_config so its trainer flags omit the band."""
    real = wac.derive_crucible_v6_config

    def _no_band(*a, **k):
        cfg = real(*a, **k)

        class _Wrap:
            def __getattr__(self, n):
                return getattr(cfg, n)

            def to_trainer_flags(self, out):
                return [(f, v) for (f, v) in cfg.to_trainer_flags(out) if f != "--lane-render-band"]

        return _Wrap()

    monkeypatch.setattr(wac, "derive_crucible_v6_config", _no_band)
    with pytest.raises(ValueError, match="lane-regime coherence gate"):
        wac.derive_crucible_v7_config(_GT, num_pairs=600, epochs=3000)


# ── (i4) SEAL v7.3 round-2 R3: per-group grad-clip ON (gnorm-hijack seg-starvation guard) ─
def test_per_group_grad_clip_present_in_v7_argv(compiled):
    """--per-group-grad-clip is ON in the v7 emitted argv (bounds the ep1 gnorm_hijack so a large early
    island/eikonal term cannot starve the seg gradient during the Road-forming window)."""
    argv = list(compiled.argv)
    assert "--per-group-grad-clip" in argv
    assert "--no-per-group-grad-clip" not in argv  # not the negated BooleanOptionalAction form
    assert float(dict(compiled.emitted_pairs).get("--grad-clip")) > 0.0  # per-group clip needs grad-clip>0


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


# ── (k) v7.3 delta 2: the Polyak tail finisher (start_epoch DERIVED, not the default-0 footgun) ──
def test_polyak_lever_armed_with_derived_start_epoch(compiled):
    """The R-7 Polyak finisher fires into the argv with the DERIVED-AT-CONFIG start-epoch (NEVER the
    default 0 — that would average the whole run, not the finishing tail)."""
    argv = list(compiled.argv)
    assert _argv_val(argv, "--polyak-finisher-arm") is None  # bare store_true flag
    start = _argv_val(argv, "--polyak-finisher-start-epoch")
    assert start is not None and int(start) > 0
    # sized to the TAIL turnpike; averages EXACTLY window=round(0.2*finishing_window) epochs over the
    # trainer's INCLUSIVE [start, epochs] loop => start = epochs - window + 1 (v7.3 round-2 MINOR-2 fix).
    prov = wac.crucible_v7_polyak_start_provenance(3000)
    assert int(start) == int(prov["polyak_start_epoch"])
    # for the sealed schedule (epochs 3000, window 455) start = 3000 - 455 + 1 = 2546 (post-Muon, dwell)
    assert int(start) == 2546
    assert int(start) > wac._CRUCIBLE_V7_MUON_CAP  # strictly post-Muon (inside the constant-τ* dwell)


def test_polyak_start_provenance_is_derived_at_config():
    """The start-epoch is DERIVED from the finisher law, not a bare literal (value-provenance ladder)."""
    prov = wac.crucible_v7_polyak_start_provenance(3000)
    assert prov["ladder_class"] == "derived_at_config"
    assert prov["equation_id"] == "muon_finisher_schedule_warmstart_and_lr_anneal_v1"
    assert prov["finishing_stage_window_epochs"] == 3000 - wac._CRUCIBLE_V7_MUON_CAP
    # start = epochs - window + 1 = 2546; relative to muon_cap = 2546 - 726 = 1820 (fencepost fix)
    assert prov["polyak_relative_start_epoch"] == 1820


def test_polyak_non_degenerate_averages_exactly_window_epochs():
    """(v7.3 round-2 MINOR-2 off-by-one) the tail averages EXACTLY window epochs over the trainer's
    inclusive [start, epochs] loop — BEHAVIOR, not just the emitted constant."""
    prov = wac.crucible_v7_polyak_start_provenance(3000)
    start, window = int(prov["polyak_start_epoch"]), int(prov["polyak_window_epochs"])
    observed = sum(1 for ep in range(start, 3000 + 1))
    assert observed == window == 455  # inclusive final fencepost: no off-by-one


def test_polyak_sizing_degenerate_is_genuinely_inert_over_the_real_loop():
    """(v7.3 round-2 MINOR-1) RSS calibration reuses the REAL config name with a tiny --calibrate-epochs
    (default 3). The Polyak sizing degenerates to a GENUINELY INERT averager — NOT start_epoch=epochs
    (which observes ONCE at the final loop epoch), but start_epoch=epochs+1 so observe never fires.
    Asserts BEHAVIOR (count==0 over the real trainer loop), not the constant (tests-verify-behavior)."""
    import numpy as np
    from tac.witness_control.polyak_finisher import PolyakTailAverager
    prov = wac.crucible_v7_polyak_start_provenance(3)
    assert prov["degenerate"] is True
    start = int(prov["polyak_start_epoch"])
    assert start == 4  # epochs(3) + 1 => strictly beyond the final loop epoch
    # BEHAVIOR: arm the averager at the derived start, run the REAL trainer loop range(1, epochs+1),
    # and confirm it never observes => byte-identical to an unarmed run.
    avg = PolyakTailAverager(start_epoch=start, arm=True)
    params = {"w": np.zeros((2, 2), np.float32)}
    for ep in range(1, 3 + 1):  # the trainer loop reaches ep == epochs
        avg.observe(ep, params)
    assert avg.count == 0, "degenerate Polyak is NOT inert — it observed the final epoch"
    # a full v7 build at tiny epochs succeeds (the calibrate-rss path)
    c = wac.compile_crucible_v7_config(_GT, num_pairs=24, epochs=3)
    assert dict(c.emitted_pairs)["--polyak-finisher-start-epoch"] == "4"
    assert "polyak_finisher_start_epoch" in c.constants_manifest


def test_polyak_start_epoch_is_derived_in_manifest_not_naked(compiled, trainer_text):
    """The DERIVED start-epoch rides the constants_manifest as a LawRef (so the schedule-provenance
    gate classifies --polyak-finisher-start-epoch DERIVED, not NAKED_PRIMARY_EPOCH)."""
    assert "polyak_finisher_start_epoch" in compiled.constants_manifest
    entry = compiled.constants_manifest["polyak_finisher_start_epoch"]
    assert entry["ladder_class"] == "derived_at_config"
    assert entry["equation_id"] == "muon_finisher_schedule_warmstart_and_lr_anneal_v1"
    assert entry["value"] == 2546  # v7.3 round-2 MINOR-2 off-by-one fix (was 2545)
    # gate confirms DERIVED classification for the flag
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=gate.schedule_when_flags(trainer_text),
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=compiled.schedule_governance,
        event_registry=gate.event_start_flags(trainer_text))
    by_flag = {v.flag: v for v in verdicts}
    assert by_flag["--polyak-finisher-start-epoch"].cls == gate.CLASS_DERIVED


# ── (l) v7.3 delta 3: safe-compile hosc region armed (b2 gate is the runtime authority) ──
def test_safe_compile_hosc_region_armed(compiled):
    """--safe-compile-regions hosc_activation is armed in the argv; the launcher b2 gate
    fingerprint-verifies the manifest at admission (device-conditional GPU ADMIT / CPU REFUSE)."""
    argv = list(compiled.argv)
    assert _argv_val(argv, "--safe-compile-regions") == "hosc_activation"
    # the manifest path is NOT emitted (defaults to .omx/state/mlx_safe_compile_manifest.json)
    assert "--safe-compile-manifest" not in {f for f, _ in compiled.emitted_pairs}
    # the config is a gpu-device config (the ADMIT device; CPU device would REFUSE the hosc flip)
    assert dict(compiled.emitted_pairs).get("--mlx-device") == "gpu"


# ── (m) v7.3 delta 1: D16 persistence-pool dispatch ON in the perf-env prefix ──
def test_d16_persistence_pool_dispatch_on_in_perf_env(compiled):
    """The D16 fused persistence/clDice pool env is ON in the launch command's perf-env prefix
    (bit-identical by construction; the launcher's perf-env class guard then requires it)."""
    cmd = compiled.to_launch_config().to_command("OUT", perf_env=True)
    assert "TAC_MLX_CUSTOM_PERSISTENCE_POOL=1" in cmd
    assert cmd.startswith("TAC_MLX_CUSTOM_GROUPED_BACKWARD=1")  # unaffected — persistence pool follows


# ── (n) v7.3 delta 5: default-OFF items are a TRACKED duty-to-measure queue (not silent) ──
def test_registered_off_levers_carry_named_triggers():
    """Items 3/6/7/10/11 stay default-OFF but each carries a NAMED trigger (the 'off is a tracked
    queue' non-negotiable). Item 3's trigger is the bounded n600 d_seg A/B — NOT bit-identity
    engineering (that crux elevation was REVOKED by frozen_scorer_forward_batch_dependence_v1)."""
    off = wac.crucible_v7_registered_off_levers()
    assert set(off) == {"micro_batch", "verdict_reclaim_330", "adaptive_eps",
                        "gpu_verdict", "fp16_cf_feats",
                        # (v7.3 round-2) M1 counter-arm + M2 Road-first fallback registered duty-to-measure
                        "lane_carried_basis_regime", "road_boundary_fallback",
                        # (v7.5 B.5) horizon-weighted margin #169 + sky=rotation-only temporal-screw
                        # refinement — both BUILT, default-OFF, exit-criterion A/B owed
                        "horizon_weighted_margin_169", "temporal_screw_sky_rotation_only"}
    for item in off.values():
        assert item["default"] == "off"
        assert item["state"] == "registered_duty_to_measure"
        assert item["trigger"] and len(item["trigger"]) > 20
    # item 3 trigger reflects the falsification (A/B, not bit-identity)
    mb = off["micro_batch"]["trigger"].lower()
    assert "a/b" in mb and "bit-identity engineering" in mb
    # (v7.3 round-2 M1) the lane_carried counter-arm names the mutually-exclusive alternative + freq_along≈26
    lc = off["lane_carried_basis_regime"]["trigger"].lower()
    assert "lane_carried" in lc and "26" in lc
    # (v7.3 round-2 M2) the Road-fallback names Road as the primary run signal + the ep200 threshold
    rd = off["road_boundary_fallback"]["trigger"].lower()
    assert "road" in rd and "0.30" in rd and "ep200" in rd


# ── (o) v7.3 delta 4: the wall-clock budget re-derived from the LIVE 3.62 incl-startup cadence ──
def test_wall_clock_budget_rederived_from_amortized_cadence(compiled):
    """SEAL v7.4 round-3 DM-MINOR-1: the DERIVED budget uses the STARTUP-AMORTIZED 3000-ep cadence
    3.47 min/ep, re-fit on the WIDER ep25->125 window (was 3.39 on the narrow ep75->100 window that
    landed in a slow-adjacent-fast trough; the full-window MEASURED steady slope is 3.4537, so
    amortized(3000)=3.47), so at 3000 ep the ceiling is ~8.31 days (7.23 anchor projection x 1.15
    slack). Direction is strictly more conservative (ceiling tightens, budget grows ~0.2 d)."""
    from tac.local_acceleration.scorer_throughput_gate import RUN1_MEASURED_MIN_PER_EP
    assert RUN1_MEASURED_MIN_PER_EP == 3.47  # startup-amortized 3000-ep cadence, ep25->125 window (measured, not a bound)
    budget = compiled.to_launch_config().wall_clock_budget_days
    assert 8.0 <= float(budget.value) <= 8.5  # ~8.31 days at 3000 ep
