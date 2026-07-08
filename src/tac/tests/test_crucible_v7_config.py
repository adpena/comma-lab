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
})
_EXPECTED_REMOVED = frozenset({
    "--tau-softplus-start-epoch", "--l7-start-epoch", "--tau-hold-frac",
})
_EXPECTED_CHANGED = frozenset({
    "--tau-anneal-shape", "--lane-band-start-epoch", "--seg-chroma-boundary-start-epoch",
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
    manifest_keys = set(compiled.constants_manifest.keys())
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=registry,
        manifest_keys=manifest_keys, governance=compiled.schedule_governance)
    ok, violations, table = gate.gate_report(verdicts)
    assert ok, f"NAKED triggers remain:\n{table}"
    assert violations == []


def test_gate_classifies_all_three_starts_as_cap(compiled, trainer_text):
    registry = gate.schedule_when_flags(trainer_text)
    verdicts = gate.classify_launch(
        list(compiled.emitted_pairs), registry=registry,
        manifest_keys=set(compiled.constants_manifest.keys()),
        governance=compiled.schedule_governance)
    by_flag = {v.flag: v for v in verdicts}
    for flag in ("--muon-start-epoch", "--lane-band-start-epoch",
                 "--seg-chroma-boundary-start-epoch"):
        assert flag in by_flag, f"{flag} should be a gated positive-epoch trigger"
        assert by_flag[flag].cls == gate.CLASS_CAP, by_flag[flag]


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


def test_diff_excludes_run_dir_placeholder(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    changed = {f for f, _, _ in diff["changed"]}
    assert "--out-dir" not in changed and "--gt-cache" not in changed


def test_diff_flag_counts(compiled):
    diff = wac.diff_crucible_v6_to_v7(compiled.v6_flags, compiled.emitted_pairs)
    # v7 = v6 - 3 removed + 25 added (out-dir/gt-cache excluded from both sides symmetrically)
    assert diff["v7_flag_count"] == diff["v6_flag_count"] - 3 + 25


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
def test_wiring_gaps_enumerated():
    gaps = wac.crucible_v7_wiring_gaps()
    assert len(gaps) == 3
    joined = " ".join(gaps).lower()
    assert "powerlaw_meat" in joined and "annulus" in joined and "nucleus" in joined
