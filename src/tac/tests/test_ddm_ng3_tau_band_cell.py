"""ddm_ng3 -- the expected-flip tau band at the MEASURED R-noise scale.

Every test here defends one of three claims: the band is the LAW's output and never a literal,
both validators refuse anything but the two admissible bands, and the lever is EXACTLY the
temperature (nothing else leaks out of the ``tau_band`` block into the loss).
"""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_qbr1_born_fairform_burn_prep as qbr1  # noqa: E402
from experiments import ddm_qbt1_qbflow_trainer as qbt  # noqa: E402
from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (  # noqa: E402
    resolve_margin_band_threshold,
)
from tac.witness_dsl.curriculum_dsl import (  # noqa: E402
    QBR1_LEGACY_TAU_BAND,
    QBR1_TAU_BAND_MODES,
    ExpectedFlipTauBandMsafe,
    compile_qbr1_tau_band_config,
)

ng3 = pytest.importorskip("experiments.ddm_ng3_tau_band_cell")

#: the three historical delta_R / m_safe decimals.  None of them may appear as a LITERAL in any
#: source this arm owns -- they must be resolved through the law at runtime.  Same discipline as
#: ddm_gm1's ``test_module_carries_no_hardcoded_m_safe_literal``.
FORBIDDEN_LITERALS = (
    "0.021881818771362305",   # delta_R, n600 (ddm_dr1)
    "0.04376363754272461",    # m_safe = 2 * delta_R, n600
    "0.019590163230895963",   # delta_R, retired n96 prefix
    "0.039180326461791926",   # m_safe, retired n96 prefix
)

#: the files whose bytes this arm moved.  The law module itself is excluded: it is where the
#: WAIVER fallback decimals are allowed to live, by its own documented class-4 waiver.
#: this test module is deliberately NOT here: it must be able to NAME the decimals it bans.
ARM_OWNED_SOURCES = ("experiments/ddm_ng3_tau_band_cell.py",)
ARM_TOUCHED_SOURCES = ARM_OWNED_SOURCES + (
    "src/tac/tests/test_ddm_ng3_tau_band_cell.py",
    "experiments/ddm_qbt1_qbflow_trainer.py",
    "experiments/ddm_qbr1_born_fairform_burn_prep.py",
    "src/tac/witness_dsl/curriculum_dsl.py",
)


@pytest.fixture(scope="module")
def resolution():
    return resolve_margin_band_threshold()


@pytest.fixture(scope="module")
def band(resolution):
    return (resolution.m_safe, resolution.delta_r)


# ---------------------------------------------------------------------------
# 1. the band is the law's output, never a literal
# ---------------------------------------------------------------------------
def _executable_source(relative: str) -> str:
    """The file's code with docstrings and comments removed.

    A docstring that REPORTS the law's current output is provenance; a literal in code is a
    second source of truth that can silently drift from the artifact.  Only the second is banned,
    which is why the two are separated here instead of grepping the raw bytes.
    """

    import ast

    text = (REPO / relative).read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


def test_no_arm_source_carries_a_delta_r_or_m_safe_literal_in_CODE():
    """The whole point of routing through the law is that no CODE path retypes its output."""

    offenders = []
    for relative in ARM_TOUCHED_SOURCES:
        code = _executable_source(relative)
        for literal in FORBIDDEN_LITERALS:
            if literal in code:
                offenders.append(f"{relative}: {literal}")
    assert not offenders, f"delta_R/m_safe literals must be resolved, not retyped: {offenders}"


def test_the_files_this_arm_authored_carry_no_such_literal_anywhere():
    """Stricter still on the files ng3 owns: not even in a docstring."""

    offenders = []
    for relative in ARM_OWNED_SOURCES:
        text = (REPO / relative).read_text(encoding="utf-8")
        for literal in FORBIDDEN_LITERALS:
            if literal in text:
                offenders.append(f"{relative}: {literal}")
    assert not offenders, f"ng3-owned sources must never carry the decimals: {offenders}"


def test_the_band_is_exactly_the_laws_two_outputs(band, resolution):
    _block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    assert (start, end) == band
    assert start == resolution.headroom * resolution.delta_r
    # the anneal SURVIVES: a band is a band, not a constant temperature.
    assert start > end > 0.0


def test_the_band_ratio_preserves_a_two_to_one_anneal(resolution):
    _block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    assert math.isclose(start / end, resolution.headroom, rel_tol=1e-12)


def test_the_resolution_is_the_n600_artifact_and_never_the_fallback(resolution):
    """A band silently resolved from the WAIVER fallback would be an unmeasured band."""

    assert resolution.n_frames == 600
    assert resolution.artifact_fallback_used is False
    assert resolution.lawref_fallback_used is False


def test_legacy_mode_reproduces_the_shipped_literal_pair():
    _block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe(mode="legacy"))
    assert (start, end) == QBR1_LEGACY_TAU_BAND == (0.15, 0.05)


def test_the_lever_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        ExpectedFlipTauBandMsafe(mode="whatever")


def test_the_lever_block_carries_its_own_provenance(resolution):
    block, _start, _end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    assert block["law"] == "margin_band_satisficing_threshold_v1"
    assert block["delta_r"] == resolution.delta_r
    assert block["m_safe"] == resolution.m_safe
    assert block["n_frames"] == 600
    assert block["artifact_path"] == resolution.artifact_path
    assert isinstance(block["lawref_manifest"], dict) and block["lawref_manifest"]


def test_compile_refuses_a_foreign_override():
    lever = ExpectedFlipTauBandMsafe()
    lever.overrides["--not-a-tau-key"] = 1.0
    with pytest.raises(ValueError, match="unexpected override"):
        compile_qbr1_tau_band_config(lever)


def test_a_headroom_that_would_invert_the_band_is_refused():
    """headroom <= 1 puts tau_start at or below tau_end; tau_for_step would refuse it at update 0."""

    with pytest.raises(ValueError, match="start > end > 0"):
        ExpectedFlipTauBandMsafe(headroom=1.0)


# ---------------------------------------------------------------------------
# 2. both validators accept exactly two bands
# ---------------------------------------------------------------------------
def test_trainer_admits_the_legacy_and_the_law_resolved_band_and_nothing_else(band):
    admissible = qbt.admissible_expected_flip_tau_bands()
    assert admissible == (QBR1_LEGACY_TAU_BAND, band)
    assert (0.15, 0.05) in admissible
    assert band in admissible
    assert (0.15, 0.04) not in admissible
    assert (band[0], band[1] / 2.0) not in admissible


def test_trainer_config_validator_accepts_the_band_and_refuses_a_third(band):
    config = qbt.compile_config(
        action="train", output=REPO / ".omx/tmp/ng3_validator_probe_never_written",
        pair_ids=qbt.SELECTION_IDS, steps=5_000, device="cpu",
    )
    qbt.validate_config(config, require_launch_authority=False)
    banded = copy.deepcopy(config)
    banded["expected_flip_tau_start"], banded["expected_flip_tau_end"] = band
    qbt.validate_config(banded, require_launch_authority=False)
    bogus = copy.deepcopy(config)
    bogus["expected_flip_tau_start"], bogus["expected_flip_tau_end"] = 0.2, 0.01
    with pytest.raises(qbt.QBT1Error, match="expected-flip schedule differs"):
        qbt.validate_config(bogus, require_launch_authority=False)


def _minimal_qbr1_config(band=None, block=None):
    """A config carrying only the fields validate_tau_band_block reads."""

    config = {"expected_flip_tau_start": 0.15, "expected_flip_tau_end": 0.05}
    if band is not None:
        config["expected_flip_tau_start"], config["expected_flip_tau_end"] = band
    if block is not None:
        config["tau_band"] = block
    return config


def test_qbr1_accepts_the_legacy_pair_with_no_block():
    qbr1.validate_tau_band_block(_minimal_qbr1_config())


def test_qbr1_refuses_a_bare_tau_change_with_no_provenance_block(band):
    """This is the HOLE ng3 closes: before this validator, ANY tau reached tau_for_step."""

    with pytest.raises(qbr1.QBR1Error, match="declares no tau_band provenance"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=band))
    with pytest.raises(qbr1.QBR1Error, match="declares no tau_band provenance"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(0.9, 0.8)))


def test_qbr1_accepts_the_law_resolved_band_with_its_block(band):
    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=block))
    assert (start, end) == band


def test_qbr1_refuses_scalars_that_disagree_with_their_own_block():
    """A hand-edited temperature that the block's stated law does not imply."""

    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    with pytest.raises(qbr1.QBR1Error, match="disagree with the tau_band block"):
        qbr1.validate_tau_band_block(
            _minimal_qbr1_config(band=(start * 1.5, end), block=block)
        )


def test_qbr1_refuses_a_block_whose_endpoints_are_not_the_laws(band):
    block, _start, _end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = dict(block)
    tampered["start"] = block["start"] * 1.25
    with pytest.raises(qbr1.QBR1Error, match="not the law's msafe_band band"):
        qbr1.validate_tau_band_block(
            _minimal_qbr1_config(band=(tampered["start"], tampered["end"]), block=tampered)
        )


def test_qbr1_refuses_a_block_with_a_tampered_delta_r():
    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = dict(block)
    tampered["delta_r"] = block["delta_r"] * 1.10
    with pytest.raises(qbr1.QBR1Error, match="tau band delta_r is not the law"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=tampered))


def test_qbr1_refuses_a_block_resolved_on_a_different_population():
    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = dict(block)
    tampered["n_frames"] = 96
    with pytest.raises(qbr1.QBR1Error, match="different delta_R population"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=tampered))


def test_qbr1_refuses_a_block_that_used_the_waiver_fallback():
    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = dict(block)
    tampered["artifact_fallback_used"] = True
    with pytest.raises(qbr1.QBR1Error, match="never the WAIVER fallback"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=tampered))


def test_qbr1_refuses_a_headroom_that_is_not_the_laws_derived_default():
    """A self-consistent block at ANY headroom would be a free knob wearing a law's name."""

    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    resolved = resolve_margin_band_threshold(headroom=3.0)
    tampered = dict(block)
    tampered.update({"headroom": 3.0, "m_safe": resolved.m_safe,
                     "start": resolved.m_safe, "end": resolved.delta_r})
    with pytest.raises(qbr1.QBR1Error, match="not the law's DERIVED default"):
        qbr1.validate_tau_band_block(
            _minimal_qbr1_config(band=(resolved.m_safe, resolved.delta_r), block=tampered)
        )
    # ...and the headroom-3 band is genuinely self-consistent, which is exactly why it needs a
    # gate rather than an internal-consistency check.
    assert resolved.m_safe == 3.0 * resolved.delta_r
    assert (start, end) != (resolved.m_safe, resolved.delta_r)


@pytest.mark.parametrize(
    "dropped", ["start", "end", "delta_r", "m_safe", "headroom", "n_frames"]
)
def test_qbr1_refuses_a_block_missing_any_provenance_key(dropped):
    """A malformed block must raise the typed contract error, never a bare KeyError."""

    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = {key: value for key, value in block.items() if key != dropped}
    with pytest.raises(qbr1.QBR1Error, match="missing required provenance"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=tampered))


def test_qbr1_refuses_a_block_citing_the_wrong_law():
    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = dict(block)
    tampered["law"] = "chan_vese_area_constraint_birth_balance_v1"
    with pytest.raises(qbr1.QBR1Error, match="must cite the registered margin-band law"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=tampered))


def test_qbr1_refuses_an_unknown_band_mode():
    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    tampered = dict(block)
    tampered["mode"] = "freestyle"
    with pytest.raises(qbr1.QBR1Error, match="tau band mode differs"):
        qbr1.validate_tau_band_block(_minimal_qbr1_config(band=(start, end), block=tampered))
    assert set(QBR1_TAU_BAND_MODES) == {"legacy", "msafe_band"}


def test_the_sealed_control_of_record_still_validates_unchanged():
    """Backward compatibility is not an argument here -- it is read off the sealed config."""

    sealed = qbr1.CONFIG_ROOT / f"seed_{ng3.SEED}_{ng3.ARM_NAME}.json"
    if not sealed.is_file():
        pytest.skip(f"sealed control config not mounted: {sealed}")
    config = json.loads(sealed.read_text(encoding="utf-8"))
    assert "tau_band" not in config
    assert (config["expected_flip_tau_start"], config["expected_flip_tau_end"]) == (0.15, 0.05)
    qbr1.validate_tau_band_block(config)


# ---------------------------------------------------------------------------
# 3. tau_for_step under the band
# ---------------------------------------------------------------------------
def test_tau_for_step_accepts_the_band_and_stays_inside_it(band):
    start, end = band
    total = 5_000
    values = [qbt.tau_for_step(step, total, start, end) for step in (0, 1, 2_500, total - 1)]
    assert values[0] == start
    assert values[-1] == end
    assert all(end <= value <= start for value in values)
    assert values == sorted(values, reverse=True)


def test_the_band_is_between_one_and_two_delta_r_for_its_whole_run(resolution):
    start, end = resolution.m_safe, resolution.delta_r
    for step in (0, 1_000, 2_500, 4_999):
        tau = qbt.tau_for_step(step, 5_000, start, end)
        assert 1.0 <= tau / resolution.delta_r <= 2.0


def test_the_legacy_band_was_six_to_two_delta_r_which_is_the_finding(resolution):
    """gm1's framing, asserted so a future reader cannot lose the units."""

    assert math.isclose(0.15 / resolution.delta_r, 6.855, abs_tol=0.01)
    assert math.isclose(0.05 / resolution.delta_r, 2.285, abs_tol=0.01)


# ---------------------------------------------------------------------------
# 4. the cell is ONE lever
# ---------------------------------------------------------------------------
def test_allowed_mutations_are_exactly_the_two_scalars_the_block_and_the_identity_keys():
    assert ng3.ALLOWED_TAU_BAND_MUTATIONS == {
        "cell_id", "output", "tau_band", "expected_flip_tau_start", "expected_flip_tau_end",
    }


def test_validate_tau_band_cell_refuses_a_second_lever():
    control = {
        "cell_id": "c", "output": "/o", "objective": {"a": 1}, "ema": {}, "schedule": {},
        "initial_state": {}, "learning_rate": 2e-4, "margin_constraints": {},
        "pair_ids": [1], "selection_weights": [1], "total_steps": 5000, "milestones": [0],
        "seed": 1, "resume_from": None, "area_cap": None, "chunk_pairs": 16,
        "checkpoint_every_steps": 5, "device": "mps", "source_pins": {},
        "expected_flip_tau_start": 0.15, "expected_flip_tau_end": 0.05,
        "launch_authorized": False,
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
    }
    cell = copy.deepcopy(control)
    cell["expected_flip_tau_start"], cell["expected_flip_tau_end"] = 0.04, 0.02
    cell["tau_band"] = {"mode": "msafe_band"}
    ng3.validate_tau_band_cell(cell, control)

    with_lr = copy.deepcopy(cell)
    with_lr["learning_rate"] = 1e-4
    with pytest.raises(ng3.NG3Error, match="moved more than the lever"):
        ng3.validate_tau_band_cell(with_lr, control)

    warm = copy.deepcopy(cell)
    warm["resume_from"] = "/some/warm/seed.pt"
    with pytest.raises(ng3.NG3Error, match="moved more than the lever"):
        ng3.validate_tau_band_cell(warm, control)


def test_validate_tau_band_cell_refuses_a_composed_area_cap():
    """ng2's cap and ng3's band are separate levers; union is not the sum of legs ([[m164]])."""

    base = {
        "cell_id": "c", "output": "/o", "objective": {}, "ema": {}, "schedule": {},
        "initial_state": {}, "learning_rate": 2e-4, "margin_constraints": {},
        "pair_ids": [1], "selection_weights": [1], "total_steps": 5000, "milestones": [0],
        "seed": 1, "resume_from": None, "area_cap": {"law": "x"}, "chunk_pairs": 16,
        "checkpoint_every_steps": 5, "device": "mps", "source_pins": {},
        "expected_flip_tau_start": 0.15, "expected_flip_tau_end": 0.05,
        "launch_authorized": False,
        "scorer_lane": {"claimed": False, "claim_id": None},
        "metal_lane": {"claimed": False, "claim_id": None},
    }
    cell = copy.deepcopy(base)
    cell["expected_flip_tau_start"], cell["expected_flip_tau_end"] = 0.04, 0.02
    cell["tau_band"] = {"mode": "msafe_band"}
    with pytest.raises(ng3.NG3Error, match="ONE-lever race"):
        ng3.validate_tau_band_cell(cell, base)


def test_the_cell_is_cold_and_unauthorized_by_construction():
    assert "resume_from" not in ng3.ALLOWED_TAU_BAND_MUTATIONS


# ---------------------------------------------------------------------------
# 5. the reference temperature must NOT move with the band
# ---------------------------------------------------------------------------
def test_the_fixed_tau_reference_row_stays_at_its_sealed_value():
    """ng2's telemetry ruler exists to compare cells across DIFFERENT schedules.

    Moving it with the band would destroy exactly the comparability it was added for, so the
    band cell inherits 0.05 unchanged even though that sits ABOVE its whole range.
    """

    assert qbt.EXPECTED_FLIP_TAU_REFERENCE == 0.05
    band_start = resolve_margin_band_threshold().m_safe
    assert qbt.EXPECTED_FLIP_TAU_REFERENCE > band_start


# ---------------------------------------------------------------------------
# 6. falsifiers are pre-registered against the control of record
# ---------------------------------------------------------------------------
def test_falsifiers_cite_the_cold_control_of_record():
    falsifiers = ng3.falsifiers()
    primary = falsifiers["1_primary_the_band_must_act_on_the_excursion"]
    assert primary["endpoint_bar"] == ng3.COLD_CONTROL_S_HAT[5_000] == 0.42514878445269977
    assert primary["peak_bar"] == ng3.COLD_CONTROL_S_HAT[2_000] == 0.48567677825279465
    assert set(falsifiers) >= {
        "1_primary_the_band_must_act_on_the_excursion",
        "2_the_fixed_tau_telemetry_must_be_faithful_in_loop",
        "3_lane_share_must_fall_as_gm1_measured",
    }


def test_the_arm_carries_gm1s_lane_cost_and_never_hides_it():
    """The band buys d_seg focus by spending Lane, which holds ~90.1% of the rate demand."""

    shares = ng3.GM1_MEASURED["lane_grad_share_pct"]
    for endpoint, key in (("tau_start", "two_delta_r"), ("tau_end", "one_delta_r")):
        ratios = [base / band for base, band in zip(shares["tau_0_15"], shares[key], strict=True)]
        low, high = ng3.GM1_MEASURED[f"lane_share_ratio_at_{endpoint}_{key}"]
        assert math.isclose(min(ratios), low, abs_tol=0.001), (endpoint, min(ratios), low)
        assert math.isclose(max(ratios), high, abs_tol=0.001), (endpoint, max(ratios), high)
        # the cost is real in both cases: Lane always loses share as tau falls.
        assert all(ratio > 1.0 for ratio in ratios)


def test_the_charter_falsifier_3_range_is_corrected_and_the_correction_is_derivable():
    """gm1's 1.60-2.08x is the 0.5*delta_R column; this band never goes below delta_R.

    The correction is re-derived here from gm1's own table rather than trusted from the constant,
    so a future edit of either cannot silently re-import the out-of-band number.
    """

    shares = ng3.GM1_MEASURED["lane_grad_share_pct"]
    out_of_band = [base / half for base, half
                   in zip(shares["tau_0_15"], shares["half_delta_r"], strict=True)]
    assert math.isclose(min(out_of_band), 1.599, abs_tol=0.001)
    assert math.isclose(max(out_of_band), 2.077, abs_tol=0.001)
    # ...which is the charter's 1.6-2.1x.  The decisive statement is that the charter's interval
    # does NOT contain this band's step-0 ratio at EITHER endpoint, so a burn scored against the
    # charter's text would have reported a miss that the physics never predicted.
    charter_low, charter_high = 1.6, 2.1
    at_start = shares["tau_0_15"][0] / shares["two_delta_r"][0]
    at_end = shares["tau_0_15"][0] / shares["one_delta_r"][0]
    assert at_start < charter_low
    assert at_end < charter_low
    # gm1's headline "1.60-2.08x" is this column's min/max ROUNDED to 2dp (min 1.5991 -> 1.60),
    # which is why the comparison is made at gm1's own precision rather than exactly.
    assert (round(min(out_of_band), 2), round(max(out_of_band), 2)) == (1.60, 2.08)
    assert charter_low - 0.005 <= min(out_of_band) and max(out_of_band) <= charter_high

    correction = ng3.CHARTER_CORRECTION_FALSIFIER_3
    assert math.isclose(correction["corrected_step0_ratio_at_tau_start"],
                        shares["tau_0_15"][0] / shares["two_delta_r"][0], abs_tol=0.001)
    assert math.isclose(correction["corrected_step0_ratio_at_tau_end"],
                        shares["tau_0_15"][0] / shares["one_delta_r"][0], abs_tol=0.001)
    assert "0.5" in correction["why_it_is_wrong"]
    assert ng3.falsifiers()["3_lane_share_must_fall_as_gm1_measured"][
        "charter_correction"] is correction


def test_the_band_removes_the_waste_gm1_measured():
    removed = ng3.GM1_MEASURED["waste_removed_pct_step0"]
    waste = ng3.GM1_MEASURED["wasted_share_pct_step0"]
    for key in ("two_delta_r", "one_delta_r"):
        derived = (waste["tau_0_15"] - waste[key]) / waste["tau_0_15"] * 100.0
        assert math.isclose(derived, removed[key], abs_tol=0.05), (key, derived, removed[key])


# ---------------------------------------------------------------------------
# 7. the memo's equations leg must actually cite its two laws
# ---------------------------------------------------------------------------
def test_the_design_memo_cites_both_required_equations():
    memo = REPO / ".omx/research/ddm_ng3_tau_band_cell_20260904.md"
    if not memo.is_file():
        pytest.skip("design memo not written yet")
    text = memo.read_text(encoding="utf-8")
    for equation in ("margin_band_satisficing_threshold_v1",
                    "scalar_top1_top2_margin_is_exact_distance_to_flip_v1"):
        assert equation in text, f"the memo must cite {equation}"
    # a memo that quotes a delta_R decimal is fine (it is a REPORT of a resolution); a memo that
    # never names the artifact is not.
    assert "delta_R_noise_floor_n600.json" in text


def test_the_arm_never_writes_authorized_configs():
    """MAIN copies sealed -> authorized.  This arm must not be able to."""

    source = (REPO / "experiments/ddm_ng3_tau_band_cell.py").read_text(encoding="utf-8")
    assert not re.search(r"authorized_configs.*(?:atomic_json|write_text|mkdir)", source)
    assert "launch_authorized\"] = True" not in source
    assert "/tmp/" not in source


# ---------------------------------------------------------------------------
# 8. the config-surface lever must not enter the argv-composable set
# ---------------------------------------------------------------------------
def test_a_config_surface_lever_is_not_argv_composable():
    """``--dsl-lever NAME`` composes onto ARGV; this lever compiles into a JSON cell.

    Before ng3 the exclusion was ACCIDENTAL -- ng2's ``AreaCapBornRareClass`` is kept out of the
    composable set only because its factory takes required arguments.  A config-surface lever
    with all-default parameters (this one) slipped straight in and failed the CI parse-test with
    an ``unrecognized arguments`` error naming neither the lever nor the reason.  The registry now
    excludes by the SHAPE of what a lever emits, which is the property that actually matters.
    """

    from tac.witness_dsl import lever_registry as lr

    composable = lr.name_composable_levers()
    assert "ExpectedFlipTauBandMsafe" not in composable
    assert "AreaCapBornRareClass" not in composable
    _lever, why = lr._composability_check("ExpectedFlipTauBandMsafe")
    assert why is not None and "CONFIG-surface keys" in why
    assert "compile_qbr1_tau_band_config" in why
    # and the exclusion is narrow: every remaining composable lever emits only trainer flags
    for name in composable:
        lever, reason = lr._composability_check(name)
        assert reason is None, (name, reason)
        assert all(str(key).startswith("--") for key in lever.overrides), name


def test_the_exclusion_did_not_swallow_a_flag_emitting_lever():
    """A lever with NO overrides at all is still composable -- the rule bans non-flag keys, not
    the absence of keys."""

    from tac.witness_dsl import lever_registry as lr

    assert "StageTransitionSoftVelocityBlend" in lr.name_composable_levers()


# ---------------------------------------------------------------------------
# 9. a sealed config's bytes must be reproducible, or its sha cannot be quoted
# ---------------------------------------------------------------------------
def test_the_compiled_band_block_is_byte_stable_across_compiles():
    """ng3's FIRST seal was not byte-stable: two identical compiles produced two shas.

    The cause was the LawRef manifest's ``resolved_at`` observation time riding inside the
    config.  A memo that quotes a config sha for MAIN to verify before firing is worthless if
    the sha moves with the clock, so the volatile fields are stripped at the compile boundary.
    """

    import time as _time

    from tac.witness_dsl.curriculum_dsl import VOLATILE_LAWREF_MANIFEST_FIELDS

    first, _s1, _e1 = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    _time.sleep(1.05)  # cross a whole-second boundary: the dropped field has 1 s resolution
    second, _s2, _e2 = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    for field in VOLATILE_LAWREF_MANIFEST_FIELDS:
        assert field not in first["lawref_manifest"]
    assert first["lawref_manifest_volatile_fields_excluded"] == list(
        VOLATILE_LAWREF_MANIFEST_FIELDS)


def test_the_dropped_timestamp_is_not_lost_only_moved():
    """Stripping provenance would be signal loss; the law still reports it, and the arm's dated
    RESOLUTION receipt is where it is kept."""

    from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (
        resolve_margin_band_threshold,
    )
    from tac.witness_dsl.curriculum_dsl import VOLATILE_LAWREF_MANIFEST_FIELDS

    manifest = resolve_margin_band_threshold().lawref_manifest
    for field in VOLATILE_LAWREF_MANIFEST_FIELDS:
        assert field in manifest, f"the law must still report {field}"
    source = (REPO / "experiments/ddm_ng3_tau_band_cell.py").read_text(encoding="utf-8")
    assert "lawref_manifest_full_including_volatile_fields" in source


def test_the_validator_does_not_read_any_volatile_field():
    """The gate must be blind to the stripped fields, or a legacy config would refuse."""

    from tac.witness_dsl.curriculum_dsl import VOLATILE_LAWREF_MANIFEST_FIELDS

    block, start, end = compile_qbr1_tau_band_config(ExpectedFlipTauBandMsafe())
    with_volatile = dict(block)
    with_volatile["lawref_manifest"] = dict(block["lawref_manifest"])
    for field in VOLATILE_LAWREF_MANIFEST_FIELDS:
        with_volatile["lawref_manifest"][field] = "1999-01-01T00:00:00Z"
    qbr1.validate_tau_band_block(
        _minimal_qbr1_config(band=(start, end), block=with_volatile)
    )


def test_recompile_determinism_reports_exactly_what_holds():
    """The seal must not claim byte-reproducibility it does not have.

    ``ema.lawref.resolved_at`` is a dated LawRef observation the QBR1 lineage has always kept
    inside the config, and ``qbt.stable_ema_law_identity`` already pops exactly that field --
    so ng3 compares by the same rule and REPORTS the residual instead of asserting a stronger
    property.  The half this arm owns (``tau_band``) is byte-stable outright.
    """

    base = {"ema": {"value": 0.999, "lawref": {"resolved_at": "A", "value": 0.999}},
            "tau_band": {"start": 1.0}, "other": 1}
    later = copy.deepcopy(base)
    later["ema"]["lawref"]["resolved_at"] = "B"
    report = ng3.recompile_determinism(base, later)
    assert report["stable_identity_reproduces"] is True
    assert report["raw_bytes_reproduce"] is False
    assert report["keys_that_moved_across_two_compiles"] == ["ema"]
    assert report["volatile_paths_excluded"] == ["ema.lawref.resolved_at"]
    assert "shasum" in report["what_this_means_for_MAIN"]

    identical = copy.deepcopy(base)
    clean = ng3.recompile_determinism(base, identical)
    assert clean["raw_bytes_reproduce"] is True
    assert clean["keys_that_moved_across_two_compiles"] == []


def test_recompile_determinism_refuses_a_real_divergence():
    base = {"ema": {"lawref": {"resolved_at": "A"}}, "learning_rate": 2e-4}
    drifted = {"ema": {"lawref": {"resolved_at": "A"}}, "learning_rate": 1e-4}
    with pytest.raises(ng3.NG3Error, match="not reproducible across two compiles"):
        ng3.recompile_determinism(base, drifted)


def test_recompile_determinism_refuses_a_moving_tau_band_block():
    """The one block ng3 owns may never be the volatile one."""

    base = {"ema": {"lawref": {"resolved_at": "A"}}, "tau_band": {"start": 1.0}}
    moved = {"ema": {"lawref": {"resolved_at": "B"}}, "tau_band": {"start": 2.0}}
    with pytest.raises(ng3.NG3Error, match="must be byte-stable"):
        ng3.recompile_determinism(base, moved)


def test_ng3_uses_the_same_volatile_rule_the_trainer_already_had():
    """Not a second convention: qbt.stable_ema_law_identity pops the identical field."""

    import inspect

    source = inspect.getsource(qbt.stable_ema_law_identity)
    assert '"resolved_at"' in source
    assert ("ema", "lawref", "resolved_at") in ng3.VOLATILE_CONFIG_PATHS

