# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.win_families.terminal_compile` (F2) and
:mod:`tac.win_families.model_axis` (F4).

The staleness detector is exercised as BEHAVIOUR: an upstream artifact is genuinely
replaced after a downstream stage ran, and the pipeline must refuse to certify.
"""

from __future__ import annotations

import pytest

from tac import gt_lineage
from tac import local_contest_instruments as lci
from tac.win_families import model_axis as ma
from tac.win_families import terminal_compile as tc

# --- helpers -----------------------------------------------------------------


def _stage(name, consumes, produces, value="v", **kwargs):
    return tc.CompileStage(
        name=name,
        consumes=tuple(consumes),
        produces=tuple(produces),
        run=lambda inputs: dict.fromkeys(produces, value),
        **kwargs,
    )


def _seg_then_carrier():
    return [
        _stage("seg_edit", (), ("tokens",), value="tokens_v1"),
        _stage("carrier_resolve", ("tokens",), ("carrier_codes",), value="codes_v1"),
    ]


# --- stage contract -----------------------------------------------------------


def test_stage_requires_a_name():
    with pytest.raises(tc.TerminalCompileError, match="needs a name"):
        _stage("", (), ("x",))


def test_stage_must_produce_something():
    with pytest.raises(tc.TerminalCompileError, match="produces nothing"):
        _stage("s", ("a",), ())


def test_stage_cannot_consume_and_produce_the_same_artifact():
    with pytest.raises(tc.TerminalCompileError, match="undecidable"):
        _stage("s", ("a",), ("a",))


def test_canonical_order_is_the_measured_topological_sort():
    assert tc.canonical_compile_order == (
        "seg_edit",
        "carrier_resolve",
        "compensation",
        "rate_reencode",
        "container_search",
    )


# --- pipeline construction ------------------------------------------------------


def test_pipeline_needs_a_stage():
    with pytest.raises(tc.TerminalCompileError, match="at least one stage"):
        tc.CompilePipeline([])


def test_pipeline_refuses_duplicate_stage_names():
    with pytest.raises(tc.TerminalCompileError, match="duplicate stage names"):
        tc.CompilePipeline([_stage("seg_edit", (), ("a",)), _stage("seg_edit", (), ("b",))])


def test_pipeline_refuses_out_of_canonical_order_stages():
    stages = [
        _stage("carrier_resolve", (), ("carrier_codes",)),
        _stage("seg_edit", (), ("tokens",)),
    ]
    with pytest.raises(tc.TerminalCompileError, match="out of canonical compile order"):
        tc.CompilePipeline(stages)


def test_pipeline_allows_omitting_stages():
    pipeline = tc.CompilePipeline(
        [_stage("seg_edit", (), ("tokens",)), _stage("container_search", ("tokens",), ("archive",))]
    )
    assert len(pipeline.stages) == 2


def test_pipeline_refuses_an_unknown_target_axis():
    with pytest.raises(lci.InstrumentRefusal, match="unknown score axis"):
        tc.CompilePipeline(_seg_then_carrier(), target_axis="contest-TPU")


# --- running --------------------------------------------------------------------


def test_pipeline_runs_every_stage_and_produces_artifacts():
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    artifacts = pipeline.run()
    assert artifacts["tokens"] == "tokens_v1"
    assert artifacts["carrier_codes"] == "codes_v1"


def test_stage_consuming_an_unproduced_artifact_refuses():
    pipeline = tc.CompilePipeline([_stage("seg_edit", ("nothing",), ("tokens",))])
    with pytest.raises(tc.TerminalCompileError, match="no earlier stage produced"):
        pipeline.run()


def test_stage_returning_a_non_dict_refuses():
    stage = tc.CompileStage(
        name="seg_edit", consumes=(), produces=("tokens",), run=lambda inputs: "oops"
    )
    with pytest.raises(tc.TerminalCompileError, match="expected a dict"):
        tc.CompilePipeline([stage]).run()


def test_stage_not_returning_declared_output_refuses():
    stage = tc.CompileStage(
        name="seg_edit", consumes=(), produces=("tokens",), run=lambda inputs: {"other": 1}
    )
    with pytest.raises(tc.TerminalCompileError, match="did not return"):
        tc.CompilePipeline([stage]).run()


# --- staleness: the bug this module extincts ---------------------------------------


def test_replacing_an_upstream_artifact_makes_the_downstream_stage_stale():
    """A carrier solved against pre-edit frames is a carrier for a body that is gone."""
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    pipeline.run()
    assert pipeline.stale_stages() == ()
    pipeline.set_artifact("tokens", "tokens_v2")
    stale = pipeline.stale_stages()
    assert len(stale) == 1
    assert stale[0].stage == "carrier_resolve"
    assert stale[0].stale_inputs == ("tokens",)


def test_certify_refuses_a_stale_pipeline():
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    pipeline.run()
    pipeline.set_artifact("tokens", "tokens_v2")
    with pytest.raises(tc.TerminalCompileError, match="stale stages"):
        pipeline.certify()


def test_rerunning_the_stale_stage_clears_staleness():
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    pipeline.run()
    pipeline.set_artifact("tokens", "tokens_v2")
    pipeline.run_stage(pipeline.stages[1])
    assert pipeline.stale_stages() == ()


def test_setting_an_artifact_to_an_equal_value_is_not_stale():
    """Staleness is content-addressed: re-setting the same content changes nothing."""
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    pipeline.run()
    pipeline.set_artifact("tokens", "tokens_v1")
    assert pipeline.stale_stages() == ()


def test_certify_refuses_when_a_stage_never_ran():
    """An unrun stage is not a passing stage -- vacuity must not read as green."""
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    pipeline.run_stage(pipeline.stages[0])
    with pytest.raises(tc.TerminalCompileError, match="never ran"):
        pipeline.certify()


def test_never_ran_lists_the_unrun_stages():
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    assert pipeline.never_ran() == ("seg_edit", "carrier_resolve")


def test_certify_passes_on_a_fresh_pipeline():
    pipeline = tc.CompilePipeline(_seg_then_carrier())
    pipeline.run()
    receipt = pipeline.certify()
    assert receipt["all_stages_measured"] is True
    assert receipt["score_claim"] is False


def test_certify_reports_modelled_stages():
    stages = [
        _stage("seg_edit", (), ("tokens",)),
        _stage("rate_reencode", ("tokens",), ("archive",), measured=False),
    ]
    pipeline = tc.CompilePipeline(stages)
    pipeline.run()
    receipt = pipeline.certify()
    assert receipt["modelled_stages"] == ["rate_reencode"]
    assert receipt["all_stages_measured"] is False


def test_artifact_digest_is_stable_and_content_addressed():
    assert tc.artifact_digest(b"abc") == tc.artifact_digest(b"abc")
    assert tc.artifact_digest(b"abc") != tc.artifact_digest(b"abd")


def test_digest_of_unknown_artifact_refuses():
    with pytest.raises(tc.TerminalCompileError, match="no artifact named"):
        tc.CompilePipeline(_seg_then_carrier()).digest_of("ghost")


# --- the qs5 GT-lineage correction ---------------------------------------------------


def test_pyav_fed_compensation_cannot_certify_a_cuda_compile():
    """ddm_qs5's Schur compensation read the PyAV GT_POSE table; the ship axis is DALI."""
    stages = [
        _stage("seg_edit", (), ("carrier_codes",)),
        tc.compensation_stage(
            gt_lineage=gt_lineage.PYAV_YUV420_TO_RGB,
            run=lambda inputs: {"compensated_codes": "c"},
        ),
    ]
    pipeline = tc.CompilePipeline(stages, target_axis=lci.AXIS_CONTEST_CUDA)
    with pytest.raises(tc.TerminalCompileError, match="ddm_qs5"):
        pipeline.run()


def test_dali_fed_compensation_certifies_a_cuda_compile():
    stages = [
        _stage("seg_edit", (), ("carrier_codes",)),
        tc.compensation_stage(
            gt_lineage=gt_lineage.DALI_NVDEC,
            run=lambda inputs: {"compensated_codes": "c"},
        ),
    ]
    pipeline = tc.CompilePipeline(stages, target_axis=lci.AXIS_CONTEST_CUDA)
    pipeline.run()
    assert pipeline.certify()["gt_lineage"] == gt_lineage.DALI_NVDEC


def test_pyav_compensation_is_fine_for_a_cpu_target():
    stages = [
        _stage("seg_edit", (), ("carrier_codes",)),
        tc.compensation_stage(
            gt_lineage=gt_lineage.PYAV_YUV420_TO_RGB,
            run=lambda inputs: {"compensated_codes": "c"},
        ),
    ]
    pipeline = tc.CompilePipeline(stages, target_axis=lci.AXIS_CONTEST_CPU)
    pipeline.run()
    assert pipeline.certify()["gt_lineage"] == gt_lineage.PYAV_YUV420_TO_RGB


def test_compensation_stage_requires_a_declared_lineage():
    with pytest.raises(tc.TerminalCompileError, match="must declare the GT lineage"):
        tc.compensation_stage(gt_lineage=None, run=lambda inputs: {})


def test_stage_without_gt_lineage_is_not_checked():
    pipeline = tc.CompilePipeline(_seg_then_carrier(), target_axis=lci.AXIS_CONTEST_CUDA)
    pipeline.assert_stage_lineages()


# --- the jg1 -> jg2 measured-rate correction --------------------------------------


def test_measured_rate_leg_reports_realized_bits_per_token():
    """ddm_jg2: +30 B over the changed tokens is the real number."""
    leg = tc.RateLeg(changed_tokens=58, archive_delta_bytes=30)
    assert leg.measured is True
    assert leg.realized_bits_per_token == pytest.approx(8 * 30 / 58)


def test_modelled_rate_leg_refuses_to_certify():
    leg = tc.RateLeg(changed_tokens=100, modelled_bits_per_token=4.718)
    assert leg.measured is False
    with pytest.raises(tc.TerminalCompileError, match="MODELLED, not measured"):
        leg.assert_measured()


def test_measured_rate_leg_certifies():
    tc.RateLeg(changed_tokens=100, archive_delta_bytes=52).assert_measured()


def test_rate_leg_needs_a_measurement_or_a_model():
    with pytest.raises(tc.TerminalCompileError, match="either a MEASURED"):
        tc.RateLeg(changed_tokens=10)


def test_rate_leg_refuses_negative_token_count():
    with pytest.raises(tc.TerminalCompileError):
        tc.RateLeg(changed_tokens=-1, archive_delta_bytes=5)


def test_rate_leg_with_zero_tokens_has_no_per_token_rate():
    assert tc.RateLeg(changed_tokens=0, archive_delta_bytes=0).realized_bits_per_token is None


def test_jg2_measured_rate_is_below_the_jg1_model():
    """The correction that moved the headline: 4.1379 measured vs 4.718 modelled."""
    leg = tc.RateLeg(changed_tokens=18_000, archive_delta_bytes=int(4.1379 * 18_000 / 8))
    assert leg.realized_bits_per_token < 4.718


# --- F4: calibration -----------------------------------------------------------------


def _calibration(**kwargs):
    base = {
        "factor": ma.FX1_PARSE_BACK_CALIBRATION,
        "source": ".omx/research/ddm_fx1_fixed_point_logistic_mixer_20260817.md",
        "regime": ma.FX1_CALIBRATION_REGIME,
    }
    base.update(kwargs)
    return ma.Calibration(**base)


def test_calibration_requires_a_source():
    with pytest.raises(ma.ModelAxisError, match="FALSE SATURATION"):
        _calibration(source="  ")


def test_calibration_requires_a_regime():
    with pytest.raises(ma.ModelAxisError, match="cross-regime"):
        _calibration(regime="")


def test_calibration_refuses_a_non_positive_factor():
    with pytest.raises(ma.ModelAxisError, match="finite and > 0"):
        _calibration(factor=0.0)


def test_calibration_applies_in_its_own_regime():
    _calibration().assert_applicable(ma.FX1_CALIBRATION_REGIME)


def test_calibration_refuses_a_silent_cross_regime_transfer():
    with pytest.raises(ma.ModelAxisError, match="Re-measure it"):
        _calibration().assert_applicable("some_other_body")


def test_calibration_allows_an_acknowledged_cross_regime_transfer():
    _calibration().assert_applicable("some_other_body", allow_cross_regime=True)


# --- F4: sectors and the reservoir -----------------------------------------------------


def test_sector_headroom_and_fraction():
    sector = ma.SectorPrice("within_miss", ceiling_bytes=1247.0, realized_bytes=100.4)
    assert sector.headroom_bytes == pytest.approx(1146.6)
    assert sector.realized_fraction == pytest.approx(100.4 / 1247.0)


def test_sector_refuses_realized_above_ceiling():
    with pytest.raises(ma.ModelAxisError, match="Either the ceiling is wrong"):
        ma.SectorPrice("s", ceiling_bytes=10.0, realized_bytes=11.0)


def test_sector_refuses_a_negative_ceiling():
    with pytest.raises(ma.ModelAxisError, match="negative ceiling"):
        ma.SectorPrice("s", ceiling_bytes=-1.0)


def test_sector_refuses_negative_realized():
    with pytest.raises(ma.ModelAxisError, match="negative realized"):
        ma.SectorPrice("s", ceiling_bytes=10.0, realized_bytes=-1.0)


def test_zero_ceiling_sector_reports_zero_fraction():
    assert ma.SectorPrice("s", ceiling_bytes=0.0).realized_fraction == 0.0


def test_reservoir_needs_a_sector():
    with pytest.raises(ma.ModelAxisError, match="at least one sector"):
        ma.ModelAxisReservoir([])


def test_reservoir_refuses_duplicate_sectors():
    with pytest.raises(ma.ModelAxisError, match="duplicate sector"):
        ma.ModelAxisReservoir(
            [ma.SectorPrice("s", 1.0), ma.SectorPrice("s", 2.0)]
        )


def test_reservoir_totals_its_sectors():
    reservoir = ma.ModelAxisReservoir(
        [ma.SectorPrice("within_miss", 1247.0, 100.4), ma.SectorPrice("hit", 560.0, 560.0)]
    )
    assert reservoir.ceiling_bytes == pytest.approx(1807.0)
    assert reservoir.realized_bytes == pytest.approx(660.4)
    assert reservoir.headroom_bytes == pytest.approx(1146.6)


def test_reservoir_flags_a_saturated_sector():
    reservoir = ma.ModelAxisReservoir(
        [ma.SectorPrice("hit", 560.0, 560.0), ma.SectorPrice("within_miss", 1247.0, 100.4)]
    )
    assert reservoir.saturated_sectors() == ("hit",)


def test_projection_deflates_the_modelled_bytes():
    """ddm_fx1's x1.260 means a modelled 560 B is NOT 560 archive bytes."""
    reservoir = ma.ModelAxisReservoir([ma.SectorPrice("hit", 560.0)])
    projected = reservoir.project_archive_bytes(
        560.0, _calibration(), regime=ma.FX1_CALIBRATION_REGIME
    )
    assert projected == pytest.approx(560.0 / 1.260)
    assert projected < 560.0


def test_projection_refuses_a_silent_cross_regime_calibration():
    reservoir = ma.ModelAxisReservoir([ma.SectorPrice("hit", 560.0)])
    with pytest.raises(ma.ModelAxisError, match="Re-measure it"):
        reservoir.project_archive_bytes(560.0, _calibration(), regime="another_body")


def test_projected_score_delta_is_negative_for_a_saving():
    reservoir = ma.ModelAxisReservoir([ma.SectorPrice("hit", 560.0)])
    delta = reservoir.projected_score_delta(
        560.0, _calibration(), regime=ma.FX1_CALIBRATION_REGIME
    )
    assert delta < 0


def test_reservoir_json_is_never_a_score_claim():
    payload = ma.ModelAxisReservoir([ma.SectorPrice("hit", 1.0)]).to_json()
    assert payload["score_claim"] is False
    assert "false saturation" in payload["note"]


# --- F4: bit accounting ------------------------------------------------------------------


def test_seven_bits_is_at_most_one_byte():
    assert ma.bits_to_bytes_ceiling(7) == 1


def test_eight_bits_is_one_byte():
    assert ma.bits_to_bytes_ceiling(8) == 1


def test_nine_bits_is_two_bytes():
    assert ma.bits_to_bytes_ceiling(9) == 2


def test_negative_bits_round_away_from_zero():
    assert ma.bits_to_bytes_ceiling(-7) == -1


def test_zero_bits_is_zero_bytes():
    assert ma.bits_to_bytes_ceiling(0) == 0
