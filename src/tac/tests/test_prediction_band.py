# SPDX-License-Identifier: MIT
"""Tests for prediction-band custody validation."""

from __future__ import annotations

from tac.optimization.prediction_band import (
    BandSource,
    BaselineRef,
    EmpiricalAnchorRef,
    PredictionBand,
    SupersessionRef,
    UncertaintyRef,
    prediction_band_to_dict,
    validate_optional_prediction_band,
    validate_prediction_band,
)


def _sha(seed: str) -> str:
    return (seed * 64)[:64]


def _valid_band() -> PredictionBand:
    return PredictionBand(
        band_id="pb_test_valid_v1",
        subject_id="z3_balle_hyperprior_bolton",
        band_kind="delta_score",
        low=-0.010,
        high=-0.001,
        axis="contest-cuda",
        baseline=BaselineRef(
            label="a1_cpu_cuda_pair",
            axis="contest-cuda",
            score=0.1928,
            archive_sha256=_sha("a"),
            runtime_tree_sha256=_sha("b"),
            artifact_path="experiments/results/a1/baseline.json",
        ),
        band_source=BandSource(
            local_ledger_paths=("file:.omx/research/z3_phase_2_council_20260514.md",),
            research_basis_ids=("balle_hyperprior_2018",),
            claim_scope="planning prior only; exact paired eval required",
        ),
        uncertainty=UncertaintyRef(
            method="empirical-prior-band",
            confidence_tag="low_n",
            n_empirical_anchors=1,
            notes="paired anchor required before promotion",
        ),
        supersession=SupersessionRef(status="active"),
        empirical_anchor=EmpiricalAnchorRef(
            status="landed",
            anchors=({
                "axis": "contest-cuda",
                "archive_sha256": _sha("c"),
                "runtime_tree_sha256": _sha("d"),
                "score": 0.1987,
                "artifact_path": "experiments/results/z3/anchor.json",
            },),
        ),
        planning_only=True,
        score_claim=False,
    )


def test_valid_prediction_band_can_influence_rank_reward():
    verdict = validate_prediction_band(
        _valid_band(),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )
    assert verdict.blockers == ()
    assert verdict.valid_for_rank_reward is True
    assert verdict.valid_for_dispatch_planning is True
    assert verdict.valid_for_promotion is False


def test_missing_nonzero_prediction_band_becomes_rank_blocker():
    verdict = validate_optional_prediction_band(
        None,
        subject_id="z5_predictive_coding_world_model",
        low=-0.038,
        high=-0.025,
        axis="mixed",
    )
    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_missing" in verdict.blockers
    assert verdict.valid_for_dispatch_planning is True


def test_zero_delta_missing_band_is_annotation_only():
    verdict = validate_optional_prediction_band(
        None,
        subject_id="magic_codec",
        low=0.0,
        high=0.0,
        axis="rate",
    )
    assert verdict.valid_for_rank_reward is True
    assert verdict.blockers == ()
    assert "prediction_band_zero_delta_no_rank_reward" in verdict.annotations


def test_unknown_research_basis_blocks_rank_reward():
    band = _valid_band()
    payload = {
        **prediction_band_to_dict(band),
        "baseline": band.baseline,
        "band_source": {
            "local_ledger_paths": ["file:.omx/research/test.md"],
            "research_basis_ids": ["missing_source_id"],
            "claim_scope": "planning prior only",
        },
    }
    verdict = validate_optional_prediction_band(
        payload,
        subject_id="z3_balle_hyperprior_bolton",
        low=-0.010,
        high=-0.001,
        axis="contest-cuda",
    )
    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_unknown_research_basis" in verdict.blockers


def test_score_claim_inside_prediction_band_fails_closed():
    band = _valid_band()
    payload = {**prediction_band_to_dict(band), "score_claim": True}
    verdict = validate_optional_prediction_band(
        payload,
        subject_id="z3_balle_hyperprior_bolton",
        low=-0.010,
        high=-0.001,
        axis="contest-cuda",
    )
    assert verdict.valid_for_dispatch_planning is False
    assert "prediction_band_score_claim_forbidden" in verdict.blockers
