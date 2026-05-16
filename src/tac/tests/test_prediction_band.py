# SPDX-License-Identifier: MIT
"""Tests for prediction-band custody validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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


def _exact_anchor(axis: str = "contest_cuda") -> dict[str, object]:
    archive_bytes = 1
    score = 25.0 * archive_bytes / 37_545_489
    return {
        "axis": axis,
        "archive_sha256": _sha("c"),
        "runtime_tree_sha256": _sha("d"),
        "score": score,
        "seg_dist": 0.0,
        "pose_dist": 0.0,
        "archive_bytes": archive_bytes,
        "n_samples": 600,
        "hardware": "modal-t4",
        "inflate_device": "cuda" if axis == "contest_cuda" else "cpu",
        "eval_device": "cuda" if axis == "contest_cuda" else "cpu",
        "auth_eval_command": f"contest_auth_eval --axis {axis}",
        "log_path": f"experiments/results/z3/{axis}.log",
        "artifact_path": "experiments/results/z3/anchor.json",
    }


def _write_anchor_files(base_dir: Path, band: PredictionBand) -> None:
    baseline_path = base_dir / band.baseline.artifact_path
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text("baseline\n", encoding="utf-8")
    for ledger_path_text in band.band_source.local_ledger_paths:
        stripped = ledger_path_text.removeprefix("file:").strip()
        ledger_path = Path(stripped)
        if ledger_path.is_absolute():
            continue
        ledger_path = base_dir / ledger_path
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text("ledger\n", encoding="utf-8")
    for anchor in band.empirical_anchor.anchors:
        for key in ("artifact_path", "log_path"):
            path = base_dir / str(anchor[key])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{key}\n", encoding="utf-8")


def _valid_band() -> PredictionBand:
    return PredictionBand(
        band_id="pb_test_valid_v1",
        subject_id="z3_balle_hyperprior_bolton",
        band_kind="delta_score",
        low=-0.010,
        high=-0.001,
        axis="contest_cuda",
        baseline=BaselineRef(
            label="a1_cpu_cuda_pair",
            axis="contest_cuda",
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
            anchors=(_exact_anchor(),),
        ),
        planning_only=True,
        score_claim=False,
    )


def test_valid_prediction_band_can_influence_rank_reward(tmp_path: Path):
    band = _valid_band()
    _write_anchor_files(tmp_path, band)

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )
    assert verdict.blockers == ()
    assert verdict.valid_for_rank_reward is True
    assert verdict.valid_for_dispatch_planning is True
    assert verdict.valid_for_promotion is False


def test_source_ledger_missing_blocks_rank_reward(tmp_path: Path):
    band = replace(
        _valid_band(),
        band_source=BandSource(
            local_ledger_paths=("file:.omx/research/missing_prediction_band.md",),
            research_basis_ids=("balle_hyperprior_2018",),
            claim_scope="planning prior only",
        ),
    )
    _write_anchor_files(tmp_path, replace(band, band_source=_valid_band().band_source))

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_source_ledger_path_missing" in verdict.blockers


def test_source_ledger_transient_path_blocks_rank_reward(tmp_path: Path):
    band = replace(
        _valid_band(),
        band_source=BandSource(
            local_ledger_paths=("file:/tmp/prediction_band_source.md",),
            research_basis_ids=("balle_hyperprior_2018",),
            claim_scope="planning prior only",
        ),
    )
    _write_anchor_files(tmp_path, band)

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_source_ledger_path_transient" in verdict.blockers


def test_source_ledger_outside_repo_blocks_rank_reward(tmp_path: Path):
    outside_path = tmp_path.parent / "outside_prediction_band_source.md"
    band = replace(
        _valid_band(),
        band_source=BandSource(
            local_ledger_paths=(f"file:{outside_path}",),
            research_basis_ids=("balle_hyperprior_2018",),
            claim_scope="planning prior only",
        ),
    )
    _write_anchor_files(tmp_path, band)

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_source_ledger_path_outside_repo" in verdict.blockers


def test_baseline_axis_mismatch_blocks_rank_reward():
    band = _valid_band()
    verdict = validate_prediction_band(
        replace(
            band,
            baseline=replace(band.baseline, axis="macOS-CPU advisory"),
        ),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_baseline_axis_mismatch" in verdict.blockers


def test_landed_anchor_axis_mismatch_blocks_rank_reward():
    band = _valid_band()
    verdict = validate_prediction_band(
        replace(
            band,
            empirical_anchor=EmpiricalAnchorRef(
                status="landed",
                anchors=(_exact_anchor("macOS-CPU advisory"),),
            ),
        ),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_empirical_anchor_axis_mismatch" in verdict.blockers


def test_landed_anchor_requires_exact_eval_metric_closure():
    band = _valid_band()
    verdict = validate_prediction_band(
        replace(
            band,
            empirical_anchor=EmpiricalAnchorRef(
                status="landed",
                anchors=({
                    "axis": "contest_cuda",
                    "archive_sha256": _sha("c"),
                    "runtime_tree_sha256": _sha("d"),
                    "score": 0.1987,
                    "artifact_path": "experiments/results/z3/anchor.json",
                },),
            ),
        ),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_empirical_anchor_n_samples_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_hardware_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_command_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_log_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_inflate_device_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_eval_device_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_archive_bytes_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_seg_dist_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_pose_dist_missing" in verdict.blockers


def test_landed_anchor_rejects_score_formula_mismatch():
    band = _valid_band()
    bad_anchor = dict(_exact_anchor())
    bad_anchor["score"] = 0.1987

    verdict = validate_prediction_band(
        replace(
            band,
            empirical_anchor=EmpiricalAnchorRef(status="landed", anchors=(bad_anchor,)),
        ),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_empirical_anchor_score_formula_mismatch" in verdict.blockers


def test_landed_anchor_missing_custody_blocks_rank_reward():
    band = _valid_band()
    verdict = validate_prediction_band(
        replace(
            band,
            empirical_anchor=EmpiricalAnchorRef(
                status="landed",
                anchors=({
                    "axis": "contest_cuda",
                    "score": "0.1987",
                    "artifact_path": "",
                },),
            ),
        ),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_empirical_anchor_score_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_custody_missing" in verdict.blockers
    assert "prediction_band_empirical_anchor_artifact_missing" in verdict.blockers


def test_landed_anchor_requires_explicit_artifact_base_dir():
    verdict = validate_prediction_band(
        _valid_band(),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert (
        "prediction_band_empirical_anchor_artifact_base_dir_missing"
        in verdict.blockers
    )


def test_landed_anchor_missing_log_file_blocks_rank_reward(tmp_path: Path):
    band = _valid_band()
    baseline_path = tmp_path / band.baseline.artifact_path
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text("baseline\n", encoding="utf-8")
    artifact_path = tmp_path / str(band.empirical_anchor.anchors[0]["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("anchor\n", encoding="utf-8")

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_empirical_anchor_log_missing" in verdict.blockers


def test_baseline_artifact_file_must_be_repo_local(tmp_path: Path):
    band = _valid_band()
    _write_anchor_files(tmp_path, band)

    verdict = validate_prediction_band(
        replace(
            band,
            baseline=replace(
                band.baseline,
                artifact_path="/tmp/stale_baseline.json",
            ),
        ),
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_baseline_artifact_transient" in verdict.blockers


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


def test_mixed_landed_band_requires_paired_cpu_cuda_anchors():
    band = replace(
        _valid_band(),
        axis="mixed",
        baseline=replace(_valid_band().baseline, axis="mixed"),
        empirical_anchor=EmpiricalAnchorRef(
            status="landed",
            anchors=(_exact_anchor("contest_cuda"),),
        ),
    )

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_empirical_anchor_paired_axes_missing" in verdict.blockers


def test_mixed_landed_band_accepts_paired_cpu_cuda_anchors(tmp_path: Path):
    base = _valid_band()
    band = replace(
        base,
        axis="mixed",
        baseline=replace(base.baseline, axis="mixed"),
        empirical_anchor=EmpiricalAnchorRef(
            status="landed",
            anchors=(_exact_anchor("contest_cpu"), _exact_anchor("contest_cuda")),
        ),
    )
    _write_anchor_files(tmp_path, band)

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
        artifact_base_dir=tmp_path,
    )

    assert verdict.blockers == ()
    assert verdict.valid_for_rank_reward is True


def test_hyphenated_contest_cuda_axis_is_not_exact_eval_authority():
    band = replace(
        _valid_band(),
        axis="contest-cuda",
        baseline=replace(_valid_band().baseline, axis="contest-cuda"),
        empirical_anchor=EmpiricalAnchorRef(
            status="landed",
            anchors=(_exact_anchor("contest-cuda"),),
        ),
    )

    verdict = validate_prediction_band(
        band,
        expected_subject_id="z3_balle_hyperprior_bolton",
        expected_low=-0.010,
        expected_high=-0.001,
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_axis_not_exact_eval" in verdict.blockers


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
        axis="contest_cuda",
    )
    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_unknown_research_basis" in verdict.blockers


def test_prediction_band_accepts_legacy_research_basis_aliases():
    band = _valid_band()
    payload = {
        **prediction_band_to_dict(band),
        "band_source": {
            "local_ledger_paths": ["file:.omx/research/test.md"],
            "research_basis_ids": ["balle_2018"],
            "claim_scope": "planning prior only",
        },
    }
    verdict = validate_optional_prediction_band(
        payload,
        subject_id="z3_balle_hyperprior_bolton",
        low=-0.010,
        high=-0.001,
        axis="contest_cuda",
    )
    assert "prediction_band_unknown_research_basis" not in verdict.blockers


def test_score_claim_inside_prediction_band_fails_closed():
    band = _valid_band()
    payload = {**prediction_band_to_dict(band), "score_claim": True}
    verdict = validate_optional_prediction_band(
        payload,
        subject_id="z3_balle_hyperprior_bolton",
        low=-0.010,
        high=-0.001,
        axis="contest_cuda",
    )
    assert verdict.valid_for_dispatch_planning is False
    assert "prediction_band_score_claim_forbidden" in verdict.blockers


def test_prediction_band_mapping_rejects_string_authority_bools():
    band = _valid_band()
    payload = {**prediction_band_to_dict(band), "score_claim": "false"}

    verdict = validate_optional_prediction_band(
        payload,
        subject_id="z3_balle_hyperprior_bolton",
        low=-0.010,
        high=-0.001,
        axis="contest_cuda",
    )

    assert verdict.valid_for_rank_reward is False
    assert "prediction_band_parse_failed" in verdict.blockers
    assert any("score_claim must be a literal JSON boolean" in item for item in verdict.annotations)
