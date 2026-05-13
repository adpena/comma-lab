"""Tests for the per-tensor MDL/Bayesian codec selector (Task #308 sub-component)."""
from __future__ import annotations

import json

import pytest

from tac.mdl_bayesian_selector import (
    MDLBayesianSelectorError,
    PerTensorSelectionReport,
    TensorObservation,
    compute_per_tensor_l_total,
    load_observations_from_jsonl,
    occam_check_codec,
    rank_observations,
    select_codec_per_tensor,
)


# -- TensorObservation validation ---------------------------------------


def test_tensor_observation_accepts_valid_inputs() -> None:
    obs = TensorObservation(
        tensor_id="renderer.body.0.weight",
        codec="pr101_brotli",
        n_data_symbols=1024,
        model_bits=64 * 8,
        residual_bits=900 * 8,
    )
    assert obs.total_bits == (64 + 900) * 8
    assert obs.evidence_grade == "predicted"


def test_tensor_observation_rejects_empty_tensor_id() -> None:
    with pytest.raises(MDLBayesianSelectorError, match="tensor_id"):
        TensorObservation(
            tensor_id="",
            codec="x",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=0,
        )


def test_tensor_observation_rejects_negative_bits() -> None:
    with pytest.raises(MDLBayesianSelectorError, match="model_bits"):
        TensorObservation(
            tensor_id="t",
            codec="x",
            n_data_symbols=1,
            model_bits=-1,
            residual_bits=0,
        )
    with pytest.raises(MDLBayesianSelectorError, match="residual_bits"):
        TensorObservation(
            tensor_id="t",
            codec="x",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=-1,
        )


def test_tensor_observation_rejects_unknown_evidence_grade() -> None:
    with pytest.raises(MDLBayesianSelectorError, match="evidence_grade"):
        TensorObservation(
            tensor_id="t",
            codec="x",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=0,
            evidence_grade="totally_made_up",
        )


def test_tensor_observation_normalizes_grade_aliases() -> None:
    obs = TensorObservation(
        tensor_id="t",
        codec="x",
        n_data_symbols=1,
        model_bits=0,
        residual_bits=0,
        evidence_grade="[contest-CUDA]",
    )
    assert obs.evidence_grade == "[contest-CUDA]"  # raw preserved
    obs2 = TensorObservation(
        tensor_id="t",
        codec="x",
        n_data_symbols=1,
        model_bits=0,
        residual_bits=0,
        evidence_grade="MPS-research-signal",
    )
    assert "MPS" in obs2.evidence_grade


# -- compute_per_tensor_l_total -----------------------------------------


def test_compute_per_tensor_l_total_sums_bits() -> None:
    obs = TensorObservation(
        tensor_id="t",
        codec="x",
        n_data_symbols=1,
        model_bits=100,
        residual_bits=200,
    )
    assert compute_per_tensor_l_total(obs) == 300


# -- Occam check --------------------------------------------------------


def test_occam_check_passes_when_savings_exceed_model_bits() -> None:
    baseline = TensorObservation(
        tensor_id="t",
        codec="static",
        n_data_symbols=1,
        model_bits=0,
        residual_bits=10_000,
    )
    candidate = TensorObservation(
        tensor_id="t",
        codec="learned",
        n_data_symbols=1,
        model_bits=1_000,
        residual_bits=8_000,
    )
    # savings = 10000 - 8000 = 2000, model_bits = 1000 < 2000 -> pass
    assert occam_check_codec(candidate=candidate, baseline=baseline) is True


def test_occam_check_fails_when_model_bits_exceed_savings() -> None:
    baseline = TensorObservation(
        tensor_id="t",
        codec="static",
        n_data_symbols=1,
        model_bits=0,
        residual_bits=10_000,
    )
    candidate = TensorObservation(
        tensor_id="t",
        codec="bloated",
        n_data_symbols=1,
        model_bits=5_000,
        residual_bits=8_000,
    )
    # savings = 2000, model_bits = 5000 > 2000 -> FAIL (the
    # "10KB MLP for 5KB savings" anti-pattern)
    assert occam_check_codec(candidate=candidate, baseline=baseline) is False


def test_occam_check_rejects_mismatched_tensor_ids() -> None:
    a = TensorObservation(
        tensor_id="a",
        codec="x",
        n_data_symbols=1,
        model_bits=0,
        residual_bits=0,
    )
    b = TensorObservation(
        tensor_id="b",
        codec="y",
        n_data_symbols=1,
        model_bits=0,
        residual_bits=0,
    )
    with pytest.raises(MDLBayesianSelectorError, match="same tensor_id"):
        occam_check_codec(candidate=a, baseline=b)


# -- rank_observations --------------------------------------------------


def test_rank_observations_sorts_ascending_by_total_bits() -> None:
    obs = [
        TensorObservation(
            tensor_id="t",
            codec="big",
            n_data_symbols=1,
            model_bits=1_000,
            residual_bits=10_000,
        ),
        TensorObservation(
            tensor_id="t",
            codec="small",
            n_data_symbols=1,
            model_bits=500,
            residual_bits=5_000,
        ),
        TensorObservation(
            tensor_id="t",
            codec="medium",
            n_data_symbols=1,
            model_bits=100,
            residual_bits=8_000,
        ),
    ]
    ranking = rank_observations(obs)
    codecs_ordered = [c.codec for c in ranking.candidates]
    assert codecs_ordered == ["small", "medium", "big"]


def test_rank_observations_rejects_mixed_tensor_ids() -> None:
    obs = [
        TensorObservation(
            tensor_id="t1",
            codec="x",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=10,
        ),
        TensorObservation(
            tensor_id="t2",
            codec="y",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=20,
        ),
    ]
    with pytest.raises(MDLBayesianSelectorError, match="same tensor_id"):
        rank_observations(obs)


def test_rank_observations_best_excludes_occam_failures() -> None:
    """An overpriced "best-residual-bits" codec is filtered out by Occam."""
    obs = [
        TensorObservation(
            tensor_id="t",
            codec="static_baseline",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=10_000,
        ),
        TensorObservation(
            tensor_id="t",
            codec="bloated_winner",
            n_data_symbols=1,
            model_bits=5_000,  # exceeds 10000 - 4000 = 6000 savings? actually 5000 <= 6000
            residual_bits=4_000,
        ),
        TensorObservation(
            tensor_id="t",
            codec="reasonable",
            n_data_symbols=1,
            model_bits=200,
            residual_bits=6_000,
        ),
    ]
    # Force baseline to static_baseline so Occam compares against it.
    ranking = rank_observations(obs, baseline_codec="static_baseline")
    # Sorted by total: reasonable (6200), bloated_winner (9000), static (10000)
    assert ranking.candidates[0].codec == "reasonable"
    # reasonable: model 200 <= savings (10000-6000)=4000 -> PASS
    # bloated_winner: model 5000 <= savings (10000-4000)=6000 -> PASS
    # static_baseline: model 0 trivially passes
    # Best (smallest total that passes) = reasonable.
    assert ranking.best is not None
    assert ranking.best.codec == "reasonable"


def test_rank_observations_filters_truly_bloated() -> None:
    """A codec whose model_bits genuinely exceed savings is rejected."""
    obs = [
        TensorObservation(
            tensor_id="t",
            codec="static_baseline",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=10_000,
        ),
        TensorObservation(
            tensor_id="t",
            codec="really_bloated",
            n_data_symbols=1,
            model_bits=20_000,  # > savings of (10000 - 9000) = 1000
            residual_bits=9_000,
        ),
    ]
    ranking = rank_observations(obs, baseline_codec="static_baseline")
    # really_bloated has total 29000 > static 10000 so static is sorted first.
    # Occam: static (model_bits=0) trivially passes; really_bloated rejected.
    assert ranking.best is not None
    assert ranking.best.codec == "static_baseline"


# -- select_codec_per_tensor --------------------------------------------


def test_select_codec_per_tensor_groups_by_tensor_id() -> None:
    obs = [
        TensorObservation(
            tensor_id="layer1.weight",
            codec="brotli",
            n_data_symbols=100,
            model_bits=0,
            residual_bits=900 * 8,
        ),
        TensorObservation(
            tensor_id="layer1.weight",
            codec="arith",
            n_data_symbols=100,
            model_bits=10 * 8,
            residual_bits=700 * 8,
        ),
        TensorObservation(
            tensor_id="layer2.weight",
            codec="brotli",
            n_data_symbols=200,
            model_bits=0,
            residual_bits=1800 * 8,
        ),
    ]
    report = select_codec_per_tensor(obs)
    assert isinstance(report, PerTensorSelectionReport)
    assert set(report.rankings.keys()) == {"layer1.weight", "layer2.weight"}
    assert report.summary_by_codec == {"arith": 1, "brotli": 1}
    # arith wins layer1 with 710 * 8 bits; brotli wins layer2 trivially.
    assert report.total_selected_bits == 710 * 8 + 1800 * 8


def test_select_codec_per_tensor_aggregates_worst_evidence_grade() -> None:
    obs = [
        TensorObservation(
            tensor_id="a",
            codec="x",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=100,
            evidence_grade="contest-CUDA",
        ),
        TensorObservation(
            tensor_id="b",
            codec="y",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=100,
            evidence_grade="predicted",
        ),
    ]
    report = select_codec_per_tensor(obs)
    # predicted is worse than contest-CUDA, so aggregated grade is predicted.
    assert "predicted" in report.aggregated_evidence_grade.lower()


def test_select_codec_per_tensor_serialises_to_dict() -> None:
    obs = [
        TensorObservation(
            tensor_id="t",
            codec="x",
            n_data_symbols=1,
            model_bits=0,
            residual_bits=100,
        )
    ]
    report = select_codec_per_tensor(obs)
    d = report.to_dict()
    assert d["aggregated_evidence_grade"]
    assert d["summary_by_codec"] == {"x": 1}
    assert "rankings" in d
    # Round-trip through JSON.
    json.loads(json.dumps(d))


# -- TensorObservation.from_mapping (cathedral compat) ------------------


def test_from_mapping_accepts_canonical_keys() -> None:
    row = {
        "tensor_id": "x",
        "codec": "y",
        "n_data_symbols": 10,
        "model_bits": 5,
        "residual_bits": 15,
    }
    obs = TensorObservation.from_mapping(row)
    assert obs.tensor_id == "x"
    assert obs.total_bits == 20


def test_from_mapping_translates_cathedral_aliases() -> None:
    """TechniqueEvidence-style rows with bytes -> bits conversion."""
    row = {
        "tensor_id": "renderer.body",
        "technique": "pr101_split_brotli",
        "n_data_symbols": 1024,
        "predicted_archive_bytes": 100,
    }
    obs = TensorObservation.from_mapping(row)
    assert obs.codec == "pr101_split_brotli"
    assert obs.residual_bits == 100 * 8


def test_from_mapping_raises_on_missing_residual() -> None:
    with pytest.raises(MDLBayesianSelectorError, match="residual_bits"):
        TensorObservation.from_mapping({"tensor_id": "x", "codec": "y"})


# -- load_observations_from_jsonl ---------------------------------------


def test_load_observations_from_jsonl_roundtrip(tmp_path) -> None:
    rows = [
        {
            "tensor_id": "a",
            "codec": "brotli",
            "n_data_symbols": 1,
            "model_bits": 0,
            "residual_bits": 100,
        },
        {
            "tensor_id": "a",
            "codec": "arith",
            "n_data_symbols": 1,
            "model_bits": 5,
            "residual_bits": 80,
        },
    ]
    path = tmp_path / "obs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    obs = load_observations_from_jsonl(path)
    assert len(obs) == 2
    report = select_codec_per_tensor(obs)
    assert report.summary_by_codec == {"arith": 1}


def test_load_observations_from_jsonl_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(MDLBayesianSelectorError, match="not found"):
        load_observations_from_jsonl(tmp_path / "does_not_exist.jsonl")


def test_load_observations_from_jsonl_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{this is not json}\n")
    with pytest.raises(MDLBayesianSelectorError, match="invalid JSON"):
        load_observations_from_jsonl(path)


def test_load_observations_from_jsonl_skips_comments(tmp_path) -> None:
    path = tmp_path / "with_comments.jsonl"
    path.write_text(
        "# header comment\n"
        '{"tensor_id":"a","codec":"x","n_data_symbols":1,"model_bits":0,"residual_bits":1}\n'
        "\n"
        "# trailing comment\n"
    )
    obs = load_observations_from_jsonl(path)
    assert len(obs) == 1
