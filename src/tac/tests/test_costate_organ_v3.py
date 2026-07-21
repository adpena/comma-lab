# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import math

import pytest

from tac.witness_control.costate_organ_v3 import (
    EMA_EQUATION_ID,
    POOL_EQUATION_ID,
    RealizedDeltaRow,
    append_realized_delta_row,
    denoise_realized_target,
    emit_m1_byte_close_row,
    graded_realizability,
    load_realized_delta_corpus,
    ndcg_at_k,
    paired_bootstrap_delta,
    pool_kkt_marginals,
    rank_metrics,
    stage_survival_distribution,
)

HISTOGRAM = {
    "killed_at_uint8": 0,
    "killed_at_resize_dilution": 0,
    "killed_at_stem": 0,
    "killed_at_head_same_rival": 204,
    "killed_at_head_wrong_rival": 0,
    "survived_but_collateral": 5,
    "survived_clean": 289,
}


def test_stage_survival_reproduces_sealed_r1b7_counts() -> None:
    distribution = stage_survival_distribution(HISTOGRAM)
    assert distribution.total == 498
    assert distribution.through_uint8 == pytest.approx(498.5 / 499.0)
    assert distribution.through_resize == distribution.through_uint8
    assert distribution.through_stem == distribution.through_uint8
    assert distribution.through_head == pytest.approx(294.5 / 499.0)
    assert distribution.clean == pytest.approx(289.5 / 499.0)
    assert distribution.collateral == pytest.approx(5.5 / 499.0)


def test_stage_survival_rejects_missing_bucket() -> None:
    invalid = dict(HISTOGRAM)
    invalid.pop("killed_at_stem")
    with pytest.raises(ValueError, match="keys"):
        stage_survival_distribution(invalid)


def test_stage_survival_rejects_non_integer_count() -> None:
    invalid = dict(HISTOGRAM)
    invalid["survived_clean"] = 289.0
    with pytest.raises(ValueError, match="integers"):
        stage_survival_distribution(invalid)


def test_graded_realizability_is_continuous_and_fail_closed() -> None:
    distribution = stage_survival_distribution(HISTOGRAM)
    half = graded_realizability(distribution=distribution, route_stage="clean", strength=0.5)
    assert half["value"] == pytest.approx(distribution.clean * 0.5)
    assert half["learned_parameters"] == 0
    invalid = graded_realizability(
        distribution=distribution,
        route_stage="clean",
        formulation_valid=False,
    )
    assert invalid["value"] == 0.0
    assert invalid["status"] == "formulation_scoped_negative"


def test_pool_kkt_competitors_cannot_double_claim_one_ceiling() -> None:
    rows = [
        {"id": "a", "corpus": "c2_carrier_smoke", "vehicle": "palette", "variant": "oneside_lane"},
        {"id": "b", "corpus": "c2_carrier_smoke", "vehicle": "palette", "variant": "oneside_lane"},
    ]
    ranked = pool_kkt_marginals(rows, [0.7, 0.5])
    assert ranked[0]["value"] == pytest.approx(0.7)
    assert ranked[1]["value"] == 0.0
    assert sum(row["value"] for row in ranked) == pytest.approx(0.7)
    assert all(row["equation_id"] == POOL_EQUATION_ID for row in ranked)


def test_composite_pool_claims_sum_to_raw_lambda() -> None:
    row = {
        "id": "a",
        "corpus": "c2_carrier_smoke",
        "vehicle": "palette",
        "variant": "oneside_shallow",
    }
    result = pool_kkt_marginals([row], [0.25])[0]
    assert sum(result["claims"].values()) == pytest.approx(0.25)
    assert set(result["claims"]) == {
        "c2:palette:road_lane_boundary",
        "c2:palette:movable_boundary",
    }


def test_ema_delag_uses_exact_constant_decay_response() -> None:
    row = {
        "corpus": "#205_asof_trajectory",
        "realized_benefit_s": 0.01,
        "apparatus_valid": True,
        "source_epochs": [350, 375],
    }
    result = denoise_realized_target(row)
    response = 1.0 - 0.997**25
    assert result["value"] == pytest.approx(0.01 / response)
    assert result["weight"] == pytest.approx(response**2)
    assert result["equation_id"] == EMA_EQUATION_ID


def test_subset_precision_weight_is_n_over_600() -> None:
    result = denoise_realized_target(
        {
            "corpus": "c2_carrier_smoke",
            "realized_benefit_s": 0.1,
            "apparatus_valid": True,
            "source_scope": "stride-5 subset (120 frames of n600)",
        }
    )
    assert result["value"] == 0.1
    assert result["weight"] == pytest.approx(0.2)


def test_rank_metrics_report_top_precision_ndcg_and_ties() -> None:
    metrics = rank_metrics([(3.0,), (2.0,), (2.0,), (0.0,)], [1.0, -1.0, 0.5, -2.0], ids=list("abcd"), k=3)
    assert metrics["top8_precision"] == pytest.approx(2.0 / 3.0)
    assert 0.0 <= metrics["decision_ndcg_at_8"] <= 1.0
    assert metrics["tie_pairs"] == 1
    assert metrics["tied_rows"] == 2
    assert metrics["tie_groups"] == 1


def test_ndcg_uses_apparatus_weight_on_gain() -> None:
    keys = [(2.0,), (1.0,)]
    targets = [1.0, 10.0]
    assert ndcg_at_k(keys, targets, weights=[1.0, 0.0], k=2) == 1.0


def test_bootstrap_identity_delta_has_exact_zero_interval() -> None:
    keys = [(float(index),) for index in range(6)]
    result = paired_bootstrap_delta(
        before_keys=keys,
        after_keys=keys,
        before_targets=[float(index) for index in range(6)],
        after_targets=[float(index) for index in range(6)],
        before_weights=[1.0] * 6,
        after_weights=[1.0] * 6,
        strata=["a"] * 3 + ["b"] * 3,
        metric=lambda k, t, w: rank_metrics(k, t, weights=w, k=2)["weighted_spearman"],
        replicates=100,
        seed=7,
    )
    assert result["delta"] == 0.0
    assert result["ci95"] == [0.0, 0.0]


def _source(tmp_path) -> tuple[str, str]:
    source = tmp_path / "receipt.json"
    source.write_text("{}\n")
    return source.name, hashlib.sha256(source.read_bytes()).hexdigest()


def _row(tmp_path, *, identifier: str = "row", benefit: float = 0.1) -> RealizedDeltaRow:
    source, digest = _source(tmp_path)
    return RealizedDeltaRow(
        id=identifier,
        factors={"exact_gap": 1.0, "visibility": 0.5, "realizability": 0.5, "byte_price": 1.0},
        factor_context={"route": "test"},
        realized_benefit_s=benefit,
        apparatus_valid=True,
        corpus="fixture",
        byte_delta=0,
        source_receipt=source,
        source_receipt_sha256=digest,
        producer="test",
    )


def test_corpus_append_roundtrip_and_exact_dedup(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    row = _row(tmp_path)
    assert append_realized_delta_row(row, corpus, source_root=tmp_path)["status"] == "APPENDED"
    assert append_realized_delta_row(row, corpus, source_root=tmp_path)["status"] == "EXACT_DUPLICATE_NOOP"
    loaded = load_realized_delta_corpus(corpus)
    assert len(loaded) == 1 and loaded[0]["id"] == "row"


def test_corpus_conflicting_duplicate_refuses(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    append_realized_delta_row(_row(tmp_path), corpus, source_root=tmp_path)
    with pytest.raises(ValueError, match="conflicting"):
        append_realized_delta_row(_row(tmp_path, benefit=0.2), corpus, source_root=tmp_path)


def test_corpus_source_hash_custody_is_enforced(tmp_path) -> None:
    row = _row(tmp_path)
    (tmp_path / "receipt.json").write_text('{"drift":true}\n')
    with pytest.raises(ValueError, match="drifted"):
        append_realized_delta_row(row, tmp_path / "corpus.jsonl", source_root=tmp_path)


def test_corpus_source_path_cannot_escape_declared_root(tmp_path) -> None:
    row = _row(tmp_path)
    invalid = RealizedDeltaRow(**{**row.__dict__, "source_receipt": "../receipt.json"})
    with pytest.raises(ValueError, match="repo-relative"):
        append_realized_delta_row(invalid, tmp_path / "corpus.jsonl", source_root=tmp_path)


def test_m1_emission_is_blocked_when_realized_row_absent(tmp_path) -> None:
    source, digest = _source(tmp_path)
    result = emit_m1_byte_close_row(
        {"schema": "dry"},
        source_receipt=source,
        source_receipt_sha256=digest,
        corpus_path=tmp_path / "corpus.jsonl",
    )
    assert result["status"] == "NOT_EMITTED_REALIZED_ROW_ABSENT"
    assert not (tmp_path / "corpus.jsonl").exists()


def test_m1_zero_byte_claim_is_rejected(tmp_path) -> None:
    source, digest = _source(tmp_path)
    receipt = {
        "costate_realized_delta": {
            "id": "m1",
            "factors": {"exact_gap": 1, "visibility": 1, "realizability": 1, "byte_price": 1},
            "factor_context": {},
            "realized_benefit_s": 0.1,
            "apparatus_valid": True,
            "corpus": "m1",
            "byte_delta": 0,
        }
    }
    with pytest.raises(ValueError, match="nonzero-byte"):
        emit_m1_byte_close_row(
            receipt,
            source_receipt=source,
            source_receipt_sha256=digest,
            corpus_path=tmp_path / "corpus.jsonl",
        )


def test_m1_receipt_types_are_not_truthy_or_integer_coerced(tmp_path) -> None:
    source, digest = _source(tmp_path)
    base = {
        "id": "m1",
        "factors": {"exact_gap": 1, "visibility": 1, "realizability": 1, "byte_price": 1},
        "factor_context": {},
        "realized_benefit_s": 0.1,
        "apparatus_valid": True,
        "corpus": "m1",
        "byte_delta": 1,
    }
    for changed, match in [
        ({"apparatus_valid": "false"}, "bool"),
        ({"byte_delta": 1.5}, "int"),
    ]:
        with pytest.raises(ValueError, match=match):
            emit_m1_byte_close_row(
                {"costate_realized_delta": {**base, **changed}},
                source_receipt=source,
                source_receipt_sha256=digest,
                corpus_path=tmp_path / "corpus.jsonl",
            )


def test_realized_row_rejects_nonfinite_target(tmp_path) -> None:
    row = _row(tmp_path)
    invalid = RealizedDeltaRow(**{**row.__dict__, "realized_benefit_s": math.nan})
    with pytest.raises(ValueError, match="finite"):
        invalid.validated()
