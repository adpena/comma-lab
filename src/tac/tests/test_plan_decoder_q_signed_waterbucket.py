from __future__ import annotations

from tools.plan_decoder_q_signed_waterbucket import _rank_atoms


def _rows() -> list[dict]:
    return [
        {
            "mutation_id": "same-regressed",
            "mutation": {"tensor_name": "decoder.weight", "q_offset": 7, "delta": 1},
            "fixed_length_runtime_compatible": True,
            "length_delta": 0,
            "op3v3_target_evidence": {
                "score_impact_abs_sum": 10.0,
                "axis_score_impact_abs_sum": {"seg": 8.0, "pose": 2.0},
            },
        },
        {
            "mutation_id": "inverse-candidate",
            "mutation": {"tensor_name": "decoder.weight", "q_offset": 7, "delta": -1},
            "fixed_length_runtime_compatible": True,
            "length_delta": 0,
            "op3v3_target_evidence": {
                "score_impact_abs_sum": 10.0,
                "axis_score_impact_abs_sum": {"seg": 8.0, "pose": 2.0},
            },
        },
    ]


def _signed_calibration() -> dict:
    return {
        "schema": "decoder_q_surface_sign_calibration_labels.v1",
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "summary": {"label_count": 1},
        "labels": [
            {
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "candidate_id": "surface-bad",
                "observed_score_delta_sign": 1,
                "atom_mutation_keys": [
                    {
                        "tensor_name": "decoder.weight",
                        "q_offset": 7,
                        "delta": 1,
                    }
                ],
            }
        ],
    }


def test_signed_calibration_prioritizes_inverse_of_regressing_surface_atom() -> None:
    ranked = _rank_atoms(
        _rows(),
        advisory_by_key={},
        signed_calibration=_signed_calibration(),
    )

    assert ranked["sign_inverted_from_bad"][0]["candidate_id"] == "inverse-candidate"
    assert (
        ranked["sign_inverted_from_bad"][0]["signed_calibration"][
            "inverse_of_regressing_candidate_count"
        ]
        == 1
    )
    assert (
        ranked["sign_inverted_from_bad"][0]["signed_calibration"][
            "priority_multiplier"
        ]
        == 4.0
    )
    assert len(ranked["sign_inverted_from_bad"]) == 2
    assert (
        ranked["seg_heavy"][-1]["signed_calibration"]["same_sign_regression_count"]
        == 1
    )


def test_signed_calibration_refuses_promotable_payload() -> None:
    signed_calibration = _signed_calibration()
    signed_calibration["promotable"] = True

    try:
        _rank_atoms(_rows(), advisory_by_key={}, signed_calibration=signed_calibration)
    except SystemExit as exc:
        assert "promotable must be false" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("promotable signed calibration payload was accepted")
