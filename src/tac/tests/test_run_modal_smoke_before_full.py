from __future__ import annotations

import json

from tac.auth_eval_result import recompute_contest_score_from_payload
from tools.run_modal_smoke_before_full import _validate_smoke_result


def _auth_payload(*, score_axis: str = "contest_cuda") -> dict:
    payload = {
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.0004,
        "archive_size_bytes": 150_000,
        "score_axis": score_axis,
        "lane_tag": "[contest-CUDA]"
        if score_axis == "contest_cuda"
        else "[diagnostic-auth-eval]",
        "evidence_grade": "contest-CUDA" if score_axis == "contest_cuda" else "B",
        "exact_cuda_eval_complete": score_axis == "contest_cuda",
        "score_claim": score_axis == "contest_cuda",
        "score_claim_valid": score_axis == "contest_cuda",
    }
    payload["canonical_score"] = recompute_contest_score_from_payload(payload)
    return payload


def test_smoke_validation_accepts_canonical_contest_cuda_score() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "contest_auth_eval_cuda.json": json.dumps(_auth_payload()),
            },
        }
    )

    assert green is True
    assert "SMOKE GREEN" in diagnostic


def test_smoke_validation_rejects_diagnostic_cuda_score() -> None:
    green, diagnostic = _validate_smoke_result(
        {
            "returncode": 0,
            "timed_out": False,
            "artifacts": {
                "contest_auth_eval_cuda.json": json.dumps(
                    _auth_payload(score_axis="diagnostic_cuda")
                ),
            },
        }
    )

    assert green is False
    assert "not a finite component-coherent contest-CUDA" in diagnostic
