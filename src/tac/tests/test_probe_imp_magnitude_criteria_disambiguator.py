# SPDX-License-Identifier: MIT
"""Tests for the Lane 17 IMP magnitude-criteria disambiguator."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from probe_imp_magnitude_criteria_disambiguator import (  # noqa: E402
    SCHEMA,
    build_probe_payload,
)


class TinyImpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3, bias=False)
        self.c2 = nn.Conv2d(2, 2, 1, bias=False)
        with torch.no_grad():
            self.c1.weight.copy_(
                torch.tensor(
                    [
                        [[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]]],
                        [[[0.10, 0.11, 0.12], [0.13, 0.14, 0.15], [0.16, 0.17, 0.18]]],
                    ]
                )
            )
            self.c2.weight.copy_(
                torch.tensor([[[[0.19]], [[0.20]]], [[[0.21]], [[0.22]]]])
            )


def _payload(**kwargs):
    model = TinyImpModel()
    return build_probe_payload(
        model=model,
        anchor_path="synthetic/tiny-renderer.bin",
        anchor_bytes=b"FP4Asynthetic",
        anchor_magic="FP4A",
        cycles=2,
        sparsity_increment=0.25,
        measure_archive=False,
        created_utc="2026-05-18T08:00:00Z",
        **kwargs,
    )


def test_probe_enumerates_catalog308_criteria_without_score_authority() -> None:
    payload = _payload()

    assert payload["schema"] == SCHEMA
    assert payload["lane_id"] == "lane_17_imp_10cycle"
    assert payload["evidence_axis"] == "[local-IMP-mask-byte-proxy advisory]"
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_paid_dispatch"] is False
    assert payload["rank_or_kill_eligible"] is False
    assert payload["summary"]["catalog308_disambiguator_landed"] is True
    assert payload["summary"]["score_gradient_sidecar_supplied"] is False
    assert "requires_catalog123_score_gradient_saliency_sidecar" in payload["blockers"]

    criteria = {item["criterion_id"]: item for item in payload["criteria"]}
    assert "l1_per_tensor_canonical_frankle" in criteria
    assert "hessian_trace_per_tensor_obd_proxy" in criteria
    assert "score_gradient_saliency_catalog123" in criteria
    assert criteria["score_gradient_saliency_catalog123"]["status"] == (
        "blocked_missing_score_gradient_saliency_sidecar_for_authority"
    )
    assert criteria["score_gradient_saliency_catalog123"]["mask_stats"] is None
    assert criteria["score_gradient_saliency_catalog123"]["archive_measurement"] is None
    assert "no_global_l1_surrogate_emitted_for_score_gradient_branch" in criteria[
        "score_gradient_saliency_catalog123"
    ]["measurement_blockers"]
    assert all(
        pair["left"] != "score_gradient_saliency_catalog123"
        and pair["right"] != "score_gradient_saliency_catalog123"
        for pair in payload["pairwise_mask_jaccard"]
    )


def test_probe_is_deterministic_for_same_model_and_inputs() -> None:
    left = _payload()
    right = _payload()

    assert left["criteria"] == right["criteria"]
    assert left["pairwise_mask_jaccard"] == right["pairwise_mask_jaccard"]


def test_score_gradient_sidecar_changes_catalog123_status_and_mask() -> None:
    baseline = _payload()
    with_saliency = _payload(
        score_gradient_saliency={
            "c1.weight": 100.0,
            "c2.weight": 0.01,
        },
        score_gradient_saliency_metadata={"loaded": True, "path": "synthetic.json"},
    )

    sal_criteria = {item["criterion_id"]: item for item in with_saliency["criteria"]}

    assert with_saliency["summary"]["score_gradient_sidecar_supplied"] is True
    assert "requires_catalog123_score_gradient_saliency_sidecar" not in (
        with_saliency["blockers"]
    )
    assert sal_criteria["score_gradient_saliency_catalog123"]["status"] == (
        "ready_with_supplied_score_gradient_saliency_sidecar"
    )
    assert sal_criteria["score_gradient_saliency_catalog123"]["mask_stats"] is not None
    assert sal_criteria["score_gradient_saliency_catalog123"]["mask_stats"][
        "mask_sha256"
    ]
    assert any(
        pair["left"] == "score_gradient_saliency_catalog123"
        or pair["right"] == "score_gradient_saliency_catalog123"
        for pair in with_saliency["pairwise_mask_jaccard"]
    )
