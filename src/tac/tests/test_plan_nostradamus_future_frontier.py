# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_nostradamus_future_frontier.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("nostradamus_future_frontier_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


planner = _load_script()


def _anchor_payload() -> dict[str, Any]:
    return {
        "archive_size_bytes": 229_756,
        "avg_posenet_dist": 0.0001894,
        "avg_segnet_dist": 0.00057185,
        "n_samples": 600,
        "provenance": {
            "archive_sha256": "c6f004",
            "device": "cuda",
            "gpu_model": "Tesla T4",
        },
        "score_recomputed_from_components": 0.25369011029397787,
    }


def test_score_math_and_break_even_bytes_are_formula_owned() -> None:
    score = planner.score_from_components(
        archive_bytes=229_756,
        seg_dist=0.00057185,
        pose_dist=0.0001894,
    )
    assert score == pytest.approx(0.2536901999275155)
    assert planner.rate_score_delta(-13_924) == pytest.approx(-0.009271420063273115)
    assert planner.bytes_for_score_delta(0.001) == math.ceil(
        0.001 / planner.RATE_POINTS_PER_BYTE
    )


def test_public_report_parser_recomputes_reported_cuda_claim() -> None:
    body = """
    Average PoseNet Distortion: 0.00018963
    Average SegNet Distortion: 0.00057675
    Submission file size: 236,516 bytes
    Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.26
    """

    report = planner.parse_report_text(body)

    assert report["archive_bytes"] == 236_516
    assert report["pose_dist"] == 0.00018963
    assert report["seg_dist"] == 0.00057675
    assert report["reported_final_score"] == 0.26
    assert report["score_recomputed_from_report_components"] == (
        planner.score_from_components(
            archive_bytes=236_516,
            seg_dist=0.00057675,
            pose_dist=0.00018963,
        )
    )


def test_public_pr_classification_marks_devices_and_loopholes() -> None:
    anchor = planner.anchor_from_exact_eval(_anchor_payload())
    pr92 = {
        "number": 92,
        "title": "qzs3_range_joint_r258 (0.26)",
        "body": "device: cuda\nAverage PoseNet Distortion: 0.00018963\n"
        "Average SegNet Distortion: 0.00057675\nSubmission file size: 236,516 bytes",
        "state": "open",
        "head": {"sha": "abc"},
        "user": {"login": "nick"},
    }
    pr87 = {
        "number": 87,
        "title": "Add 100_bytes submission",
        "body": "dummy archive with base85 payload in inflate.py",
        "state": "closed",
        "head": {"sha": "def"},
        "user": {"login": "manthedan"},
    }

    row92 = planner.public_pr_row(
        pr92,
        files=[{"filename": "submissions/qzs3_range_joint_r258/range_mask_codec.cpp"}],
        anchor=anchor,
    )
    row87 = planner.public_pr_row(
        pr87,
        files=[{"filename": "submissions/100_bytes/inflate.py"}],
        anchor=anchor,
    )

    assert "qzs3_range_joint_sidechannel" in row92["families"]
    assert row92["score_device_class"] == "public_report_cuda_unverified"
    assert row92["score_claim_status"] == "external_until_local_exact_cuda_replay"
    assert row92["delta_vs_anchor"]["archive_bytes"] == 6_760
    assert "source_embedded_payload_loophole" in row87["families"]
    assert row87["score_claim_status"] == "invalid_external_loophole"


def test_primary_pr_text_dominates_neighboring_submission_files() -> None:
    anchor = planner.anchor_from_exact_eval(_anchor_payload())
    pr94 = {
        "number": 94,
        "title": "optimization_qpose_josema",
        "body": "qpose/tile-action based variant. Evaluated locally on Mac MPS.",
        "state": "open",
        "head": {"sha": "qpose"},
        "user": {"login": "jose"},
    }

    row94 = planner.public_pr_row(
        pr94,
        files=[
            {"filename": "submissions/hpac_coder_hybrid/pr86_hpac.py"},
            {"filename": "submissions/hpac_coder_hybrid/inflate.py"},
            {"filename": "submissions/optimization_qpose/inflate.py"},
        ],
        anchor=anchor,
    )

    assert "pose_manifold_qpose_action" in row94["families"]
    assert "hpm1_hpac_mask_entropy" not in row94["families"]


def test_build_plan_outputs_ranked_top5_countermoves() -> None:
    public_prs = [
        {
            "number": 91,
            "title": "Hpac coder hybrid",
            "body": "PoseNet: `0.00018940`\nSegNet: `0.00057185`\narchive bytes: `222404`",
            "state": "open",
            "head": {"sha": "hpm1"},
            "user": {"login": "ottokunkel"},
        },
        {
            "number": 92,
            "title": "qzs3_range_joint_r258 (0.26)",
            "body": "device: cuda\nAverage PoseNet Distortion: 0.00018963\n"
            "Average SegNet Distortion: 0.00057675\nSubmission file size: 236,516 bytes",
            "state": "open",
            "head": {"sha": "qzs3"},
            "user": {"login": "nick"},
        },
        {
            "number": 94,
            "title": "optimization_qpose_josema",
            "body": "device: mps\nAverage PoseNet Distortion: 0.00061985\n"
            "Average SegNet Distortion: 0.00071020\nSubmission file size: 277,087 bytes",
            "state": "open",
            "head": {"sha": "pose"},
            "user": {"login": "jose"},
        },
    ]

    plan = planner.build_plan(anchor_payload=_anchor_payload(), public_prs=public_prs)

    assert plan["schema"] == planner.SCHEMA
    assert plan["anchor"]["score"] == 0.25369011029397787
    top = plan["anticipated_moves_top5"]
    assert [row["family"] for row in top[:3]] == [
        "hpm1_hpac_mask_entropy",
        "semantic_geometry_mask_recode",
        "qzs3_range_joint_sidechannel",
    ]
    assert top[0]["implementation_status"] == "blocked_on_hpm1_full_decode_reencode_parity"
    assert top[2]["public_prs"] == [92]
    assert any(row["score_device_class"] == "public_report_mps_invalid_for_promotion" for row in plan["public_pr_rows"])
