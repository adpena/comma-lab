from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/analyze_pr90_qma9_mask_prior_transfer.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("analyze_pr90_qma9_mask_prior_transfer_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def test_parse_pr90_qtbm_blob_extracts_component_slices() -> None:
    top_payload = b"top"
    road_payload = b"road"
    spatial_table = b"aa"
    m5_table = b"bbb"
    sparse_table = b"s"
    bitstream = b"12345"
    blob = (
        b"QTBM5\0"
        + struct.pack("<HHHBBBbb", 2, 4, 5, 12, 3, 7, -1, 2)
        + bytes([3, 2, 1, 0])
        + struct.pack("<II", len(top_payload), len(road_payload))
        + top_payload
        + road_payload
        + struct.pack("<HH", len(spatial_table), len(m5_table))
        + spatial_table
        + m5_table
        + bytes([2, 30, 31])
        + struct.pack("<HI", 256, len(sparse_table))
        + sparse_table
        + struct.pack("<I", len(bitstream))
        + bitstream
    )

    parsed = module.parse_pr90_qtbm_blob(blob)

    assert parsed["magic"] == "QTBM5\x00"
    assert parsed["n_pairs"] == 2
    assert parsed["height"] == 4
    assert parsed["width"] == 5
    assert parsed["precision"] == 12
    assert parsed["top_payload"]["bytes"] == 3
    assert parsed["road_payload"]["bytes"] == 4
    assert parsed["spatial_table"]["bytes"] == 2
    assert parsed["m5_table"]["bytes"] == 3
    assert parsed["sparse_features"] == {"count": 2, "ids": [30, 31], "threshold_q8": 256}
    assert parsed["sparse_table"]["bytes"] == 1
    assert parsed["residual_bitstream"]["bytes"] == 5
    assert parsed["_top_payload_bytes"] == top_payload
    assert parsed["_road_payload_bytes"] == road_payload


def _minimal_pr90_mask() -> dict:
    return {
        "compact_split": {
            "mask_body_bytes": 152_431,
        },
        "support_masks": {
            "top_support_pixels": 58_413_695,
            "road_non_top_pixels": 29_992_748,
            "residual_coded_pixels_after_support": 29_558_357,
        },
    }


def _minimal_pr85_mask() -> dict:
    return {
        "archive": {"archive_bytes": 236_328},
        "mask_segment": {"bytes": 159_011},
    }


def test_ranked_policy_marks_lossless_recode_archive_changing_and_runtime_blocked() -> None:
    parity = {
        "decoded_mask_equal": True,
        "diff_pixels": 0,
        "pr90_render_order_sha256": "same",
        "pr85_render_order_sha256": "same",
        "render_order_shape": [600, 384, 512],
    }

    policy = module.build_ranked_policy(
        pr90_mask=_minimal_pr90_mask(),
        pr85_mask=_minimal_pr85_mask(),
        parity=parity,
    )

    assert policy["planning_only"] is True
    assert policy["score_claim"] is False
    assert policy["dispatch_performed"] is False
    assert policy["decision"]["implementable_next_archive_builder"] is True
    assert policy["decision"]["fixed_pr85_runtime_preserved"] is False
    candidate = policy["ranked_candidates"][0]
    assert candidate["policy_id"] == "pr90_stbm1br_lossless_pr85_mask_recode"
    assert candidate["no_op_status"]["no_op"] is False
    assert candidate["no_op_status"]["archive_changing"] is True
    byte = candidate["charged_byte_estimate"]
    assert byte["candidate_mask_segment_bytes"] == 152_439
    assert byte["delta_mask_segment_bytes"] == -6_572
    assert byte["estimated_archive_bytes_if_only_mask_segment_changes"] == 229_756
    assert math.isclose(
        byte["rate_score_delta_if_components_unchanged"],
        -6_572 * module.RATE_SCORE_PER_BYTE,
    )
    blockers = {item["blocker_class"] for item in candidate["blockers"]}
    assert "fixed_pr85_runtime_does_not_decode_stbm1br" in blockers
    assert any("STBM1BR" in requirement for requirement in candidate["exact_builder_requirements"])


def test_ranked_policy_fails_closed_when_decoded_masks_differ() -> None:
    parity = {
        "decoded_mask_equal": False,
        "diff_pixels": 12,
        "pr90_render_order_sha256": "pr90",
        "pr85_render_order_sha256": "pr85",
        "render_order_shape": [600, 384, 512],
    }

    policy = module.build_ranked_policy(
        pr90_mask=_minimal_pr90_mask(),
        pr85_mask=_minimal_pr85_mask(),
        parity=parity,
    )

    assert policy["decision"]["status"] == "fail_closed_co_trained_or_mismatched_mask"
    assert policy["decision"]["implementable_next_archive_builder"] is False
    candidate = policy["ranked_candidates"][0]
    assert candidate["status"] == "fail_closed_not_buildable"
    assert candidate["no_op_status"]["no_op"] is True
    assert candidate["blockers"][0]["blocker_class"] == "decoded_mask_mismatch"
    assert candidate["blockers"][0]["status"] == "fail_closed"
