from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = REPO / "experiments" / "preflight_renderer_transplant_pose_safety.py"


def _load_preflight() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_pose_safety_preflight_test",
        PREFLIGHT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transplant_contract_requires_masks_and_poses_to_match() -> None:
    preflight = _load_preflight()
    source = {
        "renderer.bin": b"QZS3source",
        "masks.mkv": b"same-masks",
        "optimized_poses.bin": b"same-poses",
    }
    candidate = {
        "renderer.bin": b"QBF1candidate",
        "masks.mkv": b"different-masks",
        "optimized_poses.bin": b"same-poses",
    }

    contract = preflight.validate_transplant_contract(source, candidate)

    assert contract["ok"] is False
    assert "mask_payload_changed" in contract["failures"]
    assert contract["comparisons"]["renderer.bin"]["same_bytes"] is False
    assert contract["comparisons"]["optimized_poses.bin"]["same_bytes"] is True


def test_transplant_contract_rejects_source_renderer_surrogate() -> None:
    preflight = _load_preflight()
    members = {
        "renderer.bin": b"QZS3same",
        "masks.mkv": b"same-masks",
        "optimized_poses.bin": b"same-poses",
    }

    contract = preflight.validate_transplant_contract(members, dict(members))

    assert contract["ok"] is False
    assert "renderer_payload_unchanged_or_surrogate" in contract["failures"]


def test_transplant_contract_accepts_qp1_pose_and_unchanged_actions() -> None:
    preflight = _load_preflight()
    source = {
        "renderer.bin": b"QZS3source",
        "masks.mkv": b"same-masks",
        "optimized_poses.qp1": b"QP1same-poses",
        "seg_tile_actions.bin": b"same-actions",
    }
    candidate = {
        **source,
        "renderer.bin": b"QZS3candidate",
    }

    contract = preflight.validate_transplant_contract(source, candidate)

    assert contract["ok"] is True
    assert contract["source_pose_members"] == ["optimized_poses.qp1"]
    assert contract["comparisons"]["seg_tile_actions.bin"]["same_bytes"] is True


def test_transplant_contract_rejects_changed_aux_payload() -> None:
    preflight = _load_preflight()
    source = {
        "renderer.bin": b"QZS3source",
        "masks.mkv": b"same-masks",
        "optimized_poses.qp1": b"QP1same-poses",
        "seg_tile_actions.bin": b"same-actions",
    }
    candidate = {
        **source,
        "renderer.bin": b"QZS3candidate",
        "seg_tile_actions.bin": b"different-actions",
    }

    contract = preflight.validate_transplant_contract(source, candidate)

    assert contract["ok"] is False
    assert "aux_payload_changed" in contract["failures"]


def test_frame_batch_parity_accepts_identical_outputs() -> None:
    preflight = _load_preflight()
    frames = torch.arange(2 * 2 * 4 * 4 * 3, dtype=torch.float32).reshape(
        2, 2, 4, 4, 3
    )

    comparison = preflight.compare_frame_batches(
        frames,
        frames.clone(),
        max_mean_abs_delta=0.0,
        max_rms_delta=0.0,
        max_max_abs_delta=0.0,
    )

    assert comparison["ok"] is True
    assert comparison["same_uint8_hash"] is True
    assert comparison["mean_abs_delta"] == 0.0
    assert comparison["rms_delta"] == 0.0
    assert comparison["max_abs_delta"] == 0.0


def test_frame_batch_parity_fails_closed_on_large_render_delta() -> None:
    preflight = _load_preflight()
    source = torch.zeros((1, 2, 4, 4, 3), dtype=torch.float32)
    candidate = source.clone()
    candidate[:, 0] += 32.0

    comparison = preflight.compare_frame_batches(
        source,
        candidate,
        max_mean_abs_delta=3.0,
        max_rms_delta=8.0,
        max_max_abs_delta=80.0,
    )

    assert comparison["ok"] is False
    assert comparison["same_uint8_hash"] is False
    assert "mean_abs_delta_exceeds_threshold" in comparison["failures"]
    assert "rms_delta_exceeds_threshold" in comparison["failures"]


def test_default_frame_batch_parity_is_strict_for_frontier_transplants() -> None:
    preflight = _load_preflight()
    source = torch.zeros((1, 2, 4, 4, 3), dtype=torch.float32)
    candidate = source + 0.06

    comparison = preflight.compare_frame_batches(
        source,
        candidate,
        max_mean_abs_delta=preflight.DEFAULT_MAX_MEAN_ABS_DELTA,
        max_rms_delta=preflight.DEFAULT_MAX_RMS_DELTA,
        max_max_abs_delta=preflight.DEFAULT_MAX_MAX_ABS_DELTA,
    )

    assert comparison["ok"] is False
    assert "mean_abs_delta_exceeds_threshold" in comparison["failures"]
