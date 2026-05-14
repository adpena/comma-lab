# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
COMPARE_PATH = REPO / "experiments" / "compare_component_traces.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("_compare_component_traces_test", COMPARE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_trace(
    path: Path,
    *,
    archive_bytes: int,
    samples: list[tuple[float, float]],
    cuda_device_name: str | None = "Tesla T4",
) -> None:
    payload = {
        "schema_version": 1,
        "score_claim": False,
        "evidence_grade": "diagnostic_component_trace",
        "n_samples": len(samples),
        "archive_size_bytes": archive_bytes,
        "avg_posenet_dist": sum(p for p, _ in samples) / len(samples),
        "avg_segnet_dist": sum(s for _, s in samples) / len(samples),
        "samples": [
            {
                "pair_index": i,
                "video_name": "0.mkv",
                "frame_indices": [2 * i, 2 * i + 1],
                "posenet_dist": pose,
                "segnet_dist": seg,
            }
            for i, (pose, seg) in enumerate(samples)
        ],
        "trace_inputs": {
            "device": "cuda:0",
            "cuda_device_name": cuda_device_name,
            "cuda_device_index": 0,
            "cuda_device_capability": [7, 5] if cuda_device_name == "Tesla T4" else [9, 0],
            "torch_version": "test",
            "torch_cuda_version": "test",
        },
    }
    path.write_text(json.dumps(payload))


def test_compare_trace_pair_splits_rate_pose_and_seg_terms(tmp_path: Path) -> None:
    compare = _load_compare_module()
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.json"
    _write_trace(
        candidate_path,
        archive_bytes=300,
        samples=[(0.010, 0.001), (0.002, 0.030), (0.003, 0.004)],
    )
    _write_trace(
        reference_path,
        archive_bytes=200,
        samples=[(0.001, 0.001), (0.002, 0.002), (0.003, 0.003)],
    )

    payload = compare.build_comparison(
        candidate_label="candidate",
        candidate_trace=candidate_path,
        reference_specs=[("reference", reference_path)],
        top_k=2,
        uncompressed_bytes=1_000,
    )

    delta = payload["best_reference_component_delta"]
    assert payload["score_claim"] is False
    assert payload["candidate"]["archive_size_bytes"] == 300
    assert delta["archive_delta_bytes"] == 100
    assert delta["score_delta_rate_exact"] == 2.5
    assert delta["score_delta_seg_exact"] == pytest.approx(
        100.0 * ((0.001 + 0.030 + 0.004) / 3 - 0.002)
    )
    top_combined = payload["references"][0]["top_excess_combined_samples"]
    assert [row["pair_index"] for row in top_combined] == [1, 0]
    assert top_combined[0]["score_seg_excess_exact"] > top_combined[1]["score_seg_excess_exact"]
    assert payload["allocator_use_allowed"] is True
    assert payload["references"][0]["hardware_comparison"]["status"] == "same_hardware_identity"


def test_cross_hardware_trace_comparison_is_marked_untrusted(tmp_path: Path) -> None:
    compare = _load_compare_module()
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.json"
    _write_trace(
        candidate_path,
        archive_bytes=300,
        samples=[(0.010, 0.001), (0.002, 0.030), (0.003, 0.004)],
        cuda_device_name="Tesla T4",
    )
    _write_trace(
        reference_path,
        archive_bytes=200,
        samples=[(0.001, 0.001), (0.002, 0.002), (0.003, 0.003)],
        cuda_device_name="NVIDIA H100 80GB HBM3",
    )

    payload = compare.build_comparison(
        candidate_label="candidate",
        candidate_trace=candidate_path,
        reference_specs=[("reference", reference_path)],
        top_k=2,
        uncompressed_bytes=1_000,
    )

    assert payload["allocator_use_allowed"] is False
    assert payload["evidence_grade"] == "diagnostic_trace_comparison_hardware_untrusted"
    assert payload["references"][0]["hardware_comparison"]["status"] == "hardware_mismatch"


def test_rejects_promotable_or_non_diagnostic_trace(tmp_path: Path) -> None:
    compare = _load_compare_module()
    bad = tmp_path / "bad.json"
    _write_trace(bad, archive_bytes=1, samples=[(0.001, 0.001)])
    payload = json.loads(bad.read_text())
    payload["score_claim"] = True
    bad.write_text(json.dumps(payload))

    try:
        compare.build_comparison(
            candidate_label="bad",
            candidate_trace=bad,
            reference_specs=[("badref", bad)],
            top_k=1,
            uncompressed_bytes=1_000,
        )
    except ValueError as exc:
        assert "score_claim" in str(exc)
    else:
        raise AssertionError("expected ValueError")
