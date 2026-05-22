# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "tools" / "profile_mlx_scorer_response_cache.py"
SPEC = importlib.util.spec_from_file_location("profile_mlx_scorer_response_cache", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)


def test_parse_positive_int_csv_rejects_empty_and_nonpositive() -> None:
    assert profiler.parse_positive_int_csv("1, 2,4", flag_name="--batch-pairs") == [1, 2, 4]
    with pytest.raises(ValueError, match=">= 1"):
        profiler.parse_positive_int_csv("1,0", flag_name="--batch-pairs")
    with pytest.raises(ValueError, match="at least one"):
        profiler.parse_positive_int_csv(" , ", flag_name="--batch-pairs")


def test_parse_device_csv_rejects_unknown_device() -> None:
    assert profiler.parse_device_csv("cpu,gpu") == ["cpu", "gpu"]
    with pytest.raises(ValueError, match="cpu or gpu"):
        profiler.parse_device_csv("mps")


def test_profile_payload_is_non_authoritative_and_selects_best(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_allowances: list[bool] = []

    def fake_response(**kwargs):
        seen_allowances.append(bool(kwargs["allow_gpu_research_signal"]))
        batch_pairs = int(kwargs["batch_pairs"])
        elapsed = 4.0 / batch_pairs
        return {
            "n_samples": 4,
            "start_pair": kwargs["start_pair"],
            "pair_window": [kwargs["start_pair"], kwargs["start_pair"] + kwargs["max_pairs"]],
            "elapsed_seconds": elapsed,
            "canonical_score": 0.2,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.002,
            "components": {
                "posenet_sha256": f"pose-{batch_pairs}",
                "segnet_sha256": f"seg-{batch_pairs}",
            },
        }

    monkeypatch.setattr(profiler, "build_mlx_scorer_response_payload", fake_response)

    payload = profiler.build_profile_payload(
        reference_cache_dir="/tmp/ref",
        candidate_cache_dir="/tmp/cand",
        archive_size_bytes=123,
        repo_root=REPO,
        batch_pairs_values=[1],
        device_values=["cpu"],
        start_pair=8,
        max_pairs=4,
    )

    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["candidate_generation_only"] is True
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["pair_window"] == [8, 12]
    assert payload["best"]["batch_pairs"] == 1
    assert payload["best"]["pairs_per_second"] == 1.0
    assert payload["gpu_research_signal_allowed"] is False
    assert payload["batch_shape_research_signal_allowed"] is False
    assert seen_allowances == [False]


def test_profile_payload_rejects_non_singleton_batch_without_explicit_research_allowance() -> None:
    with pytest.raises(ValueError, match="--allow-batch-shape-research-signal"):
        profiler.build_profile_payload(
            reference_cache_dir="/tmp/ref",
            candidate_cache_dir="/tmp/cand",
            archive_size_bytes=123,
            repo_root=REPO,
            batch_pairs_values=[1, 2],
            device_values=["cpu"],
            start_pair=0,
            max_pairs=1,
        )


def test_profile_payload_rejects_gpu_without_explicit_research_allowance() -> None:
    with pytest.raises(ValueError, match="--allow-gpu-research-signal"):
        profiler.build_profile_payload(
            reference_cache_dir="/tmp/ref",
            candidate_cache_dir="/tmp/cand",
            archive_size_bytes=123,
            repo_root=REPO,
            batch_pairs_values=[1],
            device_values=["gpu"],
            start_pair=0,
            max_pairs=1,
        )


def test_profile_payload_rejects_non_singleton_gpu_batch_after_allowance() -> None:
    with pytest.raises(ValueError, match="--allow-batch-shape-research-signal"):
        profiler.build_profile_payload(
            reference_cache_dir="/tmp/ref",
            candidate_cache_dir="/tmp/cand",
            archive_size_bytes=123,
            repo_root=REPO,
            batch_pairs_values=[1, 2],
            device_values=["gpu"],
            start_pair=0,
            max_pairs=1,
            allow_gpu_research_signal=True,
        )


def test_profile_payload_passes_explicit_gpu_research_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_allowances: list[bool] = []

    def fake_response(**kwargs):
        seen_allowances.append(bool(kwargs["allow_gpu_research_signal"]))
        return {
            "n_samples": 1,
            "start_pair": kwargs["start_pair"],
            "pair_window": [kwargs["start_pair"], kwargs["start_pair"] + kwargs["max_pairs"]],
            "elapsed_seconds": 1.0,
            "canonical_score": 0.2,
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.002,
            "components": {
                "posenet_sha256": "p" * 64,
                "segnet_sha256": "s" * 64,
            },
        }

    monkeypatch.setattr(profiler, "build_mlx_scorer_response_payload", fake_response)

    payload = profiler.build_profile_payload(
        reference_cache_dir="/tmp/ref",
        candidate_cache_dir="/tmp/cand",
        archive_size_bytes=123,
        repo_root=REPO,
        batch_pairs_values=[1],
        device_values=["gpu"],
        start_pair=0,
        max_pairs=1,
        allow_gpu_research_signal=True,
    )

    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["device_values"] == ["gpu"]
    assert payload["gpu_research_signal_allowed"] is True
    assert seen_allowances == [True]
