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
    def fake_response(**kwargs):
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
        batch_pairs_values=[1, 2],
        device_values=["cpu"],
        start_pair=8,
        max_pairs=4,
    )

    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["candidate_generation_only"] is True
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["pair_window"] == [8, 12]
    assert payload["best"]["batch_pairs"] == 2
    assert payload["best"]["pairs_per_second"] == 2.0
