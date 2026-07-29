# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_r7_token_coder_race as race


def test_counted_pool_wrapper_roundtrips_and_refuses_boundary_changes() -> None:
    raws = [b"base" * 17, b"delta" * 31, bytes(range(64))]
    measured = race._measure_pool(
        raws,
        [lambda raw: race._generic_encode(2, raw)] * len(raws),
        [race._generic_decode] * len(raws),
        implementations=["lzma1_extreme"] * len(raws),
        pr130_lesson="test",
    )
    assert measured["parseback_exact"] is True
    assert measured["member_count"] == len(raws)
    assert measured["framed_bytes"] == (
        race.POOL_HEADER.size + len(raws) * race.POOL_LENGTH.size + sum(measured["member_framed_bytes"])
    )

    members = [race._generic_encode(2, raw) for raw in raws]
    frame = race._pool_frame(members)
    assert race._pool_split(frame) == members
    for changed in (frame[:-1], frame + b"\0"):
        with pytest.raises(race.R7RaceError):
            race._pool_split(changed)


def test_generic_lzma_frame_is_bounded_exact_and_canonical() -> None:
    raw = b"bounded-stream" * 101
    frame = race._generic_encode(2, raw)
    assert race._generic_decode(frame) == raw
    for changed in (frame[:-1], frame + b"\0"):
        with pytest.raises(race.R7RaceError):
            race._generic_decode(changed)


def test_stage_snapshot_is_create_or_identical_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stage.json"
    race._immutable_json(path, {"stage": 1, "sha256": "a" * 64})
    first = path.read_bytes()
    race._immutable_json(path, {"stage": 1, "sha256": "a" * 64})
    assert path.read_bytes() == first
    with pytest.raises(race.R7RaceError, match="already differs"):
        race._immutable_json(path, {"stage": 2, "sha256": "b" * 64})
    assert path.read_bytes() == first


def test_entropy_receipt_labels_plugin_proxy_and_packed_base() -> None:
    codes = (np.arange(4 * 3 * 5 * 2, dtype=np.uint16) % 16).astype(np.uint8).reshape(4, 3, 5, 2)
    row = race._token_entropy(codes, 16)
    assert row["mode_base_symbols"] == 30
    assert row["mode_base_packed_bytes"] == 15
    assert "residual_empirical_plugin_prev1_bytes" in row
    bound = row["argmax_equivalence_conditional_bound"]
    assert bound["exact_universal_lower_bound_bytes"] == 0
    assert "ASSUME" in bound["proxy_assumption"]
    assert "not a dump theorem" in bound["proxy_assumption"]


def test_cli_rejects_duplicate_checkpoints_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "race",
            "--checkpoint",
            "/nonexistent/same.npz",
            "--checkpoint",
            "/nonexistent/same.npz",
            "--resume-from",
            ".omx/research/progress.json",
            "--output",
            ".omx/research/receipt.json",
        ],
    )
    with pytest.raises(race.R7RaceError, match="distinct"):
        race.main()


def test_cli_rejects_evidence_targets_outside_research(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "race",
            "--checkpoint",
            "/nonexistent/early.npz",
            "--checkpoint",
            "/nonexistent/latest.npz",
            "--resume-from",
            str(tmp_path / "progress.json"),
            "--output",
            str(tmp_path / "receipt.json"),
        ],
    )
    with pytest.raises(race.R7RaceError, match=r"under \.omx/research"):
        race.main()
