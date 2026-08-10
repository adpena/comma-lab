"""Executed controls for ddm_pr1's live coder-race payload retrofits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments import ddm_hp1_learned_ar_prior_race as hp1
from experiments import ddm_tk1_semantic_stream_race as tk1


def _assert_retained(row: dict[str, object]) -> None:
    path = Path(str(row["path"]))
    payload = path.read_bytes()
    assert len(payload) == int(row["bytes"])
    assert hashlib.sha256(payload).hexdigest() == row["sha256"]


def test_tk1_generic_race_retains_raw_and_every_coded_candidate(tmp_path: Path) -> None:
    labels = np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8) % tk1.LEVELS
    result = tk1.generic_baselines(
        labels,
        stream_name="control",
        retained_dir=tmp_path / "generic",
    )
    _assert_retained(result["raw_uint8_retained_payload"])
    for name in ("lzma1_x9e", "zlib_9", "bz2_9"):
        _assert_retained(result[name]["retained_payload"])
    if result["brotli_11"]["bytes"] is not None:
        _assert_retained(result["brotli_11"]["retained_payload"])


def test_tk1_estimate_control_retains_models_and_subset_frame(tmp_path: Path) -> None:
    labels = np.zeros((4, 8, 8), dtype=np.uint8)
    labels[1:, 2:6, 2:6] = 1
    result = tk1.learned_prior_race(
        labels,
        ssd_dir=tmp_path,
        stream_name="control",
        context_modes=("prev", "prev_left_up"),
        patch=4,
        full_range=False,
    )
    for row in result["raced_context_modes"]:
        if not row["skipped"]:
            _assert_retained(row["model_retained_payload"])
    _assert_retained(result["best"]["subset_retained_payload"])


def test_hp1_run_retains_all_learned_and_baseline_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rng = np.random.default_rng(20260809)
    codes = rng.integers(0, hp1.LEVELS, size=(7, 4, 4, 2), dtype=np.uint8)
    shipped = hp1.forced_lzma_ix2_token_frame(codes)

    def _load(_archive: Path, _expected_sha256: str | None) -> dict[str, object]:
        return {
            "archive_path": "synthetic-control.zip",
            "archive_bytes": len(shipped),
            "archive_sha256": hp1.sha256_bytes(shipped),
            "token_bulk": shipped,
            "token_bulk_bytes": len(shipped),
            "token_bulk_sha256": hp1.sha256_bytes(shipped),
            "codes": codes,
        }

    monkeypatch.setattr(hp1, "load_live_token_stream", _load)
    result = hp1.run_hp1(
        archive=tmp_path / "synthetic-control.zip",
        receipt_dir=tmp_path / "receipts",
        ssd_dir=tmp_path / "ssd",
        expected_sha256=None,
        context_rows=37,
        patch=2,
        max_model_bytes=10_000,
    )

    _assert_retained(result["baselines"]["raw_token_frame"])
    for name in (
        "shipped_ix2_brotli_q11",
        "forced_ix2_lzma1",
        "raw_token_frame_lzma1",
        "raw_token_frame_brotli_q11",
    ):
        _assert_retained(result["baselines"][name]["retained_payload"])
    for row in result["learned_prior"]["raced_context_modes"]:
        if not row["skipped"]:
            _assert_retained(row["retained_payload"])
