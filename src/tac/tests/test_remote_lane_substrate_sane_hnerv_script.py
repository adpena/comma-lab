# SPDX-License-Identifier: MIT
"""Regression tests for the sane_hnerv Modal remote driver."""

from __future__ import annotations

from pathlib import Path


def test_sane_hnerv_remote_driver_surfaces_canonical_auth_eval_json() -> None:
    text = Path("scripts/remote_lane_substrate_sane_hnerv.sh").read_text(
        encoding="utf-8"
    )

    assert 'AUTH_EVAL_JSON="$SANE_HNERV_OUTPUT_DIR/contest_auth_eval_cuda.json"' in text
    assert '[ -f "$SANE_HNERV_OUTPUT_DIR/auth_eval.json" ]' in text
    assert 'ARCHIVE_PATH="$SANE_HNERV_OUTPUT_DIR/0.bin"' in text


def test_sane_hnerv_remote_driver_does_not_label_fallback_json_cuda() -> None:
    text = Path("scripts/remote_lane_substrate_sane_hnerv.sh").read_text(
        encoding="utf-8"
    )

    assert 'payload.get("score_axis") == "contest_cuda"' in text
    assert 'payload.get("score_claim_valid") is True' in text
    assert 'payload.get("exact_cuda_eval_complete") is True' in text
    assert 'AUTH_EVAL_TAG="[training-artifact]"' in text
    assert "LANE_SANE_HNERV_DONE [contest-CUDA]" not in text
