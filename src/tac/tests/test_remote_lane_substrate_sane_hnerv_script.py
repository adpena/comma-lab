# SPDX-License-Identifier: MIT
"""Regression tests for the sane_hnerv Modal remote driver."""

from __future__ import annotations

from pathlib import Path


def test_sane_hnerv_remote_driver_surfaces_canonical_auth_eval_json() -> None:
    text = Path("scripts/remote_lane_substrate_sane_hnerv.sh").read_text(
        encoding="utf-8"
    )

    assert 'AUTH_EVAL_JSON="$OUTPUT_DIR/contest_auth_eval_cuda.json"' in text
    assert '[ -f "$OUTPUT_DIR/auth_eval.json" ]' in text
    assert "LANE_SANE_HNERV_DONE [contest-CUDA]" in text

