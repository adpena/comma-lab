# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tac.local_acceleration.mlx_yuv6_primitive_parity import (
    PASS_VERDICT,
    build_mlx_yuv6_primitive_parity_manifest,
    deterministic_rgb_fixture,
)

REPO = Path(__file__).resolve().parents[3]


def test_mlx_yuv6_matches_upstream_frame_utils() -> None:
    rgb = deterministic_rgb_fixture(seed=17, batch=2, height=18, width=20)

    manifest = build_mlx_yuv6_primitive_parity_manifest(
        rgb_chw=rgb,
        repo_root=REPO,
        epsilon=1.0e-5,
        run_id="fixture",
    )

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["input_shape"] == [2, 3, 18, 20]
    assert manifest["output_shape"] == [2, 6, 9, 10]
    assert manifest["deltas"]["max_abs_delta"] <= 1.0e-5
    assert len(manifest["upstream_output_sha256"]) == 64
    assert len(manifest["mlx_output_sha256"]) == 64


def test_mlx_yuv6_parity_rejects_wrong_channel_axis() -> None:
    rgb = np.zeros((1, 4, 8, 8), dtype=np.float32)

    try:
        build_mlx_yuv6_primitive_parity_manifest(rgb_chw=rgb, repo_root=REPO)
    except ValueError as exc:
        assert "channel axis -3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected wrong-channel input to fail closed")


def test_mlx_yuv6_parity_cli_writes_manifest(tmp_path: Path) -> None:
    output = tmp_path / "yuv6_parity.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_yuv6_primitive_parity.py"),
            "--output",
            str(output),
            "--repo-root",
            str(REPO),
            "--seed",
            "3",
            "--batch",
            "1",
            "--height",
            "16",
            "--width",
            "18",
            "--epsilon",
            "1e-5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    assert stdout_payload["passed"] is True
    assert stdout_payload["score_claim"] is False
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["input_shape"] == [1, 3, 16, 18]
    assert manifest["output_shape"] == [1, 6, 8, 9]
