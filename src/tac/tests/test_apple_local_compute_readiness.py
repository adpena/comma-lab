# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.local_acceleration.apple_local_compute_readiness import (
    APPLE_LOCAL_COMPUTE_READINESS_SCHEMA,
    build_apple_local_compute_readiness,
)
from tools.build_apple_local_compute_readiness import main as tool_main


def test_apple_local_compute_readiness_prefers_mlx_when_metal_ready() -> None:
    report = build_apple_local_compute_readiness(
        backend_overrides={
            "mlx": {
                "available": True,
                "metal_available": True,
                "default_device": "Device(gpu, 0)",
            },
            "torch_mps": {"available": True, "mps_available": True},
            "numpy_accelerate": {
                "available": True,
                "accelerate_framework_present": True,
            },
            "hf_accelerate": {"available": False},
        }
    )

    assert report["schema"] == APPLE_LOCAL_COMPUTE_READINESS_SCHEMA
    assert report["recommended_dev_velocity_backend"] == "mlx_metal"
    assert report["mlx_metal_ready"] is True
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "macos_local_acceleration_false_authority" in report["blockers"]
    assert "mlx_metal_not_ready" not in report["blockers"]


def test_apple_local_compute_readiness_falls_back_to_torch_mps() -> None:
    report = build_apple_local_compute_readiness(
        backend_overrides={
            "mlx": {"available": False, "metal_available": False},
            "torch_mps": {"available": True, "mps_available": True},
            "numpy_accelerate": {
                "available": True,
                "accelerate_framework_present": True,
            },
            "hf_accelerate": {"available": True},
        }
    )

    assert report["recommended_dev_velocity_backend"] == "torch_mps"
    assert "mlx_metal_not_ready" in report["blockers"]
    assert "primary_mlx_metal_backend_not_selected" in report["blockers"]


def test_build_apple_local_compute_readiness_cli_smoke(tmp_path: Path) -> None:
    output_json = tmp_path / "readiness.json"
    output_md = tmp_path / "readiness.md"

    rc = tool_main(
        [
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--substrate-id",
            "hi_nerv",
            "--substrate-id",
            "snerv",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == APPLE_LOCAL_COMPUTE_READINESS_SCHEMA
    assert payload["substrate_ids"] == ["hi_nerv", "snerv"]
    assert payload["score_claim"] is False
    assert "Apple Local Compute Readiness" in output_md.read_text(encoding="utf-8")
