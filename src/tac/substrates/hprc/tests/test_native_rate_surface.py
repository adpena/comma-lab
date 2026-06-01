# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tac.substrates.hprc.native_rate_surface import (
    HPRC_NATIVE_RATE_RESIDUAL_PROTECTION_SURFACE_SCHEMA,
    build_hprc_native_rate_residual_protection_surface,
)
from tools import build_hprc_native_rate_surface as tool


def _p19() -> dict:
    return {
        "schema": "p19_posenet_null_pair_detection.v1",
        "n_pairs": 3,
        "selected_pair_ids": [1],
        "score_claim": False,
        "promotion_eligible": False,
    }


def _p18() -> dict:
    return {
        "schema": "p18_segnet_region_waterfill.v1",
        "n_pairs_available": 3,
        "rows": [
            {
                "pair_id": 1,
                "regions256": [
                    {
                        "region_id": 0,
                        "box": {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 0.5},
                    }
                ],
            }
        ],
        "score_claim": False,
        "promotion_eligible": False,
    }


def test_hprc_native_rate_surface_maps_p19_null_and_p18_regions() -> None:
    surface, manifest = build_hprc_native_rate_residual_protection_surface(
        p19_posenet_null_pairs=_p19(),
        p18_segnet_region_waterfill=_p18(),
        frames=6,
        residual_grid_h=4,
        residual_grid_w=4,
        p19_null_protection=0.2,
        p18_region_protection=1.0,
    )

    assert manifest["schema"] == HPRC_NATIVE_RATE_RESIDUAL_PROTECTION_SURFACE_SCHEMA
    assert manifest["evidence_scope"] == "full_video"
    assert manifest["blockers"] == []
    assert surface.shape == (6, 4, 4, 3)
    assert np.all(surface[0] == 1.0)
    assert np.all(surface[2, 2:, :, :] == 0.2)
    assert np.all(surface[3, :2, :2, :] == 1.0)


def test_hprc_native_rate_surface_cli_writes_npy_and_manifest(tmp_path: Path) -> None:
    p19 = tmp_path / "p19.json"
    p18 = tmp_path / "p18.json"
    output_npy = tmp_path / "surface.npy"
    output_json = tmp_path / "surface.json"
    p19.write_text(json.dumps(_p19()), encoding="utf-8")
    p18.write_text(json.dumps(_p18()), encoding="utf-8")

    assert (
        tool.main(
            [
                "--p19-posenet-null-pairs",
                p19.as_posix(),
                "--p18-segnet-region-waterfill",
                p18.as_posix(),
                "--frames",
                "6",
                "--residual-grid-h",
                "4",
                "--residual-grid-w",
                "4",
                "--output-npy",
                output_npy.as_posix(),
                "--out-json",
                output_json.as_posix(),
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    surface = np.load(output_npy)
    assert payload["schema"] == HPRC_NATIVE_RATE_RESIDUAL_PROTECTION_SURFACE_SCHEMA
    assert payload["output_npy"]["bytes"] == output_npy.stat().st_size
    assert surface.shape == (6, 4, 4, 3)
    assert payload["score_claim"] is False
