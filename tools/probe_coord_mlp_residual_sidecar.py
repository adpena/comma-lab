#!/usr/bin/env python3
"""Emit a proxy-safe H15 Coord-MLP residual sidecar smoke manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tac.substrates.coord_mlp_residual_sidecar import (
    CoordMlpPatch,
    CoordMlpResidualWeights,
    apply_sidecar_to_rgb,
    build_readiness_manifest,
    pack_sidecar,
)


def build_demo_sidecar() -> bytes:
    """Build a deterministic non-noop demo sidecar for local smoke probes."""

    weights = CoordMlpResidualWeights(
        w1_int8=np.array([[32, 0, 0], [0, 32, 32]], dtype=np.int8),
        b1_int16=np.array([0, 0], dtype=np.int16),
        w2_int8=np.array([[0, 0], [16, 0], [0, 16]], dtype=np.int8),
        b2_int16=np.array([16, 0, 0], dtype=np.int16),
    )
    return pack_sidecar(
        (
            CoordMlpPatch(frame_index=0, y=1, x=1, height=4, width=4),
            CoordMlpPatch(frame_index=1, y=2, x=2, height=3, width=3),
        ),
        weights,
        metadata={
            "campaign_id": "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
            "probe": "coord_mlp_residual_sidecar_local_smoke",
        },
    )


def build_probe_manifest() -> dict[str, object]:
    """Return a combined readiness + consumed-byte smoke manifest."""

    sidecar = build_demo_sidecar()
    base_frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    result = apply_sidecar_to_rgb(base_frames, sidecar)
    readiness = build_readiness_manifest(sidecar)
    return {
        "schema": "coord_mlp_residual_sidecar_probe_v1",
        "campaign_id": "h15_coord_mlp_residual_sidecar_pr103_on_pr106",
        "readiness": readiness,
        "inflate_consumption": result.to_manifest(),
        "frames_changed": bool(np.any(result.frames != base_frames)),
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/cooperative_receiver/coord_mlp_residual_sidecar_probe.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_probe_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote coord_mlp_residual_sidecar_probe "
        f"bytes={manifest['readiness']['charged_bytes']} "
        f"changed={str(manifest['frames_changed']).lower()} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_demo_sidecar",
    "build_parser",
    "build_probe_manifest",
    "main",
]
