#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Seal a production G52 codec config against one exact G51 aggregate receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (  # noqa: E402
    FreshScorerPlaneMaterializationError,
    FreshScorerPlaneOperandLoaderV1,
)
from tac.witness_dsl.taskspace_fresh_selected_plane_codec_v1 import (  # noqa: E402
    CONFIG_SCHEMA,
    FreshSelectedPlaneCodecError,
    canonical_json,
    resolve_ffmpeg_binary_identity,
    validate_config,
    write_once_or_equal,
)

_CODECS = {
    "x264rgb": {
        "encoder": "libx264rgb",
        "container": "h264",
        "encoded_pixel_format": "rgb24",
    },
    "x265-444": {
        "encoder": "libx265",
        "container": "hevc",
        "encoded_pixel_format": "yuv444p",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operand-receipt", type=Path, required=True)
    parser.add_argument("--operand-receipt-sha256", required=True)
    parser.add_argument("--encoder-contract", choices=sorted(_CODECS), required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--base-bitrate-bps", type=int, required=True)
    parser.add_argument("--enhancement-bitrate-bps", type=int, required=True)
    parser.add_argument("--required-free-bytes", type=int, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    return parser


def build_config(
    *,
    operand_receipt: Path,
    operand_receipt_sha256: str,
    encoder_contract: str,
    preset: str,
    endpoint_name: str,
    base_bitrate_bps: int,
    enhancement_bitrate_bps: int,
    required_free_bytes: int,
) -> dict[str, object]:
    try:
        codec = _CODECS[encoder_contract]
    except KeyError as exc:
        raise FreshSelectedPlaneCodecError("unknown encoder contract") from exc
    ffmpeg_identity = resolve_ffmpeg_binary_identity("ffmpeg")
    config: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "research_only": True,
        "candidate_lineage_allowed": True,
        "historical_payload_reused": False,
        "pair_count": 600,
        "pairs_per_stage": 120,
        "stage_count": 5,
        "geometry": {"height": 384, "width": 512, "channels": 3},
        "representation": {
            "mode": "DIRECT_TASK_LAYERED",
            "program_residual_layered_available": False,
            "program_residual_layered_blocker": (
                "fresh semantic predictor/base bytes are absent from the G51 provider"
            ),
        },
        "operand_provider": {
            "aggregate_receipt_path": str(operand_receipt.resolve()),
            "aggregate_receipt_sha256": operand_receipt_sha256,
        },
        "codec": {
            **ffmpeg_identity,
            **codec,
            "input_pixel_format": "rgb24",
            "decoded_pixel_format": "rgb24",
            "frame_rate": 20,
            "preset": preset,
            "threads": 1,
            "color_range": "pc",
            "colorspace": "bt709",
            "color_primaries": "bt709",
            "color_transfer": "bt709",
        },
        "endpoint": {
            "name": endpoint_name,
            "base_bitrate_bps": base_bitrate_bps,
            "enhancement_bitrate_bps": enhancement_bitrate_bps,
        },
        "required_free_bytes": required_free_bytes,
        "test_only_small_fixture": False,
    }
    validate_config(config)
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        FreshScorerPlaneOperandLoaderV1.open(
            args.operand_receipt,
            expected_sha256=args.operand_receipt_sha256,
        )
        config = build_config(
            operand_receipt=args.operand_receipt,
            operand_receipt_sha256=args.operand_receipt_sha256,
            encoder_contract=args.encoder_contract,
            preset=args.preset,
            endpoint_name=args.endpoint_name,
            base_bitrate_bps=args.base_bitrate_bps,
            enhancement_bitrate_bps=args.enhancement_bitrate_bps,
            required_free_bytes=args.required_free_bytes,
        )
        write_once_or_equal(args.output_config, canonical_json(config))
    except (FreshScorerPlaneMaterializationError, FreshSelectedPlaneCodecError, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "sealed",
                "output_config": str(args.output_config.resolve()),
                "encoder_contract": args.encoder_contract,
                "operand_receipt_sha256": args.operand_receipt_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
