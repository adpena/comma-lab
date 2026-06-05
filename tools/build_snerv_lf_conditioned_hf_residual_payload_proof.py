#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority LF-conditioned HF residual receiver payload proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import sha256_bytes, write_bytes_artifact, write_json_artifact  # noqa: E402
from tac.substrates.snerv_inverse_steg_carrier.archive import (  # noqa: E402
    decode_snerv_archive_pair_frames,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_conditioned_hf_residual import (  # noqa: E402
    build_lf_conditioned_hf_residual_receiver_proof,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-payload", required=True, type=Path)
    parser.add_argument(
        "--pair-indices",
        default="0,1",
        help="Comma-separated source pair indices to encode into the proof payload.",
    )
    parser.add_argument("--anchor-downsample", default=2, type=int)
    parser.add_argument("--residual-quant-step", default=1.0, type=float)
    parser.add_argument(
        "--unclipped-source",
        action="store_true",
        help="Use unclipped receiver output as the proof source sample.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair_indices = _parse_pair_indices(args.pair_indices)
    packet = args.packet.read_bytes()
    frames = decode_snerv_archive_pair_frames(
        packet,
        pair_indices,
        clip_to_uint8_range=not bool(args.unclipped_source),
    )
    proof, payload = build_lf_conditioned_hf_residual_receiver_proof(
        frames,
        pair_indices=pair_indices,
        packet_path=args.packet.as_posix(),
        source_packet_sha256=sha256_bytes(packet),
        source_clip_to_uint8_range=not bool(args.unclipped_source),
        anchor_downsample=int(args.anchor_downsample),
        residual_quant_step=float(args.residual_quant_step),
        payload_path=args.output_payload.as_posix(),
    )
    payload_result = write_bytes_artifact(args.output_payload, payload)
    proof = {
        **proof,
        "payload_path": payload_result.path,
        "payload_bytes": payload_result.bytes_written,
        "payload_sha256": payload_result.sha256,
    }
    json_result = write_json_artifact(args.output_json, proof)
    print(
        json.dumps(
            {
                "schema": proof["schema"],
                "output_json": json_result.path,
                "output_json_sha256": json_result.sha256,
                "output_payload": payload_result.path,
                "output_payload_sha256": payload_result.sha256,
                "payload_bytes": payload_result.bytes_written,
                "receiver_decode_proven": proof["receiver_decode_proven"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_pair_indices(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("--pair-indices must contain at least one integer")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
