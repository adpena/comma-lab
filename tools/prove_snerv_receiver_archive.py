#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Emit a scorer-free receiver proof for the SNeRV archive grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (  # noqa: E402
    export_snerv_archive_bound_candidate_package,
)
from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (  # noqa: E402
    build_snerv_receiver_archive_proof,
)


def _default_out() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_receiver_archive_proof_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument(
        "--wavelet",
        default="db2",
        help=(
            "Wavelet for the proof packet. Use 'haar' to exercise the "
            "NumPy-only receiver DWT path."
        ),
    )
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--full-frame-packet",
        action="store_true",
        help="Emit a tiny packet with n_pairs/frames/channels metadata for inflate proof.",
    )
    parser.add_argument(
        "--packet-out",
        default=None,
        help="Optional path to write the tiny receiver archive packet bytes.",
    )
    parser.add_argument(
        "--package-dir",
        default=None,
        help=(
            "Optional directory to emit archive.zip + submission runtime + "
            "receiver proof. Implies --full-frame-packet."
        ),
    )
    parser.add_argument(
        "--retain-package-raw",
        action="store_true",
        help="Keep the generated raw proof output instead of certify-and-delete.",
    )
    parser.add_argument("--package-timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)

    full_frame_packet = bool(args.full_frame_packet or args.package_dir)
    proof, archive = build_snerv_receiver_archive_proof(
        bins=args.bins,
        levels=args.levels,
        wavelet=args.wavelet,
        hw=(args.height, args.width),
        seed=args.seed,
        full_frame_packet=full_frame_packet,
    )
    payload = proof.as_jsonable()
    payload["archive_packet"] = archive.as_jsonable()
    payload["full_frame_packet"] = full_frame_packet
    out_path = Path(args.out) if args.out else _default_out()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    packet_path = Path(args.packet_out) if args.packet_out else out_path.with_suffix(".snar")
    if not packet_path.is_absolute():
        packet_path = REPO_ROOT / packet_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_bytes(archive.packet)
    packet_bytes = packet_path.read_bytes()
    packet_sha256 = hashlib.sha256(packet_bytes).hexdigest()
    payload["archive_packet_path"] = str(packet_path)
    payload["packet_artifact_bytes"] = len(packet_bytes)
    payload["packet_artifact_sha256"] = packet_sha256
    payload["packet_artifact_matches_proof"] = (
        len(packet_bytes) == proof.archive_packet_bytes
        and packet_sha256 == proof.archive_packet_sha256
    )
    if args.package_dir:
        package_dir = Path(args.package_dir)
        if not package_dir.is_absolute():
            package_dir = REPO_ROOT / package_dir
        package = export_snerv_archive_bound_candidate_package(
            packet=archive.packet,
            output_dir=package_dir,
            repo_root=REPO_ROOT,
            retain_receiver_output=bool(args.retain_package_raw),
            receiver_proof_timeout_seconds=int(args.package_timeout_seconds),
        )
        payload["runtime_package"] = package
        payload["runtime_package_dir"] = str(package_dir)
        rows = package["archive_bound_candidate_adapter_package"]["candidate_rows"]
        payload["archive_packet_proof_blockers"] = payload.get("blockers", [])
        payload["blockers"] = list(rows[0].get("blockers", [])) if rows else []
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("[SNeRV receiver archive proof] false-authority")
    print(f"  archive bytes: {proof.archive_packet_bytes}")
    print(f"  receiver matches direct: {proof.receiver_matches_direct}")
    print(f"  max abs diff: {proof.max_abs_diff:.6g}")
    print(f"  blockers: {', '.join(payload['blockers'])}")
    print(f"  wrote {out_path}")
    print(f"  wrote packet {packet_path}")
    if args.package_dir:
        print(f"  wrote runtime package {Path(args.package_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
