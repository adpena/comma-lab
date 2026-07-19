#!/usr/bin/env python3
"""Probe the PDW2 spatial receiver on read-only quotient fields.

Reads a strict PDW2 packet and a quotient `.npy` field, supports pair-cap
streams (default `24,600`), records partition and field hashes without writing label
arrays, records peak RSS, and emits a canonical JSON only report that includes the
packet-only non-identifiability witness and a canonical coefficient-canary.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.insert(0, str(REPO))

from tac.boundary_math.pdw2_spatial_receiver import (  # noqa: E402
    PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
    build_pdw2_coefficient_only_nonidentifiability_witness,
    detect_pdw2_packet_mutation_canary,
    mutate_pdw2_packet_first_relative_coefficient,
    run_pdw2_spatial_receiver,
)
from tac.boundary_math.power_diagram_witness import decode_pdw2  # noqa: E402

SCHEMA = "pdw2_spatial_receiver_probe.v1"


def _parse_pair_caps(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        cap = int(token)
        if cap <= 0:
            raise ValueError(f"pair cap must be positive: {cap}")
        if cap > 600:
            raise ValueError(f"pair cap must be <=600 for PDW2 receiver streams: {cap}")
        if cap not in seen:
            values.append(cap)
            seen.add(cap)
    if not values:
        raise ValueError("--pair-caps must contain at least one integer")
    return tuple(sorted(values))


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return int(usage * 1024)
    return int(usage)


def _validate_packet(packet: Path) -> bytes:
    packet_bytes = packet.read_bytes()
    target = decode_pdw2(packet_bytes)
    from tac.boundary_math.power_diagram_witness import encode_pdw2

    if encode_pdw2(target) != packet_bytes:
        raise ValueError("packet bytes are not strict canonical PDW2/PDP2")
    return packet_bytes


def _load_quotient_field(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if not isinstance(arr, np.memmap):
        raise ValueError("quotient field must be memmap-backed for stream-safe probe mode")
    if arr.dtype != np.float32:
        raise ValueError("quotient field must be float32")
    if arr.ndim != 4:
        raise ValueError("quotient field must be [N, 384, 512, 4]")
    if arr.shape[1:] != (384, 512, 4):
        raise ValueError("quotient field geometry must be [N, 384, 512, 4]")
    if arr.shape[0] < 2:
        raise ValueError("quotient field must contain at least two pairs for this probe")
    if arr.shape[0] < 24:
        raise ValueError("quotient field pair count must support at least 24 pairs")
    return arr


def _mutation_with_evidence(packet: bytes, field: np.ndarray) -> dict[str, object]:
    deltas = [1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0]
    for delta in deltas:
        try:
            mutated = mutate_pdw2_packet_first_relative_coefficient(packet, delta)
        except Exception:
            continue
        canary = detect_pdw2_packet_mutation_canary(packet, mutated, field)
        if canary.get("mutation_observed", False):
            return canary | {"mutation_delta": delta}
    for i in range(2, 13):
        scale = float(2 ** i)
        for sign in (1.0, -1.0):
            delta = sign * scale
            try:
                mutated = mutate_pdw2_packet_first_relative_coefficient(packet, delta)
            except Exception:
                continue
            canary = detect_pdw2_packet_mutation_canary(packet, mutated, field)
            if canary.get("mutation_observed", False):
                return canary | {"mutation_delta": delta}
    raise RuntimeError("failed to find a canonical packet mutation that changes the partition")


def run_probe(packet_path: Path, quotient_path: Path, pair_caps: tuple[int, ...]) -> dict[str, object]:
    packet = _validate_packet(packet_path)
    _ = decode_pdw2(packet)
    field = _load_quotient_field(quotient_path)

    max_cap = max(pair_caps)
    if field.shape[0] < max_cap:
        raise ValueError(
            f"field has only {field.shape[0]} pairs, cannot satisfy requested max pair cap {max_cap}"
        )

    pair_cap_receipts: list[dict[str, object]] = []
    for cap in pair_caps:
        receipt = run_pdw2_spatial_receiver(packet, field[:cap], include_labels=False)
        pair_cap_receipts.append(
            {
                "pair_cap": cap,
                "pair_count": receipt["pair_count"],
                "field_sha256": receipt["field_sha256"],
                "partition_shape": receipt["partition_shape"],
                "partition_labels_sha256": receipt["partition_labels_sha256"],
                "partition_label_counts": receipt["partition_label_counts"],
                "mapped_page_eviction_applied": receipt["mapped_page_eviction_applied"],
            }
        )

    witness = build_pdw2_coefficient_only_nonidentifiability_witness(packet)
    # A real-field canary only needs one bounded stream. Keeping this at n24
    # prevents a search over mutation magnitudes from multiplying the n600
    # measurement cost while still proving that packet bytes are consumed.
    canary = _mutation_with_evidence(packet, field[: min(24, max_cap)])

    return {
        "schema": SCHEMA,
        "pdw2_packet_sha256": witness["pdw2_packet_sha256"],
        "pdw2_promotion_blocker": PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
        "packet_to_partition_consumed": True,
        "coefficient_only_through_r_equivalent": False,
        "through_r_authority": False,
        "d_seg": None,
        "d_pose": None,
        "score_claim": False,
        "promotion_eligible": False,
        "pair_caps": list(pair_caps),
        "pair_cap_receipts": pair_cap_receipts,
        "witness": witness,
        "packet_mutation_canary": canary,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="Path to strict PDW2/PDP2 packet.")
    parser.add_argument("quotient_npy", type=Path, help="Path to quotient float32 memmap .npy")
    parser.add_argument(
        "--pair-caps",
        default="24,600",
        dest="pair_caps",
        help="Comma-separated pair caps to evaluate (default: 24,600)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, prints to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pair_caps = _parse_pair_caps(args.pair_caps)
        payload = run_probe(args.packet, args.quotient_npy, pair_caps)
    except Exception as exc:  # pragma: no cover - CLI validation path
        print(f"[pdw2-spatial-receiver-probe] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        out = args.output
        out_str = str(out)
        if out_str.startswith("/tmp/") or "/private/tmp/" in out_str or "/var/tmp/" in out_str:
            print(f"[pdw2-spatial-receiver-probe] REFUSE output path on tempfs: {out_str}", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
