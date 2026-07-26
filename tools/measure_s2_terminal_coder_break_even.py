#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Map terminal-coder economics over the exact coupled Seg/Pose/rate surface.

The C1/S2 bridge establishes that every S2 event removes one Seg mismatch if
the same baseline partition and inverse realization are preserved.  This tool
therefore compares the exact Seg score saved by all events with the exact byte
price of the counted S2 packet and with payload-only lower bounds from common
terminal coders.  Because the same receiver can also move Pose, the result is a
conditional break-even curve rather than an independent Seg/rate gate.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import math
import os
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.s2_partition_seed import decode_partition_seed  # noqa: E402

SCHEMA = "tac.s2_terminal_coder_break_even.v3"
DENOMINATOR_BYTES = 37_545_489
SEG_WEIGHT = 100.0
RATE_WEIGHT = 25.0
_PREFIX = struct.Struct("<4sIII")
DEFAULT_PACKET = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s2_compose_20260721/"
    "partition_seed/s2_partition_event_seed.bin"
)
DEFAULT_BRIDGE = REPO / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "c1_s2_exact_debt_bridge_v4.json"
)
BRIDGE_SCHEMA = "tac.c0b_s2_debt_bridge.v4"


class CoderBreakEvenError(RuntimeError):
    """Fail-closed packet, bridge, coder, or output error."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CoderBreakEvenError("value is not canonical-JSON encodable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CoderBreakEvenError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoderBreakEvenError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CoderBreakEvenError(f"{label} root must be an object")
    return value


def _load_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    """Load one immutable byte snapshot for both semantics and custody."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise CoderBreakEvenError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CoderBreakEvenError(f"{label} root must be an object")
    return value, raw


def _validate_body_hash(payload: Mapping[str, Any], field: str) -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise CoderBreakEvenError(f"{field} is missing or malformed")
    body = {key: value for key, value in payload.items() if key != field}
    if _sha256_bytes(_canonical(body)) != observed:
        raise CoderBreakEvenError(f"{field} differs from its receipt body")


def _extract_validated_raw_event_bytes(packet: bytes) -> tuple[bytes, dict[str, Any]]:
    """Return the exact pre-zlib event bytes after full S2 parse-back."""

    seed = decode_partition_seed(packet)
    if len(packet) < _PREFIX.size + 4:
        raise CoderBreakEvenError("S2 packet is truncated")
    _magic, _version, header_bytes, body_bytes = _PREFIX.unpack_from(packet)
    header_start = _PREFIX.size
    body_start = header_start + header_bytes
    body_end = body_start + body_bytes
    try:
        header = json.loads(packet[header_start:body_start].decode("ascii"))
        raw = zlib.decompress(packet[body_start:body_end])
    except (UnicodeDecodeError, json.JSONDecodeError, zlib.error) as exc:
        raise CoderBreakEvenError("cannot recover validated S2 raw event bytes") from exc
    if (
        not isinstance(header, dict)
        or header.get("event_count") != len(seed.events)
        or header.get("raw_event_bytes") != len(raw)
        or header.get("raw_event_sha256") != _sha256_bytes(raw)
    ):
        raise CoderBreakEvenError("S2 raw-event accounting differs after parse-back")
    return raw, header


def _strict_break_even_byte_cap(*, event_count: int, pair_count: int, height: int, width: int) -> int:
    if any(type(value) is not int or value <= 0 for value in (event_count, pair_count, height, width)):
        raise CoderBreakEvenError("event and geometry counts must be positive exact integers")
    # 25*B/D < 100*E/N  iff  B < 4*D*E/N.  Keep this strict
    # inequality integral so an exactly equal rational boundary is rejected.
    total_sites = pair_count * height * width
    return (4 * DENOMINATOR_BYTES * event_count - 1) // total_sites


def _conditional_pose_threshold(
    *,
    payload_bytes: int,
    baseline_d_pose: float,
    seg_score_saved: float,
) -> float | None:
    """Largest open d_pose-after boundary that can make a payload improve S.

    The return value is an *open* threshold: joint improvement requires
    ``d_pose_after < result``.  ``None`` means even zero Pose debt cannot make
    the supplied bytes beneficial under the all-Seg-events-realized premise.
    """

    if type(payload_bytes) is not int or payload_bytes < 0:
        raise CoderBreakEvenError("payload_bytes must be a nonnegative exact integer")
    if not math.isfinite(baseline_d_pose) or baseline_d_pose < 0.0:
        raise CoderBreakEvenError("baseline_d_pose must be finite and nonnegative")
    rate = RATE_WEIGHT * payload_bytes / DENOMINATOR_BYTES
    pose_radius = math.sqrt(10.0 * baseline_d_pose) + seg_score_saved - rate
    if pose_radius <= 0.0:
        return None
    return pose_radius * pose_radius / 10.0


def _conditional_total_byte_cap(
    *,
    event_count: int,
    pair_count: int,
    height: int,
    width: int,
    baseline_d_pose: float,
    d_pose_after: float,
) -> int:
    """Derived integer byte cap at one declared Pose slice of the joint surface."""

    if not math.isfinite(baseline_d_pose) or baseline_d_pose < 0.0:
        raise CoderBreakEvenError("baseline_d_pose must be finite and nonnegative")
    if not math.isfinite(d_pose_after) or d_pose_after < 0.0:
        raise CoderBreakEvenError("d_pose_after must be finite and nonnegative")
    total_sites = pair_count * height * width
    saved = SEG_WEIGHT * event_count / total_sites
    saved += math.sqrt(10.0 * baseline_d_pose) - math.sqrt(10.0 * d_pose_after)
    if saved <= 0.0:
        return -1
    threshold = DENOMINATOR_BYTES * saved / RATE_WEIGHT
    # nextafter protects the strict inequality when threshold is represented
    # exactly as an integer in binary floating point.
    return math.floor(math.nextafter(threshold, -math.inf))


def _coder_payloads(raw: bytes) -> dict[str, bytes]:
    payloads = {
        "zlib_level9": zlib.compress(raw, 9),
        "bz2_level9": bz2.compress(raw, 9),
        "lzma_xz_preset6": lzma.compress(raw, preset=6),
        "lzma_xz_preset9_extreme": lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME),
    }
    try:
        import brotli
    except ImportError as exc:
        raise CoderBreakEvenError("Brotli is required for the registered terminal-coder tournament") from exc
    payloads["brotli_generic_q11"] = brotli.compress(raw, quality=11, mode=brotli.MODE_GENERIC)
    return payloads


def measure(*, packet_path: Path, bridge_path: Path) -> dict[str, Any]:
    try:
        packet = packet_path.read_bytes()
    except OSError as exc:
        raise CoderBreakEvenError("cannot read S2 packet") from exc
    seed = decode_partition_seed(packet)
    raw, header = _extract_validated_raw_event_bytes(packet)
    bridge, bridge_bytes = _load_json_snapshot(bridge_path, "C1/S2 exact bridge")
    _validate_body_hash(bridge, "receipt_sha256")
    s2_custody = bridge.get("s2")
    identity = bridge.get("identity")
    authority_geometry = bridge.get("authority_geometry")
    decoded_event_stream = [
        [event.pair, event.row, event.col, event.target_class, event.baseline_class]
        for event in seed.events
    ]
    decoded_event_stream_sha256 = _sha256_bytes(_canonical(decoded_event_stream))
    if (
        bridge.get("schema") != BRIDGE_SCHEMA
        or bridge.get("verdict") != "EXACT_C1_LIVE_TARGET_DEBT_EQUALS_R2B_INVENTORY_EQUALS_S2_PACKET"
        or bridge.get("research_only") is not True
        or bridge.get("score_claim") is not False
        or bridge.get("promotion_eligible") is not False
        or bridge.get("pointer_moved") is not False
        or not isinstance(s2_custody, dict)
        or not isinstance(identity, dict)
        or not isinstance(authority_geometry, dict)
        or s2_custody.get("packet_sha256") != _sha256_bytes(packet)
        or s2_custody.get("packet_bytes") != len(packet)
        or identity.get("event_count") != len(seed.events)
        or identity.get("event_stream_sha256") != decoded_event_stream_sha256
        or identity.get("debt_equals_r2b") is not True
        or identity.get("debt_equals_s2") is not True
        or identity.get("r2b_equals_s2") is not True
        or identity.get("strict_site_order") is not True
        or identity.get("unique_sites") is not True
        or identity.get("pose_rows_equal_r2b") is not True
        or authority_geometry.get("pair_count") != seed.n_pairs
        or authority_geometry.get("scorer_hw") != [seed.height, seed.width]
        or authority_geometry.get("total_seg_sites") != seed.n_pairs * seed.height * seed.width
        or authority_geometry.get("s2_geometry_exact") is not True
    ):
        raise CoderBreakEvenError("bridge does not bind this exact S2 packet/population")
    event_count = len(seed.events)
    cap = _strict_break_even_byte_cap(
        event_count=event_count,
        pair_count=seed.n_pairs,
        height=seed.height,
        width=seed.width,
    )
    mean_d_seg = authority_geometry.get("mean_d_seg")
    expected_mean_d_seg = event_count / (seed.n_pairs * seed.height * seed.width)
    if (
        not isinstance(mean_d_seg, (int, float))
        or not math.isfinite(float(mean_d_seg))
        or not math.isclose(float(mean_d_seg), expected_mean_d_seg, rel_tol=0.0, abs_tol=1e-18)
    ):
        raise CoderBreakEvenError("bridge mean_d_seg differs from its exact event population")
    baseline_d_pose_value = authority_geometry.get("baseline_mean_d_pose")
    if (
        isinstance(baseline_d_pose_value, bool)
        or not isinstance(baseline_d_pose_value, (int, float))
        or not math.isfinite(float(baseline_d_pose_value))
        or float(baseline_d_pose_value) < 0.0
    ):
        raise CoderBreakEvenError("bridge lacks the exact baseline mean d_pose")
    baseline_d_pose = float(baseline_d_pose_value)
    seg_score_saved = SEG_WEIGHT * float(mean_d_seg)
    pose_term_before = math.sqrt(10.0 * baseline_d_pose)
    exact_rate_cost = RATE_WEIGHT * len(packet) / DENOMINATOR_BYTES
    coder_rows: list[dict[str, Any]] = []
    for coder, payload in sorted(_coder_payloads(raw).items()):
        payload_rate = RATE_WEIGHT * len(payload) / DENOMINATOR_BYTES
        threshold = _conditional_pose_threshold(
            payload_bytes=len(payload),
            baseline_d_pose=baseline_d_pose,
            seg_score_saved=seg_score_saved,
        )
        coder_rows.append(
            {
                "coder": coder,
                "payload_bytes": len(payload),
                "payload_sha256": _sha256_bytes(payload),
                "payload_only_rate_term": payload_rate,
                "delta_s_if_all_events_realized_and_pose_unchanged": payload_rate - seg_score_saved,
                "exceeds_seg_only_strict_total_break_even_cap": len(payload) > cap,
                "strict_joint_improvement_requires_d_pose_after_below": threshold,
                "minimum_pose_reduction_fraction_required": (
                    None
                    if threshold is None or baseline_d_pose == 0.0
                    else max(0.0, 1.0 - threshold / baseline_d_pose)
                ),
                "headers_and_container_bytes_assumed": 0,
            }
        )
    best = min(coder_rows, key=lambda row: row["payload_bytes"])
    current_pose_threshold = _conditional_pose_threshold(
        payload_bytes=len(packet),
        baseline_d_pose=baseline_d_pose,
        seg_score_saved=seg_score_saved,
    )
    pose_slices = [1.0, 0.75, 0.5, 0.25, 0.0]
    conditional_curve = [
        {
            "d_pose_after_fraction_of_baseline": fraction,
            "d_pose_after": fraction * baseline_d_pose,
            "strict_total_break_even_bytes": _conditional_total_byte_cap(
                event_count=event_count,
                pair_count=seed.n_pairs,
                height=seed.height,
                width=seed.width,
                baseline_d_pose=baseline_d_pose,
                d_pose_after=fraction * baseline_d_pose,
            ),
        }
        for fraction in pose_slices
    ]
    result = {
        "schema": SCHEMA,
        "inputs": {
            "packet": {
                "path": str(packet_path.resolve()),
                "bytes": len(packet),
                "sha256": _sha256_bytes(packet),
            },
            "bridge": {
                "path": str(bridge_path.resolve()),
                "bytes": len(bridge_bytes),
                "sha256": _sha256_bytes(bridge_bytes),
                "receipt_sha256": bridge["receipt_sha256"],
            },
            "raw_event_stream": {
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "codec": header.get("codec"),
            },
        },
        "exact_economics": {
            "event_count": event_count,
            "pair_count": seed.n_pairs,
            "scorer_hw": [seed.height, seed.width],
            "seg_score_saved_if_all_events_realized": seg_score_saved,
            "baseline_mean_d_pose": baseline_d_pose,
            "baseline_pose_score_term": pose_term_before,
            "current_packet_rate_term": exact_rate_cost,
            "current_packet_delta_s_if_all_events_realized_and_pose_unchanged": exact_rate_cost
            - seg_score_saved,
            "current_packet_strict_joint_improvement_requires_d_pose_after_below": current_pose_threshold,
            "seg_only_strict_total_break_even_bytes": cap,
            "seg_only_strict_break_even_bytes_per_event": cap / event_count,
            "full_pose_elimination_strict_total_break_even_bytes": conditional_curve[-1][
                "strict_total_break_even_bytes"
            ],
            "current_packet_bytes_per_event": len(packet) / event_count,
            "minimum_fractional_packet_reduction_required_if_pose_unchanged": 1.0 - cap / len(packet),
            "conditional_joint_break_even_curve": conditional_curve,
        },
        "terminal_coder_payload_lower_bounds": coder_rows,
        "best_payload_only": best,
        "verdict": "POSE_CONDITIONAL_TERMINAL_CODER_ECONOMICS_RECEIVER_MEASUREMENT_REQUIRED",
        "verdict_scope": (
            "the exact supplied S2 site-delta/class raw stream under the enumerated deterministic terminal coders, "
            "conditional on realizing every Seg correction. Every tested payload is rate-negative when Pose is "
            "unchanged, but none is unconditionally rate-dead because a coupled receiver may lower Pose. The open "
            "d_pose-after thresholds and full conditional byte curve are derived, not score claims"
        ),
        "routing": (
            "measure a receiver-closed realization's actual joint d_pose-after before admitting or killing the S2 "
            "packet; in parallel, use the exact S2 identity as supervision for V9 boundary/topology factorization "
            "because that transform can improve the rate side without presuming the Pose outcome"
        ),
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    result["receipt_sha256"] = _sha256_bytes(_canonical(result))
    return result


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise CoderBreakEvenError(f"write-once output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != encoded:
                raise CoderBreakEvenError(f"concurrent write-once output differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = measure(
            packet_path=args.packet.expanduser().resolve(),
            bridge_path=args.bridge.expanduser().resolve(),
        )
        output = args.output.expanduser().resolve()
        _write_once(output, result)
    except (OSError, ValueError, CoderBreakEvenError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "receipt_sha256": result["receipt_sha256"],
                "verdict": result["verdict"],
                "seg_only_strict_total_break_even_bytes": result["exact_economics"][
                    "seg_only_strict_total_break_even_bytes"
                ],
                "best_payload_bytes": result["best_payload_only"]["payload_bytes"],
                "score_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
