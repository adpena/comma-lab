#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ATW V2-1 byte-closed side-info conditioning probe.

The 2026-05-18 ATW V2 symposium narrowed the useful next local artifact:
do not dispatch the existing ATW v2 recipe from the saturated two-signature
D4 result; instead, test richer scorer-derived side-info channels under the
actual shippable-byte constraint. This helper consumes the already-rendered
alternative-reducer outputs, serializes each candidate as a deterministic
decoder-recoverable packet, and recomputes H(latent | side_info) against A1
latents.

The result is diagnostic only. It is not a contest score, does not authorize
dispatch, and deliberately records score_claim=false.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tools"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from probe_alternative_reducers_latent_class_conditioning import (  # noqa: E402
    REDUCER_MEANINGFUL_THRESHOLD_BITS,
    compute_alternative_reducer_verdict,
)
from run_atw_v2_d4_probe_from_a1 import (  # noqa: E402
    _load_a1_latents,
    _quantize_u8,
    _repo_rel,
)

SIDE_INFO_MAGIC = b"ATW21SI\0"
SIDE_INFO_VERSION = 1
SIDE_INFO_HEADER_FMT = "<8sBHHBBBBII"
SIDE_INFO_HEADER_SIZE = struct.calcsize(SIDE_INFO_HEADER_FMT)
DEFAULT_SIDE_INFO_BUDGET_BYTES = 2048
CONTEST_NORMALIZER_BYTES = 37_545_489.0

DEFAULT_REDUCER_JSON = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "alternative_reducer_probes_20260516T225900Z"
    / "tishby_ib_pure_per_pair_reducer_outputs.json"
)
DEFAULT_RESEARCH_JSON = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "atw_v2_1_byte_closed_side_info_probe_20260518_codex.json"
)
DEFAULT_RESEARCH_MD = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "atw_v2_1_byte_closed_side_info_probe_20260518_codex.md"
)
DEFAULT_STATE_JSON = (
    REPO_ROOT
    / ".omx"
    / "state"
    / "atw_v2_1_byte_closed_side_info_probe.json"
)
REDUCER_ORDER = (
    "per_pixel_histogram",
    "per_region_histogram",
    "per_pair_class_2_fraction",
    "per_frame_argmax",
)


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "experiments" / "results" / f"atw_v2_1_sideinfo_probe_{stamp}"


def replay_command() -> str:
    return " ".join([".venv/bin/python", _repo_rel(Path(__file__)), *sys.argv[1:]])


def bit_width_for_values(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot compute bit width for empty values")
    max_value = max(int(v) for v in values)
    if max_value < 0:
        raise ValueError("side-info values must be non-negative")
    return max(1, max_value.bit_length())


def pack_bits(values: Sequence[int], bit_width: int) -> bytes:
    """Pack non-negative integers little-endian into a compact bitstream."""

    if bit_width < 1 or bit_width > 64:
        raise ValueError(f"bit_width={bit_width} must be in [1, 64]")
    limit = 1 << bit_width
    out = bytearray()
    acc = 0
    acc_bits = 0
    for value in values:
        ivalue = int(value)
        if ivalue < 0 or ivalue >= limit:
            raise ValueError(f"value {ivalue} does not fit {bit_width} bits")
        acc |= ivalue << acc_bits
        acc_bits += bit_width
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def unpack_bits(raw: bytes, *, count: int, bit_width: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be >= 0")
    if bit_width < 1 or bit_width > 64:
        raise ValueError(f"bit_width={bit_width} must be in [1, 64]")
    mask = (1 << bit_width) - 1
    out: list[int] = []
    acc = 0
    acc_bits = 0
    pos = 0
    while len(out) < count:
        while acc_bits < bit_width:
            if pos >= len(raw):
                raise ValueError("bitstream truncated")
            acc |= raw[pos] << acc_bits
            acc_bits += 8
            pos += 1
        out.append(acc & mask)
        acc >>= bit_width
        acc_bits -= bit_width
    return out


def encode_side_info_packet(values: Sequence[int], *, reducer_name: str) -> bytes:
    """Encode per-pair side-info values as a deterministic dictionary packet.

    The packet is intentionally self-describing enough for an inflate-time
    decoder to recover the exact per-pair values: a sorted dictionary of unique
    reducer symbols plus a bit-packed index stream.
    """

    if not values:
        raise ValueError("values must be non-empty")
    if len(values) > 0xFFFF:
        raise ValueError("values length exceeds u16 packet contract")
    name_bytes = reducer_name.encode("ascii")
    if len(name_bytes) > 0xFF:
        raise ValueError("reducer_name too long for packet")
    values_i = [int(v) for v in values]
    if any(v < 0 for v in values_i):
        raise ValueError("side-info values must be non-negative")
    dictionary = sorted(set(values_i))
    if len(dictionary) > 0xFFFF:
        raise ValueError("dictionary too large for u16 packet contract")
    index_by_value = {value: i for i, value in enumerate(dictionary)}
    indices = [index_by_value[value] for value in values_i]

    value_bit_width = bit_width_for_values(dictionary)
    index_bit_width = max(1, (len(dictionary) - 1).bit_length())
    dictionary_blob = pack_bits(dictionary, value_bit_width)
    index_blob = pack_bits(indices, index_bit_width)
    header = struct.pack(
        SIDE_INFO_HEADER_FMT,
        SIDE_INFO_MAGIC,
        SIDE_INFO_VERSION,
        len(values_i),
        len(dictionary),
        value_bit_width,
        index_bit_width,
        len(name_bytes),
        0,
        len(dictionary_blob),
        len(index_blob),
    )
    return header + name_bytes + dictionary_blob + index_blob


def decode_side_info_packet(packet: bytes) -> tuple[str, list[int]]:
    if len(packet) < SIDE_INFO_HEADER_SIZE:
        raise ValueError("packet too short")
    (
        magic,
        version,
        n_values,
        dictionary_len,
        value_bit_width,
        index_bit_width,
        name_len,
        _reserved,
        dictionary_blob_len,
        index_blob_len,
    ) = struct.unpack(SIDE_INFO_HEADER_FMT, packet[:SIDE_INFO_HEADER_SIZE])
    if magic != SIDE_INFO_MAGIC:
        raise ValueError(f"bad side-info magic: {magic!r}")
    if version != SIDE_INFO_VERSION:
        raise ValueError(f"unsupported side-info version: {version}")
    pos = SIDE_INFO_HEADER_SIZE
    end_name = pos + name_len
    end_dictionary = end_name + dictionary_blob_len
    end_index = end_dictionary + index_blob_len
    if end_index != len(packet):
        raise ValueError("packet length does not match header section lengths")
    reducer_name = packet[pos:end_name].decode("ascii")
    dictionary = unpack_bits(
        packet[end_name:end_dictionary],
        count=dictionary_len,
        bit_width=value_bit_width,
    )
    indices = unpack_bits(
        packet[end_dictionary:end_index],
        count=n_values,
        bit_width=index_bit_width,
    )
    values: list[int] = []
    for index in indices:
        if index >= len(dictionary):
            raise ValueError(f"dictionary index {index} out of range")
        values.append(dictionary[index])
    return reducer_name, values


def load_reducer_outputs(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("per_pair_by_reducer")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} missing per_pair_by_reducer object")
    out: dict[str, list[int]] = {}
    for reducer_name in REDUCER_ORDER:
        values = raw.get(reducer_name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{path} missing non-empty reducer list {reducer_name}")
        out[reducer_name] = [int(v) for v in values]
    lengths = {len(v) for v in out.values()}
    if len(lengths) != 1:
        raise ValueError(f"reducer output lengths disagree: {sorted(lengths)}")
    return out


def load_a1_latent_bytes() -> tuple[bytes, dict[str, Any]]:
    latents, provenance = _load_a1_latents()
    quantized, quantizer = _quantize_u8(latents)
    latent_bytes = bytes(quantized.flatten().tolist())
    return latent_bytes, {
        **provenance,
        "latent_quantizer": quantizer,
        "latent_u8_sha256": sha256_bytes(latent_bytes),
        "latent_u8_bytes": len(latent_bytes),
    }


def channel_action(channel: dict[str, Any]) -> str:
    if channel["byte_budget_ok"] and channel["verdict"] == "MEANINGFUL_CONDITIONING":
        return "wave_n_plus_1_council_before_any_dispatch"
    if channel["byte_budget_ok"] and channel["verdict"] == "WEAK_CONDITIONING":
        return "design_tighter_scorer_logit_sketch_or_trained_atw_residual_probe"
    if not channel["byte_budget_ok"]:
        return "reject_or_recode_side_info_payload_before_mi_interpretation"
    return "do_not_dispatch_from_this_channel"


def build_probe_payload(
    *,
    latent_stream: bytes,
    per_pair_by_reducer: dict[str, list[int]],
    output_dir: Path,
    source_reducer_json: Path,
    budget_bytes: int = DEFAULT_SIDE_INFO_BUDGET_BYTES,
    max_pairs: int | None = None,
    latent_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build packets, recompute MI, and return the fail-closed probe payload."""

    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    n_pairs_available = len(next(iter(per_pair_by_reducer.values())))
    if max_pairs is None:
        n_pairs = n_pairs_available
    else:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        n_pairs = min(max_pairs, n_pairs_available)
    if len(latent_stream) % n_pairs_available != 0:
        raise ValueError(
            f"latent stream length {len(latent_stream)} not divisible by "
            f"available pair count {n_pairs_available}"
        )
    symbols_per_pair = len(latent_stream) // n_pairs_available
    latent_slice = latent_stream[: n_pairs * symbols_per_pair]
    output_dir.mkdir(parents=True, exist_ok=True)

    channels: list[dict[str, Any]] = []
    for reducer_name in REDUCER_ORDER:
        values = per_pair_by_reducer[reducer_name][:n_pairs]
        packet = encode_side_info_packet(values, reducer_name=reducer_name)
        decoded_name, decoded_values = decode_side_info_packet(packet)
        roundtrip_ok = decoded_name == reducer_name and decoded_values == values
        if not roundtrip_ok:
            raise ValueError(f"side-info packet roundtrip failed for {reducer_name}")
        packet_path = output_dir / f"atw_v2_1_side_info_{reducer_name}.bin"
        packet_path.write_bytes(packet)

        verdict = compute_alternative_reducer_verdict(
            substrate_id="atw_codec_v2_1_a1_sideinfo",
            reducer_name=reducer_name,  # type: ignore[arg-type]
            latent_stream=latent_slice,
            per_pair_reduced_class=values,
            symbols_per_pair=symbols_per_pair,
            notes=(
                "ATW V2-1 byte-closed side-info probe. Per-pair reducer "
                "symbols are serialized into an ATW21SI dictionary packet, "
                "round-tripped, then evaluated against A1 latent bytes. "
                "Diagnostic only; not a score claim."
            ),
        )
        packet_bytes = len(packet)
        channels.append(
            {
                "reducer_name": reducer_name,
                "verdict": verdict.verdict,
                "mutual_information_bits": verdict.mutual_information_bits,
                "h_latent_unconditional_bits_per_symbol": (
                    verdict.h_latent_unconditional_bits_per_symbol
                ),
                "h_latent_given_side_info_bits_per_symbol": (
                    verdict.h_latent_given_reduced_class_bits_per_symbol
                ),
                "wyner_ziv_gain_ceiling_fraction": (
                    verdict.wyner_ziv_gain_ceiling_fraction
                ),
                "meaningful_mi_threshold_bits": verdict.meaningful_mi_threshold_bits,
                "num_unique_side_info_values": verdict.num_unique_reduced_classes,
                "num_pairs": n_pairs,
                "symbols_per_pair": symbols_per_pair,
                "packet_path": _repo_rel(packet_path),
                "packet_sha256": sha256_bytes(packet),
                "packet_bytes": packet_bytes,
                "byte_budget": budget_bytes,
                "byte_budget_ok": packet_bytes <= budget_bytes,
                "side_info_rate_score_cost": (
                    25.0 * packet_bytes / CONTEST_NORMALIZER_BYTES
                ),
                "packet_roundtrip_ok": roundtrip_ok,
                "phase2_action": channel_action(
                    {
                        "byte_budget_ok": packet_bytes <= budget_bytes,
                        "verdict": verdict.verdict,
                    }
                ),
                "diagnostic_verdict": asdict(verdict),
            }
        )

    byte_closed = [c for c in channels if c["byte_budget_ok"]]
    meaningful = [c for c in byte_closed if c["verdict"] == "MEANINGFUL_CONDITIONING"]
    weak = [c for c in byte_closed if c["verdict"] == "WEAK_CONDITIONING"]
    best_overall_channel = max(
        channels,
        key=lambda c: (
            float(c["mutual_information_bits"]),
            -int(c["packet_bytes"]),
        ),
    )
    best_byte_closed_channel = (
        max(
            byte_closed,
            key=lambda c: (
                float(c["mutual_information_bits"]),
                -int(c["packet_bytes"]),
            ),
        )
        if byte_closed
        else None
    )
    if meaningful:
        phase2_status = "byte_closed_meaningful_channel_requires_wave_n_plus_1_council"
        recommended_next_gate = "wave_n_plus_1_atw_v2_1_council_before_any_paid_dispatch"
    elif weak:
        phase2_status = "byte_closed_channels_only_weak_conditioning"
        recommended_next_gate = (
            "design_substrate_native_scorer_logit_sketch_or_trained_atw_residual_probe"
        )
    else:
        phase2_status = "byte_closed_channels_independent_or_over_budget"
        recommended_next_gate = "pivot_to_other_base_substrate_or_g2_partial"

    return {
        "schema": "atw_v2_1_byte_closed_side_info_probe_v1",
        "observed_at_utc": utc_now(),
        "command": replay_command(),
        "source_reducer_json": _repo_rel(source_reducer_json),
        "output_dir": _repo_rel(output_dir),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "dispatch_attempted": False,
        "provider_spend_attempted": False,
        "evidence_grade": "diagnostic_cpu",
        "axis_label": "[diagnostic-CPU; ATW V2-1 byte-closed side-info MI probe]",
        "side_info_budget_bytes": budget_bytes,
        "num_pairs": n_pairs,
        "symbols_per_pair": symbols_per_pair,
        "channels": channels,
        "best_byte_closed_channel": (
            {
                key: best_byte_closed_channel[key]
                for key in (
                    "reducer_name",
                    "verdict",
                    "mutual_information_bits",
                    "packet_bytes",
                    "byte_budget_ok",
                    "wyner_ziv_gain_ceiling_fraction",
                    "phase2_action",
                )
            }
            if best_byte_closed_channel is not None
            else None
        ),
        "best_overall_channel": {
            key: best_overall_channel[key]
            for key in (
                "reducer_name",
                "verdict",
                "mutual_information_bits",
                "packet_bytes",
                "byte_budget_ok",
                "wyner_ziv_gain_ceiling_fraction",
                "phase2_action",
            )
        },
        "phase2_status": phase2_status,
        "recommended_next_gate": recommended_next_gate,
        "result_review_blockers": [
            "diagnostic_probe_not_score_claim",
            "requires_new_d4_or_v2_1_council_before_dispatch_authority",
            "requires_paired_contest_cuda_cpu_harvest_before_promotion",
        ],
        "latent_provenance": latent_provenance or {},
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ATW V2-1 Byte-Closed Side-Info Probe",
        "",
        f"- observed_at_utc: `{payload['observed_at_utc']}`",
        f"- axis_label: `{payload['axis_label']}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- dispatch_attempted: `false`",
        "- provider_spend_attempted: `false`",
        f"- source_reducer_json: `{payload['source_reducer_json']}`",
        f"- output_dir: `{payload['output_dir']}`",
        f"- side_info_budget_bytes: `{payload['side_info_budget_bytes']}`",
        f"- phase2_status: `{payload['phase2_status']}`",
        f"- recommended_next_gate: `{payload['recommended_next_gate']}`",
        "",
        "## Channel Results",
        "",
        "| Channel | Packet bytes | Budget ok | Unique values | MI bits/symbol | Threshold | Verdict | WZ ceiling | Phase 2 action |",
        "|---|---:|---|---:|---:|---:|---|---:|---|",
    ]
    for channel in payload["channels"]:
        lines.append(
            "| {name} | {bytes} | {budget} | {unique} | {mi:.12f} | {thr:.3f} | {verdict} | {wz:.12f} | {action} |".format(
                name=channel["reducer_name"],
                bytes=channel["packet_bytes"],
                budget=str(channel["byte_budget_ok"]).lower(),
                unique=channel["num_unique_side_info_values"],
                mi=channel["mutual_information_bits"],
                thr=channel["meaningful_mi_threshold_bits"],
                verdict=channel["verdict"],
                wz=channel["wyner_ziv_gain_ceiling_fraction"],
                action=channel["phase2_action"],
            )
        )
    best = payload["best_byte_closed_channel"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
        ]
    )
    if best is None:
        overall = payload["best_overall_channel"]
        lines.extend(
            [
                "No byte-closed channel fit the configured side-info budget.",
                f"Highest-MI over-budget channel: `{overall['reducer_name']}` with "
                f"verdict `{overall['verdict']}`, MI "
                f"`{overall['mutual_information_bits']:.12f}` bits/symbol, packet "
                f"bytes `{overall['packet_bytes']}`. It is advisory only and cannot "
                "authorize dispatch until the payload is recoded under budget.",
                "",
                "This probe therefore does not close the byte-budget side of the",
                "ATW V2-1 question for the configured budget. Diagnostic evidence",
                "here remains non-promotional.",
            ]
        )
    else:
        lines.extend(
            [
                f"Best byte-closed channel: `{best['reducer_name']}` with verdict "
                f"`{best['verdict']}`, MI `{best['mutual_information_bits']:.12f}` "
                f"bits/symbol, packet bytes `{best['packet_bytes']}`.",
                "",
                "This probe closes the byte-budget side of the ATW V2-1 question for",
                "the currently available richer reducer artifacts: dictionary-coded",
                "side-info packets fit the <=2KB archive sidecar budget, but Phase 2",
                "dispatch still requires a MEANINGFUL_CONDITIONING result plus Wave",
                "N+1 council. Diagnostic evidence here remains non-promotional.",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            f"- command: `{payload['command']}`",
            "- packets: see per-channel `packet_path` entries in the JSON artifact",
            "",
            "## Next Gate",
            "",
            f"`{payload['recommended_next_gate']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reducer-json", type=Path, default=DEFAULT_REDUCER_JSON)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--research-json", type=Path, default=DEFAULT_RESEARCH_JSON)
    parser.add_argument("--research-md", type=Path, default=DEFAULT_RESEARCH_MD)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--side-info-budget-bytes", type=int, default=DEFAULT_SIDE_INFO_BUDGET_BYTES)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--skip-state", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reducer_json = args.reducer_json.resolve()
    output_dir = args.output_dir.resolve()
    per_pair_by_reducer = load_reducer_outputs(reducer_json)
    latent_bytes, latent_provenance = load_a1_latent_bytes()
    payload = build_probe_payload(
        latent_stream=latent_bytes,
        per_pair_by_reducer=per_pair_by_reducer,
        output_dir=output_dir,
        source_reducer_json=reducer_json,
        budget_bytes=int(args.side_info_budget_bytes),
        max_pairs=args.max_pairs,
        latent_provenance=latent_provenance,
    )
    write_json(output_dir / "atw_v2_1_byte_closed_side_info_probe.json", payload)
    write_json(args.research_json.resolve(), payload)
    args.research_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.research_md.resolve().write_text(render_markdown(payload), encoding="utf-8")
    if not args.skip_state:
        write_json(args.state_json.resolve(), payload)
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
