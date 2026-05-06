#!/usr/bin/env python3
"""Build PR79 S2 action-stream repack candidates.

This is a local byte-screening tool only.  It preserves decoded
``seg_tile_actions.bin`` bytes exactly, keeps non-action streams byte-identical
at runtime, and does not submit remote or GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.archive_byte_profile import profile_archive


BASE_BUILDER_PATH = REPO_ROOT / "experiments/build_pr79_action_lossless_repack_candidates.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr79_action_dictionary_repack_v2_20260503_codex"
TOOL = "experiments/build_pr79_action_dictionary_repack_candidates_v2.py"
SCHEMA = "pr79_action_dictionary_repack_v2_matrix_v1"
MANIFEST_SCHEMA = "pr79_action_dictionary_repack_v2_manifest_v1"
S2_MAGIC = b"S2"
S2_MODE_ADAPTIVE_ARITH = 1
ACTION_ALPHABET = 108
PR79_S2_FRONTIER_SCORE = 0.31453355357318635  # [external: PR-79 S2 contest-CUDA T4 A++ frontier]
PR79_S2_FRONTIER_BYTES = 277_321
PR79_S2_FRONTIER_SHA256 = "5740aca7e255b00093154eb1823b5b6207d8795f8eb287d35758c4cda438ec68"
SCORE_DENOMINATOR_BYTES = 37_545_489  # [contest-defined: original video bytes]
RATE_SCORE_WEIGHT = 25.0  # [contest-defined: rate weighting from upstream/evaluate.py]
CUDA_AUTH_EVAL_REQUIRED = (
    "No dispatch from this worker. Before any exact eval, claim a non-conflicting "
    "lane with tools/claim_lane_dispatch.py claim, then run exact T4-equivalent "
    "CUDA auth eval on identical archive bytes through archive.zip -> inflate.sh "
    "-> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda "
    "with expected archive SHA and size."
)


@dataclass(frozen=True)
class ArithmeticCode:
    data: bytes
    nbits: int


def _load_base_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_s1_base_builder", BASE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base builder from {BASE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_builder()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _uleb128(value: int) -> bytes:
    return BASE._uleb128(value)  # noqa: SLF001


def _read_action_records(raw: bytes) -> list[tuple[int, int, int]]:
    return BASE._read_action_records(raw)  # noqa: SLF001


def _duplicate_examples(keys: list[Any], *, limit: int = 8) -> list[Any]:
    counts: dict[Any, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return [
        {"key": list(key) if isinstance(key, tuple) else key, "count": count}
        for key, count in sorted(counts.items())
        if count > 1
    ][:limit]


def action_record_accounting(
    source_decoded_actions: bytes,
    repacked_decoded_actions: bytes,
) -> dict[str, Any]:
    """Record ordering and duplicate facts that affect raw-output parity gates."""
    source_records = _read_action_records(source_decoded_actions)
    repacked_records = _read_action_records(repacked_decoded_actions)
    source_pair_tiles = [(pair, tile) for pair, tile, _action in source_records]
    source_pairs = [pair for pair, _tile, _action in source_records]
    source_tiles = [tile for _pair, tile, _action in source_records]
    duplicate_pair_tile_record_count = len(source_pair_tiles) - len(set(source_pair_tiles))
    duplicate_pair_record_count = len(source_pairs) - len(set(source_pairs))
    duplicate_tile_record_count = len(source_tiles) - len(set(source_tiles))
    encoder_reorders_records = repacked_records != source_records
    duplicate_sensitive = duplicate_pair_tile_record_count > 0 or duplicate_pair_record_count > 0
    parity_requirement = {
        "required": encoder_reorders_records or duplicate_sensitive,
        "requirement": (
            "raw-output parity must be recorded before exact-eval dispatch when "
            "record order or duplicate pair/tile multiplicity changes"
        ),
        "reason": (
            "S2 grouping can reorder records or expose duplicate pair/tile semantics"
            if encoder_reorders_records or duplicate_sensitive
            else "source order and pair/tile multiplicity are unchanged by this repack"
        ),
        "satisfied_by_runtime_parse_validation": not encoder_reorders_records,
    }
    return {
        "duplicate_pair_tile_accounting": {
            "duplicate_pair_examples": _duplicate_examples(source_pairs),
            "duplicate_pair_record_count": duplicate_pair_record_count,
            "duplicate_pair_tile_examples": _duplicate_examples(source_pair_tiles),
            "duplicate_pair_tile_record_count": duplicate_pair_tile_record_count,
            "duplicate_tile_examples": _duplicate_examples(source_tiles),
            "duplicate_tile_record_count": duplicate_tile_record_count,
            "total_records": len(source_records),
            "unique_pair_count": len(set(source_pairs)),
            "unique_pair_tile_count": len(set(source_pair_tiles)),
            "unique_tile_count": len(set(source_tiles)),
        },
        "raw_output_parity_requirement": parity_requirement,
        "record_order": {
            "encoder_reorders_records": encoder_reorders_records,
            "repacked_matches_source_order": not encoder_reorders_records,
            "source_record_count": len(source_records),
        },
    }


def encode_s2_meta_and_actions(decoded_actions: bytes) -> tuple[bytes, bytes, bytes]:
    """Return ``(metadata_and_deltas, action_ids, s1_raw)`` in S1 semantic order."""
    records = _read_action_records(decoded_actions)
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    for tile_id in sorted({tile for _pair, tile, _action in records}):
        entries = sorted((pair, action) for pair, tile, action in records if tile == tile_id)
        groups.append((tile_id, entries))

    meta = bytearray(b"S1")
    meta += _uleb128(len(groups))
    deltas = bytearray()
    actions = bytearray()
    previous_tile = 0
    for tile_index, (tile_id, entries) in enumerate(groups):
        tile_delta = tile_id if tile_index == 0 else tile_id - previous_tile
        if tile_delta < 0:
            raise ValueError("S2 tile groups must be sorted")
        meta += _uleb128(tile_delta)
        meta += _uleb128(len(entries))
        previous_pair = 0
        for record_index, (pair_index, action_id) in enumerate(entries):
            delta = pair_index if record_index == 0 else pair_index - previous_pair
            if delta < 0:
                raise ValueError("S2 pair deltas require sorted pairs inside a tile")
            deltas += _uleb128(delta)
            actions.append(action_id)
            previous_pair = pair_index
        previous_tile = tile_id
    return bytes(meta + deltas), bytes(actions), bytes(meta + deltas + actions)


def encode_adaptive_arithmetic_actions(actions: bytes, *, alphabet: int = ACTION_ALPHABET) -> ArithmeticCode:
    """Encode action IDs with the same adaptive arithmetic model used by inflate."""
    if not actions:
        raise ValueError("adaptive arithmetic action stream cannot be empty")
    low = Fraction(0, 1)
    high = Fraction(1, 1)
    counts = [1] * alphabet
    total = alphabet
    for symbol in actions:
        if symbol >= alphabet:
            raise ValueError(f"action symbol {symbol} outside alphabet {alphabet}")
        cumulative = sum(counts[:symbol])
        count = counts[symbol]
        width = high - low
        high = low + width * Fraction(cumulative + count, total)
        low = low + width * Fraction(cumulative, total)
        counts[symbol] += 1
        total += 1

    width = high - low
    start = max(0, width.denominator.bit_length() - width.numerator.bit_length() - 2)
    for nbits in range(start, start + 128):
        code_int = (low.numerator * (1 << nbits) + low.denominator - 1) // low.denominator
        if Fraction(code_int, 1 << nbits) < high:
            return ArithmeticCode(data=code_int.to_bytes((nbits + 7) // 8, "big"), nbits=nbits)
    raise ValueError("failed to find an arithmetic code inside the final interval")


def encode_s2_adaptive_actions(decoded_actions: bytes) -> dict[str, Any]:
    meta_and_deltas, actions, s1_raw = encode_s2_meta_and_actions(decoded_actions)
    meta_choice = BASE.best_brotli(meta_and_deltas)
    action_code = encode_adaptive_arithmetic_actions(actions)
    wire = (
        S2_MAGIC
        + _uleb128(S2_MODE_ADAPTIVE_ARITH)
        + _uleb128(len(meta_choice.data))
        + _uleb128(action_code.nbits)
        + meta_choice.data
        + action_code.data
    )
    return {
        "actions_raw": actions,
        "action_arithmetic_bytes": len(action_code.data),
        "action_arithmetic_nbits": action_code.nbits,
        "meta_and_deltas_brotli": meta_choice,
        "meta_and_deltas_raw": meta_and_deltas,
        "s1_raw": s1_raw,
        "wire": wire,
    }


def _build_fixed_payload(source: Any, actions_wire: bytes) -> bytes:
    return (
        source.raw_segments["masks.mkv"]
        + source.raw_segments["renderer.bin"]
        + actions_wire
        + source.raw_segments["optimized_poses.qp1"]
    )


def _build_p3_payload(source: Any, actions_wire: bytes) -> bytes:
    return (
        b"P3"
        + struct.pack(
            "<IHH",
            len(source.raw_segments["masks.mkv"]),
            len(source.raw_segments["renderer.bin"]),
            len(actions_wire),
        )
        + source.raw_segments["masks.mkv"]
        + source.raw_segments["renderer.bin"]
        + actions_wire
        + source.raw_segments["optimized_poses.qp1"]
    )


def _validate_payload(source: Any, payload: bytes, *, unpacker: Any) -> dict[str, Any]:
    validation = BASE._validate_payload(source, payload, unpacker=unpacker)  # noqa: SLF001
    if validation["decoded_action_sha256"] != _sha256_bytes(source.decoded["seg_tile_actions.bin"]):
        raise ValueError("decoded action SHA changed")
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    validation["decoded_actions_for_accounting"] = decoded["seg_tile_actions.bin"]
    return validation


def _build_one(
    *,
    source: Any,
    candidate_id: str,
    payload: bytes,
    actions_wire: bytes,
    stream_packing: dict[str, Any],
    output_dir: Path,
    unpacker: Any,
    force: bool,
) -> dict[str, Any]:
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    BASE._write_archive(archive_path, payload)  # noqa: SLF001
    if BASE._read_single_payload(archive_path) != payload:  # noqa: SLF001
        raise ValueError(f"{candidate_id}: archive readback mismatch")
    validation = _validate_payload(source, payload, unpacker=unpacker)
    archive_sha = _sha256_file(archive_path)
    payload_sha = _sha256_bytes(payload)
    source_action_wire = source.raw_segments["seg_tile_actions.bin"]
    accounting = action_record_accounting(
        source.decoded["seg_tile_actions.bin"],
        validation["decoded_actions_for_accounting"],
    )
    validation_manifest = {
        key: value for key, value in validation.items() if key != "decoded_actions_for_accounting"
    }
    noop_status = (
        "byte_noop"
        if payload == source.payload
        else "decoded_action_semantics_preserved_action_bytes_changed"
    )
    dispatch_recommended = archive_path.stat().st_size < source.archive_bytes
    archive_delta_vs_pr79 = archive_path.stat().st_size - source.archive_bytes
    archive_delta_vs_s2 = archive_path.stat().st_size - PR79_S2_FRONTIER_BYTES
    manifest = {
        "action_record_accounting": accounting,
        "archive_byte_profile": profile_archive(archive_path),
        "break_even_math": {
            "rate_score_weight": RATE_SCORE_WEIGHT,
            "score_denominator_bytes": SCORE_DENOMINATOR_BYTES,
            "versus_pr79": {
                "archive_byte_delta": archive_delta_vs_pr79,
                "break_even_component_improvement": max(
                    0.0,
                    RATE_SCORE_WEIGHT * archive_delta_vs_pr79 / SCORE_DENOMINATOR_BYTES,
                ),
                "reference_archive_bytes": source.archive_bytes,
                "reference_archive_sha256": source.archive_sha256,
            },
            "versus_pr79_s2": {
                "archive_byte_delta": archive_delta_vs_s2,
                "break_even_component_improvement": max(
                    0.0,
                    RATE_SCORE_WEIGHT * archive_delta_vs_s2 / SCORE_DENOMINATOR_BYTES,
                ),
                "reference_archive_bytes": PR79_S2_FRONTIER_BYTES,
                "reference_archive_sha256": PR79_S2_FRONTIER_SHA256,
                "reference_score": PR79_S2_FRONTIER_SCORE,
            },
        },
        "candidate_id": candidate_id,
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
            "decoded_action_sha256": validation["decoded_action_sha256"],
        "dispatch_recommendation": {
            "dispatch_ready_now": False,
            "exact_eval_justified_after_lane_claim": dispatch_recommended,
            "lane_claim_required": True,
            "recommended": dispatch_recommended,
            "reason": (
                "local runtime-closed decoded-action-preserving byte improvement; "
                "exact CUDA T4 gate is still required for score evidence"
                if dispatch_recommended
                else "local byte screen did not beat PR79"
            ),
        },
        "evidence_grade": "empirical_lossless_byte_screen",
        "no_op_detection": {
            "actions_wire_sha_equal_to_source": _sha256_bytes(actions_wire) == _sha256_bytes(source_action_wire),
            "archive_sha_equal_to_source": archive_sha == source.archive_sha256,
            "decoded_action_sha_equal_to_source": True,
            "payload_sha_equal_to_source": payload_sha == source.payload_sha256,
            "status": noop_status,
        },
        "output_archive": {
            "bytes": archive_path.stat().st_size,
            "path": str(archive_path),
            "repo_relative_path": _repo_rel(archive_path),
            "sha256": archive_sha,
        },
        "payload": {"bytes": len(payload), "member": BASE.MEMBER_NAME, "sha256": payload_sha},
        "remote_dispatch_performed": False,
        "runtime_parse_validation": validation_manifest,
        "schema": MANIFEST_SCHEMA,
        "score_claim": False,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "stream_delta": {
            "actions_wire_delta_bytes_vs_s1": len(actions_wire) - 1121,
            "actions_wire_delta_bytes_vs_source": len(actions_wire) - len(source_action_wire),
            "actions_wire_source_bytes": len(source_action_wire),
            "actions_wire_source_sha256": _sha256_bytes(source_action_wire),
            "actions_wire_v2_bytes": len(actions_wire),
            "actions_wire_v2_sha256": _sha256_bytes(actions_wire),
            "archive_delta_bytes_vs_pr79": archive_delta_vs_pr79,
            "archive_delta_bytes_vs_pr79_s2": archive_delta_vs_s2,
            "payload_delta_bytes_vs_pr79": len(payload) - len(source.payload),
            "source_decoded_action_bytes": len(source.decoded["seg_tile_actions.bin"]),
            "source_decoded_action_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
        },
        "stream_packing": stream_packing,
        "tool": TOOL,
    }
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["repo_relative_path"],
        "archive_sha256": archive_sha,
        "candidate_id": candidate_id,
        "decoded_action_sha256": validation["decoded_action_sha256"],
        "delta_bytes_vs_pr79": manifest["stream_delta"]["archive_delta_bytes_vs_pr79"],
        "dispatch_recommendation": manifest["dispatch_recommendation"],
        "manifest_path": _repo_rel(manifest_path),
        "no_op_status": noop_status,
        "payload_bytes": len(payload),
        "score_claim": False,
        "seg_tile_actions_wire_bytes": len(actions_wire),
        "seg_tile_actions_wire_sha256": _sha256_bytes(actions_wire),
    }


def build_candidates(
    *,
    pr79_archive: Path = BASE.DEFAULT_PR79_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    unpacker = BASE._load_unpacker()  # noqa: SLF001
    source = BASE.load_source_archive(pr79_archive, unpacker=unpacker)
    if pr79_archive == BASE.DEFAULT_PR79_ARCHIVE and source.archive_sha256 != BASE.PR79_SHA256:
        raise ValueError(f"default PR79 archive SHA mismatch: {source.archive_sha256}")

    s1_raw = BASE.encode_s1_split_actions(source.decoded["seg_tile_actions.bin"])
    s1_choice = BASE.best_brotli(s1_raw)
    s2 = encode_s2_adaptive_actions(source.decoded["seg_tile_actions.bin"])
    if brotli.decompress(s2["meta_and_deltas_brotli"].data) != s2["meta_and_deltas_raw"]:
        raise ValueError("S2 metadata Brotli roundtrip failed")
    if math.ceil(s2["action_arithmetic_nbits"] / 8) != len(s2["wire"]) - (
        len(S2_MAGIC)
        + len(_uleb128(S2_MODE_ADAPTIVE_ARITH))
        + len(_uleb128(len(s2["meta_and_deltas_brotli"].data)))
        + len(_uleb128(s2["action_arithmetic_nbits"]))
        + len(s2["meta_and_deltas_brotli"].data)
    ):
        raise ValueError("S2 arithmetic byte accounting mismatch")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    s2_stream_packing = {
        "action_codec": "S2_split_meta_delta_brotli_adaptive_arithmetic_actions",
        "action_alphabet": ACTION_ALPHABET,
        "action_arithmetic_bytes": s2["action_arithmetic_bytes"],
        "action_arithmetic_nbits": s2["action_arithmetic_nbits"],
        "actions_raw_bytes": len(s2["actions_raw"]),
        "actions_raw_sha256": _sha256_bytes(s2["actions_raw"]),
        "meta_and_deltas_brotli_bytes": len(s2["meta_and_deltas_brotli"].data),
        "meta_and_deltas_brotli_params": s2["meta_and_deltas_brotli"].params,
        "meta_and_deltas_raw_bytes": len(s2["meta_and_deltas_raw"]),
        "meta_and_deltas_raw_sha256": _sha256_bytes(s2["meta_and_deltas_raw"]),
        "preserved_segments": list(BASE.NON_ACTION_SEGMENTS),
        "s1_baseline_brotli_bytes": len(s1_choice.data),
        "s1_baseline_raw_bytes": len(s1_raw),
        "s1_baseline_raw_sha256": _sha256_bytes(s1_raw),
    }
    candidates = [
        _build_one(
            source=source,
            candidate_id="pr79_s2_fixed_adaptive_actions",
            payload=_build_fixed_payload(source, s2["wire"]),
            actions_wire=s2["wire"],
            stream_packing={**s2_stream_packing, "payload_container": "fixed_public_slices"},
            output_dir=output_dir,
            unpacker=unpacker,
            force=force,
        ),
        _build_one(
            source=source,
            candidate_id="pr79_s2_p3_adaptive_actions",
            payload=_build_p3_payload(source, s2["wire"]),
            actions_wire=s2["wire"],
            stream_packing={**s2_stream_packing, "payload_container": "self_describing_p3"},
            output_dir=output_dir,
            unpacker=unpacker,
            force=force,
        ),
    ]
    source_action_wire = source.raw_segments["seg_tile_actions.bin"]
    matrix = {
        "byte_matrix": [
            {
                "archive_bytes": source.archive_bytes,
                "archive_sha256": source.archive_sha256,
                "candidate_id": "source_pr79_noop_control",
                "decoded_action_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
                "delta_bytes_vs_pr79": 0,
                "payload_bytes": len(source.payload),
                "seg_tile_actions_wire_bytes": len(source_action_wire),
                "seg_tile_actions_wire_sha256": _sha256_bytes(source_action_wire),
                "status": "source_reference",
            },
            {
                "archive_bytes": source.archive_bytes - len(source_action_wire) + len(s1_choice.data),
                "candidate_id": "halley_s1_fixed_reference",
                "decoded_action_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
                "delta_bytes_vs_pr79": -41,
                "seg_tile_actions_wire_bytes": len(s1_choice.data),
                "seg_tile_actions_wire_sha256": _sha256_bytes(s1_choice.data),
                "status": "prior_s1_reference",
            },
            *candidates,
        ],
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_decision": {
            "best_candidate_id": min(candidates, key=lambda item: item["archive_bytes"])["candidate_id"],
            "exact_eval_justified": min(item["archive_bytes"] for item in candidates) < source.archive_bytes,
            "no_remote_dispatch_performed": True,
            "required_before_dispatch": [
                "claim lane with tools/claim_lane_dispatch.py claim",
                "run exact T4-equivalent CUDA auth eval on the exact archive SHA/bytes",
                "record contest_auth_eval.json, runtime tree hash, and component gates",
            ],
        },
        "evidence_grade": "empirical_lossless_byte_screen",
        "remote_dispatch_performed": False,
        "s2_action_stream": s2_stream_packing,
        "schema": SCHEMA,
        "score_claim": False,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "source_action_stream": {
            "decoded_bytes": len(source.decoded["seg_tile_actions.bin"]),
            "decoded_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
            "record_count": len(source.decoded["seg_tile_actions.bin"]) // 4,
            "wire_bytes": len(source_action_wire),
            "wire_sha256": _sha256_bytes(source_action_wire),
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", matrix)
    return matrix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr79-archive", type=Path, default=BASE.DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    matrix = build_candidates(
        pr79_archive=args.pr79_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    print(json.dumps(matrix["byte_matrix"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
