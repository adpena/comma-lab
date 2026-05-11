#!/usr/bin/env python3
"""Profile PR85 model-payload self-compression opportunities.

This is a local static profiler. It slices the charged PR85 ``model`` segment
through the canonical PR85 bundle parser, fingerprints the exact bytes, probes
lossless compressibility, and emits planning artifacts. It does not inflate
frames, load scorers, run CUDA, dispatch jobs, or make score claims.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tac.repo_io import json_text, read_json, sha256_bytes, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr85_bundle import (  # noqa: E402
    Pr85BundleError,
    SEGMENT_ORDER,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


SCHEMA = "pr85_model_payload_self_compression_profile_v1"
TOOL = "experiments/profile_pr85_model_payload_self_compression.py"
DEFAULT_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_BUNDLE_PROFILE_JSON = DEFAULT_INTAKE_DIR / "profile_pr85_bundle.json"
DEFAULT_BIT_BUDGET_JSON = (
    REPO_ROOT
    / "experiments/results/pr85_archive_bit_budget_20260504_codex/profile_pr85_archive_bit_budget.json"
)
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "experiments/results/pr85_model_payload_self_compression_20260504_worker"
)
DEFAULT_JSON_OUT = DEFAULT_RESULTS_DIR / "profile_pr85_model_payload_self_compression.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_RESULTS_DIR / "profile_pr85_model_payload_self_compression.md"
ORIGINAL_VIDEO_BYTES = 37_545_489


class ProfileError(RuntimeError):
    """Raised when the PR85 model profile cannot be built safely."""


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json_text(payload).encode("utf-8")


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path | None, *, required: bool) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        if required:
            raise ProfileError(f"required JSON artifact is missing: {_rel(path)}")
        return {}
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"malformed JSON artifact: {_rel(path)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProfileError(f"JSON artifact root is not an object: {_rel(path)}")
    return payload


def _magic_ascii(data: bytes, *, n: int = 8) -> str:
    out = []
    for value in data[:n]:
        if 32 <= value <= 126:
            out.append(chr(value))
        else:
            out.append(".")
    return "".join(out)


def _entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = float(len(data))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _longest_run(data: bytes) -> dict[str, Any]:
    if not data:
        return {"byte": None, "length": 0, "start": None}
    best_byte = data[0]
    best_start = 0
    best_len = 1
    cur_byte = data[0]
    cur_start = 0
    cur_len = 1
    for idx, value in enumerate(data[1:], start=1):
        if value == cur_byte:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_byte = cur_byte
                best_start = cur_start
                best_len = cur_len
            cur_byte = value
            cur_start = idx
            cur_len = 1
    if cur_len > best_len:
        best_byte = cur_byte
        best_start = cur_start
        best_len = cur_len
    return {"byte": int(best_byte), "length": int(best_len), "start": int(best_start)}


def _byte_profile(data: bytes) -> dict[str, Any]:
    entropy = _entropy_bits_per_byte(data)
    counts = Counter(data)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    entropy_bytes = int(math.ceil(entropy * len(data) / 8.0))
    zero_count = int(counts.get(0, 0))
    ff_count = int(counts.get(255, 0))
    return {
        "bytes": int(len(data)),
        "sha256": sha256_bytes(data),
        "magic_hex": data[:8].hex(),
        "magic_ascii": _magic_ascii(data),
        "entropy_bits_per_byte": round(entropy, 12),
        "zero_order_entropy_bytes": entropy_bytes,
        "zero_order_entropy_delta_vs_input_bytes": entropy_bytes - len(data),
        "unique_byte_count": int(len(counts)),
        "zero_byte_fraction": round(zero_count / len(data), 12) if data else None,
        "ff_byte_fraction": round(ff_count / len(data), 12) if data else None,
        "longest_equal_byte_run": _longest_run(data),
        "top_byte_frequencies": [
            {
                "byte": int(value),
                "hex": f"{value:02x}",
                "count": int(count),
                "fraction": round(count / len(data), 12) if data else None,
            }
            for value, count in top
        ],
    }


def _try_brotli_decompress(data: bytes) -> tuple[bytes | None, str | None]:
    try:
        import brotli
    except ImportError:
        return None, "brotli_python_package_missing"
    try:
        return brotli.decompress(data), None
    except brotli.error as exc:
        return None, f"brotli_decode_failed:{exc}"


def _try_brotli_compress(data: bytes, *, quality: int) -> bytes | None:
    try:
        import brotli
    except ImportError:
        return None
    return brotli.compress(data, quality=quality)


def _compression_probes(data: bytes) -> dict[str, Any]:
    probes: dict[str, Any] = {
        "input_bytes": int(len(data)),
        "zlib_9_bytes": int(len(zlib.compress(data, level=9))),
        "lzma_preset9_bytes": int(len(lzma.compress(data, preset=9))),
    }
    for quality in (5, 9, 11):
        compressed = _try_brotli_compress(data, quality=quality)
        if compressed is not None:
            probes[f"brotli_q{quality}_bytes"] = int(len(compressed))
    byte_items = [
        (name, int(value))
        for name, value in probes.items()
        if name.endswith("_bytes") and name != "input_bytes"
    ]
    if byte_items:
        best_codec, best_bytes = min(byte_items, key=lambda item: (item[1], item[0]))
        probes["best_probe"] = {"codec": best_codec, "bytes": best_bytes}
        probes["best_probe_delta_vs_input"] = best_bytes - len(data)
        probes["best_probe_ratio"] = round(best_bytes / len(data), 12) if data else None
        probes["generic_recompression_improves"] = best_bytes < len(data)
    else:
        probes["best_probe"] = None
        probes["best_probe_delta_vs_input"] = None
        probes["best_probe_ratio"] = None
        probes["generic_recompression_improves"] = False
    return probes


def _window_profiles(data: bytes, *, window_bytes: int = 4096, top_k: int = 10) -> dict[str, Any]:
    if not data:
        return {"window_bytes": window_bytes, "windows": [], "summary": {}}
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(data), window_bytes):
        chunk = data[offset : offset + window_bytes]
        entropy = _entropy_bits_per_byte(chunk)
        zlib_len = len(zlib.compress(chunk, level=9))
        zero_fraction = chunk.count(0) / len(chunk)
        row = {
            "offset": int(offset),
            "bytes": int(len(chunk)),
            "entropy_bits_per_byte": round(entropy, 12),
            "zlib_9_bytes": int(zlib_len),
            "zlib_9_ratio": round(zlib_len / len(chunk), 12),
            "zero_byte_fraction": round(zero_fraction, 12),
            "unique_byte_count": int(len(set(chunk))),
            "classification": _classify_region(entropy, zlib_len / len(chunk), zero_fraction),
        }
        rows.append(row)
    low_entropy = sorted(rows, key=lambda row: (row["entropy_bits_per_byte"], row["offset"]))[:top_k]
    high_entropy = sorted(rows, key=lambda row: (-row["entropy_bits_per_byte"], row["offset"]))[:top_k]
    return {
        "window_bytes": int(window_bytes),
        "window_count": int(len(rows)),
        "low_entropy_or_redundant_windows": low_entropy,
        "high_entropy_or_already_compressed_windows": high_entropy,
        "summary": {
            "low_entropy_window_count": int(
                sum(1 for row in rows if row["entropy_bits_per_byte"] < 6.5)
            ),
            "already_compressed_like_window_count": int(
                sum(1 for row in rows if row["classification"] == "already_compressed_like")
            ),
            "zero_heavy_window_count": int(
                sum(1 for row in rows if row["classification"] == "zero_or_constant_heavy")
            ),
        },
    }


def _classify_region(entropy: float, zlib_ratio: float, zero_fraction: float) -> str:
    if zero_fraction >= 0.8 or entropy <= 2.0:
        return "zero_or_constant_heavy"
    if entropy >= 7.65 and zlib_ratio >= 0.98:
        return "already_compressed_like"
    if zlib_ratio < 0.85 or entropy < 6.5:
        return "recode_candidate"
    return "mixed"


def _read_pr85_archive(archive: Path) -> tuple[bytes, dict[str, Any]]:
    if not archive.is_file():
        raise ProfileError(f"PR85 archive is missing: {_rel(archive)}")
    try:
        with zipfile.ZipFile(archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if len(infos) != 1:
                raise ProfileError(
                    f"PR85 archive must contain exactly one non-directory member, got {len(infos)}"
                )
            info = infos[0]
            validate_pr85_member_name(info.filename)
            raw = zf.read(info.filename)
    except (zipfile.BadZipFile, Pr85BundleError) as exc:
        raise ProfileError(f"failed to read strict PR85 archive: {exc}") from exc
    return raw, {
        "archive_path": _rel(archive),
        "archive_bytes": int(archive.stat().st_size),
        "archive_sha256": sha256_file(archive),
        "member_name": info.filename,
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
        "member_crc32_hex": f"{info.CRC:08x}",
        "member_sha256": sha256_bytes(raw),
        "zip_stored": bool(info.compress_type == zipfile.ZIP_STORED),
    }


def _profile_artifact_consistency(
    *,
    archive_info: Mapping[str, Any],
    bundle: Any,
    bit_budget: Mapping[str, Any],
    bundle_profile: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: list[str] = []
    budget_archive = bit_budget.get("archive", {}) if isinstance(bit_budget, Mapping) else {}
    if isinstance(budget_archive, Mapping):
        expected_sha = budget_archive.get("archive_sha256")
        expected_bytes = budget_archive.get("archive_bytes") or budget_archive.get("archive_size_bytes")
        if expected_sha and expected_sha != archive_info["archive_sha256"]:
            mismatches.append("bit_budget_archive_sha256")
        if expected_bytes is not None and int(expected_bytes) != archive_info["archive_bytes"]:
            mismatches.append("bit_budget_archive_bytes")
    profile_archive = bundle_profile.get("archive", {}) if isinstance(bundle_profile, Mapping) else {}
    if isinstance(profile_archive, Mapping):
        expected_sha = profile_archive.get("archive_sha256")
        expected_bytes = profile_archive.get("archive_size_bytes") or profile_archive.get("archive_bytes")
        if expected_sha and expected_sha != archive_info["archive_sha256"]:
            mismatches.append("bundle_profile_archive_sha256")
        if expected_bytes is not None and int(expected_bytes) != archive_info["archive_bytes"]:
            mismatches.append("bundle_profile_archive_bytes")
    profile_segments = bundle_profile.get("segments", []) if isinstance(bundle_profile, Mapping) else []
    if isinstance(profile_segments, list):
        by_name = {row.get("name"): row for row in profile_segments if isinstance(row, Mapping)}
        model_row = by_name.get("model")
        if isinstance(model_row, Mapping):
            if int(model_row.get("bytes", -1)) != len(bundle.segments["model"]):
                mismatches.append("bundle_profile_model_bytes")
            if model_row.get("sha256") and model_row["sha256"] != sha256_bytes(bundle.segments["model"]):
                mismatches.append("bundle_profile_model_sha256")
    return {
        "checked": True,
        "mismatches": mismatches,
        "matches": not mismatches,
        "bit_budget_json_present": bool(bit_budget),
        "bundle_profile_json_present": bool(bundle_profile),
    }


def _identify_model_container(encoded: bytes, decoded: bytes | None, decode_error: str | None) -> dict[str, Any]:
    decoded_magic = decoded[:8] if decoded is not None else b""
    recognized = decoded is not None and decoded_magic[:3] in (b"QH0", b"QM0")
    return {
        "encoded_container": "brotli_stream" if decoded is not None else "unknown_or_non_brotli",
        "encoded_magic_hex": encoded[:8].hex(),
        "encoded_magic_ascii": _magic_ascii(encoded),
        "brotli_decodable": decoded is not None,
        "brotli_decode_error": decode_error,
        "decoded_magic_hex": decoded_magic.hex(),
        "decoded_magic_ascii": _magic_ascii(decoded_magic),
        "decoded_payload_kind": (
            "pr85_qh0_joint_frame_model"
            if decoded_magic[:3] == b"QH0"
            else "pr85_qm0_joint_frame_model"
            if decoded_magic[:3] == b"QM0"
            else "unknown_decoded_model_payload"
            if decoded is not None
            else None
        ),
        "recognized_runtime_model_payload": recognized,
        "hilo_split_qh0": bool(decoded_magic[:3] == b"QH0"),
        "canonical_loader": (
            "tac.qh0_renderer_codec.decode_qh0_state_dict"
            if recognized
            else None
        ),
        "decoder_used_for_profile": "brotli only; no model/scorer runtime imported",
    }


def _build_candidate_routes(
    *,
    encoded: bytes,
    decoded: bytes | None,
    encoded_probes: Mapping[str, Any],
    decoded_probes: Mapping[str, Any] | None,
    container: Mapping[str, Any],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    best_lossless_bytes: int | None = None
    best_lossless_codec: str | None = None
    if decoded is not None:
        for quality in (5, 9, 11):
            compressed = _try_brotli_compress(decoded, quality=quality)
            if compressed is None:
                continue
            size = len(compressed)
            if best_lossless_bytes is None or size < best_lossless_bytes:
                best_lossless_bytes = size
                best_lossless_codec = f"brotli_q{quality}_decoded_qh0_recode"
    if best_lossless_bytes is not None:
        saved = len(encoded) - best_lossless_bytes
        routes.append(
            {
                "route_id": "lossless_brotli_recode_decoded_model_segment",
                "route_type": "lossless_container_recode",
                "actionability_rank": 1 if saved > 0 else 3,
                "planning_only": True,
                "score_claim": False,
                "dispatch_performed": False,
                "bytes_at_stake": int(len(encoded)),
                "best_static_candidate_codec": best_lossless_codec,
                "estimated_model_segment_delta_bytes": int(best_lossless_bytes - len(encoded)),
                "estimated_model_segment_bytes_saved": int(max(0, saved)),
                "contest_faithful_requirements": [
                    "replace only PR85 model segment bytes inside the single-member x bundle",
                    "repack with tac.pr85_bundle and prove bundle roundtrip for unchanged segments",
                    "prove brotli-decompressed model bytes are identical to the source decoded QH0 bytes",
                    "run runtime output parity before any exact CUDA auth eval",
                ],
                "dispatchable_now": False,
                "dispatch_blocker": "planning-only static profile; no runtime parity or CUDA auth eval performed",
                "recommendation": (
                    "Implement as the first contest-faithful route if the static delta is negative."
                    if saved > 0
                    else "Do not prioritize: generic Brotli recoding did not beat the current charged bytes."
                ),
            }
        )
    routes.append(
        {
            "route_id": "qh0_record_level_repack_or_serializer",
            "route_type": "score_affecting_model_payload_rewrite",
            "actionability_rank": 2,
            "planning_only": True,
            "score_claim": False,
            "dispatch_performed": False,
            "bytes_at_stake": int(len(encoded)),
            "recognized_qh0_payload": bool(container.get("recognized_runtime_model_payload")),
            "estimated_model_segment_bytes_saved": None,
            "contest_faithful_requirements": [
                "use the reviewed QH0 parser/loader contract, not torch.load or replay-side heuristics",
                "add a deterministic QH0 serializer with byte/tensor parity tests",
                "prove runtime output parity on the exact PR85 archive before exact eval",
                "score only through archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
            ],
            "dispatchable_now": False,
            "dispatch_blocker": "no reviewed QH0 serializer or runtime parity artifact in this profile",
            "recommendation": "Top byte-saving route when lossless Brotli recode is non-improving: build a deterministic QH0 record-level serializer/repacker and target low-entropy decoded model records under runtime parity.",
        }
    )
    routes.append(
        {
            "route_id": "generic_wrapper_noop_guard",
            "route_type": "negative_guardrail",
            "actionability_rank": 4,
            "planning_only": True,
            "score_claim": False,
            "dispatch_performed": False,
            "bytes_at_stake": int(len(encoded)),
            "encoded_best_probe_delta_vs_input": encoded_probes.get("best_probe_delta_vs_input"),
            "already_compressed_by_generic_probes": not bool(
                encoded_probes.get("generic_recompression_improves")
            ),
            "decoded_best_probe_delta_vs_input": (
                None if decoded_probes is None else decoded_probes.get("best_probe_delta_vs_input")
            ),
            "recommendation": "Do not wrap the charged model segment in another generic compressor unless the decoded-byte recode proves a smaller byte-closed model segment.",
        }
    )
    return sorted(
        routes,
        key=lambda row: (
            int(row.get("actionability_rank", 99)),
            row.get("estimated_model_segment_bytes_saved") is None,
            -int(row.get("estimated_model_segment_bytes_saved") or 0),
            row["route_id"],
        ),
    )


def _markdown_report(profile: Mapping[str, Any]) -> str:
    model = profile["model_segment"]
    container = model["container"]
    top = profile["candidate_routes"][0] if profile["candidate_routes"] else {}
    lines = [
        "# PR85 Model Payload Self-Compression Profile",
        "",
        "- planning_only: true",
        "- score_claim: false",
        "- dispatch_performed: false",
        f"- archive: `{profile['archive']['archive_path']}`",
        f"- archive_sha256: `{profile['archive']['archive_sha256']}`",
        f"- model_bytes: {model['encoded']['bytes']}",
        f"- model_sha256: `{model['encoded']['sha256']}`",
        f"- encoded_container: {container['encoded_container']}",
        f"- decoded_payload_kind: {container['decoded_payload_kind']}",
        f"- decoded_bytes: {None if model.get('decoded') is None else model['decoded']['bytes']}",
        "",
        "## Static Probes",
        "",
        f"- encoded entropy bits/byte: {model['encoded']['entropy_bits_per_byte']}",
        f"- encoded best recompression delta bytes: {model['encoded_compression_probes']['best_probe_delta_vs_input']}",
        f"- decoded best recompression delta bytes: {None if model.get('decoded_compression_probes') is None else model['decoded_compression_probes']['best_probe_delta_vs_input']}",
        "",
        "## Candidate Routes",
        "",
    ]
    for route in profile["candidate_routes"]:
        lines.extend(
            [
                f"### {route['route_id']}",
                "",
                f"- route_type: {route['route_type']}",
                f"- estimated_model_segment_bytes_saved: {route.get('estimated_model_segment_bytes_saved')}",
                f"- dispatchable_now: {route.get('dispatchable_now', False)}",
                f"- recommendation: {route['recommendation']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Implementable Route",
            "",
            f"`{top.get('route_id')}`: {top.get('recommendation')}",
            "",
        ]
    )
    return "\n".join(lines)


def build_profile(
    archive: Path = DEFAULT_ARCHIVE,
    *,
    bit_budget_json: Path | None = DEFAULT_BIT_BUDGET_JSON,
    bundle_profile_json: Path | None = DEFAULT_BUNDLE_PROFILE_JSON,
    window_bytes: int = 4096,
) -> dict[str, Any]:
    archive = Path(archive)
    raw, archive_info = _read_pr85_archive(archive)
    try:
        bundle = parse_pr85_bundle(raw)
    except Pr85BundleError as exc:
        raise ProfileError(f"canonical PR85 parser rejected bundle: {exc}") from exc
    if "model" not in bundle.segments:
        raise ProfileError("canonical PR85 parser produced no model segment")
    model = bundle.segments["model"]
    decoded, decode_error = _try_brotli_decompress(model)
    bit_budget = _read_json(bit_budget_json, required=False)
    bundle_profile = _read_json(bundle_profile_json, required=False)
    container = _identify_model_container(model, decoded, decode_error)
    encoded_probes = _compression_probes(model)
    decoded_probes = _compression_probes(decoded) if decoded is not None else None
    routes = _build_candidate_routes(
        encoded=model,
        decoded=decoded,
        encoded_probes=encoded_probes,
        decoded_probes=decoded_probes,
        container=container,
    )
    model_profile: dict[str, Any] = {
        "segment_name": "model",
        "offset_in_x_member": int(bundle.segment_offsets["model"]),
        "encoded": _byte_profile(model),
        "container": container,
        "encoded_compression_probes": encoded_probes,
        "encoded_windows": _window_profiles(model, window_bytes=window_bytes),
    }
    if decoded is not None:
        model_profile["decoded"] = _byte_profile(decoded)
        model_profile["decoded_compression_probes"] = decoded_probes
        model_profile["decoded_windows"] = _window_profiles(decoded, window_bytes=window_bytes)
    else:
        model_profile["decoded"] = None
        model_profile["decoded_compression_probes"] = None
        model_profile["decoded_windows"] = None
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "deterministic": True,
        "evidence_grade": "planning_only_static_model_payload_profile",
        "cuda_score_truth": "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA",
        "archive": archive_info,
        "inputs": {
            "archive": _rel(archive),
            "bit_budget_json": _rel(bit_budget_json),
            "bundle_profile_json": _rel(bundle_profile_json),
        },
        "bundle": {
            "format": bundle.format,
            "header_bytes": int(bundle.header_bytes),
            "segment_order": list(SEGMENT_ORDER),
            "segment_lengths": bundle.segment_lengths,
            "segment_offsets": {key: int(value) for key, value in bundle.segment_offsets.items()},
        },
        "artifact_consistency": _profile_artifact_consistency(
            archive_info=archive_info,
            bundle=bundle,
            bit_budget=bit_budget,
            bundle_profile=bundle_profile,
        ),
        "score_rate_formula": {
            "formula_only": True,
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_score_per_byte": 25.0 / ORIGINAL_VIDEO_BYTES,
            "model_rate_score_contribution": len(model) * 25.0 / ORIGINAL_VIDEO_BYTES,
            "score_claim_from_this_profile": False,
        },
        "model_segment": model_profile,
        "candidate_routes": routes,
        "top_implementable_route": routes[0] if routes else None,
        "score_claim_refusal": {
            "score_claim": False,
            "dispatch_performed": False,
            "reason": "Static byte and compressibility profiling does not execute inflate, runtime parity, or CUDA auth eval.",
        },
    }


def write_outputs(profile: Mapping[str, Any], *, json_out: Path, markdown_out: Path | None) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_bytes(_json_bytes(profile))
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(_markdown_report(profile), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--bit-budget-json", type=Path, default=DEFAULT_BIT_BUDGET_JSON)
    parser.add_argument("--bundle-profile-json", type=Path, default=DEFAULT_BUNDLE_PROFILE_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--window-bytes", type=int, default=4096)
    args = parser.parse_args(argv)

    try:
        profile = build_profile(
            args.archive.resolve(),
            bit_budget_json=args.bit_budget_json.resolve() if args.bit_budget_json else None,
            bundle_profile_json=args.bundle_profile_json.resolve() if args.bundle_profile_json else None,
            window_bytes=args.window_bytes,
        )
    except ProfileError as exc:
        print(f"{TOOL}: failed closed: {exc}", file=sys.stderr)
        return 2
    write_outputs(profile, json_out=args.json_out.resolve(), markdown_out=args.markdown_out.resolve())
    print(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
