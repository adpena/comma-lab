#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe legal OP3-V3/FEC6 decoder q-symbol mutations for fixed-runtime viability."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.fec6_decoder_mutations import (  # noqa: E402
    DecoderQMutation,
    Fec6DecoderMutationError,
    apply_q_mutation,
    build_mutation_grid,
    extract_fec6_decoder_blob,
    prepare_decoder_blob,
    probe_q_mutation,
    recompress_prepared_decoder,
    replace_fec6_decoder_blob,
    sha256_bytes,
    write_stored_archive,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_names_from_topk_map(payload: dict[str, Any]) -> list[str]:
    names = []
    for row in payload.get("decoder_tensor_shortlist", []):
        if isinstance(row, dict) and row.get("tensor_name"):
            names.append(str(row["tensor_name"]))
    return names


def _shortlist_by_tensor(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for row in payload.get("decoder_tensor_shortlist", []):
        if isinstance(row, dict) and row.get("tensor_name"):
            out[str(row["tensor_name"])] = row
    return out


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return tuple(values)


def _targeted_mutations_from_topk_map(
    *,
    prepared: Any,
    topk_map: dict[str, Any],
    tensor_names: list[str],
    deltas: tuple[int, ...],
    offset_window: int,
    max_offsets_per_tensor: int,
) -> tuple[DecoderQMutation, ...]:
    """Build q-offset probes by inverting approximate OP3 compressed-byte spans."""

    tensors = prepared.tensor_by_name()
    wanted = set(tensor_names)
    by_tensor: dict[str, list[int]] = {name: [] for name in tensor_names}
    shortlist = {
        str(row["tensor_name"]): row
        for row in topk_map.get("decoder_tensor_shortlist", [])
        if isinstance(row, dict) and row.get("tensor_name")
    }
    for tensor_name in tensor_names:
        row = shortlist.get(tensor_name)
        if not row:
            continue
        approx = row.get("approx_compressed_range", {})
        start = int(approx.get("start", 0))
        length = max(1, int(approx.get("length", 0)))
        tensor = tensors[tensor_name]
        offsets: set[int] = set()
        for byte_index in row.get("top_byte_indices", []):
            rel = (float(int(byte_index) - start) + 0.5) / float(length)
            center = int(np.clip(np.floor(rel * tensor.numel), 0, tensor.numel - 1))
            for q_offset in range(center - int(offset_window), center + int(offset_window) + 1):
                if 0 <= q_offset < tensor.numel:
                    offsets.add(int(q_offset))
        ordered_offsets = sorted(offsets)
        if max_offsets_per_tensor > 0:
            ordered_offsets = ordered_offsets[: int(max_offsets_per_tensor)]
        by_tensor[tensor_name] = ordered_offsets

    rows: list[DecoderQMutation] = []
    seen: set[tuple[str, int, int]] = set()
    # Preserve the shortlist order so higher OP3 tensor mass gets earlier rows.
    ordered_tensor_names = [
        str(row["tensor_name"])
        for row in topk_map.get("decoder_tensor_shortlist", [])
        if isinstance(row, dict) and str(row.get("tensor_name")) in wanted
    ]
    ordered_tensor_names.extend(name for name in tensor_names if name not in ordered_tensor_names)
    for tensor_name in ordered_tensor_names:
        for q_offset in by_tensor.get(tensor_name, []):
            for delta in deltas:
                key = (tensor_name, int(q_offset), int(delta))
                if int(delta) == 0 or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    DecoderQMutation(
                        tensor_name=tensor_name,
                        q_offset=int(q_offset),
                        delta=int(delta),
                    )
                )
    return tuple(rows)


def _probe_one(prepared: Any, mutation: DecoderQMutation, brotli_quality: int) -> dict[str, Any]:
    try:
        return probe_q_mutation(prepared, mutation, brotli_quality=brotli_quality).as_dict()
    except Fec6DecoderMutationError as exc:
        return {
            "mutation": mutation.as_dict(),
            "error": str(exc),
            "fixed_length_runtime_compatible": False,
        }


def _write_candidate(
    candidate_root: Path,
    *,
    member_bytes: bytes,
    prepared: Any,
    mutation_row: dict[str, Any],
    brotli_quality: int,
) -> dict[str, Any]:
    mutation_payload = mutation_row["mutation"]
    mutation = DecoderQMutation(
        tensor_name=str(mutation_payload["tensor_name"]),
        q_offset=int(mutation_payload["q_offset"]),
        delta=int(mutation_payload["delta"]),
    )
    raw, _tensor, _q_before, _q_after = apply_q_mutation(prepared, mutation)
    decoder_blob = recompress_prepared_decoder(prepared, raw, brotli_quality=brotli_quality)
    replacement = replace_fec6_decoder_blob(member_bytes, decoder_blob)
    mutation_id = str(mutation_row["mutation_id"])
    out_dir = candidate_root / mutation_id
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_bin = out_dir / "archive.bin"
    archive_zip = out_dir / "archive.zip"
    archive_bin.write_bytes(replacement)
    write_stored_archive(archive_zip, replacement)
    manifest = {
        "mutation_row": mutation_row,
        "archive_bin_path": str(archive_bin.resolve()),
        "archive_zip_path": str(archive_zip.resolve()),
        "archive_bin_bytes": len(replacement),
        "archive_zip_bytes": archive_zip.stat().st_size,
        "archive_bin_sha256": sha256_bytes(replacement),
        "archive_zip_sha256": _sha256_file(archive_zip),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next": [
            "official inflate.sh raw-output locality control",
            "advisory component-response measurement",
            "exact contest eval before any promotion claim",
        ],
    }
    _write_json(out_dir / "mutation_manifest.json", manifest)
    return manifest


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    archive_bin = args.archive_bin.resolve()
    topk_map_path = args.topk_map.resolve() if args.topk_map else None
    member_bytes = archive_bin.read_bytes()
    member_sha = sha256_bytes(member_bytes)
    if args.expected_member_sha256 and member_sha != args.expected_member_sha256:
        raise SystemExit(
            f"archive member SHA mismatch: expected {args.expected_member_sha256}, got {member_sha}"
        )

    topk_map = _read_json(topk_map_path) if topk_map_path else {}
    decoder_blob = extract_fec6_decoder_blob(member_bytes)
    prepared = prepare_decoder_blob(decoder_blob)
    identity_reencoded = recompress_prepared_decoder(prepared, prepared.raw, brotli_quality=args.brotli_quality)
    identity_same = identity_reencoded == decoder_blob

    tensor_by_name = prepared.tensor_by_name()
    tensor_names = list(args.tensor)
    if not tensor_names:
        tensor_names = _tensor_names_from_topk_map(topk_map)
    if not tensor_names:
        raise SystemExit("no tensors specified and top-K map has no decoder_tensor_shortlist")
    missing = [name for name in tensor_names if name not in tensor_by_name]
    if missing:
        raise SystemExit(f"unknown tensor names: {missing}; available={sorted(tensor_by_name)}")

    deltas = _parse_csv_ints(args.deltas)
    if args.targeted_from_topk_map:
        if not topk_map:
            raise SystemExit("--targeted-from-topk-map requires --topk-map")
        grid = _targeted_mutations_from_topk_map(
            prepared=prepared,
            topk_map=topk_map,
            tensor_names=tensor_names,
            deltas=deltas,
            offset_window=args.targeted_offset_window,
            max_offsets_per_tensor=args.max_offsets_per_tensor,
        )
    else:
        grid = build_mutation_grid(
            tensor_by_name,
            tensor_names,
            deltas=deltas,
            max_offsets_per_tensor=args.max_offsets_per_tensor,
        )
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        rows = [_probe_one(prepared, mutation, args.brotli_quality) for mutation in grid]
    else:
        with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(lambda m: _probe_one(prepared, m, args.brotli_quality), grid))

    shortlist = _shortlist_by_tensor(topk_map)
    for row in rows:
        mutation = row.get("mutation")
        if isinstance(mutation, dict):
            tensor_name = str(mutation.get("tensor_name"))
            if tensor_name in shortlist:
                row["op3v3_target_evidence"] = shortlist[tensor_name]

    length_counter = Counter()
    fixed_rows = []
    errors = 0
    for row in rows:
        if "error" in row:
            errors += 1
            continue
        length_counter[int(row["length_delta"])] += 1
        if bool(row["fixed_length_runtime_compatible"]):
            fixed_rows.append(row)
    fixed_rows.sort(
        key=lambda row: (
            -float(row.get("op3v3_target_evidence", {}).get("score_impact_abs_sum", 0.0)),
            str(row["mutation"]["tensor_name"]),
            int(row["mutation"]["q_offset"]),
            int(row["mutation"]["delta"]),
        )
    )

    written = []
    if args.write_candidates:
        if args.candidate_output_dir is None:
            raise SystemExit("--write-candidates requires --candidate-output-dir")
        candidate_root = args.candidate_output_dir.resolve()
        for row in fixed_rows[: int(args.write_candidates)]:
            written.append(
                _write_candidate(
                    candidate_root,
                    member_bytes=member_bytes,
                    prepared=prepared,
                    mutation_row=row,
                    brotli_quality=args.brotli_quality,
                )
            )

    return {
        "schema": "op3v3_fec6_decoder_mutation_feasibility_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/probe_op3v3_decoder_mutation_feasibility.py",
        "inputs": {
            "archive_bin_path": str(archive_bin),
            "archive_bin_sha256": member_sha,
            "topk_map_path": str(topk_map_path) if topk_map_path else None,
            "brotli_quality": int(args.brotli_quality),
            "jobs": jobs,
            "tensor_names": tensor_names,
            "deltas": list(deltas),
            "max_offsets_per_tensor": int(args.max_offsets_per_tensor),
            "targeted_from_topk_map": bool(args.targeted_from_topk_map),
            "targeted_offset_window": int(args.targeted_offset_window),
        },
        "decoder": prepared.as_dict(),
        "identity_reencode": {
            "same_bytes": identity_same,
            "reencoded_len": len(identity_reencoded),
            "reencoded_sha256": sha256_bytes(identity_reencoded),
            "source_len": len(decoder_blob),
            "source_sha256": prepared.decoder_sha256,
        },
        "summary": {
            "mutation_count": len(rows),
            "error_count": errors,
            "fixed_length_runtime_compatible_count": len(fixed_rows),
            "length_delta_histogram": {str(k): v for k, v in sorted(length_counter.items())},
            "candidate_archive_count": len(written),
        },
        "mutation_rows": rows,
        "fixed_length_runtime_compatible_rows": fixed_rows,
        "written_candidates": written,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "raw_byte_flips_valid_for_promotion": False,
            "notes": (
                "Rows mutate decoded q-symbols and recompress legal PR101 split-Brotli streams. "
                "Only fixed-length rows can be tested with the stock PR110 fixed-offset runtime."
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", type=Path, required=True)
    parser.add_argument("--topk-map", type=Path)
    parser.add_argument("--expected-member-sha256")
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--deltas", default="-1,1")
    parser.add_argument("--max-offsets-per-tensor", type=int, default=512)
    parser.add_argument("--targeted-from-topk-map", action="store_true")
    parser.add_argument("--targeted-offset-window", type=int, default=1)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--write-candidates", type=int, default=0)
    parser.add_argument("--candidate-output-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mutation_count": payload["summary"]["mutation_count"],
                "fixed_length_runtime_compatible_count": payload["summary"][
                    "fixed_length_runtime_compatible_count"
                ],
                "length_delta_histogram": payload["summary"]["length_delta_histogram"],
                "candidate_archive_count": payload["summary"]["candidate_archive_count"],
                "identity_reencode_same_bytes": payload["identity_reencode"]["same_bytes"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
