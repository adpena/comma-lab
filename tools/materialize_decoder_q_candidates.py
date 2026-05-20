#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize selected fixed-length decoder-q candidates as ZIP_STORED archives."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.fec6_decoder_mutations import (  # noqa: E402
    DecoderQMutation,
    apply_q_mutation,
    extract_fec6_decoder_blob,
    prepare_decoder_blob,
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_ids(args: argparse.Namespace) -> list[str]:
    ids = list(args.candidate_id)
    if args.selection:
        selection = _read_json(args.selection.resolve())
        for row in selection.get("queue", []):
            if isinstance(row, dict) and row.get("candidate_id"):
                ids.append(str(row["candidate_id"]))
    out = []
    seen = set()
    for candidate_id in ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(candidate_id)
        if args.limit is not None and len(out) >= int(args.limit):
            break
    if not out:
        raise SystemExit("no candidate ids selected; provide --candidate-id or --selection")
    return out


def _rows_by_id(feasibility: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in feasibility.get("fixed_length_runtime_compatible_rows", []):
        if isinstance(row, dict) and row.get("mutation_id"):
            rows[str(row["mutation_id"])] = row
    return rows


def _materialize_one(
    *,
    out_root: Path,
    candidate_id: str,
    mutation_row: dict[str, Any],
    member_bytes: bytes,
    prepared: Any,
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
    out_dir = out_root / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_bin = out_dir / "archive.bin"
    archive_zip = out_dir / "archive.zip"
    archive_bin.write_bytes(replacement)
    write_stored_archive(archive_zip, replacement)
    manifest = {
        "schema": "fec6_decoder_q_materialized_candidate_v1",
        "candidate_id": candidate_id,
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


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    feasibility = _read_json(args.feasibility.resolve())
    rows = _rows_by_id(feasibility)
    selected = _selected_ids(args)
    missing = [candidate_id for candidate_id in selected if candidate_id not in rows]
    if missing:
        raise SystemExit(f"candidate ids not found in feasibility fixed rows: {missing}")

    archive_bin = args.archive_bin.resolve()
    member_bytes = archive_bin.read_bytes()
    member_sha = sha256_bytes(member_bytes)
    if args.expected_member_sha256 and member_sha != args.expected_member_sha256:
        raise SystemExit(
            f"archive member SHA mismatch: expected {args.expected_member_sha256}, got {member_sha}"
        )
    prepared = prepare_decoder_blob(extract_fec6_decoder_blob(member_bytes))
    out_root = args.output_dir.resolve()
    written = [
        _materialize_one(
            out_root=out_root,
            candidate_id=candidate_id,
            mutation_row=rows[candidate_id],
            member_bytes=member_bytes,
            prepared=prepared,
            brotli_quality=args.brotli_quality,
        )
        for candidate_id in selected
    ]
    return {
        "schema": "fec6_decoder_q_materialization_batch_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/materialize_decoder_q_candidates.py",
        "inputs": {
            "archive_bin": str(archive_bin),
            "archive_bin_sha256": member_sha,
            "feasibility": str(args.feasibility.resolve()),
            "selection": str(args.selection.resolve()) if args.selection else None,
            "candidate_ids": selected,
            "brotli_quality": int(args.brotli_quality),
            "output_dir": str(out_root),
        },
        "summary": {
            "candidate_count": len(written),
        },
        "candidates": written,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-bin", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-member-sha256")
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = materialize(args)
    _write_json(args.manifest_output, payload)
    print(
        json.dumps(
            {
                "manifest_output": str(args.manifest_output),
                "candidate_count": payload["summary"]["candidate_count"],
                "first_candidate": payload["candidates"][0]["candidate_id"] if payload["candidates"] else None,
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
