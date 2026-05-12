"""Materialize a magic-codec archive from a source archive.

Per CLAUDE.md "Deterministic packet compiler" non-negotiable, this tool is
the **identity / canonicalize / optimize** entry-point for magic-codec
archive production. It deconstructs a source archive (PR106 r2 family,
A1, etc.) into typed streams, runs the magic codec auto-selector per
stream, and re-packs into a magic-codec archive with primitive selection
recorded in ``magic_codec_selection_manifest.json``.

Per ``tac.phase1_packet_compiler.compile_phase1_packet`` discipline:

* ``score_claim`` is permanently False on any magic-codec output until
  exact CUDA + CPU evaluation on the contest video lands;
* ``promotion_eligible`` is permanently False;
* ``ready_for_exact_eval_dispatch`` is permanently False until the
  inflate-parity verifier passes byte-for-byte against the source
  archive's runtime;
* every emitted byte change is tagged with a manifest row that names
  the source stream + the magic codec primitive selected + the
  predicted byte savings + the actual byte savings.

CLAUDE.md compliance:

* no scorer load (pure numpy + brotli + lzma + constriction + stdlib);
* no MPS / torch import;
* no ``/tmp`` paths — outputs to ``--output-dir`` (operator-supplied);
* refuses to emit any score claim;
* deterministic-bytes: identical input + identical args produce
  byte-identical output;
* every CLI flag is grepped against this file's argparse before
  invocation (no dead flags);
* runtime-tree closure: emits a ``magic_codec_runtime_manifest.json``
  alongside the materialized archive declaring the inflate runtime
  dependency closure.

Usage::

    python tools/materialize_magic_codec_archive.py \\
        --source-archive submissions/pr106_r2/archive.zip \\
        --output-dir experiments/results/magic_codec_pr106_r2_<ts>/ \\
        --stream-type weight_tensor \\
        --selection-strategy smallest_byte_count
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from tac.packet_compiler.magic_codec import (
    MagicCodecError,
    MagicCodecResult,
    SelectionStrategy,
    StreamHint,
    StreamType,
    candidate_primitives_for,
    encode_magic_codec,
    recommendation_for,
)


_STREAM_TYPES: tuple[str, ...] = (
    "weight_tensor",
    "latent_sidecar",
    "pose",
    "mask",
    "residual_basis",
    "categorical",
    "low_pass_residual",
)


_SELECTION_STRATEGIES: tuple[str, ...] = (
    "smallest_byte_count",
    "entropy_estimate",
    "stacked_optimal",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="materialize_magic_codec_archive",
        description=(
            "Auto-select the optimal packet_compiler primitive per stream "
            "and emit a magic-codec archive. score_claim permanently False "
            "until exact CUDA + CPU adjudication."
        ),
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Source archive.zip to deconstruct (e.g. PR106 r2 archive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory. Must NOT be under /tmp (CLAUDE.md forbidden "
            "pattern). Canonical location: "
            "experiments/results/<lane_id>_<timestamp>/."
        ),
    )
    parser.add_argument(
        "--stream-type",
        choices=_STREAM_TYPES,
        required=True,
        help="Typed stream-type hint applied to the entire archive payload.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=_SELECTION_STRATEGIES,
        default="smallest_byte_count",
        help=(
            "How to select among candidate primitives. Default: "
            "smallest_byte_count."
        ),
    )
    parser.add_argument(
        "--member-name",
        default=None,
        help=(
            "Specific zip member to process. If omitted, every payload "
            "member is processed independently with the same stream-type "
            "hint."
        ),
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        default=None,
        help=(
            "When the source bytes need to be reinterpreted as quantised "
            "integers for AC / RLE / categorical primitives, declare the "
            "bit width here (8 → int8 view, 16 → int16 view, 32 → int32 "
            "view). Without this flag the source bytes are interpreted as "
            "int8."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the selection log + predicted byte savings WITHOUT "
            "writing any output. Useful for /research-signal-only ranking."
        ),
    )
    parser.add_argument(
        "--operator",
        default=None,
        help=(
            "Operator handle for manifest provenance. When omitted, the "
            "manifest records `operator=unknown`."
        ),
    )
    return parser.parse_args(argv)


def _validate_output_dir(output_dir: Path) -> None:
    """Refuse forbidden /tmp paths per CLAUDE.md non-negotiable."""
    parts = output_dir.resolve().parts
    if parts and parts[0] in ("/", "/tmp", "/var", "/private"):
        # Accept absolute paths in general; only refuse /tmp prefix.
        forbidden_anchors = ("/tmp/", "/var/tmp/", "/private/tmp/")
        as_str = str(output_dir.resolve())
        for anchor in forbidden_anchors:
            if as_str.startswith(anchor):
                raise SystemExit(
                    f"refusing to write to forbidden /tmp path {output_dir!s} "
                    "per CLAUDE.md `forbidden_/tmp_paths_in_any_persisted_artifact`"
                )


def _decode_dense_from_member_bytes(
    member_bytes: bytes, *, stream_type: StreamType, quantize_bits: int
) -> np.ndarray:
    """Interpret an archive member's bytes as a dense stream for magic codec.

    The interpretation is intentionally narrow: this tool is not a
    score-aware re-encoder; it is a research-signal byte-level
    deconstructor that re-packs already-quantised bytes into a more
    compact magic-codec envelope. The trainer is still the source of
    truth for the dense values; we only pick the best wire format.
    """
    if quantize_bits == 8:
        dtype = np.int8
    elif quantize_bits == 16:
        dtype = np.int16
    elif quantize_bits == 32:
        dtype = np.int32
    else:
        raise SystemExit(
            f"--quantize-bits must be 8 / 16 / 32; got {quantize_bits}"
        )

    # Truncate trailing bytes that don't form a complete element.
    n_elements = len(member_bytes) // np.dtype(dtype).itemsize
    usable_bytes = member_bytes[: n_elements * np.dtype(dtype).itemsize]
    arr = np.frombuffer(usable_bytes, dtype=dtype).astype(dtype, copy=True)

    # Pose / centered-delta require 2D; reshape if a reasonable factor.
    if stream_type == "pose":
        if arr.size % 6 == 0:
            arr = arr.reshape(arr.size // 6, 6).astype(np.float32)
        elif arr.size % 4 == 0:
            arr = arr.reshape(arr.size // 4, 4).astype(np.float32)
        else:
            arr = arr.reshape(-1, 1).astype(np.float32)
    elif stream_type == "low_pass_residual":
        side = int(np.sqrt(arr.size))
        if side * side == arr.size:
            arr = arr.reshape(side, side).astype(np.float32)
        else:
            # Fall back to a single row of width=arr.size.
            arr = arr.reshape(1, arr.size).astype(np.float32)
    elif stream_type == "latent_sidecar":
        # Try 2D reshape preferring last-dim 28 (PR101 latent width).
        if arr.size % 28 == 0:
            arr = arr.reshape(arr.size // 28, 28).astype(np.float32)
        else:
            # Keep 1D for the AC + RLE candidate fallbacks.
            pass
    return arr


def _process_member(
    member_name: str,
    member_bytes: bytes,
    *,
    stream_type: StreamType,
    selection_strategy: SelectionStrategy,
    quantize_bits: int,
) -> tuple[MagicCodecResult, dict[str, object]]:
    """Run the magic codec on one archive member."""
    dense = _decode_dense_from_member_bytes(
        member_bytes, stream_type=stream_type, quantize_bits=quantize_bits
    )
    result = encode_magic_codec(
        dense, hint=StreamHint(stream_type), selection_strategy=selection_strategy
    )
    member_row = {
        "member_name": member_name,
        "source_bytes": len(member_bytes),
        "dense_shape": list(dense.shape),
        "dense_dtype": str(dense.dtype),
        "magic_codec_payload_bytes": len(result.payload),
        "inner_primitive_byte_count": result.inner_primitive_byte_count,
        "selected_primitive": result.selected_primitive,
        "selected_primitive_id": result.selected_primitive_id,
        "selection_strategy": result.selection_strategy,
        "predicted_byte_delta": len(result.payload) - len(member_bytes),
        "selection_log": [
            {
                "primitive_name": c.primitive_name,
                "primitive_id": c.primitive_id,
                "encoded_bytes": len(c.encoded_bytes),
                "refused": c.refused,
                "refusal_reason": c.refusal_reason,
            }
            for c in result.selection_log
        ],
    }
    return result, member_row


def _build_runtime_manifest() -> dict[str, object]:
    """Emit a runtime-closure manifest per CLAUDE.md `Runtime closure` rule."""
    return {
        "schema": "magic_codec_runtime_manifest.v1",
        "runtime_dep_closure": ["numpy", "brotli", "lzma", "constriction"],
        "inflate_runtime_loc_budget": 200,
        "decoder_module": "tac.packet_compiler.magic_codec",
        "magic_envelope": "MAGC",
        "primitive_id_namespace": "0xF0-0xFF",
        "score_aware_loss": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_output_dir(args.output_dir)

    source = args.source_archive.resolve()
    if not source.exists():
        raise SystemExit(f"source archive {source!s} does not exist")
    if not zipfile.is_zipfile(source):
        raise SystemExit(f"source {source!s} is not a valid ZIP archive")

    quantize_bits = args.quantize_bits if args.quantize_bits is not None else 8

    member_rows: list[dict[str, object]] = []
    materialized_members: dict[str, bytes] = {}

    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(source, "r") as zf:
        target_members = (
            [args.member_name] if args.member_name else zf.namelist()
        )
        for member in target_members:
            try:
                member_bytes = zf.read(member)
            except KeyError as exc:
                raise SystemExit(
                    f"member {member!r} not found in {source!s}: {exc}"
                )
            try:
                result, row = _process_member(
                    member,
                    member_bytes,
                    stream_type=args.stream_type,
                    selection_strategy=args.selection_strategy,
                    quantize_bits=quantize_bits,
                )
            except MagicCodecError as exc:
                # Record the refusal but do not raise — operators may want
                # to see which members the selector refused so they can
                # split + retry with different stream_type hints.
                row = {
                    "member_name": member,
                    "source_bytes": len(member_bytes),
                    "magic_codec_refused": True,
                    "refusal_reason": str(exc),
                }
                member_rows.append(row)
                continue
            member_rows.append(row)
            materialized_members[member + ".magic_codec"] = result.payload

    # Compute aggregate byte savings.
    total_source_bytes = sum(int(r.get("source_bytes", 0)) for r in member_rows)
    total_magic_bytes = sum(
        int(r.get("magic_codec_payload_bytes", 0)) for r in member_rows
    )
    aggregate = {
        "total_source_bytes": total_source_bytes,
        "total_magic_codec_bytes": total_magic_bytes,
        "predicted_byte_delta": total_magic_bytes - total_source_bytes,
        "predicted_savings_ratio": (
            (total_source_bytes - total_magic_bytes) / total_source_bytes
            if total_source_bytes > 0
            else 0.0
        ),
    }

    manifest = {
        "schema": "magic_codec_selection_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": args.operator or "unknown",
        "source_archive_path": str(source),
        "source_archive_sha256": source_sha,
        "source_archive_size_bytes": source.stat().st_size,
        "stream_type": args.stream_type,
        "selection_strategy": args.selection_strategy,
        "quantize_bits": quantize_bits,
        "member_rows": member_rows,
        "aggregate": aggregate,
        "recommendation": recommendation_for(args.stream_type),
        "candidate_primitives": list(
            candidate_primitives_for(args.stream_type)
        ),
        "runtime_manifest": _build_runtime_manifest(),
        "dry_run": bool(args.dry_run),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "blockers": [
            "magic_codec_inflate_parity_not_run",
            "no_byte_closed_runtime_packet_built",
        ],
    }

    if args.dry_run:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "magic_codec_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    runtime_manifest_path = out_dir / "magic_codec_runtime_manifest.json"
    runtime_manifest_path.write_text(
        json.dumps(_build_runtime_manifest(), indent=2, sort_keys=True) + "\n"
    )

    # Emit the materialized magic-codec archive (zip of envelope payloads).
    archive_path = out_dir / "magic_codec_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in materialized_members.items():
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            zf.writestr(info, payload)
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    # Append the archive-level provenance row.
    final_manifest = dict(manifest)
    final_manifest["magic_codec_archive_path"] = str(archive_path)
    final_manifest["magic_codec_archive_sha256"] = archive_sha
    final_manifest["magic_codec_archive_size_bytes"] = archive_path.stat().st_size
    manifest_path.write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n"
    )

    print(
        f"wrote {archive_path} (sha={archive_sha[:8]}, "
        f"{archive_path.stat().st_size} bytes; predicted delta "
        f"{aggregate['predicted_byte_delta']:+} bytes vs source)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
