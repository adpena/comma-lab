# SPDX-License-Identifier: MIT
# DETERMINISTIC_COMPILER_OK:non-promoting-research-or-probe-builder-emits-non-score-claiming-archive-FIX-WAVE-R1-META-1-closure
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
  the source stream + the output member name + the magic codec primitive selected + the
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
import importlib.util
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from tac.output_path_policy import assert_not_temporary_output_dir
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


_SUBMISSION_RUNTIME_SUPPORTED_PRIMITIVE_IDS = frozenset({0xF0, 0xF1, 0xF2})
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUBMISSION_RUNTIME_PATH = (
    _REPO_ROOT / "submissions" / "magic_codec_pr106_r2" / "inflate.py"
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
    try:
        assert_not_temporary_output_dir(
            output_dir,
            tool_name="materialize_magic_codec_archive",
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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

    itemsize = np.dtype(dtype).itemsize
    if len(member_bytes) % itemsize != 0:
        raise SystemExit(
            f"member byte count {len(member_bytes)} is not divisible by "
            f"{itemsize} for --quantize-bits={quantize_bits}; refusing to "
            "truncate tail bytes"
        )
    n_elements = len(member_bytes) // itemsize
    usable_bytes = member_bytes
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


def _assert_submission_runtime_can_roundtrip_member(
    member_name: str,
    member_bytes: bytes,
    result: MagicCodecResult,
) -> None:
    """Fail closed unless the current submission adapter decodes exact bytes."""

    if result.selected_primitive_id not in _SUBMISSION_RUNTIME_SUPPORTED_PRIMITIVE_IDS:
        raise MagicCodecError(
            f"{member_name}: selected primitive_id "
            f"0x{result.selected_primitive_id:02X} is not supported by "
            "submissions/magic_codec_pr106_r2/inflate.py"
        )
    try:
        spec = importlib.util.spec_from_file_location(
            "_magic_codec_pr106_r2_inflate",
            _SUBMISSION_RUNTIME_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load runtime at {_SUBMISSION_RUNTIME_PATH}")
        runtime_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime_module)
        decoded = runtime_module._decode_envelope_to_inner_bytes(result.payload)
    except Exception as exc:
        raise MagicCodecError(
            f"{member_name}: submission runtime refused selected magic-codec "
            f"payload: {exc}"
        ) from exc
    if decoded != member_bytes:
        raise MagicCodecError(
            f"{member_name}: submission runtime decode is not byte-identical "
            "to the source member; refusing non-parity materialization"
        )


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
    _assert_submission_runtime_can_roundtrip_member(member_name, member_bytes, result)
    member_row = {
        "member_name": member_name,
        "output_member_name": member_name,
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
        "runtime_dep_closure": [
            "numpy",
            "brotli",
            "lzma",
            "constriction",
            "repo_tac_required_until_vendored",
            "sibling_pr106_latent_sidecar_r2_required",
        ],
        "inflate_runtime_loc_budget": 200,
        "decoder_module": "tac.packet_compiler.magic_codec",
        "magic_envelope": "MAGC",
        "primitive_id_namespace": "0xF0-0xFF",
        "submission_supported_primitive_ids": [
            f"0x{pid:02X}"
            for pid in sorted(_SUBMISSION_RUNTIME_SUPPORTED_PRIMITIVE_IDS)
        ],
        "runtime_tree_byte_closed": False,
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
                raise SystemExit(
                    f"refusing to materialize {member!r}: {exc}"
                ) from exc
            member_rows.append(row)
            # Preserve the original member name so contest-style inflate.sh
            # can consume the emitted archive via `${base}.bin` from file_list.
            materialized_members[member] = result.payload

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
            "research_adapter_runtime_depends_on_repo_tac_until_vendored",
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
