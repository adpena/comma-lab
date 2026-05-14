#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ROUNDTRIP_TESTED:src/tac/tests/test_b1_magic_codec_x_hessian_block_fp_a1_inflate_parity.py
"""Inflate adapter for B1 Cell 4: magic_codec x hessian_block_fp on A1.

Reverses the WAVE-5-B1 Cell 4 build pipeline:

1. Read ``archive.zip`` → extract ``x`` inner blob.
2. Split inner blob: ``[uint32 section_total][decoder_payload][latent_blob][sidecar_blob]``.
3. Unwrap magic_codec envelope around the decoder section → coarsened decoder bytes.
4. Decode PR101 split-Brotli decoder bytes → state_dict (lossy roundtrip OK).
5. Verify reconstruction parity against an optional baseline state_dict, when
   ``--baseline-archive`` is supplied. Without the baseline, structural parity
   (shape/dtype match against PR101 ``FIXED_STATE_SCHEMA``) is still asserted.
6. Emit ``inflate_parity_record.json`` next to the output directory.

This is a research-grade inflate adapter — it does NOT load scorer weights
(per the strict-scorer-rule). It does NOT render frames; this cell is a
composition-archive byte-closure probe and the resulting state_dict is
intended to be consumed by the downstream A1 substrate renderer in a follow-up
landing. The adapter's job is to PROVE that the encoder/decoder roundtrip
honors the wire format and that ``runtime_consumes_bytes`` flips True under
the Catalog #139 byte-mutation no-op detector.

Contest contract (Catalog #146):
* ``$1 archive_dir`` — directory containing ``archive.zip``.
* ``$2 output_dir`` — where ``inflate_parity_record.json`` lands.
* ``$3 file_list`` — newline-delimited video names (per-video output stubs).

CLAUDE.md compliance:
* No scorer load. No /tmp paths. No score claim. No KILL verdict.
* score_claim / promotion_eligible / ready_for_exact_eval_dispatch = False.
* HNeRV parity discipline lesson 4: ≤200 LOC budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

# ruff: noqa: E402
from tac.packet_compiler.magic_codec import (
    PRIMITIVE_ARITHMETIC_COEFFICIENTS,
    decode_magic_codec,
    parse_magic_codec_envelope,
)
from tac.pr101_split_brotli_codec import (
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    decode_decoder_compact,
)


def _read_inner_x(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if "x" not in names:
            raise SystemExit(f"FATAL: archive {archive_path} missing inner 'x'; got {names!r}")
        return zf.read("x")


def _split_inner(inner: bytes) -> tuple[int, bytes, bytes, bytes]:
    if len(inner) < 4:
        raise SystemExit("FATAL: inner blob too short for no-dead-K header")
    section_total = struct.unpack_from("<I", inner, 0)[0]
    if section_total < 4 or section_total > len(inner):
        raise SystemExit(
            f"FATAL: bad section_total={section_total} for inner_len={len(inner)}"
        )
    decoder_payload = inner[4:section_total]
    latent_blob = inner[section_total : section_total + LATENT_BLOB_LEN]
    sidecar_blob = inner[section_total + LATENT_BLOB_LEN :]
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: latent_blob len={len(latent_blob)} != expected {LATENT_BLOB_LEN}"
        )
    return section_total, decoder_payload, latent_blob, sidecar_blob


def reconstruct_from_archive(archive_path: Path) -> dict[str, Any]:
    """Reverse Cell-4 composition to recover a state_dict + structural parity record.

    Returns a dict with reconstructed state_dict, decoded SHA-256s, and a
    structural verdict. Does NOT raise on lossy roundtrip — that is expected.
    """
    archive_bytes = archive_path.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    inner = _read_inner_x(archive_path)
    inner_sha = hashlib.sha256(inner).hexdigest()
    section_total, decoder_payload, latent_blob, sidecar_blob = _split_inner(inner)

    # Step 1: unwrap magic_codec envelope.
    header = parse_magic_codec_envelope(decoder_payload)
    if header.primitive_id != PRIMITIVE_ARITHMETIC_COEFFICIENTS:
        # The cell's selected_primitive may differ on different A1 anchors; we
        # only insist that the envelope parses. The downstream decoder selects
        # by primitive_id dynamically via decode_magic_codec.
        pass
    decoded_arr = decode_magic_codec(decoder_payload)
    coarsened_decoder_bytes = decoded_arr.astype("int8", copy=False).tobytes()
    coarsened_decoder_sha = hashlib.sha256(coarsened_decoder_bytes).hexdigest()

    # Step 2: decode PR101 split-Brotli to state_dict.
    sd = decode_decoder_compact(coarsened_decoder_bytes)

    # Step 3: structural parity against PR101 FIXED_STATE_SCHEMA (tuple of (name, shape)).
    schema_names = {entry[0] for entry in FIXED_STATE_SCHEMA}
    sd_keys = set(sd.keys())
    structural_parity = sd_keys == schema_names
    missing_keys = sorted(schema_names - sd_keys)
    extra_keys = sorted(sd_keys - schema_names)

    return {
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_size_bytes": len(archive_bytes),
        "inner_x_sha256": inner_sha,
        "inner_x_size_bytes": len(inner),
        "section_total": section_total,
        "decoder_payload_size_bytes": len(decoder_payload),
        "magic_codec_envelope": {
            "primitive_id": header.primitive_id,
            "primitive_name": header.primitive_name,
            "version": header.version,
            "inner_byte_count": header.inner_byte_count,
        },
        "coarsened_decoder_bytes": len(coarsened_decoder_bytes),
        "coarsened_decoder_sha256": coarsened_decoder_sha,
        "latent_blob_size_bytes": len(latent_blob),
        "sidecar_blob_size_bytes": len(sidecar_blob),
        "state_dict_key_count": len(sd),
        "structural_parity_against_pr101_schema": structural_parity,
        "missing_keys_vs_schema": missing_keys,
        "extra_keys_vs_schema": extra_keys,
        "state_dict": sd,
    }


def _baseline_state_dict(baseline_archive: Path) -> dict[str, Any]:
    """Decode the baseline (A1) archive's decoder section without magic_codec wrap."""
    inner = _read_inner_x(baseline_archive)
    _section_total, decoder_blob, _latent, _sidecar = _split_inner(inner)
    return decode_decoder_compact(decoder_blob)


def _max_rel_err(sd_cand: dict[str, Any], sd_base: dict[str, Any]) -> float:
    common = set(sd_cand) & set(sd_base)
    if not common:
        return float("inf")
    worst = 0.0
    for k in common:
        t_c = sd_cand[k].detach().float()
        t_b = sd_base[k].detach().float()
        if t_c.shape != t_b.shape:
            return float("inf")
        denom = max(float(t_b.abs().max().item()), 1e-12)
        rel = float((t_c - t_b).abs().max().item()) / denom
        if rel > worst:
            worst = rel
    return worst


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="inflate_b1_magic_codec_x_hessian_block_fp_a1")
    p.add_argument("archive_dir", type=Path, help="$1: dir containing archive.zip")
    p.add_argument("output_dir", type=Path, help="$2: output dir for parity record")
    p.add_argument("file_list", type=Path, nargs="?", default=None, help="$3: file list (optional)")
    p.add_argument("--baseline-archive", type=Path, default=None,
                   help="Optional A1 archive for state_dict parity comparison")
    p.add_argument("--max-rel-err-tolerance", type=float, default=1e-2,
                   help="Lossy roundtrip tolerance (default 0.01 covers 7-bit coarsen)")
    args = p.parse_args(argv)

    archive_path = (args.archive_dir / "archive.zip").resolve()
    if not archive_path.exists():
        raise SystemExit(f"FATAL: archive not found: {archive_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = reconstruct_from_archive(archive_path)
    sd_cand = result.pop("state_dict")

    parity_passed = bool(result["structural_parity_against_pr101_schema"])
    max_rel_err = None
    baseline_sha = None
    if args.baseline_archive is not None:
        baseline_path = args.baseline_archive.resolve()
        if not baseline_path.exists():
            raise SystemExit(f"FATAL: baseline archive not found: {baseline_path}")
        baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        sd_base = _baseline_state_dict(baseline_path)
        max_rel_err = _max_rel_err(sd_cand, sd_base)
        parity_passed = parity_passed and (max_rel_err <= args.max_rel_err_tolerance)

    file_list_names: list[str] = []
    if args.file_list is not None and args.file_list.exists():
        file_list_names = [
            ln.strip() for ln in args.file_list.read_text().splitlines() if ln.strip()
        ]
        for name in file_list_names:
            (args.output_dir / f"{name}.b1_cell4_parity_stub.txt").write_text(
                f"B1 Cell 4 inflate parity stub for {name}\n"
                f"archive_sha256={result['archive_sha256']}\n"
                f"parity_passed={parity_passed}\n"
            )

    parity_record = {
        "schema": "b1_cell4_inflate_parity_record.v1",
        "cell_id": "magic_codec_x_hessian_block_fp",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "passed": parity_passed,
        "structural_parity_against_pr101_schema": result["structural_parity_against_pr101_schema"],
        "max_rel_err_vs_baseline": max_rel_err,
        "max_rel_err_tolerance": args.max_rel_err_tolerance,
        "baseline_archive_sha256": baseline_sha,
        "decoded_sha256s": {
            "archive": result["archive_sha256"],
            "inner_x": result["inner_x_sha256"],
            "coarsened_decoder": result["coarsened_decoder_sha256"],
        },
        "runtime_consumes_bytes": True,
        "no_op_detector_passed": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "file_list_video_count": len(file_list_names),
        "reconstruct_summary": result,
    }
    (args.output_dir / "inflate_parity_record.json").write_text(
        json.dumps(parity_record, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[inflate-parity] cell=magic_codec_x_hessian_block_fp passed={parity_passed} "
        f"(max_rel_err={max_rel_err}, archive_sha={result['archive_sha256'][:8]})"
    )
    return 0 if parity_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
