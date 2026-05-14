"""Runtime adapter for HDM3 HNeRV decoder-section candidates.

HDM3 stores the fixed-schema HNeRV decoder q stream as Brotli(q_bytes) plus raw
scale bytes. Public PR106 runtimes expect the decoder section itself to be a
legacy Brotli stream over ``q_bytes + scales``. This module performs the
deterministic bridge:

``HDM3 decoder section -> raw decoder bytes -> legacy Brotli decoder section``.

It is intentionally score-agnostic. The adapter proves byte and raw-decoder
custody for runtime parity, but exact CUDA auth eval remains the score truth.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import brotli

from tac.hnerv_decoder_recode import (
    HnervDecoderRecodeError,
    decode_hdm3_q_brotli_split_fixture,
    decode_hdm4_q_brotli_split_fixture,
    decode_hdm6_q_brotli_tuned_fixture,
    decode_hdm7_q_brotli_len_elided_fixture,
)
from tac.hnerv_lowlevel_packer import (
    HnervLowlevelPackError,
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    sha256_bytes,
)
from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_SIDECAR_MAGIC,
    emit_pr106_sidecar_packet,
    parse_pr106_sidecar_packet,
)
from tac.repo_io import json_text, write_json

SCHEMA_VERSION = 1
LEGACY_BROTLI_QUALITY = 10


class HnervHdm3RuntimeAdapterError(ValueError):
    """Raised when an HDM3 runtime-adapter input is malformed."""


def restore_hdm3_payload_to_legacy_brotli(
    payload: bytes,
    *,
    brotli_quality: int = LEGACY_BROTLI_QUALITY,
    require_hdm3: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Return a PR106-compatible packed payload plus a custody proof.

    Legacy Brotli decoder sections pass through unchanged unless
    ``require_hdm3`` is true. HDM3 sections are decoded to raw fixed-schema
    decoder bytes and rewrapped as a legacy Brotli decoder section. Unknown
    decoder sections fail closed instead of falling through to public inflate.
    """
    if payload and payload[0] == PR106_SIDECAR_MAGIC:
        packet = parse_pr106_sidecar_packet(payload)
        restored_inner, inner_proof = restore_hdm3_payload_to_legacy_brotli(
            packet.pr106_bytes,
            brotli_quality=brotli_quality,
            require_hdm3=require_hdm3,
        )
        restored_packet = emit_pr106_sidecar_packet(
            replace(packet, pr106_bytes=restored_inner)
        )
        proof = {
            **inner_proof,
            "mode": f"pr106_sidecar_wrapper_{inner_proof['mode']}",
            "proof_scope": "pr106_sidecar_wrapper_inner_payload_normalization",
            "input_payload_sha256": sha256_bytes(payload),
            "input_payload_bytes": len(payload),
            "output_payload_sha256": sha256_bytes(restored_packet),
            "output_payload_bytes": len(restored_packet),
            "payload_changed": restored_packet != payload,
            "wrapper_format_id": packet.format_id,
            "wrapper_sidecar_kind": packet.sidecar_kind,
            "wrapper_sidecar_payload_sha256": sha256_bytes(packet.sidecar_payload),
            "wrapper_sidecar_payload_bytes": len(packet.sidecar_payload),
            "wrapper_framing_meta_sha256": (
                sha256_bytes(packet.framing_meta) if packet.framing_meta is not None else None
            ),
            "wrapper_framing_meta_bytes": (
                len(packet.framing_meta) if packet.framing_meta is not None else 0
            ),
            "wrapper_sidecar_preserved": parse_pr106_sidecar_packet(
                restored_packet
            ).sidecar_payload == packet.sidecar_payload,
            "inner_payload_proof": inner_proof,
        }
        proof["ready_for_public_runtime_inflate"] = bool(
            inner_proof.get("ready_for_public_runtime_inflate")
            and proof["wrapper_sidecar_preserved"]
        )
        return restored_packet, proof

    packed = parse_ff_packed_brotli_hnerv(payload)
    decoder = packed.decoder_packed_brotli
    input_decoder_sha = sha256_bytes(decoder)
    input_latents_sha = sha256_bytes(packed.latents_and_sidecar_brotli)

    if decoder.startswith((b"HDM3", b"HDM4", b"HDM6", b"HDM7")):
        recode_magic = decoder[:4].decode("ascii", errors="replace")
        raw_decoder = _decode_hdm_raw(decoder)
        restored_decoder = brotli.compress(raw_decoder, quality=brotli_quality)
        try:
            restored_raw = brotli.decompress(restored_decoder)
        except brotli.error as exc:  # pragma: no cover - brotli self-check guard
            raise HnervHdm3RuntimeAdapterError("restored legacy Brotli is not decompressible") from exc
        if restored_raw != raw_decoder:
            raise HnervHdm3RuntimeAdapterError("restored legacy Brotli raw decoder mismatch")
        restored_payload = PackedHnervPayload(
            header=packed.header,
            decoder_packed_brotli=restored_decoder,
            latents_and_sidecar_brotli=packed.latents_and_sidecar_brotli,
        ).to_bytes()
        proof = _proof(
            mode=f"{recode_magic.lower()}_restored_to_legacy_brotli",
            input_payload=payload,
            output_payload=restored_payload,
            input_decoder_sha=input_decoder_sha,
            output_decoder_sha=sha256_bytes(restored_decoder),
            input_decoder_bytes=len(decoder),
            output_decoder_bytes=len(restored_decoder),
            raw_decoder=raw_decoder,
            restored_raw_decoder=restored_raw,
            input_latents_sha=input_latents_sha,
            output_latents_sha=sha256_bytes(packed.latents_and_sidecar_brotli),
            brotli_quality=brotli_quality,
        )
        return restored_payload, proof

    if require_hdm3:
        raise HnervHdm3RuntimeAdapterError("payload decoder section is not HDM3/HDM4/HDM6/HDM7")

    try:
        raw_decoder = brotli.decompress(decoder)
    except brotli.error as exc:
        raise HnervHdm3RuntimeAdapterError(
            "decoder section is neither HDM3/HDM4/HDM6/HDM7 nor legacy Brotli"
        ) from exc
    proof = _proof(
        mode="legacy_brotli_passthrough",
        input_payload=payload,
        output_payload=payload,
        input_decoder_sha=input_decoder_sha,
        output_decoder_sha=input_decoder_sha,
        input_decoder_bytes=len(decoder),
        output_decoder_bytes=len(decoder),
        raw_decoder=raw_decoder,
        restored_raw_decoder=raw_decoder,
        input_latents_sha=input_latents_sha,
        output_latents_sha=input_latents_sha,
        brotli_quality=None,
    )
    return payload, proof


def restore_hdm3_file_to_legacy_brotli(
    *,
    input_path: str | Path,
    output_path: str | Path,
    json_out: str | Path | None = None,
    require_hdm3: bool = False,
    brotli_quality: int = LEGACY_BROTLI_QUALITY,
) -> dict[str, Any]:
    """Restore one extracted HNeRV payload file and optionally write proof JSON."""

    source = Path(input_path)
    target = Path(output_path)
    payload = source.read_bytes()
    restored, proof = restore_hdm3_payload_to_legacy_brotli(
        payload,
        brotli_quality=brotli_quality,
        require_hdm3=require_hdm3,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(restored)
    proof = {
        **proof,
        "input_path": source.as_posix(),
        "output_path": target.as_posix(),
    }
    if json_out is not None:
        write_json(json_out, proof)
    return proof


def _decode_hdm_raw(decoder: bytes) -> bytes:
    try:
        if decoder.startswith(b"HDM3"):
            return decode_hdm3_q_brotli_split_fixture(decoder).to_raw()
        if decoder.startswith(b"HDM4"):
            return decode_hdm4_q_brotli_split_fixture(decoder).to_raw()
        if decoder.startswith(b"HDM6"):
            return decode_hdm6_q_brotli_tuned_fixture(decoder).to_raw()
        if decoder.startswith(b"HDM7"):
            return decode_hdm7_q_brotli_len_elided_fixture(decoder).to_raw()
        raise HnervDecoderRecodeError("unsupported HDM decoder-section magic")
    except HnervDecoderRecodeError as exc:
        raise HnervHdm3RuntimeAdapterError(f"invalid HDM decoder section: {exc}") from exc


def _proof(
    *,
    mode: str,
    input_payload: bytes,
    output_payload: bytes,
    input_decoder_sha: str,
    output_decoder_sha: str,
    input_decoder_bytes: int,
    output_decoder_bytes: int,
    raw_decoder: bytes,
    restored_raw_decoder: bytes,
    input_latents_sha: str,
    output_latents_sha: str,
    brotli_quality: int | None,
) -> dict[str, Any]:
    raw_sha = sha256_bytes(raw_decoder)
    restored_raw_sha = sha256_bytes(restored_raw_decoder)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": "hnerv_hdm3_runtime_adapter_proof_v1",
        "proof_scope": "payload_identity_and_legacy_decoder_normalization_only",
        "mode": mode,
        "score_claim": False,
        "dispatch_attempted": False,
        "input_payload_sha256": sha256_bytes(input_payload),
        "input_payload_bytes": len(input_payload),
        "output_payload_sha256": sha256_bytes(output_payload),
        "output_payload_bytes": len(output_payload),
        "payload_changed": output_payload != input_payload,
        "input_decoder_section_sha256": input_decoder_sha,
        "input_decoder_section_bytes": input_decoder_bytes,
        "output_decoder_section_sha256": output_decoder_sha,
        "output_decoder_section_bytes": output_decoder_bytes,
        "decoder_section_byte_delta_runtime_only": output_decoder_bytes - input_decoder_bytes,
        "decoder_raw_sha256": raw_sha,
        "restored_decoder_raw_sha256": restored_raw_sha,
        "decoder_raw_equal": raw_sha == restored_raw_sha,
        "latents_and_sidecar_input_sha256": input_latents_sha,
        "latents_and_sidecar_output_sha256": output_latents_sha,
        "latents_and_sidecar_preserved": input_latents_sha == output_latents_sha,
        "legacy_brotli_quality": brotli_quality,
        "ready_for_public_runtime_inflate": raw_sha == restored_raw_sha
        and input_latents_sha == output_latents_sha,
        "ready_for_exact_eval_dispatch": False,
        "exact_eval_packet_readiness_artifact_required": True,
        "runtime_tree_closure_contract_required": True,
        "strict_static_compliance_required": True,
        "lane_dispatch_claim_required_before_gpu": True,
        "exact_cuda_auth_eval_required_before_score": True,
        "remaining_dispatch_blockers": [
            "exact_inflate_output_parity_missing",
            "strict_pre_submission_compliance_json_missing",
            "lane_dispatch_claim_missing",
            "exact_cuda_auth_eval_missing",
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-hdm3", action="store_true")
    parser.add_argument(
        "--legacy-brotli-quality",
        type=int,
        default=LEGACY_BROTLI_QUALITY,
        choices=range(12),
        metavar="{0..11}",
        help="Brotli quality used when restoring HDM3 to a legacy decoder section.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        proof = restore_hdm3_file_to_legacy_brotli(
            input_path=args.input_path,
            output_path=args.output_path,
            json_out=args.json_out,
            require_hdm3=args.require_hdm3,
            brotli_quality=args.legacy_brotli_quality,
        )
    except (HnervHdm3RuntimeAdapterError, HnervLowlevelPackError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    if args.json_out is None:
        print(json_text(proof), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HnervHdm3RuntimeAdapterError",
    "restore_hdm3_file_to_legacy_brotli",
    "restore_hdm3_payload_to_legacy_brotli",
]
