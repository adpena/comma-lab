# SPDX-License-Identifier: MIT
"""Runtime-consumption proof for PR106 HLM1 fixed-latent recodes.

This is deliberately narrower than full-frame parity or scoring. It imports the
candidate runtime's own ``src/codec.py`` and proves that the HLM1 fixed-latent
section is decoded by that runtime, that the decoded raw bytes match the
canonical PacketIR decoder, and that a valid HLM1 metadata mutation changes
runtime-visible decoded bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tac.hnerv_lowlevel_packer import read_packed_archive_view
from tac.packet_compiler.pr106_fixed_latent_recode import (
    HLM1_MAGIC,
    HLM2_MAGIC,
    PR106_FIXED_LATENT_META_BYTES,
    decode_pr106_fixed_latent_raw,
)
from tac.packet_compiler.pr106_runtime_consumption import load_pr106_runtime_codec
from tac.repo_io import json_text, repo_relative, sha256_bytes, sha256_file


def prove_pr106_hlm1_runtime_consumption(
    *,
    archive_path: str | Path,
    runtime_dir: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a fail-closed proof that runtime codec consumes HLM1 latents."""

    manifest = prove_pr106_hlm_runtime_consumption(
        archive_path=archive_path,
        runtime_dir=runtime_dir,
        repo_root=repo_root,
        allowed_codecs=("hlm1",),
    )
    manifest["schema"] = "pr106_hlm1_runtime_consumption_proof_v1"
    manifest["proof_scope"] = "runtime_codec_hlm1_fixed_latent_decode_not_full_frame"
    manifest["runtime_hlm1_decode_matches_canonical"] = manifest[
        "runtime_hlm_decode_matches_canonical"
    ]
    manifest["runtime_hlm1_valid_mutation"] = manifest["runtime_hlm_valid_mutation"]
    manifest["runtime_hlm1_valid_mutation_changes_raw"] = manifest[
        "runtime_hlm_valid_mutation_changes_raw"
    ]
    manifest["runtime_hlm1_decode_consumption_claim"] = manifest[
        "runtime_hlm_decode_consumption_claim"
    ]
    return manifest


def prove_pr106_hlm_runtime_consumption(
    *,
    archive_path: str | Path,
    runtime_dir: str | Path,
    repo_root: str | Path | None = None,
    allowed_codecs: tuple[str, ...] = ("hlm1", "hlm2"),
) -> dict[str, Any]:
    """Return a fail-closed proof that runtime codec consumes HLM latents."""

    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    archive = Path(archive_path)
    runtime = Path(runtime_dir)
    view = read_packed_archive_view(archive)
    latent_payload = view.packed.latents_and_sidecar_brotli
    blockers: list[str] = []
    latent_codec = _latent_codec(latent_payload)
    if latent_codec == "legacy_brotli":
        blockers.append("latent_section_not_hlm")
    if latent_codec not in set(allowed_codecs):
        blockers.append(f"latent_section_codec_not_allowed:{latent_codec}")

    codec = load_pr106_runtime_codec(runtime)
    runtime_raw = codec.decode_fixed_latents_raw(latent_payload)
    canonical_raw = decode_pr106_fixed_latent_raw(latent_payload)
    runtime_raw_sha = sha256_bytes(runtime_raw)
    canonical_raw_sha = sha256_bytes(canonical_raw)
    if runtime_raw != canonical_raw:
        blockers.append("runtime_hlm1_decode_differs_from_canonical_decoder")

    mutation = _mutate_hlm_meta_byte(latent_payload)
    mutated_runtime_raw = codec.decode_fixed_latents_raw(mutation["payload"])
    mutated_runtime_raw_sha = sha256_bytes(mutated_runtime_raw)
    if mutated_runtime_raw_sha == runtime_raw_sha:
        blockers.append("runtime_hlm1_valid_mutation_did_not_change_decoded_raw")

    return {
        "schema": "pr106_hlm_runtime_consumption_proof_v1",
        "proof_scope": "runtime_codec_hlm_fixed_latent_decode_not_full_frame",
        "archive_path": repo_relative(archive, repo),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "runtime_dir": repo_relative(runtime, repo),
        "runtime_codec_sha256": sha256_file(runtime / "src" / "codec.py"),
        "member_name": view.archive.member_name,
        "latent_section_codec": latent_codec,
        "latent_section_bytes": len(latent_payload),
        "latent_section_sha256": sha256_bytes(latent_payload),
        "runtime_decoded_raw_sha256": runtime_raw_sha,
        "canonical_decoded_raw_sha256": canonical_raw_sha,
        "runtime_hlm_decode_matches_canonical": runtime_raw == canonical_raw,
        "runtime_hlm_valid_mutation": mutation["manifest"],
        "mutated_runtime_decoded_raw_sha256": mutated_runtime_raw_sha,
        "runtime_hlm_valid_mutation_changes_raw": mutated_runtime_raw_sha != runtime_raw_sha,
        "runtime_hlm_decode_consumption_claim": not blockers,
        "full_frame_inflate_output_parity_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        "required_next_proof": (
            "full-frame source-vs-candidate inflate output parity or exact same-runtime "
            "auth eval with archive/runtime custody"
        ),
    }


def dumps_hlm1_runtime_consumption_manifest(manifest: dict[str, Any]) -> str:
    """Return canonical JSON for an HLM1 runtime-consumption manifest."""

    return json_text(manifest)


def _latent_codec(payload: bytes) -> str:
    if payload.startswith(HLM1_MAGIC):
        return "hlm1"
    if payload.startswith(HLM2_MAGIC):
        return "hlm2"
    return "legacy_brotli"


def _mutate_hlm_meta_byte(payload: bytes) -> dict[str, Any]:
    if not payload.startswith(HLM1_MAGIC):
        if not payload.startswith(HLM2_MAGIC):
            raise ValueError("cannot mutate non-HLM fixed-latent payload")
    codec = _latent_codec(payload)
    lo_len = int.from_bytes(payload[4:6], "little")
    if codec == "hlm1":
        hi_delta_len = int.from_bytes(payload[6:8], "little")
        hi_count: int | None = int.from_bytes(payload[8:10], "little")
        meta_offset = 10 + lo_len
    else:
        hi_delta_len = len(payload) - (6 + lo_len + PR106_FIXED_LATENT_META_BYTES)
        hi_count = None
        meta_offset = 6 + lo_len
    meta_end = meta_offset + PR106_FIXED_LATENT_META_BYTES
    if meta_end > len(payload):
        raise ValueError("HLM payload truncated before metadata section")
    mutated = bytearray(payload)
    old = mutated[meta_offset]
    mutated[meta_offset] = old ^ 0x01
    return {
        "payload": bytes(mutated),
        "manifest": {
            "mutation_kind": f"{codec}_meta_byte_xor_0x01",
            "offset": meta_offset,
            "old_byte": old,
            "new_byte": mutated[meta_offset],
            "lo_brotli_len": lo_len,
            "hi_delta_len": hi_delta_len,
            "hi_count": hi_count,
            "metadata_bytes": PR106_FIXED_LATENT_META_BYTES,
        },
    }
