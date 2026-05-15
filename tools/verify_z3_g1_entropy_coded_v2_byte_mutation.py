#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Byte-mutation smoke verifier for Z3-G1 entropy-coded v2 (Catalog #139).

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lesson 11
(no-op detector) + Catalog #139 + the design memo §6: a substrate that ships
distinguishing bytes MUST PROVE those bytes are CONSUMED by the inflate path
to produce different inflated outputs. This tool extincts the F1 phantom-class
that bit Z3-G1 v1 (empty hyperprior_weights_int8 + w_hat_int8 slots produced
identical inflated output to Z3 v2 baseline).

Procedure (per Catalog #272 distinguishing-feature integration contract):

  1. Build a baseline Z3G2 packet from canonical sigma + class_indices +
     residual + affine.
  2. For each distinguishing-feature byte path (sigma_table_blob,
     class_index_blob), build a MUTATED packet differing in exactly ONE byte
     of that blob.
  3. Run the inflate consumer on baseline + each mutated packet.
  4. Classify each probe:
     - semantic_output_mutation: the mutated packet decodes cleanly AND the
       inflated output hash plus at least one output tensor changes.
     - parser_bound_consumption: the mutated packet is rejected by the parser or
       entropy decoder. This proves a byte boundary is live, but it is
       lower-grade evidence and MUST NOT be overclaimed as semantic output
       mutation.
  5. Pass only when every distinguishing-feature byte path has semantic output
     mutation evidence.

Exit codes:
  0  byte mutations PROVED semantic output mutation for every blob
  1  semantic output mutation missing for one or more blobs
  2  setup error (bad inputs, missing deps)

Usage:
  PYTHONPATH=src:upstream:. .venv/bin/python tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import brotli
import torch

from tac.substrates.z3_g1_entropy_coded_v2 import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G2EntropyCodedScorerClassGatingHead,
    build_z3g2_payload_bytes,
    compute_class_prior_cdf,
    encode_z3g2_section,
    reconstruct_class_indices_and_sigma_table_from_z3g2_payload,
)
from tac.substrates.z3_g1_entropy_coded_v2.archive import (
    A1_DECODER_SECTION_TOTAL,
    Z3G2_CLASS_PRIOR_BLOB_LEN,
    Z3G2_HEADER_STRUCT,
)

PARSER_BOUND_EXCEPTIONS = (ValueError, brotli.error, OverflowError, IndexError)
OUTPUT_TENSOR_NAMES = ("latents", "sigma_table", "class_indices")


def _build_synthetic_a1_bytes() -> bytes:
    """Synthetic A1 bytes for byte-mutation testing.

    Real A1 bytes can be substituted by passing --a1-archive (see main()).
    """
    fake_decoder_blob = bytes(162164 * [42])
    fake_latent_blob = bytes(15387 * [99])
    fake_sidecar = b"sidecar_for_byte_mutation_smoke"
    return (
        struct.pack("<I", A1_DECODER_SECTION_TOTAL)
        + fake_decoder_blob
        + fake_latent_blob
        + fake_sidecar
    )


def _build_baseline_packet() -> tuple[bytes, bytes, dict[str, slice]]:
    """Build a baseline Z3G2 packet + return (a1_bytes, payload, blob_slices).

    blob_slices maps each distinguishing-feature blob name to the byte slice
    in the OUTER payload (so the caller can build a mutated copy with one
    byte flipped).
    """
    head = Z3G2EntropyCodedScorerClassGatingHead()
    sigma_int8, scale = head.quantize_sigma_table_int8()
    torch.manual_seed(123)
    class_indices_t = torch.randint(0, G1_NUM_SCORER_CLASSES, (A1_N_PAIRS,))
    class_indices_uint8 = bytes(class_indices_t.to(torch.uint8).tolist())
    class_prior_counts = compute_class_prior_cdf(class_indices_t)
    residual_int8 = bytes((A1_N_PAIRS * A1_LATENT_DIM) * [3])

    section = encode_z3g2_section(
        sigma_table_int8=sigma_int8,
        class_indices_uint8=class_indices_uint8,
        class_prior_counts=class_prior_counts,
        residual_int8=residual_int8,
        latent_offset=torch.zeros(A1_LATENT_DIM),
        latent_scale=torch.ones(A1_LATENT_DIM),
        int8_sigma_scale=scale,
        quant_step=1.0,
        min_sigma=1e-3,
        max_sigma=16.0,
    )

    a1 = _build_synthetic_a1_bytes()
    payload = build_z3g2_payload_bytes(a1_bytes=a1, z3g2_section=section)

    # Compute blob slices in OUTER payload coordinates.
    # Layout: [decoder (162168)] + [Z3G2 section (479ish)] + [sidecar].
    # Z3G2 section: header (27) + sigma_len (2) + sigma_blob (~) + class_prior (10) +
    # class_index_len (4) + class_index_blob (~) + residual_len (4) + residual_blob (~) +
    # affine (224).
    section_offset = A1_DECODER_SECTION_TOTAL
    pos = section_offset + Z3G2_HEADER_STRUCT.size
    (sigma_table_len,) = struct.unpack_from("<H", payload, pos)
    sigma_blob_start = pos + 2
    sigma_blob_end = sigma_blob_start + sigma_table_len
    class_prior_start = sigma_blob_end
    class_prior_end = class_prior_start + Z3G2_CLASS_PRIOR_BLOB_LEN
    (class_index_len,) = struct.unpack_from("<I", payload, class_prior_end)
    class_index_blob_start = class_prior_end + 4
    class_index_blob_end = class_index_blob_start + class_index_len

    blob_slices = {
        "sigma_table_blob": slice(sigma_blob_start, sigma_blob_end),
        "class_prior_cdf_blob": slice(class_prior_start, class_prior_end),
        "class_index_blob": slice(class_index_blob_start, class_index_blob_end),
    }
    return a1, payload, blob_slices


def _mutate_byte_in_blob(
    payload: bytes, blob_slice: slice, *, byte_offset_in_blob: int = 0
) -> bytes:
    """Return a copy of payload with one byte flipped inside the named blob."""
    if blob_slice.stop <= blob_slice.start:
        raise ValueError(f"blob slice empty: {blob_slice}")
    if byte_offset_in_blob >= (blob_slice.stop - blob_slice.start):
        raise ValueError(
            f"byte_offset_in_blob {byte_offset_in_blob} >= blob len "
            f"{blob_slice.stop - blob_slice.start}"
        )
    target_index = blob_slice.start + byte_offset_in_blob
    mutated = bytearray(payload)
    # XOR with 0xA5 to ensure a real change (vs e.g. flipping 0x00 -> 0x00).
    mutated[target_index] ^= 0xA5
    return bytes(mutated)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    h = hashlib.sha256()
    tensor_cpu = tensor.detach().cpu().contiguous()
    h.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
    h.update(str(tensor_cpu.dtype).encode("utf-8"))
    h.update(tensor_cpu.numpy().tobytes())
    return h.hexdigest()


def _outputs_sha256(
    outputs: tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor],
) -> str:
    h = hashlib.sha256()
    a1_shell, latents, sigma_table, class_indices = outputs
    h.update(b"a1_shell")
    h.update(len(a1_shell).to_bytes(8, "little"))
    h.update(a1_shell)
    for name, tensor in (
        ("latents", latents),
        ("sigma_table", sigma_table),
        ("class_indices", class_indices),
    ):
        tensor_cpu = tensor.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
        h.update(str(tensor_cpu.dtype).encode("utf-8"))
        h.update(tensor_cpu.numpy().tobytes())
    return h.hexdigest()


def _output_shapes(
    outputs: tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    _, latents, sigma_table, class_indices = outputs
    return {
        "latents": list(latents.shape),
        "sigma_table": list(sigma_table.shape),
        "class_indices": list(class_indices.shape),
    }


def _compare_outputs(
    baseline_outputs: tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor],
    mutated_outputs: tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Return structured evidence comparing cleanly decoded inflate outputs."""
    _, baseline_latents, baseline_sigma, baseline_class_idx = baseline_outputs
    _, mutated_latents, mutated_sigma, mutated_class_idx = mutated_outputs
    baseline_hash = _outputs_sha256(baseline_outputs)
    mutated_hash = _outputs_sha256(mutated_outputs)
    tensor_pairs = (
        ("sigma_table", baseline_sigma, mutated_sigma),
        ("class_indices", baseline_class_idx, mutated_class_idx),
        ("latents", baseline_latents, mutated_latents),
    )
    tensor_deltas: list[dict[str, Any]] = []
    changed_names: list[str] = []
    for name, baseline_tensor, mutated_tensor in tensor_pairs:
        changed = not torch.equal(baseline_tensor, mutated_tensor)
        delta: dict[str, Any] = {
            "name": name,
            "changed": changed,
            "baseline_sha256": _tensor_sha256(baseline_tensor),
            "mutated_sha256": _tensor_sha256(mutated_tensor),
        }
        if changed:
            changed_names.append(name)
            if baseline_tensor.is_floating_point() or mutated_tensor.is_floating_point():
                delta["max_abs_diff"] = float(
                    (baseline_tensor - mutated_tensor).abs().max().item()
                )
            else:
                delta["n_different"] = int(
                    (baseline_tensor != mutated_tensor).sum().item()
                )
        tensor_deltas.append(delta)

    semantic_output_mutation = bool(changed_names) and baseline_hash != mutated_hash
    if semantic_output_mutation:
        reason = "semantic output tensors changed: " + ", ".join(changed_names)
    elif baseline_hash != mutated_hash:
        reason = "output hash changed without tensor delta; not semantic proof"
    else:
        reason = "all inflated output tensors identical"
    return {
        "semantic_output_mutation": semantic_output_mutation,
        "output_sha256_changed": baseline_hash != mutated_hash,
        "baseline_output_sha256": baseline_hash,
        "mutated_output_sha256": mutated_hash,
        "tensor_deltas": tensor_deltas,
        "reason": reason,
    }


def _mutation_replacement_candidates(original: int) -> tuple[int, ...]:
    """Candidate replacement bytes for a single serialized-byte mutation."""
    candidates = (original ^ 0xA5, original ^ 0x01, (original + 1) & 0xFF, 0, 255)
    return tuple(dict.fromkeys(value for value in candidates if value != original))


def _probe_offsets(blob_len: int) -> tuple[int, ...]:
    if blob_len <= 0:
        return ()
    return tuple(dict.fromkeys((0, blob_len // 2, blob_len - 1)))


def _probe_blob_mutations(
    *,
    blob_name: str,
    payload: bytes,
    blob_slice: slice,
    baseline_outputs: tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    """Probe one distinguishing-feature blob and return artifact-safe evidence."""
    blob_len = blob_slice.stop - blob_slice.start
    result: dict[str, Any] = {
        "blob_name": blob_name,
        "blob_start": blob_slice.start,
        "blob_stop": blob_slice.stop,
        "blob_len": blob_len,
        "semantic_output_mutation": False,
        "parser_bound_consumption": False,
        "semantic_proof": None,
        "parser_bound_attempts": [],
        "no_observed_effect_attempts": [],
        "attempts": [],
    }
    if blob_len <= 0:
        result["no_observed_effect_attempts"].append(
            {"status": "empty_blob", "reason": "blob slice is empty"}
        )
        return result

    for offset in _probe_offsets(blob_len):
        original_byte = payload[blob_slice.start + offset]
        for replacement_byte in _mutation_replacement_candidates(original_byte):
            attempt: dict[str, Any] = {
                "blob_name": blob_name,
                "byte_offset_in_blob": offset,
                "payload_byte_offset": blob_slice.start + offset,
                "original_byte": int(original_byte),
                "replacement_byte": int(replacement_byte),
                "clean_decode": False,
                "semantic_output_mutation": False,
                "parser_bound_consumption": False,
                "output_sha256_changed": False,
                "status": "not_run",
            }
            try:
                mutated = bytearray(payload)
                mutated[blob_slice.start + offset] = replacement_byte
                mutated_outputs = reconstruct_class_indices_and_sigma_table_from_z3g2_payload(
                    bytes(mutated)
                )
                comparison = _compare_outputs(baseline_outputs, mutated_outputs)
                attempt.update(comparison)
                attempt["clean_decode"] = True
                if comparison["semantic_output_mutation"]:
                    attempt["status"] = "semantic_output_mutation"
                    result["semantic_output_mutation"] = True
                    if result["semantic_proof"] is None:
                        result["semantic_proof"] = attempt
                else:
                    attempt["status"] = "no_observed_effect"
                    result["no_observed_effect_attempts"].append(attempt)
            except PARSER_BOUND_EXCEPTIONS as exc:
                attempt.update(
                    {
                        "status": "parser_bound_consumption",
                        "parser_bound_consumption": True,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "reason": (
                            "mutated bytes rejected by parser/entropy decoder; "
                            "lower-grade evidence only"
                        ),
                    }
                )
                result["parser_bound_consumption"] = True
                result["parser_bound_attempts"].append(attempt)
            result["attempts"].append(attempt)

    if result["semantic_output_mutation"]:
        result["evidence_grade"] = "semantic_output_mutation"
    elif result["parser_bound_consumption"]:
        result["evidence_grade"] = "parser_bound_consumption_only"
    else:
        result["evidence_grade"] = "no_observed_effect"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Byte-mutation smoke verifier for Z3-G1 entropy-coded v2 (Catalog #139)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-blob mutation results."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for structured byte-mutation evidence JSON.",
    )
    args = parser.parse_args(argv)

    try:
        _, payload, blob_slices = _build_baseline_packet()
    except Exception as exc:
        print(f"[FAIL] baseline packet build error: {exc}", file=sys.stderr)
        return 2

    try:
        baseline_outputs = reconstruct_class_indices_and_sigma_table_from_z3g2_payload(
            payload
        )
    except Exception as exc:
        print(f"[FAIL] baseline inflate error: {exc}", file=sys.stderr)
        return 2

    baseline_hash = _outputs_sha256(baseline_outputs)
    print(
        f"[OK] baseline packet built ({len(payload)} B); inflate outputs OK "
        f"(sigma_table {tuple(baseline_outputs[2].shape)}, "
        f"class_indices {tuple(baseline_outputs[3].shape)}, "
        f"latents {tuple(baseline_outputs[1].shape)}); "
        f"output_sha256={baseline_hash}"
    )

    artifact: dict[str, Any] = {
        "schema_version": "z3_g1_entropy_coded_v2_byte_mutation_v2",
        "verifier": "tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py",
        "payload_bytes": len(payload),
        "baseline": {
            "clean_decode": True,
            "output_sha256": baseline_hash,
            "output_shapes": _output_shapes(baseline_outputs),
        },
        "blob_results": [],
    }

    for blob_name, blob_slice in blob_slices.items():
        result = _probe_blob_mutations(
            blob_name=blob_name,
            payload=payload,
            blob_slice=blob_slice,
            baseline_outputs=baseline_outputs,
        )
        artifact["blob_results"].append(result)
        if args.verbose:
            for attempt in result["attempts"]:
                status = attempt["status"]
                offset = attempt["byte_offset_in_blob"]
                replacement = attempt["replacement_byte"]
                reason = attempt.get("reason", "")
                print(
                    f"  [{blob_name}] byte {offset} -> {replacement}: "
                    f"{status} ({reason})"
                )
        if result["semantic_output_mutation"]:
            proof = result["semantic_proof"] or {}
            print(
                f"[PASS] {blob_name} semantic_output_mutation — "
                f"{proof.get('reason', 'clean decode changed output')}; "
                f"mutated_output_sha256={proof.get('mutated_output_sha256')}"
            )
            if result["parser_bound_consumption"]:
                print(
                    f"[INFO] {blob_name} also has "
                    f"{len(result['parser_bound_attempts'])} parser_bound_consumption "
                    "attempt(s); recorded as lower-grade evidence only."
                )
        elif result["parser_bound_consumption"]:
            print(
                f"[WARN] {blob_name} parser_bound_consumption_only — decoder "
                "rejected mutated bytes, but no clean semantic output mutation "
                "was observed."
            )
        else:
            print(
                f"[FAIL] {blob_name} no_observed_effect — clean mutations did not "
                "change inflated output and no parser-bound rejection was observed."
            )

    semantic_failures = [
        item["blob_name"]
        for item in artifact["blob_results"]
        if not item["semantic_output_mutation"]
    ]
    parser_bound_only = [
        item["blob_name"]
        for item in artifact["blob_results"]
        if item["evidence_grade"] == "parser_bound_consumption_only"
    ]
    artifact["semantic_output_mutation_all_blobs"] = not semantic_failures
    artifact["parser_bound_consumption_blobs"] = [
        item["blob_name"]
        for item in artifact["blob_results"]
        if item["parser_bound_consumption"]
    ]
    artifact["parser_bound_only_blobs"] = parser_bound_only
    artifact["verdict"] = "pass" if not semantic_failures else "fail"

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(f"[ARTIFACT] wrote structured evidence JSON: {args.output_json}")

    if semantic_failures:
        print(
            f"\n[VERDICT] FAIL: semantic_output_mutation missing for "
            f"{len(semantic_failures)}/{len(blob_slices)} distinguishing-feature "
            f"blob(s): {semantic_failures}",
            file=sys.stderr,
        )
        if parser_bound_only:
            print(
                f"[VERDICT] lower-grade parser_bound_consumption_only evidence "
                f"exists for: {parser_bound_only}",
                file=sys.stderr,
            )
        print(
            "Per Catalog #139 + design memo §6: semantic proof requires clean "
            "decode plus output hash/tensor delta. Parser rejection alone is "
            "not semantic output mutation.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\n[VERDICT] PASS: all {len(blob_slices)} distinguishing-feature blobs "
        f"have semantic_output_mutation evidence (clean decode + output "
        f"hash/tensor delta). Parser-bound rejections, if present, are "
        f"recorded separately as lower-grade evidence."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
