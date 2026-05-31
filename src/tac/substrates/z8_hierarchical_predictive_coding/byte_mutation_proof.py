# SPDX-License-Identifier: MIT
"""Byte-mutation consumption proof for Z8HPC1 archive sections.

The current Z8 runtime reconstructs pixels from Mallat wavelet coefficients and
applies a deterministic frame-1 top-LL correction from the decoded Wyner-Ziv /
Mamba top state. Decoder, DreamerV3, and categorical-index sections remain
archive-custody surfaces until their own mutation proofs show pixel
consumption. This module keeps that classification executable and reusable by
exporters and thin tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Z8_BYTE_MUTATION_PROOF_SCHEMA = (
    "z8_hierarchical_predictive_coding_distinguishing_feature_byte_mutation_proof.v1"
)
Z8_BYTE_MUTATION_PROBED_SECTIONS: tuple[str, ...] = (
    "wavelet_blob",
    "decoder_blob",
    "indices_blob",
    "wyner_ziv_blob",
    "dreamer_state_blob",
)
Z8_STACK_CUSTODY_SECTION_TO_MEMBER: dict[str, str] = {
    "decoder_blob": "decoder_weight_custody",
    "indices_blob": "dreamer_v3_categorical_indices_custody",
    "dreamer_state_blob": "dreamer_v3_rssm_state_custody",
}


def _reconstruct_all_pairs_small(archive_bytes: bytes) -> Any:
    """Reconstruct every trained pair at archive eval resolution."""

    import numpy as np

    from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
        reconstruct_pair_rgb_from_pyramid,
    )
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        projected_pair_pyramids_from_archive_bytes,
    )

    binding, pair_pyramids, _stats = projected_pair_pyramids_from_archive_bytes(
        archive_bytes
    )
    outs: list[Any] = []
    for pyramid in pair_pyramids:
        r0, r1 = reconstruct_pair_rgb_from_pyramid(binding, pyramid)
        outs.append(np.concatenate([r0.ravel(), r1.ravel()]))
    return np.concatenate(outs) if outs else np.zeros((0,), dtype=np.float32)


def _section_map(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    from tac.substrates.z8_hierarchical_predictive_coding.archive import (
        parse_z8hpc1_archive_bytes,
    )

    return parse_z8hpc1_archive_bytes(archive_bytes)


def probe_section_consumption(
    archive_bytes: bytes,
    base_recon: Any,
    section: str,
    *,
    num_offsets: int = 7,
) -> dict[str, Any]:
    """Flip bytes across one section and report parse/pixel-consumption verdict."""

    import numpy as np

    sections = _section_map(archive_bytes)
    if section not in sections:
        return {"section": section, "present": False}
    start, length = sections[section]
    if length <= 0:
        return {"section": section, "present": True, "length": 0, "verdict": "EMPTY"}

    skip = min(4, max(0, length - 1))
    span = length - skip
    raw_offsets = (
        [start]
        if span <= 0
        else [start + skip + (i * span) // max(num_offsets, 1) for i in range(num_offsets)]
    )
    offsets = sorted({offset for offset in raw_offsets if start <= offset < start + length})

    max_delta = 0.0
    n_pixel_changed = 0
    n_parse_error = 0
    n_zero_delta = 0
    per_offset: list[dict[str, Any]] = []
    for offset in offsets:
        mutated = bytearray(archive_bytes)
        mutated[offset] ^= 0xFF
        try:
            recon = _reconstruct_all_pairs_small(bytes(mutated))
            if recon.shape != base_recon.shape:
                n_parse_error += 1
                per_offset.append(
                    {"offset_in_section": offset - start, "result": "shape_changed"}
                )
                continue
            delta = float(np.abs(recon - base_recon).max())
            if delta > 0.0:
                n_pixel_changed += 1
                max_delta = max(max_delta, delta)
            else:
                n_zero_delta += 1
            per_offset.append({"offset_in_section": offset - start, "max_abs_delta": delta})
        except Exception as exc:
            n_parse_error += 1
            per_offset.append(
                {
                    "offset_in_section": offset - start,
                    "result": "parse_error",
                    "err": repr(exc)[:80],
                }
            )

    if n_pixel_changed > 0:
        verdict = "PIXEL_CONSUMED"
    elif n_parse_error > 0:
        verdict = "PARSE_GUARD_ONLY"
    else:
        verdict = "NO_OP"
    return {
        "section": section,
        "present": True,
        "length": length,
        "offsets_probed": len(offsets),
        "n_pixel_changed": n_pixel_changed,
        "n_parse_error": n_parse_error,
        "n_zero_delta": n_zero_delta,
        "max_abs_pixel_delta": max_delta,
        "verdict": verdict,
        "per_offset": per_offset,
    }


def probe_z8_archive_distinguishing_feature(
    archive_path: Path,
    *,
    proof_out: Path | None = None,
) -> dict[str, Any]:
    """Run the byte-mutation consumption proof on a Z8HPC1 ``0.bin`` archive."""

    archive_bytes = Path(archive_path).read_bytes()
    base_recon = _reconstruct_all_pairs_small(archive_bytes)
    sections = {
        section: probe_section_consumption(archive_bytes, base_recon, section)
        for section in Z8_BYTE_MUTATION_PROBED_SECTIONS
    }
    from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
        build_stack_context_payload_mutation_receiver_proofs,
        build_wyner_ziv_payload_mutation_receiver_proof,
    )

    semantic_wz_proof = build_wyner_ziv_payload_mutation_receiver_proof(archive_bytes)
    stack_context_proofs = build_stack_context_payload_mutation_receiver_proofs(
        archive_bytes
    )
    pixel_consumed_sections = [
        name for name, verdict in sections.items() if verdict.get("verdict") == "PIXEL_CONSUMED"
    ]
    if (
        semantic_wz_proof.get("wyner_ziv_top_state_pixel_consumption_proven")
        and "wyner_ziv_blob" not in pixel_consumed_sections
    ):
        pixel_consumed_sections.append("wyner_ziv_blob")
    for section in stack_context_proofs.get("stack_context_sections_pixel_consumed") or []:
        if section not in pixel_consumed_sections:
            pixel_consumed_sections.append(str(section))
    custody_only_sections = [
        name
        for name in Z8_STACK_CUSTODY_SECTION_TO_MEMBER
        if name not in pixel_consumed_sections
    ]
    distinguishing_feature_consumed = "wavelet_blob" in pixel_consumed_sections
    proof_path = str(proof_out) if proof_out is not None else None

    manifest = {
        "schema_version": Z8_BYTE_MUTATION_PROOF_SCHEMA,
        "archive_path": str(archive_path),
        "archive_bytes": len(archive_bytes),
        "base_recon_pixel_count": int(base_recon.size),
        "distinguishing_feature": "wavelet_blob",
        "distinguishing_feature_consumed": bool(distinguishing_feature_consumed),
        "pixel_consumed_sections": pixel_consumed_sections,
        "custody_only_sections": custody_only_sections,
        "mamba_dreamer_wyner_ziv_pixel_consumption_proven": bool(
            semantic_wz_proof.get("wyner_ziv_top_state_pixel_consumption_proven")
            and stack_context_proofs.get("stack_context_sections_pixel_consumed")
        ),
        "wyner_ziv_payload_pixel_consumption_proven": bool(
            semantic_wz_proof.get("wyner_ziv_top_state_pixel_consumption_proven")
        ),
        "stack_context_pixel_consumption_proven": bool(
            stack_context_proofs.get("stack_context_sections_pixel_consumed")
        ),
        "wyner_ziv_semantic_payload_mutation_receiver_proof": semantic_wz_proof,
        "stack_context_semantic_payload_mutation_receiver_proofs": (
            stack_context_proofs
        ),
        "sections": sections,
        "custody_section_to_member": dict(Z8_STACK_CUSTODY_SECTION_TO_MEMBER),
        "honest_architecture_note": (
            "Z8 contest inflate reconstructs pixels from wavelet_blob via Mallat "
            "perfect reconstruction and consumes wyner_ziv_blob by projecting "
            "the decoded top state into frame_1 top-LL. Decoder, indices, and "
            "Dreamer sections are consumed by the deterministic stack-context "
            "projection path; trained MLX renderer export remains a separate "
            "promotion blocker."
        ),
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "advisory",
        "score_claim": False,
        "promotable": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "proof_path": proof_path,
    }
    if proof_out is not None:
        out = Path(proof_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


__all__ = [
    "Z8_BYTE_MUTATION_PROBED_SECTIONS",
    "Z8_BYTE_MUTATION_PROOF_SCHEMA",
    "Z8_STACK_CUSTODY_SECTION_TO_MEMBER",
    "probe_section_consumption",
    "probe_z8_archive_distinguishing_feature",
]
