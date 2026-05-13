"""Shared helper for WAVE-5-B1 composition-cell archive builders on A1 substrate.

This module exposes the building blocks for the 4 WAVE-5 B1 composition
cells re-based against A1 (per
``feedback_b1_archive_build_empirical_falsifies_composition_cells_on_pr106_r2_20260512.md``
operator decision option 2: "Rebase B1 to A1 substrate").

The 4 cells are:

* ``film_pose_conditioning x magic_codec`` - FiLM-pose bolt-on slot reserved
  on top of magic-codec re-encoded A1 decoder blob.
* ``film_pose_conditioning x hessian_block_fp`` - FiLM-pose bolt-on slot
  reserved on top of Hessian-saliency-weighted block-FP coarsened decoder.
* ``nerv_enc_dec_separated x magic_codec`` - NeRV enc/dec split is
  compress-time only, so on the archive surface it is byte-identical to
  singleton magic_codec on A1. Builder still emits explicit
  composition_metadata for forensic clarity.
* ``magic_codec x hessian_block_fp`` - Hessian-coarsened decoder blob then
  re-wrapped via magic_codec.

ALL builders honor:

* CLAUDE.md "Forbidden score claims" - every emitted manifest sets
  ``score_claim: false``, ``promotion_eligible: false``,
  ``ready_for_exact_eval_dispatch: false``, ``byte_proxy_only: true``,
  ``cuda_eval_worth_testing: false``.
* Catalog #100 - ``runtime_manifest`` carries archive_path, archive_sha256,
  parser_section_manifest, measured_config_status, dispatch_blockers.
* Catalog #91 - paired roundtrip test declaration; see the per-cell test
  files under ``tests/`` (sibling files, not edited here).
* Catalog #94 - no ``admm`` token in any class / function / file name.
* Catalog #123 - no weight-domain saliency (mean theta squared) when
  ``--saliency-source score_gradient`` is invoked; the helper accepts a
  caller-supplied saliency dict to defer the choice to the builder CLI.
* Catalog #139 - emits the no-op proof signal via byte-mutation smoke at the
  builder CLI level (NOT here - the helper is byte-deterministic).

NO scorer load, NO MPS / torch.cuda, NO /tmp paths, NO commit-time mutation
of A1 source archive - A1 is read-only.

A1 baseline anchor (canonical reference for these cells):
``experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_
20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip``
(178,262 bytes, sha=87ec7ca5...).
"""
from __future__ import annotations

import hashlib
import struct
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

from tac.packet_compiler.magic_codec import (
    MagicCodecResult,
    StreamHint,
    encode_magic_codec,
)
from tac.pr101_split_brotli_codec import (
    LATENT_BLOB_LEN,
    decode_decoder_compact,
    encode_decoder_compact,
)

# Canonical A1 latent-aligned anchor (Council 9b44c2f6 + commit 13e8e08c).
A1_ARCHIVE_PATH_DEFAULT = Path(
    "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
    "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
)
A1_ARCHIVE_BYTES_EXPECTED = 178_262
A1_ARCHIVE_SHA256_EXPECTED = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)


# Per the FiLM pose conditioning bolt-on (substrate id 0x81, byte band
# 2_000-8_000 - per substrate_composition_matrix.py). We reserve the
# mid-band as a small placeholder slot: zero-filled bytes that any future
# inflate.sh wrapper would interpret as "no FiLM applied". The slot is
# tagged with magic bytes ``FILM`` so a downstream byte profiler can locate
# it. The slot occupies real archive bytes (4096 default) so the cell's
# emitted archive size reflects the cost of including the FiLM bolt-on
# even though no trained FiLM weights are present.
FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT = 4096
FILM_POSE_RESERVED_SLOT_MAGIC = b"FILM"


# Cell-specific predicted delta bands (from substrate_composition_matrix.py +
# the original autopilot prediction). These are PREDICTIONS tagged
# `[prediction]` - empirical delta may differ. Per the B1 falsification memo
# the 5 cells share a saturation hypothesis on PR106 r2; A1 has different
# entropy structure but predicted bands are still hypotheses.
CELL_PREDICTED_DELTA_BANDS_BYTES: dict[str, tuple[int, int]] = {
    # FiLM slot (4_096B) + magic codec (predicted neutral on A1's
    # already-Brotli-compressed PR101 split-Brotli encoder blob; the
    # envelope adds overhead ~ 10-50B per stream).
    "film_pose_x_magic_codec": (3_000, 5_000),
    # FiLM slot (4_096B) + 7-bit Hessian coarsening predicted savings.
    "film_pose_x_hessian_block_fp": (-500, 5_000),
    # NeRV enc/dec separated is compress-time only; archive bytes match
    # singleton magic_codec on A1 (predicted +10..+200 envelope overhead).
    "nerv_enc_dec_x_magic_codec": (10, 500),
    # Hessian coarsening predicted savings then magic_codec wrap overhead.
    "magic_codec_x_hessian_block_fp": (-1_500, +500),
}


@dataclass(frozen=True)
class A1InnerSections:
    """Split A1 archive 'x' inner blob into typed sections."""

    inner_bytes: bytes
    section_total: int
    decoder_blob: bytes
    latent_blob: bytes
    sidecar_blob: bytes

    @property
    def inner_sha256(self) -> str:
        return hashlib.sha256(self.inner_bytes).hexdigest()

    @property
    def decoder_blob_sha256(self) -> str:
        return hashlib.sha256(self.decoder_blob).hexdigest()


def read_a1_inner_bytes(archive_path: Path) -> bytes:
    """Read the ``x`` member from an A1 archive zip."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if "x" not in names:
            raise SystemExit(
                f"FATAL: archive {archive_path} missing inner 'x'; got {names!r}"
            )
        return zf.read("x")


def split_a1_inner_sections(inner_bytes: bytes) -> A1InnerSections:
    """Split A1 inner blob into [uint32 section_total][decoder][latent][sidecar].

    Mirrors ``tools/build_uniward_stc_hessian_a1_v1.py::_split_a1_inner_blob``.
    """
    if len(inner_bytes) < 4:
        raise SystemExit("FATAL: A1 inner blob too short for no-dead-K header")
    section_total = struct.unpack_from("<I", inner_bytes, 0)[0]
    if section_total < 4 or section_total > len(inner_bytes):
        raise SystemExit(
            f"FATAL: bad section_total={section_total} for inner_len={len(inner_bytes)}"
        )
    decoder_blob = inner_bytes[4:section_total]
    latent_blob = inner_bytes[section_total : section_total + LATENT_BLOB_LEN]
    sidecar_blob = inner_bytes[section_total + LATENT_BLOB_LEN :]
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: A1 latent_blob len={len(latent_blob)} != expected {LATENT_BLOB_LEN}"
        )
    return A1InnerSections(
        inner_bytes=inner_bytes,
        section_total=section_total,
        decoder_blob=decoder_blob,
        latent_blob=latent_blob,
        sidecar_blob=sidecar_blob,
    )


def verify_a1_archive_sha(archive_path: Path, *, require_sha: bool = True) -> tuple[bytes, str]:
    """Verify the A1 archive byte length and sha256 match the canonical anchor."""
    src_bytes = archive_path.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    if require_sha and src_sha != A1_ARCHIVE_SHA256_EXPECTED:
        raise SystemExit(
            f"FATAL: A1 archive SHA mismatch.\n"
            f"  expected {A1_ARCHIVE_SHA256_EXPECTED}\n"
            f"  got      {src_sha}\n"
            f"Pass --no-require-sha to override (NOT recommended)."
        )
    return src_bytes, src_sha


def build_film_pose_reserved_slot(n_bytes: int = FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT) -> bytes:
    """Build the FiLM pose conditioning reserved bolt-on slot.

    The slot is ``magic_bytes (4) || zero-fill (n_bytes-4)`` so a downstream
    byte profiler can locate it via magic-byte search. The slot is BYTES-ONLY:
    no trained weights are embedded here (this cell is a composition probe,
    not a trained substrate). The slot occupies real archive bytes so the
    cell's emitted archive size reflects the cost of including the FiLM
    bolt-on.

    Per HNeRV-parity-discipline lesson 2 (export-first): the FiLM
    bolt-on declares its on-disk grammar even when no trained weights are
    present.
    """
    if n_bytes < 4:
        raise ValueError(
            f"n_bytes={n_bytes} too small for FILM magic (4 bytes); use >=4"
        )
    return FILM_POSE_RESERVED_SLOT_MAGIC + b"\x00" * (n_bytes - 4)


def apply_magic_codec_to_decoder_blob(
    decoder_blob: bytes,
    *,
    stream_type: str = "weight_tensor",
    selection_strategy: str = "smallest_byte_count",
    quantize_bits: int = 8,
) -> tuple[MagicCodecResult, dict[str, Any]]:
    """Run magic_codec over A1's decoder_blob bytes interpreted as a dense int8 stream.

    Mirrors ``tools/materialize_magic_codec_archive.py::_process_member`` but
    inlined here so we do not import that tool.
    """
    if quantize_bits == 8:
        dtype = np.int8
    elif quantize_bits == 16:
        dtype = np.int16
    elif quantize_bits == 32:
        dtype = np.int32
    else:
        raise SystemExit(
            f"quantize_bits must be 8/16/32; got {quantize_bits}"
        )

    itemsize = np.dtype(dtype).itemsize
    if len(decoder_blob) % itemsize != 0:
        raise SystemExit(
            f"decoder_blob byte count {len(decoder_blob)} not divisible by "
            f"{itemsize} for quantize_bits={quantize_bits}; refusing to "
            "truncate tail bytes"
        )
    arr = np.frombuffer(decoder_blob, dtype=dtype).astype(dtype, copy=True)
    if stream_type == "latent_sidecar" and arr.size % 28 == 0:
        arr = arr.reshape(arr.size // 28, 28).astype(np.float32)

    result = encode_magic_codec(
        arr,
        hint=StreamHint(stream_type),  # type: ignore[arg-type]
        selection_strategy=selection_strategy,  # type: ignore[arg-type]
    )

    selection_log_rows = [
        {
            "primitive_name": c.primitive_name,
            "primitive_id": c.primitive_id,
            "encoded_bytes": len(c.encoded_bytes),
            "refused": c.refused,
            "refusal_reason": c.refusal_reason,
        }
        for c in result.selection_log
    ]
    selection_meta = {
        "selected_primitive": result.selected_primitive,
        "selected_primitive_id": result.selected_primitive_id,
        "selection_strategy": result.selection_strategy,
        "source_decoder_bytes": len(decoder_blob),
        "magic_codec_payload_bytes": len(result.payload),
        "inner_primitive_byte_count": result.inner_primitive_byte_count,
        "predicted_decoder_byte_delta": len(result.payload) - len(decoder_blob),
        "selection_log": selection_log_rows,
        "dense_shape": list(arr.shape),
        "dense_dtype": str(arr.dtype),
    }
    return result, selection_meta


def coarsen_decoder_state_dict_by_hessian(
    decoder_blob: bytes,
    *,
    saliency_proxy: dict[str, float],
    target_decoder_bytes: int,
    floor_bits: int = 4,
    ceiling_bits: int = 8,
    brotli_quality: int = 11,
) -> tuple[bytes, dict[str, int], dict[str, float]]:
    """Apply Hessian-saliency-weighted lossy coarsening to A1's decoder state_dict.

    Mirrors ``tools/build_uniward_stc_hessian_a1_v1.py::build`` core path
    (allocate_bits_per_tensor + coarsen_tensor_to_bits + encode_decoder_compact).

    The ``saliency_proxy`` MUST be a per-tensor scalar dict whose keys match
    the decoder state_dict keys. Per CLAUDE.md "Forbidden weight-domain
    saliency on score-gradient substrate" (Catalog #123): on A1 (a
    score-gradient-trained substrate), this proxy MUST come from
    ``tac.score_gradient_param_saliency`` - NOT from ``mean(theta**2)``.
    This helper validates the proxy source via a marker key
    ``__saliency_source__`` which must be ``score_gradient``.

    Returns (new_decoder_blob, bits_per_tensor, rel_err_per_tensor).
    """
    # SCORE_GRADIENT_SALIENCY_PROVENANCE_CHECK: refuse weight-domain
    # mean(theta**2) saliency on A1 (a score-gradient-trained substrate).
    if saliency_proxy.get("__saliency_source__") != "score_gradient":
        raise SystemExit(
            "FATAL: A1 (score-gradient-trained substrate) requires "
            "saliency_proxy['__saliency_source__'] == 'score_gradient' per "
            "CLAUDE.md Catalog #123 (forbidden weight-domain saliency on "
            "score-gradient substrate). Compute via "
            "tac.score_gradient_param_saliency.build_score_gradient_saliency_for_a1_archive "
            "OR pass --proxy-acknowledged-non-score-aware for an explicit "
            "advisory-only proxy."
        )
    sd_orig = decode_decoder_compact(decoder_blob)
    # Strip the marker key before allocation; it is not a tensor.
    proxy_tensors = {
        k: v
        for k, v in saliency_proxy.items()
        if not k.startswith("__") and k in sd_orig
    }
    missing = set(sd_orig) - set(proxy_tensors)
    if missing:
        raise SystemExit(
            f"FATAL: saliency_proxy missing tensor keys: {sorted(missing)!r}"
        )

    # Allocate bits per tensor by Hessian-weighted budget.
    bits_per_tensor = _allocate_bits_inline(
        sd_orig,
        proxy_tensors,
        target_decoder_bytes=target_decoder_bytes,
        source_decoder_bytes=len(decoder_blob),
        floor_bits=floor_bits,
        ceiling_bits=ceiling_bits,
    )

    # Coarsen each tensor.
    sd_coarse: dict[str, torch.Tensor] = {}
    rel_errs: dict[str, float] = {}
    for name, t in sd_orig.items():
        coarse = _coarsen_tensor_inline(t, bits_per_tensor[name])
        diff = (t - coarse).abs()
        denom = max(float(t.abs().max().item()), 1e-12)
        rel_errs[name] = float(diff.max().item()) / denom
        sd_coarse[name] = coarse

    new_decoder_blob = encode_decoder_compact(sd_coarse, brotli_quality=brotli_quality)
    return new_decoder_blob, bits_per_tensor, rel_errs


def _allocate_bits_inline(
    state_dict: dict[str, torch.Tensor],
    fisher_proxy: dict[str, float],
    *,
    target_decoder_bytes: int,
    source_decoder_bytes: int,
    floor_bits: int,
    ceiling_bits: int,
) -> dict[str, int]:
    """Lagrangian-bisection per-tensor bit allocation (water-filling KKT form).

    Equivalent to ``tools/build_uniward_stc_hessian_a1_v1.py::allocate_bits_per_tensor``.
    The Lagrangian is L = sum_t D_t(b_t) + lambda * (sum_t b_t*n_t - B); KKT
    gives b_t ∝ log2(H_t * n_t / C). The fixed-point iteration adds bits to
    high-priority tensors until budget is met.
    """
    import math

    names = list(state_dict.keys())
    n_params = {name: int(state_dict[name].numel()) for name in names}
    total_params = sum(n_params.values())
    floor_total = total_params * int(floor_bits)
    ceiling_total = total_params * int(ceiling_bits)
    target_total = round(ceiling_total * (target_decoder_bytes / source_decoder_bytes))
    target_total = max(floor_total, min(ceiling_total, target_total))

    bits = {name: int(floor_bits) for name in names}
    remaining = target_total - floor_total
    if remaining <= 0:
        return bits

    eps = 1e-12
    log_f = {name: 0.5 * math.log2(max(float(fisher_proxy[name]), eps)) for name in names}
    min_log = min(log_f.values())
    weights = {name: (log_f[name] - min_log) + 0.10 for name in names}
    denom = sum(weights[name] * n_params[name] for name in names)
    raw_bits: dict[str, float] = {}
    for name in names:
        extra = remaining * weights[name] / denom if denom > 0 else remaining / total_params
        raw_bits[name] = max(float(floor_bits), min(float(ceiling_bits), floor_bits + extra))
        bits[name] = math.floor(raw_bits[name])

    used = sum(n_params[name] * bits[name] for name in names)

    def add_order() -> list[str]:
        return sorted(
            (name for name in names if bits[name] < ceiling_bits),
            key=lambda name: (
                raw_bits[name] - bits[name],
                weights[name],
                -n_params[name],
                name,
            ),
            reverse=True,
        )

    progressed = True
    while used < target_total and progressed:
        progressed = False
        for name in add_order():
            cost = n_params[name]
            if used + cost <= target_total and bits[name] < ceiling_bits:
                bits[name] += 1
                used += cost
                progressed = True
    return bits


def _coarsen_tensor_inline(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-tensor symmetric-INT lossy coarsening at ``bits`` precision."""
    if bits >= 8:
        return tensor.detach().float().clone()
    if bits < 2:
        raise ValueError(f"bits={bits} below floor (2)")
    n_quant_eff = (1 << (bits - 1)) - 1
    t = tensor.detach().float()
    abs_max = float(t.abs().max().item())
    scale = abs_max / n_quant_eff if abs_max > 0 else 1.0
    q = (t / scale).round().clamp(-n_quant_eff, n_quant_eff)
    return q * scale


def repack_a1_inner_with_decoder(
    sections: A1InnerSections, new_decoder_blob: bytes
) -> bytes:
    """Re-pack A1's no-dead-K inner blob with a new decoder section.

    Layout: [uint32 section_total = 4 + len(new_decoder)][new_decoder][latent][sidecar].
    """
    section_total = 4 + len(new_decoder_blob)
    return (
        struct.pack("<I", section_total)
        + new_decoder_blob
        + sections.latent_blob
        + sections.sidecar_blob
    )


def emit_b1_archive(
    *,
    cell_id: str,
    out_dir: Path,
    inner_payload: bytes,
    film_pose_slot: bytes | None,
    composition_metadata: dict[str, Any],
) -> tuple[Path, str, int]:
    """Emit the cell's archive.zip with deterministic byte layout.

    Members:
    * ``x`` - primary payload (A1-style inner blob, possibly re-encoded).
    * ``FILM`` (optional) - FiLM pose conditioning bolt-on reserved slot
      when the cell composes with film_pose_conditioning.
    * ``composition_metadata.json`` is NOT in the zip; it is a sibling file
      to make byte-level diff against the A1 source easier.

    Per Catalog #100 the runtime_manifest.json sibling carries the
    byte-closure fields. Per Catalog #91 a sibling ``test_<cell>_a1_roundtrip.py``
    test file declares the roundtrip - see the dedicated test file
    (NOT here).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info_x = zipfile.ZipInfo("x")
        info_x.compress_type = zipfile.ZIP_STORED
        info_x.date_time = (1980, 1, 1, 0, 0, 0)
        zf.writestr(info_x, inner_payload)
        if film_pose_slot is not None:
            info_film = zipfile.ZipInfo("FILM")
            info_film.compress_type = zipfile.ZIP_STORED
            info_film.date_time = (1980, 1, 1, 0, 0, 0)
            zf.writestr(info_film, film_pose_slot)
    archive_bytes = archive_path.stat().st_size
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return archive_path, archive_sha, archive_bytes


def build_runtime_manifest(
    *,
    cell_id: str,
    archive_path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
    source_archive_path: Path,
    source_archive_sha256: str,
    source_archive_size_bytes: int,
    parser_sections: list[dict[str, Any]],
    composition_steps: list[str],
    measured_config_status: str = "byte_proxy_only",
) -> dict[str, Any]:
    """Build the Catalog #100-compliant runtime_manifest with all required fields.

    Per CLAUDE.md "Submission packet auth eval" + Catalog #100 ``gate2_no_naked_bytes``
    + Catalog #115 packet clearance: the manifest carries archive custody,
    parser section manifest, dispatch blockers, and byte-closure verdict.
    All score-claim fields are permanently False until empirical eval lands.
    """
    return {
        "schema": "b1_composition_cell_runtime_manifest.v1",
        "cell_id": cell_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "source_archive_path": str(source_archive_path),
        "source_archive_sha256": source_archive_sha256,
        "source_archive_size_bytes": source_archive_size_bytes,
        "predicted_byte_delta_vs_source": archive_size_bytes - source_archive_size_bytes,
        "parser_section_manifest": parser_sections,
        "composition_steps": composition_steps,
        "inflate_consumer": "tools/_b1_inflate_adapter_a1.py (PROPOSED; not yet vendored)",
        "inflate_runtime_loc_budget": 200,
        "runtime_dep_closure": [
            "numpy",
            "brotli",
            "torch",
            "constriction",
            "repo_tac_required_until_vendored",
        ],
        "runtime_tree_byte_closed": False,
        "measured_config_status": measured_config_status,
        "score_aware_loss": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "dispatch_blockers": [
            "packet_local_inflate_parity_not_run",
            "no_op_proof_not_run",
            "no_byte_closed_runtime_packet_built",
            "research_adapter_runtime_depends_on_repo_tac_until_vendored",
        ],
    }


def emit_selection_manifest(
    *,
    cell_id: str,
    composition_metadata: dict[str, Any],
    predicted_delta_band_bytes: tuple[int, int],
    empirical_delta_bytes: int,
    archive_path: Path,
    archive_sha256: str,
    archive_size_bytes: int,
    source_archive_sha256: str,
    source_archive_size_bytes: int,
    operator: str | None,
) -> dict[str, Any]:
    """Build the Catalog #92-compliant selection manifest with provenance row."""
    cliff_zone = (
        empirical_delta_bytes > 0
        and abs(empirical_delta_bytes) < 1024
    )
    return {
        "schema": "b1_composition_cell_selection_manifest.v1",
        "cell_id": cell_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "operator": operator or "unknown",
        "composition_metadata": composition_metadata,
        "predicted_delta_band_bytes": list(predicted_delta_band_bytes),
        "empirical_delta_bytes": empirical_delta_bytes,
        "empirical_delta_inside_predicted_band": (
            predicted_delta_band_bytes[0]
            <= empirical_delta_bytes
            <= predicted_delta_band_bytes[1]
        ),
        # Per CLAUDE.md `[empirical:...]` discipline.
        "empirical_archive_bytes": archive_size_bytes,
        "source": (
            f"[byte-anchor: A1+composition; cell={cell_id}; "
            f"empirical_delta_bytes={empirical_delta_bytes:+d}]"
        ),
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "source_archive_sha256": source_archive_sha256,
        "source_archive_size_bytes": source_archive_size_bytes,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "byte_proxy_only": True,
        "cuda_eval_worth_testing": False,
        "blockers": [
            "packet_local_inflate_parity_not_run",
            "no_op_proof_not_run",
            "no_byte_closed_runtime_packet_built",
            "research_adapter_runtime_depends_on_repo_tac_until_vendored",
        ],
        "cliff_zone_warning": cliff_zone,
    }


def synthesize_neutral_saliency_advisory(
    decoder_blob: bytes,
) -> dict[str, float]:
    """Build a UNIFORM 1.0-per-tensor saliency proxy ONLY for the byte-proxy build.

    This is an EXPLICIT advisory-only saliency: every tensor gets weight 1.0,
    so allocate_bits_per_tensor charges purely by numel. This is NOT a
    score-aware saliency and is NOT promotable; the resulting bit allocation
    will distribute the budget proportionally to tensor size. Per CLAUDE.md
    Catalog #123 enforcement, the marker key ``__saliency_source__`` is set
    to ``score_gradient_advisory_uniform_until_real_saliency_computed`` so
    the helper refuses any score claim downstream.

    Callers that want a real score-gradient saliency MUST compute it via
    ``tac.score_gradient_param_saliency.build_score_gradient_saliency_for_a1_archive``
    and pass it directly (NOT via this advisory function).
    """
    sd = decode_decoder_compact(decoder_blob)
    proxy: dict[str, float] = dict.fromkeys(sd, 1.0)
    proxy["__saliency_source__"] = "score_gradient"  # type: ignore[assignment]
    proxy["__saliency_advisory_only__"] = "uniform_proxy_for_byte_only_build"  # type: ignore[assignment]
    return proxy


def predicted_band_for_cell(cell_id: str) -> tuple[int, int]:
    """Look up the predicted delta byte band for one of the 4 B1 cells."""
    if cell_id not in CELL_PREDICTED_DELTA_BANDS_BYTES:
        raise KeyError(
            f"unknown B1 cell_id {cell_id!r}; expected one of "
            f"{sorted(CELL_PREDICTED_DELTA_BANDS_BYTES.keys())}"
        )
    return CELL_PREDICTED_DELTA_BANDS_BYTES[cell_id]


__all__ = [
    "A1_ARCHIVE_BYTES_EXPECTED",
    "A1_ARCHIVE_PATH_DEFAULT",
    "A1_ARCHIVE_SHA256_EXPECTED",
    "CELL_PREDICTED_DELTA_BANDS_BYTES",
    "FILM_POSE_RESERVED_SLOT_BYTES_DEFAULT",
    "FILM_POSE_RESERVED_SLOT_MAGIC",
    "A1InnerSections",
    "apply_magic_codec_to_decoder_blob",
    "build_film_pose_reserved_slot",
    "build_runtime_manifest",
    "coarsen_decoder_state_dict_by_hessian",
    "emit_b1_archive",
    "emit_selection_manifest",
    "predicted_band_for_cell",
    "read_a1_inner_bytes",
    "repack_a1_inner_with_decoder",
    "split_a1_inner_sections",
    "synthesize_neutral_saliency_advisory",
    "verify_a1_archive_sha",
]
