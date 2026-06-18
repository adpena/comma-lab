# SPDX-License-Identifier: MIT
"""Frontier-decoder PTQ — decode / requantize / re-encode the 0.19110 frontier decoder.

The contest-CPU frontier (lane ``pr110_payload_entropy_recode``, archive 177,169 B,
sha ``b46897267…``) ships its 228,958-param HNeRV decoder as PR #101 per-tensor symmetric
int8 (N_QUANT=127) → split-brotli → CTXR-recoded FP11 member. The decoder section is
161,104 B = 90.9% of the archive (the QAT-attackable share); the latent + sidecar +
selector + DQS1 tail are kept verbatim (lossless, distortion-neutral).

This module exposes the reusable round-trip that lets a $0 PTQ feasibility sweep measure
the **PTQ floor** — the actual exact-S of NAIVE (and SCORE-AWARE-allocated) bit-shrinking
the frontier decoder, which is the floor that score-aware QAT must recover from:

  decode  : frontier archive member → CTXR source → raw decoder streams → reshape →
            torch state_dict (the exact int8*fp16 weights the frontier ships).
  shrink  : ``tac.post_hoc_weight_shrink.requantize_decoder_state_dict`` (uniform int-N)
            OR :func:`score_aware_requant_state_dict` (per-stage allocation: coarsen the
            d_seg-blind early/low-res stages to int4 where 77% of the params live, protect
            the tiny output-proximal d_seg-critical heads at higher precision).
  encode  : shrunk state_dict → PR #101 ``encode_decoder_compact`` (brotli) → reassemble
            legacy source (decoder_blob + lzma(latent_raw) + sidecar) → CTXR recode →
            ``join_fp11_member`` → STORED zip → MEASURED archive bytes.

The identity round-trip (int8, unchanged weights) reproduces the frontier archive
byte-for-byte (177,169 B, state_dict max_abs_delta=0) — proven by the module's own test —
so the PTQ byte/distortion numbers are apples-to-apples against the real frontier.

AUTHORITY: every score this module helps produce is ``[contest-CPU advisory]``
NON-PROMOTABLE (local CPU ≈ contest-CPU per G3, but this is a FEASIBILITY measurement,
NOT a byte-closed dual-exact row). The frontier pointer stays pointer-only at 0.19110. No
score / frontier / promotion claim. CPU only (NO MPS — MPS owns the live train and
2x-corrupts SegNet / 23x PoseNet). $0 — no GPU dispatch, no paid spend.
"""

from __future__ import annotations

import lzma
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tac.packet_compiler.feca_selector_reparameterize import (
    FEC8_MAGIC,
    FECA_MAGIC,
    join_fp11_member,
    split_fp11_member,
)
from tac.packet_compiler.pr110_payload_entropy_recode import (
    LATENT_LZMA_FILTERS,
    encode_source_payload,
    read_single_member,
    reconstruct_raw_sections,
    write_stored_archive,
)
from tac.post_hoc_weight_shrink import intn_qdq, requantize_decoder_state_dict
from tac.pr101_split_brotli_codec import (
    CONV4_STORAGE_PERMS,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    FIXED_STATE_SCHEMA,
    decode_latents_compact,
    decode_mapped_u8,
    encode_decoder_compact,
)

# The frontier archive (contest-CPU pointer, sha b46897267…).
FRONTIER_ARCHIVE = Path(
    "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip"
)
_ALLOWED_SELECTOR_MAGICS = (FECA_MAGIC, FEC8_MAGIC)
RATE_DENOM = 37_545_489


class FrontierDecoderPtqError(ValueError):
    """Raised when the frontier PTQ round-trip is malformed."""


# ---------------------------------------------------------------------------
# decode: frontier archive → state_dict + latents (+ the verbatim member parts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierMember:
    """The decoded frontier member, with everything needed to re-encode losslessly."""

    member_name: str
    member_bytes: int
    selector_payload: bytes
    dqs1_tail: bytes
    raw_decoder_joined: bytes
    latent_raw: bytes
    sidecar: bytes
    state_dict: dict[str, torch.Tensor]
    latents: torch.Tensor  # (600, 28) decoded runtime latents (sidecar applied)


def _reshape_raw_to_state_dict(raw: bytes) -> dict[str, torch.Tensor]:
    """Reshape the post-brotli-decompress raw decoder bytes → torch state_dict.

    This is the reshape loop of PR #101 ``decode_decoder_compact`` operating on the
    ALREADY-decompressed ``raw`` (what ``reconstruct_raw_sections`` returns), so it must
    NOT re-decompress. Honors the default per-tensor byte maps + conv4 storage perms the
    frontier was encoded with."""
    pos = 0
    sd: dict[str, torch.Tensor] = {}
    for idx in DECODER_STORAGE_ORDER:
        name, shape = FIXED_STATE_SCHEMA[idx]
        numel = int(np.prod(shape))
        zz = np.frombuffer(raw, dtype=np.uint8, count=numel, offset=pos)
        pos += numel
        scale = np.frombuffer(raw, dtype=np.float16, count=1, offset=pos)[0]
        pos += 2
        decode_map = DECODER_BYTE_MAPS.get(idx, "zig")
        q = decode_mapped_u8(zz, decode_map)
        if len(shape) == 4 and idx in CONV4_STORAGE_PERMS:
            storage_perm = CONV4_STORAGE_PERMS[idx]
            inverse_perm = tuple(int(v) for v in np.argsort(storage_perm))
            stored_shape = tuple(shape[i] for i in storage_perm)
            q = q.reshape(stored_shape)
            q = np.transpose(q, inverse_perm).copy()
        else:
            q = q.reshape(shape)
        sd[name] = torch.from_numpy(q.astype(np.float32)) * float(scale)
    if pos != len(raw):
        raise FrontierDecoderPtqError(
            f"trailing/truncated raw decoder bytes: pos={pos} len={len(raw)}"
        )
    return sd


def decode_frontier_member(archive_path: str | Path = FRONTIER_ARCHIVE) -> FrontierMember:
    """Decode the frontier archive into its state_dict + latents + verbatim parts."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FrontierDecoderPtqError(f"frontier archive not found: {archive_path}")
    member_name, member = read_single_member(archive_path)
    parts = split_fp11_member(member, allowed_selector_magics=_ALLOWED_SELECTOR_MAGICS)
    raw_streams, latent_raw, sidecar = reconstruct_raw_sections(parts["source_payload"])
    raw_joined = b"".join(raw_streams)
    state_dict = _reshape_raw_to_state_dict(raw_joined)
    # Decode the runtime latents (lzma re-compress latent_raw → the compact latent_blob
    # → decode_latents_compact). The lzma re-compress is proven byte-identical (the
    # frontier's latent_blob is the canonical raw-LZMA1 form).
    latent_blob = lzma.compress(latent_raw, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS)
    latents = decode_latents_compact(latent_blob)
    return FrontierMember(
        member_name=member_name,
        member_bytes=len(member),
        selector_payload=parts["selector_payload"],
        dqs1_tail=parts["dqs1_tail"],
        raw_decoder_joined=raw_joined,
        latent_raw=latent_raw,
        sidecar=sidecar,
        state_dict=state_dict,
        latents=latents,
    )


# ---------------------------------------------------------------------------
# score-aware per-stage bit allocation (the taper-on-the-bit-axis lever)
# ---------------------------------------------------------------------------

# Output-proximal d_seg-CRITICAL stages: the late, high-resolution (192×256 / 384×512)
# stages whose argmax-flips drive d_seg. These are TINY in params (~16K = 7% of the
# decoder) so protecting them at higher precision is nearly free on bytes. Everything
# else (stem + early low-res blocks, where 77% of the params live and the signal is
# d_seg-blind low-frequency structure) goes to the coarse grid.
DSEG_CRITICAL_PREFIXES: tuple[str, ...] = (
    "blocks.4.",
    "blocks.5.",
    "skips.",
    "refine.",
    "rgb_0.",
    "rgb_1.",
)


def score_aware_requant_state_dict(
    dec_sd: dict[str, torch.Tensor],
    *,
    low_nbits: int = 4,
    high_nbits: int = 8,
    critical_prefixes: tuple[str, ...] = DSEG_CRITICAL_PREFIXES,
    weight_min_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Score-aware per-stage int-N allocation (the taper on the bit axis).

    The d_seg-critical output-proximal stages (``critical_prefixes``) are quantized at
    ``high_nbits`` (precise), the d_seg-blind early/low-res stages (where ~77% of the
    params live) at ``low_nbits`` (coarse). This is the inverse-Pareto lever: spend the
    cheap precision where d_seg lives, shed the bytes where d_seg is blind.

    Args:
        dec_sd: frontier decoder state dict (float tensors).
        low_nbits: coarse grid for the d_seg-blind early/interior stages.
        high_nbits: precise grid for the d_seg-critical output-proximal stages.
        critical_prefixes: key prefixes treated as d_seg-critical (kept at high_nbits).
        weight_min_dim: minimum tensor rank to quantize (biases/1-D params stay fp32).

    Returns:
        A new state dict (does not mutate the input).
    """
    if low_nbits < 2 or high_nbits < 2:
        raise FrontierDecoderPtqError("nbits must be >= 2")
    out: dict[str, torch.Tensor] = {}
    for k, v in dec_sd.items():
        if not torch.is_tensor(v) or v.dim() < weight_min_dim:
            out[k] = v.clone()
            continue
        nbits = high_nbits if any(k.startswith(p) for p in critical_prefixes) else low_nbits
        out[k] = intn_qdq(v, nbits)
    return out


# ---------------------------------------------------------------------------
# encode: shrunk state_dict → byte-closed frontier archive (MEASURED bytes)
# ---------------------------------------------------------------------------


def reencode_frontier_archive(
    member: FrontierMember,
    shrunk_state_dict: dict[str, torch.Tensor],
    out_path: str | Path,
    *,
    auto_select_byte_maps: bool = False,
) -> int:
    """Re-encode the frontier archive with a (possibly shrunk) decoder; return bytes.

    Only the decoder section changes; the latent payload + sidecar + selector + DQS1 tail
    are kept verbatim (lossless). Returns the MEASURED archive byte length (STORED zip).

    Args:
        member: the decoded frontier member (verbatim parts + latents).
        shrunk_state_dict: the requantized decoder state dict.
        out_path: where to write the byte-closed archive.
        auto_select_byte_maps: if True, let PR #101 pick brotli-optimal byte maps for the
            shrunk weights (a tighter, still-valid encode). Default False = the faithful
            "ship through the EXISTING frontier codec" byte count (apples-to-apples, since
            the decoder MUST then carry the chosen maps — out of scope for this floor
            measurement; the default-map number is the honest conservative floor).
    """
    if auto_select_byte_maps:
        # auto_select picks brotli-optimal per-tensor byte maps for the shrunk weights
        # (a tighter, still-valid encode). encode_decoder_compact returns just the bytes
        # (the chosen maps are computed internally); the archive bytes are the honest
        # tighter byte count. NOTE: such an archive would need the chosen maps to
        # re-decode, so this path is a BYTE estimate only — exact_eval runs on the
        # in-memory shrunk state_dict, not on a re-decode of this archive.
        decoder_blob = encode_decoder_compact(shrunk_state_dict, auto_select=True)
    else:
        decoder_blob = encode_decoder_compact(shrunk_state_dict)
    latent_blob = lzma.compress(
        member.latent_raw, format=lzma.FORMAT_RAW, filters=LATENT_LZMA_FILTERS
    )
    legacy_source = decoder_blob + latent_blob + member.sidecar
    container, _metrics = encode_source_payload(legacy_source, decoder_blob_len=len(decoder_blob))
    new_member = join_fp11_member(
        source_payload=container,
        selector_payload=member.selector_payload,
        dqs1_tail=member.dqs1_tail,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_stored_archive(out_path, member_name=member.member_name, payload=new_member)
    return out_path.stat().st_size


def build_frontier_decoder(state_dict: dict[str, torch.Tensor], *, eval_size=(384, 512)):
    """Instantiate the frontier HNeRV decoder (base_channels=36) and load ``state_dict``.

    Uses the vendored frontier ``submission_dir/src/model.py`` HNeRVDecoder so the
    forward matches the contest runtime exactly. Returns an ``.eval()`` module on CPU."""
    import importlib.util

    model_path = (
        Path("experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/src/model.py")
    )
    if not model_path.is_file():
        raise FrontierDecoderPtqError(f"frontier model.py not found: {model_path}")
    spec = importlib.util.spec_from_file_location("_frontier_model", model_path)
    if spec is None or spec.loader is None:
        raise FrontierDecoderPtqError("could not load frontier model.py spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decoder = module.HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=eval_size).eval()
    decoder.load_state_dict(state_dict)
    return decoder


def score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """The exact contest score S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/B0."""
    return 100.0 * d_seg + (10.0 * d_pose + 1e-12) ** 0.5 + 25.0 * archive_bytes / RATE_DENOM


__all__ = [
    "DSEG_CRITICAL_PREFIXES",
    "FRONTIER_ARCHIVE",
    "FrontierDecoderPtqError",
    "FrontierMember",
    "build_frontier_decoder",
    "decode_frontier_member",
    "reencode_frontier_archive",
    "requantize_decoder_state_dict",
    "score",
    "score_aware_requant_state_dict",
]
