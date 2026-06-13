# SPDX-License-Identifier: MIT
"""Lever 3 (the Quantizr STORE-pose lesson) for the torch-vehicle HNeRV path.

The design memo (``.omx/research/incurriculum_levers_design_floor_chasing_20260612.md``
Lever 3): instead of asking the decoder to RECOVER pose from pixels (the binding
``d_pose`` constraint — PoseNet runs on the rendered frame, the decoder must learn
to produce frames whose PoseNet readout matches GT, a hard inverse problem), we
STORE the 6 GT pose scalars per pair and FiLM-condition the decoder on them. The
pose enters as side information (Wyner-Ziv: the decoder HAS the pose, so it does
not pay to encode it in the weights); ``d_pose`` collapses toward the
quantization floor of the stored pose, freeing the whole decoder capacity for
``d_seg`` + rate.

This module is the torch-vehicle wire-in: a thin :class:`PoseFiLMHNeRVWrapper`
that WRAPS (never edits) the pristine vendored ``HNeRVDecoder`` and injects FiLM
at the stem (channels[0], the EARLIEST point = maximal effect per the memo). It
is DEFAULT-OFF in the driver (``cfg.pose_film_enabled=False``); when off the
driver builds the vendored decoder unchanged and adds NO pose section, so the
archive is BYTE-IDENTICAL to today (the live base_ch=20 basin is unaffected if it
resumes onto this code — proved by ``tests/test_pose_film_wire_in.py``).

Identity-at-init is MANDATORY (else it perturbs the basin): the FiLM ``fc2`` is
zero-init, so ``gamma=1+tanh(0)=1`` and ``beta=0`` → the FIRST-step FiLM-on render
is bit-equal to the no-FiLM decoder (verified by the identity test).

Export-first pose-section grammar (HNeRV parity L2 + L4): the vendored
``build_archive`` is PRISTINE (we do NOT edit it). The pose section is added at
the WRAPPER level as an ADDITIVE, length-prefixed blob APPENDED after the 3
vendored sections (the vendored ``parse_archive`` reads exactly 3 sections and
stops, ignoring trailing bytes — so a vendored-only reader is unaffected). The
codec mirrors the latent codec style (per-dim min/max → uint8 → 1st-order delta
→ zigzag → brotli q=11). :func:`build_archive_with_pose` / :func:`parse_pose_section`
are the wrapper-level grammar; the pose round-trips through inflate via
:func:`inflate_film_decoder` (numpy-portable, ≤100 LOC equivalent).

Authority: torch-CPU TRUSTED (CLAUDE.md "local CPU + MLX GPU good"); NO MPS for
the exact metric. The in-loop d_pose is ``[contest-CPU advisory]`` NON-PROMOTABLE
until the byte-closed archive is run through ``upstream/evaluate.py``.
"""

from __future__ import annotations

import io
import struct

import brotli
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_POSE_DIM = 6  # contest PoseNet pose (first 6 dims; upstream/modules.py)
# Magic for the additive pose section so a parser can detect its presence/absence
# (a vendored-only archive simply has no trailing bytes after the 3 sections).
_POSE_SECTION_MAGIC = b"PFLM"


class _PoseFiLM(nn.Module):
    """Torch stored-pose FiLM (Quantizr STORE-pose lesson).

    Identical mechanism to :class:`tac.residual_basis.cool_chic_carrier._PoseFiLM`
    and the capstone MLX bundle (``tac.mlx_pr95_port.pose_film``):

        pose (B, 6) -> sin(fc1) -> fc2 -> [gamma_pre, beta]  (each (B, channels))
        gamma = 1 + tanh(gamma_pre)  in (0, 2), =1 at init
        beta  = fc2 bias/weight output, =0 at init

    ``fc2`` is zero-init so FiLM is the EXACT identity at init (gamma=1, beta=0):
    the untrained wrapper renders EXACTLY the no-FiLM vendored decoder, so the
    live basin is unperturbed if it resumes onto this code, and FiLM only *adds*
    pose grammar as it trains.
    """

    def __init__(self, *, pose_dim: int, channels: int, hidden: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.fc1 = nn.Linear(int(pose_dim), int(hidden))
        self.fc2 = nn.Linear(int(hidden), 2 * int(channels))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.sin(self.fc1(pose))  # NeRF-style activation (matches capstone)
        gb = self.fc2(h)
        gamma_pre = gb[:, : self.channels]
        beta = gb[:, self.channels :]
        gamma = 1.0 + torch.tanh(gamma_pre)  # (0, 2), =1 at init
        return gamma, beta

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PoseFiLMHNeRVWrapper(nn.Module):
    """WRAP (never edit) the vendored ``HNeRVDecoder`` with a stored-pose FiLM at
    the stem (channels[0]) — the EARLIEST injection point = maximal effect (memo).

    The vendored ``HNeRVDecoder.forward(z)`` is::

        x = stem(z).view(B, channels[0], base_h, base_w)
        x = sin(x)
        for block, skip in zip(blocks, skips):
            identity = skip(interpolate(x, x2))
            x = ps(block(x))
            x = sin(x + identity)
        x = x + 0.1 * sin(refine(x))
        f0 = sigmoid(rgb_0(x)) * 255 ; f1 = sigmoid(rgb_1(x)) * 255
        return stack([f0, f1], dim=1)

    We REPLICATE that flow EXACTLY (calling the vendored ``stem`` / ``blocks`` /
    ``skips`` / ``ps`` / ``refine`` / ``rgb_0`` / ``rgb_1`` layers unchanged), and
    inject FiLM ON THE STEM CHANNEL DIM right after the ``stem(z).view(...)`` and
    BEFORE ``sin(x)`` + the block cascade::

        x = stem(z).view(B, channels[0], base_h, base_w)
        x = gamma[:, :, None, None] * x + beta[:, :, None, None]   # <-- FiLM (stem)
        x = sin(x)
        ... vendored cascade unchanged ...

    Because the stem feeds the SHARED ``x`` that both ``rgb_0`` and ``rgb_1`` read,
    the pose conditions the WHOLE pair render (frame0 + frame1). gamma=1/beta=0
    (identity at init) reproduces the vendored render bit-for-bit.

    The wrapper holds the GT pose as a NON-TRAINABLE ``stored_pose`` buffer
    ``(n_pairs, 6)`` (set from the GT PoseNet pose via :meth:`set_stored_pose`);
    ``forward(latents, idx)`` looks the per-pair pose up by index. The buffer is
    range-coded into the archive's additive pose section (~1 KB charged) at
    byte-close — the Quantizr STORE-pose payload, NOT a metadata stub.
    """

    def __init__(
        self,
        decoder: nn.Module,
        *,
        n_pairs: int,
        pose_dim: int = _POSE_DIM,
        film_hidden: int = 8,
    ) -> None:
        super().__init__()
        self.decoder = decoder  # vendored HNeRVDecoder (its params still train)
        self.n_pairs = int(n_pairs)
        self.pose_dim = int(pose_dim)
        # Stem channel count (channels[0]) — the FiLM injection dim.
        self.stem_channels = int(decoder.channels[0])
        self.pose_film = _PoseFiLM(
            pose_dim=self.pose_dim, channels=self.stem_channels, hidden=int(film_hidden)
        )
        self.register_buffer(
            "stored_pose", torch.zeros(self.n_pairs, self.pose_dim), persistent=True
        )

    def set_stored_pose(self, pose: torch.Tensor) -> None:
        """Set the STORED per-pair 6-dim pose buffer (from the GT PoseNet pose).

        This is the Quantizr STORE-pose payload: the contest pose is *stored* (and
        range-coded into the archive ~1 KB at byte-close), NOT reconstructed from
        pixels. ``pose`` is ``(n_pairs, pose_dim)``.
        """
        with torch.no_grad():
            p = pose.detach().to(self.stored_pose.dtype).to(self.stored_pose.device)
            if p.shape != self.stored_pose.shape:
                raise ValueError(
                    f"stored_pose expects {tuple(self.stored_pose.shape)}, "
                    f"got {tuple(p.shape)}"
                )
            self.stored_pose.copy_(p)

    def _forward_with_film(
        self, z: torch.Tensor, pose6: torch.Tensor | None
    ) -> torch.Tensor:
        """Replicate the vendored ``HNeRVDecoder.forward`` EXACTLY, injecting FiLM
        on the stem channel dim before ``sin(x)`` + the cascade.

        ``pose6 is None`` (or gamma=1/beta=0 at init) reproduces the vendored
        forward bit-for-bit (the byte-identity contract)."""
        d = self.decoder
        B = z.shape[0]
        x = d.stem(z).view(B, d.channels[0], d.base_h, d.base_w)
        if pose6 is not None:
            gamma, beta = self.pose_film(pose6)  # each (B, stem_channels)
            x = gamma[:, :, None, None] * x + beta[:, :, None, None]
        x = torch.sin(x)
        # blocks/skips are equal-length by construction (vendored decoder).
        for block, skip in zip(d.blocks, d.skips, strict=False):
            identity = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            identity = skip(identity)
            x = d.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(d.refine(x))
        f0 = torch.sigmoid(d.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(d.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)

    def forward(self, z: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``(B, 2, 3, 384, 512)`` float in ``[0, 255]`` — the pair render
        conditioned on the per-pair stored pose.

        ``z`` is ``(B, latent_dim)``; ``idx`` is the ``(B,)`` pair index used to
        look up ``stored_pose[idx]``. ``idx is None`` renders WITHOUT FiLM (the
        vendored path) — used by the byte-identity proof.
        """
        pose6 = None if idx is None else self.stored_pose[idx]
        return self._forward_with_film(z, pose6)


class _FiLMEvalDecoder(nn.Module):
    """Adapt a :class:`PoseFiLMHNeRVWrapper` to the vendored ``decoder(z)`` eval
    call (which does NOT pass an index).

    The vendored ``score.evaluate_decoder`` (and the synthetic ``exact_eval``)
    stream pairs in STRICT pair order starting at pair 0 and call ``decoder(z)``
    with ``z = latents[arange(pair_idx, pair_idx + B)]``. So a monotonic cursor
    EXACTLY reconstructs the per-pair index for each batch: the first call gets
    rows ``[0, B)``, the next ``[B, 2B)``, etc. We reset the cursor in
    :meth:`reset` before each eval and look up ``stored_pose`` by the cursor so the
    FiLM render is conditioned on the correct per-pair pose — the SAME render the
    inflate path produces (proved by the byte-closed round-trip test). This is the
    minimal-coupling way to make the FiLM eval faithful WITHOUT editing the
    vendored ``evaluate_decoder``."""

    def __init__(self, wrapper: PoseFiLMHNeRVWrapper) -> None:
        super().__init__()
        self.wrapper = wrapper
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def eval(self):  # type: ignore[override]
        # The vendored evaluate_decoder calls decoder.eval() at entry — use that as
        # the per-eval cursor reset so a fresh eval always starts at pair 0.
        self.reset()
        super().eval()
        self.wrapper.eval()
        return self

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        idx = torch.arange(
            self._cursor, self._cursor + B, device=self.wrapper.stored_pose.device
        )
        self._cursor += B
        return self.wrapper(z.to(self.wrapper.stored_pose.device), idx)


# ===========================================================================
# Export-first pose-section grammar (additive; wrapper-level; vendored codec is
# PRISTINE). Layout APPENDED after the 3 vendored sections:
#   [MAGIC:4='PFLM'] [n_pairs:u32] [pose_dim:u32]
#   [mins: pose_dim * f16] [scales: pose_dim * f16]
#   [delta_zigzag_lo: n_pairs*pose_dim * u8] [delta_zigzag_hi: ... * u8]   (brotli q=11)
# Mirrors the vendored latent codec (per-dim min/max → uint8 → 1st-order delta →
# zigzag uint16 → lo/hi byte split → brotli), so the pose stream is as small as
# the latents at equal smoothness. Pose is temporally smooth → mostly-zero hi
# stream → tiny brotli (the ~1 KB the memo predicts).
# ===========================================================================

def wrapper_sd_to_archive_decoder_sd(wrapper_sd: dict) -> dict:
    """Transform a :class:`PoseFiLMHNeRVWrapper` ``state_dict()`` into the codec's
    decoder state dict.

    The wrapper keys are ``decoder.<vendored-key>`` (the vendored submodule), the
    FiLM submodule keys, and ``stored_pose`` (the buffer). The vendored codec +
    ``HNeRVDecoder.load_state_dict`` expect the BARE vendored keys (``stem.weight``
    etc.), so we strip the ``decoder.`` prefix; the FiLM keys are KEPT (they ship in
    the decoder blob so the wrapper rebuilds at inflate) and the inflate path splits
    them back out. The ``stored_pose`` buffer is DROPPED here (it is range-coded into
    the additive pose section, NOT the int8 decoder blob).

    The ``else`` branch is key-name-AGNOSTIC: it passes through ANY non-``decoder.*``,
    non-``stored_pose`` key unchanged. This function is re-exported VERBATIM by
    ``pose_film_v2`` (so it serves BOTH versions): the kept FiLM keys are ``pose_film.*``
    for v1 and ``pose_mlp.*`` + ``film_resid.*`` for v2. Do NOT pre-filter the FiLM keys
    before calling this — the version-specific split happens at inflate
    (:func:`inflate_film_decoder` v1 / :func:`pose_film_v2.inflate_film_decoder_v2` v2)."""
    out: dict = {}
    for k, v in wrapper_sd.items():
        if k == "stored_pose":
            continue  # → additive pose section, not the decoder blob
        if k.startswith("decoder."):
            out[k[len("decoder.") :]] = v
        else:  # pose_film.* (and any other wrapper-level param)
            out[k] = v
    return out


def encode_pose_section(stored_pose: torch.Tensor) -> bytes:
    """Encode the ``(n_pairs, pose_dim)`` stored pose to the additive brotli blob.

    Per-dim min/max → uint8 → 1st-order temporal delta → zigzag uint16 → lo/hi
    byte split → brotli q=11 (the vendored latent codec, applied to pose). The
    returned bytes are the FULL section (magic + sizes + brotli payload), ready to
    append to the vendored archive."""
    t = stored_pose.detach().cpu().float()
    n, dpose = int(t.shape[0]), int(t.shape[1])
    mins = t.min(dim=0).values
    maxs = t.max(dim=0).values
    scales = ((maxs - mins) / 254.0).clamp(min=1e-10)
    q = (
        ((t - mins.unsqueeze(0)) / scales.unsqueeze(0))
        .round()
        .clamp(0, 254)
        .to(torch.uint8)
        .numpy()
    )
    delta = np.empty_like(q, dtype=np.int16)
    delta[0] = q[0]
    delta[1:] = q[1:].astype(np.int16) - q[:-1].astype(np.int16)
    delta_zz = np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint16)
    lo = (delta_zz & 0xFF).astype(np.uint8).tobytes()
    hi = (delta_zz >> 8).astype(np.uint8).tobytes()
    payload = io.BytesIO()
    payload.write(mins.to(torch.float16).numpy().tobytes())
    payload.write(scales.to(torch.float16).numpy().tobytes())
    payload.write(lo)
    payload.write(hi)
    pose_brotli = brotli.compress(payload.getvalue(), quality=11)

    out = io.BytesIO()
    out.write(_POSE_SECTION_MAGIC)
    out.write(struct.pack("<II", n, dpose))
    out.write(struct.pack("<I", len(pose_brotli)))
    out.write(pose_brotli)
    return out.getvalue()


def decode_pose_section(section_bytes: bytes) -> torch.Tensor:
    """Inverse of :func:`encode_pose_section` → ``(n_pairs, pose_dim)`` float pose.

    Reconstructs the per-dim min/max-dequantized, delta-decoded pose. Pure-numpy
    (the inflate path is numpy-portable per HNeRV parity L4)."""
    buf = io.BytesIO(section_bytes)
    magic = buf.read(4)
    if magic != _POSE_SECTION_MAGIC:
        raise ValueError(f"bad pose-section magic {magic!r} (expected {_POSE_SECTION_MAGIC!r})")
    n, dpose = struct.unpack("<II", buf.read(8))
    blob_len = struct.unpack("<I", buf.read(4))[0]
    raw = brotli.decompress(buf.read(blob_len))
    rb = io.BytesIO(raw)
    mins = np.frombuffer(rb.read(dpose * 2), dtype=np.float16).astype(np.float32)
    scales = np.frombuffer(rb.read(dpose * 2), dtype=np.float16).astype(np.float32)
    total = n * dpose
    lo = np.frombuffer(rb.read(total), dtype=np.uint8).astype(np.uint16)
    hi = np.frombuffer(rb.read(total), dtype=np.uint8).astype(np.uint16)
    delta_zz = ((hi << 8) | lo).reshape(n, dpose)
    delta = np.where(
        delta_zz % 2 == 0,
        delta_zz.astype(np.int32) // 2,
        -(delta_zz.astype(np.int32) // 2) - 1,
    ).astype(np.int16)
    q = np.empty_like(delta, dtype=np.int32)
    q[0] = delta[0]
    for i in range(1, n):
        q[i] = q[i - 1] + delta[i]
    q = q.astype(np.uint8).astype(np.float32)
    pose = q * scales[None, :] + mins[None, :]
    return torch.from_numpy(pose)


def build_archive_with_pose(
    vendored_build_archive,
    decoder_state_dict: dict,
    latents: torch.Tensor,
    meta_dict: dict,
    stored_pose: torch.Tensor,
) -> bytes:
    """Build the vendored archive, then APPEND the additive pose section.

    ``vendored_build_archive`` is the PRISTINE ``codec.build_archive`` (we never
    edit it). The result is ``vendored_archive_bytes + encode_pose_section(...)``
    — a strictly additive grammar: a vendored-only ``parse_archive`` reads its 3
    sections and ignores the trailing pose bytes; :func:`parse_archive_with_pose`
    reads all 4. The decoder_state_dict should EXCLUDE the FiLM params ONLY if you
    want the FiLM weights elsewhere; here we include the FiLM weights in the decoder
    state dict (they ship in the decoder blob), so the wrapper rebuilds exactly."""
    base = vendored_build_archive(decoder_state_dict, latents, meta_dict)
    return base + encode_pose_section(stored_pose)


def parse_pose_section(archive_bytes: bytes, vendored_parse_archive) -> torch.Tensor | None:
    """Return the decoded stored pose from an archive's additive pose section, or
    ``None`` if the archive has no pose section (a vendored-only / FiLM-off archive).

    Strategy: re-serialize the 3 vendored sections to compute their byte length
    (the vendored ``parse_archive`` does not return offsets), then slice the
    trailing bytes. We instead read the 3 length-prefixed sections directly from
    the archive header (the vendored grammar is fixed: ``[len][blob]`` × 3), which
    is robust and offset-exact without re-encoding."""
    buf = io.BytesIO(archive_bytes)
    # Walk the 3 vendored length-prefixed sections to find the trailing offset.
    for _ in range(3):
        sec_len_bytes = buf.read(4)
        if len(sec_len_bytes) < 4:
            return None  # truncated / not a valid vendored archive
        sec_len = struct.unpack("<I", sec_len_bytes)[0]
        buf.seek(sec_len, io.SEEK_CUR)
    trailing = buf.read()
    if len(trailing) < 4 or trailing[:4] != _POSE_SECTION_MAGIC:
        return None  # no pose section (FiLM-off archive)
    # Sanity: the vendored parse still succeeds on the (additive) archive.
    _ = vendored_parse_archive  # the vendored reader ignores trailing bytes
    return decode_pose_section(trailing)


def _parse_archive_variable_or_vendored(archive_bytes: bytes, vendored_parse_archive):
    """Parse a vendored archive or a D2 variable-level decoder archive.

    The vendored parser cannot decode a variable-level decoder blob, so the D2
    receiver dispatches from metadata and reuses the vendored latent decoder. This
    keeps FiLM+D2 parse-back faithful without editing the public PR95 source.
    """
    import json

    buf = io.BytesIO(archive_bytes)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta_brotli = buf.read(meta_len)
    meta = json.loads(brotli.decompress(meta_brotli))
    dec_len = struct.unpack("<I", buf.read(4))[0]
    decoder_blob = buf.read(dec_len)
    lat_len = struct.unpack("<I", buf.read(4))[0]
    latents_brotli = buf.read(lat_len)

    var_meta = meta.get("variable_level_waterfill") or {}
    is_variable = bool(var_meta.get("decoder_blob_is_variable_format"))
    if not is_variable:
        return vendored_parse_archive(archive_bytes)

    from tac.losses.variable_level_codec import decode_decoder_variable
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    decoder_sd = decode_decoder_variable(decoder_blob)
    latents = codec.decode_latents(brotli.decompress(latents_brotli))
    return decoder_sd, latents, meta


@torch.inference_mode()
def inflate_film_decoder(
    archive_bytes: bytes,
    vendored_parse_archive,
    vendored_decoder_cls,
    *,
    film_hidden: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """Inflate a FiLM archive to raw pair frames — the export-first ROUND-TRIP proof.

    Parses the vendored 3 sections (decoder + latents + meta) AND the additive pose
    section, rebuilds the vendored decoder, wraps it with a FiLM whose weights come
    from the decoder state dict (the FiLM params ship in the decoder blob under the
    ``pose_film.*`` keys), sets the stored pose from the parsed section, and renders
    every pair conditioned on its stored pose. Returns ``(n_pairs, 2, 3, 384, 512)``
    float in [0, 255]. Pure-torch/numpy, no challenge deps (numpy-portable inflate).
    """
    decoder_sd, latents, meta = _parse_archive_variable_or_vendored(
        archive_bytes, vendored_parse_archive
    )
    pose = parse_pose_section(archive_bytes, vendored_parse_archive)
    n_pairs = int(meta["n_pairs"])
    # Split the parsed state dict into vendored-decoder keys and FiLM keys.
    film_sd = {
        k[len("pose_film.") :]: v
        for k, v in decoder_sd.items()
        if k.startswith("pose_film.")
    }
    dec_sd = {k: v for k, v in decoder_sd.items() if not k.startswith("pose_film.")}
    decoder = vendored_decoder_cls(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(dec_sd)
    decoder.eval()
    wrapper = PoseFiLMHNeRVWrapper(
        decoder, n_pairs=n_pairs, pose_dim=_POSE_DIM, film_hidden=film_hidden
    ).to(device)
    if film_sd:
        wrapper.pose_film.load_state_dict(film_sd)
    if pose is not None:
        wrapper.set_stored_pose(pose.to(device))
    latents = latents.to(device)
    out = []
    for i in range(0, n_pairs, 16):
        j = min(i + 16, n_pairs)
        idx = torch.arange(i, j, device=device)
        out.append(wrapper(latents[i:j], idx))
    return torch.cat(out, dim=0)


def stored_pose_bytes(stored_pose: torch.Tensor) -> int:
    """The REAL byte cost of the additive pose section (encode then measure).

    Unlike a fp16-per-scalar estimate, this is the brotli-coded count the archive
    actually pays (the honest export cost — Catalog #304: the encode IS the
    empirical bit-spend proof)."""
    return len(encode_pose_section(stored_pose))


__all__ = [
    "PoseFiLMHNeRVWrapper",
    "_FiLMEvalDecoder",
    "_PoseFiLM",
    "build_archive_with_pose",
    "decode_pose_section",
    "encode_pose_section",
    "inflate_film_decoder",
    "parse_pose_section",
    "stored_pose_bytes",
    "wrapper_sd_to_archive_decoder_sd",
]
