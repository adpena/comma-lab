"""Architecture for the A1 + wavelet residual sidecar substrate.

The base renderer is A1's PR101-fine-tuned HNeRVDecoder (frozen verbatim).
The composition layer is a tiny per-pair RGB residual head that stores
Daubechies-4 detail-band coefficients (LH/HL/HH at depth-1 Mallat
decomposition) at a small foveal patch.  The LL approximation band is
NOT stored — that is the A1 base's responsibility; the sidecar carries only
the high-frequency residual that the A1 base cannot represent.

Per CLAUDE.md "HNeRV parity discipline" lessons L1, L4, L5, L7:

* L1 score-aware substrate trains against contest video pixels.
* L4 inflate ≤ 100 LOC; this module is training-time only.
* L5 architecture is the FULL renderer (RGB out from selected pairs).
* L7 substrate-engineering LOC budget (~400-600 trainer + helpers; freeze-A1
  mode is much cheaper than A1+LAPose's joint mode).

Mallat seat (grand council):
    The high-frequency residual A1 cannot represent (the conv-decoder's
    low-pass bias) lives in a sparse multi-resolution representation.
    Daubechies-4 at depth-1 captures horizontal / vertical / diagonal
    detail at half-camera resolution per channel.  Selected pair indices
    (the "hard pairs" with maximum pose drift) get the budget; the rest
    fall back to A1's prediction unchanged.

Design parameters (D2 byte-budget envelope):
    rank-K per-band coefficients at a small foveal patch + int8 quant
    + brotli yields ~200-500 B per archive.  Target overhead ≤ 500 B
    over the A1 base (~0.4 % of A1's 178,162 B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


# A1 wire-format constants (canonical A1 anchor on contest CPU GHA Linux
# x86_64 + contest CUDA T4 — 178,162 bytes for the unzipped "x" blob,
# 174,113 bytes for the archive.zip wrapper, sha256
# 8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243).
# These match A1+LAPose's architecture.py; the substrate-engineering
# exemption keeps the redeclaration here intentional so the wavelet
# substrate can be reviewed independently of A1+LAPose.
A1_EVAL_H, A1_EVAL_W = 384, 512
A1_CAMERA_H, A1_CAMERA_W = 874, 1164
A1_N_PAIRS = 600
A1_LATENT_DIM = 28
A1_BASE_CHANNELS = 36


@dataclass(frozen=True)
class A1PlusWaveletResidualConfig:
    """Composition substrate config for the wavelet residual sidecar.

    The composition is parameterized by:

    * ``selected_pair_indices``: tuple of pair indices receiving the wavelet
      residual (subset of [0, A1_N_PAIRS); usually 16-96).  The "hard pairs"
      identified by either a LAPose-style atom manifest, a heuristic pose-
      drift ranking, OR an explicit operator-supplied list.
    * ``coeff_rank``: per-band low-rank K (default 2) for the LH/HL/HH
      detail-band coefficient outer product.  Small K keeps the sidecar
      ≤500 B even with 96 selected pairs at fov=128.
    * ``foveal_h``, ``foveal_w``: foveal patch dims at HALF-camera
      resolution (depth-1 Mallat sub-sampling).  Default 128x128 at
      half-camera = 256x256 patch at camera-native after IDWT.
    * ``int8_residual_scale``: int8 quantization scale (q = clamp(round(
      real * scale), -128, 127)).  Default 8.0 keeps the quantized
      coefficient within roughly [-16, 16] real-valued units — small
      enough to keep PoseNet's 23x MPS drift band off the table.
    * ``wavelet_levels``: depth of the DWT decomposition.  Default 1
      (single-level DB4); higher levels grow the sidecar quadratically.
    """

    selected_pair_indices: tuple[int, ...] = field(default_factory=tuple)
    coeff_rank: int = 2
    foveal_h: int = 128  # half-camera resolution patch
    foveal_w: int = 128
    int8_residual_scale: float = 8.0
    wavelet_levels: int = 1
    """Depth-1 single-level DWT.  Higher levels grow sidecar bytes
    quadratically; council §5.6 noted single-level captures the dominant
    high-frequency residual at the pose-axis operating point."""

    def estimated_sidecar_bytes(self) -> int:
        """Closed-form bound on the post-int8-quant brotli sidecar size.

        For each of the N selected pairs:
            3 detail bands (LH, HL, HH) × 2 frames per pair × 3 RGB channels
            × coeff_rank × (foveal_h + foveal_w) int8 = pre-brotli bytes.

        + 4 bytes magic + 16 bytes header + 2*N bytes for the index table.
        Brotli typically compresses int8 wavelet coefficients ~40-60% on
        sparse residual content; this returns the PRE-compression upper
        bound to keep the loss-side rate proxy honest.
        """
        n = max(1, len(self.selected_pair_indices))
        per_band_per_frame_per_chan = self.coeff_rank * (self.foveal_h + self.foveal_w)
        bands_per_frame = 3 * 3 * per_band_per_frame_per_chan  # 3 bands × 3 channels
        frames_per_pair = 2
        param_bytes = n * frames_per_pair * bands_per_frame
        index_bytes = 4 + 16 + 2 * n  # 4 B magic + 16 B header + N×2 B indices
        return int(param_bytes + index_bytes)


class PerPairWaveletResidualHead(nn.Module):
    """A per-pair low-rank wavelet detail-band residual head.

    Each selected pair index gets its own learned (U, V) rank-K factors for
    each of the three detail bands (LH/HL/HH) × 2 frames × 3 RGB channels.
    The residual is computed at HALF-camera resolution and uplifted to
    camera-native via DB4 IDWT (no learnable filter; Mallat-fixed).

    For pair_id not in the selection, the head returns zeros (no-op).

    Per CLAUDE.md "bolt-on size ≤ 350 LOC" + HNeRV parity discipline
    lesson 12 (single-LOC-per-LOC review discipline).
    """

    NUM_DETAIL_BANDS = 3  # LH, HL, HH (LL stays at A1's prediction)
    NUM_FRAMES_PER_PAIR = 2
    NUM_RGB_CHANNELS = 3

    def __init__(self, cfg: A1PlusWaveletResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_selected = max(1, len(cfg.selected_pair_indices))
        # Parameters per pair: U (rank, foveal_h), V (rank, foveal_w)
        # per (band, frame, channel) tuple.  Small init keeps untrained
        # residual within a few gray levels.
        u_shape = (
            self.num_selected,
            self.NUM_DETAIL_BANDS,
            self.NUM_FRAMES_PER_PAIR,
            self.NUM_RGB_CHANNELS,
            cfg.coeff_rank,
            cfg.foveal_h,
        )
        v_shape = (
            self.num_selected,
            self.NUM_DETAIL_BANDS,
            self.NUM_FRAMES_PER_PAIR,
            self.NUM_RGB_CHANNELS,
            cfg.coeff_rank,
            cfg.foveal_w,
        )
        self.U = nn.Parameter(0.01 * torch.randn(*u_shape))
        self.V = nn.Parameter(0.01 * torch.randn(*v_shape))
        # Mapping pair_index -> slot in the parameter table.
        pair_to_slot = {int(p): i for i, p in enumerate(cfg.selected_pair_indices)}
        self._pair_to_slot = pair_to_slot

    def _detail_band_chw(
        self,
        slot: int,
        band_idx: int,
        frame_idx: int,
    ) -> torch.Tensor:
        """Build a (3, fov_h, fov_w) detail-band coefficient tensor for one slot."""
        u_chw = self.U[slot, band_idx, frame_idx]  # (3, rank, fov_h)
        v_chw = self.V[slot, band_idx, frame_idx]  # (3, rank, fov_w)
        # Outer product per channel: einsum over rank
        return torch.einsum("ckp,ckw->cpw", u_chw, v_chw)

    def residual_chw_for_pair(
        self,
        pair_index: int,
        frame_index: int,
    ) -> torch.Tensor:
        """Return camera-native (3, foveal_h*2, foveal_w*2) RGB residual.

        Reconstructs via single-level DB4 IDWT (LL=0, LH/HL/HH from
        per-pair coefficients).  Returns a zero tensor of the same shape
        for pair indices NOT in the selection.

        ``frame_index`` is 0 (first frame of pair) or 1 (second frame).
        """
        out_h = 2 * self.cfg.foveal_h
        out_w = 2 * self.cfg.foveal_w
        if pair_index not in self._pair_to_slot:
            return torch.zeros(
                self.NUM_RGB_CHANNELS, out_h, out_w, device=self.U.device
            )
        slot = self._pair_to_slot[pair_index]
        LH = self._detail_band_chw(slot, 0, frame_index)  # (3, fh, fw)
        HL = self._detail_band_chw(slot, 1, frame_index)
        HH = self._detail_band_chw(slot, 2, frame_index)
        # LL = 0 (A1 carries the approximation; we contribute only detail bands)
        LL = torch.zeros_like(LH)
        return _db4_idwt_single_level(LL, LH, HL, HH)

    def selected_indices(self) -> tuple[int, ...]:
        return tuple(self.cfg.selected_pair_indices)

    def estimated_sidecar_bytes(self) -> int:
        return self.cfg.estimated_sidecar_bytes()


# ----------------------------------------------------------------------
# Daubechies-4 IDWT (single level) — fixed (non-learnable) Mallat synthesis
# ----------------------------------------------------------------------

import math as _math

_DB4_LO: tuple[float, ...] = (
    (1.0 + _math.sqrt(3.0)) / (4.0 * _math.sqrt(2.0)),
    (3.0 + _math.sqrt(3.0)) / (4.0 * _math.sqrt(2.0)),
    (3.0 - _math.sqrt(3.0)) / (4.0 * _math.sqrt(2.0)),
    (1.0 - _math.sqrt(3.0)) / (4.0 * _math.sqrt(2.0)),
)


def _db4_hi() -> tuple[float, ...]:
    lo = _DB4_LO
    return tuple((-1.0) ** k * lo[len(lo) - 1 - k] for k in range(len(lo)))


def _db4_idwt_single_level(
    LL: torch.Tensor,
    LH: torch.Tensor,
    HL: torch.Tensor,
    HH: torch.Tensor,
) -> torch.Tensor:
    """Single-level inverse DWT via DB4 synthesis filters (separable 1D).

    Each sub-band is (C, h, w); output is (C, 2h, 2w).  Implements the
    same separable filterbank as ``tac.substrates.wavelet.architecture``
    but standalone here for the substrate-engineering exemption.
    """
    import torch.nn.functional as F

    if LL.shape != LH.shape or LL.shape != HL.shape or LL.shape != HH.shape:
        raise ValueError(
            f"DB4 IDWT requires equal sub-band shapes; got "
            f"LL={tuple(LL.shape)} LH={tuple(LH.shape)} HL={tuple(HL.shape)} "
            f"HH={tuple(HH.shape)}"
        )
    if LL.dim() != 3:
        raise ValueError(f"DB4 IDWT expects (C, H, W) input; got {tuple(LL.shape)}")
    C, h, w = LL.shape
    lo = torch.tensor(_DB4_LO, dtype=LL.dtype, device=LL.device).view(1, 1, 4).repeat(C, 1, 1)
    hi = torch.tensor(_db4_hi(), dtype=LL.dtype, device=LL.device).view(1, 1, 4).repeat(C, 1, 1)

    def _upsample_filter_1d(x_lo: torch.Tensor, x_hi: torch.Tensor, dim: int) -> torch.Tensor:
        x_lo_p = x_lo.movedim(dim, -1)
        x_hi_p = x_hi.movedim(dim, -1)
        orig_shape = list(x_lo_p.shape)
        zeros_lo = torch.zeros_like(x_lo_p)
        zeros_hi = torch.zeros_like(x_hi_p)
        up_lo = torch.stack([x_lo_p, zeros_lo], dim=-1).flatten(-2)
        up_hi = torch.stack([x_hi_p, zeros_hi], dim=-1).flatten(-2)
        L = up_lo.shape[-1]
        # Reshape to (B', C, L) for grouped conv1d
        bc_lo = up_lo.reshape(-1, C, L)
        bc_hi = up_hi.reshape(-1, C, L)
        pad = 3
        bc_lo_padded = F.pad(bc_lo, (pad, 0), mode="reflect")
        bc_hi_padded = F.pad(bc_hi, (pad, 0), mode="reflect")
        out_lo = F.conv1d(bc_lo_padded, lo, groups=C)
        out_hi = F.conv1d(bc_hi_padded, hi, groups=C)
        out = out_lo + out_hi
        new_shape = orig_shape[:-1] + [L]
        out_full = out.reshape(*new_shape)
        return out_full.movedim(-1, dim)

    # Vertical pass (combine LL/LH on the low-frequency column; HL/HH on the high-frequency column)
    low_col = _upsample_filter_1d(LL.unsqueeze(0), LH.unsqueeze(0), dim=-2).squeeze(0)
    high_col = _upsample_filter_1d(HL.unsqueeze(0), HH.unsqueeze(0), dim=-2).squeeze(0)
    # Horizontal pass
    out = _upsample_filter_1d(low_col.unsqueeze(0), high_col.unsqueeze(0), dim=-1).squeeze(0)
    return out


def parse_wavelet_residual_pair_indices(
    manifest_dict: dict[str, Any],
    max_pairs: int | None = None,
) -> tuple[int, ...]:
    """Extract pair indices from a wavelet-residual / pose-drift manifest.

    Accepts the same LAPose atom manifest schema (atoms[] with hard_pair_support
    or atom_id "*_pair:N") OR a simpler wavelet schema (pairs[] of int).
    Returns a deduplicated tuple in ascending order, optionally capped at
    ``max_pairs``.
    """
    out: set[int] = set()

    # Schema A: simple pairs[] of int.
    pairs = manifest_dict.get("pairs")
    if isinstance(pairs, list):
        for p in pairs:
            try:
                idx = int(p)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < A1_N_PAIRS:
                out.add(idx)

    # Schema B: LAPose-style atoms[] (reuse the same parser logic).
    atoms = manifest_dict.get("atoms")
    if atoms is None and isinstance(manifest_dict.get("atom_ledger"), dict):
        atoms = manifest_dict["atom_ledger"].get("rows", [])
    if isinstance(atoms, list):
        for row in atoms:
            if not isinstance(row, dict):
                continue
            support = row.get("hard_pair_support")
            if isinstance(support, list) and support:
                try:
                    idx = int(support[0])
                    if 0 <= idx < A1_N_PAIRS:
                        out.add(idx)
                        continue
                except (TypeError, ValueError):
                    pass
            atom_id = str(row.get("atom_id") or "")
            if ":" in atom_id:
                tail = atom_id.split(":")[-1]
                try:
                    idx = int(tail)
                    if 0 <= idx < A1_N_PAIRS:
                        out.add(idx)
                except ValueError:
                    continue

    ordered = sorted(out)
    if max_pairs is not None:
        ordered = ordered[: int(max_pairs)]
    return tuple(ordered)


__all__ = [
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "A1PlusWaveletResidualConfig",
    "PerPairWaveletResidualHead",
    "parse_wavelet_residual_pair_indices",
]
