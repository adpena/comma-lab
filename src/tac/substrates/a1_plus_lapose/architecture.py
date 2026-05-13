"""Architecture for the A1 + LAPose composition substrate.

The base renderer is A1's PR101-derived HNeRVDecoder (frozen). The
composition layer is a small per-pair RGB residual head, ENABLED only at
LAPose-atom-selected hard pairs.

The residual produces a low-rank correction to A1's predicted RGB at the
foveal region of the camera frame (the dashcam exploit). Pose-axis foveal
priors come from ``geometry_priors=foveal_*`` rows of the LAPose atom
manifest.

Per CLAUDE.md "HNeRV parity discipline" lessons L1, L4, L5, L7:
* L1 score-aware substrate trains against contest video pixels.
* L4 inflate ≤ 100 LOC; this module is training-time only.
* L5 architecture is the FULL renderer (RGB out from selected pairs).
* L7 substrate-engineering LOC budget (~600-900 trainer + helpers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


# A1 wire-format constants (canonical A1 anchor on contest CPU GHA Linux
# x86_64 + contest CUDA T4 — 178,162 bytes, sha256
# 8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243).
# These are NOT redeclared here as architecture; they are the immutable
# base substrate's contract.  See
# experiments/results/track4_sg_a1_t178000_20260509/submission_dir/inflate.py
# for the canonical reference parser.  The longer SHA ``87ec7ca5...`` cited
# in the council memo refers to the unfinetuned PR101 source; the operator
# directive uses the t178000 fork as the BUILD anchor.
A1_EVAL_H, A1_EVAL_W = 384, 512
A1_CAMERA_H, A1_CAMERA_W = 874, 1164
A1_N_PAIRS = 600
A1_LATENT_DIM = 28
A1_BASE_CHANNELS = 36


@dataclass(frozen=True)
class A1PlusLaposeConfig:
    """Composition substrate config.

    The composition is parameterized by:

    * ``residual_dim``: width of the residual head's hidden layer.
      Small (default 8) per CLAUDE.md "bolt-on size ≤ 350 LOC" + the
      3-5 KB residual budget operator-fallback D2.B.
    * ``residual_rank``: per-pair rank of the foveal residual correction
      (default 4 — produces ~3 KB Brotli-compressed residual blob at
      int8 quantization for ~64 selected pairs).
    * ``selected_pair_indices``: tuple of pair indices receiving the
      composition residual (subset of [0, A1_N_PAIRS); usually 32-96).
      Drawn from the LAPose atom manifest.
    * ``foveal_h``, ``foveal_w``: rectangular foveal patch dims at
      camera-native resolution where the residual applies. Default
      256x256 centered on (camera_h//2, camera_w//2) — the dashcam
      vanishing-point region where PoseNet's 6-DoF Lie algebra signal
      is densest (D5.C operator-fallback).
    """

    residual_dim: int = 8
    residual_rank: int = 4
    selected_pair_indices: tuple[int, ...] = field(default_factory=tuple)
    foveal_h: int = 256
    foveal_w: int = 256
    foveal_center_h: int = A1_CAMERA_H // 2
    foveal_center_w: int = A1_CAMERA_W // 2
    int8_residual_scale: float = 4.0
    """Scale factor for int8 quantization (residual lives in
    [-128/scale, 127/scale] approximately). Default 4.0 keeps the
    quantized residual in roughly [-32, 32] gray levels — well below
    the 23x MPS PoseNet drift threshold (CLAUDE.md "MPS auth eval is
    NOISE")."""


class PerPairResidualHead(nn.Module):
    """A per-pair low-rank RGB residual head.

    Each selected pair index gets its own learned (U, V) rank-K factor.
    The residual is ``U @ V^T`` projected to (3, foveal_h, foveal_w).
    This is intentionally simple — the budget is ~3-5 KB after int8
    quant + Brotli, not 100 KB.

    The output is ADDITIVE to A1's base RGB at the foveal patch. The
    base outside the foveal patch is unchanged.

    Per CLAUDE.md "bolt-on size ≤ 350 LOC" + HNeRV parity discipline
    lesson 12 (single-LOC-per-LOC review discipline).
    """

    def __init__(self, cfg: A1PlusLaposeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_selected = max(1, len(cfg.selected_pair_indices))
        # Parameters per pair: U (rank, 3*foveal_h), V (rank, foveal_w)
        # Two factors per RGB frame (left + right of the pair = 2 frames).
        # Per Selfcomp's seat: small initialization to keep the residual
        # within a few gray levels until trained.
        self.U = nn.Parameter(
            0.02
            * torch.randn(
                self.num_selected, 2, cfg.residual_rank, 3 * cfg.foveal_h
            )
        )
        self.V = nn.Parameter(
            0.02
            * torch.randn(self.num_selected, 2, cfg.residual_rank, cfg.foveal_w)
        )
        # Mapping pair_index -> slot in the parameter table.
        pair_to_slot = {
            int(p): i for i, p in enumerate(cfg.selected_pair_indices)
        }
        # Stored as a python dict; the trainer indexes by pair id.
        self._pair_to_slot = pair_to_slot

    def residual_chw(
        self, pair_index: int, frame_index: int
    ) -> torch.Tensor:
        """Return the per-frame residual at (3, foveal_h, foveal_w).

        ``frame_index`` is 0 (first frame of pair) or 1 (second frame).
        Returns zeros (no-op) for pair indices not in the selection.
        """
        if pair_index not in self._pair_to_slot:
            device = self.U.device
            return torch.zeros(
                3, self.cfg.foveal_h, self.cfg.foveal_w, device=device
            )
        slot = self._pair_to_slot[pair_index]
        # U[slot, frame] : (rank, 3 * fov_h)
        # V[slot, frame] : (rank, fov_w)
        # Outer product over rank, then sum -> (3 * fov_h, fov_w)
        u_chw = self.U[slot, frame_index]
        v_chw = self.V[slot, frame_index]
        # Result: (3, fov_h, fov_w)
        return torch.einsum("kp,kw->pw", u_chw, v_chw).view(
            3, self.cfg.foveal_h, self.cfg.foveal_w
        )

    def selected_indices(self) -> tuple[int, ...]:
        return tuple(self.cfg.selected_pair_indices)

    def total_int8_bytes(self) -> int:
        """Closed-form bound on the post-int8-quant residual size in bytes.

        Each slot has (2 frames) x (rank * (3*fov_h + fov_w)) int8 params.
        + 2 bytes header for selected pair count + 2 bytes per index.
        """
        per_frame = self.cfg.residual_rank * (
            3 * self.cfg.foveal_h + self.cfg.foveal_w
        )
        param_bytes = self.num_selected * 2 * per_frame
        index_bytes = 2 + 2 * self.num_selected
        return int(param_bytes + index_bytes)


def parse_lapose_atom_indices(
    manifest_dict: dict[str, Any], max_atoms: int | None = None
) -> tuple[int, ...]:
    """Extract pair indices from a LAPose motion-atom manifest.

    The canonical LAPose motion-atom manifest schema (see
    ``tac.analysis.lapose_motion_atoms``) has rows like::

        {
            "atom_id": "lapose_motion_pair:75",
            "byte_delta": 72,
            "hard_pair_support": [75],
            "expected_pose_dist_delta": -1.2e-05,
            ...
        }

    The pair index is the trailing integer of ``atom_id`` and is also
    surfaced as ``hard_pair_support[0]`` (the canonical hard pair).

    Returns a deduplicated tuple in ascending order. Caps at
    ``max_atoms`` if supplied.
    """
    atoms = manifest_dict.get("atoms")
    if atoms is None and isinstance(manifest_dict.get("atom_ledger"), dict):
        atoms = manifest_dict["atom_ledger"].get("rows", [])
    if not isinstance(atoms, list):
        return ()
    out: set[int] = set()
    for row in atoms:
        if not isinstance(row, dict):
            continue
        support = row.get("hard_pair_support")
        if isinstance(support, list) and support:
            try:
                out.add(int(support[0]))
                continue
            except (TypeError, ValueError):
                pass
        atom_id = str(row.get("atom_id") or "")
        if ":" in atom_id:
            tail = atom_id.split(":")[-1]
            try:
                out.add(int(tail))
            except ValueError:
                continue
    ordered = sorted(p for p in out if 0 <= p < A1_N_PAIRS)
    if max_atoms is not None:
        ordered = ordered[: int(max_atoms)]
    return tuple(ordered)


__all__ = [
    "A1_BASE_CHANNELS",
    "A1_CAMERA_H",
    "A1_CAMERA_W",
    "A1_EVAL_H",
    "A1_EVAL_W",
    "A1_LATENT_DIM",
    "A1_N_PAIRS",
    "A1PlusLaposeConfig",
    "PerPairResidualHead",
    "parse_lapose_atom_indices",
]
