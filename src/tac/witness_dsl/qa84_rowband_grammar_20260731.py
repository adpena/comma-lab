"""ddm_b2b — QA84 variable-size cell tiling (row-band D8/D16 foveation grammar).

census §4.2 / QA84: the uniform 16x16 token lattice spends equal DOF everywhere, but the
op1 foveation gate PASSED at the pre-registered >=50% criterion — 72.1-72.7% of flip-prone
mass lives in render rows ~160-240. This grammar realizes the SEPARABLE approximation of the
scorer-geometry-optimal cell field D ~ (flip-density)^(-alpha): a FINE (D8) base grid whose
BULK rows are TIED in coarse_factor x coarse_factor blocks (=> D16-effective), while the
annulus band rows stay FREE at D8. The tie is a deterministic GATHER (each bulk cell copies
its block's top-left representative), differentiable (gradient flows to the representative,
the other cells get none — like the cell_mask), backend-agnostic (numpy + MLX share one
representative-index map).

BYTE-CLOSE reuses the SHIPPED SMEVR coder: the tied field has identical values inside each
bulk block, so SMEVR's left/up + temporal contexts code them as ~free zero-delta runs — the
tie's byte savings materialize through the real coder without any kept-cell restriction. Only
a few bytes of band-spec side-info (row_lo, row_hi, coarse_factor) are added.

INSTANCE scoping (census): the gr1 nested-rungs DOMINATED receipt is scoped to the solved-token
post-hoc substrate; the trained-renderer FROM-BIRTH formulation (this grammar) is uncovered.
The Hilbert-race receipt stands: raster is the wire order (this grammar does NOT change scan
order). Prior art: HEVC CTU splitting = lessons-only.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED. Grammar/byte-close only; score_claim=False.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

#: render-frame rows carrying 72.1-72.7% of flip-prone mass (op1 foveation gate, QA74 typing).
FLIP_BAND_RENDER_ROWS: tuple[int, int] = (160, 240)


@dataclass(frozen=True)
class RowBandGrammar:
    """A row-band variable-cell tiling on a fine (D8) token grid.

    ``fine_gh``/``fine_gw``: fine grid dims (render//fine_downsample).
    ``band_row_lo``/``band_row_hi``: fine-grid row range [lo, hi) kept FREE at the fine pitch.
    ``coarse_factor``: bulk cells outside the band are tied in coarse_factor x coarse_factor
    blocks (=> coarse_factor*fine_downsample effective pitch). 2 => D8-band/D16-bulk.
    """

    fine_gh: int
    fine_gw: int
    band_row_lo: int
    band_row_hi: int
    coarse_factor: int = 2
    code_width: int = 4

    def __post_init__(self) -> None:
        if not (0 <= self.band_row_lo < self.band_row_hi <= self.fine_gh):
            raise ValueError(f"band rows [{self.band_row_lo},{self.band_row_hi}) out of "
                             f"[0,{self.fine_gh})")
        if self.coarse_factor < 1:
            raise ValueError("coarse_factor must be >= 1")
        # bulk regions (above/below the band) and the fine width must tile the coarse block
        # cleanly, else the tie would straddle the band boundary (fail-closed, never-invent).
        if (self.band_row_lo % self.coarse_factor or self.band_row_hi % self.coarse_factor
                or self.fine_gh % self.coarse_factor or self.fine_gw % self.coarse_factor):
            raise ValueError(
                f"band bounds/grid must be multiples of coarse_factor={self.coarse_factor} "
                f"(band=[{self.band_row_lo},{self.band_row_hi}), grid={self.fine_gh}x{self.fine_gw})")

    @classmethod
    def from_render_rows(cls, render_row_lo: int, render_row_hi: int, *, fine_downsample: int,
                         render_h: int, render_w: int, coarse_factor: int = 2,
                         code_width: int = 4) -> "RowBandGrammar":
        """Build from RENDER-frame row bounds (e.g. FLIP_BAND_RENDER_ROWS), snapping the band
        to coarse_factor-aligned fine-grid rows (outward = never shrink the fovea)."""
        fine_gh, fine_gw = render_h // fine_downsample, render_w // fine_downsample
        lo = (render_row_lo // fine_downsample) // coarse_factor * coarse_factor
        hi = -(-(render_row_hi // fine_downsample) // coarse_factor) * coarse_factor  # ceil-align
        return cls(fine_gh, fine_gw, max(0, lo), min(fine_gh, hi), coarse_factor, code_width)

    def _is_band_row(self, r: int) -> bool:
        return self.band_row_lo <= r < self.band_row_hi

    def representative_flat_index(self) -> np.ndarray:
        """(fine_gh*fine_gw,) int: each fine cell -> the flat index it copies from (itself if
        band or coarse_factor==1; else its block's top-left)."""
        idx = np.empty((self.fine_gh, self.fine_gw), dtype=np.int64)
        cf = self.coarse_factor
        for r in range(self.fine_gh):
            for c in range(self.fine_gw):
                if self._is_band_row(r) or cf == 1:
                    rr, cc = r, c
                else:
                    rr, cc = (r // cf) * cf, (c // cf) * cf
                idx[r, c] = rr * self.fine_gw + cc
        return idx.ravel()

    def independent_cells(self) -> int:
        """Count of DISTINCT representatives (the independent DOF = counted token cells)."""
        return int(np.unique(self.representative_flat_index()).size)

    def apply_tie_np(self, field: np.ndarray) -> np.ndarray:
        """Tie a numpy token field (..., fine_gh, fine_gw, c) -> same shape, bulk blocks shared."""
        rep = self.representative_flat_index()
        lead = field.shape[:-3]
        c = field.shape[-1]
        flat = field.reshape(*lead, self.fine_gh * self.fine_gw, c)
        tied = flat[..., rep, :]
        return tied.reshape(*lead, self.fine_gh, self.fine_gw, c)

    def apply_tie_mx(self, mx, field):
        """Tie an MLX token field (..., fine_gh, fine_gw, c). Same gather as ``apply_tie_np``
        (bit-identical representative map) so render (MLX) and byte-close (numpy) agree."""
        rep = mx.array(self.representative_flat_index())
        lead = tuple(field.shape[:-3])
        c = field.shape[-1]
        flat = field.reshape(*lead, self.fine_gh * self.fine_gw, c)
        tied = mx.take(flat, rep, axis=-2)
        return tied.reshape(*lead, self.fine_gh, self.fine_gw, c)

    def band_spec_bytes(self) -> int:
        """COUNTED side-info: the band grammar spec the decoder needs to expand the tie."""
        return len(self.spec_json().encode())

    def spec_json(self) -> str:
        return json.dumps({
            "grammar": "rowband_d_foveation.v1", "fine_gh": self.fine_gh, "fine_gw": self.fine_gw,
            "band_row_lo": self.band_row_lo, "band_row_hi": self.band_row_hi,
            "coarse_factor": self.coarse_factor, "code_width": self.code_width,
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_spec_json(cls, s: str) -> "RowBandGrammar":
        d = json.loads(s)
        return cls(int(d["fine_gh"]), int(d["fine_gw"]), int(d["band_row_lo"]),
                   int(d["band_row_hi"]), int(d["coarse_factor"]), int(d["code_width"]))

    def dof_summary(self) -> dict[str, int]:
        """Scorer-free comparison vs the uniform lattices (independent counted cells)."""
        uniform_fine = self.fine_gh * self.fine_gw
        uniform_coarse = (self.fine_gh // self.coarse_factor) * (self.fine_gw // self.coarse_factor)
        return {"rowband_cells": self.independent_cells(),
                "uniform_fine_cells": int(uniform_fine),
                "uniform_coarse_cells": int(uniform_coarse),
                "band_spec_bytes": self.band_spec_bytes()}


def default_flip_band_grammar(fine_downsample: int = 8, render_h: int = 384, render_w: int = 512,
                              coarse_factor: int = 2, code_width: int = 4) -> RowBandGrammar:
    """The pre-registered row-band arm: D8 base, D16 bulk, fovea on the op1 flip band."""
    return RowBandGrammar.from_render_rows(
        *FLIP_BAND_RENDER_ROWS, fine_downsample=fine_downsample, render_h=render_h,
        render_w=render_w, coarse_factor=coarse_factor, code_width=code_width)
