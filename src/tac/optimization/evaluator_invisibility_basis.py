# SPDX-License-Identifier: MIT
"""Evaluator invisibility basis (task #47) — the certified free-byte basis.

The contest objective is the evaluator quotient
``100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489``.  The joint scorer
map is

    M(frame_pair) = (SegNet . resize . slice_frame1) (frame_pair)        [d_seg]
                  + (PoseNet . yuv6 . resize) (frame_pair)               [d_pose]

Both scorer heads share an identical FIRST preprocessing step: a fixed bilinear
``F.interpolate((874, 1164) -> (384, 512), align_corners=False)`` applied to the
camera frame(s) (``upstream/modules.py:73`` for PoseNet, ``:109`` for SegNet, via
``upstream/frame_utils.py:11/13``).  That resize is a fixed LOW-RANK LINEAR
projection ``R``; its null space is, by construction, the set of camera-pixel
perturbations that produce a BIT-IDENTICAL scorer input — exactly invisible to
BOTH heads, at amplitude unlimited up to uint8 clipping.

This module DERIVES that null space in closed form (NOT Monte-Carlo sampled — the
sister ``tac.xray.bilinear_resize_nullspace`` is the estimator; this is the
certification-grade derivation) and exposes it as a queryable, two-TIER basis:

  TIER 1 — CERTIFIED EXACT.  Closed-form null directions of the deterministic
  linear preprocessing.  A perturbation IN tier 1, pushed through the REAL scorer
  preprocessing, yields a bit-identical scorer input (residual == 0.0).  Two
  sub-families:

    1a. RESIZE ZERO-WEIGHT PIXELS (both heads, camera-pixel domain) — the input
        rows/cols that the downsampling drops entirely (total interpolate weight
        exactly 0).  Single-pixel, axis-aligned, single-channel, amplitude
        unlimited up to clipping.  This is the robustly-certifiable primary basis.

    1b. RESIZE FULL NULL SPACE (both heads) — the complete ``ker(R)``.  Its
        DIMENSION is closed-form exact (camera_pixels - rank(R)); the zero-weight
        pixels (1a) are an axis-aligned SUBSET of it.

  Plus the FRAME0 corollary: SegNet reads only ``x[:, -1, ...]`` (frame1), so the
  ENTIRE frame0 is SegNet-invisible; frame0's PoseNet-visibility is again gated by
  the same resize, so frame0's resize zero-weight pixels are tier-1 invisible to
  BOTH heads, and ALL of frame0 is tier-1 invisible to SegNet.

  TIER 2 — MEASURED LOW-SENSITIVITY.  The nonlinear part (the scorer networks
  after preprocessing) has no exact null space, but the #36 atlas measured the
  per-pair joint cone / pose-Jacobian / seg-margin fields.  Tier 2 summarises
  those into low-singular-direction budgets PER PAIR / REGION, each tagged with a
  ``measurement_scope`` (Catalog #385 discipline).  MEASURED is NOT CERTIFIED; the
  two tiers are kept in SEPARATE schema records so a consumer can never confuse a
  measured budget for a certified zero.

Evidence grade: TIER 1 is ``mathematical-derivation`` (the residual==0.0 proof is
exact, hardware-independent).  TIER 2 is ``[macOS-CPU advisory]`` (the atlas's
scorer forwards).  No score claim; no promotion; no dispatch authority — this is a
budget surface that downstream actuators (#46 waterfiller, PR110++ atom generator)
consume.

CLAUDE.md compliance:
  - ``planning_only_no_score_claim`` / ``promotable=false``
  - ``no_mps_authoritative`` (tier 2 atlas is cpu_torch; tier 1 is pure linear algebra)
  - ``no_tmp_paths`` (durable SSD artifacts)

Cross-references
----------------
- ``upstream/modules.py:70-74`` (PoseNet preprocess: resize + yuv6),
  ``:107-109`` (SegNet preprocess: slice frame1 + resize)
- ``upstream/frame_utils.py:11`` camera_size=(1164,874), ``:13``
  segnet_model_input_size=(512,384), ``:51`` rgb_to_yuv6
- ``tac.xray.bilinear_resize_nullspace`` (the Monte-Carlo estimator this certifies)
- ``tac.optimization.frame1_joint_safe_cone`` (#35; tier-2's per-pair source)
- ``tac.optimization.evaluator_response_atlas`` (#36; tier-2's 600-pair index)
- ``tac.null_space_exploiter`` (the BYTE-space null basis; this is the PIXEL-space
  certified complement)
- ``.omx/research/evaluator_invisibility_basis_landed_20260610.md`` (landing memo)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Canonical contest dimensions (pinned from upstream/frame_utils.py).
# camera_size = (W, H) = (1164, 874); segnet_model_input_size = (W, H) = (512, 384)
# ---------------------------------------------------------------------------
CAMERA_H = 874
CAMERA_W = 1164
SCORER_INPUT_H = 384
SCORER_INPUT_W = 512
CONTEST_TOTAL_BYTES = 37_545_489  # the rate denominator

EVALUATOR_INVISIBILITY_BASIS_SCHEMA = "evaluator_invisibility_basis.v1"
TIER1_EVIDENCE_GRADE = "mathematical-derivation"
TIER2_EVIDENCE_GRADE = "[macOS-CPU advisory]"

# A bilinear-resize input pixel is "zero-weight" when its total contribution to
# EVERY output pixel is below this tolerance.  The derivation below shows the
# dropped pixels carry EXACTLY 0 weight (float64), so this tolerance only guards
# against fp roundoff; the certification test uses an exact (residual==0.0) check.
ZERO_WEIGHT_TOL = 1e-12


class EvaluatorInvisibilityBasisError(ValueError):
    """Raised when invisibility-basis inputs are malformed or a tier-1 claim
    cannot be certified (fail-closed: a non-certifiable direction never enters
    tier 1)."""


# ---------------------------------------------------------------------------
# Closed-form derivation of the bilinear-resize operator's separable structure.
# F.interpolate(bilinear, align_corners=False) is SEPARABLE: R = R_h (x) R_w.
# We derive each 1D operator's exact impulse response (a fixed deterministic
# matrix) and read off the dropped (zero-weight) input indices + the rank.
# ---------------------------------------------------------------------------
def _resize_1d_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Exact 1D bilinear-resize matrix ``M`` (shape ``(n_out, n_in)``) such that
    ``y = M @ x`` reproduces ``F.interpolate(x, size=n_out, mode='bilinear',
    align_corners=False)`` along one axis.

    Derived in closed form from the align_corners=False sampling rule
    (PyTorch / OpenCV convention): output sample ``i`` maps to input coordinate
    ``src = (i + 0.5) * n_in / n_out - 0.5``; the two neighbours
    ``floor(src), floor(src)+1`` (clamped to ``[0, n_in-1]``) receive weights
    ``(1 - frac)`` and ``frac`` where ``frac = src - floor(src)``.

    This is a DERIVATION, not a probe: no torch call, no sampling — the matrix is
    the exact algebraic kernel.  (A torch round-trip is used only inside the
    certification test, to prove this matrix == ``F.interpolate``.)
    """
    if n_in <= 0 or n_out <= 0:
        raise EvaluatorInvisibilityBasisError("n_in/n_out must be positive")
    if n_out > n_in:
        raise EvaluatorInvisibilityBasisError(
            "this derivation is for downsampling (n_out <= n_in); the contest "
            "resize is 874->384 / 1164->512"
        )
    scale = n_in / n_out
    out_idx = np.arange(n_out, dtype=np.float64)
    src = (out_idx + 0.5) * scale - 0.5
    # align_corners=False clamps source coordinates into [0, n_in-1].
    src_clamped = np.clip(src, 0.0, n_in - 1)
    lo = np.floor(src_clamped).astype(np.int64)
    frac = src_clamped - lo
    hi = np.clip(lo + 1, 0, n_in - 1)
    M = np.zeros((n_out, n_in), dtype=np.float64)
    rows = np.arange(n_out)
    M[rows, lo] += (1.0 - frac)
    M[rows, hi] += frac
    return M


@dataclass(frozen=True)
class ResizeKernelDerivation:
    """The closed-form structure of one axis of the contest resize operator.

    ``zero_weight_indices`` are the input positions the downsample DROPS (total
    output weight exactly 0): any perturbation there is exactly invisible to the
    resize output.  ``rank`` is the exact rank of the 1D operator (full-rank
    downsample => rank == n_out).
    """

    axis: str  # "h" or "w"
    n_in: int
    n_out: int
    zero_weight_indices: tuple[int, ...]
    rank: int

    @property
    def n_zero_weight(self) -> int:
        return len(self.zero_weight_indices)

    @property
    def zero_weight_fraction(self) -> float:
        return self.n_zero_weight / self.n_in if self.n_in else 0.0


def derive_resize_kernel(axis: str, n_in: int, n_out: int) -> ResizeKernelDerivation:
    """Derive (closed form) one axis of the contest bilinear-resize operator."""
    if axis not in ("h", "w"):
        raise EvaluatorInvisibilityBasisError("axis must be 'h' or 'w'")
    M = _resize_1d_matrix(n_in, n_out)
    col_weight = M.sum(axis=0)  # total output weight each input index receives
    zero_idx = tuple(int(j) for j in np.where(col_weight < ZERO_WEIGHT_TOL)[0])
    rank = int(np.linalg.matrix_rank(M))
    return ResizeKernelDerivation(
        axis=axis,
        n_in=int(n_in),
        n_out=int(n_out),
        zero_weight_indices=zero_idx,
        rank=rank,
    )


# ---------------------------------------------------------------------------
# TIER 1 — CERTIFIED EXACT basis.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tier1ResizeNullSpace:
    """TIER-1 CERTIFIED-EXACT invisibility basis for the contest resize.

    Two derived sub-families:

    - ``zero_weight_rows`` / ``zero_weight_cols`` — the dropped camera rows/cols.
      A camera pixel ``(r, c)`` is single-pixel exactly-invisible iff ``r`` is a
      zero-weight row OR ``c`` is a zero-weight col (its bilinear weight to every
      output is 0).  Single-channel, axis-aligned, amplitude unlimited up to
      uint8 clipping.

    - ``full_null_dim`` — the DIMENSION of the complete ``ker(R)`` (camera pixels
      minus rank).  The zero-weight pixels are an axis-aligned subset.
    """

    camera_h: int
    camera_w: int
    scorer_h: int
    scorer_w: int
    h_kernel: ResizeKernelDerivation
    w_kernel: ResizeKernelDerivation
    evidence_grade: str = TIER1_EVIDENCE_GRADE
    score_claim: bool = False
    promotable: bool = False

    # ---- derived scalar summary --------------------------------------------
    @property
    def zero_weight_rows(self) -> tuple[int, ...]:
        return self.h_kernel.zero_weight_indices

    @property
    def zero_weight_cols(self) -> tuple[int, ...]:
        return self.w_kernel.zero_weight_indices

    @property
    def camera_pixels(self) -> int:
        return self.camera_h * self.camera_w

    @property
    def scorer_pixels(self) -> int:
        return self.scorer_h * self.scorer_w

    @property
    def n_zero_weight_pixels_per_channel(self) -> int:
        """Camera pixels (r,c) with r OR c a zero-weight axis index.

        ``|{r zero} x W| + |H x {c zero}| - |{r zero} x {c zero}|``."""
        nr, nc = self.h_kernel.n_zero_weight, self.w_kernel.n_zero_weight
        return nr * self.camera_w + nc * self.camera_h - nr * nc

    @property
    def zero_weight_pixel_fraction(self) -> float:
        """Fraction of per-channel camera pixels that are single-pixel exactly
        invisible (the tier-1 axis-aligned basis fraction)."""
        return self.n_zero_weight_pixels_per_channel / self.camera_pixels

    @property
    def full_null_dim(self) -> int:
        """Exact dimension of ``ker(R)`` = camera_pixels - rank(R).

        For the separable downsample, ``rank(R) = rank(R_h) * rank(R_w)``."""
        rank = self.h_kernel.rank * self.w_kernel.rank
        return self.camera_pixels - rank

    @property
    def full_null_fraction(self) -> float:
        return self.full_null_dim / self.camera_pixels

    def zero_weight_pixel_mask(self) -> np.ndarray:
        """Boolean ``(camera_h, camera_w)`` mask: True where a single-pixel
        perturbation is exactly invisible to BOTH scorer heads (any channel)."""
        row_mask = np.zeros(self.camera_h, dtype=bool)
        col_mask = np.zeros(self.camera_w, dtype=bool)
        if self.zero_weight_rows:
            row_mask[np.asarray(self.zero_weight_rows)] = True
        if self.zero_weight_cols:
            col_mask[np.asarray(self.zero_weight_cols)] = True
        return row_mask[:, None] | col_mask[None, :]

    def is_pixel_invisible(self, row: int, col: int) -> bool:
        """True iff a single-pixel perturbation at camera ``(row, col)`` is
        certified exactly invisible (row OR col is a dropped axis index)."""
        if not (0 <= row < self.camera_h and 0 <= col < self.camera_w):
            raise EvaluatorInvisibilityBasisError("pixel out of camera bounds")
        return row in set(self.zero_weight_rows) or col in set(self.zero_weight_cols)

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema": EVALUATOR_INVISIBILITY_BASIS_SCHEMA,
            "tier": 1,
            "family": "resize_null_space",
            "evidence_grade": self.evidence_grade,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "camera_h": self.camera_h,
            "camera_w": self.camera_w,
            "scorer_h": self.scorer_h,
            "scorer_w": self.scorer_w,
            "camera_pixels": self.camera_pixels,
            "n_zero_weight_rows": self.h_kernel.n_zero_weight,
            "n_zero_weight_cols": self.w_kernel.n_zero_weight,
            "zero_weight_rows": list(self.zero_weight_rows),
            "zero_weight_cols": list(self.zero_weight_cols),
            "n_zero_weight_pixels_per_channel": self.n_zero_weight_pixels_per_channel,
            "zero_weight_pixel_fraction": self.zero_weight_pixel_fraction,
            "full_null_dim": self.full_null_dim,
            "full_null_fraction": self.full_null_fraction,
            "resize_rank": self.h_kernel.rank * self.w_kernel.rank,
        }


def derive_tier1_resize_null_space(
    *,
    camera_h: int = CAMERA_H,
    camera_w: int = CAMERA_W,
    scorer_h: int = SCORER_INPUT_H,
    scorer_w: int = SCORER_INPUT_W,
) -> Tier1ResizeNullSpace:
    """Derive (closed form) the tier-1 certified-exact resize null space shared
    by BOTH scorer heads.  No torch, no sampling — pure linear algebra over the
    deterministic interpolate kernel."""
    h_kernel = derive_resize_kernel("h", camera_h, scorer_h)
    w_kernel = derive_resize_kernel("w", camera_w, scorer_w)
    return Tier1ResizeNullSpace(
        camera_h=int(camera_h),
        camera_w=int(camera_w),
        scorer_h=int(scorer_h),
        scorer_w=int(scorer_w),
        h_kernel=h_kernel,
        w_kernel=w_kernel,
    )


@dataclass(frozen=True)
class Frame0SegNetCorollary:
    """The trivial corollary: SegNet reads only frame1.

    ``upstream/modules.py:108`` SegNet.preprocess_input slices ``x[:, -1, ...]``
    (the LAST frame = frame1).  Therefore the ENTIRE frame0 (all
    ``3 * camera_pixels`` directions) is exactly SegNet-invisible — no resize null
    needed.  frame0's PoseNet visibility is again gated by the same resize, so
    frame0's resize zero-weight pixels are invisible to BOTH heads (and all of
    frame0 is invisible to SegNet)."""

    camera_pixels: int
    n_channels: int = 3
    upstream_ref: str = "upstream/modules.py:108 (x[:, -1, ...])"

    @property
    def segnet_invisible_directions(self) -> int:
        """ALL of frame0 is SegNet-invisible (every pixel, every channel)."""
        return self.camera_pixels * self.n_channels

    @property
    def segnet_invisible_fraction(self) -> float:
        return 1.0  # 100% of frame0 is SegNet-invisible by construction.

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema": EVALUATOR_INVISIBILITY_BASIS_SCHEMA,
            "tier": 1,
            "family": "frame0_segnet_corollary",
            "evidence_grade": TIER1_EVIDENCE_GRADE,
            "upstream_ref": self.upstream_ref,
            "segnet_invisible_directions": self.segnet_invisible_directions,
            "segnet_invisible_fraction": self.segnet_invisible_fraction,
            "note": (
                "ALL of frame0 is SegNet-invisible (slice x[:,-1,...]); frame0 "
                "BOTH-head invisibility is the resize zero-weight pixel set."
            ),
        }


# ---------------------------------------------------------------------------
# TIER 2 — MEASURED low-sensitivity directions (atlas-consumed; kept SEPARATE).
# Reuses the Catalog #385 MeasurementScope so measured != certified is structural.
# ---------------------------------------------------------------------------
try:  # MeasurementScope is the canonical scope guard; reuse, do not refork.
    from tac.substrates._shared.constants_provenance_manifest import (
        MeasurementScope,
    )
except Exception:  # pragma: no cover - import-resilience for partial trees
    MeasurementScope = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Tier2MeasuredLowSensitivity:
    """TIER-2 MEASURED-LOW-SENSITIVITY direction summary for one pair / region.

    Sourced from the #36 atlas (the per-pair joint cone + pose-Jacobian + seg
    margin fields).  This is a MEASURED budget, NOT a certified zero: it is valid
    only inside ``measurement_scope`` (Catalog #385 discipline) and only up to the
    amplitude the cone radius permits.  Kept in a SEPARATE record from tier 1 so a
    consumer cannot treat a measured budget as a certified invisibility.

    ``pose_null_fraction`` / ``usable_budget_fraction`` etc. are the atlas's
    measured low-singular-direction summaries; ``pair_budget`` is the integrated
    free budget.  All carry the ``[macOS-CPU advisory]`` grade.
    """

    pair_index: int
    region_class: int | None
    usable_budget_fraction: float
    pose_null_fraction: float
    pair_budget: float
    mean_radius_usable: float
    pose_binds_fraction: float
    fragile_fraction: float
    cone_map_path: str
    cone_map_sha256: str
    measurement_scope: Any  # MeasurementScope (Catalog #385)
    evidence_grade: str = TIER2_EVIDENCE_GRADE
    score_claim: bool = False
    promotable: bool = False

    def __post_init__(self) -> None:
        for f in ("usable_budget_fraction", "pose_null_fraction",
                  "pose_binds_fraction", "fragile_fraction"):
            v = getattr(self, f)
            if not (0.0 <= float(v) <= 1.0):
                raise EvaluatorInvisibilityBasisError(
                    f"{f} must be a fraction in [0,1], got {v}"
                )
        if self.cone_map_path.startswith("/tmp"):
            raise EvaluatorInvisibilityBasisError(
                "cone_map_path must be durable (not /tmp) per CLAUDE.md disk hygiene"
            )

    @property
    def measurement_scope_empty(self) -> bool:
        """True when the measurement scope records no evidence (a fragility flag —
        a measured budget with an empty scope is suspect, NOT certified)."""
        sc = self.measurement_scope
        if sc is None:
            return True
        is_empty = getattr(sc, "is_empty", None)
        return bool(is_empty()) if callable(is_empty) else True

    def to_row(self) -> dict[str, Any]:
        sc = self.measurement_scope
        scope_obj = sc.as_dict() if (sc is not None and hasattr(sc, "as_dict")) else {}
        return {
            "schema": EVALUATOR_INVISIBILITY_BASIS_SCHEMA,
            "tier": 2,
            "family": "measured_low_sensitivity",
            "pair_index": self.pair_index,
            "region_class": self.region_class,
            "usable_budget_fraction": self.usable_budget_fraction,
            "pose_null_fraction": self.pose_null_fraction,
            "pair_budget": self.pair_budget,
            "mean_radius_usable": self.mean_radius_usable,
            "pose_binds_fraction": self.pose_binds_fraction,
            "fragile_fraction": self.fragile_fraction,
            "cone_map_path": self.cone_map_path,
            "cone_map_sha256": self.cone_map_sha256,
            "measurement_scope": scope_obj,
            "measurement_scope_empty": self.measurement_scope_empty,
            "evidence_grade": self.evidence_grade,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
        }


# ---------------------------------------------------------------------------
# The combined basis artifact (tier 1 + tier 2, kept structurally separate).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatorInvisibilityBasis:
    """The full two-tier invisibility basis artifact (schema
    ``evaluator_invisibility_basis.v1``).

    TIER 1 is the certified-exact closed-form resize null space + frame0
    corollary (hardware-independent, amplitude-unlimited up to clipping).
    TIER 2 is the per-pair measured low-sensitivity directions (atlas-consumed,
    scoped).  The two tiers are SEPARATE fields so consumers cannot conflate a
    measured budget with a certified zero.
    """

    tier1_resize: Tier1ResizeNullSpace
    frame0_corollary: Frame0SegNetCorollary
    tier2_rows: tuple[Tier2MeasuredLowSensitivity, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = EVALUATOR_INVISIBILITY_BASIS_SCHEMA

    # ---- query API ----------------------------------------------------------
    def tier1_pixel_invisible(self, frame_role: str, row: int, col: int,
                              channel: int = 0) -> bool:
        """Query: is a single-pixel perturbation at ``(frame_role, channel, row,
        col)`` certified tier-1 invisible to BOTH heads?

        - ``frame_role == "frame0"``: invisible to SegNet always; invisible to
          BOTH iff it is also in the resize zero-weight set (PoseNet still reads
          frame0 via the same resize).
        - ``frame_role == "frame1"``: invisible to BOTH iff in the resize
          zero-weight set (SegNet AND PoseNet both read frame1 via the resize)."""
        if frame_role not in ("frame0", "frame1"):
            raise EvaluatorInvisibilityBasisError("frame_role must be frame0/frame1")
        if not (0 <= channel < 3):
            raise EvaluatorInvisibilityBasisError("channel must be in [0,3)")
        return self.tier1_resize.is_pixel_invisible(row, col)

    def tier1_frame0_segnet_invisible(self) -> bool:
        """Query: ALL of frame0 is SegNet-invisible (the trivial corollary)."""
        return True

    def tier2_by_pair(self, pair_index: int) -> list[Tier2MeasuredLowSensitivity]:
        return [r for r in self.tier2_rows if r.pair_index == pair_index]

    def tier2_by_region(self, region_class: int) -> list[Tier2MeasuredLowSensitivity]:
        return [r for r in self.tier2_rows if r.region_class == region_class]

    def tier1_free_byte_fraction_per_channel(self) -> float:
        """The fraction of camera-pixel directions per channel that are tier-1
        certified-invisible single-pixel (the zero-weight pixel fraction)."""
        return self.tier1_resize.zero_weight_pixel_fraction

    def to_jsonl_lines(self) -> list[str]:
        """Persist as JSONL: a header (tier-1 summary + corollary + provenance)
        then one line per tier-2 row.  Tensors are NEVER inlined — tier-1 is
        derivable, tier-2 references cone-map ``.npz`` by path+sha."""
        import json

        header = {
            "kind": "header",
            "schema": self.schema,
            "tier1_resize": self.tier1_resize.to_summary(),
            "frame0_corollary": self.frame0_corollary.to_summary(),
            "n_tier2_rows": len(self.tier2_rows),
            "provenance": self.provenance,
            "evidence": {
                "tier1": TIER1_EVIDENCE_GRADE,
                "tier2": TIER2_EVIDENCE_GRADE,
                "score_claim": False,
                "promotable": False,
            },
        }
        lines = [json.dumps(header, sort_keys=True)]
        for row in self.tier2_rows:
            obj = {"kind": "tier2_row", **row.to_row()}
            lines.append(json.dumps(obj, sort_keys=True))
        return lines

    @classmethod
    def from_jsonl_lines(cls, lines: Iterable[str]) -> EvaluatorInvisibilityBasis:
        """Rebuild from JSONL.  Tier-1 is re-DERIVED from the header's camera /
        scorer sizes (the certified basis is reproducible from sizes alone — the
        header summary is an audit echo, not the source of truth)."""
        import json

        line_list = [ln for ln in lines if ln.strip()]
        if not line_list:
            raise EvaluatorInvisibilityBasisError("empty JSONL")
        header = json.loads(line_list[0])
        if header.get("kind") != "header":
            raise EvaluatorInvisibilityBasisError("first line must be header")
        t1 = header["tier1_resize"]
        tier1 = derive_tier1_resize_null_space(
            camera_h=int(t1["camera_h"]),
            camera_w=int(t1["camera_w"]),
            scorer_h=int(t1["scorer_h"]),
            scorer_w=int(t1["scorer_w"]),
        )
        corollary = Frame0SegNetCorollary(camera_pixels=tier1.camera_pixels)
        tier2_rows: list[Tier2MeasuredLowSensitivity] = []
        for ln in line_list[1:]:
            obj = json.loads(ln)
            if obj.get("kind") != "tier2_row":
                continue
            scope = None
            if MeasurementScope is not None:
                scope = MeasurementScope.from_dict(obj.get("measurement_scope", {}) or {})
            tier2_rows.append(
                Tier2MeasuredLowSensitivity(
                    pair_index=int(obj["pair_index"]),
                    region_class=(None if obj.get("region_class") is None
                                  else int(obj["region_class"])),
                    usable_budget_fraction=float(obj["usable_budget_fraction"]),
                    pose_null_fraction=float(obj["pose_null_fraction"]),
                    pair_budget=float(obj["pair_budget"]),
                    mean_radius_usable=float(obj["mean_radius_usable"]),
                    pose_binds_fraction=float(obj["pose_binds_fraction"]),
                    fragile_fraction=float(obj["fragile_fraction"]),
                    cone_map_path=str(obj["cone_map_path"]),
                    cone_map_sha256=str(obj["cone_map_sha256"]),
                    measurement_scope=scope,
                )
            )
        return cls(
            tier1_resize=tier1,
            frame0_corollary=corollary,
            tier2_rows=tuple(tier2_rows),
            provenance=dict(header.get("provenance", {})),
        )


def build_evaluator_invisibility_basis(
    *,
    camera_h: int = CAMERA_H,
    camera_w: int = CAMERA_W,
    scorer_h: int = SCORER_INPUT_H,
    scorer_w: int = SCORER_INPUT_W,
    tier2_rows: Iterable[Tier2MeasuredLowSensitivity] = (),
    provenance: Mapping[str, Any] | None = None,
) -> EvaluatorInvisibilityBasis:
    """Build the full two-tier invisibility basis.  Tier 1 is derived in closed
    form (always); tier 2 is whatever measured rows the caller supplies (atlas
    consumer)."""
    tier1 = derive_tier1_resize_null_space(
        camera_h=camera_h, camera_w=camera_w, scorer_h=scorer_h, scorer_w=scorer_w
    )
    corollary = Frame0SegNetCorollary(camera_pixels=tier1.camera_pixels)
    return EvaluatorInvisibilityBasis(
        tier1_resize=tier1,
        frame0_corollary=corollary,
        tier2_rows=tuple(tier2_rows),
        provenance=dict(provenance or {}),
    )


def tier2_rows_from_atlas(
    atlas: Any,
    *,
    pairs: int,
    authority_tier: str = "macos_cpu_advisory",
    artifact_path: str = "",
    confidence_interval: str = "",
    region_classes: Iterable[int] | None = None,
) -> list[Tier2MeasuredLowSensitivity]:
    """Project a #36 ``EvaluatorResponseAtlas`` into tier-2 measured rows.

    One row per pair (region_class=None) plus optional per-region rows.  Each
    carries a ``MeasurementScope`` (Catalog #385) recording the atlas's
    authority tier + the cone-map artifact so a consumer can never treat the
    measured budget as a certified zero.  No tensors copied: the cone-map path +
    sha point at the spatial budget surface.
    """
    if MeasurementScope is None:
        raise EvaluatorInvisibilityBasisError(
            "MeasurementScope unavailable; cannot build scoped tier-2 rows"
        )
    rows: list[Tier2MeasuredLowSensitivity] = []
    region_classes = tuple(region_classes or ())
    for pr in atlas.rows:  # AtlasPairRow iterable
        cone = pr.joint_cone_summary
        refs = pr.sensitivity_refs or {}
        scope = MeasurementScope(
            pairs=int(pairs),
            frames=2,  # frame pair
            amplitude_range=(0.0, 0.5),  # cone budget is ~half-uint8-step granular
            scorer_surfaces=("d_seg", "d_pose"),
            authority_tier=authority_tier,
            confidence_interval=confidence_interval,
            artifact_path=str(refs.get("cone_map_path", "") or artifact_path),
        )
        rows.append(
            Tier2MeasuredLowSensitivity(
                pair_index=int(pr.pair_index),
                region_class=None,
                usable_budget_fraction=float(cone.usable_budget_fraction),
                pose_null_fraction=float(cone.pose_null_fraction),
                pair_budget=float(cone.pair_budget),
                mean_radius_usable=float(cone.mean_radius_usable),
                pose_binds_fraction=float(cone.pose_binds_fraction),
                fragile_fraction=float(pr.seg_margin_field_stats.fragile_fraction),
                cone_map_path=str(refs.get("cone_map_path", "") or ""),
                cone_map_sha256=str(refs.get("cone_map_sha256", "") or ""),
                measurement_scope=scope,
            )
        )
        for rc in region_classes:
            reg = pr.per_region.get(int(rc))
            if not reg:
                continue
            rows.append(
                Tier2MeasuredLowSensitivity(
                    pair_index=int(pr.pair_index),
                    region_class=int(rc),
                    usable_budget_fraction=float(reg.get("usable_frac", 0.0)),
                    pose_null_fraction=float(cone.pose_null_fraction),
                    pair_budget=float(reg.get("budget", 0.0)),
                    mean_radius_usable=float(reg.get("mean_radius_usable", 0.0)),
                    pose_binds_fraction=float(cone.pose_binds_fraction),
                    fragile_fraction=float(pr.seg_margin_field_stats.fragile_fraction),
                    cone_map_path=str(refs.get("cone_map_path", "") or ""),
                    cone_map_sha256=str(refs.get("cone_map_sha256", "") or ""),
                    measurement_scope=scope,
                )
            )
    return rows


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CONTEST_TOTAL_BYTES",
    "EVALUATOR_INVISIBILITY_BASIS_SCHEMA",
    "SCORER_INPUT_H",
    "SCORER_INPUT_W",
    "TIER1_EVIDENCE_GRADE",
    "TIER2_EVIDENCE_GRADE",
    "EvaluatorInvisibilityBasis",
    "EvaluatorInvisibilityBasisError",
    "Frame0SegNetCorollary",
    "ResizeKernelDerivation",
    "Tier1ResizeNullSpace",
    "Tier2MeasuredLowSensitivity",
    "build_evaluator_invisibility_basis",
    "derive_resize_kernel",
    "derive_tier1_resize_null_space",
    "tier2_rows_from_atlas",
]
