# SPDX-License-Identifier: MIT
"""The composite-argmax ASSEMBLER — compose per-class fields into ONE partition.

increment-1a item 1 (SYNTHESIS_v2_v8 §A.4 / §B ``measure_1a``). Given the v8 per-class
field outputs (horizon Road↔Undriv, lane band, movable, hood, complement seas), produce
ONE argmax partition via the Laguerre / tropical argmax
(:func:`tac.boundary_math.laguerre_logit_offset.power_diagram_argmax`), with the b_c
additive offset **pluggable** over the #386 dispatcher
(:func:`tac.boundary_math.laguerre_logit_offset.solve_head_offsets`):

    * ``no_offset``  — the SAFE DEFAULT (S5-V6 binding; 0-param, never ot_newton).
    * ``flip_weighted`` — M-a flip-share-target-MASS OT (BUILT; UN-ANALYZED — may re-inherit N-1).
    * ``flip_median``   — M-b flip-density MEDIAN closed-form (S1 Hamming-optimal; the #386 landing).
    * ``menon`` / ``ot_newton`` — the prior heuristics (ot_newton = N-1-FALSIFIED, never a default).

NO-FAKE: the assembler ACTUALLY composes — it stacks the real per-class fields and runs
the real power-diagram argmax on them; ``no_offset`` returns ``argmax_k phi_k`` exactly.
It stores NOTHING derivable — the composition is deterministic code (rule-118 FREE); only
the carrier PARAMS (the callers' fields) carry bytes, never this assembler.

Reconciliation (``reconcile_partition``): a bounded, guarded fixed-point of the b_c/argmax
solve — **NOT a globally-convex Dykstra** (SegNet-argmax pre-image is non-convex, S3's
honesty). Controls: max-iter cap + per-iter d_seg-monotonicity reject. DEFAULT ``max_iter=0``
⇒ a single-pass tropical argmax IS the composition for the paint-free 1a screen. The
merge→diff→correct alternating projection with paint is the 1b stage (governed-heavy),
where the "falsifiable decoupling test" label lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tac.boundary_math.bitmask_dseg import d_seg_reference
from tac.boundary_math.laguerre_logit_offset import (
    HEAD_OFFSET_SOLVERS,
    power_diagram_argmax,
    solve_head_offsets,
)

N_SEG_CLASSES = 5

# b_c modes the assembler accepts. ``no_offset`` is the safe default; the rest delegate
# to the #386 dispatcher ``solve_head_offsets`` (which RAISES if its inputs are absent).
BC_MODES: tuple[str, ...] = ("no_offset", *HEAD_OFFSET_SOLVERS)


class Inc1aAssemblerError(ValueError):
    """Raised on malformed assembler inputs (NO silent degeneration to a fake)."""


@dataclass(frozen=True)
class CarrierField:
    """One per-class scalar field to compose into the K-field stack.

    ``phi_hw`` is the ``(H, W)`` field for class ``class_id``; wherever it is the argmax
    (after b_c) that class wins. ``geometric_home`` (binding clause A) names the UNIQUE
    geometric home of this carrier so a duplicated fact is a finding, never a silent
    double-count. ``representation_mode`` (binding clause B / §I) is one of
    ``GEOMETRIC-MINIMAL`` | ``KKT-WATERFILLED`` — an unjustified capacity is a finding.
    """

    class_id: int
    phi_hw: np.ndarray
    name: str
    geometric_home: str
    representation_mode: str = "GEOMETRIC-MINIMAL"

    def __post_init__(self) -> None:
        a = np.asarray(self.phi_hw)
        if a.ndim != 2:
            raise Inc1aAssemblerError(
                f"carrier {self.name!r} phi_hw must be (H,W); got {a.shape}"
            )
        if self.representation_mode not in ("GEOMETRIC-MINIMAL", "KKT-WATERFILLED"):
            raise Inc1aAssemblerError(
                f"carrier {self.name!r} representation_mode must be GEOMETRIC-MINIMAL|"
                f"KKT-WATERFILLED (binding clause B); got {self.representation_mode!r}"
            )


@dataclass
class CompositeResult:
    """The assembled partition + its provenance."""

    partition: np.ndarray  # (H, W) int64 composite argmax label map
    offsets: np.ndarray  # (K,) applied Laguerre b_c offsets (zero when no_offset)
    bc_mode: str
    reconcile_iters: int
    bc_info: dict[str, float] = field(default_factory=dict)
    carrier_homes: dict[int, str] = field(default_factory=dict)


def assemble_fields(
    carriers: list[CarrierField],
    *,
    shape: tuple[int, int],
    n_classes: int = N_SEG_CLASSES,
    complement_class: int,
    complement_level: float = 0.0,
    deep_negative: float | None = None,
) -> np.ndarray:
    """Stack the per-class carrier fields into a ``(H, W, K)`` K-field.

    Classes with a carrier get that carrier's ``phi_hw``; the ``complement_class`` (the
    Road complement sea) is filled with the constant ``complement_level`` (rule-118: a
    frozen threshold, 0 video-derived bytes — ``road_complement_field``); every other
    unspecified class is filled ``deep_negative`` so it only wins where its own carrier
    (if any) lifts it. Two carriers for the SAME class is a duplicate-home error (clause A).
    """

    h, w = int(shape[0]), int(shape[1])
    k = int(n_classes)
    if not (0 <= int(complement_class) < k):
        raise Inc1aAssemblerError(f"complement_class {complement_class} out of [0,{k})")
    if deep_negative is None:
        deep_negative = -float(max(h, w))
    phi = np.full((h, w, k), float(deep_negative), dtype=np.float32)
    phi[..., int(complement_class)] = float(complement_level)

    seen: dict[int, str] = {}
    for c in carriers:
        cid = int(c.class_id)
        if not (0 <= cid < k):
            raise Inc1aAssemblerError(f"carrier {c.name!r} class_id {cid} out of [0,{k})")
        if cid in seen:
            raise Inc1aAssemblerError(
                f"DUPLICATE geometric home (clause A): class {cid} claimed by both "
                f"{seen[cid]!r} and {c.name!r} — each fact has exactly ONE home"
            )
        a = np.asarray(c.phi_hw, dtype=np.float32)
        if a.shape != (h, w):
            raise Inc1aAssemblerError(
                f"carrier {c.name!r} shape {a.shape} != (H,W)=({h},{w})"
            )
        phi[..., cid] = a
        seen[cid] = c.name
    return phi


def _resolve_offsets(
    phi_hwk: np.ndarray, bc_mode: str, gt: np.ndarray | None, pred: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Resolve the (K,) Laguerre b_c offsets for ``bc_mode`` via the #386 dispatcher.

    ``no_offset`` returns a zero vector (the safe default). Every other mode delegates to
    :func:`solve_head_offsets`, which RAISES if its required inputs (``phi``/``gt``/priors)
    are absent — the assembler never silently substitutes ``no_offset`` for a failed solve.
    """

    k = int(phi_hwk.shape[-1])
    m = str(bc_mode)
    if m == "no_offset":
        return np.zeros(k, dtype=np.float64), {"solver": -1.0, "iters": 0.0, "converged": 1.0}
    if m not in HEAD_OFFSET_SOLVERS:
        raise Inc1aAssemblerError(f"unknown bc_mode {bc_mode!r}; expected one of {BC_MODES}")
    # flip_* modes derive per-class targets from gt (+ optional realized pred); menon needs priors.
    priors = None
    target_masses = None
    if m == "menon":
        if gt is None:
            raise Inc1aAssemblerError("bc_mode='menon' needs gt (to form class priors)")
        g = np.asarray(gt).reshape(-1)
        priors = np.bincount(g.astype(np.int64), minlength=k).astype(np.float64)
    elif m == "ot_newton":
        if gt is None:
            raise Inc1aAssemblerError("bc_mode='ot_newton' needs gt (GT area frequencies)")
        g = np.asarray(gt).reshape(-1)
        target_masses = np.bincount(g.astype(np.int64), minlength=k).astype(np.float64)
        target_masses = target_masses / max(float(target_masses.sum()), 1.0)
    b, info = solve_head_offsets(
        m, priors=priors, phi=phi_hwk, target_masses=target_masses, gt=gt, pred=pred,
    )
    return np.asarray(b, dtype=np.float64), {k2: float(v) for k2, v in info.items()}


def reconcile_partition(
    phi_hwk: np.ndarray,
    *,
    bc_mode: str = "no_offset",
    gt: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    max_iter: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Bounded, guarded reconciliation of the composite argmax.

    ``max_iter == 0`` (DEFAULT): a SINGLE-PASS tropical argmax — the composition for the
    paint-free 1a screen. Returns ``(partition, offsets, info)``.

    ``max_iter > 0``: a bounded fixed-point of the b_c/argmax solve — re-solve the offsets
    with ``pred`` fed back to the CURRENT partition, re-argmax, repeat. This is the honest
    linearized alternating projection (S3): it has a **max-iter cap** and a **per-iter
    d_seg-monotonicity REJECT** (an iteration that RAISES d_seg vs ``gt`` is rolled back
    and the loop STOPS). **NO Dykstra-convergence guarantee is claimed** — the SegNet-argmax
    pre-image is non-convex; this is a local projection with guards, nothing more. Requires
    ``gt`` when ``max_iter > 0`` (the monotonicity reference).
    """

    phi = np.asarray(phi_hwk, dtype=np.float64)
    if phi.ndim != 3:
        raise Inc1aAssemblerError(f"phi_hwk must be (H,W,K); got {phi.shape}")
    offsets, info = _resolve_offsets(phi, bc_mode, gt, pred)
    part = power_diagram_argmax(phi, offsets)
    used_iters = 0
    if int(max_iter) > 0:
        if gt is None:
            raise Inc1aAssemblerError(
                "reconcile_partition(max_iter>0) needs gt (the d_seg-monotonicity reference)"
            )
        g = np.asarray(gt)
        best = float(d_seg_reference(part, g))
        for _ in range(int(max_iter)):
            cand_off, cand_info = _resolve_offsets(phi, bc_mode, gt, pred=part)
            cand = power_diagram_argmax(phi, cand_off)
            cand_dseg = float(d_seg_reference(cand, g))
            if cand_dseg > best:  # monotonicity REJECT: roll back, STOP (no convergence claim)
                info = {**info, "reconcile_rejected_at_iter": float(used_iters)}
                break
            if np.array_equal(cand, part):  # fixed point reached
                part, offsets, info = cand, cand_off, cand_info
                used_iters += 1
                break
            part, offsets, info, best = cand, cand_off, cand_info, cand_dseg
            used_iters += 1
    info = {**info, "reconcile_iters": float(used_iters)}
    return part, offsets, info


def compose_partition(
    carriers: list[CarrierField] | None = None,
    *,
    phi_hwk: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
    n_classes: int = N_SEG_CLASSES,
    complement_class: int | None = None,
    complement_level: float = 0.0,
    bc_mode: str = "no_offset",
    gt: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    reconcile_max_iter: int = 0,
) -> CompositeResult:
    """Compose per-class fields → ONE argmax partition (the assembler entry point).

    Provide EITHER ``carriers`` (list of :class:`CarrierField`, stacked via
    :func:`assemble_fields` — needs ``shape`` + ``complement_class``) OR a pre-stacked
    ``phi_hwk`` ``(H,W,K)``. Applies the ``bc_mode`` Laguerre offset (default ``no_offset``)
    and runs :func:`reconcile_partition`. Returns a :class:`CompositeResult`.
    """

    homes: dict[int, str] = {}
    if phi_hwk is None:
        if carriers is None or shape is None or complement_class is None:
            raise Inc1aAssemblerError(
                "compose_partition needs either phi_hwk, or (carriers + shape + complement_class)"
            )
        phi_hwk = assemble_fields(
            carriers, shape=shape, n_classes=n_classes,
            complement_class=complement_class, complement_level=complement_level,
        )
        homes = {int(c.class_id): c.geometric_home for c in carriers}
    phi = np.asarray(phi_hwk)
    if phi.ndim != 3:
        raise Inc1aAssemblerError(f"phi_hwk must be (H,W,K); got {phi.shape}")
    part, offsets, info = reconcile_partition(
        phi, bc_mode=bc_mode, gt=gt, pred=pred, max_iter=reconcile_max_iter,
    )
    return CompositeResult(
        partition=part,
        offsets=offsets,
        bc_mode=str(bc_mode),
        reconcile_iters=int(info.get("reconcile_iters", 0.0)),
        bc_info=info,
        carrier_homes=homes,
    )
