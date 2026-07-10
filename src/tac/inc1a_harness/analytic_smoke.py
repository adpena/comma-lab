# SPDX-License-Identifier: MIT
"""$0 SMOKE of the 1a harness — the ANALYTIC-generator composite floor row.

increment-1a item 5 (SYNTHESIS_v2_v8 §A.4). Runs the assembler + meter on the EXISTING
ANALYTIC generators — the deg-3 **horizon poly** (Road↔Undriv), the **lane band** (structured
lane SDF), and the static **hood** clamp — with NO training. The composite tropical argmax vs
``L\\*`` on ``gt_n600.npz`` yields the analytic-composite MASK d_seg: a MEASURED positive
control that proves the assembler + meter run end-to-end, plus a first honest floor row for
the P8 brief.

**LABEL (binding, carried on the number every time):** ``[analytic-generators,
no-trained-fields]``. This is a HARNESS VALIDATION, **NOT the 1a verdict** — the real 1a
needs TRAINED decoupled fields + a matched-compute control arm (a later governed EVENT). The
analytic composite models {Road-complement, Undriv-above-horizon, Lane-band, Hood}; Movable
and fine multi-valued/lateral Undriv detail are UNMODELED (fold into the Road complement), so
this floor is HIGH by construction — that is the honest, expected shape (F4 lateral-undriv
under-coverage is visible here as Undriv/Movable flip mass).

Binding clause A (geometric home): every carrier states its unique home; the assembler stores
NOTHING derivable (deterministic code = rule-118 FREE). Binding clause B (§I): each carrier is
GEOMETRIC-MINIMAL (I1 horizon 4-coeff · I2 lane poly · I4 hood 1 static mask).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tac.boundary_math.hood_static_component import (
    build_static_hood_sdf,
    compute_static_hood_mask,
)
from tac.boundary_math.lane_sdf_component import build_structured_lane_sdf
from tac.boundary_math.movable_deshare import detect_seg_roles
from tac.inc1a_harness.composite_assembler import CarrierField, compose_partition
from tac.inc1a_harness.mask_dseg_meter import MaskDsegResult, measure_mask_dseg

DEFAULT_NPZ = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
ANALYTIC_LABEL = "[analytic-generators, no-trained-fields]"


def analytic_horizon_undriv_field(
    lstar: np.ndarray, *, road_cls: int, undriv_cls: int, degree: int = 3,
) -> np.ndarray:
    """The ANALYTIC deg-``degree`` horizon-poly Undrivable field (H,W) — lossy, no training.

    Reuses ``road_undriv_bulk_field._horizon_profile`` (the per-column topmost Road/Undriv arc)
    then fits a deg-``degree`` polynomial (``np.polyfit`` — the I1 GEOMETRIC-MINIMAL 4-coeff
    curve) and lifts it to an Undrivable channel: ``phi_undriv(row,col) = boundary_row(col) -
    row`` (positive ABOVE the horizon = Undriv side). Columns without dominant-arc support are
    deep-negative (fold into the Road complement — the honest single-valued limitation, F4).
    """

    from tac.boundary_math.road_undriv_bulk_field import _horizon_profile

    a = np.asarray(lstar)
    h, w = a.shape
    ys = _horizon_profile(a, int(road_cls), int(undriv_cls))  # (W,) boundary row, -1 unfittable
    phi = np.full((h, w), -float(max(h, w)), dtype=np.float32)  # deep-neg default (-> Road)
    valid = ys >= 0
    xs = np.where(valid)[0]
    if xs.size < int(degree) + 5:
        return phi
    coeffs = np.polyfit(xs.astype(np.float64), ys[valid].astype(np.float64), int(degree))
    boundary = np.rint(np.polyval(coeffs, np.arange(w, dtype=np.float64))).astype(np.int64)
    rows = np.arange(h, dtype=np.float32)[:, None]
    lifted = (boundary[None, :].astype(np.float32) - rows)  # + above horizon (Undriv side)
    phi[:, valid] = lifted[:, valid]
    return phi


def analytic_lateral_undriv_field(
    lstar: np.ndarray, *, road_cls: int, undriv_cls: int, degree: int = 2,
) -> np.ndarray:
    """The ANALYTIC lateral-extent Undrivable field (H,W) — the multi-branch side complement (owed-9).

    Fits ``x_L(y)``, ``x_R(y)`` (leftmost/rightmost Road column per row, via
    ``road_undriv_bulk_field._lateral_extents``) as deg-``degree`` polynomials and lifts them to an
    Undrivable channel POSITIVE OUTSIDE the drivable band: ``phi = max(x_L(row) - col, col -
    x_R(row))`` (positive to the LEFT of x_L or RIGHT of x_R = side Undriv), NEGATIVE inside the
    band (folds to Road/horizon via argmax). Homes the R6-MEASURED 97.54% lateral/side Undrivable
    mass the single-valued top arc (``analytic_horizon_undriv_field``) structurally cannot reach
    (F-P5-1). Rows without Road support stay deep-negative (fold to the Road complement).
    GEOMETRIC-MINIMAL (2 low-order curves). ``undriv_cls`` is accepted for signature symmetry with
    the horizon field (the lateral envelope is defined by the Road extent, not the Undriv mask).
    """

    from tac.boundary_math.road_undriv_bulk_field import _lateral_extents

    a = np.asarray(lstar)
    h, w = a.shape
    xl, xr = _lateral_extents(a, int(road_cls))
    phi = np.full((h, w), -float(max(h, w)), dtype=np.float32)  # deep-neg default (-> Road)
    valid = (xl >= 0) & (xr >= 0)
    ys = np.where(valid)[0]
    if ys.size < int(degree) + 5:
        return phi
    yy = ys.astype(np.float64)
    cl = np.polyfit(yy, xl[valid].astype(np.float64), int(degree))
    cr = np.polyfit(yy, xr[valid].astype(np.float64), int(degree))
    rows = np.arange(h, dtype=np.float64)
    xL = np.polyval(cl, rows)[:, None]  # (H,1)
    xR = np.polyval(cr, rows)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]  # (1,W)
    side = np.maximum(xL - cols, cols - xR).astype(np.float32)  # (H,W) positive OUTSIDE the band
    phi[valid] = side[valid]  # only rows with Road support; others stay deep-neg
    return phi


def build_analytic_carriers(
    lstar: np.ndarray, roles, hood_sdf: np.ndarray,
    *, include_lateral: bool = False, lateral_degree: int = 2,
) -> list[CarrierField]:
    """Build the {Undriv-horizon(+lateral), Lane-band, Hood} analytic carriers for one frame.

    Road is the COMPLEMENT sea (no positive field); Movable is UNMODELED (no analytic SDF
    generator — folds into the Road complement, F4-honest high floor).

    ``include_lateral`` (owed-9, DEFAULT OFF — the horizon-only floor 0.100403 is canonical):
    when True, the Undrivable channel is the ELEMENTWISE MAX of the single-valued top horizon arc
    (G1) and the lateral-extent envelope (G1b), merged into ONE carrier per class (clause-A: the
    assembler forbids two carriers for the same class). MEASURED n600: the naive convex
    leftmost/rightmost envelope HURTS (+0.0194: Road 0.021->0.089 as the smooth poly band cuts into
    true road) — a FORMULATION negative carved into the design space (see DAG FEED-v8unlock). The
    flag exists so the A/B is reproducible + the trained decoupled arm's lateral field (margin-aware,
    not a hard poly hull) has a measured baseline to beat.
    """

    undriv_phi = analytic_horizon_undriv_field(
        lstar, road_cls=roles.road, undriv_cls=roles.undriv,
    )
    home = "G1 ego-rigid Road<->Undriv horizon arc (deg-3 poly, single-valued)"
    name = "undriv_horizon_poly"
    if include_lateral:
        undriv_lateral = analytic_lateral_undriv_field(
            lstar, road_cls=roles.road, undriv_cls=roles.undriv, degree=int(lateral_degree),
        )
        undriv_phi = np.maximum(undriv_phi, undriv_lateral)  # G1 arc OR G1b side envelope
        home = (
            "G1 ego-rigid Road<->Undriv horizon arc (deg-3) + G1b lateral extents "
            "x_L(y),x_R(y) (multi-branch side Undriv, owed-9)"
        )
        name = "undriv_horizon_plus_lateral"
    lane_phi, _meta = build_structured_lane_sdf(np.asarray(lstar), lane_cls=roles.lane)
    return [
        CarrierField(
            class_id=int(roles.undriv), phi_hw=undriv_phi, name=name,
            geometric_home=home,
            representation_mode="GEOMETRIC-MINIMAL",
        ),
        CarrierField(
            class_id=int(roles.lane), phi_hw=np.asarray(lane_phi, np.float32), name="lane_band",
            geometric_home="G2 lane centerline band (openpilot analytic band, ground-frame)",
            representation_mode="GEOMETRIC-MINIMAL",
        ),
        CarrierField(
            class_id=int(roles.mycar), phi_hw=np.asarray(hood_sdf, np.float32), name="hood_static",
            geometric_home="G4 static frame0 ego-hood silhouette (one mask, all frames)",
            representation_mode="GEOMETRIC-MINIMAL",
        ),
    ]


def run_analytic_smoke(
    npz_path: str = DEFAULT_NPZ,
    *,
    n_frames: int | None = None,
    require_n600: bool | None = None,
    hood_agg: str = "majority",
    include_lateral: bool = False,
    lateral_degree: int = 2,
) -> MaskDsegResult:
    """Run the analytic-composite smoke: assemble → argmax → measure MASK d_seg vs L\\*.

    ``n_frames=None`` (DEFAULT) uses ALL frames in the cache (n600 = the honest floor row).
    A subset is admissible only as an explicitly non-authority dev check (``require_n600`` then
    defaults to False). Roles + the static hood are detected ONCE on the stack (per-video
    constant). Returns the :class:`MaskDsegResult` (carrying the ``[analytic-generators,
    no-trained-fields]`` label).
    """

    data = np.load(npz_path, mmap_mode="r")
    lstars_all = data["lstars"]
    total = int(lstars_all.shape[0])
    n = total if n_frames is None else min(int(n_frames), total)
    if require_n600 is None:
        require_n600 = n == total == 600
    lstars = np.asarray(lstars_all[:n])  # materialize the subset we score

    roles = detect_seg_roles(lstars)
    hood = compute_static_hood_mask(lstars, hood_cls=roles.mycar, agg=hood_agg)
    hood_sdf = build_static_hood_sdf(hood.mask)

    partitions: list[np.ndarray] = []
    for i in range(n):
        carriers = build_analytic_carriers(
            lstars[i], roles, hood_sdf,
            include_lateral=bool(include_lateral), lateral_degree=int(lateral_degree),
        )
        res = compose_partition(
            carriers, shape=(lstars.shape[1], lstars.shape[2]),
            complement_class=int(roles.road), bc_mode="no_offset",
        )
        partitions.append(res.partition)

    result = measure_mask_dseg(
        partitions, lstars, require_n600=bool(require_n600), label=ANALYTIC_LABEL,
    )
    result.extra["roles"] = float("nan")  # roles carried below in the CLI report
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="increment-1a analytic-composite smoke (harness validation)")
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="gt cache npz (default gt_n600)")
    ap.add_argument("--n-frames", type=int, default=None, help="subset for a dev check (default ALL = n600)")
    ap.add_argument("--allow-subset", action="store_true", help="permit a non-authority subset measurement")
    ap.add_argument("--hood-agg", default="majority", choices=["majority", "intersection", "union"])
    ap.add_argument("--include-lateral", action="store_true",
                    help="owed-9: add the G1b lateral-extent Undriv envelope (MEASURED WORSE at "
                    "n600 as a naive convex hull; default OFF = the canonical horizon-only floor)")
    ap.add_argument("--lateral-degree", type=int, default=2, help="owed-9 x_L/x_R poly degree")
    ap.add_argument("--json", action="store_true", help="emit the result as JSON")
    args = ap.parse_args(argv)

    require = None
    if args.n_frames is not None and args.allow_subset:
        require = False
    res = run_analytic_smoke(
        args.npz, n_frames=args.n_frames, require_n600=require, hood_agg=args.hood_agg,
        include_lateral=bool(args.include_lateral), lateral_degree=int(args.lateral_degree),
    )
    out = {
        "label": res.label,
        "agg_mask_dseg": round(res.agg_dseg, 6),
        "n_frames": res.n_frames,
        "is_n600": res.is_n600,
        "per_class_mask_dseg": {k: round(v, 6) for k, v in res.per_class_dseg.items()},
        "flip_share_by_class": {k: round(v, 4) for k, v in res.flip_share_by_class.items()},
        "total_flips": res.total_flips,
        "note": "HARNESS VALIDATION + first honest floor row; NOT the 1a verdict "
        "(needs trained decoupled fields + matched control arm).",
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{res.label}  analytic-composite MASK d_seg = {res.agg_dseg:.6f}  (N={res.n_frames}, n600={res.is_n600})")
        print("  per-class mask d_seg:", {k: round(v, 5) for k, v in res.per_class_dseg.items()})
        print("  flip share:", {k: round(v, 3) for k, v in res.flip_share_by_class.items()})
        print("  NOTE:", out["note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
