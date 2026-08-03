"""ddm_sx2 -- turn the "object-DISPLACEMENT" residual class into a MEASURED magnitude.

``ddm_sx1`` §2.5 split the seg residual by joining ``pc2``'s per-edge flip shares to its own GT
interface lengths:

    boundary-PRECISION  76.4 %  (Road<->Lane, Road<->Undriv, Road<->MyCar; 97-99 % ON the boundary)
    object-DISPLACEMENT 23.3 %  (Undriv<->Movable, Road<->Movable; flips/len 1.9-2.1x; 18-20 % >3 px OFF)

and left the displacement half with NO carrier, calling it "not separatrix-shaped".

A carrier cannot be designed against an adjective. This module measures the two signatures that
a rigid per-object TRANSLATION would produce on the ground-truth label field, and inverts each of
them independently for the displacement magnitude ``d``:

  route A (distance profile) -- translating a mask by ``d`` produces a band; the fraction of that
      band lying >3 px from the GT separatrix is a monotone function of ``d``. Match 18-20 %.
  route B (flip count) -- translating by ``d`` changes ~ perimeter * d pixels. Match the observed
      displacement-class flip count.

Two independent inversions of the same unknown. AGREEMENT corroborates rigid translation and
prices a per-object offset carrier. DISAGREEMENT REFUTES rigid translation as the mechanism, which
is the more valuable outcome because it redirects the carrier design.

A third signature discriminates translation from OBJECT DROPOUT (the decoder failing to render a
small object at all): dropout changes ``area`` pixels with a distance profile reaching the object's
inradius, whereas translation changes ``perimeter * d`` pixels confined to a thin band.

Axis: [macOS-CPU advisory]. NO contest scorer forward is run. score_claim=false.
Substrate: ``lstars`` (GT SegNet argmax, n600) only -- no decode, no vehicle, no scorer.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

SCORER_H = 384
SCORER_W = 512
DEFAULT_LSTARS = "experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz"

# Canonical comma10k order, self-detected and confirmed by ddm_sx1 §3 (NOT the forbidden luma-sort).
CLASS_ROAD, CLASS_LANE, CLASS_UNDRIV, CLASS_MOVABLE, CLASS_MYCAR = 0, 1, 2, 3, 4

# ddm_pc2 (via ddm_sx1 §2.5), on the tb1 ep399 vehicle. Carried as INPUTS to the inversion, and
# labelled: these are the numbers being inverted, not results of this module.
PC2_TOTAL_FLIPS = 458_738
PC2_FRAMES = 600  # the population pc2's total is over; NOT this module's --n-pairs
PC2_DISPLACEMENT_SHARE = 0.2332  # Undriv<->Movable 11.85 % + Road<->Movable 11.47 %
PC2_OFF_BOUNDARY_FRAC_LO = 0.179  # Road<->Movable   "18-20 % >3 px OFF"
PC2_OFF_BOUNDARY_FRAC_HI = 0.197  # Undriv<->Movable

# per-edge ">3 px OFF the GT boundary" shares and flip shares (ddm_sx1 §2.4/§2.5 <- ddm_pc2).
PC2_EDGES = {
    "Road<->Lane": {"flip_share": 0.4923, "off3": 0.026, "classes": (CLASS_ROAD, CLASS_LANE)},
    "Road<->Undriv": {"flip_share": 0.1626, "off3": 0.021, "classes": (CLASS_ROAD, CLASS_UNDRIV)},
    "Undriv<->Movable": {"flip_share": 0.1185, "off3": 0.197, "classes": (CLASS_UNDRIV, CLASS_MOVABLE)},
    "Road<->Movable": {"flip_share": 0.1147, "off3": 0.179, "classes": (CLASS_ROAD, CLASS_MOVABLE)},
    "Road<->MyCar": {"flip_share": 0.1089, "off3": 0.008, "classes": (CLASS_ROAD, CLASS_MYCAR)},
}


def label_boundary_4conn(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, dtype=bool)
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    b[1:, :] |= labels[:-1, :] != labels[1:, :]
    return b


def shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a boolean mask by (dy, dx), filling vacated pixels with False."""
    out = np.zeros_like(mask)
    ys0, ys1 = (dy, SCORER_H) if dy >= 0 else (0, SCORER_H + dy)
    xs0, xs1 = (dx, SCORER_W) if dx >= 0 else (0, SCORER_W + dx)
    out[ys0:ys1, xs0:xs1] = mask[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return out


def _dist_to_boundary(boundary: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest GT separatrix pixel."""
    return cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 5)


def run(n_pairs: int, lstars_path: str, max_d: int, out_json: str) -> dict:
    z = np.load(lstars_path)
    lstars = z["lstars"].astype(np.uint8)
    n = int(min(n_pairs, lstars.shape[0]))

    # eight directions at unit step; a real displacement has arbitrary direction, so we average
    # the signature over directions rather than privileging an axis.
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    changed = dict.fromkeys(range(1, max_d + 1), 0)
    changed_off3 = dict.fromkeys(range(1, max_d + 1), 0)
    changed_off5 = dict.fromkeys(range(1, max_d + 1), 0)
    dir_count = dict.fromkeys(range(1, max_d + 1), 0)

    movable_px = 0
    movable_components = 0
    movable_perimeter = 0
    dropout_changed = 0
    dropout_off3 = 0
    comp_area_hist: list[int] = []
    comp_inradius: list[float] = []
    # per-class "the whole object is wrong" reference: fraction of a class's own pixels lying
    # >3 px from the separatrix. This is the correct normaliser for the deep-fraction inversion --
    # a thin Lane dash has almost no deep interior, so the same observed off-3px share means a far
    # larger deep component for Lane than for a bulky Movable blob. Using one global normaliser
    # (my first formulation) silently assumes every class has the same interior geometry.
    class_px = np.zeros(5, dtype=np.int64)
    class_off3 = np.zeros(5, dtype=np.int64)

    for i in range(n):
        lab = lstars[i]
        gt_b = label_boundary_4conn(lab)
        dist = _dist_to_boundary(gt_b)
        mov = lab == CLASS_MOVABLE
        movable_px += int(mov.sum())
        deep = dist > 3.0
        for c in range(5):
            m = lab == c
            class_px[c] += int(m.sum())
            class_off3[c] += int((m & deep).sum())

        # per-component statistics: area, perimeter, inradius (max interior distance transform)
        ncomp, cc = cv2.connectedComponents(mov.astype(np.uint8), connectivity=8)
        movable_components += ncomp - 1
        mov_dt = cv2.distanceTransform(mov.astype(np.uint8), cv2.DIST_L2, 5)
        for c in range(1, ncomp):
            m = cc == c
            a = int(m.sum())
            comp_area_hist.append(a)
            comp_inradius.append(float(mov_dt[m].max()))
        movable_perimeter += int((label_boundary_4conn(mov.astype(np.uint8)) & mov).sum())

        # DROPOUT signature: the object simply is not rendered -> every one of its pixels flips.
        dropout_changed += int(mov.sum())
        dropout_off3 += int((mov & (dist > 3.0)).sum())

        # TRANSLATION signature, per displacement magnitude, averaged over 8 directions.
        for d in range(1, max_d + 1):
            for dy, dx in dirs:
                sm = shift(mov, dy * d, dx * d)
                band = mov ^ sm
                changed[d] += int(band.sum())
                changed_off3[d] += int((band & (dist > 3.0)).sum())
                changed_off5[d] += int((band & (dist > 5.0)).sum())
                dir_count[d] += 1

    # ---- inversion ----
    profile = {}
    for d in range(1, max_d + 1):
        tot = changed[d]
        profile[d] = {
            "changed_px_total": tot,
            "changed_px_per_frame_per_dir": tot / (n * len(dirs)),
            "frac_off_3px": changed_off3[d] / tot if tot else 0.0,
            "frac_off_5px": changed_off5[d] / tot if tot else 0.0,
        }

    def _invert(target: float, key: str) -> float | None:
        """Linear interpolation of d against a monotone signature."""
        ds = sorted(profile)
        vals = [profile[d][key] for d in ds]
        for j in range(len(ds) - 1):
            lo, hi = vals[j], vals[j + 1]
            if (lo - target) * (hi - target) <= 0 and hi != lo:
                return ds[j] + (target - lo) / (hi - lo)
        return None

    route_a_lo = _invert(PC2_OFF_BOUNDARY_FRAC_LO, "frac_off_3px")
    route_a_hi = _invert(PC2_OFF_BOUNDARY_FRAC_HI, "frac_off_3px")

    disp_flips = PC2_TOTAL_FLIPS * PC2_DISPLACEMENT_SHARE
    per_px_of_d = profile[1]["changed_px_total"] / (n * len(dirs))  # changed px per frame at d=1
    # pc2's total is over PC2_FRAMES frames, not this module's n -- normalise both to per-frame.
    route_b = (disp_flips / PC2_FRAMES) / per_px_of_d if per_px_of_d else None

    # ---- deep-fraction decomposition, per edge, with the per-class normaliser ----
    class_deep_ref = {
        c: (float(class_off3[c]) / float(class_px[c]) if class_px[c] else 0.0) for c in range(5)
    }
    edges_out = {}
    weighted_deep = 0.0
    for name, e in PC2_EDGES.items():
        ca, cb = e["classes"]
        # an error on this edge writes the wrong one of the two labels, so the deep reference is
        # the mean of the two classes' own interior fractions.
        ref = 0.5 * (class_deep_ref[ca] + class_deep_ref[cb])
        deep_frac = min(1.0, e["off3"] / ref) if ref > 0 else None
        edges_out[name] = {
            "flip_share": e["flip_share"],
            "observed_off3": e["off3"],
            "class_interior_reference_off3": ref,
            "deep_fraction": deep_frac,
            "boundary_shaped_fraction": (1.0 - deep_frac) if deep_frac is not None else None,
        }
        if deep_frac is not None:
            weighted_deep += e["flip_share"] * deep_frac

    areas = np.array(comp_area_hist, dtype=np.int64)
    inr = np.array(comp_inradius, dtype=np.float64)

    out = {
        "arm": "ddm_sx2",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "substrate": "lstars GT SegNet argmax n600; no decode, no vehicle",
        "n_frames": n,
        "movable": {
            "px_total": movable_px,
            "px_share_of_area": movable_px / (n * SCORER_H * SCORER_W),
            "components": movable_components,
            "components_per_frame": movable_components / n,
            "mean_area_px": float(areas.mean()) if areas.size else 0.0,
            "median_area_px": float(np.median(areas)) if areas.size else 0.0,
            "perimeter_px_total": movable_perimeter,
            "perimeter_per_frame": movable_perimeter / n,
            "mean_inradius_px": float(inr.mean()) if inr.size else 0.0,
            "median_inradius_px": float(np.median(inr)) if inr.size else 0.0,
        },
        "dropout_signature": {
            "changed_px_per_frame": dropout_changed / n,
            "frac_off_3px": dropout_off3 / dropout_changed if dropout_changed else 0.0,
        },
        "translation_profile": profile,
        "pc2_inputs_being_inverted": {
            "total_flips": PC2_TOTAL_FLIPS,
            "displacement_share": PC2_DISPLACEMENT_SHARE,
            "displacement_flips": disp_flips,
            "off_3px_frac_lo": PC2_OFF_BOUNDARY_FRAC_LO,
            "off_3px_frac_hi": PC2_OFF_BOUNDARY_FRAC_HI,
            "vehicle": "tb1 ep399 -- NOT the live cx1 vehicle (ddm_sx1 A4, untested transfer)",
        },
        "inversion": {
            "route_a_distance_profile_d_lo": route_a_lo,
            "route_a_distance_profile_d_hi": route_a_hi,
            "route_b_flip_count_d": route_b,
            "routes_agree": (
                route_a_lo is not None
                and route_b is not None
                and 0.5 <= route_b / route_a_lo <= 2.0
            ),
        },
        "class_interior_reference_off3": class_deep_ref,
        "edge_decomposition": edges_out,
        "seg_residual_deep_fraction": weighted_deep,
        "seg_residual_boundary_shaped_fraction": 1.0 - weighted_deep,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--max-d", type=int, default=10)
    ap.add_argument("--lstars", default=DEFAULT_LSTARS)
    ap.add_argument("--out", default=".omx/research/ddm_sx2_displacement_magnitude.json")
    a = ap.parse_args()
    o = run(a.n_pairs, a.lstars, a.max_d, a.out)
    m = o["movable"]
    print(f"frames={o['n_frames']}  Movable comps/frame={m['components_per_frame']:.2f}  "
          f"mean area={m['mean_area_px']:.0f}px  median area={m['median_area_px']:.0f}px  "
          f"perimeter/frame={m['perimeter_per_frame']:.0f}px  median inradius={m['median_inradius_px']:.2f}px")
    print(f"DROPOUT signature: {o['dropout_signature']['changed_px_per_frame']:.0f} px/frame changed, "
          f"frac >3px off boundary = {o['dropout_signature']['frac_off_3px']:.4f}")
    print(f"{'d':>4}{'changed px/frame':>20}{'frac >3px':>12}{'frac >5px':>12}")
    for d, p in sorted(o["translation_profile"].items()):
        print(f"{d:>4}{p['changed_px_per_frame_per_dir']:>20.1f}{p['frac_off_3px']:>12.4f}{p['frac_off_5px']:>12.4f}")
    inv = o["inversion"]
    print(f"\nroute A (distance profile) d = {inv['route_a_distance_profile_d_lo']} .. {inv['route_a_distance_profile_d_hi']}")
    print(f"route B (flip count)       d = {inv['route_b_flip_count_d']}")
    print(f"routes agree within 2x: {inv['routes_agree']}")
    print("\nclass interior reference (frac of class px >3px from separatrix):")
    print("  " + "  ".join(f"{c}:{v:.4f}" for c, v in o["class_interior_reference_off3"].items()))
    print(f"\n{'edge':<20}{'flip%':>8}{'off3':>8}{'ref':>8}{'DEEP':>9}{'bdry-shaped':>13}")
    for name, e in o["edge_decomposition"].items():
        print(f"{name:<20}{e['flip_share'] * 100:>7.2f}%{e['observed_off3']:>8.3f}"
              f"{e['class_interior_reference_off3']:>8.3f}{e['deep_fraction']:>8.3f} {e['boundary_shaped_fraction']:>12.3f}")
    print(f"\nSEG RESIDUAL: deep/object-level = {o['seg_residual_deep_fraction'] * 100:.2f}%   "
          f"boundary-shaped = {o['seg_residual_boundary_shaped_fraction'] * 100:.2f}%")


if __name__ == "__main__":
    main()
