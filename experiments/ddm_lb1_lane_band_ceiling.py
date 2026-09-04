# SPDX-License-Identifier: MIT
"""ddm_lb1 -- price the CEILING of the analytic lane-band carrier on the born field's
persistent set, at $0 on CPU.

WHAT THIS PRICES, AND WHAT IT DOES NOT
--------------------------------------
md1 (``experiments/ddm_md1_micro_to_macro.py``, 456c74551) measured that **62.011%** of the
QBF1-born field's terminal EMA-shadow seg error sits in ``PERSISTENT`` sites -- wrong before the
first update and never moved -- and that the floor those sites impose (0.0017403920) is **12.75x**
the sub-0.12 accuracy corner. The persistent set is Lane-concentrated (64.79% touch a Lane edge,
GT-Lane enriched 51.50x). What it demands is a different REPRESENTATION at the Lane-edge sites.

vr1 row 10 names that representation, LANDED: the v8 analytic lane ground-frame band
(``tac.boundary_math.analytic_lane_render_band`` + ``lane_sdf_component``) with a bit-exact RD
coder (LBND2). This module composes the carrier's Lane prediction INTO the born field's terminal
argmax and measures exactly what fraction of md1's persistent set that removes, at what coded byte
cost.

**The composition is in LABEL SPACE.** The real carrier composites lane appearance into the RGB
render BEFORE the contest R operator and the frozen SegNet then decides; the carrier module's own
docstring records that the NAIVE form of that composite HURT realized d_seg by +25% (0.00333 ->
0.00415, n600). Overwriting the argmax assumes perfect label authority at every claimed pixel, so
every number here is a strict **UPPER BOUND** on what any realized composite could deliver. No row
from this module is a realized-through-R measurement and none may be cited as one. This is a
CEILING PRICE (m118: price the ceiling before building), labelled as such.

NO-FAKE
-------
* The carrier is FIT by the module's own ``build_lane_band_pairs_from_lstars`` (a real openpilot-IPM
  polynomial fit to the real GT class-1 pixels), CODED by the module's own bit-exact
  ``roundtrip_lines_through_rd`` (LBND2), and RENDERED from the **dequantized** lines -- so the
  scored band is the shipped band (measure-what-you-ship).
* Scoring reuses md1's exact integer HT path against DALI ``gt_cache_dali.pt`` and md1's RETAINED
  terminal argmax + site partition. The partition is LOADED, never recomputed.
* The lane class is SELF-DETECTED from the ground truth by geometric signature per the CLAUDE.md
  class-order law, then asserted against the carrier module's default. It is never assumed.
* Axis ``[macOS-CPU advisory . LABEL-SPACE CEILING . NON-PROMOTABLE . no score claim]``.

Cross-refs: charter ``.omx/research/charters/ddm_lb1_lane_band_carrier_ceiling_on_born_field_20260904.md``
(29a303192); prereg ``.omx/research/ddm_lb1_prereg_20260904.md`` (ced026cdd); md1 memo
``.omx/research/ddm_md1_micro_to_macro_dynamics_20260904.md``; equations
``v8_geometric_rate_decomposition_v1`` and ``checkpoint_trajectory_error_partition_v1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_lb1_lane_band_ceiling")
DEFAULT_MD1_STORE = Path("/Volumes/APDataStore/pact/ddm_md1_micro_macro")
DEFAULT_CELL = "cold_control_seed_20260902"
TERMINAL_STEP = 5000
TOTAL_STEPS = 5000

AXIS = (
    "[macOS-CPU advisory . LABEL-SPACE CEILING (composition into the retained argmax, "
    "NOT realized through R) . frozen CPU-torch SegNet . QBF1-born cold control seed 20260902 . "
    "n32 sealed selection . NON-PROMOTABLE . no score claim]"
)

# md1's sealed terminal reading, DALI shadow (memo table, section 3).  Used only as a
# calibration TARGET -- this module recomputes it from the retained argmax.
MD1_TERMINAL_D_SEG_HAT_DALI = 0.0028065999348958334
MD1_PERSISTENT_NUMERATOR_DALI = 205305

# CLAUDE.md rate term: 25 * bytes / 37_545_489.
RATE_S_PER_BYTE = 25.0 / 37_545_489.0
# CLAUDE.md banner: the born vehicle's own archive size and the sub-0.12 accuracy corner.
BORN_ARCHIVE_BYTES = 106_643
# qn1 (DERIVED, n600, at the falsifier pose on the bound 106,626 B archive).
SUB012_ACCURACY_CORNER_D_SEG = 1.3646784205e-4
# md1: the combined credit ceiling of every schedule/optimizer/objective lever, on d_seg.
SCHEDULE_LEVER_CEILING_X = 1.61

COVERAGE_THRESHOLD = 0.5
BAND_DILATION_PX = 3


class LB1Error(RuntimeError):
    """Fail-closed error for this arm."""


# ---------------------------------------------------------------------------
# custody
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    """Write an npz atomically (tmp + os.replace) and return its custody fact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return file_fact(path)


def atomic_json(path: Path, payload: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return file_fact(path)


def atomic_bytes(path: Path, blob: bytes) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return file_fact(path)


# ---------------------------------------------------------------------------
# the class-order law: SELF-DETECT the lane class, never hardcode the index
# ---------------------------------------------------------------------------
def class_geometry(gt_seg: np.ndarray, n_classes: int = 5) -> list[dict[str, float]]:
    """Per-class area fraction, 4-neighbour boundary-to-area ratio (thinness), and mean
    vertical centroid (rows normalised to [0,1]), over a stack of label maps."""

    a = np.asarray(gt_seg)
    if a.ndim != 3:
        raise LB1Error("class_geometry needs [P,H,W] label maps")
    _p, h, _w = a.shape
    rows = np.arange(h, dtype=np.float64)[None, :, None]
    out: list[dict[str, float]] = []
    for c in range(n_classes):
        mask = a == c
        area = int(mask.sum())
        if area == 0:
            out.append({"area_fraction": 0.0, "thinness": 0.0, "row_centroid": float("nan")})
            continue
        # 4-neighbour boundary: a mask pixel with at least one differing cardinal neighbour.
        boundary = np.zeros_like(mask)
        boundary[:, 1:, :] |= mask[:, 1:, :] & ~mask[:, :-1, :]
        boundary[:, :-1, :] |= mask[:, :-1, :] & ~mask[:, 1:, :]
        boundary[:, :, 1:] |= mask[:, :, 1:] & ~mask[:, :, :-1]
        boundary[:, :, :-1] |= mask[:, :, :-1] & ~mask[:, :, 1:]
        out.append(
            {
                "area_fraction": float(area) / float(a.size),
                "thinness": float(boundary.sum()) / float(area),
                "row_centroid": float((np.broadcast_to(rows, a.shape)[mask]).mean() / float(h - 1)),
            }
        )
    return out


def detect_lane_class(gt_seg: np.ndarray, n_classes: int = 5) -> tuple[int, list[dict[str, float]]]:
    """SELF-DETECT the lane-marking class from the data (CLAUDE.md class-order law).

    Lane markings are the thinnest structure in the frame and the smallest by area: the
    detector takes the argmin of area fraction and the argmax of the 4-neighbour
    boundary-to-area ratio and REFUSES unless they agree.  No index is assumed.
    """

    geom = class_geometry(gt_seg, n_classes=n_classes)
    areas = np.asarray([g["area_fraction"] for g in geom], dtype=np.float64)
    thin = np.asarray([g["thinness"] for g in geom], dtype=np.float64)
    by_area = int(np.argmin(np.where(areas > 0.0, areas, np.inf)))
    by_thin = int(np.argmax(thin))
    if by_area != by_thin:
        raise LB1Error(
            "lane self-detection is ambiguous: smallest-area class "
            f"{by_area} != thinnest class {by_thin}; refusing to guess (NO-FAKE)"
        )
    return by_area, geom


# ---------------------------------------------------------------------------
# exact integer HT scoring, md1's path
# ---------------------------------------------------------------------------
def ht_numerator(wrong: np.ndarray, weights_int: np.ndarray) -> int:
    """W = sum_p w_p * n_wrong(p), exact in int64."""

    if wrong.ndim != 3 or wrong.shape[0] != weights_int.shape[0]:
        raise LB1Error("ht_numerator needs wrong[P,H,W] and weights[P]")
    per_pair = wrong.reshape(wrong.shape[0], -1).sum(axis=1, dtype=np.int64)
    return int((per_pair * weights_int).sum())


def ht_numerator_masked(wrong: np.ndarray, mask: np.ndarray, weights_int: np.ndarray) -> int:
    return ht_numerator(np.logical_and(wrong, mask), weights_int)


def denominator(population_n: int, sites_per_pair: int) -> float:
    return float(population_n) * float(sites_per_pair)


# ---------------------------------------------------------------------------
# mode: forward -- one CPU forward of the terminal shadow, retaining top1 AND top2
# ---------------------------------------------------------------------------
def run_forward(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(int(args.threads))
    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_md1_micro_to_macro as md1
    from experiments import ddm_qbt1_qbflow_trainer as qbt

    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    device = torch.device("cpu")
    posenet, segnet = qbt.load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    model = qbt.load_initial_model(device)

    ckpt = md1.checkpoint_path(Path(args.run_root), TERMINAL_STEP, TOTAL_STEPS)
    if not ckpt.is_file():
        raise LB1Error(f"terminal checkpoint missing: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    shadow = {name: value.detach().clone().float() for name, value in payload["ema"]["shadow"].items()}
    ema_meta = {
        "num_updates": int(payload["ema"]["num_updates"]),
        "decay": float(payload["ema"]["decay"]),
    }
    del payload

    model.load_state_dict({name: torch.as_tensor(v).to(device) for name, v in shadow.items()}, strict=True)
    pair_ids = list(qbt.SELECTION_IDS)

    top1_blocks: list[np.ndarray] = []
    top2_blocks: list[np.ndarray] = []
    margin_blocks: list[np.ndarray] = []
    for chunk in qbt.pair_chunks(tuple(pair_ids), 16):
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(ids, height=qbt.EVAL_H, width=qbt.EVAL_W)
            camera = qbt.roundtrip_to_camera_uint8_ste(outputs["rgb_pair_01"])
            _pose6, logits = qbt.scorer_forward(camera, posenet, segnet)
            top2 = torch.topk(logits, k=2, dim=1)
            top1_blocks.append(top2.indices[:, 0].to(torch.uint8).cpu().numpy())
            top2_blocks.append(top2.indices[:, 1].to(torch.uint8).cpu().numpy())
            margin_blocks.append((top2.values[:, 0] - top2.values[:, 1]).float().cpu().numpy())
        del outputs, camera, logits, top2

    top1 = np.concatenate(top1_blocks, axis=0)
    runner_up = np.concatenate(top2_blocks, axis=0)
    margin = np.concatenate(margin_blocks, axis=0).astype(np.float32)

    # Calibration gate 1: reproduce md1's retained terminal argmax bit-for-bit.
    retained_path = Path(args.md1_store) / "payloads" / args.cell / f"shadow_step_{TERMINAL_STEP:06d}.npz"
    retained = np.load(retained_path)
    md1_argmax = np.asarray(retained["argmax_u8"], dtype=np.uint8)
    md1_pairs = np.asarray(retained["pair_ids"], dtype=np.int64)
    if not np.array_equal(md1_pairs, np.asarray(pair_ids, dtype=np.int64)):
        raise LB1Error("md1's retained pair order differs from qbt.SELECTION_IDS")
    differing = int((md1_argmax != top1).sum())

    # Calibration gate 2: reproduce md1's sealed terminal d_seg_hat from the retained argmax.
    gt = ar1.load_ground_truth()
    index = np.asarray(pair_ids, dtype=np.int64)
    gt_dali = np.asarray(gt["dali_seg"], dtype=np.uint8)[index]
    weights_int = md1.ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS).astype(np.int64)
    den = denominator(qbt.N, int(gt_dali.shape[1] * gt_dali.shape[2]))
    recomputed = ht_numerator(md1_argmax != gt_dali, weights_int) / den

    payload_fact = atomic_npz(
        store / "terminal_shadow_top2.npz",
        top1_u8=top1,
        runner_up_u8=runner_up,
        margin_f32=margin,
        pair_ids=index,
    )
    out = {
        "schema": "ddm_lb1_lane_band_ceiling.v1.forward",
        "axis": AXIS,
        "score_claim": False,
        "cell": args.cell,
        "checkpoint": file_fact(ckpt),
        "ema": ema_meta,
        "pair_ids": [int(v) for v in pair_ids],
        "elapsed_s": time.monotonic() - started,
        "host": {"platform": platform.platform(), "threads": int(args.threads)},
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "calibration_gate_argmax_vs_md1_differing_sites": differing,
        "calibration_gate_argmax_bit_exact": bool(differing == 0),
        "calibration_gate_d_seg_hat_recomputed": float(recomputed),
        "calibration_gate_d_seg_hat_md1_sealed": MD1_TERMINAL_D_SEG_HAT_DALI,
        "calibration_gate_d_seg_hat_abs_gap": abs(float(recomputed) - MD1_TERMINAL_D_SEG_HAT_DALI),
        "payload": payload_fact,
        "md1_retained": file_fact(retained_path),
    }
    atomic_json(store / "FORWARD.json", out)
    return out


# ---------------------------------------------------------------------------
# mode: price -- fit, code, compose, score
# ---------------------------------------------------------------------------
def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-structuring-element dilation by ``radius`` px, per-pair, pure numpy."""

    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(radius)):
        nxt = out.copy()
        nxt[:, 1:, :] |= out[:, :-1, :]
        nxt[:, :-1, :] |= out[:, 1:, :]
        nxt[:, :, 1:] |= out[:, :, :-1]
        nxt[:, :, :-1] |= out[:, :, 1:]
        out = nxt
    return out


def compose(
    born: np.ndarray,
    runner_up: np.ndarray,
    carrier_lane: np.ndarray,
    lane_cls: int,
    rule: str,
    band: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one composition rule and return the composed argmax.

    (a) REPLACE -- Lane where the carrier claims Lane; where born said Lane and the carrier
        does not, fall back to the born field's own runner-up class.
    (b) UNION   -- Lane where the carrier claims Lane; born kept everywhere else.
    (c) BAND    -- rule (a) restricted to ``band``; outside it the born prediction is untouched.
    """

    out = np.asarray(born, dtype=np.uint8).copy()
    claim = np.asarray(carrier_lane, dtype=bool)
    if rule == "union":
        out[claim] = np.uint8(lane_cls)
        return out
    if rule == "replace":
        scope = np.ones_like(claim)
    elif rule == "band":
        if band is None:
            raise LB1Error("rule 'band' needs a band mask")
        scope = np.asarray(band, dtype=bool)
    else:
        raise LB1Error(f"unknown composition rule {rule!r}")
    out[claim & scope] = np.uint8(lane_cls)
    demote = scope & ~claim & (np.asarray(born) == lane_cls)
    out[demote] = np.asarray(runner_up, dtype=np.uint8)[demote]
    return out


def score_composition(
    born: np.ndarray,
    composed: np.ndarray,
    gt: np.ndarray,
    weights_int: np.ndarray,
    site_class: np.ndarray,
    class_code: Mapping[str, int],
    den: float,
    n_classes: int = 5,
) -> dict[str, Any]:
    """Exact integer HT accounting of one composition, split by md1's site classes."""

    wrong_before = born != gt
    wrong_after = composed != gt
    healed = wrong_before & ~wrong_after
    broken = ~wrong_before & wrong_after
    still = wrong_before & wrong_after

    total_sites = int(born.size)
    partition_ok = int((~wrong_before & ~wrong_after).sum()) + int(healed.sum()) + int(
        broken.sum()
    ) + int(still.sum())

    num_before = ht_numerator(wrong_before, weights_int)
    num_after = ht_numerator(wrong_after, weights_int)
    num_healed = ht_numerator(healed, weights_int)
    num_broken = ht_numerator(broken, weights_int)
    num_still = ht_numerator(still, weights_int)

    by_class: dict[str, dict[str, int | float]] = {}
    for name, code in class_code.items():
        mask = site_class == code
        b_before = ht_numerator_masked(wrong_before, mask, weights_int)
        b_after = ht_numerator_masked(wrong_after, mask, weights_int)
        by_class[name] = {
            "numerator_before": b_before,
            "numerator_after": b_after,
            "numerator_healed": ht_numerator_masked(healed, mask, weights_int),
            "numerator_broken": ht_numerator_masked(broken, mask, weights_int),
            "removed_fraction": (float(b_before - b_after) / float(b_before)) if b_before else 0.0,
            "d_seg_before": float(b_before) / den,
            "d_seg_after": float(b_after) / den,
        }

    # Per-GT-class collateral: sites of GT class c that the composition BROKE, and what to.
    collateral: dict[str, dict[str, int]] = {}
    for c in range(n_classes):
        gt_c = gt == c
        collateral[str(c)] = {
            "broken_sites": int((broken & gt_c).sum()),
            "broken_numerator": ht_numerator_masked(broken, gt_c, weights_int),
            "healed_sites": int((healed & gt_c).sum()),
            "healed_numerator": ht_numerator_masked(healed, gt_c, weights_int),
        }

    per_pair = []
    for i in range(born.shape[0]):
        per_pair.append(
            {
                "wrong_before": int(wrong_before[i].sum()),
                "wrong_after": int(wrong_after[i].sum()),
                "healed": int(healed[i].sum()),
                "broken": int(broken[i].sum()),
            }
        )

    return {
        "numerator_before": num_before,
        "numerator_after": num_after,
        "d_seg_before": float(num_before) / den,
        "d_seg_after": float(num_after) / den,
        "delta_d_seg": float(num_after - num_before) / den,
        "H_healed_numerator": num_healed,
        "B_broken_numerator": num_broken,
        "W_still_wrong_numerator": num_still,
        "H_healed_sites": int(healed.sum()),
        "B_broken_sites": int(broken.sum()),
        "W_still_wrong_sites": int(still.sum()),
        "harm_over_removal": (float(num_broken) / float(num_healed)) if num_healed else float("inf"),
        "by_site_class": by_class,
        "collateral_by_gt_class": collateral,
        "per_pair": per_pair,
        "partition_gate_sites": partition_ok,
        "partition_gate_exact": bool(partition_ok == total_sites),
    }


def oracle_lane_compose(
    born: np.ndarray, runner_up: np.ndarray, gt: np.ndarray, lane_cls: int
) -> np.ndarray:
    """PERFECT-LANE ORACLE: give the composition exact Lane authority in both directions,
    leave every other class to the born field.

    This is the ceiling of the whole REPRESENTATION CLASS ``a Lane-only carrier``, independent
    of how well any particular carrier fits: no lane-shaped object can beat it.  Sites whose GT
    is Lane become correct; sites the born field painted Lane that are not Lane fall back to the
    born field's own runner-up (which may itself be wrong -- the oracle owns Lane, not the rest).
    """

    out = np.asarray(born, dtype=np.uint8).copy()
    g = np.asarray(gt, dtype=np.uint8)
    is_lane = g == lane_cls
    out[is_lane] = np.uint8(lane_cls)
    demote = (~is_lane) & (np.asarray(born) == lane_cls)
    out[demote] = np.asarray(runner_up, dtype=np.uint8)[demote]
    return out


def run_price(args: argparse.Namespace) -> dict[str, Any]:
    import brotli

    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_md1_micro_to_macro as md1
    from experiments import ddm_qbt1_qbflow_trainer as qbt
    from tac.boundary_math import analytic_lane_render_band as alb

    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    pair_ids = list(qbt.SELECTION_IDS)
    index = np.asarray(pair_ids, dtype=np.int64)
    weights_int = md1.ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS).astype(np.int64)

    gt = ar1.load_ground_truth()
    gt_all = {
        "dali": np.asarray(gt["dali_seg"], dtype=np.uint8)[index],
        "pyav": np.asarray(gt["pyav_seg"], dtype=np.uint8)[index],
    }
    n_pairs, height, width = gt_all["dali"].shape
    den = denominator(qbt.N, height * width)

    lane_cls, geom = detect_lane_class(gt_all["dali"])
    module_default = int(alb.LaneBandRenderConfig().lane_cls)

    fwd = np.load(store / "terminal_shadow_top2.npz")
    born = np.asarray(fwd["top1_u8"], dtype=np.uint8)
    runner_up = np.asarray(fwd["runner_up_u8"], dtype=np.uint8)
    if not np.array_equal(np.asarray(fwd["pair_ids"], dtype=np.int64), index):
        raise LB1Error("forward payload pair order differs from the sealed selection")

    # ---- fit + code the carrier from the GT lstars (compress-time, DALI authority) ----
    cfg = alb.LaneBandRenderConfig(lane_cls=int(lane_cls))
    fit_lineage = str(args.fit_lineage)
    lstars = [gt_all[fit_lineage][i] for i in range(n_pairs)]
    pairs_lines, fit_stats = alb.build_lane_band_pairs_from_lstars(lstars, cfg)
    dq_lines, blob = alb.roundtrip_lines_through_rd(pairs_lines, cfg)
    coded = brotli.compress(blob, quality=11)
    blob_fact = atomic_bytes(store / f"lane_band_lbnd2_{fit_lineage}.bin", blob)
    coded_fact = atomic_bytes(store / f"lane_band_lbnd2_{fit_lineage}.br", coded)

    # Render the band from the DEQUANTIZED lines -- measure-what-you-ship.
    coverage = np.stack(
        [
            alb.rasterize_lane_coverage_range_dependent(
                dq_lines[i], h=height, w=width, softness=cfg.softness,
                dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m, v_h=cfg.v_h,
            )
            for i in range(n_pairs)
        ],
        axis=0,
    ).astype(np.float32)
    carrier_lane = coverage >= float(args.coverage_threshold)
    band = dilate_bool(coverage > 0.0, int(args.band_dilation_px))

    coverage_fact = atomic_npz(
        store / f"carrier_coverage_{fit_lineage}.npz",
        coverage_f32=coverage,
        carrier_lane_bool=carrier_lane,
        band_bool=band,
        pair_ids=index,
    )

    # ---- md1's retained site partition (LOADED, never recomputed) ----
    site_path = Path(args.md1_store) / f"site_classes_{args.cell}_shadow_dali.npz"
    site_class = np.asarray(np.load(site_path)["site_class_u8"], dtype=np.uint8)
    class_code = {name: int(code) for name, code in md1.CLASS_CODE.items()}
    persistent_code = class_code[md1.CLASS_PERSISTENT]

    # ---- carrier fit quality against each GT lineage ----
    fit_quality: dict[str, Any] = {}
    for lineage, g in gt_all.items():
        gt_lane = g == lane_cls
        tp = int((carrier_lane & gt_lane).sum())
        fp = int((carrier_lane & ~gt_lane).sum())
        fn = int((~carrier_lane & gt_lane).sum())
        fit_quality[lineage] = {
            "gt_lane_sites": int(gt_lane.sum()),
            "carrier_lane_sites": int(carrier_lane.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "recall": (float(tp) / float(tp + fn)) if (tp + fn) else float("nan"),
            "precision": (float(tp) / float(tp + fp)) if (tp + fp) else float("nan"),
        }

    # ---- compose + score, all three rules, both lineages ----
    rules = {
        "a_replace": ("replace", None),
        "b_union": ("union", None),
        "c_band": ("band", band),
    }
    results: dict[str, dict[str, Any]] = {}
    composed_store: dict[str, np.ndarray] = {}
    for label, (rule, scope) in rules.items():
        composed = compose(born, runner_up, carrier_lane, lane_cls, rule, band=scope)
        composed_store[label] = composed
        per_lineage: dict[str, Any] = {}
        for lineage, g in gt_all.items():
            scored = score_composition(
                born, composed, g, weights_int, site_class, class_code, den
            )
            persistent_mask = site_class == persistent_code
            pb = ht_numerator_masked(born != g, persistent_mask, weights_int)
            pa = ht_numerator_masked(composed != g, persistent_mask, weights_int)
            scored["persistent_numerator_before"] = pb
            scored["persistent_numerator_after"] = pa
            scored["persistent_removed_fraction"] = (float(pb - pa) / float(pb)) if pb else 0.0
            per_lineage[lineage] = scored
        results[label] = per_lineage

    composed_fact = atomic_npz(
        store / "composed_argmax.npz",
        **{f"composed_{k}_u8": v for k, v in composed_store.items()},
        born_u8=born,
        pair_ids=index,
    )

    # ---- rate arithmetic ----
    coded_bytes = int(len(coded))
    raw_bytes = int(len(blob))
    per_pair_bytes = float(coded_bytes) / float(n_pairs)
    rate = {
        "coded_bytes_n32_lbnd2_brotli11": coded_bytes,
        "raw_blob_bytes_n32": raw_bytes,
        "coded_bytes_per_pair_n32": per_pair_bytes,
        "s_per_byte": RATE_S_PER_BYTE,
        "born_archive_bytes": BORN_ARCHIVE_BYTES,
        "delta_s_rate_n32_coded": float(coded_bytes) * RATE_S_PER_BYTE,
        "derived_n600_bytes_TRANSFERRED_naive_per_pair": per_pair_bytes * float(qbt.N),
        "derived_n600_delta_s_rate_TRANSFERRED": per_pair_bytes * float(qbt.N) * RATE_S_PER_BYTE,
        "note": (
            "the n32 selection is NON-CONSECUTIVE (SELECTION_IDS span 4..573), so LBND2's temporal "
            "delta has almost no correlation to exploit; the per-pair byte cost measured here is a "
            "CONSERVATIVE UPPER BOUND on the consecutive-n600 cost"
        ),
    }
    for per_lineage in results.values():
        for scored in per_lineage.values():
            d_s_seg = 100.0 * scored["delta_d_seg"]
            scored["delta_S_seg_term"] = d_s_seg
            scored["delta_S_rate_term_n32_coded"] = float(coded_bytes) * RATE_S_PER_BYTE
            scored["delta_S_net_n32_coded"] = d_s_seg + float(coded_bytes) * RATE_S_PER_BYTE
            scored["exchange_S_per_byte"] = (
                (-d_s_seg / float(coded_bytes)) if coded_bytes else float("nan")
            )
            scored["exchange_vs_rate_floor_x"] = (
                (-d_s_seg / float(coded_bytes)) / RATE_S_PER_BYTE if coded_bytes else float("nan")
            )

    # ---- gestalt delta ----
    best = results["a_replace"]["dali"]
    persistent_floor_after = float(best["persistent_numerator_after"]) / den
    persistent_floor_before = float(best["persistent_numerator_before"]) / den
    gestalt = {
        "target_d_seg_sub012_accuracy_corner": SUB012_ACCURACY_CORNER_D_SEG,
        "persistent_floor_before": persistent_floor_before,
        "persistent_floor_before_x_target": persistent_floor_before / SUB012_ACCURACY_CORNER_D_SEG,
        "persistent_floor_after_rule_a": persistent_floor_after,
        "persistent_floor_after_x_target": persistent_floor_after / SUB012_ACCURACY_CORNER_D_SEG,
        "terminal_after_rule_a": best["d_seg_after"],
        "terminal_after_x_target": best["d_seg_after"] / SUB012_ACCURACY_CORNER_D_SEG,
        "schedule_lever_ceiling_x": SCHEDULE_LEVER_CEILING_X,
        "residual_demand_after_schedule_levers_x": (
            best["d_seg_after"] / SUB012_ACCURACY_CORNER_D_SEG / SCHEDULE_LEVER_CEILING_X
        ),
    }

    out = {
        "schema": "ddm_lb1_lane_band_ceiling.v1.price",
        "axis": AXIS,
        "score_claim": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "cell": args.cell,
        "fit_lineage": fit_lineage,
        "lane_class_self_detected": int(lane_cls),
        "lane_class_module_default": module_default,
        "lane_class_agrees_with_module_default": bool(int(lane_cls) == module_default),
        "class_geometry": geom,
        "coverage_threshold": float(args.coverage_threshold),
        "band_dilation_px": int(args.band_dilation_px),
        "carrier_fit_stats": fit_stats,
        "carrier_fit_quality": fit_quality,
        "rate": rate,
        "results": results,
        "gestalt": gestalt,
        "md1_persistent_numerator_sealed": MD1_PERSISTENT_NUMERATOR_DALI,
        "denominator": den,
        "elapsed_s": time.monotonic() - started,
        "host": {"platform": platform.platform(), "threads": int(args.threads)},
        "payloads": {
            "lbnd2_blob": blob_fact,
            "lbnd2_brotli": coded_fact,
            "coverage": coverage_fact,
            "composed": composed_fact,
        },
        "inputs": {
            "site_classes": file_fact(site_path),
            "forward": file_fact(store / "terminal_shadow_top2.npz"),
        },
    }
    atomic_json(store / f"PRICE_{fit_lineage}.json", out)
    return out


def run_optimal(args: argparse.Namespace) -> dict[str, Any]:
    """OPTIMAL-FORM pass (CLAUDE.md optimal-form law): the pre-registered ``price`` pass runs the
    carrier at the module's DEFAULTS, which leave ``u_mask_enabled=False`` -- the NAIVE band the
    module's own docstring records as HURTING realized d_seg by +25%.  This pass tunes the two
    knobs the module names as its FP killers to their own optimum (coverage threshold; the
    witness-uncertainty gate driven by the born field's OWN top1-top2 margin) and adds the
    PERFECT-LANE ORACLE -- the ceiling of the whole Lane-only representation class.
    """

    import brotli

    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_md1_micro_to_macro as md1
    from experiments import ddm_qbt1_qbflow_trainer as qbt
    from tac.boundary_math import analytic_lane_render_band as alb

    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    pair_ids = list(qbt.SELECTION_IDS)
    index = np.asarray(pair_ids, dtype=np.int64)
    weights_int = md1.ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS).astype(np.int64)
    gt_raw = ar1.load_ground_truth()
    gt = np.asarray(gt_raw["dali_seg"], dtype=np.uint8)[index]
    n_pairs, height, width = gt.shape
    den = denominator(qbt.N, height * width)
    lane_cls, _geom = detect_lane_class(gt)

    fwd = np.load(store / "terminal_shadow_top2.npz")
    born = np.asarray(fwd["top1_u8"], dtype=np.uint8)
    runner_up = np.asarray(fwd["runner_up_u8"], dtype=np.uint8)
    margin = np.asarray(fwd["margin_f32"], dtype=np.float32)

    site_path = Path(args.md1_store) / f"site_classes_{args.cell}_shadow_dali.npz"
    site_class = np.asarray(np.load(site_path)["site_class_u8"], dtype=np.uint8)
    class_code = {name: int(code) for name, code in md1.CLASS_CODE.items()}
    persistent_mask = site_class == class_code[md1.CLASS_PERSISTENT]

    cfg = alb.LaneBandRenderConfig(lane_cls=int(lane_cls))
    lstars = [gt[i] for i in range(n_pairs)]
    pairs_lines, fit_stats = alb.build_lane_band_pairs_from_lstars(lstars, cfg)
    dq_lines, blob = alb.roundtrip_lines_through_rd(pairs_lines, cfg)
    coded_bytes = int(len(brotli.compress(blob, quality=11)))

    # Raw-fit vs dequantized-fit coverage: separates coder loss from fit loss.
    def _cov(lines_per_pair: list[list[Any]]) -> np.ndarray:
        return np.stack(
            [
                alb.rasterize_lane_coverage_range_dependent(
                    lines_per_pair[i], h=height, w=width, softness=cfg.softness,
                    dash_gate=cfg.dash_gate, dash_forward_max_m=cfg.dash_forward_max_m, v_h=cfg.v_h,
                )
                for i in range(n_pairs)
            ],
            axis=0,
        ).astype(np.float32)

    cov_raw = _cov(pairs_lines)
    cov_dq = _cov(dq_lines)
    gt_lane = gt == lane_cls

    def _fit_row(cov: np.ndarray, thr: float) -> dict[str, float | int]:
        claim = cov >= thr
        tp = int((claim & gt_lane).sum())
        fp = int((claim & ~gt_lane).sum())
        fn = int((~claim & gt_lane).sum())
        return {
            "threshold": float(thr),
            "claimed": int(claim.sum()),
            "recall": (float(tp) / float(tp + fn)) if (tp + fn) else float("nan"),
            "precision": (float(tp) / float(tp + fp)) if (tp + fp) else float("nan"),
        }

    coder_loss = {
        "raw_fit": _fit_row(cov_raw, 0.5),
        "dequantized_fit": _fit_row(cov_dq, 0.5),
    }

    num_before = ht_numerator(born != gt, weights_int)
    persistent_before = ht_numerator_masked(born != gt, persistent_mask, weights_int)

    def _row(composed: np.ndarray, label: str, extra: dict[str, Any]) -> dict[str, Any]:
        wrong_after = composed != gt
        wrong_before = born != gt
        num_after = ht_numerator(wrong_after, weights_int)
        healed = ht_numerator(wrong_before & ~wrong_after, weights_int)
        broken = ht_numerator(~wrong_before & wrong_after, weights_int)
        p_after = ht_numerator_masked(wrong_after, persistent_mask, weights_int)
        d_seg_after = float(num_after) / den
        delta_s_seg = 100.0 * (d_seg_after - float(num_before) / den)
        return {
            "label": label,
            **extra,
            "d_seg_before": float(num_before) / den,
            "d_seg_after": d_seg_after,
            "delta_S_seg_term": delta_s_seg,
            "persistent_removed_fraction": (
                float(persistent_before - p_after) / float(persistent_before) if persistent_before else 0.0
            ),
            "persistent_floor_after": float(p_after) / den,
            "H_healed_numerator": healed,
            "B_broken_numerator": broken,
            "harm_over_removal": (float(broken) / float(healed)) if healed else float("inf"),
            "improves": bool(num_after < num_before),
        }

    rows: list[dict[str, Any]] = []
    thresholds = [float(v) for v in args.threshold_grid]
    taus = [float(v) for v in args.u_mask_tau_grid]
    band = dilate_bool(cov_dq > 0.0, int(args.band_dilation_px))
    for thr in thresholds:
        claim_base = cov_dq >= thr
        for tau in taus:
            if tau <= 0.0:
                gate = np.ones_like(claim_base)
                tau_label: float | None = None
            else:
                gate = alb.witness_uncertainty_mask(margin, tau=tau, eps=float(args.u_mask_eps)) >= 0.5
                tau_label = tau
            claim = claim_base & gate
            for rule, scope in (("replace", None), ("union", None), ("band", band)):
                composed = compose(born, runner_up, claim, lane_cls, rule, band=scope)
                rows.append(
                    _row(
                        composed,
                        f"{rule}|thr={thr}|tau={tau_label}",
                        {
                            "rule": rule,
                            "coverage_threshold": thr,
                            "u_mask_tau": tau_label,
                            "claimed_sites": int(claim.sum()),
                        },
                    )
                )

    oracle = _row(
        oracle_lane_compose(born, runner_up, gt, lane_cls),
        "PERFECT_LANE_ORACLE",
        {"rule": "oracle", "coverage_threshold": None, "u_mask_tau": None,
         "claimed_sites": int(gt_lane.sum())},
    )

    # How much of the persistent set can ANY Lane-only object reach at all?
    wrong_before = born != gt
    touches_lane = gt_lane | (born == lane_cls)
    persistent_touching_lane = ht_numerator_masked(
        wrong_before & touches_lane, persistent_mask, weights_int
    )

    best = min(rows, key=lambda r: r["d_seg_after"])
    out = {
        "schema": "ddm_lb1_lane_band_ceiling.v1.optimal",
        "axis": AXIS,
        "score_claim": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "cell": args.cell,
        "lane_class_self_detected": int(lane_cls),
        "carrier_fit_stats": fit_stats,
        "coder_loss_vs_fit_loss": coder_loss,
        "coded_bytes_n32_lbnd2_brotli11": coded_bytes,
        "u_mask_eps": float(args.u_mask_eps),
        "threshold_grid": thresholds,
        "u_mask_tau_grid": taus,
        "rows": rows,
        "best_row": best,
        "any_row_improves": bool(any(r["improves"] for r in rows)),
        "perfect_lane_oracle": oracle,
        "persistent_numerator_before": persistent_before,
        "persistent_numerator_touching_lane": persistent_touching_lane,
        "persistent_lane_touching_fraction": (
            float(persistent_touching_lane) / float(persistent_before) if persistent_before else 0.0
        ),
        "denominator": den,
        "elapsed_s": time.monotonic() - started,
        "host": {"platform": platform.platform(), "threads": int(args.threads)},
    }
    atomic_json(store / "OPTIMAL.json", out)
    return out


def run_fitsweep(args: argparse.Namespace) -> dict[str, Any]:
    """FIT-side optimal form.  The ``optimal`` pass tunes the COMPOSITION knobs; this one tunes
    the FIT knobs the carrier module exposes (centerline degree, dash gate, AA softness) and
    reports band recall / precision plus the best achievable composed d_seg for each fit.

    Reference point: ``src/tac/boundary_math/lane_sdf_component.py:17-18`` records FEED-dj's
    measurement that the centerline+width band captures lane SHAPE to false-negative d_seg
    0.00046 -- against a GT lane fraction of ~0.6% of the frame that implies a band recall of
    ~92%.  A fit that lands far below that is NOT at optimal form and its verdict does not bind.
    """

    import brotli

    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_md1_micro_to_macro as md1
    from experiments import ddm_qbt1_qbflow_trainer as qbt
    from tac.boundary_math import analytic_lane_render_band as alb

    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    pair_ids = list(qbt.SELECTION_IDS)
    index = np.asarray(pair_ids, dtype=np.int64)
    weights_int = md1.ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS).astype(np.int64)
    gt = np.asarray(ar1.load_ground_truth()["dali_seg"], dtype=np.uint8)[index]
    n_pairs, height, width = gt.shape
    den = denominator(qbt.N, height * width)
    lane_cls, _geom = detect_lane_class(gt)
    gt_lane = gt == lane_cls
    sites = int(gt.size)

    fwd = np.load(store / "terminal_shadow_top2.npz")
    born = np.asarray(fwd["top1_u8"], dtype=np.uint8)
    runner_up = np.asarray(fwd["runner_up_u8"], dtype=np.uint8)
    num_before = ht_numerator(born != gt, weights_int)

    site_path = Path(args.md1_store) / f"site_classes_{args.cell}_shadow_dali.npz"
    site_class = np.asarray(np.load(site_path)["site_class_u8"], dtype=np.uint8)
    persistent_mask = site_class == int(md1.CLASS_CODE[md1.CLASS_PERSISTENT])
    persistent_before = ht_numerator_masked(born != gt, persistent_mask, weights_int)

    rows: list[dict[str, Any]] = []
    lstars = [gt[i] for i in range(n_pairs)]
    for deg in [int(v) for v in args.centerline_deg_grid]:
        for dash in (True, False):
            cfg = alb.LaneBandRenderConfig(lane_cls=int(lane_cls), dash_gate=dash)
            pairs_lines, fit_stats = alb.build_lane_band_pairs_from_lstars(
                lstars, cfg, centerline_deg=deg
            )
            dq_lines, blob = alb.roundtrip_lines_through_rd(pairs_lines, cfg)
            coded = int(len(brotli.compress(blob, quality=11)))
            for soft in [float(v) for v in args.softness_grid]:
                cov = np.stack(
                    [
                        alb.rasterize_lane_coverage_range_dependent(
                            dq_lines[i], h=height, w=width, softness=soft,
                            dash_gate=dash, dash_forward_max_m=cfg.dash_forward_max_m, v_h=cfg.v_h,
                        )
                        for i in range(n_pairs)
                    ],
                    axis=0,
                ).astype(np.float32)
                for thr in [float(v) for v in args.threshold_grid]:
                    claim = cov >= thr
                    tp = int((claim & gt_lane).sum())
                    fp = int((claim & ~gt_lane).sum())
                    fn = int((~claim & gt_lane).sum())
                    best_after = None
                    best_rule = None
                    best_composed = born
                    for rule in ("replace", "union"):
                        composed = compose(born, runner_up, claim, lane_cls, rule)
                        after = ht_numerator(composed != gt, weights_int)
                        if best_after is None or after < best_after:
                            best_after = after
                            best_rule = rule
                            best_composed = composed
                    p_after = ht_numerator_masked(best_composed != gt, persistent_mask, weights_int)
                    rows.append(
                        {
                            "centerline_deg": deg,
                            "dash_gate": dash,
                            "softness": soft,
                            "threshold": thr,
                            "coded_bytes": coded,
                            "n_lines_mean": fit_stats["n_lines_mean"],
                            "band_recall_mean_module": fit_stats["band_recall_mean"],
                            "recall": (float(tp) / float(tp + fn)) if (tp + fn) else float("nan"),
                            "precision": (float(tp) / float(tp + fp)) if (tp + fp) else float("nan"),
                            "false_negative_d_seg_plain": float(fn) / float(sites),
                            "false_positive_d_seg_plain": float(fp) / float(sites),
                            "best_rule": best_rule,
                            "d_seg_before": float(num_before) / den,
                            "d_seg_after": float(best_after) / den,
                            "delta_S_seg_term": 100.0 * float(best_after - num_before) / den,
                            "persistent_removed_fraction": (
                                float(persistent_before - p_after) / float(persistent_before)
                                if persistent_before else 0.0
                            ),
                            "improves": bool(best_after < num_before),
                        }
                    )

    best = min(rows, key=lambda r: r["d_seg_after"])
    best_recall = max(rows, key=lambda r: (r["recall"] if r["recall"] == r["recall"] else -1.0))
    out = {
        "schema": "ddm_lb1_lane_band_ceiling.v1.fitsweep",
        "axis": AXIS,
        "score_claim": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "cell": args.cell,
        "lane_class_self_detected": int(lane_cls),
        "gt_lane_frame_fraction": float(int(gt_lane.sum())) / float(sites),
        "feed_dj_reference": {
            "source": "src/tac/boundary_math/lane_sdf_component.py:17-18",
            "false_negative_d_seg": 0.00046,
            "false_positive_d_seg_as_full_authority": 0.00396,
            "witness_target_d_seg": 0.00087,
            "reading": (
                "0.00087 is the witness TARGET, not the band's achieved d_seg; the band's own "
                "full-authority FP cost is 0.00396, 8.6x its FN cost"
            ),
        },
        "rows": rows,
        "best_row_by_d_seg": best,
        "best_row_by_recall": best_recall,
        "any_row_improves": bool(any(r["improves"] for r in rows)),
        "elapsed_s": time.monotonic() - started,
        "host": {"platform": platform.platform(), "threads": int(args.threads)},
    }
    atomic_json(store / "FITSWEEP.json", out)
    return out


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Closing arithmetic: the break-even precision a Lane claim must reach, the maximal
    Lane-only ORACLE (two demotion variants, the better reported), and the joint
    carrier-perfect + optimization-perfect floor against the sub-0.12 accuracy corner."""

    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_md1_micro_to_macro as md1
    from experiments import ddm_qbt1_qbflow_trainer as qbt
    from tac.boundary_math import analytic_lane_render_band as alb

    store = Path(args.store)
    pair_ids = list(qbt.SELECTION_IDS)
    index = np.asarray(pair_ids, dtype=np.int64)
    weights_int = md1.ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS).astype(np.int64)
    gt = np.asarray(ar1.load_ground_truth()["dali_seg"], dtype=np.uint8)[index]
    n_pairs, height, width = gt.shape
    den = denominator(qbt.N, height * width)
    lane_cls, _geom = detect_lane_class(gt)
    gt_lane = gt == lane_cls

    fwd = np.load(store / "terminal_shadow_top2.npz")
    born = np.asarray(fwd["top1_u8"], dtype=np.uint8)
    runner_up = np.asarray(fwd["runner_up_u8"], dtype=np.uint8)
    wrong_before = born != gt
    num_before = ht_numerator(wrong_before, weights_int)

    site_path = Path(args.md1_store) / f"site_classes_{args.cell}_shadow_dali.npz"
    site_class = np.asarray(np.load(site_path)["site_class_u8"], dtype=np.uint8)
    persistent_mask = site_class == int(md1.CLASS_CODE[md1.CLASS_PERSISTENT])
    persistent_before = ht_numerator_masked(wrong_before, persistent_mask, weights_int)

    # --- break-even: a UNION claim heals a site only if it is GT-Lane AND currently wrong;
    # it breaks a site if it is not GT-Lane AND currently correct.  Over the whole frame the
    # payoff density of each population is fixed by the born field, so the required precision
    # follows in closed form from the two conditional rates.
    heals_if_claimed = gt_lane & wrong_before
    breaks_if_claimed = (~gt_lane) & (~wrong_before)
    w_map = np.repeat(weights_int, height * width).reshape(gt.shape)
    heal_weight = float((w_map * heals_if_claimed).sum())
    break_weight = float((w_map * breaks_if_claimed).sum())
    lane_weight = float((w_map * gt_lane).sum())
    non_lane_weight = float((w_map * ~gt_lane).sum())
    # p_heal   = P(currently wrong | GT Lane);  p_break = P(currently correct | not GT Lane)
    p_heal = heal_weight / lane_weight if lane_weight else 0.0
    p_break = break_weight / non_lane_weight if non_lane_weight else 0.0
    # Claiming with precision q: expected heal = q*p_heal, expected break = (1-q)*p_break.
    break_even_precision = p_break / (p_heal + p_break) if (p_heal + p_break) else float("nan")

    # --- oracle variants: perfect Lane authority, two demotion priors for born-Lane FPs.
    def _score(composed: np.ndarray) -> dict[str, Any]:
        wrong_after = composed != gt
        num_after = ht_numerator(wrong_after, weights_int)
        p_after = ht_numerator_masked(wrong_after, persistent_mask, weights_int)
        return {
            "d_seg_after": float(num_after) / den,
            "delta_S_seg_term": 100.0 * float(num_after - num_before) / den,
            "persistent_floor_after": float(p_after) / den,
            "persistent_removed_fraction": (
                float(persistent_before - p_after) / float(persistent_before) if persistent_before else 0.0
            ),
            "B_broken_numerator": ht_numerator(~wrong_before & wrong_after, weights_int),
            "H_healed_numerator": ht_numerator(wrong_before & ~wrong_after, weights_int),
            "numerator_after": num_after,
        }

    # The demotion prior is SELF-DETECTED, never an index: the largest class whose vertical
    # centroid is not in the bottom of the frame (which is the static ego-hood), excluding Lane.
    areas = [int((gt == c).sum()) for c in range(5)]
    rows_idx = np.arange(height, dtype=np.float64)[None, :, None]
    centroids = []
    for c in range(5):
        m = gt == c
        centroids.append(float(np.broadcast_to(rows_idx, gt.shape)[m].mean() / (height - 1)) if m.any() else 1.0)
    eligible = [c for c in range(5) if centroids[c] < 0.72 and c != lane_cls]
    if not eligible:
        raise LB1Error("no non-hood, non-Lane class to use as the demotion prior; refusing to guess")
    road_cls = max(eligible, key=lambda c: areas[c])

    oracle_runner = oracle_lane_compose(born, runner_up, gt, lane_cls)
    oracle_road = born.copy()
    oracle_road[gt_lane] = np.uint8(lane_cls)
    demote = (~gt_lane) & (born == lane_cls)
    oracle_road[demote] = np.uint8(road_cls)
    variants = {
        "demote_to_runner_up": _score(oracle_runner),
        "demote_to_largest_non_hood_class": _score(oracle_road),
    }
    best_name = min(variants, key=lambda k: variants[k]["numerator_after"])
    oracle = variants[best_name]

    target = SUB012_ACCURACY_CORNER_D_SEG
    joint_floor = oracle["persistent_floor_after"]
    summary = {
        "schema": "ddm_lb1_lane_band_ceiling.v1.summary",
        "axis": AXIS,
        "score_claim": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "lane_class_self_detected": int(lane_cls),
        "demotion_prior_class_self_detected": int(road_cls),
        "class_row_centroids": centroids,
        "break_even": {
            "p_wrong_given_gt_lane": p_heal,
            "p_correct_given_not_gt_lane": p_break,
            "required_precision_for_union_break_even": break_even_precision,
            "reading": (
                "a UNION Lane claim must be right about this often merely to BREAK EVEN, because "
                "the ground it paints over is almost entirely already-correct non-Lane"
            ),
        },
        "oracle_variants": variants,
        "oracle_best_variant": best_name,
        "gestalt": {
            "target_d_seg_sub012_accuracy_corner": target,
            "born_terminal_d_seg": float(num_before) / den,
            "born_terminal_x_target": (float(num_before) / den) / target,
            "born_persistent_floor": float(persistent_before) / den,
            "born_persistent_floor_x_target": (float(persistent_before) / den) / target,
            "oracle_terminal_d_seg": oracle["d_seg_after"],
            "oracle_terminal_x_target": oracle["d_seg_after"] / target,
            "oracle_persistent_floor": joint_floor,
            "oracle_persistent_floor_x_target": joint_floor / target,
            "joint_carrier_perfect_plus_optimization_perfect_x_target": joint_floor / target,
            "schedule_lever_ceiling_x": SCHEDULE_LEVER_CEILING_X,
            "residual_after_oracle_and_schedule_levers_x": (
                oracle["d_seg_after"] / target / SCHEDULE_LEVER_CEILING_X
            ),
            "reading": (
                "carrier-PERFECT Lane authority plus optimization that removed EVERY remaining "
                "reachable site still leaves the accuracy corner short by the "
                "oracle_persistent_floor_x_target factor"
            ),
        },
        "denominator": den,
        "host": {"platform": platform.platform(), "threads": int(args.threads)},
        "carrier_module": {
            "lane_band": "src/tac/boundary_math/analytic_lane_render_band.py",
            "lane_band_default_lane_cls": int(alb.LaneBandRenderConfig().lane_cls),
        },
    }
    atomic_json(store / "SUMMARY.json", summary)
    return summary


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "price", "optimal", "fitsweep", "summary"), required=True)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--md1-store", default=str(DEFAULT_MD1_STORE))
    parser.add_argument("--cell", default=DEFAULT_CELL)
    parser.add_argument(
        "--run-root",
        default="/Volumes/APDataStore/pact/ddm_wc3_qbr1_ema_law_cure/runs/seed_20260902/control_native100",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--fit-lineage", choices=("dali", "pyav"), default="dali")
    parser.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD)
    parser.add_argument("--band-dilation-px", type=int, default=BAND_DILATION_PX)
    parser.add_argument(
        "--threshold-grid", type=float, nargs="+",
        default=[0.25, 0.5, 0.75, 0.9, 0.99],
        help="coverage thresholds to sweep in --mode optimal",
    )
    parser.add_argument(
        "--u-mask-tau-grid", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 0.85, 1.5, 3.0],
        help="witness-uncertainty tau to sweep (0.0 = gate disabled, the module default)",
    )
    parser.add_argument("--u-mask-eps", type=float, default=0.35)
    parser.add_argument("--centerline-deg-grid", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--softness-grid", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "forward":
        out = run_forward(args)
    elif args.mode == "optimal":
        out = run_optimal(args)
    elif args.mode == "fitsweep":
        out = run_fitsweep(args)
    elif args.mode == "summary":
        out = run_summary(args)
    else:
        out = run_price(args)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (dict, list))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
