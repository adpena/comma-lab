#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ddm_b4r (#807) — burn-4 ENDPOINT EXTRAS: the gc13 R1-bundle measurements beyond the scalar.

``ddm_pa1r_endpoint_verdict.py`` answers ONE question (n600 realized d_seg + bytes -> additive S).
The gc13 R1 bundle MAIN consumes needs the DECOMPOSITION (decompose-every-headline: a bare composite
is an unmeasured number):

  1. **per-class n600 d_seg**, all 5 classes, in xp1's EXACT convention/units;
  2. the **5x5 class-PAIR flip matrix** (which class is being mistaken for which);
  3. **per-class descent rates across the burn's accepted checkpoints** (the protected-vs-eroding
     read that the betti0 topology watch can only see as component counts);
  4. **Undrivable priced in S across the whole burn** — the class whose erosion ALARMed at
     window_02 and which MAIN adjudicated CONTINUE as a priced trade.

CONVENTION (xp1's, reused VERBATIM via ``ddm_qa92_carrier_discriminator._per_class_flip_counts`` --
NOT re-derived): a flip is charged to its **GT class**.  ``d_seg_c = flips_gt_c / (N*H*W)`` so the
per-class values SUM to the aggregate ``d_seg`` (a sum-check is emitted and asserted); ``S_c =
100*d_seg_c``.  The secondary within-class rate ``flips_gt_c / px_gt_c`` is emitted alongside,
explicitly labelled -- it does NOT sum to d_seg and is never the headline.

COST: ZERO extra SegNet forwards over a plain d_seg verdict --
``cpu_verdict_d_seg_argmax_batch`` already returns the realized argmax maps from the SAME forward
the per-pair count uses.  Everything here is numpy bookkeeping on maps we already paid for.

Rendering/EMA/quant handling mirrors ``ddm_pa1r_endpoint_verdict.endpoint_verdict`` exactly (EMA
shadow shipped; STE quantization re-engaged past the knee) so the aggregate d_seg reproduces pa1r's.

score_claim=False; every number [macOS-CPU/MLX advisory]; pointer 0.1910828242 [contest-CPU] UNMOVED.

Usage:
  ddm_b4r_endpoint_extras.py <final_window_dir> [--root <burn root>] [--parent-ckpt <path>]
                             [--chunk 100] [--output-json <path>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling experiments/ imports

GT_CACHE = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
CONTEST_N = 37_545_489
N_PAIRS = 600
CKPT_NAME = "stage_seg_trunk_tau_final.npz"

# comma10k CANONICAL order — MEASURED, never luma-sorted (CLAUDE.md class-index non-negotiable).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
UNDRIV = 2

# --- COMPARISON ANCHORS (measured elsewhere; carried for delta arithmetic, never re-derived here).
# xp1 ep641 per-class residual, S units, r1c parent = the burn's PRE-BURN state.
#   source: /Volumes/VertigoDataTier/pact/ddm_xp1_20260731/xp1_verdict.json (n600 exact)
XP1_EP641_PER_CLASS_S = (0.18845, 0.12589, 0.05574, 0.03792, 0.01840)  # sums 0.42640 = 100*d_seg
# fl1 (#813) per-class GT-flicker FLOOR, S units — the smooth-label lower bound per class.
#   source: .omx/research/ddm_fl1_perclass_flicker_floors_20260731.md §2 (sums 0.53179)
FL1_PER_CLASS_FLOOR_S = (0.18890, 0.23160, 0.03939, 0.02850, 0.04340)


def _cfg_from_json(cfg_path: Path):
    import experiments.train_tr1_partition_renderer_mlx as T
    d = json.loads(cfg_path.read_text())
    fields = {f for f in T.TR1Config.__dataclass_fields__}
    return T.TR1Config(**{k: v for k, v in d.items() if k in fields})


def _ckpt_epoch(ckpt: Path) -> int:
    """Checkpoint epoch from the npz's top-level ``meta::epoch`` key — the SAME read
    ``tools/supervise_ddm_b4s_burn4.py::_ckpt_epoch`` uses. (The loaded state dict's
    ``meta`` sub-mapping does NOT carry it; reading it there silently yields 0, which
    would zero every per-epoch descent rate — caught by the ddm_b4r code-path smoke.)"""
    try:
        return int(np.load(ckpt, allow_pickle=False)["meta::epoch"][0])
    except Exception:
        return 0


def _load_model(arm_dir: Path, ckpt: Path):
    """EMA shadow + past-knee STE re-engage — VERBATIM pa1r semantics (see its §3.3(a) note:
    without the re-engage the verdict under-reports d_seg by ~1e-4 = the quantization gap)."""
    import mlx.core as mx

    import experiments.train_tr1_partition_renderer_mlx as T
    cfg = replace(_cfg_from_json(arm_dir / "tr1_config.json"), num_pairs=N_PAIRS)
    model = T.build_module(cfg)
    st = T.load_checkpoint(ckpt, model)
    T.ema_snapshot_swap(model, st["ema"])
    stage = st.get("meta", {}).get("stage", "")
    if cfg.token_quant_anneal == "at_knee" and stage != "seg_trunk_ce":
        model._quant_engaged = True
    mx.eval(model.parameters())
    return model, cfg, _ckpt_epoch(ckpt)


def measure_checkpoint(arm_dir: Path, ckpt: Path, chunk: int, pairs: int = N_PAIRS) -> dict:
    """n600 realized decomposition for ONE checkpoint: aggregate d_seg (pa1r-identical),
    per-class flip mass in xp1 convention, and the 5x5 GT-class x realized-class matrix."""
    import mlx.core as mx

    import experiments.train_tr1_partition_renderer_mlx as T
    from ddm_qa92_carrier_discriminator import N_CLASSES, _per_class_flip_counts
    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_argmax_batch,
    )
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet

    model, cfg, meta_epoch = _load_model(arm_dir, ckpt)
    lstars = open_stored_npy_memmap(GT_CACHE, "lstars")
    seg_cpu = load_real_segnet("cpu")

    dsegs: list[float] = []
    cls_flip = np.zeros(N_CLASSES, dtype=np.int64)   # flips charged to GT class (xp1)
    cls_gt = np.zeros(N_CLASSES, dtype=np.int64)     # GT pixel population per class
    pair_mx = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)  # [gt, realized]
    total_px = 0

    for s in range(0, pairs, chunk):
        block = list(range(s, min(s + chunk, pairs)))
        frames = []
        with mx.stream(mx.cpu):
            for i in block:
                rgb = model.render_frame(int(i))
                mx.eval(rgb)
                frames.append(np.asarray(rgb, dtype=np.float32)[0])
        cams = [_torch_R_to_camera_uint8(f) for f in frames]
        gts = [np.asarray(lstars[i], dtype=np.int64) for i in block]
        ds, realized = cpu_verdict_d_seg_argmax_batch(seg_cpu, cams, gts)
        dsegs.extend(ds)
        for k, g in enumerate(gts):
            cls_flip += _per_class_flip_counts(realized[k], g)
            # np.bincount on the joint index is the 5x5 matrix in one pass (no python loop)
            pair_mx += np.bincount((g.ravel() * N_CLASSES + realized[k].ravel()),
                                   minlength=N_CLASSES * N_CLASSES).reshape(
                                       N_CLASSES, N_CLASSES).astype(np.int64)
            for c in range(N_CLASSES):
                cls_gt[c] += int((g == c).sum())
            total_px += g.size
        del frames, cams, gts, realized

    d_seg = float(np.mean(dsegs))
    per_class_dseg = (cls_flip / float(total_px)).astype(np.float64)
    ledger = T.counted_bytes_ledger(model, cfg)
    total_bytes = int(ledger["total_counted_bytes"])
    # SUM-CHECK: the xp1 convention makes per-class d_seg PARTITION the aggregate exactly.
    sum_err = float(abs(per_class_dseg.sum() - d_seg))
    return {
        "arm_dir": str(arm_dir), "ckpt": str(ckpt), "meta_epoch": meta_epoch,
        "n_pairs": int(pairs),
        "CODE_PATH_SMOKE_NOT_A_MEASUREMENT": bool(pairs != N_PAIRS),
        "measurement_valid": bool(pairs == N_PAIRS),
        "n600_d_seg": d_seg, "d_seg_per_pair_max": float(np.max(dsegs)),
        "total_counted_bytes": total_bytes,
        "seg_term_s": 100.0 * d_seg, "rate_term_s": 25.0 * total_bytes / CONTEST_N,
        "s_additive": 100.0 * d_seg + 25.0 * total_bytes / CONTEST_N,
        "per_class": {
            "convention": "flip charged to its GT class (xp1 _per_class_flip_counts, reused "
                          "verbatim); d_seg_c = flips_gt_c/(N*H*W) so the 5 values PARTITION "
                          "the aggregate d_seg",
            "class_names": list(CLASS_NAMES),
            "flip_px": cls_flip.tolist(), "gt_px": cls_gt.tolist(),
            "d_seg": per_class_dseg.tolist(),
            "s_units": (100.0 * per_class_dseg).tolist(),
            "within_class_error_rate": [
                float(cls_flip[c] / cls_gt[c]) if cls_gt[c] else None
                for c in range(N_CLASSES)],
            "within_class_error_rate_note": "SECONDARY (flips_gt_c/px_gt_c); does NOT sum to "
                                            "d_seg — never the headline",
            "sum_check_abs_err": sum_err,
            "sum_check_ok": bool(sum_err < 1e-12),
        },
        "class_pair_flip_matrix": {
            "layout": "M[gt_class][realized_class]; diagonal = correct px, off-diagonal = flips",
            "class_names": list(CLASS_NAMES),
            "counts": pair_mx.tolist(),
            "off_diagonal_s_units": [
                [0.0 if i == j else 100.0 * pair_mx[i][j] / float(total_px)
                 for j in range(N_CLASSES)] for i in range(N_CLASSES)],
            "total_px": int(total_px),
        },
        "score_claim": False, "axis": "[macOS-CPU/MLX advisory]",
    }


def _accepted_checkpoints(final_dir: Path, root: Path, parent_ckpt: Path | None) -> list[dict]:
    """The burn's accepted state trajectory, oldest first: the r1c parent (pre-burn), then every
    EXACT ``window_NN`` that produced a final ckpt (retired/rejected debris excluded by the same
    ``re.fullmatch`` rule the supervisor uses)."""
    stops: list[dict] = []
    if parent_ckpt is not None and parent_ckpt.is_file():
        stops.append({"label": "parent_r1c_ep641", "arm_dir": parent_ckpt.parent.parent,
                      "ckpt": parent_ckpt})
    for w in sorted(p for p in root.glob("window_*")
                    if p.is_dir() and re.fullmatch(r"window_\d{2}", p.name)):
        ck = w / "checkpoints" / CKPT_NAME
        if ck.is_file():
            stops.append({"label": w.name, "arm_dir": w, "ckpt": ck})
    fck = final_dir / "checkpoints" / CKPT_NAME
    if not any(s["ckpt"] == fck for s in stops) and fck.is_file():
        stops.append({"label": final_dir.name, "arm_dir": final_dir, "ckpt": fck})
    return stops


def build_extras(final_dir: Path, root: Path, parent_ckpt: Path | None, chunk: int,
                 pairs: int = N_PAIRS) -> dict:
    stops = _accepted_checkpoints(final_dir, root, parent_ckpt)
    measured: list[dict] = []
    for st in stops:
        try:
            m = measure_checkpoint(st["arm_dir"], st["ckpt"], chunk, pairs)
            m["label"] = st["label"]
            measured.append(m)
        except Exception as exc:  # one bad stop never costs the others
            measured.append({"label": st["label"], "ckpt": str(st["ckpt"]),
                             "error": repr(exc)})
    ok = [m for m in measured if "error" not in m]

    # --- per-class DESCENT RATES across the accepted trajectory (S units per stop and per epoch)
    descent: list[dict] = []
    for a, b in zip(ok, ok[1:]):
        de = (b["meta_epoch"] or 0) - (a["meta_epoch"] or 0)
        seg = [b["per_class"]["s_units"][c] - a["per_class"]["s_units"][c] for c in range(5)]
        descent.append({
            "from": a["label"], "to": b["label"],
            "epochs": de,
            "per_class_delta_s": dict(zip(CLASS_NAMES, seg)),
            "per_class_delta_s_per_100ep": dict(zip(
                CLASS_NAMES, [100.0 * s / de if de else None for s in seg])),
            "aggregate_delta_seg_s": b["seg_term_s"] - a["seg_term_s"],
            "aggregate_delta_rate_s": b["rate_term_s"] - a["rate_term_s"],
            "aggregate_net_delta_s": b["s_additive"] - a["s_additive"],
            "improving_classes": [CLASS_NAMES[c] for c in range(5) if seg[c] < 0],
            "eroding_classes": [CLASS_NAMES[c] for c in range(5) if seg[c] > 0],
        })

    # --- UNDRIVABLE priced across the WHOLE burn (the class MAIN adjudicated as a priced trade)
    undriv: dict = {"note": "insufficient measured stops"}
    if len(ok) >= 2:
        first, last = ok[0], ok[-1]
        u0 = first["per_class"]["s_units"][UNDRIV]
        u1 = last["per_class"]["s_units"][UNDRIV]
        floor = FL1_PER_CLASS_FLOOR_S[UNDRIV]
        agg = last["s_additive"] - first["s_additive"]
        undriv = {
            "class": "Undrivable", "class_index": UNDRIV,
            "burn_start_label": first["label"], "burn_end_label": last["label"],
            "s_start": u0, "s_end": u1, "s_delta_over_burn": u1 - u0,
            "direction": "ERODED (cost)" if u1 > u0 else "improved",
            "fl1_gt_flicker_floor_s": floor,
            "s_above_floor_end": u1 - floor,
            "fl1_total_reachable_headroom_s": XP1_EP641_PER_CLASS_S[UNDRIV] - floor,
            "aggregate_net_delta_s_over_burn": agg,
            "trade_verdict": (
                "PRICED TRADE HONOURED: the aggregate net delta is score-LOWERING and its "
                "magnitude exceeds the Undrivable cost" if (agg < 0 and abs(agg) > abs(u1 - u0))
                else "REVIEW: Undrivable cost is NOT dominated by the aggregate gain — MAIN "
                     "should re-price before the next continuation"),
            "xp1_ep641_anchor_s": XP1_EP641_PER_CLASS_S[UNDRIV],
            "routes_to": "cg1 (#809) trainer-side lambda_undriv, calibrated by THIS endpoint",
        }

    # --- endpoint per-class vs the two carried anchors (pre-burn residual + flicker floor)
    endpoint_vs_anchors: dict = {}
    if ok:
        last = ok[-1]
        endpoint_vs_anchors = {
            "class_names": list(CLASS_NAMES),
            "endpoint_s": last["per_class"]["s_units"],
            "xp1_ep641_s": list(XP1_EP641_PER_CLASS_S),
            "delta_vs_ep641_s": [last["per_class"]["s_units"][c] - XP1_EP641_PER_CLASS_S[c]
                                 for c in range(5)],
            "fl1_floor_s": list(FL1_PER_CLASS_FLOOR_S),
            "s_above_floor": [last["per_class"]["s_units"][c] - FL1_PER_CLASS_FLOOR_S[c]
                              for c in range(5)],
            "below_floor_classes": [CLASS_NAMES[c] for c in range(5)
                                    if last["per_class"]["s_units"][c] < FL1_PER_CLASS_FLOOR_S[c]],
            "note": "a class BELOW its smooth-label flicker floor is piercing that floor "
                    "(fl1 verdict_scope: formulation-scoped, NOT a hard bound)",
        }

    return {
        "schema": "ddm_b4r_endpoint_extras.v1",
        "final_window_dir": str(final_dir),
        "n_pairs": int(pairs),
        "CODE_PATH_SMOKE_NOT_A_MEASUREMENT": bool(pairs != N_PAIRS),
        "measurement_valid": bool(pairs == N_PAIRS),
        "class_order": list(CLASS_NAMES),
        "class_order_provenance": "comma10k CANONICAL [Road, Lane, Undrivable, Movable, MyCar] "
                                  "— MEASURED (CLAUDE.md non-negotiable); NEVER luma-sorted",
        "per_checkpoint": measured,
        "per_class_descent": descent,
        "undrivable_priced_over_burn": undriv,
        "endpoint_vs_anchors": endpoint_vs_anchors,
        "measured_stops": len(ok), "failed_stops": len(measured) - len(ok),
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir", type=Path)
    ap.add_argument("--root", type=Path, default=None,
                    help="burn root holding window_NN dirs (default: final_dir's parent)")
    ap.add_argument("--parent-ckpt", type=Path, default=None,
                    help="pre-burn parent checkpoint (the r1c ep641 endpoint)")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--pairs", type=int, default=N_PAIRS,
                    help="CODE-PATH SMOKE ONLY when != 600. n600 or it is NOT evidence "
                         "(CLAUDE.md allergic-to-non-n600-scale); sub-600 output is stamped "
                         "CODE_PATH_SMOKE_NOT_A_MEASUREMENT and measurement_valid=False.")
    args = ap.parse_args()
    if args.pairs < 1 or args.chunk < 1:
        ap.error("--pairs and --chunk must be >= 1")
    root = args.root or args.final_dir.parent
    out = build_extras(args.final_dir, root, args.parent_ckpt, args.chunk, args.pairs)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        tmp.replace(args.output_json)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
