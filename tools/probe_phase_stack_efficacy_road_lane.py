# SPDX-License-Identifier: MIT
"""$0 PHASE-STACK EFFICACY PROBE — the SPEC_v10 §14.5(b) decisive gate.

QUESTION (SPEC_v10 §14.2 curriculum-order law, UNMEASURED until now): does firing the
phase stack (#424 phase-advection / #360 Force-3 tie-locus conditioning + the #425
phase-carrier store leg) at organ-B's amplitude-OPEN ``Road->Lane`` strata REDUCE d_seg
through the real byte-closed decode (R -> frozen CPU-torch SegNet)?

MECHANISM (all REAL inputs; no surrogates):
  1. decode the BANKED v9c2 EMA through the canonical R decode
     (``tac.witness_control.factorized_features.snapshot_witness_margins`` -> real
     camera uint8 frames + frozen-SegNet flips + exact pairwise margins vs bit-exact GT
     ``lstars``);
  2. the phase stack's TARGET is the GT sub-pixel tie coordinate
     (:func:`tac.boundary_math.phase_primitives.gt_tie_targets_numpy` — op-for-op the
     #424/Force-3 target). Post-hoc on a frozen checkpoint, "fire the phase stack at a
     stratum" = realize the GT-phase correction at that stratum's flip pixels: the
     min-norm camera-space displacement that crosses the exact pairwise margin
     (``delta* = -s * m * g / ||g||^2`` — the SAME full-chain VJP convention as
     ``tac.witness_control.realization_regime`` / the necessity solver), APPLIED to the
     camera frame and ROUNDED TO uint8 (the realization constraint that decides whether
     phase moves survive);
  3. re-run the frozen SegNet on the treated frames (real ``preprocess_input``-parity
     resize) and RECOUNT d_seg vs GT — fixes AND collateral both counted.

READ THE VERDICT CAREFULLY (honesty contract):
  * This is the ORACLE-CEILING of the phase-conditioning lever at the targeted strata —
    the correction consumes GT (legal for a $0 gate measurement, NEVER shippable). If
    even the ceiling is FLAT through uint8+R, the §14.2 stage cannot pay at that
    stratum post-hoc; if it PAYS, the lever CAN pay and train-side/store-side must then
    realize it without the oracle.
  * Post-hoc-store vs train-side scoping: this probe applies MIN-NORM per-pixel moves.
    Sub-LSB deaths here bound the *amplitude-style post-hoc* path only; train-side
    joint descent can shape wider-support (>= 1 LSB, spread) patterns — exactly the
    honesty note in ``realization_regime`` (necessary-side indicator, not
    impossibility). The probe reports the organ's a_max sub-LSB prediction next to the
    REALIZED fix per pixel so the two channels are distinguishable.
  * Advisory ``[macOS-CPU advisory] NON-PROMOTABLE``; ``score_claim=False``; stratified
    labeled sample (the organ's stride convention); the pointer moves ONLY through
    ``upstream/evaluate.py``. MPS is NEVER touched.

Resumable (P0): per-frame results checkpoint to ``--out`` (tmp+rename) after every
frame; ``--resume`` skips frames already present (decode + sampling are deterministic).

Memory-safe co-run: refuses to start when available RAM < ``--min-free-gib`` (the live
c2 run is SACRED; read-only inputs; no full-n600 batch anywhere — pairs are decoded per
selection and scored frame-by-frame).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for p in (str(_REPO / "src"), str(_REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tac.boundary_math.phase_primitives import gt_tie_targets_numpy  # noqa: E402
from tac.witness_control.factorized_features import (  # noqa: E402
    AXIS_TAG,
    SCORER_HW,
    MarginSnapshot,
    default_pair_sample,
    load_frozen_segnet_cpu,
    locked_append_jsonl,
    oriented_key,
    snapshot_witness_margins,
    utc_stamp,
)
from tac.witness_control.realization_regime import (  # noqa: E402
    SUB_LSB_MAX_COORD,
    _assert_preprocess_parity,
    min_norm_crossing_max_coord,
)

STATE_JSONL = _REPO / ".omx" / "state" / "phase_stack_efficacy_probe.jsonl"
GT_CACHE_DEFAULT = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)


def _refuse_if_low_ram(min_free_gib: float) -> float:
    from tools.mem_basis import conservative_free_gib

    avail = conservative_free_gib()
    if avail < min_free_gib:
        raise SystemExit(
            f"REFUSE: available RAM {avail:.1f} GiB < floor {min_free_gib:.1f} GiB "
            "(live c2 run co-resident; probe must stay light)"
        )
    return avail


def load_gt_margins_slice(gt_cache: Path, pair_indices: list[int]) -> np.ndarray:
    """(N,H,W) float32 GT margin field for the selected pairs (transient full-array
    load, sliced immediately — the 472 MB member is freed before decode starts)."""
    z = np.load(gt_cache, allow_pickle=False)
    if "margins" not in z.files:
        raise KeyError(f"GT cache {gt_cache} lacks 'margins' (needed for the GT phase band)")
    mg = np.asarray(z["margins"])
    out = mg[np.asarray(pair_indices, dtype=np.int64)].astype(np.float32).copy()
    del mg
    return out


def gt_phase_band(lstars_pair: np.ndarray, margins_pair: np.ndarray, band: float) -> np.ndarray:
    """The GT phase straddle band (the #424/#425 addressable set) for one pair,
    dilated to include the straddle PARTNER pixels (q = right/down of the active p —
    a DERIVED choice, stated: the tie coordinate lives BETWEEN p and q, so both carry
    the phase)."""
    _t, _d, active = gt_tie_targets_numpy(lstars_pair, margins_pair, band=band)
    dil = active.copy()
    dil[:, 1:] |= active[:, :-1]   # partner right of an active p
    dil[1:, :] |= active[:-1, :]   # partner below an active p
    return dil


def isolation_pass(
    snap: MarginSnapshot,
    segnet,
    lstars: np.ndarray,
    sel_by_frame: dict[int, list[int]],
    *,
    n_isolate: int,
    scale: float,
    seed: int,
) -> dict:
    """Mechanism disambiguation: apply each pixel's min-norm phase move ALONE (no
    superposition) and count its own fix + collateral. Distinguishes 'the individual
    move spills' from 'the composed treatment spills'."""
    import torch
    import torch.nn.functional as tfun

    rng = np.random.default_rng(seed + 1)
    all_items = [(fi, i) for fi, items in sel_by_frame.items() for i in items]
    if len(all_items) > n_isolate:
        pick = rng.choice(len(all_items), size=n_isolate, replace=False)
        all_items = [all_items[int(k)] for k in pick]
    by_frame: dict[int, list[int]] = {}
    for fi, i in all_items:
        by_frame.setdefault(fi, []).append(i)

    rows: list[dict] = []
    for fi, items in sorted(by_frame.items()):
        frame = np.ascontiguousarray(snap.frames1[fi])
        base_wrong = snap.witness_argmax[fi] != lstars[fi]
        x_cam = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        x_cam.requires_grad_(True)
        x_s = tfun.interpolate(x_cam, size=SCORER_HW, mode="bilinear")
        logits = segnet(x_s)[0]
        for i in items:
            y, x = int(snap.flip_y[i]), int(snap.flip_x[i])
            w, g = int(snap.flip_wrong[i]), int(snap.flip_gt[i])
            m_t = logits[w, y, x] - logits[g, y, x]
            m = float(m_t.item())
            if x_cam.grad is not None:
                x_cam.grad = None
            m_t.backward(retain_graph=True)
            gr = x_cam.grad[0].detach().numpy()
            gn2 = float((gr * gr).sum())
            delta = -(scale * m / gn2) * np.transpose(gr, (1, 2, 0))
            treated = np.clip(np.rint(frame.astype(np.float64) + delta), 0, 255).astype(
                np.uint8
            )
            changed = int(np.count_nonzero(treated != frame))
            if changed == 0:
                rows.append({"fi": fi, "y": y, "x": x, "fixed": False,
                             "collateral": 0, "rounded_away": True})
                continue
            xt = torch.from_numpy(treated[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
            with torch.inference_mode():
                lg = segnet(segnet.preprocess_input(xt))[0].cpu().numpy()
            na = lg.argmax(axis=0).astype(np.int64)
            nw = na != lstars[fi]
            rows.append(
                {
                    "fi": fi, "y": y, "x": x,
                    "fixed": bool(not nw[y, x]),
                    "collateral": int(np.count_nonzero(nw & ~base_wrong)),
                    "others_fixed": int(
                        np.count_nonzero(~nw & base_wrong) - int(not nw[y, x])
                    ),
                    "rounded_away": False,
                }
            )
        del logits, x_s, x_cam
    n = len(rows)
    n_fix = sum(r["fixed"] for r in rows)
    n_round = sum(r.get("rounded_away", False) for r in rows)
    coll = [r["collateral"] for r in rows]
    others = [r.get("others_fixed", 0) for r in rows]
    return {
        "scale": scale,
        "n_isolated": n,
        "n_fixed": n_fix,
        "n_rounded_away": n_round,
        "fix_rate": (n_fix / n) if n else None,
        "collateral_mean": float(np.mean(coll)) if rows else None,
        "collateral_median": float(np.median(coll)) if rows else None,
        "others_fixed_mean": float(np.mean(others)) if rows else None,
        "net_flips_per_treated_px": (
            float(np.mean([(-1 if r["fixed"] else 0)
                           + r["collateral"] - r.get("others_fixed", 0) for r in rows]))
            if rows else None
        ),
        "rows": rows,
    }


def probe(args: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(max(1, int(args.torch_threads)))
    avail0 = _refuse_if_low_ram(args.min_free_gib)

    ema = Path(args.ema)
    gt_cache = Path(args.gt_cache)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = default_pair_sample(600, args.n_pairs)
    scales = [float(s) for s in args.cross_scales.split(",") if s.strip()]

    segnet = load_frozen_segnet_cpu()
    snap: MarginSnapshot = snapshot_witness_margins(
        ema, gt_cache, pairs, segnet_cpu=segnet, keep_frames=True, keep_argmax=True,
        run_ref=args.run_ref,
    )
    if not snap.frames1 or snap.witness_argmax is None:
        raise AssertionError("snapshot lacks frames/argmax — cannot probe (no surrogates)")

    gt_margins = load_gt_margins_slice(gt_cache, pairs)
    # re-load GT lstars slice aligned with the snapshot for band + recount
    from tac.witness_control.factorized_features import load_gt_slices

    lstars = load_gt_slices(gt_cache, pairs)["lstars"].astype(np.int64)

    # --- select target pixels per stratum -----------------------------------------
    want_all = args.strata.strip().lower() == "all"
    target_keys: set[str] = set()
    if want_all:
        for w in range(5):
            for g in range(5):
                if w != g:
                    target_keys.add(oriented_key(w, g))
    else:
        target_keys = {k.strip() for k in args.strata.split(",") if k.strip()}

    rng = np.random.default_rng(args.seed)
    sel_by_frame: dict[int, list[int]] = {}
    n_by_key: dict[str, int] = {}
    for key in sorted(target_keys):
        from tac.witness_control.factorized_features import parse_oriented_key

        w, g = parse_oriented_key(key)
        idx = np.nonzero((snap.flip_wrong == w) & (snap.flip_gt == g))[0]
        n_by_key[key] = int(idx.size)
        if idx.size == 0:
            continue
        if args.max_pixels_per_stratum and idx.size > args.max_pixels_per_stratum:
            idx = rng.choice(idx, size=args.max_pixels_per_stratum, replace=False)
        for i in idx:
            sel_by_frame.setdefault(int(snap.flip_pair_idx[i]), []).append(int(i))
    n_target = sum(len(v) for v in sel_by_frame.values())
    if n_target == 0:
        raise SystemExit(f"REFUSE (no-op guard): zero flip pixels in strata {sorted(target_keys)}")

    # --- resume state ---------------------------------------------------------------
    per_frame: dict[str, dict] = {}
    if args.resume and out_path.is_file():
        try:
            prior = json.loads(out_path.read_text())
            per_frame = dict(prior.get("per_frame", {}))
            print(f"[resume] {len(per_frame)} frames already done in {out_path}")
        except Exception as exc:
            print(f"[resume] could not load prior partial ({exc}); starting fresh")

    H, W = SCORER_HW

    def _flush(final: bool = False) -> dict:
        doc = {
            "schema": "phase_stack_efficacy_probe.v1",
            "generated_at": utc_stamp(),
            "ema_ckpt": str(ema),
            "ema_epoch": int(snap.ema_epoch),
            "gt_cache": str(gt_cache),
            "pair_indices": pairs,
            "n_pairs_sampled": len(pairs),
            "strata_targeted": sorted(target_keys),
            "n_flips_by_stratum_on_sample": n_by_key,
            "n_target_pixels": n_target,
            "cross_scales": scales,
            "gt_band": args.gt_band,
            "baseline_d_seg_sample": snap.d_seg_sample,
            "sub_lsb_max_coord": SUB_LSB_MAX_COORD,
            "per_frame": per_frame,
            "axis_tag": AXIS_TAG,
            "score_claim": False,
            "oracle_ceiling": True,
            "complete": bool(final),
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, out_path)
        return doc

    import torch.nn.functional as tfun

    for fi, items in sorted(sel_by_frame.items()):
        fkey = str(fi)
        if fkey in per_frame:
            continue
        frame = np.ascontiguousarray(snap.frames1[fi])            # (874,1164,3) uint8
        band = gt_phase_band(lstars[fi], gt_margins[fi], args.gt_band)

        x_cam = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        x_cam.requires_grad_(True)
        _assert_preprocess_parity(segnet, x_cam)
        x_s = tfun.interpolate(x_cam, size=SCORER_HW, mode="bilinear")
        logits = segnet(x_s)[0]

        deltas = {s: np.zeros((874, 1164, 3), dtype=np.float64) for s in scales}
        px_rows: list[dict] = []
        for i in items:
            y, x = int(snap.flip_y[i]), int(snap.flip_x[i])
            w, g = int(snap.flip_wrong[i]), int(snap.flip_gt[i])
            m_t = logits[w, y, x] - logits[g, y, x]
            m = float(m_t.item())
            if abs(m - float(snap.flip_margin[i])) > 1e-3:
                raise AssertionError(
                    f"graph margin {m:.6f} != snapshot margin "
                    f"{float(snap.flip_margin[i]):.6f} at frame-slot {fi} px({y},{x}) — "
                    "stale snapshot; refusing"
                )
            if x_cam.grad is not None:
                x_cam.grad = None
            m_t.backward(retain_graph=True)
            gr = x_cam.grad[0].detach().numpy()                    # (3,874,1164)
            a_max, flipdist_l2 = min_norm_crossing_max_coord(m, gr)
            gn2 = float((gr * gr).sum())
            base = -(m / gn2) * np.transpose(gr, (1, 2, 0))        # (874,1164,3) min-norm
            for s in scales:
                deltas[s] += s * base
            px_rows.append(
                {
                    "i": int(i), "y": y, "x": x, "key": oriented_key(w, g),
                    "margin": m, "a_max": float(a_max),
                    "sub_lsb_pred": bool(a_max < SUB_LSB_MAX_COORD),
                    "on_gt_phase_band": bool(band[y, x]),
                    "flipdist_l2_cam": float(flipdist_l2),
                }
            )
        del logits, x_s, x_cam

        # apply + realize (uint8) + re-score per scale
        frame_res: dict = {"n_treated": len(px_rows), "pixels": px_rows, "scales": {}}
        base_wrong = snap.witness_argmax[fi] != lstars[fi]
        for s in scales:
            treated = np.clip(np.rint(frame.astype(np.float64) + deltas[s]), 0, 255).astype(
                np.uint8
            )
            changed = int(np.count_nonzero(treated != frame))
            if changed == 0:
                # loud, not silent: the whole treatment rounded away (all sub-LSB)
                frame_res["scales"][str(s)] = {
                    "bytes_changed": 0, "note": "entire delta rounded away under uint8",
                    "d_seg_frame_before": float(base_wrong.mean()),
                    "d_seg_frame_after": float(base_wrong.mean()),
                    "n_fixed": 0, "n_collateral_new": 0,
                    "fixed_flags": [False] * len(px_rows),
                }
                continue
            xt = torch.from_numpy(treated[None, None]).permute(0, 1, 4, 2, 3).contiguous().float()
            with torch.inference_mode():
                lg = segnet(segnet.preprocess_input(xt))[0].cpu().numpy()
            new_arg = lg.argmax(axis=0).astype(np.int64)
            new_wrong = new_arg != lstars[fi]
            fixed_flags = [
                bool(not new_wrong[r["y"], r["x"]]) for r in px_rows
            ]
            treated_mask = np.zeros((H, W), dtype=bool)
            for r in px_rows:
                treated_mask[r["y"], r["x"]] = True
            coll_mask = new_wrong & ~base_wrong
            collateral_new = int(np.count_nonzero(coll_mask))
            # WHERE the collateral lands (oriented stratum of each NEW flip) — is the
            # spill Lane-adjacent phase jitter (boundary moved elsewhere) or foreign?
            coll_by_key: dict[str, int] = {}
            if collateral_new:
                cy, cx = np.nonzero(coll_mask)
                cw = new_arg[cy, cx]
                cg = lstars[fi][cy, cx]
                for wv, gv in zip(cw.tolist(), cg.tolist(), strict=True):
                    k = oriented_key(int(wv), int(gv))
                    coll_by_key[k] = coll_by_key.get(k, 0) + 1
            untargeted_fixed = int(
                np.count_nonzero(~new_wrong & base_wrong & ~treated_mask)
            )
            frame_res["scales"][str(s)] = {
                "bytes_changed": changed,
                "d_seg_frame_before": float(base_wrong.mean()),
                "d_seg_frame_after": float(new_wrong.mean()),
                "n_fixed": int(sum(fixed_flags)),
                "n_collateral_new": collateral_new,
                "collateral_by_stratum": coll_by_key,
                "n_untargeted_also_fixed": untargeted_fixed,
                "fixed_flags": fixed_flags,
            }
        per_frame[fkey] = frame_res
        _flush()
        print(
            f"[frame {fi}] treated {len(px_rows)} px; "
            + "; ".join(
                f"s={s}: fixed {frame_res['scales'][str(s)].get('n_fixed', 0)}"
                f"/{len(px_rows)}, collateral +{frame_res['scales'][str(s)].get('n_collateral_new', 0)}"
                for s in scales
            ),
            flush=True,
        )

    doc = _flush(final=True)

    # ---- aggregate ------------------------------------------------------------------
    agg: dict = {"schema": "phase_stack_efficacy_probe_summary.v1"}
    base_dseg = snap.d_seg_sample
    for s in scales:
        skey = str(s)
        n_treat = n_fixed = n_coll = n_unt = 0
        n_onband = n_onband_fixed = n_sublsb = n_sublsb_fixed = 0
        coll_by_key_total: dict[str, int] = {}
        dseg_after_frames = []
        dseg_before_frames = []
        for fr in per_frame.values():
            sc = fr["scales"].get(skey)
            if sc is None:
                continue
            flags = sc["fixed_flags"]
            n_treat += len(flags)
            n_fixed += sum(bool(f) for f in flags)
            n_coll += sc.get("n_collateral_new", 0)
            n_unt += sc.get("n_untargeted_also_fixed", 0)
            for k, v in (sc.get("collateral_by_stratum") or {}).items():
                coll_by_key_total[k] = coll_by_key_total.get(k, 0) + int(v)
            dseg_after_frames.append(sc["d_seg_frame_after"])
            dseg_before_frames.append(sc["d_seg_frame_before"])
            for r, f in zip(fr["pixels"], flags, strict=True):
                if r["on_gt_phase_band"]:
                    n_onband += 1
                    n_onband_fixed += bool(f)
                if r["sub_lsb_pred"]:
                    n_sublsb += 1
                    n_sublsb_fixed += bool(f)
        # exact recomposition: sample d_seg = mean over frames of per-frame flip rate;
        # untouched frames keep their baseline per-frame rate (per-pair separable
        # objective, evaluate.py:96).
        base_by_frame = {
            str(fi): float((snap.witness_argmax[fi] != lstars[fi]).mean())
            for fi in range(len(pairs))
        }
        total_after = 0.0
        for fi in range(len(pairs)):
            fr = per_frame.get(str(fi))
            if fr is not None and str(s) in fr["scales"]:
                total_after += fr["scales"][skey]["d_seg_frame_after"]
            else:
                total_after += base_by_frame[str(fi)]
        d_seg_after = total_after / len(pairs)
        agg[skey] = {
            "n_treated": n_treat,
            "collateral_by_stratum": dict(
                sorted(coll_by_key_total.items(), key=lambda kv: -kv[1])
            ),
            "n_fixed": n_fixed,
            "fix_rate": (n_fixed / n_treat) if n_treat else None,
            "n_collateral_new": n_coll,
            "n_untargeted_also_fixed": n_unt,
            "on_band": {"n": n_onband, "fixed": n_onband_fixed},
            "sub_lsb_pred": {"n": n_sublsb, "fixed": n_sublsb_fixed},
            "d_seg_baseline_sample": base_dseg,
            "d_seg_treated_sample": d_seg_after,
            "delta_d_seg_sample": d_seg_after - base_dseg,
            "delta_d_seg_pct": 100.0 * (d_seg_after - base_dseg) / base_dseg,
        }

    if args.isolate_n > 0:
        print(f"[isolation] measuring {args.isolate_n} single-pixel treatments "
              f"at scale {args.isolate_scale} ...", flush=True)
        doc["isolation"] = isolation_pass(
            snap, segnet, lstars, sel_by_frame,
            n_isolate=args.isolate_n, scale=args.isolate_scale, seed=args.seed,
        )

    doc["summary"] = agg
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, out_path)

    row = {
        "schema": "phase_stack_efficacy_probe_row.v1",
        "generated_at": utc_stamp(),
        "ema_ckpt": str(ema),
        "ema_epoch": int(snap.ema_epoch),
        "pairs": len(pairs),
        "strata": sorted(target_keys),
        "summary": {k: v for k, v in agg.items() if k != "schema"},
        "artifact": str(out_path),
        "axis_tag": AXIS_TAG,
        "score_claim": False,
        "oracle_ceiling": True,
    }
    if "isolation" in doc:
        row["isolation"] = {
            k: v for k, v in doc["isolation"].items() if k != "rows"
        }
    locked_append_jsonl(args.state or STATE_JSONL, row)

    import psutil

    peak = psutil.Process().memory_info().rss / 2**30
    print(f"\n=== SUMMARY (baseline d_seg {base_dseg:.6f} on n{len(pairs)} stride sample) ===")
    for s in scales:
        a = agg[str(s)]
        print(
            f" scale {s}: fixed {a['n_fixed']}/{a['n_treated']} "
            f"(fix_rate {a['fix_rate']:.3f}), collateral +{a['n_collateral_new']}, "
            f"d_seg {a['d_seg_baseline_sample']:.6f} -> {a['d_seg_treated_sample']:.6f} "
            f"(Δ {a['delta_d_seg_sample']:+.6f}, {a['delta_d_seg_pct']:+.1f}%) "
            f"| on-band fixed {a['on_band']['fixed']}/{a['on_band']['n']} "
            f"| sub-LSB-pred fixed {a['sub_lsb_pred']['fixed']}/{a['sub_lsb_pred']['n']}"
        )
    print(f" probe RSS now {peak:.1f} GiB (avail at start {avail0:.1f} GiB) {AXIS_TAG}")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ema", required=True, help="banked EMA npz (read-only)")
    ap.add_argument("--gt-cache", default=str(GT_CACHE_DEFAULT))
    ap.add_argument("--n-pairs", type=int, default=24,
                    help="stride sample size over the 600 scored pairs (organ convention)")
    ap.add_argument("--strata", default="Road->Lane",
                    help="comma-separated oriented keys (e.g. 'Road->Lane') or 'all'")
    ap.add_argument("--max-pixels-per-stratum", type=int, default=0,
                    help="0 = treat ALL flip pixels of each targeted stratum")
    ap.add_argument("--cross-scales", default="1.05,2.0",
                    help="margin-crossing scale arms (1.05=minimal, 2.0=firm/mirror)")
    ap.add_argument("--gt-band", type=float, default=1.0,
                    help="GT straddle band threshold (the #424 target convention)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--isolate-n", type=int, default=0,
                    help="if >0, also measure N SINGLE-pixel treatments (no superposition)")
    ap.add_argument("--isolate-scale", type=float, default=1.5)
    ap.add_argument("--torch-threads", type=int, default=3)
    ap.add_argument("--min-free-gib", type=float, default=14.0)
    ap.add_argument("--run-ref", default="v9c2_defensive_bank_20260718")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    probe(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
