# SPDX-License-Identifier: MIT
"""MATH-OPTIMAL reverse-waterfill A/B — does the KKT allocation flip the net NET-POSITIVE?

THE FINDING THIS IMPROVES ON. The crude variable-level-codec probe
(``probe_variable_level_codec_byte_distortion.py``) used a uniform ``min_level_ratio``
knob that coarsens ~27/28 tensors to the same fraction. MEASURED on the real basin EMA +
real frozen SegNet/PoseNet: byte −789..−1721 B (real, survives inflate) but advisory net
S Δ = **+0.001053 / +0.006030 (WORSE)** — the uniform allocation coarsens pose-sensitive
tensors it shouldn't. The more-aggressive arm had the SMALLER penalty, a strong sign the
allocation is mis-shaped.

THE MATH-OPTIMAL FIX (this probe). Instead of a uniform band, this probe:

 1. MEASURES per-tensor RD curves on the REAL frozen scorer: for each weight tensor t and
    each coarser level n in {96,64,48,32,16}, build the deployed decoder blob with ONLY t
    at n (all others at 127), dequant, score the advisory d_seg/d_pose on the SAME real
    pairs, and record ``(byte_saving_t(n), dist_cost_t(n))`` (cumulative vs the all-127
    baseline). This is the empirical bit-spend + distortion table (Catalog #304 — NOT a
    closed-form assumption).
 2. SOLVES the KKT reverse-waterfill (``solve_waterfill_allocation``): pick each tensor's
    n_levels to MINIMIZE total advisory distortion subject to a byte target — equivalently
    take the cheapest marginal ``dΔS_dist/dByte`` step next, stop where it exceeds the byte
    value 25/N. At the optimum the marginal is EQUALIZED across coarsened tensors (the
    Lagrange-multiplier / reverse-waterfilling condition).
 3. MEASURES the net at the optimal allocation on the real scorer: byte-close the FULL
    deployed blob at the waterfill levels, dequant, score, report deployed byte Δ AND
    advisory net S Δ.
 4. Sweeps the byte target to trace the RD frontier and reports the BEST net.

ROBUSTNESS (the prompt's Lens-3). The crude probe's +0.001-vs-+0.006 inversion warns
d_pose is noisy on small slices. This probe RE-MEASURES the chosen allocation's net on a
SECOND independent slice (``--confirm-slice``) and reports both; the verdict is taken from
the slice-robust net (the worse of the two, conservatively).

VERDICT (the operator's "is it net-positive?" question):
  * NET_POSITIVE_AT_$0 — the math-optimal allocation makes advisory net S Δ < 0 with NO
    retrain on BOTH slices -> a ready $0 rate win; recommend the driver wire-in (deferred).
  * STILL_NEAR_BREAKEVEN — the optimum is closer to 0 than the crude probe but not
    net-negative (or not robustly so) -> report the best net + the gap, recommend folding
    the variable grid into the curriculum's QAT stages (train coarse-grid-robust).

Authority: $0 / local / torch-CPU (TRUSTED for advisory). Every number is
``[macOS-CPU advisory]`` NON-PROMOTABLE — NOT a 600-pair contest eval. MPS is NEVER used
for any score.

Usage::

    .venv/bin/python experiments/probe_variable_level_waterfill_net.py --rd-pairs 16 --confirm-pairs 16
    .venv/bin/python experiments/probe_variable_level_waterfill_net.py --rd-pairs 12 --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

_BASIN_CKPT = (
    _REPO_ROOT
    / "experiments/results/forkpoints/basin_bc20_20260612T121523Z"
    / "torch_vehicle_checkpoint_state.pt"
)
_VIDEO = _REPO_ROOT / "upstream/videos/0.mkv"
_EVAL_H, _EVAL_W = 384, 512
_N = 37_545_489


def _contest_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose + 1e-12) + 25.0 * archive_bytes / _N


def _decode_and_score(sd, latents, ctx, n, model, idx):
    """Advisory d_seg (argmax-flip RATE vs GT) + d_pose (MSE on 6 pose dims) for a
    given DEPLOYED state dict over the pair indices ``idx``. Real frozen scorer, no MPS."""
    dec = model.HNeRVDecoder(latent_dim=28, base_channels=20, eval_size=(_EVAL_H, _EVAL_W))
    dec.load_state_dict(sd)
    with torch.no_grad():
        decoded_pair = dec(latents[idx])
        nn_ = len(idx)
        flat = decoded_pair.reshape(nn_ * 2, 3, _EVAL_H, _EVAL_W)
        up = F.interpolate(flat, size=(874, 1164), mode="bicubic", align_corners=False)
        down = F.interpolate(up, size=(_EVAL_H, _EVAL_W), mode="bilinear", align_corners=False)
        decoded_bhwc = down.reshape(nn_, 2, 3, _EVAL_H, _EVAL_W).permute(0, 1, 3, 4, 2)
        decoded_bhwc = decoded_bhwc.clamp(0, 255).round()
        net = ctx.distortion_net
        posenet_in, segnet_in = net.preprocess_input(decoded_bhwc)
        seg_out = net.segnet(segnet_in)
        pose6 = net.posenet(posenet_in)["pose"][:, :6]
        d_seg = float((seg_out.argmax(dim=1) != ctx.seg_targets_hard[idx]).float().mean().item())
        d_pose = float(F.mse_loss(pose6, ctx.pose_targets[idx]).item())
    return d_seg, d_pose


def _measure_rd_curves(sd, latents, ctx, n, model, codec, level_grid, idx, *, verbose):
    """MEASURE the per-tensor RD curve on the real scorer.

    For each weight tensor t and each coarser level n in level_grid (excluding 127),
    build the DEPLOYED decoder blob with ONLY t at n, dequant + score, and record the
    cumulative ``(byte_saving, dist_cost)`` vs the all-127 baseline (deployed byte saving
    = vendored_blob_bytes − blob_with_t_coarsened; dist_cost = ΔS_dist vs baseline).

    Returns ``(rd_table, baseline)`` where ``baseline`` is the (d_seg, d_pose, score,
    archive_bytes) at all-127 (vendored deployed) and ``rd_table`` maps tensor name ->
    {n_levels: (byte_saving, dist_cost)} with the 127 entry (0,0).
    """
    from tac.losses.variable_level_codec import (
        build_decoder_blob_variable_or_vendored,
        decode_decoder_variable,
    )
    from tac.losses.variable_level_waterfill_allocator import sqrt_pose_term

    names = list(sd.keys())
    weight_keys = [k for k in names if k.endswith(".weight")]

    # full-archive constants (so byte saving is the deployed full-archive saving)
    import brotli
    meta = {"base_channels": 20, "latent_dim": 28}
    lat_brotli_len = len(brotli.compress(codec.encode_latents(latents), quality=11))
    meta_brotli_len = len(brotli.compress(json.dumps(meta).encode(), quality=11))

    # --- baseline: vendored all-127 deployed ---
    vend_blob = codec.encode_decoder(codec.quantize_state_dict(sd))
    vend_archive = 4 + meta_brotli_len + 4 + len(vend_blob) + 4 + lat_brotli_len
    vend_sd = codec.decode_decoder(vend_blob)
    base_dseg, base_dpose = _decode_and_score(vend_sd, latents, ctx, n, model, idx)
    base_sqrt = sqrt_pose_term(base_dpose)
    baseline = {
        "d_seg": base_dseg, "d_pose": base_dpose,
        "score": _contest_score(base_dseg, base_dpose, vend_archive),
        "archive_bytes": vend_archive, "decoder_blob_bytes": len(vend_blob),
    }

    coarser_levels = [lv for lv in sorted(set(level_grid), reverse=True) if lv < 127]
    rd_table: dict[str, dict[int, tuple[float, float]]] = {}
    t0 = time.time()
    n_meas = 0
    for ki, wk in enumerate(weight_keys):
        curve: dict[int, tuple[float, float]] = {127: (0.0, 0.0)}
        for lv in coarser_levels:
            levels = dict.fromkeys(names, 127)
            levels[wk] = lv
            blob, is_var = build_decoder_blob_variable_or_vendored(sd, levels)
            blob_archive = 4 + meta_brotli_len + 4 + len(blob) + 4 + lat_brotli_len
            dep_sd = decode_decoder_variable(blob) if is_var else codec.decode_decoder(blob)
            dseg, dpose = _decode_and_score(dep_sd, latents, ctx, n, model, idx)
            # deployed byte saving vs vendored (positive = smaller)
            byte_saving = float(vend_archive - blob_archive)
            # advisory dist cost vs baseline (ΔS_dist = 100*Δd_seg + Δsqrt(10*d_pose))
            dist_cost = 100.0 * (dseg - base_dseg) + (sqrt_pose_term(dpose) - base_sqrt)
            curve[lv] = (byte_saving, dist_cost)
            n_meas += 1
        rd_table[wk] = curve
        if verbose:
            elapsed = time.time() - t0
            print(f"  RD curve {ki+1}/{len(weight_keys)} {wk[:40]:40s} "
                  f"({n_meas} measured, {elapsed:.0f}s)", flush=True)
    return rd_table, baseline, vend_blob, (meta_brotli_len, lat_brotli_len)


def _net_at_allocation(sd, latents, ctx, n, model, codec, levels, idx, blob_consts):
    """MEASURE the advisory net at a specific allocation on the real scorer (deployed)."""
    from tac.losses.variable_level_codec import (
        build_decoder_blob_variable_or_vendored,
        decode_decoder_variable,
    )
    meta_brotli_len, lat_brotli_len = blob_consts
    blob, is_var = build_decoder_blob_variable_or_vendored(sd, levels)
    archive = 4 + meta_brotli_len + 4 + len(blob) + 4 + lat_brotli_len
    dep_sd = decode_decoder_variable(blob) if is_var else codec.decode_decoder(blob)
    dseg, dpose = _decode_and_score(dep_sd, latents, ctx, n, model, idx)
    score = _contest_score(dseg, dpose, archive)
    return {
        "d_seg": dseg, "d_pose": dpose, "score": score,
        "archive_bytes": archive, "decoder_blob_bytes": len(blob),
        "is_variable_format": bool(is_var),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rd-pairs", type=int, default=16,
                    help="real pairs for the RD-curve measurement + primary net (be contention-aware)")
    ap.add_argument("--confirm-pairs", type=int, default=16,
                    help="real pairs for the SECOND slice (d_pose-noise robustness); 0 to skip")
    ap.add_argument("--confirm-offset", type=int, default=300,
                    help="start the confirm slice at this pair offset (independent of the RD slice)")
    ap.add_argument("--byte-targets", type=str, default="",
                    help="comma byte targets for the RD frontier sweep; empty = net-stop only")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not _BASIN_CKPT.exists():
        raise FileNotFoundError(f"basin checkpoint not found: {_BASIN_CKPT}")

    from tac.losses.variable_level_waterfill_allocator import (
        DEFAULT_LEVEL_GRID,
        solve_waterfill_allocation,
        verify_kkt_marginal_equalization,
    )
    from tac.torch_vehicle.scorer_context import RealScorerContext
    from tac.torch_vehicle.vendored_imports import import_vendored

    codec = import_vendored("codec")
    model = import_vendored("model")

    ck = torch.load(_BASIN_CKPT, map_location="cpu", weights_only=False)
    sd = {k: v.detach().float() for k, v in ck["ema_decoder"].items()}
    latents = ck["ema_latents"].detach().float()

    # The RD slice (also the primary net slice). max_pairs caps the precompute.
    rd_n = args.rd_pairs
    confirm_n = args.confirm_pairs
    total_needed = max(rd_n, args.confirm_offset + confirm_n if confirm_n > 0 else rd_n)
    ctx = RealScorerContext(
        str(_VIDEO), device="cpu", max_pairs=total_needed,
        targets_cache=str(_REPO_ROOT / ".omx/tmp/waterfill_probe_targets"),
    )
    n = ctx.n_pairs
    rd_n = min(rd_n, n)
    rd_idx = torch.arange(rd_n)

    if not args.json:
        print("=" * 80)
        print("MATH-OPTIMAL REVERSE-WATERFILL (KKT) — real-scorer RD curves + net A/B")
        print("=" * 80)
        print(f"basin: {_BASIN_CKPT.relative_to(_REPO_ROOT)}  ({n} pairs available)")
        print(f"RD/primary slice: pairs [0:{rd_n}]")
        print("measuring per-tensor RD curves on the real frozen scorer ...", flush=True)

    rd_table, baseline, _vend_blob, blob_consts = _measure_rd_curves(
        sd, latents, ctx, n, model, codec, DEFAULT_LEVEL_GRID, rd_idx,
        verbose=not args.json,
    )

    # Persist the (expensive) measured RD table so a re-solve (e.g. after an allocator fix)
    # is FREE — the RD measurement is the cost; re-solving + re-verifying KKT is instant.
    rd_dump = _REPO_ROOT / ".omx/research" / f"track_a_itemB_waterfill_rd_table_rd{rd_n}_20260612.json"
    try:
        rd_dump.write_text(json.dumps(
            {t: {str(k): list(v) for k, v in curve.items()} for t, curve in rd_table.items()},
            indent=2,
        ))
    except OSError:
        pass

    # --- solve the net-maximizing waterfill ---
    alloc = solve_waterfill_allocation(rd_table, net_stop=True)
    kkt_holds, kkt_msg = verify_kkt_marginal_equalization(alloc)

    # MEASURE the primary-slice net at the waterfill allocation (re-decode the FULL deployed
    # blob, not the per-tensor marginal sum — the deployed bytes are the truth).
    wf_primary = _net_at_allocation(
        sd, latents, ctx, n, model, codec, alloc.levels, rd_idx, blob_consts,
    )
    primary_net = wf_primary["score"] - baseline["score"]
    primary_byte = wf_primary["archive_bytes"] - baseline["archive_bytes"]

    # --- SECOND slice index (d_pose-noise robustness) — built FIRST so the frontier sweep
    #     can re-measure EACH byte target on BOTH slices (the operating-point honesty fix:
    #     a target that is net-negative on the RD/primary slice can REGRESS on an unseen
    #     slice because the RD table is fit on the primary slice; the robust operating point
    #     is the target whose WORST slice is most negative, NOT the RD-table prediction). ---
    have_confirm = confirm_n > 0 and args.confirm_offset + confirm_n <= n
    c_idx = (
        torch.arange(args.confirm_offset, args.confirm_offset + confirm_n)
        if have_confirm else None
    )

    # --- RD frontier sweep over byte targets (if requested) — measured on BOTH slices ---
    frontier = []
    targets = [float(x) for x in args.byte_targets.split(",") if x.strip()] if args.byte_targets else []
    for tgt in targets:
        a_t = solve_waterfill_allocation(rd_table, byte_target=tgt, net_stop=False)
        m_t = _net_at_allocation(sd, latents, ctx, n, model, codec, a_t.levels, rd_idx, blob_consts)
        row = {
            "byte_target": tgt,
            "achieved_byte_delta": m_t["archive_bytes"] - baseline["archive_bytes"],
            "primary_net_score_delta": round(m_t["score"] - baseline["score"], 6),
            "n_coarsened": a_t.n_coarsened,
        }
        if have_confirm:
            cb_t = _net_at_allocation(
                sd, latents, ctx, n, model, codec, dict.fromkeys(sd, 127), c_idx, blob_consts,
            )
            cw_t = _net_at_allocation(
                sd, latents, ctx, n, model, codec, a_t.levels, c_idx, blob_consts,
            )
            row["confirm_net_score_delta"] = round(cw_t["score"] - cb_t["score"], 6)
            row["worst_slice_net"] = round(
                max(row["primary_net_score_delta"], row["confirm_net_score_delta"]), 6,
            )
        frontier.append(row)

    # --- SECOND slice (d_pose-noise robustness): re-measure the net-stop allocation ---
    confirm = None
    if have_confirm:
        # baseline on the confirm slice (all-127 dict routes through the byte-identical
        # vendored path inside _net_at_allocation — no separate decode needed).
        c_base = _net_at_allocation(
            sd, latents, ctx, n, model, codec, dict.fromkeys(sd, 127), c_idx, blob_consts,
        )
        c_wf = _net_at_allocation(
            sd, latents, ctx, n, model, codec, alloc.levels, c_idx, blob_consts,
        )
        confirm = {
            "confirm_slice": f"[{args.confirm_offset}:{args.confirm_offset+confirm_n}]",
            "baseline_score": round(c_base["score"], 6),
            "waterfill_score": round(c_wf["score"], 6),
            "net_score_delta": round(c_wf["score"] - c_base["score"], 6),
            "byte_delta": c_wf["archive_bytes"] - c_base["archive_bytes"],
            "baseline_d_seg": round(c_base["d_seg"], 6),
            "waterfill_d_seg": round(c_wf["d_seg"], 6),
            "baseline_d_pose": round(c_base["d_pose"], 8),
            "waterfill_d_pose": round(c_wf["d_pose"], 8),
        }

    # --- verdict (slice-robust: worse of the two nets, conservatively) ---
    # The net_stop allocation's slice-robust net (the worse of the two slices).
    nets = [primary_net]
    if confirm is not None:
        nets.append(confirm["net_score_delta"])
    worst_net = max(nets)  # most-positive (worst) net across slices for net_stop
    best_net = min(nets)
    net_stop_robust_negative = all(x < 0 for x in nets)

    # The ROBUST OPERATING POINT from the byte-target frontier: the target whose WORST
    # slice net is most negative. This is the honest wire-in operating point — net_stop
    # over-spends on the RD-table prediction (fit on the primary slice), so its WORST slice
    # can REGRESS even when the predicted net is deeply negative. The frontier-with-confirm
    # picks the operating point that GENERALIZES (worst slice still net-negative).
    robust_op = None
    if frontier and all("worst_slice_net" in r for r in frontier):
        # candidate rows whose worst slice is net-negative (a real robust win)
        neg_rows = [r for r in frontier if r["worst_slice_net"] < 0.0]
        if neg_rows:
            robust_op = min(neg_rows, key=lambda r: r["worst_slice_net"])
        else:
            # no robust-negative target: report the least-bad (most-negative worst) anyway
            robust_op = min(frontier, key=lambda r: r["worst_slice_net"])

    # The verdict is robust-positive ONLY IF a frontier operating point exists whose WORST
    # slice is net-negative (when a confirm slice is present). Fall back to the net_stop
    # both-slices check when no frontier was swept.
    if frontier and confirm is not None:
        robust_negative = robust_op is not None and robust_op["worst_slice_net"] < 0.0
    else:
        robust_negative = net_stop_robust_negative

    if robust_negative:
        verdict = "NET_POSITIVE_AT_$0"
        recommendation = (
            "ready $0 rate win at the ROBUST OPERATING POINT (the byte target whose WORST "
            "slice is net-negative) — recommend driver wire-in at that conservative target "
            "(DEFERRED, next gate). NOTE: net_stop OVER-SPENDS (RD table fit on primary "
            "slice) — wire the conservative byte target, not net_stop."
        )
    else:
        verdict = "STILL_NEAR_BREAKEVEN"
        recommendation = (
            "fold the variable grid into the curriculum QAT stages (train coarse-grid-robust) — "
            "path (b); the byte lever is real but the no-retrain net is not robustly negative "
            "at any byte target (the confirm slice regresses)"
        )

    crude_net_05, crude_net_075 = 0.001053, 0.006030  # the crude probe's measured nets
    result = {
        "probe": "variable_level_waterfill_net",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE — RD curves + net on real frozen scorer, NOT a 600-pair contest eval",
        "basin_checkpoint": str(_BASIN_CKPT.relative_to(_REPO_ROOT)),
        "rd_pairs": rd_n,
        "level_grid": list(DEFAULT_LEVEL_GRID),
        "baseline_score": round(baseline["score"], 6),
        "baseline_d_seg": round(baseline["d_seg"], 6),
        "baseline_d_pose": round(baseline["d_pose"], 8),
        "baseline_archive_bytes": baseline["archive_bytes"],
        "waterfill_levels_coarsened": alloc.n_coarsened,
        "waterfill_levels": {k: v for k, v in alloc.levels.items() if v < 127},
        "waterfill_total_byte_saving_predicted": round(alloc.total_byte_saving, 1),
        "waterfill_net_predicted_from_rd_table": round(alloc.net_score_delta, 6),
        "kkt_marginal_equalization_holds": bool(kkt_holds),
        "kkt_explanation": kkt_msg,
        "kkt_byte_value_per_byte": alloc.byte_value_per_byte,
        "kkt_max_accepted_ratio": alloc.kkt_max_accepted_ratio,
        "kkt_min_rejected_ratio": alloc.kkt_min_rejected_ratio,
        "primary_net_score_delta_MEASURED": round(primary_net, 6),
        "primary_byte_delta_MEASURED": primary_byte,
        "primary_d_seg": round(wf_primary["d_seg"], 6),
        "primary_d_pose": round(wf_primary["d_pose"], 8),
        "confirm_slice": confirm,
        "rd_frontier": frontier,
        "net_stop_slice_robust_best_net": round(best_net, 6),
        "net_stop_slice_robust_worst_net": round(worst_net, 6),
        "net_stop_robust_negative": bool(net_stop_robust_negative),
        "robust_operating_point": robust_op,
        "crude_probe_net_ratio050": crude_net_05,
        "crude_probe_net_ratio075": crude_net_075,
        "improvement_vs_crude_best": round(crude_net_05 - best_net, 6),
        "verdict": verdict,
        "recommendation": recommendation,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print(f"  baseline (vendored-127): score {baseline['score']:.6f}  "
              f"d_seg {baseline['d_seg']:.5f}  d_pose {baseline['d_pose']:.6f}  "
              f"archive {baseline['archive_bytes']} B")
        print(f"  waterfill coarsened {alloc.n_coarsened} tensors; predicted net (RD table) "
              f"{alloc.net_score_delta:+.6f}")
        print(f"  KKT marginal-equalization holds: {kkt_holds}")
        print(f"    {kkt_msg}")
        print()
        print(f"  PRIMARY slice [0:{rd_n}]: net {primary_net:+.6f}  byte Δ {primary_byte:+d} B "
              f"(d_seg {wf_primary['d_seg']:.5f}, d_pose {wf_primary['d_pose']:.6f})")
        if confirm is not None:
            print(f"  CONFIRM slice {confirm['confirm_slice']}: net {confirm['net_score_delta']:+.6f}  "
                  f"byte Δ {confirm['byte_delta']:+d} B "
                  f"(d_seg {confirm['waterfill_d_seg']:.5f}, d_pose {confirm['waterfill_d_pose']:.6f})")
        if frontier:
            print("  RD frontier (byte target -> primary / confirm net) [robust = worst slice]:")
            for f in frontier:
                conf = f.get("confirm_net_score_delta")
                worst = f.get("worst_slice_net")
                conf_s = f" confirm {conf:+.6f} worst {worst:+.6f}" if conf is not None else ""
                print(f"    target {f['byte_target']:>7.0f} B -> achieved {f['achieved_byte_delta']:+d} B, "
                      f"primary {f['primary_net_score_delta']:+.6f}{conf_s}, {f['n_coarsened']} coarsened")
            if robust_op is not None:
                print(f"  ROBUST OPERATING POINT: target {robust_op['byte_target']:.0f} B -> "
                      f"achieved {robust_op['achieved_byte_delta']:+d} B, worst-slice net "
                      f"{robust_op['worst_slice_net']:+.6f} ({robust_op['n_coarsened']} coarsened)")
        print()
        print(f"  vs CRUDE probe: best slice net {best_net:+.6f} vs crude +0.001053 "
              f"(improvement {crude_net_05 - best_net:+.6f})")
        print()
        print(f"VERDICT: {verdict}")
        print(f"  {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
