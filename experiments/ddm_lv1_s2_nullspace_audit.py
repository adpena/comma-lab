"""ddm_lv1 — pn1 S2+S4 fused: token-nullspace audit on the T2 lotto dump (the $0 ν measurement).

Protocol = pn1 §3 verbatim (consumed, not re-derived):
  1. SENSITIVITY MAP: one full-data MLX backward of the canonical seg loss w.r.t. ALL token
     params → dense |g| per token quantum (~1 epoch-equivalent).
  2. HARD-NULL VERIFICATION (stratified): ~2,000 delta-token quanta sampled across |g| deciles;
     ±1-quantum perturbation → realized argmax flip count on the perturbed pair's frame
     (a delta token lives in ONE pair ⇒ single-frame probe; fp32 MLX-CPU render → torch R →
     frozen CPU SegNet, the A1 verdict path). Calibrates |g| ↔ true-null (the STE/uint8
     dead-zone makes small-|g| a proxy, not a proof).
  3. NULL-SNAP RATE READOUT (encode-side gauge-fix, receiver-free): snap the bottom-q% |g|
     delta tokens to the shared-base prediction (delta → 0 ⇒ near-zero coded cost);
     q ∈ {25,50,70,80,90}; re-code (zlib tdelta + kt_prev1 context-arith, zero counted prior
     bytes per pn1 row-2) + FULL n600 realized confirm per q (chunked ≤120, EMA basis).
     Pre-registered tolerance: Δd_seg ≤ +2e-4. Output: (bytes(q), d_seg(q)) curve and
     ν := max tolerable q; G4 feasibility pivot band ν ∈ [0.55, 0.75] (pn1 §4).

Falsifier (both directions informative, pn1): flat bytes(q) ⇒ tb1 zero-init gauge design
VERIFIED (coder already pays ~nothing for null tokens); steep bytes(q) at flat d_seg(q) ⇒
measured free rate ⇒ standing encode-side projection in the burn's export path.

Evidence axis [macOS-CPU/MLX advisory]; score_claim=false; pointer UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from experiments.ddm_lv1_token_coder_race import (
    DEFAULT_CKPT,
    DEFAULT_GT,
    factor_static_delta,
    kt_bytes,
    _tprev,
)

POINTER_LINE = "0.1910828242 [contest-CPU] UNMOVED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--gt-cache", default=DEFAULT_GT)
    ap.add_argument("--probes", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--qs", default="25,50,70,80,90")
    ap.add_argument("--out", default="/Volumes/VertigoDataTier/pact/ddm_lv1_20260728/"
                                     "s2_nullspace_audit/receipt.json")
    args = ap.parse_args()
    if args.chunk > 120:
        raise SystemExit("chunk must be <= 120 (charter law)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    from experiments.train_tr1_partition_renderer_mlx import (
        SEG_H, SEG_W, TR1Config, build_module, make_render_fn)
    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8, cpu_verdict_d_seg_argmax_batch,
        cpu_verdict_d_seg_batch, make_loss_fn)
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream)
    import sys

    z = np.load(args.ckpt, allow_pickle=False)
    meta = json.loads(bytes(z["meta::json"]).decode())
    cfgd = dict(meta["cfg"])
    cfgd.setdefault("token_init_mode", "zero")
    cfgd.setdefault("basin_handoff", "off")
    cfg = TR1Config(**{k: cfgd[k] for k in TR1Config.__dataclass_fields__ if k in cfgd})
    model = build_module(cfg)
    model.update(tree_unflatten(
        [(k[len("ema::"):], mx.array(z[k])) for k in z.files if k.startswith("ema::")]))
    mx.eval(model.parameters())
    L = cfg.token_quant_levels
    quantum = 2.0 / (L - 1)
    lstars = open_stored_npy_memmap(Path(args.gt_cache), "lstars")
    margins = open_stored_npy_memmap(Path(args.gt_cache), "margins")
    seg_cpu = load_real_segnet("cpu")

    receipt: dict = {"schema": "ddm_lv1_s2_nullspace_audit.v1", "pointer": POINTER_LINE,
                     "score_claim": False, "evidence_axis": "[macOS-CPU/MLX advisory]",
                     "ckpt": args.ckpt, "config_hash": meta.get("config_hash"),
                     "protocol": "pn1 §3 S2 (sensitivity map + stratified hard-null probes "
                                 "+ null-snap curve); prereg dseg tolerance +2e-4; "
                                 "DEVIATION recorded: probes sample tokens across ALL "
                                 "pairs (each token probed on its OWN single frame) "
                                 "instead of the fd2-36 restriction — strictly wider "
                                 "coverage at equal cost"}

    # ---- 1. dense sensitivity map (one full-data backward over token params) ----
    upstream_root = str(Path(sys.modules["tac"].__file__).resolve().parents[2] / "upstream")
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(upstream_root, device="cpu")
    loss_fn = make_loss_fn(adapter, SEG_H, SEG_W, score_domain=True,
                           seg_loss="tau_softplus", margin_weighted=False,
                           render_fn=make_render_fn())

    def pair_loss(mdl, idx: int):
        lstar = np.asarray(lstars[idx], dtype=np.int64)
        lstar_oh = mx.array((lstar[..., None] == np.arange(5)).astype(np.float32))[None]
        margin = mx.array(np.asarray(margins[idx], dtype=np.float32))
        return loss_fn(mdl, None, idx, idx, lstar_oh, margin, mx.zeros((6,)),
                       cfg.w_seg, 0.0, 0.0, cfg.margin_target, seg_form="tau_softplus",
                       compute_pose=False)

    def batch_loss(mdl, ids):
        acc = None
        for i in ids:
            li = pair_loss(mdl, int(i))
            acc = li if acc is None else acc + li
        return acc / len(ids)

    t0 = time.monotonic()
    vg = nn.value_and_grad(model, batch_loss)
    g_abs = np.zeros((cfg.num_pairs, cfg.grid_h, cfg.grid_w, cfg.code_width), np.float64)
    for b0 in range(0, cfg.num_pairs, cfg.batch_pairs):
        ids = list(range(b0, min(b0 + cfg.batch_pairs, cfg.num_pairs)))
        _, grads = vg(model, ids)
        gd = grads["tokens_delta"]
        mx.eval(gd)
        g_abs[ids] += np.abs(np.asarray(gd, dtype=np.float64))[ids]
    map_wall = time.monotonic() - t0
    flat_g = g_abs.reshape(-1)
    receipt["sensitivity_map"] = {
        "wall_s": round(map_wall, 1),
        "quantiles_absg": {str(p): float(np.quantile(flat_g, p / 100))
                           for p in (10, 25, 50, 75, 90, 99)},
        "frac_exact_zero": float(np.mean(flat_g == 0.0))}

    # ---- 2. stratified hard-null probes (±1 quantum, single-frame realized flips) ----
    rng = np.random.default_rng(0)
    deciles = np.quantile(flat_g, np.linspace(0, 1, 11))
    per_dec = max(1, args.probes // 10)
    probe_rows = []
    base_delta = np.asarray(model.tokens_delta, dtype=np.float32).copy()
    t0 = time.monotonic()
    from collections import defaultdict
    by_pair: dict[int, list[tuple]] = defaultdict(list)
    for d in range(10):
        lo, hi = deciles[d], deciles[d + 1]
        idxs = np.where((flat_g >= lo) & (flat_g <= hi))[0]
        if len(idxs) == 0:
            continue
        for fi in rng.choice(idxs, size=min(per_dec, len(idxs)), replace=False):
            coord = np.unravel_index(int(fi), g_abs.shape)
            by_pair[int(coord[0])].append((d, coord))
    dec_flips: dict[int, list[int]] = defaultdict(list)
    for pair, items in by_pair.items():
        with mx.stream(mx.cpu):
            rgb = model.render_frame(pair)
            mx.eval(rgb)
        cam0 = _torch_R_to_camera_uint8(np.asarray(rgb, dtype=np.float32)[0])
        _, real0 = cpu_verdict_d_seg_argmax_batch(
            seg_cpu, [cam0], [np.asarray(lstars[pair], dtype=np.int64)])
        real0 = np.asarray(real0)[0]
        for d, coord in items:
            sign = 1.0 if base_delta[coord] <= 0 else -1.0  # push toward in-range
            pert = base_delta.copy()
            pert[coord] += sign * quantum
            model.tokens_delta = mx.array(pert)
            with mx.stream(mx.cpu):
                rgb = model.render_frame(pair)
                mx.eval(rgb)
            cam = _torch_R_to_camera_uint8(np.asarray(rgb, dtype=np.float32)[0])
            _, real = cpu_verdict_d_seg_argmax_batch(
                seg_cpu, [cam], [np.asarray(lstars[pair], dtype=np.int64)])
            dec_flips[d].append(int(np.count_nonzero(np.asarray(real)[0] != real0)))
        model.tokens_delta = mx.array(base_delta)
    mx.eval(model.parameters())
    receipt["hard_null_probes"] = {
        "n_probes": int(sum(len(v) for v in dec_flips.values())),
        "wall_s": round(time.monotonic() - t0, 1),
        "per_decile": {str(d): {"n": len(v),
                                "zero_flip_frac": float(np.mean(np.array(v) == 0)),
                                "mean_flips": float(np.mean(v)) if v else None}
                       for d, v in sorted(dec_flips.items())}}

    # ---- 3. null-snap curve: snap bottom-q% |g| deltas to base; bytes + full confirm ----
    t_full = np.clip(base_delta + np.asarray(model.tokens_base, dtype=np.float32)[None],
                     -1.0, 1.0)
    q_base = np.round((t_full + 1.0) * 0.5 * (L - 1)).astype(np.uint8)
    order = np.argsort(flat_g)  # ascending |g|
    curve = {}
    for qpct in [int(x) for x in args.qs.split(",")]:
        k = int(len(order) * qpct / 100)
        snap_mask = np.zeros(len(order), bool)
        snap_mask[order[:k]] = True
        snap_mask = snap_mask.reshape(g_abs.shape)
        d_snap = base_delta.copy()
        d_snap[snap_mask] = 0.0  # delta -> 0 (shared-base prediction)
        t_s = np.clip(d_snap + np.asarray(model.tokens_base, dtype=np.float32)[None],
                      -1.0, 1.0)
        q_s = np.round((t_s + 1.0) * 0.5 * (L - 1)).astype(np.uint8)
        # bytes: F1 factorization + {zlib tdelta, kt_prev1} on the snapped stream
        bhat, dstream = factor_static_delta(q_s, L)
        dl = dstream.copy()
        dl[1:] = (dstream[1:].astype(np.int16) - dstream[:-1].astype(np.int16)) % 256
        by_zlib = len(zlib.compress(dl.astype(np.uint8).tobytes(), 9)) + \
            len(zlib.compress(bhat.tobytes(), 9))
        kt_tot = 0.0
        for ch in range(q_s.shape[-1]):
            f = dstream[..., ch].astype(np.int64)
            b, _ = kt_bytes(f, _tprev(f), L)
            kt_tot += b
        # full n600 realized confirm with the snapped tokens
        model.tokens_delta = mx.array(d_snap)
        mx.eval(model.parameters())
        dsegs = []
        t0 = time.monotonic()
        for c0 in range(0, cfg.num_pairs, args.chunk):
            cams, gts = [], []
            with mx.stream(mx.cpu):
                for i in range(c0, min(c0 + args.chunk, cfg.num_pairs)):
                    rgb = model.render_frame(i)
                    mx.eval(rgb)
                    cams.append(_torch_R_to_camera_uint8(
                        np.asarray(rgb, dtype=np.float32)[0]))
                    gts.append(np.asarray(lstars[i], dtype=np.int64))
            dsegs.extend(cpu_verdict_d_seg_batch(seg_cpu, cams, gts))
        model.tokens_delta = mx.array(base_delta)
        mx.eval(model.parameters())
        curve[str(qpct)] = {
            "snapped_quanta": int(k),
            "bytes_zlib_f1": int(by_zlib),
            "bytes_kt_prev1_f1": round(kt_tot, 1),
            "full_dseg": float(np.mean(dsegs)),
            "confirm_wall_s": round(time.monotonic() - t0, 1)}
        print(json.dumps({"q": qpct, **curve[str(qpct)]}), flush=True)
        out_path.write_text(json.dumps(receipt | {"null_snap_curve": curve},
                                       indent=2, sort_keys=True) + "\n")

    receipt["null_snap_curve"] = curve
    receipt["prereg"] = {"dseg_tolerance": 2e-4,
                         "baseline_full_dseg": 0.013832855224609374,
                         "g4_pivot_band": [0.55, 0.75]}
    tol = 0.013832855224609374 + 2e-4
    ok_qs = [int(qq) for qq, row in curve.items() if row["full_dseg"] <= tol]
    receipt["nu_max_tolerable_q"] = max(ok_qs) / 100 if ok_qs else 0.0
    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"nu": receipt["nu_max_tolerable_q"],
                      "receipt": str(out_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
