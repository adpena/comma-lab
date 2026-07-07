#!/usr/bin/env python
"""$0 TRAINING-FLOW tau-crossover probe — the owed anchor of `dash_erasure_homogenization_v1`.

[macOS-CPU advisory] NON-PROMOTABLE. Renders the WITNESS ALONE (no band compositing) from the
live #205 run's OWN per-stage checkpoints (each carrying its annealed ``__cfg_softmax_temp``,
cross-checked against run.log loss_terms telemetry) and measures, through the EXACT contest R
(bicubic^ 874x1164 -> round/clamp/uint8 -> SegNet contest bilinear -> argmax) vs GT lstars on
ALL 600 pairs, per checkpoint:

  d_seg, lane recall / FP / FN, dash-gap FP (same definition + normalization as
  tools/dash_comb_probe_n600.py: realized-lane FP inside solid-band footprint where the
  ego-phase comb gates OFF), and the amplitude-normalized HOMOGENIZATION INDEX
  H = P(realized==lane | gap) / P(realized==lane | mark), total and per forward-range band.

THIN WRAPPER discipline: imports the dash-comb probe's Renderer + verdict path + cached
per-pair line fits + global comb fit (deterministic, GT-derived, checkpoint-independent) —
no duplicated render code. Pre-registration:
.omx/research/tau_crossover_trainflow_probe_20260707.md (design written BEFORE measurement).

LIVE RUN IS SACRED: read-only; every checkpoint is snapshotted into OUT_DIR before loading
(the live EMA can change under us mid-measurement). Chunked resumable foreground (atomic
tmp+replace state; kills become progress); free-RAM >= 20 GiB gate per invocation; verdict
batches VBATCH<=12; NO MPS. Pointer 0.19110 UNMOVED (this is a means).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import dash_comb_probe_n600 as dcp  # noqa: E402  (sets env caps, sys.path, torch threads)
import numpy as np  # noqa: E402

from tac.boundary_math.analytic_lane_render_band import (  # noqa: E402
    rasterize_lane_coverage_range_dependent,
)
from tac.boundary_math.dash_comb import (  # noqa: E402
    ego_cumulative_distance,
    rasterize_lane_coverage_combed,
)
from experiments.train_witness_realized_through_R_mlx import (  # noqa: E402
    _torch_R_to_camera_uint8,
)

RUN_DIR = REPO / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"
OUT_DIR = REPO / "experiments/results/tau_crossover_trainflow_20260707"
STATE = OUT_DIR / "probe_state.ckpt.npz"
CKPTS_MANIFEST = OUT_DIR / "ckpts.json"
OUT = OUT_DIR / "tau_crossover_trainflow_n600_20260707.json"

# measurement order = most-informative-first (extremes -> fixed-tau segment -> mid anneal point),
# so a budget cut still leaves complete n600 rows for the max-contrast comparison.
DEFAULT_CKPTS = [
    ("ep299_CEend", "levelset_ckpt_stageCE_ep299.npz"),
    ("ep925_liveEMA", "levelset_witness_ema_mlx.npz"),
    ("ep726_MuonStart", "levelset_ckpt_stageMuonStart_ep726.npz"),
    ("ep650_tauBest", "levelset_witness_ema_BEST.npz"),
]
# NOTE: there is no literal "stageFinal"/"ema_FINAL" checkpoint — the trainer
# preserves the FINAL checkpoint under its final *stage* name (e.g.
# levelset_ckpt_stageTau_muon_ep1000.npz), per the per-stage-checkpoint rule.
# --add-final therefore DISCOVERS the real final checkpoint (highest-epoch
# preserved stage ckpt; fallback = rolling EMA latest) rather than inventing a
# filename. (Prior FINAL_CANDIDATES = ema_FINAL/stageFinal_ep1000 were names no
# run ever emits → --add-final was a dead no-op. Fixed 2026-07-07.)


def _final_ckpt_epoch(p: Path) -> int:
    tail = p.stem.rsplit("_ep", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _resolve_final_ckpt(run_dir: Path) -> Path | None:
    """The ACTUAL final checkpoint file in run_dir — highest-epoch preserved stage
    ckpt (``levelset_ckpt_stage*_ep*.npz``), fallback = rolling EMA latest
    (``levelset_witness_ema_mlx.npz``). Returns None ONLY when the run has
    neither, which is a genuine anomaly the caller MUST surface loudly (never a
    silent skip)."""
    stage_ckpts = sorted(run_dir.glob("levelset_ckpt_stage*_ep*.npz"),
                         key=_final_ckpt_epoch)
    if stage_ckpts:
        return stage_ckpts[-1]
    ema = run_dir / "levelset_witness_ema_mlx.npz"
    return ema if ema.exists() else None

VBATCH = int(dcp.os.environ.get("TAUXOVER_VBATCH", "6"))
SAVE_EVERY = int(dcp.os.environ.get("TAUXOVER_SAVE_EVERY", "24"))
METRICS = ["dseg", "rec", "fn", "fp", "gapfp",
           "mark_ct", "mark_lane_ct", "gap_ct", "gap_lane_ct"]


def _load_ckpt_from(path: Path):
    """load_ckpt parametrized by path (mirrors dcp.load_ckpt, which is hardwired to its global)."""
    z = np.load(path, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}
    code = params.pop("code")
    m = dict(
        render_h=384, render_w=512, n_pairs=code.shape[0] // 2,
        bank_n_scales=int(cfg["__bank_n_scales"]), bank_n_orient0=int(cfg["__bank_n_orient0"]),
        bank_f0=float(cfg["__bank_f0"]), bank_base=float(cfg["__bank_base"]),
        bank_n_iso=int(cfg["__bank_n_iso"]), max_bank_freq=float(cfg["__cfg_max_bank_freq"]),
        self_orient=bool(int(cfg["__cfg_self_orient"])), n_dir_freqs=int(cfg["__cfg_n_dir_freqs"]),
        so_iters=4, so_freq_along=float(cfg["__cfg_freq_along"]),
        so_freq_across=float(cfg["__cfg_freq_across"]), so_tau=4.0,
        activation=str(cfg["__cfg_activation"]), wire_w0=float(cfg["__cfg_wire_w0"]),
        wire_s0=float(cfg["__cfg_wire_s0"]), hosc_beta=float(cfg["__cfg_hosc_beta"]),
        hosc_omega=float(cfg["__cfg_hosc_omega"]), n_hidden=int(cfg["__cfg_n_hidden"]),
        hidden_dim=int(cfg["__cfg_hidden_dim"]), softmax_temp=float(cfg["__cfg_softmax_temp"]),
        chroma=bool(int(cfg["__cfg_chroma"])),
    )
    return params, code, m, cfg


def _snapshot_ckpts(include_final: bool) -> list[dict]:
    """Snapshot every checkpoint into OUT_DIR (read-only live-run discipline); cache manifest."""
    if CKPTS_MANIFEST.exists():
        rows = json.loads(CKPTS_MANIFEST.read_text())
        if not include_final:
            return rows
        have = {r["label"] for r in rows}
        if any(lbl.startswith("final") for lbl in have):
            return rows
    else:
        rows = []
    listed = {r["src"] for r in rows}
    wanted = [(lbl, name) for lbl, name in DEFAULT_CKPTS]
    if include_final:
        final_src = _resolve_final_ckpt(RUN_DIR)
        already = {name for _, name in wanted}
        if final_src is None:
            # NO SILENT ERROR: neither a preserved stage ckpt nor a rolling EMA
            # exists — say exactly why, loudly (main() also exits non-zero).
            print(f"[add-final] ERROR: no final checkpoint in {RUN_DIR} — expected the "
                  "highest-epoch levelset_ckpt_stage*_ep*.npz or "
                  "levelset_witness_ema_mlx.npz. Run incomplete or RUN_DIR wrong; the "
                  "fifth point cannot be added.", flush=True)
        elif final_src.name in already:
            print(f"[add-final] final checkpoint {final_src.name} already coincides with "
                  "a measured point (the rolling EMA IS the final) — no distinct fifth "
                  "point to add.", flush=True)
        else:
            ep = _final_ckpt_epoch(final_src)
            label = f"final_ep{ep}" if ep >= 0 else "final_emaLatest"
            wanted.append((label, final_src.name))
            print(f"[add-final] final checkpoint discovered: {final_src.name} -> {label} "
                  f"(ep{ep})", flush=True)
    for label, name in wanted:
        src = RUN_DIR / name
        if str(src) in listed:
            continue
        dst = OUT_DIR / f"frozen_{label}.npz"
        if not dst.exists():
            shutil.copy2(src, dst)
        _, _, m, cfg = _load_ckpt_from(dst)
        rows.append({
            "label": label, "src": str(src), "snapshot": str(dst),
            "epoch": int(cfg.get("__epoch", -1)), "softmax_temp": m["softmax_temp"],
        })
        print(f"[snapshot] {label}: ep{rows[-1]['epoch']} tau={rows[-1]['softmax_temp']:.5f} "
              f"<- {name}", flush=True)
    CKPTS_MANIFEST.write_text(json.dumps(rows, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-seconds", type=float, default=520.0,
                    help="exit cleanly (state saved) after this many seconds; re-invoke to resume.")
    ap.add_argument("--num-pairs", type=int, default=600, help="n600 discipline: keep 600.")
    ap.add_argument("--add-final", action="store_true",
                    help="also snapshot+measure a final checkpoint if the live run has finished.")
    args = ap.parse_args()

    free = dcp._free_gib()
    if free < dcp.MIN_FREE_GIB:
        print(f"REFUSE: free RAM {free:.1f} GiB < {dcp.MIN_FREE_GIB} GiB "
              f"(live #205 run protection)", flush=True)
        sys.exit(3)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ck_rows = _snapshot_ckpts(args.add_final)
    # World-class DX: --add-final must not silently look like success. If it was
    # requested but produced no distinct final row AND the final is genuinely
    # absent (not merely coincident with an existing point), exit non-zero.
    if args.add_final and not any(str(r["label"]).startswith("final") for r in ck_rows):
        if _resolve_final_ckpt(RUN_DIR) is None:
            print("[add-final] FAILED: the fifth (final) point was requested but no final "
                  "checkpoint exists in the run dir (see ERROR above). Exiting rc=4 so this "
                  "is never mistaken for success.", flush=True)
            sys.exit(4)
        print("[add-final] note: the final checkpoint coincides with an already-measured "
              "point — this is legitimate (rolling EMA == final); 4 rows stand.", flush=True)
    labels = [r["label"] for r in ck_rows]

    seg = dcp.load_real_segnet("cpu")
    z = np.load(dcp.CACHE, allow_pickle=False)
    lst_all = z["lstars"]
    gt_poses = np.asarray(z["gt_poses"], np.float64)
    P = min(int(args.num_pairs), int(z["n_pairs"]))
    # reuse the dash-comb probe's cached, deterministic GT-derived line fits + global comb fit
    per_pair_lines, fit = dcp._stage0_lines_and_fit(lst_all, gt_poses, P)
    E = ego_cumulative_distance(gt_poses[:P, 0], fit.scale)
    H_, W_ = 384, 512
    fwd_rows = dcp._forward_of_rows(H_)
    band_row_idx = np.full(H_, -1, np.int64)
    for bi, (lo, hi) in enumerate(dcp.FWD_BANDS):
        band_row_idx[(fwd_rows >= lo) & (fwd_rows < hi)] = bi
    NB = len(dcp.FWD_BANDS)
    print(f"[{time.time() - t0:.1f}s] segnet + cache + cached comb fit loaded; "
          f"P={P} ckpts={labels}", flush=True)

    st: dict[str, np.ndarray] = {}
    done: dict[str, np.ndarray] = {}
    if STATE.exists():
        ck = np.load(STATE, allow_pickle=False)
        for k in ck.files:
            if k.startswith("done__"):
                done[k[len("done__"):]] = ck[k].astype(bool)
            else:
                st[k] = ck[k]
    for lbl in labels:
        for mt in METRICS:
            st.setdefault(f"{lbl}__{mt}", np.full(P, np.nan))
        for mt in ("mark_ct", "mark_lane_ct", "gap_ct", "gap_lane_ct"):
            st.setdefault(f"{lbl}__{mt}_band", np.full((P, NB), np.nan))
        done.setdefault(lbl, np.zeros(P, bool))
    if any(d.any() for d in done.values()):
        print("[resume] " + " ".join(f"{lbl}:{int(done[lbl].sum())}/{P}" for lbl in labels),
              flush=True)

    def save():
        payload = dict(st)
        for lbl, d in done.items():
            payload[f"done__{lbl}"] = d
        tmp = STATE.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(STATE)

    def measure_ckpt(row: dict) -> bool:
        """Measure one checkpoint's remaining pairs. Returns False on chunk-budget exit."""
        lbl = row["label"]
        todo = [i for i in range(P) if not done[lbl][i]]
        if not todo:
            return True
        params, code, m, _cfg = _load_ckpt_from(Path(row["snapshot"]))
        R = dcp.Renderer(params, code, m)
        last = int(done[lbl].sum())
        for s0 in range(0, len(todo), VBATCH):
            chunk = todo[s0:s0 + VBATCH]
            frames, regions = [], []
            for pi in chunk:
                bulk, _lane = R.render_pair(pi)
                frames.append(_torch_R_to_camera_uint8(bulk.astype(np.float64)))
                lines = per_pair_lines[pi]
                cov_solid = rasterize_lane_coverage_range_dependent(
                    lines, h=H_, w=W_, softness=dcp.BAND_SOFTNESS_PX, dash_gate=False)
                cov_comb = rasterize_lane_coverage_combed(
                    lines, fit, float(E[pi]), pair_idx=pi, h=H_, w=W_,
                    softness=dcp.BAND_SOFTNESS_PX,
                    dash_forward_max_m=dcp.DASH_FORWARD_MAX_M,
                    comb_softness_m=dcp.COMB_SOFTNESS_M)
                mark = (cov_solid >= 0.5) & (cov_comb >= 0.5)
                gap = (cov_solid >= 0.5) & (cov_comb < 0.5)
                regions.append((mark, gap))
            realized = dcp.seg_argmax_batch(seg, frames)
            for j, pi in enumerate(chunk):
                gt_l = np.asarray(lst_all[pi], np.int64)
                r = realized[j]
                n = float(gt_l.size)
                is_lane = gt_l == 1
                nlane = int(is_lane.sum())
                r_lane = r == 1
                fp_px = (~is_lane) & r_lane
                mark, gap = regions[j]
                st[f"{lbl}__dseg"][pi] = float(np.count_nonzero(r != gt_l)) / n
                st[f"{lbl}__rec"][pi] = (float(np.count_nonzero(r[is_lane] == 1)) / nlane) if nlane else np.nan
                st[f"{lbl}__fn"][pi] = float((is_lane & ~r_lane).sum()) / n
                st[f"{lbl}__fp"][pi] = float(fp_px.sum()) / n
                st[f"{lbl}__gapfp"][pi] = float((fp_px & gap).sum()) / n
                st[f"{lbl}__mark_ct"][pi] = float(mark.sum())
                st[f"{lbl}__mark_lane_ct"][pi] = float((r_lane & mark).sum())
                st[f"{lbl}__gap_ct"][pi] = float(gap.sum())
                st[f"{lbl}__gap_lane_ct"][pi] = float((r_lane & gap).sum())
                for bi in range(NB):
                    bm = band_row_idx == bi
                    st[f"{lbl}__mark_ct_band"][pi, bi] = float(mark[bm].sum())
                    st[f"{lbl}__mark_lane_ct_band"][pi, bi] = float((r_lane & mark)[bm].sum())
                    st[f"{lbl}__gap_ct_band"][pi, bi] = float(gap[bm].sum())
                    st[f"{lbl}__gap_lane_ct_band"][pi, bi] = float((r_lane & gap)[bm].sum())
                done[lbl][pi] = True
            nd = int(done[lbl].sum())
            if nd - last >= SAVE_EVERY or nd == P:
                save()
                last = nd
                gp = np.nansum(st[f"{lbl}__gap_lane_ct"]) / max(np.nansum(st[f"{lbl}__gap_ct"]), 1.0)
                mk = np.nansum(st[f"{lbl}__mark_lane_ct"]) / max(np.nansum(st[f"{lbl}__mark_ct"]), 1.0)
                print(f"[{time.time() - t0:.1f}s] {lbl} {nd}/{P} | "
                      f"dseg={np.nanmean(st[f'{lbl}__dseg']):.5f} "
                      f"rec={np.nanmean(st[f'{lbl}__rec']):.4f} "
                      f"gapfp={np.nanmean(st[f'{lbl}__gapfp']):.6f} "
                      f"H={gp / mk if mk > 0 else float('nan'):.4f} "
                      f"| rss={dcp._peak_rss_gib():.1f}GiB", flush=True)
            if time.time() - t0 > float(args.chunk_seconds) and nd < P:
                save()
                print(f"[chunk-exit] {lbl} {nd}/{P} at {time.time() - t0:.1f}s; re-invoke to "
                      f"resume. peak_rss={dcp._peak_rss_gib():.2f}GiB", flush=True)
                return False
        return True

    for row in ck_rows:
        if not measure_ckpt(row):
            sys.exit(0)
    save()

    def pooled(lbl, num, den, bi=None):
        a = st[f"{lbl}__{num}" + ("_band" if bi is not None else "")]
        b = st[f"{lbl}__{den}" + ("_band" if bi is not None else "")]
        if bi is not None:
            a, b = a[:, bi], b[:, bi]
        sa, sb = float(np.nansum(a)), float(np.nansum(b))
        return sa / sb if sb > 0 else float("nan")

    table = []
    for row in ck_rows:
        lbl = row["label"]
        r_mark = pooled(lbl, "mark_lane_ct", "mark_ct")
        r_gap = pooled(lbl, "gap_lane_ct", "gap_ct")
        table.append({
            "label": lbl, "epoch": row["epoch"], "tau": row["softmax_temp"],
            "n_done": int(done[lbl].sum()),
            "d_seg": float(np.nanmean(st[f"{lbl}__dseg"])),
            "lane_recall": float(np.nanmean(st[f"{lbl}__rec"])),
            "lane_fp": float(np.nanmean(st[f"{lbl}__fp"])),
            "lane_fn": float(np.nanmean(st[f"{lbl}__fn"])),
            "dash_gap_fp": float(np.nanmean(st[f"{lbl}__gapfp"])),
            "r_mark": r_mark, "r_gap": r_gap,
            "H_index": (r_gap / r_mark) if r_mark and r_mark > 0 else float("nan"),
            "H_by_band": [
                (lambda rg, rm: rg / rm if rm and rm > 0 else float("nan"))(
                    pooled(lbl, "gap_lane_ct", "gap_ct", bi),
                    pooled(lbl, "mark_lane_ct", "mark_ct", bi))
                for bi in range(NB)
            ],
            "r_mark_by_band": [pooled(lbl, "mark_lane_ct", "mark_ct", bi) for bi in range(NB)],
            "r_gap_by_band": [pooled(lbl, "gap_lane_ct", "gap_ct", bi) for bi in range(NB)],
        })
    out = {
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "task": "training-flow tau-crossover probe (owed anchor of dash_erasure_homogenization_v1): "
                "witness-alone renders from the live #205 run's own per-stage checkpoints, each at "
                "its annealed softmax_temp, n600 through exact R + frozen CPU SegNet",
        "pre_registration": ".omx/research/tau_crossover_trainflow_probe_20260707.md",
        "deconfound_limit": "single trajectory: epoch and tau co-vary; ep726->ep925 is the "
                            "quasi-fixed-tau internal contrast (delta tau ~ -0.5%); definitive "
                            "deconfound = fixed-tau control arm (next-run design item)",
        "n_pairs": P,
        "checkpoints": ck_rows,
        "table": table,
        "H_definition": "H = P(realized==lane | solid-band & comb-OFF gap) / "
                        "P(realized==lane | solid-band & comb-ON mark); pooled (micro-avg) over pairs",
        "dash_gap_fp_definition": "identical to tools/dash_comb_probe_n600.py: "
                                  "(GT!=lane & realized==lane & gap-region).sum()/total_px, mean over pairs",
        "forward_bands_m": [[lo, hi] for lo, hi in dcp.FWD_BANDS],
        "delta_along_px_per_band": json.loads(dcp.LINES_JSON.read_text()).get("fit", {}).get(
            "provenance", {}).get("delta_along_note", "see dash_comb_probe JSON "
            "probe2_delta_along_px_per_band: 110/19.8/5.4/1.9/0.39 px, dominant slot -1 T=7.54 m"),
        "R_path": "torch bicubic^ 874x1164 -> round/clamp/uint8 -> SegNet.preprocess_input "
                  "(contest bilinear) -> argmax vs GT lstars (frozen CPU-torch; never MPS)",
        "peak_rss_gib": dcp._peak_rss_gib(),
        "note": "means not ends: advisory row; pointer 0.19110 moves only via upstream/evaluate.py "
                "on exact archive bytes",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE -> {OUT}", flush=True)
    for r in table:
        print(f"  {r['label']:16s} ep{r['epoch']:4d} tau={r['tau']:.5f} n={r['n_done']} "
              f"gapfp={r['dash_gap_fp']:.6f} rec={r['lane_recall']:.4f} "
              f"dseg={r['d_seg']:.5f} H={r['H_index']:.4f}", flush=True)


if __name__ == "__main__":
    main()
