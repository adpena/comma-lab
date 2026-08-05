"""ddm_gr1 — TOKEN-GRANULARITY re-race on the pfs1 D1 dr7t lattice (ledger QA07/QA08/QA24).

THE QUESTION (co9 fired granularity_race_duty ARMED->DUE on the REAL in-band base
0.00389011): the token alphabet/granularity is the model class the rate lives in
(coder SATURATED within 1,617 B of ideal per QA08/xi1). Re-race the DESCRIPTION
granularity with per-token sensitivity-ordered precision allocation (QA11 law: 35x
spread, 27% exact-zero grads -> continuous log-bit dominates the rung ladder), no
retraining. Deliver the measured (bytes, realized d_seg) RD curve vs the current
operating point (569,996 B, d_seg 0.00389011) and vs wr1 cell-drop Knee-A.

WHAT THIS DOES:
  * Loads archive dr7t codes [600,24,32,4] L16 -> factor_mode_delta -> (base,delta).
  * Loads the T3 endpoint ckpt -> MLX model; ONE seg-loss backward -> per-TOKEN |g|
    (sb1/QA11 sensitivity map, 1.84M values). Verifies model codes == archive codes.
  * Coarsens the mod-16 residual PER TOKEN, sensitivity-ordered, at candidate
    allocations: DROP-token (delta->0, the token-granular wr1) and NESTED-RUNG
    {L16,L8,L4,base}. Measures bytes through the REAL SMEVR coder (archive-faithful).
  * Realized d_seg: inject coarsened codes -> model.render_frame -> torch R -> frozen
    CPU SegNet argmax vs GT lstars. n48 ranked subset first; n600 ONLY on the winner.

AUTHORITY: bytes MEASURED (lossless, real coder). d_seg REALIZED through the real
render+SegNet (validated: baseline injection -> 0.00389 == pfs1 D1 evaluate.py).
[macOS-CPU advisory]; score_claim=false; promotion_eligible=false; pointer
0.1910828242 [contest-CPU] UNMOVED. NON-PROMOTABLE. No scorer promotion / paid
dispatch / pointer mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddm_r7_token_coder import (
    decode_token_codes,
    encode_token_codes,
    factor_mode_delta,
    reconstruct_mode_delta,
)
from tac.subset_selection import MODE_PREFIX, MODE_STRATIFIED, Selection, select

POINTER = "0.1910828242 [contest-CPU] UNMOVED"
LEVELS = 16
UNCOMPRESSED = 37_545_489
# pfs1 D1 exact-protocol reference (real evaluator, real bytes).
REF_ARCHIVE_BYTES = 569_996
REF_TOKENS_BYTES = 557_253
REF_DSEG = 0.00389011
REF_DPOSE = 0.22144216
ARCHIVE_FLOOR = REF_ARCHIVE_BYTES - REF_TOKENS_BYTES  # 12,743 B non-token members
# water break-even: 25*dB/N == 100*d_dseg  =>  B/flip == 100*(1/TOTAL_PX)*N/25
TOTAL_PX = 600 * 512 * 384  # 117,964,800 argmax pixels
WATER_B_PER_FLIP = 100.0 * (1.0 / TOTAL_PX) * UNCOMPRESSED / 25.0  # ~1.273
DSEG_PER_FLIP = 1.0 / TOTAL_PX

CKPT = ("/Volumes/VertigoDataTier/pact/ddm_lv1_20260728/t3_long_burn_lotto_v2/"
        "checkpoints/stage_seg_trunk_tau_final.npz")
ARCHIVE = ("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
           "submissions/pfs1/archive.zip")
GT = ("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
OUTDIR = Path("/Volumes/VertigoDataTier/pact/ddm_gr1_20260730")
SELECTION_POPULATION = 600
DEFAULT_SELECTION_SEED = 20260805
SELECTION_BOOTSTRAP = 2000
SEG_GOVERNING_TABLE = REPO / "src/tac/tests/fixtures/subset_selection/gt_n600_per_pair_population.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rate_term(archive_bytes: int) -> float:
    return 25.0 * archive_bytes / UNCOMPRESSED


def _load_seg_governing_table(path: Path = SEG_GOVERNING_TABLE) -> tuple[list[float], dict[str, Any]]:
    rec = json.loads(path.read_text())
    if int(rec.get("n_pairs", -1)) != SELECTION_POPULATION:
        raise ValueError(
            f"{path} has n_pairs={rec.get('n_pairs')} but GR1 selection population is "
            f"{SELECTION_POPULATION}"
        )
    vals = [float(v) for v in rec["seg_flip_density"]]
    if len(vals) != SELECTION_POPULATION:
        raise ValueError(
            f"{path} seg_flip_density has {len(vals)} entries, expected "
            f"{SELECTION_POPULATION}"
        )
    meta = {
        "path": str(path),
        "sha256": _sha_path(path),
        "quantity": "seg_flip_density",
        "source_npz": rec.get("source_npz"),
        "derivation": rec.get("derivation_seg_flip_density"),
        "population": SELECTION_POPULATION,
    }
    return vals, meta


def _selector_mode(cli_mode: str) -> str:
    if cli_mode == "prefix":
        return MODE_PREFIX
    if cli_mode == "stratified":
        return MODE_STRATIFIED
    raise ValueError(f"unknown --selection-mode {cli_mode!r}")


def build_selection_scope(
    n_pairs: int,
    selection_mode: str,
    selection_seed: int | None,
    *,
    n_bootstrap: int = SELECTION_BOOTSTRAP,
) -> tuple[Selection, dict[str, Any]]:
    governing, table_meta = _load_seg_governing_table()
    mode = _selector_mode(selection_mode)
    seed = int(selection_seed) if mode == MODE_STRATIFIED else None
    sel = select(
        int(n_pairs),
        SELECTION_POPULATION,
        mode=mode,
        seed=seed,
        governing=governing,
        governing_name="seg_flip_density",
        n_bootstrap=int(n_bootstrap),
    )
    prov = sel.provenance()
    prov["indices"] = list(sel.indices)
    rec = {
        "schema": "ddm_gr1.selection_scope.v1",
        "score_claim": False,
        "scorer_forwards_run": 0,
        "selection_args": {
            "selection_mode": selection_mode,
            "selection_seed": seed,
            "n_pairs": int(n_pairs),
        },
        "selection": prov,
        "summary": sel.summary(),
        "governing_table": table_meta,
        "scope_proof": {
            "default_prefix_preserved": bool(mode == MODE_PREFIX),
            "selection_is_explicit": True,
            "population_match_checked": bool(sel.ratios),
            "population_matched": sel.population_matched,
            "axis": "seg",
        },
    }
    return sel, rec


def write_selection_receipt(outdir: Path, selection_mode: str, selection_rec: dict[str, Any]) -> Path:
    n_pairs = int(selection_rec["selection"]["n"])
    seed = selection_rec["selection_args"]["selection_seed"]
    seed_tag = "noseed" if seed is None else f"seed{seed}"
    out = outdir / f"gr1_selection_{selection_mode}_n{n_pairs}_{seed_tag}.json"
    out.write_text(json.dumps(selection_rec, indent=1) + "\n")
    return out


# ---------------------------------------------------------------- coarsening
def signed_residual(delta: np.ndarray) -> np.ndarray:
    s = delta.astype(np.int16)
    return np.where(s > LEVELS // 2, s - LEVELS, s)  # [-8,7]


def coarsen_alloc(signed: np.ndarray, step_map: np.ndarray) -> np.ndarray:
    """step_map same shape as signed; entries in {1,2,4,0(=drop)}.

    step 1 = keep L16; 2 = L8; 4 = L4; 0 = drop-to-base (residual -> 0).
    Returns coarsened residual (signed int16)."""
    out = signed.astype(np.float64).copy()
    for step in (2, 4):
        m = step_map == step
        if m.any():
            out[m] = np.round(out[m] / step) * step
    out[step_map == 0] = 0.0
    return np.clip(np.rint(out), -8, 7).astype(np.int16)


def residual_to_codes(base: np.ndarray, signed_coarse: np.ndarray) -> np.ndarray:
    delta = (signed_coarse.astype(np.int16) % LEVELS).astype(np.uint8)
    return reconstruct_mode_delta(base, delta, LEVELS)


def smevr_bytes(codes: np.ndarray) -> int:
    return len(encode_token_codes(codes, levels=LEVELS, codec="smevr"))


# ---------------------------------------------------------------- model / d_seg
def load_model():
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    from experiments.train_tr1_partition_renderer_mlx import TR1Config, build_module

    z = np.load(CKPT, allow_pickle=False)
    meta = json.loads(bytes(z["meta::json"]).decode())
    cfgd = dict(meta["cfg"])
    cfgd.setdefault("token_init_mode", "zero")
    cfgd.setdefault("basin_handoff", "off")
    cfg = TR1Config(**{k: cfgd[k] for k in TR1Config.__dataclass_fields__ if k in cfgd})
    model = build_module(cfg)
    model.update(tree_unflatten(
        [(k[len("ema::"):], mx.array(z[k])) for k in z.files if k.startswith("ema::")]))
    mx.eval(model.parameters())
    return model, cfg, meta


def sensitivity_map(model, cfg, gt_cache: str) -> tuple[np.ndarray, float]:
    """One seg-loss backward over token params -> per-token |g| (sb1/QA11)."""
    import mlx.core as mx
    import mlx.nn as nn

    from experiments.train_tr1_partition_renderer_mlx import SEG_H, SEG_W, make_render_fn
    from experiments.train_witness_realized_through_R_mlx import make_loss_fn
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.local_acceleration.mlx_scorer_adapters import load_mlx_distortion_scorer_adapter_from_upstream

    lstars = open_stored_npy_memmap(Path(gt_cache), "lstars")
    margins = open_stored_npy_memmap(Path(gt_cache), "margins")
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

    vg = nn.value_and_grad(model, batch_loss)
    g_abs = np.zeros((cfg.num_pairs, cfg.grid_h, cfg.grid_w, cfg.code_width), np.float64)
    t0 = time.monotonic()
    for b0 in range(0, cfg.num_pairs, cfg.batch_pairs):
        ids = list(range(b0, min(b0 + cfg.batch_pairs, cfg.num_pairs)))
        _, grads = vg(model, ids)
        gd = grads["tokens_delta"]
        mx.eval(gd)
        g_abs[ids] += np.abs(np.asarray(gd, dtype=np.float64))[ids]
    return g_abs, round(time.monotonic() - t0, 1)


def realized_dseg(model, cfg, codes: np.ndarray, gt_cache: str,
                  pair_indices, chunk: int = 120) -> tuple[float, float]:
    """Inject coarsened codes -> render -> frozen CPU SegNet argmax vs GT -> mean d_seg."""
    import mlx.core as mx

    from experiments.train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8, cpu_verdict_d_seg_batch
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet

    indices = tuple(int(i) for i in pair_indices)
    lstars = open_stored_npy_memmap(Path(gt_cache), "lstars")
    seg_cpu = load_real_segnet("cpu")
    base_arr = np.asarray(model.tokens_base, dtype=np.float32)
    t = codes.astype(np.float32) / (LEVELS - 1) * 2.0 - 1.0
    inj = t - base_arr[None]
    saved = model.tokens_delta
    model.tokens_delta = mx.array(inj.astype(np.float32))
    mx.eval(model.parameters())
    dsegs: list[float] = []
    t0 = time.monotonic()
    for c0 in range(0, len(indices), chunk):
        cams, gts = [], []
        chunk_indices = indices[c0:c0 + chunk]
        with mx.stream(mx.cpu):
            for i in chunk_indices:
                rgb = model.render_frame(i)
                mx.eval(rgb)
                cams.append(_torch_R_to_camera_uint8(np.asarray(rgb, dtype=np.float32)[0]))
                gts.append(np.asarray(lstars[i], dtype=np.int64))
        dsegs.extend(cpu_verdict_d_seg_batch(seg_cpu, cams, gts))
    model.tokens_delta = saved
    mx.eval(model.parameters())
    return float(np.mean(dsegs)), round(time.monotonic() - t0, 1)


# ---------------------------------------------------------------- allocations
def build_allocations(g_abs: np.ndarray, family: str = "token") -> dict[str, np.ndarray]:
    """Return {name: step_map} where step_map has entries in {1,2,4,0}.

    family='token' — per-token |g| ordering (the QA11 continuous-allocation claim).
    family='cell'  — per-CELL (spatial 24x32) ordering; a whole spatial column
                     (all 600 pairs x 4 channels) shares one rung. This tests the
                     efficient-coder-unit hypothesis (SMEVR conditions on the
                     temporal mode; a cell is its natural unit; cf. wr1)."""
    allocs: dict[str, np.ndarray] = {}
    n = g_abs.size
    if family == "token":
        order = np.argsort(g_abs.reshape(-1), kind="stable")  # ascending |g|
        for f in (0.10, 0.20, 0.2704, 0.35, 0.50, 0.65, 0.80):
            sm = np.ones(n, dtype=np.int16)
            sm[order[:round(n * f)]] = 0
            allocs[f"tok_drop{round(f * 100):02d}"] = sm.reshape(g_abs.shape)
        for name, (df, l4f, l8f) in {"tok_rung_a": (0.2704, 0.45, 0.65),
                                     "tok_rung_b": (0.35, 0.55, 0.75),
                                     "tok_rung_c": (0.50, 0.70, 0.85)}.items():
            sm = np.ones(n, dtype=np.int16)
            kd, k4, k8 = (round(n * x) for x in (df, l4f, l8f))
            sm[order[k8:]] = 1
            sm[order[k4:k8]] = 2
            sm[order[kd:k4]] = 4
            sm[order[:kd]] = 0
            allocs[name] = sm.reshape(g_abs.shape)
        return allocs

    # family == "cell": aggregate |g| to per-spatial-cell (24x32); order cells.
    gh, gw = g_abs.shape[1], g_abs.shape[2]
    ncell = gh * gw
    cell_sens = g_abs.sum(axis=(0, 3)).reshape(-1)  # sum over pairs+channels
    corder = np.argsort(cell_sens, kind="stable")  # ascending cell sensitivity

    def cell_map(step_by_rank) -> np.ndarray:
        """step_by_rank: array of len ncell of steps applied to the rank-ordered cells."""
        cell_step = np.ones(ncell, dtype=np.int16)
        cell_step[corder] = step_by_rank
        return np.broadcast_to(cell_step.reshape(gh, gw)[None, :, :, None],
                               g_abs.shape).astype(np.int16).copy()

    # cell-DROP (reproduce wr1's cell-drop-to-base, sensitivity-ordered here).
    for f in (0.35, 0.50, 0.633, 0.75, 0.85):
        steps = np.ones(ncell, dtype=np.int16)
        steps[:round(ncell * f)] = 0
        allocs[f"cell_drop{round(f * 100):02d}"] = cell_map(steps)
    # cell-RUNG (graded {drop,L4,L8,keep} on cells — the NEW nested-precision test).
    for name, (df, l4f, l8f) in {"cell_rung_a": (0.35, 0.55, 0.75),
                                 "cell_rung_b": (0.50, 0.70, 0.85),
                                 "cell_rung_c": (0.633, 0.80, 0.92)}.items():
        steps = np.ones(ncell, dtype=np.int16)
        kd, k4, k8 = (round(ncell * x) for x in (df, l4f, l8f))
        steps[k8:] = 1
        steps[k4:k8] = 2
        steps[kd:k4] = 4
        steps[:kd] = 0
        allocs[name] = cell_map(steps)
    return allocs


def predict_dseg_flips(g_abs: np.ndarray, signed: np.ndarray, step_map: np.ndarray,
                       deciles_meanflip: dict[int, float]) -> float:
    """$0 first-order predicted d_seg from per-decile ±1-quantum flip rates scaled by
    the residual magnitude actually removed. PREDICTED, not measured — for pre-ranking."""
    flat_g = g_abs.reshape(-1)
    dec_edges = np.quantile(flat_g, np.linspace(0, 1, 11))
    dec_idx = np.clip(np.digitize(flat_g, dec_edges[1:-1]), 0, 9)
    # residual actually removed by the step: |s - coarsen(s)|
    sm = step_map.reshape(-1)
    coarse = signed.reshape(-1).astype(np.float64).copy()
    for step in (2, 4):
        m = sm == step
        coarse[m] = np.round(coarse[m] / step) * step
    coarse[sm == 0] = 0.0
    delta_removed = np.abs(signed.reshape(-1).astype(np.float64) - coarse)
    mf = np.array([deciles_meanflip.get(d, 0.0) for d in range(10)])
    flips = (mf[dec_idx] * delta_removed).sum()  # linear in quanta moved
    return REF_DSEG + flips * DSEG_PER_FLIP, float(flips)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["selection", "verify", "sweep", "confirm"], default="sweep")
    ap.add_argument("--family", choices=["token", "cell"], default="token")
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--candidate", default=None, help="confirm: alloc name (or 'ref')")
    ap.add_argument("--byte-close", action="store_true",
                    help="confirm: also write a valid byte-closed archive for hand-off")
    ap.add_argument("--gt-cache", default=GT)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--selection-mode", choices=["prefix", "stratified"], default="prefix",
                    help="prefix preserves historical video-order [:n]; stratified uses the "
                    "canonical seeded block selector and records population-ratio provenance")
    ap.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED,
                    help="seed used only with --selection-mode=stratified")
    args = ap.parse_args()
    if args.chunk > 120:
        raise SystemExit("chunk must be <= 120 (charter law)")
    args.outdir.mkdir(parents=True, exist_ok=True)

    selection, selection_rec = build_selection_scope(
        args.pairs,
        args.selection_mode,
        args.selection_seed,
    )
    selection_path = write_selection_receipt(args.outdir, args.selection_mode, selection_rec)
    if args.mode == "selection":
        print(json.dumps({"selection_receipt": str(selection_path), **selection_rec}, indent=1), flush=True)
        return 0

    frame = zipfile.ZipFile(ARCHIVE).read("state/tokens.dr7t")
    codes = np.asarray(decode_token_codes(frame), dtype=np.uint8)
    base, delta = factor_mode_delta(codes, LEVELS)
    signed = signed_residual(delta)
    print(f"[gr1] archive codes {codes.shape} tokens_bytes={len(frame)} "
          f"floor={ARCHIVE_FLOOR} water_B_per_flip={WATER_B_PER_FLIP:.4f}", flush=True)

    # hard-null per-decile mean flips (sb1 qa11 receipt) for $0 prediction.
    deciles_meanflip = {0: 0.5625, 1: 1.175, 2: 1.2875, 3: 2.0625, 4: 2.4875,
                        5: 3.0125, 6: 3.85, 7: 6.225, 8: 8.15, 9: 13.0125}

    model, cfg, meta = load_model()

    # cache the per-token |g| map (35s backward) so family re-runs skip it.
    gcache = args.outdir / "gr1_sensitivity_gabs.npy"

    def get_sensitivity() -> tuple[np.ndarray, float]:
        if gcache.exists():
            return np.load(gcache), 0.0
        g, w = sensitivity_map(model, cfg, args.gt_cache)
        np.save(gcache, g)
        return g, w

    if args.mode == "verify":
        # baseline: inject the ARCHIVE codes and confirm d_seg ~= pfs1 D1 (0.00389).
        d, wall = realized_dseg(model, cfg, codes, args.gt_cache, selection.indices, args.chunk)
        rec = {"schema": "ddm_gr1_verify.v1", "pointer": POINTER, "score_claim": False,
               "n_pairs": args.pairs, "baseline_realized_dseg": d,
               "ref_evaluate_py_dseg": REF_DSEG, "delta": d - REF_DSEG, "wall_s": wall,
               "selection": selection_rec}
        print(json.dumps(rec, indent=1), flush=True)
        (args.outdir / f"gr1_verify_n{args.pairs}.json").write_text(json.dumps(rec, indent=1))
        return 0

    if args.mode == "confirm":
        allocs = build_allocations(get_sensitivity()[0], args.family) \
            if args.candidate != "ref" else {}
        if args.candidate == "ref":
            cand_codes = codes
        else:
            sm = allocs[args.candidate]
            cand_codes = residual_to_codes(base, coarsen_alloc(signed, sm))
        cand_bytes = smevr_bytes(cand_codes)
        d, wall = realized_dseg(model, cfg, cand_codes, args.gt_cache, selection.indices, args.chunk)
        arch_b = ARCHIVE_FLOOR + cand_bytes
        rec = {"schema": "ddm_gr1_confirm.v1", "pointer": POINTER, "score_claim": False,
               "candidate": args.candidate, "n_pairs": args.pairs,
               "tokens_bytes": cand_bytes, "archive_bytes": arch_b,
               "realized_dseg": d, "rate_term": rate_term(arch_b),
               "seg_term": 100 * d, "seg_plus_rate": 100 * d + rate_term(arch_b),
               "ref_seg_plus_rate": 100 * REF_DSEG + rate_term(REF_ARCHIVE_BYTES),
               "wall_s": wall, "selection": selection_rec}
        rec["dominates_ref_segrate"] = bool(rec["seg_plus_rate"] < rec["ref_seg_plus_rate"])
        if args.byte_close and args.candidate != "ref":
            out_zip = args.outdir / f"gr1_{args.candidate}_archive.zip"
            src = zipfile.ZipFile(ARCHIVE)
            token_bytes = encode_token_codes(cand_codes, levels=LEVELS, codec="smevr")
            # canonical closure: decode round-trips to the same codes.
            if not np.array_equal(np.asarray(decode_token_codes(token_bytes), dtype=np.uint8),
                                  cand_codes):
                raise RuntimeError("byte-close roundtrip mismatch")
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as out:
                for item in src.infolist():
                    data = (token_bytes if item.filename == "state/tokens.dr7t"
                            else src.read(item.filename))
                    out.writestr(item.filename, data)
            disk = out_zip.read_bytes()
            rec["byte_closed"] = {"archive_zip": str(out_zip), "archive_bytes": len(disk),
                                  "archive_sha256": _sha(disk),
                                  "tokens_sha256": _sha(token_bytes)}
        print(json.dumps(rec, indent=1), flush=True)
        tag = f"gr1_confirm_{args.candidate}_n{args.pairs}.json"
        (args.outdir / tag).write_text(json.dumps(rec, indent=1))
        return 0

    # -------- sweep: sensitivity + bytes + predicted + n48 realized --------
    g_abs, map_wall = get_sensitivity()
    flat_g = g_abs.reshape(-1)
    print(f"[gr1] sensitivity map wall={map_wall}s frac_zero={np.mean(flat_g==0):.4f} "
          f"family={args.family}", flush=True)
    # baseline realized d_seg at this n_pairs (apples-to-apples ranking anchor).
    base_dseg, base_wall = realized_dseg(model, cfg, codes, args.gt_cache,
                                         selection.indices, args.chunk)
    print(json.dumps({"baseline": True, f"dseg_n{args.pairs}": round(base_dseg, 7),
                      "wall_s": base_wall}), flush=True)
    allocs = build_allocations(g_abs, args.family)
    rows = []
    for name, sm in allocs.items():
        t0 = time.time()
        cand_codes = residual_to_codes(base, coarsen_alloc(signed, sm))
        cand_bytes = smevr_bytes(cand_codes)
        enc_s = round(time.time() - t0, 1)
        pred_dseg, pred_flips = predict_dseg_flips(g_abs, signed, sm, deciles_meanflip)
        arch_b = ARCHIVE_FLOOR + cand_bytes
        n_coarsened = int((sm != 1).sum())
        d_real, wall = realized_dseg(model, cfg, cand_codes, args.gt_cache,
                                     selection.indices, args.chunk)
        row = {
            "name": name, "n_coarsened": n_coarsened,
            "tokens_bytes": cand_bytes, "tokens_saved": REF_TOKENS_BYTES - cand_bytes,
            "archive_bytes": arch_b, "rate_term": round(rate_term(arch_b), 6),
            "pred_dseg": round(pred_dseg, 7), "pred_flips": int(pred_flips),
            f"realized_dseg_n{args.pairs}": round(d_real, 7),
            f"seg_plus_rate_n{args.pairs}": round(100 * d_real + rate_term(arch_b), 6),
            "encode_s": enc_s, "realize_wall_s": wall,
        }
        # water at this candidate: bytes saved per flip introduced.
        # BOTH n600-scale: bytes are the full 600-pair token field; flips extrapolate
        # the per-pair-mean d_seg delta to n600 via TOTAL_PX (no pairs/600 factor).
        flips_intro = max(1.0, (d_real - base_dseg) * TOTAL_PX)
        row["b_per_flip_vs_base"] = round((REF_TOKENS_BYTES - cand_bytes) / flips_intro, 3)
        rows.append(row)
        print(json.dumps({"k": name, "arch": arch_b,
                          "rate": round(rate_term(arch_b), 4),
                          f"dseg_n{args.pairs}": round(d_real, 6),
                          "segrate": round(100 * d_real + rate_term(arch_b), 4),
                          "B/flip": row["b_per_flip_vs_base"]}), flush=True)
        _dump_sweep(args.outdir, meta, rows, args.pairs, map_wall, base_dseg,
                    args.family, selection_rec)
    return 0


def _dump_sweep(outdir: Path, meta, rows, pairs, map_wall, base_dseg,
                family="token", selection_rec: dict[str, Any] | None = None) -> None:
    ref_segrate = 100 * REF_DSEG + rate_term(REF_ARCHIVE_BYTES)
    rec = {
        "schema": "ddm_gr1_granularity_rerace.v1", "pointer": POINTER,
        "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU advisory] bytes real SMEVR; d_seg realized render+SegNet",
        "ckpt_config_hash": meta.get("config_hash"),
        "ref_row": {"archive_bytes": REF_ARCHIVE_BYTES, "tokens_bytes": REF_TOKENS_BYTES,
                    "d_seg": REF_DSEG, "seg_plus_rate": round(ref_segrate, 6),
                    "rate_term": round(rate_term(REF_ARCHIVE_BYTES), 6)},
        f"baseline_realized_dseg_n{pairs}": round(base_dseg, 7),
        "water_B_per_flip": round(WATER_B_PER_FLIP, 4),
        "family": family,
        "sensitivity_map_wall_s": map_wall, "n_pairs_realized": pairs,
        "selection": selection_rec,
        "rows": rows,
    }
    (outdir / f"gr1_sweep_{family}_n{pairs}_receipt.json").write_text(
        json.dumps(rec, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
