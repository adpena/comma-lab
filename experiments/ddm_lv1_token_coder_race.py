"""ddm_lv1 Phase C — token-stream compression stack race on the REAL tb1 payload.

Operator steer 2026-07-28 ("tokens might be well suited to factorization and
compactness and possibly dynamic adaptive quantization and/or truncation") applied
in the PROVEN fc1/oc1 stage order: solved-object -> FACTORIZE -> dynamic-adaptive-
QUANTIZE -> TRUNCATE -> ENTROPY-CODE (-> byte-close owed at the E4/WS1 exporter).
Entropy race per the lv1 charter C: MPEG-4 INTER-CAE (xi-motion-compensated template
contexts; identity-xi in the image chart per op1 98.806% stationary flip mass) vs
pp1-style KT context-arith vs order-0 (rANS-class) vs general-purpose coders.

Payload: the DECODE-RELEVANT description field q = quantize(clip(base+delta), L)
from the tb1 T2 lotto FINAL EMA checkpoint (the sealed burn arm's precedent payload)
— (600, 24, 32, 4) uint8 in [0, L-1].

Honesty labels:
  * Closed-form Dirichlet-multinomial (KT alpha=0.5) code lengths equal a real
    single-pass adaptive-arithmetic coder's bytes to <0.01% (pp1 roundtrip_proof
    precedent, experiments/ddm_pp1_direct_partition_coder.py). All contexts are
    STRICTLY CAUSAL (raster scan / previous frame / higher bit-planes), so a
    symmetric decoder exists; decoder WIRING is owed at the exporter (E4/WS1).
  * Stage-2/3 rows are LOSSY: bytes are measured immediately; each variant's
    full-n600 realized d_seg through the frozen CPU-torch scorer (--stage validate)
    is REQUIRED before any adopt verdict (realized-flip gate law — never truncate
    on smooth-loss/byte say-so; fd2 lesson applied to coding decisions).
  * Evidence axis [macOS-CPU/MLX advisory]; score_claim=false; POINTER UNMOVED.
"""

from __future__ import annotations

import argparse
import json
import lzma
import time
import zlib
from pathlib import Path

import numpy as np

from tac.payload_retention import retain_candidates, retain_payload, retention_root

POINTER_LINE = "0.1910828242 [contest-CPU] UNMOVED"
DEFAULT_CKPT = ("/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/t2_n600_lotto/"
                "checkpoints/stage_seg_trunk_tau_final.npz")
DEFAULT_GT = ("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/"
              "gt_n600.npz")
BORDER = 99  # context fill sentinel (outside any token alphabet)


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------
def load_payload(ckpt: str) -> tuple[np.ndarray, dict]:
    z = np.load(ckpt, allow_pickle=False)
    meta = json.loads(bytes(z["meta::json"]).decode())
    cfg = meta["cfg"]
    levels = int(cfg["token_quant_levels"])
    base = z["ema::tokens_base"].astype(np.float32)
    delta = z["ema::tokens_delta"].astype(np.float32)
    t = np.clip(base[None] + delta, -1.0, 1.0)
    q = np.round((t + 1.0) * 0.5 * (levels - 1)).astype(np.uint8)  # (P,gh,gw,c)
    return q, {"levels": levels, "cfg": cfg, "ckpt": ckpt,
               "config_hash": meta.get("config_hash")}


# ---------------------------------------------------------------------------
# closed-form adaptive (KT) code length — generalization of pp1's proven form
# ---------------------------------------------------------------------------
def kt_bytes(symbols: np.ndarray, ctx: np.ndarray, alphabet: int,
             alpha: float = 0.5) -> tuple[float, int]:
    from scipy.special import gammaln
    x = symbols.reshape(-1).astype(np.int64)
    cid = np.unique(ctx.reshape(-1), return_inverse=True)[1]
    ncomb = int(cid.max()) + 1
    cnt = np.bincount(cid * alphabet + x,
                      minlength=ncomb * alphabet).reshape(ncomb, alphabet).astype(np.float64)
    nc = cnt.sum(1)
    cc = gammaln(alphabet * alpha) - alphabet * gammaln(alpha)
    bits = -(ncomb * cc - gammaln(nc + alphabet * alpha).sum()
             + gammaln(cnt + alpha).sum()) / np.log(2.0)
    return float(bits / 8.0), ncomb


def _shift2(f: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Causal spatial shift on (P,H,W) with BORDER fill."""
    out = np.full_like(f, BORDER, dtype=np.int64)
    h, w = f.shape[1], f.shape[2]
    ys0, ye0 = max(0, dy), h + min(0, dy)
    xs0, xe0 = max(0, dx), w + min(0, dx)
    ys1, ye1 = max(0, -dy), h + min(0, -dy)
    xs1, xe1 = max(0, -dx), w + min(0, -dx)
    out[:, ys0:ye0, xs0:xe0] = f[:, ys1:ye1, xs1:xe1]
    return out


def _tprev(f: np.ndarray) -> np.ndarray:
    out = np.full_like(f, BORDER, dtype=np.int64)
    out[1:] = f[:-1]
    return out


def ctx_pack(maps: list[np.ndarray]) -> np.ndarray:
    ctx = np.zeros_like(maps[0], dtype=np.int64)
    for m in maps:
        ctx = ctx * (BORDER + 1) + m
    return ctx


def race_channel_coders(q: np.ndarray, alphabet: int) -> dict[str, float]:
    """Adaptive-context rows on a (P,gh,gw,c) stream; per-channel tables, summed."""
    rows: dict[str, float] = {}
    p, gh, gw, c = q.shape
    for name in ("kt_o0", "kt_intra_o4", "kt_prev1", "kt_inter_cae"):
        tot = 0.0
        for ch in range(c):
            f = q[..., ch].astype(np.int64)
            if name == "kt_o0":
                ctx = np.zeros_like(f)
            elif name == "kt_intra_o4":
                ctx = ctx_pack([_shift2(f, 0, -1), _shift2(f, -1, 0),
                                _shift2(f, -1, -1), _shift2(f, -1, 1)])
            elif name == "kt_prev1":
                ctx = _tprev(f)
            else:  # INTER-CAE: causal spatial template + xi-advected (identity
                #    image-chart) co-located previous-frame token — MPEG-4 INTER-CAE
                #    context structure adapted to the L-ary token lattice (#574).
                ctx = ctx_pack([_shift2(f, 0, -1), _shift2(f, -1, 0), _tprev(f)])
            b, _ = kt_bytes(f, ctx, alphabet)
            tot += b
        rows[name] = round(tot, 1)
    # CAE-BITPLANE INTER (the literal binary-CAE adaptation): MSB->LSB planes;
    # ctx = 4 causal same-plane bits + co-located prev-frame same-plane bit +
    # the cell's already-decoded higher bits (progressive L-ary via binary planes).
    nbits = int(np.ceil(np.log2(alphabet)))
    tot = 0.0
    for ch in range(c):
        f = q[..., ch].astype(np.int64)
        for plane in range(nbits - 1, -1, -1):
            bit = (f >> plane) & 1
            higher = f >> (plane + 1)
            pb = _tprev(bit)
            ctx = ctx_pack([_shift2(bit, 0, -1), _shift2(bit, -1, 0),
                            _shift2(bit, -1, -1), _shift2(bit, -1, 1), pb, higher])
            b, _ = kt_bytes(bit, ctx, 2)
            tot += b
    rows["cae_bitplane_inter"] = round(tot, 1)
    return rows


def race_generic_coders(
    q: np.ndarray,
    *,
    retain_dir: Path,
    label: str,
) -> dict[str, object]:
    """Race the general-purpose coders, RETAINING every candidate's bytes.

    ALWAYS KEEP THE PAYLOAD (P0, operator 2026-08-09): this race is the exact shape of
    the ``ans_real_n600.py`` incident — six real coder payloads materialized, only their
    lengths kept. Every candidate is persisted, not just the winner, because the anchor's
    discarded loser turned out to be a -2,120 B win. ``_retained`` carries each payload's
    path, byte count and sha256 into the receipt so the exporter can consume these bytes
    byte-identically instead of re-encoding them.
    """
    raw = q.tobytes()
    d = q.copy()
    d[1:] = (q[1:].astype(np.int16) - q[:-1].astype(np.int16)) % 256
    dt = d.astype(np.uint8).tobytes()
    payloads: dict[str, bytes] = {
        "zlib9_raw": zlib.compress(raw, 9),
        "zlib9_tdelta": zlib.compress(dt, 9),
        "lzma_raw": lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME),
        "lzma_tdelta": lzma.compress(dt, preset=9 | lzma.PRESET_EXTREME),
    }
    brotli_error: str | None = None
    try:
        import brotli
        payloads["brotli_q11_raw"] = brotli.compress(raw, quality=11)
        payloads["brotli_q11_tdelta"] = brotli.compress(dt, quality=11)
    except Exception as exc:  # dep policy: report, never silently skip
        brotli_error = str(exc)

    rows: dict = {name: len(blob) for name, blob in payloads.items()}
    # The coder INPUTS are retained too: without them the race is not reproducible
    # byte-identically, which is the property the retention rule exists to protect.
    custody = retain_candidates(
        retain_dir, {f"{label}.{name}": blob for name, blob in payloads.items()}
    )
    custody[f"{label}.source_raw"] = retain_payload(
        retain_dir / f"{label}.source_raw.bin", raw
    )
    custody[f"{label}.source_tdelta"] = retain_payload(
        retain_dir / f"{label}.source_tdelta.bin", dt
    )
    rows["_retained"] = custody
    if brotli_error is not None:
        rows["brotli_error"] = brotli_error
    return rows


# ---------------------------------------------------------------------------
# Stage 1 — factorization (lossless re-representations of q)
# ---------------------------------------------------------------------------
def factor_static_delta(q: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """base_hat = per-cell temporal MODE; d = (q - base_hat) mod levels.
    Exact reconstruction: q = (base_hat + d) mod levels."""
    p, gh, gw, c = q.shape
    flat = q.reshape(p, -1)
    base = np.empty(flat.shape[1], dtype=np.uint8)
    for j in range(flat.shape[1]):
        base[j] = np.bincount(flat[:, j], minlength=levels).argmax()
    base = base.reshape(gh, gw, c)
    d = (q.astype(np.int16) - base[None].astype(np.int16)) % levels
    return base, d.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2/3 — margin-slack-priced quantization + truncation (LOSSY)
# ---------------------------------------------------------------------------
def cell_min_margin(gt_npz: str, p: int, gh: int, gw: int, d_pix: int) -> np.ndarray:
    """Per (pair, cell) minimum GT margin over the DxD pixel block (flip-proneness)."""
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    margins = open_stored_npy_memmap(Path(gt_npz), "margins")  # (P,384,512) f32
    out = np.empty((p, gh, gw), dtype=np.float32)
    for i in range(p):
        m = np.asarray(margins[i], dtype=np.float32)
        out[i] = m.reshape(gh, d_pix, gw, d_pix).min(axis=(1, 3))
    return out


def requant_cells(q: np.ndarray, coarse_mask: np.ndarray, sub_levels: int,
                  levels: int) -> np.ndarray:
    """Requantize masked cells onto a sub_levels lattice re-expressed on the
    levels lattice (single decode path; entropy drop from reduced support)."""
    qq = q.astype(np.float32)
    sub = np.round(np.round(qq * (sub_levels - 1) / (levels - 1))
                   * (levels - 1) / (sub_levels - 1)).astype(np.uint8)
    out = q.copy()
    out[coarse_mask] = sub[coarse_mask]
    return out


def truncate_to_base(q: np.ndarray, base: np.ndarray, cell_mask: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[cell_mask] = np.broadcast_to(base[None], q.shape)[cell_mask]
    return out


# ---------------------------------------------------------------------------
# validity: full-n600 realized d_seg through the frozen CPU scorer (LOSSY gates)
# ---------------------------------------------------------------------------
def realized_dseg_full(q_variant: np.ndarray, payload_meta: dict, gt_npz: str,
                       chunk: int = 32) -> dict:
    import mlx.core as mx

    from experiments.train_tr1_partition_renderer_mlx import TR1Config, build_module
    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_batch,
    )
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet

    if chunk > 120:
        raise SystemExit("chunk must be <= 120 (charter n600 chunk rule)")
    cfgd = dict(payload_meta["cfg"])
    cfgd.setdefault("token_init_mode", "zero")
    cfg = TR1Config(**{k: cfgd[k] for k in TR1Config.__dataclass_fields__ if k in cfgd})
    model = build_module(cfg)
    z = np.load(payload_meta["ckpt"], allow_pickle=False)
    # EMA params are the verdict basis (tb1 law); override tokens with the variant
    # lattice points (base=0, delta=lattice values -> round-STE is idempotent there).
    updates = []
    for k in z.files:
        if k.startswith("ema::"):
            updates.append((k[len("ema::"):], mx.array(z[k])))
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(updates))
    lv = float(payload_meta["levels"] - 1)
    t_lat = (q_variant.astype(np.float32) / lv) * 2.0 - 1.0
    model.tokens_base = mx.zeros(t_lat.shape[1:])
    model.tokens_delta = mx.array(t_lat)
    mx.eval(model.parameters())
    lstars = open_stored_npy_memmap(Path(gt_npz), "lstars")
    seg_cpu = load_real_segnet("cpu")
    p = q_variant.shape[0]
    dsegs: list[float] = []
    t0 = time.monotonic()
    for c0 in range(0, p, chunk):
        cams, gts = [], []
        with mx.stream(mx.cpu):
            for i in range(c0, min(c0 + chunk, p)):
                rgb = model.render_frame(int(i))
                mx.eval(rgb)
                cams.append(_torch_R_to_camera_uint8(np.asarray(rgb, dtype=np.float32)[0]))
                gts.append(np.asarray(lstars[i], dtype=np.int64))
        dsegs.extend(cpu_verdict_d_seg_batch(seg_cpu, cams, gts))
    return {"full_dseg": float(np.mean(dsegs)), "max_pair": float(np.max(dsegs)),
            "pairs": p, "wall_s": round(time.monotonic() - t0, 1),
            "basis": "ema_shadow", "chunk": chunk,
            "evidence_axis": "[macOS-CPU/MLX advisory]", "score_claim": False}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_variants(q: np.ndarray, base: np.ndarray, meta: dict, gt_npz: str):
    levels = meta["levels"]
    p, gh, gw, c = q.shape
    d_pix = 384 // gh
    mm = cell_min_margin(gt_npz, p, gh, gw, d_pix)  # (P,gh,gw)
    deep = (mm > 0.25)[..., None].repeat(c, axis=3)   # op1 gate-stable threshold set
    mid = ((mm > 0.10) & (mm <= 0.25))[..., None].repeat(c, axis=3)
    variants: dict[str, np.ndarray] = {}
    variants["Q1_deepL8"] = requant_cells(q, deep, 8, levels)
    q2 = requant_cells(q, deep, 4, levels)
    variants["Q2_deepL4_midL8"] = requant_cells(q2, mid, 8, levels)
    d_int = (q.astype(np.int16) - base[None].astype(np.int16))
    small = np.abs(d_int) <= 1
    variants["T1_revert_small_deep"] = truncate_to_base(q, base, small & deep)
    variants["QT_Q2_plus_T1"] = truncate_to_base(variants["Q2_deepL4_midL8"], base,
                                                 small & deep)
    stats = {k: {"cells_changed": int(np.count_nonzero(v != q)),
                 "frac_changed": round(float(np.mean(v != q)), 4)}
             for k, v in variants.items()}
    return variants, stats, {"deep_frac": round(float(deep.mean()), 4),
                             "mid_frac": round(float(mid.mean()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--gt-cache", default=DEFAULT_GT)
    ap.add_argument("--stage", choices=("bytes", "validate"), default="bytes")
    ap.add_argument("--variants", default="Q1_deepL8,Q2_deepL4_midL8,"
                                          "T1_revert_small_deep,QT_Q2_plus_T1")
    ap.add_argument("--out", default="/Volumes/VertigoDataTier/pact/ddm_lv1_20260728/"
                                     "c_token_stack_race/receipt.json")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--retain-dir", default=None,
                    help="where coder payloads are retained; default resolves the SSD "
                         "tier waterfall (ALWAYS KEEP THE PAYLOAD, P0)")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resolved BEFORE any encode: retention is a precondition for running, so a full
    # tier must fail the run here rather than after minutes of discarded compression.
    retain_dir = (Path(args.retain_dir) if args.retain_dir
                  else retention_root("ddm_lv1_token_coder_race"))
    retain_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict = {}
    if out_path.exists():
        receipt = json.loads(out_path.read_text())
    receipt.setdefault("pointer", POINTER_LINE)
    receipt.setdefault("score_claim", False)
    receipt.setdefault("evidence_axis", "[macOS-CPU/MLX advisory]")

    q, meta = load_payload(args.ckpt)
    receipt["payload"] = {"ckpt": meta["ckpt"], "shape": list(q.shape),
                          "levels": meta["levels"],
                          "config_hash": meta["config_hash"]}
    base, d_stream = factor_static_delta(q, meta["levels"])

    if args.stage == "bytes":
        rows: dict = {}
        rows["F0_monolithic"] = {
            **race_generic_coders(q, retain_dir=retain_dir, label="F0_monolithic"),
            **race_channel_coders(q, meta["levels"])}
        base_payload = zlib.compress(base.tobytes(), 9)
        base_bytes = {
            "zlib9": len(base_payload),
            "_retained": retain_payload(
                retain_dir / "F1_staticbase.zlib9.bin", base_payload)}
        rows["F1_staticbase_delta"] = {
            "base": base_bytes,
            "delta": {
                **race_generic_coders(
                    d_stream, retain_dir=retain_dir, label="F1_delta"),
                **race_channel_coders(d_stream, meta["levels"])}}
        variants, vstats, mstats = build_variants(q, base, meta, args.gt_cache)
        rows["stage23_variants"] = {"margin_cell_stats": mstats, "changed": vstats}
        for name, v in variants.items():
            vb, vd = factor_static_delta(v, meta["levels"])
            delta_race = race_generic_coders(
                vd, retain_dir=retain_dir, label=f"variant_{name}_delta")
            vbase_payload = zlib.compress(vb.tobytes(), 9)
            rows["stage23_variants"][name] = {
                "F1_delta_kt_inter_cae": race_channel_coders(vd, meta["levels"])[
                    "kt_inter_cae"],
                "F1_delta_zlib9_tdelta": delta_race["zlib9_tdelta"],
                "F1_delta_retained": delta_race["_retained"],
                "F1_base_zlib9": len(vbase_payload),
                "F1_base_retained": retain_payload(
                    retain_dir / f"variant_{name}_base.zlib9.bin", vbase_payload),
                "dseg_validity": "PENDING (--stage validate; lossy adopt gate)"}
        receipt["bytes_race"] = rows
        receipt["ledger_anchor_note"] = (
            "tb1 T2 ledger tokens 531,097 B coded quantize(base)+quantize(delta) "
            "SEPARATELY (COUNTED-ESTIMATE); this race codes the DECODE-RELEVANT "
            "field q=quantize(clip(base+delta)) and its exact lossless "
            "re-factorizations — the object the exporter will actually ship")
    else:
        variants, _, _ = build_variants(q, base, meta, args.gt_cache)
        receipt.setdefault("validate", {})
        receipt["validate"]["baseline_q"] = realized_dseg_full(
            q, meta, args.gt_cache, args.chunk)
        for name in args.variants.split(","):
            name = name.strip()
            if not name:
                continue
            receipt["validate"][name] = realized_dseg_full(
                variants[name], meta, args.gt_cache, args.chunk)
            out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    out_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt.get("bytes_race", receipt.get("validate")), indent=2,
                     sort_keys=True)[:4000])
    print("receipt:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
