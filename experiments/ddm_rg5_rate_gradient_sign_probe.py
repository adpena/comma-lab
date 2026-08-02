"""ddm_rg5 — does the in-loop RATE gradient point DOWNHILL in real shipped bytes?

`ddm_gc17` §3c concludes from `ddm_rsf1` that the live `--rate-model entropy` term is an
"ANTI-CORRELATED surrogate" and that the burn's rate leg "points backwards". That conclusion is
drawn from a **trajectory rank correlation** (Spearman rho between the surrogate level and real
SMEVR bytes across checkpoints of one run). A trajectory correlation cannot establish the sign of
a GRADIENT: along a burn both quantities move under the *seg* force (`w_seg = 100` vs
`w_rate = 0.05`), so co-movement measures the joint dynamics, not the rate term's own push.

This harness measures the two things a trajectory correlation cannot:

  **P1 DIRECTIONAL-DERIVATIVE (the sign test).** Take the live field, compute the EXACT gradient
  of the trainer's own `token_rate_term` w.r.t. the token field, step along `-grad`, requantize,
  and encode with the SHIPPED r7 SMEVR coder. If real bytes FALL, the term's sign is correct at
  the operating point; if they RISE, the gradient is genuinely backwards. Ascent (`+grad`) and a
  random direction of identical RMS are run as controls, and step size is swept so the answer is
  reported as a curve, not a point.

  **P2 PERMUTATION SENSITIVITY (the mechanism test).** gc17's mechanism claim is that a marginal
  histogram is permutation-invariant while SMEVR's cost is temporal, so the surrogate is
  structurally blind. That is checkable directly and without any correlation: permute the pair
  axis (which leaves the pooled histogram's multiset identical BY CONSTRUCTION) and measure how
  far real SMEVR bytes move. The permutation-reachable byte spread is the size of the surrogate's
  provable blind subspace; comparing it to the trajectory's own byte range says whether blindness
  can account for the observed anti-correlation.

Controls (P4 "no meter without a canary"): identity permutation and zero-step must reproduce the
baseline byte count EXACTLY (negative control); a lattice-scale value perturbation must move both
the surrogate and the bytes (positive control).

Authority: `[macOS-CPU advisory]`, `score_claim=false`, `promotable=false`. $0, scorer-free,
training-free. Byte columns are EXACT (the r7 encoder is deterministic and lossless). Surrogate
columns call the trainer's own `_soft_hist_entropy_bits` / `token_rate_term` branches, never a
reimplementation. Field reconstruction reuses `ddm_rsf1`'s `load_field` verbatim.

Usage:

    .venv/bin/python experiments/ddm_rg5_rate_gradient_sign_probe.py \
        --run ddm_r1c_20260731/window_01 --ckpt intra_seg_trunk_tau_ep00640.npz \
        --out .omx/research/ddm_rg5_rows_20260801.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Run-as-script support: resolve the repo root from THIS file (never a hardcoded absolute path,
# and never the CWD) so `experiments.*` imports resolve from the same tree this file lives in —
# the shared-venv editable-install hijack guard.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_ROOT = Path("/Volumes/VertigoDataTier/pact")


def _lattice_step(levels: int) -> float:
    """Field-space width of one quantization level: the field lives in [-1, 1] and
    `quantize_tokens_np` maps it onto `levels` bins, so one bin is `2 / (levels - 1)` wide."""
    return 2.0 / float(levels - 1)


def real_bytes(field: np.ndarray, levels: int) -> dict:
    """EXACT shipped-coder bytes for a float field, via the trainer's quantizer + r7 SMEVR."""
    import struct

    from experiments.ddm_r7_token_coder import HEADER, encode_token_codes
    from experiments.train_tr1_partition_renderer_mlx import quantize_tokens_np

    q = quantize_tokens_np(field, levels)
    frame = encode_token_codes(np.ascontiguousarray(q), levels=levels, codec="smevr")
    f = HEADER.unpack_from(frame)
    base_len, delta_len = int(f[-3]), int(f[-2])
    occ_len = struct.unpack_from("<I", frame[HEADER.size + base_len:])[0]
    return {
        "smevr_bytes": len(frame),
        "smevr_base_bytes": base_len,
        "smevr_occupancy_bytes": int(occ_len),
        "smevr_value_bytes": int(delta_len - 4 - occ_len),
    }


def surrogate_values(field: np.ndarray, keep_bool: np.ndarray, levels: int) -> dict:
    """The trainer's two `--rate-model` branches, evaluated on the FULL pair set."""
    import mlx.core as mx

    from experiments.train_tr1_partition_renderer_mlx import _soft_hist_entropy_bits

    P, gh, gw, c = field.shape
    ki = mx.array(np.flatnonzero(keep_bool.ravel()).astype(np.int64))
    kept = mx.take(mx.array(field.reshape(P, gh * gw, c)), ki, axis=1)
    return {
        "surr_entropy_bits": float(_soft_hist_entropy_bits(mx.reshape(kept, (-1, c)), levels)),
        "surr_smevr_surrogate_bits": float(
            _soft_hist_entropy_bits(mx.reshape(0.5 * (kept[1:] - kept[:-1]), (-1, c)), levels)),
    }


def rate_gradient(field: np.ndarray, keep_bool: np.ndarray, levels: int, model: str,
                  batch: int | None = None, n_draws: int = 32, seed: int = 0) -> np.ndarray:
    """d(token_rate_term)/d(field) for one `--rate-model` branch, shaped like `field`.

    `batch=None` gives the POPULATION gradient (all pairs, one term). An integer `batch`
    averages the gradient of `n_draws` independent batch-`batch` estimates — the object the
    trainer's optimizer actually accumulates, whose bias relative to the population gradient is
    exactly the question a population-only measurement would beg.
    """
    import mlx.core as mx

    from experiments.train_tr1_partition_renderer_mlx import _soft_hist_entropy_bits

    P, gh, gw, c = field.shape
    ki = mx.array(np.flatnonzero(keep_bool.ravel()).astype(np.int64))
    flat = mx.array(field.reshape(P, gh * gw, c))

    def term(x, ids):
        kept = mx.take(mx.take(x, mx.array(np.asarray(ids, dtype=np.int64)), axis=0), ki, axis=1)
        if model == "smevr_surrogate":
            kept = 0.5 * (kept[1:] - kept[:-1])
        return _soft_hist_entropy_bits(mx.reshape(kept, (-1, c)), levels)

    gfn = mx.grad(term)
    if batch is None:
        g = gfn(flat, np.arange(P))
    else:
        rng = np.random.default_rng(seed)
        acc = None
        for _ in range(n_draws):
            ids = np.sort(rng.choice(P, size=min(batch, P), replace=False))
            gi = gfn(flat, ids)
            acc = gi if acc is None else acc + gi
        g = acc / float(n_draws)
    return np.asarray(g, dtype=np.float32).reshape(P, gh, gw, c)


def unit_rms(g: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize a direction to unit RMS so step sizes are comparable across directions."""
    rms = float(np.sqrt(np.mean(np.square(g))))
    if rms <= 0.0:
        raise SystemExit("rate gradient is identically zero — refuse to normalize (fail closed)")
    return g / rms, rms


def run_direction_sweep(field, keep_bool, levels, direction, label, alphas, base, out_rows):
    """Step `field + alpha * lattice * direction` and record EXACT bytes at each alpha."""
    lat = _lattice_step(levels)
    for alpha in alphas:
        t0 = time.time()
        moved = np.clip(field + (alpha * lat) * direction, -1.0, 1.0)
        row = {"probe": "P1_direction", "direction": label, "alpha_lattice": float(alpha)}
        row.update(surrogate_values(moved, keep_bool, levels))
        row.update(real_bytes(moved, levels))
        row["d_bytes"] = row["smevr_bytes"] - base["smevr_bytes"]
        row["d_value_bytes"] = row["smevr_value_bytes"] - base["smevr_value_bytes"]
        row["d_occupancy_bytes"] = row["smevr_occupancy_bytes"] - base["smevr_occupancy_bytes"]
        row["d_surr_entropy"] = row["surr_entropy_bits"] - base["surr_entropy_bits"]
        row["d_surr_smevr"] = row["surr_smevr_surrogate_bits"] - base["surr_smevr_surrogate_bits"]
        row["secs"] = round(time.time() - t0, 2)
        out_rows.append(row)
        print(f"  [{label:>22s}] a={alpha:+6.3f}  bytes {row['smevr_bytes']:8d} "
              f"({row['d_bytes']:+7d})  dH_ent {row['d_surr_entropy']:+.5f}  "
              f"dH_smevr {row['d_surr_smevr']:+.5f}", flush=True)


def run_permutation_probe(field, keep_bool, levels, base, n_perm, seed, out_rows):
    """P2: permute the PAIR axis. The pooled marginal histogram is invariant by construction;
    measure how far the SHIPPED bytes move to size the surrogate's provable blind subspace."""
    rng = np.random.default_rng(seed)
    P = field.shape[0]
    perms = [("identity", np.arange(P)), ("reverse", np.arange(P)[::-1].copy())]
    perms += [(f"random{i}", rng.permutation(P)) for i in range(n_perm)]
    for label, perm in perms:
        t0 = time.time()
        moved = np.ascontiguousarray(field[perm])
        row = {"probe": "P2_permutation", "perm": label}
        row.update(surrogate_values(moved, keep_bool, levels))
        row.update(real_bytes(moved, levels))
        row["d_bytes"] = row["smevr_bytes"] - base["smevr_bytes"]
        row["d_surr_entropy"] = row["surr_entropy_bits"] - base["surr_entropy_bits"]
        row["d_surr_smevr"] = row["surr_smevr_surrogate_bits"] - base["surr_smevr_surrogate_bits"]
        row["secs"] = round(time.time() - t0, 2)
        out_rows.append(row)
        print(f"  [{label:>10s}] bytes {row['smevr_bytes']:8d} ({row['d_bytes']:+7d})  "
              f"dH_ent {row['d_surr_entropy']:+.3e}  dH_smevr {row['d_surr_smevr']:+.5f}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir relative to --root")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alphas", default="0.0,0.05,0.1,0.25,0.5",
                    help="step sizes in LATTICE units (one level = 2/(levels-1) in field space)")
    ap.add_argument("--n-perm", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8, help="in-loop batch for the batch gradient")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-permutation", action="store_true")
    args = ap.parse_args()

    from experiments.ddm_rsf1_rate_surrogate_fidelity import load_field

    d = args.root / args.run
    cfg = json.loads((d / "tr1_config.json").read_text())
    field, basis, keep_bool = load_field(d / "checkpoints" / args.ckpt, cfg)
    levels = int(cfg["token_quant_levels"])

    base = {}
    base.update(surrogate_values(field, keep_bool, levels))
    base.update(real_bytes(field, levels))
    print(f"BASELINE {args.run}/{args.ckpt} basis={basis} shape={list(field.shape)} "
          f"levels={levels} keep={keep_bool.mean():.3f}")
    print(f"  smevr_bytes {base['smevr_bytes']} (value {base['smevr_value_bytes']}, "
          f"occ {base['smevr_occupancy_bytes']}, base {base['smevr_base_bytes']})")
    print(f"  entropy {base['surr_entropy_bits']:.6f} bits | "
          f"smevr_surrogate {base['surr_smevr_surrogate_bits']:.6f} bits")

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    rows: list[dict] = []
    meta = {"probe": "meta", "run": args.run, "ckpt": args.ckpt, "basis": basis,
            "levels": levels, "shape": list(field.shape), "keep_frac": float(keep_bool.mean()),
            "cfg_w_rate": cfg.get("w_rate"), "cfg_rate_model": cfg.get("rate_model"),
            "lattice_step": _lattice_step(levels), "baseline": base}

    print("\n=== P1 DIRECTIONAL DERIVATIVE (sign of the live rate gradient) ===")
    g_pop, rms_pop = unit_rms(rate_gradient(field, keep_bool, levels, "entropy"))
    g_bat, rms_bat = unit_rms(
        rate_gradient(field, keep_bool, levels, "entropy", batch=args.batch, seed=args.seed))
    cos_pop_bat = float(np.mean(g_pop * g_bat))  # both unit-RMS => mean product IS the cosine
    g_sm, rms_sm = unit_rms(rate_gradient(field, keep_bool, levels, "smevr_surrogate"))
    cos_pop_sm = float(np.mean(g_pop * g_sm))
    rng = np.random.default_rng(args.seed + 991)
    g_rand, _ = unit_rms(rng.standard_normal(field.shape).astype(np.float32))
    meta.update({"grad_rms_population": rms_pop, "grad_rms_batch": rms_bat,
                 "grad_rms_smevr_surrogate": rms_sm,
                 "cos_population_vs_batch": cos_pop_bat,
                 "cos_entropy_vs_smevr_surrogate": cos_pop_sm})
    print(f"  |grad| RMS: population {rms_pop:.3e} | batch-{args.batch} {rms_bat:.3e} | "
          f"smevr_surrogate {rms_sm:.3e}")
    print(f"  cosine(population, batch-{args.batch}) = {cos_pop_bat:+.4f}   "
          f"cosine(entropy, smevr_surrogate) = {cos_pop_sm:+.4f}")

    # Adam is a SIGN-PRESERVING preconditioner (it divides each coordinate by a positive RMS
    # estimate), so the realized update lies in the same orthant as -grad but NOT along it. The
    # sign-SGD limit is the far end of that family; running it bounds the preconditioner's effect
    # instead of assuming the raw-gradient answer transfers.
    g_sign, _ = unit_rms(np.sign(g_pop).astype(np.float32))

    pos = [a for a in alphas if a > 0.0]
    run_direction_sweep(field, keep_bool, levels, -g_pop, "entropy_DESCENT", alphas, base, rows)
    run_direction_sweep(field, keep_bool, levels, +g_pop, "entropy_ASCENT", pos, base, rows)
    run_direction_sweep(field, keep_bool, levels, -g_sign, "entropy_DESCENT_signSGD", pos, base,
                        rows)
    run_direction_sweep(field, keep_bool, levels, -g_bat, f"entropy_DESCENT_b{args.batch}", pos,
                        base, rows)
    run_direction_sweep(field, keep_bool, levels, -g_sm, "smevrsurr_DESCENT", pos, base, rows)
    run_direction_sweep(field, keep_bool, levels, g_rand, "random_CONTROL", pos, base, rows)

    if not args.skip_permutation:
        print("\n=== P2 PERMUTATION SENSITIVITY (size of the provable blind subspace) ===")
        run_permutation_probe(field, keep_bool, levels, base, args.n_perm, args.seed, rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        fh.write(json.dumps(meta) + "\n")
        for r in rows:
            r["run"] = args.run
            r["ckpt"] = args.ckpt
            fh.write(json.dumps(r) + "\n")
    print(f"\ndone -> {args.out}  ({len(rows) + 1} rows appended)")


if __name__ == "__main__":
    main()
