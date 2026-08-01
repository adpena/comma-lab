"""ddm_rsf1 — is the in-loop RATE surrogate aimed at the coder we actually ship?

Extends `.omx/research/ddm_fh1_forces_harvest_20260731.md` §1 row **A3** (which catalogues the
`--rate-model entropy|smevr_surrogate` option set) and closes the measurement gap left open by
`.omx/research/ddm_gd1_generic_default_census_20260731.md` row **T4** (which classifies the live
`entropy` default GENERIC-CHOSEN-UNRACED and ASSERTS "the burn's rate gradient is steered by a
coder-mismatched objective" without measuring it).

The decision statistic is RANK fidelity: a surrogate need not match bytes in magnitude, it must
ORDER token fields the way the SHIPPED coder prices them. This harness measures, per token field:

  (a) `entropy`          — the LIVE default: marginal soft-histogram entropy of kept-cell token
                           VALUES.                            [trainer's own `token_rate_term`]
  (b) `smevr_surrogate`  — the BUILT alternate: soft-histogram entropy of CONSECUTIVE-FRAME
                           deltas.                            [trainer's own `token_rate_term`]
  (c) mode-base residual — gd1 T4's named "derived candidate": soft-histogram entropy of
                           (value - stopgrad(per-cell temporal MODE)), i.e. SMEVR's OWN residual.
                           Differentiable in the values => buildable as an in-loop term.
  (d) soft occupancy     — differentiable P(mode-residual != 0): SMEVR's event stream, expressed
                           as a loss-shaped quantity.
  (e) REAL `smevr` bytes and REAL `brotli11` bytes via `ddm_r7_token_coder.encode_token_codes`,
      plus the SMEVR stream decomposition (base / occupancy / value) so each surrogate can be
      attributed to the part of the coder it actually tracks.

Authority: $0, scorer-free, training-free. Byte columns are EXACT (the r7 encoder is
deterministic and lossless); surrogate columns use the trainer's own `_soft_hist_entropy_bits`,
never a reimplementation. Field reconstruction is verified against the runs' own logged
`tokens_bytes` telemetry (see the receipt).

Receipt: `.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md`.

Usage (resumable; re-running skips rows already present in --out):

    .venv/bin/python experiments/ddm_rsf1_rate_surrogate_fidelity.py \
        --manifest .omx/research/ddm_rsf1_manifest_20260801.json \
        --out .omx/research/ddm_rsf1_rows_20260801.jsonl

The manifest is a JSON list of {"dir": <run dir relative to --root>, "ckpt": <npz name>,
"group": <comma-joined population tags>}.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

# Run-as-script support: resolve the repo root from THIS file (never a hardcoded absolute
# path, and never the CWD) so `experiments.*` imports resolve from the same tree this file
# lives in — the shared-venv editable-install hijack guard.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_ROOT = Path("/Volumes/VertigoDataTier/pact")


# --------------------------------------------------------------------------------------
# field reconstruction — must match the trainer's byte-close path EXACTLY
# --------------------------------------------------------------------------------------
def load_field(ckpt: Path, cfg: dict) -> tuple[np.ndarray, str, np.ndarray]:
    """Reconstruct the FULL float token field (P,gh,gw,c) exactly as the trainer's
    `_full_token_field_np` does: (base + delta) * cell_keep [-> row-band tie].

    Returns (field, basis, keep_bool). `basis` is 'ema' when the EMA shadow is present (the
    basis the trainer's own gate/ledger uses) else 'param'.
    """
    z = np.load(ckpt)
    pre = "ema::" if any(k.startswith("ema::tokens") for k in z.files) else "param::"
    if cfg["token_temporal_mode"] == "shared_base":
        field = z[pre + "tokens_base"][None] + z[pre + "tokens_delta"]
    else:
        field = z[pre + "tokens"]
    cm = cfg.get("token_cell_mask")
    if cm:
        keep = np.load(cm).astype(np.float32)[..., None]
        field = field * keep
        keep_bool = keep[..., 0] > 0.5
    else:
        keep_bool = np.ones(field.shape[1:3], dtype=bool)
    if cfg.get("token_rowband_spec"):
        raise SystemExit(
            f"row-band grammar present in {ckpt}: the offline reconstruction would not match the "
            "trainer's tied field — fail closed (never-invent geometry)")
    basis = "ema" if pre == "ema::" else "param"
    return np.ascontiguousarray(field.astype(np.float32)), basis, keep_bool


# --------------------------------------------------------------------------------------
# surrogates — evaluated with the TRAINER'S OWN entropy function
# --------------------------------------------------------------------------------------
def _kept(mx, field: np.ndarray, keep_bool: np.ndarray):
    P, gh, gw, c = field.shape
    ki = mx.array(np.flatnonzero(keep_bool.ravel()).astype(np.int64))
    return mx.take(mx.array(field.reshape(P, gh * gw, c)), ki, axis=1), ki, c


def surrogates(field: np.ndarray, keep_bool: np.ndarray, levels: int, q: np.ndarray) -> dict:
    """All four surrogate columns on the FULL pair set (ids = every pair).

    Mirrors `token_rate_term` exactly (reshape (-1,c) -> gather kept rows -> soft-hist entropy);
    the 'entropy' and 'smevr_surrogate' branches ARE the trainer's two `--rate-model` values.
    """
    import mlx.core as mx

    from experiments.ddm_r7_token_coder import factor_mode_delta
    from experiments.train_tr1_partition_renderer_mlx import _soft_hist_entropy_bits

    kept, ki, c = _kept(mx, field, keep_bool)
    out = {
        "surr_entropy_bits": float(_soft_hist_entropy_bits(mx.reshape(kept, (-1, c)), levels)),
        "surr_smevr_surrogate_bits": float(
            _soft_hist_entropy_bits(mx.reshape(0.5 * (kept[1:] - kept[:-1]), (-1, c)), levels)),
    }
    # (c)/(d): SMEVR's own mode-referenced residual.
    gh, gw = field.shape[1], field.shape[2]
    base_q, _ = factor_mode_delta(np.ascontiguousarray(q), levels)
    base_val = base_q.astype(np.float32) / (levels - 1) * 2.0 - 1.0
    bkept = mx.take(mx.array(base_val.reshape(gh * gw, c)), ki, axis=0)
    resid = mx.reshape(0.5 * (kept - mx.stop_gradient(bkept)[None]), (-1, c))
    out["surr_modebase_bits"] = float(_soft_hist_entropy_bits(resid, levels))
    L = float(levels - 1)
    x01 = mx.clip((resid + 1.0) * 0.5, 0.0, 1.0) * L
    d2 = (mx.reshape(x01, (-1, 1)) - mx.reshape(mx.arange(levels).astype(mx.float32), (1, -1))) ** 2
    out["surr_soft_occupancy"] = float(
        1.0 - mx.mean(mx.softmax(-d2 / 0.15, axis=-1)[:, int(round(0.5 * L))]))
    return out


def batch_dispersion(field, keep_bool, levels, n_draws=64, batch=8, seed=0) -> dict:
    """The in-loop term is a BATCH-`batch` estimate, not the population value. Measure its
    spread so ranking signal can be compared against the estimator's own sampling noise."""
    import mlx.core as mx

    from experiments.train_tr1_partition_renderer_mlx import _soft_hist_entropy_bits

    kept_all, _ki, c = _kept(mx, field, keep_bool)
    P = int(field.shape[0])
    batch = min(int(batch), P)
    if batch < 2:  # the consecutive-delta branch needs >=2 ids; report absent, never fake a 0.
        return {}
    rng = np.random.default_rng(seed)
    e, s = [], []
    for _ in range(n_draws):
        ids = np.sort(rng.choice(P, size=batch, replace=False))
        k = mx.take(kept_all, mx.array(ids.astype(np.int64)), axis=0)
        e.append(float(_soft_hist_entropy_bits(mx.reshape(k, (-1, c)), levels)))
        s.append(float(_soft_hist_entropy_bits(mx.reshape(0.5 * (k[1:] - k[:-1]), (-1, c)), levels)))
    return {"batch8_entropy_mean": float(np.mean(e)), "batch8_entropy_std": float(np.std(e)),
            "batch8_smevrsurr_mean": float(np.mean(s)), "batch8_smevrsurr_std": float(np.std(s))}


# --------------------------------------------------------------------------------------
# real coder bytes + SMEVR stream decomposition
# --------------------------------------------------------------------------------------
def coder_bytes(q: np.ndarray, levels: int) -> dict:
    """EXACT bytes from the shipped r7 coder, plus SMEVR's internal stream split.

    A SMEVR frame is HEADER + base_stream + delta_stream, and delta_stream is
    u32(len(occupancy)) + occupancy_stream + value_stream (see `_encode_smevr`). Splitting it
    attributes every shipped byte to the temporal mode base, the EVENT (occupancy) stream, or
    the VALUE (residual magnitude) stream.
    """
    from experiments.ddm_r7_token_coder import HEADER, encode_token_codes

    qc = np.ascontiguousarray(q)
    frame = encode_token_codes(qc, levels=levels, codec="smevr")
    f = HEADER.unpack_from(frame)
    base_len, delta_len = int(f[-3]), int(f[-2])
    body = frame[HEADER.size:]
    if len(body) != base_len + delta_len:
        raise RuntimeError(f"SMEVR frame length mismatch: {len(body)} != {base_len}+{delta_len}")
    occ_len = struct.unpack_from("<I", body[base_len:])[0]
    return {
        "smevr_bytes": len(frame),
        "brotli11_bytes": len(encode_token_codes(qc, levels=levels, codec="brotli11")),
        "smevr_header_bytes": HEADER.size, "smevr_base_bytes": base_len,
        "smevr_occupancy_bytes": int(occ_len),
        "smevr_value_bytes": int(delta_len - 4 - occ_len),
    }


def mechanism(q: np.ndarray, levels: int) -> dict:
    """Hard (non-differentiable) mechanism diagnostics: the ORACLE versions of (c)/(d)."""
    from experiments.ddm_r7_token_coder import factor_mode_delta

    _base, delta = factor_mode_delta(np.ascontiguousarray(q), levels)
    consec = (q[1:].astype(np.int16) - q[:-1].astype(np.int16)) % levels
    return {"mode_occupancy": float((delta != 0).mean()),
            "consec_occupancy": float((consec != 0).mean()),
            "n_distinct_symbols": int(np.unique(q).size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    from experiments.train_tr1_partition_renderer_mlx import quantize_tokens_np

    manifest = json.loads(args.manifest.read_text())
    seen: set[tuple[str, str]] = set()
    if args.out.exists():  # resumable-from-disk (P0): never redo a completed row
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen.add((r["run"], r["ckpt"]))
    print(f"{len(manifest)} fields in manifest; {len(seen)} already done")

    with args.out.open("a") as fh:
        for i, item in enumerate(manifest, 1):
            if (item["dir"], item["ckpt"]) in seen:
                continue
            d = args.root / item["dir"]
            cfg = json.loads((d / "tr1_config.json").read_text())
            t0 = time.time()
            field, basis, keep_bool = load_field(d / "checkpoints" / item["ckpt"], cfg)
            L = int(cfg["token_quant_levels"])
            q = quantize_tokens_np(field, L)
            row = {
                "run": item["dir"], "ckpt": item["ckpt"], "group": item["group"],
                "basis": basis, "shape": list(q.shape), "levels": L, "n_tokens": int(q.size),
                "variant": cfg["variant"], "init": cfg.get("token_init_mode"),
                "mask": (os.path.basename(cfg["token_cell_mask"])
                         if cfg.get("token_cell_mask") else None),
                "keep_frac": float(keep_bool.mean()),
                "cfg_w_rate": cfg.get("w_rate"), "cfg_rate_model": cfg.get("rate_model"),
                "margin_quant": cfg.get("token_quant_margin_coupling"),
                "delta_sparsity_weight": cfg.get("delta_sparsity_weight"),
            }
            row.update(surrogates(field, keep_bool, L, q))
            row.update(mechanism(q, L))
            row.update(coder_bytes(q, L))
            row.update(batch_dispersion(field, keep_bool, L))
            row["smevr_bits_per_token"] = row["smevr_bytes"] * 8.0 / q.size
            row["secs"] = round(time.time() - t0, 2)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"[{i:3d}/{len(manifest)}] {item['dir']:44s} {item['ckpt']:34s} "
                  f"smevr={row['smevr_bytes']:7d} ent={row['surr_entropy_bits']:.4f} "
                  f"consec={row['surr_smevr_surrogate_bits']:.4f} "
                  f"modeb={row['surr_modebase_bits']:.4f} ({row['secs']}s)", flush=True)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
