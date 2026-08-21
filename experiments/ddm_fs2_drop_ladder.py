"""ddm_fs2 stage 1 - the two rate ladders on the LIVE rc2 body, per pair.

Consumes the retained per-pair `u = -log2(1 - p_max)` histograms and the retained
argmax / u-index fields from `ddm_fs2_rc2_token_replay.py`.

Two ladders, identical decoded field, different receivers:

* **PATH A (skip decode)** - the receiver stops decoding wherever `p_max >= tau`
  and takes the model argmax. Credit = the whole code length above tau. This is
  the ddm_rc4 ladder and it needs a receiver change.
* **PATH B (argmax substitution)** - the ENCODER writes the argmax into the token
  field above tau; the receiver is untouched. Credit = the bits currently spent at
  the positions that actually move (confident disagreements) minus the cost of
  coding the argmax there.

Both are DERIVED-first-order on the rate axis: substituting a token perturbs the
autoregressive context of later positions, so the realised credit is whatever
`ddm_jg2_tail_reencode` measures when it re-encodes the field for real. This tool
selects the threshold; the re-encoder prices it.

`mode=field` materialises the substituted token field at a chosen threshold and
retains it with sha256 + bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STORE = Path("/Volumes/APDataStore/pact/ddm_fs2")
RETAINED = STORE / "retained" / "token_rd"
FIELDS = STORE / "retained" / "fields"

N_PAIRS = 600
EVAL_H, EVAL_W = 384, 512
POSITIONS = N_PAIRS * EVAL_H * EVAL_W
U_STEP = 0.125

#: Contest score partials. Both are exact constants of the scoring function.
S_PER_BYTE = 25.0 / 37_545_489.0
S_PER_SEG_FLIP = 100.0 / POSITIONS

#: ddm_rc4 Stage 1b measured the seg amplification A (net SegNet argmax flips per
#: token flip) on the hv1 body at three thresholds: 0.78475 / 0.79844 / 0.80686 at
#: u = 5.0 / 7.0 / 8.5. It is FLAT in the threshold over a 20x range of (1 - p_max).
#: Used here ONLY to rank thresholds; the live body's A is measured downstream and
#: the candidate's seg leg is never priced from this number.
RC4_A_PRIOR = ((5.0, 0.78475), (7.0, 0.79844), (8.5, 0.80686))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(name: str) -> tuple[np.ndarray, dict]:
    path = RETAINED / f"{name}.npy"
    digest = sha256_bytes(path.read_bytes())
    return np.load(path, mmap_mode="r"), {"path": str(path), "sha256": digest}


def a_prior(u: float) -> float:
    """Linear interpolation / flat extrapolation of rc4's measured A."""
    us = [p[0] for p in RC4_A_PRIOR]
    values = [p[1] for p in RC4_A_PRIOR]
    if u <= us[0]:
        return values[0]
    if u >= us[-1]:
        return values[-1]
    return float(np.interp(u, us, values))


def build_ladders() -> dict:
    pair_bits, d_bits = _load("pair_hist_bits")
    pair_bits_dis, d_bits_dis = _load("pair_hist_bits_disagree")
    pair_n_dis, d_n_dis = _load("pair_hist_n_disagree")
    pair_n, d_n = _load("pair_hist_n")

    bits = np.asarray(pair_bits).sum(axis=0)
    bits_dis = np.asarray(pair_bits_dis).sum(axis=0)
    n_dis = np.asarray(pair_n_dis).sum(axis=0)
    n_all = np.asarray(pair_n).sum(axis=0)
    u_bins = bits.shape[0]
    us = np.arange(u_bins) * U_STEP
    tau = 1.0 - np.power(2.0, -us)
    # Cost of coding the argmax at a bin-j position, bounded by the bin's LOWER tau
    # (p_max >= tau there, so -log2(p_max) <= -log2(tau)). Conservative for Path B.
    # tau == 0 at the u == 0 bin: coding the argmax there is unbounded, so that bin
    # contributes no Path-B credit. Zero it BEFORE the multiply rather than after,
    # so no inf * 0 nan is ever formed.
    with np.errstate(divide="ignore"):
        cost_argmax = np.where(tau > 0.0, -np.log2(np.clip(tau, 1e-300, 1.0)), 0.0)
    finite_cost = np.where(tau > 0.0, cost_argmax, 0.0)

    def suffix(arr: np.ndarray) -> np.ndarray:
        return np.cumsum(arr[::-1])[::-1]

    suf_all = suffix(bits)
    suf_dis = suffix(bits_dis)
    suf_flips = suffix(n_dis)
    suf_positions = suffix(n_all)
    suf_sub_cost = suffix(n_dis * finite_cost)
    path_a_bytes = suf_all / 8.0
    path_b_bytes = (suf_dis - suf_sub_cost) / 8.0

    rows = []
    for k in range(u_bins):
        flips = int(suf_flips[k])
        if flips == 0:
            continue
        a = a_prior(float(us[k]))
        seg_flips = flips * a
        for label, saved in (("A", float(path_a_bytes[k])), ("B", float(path_b_bytes[k]))):
            rows.append(
                {
                    "path": label,
                    "u": float(us[k]),
                    "p_max_threshold": float(tau[k]),
                    "bytes_saved": saved,
                    "token_flips": flips,
                    "positions_above_tau": int(suf_positions[k]),
                    "A_prior": a,
                    "net_seg_flips_prior": seg_flips,
                    "dS_rate": -saved * S_PER_BYTE,
                    "dS_seg_prior": seg_flips * S_PER_SEG_FLIP,
                    "net_dS_pose_free_prior": -saved * S_PER_BYTE + seg_flips * S_PER_SEG_FLIP,
                    "bytes_per_token_flip": saved / flips,
                }
            )
    best = {}
    for label in ("A", "B"):
        subset = [r for r in rows if r["path"] == label]
        best[label] = min(subset, key=lambda r: r["net_dS_pose_free_prior"])
    return {
        "inputs": {
            "pair_hist_bits": d_bits,
            "pair_hist_bits_disagree": d_bits_dis,
            "pair_hist_n_disagree": d_n_dis,
            "pair_hist_n": d_n,
        },
        "exchange_rates": {
            "S_per_byte": S_PER_BYTE,
            "S_per_seg_flip": S_PER_SEG_FLIP,
            "breakeven_bytes_per_net_seg_flip": S_PER_SEG_FLIP / S_PER_BYTE,
        },
        "ladder": rows,
        "best_pose_free_prior": best,
    }


def per_pair_credit(u_threshold: float) -> dict:
    """Exact per-pair Path A / Path B rate credit at one threshold."""
    pair_bits, _ = _load("pair_hist_bits")
    pair_bits_dis, _ = _load("pair_hist_bits_disagree")
    pair_n_dis, _ = _load("pair_hist_n_disagree")
    k = round(u_threshold / U_STEP)
    us = np.arange(np.asarray(pair_bits).shape[1]) * U_STEP
    tau = 1.0 - np.power(2.0, -us)
    with np.errstate(divide="ignore"):
        cost = np.where(tau > 0.0, -np.log2(np.clip(tau, 1e-300, 1.0)), 0.0)
    cost = np.where(np.isfinite(cost), cost, 0.0)
    a_bits = np.asarray(pair_bits)[:, k:].sum(axis=1)
    d_bits = np.asarray(pair_bits_dis)[:, k:].sum(axis=1)
    flips = np.asarray(pair_n_dis)[:, k:].sum(axis=1)
    sub_cost = (np.asarray(pair_n_dis)[:, k:] * cost[k:][None, :]).sum(axis=1)
    return {
        "u_threshold": float(us[k]),
        "p_max_threshold": float(tau[k]),
        "path_a_bytes_per_pair": a_bits / 8.0,
        "path_b_bytes_per_pair": (d_bits - sub_cost) / 8.0,
        "token_flips_per_pair": flips.astype(np.int64),
    }


def run_ladder(args) -> int:
    out = build_ladders()
    out["arm"] = "ddm_fs2"
    out["stage"] = "1_drop_ladder_live_rc2"
    out["score_claim"] = False
    out["promotable"] = False
    out["rate_leg_label"] = (
        "DERIVED-first-order: substitution perturbs later contexts; the realised "
        "credit is whatever ddm_jg2_tail_reencode measures on the built field"
    )
    path = STORE / "FS2_DROP_LADDER.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for label in ("A", "B"):
        best = out["best_pose_free_prior"][label]
        print(
            f"PATH {label}: best u={best['u']:.3f} p_max>={best['p_max_threshold']:.7f} "
            f"bytes={best['bytes_saved']:.1f} flips={best['token_flips']} "
            f"net_dS(pose free, A prior)={best['net_dS_pose_free_prior']:.6e}"
        )
    print(f"wrote {path}")
    return 0


def run_field(args) -> int:
    FIELDS.mkdir(parents=True, exist_ok=True)
    argmax, d_arg = _load("argmax_field")
    u_index, d_u = _load("u_index_field")
    tokens = np.fromfile(args.tokens, dtype=np.uint8).reshape(N_PAIRS, EVAL_H * EVAL_W)
    k = round(args.u_threshold / U_STEP)
    mask = np.asarray(u_index) >= k
    new = np.where(mask, np.asarray(argmax), tokens).astype(np.uint8)
    changed = int((new != tokens).sum())
    changed_per_pair = (new != tokens).sum(axis=1).astype(np.int64)

    credit = per_pair_credit(args.u_threshold)
    tag = f"u{args.u_threshold:g}".replace(".", "p")
    field_path = FIELDS / f"tokens_substituted_{tag}.u8"
    field_path.write_bytes(new.tobytes())
    payload = field_path.read_bytes()

    # The same substitution in the shape `ddm_jg2_tail_reencode --edits` consumes,
    # so the MEASURED rate is priced by the proven re-encoder and never modelled.
    edits_path = FIELDS / f"edits_{tag}.npz"
    moved = np.flatnonzero(changed_per_pair > 0)
    np.savez_compressed(
        edits_path,
        **{str(int(p)): new[p].reshape(EVAL_H, EVAL_W) for p in moved},
    )
    edits_payload = edits_path.read_bytes()
    np.save(FIELDS / f"changed_per_pair_{tag}.npy", changed_per_pair)
    np.save(FIELDS / f"path_a_bytes_per_pair_{tag}.npy", credit["path_a_bytes_per_pair"])
    np.save(FIELDS / f"path_b_bytes_per_pair_{tag}.npy", credit["path_b_bytes_per_pair"])

    out = {
        "arm": "ddm_fs2",
        "stage": "1b_substituted_token_field",
        "u_threshold": credit["u_threshold"],
        "p_max_threshold": credit["p_max_threshold"],
        "source_tokens": {"path": str(args.tokens), "sha256": sha256_bytes(tokens.tobytes())},
        "argmax_field": d_arg,
        "u_index_field": d_u,
        "substituted_field": {
            "path": str(field_path),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        },
        "edits_npz": {
            "path": str(edits_path),
            "bytes": len(edits_payload),
            "sha256": sha256_bytes(edits_payload),
            "planes": int(moved.size),
        },
        "changed_tokens": changed,
        "changed_pairs": int((changed_per_pair > 0).sum()),
        "predicted_path_a_bytes": float(credit["path_a_bytes_per_pair"].sum()),
        "predicted_path_b_bytes": float(credit["path_b_bytes_per_pair"].sum()),
        "score_claim": False,
        "promotable": False,
    }
    path = STORE / f"FS2_FIELD_{tag}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("ladder").set_defaults(func=run_ladder)
    field = sub.add_parser("field")
    field.add_argument("--u-threshold", type=float, required=True)
    field.add_argument(
        "--tokens",
        type=Path,
        default=Path(
            "/Volumes/APDataStore/pact/ddm_rc2/composed_decode_r2/inflated/"
            ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
        ),
    )
    field.set_defaults(func=run_field)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
