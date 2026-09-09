#!/usr/bin/env python3
"""ddm_rp1 -- the mispredicted-token census: where the tail's payable bytes actually are.

WHY THIS IS THE DURABLE DELIVERABLE
-----------------------------------
``rank-mixer`` measured the shipped tail's exact split under the LIVE coder: of 117,964,800
tokens, **99.80075 % are the coder's own most probable class** and cost 292,155 bits
(36,519 B) in total -- 0.0024 bits each, a flag mass no field change can touch, because a
token cannot be cheaper than the prediction it already equals.  The whole payable object is
the other **0.19925 % -- 235,044 tokens carrying 666,069 bits = 83,259 B**, and the sub-0.12
rate corner (26,908.6 B at the move-37 pointer) is 32.3 % of exactly that.

``ddm_rp1`` measured what a single actuator gets out of it (0.146 of a first-order saving,
95.4 % of proposals refused by the argmax).  This module describes the OBJECT rather than
the actuator, so the next rate charter can aim at it without re-deriving it: the cost
distribution, the class transition matrix weighted by BITS, the vertical geometry, whether
the expensive tokens sit on GT edges or in region interiors, and whether they agree with GT
at all.

``[macOS-CPU advisory]``; ``score_claim=false``.  Every number is read from the live coder's
own probability rows via ``rank_mixer/candidates.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

EVAL_H, EVAL_W = jg1.EVAL_H, jg1.EVAL_W
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
#: comma10k vertical geometry, MEASURED 2026-06-27 and pinned in CLAUDE.md.
ROW_BANDS = (
    (0, 128, "upper (Undrivable/sky dominant)"),
    (128, 256, "mid (the seg residual band, rows 128-319)"),
    (256, 320, "lower-mid (residual band continues)"),
    (320, 384, "hood (MyCar static core)"),
)


def gt_edge_mask(gt_plane: np.ndarray) -> np.ndarray:
    """True where the GT label differs from any 4-neighbour -- the codim-1 boundary."""
    edge = np.zeros_like(gt_plane, dtype=bool)
    edge[:-1, :] |= gt_plane[:-1, :] != gt_plane[1:, :]
    edge[1:, :] |= gt_plane[:-1, :] != gt_plane[1:, :]
    edge[:, :-1] |= gt_plane[:, :-1] != gt_plane[:, 1:]
    edge[:, 1:] |= gt_plane[:, :-1] != gt_plane[:, 1:]
    return edge


def _bucket(values: np.ndarray, bits: np.ndarray, keys: list[Any]) -> list[dict[str, Any]]:
    out = []
    total_bits = float(bits.sum())
    for key in keys:
        mask = values == key
        n = int(mask.sum())
        if not n:
            continue
        b = float(bits[mask].sum())
        out.append(
            {
                "key": key,
                "tokens": n,
                "bits": b,
                "bytes": b / 8.0,
                "share_of_bits": b / total_bits if total_bits else 0.0,
                "mean_bits_per_token": b / n,
            }
        )
    return sorted(out, key=lambda r: -r["bits"])


def cmd_census(args: argparse.Namespace) -> int:
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    rank = Path(args.rank_dir)
    receipt = json.loads((rank / "RANK.json").read_text())
    if not receipt["identity_control"]["byte_identical"]:
        raise rp1.Rp1Error("rank dir's identity control did not pass")
    with np.load(rank / "candidates.npz", allow_pickle=False) as blob:
        cand = {k: blob[k] for k in blob.files}

    frame = cand["frame"].astype(np.int64)
    pos = cand["pos"].astype(np.int64)
    sym = cand["sym"].astype(np.int64)
    best = cand["best"].astype(np.int64)
    bits = cand["bits_sym"].astype(np.float64)
    saving = bits - cand["bits_best"].astype(np.float64)
    is_sj1 = cand["is_sj1_edit"].astype(bool)
    row = pos // EVAL_W

    census = receipt["census"]
    covered_bits = float(bits.sum())
    total_nonargmax_bits = census["bits_on_nonargmax_symbols"]

    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    base, live, _ = rp1.load_live_field()

    on_edge = np.zeros(pos.size, dtype=bool)
    gt_here = np.zeros(pos.size, dtype=np.int64)
    for f in np.unique(frame):
        sel = frame == f
        plane_gt = np.asarray(gt[int(f)])
        edge = gt_edge_mask(plane_gt).reshape(-1)
        on_edge[sel] = edge[pos[sel]]
        gt_here[sel] = plane_gt.reshape(-1)[pos[sel]]

    agrees_gt = sym == gt_here
    best_is_gt = best == gt_here

    band_key = np.zeros(pos.size, dtype=np.int64)
    for i, (lo, hi, _) in enumerate(ROW_BANDS):
        band_key[(row >= lo) & (row < hi)] = i

    transition = {}
    for a in range(5):
        for b in range(5):
            mask = (sym == a) & (best == b)
            n = int(mask.sum())
            if n:
                bb = float(bits[mask].sum())
                transition[f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}"] = {
                    "tokens": n,
                    "bits": bb,
                    "bytes": bb / 8.0,
                    "share_of_bits": bb / covered_bits,
                }

    verdict = {
        "schema": "ddm_rp1_mispredicted_census.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT coder measurement]",
        "score_claim": False,
        "coder": receipt.get("coder", "unknown"),
        "pointer": rp1.verify_pointer(expect_sha=args.expect_pointer_sha)
        if args.expect_pointer_sha
        else rp1.verify_pointer.__doc__,
        "denominator": {
            "tokens_total": census["total_tokens"],
            "tokens_that_are_coder_argmax": census["n_symbol_is_coder_argmax"],
            "tokens_mispredicted": census["total_tokens"]
            - census["n_symbol_is_coder_argmax"],
            "bits_on_mispredicted": total_nonargmax_bits,
            "bytes_on_mispredicted": total_nonargmax_bits / 8.0,
            "flag_mass_bits": census["bits_on_argmax_symbols"],
            "flag_mass_bytes": census["bits_on_argmax_symbols"] / 8.0,
            "census_covers_tokens": int(pos.size),
            "census_covers_bits": covered_bits,
            "census_coverage_of_mispredicted_bits": covered_bits / total_nonargmax_bits,
            "coverage_note": (
                "the dump retains positions whose best-alternative saving is >= "
                f"{receipt['candidates_dump']['min_saving_bits']} bits; the shortfall is "
                "mispredicted tokens the coder charges almost nothing for"
            ),
        },
        "by_stored_class": _bucket(sym, bits, list(range(5))),
        "by_coder_best_class": _bucket(best, bits, list(range(5))),
        "class_transitions_weighted_by_bits": dict(
            sorted(transition.items(), key=lambda kv: -kv[1]["bits"])
        ),
        "by_row_band": [
            {
                "band": ROW_BANDS[i][2],
                "rows": [ROW_BANDS[i][0], ROW_BANDS[i][1]],
                **{
                    k: v
                    for k, v in (_bucket(band_key, bits, [i])[0] if (band_key == i).any() else {}).items()
                    if k != "key"
                },
            }
            for i in range(len(ROW_BANDS))
        ],
        "gt_geometry": {
            "on_gt_edge": {
                "tokens": int(on_edge.sum()),
                "bits": float(bits[on_edge].sum()),
                "bytes": float(bits[on_edge].sum()) / 8.0,
                "share_of_bits": float(bits[on_edge].sum()) / covered_bits,
                "mean_bits_per_token": float(bits[on_edge].mean()) if on_edge.any() else 0.0,
            },
            "in_region_interior": {
                "tokens": int((~on_edge).sum()),
                "bits": float(bits[~on_edge].sum()),
                "bytes": float(bits[~on_edge].sum()) / 8.0,
                "share_of_bits": float(bits[~on_edge].sum()) / covered_bits,
                "mean_bits_per_token": float(bits[~on_edge].mean())
                if (~on_edge).any()
                else 0.0,
            },
        },
        "agreement_with_gt": {
            "stored_token_equals_gt": {
                "tokens": int(agrees_gt.sum()),
                "bits": float(bits[agrees_gt].sum()),
                "share_of_bits": float(bits[agrees_gt].sum()) / covered_bits,
            },
            "coder_best_equals_gt": {
                "tokens": int(best_is_gt.sum()),
                "bits": float(bits[best_is_gt].sum()),
                "share_of_bits": float(bits[best_is_gt].sum()) / covered_bits,
                "reading": (
                    "where the coder's favourite IS the GT label, the stored token is a "
                    "deliberate or incidental departure from GT and the coder is paying "
                    "for it; where it is NOT, the coder and GT disagree and the bits are "
                    "genuine content"
                ),
            },
            "is_an_sj1_predistortion_edit": {
                "tokens": int(is_sj1.sum()),
                "bits": float(bits[is_sj1].sum()),
                "bytes": float(bits[is_sj1].sum()) / 8.0,
                "share_of_bits": float(bits[is_sj1].sum()) / covered_bits,
            },
        },
        "cost_concentration": {
            f"top_{k}": {
                "tokens": k,
                "bits": float(np.sort(bits)[-k:].sum()),
                "bytes": float(np.sort(bits)[-k:].sum()) / 8.0,
                "share_of_mispredicted_bits": float(np.sort(bits)[-k:].sum())
                / total_nonargmax_bits,
            }
            for k in (1_000, 10_000, 50_000, 100_000)
            if k <= bits.size
        },
        "saving_headroom_by_class": _bucket(sym, saving, list(range(5))),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    print(json.dumps(verdict, indent=2, default=str)[:4000])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    census = sub.add_parser("census")
    census.add_argument("--rank-dir", required=True)
    census.add_argument("--out", required=True)
    census.add_argument("--expect-pointer-sha", default=None)
    census.set_defaults(func=cmd_census)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
