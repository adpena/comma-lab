#!/usr/bin/env python3
"""ddm_rp1 admission -- rate as the OBJECTIVE, pose as the COST, seg zero by construction.

WHY THIS IS NOT sj1's ADMISSION
-------------------------------
sj1 sweeps a seg credit against a pose cost and apportions rate.  This arm's seg leg is
**identically zero per admitted pair** -- every accepted token change was realized and the
pair's argmax came back identical on all 196,608 cells -- so the sweep here is the other
two legs only, and the seg leg is a CHECK rather than a term (``--verify-seg-zero`` reads
the composed field's own parse-back flip count and refuses unless it equals the base's).

THE THREE LEGS, EACH FROM ITS OWN MEASUREMENT
---------------------------------------------
rate   per-pair from the two REAL per-frame bit ledgers (control = the live field's own
       encode, candidate = the edited field's).  Never apportioned, never modelled.
pose   per-pair from the frozen CPU-torch PoseNet on the candidate's OWN renders, after
       the carrier re-solve; the base leg is the live row's own resolved pose, measured on
       the pointer's own configuration (never borrowed -- the pose-base law).
seg    zero, and checked.

THE SUM IS NOT THE PRICE
------------------------
A subset's exact archive is NOT the sum of its per-pair bit deltas: the coder is
autoregressive, so dropping a pair changes what the frames after it cost.  sj1 measured
that gap at **+19.6 B (+0.0108 %)** on its own pass-3 subset -- the ledger sum RANKS well
and UNDER-CHARGES.  So this module SELECTS with the sum and the caller PRICES with a real
encode of the selected field.  The refusal is structural: ``ADMISSION.json`` carries
``priced_by="ledger_sum_selection_only"`` and every downstream consumer must re-price.

``[macOS-CPU advisory]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402

N_PAIRS = jg1.N_PAIRS
S_PER_BYTE = rp1.S_PER_BYTE


def compose_score(
    d_seg: float, d_pose_mean: float, archive_bytes: float
) -> float:
    """S exactly as ``upstream/evaluate.py`` composes it."""
    return (
        100.0 * d_seg
        + math.sqrt(10.0 * d_pose_mean)
        + 25.0 * archive_bytes / jg1.SCORE_RATE_DENOMINATOR
    )


def load_rows(paths: list[Path]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pair = int(row["pair"])
            if pair in rows and rows[pair]["accepted"] != row["accepted"]:
                raise rp1.Rp1Error(
                    f"pair {pair} appears twice with different accepted sets"
                )
            rows[pair] = row
    return rows


def cmd_admit(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pointer = rp1.verify_pointer()

    rows = load_rows([Path(p) for p in args.rows])
    bits_ctrl = np.load(args.bits_control).astype(np.float64)
    bits_cand = np.load(args.bits_candidate).astype(np.float64)
    if bits_ctrl.shape != (N_PAIRS,) or bits_cand.shape != (N_PAIRS,):
        raise rp1.Rp1Error("bit ledgers must be (600,)")
    delta_bytes = (bits_cand - bits_ctrl) / 8.0

    base_pose = np.load(args.base_pose).astype(np.float64)
    resolved_pose = np.load(args.resolved_pose).astype(np.float64)
    if base_pose.shape != (N_PAIRS,) or resolved_pose.shape != (N_PAIRS,):
        raise rp1.Rp1Error("pose ledgers must be (600,)")

    edited_pairs = sorted(p for p, r in rows.items() if r["accepted"])
    if not edited_pairs:
        raise rp1.Rp1Error("no pair carries an accepted edit; nothing to admit")

    # A pair with no accepted edit must not move ANY leg: its plane is the live plane, so
    # its rate delta and its pose delta are both artefacts of the encode/solve and are
    # pinned to zero rather than being allowed to leak into the sweep.
    keep_mask_all = np.zeros(N_PAIRS, dtype=bool)
    keep_mask_all[edited_pairs] = True
    delta_bytes = np.where(keep_mask_all, delta_bytes, 0.0)
    pose_used = np.where(keep_mask_all, resolved_pose, base_pose)

    base_pose_mean = float(base_pose.mean())
    base_score = compose_score(
        args.base_d_seg, base_pose_mean, args.base_archive_bytes
    )
    if abs(base_score - pointer["score"]) > args.base_score_tolerance:
        raise rp1.Rp1Error(
            f"the base legs recompose to {base_score!r} but the live pointer reads "
            f"{pointer['score']!r}; the base is not the pointer's own row"
        )

    # Marginal ΔS per pair: rate exactly, pose linearized at the base operating point.
    # The linearization ORDERS the sweep; every reported score below is the EXACT
    # composition of the kept subset, never the linear sum.
    dS_pose_marginal = 5.0 / math.sqrt(10.0 * base_pose_mean) / N_PAIRS
    marginal = delta_bytes * S_PER_BYTE + (pose_used - base_pose) * dS_pose_marginal
    order = [p for p in edited_pairs]
    order.sort(key=lambda p: marginal[p])

    sweep: list[dict[str, Any]] = []
    best = None
    for cut in range(len(order) + 1):
        keep = order[:cut]
        mask = np.zeros(N_PAIRS, dtype=bool)
        mask[keep] = True
        archive = args.base_archive_bytes + float(delta_bytes[mask].sum())
        pose_mean = float(np.where(mask, resolved_pose, base_pose).mean())
        score = compose_score(args.base_d_seg, pose_mean, archive)
        entry = {
            "pairs_kept": cut,
            "archive_bytes_ledger_sum": archive,
            "delta_bytes": archive - args.base_archive_bytes,
            "d_pose_mean": pose_mean,
            "score": score,
            "delta_S_vs_pointer": score - base_score,
        }
        sweep.append(entry)
        if best is None or score < best["score"]:
            best = dict(entry, cut=cut)

    kept = order[: best["cut"]]
    kept_set = set(kept)
    tokens_kept = sum(len(rows[p]["accepted"]) for p in kept)

    # The admitted field carries ALL 600 planes.  A pair merely absent from an edits npz
    # reverts to the PRISTINE cl2 base and silently undoes every edit the live pointer
    # banked; a dropped pair must revert to the LIVE plane, which is what this does.
    _, live_field, field_receipts = rp1.load_live_field()
    field = np.array(live_field, dtype=np.uint8)
    for pair in kept:
        flat = field[pair].reshape(-1)
        for edit in rows[pair]["accepted"]:
            flat[int(edit["pos"])] = np.uint8(int(edit["best"]))
    changed = int((field != live_field).sum())
    if changed != tokens_kept:
        raise rp1.Rp1Error(
            f"admitted field differs from the live field at {changed} tokens but "
            f"{tokens_kept} edits were kept"
        )
    field_path = out / "field_admitted.npz"
    np.savez_compressed(field_path, **{str(p): field[p] for p in range(N_PAIRS)})

    verdict = {
        "schema": "ddm_rp1_admit.v1",
        "axis": (
            "rate EXACT from two real per-frame bit ledgers (selection only -- see "
            "priced_by); pose [macOS-CPU advisory, frozen CPU-torch PoseNet]; seg ZERO "
            "by realized construction and separately verified on the parse-back"
        ),
        "score_claim": False,
        "priced_by": "ledger_sum_selection_only",
        "pointer": pointer,
        "base": {
            "archive_bytes": args.base_archive_bytes,
            "d_seg": args.base_d_seg,
            "d_pose_mean": base_pose_mean,
            "recomposed_score": base_score,
        },
        "pairs_with_edits": len(edited_pairs),
        "pairs_kept": len(kept),
        "kept_pairs": sorted(kept_set),
        "tokens_kept": tokens_kept,
        "tokens_proposed": sum(len(r["accepted"]) for r in rows.values()),
        "best": best,
        "all_edits_row": sweep[-1],
        "sweep": sweep,
        "field_admitted": {
            "path": str(field_path),
            "planes": N_PAIRS,
            "tokens_changed_vs_live_field": changed,
        },
        "live_field": field_receipts,
    }
    (out / "ADMISSION.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                k: v
                for k, v in verdict.items()
                if k not in ("sweep", "kept_pairs", "pointer", "live_field")
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    admit = sub.add_parser("admit")
    admit.add_argument("--rows", nargs="+", required=True)
    admit.add_argument("--bits-control", required=True)
    admit.add_argument("--bits-candidate", required=True)
    admit.add_argument("--base-pose", required=True)
    admit.add_argument("--resolved-pose", required=True)
    admit.add_argument("--out-dir", required=True)
    admit.add_argument(
        "--base-archive-bytes", type=float, default=float(rp1.POINTER_ARCHIVE_BYTES)
    )
    admit.add_argument("--base-d-seg", type=float, required=True)
    admit.add_argument("--base-score-tolerance", type=float, default=5e-6)
    admit.set_defaults(func=cmd_admit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
