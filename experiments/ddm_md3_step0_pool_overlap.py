#!/usr/bin/env python
"""ddm_md3 — is the born vehicle's step-0 WRONG POOL a property of the START or of the DATA?

md2 measured that the PERSISTENT (optimizer-unreachable) partition is a subset of the step-0 wrong
pool, and that the pool is **bit-identical across every burn cell** --- because every cell declares
the same ``initial_state`` file.  This module asks the next question at its cheapest possible
scope: point the start somewhere else and see whether the pool moves.

Each candidate start was swept by ``experiments/ddm_md1_micro_to_macro.py --mode sweep
--max-step 0``, UNCHANGED, which retains ``payloads/<cell>/shadow_step_000000.npz`` holding the
exact argmax over the sealed n32 selection.  This module reads those payloads, forms each start's
wrong pool against md1's own ground truth (``ar1.load_ground_truth``, both lineages), and reports:

* pool size per start, and the **weighted** ``d_seg_hat`` at step 0 (md1's HT estimator);
* **Jaccard of each candidate's pool against the incumbent's**, next to the weight-space distance
  the start sits at --- the dose-response curve;
* **containment**: what fraction of the incumbent's PERSISTENT sites (md1's retained class map) is
  ALSO wrong at each candidate start.  Containment is the statistic that survives a size mismatch,
  and an untrained null anchor has a large pool by construction;
* the **null models**: chance Jaccard for two independent draws of the observed sizes from the
  frame, and the maximum Jaccard the two observed sizes allow.

Verdict words are the caller's; this module emits the numbers and refuses to guess.

$0.  CPU only.  Reads retained payloads; writes only under ``--out-store``.  No score claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ar1_aa_render_price as ar1
from experiments import ddm_md1_micro_to_macro as md1
from experiments import ddm_qbt1_qbflow_trainer as qbt

SCHEMA = "ddm_md3_step0_pool_overlap.v1"
AXIS = (
    "[macOS-CPU advisory; step-0 wrong pools reconstructed by ddm_md1_micro_to_macro from retained "
    "starting states; frozen CPU-torch SegNet; n32 sealed selection; not contest authority]"
)
CLASS_PERSISTENT_CODE = md1.CLASS_CODE[md1.CLASS_PERSISTENT]


class MD3OverlapError(RuntimeError):
    """ddm_md3 refuses rather than emitting an unlabelled or unfaithful overlap."""


# ---------------------------------------------------------------------------
def _argmax(store: Path, cell: str, step: int = 0, forward: str = "shadow") -> np.ndarray:
    path = store / "payloads" / cell / f"{forward}_step_{step:06d}.npz"
    if not path.is_file():
        raise MD3OverlapError(f"retained step-{step} payload absent: {path}")
    with np.load(path, allow_pickle=False) as payload:
        argmax = np.asarray(payload["argmax_u8"], dtype=np.uint8)
        pair_ids = np.asarray(payload["pair_ids"], dtype=np.int64)
    if list(pair_ids) != list(qbt.SELECTION_IDS):
        raise MD3OverlapError(f"{cell}: payload pair ids differ from the sealed n32 selection")
    return argmax


def _jaccard(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    na = int(np.count_nonzero(a))
    nb = int(np.count_nonzero(b))
    return {
        "a_sites": na,
        "b_sites": nb,
        "intersection": inter,
        "union": union,
        "jaccard": (inter / union) if union else float("nan"),
        "a_in_b_fraction": (inter / na) if na else float("nan"),
        "b_in_a_fraction": (inter / nb) if nb else float("nan"),
    }


def _chance_jaccard(na: int, nb: int, total: int) -> float:
    """Two independent uniform draws of sizes na, nb from `total` sites."""

    expected_inter = na * nb / total
    expected_union = na + nb - expected_inter
    return float(expected_inter / expected_union) if expected_union else float("nan")


def _max_jaccard(na: int, nb: int) -> float:
    return float(min(na, nb) / max(na, nb)) if max(na, nb) else float("nan")


def _weighted_dseg(wrong: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """md1's HT estimator: per-pair weight times per-pair wrong count, over the same denominator."""

    n_pairs, height, width = wrong.shape
    per_pair = wrong.reshape(n_pairs, -1).sum(axis=1).astype(np.int64)
    numerator = int((per_pair * weights.astype(np.int64)).sum())
    denominator = int(weights.astype(np.int64).sum() * height * width)
    return {
        "weighted_numerator": numerator,
        "denominator": denominator,
        "d_seg_hat": numerator / denominator,
    }


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    out_store = Path(args.out_store)
    probe_store = Path(args.probe_store)
    index = json.loads(Path(args.states_index).read_text(encoding="utf-8"))

    gt = ar1.load_ground_truth()
    pair_ids = np.asarray(list(qbt.SELECTION_IDS), dtype=np.int64)
    weights = md1.ht_weights_vector(list(qbt.SELECTION_IDS), qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)

    blocks: dict[str, Any] = {}
    for lineage in ("dali", "pyav"):
        gt_seg = np.asarray(gt["dali_seg" if lineage == "dali" else "pyav_seg"], dtype=np.uint8)[pair_ids]
        total_sites = int(gt_seg.size)

        incumbent_argmax = _argmax(Path(args.incumbent_store), args.incumbent_cell)
        incumbent_pool = incumbent_argmax != gt_seg

        # md1's retained PERSISTENT class map for the incumbent cell (the unreachable set).
        persistent_path = (
            Path(args.incumbent_store)
            / f"site_classes_{args.incumbent_cell}_shadow_{lineage}.npz"
        )
        with np.load(persistent_path, allow_pickle=False) as payload:
            codes = np.asarray(payload["site_class_u8"], dtype=np.uint8)
        persistent = (codes == CLASS_PERSISTENT_CODE).reshape(gt_seg.shape)
        if not bool(np.all(persistent <= incumbent_pool)):
            raise MD3OverlapError(
                f"{lineage}: the retained PERSISTENT set is not a subset of the reconstructed "
                "step-0 pool -- the payload and the class map disagree"
            )

        rows: list[dict[str, Any]] = []
        for candidate in index["candidates"]:
            cid = candidate["candidate_id"]
            cell = f"md3_start_{cid}"
            try:
                argmax = _argmax(probe_store, cell)
            except MD3OverlapError:
                if args.skip_missing:
                    continue
                raise
            pool = argmax != gt_seg
            overlap = _jaccard(pool.reshape(-1), incumbent_pool.reshape(-1))
            persistent_still_wrong = int(np.count_nonzero(persistent & pool))
            rows.append(
                {
                    "candidate_id": cid,
                    "born_gate_eligible": bool(candidate["born_gate_eligible"]),
                    "relative_l2_from_incumbent": float(
                        candidate["distance_from_incumbent"]["relative_l2"]
                    ),
                    "state_sha256": candidate["state"]["sha256"],
                    "step0": _weighted_dseg(pool, weights),
                    "pool_vs_incumbent": overlap,
                    "chance_jaccard": _chance_jaccard(
                        overlap["a_sites"], overlap["b_sites"], total_sites
                    ),
                    "max_attainable_jaccard": _max_jaccard(overlap["a_sites"], overlap["b_sites"]),
                    "incumbent_persistent_sites": int(np.count_nonzero(persistent)),
                    "incumbent_persistent_still_wrong_here": persistent_still_wrong,
                    "incumbent_persistent_containment": (
                        persistent_still_wrong / int(np.count_nonzero(persistent))
                        if np.count_nonzero(persistent)
                        else float("nan")
                    ),
                }
            )

        blocks[lineage] = {
            "total_sites": total_sites,
            "incumbent": {
                "cell": args.incumbent_cell,
                "pool_sites": int(np.count_nonzero(incumbent_pool)),
                "persistent_sites": int(np.count_nonzero(persistent)),
                "step0": _weighted_dseg(incumbent_pool, weights),
            },
            "candidates": rows,
        }

    receipt = {
        "schema": SCHEMA,
        "axis": AXIS,
        "states_index": str(Path(args.states_index).resolve()),
        "probe_store": str(probe_store.resolve()),
        "incumbent_store": str(Path(args.incumbent_store).resolve()),
        "n32_selection_ids": [int(x) for x in qbt.SELECTION_IDS],
        "n32_caveat": (
            "the n32 stratified Horvitz-Thompson selection estimates the n600 population; qn1's "
            "caveat that n32 -> n600 transfer is untested on this vehicle travels with every row"
        ),
        "scope_note": index.get("scope_note"),
        "blocks": blocks,
        "elapsed_s": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3),
        "host": platform.node(),
    }
    qbt.atomic_json(out_store / "STEP0_POOL_OVERLAP.json", receipt)
    return receipt


# ---------------------------------------------------------------------------
def render(receipt: dict[str, Any]) -> str:
    lines: list[str] = []
    for lineage, block in receipt["blocks"].items():
        inc = block["incumbent"]
        lines.append(f"### {lineage} — incumbent pool {inc['pool_sites']:,} sites, "
                     f"persistent {inc['persistent_sites']:,}, "
                     f"step-0 d_seg_hat {inc['step0']['d_seg_hat']:.10f}")
        lines.append("")
        lines.append("| start | born-gate | rel L2 | pool sites | step-0 d_seg_hat | J vs incumbent "
                     "| chance J | max J | persistent still wrong |")
        lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|")
        for row in block["candidates"]:
            lines.append(
                f"| {row['candidate_id']} | {'yes' if row['born_gate_eligible'] else 'NO'} "
                f"| {row['relative_l2_from_incumbent']:.4f} "
                f"| {row['pool_vs_incumbent']['a_sites']:,} "
                f"| {row['step0']['d_seg_hat']:.10f} "
                f"| {row['pool_vs_incumbent']['jaccard']:.4f} "
                f"| {row['chance_jaccard']:.4f} "
                f"| {row['max_attainable_jaccard']:.4f} "
                f"| {100 * row['incumbent_persistent_containment']:.2f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-store", required=True, help="the only directory written")
    parser.add_argument("--probe-store", required=True, help="store holding the --max-step 0 probe payloads")
    parser.add_argument("--states-index", required=True, help="ALTERNATE_INITIAL_STATES.json")
    parser.add_argument("--incumbent-store", required=True, help="md1's store (the cold control)")
    parser.add_argument("--incumbent-cell", default="cold_control_seed_20260902")
    parser.add_argument("--skip-missing", action="store_true", help="skip candidates with no probe payload")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(args)
    print(render(receipt))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
