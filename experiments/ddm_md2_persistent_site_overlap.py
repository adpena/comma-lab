#!/usr/bin/env python
"""ddm_md2 — site-level overlap between two cells' PERSISTENT partitions, plus the birth step.

`experiments/ddm_md1_micro_to_macro.py` is used UNCHANGED to sweep and analyze each cell.  It
already persists, per (cell, forward, gt_lineage):

* ``site_classes_<cell>_<forward>_<lineage>.npz``  — ``site_class_u8`` [P,H,W] falling-rule class
* ``excursion_<cell>_<forward>_<lineage>.npz``     — ``trajectory_code_u8`` = at_zero + 2*at_peak
                                                     + 4*at_terminal
* ``ANALYSIS_<cell>_<lineage>.json``               — class rows, terminal edges, reachability

md1's own ``--mode compare`` answers a DIFFERENT question (BORN-site overlap between the sealed
cold and warm cells) and is hardcoded to those two cell names, so it cannot be pointed at a third
cell.  This module adds only what md2's charter asks for on top of md1's persisted payloads:

1. the **Jaccard site overlap of the PERSISTENT sets** across two cells, on the whole class and on
   its terminal-wrong subset, against the chance expectation for two independent draws of the same
   sizes (same sites => capacity-limited regardless of schedule; different sites =>
   optimizer-path-dependent);
2. the **Lane-touching share and GT=Lane enrichment**, computed with md1's own two formulas —
   Lane-touching = terminal edges of the class with `Lane` on either side; enrichment = the class's
   GT=Lane site fraction over Lane's HT GT area fraction;
3. the **birth**: the over-paint birth step per class, the excursion peak, and the first-wrong step
   of every NEW_PERSISTENT site, streamed one swept payload at a time.

$0.  CPU only.  Reads md1's payloads; writes only under ``--out-store``.  No score claim.
"""

from __future__ import annotations

import argparse
import json
import os
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

SCHEMA = "ddm_md2_persistent_site_overlap.v1"
AXIS = (
    "[macOS-CPU advisory; site sets reconstructed by ddm_md1_micro_to_macro from retained 16-step "
    "checkpoints; not contest authority]"
)


class MD2Error(RuntimeError):
    """ddm_md2 refuses rather than emitting an unlabelled or unfaithful number."""


def _load_site_classes(store: Path, cell: str, forward: str, lineage: str) -> tuple[np.ndarray, np.ndarray]:
    path = store / f"site_classes_{cell}_{forward}_{lineage}.npz"
    if not path.is_file():
        raise MD2Error(f"missing site-class payload: {path}")
    with np.load(path) as block:
        codes = np.asarray(block["site_class_u8"], dtype=np.uint8)
        pair_ids = np.asarray(block["pair_ids"], dtype=np.int64)
    return codes.reshape(-1), pair_ids


def _load_terminal_wrong(store: Path, cell: str, forward: str, lineage: str) -> tuple[np.ndarray, int]:
    path = store / f"excursion_{cell}_{forward}_{lineage}.npz"
    if not path.is_file():
        raise MD2Error(f"missing excursion payload: {path}")
    with np.load(path) as block:
        code = np.asarray(block["trajectory_code_u8"], dtype=np.uint8).reshape(-1)
        peak_step = int(block["peak_step"][0])
    return (code & 4) != 0, peak_step


def _overlap(a: np.ndarray, b: np.ndarray, *, sites: int) -> dict[str, Any]:
    """Jaccard plus the chance expectation for two independent draws of the same sizes."""

    a_n = int(a.sum())
    b_n = int(b.sum())
    inter = int(np.count_nonzero(a & b))
    union = a_n + b_n - inter
    chance = (a_n * b_n) / float(sites) if sites else 0.0
    return {
        "a_sites": a_n,
        "b_sites": b_n,
        "intersection_sites": inter,
        "union_sites": int(union),
        "jaccard": float(inter / union) if union else 0.0,
        "a_covered_by_b_fraction": float(inter / a_n) if a_n else 0.0,
        "b_covered_by_a_fraction": float(inter / b_n) if b_n else 0.0,
        "a_only_sites": a_n - inter,
        "b_only_sites": b_n - inter,
        "chance_intersection_sites_if_independent": float(chance),
        "intersection_over_chance": float(inter / chance) if chance > 0 else None,
    }


def _lane_rows(analysis: dict[str, Any], forward: str, class_name: str, gt_area_lane: float) -> dict[str, Any]:
    """md1's two Lane formulas, applied verbatim to whichever cell's ANALYSIS is passed in."""

    block = analysis["forwards"][forward]
    row = block["classes"][class_name]
    edges = block["terminal_edges"].get(class_name, {})
    edge_total = int(sum(edges.values()))
    lane_touching = int(sum(v for k, v in edges.items() if "Lane" in k))
    hist = [int(v) for v in row["gt_class_histogram"]]
    hist_total = int(sum(hist))
    gt_lane_fraction = float(hist[1] / hist_total) if hist_total else 0.0
    return {
        "terminal_wrong_sites": edge_total,
        "lane_touching_terminal_wrong_sites": lane_touching,
        "lane_touching_share_of_terminal_wrong": float(lane_touching / edge_total) if edge_total else 0.0,
        "class_sites": int(row["sites"]),
        "gt_class_histogram": hist,
        "gt_lane_site_fraction_of_class": gt_lane_fraction,
        "gt_lane_ht_area_fraction": float(gt_area_lane),
        "gt_lane_enrichment": float(gt_lane_fraction / gt_area_lane) if gt_area_lane > 0 else None,
    }


def _rows_by_forward(store: Path, cell: str) -> dict[str, dict[int, dict[str, Any]]]:
    path = store / f"sweep_rows_{cell}.jsonl"
    if not path.is_file():
        raise MD2Error(f"missing swept rows: {path}")
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row["forward"]), {})[int(row["step"])] = row
    return grouped


def _overpaint_birth(
    rows: dict[int, dict[str, Any]],
    gt_area: Sequence[float],
    threshold: float,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """First swept step whose HT predicted area reaches ``threshold`` x the class's HT GT area."""

    steps = sorted(rows)
    out: dict[str, Any] = {}
    for class_index, name in enumerate(class_names):
        birth: int | None = None
        for step in steps:
            area = rows[step]["pred_area_ht_by_class"][class_index]
            if gt_area[class_index] > 0 and area / gt_area[class_index] >= threshold:
                birth = int(step)
                break
        out[str(name)] = birth
    return out


def _first_wrong_step(
    payload_root: Path, forward: str, steps: Sequence[int], gt_flat: np.ndarray
) -> np.ndarray:
    """Streamed: the first swept step at which each site is wrong (-1 if never).  One payload at a time.

    At step 0 the EMA shadow is initialised FROM the model, so md1 writes a single shared payload
    under the ``shadow`` name; the live series falls back to it exactly as md1's analyze does.
    """

    first = np.full(gt_flat.shape[0], -1, dtype=np.int32)
    for step in steps:
        path = payload_root / f"{forward}_step_{step:06d}.npz"
        if not path.is_file() and step == 0:
            # step 0 only: the shared payload.  A gap at any OTHER step is a hole in the sweep and
            # must refuse rather than silently substitute the other forward's field.
            path = payload_root / f"shadow_step_{step:06d}.npz"
        if not path.is_file():
            raise MD2Error(f"missing swept payload: {path}")
        with np.load(path) as payload:
            argmax = np.asarray(payload["argmax_u8"], dtype=np.uint8).reshape(-1)
        wrong = argmax != gt_flat
        fresh = wrong & (first < 0)
        first[fresh] = int(step)
        del argmax, wrong, fresh
    return first


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-store", required=True, help="md2's store; the only directory written")
    parser.add_argument("--a-store", required=True, help="store holding cell A (the new cell)")
    parser.add_argument("--a-cell", required=True)
    parser.add_argument("--b-store", required=True, help="store holding cell B (the comparator)")
    parser.add_argument("--b-cell", required=True)
    parser.add_argument("--gt-lineage", choices=("dali", "pyav"), default="dali")
    parser.add_argument("--overpaint-threshold", type=float, default=1.05)
    parser.add_argument("--skip-first-wrong", action="store_true", help="skip the streamed birth-step pass")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()

    from experiments import ddm_ar1_aa_render_price as ar1
    from experiments import ddm_md1_micro_to_macro as md1
    from experiments import ddm_qbt1_qbflow_trainer as qbt

    out_store = Path(args.out_store)
    out_store.mkdir(parents=True, exist_ok=True)
    a_store = Path(args.a_store)
    b_store = Path(args.b_store)
    lineage = args.gt_lineage

    gt = ar1.load_ground_truth()
    pair_ids = list(qbt.SELECTION_IDS)
    index = np.asarray(pair_ids, dtype=np.int64)
    weights = md1.ht_weights_vector(pair_ids, qbt.SELECTION_IDS, qbt.SELECTION_WEIGHTS)
    gt_key = "dali_seg" if lineage == "dali" else "pyav_seg"
    gt_seg = np.asarray(gt[gt_key], dtype=np.uint8)[index]
    gt_flat = gt_seg.reshape(-1)
    gt_area = md1.gt_area_fractions(gt_seg, weights, qbt.N)
    sites = int(gt_flat.shape[0])

    a_analysis = json.loads(
        (a_store / f"ANALYSIS_{args.a_cell}_{lineage}.json").read_text(encoding="utf-8")
    )
    b_analysis = json.loads(
        (b_store / f"ANALYSIS_{args.b_cell}_{lineage}.json").read_text(encoding="utf-8")
    )

    out: dict[str, Any] = {
        "schema": SCHEMA,
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "metal_invocations": 0,
        "modal_invocations": 0,
        "contest_eval_invocations": 0,
        "gt_lineage": lineage,
        "a_cell": args.a_cell,
        "b_cell": args.b_cell,
        "sites": sites,
        "gt_area_ht_by_class": [float(v) for v in gt_area],
        "class_names": list(ar1.CLASS_NAMES),
        "overpaint_threshold": float(args.overpaint_threshold),
        "forwards": {},
        "forwards_skipped": {},
    }

    for forward in ("shadow", "live"):
        try:
            a_codes, a_pairs = _load_site_classes(a_store, args.a_cell, forward, lineage)
            b_codes, b_pairs = _load_site_classes(b_store, args.b_cell, forward, lineage)
        except MD2Error as missing:
            # Silence is the failure mode this records: a forward absent from either store must
            # appear in the artifact, never vanish into an empty table.
            out["forwards_skipped"][forward] = str(missing)
            continue
        if not np.array_equal(a_pairs, b_pairs) or not np.array_equal(a_pairs, index):
            raise MD2Error("the two cells do not share the sealed pair selection; sites are not comparable")
        a_terminal, a_peak = _load_terminal_wrong(a_store, args.a_cell, forward, lineage)
        b_terminal, b_peak = _load_terminal_wrong(b_store, args.b_cell, forward, lineage)

        block: dict[str, Any] = {"a_peak_step": a_peak, "b_peak_step": b_peak, "overlap": {}}
        for class_name in md1.SITE_CLASSES:
            code = md1.CLASS_CODE[class_name]
            block["overlap"][class_name] = _overlap(a_codes == code, b_codes == code, sites=sites)
        persistent = md1.CLASS_CODE[md1.CLASS_PERSISTENT]
        block["overlap_persistent_terminal_wrong"] = _overlap(
            (a_codes == persistent) & a_terminal,
            (b_codes == persistent) & b_terminal,
            sites=sites,
        )
        block["overlap_any_terminal_wrong"] = _overlap(a_terminal, b_terminal, sites=sites)
        always = md1.CLASS_CODE[md1.CLASS_ALWAYS_CORRECT]
        block["overlap_any_error_class"] = _overlap(a_codes != always, b_codes != always, sites=sites)
        block["lane_a"] = _lane_rows(a_analysis, forward, md1.CLASS_PERSISTENT, gt_area[1])
        block["lane_b"] = _lane_rows(b_analysis, forward, md1.CLASS_PERSISTENT, gt_area[1])
        out["forwards"][forward] = block

    # --- birth: over-paint crossing per class, and the first-wrong step of NEW_PERSISTENT sites ---
    a_rows = _rows_by_forward(a_store, args.a_cell)
    b_rows = _rows_by_forward(b_store, args.b_cell)
    birth: dict[str, Any] = {"overpaint_birth_step_by_class": {}, "d_seg_hat_trajectory": {}}
    for label, rows in (("a", a_rows), ("b", b_rows)):
        birth["overpaint_birth_step_by_class"][label] = {
            forward: _overpaint_birth(
                series, gt_area, float(args.overpaint_threshold), ar1.CLASS_NAMES
            )
            for forward, series in rows.items()
        }
        birth["d_seg_hat_trajectory"][label] = {
            forward: {
                "steps": sorted(series),
                f"d_seg_hat_{lineage}": [series[s][f"d_seg_hat_{lineage}"] for s in sorted(series)],
                "pose_mse_hat": [series[s]["pose_mse_hat"] for s in sorted(series)],
            }
            for forward, series in rows.items()
        }
    out["birth"] = birth

    if not args.skip_first_wrong:
        new_persistent = md1.CLASS_CODE[md1.CLASS_NEW_PERSISTENT]
        first_wrong: dict[str, Any] = {}
        for label, store, cell, rows in (
            ("a", a_store, args.a_cell, a_rows),
            ("b", b_store, args.b_cell, b_rows),
        ):
            payload_root = store / "payloads" / cell
            shadow_steps = set(rows.get("shadow", {}))
            per_forward: dict[str, Any] = {}
            for forward in ("shadow", "live"):
                if forward not in rows:
                    continue
                steps = sorted(rows[forward])
                if 0 not in steps and 0 in shadow_steps:
                    steps = [0, *steps]
                codes, _ = _load_site_classes(store, cell, forward, lineage)
                firsts = _first_wrong_step(payload_root, forward, steps, gt_flat)
                mask = codes == new_persistent
                selected = firsts[mask]
                selected = selected[selected >= 0]
                if selected.size:
                    values, counts = np.unique(selected, return_counts=True)
                    order = np.argsort(-counts)[:12]
                    per_forward[forward] = {
                        "new_persistent_sites": int(mask.sum()),
                        "first_wrong_step_min": int(selected.min()),
                        "first_wrong_step_median": float(np.median(selected)),
                        "first_wrong_step_max": int(selected.max()),
                        "born_at_or_before_step_16_fraction": float((selected <= 16).mean()),
                        "born_at_or_before_step_64_fraction": float((selected <= 64).mean()),
                        "top_first_wrong_steps": [
                            {"step": int(values[i]), "sites": int(counts[i])} for i in order
                        ],
                    }
                else:
                    per_forward[forward] = {"new_persistent_sites": int(mask.sum())}
                del codes, firsts, mask, selected
            first_wrong[label] = per_forward
        out["birth"]["new_persistent_first_wrong_step"] = first_wrong

    out["elapsed_s"] = float(time.monotonic() - started)
    out["peak_rss_gib"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3))
    out["host"] = {"platform": platform.platform()}
    path = out_store / f"OVERLAP_{args.a_cell}_vs_{args.b_cell}_{lineage}.json"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(out, indent=2, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    print(json.dumps({k: v for k, v in out.items() if k not in ("birth",)}, indent=2, sort_keys=True)[:6000])
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
