#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""OD4 scorer-free weak Stage-1 packet builder.

This prices a sparse mask-domain receiver packet for OD2's n32 Stage-1 rows.
It consumes OD2's real row receipt plus cached argmax arrays, builds counted
weak constraints, proves parse-back/replay in mask space, and races real
coders.  It does not run SegNet, PoseNet, upstream/evaluate.py, or an inflate.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for path in (REPO / "src", REPO / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ddm_bd1_class_field_receiver as bd1  # noqa: E402
from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks  # noqa: E402
from tac.optimization import ddm_od4_weak_stage1_packet as od4  # noqa: E402

DEFAULT_OD2_DIR: Final = REPO / ".omx/research/ddm_od2_20260805"
DEFAULT_OD2_JSON: Final = DEFAULT_OD2_DIR / "od2_js1_n32_cprime_k4.json"
DEFAULT_PAIR_SELECTION: Final = DEFAULT_OD2_DIR / "PAIR_SELECTION.json"
DEFAULT_ARGMAX_CACHE: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_od4_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_od4_20260805")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return data


def _best_coder(rows: tuple[od4.CoderRow, ...]) -> od4.CoderRow:
    candidates = [row for row in rows if row.parseback_exact and row.bytes > 0]
    if not candidates:
        raise SystemExit("no exact coder row survived")
    return min(candidates, key=lambda row: row.bytes)


def _store_best_packet(
    *,
    packet: bytes,
    coder: od4.CoderRow,
    fraction_tag: str,
    ssd_dir: Path,
) -> dict[str, Any]:
    packet_dir = ssd_dir / "packets"
    packet_dir.mkdir(parents=True, exist_ok=True)
    path = packet_dir / f"od4_stage1_sparse_{fraction_tag}.raw_packet"
    path.write_bytes(packet)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "selected_coder": coder.codec,
        "selected_coder_bytes": coder.bytes,
        "selected_coder_sha256": coder.sha256,
    }


def _derive_pair_record(
    *,
    pair: int,
    od2_row: dict[str, Any],
    current: np.ndarray,
    gt: np.ndarray,
    block: int,
    rmax: int,
    fraction: float,
) -> tuple[od4.SparsePairCorrections, dict[str, Any]]:
    before = int((current != gt).sum())
    if before != int(od2_row["flips_before"]):
        raise SystemExit(f"pair {pair}: cached current flips {before} != OD2 {od2_row['flips_before']}")
    offsets = solve_blocks(current, gt, block, rmax)
    target = translate_blocks(current, offsets.reshape(-1, 2), block)
    n_described = before - int((target != gt).sum())
    if n_described != int(od2_row["n_described"]):
        raise SystemExit(f"pair {pair}: recomputed n_described {n_described} != OD2 {od2_row['n_described']}")
    od2_after = int(od2_row["stage1"]["flips_after"])
    desired = before - od2_after
    record = od4.select_sparse_corrections(
        pair=pair,
        current_argmax=current,
        gt_argmax=gt,
        target_argmax=target,
        desired_fix_count=desired,
        fraction=fraction,
    )
    useful_count = int(((current != gt) & (target == gt) & (target != current)).sum())
    return record, {
        "pair": pair,
        "flips_before": before,
        "od2_flips_after": od2_after,
        "od2_fix_count": desired,
        "n_described": n_described,
        "candidate_useful_cells": useful_count,
        "selected_cells": record.count,
        "selected_fraction": fraction,
        "offsets_sha256": od4.sha256_bytes(np.ascontiguousarray(offsets.astype(np.int8)).tobytes()),
    }


def _rung_receipt(
    *,
    fraction: float,
    od2_rows_by_pair: dict[int, dict[str, Any]],
    pairs: list[int],
    current_argmax: np.ndarray,
    gt_argmax: np.ndarray,
    block: int,
    rmax: int,
    ssd_dir: Path,
) -> dict[str, Any]:
    records: list[od4.SparsePairCorrections] = []
    build_rows: list[dict[str, Any]] = []
    for pair in pairs:
        record, build_row = _derive_pair_record(
            pair=pair,
            od2_row=od2_rows_by_pair[pair],
            current=np.asarray(current_argmax[pair], dtype=np.uint8),
            gt=np.asarray(gt_argmax[pair], dtype=np.uint8),
            block=block,
            rmax=rmax,
            fraction=fraction,
        )
        records.append(record)
        build_rows.append(build_row)

    packet = od4.serialize_sparse_packet(records)
    parsed = od4.parse_sparse_packet(packet)
    if od4.serialize_sparse_packet(parsed.pair_records) != packet:
        raise SystemExit("OD4 packet parse-back serialization changed bytes")
    fidelity = od4.fidelity_for_packet(
        current_argmax=current_argmax,
        gt_argmax=gt_argmax,
        packet=parsed,
        od2_rows_by_pair=od2_rows_by_pair,
    )
    coder_rows = od4.race_packet_coders(
        packet,
        smevr_encode=bd1.smevr_records,
        smevr_decode=bd1.unsmevr_records,
    )
    best = _best_coder(coder_rows)
    fraction_tag = f"f{int(round(fraction * 1000)):04d}"
    artifact = _store_best_packet(packet=packet, coder=best, fraction_tag=fraction_tag, ssd_dir=ssd_dir)
    totals = fidelity["totals"]
    return {
        "fraction": fraction,
        "fraction_tag": fraction_tag,
        "packet": {
            "schema": od4.PACKET_SCHEMA,
            "raw_packet_bytes": len(packet),
            "raw_packet_sha256": od4.sha256_bytes(packet),
            "pair_records": len(records),
            "correction_count": parsed.correction_count,
            "artifact": artifact,
            "coder_race": [row.as_json() for row in coder_rows],
        },
        "build_rows": build_rows,
        "fidelity": fidelity,
        "projection_seg_only": od4.projection_rows(
            n32_packet_bytes=best.bytes,
            n_pairs=len(pairs),
            retained_fix_count=int(totals["retained_fix_count"]),
            include_od2_pose_credit=False,
        ),
        "projection_with_od2_stage2_pose_credit": od4.projection_rows(
            n32_packet_bytes=best.bytes,
            n_pairs=len(pairs),
            retained_fix_count=int(totals["retained_fix_count"]),
            include_od2_pose_credit=True,
        ),
    }


def _price_table_markdown(receipt: dict[str, Any]) -> str:
    rows = receipt["rungs"]
    lines = [
        "| retained target | corrections | exact n32 bytes | projected n600 bytes | retained eta | S seg-only | S w/ OD2 pose credit | best coder |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        totals = row["fidelity"]["totals"]
        proj0 = row["projection_seg_only"]
        proj1 = row["projection_with_od2_stage2_pose_credit"]
        packet = row["packet"]
        best = min(
            (item for item in packet["coder_race"] if item["parseback_exact"] and item["bytes"] > 0),
            key=lambda item: item["bytes"],
        )
        lines.append(
            "| "
            f"{row['fraction']:.2f} | "
            f"{packet['correction_count']} | "
            f"{best['bytes']} | "
            f"{proj0['packet_bytes_n600_linear_projection']} | "
            f"{totals['eta_receiver']:.6f} | "
            f"{proj0['projected_s']:.9f} | "
            f"{proj1['projected_s']:.9f} | "
            f"{best['codec']} |"
        )
    return "\n".join(lines)


def _write_gate_script(path: Path) -> None:
    content = """#!/usr/bin/env bash
set -euo pipefail

# OD4 queued scorer gate. Fill SUB_DIR only after a receiver-closed staged
# submission exists. od3 owns the scorer slot at OD4 build time, so this script
# is a fire-order artifact, not an active launch.
SUB_DIR="${SUB_DIR:?set SUB_DIR to the receiver-closed staged submission directory}"
OUT="${OUT:-.omx/research/ddm_od4_20260805/od4_receiver_gate_fz2_receipt.json}"

.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \\
  --sub-dir "${SUB_DIR}" \\
  --out "${OUT}" \\
  --inflate-out "${SUB_DIR}/inflated" \\
  --device cpu \\
  --batch-size 16 \\
  --num-threads 6
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_markdown(path: Path, receipt: dict[str, Any]) -> None:
    best = receipt["best_rung_by_projected_s_with_od2_pose_credit"]
    best_projection = best["projection_with_od2_stage2_pose_credit"]
    falsifier_fires = not bool(best_projection["beats_current_own_line"])
    verdict = (
        "fires"
        if falsifier_fires
        else "does not fire"
    )
    verdict_detail = (
        f"above the live own line by `{best_projection['projected_s'] - od4.CURRENT_OWN_S:.9f}`"
        if falsifier_fires
        else "below the live own line"
    )
    md = f"""# OD4 weak Stage-1 packet receipt - 2026-08-05

Status: `SCORER_FREE_MASK_DOMAIN_PACKET_PRICED / NO FRONTIER MOVE`.

Axis: `[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`, `scorer_forwards_run=0`.

## Answer First

OD4 built a receiver-parsed sparse weak packet for OD2's same n32 rows and priced it with a real coder race. The best measured rung is `{best['fraction']:.2f}` of OD2's retained Stage-1 mask fixes: `{best['packet']['correction_count']}` sparse target constraints, `{best['selected_coder_bytes']}` exact n32 counted bytes, projected `{best['projection_with_od2_stage2_pose_credit']['packet_bytes_n600_linear_projection']}` B at n600 by linear per-pair scaling.

The gc17 rank-1 falsifier **{verdict}** for this sparse packet formulation: under the OD2 same-row Stage-2 pose-credit projection, the best rung projects to `S = {best_projection['projected_s']:.9f}`, {verdict_detail}. This is not bankable: it is n32, cap-bound, mask-domain, and receiver-open to RGB/inflate/scorer. The frozen-scorer slot is occupied by od3, so the final n>=32 scorer gate is queued, not run.

## Price Table

{_price_table_markdown(receipt)}

## RECALL EVIDENCE

| query / source | beyond-charter finding | plan impact |
|---|---|---|
| `MEMORY.md: od4, gc17, #899/#904, margin_targets` | No OD4 prior memory hit; #899/#904 history is separate apparatus. | Did not reuse stale apparatus work; kept OD4 focused on Stage-1 representation pricing. |
| `_common_contract.md`, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md` | od3 owns the scorer slot; current own line is `S=0.7539807296911207 @ 357,836 B`; protected files and staged index are off limits. | Built scorer-free packet only, avoided protected paths, queued the scorer gate. |
| `GC17_CONVOCATION_RECEIPT.md`, OD2/ST2/BN1 receipts, operator addenda 4-6 | Stage-1 bytes are the named crux; ST2 is a prior, not a receiver; weakness must still minimize counted bytes. | Implemented counted sparse constraints with parse-back instead of shipping dense solved fields or hidden scorer tables. |
| content search: `Weak Stage-1`, `receiver-close`, `worldsheet`, `Road/Lane`, `description-side targets` over `.omx/research`, `.omx/state`, docs, and `src/tac` | PE1/SE3/BD1 already supply exact coder-race patterns and Road/Lane/per-edge description context; no finished OD4 packet existed. | Reused BD1-style Brotli/LZMA1/SMEVR round-trip discipline; did not claim edge/worldsheet survival. |
| canonical equations registry search for `receiver`, `worldsheet`, `quotient`, `trajectory` | Existing laws emphasize receiver closure, score quotient, trajectory cap-not-convergence, and exact counted bytes. | Kept cap-bound OD2 as a floor and labeled n600 byte figures as linear projections from exact n32 bytes. |

## SHA Table

| artifact | bytes | sha256 |
|---|---:|---|
| `{receipt['source_files']['od2_json']['path']}` | {receipt['source_files']['od2_json']['bytes']} | `{receipt['source_files']['od2_json']['sha256']}` |
| `{receipt['source_files']['pair_selection']['path']}` | {receipt['source_files']['pair_selection']['bytes']} | `{receipt['source_files']['pair_selection']['sha256']}` |
| `{receipt['receipt_json_path']}` | {Path(receipt['receipt_json_path']).stat().st_size if Path(receipt['receipt_json_path']).exists() else 0} | `{_sha256_file(Path(receipt['receipt_json_path'])) if Path(receipt['receipt_json_path']).exists() else 'pending'}` |

## NEXT_IF_RESUMED

1. Re-price this same packet format against od3 terminal fields, preserving the OD4 schema and changing only the values.
2. Build the actual receiver-closed RGB/inflate candidate only after terminal fields exist or MAIN explicitly accepts the cap-bound OD2 floor as a bounded prototype.
3. When the scorer slot is free, run `.omx/research/ddm_od4_20260805/OD4_SCORER_GATE_FIRE_ORDER.sh` with `SUB_DIR` bound to the exact receiver-closed staged submission.
4. Promote to n600 only if the n>=32 receiver-closed scorer gate beats the live own line after recomputing S from components.

## Boundaries

- No `upstream/evaluate.py`, SegNet, PoseNet, full n600, contest-CPU, or contest-CUDA run.
- This is a sparse mask-domain receiver packet, not a dense RGB field and not a legal archive row.
- n32 packet bytes are exact measured coder bytes; n600 packet bytes are linear projections until terminal n600 values exist.
- OD2 pose credit is inherited only for the same-row projection and was not remeasured by OD4.
- Stage 1 remains cap-bound; od3 terminality is still first.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
"""
    path.write_text(md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--od2-json", type=Path, default=DEFAULT_OD2_JSON)
    ap.add_argument("--pair-selection", type=Path, default=DEFAULT_PAIR_SELECTION)
    ap.add_argument("--argmax-cache", type=Path, default=DEFAULT_ARGMAX_CACHE)
    ap.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    ap.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    args = ap.parse_args()

    fractions = [float(item) for item in args.fractions.split(",") if item.strip()]
    if not fractions:
        raise SystemExit("--fractions produced no rungs")
    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)

    od2_json = _load_json(args.od2_json)
    pair_selection = _load_json(args.pair_selection)
    rows = od2_json.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("OD2 JSON does not contain rows")
    od2_rows_by_pair = {int(row["pair"]): row for row in rows}
    pairs = [int(pair) for pair in pair_selection["pairs"]]
    missing = [pair for pair in pairs if pair not in od2_rows_by_pair]
    if missing:
        raise SystemExit(f"OD2 JSON missing selected pairs: {missing}")

    current_argmax = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gt_argmax = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")
    rungs = [
        _rung_receipt(
            fraction=fraction,
            od2_rows_by_pair=od2_rows_by_pair,
            pairs=pairs,
            current_argmax=current_argmax,
            gt_argmax=gt_argmax,
            block=args.block,
            rmax=args.rmax,
            ssd_dir=args.ssd_dir,
        )
        for fraction in fractions
    ]

    best = min(
        rungs,
        key=lambda row: row["projection_with_od2_stage2_pose_credit"]["projected_s"],
    )
    best_coder = min(
        (item for item in best["packet"]["coder_race"] if item["parseback_exact"] and item["bytes"] > 0),
        key=lambda item: item["bytes"],
    )
    receipt_json_path = args.research_dir / "ddm_od4_weak_packet_receipt.json"
    gate_script = args.research_dir / "OD4_SCORER_GATE_FIRE_ORDER.sh"
    receipt: dict[str, Any] = {
        "schema": od4.RECEIPT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "axis": "[macOS-CPU cache-derived advisory / scorer-free mask-domain replay]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "scorer_forwards_run": 0,
        "current_own_vehicle_frontier": {
            "S": od4.CURRENT_OWN_S,
            "archive_bytes": od4.CURRENT_OWN_BYTES,
            "axis": od4.CURRENT_OWN_AXIS,
        },
        "denominators": {
            "n_pairs": len(pairs),
            "population_pairs": od4.N_PAIRS,
            "selection_mode": pair_selection["selection"]["pair_selection"],
            "selection_seed": pair_selection["seed"],
            "height": od4.SEG_H,
            "width": od4.SEG_W,
            "rate_denominator_bytes": od4.RATE_DENOMINATOR_BYTES,
        },
        "source_files": {
            "od2_json": {
                "path": str(args.od2_json),
                "bytes": args.od2_json.stat().st_size,
                "sha256": _sha256_file(args.od2_json),
            },
            "pair_selection": {
                "path": str(args.pair_selection),
                "bytes": args.pair_selection.stat().st_size,
                "sha256": _sha256_file(args.pair_selection),
            },
            "argmax_cache": {
                "path": str(args.argmax_cache),
                "cx1_sha256": _sha256_file(args.argmax_cache / "cx1_argmax_n600.npy"),
                "gt_sha256": _sha256_file(args.argmax_cache / "gt_argmax_n600.npy"),
            },
        },
        "packet_scope": (
            "sparse mask-domain constraints selected from OD2 same-row Stage-1 target fixes; "
            "not dense RGB, not frozen-scorer runtime, not a legal archive row"
        ),
        "rungs": rungs,
        "best_rung_by_projected_s_with_od2_pose_credit": {
            **best,
            "selected_coder": best_coder["codec"],
            "selected_coder_bytes": best_coder["bytes"],
        },
        "queued_scorer_gate_script": str(gate_script),
        "boundaries": [
            "No upstream/evaluate.py run",
            "No SegNet/PoseNet scorer job",
            "No full n600 dispatch",
            "No receiver-closed RGB/inflate archive",
            "n600 packet bytes are linear projections from exact n32 packet bytes",
        ],
        "frontier_line": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.",
        "receipt_json_path": str(receipt_json_path),
    }
    receipt_json_path.write_text(json.dumps(_jsonable(receipt), indent=2, sort_keys=True), encoding="utf-8")
    _write_gate_script(gate_script)
    _write_markdown(args.research_dir / "OD4_WEAK_PACKET_RECEIPT.md", receipt)

    print(_price_table_markdown(receipt))
    print(f"receipt_json={receipt_json_path}")
    print(f"receipt_md={args.research_dir / 'OD4_WEAK_PACKET_RECEIPT.md'}")
    print(f"gate_script={gate_script}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
