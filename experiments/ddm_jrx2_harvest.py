#!/usr/bin/env python3
"""Verify and summarize completed jrx2 bytes; never score or launch a candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "experiments"), str(REPO / "src"), str(REPO)]
import ddm_jrx2_control as c
import numpy as np

RATE = 25 / 37_545_489


def distribution(values: list[float | int | None]) -> dict:
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {
        "k": len(finite),
        "n": 24,
        "median": float(np.median(finite)) if finite else None,
        "iqr": np.quantile(finite, [0.25, 0.75]).tolist() if finite else None,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def prices() -> dict:
    """Whole archives charge the chosen proposal; sheet ledgers only ranked it."""
    denominator = c.load(c.STORE / "DISCOVERY_DENOMINATOR.json")
    winners = c.load(c.STORE / "prices/WINNERS.json")
    if winners["pairs"] != denominator["finite"]:
        raise ValueError("winner population differs from fixed sample's finite subset")
    base = c.load(c.price.TAIL_CONTROL)
    mean_pose = float(np.load(c.control.PD4 / "base/pose_base_move52.npy").mean())
    rows = []
    for selection in winners["jobs"]:
        root = c.STORE / "prices" / selection["name"]
        job, done = c.load(root / "JOB.json"), c.load(root / "DONE.json")
        if job["rows"] != [selection["row"]] or done["rows"] != job["rows"]:
            raise ValueError("individual price is not the selected proposal")
        for fact in [job["field_npz"], done["encode"], done["raw_expansion_certificate"], *done["archives"]]:
            c.price.checked_fact(fact)
        encoded = c.load(Path(done["encode"]["path"]))
        for fact in encoded["outputs"].values():
            c.price.checked_fact(fact)
        if not encoded["output_lossless"] or encoded["decoded_field_sha256"] != job["field_raw"]["sha256"]:
            raise ValueError("known-symbol receiver field reproduction failed")
        if done["archives"][0]["sha256"] != done["archives"][1]["sha256"]:
            raise ValueError("individual price twins disagree")
        row = selection["row"]
        pair = row["pair"]
        tokens = len(c.pd4.row_edits(row))
        if not tokens:
            raise ValueError("no-op cannot have a changed-token price")
        delta_bytes = done["archives"][0]["bytes"] - 179332
        if delta_bytes != done["delta_archive_bytes"]:
            raise ValueError("archive byte charge differs")
        credit = float(row["benefit_S"])
        finite_pose = mean_pose + row["credit_d_pose"] / 600
        if finite_pose < 0:
            raise ValueError("negative composed pose MSE")
        finite_credit = (
            math.sqrt(10 * mean_pose) - math.sqrt(10 * finite_pose)
            - row["d_cells"] * c.pd4.seg_s_per_cell()
        )
        rows.append({
            "pair": pair, "tokens_changed": tokens, "delta_archive_bytes": delta_bytes,
            "whole_archive_bits_per_changed_token": delta_bytes * 8 / tokens,
            "individual_frame_ledger_delta_bits": encoded["per_frame_bits"][pair] - base["per_frame_bits"][pair],
            "ranking_sheet_delta_bits": row["ranking_bits"],
            "matched_pd4_first_order_credit_S": credit,
            "nonlinear_isolated_credit_S": finite_credit,
            "signed_credit_S_per_byte": credit / delta_bytes if delta_bytes else None,
            "positive_cost_credit_S_per_byte": credit / delta_bytes if delta_bytes > 0 else None,
            "diagnostic_net_credit_S": finite_credit - RATE * delta_bytes,
            "d_cells": row["d_cells"], "credit_d_pose": row["credit_d_pose"],
            "done": c.fact(root / "DONE.json"), "archives": done["archives"],
        })
    result = {
        "rows": rows, "null_pairs": denominator["null_pairs"], "k": len(rows), "n": 24,
        "axis": c.AXIS, "score_claim": False,
        "boundary": "diagnostic resolved credit; priced tail archives hold shipped carrier, so not a byte-closed score",
        "field_check": "known-symbol receiver loop reproduces input field; packed sections parse back; no separate full entropy decode",
        "twins": "two independent in-process range-encoder states consume the same probabilities and symbols",
        "ratio_boundary": "signed ratios are algebraic; nonpositive cost is not a positive-spend exchange",
        "summaries": {key: distribution([r[key] for r in rows]) for key in (
            "tokens_changed", "delta_archive_bytes", "whole_archive_bits_per_changed_token",
            "individual_frame_ledger_delta_bits", "matched_pd4_first_order_credit_S",
            "nonlinear_isolated_credit_S", "signed_credit_S_per_byte", "positive_cost_credit_S_per_byte",
            "diagnostic_net_credit_S",
        )},
        "positive_diagnostic_net_count": sum(r["diagnostic_net_credit_S"] > 0 for r in rows),
        "input_winners": c.fact(c.STORE / "prices/WINNERS.json"),
        "source": c.fact(Path(__file__)),
    }
    c.save(c.STORE / "DISCOVERY_RESULTS.json", result)
    return result


def verify() -> dict:
    """Reconstruct every saved render, and recompute collateral from saved argmax."""
    import ddm_jg1_seg_solve as jg1

    # This checks target lineage only; all observations remain macOS CPU advisory.
    lineage = jg1.up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=jg1.up2.LINEAGE_DALI)
    binding = c.load(c.PREDECESSOR / "BINDING.json")["scorer_binding"]
    raw_path = Path(binding["raw_path"])
    raw_fact = c.fact(raw_path)
    if raw_fact["sha256"] != binding["raw_sha256"] or raw_fact["bytes"] != binding["raw_bytes"]:
        raise ValueError("raw reconstruction base drift")
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(1200, 874, 1164, 3))
    gt = jg1.load_gt_seg_labels(lineage["gt_lineage"])
    price_jobs = 0
    for path in sorted((c.STORE / "prices").glob("*/DONE.json")):
        done = c.load(path)
        for fact in [done["encode"], done["raw_expansion_certificate"], *done["archives"]]:
            c.price.checked_fact(fact)
        encoded = c.load(Path(done["encode"]["path"]))
        for fact in encoded["outputs"].values():
            c.price.checked_fact(fact)
        if not encoded["output_lossless"] or done["archives"][0]["sha256"] != done["archives"][1]["sha256"]:
            raise ValueError("a completed price lost known-symbol field reproduction or twin identity")
        price_jobs += 1
    packs = 0
    for path in sorted((c.STORE / "renderer").glob("*/PACK.json")):
        pack = c.load(path)
        for fact in pack["archives"]:
            c.price.checked_fact(fact)
        if pack["archives"][0]["sha256"] != pack["archives"][1]["sha256"]:
            raise ValueError("renderer twin identity lost")
        packs += 1
    frames = 0
    for metadata in sorted((c.STORE / "frames").glob("*.json")):
        if metadata.name.startswith("._"):
            continue
        doc = c.load(metadata)
        c.price.checked_fact(doc["payload"])
        payload = zlib.decompress(Path(doc["payload"]["path"]).read_bytes())
        if len(payload) != doc["raw_bytes"] or c.control.digest(payload) != doc["raw_sha256"]:
            raise ValueError("retained calibration render mismatch")
        frames += 1
    if frames != len([p for p in (c.STORE / "frames").glob("*.zlib") if not p.name.startswith("._")]):
        raise ValueError("orphaned calibration render payload")
    solves = 0
    for checksum in sorted((c.STORE / "solves").glob("*.sha.json")):
        if not checksum.name.startswith("._"):
            c.price.checked_fact(c.load(checksum))
            solves += 1
    renderer_rows, census_frames = [], 0
    for path in sorted((c.STORE / "renderer").glob("pair_*/COLLATERAL.json")):
        doc = c.load(path)
        rows = []
        for start in range(0, 600, 25):
            chunk = c.load(Path(doc["shared_collateral"]) / f"chunk_{start:03d}.json")
            for fact in [*chunk["payloads"], chunk["base_argmax"]]:
                c.price.checked_fact(fact)
            with np.load(chunk["payloads"][0]["path"], allow_pickle=False) as data:
                xor, predictions = data["xor_frames"], data["argmax"]
            with np.load(chunk["base_argmax"]["path"], allow_pickle=False) as data:
                baseline = data["argmax"]
            for offset, row in enumerate(chunk["rows"]):
                p = row["pair"]
                if p != start + offset or row["archive_sha256"] != doc["archive_sha256"]:
                    raise ValueError("collateral ordering or archive identity mismatch")
                restored = np.bitwise_xor(xor[offset], raw[2 * p + 1])
                if c.control.digest(restored.tobytes()) != row["raw_sha256"]:
                    raise ValueError("saved renderer frame reconstruction mismatch")
                before, after = baseline[offset] != gt[p], predictions[offset] != gt[p]
                expected = {
                    "base_errors": int(before.sum()), "errors": int(after.sum()),
                    "delta_cells": int(after.sum()) - int(before.sum()),
                    "new_errors": int((after & ~before).sum()), "fixed_errors": int((before & ~after).sum()),
                }
                if any(row[key] != value for key, value in expected.items()):
                    raise ValueError("collateral disagrees with saved argmax")
                rows.append(row)
                census_frames += 1
        if len(rows) != 600 or [r["pair"] for r in rows] != list(range(600)):
            raise ValueError("incomplete collateral census")
        credit = -rows[doc["pair"]]["delta_cells"]
        collateral = sum(r["delta_cells"] for r in rows if r["pair"] != doc["pair"])
        if (credit != doc["target_credit_cells"] or collateral != doc["other599_net_collateral_cells"]
                or (collateral > credit) != doc["collateral_exceeds_target_seg_credit"]):
            raise ValueError("collateral summary disagrees with 600 observed pairs")
        renderer_rows.append(doc)
    result = {
        "source": c.fact(Path(__file__)), "raw_base": raw_fact,
        "gt_lineage_check": lineage, "price_jobs_verified": price_jobs, "renderer_packs_verified": packs,
        "calibration_and_extension_frames": frames, "completed_solve_checksums": solves,
        "renderer_census_frames": census_frames, "renderer_actions": len(renderer_rows),
        "renderer_rows": renderer_rows, "axis": c.AXIS, "score_claim": False,
        "storage": c.storage(), "passed": True,
    }
    c.save(c.STORE / "PAYLOAD_VERIFICATION.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prices", "verify"))
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != c.STORE.resolve():
        raise ValueError("wrong resume store")
    c.bind()
    result = prices() if args.stage == "prices" else verify()
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "renderer_rows")}), flush=True)


if __name__ == "__main__":
    main()
