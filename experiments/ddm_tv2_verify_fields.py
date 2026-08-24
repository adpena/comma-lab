#!/usr/bin/env python3
"""ddm_tv2 -- independently re-derive every number the tolerance verdict rests on.

WHY RE-DERIVE RATHER THAN READ THE MANIFEST
-------------------------------------------
The tolerance verdict is a ratio, tau = delta_d_seg / (k / N).  Its denominator
comes from the perturbed field and its numerator from the scorer, so a wrong k
silently rescales every conclusion.  ``ddm_tv1`` recorded ``k_changed_verified``
in each field manifest, but a manifest is the producer's own claim about its
own output; confirming it against itself proves nothing.  This script recounts
k directly from the retained field bytes.

The credit column is cross-checked the same way.  ``tv1`` priced the changed
positions with the coding conditional it captured.  ``ddm_tb2`` independently
produced a per-position cost field from the same object by a different route.
Agreement between two producers is evidence; agreement of one producer with
itself is not.  Where they disagree, the disagreement is the finding.

Reported, per field:
  * k_recounted vs the manifest's k_changed_verified (and vs k_target);
  * frames touched -- the prefix-bias guard.  Prefix bias on this object is
    measured and axis-dependent (pose 2.5-4.2x harder on a contiguous prefix,
    seg ~0.96x), and this is a pose-bearing measurement, so a field that failed
    to touch all 600 frames would be disqualified rather than discounted;
  * bits held at the changed positions under tb2's cost field, and the byte and
    S-credit that follow at the CITED exchange rate;
  * the class-transition matrix, recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NUM_CLASSES = 5
FRAMES, EVAL_H, EVAL_W = 600, 384, 512
PLANE = EVAL_H * EVAL_W
POSITIONS = FRAMES * PLANE
# ddm_tx1_toolbox_crosswalk_20260819.md section 0 -- CITED, never re-derived.
S_PER_BYTE = 6.658590e-07


class VerifyError(RuntimeError):
    """Fail-closed error for verification."""


def load_field(path: Path) -> np.ndarray:
    """Memory-map a uint8 token field, refusing any wrong-sized file."""
    size = path.stat().st_size
    if size != POSITIONS:
        raise VerifyError(f"{path.name}: {size} B, expected {POSITIONS} B")
    return np.memmap(path, dtype=np.uint8, mode="r")


def verify_one(label: str, field_path: Path, base: np.ndarray,
               cost: np.ndarray | None, manifest: dict | None) -> dict:
    """Recount one perturbed field and price its changed positions."""
    pert = load_field(field_path)

    # Recount in blocks so peak memory stays bounded while the machine is busy
    # running scorer rows.
    block = 8 * PLANE
    changed_idx_parts: list[np.ndarray] = []
    for start in range(0, POSITIONS, block):
        stop = min(start + block, POSITIONS)
        diff = np.nonzero(np.asarray(base[start:stop]) != np.asarray(pert[start:stop]))[0]
        if diff.size:
            changed_idx_parts.append(diff + start)
    changed = (np.concatenate(changed_idx_parts) if changed_idx_parts
               else np.zeros(0, dtype=np.int64))
    k = int(changed.size)

    frames_touched = int(np.unique(changed // PLANE).size) if k else 0

    out: dict = {
        "label": label,
        "k_recounted": k,
        "frames_touched": frames_touched,
        "touches_all_600_frames": frames_touched == FRAMES,
        "one_to_one_transfer_delta_d_seg": k / POSITIONS,
    }

    if manifest is not None:
        claimed = manifest.get("k_changed_verified")
        out["k_manifest"] = claimed
        out["k_target"] = manifest.get("k_target")
        out["k_matches_manifest"] = (claimed == k)
        out["manifest_credit_S"] = manifest.get("addressing_free_rate_credit_S")
        out["manifest_changed_bits"] = manifest.get("changed_bits_in_shipped_stream")

    if cost is not None and k:
        bits = float(np.asarray(cost[changed], dtype=np.float64).sum())
        out["bits_held_tb2"] = bits
        out["bytes_held_tb2"] = bits / 8.0
        out["credit_S_tb2"] = (bits / 8.0) * S_PER_BYTE
        if manifest is not None and manifest.get("changed_bits_in_shipped_stream"):
            claimed_bits = float(manifest["changed_bits_in_shipped_stream"])
            out["bits_ratio_tb2_over_manifest"] = (
                bits / claimed_bits if claimed_bits else None)

    if k:
        trans = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        src = np.asarray(base[changed])
        dst = np.asarray(pert[changed])
        np.add.at(trans, (src, dst), 1)
        out["class_transitions_from_to"] = trans.tolist()
        out["self_transitions"] = int(np.trace(trans))

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-field", required=True, type=Path)
    parser.add_argument("--perturbed-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--cost-field", type=Path, default=None,
                        help="tb2 per-position cost field, float64 LE bits")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    base = load_field(args.base_field)

    cost = None
    if args.cost_field is not None:
        expect = POSITIONS * 8
        actual = args.cost_field.stat().st_size
        if actual != expect:
            raise VerifyError(f"cost field {actual} B, expected {expect} B")
        cost = np.memmap(args.cost_field, dtype="<f8", mode="r")

    results = []
    for label in args.labels:
        field_path = args.perturbed_dir / f"{label}.u8"
        if not field_path.is_file():
            sys.stdout.write(f"[skip] {label}: field absent\n")
            continue
        manifest = None
        if args.manifest_dir is not None:
            mpath = args.manifest_dir / f"{label}.json"
            if mpath.is_file():
                manifest = json.loads(mpath.read_text())
        row = verify_one(label, field_path, base, cost, manifest)
        results.append(row)
        sys.stdout.write(
            f"[ok] {row['label']}: k={row['k_recounted']} "
            f"match={row.get('k_matches_manifest')} "
            f"frames={row['frames_touched']} "
            f"bytes_tb2={row.get('bytes_held_tb2', float('nan')):.1f} "
            f"ratio={row.get('bits_ratio_tb2_over_manifest')}\n")
        sys.stdout.flush()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"schema": "ddm_tv2_field_verification.v1",
             "s_per_byte": S_PER_BYTE,
             "base_field": str(args.base_field),
             "cost_field": str(args.cost_field) if args.cost_field else None,
             "results": results}, indent=2, sort_keys=True))
        sys.stdout.write(f"[write] {args.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
