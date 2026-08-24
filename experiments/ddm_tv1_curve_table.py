#!/usr/bin/env python3
"""ddm_tv1 -- assemble the evaluator-tolerance curve from advisory rows.

Every number here is RECOMPUTED from the 8-decimal report components, never read
from a rounded ``final_score`` display (#877).  The rate term is held FIXED at
the shipped dx2 archive across every row, because no row changes the archive:
these are counterfactual FIELDS pushed through the same 180,368 B receiver, so
the only movement that exists is distortion.

Three quantities carry the finding:

``transfer``      Delta d_seg divided by k/117,964,800.  1.0 means one changed
                  token buys exactly one changed SegNet argmax pixel; below 1.0
                  the renderer plus R plus the stride-2 stem are absorbing the
                  movement, and that absorption IS the tolerance.

``credit_S``      the rate the changed positions actually hold in the shipped
                  stream, converted at 6.658590e-07 S/B.  This is an
                  ADDRESSING-FREE UPPER BOUND: it assumes the changed set is
                  free to name, which no real representation gets.

``cost_over_credit``  Delta S from distortion divided by credit_S.  Above 1.0
                  the movement costs more than the bits it could ever release,
                  even with the address given away.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

POSITIONS = 117_964_800
ORIGINAL_BYTES = 37_545_489
S_PER_BYTE = 6.658590e-07


def components(row: dict[str, Any]) -> tuple[float, float, float]:
    """(d_seg, d_pose, archive_bytes) from the 8dp report-derived fields."""
    d_seg = float(row["avg_segnet_dist_report_8dp_derived"])
    d_pose = float(row["avg_posenet_dist_report_8dp_derived"])
    return d_seg, d_pose, int(row["archive_size_bytes"])


def score_of(d_seg: float, d_pose: float, archive_bytes: int) -> dict[str, float]:
    seg = 100.0 * d_seg
    pose = math.sqrt(10.0 * d_pose)
    rate = 25.0 * archive_bytes / ORIGINAL_BYTES
    return {"seg": seg, "pose": pose, "rate": rate, "S": seg + pose + rate}


def build_table(base_path: Path, rows: list[tuple[Path, Path]]) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    b_seg, b_pose, b_bytes = components(base)
    base_score = score_of(b_seg, b_pose, b_bytes)

    out: list[dict[str, Any]] = []
    for manifest_path, eval_path in rows:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        row = json.loads(eval_path.read_text(encoding="utf-8"))
        d_seg, d_pose, archive_bytes = components(row)
        if archive_bytes != b_bytes:
            raise ValueError(
                f"{eval_path} scored a {archive_bytes} B archive; the tolerance "
                f"curve requires the shipped {b_bytes} B archive on every row"
            )
        score = score_of(d_seg, d_pose, archive_bytes)
        k = int(manifest["k_changed_verified"])
        credit = float(manifest["addressing_free_rate_credit_S"])
        delta_seg = d_seg - b_seg
        delta_pose = d_pose - b_pose
        delta_s = (score["seg"] - base_score["seg"]) + (score["pose"] - base_score["pose"])
        out.append(
            {
                "arm": manifest["arm"],
                "k": k,
                "field_sha256": manifest["field"]["sha256"],
                "derived_seed": manifest["derived_seed"],
                "d_seg": d_seg,
                "d_pose": d_pose,
                "delta_d_seg": delta_seg,
                "delta_d_pose": delta_pose,
                "delta_S_seg": score["seg"] - base_score["seg"],
                "delta_S_pose": score["pose"] - base_score["pose"],
                "delta_S_distortion": delta_s,
                "transfer": delta_seg / (k / POSITIONS) if k else None,
                "changed_bytes_in_shipped_stream": manifest[
                    "changed_bytes_in_shipped_stream"
                ],
                "credit_S_addressing_free": credit,
                "cost_over_credit": (delta_s / credit) if credit > 0 else None,
                "boundary_distance_histogram_0_to_4": manifest.get(
                    "boundary_distance_histogram_0_to_4"
                ),
                "value_marginal_fallbacks": manifest.get("value_marginal_fallbacks"),
                "eval_json": str(eval_path),
            }
        )
    out.sort(key=lambda r: (r["arm"], r["k"]))
    return {
        "schema": "ddm_tv1_tolerance_curve.v1",
        "positions": POSITIONS,
        "s_per_byte": S_PER_BYTE,
        "matched_base": {
            "path": str(base_path),
            "d_seg": b_seg,
            "d_pose": b_pose,
            "archive_bytes": b_bytes,
            **base_score,
        },
        "rows": out,
    }


def render(table: dict[str, Any]) -> str:
    base = table["matched_base"]
    lines = [
        f"matched base  d_seg {base['d_seg']:.8f}  d_pose {base['d_pose']:.8f}  "
        f"S {base['S']:.17g}",
        "",
        f"{'arm':<10} {'k':>9} {'d_seg':>11} {'Δd_seg':>12} {'transfer':>9} "
        f"{'Δd_pose':>12} {'ΔS_dist':>11} {'credit_S':>11} {'cost/credit':>12}",
    ]
    for row in table["rows"]:
        transfer = "-" if row["transfer"] is None else f"{row['transfer']:.4f}"
        ratio = "-" if row["cost_over_credit"] is None else f"{row['cost_over_credit']:.1f}"
        lines.append(
            f"{row['arm']:<10} {row['k']:>9,} {row['d_seg']:>11.8f} "
            f"{row['delta_d_seg']:>+12.8f} {transfer:>9} {row['delta_d_pose']:>+12.8f} "
            f"{row['delta_S_distortion']:>+11.6f} {row['credit_S_addressing_free']:>11.3e} "
            f"{ratio:>12}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        metavar="MANIFEST:EVAL_JSON",
        help="repeatable; a perturbed-field manifest and the row it produced",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for spec in args.row:
        manifest, _, evaluation = spec.rpartition(":")
        if not manifest or not evaluation:
            raise SystemExit(f"--row wants MANIFEST:EVAL_JSON, got {spec!r}")
        rows.append((Path(manifest), Path(evaluation)))

    table = build_table(args.base_eval, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    print(render(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
