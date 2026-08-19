"""ddm_br1 byte-close: turn the Gauss-Newton solve into a real archive, and price it.

Why a separate module from the solver: measurement and packaging are different jobs
with different failure modes, and ``ddm_up2``/``ddm_up3`` already split them the same
way.  This one owns only the question "what does the candidate actually weigh, and is
it still worth it once the container has had its say?"

WHY THE ARCHIVE, NOT THE RICE PAYLOAD
-------------------------------------
The Rice payload bits are the obvious cost and they are not the whole cost.  ``ddm_up3``
measured a candidate whose Rice payload was byte-identical to shipped yet whose archive
grew 48 B, because brotli compressed the perturbed stream worse -- 47% of that arm's
pose gain, invisible to a payload-level pricer.  So this module prices the **archive
bytes**, built through ``ddm_up3_carrier_splice.build_archive`` with its container
search and its parse-back verification, and never infers the archive size from the
payload size.

THE BASE BODY
-------------
``build_archive`` rebuilds from the **to1** base body and carries a codes array; the
live pointer ``7ce46fd7`` is exactly that base carrying ``ddm_up2``'s solved codes
(verified here as a control).  So a br1 candidate is the same base carrying br1's
codes, and the byte comparison against 176,420 B is like-for-like.

RATE-AWARE SELECTION
--------------------
Pairs are admitted greedily by pose gain, and every prefix is priced by BUILDING the
archive.  A pair whose gain does not cover its bytes is dropped rather than shipped,
so the reported net is the best admissible subset, not the full solve assumed free.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_up3_carrier_splice as splice

#: The live pointer row this candidate must beat.
LIVE_ARCHIVE_BYTES = 176_420
LIVE_D_POSE = br1.LIVE_D_POSE
LIVE_SCORE = br1.LIVE_POINTER_SCORE
BYTE_TO_SCORE = br1.BYTE_TO_SCORE
N_PAIRS = br1.up2.N_PAIRS_TOTAL


class ByteCloseError(RuntimeError):
    """A ddm_br1 byte-close precondition failed. Fail closed."""


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def pose_report_bound(d_pose: float) -> float:
    """Half-ULP of the 8dp d_pose report, in score units.

    The pose leg's sensitivity GROWS as d_pose falls, so this bound must be quoted
    per row and never folded into a single summed figure.
    """
    if d_pose <= 0.0:
        return pose_leg(br1.up2.REPORT_HALF_ULP)
    return 5.0 / math.sqrt(10.0 * d_pose) * br1.up2.REPORT_HALF_ULP


def live_codes() -> np.ndarray:
    """The codes actually inside the live pointer body, read from its bytes."""
    data = Path(br1.LIVE_RUNTIME / "archive.zip").read_bytes()
    codes, _info = splice.parse_back_codes(data)
    return np.asarray(codes, dtype=np.int32)


def control_live_body_is_up2(codes: np.ndarray) -> bool:
    """The live body must be the to1 base carrying up2's solved codes."""
    if not br1.UP2_SOLVED_CODES.is_file():
        return False
    return bool(np.array_equal(codes, np.load(br1.UP2_SOLVED_CODES).astype(np.int32)))


def build_and_measure(codes: np.ndarray, *, runtime: Path) -> dict[str, Any]:
    """Build the archive for ``codes`` and return its measured size and sha."""
    body = splice.parse_shipped_body(runtime_dir=runtime)
    built = splice.build_archive(
        body, codes, runtime_dir=runtime, container_search=True, verify=True
    )
    return built


def run(args) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = br1.load_done(Path(args.rows))
    if not rows:
        raise ByteCloseError(f"no solved rows at {args.rows}")

    base = live_codes()
    control_up2 = control_live_body_is_up2(base)
    if not control_up2 and not args.allow_base_mismatch:
        raise ByteCloseError(
            "CONTROL FAILED: the live body's codes are not ddm_up2's solved codes; "
            "the base this arm solved on top of is not the one it is pricing against"
        )

    improving = [
        r for r in rows.values() if r["final_d_pose"] < r["start_d_pose"]
    ]
    improving.sort(key=lambda r: r["start_d_pose"] - r["final_d_pose"], reverse=True)
    if not improving:
        raise ByteCloseError("no improving pairs to close")

    runtime = Path(args.base_runtime)
    levels = sorted({
        max(1, int(round(len(improving) * f)))
        for f in (0.1, 0.25, 0.5, 0.75, 1.0)
    })

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    payload_dir = out_dir / "archives"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for count in levels:
        chosen = improving[:count]
        candidate = base.copy()
        for row in chosen:
            candidate[int(row["pair"])] = np.asarray(row["codes"], dtype=np.int32)
        built = build_and_measure(candidate, runtime=runtime)
        # ALWAYS KEEP THE PAYLOAD: the archive bytes exist in memory here, so they are
        # written before anything is measured from them. Persisting only the LENGTH of
        # a payload we materialised is the forbidden measure-and-discard pattern, and
        # every level is retained, not only the winner.
        archive_bytes = int(built["archive_size"])
        level_path = payload_dir / f"archive_level_{count:04d}.zip"
        level_path.write_bytes(built["archive_bytes"])
        np.save(payload_dir / f"codes_level_{count:04d}.npy", candidate)
        gain = sum(r["start_d_pose"] - r["final_d_pose"] for r in chosen)
        d_pose_new = LIVE_D_POSE - gain / N_PAIRS
        delta_pose = pose_leg(d_pose_new) - pose_leg(LIVE_D_POSE)
        delta_rate = (archive_bytes - LIVE_ARCHIVE_BYTES) * BYTE_TO_SCORE
        net = delta_pose + delta_rate
        row_out = {
            "pairs_admitted": count,
            "archive_bytes": archive_bytes,
            "delta_bytes": archive_bytes - LIVE_ARCHIVE_BYTES,
            "archive_sha256": built.get("archive_sha256"),
            "archive_path": str(level_path),
            "d_pose_final": d_pose_new,
            "delta_score_pose": delta_pose,
            "delta_score_rate": delta_rate,
            "net_delta_score": net,
            "pose_report_bound_live": pose_report_bound(LIVE_D_POSE),
            "pose_report_bound_candidate": pose_report_bound(d_pose_new),
        }
        results.append(row_out)
        print(
            f"[level {count}/{len(improving)}] bytes {archive_bytes} "
            f"(dB {archive_bytes - LIVE_ARCHIVE_BYTES:+d})  pose {delta_pose:+.4e}  "
            f"rate {delta_rate:+.4e}  NET {net:+.4e}",
            flush=True,
        )
        if best is None or net < best["net_delta_score"]:
            best = dict(row_out)
            best["codes"] = candidate

    if best is None:
        raise ByteCloseError("no level produced a result")

    codes_path = out_dir / "br1_candidate_codes.npy"
    np.save(codes_path, best.pop("codes"))

    summary = {
        "schema": "ddm_br1_byte_close.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "control_live_body_is_to1_base_plus_up2_codes": control_up2,
        "live_pointer": {
            "archive_sha256": br1.LIVE_ARCHIVE_SHA256,
            "archive_bytes": LIVE_ARCHIVE_BYTES,
            "score": LIVE_SCORE,
            "d_pose_dali": LIVE_D_POSE,
        },
        "rows_solved": len(rows),
        "pairs_improving": len(improving),
        "admit_bar": br1.ADMIT_BAR,
        "levels": results,
        "best": best,
        "clears_bar": bool(best["net_delta_score"] < br1.ADMIT_BAR),
        "candidate_codes": str(codes_path),
    }
    (out_dir / "BYTE_CLOSE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "levels"}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="rows.jsonl from br1 mode=gn")
    parser.add_argument(
        "--base-runtime",
        default=str(splice.DEFAULT_RUNTIME),
        help="the to1 BASE body that build_archive rebuilds from",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="proceed even if the live body is not to1-base + up2 codes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
