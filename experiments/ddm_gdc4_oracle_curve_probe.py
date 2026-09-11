"""ddm_gdc4 stage-00e: the K+R curve of the BEST POSSIBLE bounded-endpoint generator.

Stage 00a proved that gdc1's retained scanline renders are exactly the optimal
``E``-bounded endpoint approximations of the move-44 field: their mismatch
counts reproduce this arm's independently computed ORACLE floor ``M(E)`` to the
unit. No generator that emits at most ``E`` ``(x_stop, class)`` symbols per row
-- learned, fitted, or hand-built -- can be more accurate than these renders.

This probe therefore prices the whole family at once. For each ``E`` it codes
the oracle render with this arm's own coders and reports ``K(E)``. Paired with
the residual ``R_exact(E)`` that gdc2 measured with exact closure on these same
retained renders against this same field, that is the family's complete
``K + R_exact`` curve. If its minimum is above the ``94,010 B`` door, no
bounded-endpoint generator at any accuracy can pass, and training one is
refused by arithmetic rather than by a run.

``research_only=true`` · ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac.gdc4_run_native_endpoint import (
    encode_field_transitions,
    frame_run_counts,
)

FIELD_SHAPE = (600, 384, 512)
DOOR_TOTAL_B = 94_010
SHIPPED_TAIL_B = 119_969

# R_exact measured by ddm_gdc2 with exact field closure on these exact retained
# renders against this exact field (memo sha 2f534b1d2ec91038..., Finding 2).
# Cited, not re-derived: the arm must not re-measure what an artifact settled.
GDC2_R_EXACT = {4: 174_680, 6: 105_628, 8: 60_520, 12: 11_772, 16: 1_238, 24: 0}
GDC1_PACKET = {4: 133_426, 6: 223_494, 8: 306_042, 12: 421_886, 16: 459_394, 24: 475_002}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def adaptive_cost_of(field_path: Path, out_dir: Path, tag: str) -> dict:
    """Run the arm's adaptive-cost probe on an arbitrary field file."""
    target = out_dir / f"adaptive_{tag}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("ddm_gdc4_adaptive_cost_probe.py")),
        "--field",
        str(field_path),
        "--out-dir",
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"adaptive probe failed for {tag}: {proc.stderr[-2000:]}")
    return json.loads((target / "RESULT.json").read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, help="the exact target field")
    parser.add_argument("--renders-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budgets", default="4,6,8,12,16,24")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)

    field_path = Path(args.field)
    root = Path(args.renders_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = [int(v) for v in args.budgets.split(",")]

    target = np.memmap(field_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)
    started = time.time()
    rows = []
    for e in budgets:
        render_path = root / f"k{e:02d}" / "receiver_render.u8"
        if not render_path.exists():
            raise SystemExit(f"missing retained oracle render: {render_path}")
        render = np.memmap(render_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)

        mismatches = 0
        max_runs = 0
        for f in range(FIELD_SHAPE[0]):
            frame = np.asarray(render[f])
            mismatches += int(np.count_nonzero(frame != np.asarray(target[f])))
            max_runs = max(max_runs, int(frame_run_counts(frame).max()))

        adaptive = adaptive_cost_of(render_path, out_dir, f"k{e:02d}")
        k_adaptive = int(adaptive["adaptive_total_bytes"])

        streams = encode_field_transitions(np.asarray(render))
        raw_total = sum(len(v) for v in streams.values())

        r_exact = GDC2_R_EXACT.get(e)
        total = k_adaptive + r_exact if r_exact is not None else None
        rows.append(
            {
                "endpoint_budget_E": e,
                "render_path": str(render_path),
                "render_sha256": sha256_file(render_path),
                "mismatches_vs_field": mismatches,
                "max_runs_in_any_row": max_runs,
                "gdc1_explicit_packet_bytes": GDC1_PACKET.get(e),
                "gdc4_adaptive_packet_bytes": k_adaptive,
                "gdc4_transition_stream_raw_bytes": raw_total,
                "r_exact_bytes_MEASURED_BY_GDC2": r_exact,
                "gdc4_total_bytes": total,
                "vs_door_bytes": (total - DOOR_TOTAL_B) if total is not None else None,
                "vs_door_ratio": (total / DOOR_TOTAL_B) if total is not None else None,
            }
        )
        print(
            f"[gdc4-curve] E={e:>2} M={mismatches:>9,} K_adaptive={k_adaptive:>9,} "
            f"R={r_exact} total={total}",
            flush=True,
        )

    feasible = [r for r in rows if r["gdc4_total_bytes"] is not None]
    best = min(feasible, key=lambda r: r["gdc4_total_bytes"])
    result = {
        "arm": "ddm_gdc4",
        "stage": "stage00e_oracle_bounded_endpoint_curve",
        "research_only": True,
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS-CPU byte-only n600]",
        "seed": int(args.seed),
        "field": {"path": str(field_path), "sha256": sha256_file(field_path)},
        "rows": rows,
        "best_endpoint_budget_E": best["endpoint_budget_E"],
        "best_total_bytes": best["gdc4_total_bytes"],
        "door_total_bytes": DOOR_TOTAL_B,
        "shipped_tail_bytes": SHIPPED_TAIL_B,
        "best_vs_door_ratio": best["gdc4_total_bytes"] / DOOR_TOTAL_B,
        "gate_pass": best["gdc4_total_bytes"] <= DOOR_TOTAL_B,
        "elapsed_s": time.time() - started,
        "note": (
            "K(E) is a free ONLINE-model code length: no weights are "
            "transmitted. Any learned generator must beat this number with a "
            "model whose weights it also has to pay for, at an accuracy the "
            "oracle already caps."
        ),
    }
    out_path = out_dir / "RESULT.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, out_path)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
