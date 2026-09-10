"""ddm_gdc4 stage-00a: size the run-native endpoint language on the move-44 field.

This is the pre-build screen the packet ABI needs. It measures, on the real
n600 token field and with no training, no scorer and no archive:

* the run population (runs per row, total runs, transition count);
* the ORACLE floor ``M_oracle(E)``: the minimum mismatch count achievable by
  ANY generator restricted to ``E`` ``(x_stop, class)`` symbols per row;
* free causal-context repeat rates (row-above and previous-frame-same-row),
  which bound what the receiver gets for zero counted bytes.

``M_oracle(E)`` is a hard floor on the charter's ``M <= 65,000`` screen and on
the endpoint budget the trainer may declare. Every row is labelled
``research_only=true`` / ``score_claim=false``.

Usage::

    .venv/bin/python experiments/ddm_gdc4_run_geometry_probe.py \
        --field /Volumes/.../pass6.u8 --out-dir /Volumes/.../run_geometry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tac.gdc4_run_native_endpoint import (
    frame_run_counts,
    optimal_bounded_endpoint_cost,
)

FIELD_SHAPE = (600, 384, 512)
BUDGETS = np.array([2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64], dtype=np.int64)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=FIELD_SHAPE[0])
    parser.add_argument("--oracle-frames", type=int, default=FIELD_SHAPE[0])
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)

    field_path = Path(args.field)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_frames = int(args.frames)
    n_rows, width = FIELD_SHAPE[1], FIELD_SHAPE[2]
    expected = FIELD_SHAPE[0] * n_rows * width
    size = field_path.stat().st_size
    if size != expected:
        raise SystemExit(f"field size {size} != expected {expected}")

    started = time.time()
    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)

    total_runs = 0
    max_runs = 0
    run_count_hist = np.zeros(width + 2, dtype=np.int64)
    class_cells = np.zeros(5, dtype=np.int64)
    rows_equal_above = 0
    rows_equal_prev_frame = 0
    rows_considered_above = 0
    rows_considered_prev = 0

    oracle_frames = min(int(args.oracle_frames), n_frames)
    oracle_total = np.zeros(BUDGETS.shape[0], dtype=np.int64)
    oracle_rows = 0

    prev_frame = None
    for f in range(n_frames):
        frame = np.asarray(field[f])
        counts = frame_run_counts(frame)
        total_runs += int(counts.sum())
        max_runs = max(max_runs, int(counts.max()))
        np.add.at(run_count_hist, counts, 1)
        class_cells += np.bincount(frame.reshape(-1), minlength=5).astype(np.int64)

        same_above = (frame[1:] == frame[:-1]).all(axis=1)
        rows_equal_above += int(same_above.sum())
        rows_considered_above += int(same_above.shape[0])
        if prev_frame is not None:
            same_prev = (frame == prev_frame).all(axis=1)
            rows_equal_prev_frame += int(same_prev.sum())
            rows_considered_prev += int(same_prev.shape[0])
        prev_frame = frame

        if f < oracle_frames:
            for r in range(n_rows):
                oracle_total += optimal_bounded_endpoint_cost(frame[r], BUDGETS)
                oracle_rows += 1

        if f % 50 == 0:
            elapsed = time.time() - started
            print(
                f"[gdc4-geom] frame {f}/{n_frames} runs={total_runs} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    rows_seen = n_frames * n_rows
    cells_seen = rows_seen * width
    elapsed = time.time() - started

    nz = np.flatnonzero(run_count_hist)
    counts_values = nz.astype(np.int64)
    counts_weights = run_count_hist[nz].astype(np.int64)
    cum = np.cumsum(counts_weights)
    total_rows_hist = int(cum[-1])

    def quantile(q: float) -> int:
        idx = int(np.searchsorted(cum, q * total_rows_hist))
        idx = min(idx, counts_values.shape[0] - 1)
        return int(counts_values[idx])

    mean_runs = float(total_runs) / float(rows_seen)

    # Byte arithmetic for the charter's door, DERIVED here, never a result.
    door_total_b = 94_010
    packet_screen_b = 60_000
    bits_available = packet_screen_b * 8
    bits_per_run_at_screen = bits_available / float(total_runs) if total_runs else 0.0

    oracle_rows_scaled = {}
    if oracle_rows:
        scale = float(rows_seen) / float(oracle_rows)
        for budget, value in zip(
            BUDGETS.tolist(), oracle_total.tolist(), strict=True
        ):
            oracle_rows_scaled[str(budget)] = {
                "measured_rows": oracle_rows,
                "measured_mismatches": int(value),
                "scaled_to_all_rows": float(value) * scale,
            }

    result = {
        "arm": "ddm_gdc4",
        "stage": "stage00a_run_geometry",
        "research_only": True,
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS-CPU byte-only n600]",
        "seed": int(args.seed),
        "field": {
            "path": str(field_path),
            "bytes": size,
            "sha256": sha256_file(field_path),
            "shape": list(FIELD_SHAPE),
        },
        "frames_scanned": n_frames,
        "rows_scanned": rows_seen,
        "cells_scanned": cells_seen,
        "run_population": {
            "total_runs": int(total_runs),
            "total_transitions": int(total_runs - rows_seen),
            "runs_per_row_mean": mean_runs,
            "runs_per_row_p50": quantile(0.50),
            "runs_per_row_p90": quantile(0.90),
            "runs_per_row_p99": quantile(0.99),
            "runs_per_row_max": int(max_runs),
        },
        "class_cell_counts": class_cells.tolist(),
        "free_causal_context": {
            "rows_identical_to_row_above": rows_equal_above,
            "rows_considered_above": rows_considered_above,
            "row_above_repeat_rate": (
                rows_equal_above / rows_considered_above if rows_considered_above else 0.0
            ),
            "rows_identical_to_prev_frame_same_row": rows_equal_prev_frame,
            "rows_considered_prev_frame": rows_considered_prev,
            "prev_frame_repeat_rate": (
                rows_equal_prev_frame / rows_considered_prev if rows_considered_prev else 0.0
            ),
        },
        "oracle_bounded_endpoint_floor": oracle_rows_scaled,
        "door_arithmetic_DERIVED": {
            "door_total_bytes": door_total_b,
            "packet_screen_bytes": packet_screen_b,
            "bits_per_run_at_packet_screen": bits_per_run_at_screen,
            "note": (
                "bits_per_run_at_packet_screen is the counted budget divided by "
                "the field's own run count: how many bits the packet could spend "
                "per run if it spent everything on runs and nothing on the model."
            ),
        },
        "elapsed_s": elapsed,
        "run_count_histogram": {
            str(int(v)): int(w)
            for v, w in zip(counts_values, counts_weights, strict=True)
        },
    }

    out_path = out_dir / "RESULT.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, out_path)
    print(json.dumps({k: v for k, v in result.items() if k != "run_count_histogram"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
