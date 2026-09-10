#!/usr/bin/env python3
"""Measure the REAL exact residual cost of the retained GDC1 K=8 scanline teacher.

GDC1 screened the categorical-distillation door with GF1's transferred rate of
0.2909 B per mismatch, which GDC1 itself measured to be optimistic (K=6 closed at
0.4508 B per mismatch on the same field).  This probe replaces the transferred
constant with the measured one for the exact object GDC2 distills: it codes the
real exact residual that carries the K=8 teacher render back to the move-43
token field, races the three physical coders over all eight retained residual orders, proves
the winning residual closes the target exactly, and reports the packet budget
that the distilled decoder must fit inside.

No scorer, no Modal, no candidate archive, no score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

N_PAIRS: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
SHAPE: Final = (N_PAIRS, HEIGHT, WIDTH)
FIELD_PATH: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_pass6/retained/fields/pass6.u8"
)
FIELD_SHA256: Final = "78e57545439515eb29f806cc5a5f7d8b14acf658955cdd7561debb4edf3b7db6"
TEACHER_PATH: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gdc1_generator_door/scanline_v1/k08/receiver_render.u8"
)
TEACHER_SHA256: Final = "abd130921beec23255112e8b56148d55c49cf4fede03e99e0e99b55378b16ea2"
TEACHER_PACKET_BYTES: Final = 306_042
REPLACEMENT_INTEGER_CAP: Final = 94_010
GF1_BYTES_PER_MISMATCH: Final = 0.2909
RESIDUAL_ORDERS: Final = (
    "frame_raster",
    "class_frame_raster",
    "tile8_time",
    "tile16_time",
    "tile32_time",
    "tile64_time",
    "class_tile16_time",
    "pair_tile16",
)


class ProbeError(RuntimeError):
    """A GDC2 residual-probe source or custody invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_verified(path: Path, expected_sha: str) -> np.ndarray:
    if not path.is_file():
        raise ProbeError(f"missing input: {path}")
    if path.stat().st_size != N_PAIRS * HEIGHT * WIDTH:
        raise ProbeError(f"unexpected size for {path}: {path.stat().st_size}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise ProbeError(f"sha mismatch for {path}: {observed} != {expected_sha}")
    return np.memmap(path, dtype=np.uint8, mode="r").reshape(SHAPE)


def class_mismatch_table(target: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    """Per-target-class mismatch counts, streamed pair by pair (bounded RSS)."""
    counts = np.zeros(5, dtype=np.int64)
    total = 0
    for pair in range(N_PAIRS):
        differ = np.asarray(target[pair]) != np.asarray(generated[pair])
        total += int(differ.sum())
        if differ.any():
            counts += np.bincount(
                np.asarray(target[pair])[differ].astype(np.int64), minlength=5
            )
    return {"total": total, "by_target_class_0_to_4": [int(value) for value in counts]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render", type=Path, default=TEACHER_PATH,
                        help="generated field to code the exact residual from")
    parser.add_argument("--render-sha256", type=str, default="",
                        help="expected sha of --render; defaults to the K=8 teacher's sha")
    parser.add_argument("--render-packet-bytes", type=int, default=TEACHER_PACKET_BYTES,
                        help="the counted packet that produced --render")
    args = parser.parse_args(argv)

    started = time.monotonic()
    root: Path = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    render_path: Path = args.render
    expected = args.render_sha256 or (
        TEACHER_SHA256 if render_path == TEACHER_PATH else sha256_file(render_path))
    target = load_verified(FIELD_PATH, FIELD_SHA256)
    teacher = load_verified(render_path, expected)

    table = class_mismatch_table(target, teacher)
    mismatches = table["total"]
    if mismatches <= 0:
        raise ProbeError("teacher matched the field exactly; the probe premise is void")

    rows: list[dict[str, Any]] = []
    for order in RESIDUAL_ORDERS:
        raw_path = root / order / "residual.raw"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        fact = hg1.encode_residual(target, teacher, raw_path, None, order)
        race = hg1.coder_race(f"gdc2_teacher_residual_{order}", raw_path, root)
        winner = str(race["winner"])
        coded = race["coders"][winner]["coded"]
        rows.append(
            {
                "order": order,
                "raw": fact,
                "coder_race": race,
                "winner": winner,
                "coded": coded,
                "coded_bytes": int(coded["bytes"]),
            }
        )

    best = min(rows, key=lambda row: (row["coded_bytes"], row["order"]))
    corrected = np.array(teacher, copy=True)
    hg1.apply_residual(Path(best["raw"]["path"]).read_bytes(), corrected)
    if not np.array_equal(corrected, np.asarray(target)):
        raise ProbeError("winning residual did not close the target exactly")
    del corrected

    measured_rate = best["coded_bytes"] / mismatches
    result = {
        "schema": "ddm_gdc2_teacher_residual_probe.v1",
        "axis": "[macOS-CPU scorer-free exact-field measurement, n600]",
        "score_claim": False,
        "research_only": True,
        "promotable": False,
        "field": file_fact(FIELD_PATH),
        "teacher_render": file_fact(render_path),
        "teacher_packet_bytes": args.render_packet_bytes,
        "mismatch_table": table,
        "residual_rows": rows,
        "best_order": best["order"],
        "best_coded_bytes": best["coded_bytes"],
        "exact_target_closure": True,
        "measured_bytes_per_mismatch": measured_rate,
        "transferred_gf1_bytes_per_mismatch": GF1_BYTES_PER_MISMATCH,
        "transferred_over_measured_ratio": GF1_BYTES_PER_MISMATCH / measured_rate,
        "replacement_integer_cap": REPLACEMENT_INTEGER_CAP,
        "gdc1_clean_screen_packet_cap": REPLACEMENT_INTEGER_CAP
        - GF1_BYTES_PER_MISMATCH * mismatches,
        "real_packet_cap_at_exact_teacher_identity": REPLACEMENT_INTEGER_CAP
        - best["coded_bytes"],
        "teacher_packet_plus_real_residual": args.render_packet_bytes + best["coded_bytes"],
        "mismatch_budget_at_measured_rate_per_packet_bytes": {
            str(packet): math.floor((REPLACEMENT_INTEGER_CAP - packet) / measured_rate)
            for packet in (30_000, 40_000, 50_000, 60_000, 68_322)
        },
        "elapsed_seconds": time.monotonic() - started,
        "host": os.uname().nodename,
    }
    out = root / "RESULT.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "best_order", "best_coded_bytes", "measured_bytes_per_mismatch",
        "real_packet_cap_at_exact_teacher_identity", "exact_target_closure",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
