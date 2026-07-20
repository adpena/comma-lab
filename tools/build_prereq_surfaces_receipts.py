#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic local readiness receipts for prerequisite surfaces 1-3.

This is a small CPU-only structural builder.  It loads only the frozen SegNet
head tensors, performs no scorer forward, launches no training/eval job, and
writes no bulk artifact.  The surface-1 canary oracle checks exact decoded
resize numerators; the frozen-SegNet n6 hard-oracle anchor remains the separate
real measurement named in the emitted receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.prereq_surfaces import (  # noqa: E402
    build_frozen_rank4_prototype_bank,
    compare_affine_cell_representatives_same_coder,
    matched_continuous_to_uint8_hard_accept,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
    HardOracleEvaluation,
)

DEFAULT_WEIGHTS = Path("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors")
REAL_HARD_ACCEPT_ANCHOR = REPO / ".omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_receipts(weights: Path) -> dict[str, dict[str, Any]]:
    weights = weights.expanduser().resolve(strict=True)
    bank = build_frozen_rank4_prototype_bank(weights)
    comparator = compare_affine_cell_representatives_same_coder(bank)

    operator = DisjointResizeOperator.build(
        camera_h=8,
        camera_w=10,
        scorer_h=3,
        scorer_w=4,
    )
    source = np.full((8, 10, 1), 127, dtype=np.uint8)
    numerators, denominator = operator.apply_numerators(source)
    target = numerators / denominator

    def exact_decoded_numerator_oracle(frame: np.ndarray) -> HardOracleEvaluation:
        accepted = bool(np.array_equal(operator.apply_numerators(frame)[0], numerators))
        return HardOracleEvaluation(
            satisfied=np.array([accepted], dtype=bool),
            margins=np.array([1.0 if accepted else -1.0], dtype=np.float64),
        )

    adapter = matched_continuous_to_uint8_hard_accept(
        operator,
        np.linspace(-10.0, 265.0, 12).reshape(3, 4, 1),
        target,
        numerators,
        exact_decoded_numerator_oracle,
        pre_step_family="matched_receiver_structural_canary",
        max_nodes_per_block=50_000,
    )
    surface_1 = dict(adapter.receipt)
    hard_anchor = json.loads(REAL_HARD_ACCEPT_ANCHOR.read_text(encoding="ascii"))
    if (
        hard_anchor.get("schema") != "v10_uint8_lattice_feasibility_receipt.v1"
        or hard_anchor.get("authority", {}).get("score_claim") is not False
    ):
        raise ValueError("real hard-accept anchor schema/authority mismatch")
    exact_search = hard_anchor["aggregate"]["exact_search"]
    exact_arm = hard_anchor["aggregate"]["arms"]["exact_uint8_lattice_candidate"]
    pair_ids = hard_anchor["configuration"]["pair_ids"]
    surface_1.update(
        {
            "canary_scope": (
                "structural decoded-numerator HARD_ACCEPT only; each regmax probe must "
                "supply its own fresh frozen-SegNet hard oracle"
            ),
            "real_frozen_hard_accept_anchor": {
                "path": str(REAL_HARD_ACCEPT_ANCHOR.relative_to(REPO)),
                "sha256": _sha256_file(REAL_HARD_ACCEPT_ANCHOR),
                "measured_pairs": len(pair_ids),
                "pair_ids": pair_ids,
                "decoded_frames_with_exact_numerator_equality": exact_search[
                    "decoded_frames_with_exact_numerator_equality"
                ],
                "frozen_segnet_decoded_uint8_d_seg": exact_arm["d_seg"],
                "transfer_status": "ANCHOR_ONLY_NOT_A_RERUN_OF_CONTINUOUS_PREFERENCE",
            },
        }
    )
    return {
        "surface_1_matched_preimage_adapter": surface_1,
        "surface_2_rank4_prototype_bank": dict(bank.receipt),
        "surface_3_same_coder_comparator": comparator,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional durable directory for three JSON receipts; omitted prints a bundle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipts = build_receipts(args.weights)
    if args.output_dir is None:
        print(json.dumps(receipts, sort_keys=True, indent=2, allow_nan=False))
        return 0
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp")) or any(
        root in output_dir.parents
        for root in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"))
    ):
        raise SystemExit("output-dir must be durable and may not be a temporary directory")
    filenames = {
        "surface_1_matched_preimage_adapter": "surface_1_matched_preimage_adapter.json",
        "surface_2_rank4_prototype_bank": "surface_2_rank4_prototype_bank.json",
        "surface_3_same_coder_comparator": "surface_3_same_coder_comparator.json",
    }
    for key, filename in filenames.items():
        destination = output_dir / filename
        if destination.exists():
            raise SystemExit(f"refusing to overwrite preserved receipt: {destination}")
        _atomic_json(destination, receipts[key])
        print(f"{destination}\t{_sha256_file(destination)}")
    bundle = {
        "schema": "prereq_surfaces_flush_receipt_manifest.v1",
        "receipts": {
            filename: {
                "sha256": _sha256_file(output_dir / filename),
                "bytes": (output_dir / filename).stat().st_size,
            }
            for filename in filenames.values()
        },
        "builder_sha256": _sha256_file(Path(__file__).resolve()),
        "inputs_sha256": {
            "frozen_segnet_weights": _sha256_file(args.weights.expanduser().resolve()),
            "real_hard_accept_anchor": _sha256_file(REAL_HARD_ACCEPT_ANCHOR),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_json(output_dir / "manifest.json", bundle)
    print(
        f"{output_dir / 'manifest.json'}\t"
        f"{_sha256_file(output_dir / 'manifest.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
