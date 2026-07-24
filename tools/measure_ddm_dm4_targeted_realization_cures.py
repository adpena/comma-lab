#!/usr/bin/env python3
"""Run the bounded DDM DM4 targeted realization-cure measurement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from tac.optimization.ddm_dm4_targeted_realization_cures import materialize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.config, args.output_dir)
    aggregate = receipt["aggregate"]
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "row_count": receipt["row_count"],
                "joint_semantics_exact": aggregate["semantic_records_joint_exact_after_composition"],
                "effective_ratio": aggregate["ratio"]["effective_bytes_per_semantic_byte"],
                "receipt": str(args.output_dir / "ddm_dm4_targeted_realization_cures_receipt.json"),
                "score_claim": receipt["score_claim"],
                "pointer": receipt["pointer"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
