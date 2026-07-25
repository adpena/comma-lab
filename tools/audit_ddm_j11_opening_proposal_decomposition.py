#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the fail-closed J11 opening-proposal decomposition custody receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.optimization.ddm_j11_opening_proposal_decomposition import (  # noqa: E402
    J11ProposalDecompositionConfigV1,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    DirectDescriptionJointDescentTypedConfigV1,
)


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = J11ProposalDecompositionConfigV1.from_path(args.config)
    output = args.output or Path(config.output_path)
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(
        config.source_artifacts["j10_ticket"].resolve(config.repository_root)
    )
    receipt = typed.j11_opening_proposal_decomposition_source(
        audit_config_path=args.config,
    )
    payload = _canonical_json(receipt)
    if output.exists() and output.read_bytes() != payload:
        raise RuntimeError(f"immutable J11 receipt differs: {output}")
    if not output.exists():
        _atomic_write(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": receipt["verdict"],
                "materialized_single_components": receipt["candidate_counts"]["materialized_single_components"],
                "materialized_composed_pairs": receipt["candidate_counts"]["materialized_composed_pairs"],
                "n600_scorer_invoked": receipt["bounded_smoke"]["n600_scorer_invoked"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
