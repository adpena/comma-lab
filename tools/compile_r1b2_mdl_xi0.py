#!/usr/bin/env python3
"""Audit R1b2 production custody and compile only a fully counted candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.r1b2_mdl_xi0_compile import (  # noqa: E402
    R1B2CompileError,
    atomic_json,
    audit_control_receipt,
    audit_full_kernel,
    audit_rank4_secants,
    audit_vjp_campaign,
    audit_xi0,
    build_receipt,
    compile_candidate_archive,
)

DEFAULT_CONTROL = REPO / ".omx/research/r1b_boundary_generator_solve_20260720.json"
DEFAULT_VJP = Path(
    "/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--control-receipt", type=Path, default=DEFAULT_CONTROL)
    result.add_argument("--vjp-campaign", type=Path, default=DEFAULT_VJP)
    result.add_argument("--rank4-secants", type=Path)
    result.add_argument("--full-kernel-mdl", type=Path)
    result.add_argument("--xi0", type=Path)
    result.add_argument("--candidate-output", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def execute(args: argparse.Namespace) -> int:
    control = audit_control_receipt(args.control_receipt)
    vjp = audit_vjp_campaign(args.vjp_campaign)
    blockers = list(vjp["blockers"])
    rank4, rank4_blockers = audit_rank4_secants(
        args.rank4_secants,
        vjp_campaign_sha256=vjp["campaign"]["sha256"],
    )
    full_kernel, full_kernel_blockers = audit_full_kernel(args.full_kernel_mdl)
    xi0, xi0_blockers = audit_xi0(args.xi0)
    blockers.extend(rank4_blockers)
    blockers.extend(full_kernel_blockers)
    blockers.extend(xi0_blockers)

    candidate = None
    if not blockers:
        if args.candidate_output is None:
            blockers.append("R1B2_CANDIDATE_OUTPUT_PATH_ABSENT")
        else:
            assert rank4 is not None and full_kernel is not None and xi0 is not None
            candidate = compile_candidate_archive(
                control_archive=Path(control["archive"]["path"]),
                boundary_packet=Path(rank4["boundary_packet"]["path"]),
                replay_payload=Path(full_kernel["replay"]["path"]),
                xi0_payload=Path(xi0["payload"]["path"]),
                source_manifest_hashes={
                    "vjp_campaign": vjp["campaign"]["sha256"],
                    "rank4_secants": rank4["custody"]["sha256"],
                    "full_kernel_mdl": full_kernel["custody"]["sha256"],
                    "xi0": xi0["custody"]["sha256"],
                },
                output=args.candidate_output,
            )
    receipt = build_receipt(
        control=control,
        vjp=vjp,
        rank4=rank4,
        full_kernel=full_kernel,
        xi0=xi0,
        blockers=blockers,
        candidate=candidate,
    )
    atomic_json(args.output.expanduser().resolve(), receipt)
    print(
        json.dumps(
            {
                "receipt": str(args.output.expanduser().resolve()),
                "verdict": receipt["verdict"],
                "blocker_count": len(receipt["blockers"]),
                "candidate": None if candidate is None else candidate["archive"],
            },
            sort_keys=True,
        )
    )
    return 0 if candidate is not None and not blockers else 3


def main() -> None:
    try:
        raise SystemExit(execute(parser().parse_args()))
    except R1B2CompileError as exc:
        raise SystemExit(f"R1B2_COMPILE_REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
