#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Advance one immutable G59 task-space codec campaign boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.witness_control.taskspace_codec_adversarial_gate_v2 import (  # noqa: E402
    AdversarialGateError,
    admit_encode,
    admit_post_eval,
    admit_pre_encode,
    admit_pre_promotion,
    admit_pre_public_closure,
    audit_g57_retrospective,
    canonical_json,
    seal_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--campaign-id", required=True)
    seal.add_argument("--representation", required=True)
    seal.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--assert-target-score", type=float)
    seal.add_argument("--assert-pointer-sha256")

    pre_encode = subparsers.add_parser("pre-encode")
    pre_encode.add_argument("--predecessor", type=Path, required=True)
    pre_encode.add_argument("--output", type=Path, required=True)
    pre_encode.add_argument("--producer-config", type=Path, required=True)
    pre_encode.add_argument("--g58-identity-receipt", type=Path)
    pre_encode.add_argument("--g58-terminal-stage-chain", type=Path)
    pre_encode.add_argument("--g58-outer-proof", type=Path)
    pre_encode.add_argument("--assert-representation")

    encode = subparsers.add_parser("encode")
    encode.add_argument("--predecessor", type=Path, required=True)
    encode.add_argument("--archive", type=Path, required=True)
    encode.add_argument("--decoded-raw", type=Path, required=True)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--assert-representation")

    post_eval = subparsers.add_parser("post-eval")
    post_eval.add_argument("--predecessor", type=Path, required=True)
    post_eval.add_argument("--eval-receipt", type=Path, required=True)
    post_eval.add_argument("--eval-report", type=Path, required=True)
    post_eval.add_argument("--integration-receipt", type=Path, action="append", default=[])
    post_eval.add_argument("--blocker", type=Path)
    post_eval.add_argument("--assert-representation")
    post_eval.add_argument("--output", type=Path, required=True)

    pre_public = subparsers.add_parser("pre-public")
    pre_public.add_argument("--predecessor", type=Path, required=True)
    pre_public.add_argument("--assert-archive", type=Path)
    pre_public.add_argument("--output", type=Path, required=True)

    pre_promotion = subparsers.add_parser("pre-promotion")
    pre_promotion.add_argument("--predecessor", type=Path, required=True)
    pre_promotion.add_argument("--public-auth-receipt", type=Path, required=True)
    pre_promotion.add_argument("--output", type=Path, required=True)

    retrospective = subparsers.add_parser("retrospective-g57")
    retrospective.add_argument("--campaign-id", required=True)
    retrospective.add_argument("--representation", required=True)
    retrospective.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    retrospective.add_argument("--g57-request", type=Path, required=True)
    retrospective.add_argument("--g57-receipt", type=Path, required=True)
    retrospective.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "seal":
            receipt = seal_campaign(
                campaign_id=args.campaign_id,
                requested_representation=args.representation,
                repo_root=args.repo_root,
                expected_repo_root=REPO_ROOT,
                output_path=args.output,
                asserted_target_score=args.assert_target_score,
                asserted_pointer_sha256=args.assert_pointer_sha256,
            )
        elif args.command == "pre-encode":
            receipt = admit_pre_encode(
                campaign_seal_path=args.predecessor,
                output_path=args.output,
                producer_config_path=args.producer_config,
                g58_identity_receipt_path=args.g58_identity_receipt,
                g58_terminal_stage_chain_path=args.g58_terminal_stage_chain,
                g58_outer_proof_path=args.g58_outer_proof,
                asserted_representation=args.assert_representation,
            )
        elif args.command == "encode":
            receipt = admit_encode(
                pre_encode_receipt_path=args.predecessor,
                archive_path=args.archive,
                decoded_raw_path=args.decoded_raw,
                output_path=args.output,
                asserted_representation=args.assert_representation,
            )
        elif args.command == "post-eval":
            receipt = admit_post_eval(
                encode_receipt_path=args.predecessor,
                eval_receipt_path=args.eval_receipt,
                eval_report_path=args.eval_report,
                integration_receipt_paths=args.integration_receipt,
                blocker_path=args.blocker,
                asserted_representation=args.assert_representation,
                output_path=args.output,
            )
        elif args.command == "pre-public":
            receipt = admit_pre_public_closure(
                post_eval_receipt_path=args.predecessor,
                asserted_archive_path=args.assert_archive,
                output_path=args.output,
            )
        elif args.command == "pre-promotion":
            receipt = admit_pre_promotion(
                pre_public_receipt_path=args.predecessor,
                public_auth_receipt_path=args.public_auth_receipt,
                output_path=args.output,
            )
        else:
            receipt = audit_g57_retrospective(
                campaign_id=args.campaign_id,
                requested_representation=args.representation,
                repo_root=args.repo_root,
                g57_request_path=args.g57_request,
                g57_receipt_path=args.g57_receipt,
                output_path=args.output,
            )
    except (AdversarialGateError, OSError, ValueError) as exc:
        sys.stderr.buffer.write(canonical_json({"status": "REFUSE", "error": str(exc)}))
        return 20
    sys.stdout.buffer.write(canonical_json(receipt))
    return 0 if receipt.get("candidate_admission") is True else 20


if __name__ == "__main__":
    raise SystemExit(main())
