#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build or promote the typed G55 public selected-plane archive closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.witness_control.taskspace_codec_adversarial_gate_v2 import (  # noqa: E402
    DIRECT_TASK_LAYERED_CONTROL,
    PRE_PROMOTION,
    AdversarialGateError,
    require_live_admission_receipt,
)
from tac.witness_dsl.taskspace_layered_public_closure_v1 import (  # noqa: E402
    ClosureError,
    build_preview,
    promote,
    read_json,
    sha256_file,
    stage_exact_eval,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument(
        "--adversarial-pre-promotion-receipt",
        type=Path,
        help="same-object live G59 PRE_PROMOTION receipt required for promotion",
    )
    parser.add_argument("--auth-receipt", type=Path)
    parser.add_argument("--auth-receipt-sha256")
    parser.add_argument(
        "--stage-exact-eval",
        action="store_true",
        help=("materialize archive.zip for upstream/evaluate.sh without making a promotion or score claim"),
    )
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        receipt = build_preview(config)
        preview_path = Path(receipt["archive_preview"]["path"])
        if args.stage_exact_eval:
            staged_archive, staging_receipt = stage_exact_eval(
                preview_path,
                receipt,
            )
            receipt["exact_eval_staged_archive_path"] = str(staged_archive)
            receipt["exact_eval_staging_receipt"] = staging_receipt
        if args.auth_receipt is not None:
            if not args.auth_receipt_sha256:
                raise ClosureError("--auth-receipt-sha256 is required for promotion")
            if args.campaign_id is None or args.adversarial_pre_promotion_receipt is None:
                raise AdversarialGateError(
                    "--campaign-id and --adversarial-pre-promotion-receipt are required for promotion"
                )
            require_live_admission_receipt(
                args.adversarial_pre_promotion_receipt,
                expected_stage=PRE_PROMOTION,
                expected_campaign_id=args.campaign_id,
                expected_repo_root=REPO_ROOT,
                expected_representation=DIRECT_TASK_LAYERED_CONTROL,
                expected_config_path=args.config,
                expected_archive_path=preview_path,
                expected_public_auth_path=args.auth_receipt,
            )
            build_receipt_path = Path(receipt["receipt_path"])
            build_receipt = read_json(
                build_receipt_path,
                sha256_file(build_receipt_path),
                "G55 build receipt",
            )
            archive_path = promote(
                Path(receipt["archive_preview"]["path"]),
                build_receipt,
                args.auth_receipt,
                args.auth_receipt_sha256,
                repo_root=Path(config.get("repo_root", ".")),
            )
            receipt["promoted_archive_path"] = str(archive_path)
            receipt["promoted_archive_sha256"] = sha256_file(archive_path)
    except (AdversarialGateError, ClosureError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
