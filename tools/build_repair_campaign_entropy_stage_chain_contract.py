#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the entropy-stage repair campaign chain contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.repair_campaign_chain_contract import (  # noqa: E402
    RepairCampaignChainContractError,
    build_repair_campaign_entropy_stage_chain_contract,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-report", required=True, type=Path)
    parser.add_argument("--chain-contract-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        score_report_path = _resolve(args.score_report)
        score_report = json.loads(score_report_path.read_text(encoding="utf-8"))
        if not isinstance(score_report, dict):
            raise RepairCampaignChainContractError("score report must be a JSON object")
        contract = build_repair_campaign_entropy_stage_chain_contract(
            score_report=score_report,
            score_report_path=args.score_report,
            repo_root=REPO_ROOT,
        )
        out = _resolve(args.chain_contract_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if out.exists() and args.overwrite:
            existing_text = out.read_text(encoding="utf-8")
            next_text = json_text(contract)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                out,
                contract,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignChainContractError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"FATAL: repair campaign chain contract build failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_entropy_stage_chain_contract_cli_result.v1",
                "score_report": str(args.score_report),
                "chain_contract_out": str(args.chain_contract_out),
                "chain_node_count": contract["chain_node_count"],
                "ready_for_exact_eval_dispatch": False,
                "budget_spend_allowed": False,
                "bytes_written": (write_result.bytes_written if write_result is not None else 0),
                "skipped_identical_existing_artifact": (skipped_identical_existing_artifact),
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
