#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit a fail-closed SNeRV source-forward witness from a byte-real packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_source_forward_producer import (  # noqa: E402
    build_snerv_source_forward_proof_from_archive_packet,
)
from tac.analysis.snerv_source_forward_proof import (  # noqa: E402
    validate_snerv_source_forward_proof_action_effect,
)
from tac.repo_io import sha256_bytes, write_json_artifact  # noqa: E402

SNERV_SOURCE_FORWARD_WITNESS_SCHEMA = "snerv_source_forward_witness_cli.v1"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _default_output() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_source_forward_witness_{stamp}.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pair-ids", default="0")
    parser.add_argument("--action-id", default=None)
    parser.add_argument("--bitflip-section", default="decoder_payload")
    parser.add_argument("--bitflip-offset", default=0, type=int)
    parser.add_argument("--bitflip-mask", default=1, type=int)
    parser.add_argument("--capture-pact-mlx-from-archive", action="store_true")
    parser.add_argument(
        "--capture-official-torch-from-archive-diagnostic",
        action="store_true",
        help=(
            "Receiver-bound diagnostic only. This does not prove upstream "
            "SNeRV_T source-forward authority."
        ),
    )
    parser.add_argument(
        "--capture-official-torch-from-upstream-fixture",
        action="store_true",
    )
    parser.add_argument("--official-snerv-repo-dir", type=Path, default=None)
    parser.add_argument("--official-torch-train-one-step", action="store_true")
    parser.add_argument("--official-torch-checkpoint-state-dict", type=Path, default=None)
    parser.add_argument(
        "--official-torch-checkpoint-state-dict-kind",
        default="official_trained_checkpoint_state_dict",
    )
    parser.add_argument("--fail-on-blockers", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    parser.add_argument(
        "--proof-row-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path for the nested "
            "snerv_source_forward_proof_action_effect row. Writes zero rows "
            "when row construction failed, preserving fail-closed semantics."
        ),
    )
    parser.add_argument("--expected-proof-row-jsonl-sha256", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    out = args.out or _default_output()
    if not out.is_absolute():
        out = REPO_ROOT / out
    payload = build_source_forward_witness_payload(
        packet_path=args.packet,
        pair_ids=_parse_pair_ids(args.pair_ids),
        action_id=args.action_id,
        bitflip_section=args.bitflip_section,
        bitflip_offset=args.bitflip_offset,
        bitflip_mask=args.bitflip_mask,
        capture_pact_mlx_from_archive=bool(args.capture_pact_mlx_from_archive),
        capture_official_torch_from_archive=bool(
            args.capture_official_torch_from_archive_diagnostic
        ),
        capture_official_torch_from_upstream_fixture=bool(
            args.capture_official_torch_from_upstream_fixture
        ),
        official_snerv_repo_dir=(
            None if args.official_snerv_repo_dir is None else args.official_snerv_repo_dir
        ),
        official_torch_train_one_step=bool(args.official_torch_train_one_step),
        official_torch_checkpoint_state_dict_path=(
            None
            if args.official_torch_checkpoint_state_dict is None
            else args.official_torch_checkpoint_state_dict
        ),
        official_torch_checkpoint_state_dict_kind=str(
            args.official_torch_checkpoint_state_dict_kind
        ),
    )
    result = write_json_artifact(
        out,
        payload,
        allow_overwrite=bool(args.allow_overwrite),
        expected_existing_sha256=args.expected_output_sha256,
    )
    proof_row_result: dict[str, Any] | None = None
    if args.proof_row_jsonl is not None:
        proof_row_result = write_proof_row_jsonl(
            args.proof_row_jsonl,
            payload,
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_sha256=args.expected_proof_row_jsonl_sha256,
        )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "passed": payload["passed"],
                "launch_gate_clearable": payload["launch_gate_clearable"],
                "output2_verdict": payload.get("output2_verdict"),
                "first_failed_tensor": payload.get("first_failed_tensor"),
                "blocker_count": len(payload.get("blockers") or []),
                "output_json": result.path,
                "output_json_sha256": result.sha256,
                **(
                    {
                        "proof_row_jsonl": proof_row_result["path"],
                        "proof_row_jsonl_count": proof_row_result["row_count"],
                        "proof_row_jsonl_sha256": proof_row_result["sha256"],
                    }
                    if proof_row_result is not None
                    else {}
                ),
            },
            sort_keys=True,
        )
    )
    if args.fail_on_blockers and payload.get("blockers"):
        return 2
    return 0


def write_proof_row_jsonl(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    allow_overwrite: bool = False,
    expected_existing_sha256: str | None = None,
) -> dict[str, Any]:
    out = Path(path)
    if not out.is_absolute():
        out = REPO_ROOT / out
    if out.exists():
        existing_sha256 = sha256_bytes(out.read_bytes())
        if expected_existing_sha256 is not None and existing_sha256 != expected_existing_sha256:
            raise ValueError(
                f"existing proof row JSONL sha256 mismatch for {out}: "
                f"expected {expected_existing_sha256}, got {existing_sha256}"
            )
        if not allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing proof row JSONL: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    row = payload.get("source_forward_proof_action_effect")
    if row is None:
        content = ""
        row_count = 0
    elif isinstance(row, Mapping):
        content = json.dumps(dict(row), sort_keys=True) + "\n"
        row_count = 1
    else:
        raise TypeError("source_forward_proof_action_effect must be an object or null")
    out.write_text(content, encoding="utf-8")
    return {
        "path": out.as_posix(),
        "sha256": sha256_bytes(out.read_bytes()),
        "row_count": row_count,
    }


def build_source_forward_witness_payload(
    *,
    packet_path: str | Path,
    pair_ids: list[int],
    action_id: str | None = None,
    bitflip_section: str = "decoder_payload",
    bitflip_offset: int = 0,
    bitflip_mask: int = 1,
    capture_pact_mlx_from_archive: bool = False,
    capture_official_torch_from_archive: bool = False,
    capture_official_torch_from_upstream_fixture: bool = False,
    official_snerv_repo_dir: str | Path | None = None,
    official_torch_train_one_step: bool = False,
    official_torch_checkpoint_state_dict_path: str | Path | None = None,
    official_torch_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
    generated_utc: str | None = None,
) -> dict[str, Any]:
    packet = Path(packet_path).expanduser().resolve(strict=False)
    generated = generated_utc or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    packet_bytes = packet.read_bytes()
    packet_sha256 = sha256_bytes(packet_bytes)
    resolved_action_id = action_id or _default_action_id(packet_sha256, pair_ids)
    base: dict[str, Any] = {
        "schema": SNERV_SOURCE_FORWARD_WITNESS_SCHEMA,
        "generated_utc": generated,
        "family": "snerv",
        "packet_path": packet.as_posix(),
        "packet_bytes": len(packet_bytes),
        "packet_sha256": packet_sha256,
        "pair_ids": list(pair_ids),
        "action_id": resolved_action_id,
        "capture_modes": {
            "pact_mlx_from_archive": bool(capture_pact_mlx_from_archive),
            "official_torch_from_archive_diagnostic": bool(
                capture_official_torch_from_archive
            ),
            "official_torch_from_upstream_fixture": bool(
                capture_official_torch_from_upstream_fixture
            ),
        },
        "source_forward_proof_action_effect": None,
        "validation_status": {"passed": False, "blockers": []},
        "passed": False,
        "launch_gate_clearable": False,
        "output2_verdict": None,
        "first_failed_tensor": None,
        "blockers": [],
        **FALSE_AUTHORITY,
    }
    try:
        row = build_snerv_source_forward_proof_from_archive_packet(
            action_id=resolved_action_id,
            archive_packet=packet_bytes,
            pair_ids=pair_ids,
            capture_official_torch_from_archive=capture_official_torch_from_archive,
            capture_official_torch_from_upstream_fixture=(
                capture_official_torch_from_upstream_fixture
            ),
            official_snerv_repo_dir=(
                None
                if official_snerv_repo_dir is None
                else Path(official_snerv_repo_dir).expanduser().as_posix()
            ),
            official_torch_train_one_step=official_torch_train_one_step,
            official_torch_checkpoint_state_dict_path=(
                None
                if official_torch_checkpoint_state_dict_path is None
                else Path(official_torch_checkpoint_state_dict_path)
                .expanduser()
                .as_posix()
            ),
            official_torch_checkpoint_state_dict_kind=(
                official_torch_checkpoint_state_dict_kind
            ),
            capture_pact_mlx_from_archive=capture_pact_mlx_from_archive,
            bitflip_section=bitflip_section,
            bitflip_offset=bitflip_offset,
            bitflip_mask=bitflip_mask,
            generated_utc=generated,
        )
    except Exception as exc:
        blocker = f"snerv_source_forward_witness_build_failed:{type(exc).__name__}"
        return {
            **base,
            "build_exception": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "blockers": [blocker],
        }
    validation = validate_snerv_source_forward_proof_action_effect(row)
    output2 = row.get("output2_boundary_verdict")
    blockers = _ordered_unique(
        [
            *[str(value) for value in row.get("blockers") or []],
            *[
                f"snerv_source_forward_proof_invalid:{value}"
                for value in validation.get("blockers") or []
            ],
        ]
    )
    return {
        **base,
        "source_forward_proof_action_effect": row,
        "validation_status": validation,
        "passed": bool(validation.get("passed") is True),
        "launch_gate_clearable": bool(row.get("launch_gate_clearable") is True),
        "output2_verdict": (
            output2.get("verdict") if isinstance(output2, dict) else None
        ),
        "first_failed_tensor": row.get("first_failed_tensor"),
        "blockers": blockers,
    }


def _parse_pair_ids(raw: str) -> list[int]:
    values = [item.strip() for item in raw.replace(",", " ").split()]
    pair_ids = [int(item) for item in values if item]
    if not pair_ids:
        raise ValueError("--pair-ids must name at least one pair id")
    if any(pair_id < 0 for pair_id in pair_ids):
        raise ValueError("--pair-ids must be non-negative")
    return pair_ids


def _default_action_id(packet_sha256: str, pair_ids: list[int]) -> str:
    material = f"snerv-source-forward-witness:{packet_sha256}:{pair_ids}".encode()
    return hashlib.sha256(material).hexdigest()


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
