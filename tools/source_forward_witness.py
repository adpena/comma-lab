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

import numpy as np

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
    parser.add_argument(
        "--packet",
        type=Path,
        default=None,
        help=(
            "Receiver packet bytes. Optional only when "
            "--checkpoint-export-report contains packet_path."
        ),
    )
    parser.add_argument(
        "--checkpoint-export-report",
        type=Path,
        default=None,
        help=(
            "SNeRV checkpoint archive export report. Resolves packet_path and "
            "the exported trained checkpoint state_dict slice for the strict "
            "source-forward witness; exact official source config and real "
            "frame triplets remain explicit unless the report already carries "
            "source-config/triplet paths."
        ),
    )
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
    parser.add_argument(
        "--capture-official-torch-from-upstream-source-graph",
        action="store_true",
        help=(
            "Strict source authority path. Requires exact trained source config, "
            "strict checkpoint state_dict, and real frame triplets."
        ),
    )
    parser.add_argument("--official-snerv-repo-dir", type=Path, default=None)
    parser.add_argument("--official-torch-train-one-step", action="store_true")
    parser.add_argument("--official-torch-checkpoint-state-dict", type=Path, default=None)
    parser.add_argument(
        "--official-torch-checkpoint-state-dict-kind",
        default="official_trained_checkpoint_state_dict",
    )
    parser.add_argument("--official-torch-source-config", type=Path, default=None)
    parser.add_argument(
        "--official-torch-source-config-kind",
        default="official_trained_run_config",
    )
    parser.add_argument(
        "--official-torch-source-frame-triplets-npy",
        type=Path,
        default=None,
        help=(
            "Numpy array with shape (pairs, 3, 3, H, W) in NCHW/0..255 order: "
            "current, previous, next."
        ),
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
    report_resolution = resolve_checkpoint_export_report_witness_inputs(
        args.checkpoint_export_report,
        packet_path=args.packet,
        official_torch_checkpoint_state_dict_path=(
            args.official_torch_checkpoint_state_dict
        ),
        official_torch_checkpoint_state_dict_kind=(
            args.official_torch_checkpoint_state_dict_kind
        ),
        official_torch_source_config_path=args.official_torch_source_config,
        official_torch_source_config_kind=args.official_torch_source_config_kind,
        official_torch_source_frame_triplets_npy=(
            args.official_torch_source_frame_triplets_npy
        ),
    )
    resolved_packet_path = report_resolution.get("packet_path")
    if resolved_packet_path is None:
        raise SystemExit(
            "pass --packet or a --checkpoint-export-report with packet_path"
        )
    payload = build_source_forward_witness_payload(
        packet_path=resolved_packet_path,
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
        capture_official_torch_from_upstream_source_graph=bool(
            args.capture_official_torch_from_upstream_source_graph
        ),
        official_snerv_repo_dir=(
            None if args.official_snerv_repo_dir is None else args.official_snerv_repo_dir
        ),
        official_torch_train_one_step=bool(args.official_torch_train_one_step),
        official_torch_checkpoint_state_dict_path=(
            None
            if report_resolution.get("official_torch_checkpoint_state_dict_path")
            is None
            else report_resolution["official_torch_checkpoint_state_dict_path"]
        ),
        official_torch_checkpoint_state_dict_kind=str(
            report_resolution.get("official_torch_checkpoint_state_dict_kind")
            or args.official_torch_checkpoint_state_dict_kind
        ),
        official_torch_source_config_path=(
            None
            if report_resolution.get("official_torch_source_config_path") is None
            else report_resolution["official_torch_source_config_path"]
        ),
        official_torch_source_config_kind=str(
            report_resolution.get("official_torch_source_config_kind")
            or args.official_torch_source_config_kind
        ),
        official_torch_source_frame_triplets_npy=(
            None
            if report_resolution.get("official_torch_source_frame_triplets_npy")
            is None
            else report_resolution["official_torch_source_frame_triplets_npy"]
        ),
        checkpoint_export_report_resolution=report_resolution,
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


def resolve_checkpoint_export_report_witness_inputs(
    checkpoint_export_report: str | Path | None,
    *,
    packet_path: str | Path | None = None,
    official_torch_checkpoint_state_dict_path: str | Path | None = None,
    official_torch_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
    official_torch_source_config_path: str | Path | None = None,
    official_torch_source_config_kind: str = "official_trained_run_config",
    official_torch_source_frame_triplets_npy: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve strict witness inputs without turning export metadata into proof."""

    blockers: list[str] = []
    report_path = (
        None
        if checkpoint_export_report is None
        else Path(checkpoint_export_report).expanduser().resolve(strict=False)
    )
    report: Mapping[str, Any] = {}
    if report_path is not None:
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            loaded = {}
            blockers.append(
                "snerv_source_forward_witness_checkpoint_export_report_unreadable:"
                f"{type(exc).__name__}"
            )
        if isinstance(loaded, Mapping):
            report = loaded
        else:
            blockers.append(
                "snerv_source_forward_witness_checkpoint_export_report_not_mapping"
            )
        if report.get("schema") not in {None, "snerv_checkpoint_archive_export.v1"}:
            blockers.append(
                "snerv_source_forward_witness_checkpoint_export_report_schema_mismatch"
            )
    base_dir = report_path.parent if report_path is not None else Path.cwd()
    binding = report.get("official_checkpoint_export_binding")
    binding = binding if isinstance(binding, Mapping) else {}
    resolved_packet = _resolve_first_path(
        base_dir,
        packet_path,
        report.get("packet_path"),
    )
    checkpoint_source = "explicit_cli"
    resolved_state_dict = _resolve_first_path(
        base_dir,
        official_torch_checkpoint_state_dict_path,
    )
    resolved_state_kind = str(official_torch_checkpoint_state_dict_kind)
    if resolved_state_dict is None:
        resolved_state_dict = _resolve_first_path(
            base_dir,
            binding.get("official_trained_checkpoint_state_dict_slice_path"),
            report.get("official_trained_checkpoint_state_dict_slice_path"),
        )
        if resolved_state_dict is not None:
            checkpoint_source = "checkpoint_export_official_state_dict_slice"
            resolved_state_kind = "checkpoint_export_official_trained_checkpoint_state_dict"
    if resolved_state_dict is None:
        resolved_state_dict = _resolve_first_path(base_dir, report.get("checkpoint_state_path"))
        if resolved_state_dict is not None:
            checkpoint_source = "checkpoint_export_native_mlx_checkpoint_state"
            resolved_state_kind = "checkpoint_export_native_mlx_receiver_state_dict"

    source_config_source = "explicit_cli"
    resolved_config = _resolve_first_path(base_dir, official_torch_source_config_path)
    resolved_config_kind = str(official_torch_source_config_kind)
    if resolved_config is None:
        resolved_config = _resolve_first_path(
            base_dir,
            report.get("official_torch_source_config_path"),
            report.get("official_trained_source_config_path"),
            report.get("official_source_config_path"),
            report.get("source_config_path"),
        )
        if resolved_config is not None:
            source_config_source = "checkpoint_export_report_source_config"
            resolved_config_kind = str(
                report.get("official_torch_source_config_kind")
                or report.get("official_trained_source_config_kind")
                or report.get("source_config_kind")
                or "checkpoint_export_official_trained_run_config"
            )

    triplets_source = "explicit_cli"
    resolved_triplets = _resolve_first_path(base_dir, official_torch_source_frame_triplets_npy)
    if resolved_triplets is None:
        resolved_triplets = _resolve_first_path(
            base_dir,
            report.get("official_torch_source_frame_triplets_npy"),
            report.get("official_torch_source_frame_triplets_path"),
            report.get("source_frame_triplets_npy"),
            report.get("source_frame_triplets_path"),
        )
        if resolved_triplets is not None:
            triplets_source = "checkpoint_export_report_source_frame_triplets"

    if report_path is not None:
        if resolved_packet is None:
            blockers.append("snerv_source_forward_witness_report_packet_path_missing")
        elif not resolved_packet.is_file():
            blockers.append(
                "snerv_source_forward_witness_report_packet_path_missing_on_disk"
            )
        if resolved_state_dict is None:
            blockers.append(
                "snerv_source_forward_witness_report_checkpoint_state_dict_path_missing"
            )
        elif not resolved_state_dict.is_file():
            blockers.append(
                "snerv_source_forward_witness_report_checkpoint_state_dict_path_missing_on_disk"
            )
        if resolved_config is None:
            blockers.append(
                "snerv_source_forward_witness_report_source_config_path_missing"
            )
        elif not resolved_config.is_file():
            blockers.append(
                "snerv_source_forward_witness_report_source_config_path_missing_on_disk"
            )
        if resolved_triplets is None:
            blockers.append(
                "snerv_source_forward_witness_report_source_frame_triplets_missing"
            )
        elif not resolved_triplets.is_file():
            blockers.append(
                "snerv_source_forward_witness_report_source_frame_triplets_missing_on_disk"
            )
    return {
        "schema": "snerv_source_forward_witness_input_resolution.v1",
        "checkpoint_export_report_path": (
            None if report_path is None else report_path.as_posix()
        ),
        "checkpoint_export_report_requested": report_path is not None,
        "packet_path": None if resolved_packet is None else resolved_packet.as_posix(),
        "official_torch_checkpoint_state_dict_path": (
            None if resolved_state_dict is None else resolved_state_dict.as_posix()
        ),
        "official_torch_checkpoint_state_dict_kind": resolved_state_kind,
        "official_torch_checkpoint_state_dict_source": checkpoint_source,
        "official_torch_source_config_path": (
            None if resolved_config is None else resolved_config.as_posix()
        ),
        "official_torch_source_config_kind": resolved_config_kind,
        "official_torch_source_config_source": source_config_source,
        "official_torch_source_frame_triplets_npy": (
            None if resolved_triplets is None else resolved_triplets.as_posix()
        ),
        "official_torch_source_frame_triplets_source": triplets_source,
        "startup_json_path_not_source_authority": report.get("startup_json_path"),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
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
    capture_official_torch_from_upstream_source_graph: bool = False,
    official_snerv_repo_dir: str | Path | None = None,
    official_torch_train_one_step: bool = False,
    official_torch_checkpoint_state_dict_path: str | Path | None = None,
    official_torch_checkpoint_state_dict_kind: str = (
        "official_trained_checkpoint_state_dict"
    ),
    official_torch_source_config_path: str | Path | None = None,
    official_torch_source_config_kind: str = "official_trained_run_config",
    official_torch_source_frame_triplets_npy: str | Path | None = None,
    checkpoint_export_report_resolution: Mapping[str, Any] | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    packet = Path(packet_path).expanduser().resolve(strict=False)
    generated = generated_utc or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolution_blockers = [
        str(value)
        for value in (
            (checkpoint_export_report_resolution or {}).get("blockers")
            if isinstance(checkpoint_export_report_resolution, Mapping)
            else []
        )
        or []
    ]
    if not packet.is_file():
        missing_blocker = "snerv_source_forward_witness_packet_path_missing_on_disk"
        fallback_material = f"missing-packet:{packet.as_posix()}:{pair_ids}".encode()
        return {
            "schema": SNERV_SOURCE_FORWARD_WITNESS_SCHEMA,
            "generated_utc": generated,
            "family": "snerv",
            "packet_path": packet.as_posix(),
            "packet_bytes": None,
            "packet_sha256": None,
            "pair_ids": list(pair_ids),
            "action_id": action_id or hashlib.sha256(fallback_material).hexdigest(),
            "capture_modes": {
                "checkpoint_export_report_requested": bool(
                    (checkpoint_export_report_resolution or {}).get(
                        "checkpoint_export_report_requested"
                    )
                ),
                "pact_mlx_from_archive": bool(capture_pact_mlx_from_archive),
                "official_torch_from_archive_diagnostic": bool(
                    capture_official_torch_from_archive
                ),
                "official_torch_from_upstream_fixture": bool(
                    capture_official_torch_from_upstream_fixture
                ),
                "official_torch_from_upstream_source_graph": bool(
                    capture_official_torch_from_upstream_source_graph
                ),
                "official_torch_source_config_requested": (
                    official_torch_source_config_path is not None
                ),
                "official_torch_source_frame_triplets_requested": (
                    official_torch_source_frame_triplets_npy is not None
                ),
            },
            "checkpoint_export_report_resolution": (
                dict(checkpoint_export_report_resolution)
                if isinstance(checkpoint_export_report_resolution, Mapping)
                else None
            ),
            "source_forward_proof_action_effect": None,
            "validation_status": {"passed": False, "blockers": []},
            "passed": False,
            "launch_gate_clearable": False,
            "output2_verdict": None,
            "first_failed_tensor": None,
            "blockers": _ordered_unique([*resolution_blockers, missing_blocker]),
            **FALSE_AUTHORITY,
        }
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
            "checkpoint_export_report_requested": bool(
                (checkpoint_export_report_resolution or {}).get(
                    "checkpoint_export_report_requested"
                )
            ),
            "pact_mlx_from_archive": bool(capture_pact_mlx_from_archive),
            "official_torch_from_archive_diagnostic": bool(
                capture_official_torch_from_archive
            ),
            "official_torch_from_upstream_fixture": bool(
                capture_official_torch_from_upstream_fixture
            ),
            "official_torch_from_upstream_source_graph": bool(
                capture_official_torch_from_upstream_source_graph
            ),
            "official_torch_source_config_requested": (
                official_torch_source_config_path is not None
            ),
            "official_torch_source_frame_triplets_requested": (
                official_torch_source_frame_triplets_npy is not None
            ),
        },
        "checkpoint_export_report_resolution": (
            dict(checkpoint_export_report_resolution)
            if isinstance(checkpoint_export_report_resolution, Mapping)
            else None
        ),
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
        source_frame_triplets = (
            None
            if official_torch_source_frame_triplets_npy is None
            else np.load(
                Path(official_torch_source_frame_triplets_npy).expanduser(),
                allow_pickle=False,
            )
        )
        row = build_snerv_source_forward_proof_from_archive_packet(
            action_id=resolved_action_id,
            archive_packet=packet_bytes,
            pair_ids=pair_ids,
            capture_official_torch_from_archive=capture_official_torch_from_archive,
            capture_official_torch_from_upstream_fixture=(
                capture_official_torch_from_upstream_fixture
            ),
            capture_official_torch_from_upstream_source_graph=(
                capture_official_torch_from_upstream_source_graph
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
            official_torch_source_config_path=(
                None
                if official_torch_source_config_path is None
                else Path(official_torch_source_config_path)
                .expanduser()
                .as_posix()
            ),
            official_torch_source_config_kind=official_torch_source_config_kind,
            official_torch_source_frame_triplets_nchw255=source_frame_triplets,
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
            "blockers": _ordered_unique([*resolution_blockers, blocker]),
        }
    validation = validate_snerv_source_forward_proof_action_effect(row)
    output2 = row.get("output2_boundary_verdict")
    blockers = _ordered_unique(
        [
            *resolution_blockers,
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


def _resolve_first_path(base_dir: Path, *values: Any) -> Path | None:
    for value in values:
        if value is None:
            continue
        raw = str(value).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve(strict=False)
    return None


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
