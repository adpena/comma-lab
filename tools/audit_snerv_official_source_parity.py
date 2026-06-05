#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit official SNeRV OSS controls against the local receiver-safe adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.snerv_official_source_forward_harness import (  # noqa: E402
    build_snerv_official_source_forward_harness_artifact,
    build_snerv_official_trained_checkpoint_mapping_manifest,
)
from tac.analysis.snerv_official_source_parity_audit import (  # noqa: E402
    build_snerv_official_source_parity_audit,
    render_snerv_official_source_parity_markdown,
)
from tac.repo_io import sha256_file, write_json_artifact, write_text_artifact  # noqa: E402
from tac.substrates._shared.numpy_portable_inflate import (  # noqa: E402
    unpack_state_dict_numpy,
)


def _default_output_json() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_official_source_parity_audit_{stamp}.json"


def _default_forward_parity_artifact_json() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / f".omx/research/snerv_official_mfu_hfr_tub_forward_parity_{stamp}.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--official-repo-dir",
        type=Path,
        required=True,
        help="SSD-backed checkout of https://github.com/qwertja/SNeRV.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--official-forward-parity-artifact",
        type=Path,
        help=(
            "Optional JSON proof artifact with schema "
            "snerv_official_mfu_hfr_tub_forward_parity.v1. A marker constant "
            "alone is never enough to prove official MFU/HFR/TUB parity."
        ),
    )
    parser.add_argument(
        "--output-forward-parity-artifact",
        type=Path,
        nargs="?",
        const=_default_forward_parity_artifact_json(),
        help=(
            "Write a source-backed MFU/HFR/TUB proof-or-falsification artifact "
            "and consume it in the audit report. Provide a path or omit the value "
            "to use .omx/research."
        ),
    )
    parser.add_argument(
        "--checkpoint-export-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_checkpoint_archive_export.v1 JSON carrying "
            "official_checkpoint_export_binding evidence. Repeatable; closes "
            "only receiver-bound export debt, not source-forward authority."
        ),
    )
    parser.add_argument(
        "--trained-checkpoint-state-dict",
        action="append",
        default=[],
        type=Path,
        help=(
            "Optional trusted local NumPy-portable .npsd trained checkpoint "
            "state. Loaded into a mapping manifest; native MLX keys still do "
            "not prove official decoder source-forward parity."
        ),
    )
    parser.add_argument(
        "--tub-source-forward-artifact",
        type=Path,
        help=(
            "Optional snerv_official_tub_source_forward_replay.v1 JSON. When "
            "omitted, the audit builds the bounded local TUB source-fixture "
            "artifact itself."
        ),
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    parser.add_argument("--expected-output-forward-parity-artifact-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.official_forward_parity_artifact and args.output_forward_parity_artifact:
        parser.error("--official-forward-parity-artifact and --output-forward-parity-artifact are mutually exclusive")
    output_json = args.output_json or _default_output_json()
    if not output_json.is_absolute():
        output_json = REPO_ROOT / output_json
    output_md = args.output_md
    if output_md is not None and not output_md.is_absolute():
        output_md = REPO_ROOT / output_md
    output_forward_parity_artifact = args.output_forward_parity_artifact
    if output_forward_parity_artifact is not None and not output_forward_parity_artifact.is_absolute():
        output_forward_parity_artifact = REPO_ROOT / output_forward_parity_artifact

    checkpoint_export_reports = tuple(
        _load_json(path) for path in args.checkpoint_export_report
    )
    trained_checkpoint_mapping_manifests = (
        *_trained_checkpoint_manifests_from_export_reports(checkpoint_export_reports),
        *(
            _load_trained_checkpoint_mapping_manifest(path)
            for path in args.trained_checkpoint_state_dict
        ),
    )

    forward_artifact_result = None
    forward_artifact_path = args.official_forward_parity_artifact
    if output_forward_parity_artifact is not None:
        forward_artifact = build_snerv_official_source_forward_harness_artifact(
            official_repo_dir=args.official_repo_dir,
            repo_root=args.repo_root,
            checkpoint_export_reports=checkpoint_export_reports,
            trained_checkpoint_mapping_manifests=(
                trained_checkpoint_mapping_manifests
            ),
            tub_source_forward_artifact=(
                None
                if args.tub_source_forward_artifact is None
                else _load_json(args.tub_source_forward_artifact)
            ),
        )
        forward_artifact_result = write_json_artifact(
            output_forward_parity_artifact,
            forward_artifact,
            allow_overwrite=args.expected_output_forward_parity_artifact_sha256 is not None,
            expected_existing_sha256=args.expected_output_forward_parity_artifact_sha256,
        )
        forward_artifact_path = Path(forward_artifact_result.path)

    report = build_snerv_official_source_parity_audit(
        official_repo_dir=args.official_repo_dir,
        repo_root=args.repo_root,
        official_forward_parity_artifact_path=forward_artifact_path,
    )
    json_result = write_json_artifact(
        output_json,
        report,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    md_result = None
    if output_md is not None:
        md_result = write_text_artifact(
            output_md,
            render_snerv_official_source_parity_markdown(report),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "authority": report["authority"],
                "official_source_markers_present": report["official_source_markers_present"],
                "local_receiver_safe_adapter_present": report["local_receiver_safe_adapter_present"],
                "official_mfu_hfr_tub_parity_proven": report["official_mfu_hfr_tub_parity_proven"],
                "blocker_count": len(report["blockers"]),
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
                "output_json": json_result.path,
                "output_json_sha256": json_result.sha256,
                "output_md": None if md_result is None else md_result.path,
                "output_md_sha256": None if md_result is None else md_result.sha256,
                "output_forward_parity_artifact": (
                    None if forward_artifact_result is None else forward_artifact_result.path
                ),
                "output_forward_parity_artifact_sha256": (
                    None if forward_artifact_result is None else forward_artifact_result.sha256
                ),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    payload.setdefault("_source_path", path.as_posix())
    return payload


def _trained_checkpoint_manifests_from_export_reports(
    reports: tuple[dict, ...],
) -> tuple[dict, ...]:
    manifests = []
    for report in reports:
        state_path_raw = report.get("checkpoint_state_path")
        if not state_path_raw:
            continue
        manifests.append(
            _load_trained_checkpoint_mapping_manifest(
                Path(str(state_path_raw)),
                expected_sha256=str(report.get("checkpoint_state_sha256") or ""),
                state_dict_kind=(
                    "checkpoint_export_native_mlx_state_dict_"
                    f"{report.get('checkpoint_state_kind') or 'unknown'}"
                ),
                source_label=str(report.get("_source_path") or ""),
            )
        )
    return tuple(manifests)


def _load_trained_checkpoint_mapping_manifest(
    path: Path,
    *,
    expected_sha256: str = "",
    state_dict_kind: str = "trusted_local_native_mlx_checkpoint_state_dict",
    source_label: str | None = None,
) -> dict:
    resolved = path.expanduser().resolve(strict=False)
    source = source_label or resolved.as_posix()
    if not resolved.is_file():
        return _failed_trained_checkpoint_mapping_manifest(
            path=resolved,
            source=source,
            state_dict_kind=state_dict_kind,
            blocker="snerv_official_trained_checkpoint_state_dict_path_missing",
        )
    actual_sha256 = sha256_file(resolved)
    expected = str(expected_sha256 or "").strip().lower()
    if expected and actual_sha256 != expected:
        manifest = _failed_trained_checkpoint_mapping_manifest(
            path=resolved,
            source=source,
            state_dict_kind=state_dict_kind,
            blocker="snerv_official_trained_checkpoint_state_dict_sha256_mismatch",
        )
        manifest["expected_state_dict_sha256"] = expected
        manifest["actual_state_dict_sha256"] = actual_sha256
        return manifest
    try:
        state = unpack_state_dict_numpy(resolved.read_bytes())
    except Exception as exc:  # pragma: no cover - fail-closed file-format guard.
        manifest = _failed_trained_checkpoint_mapping_manifest(
            path=resolved,
            source=source,
            state_dict_kind=state_dict_kind,
            blocker="snerv_official_trained_checkpoint_state_dict_load_failed",
        )
        manifest["load_error_type"] = type(exc).__name__
        manifest["load_error"] = str(exc)
        return manifest
    manifest = build_snerv_official_trained_checkpoint_mapping_manifest(
        state,
        decoder_len=None,
        state_dict_kind=state_dict_kind,
        source=source,
    )
    manifest["state_dict_path"] = resolved.as_posix()
    manifest["state_dict_file_sha256"] = actual_sha256
    return manifest


def _failed_trained_checkpoint_mapping_manifest(
    *,
    path: Path,
    source: str,
    state_dict_kind: str,
    blocker: str,
) -> dict:
    return {
        "schema": "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1",
        "state_dict_kind": state_dict_kind,
        "state_dict_source": source,
        "state_dict_path": path.as_posix(),
        "state_dict_key_count": 0,
        "decoder_len": None,
        "official_trained_checkpoint_loaded": False,
        "official_mfu_hfr_trained_checkpoint_weight_mapping_proven": False,
        "official_tub_temporal_encoder_weight_mapping_proven": False,
        "mapped_weight_key_count": 0,
        "weight_entries": [],
        "component_rows": [],
        "closed_campaign_blockers": [],
        "blockers": [
            blocker,
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_trained_checkpoint_source_forward_replay_missing",
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "production_hardened_claim": False,
        "source_faithful_stack_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
