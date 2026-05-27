#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Safely rerun and diff a repair stackability replay bundle.

The rerun path rewrites the probe output into an operator-selected directory
and writes a separate rerun summary. It is still a macOS-MLX research-signal
verifier, never score, promotion, spend, or exact-dispatch authority.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY  # noqa: E402
from tac.optimization.proxy_candidate_contract import (  # noqa: E402
    require_no_truthy_authority_fields,
)
from tac.optimization.repair_campaign_replay_bundle import (  # noqa: E402
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_COMMAND_SCHEMA,
    REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA,
    RepairCampaignReplayBundleError,
    build_repair_campaign_stackability_replay_bundle,
    diff_repair_campaign_stackability_replay_bundles,
    stable_json_sha256,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)

TARGET_TOOL = "tools/run_repair_campaign_stackability_probe.py"
VALUE_OPTIONS_TO_REWRITE = {"--output"}


def _utc_compact() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _safe_id(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    return out.strip("._-").lower() or "repair_stackability_replay"


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else REPO_ROOT / value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairCampaignReplayBundleError(f"{path}: expected JSON object")
    return payload


def _write_json_artifact(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    expected_existing_sha256 = None
    if path.exists() and overwrite:
        existing_text = path.read_text(encoding="utf-8")
        next_text = json_text(payload)
        if existing_text == next_text:
            return
        expected_existing_sha256 = sha256_file(path)
    write_json_artifact(
        path,
        payload,
        allow_overwrite=overwrite,
        expected_existing_sha256=expected_existing_sha256,
    )


def _expect_bundle(payload: Mapping[str, Any], *, path: Path) -> None:
    if payload.get("schema") != REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA:
        raise RepairCampaignReplayBundleError(
            f"{path}: schema must be {REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_SCHEMA}, "
            f"got {payload.get('schema')!r}"
        )
    if payload.get("replay_target_tool") != TARGET_TOOL:
        raise RepairCampaignReplayBundleError(
            f"{path}: replay target must be {TARGET_TOOL}"
        )
    gate = payload.get("calibration_gate")
    if not isinstance(gate, Mapping):
        raise RepairCampaignReplayBundleError(f"{path}: calibration_gate missing")
    try:
        require_no_truthy_authority_fields(
            payload,
            context="repair_campaign_stackability_replay_rerun_source_bundle",
        )
    except ValueError as exc:
        raise RepairCampaignReplayBundleError(str(exc)) from exc


def _target_tool_args(argv: Sequence[str]) -> list[str]:
    for index, token in enumerate(argv):
        text = str(token)
        if text == TARGET_TOOL or text.endswith(f"/{TARGET_TOOL}"):
            return [str(item) for item in argv[index + 1 :]]
    raise RepairCampaignReplayBundleError(
        f"replay_argv does not call expected target tool: {TARGET_TOOL}"
    )


def _rewrite_output_arg(argv: Sequence[str], output_path: Path) -> list[str]:
    out: list[str] = []
    skip_next = False
    replaced = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        text = str(token)
        if text == "--output":
            out.extend(["--output", str(output_path)])
            skip_next = True
            replaced = True
            continue
        if text.startswith("--output="):
            out.append(f"--output={output_path}")
            replaced = True
            continue
        out.append(text)
    if not replaced:
        out.extend(["--output", str(output_path)])
    return out


def _source_record_path(bundle: Mapping[str, Any], *, label: str) -> Path:
    manifest = bundle.get("hash_manifest")
    records = manifest.get("source_records") if isinstance(manifest, Mapping) else []
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        if record.get("label") == label:
            path = str(record.get("path") or "").strip()
            if path:
                return _resolve(path)
    raise RepairCampaignReplayBundleError(f"source record missing: {label}")


def _custody_hashes(bundle: Mapping[str, Any]) -> dict[str, str]:
    manifest = bundle.get("hash_manifest")
    records = manifest.get("source_records") if isinstance(manifest, Mapping) else []
    hashes: dict[str, str] = {}
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        label = str(record.get("label") or "")
        if not label.startswith("local_mlx_custody:"):
            continue
        hashes[label] = str(record.get("sha256") or "")
    return hashes


def build_rerun_command(
    bundle: Mapping[str, Any],
    *,
    bundle_path: Path,
    output_dir: Path,
    python_executable: str,
    summary_out: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build a side-effect-contained repair stackability rerun command."""

    _expect_bundle(bundle, path=bundle_path)
    argv = bundle.get("replay_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise RepairCampaignReplayBundleError(f"{bundle_path}: replay_argv must be strings")
    typed_response_id = str(bundle.get("typed_response_id") or "repair_stackability")
    resolved_run_id = run_id or f"{bundle_path.stem}_{typed_response_id}_{_utc_compact()}"
    run_dir = output_dir / _safe_id(resolved_run_id)
    rerun_probe_path = run_dir / "repair_campaign_stackability_probe.rerun.json"
    rerun_bundle_path = run_dir / "repair_campaign_stackability_replay_bundle.rerun.json"
    diff_path = run_dir / "repair_campaign_stackability_replay_bundle.diff.json"
    summary_path = summary_out or (
        run_dir / "repair_campaign_stackability_replay_rerun_summary.json"
    )
    rewritten_args = _rewrite_output_arg(
        _target_tool_args(argv),
        rerun_probe_path,
    )
    command = [
        python_executable,
        str(REPO_ROOT / TARGET_TOOL),
        *rewritten_args,
    ]
    record = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_COMMAND_SCHEMA,
        "command": command,
        "run_dir": str(run_dir),
        "rerun_probe_path": str(rerun_probe_path),
        "rerun_bundle_path": str(rerun_bundle_path),
        "diff_path": str(diff_path),
        "summary_path": str(summary_path),
        "rewritten_options": sorted(VALUE_OPTIONS_TO_REWRITE),
        "side_effect_policy": {
            "original_probe_output_rewritten": True,
            "canonical_posterior_append_disabled": True,
            "score_or_dispatch_authority_granted": False,
        },
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        record,
        context="repair_campaign_stackability_replay_rerun_command",
    )
    return record


def rerun_and_diff(
    *,
    bundle_path: Path,
    output_dir: Path,
    python_executable: str,
    summary_out: Path | None = None,
    run_id: str | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], int]:
    bundle = _load_json(bundle_path)
    command_record = build_rerun_command(
        bundle,
        bundle_path=bundle_path,
        output_dir=output_dir,
        python_executable=python_executable,
        summary_out=summary_out,
        run_id=run_id,
    )
    run_dir = Path(command_record["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command_record["command"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    rerun_probe_path = Path(command_record["rerun_probe_path"])
    rerun_bundle_path = Path(command_record["rerun_bundle_path"])
    diff_path = Path(command_record["diff_path"])
    diff: dict[str, Any] | None = None
    original_probe_hash = None
    rerun_probe_hash = None
    probe_payload_matched = False
    custody_hashes_matched = False
    if proc.returncode == 0 and rerun_probe_path.is_file():
        original_probe_path = _source_record_path(
            bundle,
            label="repair_campaign_stackability_probe",
        )
        score_report_path = _source_record_path(
            bundle,
            label="repair_campaign_score_report",
        )
        score_report = _load_json(score_report_path)
        original_probe = _load_json(original_probe_path)
        rerun_probe = _load_json(rerun_probe_path)
        original_probe_hash = stable_json_sha256(original_probe)
        rerun_probe_hash = stable_json_sha256(rerun_probe)
        probe_payload_matched = original_probe_hash == rerun_probe_hash
        rerun_bundle = build_repair_campaign_stackability_replay_bundle(
            score_report_path=score_report_path,
            probe_path=rerun_probe_path,
            score_report=score_report,
            probe=rerun_probe,
            replay_argv=bundle["replay_argv"],
            invocation_argv=command_record["command"],
            repo_root=REPO_ROOT,
        )
        _write_json_artifact(
            rerun_bundle_path,
            rerun_bundle,
            overwrite=overwrite,
        )
        diff = diff_repair_campaign_stackability_replay_bundles(bundle, rerun_bundle)
        _write_json_artifact(diff_path, diff, overwrite=overwrite)
        custody_hashes_matched = _custody_hashes(bundle) == _custody_hashes(rerun_bundle)

    matched = bool(
        proc.returncode == 0
        and probe_payload_matched
        and custody_hashes_matched
    )
    summary = {
        "schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_RERUN_SCHEMA,
        "bundle_path": str(bundle_path),
        "command_record": command_record,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "original_probe_stable_json_sha256": original_probe_hash,
        "rerun_probe_stable_json_sha256": rerun_probe_hash,
        "probe_payload_matched": probe_payload_matched,
        "local_mlx_custody_hashes_matched": custody_hashes_matched,
        "bundle_diff_schema": REPAIR_CAMPAIGN_STACKABILITY_REPLAY_BUNDLE_DIFF_SCHEMA,
        "bundle_diff": diff,
        "bundle_diff_matched": bool(diff and diff.get("matched") is True),
        "matched": matched,
        "side_effect_policy": command_record["side_effect_policy"],
        "component_response_axis": "[macOS-MLX research-signal]",
        "budget_spend_allowed": False,
        "ready_for_budget_spend": False,
        "ready_for_exact_eval_dispatch": False,
        "allowed_use": "deterministic_local_mlx_repair_stackability_rerun_only",
        "forbidden_use": "score_claim_or_budget_spend_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(
        summary,
        context="repair_campaign_stackability_replay_rerun_summary",
    )
    _write_json_artifact(Path(command_record["summary_path"]), summary, overwrite=overwrite)
    return summary, 0 if matched else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary, code = rerun_and_diff(
            bundle_path=_resolve(args.bundle),
            output_dir=_resolve(args.output_dir),
            python_executable=args.python_executable,
            summary_out=_resolve(args.summary_out) if args.summary_out else None,
            run_id=args.run_id,
            overwrite=bool(args.overwrite),
        )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignReplayBundleError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair stackability replay rerun failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_stackability_replay_rerun_cli_result.v1",
                "bundle": str(args.bundle),
                "summary_out": summary["command_record"]["summary_path"],
                "matched": summary["matched"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
