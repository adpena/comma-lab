#!/usr/bin/env python3
"""Plan paired CPU and CUDA auth evals for the same archive/runtime.

CUDA remains the internal promotion/ranking gate. CPU replay is a separate
public-leaderboard reproduction axis. This planner keeps both commands tied to
the same archive bytes so CUDA-vs-CPU drift is measured rather than inferred.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.auth_eval_records import (
        inflated_output_manifest_summary,
        parse_auth_eval_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - script execution from tools/
    from auth_eval_records import inflated_output_manifest_summary, parse_auth_eval_payload

DEFAULT_LEDGER = Path("experiments/results/pr100_107_reproduction_ledger_20260507_codex/ledger.json")
CUDA_SCORE_GRADES = {
    "a++",
    "[contest-cuda]",
    "contest-cuda",
    "contest-cuda-t4",
    "[contest-cuda t4]",
}


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._-") or "auth_eval"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _input_closure(paths: dict[str, Path]) -> dict[str, Any]:
    entries = {}
    missing = []
    for name, path in paths.items():
        exists = path.exists()
        entries[name] = {
            "path": str(path),
            "exists": exists,
            "is_file": path.is_file() if exists else False,
        }
        if not exists:
            missing.append(name)
    return {
        "required_inputs": entries,
        "missing_inputs": missing,
        "ready_to_execute": not missing,
    }


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _runtime_tree_sha256(payload: dict[str, Any]) -> str | None:
    candidates = (
        payload.get("runtime_content_tree_sha256"),
        _nested_get(payload, "provenance", "runtime_content_tree_sha256"),
        _nested_get(
            payload,
            "provenance",
            "inflate_runtime_manifest",
            "runtime_content_tree_sha256",
        ),
        _nested_get(payload, "inflate_runtime_manifest", "runtime_content_tree_sha256"),
        _nested_get(payload, "runtime", "runtime_content_tree_sha256"),
        payload.get("runtime_tree_sha256"),
        _nested_get(payload, "provenance", "runtime_tree_sha256"),
        _nested_get(payload, "provenance", "inflate_runtime_manifest", "runtime_tree_sha256"),
        _nested_get(payload, "inflate_runtime_manifest", "runtime_tree_sha256"),
        _nested_get(payload, "runtime", "runtime_tree_sha256"),
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _artifact_axis_summary(
    *,
    artifact_json: Path | None,
    expected_device: str,
) -> tuple[dict[str, Any], list[str]]:
    axis_label = "contest_cuda" if expected_device == "cuda" else "contest_cpu"
    summary: dict[str, Any] = {
        "axis": axis_label,
        "device": expected_device,
        "path": str(artifact_json) if artifact_json else None,
        "provided": artifact_json is not None,
        "exists": False,
        "valid_for_axis": False,
    }
    blockers: list[str] = []
    if artifact_json is None:
        blockers.append(f"missing_{axis_label}_score_artifact")
        return summary, blockers
    if not artifact_json.exists():
        blockers.append(f"missing_{axis_label}_score_artifact")
        return summary, blockers

    summary["exists"] = True
    try:
        payload = json.loads(artifact_json.read_text(encoding="utf-8"))
    except Exception as exc:
        blockers.append(f"{axis_label}_score_artifact_unreadable:{type(exc).__name__}")
        return summary, blockers
    if not isinstance(payload, dict):
        blockers.append(f"{axis_label}_score_artifact_not_json_object")
        return summary, blockers

    record = parse_auth_eval_payload(payload)
    if record is None:
        blockers.append(f"{axis_label}_score_artifact_unparseable")
        return summary, blockers

    grade = (record.evidence_grade or "").strip()
    grade_key = grade.lower()
    device = record.device.lower()
    summary.update(
        {
            "device": device,
            "score": record.score,
            "archive_bytes": record.archive_bytes,
            "archive_sha256": record.archive_sha256,
            "avg_segnet_dist": record.avg_segnet_dist,
            "avg_posenet_dist": record.avg_posenet_dist,
            "n_samples": record.samples,
            "evidence_grade": grade,
            "promotion_eligible": record.promotion_eligible,
            "score_claim_valid": record.score_claim_valid,
            "runtime_tree_sha256": _runtime_tree_sha256(payload),
            "inflated_output_manifest": inflated_output_manifest_summary(payload),
        }
    )

    if device != expected_device:
        blockers.append(f"{axis_label}_device_mismatch:{device}")
    if record.score is None:
        blockers.append(f"{axis_label}_score_missing")
    if record.archive_sha256 is None:
        blockers.append(f"{axis_label}_archive_sha256_missing")
    if record.archive_bytes is None:
        blockers.append(f"{axis_label}_archive_bytes_missing")
    if record.samples != 600:
        blockers.append(f"{axis_label}_not_full_sample_600")
    if expected_device == "cpu" and "contest-cpu" not in grade_key:
        blockers.append("contest_cpu_evidence_grade_missing")
    if expected_device == "cuda" and not (
        record.promotion_eligible
        or grade_key in CUDA_SCORE_GRADES
        or "contest-cuda" in grade_key
    ):
        blockers.append("contest_cuda_evidence_grade_missing")

    summary["valid_for_axis"] = not blockers
    return summary, blockers


def _dual_axis_completion(
    *,
    archive_meta: dict[str, Any],
    cpu_artifact_json: Path | None,
    cuda_artifact_json: Path | None,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    blockers: list[str] = []
    for device, path in (("cuda", cuda_artifact_json), ("cpu", cpu_artifact_json)):
        summary, axis_blockers = _artifact_axis_summary(
            artifact_json=path,
            expected_device=device,
        )
        artifacts[device] = summary
        blockers.extend(axis_blockers)

    cuda = artifacts["cuda"]
    cpu = artifacts["cpu"]
    same_archive_sha256 = (
        cuda.get("archive_sha256") == cpu.get("archive_sha256")
        if cuda.get("archive_sha256") and cpu.get("archive_sha256")
        else False
    )
    same_archive_bytes = (
        cuda.get("archive_bytes") == cpu.get("archive_bytes")
        if cuda.get("archive_bytes") is not None and cpu.get("archive_bytes") is not None
        else False
    )
    same_runtime_tree_sha256 = None
    for device, artifact in artifacts.items():
        if artifact.get("provided") and not artifact.get("runtime_tree_sha256"):
            blockers.append(f"{device}_artifact_runtime_tree_sha256_missing")
    if cuda.get("runtime_tree_sha256") and cpu.get("runtime_tree_sha256"):
        same_runtime_tree_sha256 = cuda["runtime_tree_sha256"] == cpu["runtime_tree_sha256"]
        if not same_runtime_tree_sha256:
            blockers.append("cpu_cuda_runtime_tree_sha256_mismatch")

    cpu_raw = cpu.get("inflated_output_manifest")
    cuda_raw = cuda.get("inflated_output_manifest")
    same_inflated_output_aggregate_sha256 = None
    raw_output_pairing_status = "raw_output_manifest_missing"
    if cpu_raw and cuda_raw:
        same_inflated_output_aggregate_sha256 = (
            cpu_raw.get("aggregate_sha256") == cuda_raw.get("aggregate_sha256")
        )
        raw_output_pairing_status = (
            "same_inflated_outputs"
            if same_inflated_output_aggregate_sha256
            else "different_inflated_outputs"
        )
    elif cpu_raw or cuda_raw:
        raw_output_pairing_status = "partial_raw_output_manifest"

    if cuda.get("provided") and cpu.get("provided") and not same_archive_sha256:
        blockers.append("cpu_cuda_archive_sha256_mismatch")
    if cuda.get("provided") and cpu.get("provided") and not same_archive_bytes:
        blockers.append("cpu_cuda_archive_bytes_mismatch")

    planned_sha = archive_meta.get("sha256")
    planned_bytes = archive_meta.get("bytes")
    for device, artifact in artifacts.items():
        if planned_sha and artifact.get("archive_sha256"):
            matches = artifact["archive_sha256"] == planned_sha
            artifact["matches_planned_archive_sha256"] = matches
            if not matches:
                blockers.append(f"{device}_artifact_archive_sha256_mismatch_with_plan")
        if planned_bytes is not None and artifact.get("archive_bytes") is not None:
            matches = artifact["archive_bytes"] == planned_bytes
            artifact["matches_planned_archive_bytes"] = matches
            if not matches:
                blockers.append(f"{device}_artifact_archive_bytes_mismatch_with_plan")

    unique_blockers = list(dict.fromkeys(blockers))
    paired_complete = not unique_blockers
    return {
        "schema": "dual_axis_auth_eval_completion.v1",
        "required_axes": ["contest_cuda", "contest_cpu"],
        "artifacts": artifacts,
        "same_archive_sha256": same_archive_sha256,
        "same_archive_bytes": same_archive_bytes,
        "same_runtime_tree_sha256": same_runtime_tree_sha256,
        "same_inflated_output_aggregate_sha256": same_inflated_output_aggregate_sha256,
        "raw_output_pairing_status": raw_output_pairing_status,
        "paired_score_artifacts_complete": paired_complete,
        "frontier_or_medal_band_complete": paired_complete,
        "global_priority_eligible": paired_complete,
        "rank_or_kill_eligible": paired_complete,
        "blockers": unique_blockers,
        "notes": [
            "A candidate is paired/complete only when both contest-CUDA and contest-CPU score artifacts are present.",
            "Both artifacts must parse as full-sample auth evals for the same archive bytes and SHA-256.",
            "Runtime tree hashes are compared when both artifacts expose them.",
            "Inflated raw-output hashes diagnose render-device drift but do not by themselves block paired score custody.",
        ],
    }


def _load_ledger_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported ledger shape: {path}")


def _public_pr_inputs(ledger: Path, pr: int) -> tuple[Path, Path, str, dict[str, Any]]:
    for row in _load_ledger_rows(ledger):
        if row.get("pr") != pr:
            continue
        archive_path = (row.get("archive") or {}).get("path")
        inflate_path = (((row.get("source") or {}).get("key_files") or {}).get("inflate_sh") or {}).get("path")
        if not archive_path:
            raise ValueError(f"PR {pr} row has no archive path")
        if not inflate_path:
            raise ValueError(f"PR {pr} row has no inflate.sh path")
        label = f"public-pr{pr}-{row.get('leaderboard_name') or row.get('title') or 'submission'}"
        return Path(archive_path), Path(inflate_path), label, row
    raise KeyError(f"PR {pr} not found in {ledger}")


def _command(
    *,
    archive: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    work_dir: Path,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> list[str]:
    return [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive),
        "--inflate-sh",
        str(inflate_sh),
        "--upstream-dir",
        str(upstream_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        device,
        "--work-dir",
        str(work_dir),
        "--inflate-timeout",
        str(inflate_timeout),
        "--evaluate-timeout",
        str(evaluate_timeout),
        "--keep-work-dir",
    ]


def build_plan(
    *,
    archive: Path,
    inflate_sh: Path,
    label: str,
    repo_root: Path,
    run_id: str | None,
    output_root: Path,
    upstream_dir: Path,
    video_names_file: Path,
    inflate_timeout: int,
    evaluate_timeout: int,
    public_pr_row: dict[str, Any] | None = None,
    cpu_artifact_json: Path | None = None,
    cuda_artifact_json: Path | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"{_safe_slug(label)}-dual-auth-{_utc_now_compact()}"
    work_root = output_root / run_id
    archive_meta: dict[str, Any] = {"path": str(archive)}
    if archive.exists():
        archive_meta.update({"bytes": archive.stat().st_size, "sha256": _sha256(archive)})
    input_closure = _input_closure(
        {
            "archive": archive,
            "inflate_sh": inflate_sh,
            "upstream_dir": upstream_dir,
            "video_names_file": video_names_file,
        }
    )

    evals: dict[str, dict[str, Any]] = {}
    for device in ("cuda", "cpu"):
        semantics = (
            "contest_cuda_exact_auth_eval_promotion_axis"
            if device == "cuda"
            else "public_leaderboard_cpu_reproduction_axis"
        )
        evals[device] = {
            "device": device,
            "work_dir": str(work_root / device),
            "command": _command(
                archive=archive,
                inflate_sh=inflate_sh,
                upstream_dir=upstream_dir,
                video_names_file=video_names_file,
                device=device,
                work_dir=work_root / device,
                inflate_timeout=inflate_timeout,
                evaluate_timeout=evaluate_timeout,
            ),
            "evidence_semantics": semantics,
            "promotion_eligible_from_this_axis": device == "cuda",
        }
    return {
        "schema": "dual_device_auth_eval_plan.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "run_id": run_id,
        "label": label,
        "archive": archive_meta,
        "inflate_sh": str(inflate_sh),
        "upstream_dir": str(upstream_dir),
        "video_names_file": str(video_names_file),
        "input_closure": input_closure,
        "evals": evals,
        "dual_axis_completion": _dual_axis_completion(
            archive_meta=archive_meta,
            cpu_artifact_json=cpu_artifact_json,
            cuda_artifact_json=cuda_artifact_json,
        ),
        "public_pr_row": public_pr_row,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "notes": [
            "Run both commands on the same archive bytes before PR/frontier claims.",
            "CUDA is the internal promotion/ranking axis; CPU is the public leaderboard reproduction axis.",
            "Do not extrapolate CPU from CUDA or CUDA from CPU; compare paired JSON artifacts only.",
            "Do not assume CPU or CUDA is universally better; treat the CPU/CUDA gap as per-submission and per-runtime.",
            "Use inflated_outputs_manifest aggregate hashes to separate render-device drift from scorer-device drift.",
            "Use dual_axis_completion.paired_score_artifacts_complete as the machine-checkable pair-completion guard.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--public-pr", type=int, help="Resolve archive/runtime from the PR100-107 ledger.")
    source.add_argument("--archive", type=Path, help="Archive path for an explicit submission.")
    parser.add_argument("--inflate-sh", type=Path, help="inflate.sh for --archive mode.")
    parser.add_argument("--label", help="Human-readable run label.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/dual_device_auth_eval"))
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--video-names-file", type=Path, default=Path("upstream/public_test_video_names.txt"))
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--evaluate-timeout", type=int, default=1800)
    parser.add_argument("--cpu-artifact-json", type=Path, help="Existing contest-CPU auth-eval JSON for pair-completion checks.")
    parser.add_argument("--cuda-artifact-json", type=Path, help="Existing contest-CUDA auth-eval JSON for pair-completion checks.")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--execute", choices=("cpu", "cuda", "both"), help="Run planned local command(s).")
    args = parser.parse_args()

    public_pr_row = None
    if args.public_pr is not None:
        archive, inflate_sh, label, public_pr_row = _public_pr_inputs(args.ledger, args.public_pr)
    else:
        if args.inflate_sh is None:
            raise SystemExit("--inflate-sh is required with --archive")
        archive = args.archive
        inflate_sh = args.inflate_sh
        label = args.label or archive.stem

    plan = build_plan(
        archive=archive,
        inflate_sh=inflate_sh,
        label=label,
        repo_root=args.repo_root,
        run_id=args.run_id,
        output_root=args.output_root,
        upstream_dir=args.upstream_dir,
        video_names_file=args.video_names_file,
        inflate_timeout=args.inflate_timeout,
        evaluate_timeout=args.evaluate_timeout,
        public_pr_row=public_pr_row,
        cpu_artifact_json=args.cpu_artifact_json,
        cuda_artifact_json=args.cuda_artifact_json,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.execute:
        if not plan["input_closure"]["ready_to_execute"]:
            missing = ", ".join(plan["input_closure"]["missing_inputs"])
            raise SystemExit(f"refusing to execute dual-device auth eval plan with missing inputs: {missing}")
        devices = ("cpu", "cuda") if args.execute == "both" else (args.execute,)
        for device in devices:
            result = subprocess.run(plan["evals"][device]["command"], cwd=args.repo_root)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
