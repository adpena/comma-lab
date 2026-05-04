#!/usr/bin/env python3
"""Prepare a local handoff plan for trained-renderer transplants.

This is a deterministic dry-run planner. It does not build archives, contact
remote GPU providers, or dispatch jobs. It verifies the source archive custody
record, checks whether a recovered renderer export exists, computes the
byte-only break-even target for sub-0.314, and emits the exact local commands
needed to turn that export into a contest-faithful candidate archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "trained_renderer_transplant_dispatch_prepare_v1"
TOOL = "experiments/prepare_trained_renderer_transplant_dispatch.py"
POSE_SAFETY_SCHEMA = "renderer_transplant_pose_safety_preflight_v1"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
DEFAULT_TARGET_SCORE = 0.314
DEFAULT_LANE_ID = "c091_trained_renderer_self_compression_transplant"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_EVAL_JSON = DEFAULT_SOURCE_ARCHIVE.with_name(
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/trained_renderer_transplant_recovery_worker_20260503/"
    "c091_readiness"
)
DEFAULT_BLOCK_SIZES = (64, 128, 256, 512, 1024)
DEFAULT_ACTIVE_MODAL_CALL_IDS = (
    "fc-01KQP9K42CAWJH7XEV4KC0V28M",
    "fc-01KQP9T1VD14785MG63H7JM5VK",
    "fc-01KQP9T19Y7PMDETDN99WDMF2W",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _cmd(args: Sequence[str | Path]) -> dict[str, Any]:
    as_strings = [str(item) for item in args]
    return {"argv": as_strings, "text": shlex.join(as_strings)}


def _finite_float(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite float") from exc
    if not math.isfinite(out):
        raise ValueError(f"{label} must be finite")
    return out


def _read_eval_score(eval_json: Path) -> dict[str, Any]:
    payload = _load_json(eval_json)
    if not isinstance(payload, Mapping):
        raise ValueError(f"eval JSON must be an object: {eval_json}")
    archive_bytes = int(payload["archive_size_bytes"])
    score = _finite_float(
        payload.get("score_recomputed_from_components", payload.get("final_score")),
        "score",
    )
    seg_dist = _finite_float(payload.get("avg_segnet_dist"), "avg_segnet_dist")
    pose_dist = _finite_float(payload.get("avg_posenet_dist"), "avg_posenet_dist")
    return {
        "archive_size_bytes": archive_bytes,
        "score_recomputed_from_components": score,
        "avg_segnet_dist": seg_dist,
        "avg_posenet_dist": pose_dist,
        "device": (payload.get("provenance") or {}).get("device", payload.get("device")),
        "sample_count": payload.get(
            "num_samples",
            payload.get("n_samples", payload.get("sample_count")),
        ),
    }


def _source_record(
    *,
    source_archive: Path,
    eval_json: Path,
    expected_source_sha256: str | None,
    expected_source_bytes: int | None,
) -> dict[str, Any]:
    if not source_archive.exists():
        raise FileNotFoundError(f"source archive does not exist: {source_archive}")
    if not source_archive.is_file():
        raise ValueError(f"source archive is not a file: {source_archive}")
    if not eval_json.exists():
        raise FileNotFoundError(f"source eval JSON does not exist: {eval_json}")
    actual_bytes = source_archive.stat().st_size
    actual_sha = _sha256_file(source_archive)
    score = _read_eval_score(eval_json)
    if score["archive_size_bytes"] != actual_bytes:
        raise ValueError(
            "source archive bytes do not match eval JSON: "
            f"{actual_bytes} != {score['archive_size_bytes']}"
        )
    if expected_source_bytes is not None and expected_source_bytes != actual_bytes:
        raise ValueError(
            "source archive bytes do not match expected custody record: "
            f"{actual_bytes} != {expected_source_bytes}"
        )
    if expected_source_sha256 is not None and expected_source_sha256 != actual_sha:
        raise ValueError(
            "source archive SHA-256 does not match expected custody record: "
            f"{actual_sha} != {expected_source_sha256}"
        )
    return {
        "path": str(source_archive.resolve()),
        "repo_relative_path": _repo_rel(source_archive),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "eval_json": str(eval_json.resolve()),
        "eval_score": score,
        "verified": True,
        "expected_bytes": expected_source_bytes,
        "expected_sha256": expected_source_sha256,
    }


def _renderer_export_record(renderer_export: Path) -> dict[str, Any]:
    if not renderer_export.exists():
        raise FileNotFoundError(f"renderer export does not exist: {renderer_export}")
    if not renderer_export.is_file():
        raise ValueError(f"renderer export is not a file: {renderer_export}")
    size = renderer_export.stat().st_size
    if size <= 0:
        raise ValueError(f"renderer export is empty: {renderer_export}")
    return {
        "path": str(renderer_export.resolve()),
        "repo_relative_path": _repo_rel(renderer_export),
        "bytes": size,
        "sha256": _sha256_file(renderer_export),
        "magic4_hex": renderer_export.read_bytes()[:4].hex(),
        "exists": True,
    }


def _missing_renderer_export_record(renderer_export: Path | None) -> dict[str, Any]:
    return {
        "path": str(renderer_export.resolve()) if renderer_export is not None else None,
        "repo_relative_path": _repo_rel(renderer_export) if renderer_export is not None else None,
        "bytes": None,
        "sha256": None,
        "magic4_hex": None,
        "exists": False,
        "missing": True,
        "blockers": [
            "no terminal Modal renderer export is available locally",
            "recover the active Modal call, then provide the recovered QZS3 renderer export path",
        ],
    }


def _break_even(
    *,
    source_bytes: int,
    current_score: float,
    target_score: float,
) -> dict[str, Any]:
    gap = current_score - target_score
    bytes_to_save = 0
    if gap >= 0:
        bytes_to_save = int(math.floor(gap / RATE_SCORE_PER_BYTE)) + 1
    target_archive_bytes = source_bytes - bytes_to_save
    return {
        "target_score_strictly_below": target_score,
        "current_score": current_score,
        "score_gap": gap,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "bytes_to_save_at_unchanged_distortion": bytes_to_save,
        "max_archive_bytes_for_byte_only_crossing": target_archive_bytes,
        "score_at_max_archive_bytes": current_score
        - bytes_to_save * RATE_SCORE_PER_BYTE,
    }


def _candidate_from_preflight_summary(
    preflight_summary: Path,
) -> dict[str, Any] | None:
    if not preflight_summary.exists():
        return None
    payload = _load_json(preflight_summary)
    if not isinstance(payload, Mapping):
        raise ValueError(f"preflight summary must be an object: {preflight_summary}")
    selected = payload.get("best_by_archive_bytes")
    if not isinstance(selected, Mapping):
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        selected = min(
            (item for item in candidates if isinstance(item, Mapping)),
            key=lambda item: (int(item["archive_bytes"]), str(item["candidate_id"])),
        )
    archive_raw = selected.get("archive_path", selected.get("archive"))
    if archive_raw is None:
        return None
    archive_path = Path(str(archive_raw))
    if not archive_path.is_absolute():
        archive_path = (REPO_ROOT / archive_path).resolve()
    manifest_path = selected.get("manifest_path", selected.get("manifest"))
    record = {
        "candidate_id": str(selected["candidate_id"]),
        "archive_path": str(archive_path),
        "repo_relative_archive_path": _repo_rel(archive_path),
        "archive_bytes": int(selected["archive_bytes"]),
        "archive_sha256": str(selected["archive_sha256"]),
        "manifest_path": manifest_path,
        "preflight_summary_path": str(preflight_summary.resolve()),
        "preflight_summary_schema": payload.get("schema"),
        "exists": archive_path.exists(),
    }
    if archive_path.exists():
        actual_bytes = archive_path.stat().st_size
        actual_sha = _sha256_file(archive_path)
        record.update(
            {
                "actual_archive_bytes": actual_bytes,
                "actual_archive_sha256": actual_sha,
                "archive_matches_preflight_summary": (
                    actual_bytes == record["archive_bytes"]
                    and actual_sha == record["archive_sha256"]
                ),
            }
        )
    else:
        record["archive_matches_preflight_summary"] = False
    return record


def _pose_safety_gate(
    *,
    pose_safety_json: Sequence[Path],
    source_sha256: str,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if candidate is None:
        return {
            "required": True,
            "status": "waiting_for_candidate_archive",
            "safe_for_exact_eval_dispatch": False,
            "matching_report_path": None,
            "blockers": ["candidate archive is not available yet"],
        }
    candidate_sha = str(candidate["archive_sha256"])
    reports: list[dict[str, Any]] = []
    for path in pose_safety_json:
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"pose-safety JSON must be an object: {path}")
        reports.append({"path": str(path.resolve()), "payload": dict(payload)})
    matching = []
    for report in reports:
        payload = report["payload"]
        source = payload.get("source_archive") or {}
        target = payload.get("candidate_archive") or {}
        if source.get("sha256") == source_sha256 and target.get("sha256") == candidate_sha:
            matching.append(report)
    if not matching:
        return {
            "required": True,
            "status": "missing_pose_safety_json",
            "safe_for_exact_eval_dispatch": False,
            "matching_report_path": None,
            "checked_report_paths": [report["path"] for report in reports],
            "blockers": [
                "missing pose-safety JSON for exact source and candidate archive SHA pair"
            ],
        }
    report = sorted(matching, key=lambda item: item["path"])[-1]
    payload = report["payload"]
    blockers: list[str] = []
    if payload.get("schema") != POSE_SAFETY_SCHEMA:
        blockers.append("pose-safety JSON schema mismatch")
    if payload.get("score_claim") is not False:
        blockers.append("pose-safety JSON must be no-score evidence")
    if payload.get("promotion_eligible") is not False:
        blockers.append("pose-safety JSON must not claim promotion")
    if payload.get("remote_gpu_dispatch_performed") is not False:
        blockers.append("pose-safety JSON must be local-only")
    if payload.get("safe_for_exact_eval_dispatch") is not True:
        blockers.extend(
            str(item)
            for item in (
                payload.get("fail_closed_reasons")
                or ["pose-safety JSON failed closed"]
            )
        )
    return {
        "required": True,
        "status": "pass" if not blockers else "failed",
        "safe_for_exact_eval_dispatch": not blockers,
        "matching_report_path": report["path"],
        "failure_class": payload.get("failure_class"),
        "fail_closed_reasons": payload.get("fail_closed_reasons") or [],
        "blockers": sorted(set(blockers)),
    }


def _preflight_command(
    *,
    source_archive: Path,
    renderer_export: Path | None,
    preflight_output_dir: Path,
    block_sizes: Sequence[int],
    force_preflight: bool,
) -> dict[str, Any]:
    renderer_export_arg: str | Path = (
        _repo_rel(renderer_export)
        if renderer_export is not None
        else "<recovered_renderer_qzs3.bin>"
    )
    args: list[str | Path] = [
        ".venv/bin/python",
        "experiments/build_renderer_shrink_candidate.py",
        "--source-archive",
        _repo_rel(source_archive),
        "--renderer-export",
        renderer_export_arg,
        "--output-dir",
        _repo_rel(preflight_output_dir),
        "--qzs3-block-sizes",
        ",".join(str(item) for item in block_sizes),
    ]
    if force_preflight:
        args.append("--force")
    return _cmd(args)


def _pose_safety_command(
    *,
    source_archive: Path,
    output_dir: Path,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = (
        str(candidate["candidate_id"]) if candidate is not None else "${CANDIDATE_ID}"
    )
    candidate_archive = (
        str(candidate["repo_relative_archive_path"])
        if candidate is not None
        else "${CANDIDATE_ARCHIVE}"
    )
    output_json = output_dir / "pose_safety" / f"{candidate_id}.json"
    return _cmd(
        [
            ".venv/bin/python",
            "experiments/preflight_renderer_transplant_pose_safety.py",
            "--source-archive",
            _repo_rel(source_archive),
            "--candidate-archive",
            candidate_archive,
            "--output-json",
            _repo_rel(output_json),
        ]
    )


def _dispatch_commands(
    *,
    lane_id: str,
    source_sha256: str,
    baseline_score: float,
    baseline_archive_bytes: int,
    renderer_sha256: str | None,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = (
        str(candidate["candidate_id"]) if candidate is not None else "${CANDIDATE_ID}"
    )
    archive_path = (
        str(candidate["repo_relative_archive_path"])
        if candidate is not None
        else "${CANDIDATE_ARCHIVE}"
    )
    archive_sha = (
        str(candidate["archive_sha256"]) if candidate is not None else "${ARCHIVE_SHA256}"
    )
    renderer_sha = renderer_sha256 or "${RENDERER_SHA256}"
    job_name = f"exact_eval_{candidate_id}_trained_transplant_READY"
    claim = _cmd(
        [
            ".venv/bin/python",
            "tools/claim_lane_dispatch.py",
            "claim",
            "--lane-id",
            lane_id,
            "--platform",
            "lightning",
            "--instance-job-id",
            job_name,
            "--agent",
            "codex:gpt-5",
            "--status",
            "eval",
            "--notes",
            (
                "trained_renderer_transplant "
                f"candidate={candidate_id} archive_sha256={archive_sha}"
            ),
        ]
    )
    exact_eval_base: list[str | Path] = [
        ".venv/bin/python",
        "scripts/launch_lightning_batch_job.py",
        "exact-eval",
        "--job-name",
        job_name,
        "--archive",
        archive_path,
        "--repo-dir",
        "/teamspace/studios/this_studio/pact",
        "--upstream-dir",
        "/teamspace/studios/this_studio/pact/upstream",
        "--machine",
        "g7e.4xlarge",
        "--adjudicate",
        "--baseline-score",
        f"{baseline_score:.17g}",
        "--baseline-archive-bytes",
        str(int(baseline_archive_bytes)),
        "--predicted-band",
        "0.0",
        "10.0",
        "--regression-threshold",
        "10.0",
        "--infer-expected-archive",
        "--dispatch-lane-id",
        lane_id,
        "--queue-metadata",
        f"lane_id={lane_id}",
        "--queue-metadata",
        f"candidate_id={candidate_id}",
        "--queue-metadata",
        f"source_archive_sha256={source_sha256}",
        "--queue-metadata",
        f"trained_renderer_sha256={renderer_sha}",
        "--queue-metadata",
        "purpose=trained_renderer_transplant_dispatch",
        "--component-trace",
        "--component-trace-top-k",
        "80",
        "--max-sane-score",
        "10.0",
    ]
    return {
        "dispatch_claim_command": claim,
        "lightning_exact_eval_dry_run_command": _cmd([*exact_eval_base, "--dry-run"]),
        "lightning_exact_eval_submit_command_shape_after_claim": _cmd(
            [
                *exact_eval_base,
                "--studio",
                "${LIGHTNING_STUDIO}",
                "--source-manifest",
                "${LIGHTNING_SOURCE_MANIFEST_JSON}",
                "--remote-preflight-ssh-target",
                "${LIGHTNING_PREFLIGHT_SSH_TARGET}",
            ]
        ),
    }


def _recover_commands(call_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [
        _cmd(
            [
                ".venv/bin/python",
                "experiments/modal_recover_lane.py",
                "--call-id",
                call_id,
            ]
        )
        for call_id in call_ids
    ]


def build_plan(
    *,
    renderer_export: Path | None,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    eval_json: Path = DEFAULT_EVAL_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    preflight_summary: Path | None = None,
    expected_source_sha256: str | None = None,
    expected_source_bytes: int | None = 276_481,
    target_score: float = DEFAULT_TARGET_SCORE,
    block_sizes: Sequence[int] = DEFAULT_BLOCK_SIZES,
    pose_safety_json: Sequence[Path] = (),
    modal_call_ids: Sequence[str] = DEFAULT_ACTIVE_MODAL_CALL_IDS,
    lane_id: str = DEFAULT_LANE_ID,
    force_preflight: bool = False,
    allow_missing_renderer_export: bool = False,
) -> dict[str, Any]:
    """Build a deterministic dry-run handoff manifest."""

    source_archive = source_archive.resolve()
    eval_json = eval_json.resolve()
    renderer_export = renderer_export.resolve() if renderer_export is not None else None
    output_dir = output_dir.resolve()
    preflight_output_dir = output_dir / "preflight"
    if preflight_summary is None:
        preflight_summary = preflight_output_dir / "summary.json"
    else:
        preflight_summary = preflight_summary.resolve()

    source = _source_record(
        source_archive=source_archive,
        eval_json=eval_json,
        expected_source_sha256=expected_source_sha256,
        expected_source_bytes=expected_source_bytes,
    )
    if renderer_export is None or not renderer_export.exists():
        if not allow_missing_renderer_export:
            missing = renderer_export or Path("<missing_renderer_export>")
            raise FileNotFoundError(f"renderer export does not exist: {missing}")
        renderer = _missing_renderer_export_record(renderer_export)
    else:
        renderer = _renderer_export_record(renderer_export)
    score = source["eval_score"]
    break_even = _break_even(
        source_bytes=int(source["bytes"]),
        current_score=float(score["score_recomputed_from_components"]),
        target_score=target_score,
    )
    candidate = _candidate_from_preflight_summary(preflight_summary)
    if candidate is not None:
        bytes_saved = int(source["bytes"]) - int(candidate["archive_bytes"])
        candidate["bytes_saved_vs_source"] = bytes_saved
        candidate["projected_score_if_distortion_unchanged"] = (
            float(score["score_recomputed_from_components"])
            - bytes_saved * RATE_SCORE_PER_BYTE
        )
        candidate["byte_only_crosses_target"] = (
            candidate["projected_score_if_distortion_unchanged"] < target_score
        )
    pose_gate = _pose_safety_gate(
        pose_safety_json=pose_safety_json,
        source_sha256=str(source["sha256"]),
        candidate=candidate,
    )
    candidate_ready_blockers: list[str] = []
    if not renderer.get("exists"):
        candidate_ready_blockers.extend(str(item) for item in renderer.get("blockers") or [])
    if candidate is None:
        candidate_ready_blockers.append("run transplant preflight to build candidate archive")
    elif not candidate.get("exists"):
        candidate_ready_blockers.append("candidate archive from preflight summary is missing")
    elif not candidate.get("archive_matches_preflight_summary"):
        candidate_ready_blockers.append("candidate archive does not match preflight summary")
    if not pose_gate["safe_for_exact_eval_dispatch"]:
        candidate_ready_blockers.extend(str(item) for item in pose_gate["blockers"])
    exact_eval_ready = not candidate_ready_blockers

    commands = {
        "recover_modal_exports": _recover_commands(modal_call_ids),
        "build_candidate_archives": _preflight_command(
            source_archive=source_archive,
            renderer_export=renderer_export,
            preflight_output_dir=preflight_output_dir,
            block_sizes=block_sizes,
            force_preflight=force_preflight,
        ),
        "run_pose_safety_preflight": _pose_safety_command(
            source_archive=source_archive,
            output_dir=output_dir,
            candidate=candidate,
        ),
        **_dispatch_commands(
            lane_id=lane_id,
            source_sha256=str(source["sha256"]),
            baseline_score=float(score["score_recomputed_from_components"]),
            baseline_archive_bytes=int(source["bytes"]),
            renderer_sha256=renderer.get("sha256"),
            candidate=candidate,
        ),
    }
    missing_prerequisites = sorted(
        set(
            [
                *(str(item) for item in renderer.get("blockers") or []),
                *(str(item) for item in candidate_ready_blockers),
            ]
        )
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "deterministic_dry_run": True,
        "source_archive": source,
        "renderer_export": renderer,
        "terminal_exports_exist": bool(renderer.get("exists")),
        "missing_prerequisites": missing_prerequisites,
        "preflight_summary": {
            "path": str(preflight_summary),
            "exists": preflight_summary.exists(),
        },
        "selected_candidate": candidate,
        "score_break_even": break_even,
        "pose_safety_gate": pose_gate,
        "exact_eval_dispatch_ready": exact_eval_ready,
        "exact_eval_dispatch_blockers": sorted(set(candidate_ready_blockers)),
        "lane_id": lane_id,
        "active_modal_call_ids": list(modal_call_ids),
        "commands": commands,
        "handoff_order": [
            "recover_modal_exports",
            "build_candidate_archives",
            "run_pose_safety_preflight",
            "dispatch_claim_command",
            "lightning_exact_eval_dry_run_command",
            "lightning_exact_eval_submit_command_shape_after_claim",
        ],
        "contest_faithful_constraints": {
            "canonical_eval_path": "archive.zip -> inflate.sh -> upstream/evaluate.py",
            "requires_cuda_auth_eval_for_score": True,
            "requires_dispatch_claim_before_remote_exact_eval": True,
            "requires_pose_safety_json_before_exact_eval_ready": True,
            "hidden_sidecars_allowed": False,
        },
    }


def write_markdown(path: Path, plan: Mapping[str, Any]) -> None:
    score = plan["score_break_even"]
    candidate = plan.get("selected_candidate") or {}
    lines = [
        "# Trained Renderer Transplant Dispatch Handoff - 2026-05-03",
        "",
        "Evidence grade: empirical dry-run planning only. Score claim: false. "
        "Remote dispatch: none.",
        "",
        "## Custody",
        "",
        f"- Source archive: `{plan['source_archive']['repo_relative_path']}`",
        f"- Source bytes: `{plan['source_archive']['bytes']}`",
        f"- Source SHA-256: `{plan['source_archive']['sha256']}`",
        f"- Renderer export: `{plan['renderer_export']['repo_relative_path']}`",
        f"- Renderer export SHA-256: `{plan['renderer_export']['sha256']}`",
        "",
        "## Break-Even",
        "",
        f"- Current score: `{score['current_score']}`",
        f"- Strict target: `< {score['target_score_strictly_below']}`",
        "- Bytes to save at unchanged distortion: "
        f"`{score['bytes_to_save_at_unchanged_distortion']}`",
        "- Max byte-only crossing archive bytes: "
        f"`{score['max_archive_bytes_for_byte_only_crossing']}`",
        "",
        "## Readiness",
        "",
        f"- Exact-eval dispatch ready: `{plan['exact_eval_dispatch_ready']}`",
    ]
    if candidate:
        lines.append(f"- Selected candidate: `{candidate['candidate_id']}`")
        lines.append(f"- Candidate bytes: `{candidate['archive_bytes']}`")
        lines.append(
            "- Byte-only projected score: "
            f"`{candidate['projected_score_if_distortion_unchanged']}`"
        )
    if plan["exact_eval_dispatch_blockers"]:
        lines.append(
            "- Blockers: "
            + "; ".join(f"`{item}`" for item in plan["exact_eval_dispatch_blockers"])
        )
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "```bash",
            *[
                item["text"]
                for item in plan["commands"].get("recover_modal_exports", [])
            ],
            plan["commands"]["build_candidate_archives"]["text"],
            plan["commands"]["run_pose_safety_preflight"]["text"],
            plan["commands"]["dispatch_claim_command"]["text"],
            plan["commands"]["lightning_exact_eval_dry_run_command"]["text"],
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_block_sizes(value: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in value.split(",") if item.strip())
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("block sizes must be positive integers")
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer-export", type=Path)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--preflight-summary", type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-source-bytes", type=int, default=276_481)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--block-sizes", type=parse_block_sizes, default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--pose-safety-json", action="append", type=Path, default=[])
    parser.add_argument("--modal-call-id", action="append", default=[])
    parser.add_argument("--lane-id", default=DEFAULT_LANE_ID)
    parser.add_argument("--force-preflight", action="store_true")
    parser.add_argument(
        "--allow-missing-renderer-export",
        action="store_true",
        help=(
            "Emit a blocked readiness plan when the active Modal run has not "
            "returned a terminal renderer export yet."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_json = args.output_json or output_dir / "handoff_manifest.json"
    output_md = args.output_md or output_dir / "handoff_manifest.md"
    modal_call_ids = tuple(args.modal_call_id) if args.modal_call_id else DEFAULT_ACTIVE_MODAL_CALL_IDS
    try:
        plan = build_plan(
            renderer_export=args.renderer_export,
            source_archive=args.source_archive,
            eval_json=args.eval_json,
            output_dir=output_dir,
            preflight_summary=args.preflight_summary,
            expected_source_sha256=args.expected_source_sha256,
            expected_source_bytes=args.expected_source_bytes,
            target_score=args.target_score,
            block_sizes=args.block_sizes,
            pose_safety_json=tuple(args.pose_safety_json),
            modal_call_ids=modal_call_ids,
            lane_id=args.lane_id,
            force_preflight=args.force_preflight,
            allow_missing_renderer_export=args.allow_missing_renderer_export,
        )
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(_json_bytes(plan))
    write_markdown(output_md, plan)
    print(
        json.dumps(
            {
                "exact_eval_dispatch_ready": plan["exact_eval_dispatch_ready"],
                "output_json": str(output_json.resolve()),
                "output_md": str(output_md.resolve()),
                "remote_gpu_dispatch_performed": False,
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
