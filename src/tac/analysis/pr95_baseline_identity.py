# SPDX-License-Identifier: MIT
"""False-authority PR95 baseline identity intake.

This helper turns existing PR95/HNeRV public-archive reports and receiver
proofs into a reusable control-arm packet for SNeRV/HiNeRV work. It does not
run auth eval and never grants score authority. Baseline/control-arm validation
stays local CPU plus MLX; Modal/provider exact auth eval is reserved for
frontier-qualified candidate promotion only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tac.auth_eval_result import parse_auth_eval_score_claim

SCHEMA = "pr95_baseline_identity.v1"
ARTIFACT_SCHEMA = "pr95_baseline_identity_source_artifact.v1"
ARCHIVE_RECORD_SCHEMA = "pr95_baseline_identity_archive_record.v1"
EXACT_AXIS_STATUS_SCHEMA = "pr95_baseline_identity_exact_axis_status.v1"
LOCAL_CPU_MLX_WORK_ORDER_SCHEMA = "pr95_baseline_local_cpu_mlx_work_order.v1"
MODAL_DISPATCH_POLICY_SCHEMA = "pr95_baseline_modal_dispatch_policy.v1"
CONTROL_ARM_ID = "pr95_public_hnerv_muon_control_arm"
AUTHORITY = "false_authority_pr95_baseline_identity_no_score_claim"

FALSE_AUTHORITY: dict[str, Any] = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

_ONLY_EXACT_EVAL_MISSING = {
    "contest_cpu_cuda_exact_eval_missing",
    "contest_cpu_exact_eval_missing",
    "contest_cuda_exact_eval_missing",
}


class Pr95BaselineIdentityError(ValueError):
    """Raised when PR95 baseline identity input is malformed."""


def build_pr95_baseline_identity(
    *,
    source_artifacts: Sequence[str | Path],
    output_root: str | Path = "/Volumes/VertigoDataTier/pact/pr95_baseline_identity",
) -> dict[str, Any]:
    """Build a reusable PR95 control-arm identity packet.

    ``source_artifacts`` may include PR95 Stage-8 lane reports, receiver/runtime
    proof reports, direct archive ZIP paths, and auth-eval JSON artifacts. The
    returned packet is deterministic and false-authority.
    """

    artifacts = [_artifact_record(Path(path)) for path in source_artifacts]
    payloads = [
        (Path(record["path"]), _read_json(Path(record["path"])))
        for record in artifacts
        if record["exists"] and str(record["kind"]) == "json"
    ]
    source_archive = _select_source_archive(payloads)
    candidate_archives = _candidate_archive_records(payloads, artifacts)
    selected_archive = _select_reusable_candidate_archive(candidate_archives)
    exact_axis_status = _exact_axis_status(
        payloads=payloads,
        selected_archive_sha256=(
            str(selected_archive.get("sha256") or "") if selected_archive else ""
        ),
    )
    baseline_reusable = selected_archive is not None
    blockers = _blockers(
        source_archive=source_archive,
        selected_archive=selected_archive,
        exact_axis_status=exact_axis_status,
    )
    local_cpu_mlx_work_order = _local_cpu_mlx_work_order(
        selected_archive=selected_archive,
        runtime_root=_select_runtime_root(payloads),
        output_root=Path(output_root),
    )
    modal_policy = _modal_dispatch_policy()
    paired_work_order = _paired_exact_eval_work_order(
        selected_archive=selected_archive,
        runtime_root=_select_runtime_root(payloads),
        output_root=Path(output_root),
        modal_policy=modal_policy,
    )
    return {
        "schema": SCHEMA,
        "baseline_id": CONTROL_ARM_ID,
        "authority": AUTHORITY,
        "role": "control_arm_for_hi_nerv_snerv_pr95_or_better_work",
        "source_artifact_count": len(artifacts),
        "source_artifacts": artifacts,
        "source_archive": source_archive,
        "candidate_archives": candidate_archives,
        "candidate_archive_count": len(candidate_archives),
        "selected_reusable_candidate_archive": selected_archive,
        "baseline_identity_reusable": bool(baseline_reusable),
        "reusable_for": [
            "pr95_control_arm_identity",
            "aurora_like_optimizer_timing_smoke_anchor",
            "hinerv_snerv_exact_eval_dispatch_candidate_selection",
        ],
        "not_reusable_for": [
            "score_claim",
            "rank_or_kill",
            "promotion",
            "cpu_cuda_axis_collapse",
        ],
        "exact_axis_status": exact_axis_status,
        "local_cpu_mlx_work_order": local_cpu_mlx_work_order,
        "modal_dispatch_policy": modal_policy,
        "paired_exact_eval_work_order": paired_work_order,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def render_pr95_baseline_identity_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing PR95 baseline identity summary."""

    selected = report.get("selected_reusable_candidate_archive")
    selected_line = "none"
    if isinstance(selected, Mapping):
        selected_line = (
            f"`{selected.get('path')}` ({selected.get('bytes')} bytes, "
            f"sha `{str(selected.get('sha256') or '')[:12]}`)"
        )
    lines = [
        "# PR95 Baseline Identity",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Reusable identity: `{report.get('baseline_identity_reusable')}`",
        f"Score claim: `{report.get('score_claim')}`",
        f"Selected archive: {selected_line}",
        "",
        "## Local Work Order",
        "",
    ]
    local = report.get("local_cpu_mlx_work_order")
    if isinstance(local, Mapping):
        lines.extend(
            [
                f"- ready: `{local.get('ready')}`",
                f"- local_cpu_axis: `{local.get('local_cpu_axis_tag')}`",
                f"- mlx_axis: `{local.get('mlx_axis_tag')}`",
                f"- blockers: `{len(local.get('blockers') or [])}`",
            ]
        )
    modal = report.get("modal_dispatch_policy")
    if isinstance(modal, Mapping):
        lines.extend(
            [
                f"- modal_allowed: `{modal.get('modal_dispatch_allowed')}`",
                f"- modal_reason: `{modal.get('reason')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Exact Axis Status",
            "",
        ]
    )
    exact = report.get("exact_axis_status")
    if isinstance(exact, Mapping):
        for axis in ("contest_cpu", "contest_cuda"):
            row = exact.get(axis)
            if isinstance(row, Mapping):
                lines.append(
                    f"- `{axis}` present: `{row.get('present')}` "
                    f"blockers: `{len(row.get('blockers') or [])}`"
                )
    lines.extend(["", "## Blockers", ""])
    blockers = list(report.get("blockers") or ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _artifact_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    suffix = path.suffix.lower()
    return {
        "schema": ARTIFACT_SCHEMA,
        "path": path.as_posix(),
        "exists": exists,
        "kind": "json" if suffix == ".json" else "zip" if suffix == ".zip" else "file",
        "bytes": path.stat().st_size if exists else None,
        "sha256": _sha256_file(path) if exists else None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Pr95BaselineIdentityError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Pr95BaselineIdentityError(f"JSON artifact must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _select_source_archive(payloads: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, Any] | None:
    for source_path, payload in payloads:
        archive_path = _path_or_none(payload.get("source_archive_zip"))
        if archive_path is None:
            continue
        return _archive_record(
            archive_path=archive_path,
            source_artifact=source_path,
            declared_sha256=payload.get("source_archive_zip_sha256"),
            role="public_pr95_source_archive",
            blockers=[],
            score_axis=payload.get("score_axis"),
            score_authority=payload.get("score_authority"),
        )
    return None


def _candidate_archive_records(
    payloads: Sequence[tuple[Path, Mapping[str, Any]]],
    artifacts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_path, payload in payloads:
        rows.extend(_candidate_archives_from_stage8_report(source_path, payload))
        rows.extend(_candidate_archives_from_receiver_proof(source_path, payload))
    for artifact in artifacts:
        if artifact.get("kind") == "zip":
            rows.append(
                _archive_record(
                    archive_path=Path(str(artifact["path"])),
                    source_artifact=Path(str(artifact["path"])),
                    declared_sha256=artifact.get("sha256"),
                    role="direct_archive_zip_input",
                    blockers=["direct_archive_missing_runtime_context"],
                )
            )
    return _dedupe_archives(rows)


def _candidate_archives_from_stage8_report(
    source_path: Path,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if payload.get("schema") != "pr95_stage8_from_public_archive_lane.v1":
        return []
    archive_path = _path_or_none(payload.get("candidate_archive_zip_path"))
    if archive_path is None:
        return []
    gate = payload.get("exact_gate")
    gate_blockers = (
        [str(blocker) for blocker in gate.get("blockers") or () if blocker]
        if isinstance(gate, Mapping)
        else []
    )
    return [
        _archive_record(
            archive_path=archive_path,
            source_artifact=source_path,
            declared_bytes=payload.get("candidate_archive_zip_bytes"),
            declared_sha256=payload.get("candidate_archive_zip_sha256"),
            role="stage8_public_archive_candidate",
            blockers=gate_blockers,
            score_axis=payload.get("score_axis"),
            score_authority=payload.get("score_authority"),
            exact_gate=gate if isinstance(gate, Mapping) else None,
            runtime_root=_runtime_root_from_stage8_report(payload),
        )
    ]


def _candidate_archives_from_receiver_proof(
    source_path: Path,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    schema = str(payload.get("schema") or "")
    if schema not in {
        "pr95_hnerv_receiver_proof.v1",
        "pr95_hnerv_public_runtime_consumption_proof.v1",
    }:
        return []
    archive_path = _path_or_none(
        payload.get("archive_zip_path") or payload.get("archive_path")
    )
    if archive_path is None:
        return []
    blockers = [str(blocker) for blocker in payload.get("blockers") or () if blocker]
    blockers.append("receiver_proof_archive_not_stage8_public_archive_candidate")
    if payload.get("runtime_consumption_proven") is not True and schema.endswith(
        "runtime_consumption_proof.v1"
    ):
        blockers.append("runtime_consumption_not_proven")
    return [
        _archive_record(
            archive_path=archive_path,
            source_artifact=source_path,
            declared_bytes=payload.get("archive_bytes"),
            declared_sha256=payload.get("archive_sha256"),
            role="receiver_runtime_consumption_identity",
            blockers=blockers,
            runtime_consumption_proven=payload.get("runtime_consumption_proven"),
        )
    ]


def _archive_record(
    *,
    archive_path: Path,
    source_artifact: Path,
    role: str,
    blockers: Sequence[str],
    declared_bytes: Any = None,
    declared_sha256: Any = None,
    score_axis: Any = None,
    score_authority: Any = None,
    exact_gate: Mapping[str, Any] | None = None,
    runtime_root: str | None = None,
    runtime_consumption_proven: Any = None,
) -> dict[str, Any]:
    exists = archive_path.is_file()
    observed_bytes = archive_path.stat().st_size if exists else None
    observed_sha = _sha256_file(archive_path) if exists else None
    declared_sha = str(declared_sha256 or "") or None
    declared_size = _int_or_none(declared_bytes)
    integrity_blockers = []
    if not exists:
        integrity_blockers.append("archive_path_missing")
    if declared_size is not None and observed_bytes is not None and declared_size != observed_bytes:
        integrity_blockers.append("declared_archive_bytes_mismatch")
    if declared_sha and observed_sha and declared_sha.lower() != observed_sha.lower():
        integrity_blockers.append("declared_archive_sha256_mismatch")
    all_blockers = _dedupe([*blockers, *integrity_blockers])
    reusable = exists and not [
        blocker for blocker in all_blockers if blocker not in _ONLY_EXACT_EVAL_MISSING
    ]
    return {
        "schema": ARCHIVE_RECORD_SCHEMA,
        "role": role,
        "path": archive_path.as_posix(),
        "exists": exists,
        "bytes": observed_bytes if observed_bytes is not None else declared_size,
        "sha256": observed_sha or declared_sha,
        "declared_bytes": declared_size,
        "declared_sha256": declared_sha,
        "source_artifact": source_artifact.as_posix(),
        "score_axis": score_axis,
        "score_authority": score_authority,
        "runtime_root": runtime_root,
        "runtime_consumption_proven": runtime_consumption_proven,
        "exact_gate": dict(exact_gate) if isinstance(exact_gate, Mapping) else None,
        "reusable_identity": bool(reusable),
        "blockers": all_blockers,
        **FALSE_AUTHORITY,
    }


def _select_reusable_candidate_archive(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    reusable = [dict(row) for row in rows if row.get("reusable_identity") is True]
    if not reusable:
        return None

    def key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        blockers = [str(blocker) for blocker in row.get("blockers") or ()]
        non_exact = [blocker for blocker in blockers if blocker not in _ONLY_EXACT_EVAL_MISSING]
        exact_only_count = len([blocker for blocker in blockers if blocker in _ONLY_EXACT_EVAL_MISSING])
        return (
            len(non_exact),
            exact_only_count,
            int(row.get("bytes") or 1 << 60),
            str(row.get("path") or ""),
        )

    return sorted(reusable, key=key)[0]


def _exact_axis_status(
    *,
    payloads: Sequence[tuple[Path, Mapping[str, Any]]],
    selected_archive_sha256: str,
) -> dict[str, Any]:
    axes = {
        "contest_cpu": _axis_row("contest_cpu"),
        "contest_cuda": _axis_row("contest_cuda"),
    }
    for source_path, payload in payloads:
        for axis, required in (("contest_cpu", "contest_cpu"), ("contest_cuda", "contest_cuda")):
            claim = parse_auth_eval_score_claim(
                payload,
                required_score_axis=required,
                require_component_recompute=False,
            )
            if claim is None:
                continue
            nested_archive = payload.get("archive")
            nested_archive = nested_archive if isinstance(nested_archive, Mapping) else {}
            archive_sha = str(
                payload.get("archive_sha256")
                or payload.get("archive_zip_sha256")
                or nested_archive.get("sha256")
                or ""
            )
            if selected_archive_sha256 and archive_sha and archive_sha != selected_archive_sha256:
                continue
            axes[axis].update(
                {
                    "present": True,
                    "source_artifact": source_path.as_posix(),
                    "score": claim.score,
                    "archive_sha256": archive_sha or selected_archive_sha256,
                    "blockers": [],
                }
            )
    return {
        "schema": EXACT_AXIS_STATUS_SCHEMA,
        **axes,
        **FALSE_AUTHORITY,
    }


def _axis_row(axis: str) -> dict[str, Any]:
    return {
        "axis": axis,
        "present": False,
        "source_artifact": None,
        "score": None,
        "archive_sha256": None,
        "blockers": [f"pr95_{axis}_exact_eval_missing"],
        **FALSE_AUTHORITY,
    }


def _local_cpu_mlx_work_order(
    *,
    selected_archive: Mapping[str, Any] | None,
    runtime_root: str | None,
    output_root: Path,
) -> dict[str, Any]:
    if not isinstance(selected_archive, Mapping):
        return {
            "schema": LOCAL_CPU_MLX_WORK_ORDER_SCHEMA,
            "ready": False,
            "blockers": ["pr95_reusable_candidate_archive_missing"],
            **FALSE_AUTHORITY,
        }
    blockers = []
    local_cpu_out = (
        output_root / "local_cpu_advisory" / "contest_auth_eval_cpu_advisory.json"
    )
    local_cpu_command = [
        "uv",
        "run",
        "python",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(selected_archive["path"]),
        "--upstream-dir",
        "upstream",
        "--device",
        "cpu",
        "--json-out",
        local_cpu_out.as_posix(),
    ]
    if runtime_root:
        local_cpu_command.extend(
            ["--inflate-sh", (Path(runtime_root) / "inflate.sh").as_posix()]
        )
    else:
        blockers.append("pr95_runtime_root_missing_for_local_cpu_replay")
    return {
        "schema": LOCAL_CPU_MLX_WORK_ORDER_SCHEMA,
        "ready": not blockers,
        "local_cpu_axis_tag": "[macOS-CPU advisory]",
        "local_cpu_command_argv": local_cpu_command,
        "local_cpu_output_json": local_cpu_out.as_posix(),
        "mlx_axis_tag": "[macOS-MLX research-signal]",
        "mlx_next_action": (
            "Bind PR95/HNeRV MLX-vs-PyTorch parity or scorer-response artifacts "
            "as research signal only before local CPU replay spend triage."
        ),
        "modal_dispatch_allowed": False,
        "blockers": blockers,
        "plan_only_default": True,
        **FALSE_AUTHORITY,
    }


def _modal_dispatch_policy() -> dict[str, Any]:
    return {
        "schema": MODAL_DISPATCH_POLICY_SCHEMA,
        "modal_dispatch_allowed": False,
        "reason": "non_frontier_control_arm_modal_dispatch_forbidden",
        "allowed_only_for": "frontier_candidate_exact_auth_eval_after_local_cpu_mlx_gates",
        "baseline_control_arm_policy": "local_cpu_and_mlx_only",
        "forbidden_work_order_blocker": "modal_reserved_for_frontier_candidates",
        **FALSE_AUTHORITY,
    }


def _paired_exact_eval_work_order(
    *,
    selected_archive: Mapping[str, Any] | None,
    runtime_root: str | None,
    output_root: Path,
    modal_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(selected_archive, Mapping):
        blockers = ["pr95_reusable_candidate_archive_missing"]
    else:
        blockers = [str(modal_policy["forbidden_work_order_blocker"])]
        if not runtime_root:
            blockers.append("pr95_runtime_root_missing_for_paired_exact_eval")
    return {
        "schema": "pr95_baseline_paired_exact_eval_work_order.v1",
        "ready": False,
        "modal_dispatch_allowed": False,
        "reason": modal_policy.get("reason"),
        "allowed_only_for": modal_policy.get("allowed_only_for"),
        "command_argv": [],
        "blockers": blockers,
        "superseded_by": LOCAL_CPU_MLX_WORK_ORDER_SCHEMA,
        "output_root": output_root.as_posix(),
        **FALSE_AUTHORITY,
    }


def _blockers(
    *,
    source_archive: Mapping[str, Any] | None,
    selected_archive: Mapping[str, Any] | None,
    exact_axis_status: Mapping[str, Any],
) -> list[str]:
    out: list[str] = []
    if source_archive is None:
        out.append("pr95_source_archive_identity_missing")
    if selected_archive is None:
        out.append("pr95_reusable_candidate_archive_missing")
    for axis in ("contest_cpu", "contest_cuda"):
        row = exact_axis_status.get(axis)
        if isinstance(row, Mapping):
            out.extend(str(blocker) for blocker in row.get("blockers") or ())
    return _dedupe(out)


def _runtime_root_from_stage8_report(payload: Mapping[str, Any]) -> str | None:
    repro = payload.get("reproducibility")
    if not isinstance(repro, Mapping):
        return None
    argv = repro.get("argv_template")
    if not isinstance(argv, list):
        return None
    for idx, value in enumerate(argv):
        if value == "--public-submission-root" and idx + 1 < len(argv):
            return str(argv[idx + 1])
    return None


def _select_runtime_root(payloads: Sequence[tuple[Path, Mapping[str, Any]]]) -> str | None:
    for _, payload in payloads:
        runtime_root = _runtime_root_from_stage8_report(payload)
        if runtime_root and Path(runtime_root).is_dir():
            return runtime_root
    return None


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value)


def _int_or_none(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_archives(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("path") or ""), str(row.get("sha256") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
    return out


__all__ = [
    "AUTHORITY",
    "CONTROL_ARM_ID",
    "FALSE_AUTHORITY",
    "SCHEMA",
    "Pr95BaselineIdentityError",
    "build_pr95_baseline_identity",
    "render_pr95_baseline_identity_markdown",
]
