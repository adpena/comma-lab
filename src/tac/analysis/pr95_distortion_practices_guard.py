# SPDX-License-Identifier: MIT
"""Source-derived PR95 distortion-practice guard for NeRV launch rows.

This guard is deliberately narrower than the full PR95 stack-binding matrix:
it only checks practices that directly affect scorer-domain distortion and
renderer collapse risk before a HiNeRV/SNeRV local-MLX row is admitted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.pr95_stack_binding_requirements import FALSE_AUTHORITY

SCHEMA = "pr95_distortion_practices_guard.v1"
SOURCE_INVENTORY_SCHEMA = "pr95_distortion_source_inventory.v1"
PRACTICE_ROW_SCHEMA = "pr95_distortion_practice_row.v1"
PAYLOAD_GUARD_SCHEMA = "pr95_distortion_practices_payload_guard.v1"

PR95_SOURCE_REL = Path(
    "experiments/results/public_pr_archive_release_view/"
    "public_pr95_intake_20260505_auto/source/submissions/hnerv_muon"
)
UPSTREAM_REL = Path("upstream")
RUNNER_REL = Path("tools/run_compact_renderer_mlx_spine_runner.py")


@dataclass(frozen=True)
class Practice:
    practice_id: str
    title: str
    why_it_matters: str
    source_check_ids: tuple[str, ...]


PRACTICES: tuple[Practice, ...] = (
    Practice(
        practice_id="official_non_overlapping_seq2_pair_geometry",
        title="official non-overlapping seq_len=2 pair geometry",
        why_it_matters=(
            "upstream evaluate.py scores 600 non-overlapping 2-frame samples; "
            "overlapping pair construction changes PoseNet targets and can hide "
            "collapsed temporal geometry."
        ),
        source_check_ids=(
            "upstream_frame_utils_seq_len_2",
            "upstream_evaluate_seq_len_shape_assert",
            "pr95_score_streams_non_overlapping_pairs",
        ),
    ),
    Practice(
        practice_id="scorer_preprocess_eval_roundtrip_yuv6",
        title="PoseNet YUV6 plus uint8 eval-roundtrip STE",
        why_it_matters=(
            "PR95 trains through the camera-resize/rounding scorer surface and "
            "PoseNet consumes two-frame YUV6; skipping either lets a renderer fit "
            "RGB-looking proxies while collapsing scorer inputs."
        ),
        source_check_ids=(
            "upstream_posenet_uses_yuv6_pair",
            "upstream_rgb_to_yuv6_is_no_grad",
            "pr95_training_eval_roundtrip_ste",
            "runner_hinerv_eval_roundtrip_metadata",
        ),
    ),
    Practice(
        practice_id="dual_component_real_scorer_pressure",
        title="both real SegNet last-frame and PoseNet pair pressure",
        why_it_matters=(
            "upstream SegNet scores only the last frame by hard argmax while "
            "PoseNet scores pair motion; single-component pressure can look "
            "healthy while the other component collapses."
        ),
        source_check_ids=(
            "upstream_segnet_last_frame",
            "upstream_segnet_argmax_distortion",
            "pr95_losses_include_seg_margin_and_pose",
        ),
    ),
    Practice(
        practice_id="pr95_staged_qat_coder_curriculum",
        title="PR95 staged curriculum with QAT and coder pressure",
        why_it_matters=(
            "PR95 did not run one homogeneous proxy loss: it staged CE, margin "
            "losses, QAT, C1a entropy pressure, sigma tightening, and final "
            "optimizer polish so distortion stayed attached to the charged packet."
        ),
        source_check_ids=(
            "pr95_eight_stage_curriculum_present",
            "pr95_qat_and_c1a_present",
            "pr95_stage8_muon_present",
        ),
    ),
)


class PR95DistortionPracticesGuardError(ValueError):
    """Raised when a PR95 distortion guard input is malformed."""


def build_pr95_distortion_source_inventory(
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build the source evidence inventory used by distortion-practice rows."""

    repo = Path(repo_root).expanduser().resolve(strict=False)
    pr95 = repo / PR95_SOURCE_REL
    runner = repo / RUNNER_REL
    records: list[dict[str, Any]] = []
    blockers: list[str] = []

    def add_record(
        *,
        rel_path: Path,
        check_id: str,
        required_tokens: Sequence[str],
    ) -> None:
        path = repo / rel_path
        record = _source_record(path, repo_root=repo)
        token_rows = []
        for token in required_tokens:
            token_rows.append(
                {
                    "token": token,
                    "present": bool(record.get("exists")) and token in record.get("text", ""),
                }
            )
        passed = bool(record.get("exists")) and all(row["present"] for row in token_rows)
        if not passed:
            blockers.append(f"source_check_failed:{check_id}")
        record.update(
            {
                "check_id": check_id,
                "required_tokens": token_rows,
                "passed": passed,
            }
        )
        record.pop("text", None)
        records.append(record)

    add_record(
        rel_path=UPSTREAM_REL / "frame_utils.py",
        check_id="upstream_frame_utils_seq_len_2",
        required_tokens=("seq_len = 2", "seq_buf = []"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "evaluate.py",
        check_id="upstream_evaluate_seq_len_shape_assert",
        required_tokens=(
            "TensorVideoDataset",
            "[seq_len, camera_size[1], camera_size[0], 3]",
            "DistortionNet().eval()",
        ),
    )
    add_record(
        rel_path=UPSTREAM_REL / "modules.py",
        check_id="upstream_posenet_uses_yuv6_pair",
        required_tokens=("IN_CHANS = 6 * 2", "rgb_to_yuv6(x)", "b (t c) h w"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "frame_utils.py",
        check_id="upstream_rgb_to_yuv6_is_no_grad",
        required_tokens=("@torch.no_grad()", "def rgb_to_yuv6"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "modules.py",
        check_id="upstream_segnet_last_frame",
        required_tokens=("x = x[:, -1, ...]", "SegNet"),
    )
    add_record(
        rel_path=UPSTREAM_REL / "modules.py",
        check_id="upstream_segnet_argmax_distortion",
        required_tokens=("out1.argmax(dim=1)", "out2.argmax(dim=1)"),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "score.py",
        check_id="pr95_score_streams_non_overlapping_pairs",
        required_tokens=("gt_pairs_iter_state['prev'] = None", "compute_distortion"),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "stages" / "common.py",
        check_id="pr95_training_eval_roundtrip_ste",
        required_tokens=(
            "F.interpolate(up, size=(384, 512)",
            "decoded_clamped.round()",
            "distortion_net.preprocess_input(decoded_bhwc)",
        ),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "losses.py",
        check_id="pr95_losses_include_seg_margin_and_pose",
        required_tokens=(
            "ce_seg_loss",
            "smooth_disagreement_seg_loss",
            "l7_softplus_seg_loss",
            "def pose_loss",
        ),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "losses.py",
        check_id="pr95_qat_and_c1a_present",
        required_tokens=("cat_entropy_v2", "fake_quantize", "apply_qat"),
    )
    add_record(
        rel_path=PR95_SOURCE_REL / "src" / "stages" / "stage8_muon_finetune.py",
        check_id="pr95_stage8_muon_present",
        required_tokens=("use_muon=True", "muon_lr", "Muon"),
    )
    add_record(
        rel_path=RUNNER_REL,
        check_id="runner_hinerv_eval_roundtrip_metadata",
        required_tokens=(
            "_hi_nerv_eval_roundtrip_ste_metadata",
            "eval_roundtrip_ste_enabled",
            "pose_student_input_preprocess",
        ),
    )

    stage_paths = sorted((pr95 / "src" / "stages").glob("stage*.py"))
    expected_stage_names = {
        "stage1_v328_ce.py",
        "stage2_v331_softplus.py",
        "stage3_v332_smooth.py",
        "stage4_v332_qat.py",
        "stage5_c1a_l7.py",
        "stage6_lambda_sweep.py",
        "stage7_sigma_sweep.py",
        "stage8_muon_finetune.py",
    }
    present_stage_names = {path.name for path in stage_paths}
    eight_stage_passed = expected_stage_names.issubset(present_stage_names)
    if not eight_stage_passed:
        blockers.append("source_check_failed:pr95_eight_stage_curriculum_present")
    records.append(
        {
            "schema": "pr95_distortion_source_record.v1",
            "check_id": "pr95_eight_stage_curriculum_present",
            "path": (PR95_SOURCE_REL / "src" / "stages").as_posix(),
            "exists": (pr95 / "src" / "stages").is_dir(),
            "sha256": _combined_sha256(stage_paths),
            "bytes": sum(path.stat().st_size for path in stage_paths if path.is_file()),
            "required_stage_files": sorted(expected_stage_names),
            "present_stage_files": sorted(present_stage_names),
            "passed": eight_stage_passed,
        }
    )

    check_passed = {
        str(record.get("check_id")): bool(record.get("passed")) for record in records
    }
    practice_source_rows = []
    for practice in PRACTICES:
        missing = [
            check_id
            for check_id in practice.source_check_ids
            if check_passed.get(check_id) is not True
        ]
        practice_source_rows.append(
            {
                "schema": "pr95_distortion_practice_source_row.v1",
                "practice_id": practice.practice_id,
                "title": practice.title,
                "source_check_ids": list(practice.source_check_ids),
                "source_ready": not missing,
                "missing_source_check_ids": missing,
            }
        )

    inventory = {
        "schema": SOURCE_INVENTORY_SCHEMA,
        "repo_root": repo.as_posix(),
        "pr95_source_dir": (repo / PR95_SOURCE_REL).as_posix(),
        "upstream_dir": (repo / UPSTREAM_REL).as_posix(),
        "runner_path": runner.as_posix(),
        "source_records": records,
        "practice_source_rows": practice_source_rows,
        "source_ready": not blockers,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    inventory["sha256"] = _payload_sha256(inventory)
    return inventory


def build_pr95_distortion_practices_row_guard(
    row: Mapping[str, Any],
    *,
    repo_root: str | Path,
    source_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed PR95 distortion-practice guard for one launch row."""

    if not isinstance(row, Mapping):
        raise PR95DistortionPracticesGuardError("row must be a mapping")
    inventory = (
        dict(source_inventory)
        if isinstance(source_inventory, Mapping)
        else build_pr95_distortion_source_inventory(repo_root)
    )
    command = _command_list(row)
    family = str(row.get("family") or _arg_value(command, "--execute-family") or "unknown")
    row_id = str(row.get("id") or row.get("row_id") or row.get("candidate_id") or "")
    required = family in {"hi_nerv", "snerv"}
    source_by_practice = {
        str(item.get("practice_id")): item
        for item in _mapping_list(inventory.get("practice_source_rows"))
    }

    practice_rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for practice in PRACTICES:
        source_row = source_by_practice.get(practice.practice_id, {})
        source_ready = source_row.get("source_ready") is True
        observed, observed_evidence = _observe_practice(
            practice.practice_id,
            family=family,
            row=row,
            command=command,
            source_inventory=inventory,
        )
        passed = (not required) or (source_ready and observed)
        blocker = None
        if required and not passed:
            blocker = f"{family}_pr95_distortion_{practice.practice_id}_missing"
            blockers.append(blocker)
        practice_rows.append(
            {
                "schema": PRACTICE_ROW_SCHEMA,
                "practice_id": practice.practice_id,
                "title": practice.title,
                "required_for_family": required,
                "source_ready": source_ready,
                "observed": observed,
                "passed": passed,
                "blocker": blocker,
                "observed_evidence": observed_evidence,
                "source_check_ids": list(practice.source_check_ids),
                "why_it_matters": practice.why_it_matters,
            }
        )

    if required and inventory.get("source_ready") is not True:
        blockers.insert(0, "pr95_distortion_source_inventory_incomplete")

    passed_count = sum(1 for item in practice_rows if item["passed"])
    guard = {
        "schema": SCHEMA,
        "family": family,
        "row_id": row_id,
        "required_for_family": required,
        "command": command,
        "source_inventory_schema": inventory.get("schema"),
        "source_inventory_sha256": inventory.get("sha256"),
        "required_practice_count": len(PRACTICES) if required else 0,
        "passed_practice_count": passed_count if required else 0,
        "launch_allowed": not blockers,
        "practice_rows": practice_rows,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    guard["sha256"] = _payload_sha256(guard)
    return guard


def build_pr95_distortion_practices_payload_guard(
    payload: Mapping[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build row guards for a verdict, queue, report, or single row payload."""

    if not isinstance(payload, Mapping):
        raise PR95DistortionPracticesGuardError("payload must be a mapping")
    inventory = build_pr95_distortion_source_inventory(repo_root)
    rows = _extract_candidate_rows(payload)
    row_guards = [
        build_pr95_distortion_practices_row_guard(
            row,
            repo_root=repo_root,
            source_inventory=inventory,
        )
        for row in rows
    ]
    blockers = [
        blocker
        for guard in row_guards
        for blocker in _string_list(guard.get("blockers"))
    ]
    if not rows:
        blockers.append("pr95_distortion_guard_no_candidate_rows_found")
    if inventory.get("source_ready") is not True:
        blockers.append("pr95_distortion_source_inventory_incomplete")
    out = {
        "schema": PAYLOAD_GUARD_SCHEMA,
        "source_inventory": inventory,
        "candidate_row_count": len(rows),
        "row_guards": row_guards,
        "launch_allowed": not blockers,
        "blockers": _dedupe(blockers),
        **FALSE_AUTHORITY,
    }
    out["sha256"] = _payload_sha256(out)
    return out


def render_pr95_distortion_practices_markdown(payload: Mapping[str, Any]) -> str:
    """Render a compact Markdown summary for guard artifacts."""

    lines = [
        "# PR95 Distortion Practices Guard",
        "",
        f"Schema: `{payload.get('schema')}`",
        f"Launch allowed: `{payload.get('launch_allowed')}`",
        f"Candidate rows: `{payload.get('candidate_row_count', 1)}`",
        "",
        "## Practices",
        "",
    ]
    row_guards = _mapping_list(payload.get("row_guards"))
    if not row_guards and payload.get("schema") == SCHEMA:
        row_guards = [payload]
    for guard in row_guards:
        lines.append(f"### `{guard.get('row_id') or guard.get('family')}`")
        for row in _mapping_list(guard.get("practice_rows")):
            lines.append(
                f"- `{row.get('practice_id')}` passed=`{row.get('passed')}` "
                f"observed=`{row.get('observed')}`"
            )
        lines.append("")
    lines.append("## Blockers")
    lines.append("")
    blockers = _string_list(payload.get("blockers"))
    if not blockers and row_guards:
        blockers = [
            blocker
            for guard in row_guards
            for blocker in _string_list(guard.get("blockers"))
        ]
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in _dedupe(blockers))
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _observe_practice(
    practice_id: str,
    *,
    family: str,
    row: Mapping[str, Any],
    command: Sequence[str],
    source_inventory: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if practice_id == "official_non_overlapping_seq2_pair_geometry":
        num_pairs = _positive_int_arg(command, "--num-pairs")
        batch_pairs = _positive_int_arg(command, "--batch-pairs") or _positive_int_arg(
            command, "--snerv-score-aware-long-training-batch-pairs"
        )
        evidence = []
        if num_pairs:
            evidence.append(f"command_num_pairs={num_pairs}")
        if batch_pairs:
            evidence.append(f"command_batch_pairs={batch_pairs}")
        return bool(num_pairs and batch_pairs), evidence

    if practice_id == "scorer_preprocess_eval_roundtrip_yuv6":
        evidence = []
        pose_weight = _positive_float_arg(command, "--pose-distillation-weight")
        if pose_weight:
            evidence.append(f"pose_distillation_weight={pose_weight:g}")
        explicit_snerv = _has_flag(
            command, "--snerv-score-aware-long-training-eval-roundtrip-ste"
        )
        if explicit_snerv:
            evidence.append("snerv_eval_roundtrip_ste_flag")
        hinerv_runner_support = _source_check_passed(
            source_inventory, "runner_hinerv_eval_roundtrip_metadata"
        )
        if family == "hi_nerv" and hinerv_runner_support:
            evidence.append("hinerv_runner_eval_roundtrip_metadata_source_verified")
        payload_declares = _deep_truthy(
            row,
            {
                "eval_roundtrip_ste_enabled",
                "eval_roundtrip_ste_attached",
                "native_mlx_eval_roundtrip_ste_bound",
            },
        )
        if payload_declares:
            evidence.append("payload_eval_roundtrip_truthy")
        observed = bool(pose_weight) and (
            explicit_snerv or payload_declares or (family == "hi_nerv" and hinerv_runner_support)
        )
        return observed, evidence

    if practice_id == "dual_component_real_scorer_pressure":
        seg_weight = _positive_float_arg(command, "--segnet-distillation-weight") or _positive_float_arg(
            command, "--segnet-direct-live-distillation-weight"
        )
        pose_weight = _positive_float_arg(command, "--pose-distillation-weight")
        device = _arg_value(command, "--distillation-device")
        evidence = []
        if seg_weight:
            evidence.append(f"segnet_distillation_weight={seg_weight:g}")
        if pose_weight:
            evidence.append(f"pose_distillation_weight={pose_weight:g}")
        if device:
            evidence.append(f"distillation_device={device}")
        return bool(seg_weight and pose_weight and device), evidence

    if practice_id == "pr95_staged_qat_coder_curriculum":
        evidence = []
        hinerv_policy = _arg_value(command, "--hi-nerv-optimizer-policy")
        snerv_curriculum = _has_flag(
            command, "--snerv-score-aware-long-training-pr95-faithful-curriculum"
        )
        coder_qat = _has_flag(command, "--coder-aware-qat")
        c1a_weight = _positive_float_arg(command, "--coder-qat-c1a-entropy-weight")
        optimizer = _arg_value(command, "--optimizer-kind") or _arg_value(
            command, "--snerv-score-aware-long-training-optimizer"
        )
        if hinerv_policy:
            evidence.append(f"hi_nerv_optimizer_policy={hinerv_policy}")
        if snerv_curriculum:
            evidence.append("snerv_pr95_faithful_curriculum_flag")
        if coder_qat:
            evidence.append("coder_aware_qat_flag")
        if c1a_weight:
            evidence.append(f"coder_qat_c1a_entropy_weight={c1a_weight:g}")
        if optimizer:
            evidence.append(f"optimizer={optimizer}")
        curriculum = (
            hinerv_policy == "pr95_curriculum"
            or snerv_curriculum
            or _deep_truthy(row, {"pr95_faithful_curriculum_enabled", "pr95_staged_curriculum"})
        )
        return bool(curriculum and coder_qat and c1a_weight), evidence

    return False, []


def _source_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    exists = path.is_file()
    text = path.read_text(encoding="utf-8") if exists else ""
    try:
        rel = path.relative_to(repo_root).as_posix()
    except ValueError:
        rel = path.as_posix()
    return {
        "schema": "pr95_distortion_source_record.v1",
        "path": rel,
        "exists": exists,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if exists else None,
        "bytes": len(text.encode("utf-8")) if exists else 0,
        "text": text,
    }


def _extract_candidate_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    direct = _mapping_list(payload.get("selected_local_mlx_experiments"))
    if direct:
        return direct
    queue = payload.get("experiment_queue")
    if isinstance(queue, Mapping):
        experiments = _mapping_list(queue.get("experiments"))
        rows = [row for row in experiments if _command_list(row)]
        if rows:
            return rows
    experiments = _mapping_list(payload.get("experiments"))
    if experiments:
        rows = [row for row in experiments if _command_list(row)]
        if rows:
            return rows
    selected_rows = _mapping_list(payload.get("selected_rows"))
    if selected_rows:
        rows = [row for row in selected_rows if _command_list(row)]
        if rows:
            return rows
    if _command_list(payload):
        return [payload]
    return []


def _command_list(row: Mapping[str, Any]) -> list[str]:
    for key in ("command", "command_argv", "argv"):
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
    steps = row.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        for step in steps:
            if isinstance(step, Mapping):
                value = step.get("command")
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    return [str(item) for item in value]
    return []


def _arg_value(command: Sequence[str], flag: str) -> str | None:
    try:
        return str(command[command.index(flag) + 1])
    except (ValueError, IndexError):
        return None


def _has_flag(command: Sequence[str], flag: str) -> bool:
    return flag in command


def _positive_int_arg(command: Sequence[str], flag: str) -> int | None:
    value = _arg_value(command, flag)
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_float_arg(command: Sequence[str], flag: str) -> float | None:
    value = _arg_value(command, flag)
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def _source_check_passed(inventory: Mapping[str, Any], check_id: str) -> bool:
    for record in _mapping_list(inventory.get("source_records")):
        if record.get("check_id") == check_id:
            return record.get("passed") is True
    return False


def _deep_truthy(value: Any, keys: set[str]) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in keys and item is True:
                return True
            if _deep_truthy(item, keys):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_deep_truthy(item, keys) for item in value)
    return False


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item)]
    return []


def _combined_sha256(paths: Sequence[Path]) -> str | None:
    h = hashlib.sha256()
    count = 0
    for path in paths:
        if not path.is_file():
            continue
        count += 1
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest() if count else None


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    import json

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "PAYLOAD_GUARD_SCHEMA",
    "PRACTICES",
    "PRACTICE_ROW_SCHEMA",
    "SCHEMA",
    "SOURCE_INVENTORY_SCHEMA",
    "PR95DistortionPracticesGuardError",
    "build_pr95_distortion_practices_payload_guard",
    "build_pr95_distortion_practices_row_guard",
    "build_pr95_distortion_source_inventory",
    "render_pr95_distortion_practices_markdown",
]
