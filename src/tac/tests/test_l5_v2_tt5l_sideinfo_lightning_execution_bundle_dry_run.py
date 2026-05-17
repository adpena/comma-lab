# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH,
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
    T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_SCHEMA,
    DryRunCommandResult,
    build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification,
    l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_json,
    render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_markdown,
)


def _sha(seed: object) -> str:
    return hashlib.sha256(str(seed).encode("utf-8")).hexdigest()


def _arg_value(argv: list[str], flag: str) -> str:
    try:
        idx = argv.index(flag)
    except ValueError:
        return ""
    if idx + 1 >= len(argv):
        return ""
    return argv[idx + 1]


def _metadata(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    idx = 0
    while idx < len(argv):
        if argv[idx] == "--queue-metadata" and idx + 1 < len(argv):
            key, value = argv[idx + 1].split("=", 1)
            out[key] = value
            idx += 2
            continue
        idx += 1
    return out


def _with_t4_runtime_env_args(argv: list[str]) -> list[str]:
    out = list(argv)
    for item in T4_LIGHTNING_EXACT_EVAL_RUNTIME_ENV:
        out.extend(["--env", item])
    return out


def _without_env_args(command: str) -> str:
    argv = shlex.split(command)
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        if argv[idx] == "--env":
            idx += 2
            continue
        out.append(argv[idx])
        idx += 1
    return shlex.join(out)


def _without_arg_value(command: str, flag: str) -> str:
    argv = shlex.split(command)
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        if argv[idx] == flag:
            idx += 2
            continue
        out.append(argv[idx])
        idx += 1
    return shlex.join(out)


def _fake_bundle(repo_root: Path) -> dict[str, object]:
    cells: list[dict[str, object]] = []
    variants: list[dict[str, object]] = []
    inflate_sh = repo_root / "runtime" / "inflate.sh"
    inflate_sh.parent.mkdir(parents=True, exist_ok=True)
    inflate_sh.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate_sh.chmod(0o755)
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        archive_path = repo_root / "archives" / variant / "archive.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_bytes = (f"archive:{variant}:".encode() * 4000)[:34373]
        archive_path.write_bytes(archive_bytes)
        archive_sha = hashlib.sha256(archive_bytes).hexdigest()
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            device = "cpu" if axis == "contest_cpu" else "cuda"
            lane_id = f"lane_l5_v2_tt5l_sideinfo_effect_curve_{variant}_{axis}"
            source_sha = _sha(f"{variant}:{axis}:source")
            pair_group_id = f"pair_l5_v2_tt5l_sideinfo_effect_curve_{variant}_{archive_sha[:12]}"
            run_id = f"l5_v2_tt5l_sideinfo_effect_curve_{variant}_{archive_sha[:12]}"
            local_artifact_dir = (
                "experiments/results/lightning_batch/"
                f"l5_v2_tt5l_sideinfo_effect_curve_paired_axes/{variant}/{axis}"
            )
            command = shlex.join(
                _with_t4_runtime_env_args(
                    [
                        ".venv/bin/python",
                        "scripts/launch_lightning_batch_job.py",
                        "exact-eval",
                        "--state-path",
                        f"{local_artifact_dir}/launcher_dry_run_state.json",
                        "--job-name",
                        f"l5-v2-tt5l-sideinfo-{variant}-{device}-test",
                        "--machine",
                        "T4",
                        "--inflate-sh",
                        inflate_sh.relative_to(repo_root).as_posix(),
                        "--expected-archive-sha256",
                        archive_sha,
                        "--expected-archive-size-bytes",
                        "34373",
                        "--local-artifact-dir",
                        local_artifact_dir,
                        "--dispatch-lane-id",
                        lane_id,
                        "--eval-device",
                        device,
                        "--source-manifest",
                        f"experiments/results/lightning_batch/{variant}-{axis}/source_manifest.json",
                        "--queue-metadata",
                        f"variant={variant}",
                        "--queue-metadata",
                        f"axis={axis}",
                        "--queue-metadata",
                        f"lane_id={lane_id}",
                        "--queue-metadata",
                        f"pair_group_id={pair_group_id}",
                        "--queue-metadata",
                        f"run_id={run_id}",
                        "--queue-metadata",
                        f"archive_sha256={archive_sha}",
                        "--queue-metadata",
                        "source_plan="
                        f"{L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_ARTIFACT_PATH}",
                        "--queue-metadata",
                        f"source_spec_command_sha256={source_sha}",
                        "--adjudicate",
                        "--dry-run",
                    ]
                )
            )
            non_dry = shlex.join(
                _with_t4_runtime_env_args(
                    [
                        ".venv/bin/python",
                        "scripts/launch_lightning_batch_job.py",
                        "exact-eval",
                        "--job-name",
                        f"l5-v2-tt5l-sideinfo-{variant}-{device}-test",
                        "--machine",
                        "T4",
                        "--inflate-sh",
                        inflate_sh.relative_to(repo_root).as_posix(),
                        "--studio",
                        "<lightning-studio>",
                        "--remote-preflight-ssh-target",
                        "<lightning-ssh-target>",
                    ]
                )
            )
            cells.append(
                {
                    "planning_only": True,
                    "score_claim_valid": False,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_provider_dispatch": False,
                    "dispatch_attempted": False,
                    "variant": variant,
                    "axis": axis,
                    "axis_label": (
                        "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]"
                    ),
                    "eval_device": device,
                    "lane_id": lane_id,
                    "archive_path": archive_path.relative_to(repo_root).as_posix(),
                    "archive_sha256": archive_sha,
                    "archive_size_bytes": 34373,
                    "pair_group_id": pair_group_id,
                    "run_id": run_id,
                    "local_artifact_dir": local_artifact_dir,
                    "dry_run_state_path": f"{local_artifact_dir}/launcher_dry_run_state.json",
                    "dry_run_submit_command": command,
                    "non_dry_run_submit_command_template": non_dry,
                    "ready_for_dry_run_submit": True,
                    "ready_for_non_dry_run_submit": False,
                    "source_spec_command_sha256": source_sha,
                    "blockers": [],
                }
            )
        variants.append(
            {
                "variant": variant,
                "archive_path": archive_path.relative_to(repo_root).as_posix(),
                "archive_sha256": archive_sha,
                "archive_bytes": 34373,
            }
        )
    manifest_path = repo_root / ".omx/research/fake_variant_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"schema": "fake", "variants": variants}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_plan_cells = [
        {
            "variant": cell["variant"],
            "axis": cell["axis"],
            "archive_sha256": cell["archive_sha256"],
            "archive_size_bytes": cell["archive_size_bytes"],
            "pair_group_id": cell["pair_group_id"],
            "run_id": cell["run_id"],
            "local_artifact_dir": cell["local_artifact_dir"],
            "command_sha256": cell["source_spec_command_sha256"],
        }
        for cell in cells
    ]
    source_plan = {
        "schema": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_LIGHTNING_PAIRED_AXIS_PLAN_SCHEMA,
        "cells": source_plan_cells,
    }
    source_plan_bytes = json.dumps(source_plan, sort_keys=True).encode("utf-8") + b"\n"
    source_plan_path = repo_root / ".omx/research/fake_lightning_plan.json"
    source_plan_path.write_bytes(source_plan_bytes)
    return {
        "schema": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
        "tool": "tools/build_l5_v2_tt5l_sideinfo_lightning_execution_bundle.py",
        "planning_only": True,
        "score_claim_valid": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "ready_for_non_dry_run_submit": False,
        "ready_for_dry_run_submit": True,
        "source_plan": source_plan_path.relative_to(repo_root).as_posix(),
        "source_plan_sha256": hashlib.sha256(source_plan_bytes).hexdigest(),
        "source_variant_manifest": manifest_path.relative_to(repo_root).as_posix(),
        "cell_count": len(cells),
        "ready_dry_run_cell_count": len(cells),
        "cells": cells,
        "blockers": [],
    }


def _fake_runner(
    command: str,
    repo_root: Path,
    _timeout_seconds: int,
    *,
    cuda_marker: bool = True,
    metadata_override: dict[str, str] | None = None,
) -> DryRunCommandResult:
    argv = shlex.split(command)
    device = _arg_value(argv, "--eval-device")
    metadata = _metadata(argv)
    if metadata_override:
        metadata.update(metadata_override)
    role = f"exact_{device}_eval"
    command_text = (
        "set -euo pipefail\n"
        "scripts/scan_lightning_supply_chain.py --phase pre\n"
    )
    if device == "cuda":
        command_text += "LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK\n"
        command_text += "LIGHTNING_RUNNER_DALI_PREFLIGHT_OK\n"
        if cuda_marker:
            command_text += "export INFLATE_REQUIRE_CUDA=1\n"
    else:
        command_text += "LIGHTNING_RUNNER_CPU_PREFLIGHT_OK\n"
    command_text += (
        "experiments/contest_auth_eval.py "
        f"--device {device} "
        "--archive archive.zip --output contest_auth_eval.json\n"
    )
    record = {
        "dry_run": True,
        "queue": {
            "role": role,
            "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
            "expected_archive_sha256": _arg_value(argv, "--expected-archive-sha256"),
            "expected_archive_size_bytes": int(
                _arg_value(argv, "--expected-archive-size-bytes")
            ),
            "local_artifact_dir": _arg_value(argv, "--local-artifact-dir"),
            "queue_metadata": metadata,
            "adjudication": {
                "required_device": device,
                "required_samples": 600,
            },
        },
        "spec": {"command": command_text},
    }
    state_path = _arg_value(argv, "--state-path")
    if state_path:
        resolved_state_path = repo_root / state_path
        resolved_state_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_state_path.write_text(
            json.dumps([record], sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return DryRunCommandResult(
        returncode=0,
        stdout=json.dumps(record, sort_keys=True),
        stderr="",
        argv=tuple(argv),
    )


def test_tt5l_lightning_bundle_dry_run_verifier_accepts_all_cells(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
        generated_at_utc="2026-05-17T00:00:00Z",
    )

    assert payload["schema"] == L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_DRY_RUN_SCHEMA
    assert payload["all_dry_runs_passed"] is True
    assert payload["passed_cell_count"] == 10
    assert payload["ready_for_dry_run_submit"] is True
    assert payload["ready_for_non_dry_run_submit"] is False
    assert payload["ready_for_provider_dispatch"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["blockers"] == []
    first = payload["cells"][0]
    assert first["verified"] is True
    assert first["inflate_runtime"]["exists"] is True
    assert first["inflate_runtime"]["executable"] is True
    assert first["source_plan_cell"]["matched"] is True
    assert (
        first["source_plan_cell"]["command_sha256"]
        == first["queue"]["source_spec_command_sha256"]
    )
    assert first["dry_run_state_file"]["stdout_core_matched"] is True
    assert first["queue"]["launcher_command_sha_matches_source_spec"] is False
    assert first["queue"]["command_sha_delta_classification"] == "expected_submit_layer_delta"


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_cuda_without_cuda_requirement(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=lambda command, repo_root, timeout: _fake_runner(
            command,
            repo_root,
            timeout,
            cuda_marker=False,
        ),
    )

    assert payload["all_dry_runs_passed"] is False
    assert payload["ready_for_dry_run_submit"] is False
    assert any(
        "dry_run_spec_cuda_inflate_requirement_missing" in blocker
        for blocker in payload["blockers"]
    )
    cuda_cells = [cell for cell in payload["cells"] if cell["axis"] == "contest_cuda"]
    assert cuda_cells
    assert all(cell["verified"] is False for cell in cuda_cells)


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_t4_without_runtime_env(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    first_cell = bundle["cells"][0]
    assert isinstance(first_cell, dict)
    first_cell["dry_run_submit_command"] = _without_env_args(
        str(first_cell["dry_run_submit_command"])
    )
    first_cell["non_dry_run_submit_command_template"] = _without_env_args(
        str(first_cell["non_dry_run_submit_command_template"])
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert payload["ready_for_dry_run_submit"] is False
    assert any(
        "dry_run_t4_runtime_env_missing:INFLATE_TORCH_SPEC=torch==2.5.1+cu124"
        in blocker
        for blocker in payload["blockers"]
    )
    assert any(
        "non_dry_run_t4_runtime_env_missing:UV_INDEX_STRATEGY=unsafe-best-match"
        in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_stale_state_file(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    def stale_state_runner(
        command: str,
        repo_root: Path,
        timeout: int,
    ) -> DryRunCommandResult:
        result = _fake_runner(command, repo_root, timeout)
        argv = shlex.split(command)
        state_path = repo_root / _arg_value(argv, "--state-path")
        stale_record = json.loads(result.stdout)
        stale_record["queue"]["queue_metadata"]["axis"] = "stale_axis"
        state_path.write_text(
            json.dumps([stale_record], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=stale_state_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert payload["ready_for_dry_run_submit"] is False
    assert any(
        "dry_run_state_stdout_queue_mismatch" in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_missing_inflate_runtime(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    first_cell = bundle["cells"][0]
    assert isinstance(first_cell, dict)
    first_cell["dry_run_submit_command"] = _without_arg_value(
        str(first_cell["dry_run_submit_command"]),
        "--inflate-sh",
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert payload["ready_for_dry_run_submit"] is False
    assert any(
        "dry_run_inflate_sh_arg_missing" in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_metadata_axis_drift(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=lambda command, repo_root, timeout: _fake_runner(
            command,
            repo_root,
            timeout,
            metadata_override={"axis": "contest_cuda"},
        ),
    )

    assert payload["all_dry_runs_passed"] is False
    assert any("dry_run_queue_metadata_axis_mismatch" in blocker for blocker in payload["blockers"])


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_source_plan_sha_drift(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    first_cell = bundle["cells"][0]
    assert isinstance(first_cell, dict)
    first_cell["source_spec_command_sha256"] = "0" * 64
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert any(
        "source_spec_command_sha256_mismatch_source_plan" in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_absolute_source_plan(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle["source_plan"] = str(
        tmp_path / ".omx/research/fake_lightning_plan.json"
    )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert "source_plan_path_absolute_rejected" in payload["blockers"]


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_source_plan_traversal(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle["source_plan"] = "../outside_plan.json"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert "source_plan_path_parent_traversal_rejected" in payload["blockers"]


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_unpaired_axis_cells(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    first_variant = L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS[0]
    for cell in bundle["cells"]:
        assert isinstance(cell, dict)
        if cell["variant"] == first_variant and cell["axis"] == "contest_cuda":
            cell["pair_group_id"] = "pair_wrong_unpaired_cuda"
            break
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert payload["ready_for_dry_run_submit"] is False
    assert any(
        f"paired_axis_pair_group_id_mismatch:{first_variant}" in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_lightning_bundle_dry_run_verifier_rejects_duplicate_and_extra_cells(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    cells = bundle["cells"]
    assert isinstance(cells, list)
    first = dict(cells[0])
    extra = dict(cells[1])
    extra["variant"] = "unexpected_variant"
    extra["axis"] = "contest_cpu"
    cells.append(first)
    cells.append(extra)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")

    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    assert payload["all_dry_runs_passed"] is False
    assert payload["ready_for_dry_run_submit"] is False
    assert payload["coverage"]["duplicate_cells"] == ["zero:contest_cpu"]
    assert payload["coverage"]["extra_cells"] == ["unexpected_variant:contest_cpu"]
    assert any(
        "source_bundle_duplicate_cells:zero:contest_cpu" in blocker
        for blocker in payload["blockers"]
    )
    assert any(
        "source_bundle_extra_cells:unexpected_variant:contest_cpu" in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_lightning_bundle_dry_run_json_and_markdown_keep_false_authority(
    tmp_path: Path,
) -> None:
    bundle = _fake_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
    payload = build_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_verification(
        bundle=bundle,
        bundle_path=bundle_path,
        repo_root=tmp_path,
        runner=_fake_runner,
    )

    decoded = json.loads(
        l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_json(payload)
    )
    report = render_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run_markdown(
        payload
    )

    assert decoded["score_claim"] is False
    assert decoded["promotion_eligible"] is False
    assert decoded["provider_spend_attempted"] is False
    assert "no provider work was dispatched" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report
    assert "ready_for_provider_dispatch: `false`" in report
