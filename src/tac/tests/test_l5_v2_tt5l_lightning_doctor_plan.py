# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_tt5l_lightning_doctor_plan import (
    L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA,
    build_l5_v2_tt5l_lightning_doctor_plan,
    l5_v2_tt5l_lightning_doctor_plan_json,
    render_l5_v2_tt5l_lightning_doctor_plan_markdown,
)
from tac.optimization.l5_v2_tt5l_lightning_route_unblock_packet import (
    L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _route_packet(tmp_path: Path, *, artifact_blockers: list[str] | None = None) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "schema": L5V2_TT5L_LIGHTNING_ROUTE_UNBLOCK_PACKET_SCHEMA,
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "provider_spend_attempted": False,
        "current_head_commit": "a" * 40,
        "blockers": artifact_blockers or [],
        "remaining_blockers": [
            "Lightning credits or quota not checked",
            "LIGHTNING_SDK_USER or LIGHTNING_ORG missing",
            "LIGHTNING_SSH_TARGET missing",
            "LIGHTNING_TEAMSPACE missing",
            "active dispatch claims not created for non-dry-run cells",
            "Lightning machine inventory not checked",
            "source manifest not staged to remote Lightning workspace",
            "remote CUDA runtime not probed",
        ],
        "source_artifacts": {
            "sideinfo_lightning_paired_axis_plan": {
                "source_relevant_paths_match_current_head": True,
            }
        },
    }
    path = _write_json(tmp_path / ".omx/research/route_packet.json", payload)
    return path, payload


def test_tt5l_lightning_doctor_plan_builds_two_valid_identity_commands(
    tmp_path: Path,
) -> None:
    route_path, route = _route_packet(tmp_path)

    plan = build_l5_v2_tt5l_lightning_doctor_plan(
        route_packet=route,
        route_packet_path=route_path,
        repo_root=tmp_path,
        current_head_commit="b" * 40,
    )

    assert plan["schema"] == L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA
    assert plan["planning_only"] is True
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_provider_dispatch"] is False
    assert plan["ready_for_operator_doctor"] is True
    assert plan["ready_for_non_dry_run_submit"] is False
    assert plan["source_route_packet_commit"] == "a" * 40
    assert plan["source_route_remaining_blockers"] == route["remaining_blockers"]
    modes = {mode["mode"]: mode for mode in plan["identity_modes"]}
    user_command = modes["user"]["doctor_command_template"]
    org_command = modes["org"]["doctor_command_template"]
    assert "--user \"$LIGHTNING_SDK_USER\"" in user_command
    assert "--org \"$LIGHTNING_ORG\"" in org_command
    for command in (user_command, org_command):
        assert "scripts/launch_lightning_batch_job.py doctor" in command
        assert "--require-ssh" in command
        assert "--require-remote-supply-chain" in command
        assert "--require-machine-inventory" in command
        assert "--machine T4" in command
        assert "--gpu-only" in command
        assert "<--user-or---org>" not in command
        assert "<lightning-user-or-org>" not in command
    assert plan["doctor_required_checks"]["expected_status"] == "OK"
    assert "ssh_auth" in plan["doctor_required_checks"]["required_checks"]
    assert "machine_inventory" in plan["doctor_required_checks"]["required_checks"]


def test_tt5l_lightning_doctor_plan_fails_closed_on_route_artifact_blockers(
    tmp_path: Path,
) -> None:
    route_path, route = _route_packet(
        tmp_path,
        artifact_blockers=["paired_axis_plan:source_relevant_paths_changed"],
    )

    plan = build_l5_v2_tt5l_lightning_doctor_plan(
        route_packet=route,
        route_packet_path=route_path,
        repo_root=tmp_path,
    )

    assert plan["ready_for_operator_doctor"] is False
    assert "source_route_packet_has_artifact_blockers" in plan["blockers"]
    assert plan["score_claim"] is False
    assert plan["ready_for_provider_dispatch"] is False


def test_tt5l_lightning_doctor_plan_json_and_markdown_are_false_authority(
    tmp_path: Path,
) -> None:
    route_path, route = _route_packet(tmp_path)
    plan = build_l5_v2_tt5l_lightning_doctor_plan(
        route_packet=route,
        route_packet_path=route_path,
        repo_root=tmp_path,
    )

    decoded = json.loads(l5_v2_tt5l_lightning_doctor_plan_json(plan))
    report = render_l5_v2_tt5l_lightning_doctor_plan_markdown(plan)

    assert decoded["schema"] == L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA
    assert "score_claim=false" in report
    assert "promotion_eligible=false" in report
    assert "ready_for_provider_dispatch=false" in report
    assert "doctor" in report
    assert "--user \"$LIGHTNING_SDK_USER\"" in report
    assert "--org \"$LIGHTNING_ORG\"" in report


def test_tt5l_lightning_doctor_plan_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    route_path, _route = _route_packet(tmp_path)
    output_json = tmp_path / "doctor_plan.json"
    output_md = tmp_path / "doctor_plan.md"

    proc = subprocess.run(
        [
            str(root / "tools" / "build_l5_v2_tt5l_lightning_doctor_plan.py"),
            "--repo-root",
            str(tmp_path),
            "--route-packet-json",
            str(route_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--current-head-commit",
            "c" * 40,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["current_head_commit"] == "c" * 40
    assert payload["ready_for_operator_doctor"] is True
    assert "score_claim=false" in proc.stdout
    assert output_md.is_file()
