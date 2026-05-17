# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_tt5l_lightning_doctor_plan import (
    L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA,
)
from tac.optimization.l5_v2_tt5l_lightning_non_dry_run_gate import (
    L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_SCHEMA,
    build_l5_v2_tt5l_lightning_non_dry_run_gate,
    l5_v2_tt5l_lightning_non_dry_run_gate_json,
    render_l5_v2_tt5l_lightning_non_dry_run_gate_markdown,
)
from tac.optimization.l5_v2_tt5l_sideinfo_lightning_execution_bundle import (
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH,
    L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
)

COMMIT = "a" * 40


def _manifest_sha(payload: dict[str, object]) -> str:
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _doctor_plan() -> dict[str, object]:
    return {
        "schema": L5V2_TT5L_LIGHTNING_DOCTOR_PLAN_SCHEMA,
        "ready_for_operator_doctor": True,
        "doctor_required_checks": {
            "expected_status": "OK",
            "required_json_fields": [
                "schema_version",
                "tool",
                "recorded_at_utc",
                "checks",
                "status",
                "failed_checks",
            ],
            "required_checks": [
                "local_supply_chain",
                "ssh_auth",
                "remote_supply_chain",
                "machine_inventory",
            ],
        },
    }


def _doctor_output() -> dict[str, object]:
    return {
        "schema_version": 1,
        "tool": "scripts/launch_lightning_batch_job.py doctor",
        "recorded_at_utc": "2026-05-17T00:00:00Z",
        "status": "OK",
        "failed_checks": [],
        "checks": {
            "local_supply_chain": {"ok": True},
            "ssh_auth": {"ok": True},
            "remote_supply_chain": {"ok": True},
            "machine_inventory": {"ok": True, "machine_count": 1},
        },
    }


def _source_manifest(
    *,
    run_id: str,
    archive_path: str,
    archive_sha256: str,
    archive_bytes: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": "2026-05-17T00:00:00Z",
        "tool": "scripts/lightning_repro_workspace.py",
        "run_id": run_id,
        "repo_root": "/repo",
        "remote": "studio@ssh.lightning.ai",
        "remote_pact": "/teamspace/studios/this_studio/pact",
        "source_paths": ["src", "experiments", "submissions", "scripts", "upstream", "tools"],
        "artifact_paths": [archive_path],
        "install_runtime": False,
        "requirements_mode": "no-install",
        "python_bin": None,
        "require_cuda": False,
        "runtime_security": {},
        "uv_locked": True,
        "git": {"head": COMMIT, "branch": "main", "status_short": []},
        "files": [
            {
                "path": archive_path,
                "role": "artifact",
                "bytes": archive_bytes,
                "sha256": archive_sha256,
            }
        ],
        "file_count": 1,
        "total_bytes": archive_bytes,
    }
    payload["manifest_sha256"] = _manifest_sha(payload)
    return payload


def _stage_arg(command: str, flag: str) -> Path:
    argv = shlex.split(command)
    return Path(argv[argv.index(flag) + 1])


def _receipt_from_manifest(
    *,
    manifest: dict[str, object],
    receipt_path: str,
    status: str = "OK",
    dry_run: bool = False,
    remote_sha256_verified: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "tool": "scripts/lightning_repro_workspace.py",
        "status": status,
        "dry_run": dry_run,
        "remote_sha256_verified": remote_sha256_verified,
        "manifest": str(receipt_path).replace("source_manifest_receipt.json", "source_manifest.json"),
        "remote_manifest": (
            "studio@ssh.lightning.ai:/teamspace/studios/this_studio/pact/.omx/state/"
            f"{manifest['run_id']}_manifest.json"
        ),
        "remote": "studio@ssh.lightning.ai",
        "remote_pact": "/teamspace/studios/this_studio/pact",
        "run_id": manifest["run_id"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def _non_dry_command(
    *,
    cell: dict[str, object],
    source_manifest_path: str,
    placeholder: bool = False,
) -> str:
    axis = str(cell["axis"])
    device = "cpu" if axis == "contest_cpu" else "cuda"
    studio = "<lightning-studio>" if placeholder else "tt5l-studio"
    ssh_target = "<lightning-ssh-target>" if placeholder else "studio@ssh.lightning.ai"
    return shlex.join(
        [
            ".venv/bin/python",
            "scripts/launch_lightning_batch_job.py",
            "exact-eval",
            "--job-name",
            str(cell["job_name"]),
            "--archive",
            f"/teamspace/studios/this_studio/pact/{cell['archive_path']}",
            "--repo-dir",
            "/teamspace/studios/this_studio/pact",
            "--upstream-dir",
            "/teamspace/studios/this_studio/pact/upstream",
            "--output-dir",
            f"/teamspace/studios/this_studio/pact/out/{cell['job_name']}",
            "--machine",
            "T4",
            "--inflate-sh",
            "/teamspace/studios/this_studio/pact/submissions/robust_current/inflate.sh",
            "--expected-archive-sha256",
            str(cell["archive_sha256"]),
            "--expected-archive-size-bytes",
            str(cell["archive_size_bytes"]),
            "--local-artifact-dir",
            f"experiments/results/lightning_batch/{cell['job_name']}",
            "--dispatch-lane-id",
            str(cell["lane_id"]),
            "--dispatch-claims-path",
            L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_CLAIMS_PATH,
            "--eval-device",
            device,
            "--source-manifest",
            source_manifest_path,
            "--adjudicate",
            "--studio",
            studio,
            "--teamspace",
            "tt5l-teamspace",
            "--user",
            "tt5l-user",
            "--remote-preflight-ssh-target",
            ssh_target,
        ]
    )


def _fake_bundle(tmp_path: Path, *, placeholder: bool = False) -> tuple[dict[str, object], str]:
    cells: list[dict[str, object]] = []
    claims_rows = [
        "# Active lane dispatch claims",
        "",
        "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS:
        archive_rel = f"archives/{variant}/archive.zip"
        archive_path = tmp_path / archive_rel
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_bytes = (f"{variant}:tt5l".encode() * 2000)[:12345]
        archive_path.write_bytes(archive_bytes)
        archive_sha = hashlib.sha256(archive_bytes).hexdigest()
        for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES:
            lane_id = f"lane_l5_v2_tt5l_sideinfo_effect_curve_{variant}_{axis}"
            job_name = f"l5-v2-tt5l-sideinfo-{variant}-{axis}-test"
            source_manifest_path = f"experiments/results/lightning_batch/{job_name}/source_manifest.json"
            source_receipt_path = (
                f"experiments/results/lightning_batch/{job_name}/"
                "source_manifest_receipt.json"
            )
            cell: dict[str, object] = {
                "variant": variant,
                "axis": axis,
                "axis_label": "[contest-CPU]" if axis == "contest_cpu" else "[contest-CUDA]",
                "lane_id": lane_id,
                "job_name": job_name,
                "archive_path": archive_rel,
                "archive_sha256": archive_sha,
                "archive_size_bytes": len(archive_bytes),
                "stage_source_manifest_command_template": shlex.join(
                    [
                        ".venv/bin/python",
                        "scripts/lightning_repro_workspace.py",
                        "--manifest-out",
                        source_manifest_path,
                        "--receipt-out",
                        source_receipt_path,
                        "--artifact",
                        archive_rel,
                    ]
                ),
            }
            cell["non_dry_run_submit_command_template"] = _non_dry_command(
                cell=cell,
                source_manifest_path=source_manifest_path,
                placeholder=placeholder,
            )
            cells.append(cell)
            manifest_path = tmp_path / source_manifest_path
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = _source_manifest(
                run_id=job_name,
                archive_path=archive_rel,
                archive_sha256=archive_sha,
                archive_bytes=len(archive_bytes),
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt_path = tmp_path / source_receipt_path
            receipt_path.write_text(
                json.dumps(
                    _receipt_from_manifest(
                        manifest=manifest,
                        receipt_path=source_receipt_path,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            notes = (
                f"score_claim=false; archive_sha256={archive_sha}; axis={axis}; "
                f"variant={variant}"
            )
            claims_rows.append(
                f"| 2026-05-17T00:00:00Z | codex:test | {lane_id} | lightning | "
                f"{job_name} | 2026-05-17T01:00:00Z | active_dispatching | {notes} |"
            )
    bundle = {
        "schema": L5V2_TT5L_SIDEINFO_LIGHTNING_EXECUTION_BUNDLE_SCHEMA,
        "planning_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_provider_dispatch": False,
        "dispatch_attempted": False,
        "ready_for_dry_run_submit": True,
        "ready_for_non_dry_run_submit": False,
        "current_head_commit": COMMIT,
        "cells": cells,
    }
    return bundle, "\n".join(claims_rows) + "\n"


def _build_gate(
    tmp_path: Path,
    *,
    bundle: dict[str, object],
    claims_text: str,
    doctor_output: dict[str, object] | None = None,
    current_head_commit: str = COMMIT,
) -> dict[str, object]:
    bundle_path = tmp_path / ".omx/research/bundle.json"
    doctor_plan_path = tmp_path / ".omx/research/doctor_plan.json"
    doctor_output_path = tmp_path / ".omx/research/doctor_output.json"
    claims_path = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    for path, payload in (
        (bundle_path, bundle),
        (doctor_plan_path, _doctor_plan()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if doctor_output is not None:
        doctor_output_path.parent.mkdir(parents=True, exist_ok=True)
        doctor_output_path.write_text(
            json.dumps(doctor_output, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    claims_path.write_text(claims_text, encoding="utf-8")
    return build_l5_v2_tt5l_lightning_non_dry_run_gate(
        bundle=bundle,
        bundle_path=bundle_path,
        doctor_plan=_doctor_plan(),
        doctor_plan_path=doctor_plan_path,
        doctor_output=doctor_output or {},
        doctor_output_path=doctor_output_path,
        claims_text=claims_text,
        claims_path=claims_path,
        repo_root=tmp_path,
        current_head_commit=current_head_commit,
        generated_at_utc="2026-05-17T00:00:00Z",
    )


def test_tt5l_non_dry_run_gate_allows_fully_closed_packet(tmp_path: Path) -> None:
    bundle, claims_text = _fake_bundle(tmp_path)

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=_doctor_output(),
    )

    assert payload["schema"] == L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_SCHEMA
    assert payload["ready_for_non_dry_run_submit"] is True
    assert payload["ready_for_provider_dispatch"] is True
    assert payload["ready_cell_count"] == 10
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["provider_spend_attempted"] is False
    assert payload["blockers"] == []


def test_tt5l_non_dry_run_gate_blocks_missing_doctor_and_placeholders(
    tmp_path: Path,
) -> None:
    bundle, claims_text = _fake_bundle(tmp_path, placeholder=True)

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=None,
    )

    assert payload["ready_for_non_dry_run_submit"] is False
    assert payload["ready_for_provider_dispatch"] is False
    blockers = payload["blockers"]
    assert isinstance(blockers, list)
    assert "doctor_output_missing" in blockers
    assert any(
        "non_dry_run_command_placeholders_present" in blocker for blocker in blockers
    )
    assert any(
        "non_dry_run_command_arg_placeholder:--studio" in blocker
        for blocker in blockers
    )
    assert any(
        "non_dry_run_command_arg_placeholder:--remote-preflight-ssh-target" in blocker
        for blocker in blockers
    )


def test_tt5l_non_dry_run_gate_blocks_missing_active_claims(
    tmp_path: Path,
) -> None:
    bundle, _claims_text = _fake_bundle(tmp_path)

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text="",
        doctor_output=_doctor_output(),
    )

    assert payload["ready_for_non_dry_run_submit"] is False
    assert "claims_ledger_has_no_rows" in payload["blockers"]
    assert any("active_lightning_claim_missing" in blocker for blocker in payload["blockers"])


def test_tt5l_non_dry_run_gate_blocks_stale_manifest_head(tmp_path: Path) -> None:
    bundle, claims_text = _fake_bundle(tmp_path)
    first_cell = bundle["cells"][0]
    assert isinstance(first_cell, dict)
    stage_command = str(first_cell["stage_source_manifest_command_template"])
    source_manifest_path = _stage_arg(stage_command, "--manifest-out")
    manifest_file = tmp_path / source_manifest_path
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["git"]["head"] = "b" * 40
    manifest["manifest_sha256"] = _manifest_sha(manifest)
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=_doctor_output(),
    )

    assert payload["ready_for_non_dry_run_submit"] is False
    assert any(
        "source_manifest_git_head_mismatch_bundle" in blocker
        for blocker in payload["blockers"]
    )


def test_tt5l_non_dry_run_gate_blocks_missing_stage_receipt(tmp_path: Path) -> None:
    bundle, claims_text = _fake_bundle(tmp_path)
    first_cell = bundle["cells"][0]
    assert isinstance(first_cell, dict)
    stage_command = str(first_cell["stage_source_manifest_command_template"])
    receipt_file = tmp_path / _stage_arg(stage_command, "--receipt-out")
    receipt_file.unlink()

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=_doctor_output(),
    )

    assert payload["ready_for_non_dry_run_submit"] is False
    assert any("stage_receipt_missing" in blocker for blocker in payload["blockers"])


def test_tt5l_non_dry_run_gate_blocks_dry_run_stage_receipt(tmp_path: Path) -> None:
    bundle, claims_text = _fake_bundle(tmp_path)
    first_cell = bundle["cells"][0]
    assert isinstance(first_cell, dict)
    stage_command = str(first_cell["stage_source_manifest_command_template"])
    manifest_file = tmp_path / _stage_arg(stage_command, "--manifest-out")
    receipt_file = tmp_path / _stage_arg(stage_command, "--receipt-out")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    receipt = _receipt_from_manifest(
        manifest=manifest,
        receipt_path=str(_stage_arg(stage_command, "--receipt-out")),
        status="DRY_RUN",
        dry_run=True,
        remote_sha256_verified=False,
    )
    receipt_file.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=_doctor_output(),
    )

    assert payload["ready_for_non_dry_run_submit"] is False
    blockers = payload["blockers"]
    assert any("stage_receipt_status_not_ok" in blocker for blocker in blockers)
    assert any("stage_receipt_dry_run_true" in blocker for blocker in blockers)
    assert any(
        "stage_receipt_remote_sha256_not_verified" in blocker for blocker in blockers
    )


def test_tt5l_non_dry_run_gate_blocks_missing_paired_axis_cell(
    tmp_path: Path,
) -> None:
    bundle, claims_text = _fake_bundle(tmp_path)
    cells = bundle["cells"]
    assert isinstance(cells, list)
    first_variant = L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS[0]
    bundle["cells"] = [
        cell
        for cell in cells
        if not (
            isinstance(cell, dict)
            and cell["variant"] == first_variant
            and cell["axis"] == "contest_cuda"
        )
    ]

    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=_doctor_output(),
    )

    assert payload["ready_for_non_dry_run_submit"] is False
    assert (
        f"source_bundle_missing_cell:{first_variant}:contest_cuda"
        in payload["blockers"]
    )


def test_tt5l_non_dry_run_gate_json_markdown_and_cli(tmp_path: Path) -> None:
    bundle, claims_text = _fake_bundle(tmp_path)
    payload = _build_gate(
        tmp_path,
        bundle=bundle,
        claims_text=claims_text,
        doctor_output=_doctor_output(),
    )
    decoded = json.loads(l5_v2_tt5l_lightning_non_dry_run_gate_json(payload))
    report = render_l5_v2_tt5l_lightning_non_dry_run_gate_markdown(payload)
    assert decoded["schema"] == L5V2_TT5L_LIGHTNING_NON_DRY_RUN_GATE_SCHEMA
    assert "spend-readiness only" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report
    assert "score_claim: `false`" in report

    repo_root = Path.cwd()
    bundle_path = tmp_path / ".omx/research/bundle.json"
    doctor_plan_path = tmp_path / ".omx/research/doctor_plan.json"
    doctor_output_path = tmp_path / ".omx/research/doctor_output.json"
    claims_path = tmp_path / ".omx/state/active_lane_dispatch_claims.md"
    out_json = tmp_path / "out/gate.json"
    out_md = tmp_path / "out/gate.md"
    proc = subprocess.run(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "tools/check_l5_v2_tt5l_lightning_non_dry_run_gate.py"),
            "--repo-root",
            str(tmp_path),
            "--bundle-json",
            str(bundle_path),
            "--doctor-plan-json",
            str(doctor_plan_path),
            "--doctor-output-json",
            str(doctor_output_path),
            "--claims-md",
            str(claims_path),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--current-head-commit",
            COMMIT,
            "--strict-ready",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out_json.read_text(encoding="utf-8"))[
        "ready_for_non_dry_run_submit"
    ] is True
    assert "ready_for_non_dry_run_submit=True" in proc.stdout
