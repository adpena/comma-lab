# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
    L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA,
)
from tac.optimization.l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan import (
    L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA,
    build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan,
    l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json,
    render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown,
)
from tac.optimizer.exact_readiness import runtime_dependency_manifest


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runtime(root: Path) -> Path:
    runtime = root / "experiments/results/tt5l/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    os.chmod(runtime / "inflate.sh", 0o755)
    (runtime / "inflate.py").write_text("# fixture inflate\n", encoding="utf-8")
    (runtime / "report.txt").write_text(
        "TT5L side-info effect-curve test runtime\n",
        encoding="utf-8",
    )
    inflate_sh_sha = _sha256(runtime / "inflate.sh")
    inflate_py_sha = _sha256(runtime / "inflate.py")
    (runtime / "runtime_packet_manifest.json").write_text(
        json.dumps(
            {
                "schema": "pr101_kaggle_proxy_runtime_packet_v1",
                "packet_dir": runtime.relative_to(root).as_posix(),
                "runtime_custody": {
                    "runtime_files": [
                        {"relpath": "inflate.sh", "sha256": inflate_sh_sha},
                        {"relpath": "inflate.py", "sha256": inflate_py_sha},
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return runtime


def _write_runtime_consumption_proof(
    root: Path,
    runtime: Path,
    archive_sha256: str,
    proof_path: Path,
) -> None:
    inflate_sh_sha = _sha256(runtime / "inflate.sh")
    inflate_py_sha = _sha256(runtime / "inflate.py")
    manifest_path = runtime / "runtime_packet_manifest.json"
    proof_path.write_text(
        json.dumps(
            {
                "schema": "pr101_kaggle_proxy_runtime_consumption_proof_v1",
                "proof_kind": "fixture_runtime_bound_tt5l_proof",
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": _sha256(manifest_path),
                "packet_dir": runtime.relative_to(root).as_posix(),
                "runtime_consumption_proven_for_supported_bias_params": True,
                "inflate_sh_routes_to_packet_inflate_py": True,
                "archive_unchanged_proof": {"archive_sha256": archive_sha256},
                "inflate_wrapper_route_proof": {
                    "wrapper_invoked_packet_inflate_py": True,
                    "inflate_sh_sha256": inflate_sh_sha,
                    "packet_inflate_py_sha256": inflate_py_sha,
                },
                "inflate_static_bias_patch_proof": {
                    "inflate_sha256": inflate_py_sha,
                },
                "inflate_runtime_bias_logic_proof": {
                    "packet_inflate_function_executed": True,
                    "inflate_py_sha256": inflate_py_sha,
                },
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
                "dispatch_attempted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(root: Path) -> Path:
    runtime = _write_runtime(root)
    runtime_manifest = runtime_dependency_manifest(runtime, root)
    variants = []
    for index, variant in enumerate(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS):
        archive = root / f"experiments/results/tt5l/{variant}/archive.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive, "w") as zf:
            zf.writestr("0.bin", f"archive:{variant}:{index}".encode())
        proof_path = archive.parent / "runtime_consumption_proof.json"
        _write_runtime_consumption_proof(root, runtime, _sha256(archive), proof_path)
        archive_manifest = archive.parent / "archive_manifest.json"
        archive_manifest.write_text(
            json.dumps(
                {
                    "schema": "tt5l_sideinfo_variant_archive_manifest_v1",
                    "archive_path": archive.relative_to(root).as_posix(),
                    "archive_sha256": _sha256(archive),
                    "archive_bytes": archive.stat().st_size,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "members": [{"name": "0.bin"}],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        variants.append(
            {
                "variant": variant,
                "generation_rule": f"test-generation-rule:{variant}",
                "variant_seed": 20260517,
                "archive_path": archive.relative_to(root).as_posix(),
                "archive_sha256": _sha256(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_manifest_path": archive_manifest.relative_to(root).as_posix(),
                "runtime_consumption_proof_path": proof_path.relative_to(root).as_posix(),
                "source_archive_sha256": "1" * 64,
                "source_archive_member_sha256": "2" * 64,
                "source_sideinfo_section_sha256": "3" * 64,
                "sideinfo_changed_from_source": variant != "trained",
                "archive_sha_changed_from_source": True,
                "archive_member_sha_changed_from_source": variant != "trained",
                "sideinfo_section_sha_changed_from_source": variant != "trained",
                "sideinfo_liveness": {
                    "checked": True,
                    "nonzero_values": 0 if variant == "zero" else 8,
                },
                "score_claim": False,
                "promotion_eligible": False,
                "dispatch_attempted": False,
                "blockers": ["requires_paired_cpu_cuda_exact_eval_before_score_claim"],
            }
        )
    manifest = root / ".omx/research/variant_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": L5V2_TT5L_SIDEINFO_VARIANT_PACKET_SCHEMA,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "dispatch_attempted": False,
                "runtime": {
                    "available": True,
                    "submission_dir": runtime.relative_to(root).as_posix(),
                    "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
                    "runtime_content_tree_sha256": runtime_manifest[
                        "runtime_content_tree_sha256"
                    ],
                    "runtime_file_count": runtime_manifest["runtime_file_count"],
                    "blockers": [],
                },
                "required_effect_curve_variants": list(
                    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
                ),
                "variants": variants,
                "blockers": [
                    "requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_tt5l_sideinfo_dispatch_plan_materializes_five_byte_closed_work_units(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    plan = build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
    )

    assert plan["schema"] == L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA
    assert plan["score_claim"] is False
    assert plan["promotion_eligible"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["work_unit_count"] == len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
    assert plan["ready_work_unit_count"] == len(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS)
    assert plan["ready_for_operator_dispatch"] is True
    assert plan["ready_for_provider_dispatch"] is False
    assert {row["variant"] for row in plan["work_units"]} == set(
        L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    )
    for row in plan["work_units"]:
        command = row["dispatch_command_template"]
        execute_command = row["operator_execute_command_template_after_review"]
        assert row["dispatch_command_executable"] is False
        assert row["operator_execute_required"] is True
        assert row["ready_for_operator_dispatch"] is True
        assert row["required_axes"] == ["contest_cpu", "contest_cuda"]
        assert row["required_cells"] == [
            {"axis": "contest_cpu", "variant": row["variant"]},
            {"axis": "contest_cuda", "variant": row["variant"]},
        ]
        assert row["archive"]["expected_sha256_match"] is True
        assert row["archive"]["path"] in command
        assert row["archive"]["sha256"] in command
        assert "tools/dispatch_modal_paired_auth_eval.py" in command
        assert "experiments/modal_auth_eval.py" not in command
        assert "experiments/modal_auth_eval_cpu.py" not in command
        assert "--expected-runtime-tree-sha256 auto" in command
        assert "--skip-axis-if-promotable-anchor-exists" in command
        assert "--execute" not in row["dispatch_command"]
        assert execute_command.endswith(" --execute")
        assert row["pair_group_id"] in command
        assert row["pair_group_id"] in execute_command
        assert row["dispatch_blockers"] == []
        assert row["exact_dispatch_authority"]["authorized"] is True
        assert row["exact_dispatch_authority"]["blockers"] == []
        assert "requires_paired_cpu_cuda_exact_eval_before_score_claim" in row[
            "score_claim_blockers"
        ]


def test_tt5l_sideinfo_dispatch_plan_fails_closed_on_archive_sha_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["variants"][0]["archive_sha256"] = "0" * 64

    plan = build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
    )

    first = plan["work_units"][0]
    assert first["ready_for_operator_dispatch"] is False
    assert "variant_archive_sha_mismatch:zero" in first["dispatch_blockers"]
    assert "paired_dispatch_command_not_materialized" in first["dispatch_blockers"]
    assert plan["ready_for_operator_dispatch"] is False
    assert "variant_archive_sha_mismatch:zero" in plan["blockers"]


def test_tt5l_sideinfo_dispatch_plan_fails_closed_on_missing_noop_custody(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["variants"][1].pop("source_sideinfo_section_sha256")
    payload["variants"][1].pop("sideinfo_section_sha_changed_from_source")

    plan = build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
    )

    row = next(item for item in plan["work_units"] if item["variant"] == "random_lsb")
    assert row["ready_for_operator_dispatch"] is False
    assert (
        "variant_custody_field_missing:random_lsb:source_sideinfo_section_sha256"
        in row["dispatch_blockers"]
    )
    assert (
        "variant_custody_field_missing:random_lsb:"
        "sideinfo_section_sha_changed_from_source"
    ) in row["dispatch_blockers"]
    assert plan["ready_for_operator_dispatch"] is False
    assert (
        "variant_custody_field_missing:random_lsb:source_sideinfo_section_sha256"
        in plan["blockers"]
    )


def test_tt5l_sideinfo_dispatch_plan_requires_exact_dispatch_custody(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    runtime = tmp_path / "experiments/results/tt5l/runtime"
    (runtime / "report.txt").unlink()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    first_manifest = (
        tmp_path
        / str(payload["variants"][0]["archive_manifest_path"])
    )
    first_manifest.unlink()

    plan = build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
    )

    first = plan["work_units"][0]
    assert first["ready_for_operator_dispatch"] is False
    blockers = first["exact_dispatch_authority"]["blockers"]
    assert (
        "exact_dispatch_authority:contest_cpu:archive_manifest_missing"
        in blockers
    )
    assert "exact_dispatch_authority:contest_cpu:report_txt_missing" in blockers
    assert "exact_dispatch_authority:contest_cuda:report_txt_missing" in blockers
    assert plan["ready_for_operator_dispatch"] is False


def test_tt5l_sideinfo_dispatch_plan_json_and_markdown_are_durable(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    plan = build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan(
        manifest=payload,
        manifest_path=manifest,
        repo_root=tmp_path,
    )

    decoded = json.loads(l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_json(plan))
    report = render_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_markdown(plan)

    assert decoded["schema"] == L5V2_TT5L_SIDEINFO_EFFECT_CURVE_DISPATCH_PLAN_SCHEMA
    assert "L5 v2 TT5L side-info effect-curve dispatch plan" in report
    assert "planning_only: `true`" in report
    assert "score_claim: `false`" in report
    assert "ready_for_provider_dispatch: `false`" in report
    assert "operator_execute_command_after_review" in report
    assert "[contest-CPU]" in report
    assert "[contest-CUDA]" in report


def test_tt5l_sideinfo_dispatch_plan_cli_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = root / "experiments/results/time_traveler_l5_v2" / (
        f"test_sideinfo_dispatch_{tmp_path.name}"
    )
    manifest = artifact_root / ".omx/research/variant_manifest.json"
    output_json = artifact_root / "dispatch_plan.json"
    output_md = artifact_root / "dispatch_plan.md"
    try:
        manifest_source = _write_manifest(artifact_root)
        manifest = manifest_source

        proc = subprocess.run(
            [
                str(root / "tools" / "build_l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py"),
                "--variant-manifest",
                str(manifest),
                "--output-json",
                str(output_json.relative_to(root)),
                "--output-md",
                str(output_md.relative_to(root)),
                "--repo-root",
                str(artifact_root),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 0, proc.stdout + proc.stderr
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["work_unit_count"] == len(
            L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        )
        assert payload["ready_work_unit_count"] == len(
            L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        )
        assert "score_claim=false dispatch_attempted=false" in proc.stdout
        assert output_md.is_file()
    finally:
        if artifact_root.exists():
            import shutil

            shutil.rmtree(artifact_root)
