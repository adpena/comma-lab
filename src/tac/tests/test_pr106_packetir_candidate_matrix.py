# SPDX-License-Identifier: MIT
"""Tests for the PR106 PacketIR candidate evidence matrix."""

from __future__ import annotations

import json
from pathlib import Path

from tac.exact_eval_custody import contest_score
from tac.packet_compiler.pr106_candidate_matrix import (
    DEFAULT_PR106_PACKETIR_CANDIDATES,
    PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA,
    PR106PacketIRCandidateSpec,
    build_pr106_packetir_candidate_matrix,
    render_pr106_packetir_candidate_matrix_markdown,
    write_pr106_packetir_candidate_matrix,
)


def _row_by_id(matrix: dict[str, object], candidate_id: str) -> dict[str, object]:
    rows = matrix["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["candidate_id"] == candidate_id:
            return row
    raise AssertionError(candidate_id)


def test_pr106_packetir_candidate_matrix_covers_active_candidates() -> None:
    matrix = build_pr106_packetir_candidate_matrix()

    assert matrix["schema"] == PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA
    assert matrix["candidate_count"] >= 15
    assert matrix["status_counts"] == {
        "paired_exact_blocked": 3,
        "runtime_consumed_needs_paired_exact_eval": 4,
        "single_axis_exact_measured_needs_pair": 9,
    }
    assert matrix["next_exact_eval_target_count"] == 17
    assert matrix["score_claim"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    targets = matrix["next_exact_eval_targets"]
    assert isinstance(targets, list)
    assert all(target["score_claim"] is False for target in targets)
    assert all(target["promotion_eligible"] is False for target in targets)
    assert all(target["ready_for_exact_eval_dispatch"] is False for target in targets)
    rows = matrix["rows"]
    assert isinstance(rows, list)
    assert all(row["score_claim"] is False for row in rows if isinstance(row, dict))
    assert all(
        row["promotion_eligible"] is False for row in rows if isinstance(row, dict)
    )
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in rows
        if isinstance(row, dict)
    )

    format_0c = _row_by_id(matrix, "format_0x0c_exact_radix")
    assert format_0c["format_id"] == "0x0C"
    assert format_0c["status"] == "paired_exact_blocked"
    assert (
        "paired_exact_eval_runtime_consumption_content_tree_sha_missing"
        in format_0c["status_blockers"]
    )
    exact_0c = format_0c["exact_axis_evidence"]
    assert isinstance(exact_0c, dict)
    assert set(exact_0c) == {"contest_cpu", "contest_cuda"}
    assert exact_0c["contest_cpu"]["valid"] is True
    assert exact_0c["contest_cuda"]["valid"] is True

    format_0d = _row_by_id(matrix, "format_0x0d_latent_score_table")
    assert format_0d["format_id"] == "0x0D"
    assert format_0d["packet_ir_identity"]["passed"] is True
    assert format_0d["runtime_consumption"]["valid"] is True
    assert format_0d["status"] == "paired_exact_blocked"
    assert (
        "paired_exact_eval_runtime_content_tree_sha_mismatch_with_consumption"
        in format_0d["status_blockers"]
    )

    prefix_1 = _row_by_id(matrix, "prefix_top_1_pr101grammar")
    assert prefix_1["status"] == "runtime_consumed_needs_paired_exact_eval"
    assert prefix_1["status_blockers"] == [
        "paired_exact_eval_missing:contest_cpu,contest_cuda"
    ]

    r2 = _row_by_id(matrix, "format_0x01_r2_release")
    assert r2["status"] == "runtime_consumed_needs_paired_exact_eval"
    exact_r2 = r2["exact_axis_evidence"]
    assert isinstance(exact_r2, dict)
    assert exact_r2["contest_cpu"]["valid"] is False
    assert "exact_eval_score_formula_mismatch" in exact_r2["contest_cpu"]["blockers"]


def _targets_for_id(
    matrix: dict[str, object], candidate_id: str
) -> list[dict[str, object]]:
    targets = matrix["next_exact_eval_targets"]
    assert isinstance(targets, list)
    out: list[dict[str, object]] = []
    for target in targets:
        assert isinstance(target, dict)
        if target["candidate_id"] == candidate_id:
            out.append(target)
    return out


def test_pr106_packetir_candidate_matrix_emits_fail_fast_exact_eval_targets() -> None:
    matrix = build_pr106_packetir_candidate_matrix()

    r2_targets = _targets_for_id(matrix, "format_0x01_r2_release")
    assert {target["missing_axis"] for target in r2_targets} == {
        "contest_cpu",
        "contest_cuda",
    }
    assert {target["recommended_provider"] for target in r2_targets} == {
        "modal_linux_x86_64_cpu",
        "modal_t4_cuda",
    }
    assert all(
        target["dispatch_status"]
        == "requires_claim_lane_dispatch_before_provider_launch"
        for target in r2_targets
    )
    assert all("--pair-group-id" in target["command_template"] for target in r2_targets)
    assert all("--submission-dir" in target["command_template"] for target in r2_targets)
    assert all(
        "--expected-runtime-tree-sha256" in target["command_template"]
        for target in r2_targets
    )
    assert all(
        target["expected_runtime_tree_sha256_policy"]
        == "compute_axis_specific_modal_uploaded_runtime_tree_sha256"
        for target in r2_targets
    )
    assert all("/Users/" not in target["command_template"] for target in r2_targets)

    format_04_targets = _targets_for_id(matrix, "format_0x04_rank_elided")
    assert len(format_04_targets) == 1
    target = format_04_targets[0]
    assert target["missing_axis"] == "contest_cpu"
    assert target["existing_valid_axes"] == ["contest_cuda"]
    assert target["modal_entrypoint"] == "experiments/modal_auth_eval_cpu.py"
    assert "experiments/modal_auth_eval_cpu.py" in target["command_template"]
    assert "--gpu" not in target["command_template"]

    format_05_targets = _targets_for_id(matrix, "format_0x05_fixed_meta")
    assert len(format_05_targets) == 1
    assert format_05_targets[0]["runtime_dir"] == (
        "submissions/pr106_latent_sidecar_r2_pr101_grammar"
    )
    assert "/Users/" not in format_05_targets[0]["command_template"]


def _fake_exact_eval_artifact(
    tmp_path: Path,
    *,
    axis: str = "contest_cuda",
    hardware: str = "Tesla T4",
    include_runtime_tree: bool = True,
    score_offset: float = 0.0,
) -> Path:
    artifact_dir = tmp_path / "exact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "contest_auth_eval.stdout.log").write_text("ok\n", encoding="utf-8")
    (artifact_dir / "modal_cuda_auth_eval_local_request.json").write_text(
        json.dumps(
            {
                "axis": axis,
                "canonical_path": f"archive.zip -> inflate.sh -> upstream/evaluate.py --device {'cuda' if axis == 'contest_cuda' else 'cpu'}",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive_sha = (
        "9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4"
    )
    archive_bytes = 186_876
    seg_dist = 0.00063198
    pose_dist = 0.00016402
    score = contest_score(seg_dist, pose_dist, archive_bytes) + score_offset
    runtime_manifest: dict[str, object] = {}
    if include_runtime_tree:
        runtime_manifest = {
            "runtime_tree_sha256": "1" * 64,
            "runtime_content_tree_sha256": "2" * 64,
        }
    payload = {
        "schema_version": 1,
        "score_axis": axis,
        "canonical_score": score,
        "score_recomputed_from_components": score,
        "archive_size_bytes": archive_bytes,
        "avg_segnet_dist": seg_dist,
        "avg_posenet_dist": pose_dist,
        "n_samples": 600,
        "evidence_grade": "contest-CUDA" if axis == "contest_cuda" else "contest-CPU",
        "provenance": {
            "archive_sha256": archive_sha,
            "device": "cuda" if axis == "contest_cuda" else "cpu",
            "gpu_model": hardware,
            "inflate_runtime_manifest": runtime_manifest,
        },
    }
    path = artifact_dir / "contest_auth_eval.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _format_0d_spec_with_exact(path: Path) -> PR106PacketIRCandidateSpec:
    base = next(
        spec
        for spec in DEFAULT_PR106_PACKETIR_CANDIDATES
        if spec.candidate_id == "format_0x0d_latent_score_table"
    )
    return PR106PacketIRCandidateSpec(
        candidate_id="spoofed_format_0x0d",
        expected_format_id=base.expected_format_id,
        archive_path=str(Path(base.archive_path).resolve()),
        runtime_consumption_path=str(Path(base.runtime_consumption_path).resolve()),
        exact_eval_paths={"contest_cuda": str(path)},
    )


def test_pr106_packetir_matrix_rejects_spoofed_cuda_hardware(tmp_path: Path) -> None:
    exact_path = _fake_exact_eval_artifact(tmp_path, hardware="cpu")

    matrix = build_pr106_packetir_candidate_matrix(
        repo_root=tmp_path,
        candidates=(_format_0d_spec_with_exact(exact_path),),
    )

    row = _row_by_id(matrix, "spoofed_format_0x0d")
    exact = row["exact_axis_evidence"]
    assert isinstance(exact, dict)
    assert exact["contest_cuda"]["valid"] is False
    assert "exact_eval_hardware_not_cuda" in exact["contest_cuda"]["blockers"]


def test_pr106_packetir_matrix_rejects_missing_runtime_tree(tmp_path: Path) -> None:
    exact_path = _fake_exact_eval_artifact(tmp_path, include_runtime_tree=False)

    matrix = build_pr106_packetir_candidate_matrix(
        repo_root=tmp_path,
        candidates=(_format_0d_spec_with_exact(exact_path),),
    )

    row = _row_by_id(matrix, "spoofed_format_0x0d")
    exact = row["exact_axis_evidence"]
    assert isinstance(exact, dict)
    assert "exact_eval_runtime_tree_sha_invalid" in exact["contest_cuda"]["blockers"]
    assert "exact_eval_runtime_content_tree_sha_invalid" in exact["contest_cuda"]["blockers"]


def test_pr106_packetir_matrix_rejects_score_formula_mismatch(tmp_path: Path) -> None:
    exact_path = _fake_exact_eval_artifact(tmp_path, score_offset=0.01)

    matrix = build_pr106_packetir_candidate_matrix(
        repo_root=tmp_path,
        candidates=(_format_0d_spec_with_exact(exact_path),),
    )

    row = _row_by_id(matrix, "spoofed_format_0x0d")
    exact = row["exact_axis_evidence"]
    assert isinstance(exact, dict)
    assert "exact_eval_score_formula_mismatch" in exact["contest_cuda"]["blockers"]


def test_pr106_packetir_candidate_matrix_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "matrix.json"
    output_md = tmp_path / "matrix.md"

    matrix = write_pr106_packetir_candidate_matrix(
        output_json=output_json,
        output_md=output_md,
    )

    assert output_json.is_file()
    assert output_md.is_file()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["schema"] == PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA
    assert matrix["artifact_paths"]["json"] == str(output_json)
    markdown = output_md.read_text(encoding="utf-8")
    assert "PR106 PacketIR candidate matrix" in markdown
    assert "format_0x0d_latent_score_table" in markdown


def test_pr106_packetir_candidate_matrix_markdown_is_nonpromotional() -> None:
    matrix = build_pr106_packetir_candidate_matrix()
    markdown = render_pr106_packetir_candidate_matrix_markdown(matrix)

    assert "score_claim=false" in markdown
    assert "promotion_eligible=false" in markdown
    assert "contest_cpu" in markdown
    assert "contest_cuda" in markdown
    assert "paired_exact_eval_missing:contest_cpu,contest_cuda" in markdown
    assert "Next exact eval targets" in markdown
    assert "modal_linux_x86_64_cpu" in markdown
    assert "requires_claim_lane_dispatch_before_provider_launch" in markdown
