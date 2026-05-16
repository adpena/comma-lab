# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from tac.deploy.modal.auth_eval import (
    ModalAuthEvalPairingError,
    validate_modal_auth_eval_pairing,
)


def _load_tool():
    repo_root = Path(__file__).resolve().parents[3]
    tool_path = repo_root / "tools" / "dispatch_modal_paired_auth_eval.py"
    spec = importlib.util.spec_from_file_location("dispatch_modal_paired_auth_eval", tool_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pairing_contract_requires_pair_or_waiver() -> None:
    with pytest.raises(ModalAuthEvalPairingError, match="paired-by-default"):
        validate_modal_auth_eval_pairing(
            axis="contest_cuda",
            pair_group_id="",
            single_axis_waiver_reason="",
        )

    paired = validate_modal_auth_eval_pairing(
        axis="contest_cuda",
        pair_group_id="pair_001",
        single_axis_waiver_reason="",
    )
    assert paired["paired_axis_required"] is True
    assert paired["pair_group_id"] == "pair_001"
    assert paired["single_axis_waiver_used"] is False


def test_paired_modal_plan_emits_cpu_and_cuda_commands_with_same_pair_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")

    monkeypatch.setattr(
        mod,
        "find_promotable_anchor_for_axis_and_sha",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mod,
        "_resolve_axis_runtime_expectations",
        lambda **_kwargs: {
            "contest_cuda": "",
            "contest_cpu": "",
            "observed_uploaded_runtime": {},
            "common_expected_runtime_tree_sha256": None,
        },
    )

    plan = mod.build_plan(
        archive=archive,
        submission_dir="submissions/pr106_latent_sidecar_r2_pr101_grammar",
        inflate_sh="inflate.sh",
        run_id="unit_pair_run",
        pair_group_id="unit_pair_group",
        lane_id_base="lane_unit_pair",
        output_root=Path("experiments/results"),
        modal_bin=".venv/bin/modal",
        gpu="T4",
        claim_agent="codex:test",
        claim_notes="unit test",
    )

    assert plan["required_axes"] == ["contest_cuda", "contest_cpu"]
    assert plan["pair_group_id"] == "unit_pair_group"
    cuda_cmd = plan["commands"]["contest_cuda"]
    cpu_cmd = plan["commands"]["contest_cpu"]
    expected_archive_sha = mod._sha256(archive)
    assert plan["archive"]["expected_sha256"] == expected_archive_sha
    assert plan["archive"]["expected_sha256_match"] is True
    for cmd in (cuda_cmd, cpu_cmd):
        assert cmd[cmd.index("--pair-group-id") + 1] == "unit_pair_group"
        assert cmd[cmd.index("--archive") + 1] == str(archive.resolve())
        assert cmd[cmd.index("--expected-archive-sha256") + 1] == expected_archive_sha
        assert "--single-axis-waiver-reason" not in cmd
    assert "experiments/modal_auth_eval.py" in cuda_cmd
    assert "experiments/modal_auth_eval_cpu.py" in cpu_cmd


def test_paired_modal_plan_relativizes_inflate_sh_under_submission_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    submission_dir = tmp_path / "candidate" / "submission_dir"
    submission_dir.mkdir(parents=True)
    inflate_sh = submission_dir / "inflate.sh"
    inflate_sh.write_text("#!/bin/sh\n")

    monkeypatch.setattr(
        mod,
        "find_promotable_anchor_for_axis_and_sha",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mod,
        "_resolve_axis_runtime_expectations",
        lambda **_kwargs: {
            "contest_cuda": "",
            "contest_cpu": "",
            "observed_uploaded_runtime": {},
            "common_expected_runtime_tree_sha256": None,
        },
    )

    plan = mod.build_plan(
        archive=archive,
        submission_dir=str(submission_dir),
        inflate_sh=str(inflate_sh),
        run_id="unit_pair_run",
        pair_group_id="unit_pair_group",
        lane_id_base="lane_unit_pair",
        output_root=Path("experiments/results"),
        modal_bin=".venv/bin/modal",
        gpu="T4",
        claim_agent="codex:test",
        claim_notes="unit test",
    )

    assert plan["runtime"]["inflate_sh"] == "inflate.sh"
    assert plan["runtime"]["inflate_sh_original"] == str(inflate_sh)
    for cmd in (plan["commands"]["contest_cuda"], plan["commands"]["contest_cpu"]):
        assert cmd[cmd.index("--inflate-sh") + 1] == "inflate.sh"


def test_cli_defaults_inflate_sh_to_uploaded_submission_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "inflate.sh").write_text("#!/bin/sh\n")
    json_out = tmp_path / "plan.json"

    monkeypatch.setattr(
        mod,
        "find_promotable_anchor_for_axis_and_sha",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mod,
        "_resolve_axis_runtime_expectations",
        lambda **_kwargs: {
            "contest_cuda": "",
            "contest_cpu": "",
            "observed_uploaded_runtime": {},
            "common_expected_runtime_tree_sha256": None,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dispatch_modal_paired_auth_eval.py",
            "--archive",
            str(archive),
            "--submission-dir",
            str(submission_dir),
            "--run-id",
            "unit_pair_run",
            "--pair-group-id",
            "unit_pair_group",
            "--lane-id-base",
            "lane_unit_pair",
            "--json-out",
            str(json_out),
        ],
    )

    assert mod.main() == 0
    plan = json.loads(json_out.read_text(encoding="utf-8"))
    assert plan["runtime"]["submission_dir"] == str(submission_dir)
    assert plan["runtime"]["inflate_sh"] == "inflate.sh"
    assert plan["runtime"]["inflate_sh_original"] is None
    for cmd in (plan["commands"]["contest_cuda"], plan["commands"]["contest_cpu"]):
        assert cmd[cmd.index("--inflate-sh") + 1] == "inflate.sh"
        assert mod.DEFAULT_REPO_INFLATE_SH not in cmd


def test_paired_modal_plan_can_skip_existing_cuda_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")

    def fake_anchor(
        axis: str,
        archive_sha256: str,
        *,
        repo_root: Path,
        expected_runtime_tree_sha256: str = "",
    ):
        assert expected_runtime_tree_sha256 == "a" * 64
        if axis == "cuda":
            return {
                "axis": "contest_cuda",
                "archive_sha256": archive_sha256,
                "score": 0.2,
                "runtime_tree_sha256": expected_runtime_tree_sha256,
                "result_path": "existing/cuda.json",
                "source": "filesystem_scan",
                "custody_match": True,
            }
        return None

    monkeypatch.setattr(mod, "find_promotable_anchor_for_axis_and_sha", fake_anchor)
    monkeypatch.setattr(
        mod,
        "_resolve_axis_runtime_expectations",
        lambda **_kwargs: {
            "contest_cuda": "a" * 64,
            "contest_cpu": "a" * 64,
            "observed_uploaded_runtime": {},
            "common_expected_runtime_tree_sha256": "a" * 64,
        },
    )

    plan = mod.build_plan(
        archive=archive,
        submission_dir="submissions/pr106_latent_sidecar_r2_pr101_grammar",
        inflate_sh="inflate.sh",
        run_id="unit_pair_run",
        pair_group_id="unit_pair_group",
        lane_id_base="lane_unit_pair",
        output_root=Path("experiments/results"),
        modal_bin=".venv/bin/modal",
        gpu="T4",
        claim_agent="codex:test",
        claim_notes="unit test",
        expected_runtime_tree_sha256="a" * 64,
        skip_axis_if_promotable_anchor_exists=True,
        repo_root=tmp_path,
    )

    assert plan["skip_axis_if_promotable_anchor_exists"] is True
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cuda"] is True
    assert plan["axes_skipped_due_to_existing_anchor"]["contest_cpu"] is False
    assert plan["existing_anchors_reused"]["contest_cuda"]["score"] == 0.2


def test_execute_skip_records_terminal_claims_and_repointer_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    output_root = tmp_path / "results"
    runtime_hash = "a" * 64
    subprocess_calls: list[list[str]] = []

    def fake_anchor(
        axis: str,
        archive_sha256: str,
        *,
        repo_root: Path,
        expected_runtime_tree_sha256: str = "",
    ):
        assert repo_root == tmp_path
        assert expected_runtime_tree_sha256 == runtime_hash
        return {
            "axis": f"contest_{axis}",
            "archive_sha256": archive_sha256,
            "score": 0.19 if axis == "cuda" else 0.18,
            "runtime_tree_sha256": runtime_hash,
            "result_path": f"existing/{axis}.json",
            "source": "unit_anchor_scan",
            "custody_match": True,
        }

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and "tools/claim_lane_dispatch.py" in " ".join(map(str, cmd)):
            return original_subprocess_run(cmd, *args, **kwargs)
        subprocess_calls.append(list(cmd))
        raise AssertionError(f"provider command should not run when anchors are reused: {cmd!r}")

    original_subprocess_run = mod.subprocess.run
    monkeypatch.setattr(mod, "find_promotable_anchor_for_axis_and_sha", fake_anchor)
    monkeypatch.setattr(
        mod,
        "_resolve_axis_runtime_expectations",
        lambda **_kwargs: {
            "contest_cuda": runtime_hash,
            "contest_cpu": runtime_hash,
            "observed_uploaded_runtime": {},
            "common_expected_runtime_tree_sha256": runtime_hash,
        },
    )
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dispatch_modal_paired_auth_eval.py",
            "--archive",
            str(archive),
            "--submission-dir",
            "submissions/pr106_latent_sidecar_r2_pr101_grammar",
            "--inflate-sh",
            "inflate.sh",
            "--run-id",
            "unit_pair_run",
            "--pair-group-id",
            "unit_pair_group",
            "--lane-id-base",
            "lane_unit_pair",
            "--output-root",
            str(output_root),
            "--repo-root",
            str(tmp_path),
            "--expected-runtime-tree-sha256",
            runtime_hash,
            "--skip-axis-if-promotable-anchor-exists",
            "--execute",
        ],
    )

    assert mod.main() == 0
    assert subprocess_calls == []

    claims_text = (
        tmp_path / ".omx" / "state" / "active_lane_dispatch_claims.md"
    ).read_text(encoding="utf-8")
    assert "lane_unit_pair_contest_cuda" in claims_text
    assert "lane_unit_pair_contest_cpu" in claims_text
    assert claims_text.count("completed_reused_existing_anchor") == 2

    for axis in ("contest_cuda", "contest_cpu"):
        repointer = (
            output_root
            / ("modal_auth_eval" if axis == "contest_cuda" else "modal_auth_eval_cpu")
            / f"unit_pair_run_{'cuda' if axis == 'contest_cuda' else 'cpu'}"
            / f"anchor_repointer_{axis}.json"
        )
        payload = json.loads(repointer.read_text(encoding="utf-8"))
        assert payload["terminal_dispatch_claim"]["lane_id"] == f"lane_unit_pair_{axis}"
        assert (
            payload["terminal_dispatch_claim"]["status"]
            == "completed_reused_existing_anchor"
        )
        assert payload["reused_anchor"]["source"] == "unit_anchor_scan"


def test_paired_modal_plan_auto_computes_axis_specific_uploaded_runtime_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/bin/sh\n")
    (runtime / "inflate.py").write_text("print('ok')\n")

    cuda_hash = "c" * 64
    cpu_hash = "d" * 64

    def fake_uploaded_hashes(*, remote_submission_dir: str, **_kwargs):
        if remote_submission_dir == mod.CUDA_REMOTE_SUBMISSION_DIR:
            return {
                "runtime_tree_sha256": cuda_hash,
                "runtime_content_tree_sha256": "e" * 64,
            }
        if remote_submission_dir == mod.CPU_REMOTE_SUBMISSION_DIR:
            return {
                "runtime_tree_sha256": cpu_hash,
                "runtime_content_tree_sha256": "e" * 64,
            }
        raise AssertionError(remote_submission_dir)

    monkeypatch.setattr(mod, "_modal_uploaded_runtime_hashes_for_axis", fake_uploaded_hashes)
    monkeypatch.setattr(
        mod,
        "find_promotable_anchor_for_axis_and_sha",
        lambda *_args, **_kwargs: None,
    )

    plan = mod.build_plan(
        archive=archive,
        submission_dir=str(runtime),
        inflate_sh=str(runtime / "inflate.sh"),
        run_id="unit_pair_run",
        pair_group_id="unit_pair_group",
        lane_id_base="lane_unit_pair",
        output_root=Path("experiments/results"),
        modal_bin=".venv/bin/modal",
        gpu="T4",
        claim_agent="codex:test",
        claim_notes="unit test",
    )

    assert plan["runtime"]["expected_runtime_tree_sha256"] is None
    assert plan["runtime"]["expected_runtime_tree_sha256_by_axis"] == {
        "contest_cuda": cuda_hash,
        "contest_cpu": cpu_hash,
    }
    assert (
        plan["commands"]["contest_cuda"][
            plan["commands"]["contest_cuda"].index("--expected-runtime-tree-sha256") + 1
        ]
        == cuda_hash
    )
    assert (
        plan["commands"]["contest_cpu"][
            plan["commands"]["contest_cpu"].index("--expected-runtime-tree-sha256") + 1
        ]
        == cpu_hash
    )


def test_paired_modal_plan_rejects_legacy_single_hash_before_provider_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/bin/sh\n")
    (runtime / "inflate.py").write_text("print('ok')\n")

    monkeypatch.setattr(
        mod,
        "_modal_uploaded_runtime_hashes_for_axis",
        lambda **_kwargs: {
            "runtime_tree_sha256": "b" * 64,
            "runtime_content_tree_sha256": "c" * 64,
        },
    )

    with pytest.raises(ValueError, match="expected runtime tree does not match"):
        mod.build_plan(
            archive=archive,
            submission_dir=str(runtime),
            inflate_sh="inflate.sh",
            run_id="unit_pair_run",
            pair_group_id="unit_pair_group",
            lane_id_base="lane_unit_pair",
            output_root=Path("experiments/results"),
            modal_bin=".venv/bin/modal",
            gpu="T4",
            claim_agent="codex:test",
            claim_notes="unit test",
            expected_runtime_tree_sha256="a" * 64,
        )


def test_paired_modal_plan_rejects_expected_archive_sha_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_tool()
    archive = tmp_path / "archive.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("x", b"payload")

    with pytest.raises(ValueError, match="expected archive sha does not match"):
        mod.build_plan(
            archive=archive,
            submission_dir="",
            inflate_sh="inflate.sh",
            run_id="unit_pair_run",
            pair_group_id="unit_pair_group",
            lane_id_base="lane_unit_pair",
            output_root=Path("experiments/results"),
            modal_bin=".venv/bin/modal",
            gpu="T4",
            claim_agent="codex:test",
            claim_notes="unit test",
            expected_archive_sha256="0" * 64,
        )
