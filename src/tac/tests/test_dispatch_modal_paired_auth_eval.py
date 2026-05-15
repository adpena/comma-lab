# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
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
    for cmd in (cuda_cmd, cpu_cmd):
        assert cmd[cmd.index("--pair-group-id") + 1] == "unit_pair_group"
        assert cmd[cmd.index("--archive") + 1] == str(archive.resolve())
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
