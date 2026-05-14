# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tac.composition.stack_of_stacks import (
    InnerStackSpec,
    MiddleStackSpec,
    OuterStackSpec,
    compose_stack_of_stacks,
)

REPO = Path(__file__).resolve().parents[3]
RECIPE = (
    REPO
    / ".omx/operator_authorize_recipes/substrate_stack_of_stacks_modal_a100_dispatch.yaml"
)
TRAINER = REPO / "experiments/train_substrate_stack_of_stacks.py"
REMOTE_DRIVER = REPO / "scripts/remote_lane_substrate_stack_of_stacks.sh"


def _write_fake_base_runtime(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "inflate.py").write_text("# fake base runtime marker\n", encoding="utf-8")
    inflate_sh = path / "inflate.sh"
    inflate_sh.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "data_dir=\"$1\"\n"
        "output_dir=\"$2\"\n"
        "file_list=\"$3\"\n"
        "mkdir -p \"$output_dir\"\n"
        "while IFS= read -r base; do\n"
        "  [ -z \"$base\" ] && continue\n"
        "  cp \"$data_dir/x\" \"$output_dir/${base}.raw\"\n"
        "done < \"$file_list\"\n",
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)


def test_stack_of_stacks_recipe_has_no_missing_local_dispatch_paths() -> None:
    text = RECIPE.read_text(encoding="utf-8")

    assert "dispatch_enabled: false" in text
    assert "scripts/remote_lane_substrate_stack_of_stacks.sh" in text
    assert "submissions/a1/archive.zip" in text
    assert "remote_driver_missing" not in text
    assert "runtime_frame_emission_missing" not in text
    assert "modal_base_archive_mount_missing" not in text
    assert "[MISSING]" not in text
    assert "[LOCAL_ONLY_NOT_MOUNTED_BY_MODAL]" not in text
    assert REMOTE_DRIVER.is_file()
    assert os.access(REMOTE_DRIVER, os.X_OK)


def test_stack_of_stacks_recipe_keeps_full_score_lowering_blockers_explicit() -> None:
    text = RECIPE.read_text(encoding="utf-8")

    assert "score_aware_training_missing" in text
    assert "eval_roundtrip_training_missing" in text
    assert "per_arm_archive_inputs_missing" in text
    assert "multi_arm_frame_stitching_missing" in text
    assert "full_stack_score_lowering_unproven" in text


def test_stack_of_stacks_single_arm_trainer_builds_exact_eval_canary(tmp_path: Path) -> None:
    output_dir = tmp_path / "sos_canary"
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--base-archive",
            str(REPO / "submissions/a1/archive.zip"),
            "--base-runtime-dir",
            str(REPO / "submissions/a1"),
            "--video-path",
            str(REPO / "upstream/videos/0.mkv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "0",
            "--device",
            "cpu",
            "--smoke",
            "--max-pairs",
            "1",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    summary = (output_dir / "stack_of_stacks_compose_summary.json").read_text(
        encoding="utf-8"
    )
    assert '"ready_for_exact_eval_dispatch": true' in summary
    assert '"canary_exact_eval_ready": true' in summary
    assert '"score_lowering_dispatch_ready": false' in summary
    assert '"operator_dispatch_enabled": false' in summary
    assert '"research_only": false' in summary
    assert '"dispatch_blockers": []' in summary
    assert (output_dir / "submission_dir/archive.zip").is_file()
    inflate_sh = output_dir / "submission_dir/inflate.sh"
    assert inflate_sh.is_file()
    inflate_text = inflate_sh.read_text(encoding="utf-8")
    assert "--with brotli==1.1.0" in inflate_text
    assert "--with numpy" in inflate_text
    assert (output_dir / "submission_dir/base_runtime/inflate.sh").is_file()


def test_emitted_runtime_strips_sos_and_delegates_single_arm(tmp_path: Path) -> None:
    from experiments.train_substrate_stack_of_stacks import _build_archive_from_compose

    base_runtime = tmp_path / "base_runtime"
    _write_fake_base_runtime(base_runtime)
    composed, _ = compose_stack_of_stacks(
        middle_stack_spec=MiddleStackSpec(
            inner_specs=(InnerStackSpec(substrate_id="a1", base_bytes=b"BASE_BYTES"),)
        ),
        outer_stack_spec=OuterStackSpec(k=1, per_pair_arm=(0,)),
        n_pairs=1,
    )
    archive = _build_archive_from_compose(
        composed_bytes=composed,
        output_dir=tmp_path / "build",
        base_runtime_dir=base_runtime,
    )
    submission_dir = archive.parent
    output_dir = tmp_path / "inflated"
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("000001\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(submission_dir / "inflate.py"),
            str(submission_dir),
            str(output_dir),
            str(file_list),
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "000001.raw").read_bytes() == b"BASE_BYTES"


def test_remote_driver_closes_dispatch_claim_terminally() -> None:
    text = REMOTE_DRIVER.read_text(encoding="utf-8")

    assert "close_dispatch_claim()" in text
    assert "completed_stack_of_stacks_auth_eval" in text
    assert "failed_stack_of_stacks_rc_" in text
    assert "claim_lane_dispatch.py\" claim" in text
    assert "trap finalize_remote_run EXIT" in text


def test_emitted_runtime_fails_closed_for_mixed_arm_selector(tmp_path: Path) -> None:
    from experiments.train_substrate_stack_of_stacks import _build_archive_from_compose

    base_runtime = tmp_path / "base_runtime"
    _write_fake_base_runtime(base_runtime)
    composed, _ = compose_stack_of_stacks(
        middle_stack_spec=MiddleStackSpec(
            inner_specs=(
                InnerStackSpec(substrate_id="a1", base_bytes=b"ARM0"),
                InnerStackSpec(substrate_id="lapose", base_bytes=b"ARM1"),
            )
        ),
        outer_stack_spec=OuterStackSpec(k=2, per_pair_arm=(0, 1)),
        n_pairs=2,
    )
    archive = _build_archive_from_compose(
        composed_bytes=composed,
        output_dir=tmp_path / "build",
        base_runtime_dir=base_runtime,
    )
    file_list = tmp_path / "file_list.txt"
    file_list.write_text("000001\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(archive.parent / "inflate.py"),
            str(archive.parent),
            str(tmp_path / "inflated"),
            str(file_list),
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "requires a single arm-0 selector" in result.stderr
