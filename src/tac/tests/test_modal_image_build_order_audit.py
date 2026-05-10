from __future__ import annotations

from pathlib import Path

from tac.source_index import source_index_context
from tools.audit_modal_image_build_order import (
    audit_modal_image_build_order,
    modal_image_candidate_paths,
)


def _write_repo(root: Path, source: str) -> None:
    (root / "experiments").mkdir()
    (root / "experiments" / "modal_demo.py").write_text(source, encoding="utf-8")


def _init_git(root: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)


def test_modal_image_build_order_accepts_env_before_local_mounts(tmp_path: Path) -> None:
    _write_repo(
        tmp_path,
        """
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch")
    .env({"DALI_DISABLE_NVML": "1", "PYTHONPATH": "/workspace/pact/src"})
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_file("pyproject.toml", remote_path="/workspace/pyproject.toml")
)
""",
    )
    _init_git(tmp_path)

    payload = audit_modal_image_build_order(tmp_path)

    assert payload["violation_count"] == 0


def test_modal_image_build_order_blocks_build_steps_after_local_mounts(tmp_path: Path) -> None:
    _write_repo(
        tmp_path,
        """
import modal

image = (
    modal.Image.debian_slim()
    .add_local_file("pyproject.toml", remote_path="/workspace/pyproject.toml")
    .env({"DALI_DISABLE_NVML": "1", "PYTHONPATH": "/workspace/pact/src"})
    .run_commands("echo late")
)
""",
    )
    _init_git(tmp_path)

    payload = audit_modal_image_build_order(tmp_path)

    assert payload["violation_count"] == 2
    methods = {row["method"] for row in payload["violations"]}
    assert methods == {"env", "run_commands"}


def test_modal_image_build_order_blocks_src_mount_without_pythonpath(tmp_path: Path) -> None:
    _write_repo(
        tmp_path,
        """
import modal

image = (
    modal.Image.debian_slim()
    .env({"DALI_DISABLE_NVML": "1"})
    .add_local_dir("src", remote_path="/workspace/pact/src")
)
""",
    )
    _init_git(tmp_path)

    payload = audit_modal_image_build_order(tmp_path)

    assert payload["violation_count"] == 1
    assert payload["violations"][0]["method"] == "PYTHONPATH"


def test_modal_image_candidate_filter_keeps_tracked_boundary(tmp_path: Path) -> None:
    _write_repo(
        tmp_path,
        """
def ordinary_helper():
    return "no modal image here"
""",
    )
    _init_git(tmp_path)
    (tmp_path / "experiments" / "untracked_modal.py").write_text(
        """
import modal

image = modal.Image.debian_slim().add_local_file("pyproject.toml")
""",
        encoding="utf-8",
    )

    checked, candidates, violations = modal_image_candidate_paths(tmp_path)

    assert checked == 1
    assert candidates == []
    assert violations == []


def test_modal_image_order_source_index_matches_plain_scan(tmp_path: Path) -> None:
    _write_repo(
        tmp_path,
        """
import modal

image = (
    modal.Image.debian_slim()
    .add_local_dir("src", remote_path="/workspace/pact/src")
    .run_commands("echo late")
)
""",
    )
    _init_git(tmp_path)

    plain_payload = audit_modal_image_build_order(tmp_path)
    with source_index_context(tmp_path) as source_index:
        indexed_payload = audit_modal_image_build_order(tmp_path, source_index=source_index)

    assert indexed_payload == plain_payload
