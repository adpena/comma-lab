# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.deploy.modal.static_manifest import (
    validate_static_manifest_covers_trainer_metadata,
)


def test_static_manifest_accepts_required_input_file_covered_by_parent_dir(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "train_demo.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
        "    '--video-path': {\n"
        "        'default': 'upstream/videos/0.mkv',\n"
        "        'required_input_file': True,\n"
        "    },\n"
        "}\n"
    )
    manifest = (
        {"kind": "dir", "local_path": "upstream/videos"},
    )

    assert (
        validate_static_manifest_covers_trainer_metadata(
            manifest, trainer_path=trainer, repo_root=tmp_path
        )
        == []
    )


def test_static_manifest_rejects_missing_required_input_file_coverage(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "train_demo.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "TIER_1_OPERATOR_REQUIRED_FLAGS = {\n"
        "    '--profile': {\n"
        "        'default': '.omx/research/profile.json',\n"
        "        'required_input_file': True,\n"
        "    },\n"
        "}\n"
    )
    manifest = (
        {"kind": "dir", "local_path": "src"},
    )

    violations = validate_static_manifest_covers_trainer_metadata(
        manifest, trainer_path=trainer, repo_root=tmp_path
    )

    assert len(violations) == 1
    assert "--profile required_input_file" in violations[0]
    assert ".omx/research/profile.json" in violations[0]


def test_static_manifest_rejects_missing_extra_mount_coverage(tmp_path: Path) -> None:
    trainer = tmp_path / "experiments" / "train_demo.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "TIER_1_EXTRA_MOUNT_PATHS = ('experiments/results/base/archive.zip',)\n"
    )
    manifest = (
        {"kind": "dir", "local_path": "experiments"},
    )

    assert (
        validate_static_manifest_covers_trainer_metadata(
            manifest, trainer_path=trainer, repo_root=tmp_path
        )
        == []
    )

    violations = validate_static_manifest_covers_trainer_metadata(
        (), trainer_path=trainer, repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "TIER_1_EXTRA_MOUNT_PATHS" in violations[0]


def test_static_manifest_flags_unresolved_metadata(tmp_path: Path) -> None:
    trainer = tmp_path / "experiments" / "train_demo.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "TIER_1_EXTRA_MOUNT_PATHS = (str(SOME_PATH.relative_to(REPO_ROOT)),)\n"
    )

    violations = validate_static_manifest_covers_trainer_metadata(
        (), trainer_path=trainer, repo_root=tmp_path
    )

    assert len(violations) == 1
    assert "unresolved_metadata:TIER_1_EXTRA_MOUNT_PATHS" in violations[0]
