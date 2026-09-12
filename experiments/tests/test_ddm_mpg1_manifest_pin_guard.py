"""Behavioral controls for ddm_mpg1's archive-pin manifest rebind."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments import ddm_ntb2_public as ntb2_public
from experiments import ddm_sj1_joint_admission as joint
from tac.candidate_seal import SealContractError
from tac.decode_wall_clock import validate_receiver_manifest
from tac.receiver_manifest import (
    canonical_receiver_manifest_bytes,
    rebind_receiver_manifest,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "inflate.py").write_text(
        f'ARCHIVE_SHA256 = "{joint.sj1.POINTER_ARCHIVE_SHA256}"\n'
        f"ARCHIVE_BYTES = {joint.sj1.POINTER_ARCHIVE_BYTES}\n",
        encoding="utf-8",
    )
    (root / "inflate.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (root / "MANIFEST.sha256").write_bytes(canonical_receiver_manifest_bytes(root))
    return root


def test_patch_inflate_pins_rebinds_the_inflate_row(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    before = _sha(root / "inflate.py")
    fact = joint.patch_inflate_pins(root, "ab" * 32, 123_456)
    listing = (root / "MANIFEST.sha256").read_text(encoding="utf-8")
    assert before not in listing
    assert f"{_sha(root / 'inflate.py')}  inflate.py\n" in listing
    assert fact["manifest_rebind"]["relative_paths"] == ["inflate.py"]


def test_patch_inflate_pins_noop_keeps_manifest_byte_identical(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    joint.patch_inflate_pins(root, "ab" * 32, 123_456)
    before = (root / "MANIFEST.sha256").read_bytes()
    fact = joint.patch_inflate_pins(root, "ab" * 32, 123_456)
    assert (root / "MANIFEST.sha256").read_bytes() == before
    assert fact["manifest_rebind"]["changed"] is False


def test_patch_inflate_pins_missing_manifest_row_refuses_without_mutation(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    original = (root / "inflate.py").read_bytes()
    manifest = root / "MANIFEST.sha256"
    manifest.write_text(f"{_sha(root / 'inflate.sh')}  inflate.sh\n", encoding="utf-8")
    with pytest.raises(joint.Sj1JointError, match="manifest omits rewritten file"):
        joint.patch_inflate_pins(root, "ab" * 32, 123_456)
    assert (root / "inflate.py").read_bytes() == original


def test_two_rewritten_files_rebind_both_rows(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "inflate.py").write_text("ARCHIVE_SHA256 = " + repr("cd" * 32) + "\nARCHIVE_BYTES = 7\n")
    (root / "inflate.sh").write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
    result = rebind_receiver_manifest(root, ("inflate.py", "inflate.sh"))
    listing = (root / "MANIFEST.sha256").read_text(encoding="utf-8")
    assert f"{_sha(root / 'inflate.py')}  inflate.py\n" in listing
    assert f"{_sha(root / 'inflate.sh')}  inflate.sh\n" in listing
    assert result["relative_paths"] == ["inflate.py", "inflate.sh"]


def test_existing_manifest_validator_accepts_the_patched_tree(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    joint.patch_inflate_pins(root, "ef" * 32, 999)
    validation = validate_receiver_manifest(root)
    assert validation["present"] is True
    assert validation["listed_file_count"] == 2


def test_existing_seal_inputs_manifest_validation_detects_staleness(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "inflate.py").write_text('ARCHIVE_SHA256 = "' + "12" * 32 + '"\nARCHIVE_BYTES = 4\n')
    with pytest.raises(SealContractError, match=r"manifest hash differs.*inflate\.py"):
        validate_receiver_manifest(root)


def test_ntb2_writer_delegates_to_the_shared_canonical_bytes(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    assert ntb2_public.regenerate_manifest(root) == canonical_receiver_manifest_bytes(root)


def test_unrelated_stale_row_is_not_laundered(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "inflate.py").write_text('ARCHIVE_SHA256 = "' + "34" * 32 + '"\nARCHIVE_BYTES = 5\n')
    (root / "inflate.sh").write_text("#!/bin/sh\necho stale-other-row\n", encoding="utf-8")
    before = (root / "MANIFEST.sha256").read_bytes()
    with pytest.raises(SealContractError, match=r"unrelated stale row.*inflate\.sh"):
        rebind_receiver_manifest(root, ("inflate.py",))
    assert (root / "MANIFEST.sha256").read_bytes() == before


def test_unrelated_missing_row_is_not_silently_added(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    manifest = root / "MANIFEST.sha256"
    manifest.write_text(
        f"{_sha(root / 'inflate.py')}  inflate.py\n", encoding="utf-8"
    )
    (root / "inflate.py").write_text(
        'ARCHIVE_SHA256 = "' + "56" * 32 + '"\nARCHIVE_BYTES = 6\n'
    )
    before = manifest.read_bytes()
    with pytest.raises(SealContractError, match="omits unrelated shipped file"):
        rebind_receiver_manifest(root, ("inflate.py",))
    assert manifest.read_bytes() == before
