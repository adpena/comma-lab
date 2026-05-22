# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tools.build_hfv1_sparse_sidecar_candidate import (
    DEFAULT_SUBMISSION_DIR,
    _patch_inflate_py,
    _validate_hfv2_archive_members,
    _write_stored_zip,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OLD_HFV2_RUNTIME = REPO_ROOT / "experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/submission_dir_hfv2/inflate.py"


def test_hfv2_sparse_builder_patches_hfv1_runtime_to_magic_bin_dispatch() -> None:
    source = (REPO_ROOT / DEFAULT_SUBMISSION_DIR / "inflate.py").read_text(
        encoding="utf-8"
    )

    patched = _patch_inflate_py(source)

    assert 'HFV2_MAGIC = b"HFV2"' in patched
    assert "foveation_params.bin unknown magic" in patched
    assert 'src_bin.with_name("foveation_params.hfv2")' not in patched
    assert "row_by_frame.get(int(frame_index), default_row)" in patched
    compile(patched, "patched_hfv2_inflate.py", "exec")
    assert _patch_inflate_py(patched) == patched


def test_hfv2_sparse_builder_migrates_old_hfv2_runtime_to_magic_bin_dispatch() -> None:
    source = OLD_HFV2_RUNTIME.read_text(encoding="utf-8")
    assert 'src_bin.with_name("foveation_params.hfv2")' in source

    patched = _patch_inflate_py(source)

    assert 'src_bin.with_name("foveation_params.hfv2")' not in patched
    assert "foveation_params.bin unknown magic" in patched
    assert "row_by_frame.get(int(frame_index), default_row)" in patched
    compile(patched, "patched_old_hfv2_inflate.py", "exec")
    assert _patch_inflate_py(patched) == patched


def test_hfv2_sparse_builder_enforces_exact_archive_member_shape(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_stored_zip(archive, [("foveation_params.bin", b"HFV2"), ("x", b"x")])

    _validate_hfv2_archive_members(archive)

    bad = tmp_path / "bad.zip"
    _write_stored_zip(
        bad,
        [("foveation_params.bin", b"HFV2"), ("x", b"x"), ("debug.json", b"{}")],
    )
    with pytest.raises(ValueError, match="member order mismatch"):
        _validate_hfv2_archive_members(bad)
