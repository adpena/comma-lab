from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO / "experiments" / "build_c091_relative_pr75_pr77_action_pose_matrix.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("c091_relative_matrix_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, name: str, payload: bytes, *, stored: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        info.create_system = 3
        zf.writestr(info, payload)


def test_break_even_sub314_matches_c091_rate_math() -> None:
    builder = _load_builder()

    row = builder._break_even(276_329)

    assert row["archive_delta_bytes_vs_c091"] == -152
    assert row["score_if_components_unchanged"] == pytest.approx(
        0.31516575028285976 - 152 * 25.0 / 37_545_489
    )
    assert row["sub314_component_score_improvement_needed"] == pytest.approx(
        0.0010645397219851693
    )
    assert row["sub314_equivalent_bytes_needed_after_candidate"] == 1599


def test_archive_profile_requires_single_stored_p_member(tmp_path: Path) -> None:
    builder = _load_builder()
    ok = tmp_path / "ok.zip"
    bad_name = tmp_path / "bad-name.zip"
    bad_compression = tmp_path / "bad-compression.zip"
    _write_zip(ok, "p", b"payload")
    _write_zip(bad_name, "payload.bin", b"payload")
    _write_zip(bad_compression, "p", b"payload", stored=False)

    assert builder._archive_profile(ok)["status"] == "passed"
    assert builder._archive_profile(bad_name)["status"] == "failed"
    assert builder._archive_profile(bad_compression)["status"] == "failed"


def test_archive_profile_rejects_zip_slip(tmp_path: Path) -> None:
    builder = _load_builder()
    path = tmp_path / "zipslip.zip"
    _write_zip(path, "../p", b"payload")

    with pytest.raises(ValueError, match="unsafe archive member"):
        builder._archive_profile(path)
