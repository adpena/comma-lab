"""Hardening tests for tools/materialize_magic_codec_archive.py."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.fixture()
def repo_output_dir(tmp_path):
    root = REPO_ROOT / "experiments" / "results" / ".pytest_tmp_outputs" / tmp_path.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    def make(name: str) -> Path:
        return root / name

    yield make
    shutil.rmtree(root, ignore_errors=True)


def _import_tool():
    spec = importlib.util.spec_from_file_location(
        "materialize_magic_codec_archive",
        REPO_ROOT / "tools" / "materialize_magic_codec_archive.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, member: str, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
        zf.writestr(info, payload)


def test_runtime_manifest_declares_not_byte_closed() -> None:
    tool = _import_tool()
    manifest = tool._build_runtime_manifest()
    assert manifest["runtime_tree_byte_closed"] is False
    assert "repo_tac_required_until_vendored" in manifest["runtime_dep_closure"]
    assert "sibling_pr106_latent_sidecar_r2_required" in manifest["runtime_dep_closure"]
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_refuses_tail_truncation_for_wide_quantization() -> None:
    tool = _import_tool()
    with pytest.raises(SystemExit, match="refusing to truncate tail bytes"):
        tool._decode_dense_from_member_bytes(
            b"abc", stream_type="weight_tensor", quantize_bits=16
        )


def test_refuses_runtime_unsupported_primitive() -> None:
    tool = _import_tool()
    with pytest.raises(tool.MagicCodecError, match="not supported"):
        tool._process_member(
            "0.bin",
            bytes([1]) * 64,
            stream_type="low_pass_residual",
            selection_strategy="smallest_byte_count",
            quantize_bits=8,
        )


def test_materializes_supported_member_under_original_name(
    tmp_path: Path,
    repo_output_dir,
) -> None:
    tool = _import_tool()
    source = tmp_path / "source.zip"
    _write_zip(source, "0.bin", bytes([0]) * 64)
    out_dir = repo_output_dir("out")

    rc = tool.main(
        [
            "--source-archive",
            str(source),
            "--output-dir",
            str(out_dir),
            "--stream-type",
            "residual_basis",
            "--selection-strategy",
            "smallest_byte_count",
        ]
    )

    assert rc == 0
    archive = out_dir / "magic_codec_archive.zip"
    with zipfile.ZipFile(archive, "r") as zf:
        assert zf.namelist() == ["0.bin"]
        assert zf.read("0.bin").startswith(b"MAGC")
    manifest = json.loads((out_dir / "magic_codec_selection_manifest.json").read_text())
    row = manifest["member_rows"][0]
    assert row["member_name"] == "0.bin"
    assert row["output_member_name"] == "0.bin"
    assert row["selected_primitive_id"] == 0xF0
