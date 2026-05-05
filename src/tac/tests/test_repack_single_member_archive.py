from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "experiments" / "repack_single_member_archive.py"
spec = importlib.util.spec_from_file_location("repack_single_member_archive", TOOL_PATH)
assert spec is not None
tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tool)


def _write_archive(path: Path, member: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(path, "w", strict_timestamps=True) as zf:
        zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def test_repack_preserves_payload_and_renames_single_member(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    output = tmp_path / "output.zip"
    manifest = tmp_path / "manifest.json"
    payload = b"hnerv-payload-bytes"
    _write_archive(source, "0.bin", payload)

    assert (
        tool.main(
            [
            "--input",
            str(source),
            "--output",
            str(output),
            "--member-name",
            "x",
            "--json-out",
            str(manifest),
            ]
        )
        == 0
    )

    with zipfile.ZipFile(output) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        assert [info.filename for info in infos] == ["x"]
        assert infos[0].date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.read("x") == payload

    record = json.loads(manifest.read_text())
    assert record["source_member_name"] == "0.bin"
    assert record["target_member_name"] == "x"
    assert record["source_member_bytes"] == len(payload)
    assert record["score_claim"] is False
    assert record["evidence_grade"] == "byte_repack_only"


def test_repack_rejects_multi_member_archives(tmp_path: Path) -> None:
    source = tmp_path / "multi.zip"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("a", b"a")
        zf.writestr("b", b"b")

    try:
        tool.read_single_member(source)
    except SystemExit as exc:
        assert "expected exactly one" in str(exc)
    else:
        raise AssertionError("multi-member archive was accepted")
