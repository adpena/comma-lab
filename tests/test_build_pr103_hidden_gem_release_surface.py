from __future__ import annotations

import importlib.util
import json
import stat
import sys
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "build_pr103_hidden_gem_release_surface.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_pr103_hidden_gem_release_surface", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_zip(path: Path) -> tuple[int, str]:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, b"payload", compress_type=zipfile.ZIP_STORED)
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def test_builds_consolidated_release_surface(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "candidate"
    archive_bytes, archive_sha = _write_zip(source / "archive.zip")
    runtime = source / "runtime"
    runtime.mkdir(parents=True)
    (runtime / "inflate.py").write_text("# runtime\n", encoding="utf-8")
    inflate = runtime / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    inflate.chmod(inflate.stat().st_mode | stat.S_IXUSR)
    (source / "report.txt").write_text("report\n", encoding="utf-8")
    _write_json(
        source / "manifest.json",
        {
            "schema_version": 1,
            "candidate_id": "hidden",
            "score_claim": False,
            "dispatch_attempted": False,
            "candidate_archive": {
                "path": str(source / "archive.zip"),
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
            },
            "source_archive": {
                "archive_bytes": archive_bytes + 4,
                "archive_sha256": "a" * 64,
            },
            "charged_byte_proof": {"charged_archive_delta_bytes": -4},
            "section_sha256_proof": {"runtime_consumed_section_changed": True},
            "runtime_consumption_no_op_proof": {"state_dict_changed_vs_source": True},
        },
    )

    manifest = tool.build_release_surface(source_dir=source)
    release = source / "release_surface"

    assert manifest["schema"] == "pr103_hidden_gem_release_surface_manifest_v1"
    assert (release / "archive.zip").read_bytes() == (source / "archive.zip").read_bytes()
    assert (release / "inflate.sh").stat().st_mode & stat.S_IXUSR
    inflate_sh = (release / "inflate.sh").read_text(encoding="utf-8")
    assert ".venv" not in inflate_sh
    assert "$HERE/inflate.py" in inflate_sh
    assert 'PYBIN="python"' in inflate_sh
    assert (release / "archive_manifest.json").is_file()
    assert manifest["score_affecting_payload_changed"] is True
    assert manifest["charged_bits_changed"] is True
