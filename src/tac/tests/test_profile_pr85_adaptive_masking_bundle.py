from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_adaptive_masking_bundle.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("profile_pr85_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _write_fixture_archive(path: Path) -> dict[str, bytes]:
    segments = {
        "mask": b"QMA9" + b"m" * 1001,
        "model": b"model" + b"a" * 1001,
        "pose": b"pose" + b"b" * 101,
        "post": b"post" + b"c" * 7,
        "shift": b"shift" + b"d",
        "frac": b"frac" + b"e",
        "frac2": b"frac2" + b"f",
        "frac3": b"frac3" + b"g",
        "bias": b"b" * 223,
        "region": b"r" * 273,
        "randmulti": b"randmulti-tail",
    }
    header = b"".join(
        _u24(len(segments[name]))
        for name in ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3")
    )
    raw = header + b"".join(segments[name] for name in module.SEGMENT_ORDER)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as zf:
        zf.writestr("x", raw)
    return segments


module = _load_script()


def test_pr85_profiler_parses_v5_micro_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    segments = _write_fixture_archive(archive)

    payload = module.profile_archive(archive)

    assert payload["score_claim"] is False
    assert payload["bundle_format"] == "pr85_v5_micro_24bit_lengths_fixed_bias_region"
    rows = {row["name"]: row for row in payload["segments"]}
    assert rows["mask"]["bytes"] == len(segments["mask"])
    assert rows["mask"]["magic_ascii"] == "QMA9mmmm"
    assert rows["bias"]["bytes"] == 223
    assert rows["region"]["bytes"] == 273
    assert rows["randmulti"]["bytes"] == len(segments["randmulti"])


def test_pr85_profiler_cli_writes_json(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "archive.zip"
    out = tmp_path / "profile.json"
    _write_fixture_archive(archive)

    assert module.main(["--archive", str(archive), "--json-out", str(out)]) == 0

    assert json.loads(out.read_text())["archive"]["member_name"] == "x"
    assert '"score_claim": false' in capsys.readouterr().out
