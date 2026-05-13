from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "probe_inflate_shell_output_parity.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("probe_inflate_shell_output_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_packet(root: Path, *, payload: bytes, output_suffix: str = "") -> tuple[Path, Path]:
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload)
    inflate = root / "inflate.sh"
    inflate.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "DATA_DIR=\"$1\"",
                "OUTPUT_DIR=\"$2\"",
                "FILE_LIST=\"$3\"",
                "mkdir -p \"$OUTPUT_DIR\"",
                "while IFS= read -r line; do",
                "  [ -z \"$line\" ] && continue",
                "  base=\"${line%.*}\"",
                f"  printf '%s{output_suffix}' \"$(cat \"$DATA_DIR/x\")\" > \"$OUTPUT_DIR/${{base}}.raw\"",
                "done < \"$FILE_LIST\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inflate.chmod(0o755)
    return archive, inflate


def _write_python3_packet(root: Path, *, payload: bytes) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    archive = root / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", payload)
    inflate = root / "inflate.sh"
    inflate.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "DATA_DIR=\"$1\"",
                "OUTPUT_DIR=\"$2\"",
                "FILE_LIST=\"$3\"",
                "mkdir -p \"$OUTPUT_DIR\"",
                "while IFS= read -r line; do",
                "  [ -z \"$line\" ] && continue",
                "  base=\"${line%.*}\"",
                "  python3 \"$DATA_DIR/x\" \"$OUTPUT_DIR/${base}.raw\"",
                "done < \"$FILE_LIST\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inflate.chmod(0o755)
    return archive, inflate


def _write_fake_python(root: Path) -> Path:
    fake_python = root / "fake-python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cat \"$1\" > \"$2\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_python


def test_inflate_shell_parity_probe_passes_for_identical_outputs(tmp_path: Path) -> None:
    mod = _load_module()
    source_archive, source_inflate = _write_packet(tmp_path / "source", payload=b"payload")
    candidate_archive, candidate_inflate = _write_packet(
        tmp_path / "candidate",
        payload=b"payload",
    )

    report = mod.build_report(
        mod.parse_args(
            [
                "--source-archive",
                str(source_archive),
                "--source-inflate-sh",
                str(source_inflate),
                "--candidate-archive",
                str(candidate_archive),
                "--candidate-inflate-sh",
                str(candidate_inflate),
            ]
        ),
        raw_argv=[],
    )

    assert report["passed"] is True
    assert report["parity_method"] == "exact_inflate_sh_archive_dir_output_dir_file_list"
    assert report["source"]["outputs"] == report["candidate"]["outputs"]
    assert report["output_mismatches"] == []


def test_inflate_shell_parity_probe_rejects_output_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    source_archive, source_inflate = _write_packet(tmp_path / "source", payload=b"payload")
    candidate_archive, candidate_inflate = _write_packet(
        tmp_path / "candidate",
        payload=b"payload",
        output_suffix="-changed",
    )

    report = mod.build_report(
        mod.parse_args(
            [
                "--source-archive",
                str(source_archive),
                "--source-inflate-sh",
                str(source_inflate),
                "--candidate-archive",
                str(candidate_archive),
                "--candidate-inflate-sh",
                str(candidate_inflate),
            ]
        ),
        raw_argv=[],
    )

    assert report["passed"] is False
    assert report["output_mismatches"][0]["relative_path"] == "0.raw"


def test_inflate_shell_parity_probe_cli_json_contract(tmp_path: Path) -> None:
    source_archive, source_inflate = _write_packet(tmp_path / "source", payload=b"payload")
    candidate_archive, candidate_inflate = _write_packet(
        tmp_path / "candidate",
        payload=b"payload",
    )
    out = tmp_path / "report.json"
    mod = _load_module()

    rc = mod.main(
        [
            "--source-archive",
            str(source_archive),
            "--source-inflate-sh",
            str(source_inflate),
            "--candidate-archive",
            str(candidate_archive),
            "--candidate-inflate-sh",
            str(candidate_inflate),
            "--json-out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["schema"] == "pact.inflate_shell_output_parity_v1"
    assert payload["passed"] is True


def test_inflate_shell_parity_probe_resolves_relative_inputs(tmp_path: Path, monkeypatch) -> None:
    source_archive, source_inflate = _write_packet(tmp_path / "source", payload=b"payload")
    candidate_archive, candidate_inflate = _write_packet(
        tmp_path / "candidate",
        payload=b"payload",
    )
    monkeypatch.chdir(tmp_path)
    mod = _load_module()

    report = mod.build_report(
        mod.parse_args(
            [
                "--source-archive",
                str(source_archive.relative_to(tmp_path)),
                "--source-inflate-sh",
                str(source_inflate.relative_to(tmp_path)),
                "--candidate-archive",
                str(candidate_archive.relative_to(tmp_path)),
                "--candidate-inflate-sh",
                str(candidate_inflate.relative_to(tmp_path)),
            ]
        ),
        raw_argv=[],
    )

    assert report["passed"] is True
    assert report["source"]["outputs"] == report["candidate"]["outputs"]


def test_inflate_shell_parity_probe_python_bin_shims_python3(tmp_path: Path) -> None:
    mod = _load_module()
    source_archive, source_inflate = _write_python3_packet(tmp_path / "source", payload=b"payload")
    candidate_archive, candidate_inflate = _write_python3_packet(
        tmp_path / "candidate",
        payload=b"payload",
    )
    fake_python = _write_fake_python(tmp_path)

    report = mod.build_report(
        mod.parse_args(
            [
                "--source-archive",
                str(source_archive),
                "--source-inflate-sh",
                str(source_inflate),
                "--candidate-archive",
                str(candidate_archive),
                "--candidate-inflate-sh",
                str(candidate_inflate),
                "--python-bin",
                str(fake_python),
            ]
        ),
        raw_argv=[],
    )

    resolution = report["runtime_environment"]["python_resolution"]
    assert report["passed"] is True
    assert sorted(resolution["python_shims"]) == ["python", "python3"]
    assert sorted(resolution["python_shim_sha256s"]) == ["python", "python3"]
    assert report["source"]["outputs"] == report["candidate"]["outputs"]
