from __future__ import annotations

import zipfile
from pathlib import Path

_FIXED_ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)


def validate_submission_inputs(*, payload_dir: Path, decompress_path: Path) -> None:
    payload_dir = payload_dir.resolve()
    decompress_path = decompress_path.resolve()

    if not payload_dir.is_dir():
        raise ValueError(f"payload_dir must be an existing directory: {payload_dir}")
    if not decompress_path.is_file():
        raise ValueError(f"decompress.py must be an existing file: {decompress_path}")
    if decompress_path.name != "decompress.py":
        raise ValueError(f"expected decompress.py, got: {decompress_path.name}")

    try:
        decompress_path.relative_to(payload_dir)
    except ValueError:
        pass
    else:
        raise ValueError("decompress.py must live outside the payload directory")


def _iter_payload_files(payload_dir: Path) -> list[Path]:
    return sorted(path for path in payload_dir.rglob("*") if path.is_file())


def _zip_info(name: str, source: Path) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    mode = source.stat().st_mode & 0o777
    info.external_attr = (0o100000 | mode) << 16
    return info


def build_submission_zip(*, payload_dir: Path, decompress_path: Path, output_path: Path) -> Path:
    validate_submission_inputs(payload_dir=payload_dir, decompress_path=decompress_path)

    payload_dir = payload_dir.resolve()
    decompress_path = decompress_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, mode="w") as zf:
        for source in _iter_payload_files(payload_dir):
            arcname = source.relative_to(payload_dir).as_posix()
            if arcname == "decompress.py":
                raise ValueError("payload directory must not contain decompress.py")
            zf.writestr(_zip_info(arcname, source), source.read_bytes(), compresslevel=9)
        zf.writestr(
            _zip_info("decompress.py", decompress_path),
            decompress_path.read_bytes(),
            compresslevel=9,
        )

    return output_path
