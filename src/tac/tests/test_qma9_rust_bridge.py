from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tac.qma9_range_mask_contract import (
    decode_qma9_mask,
    encode_qma9_mask,
    sha256_bytes,
)


REPO = Path(__file__).resolve().parents[3]
RUNTIME_RS = REPO / "runtime-rs"


def _build_qma_cli() -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "-p", "qma-codec"],
        cwd=RUNTIME_RS,
        check=True,
        text=True,
        capture_output=True,
    )
    cli = RUNTIME_RS / "target/debug/qma-codec"
    assert cli.is_file()
    return cli


def test_qma9_rust_cli_matches_python_reference_bytes(tmp_path: Path) -> None:
    raw = bytes(
        (t * 3 + y * 2 + x) % 5
        for t in range(4)
        for y in range(6)
        for x in range(7)
    )
    payload = encode_qma9_mask(raw, frame_count=4, width=6, height=7)
    assert decode_qma9_mask(payload).data == raw

    cli = _build_qma_cli()
    qma_path = tmp_path / "mask.qma9"
    raw_path = tmp_path / "mask.raw"
    metadata_path = tmp_path / "mask.json"
    qma_path.write_bytes(payload)

    proc = subprocess.run(
        [
            str(cli),
            "decode",
            str(qma_path),
            str(raw_path),
            "--expected-frames",
            "4",
            "--expected-width",
            "6",
            "--expected-height",
            "7",
            "--metadata-json",
            str(metadata_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    decoded = raw_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "score" not in proc.stderr.lower()
    assert decoded == raw
    assert sha256_bytes(decoded) == sha256_bytes(raw)
    assert metadata["written_mask_bytes"] == len(raw)
    assert metadata["decoded_mask_bytes"] == len(raw)


def test_qma9_rust_cli_prefix_matches_python_prefix(tmp_path: Path) -> None:
    raw = bytes((t + y + 2 * x) % 5 for t in range(3) for y in range(5) for x in range(4))
    payload = encode_qma9_mask(raw, frame_count=3, width=5, height=4)
    cli = _build_qma_cli()
    qma_path = tmp_path / "mask.qma9"
    prefix_path = tmp_path / "prefix.raw"
    qma_path.write_bytes(payload)

    subprocess.run(
        [
            str(cli),
            "decode",
            str(qma_path),
            str(prefix_path),
            "--prefix-frames",
            "2",
            "--expected-frames",
            "2",
            "--expected-width",
            "5",
            "--expected-height",
            "4",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert prefix_path.read_bytes() == raw[: 2 * 5 * 4]
