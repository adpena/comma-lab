from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tac.stbm1br_mask_codec import decode_stbm1br_mask_segment
from tac.stbm1br_rust_bridge import (
    STBM1BRRustBridgeError,
    decode_stbm1br_mask_segment_via_rust,
    resolve_stbm1br_rust_decoder,
)


REPO = Path(__file__).resolve().parents[3]
RUNTIME_RS = REPO / "runtime-rs"
STBM_SEGMENT = (
    REPO
    / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
    "pr90_stbm1br_lossless_pr85_mask_recode/mask_segment.stbm1br"
)
EXPECTED_SHAPE = (600, 384, 512)
EXPECTED_RENDER_SHA256 = "0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45"


def _build_debug_cli() -> Path:
    subprocess.run(
        ["cargo", "build", "--quiet", "-p", "stbm1br-codec"],
        cwd=RUNTIME_RS,
        check=True,
        timeout=180,
    )
    cli = RUNTIME_RS / "target/debug/stbm1br-codec"
    assert cli.is_file()
    return cli


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _fake_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def test_rust_bridge_requires_explicit_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PACT_STBM1BR_RUST_DECODER", raising=False)
    with pytest.raises(STBM1BRRustBridgeError, match="no decoder path"):
        resolve_stbm1br_rust_decoder()


def test_rust_cli_rejects_bad_magic(tmp_path: Path) -> None:
    cli = _build_debug_cli()
    bad = tmp_path / "bad.stbm1br"
    out = tmp_path / "out.raw"
    bad.write_bytes(b"QMA9" + b"\0" * 64)

    proc = subprocess.run(
        [str(cli), "decode", str(bad), str(out)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert proc.returncode != 0
    assert "bad STBM1BR magic" in proc.stderr
    assert not out.exists()


def test_rust_bridge_can_infer_shape_from_metadata_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _fake_executable(tmp_path / "stbm1br-codec")

    def fake_run(cmd, **_kwargs):
        output = Path(cmd[3])
        metadata = Path(cmd[cmd.index("--metadata-json") + 1])
        output.write_bytes(bytes(range(12)))
        metadata.write_text(
            json.dumps({"n_pairs": 2, "height": 2, "width": 3}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("tac.stbm1br_rust_bridge.subprocess.run", fake_run)

    decoded = decode_stbm1br_mask_segment_via_rust(
        b"STBM1BR\0fake",
        decoder_path=decoder,
    )

    assert decoded.dtype == np.uint8
    assert decoded.shape == (2, 2, 3)
    assert decoded.reshape(-1).tolist() == list(range(12))


def test_public_stbm_decoder_delegates_to_explicit_rust_binary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _fake_executable(tmp_path / "stbm1br-codec")

    def fake_run(cmd, **_kwargs):
        output = Path(cmd[3])
        metadata = Path(cmd[cmd.index("--metadata-json") + 1])
        assert cmd[cmd.index("--expected-frames") + 1] == "1"
        assert cmd[cmd.index("--expected-height") + 1] == "2"
        assert cmd[cmd.index("--expected-width") + 1] == "3"
        output.write_bytes(bytes([4, 3, 2, 1, 0, 4]))
        metadata.write_text(
            json.dumps({"n_pairs": 1, "height": 2, "width": 3}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("tac.stbm1br_rust_bridge.subprocess.run", fake_run)
    monkeypatch.setenv("PACT_STBM1BR_RUST_DECODER", str(decoder))

    decoded = decode_stbm1br_mask_segment(
        b"STBM1BR\0fake",
        expected_shape=(1, 2, 3),
    )

    assert decoded.shape == (1, 2, 3)
    assert decoded.tolist() == [[[4, 3, 2], [1, 0, 4]]]


def test_rust_bridge_rejects_metadata_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder = _fake_executable(tmp_path / "stbm1br-codec")

    def fake_run(cmd, **_kwargs):
        Path(cmd[3]).write_bytes(b"\0" * 6)
        Path(cmd[cmd.index("--metadata-json") + 1]).write_text(
            json.dumps({"n_pairs": 1, "height": 2, "width": 3}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("tac.stbm1br_rust_bridge.subprocess.run", fake_run)

    with pytest.raises(STBM1BRRustBridgeError, match="metadata shape"):
        decode_stbm1br_mask_segment_via_rust(
            b"STBM1BR\0fake",
            expected_shape=(1, 3, 2),
            decoder_path=decoder,
        )


@pytest.mark.timeout(240)
def test_rust_bridge_matches_public_decoder_on_real_stbm_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not STBM_SEGMENT.is_file():
        pytest.skip("real PR85 STBM1BR segment fixture is not present")
    cli = _build_debug_cli()
    segment = STBM_SEGMENT.read_bytes()

    monkeypatch.setenv("PACT_STBM1BR_RUST_DECODER", os.fspath(cli))
    python_decoded = decode_stbm1br_mask_segment(segment, expected_shape=EXPECTED_SHAPE)
    rust_decoded = decode_stbm1br_mask_segment_via_rust(
        segment,
        expected_shape=EXPECTED_SHAPE,
        decoder_path=cli,
        timeout_seconds=120.0,
    )

    assert rust_decoded.dtype == np.uint8
    assert rust_decoded.shape == python_decoded.shape == EXPECTED_SHAPE
    assert _sha256_array(python_decoded) == EXPECTED_RENDER_SHA256
    assert _sha256_array(rust_decoded) == EXPECTED_RENDER_SHA256
    assert np.array_equal(rust_decoded, python_decoded)


def test_rust_cli_real_segment_hash_matches_python_manifest(tmp_path: Path) -> None:
    if not STBM_SEGMENT.is_file():
        pytest.skip("real PR85 STBM1BR segment fixture is not present")
    cli = _build_debug_cli()
    out = tmp_path / "masks.raw"

    subprocess.run(
        [
            str(cli),
            "decode",
            str(STBM_SEGMENT),
            str(out),
            "--expected-frames",
            str(EXPECTED_SHAPE[0]),
            "--expected-height",
            str(EXPECTED_SHAPE[1]),
            "--expected-width",
            str(EXPECTED_SHAPE[2]),
        ],
        check=True,
        timeout=120,
    )

    raw = out.read_bytes()
    assert len(raw) == EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1] * EXPECTED_SHAPE[2]
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RENDER_SHA256
