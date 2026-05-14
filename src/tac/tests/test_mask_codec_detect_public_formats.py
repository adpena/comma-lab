# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from tac.mask_codec import (
    DETECTABLE_MASK_FORMATS,
    SUPPORTED_MASK_FORMATS,
    detect_mask_codec,
)


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_detect_mask_codec_recognizes_public_categorical_magics(tmp_path: Path) -> None:
    fixtures = {
        "qma9.bin": (b"QMA9" + b"\x00" * 28, "qma9_range"),
        "qmb1.bin": (b"QMB1" + b"\x00" * 28, "qma9_range"),
        "qmf1.bin": (b"QMF1" + b"\x00" * 28, "qma9_range"),
        "hpm1.bin": (b"HPM1" + b"\x00" * 28, "hpm1_hpac"),
        "stbm1br.bin": (b"STBM1BR\x00" + b"\x00" * 24, "stbm1br_public"),
    }

    for filename, (payload, expected) in fixtures.items():
        assert detect_mask_codec(_write(tmp_path / filename, payload)) == expected


def test_detectable_only_public_formats_are_not_generic_encode_targets() -> None:
    detectable_only = set(DETECTABLE_MASK_FORMATS) - set(SUPPORTED_MASK_FORMATS)

    assert {"hpm1_hpac", "stbm1br_public"}.issubset(detectable_only)


def test_detect_mask_codec_uses_content_before_extension(tmp_path: Path) -> None:
    mislabeled = _write(tmp_path / "masks.mkv", b"HPM1" + b"\x00" * 28)

    assert detect_mask_codec(mislabeled) == "hpm1_hpac"


def test_detect_mask_codec_fails_closed_on_unknown_payload(tmp_path: Path) -> None:
    unknown = _write(tmp_path / "masks.bin", b"????" + b"\x00" * 28)

    with pytest.raises(ValueError, match="Unrecognized mask codec"):
        detect_mask_codec(unknown)
