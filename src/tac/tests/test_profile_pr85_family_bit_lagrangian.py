# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

from tac.pr85_bundle import FIXED_V5_LENGTHS, SEGMENT_ORDER, pack_pr85_bundle


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments" / "profile_pr85_family_bit_lagrangian.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("profile_pr85_family_bit_lagrangian", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _zip_x(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _segments(mask: bytes) -> dict[str, bytes]:
    return {
        "mask": mask,
        "model": b"QH0" + b"m" * 20,
        "pose": b"P1D1" + b"p" * 8,
        "post": b"post",
        "shift": b"shift",
        "frac": b"frac",
        "frac2": b"frac2",
        "frac3": b"frac3",
        "bias": b"B" * FIXED_V5_LENGTHS["bias"],
        "region": b"R" * FIXED_V5_LENGTHS["region"],
        "randmulti": b"randmulti",
    }


def _hpm1(tokens: bytes = b"tokn" * 2, hpac: bytes = b"hpac") -> bytes:
    return b"HPM1" + struct.pack(
        "<" + "I" * 11,
        600,
        384,
        512,
        32,
        2,
        64,
        1,
        8,
        len(tokens),
        len(hpac),
        4,
    ) + tokens + hpac


def test_build_report_compares_pr85_family_segments_and_hpm1(tmp_path: Path) -> None:
    module = _load_module()
    left = tmp_path / "left.zip"
    right = tmp_path / "right.zip"
    _zip_x(left, pack_pr85_bundle(_segments(b"QMA9" + b"a" * 64), header_mode="v5"))
    _zip_x(right, pack_pr85_bundle(_segments(_hpm1()), header_mode="v5"))

    report = module.build_report(
        left,
        right,
        left_label="qma9",
        right_label="hpm1",
        target_score_buffer=0.004,
    )

    assert report["schema"] == "pr85_family_bit_lagrangian_profile_v1"
    assert report["score_claim"] is False
    assert report["comparison"]["left_label"] == "qma9"
    rows = {row["name"]: row for row in report["comparison"]["segment_rows"]}
    assert set(rows) == set(SEGMENT_ORDER)
    assert rows["mask"]["left_codec"] != rows["mask"]["right_codec"]
    assert rows["mask"]["right_hpm1_metadata"]["tokens_len"] == 8
    assert rows["model"]["same_sha256"] is True
    assert report["lagrangian_target"]["neutral_bytes_needed_for_buffer"] > 0


def test_main_writes_json_and_markdown(tmp_path: Path) -> None:
    module = _load_module()
    left = tmp_path / "left.zip"
    right = tmp_path / "right.zip"
    out_json = tmp_path / "profile.json"
    out_md = tmp_path / "profile.md"
    payload = pack_pr85_bundle(_segments(b"QMA9" + b"a" * 64), header_mode="v5")
    _zip_x(left, payload)
    _zip_x(right, payload)

    assert module.main(
        [
            str(left),
            str(right),
            "--left-label",
            "a",
            "--right-label",
            "b",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    ) == 0

    assert out_json.exists()
    assert "PR85-Family Bit/Lagrangian Profile" in out_md.read_text(encoding="utf-8")
