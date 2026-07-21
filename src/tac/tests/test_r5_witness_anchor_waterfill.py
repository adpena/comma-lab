from __future__ import annotations

import importlib.util
import itertools
import json
import struct
import zlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "r5_witness_anchor_waterfill.py"
    spec = importlib.util.spec_from_file_location("r5_witness_anchor_waterfill", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def test_lvls1_pack_parse_exact_roundtrip() -> None:
    manifest = json.dumps({"format_version": 1}, separators=(",", ":")).encode()
    blocks = [manifest, b"base", b"code", b""]
    blob = tool.pack_lvls1(blocks)

    parsed_manifest, parsed_blocks = tool.parse_lvls1(blob)

    assert parsed_manifest == {"format_version": 1}
    assert parsed_blocks == blocks


def test_lvls1_parser_rejects_trailing_bytes() -> None:
    manifest = json.dumps({"format_version": 1}).encode()
    with pytest.raises(tool.R5Error, match="trailing"):
        tool.parse_lvls1(tool.pack_lvls1([manifest, b"", b"", b""]) + b"x")


def _uvarint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def test_donor_keys_parse_self_delimited_component_packets(tmp_path: Path) -> None:
    sites = [7, 11, 300]
    raw = bytearray(tool.COMPONENT_HEADER.pack(4, 2, 1, len(sites), sites[0]))
    for left, right in itertools.pairwise(sites):
        raw.extend(_uvarint(right - left))
    compressed = zlib.compress(bytes(raw), 9)
    path = tmp_path / "donor.pcomp3"
    path.write_bytes(struct.pack("<I", len(compressed)) + compressed)

    keys, receipt = tool.donor_keys(path)

    expected = np.asarray(
        [((4 * tool.SEG_HEIGHT * tool.SEG_WIDTH + site) * 5 + 2) for site in sites],
        dtype=np.uint64,
    )
    assert np.array_equal(keys, expected)
    assert receipt["packets"] == 1
    assert receipt["unique_description_repairable_sites"] == 3


def test_spatial_view_preserves_elements_and_adds_explicit_geometry() -> None:
    one = np.arange(5, dtype=np.int8)
    two = np.arange(12, dtype=np.int8).reshape(3, 4)

    assert tool._spatial_view(one).shape == (1, 1, 5, 1)
    assert tool._spatial_view(two).shape == (1, 3, 4, 1)
    assert np.array_equal(tool._spatial_view(two).reshape(two.shape), two)


def test_compose_admits_only_measured_singleton_and_closes_interaction(tmp_path: Path) -> None:
    def write_json(name: str, payload: dict) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return path

    anchor_score = write_json(
        "anchor.json",
        {"measurement": {"aggregate": {"d_seg_official_float32": 0.1, "d_pose_official_float32": 0.1}}},
    )
    anchor_archive = tmp_path / "anchor.zip"
    anchor_archive.write_bytes(b"a" * 100)

    def candidate(name: str, *, d_seg: float, archive_bytes: int) -> tuple[Path, Path]:
        score = write_json(
            f"{name}_score.json",
            {"measurement": {"aggregate": {"d_seg_official_float32": d_seg, "d_pose_official_float32": 0.1}}},
        )
        build = write_json(
            f"{name}_build.json",
            {
                "candidate": {
                    "archive": {"path": f"/{name}.zip", "bytes": archive_bytes, "sha256": "b" * 64},
                    "plan": [],
                }
            },
        )
        return score, build

    jrd_score, jrd_build = candidate("jrd", d_seg=0.2, archive_bytes=90)
    requant_score, requant_build = candidate("requant", d_seg=0.1, archive_bytes=99)
    coder_race = write_json("coder.json", {})
    r3_overlap = write_json(
        "overlap.json",
        {"donor": {"bytes": 1000}, "overlap_sites": 0},
    )
    output = tmp_path / "composed.json"

    assert tool.compose(
        Namespace(
            anchor_score=anchor_score,
            anchor_archive=anchor_archive,
            jrd_score=jrd_score,
            jrd_build=jrd_build,
            requant_score=requant_score,
            requant_build=requant_build,
            coder_race=coder_race,
            r3_overlap=r3_overlap,
            pointer=0.1910828242,
            output=output,
        )
    ) == 0
    payload = json.loads(output.read_text())
    assert payload["D2_curves"]["jrd_prefix_453"]["waterfill_admit"] is False
    assert payload["D2_curves"]["requant_336"]["waterfill_admit"] is True
    assert payload["D3_interaction_matrix"]["status"] == "CLOSED_ZERO_OR_ONE_ADMITTED_STREAM"
    assert payload["D3_composed_v5"]["selected_stream"] == "requant_336"
    assert sum(payload["D4_pointer_comparison"]["gap_component_percent"].values()) == pytest.approx(
        100.0
    )
