# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "pr101_fec6_wrapper_profile.py"
FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location("pr101_fec6_wrapper_profile", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_pr101_fec6_archive_profile_reconstructs_wrapper_offsets() -> None:
    if not FEC6_ARCHIVE.exists() or not SOURCE_ARCHIVE.exists():
        pytest.skip(
            "missing local PR101/FEC6 forensic artifacts: "
            f"{FEC6_ARCHIVE.relative_to(REPO_ROOT)} or {SOURCE_ARCHIVE.relative_to(REPO_ROOT)}"
        )
    tool = _load_tool()

    profile = tool.profile_archive(FEC6_ARCHIVE, source_archive=SOURCE_ARCHIVE)

    assert profile["schema"] == "pr101_fec6_wrapper_profile.v1"
    assert profile["score_claim"] is False
    assert profile["dispatch_attempted"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["archive"]["member_name"] == "x"
    assert profile["archive"]["bytes"] == 178_517
    assert (
        profile["archive"]["sha256"]
        == "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
    )
    assert profile["wrapper"]["wrapper_payload_bytes"] == 178_417
    assert profile["wrapper"]["source_payload"]["bytes"] == 178_158
    assert profile["wrapper"]["selector_length_field"] == {
        "offset": 178_166,
        "bytes": 2,
        "value": 249,
    }
    assert profile["wrapper"]["runtime_parser_offsets"]["selector_payload"] == [
        178_168,
        178_417,
    ]
    assert profile["source_reference"]["source_payload_matches_wrapper"] is True
    assert profile["repeatability_gates"]["source_payload_reference_match"] is True


def test_real_pr101_fec6_selector_profile_matches_manifest_counts() -> None:
    if not FEC6_ARCHIVE.exists():
        pytest.skip(f"missing local PR101/FEC6 clean archive: {FEC6_ARCHIVE.relative_to(REPO_ROOT)}")
    tool = _load_tool()

    selector = tool.profile_archive(FEC6_ARCHIVE, source_archive=None)["wrapper"][
        "selector_payload"
    ]

    assert selector["payload_bytes"] == 249
    assert selector["selector_index_bytes"] == 243
    assert selector["payload_sha256"] == "fc5c431b5d793c33e2f320076fe6f0dd76c2d91e3826ae4b05abfb4f86f453ca"
    assert selector["n_pairs"] == 600
    assert selector["selector_code_bits_total"] == 1_944
    assert selector["selector_avg_bits_per_pair"] == 3.24
    assert selector["zero_padding_bits"] == 0
    assert selector["entropy_floor_bytes"] == 241
    assert selector["gap_to_entropy_floor_bytes"] == 8
    assert selector["code_histogram"] == {
        "0": 134,
        "1": 35,
        "2": 129,
        "3": 9,
        "4": 25,
        "5": 13,
        "6": 11,
        "7": 71,
        "8": 10,
        "9": 24,
        "10": 7,
        "11": 16,
        "12": 6,
        "13": 92,
        "14": 17,
        "15": 1,
    }


def test_fec6_parser_rejects_nonzero_padding_bits() -> None:
    tool = _load_tool()
    selector = b"FEC6" + struct.pack("<H", 1) + b"\x01"

    with pytest.raises(ValueError, match="non-zero padding bits"):
        tool.parse_fec6_selector_payload(selector)


def test_fp11_wrapper_rejects_truncated_source_payload() -> None:
    tool = _load_tool()
    wrapper = b"FP11" + struct.pack("<I", 99) + b"short"

    with pytest.raises(ValueError, match="truncated in source payload"):
        tool.parse_fp11_wrapper_payload(wrapper)
