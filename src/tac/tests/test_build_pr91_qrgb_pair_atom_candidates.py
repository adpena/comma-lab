from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle, parse_pr85_bundle


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr91_qrgb_pair_atom_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr91_qrgb_pair_atom_candidates_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()
pair_builder = module.pair_builder


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, pair_builder.FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hpm1_mask() -> bytes:
    tokens = b"\x00\x00\x00\x00" * 8
    hpac = b"fixture-hpac"
    return (
        b"HPM1"
        + struct.pack("<" + "I" * 11, 600, 384, 512, 32, 2, 64, 1, 8, len(tokens), len(hpac), 4)
        + tokens
        + hpac
    )


def _write_pr91_archive(path: Path) -> Path:
    bias_raw = bytearray([0] * pair_builder.PAIR_COUNT)
    bias_raw[1] = 24 + 1
    region_raw = bytearray([0] * pair_builder.PAIR_COUNT)
    region_raw[3] = 60 + 1
    values = bytes([4]) * pair_builder.PAIR_COUNT
    segments = {
        "mask": _hpm1_mask(),
        "model": brotli.compress(b"QH0" + b"R" * 64, quality=5),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0]) + (600).to_bytes(2, "little") + bytes(600), quality=5),
        "post": brotli.compress(bytes(pair_builder.PAIR_COUNT * 4), quality=5),
        "shift": brotli.compress(b"SD4" + bytes(pair_builder.PAIR_COUNT), quality=5),
        "frac": brotli.compress(b"FH1" + values, quality=5),
        "frac2": brotli.compress(b"FH2" + values, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes(pair_builder.PAIR_COUNT), quality=5),
        "bias": brotli.compress(b"BD1" + bytes(bias_raw), quality=5),
        "region": brotli.compress(b"RH1" + bytes(region_raw), quality=5),
        "randmulti": brotli.compress(bytes(72), quality=5),
    }
    assert set(segments) == set(SEGMENT_ORDER)
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), raw)
    return path


def _action_spec(path: Path, *, source_value: int = 24) -> Path:
    return _write_json(
        path,
        {
            "schema": pair_builder.ACTION_SPEC_SCHEMA,
            "score_claim": False,
            "dispatch_performed": False,
            "remote_jobs_dispatched": False,
            "inflate_time_scorer_load_allowed": False,
            "candidates": [
                {
                    "candidate_id": "pr85_qrgb_fixture_bias_pair_0001",
                    "header_mode": "explicit_30",
                    "actions": [
                        {
                            "op": "set",
                            "pair_index": 1,
                            "stream": "bias",
                            "source_value": source_value,
                            "value": 5,
                            "source_atom_id": "fixture:pair_0001",
                            "source_artifact_sha256": "a" * 64,
                        }
                    ],
                }
            ],
        },
    )


def _segment_diff(path: Path, *, sha_equal: bool = True) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "score_claim": False,
            "segments": [
                {"name": "mask", "sha_equal": False},
                {"name": "bias", "sha_equal": sha_equal},
                {"name": "region", "sha_equal": True},
            ],
        },
    )


def test_pr91_qrgb_transfer_builds_byte_closed_hpm1_candidate(tmp_path: Path) -> None:
    archive = _write_pr91_archive(tmp_path / "source.zip")
    action = _action_spec(tmp_path / "actions.json")
    diff = _segment_diff(tmp_path / "diff.json")

    summary = module.build_pr91_qrgb_pair_atom_candidates(
        source_archive=archive,
        action_spec_json=action,
        segment_diff_json=diff,
        transfer_decisions_json=tmp_path / "missing_transfer.json",
        out_dir=tmp_path / "out",
    )

    assert summary["candidate_archive_count"] == 1
    assert summary["dispatch_unlocked"] is False
    candidate = summary["candidates"][0]
    assert candidate["build_status"] == "built"
    assert candidate["scorer_gradient_claim"] is False
    assert candidate["pr91_scorer_gradient_consumed"] is False
    assert candidate["non_noop_proof"]["hpm1_mask_unchanged"] is True
    assert candidate["changed_segments"] == ["bias"]

    with zipfile.ZipFile(REPO / candidate["candidate_archive"]["archive_path"]) as zf:
        parsed = parse_pr85_bundle(zf.read("x"))
    assert parsed.segment_contracts["mask"].codec == "HPM1"
    decoded_bias = brotli.decompress(parsed.segments["bias"])
    assert decoded_bias.startswith(b"BD1")
    assert decoded_bias[3 + 1] == 5 + 1


def test_pr91_qrgb_transfer_fails_on_source_value_mismatch(tmp_path: Path) -> None:
    archive = _write_pr91_archive(tmp_path / "source.zip")
    action = _action_spec(tmp_path / "actions.json", source_value=99)
    diff = _segment_diff(tmp_path / "diff.json")

    summary = module.build_pr91_qrgb_pair_atom_candidates(
        source_archive=archive,
        action_spec_json=action,
        segment_diff_json=diff,
        transfer_decisions_json=tmp_path / "missing_transfer.json",
        out_dir=tmp_path / "out",
    )

    assert summary["candidate_archive_count"] == 0
    assert summary["dispatch_unlocked"] is False
    assert summary["candidates"][0]["blocker_class"] == "pr91_source_value_mismatch"


def test_pr91_qrgb_transfer_fails_on_pr85_pr91_segment_mismatch(tmp_path: Path) -> None:
    archive = _write_pr91_archive(tmp_path / "source.zip")
    action = _action_spec(tmp_path / "actions.json")
    diff = _segment_diff(tmp_path / "diff.json", sha_equal=False)

    summary = module.build_pr91_qrgb_pair_atom_candidates(
        source_archive=archive,
        action_spec_json=action,
        segment_diff_json=diff,
        transfer_decisions_json=tmp_path / "missing_transfer.json",
        out_dir=tmp_path / "out",
    )

    assert summary["candidate_archive_count"] == 0
    assert summary["blocker_class"] == "pr91_pr85_source_mismatch"
