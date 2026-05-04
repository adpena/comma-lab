from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pr85_adaptive_masking_sidechannel_attribution.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("pr85_sidechannel_attribution_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _write_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x")
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _synthetic_randmulti_headerless() -> bytes:
    rows = sum(spec[3] for spec in module.HEADERLESS_RANDMULTI_SPECS)
    return b"\x00" * rows


def _synthetic_pr85_archive(path: Path) -> dict[str, bytes]:
    mask_bitstream = b"abcde"
    decoded = {
        "mask": b"QMA9"
        + (2).to_bytes(4, "little")
        + (3).to_bytes(4, "little")
        + (4).to_bytes(4, "little")
        + len(mask_bitstream).to_bytes(4, "little")
        + mask_bitstream,
        "model": b"QH0" + b"\x00\x01model",
        "pose": b"P1D1" + bytes([2, 0]) + (1).to_bytes(2, "little") + bytes([1]) + (1).to_bytes(2, "little") + b"\x00\x00",
        "post": bytes([1]) * module.PAIR_COUNT + bytes([2]) * module.PAIR_COUNT,
        "shift": b"SD4" + bytes([0]) * module.PAIR_COUNT,
        "frac": b"FV1" + (2).to_bytes(2, "little") + _varint(0) + _varint(2) + bytes([5, 6]),
        "frac2": b"FH2" + bytes([4]) * module.PAIR_COUNT,
        "frac3": b"FD3" + bytes([0]) * module.PAIR_COUNT,
        "bias": b"BD1" + bytes([0]) * module.PAIR_COUNT,
        "region": b"RH1" + bytes([0]) * module.PAIR_COUNT,
        "randmulti": _synthetic_randmulti_headerless(),
    }
    encoded = {
        name: (payload if name == "mask" else brotli.compress(payload, quality=5))
        for name, payload in decoded.items()
    }
    encoded["bias"] = b"B" * module.FIXED_V5_BIAS_BYTES
    encoded["region"] = b"R" * module.FIXED_V5_REGION_BYTES
    header = b"".join(_u24(len(encoded[name])) for name in module.SEGMENT_ORDER[:8])
    raw = header + b"".join(encoded[name] for name in module.SEGMENT_ORDER)
    _write_zip(path, raw)
    return decoded


def test_static_pr85_attribution_parses_first_level_schema_facts(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _synthetic_pr85_archive(archive)

    payload = module.build_profile(archive)
    rows = {row["name"]: row for row in payload["segments"]}

    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["gpu_required"] is False
    assert payload["deterministic"] is True
    assert payload["qma9_mask_token_prefix_profile"]["status"] == "passed"
    assert payload["qma9_mask_token_prefix_profile"]["promotable"] is False
    assert payload["qma9_mask_token_prefix_profile"]["prefix_pixels_traced"] == 24
    assert payload["qma9_mask_token_prefix_profile"]["qma9_header"]["payload_sha256"] == rows["mask"]["raw_sha256"]
    assert payload["qma9_mask_token_prefix_profile"]["prefix_self_roundtrip"]["matches_prefix"] is True
    assert rows["mask"]["schema_facts"] == {
        "actual_bitstream_bytes": 5,
        "class_count_assumption": 5,
        "decoded_mask_bytes": 24,
        "declared_bitstream_bytes": 5,
        "declared_length_matches_payload": True,
        "frame_count": 2,
        "header_bytes": 20,
        "height": 3,
        "kind": "qma9_range_mask",
        "magic": "QMA9",
        "recognized": True,
        "width": 4,
    }
    assert rows["model"]["schema_facts"]["magic"] == "QH0"
    assert rows["model"]["schema_facts"]["runtime_imported"] is False
    assert rows["pose"]["schema_facts"]["dimensions"] == [0, 1]
    assert rows["pose"]["schema_facts"]["payload_length_matches_streams"] is True
    assert rows["post"]["schema_facts"]["stage_count"] == 2
    assert rows["post"]["schema_facts"]["pairs_per_stage"] == module.PAIR_COUNT
    assert rows["post"]["schema_facts"]["choice_stats_by_stage"][0]["dominant_symbol"] == 1
    assert rows["post"]["schema_facts"]["choice_stats_by_stage"][0]["dominant_symbol_fraction"] == 1.0
    assert rows["post"]["schema_facts"]["ideal_entropy_bytes_total"] == 0.0
    assert rows["frac"]["schema_facts"]["sparse_override_count"] == 2
    assert rows["frac"]["schema_facts"]["sparse_override_fraction"] == round(2 / module.PAIR_COUNT, 6)
    assert rows["frac"]["schema_facts"]["value_stats"]["unique_count"] == 2
    assert rows["randmulti"]["schema_facts"]["kind"] == "headerless_randmulti_sparse_tables"
    assert rows["randmulti"]["schema_facts"]["payload_length_matches_specs"] is True
    assert rows["randmulti"]["schema_facts"]["nonzero_entries"] == 0
    assert rows["randmulti"]["schema_facts"]["nonzero_density"] == 0.0
    assert rows["randmulti"]["schema_facts"]["top_groups_by_nonzero"][0]["nonzero_entries"] == 0


def test_attribution_candidates_are_ranked_and_planning_only(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _synthetic_pr85_archive(archive)

    payload = module.build_profile(archive)
    candidates = payload["recommended_isolated_transplant_eval_candidates"]

    assert [candidate["rank"] for candidate in candidates] == list(range(1, len(candidates) + 1))
    assert all(candidate["score_claim"] is False for candidate in candidates)
    assert all(candidate["dispatch_status"] == "planning_only_no_dispatch" for candidate in candidates)
    assert all(candidate["evidence_grade"] == module.EVIDENCE_GRADE for candidate in candidates)
    assert candidates == sorted(candidates, key=lambda item: (-item["expected_ev_score"], item["candidate_id"]))
    assert {candidate["candidate_id"] for candidate in candidates} >= {
        "randmulti_sparse_table_isolation",
        "post_code_stage_ablation",
        "f1_micro_sidechannel_stack",
    }


def test_cli_writes_deterministic_json(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "archive.zip"
    out = tmp_path / "plan.json"
    _synthetic_pr85_archive(archive)

    assert module.main(["--archive", str(archive), "--json-out", str(out)]) == 0
    first = out.read_text()
    stdout = capsys.readouterr().out
    assert stdout == first

    assert module.main(["--archive", str(archive), "--json-out", str(out)]) == 0
    second = out.read_text()
    capsys.readouterr()
    assert first == second
    assert json.loads(second)["schema"] == module.SCHEMA


def test_qma9_prefix_token_profile_can_be_disabled(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _synthetic_pr85_archive(archive)

    payload = module.build_profile(archive, qma9_prefix_pixels=0)

    assert payload["qma9_mask_token_prefix_profile"] == {
        "dispatch_status": "planning_only_no_dispatch",
        "reason": "qma9 prefix token extraction disabled",
        "score_claim": False,
        "status": "skipped",
    }
