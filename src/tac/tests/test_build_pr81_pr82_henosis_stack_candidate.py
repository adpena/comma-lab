# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import brotli
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/build_pr81_pr82_henosis_stack_candidate.py"
PR81_ARCHIVE = REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip"
PR81_PROFILE = (
    REPO_ROOT
    / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/pr81_qma9_semantic_range_mask_profile.json"
)
PR82_ARCHIVE = REPO_ROOT / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/archive.zip"
PR82_REPLAY = (
    REPO_ROOT
    / "experiments/results/public_pr82_henosis_frontier_intake_20260503_codex/replay_submission/inflate.py"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("build_pr81_pr82_henosis_stack_candidate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _vlq(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _randmulti_sparse_row(indices: list[int], values: list[int]) -> bytes:
    out = bytearray([len(indices)])
    previous = -1
    for index in indices:
        out.extend(_vlq(index - previous - 1))
        previous = index
    out.extend(values)
    return bytes(out)


def _synthetic_pr82_archive(path: Path, replay: Path) -> None:
    post = bytes([0] * 600 + [1] + [0] * 599 + [0] * 1200)
    shift = b"SH4" + bytes([40] * 600)
    frac = b"FH1" + bytes([4] * 600)
    frac2 = b"FH2" + bytes([4] * 600)
    frac3 = b"FH3" + bytes([4] * 600)
    bias = b"BH1" + bytes([13] * 600)
    region = b"RH1" + bytes([0] * 600)
    specs = [
        (24, 32, 1, 12),
        (12, 16, 1, 1),
        (6, 8, 1, 1),
        (3, 4, 1, 1),
        (2, 2, 1, 1),
        (8, 8, 1, 1),
        (4, 4, 1, 1),
    ]
    randmulti_rows = []
    for group_index, spec in enumerate(specs):
        for row_index in range(spec[3]):
            if group_index == 6 and row_index == 0:
                randmulti_rows.append(_randmulti_sparse_row([5], [7]))
            else:
                randmulti_rows.append(_randmulti_sparse_row([], []))
    randmulti_raw = b"".join(randmulti_rows)
    encoded = {
        "mask": brotli.compress(b"mask"),
        "model": brotli.compress(b"QH0" + bytes(16)),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0, 0, 0])),
        "post": brotli.compress(post),
        "shift": brotli.compress(shift),
        "frac": brotli.compress(frac),
        "frac2": brotli.compress(frac2),
        "frac3": brotli.compress(frac3),
        "bias": brotli.compress(bias),
        "region": brotli.compress(region),
        "randmulti": brotli.compress(randmulti_raw),
    }
    replay.write_text(
        "\n".join(
            [
                "def load_compact_archive_bundle():",
                f"    l_bias = {len(encoded['bias'])}",
                f"    l_region = {len(encoded['region'])}",
                "def main():",
                f"    specs_n = {specs!r}",
            ]
        ),
        encoding="utf-8",
    )
    header_names = ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3")
    payload = b"".join(len(encoded[name]).to_bytes(3, "little") for name in header_names)
    payload += b"".join(encoded[name] for name in header_names)
    payload += encoded["bias"] + encoded["region"] + encoded["randmulti"]
    _write_zip(path, "x", payload)


def test_score_formula_uses_pr82_components_and_candidate_bytes() -> None:
    script = _load_script()

    score = script.contest_score_from_components(215_960, segnet_dist=0.00057185, posenet_dist=0.0001894)

    assert 0.244 < score < 0.246


def test_synthetic_pr81_pr82_stack_candidates_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.setattr(
        script,
        "_pr81_reordered_model_restore_preflight",
        lambda model_payload: {
            "input_model_payload_bytes": len(model_payload),
            "input_model_payload_sha256": script.sha256_bytes(model_payload),
            "runtime_inflate_renderer": "synthetic-test-double",
            "runtime_inflate_renderer_sha256": "0" * 64,
            "restored_block_size": 32,
            "restored_model_bytes": len(model_payload) + 4,
            "restored_model_sha256": script.sha256_bytes(b"QZS3" + model_payload),
            "status": "synthetic_test_double",
        },
    )
    mask = struct.pack("<4sIIII", b"QMA9", 1, 2, 2, 1) + b"x"
    pr81_payload = mask + b"model" + b"pose" + b"router"
    pr81_archive = tmp_path / "pr81.zip"
    _write_zip(pr81_archive, "p", pr81_payload)
    pr81_profile = tmp_path / "pr81_profile.json"
    pr81_profile.write_text(
        json.dumps(
            {
                "qma9": {"decoded_mask_bytes": 4},
                "split_constants": {
                    "RANGE_MASK_BYTES": len(mask),
                    "SPLIT_MODEL_REORDERED_BYTES": 5,
                    "POSE_STREAM_BYTES": 4,
                    "ROUTER_ACTION_BYTES": 6,
                },
            }
        ),
        encoding="utf-8",
    )
    pr82_archive = tmp_path / "pr82.zip"
    replay = tmp_path / "inflate.py"
    _synthetic_pr82_archive(pr82_archive, replay)

    summary = script.build_candidates(
        pr81_archive=pr81_archive,
        pr81_profile_json=pr81_profile,
        pr82_archive=pr82_archive,
        replay_inflate=replay,
        output_dir=tmp_path / "out",
        expected_pr81_sha256=None,
        expected_pr82_sha256=None,
    )

    assert summary["score_claim"] is False
    assert summary["no_remote_dispatch"] is True
    assert summary["candidate_count"] == 6
    assert summary["highest_ev_local_candidate"]["dispatch_gate"]["dispatch_ready_now"] is True
    assert summary["highest_ev_qrm1_compatible_candidate"]["dispatch_gate"]["dispatch_ready_now"] is True
    assert summary["pr81_profile"]["segments"][0]["name"] == "range_mask.qma9"
    assert summary["pr82_profile"]["contract"]["randmulti_group_count"] == 7
    assert summary["static_lower_bounds"][0]["label"] == "ideal_pr81_archive_bytes_with_pr82_components"
    assert (tmp_path / "out/candidate_summary.json").exists()
    manifest = json.loads((tmp_path / "out/pr81_qma9_pr82_qps1_controls_qrm1_all072/manifest.json").read_text())
    assert manifest["dispatch_gate"]["dispatch_ready_now"] is True
    assert manifest["randmulti"]["runtime_support_report"]["dispatchable_qrm1"] is True
    subset_manifest = json.loads(
        (tmp_path / "out/pr81_qma9_pr82_qps1_controls_qrm1_supported_subset/manifest.json").read_text()
    )
    assert subset_manifest["dispatch_gate"]["dispatch_ready_now"] is True
    assert subset_manifest["archive_bytes"] == subset_manifest["output_archive"]["bytes"]
    assert subset_manifest["archive_sha256"] == subset_manifest["output_archive"]["sha256"]
    assert subset_manifest["randmulti"]["excluded_group_policy"]["excluded_active_unsupported_group_ids"] == []
    assert subset_manifest["raw_output_delta_proof"]["changed_values"] > 0


@pytest.mark.skipif(
    not (PR81_ARCHIVE.exists() and PR81_PROFILE.exists() and PR82_ARCHIVE.exists() and PR82_REPLAY.exists()),
    reason="PR81/PR82 public intake artifacts missing",
)
def test_actual_pr81_pr82_stack_builds_local_candidates(tmp_path: Path) -> None:
    script = _load_script()

    summary = script.build_candidates(
        pr81_archive=PR81_ARCHIVE,
        pr81_profile_json=PR81_PROFILE,
        pr82_archive=PR82_ARCHIVE,
        replay_inflate=PR82_REPLAY,
        output_dir=tmp_path,
        expected_pr81_sha256=None,
        expected_pr82_sha256=None,
    )

    assert summary["pr81_profile"]["archive_bytes"] == 215_960
    assert summary["pr82_profile"]["archive_bytes"] == 296_789
    assert summary["highest_ev_local_candidate"]["dispatch_gate"]["dispatch_ready_now"] is True
    assert summary["highest_ev_qrm1_compatible_candidate"]["dispatch_gate"]["dispatch_ready_now"] is True
    assert summary["highest_ev_local_candidate"]["archive_bytes"] < 240_000
    assert summary["static_lower_bounds"][0]["expected_score_if_pr82_components_carry"] < 0.246
    all072_manifest = json.loads((tmp_path / "pr81_qma9_pr82_qps1_controls_qrm1_all072/manifest.json").read_text())
    assert all072_manifest["dispatch_gate"]["dispatch_ready_now"] is True
    assert all072_manifest["randmulti"]["runtime_support_report"]["source_mask_required_group_ids"] == [
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        70,
    ]
    assert all072_manifest["raw_output_delta_proof"]["changed_values"] > 0
    manifest = json.loads((tmp_path / "pr81_qma9_pr82_qps1_controls_qrm1_supported_subset/manifest.json").read_text())
    assert manifest["randmulti"]["excluded_group_policy"]["excluded_active_unsupported_group_ids"] == []
    assert manifest["randmulti"]["runtime_support_report_after_exclusion"]["dispatchable_qrm1"] is True
    assert manifest["raw_output_delta_proof"]["changed_values"] > 0
