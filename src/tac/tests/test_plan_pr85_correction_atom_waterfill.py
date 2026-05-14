# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

brotli = pytest.importorskip("brotli")

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_pr85_correction_atom_waterfill.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plan_pr85_correction_atom_waterfill_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


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


def _sparse_choice(magic: bytes, values: bytes, *, default_choice: int) -> bytes:
    indices = [index for index, value in enumerate(values) if value != default_choice]
    out = bytearray(magic + len(indices).to_bytes(2, "little"))
    previous = -1
    for index in indices:
        out += _varint(index - previous - 1)
        previous = index
    out += bytes(values[index] + 1 for index in indices)
    return bytes(out)


def _randmulti_raw() -> bytes:
    raw = bytearray()
    for group_index, (_height, _width, _amplitude, row_count) in enumerate(
        module.PR85_HEADERLESS_RANDMULTI_SPECS
    ):
        for row_index in range(row_count):
            if group_index == 0 and row_index == 0:
                raw += b"\x02" + _varint(0) + _varint(2) + bytes([7, 8])
            elif group_index == 1 and row_index == 0:
                raw += b"\x01" + _varint(1) + bytes([5])
            else:
                raw += b"\x00"
    return bytes(raw)


def _write_archive(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _fixture_archive(path: Path) -> None:
    post = b"".join(
        bytes((stage + pair) % (stage + 3) for pair in range(module.PAIR_COUNT))
        for stage in range(1, 5)
    )
    shift_values = bytes(41 if pair % 5 == 0 else 40 for pair in range(module.PAIR_COUNT))
    frac_values = bytes(6 if pair in {3, 9, 27} else 4 for pair in range(module.PAIR_COUNT))
    frac2_values = bytes(2 if pair % 11 == 0 else 4 for pair in range(module.PAIR_COUNT))
    frac3_values = bytes(7 if pair % 13 == 0 else 4 for pair in range(module.PAIR_COUNT))
    bias_values = bytes(12 if pair % 17 == 0 else 13 for pair in range(module.PAIR_COUNT))
    region_values = bytes(2 if pair % 19 == 0 else 0 for pair in range(module.PAIR_COUNT))
    segments = {
        "mask": b"QMA9" + b"M" * 1001,
        "model": brotli.compress(b"QH0" + b"W" * 1001, quality=5),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0]) + (600).to_bytes(2, "little") + bytes(600), quality=5),
        "post": brotli.compress(post, quality=5),
        "shift": brotli.compress(b"SD4" + bytes(0 if value == 40 else value + 1 for value in shift_values), quality=5),
        "frac": brotli.compress(_sparse_choice(b"FV1", frac_values, default_choice=4), quality=5),
        "frac2": brotli.compress(b"FH2" + frac2_values, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes(0 if value == 4 else value + 1 for value in frac3_values), quality=5),
        "bias": brotli.compress(b"BD1" + bytes(0 if value == 13 else value + 1 for value in bias_values), quality=5),
        "region": brotli.compress(b"RH1" + region_values, quality=5),
        "randmulti": brotli.compress(_randmulti_raw(), quality=5),
    }
    payload = pack_pr85_bundle(segments, header_mode="explicit_30")
    assert payload[30:34] == b"QMA9"
    assert set(segments) == set(SEGMENT_ORDER)
    _write_archive(path, payload)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _eval(path: Path, *, score: float, pose_score: float, rate_score: float, bytes_: int = 1000) -> None:
    _write_json(
        path,
        {
            "archive_size_bytes": bytes_,
            "avg_posenet_dist": 0.0002,
            "avg_segnet_dist": 0.0005,
            "n_samples": module.PAIR_COUNT,
            "score_pose_contribution": pose_score,
            "score_rate_contribution": rate_score,
            "score_recomputed_from_components": score,
            "score_seg_contribution": 0.05,
        },
    )


def _manifest(path: Path) -> None:
    _write_json(
        path,
        {
            "byte_delta_vs_source_archive": -97,
            "changed_segments": ["frac"],
            "neutralized_groups": ["motion_frac"],
            "policy_id": "preserve_post_all_shift_frac2_frac3",
            "selected_groups": [
                "post_stage1",
                "post_stage2",
                "post_stage3",
                "post_stage4",
                "motion_shift",
                "motion_frac2",
                "motion_frac3",
            ],
            "whole_stream_negative_context": {
                "minus_motion_stack": {"score": 0.36},
                "minus_post": {"score": 0.31},
            },
        },
    )


def test_build_plan_emits_fine_grained_atoms_and_strict_gates(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    baseline = tmp_path / "baseline.json"
    preserve = tmp_path / "preserve.json"
    manifest = tmp_path / "manifest.json"
    _fixture_archive(archive)
    _eval(baseline, score=0.258, pose_score=0.043, rate_score=0.157)
    _eval(preserve, score=0.271, pose_score=0.056, rate_score=0.1569)
    _manifest(manifest)

    payload = module.build_plan(
        archive=archive,
        baseline_eval_json=baseline,
        preserve_eval_json=preserve,
        preserve_manifest_json=manifest,
    )

    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    ledger = payload["atom_ledger"]
    atoms = {atom["atom_id"]: atom for atom in ledger["atoms"]}
    assert ledger["bundle"]["format"] == "pr85_explicit_30byte_lengths"
    assert {"pr85_post_stage1", "pr85_motion_frac", "pr85_bias", "pr85_region", "pr85_randmulti_g000"} <= set(atoms)
    assert atoms["pr85_motion_frac"]["stats"]["nondefault_count"] == 3
    assert atoms["pr85_randmulti_g000"]["raw_group_payload_bytes"] > 1
    assert atoms["pr85_bias"]["neutralization_gate"] == "requires_exact_component_response_before_eval"
    assert atoms["pr85_region"]["recode_gate"] == "requires_decoded_output_parity_before_eval"
    assert ledger["exact_negative_constraints"]["status"] == "exact_negative"
    assert ledger["exact_negative_constraints"]["blocked_atoms"] == ["motion_frac"]

    policies = payload["candidate_policies"]["policies"]
    assert all(policy["planning_only"] is True for policy in policies)
    assert all(policy["dispatch_gate"] == "planning_only/no_remote_dispatch" for policy in policies)
    frac_policy = next(
        policy for policy in policies if policy["candidate_policy_id"] == "component_response_motion_frac_microatoms"
    )
    assert frac_policy["blocked_by"] == "preserve_post_all_shift_frac2_frac3 exact negative"
    assert frac_policy["eval_gate"] == "requires_exact_component_response_on_motion_frac_atoms_before_eval"


def test_missing_exact_negative_inputs_keep_policy_blocked_without_score_claim(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _fixture_archive(archive)

    payload = module.build_plan(
        archive=archive,
        baseline_eval_json=tmp_path / "missing_baseline.json",
        preserve_eval_json=tmp_path / "missing_preserve.json",
        preserve_manifest_json=tmp_path / "missing_manifest.json",
    )

    constraints = payload["atom_ledger"]["exact_negative_constraints"]
    assert constraints["status"] == "missing_optional_inputs"
    assert constraints["rule"].startswith("motion_frac")
    assert payload["candidate_policies"]["score_claim"] is False
    assert "remote/GPU dispatch forbidden for this worker" in payload["candidate_policies"]["dispatch_blockers"]


def test_cli_writes_json_when_requested(tmp_path: Path, capsys) -> None:
    archive = tmp_path / "archive.zip"
    baseline = tmp_path / "baseline.json"
    preserve = tmp_path / "preserve.json"
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "plan.json"
    _fixture_archive(archive)
    _eval(baseline, score=0.258, pose_score=0.043, rate_score=0.157)
    _eval(preserve, score=0.271, pose_score=0.056, rate_score=0.1569)
    _manifest(manifest)

    assert module.main(
        [
            "--archive",
            str(archive),
            "--baseline-eval-json",
            str(baseline),
            "--preserve-eval-json",
            str(preserve),
            "--preserve-manifest-json",
            str(manifest),
            "--json-out",
            str(out),
        ]
    ) == 0

    written = json.loads(out.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert written == stdout
    assert written["schema"] == module.SCHEMA
