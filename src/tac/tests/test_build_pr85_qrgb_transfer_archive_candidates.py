# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

from tac.pr85_bundle import PR85_HEADERLESS_RANDMULTI_SPECS, SEGMENT_ORDER, pack_pr85_bundle, parse_pr85_bundle


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_qrgb_transfer_archive_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_qrgb_transfer_archive_candidates_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _zip_info(name: str = "x") -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, module.FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_archive(path: Path) -> Path:
    values = bytes([4]) * module.PAIR_COUNT
    randmulti_rows = [
        [bytearray(module.PAIR_COUNT) for _ in range(row_count)]
        for _lh, _lw, _amp, row_count in PR85_HEADERLESS_RANDMULTI_SPECS
    ]
    randmulti, _meta = module._encode_randmulti_stream(randmulti_rows)
    segments = {
        "mask": b"QMA9" + b"M" * 64,
        "model": brotli.compress(b"QH0" + b"R" * 64, quality=5),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0]) + (600).to_bytes(2, "little") + bytes(600), quality=5),
        "post": brotli.compress(bytes(module.PAIR_COUNT * 4), quality=5),
        "shift": brotli.compress(b"SD4" + bytes(module.PAIR_COUNT), quality=5),
        "frac": brotli.compress(b"FH1" + values, quality=5),
        "frac2": brotli.compress(b"FH2" + values, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes(module.PAIR_COUNT), quality=5),
        "bias": brotli.compress(b"BD1" + bytes(module.PAIR_COUNT), quality=5),
        "region": brotli.compress(b"RH1" + bytes(module.PAIR_COUNT), quality=5),
        "randmulti": randmulti,
    }
    assert set(segments) == set(SEGMENT_ORDER)
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), raw)
    return path


def _source_member(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read("x")


def _basis(stream: str) -> dict:
    if stream == "randmulti":
        return {
            "target_stream": "randmulti",
            "target_randmulti_group": {
                "group_index": 61,
                "lh": 222,
                "lw": 222,
                "amp": 4,
                "selection_row": 0,
            },
        }
    return {"target_stream": stream, "source_basis": "fixture"}


def _candidate(candidate_id: str, pair_index: int, stream: str, source_value: int, value: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "header_mode": "explicit_30",
        "actions": [
            {
                "op": "set",
                "pair_index": pair_index,
                "stream": stream,
                "source_value": source_value,
                "candidate_value": value,
                "source_artifact_sha256": "a" * 64,
                "source_atom_id": f"fixture:pair_{pair_index:04d}",
                "rationale": "fixture QRGB transfer action",
            }
        ],
        "charged_bytes_proxy": {
            "candidate_action_bytes": 7,
            "source_artifact_sha256": "a" * 64,
        },
        "score_claim": False,
    }


def _write_inputs(tmp_path: Path, archive: Path, candidates: list[dict], *, source_sha: str | None = None) -> tuple[Path, Path]:
    archive_sha = module._sha256_file(archive)
    source_sha = source_sha or archive_sha
    evidence = {
        "schema": module.PAIR_ACTION_EVIDENCE_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "candidates": candidates,
        "thresholds": [],
    }
    plan_rows = []
    for rank, row in enumerate(candidates, start=1):
        action = row["actions"][0]
        plan_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "rank": rank,
                "can_feed_pair_action_evidence": True,
                "source_archive_bytes": archive.stat().st_size,
                "source_archive_sha256": source_sha,
                "pr90_source_archive_sha256": "b" * 64,
                "pr90_source_evidence_sha256": "c" * 64,
                "source_atom_id": action["source_atom_id"],
                "actions": [dict(action)],
                "basis_action_schema": _basis(action["stream"]),
                "expected_break_even": {
                    "combined_max_charged_bytes_for_zero_net_change": 100.0,
                },
                "qrgb_pair_summary": {"pair_index": action["pair_index"]},
                "score_claim": False,
                "dispatch_unlocked": False,
            }
        )
    plan = {
        "schema": module.TRANSFER_PLAN_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "pair_planning_source": {"source_sha_for_pair_action": "a" * 64},
        "ranked_candidates": plan_rows,
    }
    return (
        _write_json(tmp_path / "pair_action_evidence.json", evidence),
        _write_json(tmp_path / "planning.json", plan),
    )


def _ready_preflight(*_args, **_kwargs) -> dict:
    return {
        "ready_for_fixed_runtime_exact_eval": True,
        "readiness_status": "ready",
        "blockers": [],
    }


def test_builder_emits_top_three_unique_pair_archives_with_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    candidates = [
        _candidate("pr85_qrgb_f2_randglobal_pair_0001", 1, "randmulti", 0, 20),
        _candidate("pr85_qrgb_f1_bias_pair_0002", 2, "bias", 13, 5),
        _candidate("pr85_qrgb_f1_region_pair_0003", 3, "region", 0, 45),
        _candidate("pr85_qrgb_f1_bias_pair_0004", 4, "bias", 13, 21),
    ]
    evidence, plan = _write_inputs(tmp_path, archive, candidates)
    monkeypatch.setattr(module.fixed_preflight, "build_preflight", _ready_preflight)

    summary = module.build_qrgb_transfer_archive_candidates(
        source_archive=archive,
        pair_action_evidence_json=evidence,
        transfer_plan_json=plan,
        out_dir=tmp_path / "out",
        max_candidates=3,
        require_known_pr85_anchor=False,
    )

    assert summary["candidate_archive_count"] == 3
    assert summary["ready_candidate_count"] == 3
    assert summary["dispatch_unlocked"] is True
    assert [row["candidate_id"] for row in summary["candidates"]] == [
        "pr85_qrgb_f2_randglobal_pair_0001",
        "pr85_qrgb_f1_bias_pair_0002",
        "pr85_qrgb_f1_region_pair_0003",
    ]
    assert (tmp_path / "out" / "planning.json").is_file()

    rand_manifest = json.loads(
        (tmp_path / "out" / "pr85_qrgb_f2_randglobal_pair_0001" / "manifest.json").read_text(encoding="utf-8")
    )
    assert rand_manifest["score_claim"] is False
    assert rand_manifest["dispatch_performed"] is False
    assert rand_manifest["candidate_archive"]["archive_sha256"]
    assert rand_manifest["dispatch_unlocked"] is True
    with zipfile.ZipFile(REPO / rand_manifest["candidate_archive"]["archive_path"]) as zf:
        parsed = parse_pr85_bundle(zf.read("x"))
    groups, _meta = module._decode_randmulti_stream(parsed.segments["randmulti"])
    assert groups[61][0][1] == 20

    bias_manifest = json.loads(
        (tmp_path / "out" / "pr85_qrgb_f1_bias_pair_0002" / "manifest.json").read_text(encoding="utf-8")
    )
    with zipfile.ZipFile(REPO / bias_manifest["candidate_archive"]["archive_path"]) as zf:
        parsed = parse_pr85_bundle(zf.read("x"))
    bias_values, _report = module._decode_choice_stream("bias", parsed.segments["bias"])
    assert bias_values[2] == 5


def test_builder_fails_closed_on_source_sha_mismatch(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    evidence, plan = _write_inputs(
        tmp_path,
        archive,
        [_candidate("pr85_qrgb_f1_bias_pair_0002", 2, "bias", 13, 5)],
        source_sha="0" * 64,
    )

    summary = module.build_qrgb_transfer_archive_candidates(
        source_archive=archive,
        pair_action_evidence_json=evidence,
        transfer_plan_json=plan,
        out_dir=tmp_path / "out",
        require_known_pr85_anchor=False,
    )

    assert summary["candidate_archive_count"] == 0
    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "source_sha_mismatch"


def test_builder_fails_closed_on_source_preserving_action(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    evidence, plan = _write_inputs(
        tmp_path,
        archive,
        [_candidate("pr85_qrgb_f1_bias_pair_0002", 2, "bias", 13, 13)],
    )

    summary = module.build_qrgb_transfer_archive_candidates(
        source_archive=archive,
        pair_action_evidence_json=evidence,
        transfer_plan_json=plan,
        out_dir=tmp_path / "out",
        require_known_pr85_anchor=False,
    )

    assert summary["candidate_archive_count"] == 0
    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "source_preserving_edit"


def test_builder_fails_closed_on_duplicate_actions(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    duplicate = _candidate("pr85_qrgb_f1_bias_pair_0002_dup", 2, "bias", 13, 5)
    evidence, plan = _write_inputs(
        tmp_path,
        archive,
        [
            _candidate("pr85_qrgb_f1_bias_pair_0002", 2, "bias", 13, 5),
            duplicate,
        ],
    )

    summary = module.build_qrgb_transfer_archive_candidates(
        source_archive=archive,
        pair_action_evidence_json=evidence,
        transfer_plan_json=plan,
        out_dir=tmp_path / "out",
        require_known_pr85_anchor=False,
    )

    assert summary["candidate_archive_count"] == 0
    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "duplicate_actions"


def test_builder_fails_closed_on_ungrounded_stream_value_evidence(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    candidate = _candidate("pr85_qrgb_f1_bias_pair_0002", 2, "bias", 13, 5)
    evidence, plan = _write_inputs(tmp_path, archive, [candidate])
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["ranked_candidates"][0]["basis_action_schema"] = {"target_stream": "region"}
    _write_json(plan, plan_payload)

    summary = module.build_qrgb_transfer_archive_candidates(
        source_archive=archive,
        pair_action_evidence_json=evidence,
        transfer_plan_json=plan,
        out_dir=tmp_path / "out",
        require_known_pr85_anchor=False,
    )

    assert summary["candidate_archive_count"] == 0
    assert summary["dispatch_unlocked"] is False
    assert summary["blocker_class"] == "ungrounded_stream_value_evidence"
