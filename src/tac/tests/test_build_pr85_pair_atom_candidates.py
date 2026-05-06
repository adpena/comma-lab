from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle, parse_pr85_bundle


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_pair_atom_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_pair_atom_candidates_test", SCRIPT)
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
        "randmulti": brotli.compress(bytes(72), quality=5),
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


def _plan(path: Path, archive: Path, *, pair_index: int = 1, stale: bool = False) -> Path:
    archive_sha = module._sha256_file(archive)
    payload = {
        "schema": module.SCORER_PLAN_SCHEMA,
        "producer": "fixture",
        "planning_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "compression_time_only": True,
        "inflate_time_scorer_load_allowed": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "exact_eval": {
            "archive_size_bytes": archive.stat().st_size,
            "avg_posenet_dist": 0.0002,
            "avg_segnet_dist": 0.0005,
            "n_samples": module.PAIR_COUNT,
            "reported_score": 0.25,
            "provenance": {
                "archive_sha256": "0" * 64 if stale else archive_sha,
                "archive_size_bytes": archive.stat().st_size,
                "device": "cuda",
                "gpu_model": "fixture",
                "tool": "experiments/contest_auth_eval.py",
            },
        },
        "dispatch_gates": {"dispatchable": False, "status": "blocked_planning_only"},
        "atom_ranking": [
            {
                "atom_id": f"fixture:pair_{pair_index:04d}",
                "pair_index": pair_index,
                "frame_indices": [pair_index * 2, pair_index * 2 + 1],
                "ranking_score": 0.0003,
                "byte_break_even": {
                    "combined": {
                        "max_charged_bytes_for_zero_net_change": 450.0,
                        "dscore_darchive_byte": module.RATE_SCORE_PER_BYTE,
                    }
                },
                "dispatch_gate": {"dispatchable": False, "status": "blocked_planning_only"},
            }
        ],
    }
    payload["stable_plan_digest_sha256"] = module._stable_digest(payload)
    return _write_json(path, payload)


def _action_spec(path: Path, *, pair_index: int = 1, value: int = 9) -> Path:
    return _write_json(
        path,
        {
            "schema": module.ACTION_SPEC_SCHEMA,
            "score_claim": False,
            "dispatch_performed": False,
            "inflate_time_scorer_load_allowed": False,
            "candidate_id": "pair1_frac2",
            "header_mode": "explicit_30",
            "actions": [
                {
                    "pair_index": pair_index,
                    "stream": "frac2",
                    "value": value,
                    "rationale": "fixture explicit pair action",
                }
            ],
        },
    )


def _runtime_contract(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": module.RUNTIME_CONTRACT_SCHEMA,
            "supports_pair_specific_actions": True,
            "scorer_load_allowed": False,
            "sidecars_allowed": False,
            "archive_member_contract": "single_member_x",
            "supported_streams": ["frac2"],
            "supported_header_modes": ["explicit_30"],
        },
    )


def test_synthetic_pair_atom_archive_and_manifest_are_deterministic(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    plan = _plan(tmp_path / "plan.json", archive)
    action = _action_spec(tmp_path / "action.json")
    runtime = _runtime_contract(tmp_path / "runtime.json")
    out_dir = tmp_path / "out"

    first = module.build_pair_atom_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=runtime,
        out_dir=out_dir,
        top_pairs=(1,),
        require_known_pr85_anchor=False,
    )
    first_manifest = json.loads((out_dir / "pair1_frac2" / "manifest.json").read_text(encoding="utf-8"))
    second = module.build_pair_atom_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=runtime,
        out_dir=out_dir,
        top_pairs=(1,),
        require_known_pr85_anchor=False,
    )
    second_manifest = json.loads((out_dir / "pair1_frac2" / "manifest.json").read_text(encoding="utf-8"))

    assert first["dispatch_unlocked"] is True
    assert second["dispatch_unlocked"] is True
    assert first_manifest == second_manifest
    assert first_manifest["non_noop_proof"]["status"] == "passed"
    assert first_manifest["dispatch_gate"] == "eligible_for_exact_eval_after_lane_claim"
    assert first_manifest["lane_claim_required_before_exact_eval"] is True
    assert first_manifest["score_claim"] is False
    assert first_manifest["dispatch_performed"] is False
    assert first_manifest["candidate_archive"]["archive_sha256"] == second_manifest["candidate_archive"]["archive_sha256"]

    candidate_raw = _source_member(out_dir / "pair1_frac2" / "archive.zip")
    assert module._sha256_bytes(candidate_raw) != module._sha256_bytes(_source_member(archive))
    candidate_bundle = parse_pr85_bundle(candidate_raw)
    decoded = brotli.decompress(candidate_bundle.segments["frac2"])
    assert decoded.startswith(b"FH2")
    assert decoded[3 + 1] == 9


def test_fail_closed_on_missing_runtime_support(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    plan = _plan(tmp_path / "plan.json", archive)
    action = _action_spec(tmp_path / "action.json")

    summary = module.build_pair_atom_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=None,
        out_dir=tmp_path / "out",
        top_pairs=(1,),
        require_known_pr85_anchor=False,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["candidate_archive_count"] == 0
    assert summary["blocker_class"] == "missing_pair_atom_runtime_contract"
    assert summary["candidates"][0]["dispatch_unlocked"] is False


def test_fail_closed_on_stale_scorer_gradient_source(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    plan = _plan(tmp_path / "stale_plan.json", archive, stale=True)
    action = _action_spec(tmp_path / "action.json")
    runtime = _runtime_contract(tmp_path / "runtime.json")

    summary = module.build_pair_atom_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=runtime,
        out_dir=tmp_path / "out",
        top_pairs=(1,),
        require_known_pr85_anchor=False,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["candidate_archive_count"] == 0
    assert summary["blocker_class"] == "stale_scorer_gradient_source"
    assert summary["candidates"][0]["build_status"] == "blocked"


def test_no_dispatch_unlocked_without_non_noop_proof(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    plan = _plan(tmp_path / "plan.json", archive)
    action = _action_spec(tmp_path / "noop_action.json", value=4)
    runtime = _runtime_contract(tmp_path / "runtime.json")

    summary = module.build_pair_atom_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=runtime,
        out_dir=tmp_path / "out",
        top_pairs=(1,),
        require_known_pr85_anchor=False,
    )

    assert summary["dispatch_unlocked"] is False
    assert summary["candidate_archive_count"] == 0
    assert summary["blocker_class"] == "non_noop_proof_failed"
    assert summary["candidates"][0]["blocker_class"] == "non_noop_proof_failed"
    assert not (tmp_path / "out" / "pair1_frac2" / "archive.zip").exists()
