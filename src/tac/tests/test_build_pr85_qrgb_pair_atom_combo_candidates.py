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
SCRIPT = REPO / "experiments" / "build_pr85_qrgb_pair_atom_combo_candidates.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_pr85_qrgb_pair_atom_combo_candidates_test", SCRIPT)
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


def _write_archive(path: Path) -> Path:
    values = bytes([4]) * pair_builder.PAIR_COUNT
    segments = {
        "mask": b"QMA9" + b"M" * 64,
        "model": brotli.compress(b"QH0" + b"R" * 64, quality=5),
        "pose": brotli.compress(b"P1D1" + bytes([1, 0]) + (600).to_bytes(2, "little") + bytes(600), quality=5),
        "post": brotli.compress(bytes(pair_builder.PAIR_COUNT * 4), quality=5),
        "shift": brotli.compress(b"SD4" + bytes(pair_builder.PAIR_COUNT), quality=5),
        "frac": brotli.compress(b"FH1" + values, quality=5),
        "frac2": brotli.compress(b"FH2" + values, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes(pair_builder.PAIR_COUNT), quality=5),
        "bias": brotli.compress(b"BD1" + bytes(pair_builder.PAIR_COUNT), quality=5),
        "region": brotli.compress(b"RH1" + bytes(pair_builder.PAIR_COUNT), quality=5),
        "randmulti": brotli.compress(bytes(72), quality=5),
    }
    assert set(segments) == set(SEGMENT_ORDER)
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("x"), raw)
    return path


def _plan(path: Path, archive: Path, pair_indices: tuple[int, ...] = (1, 2, 3)) -> Path:
    archive_sha = pair_builder._sha256_file(archive)
    payload = {
        "schema": pair_builder.SCORER_PLAN_SCHEMA,
        "producer": "fixture",
        "planning_only": True,
        "score_claim": False,
        "compression_time_only": True,
        "inflate_time_scorer_load_allowed": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "exact_eval": {
            "archive_size_bytes": archive.stat().st_size,
            "avg_posenet_dist": 0.0002,
            "avg_segnet_dist": 0.0005,
            "n_samples": pair_builder.PAIR_COUNT,
            "reported_score": 0.25,
            "provenance": {
                "archive_sha256": archive_sha,
                "archive_size_bytes": archive.stat().st_size,
                "device": "cuda",
                "tool": "experiments/contest_auth_eval.py",
            },
        },
        "atom_ranking": [
            {
                "atom_id": f"fixture:pair_{pair_index:04d}",
                "pair_index": pair_index,
                "frame_indices": [pair_index * 2, pair_index * 2 + 1],
                "ranking_score": 0.0003,
                "byte_break_even": {
                    "combined": {
                        "max_charged_bytes_for_zero_net_change": 450.0,
                        "dscore_darchive_byte": pair_builder.RATE_SCORE_PER_BYTE,
                    }
                },
                "dispatch_gate": {"dispatchable": False, "status": "blocked_planning_only"},
            }
            for pair_index in pair_indices
        ],
    }
    payload["stable_plan_digest_sha256"] = pair_builder._stable_digest(payload)
    return _write_json(path, payload)


def _runtime_contract(path: Path) -> Path:
    return _write_json(
        path,
        {
            "schema": module.RUNTIME_CONTRACT_SCHEMA,
            "supports_pair_specific_actions": True,
            "scorer_load_allowed": False,
            "sidecars_allowed": False,
            "archive_member_contract": "single_member_x",
            "supported_streams": ["bias", "region"],
            "supported_header_modes": ["explicit_30"],
        },
    )


def _action_spec(path: Path, *, duplicate_pair: bool = False, unsupported: bool = False) -> Path:
    rows = [
        ("pr85_qrgb_fixture_bias_pair_0001", 1, "bias", 9, 0),
        ("pr85_qrgb_fixture_bias_pair_0002", 1 if duplicate_pair else 2, "bias", 10, 0),
        ("pr85_qrgb_fixture_region_pair_0003", 3, "randmulti" if unsupported else "region", 11, 0),
    ]
    return _write_json(
        path,
        {
            "schema": module.ACTION_SPEC_SCHEMA,
            "score_claim": False,
            "dispatch_performed": False,
            "remote_jobs_dispatched": False,
            "inflate_time_scorer_load_allowed": False,
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "header_mode": "explicit_30",
                    "actions": [
                        {
                            "op": "set",
                            "pair_index": pair_index,
                            "stream": stream,
                            "value": value,
                            "source_value": source_value,
                            "rationale": "fixture combo atom",
                        }
                    ],
                }
                for candidate_id, pair_index, stream, value, source_value in rows
            ],
        },
    )


def _seed_singletons(singleton_dir: Path, source_archive: Path, expected: dict[str, dict]) -> None:
    for candidate_id, row in expected.items():
        candidate_dir = singleton_dir / candidate_id
        archive = {
            "archive_bytes": row["archive_bytes"],
            "archive_sha256": row["archive_sha256"],
            "archive_path": f"fixture/{candidate_id}/archive.zip",
            "member_sha256": "f" * 64,
        }
        _write_json(
            candidate_dir / "manifest.json",
            {
                "build_status": "built",
                "score_claim": False,
                "candidate_archive": archive,
                "source_archive": {"archive_sha256": pair_builder._sha256_file(source_archive)},
            },
        )
        _write_json(
            candidate_dir / "fixed_runtime_atom_preflight.json",
            {
                "ready_for_fixed_runtime_exact_eval": True,
                "archive": archive,
            },
        )


def test_combo_builder_emits_two_and_three_atom_archives_from_fixture(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    plan = _plan(tmp_path / "plan.json", archive)
    action = _action_spec(tmp_path / "action.json")
    runtime = _runtime_contract(tmp_path / "runtime.json")
    expected = {
        "pr85_qrgb_fixture_bias_pair_0001": {"archive_bytes": 101, "archive_sha256": "1" * 64},
        "pr85_qrgb_fixture_bias_pair_0002": {"archive_bytes": 102, "archive_sha256": "2" * 64},
        "pr85_qrgb_fixture_region_pair_0003": {"archive_bytes": 103, "archive_sha256": "3" * 64},
    }
    _seed_singletons(tmp_path / "singletons", archive, expected)

    summary = module.build_combo_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=runtime,
        singleton_dir=tmp_path / "singletons",
        out_dir=tmp_path / "out",
        combo_sizes=(2, 3),
        run_fixed_runtime_preflight=False,
        require_known_pr85_anchor=False,
        expected_singletons=expected,
    )

    assert summary["combo_archive_count"] == 4
    assert summary["ready_combo_count"] == 0
    assert summary["blocker_class"] == "missing_preflight"
    assert (tmp_path / "out" / "action_spec_combos.json").is_file()
    built = [row for row in summary["candidates"] if row["build_status"] == "built"]
    assert len(built) == 4
    for row in built:
        assert row["score_claim"] is False
        assert row["dispatch_performed"] is False
        assert row["dispatch_unlocked"] is False
        with zipfile.ZipFile(REPO / row["candidate_archive"]["archive_path"]) as zf:
            parsed = parse_pr85_bundle(zf.read("x"))
        assert set(parsed.segments) == set(SEGMENT_ORDER)


def test_combo_builder_fails_closed_on_duplicate_pair_conflict(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "source.zip")
    plan = _plan(tmp_path / "plan.json", archive, pair_indices=(1, 3))
    action = _action_spec(tmp_path / "action.json", duplicate_pair=True)
    runtime = _runtime_contract(tmp_path / "runtime.json")
    expected = {
        "pr85_qrgb_fixture_bias_pair_0001": {"archive_bytes": 101, "archive_sha256": "1" * 64},
        "pr85_qrgb_fixture_bias_pair_0002": {"archive_bytes": 102, "archive_sha256": "2" * 64},
        "pr85_qrgb_fixture_region_pair_0003": {"archive_bytes": 103, "archive_sha256": "3" * 64},
    }
    _seed_singletons(tmp_path / "singletons", archive, expected)

    summary = module.build_combo_candidates(
        source_archive=archive,
        scorer_plan_json=plan,
        action_spec_json=action,
        runtime_contract_json=runtime,
        singleton_dir=tmp_path / "singletons",
        out_dir=tmp_path / "out",
        combo_sizes=(2,),
        run_fixed_runtime_preflight=False,
        require_known_pr85_anchor=False,
        expected_singletons=expected,
    )

    blocked = [row for row in summary["candidates"] if row.get("blocker_class") == "duplicate_pair_segment_conflict"]
    assert blocked
    assert all(row["dispatch_unlocked"] is False for row in blocked)


def test_combo_builder_fails_closed_on_unsupported_stream(tmp_path: Path) -> None:
    action = _action_spec(tmp_path / "action.json", unsupported=True)
    report, candidates = module._action_candidates(action)

    assert report["status"] == "blocked"
    assert report["blocker_class"] == "unsupported_stream"
    assert {row["candidate_id"] for row in candidates} == {
        "pr85_qrgb_fixture_bias_pair_0001",
        "pr85_qrgb_fixture_bias_pair_0002",
    }
