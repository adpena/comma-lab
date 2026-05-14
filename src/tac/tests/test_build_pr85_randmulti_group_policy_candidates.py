# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "build_pr85_randmulti_group_policy_candidates.py"


def _load_script():
    sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location("build_pr85_randmulti_policy_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _u24(value: int) -> bytes:
    return int(value).to_bytes(3, "little")


def _zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _randmulti_raw() -> bytes:
    raw = bytearray()
    for group_index, (_height, _width, _amplitude, scount) in enumerate(
        module.plan.HEADERLESS_RANDMULTI_SPECS
    ):
        for row_index in range(scount):
            row = bytearray(module.plan.PAIR_COUNT)
            if group_index == 0 and row_index == 0:
                row[0] = 3
                row[17] = 1
            if group_index == 1 and row_index == 0:
                row[1] = 2
            raw += module.plan._encode_sparse_row(bytes(row))
    return bytes(raw)


def _fixture_archive(path: Path) -> None:
    segments = {
        "mask": b"QMA9" + b"M" * 1001,
        "model": b"QH0" + b"W" * 1001,
        "pose": b"P1D1" + b"P" * 101,
        "post": brotli.compress(bytes(600 * 4), quality=5),
        "shift": brotli.compress(b"SH4" + bytes([40]) * 600, quality=5),
        "frac": brotli.compress(b"FH1" + bytes([4]) * 600, quality=5),
        "frac2": brotli.compress(b"FH2" + bytes([4]) * 600, quality=5),
        "frac3": brotli.compress(b"FH3" + bytes([4]) * 600, quality=5),
        "bias": b"B" * module.recode.FIXED_V5_LENGTHS["bias"],
        "region": b"R" * module.recode.FIXED_V5_LENGTHS["region"],
        "randmulti": brotli.compress(_randmulti_raw(), quality=5),
    }
    header = b"".join(_u24(len(segments[name])) for name in module.recode.SEGMENT_ORDER[:8])
    payload = header + b"".join(segments[name] for name in module.recode.SEGMENT_ORDER)
    _zip(path, payload)


def _policy_json(path: Path) -> None:
    payload = {
        "schema": "pr85_randmulti_group_policy_candidates_v1",
        "score_claim": False,
        "planning_only": True,
        "remote_jobs_dispatched": False,
        "policies": [
            {
                "candidate_policy_id": "keep_group0",
                "estimated_component_score_rescue": 0.01,
                "estimated_net_score_rescue_after_rate": 0.009,
                "planning_only": True,
                "policy_hash": "fixture",
                "score_claim": False,
                "selected_group_ids": [0],
            },
            {
                "candidate_policy_id": "keep_nothing",
                "estimated_component_score_rescue": 0.0,
                "estimated_net_score_rescue_after_rate": 0.0,
                "planning_only": True,
                "policy_hash": "zero",
                "score_claim": False,
                "selected_group_ids": [],
            },
        ],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _candidate_groups(archive: Path):
    _source_archive, raw = module.recode._read_pr85_archive(archive)
    _bundle, segments = module.recode._parse_bundle(raw)
    return module.plan._decode_pr85_randmulti_groups(brotli.decompress(segments["randmulti"]))


def test_builds_group_policy_candidate_with_selected_group_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    policies = tmp_path / "policies.json"
    out_dir = tmp_path / "out"
    _fixture_archive(source)
    _policy_json(policies)

    payload = module.build_candidates(
        archive=source,
        policy_json=policies,
        out_dir=out_dir,
        policy_ids=["keep_group0"],
    )

    assert payload["score_claim"] is False
    assert payload["dispatch_performed"] is False
    assert payload["candidate_count"] == 1
    manifest = payload["candidates"][0]
    assert manifest["policy_id"] == "keep_group0"
    assert manifest["candidate"]["member_name"] == "x"
    assert manifest["candidate"]["zip_stored"] is True
    assert manifest["candidate_bundle_validation"]["status"] == "passed"
    assert manifest["randmulti_transform"]["group_semantics"]["status"] == "passed"
    assert manifest["randmulti_transform"]["group_semantics"]["preserved_group_ids"] == [0]
    assert 1 in manifest["randmulti_transform"]["group_semantics"]["zero_group_ids"]
    assert manifest["randmulti_transform"]["group_semantics"]["source_nonzero_choice_total"] == 3
    assert manifest["randmulti_transform"]["group_semantics"]["candidate_nonzero_choice_total"] == 2
    group_profiles = {
        row["group_index"]: row
        for row in manifest["randmulti_transform"]["group_semantics"]["group_profiles"]
    }
    assert group_profiles[0]["selected"] is True
    assert group_profiles[0]["source_nonzero_choice_count"] == 2
    assert group_profiles[0]["candidate_nonzero_choice_count"] == 2
    assert group_profiles[1]["selected"] is False
    assert group_profiles[1]["source_nonzero_choice_count"] == 1
    assert group_profiles[1]["candidate_nonzero_choice_count"] == 0
    assert manifest["byte_delta_vs_source_archive"] < 0
    assert manifest["dispatch_gate"] == "eligible_for_cuda_auth_eval_after_lane_claim"

    source_groups = _candidate_groups(source)
    candidate_groups = _candidate_groups(out_dir / "keep_group0" / "archive.zip")
    assert candidate_groups[0].rows == source_groups[0].rows
    assert all(row == bytes(module.plan.PAIR_COUNT) for row in candidate_groups[1].rows)


def test_empty_policy_is_planning_only_not_dispatchable(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    policies = tmp_path / "policies.json"
    out_dir = tmp_path / "out"
    _fixture_archive(source)
    _policy_json(policies)

    payload = module.build_candidates(
        archive=source,
        policy_json=policies,
        out_dir=out_dir,
        policy_ids=["keep_nothing"],
    )

    manifest = payload["candidates"][0]
    assert manifest["policy_id"] == "keep_nothing"
    assert manifest["source_policy"]["selected_group_ids"] == []
    assert manifest["dispatch_gate"] == "planning_only/no_remote_dispatch"
    assert payload["dispatchable_candidate_count"] == 0


def test_duplicate_group_policy_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    policies = tmp_path / "policies.json"
    out_dir = tmp_path / "out"
    _fixture_archive(source)
    policies.write_text(
        json.dumps(
            {
                "policies": [
                    {
                        "candidate_policy_id": "duplicate",
                        "selected_group_ids": [0, 0],
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate selected_group_ids"):
        module.build_candidates(
            archive=source,
            policy_json=policies,
            out_dir=out_dir,
            policy_ids=["duplicate"],
        )


def test_cli_writes_summary(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.zip"
    policies = tmp_path / "policies.json"
    out_dir = tmp_path / "out"
    _fixture_archive(source)
    _policy_json(policies)

    assert module.main(
        [
            "--archive",
            str(source),
            "--policy-json",
            str(policies),
            "--out-dir",
            str(out_dir),
            "--policy",
            "keep_group0",
        ]
    ) == 0

    assert (out_dir / "candidate_summary.json").is_file()
    assert '"policy_id": "keep_group0"' in capsys.readouterr().out
