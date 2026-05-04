from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

brotli = pytest.importorskip("brotli")

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_pr85_randmulti_group_waterfill.py"


def _load_script():
    sys.path.insert(0, str(REPO / "experiments"))
    spec = importlib.util.spec_from_file_location("plan_pr85_randmulti_group_waterfill_test", SCRIPT)
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


def _randmulti_raw(*, pair0_value: int = 1, pair1_value: int = 1) -> bytes:
    raw = bytearray()
    for group_index, (_height, _width, _amplitude, scount) in enumerate(module.HEADERLESS_RANDMULTI_SPECS):
        for row_index in range(scount):
            row = bytearray(module.PAIR_COUNT)
            if group_index == 0 and row_index == 0:
                row[0] = pair0_value
            if group_index == 1 and row_index == 0:
                row[1] = pair1_value
            raw += module._encode_sparse_row(bytes(row))
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _auth(path: Path, *, archive_bytes: int, component_score: float) -> None:
    rate = archive_bytes * module.RATE_SCORE_PER_BYTE
    _write_json(
        path,
        {
            "archive_size_bytes": archive_bytes,
            "avg_posenet_dist": 0.0004,
            "avg_segnet_dist": 0.0005,
            "n_samples": module.EXPECTED_CONTEST_SAMPLES,
            "score_pose_contribution": component_score / 2.0,
            "score_rate_contribution": rate,
            "score_recomputed_from_components": component_score + rate,
            "score_seg_contribution": component_score / 2.0,
        },
    )


def _trace(path: Path, *, archive_bytes: int, pair_scores: dict[int, float], archive_sha: str = "a" * 64) -> None:
    samples = []
    for pair_index in range(module.EXPECTED_CONTEST_SAMPLES):
        score = pair_scores.get(pair_index, 0.0)
        samples.append(
            {
                "frame_indices": [pair_index * 2, pair_index * 2 + 1],
                "frame_start": pair_index * 2,
                "pair_index": pair_index,
                "posenet_dist": 0.0001,
                "score_combined_contribution_first_order": score,
                "score_pose_contribution_first_order": score / 2.0,
                "score_seg_contribution_exact": score / 2.0,
                "segnet_dist": 0.0002,
            }
        )
    _write_json(
        path,
        {
            "archive_size_bytes": archive_bytes,
            "n_samples": module.EXPECTED_CONTEST_SAMPLES,
            "samples": samples,
            "score_claim": False,
            "trace_inputs": {"archive_sha256": archive_sha},
        },
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    archive = tmp_path / "source.zip"
    _fixture_archive(archive)
    source_bytes = archive.stat().st_size
    baseline = tmp_path / "baseline"
    minus_randmulti = tmp_path / "minus_randmulti"
    minus_post = tmp_path / "minus_post"
    minus_motion = tmp_path / "minus_motion"

    _auth(baseline / "contest_auth_eval.adjudicated.json", archive_bytes=source_bytes, component_score=0.20)
    _auth(minus_randmulti / "contest_auth_eval.adjudicated.json", archive_bytes=source_bytes - 20, component_score=0.28)
    _auth(minus_post / "contest_auth_eval.adjudicated.json", archive_bytes=source_bytes - 10, component_score=0.25)
    _auth(minus_motion / "contest_auth_eval.adjudicated.json", archive_bytes=source_bytes - 8, component_score=0.30)

    _trace(baseline / "component_trace.json", archive_bytes=source_bytes, pair_scores={0: 0.01, 1: 0.01})
    _trace(
        minus_randmulti / "component_trace.json",
        archive_bytes=source_bytes - 20,
        pair_scores={0: 0.04, 1: 0.02},
    )
    _trace(minus_post / "component_trace.json", archive_bytes=source_bytes - 10, pair_scores={0: 0.03, 1: 0.01})
    _trace(minus_motion / "component_trace.json", archive_bytes=source_bytes - 8, pair_scores={0: 0.02, 1: 0.05})
    return {
        "archive": archive,
        "baseline_auth": baseline / "contest_auth_eval.adjudicated.json",
        "baseline_trace": baseline / "component_trace.json",
        "minus_randmulti_auth": minus_randmulti / "contest_auth_eval.adjudicated.json",
        "minus_randmulti_trace": minus_randmulti / "component_trace.json",
        "minus_post_auth": minus_post / "contest_auth_eval.adjudicated.json",
        "minus_post_trace": minus_post / "component_trace.json",
        "minus_motion_auth": minus_motion / "contest_auth_eval.adjudicated.json",
        "minus_motion_trace": minus_motion / "component_trace.json",
    }


def test_build_plan_decomposes_randmulti_groups_and_policies(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    out_dir = tmp_path / "out"

    summary = module.build_plan(
        archive=inputs["archive"],
        out_dir=out_dir,
        baseline_auth_json=inputs["baseline_auth"],
        baseline_trace_json=inputs["baseline_trace"],
        minus_randmulti_auth_json=inputs["minus_randmulti_auth"],
        minus_randmulti_trace_json=inputs["minus_randmulti_trace"],
        minus_post_auth_json=inputs["minus_post_auth"],
        minus_post_trace_json=inputs["minus_post_trace"],
        minus_motion_auth_json=inputs["minus_motion_auth"],
        minus_motion_trace_json=inputs["minus_motion_trace"],
        topks=(1, 2),
        budget_fractions=(1.0,),
    )

    assert summary["score_claim"] is False
    assert summary["remote_jobs_dispatched"] is False
    assert summary["top_group_ids_by_waterfill"][:2] == [0, 1]
    assert (out_dir / "randmulti_group_ledger.json").is_file()
    assert (out_dir / "candidate_policies.json").is_file()
    assert (out_dir / "candidate_summary.json").is_file()

    ledger = json.loads((out_dir / "randmulti_group_ledger.json").read_text())
    assert ledger["pair_delta_calibration"]["minus_randmulti"]["calibration_scale"] == pytest.approx(2.0)
    groups = {row["group_index"]: row for row in ledger["groups"]}
    assert groups[0]["estimated_component_score_rescue"] == pytest.approx(0.06)
    assert groups[1]["estimated_component_score_rescue"] == pytest.approx(0.02)
    assert groups[0]["post_trace_overlap_component_delta_mean"] > 0
    assert ledger["allocation_summary"]["allocated_component_score_delta"] == pytest.approx(0.08)
    assert ledger["atoms"][0]["group_index"] == 0

    policies = json.loads((out_dir / "candidate_policies.json").read_text())
    top1 = next(row for row in policies["policies"] if row["candidate_policy_id"] == "waterfill_top001")
    assert top1["selected_group_ids"] == [0]
    assert top1["planning_only"] is True
    assert top1["dispatch_gate"] == "planning_only/no_remote_dispatch"
    assert top1["score_claim"] is False


def test_plan_fails_closed_when_trace_custody_mismatches_auth(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _trace(
        inputs["minus_randmulti_trace"],
        archive_bytes=123,
        pair_scores={0: 0.04, 1: 0.02},
    )

    with pytest.raises(module.PlannerError, match="archive_size_bytes=123 does not match"):
        module.build_plan(
            archive=inputs["archive"],
            out_dir=tmp_path / "out",
            baseline_auth_json=inputs["baseline_auth"],
            baseline_trace_json=inputs["baseline_trace"],
            minus_randmulti_auth_json=inputs["minus_randmulti_auth"],
            minus_randmulti_trace_json=inputs["minus_randmulti_trace"],
            minus_post_auth_json=inputs["minus_post_auth"],
            minus_post_trace_json=inputs["minus_post_trace"],
            minus_motion_auth_json=inputs["minus_motion_auth"],
            minus_motion_trace_json=inputs["minus_motion_trace"],
        )


def test_cli_writes_candidate_summary(tmp_path: Path, capsys) -> None:
    inputs = _inputs(tmp_path)
    out_dir = tmp_path / "out"

    assert module.main(
        [
            "--archive",
            str(inputs["archive"]),
            "--out-dir",
            str(out_dir),
            "--baseline-auth-json",
            str(inputs["baseline_auth"]),
            "--baseline-trace-json",
            str(inputs["baseline_trace"]),
            "--minus-randmulti-auth-json",
            str(inputs["minus_randmulti_auth"]),
            "--minus-randmulti-trace-json",
            str(inputs["minus_randmulti_trace"]),
            "--minus-post-auth-json",
            str(inputs["minus_post_auth"]),
            "--minus-post-trace-json",
            str(inputs["minus_post_trace"]),
            "--minus-motion-auth-json",
            str(inputs["minus_motion_auth"]),
            "--minus-motion-trace-json",
            str(inputs["minus_motion_trace"]),
            "--topk",
            "1,2",
            "--budget-fraction",
            "1.0",
        ]
    ) == 0

    assert (out_dir / "candidate_summary.json").is_file()
    assert '"planning_only": true' in capsys.readouterr().out
