from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "plan_frontier_stack_reconstruction.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("frontier_stack_reconstruction_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_script()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_zip(path: Path, *, member: str, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        info = zipfile.ZipInfo(member)
        info.compress_type = zipfile.ZIP_STORED
        zf.writestr(info, payload)
    return path


def _sha(path: Path) -> str:
    return module._sha256_file(path)


def _exact_score(archive_bytes: int, seg: float = 0.0005, pose: float = 0.0001) -> float:
    return module._score_from_components(
        archive_bytes=archive_bytes,
        seg_dist=seg,
        pose_dist=pose,
    )


def _fixture_inputs(tmp_path: Path) -> dict[str, list[str]]:
    pr85_archive = _write_zip(
        tmp_path / "experiments/results/public_pr85_intake_20260503_codex/archive.zip",
        member="x",
        payload=b"a" * 200,
    )
    pr91_archive = _write_zip(
        tmp_path / "experiments/results/public_pr91_intake_20260504_worker/archive.zip",
        member="x",
        payload=b"b" * 180,
    )
    pr90_archive = _write_zip(
        tmp_path / "experiments/results/public_pr90_intake_20260504_worker/archive.zip",
        member="p",
        payload=b"c" * 190,
    )
    stbm_archive = _write_zip(
        tmp_path
        / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip",
        member="x",
        payload=b"d" * 170,
    )
    qrgb_archive = _write_zip(
        tmp_path
        / "experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/pr85_qrgb_f1_bias_pair_0060/archive.zip",
        member="x",
        payload=b"e" * 205,
    )
    pr85_eval = _write_json(
        tmp_path
        / "experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4/contest_auth_eval.adjudicated.json",
        {
            "archive_size_bytes": pr85_archive.stat().st_size,
            "avg_posenet_dist": 0.0001,
            "avg_segnet_dist": 0.0005,
            "n_samples": 600,
            "provenance": {
                "archive_sha256": _sha(pr85_archive),
                "archive_size_bytes": pr85_archive.stat().st_size,
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "gpu_t4_match": True,
            },
            "score_recomputed_from_components": _exact_score(pr85_archive.stat().st_size),
        },
    )
    qrgb_eval = _write_json(
        tmp_path
        / "experiments/results/lightning_batch/exact_eval_pr85_qrgb_f1_bias_pair_0060_t4/contest_auth_eval.adjudicated.json",
        {
            "archive_size_bytes": qrgb_archive.stat().st_size,
            "avg_posenet_dist": 0.000101,
            "avg_segnet_dist": 0.000501,
            "n_samples": 600,
            "provenance": {
                "archive_sha256": _sha(qrgb_archive),
                "archive_size_bytes": qrgb_archive.stat().st_size,
                "device": "cuda",
                "gpu_model": "Tesla T4",
                "gpu_t4_match": True,
            },
            "score_recomputed_from_components": _exact_score(
                qrgb_archive.stat().st_size,
                seg=0.000501,
                pose=0.000101,
            ),
        },
    )
    pr85_profile = _write_json(
        tmp_path / "experiments/results/public_pr85_intake_20260503_codex/profile_pr85_bundle.json",
        {
            "segments": [
                {"name": "mask", "bytes": 100, "sha256": "mask85"},
                {"name": "model", "bytes": 60, "sha256": "model"},
            ],
            "score_claim": False,
        },
    )
    pr91_anatomy = _write_json(
        tmp_path / "experiments/results/public_pr91_intake_20260504_worker/pr91_archive_anatomy.json",
        {
            "segments": [
                {"name": "mask", "bytes": 80, "sha256": "mask91"},
                {"name": "model", "bytes": 60, "sha256": "model"},
            ],
            "score_claim": False,
        },
    )
    pr91_decisions = _write_json(
        tmp_path / "experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json",
        {
            "pr85_diff": {"archive_delta_bytes": -20, "mask_delta_bytes": -20},
            "source_pr": {
                "claimed_report": {
                    "archive_bytes": pr91_archive.stat().st_size,
                    "exact_score_external_text": 0.2,
                    "pose_dist": 0.0001,
                    "seg_dist": 0.0005,
                }
            },
            "score_claim": False,
            "transfer_opportunities": [],
        },
    )
    pr90_probe = _write_json(
        tmp_path / "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json",
        {
            "slices": {
                "mask_body": {"offset": 0, "len": 90, "runtime_magic": "STBM1BR\\0"},
                "model_body": {"offset": 90, "len": 55, "runtime_magic": "QFQ4\\0"},
                "pose_qrgb_body": {"offset": 145, "len": 45},
            },
            "score_claim": False,
        },
    )
    pr90_pull = _write_json(
        tmp_path / "experiments/results/public_pr90_intake_20260504_worker/pr90_pull.json",
        {
            "body": (
                "Average PoseNet Distortion: 0.0004\n"
                "Average SegNet Distortion: 0.0006\n"
                "Submission file size: 190 bytes\n"
                "Final score: 0.28\n"
            )
        },
    )
    stbm_summary = _write_json(
        tmp_path / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/candidate_summary.json",
        {"score_claim": False, "dispatch_performed": False},
    )
    stack_summary = _write_json(
        tmp_path / "experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/candidate_summary.json",
        {
            "candidate_archive": {"archive_bytes": stbm_archive.stat().st_size},
            "fixed_runtime_preflight": {"readiness_status": "ready"},
            "score_claim": False,
            "dispatch_unlocked": False,
        },
    )
    return {
        "archive_globs": [
            str(pr85_archive.relative_to(tmp_path)),
            str(pr91_archive.relative_to(tmp_path)),
            str(pr90_archive.relative_to(tmp_path)),
            str(stbm_archive.relative_to(tmp_path)),
            str(qrgb_archive.relative_to(tmp_path)),
        ],
        "exact_eval_globs": [
            str(pr85_eval.relative_to(tmp_path)),
            str(qrgb_eval.relative_to(tmp_path)),
        ],
        "json_globs": [
            str(pr85_profile.relative_to(tmp_path)),
            str(pr91_anatomy.relative_to(tmp_path)),
            str(pr91_decisions.relative_to(tmp_path)),
            str(pr90_probe.relative_to(tmp_path)),
            str(pr90_pull.relative_to(tmp_path)),
            str(stbm_summary.relative_to(tmp_path)),
            str(stack_summary.relative_to(tmp_path)),
        ],
    }


def _by_id(plan: dict) -> dict[str, dict]:
    return {row["opportunity_id"]: row for row in plan["ranked_opportunities"]}


def test_frontier_plan_profiles_archives_and_ranks_blocked_frontier_work(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)

    plan = module.build_plan(repo_root=tmp_path, **inputs)
    records = _by_id(plan)

    assert plan["planning_only"] is True
    assert plan["score_claim"] is False
    assert plan["dispatch_performed"] is False
    assert plan["baseline_pr85_exact_anchor"]["evidence_status"] == "exact_cuda_full_600"
    assert all(profile["strict_zip_ok"] for profile in plan["archive_profiles"])
    assert all(profile["zip_overhead_bytes"] == 100 for profile in plan["archive_profiles"])
    assert plan["external_public_reports"]["pr91"]["score_claim"] is False
    assert plan["external_public_reports"]["pr90"]["evidence_status"] == (
        "external_public_pr_text_not_local_exact"
    )

    hpm1 = records["recover_pr91_hpm1_mask_contract_on_pr85_runtime"]
    assert hpm1["expected_bytes_saved_vs_pr85"] == 20
    assert hpm1["blocked"] is True
    assert hpm1["score_claim"] is False
    assert "full local HPM1 decode" in hpm1["required_gates"][0]

    stbm = records["exact_eval_pr85_stbm1br_lossless_mask_recode"]
    assert stbm["expected_bytes_saved_vs_pr85"] == 30
    assert stbm["blocked"] is False

    pr90 = records["pr90_topband_geometry_mask_prior_for_pr85"]
    assert pr90["expected_bytes_saved_vs_pr85"] == 10
    assert pr90["dispatch_readiness"] == "needs_local_builder"

    qrgb = records["pr85_qrgb_pair_atoms_negative_guardrail"]
    assert qrgb["refuted"] is True
    assert qrgb["evidence_status"] == "exact_cuda_negative_or_non_improving"


def test_frontier_plan_digest_and_outputs_are_stable(tmp_path: Path) -> None:
    inputs = _fixture_inputs(tmp_path)

    first = module.build_plan(repo_root=tmp_path, **inputs)
    second = module.build_plan(repo_root=tmp_path, **inputs)
    outputs = module.write_outputs(first, tmp_path / "out")

    assert first == second
    assert first["stable_plan_digest_sha256"] == second["stable_plan_digest_sha256"]
    json_path = tmp_path / "out" / "frontier_stack_reconstruction_plan.json"
    md_path = tmp_path / "out" / "frontier_stack_reconstruction_plan.md"
    assert outputs["json"].endswith("frontier_stack_reconstruction_plan.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["planning_only"] is True
    assert "Frontier Stack Reconstruction Plan" in md_path.read_text(encoding="utf-8")
