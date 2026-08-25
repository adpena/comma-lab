"""Behavioral controls for the S1A both-OFF endpoint adjudicator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO / "tools/s1a_off_floor_adjudicator.py"
    spec = importlib.util.spec_from_file_location("s1a_off_floor_adjudicator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADJ = _load_module()


def _sha_token(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _make_seed(
    root: Path,
    *,
    seed: int,
    epochs: tuple[int, ...] = (5, 65),
    packet_bytes: int = 30_000,
    hard_d_seg: float = ADJ.GB1_HARD_D_SEG,
    d_pose: float = ADJ.GB1_D_POSE,
    archive_bytes: int = 1,
) -> None:
    seed_root = root / f"off_seed_{seed}" / "W96_flattened"
    allocation_sha = _sha_token(f"allocation-{seed}")
    selection_sha = _sha_token(f"selection-{seed}")
    controller = {
        "schema": "ddm_wd3_stage_controller.v1",
        "complete": True,
        "all_payloads_retained": True,
        "chosen_allocation": {
            "schema": "ddm_wd3_adaptive_quant_allocation.v1",
            "policy": "uniform_int4_degenerate",
            "selection_sha256": selection_sha,
            "bits": {"weight": [4, 4]},
        },
        "chosen_allocation_sha256": allocation_sha,
        "cheap_to_shrink_ladder": {
            "active": False,
            "allocation_family": "uniform_bits",
            "base_bytes": packet_bytes,
            "byte_cost_checked": True,
            "rung_bytes": [],
        },
        "quantization_race": [
            {
                "allocation_id": "uniform4",
                "hard_cell_gate_pass": True,
                "measured": True,
                "packet_bytes": packet_bytes,
                "parse_back_exact": True,
                "pose_gate_pass": True,
                "retained_payload": True,
                "road_lane_gate_pass": True,
            }
        ],
    }
    _write_json(
        seed_root / "stage_controllers/stage_04_from_epoch_0000/STAGE_CONTROLLER_RESULT.json",
        controller,
    )
    for epoch in epochs:
        checkpoint = seed_root / "checkpoints" / f"wd3_epoch_{epoch:04d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{seed}-{epoch}".encode())
        retained = seed_root / "retained/evaluations" / f"epoch_{epoch:04d}_n60"
        archive = _artifact(retained / "candidate/archive.zip", b"a" * archive_bytes)
        archive_repeat = _artifact(retained / "candidate/archive.repeat.zip", b"a" * archive_bytes)
        member = _artifact(retained / "candidate/p", b"member")
        semantic = _artifact(retained / "candidate/semantic.ck2.br", b"semantic")
        student_packet = _artifact(retained / "candidate/semantic.wd3q", b"s" * packet_bytes)
        submission_archive = _artifact(
            retained / "candidate/submission/archive.zip",
            b"a" * archive_bytes,
        )
        receiver_pairs = _artifact(retained / "receiver_pairs.rgb.u8", b"receiver-pairs")
        scorer_bundle = _artifact(retained / "scorer_outputs.npz", b"scorer-bundle")
        transcript = _artifact(retained / "candidate/PARSEBACK_TRANSCRIPT.txt", b"pass")
        section_receipt = _artifact(retained / "candidate/SECTION_PRESERVATION.json", b"{}")
        _write_json(
            seed_root / "evaluations" / f"epoch_{epoch:04d}_n60.json",
            {
                "schema": "ddm_wd3_retained_subset_evaluation.v1",
                "axis": ADJ.EXPECTED_AXIS,
                "score_claim": False,
                "all_payloads_retained": True,
                "n_pairs": 60,
                "pair_ids": list(ADJ.EXPECTED_PAIR_IDS),
                "hard_d_seg": hard_d_seg,
                "d_pose": d_pose,
                "seg_contribution": 100.0 * hard_d_seg,
                "pose_contribution": (10.0 * d_pose) ** 0.5,
                "evaluation_binding": {
                    "allocation_sha256": allocation_sha,
                    "student_packet_sha256": student_packet["sha256"],
                },
                "receiver_pairs": receiver_pairs,
                "scorer_bundle": scorer_bundle,
                "packet_archive": {
                    "schema": "ddm_wd3_retained_packet_archive.v1",
                    "archive_binding": submission_archive,
                    "archive_bytes": archive_bytes,
                    "archive_repeat_byte_identical": True,
                    "receiver_parse_back_exact": True,
                    "untouched_sections_byte_identical": True,
                    "payloads": {
                        "archive": archive,
                        "archive_repeat": archive_repeat,
                        "member": member,
                        "semantic_ck2_brotli_q11": semantic,
                        "student_packet": student_packet,
                    },
                    "parseback": {
                        "status": "PASS",
                        "report": {"packet_exact": True, "repack_exact": True},
                        "transcript": transcript,
                    },
                    "section_preservation_receipt": section_receipt,
                    "allocation": {
                        "packet_bytes": packet_bytes,
                        "policy": "uniform_int4_degenerate",
                        "selection_sha256": selection_sha,
                    },
                },
            },
        )


def test_two_complete_seeds_refuse_all_points_when_damage_exceeds_credit(
    tmp_path: Path,
) -> None:
    for seed in ADJ.EXPECTED_SEEDS:
        _make_seed(
            tmp_path,
            seed=seed,
            packet_bytes=30_000,
            hard_d_seg=0.001,
            d_pose=0.01,
        )
    result = ADJ.adjudicate(tmp_path)
    assert result["falsifier_verdict"] == ADJ.ENTERED_AND_REFUSED
    assert result["all_seed_endpoints_complete"] is True
    assert result["row_count"] == 4
    assert all(row["falsifier_verdict"] == ADJ.ENTERED_AND_REFUSED for row in result["rows"])
    assert all(row["point_crosses_renderer_corner"] is False for row in result["rows"])
    assert result["score_claim"] is False


def test_any_negative_composed_delta_crosses_corner(tmp_path: Path) -> None:
    for seed in ADJ.EXPECTED_SEEDS:
        _make_seed(tmp_path, seed=seed, packet_bytes=30_000)
    result = ADJ.adjudicate(tmp_path)
    assert result["falsifier_verdict"] == ADJ.CORNER_CROSSED
    assert result["corner_crossing_point_count"] == 4
    assert all(row["composed_delta_S_vs_break_even"] < 0 for row in result["rows"])


def test_one_finished_seed_is_a_valid_incomplete_positive_control(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815, packet_bytes=38_847)
    result = ADJ.adjudicate(tmp_path)
    assert result["falsifier_verdict"] == ADJ.INCOMPLETE_DATA
    assert result["all_seed_endpoints_complete"] is False
    assert result["seed_summaries"][0]["status"] == "ENDPOINT_COMPLETE"
    assert result["seed_summaries"][1]["status"] == "MISSING_SEED_STORE"
    assert all(row["falsifier_verdict"] == ADJ.INCOMPLETE_DATA for row in result["rows"])


def test_byte_numerator_comes_from_controller_not_archive_size(tmp_path: Path) -> None:
    for seed in ADJ.EXPECTED_SEEDS:
        _make_seed(
            tmp_path,
            seed=seed,
            packet_bytes=38_847,
            archive_bytes=17,
            hard_d_seg=0.001,
            d_pose=0.01,
        )
    result = ADJ.adjudicate(tmp_path)
    row = result["rows"][0]
    assert row["bytes_shed_vs_gb1_renderer_30856B"] == 30_856 - 38_847
    assert row["candidate_archive_observation"]["archive_bytes"] == 17
    assert row["renderer_rate_credit_S_at_6_658e_7_per_B"] == pytest.approx((30_856 - 38_847) * ADJ.RATE_PER_BYTE)


def test_prior_law_falsifier_counts_seg_dominant_points(tmp_path: Path) -> None:
    for seed in ADJ.EXPECTED_SEEDS:
        _make_seed(
            tmp_path,
            seed=seed,
            packet_bytes=30_000,
            hard_d_seg=0.02,
            d_pose=ADJ.GB1_D_POSE,
        )
    result = ADJ.adjudicate(tmp_path)
    prediction = result["prior_law_prediction"]
    assert prediction["falsified_at_least_one_checkpoint"] is True
    assert prediction["seg_term_exceeds_pose_term_count"] == result["row_count"]
    assert len(prediction["falsifying_points"]) == result["row_count"]


def test_refuses_controller_race_bytes_that_do_not_match_base(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815, packet_bytes=30_000)
    controller_path = next(tmp_path.rglob("STAGE_CONTROLLER_RESULT.json"))
    controller = json.loads(controller_path.read_text())
    controller["quantization_race"][0]["packet_bytes"] = 29_999
    _write_json(controller_path, controller)
    with pytest.raises(ADJ.AdjudicationError, match="ladder base bytes"):
        ADJ.adjudicate(tmp_path)


def test_refuses_prefix_or_other_subset_selection(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815, packet_bytes=30_000)
    evaluation_path = next(tmp_path.rglob("epoch_0005_n60.json"))
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["pair_ids"] = list(range(60))
    _write_json(evaluation_path, evaluation)
    with pytest.raises(ADJ.AdjudicationError, match="evenly-strided"):
        ADJ.adjudicate(tmp_path)


def test_refuses_evaluation_that_claims_a_score(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815, packet_bytes=30_000)
    evaluation_path = next(tmp_path.rglob("epoch_0005_n60.json"))
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["score_claim"] = True
    _write_json(evaluation_path, evaluation)
    with pytest.raises(ADJ.AdjudicationError, match="score_claim=false"):
        ADJ.adjudicate(tmp_path)


def test_cli_refuses_output_inside_read_only_training_tree(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815, packet_bytes=30_000)
    assert ADJ.main(["--training-root", str(tmp_path), "--out", str(tmp_path / "out.json")]) == 2


def test_existing_seed_store_without_controller_result_is_incomplete(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815, hard_d_seg=0.001, d_pose=0.01)
    partial = tmp_path / "off_seed_20260816/W96_flattened/stage_controllers/stage_04_from_epoch_0000"
    partial.mkdir(parents=True)
    result = ADJ.adjudicate(tmp_path)
    assert result["falsifier_verdict"] == ADJ.INCOMPLETE_DATA
    assert result["seed_summaries"][1]["status"] == "MISSING_STAGE_CONTROLLER_RESULT"


def test_refuses_tampered_retained_payload(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815)
    evaluation_path = next(tmp_path.rglob("epoch_0005_n60.json"))
    evaluation = json.loads(evaluation_path.read_text())
    retained_path = Path(evaluation["receiver_pairs"]["path"])
    retained_path.write_bytes(b"tampered")
    with pytest.raises(ADJ.AdjudicationError, match="receiver_pairs retained bytes mismatch"):
        ADJ.adjudicate(tmp_path)


def test_refuses_same_size_tamper_of_hashed_student_packet(tmp_path: Path) -> None:
    _make_seed(tmp_path, seed=20260815)
    evaluation_path = next(tmp_path.rglob("epoch_0005_n60.json"))
    evaluation = json.loads(evaluation_path.read_text())
    retained_path = Path(evaluation["packet_archive"]["payloads"]["student_packet"]["path"])
    retained_path.write_bytes(b"x" * retained_path.stat().st_size)
    with pytest.raises(ADJ.AdjudicationError, match="student_packet retained SHA-256 mismatch"):
        ADJ.adjudicate(tmp_path)
