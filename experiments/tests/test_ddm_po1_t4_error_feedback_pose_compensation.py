from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_js1b_cuda_argmax_field_materializer_worker as js1b
from experiments import ddm_po1_modal_t4_pose_feedback as dispatcher
from experiments import ddm_po1_t4_error_feedback_pose_compensation as solver
from experiments import ddm_po1_t4_pose_feedback_worker as worker


def test_pose_feedback_metrics_uses_first6_and_closes_repeat_floor() -> None:
    gt = np.zeros((worker.N_PAIRS, 6), dtype=np.float32)
    first = np.ones_like(gt)
    stable = first.copy()
    stable_result = worker.pose_feedback_metrics(gt, first, stable)
    assert stable_result["d_pose_decoded_first"] == pytest.approx(1.0)
    assert stable_result["repeat_noise_mse"] == pytest.approx(0.0)
    assert stable_result["f1_instrument_floor_closed"] is False

    noisy = np.full_like(gt, 2.0)
    noisy_result = worker.pose_feedback_metrics(gt, first, noisy)
    assert noisy_result["noise_comparable_pairs"] == worker.N_PAIRS
    assert noisy_result["f1_instrument_floor_closed"] is True


def test_score_pose_vectors_retains_inputs_outputs_and_scored_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setattr(js1b, "N_PAIRS", 2)
    batch = torch.arange(12, dtype=torch.uint8).reshape(2, 2, 1, 1, 3)
    monkeypatch.setattr(js1b, "_dataset", lambda *_args, **_kwargs: [("0", 0, batch)])

    class FakePoseNet:
        @staticmethod
        def preprocess_input(value):
            return value

        @staticmethod
        def __call__(value):
            total = value.reshape(value.shape[0], -1).sum(dim=1, keepdim=True)
            return {"pose": total.repeat(1, 12)}

    result = js1b.score_pose_vectors(
        source="gt",
        raw_root=None,
        raw_record=None,
        scorer=FakePoseNet(),
        device=torch.device("cpu"),
        run_root=tmp_path,
    )
    vectors = np.load(result["first6_vectors"]["path"], allow_pickle=False)
    assert vectors.shape == (2, 6)
    assert np.all(vectors[0] == 15.0)
    assert np.all(vectors[1] == 51.0)
    batch_result = json.loads(
        (tmp_path / "retained/pose/gt/batches/batch_0000/BATCH_RESULT.json").read_text()
    )
    assert Path(batch_result["pose_input"]["path"]).is_file()
    assert Path(batch_result["pose_output_full"]["path"]).is_file()


def test_damped_step_moves_toward_residual_and_clamps_int12() -> None:
    jacobian = np.zeros((6, 12), dtype=np.float64)
    jacobian[:, :6] = np.eye(6)
    residual = np.arange(1.0, 7.0)
    update, diagnostics = solver.solve_damped_least_squares(
        jacobian,
        residual,
        damping=0.0,
        max_code_step=4.0,
    )
    assert diagnostics["rank"] == 6
    assert np.allclose(update[:4], [1.0, 2.0, 3.0, 4.0])
    assert np.all(update[4:6] == 4.0)
    current = np.array([2047, -2048, *([0] * 10)], dtype=np.int16)
    quantized = solver.quantize_int12_update(
        current,
        np.array([10.0, -10.0, *([0.0] * 10)]),
    )
    assert quantized[0] == 2047
    assert quantized[1] == -2048


def test_cp135_identity_rebuild_is_byte_identical(tmp_path: Path) -> None:
    archive = solver.DEFAULT_ARCHIVE
    runtime = solver.DEFAULT_RUNTIME
    if not archive.is_file() or not runtime.is_dir():
        pytest.skip("custodied CP135 runtime is unavailable")
    parts, state = solver.load_carrier(archive, runtime)
    candidate = solver.build_candidate(
        base_archive=archive,
        source_runtime=runtime,
        output_runtime=tmp_path / "candidate_runtime",
        parts=parts,
        state=state,
        codes=state.codes.copy(),
    )
    assert candidate["cap1_metadata_packed"] is True
    assert candidate["archive"]["bytes"] == solver.CP135_BYTES
    assert candidate["archive"]["sha256"] == solver.CP135_SHA256


def test_round2_adjudication_requires_exact_seg_field_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(solver, "N", 2)
    monkeypatch.setattr(solver, "SEG_H", 2)
    monkeypatch.setattr(solver, "SEG_W", 2)
    round1 = tmp_path / "round1"
    round2 = tmp_path / "round2"
    for root, pose, size in ((round1, 0.01, 100), (round2, 0.005, 101)):
        (root / "retained/fields").mkdir(parents=True)
        np.save(
            root / "retained/fields/candidate_argmax_n600.npy",
            np.zeros((solver.N, solver.SEG_H, solver.SEG_W), dtype=np.uint8),
            allow_pickle=False,
        )
        field_record = solver.file_record(
            root / "retained/fields/candidate_argmax_n600.npy"
        )
        (root / "FINAL_RESULT.json").write_text(
            json.dumps(
                {
                    "pose_feedback": {"d_pose_decoded_first": pose},
                    "seg_feedback": {"d_seg": 0.001},
                    "seg_scorers": {"candidate": {"argmax": field_record}},
                    "candidate_archive": {"bytes": size, "sha256": "0" * 64},
                }
            )
        )
    solve_result = tmp_path / "SOLVE_RESULT.json"
    solve_result.write_text(json.dumps({"predicted_candidate_d_pose_t4_origin": 0.004}))
    output = tmp_path / "ADJUDICATION.json"
    args = argparse.Namespace(
        round1=round1,
        round2=round2,
        solve_result=solve_result,
        output=output,
    )
    assert solver.adjudicate(args) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "ADMITTED_COMPONENT_IMPROVEMENT"
    assert result["seg_field_changed_pixels"] == 0
    assert result["third_round_allowed"] is True

    changed = np.load(
        round2 / "retained/fields/candidate_argmax_n600.npy",
        allow_pickle=False,
    )
    changed[0, 0, 0] = 1
    np.save(
        round2 / "retained/fields/candidate_argmax_n600.npy",
        changed,
        allow_pickle=False,
    )
    round2_result = json.loads((round2 / "FINAL_RESULT.json").read_text())
    round2_result["seg_scorers"]["candidate"]["argmax"] = solver.file_record(
        round2 / "retained/fields/candidate_argmax_n600.npy"
    )
    (round2 / "FINAL_RESULT.json").write_text(json.dumps(round2_result))
    assert solver.adjudicate(args) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "CLOSED_F3_SEGNET_MOVED"
    assert result["disposition"] == "do not ship"


def test_dispatch_recover_is_idempotent_for_both_terminal_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "recover"
    output_dir.mkdir()
    call_id = "fc-double-recover"
    run_id = "ddm_po1_double_recover"
    spawn = {
        "call_id": call_id,
        "lane_id": "ddm_po1_double_recover_lane",
        "instance_job_id": f"modal:{run_id}",
        "claim_agent": "test:po1",
    }
    (output_dir / "modal_auth_eval_spawn.json").write_text(json.dumps(spawn))
    remote_result = {
        "schema": "ddm_po1_modal_return.v1",
        "passed": True,
        "returncode": 0,
        "run_id": run_id,
        "artifacts": {"PO1_FINAL_RESULT.json": b"retained"},
        "score_claim": False,
        "promotion_eligible": False,
    }

    class FakeCall:
        @staticmethod
        def get(*, timeout: float):
            assert timeout == 0.0
            return copy.deepcopy(remote_result)

    monkeypatch.setattr(
        dispatcher.modal.functions.FunctionCall,
        "from_id",
        lambda value: FakeCall() if value == call_id else None,
    )
    outcome_rows: list[dict] = []
    monkeypatch.setattr(dispatcher, "query_by_call_id", lambda _call_id: outcome_rows)

    def fake_update(**kwargs):
        outcome_rows.append(
            {
                "call_id": kwargs["call_id"],
                "status": kwargs["status"],
                "harvest_result": kwargs["harvest_result"],
            }
        )

    monkeypatch.setattr(dispatcher, "update_call_id_outcome", fake_update)
    fake_repo = tmp_path / "repo"
    claims_path = fake_repo / ".omx/state/active_lane_dispatch_claims.md"
    monkeypatch.setattr(dispatcher, "REPO", fake_repo)
    claim_rows: list[dict] = []

    def fake_terminal(*, repo_root, spec, status, notes):
        assert repo_root == fake_repo
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        row = (
            f"| now | {spec.agent} | {spec.lane_id} | modal | "
            f"{spec.instance_job_id} | now | {status} | {notes} |\n"
        )
        with claims_path.open("a", encoding="utf-8") as handle:
            handle.write(row)
        claim_rows.append({"status": status, "notes": notes})

    monkeypatch.setattr(dispatcher, "terminal_modal_auth_eval_claim", fake_terminal)

    assert dispatcher.recover(output_dir) == 0
    assert dispatcher.recover(output_dir) == 0
    assert len(outcome_rows) == 1
    assert len(claim_rows) == 1
