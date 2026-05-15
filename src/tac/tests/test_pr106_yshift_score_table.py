# SPDX-License-Identifier: MIT
"""Tests for PR106 yshift scorer-table producer scaffolding.

These tests intentionally avoid loading CUDA scorers. They cover the deterministic
plan/manifest path, lane-claim parser, and torch yshift arithmetic used by the
real CUDA producer.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tac.repo_io import read_json
from tac.sidechannel_score_table import is_cuda_oom, mirror_provider_local_active_claim

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/build_pr106_yshift_score_table.py"
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
REMOTE_SCRIPT = REPO_ROOT / "scripts/remote_lane_pr106_yshift_sidechannel.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_pr106_yshift_score_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    spec.loader.exec_module(module)
    return module


def test_score_without_rate_matches_contest_formula_without_rate():
    mod = _load_module()
    pose = torch.tensor([0.004, 0.0])
    seg = torch.tensor([0.001, 0.002])

    out = mod.score_without_rate(pose, seg)

    expected = 100.0 * seg + torch.sqrt(10.0 * pose)
    assert torch.allclose(out, expected)


def test_apply_yshift_candidates_torch_matches_integer_shift_semantics():
    mod = _load_module()
    frame = torch.zeros((2, 4, 5, 3), dtype=torch.uint8)
    frame[:, 1, 1] = torch.tensor([100, 110, 120], dtype=torch.uint8)
    candidates = torch.tensor(
        [
            [10, 0, 0],
            [0, 1, 2],
        ],
        dtype=torch.int8,
    )

    out = mod.apply_yshift_candidates_torch(frame, candidates, step=1.0)

    assert tuple(out[0, 1, 1].tolist()) == (110, 120, 130)
    assert tuple(out[1, 2, 3].tolist()) == (100, 110, 120)
    assert tuple(out[1, 0, 0].tolist()) == (0, 0, 0)


def test_batched_candidate_table_matches_pairwise_reference():
    mod = _load_module()

    class FakeDistortionNet:
        def compute_distortion(self, gt_batch: torch.Tensor, cand_batch: torch.Tensor):
            diff = (gt_batch.float() - cand_batch.float()).abs()
            pose = diff.mean(dim=(1, 2, 3, 4))
            seg = diff[:, 1].mean(dim=(1, 2, 3))
            return pose, seg

    gt_pairs = torch.zeros((2, 2, 4, 5, 3), dtype=torch.uint8)
    comp_pairs = torch.zeros_like(gt_pairs)
    comp_pairs[0, 0, 1, 1] = torch.tensor([40, 50, 60], dtype=torch.uint8)
    comp_pairs[0, 1, 2, 2] = torch.tensor([80, 90, 100], dtype=torch.uint8)
    comp_pairs[1, 0, 1, 3] = torch.tensor([120, 90, 30], dtype=torch.uint8)
    comp_pairs[1, 1, 3, 4] = torch.tensor([15, 25, 35], dtype=torch.uint8)
    candidates = torch.tensor(
        [
            [0, 0, 0],
            [5, 0, 0],
            [0, 1, -1],
            [-3, -1, 1],
        ],
        dtype=torch.int8,
    )

    batched = mod.score_pair_batch_candidate_table(
        FakeDistortionNet(),
        gt_pairs=gt_pairs,
        comp_pairs=comp_pairs,
        candidates=candidates,
        candidate_batch_size=2,
        score_step=1.0,
    )
    pairwise_rows = []
    for pair_index in range(gt_pairs.shape[0]):
        row0, row1 = mod.score_frame_candidate_table(
            FakeDistortionNet(),
            gt_pair=gt_pairs[pair_index:pair_index + 1],
            comp_pair=comp_pairs[pair_index:pair_index + 1],
            candidates=candidates,
            candidate_batch_size=2,
            score_step=1.0,
        )
        pairwise_rows.extend([row0, row1])
    pairwise = np.stack(pairwise_rows, axis=0)

    assert batched.shape == pairwise.shape == (4, 4)
    assert np.allclose(batched, pairwise, rtol=0.0, atol=1e-6)


def test_adaptive_candidate_table_retries_cuda_oom_without_changing_scores():
    mod = _load_module()

    class FakeDistortionNet:
        def __init__(self, *, max_rows: int | None = None):
            self.max_rows = max_rows

        def compute_distortion(self, gt_batch: torch.Tensor, cand_batch: torch.Tensor):
            if self.max_rows is not None and int(cand_batch.shape[0]) > self.max_rows:
                raise RuntimeError("CUDA out of memory. Tried to allocate test bytes")
            diff = (gt_batch.float() - cand_batch.float()).abs()
            pose = diff.mean(dim=(1, 2, 3, 4))
            seg = diff[:, 1].mean(dim=(1, 2, 3))
            return pose, seg

    gt_pairs = torch.zeros((2, 2, 4, 5, 3), dtype=torch.uint8)
    comp_pairs = torch.zeros_like(gt_pairs)
    comp_pairs[0, 0, 1, 1] = torch.tensor([40, 50, 60], dtype=torch.uint8)
    comp_pairs[0, 1, 2, 2] = torch.tensor([80, 90, 100], dtype=torch.uint8)
    comp_pairs[1, 0, 1, 3] = torch.tensor([120, 90, 30], dtype=torch.uint8)
    comp_pairs[1, 1, 3, 4] = torch.tensor([15, 25, 35], dtype=torch.uint8)
    candidates = torch.tensor(
        [
            [0, 0, 0],
            [5, 0, 0],
            [0, 1, -1],
            [-3, -1, 1],
        ],
        dtype=torch.int8,
    )

    reference = mod.score_pair_batch_candidate_table(
        FakeDistortionNet(),
        gt_pairs=gt_pairs,
        comp_pairs=comp_pairs,
        candidates=candidates,
        candidate_batch_size=4,
        score_step=1.0,
    )
    adaptive, telemetry = mod.score_pair_batch_candidate_table_adaptive(
        FakeDistortionNet(max_rows=2),
        gt_pairs=gt_pairs,
        comp_pairs=comp_pairs,
        candidates=candidates,
        pair_chunk_size=2,
        candidate_batch_size=4,
        score_step=1.0,
        device=torch.device("cpu"),
    )

    assert np.allclose(adaptive, reference, rtol=0.0, atol=1e-6)
    assert telemetry["oom_retry_count"] > 0
    assert telemetry["min_candidate_batch_size_used"] == 1


def test_adaptive_candidate_table_reduces_pair_chunk_after_candidate_floor():
    mod = _load_module()

    class FakeDistortionNet:
        def compute_distortion(self, gt_batch: torch.Tensor, cand_batch: torch.Tensor):
            if int(cand_batch.shape[0]) > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate test bytes")
            diff = (gt_batch.float() - cand_batch.float()).abs()
            pose = diff.mean(dim=(1, 2, 3, 4))
            seg = diff[:, 1].mean(dim=(1, 2, 3))
            return pose, seg

    gt_pairs = torch.zeros((2, 2, 2, 2, 3), dtype=torch.uint8)
    comp_pairs = torch.zeros_like(gt_pairs)
    candidates = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int8)

    rows, telemetry = mod.score_pair_batch_candidate_table_adaptive(
        FakeDistortionNet(),
        gt_pairs=gt_pairs,
        comp_pairs=comp_pairs,
        candidates=candidates,
        pair_chunk_size=2,
        candidate_batch_size=2,
        score_step=1.0,
        device=torch.device("cpu"),
    )

    assert rows.shape == (4, 2)
    assert np.isfinite(rows).all()
    assert telemetry["min_candidate_batch_size_used"] == 1
    assert telemetry["min_pair_chunk_size_used"] == 1


def test_is_cuda_oom_requires_cuda_oom_signal():
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 4 GiB"))
    assert not is_cuda_oom(RuntimeError("cpu out of memory in unrelated parser"))
    assert not is_cuda_oom(ValueError("candidate shape mismatch"))


def test_verify_active_lane_claim_accepts_newest_nonterminal(tmp_path):
    mod = _load_module()
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-06T10:01:00Z | codex | lane_a | modal | job1 |  | running | current |",
                "| 2026-05-06T10:00:00Z | codex | lane_a | modal | job1 |  | completed_ok | older |",
            ]
        ),
        encoding="utf-8",
    )

    row = mod.verify_active_lane_claim(claims, lane_id="lane_a", instance_job_id="job1")

    assert row["status"] == "running"
    assert row["notes"] == "current"


def test_verify_active_lane_claim_rejects_terminal_newest(tmp_path):
    mod = _load_module()
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-06T10:02:00Z | codex | lane_a | modal | job1 |  | failed_cuda | newest |",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="terminal"):
        mod.verify_active_lane_claim(claims, lane_id="lane_a", instance_job_id="job1")


def test_provider_local_claim_mirror_prepends_strict_active_row(tmp_path):
    mod = _load_module()
    claims = tmp_path / "claims.md"
    claims.write_text(
        "\n".join(
            [
                "| timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |",
                "|---|---|---|---|---|---|---|---|",
                "| 2026-05-06T10:02:00Z | codex | lane_a | kaggle | job1 |  | failed_cuda | newest stale in bundled ledger |",
            ]
        ),
        encoding="utf-8",
    )

    row = mirror_provider_local_active_claim(
        claims,
        lane_id="lane_a",
        instance_job_id="job1",
        platform="kaggle",
        agent="provider-test",
        notes="mirror for isolated provider workspace",
    )

    assert row["status"] == "active_dispatching"
    assert row["agent"] == "provider-test"
    assert row["notes"] == "mirror for isolated provider workspace"
    assert mod.verify_active_lane_claim(claims, lane_id="lane_a", instance_job_id="job1") == row


def _checkpoint_args(tmp_path: Path) -> SimpleNamespace:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fake-pr106-archive")
    return SimpleNamespace(
        pr106_archive=archive,
        out_dir=tmp_path,
        candidate_radius=1,
        n_pairs=3,
        max_frames=None,
        score_step=1.0,
    )


def test_score_table_checkpoint_rounds_back_half_scored_pair(tmp_path):
    mod = _load_module()
    args = _checkpoint_args(tmp_path)
    candidates = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int8)
    candidates_path = tmp_path / "candidate_grid.npy"
    np.save(candidates_path, candidates, allow_pickle=False)
    contract = mod._score_table_contract(
        args,
        candidates_np=candidates,
        candidates_path=candidates_path,
        n_frames=6,
    )
    table = np.full((6, 2), np.nan, dtype=np.float32)
    table[:3] = 1.0

    mod._write_score_table_checkpoint(
        args,
        table=table,
        contract=contract,
        claim_row={"lane_id": "lane_pr106_yshift_score_table"},
        started_at=0.0,
        terminal=False,
    )
    loaded = mod._load_score_table_checkpoint(args, contract=contract)

    assert loaded is not None
    resumed_table, complete_frames = loaded
    assert complete_frames == 2
    assert np.isfinite(resumed_table[:2]).all()
    assert np.isnan(resumed_table[2:]).all()


def test_score_table_checkpoint_rejects_contract_drift(tmp_path):
    mod = _load_module()
    args = _checkpoint_args(tmp_path)
    candidates = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int8)
    candidates_path = tmp_path / "candidate_grid.npy"
    np.save(candidates_path, candidates, allow_pickle=False)
    contract = mod._score_table_contract(
        args,
        candidates_np=candidates,
        candidates_path=candidates_path,
        n_frames=6,
    )
    table = np.zeros((6, 2), dtype=np.float32)
    mod._write_score_table_checkpoint(
        args,
        table=table,
        contract=contract,
        claim_row={"lane_id": "lane_pr106_yshift_score_table"},
        started_at=0.0,
        terminal=False,
    )

    changed = {**contract, "score_step": 0.5}
    with pytest.raises(ValueError, match="score_step"):
        mod._load_score_table_checkpoint(args, contract=changed)


def test_completed_score_table_reuse_validates_contract(tmp_path):
    mod = _load_module()
    args = _checkpoint_args(tmp_path)
    candidates = np.array([[0, 0, 0], [0, 1, 0]], dtype=np.int8)
    candidates_path = tmp_path / "candidate_grid.npy"
    np.save(candidates_path, candidates, allow_pickle=False)
    contract = mod._score_table_contract(
        args,
        candidates_np=candidates,
        candidates_path=candidates_path,
        n_frames=6,
    )
    np.save(tmp_path / "score_table.npy", np.zeros((6, 2), dtype=np.float32), allow_pickle=False)
    (tmp_path / "score_table_manifest.json").write_text(
        json.dumps({"ready_for_builder": True, "score_claim": False, **contract}),
        encoding="utf-8",
    )

    assert mod._reuse_completed_score_table_if_valid(args, contract=contract) is True

    drifted = {**contract, "candidate_radius": 2}
    with pytest.raises(ValueError, match="candidate_radius"):
        mod._reuse_completed_score_table_if_valid(args, contract=drifted)


def test_remote_score_table_path_opts_into_resume_checkpoint():
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'PR106_YSHIFT_SCORE_TABLE_RESUME="${PR106_YSHIFT_SCORE_TABLE_RESUME:-1}"' in text
    assert "Stage 1a RESUME: validating completed score table" in text
    assert "SCORE_TABLE_ARGS+=(--resume-checkpoint)" in text


def test_dry_run_plan_writes_candidate_grid_and_manifest(tmp_path):
    if not PR106_ARCHIVE.is_file():
        pytest.skip(f"missing PR106 archive at {PR106_ARCHIVE}")
    out_dir = tmp_path / "plan"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pr106-archive",
            str(PR106_ARCHIVE),
            "--out-dir",
            str(out_dir),
            "--candidate-radius",
            "1",
            "--n-pairs",
            "2",
            "--max-frames",
            "3",
            "--dry-run-plan",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    grid = np.load(out_dir / "candidate_grid.npy", allow_pickle=False)
    manifest = read_json(out_dir / "score_table_manifest.json")
    assert grid.shape == (27, 3)
    assert manifest["score_claim"] is False
    assert manifest["dry_run_plan"] is True
    assert manifest["ready_for_builder"] is False
    assert manifest["n_frames"] == 3
    assert manifest["max_frames"] == 3
    assert manifest["expected_score_table_shape"] == [3, 27]
    assert "requires_real_cuda_score_table" in manifest["dispatch_blockers"]
