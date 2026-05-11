"""Tests for PR106 latent-sidecar scorer-table producer scaffolding."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tac.repo_io import read_json

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "experiments/build_pr106_latent_score_table.py"
PR106_ARCHIVE = REPO_ROOT / (
    "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_pr106_latent_score_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    spec.loader.exec_module(module)
    return module


def test_apply_latent_candidates_torch_expands_and_applies_deltas():
    mod = _load_module()
    latents = torch.zeros((2, 4), dtype=torch.float32)
    candidates = torch.tensor(
        [
            [mod.NO_OP_DIM, 0],
            [1, -2],
            [3, 1],
        ],
        dtype=torch.int16,
    )

    out = mod.apply_latent_candidates_torch(latents, candidates)

    assert tuple(out.shape) == (6, 4)
    assert torch.equal(out[0], torch.zeros(4))
    assert out[1, 1].item() == pytest.approx(-0.02)
    assert out[2, 3].item() == pytest.approx(0.01)
    assert torch.equal(out[3], torch.zeros(4))
    assert out[4, 1].item() == pytest.approx(-0.02)
    assert out[5, 3].item() == pytest.approx(0.01)


def _checkpoint_args(tmp_path: Path) -> SimpleNamespace:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"fake-pr106-archive")
    return SimpleNamespace(
        pr106_archive=archive,
        out_dir=tmp_path,
        delta_radius=1,
        latent_dim=2,
        n_pairs=3,
        max_pairs=None,
    )


def test_latent_score_table_checkpoint_validates_contract(tmp_path):
    mod = _load_module()
    args = _checkpoint_args(tmp_path)
    candidates = np.array([[mod.NO_OP_DIM, 0], [0, -1], [0, 1]], dtype=np.int16)
    candidates_path = tmp_path / "candidate_grid.npy"
    np.save(candidates_path, candidates, allow_pickle=False)
    contract = mod._score_table_contract(
        args,
        candidates_np=candidates,
        candidates_path=candidates_path,
        n_pairs=3,
    )
    table = np.full((3, 3), np.nan, dtype=np.float32)
    table[:2] = 1.0

    mod._write_score_table_checkpoint(
        args,
        table=table,
        contract=contract,
        claim_row={"lane_id": "lane_pr106_latent_score_table"},
        started_at=0.0,
        terminal=False,
    )
    loaded = mod._load_score_table_checkpoint(args, contract=contract)

    assert loaded is not None
    resumed_table, complete_pairs = loaded
    assert complete_pairs == 2
    assert np.isfinite(resumed_table[:2]).all()
    assert np.isnan(resumed_table[2:]).all()

    changed = {**contract, "delta_radius": 2}
    with pytest.raises(ValueError, match="delta_radius"):
        mod._load_score_table_checkpoint(args, contract=changed)


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
            "--delta-radius",
            "1",
            "--latent-dim",
            "3",
            "--n-pairs",
            "5",
            "--max-pairs",
            "2",
            "--dry-run-plan",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    grid = np.load(out_dir / "candidate_grid.npy", allow_pickle=False)
    manifest = read_json(out_dir / "score_table_manifest.json")
    assert grid.shape == (1 + 3 * 2, 2)
    assert grid[0].tolist() == [255, 0]
    assert manifest["score_claim"] is False
    assert manifest["dry_run_plan"] is True
    assert manifest["ready_for_builder"] is False
    assert manifest["n_pairs"] == 2
    assert manifest["max_pairs"] == 2
    assert manifest["expected_score_table_shape"] == [2, 7]
    assert "requires_real_cuda_score_table" in manifest["dispatch_blockers"]
