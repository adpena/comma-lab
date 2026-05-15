# SPDX-License-Identifier: MIT
"""Tests for PR106 latent-sidecar scorer-table producer scaffolding."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
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
PR106_FORMAT0C_ARCHIVE = REPO_ROOT / (
    "experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/"
    "candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip"
)
REMOTE_SCRIPT = REPO_ROOT / "scripts/remote_lane_pr106_latent_sidecar.sh"


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


def test_adaptive_latent_table_retries_cuda_oom_without_changing_scores(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "EVAL_SIZE", (2, 2))
    monkeypatch.setattr(mod, "CAMERA_H", 2)
    monkeypatch.setattr(mod, "CAMERA_W", 2)

    class FakeDecoder:
        def __call__(self, latents: torch.Tensor) -> torch.Tensor:
            values = latents.sum(dim=1).view(-1, 1, 1, 1, 1)
            return values.expand(-1, 2, 3, 2, 2)

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

    gt_pairs = torch.zeros((2, 2, 2, 2, 3), dtype=torch.uint8)
    latents = torch.tensor([[0.0, 0.5], [1.0, -0.5]], dtype=torch.float32)
    candidates = torch.tensor(
        [
            [mod.NO_OP_DIM, 0],
            [0, 1],
            [1, -1],
        ],
        dtype=torch.int16,
    )

    reference = mod.score_pair_batch_candidate_table(
        FakeDistortionNet(),
        FakeDecoder(),
        gt_pairs=gt_pairs,
        latents_batch=latents,
        candidates=candidates,
        candidate_batch_size=3,
    )
    adaptive, telemetry = mod.score_pair_batch_candidate_table_adaptive(
        FakeDistortionNet(max_rows=2),
        FakeDecoder(),
        gt_pairs=gt_pairs,
        latents_batch=latents,
        candidates=candidates,
        pair_chunk_size=2,
        candidate_batch_size=3,
        device=torch.device("cpu"),
    )

    assert np.allclose(adaptive, reference, rtol=0.0, atol=1e-6)
    assert telemetry["oom_retry_count"] > 0
    assert telemetry["min_candidate_batch_size_used"] == 1


def test_adaptive_latent_table_reduces_pair_chunk_after_candidate_floor(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "EVAL_SIZE", (2, 2))
    monkeypatch.setattr(mod, "CAMERA_H", 2)
    monkeypatch.setattr(mod, "CAMERA_W", 2)

    class FakeDecoder:
        def __call__(self, latents: torch.Tensor) -> torch.Tensor:
            values = latents.sum(dim=1).view(-1, 1, 1, 1, 1)
            return values.expand(-1, 2, 3, 2, 2)

    class FakeDistortionNet:
        def compute_distortion(self, gt_batch: torch.Tensor, cand_batch: torch.Tensor):
            if int(cand_batch.shape[0]) > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate test bytes")
            diff = (gt_batch.float() - cand_batch.float()).abs()
            pose = diff.mean(dim=(1, 2, 3, 4))
            seg = diff[:, 1].mean(dim=(1, 2, 3))
            return pose, seg

    gt_pairs = torch.zeros((2, 2, 2, 2, 3), dtype=torch.uint8)
    latents = torch.zeros((2, 2), dtype=torch.float32)
    candidates = torch.tensor([[mod.NO_OP_DIM, 0], [0, 1]], dtype=torch.int16)

    rows, telemetry = mod.score_pair_batch_candidate_table_adaptive(
        FakeDistortionNet(),
        FakeDecoder(),
        gt_pairs=gt_pairs,
        latents_batch=latents,
        candidates=candidates,
        pair_chunk_size=2,
        candidate_batch_size=2,
        device=torch.device("cpu"),
    )

    assert rows.shape == (2, 2)
    assert np.isfinite(rows).all()
    assert telemetry["min_candidate_batch_size_used"] == 1
    assert telemetry["min_pair_chunk_size_used"] == 1


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


def test_latent_score_table_member_reader_autodetects_x_member(tmp_path):
    mod = _load_module()
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("x", b"\xfepr106-sidecar-packet")

    member = mod._read_source_archive_member(archive)

    assert member.name == "x"
    assert member.payload == b"\xfepr106-sidecar-packet"
    assert mod._source_member_contract(member)["source_archive_member_name"] == "x"


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
    assert manifest["source_archive_member_name"] == "0.bin"
    assert manifest["runtime_dir"].endswith("submissions/pr106_latent_sidecar_r2_pr101_grammar")
    assert "requires_real_cuda_score_table" in manifest["dispatch_blockers"]


def test_dry_run_plan_binds_format0c_x_member_and_runtime(tmp_path):
    if not PR106_FORMAT0C_ARCHIVE.is_file():
        pytest.skip(f"missing format0C archive at {PR106_FORMAT0C_ARCHIVE}")
    out_dir = tmp_path / "format0c-plan"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pr106-archive",
            str(PR106_FORMAT0C_ARCHIVE),
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

    manifest = read_json(out_dir / "score_table_manifest.json")
    assert manifest["source_archive_member_name"] == "x"
    assert manifest["source_payload_kind"] == "pr106_sidecar_packet"
    assert manifest["source_archive_member_sha256"]
    assert manifest["runtime_dir"].endswith("submissions/pr106_latent_sidecar_r2_pr101_grammar")
    assert manifest["score_claim"] is False


def test_load_pr106_decoder_applies_format0c_runtime_sidecar_cpu():
    if not PR106_FORMAT0C_ARCHIVE.is_file():
        pytest.skip(f"missing format0C archive at {PR106_FORMAT0C_ARCHIVE}")
    mod = _load_module()

    _, latents, meta, _, source_info = mod._load_pr106_decoder(
        PR106_FORMAT0C_ARCHIVE,
        torch.device("cpu"),
        runtime_dir=mod.DEFAULT_RUNTIME_DIR,
    )

    assert source_info["source_archive_member_name"] == "x"
    assert source_info["source_payload_kind"] == "pr106_sidecar_packet"
    assert source_info["sidecar_format_id"] == "0x0C"
    assert int(meta["n_pairs"]) == 600
    assert tuple(latents.shape) == (600, int(meta["latent_dim"]))


def test_remote_latent_lane_defaults_to_score_table_and_resume():
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'PR106_LATENT_MODE="${PR106_LATENT_MODE:-score_table}"' in text
    assert 'PR106_LATENT_SCORE_TABLE_RESUME="${PR106_LATENT_SCORE_TABLE_RESUME:-1}"' in text
    assert 'LANE_ID="lane_pr106_latent_sidecar"' in text
    assert 'PR106_LATENT_SCORE_TABLE_LANE_ID="${PR106_LATENT_SCORE_TABLE_LANE_ID:-$LANE_ID}"' in text
    assert "experiments/build_pr106_latent_score_table.py" in text
    assert 'PR106_RUNTIME_DIR="${PR106_RUNTIME_DIR:-submissions/pr106_latent_sidecar_r2_pr101_grammar}"' in text
    assert "--runtime-dir \"$PR106_RUNTIME_DIR\"" in text
    assert "tools/prove_pr106_sidecar_runtime_consumption.py" in text
    assert "EXPECTED_ARCHIVE_SHA=" in text
    assert "EXPECTED_RUNTIME_TREE_SHA=" in text
    assert "--expected-archive-sha256 \"$EXPECTED_ARCHIVE_SHA\"" in text
    assert (
        "--expected-runtime-source-tree-sha256 \"$EXPECTED_RUNTIME_TREE_SHA\""
        in text
    )
    assert 'INFLATE_SH="$WORKSPACE/$PR106_RUNTIME_DIR/inflate.sh"' in text
    assert "Stage 1a RESUME: validating completed latent score table" in text
    assert "SCORE_TABLE_ARGS+=(--resume-checkpoint)" in text
    assert "--search-mode \"$PR106_LATENT_MODE\"" in text
    assert "--score-table-manifest \"$PR106_LATENT_SCORE_TABLE_MANIFEST\"" in text


def test_remote_latent_lane_defines_log_before_nvdec_probe():
    text = REMOTE_SCRIPT.read_text(encoding="utf-8")

    log_pos = text.index("log() {")
    probe_pos = text.index("probe MUST come before")
    assert log_pos < probe_pos
    assert "NVDEC/DALI probe failed" in text
