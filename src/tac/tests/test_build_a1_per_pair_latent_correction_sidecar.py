"""Tests for the A1 per-pair latent sidecar resampling helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "build_a1_per_pair_latent_correction_sidecar.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("a1_sidecar_tool", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_packed_sidecar_encodes_high_dims_that_uint8_layout_cannot() -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    dims[0] = 27
    delta_idx[0] = 15

    packed = tool.encode_sidecar_huff_enum(dims, delta_idx)

    assert len(packed) == 661
    assert packed != b"\x00" * 661


def test_uint8_sidecar_rejects_high_dim_choices() -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    dims[0] = 27
    delta_idx[0] = 15

    try:
        tool.encode_sidecar_n_pairs(dims, delta_idx)
    except ValueError as exc:
        assert "cannot encode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected high-dim uint8 sidecar choice to fail")


def test_manifest_path_allows_external_outputs(tmp_path: Path) -> None:
    tool = load_tool()

    assert tool.manifest_path(REPO_ROOT / "tools" / "x.py") == "tools/x.py"
    assert tool.manifest_path(tmp_path / "x.py") == str(tmp_path / "x.py")


def test_ground_truth_loader_uses_upstream_yuv420_helper() -> None:
    tool_text = TOOL_PATH.read_text()

    assert "load_upstream_yuv420_to_rgb" in tool_text
    assert "yuv420_to_rgb(f)" in tool_text
    assert "to_ndarray(format=\"rgb24\")" not in tool_text


def _write_runtime_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sub_dir = tmp_path / "submission_dir"
    src_dir = sub_dir / "src"
    src_dir.mkdir(parents=True)
    archive = sub_dir / "archive.zip"
    archive.write_bytes(b"archive bytes")
    (sub_dir / "inflate.py").write_text("print('inflate')\n")
    inflate_sh = sub_dir / "inflate.sh"
    inflate_sh.write_text("#!/bin/sh\npython inflate.py\n")
    inflate_sh.chmod(0o755)
    (src_dir / "codec.py").write_text("LATENT_DIM = 28\n")
    (src_dir / "model.py").write_text("class HNeRVDecoder: pass\n")
    return sub_dir, archive


def _changed_sidecar_no_op() -> dict[str, object]:
    return {
        "old_inner_sidecar_sha256": "1" * 64,
        "new_inner_sidecar_sha256": "2" * 64,
        "sidecar_changed": True,
    }


def _runtime_smoke_evidence(tool, archive: Path, custody: dict[str, object]) -> dict[str, object]:
    return {
        "command": ["./inflate.sh", "archive_dir", "out", "file_list.txt"],
        "exit_code": 0,
        "archive_sha256": tool.sha256_of(archive),
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "output_digest_sha256": "3" * 64,
    }


def test_infer_sidecar_choices_preserves_old_unsearched_pairs() -> None:
    tool = load_tool()
    base = torch.zeros((tool.N_PAIRS, tool.LATENT_DIM), dtype=torch.float32)
    old = base.clone()
    old[0, 3] += 0.04
    old[10, 27] -= 0.10

    dims, delta_idx = tool.infer_sidecar_choices_from_latents(
        {
            "latents_pre_sidecar": base,
            "latents_with_sidecar_old": old,
        }
    )

    assert dims[0] == 3
    assert tool.SIDECAR_DELTAS_X100[delta_idx[0]] == 4
    assert dims[10] == 27
    assert tool.SIDECAR_DELTAS_X100[delta_idx[10]] == -10
    assert dims[1] == 255
    assert delta_idx[1] == -1


def test_infer_sidecar_rejects_multi_dim_pair_delta() -> None:
    tool = load_tool()
    base = torch.zeros((tool.N_PAIRS, tool.LATENT_DIM), dtype=torch.float32)
    old = base.clone()
    old[0, 0] += 0.02
    old[0, 1] += 0.03

    with pytest.raises(ValueError, match="<=1 changed latent dim"):
        tool.infer_sidecar_choices_from_latents(
            {
                "latents_pre_sidecar": base,
                "latents_with_sidecar_old": old,
            }
        )


def test_manifest_readiness_fails_closed_without_materialized_archive(tmp_path: Path) -> None:
    tool = load_tool()
    manifest = {
        "dispatch_blockers": [],
        "new_archive_sha256": "missing",
        "new_archive_bytes": 123,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": {"sidecar_changed": True},
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=tmp_path / "submission_dir" / "archive.zip",
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert any(
        item.startswith("materialized_archive_missing:")
        for item in out["dispatch_blockers"]
    )


def test_manifest_readiness_requires_sidecar_lane_and_archive_path(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    manifest = {
        "lane_id": "lane_a1_inflate_time_bias_correction_sweep",
        "dispatch_blockers": [],
        "new_archive_sha256": tool.sha256_of(archive),
        "new_archive_bytes": archive.stat().st_size,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": {"sidecar_changed": True},
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "sidecar_lane_id_missing_or_mismatch" in out["dispatch_blockers"]
    assert "archive_path_missing" in out["dispatch_blockers"]


def test_manifest_readiness_accepts_complete_local_custody(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "candidate_archive_path": tool.manifest_path(archive),
        "candidate_archive_sha256": archive_sha,
        "candidate_archive_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": _changed_sidecar_no_op(),
        "local_runtime_custody": custody,
        "runtime_manifest": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is True
    assert out["dispatch_blockers"] == []


def test_collect_local_runtime_custody_covers_extra_runtime_files(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    (sub_dir / "src" / "helper.py").write_text("HELPER = 1\n")
    (sub_dir / "weights.pt").write_bytes(b"runtime sidecar")

    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)

    rels = {row["relative_path"] for row in custody["files"]}
    assert "src/helper.py" in rels
    assert "weights.pt" in rels


def test_manifest_readiness_rejects_extra_runtime_file_drift(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    (sub_dir / "src" / "helper.py").write_text("HELPER = 1\n")
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "smoke_only": False,
        "no_op_detector": _changed_sidecar_no_op(),
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    blockers = out["dispatch_blockers"]
    assert "local_runtime_custody_file_record_missing:src/helper.py" in blockers
    assert "local_runtime_custody_file_count_mismatch" in blockers
    assert "local_runtime_custody_runtime_tree_sha256_mismatch" in blockers


def test_manifest_readiness_rejects_no_op_hash_without_sidecar_delta(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "smoke_only": False,
        "no_op_detector": {
            "old_inner_sidecar_sha256": "a" * 64,
            "new_inner_sidecar_sha256": "a" * 64,
            "sidecar_changed": True,
        },
        "local_runtime_custody": custody,
        "runtime_manifest": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "sidecar_no_op_hashes_equal" in out["dispatch_blockers"]


def test_manifest_readiness_rejects_bare_runtime_smoke_boolean(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": _changed_sidecar_no_op(),
        "local_runtime_custody": custody,
        "runtime_manifest": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "runtime_smoke_evidence_missing" in out["dispatch_blockers"]


def test_manifest_readiness_rejects_runtime_custody_drift(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    (sub_dir / "inflate.py").write_text("print('mutated')\n")
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": {"sidecar_changed": True},
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    blockers = out["dispatch_blockers"]
    assert "local_runtime_custody_inflate.py_sha256_mismatch" in blockers
    assert "local_runtime_custody_runtime_tree_sha256_mismatch" in blockers


def test_manifest_readiness_rejects_archive_sha_and_byte_drift(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": "0" * 64,
        "archive_size_bytes": archive.stat().st_size + 1,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": "0" * 64,
        "new_archive_bytes": archive.stat().st_size + 1,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "smoke_only": False,
        "no_op_detector": {"sidecar_changed": True},
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    blockers = out["dispatch_blockers"]
    assert "new_archive_sha256_mismatch" in blockers
    assert "new_archive_bytes_mismatch" in blockers
    assert "materialized_archive_sha256_mismatch" in blockers
    assert "materialized_archive_size_mismatch" in blockers


def test_manifest_readiness_fails_closed_for_smoke_or_no_runtime_smoke(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": False,
        "smoke_only": True,
        "no_op_detector": {"sidecar_changed": True},
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "smoke_only_not_exact_eval_ready" in out["dispatch_blockers"]
    assert "runtime_smoke_not_checked" in out["dispatch_blockers"]


def test_ground_truth_pairs_needed_decodes_only_required_prefix() -> None:
    tool = load_tool()

    assert tool.ground_truth_pairs_needed([0, 1, 9], 600) == 10
    assert tool.ground_truth_pairs_needed([], 600) == 0
    with pytest.raises(ValueError, match="outside n_pairs"):
        tool.ground_truth_pairs_needed([600], 600)


def test_batched_pair_proxy_search_preserves_best_candidate_semantics() -> None:
    tool = load_tool()

    class Decoder(torch.nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            value = latents[:, 1].reshape(-1, 1, 1, 1)
            return value.repeat(1, 6, 1, 1)

    decoder = Decoder()
    base_lat = torch.zeros((1, 2), dtype=torch.float32)
    gt_eval = np.ones((2, 1, 1, 3), dtype=np.float32)
    deltas = np.array([-1.0, 1.0], dtype=np.float32)

    scalar = tool._best_pair_proxy_mse_candidate(
        decoder=decoder,
        base_lat=base_lat,
        gt_eval=gt_eval,
        eval_h=1,
        eval_w=1,
        deltas=deltas,
        candidate_batch_size=1,
    )
    batched = tool._best_pair_proxy_mse_candidate(
        decoder=decoder,
        base_lat=base_lat,
        gt_eval=gt_eval,
        eval_h=1,
        eval_w=1,
        deltas=deltas,
        candidate_batch_size=4,
    )

    assert scalar == batched
    base_mse, best_mse, best_dim, best_didx = batched
    assert base_mse == pytest.approx(1.0)
    assert best_mse == pytest.approx(0.0)
    assert best_dim == 1
    assert best_didx == 1
