"""Tests for the A1 per-pair latent sidecar resampling helper."""

from __future__ import annotations

import importlib.util
import json
import sys
import struct
import types
import zipfile
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


def _write_runtime_fixture(tmp_path: Path, sidecar: bytes = b"sidecar") -> tuple[Path, Path]:
    sub_dir = tmp_path / "submission_dir"
    src_dir = sub_dir / "src"
    src_dir.mkdir(parents=True)
    archive = sub_dir / "archive.zip"
    inner = struct.pack("<I", 8) + b"abcd" + (b"\x00" * 15387) + sidecar
    zinfo = zipfile.ZipInfo(filename="x", date_time=(2024, 1, 1, 0, 0, 0))
    zinfo.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(zinfo, inner)
    (sub_dir / "inflate.py").write_text("print('inflate')\n")
    inflate_sh = sub_dir / "inflate.sh"
    inflate_sh.write_text("#!/bin/sh\npython inflate.py\n")
    inflate_sh.chmod(0o755)
    (src_dir / "codec.py").write_text("LATENT_DIM = 28\n")
    (src_dir / "model.py").write_text("class HNeRVDecoder: pass\n")
    return sub_dir, archive


def _changed_sidecar_no_op(new_sha: str | None = None) -> dict[str, object]:
    return {
        "old_inner_sidecar_sha256": "1" * 64,
        "new_inner_sidecar_sha256": new_sha or "2" * 64,
        "sidecar_changed": True,
    }


def _proof_sidecar_sha(proof: dict[str, object]) -> str:
    inner = proof["inner_sections"]
    assert isinstance(inner, dict)
    sidecar = inner["sidecar_blob"]
    assert isinstance(sidecar, dict)
    value = sidecar["sha256"]
    assert isinstance(value, str)
    return value


def _proof_sidecar_bytes(proof: dict[str, object]) -> int:
    inner = proof["inner_sections"]
    assert isinstance(inner, dict)
    sidecar = inner["sidecar_blob"]
    assert isinstance(sidecar, dict)
    value = sidecar["bytes"]
    assert isinstance(value, int)
    return value


def _runtime_smoke_evidence(tool, archive: Path, custody: dict[str, object]) -> dict[str, object]:
    return {
        "command": ["./inflate.sh", "archive_dir", "out", "file_list.txt"],
        "exit_code": 0,
        "archive_sha256": tool.sha256_of(archive),
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "output_digest_sha256": "3" * 64,
    }


def _write_choice_state_record(
    tool,
    tmp_path: Path,
    *,
    dims: np.ndarray | None = None,
    delta_idx: np.ndarray | None = None,
    searched_mask: np.ndarray | None = None,
    old_archive_sha256: str = "a" * 64,
    old_sidecar_sha256: str = "1" * 64,
    search_signal: str = "proxy_mse",
    search_device: str = "cpu",
    encode_format: str = "packed_661",
) -> dict[str, object]:
    dims = (
        np.full(tool.N_PAIRS, 255, dtype=np.int64)
        if dims is None
        else dims.astype(np.int64, copy=True)
    )
    delta_idx = (
        np.full(tool.N_PAIRS, -1, dtype=np.int64)
        if delta_idx is None
        else delta_idx.astype(np.int64, copy=True)
    )
    searched_mask = (
        np.ones(tool.N_PAIRS, dtype=bool)
        if searched_mask is None
        else searched_mask.astype(bool, copy=True)
    )
    state_path = tmp_path / "sidecar_choice_state.json"
    tool.write_sidecar_choice_state(
        state_path,
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        old_archive_sha256=old_archive_sha256,
        old_sidecar_sha256=old_sidecar_sha256,
        search_signal=search_signal,
        search_device=search_device,
        encode_format=encode_format,
        total_pairs=tool.N_PAIRS,
        latent_dim=tool.LATENT_DIM,
    )
    return tool.sidecar_choice_state_manifest_record(state_path)


def test_member_section_proof_records_a1_archive_sections(tmp_path: Path) -> None:
    tool = load_tool()
    _, archive = _write_runtime_fixture(tmp_path)

    proof = tool.collect_member_section_proof(archive)

    assert proof["schema_version"] == tool.MEMBER_SECTION_PROOF_SCHEMA
    assert proof["member_count"] == 1
    assert proof["single_member_name"] == "x"
    inner = proof["inner_sections"]
    assert inner["decoder_section_total"] == 8
    assert inner["decoder_blob"]["bytes"] == 4
    assert inner["latent_blob"]["bytes"] == tool.A1_LATENT_BLOB_LEN
    assert inner["sidecar_blob"]["bytes"] == len(b"sidecar")


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
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    sidecar = tool.encode_sidecar_huff_enum(dims, delta_idx)
    sub_dir, archive = _write_runtime_fixture(tmp_path, sidecar=sidecar)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    proof = tool.collect_member_section_proof(archive)
    choice_state = _write_choice_state_record(
        tool,
        tmp_path,
        dims=dims,
        delta_idx=delta_idx,
        old_archive_sha256="a" * 64,
        old_sidecar_sha256="1" * 64,
    )
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
        "new_sidecar_bytes": _proof_sidecar_bytes(proof),
        "old_archive_sha256": "a" * 64,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "search_signal": "proxy_mse",
        "search_device": "cpu",
        "encode_format": "packed_661",
        "smoke_only": False,
        "full_non_smoke_search": True,
        "no_op_detector": _changed_sidecar_no_op(_proof_sidecar_sha(proof)),
        "local_runtime_custody": custody,
        "runtime_manifest": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "member_section_proof": proof,
        "sidecar_choice_state": choice_state,
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is True
    assert out["dispatch_blockers"] == []


def test_manifest_readiness_keeps_smoke_only_blocked_even_with_runtime_smoke(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    proof = tool.collect_member_section_proof(archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "new_sidecar_bytes": _proof_sidecar_bytes(proof),
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "smoke_only": True,
        "full_non_smoke_search": False,
        "no_op_detector": _changed_sidecar_no_op(_proof_sidecar_sha(proof)),
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "member_section_proof": proof,
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "smoke_only_not_exact_eval_ready" in out["dispatch_blockers"]
    assert "non_full_sidecar_search_not_exact_eval_ready" in out["dispatch_blockers"]


def test_manifest_readiness_rejects_partial_non_smoke_search(tmp_path: Path) -> None:
    tool = load_tool()
    sub_dir, archive = _write_runtime_fixture(tmp_path)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    proof = tool.collect_member_section_proof(archive)
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "new_sidecar_bytes": _proof_sidecar_bytes(proof),
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "smoke_only": False,
        "full_non_smoke_search": False,
        "no_op_detector": _changed_sidecar_no_op(_proof_sidecar_sha(proof)),
        "local_runtime_custody": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "member_section_proof": proof,
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "non_full_sidecar_search_not_exact_eval_ready" in out["dispatch_blockers"]


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


def test_sidecar_choice_state_round_trips_and_refuses_context_drift(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    searched_mask = np.zeros(tool.N_PAIRS, dtype=bool)
    dims[3] = 7
    delta_idx[3] = 12
    searched_mask[3] = True
    state_path = tmp_path / "sidecar_choice_state.json"

    payload = tool.write_sidecar_choice_state(
        state_path,
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        old_archive_sha256="a" * 64,
        old_sidecar_sha256="b" * 64,
        search_signal="proxy_mse",
        search_device="cpu",
        encode_format="packed_661",
        total_pairs=tool.N_PAIRS,
        latent_dim=tool.LATENT_DIM,
    )

    loaded_dims, loaded_delta_idx, loaded_mask, loaded_payload = (
        tool.load_sidecar_choice_state(
            state_path,
            old_archive_sha256="a" * 64,
            old_sidecar_sha256="b" * 64,
            search_signal="proxy_mse",
            search_device="cpu",
            encode_format="packed_661",
            total_pairs=tool.N_PAIRS,
            latent_dim=tool.LATENT_DIM,
        )
    )
    assert payload == loaded_payload
    assert np.array_equal(loaded_dims, dims)
    assert np.array_equal(loaded_delta_idx, delta_idx)
    assert np.array_equal(loaded_mask, searched_mask)

    with pytest.raises(ValueError, match="context mismatch"):
        tool.load_sidecar_choice_state(
            state_path,
            old_archive_sha256="0" * 64,
            old_sidecar_sha256="b" * 64,
            search_signal="proxy_mse",
            search_device="cpu",
            encode_format="packed_661",
            total_pairs=tool.N_PAIRS,
            latent_dim=tool.LATENT_DIM,
        )


def test_manifest_readiness_rejects_full_search_without_complete_choice_state(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    sidecar = tool.encode_sidecar_huff_enum(dims, delta_idx)
    sub_dir, archive = _write_runtime_fixture(tmp_path, sidecar=sidecar)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    proof = tool.collect_member_section_proof(archive)
    searched_mask = np.ones(tool.N_PAIRS, dtype=bool)
    searched_mask[-1] = False
    choice_state = _write_choice_state_record(
        tool,
        tmp_path,
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        old_archive_sha256="a" * 64,
        old_sidecar_sha256="1" * 64,
    )
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "new_sidecar_bytes": _proof_sidecar_bytes(proof),
        "old_archive_sha256": "a" * 64,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "search_signal": "proxy_mse",
        "search_device": "cpu",
        "encode_format": "packed_661",
        "smoke_only": False,
        "full_non_smoke_search": True,
        "no_op_detector": _changed_sidecar_no_op(_proof_sidecar_sha(proof)),
        "local_runtime_custody": custody,
        "runtime_manifest": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "member_section_proof": proof,
        "sidecar_choice_state": choice_state,
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert "sidecar_choice_state_incomplete_for_full_search" in out["dispatch_blockers"]


def test_manifest_readiness_rejects_malformed_choice_state_fail_closed(
    tmp_path: Path,
) -> None:
    tool = load_tool()
    dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    sidecar = tool.encode_sidecar_huff_enum(dims, delta_idx)
    sub_dir, archive = _write_runtime_fixture(tmp_path, sidecar=sidecar)
    archive_sha = tool.sha256_of(archive)
    archive_bytes = archive.stat().st_size
    custody = tool.collect_local_runtime_custody(sub_dir, archive_path=archive)
    proof = tool.collect_member_section_proof(archive)
    state_path = tmp_path / "sidecar_choice_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": tool.SIDECAR_CHOICE_STATE_SCHEMA,
                "total_pairs": tool.N_PAIRS,
                "latent_dim": tool.LATENT_DIM,
                "dims": None,
                "delta_idx": [],
                "searched_mask": [],
            }
        )
        + "\n"
    )
    manifest = {
        "lane_id": tool.SIDECAR_LANE_ID,
        "dispatch_blockers": [],
        "archive_path": tool.manifest_path(archive),
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_bytes,
        "new_archive_path": tool.manifest_path(archive),
        "new_archive_sha256": archive_sha,
        "new_archive_bytes": archive_bytes,
        "new_sidecar_bytes": _proof_sidecar_bytes(proof),
        "old_archive_sha256": "a" * 64,
        "ready_for_exact_eval_dispatch": True,
        "runtime_smoke_checked": True,
        "runtime_smoke_evidence": _runtime_smoke_evidence(tool, archive, custody),
        "search_signal": "proxy_mse",
        "search_device": "cpu",
        "encode_format": "packed_661",
        "smoke_only": False,
        "full_non_smoke_search": True,
        "no_op_detector": _changed_sidecar_no_op(_proof_sidecar_sha(proof)),
        "local_runtime_custody": custody,
        "runtime_manifest": custody,
        "runtime_tree_sha256": custody["runtime_tree_sha256"],
        "member_section_proof": proof,
        "sidecar_choice_state": {
            "schema_version": tool.SIDECAR_CHOICE_STATE_SCHEMA,
            "path": tool.manifest_path(state_path),
            "sha256": tool.sha256_of(state_path),
            "n_pairs_completed_total": tool.N_PAIRS,
            "total_pairs": tool.N_PAIRS,
            "full_coverage": True,
            "searched_pair_indices": list(range(tool.N_PAIRS)),
        },
    }

    out = tool.enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=archive,
        submission_dir=sub_dir,
    )

    assert out["ready_for_exact_eval_dispatch"] is False
    assert any(
        str(reason).startswith("sidecar_choice_state_unreadable:")
        for reason in out["dispatch_blockers"]
    )


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


def test_candidate_batch_profiler_selects_fastest_semantic_match(monkeypatch) -> None:
    tool = load_tool()

    def fake_best_pair_proxy_mse_candidate(**kwargs):
        assert kwargs["candidate_batch_size"] in {1, 4}
        return 1.0, 0.5, 1, 1

    ticks = iter([0.0, 10.0, 10.0, 12.0])
    monkeypatch.setattr(
        tool,
        "_best_pair_proxy_mse_candidate",
        fake_best_pair_proxy_mse_candidate,
    )
    monkeypatch.setattr(tool.time, "perf_counter", lambda: next(ticks))

    profile = tool.profile_pair_proxy_mse_candidate_batches(
        decoder=object(),
        base_lat=torch.zeros((1, 2), dtype=torch.float32),
        gt_eval=np.ones((2, 1, 1, 3), dtype=np.float32),
        eval_h=1,
        eval_w=1,
        deltas=np.array([-1.0, 1.0], dtype=np.float32),
        candidate_batch_sizes=[1, 4],
    )

    assert profile["schema_version"] == "a1_sidecar_candidate_batch_profile_v1"
    assert profile["selected_candidate_batch_size"] == 4
    assert profile["selection_reason"] == "fastest_semantic_match"
    assert all(row["semantic_match_scalar_reference"] for row in profile["records"])


def test_candidate_batch_profiler_falls_back_on_semantic_mismatch(monkeypatch) -> None:
    tool = load_tool()

    def fake_best_pair_proxy_mse_candidate(**kwargs):
        if kwargs["candidate_batch_size"] == 1:
            return 1.0, 0.5, 0, 0
        return 1.0, 0.25, 1, 0

    monkeypatch.setattr(
        tool,
        "_best_pair_proxy_mse_candidate",
        fake_best_pair_proxy_mse_candidate,
    )

    profile = tool.profile_pair_proxy_mse_candidate_batches(
        decoder=object(),
        base_lat=torch.zeros((1, 2), dtype=torch.float32),
        gt_eval=np.ones((2, 1, 1, 3), dtype=np.float32),
        eval_h=1,
        eval_w=1,
        deltas=np.array([-1.0, 1.0], dtype=np.float32),
        candidate_batch_sizes=[4, 1],
    )

    assert profile["profiled_candidate_batch_sizes"] == [1, 4]
    assert profile["selected_candidate_batch_size"] == 1
    assert profile["selection_reason"] == "non_scalar_batches_semantic_mismatch"
    assert profile["records"][1]["semantic_match_scalar_reference"] is False


def test_proxy_search_resume_skips_completed_pairs_and_updates_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool = load_tool()

    class Decoder(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            value = latents[:, 0].reshape(-1, 1, 1, 1)
            return value.repeat(1, 6, 1, 1)

    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(HNeRVDecoder=Decoder))
    base = torch.zeros((tool.N_PAIRS, tool.LATENT_DIM), dtype=torch.float32)
    initial_dims = np.full(tool.N_PAIRS, 255, dtype=np.int64)
    initial_delta_idx = np.full(tool.N_PAIRS, -1, dtype=np.int64)
    initial_mask = np.zeros(tool.N_PAIRS, dtype=bool)
    initial_dims[0] = 2
    initial_delta_idx[0] = 3
    initial_mask[0] = True
    state_path = tmp_path / "sidecar_choice_state.json"
    components = {
        "decoder_sd": {},
        "latents_pre_sidecar": base,
        "latents_with_sidecar_old": base.clone(),
        "eval_size": (1, 1),
        "latent_dim": tool.LATENT_DIM,
        "base_channels": 1,
        "n_pairs": tool.N_PAIRS,
    }
    gt_eval = np.ones((4, 1, 1, 3), dtype=np.uint8)

    dims, delta_idx, meta = tool.search_per_pair_proxy_mse(
        components,
        gt_eval,
        pair_indices=[0, 1],
        candidate_batch_size=4,
        search_device="cpu",
        initial_dims=initial_dims,
        initial_delta_idx=initial_delta_idx,
        initial_searched_mask=initial_mask,
        state_path=state_path,
        state_context={
            "old_archive_sha256": "a" * 64,
            "old_sidecar_sha256": "b" * 64,
            "search_signal": "proxy_mse",
            "search_device": "cpu",
            "encode_format": "packed_661",
            "total_pairs": tool.N_PAIRS,
            "latent_dim": tool.LATENT_DIM,
        },
    )

    assert meta["n_pairs_skipped_already_completed"] == 1
    assert meta["n_pairs_completed_this_run"] == 1
    assert meta["n_pairs_completed_total"] == 2
    assert dims[0] == 2
    assert delta_idx[0] == 3
    assert dims[1] == 0
    state_payload = json.loads(state_path.read_text())
    assert state_payload["searched_pair_indices"] == [0, 1]


def test_proxy_search_explicit_cpu_device_path(monkeypatch) -> None:
    tool = load_tool()

    class Decoder(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            value = latents[:, 0].reshape(-1, 1, 1, 1)
            return value.repeat(1, 6, 1, 1)

    monkeypatch.setitem(sys.modules, "model", types.SimpleNamespace(HNeRVDecoder=Decoder))
    base = torch.zeros((tool.N_PAIRS, tool.LATENT_DIM), dtype=torch.float32)
    components = {
        "decoder_sd": {},
        "latents_pre_sidecar": base,
        "latents_with_sidecar_old": base.clone(),
        "eval_size": (1, 1),
        "latent_dim": tool.LATENT_DIM,
        "base_channels": 1,
        "n_pairs": tool.N_PAIRS,
    }
    gt_eval = np.ones((2, 1, 1, 3), dtype=np.uint8)

    dims, delta_idx, meta = tool.search_per_pair_proxy_mse(
        components,
        gt_eval,
        pair_indices=[0],
        candidate_batch_size=4,
        search_device="cpu",
    )

    assert meta["search_device"] == "cpu"
    assert dims[0] == 0
    assert tool.SIDECAR_DELTAS_X100[delta_idx[0]] == 10
