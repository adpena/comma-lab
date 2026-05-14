# SPDX-License-Identifier: MIT
"""Hardening tests for the A1 per-pair latent sidecar builder."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools/build_a1_per_pair_latent_correction_sidecar.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("a1_sidecar_builder", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_arrays(mod, searched: tuple[int, ...] = (0, 1)):
    dims = np.full(mod.N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(mod.N_PAIRS, -1, dtype=np.int64)
    searched_mask = np.zeros(mod.N_PAIRS, dtype=bool)
    for pair_index in searched:
        searched_mask[pair_index] = True
    return dims, delta_idx, searched_mask


def _manifest_for_state(mod, state_path: Path) -> dict:
    record = mod.sidecar_choice_state_manifest_record(state_path)
    payload = mod._load_sidecar_choice_state_payload(state_path)
    sidecar = mod._encode_sidecar_from_state_payload(payload, encode_format="packed_661")
    return {
        "sidecar_choice_state": record,
        "full_non_smoke_search": False,
        "total_pairs": mod.N_PAIRS,
        "old_archive_sha256": "old-archive",
        "search_signal": "proxy_mse",
        "search_device": payload["search_device"],
        "encode_format": "packed_661",
        "new_sidecar_bytes": len(sidecar),
        "no_op_detector": {
            "old_inner_sidecar_sha256": "old-sidecar",
            "new_inner_sidecar_sha256": hashlib.sha256(sidecar).hexdigest(),
        },
    }


def test_choice_state_blocks_completed_pairs_without_scalar_provenance(tmp_path: Path):
    mod = _load_tool()
    dims, delta_idx, searched_mask = _state_arrays(mod)
    state_path = tmp_path / "sidecar_choice_state.json"
    mod.write_sidecar_choice_state(
        state_path,
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        pair_search_records={},
        old_archive_sha256="old-archive",
        old_sidecar_sha256="old-sidecar",
        search_signal="proxy_mse",
        search_device="cpu",
        encode_format="packed_661",
        total_pairs=mod.N_PAIRS,
        latent_dim=mod.LATENT_DIM,
    )

    blockers = mod._sidecar_choice_state_blockers(_manifest_for_state(mod, state_path))

    assert "sidecar_pair_search_records_missing_for_completed_pairs:2" in blockers
    assert "sidecar_choice_state_payload_dispatch_safe_flag_mismatch" not in blockers


def test_choice_state_accepts_scalar_direct_pair_records(tmp_path: Path):
    mod = _load_tool()
    dims, delta_idx, searched_mask = _state_arrays(mod)
    records = {
        pair_index: mod._sidecar_pair_search_record(
            pair_index=pair_index,
            search_signal="proxy_mse",
            search_device="cpu",
            requested_candidate_batch_size=1,
            candidate_batch_size=1,
            base_mse=1.0,
            best_mse=1.0,
            best_dim=255,
            best_delta_idx=-1,
            scalar_reference_status="scalar_direct",
        )
        for pair_index in (0, 1)
    }
    state_path = tmp_path / "sidecar_choice_state.json"
    mod.write_sidecar_choice_state(
        state_path,
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        pair_search_records=records,
        old_archive_sha256="old-archive",
        old_sidecar_sha256="old-sidecar",
        search_signal="proxy_mse",
        search_device="cpu",
        encode_format="packed_661",
        total_pairs=mod.N_PAIRS,
        latent_dim=mod.LATENT_DIM,
    )

    blockers = mod._sidecar_choice_state_blockers(_manifest_for_state(mod, state_path))

    assert not [item for item in blockers if item.startswith("sidecar_pair_search")]


def test_choice_state_blocks_mps_proxy_search_for_dispatch(tmp_path: Path):
    mod = _load_tool()
    dims, delta_idx, searched_mask = _state_arrays(mod)
    state_path = tmp_path / "sidecar_choice_state.json"
    mod.write_sidecar_choice_state(
        state_path,
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        pair_search_records={},
        old_archive_sha256="old-archive",
        old_sidecar_sha256="old-sidecar",
        search_signal="proxy_mse",
        search_device="mps",
        encode_format="packed_661",
        total_pairs=mod.N_PAIRS,
        latent_dim=mod.LATENT_DIM,
    )

    blockers = mod._sidecar_choice_state_blockers(_manifest_for_state(mod, state_path))

    assert "sidecar_choice_state_mps_proxy_search_advisory_only" in blockers


def test_manifest_top_level_blocks_mps_proxy_search_for_dispatch():
    mod = _load_tool()

    blockers = mod._manifest_top_level_dispatch_blockers({"search_device": "mps"})

    assert "mps_search_device_advisory_only_not_exact_eval_ready" in blockers


def test_recheck_unproven_pairs_reopens_legacy_completed_pairs():
    mod = _load_tool()
    _, _, searched_mask = _state_arrays(mod, searched=(0, 1, 2))
    safe_record = mod._sidecar_pair_search_record(
        pair_index=2,
        search_signal="proxy_mse",
        search_device="cpu",
        requested_candidate_batch_size=1,
        candidate_batch_size=1,
        base_mse=1.0,
        best_mse=1.0,
        best_dim=255,
        best_delta_idx=-1,
        scalar_reference_status="scalar_direct",
    )

    plan = mod._sidecar_pair_work_plan(
        pair_indices=[0, 1, 2, 3],
        searched_mask=searched_mask,
        pair_search_records={2: safe_record},
        recheck_unproven_pairs=True,
    )

    assert plan["recheck_pairs"] == {0, 1}
    assert plan["work_pair_indices"] == [0, 1, 3]
    assert plan["skipped_already_completed"] == [2]


def test_recheck_unproven_pairs_disabled_preserves_existing_resume_behavior():
    mod = _load_tool()
    _, _, searched_mask = _state_arrays(mod, searched=(0, 1, 2))

    plan = mod._sidecar_pair_work_plan(
        pair_indices=[0, 1, 2, 3],
        searched_mask=searched_mask,
        pair_search_records={},
        recheck_unproven_pairs=False,
    )

    assert plan["recheck_pairs"] == set()
    assert plan["work_pair_indices"] == [3]
    assert plan["skipped_already_completed"] == [0, 1, 2]


def test_sidecar_output_lock_blocks_duplicate_writers(tmp_path: Path):
    mod = _load_tool()
    lock = mod.SidecarOutputLock(tmp_path)
    lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            mod.SidecarOutputLock(tmp_path).acquire()
    finally:
        lock.release()
