# SPDX-License-Identifier: MIT
"""Portable n4-fixture tests for the QA75 solve-frame targets loader + helpers.

The materializer itself is exercised by the real ms2r_r3 archive run (recorded in
the ddm_b2p memo with a determinism spot-check); these tests cover the LOADER
contract the burn-2 distill stage consumes, plus the atomic-write + sha helpers,
without needing the 277 MB SSD archive.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tac.witness_dsl.qa75_solve_frame_targets import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    SolveFrameTargets,
    SolveFrameTargetsError,
    _atomic_write_npy,
    _sha256_file,
)


def _make_fixture(root, *, n_pairs=4, h=4, w=6, corrupt_sha_pair=None, schema=MANIFEST_SCHEMA):
    root.mkdir(parents=True, exist_ok=True)
    pairs = []
    rng = np.random.default_rng(0)
    for i in range(n_pairs):
        stacked = rng.integers(0, 256, size=(2, h, w, 3), dtype=np.uint8)
        name = f"pair-{i:06d}.npy"
        _atomic_write_npy(root / name, stacked)
        sha = _sha256_file(root / name)
        if corrupt_sha_pair == i:
            sha = "0" * 64
        pairs.append({"pair_id": i, "path": name, "sha256": sha, "bytes": (root / name).stat().st_size})
    manifest = {
        "schema": schema,
        "geometry": {"camera_h": h, "camera_w": w, "channels": 3, "frame_shape": [2, h, w, 3]},
        "frame0_described": True,
        "pair_count_total": n_pairs,
        "pair_count_materialized": n_pairs,
        "pairs": pairs,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def test_loader_roundtrip_frames_and_geometry(tmp_path):
    _make_fixture(tmp_path)
    targets = SolveFrameTargets.load(tmp_path)
    assert targets.pair_count == 4
    assert targets.frame_shape == (2, 4, 6, 3)
    assert targets.frame0_described is True
    f0, f1 = targets.pair(0)
    assert f0.shape == (4, 6, 3) and f0.dtype == np.uint8
    assert f1.shape == (4, 6, 3)
    assert targets.frame1(3).shape == (4, 6, 3)
    assert targets.verify_sha(0) is True


def test_loader_verify_flag_raises_on_corrupt_sha(tmp_path):
    _make_fixture(tmp_path, corrupt_sha_pair=2)
    targets = SolveFrameTargets.load(tmp_path)
    assert targets.verify_sha(2) is False
    with pytest.raises(SolveFrameTargetsError):
        targets.pair(2, verify=True)
    # a clean pair still loads
    targets.pair(0, verify=True)


def test_loader_fail_closed_missing_manifest_and_bad_schema(tmp_path):
    with pytest.raises(SolveFrameTargetsError):
        SolveFrameTargets.load(tmp_path)
    _make_fixture(tmp_path, schema="not_qa75.v9")
    with pytest.raises(SolveFrameTargetsError):
        SolveFrameTargets.load(tmp_path)


def test_loader_pair_id_out_of_range_raises(tmp_path):
    _make_fixture(tmp_path, n_pairs=4)
    targets = SolveFrameTargets.load(tmp_path)
    with pytest.raises(SolveFrameTargetsError):
        targets.pair(4)
    with pytest.raises(SolveFrameTargetsError):
        targets.pair(-1)


def test_atomic_write_and_sha_helpers(tmp_path):
    arr = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
    path = tmp_path / "pair-000000.npy"
    _atomic_write_npy(path, arr)
    assert path.is_file()
    loaded = np.load(path, allow_pickle=False)
    assert np.array_equal(loaded, arr)
    # deterministic sha
    assert _sha256_file(path) == _sha256_file(path)
