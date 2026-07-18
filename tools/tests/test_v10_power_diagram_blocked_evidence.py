# SPDX-License-Identifier: MIT
"""Tests for the minimal read-only v10 blocked-evidence math surface."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import tools.v10_power_diagram_blocked_evidence as evidence
from tac.boundary_math.power_diagram_witness import power_assign


def _fixture_immutable_identity(
    *,
    custody: dict[object, object],
    expected_pairs: int,
    seg_hw: tuple[int, int],
    camera_hwc: tuple[int, int, int],
    n_classes: int,
    head_rank: int,
    ridge: float,
    torch_threads_requested: int,
    torch_threads_effective: int,
    torch_interop_threads_requested: int,
    torch_interop_threads_effective: int,
    implementation: dict[object, object],
) -> dict[str, object]:
    """Build a test-only checkpoint identity; production exposes validators only."""
    return {
        "custody_derivation": evidence.CUSTODY_DERIVATION,
        "custody": custody,
        "geometry": {
            "expected_pairs": expected_pairs,
            "seg_hw": list(seg_hw),
            "camera_hwc": list(camera_hwc),
            "n_classes": n_classes,
            "head_rank": head_rank,
        },
        "config": {
            "ridge": float(ridge),
            "batch_size": 1,
            "device": "cpu",
            "dtype": "torch.float32",
            "deterministic_algorithms": True,
            "torch_threads_requested": torch_threads_requested,
            "torch_threads_effective": torch_threads_effective,
            "torch_interop_threads_requested": torch_interop_threads_requested,
            "torch_interop_threads_effective": torch_interop_threads_effective,
        },
        "implementation": implementation,
    }


def _fixture_checkpoint_payload(
    state: evidence.ExtractionState, immutable_identity: dict[str, object]
) -> dict[str, object]:
    """Build a test-only immutable checkpoint payload for parser coverage."""
    return {
        "schema": evidence.CHECKPOINT_SCHEMA,
        "status": state.status,
        "next_canonical_frame": state.next_frame,
        "immutable_identity": immutable_identity,
        "statistics": {
            "gram": state.statistics.gram.tolist(),
            "rhs": state.statistics.rhs.tolist(),
            "label_counts": state.statistics.label_counts.tolist(),
            "sample_count": state.statistics.sample_count,
        },
        "adjacency": [list(edge) for edge in sorted(state.adjacency)],
        "positive_control": {
            "power_target_mismatch_count": state.positive_power_mismatches,
            "cpu_torch_forward_mismatch_count": state.positive_forward_mismatches,
        },
        "blocked_reason": state.blocked_reason,
        "updated_utc": "fixture",
    }


def test_safe_module_has_no_destructive_resume_or_scratch_write_capability() -> None:
    source_path = Path(evidence.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_names = {
        "unlink",
        "rmtree",
        "remove",
        "replace",
        "rename",
        "open_memmap",
        "atomic_write_json",
        "cleanup_certified_scratch",
        "certify_feature_cache",
        "prepare_extraction_scratch",
        "write_extraction_checkpoint",
    }
    seen = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    seen.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    assert forbidden_names.isdisjoint(seen)
    assert "import os" not in source
    assert "import shutil" not in source
    assert "import tempfile" not in source


def test_safe_module_exposes_no_checkpoint_or_resume_construction_api() -> None:
    construction_api = {"make_immutable_identity", "extraction_checkpoint_payload"}
    assert construction_api.isdisjoint(evidence.__all__)
    assert all(not hasattr(evidence, name) for name in construction_api)


def test_streaming_fit_matches_dense_reference_deterministically() -> None:
    rng = np.random.default_rng(543)
    features = rng.normal(size=(97, 4))
    labels = np.argmax(features @ rng.normal(size=(4, 5)) + rng.normal(size=5), axis=1)
    statistics = evidence.StreamingRidgeSufficientStatistics(4, 5)
    for start, stop in ((0, 11), (11, 43), (43, 97)):
        statistics.update(features[start:stop], labels[start:stop])
    weight, bias = statistics.solve(1e-6)
    design = np.concatenate((features, np.ones((features.shape[0], 1))), axis=1)
    desired = np.eye(5)[labels] - 1.0 / 5.0
    dense = np.linalg.solve(design.T @ design + 1e-6 * np.eye(5), design.T @ desired)
    np.testing.assert_allclose(weight, dense[:-1].T, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(bias, dense[-1], atol=1e-12, rtol=1e-12)
    target = evidence.affine_scores_to_power_target(
        weight,
        bias,
        adjacency=tuple((i, j) for i in range(5) for j in range(i + 1, 5)),
    )
    first = power_assign(features, target)
    second = power_assign(features, target)
    np.testing.assert_array_equal(first, second)


def test_checkpoint_parse_is_pure_and_preserves_float64_statistics() -> None:
    identity = _fixture_immutable_identity(
        custody={"fixture": True},
        expected_pairs=2,
        seg_hw=(2, 3),
        camera_hwc=(4, 5, 3),
        n_classes=3,
        head_rank=2,
        ridge=1e-6,
        torch_threads_requested=6,
        torch_threads_effective=6,
        torch_interop_threads_requested=18,
        torch_interop_threads_effective=18,
        implementation={"fixture": True},
    )
    statistics = evidence.StreamingRidgeSufficientStatistics(2, 3)
    features = np.arange(12, dtype=np.float64).reshape(6, 2)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    statistics.update(features, labels)
    state = evidence.ExtractionState(next_frame=1, statistics=statistics, adjacency={(0, 1)})
    payload = _fixture_checkpoint_payload(state, identity)
    before = repr(payload)
    restored = evidence.validate_extraction_checkpoint(payload, expected_identity=identity)
    assert repr(payload) == before
    assert restored.statistics.gram.dtype == np.float64
    assert restored.statistics.rhs.dtype == np.float64
    np.testing.assert_array_equal(restored.statistics.gram, statistics.gram)
    np.testing.assert_array_equal(restored.statistics.rhs, statistics.rhs)


def test_order0_semantics_are_ideal_entropy_not_a_realizable_bound() -> None:
    estimate = evidence.order0_ideal_entropy_estimate(b"aaabccccddddd")
    assert estimate["label"] == "DERIVED_OPTIMISTIC_ROUNDED_UP_IDEAL_ENTROPY_BYTES"
    assert estimate["assumptions"] == "empirical PMF free; no model/header/termination overhead"
    assert estimate["rounded_up_ideal_entropy_bytes"] >= 1
    assert all("lower" not in key and "ceiling" not in key for key in estimate)
