from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _tool():
    name = "_measure_yhat_native_equivalence_for_tests"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[3] / "tools/measure_yhat_native_equivalence.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_float32_ulp_distance_covers_equal_adjacent_signed_zero_and_nonfinite() -> None:
    tool = _tool()
    below = np.nextafter(np.float32(1.0), np.float32(2.0))
    negative = np.nextafter(np.float32(-1.0), np.float32(-2.0))
    values = tool.float32_ulp_distance(
        np.array([1.0, 1.0, -1.0, 0.0, np.inf]), np.array([1.0, below, negative, -0.0, np.inf], dtype=np.float32)
    )
    assert values.tolist() == [0, 1, 1, 0, 0]
    negative_tiny = np.nextafter(np.float32(0.0), np.float32(-1.0))
    positive_tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    assert tool.float32_ulp_distance(negative_tiny, np.float32(0.0)).item() == 1
    assert tool.float32_ulp_distance(negative_tiny, positive_tiny).item() == 2
    assert tool.float32_ulp_distance(np.array([np.inf]), np.array([-np.inf]))[0] == np.iinfo(np.int64).max


def test_tensor_bit_equality_distinguishes_signed_zero() -> None:
    tool = _tool()

    class FakeTensor:
        def __init__(self, values: list[float]) -> None:
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self.values

    comparison = tool._tensor_comparison(FakeTensor([0.0]), FakeTensor([-0.0]))
    assert comparison["bit_identical"] is False
    assert comparison["max_native_f32_ulp"] == 0


def test_narrow_receipt_classification_rules() -> None:
    tool = _tool()
    assert (
        tool.classify_equivalence(
            rational_plane_exact=False,
            oracle_bit_identical=True,
            metrics_bit_identical=True,
            native_f32_deltas_described=False,
        )
        == "BIT_IDENTICAL"
    )
    assert (
        tool.classify_equivalence(
            rational_plane_exact=True,
            oracle_bit_identical=False,
            metrics_bit_identical=False,
            native_f32_deltas_described=True,
        )
        == "EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS"
    )
    assert (
        tool.classify_equivalence(
            rational_plane_exact=True,
            oracle_bit_identical=False,
            metrics_bit_identical=False,
            native_f32_deltas_described=False,
        )
        == "NOT_EQUIVALENT_UNDER_NARROW_SCOPE"
    )


def test_scored_posenet_extraction_uses_declared_head_order_and_first_half_only() -> None:
    tool = _tool()
    model = SimpleNamespace(
        posenet=SimpleNamespace(
            hydra=SimpleNamespace(heads=[SimpleNamespace(name="aux", out=4), SimpleNamespace(name="pose", out=12)])
        )
    )
    fake_torch = SimpleNamespace(cat=lambda values, dim: np.concatenate(values, axis=dim))
    outputs = {
        "aux": np.array([[99, 98, 97, 96]], dtype=np.float32),
        "pose": np.arange(12, dtype=np.float32)[None, :],
    }
    assert tool.scored_posenet_output(model, outputs, fake_torch).tolist() == [list(range(6))]


def test_finite_ulp_class_predicate_requires_proof_finite_metrics_and_no_sentinel() -> None:
    tool = _tool()
    comparisons = {
        "segnet_logits": {"has_nonfinite": False, "sentinel": None, "max_abs_delta": 0.25, "max_native_f32_ulp": 2}
    }
    assert tool.native_f32_deltas_described(
        rational_plane_exact=True,
        direct={"d_seg": 0.0, "d_pose": 1.0},
        native={"d_seg": 0.0, "d_pose": 1.25},
        metric_deltas={"d_seg": 0.0, "d_pose": 0.25},
        comparisons=comparisons,
    )
    comparisons["segnet_logits"]["sentinel"] = tool.NONFINITE_SENTINEL
    assert not tool.native_f32_deltas_described(
        rational_plane_exact=True,
        direct={"d_seg": 0.0},
        native={"d_seg": 0.0},
        metric_deltas={"d_seg": 0.0},
        comparisons=comparisons,
    )


def test_atomic_stage_is_preserved_and_different_rebuild_refuses(tmp_path: Path) -> None:
    tool = _tool()
    path = tmp_path / "stages/pair_0000.json"
    assert tool._write_stage_once(path, {"pair": 0}) == {"pair": 0}
    assert json.loads(path.read_text()) == {"pair": 0}
    with pytest.raises(tool.YhatNativeMeasurementError, match="differs"):
        tool._write_stage_once(path, {"pair": 1})
    assert tool._write_stage_once(path, {"pair": 0, "timing": {"pair_runtime_seconds": 99.0}}) == {"pair": 0}


def test_pair_ids_are_deterministic_and_require_n24() -> None:
    tool = _tool()
    raw = ",".join(str(value) for value in range(24))
    assert tool.parse_pair_ids(raw) == tuple(range(24))
    with pytest.raises(tool.YhatNativeMeasurementError, match="at least 24"):
        tool.parse_pair_ids("0,1")


def test_normalized_resume_binding_ignores_transport_but_refuses_scientific_changes() -> None:
    tool = _tool()
    kwargs = {
        "pair_ids": tuple(range(24)),
        "cpu_threads": 1,
        "max_nodes_per_block": 4096,
        "files": {"donor": {"sha256": "one", "bytes": 1}},
        "sacred": {"before": {"metadata_sha256": "sacred"}},
        "checkpoint_metadata": {"epoch": 725, "render_hw": [384, 512]},
        "git_commit": "abc",
    }
    binding = tool.normalized_scientific_binding(**kwargs)
    sha = __import__("hashlib").sha256(tool._canonical(binding)).hexdigest()
    state = {"schema": tool.STATE_SCHEMA, "binding": binding, "binding_sha256": sha}
    assert tool.resume_binding_matches(state, binding, sha)
    assert not tool.resume_binding_matches(
        state,
        tool.normalized_scientific_binding(**{**kwargs, "cpu_threads": 2}),
        sha,
    )
    assert tool.normalized_scientific_binding(**{**kwargs, "pair_ids": tuple(range(1, 25))}) != binding
    assert tool.normalized_scientific_binding(**{**kwargs, "max_nodes_per_block": 1}) != binding
    changed_files = {"donor": {"sha256": "two", "bytes": 1}}
    assert tool.normalized_scientific_binding(**{**kwargs, "files": changed_files}) != binding


def test_completed_resume_stages_require_prefix_and_hash_custody(tmp_path: Path) -> None:
    tool = _tool()
    pairs = tuple(range(24))
    stage = {
        "schema": tool.SCHEMA,
        "pair_id": 0,
        "preimage_policy": tool.PREIMAGE_POLICY,
        "f32_receiver_arithmetic_admissibility": tool.F32_ADMISSIBILITY,
    }
    stages = tmp_path / "stages"
    stages.mkdir()
    (stages / "pair_0000.json").write_text(json.dumps(stage))
    state = {"completed_pairs": [0], "completed_stage_sha256": {"0": tool._stage_sha256(stage)}}
    assert tool._load_completed_rows(state=state, pairs=pairs, stages=stages) == {0: stage}
    with pytest.raises(tool.YhatNativeMeasurementError, match="ordered prefix"):
        tool._load_completed_rows(
            state={"completed_pairs": [1], "completed_stage_sha256": {"1": tool._stage_sha256(stage)}},
            pairs=pairs,
            stages=stages,
        )
    stage["pair_id"] = 9
    (stages / "pair_0000.json").write_text(json.dumps(stage))
    with pytest.raises(tool.YhatNativeMeasurementError, match="contract mismatch"):
        tool._load_completed_rows(state=state, pairs=pairs, stages=stages)


def test_exact_raw_size_refusal(tmp_path: Path) -> None:
    tool = _tool()
    raw = tmp_path / "donor.raw"
    raw.write_bytes(b"too-short")
    with pytest.raises(tool.YhatNativeMeasurementError, match="donor raw bytes"):
        tool.validate_donor_raw_size(raw, 24)


def test_aggregate_math_is_exact_and_narrow() -> None:
    tool = _tool()
    comparison = {"max_abs_delta": 0.5, "max_native_f32_ulp": 3}
    rows = [
        {
            "exact_rational_planes": [{"exact_blocks": 2, "exact_samples": 4, "failures": 0}],
            "segnet_argmax_disagreement": 1,
            "classification": "BIT_IDENTICAL",
            "native_f32_deltas_described": False,
            "direct": {"d_seg": 0.0, "d_pose": 2.0},
            "yhat_native": {"d_seg": 0.0, "d_pose": 2.0},
            "comparisons": {"segnet_logits": comparison},
            "timing": {"pair_runtime_seconds": 2.0},
        },
        {
            "exact_rational_planes": [{"exact_blocks": 3, "exact_samples": 5, "failures": 0}],
            "segnet_argmax_disagreement": 2,
            "classification": "EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS",
            "native_f32_deltas_described": True,
            "direct": {"d_seg": 0.2, "d_pose": 4.0},
            "yhat_native": {"d_seg": 0.3, "d_pose": 5.0},
            "comparisons": {"segnet_logits": {"max_abs_delta": 1.0, "max_native_f32_ulp": 7}},
            "timing": {"pair_runtime_seconds": 4.0},
        },
    ]
    rows[0]["exact_rational_planes"][0]["solve_runtime_seconds"] = 0.5
    rows[1]["exact_rational_planes"][0]["solve_runtime_seconds"] = 1.5
    aggregate = tool.aggregate_rows(rows)
    assert aggregate["exact_blocks"] == 5 and aggregate["exact_samples"] == 9
    assert aggregate["segnet_argmax_disagreements"] == 3
    assert aggregate["classification"] == "EXACT_RATIONAL_PLANE_NATIVE_F32_ULP_CLASS"
    assert aggregate["direct_mean"] == {"d_seg": 0.1, "d_pose": 3.0}
    assert aggregate["yhat_native_mean"] == {"d_seg": 0.15, "d_pose": 3.5}
    assert aggregate["surface_maxima"]["segnet_logits"] == {"max_abs_delta": 1.0, "max_native_f32_ulp": 7}
    assert aggregate["timing"]["observed_pair_seconds_mean"] == 3.0
    assert aggregate["timing"]["derived_n600_preimage_minutes"] == 10.0


def test_sacred_snapshot_comparison_detects_fixture_mutation(tmp_path: Path) -> None:
    tool = _tool()
    fixture = tmp_path / "sacred"
    fixture.mkdir()
    (fixture / "checkpoint.npz").write_bytes(b"one")
    before = tool.tree_snapshot(fixture)
    assert before == tool.tree_snapshot(fixture)
    (fixture / "checkpoint.npz").write_bytes(b"two")
    assert before != tool.tree_snapshot(fixture)
