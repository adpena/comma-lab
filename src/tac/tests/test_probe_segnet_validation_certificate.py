from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[3] / "tools/probe_segnet_validation_certificate.py"
    spec = importlib.util.spec_from_file_location("probe_segnet_validation_certificate_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canary_receipt_has_real_controls_and_false_authority() -> None:
    receipt = _module().build_canary_receipt(["probe", "--canaries-only"])
    assert receipt["controls"]["known_linear_map_inside_and_outside"]["status"] == "PASS"
    assert receipt["controls"]["confusion_meter_positive_and_negative"]["status"] == "PASS"
    assert receipt["authority"] == {
        "axis": "[local mechanism only]",
        "score_claim": False,
        "promotion_eligible": False,
        "review_status": "unreviewed_fix_round_2",
    }
    assert receipt["controls"]["calibration_content_mutation"]["status"] == "PASS"
    assert receipt["rigorous_mechanism"]["verdict"] == "NO-GO"


def test_atomic_receipt_replaces_complete_json(tmp_path) -> None:
    module = _module()
    path = tmp_path / "receipt.json"
    module.atomic_write_json(path, {"stage": 1})
    module.atomic_write_json(path, {"stage": 2})
    assert json.loads(path.read_text()) == {"stage": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_terminal_verdict_fails_empirical_on_unsafe_accept_and_never_grants_throughput() -> None:
    verdict = _module()._terminal_verdict(
        [{"confusion": {"unsafe_accepts_any": 1}}],
        economics={"cadences": {"2": {"derived_speedup": 99.0}}},
    )
    assert verdict["empirical"] == "NO-GO"
    assert verdict["joint_held_descent"] == "NO-GO"
    assert verdict["segnet_dseg_proxy"] == "ADVISORY-HOLDOUT-PASS"
    assert verdict["rigorous"] == "NO-GO"
    assert verdict["throughput"] == "NEEDS-MORE"


def test_terminal_go_requires_sequence_integrated_measurement() -> None:
    module = _module()
    no_sequence = module._terminal_verdict([], economics={"cadences": {"4": {"derived_speedup": 10.0}}})
    fake_sequence = module._terminal_verdict([], sequence_integrated_whole_step={"measured_speedup": 1.3})
    laundered_sequence = module._terminal_verdict(
        [],
        sequence_integrated_whole_step={
            "authority": "MEASURED_SEQUENCE_INTEGRATED_WHOLE_STEP",
            "receipt_path": "/does/not/exist.json",
            "receipt_sha256": "a" * 64,
            "measured_speedup": 1.3,
        },
    )
    assert no_sequence["throughput"] == "NEEDS-MORE"
    assert fake_sequence["throughput"] == "NEEDS-MORE"
    assert laundered_sequence["throughput"] == "NEEDS-MORE"


def test_camera_boundary_accepts_numpy_renderer_output(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    torch = pytest.importorskip("torch")

    class Renderer:
        def __init__(self) -> None:
            self.code = [None, None]

        def render_pair(self, _pair: int):
            return np.ones((2, 3, 3), dtype=np.float32), None

    seen: list[np.ndarray] = []
    monkeypatch.setattr(module, "_torch_R_to_camera_uint8", lambda value: seen.append(value) or value.astype(np.uint8))
    result = module._camera_frame(Renderer(), torch.zeros(1))
    assert result.dtype == np.uint8
    assert seen[0].shape == (2, 3, 3)


def test_calibration_sha_binds_numeric_content_and_identity() -> None:
    module = _module()
    base = {
        "regime": "early",
        "anchor_margins": np.array([2.0, 3.0]),
        "candidate_margin_arrays": np.array([[1.0, 2.0]]),
        "feature_displacements_linf": np.array([0.5]),
        "derived_bounds": np.array([2.0, 2.0]),
        "candidate_indices": [0],
        "fractions": [0.01],
        "anchor_feature": np.array([1.0], dtype=np.float32),
        "anchor_split_identity_sha256": "a" * 64,
    }
    reference = module._json_sha256(module._calibration_custody_payload(**base))
    mutations = (
        ("anchor_margins", np.array([2.0, 4.0])),
        ("candidate_margin_arrays", np.array([[1.0, 2.5]])),
        ("feature_displacements_linf", np.array([0.6])),
        ("derived_bounds", np.array([2.0, 2.5])),
        ("candidate_indices", [1]),
        ("fractions", [0.02]),
        ("anchor_feature", np.array([2.0], dtype=np.float32)),
        ("anchor_split_identity_sha256", "b" * 64),
    )
    for key, value in mutations:
        changed = dict(base)
        changed[key] = value
        assert module._json_sha256(module._calibration_custody_payload(**changed)) != reference


def test_component_economics_uses_only_prefix_plus_array_gate_as_cheap_timing() -> None:
    module = _module()
    regimes = [{
        "holdout": [{
            "proxy_decision": {"status": "PROXY_ACCEPT"},
            "timing_measured_seconds": {
                "camera_render_R": 1000.0,
                "cheap_prefix_only": 0.25,
                "array_gate_only": 0.05,
                "exact_segnet_ce_dseg": 10.0,
                "posenet": 20.0,
            },
        }],
    }]
    economics = module._component_economics(regimes)
    assert economics["cadences"]["2"]["t_validate_cheap_seconds_current_measured_median"] == 0.3
    assert economics["cadences"]["4"]["status"] == "DERIVED_COMPONENT_FORMULA"
    assert economics["cadences"]["4"]["t_approx_nonrefresh_rows"]
    assert not economics["master_gate_rerun"]


def test_rejection_fallback_uses_inherited_full_teacher_not_exact_safety_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    inherited = {
        cadence: {
            "t_exact_rows": [{"seconds": 10.0, "K": cadence}],
            "t_approx_nonrefresh_rows": [{"seconds": 1.0, "K": cadence}],
        }
        for cadence in (2, 4)
    }
    monkeypatch.setattr(module, "_inherited_yopo_component_rows", lambda: inherited)
    regimes = [{
        "holdout": [{
            "proxy_decision": {"status": "REFRESH"},
            "timing_measured_seconds": {
                "cheap_prefix_only": 0.25,
                "array_gate_only": 0.05,
                "exact_segnet_ce_dseg": 100.0,
                "posenet": 200.0,
            },
        }],
    }]
    economics = module._component_economics(regimes)
    row = economics["cadences"]["2"]
    assert row["t_fallback_seconds_derived_rejection_weighted_inherited_full_teacher_forward_backward"] == 10.0
    assert row["fallback_action"] == "full_teacher_and_refresh"
    assert row["sequence_integration_status"] == "UNINTEGRATED_COMPONENT_ECONOMICS_ONLY"
    safety = economics["current_exact_scorer_safety_measurement"]
    assert safety["exact_segnet_ce_dseg_seconds_current_measured_median"] == 100.0
    assert safety["posenet_seconds_current_measured_median"] == 200.0


def test_terminal_separates_dseg_proxy_from_joint_held_descent() -> None:
    module = _module()
    verdict = module._terminal_verdict([{
        "confusion": {
            "unsafe_accepts_any": 1,
            "unsafe_accepts_dseg": 0,
            "exact_safe_accepts": 2,
        },
    }])
    assert verdict["segnet_dseg_proxy"] == "ADVISORY-HOLDOUT-PASS"
    assert "among 3 proxy accepts" in verdict["segnet_dseg_proxy_reason"]
    assert verdict["joint_held_descent"] == "NO-GO"
    assert verdict["throughput"] == "NEEDS-MORE"


def test_storage_receipt_has_local_and_ssd_waterfall() -> None:
    rows = _module()._storage_free_bytes()
    assert rows["local"]["free_bytes"] > 0
    assert "vertigo_ssd" in rows and "apdatastore_ssd" in rows
