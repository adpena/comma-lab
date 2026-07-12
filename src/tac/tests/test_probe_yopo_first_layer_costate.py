"""Focused no-scorer tests for the YOPO measurement receipt contract."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("yopo_probe", ROOT / "tools/probe_yopo_first_layer_costate.py")
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _regime(*, ks: tuple[int, ...], descent: bool = True) -> dict:
    row = {
        "status": "MEASURED",
        "controls": {"status": "PASS"},
        "arms": [
            {
                "K": k,
                "status": "MEASURED",
                "steps": [
                    {
                        "candidate_non_descent": not descent,
                        "bank_age_steps": step % k,
                        "timing_measured_seconds": {"operational_cycle": 10.0 / k},
                        "controls": {"sign_reversed_behavioral_negative": {"status": "PASS"}},
                    }
                    for step in range(max(k, 4))
                ],
                "controls": {
                    "status": "PASS",
                    "full_age_horizon_exercised": True,
                },
                "summary": {
                    "mean_operational_cycle_seconds": 10.0 / k,
                    "mean_ce_regret_vs_exact_reference": float(k - 1),
                    "mean_dseg_regret_vs_exact_reference": float(k - 1),
                },
            }
            for k in ks
        ],
    }
    row["pareto"] = probe._pareto_knee(row["arms"])
    return row


def test_admission_requires_all_registered_regimes_and_matching_knee():
    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    verdict = probe._admission(rows)
    assert verdict["status"] == "GO"
    assert verdict["selected_K"] == 2
    rows["late"] = _regime(ks=(1, 2))
    assert probe._admission(rows)["status"] == "NEEDS-MORE"


def test_admission_refuses_when_no_shared_descent_cadence_survives():
    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    rows["boundary"] = _regime(ks=(1, 2, 4), descent=False)
    verdict = probe._admission(rows)
    assert verdict["status"] == "NO-GO"
    assert verdict["pareto_knee_by_regime"] != dict.fromkeys(probe.REGIMES, 2)
    assert verdict["falsified_cadences_by_regime"]["boundary"] == [1, 2, 4]


def test_admission_allows_clean_shared_knee_when_other_cadence_is_falsified():
    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    for row in rows.values():
        failed = next(arm for arm in row["arms"] if arm["K"] == 4)
        failed["status"] = "NO_GO_NON_DESCENT"
        failed["steps"] = [
            {
                "candidate_non_descent": True,
                "provider_fallback": False,
                "bank_age_steps": 0,
                "timing_measured_seconds": {"operational_cycle": 1.0},
            }
        ]
        surviving = next(arm for arm in row["arms"] if arm["K"] == 2)
        surviving["summary"]["mean_ce_regret_vs_exact_reference"] = 0.0
        surviving["summary"]["mean_dseg_regret_vs_exact_reference"] = 0.0
        row["pareto"] = probe._pareto_knee(row["arms"])
    verdict = probe._admission(rows)
    assert verdict["status"] == "GO"
    assert verdict["selected_K"] == 2
    assert verdict["falsified_cadences_by_regime"] == {name: [4] for name in probe.REGIMES}


def test_admission_refuses_when_selected_cadence_does_not_survive_every_regime():
    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    failed = next(arm for arm in rows["late"]["arms"] if arm["K"] == 2)
    failed["status"] = "NO_GO_PROVIDER_FALLBACK"
    failed["steps"] = [
        {
            "candidate_non_descent": False,
            "provider_fallback": True,
            "bank_age_steps": 0,
            "timing_measured_seconds": {"operational_cycle": 1.0},
        }
    ]
    rows["late"]["pareto"] = probe._pareto_knee(rows["late"]["arms"])
    verdict = probe._admission(rows)
    assert verdict["status"] == "NO-GO"
    assert verdict["falsified_cadences_by_regime"]["late"] == [2]


def test_admission_requires_measurement_canaries():
    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    rows["early"]["controls"]["status"] = "FAIL"
    verdict = probe._admission(rows)
    assert verdict["status"] == "NEEDS-MORE"
    assert "controls" in " ".join(verdict["reason"])


def test_pareto_knee_requires_measured_speed_regret_nondominance_not_max_descent_k():
    arms = _regime(ks=(1, 2, 4))["arms"]
    # K=4 is fast but loses on both regrets to K=2, so it is dominated even
    # though every arm descends. The knee follows the measured Pareto front.
    arms[-1]["summary"]["mean_ce_regret_vs_exact_reference"] = 3.0
    arms[-1]["summary"]["mean_dseg_regret_vs_exact_reference"] = 3.0
    arms[-1]["summary"]["mean_operational_cycle_seconds"] = 8.0
    for step in arms[-1]["steps"]:
        step["timing_measured_seconds"]["operational_cycle"] = 8.0
    arms[-2]["summary"]["mean_ce_regret_vs_exact_reference"] = 1.0
    arms[-2]["summary"]["mean_dseg_regret_vs_exact_reference"] = 1.0
    result = probe._pareto_knee(arms)
    assert result["pareto_nondominated_K"] == [1, 2]
    assert result["pareto_knee_K"] == 1
    assert "utopia" in result["rule"]


def test_admission_rejects_k1_only_or_no_measured_whole_step_speedup():
    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    for row in rows.values():
        row["pareto"] = {
            "pareto_nondominated_K": [1],
            "pareto_knee_K": 1,
            "rows": [
                {
                    "K": 1,
                    "measured_operational_speedup_vs_K1": 1.0,
                    "conservative_speedup_lower_bound_vs_K1": 1.0,
                }
            ],
        }
    verdict = probe._admission(rows)
    assert verdict["status"] == "NO-GO"
    assert verdict["selected_K"] == 1

    rows = {name: _regime(ks=(1, 2, 4)) for name in probe.REGIMES}
    for row in rows.values():
        row["pareto"] = {
            "pareto_nondominated_K": [2],
            "pareto_knee_K": 2,
            "rows": [
                {
                    "K": 2,
                    "measured_operational_speedup_vs_K1": 10.0 / 11.0,
                    "conservative_speedup_lower_bound_vs_K1": 0.9,
                }
            ],
        }
    verdict = probe._admission(rows)
    assert verdict["status"] == "NO-GO"
    assert verdict["selected_K"] == 2
    assert all(speed < 1.0 for speed in verdict["measured_operational_speedup_vs_K1_by_regime"].values())


def test_pareto_excludes_underexercised_and_failed_control_arms():
    arms = _regime(ks=(1, 2, 4))["arms"]
    arms[1]["controls"]["status"] = "FAIL"
    arms[2]["steps"] = arms[2]["steps"][:3]
    result = probe._pareto_knee(arms)
    assert result["pareto_nondominated_K"] == [1]
    assert result["excluded_control_failure_K"] == [2]
    assert result["excluded_underexercised_K"] == [4]


def test_incomplete_k4_horizon_is_rejected_before_measurement():
    with pytest.raises(ValueError, match="K=4"):
        probe._require_complete_horizon(2)
    args = Namespace(steps=2, seed=1)
    with pytest.raises(ValueError, match="K=4"):
        probe._base_receipt(args)


def test_provider_rejection_automatically_selects_exact_teacher_fallback_and_binds_all_args():
    required = {
        "segnet",
        "current_frame",
        "bank_path",
        "expected_bank_sha256",
        "objective_context_fingerprint",
        "scorer_fingerprint",
        "current_step",
        "expected_split_identity_sha256",
        "expected_anchor_frame_sha256",
        "expected_source_step",
        "max_staleness_steps",
    }
    kwargs = {name: object() for name in required}

    def rejected_provider(**received):
        assert set(received) == required
        raise ValueError("changed bank fingerprint")

    exact = object()
    candidate, metadata, elapsed, fallback = probe._provider_or_full_teacher_fallback(
        provider=rejected_provider,
        provider_kwargs=kwargs,
        exact_costate=exact,
    )
    assert candidate is exact
    assert fallback is True
    assert metadata["selected_mode"] == "full_teacher_fallback"
    assert "changed bank fingerprint" in metadata["provider_failure"]
    assert elapsed >= 0.0

    candidate, metadata, elapsed, fallback = probe._provider_or_full_teacher_fallback(
        provider=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider must not run")),
        provider_kwargs=None,
        exact_costate=exact,
        preflight_failure="ValueError: live objective custody changed",
    )
    assert candidate is exact
    assert fallback is True
    assert elapsed == 0.0
    assert "objective custody changed" in metadata["provider_failure"]

    candidate, metadata, _elapsed, fallback = probe._provider_or_full_teacher_fallback(
        provider=lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError("bank vanished after is_file")),
        provider_kwargs=kwargs,
        exact_costate=exact,
    )
    assert candidate is exact
    assert fallback is True
    assert "bank vanished after is_file" in metadata["provider_failure"]


def test_pareto_timing_lower_bound_contains_skewed_observed_support():
    arms = _regime(ks=(1, 2))["arms"]
    for step, seconds in zip(arms[0]["steps"], [10.0, 10.0, 10.0, 1.0], strict=True):
        step["timing_measured_seconds"]["operational_cycle"] = seconds
    for step, seconds in zip(arms[1]["steps"], [1.0, 1.0, 1.0, 2.0], strict=True):
        step["timing_measured_seconds"]["operational_cycle"] = seconds
    arms[0]["summary"]["mean_operational_cycle_seconds"] = 7.75
    arms[1]["summary"]["mean_operational_cycle_seconds"] = 1.25
    result = probe._pareto_knee(arms)
    k2 = next(row for row in result["rows"] if row["K"] == 2)
    assert k2["conservative_speedup_lower_bound_vs_K1"] <= 0.5
    assert k2["K1_observed_timing_support_seconds"][0] <= 1.0
    assert k2["arm_observed_timing_support_seconds"][1] >= 2.0


def test_operational_wall_interval_owns_unattributed_overhead():
    timing = probe._operational_timing_record(
        step_started=10.0,
        operational_completed=15.0,
        component_seconds={"named_a": 1.0, "named_b": 2.0},
        path="test",
    )
    assert timing["wall_seconds"] == 5.0
    assert timing["component_sum_seconds_diagnostic_only"] == 3.0
    assert timing["unattributed_overhead_seconds"] == 2.0
    assert timing["authority"].startswith("wall interval")


def test_run_frees_operational_vjp_graph_and_rerenders_for_diagnostics():
    source = inspect.getsource(probe.run)
    assert "retain_graph=True" not in source
    assert "diagnostic_frame = _render_chart(renderer, theta)" in source
    assert "operational_timing = _operational_timing_record(" in source


def test_run_enforces_label_floor_on_every_comparison_and_successful_arm():
    source = inspect.getsource(probe.run)
    assert source.count("_require_teacher_label_path_agreement(") == 3
    assert source.count('step["controls"]["teacher_label_path_agreement"]["status"] == "PASS"') == 1
    assert "and label_path_floor_pass" in source
    assert (
        'template["measurement_canaries"]["same_frame_teacher_label_path_float32_floor"]["status"] != "PASS"'
        in source
    )
    assert "same-frame teacher label-path floor canary failed before measurement" in source
    assert '"status": "NOT_APPLICABLE_PROVIDER_FALLBACK"' in source
    assert "a rejected provider uses one operational exact-teacher label path" in source


def test_terminal_non_descent_receipt_keeps_fresh_metrics_and_teacher_work():
    source = inspect.getsource(probe.run)
    terminal = source.split("if target_norm is None or candidate_theta is None:", maxsplit=1)[1]
    terminal = terminal.split("selected_candidate = next", maxsplit=1)[0]
    for required in (
        '"costate_metrics_global"',
        '"costate_metrics_gt_boundary_annulus_bottom_k_0p05"',
        '"renderer_gradient_cosine"',
        '"teacher_work_counts"',
        '"operational_teacher_forward_backward_including_labels"',
        '"operational_validation_forwards_including_labels"',
        '"algebraic_speed_ceiling_derived"',
        '"regret_status": "UNKNOWN_NO_ADMISSIBLE_DESCENDING_CANDIDATE"',
        '"teacher_label_path_agreement"',
        '"terminal_stop_requires_no_parameter_update": True',
        '"full_teacher_fallbacks": int(provider_fallback)',
        '"candidate_ce": None',
        '"candidate_dseg": None',
        '"exact_reference_ce": None',
        '"exact_reference_dseg": None',
        '"ce_regret_vs_exact_reference": None',
        '"dseg_regret_vs_exact_reference": None',
        '"actual_probe_teacher_forward_backward_including_labels"',
        '"measurement_only_teacher_forward_backward_including_labels"',
        '"measurement_only_control_forwards_including_labels": 0',
        '"actual_probe_teacher_forwards_total_including_labels"',
    ):
        assert required in terminal
    assert "_metrics(exact_costate, candidate_costate)" in terminal
    assert "_cosine(\n                                exact_grad.detach().numpy(), candidate_grad.detach().numpy()" in terminal
    assert "1 + int(refresh)" not in source
    assert source.count("operational_teacher_forward_backward_count += 1") == 4
    assert source.count("measurement_only_teacher_forward_backward_count += 1") == 3
    assert source.count('"actual_probe_teacher_forward_backward_including_labels": (') == 2


@pytest.mark.parametrize("terminal_status", ["NO_GO_NON_DESCENT", "NO_GO_PROVIDER_FALLBACK"])
def test_last_step_terminal_arm_cannot_enter_success_finalizer(terminal_status):
    arm = _regime(ks=(4,))["arms"][0]
    for step in arm["steps"]:
        step["status"] = "MEASURED"
        step["provider_fallback"] = False
    arm["status"] = terminal_status
    arm["steps"][-1]["candidate_non_descent"] = terminal_status == "NO_GO_NON_DESCENT"
    arm["steps"][-1]["provider_fallback"] = terminal_status == "NO_GO_PROVIDER_FALLBACK"
    assert probe._arm_ready_for_success_finalization(arm, expected_steps=4) is False

    arm["status"] = "RUNNING"
    for step in arm["steps"]:
        step["provider_fallback"] = False
    assert probe._arm_ready_for_success_finalization(arm, expected_steps=4) is (terminal_status != "NO_GO_NON_DESCENT")


def test_live_decision_custody_comparison_is_fail_closed():
    receipt = {
        "inputs": {
            "gt_cache": {"sha256": "a" * 64},
            "segnet": {"sha256": "b" * 64},
            "checkpoints": {"early": {"sha256": "e" * 64}},
        },
        "source_custody": {
            "src/tac/cuda_levelset_training.py": {"sha256": "c" * 64},
            "src/tac/boundary_math/seg_core.py": {"sha256": "d" * 64},
        },
    }
    live = {
        "gt_sha256": "a" * 64,
        "segnet_sha256": "b" * 64,
        "checkpoint_sha256": "e" * 64,
        "source_sha256": {
            "src/tac/cuda_levelset_training.py": "c" * 64,
            "src/tac/boundary_math/seg_core.py": "d" * 64,
        },
    }
    assert probe._decision_custody_changed(receipt, "early", live) is False
    live["source_sha256"]["src/tac/boundary_math/seg_core.py"] = "f" * 64
    assert probe._decision_custody_changed(receipt, "early", live) is True


def test_fresh_teacher_timer_includes_bound_ce_and_dseg_labels():
    import torch

    logits = torch.tensor([[[[2.0]], [[-1.0]]]], requires_grad=True)
    labels = torch.zeros((1, 1, 1), dtype=torch.long)

    def capture(**kwargs):
        loss = kwargs["teacher_loss_fn"](logits)
        assert loss.requires_grad
        return "bank", torch.ones((1, 3, 1, 1))

    bank, _costate, holder, elapsed = probe._capture_labeled_teacher(
        capture=capture,
        segnet=object(),
        frame_nchw=torch.zeros((1, 3, 1, 1)),
        labels=labels,
        objective_context="a" * 64,
        scorer_fingerprint="b" * 64,
        step_index=0,
    )
    assert bank == "bank"
    assert holder["ce"] > 0.0
    assert holder["dseg"] == 0.0
    assert elapsed >= 0.0


def test_ordinary_exact_teacher_baseline_excludes_yopo_bank_capture():
    import torch

    class Scorer(torch.nn.Module):
        def forward(self, frame):
            return frame[:, :2]

    frame = torch.tensor([[[[2.0]], [[-1.0]], [[0.5]]]])
    labels = torch.zeros((1, 1, 1), dtype=torch.long)
    costate, holder, elapsed = probe._capture_exact_teacher_costate(
        segnet=Scorer().eval(), frame_nchw=frame, labels=labels
    )
    assert tuple(costate.shape) == tuple(frame.shape)
    assert holder["ce"] > 0.0
    assert holder["dseg"] == 0.0
    assert elapsed >= 0.0


def test_teacher_label_path_floor_accepts_measured_four_ulp_and_rejects_five():
    import numpy as np

    start = np.float32(1.0)
    values = [start]
    for _ in range(5):
        values.append(np.nextafter(values[-1], np.float32(np.inf), dtype=np.float32))
    within = probe._teacher_label_path_agreement(
        {"ce": float(start), "dseg": 0.25}, {"ce": float(values[4]), "dseg": 0.25}
    )
    outside = probe._teacher_label_path_agreement(
        {"ce": float(start), "dseg": 0.25}, {"ce": float(values[5]), "dseg": 0.25}
    )
    assert within["status"] == "PASS"
    assert within["ce_float32_ulp_distance"] == 4
    assert outside["status"] == "FAIL"
    assert outside["ce_float32_ulp_distance"] == 5
    assert probe._require_teacher_label_path_agreement(
        {"ce": float(start), "dseg": 0.25}, {"ce": float(values[4]), "dseg": 0.25}
    )["status"] == "PASS"
    with pytest.raises(RuntimeError, match="exceed registered numerical floor"):
        probe._require_teacher_label_path_agreement(
            {"ce": float(start), "dseg": 0.25}, {"ce": float(values[5]), "dseg": 0.25}
        )


def test_teacher_label_path_floor_never_relaxes_dseg_or_nonfinite_ce():
    dseg_mismatch = probe._teacher_label_path_agreement(
        {"ce": 1.0, "dseg": 0.25}, {"ce": 1.0, "dseg": 0.5}
    )
    nonfinite = probe._teacher_label_path_agreement(
        {"ce": 1.0, "dseg": 0.25}, {"ce": float("nan"), "dseg": 0.25}
    )
    nonfinite_dseg = probe._teacher_label_path_agreement(
        {"ce": 1.0, "dseg": float("inf")}, {"ce": 1.0, "dseg": float("inf")}
    )
    assert dseg_mismatch["status"] == "FAIL"
    assert dseg_mismatch["dseg_exact_match"] is False
    assert nonfinite["status"] == "FAIL"
    assert nonfinite["ce_float32_ulp_distance"] is None
    assert nonfinite_dseg["status"] == "FAIL"
    assert nonfinite_dseg["dseg_finite"] is False


def test_teacher_label_path_floor_meter_has_positive_and_negative_canaries():
    canary = probe._teacher_label_path_floor_canary()
    assert canary["status"] == "PASS"
    assert canary["positive_control_at_registered_floor"]["status"] == "PASS"
    assert canary["negative_control_one_ulp_beyond_floor"]["status"] == "FAIL"
    assert canary["negative_control_dseg_mismatch"]["status"] == "FAIL"
    assert canary["measured_anchor"]["status"] == "PASS"


def test_teacher_label_path_floor_canary_rejects_anchor_hash_drift(monkeypatch):
    monkeypatch.setitem(probe.SAME_FRAME_TEACHER_CE_PATH_FLOOR_ANCHOR, "artifact_sha256", "0" * 64)
    canary = probe._teacher_label_path_floor_canary()
    assert canary["status"] == "FAIL"
    assert canary["measured_anchor"]["status"] == "FAIL"


def test_candidate_recess_is_event_conditioned_and_has_completion_rule(monkeypatch):
    import torch

    theta = torch.zeros(2)
    gradient = torch.ones(2)
    monkeypatch.setattr(probe, "_render_chart", lambda _renderer, candidate: candidate)
    monkeypatch.setattr(probe, "_evaluate_teacher", lambda _segnet, _frame, _labels: (0.9, 0.0))
    candidate, norm, trials, forwards, elapsed = probe._select_candidate_recess(
        renderer=object(),
        theta=theta,
        candidate_grad=gradient,
        segnet=object(),
        labels=object(),
        current_loss=1.0,
        current_dseg=0.0,
    )
    assert candidate is not None
    assert norm == 0.01
    assert trials == [
        {
            "fraction_of_max_theta_norm": 0.01,
            "target_parameter_step_norm": 0.01,
            "candidate_ce": 0.9,
            "candidate_dseg": 0.0,
            "current_ce": 1.0,
            "current_dseg": 0.0,
            "accepted": True,
        }
    ]
    assert forwards == 1
    assert elapsed >= 0.0


def test_candidate_recess_counts_only_executed_teacher_validations(monkeypatch):
    import torch

    calls = 0

    def evaluate(_segnet, _frame, _labels):
        nonlocal calls
        calls += 1
        return 1.0, 0.0

    theta = torch.zeros(1, dtype=torch.float32)
    monkeypatch.setattr(probe, "_render_chart", lambda _renderer, candidate: candidate)
    monkeypatch.setattr(probe, "_evaluate_teacher", evaluate)
    candidate, norm, trials, forwards, elapsed = probe._select_candidate_recess(
        renderer=object(),
        theta=theta,
        candidate_grad=torch.ones_like(theta),
        segnet=object(),
        labels=object(),
        current_loss=1.0,
        current_dseg=0.0,
    )
    assert candidate is None
    assert norm is None
    assert trials[-1]["status"] == "BIT_IDENTICAL_TERMINATION"
    assert forwards == calls
    assert forwards == len(trials) - 1
    assert elapsed >= 0.0


def test_renderer_parity_canary_uses_canonical_receiver(monkeypatch):
    import numpy as np
    import torch

    from tac.cuda_levelset_training import contest_r

    bulk = np.zeros((384, 512, 3), np.float32)
    expected = contest_r(torch.as_tensor(bulk).unsqueeze(0))

    class Renderer:
        def __init__(self):
            self.code = [None, None]

        @staticmethod
        def render_pair(_pair):
            return bulk, None

    monkeypatch.setattr(probe, "_render_chart", lambda _renderer, _theta: expected.clone().requires_grad_(True))
    result = probe._renderer_parity_canary(Renderer(), torch.zeros(1, requires_grad=True))
    assert result["max_abs"] == 0.0
    assert result["different_elements"] == 0
    assert result["receiver"] == "tac.cuda_levelset_training.contest_r"


def test_self_orient_checkpoint_metadata_is_bound_but_not_used_as_a_false_blocker(tmp_path):
    import numpy as np

    checkpoint = tmp_path / "snapshot.npz"
    np.savez(checkpoint, __cfg_self_orient=np.asarray(1), __epoch=np.asarray(299))
    metadata = probe._checkpoint_metadata(checkpoint)
    assert metadata["cfg_self_orient"] == 1
    assert not hasattr(probe, "_exact_renderer_blocker")


def test_atomic_write_and_resume_payload_are_json_and_false_authority(tmp_path):
    path = tmp_path / "receipt.json"
    payload = {"authority": {"score_claim": False, "promotion_eligible": False}, "regimes": {}}
    probe._atomic_write(path, payload)
    assert probe._load_existing(path) == payload


def test_pending_atomic_state_promotes_one_unwritten_step_on_resume(tmp_path):
    import numpy as np

    old_bank = {"sha256": "a" * 64, "source_step": 0}
    new_bank = {"sha256": "b" * 64, "source_step": 4}
    arm = {"steps": [], "active_bank": old_bank}
    state = tmp_path / "state.npz"
    step = {"step": 0, "status": "MEASURED", "candidate_non_descent": False, "refresh": True}
    probe._stage_step_state(state, np.asarray([1.0, 2.0], np.float32), 1, step, new_bank)
    theta, promoted = probe._recover_pending_step(arm, state)
    assert promoted is True
    assert arm["steps"] == [step]
    assert arm["active_bank"] == new_bank
    assert np.array_equal(theta, np.asarray([1.0, 2.0], np.float32))
    probe._clear_staged_step_state(state, theta, len(arm["steps"]), arm["active_bank"])
    _theta, promoted_again = probe._recover_pending_step(arm, state)
    assert promoted_again is False


@pytest.mark.parametrize(
    ("provider_fallback", "candidate_non_descent", "expected_status"),
    [
        (True, False, "NO_GO_PROVIDER_FALLBACK"),
        (False, True, "NO_GO_NON_DESCENT"),
        (True, True, "NO_GO_PROVIDER_FALLBACK"),
    ],
)
@pytest.mark.parametrize("terminal_step", [2, 3])
def test_pending_terminal_step_restores_exact_no_go_status_on_resume(
    tmp_path, provider_fallback, candidate_non_descent, expected_status, terminal_step
):
    import numpy as np

    active_bank = {"sha256": "a" * 64, "source_step": 0, "rebuild": "test"}
    arm = {
        "status": "RUNNING",
        "steps": [{"step": step, "status": "MEASURED"} for step in range(terminal_step)],
        "active_bank": active_bank,
    }
    pending = {
        "step": terminal_step,
        "status": "MEASURED",
        "provider_fallback": provider_fallback,
        "candidate_non_descent": candidate_non_descent,
    }
    assert probe._terminal_status_for_step(pending) == expected_status
    state = tmp_path / "state.npz"
    probe._stage_step_state(
        state,
        np.asarray([1.0], np.float32),
        terminal_step + 1,
        pending,
        active_bank,
    )
    _theta, promoted = probe._recover_pending_step(arm, state)
    assert promoted is True
    assert arm["status"] == expected_status
    assert arm["steps"][-1] == pending


def test_cleanup_certificate_is_durable_before_unlink_and_resume_reconciles(tmp_path, monkeypatch):
    scratch = tmp_path / "bank.npz"
    scratch.write_bytes(b"content-addressed-bank")
    receipt_path = tmp_path / "receipt.json"
    rebuild = "fresh exact teacher plus recorded code-row state"
    arm = {
        "active_bank": {"path": str(scratch), "sha256": probe._sha256(scratch), "rebuild": rebuild},
        "cleanup": [],
    }
    receipt = {
        "authority": {"score_claim": False},
        "config": {"seed": 1},
        "runtime_provenance": {"argv": ["probe"]},
        "inputs": {},
        "source_custody": {},
        "arm": arm,
    }
    real_atomic_write = probe._atomic_write
    calls = 0

    def fail_second_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated crash after unlink")
        real_atomic_write(path, payload)

    monkeypatch.setattr(probe, "_atomic_write", fail_second_write)
    with pytest.raises(RuntimeError, match="simulated crash"):
        probe._certify_then_remove_scratch(
            receipt_path=receipt_path,
            receipt=receipt,
            arm=arm,
            scratch_path=scratch,
            reason="test cleanup",
        )
    assert not scratch.exists()
    durable = probe._load_existing(receipt_path)
    assert durable is not None
    durable_record = durable["arm"]["cleanup"][0]
    assert durable_record["status"] == "CERTIFIED_REBUILDABLE_SCRATCH_PENDING_REMOVAL"
    assert durable_record["sha256"]
    assert durable_record["bytes"] == len(b"content-addressed-bank")

    monkeypatch.setattr(probe, "_atomic_write", real_atomic_write)
    probe._certify_then_remove_scratch(
        receipt_path=receipt_path,
        receipt=durable,
        arm=durable["arm"],
        scratch_path=scratch,
        reason="test cleanup",
    )
    reconciled = probe._load_existing(receipt_path)
    assert reconciled["arm"]["cleanup"][0]["status"] == "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED"


def test_cleanup_without_rebuild_proof_keeps_bytes(tmp_path):
    scratch = tmp_path / "bank.npz"
    scratch.write_bytes(b"keep-me")
    receipt = {"arm": {"active_bank": {}, "cleanup": []}}
    with pytest.raises(RuntimeError, match="rebuild proof"):
        probe._certify_then_remove_scratch(
            receipt_path=tmp_path / "receipt.json",
            receipt=receipt,
            arm=receipt["arm"],
            scratch_path=scratch,
            reason="test cleanup",
        )
    assert scratch.read_bytes() == b"keep-me"


def test_resumed_terminal_arm_reconciles_cleanup_before_skip(tmp_path):
    scratch = tmp_path / "bank.npz"
    scratch.write_bytes(b"terminal-bank")
    rebuild = "fresh exact teacher plus recorded code-row state"
    arm = {
        "status": "NO_GO_PROVIDER_FALLBACK",
        "active_bank": {"path": str(scratch), "sha256": probe._sha256(scratch), "rebuild": rebuild},
        "cleanup": [],
    }
    receipt = {
        "config": {},
        "runtime_provenance": {"argv": ["probe"]},
        "inputs": {},
        "source_custody": {},
        "arm": arm,
    }
    receipt_path = tmp_path / "receipt.json"
    assert probe._reconcile_terminal_arm_cleanup(
        receipt_path=receipt_path,
        receipt=receipt,
        arm=arm,
        scratch_path=scratch,
    )
    assert not scratch.exists()
    durable = probe._load_existing(receipt_path)
    assert durable["arm"]["status"] == "NO_GO_PROVIDER_FALLBACK"
    assert durable["arm"]["active_bank"] is None
    assert durable["arm"]["cleanup"][0]["status"] == "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED"


def test_missing_active_scratch_without_cleanup_certificate_blocks_all_finalizers(tmp_path):
    scratch = tmp_path / "missing-bank.npz"
    arm = {
        "status": "NO_GO_NON_DESCENT",
        "active_bank": {
            "path": str(scratch),
            "sha256": "a" * 64,
            "rebuild": "fresh exact teacher plus recorded code-row state",
        },
        "cleanup": [],
    }
    with pytest.raises(RuntimeError, match="missing without a durable cleanup certificate"):
        probe._require_active_scratch_custody(arm, scratch)
    with pytest.raises(RuntimeError, match="missing without a durable cleanup certificate"):
        probe._reconcile_terminal_arm_cleanup(
            receipt_path=tmp_path / "receipt.json",
            receipt={"arm": arm},
            arm=arm,
            scratch_path=scratch,
        )
    assert arm["active_bank"] is not None
    assert arm["cleanup"] == []

    partial = {"path": str(scratch), "status": "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED"}
    arm["cleanup"].append(partial)
    with pytest.raises(RuntimeError, match="schema is incomplete"):
        probe._require_active_scratch_custody(arm, scratch)
    arm["cleanup"][0] = {
        "status": "CERTIFIED_REBUILDABLE_SCRATCH_REMOVED",
        "path": str(scratch),
        "bytes": 123,
        "sha256": "a" * 64,
        "rebuild": "fresh exact teacher plus recorded code-row state",
        "reason": "certified test removal",
        "cold_store_destination": None,
        "reproducibility_context": probe._CLEANUP_REPRODUCIBILITY_CONTEXT,
        "false_authority": {"score_claim": False, "promotion_eligible": False},
    }
    assert probe._require_active_scratch_custody(arm, scratch) == (False, True)
    arm["cleanup"][0]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="does not match the active bank custody"):
        probe._require_active_scratch_custody(arm, scratch)


def test_resume_rejects_receipt_bank_metadata_that_differs_from_cleared_state(tmp_path):
    import numpy as np

    state = tmp_path / "state.npz"
    bank = {"sha256": "a" * 64, "source_step": 0}
    arm = {"steps": [], "active_bank": bank}
    probe._clear_staged_step_state(state, np.asarray([1.0], np.float32), 0, bank)
    arm["active_bank"] = {"sha256": "b" * 64, "source_step": 0}
    with pytest.raises(RuntimeError, match="active-bank metadata"):
        probe._recover_pending_step(arm, state)


def test_single_writer_lock_fails_closed_until_owner_releases(tmp_path):
    fd = probe._acquire_output_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="active probe writer"):
            probe._acquire_output_lock(tmp_path)
    finally:
        probe._release_output_lock(fd)


def test_resume_custody_rejects_changed_source_and_terminal_receipt():
    stable_runtime = {
        "python": "3.12.0",
        "platform": "macOS-test",
        "machine": "arm64",
        "deterministic_algorithms": True,
        "git_head": "a" * 40,
        "numpy": "2.0.0",
        "torch": "2.5.0",
        "torch_num_threads": 4,
    }
    template = {
        "schema": "schema-v1",
        "authority": {"score_claim": False, "promotion_eligible": False},
        "config": {"seed": 1},
        "runtime_provenance": {"argv": ["current", "--resume"], **stable_runtime},
        "inputs": {"video": {"sha256": "a"}},
        "source_custody": {"probe": {"sha256": "b"}},
        "split": {"name": "block0"},
        "objective": "frozen SegNet CE",
        "measurement_canaries": {"same_frame_teacher_label_path_float32_floor": {"status": "PASS"}},
    }
    receipt = deepcopy(template)
    receipt["runtime_provenance"]["argv"] = ["original"]
    probe._validate_resume_custody(receipt, template)
    for key in (
        "schema",
        "authority",
        "config",
        "inputs",
        "source_custody",
        "split",
        "objective",
        "measurement_canaries",
    ):
        changed = deepcopy(receipt)
        changed[key] = "changed"
        with pytest.raises(RuntimeError, match="immutable field"):
            probe._validate_resume_custody(changed, template)
    for key in stable_runtime:
        changed = deepcopy(receipt)
        changed["runtime_provenance"][key] = "changed"
        with pytest.raises(RuntimeError, match="runtime provenance"):
            probe._validate_resume_custody(changed, template)
    terminal = deepcopy(receipt)
    terminal["completed_at_utc"] = "2026-07-12T00:00:00Z"
    with pytest.raises(RuntimeError, match="terminal"):
        probe._validate_resume_custody(terminal, template)


def test_resume_rederives_regime_checkpoint_and_objective_metadata():
    checkpoint_metadata = {"epoch": 300, "cfg_softmax_temp": 0.1}
    objective_metadata = {"loss_name": "frozen_segnet_ce", "regime": "early"}
    row = {
        "checkpoint_metadata": deepcopy(checkpoint_metadata),
        "objective_metadata": deepcopy(objective_metadata),
    }
    probe._validate_regime_resume_custody(
        row,
        checkpoint_metadata=checkpoint_metadata,
        objective_metadata=objective_metadata,
    )
    for key in ("checkpoint_metadata", "objective_metadata"):
        changed = deepcopy(row)
        changed[key]["tampered"] = True
        with pytest.raises(RuntimeError, match="differs"):
            probe._validate_regime_resume_custody(
                changed,
                checkpoint_metadata=checkpoint_metadata,
                objective_metadata=objective_metadata,
            )
