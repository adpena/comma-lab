# SPDX-License-Identifier: MIT
"""#507 composed C1 optimal-form config — consumed-proof + refusal tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.witness_dsl.spec_c1_optimal_form_20260715 import (
    C1_CURVELET_SLOT_LEVER,
    C1_DEEP_MATH_SLOTS,
    C1_OPTIMAL_FORM_EXPECTED_ADDITIONS,
    C1_SPEED_DISPOSITIONS,
    DEFAULT_OUT_DIR,
    PROGRAM_NAME,
    compile_c1_optimal_form_launch_config,
)


def _pairs(cfg) -> dict:
    from tac.witness_autoconfig import _crucible_v7_argv_pairs

    return dict(_crucible_v7_argv_pairs(tuple(cfg.typed.to_program().compile_trainer_argv())))


@pytest.fixture(scope="module")
def composed():
    return compile_c1_optimal_form_launch_config()


@pytest.fixture(scope="module")
def parent():
    from tac.witness_dsl.spec_v9_cgauge import (
        compile_v9_cgauge_ideal_mod19_sR_launch_config,
    )

    return compile_v9_cgauge_ideal_mod19_sR_launch_config(out_dir=DEFAULT_OUT_DIR)


def test_composed_actuation_carries_every_leg(composed) -> None:
    assert composed.name == PROGRAM_NAME
    pairs = _pairs(composed)
    # Leg A — the S_R treatment survives composition and pins the serial route.
    assert "--margin-saliency-reachability" in pairs
    assert pairs["--micro-batch-pairs"] == "1"
    assert pairs["--margin-saliency-weight"] == "1.0"
    assert pairs["--mod-dim"] == "19"
    # Leg B — the parent-carried speed core + the folded telemetry lever.
    for flag in ("--fused-r-kernel", "--cache-gt-skeleton", "--async-verdict",
                 "--component-wallclock-telemetry", "--profile-timing"):
        assert flag in pairs, flag
    assert pairs["--verdict-batch"] == "32"
    assert pairs["--verdict-pairs"] == "0"
    assert pairs["--safe-compile-regions"] == "hosc_activation"
    assert pairs["--component-wallclock-probe-every"] == "1"
    # Leg C — the consumable deep-math folds.
    assert "--pose-training-compute-gate" in pairs
    assert "--verdict-pose-gate" in pairs
    assert pairs["--head-offset-solver"] == "flip_median"
    assert pairs["--head-offset-solver-tau"] == "1.0"


def test_expected_levers_and_manifest_records(composed, parent) -> None:
    got = sorted(lv.name for lv in composed.typed.levers)
    want = sorted(tuple(lv.name for lv in parent.typed.levers)
                  + C1_OPTIMAL_FORM_EXPECTED_ADDITIONS)
    assert got == want
    m = composed.dsl_program_manifest
    assert m["held"] is True
    assert m["operator_go_required"] is True
    assert m["launch_blockers"], "the composed-path bench + sR custody blockers must be visible"
    rec = m["sr_microbatch_reconciliation"]
    assert rec["sr_requires_micro_batch_pairs"] == 1
    assert "batched LEVER-4 twin" in rec["cite"]
    slots = m["deep_math_slots"]
    assert set(slots) == set(C1_DEEP_MATH_SLOTS)
    assert slots["bregman_504"]["status"] == "SLOT_NO_TRAINER_CONSUMER"
    assert slots["fisher_natural_trust_region"]["status"] == "SLOT_BUILT_NOT_ACTIVATED"
    assert slots["hessian_preconditioned_423"]["status"] == "SLOT_NOT_ARGV_REACHABLE"
    assert slots["curvelet_basis"]["status"] == "SLOT_OPTIMAL_FORM_RECEIPT_OWED"
    assert m["speed_dispositions"] == C1_SPEED_DISPOSITIONS
    assert (m["speed_dispositions"]["custom_grouped_backward_vjp"]["status"]
            == "ON_VIA_PERF_ENV")


def test_delta_vs_parent_is_exactly_the_addition_flags(composed, parent) -> None:
    absent = "<ABSENT>"
    ppairs, cpairs = _pairs(parent), _pairs(composed)
    diff = {
        flag: [ppairs.get(flag, absent), cpairs.get(flag, absent)]
        for flag in sorted(set(ppairs) | set(cpairs))
        if flag != "--out-dir" and ppairs.get(flag, absent) != cpairs.get(flag, absent)
    }
    assert diff == composed.dsl_program_manifest["composition_contract"]["argv_delta_vs_parent"]
    addition_flags = set()
    for lever in composed.typed.levers:
        if lever.name in C1_OPTIMAL_FORM_EXPECTED_ADDITIONS:
            addition_flags |= set(lever.overrides)
    assert set(diff) <= addition_flags
    # every addition flag reached argv (consumed-not-inert, #417)
    assert addition_flags <= set(cpairs)


def test_curvelet_slot_refuses_missing_receipt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="curvelet-slot REFUSE"):
        compile_c1_optimal_form_launch_config(
            curvelet_optimal_form_receipt=tmp_path / "does_not_exist.json")


def test_curvelet_slot_folds_with_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "curvelet_optimal_form_receipt.json"
    receipt.write_text(json.dumps({"arm": "curvelet_optimal_form_crux",
                                   "status": "OPTIMAL_FORM_LANDED"}))
    cfg = compile_c1_optimal_form_launch_config(curvelet_optimal_form_receipt=receipt)
    assert C1_CURVELET_SLOT_LEVER in {lv.name for lv in cfg.typed.levers}
    pairs = _pairs(cfg)
    assert pairs["--basis"] == "windowed_curvelet"
    rec = cfg.dsl_program_manifest["deep_math_slots"]["curvelet_basis"]
    assert rec["status"] == "FOLDED_WITH_RECEIPT"
    assert rec["receipt_path"] == str(receipt)
    assert len(rec["receipt_sha256"]) == 64


def test_sr_microbatch_guard_refuses_parent_drift(monkeypatch, parent) -> None:
    """If the parent ever drifts to micro-batch>1 while S_R is emitted, the compile
    must refuse (mirrors the trainer's own fail-close)."""
    doctored = parent._rebind_typed(parent.typed.model_copy(update={
        "base": {**parent.typed.base, "--micro-batch-pairs": 4},
    }))
    from tac.witness_dsl import spec_v9_cgauge

    monkeypatch.setattr(
        spec_v9_cgauge, "compile_v9_cgauge_ideal_mod19_sR_launch_config",
        lambda **kwargs: doctored)
    with pytest.raises(ValueError, match="S_R/micro-batch REFUSE"):
        compile_c1_optimal_form_launch_config()
