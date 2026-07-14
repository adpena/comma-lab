import hashlib
import json
import shutil
import uuid
from pathlib import Path
from dataclasses import fields

import pytest

from tac.witness_control.exact_costate_reuse import (
    AnchorIdentity,
    ControllerState,
    GuardMetrics,
    Phase,
    ReuseAttemptIdentity,
    StepAction,
    evaluate_reuse_guard,
    force_refresh_boundary,
    plan_step,
    record_exact_anchor,
    record_full_teacher_refresh,
)

BASELINE = GuardMetrics(1.0, 0.2, 0.3)


@pytest.fixture
def durable_dir():
    path = Path.cwd() / ".pytest_artifacts" / f"exact-costate-reuse-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        parent = path.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()


def identity(seed: str = "a") -> AnchorIdentity:
    payload_path = Path(__file__)
    payload_sha256 = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    chars = [seed, "b", "c", "d", "e", "f"]
    return AnchorIdentity(
        payload_path=str(payload_path),
        payload_sha256=payload_sha256,
        frame_sha256=chars[1] * 64,
        costate_sha256=chars[2] * 64,
        objective_sha256=chars[3] * 64,
        scorer_sha256=chars[4] * 64,
        control_scope_sha256=chars[5] * 64,
    )


def attempt(anchor: AnchorIdentity, *, current_frame: str = "9", **overrides: str) -> ReuseAttemptIdentity:
    values = {
        "current_frame_sha256": current_frame * 64,
        "anchor_payload_sha256": anchor.payload_sha256,
        "anchor_frame_sha256": anchor.frame_sha256,
        "anchor_costate_sha256": anchor.costate_sha256,
        "objective_sha256": anchor.objective_sha256,
        "scorer_sha256": anchor.scorer_sha256,
        "control_scope_sha256": anchor.control_scope_sha256,
    }
    values.update(overrides)
    return ReuseAttemptIdentity(**values)


def test_changed_frame_can_attempt_reuse_without_current_exact_costate():
    anchor = identity()
    ready = record_exact_anchor(ControllerState(), anchor, BASELINE)
    planned = plan_step(ready, attempt(anchor))
    assert planned.action is StepAction.STALE_REUSE_ATTEMPT
    assert "current_costate_sha256" not in {field.name for field in fields(ReuseAttemptIdentity)}


def test_event_controlled_kmax_two_returns_to_exact_after_one_guarded_attempt():
    initial = ControllerState()
    assert plan_step(initial).action is StepAction.EXACT_ANCHOR
    ready = record_exact_anchor(initial, identity(), BASELINE)
    guarded = evaluate_reuse_guard(
        ready,
        candidate_metrics=GuardMetrics(0.9, 0.2, 0.29),
    )
    assert guarded.action is StepAction.EXACT_ANCHOR
    assert guarded.state.phase is Phase.NEEDS_EXACT_ANCHOR
    assert "K_max=2" in guarded.reason


def test_resume_round_trip_preserves_payload_custody_and_full_facet_metrics():
    initial = ControllerState(step_index=4)
    ready = record_exact_anchor(initial, identity(), BASELINE)
    payload = json.loads(json.dumps(ready.to_dict()))
    restored = ControllerState.from_dict(payload)
    assert restored == ready
    assert restored.anchor is not None
    assert restored.anchor.payload_path.endswith("test_exact_costate_reuse.py")
    assert restored.anchor_metrics == BASELINE


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/costate.npy",
        "/private/tmp/costate.npy",
        "/private/var/folders/test/costate.npy",
        "../costate.npy",
    ],
)
def test_anchor_payload_path_must_be_durable(path):
    values = identity().to_dict()
    values["payload_path"] = path
    with pytest.raises(ValueError):
        AnchorIdentity.from_dict(values)


def test_scope_or_anchor_hash_mismatch_fails_closed_and_latches_refresh():
    anchor = identity()
    ready = record_exact_anchor(ControllerState(), anchor, BASELINE)
    decision = plan_step(ready, attempt(anchor, control_scope_sha256="7" * 64))
    assert decision.action is StepAction.FULL_TEACHER_REFRESH
    assert decision.state.rollback_latched is True
    assert decision.state.anchor is None
    with pytest.raises(ValueError):
        record_exact_anchor(decision.state, identity(), BASELINE)
    refreshed = record_full_teacher_refresh(decision.state, identity(), BASELINE)
    assert refreshed.rollback_latched is False


def test_bit_identical_current_frame_fails_closed():
    anchor = identity()
    ready = record_exact_anchor(ControllerState(), anchor, BASELINE)
    decision = plan_step(ready, attempt(anchor, current_frame="b"))
    assert decision.action is StepAction.FULL_TEACHER_REFRESH
    assert "bit-identical" in decision.reason


@pytest.mark.parametrize(
    "candidate",
    [
        GuardMetrics(1.0, 0.2, 0.3),
        GuardMetrics(0.9, 0.21, 0.3),
        GuardMetrics(0.9, 0.2, 0.31),
    ],
)
def test_guard_rejects_non_strict_ce_or_dseg_dpose_regression(candidate):
    ready = record_exact_anchor(ControllerState(), identity(), BASELINE)
    decision = evaluate_reuse_guard(ready, candidate_metrics=candidate)
    assert decision.action is StepAction.FULL_TEACHER_REFRESH
    assert decision.state.rollback_latched is True


@pytest.mark.parametrize("boundary", ["event", "stage", "custody_change"])
def test_every_boundary_invalidates_anchor_metrics_and_payload(boundary):
    ready = record_exact_anchor(ControllerState(), identity(), BASELINE)
    decision = force_refresh_boundary(ready, boundary)
    assert decision.action is StepAction.FULL_TEACHER_REFRESH
    assert decision.state.anchor is None
    assert decision.state.anchor_metrics is None


def test_missing_payload_refuses_resume_and_latches_reuse(durable_dir):
    payload = durable_dir / "anchor.npy"
    payload.write_bytes(b"exact-costate")
    values = identity().to_dict()
    values["payload_path"] = str(payload)
    values["payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    anchor = AnchorIdentity.from_dict(values)
    ready = record_exact_anchor(ControllerState(), anchor, BASELINE)
    payload.unlink()
    with pytest.raises(ValueError, match="payload bytes are unavailable"):
        ControllerState.from_dict(json.loads(json.dumps(ready.to_dict())))
    decision = plan_step(ready, attempt(anchor))
    assert decision.action is StepAction.FULL_TEACHER_REFRESH
    assert decision.state.rollback_latched is True


def test_replaced_payload_refuses_record_and_reuse(durable_dir):
    payload = durable_dir / "anchor.npy"
    payload.write_bytes(b"anchor-a")
    values = identity().to_dict()
    values["payload_path"] = str(payload)
    values["payload_sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    anchor = AnchorIdentity.from_dict(values)
    payload.write_bytes(b"anchor-b")
    with pytest.raises(ValueError, match="payload sha256 mismatch"):
        record_exact_anchor(ControllerState(), anchor, BASELINE)
    ready_payload = durable_dir / "ready.npy"
    ready_payload.write_bytes(b"anchor-c")
    values["payload_path"] = str(ready_payload)
    values["payload_sha256"] = hashlib.sha256(ready_payload.read_bytes()).hexdigest()
    ready_anchor = AnchorIdentity.from_dict(values)
    ready = record_exact_anchor(ControllerState(), ready_anchor, BASELINE)
    ready_payload.write_bytes(b"anchor-d")
    decision = plan_step(ready, attempt(ready_anchor))
    assert decision.action is StepAction.FULL_TEACHER_REFRESH
    assert "payload custody failed" in decision.reason
