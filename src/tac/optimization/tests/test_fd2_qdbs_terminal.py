from __future__ import annotations

import json
import struct
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

import tac.optimization.fd2_qdbs_terminal as qdbs_module
from tac.optimization.fd2_qdbs_terminal import (
    FULL_N600_AUTHORITY_MARKER,
    STALE_REHEARSAL_AUTHORITY_MARKER,
    ConsumedDescriptionCandidate,
    ContestAxis,
    DescriptionDelta,
    DescriptionHardOracleCallbacks,
    DescriptionProposal,
    FD2QDBSError,
    ParsedDescriptionCandidate,
    ProductionCustody,
    ProposalClass,
    QDBSAuthorityMode,
    QDBSStatus,
    RealizedJointAction,
    precommit_qdbs_schedule,
    run_fd2_qdbs_terminal,
)

_MAGIC = b"QDBSTST1"
_DEFAULT = object()


def _active_indices() -> tuple[int, ...]:
    return tuple(range(32)) + tuple(range(48, 64))


def _base_theta() -> np.ndarray:
    base = np.zeros(64, dtype=np.int64)
    base[np.asarray(_active_indices(), dtype=np.int64)] = 1
    return base


def _archive_payload(theta: np.ndarray) -> bytes:
    return _MAGIC + struct.pack(">I", theta.size) + np.asarray(theta, dtype=">i8").tobytes()


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _custody(
    base: np.ndarray,
    *,
    parent_archive_sha256: str | None = None,
) -> ProductionCustody:
    parent = _archive_payload(base)
    return ProductionCustody(
        parent_checkpoint_sha256=_digest("parent checkpoint"),
        parent_archive_sha256=(sha256(parent).hexdigest() if parent_archive_sha256 is None else parent_archive_sha256),
        parent_archive_bytes=len(parent),
        compiler_sha256=_digest("compiler"),
        receiver_sha256=_digest("receiver"),
        evaluator_sha256=_digest("evaluator"),
        evaluation_protocol_sha256=_digest("durable evaluator protocol v1"),
        upstream_sha256=_digest("upstream"),
        n_pairs=600,
        axis=ContestAxis.CONTEST_CPU,
        command="python upstream/evaluate.py submission/",
        hardware="synthetic test fixture; never promotion authority",
    )


def _proposals() -> tuple[tuple[DescriptionProposal, ...], tuple[DescriptionProposal, ...]]:
    singletons = tuple(
        DescriptionProposal(
            identity=f"scorer_singleton_{index:02d}",
            proposal_class=ProposalClass.SCORER_SINGLETON,
            deltas=(DescriptionDelta(index, -1),),
            signal_label="frozen_scorer_rank",
            signal_value=float(100 - index),
        )
        for index in range(16)
    )
    groups = tuple(
        DescriptionProposal(
            identity=f"scorer_group_{group:02d}",
            proposal_class=ProposalClass.SCORER_GROUP,
            deltas=(
                DescriptionDelta(16 + 2 * group, -1),
                DescriptionDelta(17 + 2 * group, -1),
            ),
            signal_label="frozen_scorer_block_rank",
            signal_value=float(80 - group),
        )
        for group in range(8)
    )
    return singletons, groups


def _callbacks(
    *,
    production: bool,
    custody: ProductionCustody | None = None,
    target: np.ndarray | None = None,
    counters: dict[str, int] | None = None,
    corrupt_parseback: bool = False,
    action_archive_sha256: object = _DEFAULT,
    action_archive_bytes_delta: int = 0,
    action_custody_digest: object = _DEFAULT,
    action_evaluation_key: object = _DEFAULT,
    fail_after: int | None = None,
    compiler_sha256: str | None = None,
    receiver_sha256: str | None = None,
    evaluator_sha256: str | None = None,
    evaluation_protocol_sha256: str | None = None,
    compiled_thetas: list[np.ndarray] | None = None,
    durable_actions: dict[str, RealizedJointAction] | None = None,
    fresh_by_key: dict[str, int] | None = None,
) -> DescriptionHardOracleCallbacks:
    target_array = np.zeros(64, dtype=np.int64) if target is None else target
    counts = counters if counters is not None else {}
    action_cache = durable_actions if durable_actions is not None else {}
    fresh_counts = fresh_by_key if fresh_by_key is not None else {}

    def compile_archive(theta: np.ndarray, _proposal: DescriptionProposal | None) -> bytes:
        counts["compile"] = counts.get("compile", 0) + 1
        if compiled_thetas is not None:
            compiled_thetas.append(np.array(theta, dtype=np.int64, copy=True))
        return _archive_payload(theta)

    def parse_archive(payload: bytes) -> ParsedDescriptionCandidate:
        counts["parse"] = counts.get("parse", 0) + 1
        assert payload[:8] == _MAGIC
        size = struct.unpack(">I", payload[8:12])[0]
        theta = np.frombuffer(payload[12:], dtype=">i8").astype(np.int64)
        assert theta.size == size
        if corrupt_parseback:
            theta = theta.copy()
            theta[0] += 1
        return ParsedDescriptionCandidate(
            realized_theta=theta,
            archive_sha256=sha256(payload).hexdigest(),
            exact_parseback=True,
            value=payload,
        )

    def consume_archive(parsed: ParsedDescriptionCandidate) -> ConsumedDescriptionCandidate:
        counts["consume"] = counts.get("consume", 0) + 1
        return ConsumedDescriptionCandidate(
            realized_theta=parsed.realized_theta,
            archive_sha256=parsed.archive_sha256,
            exact_consumption=True,
            value=parsed.value,
        )

    def evaluate(
        consumed: ConsumedDescriptionCandidate,
        evaluation_idempotency_key: str,
    ) -> RealizedJointAction:
        counts["evaluate"] = counts.get("evaluate", 0) + 1
        cached = action_cache.get(evaluation_idempotency_key)
        if cached is not None:
            return cached
        if fail_after is not None and counts["evaluate"] > fail_after:
            raise RuntimeError("injected interruption")
        error = np.asarray(consumed.realized_theta, dtype=np.float64) - target_array
        payload = consumed.value
        assert isinstance(payload, bytes)
        archive_sha = consumed.archive_sha256 if action_archive_sha256 is _DEFAULT else action_archive_sha256
        custody_digest = custody.digest if production and custody is not None else None
        if action_custody_digest is not _DEFAULT:
            custody_digest = action_custody_digest
        action = RealizedJointAction(
            d_seg=float(np.mean(error * error)) / 1000.0,
            d_pose=0.01,
            archive_sha256=archive_sha,
            archive_bytes=len(payload) + action_archive_bytes_delta,
            sample_count=600 if production else 1,
            authority_marker=(FULL_N600_AUTHORITY_MARKER if production else STALE_REHEARSAL_AUTHORITY_MARKER),
            custody_digest=custody_digest,
            evaluation_idempotency_key=(
                evaluation_idempotency_key if action_evaluation_key is _DEFAULT else action_evaluation_key
            ),
            realized=True,
        )
        fresh_counts[evaluation_idempotency_key] = fresh_counts.get(evaluation_idempotency_key, 0) + 1
        action_cache[evaluation_idempotency_key] = action
        return action

    return DescriptionHardOracleCallbacks(
        compile_archive=compile_archive,
        parse_archive=parse_archive,
        consume_archive=consume_archive,
        evaluate_joint_action_idempotent=evaluate,
        compiler_sha256=(
            compiler_sha256
            if compiler_sha256 is not None
            else custody.compiler_sha256
            if custody is not None
            else _digest("compiler")
        ),
        receiver_sha256=(
            receiver_sha256
            if receiver_sha256 is not None
            else custody.receiver_sha256
            if custody is not None
            else _digest("receiver")
        ),
        evaluator_sha256=(
            evaluator_sha256
            if evaluator_sha256 is not None
            else custody.evaluator_sha256
            if custody is not None
            else _digest("evaluator")
        ),
        evaluation_protocol_sha256=(
            evaluation_protocol_sha256
            if evaluation_protocol_sha256 is not None
            else custody.evaluation_protocol_sha256
            if custody is not None
            else _digest("durable evaluator protocol v1")
        ),
    )


def _run_production(
    tmp_path: Path,
    *,
    seed: int,
    custody: ProductionCustody,
    callbacks: DescriptionHardOracleCallbacks,
    ledger_name: str = "qdbs_resume.json",
):
    singletons, groups = _proposals()
    return run_fd2_qdbs_terminal(
        _base_theta(),
        singletons,
        groups,
        callbacks,
        active_indices=_active_indices(),
        seed=seed,
        authority_mode=QDBSAuthorityMode.PRODUCTION_FULL_N600,
        production_custody=custody,
        resume_ledger_path=tmp_path / ledger_name,
    )


def test_schedule_is_exact_count_matched_active_and_deterministic() -> None:
    singletons, groups = _proposals()
    first = precommit_qdbs_schedule(
        singletons,
        groups,
        coordinate_count=64,
        active_indices=_active_indices(),
        seed=8128,
    )
    second = precommit_qdbs_schedule(
        singletons,
        groups,
        coordinate_count=64,
        active_indices=_active_indices(),
        seed=8128,
    )
    assert first.schedule_sha256 == second.schedule_sha256
    assert first.random_controls == second.random_controls
    assert len(first.scorer_proposals) == 24
    assert len(first.random_controls) == 24
    assert len(first.candidates) == 48
    active = frozenset(_active_indices())
    for scorer, control in zip(first.scorer_proposals, first.random_controls, strict=True):
        assert control.matched_scorer_identity == scorer.identity
        assert len(control.deltas) == len(scorer.deltas)
        assert sorted(delta.delta for delta in control.deltas) == sorted(delta.delta for delta in scorer.deltas)
        assert all(delta.index in active for delta in scorer.deltas)
        assert all(delta.index in active for delta in control.deltas)


def test_active_domain_and_nonzero_inactive_coordinates_remain_frozen() -> None:
    singletons, groups = _proposals()
    with pytest.raises(FD2QDBSError, match="outside active_indices"):
        precommit_qdbs_schedule(
            singletons,
            groups,
            coordinate_count=64,
            active_indices=tuple(range(31)),
            seed=1,
        )
    nonzero_inactive = _base_theta()
    nonzero_inactive[32:48] = np.arange(16, dtype=np.int64) + 7
    compiled_thetas: list[np.ndarray] = []
    run_fd2_qdbs_terminal(
        nonzero_inactive,
        singletons,
        groups,
        _callbacks(production=False, compiled_thetas=compiled_thetas),
        active_indices=_active_indices(),
        seed=1,
        authority_mode=QDBSAuthorityMode.STALE_REHEARSAL,
    )
    assert len(compiled_thetas) == 49
    for compiled_theta in compiled_thetas:
        assert np.array_equal(compiled_theta[32:48], nonzero_inactive[32:48])


def test_rehearsal_runs_exact_budget_but_never_promotes() -> None:
    singletons, groups = _proposals()
    counters: dict[str, int] = {}
    result = run_fd2_qdbs_terminal(
        _base_theta(),
        singletons,
        groups,
        _callbacks(production=False, counters=counters),
        active_indices=_active_indices(),
        seed=7,
        authority_mode=QDBSAuthorityMode.STALE_REHEARSAL,
    )
    assert result.status is QDBSStatus.REHEARSAL_NONPROMOTABLE
    assert result.candidate_evaluations == 48
    assert result.shared_base_evaluations == 1
    assert result.best_strict_improvement_identity is not None
    assert result.governed_handoff_identity is None
    assert not result.governed_handoff_eligible
    assert not result.promotion_allowed
    assert result.resume_ledger_path is None
    assert counters == {"compile": 49, "parse": 49, "consume": 49, "evaluate": 49}


def test_full_n600_self_attested_custody_requires_external_governor(
    tmp_path: Path,
) -> None:
    base = _base_theta()
    custody = _custody(base)
    result = _run_production(
        tmp_path,
        seed=9,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody),
    )
    assert result.status is QDBSStatus.REQUIRES_EXTERNAL_GOVERNOR
    assert result.governed_handoff_identity is None
    assert result.best_strict_improvement_identity is not None
    winning = next(trace for trace in result.traces if trace.identity == result.best_strict_improvement_identity)
    assert winning.strict_realized_improvement
    assert not winning.governed_handoff_eligible
    assert winning.delta_vs_base < 0.0
    payload = result.to_payload()
    assert payload["production_custody"]["verification_status"] == ("SELF_ATTESTED_REQUIRES_EXTERNAL_GOVERNOR")
    assert payload["governed_handoff_eligible"] is False
    assert payload["outer_governor_required"] is True
    assert "does not independently verify" in payload["external_governor_blocker"]
    assert payload["promotion_allowed"] is False
    assert payload["score_claim"] is False
    assert payload["pointer_moved"] is False
    with pytest.raises(FD2QDBSError, match="cannot authorize governed handoff"):
        replace(result, governed_handoff_identity="self-authorized")


def test_no_strict_improvement_is_not_handoff_eligible(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    result = _run_production(
        tmp_path,
        seed=10,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody, target=base),
    )
    assert result.status is QDBSStatus.NO_STRICT_IMPROVEMENT
    assert result.governed_handoff_identity is None
    assert not result.promotion_allowed


def test_malformed_count_and_parseback_fail_closed() -> None:
    singletons, groups = _proposals()
    with pytest.raises(FD2QDBSError, match="exactly 16"):
        precommit_qdbs_schedule(
            singletons[:-1],
            groups,
            coordinate_count=64,
            active_indices=_active_indices(),
            seed=1,
        )
    with pytest.raises(FD2QDBSError, match="parse-back realized"):
        run_fd2_qdbs_terminal(
            _base_theta(),
            singletons,
            groups,
            _callbacks(production=False, corrupt_parseback=True),
            active_indices=_active_indices(),
            seed=1,
            authority_mode=QDBSAuthorityMode.STALE_REHEARSAL,
        )


def test_scorer_delta_sets_must_be_distinct_across_all_24_proposals() -> None:
    singletons, groups = _proposals()
    duplicate_singletons = list(singletons)
    duplicate_singletons[1] = replace(
        duplicate_singletons[1],
        deltas=duplicate_singletons[0].deltas,
    )
    with pytest.raises(FD2QDBSError, match="delta sets must be distinct"):
        precommit_qdbs_schedule(
            duplicate_singletons,
            groups,
            coordinate_count=64,
            active_indices=_active_indices(),
            seed=2,
        )
    duplicate_groups = list(groups)
    duplicate_groups[1] = replace(
        duplicate_groups[1],
        deltas=duplicate_groups[0].deltas,
    )
    with pytest.raises(FD2QDBSError, match="delta sets must be distinct"):
        precommit_qdbs_schedule(
            singletons,
            duplicate_groups,
            coordinate_count=64,
            active_indices=_active_indices(),
            seed=2,
        )


def test_production_requires_custody_resume_path_and_full_n600(tmp_path: Path) -> None:
    singletons, groups = _proposals()
    base = _base_theta()
    custody = _custody(base)
    with pytest.raises(FD2QDBSError, match="typed production custody"):
        run_fd2_qdbs_terminal(
            base,
            singletons,
            groups,
            _callbacks(production=True),
            active_indices=_active_indices(),
            seed=1,
            authority_mode=QDBSAuthorityMode.PRODUCTION_FULL_N600,
            resume_ledger_path=tmp_path / "missing_custody.json",
        )
    with pytest.raises(FD2QDBSError, match="resume ledger Path"):
        run_fd2_qdbs_terminal(
            base,
            singletons,
            groups,
            _callbacks(production=True, custody=custody),
            active_indices=_active_indices(),
            seed=1,
            authority_mode=QDBSAuthorityMode.PRODUCTION_FULL_N600,
            production_custody=custody,
        )
    with pytest.raises(FD2QDBSError, match="must be absolute"):
        run_fd2_qdbs_terminal(
            base,
            singletons,
            groups,
            _callbacks(production=True, custody=custody),
            active_indices=_active_indices(),
            seed=1,
            authority_mode=QDBSAuthorityMode.PRODUCTION_FULL_N600,
            production_custody=custody,
            resume_ledger_path=Path("relative/qdbs_resume.json"),
        )
    with pytest.raises(FD2QDBSError, match="full-n600"):
        _run_production(
            tmp_path,
            seed=1,
            custody=custody,
            callbacks=_callbacks(production=False),
            ledger_name="subset.json",
        )


@pytest.mark.parametrize(
    ("callback_kwargs", "message"),
    [
        ({"action_archive_sha256": "f" * 64}, "archive SHA differs"),
        ({"action_archive_bytes_delta": 1}, "byte count differs"),
        ({"action_custody_digest": "e" * 64}, "custody digest differs"),
        ({"action_evaluation_key": "d" * 64}, "idempotency key differs"),
    ],
)
def test_production_action_must_bind_archive_and_custody(
    tmp_path: Path,
    callback_kwargs: dict[str, object],
    message: str,
) -> None:
    base = _base_theta()
    custody = _custody(base)
    with pytest.raises(FD2QDBSError, match=message):
        _run_production(
            tmp_path,
            seed=3,
            custody=custody,
            callbacks=_callbacks(
                production=True,
                custody=custody,
                **callback_kwargs,
            ),
            ledger_name=f"{message.replace(' ', '_')}.json",
        )


def test_production_base_must_match_custody_parent(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base, parent_archive_sha256="0" * 64)
    ledger = tmp_path / "qdbs_resume.json"
    with pytest.raises(FD2QDBSError, match="custody parent"):
        _run_production(
            tmp_path,
            seed=4,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
        )
    assert not ledger.exists()


def test_resume_ledger_reuses_complete_prefix_without_callbacks(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    first_counts: dict[str, int] = {}
    first = _run_production(
        tmp_path,
        seed=11,
        custody=custody,
        callbacks=_callbacks(
            production=True,
            custody=custody,
            counters=first_counts,
        ),
    )
    assert first.resume_records_reused == 0
    assert first.resume_records_written == 49
    artifact_directory = tmp_path / "qdbs_resume.json.archives"
    assert artifact_directory.is_dir()
    assert len(tuple(artifact_directory.iterdir())) == 49
    assert first_counts == {
        "compile": 49,
        "parse": 49,
        "consume": 49,
        "evaluate": 49,
    }
    second_counts: dict[str, int] = {}
    second = _run_production(
        tmp_path,
        seed=11,
        custody=custody,
        callbacks=_callbacks(
            production=True,
            custody=custody,
            counters=second_counts,
        ),
    )
    assert second.resume_records_reused == 49
    assert second.resume_records_written == 0
    assert second_counts == {}
    assert second.to_payload()["traces"] == first.to_payload()["traces"]
    assert second.resume_ledger_sha256 == first.resume_ledger_sha256
    assert not second.promotion_allowed


def test_resume_rejects_foreign_callback_identity_before_cache_reuse(
    tmp_path: Path,
) -> None:
    base = _base_theta()
    custody = _custody(base)
    _run_production(
        tmp_path,
        seed=16,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody),
    )
    counters: dict[str, int] = {}
    with pytest.raises(FD2QDBSError, match="compiler_sha256 differs"):
        _run_production(
            tmp_path,
            seed=16,
            custody=custody,
            callbacks=_callbacks(
                production=True,
                custody=custody,
                counters=counters,
                compiler_sha256=_digest("foreign compiler"),
            ),
        )
    assert counters == {}
    with pytest.raises(FD2QDBSError, match="evaluation_protocol_sha256 differs"):
        _run_production(
            tmp_path,
            seed=16,
            custody=custody,
            callbacks=_callbacks(
                production=True,
                custody=custody,
                evaluation_protocol_sha256=_digest("foreign protocol"),
            ),
        )


def test_crash_after_evaluator_return_reuses_one_durable_idempotency_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_theta()
    custody = _custody(base)
    durable_actions: dict[str, RealizedJointAction] = {}
    fresh_by_key: dict[str, int] = {}
    counters: dict[str, int] = {}
    callbacks = _callbacks(
        production=True,
        custody=custody,
        counters=counters,
        durable_actions=durable_actions,
        fresh_by_key=fresh_by_key,
    )
    real_write = qdbs_module._atomic_write_ledger
    crashed = False

    def crash_before_complete_commit(
        path: Path,
        binding: dict[str, str],
        records: list[dict[str, object]],
    ) -> str:
        nonlocal crashed
        final_record = records[-1]
        if not crashed and final_record["ordinal"] == 4 and final_record["state"] == "COMPLETE":
            crashed = True
            raise RuntimeError("injected crash after evaluator return")
        return real_write(path, binding, records)

    monkeypatch.setattr(
        qdbs_module,
        "_atomic_write_ledger",
        crash_before_complete_commit,
    )
    with pytest.raises(RuntimeError, match="after evaluator return"):
        _run_production(
            tmp_path,
            seed=18,
            custody=custody,
            callbacks=callbacks,
        )
    ledger = json.loads((tmp_path / "qdbs_resume.json").read_text(encoding="ascii"))
    pending = ledger["records"][-1]
    assert pending["ordinal"] == 4
    assert pending["state"] == "PENDING_EVALUATION"
    assert pending["action"] is None
    pending_key = pending["evaluation_idempotency_key"]
    assert pending_key in durable_actions
    assert fresh_by_key[pending_key] == 1

    monkeypatch.setattr(qdbs_module, "_atomic_write_ledger", real_write)
    resumed = _run_production(
        tmp_path,
        seed=18,
        custody=custody,
        callbacks=callbacks,
    )
    assert resumed.status is QDBSStatus.REQUIRES_EXTERNAL_GOVERNOR
    assert len(durable_actions) == 49
    assert len(fresh_by_key) == 49
    assert set(fresh_by_key.values()) == {1}
    assert counters["evaluate"] == 50
    assert not resumed.governed_handoff_eligible
    assert not resumed.promotion_allowed


def test_resume_reopens_and_verifies_retained_archive_bytes(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    _run_production(
        tmp_path,
        seed=17,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody),
    )
    artifact_directory = tmp_path / "qdbs_resume.json.archives"
    artifact = sorted(artifact_directory.iterdir())[0]
    corrupted = bytearray(artifact.read_bytes())
    corrupted[-1] ^= 1
    artifact.write_bytes(corrupted)
    with pytest.raises(FD2QDBSError, match="artifact SHA differs"):
        _run_production(
            tmp_path,
            seed=17,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
        )


def test_resume_ledger_recovers_interrupted_prefix(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    ledger = tmp_path / "qdbs_resume.json"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _run_production(
            tmp_path,
            seed=12,
            custody=custody,
            callbacks=_callbacks(
                production=True,
                custody=custody,
                fail_after=10,
            ),
        )
    persisted = json.loads(ledger.read_text(encoding="ascii"))
    assert len(persisted["records"]) == 11
    assert persisted["records"][-1]["state"] == "PENDING_EVALUATION"
    assert persisted["records"][-1]["action"] is None
    resumed_counts: dict[str, int] = {}
    resumed = _run_production(
        tmp_path,
        seed=12,
        custody=custody,
        callbacks=_callbacks(
            production=True,
            custody=custody,
            counters=resumed_counts,
        ),
    )
    assert resumed.resume_records_reused == 10
    assert resumed.resume_records_written == 39
    assert resumed_counts == {
        "compile": 38,
        "parse": 39,
        "consume": 39,
        "evaluate": 39,
    }


def test_resume_ledger_corruption_and_binding_drift_fail_closed(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    _run_production(
        tmp_path,
        seed=13,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody),
    )
    ledger = tmp_path / "qdbs_resume.json"
    original = ledger.read_text(encoding="ascii")
    payload = json.loads(original)
    payload["records"][0]["identity"] = "tampered"
    ledger.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(FD2QDBSError, match="checksum differs"):
        _run_production(
            tmp_path,
            seed=13,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
        )
    ledger.write_text(original, encoding="ascii")
    with pytest.raises(FD2QDBSError, match="binding differs"):
        _run_production(
            tmp_path,
            seed=14,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
        )


def test_resume_ledger_symlink_fails_closed(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    _run_production(
        tmp_path,
        seed=15,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody),
        ledger_name="target.json",
    )
    alias = tmp_path / "alias.json"
    alias.symlink_to("target.json")
    with pytest.raises(FD2QDBSError, match="symlink"):
        _run_production(
            tmp_path,
            seed=15,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
            ledger_name="alias.json",
        )
    lock_alias = tmp_path / "lock_alias.json.lock"
    lock_alias.symlink_to("target.json")
    with pytest.raises(FD2QDBSError, match="symlink"):
        _run_production(
            tmp_path,
            seed=15,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
            ledger_name="lock_alias.json",
        )
    target_directory = tmp_path / "target_directory"
    target_directory.mkdir()
    alias_directory = tmp_path / "alias_directory"
    alias_directory.symlink_to(target_directory, target_is_directory=True)
    with pytest.raises(FD2QDBSError, match="symlink"):
        _run_production(
            alias_directory / "nested" / "resume",
            seed=15,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
        )


def test_resume_archive_artifact_symlink_fails_closed(tmp_path: Path) -> None:
    base = _base_theta()
    custody = _custody(base)
    _run_production(
        tmp_path,
        seed=18,
        custody=custody,
        callbacks=_callbacks(production=True, custody=custody),
    )
    artifact_directory = tmp_path / "qdbs_resume.json.archives"
    artifact = sorted(artifact_directory.iterdir())[0]
    backing = artifact.with_suffix(".backing")
    artifact.rename(backing)
    artifact.symlink_to(backing.name)
    with pytest.raises(FD2QDBSError, match="symlink"):
        _run_production(
            tmp_path,
            seed=18,
            custody=custody,
            callbacks=_callbacks(production=True, custody=custody),
        )
