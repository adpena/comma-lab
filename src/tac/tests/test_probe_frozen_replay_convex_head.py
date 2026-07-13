# SPDX-License-Identifier: MIT
"""No-heavy regressions for the resumable frozen-replay probe apparatus."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import probe_frozen_replay_convex_head as probe


def _fake_torch() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        set_num_threads=lambda _count: None,
        set_num_interop_threads=lambda _count: None,
        manual_seed=lambda _seed: None,
        use_deterministic_algorithms=lambda _enabled: None,
    )


def _file_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _teacher_event_rows(
    assignments: tuple[probe.ReplayAssignment, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        batch_id = f"state-{assignment.pair_index:04d}"
        common = {
            "stage": "train_cache" if assignment.split == "train" else "heldout_validation",
            "batch_id": batch_id,
            "split": assignment.split,
            "pair_index": assignment.pair_index,
        }
        rows.extend(
            (
                {"event": "exact_teacher_state_call_started", **common},
                {"event": "exact_teacher_state_call_completed", **common},
            )
        )
    return rows


def _sealed_source_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for index, relative in enumerate(probe.SOURCE_FILES, start=1):
        rows[relative] = {
            "path": f"source_bundle/{relative}",
            "bytes": index,
            "sha256": probe.SOURCE_AMENDMENT_OLD_SHA256.get(
                relative, hashlib.sha256(relative.encode()).hexdigest()
            ),
        }
    return rows


def _changed_fingerprint(
    fingerprints: dict[str, dict[str, Any]], relative: str
) -> None:
    fingerprints[relative] = {
        "bytes": int(fingerprints[relative]["bytes"]) + 1,
        "sha256": hashlib.sha256(f"changed:{relative}".encode()).hexdigest(),
    }


def _write_recovery_boundary(
    output_dir: Path, *, pair_count: int
) -> tuple[dict[int, Path], dict[int, Path]]:
    record_paths: dict[int, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for pair_index in range(pair_count):
        path = output_dir / "train_cache" / f"pair_{pair_index:04d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed-record-{pair_index}".encode())
        record_paths[pair_index] = path
        records[str(pair_index)] = {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": probe._sha256(path),
        }
    probe._atomic_json(
        output_dir / "stage_train_cache_complete.json",
        {"state_count": pair_count, "records": records},
    )

    event_dir = output_dir / "teacher_calls.jsonl.events"
    event_dir.mkdir(parents=True)
    batch_paths: dict[int, Path] = {}
    event_index = 0
    for pair_index in range(pair_count):
        batch_id = f"state-{pair_index:04d}"
        events = (
            {
                "event": "exact_teacher_state_call_started",
                "stage": "train_cache",
                "split": "train",
                "pair_index": pair_index,
                "batch_id": batch_id,
            },
            {
                "event": "exact_teacher_state_call_completed",
                "stage": "train_cache",
                "split": "train",
                "pair_index": pair_index,
                "batch_id": batch_id,
            },
            {
                "event": "exact_teacher_batch_completed",
                "stage": "train_cache",
                "split": "train",
                "state_count": 1,
                "batch_id": batch_id,
            },
        )
        for offset, event in enumerate(events):
            path = event_dir / f"{event_index:06d}.json"
            path.write_text(json.dumps(event, sort_keys=True))
            if offset == 2:
                batch_paths[pair_index] = path
            event_index += 1

    weights_path = output_dir / "fit" / "convex_head_weights.npz"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"sealed-pre-manifest-weights")
    return record_paths, batch_paths


def test_policy_and_probe_refuse_nonunit_teacher_batches_and_keep_bars_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = probe.FrozenReplayConvexHeadPolicy()
    contract = policy.compile_measurement_contract()
    assert policy.teacher_batch_size == contract["teacher_batch_size"] == 1
    assert policy.operator_early_regime_cosine_bar == pytest.approx(-0.16153190769629602)
    assert policy.legacy_nonnegative_policy_overlay == pytest.approx(0.0)
    assert contract["operator_early_regime_cosine_bar"] < contract[
        "legacy_nonnegative_policy_overlay"
    ]
    with pytest.raises(ValueError, match="mean cross-entropy"):
        probe.FrozenReplayConvexHeadPolicy(teacher_batch_size=4)

    verdict = probe.derive_mission_verdict(
        heldout_costate_cosine=-0.1,
        teacher_call_amortization_x=5.0,
    )
    assert verdict["verdict"] == "GO"
    assert verdict["cosine_gate_pass"] is True
    assert verdict["legacy_nonnegative_policy_overlay_pass"] is False
    assert verdict["legacy_nonnegative_policy_overlay_is_decision_gate"] is False

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    forged_policy = types.SimpleNamespace(teacher_batch_size=4)
    with pytest.raises(probe.ProbeError, match="per-state exact-label parity requires batch size 1"):
        probe._build_training_cache(
            output_dir=tmp_path,
            assignments=(),
            labels=None,
            margins=None,
            policy=forged_policy,
            segnet=None,
            yopo=None,
        )


def test_atomic_event_files_repair_a_truncated_jsonl_projection(tmp_path: Path) -> None:
    ledger = tmp_path / "teacher_calls.jsonl"
    expected = [
        {"event": "exact_teacher_state_call_started", "pair_index": 7},
        {"event": "exact_teacher_state_call_completed", "pair_index": 7},
    ]
    for row in expected:
        probe._append_jsonl(ledger, row)
    with ledger.open("ab") as handle:
        handle.write(b'{"event":"truncated')

    rows = probe._canonicalize_event_ledger(ledger)
    projected = [json.loads(line) for line in ledger.read_text().splitlines()]
    event_files = sorted((tmp_path / "teacher_calls.jsonl.events").glob("*.json"))
    assert rows == expected
    assert projected == expected
    assert len(event_files) == 2
    assert all(
        json.loads(path.read_text())["event"] == row["event"]
        for path, row in zip(event_files, expected, strict=True)
    )


def test_cleanup_recovers_valid_atomic_temps_and_preserves_prior_actions(tmp_path: Path) -> None:
    first_temp = tmp_path / ".receipt.json.tmp.101"
    first_payload = b'{"receipt":"complete"}\n'
    first_temp.write_bytes(first_payload)
    first_sha = hashlib.sha256(first_payload).hexdigest()

    first = probe._write_cleanup_manifest(tmp_path, phase="ARMED")
    assert not first_temp.exists()
    assert (tmp_path / "receipt.json").read_bytes() == first_payload
    assert len(first["removed_scratch"]) == 1
    assert first["removed_scratch"][0]["action"] == "lossless_atomic_recovery_to_destination"
    assert first["removed_scratch"][0]["destination_sha256"] == first_sha

    prior_actions = list(first["removed_scratch"])
    second_temp = tmp_path / ".stage.json.tmp.202"
    second_payload = b'{"stage":"fit"}\n'
    second_temp.write_bytes(second_payload)
    second = probe._write_cleanup_manifest(tmp_path, phase="RESUME_PREFLIGHT")

    assert not second_temp.exists()
    assert (tmp_path / "stage.json").read_bytes() == second_payload
    assert second["removed_scratch"][: len(prior_actions)] == prior_actions
    assert len(second["removed_scratch"]) == 2
    assert all(
        row["action"] == "lossless_atomic_recovery_to_destination"
        for row in second["removed_scratch"]
    )
    assert second["preserved_scratch_blockers"] == []

    unrecognized_temp = tmp_path / ".orphan.bin.tmp.303"
    unrecognized_payload = b"unrecognized-but-preserved"
    unrecognized_temp.write_bytes(unrecognized_payload)
    third = probe._write_cleanup_manifest(tmp_path, phase="RESUME_PREFLIGHT")
    assert third["removed_scratch"][:2] == second["removed_scratch"]
    quarantined = third["removed_scratch"][-1]
    assert quarantined["action"] == "moved_losslessly_to_recovery_quarantine"
    assert (tmp_path / quarantined["quarantine_path"]).read_bytes() == unrecognized_payload
    assert quarantined["retry_teacher_calls_are_conservatively_charged"] is True
    assert third["preserved_scratch_blockers"] == []


def test_ordinary_resume_refuses_source_drift_without_explicit_amendment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_sources = _sealed_source_rows()
    current = probe._custody_fingerprints(prior_sources)
    _changed_fingerprint(current, next(iter(probe.SOURCE_AMENDMENT_CHANGED_PATHS)))
    monkeypatch.setattr(probe, "_source_fingerprints", lambda: current)

    with pytest.raises(
        probe.ProbeError,
        match="explicit verifier source amendment was not requested",
    ):
        probe._resolve_source_custody(
            tmp_path,
            prior_contract={"sources": prior_sources},
            requested_amendment=None,
            expected_train_pairs=set(range(480)),
        )


def test_source_amendment_refuses_unapproved_changed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_sources = _sealed_source_rows()
    current = probe._custody_fingerprints(prior_sources)
    for relative in probe.SOURCE_AMENDMENT_CHANGED_PATHS:
        _changed_fingerprint(current, relative)
    unexpected = next(
        relative
        for relative in probe.SOURCE_FILES
        if relative not in probe.SOURCE_AMENDMENT_CHANGED_PATHS
    )
    _changed_fingerprint(current, unexpected)
    monkeypatch.setattr(probe, "_source_fingerprints", lambda: current)

    with pytest.raises(probe.ProbeError, match="unapproved source delta"):
        probe._resolve_source_custody(
            tmp_path,
            prior_contract={"sources": prior_sources},
            requested_amendment=probe.SOURCE_AMENDMENT_ID,
            expected_train_pairs=set(range(480)),
        )


def test_source_amendment_refuses_old_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_sources = _sealed_source_rows()
    drifted_old_path = next(iter(probe.SOURCE_AMENDMENT_CHANGED_PATHS))
    prior_sources[drifted_old_path]["sha256"] = hashlib.sha256(b"wrong-old").hexdigest()
    current = probe._custody_fingerprints(prior_sources)
    for relative in probe.SOURCE_AMENDMENT_CHANGED_PATHS:
        _changed_fingerprint(current, relative)
    monkeypatch.setattr(probe, "_source_fingerprints", lambda: current)

    with pytest.raises(probe.ProbeError, match="source amendment old hash drift"):
        probe._resolve_source_custody(
            tmp_path,
            prior_contract={"sources": prior_sources},
            requested_amendment=probe.SOURCE_AMENDMENT_ID,
            expected_train_pairs=set(range(480)),
        )


def test_recovery_boundary_requires_exact_480_teacher_triplets_and_record_custody(
    tmp_path: Path,
) -> None:
    expected_pairs = set(range(480))
    record_paths, batch_paths = _write_recovery_boundary(tmp_path, pair_count=480)

    boundary = probe._verify_recovery_boundary(
        tmp_path, expected_train_pairs=expected_pairs
    )
    assert boundary["train_cache_stage"]["record_count"] == 480
    assert boundary["teacher_events"] == {
        "atomic_event_directory": "teacher_calls.jsonl.events",
        "atomic_event_file_count": 1_440,
        "atomic_event_tree_sha256": boundary["teacher_events"][
            "atomic_event_tree_sha256"
        ],
        "unique_started_states": 480,
        "unique_completed_states": 480,
        "batch_completions": 480,
        "teacher_calls_recomputed_by_amendment": 0,
    }

    missing_batch = batch_paths[479]
    missing_payload = missing_batch.read_bytes()
    missing_batch.unlink()
    with pytest.raises(probe.ProbeError, match="teacher batch coverage drift"):
        probe._verify_recovery_boundary(tmp_path, expected_train_pairs=expected_pairs)
    missing_batch.write_bytes(missing_payload)

    record_paths[0].write_bytes(b"custody-drift")
    with pytest.raises(probe.ProbeError, match="cache record drift: pair 0"):
        probe._verify_recovery_boundary(tmp_path, expected_train_pairs=expected_pairs)


def test_completed_run_resume_is_hash_verified_read_only_and_nonresume_refuses(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "completed"
    output_dir.mkdir()
    def write_custody(relative: str, payload: bytes, *, with_event_tree: bool) -> dict[str, Any]:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        custody: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": probe._sha256(path),
        }
        if with_event_tree:
            event_dir = output_dir / f"{relative}.events"
            event_dir.mkdir(parents=True)
            event = event_dir / "0000.json"
            event.write_text('{"event":"sealed"}\n')
            manifest = [
                {
                    "path": str(event.relative_to(output_dir)),
                    "bytes": event.stat().st_size,
                    "sha256": probe._sha256(event),
                }
            ]
            custody.update(
                {
                    "atomic_event_directory": str(event_dir.relative_to(output_dir)),
                    "atomic_event_file_count": 1,
                    "atomic_event_tree_sha256": hashlib.sha256(
                        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
        return custody

    run_contract_custody = write_custody("run_contract.json", b'{"sealed":true}\n', with_event_tree=False)
    invocation_custody = write_custody(
        "invocations.jsonl", b'{"event":"completed"}\n', with_event_tree=True
    )
    cleanup_custody = write_custody(
        "cleanup_manifest.json", b'{"phase":"COMPLETE_NO_BULK"}\n', with_event_tree=False
    )
    teacher_custody = write_custody(
        "teacher_calls.jsonl", b'{"event":"teacher"}\n', with_event_tree=True
    )
    receipt_path = output_dir / "measurement_receipt.json"
    receipt = {
        "schema": probe.SCHEMA,
        "verdict": {"verdict": "NO-GO"},
        "sealed": True,
        "initial_run_contract_custody": run_contract_custody,
        "invocation_custody": invocation_custody,
        "cleanup_custody": cleanup_custody,
        "teacher_call_accounting": {"teacher_call_ledger": teacher_custody},
    }
    probe._atomic_json(receipt_path, receipt)
    probe._atomic_json(
        output_dir / "complete.json",
        {
            "receipt": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": probe._sha256(receipt_path),
        },
    )
    before = _file_snapshot(output_dir)

    assert probe.run(output_dir=output_dir, resume=True, teacher_batch_size=1) == receipt
    assert _file_snapshot(output_dir) == before
    with pytest.raises(probe.ProbeError, match="completed run directory is sacred"):
        probe.run(output_dir=output_dir, resume=False, teacher_batch_size=1)
    assert _file_snapshot(output_dir) == before

    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
    with pytest.raises(probe.ProbeError, match="receipt custody drifted"):
        probe.run(output_dir=output_dir, resume=True, teacher_batch_size=1)


def test_terminal_validator_binds_source_amendment_and_effective_source_copies(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "completed-amendment"
    output_dir.mkdir()

    def write_custody(relative: str, payload: bytes) -> dict[str, Any]:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": probe._sha256(path),
        }

    run_contract = write_custody("run_contract.json", b"sealed-run-contract")
    invocation = write_custody("invocations.jsonl", b"sealed-invocation")
    cleanup = write_custody("cleanup_manifest.json", b"sealed-cleanup")
    teacher = write_custody("teacher_calls.jsonl", b"sealed-teacher-ledger")
    amendment = write_custody(
        f"source_amendment_{probe.SOURCE_AMENDMENT_ID}.json",
        b"sealed-source-amendment",
    )
    amendment["amendment_id"] = probe.SOURCE_AMENDMENT_ID
    effective_source = write_custody(
        "source_bundle_amendments/fit-ratio-scale-floor-v1/tool.py",
        b"effective-source-copy",
    )
    receipt = {
        "schema": probe.SCHEMA,
        "initial_run_contract_custody": run_contract,
        "invocation_custody": invocation,
        "cleanup_custody": cleanup,
        "teacher_call_accounting": {"teacher_call_ledger": teacher},
        "source_amendment_custody": amendment,
        "effective_source_custody": {"tool.py": effective_source},
    }
    receipt_path = output_dir / "measurement_receipt.json"
    probe._atomic_json(receipt_path, receipt)
    probe._atomic_json(
        output_dir / "complete.json",
        {
            "receipt": receipt_path.name,
            "bytes": receipt_path.stat().st_size,
            "sha256": probe._sha256(receipt_path),
        },
    )

    assert probe._completed_receipt_or_none(output_dir, resume=True) == receipt

    amendment_path = output_dir / amendment["path"]
    amendment_payload = amendment_path.read_bytes()
    amendment_path.write_bytes(amendment_payload + b"drift")
    with pytest.raises(probe.ProbeError, match="completed dependent custody drifted"):
        probe._completed_receipt_or_none(output_dir, resume=True)
    amendment_path.write_bytes(amendment_payload)

    effective_path = output_dir / effective_source["path"]
    effective_path.write_bytes(effective_path.read_bytes() + b"drift")
    with pytest.raises(probe.ProbeError, match="completed effective-source custody drifted"):
        probe._completed_receipt_or_none(output_dir, resume=True)


def test_fit_stage_preserves_matching_pre_manifest_weights_byte_for_byte(
    tmp_path: Path,
) -> None:
    assignments: list[probe.ReplayAssignment] = []
    records: list[probe.StateSufficientStatistics] = []
    for pair_index in range(2):
        assignment = probe.ReplayAssignment(
            pair_index=pair_index,
            checkpoint_index=pair_index,
            checkpoint_name=f"checkpoint-{pair_index}",
            split="train",
        )
        rng = np.random.default_rng(700 + pair_index)
        features = rng.normal(size=(16, len(probe.FEATURE_NAMES))).astype(np.float32)
        targets = rng.normal(size=(16, 3)).astype(np.float32)
        record = probe.cache_exact_label_sufficient_statistics(features, targets)
        probe._save_train_record(
            probe._train_record_path(tmp_path, pair_index),
            assignment,
            record,
            frame_sha256="f" * 64,
            label_sha256="1" * 64,
            margin_sha256="2" * 64,
            teacher_metrics={"ce": 1.0, "dseg": 0.5},
            teacher_elapsed_seconds=0.0,
        )
        assignments.append(assignment)
        records.append(record)

    policy = types.SimpleNamespace(fit_epochs=3, effective_training_state_steps=6)
    aggregate = probe.aggregate_sufficient_statistics(records)
    expected_fit = probe.fit_cached_convex_head(aggregate, epochs=policy.fit_epochs)
    weights_path = tmp_path / "fit" / "convex_head_weights.npz"
    probe._atomic_npz(
        weights_path,
        weights=expected_fit.weights,
        optimum_weights=expected_fit.optimum_weights,
        feature_names=np.asarray(probe.FEATURE_NAMES),
        hessian_sha256=np.asarray(expected_fit.certificate.hessian_sha256),
    )
    before_bytes = weights_path.read_bytes()
    before_mtime = weights_path.stat().st_mtime_ns

    fit, manifest = probe._fit_stage(
        output_dir=tmp_path,
        assignments=tuple(assignments),
        policy=policy,
    )

    assert np.array_equal(fit.weights, expected_fit.weights)
    assert manifest["recovered_pre_manifest_weights_without_rewrite"] is True
    assert weights_path.read_bytes() == before_bytes
    assert weights_path.stat().st_mtime_ns == before_mtime


def test_teacher_accounting_requires_exact_480_120_coverage_and_binds_ledgers(
    tmp_path: Path,
) -> None:
    policy = probe.FrozenReplayConvexHeadPolicy()
    assignments = probe.deterministic_replay_assignments(
        n_pairs=policy.n_pairs,
        checkpoint_names=("a", "b", "c"),
        holdout_period=policy.holdout_period,
        seed=policy.seed,
    )
    assert sum(row.split == "train" for row in assignments) == 480
    assert sum(row.split == "heldout" for row in assignments) == 120

    event_dir = tmp_path / "teacher_calls.jsonl.events"
    event_dir.mkdir()
    event_paths: list[Path] = []
    for index, row in enumerate(_teacher_event_rows(assignments)):
        path = event_dir / f"{index:04d}.json"
        path.write_text(json.dumps(row, sort_keys=True))
        event_paths.append(path)

    accounting = probe._teacher_accounting(tmp_path, policy, assignments)
    ledger = tmp_path / "teacher_calls.jsonl"
    event_manifest = [
        {
            "path": str(path.relative_to(tmp_path)),
            "bytes": path.stat().st_size,
            "sha256": probe._sha256(path),
        }
        for path in event_paths
    ]
    expected_tree_sha = hashlib.sha256(
        json.dumps(event_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert accounting["sealed_assignment_coverage"] == "PASS"
    assert accounting["completed_unique_state_calls"] == 600
    assert accounting["completed_unique_train_state_calls"] == 480
    assert accounting["completed_unique_heldout_state_calls"] == 120
    assert accounting["teacher_call_amortization_x"] == pytest.approx(12.0)
    assert accounting["teacher_call_ledger"]["rows"] == 1_200
    assert accounting["teacher_call_ledger"]["sha256"] == probe._sha256(ledger)
    assert accounting["teacher_call_ledger"]["atomic_event_file_count"] == 1_200
    assert accounting["teacher_call_ledger"]["atomic_event_tree_sha256"] == expected_tree_sha

    event_paths[1].unlink()
    with pytest.raises(probe.ProbeError, match="completion-ledger state coverage drift"):
        probe._teacher_accounting(tmp_path, policy, assignments)


@pytest.mark.parametrize(
    ("repeat_equal", "settled_equal", "message"),
    [
        (False, True, "cached held-out render repeat drift"),
        (True, False, "cached held-out settled parity drift"),
    ],
)
def test_cached_heldout_rows_refuse_repeat_or_settled_parity_drift(
    tmp_path: Path,
    repeat_equal: bool,
    settled_equal: bool,
    message: str,
) -> None:
    assignment = probe.ReplayAssignment(
        pair_index=0,
        checkpoint_index=0,
        checkpoint_name=probe.CHECKPOINTS[0][0],
        split="heldout",
    )
    record = probe._heldout_record_path(tmp_path, assignment.pair_index)
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps(
            {
                "assignment": assignment.to_dict(),
                "deterministic_render_repeat_equal": repeat_equal,
                "settled_renderer_parity": {
                    "equal": settled_equal,
                    "max_abs": 0.0,
                    "different_elements": 0,
                },
            }
        )
    )

    policy = types.SimpleNamespace(heldout_state_count=1)
    with pytest.raises(probe.ProbeError, match=message):
        probe._heldout_stage(
            output_dir=tmp_path,
            assignments=(assignment,),
            labels=None,
            margins=None,
            policy=policy,
            weights=None,
            segnet=None,
            yopo=None,
            matched=None,
        )


def test_teacher_completion_event_precedes_cached_record_write_in_source() -> None:
    source = textwrap.dedent(inspect.getsource(probe._build_training_cache))
    tree = ast.parse(source)
    call_lines: dict[str, list[int]] = {"_teacher_complete": [], "_save_train_record": []}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in call_lines
        ):
            call_lines[node.func.id].append(node.lineno)

    assert len(call_lines["_teacher_complete"]) == 1
    assert len(call_lines["_save_train_record"]) == 1
    assert call_lines["_teacher_complete"][0] < call_lines["_save_train_record"][0]
