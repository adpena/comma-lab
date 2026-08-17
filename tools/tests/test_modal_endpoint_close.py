from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest

from tac.deploy.modal.call_id_ledger import (
    query_by_call_id,
    register_dispatched_call_id,
)
from tools import modal_endpoint_close as close
from tools import preflight_hook
from tools.claim_lane_dispatch import HEADER
from tools.modal_harvest_poller import (
    POLL_DEADLINE,
    POLL_REMOTE_FAILURE,
    POLL_RESULT,
    poll_modal_call,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _entry(payload: bytes = b"payload", remote_path: str = "run/retained/a.bin") -> dict:
    return {
        "name": "a",
        "remote_path": remote_path,
        "bytes": len(payload),
        "sha256": _digest(payload),
    }


def _manifest(payloads: list[dict] | None = None) -> dict:
    return {
        "schema": close.CLOSURE_MANIFEST_SCHEMA,
        "volume_name": "retained-volume",
        "lane_id": "lane_ac1_test",
        "instance_job_id": "modal:ac1-test",
        "payloads": payloads or [],
    }


def _result(payloads: list[dict] | None = None) -> dict:
    return {
        "schema": "fixture.v1",
        "training_complete": True,
        "returncode": 0,
        "closure_manifest": _manifest(payloads),
    }


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def test_poller_returns_after_timeout_then_result() -> None:
    attempts = iter([TimeoutError(), {"ok": True}])

    def get_result(_timeout: float):
        value = next(attempts)
        if isinstance(value, Exception):
            raise value
        return value

    clock = _Clock()
    outcome = poll_modal_call(
        call_id="fc-test",
        deadline_s=10,
        poll_s=1,
        get_result=get_result,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert outcome == {"kind": POLL_RESULT, "result": {"ok": True}}


def test_poller_classifies_remote_exception() -> None:
    def get_result(_timeout: float):
        raise RuntimeError("remote failed")

    outcome = poll_modal_call(
        call_id="fc-test",
        deadline_s=10,
        poll_s=0,
        get_result=get_result,
    )
    assert outcome["kind"] == POLL_REMOTE_FAILURE
    assert outcome["error_class"] == "RuntimeError"


def test_poller_deadline_is_not_terminal_provider_evidence() -> None:
    clock = _Clock()

    def get_result(_timeout: float):
        clock.value += 1
        raise TimeoutError

    outcome = poll_modal_call(
        call_id="fc-test",
        deadline_s=2,
        poll_s=0,
        get_result=get_result,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    assert outcome["kind"] == POLL_DEADLINE
    assert not close.derive_terminal_status({}, poll_kind=POLL_DEADLINE).terminal


@pytest.mark.parametrize(
    ("result", "success", "ledger_status", "reason"),
    [
        ({"training_complete": True}, True, "harvested", "training_complete_true"),
        ({"returncode": 0}, True, "harvested", "zero_returncode"),
        ({"status": "success"}, True, "harvested", "success_status_word"),
        ({"returncode": 9, "training_complete": True}, False, "failed", "nonzero_returncode"),
        ({"error": "boom"}, False, "failed", "error_field_present"),
        ({"training_complete": False}, False, "failed", "training_complete_false"),
        ({"status": "timed_out"}, False, "failed", "failure_status_word"),
        ({"unknown": True}, False, "failed", "terminal_result_lacks_success_evidence"),
    ],
)
def test_terminal_status_derivation_table(result: dict, success: bool, ledger_status: str, reason: str) -> None:
    decision = close.derive_terminal_status(result)
    assert decision.terminal
    assert decision.remote_success is success
    assert decision.ledger_status == ledger_status
    assert decision.reason == reason


def test_remote_exception_is_terminal_failure() -> None:
    decision = close.derive_terminal_status({}, poll_kind=POLL_REMOTE_FAILURE)
    assert decision.terminal and not decision.remote_success
    assert decision.rc == 1


def test_legacy_ec2_final_fixture_is_recognized() -> None:
    decision = close.derive_terminal_status(
        {
            "schema": "ddm_ec2_adapter_trainer_result.v1",
            "endpoint": {},
            "selected": {},
        }
    )
    assert decision.remote_success


def test_payload_extractor_reads_both_manifest_block_names() -> None:
    payload = b"abc"
    record = {
        "path": "/mounted/run-a/retained/a.bin",
        "bytes": len(payload),
        "sha256": _digest(payload),
    }
    rows = close.payload_entries_from_result(
        {"endpoint": {"retained_payloads": {"a": record}}, "selected": {"payloads": {"b": record}}},
        run_id="run-a",
    )
    assert len(rows) == 2
    assert {row["remote_path"] for row in rows} == {"run-a/retained/a.bin"}


def test_absolute_remote_path_requires_run_id() -> None:
    with pytest.raises(close.EndpointClosureError, match="does not contain"):
        close.remote_path_from_record("/mounted/other/a.bin", run_id="run-a")


def test_unsafe_relative_remote_path_is_rejected() -> None:
    with pytest.raises(close.EndpointClosureError, match="unsafe"):
        close.remote_path_from_record("../escape.bin", run_id="run-a")


def test_manifest_rejects_lane_mismatch() -> None:
    manifest = _manifest()
    manifest["lane_id"] = "other"
    with pytest.raises(close.EndpointClosureError, match="lane mismatch"):
        close.validate_closure_manifest(
            manifest,
            lane_id="lane_ac1_test",
            instance_job_id="modal:ac1-test",
        )


def test_manifest_refuses_omitted_result_payload() -> None:
    payload = b"abc"
    result = _result()
    result["run_id"] = "run-a"
    result["endpoint"] = {
        "retained_payloads": {
            "a": {
                "path": "/mounted/run-a/a.bin",
                "bytes": len(payload),
                "sha256": _digest(payload),
            }
        }
    }
    with pytest.raises(close.EndpointClosureError, match="omits"):
        close.resolve_closure_manifest(
            result,
            lane_id="lane_ac1_test",
            instance_job_id="modal:ac1-test",
        )


def test_payload_download_precreates_destination_directories(tmp_path: Path) -> None:
    payload = b"exact payload"
    seen: list[Path] = []

    def getter(**kwargs) -> None:
        destination = kwargs["destination"]
        assert destination.parent.is_dir()
        seen.append(destination)
        destination.write_bytes(payload)

    rows, _ = close.harvest_payloads(
        manifest=_manifest([_entry(payload)]),
        local_store=tmp_path / "store",
        modal_executable=Path("modal"),
        volume_get=getter,
        reserve_bytes=0,
    )
    assert seen
    assert rows[0]["status"] == "downloaded_verified"


def test_sha_mismatch_refuses_and_preserves_download(tmp_path: Path) -> None:
    def getter(**kwargs) -> None:
        kwargs["destination"].write_bytes(b"wrong")

    with pytest.raises(close.PayloadCustodyError) as caught:
        close.harvest_payloads(
            manifest=_manifest([_entry(b"right")]),
            local_store=tmp_path / "store",
            modal_executable=Path("modal"),
            volume_get=getter,
            reserve_bytes=0,
        )
    mismatch = Path(caught.value.rows[0]["local_path"])
    assert mismatch.is_file() and mismatch.read_bytes() == b"wrong"
    assert caught.value.rows[0]["status"] == "download_mismatch_preserved"


def test_volume_get_exception_becomes_typed_custody_refusal(tmp_path: Path) -> None:
    def getter(**_kwargs) -> None:
        raise OSError("network unavailable")

    with pytest.raises(close.PayloadCustodyError) as caught:
        close.harvest_payloads(
            manifest=_manifest([_entry(b"right")]),
            local_store=tmp_path / "store",
            modal_executable=Path("modal"),
            volume_get=getter,
            reserve_bytes=0,
        )
    assert caught.value.rows[0]["status"] == "download_failed_staging_preserved"
    assert "OSError" in caught.value.rows[0]["error"]


def test_existing_exact_payload_is_noop(tmp_path: Path) -> None:
    payload = b"already here"
    destination = tmp_path / "store/run/retained/a.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    def getter(**_kwargs) -> None:
        raise AssertionError("volume get must not run")

    rows, _ = close.harvest_payloads(
        manifest=_manifest([_entry(payload)]),
        local_store=tmp_path / "store",
        modal_executable=Path("modal"),
        volume_get=getter,
        reserve_bytes=0,
    )
    assert rows[0]["status"] == "already_present_verified"


def test_dry_run_verifies_available_fixture_and_marks_missing(tmp_path: Path) -> None:
    available = b"available"
    fixture = tmp_path / "fixtures/a.bin"
    fixture.parent.mkdir()
    fixture.write_bytes(available)
    rows, storage = close.harvest_payloads(
        manifest=_manifest(
            [
                _entry(available, "run/a.bin"),
                _entry(b"missing", "run/missing.bin"),
            ]
        ),
        local_store=tmp_path / "store",
        modal_executable=Path("modal"),
        dry_run=True,
        fixture_roots=[fixture.parent],
    )
    assert [row["status"] for row in rows] == ["fixture_verified", "would_pull"]
    assert storage["dry_run"]


def test_memo_handoff_sha_gate(tmp_path: Path) -> None:
    memo = tmp_path / "memo.md"
    memo.write_text("real memo\n")
    with pytest.raises(close.EndpointClosureError, match="sha gate"):
        close.commit_memo_handoff(
            {
                "schema": close.MEMO_HANDOFF_SCHEMA,
                "path": "memo.md",
                "sha256": "0" * 64,
                "message": "memo [no-triality] [p0-ledger-ok]",
            },
            repo_root=tmp_path,
            dry_run=True,
        )


def test_memo_handoff_uses_serializer_and_required_tags(tmp_path: Path) -> None:
    memo = tmp_path / "memo.md"
    memo.write_text("real memo\n")
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="committed", stderr="")

    result = close.commit_memo_handoff(
        {
            "schema": close.MEMO_HANDOFF_SCHEMA,
            "path": "memo.md",
            "sha256": close.sha256_file(memo),
            "message": "memo [no-triality] [p0-ledger-ok]",
        },
        repo_root=tmp_path,
        dry_run=False,
        serializer_runner=runner,
    )
    assert result["status"] == "committed"
    assert "subagent_commit_serializer.py" in " ".join(calls[0])
    assert "--no-co-author" in calls[0]


def test_np1_exact_heading_is_extracted_and_omission_stays_empty(tmp_path: Path) -> None:
    with_next = tmp_path / "with.md"
    without = tmp_path / "without.md"
    with_next.write_text("## NEXT_IF_RESUMED\n\n- run the endpoint\n\n## LIVE-HYPOTHESES\n")
    without.write_text("No future action remains.\n")
    result = close.extract_next_surface(
        final_message_paths=[with_next, without],
        inline_final_message=None,
        dry_run=True,
        next_store=tmp_path / "next.jsonl",
        name="ac1",
    )
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["heading"] == "NEXT_IF_RESUMED"


def test_receipt_schema_round_trip(tmp_path: Path) -> None:
    receipt = {
        "schema": close.RECEIPT_SCHEMA,
        "call_id": "fc-test",
        "status": "CLOSED",
        "process_rc": 0,
        "terminal_decision": {},
        "ledgers": {},
        "payloads": [],
        "memo_commit": {},
        "next_if_resumed": {},
    }
    close._write_receipt_and_done(tmp_path, receipt)
    assert close.validate_receipt(json.loads((tmp_path / close.RECEIPT_NAME).read_text())) == receipt
    assert json.loads((tmp_path / close.DONE_NAME).read_text())["receipt"]["sha256"]


def _active_ledgers(tmp_path: Path) -> tuple[Path, Path, str]:
    claims = tmp_path / "claims.md"
    claims.write_text(
        HEADER + "| 2026-08-14T12:00:00Z | test-agent | lane_ac1_test | modal | "
        "modal:ac1-test | 2026-08-14T15:00:00Z | active_test | test claim |\n"
    )
    ledger = tmp_path / "calls.jsonl"
    call_id = "fc-ac1-test"
    register_dispatched_call_id(
        call_id=call_id,
        lane_id="lane_ac1_test",
        label="ac1-test",
        agent="test-agent",
        path=ledger,
        lock_path=ledger.with_suffix(".jsonl.lock"),
    )
    return claims, ledger, call_id


def _authenticated_spawn_metadata(call_id: str = "fc-ac1-test") -> dict:
    return {
        "schema_version": "modal_auth_eval_spawn_v1",
        "tool": "experiments/modal_auth_eval_cpu.py",
        "app": "comma-auth-eval-cpu",
        "axis": "contest_cpu",
        "call_id": call_id,
        "lane_id": "lane_ac1_test",
        "instance_job_id": "modal:ac1-test",
        "claim_agent": "test-agent",
        "claim_platform": "modal",
        "local_request": {
            "archive_sha256": "a" * 64,
            "source_repo_commit": "b" * 40,
            "pair_group_id": "pair-test",
        },
    }


def test_dual_ledgers_close_and_idempotent_rerun_adds_no_rows(tmp_path: Path) -> None:
    claims, ledger, call_id = _active_ledgers(tmp_path)
    kwargs = {
        "call_id": call_id,
        "result": _result(),
        "poll_kind": POLL_RESULT,
        "output_dir": tmp_path / "output",
        "local_store": tmp_path / "store",
        "lane_id": "lane_ac1_test",
        "instance_job_id": "modal:ac1-test",
        "agent": "test-agent",
        "claims_path": claims,
        "call_ledger_path": ledger,
    }
    first = close.execute_endpoint_closure(**kwargs)
    ledger_lines = ledger.read_text().splitlines()
    claim_rows = [line for line in claims.read_text().splitlines() if "lane_ac1_test" in line]
    second = close.execute_endpoint_closure(**kwargs)
    assert first["status"] == second["status"] == "CLOSED"
    assert first["ledgers"]["both_terminal"]
    assert len(ledger_lines) == len(ledger.read_text().splitlines()) == 2
    assert len(claim_rows) == len([line for line in claims.read_text().splitlines() if "lane_ac1_test" in line]) == 2
    assert query_by_call_id(call_id, path=ledger)[-1]["status"] == "harvested"


def test_missing_call_registration_is_recovered_from_exact_spawn_metadata(
    tmp_path: Path,
) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        HEADER + "| 2026-08-14T12:00:00Z | test-agent | lane_ac1_test | modal | "
        "modal:ac1-test | 2026-08-14T15:00:00Z | active_test | test claim |\n"
    )
    ledger = tmp_path / "calls.jsonl"
    receipt = close.execute_endpoint_closure(
        call_id="fc-ac1-test",
        result=_result(),
        poll_kind=POLL_RESULT,
        output_dir=tmp_path / "output",
        local_store=tmp_path / "store",
        lane_id="lane_ac1_test",
        instance_job_id="modal:ac1-test",
        agent="test-agent",
        claims_path=claims,
        call_ledger_path=ledger,
        spawn_metadata=_authenticated_spawn_metadata(),
    )
    assert receipt["status"] == "CLOSED"
    assert receipt["ledgers"]["call_id"]["registration_action"] == (
        "recovered_from_spawn_metadata"
    )
    rows = query_by_call_id("fc-ac1-test", path=ledger)
    assert [row["status"] for row in rows] == ["dispatched", "harvested"]


def test_spawn_metadata_mismatch_refuses_registration_recovery(tmp_path: Path) -> None:
    claims = tmp_path / "claims.md"
    claims.write_text(
        HEADER + "| 2026-08-14T12:00:00Z | test-agent | lane_ac1_test | modal | "
        "modal:ac1-test | 2026-08-14T15:00:00Z | active_test | test claim |\n"
    )
    metadata = _authenticated_spawn_metadata()
    metadata["lane_id"] = "lane-wrong"
    receipt = close.execute_endpoint_closure(
        call_id="fc-ac1-test",
        result=_result(),
        poll_kind=POLL_RESULT,
        output_dir=tmp_path / "output",
        local_store=tmp_path / "store",
        lane_id="lane_ac1_test",
        instance_job_id="modal:ac1-test",
        agent="test-agent",
        claims_path=claims,
        call_ledger_path=tmp_path / "calls.jsonl",
        spawn_metadata=metadata,
    )
    assert receipt["status"] == "REFUSED_DUAL_LEDGER"
    assert "does not authenticate" in receipt["errors"][0]


@pytest.mark.parametrize(
    "relative_path",
    ["experiments/modal_auth_eval.py", "experiments/modal_auth_eval_cpu.py"],
)
def test_auth_dispatchers_register_before_writing_spawn_metadata(
    relative_path: str,
) -> None:
    source = (close.REPO_ROOT / relative_path).read_text()
    spawn = source.index(".spawn(*call_args)")
    register = source.index("register_dispatched_call_id_fail_closed(", spawn)
    metadata = source.index("write_spawn_metadata(", spawn)
    assert spawn < register < metadata


def test_dry_run_does_not_mutate_either_ledger(tmp_path: Path) -> None:
    claims, ledger, call_id = _active_ledgers(tmp_path)
    claims_before = claims.read_bytes()
    ledger_before = ledger.read_bytes()
    receipt = close.execute_endpoint_closure(
        call_id=call_id,
        result=_result(),
        poll_kind=POLL_RESULT,
        output_dir=tmp_path / "output",
        local_store=tmp_path / "store",
        lane_id="lane_ac1_test",
        instance_job_id="modal:ac1-test",
        agent="test-agent",
        claims_path=claims,
        call_ledger_path=ledger,
        dry_run=True,
    )
    assert receipt["status"] == "DRY_RUN_VALIDATED"
    assert claims.read_bytes() == claims_before
    assert ledger.read_bytes() == ledger_before


def test_call_ledger_writer_failure_still_emits_refusal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claims, ledger, call_id = _active_ledgers(tmp_path)

    def fail_writer(**_kwargs) -> None:
        raise OSError("ledger unavailable")

    monkeypatch.setattr(close, "update_call_id_outcome", fail_writer)
    receipt = close.execute_endpoint_closure(
        call_id=call_id,
        result=_result(),
        poll_kind=POLL_RESULT,
        output_dir=tmp_path / "output",
        local_store=tmp_path / "store",
        lane_id="lane_ac1_test",
        instance_job_id="modal:ac1-test",
        agent="test-agent",
        claims_path=claims,
        call_ledger_path=ledger,
    )
    assert receipt["status"] == "REFUSED_DUAL_LEDGER"
    assert "canonical call-id closure failed" in receipt["errors"][0]
    assert (tmp_path / "output" / close.RECEIPT_NAME).is_file()


def test_new_modal_dispatcher_without_manifest_warns() -> None:
    warning = preflight_hook.scan_modal_closure_manifest_source(
        "experiments/new_modal_runner.py", "call = fn.spawn(payload)\n"
    )
    assert warning and "no closure manifest" in warning


def test_new_modal_dispatcher_with_manifest_passes() -> None:
    source = (
        "SCHEMA = \"modal_endpoint_closure_manifest.v1\"\nresult = {'closure_manifest': {}}\ncall = fn.spawn(payload)\n"
    )
    assert preflight_hook.scan_modal_closure_manifest_source("experiments/new_modal_runner.py", source) is None


def test_manifest_free_same_line_waiver_requires_substantive_reason() -> None:
    source = "call = fn.spawn(payload)  # MODAL_CLOSURE_MANIFEST_FREE: returns no data and owns no retained artifacts\n"
    assert preflight_hook.scan_modal_closure_manifest_source("experiments/new_modal_runner.py", source) is None


def test_manifest_free_placeholder_waiver_is_rejected() -> None:
    source = "call = fn.spawn(payload)  # MODAL_CLOSURE_MANIFEST_FREE: TODO later\n"
    warning = preflight_hook.scan_modal_closure_manifest_source("experiments/new_modal_runner.py", source)
    assert warning and "placeholder" in warning


def test_ec2_reference_manifest_covers_nested_and_explicit_payloads(tmp_path: Path) -> None:
    ec2 = importlib.import_module("experiments.ddm_ec2_modal_oriented_adapter_trainer")
    run_id = "ddm_ec2_fixture"
    root = tmp_path / run_id
    root.mkdir()
    nested = root / "nested.bin"
    final = root / "FINAL_RESULT.json"
    nested.write_bytes(b"nested")
    final.write_bytes(b"final")
    result = {
        "endpoint": {
            "retained_payloads": {
                "nested": {
                    "path": str(nested),
                    "bytes": nested.stat().st_size,
                    "sha256": ec2.sha256_file(nested),
                }
            }
        }
    }
    manifest = ec2.build_closure_manifest(
        request={
            "run_id": run_id,
            "lane_id": "lane",
            "instance_job_id": "modal:job",
        },
        result=result,
        explicit_paths={"final_result": final},
    )
    assert manifest["schema"] == close.CLOSURE_MANIFEST_SCHEMA
    assert {row["remote_path"] for row in manifest["payloads"]} == {
        f"{run_id}/nested.bin",
        f"{run_id}/FINAL_RESULT.json",
    }


def test_ec2_arming_uses_detached_launcher_and_done_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ec2 = importlib.import_module("experiments.ddm_ec2_modal_oriented_adapter_trainer")
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="launched", stderr="")

    monkeypatch.setattr(ec2.subprocess, "run", fake_run)
    receipt = ec2.arm_endpoint_closer(
        call_id="fc-test",
        request={
            "run_id": "run-test",
            "lane_id": "lane-test",
            "instance_job_id": "modal:run-test",
            "claim_agent": "main:test",
        },
        output=tmp_path,
    )
    command = calls[0]
    assert "tools/launch_detached_process.py" in " ".join(command)
    assert "tools/modal_endpoint_close.py" in " ".join(command)
    assert "--done-receipt" in command
    assert receipt["returncode"] == 0


def test_endpoint_closure_extract_inherits_the_retraction_channel(tmp_path: Path) -> None:
    """The closer is a PRODUCER, not a reader of the JSONL: it extracts blocks from an
    arm's final message and appends them. It must therefore inherit the auto-retraction
    that fires when a corrected memo is re-extracted, or a closure re-run after a
    correction would leave the stale plan row live beside the corrected one."""
    store = tmp_path / "next.jsonl"
    memo = tmp_path / "ddm_zz1_final.md"
    memo.write_text("## NEXT_IF_RESUMED\n\n- fire below 186,269 B\n")
    close.extract_next_surface(
        final_message_paths=[memo],
        inline_final_message=None,
        dry_run=False,
        next_store=store,
        name="zz1",
    )
    memo.write_text("## NEXT_IF_RESUMED\n\n- fire at or below the derived pure-rate bar\n")
    result = close.extract_next_surface(
        final_message_paths=[memo],
        inline_final_message=None,
        dry_run=False,
        next_store=store,
        name="zz1",
    )

    assert result["extract_summary"]["auto_retracted"] == 1
    rows = [json.loads(line) for line in store.read_text().splitlines()]
    plans = [r for r in rows if r["schema"] == "codex_arm_queue.next_if_resumed.v1"]
    retractions = [r for r in rows if r["schema"].endswith("retraction.v1")]
    assert len(plans) == 2 and len(retractions) == 1  # nothing deleted; the stale one is flagged
    assert retractions[0]["target_row_id"] == plans[0]["row_id"]
