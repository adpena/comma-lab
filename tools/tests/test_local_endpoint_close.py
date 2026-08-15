"""Controls for the armed scorer-free local endpoint closer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO / "tools/local_endpoint_close.py"
    spec = importlib.util.spec_from_file_location("local_endpoint_close", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEC = _load_module()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_run(tmp_path: Path, *, source_rc: int = 0, checkpoints: bool = True):
    run_root = (tmp_path / "run").resolve()
    manifest_path = run_root / "launcher/launch_manifest.json"
    save = run_root / "checkpoints/full_mps_e960.pt"
    stage = save.with_name(save.stem + ".checkpoints") / "qat_stage_end_epoch_0960.pt"
    if checkpoints:
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_bytes(b"retained-final-best-checkpoint")
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_bytes(b"retained-stage-end-checkpoint")
    argv = [
        ".venv/bin/python",
        "tools/train_ddm_cl1_hpac_capacity_mps.py",
        "--epochs",
        "960",
        "--save",
        str(save),
        "--out",
        str(run_root / "reports/trainer.json"),
    ]
    _write_json(manifest_path, {"schema": "detached_local_process_launch.v2", "argv": argv})
    log = run_root / "launcher/run.log"
    rows = []
    for index, epoch in enumerate(range(482, 498, 2)):
        rows.append(
            json.dumps(
                {
                    "epoch": epoch,
                    "phase": "discrete_qat",
                    "estimated_joint_bytes": 132000 - 100 * index,
                    "bpp": 0.01,
                    "top1_error": 0.001,
                }
            )
        )
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    done = tmp_path / "source.done"
    _write_json(
        done,
        {
            "schema": LEC.DONE_SCHEMA,
            "rc": source_rc,
            "launch_id": {"manifest_path": str(manifest_path), "pid": 123, "monotonic_launch_counter": 1},
        },
    )
    return run_root, done, save, stage


def test_completed_endpoint_refits_hashes_payloads_and_emits_contained_chain(tmp_path: Path) -> None:
    run_root, done, save, stage = _completed_run(tmp_path)
    output = tmp_path / "closure"
    receipt = LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)
    assert receipt["status"] == "CLOSED"
    assert receipt["retained_source_done_receipt"]["sha256"] == receipt["source_done_receipt"]["sha256"]
    assert {row["path"] for row in receipt["payloads"]} == {str(save), str(stage)}
    assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in receipt["payloads"])
    fit = json.loads((output / "descent_law_refit.json").read_text())
    assert fit["schema"] == "hpac_descent_law_fit.v1"
    order = json.loads((output / "NEXT_FIRE_ORDER.json").read_text())
    assert [step["order"] for step in order["steps"]] == [1, 2, 3, 4]
    assert [step["disposition"] for step in order["steps"]] == [
        "FIRED",
        "QUEUED-WITH-A-FIRE-ORDER",
        "QUEUED-WITH-A-FIRE-ORDER",
        "QUEUED-WITH-A-FIRE-ORDER",
    ]
    assert all(
        all(step.get(key) for key in ("owner", "consumer_store", "fire_trigger"))
        for step in order["steps"]
    )
    assert order["paid_or_scorer_work_launched"] is False
    note = (output / "TERMINAL_NOTE.md").read_text()
    assert "PUSH NOTIFICATION" in note and "no paid or scorer work" in note
    assert (output / "ENDPOINT_CLOSURE.done.json").is_file()
    # Terminal replay is idempotent and does not re-run the fit.
    prior_mtime = (output / "descent_law_refit.json").stat().st_mtime_ns
    replay = LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)
    assert replay == receipt
    assert (output / "descent_law_refit.json").stat().st_mtime_ns == prior_mtime


def test_source_failure_closes_without_refit_or_downstream_order(tmp_path: Path) -> None:
    run_root, done, _, _ = _completed_run(tmp_path, source_rc=9, checkpoints=False)
    output = tmp_path / "closure"
    receipt = LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)
    assert receipt["status"] == "SOURCE_FAILED" and receipt["process_rc"] == 9
    assert not (output / "descent_law_refit.json").exists()
    assert not (output / "NEXT_FIRE_ORDER.json").exists()


def test_missing_final_checkpoint_refuses_after_retaining_fit_receipt(tmp_path: Path) -> None:
    run_root, done, _, _ = _completed_run(tmp_path, checkpoints=False)
    output = tmp_path / "closure"
    receipt = LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)
    assert receipt["status"] == "REFUSED_CHECKPOINT_OR_REFIT_CUSTODY"
    assert any("final checkpoint custody is incomplete" in error for error in receipt["errors"])
    assert (output / "descent_law_refit.json").is_file()
    assert not (output / "NEXT_FIRE_ORDER.json").exists()


def test_read_only_arm_without_done_receipt_is_resumable_pending(tmp_path: Path) -> None:
    run_root = (tmp_path / "live_run").resolve()
    output = tmp_path / "closure"
    result = LEC.wait_and_close(
        run_root=run_root,
        done_receipt_path=tmp_path / "future.done",
        output_dir=output,
        deadline_s=60,
        poll_s=0.001,
        once=True,
    )
    assert result is None
    state = json.loads((output / "POLL_STATE.json").read_text())
    assert state["status"] == "ARMED_WAITING"
    assert state["source_deadline_is_process_failure"] is False
    assert "rerun the same argv" in json.loads((output / "ARMED.json").read_text())["resumable_by"]
    assert not (output / LEC.RECEIPT_NAME).exists()


def test_done_receipt_for_another_run_refuses(tmp_path: Path) -> None:
    run_root, done, _, _ = _completed_run(tmp_path)
    payload = json.loads(done.read_text())
    payload["launch_id"]["manifest_path"] = str(tmp_path / "other/launch_manifest.json")
    _write_json(done, payload)
    with pytest.raises(LEC.LocalEndpointCloseError, match="another run manifest"):
        LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=tmp_path / "closure")


def test_corrupt_prior_terminal_receipt_is_not_silently_adopted(tmp_path: Path) -> None:
    run_root, done, _, _ = _completed_run(tmp_path)
    output = tmp_path / "closure"
    _write_json(output / LEC.RECEIPT_NAME, {"schema": LEC.RECEIPT_SCHEMA, "status": "CLOSED"})
    with pytest.raises(LEC.LocalEndpointCloseError, match="receipt missing"):
        LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)


def test_unknown_prior_terminal_status_is_not_silently_adopted(tmp_path: Path) -> None:
    run_root, done, _, _ = _completed_run(tmp_path)
    output = tmp_path / "closure"
    receipt = LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)
    receipt["status"] = "MYSTERY"
    _write_json(output / LEC.RECEIPT_NAME, receipt)
    with pytest.raises(LEC.LocalEndpointCloseError, match="status is unknown"):
        LEC.execute_closure(run_root=run_root, done_receipt_path=done, output_dir=output)
