from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import ddm_qbr1_cell_chain as chain

NOW = "2026-09-03T19:00:00Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _claims(path: Path, *, timestamp: str = "2026-09-03T18:07:03Z") -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            # Claims

            | timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |
            |---|---|---|---|---|---|---|---|
            | {timestamp} | MAIN | {chain.DEFAULT_SCORER_CLAIM} | local_macos_cpu | qbr1 | | active_eval | test |
            | {timestamp} | MAIN | {chain.DEFAULT_METAL_CLAIM} | local_mlx_metal | qbr1 | | active_eval | test |
            """
        ),
        encoding="utf-8",
    )


def _fake_process(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """\
            from __future__ import annotations
            import hashlib
            import json
            import os
            import sys
            from pathlib import Path

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n")

            def append_record(path, mode):
                with path.open("a") as handle:
                    handle.write(json.dumps({"mode": mode, "argv": [sys.executable, *sys.argv]}) + "\\n")

            mode = sys.argv[1]
            record = Path(sys.argv[sys.argv.index("--record") + 1])
            append_record(record, mode)
            if mode == "launcher":
                output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
                done_path = output_dir.parent / "SHARED.done"
                consumed_path = done_path.with_name(done_path.name + ".consumed.json")
                if done_path.exists():
                    digest = hashlib.sha256(done_path.read_bytes()).hexdigest()
                    if not consumed_path.exists() or json.loads(consumed_path.read_text()).get("receipt_sha256") != digest:
                        raise SystemExit(6)
                    done_path.unlink()
                    consumed_path.unlink()
                child = sys.argv[sys.argv.index("--") + 1:]
                config = json.loads(Path(child[-1]).read_text())
                result_path = Path(config["output"]) / "RESULT.json"
                write(result_path, {
                    "schema": "fake_qbr1_result.v1",
                    "complete": True,
                    "cell_id": config["cell_id"],
                    "completed_steps": config["total_steps"],
                })
                output_dir.mkdir(parents=True, exist_ok=True)
                pid = os.getpid()
                manifest_path = output_dir / "launch_manifest.json"
                launch_id = {"manifest_path": str(manifest_path), "pid": pid, "monotonic_launch_counter": int(config["order"])}
                (output_dir / "run.pid").write_text(f"{pid}\\n")
                write(manifest_path, {
                    "schema": "detached_local_process_launch.v2",
                    "dry_run": False,
                    "output_dir": str(output_dir),
                    "pid": pid,
                    "argv": child,
                    "launch_id": launch_id,
                    "done_receipt_path": str(done_path),
                })
                write(done_path, {
                    "schema": "detached_local_process_done.v2",
                    "launch_id": launch_id,
                    "receipt_name": "SHARED",
                    "rc": 0,
                    "adjudicated_at_launch": False,
                })
            elif mode == "adjudicate":
                output = Path(sys.argv[sys.argv.index("--output") + 1])
                write(output, {
                    "schema": "ddm_qbr1_adjudication_result.v1",
                    "disposition": "INCONCLUSIVE_MIXED_NO_FAMILY_CLOSURE",
                })
            else:
                raise SystemExit(9)
            """
        ),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path, *, completed: set[int] | None = None) -> SimpleNamespace:
    completed = completed or set()
    root = tmp_path / "qbr1"
    sealed_root = root / "sealed_configs"
    run_root = root / "runs"
    launch_root = root / "launch"
    claims = tmp_path / "claims.md"
    record = tmp_path / "invocations.jsonl"
    fake = tmp_path / "fake_process.py"
    source = tmp_path / "source.bin"
    source.write_bytes(b"sealed-source-payload")
    _claims(claims)
    _fake_process(fake)
    inputs = {"source": _fact(source)}
    cells = []
    result_paths = []
    for order in range(1, 7):
        cell_id = f"cell_{order}"
        output = run_root / cell_id
        config = {
            "schema": "fake_qbr1_config.v1",
            "cell_id": cell_id,
            "order": order,
            "output": str(output),
            "total_steps": 1,
            "launch_authorized": False,
            "scorer_lane": {"claimed": False, "claim_id": None},
            "metal_lane": {"claimed": False, "claim_id": None},
            "source_pins": inputs,
        }
        config_path = sealed_root / f"{cell_id}.json"
        _write_json(config_path, config)
        result_path = output / "RESULT.json"
        result_paths.append(str(result_path))
        if order in completed:
            _write_json(
                result_path,
                {
                    "schema": "fake_qbr1_result.v1",
                    "complete": True,
                    "cell_id": cell_id,
                    "completed_steps": 1,
                },
            )
        cells.append(
            {
                "order": order,
                "cell_id": cell_id,
                "config": _fact(config_path),
                "claim_mutation": "test exact five-field mutation",
                "launcher_argv": [
                    sys.executable,
                    str(fake),
                    "launcher",
                    "--record",
                    str(record),
                    "--output-dir",
                    str(launch_root / cell_id),
                    "--done-receipt",
                    "SHARED",
                    "--",
                    sys.executable,
                    str(fake),
                    "run-config",
                    "AUTHORIZED_CONFIG_PATH",
                ],
            }
        )
    adjudication = root / "ADJUDICATION_RESULT.json"
    adjudication_argv = [
        sys.executable,
        str(fake),
        "adjudicate",
        "--record",
        str(record),
        "--output",
        str(adjudication),
        *result_paths,
    ]
    fire_order = root / "SEALED_MAIN_FIRE_ORDER.json"
    _write_json(
        fire_order,
        {
            "schema": chain.FIRE_ORDER_SCHEMA,
            "inputs": inputs,
            "cells": cells,
            "adjudication_argv": adjudication_argv,
        },
    )
    return SimpleNamespace(
        root=root,
        fire_order=fire_order,
        claims=claims,
        record=record,
        fake=fake,
        source=source,
        cells=cells,
        adjudication_argv=adjudication_argv,
    )


def _args(fixture: SimpleNamespace, *extra: str) -> list[str]:
    return [
        "--fire-order",
        str(fixture.fire_order),
        "--claims",
        str(fixture.claims),
        "--now-utc",
        NOW,
        "--reserve-bytes",
        "0",
        "--poll-seconds",
        "0.001",
        "--terminal-grace-seconds",
        "0.01",
        *extra,
    ]


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_completed_cells_are_skipped_and_not_refired(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, completed={1, 2})

    assert chain.main(_args(fixture)) == 0

    records = _records(fixture.record)
    assert [row["mode"] for row in records].count("launcher") == 4
    ledger = _records(fixture.root / "CHAIN_LEDGER.jsonl")
    assert [row["action"] for row in ledger[:2]] == ["SKIPPED_COMPLETE", "SKIPPED_COMPLETE"]
    assert json.loads((fixture.root / "CHAIN_DONE.json").read_text())["status"] == "COMPLETE"


def test_other_live_cell_refuses_single_flight(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    live_dir = chain.launcher_output_dir(fixture.cells[1]["launcher_argv"])
    live_dir.mkdir(parents=True)
    (live_dir / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    assert chain.main(_args(fixture)) == 2

    refusal = json.loads(capsys.readouterr().out)
    assert refusal["reason"] == "OTHER_CELL_LIVE"
    assert not fixture.record.exists()


@pytest.mark.parametrize("failure", ["source", "claim", "storage"])
def test_preconditions_refuse_closed(failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    expected = {
        "source": "SOURCE_PIN_MISMATCH",
        "claim": "CLAIM_NOT_LIVE",
        "storage": "AP_RESERVE",
    }[failure]
    if failure == "source":
        fixture.source.write_bytes(b"drifted")
    elif failure == "claim":
        _claims(fixture.claims, timestamp="2026-09-02T18:00:00Z")
    else:
        monkeypatch.setattr(chain.os, "statvfs", lambda _path: SimpleNamespace(f_bavail=0, f_frsize=1))
        args[args.index("--reserve-bytes") + 1] = "1"

    assert chain.main(args) == 2

    refusal = json.loads(capsys.readouterr().out)
    assert refusal["reason"] == expected
    assert not fixture.record.exists()


def test_launcher_argv_binding_is_verbatim_except_config_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    assert chain.main(_args(fixture)) == 0

    launcher_rows = [row for row in _records(fixture.record) if row["mode"] == "launcher"]
    assert len(launcher_rows) == 6
    for cell, row in zip(fixture.cells, launcher_rows, strict=True):
        sealed_path = Path(cell["config"]["path"])
        auth_path = chain.authorized_path(sealed_path)
        assert row["argv"] == chain.bind_argv(cell["launcher_argv"], auth_path)
        authorized = json.loads(auth_path.read_text())
        sealed = json.loads(sealed_path.read_text())
        assert authorized == chain.authorized_config(
            sealed, chain.DEFAULT_SCORER_CLAIM, chain.DEFAULT_METAL_CLAIM
        )


def test_adjudication_runs_only_after_all_six_terminal_results(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    assert chain.main(_args(fixture)) == 0

    records = _records(fixture.record)
    assert [row["mode"] for row in records] == ["launcher"] * 6 + ["adjudicate"]
    assert records[-1]["argv"] == fixture.adjudication_argv
    done = json.loads((fixture.root / "CHAIN_DONE.json").read_text())
    assert done["cells_complete"] == [f"cell_{index}" for index in range(1, 7)]
    assert Path(done["adjudication"]["path"]) == fixture.root / "ADJUDICATION_RESULT.json"


def test_dry_run_uses_temporary_authorization_and_launches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)

    assert chain.main(_args(fixture, "--dry-run")) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "DRY_RUN_ONLY_NO_LAUNCH"
    assert len(plan["sequence"]) == 6
    assert all(row["argv_difference_from_seal"] == ["AUTHORIZED_CONFIG_PATH"] for row in plan["sequence"])
    assert not fixture.record.exists()
    assert not (fixture.root / "authorized_configs").exists()


def test_attach_live_current_cell_without_relaunch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    first = fixture.cells[0]
    sealed_path = Path(first["config"]["path"])
    sealed = json.loads(sealed_path.read_text())
    auth_path = chain.authorized_path(sealed_path)
    expected = chain.authorized_config(sealed, chain.DEFAULT_SCORER_CLAIM, chain.DEFAULT_METAL_CLAIM)
    chain.write_or_verify_authorized(auth_path, expected)
    bound = chain.bind_argv(first["launcher_argv"], auth_path)
    launch_dir = chain.launcher_output_dir(first["launcher_argv"])
    launch_dir.mkdir(parents=True)
    manifest_path = launch_dir / "launch_manifest.json"
    launch_id = {"manifest_path": str(manifest_path), "pid": os.getpid(), "monotonic_launch_counter": 1}
    (launch_dir / "run.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_json(
        manifest_path,
        {
            "schema": "detached_local_process_launch.v2",
            "dry_run": False,
            "output_dir": str(launch_dir),
            "pid": os.getpid(),
            "argv": chain.child_argv(bound),
            "launch_id": launch_id,
            "done_receipt_path": str(launch_dir.parent / "SHARED.done"),
        },
    )
    real_wait = chain.wait_for_terminal

    def finish_attached(cell, config, manifest, **kwargs):
        if cell["cell_id"] == "cell_1":
            result = Path(config["output"]) / "RESULT.json"
            _write_json(
                result,
                {"complete": True, "cell_id": "cell_1", "completed_steps": 1},
            )
            (launch_dir / "run.pid").write_text("99999999\n", encoding="utf-8")
            return {"attached_test": True}
        return real_wait(cell, config, manifest, **kwargs)

    monkeypatch.setattr(chain, "wait_for_terminal", finish_attached)

    assert chain.main(_args(fixture)) == 0

    launcher_rows = [row for row in _records(fixture.record) if row["mode"] == "launcher"]
    assert len(launcher_rows) == 5
    ledger = _records(fixture.root / "CHAIN_LEDGER.jsonl")
    assert ledger[0]["action"] == "ATTACHED_LIVE"


def test_storage_preflight_uses_exact_available_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        chain.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=4, f_frsize=1024),
    )
    assert chain.storage_preflight(tmp_path, 4096)["available_bytes"] == 4096
    with pytest.raises(chain.ChainRefusal, match="below the required reserve"):
        chain.storage_preflight(tmp_path, 4097)


def test_claim_expiry_is_rechecked_against_supplied_time(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    now = dt.datetime(2026, 9, 4, 18, 7, 4, tzinfo=dt.UTC)
    with pytest.raises(chain.ChainRefusal) as error:
        chain.verify_claim(
            fixture.claims,
            chain.DEFAULT_SCORER_CLAIM,
            "local_macos_cpu",
            now=now,
            ttl_hours=24,
        )
    assert error.value.reason == "CLAIM_NOT_LIVE"
