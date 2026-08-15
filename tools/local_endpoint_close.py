#!/usr/bin/env python3
"""Close a completed local HPAC continuation without firing downstream work.

The process is safe to arm at launch: it waits for the canonical detached-run
done receipt, fits the retained log, inventories both final checkpoints, and
publishes a typed fire order for MAIN.  It never invokes a scorer, provider,
identity race, archive compiler, or auth evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FITTER = REPO_ROOT / "tools/fit_hpac_descent_law.py"
DONE_SCHEMA = "detached_local_process_done.v2"
ARM_SCHEMA = "detached_local_process_receipt_arm.v1"
RECEIPT_SCHEMA = "local_hpac_endpoint_closure_receipt.v1"
FIRE_ORDER_SCHEMA = "local_hpac_endpoint_next_fire_order.v1"
RECEIPT_NAME = "ENDPOINT_CLOSURE.receipt.json"


class LocalEndpointCloseError(RuntimeError):
    """The endpoint cannot be closed without losing custody or provenance."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(payload)
    tmp.replace(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, _canonical_json(payload))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalEndpointCloseError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LocalEndpointCloseError(f"JSON root is not an object: {path}")
    return value


def _option_value(argv: Sequence[str], flag: str) -> str:
    for index, token in enumerate(argv):
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
        if token == flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                break
            return argv[index + 1]
    raise LocalEndpointCloseError(f"launch argv lacks required option {flag}")


def _validate_source_done(done: Mapping[str, Any], run_root: Path) -> Path:
    if done.get("schema") != DONE_SCHEMA:
        raise LocalEndpointCloseError(f"source done receipt schema differs: {done.get('schema')!r}")
    launch_id = done.get("launch_id")
    if not isinstance(launch_id, dict) or not isinstance(launch_id.get("manifest_path"), str):
        raise LocalEndpointCloseError("source done receipt has no typed launch manifest identity")
    observed = Path(launch_id["manifest_path"]).expanduser().resolve(strict=False)
    expected = (run_root / "launcher/launch_manifest.json").resolve(strict=False)
    if observed != expected:
        raise LocalEndpointCloseError(
            f"source done receipt points at another run manifest: observed {observed}, expected {expected}"
        )
    return observed


def _checkpoint_records(manifest: Mapping[str, Any], run_root: Path) -> list[dict[str, Any]]:
    argv = manifest.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise LocalEndpointCloseError("launch manifest argv is malformed")
    save = Path(_option_value(argv, "--save")).expanduser().resolve(strict=False)
    epochs = int(_option_value(argv, "--epochs"))
    if run_root not in save.parents:
        raise LocalEndpointCloseError(f"final checkpoint is outside the declared run root: {save}")
    checkpoint_root = save.with_name(save.stem + ".checkpoints")
    stage = checkpoint_root / f"qat_stage_end_epoch_{epochs:04d}.pt"
    paths = [save, stage]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise LocalEndpointCloseError(f"final checkpoint custody is incomplete: missing {missing}")
    records = [file_record(path) for path in paths]
    if any(row["bytes"] < 1 for row in records):
        raise LocalEndpointCloseError("a final checkpoint payload is empty")
    return records


def _run_fitter(run_root: Path, output_dir: Path) -> dict[str, Any]:
    fit_path = output_dir / "descent_law_refit.json"
    log_path = run_root / "launcher/run.log"
    argv = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(FITTER),
        "--log",
        str(log_path),
        "--out",
        str(fit_path),
    ]
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    fit_log = output_dir / "descent_law_refit.log"
    atomic_text(fit_log, proc.stdout)
    process_receipt = {
        "schema": "local_hpac_endpoint_fit_process.v1",
        "argv": argv,
        "rc": proc.returncode,
        "source_log": file_record(log_path),
        "process_log": file_record(fit_log),
    }
    if proc.returncode != 0 or not fit_path.is_file():
        raise LocalEndpointCloseError(
            f"descent-law refit failed rc={proc.returncode}; retained log {fit_log}"
        )
    fit_payload = _read_object(fit_path)
    if fit_payload.get("schema") != "hpac_descent_law_fit.v1":
        raise LocalEndpointCloseError("descent-law refit receipt schema differs")
    source_records = fit_payload.get("sources")
    source_log = process_receipt["source_log"]
    if not isinstance(source_records, list) or not any(
        isinstance(row, dict)
        and row.get("path") == source_log["path"]
        and row.get("sha256") == source_log["sha256"]
        for row in source_records
    ):
        raise LocalEndpointCloseError("descent-law refit did not bind the exact retained run log")
    process_receipt["fit_receipt"] = file_record(fit_path)
    atomic_json(output_dir / "descent_law_refit_process.json", process_receipt)
    return process_receipt


def _next_fire_order(
    *, run_root: Path, checkpoint_records: Sequence[Mapping[str, Any]], fit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": FIRE_ORDER_SCHEMA,
        "generated_utc": _utc_now(),
        "source_run_root": str(run_root),
        "containment": "MAIN adjudicates every row; this closer launches none of them",
        "steps": [
            {
                "order": 1,
                "action": "endpoint descent-law refit and checkpoint custody",
                "disposition": "FIRED",
                "owner": "local_endpoint_close",
                "consumer_store": str(Path(fit["fit_receipt"]["path"]).parent),
                "fire_trigger": "source detached done receipt rc=0",
                "checkpoint_payloads": list(checkpoint_records),
            },
            {
                "order": 2,
                "action": "run the RX2 terminal CPU identity race on the retained final checkpoint",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN identity-race owner",
                "consumer_store": "experiments/ddm_rx2_mc36_identity_race.py and the RX2 retained run store",
                "fire_trigger": "endpoint refit and both checkpoint SHA-256 custody records pass",
            },
            {
                "order": 3,
                "action": "recompile the QS2 and RE1 micro-edits against the identity-race-selected final coder",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN micro-edit composition owner",
                "consumer_store": "final-coder micro-edit compilation receipt and retained candidate archive store",
                "fire_trigger": "identity race selects a byte-identical receiver-valid final coder",
            },
            {
                "order": 4,
                "action": "evaluate one composed T4 row for task #1058",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN T4 auth-eval owner",
                "consumer_store": "canonical exact-row receipt store and frontier pointer adjudication",
                "fire_trigger": "micro-edit recompile emits a retained receiver-valid archive and deterministic repeat",
            },
        ],
        "score_claim": False,
        "paid_or_scorer_work_launched": False,
    }


def _terminal_note(status: str, run_root: Path, fire_order_path: Path | None) -> str:
    next_text = (
        f" MAIN should adjudicate the typed #1058 chain at {fire_order_path}."
        if fire_order_path is not None
        else " No downstream chain was emitted."
    )
    return (
        f"PUSH NOTIFICATION: local HPAC endpoint closer status={status} for {run_root}."
        f"{next_text} The closer launched no paid or scorer work.\n"
    )


def _write_terminal(output_dir: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    validate_receipt(receipt)
    receipt_path = output_dir / RECEIPT_NAME
    atomic_json(receipt_path, receipt)
    done = {
        "schema": "local_hpac_endpoint_closure_done.v1",
        "generated_utc": _utc_now(),
        "status": receipt["status"],
        "process_rc": receipt["process_rc"],
        "receipt": file_record(receipt_path),
    }
    atomic_json(output_dir / "ENDPOINT_CLOSURE.done.json", done)
    return receipt


def validate_receipt(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise LocalEndpointCloseError("invalid local endpoint closure receipt schema")
    required = {
        "generated_utc",
        "status",
        "process_rc",
        "run_root",
        "source_done_receipt",
        "retained_source_done_receipt",
        "source_launch_manifest",
        "fit",
        "payloads",
        "next_fire_order",
        "terminal_note",
        "errors",
        "score_claim",
        "paid_or_scorer_work_launched",
    }
    missing = required - set(receipt)
    if missing:
        raise LocalEndpointCloseError(f"local endpoint closure receipt missing {sorted(missing)}")
    if not isinstance(receipt["process_rc"], int):
        raise LocalEndpointCloseError("local endpoint closure process_rc must be int")
    allowed_statuses = {"CLOSED", "SOURCE_FAILED", "REFUSED_CHECKPOINT_OR_REFIT_CUSTODY"}
    if receipt["status"] not in allowed_statuses:
        raise LocalEndpointCloseError(f"local endpoint closure status is unknown: {receipt['status']!r}")
    if not isinstance(receipt["payloads"], list) or not isinstance(receipt["errors"], list):
        raise LocalEndpointCloseError("local endpoint closure payloads/errors must be lists")
    if receipt["score_claim"] is not False or receipt["paid_or_scorer_work_launched"] is not False:
        raise LocalEndpointCloseError("local endpoint closure containment fields must remain false")
    if receipt["status"] == "CLOSED":
        if receipt["process_rc"] != 0 or not receipt["payloads"] or receipt["next_fire_order"] is None:
            raise LocalEndpointCloseError("closed endpoint receipt lacks payload or fire-order custody")
    elif receipt["next_fire_order"] is not None:
        raise LocalEndpointCloseError("non-closed endpoint receipt must not emit a downstream fire order")
    return receipt


def execute_closure(*, run_root: Path, done_receipt_path: Path, output_dir: Path) -> dict[str, Any]:
    prior_path = output_dir / RECEIPT_NAME
    if prior_path.is_file():
        return validate_receipt(_read_object(prior_path))
    done = _read_object(done_receipt_path)
    manifest_path = _validate_source_done(done, run_root)
    retained_done_path = output_dir / "SOURCE_DONE_RECEIPT.json"
    atomic_bytes(retained_done_path, done_receipt_path.read_bytes())
    if _sha256(retained_done_path) != _sha256(done_receipt_path):
        raise LocalEndpointCloseError("retained source done receipt differs bytewise")
    base: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "generated_utc": _utc_now(),
        "status": "",
        "process_rc": 0,
        "run_root": str(run_root),
        "source_done_receipt": file_record(done_receipt_path),
        "retained_source_done_receipt": file_record(retained_done_path),
        "source_launch_manifest": file_record(manifest_path),
        "fit": None,
        "payloads": [],
        "next_fire_order": None,
        "terminal_note": None,
        "errors": [],
        "score_claim": False,
        "paid_or_scorer_work_launched": False,
    }
    if int(done.get("rc", -1)) != 0:
        base["status"] = "SOURCE_FAILED"
        base["process_rc"] = int(done.get("rc", 1))
        base["errors"].append(f"source detached process rc={done.get('rc')}")
        note_path = output_dir / "TERMINAL_NOTE.md"
        atomic_text(note_path, _terminal_note(base["status"], run_root, None))
        base["terminal_note"] = file_record(note_path)
        return _write_terminal(output_dir, base)
    manifest = _read_object(manifest_path)
    try:
        fit = _run_fitter(run_root, output_dir)
        checkpoints = _checkpoint_records(manifest, run_root)
        order = _next_fire_order(run_root=run_root, checkpoint_records=checkpoints, fit=fit)
        order_path = output_dir / "NEXT_FIRE_ORDER.json"
        atomic_json(order_path, order)
        note_path = output_dir / "TERMINAL_NOTE.md"
        atomic_text(note_path, _terminal_note("CLOSED", run_root, order_path))
        base.update(
            {
                "status": "CLOSED",
                "fit": fit,
                "payloads": checkpoints,
                "next_fire_order": file_record(order_path),
                "terminal_note": file_record(note_path),
            }
        )
    except (LocalEndpointCloseError, OSError, ValueError) as exc:
        base["status"] = "REFUSED_CHECKPOINT_OR_REFIT_CUSTODY"
        base["process_rc"] = 2
        base["errors"].append(f"{type(exc).__name__}: {exc}")
        note_path = output_dir / "TERMINAL_NOTE.md"
        atomic_text(note_path, _terminal_note(base["status"], run_root, None))
        base["terminal_note"] = file_record(note_path)
    print((output_dir / "TERMINAL_NOTE.md").read_text(encoding="utf-8"), end="", flush=True)
    return _write_terminal(output_dir, base)


def wait_and_close(
    *,
    run_root: Path,
    done_receipt_path: Path,
    output_dir: Path,
    deadline_s: float,
    poll_s: float,
    once: bool,
) -> dict[str, Any] | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    armed = {
        "schema": "local_hpac_endpoint_closer_armed.v1",
        "generated_utc": _utc_now(),
        "run_root": str(run_root),
        "done_receipt_path": str(done_receipt_path),
        "deadline_s": deadline_s,
        "poll_s": poll_s,
        "resumable_by": "rerun the same argv; terminal receipt is idempotent",
        "closer_tool": file_record(Path(__file__).resolve()),
    }
    atomic_json(output_dir / "ARMED.json", armed)
    started = time.monotonic()
    while True:
        if done_receipt_path.is_file():
            try:
                candidate = _read_object(done_receipt_path)
            except LocalEndpointCloseError:
                candidate = {}
            if candidate.get("schema") == DONE_SCHEMA:
                return execute_closure(
                    run_root=run_root,
                    done_receipt_path=done_receipt_path,
                    output_dir=output_dir,
                )
            if candidate.get("schema") not in {ARM_SCHEMA, None}:
                raise LocalEndpointCloseError(
                    f"done-receipt path contains unexpected schema {candidate.get('schema')!r}"
                )
        elapsed = time.monotonic() - started
        state = {
            "schema": "local_hpac_endpoint_closer_poll_state.v1",
            "generated_utc": _utc_now(),
            "status": "ARMED_WAITING",
            "elapsed_s": elapsed,
            "source_deadline_is_process_failure": False,
        }
        atomic_json(output_dir / "POLL_STATE.json", state)
        if once or elapsed >= deadline_s:
            if elapsed >= deadline_s:
                state["status"] = "BOUNDED_PENDING_REARM"
                atomic_json(output_dir / "POLL_STATE.json", state)
            return None
        time.sleep(poll_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--done-receipt-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deadline-s", type=float, default=12 * 3600)
    parser.add_argument("--poll-s", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.deadline_s < 0 or args.poll_s <= 0:
        parser.error("--deadline-s must be nonnegative and --poll-s must be positive")
    run_root = args.run_root.expanduser().resolve(strict=False)
    done = args.done_receipt_path.expanduser().resolve(strict=False)
    output = args.output_dir.expanduser().resolve(strict=False)
    try:
        receipt = wait_and_close(
            run_root=run_root,
            done_receipt_path=done,
            output_dir=output,
            deadline_s=args.deadline_s,
            poll_s=args.poll_s,
            once=args.once,
        )
    except LocalEndpointCloseError as exc:
        print(json.dumps({"error": str(exc), "schema": RECEIPT_SCHEMA}), file=sys.stderr)
        return 2
    if receipt is None:
        print(_canonical_json(_read_object(output / "POLL_STATE.json")), end="")
        return 0
    print(_canonical_json(receipt), end="")
    return int(receipt["process_rc"])


if __name__ == "__main__":
    raise SystemExit(main())
