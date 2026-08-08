#!/usr/bin/env python3
"""DDM seal orchestrator - ticket-driven, idempotent, fail-closed seal-gate walker.

WHY THIS EXISTS
---------------
The M1 seal chain (mem-probe -> per-key fire guard -> sigma runs -> sigma harvest ->
independent review passes -> fire guard -> FIRE) was hand-choreographed across shell
heredocs, hand-typed argv, and a human remembering the 6h receipt-freshness window.
That is a person-shaped dependency inside an apparatus whose standing discipline is
"the controller remembers and surfaces; the operator never has to"
(CLAUDE.md: '"Off" is a tracked queue, never a forgotten default').

This tool makes the LAUNCH TICKET the single source of every command. It never
re-types an argv: each gate executes ``ticket[argv_key]`` verbatim, which is the same
list the fire guard evaluated, so guard-evaluated config and executed config cannot
drift.

CONTRACT
--------
* IDEMPOTENT  - a gate whose receipt exists and passes is SATISFIED and is skipped.
* RESUMABLE   - state lives entirely in on-disk receipts; a crashed walk re-walks.
* FAIL-CLOSED - a missing/failed/stale receipt is a typed blocker, never a silent skip.
* CONTAINED   - the FIRE gate is reported, never executed here (heavy Metal launches
                stay an explicit MAIN/operator action per the governor discipline).
* HONEST      - the sigma harvest refuses to compare a metric against a bar in
                different units; it emits UNIT_MISMATCH plus the resolving command.

Usage
-----
    ddm_seal_orchestrator.py --ticket <ticket.json> --status [--json]
    ddm_seal_orchestrator.py --ticket <ticket.json> --run [--gate NAME] [--max-gates N]

Exit codes: 0 all reachable gates satisfied / ran clean; 3 blocked; 4 gate failure.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mx1_fire_guard import RECEIPT_FRESHNESS_WINDOW_SECONDS  # noqa: E402

SATISFIED = "satisfied"
PENDING = "pending"
BLOCKED = "blocked"
MANUAL = "manual"

# Gate kinds
KIND_MEM_PROBE = "mem_probe"
KIND_GUARD = "guard"
KIND_TRAIN = "train"
KIND_VERDICT = "dseg_verdict"
KIND_HARVEST = "sigma_harvest"
KIND_REVIEW = "review_counter"
KIND_FIRE = "fire"


class TicketError(RuntimeError):
    """The ticket cannot answer a question the orchestrator must not guess."""


@dataclass(frozen=True)
class Gate:
    name: str
    kind: str
    detail: str
    argv_key: str | None = None
    command_key: str | None = None
    receipt: Path | None = None
    depends_on: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateState:
    gate: Gate
    status: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- ticket


def load_ticket(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TicketError(f"ticket not found: {path}")
    return json.loads(path.read_text())


def argv_flag(argv: list[str], flag: str) -> str | None:
    """Read a flag value from the ticket's own argv (never invent one)."""
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def receipt_age_seconds(path: Path) -> float:
    return max(0.0, time.time() - path.stat().st_mtime)


# ----------------------------------------------------------------------- gate build


def build_gates(ticket: dict[str, Any], ticket_path: Path) -> list[Gate]:
    """Derive the gate DAG from the ticket itself - no hardcoded run layout."""
    gates: list[Gate] = []
    receipts = ticket.get("mem_probe_receipt_paths") or {}
    singular = ticket.get("mem_probe_receipt_path")

    # 1. memory probes, one per distinct receipt the ticket declares.
    probe_commands = {
        "mem_probe_fp16": "mem_probe_command",
        "mem_probe_fp32": "mem_probe_command_fp32",
    }
    probe_gate_for_receipt: dict[str, str] = {}
    for gate_name, command_key in probe_commands.items():
        command = ticket.get(command_key)
        if not command:
            continue
        out = argv_flag(command, "--out")
        if out is None:
            raise TicketError(f"{command_key} carries no --out; cannot locate its receipt")
        receipt = Path(out).parent / "mem_probe_receipt.json"
        gates.append(
            Gate(
                name=gate_name,
                kind=KIND_MEM_PROBE,
                detail=f"memory probe at the real config ({command_key})",
                command_key=command_key,
                receipt=receipt,
            )
        )
        probe_gate_for_receipt[str(receipt)] = gate_name

    # 2/3. per-sigma-run guard + train pairs, ordered by argv key.
    sigma_keys = sorted(k for k in ticket if k.startswith("argv_sigma_"))
    for key in sigma_keys:
        argv = ticket[key]
        verdict = argv_flag(argv, "--fire-guard-verdict")
        out = argv_flag(argv, "--out")
        if verdict is None or out is None:
            raise TicketError(f"{key} lacks --fire-guard-verdict/--out; cannot gate it")
        receipt_path = receipts.get(key, singular)
        depends = (probe_gate_for_receipt[str(Path(receipt_path))],) if (
            receipt_path and str(Path(receipt_path)) in probe_gate_for_receipt
        ) else ()
        guard_name = f"guard::{key}"
        gates.append(
            Gate(
                name=guard_name,
                kind=KIND_GUARD,
                detail=f"fire guard for {key}",
                argv_key=key,
                receipt=Path(verdict),
                depends_on=depends,
            )
        )
        gates.append(
            Gate(
                name=f"run::{key}",
                kind=KIND_TRAIN,
                detail=f"governed sigma run {key}",
                argv_key=key,
                receipt=Path(out),
                depends_on=(guard_name,),
            )
        )

    # 3b. d_seg-unit verdicts (CPU-torch authority units - the falsifier bars speak d_seg).
    verdict_keys = sorted(k for k in ticket if k.startswith("argv_dseg_verdict_"))
    for key in verdict_keys:
        out = argv_flag(ticket[key], "--out")
        if out is None:
            raise TicketError(f"{key} lacks --out; cannot locate its verdict result")
        gates.append(
            Gate(
                name=f"dseg::{key}",
                kind=KIND_VERDICT,
                detail=f"CPU-torch d_seg verdict ({key})",
                argv_key=key,
                receipt=Path(out),
                depends_on=tuple(f"run::{k}" for k in sigma_keys),
            )
        )

    # 4. sigma harvest.
    if sigma_keys:
        fp16 = [k for k in sigma_keys if "fp32" not in k]
        fp32 = [k for k in sigma_keys if "fp32" in k]
        harvest_receipt = ticket_path.parent / "sigma" / "sigma_harvest_receipt.json"
        gates.append(
            Gate(
                name="sigma_harvest",
                kind=KIND_HARVEST,
                detail="determinism proof + fp16/fp32 d_seg delta -> ticket fields + falsifiers",
                receipt=harvest_receipt,
                depends_on=tuple(f"run::{k}" for k in sigma_keys)
                + tuple(f"dseg::{k}" for k in verdict_keys),
                extra={"fp16_keys": fp16, "fp32_keys": fp32, "verdict_keys": verdict_keys},
            )
        )

    # 5. independent review counter (gc21 discipline: any finding resets to 0).
    gates.append(
        Gate(
            name="review_passes",
            kind=KIND_REVIEW,
            detail="independent adversarial review, 3 consecutive clean passes",
            depends_on=("sigma_harvest",) if sigma_keys else (),
            extra={"required": int(ticket.get("review_passes_required", 3))},
        )
    )

    # 6. the fire gate: reported, never executed by this tool.
    fire_key = ticket.get("fire_argv_key", "argv_m1_n120_cap_saturated")
    if fire_key in ticket:
        argv = ticket[fire_key]
        verdict = argv_flag(argv, "--fire-guard-verdict")
        gates.append(
            Gate(
                name=f"guard::{fire_key}",
                kind=KIND_GUARD,
                detail=f"fire guard for the burn ({fire_key})",
                argv_key=fire_key,
                receipt=Path(verdict) if verdict else None,
                depends_on=("review_passes",),
            )
        )
        gates.append(
            Gate(
                name="FIRE",
                kind=KIND_FIRE,
                detail="the n120 burn - MAIN/operator action, never auto-executed",
                argv_key=fire_key,
                depends_on=(f"guard::{fire_key}",),
            )
        )
    return gates


# ------------------------------------------------------------------------ evaluate


def unmet_dependencies(gate: Gate, states: dict[str, GateState]) -> list[str]:
    """Dependencies not proven SATISFIED — fail-closed on ABSENCE.

    The prior form was ``states.get(d) and states[d].status != SATISFIED``: a
    dependency with NO state yet is falsy, so an UNEVALUATED dependency counted
    as met. Absent == pass is the vacuity genus. A dependency we have not
    evaluated is UNMET.
    """
    return [d for d in gate.depends_on
            if states.get(d) is None or states[d].status != SATISFIED]


def evaluate_gate(gate: Gate, ticket: dict[str, Any], states: dict[str, GateState]) -> GateState:
    unmet = unmet_dependencies(gate, states)
    if gate.kind == KIND_FIRE:
        if unmet:
            return GateState(gate, MANUAL, f"held: {', '.join(unmet)} not satisfied")
        return GateState(gate, MANUAL, "READY - run the ticket argv from MAIN (one Metal fire)")

    if gate.kind == KIND_MEM_PROBE:
        state = _eval_mem_probe(gate)
    elif gate.kind == KIND_GUARD:
        state = _eval_receipt_status(gate, "status", ("passed",))
    elif gate.kind == KIND_TRAIN:
        state = _eval_train(gate)
    elif gate.kind == KIND_VERDICT:
        state = _eval_verdict(gate)
    elif gate.kind == KIND_HARVEST:
        state = _eval_harvest(gate, ticket)
    elif gate.kind == KIND_REVIEW:
        state = _eval_review(gate, ticket)
    else:
        return GateState(gate, BLOCKED, f"unknown gate kind {gate.kind!r}")

    # NO gate may REPORT satisfied while its own dependencies are unmet.
    # `unmet` was computed for every kind but consulted ONLY by KIND_FIRE, so a
    # stale burn-guard receipt (status=passed on disk) satisfied guard::<key>
    # while review_passes sat at 0/3 — and FIRE, whose only dependency IS that
    # guard, then read READY. That is a fire-gate bypass at zero clean passes
    # (M1R4C-F1, 2026-08-08). PENDING not BLOCKED: the dependency may yet be
    # satisfied, and BLOCKED breaks the evaluation loop.
    if unmet and state.status == SATISFIED:
        return GateState(
            gate,
            PENDING,
            f"held: {', '.join(unmet)} not satisfied (receipt says {state.reason!r})",
        )
    return state


def _eval_mem_probe(gate: Gate) -> GateState:
    receipt = gate.receipt
    if receipt is None or not receipt.exists():
        return GateState(gate, PENDING, "no receipt on disk")
    payload = read_json(receipt) or {}
    status = payload.get("status")
    if status != "passed":
        return GateState(gate, BLOCKED, f"receipt status={status!r}", {"receipt": str(receipt)})
    age = receipt_age_seconds(receipt)
    if age > RECEIPT_FRESHNESS_WINDOW_SECONDS:
        return GateState(
            gate,
            PENDING,
            f"receipt stale ({age / 3600:.2f}h > {RECEIPT_FRESHNESS_WINDOW_SECONDS / 3600:.0f}h) - re-probe",
            {"receipt": str(receipt), "age_seconds": age},
        )
    peak = (payload.get("peak") or {})
    return GateState(
        gate,
        SATISFIED,
        f"passed, age {age / 3600:.2f}h",
        {
            "receipt": str(receipt),
            "peak_mlx_reported_gib": peak.get("peak_mlx_reported"),
            "peak_rss_gib": peak.get("peak_rss"),
        },
    )


def _eval_receipt_status(gate: Gate, key: str, ok: tuple[str, ...]) -> GateState:
    receipt = gate.receipt
    if receipt is None or not receipt.exists():
        return GateState(gate, PENDING, "no verdict on disk")
    payload = read_json(receipt) or {}
    value = payload.get(key)
    if value in ok:
        return GateState(gate, SATISFIED, f"{key}={value}", {"receipt": str(receipt)})
    return GateState(
        gate,
        PENDING,
        f"{key}={value!r} - re-evaluate",
        {"receipt": str(receipt), "reason_code": payload.get("reason_code")},
    )


def _eval_train(gate: Gate) -> GateState:
    receipt = gate.receipt
    if receipt is None or not receipt.exists():
        return GateState(gate, PENDING, "no result.json on disk")
    payload = read_json(receipt) or {}
    train = payload.get("mlx_train") or {}
    status = train.get("status")
    if status != "passed":
        return GateState(gate, BLOCKED, f"mlx_train.status={status!r}", {"receipt": str(receipt)})
    history = train.get("history") or []
    return GateState(
        gate,
        SATISFIED,
        f"passed, {len(history)} steps",
        {"receipt": str(receipt), "seconds_per_step": train.get("seconds_per_step")},
    )


def _verdict_d_seg(result_path: Path) -> float | None:
    """Pull aggregate_d_seg out of a CPU-torch verdict result (authority units)."""
    verdict = (read_json(result_path) or {}).get("torch_verdict") or {}
    # torch-verdict emits aggregate_d_seg at the top level; the facets path nests it in rows.
    value = verdict.get("aggregate_d_seg")
    if not isinstance(value, (int, float)):
        rows = verdict.get("checkpoint_rows") or []
        value = rows[-1].get("aggregate_d_seg") if rows else None
    return float(value) if isinstance(value, (int, float)) else None


def _eval_verdict(gate: Gate) -> GateState:
    receipt = gate.receipt
    if receipt is None or not receipt.exists():
        return GateState(gate, PENDING, "no verdict result on disk")
    d_seg = _verdict_d_seg(receipt)
    if d_seg is None:
        return GateState(gate, BLOCKED, "result carries no aggregate_d_seg row", {"receipt": str(receipt)})
    return GateState(gate, SATISFIED, f"d_seg={d_seg:.9g}", {"receipt": str(receipt), "d_seg": d_seg})


def _final_loss(result_path: Path) -> float | None:
    payload = read_json(result_path) or {}
    history = ((payload.get("mlx_train") or {}).get("history")) or []
    if not history:
        return None
    last = history[-1]
    value = last.get("loss")
    return float(value) if isinstance(value, (int, float)) else None


def _harvest_input_paths(gate: Gate, ticket: dict[str, Any]) -> list[Path]:
    keys = list(gate.extra.get("fp16_keys", [])) + list(gate.extra.get("fp32_keys", []))
    keys += list(gate.extra.get("verdict_keys", []))
    paths = []
    for key in keys:
        out = argv_flag(ticket.get(key) or [], "--out")
        if out:
            paths.append(Path(out))
    return paths


def _eval_harvest(gate: Gate, ticket: dict[str, Any]) -> GateState:
    receipt = gate.receipt
    payload = read_json(receipt) if receipt else None
    if payload is None:
        return GateState(gate, PENDING, "sigma not harvested yet")
    # Freshness AT CONSUMPTION: a harvest older than any of its inputs is stale,
    # not terminal - otherwise a stale UNEVALUABLE verdict outlives the evidence
    # that would resolve it.
    assert receipt is not None
    harvest_mtime = receipt.stat().st_mtime
    newer = [p for p in _harvest_input_paths(gate, ticket) if p.exists() and p.stat().st_mtime > harvest_mtime]
    if newer:
        return GateState(
            gate,
            PENDING,
            f"stale: {len(newer)} input receipt(s) newer than the harvest - re-harvest",
            {"newer_inputs": [str(p) for p in newer[:4]]},
        )
    calibration = (ticket.get("sigma_calibration") or {})
    if calibration.get("sanity_sigma_measured") in (None, {}):
        return GateState(gate, PENDING, "harvest receipt exists but ticket fields unfilled")
    falsifiers = payload.get("falsifiers") or {}
    fired = [name for name, row in falsifiers.items() if row.get("fired") is True]
    if fired:
        return GateState(gate, BLOCKED, f"seal falsifier fired: {', '.join(fired)}", {"receipt": str(receipt)})
    unresolved = [name for name, row in falsifiers.items() if row.get("fired") is None]
    if unresolved:
        return GateState(
            gate,
            BLOCKED,
            f"falsifier unevaluable: {', '.join(unresolved)}",
            {"receipt": str(receipt), "resolving_commands": payload.get("resolving_commands")},
        )
    return GateState(gate, SATISFIED, "sigma measured, no falsifier fired", {"receipt": str(receipt)})


def _eval_review(gate: Gate, ticket: dict[str, Any]) -> GateState:
    required = int(gate.extra.get("required", 3))
    rounds = ticket.get("review_passes") or []
    clean = 0
    for row in rounds:  # any finding resets the counter (gc21 discipline)
        clean = clean + 1 if int(row.get("findings", 1)) == 0 else 0
    if clean >= required:
        return GateState(gate, SATISFIED, f"{clean}/{required} consecutive clean passes")
    return GateState(
        gate,
        PENDING,
        f"{clean}/{required} clean passes - spawn an independent review over the sealed ticket",
        {"rounds_recorded": len(rounds)},
    )


# ------------------------------------------------------------------------- execute


def execute_gate(gate: Gate, ticket: dict[str, Any], ticket_path: Path, verbose: bool) -> tuple[int, str]:
    if gate.kind == KIND_MEM_PROBE:
        command = ticket[gate.command_key]
        return _run(command, verbose), f"ran {gate.command_key}"
    if gate.kind == KIND_GUARD:
        command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "mx1_fire_guard.py"),
            "--ticket",
            str(ticket_path),
            "--argv-key",
            str(gate.argv_key),
        ]
        if gate.receipt is not None:
            gate.receipt.parent.mkdir(parents=True, exist_ok=True)
            command += ["--out", str(gate.receipt)]
        return _run(command, verbose), f"evaluated guard for {gate.argv_key}"
    if gate.kind in (KIND_TRAIN, KIND_VERDICT):
        return _run(ticket[gate.argv_key], verbose), f"ran {gate.argv_key}"
    if gate.kind == KIND_HARVEST:
        return harvest_sigma(gate, ticket, ticket_path), "harvested sigma"
    if gate.kind in (KIND_REVIEW, KIND_FIRE):
        return 0, "manual gate - not executed by the orchestrator"
    return 4, f"unknown gate kind {gate.kind!r}"


def _run(command: list[str], verbose: bool) -> int:
    if verbose:
        print(f"    $ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=str(REPO_ROOT)).returncode


def _falsifier(*, bar: float, value: float | None, fires_when_ge: bool, unit: str, basis: str) -> dict[str, Any]:
    """A falsifier row. `fired=None` means UNEVALUABLE - never silently 'passed'."""
    if value is None:
        return {
            "bar": bar,
            "unit": unit,
            "measured": None,
            "fired": None,
            "reason": f"UNEVALUABLE: {basis}",
        }
    fired = (value >= bar) if fires_when_ge else (value < bar)
    return {
        "bar": bar,
        "unit": unit,
        "measured": value,
        "fired": bool(fired),
        "reason": f"{'FIRED' if fired else 'clear'}: {basis}",
    }


def harvest_sigma(gate: Gate, ticket: dict[str, Any], ticket_path: Path) -> int:
    """Compute sigma over the repeats, fill the ticket, evaluate seal falsifiers.

    The metric available in an mlx-train receipt is the final-step TRAINING LOSS.
    The gc21 fp16 guard bar and the event-predicate marginal bar are stated in
    d_seg units. Comparing them would launder a unit mismatch into a verdict, so
    any bar whose unit differs is emitted UNEVALUABLE with its resolving command.
    """
    fp16_losses: dict[str, float] = {}
    for key in gate.extra["fp16_keys"]:
        out = argv_flag(ticket[key], "--out")
        value = _final_loss(Path(out)) if out else None
        if value is None:
            print(f"[seal] missing final loss for {key}", file=sys.stderr)
            return 4
        fp16_losses[key] = value
    fp32_losses = {}
    for key in gate.extra["fp32_keys"]:
        out = argv_flag(ticket[key], "--out")
        value = _final_loss(Path(out)) if out else None
        if value is not None:
            fp32_losses[key] = value

    values = list(fp16_losses.values())
    mean = statistics.fmean(values)
    sigma = statistics.stdev(values) if len(values) > 1 else 0.0
    delta = (
        abs(mean - statistics.fmean(list(fp32_losses.values()))) if fp32_losses else None
    )

    calibration = ticket.setdefault("sigma_calibration", {})
    measured = {
        "metric": "final_step_training_loss",
        "unit": "training_loss",
        "scope": "same-seed backend nondeterminism floor (n=%d repeats, identical config)" % len(values),
        "n": len(values),
        "mean": mean,
        "sigma": sigma,
        "relative_sigma": (sigma / mean) if mean else None,
        "per_run": fp16_losses,
    }
    calibration["sanity_sigma_measured"] = measured
    calibration["fp16_fp32_delta_measured"] = {
        "metric": "final_step_training_loss",
        "unit": "training_loss",
        "fp16_mean": mean,
        "fp32": fp32_losses,
        "abs_delta": delta,
        "delta_in_sigma": (delta / sigma) if (delta is not None and sigma > 0) else None,
    }

    # DETERMINISM PROOF: identical checkpoint bytes across repeats => sigma is EXACTLY 0
    # in every checkpoint-derived metric (d_seg included), because the CPU-torch verdict
    # is a deterministic function of those bytes. This is a proof, not a unit transfer.
    ckpt_shas: dict[str, str | None] = {}
    for key in gate.extra["fp16_keys"]:
        out = argv_flag(ticket[key], "--out")
        payload = read_json(Path(out)) if out else None
        ckpt_shas[key] = ((payload or {}).get("mlx_train") or {}).get("latest_checkpoint_sha256")
    distinct = {sha for sha in ckpt_shas.values() if sha}
    deterministic = len(distinct) == 1 and len(ckpt_shas) == len(gate.extra["fp16_keys"])
    measured["checkpoint_sha256_per_run"] = ckpt_shas
    measured["repeat_determinism"] = {
        "bit_identical_checkpoints": deterministic,
        "distinct_sha_count": len(distinct),
        "implies_sigma_zero_in_all_checkpoint_derived_metrics": deterministic,
    }

    # d_seg-unit adjudication from the CPU-torch verdicts (authority units).
    dseg: dict[str, float | None] = {}
    for key in gate.extra.get("verdict_keys", []):
        out = argv_flag(ticket[key], "--out")
        dseg[key] = _verdict_d_seg(Path(out)) if out else None
    fp16_dseg = next((v for k, v in dseg.items() if "fp32" not in k), None)
    fp32_dseg = next((v for k, v in dseg.items() if "fp32" in k), None)
    dseg_delta = abs(fp16_dseg - fp32_dseg) if (fp16_dseg is not None and fp32_dseg is not None) else None
    calibration["dseg_unit_measurement"] = {
        "unit": "d_seg",
        "authority": "frozen CPU-torch verdict over the run checkpoints",
        "fp16": fp16_dseg,
        "fp32": fp32_dseg,
        "abs_delta": dseg_delta,
        "sigma_repeat": 0.0 if deterministic else None,
    }

    falsifiers = {
        "F1_sigma_below_fp16_guard_bar": _falsifier(
            bar=2.0e-6,
            value=0.0 if deterministic else None,
            fires_when_ge=True,
            unit="d_seg",
            basis="repeat sigma = 0 by bit-identical checkpoints" if deterministic
            else "repeats are NOT bit-identical; per-repeat d_seg verdicts required",
        ),
        "F2_fp16_fp32_delta_within_envelope": _falsifier(
            bar=2.0e-6,
            value=dseg_delta,
            fires_when_ge=True,
            unit="d_seg",
            basis="|d_seg(fp16) - d_seg(fp32)| vs max(2.0e-6, 3*sigma); sigma=0 so the bar is 2.0e-6",
        ),
        "F3_sigma_below_event_threshold": _falsifier(
            bar=2.5e-6,
            value=0.0 if deterministic else None,
            fires_when_ge=True,
            unit="d_seg",
            basis="same determinism proof; event thresholds are objective/d_seg-denominated",
        ),
    }
    receipt = {
        "schema": "ddm_seal_sigma_harvest.v2_dseg_adjudicated",
        "ticket": str(ticket_path),
        "measured": measured,
        "fp16_fp32_delta": calibration["fp16_fp32_delta_measured"],
        "dseg_unit_measurement": calibration["dseg_unit_measurement"],
        "falsifiers": falsifiers,
        "resolving_commands": {
            "d_seg_unit_verdicts": "ticket keys argv_dseg_verdict_fp16 / argv_dseg_verdict_fp32",
        },
        "scope": (
            "measured at the CALIBRATION horizon (the sigma runs' step count), not at the "
            "full burn horizon; the in-loop fp16 fallback guard remains the live protection "
            "for the burn itself"
        ),
        "axis": "[macOS-CPU advisory] for d_seg rows; [macOS-MLX research-signal] for loss rows",
        "score_claim": False,
    }
    assert gate.receipt is not None
    gate.receipt.parent.mkdir(parents=True, exist_ok=True)
    gate.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    ticket_path.write_text(json.dumps(ticket, indent=1, sort_keys=True) + "\n")
    print(
        f"[seal] sigma n={len(values)} mean={mean:.9g} sigma={sigma:.3g} "
        f"rel={measured['relative_sigma']:.3g}"
        if measured["relative_sigma"] is not None
        else f"[seal] sigma n={len(values)} mean={mean:.9g} sigma={sigma:.3g}"
    )
    return 0


# ------------------------------------------------------------------------- walking


def walk(ticket_path: Path, run: bool, only: str | None, max_gates: int, verbose: bool) -> dict[str, Any]:
    ticket = load_ticket(ticket_path)
    gates = build_gates(ticket, ticket_path)
    states: dict[str, GateState] = {}
    executed: list[str] = []
    blocked: list[str] = []

    for gate in gates:
        state = evaluate_gate(gate, ticket, states)
        manual_kind = gate.kind in (KIND_REVIEW, KIND_FIRE)
        if run and state.status == PENDING and not manual_kind and (only is None or gate.name == only):
            unmet = [d for d in gate.depends_on if states.get(d, GateState(gate, PENDING, "")).status != SATISFIED]
            if unmet:
                state = GateState(gate, BLOCKED, f"dependency not satisfied: {', '.join(unmet)}")
            elif len(executed) >= max_gates:
                state = GateState(gate, PENDING, "max-gates budget reached")
            else:
                if verbose:
                    print(f"[seal] running {gate.name}: {gate.detail}", flush=True)
                rc, note = execute_gate(gate, ticket, ticket_path, verbose)
                executed.append(gate.name)
                if rc != 0:
                    state = GateState(gate, BLOCKED, f"{note} -> rc={rc}")
                else:
                    ticket = load_ticket(ticket_path)  # gates may amend the ticket
                    state = evaluate_gate(gate, ticket, states)
        states[gate.name] = state
        if state.status == BLOCKED:
            blocked.append(gate.name)
            break  # fail-closed: never walk past a blocker

    return {
        "ticket": str(ticket_path),
        "gates": [
            {
                "name": s.gate.name,
                "kind": s.gate.kind,
                "status": s.status,
                "reason": s.reason,
                "detail": s.gate.detail,
                "evidence": s.evidence,
            }
            for s in states.values()
        ],
        "executed": executed,
        "blocked": blocked,
        "all_satisfied": all(
            s.status in (SATISFIED, MANUAL) for s in states.values()
        ),
    }


ICONS = {SATISFIED: "OK  ", PENDING: "..  ", BLOCKED: "STOP", MANUAL: "HOLD"}


def render(report: dict[str, Any]) -> str:
    lines = [f"DDM SEAL ORCHESTRATOR  ticket={report['ticket']}", ""]
    for row in report["gates"]:
        lines.append(f"  [{ICONS.get(row['status'], '?')}] {row['name']:<34} {row['reason']}")
    if report["executed"]:
        lines += ["", f"  executed this walk: {', '.join(report['executed'])}"]
    if report["blocked"]:
        lines += ["", f"  BLOCKED at: {', '.join(report['blocked'])}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticket", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="execute pending gates in order")
    parser.add_argument("--gate", help="restrict execution to a single gate name")
    parser.add_argument("--max-gates", type=int, default=64)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = walk(args.ticket, args.run, args.gate, args.max_gates, not args.quiet)
    except TicketError as exc:
        print(f"[seal] TICKET ERROR: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(report, indent=1, sort_keys=True) if args.json else render(report))
    if report["blocked"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
