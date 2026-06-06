# SPDX-License-Identifier: MIT
"""Fail-closed launch gate for NeRV-family long MLX training runs.

Evidence for a long run accumulates across many producers (birth receipts,
pose-trust rows, survival receipts, parse-back manifests, SNeRV source-forward
proofs).  Nothing may consume that pile and say "approved" by vibes.  This
module is the single machine arbiter: it scans a run root for typed evidence,
walks the family ladder (L2 physical birth -> L3 pose-trusted -> L4 same-action
survival -> L5 representative/inflate), and emits a machine-readable verdict
whose only approval path is every required row present and consistent.

It also DEFINES the survival/hysteresis/coverage schemas before their
producers exist: the gate is the contract; producers conform to it:

* ``hi_nerv_target_region_birth_survival.v1`` - re-measurement of the same
  accepted birth (matching ``action_id``) on one surface in
  {``fakequant_mlx``, ``parseback_mlx``, ``inflated_torch_cpu``} with
  ``survived: bool`` and target-support stats re-measured on that surface.
* ``hi_nerv_target_region_birth_hysteresis.v1`` - same ``action_id`` after M
  further steps with ``passed: bool`` (no hard-won collapse, debt not above
  pre-birth).
* ``hi_nerv_representative_region_coverage.v1`` - hard-region-miner coverage
  row proving birth beyond the birth-at-init easy case.

L4 survival is proof ONLY when the same action identity is traced through
surfaces: survival rows must CARRY the live receipt's ``action_id``; a
re-solved birth under fakequant is a different experiment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.analysis.receiver_surface_metrics import receiver_surface_target_support_breakdown
from tac.optimization.proxy_candidate_contract import (
    PROXY_FALSE_AUTHORITY_FIELDS,
    require_no_truthy_authority_fields,
)

NERV_LONG_RUN_LAUNCH_GATE_SCHEMA = "nerv_long_run_launch_gate.v1"
BIRTH_RECEIPT_SCHEMA = "hi_nerv_target_region_birth_receipt.v1"
BIRTH_SURVIVAL_SCHEMA = "hi_nerv_target_region_birth_survival.v1"
BIRTH_HYSTERESIS_SCHEMA = "hi_nerv_target_region_birth_hysteresis.v1"
REPRESENTATIVE_COVERAGE_SCHEMA = "hi_nerv_representative_region_coverage.v1"
SNERV_SOURCE_FORWARD_SCHEMA = "snerv_official_tub_source_forward_replay.v1"
SUPPORTED_FAMILIES = ("hinerv", "hi_nerv", "snerv")
SURVIVAL_SURFACES_L4 = ("fakequant_mlx", "parseback_mlx")
SURVIVAL_SURFACE_L5 = "inflated_torch_cpu"
FRONTIER_POINTER_MAX_AGE_HOURS = 24.0
_MAX_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024


class NervLongRunLaunchGateError(ValueError):
    """Raised when launch-gate inputs are structurally invalid."""


def _read_json(path: Path) -> Any | None:
    try:
        if path.stat().st_size > _MAX_EVIDENCE_FILE_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _iter_evidence_payloads(run_root: Path):
    for path in sorted(run_root.rglob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            yield path, payload
        # JSONL receipts: one payload per line.
    for path in sorted(run_root.rglob("*.jsonl")):
        try:
            if path.stat().st_size > _MAX_EVIDENCE_FILE_BYTES:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield path, payload
        except OSError:
            continue


def _walk_schema(payload: Any, schema: str, hits: list[dict[str, Any]]) -> None:
    if isinstance(payload, Mapping):
        if str(payload.get("schema") or "") == schema:
            hits.append(dict(payload))
        for value in payload.values():
            _walk_schema(value, schema, hits)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _walk_schema(value, schema, hits)


def _collect_schema_rows(
    run_root: Path,
    schema: str,
    *,
    index: dict[str, list[str]],
    blockers: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, payload in _iter_evidence_payloads(run_root):
        hits: list[dict[str, Any]] = []
        _walk_schema(payload, schema, hits)
        if not hits:
            continue
        try:
            require_no_truthy_authority_fields(payload, context=str(path))
        except Exception:
            blockers.append(f"evidence_truthy_authority:{path.name}")
            continue
        rows.extend(hits)
        index.setdefault(schema, []).append(path.as_posix())
    return rows


def _frontier_pointer_status(
    frontier_pointer: Path | None,
    *,
    now_utc: datetime,
) -> dict[str, Any]:
    if frontier_pointer is None or not frontier_pointer.is_file():
        return {"present": False, "fresh": False, "last_refreshed_utc": None}
    payload = _read_json(frontier_pointer)
    if not isinstance(payload, dict):
        return {"present": False, "fresh": False, "last_refreshed_utc": None}
    raw = str(payload.get("last_refreshed_utc") or "")
    try:
        refreshed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if refreshed.tzinfo is None:
            refreshed = refreshed.replace(tzinfo=UTC)
        age_hours = (now_utc - refreshed).total_seconds() / 3600.0
        fresh = math.isfinite(age_hours) and 0.0 <= age_hours <= (FRONTIER_POINTER_MAX_AGE_HOURS)
    except ValueError:
        refreshed = None
        fresh = False
    return {
        "present": True,
        "fresh": bool(fresh),
        "last_refreshed_utc": raw or None,
    }


def _accepted_live_birth(
    rows: list[dict[str, Any]],
    *,
    blockers: list[str],
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("surface") or "live_mlx") != "live_mlx":
            continue
        if int(row.get("accepted_step_count") or 0) <= 0:
            continue
        if not _target_support_positive(
            row,
            blocker="live_birth_target_support_missing",
            not_positive_blocker="live_birth_target_support_not_positive",
            blockers=blockers,
        ):
            continue
        if not row.get("action_id"):
            continue
        return row
    return None


def _pose_trusted(row: Mapping[str, Any]) -> bool:
    pose_guard = row.get("pose_guard") or {}
    nonrate = row.get("exact_nonrate") or {}
    delta = nonrate.get("delta_score_nonrate")
    return bool(
        pose_guard.get("available")
        and pose_guard.get("pose_input_contest_resolution")
        and nonrate.get("pose_term_available")
        and isinstance(delta, (int, float))
        and float(delta) < 0.0
    )


def _survival_rows_for_action(
    rows: list[dict[str, Any]],
    *,
    action_id: str,
    surface: str,
    blockers: list[str],
) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("surface") or "") != surface:
            continue
        if str(row.get("action_id") or "") != action_id:
            blockers.append(f"l4_survival_action_id_mismatch:{surface}")
            continue
        if row.get("survived") is True:
            if _target_support_positive(
                row,
                blocker=f"birth_survival_target_support_missing:{surface}",
                not_positive_blocker=(
                    f"birth_survival_target_support_not_positive:{surface}"
                ),
                blockers=blockers,
            ):
                return row
            continue
        blockers.append(f"birth_not_survived:{surface}")
    return None


def _target_support_positive(
    row: Mapping[str, Any],
    *,
    blocker: str,
    not_positive_blocker: str,
    blockers: list[str],
) -> bool:
    surface = dict(row)
    transitions = row.get("argmax_transitions")
    if isinstance(transitions, Mapping):
        surface.update(transitions)
    support = receiver_surface_target_support_breakdown(surface)
    hard_won = support.get("receiver_surface_target_hard_won_count")
    net = support.get("receiver_surface_net_target_support_delta")
    if hard_won is None or net is None:
        blockers.append(blocker)
        return False
    if float(hard_won) <= 0.0 or float(net) <= 0.0:
        blockers.append(not_positive_blocker)
        return False
    return True


def evaluate_nerv_long_run_launch_gate(
    *,
    family: str,
    run_root: str | Path,
    frontier_pointer: str | Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Return the fail-closed machine verdict for one family's long run."""

    family = _canonical_family(family)
    if family not in {"hinerv", "snerv"}:
        raise NervLongRunLaunchGateError(f"family must be one of {SUPPORTED_FAMILIES}; got {family!r}")
    root = Path(run_root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise NervLongRunLaunchGateError(f"run_root is not a directory: {root}")
    pointer_path = Path(frontier_pointer).expanduser().resolve(strict=False) if frontier_pointer is not None else None
    now = now_utc or datetime.now(UTC)

    blockers: list[str] = []
    warnings: list[str] = []
    evidence_index: dict[str, list[str]] = {}
    highest_level = "none"

    pointer = _frontier_pointer_status(pointer_path, now_utc=now)
    if not pointer["present"]:
        blockers.append("frontier_pointer_missing")
    elif not pointer["fresh"]:
        blockers.append("frontier_pointer_stale")

    if family == "hinerv":
        birth_rows = _collect_schema_rows(root, BIRTH_RECEIPT_SCHEMA, index=evidence_index, blockers=blockers)
        survival_rows = _collect_schema_rows(root, BIRTH_SURVIVAL_SCHEMA, index=evidence_index, blockers=blockers)
        hysteresis_rows = _collect_schema_rows(root, BIRTH_HYSTERESIS_SCHEMA, index=evidence_index, blockers=blockers)
        coverage_rows = _collect_schema_rows(
            root,
            REPRESENTATIVE_COVERAGE_SCHEMA,
            index=evidence_index,
            blockers=blockers,
        )
        live = _accepted_live_birth(birth_rows, blockers=blockers)
        if live is None:
            blockers.append("real_video_birth_receipt_missing")
        else:
            highest_level = "L2"
            if _pose_trusted(live):
                highest_level = "L3"
            else:
                blockers.append("pose_trusted_birth_receipt_missing")
            action_id = str(live.get("action_id"))
            l4_ok = highest_level == "L3"
            for surface in SURVIVAL_SURFACES_L4:
                row = _survival_rows_for_action(
                    survival_rows,
                    action_id=action_id,
                    surface=surface,
                    blockers=blockers,
                )
                if row is None:
                    blockers.append(f"birth_survival_receipt_missing:{surface}")
                    l4_ok = False
            hysteresis = next(
                (
                    row
                    for row in hysteresis_rows
                    if str(row.get("action_id") or "") == action_id and row.get("passed") is True
                ),
                None,
            )
            if hysteresis is None:
                blockers.append("hysteresis_receipt_missing")
                l4_ok = False
            if l4_ok:
                highest_level = "L4"
            l5_ok = l4_ok
            inflate_row = _survival_rows_for_action(
                survival_rows,
                action_id=action_id,
                surface=SURVIVAL_SURFACE_L5,
                blockers=blockers,
            )
            if inflate_row is None:
                blockers.append(f"birth_survival_receipt_missing:{SURVIVAL_SURFACE_L5}")
                l5_ok = False
            if not coverage_rows:
                blockers.append("representative_region_coverage_missing")
                warnings.append("birth_at_init_only")
                l5_ok = False
            if l5_ok:
                highest_level = "L5"
            delta_bytes = int(live.get("runtime_sidecar_bytes") or 0)
            if delta_bytes != 0:
                blockers.append("value_per_byte_ledger_missing")
    else:  # snerv
        proof_rows = _collect_schema_rows(root, SNERV_SOURCE_FORWARD_SCHEMA, index=evidence_index, blockers=blockers)
        proven = next(
            (row for row in proof_rows if row.get("full_tub_source_forward_parity_proven") is True),
            None,
        )
        if proven is None:
            blockers.append("snerv_full_source_forward_parity_missing")
        else:
            highest_level = "L3"
        bitflip = next(
            (
                row
                for row in proof_rows
                if row.get("bitflip_section") and row.get("proof_passed") is False and row.get("first_failed_tensor")
            ),
            None,
        )
        if bitflip is None:
            blockers.append("snerv_payload_bitflip_falsification_missing")
        elif proven is not None:
            highest_level = "L4"

    approved = not blockers and highest_level == ("L5" if family == "hinerv" else "L4")
    verdict = {
        "schema": NERV_LONG_RUN_LAUNCH_GATE_SCHEMA,
        "family": family,
        "run_root": root.as_posix(),
        "approved": bool(approved),
        "highest_level": highest_level,
        "blocking_evidence": sorted(set(blockers)),
        "non_blocking_warnings": sorted(set(warnings)),
        "evidence_index": evidence_index,
        "frontier_pointer": pointer,
        "evaluated_at_utc": now.isoformat(),
        "authority": "[planning/launch-gate]",
    }
    verdict.update(PROXY_FALSE_AUTHORITY_FIELDS)
    return verdict


def _canonical_family(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    if family == "hi_nerv":
        return "hinerv"
    return family
