#!/usr/bin/env python3
"""Lower PR85 pair-gradient planning signal into explicit pair-action specs.

This tool is a deterministic, local-only bridge between scorer-gradient or
pair-atom opportunity plans and archive-relevant candidate specifications. It
does not build archives, import scorers, run GPUs, touch dispatch state, or make
score claims. Dispatch remains locked unless an input action evidence record
points at a real, non-noop archive-changing path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/build_pr85_pair_action_candidates.py"
SCHEMA = "pr85_pair_action_candidate_specs_v1"
CANDIDATE_SCHEMA = "pr85_pair_action_candidate_spec_v1"
ACTION_EVIDENCE_SCHEMA = "pr85_pair_action_lowering_evidence_v1"
PAIR_ATOM_READINESS_SCHEMA = "pr85_pair_atom_candidate_readiness_v1"
PAIR_ATOM_ACTION_SPEC_SCHEMA = "pr85_pair_atom_action_spec_v1"
SCORER_PLAN_SCHEMA = "pr85_scorer_gradient_atom_opportunity_v1"

DEFAULT_PAIR_ATOM_PLANNING = (
    REPO_ROOT
    / "experiments/results/pr85_pair_atom_candidates_20260504_orchestrator/planning.json"
)
DEFAULT_OUT_JSON = Path("/tmp/pr85_pair_action_candidates_worker_20260504/candidate_specs.json")
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr85_pair_action_lowering_worker_20260504.md"

PAIR_COUNT = 600
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
ACTION_STREAMS = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)


class PairActionLoweringError(ValueError):
    """Raised when a caller supplies structurally invalid arguments."""


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise PairActionLoweringError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PairActionLoweringError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key != "stable_plan_digest_sha256"
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _path_meta(path: Path) -> dict[str, Any]:
    return {
        "path": _rel(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_candidate_id(value: str) -> str:
    if not value or not SAFE_ID.match(value):
        raise PairActionLoweringError(f"unsafe candidate_id: {value!r}")
    return value


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.match(value) is not None


def _blocker(blocker_class: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"blocker_class": blocker_class, "reason": reason, **extra}


def _canonical_source_archive(pair_payload: Mapping[str, Any], scorer_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = pair_payload.get("source_archive")
    source = source if isinstance(source, dict) else {}
    exact_eval = scorer_payload.get("exact_eval") if isinstance(scorer_payload, dict) else {}
    exact_eval = exact_eval if isinstance(exact_eval, dict) else {}
    provenance = exact_eval.get("provenance") if isinstance(exact_eval.get("provenance"), dict) else {}
    return {
        "path": source.get("path"),
        "archive_bytes": source.get("archive_bytes", exact_eval.get("archive_size_bytes")),
        "archive_sha256": source.get("archive_sha256", provenance.get("archive_sha256")),
        "member_sha256": source.get("member_sha256"),
        "evidence_context": {
            "scorer_exact_eval_score": exact_eval.get("reported_score"),
            "scorer_exact_eval_samples": exact_eval.get("n_samples"),
            "scorer_exact_eval_device": provenance.get("device"),
            "scorer_exact_eval_gpu_model": provenance.get("gpu_model"),
            "runtime_tree_sha256": provenance.get("runtime_tree_sha256"),
        },
    }


def _resolve_scorer_path(pair_payload: Mapping[str, Any], scorer_plan_json: Path | None) -> Path | None:
    if scorer_plan_json is not None:
        return scorer_plan_json
    report = pair_payload.get("scorer_gradient_plan")
    if isinstance(report, Mapping):
        raw_path = report.get("path")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            return path if path.is_absolute() else REPO_ROOT / path
    return None


def _load_pair_atom_planning(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not path.is_file():
        return (
            {
                "status": "blocked",
                "path": _rel(path),
                "blocker_class": "missing_pair_atom_planning_source",
                "blockers": [
                    _blocker(
                        "missing_pair_atom_planning_source",
                        "pair-atom planning JSON does not exist",
                    )
                ],
            },
            None,
        )
    payload = _load_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != PAIR_ATOM_READINESS_SCHEMA:
        blockers.append(
            _blocker(
                "unsupported_pair_atom_planning_source",
                "unexpected pair-atom readiness schema",
                observed_schema=payload.get("schema"),
            )
        )
    if payload.get("score_claim") is not False:
        blockers.append(_blocker("unsafe_pair_atom_planning_source", "score_claim must be false"))
    if payload.get("dispatch_performed") is not False:
        blockers.append(_blocker("unsafe_pair_atom_planning_source", "dispatch_performed must be false"))
    if payload.get("remote_jobs_dispatched") is not False:
        blockers.append(_blocker("unsafe_pair_atom_planning_source", "remote_jobs_dispatched must be false"))
    report = {
        **_path_meta(path),
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else str(blockers[0]["blocker_class"]),
        "blockers": blockers,
        "schema": payload.get("schema"),
        "dispatch_unlocked": payload.get("dispatch_unlocked"),
        "candidate_archive_count": payload.get("candidate_archive_count"),
    }
    return report, payload


def _source_top_atoms_from_pair_payload(pair_payload: Mapping[str, Any], *, limit: int) -> list[dict[str, Any]]:
    report = pair_payload.get("scorer_gradient_plan")
    report = report if isinstance(report, Mapping) else {}
    top_atoms = report.get("top_atoms")
    if not isinstance(top_atoms, list):
        return []
    return [dict(row) for row in top_atoms[:limit] if isinstance(row, Mapping)]


def _load_scorer_plan(
    path: Path | None,
    *,
    pair_payload: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    limit: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    if path is None:
        fallback = _source_top_atoms_from_pair_payload(pair_payload, limit=limit)
        blockers = []
        if not fallback:
            blockers.append(
                _blocker(
                    "missing_scorer_gradient_source",
                    "no scorer-gradient plan path or embedded top atoms are available",
                )
            )
        return (
            {
                "status": "passed_from_pair_atom_embedded_report" if fallback else "blocked",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "blocker_class": "none" if not blockers else str(blockers[0]["blocker_class"]),
                "blockers": blockers,
                "top_atoms": fallback,
            },
            None,
            fallback,
        )
    if not path.is_file():
        fallback = _source_top_atoms_from_pair_payload(pair_payload, limit=limit)
        blockers = [
            _blocker(
                "missing_scorer_gradient_source",
                "scorer-gradient plan path does not exist",
                requested_path=_rel(path),
                embedded_top_atoms_used=bool(fallback),
            )
        ]
        return (
            {
                "status": "blocked",
                "path": _rel(path),
                "sha256": None,
                "size_bytes": None,
                "blocker_class": str(blockers[0]["blocker_class"]),
                "blockers": blockers,
                "top_atoms": fallback,
            },
            None,
            fallback,
        )

    payload = _load_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != SCORER_PLAN_SCHEMA:
        blockers.append(
            _blocker(
                "unsupported_scorer_gradient_source",
                "unexpected scorer-gradient schema",
                observed_schema=payload.get("schema"),
            )
        )
    if payload.get("planning_only") is not True:
        blockers.append(_blocker("unsafe_scorer_gradient_source", "planning_only must be true"))
    if payload.get("score_claim") is not False:
        blockers.append(_blocker("unsafe_scorer_gradient_source", "score_claim must be false"))
    for key in ("dispatch_performed", "remote_jobs_dispatched", "inflate_time_scorer_load_allowed"):
        if payload.get(key) not in (False, None):
            blockers.append(_blocker("unsafe_scorer_gradient_source", f"{key} must be false"))
    declared_digest = payload.get("stable_plan_digest_sha256")
    computed_digest = _stable_digest(payload)
    if isinstance(declared_digest, str) and declared_digest != computed_digest:
        blockers.append(
            _blocker(
                "stale_scorer_gradient_source",
                "stable_plan_digest_sha256 mismatch",
                declared=declared_digest,
                computed=computed_digest,
            )
        )
    exact_eval = payload.get("exact_eval") if isinstance(payload.get("exact_eval"), dict) else {}
    provenance = exact_eval.get("provenance") if isinstance(exact_eval.get("provenance"), dict) else {}
    source_sha = source_archive.get("archive_sha256")
    source_bytes = source_archive.get("archive_bytes")
    plan_sha = provenance.get("archive_sha256")
    plan_bytes = provenance.get("archive_size_bytes", exact_eval.get("archive_size_bytes"))
    if source_sha and plan_sha and source_sha != plan_sha:
        blockers.append(
            _blocker(
                "stale_scorer_gradient_source",
                "scorer-gradient source archive sha does not match pair-action source archive",
                source_archive_sha256=source_sha,
                scorer_archive_sha256=plan_sha,
            )
        )
    if source_bytes is not None and plan_bytes is not None and int(source_bytes) != int(plan_bytes):
        blockers.append(
            _blocker(
                "stale_scorer_gradient_source",
                "scorer-gradient source archive bytes do not match pair-action source archive",
                source_archive_bytes=source_bytes,
                scorer_archive_bytes=plan_bytes,
            )
        )
    atoms = payload.get("atom_ranking")
    top_atoms = [dict(row) for row in atoms[:limit] if isinstance(row, Mapping)] if isinstance(atoms, list) else []
    seen_pairs: set[int] = set()
    duplicate_pairs: list[int] = []
    for row in top_atoms:
        pair = row.get("pair_index")
        if isinstance(pair, int) and not isinstance(pair, bool):
            if pair in seen_pairs:
                duplicate_pairs.append(pair)
            seen_pairs.add(pair)
    if duplicate_pairs:
        blockers.append(
            _blocker(
                "duplicate_source_pairs",
                "scorer-gradient source top atoms contain duplicate pair indices",
                duplicate_pair_indices=duplicate_pairs,
            )
        )
    report = {
        **_path_meta(path),
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else str(blockers[0]["blocker_class"]),
        "blockers": blockers,
        "schema": payload.get("schema"),
        "stable_plan_digest_sha256": declared_digest,
        "computed_stable_plan_digest_sha256": computed_digest,
        "atom_count": len(atoms) if isinstance(atoms, list) else None,
        "top_atoms": [_summarize_atom(row, rank=index + 1) for index, row in enumerate(top_atoms)],
    }
    return report, payload, top_atoms


def _break_even(row: Mapping[str, Any], component: str) -> float | None:
    byte_break_even = row.get("byte_break_even")
    if not isinstance(byte_break_even, Mapping):
        return None
    section = byte_break_even.get(component)
    if not isinstance(section, Mapping):
        return None
    value = section.get("max_charged_bytes_for_zero_net_change")
    return float(value) if _finite_number(value) else None


def _targeted_component(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    combined = _break_even(row, "combined")
    pose = _break_even(row, "pose_only")
    seg = _break_even(row, "seg_only")
    if pose is None and seg is None:
        dominant = None
    elif seg is None or (pose is not None and pose >= seg):
        dominant = "pose"
    else:
        dominant = "seg"
    return (
        "combined",
        {
            "dominant_single_component": dominant,
            "combined_break_even_bytes": combined,
            "pose_break_even_bytes": pose,
            "seg_break_even_bytes": seg,
        },
    )


def _summarize_atom(row: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
    targeted, component_signal = _targeted_component(row)
    pair_index = row.get("pair_index")
    return {
        "rank": rank,
        "atom_id": row.get("atom_id"),
        "pair_index": pair_index,
        "frame_indices": row.get("frame_indices"),
        "ranking_score": row.get("ranking_score"),
        "targeted_component": targeted,
        "component_signal": component_signal,
        "dispatch_gate": row.get("dispatch_gate"),
    }


def _unlowered_candidate(
    atom: Mapping[str, Any],
    *,
    rank: int,
    source_archive: Mapping[str, Any],
    pair_source_sha256: str | None,
    scorer_source_sha256: str | None,
) -> dict[str, Any]:
    summary = _summarize_atom(atom, rank=rank)
    pair_index = summary["pair_index"]
    candidate_id = f"pr85_pair_{int(pair_index):04d}_unlowered" if isinstance(pair_index, int) else f"pr85_rank_{rank:04d}_unlowered"
    reason = (
        "scorer-gradient pair ranking has no explicit stream/value delta, "
        "non-noop proof, or archive-changing runtime path"
    )
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "lowering_status": "blocked",
        "blocker_class": "missing_explicit_pair_action",
        "blockers": [_blocker("missing_explicit_pair_action", reason)],
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": False,
        "ready_for_pair_atom_archive_build": False,
        "ready_for_exact_eval_after_lane_claim": False,
        "selected_pair_indices": [pair_index] if isinstance(pair_index, int) else [],
        "selected_pairs": [summary],
        "targeted_component": summary["targeted_component"],
        "actions": [],
        "pair_atom_action_spec": None,
        "charged_bytes_proxy": {
            "status": "blocked_no_charged_action",
            "candidate_action_bytes": None,
            "formula_rate_score_per_byte": RATE_SCORE_PER_BYTE,
            "break_even_bytes": summary["component_signal"].get("combined_break_even_bytes"),
            "basis": "scorer-gradient formula-only opportunity; no charged stream action selected",
        },
        "input_archive": {
            "archive_sha256": source_archive.get("archive_sha256"),
            "archive_bytes": source_archive.get("archive_bytes"),
        },
        "source_artifacts": {
            "pair_atom_planning_sha256": pair_source_sha256,
            "scorer_gradient_plan_sha256": scorer_source_sha256,
            "source_atom_id": summary.get("atom_id"),
        },
        "no_op_status": {
            "status": "missing_action_delta",
            "is_noop": None,
            "reason": "no stream/value action exists to compare with source semantics",
        },
        "next_action": "Provide grounded pair-action evidence with stream, source_value, candidate_value, charged_bytes_proxy, and archive-changing path, or run a local archive builder to prove one.",
    }


def _known_source_shas(*reports: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for report in reports:
        sha = report.get("sha256")
        if isinstance(sha, str) and sha:
            out.add(sha)
        stable = report.get("stable_plan_digest_sha256")
        if isinstance(stable, str) and stable:
            out.add(stable)
    return out


def _validate_thresholds(payload: Mapping[str, Any], *, known_shas: set[str]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    thresholds = payload.get("thresholds", [])
    if thresholds in (None, []):
        return blockers
    if not isinstance(thresholds, list):
        return [_blocker("ungrounded_threshold", "thresholds must be a list when supplied")]
    for index, row in enumerate(thresholds):
        if not isinstance(row, Mapping):
            blockers.append(_blocker("ungrounded_threshold", f"thresholds[{index}] must be an object"))
            continue
        value = row.get("value")
        if not _finite_number(value):
            blockers.append(_blocker("ungrounded_threshold", f"thresholds[{index}].value must be finite"))
        grounded_by = row.get("grounded_by")
        source_sha = row.get("source_artifact_sha256")
        if not isinstance(grounded_by, str) or not grounded_by.strip():
            blockers.append(_blocker("ungrounded_threshold", f"thresholds[{index}] lacks grounded_by"))
        if source_sha not in known_shas:
            blockers.append(
                _blocker(
                    "ungrounded_threshold",
                    f"thresholds[{index}] source_artifact_sha256 is not a known input artifact",
                    source_artifact_sha256=source_sha,
                )
            )
    return blockers


def _charged_proxy_report(
    raw: Any,
    *,
    known_shas: set[str],
    candidate_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    if not isinstance(raw, Mapping):
        return (
            {
                "status": "blocked",
                "candidate_action_bytes": None,
                "formula_rate_score_per_byte": RATE_SCORE_PER_BYTE,
            },
            [_blocker("missing_charged_bytes_proxy", f"{candidate_id} lacks charged_bytes_proxy")],
        )
    bytes_value = raw.get("candidate_action_bytes", raw.get("bytes"))
    basis = raw.get("basis")
    source_sha = raw.get("source_artifact_sha256")
    if not _finite_number(bytes_value) or float(bytes_value) < 0:
        blockers.append(_blocker("missing_charged_bytes_proxy", f"{candidate_id} charged byte proxy must be finite and non-negative"))
    if not isinstance(basis, str) or not basis.strip():
        blockers.append(_blocker("missing_charged_bytes_proxy", f"{candidate_id} charged byte proxy lacks basis"))
    if source_sha not in known_shas:
        blockers.append(
            _blocker(
                "missing_charged_bytes_proxy",
                f"{candidate_id} charged byte proxy is not grounded in a known input artifact",
                source_artifact_sha256=source_sha,
            )
        )
    report = {
        "status": "passed" if not blockers else "blocked",
        "candidate_action_bytes": bytes_value,
        "formula_rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "formula_only_rate_score_delta": (
            float(bytes_value) * RATE_SCORE_PER_BYTE if _finite_number(bytes_value) else None
        ),
        "basis": basis,
        "source_artifact_sha256": source_sha,
    }
    return report, blockers


def _validate_action(
    raw: Any,
    *,
    row_index: int,
    candidate_id: str,
    known_shas: set[str],
    top_by_pair: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    if not isinstance(raw, Mapping):
        return None, [_blocker("unsupported_pair_action", f"{candidate_id}.actions[{row_index}] must be an object")]
    pair_index = raw.get("pair_index")
    if not isinstance(pair_index, int) or isinstance(pair_index, bool) or not 0 <= pair_index < PAIR_COUNT:
        blockers.append(_blocker("unsupported_pair_action", f"{candidate_id}.actions[{row_index}].pair_index must be in [0,{PAIR_COUNT})"))
    stream = raw.get("stream")
    if stream not in ACTION_STREAMS:
        blockers.append(
            _blocker(
                "unsupported_pair_action",
                f"{candidate_id}.actions[{row_index}].stream is unsupported",
                supported_streams=list(ACTION_STREAMS),
            )
        )
    op = raw.get("op", "set")
    if op not in {"set", "delta"}:
        blockers.append(_blocker("unsupported_pair_action", f"{candidate_id}.actions[{row_index}].op must be set or delta"))
    source_sha = raw.get("source_artifact_sha256")
    if source_sha not in known_shas:
        blockers.append(
            _blocker(
                "missing_source_evidence",
                f"{candidate_id}.actions[{row_index}] is not grounded in a known source artifact",
                source_artifact_sha256=source_sha,
            )
        )
    source_value = raw.get("source_value")
    candidate_value = raw.get("candidate_value", raw.get("value"))
    no_op_status: dict[str, Any]
    if not _finite_number(source_value) or not _finite_number(candidate_value):
        blockers.append(_blocker("missing_non_noop_proof", f"{candidate_id}.actions[{row_index}] lacks source_value/candidate_value proof"))
        no_op_status = {
            "status": "unknown_missing_values",
            "is_noop": None,
            "source_value": source_value,
            "candidate_value": candidate_value,
        }
    elif float(source_value) == float(candidate_value):
        blockers.append(_blocker("no_op_action", f"{candidate_id}.actions[{row_index}] preserves the source value"))
        no_op_status = {
            "status": "noop",
            "is_noop": True,
            "source_value": source_value,
            "candidate_value": candidate_value,
        }
    else:
        no_op_status = {
            "status": "non_noop_value_change",
            "is_noop": False,
            "source_value": source_value,
            "candidate_value": candidate_value,
        }
    source_atom = top_by_pair.get(int(pair_index)) if isinstance(pair_index, int) and not isinstance(pair_index, bool) else None
    source_atom_id = raw.get("source_atom_id")
    if source_atom is None:
        blockers.append(
            _blocker(
                "action_pair_not_in_source_signal",
                f"{candidate_id}.actions[{row_index}] pair was not present in the selected scorer-gradient source atoms",
                pair_index=pair_index,
            )
        )
    elif source_atom_id is not None and source_atom_id != source_atom.get("atom_id"):
        blockers.append(
            _blocker(
                "missing_source_evidence",
                f"{candidate_id}.actions[{row_index}] source_atom_id does not match scorer-gradient atom",
                source_atom_id=source_atom_id,
                expected_source_atom_id=source_atom.get("atom_id"),
            )
        )
    action = {
        "pair_index": pair_index,
        "stream": stream,
        "op": op,
        "source_value": source_value,
        "candidate_value": candidate_value,
        "value": candidate_value,
        "source_artifact_sha256": source_sha,
        "source_atom_id": source_atom.get("atom_id") if isinstance(source_atom, Mapping) else source_atom_id,
        "rationale": raw.get("rationale"),
        "no_op_status": no_op_status,
    }
    return action, blockers


def _archive_path_report(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if raw is None:
        return (
            {
                "status": "missing",
                "archive_changing_path_exists": False,
                "dispatch_unlocked": False,
            },
            [_blocker("no_archive_changing_path", "no built archive-changing path was supplied")],
        )
    if not isinstance(raw, Mapping):
        return (
            {"status": "blocked", "archive_changing_path_exists": False, "dispatch_unlocked": False},
            [_blocker("no_archive_changing_path", "archive_changing_path must be an object")],
        )
    blockers: list[dict[str, Any]] = []
    status = raw.get("status")
    sha = raw.get("candidate_archive_sha256")
    bytes_value = raw.get("candidate_archive_bytes")
    non_noop = raw.get("non_noop_proof")
    non_noop = non_noop if isinstance(non_noop, Mapping) else {}
    if status != "built":
        blockers.append(_blocker("no_archive_changing_path", "archive_changing_path.status must be built"))
    if not _is_hex64(sha):
        blockers.append(_blocker("no_archive_changing_path", "candidate_archive_sha256 must be a lowercase sha256"))
    if not isinstance(bytes_value, int) or isinstance(bytes_value, bool) or bytes_value <= 0:
        blockers.append(_blocker("no_archive_changing_path", "candidate_archive_bytes must be a positive integer"))
    if non_noop.get("status") != "passed":
        blockers.append(_blocker("no_archive_changing_path", "non_noop_proof.status must be passed"))
    if raw.get("score_claim") not in (False, None):
        blockers.append(_blocker("no_archive_changing_path", "archive-changing path must not carry a score claim"))
    if raw.get("lane_claim_required_before_exact_eval") is not True:
        blockers.append(_blocker("no_archive_changing_path", "lane_claim_required_before_exact_eval must be true"))
    return (
        {
            "status": "passed" if not blockers else "blocked",
            "archive_changing_path_exists": not blockers,
            "dispatch_unlocked": not blockers,
            "candidate_archive_sha256": sha,
            "candidate_archive_bytes": bytes_value,
            "manifest_path": raw.get("manifest_path"),
            "non_noop_proof": dict(non_noop),
            "lane_claim_required_before_exact_eval": raw.get("lane_claim_required_before_exact_eval"),
        },
        blockers,
    )


def _candidate_from_evidence(
    raw: Any,
    *,
    candidate_index: int,
    source_archive: Mapping[str, Any],
    known_shas: set[str],
    top_by_pair: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(raw, Mapping):
        candidate_id = f"pr85_pair_action_{candidate_index:03d}"
        blockers = [_blocker("unsupported_pair_action_spec", f"candidates[{candidate_index}] must be an object")]
        return _blocked_candidate(candidate_id, blockers, source_archive=source_archive), blockers
    candidate_id = _safe_candidate_id(str(raw.get("candidate_id") or f"pr85_pair_action_{candidate_index:03d}"))
    blockers: list[dict[str, Any]] = []
    action_rows = raw.get("actions")
    if not isinstance(action_rows, list) or not action_rows:
        blockers.append(_blocker("missing_explicit_pair_action", f"{candidate_id} has no actions"))
        action_rows = []
    actions: list[dict[str, Any]] = []
    seen_pairs: set[int] = set()
    duplicate_pairs: list[int] = []
    for index, action_raw in enumerate(action_rows):
        action, action_blockers = _validate_action(
            action_raw,
            row_index=index,
            candidate_id=candidate_id,
            known_shas=known_shas,
            top_by_pair=top_by_pair,
        )
        blockers.extend(action_blockers)
        if action is None:
            continue
        pair = action.get("pair_index")
        if isinstance(pair, int):
            if pair in seen_pairs:
                duplicate_pairs.append(pair)
            seen_pairs.add(pair)
        actions.append(action)
    if duplicate_pairs:
        blockers.append(
            _blocker(
                "duplicate_pair_action",
                f"{candidate_id} selects the same pair more than once",
                duplicate_pair_indices=sorted(set(duplicate_pairs)),
            )
        )

    proxy, proxy_blockers = _charged_proxy_report(
        raw.get("charged_bytes_proxy"),
        known_shas=known_shas,
        candidate_id=candidate_id,
    )
    blockers.extend(proxy_blockers)
    archive_path, archive_blockers = _archive_path_report(raw.get("archive_changing_path"))
    archive_only_blockers = [item for item in archive_blockers if item.get("blocker_class") == "no_archive_changing_path"]
    if raw.get("archive_changing_path") is not None:
        blockers.extend(archive_blockers)

    selected_pairs = []
    for rank, pair in enumerate(sorted(seen_pairs), start=1):
        atom = top_by_pair.get(pair, {"pair_index": pair})
        selected_pairs.append(_summarize_atom(atom, rank=rank))
    targeted_component = raw.get("targeted_component") or (
        selected_pairs[0]["targeted_component"] if selected_pairs else "combined"
    )
    pair_atom_action_spec = None
    if actions and not blockers:
        pair_atom_action_spec = {
            "schema": PAIR_ATOM_ACTION_SPEC_SCHEMA,
            "score_claim": False,
            "dispatch_performed": False,
            "inflate_time_scorer_load_allowed": False,
            "candidate_id": candidate_id,
            "header_mode": raw.get("header_mode", "explicit_30"),
            "actions": [
                {
                    "pair_index": action["pair_index"],
                    "stream": action["stream"],
                    "value": action["candidate_value"],
                    "rationale": action.get("rationale"),
                }
                for action in actions
            ],
        }
    dispatch_unlocked = not blockers and archive_path["dispatch_unlocked"] is True
    ready_for_archive_build = bool(actions and not blockers and archive_path["dispatch_unlocked"] is False)
    candidate_blockers = blockers if blockers else archive_only_blockers
    status = (
        "archive_path_unlocked"
        if dispatch_unlocked
        else "action_spec_emitted"
        if ready_for_archive_build
        else "blocked"
    )
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "lowering_status": status,
        "blocker_class": "none" if not candidate_blockers else str(candidate_blockers[0]["blocker_class"]),
        "blockers": candidate_blockers,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": dispatch_unlocked,
        "ready_for_pair_atom_archive_build": ready_for_archive_build,
        "ready_for_exact_eval_after_lane_claim": dispatch_unlocked,
        "selected_pair_indices": sorted(seen_pairs),
        "selected_pairs": selected_pairs,
        "targeted_component": targeted_component,
        "actions": actions,
        "pair_atom_action_spec": pair_atom_action_spec,
        "charged_bytes_proxy": proxy,
        "archive_changing_path": archive_path,
        "input_archive": {
            "archive_sha256": source_archive.get("archive_sha256"),
            "archive_bytes": source_archive.get("archive_bytes"),
        },
        "source_artifacts": {
            "known_source_sha256s": sorted(known_shas),
        },
        "no_op_status": _candidate_noop_status(actions),
        "next_action": (
            "Run experiments/build_pr85_pair_atom_candidates.py with this action spec and a reviewed runtime contract."
            if ready_for_archive_build
            else "Resolve blockers before any exact eval dispatch."
        ),
    }
    return candidate, blockers


def _candidate_noop_status(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = [
        action.get("no_op_status", {})
        for action in actions
        if isinstance(action.get("no_op_status"), Mapping)
    ]
    if not statuses:
        return {
            "status": "missing_action_delta",
            "is_noop": None,
            "reason": "no explicit actions were emitted",
        }
    if any(status.get("is_noop") is True for status in statuses):
        return {"status": "contains_noop_action", "is_noop": True}
    if all(status.get("is_noop") is False for status in statuses):
        return {"status": "non_noop_value_change", "is_noop": False}
    return {"status": "unknown", "is_noop": None}


def _blocked_candidate(
    candidate_id: str,
    blockers: Sequence[Mapping[str, Any]],
    *,
    source_archive: Mapping[str, Any],
) -> dict[str, Any]:
    blocker_list = [dict(item) for item in blockers]
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate_id": candidate_id,
        "lowering_status": "blocked",
        "blocker_class": "blocked" if not blocker_list else str(blocker_list[0]["blocker_class"]),
        "blockers": blocker_list,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": False,
        "ready_for_pair_atom_archive_build": False,
        "ready_for_exact_eval_after_lane_claim": False,
        "selected_pair_indices": [],
        "selected_pairs": [],
        "targeted_component": "combined",
        "actions": [],
        "pair_atom_action_spec": None,
        "charged_bytes_proxy": {
            "status": "blocked",
            "candidate_action_bytes": None,
            "formula_rate_score_per_byte": RATE_SCORE_PER_BYTE,
        },
        "input_archive": {
            "archive_sha256": source_archive.get("archive_sha256"),
            "archive_bytes": source_archive.get("archive_bytes"),
        },
        "no_op_status": {"status": "blocked", "is_noop": None},
    }


def _load_action_evidence(
    path: Path | None,
    *,
    source_archive: Mapping[str, Any],
    known_shas: set[str],
    top_atoms: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        blockers = [
            _blocker(
                "missing_pair_action_evidence",
                "no grounded pair-action evidence JSON was supplied",
            )
        ]
        return (
            {
                "status": "blocked",
                "path": None,
                "blocker_class": str(blockers[0]["blocker_class"]),
                "blockers": blockers,
            },
            [],
            blockers,
        )
    if not path.is_file():
        blockers = [
            _blocker(
                "missing_pair_action_evidence",
                "pair-action evidence JSON does not exist",
                requested_path=_rel(path),
            )
        ]
        return (
            {
                "status": "blocked",
                "path": _rel(path),
                "blocker_class": str(blockers[0]["blocker_class"]),
                "blockers": blockers,
            },
            [],
            blockers,
        )
    payload = _load_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != ACTION_EVIDENCE_SCHEMA:
        blockers.append(
            _blocker(
                "unsupported_pair_action_evidence",
                "unexpected pair-action evidence schema",
                observed_schema=payload.get("schema"),
            )
        )
    if payload.get("score_claim") is not False:
        blockers.append(_blocker("unsafe_pair_action_evidence", "score_claim must be false"))
    if payload.get("dispatch_performed") is not False:
        blockers.append(_blocker("unsafe_pair_action_evidence", "dispatch_performed must be false"))
    if payload.get("remote_jobs_dispatched") is not False:
        blockers.append(_blocker("unsafe_pair_action_evidence", "remote_jobs_dispatched must be false"))
    blockers.extend(_validate_thresholds(payload, known_shas=known_shas))
    root_blockers = list(blockers)
    candidates_raw = payload.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        blockers.append(_blocker("missing_explicit_pair_action", "pair-action evidence contains no candidates"))
        candidates_raw = []
    top_by_pair = {
        int(row["pair_index"]): row
        for row in top_atoms
        if isinstance(row.get("pair_index"), int) and not isinstance(row.get("pair_index"), bool)
    }
    candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    seen_pairs_global: set[int] = set()
    for index, row in enumerate(candidates_raw):
        try:
            candidate, candidate_blockers = _candidate_from_evidence(
                row,
                candidate_index=index,
                source_archive=source_archive,
                known_shas=known_shas,
                top_by_pair=top_by_pair,
            )
        except PairActionLoweringError as exc:
            candidate_id = f"pr85_pair_action_{index:03d}"
            candidate_blockers = [_blocker("unsupported_pair_action_spec", str(exc))]
            candidate = _blocked_candidate(candidate_id, candidate_blockers, source_archive=source_archive)
        if candidate["candidate_id"] in seen_candidate_ids:
            duplicate = _blocker(
                "duplicate_candidate_id",
                "pair-action evidence repeats a candidate_id",
                candidate_id=candidate["candidate_id"],
            )
            candidate["blockers"].append(duplicate)
            candidate["blocker_class"] = "duplicate_candidate_id"
            candidate["lowering_status"] = "blocked"
            candidate["dispatch_unlocked"] = False
            candidate["ready_for_pair_atom_archive_build"] = False
            candidate_blockers.append(duplicate)
        seen_candidate_ids.add(candidate["candidate_id"])
        duplicate_pairs = sorted(set(candidate["selected_pair_indices"]) & seen_pairs_global)
        if duplicate_pairs:
            duplicate = _blocker(
                "duplicate_pair_action",
                "pair-action evidence assigns a pair to more than one candidate",
                duplicate_pair_indices=duplicate_pairs,
            )
            candidate["blockers"].append(duplicate)
            candidate["blocker_class"] = "duplicate_pair_action"
            candidate["lowering_status"] = "blocked"
            candidate["dispatch_unlocked"] = False
            candidate["ready_for_pair_atom_archive_build"] = False
            candidate_blockers.append(duplicate)
        seen_pairs_global.update(candidate["selected_pair_indices"])
        if root_blockers:
            candidate["blockers"].extend(root_blockers)
            candidate["blocker_class"] = str(root_blockers[0]["blocker_class"])
            candidate["lowering_status"] = "blocked"
            candidate["dispatch_unlocked"] = False
            candidate["ready_for_pair_atom_archive_build"] = False
            candidate["ready_for_exact_eval_after_lane_claim"] = False
            candidate["pair_atom_action_spec"] = None
            candidate_blockers.extend(root_blockers)
        candidates.append(candidate)
        blockers.extend(candidate_blockers)
    report = {
        **_path_meta(path),
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else str(blockers[0]["blocker_class"]),
        "blockers": blockers,
        "schema": payload.get("schema"),
        "candidate_count": len(candidates),
    }
    return report, candidates, blockers


def build_pair_action_candidates(
    *,
    pair_atom_planning_json: Path = DEFAULT_PAIR_ATOM_PLANNING,
    scorer_plan_json: Path | None = None,
    action_evidence_json: Path | None = None,
    top_n: int = 8,
    write_json: Path | None = None,
) -> dict[str, Any]:
    if top_n <= 0:
        raise PairActionLoweringError("top_n must be positive")
    pair_report, pair_payload = _load_pair_atom_planning(pair_atom_planning_json)
    if pair_payload is None:
        source_archive = _canonical_source_archive({}, None)
        summary = _summary(
            pair_report=pair_report,
            scorer_report=None,
            action_report=None,
            source_archive=source_archive,
            candidates=[],
            blockers=pair_report["blockers"],
            top_n=top_n,
        )
        if write_json is not None:
            _write_json(write_json, summary)
        return summary

    scorer_path = _resolve_scorer_path(pair_payload, scorer_plan_json)
    scorer_payload_for_source = _load_json(scorer_path) if scorer_path is not None and scorer_path.is_file() else None
    source_archive = _canonical_source_archive(pair_payload, scorer_payload_for_source)
    scorer_report, scorer_payload, top_atoms = _load_scorer_plan(
        scorer_path,
        pair_payload=pair_payload,
        source_archive=source_archive,
        limit=top_n,
    )
    known_shas = _known_source_shas(pair_report, scorer_report)
    action_report, action_candidates, action_blockers = _load_action_evidence(
        action_evidence_json,
        source_archive=source_archive,
        known_shas=known_shas,
        top_atoms=top_atoms,
    )
    candidates = action_candidates
    if action_evidence_json is None:
        candidates = [
            _unlowered_candidate(
                atom,
                rank=index + 1,
                source_archive=source_archive,
                pair_source_sha256=pair_report.get("sha256"),
                scorer_source_sha256=scorer_report.get("sha256"),
            )
            for index, atom in enumerate(top_atoms[:top_n])
        ]
    blockers = []
    for report in (pair_report, scorer_report, action_report):
        blockers.extend(report.get("blockers", []) if isinstance(report, Mapping) else [])
    blockers.extend(
        blocker
        for candidate in candidates
        for blocker in candidate.get("blockers", [])
        if isinstance(blocker, Mapping)
    )
    if action_evidence_json is not None:
        blockers.extend(action_blockers)
    summary = _summary(
        pair_report=pair_report,
        scorer_report=scorer_report,
        action_report=action_report,
        source_archive=source_archive,
        candidates=candidates,
        blockers=blockers,
        top_n=top_n,
    )
    if write_json is not None:
        _write_json(write_json, summary)
    return summary


def _summary(
    *,
    pair_report: Mapping[str, Any],
    scorer_report: Mapping[str, Any] | None,
    action_report: Mapping[str, Any] | None,
    source_archive: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, Any]],
    top_n: int,
) -> dict[str, Any]:
    blocker_list = _dedupe_blockers([dict(item) for item in blockers if isinstance(item, Mapping)])
    dispatch_unlocked_count = sum(1 for row in candidates if row.get("dispatch_unlocked") is True)
    archive_build_ready_count = sum(1 for row in candidates if row.get("ready_for_pair_atom_archive_build") is True)
    exact_ready_count = sum(1 for row in candidates if row.get("ready_for_exact_eval_after_lane_claim") is True)
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "scorer_load_performed": False,
        "sidecars_required": False,
        "dispatch_unlocked": dispatch_unlocked_count > 0,
        "dispatch_unlocked_count": dispatch_unlocked_count,
        "ready_for_pair_atom_archive_build_count": archive_build_ready_count,
        "ready_for_exact_eval_after_lane_claim_count": exact_ready_count,
        "candidate_count": len(candidates),
        "top_n": top_n,
        "blocker_class": "none" if not blocker_list else str(blocker_list[0]["blocker_class"]),
        "blockers": blocker_list,
        "source_archive": dict(source_archive),
        "pair_atom_planning": dict(pair_report),
        "scorer_gradient_plan": dict(scorer_report) if scorer_report is not None else None,
        "action_evidence": dict(action_report) if action_report is not None else None,
        "candidates": [dict(row) for row in candidates],
        "dispatch_rule": (
            "No exact eval dispatch is unlocked unless a non-noop archive-changing path "
            "is present and a lane claim is made first."
        ),
    }


def _dedupe_blockers(blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in blockers:
        key = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def render_ledger(summary: Mapping[str, Any], *, command: str | None = None) -> str:
    source = summary.get("source_archive", {})
    scorer = summary.get("scorer_gradient_plan") or {}
    action = summary.get("action_evidence") or {}
    top_rows = []
    for row in summary.get("candidates", [])[:8]:
        pairs = row.get("selected_pair_indices")
        top_rows.append(
            "- "
            f"{row.get('candidate_id')}: status={row.get('lowering_status')} "
            f"pairs={pairs} blocker={row.get('blocker_class')} "
            f"exact_ready={str(row.get('ready_for_exact_eval_after_lane_claim')).lower()}"
        )
    if not top_rows:
        top_rows.append("- no candidate specs emitted")
    blockers = summary.get("blockers", [])
    blocker_rows = [
        f"- {item.get('blocker_class')}: {item.get('reason')}"
        for item in blockers[:10]
        if isinstance(item, Mapping)
    ]
    if not blocker_rows:
        blocker_rows.append("- none")
    command_row = f"- command: `{command}`" if command else "- command: not recorded"
    return "\n".join(
        [
            "# PR85 Pair-Action Lowering Worker",
            "",
            "## Contract",
            "",
            "- tool: `experiments/build_pr85_pair_action_candidates.py`",
            "- score_claim: false",
            "- dispatch_performed: false",
            "- remote_jobs_dispatched: false",
            f"- dispatch_unlocked: {str(summary.get('dispatch_unlocked')).lower()}",
            f"- ready_for_exact_eval_after_lane_claim_count: {summary.get('ready_for_exact_eval_after_lane_claim_count')}",
            f"- blocker_class: `{summary.get('blocker_class')}`",
            "",
            "## Source Evidence",
            "",
            f"- source archive bytes: {source.get('archive_bytes')}",
            f"- source archive sha256: `{source.get('archive_sha256')}`",
            f"- scorer plan: `{scorer.get('path')}`",
            f"- scorer plan sha256: `{scorer.get('sha256')}`",
            f"- action evidence: `{action.get('path')}`",
            "",
            "## Implementation Decision",
            "",
            (
                "The lowering surface emits pair-action candidate specs only when a "
                "grounded action evidence file provides stream, source value, "
                "candidate value, charged-byte proxy, and source-artifact custody. "
                "The current PR85 scorer-gradient/pair-atom artifacts provide ranked "
                "pairs and break-even bytes but no stream/value direction, so the "
                "default output records blocked unlowered specs instead of inventing "
                "candidate actions."
            ),
            "",
            "## Candidate Specs",
            "",
            *top_rows,
            "",
            "## Blockers",
            "",
            *blocker_rows,
            "",
            "## Command Output",
            "",
            command_row,
            f"- emitted candidate_count: {summary.get('candidate_count')}",
            f"- ready_for_pair_atom_archive_build_count: {summary.get('ready_for_pair_atom_archive_build_count')}",
            f"- dispatch_unlocked_count: {summary.get('dispatch_unlocked_count')}",
            "",
            "## Exact Next Action",
            "",
            (
                "Generate measured, grounded pair-action evidence for at least one "
                "ranked pair, then run this lowering tool with `--action-evidence-json`. "
                "If it emits a non-noop `pair_atom_action_spec`, feed that spec plus a "
                "reviewed runtime contract into `experiments/build_pr85_pair_atom_candidates.py`; "
                "claim the lane before any exact CUDA auth eval."
            ),
            "",
        ]
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-atom-planning-json",
        type=Path,
        default=DEFAULT_PAIR_ATOM_PLANNING,
        help="Pair-atom readiness JSON to lower from.",
    )
    parser.add_argument(
        "--scorer-plan-json",
        type=Path,
        default=None,
        help="Optional scorer-gradient plan JSON. Defaults to the path embedded in pair planning.",
    )
    parser.add_argument(
        "--action-evidence-json",
        type=Path,
        default=None,
        help="Optional grounded pair-action evidence JSON.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of ranked pair opportunities to report when no action evidence is supplied.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Output candidate-spec JSON path.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="Dated worker ledger path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = build_pair_action_candidates(
        pair_atom_planning_json=args.pair_atom_planning_json,
        scorer_plan_json=args.scorer_plan_json,
        action_evidence_json=args.action_evidence_json,
        top_n=args.top_n,
        write_json=args.out_json,
    )
    command = " ".join([Path(sys.argv[0]).as_posix(), *(sys.argv[1:] if argv is None else argv)])
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(render_ledger(summary, command=command), encoding="utf-8")
    print(_json_text({
        "out_json": _rel(args.out_json),
        "ledger": _rel(args.ledger),
        "candidate_count": summary["candidate_count"],
        "blocker_class": summary["blocker_class"],
        "dispatch_unlocked": summary["dispatch_unlocked"],
        "ready_for_exact_eval_after_lane_claim_count": summary["ready_for_exact_eval_after_lane_claim_count"],
    }), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
