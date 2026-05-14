#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic PR85 QRGB pair-atom combo archives.

This tool is intentionally local-only.  It combines already-lowered PR85 QRGB
pair actions for the supported ``bias`` and ``region`` streams, delegates the
actual byte mutation to ``build_pr85_pair_atom_candidates.py``, and then runs
the fixed-runtime readiness preflight for each combo archive.

It does not consume singleton exact scores by default, does not make a score
claim, and does not dispatch remote GPU work.  If future selection logic passes
singleton exact-result JSON, those records must match the exact singleton
archive bytes/SHA and CUDA/eval provenance or the run fails closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import build_pr85_pair_atom_candidates as pair_builder  # noqa: E402
from experiments import preflight_pr85_fixed_runtime_readiness as fixed_preflight  # noqa: E402


TOOL = "experiments/build_pr85_qrgb_pair_atom_combo_candidates.py"
SCHEMA = "pr85_qrgb_pair_atom_combo_planning_v1"
ACTION_SPEC_SCHEMA = pair_builder.ACTION_SPEC_SCHEMA
RUNTIME_CONTRACT_SCHEMA = pair_builder.RUNTIME_CONTRACT_SCHEMA
SUPPORTED_STREAMS = ("bias", "region")
COMBO_SIZES = (2, 3)
DEFAULT_SOURCE_ARCHIVE = pair_builder.DEFAULT_SOURCE_ARCHIVE
DEFAULT_SCORER_PLAN = pair_builder.DEFAULT_SCORER_PLAN
DEFAULT_ACTION_SPEC = (
    REPO_ROOT
    / "experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/action_spec_bias_region.json"
)
DEFAULT_RUNTIME_CONTRACT = (
    REPO_ROOT
    / "experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/runtime_contract_bias_region.json"
)
DEFAULT_SINGLETON_DIR = (
    REPO_ROOT / "experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_qrgb_pair_atom_combo_candidates_20260504_worker"
)
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
KNOWN_PR85 = pair_builder.KNOWN_PR85
KNOWN_SINGLETONS: dict[str, dict[str, Any]] = {
    "pr85_qrgb_f1_bias_pair_0060": {
        "pair_index": 60,
        "stream": "bias",
        "archive_bytes": 236_336,
        "archive_sha256": "81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81",
    },
    "pr85_qrgb_f1_bias_pair_0164": {
        "pair_index": 164,
        "stream": "bias",
        "archive_bytes": 236_335,
        "archive_sha256": "d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e",
    },
    "pr85_qrgb_f1_region_pair_0197": {
        "pair_index": 197,
        "stream": "region",
        "archive_bytes": 236_335,
        "archive_sha256": "236751af46a9c98fa286ecfe613c23a2b96bffbe31784da052304701e02b71c6",
    },
}


class ComboBuilderError(ValueError):
    """Raised for malformed combo-builder inputs."""


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComboBuilderError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ComboBuilderError(f"{path} must contain a JSON object")
    return payload


def _safe_combo_id(candidate_ids: Sequence[str]) -> str:
    suffixes = []
    for candidate_id in candidate_ids:
        if not candidate_id.startswith("pr85_qrgb_"):
            raise ComboBuilderError(f"unsupported candidate_id prefix: {candidate_id}")
        suffixes.append(candidate_id.removeprefix("pr85_qrgb_"))
    combo_id = "pr85_qrgb_combo_" + "__".join(suffixes)
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in combo_id):
        raise ComboBuilderError(f"unsafe combo candidate_id: {combo_id!r}")
    return combo_id


def _action_candidates(action_spec_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(action_spec_json)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != ACTION_SPEC_SCHEMA:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "unexpected action spec schema"})
    if payload.get("score_claim") is not False:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "score_claim must be false"})
    if payload.get("dispatch_performed") is not False or payload.get("remote_jobs_dispatched") is not False:
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "dispatch flags must be false"})
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "candidates must be a list"})
        raw_candidates = []

    candidates: list[dict[str, Any]] = []
    for row in raw_candidates:
        if not isinstance(row, Mapping):
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "candidate row must be an object"})
            continue
        candidate_id = row.get("candidate_id")
        actions = row.get("actions")
        if not isinstance(candidate_id, str) or not candidate_id:
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": "candidate_id must be a string"})
            continue
        if not isinstance(actions, list) or len(actions) != 1:
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": f"{candidate_id} must contain exactly one singleton action"})
            continue
        action = actions[0]
        if not isinstance(action, Mapping):
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": f"{candidate_id} action must be an object"})
            continue
        stream = action.get("stream")
        if stream not in SUPPORTED_STREAMS:
            blockers.append({"blocker_class": "unsupported_stream", "reason": "only bias/region QRGB pair atoms are supported", "candidate_id": candidate_id, "stream": stream})
            continue
        pair_index = action.get("pair_index")
        value = action.get("value")
        source_value = action.get("source_value")
        if not isinstance(pair_index, int) or isinstance(pair_index, bool):
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": f"{candidate_id} pair_index must be an integer"})
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            blockers.append({"blocker_class": "unsupported_pair_action_spec", "reason": f"{candidate_id} value must be an integer"})
            continue
        if source_value == value:
            blockers.append({"blocker_class": "source_preserving_noop", "reason": f"{candidate_id} action preserves source value"})
            continue
        candidates.append(dict(row))
    report = {
        "path": _rel(action_spec_json),
        "sha256": _sha256_file(action_spec_json) if action_spec_json.is_file() else None,
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "candidate_count": len(candidates),
        "supported_streams": list(SUPPORTED_STREAMS),
    }
    return report, candidates


def _runtime_contract_report(runtime_contract_json: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    payload = _load_json(runtime_contract_json)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != RUNTIME_CONTRACT_SCHEMA:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "unexpected runtime contract schema"})
    if payload.get("supports_pair_specific_actions") is not True:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "supports_pair_specific_actions must be true"})
    if payload.get("scorer_load_allowed") is not False or payload.get("sidecars_allowed") is not False:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "scorer loads and sidecars must be forbidden"})
    supported = payload.get("supported_streams")
    if not isinstance(supported, list) or not set(SUPPORTED_STREAMS).issubset(set(supported)):
        blockers.append({"blocker_class": "unsupported_stream", "reason": "runtime contract must support bias and region"})
    modes = payload.get("supported_header_modes")
    if not isinstance(modes, list) or "explicit_30" not in modes:
        blockers.append({"blocker_class": "unsupported_runtime_contract", "reason": "explicit_30 header mode is required"})
    report = {
        "path": _rel(runtime_contract_json),
        "sha256": _sha256_file(runtime_contract_json) if runtime_contract_json.is_file() else None,
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "contract": payload,
    }
    return report, None if blockers else payload


def _singleton_report(
    candidate: Mapping[str, Any],
    singleton_dir: Path,
    *,
    expected_singletons: Mapping[str, Mapping[str, Any]],
    expected_source_sha256: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    manifest_path = singleton_dir / candidate_id / "manifest.json"
    preflight_path = singleton_dir / candidate_id / "fixed_runtime_atom_preflight.json"
    blockers: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    preflight: dict[str, Any] = {}
    expected = expected_singletons.get(candidate_id)
    if expected is None:
        blockers.append({"blocker_class": "archive_sha_mismatch", "reason": "candidate is not one of the expected PR85 QRGB singleton atoms", "candidate_id": candidate_id})
    if not manifest_path.is_file():
        blockers.append({"blocker_class": "missing_singleton_manifest", "reason": "singleton manifest is missing", "path": _rel(manifest_path)})
    else:
        manifest = _load_json(manifest_path)
        archive = manifest.get("candidate_archive", {}) if isinstance(manifest.get("candidate_archive"), dict) else {}
        source = manifest.get("source_archive", {}) if isinstance(manifest.get("source_archive"), dict) else {}
        if manifest.get("build_status") != "built" or manifest.get("score_claim") is not False:
            blockers.append({"blocker_class": "missing_singleton_preflight", "reason": "singleton manifest is not a built non-claim archive"})
        if source.get("archive_sha256") != expected_source_sha256:
            blockers.append({"blocker_class": "archive_sha_mismatch", "reason": "singleton source archive SHA does not match PR85 frontier", "actual": source.get("archive_sha256")})
        if expected and (
            archive.get("archive_sha256") != expected["archive_sha256"]
            or archive.get("archive_bytes") != expected["archive_bytes"]
        ):
            blockers.append(
                {
                    "blocker_class": "archive_sha_mismatch",
                    "reason": "singleton archive bytes/SHA do not match expected queued candidate",
                    "expected": expected,
                    "actual": {
                        "archive_bytes": archive.get("archive_bytes"),
                        "archive_sha256": archive.get("archive_sha256"),
                    },
                }
            )
    if not preflight_path.is_file():
        blockers.append({"blocker_class": "missing_preflight", "reason": "singleton fixed-runtime preflight JSON is missing", "path": _rel(preflight_path)})
    else:
        preflight = _load_json(preflight_path)
        preflight_archive = preflight.get("archive", {}) if isinstance(preflight.get("archive"), dict) else {}
        manifest_archive = manifest.get("candidate_archive", {}) if isinstance(manifest.get("candidate_archive"), dict) else {}
        if preflight.get("ready_for_fixed_runtime_exact_eval") is not True:
            blockers.append({"blocker_class": "missing_preflight", "reason": "singleton fixed-runtime preflight is not ready"})
        if manifest_archive and preflight_archive.get("archive_sha256") != manifest_archive.get("archive_sha256"):
            blockers.append({"blocker_class": "archive_sha_mismatch", "reason": "singleton preflight archive SHA does not match manifest"})
    return {
        "candidate_id": candidate_id,
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "manifest_path": _rel(manifest_path),
        "preflight_path": _rel(preflight_path),
        "expected_singleton": expected,
        "archive": manifest.get("candidate_archive") if manifest else None,
    }


def _combo_conflicts(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    pair_seen: dict[int, str] = {}
    segment_pair_seen: dict[tuple[str, int], str] = {}
    id_seen: set[str] = set()
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        if candidate_id in id_seen:
            blockers.append({"blocker_class": "duplicate_pair_segment_conflict", "reason": "candidate repeated in combo", "candidate_id": candidate_id})
        id_seen.add(candidate_id)
        for action in row["actions"]:
            pair_index = int(action["pair_index"])
            stream = str(action["stream"])
            previous_pair = pair_seen.get(pair_index)
            if previous_pair is not None:
                blockers.append(
                    {
                        "blocker_class": "duplicate_pair_segment_conflict",
                        "reason": "combo contains more than one atom for the same pair",
                        "pair_index": pair_index,
                        "first_candidate_id": previous_pair,
                        "second_candidate_id": candidate_id,
                    }
                )
            pair_seen[pair_index] = candidate_id
            key = (stream, pair_index)
            previous_segment = segment_pair_seen.get(key)
            if previous_segment is not None:
                blockers.append(
                    {
                        "blocker_class": "duplicate_pair_segment_conflict",
                        "reason": "combo contains duplicate stream/pair mutation",
                        "stream": stream,
                        "pair_index": pair_index,
                        "first_candidate_id": previous_segment,
                        "second_candidate_id": candidate_id,
                    }
                )
            segment_pair_seen[key] = candidate_id
    return blockers


def _combo_action_spec(combo_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_ids = [str(row["candidate_id"]) for row in combo_rows]
    actions: list[dict[str, Any]] = []
    for row in combo_rows:
        for action in row["actions"]:
            actions.append(
                {
                    "op": "set",
                    "pair_index": int(action["pair_index"]),
                    "stream": str(action["stream"]),
                    "value": int(action["value"]),
                    "source_value": action.get("source_value"),
                    "source_atom_id": action.get("source_atom_id"),
                    "source_artifact_sha256": action.get("source_artifact_sha256"),
                    "rationale": action.get("rationale"),
                }
            )
    return {
        "candidate_id": _safe_combo_id(candidate_ids),
        "header_mode": "explicit_30",
        "combo_source_candidate_ids": candidate_ids,
        "actions": actions,
    }


def _exact_result_rows(path: Path | None) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if path is None:
        return (
            {
                "consumed": False,
                "status": "not_requested",
                "blocker_class": "none",
                "blockers": [],
                "reactivation_criteria": [
                    "Consume exact singleton evidence only after T4/L40S jobs return structured contest_auth_eval JSON.",
                    "Each singleton record must match the queued archive bytes and SHA-256 exactly.",
                    "Each singleton record must be CUDA, n_samples=600, archive.zip -> inflate.sh -> upstream/evaluate.py, and score_claim=false in this planner.",
                    "Combo selection may use singleton components only as selection input; combo score still requires its own exact CUDA auth eval after a lane claim.",
                ],
            },
            {},
        )
    payload = _load_json(path)
    raw_rows = payload.get("results", payload.get("candidates", []))
    if isinstance(raw_rows, Mapping):
        rows = list(raw_rows.values())
    elif isinstance(raw_rows, list):
        rows = raw_rows
    else:
        rows = []
    by_id: dict[str, Mapping[str, Any]] = {}
    blockers: list[dict[str, Any]] = []
    now = dt.datetime.now(dt.timezone.utc)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str):
            blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "exact singleton row is missing candidate_id"})
            continue
        expected = KNOWN_SINGLETONS.get(candidate_id)
        archive = row.get("archive", row.get("candidate_archive", row))
        if not isinstance(archive, Mapping):
            archive = {}
        if expected and archive.get("archive_sha256") != expected["archive_sha256"]:
            blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "singleton exact archive SHA mismatch", "candidate_id": candidate_id})
        if expected and archive.get("archive_bytes") != expected["archive_bytes"]:
            blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "singleton exact archive bytes mismatch", "candidate_id": candidate_id})
        device = str(row.get("device", row.get("provenance", {}).get("device", ""))).lower() if isinstance(row.get("provenance", {}), Mapping) else str(row.get("device", "")).lower()
        if "cuda" not in device:
            blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "singleton exact evidence is not CUDA", "candidate_id": candidate_id})
        if row.get("n_samples", row.get("sample_count")) not in (600, None):
            blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "singleton exact evidence sample count is not 600", "candidate_id": candidate_id})
        generated = row.get("generated_at_utc") or row.get("harvested_at_utc") or row.get("completed_at_utc")
        if isinstance(generated, str):
            try:
                parsed = dt.datetime.fromisoformat(generated.replace("Z", "+00:00"))
            except ValueError:
                blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "singleton exact evidence timestamp is unparsable", "candidate_id": candidate_id})
            else:
                if now - parsed.astimezone(dt.timezone.utc) > dt.timedelta(hours=72):
                    blockers.append({"blocker_class": "stale_exact_singleton_evidence", "reason": "singleton exact evidence is older than 72 hours", "candidate_id": candidate_id})
        by_id[candidate_id] = row
    report = {
        "consumed": True,
        "path": _rel(path),
        "sha256": _sha256_file(path),
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "result_count": len(by_id),
    }
    return report, by_id


def _run_combo_preflight(
    *,
    manifest: Mapping[str, Any],
    source_archive: Path,
    robust_current_dir: Path,
    write_outputs: bool,
) -> dict[str, Any]:
    archive = manifest.get("candidate_archive", {}) if isinstance(manifest.get("candidate_archive"), Mapping) else {}
    archive_path = REPO_ROOT / str(archive.get("archive_path"))
    payload = fixed_preflight.build_preflight(
        archive_path,
        robust_current_dir,
        atom_source_archive=source_archive,
        expected_archive_sha256=archive.get("archive_sha256"),
        expected_member_sha256=archive.get("member_sha256"),
    )
    out_path = archive_path.parent / "fixed_runtime_combo_preflight.json"
    if write_outputs:
        _write_json(out_path, payload)
        manifest_path = archive_path.parent / "manifest.json"
        updated = dict(manifest)
        updated["fixed_runtime_combo_preflight"] = {
            "path": _rel(out_path),
            "ready_for_fixed_runtime_exact_eval": payload.get("ready_for_fixed_runtime_exact_eval"),
            "blocker_count": len(payload.get("blockers", [])),
        }
        if payload.get("ready_for_fixed_runtime_exact_eval") is not True:
            updated["dispatch_unlocked"] = False
            updated["dispatch_gate"] = "blocked_fixed_runtime_combo_preflight"
            updated["blockers"] = list(updated.get("blockers", [])) + [
                {
                    "blocker_class": "missing_preflight",
                    "reason": "combo fixed-runtime preflight is not ready",
                    "preflight_path": _rel(out_path),
                }
            ]
        _write_json(manifest_path, updated)
    return {
        "path": _rel(out_path),
        "ready_for_fixed_runtime_exact_eval": payload.get("ready_for_fixed_runtime_exact_eval"),
        "readiness_status": payload.get("readiness_status"),
        "blocker_count": len(payload.get("blockers", [])),
        "archive_sha256": archive.get("archive_sha256"),
    }


def build_combo_candidates(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    scorer_plan_json: Path = DEFAULT_SCORER_PLAN,
    action_spec_json: Path = DEFAULT_ACTION_SPEC,
    runtime_contract_json: Path = DEFAULT_RUNTIME_CONTRACT,
    singleton_dir: Path = DEFAULT_SINGLETON_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
    singleton_exact_results_json: Path | None = None,
    combo_sizes: Sequence[int] = COMBO_SIZES,
    run_fixed_runtime_preflight: bool = True,
    require_known_pr85_anchor: bool = True,
    expected_singletons: Mapping[str, Mapping[str, Any]] = KNOWN_SINGLETONS,
    write_outputs: bool = True,
) -> dict[str, Any]:
    action_report, action_candidates = _action_candidates(action_spec_json)
    runtime_report, _runtime_contract = _runtime_contract_report(runtime_contract_json)
    exact_report, _exact_rows = _exact_result_rows(singleton_exact_results_json)
    expected_source_sha256 = KNOWN_PR85["archive_sha256"] if require_known_pr85_anchor else _sha256_file(source_archive)
    singleton_reports = [
        _singleton_report(
            row,
            singleton_dir,
            expected_singletons=expected_singletons,
            expected_source_sha256=expected_source_sha256,
        )
        for row in action_candidates
    ]
    singleton_by_id = {row["candidate_id"]: row for row in singleton_reports}

    global_blockers = (
        list(action_report.get("blockers", []))
        + list(runtime_report.get("blockers", []))
        + list(exact_report.get("blockers", []))
        + [
            blocker
            for row in singleton_reports
            for blocker in row.get("blockers", [])
            if isinstance(blocker, Mapping)
        ]
    )

    combo_specs: list[dict[str, Any]] = []
    combo_reports: list[dict[str, Any]] = []
    prebuild_blocked_combo_reports: list[dict[str, Any]] = []
    if not global_blockers:
        for size in combo_sizes:
            for rows in combinations(action_candidates, int(size)):
                candidate_ids = [str(row["candidate_id"]) for row in rows]
                conflicts = _combo_conflicts(rows)
                combo_id = _safe_combo_id(candidate_ids)
                if conflicts:
                    blocked_combo = (
                        {
                            "candidate_id": combo_id,
                            "combo_source_candidate_ids": candidate_ids,
                            "build_status": "blocked",
                            "blocker_class": conflicts[0]["blocker_class"],
                            "blockers": conflicts,
                            "dispatch_unlocked": False,
                            "candidate_archive": None,
                        }
                    )
                    combo_reports.append(blocked_combo)
                    prebuild_blocked_combo_reports.append(blocked_combo)
                    continue
                spec = _combo_action_spec(rows)
                combo_specs.append(spec)
    else:
        for size in combo_sizes:
            for rows in combinations(action_candidates, int(size)):
                candidate_ids = [str(row["candidate_id"]) for row in rows]
                combo_reports.append(
                    {
                        "candidate_id": _safe_combo_id(candidate_ids),
                        "combo_source_candidate_ids": candidate_ids,
                        "build_status": "blocked",
                        "blocker_class": global_blockers[0]["blocker_class"],
                        "blockers": global_blockers,
                        "dispatch_unlocked": False,
                        "candidate_archive": None,
                    }
                )

    combo_action_spec = {
        "schema": ACTION_SPEC_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "inflate_time_scorer_load_allowed": False,
        "candidates": combo_specs,
    }
    combo_action_spec_path = out_dir / "action_spec_combos.json"
    if write_outputs:
        _write_json(combo_action_spec_path, combo_action_spec)

    pair_summary: dict[str, Any] | None = None
    if combo_specs and not global_blockers:
        pair_summary = pair_builder.build_pair_atom_candidates(
            source_archive=source_archive,
            scorer_plan_json=scorer_plan_json,
            action_spec_json=combo_action_spec_path,
            runtime_contract_json=runtime_contract_json,
            out_dir=out_dir,
            top_pairs=tuple(sorted({int(action["pair_index"]) for spec in combo_specs for action in spec["actions"]})),
            require_known_pr85_anchor=require_known_pr85_anchor,
            top_limit=8,
            write_outputs=write_outputs,
        )
        combo_reports = prebuild_blocked_combo_reports + list(pair_summary.get("candidates", []))

    preflight_reports: dict[str, dict[str, Any]] = {}
    if run_fixed_runtime_preflight:
        for manifest in combo_reports:
            if manifest.get("build_status") != "built":
                continue
            preflight = _run_combo_preflight(
                manifest=manifest,
                source_archive=source_archive,
                robust_current_dir=robust_current_dir,
                write_outputs=write_outputs,
            )
            preflight_reports[str(manifest["candidate_id"])] = preflight
            manifest["fixed_runtime_combo_preflight"] = preflight
            if preflight.get("ready_for_fixed_runtime_exact_eval") is not True:
                manifest["dispatch_unlocked"] = False
                manifest["dispatch_gate"] = "blocked_fixed_runtime_combo_preflight"
    else:
        for manifest in combo_reports:
            if manifest.get("build_status") == "built":
                manifest["dispatch_unlocked"] = False
                manifest["dispatch_gate"] = "blocked_fixed_runtime_combo_preflight_not_run"
                manifest["blockers"] = list(manifest.get("blockers", [])) + [
                    {
                        "blocker_class": "missing_preflight",
                        "reason": "combo fixed-runtime preflight was not run",
                    }
                ]

    built = [row for row in combo_reports if row.get("build_status") == "built"]
    ready = [
        row
        for row in built
        if row.get("fixed_runtime_combo_preflight", {}).get("ready_for_fixed_runtime_exact_eval") is True
    ]
    all_blockers = global_blockers + [
        blocker
        for row in combo_reports
        for blocker in row.get("blockers", [])
        if isinstance(blocker, Mapping)
    ]
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "scorer_load_performed": False,
        "source_archive": {
            "path": _rel(source_archive),
            "archive_bytes": source_archive.stat().st_size if source_archive.is_file() else None,
            "archive_sha256": _sha256_file(source_archive) if source_archive.is_file() else None,
            "known_pr85_anchor": KNOWN_PR85,
        },
        "action_spec": action_report,
        "runtime_contract": runtime_report,
        "singleton_preflight_gate": {
            "singleton_dir": _rel(singleton_dir),
            "status": "passed" if all(row["status"] == "passed" for row in singleton_reports) else "blocked",
            "singletons": singleton_reports,
        },
        "exact_singleton_evidence": exact_report,
        "combo_action_spec_path": _rel(combo_action_spec_path),
        "pair_builder_summary_path": _rel(out_dir / "planning.json"),
        "combo_attempt_count": len(combo_reports),
        "combo_archive_count": len(built),
        "fixed_runtime_combo_preflight_count": len(preflight_reports),
        "ready_combo_count": len(ready),
        "dispatch_unlocked": bool(ready),
        "blocker_class": "none" if not all_blockers else str(all_blockers[0]["blocker_class"]),
        "blockers": all_blockers,
        "candidates": combo_reports,
        "reactivation_criteria": exact_report.get("reactivation_criteria", []),
    }
    if write_outputs:
        _write_json(out_dir / "combo_planning.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--scorer-plan-json", type=Path, default=DEFAULT_SCORER_PLAN)
    parser.add_argument("--action-spec-json", type=Path, default=DEFAULT_ACTION_SPEC)
    parser.add_argument("--runtime-contract-json", type=Path, default=DEFAULT_RUNTIME_CONTRACT)
    parser.add_argument("--singleton-dir", type=Path, default=DEFAULT_SINGLETON_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--singleton-exact-results-json", type=Path, default=None)
    parser.add_argument("--combo-size", type=int, action="append", dest="combo_sizes")
    parser.add_argument("--skip-fixed-runtime-preflight", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_combo_candidates(
        source_archive=args.source_archive,
        scorer_plan_json=args.scorer_plan_json,
        action_spec_json=args.action_spec_json,
        runtime_contract_json=args.runtime_contract_json,
        singleton_dir=args.singleton_dir,
        out_dir=args.out_dir,
        robust_current_dir=args.robust_current_dir,
        singleton_exact_results_json=args.singleton_exact_results_json,
        combo_sizes=tuple(args.combo_sizes or COMBO_SIZES),
        run_fixed_runtime_preflight=not args.skip_fixed_runtime_preflight,
    )
    if args.stdout:
        sys.stdout.write(_json_text(summary))
    else:
        print(
            _json_text(
                {
                    "combo_planning_json": _rel(args.out_dir / "combo_planning.json"),
                    "combo_archive_count": summary["combo_archive_count"],
                    "ready_combo_count": summary["ready_combo_count"],
                    "dispatch_unlocked": summary["dispatch_unlocked"],
                    "blocker_class": summary["blocker_class"],
                }
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
