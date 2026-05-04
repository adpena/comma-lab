#!/usr/bin/env python3
"""Build local PR91 HPM1 + QRGB transfer candidates.

This tool is intentionally local-only. It consumes explicit PR85 QRGB
pair-action specs and applies them to PR91 only when the touched side-channel
segments are byte-identical between PR85 and PR91 and the action's declared
source value matches the PR91 decoded stream. It does not consume or synthesize
PR91 scorer gradients and it keeps dispatch locked.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

try:
    from experiments import build_pr85_pair_atom_candidates as pair_builder  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    import build_pr85_pair_atom_candidates as pair_builder  # type: ignore[no-redef]  # noqa: E402
from tac.pr85_bundle import (  # noqa: E402
    Pr85BundleError,
    SEGMENT_ORDER,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/build_pr91_qrgb_pair_atom_candidates.py"
SCHEMA = "pr91_hpm1_qrgb_pair_atom_transfer_v1"
SUPPORTED_STREAMS = ("bias", "region")
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT / "experiments/results/public_pr91_intake_20260504_worker/archive.zip"
)
DEFAULT_ACTION_SPEC = (
    REPO_ROOT
    / "experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/action_spec_bias_region.json"
)
DEFAULT_SEGMENT_DIFF = (
    REPO_ROOT / "experiments/results/public_pr91_intake_20260504_worker/pr91_vs_pr85_segment_diff.json"
)
DEFAULT_TRANSFER_DECISIONS = (
    REPO_ROOT / "experiments/results/public_pr91_intake_20260504_worker/pr91_transfer_decisions.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/pr91_qrgb_pair_atom_candidates_20260504_codex"
PR91_ARCHIVE_SHA256 = "4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f"
PR91_ARCHIVE_BYTES = 222_404


class Pr91QrgbBuilderError(ValueError):
    """Raised when PR91 QRGB transfer inputs are malformed."""


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


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
        raise Pr91QrgbBuilderError(f"{_rel(path)} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise Pr91QrgbBuilderError(f"{_rel(path)} must contain a JSON object")
    return payload


def _safe_candidate_id(value: str) -> str:
    if not value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in value):
        raise Pr91QrgbBuilderError(f"unsafe candidate_id: {value!r}")
    return value


def _read_single_x_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise Pr91QrgbBuilderError(f"source archive is missing: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise Pr91QrgbBuilderError(f"archive must contain exactly one member 'x'; got {names!r}")
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    return (
        {
            "path": _rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": pair_builder._sha256_file(path),
            "member_name": info.filename,
            "member_bytes": int(info.file_size),
            "member_sha256": pair_builder._sha256_bytes(raw),
            "zip_stored": info.compress_type == zipfile.ZIP_STORED,
            "known_pr91_anchor_match": {
                "matches": (
                    int(path.stat().st_size) == PR91_ARCHIVE_BYTES
                    and pair_builder._sha256_file(path) == PR91_ARCHIVE_SHA256
                ),
                "expected_archive_bytes": PR91_ARCHIVE_BYTES,
                "expected_archive_sha256": PR91_ARCHIVE_SHA256,
            },
        },
        raw,
    )


def _load_action_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path)
    blockers: list[dict[str, Any]] = []
    if payload.get("schema") != pair_builder.ACTION_SPEC_SCHEMA:
        blockers.append({"blocker_class": "unsupported_action_spec", "reason": "unexpected action spec schema"})
    if payload.get("score_claim") is not False:
        blockers.append({"blocker_class": "unsupported_action_spec", "reason": "score_claim must be false"})
    for key in ("dispatch_performed", "remote_jobs_dispatched", "inflate_time_scorer_load_allowed"):
        if payload.get(key) not in (False, None):
            blockers.append({"blocker_class": "unsupported_action_spec", "reason": f"{key} must be false"})
    raw_rows = payload.get("candidates")
    if not isinstance(raw_rows, list):
        blockers.append({"blocker_class": "unsupported_action_spec", "reason": "candidates must be a list"})
        raw_rows = []

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            blockers.append({"blocker_class": "unsupported_action_spec", "reason": "candidate row must be an object"})
            continue
        candidate_id = row.get("candidate_id")
        actions = row.get("actions")
        if not isinstance(candidate_id, str):
            blockers.append({"blocker_class": "unsupported_action_spec", "reason": "candidate_id must be a string"})
            continue
        _safe_candidate_id(candidate_id)
        if not isinstance(actions, list) or not actions:
            blockers.append({"blocker_class": "unsupported_action_spec", "reason": f"{candidate_id} has no actions"})
            continue
        for action in actions:
            if not isinstance(action, Mapping):
                blockers.append({"blocker_class": "unsupported_action_spec", "reason": f"{candidate_id} action must be an object"})
                continue
            if action.get("stream") not in SUPPORTED_STREAMS:
                blockers.append({"blocker_class": "unsupported_stream", "reason": "PR91 QRGB transfer is limited to bias/region", "candidate_id": candidate_id, "stream": action.get("stream")})
            if not isinstance(action.get("pair_index"), int) or isinstance(action.get("pair_index"), bool):
                blockers.append({"blocker_class": "unsupported_action_spec", "reason": f"{candidate_id} pair_index must be an integer"})
            if not isinstance(action.get("value"), int) or isinstance(action.get("value"), bool):
                blockers.append({"blocker_class": "unsupported_action_spec", "reason": f"{candidate_id} value must be an integer"})
            if not isinstance(action.get("source_value"), int) or isinstance(action.get("source_value"), bool):
                blockers.append({"blocker_class": "missing_source_value", "reason": f"{candidate_id} action must carry source_value for PR91 transfer"})
        rows.append(dict(row))
    report = {
        "path": _rel(path),
        "sha256": pair_builder._sha256_file(path) if path.is_file() else None,
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "candidate_count": len(rows),
        "supported_streams": list(SUPPORTED_STREAMS),
    }
    return report, rows


def _segment_identity_report(path: Path, required_streams: Sequence[str]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": _rel(path),
            "status": "blocked",
            "blocker_class": "missing_pr91_pr85_segment_diff",
            "blockers": [{"blocker_class": "missing_pr91_pr85_segment_diff", "reason": "PR91/PR85 segment diff JSON is missing"}],
            "identical_streams": [],
        }
    payload = _load_json(path)
    rows = payload.get("segments")
    if not isinstance(rows, list):
        rows = []
    by_name = {row.get("name"): row for row in rows if isinstance(row, Mapping)}
    blockers: list[dict[str, Any]] = []
    for stream in sorted(set(required_streams)):
        row = by_name.get(stream)
        if not isinstance(row, Mapping) or row.get("sha_equal") is not True:
            blockers.append(
                {
                    "blocker_class": "pr91_pr85_source_mismatch",
                    "reason": "touched stream is not byte-identical between PR85 and PR91",
                    "stream": stream,
                    "segment_diff": row,
                }
            )
    return {
        "path": _rel(path),
        "sha256": pair_builder._sha256_file(path),
        "status": "passed" if not blockers else "blocked",
        "blocker_class": "none" if not blockers else blockers[0]["blocker_class"],
        "blockers": blockers,
        "identical_streams": [
            str(name)
            for name, row in by_name.items()
            if isinstance(row, Mapping) and row.get("sha_equal") is True
        ],
    }


def _transfer_decision_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": _rel(path), "status": "missing", "blockers": []}
    payload = _load_json(path)
    return {
        "path": _rel(path),
        "sha256": pair_builder._sha256_file(path),
        "status": "loaded",
        "evidence_grade": payload.get("evidence_grade"),
        "source_blockers": payload.get("blockers", []),
    }


def _candidate_id(source_id: str) -> str:
    suffix = source_id.removeprefix("pr85_")
    return _safe_candidate_id(f"pr91_hpm1_{suffix}")


def _candidate_blocked(candidate_id: str, blocker_class: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": "pr91_hpm1_qrgb_pair_atom_candidate_v1",
        "tool": TOOL,
        "candidate_id": candidate_id,
        "build_status": "blocked",
        "blocker_class": blocker_class,
        "blockers": [{"blocker_class": blocker_class, "reason": reason, **extra}],
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "dispatch_unlocked": False,
        "candidate_archive": None,
    }


def _build_candidate(
    *,
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    source_raw: bytes,
    out_dir: Path,
) -> dict[str, Any]:
    source_id = str(row["candidate_id"])
    candidate_id = _candidate_id(source_id)
    bundle = parse_pr85_bundle(source_raw)
    contracts = bundle.segment_contracts
    mask_contract = contracts["mask"]
    if mask_contract.codec != "HPM1":
        return _candidate_blocked(candidate_id, "source_not_pr91_hpm1", "source mask segment is not typed HPM1")

    source_segments = {name: bytes(bundle.segments[name]) for name in SEGMENT_ORDER}
    actions = row.get("actions")
    if not isinstance(actions, list):
        return _candidate_blocked(candidate_id, "unsupported_action_spec", "actions must be a list")
    streams = sorted({str(action["stream"]) for action in actions if isinstance(action, Mapping)})
    values_by_stream: dict[str, bytearray] = {}
    stream_reports: dict[str, dict[str, Any]] = {}
    try:
        for stream in streams:
            values, report = pair_builder._decode_choice_stream(stream, source_segments[stream])
            values_by_stream[stream] = values
            stream_reports[stream] = report
    except Exception as exc:
        return _candidate_blocked(candidate_id, "sidechannel_decode_failed", str(exc))

    changed = False
    action_proofs: list[dict[str, Any]] = []
    for action in actions:
        stream = str(action["stream"])
        pair_index = int(action["pair_index"])
        source_value = int(action["source_value"])
        candidate_value = int(action["value"])
        values = values_by_stream[stream]
        actual_source_value = int(values[pair_index])
        if actual_source_value != source_value:
            return _candidate_blocked(
                candidate_id,
                "pr91_source_value_mismatch",
                "PR85 action source_value does not match PR91 decoded stream",
                stream=stream,
                pair_index=pair_index,
                expected_source_value=source_value,
                actual_source_value=actual_source_value,
            )
        values[pair_index] = candidate_value
        changed_here = actual_source_value != candidate_value
        changed = changed or changed_here
        action_proofs.append(
            {
                "stream": stream,
                "pair_index": pair_index,
                "source_value": actual_source_value,
                "candidate_value": candidate_value,
                "changed": changed_here,
                "source_atom_id": action.get("source_atom_id"),
                "source_artifact_sha256": action.get("source_artifact_sha256"),
                "rationale": action.get("rationale"),
            }
        )
    if not changed:
        return _candidate_blocked(candidate_id, "source_preserving_noop", "all actions preserved PR91 source values")

    candidate_segments = dict(source_segments)
    transforms: list[dict[str, Any]] = []
    try:
        for stream in streams:
            encoded, encode_meta = pair_builder._encode_choice_stream(
                stream,
                bytes(values_by_stream[stream]),
                codec=stream_reports[stream]["codec"],
            )
            candidate_segments[stream] = encoded
            transforms.append(
                {
                    **stream_reports[stream],
                    **encode_meta,
                    "segment_byte_delta": len(encoded) - len(source_segments[stream]),
                    "semantic_sha256_before": pair_builder._sha256_bytes(
                        bytes(pair_builder._decode_choice_stream(stream, source_segments[stream])[0])
                    ),
                    "semantic_sha256_after": pair_builder._sha256_bytes(bytes(values_by_stream[stream])),
                    "changed_pair_indices": [
                        int(action["pair_index"])
                        for action in actions
                        if isinstance(action, Mapping) and action.get("stream") == stream
                    ],
                }
            )
        payload = pack_pr85_bundle(candidate_segments, header_mode="explicit_30")
        reparsed = parse_pr85_bundle(payload)
    except (Pr85BundleError, Exception) as exc:
        return _candidate_blocked(candidate_id, "candidate_bundle_validation_failed", str(exc))
    if {name: bytes(reparsed.segments[name]) for name in SEGMENT_ORDER} != candidate_segments:
        return _candidate_blocked(candidate_id, "candidate_bundle_validation_failed", "reparsed segments do not match")
    if bytes(reparsed.segments["mask"]) != source_segments["mask"]:
        return _candidate_blocked(candidate_id, "hpm1_mask_changed", "QRGB transfer changed the PR91 HPM1 mask segment")

    candidate_dir = out_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    pair_builder._write_single_member_archive(archive_path, payload)
    candidate_archive = pair_builder._archive_info(archive_path)
    changed_segments = [name for name in SEGMENT_ORDER if candidate_segments[name] != source_segments[name]]
    manifest = {
        "schema": "pr91_hpm1_qrgb_pair_atom_candidate_v1",
        "tool": TOOL,
        "candidate_id": candidate_id,
        "source_pr85_candidate_id": source_id,
        "build_status": "built",
        "evidence_grade": "empirical_local_archive_build_only",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "scorer_gradient_claim": False,
        "pr91_scorer_gradient_consumed": False,
        "source_archive": dict(source),
        "candidate_archive": candidate_archive,
        "source_bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "member_sha256": pair_builder._sha256_bytes(source_raw),
            "segment_lengths": bundle.segment_lengths,
            "mask_contract": {
                "codec": mask_contract.codec,
                "sha256": mask_contract.sha256,
                "metadata": dict(mask_contract.metadata),
            },
        },
        "candidate_bundle": {
            "header_mode": "explicit_30",
            "member_bytes": len(payload),
            "member_sha256": pair_builder._sha256_bytes(payload),
            "segment_lengths": {name: len(candidate_segments[name]) for name in SEGMENT_ORDER},
            "mask_contract_preserved": True,
        },
        "transfer_basis": {
            "source": "PR85 explicit QRGB action spec transferred only over PR85-identical PR91 side-channel streams",
            "no_pr91_gradient_synthesized": True,
            "action_spec_source": _rel(DEFAULT_ACTION_SPEC),
        },
        "selected_pair_indices": sorted({int(action["pair_index"]) for action in actions if isinstance(action, Mapping)}),
        "selected_streams": streams,
        "action_proofs": action_proofs,
        "transforms": transforms,
        "changed_segments": changed_segments,
        "charged_bytes": {
            "candidate_archive_bytes": candidate_archive["archive_bytes"],
            "source_archive_bytes": source["archive_bytes"],
            "byte_delta_vs_pr91_source_archive": int(candidate_archive["archive_bytes"] - source["archive_bytes"]),
            "formula_only_rate_score_delta_vs_pr91": (
                int(candidate_archive["archive_bytes"] - source["archive_bytes"])
                * pair_builder.RATE_SCORE_PER_BYTE
            ),
        },
        "non_noop_proof": {
            "status": "passed",
            "payload_changed": pair_builder._sha256_bytes(source_raw) != pair_builder._sha256_bytes(payload),
            "decoded_sidechannel_semantics_changed": True,
            "hpm1_mask_unchanged": True,
            "changed_segments": changed_segments,
        },
        "dispatch_unlocked": False,
        "dispatch_gate": "blocked_pr91_hpm1_runtime_preflight_and_exact_evidence",
        "next_gate": "Add reviewed PR91 HPM1 inflate support, run local runtime parity/preflight, then claim lane before exact CUDA auth eval.",
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def build_pr91_qrgb_pair_atom_candidates(
    *,
    source_archive: Path = DEFAULT_SOURCE_ARCHIVE,
    action_spec_json: Path = DEFAULT_ACTION_SPEC,
    segment_diff_json: Path = DEFAULT_SEGMENT_DIFF,
    transfer_decisions_json: Path = DEFAULT_TRANSFER_DECISIONS,
    out_dir: Path = DEFAULT_OUT_DIR,
    write_outputs: bool = True,
) -> dict[str, Any]:
    source, source_raw = _read_single_x_archive(source_archive)
    bundle = parse_pr85_bundle(source_raw)
    source_contracts = bundle.segment_contracts
    action_report, rows = _load_action_rows(action_spec_json)
    required_streams = [
        str(action["stream"])
        for row in rows
        for action in row.get("actions", [])
        if isinstance(action, Mapping) and isinstance(action.get("stream"), str)
    ]
    identity_report = _segment_identity_report(segment_diff_json, required_streams)
    transfer_report = _transfer_decision_report(transfer_decisions_json)
    global_blockers = list(action_report["blockers"]) + list(identity_report["blockers"])
    candidates: list[dict[str, Any]] = []
    if not global_blockers:
        for row in rows:
            candidates.append(
                _build_candidate(
                    row=row,
                    source=source,
                    source_raw=source_raw,
                    out_dir=out_dir,
                )
            )
    else:
        for row in rows:
            candidates.append(
                _candidate_blocked(
                    _candidate_id(str(row.get("candidate_id", "unknown"))),
                    str(global_blockers[0]["blocker_class"]),
                    str(global_blockers[0]["reason"]),
                )
            )
    candidate_blockers = [
        blocker
        for row in candidates
        for blocker in row.get("blockers", [])
        if isinstance(blocker, Mapping)
    ]
    built = [row for row in candidates if row.get("build_status") == "built"]
    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "scorer_load_performed": False,
        "source_archive": source,
        "source_bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
            "mask_contract": {
                "codec": source_contracts["mask"].codec,
                "sha256": source_contracts["mask"].sha256,
                "metadata": dict(source_contracts["mask"].metadata),
            },
        },
        "action_spec": action_report,
        "segment_identity": identity_report,
        "transfer_decisions": transfer_report,
        "candidate_attempt_count": len(candidates),
        "candidate_archive_count": len(built),
        "dispatch_unlocked_count": 0,
        "dispatch_unlocked": False,
        "blocker_class": "none" if not (global_blockers + candidate_blockers) else str((global_blockers + candidate_blockers)[0]["blocker_class"]),
        "blockers": global_blockers + candidate_blockers,
        "candidates": candidates,
        "planning_json_path": _rel(out_dir / "planning.json"),
        "next_dispatch_recommendation": "Do not dispatch yet; candidate archives are local byte-closed design artifacts pending PR91 HPM1 runtime/preflight parity and a lane claim.",
    }
    if write_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "planning.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--action-spec-json", type=Path, default=DEFAULT_ACTION_SPEC)
    parser.add_argument("--segment-diff-json", type=Path, default=DEFAULT_SEGMENT_DIFF)
    parser.add_argument("--transfer-decisions-json", type=Path, default=DEFAULT_TRANSFER_DECISIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_pr91_qrgb_pair_atom_candidates(
        source_archive=args.source_archive,
        action_spec_json=args.action_spec_json,
        segment_diff_json=args.segment_diff_json,
        transfer_decisions_json=args.transfer_decisions_json,
        out_dir=args.out_dir,
    )
    if args.stdout:
        sys.stdout.write(_json_text(summary))  # DETERMINISTIC_ZIP_OK — stdout JSON, NOT inside any ZipFile context
    else:
        print(
            _json_text(
                {
                    "planning_json": summary["planning_json_path"],
                    "candidate_archive_count": summary["candidate_archive_count"],
                    "dispatch_unlocked": summary["dispatch_unlocked"],
                    "blocker_class": summary["blocker_class"],
                }
            ),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
