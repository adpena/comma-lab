#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local PR65 qpost x PR75 action interaction candidates.

This worker is local-only.  It combines reviewed PR65 qpost bias atoms with
existing PR75/C089 action-runtime-compatible archives, records byte/trace
provenance, and refuses active exact-eval duplicates.  It never dispatches
remote GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CORE_BUILDER_PATH = REPO_ROOT / "experiments/build_pr65_qpost_atom_candidates.py"
PRODUCER = "experiments/build_pr65_qpost_interaction_candidates.py"
SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr65_qpost_interaction_worker_20260503"
DEFAULT_ACTIVE_CLAIMS = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
DEFAULT_TARGET_SCORE = 0.314  # [heuristic: aspirational floor below PR-65 frontier 0.3155]
BASELINE_SCORE = 0.3154707273953505  # [external: PR-65 contest-CUDA T4 frontier]
RATE_DENOM = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / RATE_DENOM
EXPECTED_C089_SHA256 = "0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8"
NONTERMINAL_CLAIM_STATUSES = {
    "eval",
    "exact_eval_queued",
    "provisioning",
    "training",
}


class QPostInteractionError(ValueError):
    """Raised when interaction candidate planning fails a closed-world guard."""


@dataclass(frozen=True)
class ActionBase:
    base_id: str
    archive: Path
    exact_score: float
    exact_score_source: str
    expected_sha256: str
    runtime_family: str = "pr75_p6_single_member"


@dataclass(frozen=True)
class QPostSpec:
    family_id: str
    include_streams: tuple[str, ...]
    top_pairs: int
    risk_family: str


DEFAULT_ACTION_BASES: tuple[ActionBase, ...] = (
    ActionBase(
        "c089_top40_p6",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip",
        BASELINE_SCORE,
        "A++ T4 exact eval C089 frontier",
        EXPECTED_C089_SHA256,
    ),
    ActionBase(
        "pr75_lagtop67_p6",
        REPO_ROOT
        / "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
        "c067_pr75_actions_lag_eval_top67_p6/archive.zip",
        0.3154979650614253,
        "A++ T4 no-frontier exact eval exact_eval_c067_pr75_qp1_lag_eval_top67_p6_t4_20260503T0608Z",
        "d73f0957cf2d8da0526c1786613443331ccb913f5648131cfaaac5e6f7eae972",
    ),
    ActionBase(
        "pr75_pose2_top67_p6",
        REPO_ROOT
        / "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
        "c067_pr75_actions_lag_eval_pose2_top67_p6/archive.zip",
        0.3155227462217741,
        "A++ T4 no-frontier exact eval exact_eval_c067_pr75_qp1_lag_eval_pose2_top67_p6_t4_20260503T0608Z",
        "af7a34cb1c051b1accebe2768245a44f55280e2596b315f8e4809a73a23926cd",
    ),
    ActionBase(
        "pr75_pose_safe_ampm1_p6",
        REPO_ROOT
        / "experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/"
        "action_compiler_candidates/c067_pr75_actions_pose_safe_positive_ampminus1_p6/archive.zip",
        0.31556196759570776,
        "A++ T4 no-frontier exact eval exact_eval_c067_pr75_qp1_pose_safe_positive_ampminus1_p6_t4_20260503T0632Z",
        "6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796",
    ),
)

DEFAULT_QPOST_SPECS: tuple[QPostSpec, ...] = (
    QPostSpec("bias_top040", ("bias",), 40, "low_byte_bias_only_interaction"),
    QPostSpec("bias_top080", ("bias",), 80, "low_byte_bias_only_interaction_scale"),
)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_core_builder():
    spec = importlib.util.spec_from_file_location("pact_pr65_qpost_interaction_core", CORE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise QPostInteractionError(f"cannot load core qpost builder: {CORE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_single_member_magic(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        if names != ["p"]:
            raise QPostInteractionError(f"{path} must be a single-member PR75/C089 archive, got {names!r}")
        payload = zf.read("p")
    if len(payload) < 2:
        raise QPostInteractionError(f"{path} payload is too short")
    magic = payload[:2].decode("ascii", errors="replace")
    if magic != "P6":
        raise QPostInteractionError(f"{path} is not a reviewed P6 action-runtime archive, got magic {magic!r}")
    return {
        "member_names": names,
        "payload_magic": magic,
        "payload_bytes": len(payload),
        "runtime_compatible": True,
    }


def _verify_action_base(base: ActionBase, *, allow_source_sha_mismatch: bool) -> dict[str, Any]:
    archive = base.archive.resolve()
    if not archive.is_file():
        raise QPostInteractionError(f"missing action base archive: {archive}")
    sha256 = _sha256_path(archive)
    if not allow_source_sha_mismatch and sha256 != base.expected_sha256:
        raise QPostInteractionError(
            f"action base {base.base_id} SHA mismatch: expected {base.expected_sha256}, got {sha256}"
        )
    return {
        "base_id": base.base_id,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256,
        "expected_sha256": base.expected_sha256,
        "exact_score": base.exact_score,
        "exact_score_source": base.exact_score_source,
        "runtime_family": base.runtime_family,
        "runtime_probe": _read_single_member_magic(archive),
    }


def _claim_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line.startswith("| 20"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 8:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _active_claims(path: Path) -> list[dict[str, str]]:
    return [
        row
        for row in _claim_rows(path)
        if row["status"] in NONTERMINAL_CLAIM_STATUSES
    ]


def _active_duplicate_status(
    *,
    source_sha256: str,
    qpost: QPostSpec,
    candidate_archive_sha256: str | None,
    active_claims: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    matches: list[dict[str, str]] = []
    active_qpost_names = {
        f"pr65_qpost_bias_poseadv_top{qpost.top_pairs:03d}",
        f"v2_pr65_qpost_bias_poseadv_top{qpost.top_pairs:03d}",
        f"pr65_qpost_v2_bias_poseadv_top{qpost.top_pairs:03d}",
    }
    for row in active_claims:
        haystack = " ".join(
            str(row.get(key, ""))
            for key in ("lane_id", "instance_job_id", "notes", "status")
        )
        if candidate_archive_sha256 and candidate_archive_sha256 in haystack:
            matches.append(dict(row))
            continue
        if source_sha256 == EXPECTED_C089_SHA256 and any(name in haystack for name in active_qpost_names):
            matches.append(dict(row))
    return {
        "duplicates_active_job": bool(matches),
        "matched_claims": matches,
        "active_claim_count": len(active_claims),
        "source_sha256": source_sha256,
        "qpost_top_pairs": qpost.top_pairs,
    }


def _candidate_id(base_id: str, qpost: QPostSpec) -> str:
    return f"ix_{base_id}_{qpost.family_id}"


def _qpost_spec_to_core(core: Any, base_id: str, qpost: QPostSpec) -> Any:
    return core.CandidateSpec(
        _candidate_id(base_id, qpost),
        tuple(qpost.include_streams),
        int(qpost.top_pairs),
        qpost.risk_family,
    )


def _screen_candidate(
    *,
    row: Mapping[str, Any],
    base: Mapping[str, Any],
    qpost: QPostSpec,
    target_score: float,
    duplicate_status: Mapping[str, Any],
) -> dict[str, Any]:
    added_rate = float(row.get("formula_rate_score_delta", 0.0))
    source_score = float(base["exact_score"])
    trace_opportunity = float(row.get("public_trace_opportunity_bound", 0.0))
    break_even = (source_score - target_score) + added_rate
    trace_slack = trace_opportunity - break_even
    no_op = not bool(row.get("built")) or int(row.get("selected_active_atoms_total", 0)) <= 0
    core_blockers = [str(value) for value in row.get("dispatch_blockers", [])]
    dispatchable_later = (
        bool(row.get("built"))
        and not no_op
        and not duplicate_status["duplicates_active_job"]
        and not core_blockers
        and trace_slack > 0.0
    )
    blockers: list[str] = []
    if no_op:
        blockers.append("no non-noop qpost atoms selected")
    if duplicate_status["duplicates_active_job"]:
        blockers.append("matches active qpost exact-eval job")
    blockers.extend(core_blockers)
    if trace_slack <= 0.0:
        blockers.append(
            f"trace opportunity {trace_opportunity:.9f} does not clear sub-0.314 break-even {break_even:.9f}"
        )
    return {
        "candidate_id": row.get("candidate_id"),
        "score_claim": False,
        "dispatchable_later": dispatchable_later,
        "remote_dispatched": False,
        "target_score": target_score,
        "evidence_grade": "byte_trace_planning_only_until_exact_cuda",
        "source_action_base": base,
        "qpost_spec": {
            "family_id": qpost.family_id,
            "include_streams": list(qpost.include_streams),
            "top_pairs": qpost.top_pairs,
            "risk_family": qpost.risk_family,
        },
        "archive": row.get("archive"),
        "archive_bytes": row.get("archive_bytes"),
        "archive_sha256": row.get("archive_sha256"),
        "archive_byte_delta_vs_source": row.get("archive_byte_delta"),
        "qpost_bytes": row.get("qpost_bytes"),
        "selected_pairs": list(row.get("selected_pairs", [])),
        "selected_active_atoms_total": row.get("selected_active_atoms_total"),
        "pair_selection_provenance": "PR65 qpost bias atoms ranked by C089-vs-PR65 public component trace opportunity",
        "trace_opportunity_estimates": {
            "public_trace_opportunity_bound": trace_opportunity,
            "source_exact_score": source_score,
            "added_rate_score": added_rate,
            "break_even_component_gain_for_target": break_even,
            "trace_slack_vs_target": trace_slack,
        },
        "active_job_duplication": duplicate_status,
        "core_dispatch_blockers": core_blockers,
        "dispatch_blockers": blockers,
    }


def _exact_eval_command(archive: str, candidate_id: str, output_dir: Path) -> str:
    work_dir = output_dir / "exact_eval_work" / candidate_id
    return (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {archive} "
        "--inflate-sh submissions/robust_current/inflate.sh "
        "--upstream-dir upstream --device cuda --keep-work-dir "
        f"--work-dir {work_dir}"
    )


def _write_interaction_manifest(
    *,
    candidate_dir: Path,
    screen: Mapping[str, Any],
    selected_pair_rank_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    candidate_id = str(screen["candidate_id"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "dispatchable_later": bool(screen["dispatchable_later"]),
        "remote_dispatch": {
            "dispatched": False,
            "requires_lane_claim_before_any_eval": True,
            "note": "Local builder only; no remote GPU, training, or eval job was dispatched.",
        },
        "candidate_screen": screen,
        "selected_pair_rank_records": list(selected_pair_rank_records),
        "exact_eval_command_template": _exact_eval_command(
            str(screen["archive"]), candidate_id, output_dir
        ),
    }
    _write_json(candidate_dir / "interaction_manifest.json", manifest)


def _best_candidate(screens: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    dispatchable = [row for row in screens if row.get("dispatchable_later")]
    if not dispatchable:
        return None
    return sorted(
        dispatchable,
        key=lambda row: (
            -float((row["trace_opportunity_estimates"])["trace_slack_vs_target"]),
            int(row.get("archive_bytes") or 0),
            str(row.get("candidate_id")),
        ),
    )[0]


def build_interactions(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    active_claims_path: Path = DEFAULT_ACTIVE_CLAIMS,
    action_bases: Sequence[ActionBase] = DEFAULT_ACTION_BASES,
    qpost_specs: Sequence[QPostSpec] = DEFAULT_QPOST_SPECS,
    target_score: float = DEFAULT_TARGET_SCORE,
    allow_source_sha_mismatch: bool = False,
) -> dict[str, Any]:
    core = _load_core_builder()
    output_dir = output_dir.resolve()
    active_claims = _active_claims(active_claims_path.resolve())
    verified_bases = [
        _verify_action_base(base, allow_source_sha_mismatch=allow_source_sha_mismatch)
        for base in action_bases
    ]
    screens: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for base_spec, base in zip(action_bases, verified_bases, strict=True):
        for qpost in qpost_specs:
            preflight_duplicate = _active_duplicate_status(
                source_sha256=str(base["archive_sha256"]),
                qpost=qpost,
                candidate_archive_sha256=None,
                active_claims=active_claims,
            )
            if preflight_duplicate["duplicates_active_job"]:
                skipped.append(
                    {
                        "candidate_id": _candidate_id(base_spec.base_id, qpost),
                        "score_claim": False,
                        "dispatchable_later": False,
                        "skip_reason": "active_job_duplicate_preflight",
                        "active_job_duplication": preflight_duplicate,
                        "source_action_base": base,
                    }
                )
                continue
            candidate_root = output_dir / "candidates" / base_spec.base_id
            matrix = core.build_matrix(
                source_archive=Path(str(base["archive"])),
                pr65_archive=core.DEFAULT_PR65_ARCHIVE,
                output_dir=candidate_root,
                anatomy_json=core.DEFAULT_ANATOMY_JSON,
                c089_trace=core.DEFAULT_C089_TRACE,
                pr65_trace=core.DEFAULT_PR65_TRACE,
                specs=(_qpost_spec_to_core(core, base_spec.base_id, qpost),),
                positive_trace_only=True,
                expected_source_sha256=str(base["archive_sha256"]),
                expected_pr65_sha256=core.EXPECTED_PR65_SHA256,
                expected_pr65_head_sha=core.EXPECTED_PR65_HEAD_SHA,
            )
            rows = [row for row in matrix.get("candidate_summary", []) if isinstance(row, dict)]
            if len(rows) != 1:
                raise QPostInteractionError(
                    f"expected one candidate row for {base_spec.base_id}/{qpost.family_id}, got {len(rows)}"
                )
            row = rows[0]
            duplicate = _active_duplicate_status(
                source_sha256=str(base["archive_sha256"]),
                qpost=qpost,
                candidate_archive_sha256=str(row.get("archive_sha256") or ""),
                active_claims=active_claims,
            )
            screen = _screen_candidate(
                row=row,
                base=base,
                qpost=qpost,
                target_score=target_score,
                duplicate_status=duplicate,
            )
            screens.append(screen)
            if row.get("built"):
                candidate_dir = Path(str(row["archive"])).resolve().parent
                core_manifest = json.loads((candidate_dir / "manifest.json").read_text())
                _write_interaction_manifest(
                    candidate_dir=candidate_dir,
                    screen=screen,
                    selected_pair_rank_records=core_manifest.get("selected_pair_rank_records", []),
                    output_dir=output_dir,
                )

    best = _best_candidate(screens)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "remote_dispatch": {
            "dispatched": False,
            "requires_lane_claim_before_any_eval": True,
        },
        "target_score": target_score,
        "active_claims_path": str(active_claims_path.resolve()),
        "active_claim_count": len(active_claims),
        "action_bases": verified_bases,
        "qpost_specs": [
            {
                "family_id": spec.family_id,
                "include_streams": list(spec.include_streams),
                "top_pairs": spec.top_pairs,
                "risk_family": spec.risk_family,
            }
            for spec in qpost_specs
        ],
        "skipped_candidates": skipped,
        "candidate_screens": sorted(
            screens,
            key=lambda row: (
                not bool(row.get("dispatchable_later")),
                -float((row["trace_opportunity_estimates"])["trace_slack_vs_target"]),
                str(row.get("candidate_id")),
            ),
        ),
        "best_candidate": best,
    }
    _write_json(output_dir / "candidate_summary.json", summary)
    return summary


def _parse_action_base(raw: str) -> ActionBase:
    parts = raw.split(":", 4)
    if len(parts) != 5:
        raise QPostInteractionError(
            "--action-base must be BASE_ID:ARCHIVE:EXACT_SCORE:EXPECTED_SHA256:SCORE_SOURCE"
        )
    return ActionBase(parts[0], Path(parts[1]), float(parts[2]), parts[4], parts[3])


def _parse_qpost_spec(raw: str) -> QPostSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise QPostInteractionError("--qpost-spec must be FAMILY_ID:STREAM1,STREAM2:TOP_PAIRS")
    family_id = parts[0]
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]{2,80}", family_id):
        raise QPostInteractionError(f"invalid qpost family id: {family_id!r}")
    streams = tuple(value.strip() for value in parts[1].split(",") if value.strip())
    if not streams:
        raise QPostInteractionError(f"qpost spec {family_id!r} has no streams")
    top_pairs = int(parts[2])
    if top_pairs <= 0 or top_pairs > 600:
        raise QPostInteractionError(f"invalid top_pairs for {family_id}: {top_pairs}")
    return QPostSpec(family_id, streams, top_pairs, "custom_interaction")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--active-claims", type=Path, default=DEFAULT_ACTIVE_CLAIMS)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--action-base", action="append", default=[])
    parser.add_argument("--qpost-spec", action="append", default=[])
    parser.add_argument(
        "--allow-source-sha-mismatch",
        action="store_true",
        help="Planning/test override; default fails closed on action-base SHA drift.",
    )
    args = parser.parse_args(argv)
    action_bases = tuple(_parse_action_base(value) for value in args.action_base) or DEFAULT_ACTION_BASES
    qpost_specs = tuple(_parse_qpost_spec(value) for value in args.qpost_spec) or DEFAULT_QPOST_SPECS
    summary = build_interactions(
        output_dir=args.output_dir,
        active_claims_path=args.active_claims,
        action_bases=action_bases,
        qpost_specs=qpost_specs,
        target_score=args.target_score,
        allow_source_sha_mismatch=args.allow_source_sha_mismatch,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
