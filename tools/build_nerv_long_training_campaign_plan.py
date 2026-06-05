#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the false-authority HiNeRV/SNeRV long-training campaign plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.nerv_long_training_campaign_plan import (  # noqa: E402
    DEFAULT_BATCH_PAIRS,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER_KINDS,
    build_nerv_long_training_campaign_plan,
    render_nerv_long_training_campaign_plan_markdown,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_bytes,
    sha256_file,
    write_json_artifact,
    write_text_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hinerv-modelsize-budget", type=Path, required=True)
    parser.add_argument("--snerv-modelsize-budget", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-queue", type=Path)
    parser.add_argument("--output-snerv-lf-reroute-queue", type=Path)
    parser.add_argument(
        "--experiment-queue-id",
        help=(
            "Queue id for the emitted experiment_queue.v1. Defaults to a fresh "
            "id derived from --output-json so normal operator builds do not "
            "reuse stale SQLite state."
        ),
    )
    parser.add_argument("--expected-output-json-sha256")
    parser.add_argument("--expected-output-md-sha256")
    parser.add_argument("--expected-output-queue-sha256")
    parser.add_argument("--expected-output-snerv-lf-reroute-queue-sha256")
    parser.add_argument("--optimizer-kind", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-pairs", type=int, default=DEFAULT_BATCH_PAIRS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--joint-recon-weight-manifest",
        action="append",
        default=[],
        type=Path,
        help=(
            "Verified joint P18/P19 recon-pixel-weight manifest to pin in "
            "HiNeRV campaign rows. Repeatable for multiple pair counts."
        ),
    )
    parser.add_argument(
        "--candidate-feedback-source",
        action="append",
        default=[],
        type=Path,
        help=(
            "Runner report or nerv_candidate_feedback_row JSON to feed measured "
            "archive/proof/prefilter/replay evidence back into campaign rows. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--auto-candidate-feedback-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Search this artifact root for nerv_candidate_feedback_row.v1 JSON "
            "and feed newest rows into campaign rows. Repeatable."
        ),
    )
    parser.add_argument(
        "--auto-candidate-feedback-limit",
        type=int,
        default=8,
        help="Maximum discovered candidate feedback rows to consume.",
    )
    parser.add_argument(
        "--modelsize-byte-cap-feedback-json",
        action="append",
        default=[],
        type=Path,
        help=(
            "Measured checkpoint/archive export JSON consumed by "
            "run_compact_renderer_mlx_spine_runner.py for calibrated hard-byte "
            "cap modelsize selection. Repeatable."
        ),
    )
    parser.add_argument(
        "--auto-modelsize-byte-cap-feedback-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Search this artifact root for receiver-proof HiNeRV/SNeRV "
            "checkpoint export reports and feed the newest rows into calibrated "
            "hard-byte-cap modelsize selection. Repeatable."
        ),
    )
    parser.add_argument(
        "--auto-modelsize-byte-cap-feedback-limit",
        type=int,
        default=8,
        help="Maximum discovered checkpoint export reports to consume.",
    )
    parser.add_argument(
        "--decoder-weight-waterfill-source",
        action="append",
        default=[],
        type=Path,
        help=(
            "Direct nerv_decoder_weight_waterfill.v1 JSON or "
            "hinerv_archive_ladder_waterfill.v1 bundle. Nested archive-ladder "
            "plans are materialized as deterministic sidecar JSON files next "
            "to --output-json before queue rows attach them."
        ),
    )
    parser.add_argument(
        "--snerv-lf-payload-recode-source",
        action="append",
        default=[],
        type=Path,
        help=(
            "Receiver-proof snerv_lf_payload_archive_recode.v1 or admission "
            "JSON to feed the SNeRV LF recode admission and over-ceiling "
            "reroute queue. Repeatable."
        ),
    )
    parser.add_argument(
        "--snerv-lf-payload-byte-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "Measured LF payload byte report for the SNeRV over-ceiling "
            "reroute queue. Accepts snerv_checkpoint_archive_export.v1, "
            "snerv_lf_payload_codec_sweep.v1, or recode/admission JSON. "
            "Repeatable."
        ),
    )
    parser.add_argument(
        "--snerv-snar-header-grammar-profile",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_snar_header_grammar_profile.v1 JSON to attach exact "
            "post-recode SNAR header byte accounting to the over-ceiling "
            "reroute queue. Repeatable."
        ),
    )
    parser.add_argument(
        "--snerv-snar-header-minimization-report",
        action="append",
        default=[],
        type=Path,
        help=(
            "snerv_snar_header_minimization.v1 JSON to attach materialized "
            "SNAR1 header-prune packet/archive bytes to the over-ceiling "
            "reroute queue. Repeatable."
        ),
    )
    parser.add_argument(
        "--snerv-official-source-audit",
        type=Path,
        help=(
            "Optional snerv_official_source_parity_audit.v1 JSON. When present, "
            "SNeRV rows carry official-source marker custody and remaining "
            "parity debt without granting promotion authority."
        ),
    )
    parser.add_argument(
        "--pr95-baseline-identity",
        type=Path,
        help=(
            "Optional pr95_baseline_identity.v1 JSON. When present, campaign "
            "rows carry the selected PR95 control-arm archive identity and "
            "paired exact-eval work order without granting score authority."
        ),
    )
    parser.add_argument(
        "--snerv-scorer-tether-smoke-report",
        type=Path,
        help=(
            "Optional snerv_scorer_tether_smoke.v1 JSON. Failed reports block "
            "SNeRV queue launch; passing reports prove the short PR95 tether "
            "aliases and lambda activation before long training."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/Volumes/VertigoDataTier/pact/nerv_long_training_campaigns",
    )
    parser.add_argument("--max-candidates-per-family", type=int, default=3)
    parser.add_argument(
        "--snerv-bounded-proof-only",
        action="store_true",
        help=(
            "Keep SNeRV rows in the historical bounded proof mode. By default "
            "SNeRV is admitted as a real long-training campaign row."
        ),
    )
    parser.add_argument("--snerv-bounded-proof-epochs", type=int, default=3)
    args = parser.parse_args(argv)

    experiment_queue_id = args.experiment_queue_id or (
        f"nerv_long_training_campaign_{_safe_token(args.output_json.stem)}.v1"
    )
    planner_row_queue_artifact_path = args.output_queue or args.output_json
    modelsize_byte_cap_feedback_paths = _dedupe_paths(
        [
            *(path for path in args.modelsize_byte_cap_feedback_json),
            *_discover_modelsize_byte_cap_feedback_paths(
                args.auto_modelsize_byte_cap_feedback_root,
                limit=int(args.auto_modelsize_byte_cap_feedback_limit),
            ),
        ]
    )
    candidate_feedback_paths = _dedupe_paths(
        [
            *(path for path in args.candidate_feedback_source),
            *_discover_candidate_feedback_paths(
                args.auto_candidate_feedback_root,
                limit=int(args.auto_candidate_feedback_limit),
            ),
        ]
    )
    report = build_nerv_long_training_campaign_plan(
        hinerv_modelsize_budget=_load(args.hinerv_modelsize_budget),
        snerv_modelsize_budget=_load(args.snerv_modelsize_budget),
        optimizer_kinds=tuple(args.optimizer_kind or DEFAULT_OPTIMIZER_KINDS),
        epochs=args.epochs,
        batch_pairs=args.batch_pairs,
        learning_rate=args.learning_rate,
        output_root=args.output_root,
        max_candidates_per_family=args.max_candidates_per_family,
        joint_recon_weight_manifest_paths=tuple(args.joint_recon_weight_manifest),
        candidate_feedback_sources=tuple(_load_feedback_sources(candidate_feedback_paths)),
        modelsize_byte_cap_feedback_paths=tuple(
            path.as_posix() for path in modelsize_byte_cap_feedback_paths
        ),
        decoder_weight_waterfill_sources=tuple(
            _load_decoder_weight_waterfill_sources(
                args.decoder_weight_waterfill_source,
                sidecar_root=args.output_json.parent / "decoder_weight_waterfill_sidecars",
            )
        ),
        snerv_lf_payload_recode_sources=tuple(
            _load(path) for path in args.snerv_lf_payload_recode_source
        ),
        snerv_lf_payload_byte_report_sources=tuple(
            _load(path) for path in args.snerv_lf_payload_byte_report
        ),
        snerv_snar_header_grammar_profile_sources=tuple(
            _load(path) for path in args.snerv_snar_header_grammar_profile
        ),
        snerv_snar_header_minimization_report_sources=tuple(
            _load(path) for path in args.snerv_snar_header_minimization_report
        ),
        snerv_official_source_audit=(
            None if args.snerv_official_source_audit is None else _load(args.snerv_official_source_audit)
        ),
        pr95_baseline_identity=(
            None if args.pr95_baseline_identity is None else _load(args.pr95_baseline_identity)
        ),
        snerv_scorer_tether_smoke_report=(
            None
            if args.snerv_scorer_tether_smoke_report is None
            else _load(args.snerv_scorer_tether_smoke_report)
        ),
        snerv_bounded_proof_only=bool(args.snerv_bounded_proof_only),
        snerv_bounded_proof_epochs=int(args.snerv_bounded_proof_epochs),
        experiment_queue_id=experiment_queue_id,
        planner_row_queue_artifact_path=planner_row_queue_artifact_path,
    )
    write_json_artifact(
        args.output_json,
        report,
        allow_overwrite=args.expected_output_json_sha256 is not None,
        expected_existing_sha256=args.expected_output_json_sha256,
    )
    if args.output_queue:
        write_json_artifact(
            args.output_queue,
            report["experiment_queue"],
            allow_overwrite=args.expected_output_queue_sha256 is not None,
            expected_existing_sha256=args.expected_output_queue_sha256,
        )
    if args.output_snerv_lf_reroute_queue:
        write_json_artifact(
            args.output_snerv_lf_reroute_queue,
            report["snerv_lf_over_ceiling_reroute_queue"],
            allow_overwrite=args.expected_output_snerv_lf_reroute_queue_sha256 is not None,
            expected_existing_sha256=args.expected_output_snerv_lf_reroute_queue_sha256,
        )
    if args.output_md:
        write_text_artifact(
            args.output_md,
            render_nerv_long_training_campaign_plan_markdown(report),
            allow_overwrite=args.expected_output_md_sha256 is not None,
            expected_existing_sha256=args.expected_output_md_sha256,
        )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "campaign_row_count": report["campaign_row_count"],
                "candidate_feedback_source_count": report[
                    "candidate_feedback_source_count"
                ],
                "launchable_local_row_count": report["launchable_local_row_count"],
                "blocked_row_count": report["blocked_row_count"],
                "decoder_weight_waterfill_attached_row_count": report["decoder_weight_waterfill_attached_row_count"],
                "snerv_lf_over_ceiling_reroute_queue_row_count": report[
                    "snerv_lf_over_ceiling_reroute_queue_row_count"
                ],
                "snerv_snar_header_grammar_profile_source_count": report[
                    "snerv_snar_header_grammar_profile_source_count"
                ],
                "snerv_snar_header_minimization_report_source_count": report[
                    "snerv_snar_header_minimization_report_source_count"
                ],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
                "output_json": args.output_json.as_posix(),
                "output_queue": (None if args.output_queue is None else args.output_queue.as_posix()),
                "output_snerv_lf_reroute_queue": (
                    None
                    if args.output_snerv_lf_reroute_queue is None
                    else args.output_snerv_lf_reroute_queue.as_posix()
                ),
                "experiment_queue_id": report["experiment_queue_id"],
            },
            sort_keys=True,
        )
    )
    return 0


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected JSON object")
    return payload


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve(strict=False)
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _discover_modelsize_byte_cap_feedback_paths(
    roots: list[Path],
    *,
    limit: int,
) -> list[Path]:
    if limit <= 0:
        return []
    candidates: list[tuple[float, Path]] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            continue
        for name in (
            "export_report.json",
            "hinerv_checkpoint_archive_export.json",
            "snerv_checkpoint_archive_export.json",
            "snerv_binary_profile.json",
        ):
            for path in root.rglob(name):
                if _is_receiver_proof_modelsize_byte_cap_export(path):
                    candidates.append((path.stat().st_mtime, path))
    candidates.sort(key=lambda item: (item[0], item[1].as_posix()), reverse=True)
    return _dedupe_paths([path for _mtime, path in candidates])[:limit]


def _discover_candidate_feedback_paths(
    roots: list[Path],
    *,
    limit: int,
) -> list[Path]:
    if limit <= 0:
        return []
    names = (
        "snerv_upstream_eval_candidate_feedback_row.json",
        "nerv_candidate_training_telemetry_feedback_row.json",
        "nerv_candidate_feedback_row.json",
        "candidate_feedback_row.json",
    )
    candidates: list[tuple[float, Path]] = []
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            continue
        for pattern in (*names, "*candidate_feedback*row*.json"):
            for path in root.rglob(pattern):
                key = path.resolve(strict=False).as_posix()
                if key in seen:
                    continue
                seen.add(key)
                if _is_candidate_feedback_source(path):
                    candidates.append((path.stat().st_mtime, path))
    candidates.sort(key=lambda item: (item[0], item[1].as_posix()), reverse=True)
    return _dedupe_paths([path for _mtime, path in candidates])[:limit]


def _is_candidate_feedback_source(path: Path) -> bool:
    try:
        payload = _load(path)
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    if payload.get("schema") == "nerv_candidate_feedback_row.v1":
        return True
    if payload.get("schema") == "nerv_candidate_byte_feedback_ledger.v1":
        row = payload.get("row")
        return (
            isinstance(row, dict)
            and row.get("schema") == "nerv_candidate_feedback_row.v1"
        )
    if payload.get("schema") == "nerv_queue_training_feedback_refresh.v1":
        rows = payload.get("rows")
        return isinstance(rows, list) and any(
            isinstance(item, dict)
            and isinstance(item.get("row"), dict)
            and item["row"].get("schema") == "nerv_candidate_feedback_row.v1"
            for item in rows
        )
    return False


def _is_receiver_proof_modelsize_byte_cap_export(path: Path) -> bool:
    try:
        payload = _load(path)
    except (OSError, TypeError, json.JSONDecodeError):
        return False
    if payload.get("schema") == "snerv_binary_profile.v1":
        return _is_receiver_proof_snerv_binary_profile(path, payload)
    if payload.get("schema") not in {
        "hinerv_checkpoint_archive_export.v1",
        "snerv_checkpoint_archive_export.v1",
    }:
        return False
    if payload.get("family") not in {"hi_nerv", "snerv"}:
        return False
    candidate = payload.get("modelsize_candidate")
    if not isinstance(candidate, dict):
        return False
    runtime_ready = any(
        bool(payload.get(key))
        for key in (
            "receiver_proof_ready",
            "receiver_proof_passed",
            "runtime_consumption_proof_ready",
        )
    )
    if not runtime_ready:
        return False
    if payload.get("receiver_contract_satisfied") is False:
        return False
    try:
        return int(payload.get("archive_bytes") or 0) > 0
    except (TypeError, ValueError):
        return False


def _is_receiver_proof_snerv_binary_profile(
    path: Path,
    payload: dict,
) -> bool:
    if _positive_int(payload.get("charged_archive_bytes")) is None:
        return False
    startup = _startup_payload_for_artifact(path, payload)
    candidate = startup.get("modelsize_candidate")
    if not isinstance(candidate, dict):
        return False
    if not _snerv_binary_profile_scope_matches_candidate(payload, candidate):
        return False
    proof = _receiver_proof_payload_for_artifact(path, payload)
    if not proof:
        return False
    runtime_ready = bool(
        proof.get("runtime_consumption_proof_ready")
        or proof.get("runtime_consumption_proof_passed")
        or proof.get("receiver_proof_ready")
        or proof.get("receiver_proof_passed")
    )
    if not (runtime_ready and proof.get("receiver_contract_satisfied") is True):
        return False
    proof_archive_sha = str(proof.get("archive_sha256") or "").strip().lower()
    profile_archive_sha = str(payload.get("input_sha256") or "").strip().lower()
    if proof_archive_sha and profile_archive_sha and proof_archive_sha != profile_archive_sha:
        return False
    proof_archive_bytes = _positive_int(proof.get("archive_bytes"))
    profile_archive_bytes = _positive_int(payload.get("charged_archive_bytes"))
    return not (
        proof_archive_bytes is not None
        and profile_archive_bytes is not None
        and int(proof_archive_bytes) != int(profile_archive_bytes)
    )


def _snerv_binary_profile_scope_matches_candidate(
    payload: dict,
    candidate: dict,
) -> bool:
    metadata = payload.get("snar1_metadata")
    measured_pairs = _positive_int(payload.get("measured_num_pairs"))
    if measured_pairs is None and isinstance(metadata, dict):
        measured_pairs = _positive_int(metadata.get("n_pairs"))
    candidate_pairs = _positive_int(candidate.get("num_pairs"))
    return not (
        measured_pairs is not None
        and candidate_pairs is not None
        and int(measured_pairs) != int(candidate_pairs)
    )


def _startup_payload_for_artifact(path: Path, payload: dict) -> dict:
    for artifact_path in _artifact_reference_paths(path, payload):
        for parent in _self_and_parents(artifact_path):
            startup = parent / "compact_renderer_mlx_spine_runner_startup.json"
            if not startup.is_file():
                continue
            try:
                loaded = _load(startup)
            except (OSError, TypeError, json.JSONDecodeError):
                continue
            return loaded
    return {}


def _receiver_proof_payload_for_artifact(path: Path, payload: dict) -> dict:
    for artifact_path in _artifact_reference_paths(path, payload):
        for parent in _self_and_parents(artifact_path):
            for name in (
                "snerv_inverse_steg_receiver_proof.json",
                "hi_nerv_mlx_receiver_proof.json",
            ):
                for proof in (parent / name, parent / "receiver_proof" / name):
                    if not proof.is_file():
                        continue
                    try:
                        loaded = _load(proof)
                    except (OSError, TypeError, json.JSONDecodeError):
                        continue
                    return loaded
    return {}


def _artifact_reference_paths(path: Path, payload: dict) -> list[Path]:
    out = [Path(path).expanduser().resolve(strict=False)]
    for key in (
        "input_path",
        "archive_path",
        "source_archive_path",
        "candidate_archive_path",
        "proof_path",
        "receiver_proof_path",
        "report_path",
    ):
        value = payload.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser().resolve(strict=False)
        if candidate not in out:
            out.append(candidate)
    return out


def _self_and_parents(path: Path) -> list[Path]:
    base = path if path.is_dir() else path.parent
    return [base, *list(base.parents)]


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_feedback_sources(paths: list[Path]) -> list[dict]:
    out: list[dict] = []
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    payload = json.loads(text)
                    if not isinstance(payload, dict):
                        raise TypeError(f"{path}:{line_no}: expected JSON object")
                    out.append(_feedback_payload_with_path(payload, path))
            continue
        payload = _load(path)
        row = payload.get("row")
        if payload.get("schema") == "nerv_candidate_byte_feedback_ledger.v1":
            if not isinstance(row, dict):
                raise TypeError(f"{path}: ledger wrapper missing feedback row")
            out.append(_feedback_payload_with_path(row, path))
        elif payload.get("schema") == "nerv_queue_training_feedback_refresh.v1":
            rows = payload.get("rows")
            if not isinstance(rows, list):
                raise TypeError(f"{path}: refresh wrapper missing rows list")
            for index, item in enumerate(rows):
                if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                    raise TypeError(f"{path}: rows[{index}] missing feedback row")
                out.append(_feedback_payload_with_path(item["row"], path))
        else:
            out.append(_feedback_payload_with_path(payload, path))
    return out


def _feedback_payload_with_path(payload: dict, path: Path) -> dict:
    out = dict(payload)
    out.setdefault("_candidate_feedback_source_path", path.as_posix())
    return out


def _load_decoder_weight_waterfill_sources(
    paths: list[Path],
    *,
    sidecar_root: Path,
) -> list[dict]:
    out: list[dict] = []
    for path in paths:
        payload = _load(path)
        schema = payload.get("schema")
        if schema == "nerv_decoder_weight_waterfill.v1":
            out.append(_waterfill_payload_with_path(payload, plan_path=path, source_path=path))
            continue
        if schema == "hinerv_archive_ladder_waterfill.v1":
            out.extend(_materialize_archive_ladder_waterfill_sidecars(payload, path, sidecar_root))
            continue
        if schema == "hinerv_archive_size_ladder.v1":
            out.extend(_load_archive_size_ladder_waterfill_sidecars(payload, path))
            continue
        raise TypeError(f"{path}: unsupported decoder waterfill source schema {schema!r}")
    return out


def _load_archive_size_ladder_waterfill_sidecars(
    payload: dict,
    source_path: Path,
) -> list[dict]:
    rows = payload.get("archive_rows")
    if not isinstance(rows, list):
        raise TypeError(f"{source_path}: archive size ladder rows must be a list")
    out: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"{source_path}: archive_rows[{index}] must be an object")
        plan_path_raw = row.get("decoder_weight_waterfill_plan_path")
        if not plan_path_raw:
            continue
        plan_path = Path(str(plan_path_raw)).expanduser().resolve(strict=False)
        plan = _load(plan_path)
        out.append(
            _waterfill_payload_with_path(
                plan,
                plan_path=plan_path,
                source_path=source_path,
                extra={
                    "_modelsize_row_id": row.get("row_id"),
                    "_modelsize_candidate": row.get("modelsize_candidate"),
                    "_archive_size_ladder_row_index": index,
                    "_archive_size_ladder_source_schema": payload.get("schema"),
                    "_archive_size_ladder_full_video_coverage": row.get(
                        "full_video_coverage"
                    ),
                    "_archive_size_ladder_archive_sha256": row.get("archive_sha256"),
                    "_archive_size_ladder_receiver_proof_path": row.get(
                        "receiver_proof_path"
                    ),
                    "_archive_size_ladder_receiver_proof_sha256": row.get(
                        "receiver_proof_sha256"
                    ),
                    "_archive_size_ladder_runtime_consumption_proof_ready": row.get(
                        "runtime_consumption_proof_ready"
                    ),
                },
            )
        )
    return out


def _materialize_archive_ladder_waterfill_sidecars(
    payload: dict,
    source_path: Path,
    sidecar_root: Path,
) -> list[dict]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{source_path}: archive-ladder waterfill rows must be a list")
    archive_ladder_report_path_raw = (
        payload.get("archive_ladder_report_path")
        or payload.get("archive_ladder_path")
        or payload.get("source_archive_ladder_path")
    )
    archive_ladder_report_path = (
        Path(str(archive_ladder_report_path_raw)).expanduser().resolve(strict=False)
        if archive_ladder_report_path_raw
        else source_path
    )
    archive_ladder_rows_by_id: dict[str, dict] = {}
    if archive_ladder_report_path.is_file():
        archive_ladder_payload = _load(archive_ladder_report_path)
        for archive_row in archive_ladder_payload.get("archive_rows") or ():
            if isinstance(archive_row, dict) and archive_row.get("row_id") is not None:
                archive_ladder_rows_by_id[str(archive_row["row_id"])] = archive_row
    out: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        plan = row.get("waterfill_plan")
        if not isinstance(plan, dict):
            continue
        if plan.get("schema") != "nerv_decoder_weight_waterfill.v1":
            raise TypeError(
                f"{source_path}: row {index} has unsupported nested waterfill schema {plan.get('schema')!r}"
            )
        candidate_key = str(row.get("row_id") or plan.get("candidate_id") or f"waterfill_row_{index:04d}")
        archive_ladder_row = archive_ladder_rows_by_id.get(candidate_key, {})
        row_blockers = _merged_blockers(
            plan.get("blockers"),
            row.get("blockers"),
            row.get("saliency_replay_blockers"),
        )
        sidecar = (
            sidecar_root / f"{_safe_token(source_path.stem)}__{_safe_token(candidate_key)}"
            ".decoder_weight_waterfill.json"
        )
        _write_json_sidecar_if_identical_or_missing(sidecar, plan)
        out.append(
            _waterfill_payload_with_path(
                plan,
                plan_path=sidecar,
                source_path=archive_ladder_report_path,
                extra={
                    "_modelsize_row_id": candidate_key,
                    "_archive_ladder_row_index": index,
                    "_archive_ladder_source_schema": payload.get("schema"),
                    "_archive_ladder_full_video_coverage": (
                        row.get("full_video_coverage")
                        if row.get("full_video_coverage") is not None
                        else payload.get("full_video_coverage")
                    ),
                    "_archive_ladder_num_pairs": payload.get("num_pairs"),
                    "_archive_ladder_waterfill_report_path": source_path.as_posix(),
                    "_archive_ladder_waterfill_row_blockers": row_blockers,
                    "_archive_size_ladder_source_schema": payload.get("source_schema"),
                    "_modelsize_candidate": archive_ladder_row.get("modelsize_candidate"),
                    "_archive_size_ladder_archive_sha256": row.get(
                        "archive_sha256", archive_ladder_row.get("archive_sha256")
                    ),
                    "_archive_size_ladder_receiver_proof_path": row.get(
                        "receiver_proof_path",
                        archive_ladder_row.get("receiver_proof_path"),
                    ),
                    "_archive_size_ladder_receiver_proof_sha256": row.get(
                        "receiver_proof_sha256",
                        archive_ladder_row.get("receiver_proof_sha256"),
                    ),
                    "_archive_size_ladder_runtime_consumption_proof_ready": row.get(
                        "runtime_consumption_proof_ready",
                        archive_ladder_row.get("runtime_consumption_proof_ready"),
                    ),
                    "blockers": row_blockers,
                },
            )
        )
    return out


def _waterfill_payload_with_path(
    payload: dict,
    *,
    plan_path: Path,
    source_path: Path,
    extra: dict | None = None,
) -> dict:
    resolved_plan = plan_path.expanduser().resolve(strict=False)
    resolved_source = source_path.expanduser().resolve(strict=False)
    out = dict(payload)
    out.setdefault("_decoder_weight_waterfill_plan_path", resolved_plan.as_posix())
    out.setdefault("_decoder_weight_waterfill_source_path", resolved_source.as_posix())
    if resolved_plan.is_file():
        out.setdefault("_decoder_weight_waterfill_plan_sha256", sha256_file(resolved_plan))
    if extra:
        out.update(extra)
    return out


def _write_json_sidecar_if_identical_or_missing(path: Path, payload: dict) -> None:
    text = json_text(payload)
    expected_sha = sha256_bytes(text.encode("utf-8"))
    if path.exists():
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ArtifactWriteError(
                f"{path}: refusing to overwrite non-identical decoder waterfill sidecar "
                f"expected={expected_sha} actual={actual_sha}"
            )
        return
    write_json_artifact(path, payload)


def _safe_token(value: str) -> str:
    text = str(value).strip().lower()
    chars = [ch if ch.isalnum() else "_" for ch in text]
    token = "_".join("".join(chars).split("_"))
    return token[:160] or "waterfill"


def _merged_blockers(*groups: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for item in group:
            text = str(item)
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
