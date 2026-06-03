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
        "--snerv-official-source-audit",
        type=Path,
        help=(
            "Optional snerv_official_source_parity_audit.v1 JSON. When present, "
            "SNeRV rows carry official-source marker custody and remaining "
            "parity debt without granting promotion authority."
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
        candidate_feedback_sources=tuple(_load_feedback_sources(args.candidate_feedback_source)),
        decoder_weight_waterfill_sources=tuple(
            _load_decoder_weight_waterfill_sources(
                args.decoder_weight_waterfill_source,
                sidecar_root=args.output_json.parent / "decoder_weight_waterfill_sidecars",
            )
        ),
        snerv_official_source_audit=(
            None if args.snerv_official_source_audit is None else _load(args.snerv_official_source_audit)
        ),
        snerv_bounded_proof_only=bool(args.snerv_bounded_proof_only),
        snerv_bounded_proof_epochs=int(args.snerv_bounded_proof_epochs),
        experiment_queue_id=experiment_queue_id,
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
                "launchable_local_row_count": report["launchable_local_row_count"],
                "blocked_row_count": report["blocked_row_count"],
                "decoder_weight_waterfill_attached_row_count": report["decoder_weight_waterfill_attached_row_count"],
                "score_claim": report["score_claim"],
                "ready_for_exact_eval_dispatch": report["ready_for_exact_eval_dispatch"],
                "output_json": args.output_json.as_posix(),
                "output_queue": (None if args.output_queue is None else args.output_queue.as_posix()),
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
        raise TypeError(f"{path}: unsupported decoder waterfill source schema {schema!r}")
    return out


def _materialize_archive_ladder_waterfill_sidecars(
    payload: dict,
    source_path: Path,
    sidecar_root: Path,
) -> list[dict]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"{source_path}: archive-ladder waterfill rows must be a list")
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
        sidecar = (
            sidecar_root / f"{_safe_token(source_path.stem)}__{_safe_token(candidate_key)}"
            ".decoder_weight_waterfill.json"
        )
        _write_json_sidecar_if_identical_or_missing(sidecar, plan)
        out.append(
            _waterfill_payload_with_path(
                plan,
                plan_path=sidecar,
                source_path=source_path,
                extra={
                    "_modelsize_row_id": candidate_key,
                    "_archive_ladder_row_index": index,
                    "_archive_ladder_source_schema": payload.get("schema"),
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
