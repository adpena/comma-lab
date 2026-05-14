#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a fail-closed delta-epsilon-zeta PR106/HNeRV candidate plan.

This tool is a narrow bridge between the existing per-tensor Shannon target
table and a future byte-closed trained payload/archive. It does not train,
encode, score, or dispatch. It proves that the target table and baseline
renderer/payload input are loadable, selects the highest-pressure tensors, and
records the exact byte-change gates that must pass before a candidate can move
to exact-eval review.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import pathlib
import sys
from typing import Any
from zipfile import BadZipFile, ZipFile

from tac.codec_pipeline_deltaepszeta_callback import (
    CodecPipelineAwareTrainingCallback,
)
from tac.shannon_h2_loss import shannon_h2_loss

TARGET_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
TOOL_NAME = "experiments/build_deltaepszeta_pr106_candidate.py"


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: pathlib.Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist or is not a file: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _zip_member_is_unsafe(name: str) -> bool:
    pure = pathlib.PurePosixPath(name)
    parts = pure.parts
    return (
        name.startswith("/")
        or ".." in parts
        or any(part.startswith(".") for part in parts)
        or "__MACOSX" in parts
        or any(part == "" for part in parts)
    )


def _inspect_zip_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Return a safe, deterministic ZIP manifest for a candidate archive."""
    try:
        with ZipFile(path) as zf:
            members = []
            unsafe_members = []
            duplicate_names = []
            seen: set[str] = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if name in seen:
                    duplicate_names.append(name)
                seen.add(name)
                data = zf.read(name)
                unsafe = _zip_member_is_unsafe(name)
                if unsafe:
                    unsafe_members.append(name)
                members.append(
                    {
                        "name": name,
                        "bytes": len(data),
                        "compressed_bytes": int(info.compress_size),
                        "crc32": f"{info.CRC:08x}",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "unsafe_name": unsafe,
                    }
                )
    except BadZipFile:
        return {
            "is_zip": False,
            "safe_for_candidate_review": False,
            "members": [],
            "blockers": ["candidate_archive_is_not_a_zip_file"],
        }

    blockers = []
    if not members:
        blockers.append("candidate_archive_has_no_file_members")
    if unsafe_members:
        blockers.append("candidate_archive_has_unsafe_member_names")
    if duplicate_names:
        blockers.append("candidate_archive_has_duplicate_members")
    return {
        "is_zip": True,
        "safe_for_candidate_review": not blockers,
        "members": sorted(members, key=lambda m: m["name"]),
        "member_count": len(members),
        "unsafe_members": sorted(unsafe_members),
        "duplicate_members": sorted(duplicate_names),
        "blockers": blockers,
    }


def load_targets(path: pathlib.Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"targets JSON must be an object: {path}")
    if raw.get("score_claim") is not False:
        raise ValueError("targets JSON must be non-scoring: score_claim must be false")
    if raw.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported targets schema_version "
            f"{raw.get('schema_version')!r}; expected {TARGET_SCHEMA_VERSION}"
        )
    per_tensor = raw.get("per_tensor")
    if not isinstance(per_tensor, list) or not per_tensor:
        raise ValueError("targets JSON must contain a non-empty per_tensor list")
    summary = raw.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("targets JSON must contain a summary object")
    if float(summary.get("total_prize_bytes", 0.0)) <= 0.0:
        raise ValueError("targets JSON has no positive H0-H2 prize bytes")
    return raw


def select_top_tensors(
    targets: dict[str, Any],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k!r}")
    rows = []
    for row in targets["per_tensor"]:
        prize = float(row["prize_bytes"])
        headroom = float(row["headroom_bits"])
        weight = float(row["loss_weight_normalized"])
        if prize <= 0.0 or headroom <= 0.0 or weight <= 0.0:
            continue
        rows.append(
            {
                "idx": int(row["idx"]),
                "name": str(row["name"]),
                "n_symbols": int(row["n_symbols"]),
                "H0_bits": float(row["H0_bits"]),
                "H2_bits": float(row["H2_bits"]),
                "headroom_bits": headroom,
                "prize_bytes": prize,
                "prize_bytes_ceiled": math.ceil(prize),
                "loss_weight_normalized": weight,
                "in_pr103_ac_set": bool(row.get("in_pr103_ac_set", False)),
            }
        )
    rows.sort(key=lambda r: (-r["prize_bytes"], r["idx"], r["name"]))
    selected = rows[:top_k]
    if not selected:
        raise ValueError("no positive H2-pressure tensors were selected")
    return selected


def _pressure_plan(selected: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_weight = sum(float(r["loss_weight_normalized"]) for r in selected)
    aggregate_prize = sum(float(r["prize_bytes"]) for r in selected)
    return {
        "nonzero_h2_pressure": aggregate_weight > 0.0 and aggregate_prize > 0.0,
        "selected_tensor_count": len(selected),
        "aggregate_loss_weight": aggregate_weight,
        "aggregate_prize_bytes": aggregate_prize,
        "aggregate_prize_bytes_ceiled": math.ceil(aggregate_prize),
        "loss_expression": (
            "sum(row.loss_weight_normalized * "
            "shannon_h2_loss(state_dict[row.name], n_bits=8) "
            "for row in selected_top_tensors)"
        ),
        "shannon_h2_loss_api": (
            f"{shannon_h2_loss.__module__}.{shannon_h2_loss.__name__}"
        ),
        "training_byte_telemetry_api": (
            f"{CodecPipelineAwareTrainingCallback.__module__}."
            f"{CodecPipelineAwareTrainingCallback.__name__}"
        ),
        "notes": [
            "H2 pressure is a training/planning signal only.",
            "Final byte evidence must come from a distinct trained payload and archive.",
            "Final score evidence must come from exact CUDA auth eval on archive bytes.",
        ],
    }


def _byte_change_requirement(
    renderer_input: dict[str, Any],
    trained_payload_path: pathlib.Path | None,
    candidate_archive_path: pathlib.Path | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    blockers: list[str] = []
    trained_payload_record = None
    candidate_archive_record = None
    archive_manifest = None
    candidate_archive_contains_trained_payload_sha = None

    if trained_payload_path is None:
        blockers.append("missing_trained_payload")
    else:
        trained_payload_record = _file_record(
            trained_payload_path,
            role="trained_payload",
        )
        if trained_payload_record["sha256"] == renderer_input["sha256"]:
            blockers.append("trained_payload_matches_renderer_input")

    if candidate_archive_path is None:
        blockers.append("missing_candidate_archive")
    else:
        candidate_archive_record = _file_record(
            candidate_archive_path,
            role="candidate_archive",
        )
        if candidate_archive_record["sha256"] == renderer_input["sha256"]:
            blockers.append("candidate_archive_matches_renderer_input")
        archive_manifest = _inspect_zip_manifest(candidate_archive_path)
        blockers.extend(archive_manifest["blockers"])
        if trained_payload_record is not None and archive_manifest["is_zip"]:
            member_shas = {
                str(member["sha256"]) for member in archive_manifest["members"]
            }
            candidate_archive_contains_trained_payload_sha = (
                trained_payload_record["sha256"] in member_shas
            )
            if not candidate_archive_contains_trained_payload_sha:
                blockers.append("candidate_archive_does_not_contain_trained_payload_sha")

    requirement = {
        "required": True,
        "satisfied": not blockers,
        "baseline_renderer_input_sha256": renderer_input["sha256"],
        "trained_payload_sha256": (
            None if trained_payload_record is None else trained_payload_record["sha256"]
        ),
        "trained_payload_differs_from_renderer_input": (
            False
            if trained_payload_record is None
            else trained_payload_record["sha256"] != renderer_input["sha256"]
        ),
        "trained_payload_byte_delta_vs_renderer_input": (
            None
            if trained_payload_record is None
            else int(trained_payload_record["bytes"]) - int(renderer_input["bytes"])
        ),
        "candidate_archive_present": candidate_archive_record is not None,
        "candidate_archive_sha256": (
            None
            if candidate_archive_record is None
            else candidate_archive_record["sha256"]
        ),
        "candidate_archive_differs_from_renderer_input": (
            False
            if candidate_archive_record is None
            else candidate_archive_record["sha256"] != renderer_input["sha256"]
        ),
        "candidate_archive_contains_trained_payload_sha": (
            candidate_archive_contains_trained_payload_sha
        ),
        "blockers": blockers,
    }
    records = {
        "trained_payload": trained_payload_record,
        "candidate_archive": candidate_archive_record,
        "candidate_archive_zip_manifest": archive_manifest,
    }
    return requirement, records, blockers


def build_plan(
    *,
    targets_json: pathlib.Path,
    renderer_input: pathlib.Path,
    top_k: int = 8,
    trained_payload: pathlib.Path | None = None,
    candidate_archive: pathlib.Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    targets = load_targets(targets_json)
    selected = select_top_tensors(targets, top_k=top_k)
    renderer_record = _file_record(renderer_input, role="renderer_or_payload_input")
    targets_record = _file_record(targets_json, role="targets_json")
    pressure = _pressure_plan(selected)
    byte_requirement, byte_records, byte_blockers = _byte_change_requirement(
        renderer_record,
        trained_payload,
        candidate_archive,
    )
    readiness_blockers = list(byte_blockers)
    if not pressure["nonzero_h2_pressure"]:
        readiness_blockers.append("zero_h2_pressure")

    ready_for_byte_closed_candidate = not readiness_blockers
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_at_utc": created_at_utc or _utc_iso(),
        "score_claim": False,
        "evidence_grade": "[empirical]",
        "candidate": {
            "family": "delta_epsilon_zeta_pr106_hnerv",
            "purpose": (
                "planning bridge from PR106 Shannon targets to a future "
                "byte-closed trained payload/archive"
            ),
        },
        "targets": {
            "loaded": True,
            "path": str(targets_json),
            "source_shannon_analysis": targets.get("source_shannon_analysis"),
            "substrate": targets.get("substrate"),
            "n_tensors": int(targets["n_tensors"]),
            "summary": targets["summary"],
        },
        "selected_top_tensors": selected,
        "h2_pressure_plan": pressure,
        "byte_change_requirement": byte_requirement,
        "readiness": {
            "fail_closed": not ready_for_byte_closed_candidate,
            "ready_for_byte_closed_candidate": ready_for_byte_closed_candidate,
            "ready_for_exact_eval_dispatch": False,
            "ready_for_exact_eval_dispatch_reason": (
                "planning-only bridge; run contest compliance, dispatch claim, "
                "and exact CUDA auth eval after a real trained archive exists"
            ),
            "blockers": readiness_blockers,
        },
        "manifest": {
            "targets_json": targets_record,
            "renderer_or_payload_input": renderer_record,
            "trained_payload": byte_records["trained_payload"],
            "candidate_archive": byte_records["candidate_archive"],
            "candidate_archive_zip_manifest": byte_records[
                "candidate_archive_zip_manifest"
            ],
            "reused_apis": {
                "target_builder_output": "tools/build_deltaepszeta_training_targets.py schema_version=1",
                "shannon_h2_loss": (
                    f"{shannon_h2_loss.__module__}.{shannon_h2_loss.__name__}"
                ),
                "codec_pipeline_deltaepszeta_callback": (
                    f"{CodecPipelineAwareTrainingCallback.__module__}."
                    f"{CodecPipelineAwareTrainingCallback.__name__}"
                ),
            },
        },
        "score_fields": {
            "component_distances": None,
            "archive_score": None,
            "score_formula_applied": False,
            "score_claim_reason": "no scorer was run by this planning tool",
        },
    }


def write_plan(plan: dict[str, Any], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a fail-closed delta-epsilon-zeta PR106 candidate plan"
    )
    p.add_argument("--targets-json", required=True, help="targets.json from delta-epsilon-zeta target builder")
    p.add_argument("--renderer-input", required=True, help="baseline renderer/payload input to improve")
    p.add_argument("--trained-payload", default=None, help="optional distinct trained payload candidate")
    p.add_argument("--candidate-archive", default=None, help="optional byte-closed candidate archive.zip")
    p.add_argument("--top-k", type=int, default=8, help="number of positive-pressure tensors to select")
    p.add_argument(
        "--output-dir",
        default=None,
        help="output dir (default: experiments/results/lane_deltaepszeta_pr106_candidate_<UTC>/)",
    )
    p.add_argument(
        "--created-at-utc",
        default=None,
        help="fixed ISO-8601 UTC timestamp for byte-reproducible output",
    )
    args = p.parse_args(argv)

    if args.output_dir is None:
        out_dir = pathlib.Path(
            f"experiments/results/lane_deltaepszeta_pr106_candidate_{_utc_timestamp()}"
        )
    else:
        out_dir = pathlib.Path(args.output_dir)

    plan = build_plan(
        targets_json=pathlib.Path(args.targets_json),
        renderer_input=pathlib.Path(args.renderer_input),
        top_k=args.top_k,
        trained_payload=(
            None if args.trained_payload is None else pathlib.Path(args.trained_payload)
        ),
        candidate_archive=(
            None if args.candidate_archive is None else pathlib.Path(args.candidate_archive)
        ),
        created_at_utc=args.created_at_utc,
    )
    out_path = out_dir / "candidate_plan.json"
    write_plan(plan, out_path)

    readiness = plan["readiness"]
    pressure = plan["h2_pressure_plan"]
    print(f"candidate_plan: {out_path}")
    print(f"score_claim: {plan['score_claim']}")
    print(f"selected_tensors: {pressure['selected_tensor_count']}")
    print(f"aggregate_prize_bytes_ceiled: {pressure['aggregate_prize_bytes_ceiled']:,}")
    print(f"ready_for_byte_closed_candidate: {readiness['ready_for_byte_closed_candidate']}")
    print(f"ready_for_exact_eval_dispatch: {readiness['ready_for_exact_eval_dispatch']}")
    if readiness["blockers"]:
        print("blockers:")
        for blocker in readiness["blockers"]:
            print(f"  - {blocker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
