#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the frozen batch16 scorer on one exact G52 V10 raw bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.witness_dsl.taskspace_lossy_selected_plane_codec_v1 import (  # noqa: E402
    LossySelectedPlaneCodecError,
    canonical_json,
    sha256_file,
    write_once_or_equal,
)
from tools import score_coupled_witness_raw_debt as debt  # noqa: E402

CONFIG_SCHEMA = "taskspace_lossy_selected_plane_advisory_score_config.v2"
RECEIPT_SCHEMA = "taskspace_lossy_selected_plane_advisory_score_receipt.v2"
ORIGINAL_UNCOMPRESSED_SIZE_BYTES = 37_545_489
_FIELDS = frozenset(
    {
        "schema",
        "research_only",
        "candidate_lineage_allowed",
        "score_claim",
        "raw_path",
        "raw_sha256",
        "bridge_receipt_path",
        "bridge_receipt_sha256",
        "diagnostic_bundle_path",
        "diagnostic_bundle_sha256",
        "diagnostic_bundle_bytes",
        "target_raw",
        "target_receipt",
        "cache",
        "upstream",
        "output",
        "state",
        "stage_dir",
        "launch_manifest",
        "composition_receipt",
        "pair_count",
        "stage_pairs",
        "cpu_threads",
    }
)


def _receipt_hash(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _load_config(path: Path) -> dict:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise LossySelectedPlaneCodecError(f"cannot load advisory score config: {path}") from exc
    if not isinstance(value, dict) or frozenset(value) != _FIELDS:
        raise LossySelectedPlaneCodecError("advisory score config has missing or unknown fields")
    if (
        value["schema"] != CONFIG_SCHEMA
        or value["research_only"] is not True
        or value["candidate_lineage_allowed"] is not False
        or value["score_claim"] is not False
        or value["pair_count"] != 600
        or value["stage_pairs"] != 16
        or value["cpu_threads"] < 1
        or value["diagnostic_bundle_bytes"] < 1
    ):
        raise LossySelectedPlaneCodecError("advisory score type/authority contract drift")
    for path_field, sha_field in (
        ("raw_path", "raw_sha256"),
        ("bridge_receipt_path", "bridge_receipt_sha256"),
        ("diagnostic_bundle_path", "diagnostic_bundle_sha256"),
    ):
        if sha256_file(value[path_field]) != value[sha_field]:
            raise LossySelectedPlaneCodecError(f"{path_field} custody drift")
    if Path(value["diagnostic_bundle_path"]).stat().st_size != value["diagnostic_bundle_bytes"]:
        raise LossySelectedPlaneCodecError("diagnostic bundle byte count drift")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_governed_admission("taskspace_lossy_codec_advisory_score_n600")
    try:
        config = _load_config(args.config)
        debt_args = argparse.Namespace(
            raw=Path(config["raw_path"]),
            target_raw=Path(config["target_raw"]),
            target_receipt=Path(config["target_receipt"]),
            output=Path(config["output"]),
            state=Path(config["state"]),
            stage_dir=Path(config["stage_dir"]),
            launch_manifest=Path(config["launch_manifest"]),
            cache=Path(config["cache"]),
            upstream=Path(config["upstream"]),
            contest_reference=None,
            contest_archive=None,
            pair_count=config["pair_count"],
            stage_pairs=config["stage_pairs"],
            cpu_threads=config["cpu_threads"],
            resume=True,
            allow_noncanonical_cache=False,
        )
        scorer_receipt = debt.run(debt_args)
        d_seg = float(scorer_receipt["aggregate"]["mean_d_seg"])
        d_pose = float(scorer_receipt["aggregate"]["mean_d_pose"])
        distortion_term = 100.0 * d_seg + math.sqrt(10.0 * d_pose)
        rate_term = 25.0 * config["diagnostic_bundle_bytes"] / ORIGINAL_UNCOMPRESSED_SIZE_BYTES
        advisory_score = distortion_term + rate_term
        pointer = debt.scorer._effective_pointer_target()
        pointer_target = float(pointer["score"])
        composition = {
            "schema": RECEIPT_SCHEMA,
            "research_only": True,
            "candidate_lineage_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "scorer_receipt_path": str(Path(config["output"]).resolve()),
            "scorer_receipt_sha256": sha256_file(config["output"]),
            "scorer_internal_receipt_sha256": scorer_receipt["receipt_sha256"],
            "advisory_axis": scorer_receipt["axis"],
            "raw_path": config["raw_path"],
            "raw_sha256": config["raw_sha256"],
            "bridge_receipt_path": config["bridge_receipt_path"],
            "bridge_receipt_sha256": config["bridge_receipt_sha256"],
            "diagnostic_bundle_path": config["diagnostic_bundle_path"],
            "diagnostic_bundle_sha256": config["diagnostic_bundle_sha256"],
            "diagnostic_bundle_bytes": config["diagnostic_bundle_bytes"],
            "pair_count": config["pair_count"],
            "scorer_batch_pairs": config["stage_pairs"],
            "mean_d_seg": d_seg,
            "mean_d_pose": d_pose,
            "seg_term": 100.0 * d_seg,
            "pose_term": math.sqrt(10.0 * d_pose),
            "distortion_term": distortion_term,
            "rate_term": rate_term,
            "advisory_score": advisory_score,
            "dynamic_frontier_pointer": pointer,
            "pointer_target": pointer_target,
            "margin_to_pointer": pointer_target - advisory_score,
            "margin_to_sub015": 0.15 - advisory_score,
            "interpretation": "historical-C1 advisory codec measurement; not upstream evaluate authority",
        }
        composition["receipt_sha256"] = _receipt_hash(composition)
        write_once_or_equal(Path(config["composition_receipt"]), canonical_json(composition))
    except (OSError, LossySelectedPlaneCodecError, debt.RawDebtError, debt.scorer.PredictorFloorError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "closed",
                "d_seg": d_seg,
                "d_pose": d_pose,
                "distortion_term": distortion_term,
                "rate_term": rate_term,
                "advisory_score": advisory_score,
                "margin_to_pointer": pointer_target - advisory_score,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
