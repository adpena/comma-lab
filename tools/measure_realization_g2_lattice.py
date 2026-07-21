#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Audit the real seed-compose ladder for a valid G2 RGB-lattice handoff.

The factor-2 integer solver consumes an RGB scorer plane.  The merged
seed-compose campaign preserved class-ID fields and cache-replay cell/tube
checks only.  This tool walks the real n16/n64/n600 stages, records that input
domain distinction per pair, and emits resumable prefix checkpoints plus one
fail-closed advisory receipt.  It never substitutes a palette or source frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.predict_project_schema import parse_constraint_seed  # noqa: E402

SCHEMA: Final = "realization_g2_lattice_receipt.v1"
PAIR_STAGE_SCHEMA: Final = "predict_project_pair_stage.v0"
HARD_ORACLE_SCHEMA: Final = "predict_project_hard_oracle_pair.v0"
PREFIXES: Final = (16, 64, 600)
RGB_REALIZATION_FIELDS: Final = frozenset(
    {
        "projected_rgb_sha256",
        "camera_uint8_sha256",
        "factor2_verification",
        "projection_custody",
    }
)
POINTER: Final = "0.1910828242 [contest-CPU] UNMOVED"
AXIS: Final = "[macOS-CPU advisory]"


class RealizationAuditError(ValueError):
    """Missing, mixed, or falsely promoted G2 audit evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealizationAuditError(f"cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise RealizationAuditError(f"evidence must be one JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RealizationAuditError(f"{label} must be an exact nonnegative integer")
    return value


def audit_prefix(
    *,
    prefix: int,
    stage_rows: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify one real prefix without inventing an RGB projection."""

    if prefix not in PREFIXES or len(stage_rows) != prefix:
        raise RealizationAuditError("prefix rows do not match the canonical n16/n64/n600 ladder")
    if [row.get("pair_index") for row in stage_rows] != list(range(prefix)):
        raise RealizationAuditError("pair stages are not contiguous from zero")
    label_only_pairs = 0
    for index, row in enumerate(stage_rows):
        if row.get("schema") != PAIR_STAGE_SCHEMA:
            raise RealizationAuditError(f"pair {index} stage schema mismatch")
        hard = row.get("hard_oracle")
        if not isinstance(hard, Mapping) or hard.get("schema") != HARD_ORACLE_SCHEMA:
            raise RealizationAuditError(f"pair {index} lacks the real seed-compose hard row")
        if hard.get("cell_exact") is not True or hard.get("pose_within_tube") is not True:
            raise RealizationAuditError(f"pair {index} lost the settled plane-level cell/tube invariant")
        if hard.get("uint8_factor2_exact") is not False:
            raise RealizationAuditError(f"pair {index} no longer matches the seed-compose G2 blocker")
        if RGB_REALIZATION_FIELDS.intersection(hard) or RGB_REALIZATION_FIELDS.intersection(row):
            raise RealizationAuditError(f"pair {index} unexpectedly carries unaudited RGB realization fields")
        label_only_pairs += 1

    selected = []
    for row in constraints:
        time_value = _exact_nonnegative_int(row.get("time"), "constraint time")
        if time_value < prefix:
            selected.append(row)
    by_class = Counter(int(row["cell_id"]) for row in selected)
    by_stratum = Counter(str(row["stratum"]) for row in selected)

    def blocked_rows(counter: Counter[Any], name: str) -> list[dict[str, Any]]:
        return [
            {
                name: key,
                "declared_writes": count,
                "surviving_writes": 0,
                "dying_writes": 0,
                "not_attempted_missing_rgb_projection": count,
                "exact_fraction": None,
            }
            for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
        ]

    hard_rows = [row["hard_oracle"] for row in stage_rows]
    return {
        "schema": "realization_g2_prefix_audit.v1",
        "n": prefix,
        "pair_count": prefix,
        "label_only_pair_count": label_only_pairs,
        "rgb_projection_pair_count": 0,
        "lattice_attempted_pair_count": 0,
        "uint8_factor2_exact_pair_count": 0,
        "uint8_factor2_exact_fraction": None,
        "declared_constraint_count": len(selected),
        "by_class": blocked_rows(by_class, "class_id"),
        "by_stratum": blocked_rows(by_stratum, "stratum"),
        "plane_level_cache_replay": {
            "cell_exact_pairs": sum(row["cell_exact"] is True for row in hard_rows),
            "pose_within_tube_pairs": sum(row["pose_within_tube"] is True for row in hard_rows),
            "mean_d_seg_description": sum(float(row["d_seg"]) for row in hard_rows) / prefix,
            "mean_d_pose_tube_debt": sum(float(row["d_pose"]) for row in hard_rows) / prefix,
            "authority": "MEASURED_EXISTING_LABEL_AND_TUBE_CACHE_REPLAY_NOT_REALIZED_RGB",
        },
        "status": "BLOCKED_INPUT_DOMAIN_LABEL_FIELD_IS_NOT_RGB_PLANE",
        "verdict_scope": "seed_compose_b2 projected class IDs -> composed RGB lattice handoff",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def run_audit(
    *,
    seed_path: Path,
    stage_root: Path,
    m2_receipt_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    seed_bytes = seed_path.read_bytes()
    seed = parse_constraint_seed(seed_bytes)
    constraints = seed["constraint_seeds"]
    stage_paths = sorted(stage_root.glob("pair_*.json"))
    if len(stage_paths) != 600:
        raise RealizationAuditError(f"expected 600 preserved pair stages, found {len(stage_paths)}")
    stage_rows = [_load_json(path) for path in stage_paths]
    m2 = _load_json(m2_receipt_path)
    if m2.get("schema") != "m2_live_target_selection_receipt.v1":
        raise RealizationAuditError("M2 existence comparator schema mismatch")

    prefix_rows = []
    checkpoints = output_root / "checkpoints"
    for prefix in PREFIXES:
        row = audit_prefix(prefix=prefix, stage_rows=stage_rows[:prefix], constraints=constraints)
        checkpoint_path = checkpoints / f"prefix_n{prefix}.json"
        _atomic_json(checkpoint_path, row)
        row["checkpoint_path"] = str(checkpoint_path)
        row["checkpoint_sha256"] = _sha256(checkpoint_path)
        prefix_rows.append(row)

    implementation_paths = (
        REPO / "src/tac/optimization/predict_project_receiver.py",
        REPO / "tools/measure_realization_g2_lattice.py",
        REPO / "src/tac/tests/test_predict_project_receiver.py",
    )
    implementation = {str(path.relative_to(REPO)): _sha256(path) for path in implementation_paths}
    receipt = {
        "schema": SCHEMA,
        "lane_id": "lane_realization_g2_lattice_578_20260721",
        "task_id": "578",
        "verdict": "COMPOSED_RGB_LATTICE_BUILT_SEED_TO_RGB_PROJECTION_PREMISE_FALSIFIED",
        "verdict_scope": (
            "the current seed_compose_b2 class-ID projection cannot enter the RGB factor-2 lattice; "
            "this is a formulation handoff gap, not a realization-family negative"
        ),
        "D1_prefix_ladder": prefix_rows,
        "D1_implementation": {
            "status": "BUILT_STRICT_RGB_INPUT_CONTRACT",
            "callable": "tac.optimization.predict_project_receiver.realize_projected_rgb_plane_camera_uint8",
            "structural_fixture_factor2_exact": True,
            "real_seed_rgb_input_status": "ABSENT",
            "real_n600_uint8_factor2_exact": None,
            "reason": "the real seed and preserved stages contain 2D uint8 class IDs, no HxWx3 uint8 projected RGB plane",
        },
        "D2_cost": {
            "added_decode_seconds_per_pair": None,
            "additional_seed_bytes": None,
            "zero_byte_target_met": False,
            "status": "UNMEASURED_NO_COMPOSED_RGB_FRAMES",
            "M2_existence_comparator": {
                "receipt_path": str(m2_receipt_path),
                "receipt_sha256": _sha256(m2_receipt_path),
                "archive_bytes": m2["candidate"]["archive_bytes"],
                "d_seg": m2["candidate"]["d_seg"],
                "d_pose": m2["candidate"]["d_pose"],
                "interpretation": "exact realization exists when source RGB target values are counted; this does not supply the missing zero-byte seed-to-RGB map",
            },
        },
        "D3_pose": {
            "realized_frame_d_pose": None,
            "plane_level_tube_debt_d_pose": prefix_rows[-1]["plane_level_cache_replay"]["mean_d_pose_tube_debt"],
            "status": "BLOCKED_NO_COMPOSED_REALIZED_FRAMES",
            "transfer_forbidden": True,
        },
        "D4_equation": {
            "registered": False,
            "blocker": "D1_REAL_N600_COMPOSED_RGB_LATTICE_ANCHOR_ABSENT",
        },
        "reuse_manifest": {
            "seed_schema_and_parser": "tac.optimization.predict_project_schema",
            "receiver_project_stage_extended": "tac.optimization.predict_project_receiver",
            "uint8_lattice": "tac.optimization.uint8_lattice_feasibility",
            "joint_interval_solver": "tac.optimization.joint_seg_pose_rate",
            "full_kernel": "tac.optimization.resize_full_kernel",
            "seed_compose_measurement": str(stage_root.parent / "receipt.json"),
            "M2_realization_existence_anchor": str(m2_receipt_path),
            "new_with_justification": (
                "one bounded audit CLI is required because the existing measurement runner cannot distinguish "
                "a projected class-ID field from an RGB lattice input"
            ),
        },
        "input_custody": {
            "seed_path": str(seed_path),
            "seed_bytes": len(seed_bytes),
            "seed_sha256": hashlib.sha256(seed_bytes).hexdigest(),
            "stage_root": str(stage_root),
            "stage_count": len(stage_paths),
            "stage_first_sha256": _sha256(stage_paths[0]),
            "stage_last_sha256": _sha256(stage_paths[-1]),
        },
        "implementation_sources": implementation,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "storage_free_bytes_at_start": usage.free,
            "mmap_or_chunk_policy": "600 preserved JSON stages read in bounded pair order; no camera tensor materialized",
            "automatic_cleanup": "atomic temporary JSON files removed after replace; no rebuildable bulk produced",
        },
        "authority": {
            "axis": AXIS,
            "pointer": POINTER,
            "pointer_moved": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
        },
    }
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/seeds/seed_compose_b2_loose.ppcs"),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721/hard_oracle_n600/stages"),
    )
    parser.add_argument(
        "--m2-receipt",
        type=Path,
        default=REPO / ".omx/research/m2_live_target_selection_20260720T1548Z.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/realization_g2_20260721"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_audit(
        seed_path=args.seed.resolve(),
        stage_root=args.stage_root.resolve(),
        m2_receipt_path=args.m2_receipt.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "receipt": str(args.output_root / "receipt.json"),
                "verdict": receipt["verdict"],
                "n600_status": receipt["D1_prefix_ladder"][-1]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
