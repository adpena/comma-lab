#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove exact identity between C1 live-target debt, R2b, and the S2 seed.

This is a read-only scientific bridge.  It does not claim that S2 realizes a
camera-space witness: S2 still requires a baseline partition and a receiver
that inverse-realizes the corrected cells through the frozen resize/scorer
surface.  The bridge closes the narrower but valuable fact that the finite S2
packet is exactly the complete Seg-cell error topology of the custodied C1
baseline under the evaluator's batch-16 geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.s2_partition_seed import (  # noqa: E402
    SEMANTIC_NAMES,
    decode_partition_seed,
    encode_partition_seed,
)

SCHEMA = "tac.c0b_s2_debt_bridge.v4"
DEBT_SCHEMA = "tac.coupled_witness_raw_debt.v2"
R2B_SCHEMA = "r2b_sparse_target_selection_receipt.v1"
R2B_STAGE_SCHEMA = "r2b_hard_oracle_batch.v1"
S2_SCHEMA = "s2_partition_seed_measurement.v1"
DEFAULT_DEBT = REPO / (
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
    "c1_live_target_debt_n600_batch16.json"
)
DEFAULT_R2B_RECEIPT = Path(
    "/Volumes/VertigoDataTier/pact/evidence/"
    "r2b_sparse_target_selection_20260720T1621Z/receipt.json"
)
DEFAULT_R2B_STAGES = Path(
    "/Volumes/VertigoDataTier/pact/evidence/r2b_sparse_target_selection_20260720T1621Z/"
    "baseline_stages_a7192f938785_31d77be9ab9f_107a7d3a179d"
)
DEFAULT_S2_PACKET = Path(
    "/Volumes/VertigoDataTier/pact/evidence/s2_compose_20260721/"
    "partition_seed/s2_partition_event_seed.bin"
)
DEFAULT_S2_RECEIPT = REPO / ".omx/research/s2_compose_full_partition_20260721T041640Z.json"


class BridgeError(RuntimeError):
    """Fail-closed receipt, custody, or identity error."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BridgeError("value is not canonical-JSON encodable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BridgeError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return _load_json_snapshot(path, label)[0]


def _load_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise BridgeError(f"{label} root must be an object")
    return value, raw


def _validate_body_hash(payload: Mapping[str, Any], field: str) -> None:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise BridgeError(f"{field} is absent or malformed")
    body = {key: value for key, value in payload.items() if key != field}
    if _sha256_bytes(_canonical(body)) != observed:
        raise BridgeError(f"{field} differs from its receipt body")


def _tree_hash(files: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in files:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise BridgeError("R2b stage escaped its root") from exc
        digest.update(str(relative).encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _event_stream_from_debt(
    path: Path,
) -> tuple[dict[str, Any], list[list[int]], list[float], dict[str, Any]]:
    receipt, receipt_bytes = _load_json_snapshot(path, "batch-16 debt receipt")
    _validate_body_hash(receipt, "receipt_sha256")
    if receipt.get("schema") != DEBT_SCHEMA:
        raise BridgeError("debt receipt is not the superseding batch-16/live-target schema")
    if any(receipt.get(key) is not False for key in ("score_claim", "promotion_eligible", "pointer_moved")):
        raise BridgeError("debt receipt authority flags differ")
    config = receipt.get("config")
    aggregate = receipt.get("aggregate")
    rows = receipt.get("pairs")
    stages = receipt.get("stages")
    if not all(isinstance(value, dict) for value in (config, aggregate)):
        raise BridgeError("debt config/aggregate is malformed")
    if not isinstance(rows, list) or not isinstance(stages, list):
        raise BridgeError("debt rows/stages are malformed")
    pair_count = config.get("pair_count")
    stage_pairs = config.get("stage_pairs")
    scorer_hw = config.get("scorer_hw")
    if (
        not isinstance(pair_count, int)
        or pair_count <= 0
        or not isinstance(stage_pairs, int)
        or stage_pairs <= 0
        or config.get("scorer_batch_pairs") != stage_pairs
        or not isinstance(scorer_hw, list)
        or len(scorer_hw) != 2
        or any(not isinstance(value, int) or value <= 0 for value in scorer_hw)
    ):
        raise BridgeError("debt receipt lacks exact scorer population geometry")
    if [row.get("pair_id") for row in rows if isinstance(row, dict)] != list(range(pair_count)):
        raise BridgeError("debt pair rows are not the canonical complete sequence")

    events: list[list[int]] = []
    pose_mse: list[float] = []
    target_class_counts: Counter[int] = Counter()
    prior_site = -1
    height, width = scorer_hw
    for pair_id, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BridgeError("debt pair row must be an object")
        local_events = row.get("seg_events")
        if not isinstance(local_events, list) or row.get("seg_mismatched_pixels") != len(local_events):
            raise BridgeError("debt per-pair Seg event count differs")
        if _sha256_bytes(_canonical(local_events)) != row.get("seg_events_sha256"):
            raise BridgeError("debt per-pair Seg event hash differs")
        for local in local_events:
            if not isinstance(local, list) or len(local) != 4 or any(type(value) is not int for value in local):
                raise BridgeError("debt event grammar differs")
            event = [pair_id, *local]
            _, event_row, col, target_class, candidate_class = event
            if (
                not 0 <= event_row < height
                or not 0 <= col < width
                or not 0 <= target_class < len(SEMANTIC_NAMES)
                or not 0 <= candidate_class < len(SEMANTIC_NAMES)
                or target_class == candidate_class
            ):
                raise BridgeError("debt event is outside the frozen geometry/alphabet")
            site = (pair_id * height + event_row) * width + col
            if site <= prior_site:
                raise BridgeError("debt events are not unique and strictly site-sorted")
            prior_site = site
            events.append(event)
            target_class_counts[target_class] += 1
        pose_value = row.get("d_pose")
        if not isinstance(pose_value, (int, float)) or not math.isfinite(float(pose_value)):
            raise BridgeError("debt pose row is non-finite")
        pose_mse.append(float(pose_value))

    if aggregate.get("pair_count") != pair_count or aggregate.get("seg_event_count") != len(events):
        raise BridgeError("debt aggregate population/event count differs")
    if aggregate.get("seg_event_stream_sha256") != _sha256_bytes(_canonical(events)):
        raise BridgeError("debt aggregate event-stream hash differs")
    if aggregate.get("pair_rows_sha256") != _sha256_bytes(_canonical(rows)):
        raise BridgeError("debt aggregate pair-row hash differs")
    expected_dseg = len(events) / (pair_count * height * width)
    if not math.isclose(float(aggregate.get("mean_d_seg")), expected_dseg, rel_tol=0.0, abs_tol=1e-18):
        raise BridgeError("debt aggregate d_seg does not equal the event population")
    if not math.isclose(
        float(aggregate.get("mean_d_pose")), math.fsum(pose_mse) / pair_count, rel_tol=0.0, abs_tol=1e-18
    ):
        raise BridgeError("debt aggregate d_pose does not equal its pair rows")
    per_class = aggregate.get("per_class")
    if not isinstance(per_class, dict):
        raise BridgeError("debt aggregate lacks per-class routing debt")
    for class_id, name in enumerate(SEMANTIC_NAMES):
        row = per_class.get(name)
        if not isinstance(row, dict) or row.get("class_id") != class_id or row.get("errors") != target_class_counts[class_id]:
            raise BridgeError("debt per-class event attribution differs")

    expected_starts = list(range(0, pair_count, stage_pairs))
    if len(stages) != len(expected_starts):
        raise BridgeError("debt stage count differs from batch geometry")
    for index, (stage_row, start) in enumerate(zip(stages, expected_starts, strict=True)):
        if not isinstance(stage_row, dict):
            raise BridgeError("debt stage custody row is malformed")
        end = min(start + stage_pairs, pair_count)
        stage_path = Path(str(stage_row.get("path", "")))
        if (
            stage_row.get("pair_start") != start
            or stage_row.get("pair_end_exclusive") != end
            or not stage_path.is_file()
        ):
            raise BridgeError(f"debt stage custody differs at batch {index}")
        stage, stage_bytes = _load_json_snapshot(stage_path, "debt stage")
        _validate_body_hash(stage, "stage_sha256")
        if (
            stage_row.get("bytes") != len(stage_bytes)
            or stage_row.get("sha256") != _sha256_bytes(stage_bytes)
            or stage.get("config_sha256") != receipt.get("config_sha256")
            or stage.get("stage_sha256") != stage_row.get("stage_sha256")
            or stage.get("pair_start") != start
            or stage.get("pair_end_exclusive") != end
            or stage.get("rows") != rows[start:end]
        ):
            raise BridgeError(f"debt stage content differs at batch {index}")
    custody = {
        "path": str(path.resolve()),
        "bytes": len(receipt_bytes),
        "sha256": _sha256_bytes(receipt_bytes),
        "receipt_sha256": receipt["receipt_sha256"],
    }
    return receipt, events, pose_mse, custody


def _event_stream_from_r2b(
    receipt_path: Path,
    stage_dir: Path,
    *,
    debt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[list[int]], list[float], dict[str, Any]]:
    receipt, receipt_bytes = _load_json_snapshot(receipt_path, "R2b receipt")
    if receipt.get("schema") != R2B_SCHEMA or receipt.get("score_claim") is not False:
        raise BridgeError("R2b receipt schema/authority differs")
    config = debt["config"]
    aggregate = debt["aggregate"]
    pair_count = config["pair_count"]
    batch_pairs = config["stage_pairs"]
    for r2b_key, debt_key in (("baseline_raw", "raw"), ("target_raw", "target_raw")):
        r2b_raw = receipt.get(r2b_key)
        debt_raw = config.get(debt_key)
        if not isinstance(r2b_raw, dict) or not isinstance(debt_raw, dict):
            raise BridgeError("R2b/debt raw custody is malformed")
        if r2b_raw.get("bytes") != debt_raw.get("bytes") or r2b_raw.get("sha256") != debt_raw.get("sha256"):
            raise BridgeError("R2b and debt decoded-raw identities differ")
    if receipt.get("hard_oracle_batch_size") != batch_pairs:
        raise BridgeError("R2b batch geometry differs from debt receipt")
    baseline = receipt.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("flip_count") != aggregate.get("seg_event_count"):
        raise BridgeError("R2b baseline event count differs")

    files = sorted(stage_dir.glob("batch-*.json"))
    expected_starts = list(range(0, pair_count, batch_pairs))
    if len(files) != len(expected_starts):
        raise BridgeError("R2b stage count differs")
    events: list[list[int]] = []
    pose_mse: list[float] = []
    cache_mismatches = 0
    stage_snapshots: list[tuple[Path, bytes]] = []
    for path, start in zip(files, expected_starts, strict=True):
        stage, stage_bytes = _load_json_snapshot(path, "R2b stage")
        stage_snapshots.append((path, stage_bytes))
        end = min(start + batch_pairs, pair_count)
        flips = stage.get("flips")
        pose_squared = stage.get("pose_squared_error")
        if (
            stage.get("schema") != R2B_STAGE_SCHEMA
            or stage.get("pair_start") != start
            or stage.get("pair_stop") != end
            or not isinstance(flips, list)
            or stage.get("flip_count") != len(flips)
            or not isinstance(pose_squared, list)
            or len(pose_squared) != end - start
        ):
            raise BridgeError(f"R2b stage grammar differs: {path}")
        for flip in flips:
            if not isinstance(flip, list) or len(flip) < 6:
                raise BridgeError("R2b flip grammar differs")
            event = flip[:5]
            if any(type(value) is not int for value in event):
                raise BridgeError("R2b event identity is not integral")
            events.append(event)
        for dimensions in pose_squared:
            if not isinstance(dimensions, list) or len(dimensions) != 6:
                raise BridgeError("R2b Pose squared-error row differs")
            values = [float(value) for value in dimensions]
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise BridgeError("R2b Pose squared-error component is invalid")
            pose_mse.append(math.fsum(values) / 6.0)
        cache_mismatches += int(stage.get("cache_label_mismatches", -1))
    if len(pose_mse) != pair_count or events != sorted(events, key=lambda row: row[:3]):
        raise BridgeError("R2b population/order differs")
    if len({tuple(event[:3]) for event in events}) != len(events):
        raise BridgeError("R2b sites are not unique")
    tree_digest = hashlib.sha256()
    for stage_path, stage_bytes in stage_snapshots:
        tree_digest.update(str(stage_path.relative_to(stage_dir)).encode("utf-8") + b"\0")
        tree_digest.update(bytes.fromhex(_sha256_bytes(stage_bytes)))
    tree = tree_digest.hexdigest()
    custody = {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_bytes": len(receipt_bytes),
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "stage_dir": str(stage_dir.resolve()),
        "stage_count": len(files),
        "stage_tree_sha256": tree,
        "cache_label_mismatches": cache_mismatches,
    }
    return receipt, events, pose_mse, custody


def _event_stream_from_s2(
    packet_path: Path,
    receipt_path: Path,
    *,
    r2b_custody: Mapping[str, Any],
) -> tuple[dict[str, Any], list[list[int]], dict[str, Any], Any]:
    receipt, receipt_bytes = _load_json_snapshot(receipt_path, "S2 measurement receipt")
    if receipt.get("schema") != S2_SCHEMA or receipt.get("score_claim") is not False:
        raise BridgeError("S2 receipt schema/authority differs")
    try:
        packet = packet_path.read_bytes()
    except OSError as exc:
        raise BridgeError("cannot read S2 packet") from exc
    finite = receipt.get("finite_packet")
    inventory = receipt.get("inventory_custody")
    if not isinstance(finite, dict) or not isinstance(inventory, dict):
        raise BridgeError("S2 packet/inventory custody is malformed")
    packet_sha = _sha256_bytes(packet)
    if (
        finite.get("packet_bytes") != len(packet)
        or finite.get("counted_seed_bytes") != len(packet)
        or finite.get("packet_sha256") != packet_sha
        or finite.get("parse_back_event_identity") is not True
    ):
        raise BridgeError("S2 packet accounting differs")
    if (
        inventory.get("stage_count") != r2b_custody.get("stage_count")
        or inventory.get("stage_tree_sha256") != r2b_custody.get("stage_tree_sha256")
        or inventory.get("cache_label_mismatches") != r2b_custody.get("cache_label_mismatches")
    ):
        raise BridgeError("S2 receipt does not bind the selected R2b inventory")
    seed = decode_partition_seed(packet)
    if encode_partition_seed(seed) != packet:
        raise BridgeError("S2 packet is not the canonical encoding of its decoded seed")
    if list(seed.semantic_class_ids) != list(range(len(SEMANTIC_NAMES))):
        raise BridgeError("S2 semantic class registry differs from the frozen scorer")
    events = [
        [event.pair, event.row, event.col, event.target_class, event.baseline_class]
        for event in seed.events
    ]
    if len(events) != finite.get("event_count") or seed.n_pairs != receipt.get("n_pairs"):
        raise BridgeError("S2 decoded population differs from its receipt")
    custody = {
        "receipt_path": str(receipt_path.resolve()),
        "receipt_bytes": len(receipt_bytes),
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "packet_path": str(packet_path.resolve()),
        "packet_bytes": len(packet),
        "packet_sha256": packet_sha,
    }
    # Return the seed decoded from the exact packet byte snapshot above.  The
    # caller must not reopen the path for geometry: doing so would permit a
    # path-swap race to combine packet identity from one file with dimensions
    # from another.
    return receipt, events, custody, seed


def verify_bridge(
    *,
    debt_path: Path,
    r2b_receipt_path: Path,
    r2b_stage_dir: Path,
    s2_packet_path: Path,
    s2_receipt_path: Path,
) -> dict[str, Any]:
    debt, debt_events, debt_pose, debt_custody = _event_stream_from_debt(debt_path)
    r2b, r2b_events, r2b_pose, r2b_custody = _event_stream_from_r2b(
        r2b_receipt_path, r2b_stage_dir, debt=debt
    )
    _s2, s2_events, s2_custody, seed = _event_stream_from_s2(
        s2_packet_path, s2_receipt_path, r2b_custody=r2b_custody
    )
    debt_geometry = debt["config"]["scorer_hw"]
    debt_pairs = debt["config"]["pair_count"]
    if [seed.height, seed.width] != debt_geometry or seed.n_pairs != debt_pairs:
        raise BridgeError("S2 population geometry differs from the batch-16 debt authority grid")
    if debt_events != r2b_events:
        raise BridgeError("batch-16 debt events differ from the R2b live-target inventory")
    if debt_events != s2_events:
        raise BridgeError("batch-16 debt events differ from the decoded S2 packet")
    pose_deltas = [abs(left - right) for left, right in zip(debt_pose, r2b_pose, strict=True)]
    max_pose_delta = max(pose_deltas, default=0.0)
    if max_pose_delta > 1e-18:
        raise BridgeError("batch-16 debt Pose rows differ from R2b")
    stream_sha = _sha256_bytes(_canonical(debt_events))
    class_counts = Counter(event[3] for event in debt_events)
    result = {
        "schema": SCHEMA,
        "verdict": "EXACT_C1_LIVE_TARGET_DEBT_EQUALS_R2B_INVENTORY_EQUALS_S2_PACKET",
        "verdict_scope": (
            "all scorer-grid Seg mismatch sites/classes for the exact C1 baseline against the exact M2 "
            "zero-distortion target under the frozen batch-16 local CPU geometry; this is not a camera-space "
            "receiver, archive, score, or authority result"
        ),
        "debt_receipt": {
            **debt_custody,
            "candidate_raw": debt["config"]["raw"],
            "target_raw": debt["config"]["target_raw"],
        },
        "r2b": {
            **r2b_custody,
            "receipt_schema": r2b["schema"],
            "hard_oracle_batch_size": r2b["hard_oracle_batch_size"],
        },
        "s2": s2_custody,
        "identity": {
            "event_count": len(debt_events),
            "event_stream_schema": "[pair,row,col,target_class,baseline_class]",
            "event_stream_sha256": stream_sha,
            "debt_equals_r2b": True,
            "debt_equals_s2": True,
            "r2b_equals_s2": True,
            "strict_site_order": True,
            "unique_sites": True,
            "per_target_class_errors": {
                name: class_counts[class_id] for class_id, name in enumerate(SEMANTIC_NAMES)
            },
            "pose_rows_equal_r2b": True,
            "pose_max_abs_mse_delta": max_pose_delta,
        },
        "authority_geometry": {
            "pair_count": debt_pairs,
            "scorer_hw": debt_geometry,
            "total_seg_sites": debt_pairs * debt_geometry[0] * debt_geometry[1],
            "mean_d_seg": debt["aggregate"]["mean_d_seg"],
            "baseline_mean_d_pose": debt["aggregate"]["mean_d_pose"],
            "s2_geometry_exact": True,
        },
        "composition_consequence": {
            "new_fact": (
                "S2 is the exact finite Seg-error topology/class correction of the C1 baseline, not merely a prior"
            ),
            "direct_route": (
                "semantic base -> baseline partition -> apply S2 cell constraints -> inverse-realize those "
                "classes through R -> encode only remaining quotient; solve Qpose conditionally with Y1 fixed"
            ),
            "still_open": [
                "compact source-derived baseline partition predictor",
                "camera-space uint8 inverse realization of S2 class constraints",
                "joint Pose/rate closure and exact archive runtime",
            ],
        },
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    result["receipt_sha256"] = _sha256_bytes(_canonical(result))
    return result


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise BridgeError(f"write-once output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.new-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if not path.is_file() or path.read_bytes() != encoded:
                raise BridgeError(f"concurrent write-once output differs: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debt", type=Path, default=DEFAULT_DEBT)
    parser.add_argument("--r2b-receipt", type=Path, default=DEFAULT_R2B_RECEIPT)
    parser.add_argument("--r2b-stage-dir", type=Path, default=DEFAULT_R2B_STAGES)
    parser.add_argument("--s2-packet", type=Path, default=DEFAULT_S2_PACKET)
    parser.add_argument("--s2-receipt", type=Path, default=DEFAULT_S2_RECEIPT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = verify_bridge(
            debt_path=args.debt.expanduser().resolve(),
            r2b_receipt_path=args.r2b_receipt.expanduser().resolve(),
            r2b_stage_dir=args.r2b_stage_dir.expanduser().resolve(),
            s2_packet_path=args.s2_packet.expanduser().resolve(),
            s2_receipt_path=args.s2_receipt.expanduser().resolve(),
        )
        output = args.output.expanduser().resolve()
        _write_once(output, receipt)
    except (OSError, BridgeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "receipt_sha256": receipt["receipt_sha256"],
                "verdict": receipt["verdict"],
                "event_count": receipt["identity"]["event_count"],
                "score_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
