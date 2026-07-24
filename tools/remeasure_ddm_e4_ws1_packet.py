#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Exact batch-32 advisory remeasure of one decoded E4/WS1 packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_realized_flip_menu import (  # noqa: E402
    SEG_HW,
    advisory_objective,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tools.materialize_ddm_ws2_warm_start_custody import (  # noqa: E402
    CustodyConfig,
)
from tools.measure_ddm_menu1_realized_flip_menu import (  # noqa: E402
    _config_and_inputs,
    _forward,
    _load_models,
)

SCHEMA = "ddm_e4_ws1_packet_batch32_remeasure.v1"
CONFIG_SCHEMA = "DDME4WS1PacketRemeasureConfigV1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
CAMERA_SHAPE = (600, 2, 874, 1164, 3)


class RemeasureError(RuntimeError):
    """Input, scorer, or endpoint custody failed closed."""


class DDME4WS1PacketRemeasureConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDME4WS1PacketRemeasureConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    candidate: Literal["W_seg", "W_joint"]
    export_receipt_path: str
    export_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ws2_config_path: str
    ws2_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ws2_receipt_path: str
    ws2_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoded_raw_path: str
    decoded_raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_root: str
    expected_d_seg: float = Field(ge=0.0)
    expected_d_pose: float = Field(ge=0.0)
    scorer_batch_size: Literal[32] = 32
    scorer_threads: Literal[4] = 4
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    pointer: Literal["0.1910828242 [contest-CPU]"] = "0.1910828242 [contest-CPU]"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _bound_json(path_value: str, sha256: str, label: str) -> dict[str, Any]:
    path = _resolve(path_value)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise RemeasureError(f"{label} SHA-256 differs")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RemeasureError(f"{label} is malformed JSON") from exc
    return value


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = rfc8785_canonicalize(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RemeasureError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_preserved_row(
    path: Path,
    *,
    candidate: str,
    decoded_raw_sha256: str,
    camera_sha256: str,
    start: int,
    stop: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise RemeasureError(f"preserved scorer row is malformed: {path}") from exc
    expected_keys = {
        "camera_sha256",
        "candidate",
        "decoded_raw_sha256",
        "errors",
        "evidence_axis",
        "pair_range",
        "per_class",
        "pose_coordinates",
        "pose_squared_error_sum",
        "schema",
        "score_claim",
        "sites",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value["schema"] != "ddm_e4_ws1_packet_batch32_row.v1"
        or value["candidate"] != candidate
        or value["decoded_raw_sha256"] != decoded_raw_sha256
        or value["camera_sha256"] != camera_sha256
        or value["pair_range"] != [start, stop]
        or value["evidence_axis"] != EVIDENCE_AXIS
        or value["score_claim"] is not False
    ):
        raise RemeasureError(f"preserved scorer row custody differs: {path}")
    return value


def remeasure(
    config_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    config_payload = config_path.read_bytes()
    config = DDME4WS1PacketRemeasureConfigV1.model_validate_json(
        config_payload,
        strict=True,
    )
    if (
        rfc8785_canonicalize(config.model_dump(mode="json", by_alias=True)) + b"\n"
        != config_payload
    ):
        raise RemeasureError("remeasure config must be canonical JSON")
    export = _bound_json(
        config.export_receipt_path,
        config.export_receipt_sha256,
        "export receipt",
    )
    ws2_value = _bound_json(
        config.ws2_receipt_path,
        config.ws2_receipt_sha256,
        "WS2 receipt",
    )
    ws2_config_path = _resolve(config.ws2_config_path)
    ws2_config_payload = ws2_config_path.read_bytes()
    if hashlib.sha256(ws2_config_payload).hexdigest() != config.ws2_config_sha256:
        raise RemeasureError("WS2 config SHA-256 differs")
    ws2_config = CustodyConfig.model_validate_json(ws2_config_payload)
    expected = ws2_value["fresh_batch32_endpoints"][config.candidate]
    if (
        float(expected["d_seg"]) != config.expected_d_seg
        or float(expected["d_pose"]) != config.expected_d_pose
        or export["source"]["sha256"] != expected["archive_sha256"]
        or export["source"]["bytes"] != expected["archive_bytes"]
    ):
        raise RemeasureError("sealed endpoint and packet source custody differ")

    raw_path = Path(config.decoded_raw_path)
    raw_bytes, raw_sha256 = _sha256_file(raw_path)
    if (
        raw_sha256 != config.decoded_raw_sha256
        or raw_sha256 != export["output_identity"]["sha256"]
        or raw_bytes != export["output_identity"]["bytes"]
        or raw_bytes != int(np.prod(CAMERA_SHAPE, dtype=np.int64))
    ):
        raise RemeasureError("decoded raw identity differs from export receipt")

    menu_config, _ = _config_and_inputs(_resolve(ws2_config.menu1_config_path))
    with np.load(
        Path(menu_config.target_cache_path),
        allow_pickle=False,
    ) as target_cache:
        labels = np.asarray(target_cache["lstars"], dtype=np.uint8)
        poses = np.asarray(target_cache["gt_poses"], dtype=np.float64)
    if labels.shape != (600, *SEG_HW) or poses.shape != (600, 6):
        raise RemeasureError("target cache geometry differs")
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    segnet, posenet, scorer_custody = _load_models(menu_config)
    scorer_custody["batch_size"] = 32

    raw = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=CAMERA_SHAPE)
    checkpoint_root = Path(config.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        row_path = checkpoint_root / f"batch_{start:04d}_{stop:04d}.json"
        camera = np.array(raw[start:stop], copy=True, order="C")
        camera_sha256 = hashlib.sha256(camera.tobytes()).hexdigest()
        preserved = _load_preserved_row(
            row_path,
            candidate=config.candidate,
            decoded_raw_sha256=raw_sha256,
            camera_sha256=camera_sha256,
            start=start,
            stop=stop,
        )
        if preserved is not None:
            rows.append(preserved)
            print(
                f"[E4-WS1] {config.candidate} scorer {start:04d}:{stop:04d} preserved",
                flush=True,
            )
            continue
        cells, pose6 = _forward(segnet, posenet, camera)
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        per_class = {
            class_name: {
                "errors": int(
                    np.count_nonzero((cells != target) & (target == class_id))
                ),
                "sites": int(np.count_nonzero(target == class_id)),
            }
            for class_id, class_name in enumerate(CLASS_ORDER)
        }
        row = {
            "camera_sha256": camera_sha256,
            "candidate": config.candidate,
            "decoded_raw_sha256": raw_sha256,
            "errors": int(np.count_nonzero(cells != target)),
            "evidence_axis": EVIDENCE_AXIS,
            "pair_range": [start, stop],
            "per_class": per_class,
            "pose_coordinates": int(pose6.size),
            "pose_squared_error_sum": float(
                np.square(pose6 - target_pose).sum(dtype=np.float64)
            ),
            "schema": "ddm_e4_ws1_packet_batch32_row.v1",
            "score_claim": False,
            "sites": int(cells.size),
        }
        _publish(row_path, row)
        rows.append(row)
        print(
            f"[E4-WS1] {config.candidate} scorer {start:04d}:{stop:04d}",
            flush=True,
        )
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    if d_seg != config.expected_d_seg or d_pose != config.expected_d_pose:
        raise RemeasureError(
            f"decoded packet did not reproduce the exact sealed batch-32 endpoint: {(d_seg, d_pose)!r}"
        )
    per_class = {
        class_name: {
            "errors": sum(int(row["per_class"][class_name]["errors"]) for row in rows),
            "sites": sum(int(row["per_class"][class_name]["sites"]) for row in rows),
        }
        for class_name in CLASS_ORDER
    }
    for value in per_class.values():
        value["d_seg"] = value["errors"] / value["sites"]
    result = {
        "candidate": config.candidate,
        "checkpoint_root": str(checkpoint_root),
        "decoded_raw": {
            "bytes": raw_bytes,
            "path": str(raw_path),
            "sha256": raw_sha256,
        },
        "endpoint": {
            "advisory_objective_at_packed_bytes": advisory_objective(
                errors=errors,
                sites=sites,
                d_pose=d_pose,
                bytes_=int(export["archive"]["bytes"]),
            ),
            "d_pose": d_pose,
            "d_seg": d_seg,
            "errors": errors,
            "per_class": per_class,
            "sites": sites,
        },
        "evidence_axis": EVIDENCE_AXIS,
        "export_receipt": {
            "path": config.export_receipt_path,
            "sha256": config.export_receipt_sha256,
        },
        "pointer": config.pointer,
        "pointer_moved": False,
        "research_only": True,
        "schema": SCHEMA,
        "score_claim": False,
        "scorer_batch_size": 32,
        "scorer_custody": scorer_custody,
        "scorer_threads": 4,
        "stage_count": len(rows),
        "verdict": "EXACT_SEALED_BATCH32_ENDPOINT_REPRODUCED_FROM_DECODED_PACKET",
        "verdict_scope": (
            "INSTANCE: typed E4/WS1 packet decoded on macOS CPU with frozen "
            "scorers; no contest eval, promotion, or frontier mutation."
        ),
    }
    _publish(receipt_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    result = remeasure(
        Path(args.config).resolve(),
        Path(args.receipt).resolve(),
    )
    print(
        json.dumps(
            {
                "candidate": result["candidate"],
                "endpoint": result["endpoint"],
                "verdict": result["verdict"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
