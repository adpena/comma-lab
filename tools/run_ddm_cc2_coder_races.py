#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run resumable DDM CC2 Race 2 and Race 3 on exact counted payloads."""

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
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_cc2_coder_races import (  # noqa: E402
    EVIDENCE_AXIS,
    POINTER,
    RACE2_SCHEMA,
    build_per_stream_price_table,
    build_quantization_arms,
    race2_delta,
    reprice_zero_pose_composition,
    sha256_bytes,
)
from tac.optimization.ddm_pc1_pose_stream import parse_pc1_packet  # noqa: E402
from tools.launch_ddm_joint_descent import (  # noqa: E402
    _chunked_n600_verdict,
    _load_cpu_frozen_scorers,
)


class CC2RunnerError(ValueError):
    """A typed config, checkpoint, custody binding, or stage differs."""


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _canonical_json(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CC2RunnerError(f"JSON root must be an object: {path}")
    return value


def _read_bound(
    binding: dict[str, Any],
    *,
    require_bytes: bool = True,
    materialize: bool = True,
) -> tuple[bytes | None, dict[str, Any]]:
    path = Path(binding["path"])
    if not path.is_file() or path.is_symlink():
        raise CC2RunnerError(f"bound file is absent or not regular: {path}")
    actual = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if require_bytes and actual["bytes"] != binding.get("bytes"):
        raise CC2RunnerError(f"bound file byte count differs: {path}")
    if actual["sha256"] != binding.get("sha256"):
        raise CC2RunnerError(f"bound file SHA-256 differs: {path}")
    return path.read_bytes() if materialize else None, actual


def _config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(_canonical_json(config))


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    required = {
        "schema",
        "run_id",
        "seed",
        "batch_size",
        "minimum_free_bytes",
        "source_archive",
        "j8f_checkpoint",
        "j8f_receipt",
        "pose_packet",
        "target_cache",
        "upstream_root",
        "q8_parent_sha256",
        "output_root",
        "source_harvest",
    }
    if set(config) != required or config["schema"] != "ddm_cc2_coder_races_config.v1":
        raise CC2RunnerError("typed config keys/schema differ")
    if config["seed"] != 0 or config["batch_size"] <= 0:
        raise CC2RunnerError("typed seed or batch size differs")
    output = Path(config["output_root"])
    if not output.is_absolute() or not str(output).startswith(
        ("/Volumes/VertigoDataTier/pact/", "/Volumes/APDataStore/pact/")
    ):
        raise CC2RunnerError("CC2 output root must be on the governed SSD tier")
    return config


def _load_checkpoint(path: Path, *, schema: str, config_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read_json(path)
    if value.get("schema") != schema or value.get("typed_config_hash") != config_hash:
        raise CC2RunnerError(f"preserved checkpoint identity differs: {path}")
    return value


def _file_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CC2RunnerError(f"expected artifact is absent or not regular: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _preflight(config: dict[str, Any], output: Path, config_hash: str) -> dict[str, Any]:
    source, source_custody = _read_bound(config["source_archive"])
    checkpoint, checkpoint_custody = _read_bound(config["j8f_checkpoint"])
    j8f_payload, j8f_custody = _read_bound(config["j8f_receipt"])
    pose_payload, pose_custody = _read_bound(config["pose_packet"])
    _, target_custody = _read_bound(config["target_cache"], materialize=False)
    harvest_custody: dict[str, Any] = {}
    for name, binding in config["source_harvest"].items():
        _, harvest_custody[name] = _read_bound(binding, require_bytes=False)
    del source, checkpoint, j8f_payload, pose_payload

    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < int(config["minimum_free_bytes"]):
        raise CC2RunnerError("REFUSE_STORAGE_PREFLIGHT_INSUFFICIENT_SSD_FREE_BYTES")
    upstream = Path(config["upstream_root"])
    if not (upstream / "evaluate.py").is_file() or not (upstream / "modules.py").is_file():
        raise CC2RunnerError("frozen upstream scorer root is incomplete")
    return {
        "schema": "ddm_cc2_stage00_preflight.v1",
        "typed_config_hash": config_hash,
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_custody": {
            "source_archive": source_custody,
            "j8f_checkpoint": checkpoint_custody,
            "j8f_receipt": j8f_custody,
            "pose_packet": pose_custody,
            "target_cache": target_custody,
            "source_harvest": harvest_custody,
        },
        "storage": {
            "output_root": str(output),
            "free_bytes": usage.free,
            "minimum_free_bytes": config["minimum_free_bytes"],
            "waterfall_tier": "VertigoDataTier",
            "auto_cleanup": (
                "ATOMIC_TEMP_FILES_REMOVED_ON_SUCCESS_OR_FAILURE; PRESERVED CANDIDATES "
                "ARE SMALL NON_REBUILDABLE MEASUREMENT ARTIFACTS"
            ),
        },
        "hardware": platform.platform(),
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "pointer": POINTER,
        "pointer_moved": False,
    }


def _build_stage(
    config: dict[str, Any],
    output: Path,
    config_hash: str,
) -> dict[str, Any]:
    source, _ = _read_bound(config["source_archive"])
    checkpoint_path = Path(config["j8f_checkpoint"]["path"])
    with np.load(checkpoint_path, allow_pickle=False) as loaded:
        theta = np.ascontiguousarray(loaded["theta"], dtype=np.float32)
        checkpoint_step = int(loaded["__ddmjd_step"])
    pose_payload, _ = _read_bound(config["pose_packet"])
    pose_packet = parse_pc1_packet(pose_payload)
    arms = build_quantization_arms(
        source_archive=source,
        theta=theta,
        pose_packet=pose_packet,
        seed=int(config["seed"]),
    )
    if arms["CAMERA_Q8_EXACT"].parent_archive.hex() == "":
        raise CC2RunnerError("Q8 arm unexpectedly empty")
    if sha256_bytes(arms["CAMERA_Q8_EXACT"].parent_archive) != config["q8_parent_sha256"]:
        raise CC2RunnerError("Q8 arm does not reproduce the landed J8F exact archive")

    arm_rows: dict[str, Any] = {}
    for arm_id, arm in arms.items():
        arm_dir = output / "race2_candidates" / arm_id.lower()
        parent_path = arm_dir / "w_joint_parent.zip"
        composition_path = arm_dir / "w_joint_plus_pc1.zip"
        _atomic_bytes(parent_path, arm.parent_archive)
        _atomic_bytes(composition_path, arm.composition_archive)
        arm_rows[arm_id] = {
            "arm_id": arm_id,
            "schedule": arm.schedule,
            "realized_parameter_count": int(np.count_nonzero(arm.realized_theta)),
            "realized_theta_sha256": sha256_bytes(np.ascontiguousarray(arm.realized_theta, dtype="<f4").tobytes()),
            "parent_archive": _file_receipt(parent_path),
            "composition_archive": _file_receipt(composition_path),
        }
    return {
        "schema": "ddm_cc2_stage01_candidates.v1",
        "typed_config_hash": config_hash,
        "checkpoint_step": checkpoint_step,
        "theta_sha256": sha256_bytes(theta.astype("<f4", copy=False).tobytes()),
        "arms": arm_rows,
        "score_claim": False,
        "pointer": POINTER,
        "pointer_moved": False,
    }


def _load_candidate(stage1: dict[str, Any], arm_id: str) -> tuple[bytes, bytes]:
    row = stage1["arms"][arm_id]
    parent, parent_custody = _read_bound(row["parent_archive"])
    composition, composition_custody = _read_bound(row["composition_archive"])
    if parent_custody != row["parent_archive"] or composition_custody != row["composition_archive"]:
        raise CC2RunnerError(f"{arm_id} candidate file custody differs")
    return parent, composition


def _score_arm(
    *,
    arm_id: str,
    stage1: dict[str, Any],
    config: dict[str, Any],
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    config_hash: str,
) -> dict[str, Any]:
    parent, composition = _load_candidate(stage1, arm_id)
    parent_verdict = _chunked_n600_verdict(
        archive=parent,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
        batch_size=int(config["batch_size"]),
    )
    verdict = reprice_zero_pose_composition(
        parent_verdict,
        parent_archive=parent,
        composition_archive=composition,
    )
    verdict["schema"] = "ddm_cc2_stage_quantizer_verdict.v1"
    verdict["typed_config_hash"] = config_hash
    verdict["arm_id"] = arm_id
    verdict["decision_basis"] = (
        "REAL_PARENT_RECEIVER_CAMERA_UINT8_R_FROZEN_SCORERS_N600_PLUS_EXACT_COUNTED_ZERO_EFFECT_PC1_WRAPPER"
    )
    verdict["verdict_scope"] = (
        "ONE_TERMINAL_SINGLE_PASS_QUANTIZER_PROXY_ON_LANDED_J8F_THETA; NOT A RETRAINING OR FAMILY VERDICT"
    )
    return verdict


def _reference_verdict(
    stage1: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    parent, composition = _load_candidate(stage1, "CAMERA_Q8_EXACT")
    receipt = _read_json(Path(config["j8f_receipt"]["path"]))
    reference = receipt.get("step4", {}).get("reference")
    if not isinstance(reference, dict):
        raise CC2RunnerError("J8F receipt lacks its exact Step-4 reference")
    if (
        reference.get("archive_sha256") != sha256_bytes(parent)
        or reference.get("archive_bytes") != len(parent)
        or reference.get("num_pairs") != 600
    ):
        raise CC2RunnerError("J8F reference does not bind the reproduced Q8 parent")
    verdict = reprice_zero_pose_composition(
        reference,
        parent_archive=parent,
        composition_archive=composition,
    )
    verdict["schema"] = "ddm_cc2_stage_quantizer_verdict.v1"
    verdict["typed_config_hash"] = config_hash
    verdict["arm_id"] = "CAMERA_Q8_EXACT"
    verdict["decision_basis"] = (
        "REUSED_LANDED_EXACT_N600_VERDICT_BY_IDENTICAL_PARENT_SHA_PLUS_EXACT_COUNTED_ZERO_EFFECT_PC1_WRAPPER"
    )
    verdict["verdict_scope"] = "EXACT SAME ARCHIVE BYTES AS LANDED J8F STEP4"
    return verdict


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    config_hash = _config_hash(config)
    output = Path(config["output_root"])
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    stage00_path = checkpoints / "00_preflight.json"
    stage00 = _load_checkpoint(
        stage00_path,
        schema="ddm_cc2_stage00_preflight.v1",
        config_hash=config_hash,
    )
    if stage00 is None:
        stage00 = _preflight(config, output, config_hash)
        _atomic_json(stage00_path, stage00)

    stage01_path = checkpoints / "01_candidates.json"
    stage01 = _load_checkpoint(
        stage01_path,
        schema="ddm_cc2_stage01_candidates.v1",
        config_hash=config_hash,
    )
    if stage01 is None:
        stage01 = _build_stage(config, output, config_hash)
        _atomic_json(stage01_path, stage01)

    reference_path = checkpoints / "02_camera_q8_exact.json"
    reference = _load_checkpoint(
        reference_path,
        schema="ddm_cc2_stage_quantizer_verdict.v1",
        config_hash=config_hash,
    )
    if reference is None:
        reference = _reference_verdict(stage01, config, config_hash)
        _atomic_json(reference_path, reference)

    scored: dict[str, dict[str, Any]] = {"CAMERA_Q8_EXACT": reference}
    scorer_state: tuple[np.ndarray, np.ndarray, Any, Any] | None = None
    for stage_number, arm_id in (
        (3, "C3_ORIGINAL_TERMINAL_PROXY"),
        (4, "COOL_CHIC_V5_TERMINAL_PROXY"),
    ):
        path = checkpoints / f"{stage_number:02d}_{arm_id.lower()}.json"
        verdict = _load_checkpoint(
            path,
            schema="ddm_cc2_stage_quantizer_verdict.v1",
            config_hash=config_hash,
        )
        if verdict is None:
            if scorer_state is None:
                labels = open_stored_npy_memmap(config["target_cache"]["path"], "lstars")
                poses = open_stored_npy_memmap(config["target_cache"]["path"], "gt_poses")
                segnet, posenet = _load_cpu_frozen_scorers(config["upstream_root"])
                scorer_state = labels, poses, segnet, posenet
            labels, poses, segnet, posenet = scorer_state
            verdict = _score_arm(
                arm_id=arm_id,
                stage1=stage01,
                config=config,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
                config_hash=config_hash,
            )
            _atomic_json(path, verdict)
        scored[arm_id] = verdict

    race3_path = checkpoints / "05_per_stream_coder_race.json"
    race3 = _load_checkpoint(
        race3_path,
        schema="ddm_cc2_per_counted_stream_coder_race.v1",
        config_hash=config_hash,
    )
    if race3 is None:
        _, q8_composition = _load_candidate(stage01, "CAMERA_Q8_EXACT")
        race3 = build_per_stream_price_table(q8_composition)
        race3["typed_config_hash"] = config_hash
        _atomic_json(race3_path, race3)

    race2_rows: list[dict[str, Any]] = []
    for arm_id in (
        "CAMERA_Q8_EXACT",
        "C3_ORIGINAL_TERMINAL_PROXY",
        "COOL_CHIC_V5_TERMINAL_PROXY",
    ):
        verdict = scored[arm_id]
        race2_rows.append(
            {
                "arm_id": arm_id,
                "schedule": stage01["arms"][arm_id]["schedule"],
                "verdict": verdict,
                "delta_vs_camera_q8": race2_delta(reference, verdict),
            }
        )
    winner = min(
        race2_rows,
        key=lambda row: (float(row["verdict"]["advisory_action"]), str(row["arm_id"])),
    )
    receipt = {
        "schema": "ddm_cc2_coder_races_receipt.v1",
        "run_id": config["run_id"],
        "typed_config_hash": config_hash,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "seed": config["seed"],
        "deterministic_algorithms": True,
        "hardware": platform.platform(),
        "evidence_axis": EVIDENCE_AXIS,
        "race2": {
            "schema": RACE2_SCHEMA,
            "reference_arm": "CAMERA_Q8_EXACT",
            "rows": race2_rows,
            "winner": {
                "arm_id": winner["arm_id"],
                "advisory_action": winner["verdict"]["advisory_action"],
                "delta_vs_camera_q8": winner["delta_vs_camera_q8"],
            },
            "verdict_scope": (
                "TERMINAL_SINGLE_PASS_SCHEDULE_PROXY_RACE_ON_ONE_LANDED_J8F_THETA; C3/V5 WERE NOT RETRAINED"
            ),
        },
        "race3": race3,
        "stage_checkpoints": [
            _file_receipt(path)
            for path in (
                stage00_path,
                stage01_path,
                reference_path,
                checkpoints / "03_c3_original_terminal_proxy.json",
                checkpoints / "04_cool_chic_v5_terminal_proxy.json",
                race3_path,
            )
        ],
        "resumable_from_disk": True,
        "all_stage_checkpoints_preserved": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "research_only": True,
        "main_review_required": True,
        "verdict": (
            "ADVISORY_QUANTIZER_AND_PER_STREAM_PRICE_RACES_COMPLETE; NEW_CONTEXT_FRAME_RECEIVER_INTEGRATION_OWED"
        ),
    }
    receipt_path = output / "ddm_cc2_coder_races_receipt.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.config)
    print(json.dumps(receipt["race2"]["winner"], sort_keys=True))
    print(
        json.dumps(
            {"receipt": str(Path(_load_config(args.config)["output_root"]) / "ddm_cc2_coder_races_receipt.json")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
