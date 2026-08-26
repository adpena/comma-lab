#!/usr/bin/env python3
"""Build-only scorer-aware continuation of the retained WD2 student family.

The module contains real cache, paired-receiver, frozen-scorer, adaptive packet,
training, checkpoint, and retention paths.  Nothing runs at import time.  Heavy
entrypoints accept only a previously compiled config and fail closed on custody,
lane, resume, subset, or launch-authority drift.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for root in (REPO, SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_ds1_cheap_to_shrink as ds1
from experiments import ddm_wd2_width_distillation_build as wd2_build
from experiments import ddm_wd3_student_receiver as receiver
from tac.admission_guard import assert_governed_admission
from tac.scorer import load_differentiable_scorers
from tac.witness_control.resume_registry import ResumeRegistry

SCHEMA = "ddm_wd3_compiled_config.v1"
CACHE_SCHEMA = "ddm_wd3_teacher_scorer_cache.v1"
STAGE_SCHEMA = "ddm_wd3_stage_controller.v1"
CHECKPOINT_SCHEMA = "ddm_wd3_checkpoint.v1"
ARM_BIRTH_SCHEMA = "ddm_wd3_arm_birth_checkpoint.v1"
RESULT_SCHEMA = "ddm_wd3_train_result.v1"
SEED = 20260815
RATE_DENOMINATOR = 37_545_489
BASE_BYTES = 182_759
BASE_DSEG = 0.00042714
BASE_DPOSE = 0.00014747
DECODE_MSE_CEILING = 50.6728233448345
MAX_CHUNK = 120
OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation")
STAGE_A_OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter")
STAGE_A_SEEDS = (20260815, 20260816)
LANE_LEDGER = REPO / ".omx/state/active_lane_dispatch_claims.md"
BASE_RECEIPT = Path("/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json")
TEACHER_MASTER = Path(
    "/Volumes/APDataStore/pact/ddm_wd2_width_distillation/teacher_cache_e480b/"
    "retained/teacher/teacher_master_camera.rgb.u8"
)
WARM_CHECKPOINT = Path(
    "/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/"
    "checkpoints/flattened_d4_w64/distill_qat_stage_end_epoch_0060.pt"
)
WD2_TRAIN_RESULT = Path(
    "/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/TRAIN_RESULT.json"
)
WD2_ARCHIVE = Path(
    "/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/"
    "retained/candidates/flattened_d4_w64_epoch_0060/attempt_0000/archive.zip"
)
FIXED_PAIR_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/"
    "retained/candidates/flattened_d4_w64_epoch_0060/attempt_0000/"
    "advisory_n600_cpu/work/inflated/0.raw"
)
GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DESIGN = REPO / ".omx/research/ddm_wd3_scorer_aware_width_distillation_20260815.json"
POSENET = REPO / "upstream/models/posenet.safetensors"
SEGNET = REPO / "upstream/models/segnet.safetensors"

PINS: dict[str, tuple[Path, int, str]] = {
    "base_receipt": (
        BASE_RECEIPT,
        23_416,
        "cfdac1fd0965095152ffd88c878d9c4b8f38c644d755e594ad028a798daf3a7f",
    ),
    "teacher_master": (
        TEACHER_MASTER,
        1_831_204_800,
        "695023d4ca56e14f53f1e90b56134821c3c0a0c66f9b07f6aa6bd6ffdf9f4ebd",
    ),
    "warm_checkpoint": (
        WARM_CHECKPOINT,
        583_929,
        "046ee7d0171e04c3d468edd747a82bc81eb91642e5e85f17316b4419fe615071",
    ),
    "wd2_train_result": (
        WD2_TRAIN_RESULT,
        16_234,
        "c4260cf03eb4cb19f1788150592bf84f5468ebc5052dc0ab8a0ee123c3577918",
    ),
    "wd2_archive": (
        WD2_ARCHIVE,
        165_387,
        "e9c4a9ed5e6bef89d228ca877a9f9e37345e3c79dc07ba20087c218ff89fcf87",
    ),
    "fixed_pair_raw": (
        FIXED_PAIR_RAW,
        3_662_409_600,
        "7a065f110f0b8202f098cec9dc2267d6be7e99a179c911e404226d6a289f2c56",
    ),
    "gt_cache": (
        GT_CACHE,
        5_078_017_610,
        "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    ),
    "posenet": (
        POSENET,
        55_835_560,
        "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576",
    ),
    "segnet": (
        SEGNET,
        38_502_892,
        "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6",
    ),
}

ARM_SPECS = {
    "W0_warm": receiver.StudentSpec("flattened_d4_w64", "flattened", 64, 4),
    "W0_reset": receiver.StudentSpec("flattened_d4_w64", "flattened", 64, 4),
    "D56": receiver.StudentSpec("dense_d4_w56", "dense", 56, 4),
    "F64": receiver.StudentSpec("factorized_d4_w64_r19", "factorized", 64, 4, 19),
    "W96_factorized": receiver.StudentSpec("factorized_d4_w96_r20", "factorized", 96, 4, 20),
    "W96_flattened": receiver.StudentSpec("flattened_d4_w96", "flattened", 96, 4),
    "fresh": receiver.StudentSpec("fresh_flattened_d4_w64", "flattened", 64, 4),
}
ARM_ORDER = ("W0_warm", "W0_reset", "D56", "F64")


class WD3Error(RuntimeError):
    """WD3 refused an unclosed build, launch, measurement, or claim."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WD3Error(f"required file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def atomic_torch(path: Path, value: object) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


def storage_preflight(
    output: Path,
    minimum_free_bytes: int,
    *,
    allowed_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    resolved = output.resolve()
    root = allowed_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise WD3Error(f"WD3 bulk output must stay under {root}")
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    if free < minimum_free_bytes:
        raise WD3Error(f"storage preflight failed: {free} free bytes < {minimum_free_bytes} required")
    return {
        "schema": "ddm_wd3_storage_preflight.v1",
        "root": str(resolved),
        "free_bytes": free,
        "required_free_bytes": minimum_free_bytes,
        "cleanup": "certify-or-block; no retained payload is deleted or moved",
        "status": "PASS",
    }


def verify_pins(*, facts: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    records = {}
    for name, (path, size, digest) in PINS.items():
        record = dict(facts[name]) if facts is not None else file_record(path)
        if record.get("bytes") != size or record.get("sha256") != digest:
            raise WD3Error(f"pinned input changed: {name}")
        records[name] = record
    return records


def evenly_strided_indices(population: int = 600, count: int = 60) -> np.ndarray:
    if population < 1 or count < 1 or population % count:
        raise WD3Error("evenly-strided controller subset must divide the population")
    result = np.arange(0, population, population // count, dtype=np.int64)
    if result.size != count or np.array_equal(result, np.arange(count)):
        raise WD3Error("controller subset unexpectedly became a prefix")
    return result


def stratified_random_indices(strata: Sequence[int], *, count: int = 120, seed: int = SEED) -> np.ndarray:
    labels = np.asarray(strata)
    population = labels.size
    if count < 1 or count > population or population < 2:
        raise WD3Error("invalid stratified subset geometry")
    groups = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
    if any(indices.size == 0 for indices in groups.values()):
        raise WD3Error("empty stratum")
    raw = {label: count * indices.size / population for label, indices in groups.items()}
    quotas = {label: min(indices.size, math.floor(raw[label])) for label, indices in groups.items()}
    remaining = count - sum(quotas.values())
    for label in sorted(groups, key=lambda item: (-(raw[item] - quotas[item]), item)):
        if remaining == 0:
            break
        if quotas[label] < groups[label].size:
            quotas[label] += 1
            remaining -= 1
    if remaining:
        raise WD3Error("could not fill stratified subset")
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(groups[label], quotas[label], replace=False) for label in sorted(groups)])
    selected.sort()
    if selected.size != count or np.unique(selected).size != count:
        raise WD3Error("stratified subset is not unique and complete")
    if np.array_equal(selected, np.arange(count)):
        raise WD3Error("stratified subset became a forbidden prefix")
    return selected.astype(np.int64, copy=False)


def derive_negative_population_strata(original_argmax: np.ndarray) -> np.ndarray:
    """Joint temporal/class strata; derived from the real target population, not a prefix."""

    if original_argmax.shape != (receiver.N, 384, 512):
        raise WD3Error("negative-strata target population geometry differs")
    dominant = np.empty(receiver.N, dtype=np.int64)
    for pair in range(receiver.N):
        dominant[pair] = int(np.argmax(np.bincount(np.asarray(original_argmax[pair]).reshape(-1), minlength=5)))
    temporal_block = np.arange(receiver.N, dtype=np.int64) // 60
    return temporal_block * 5 + dominant


def _strict_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown or missing:
        raise WD3Error(f"{label} fields differ; missing={missing}, unknown={unknown}")


CONFIG_FIELDS = {
    "schema",
    "action",
    "output",
    "seed",
    "device",
    "chunk_pairs",
    "retain_all_payloads",
    "checkpoint_every_epochs",
    "minimum_free_bytes",
    "base_receipt",
    "teacher_cache_result",
    "resume_from",
    "resume_root",
    "arm",
    "completed_arms",
    "negative_confirmed_arms",
    "capacity_pressure_confirmed",
    "real_coder_override_dense_w96",
    "subsets",
    "objective",
    "optimizer",
    "epochs",
    "batch_pairs",
    "scorer_lane",
    "metal_lane",
    "launch_authorized",
    "r5_exit_verified",
    "source_pins",
    "expected_builder_sha256",
    "expected_receiver_sha256",
    "stage_a",
    "cheap_to_shrink",
}

STAGE_A_FIELDS = {
    "schema",
    "enabled",
    "base_archive",
    "base_archive_sha256",
    "base_runtime",
    "initializer",
    "initializer_sha256",
    "custody_receipt",
    "adapter_module",
    "adapter_sha256",
    "registered_seeds",
    "renderer_only_mutable",
    "untouched_sections",
}

CHEAP_TO_SHRINK_FIELDS = {
    "mode",
    "allocation_family",
    "uniform_bits",
    "rung_weights",
    "base_weight",
    "sampler_seed",
}


def _stage_a_binding(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = config.get("stage_a")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WD3Error("stage_a binding is not a mapping")
    _strict_fields(value, STAGE_A_FIELDS, "stage_a")
    if value["schema"] != "ddm_s1a_stage_a_binding.v1" or value["enabled"] is not True:
        raise WD3Error("stage_a binding schema/enabled flag differs")
    if tuple(map(int, value["registered_seeds"])) != STAGE_A_SEEDS:
        raise WD3Error("stage_a registered seeds differ")
    if value["renderer_only_mutable"] is not True:
        raise WD3Error("stage_a must seal renderer_only_mutable")
    if tuple(value["untouched_sections"]) != ("hpac", "carrier", "fixed_residual", "token_stream", "framing"):
        raise WD3Error("stage_a untouched-section contract differs")
    expected = {
        "base_archive": value["base_archive_sha256"],
        "initializer": value["initializer_sha256"],
        "adapter_module": value["adapter_sha256"],
    }
    for field, digest in expected.items():
        path = Path(value[field])
        if not path.is_file() or sha256_file(path) != digest:
            raise WD3Error(f"stage_a {field} custody differs")
    if not Path(value["base_runtime"]).is_dir():
        raise WD3Error("stage_a base runtime is absent")
    receipt = Path(value["custody_receipt"])
    if not receipt.is_file():
        raise WD3Error("stage_a custody re-proof receipt is absent")
    return value


def cheap_to_shrink_config(config: Mapping[str, Any]) -> ds1.CheapToShrinkConfig:
    value = config.get("cheap_to_shrink")
    if not isinstance(value, Mapping):
        raise WD3Error("cheap_to_shrink binding is absent")
    _strict_fields(value, CHEAP_TO_SHRINK_FIELDS, "cheap_to_shrink")
    try:
        return ds1.CheapToShrinkConfig(
            mode=str(value["mode"]),
            allocation_family=str(value["allocation_family"]),
            uniform_bits=tuple(map(int, value["uniform_bits"])),
            rung_weights=tuple(map(float, value["rung_weights"])),
            base_weight=float(value["base_weight"]),
            seed=int(value["sampler_seed"]),
        )
    except ds1.DS1Error as error:
        raise WD3Error(f"cheap_to_shrink config differs: {error}") from error


def _validate_arm(config: Mapping[str, Any]) -> None:
    arm = config["arm"]
    completed = tuple(config["completed_arms"])
    negatives = set(config["negative_confirmed_arms"])
    if arm is None:
        if config["action"] in {"train", "prepare_arm_birth"}:
            raise WD3Error("arm-bound config has no arm")
        return
    if arm not in ARM_SPECS:
        raise WD3Error(f"unknown WD3 arm: {arm}")
    if arm in ARM_ORDER:
        position = ARM_ORDER.index(arm)
        if tuple(completed) != ARM_ORDER[:position]:
            raise WD3Error("WD3 smaller-arm order is not sealed")
    elif arm.startswith("W96"):
        if tuple(completed[:4]) != ARM_ORDER or not config["capacity_pressure_confirmed"]:
            raise WD3Error("W96 requires all smaller arms and measured capacity pressure")
    elif arm == "fresh":
        if not {"W0_warm", "W0_reset"}.issubset(negatives):
            raise WD3Error("fresh birth requires seeded n120 failures for both W0 arms")
    if arm == "W96_dense" and not config["real_coder_override_dense_w96"]:
        raise WD3Error("dense W96 is projection-priced out without real-coder evidence")


def _verify_live_lane_claim(claim: Mapping[str, Any], *, resource: str) -> None:
    _strict_fields(claim, {"claimed", "claim_id", "agent", "platform"}, f"{resource} claim")
    if claim["claimed"] is not True or claim["agent"] != "MAIN":
        raise WD3Error(f"{resource} lane is not claimed by MAIN")
    allowed_platforms = {
        "scorer": {"macos-cpu", "local_macos_cpu", "macos-mps"},
        "metal": {"macos-mps", "local_macos_metal"},
    }
    if claim["platform"] not in allowed_platforms[resource]:
        raise WD3Error(f"{resource} lane platform differs")
    rows = []
    for line in LANE_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "timestamp_utc" in line or line.startswith("|---"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 8:
            continue
        timestamp, agent, lane_id, platform_name, _job, _eta, status, _notes = fields
        if lane_id == claim["claim_id"] and agent == claim["agent"]:
            rows.append((timestamp, platform_name, status))
    if not rows:
        raise WD3Error(f"{resource} lane claim is absent from the coordination ledger")
    timestamp, platform_name, status = rows[0]
    if platform_name != claim["platform"]:
        raise WD3Error(f"{resource} lane ledger platform differs")
    try:
        age = datetime.now(UTC) - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise WD3Error(f"{resource} lane timestamp is invalid") from error
    if age.total_seconds() < 0 or age.total_seconds() > 24 * 3600:
        raise WD3Error(f"{resource} lane claim is outside the 24-hour live window")
    status_lower = status.lower()
    if not any(word in status_lower for word in ("active", "dispatching", "training", "eval")):
        raise WD3Error(f"{resource} lane claim is not live: {status}")


def validate_compiled_config(
    config: Mapping[str, Any],
    *,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    """Validate G0--G5 without loading a scorer or model."""

    _strict_fields(config, CONFIG_FIELDS, "compiled config")
    exists = path_exists or Path.is_file
    if config["schema"] != SCHEMA or config["action"] not in {
        "prepare_arm_birth",
        "prepare_teacher_scorer_cache",
        "train",
    }:
        raise WD3Error("compiled schema/action differs")
    stage_a = _stage_a_binding(config)
    allowed_seeds = STAGE_A_SEEDS if stage_a is not None else (SEED,)
    if int(config["seed"]) not in allowed_seeds:
        raise WD3Error(f"WD3 seed differs from the sealed seeds {allowed_seeds}")
    cheap = cheap_to_shrink_config(config)
    if stage_a is None and not ds1.is_inert(cheap):
        raise WD3Error("cheap_to_shrink treatment is admitted only through the reviewed Stage-A binding")
    if not 1 <= int(config["chunk_pairs"]) <= MAX_CHUNK:
        raise WD3Error("chunk size exceeds 120 or is nonpositive")
    if config["retain_all_payloads"] is not True:
        raise WD3Error("non-retaining WD3 config is forbidden")
    if not 1 <= int(config["checkpoint_every_epochs"]) <= 5:
        raise WD3Error("checkpoint cadence must be at most five epochs")
    output = Path(config["output"]).resolve()
    allowed_output_root = STAGE_A_OUTPUT_ROOT if stage_a is not None else OUTPUT_ROOT
    if output != allowed_output_root.resolve() and allowed_output_root.resolve() not in output.parents:
        raise WD3Error("WD3 output is outside the sealed APDataStore consumer root")
    if Path(config["base_receipt"]) != BASE_RECEIPT:
        raise WD3Error("same-instrument base receipt path differs")
    pins = verify_pins(facts=facts)
    expected_source_pins = {name: {"bytes": row[1], "sha256": row[2]} for name, row in PINS.items()}
    if config["source_pins"] != expected_source_pins:
        raise WD3Error("compiled source pins differ")
    subsets = config["subsets"]
    _strict_fields(
        subsets,
        {"controller_n60", "negative_n120", "controller_kind", "negative_kind", "prefix"},
        "subset",
    )
    if subsets["prefix"] is not False:
        raise WD3Error("prefix subset is forbidden")
    if subsets["controller_kind"] != "evenly_strided" or len(subsets["controller_n60"]) != 60:
        raise WD3Error("controller subset must be fixed evenly-strided n60")
    if list(map(int, subsets["controller_n60"])) != evenly_strided_indices().tolist():
        raise WD3Error("controller n60 IDs differ from the sealed strided set")
    negative = np.asarray(subsets["negative_n120"], dtype=np.int64)
    if (
        subsets["negative_kind"] != "seeded_stratified_random"
        or negative.size != 120
        or np.unique(negative).size != 120
        or np.array_equal(np.sort(negative), np.arange(120))
    ):
        raise WD3Error("negative confirmation must be seeded stratified nonprefix n120")
    objective = config["objective"]
    _strict_fields(
        objective,
        {
            "scoreaware",
            "seg_score_coefficient",
            "pose_exact_nonlinear",
            "temperature",
            "adaptive_duals",
            "decode_mse_ceiling",
            "packet_quantizer_in_loop",
        },
        "objective",
    )
    if (
        objective["scoreaware"] is not True
        or float(objective["seg_score_coefficient"]) != 100.0
        or objective["pose_exact_nonlinear"] is not True
        or float(objective["temperature"]) != 2.0
        or objective["adaptive_duals"] is not True
        or float(objective["decode_mse_ceiling"]) != DECODE_MSE_CEILING
        or objective["packet_quantizer_in_loop"] is not True
    ):
        raise WD3Error("score-aware objective is incomplete or inactive")
    _validate_arm(config)
    if not config["resume_root"] or not Path(config["resume_root"]).is_absolute():
        raise WD3Error("durable cache/run resume root is absent")
    if config["action"] == "train":
        if not config["teacher_cache_result"] or not exists(Path(config["teacher_cache_result"])):
            raise WD3Error("complete teacher/scorer cache receipt is absent")
        if not config["resume_from"] or not exists(Path(config["resume_from"])):
            raise WD3Error("training resume checkpoint/state is absent")
        if facts is None:
            cache_receipt = json.loads(Path(config["teacher_cache_result"]).read_text(encoding="utf-8"))
            if (
                cache_receipt.get("schema") != CACHE_SCHEMA
                or cache_receipt.get("complete") is not True
                or list(map(int, subsets["negative_n120"])) != list(map(int, cache_receipt.get("negative_n120", [])))
            ):
                raise WD3Error("training n120 is not the cache-derived stratified population subset")
    for resource in ("scorer", "metal"):
        _strict_fields(
            config[f"{resource}_lane"],
            {"claimed", "claim_id", "agent", "platform"},
            f"{resource} claim",
        )
    if config["action"] == "prepare_arm_birth":
        if config["device"] != "cpu":
            raise WD3Error("arm-birth materialization is scorer-free CPU only")
        if (
            config["launch_authorized"] is not False
            or config["r5_exit_verified"] is not False
            or config["scorer_lane"].get("claimed") is not False
            or config["metal_lane"].get("claimed") is not False
        ):
            raise WD3Error("arm-birth materialization cannot carry launch authority")
        return {"status": "PASS", "pins": pins, "config_sha256": canonical_sha256(config)}
    if not config["scorer_lane"].get("claimed") or not config["scorer_lane"].get("claim_id"):
        raise WD3Error("global scorer lane is unclaimed")
    if not config["metal_lane"].get("claimed") or not config["metal_lane"].get("claim_id"):
        raise WD3Error("global Metal lane is unclaimed")
    if config["scorer_lane"]["claim_id"] == config["metal_lane"]["claim_id"]:
        raise WD3Error("scorer and Metal lane claims must be distinct")
    if config["launch_authorized"] is not True:
        raise WD3Error("charter launch authorization remains false")
    if config["r5_exit_verified"] is not True:
        raise WD3Error("r5 PID 63183 exit is not verified")
    if facts is None:
        try:
            os.kill(63183, 0)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            raise WD3Error("cannot verify r5 PID 63183 exit") from error
        else:
            raise WD3Error("r5 PID 63183 is still alive")
        _verify_live_lane_claim(config["scorer_lane"], resource="scorer")
        _verify_live_lane_claim(config["metal_lane"], resource="metal")
    return {"status": "PASS", "pins": pins, "config_sha256": canonical_sha256(config)}


def compile_fire_order(
    config: Mapping[str, Any],
    *,
    facts: Mapping[str, Mapping[str, Any]] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    blockers = []
    try:
        validation = validate_compiled_config(config, facts=facts, path_exists=path_exists)
    except (WD3Error, KeyError, TypeError, ValueError) as error:
        validation = None
        blockers.append(str(error))
    if config.get("action") != "prepare_arm_birth":
        declared_blockers = []
        if config.get("action") == "train" and not config.get("teacher_cache_result"):
            declared_blockers.append("complete teacher/scorer cache receipt is absent")
        for resource in ("scorer", "metal"):
            claim = config.get(f"{resource}_lane", {})
            if not isinstance(claim, Mapping) or not claim.get("claimed") or not claim.get("claim_id"):
                declared_blockers.append(f"global {resource} lane is unclaimed")
        if config.get("launch_authorized") is not True:
            declared_blockers.append("charter launch authorization remains false")
        if config.get("r5_exit_verified") is not True:
            declared_blockers.append("r5 PID 63183 exit is not verified in the compiled config")
        for blocker in declared_blockers:
            if blocker not in blockers:
                blockers.append(blocker)
    ready = not blockers
    disposition = (
        "READY_TO_MATERIALIZE_BUILD"
        if ready and config.get("action") == "prepare_arm_birth"
        else "READY_TO_FIRE"
        if ready
        else "BLOCKED_NOT_LAUNCHABLE"
    )
    consumer_root = STAGE_A_OUTPUT_ROOT if config.get("stage_a") is not None else OUTPUT_ROOT
    return {
        "schema": "ddm_wd3_fire_order.v1",
        "disposition": disposition,
        "owner": "MAIN",
        "consumer_store": str(consumer_root) + "/",
        "fire_trigger": (
            "G0-G5 pass; r5 exited; distinct scorer and Metal lanes claimed; reviewed WD3 code/config dry-run landed"
        ),
        "config_sha256": canonical_sha256(config),
        "validation": validation,
        "blockers": blockers,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "training_launched": False,
    }


def boundary_band(labels: torch.Tensor) -> torch.Tensor:
    """One-cell codimension-1 band of the actual argmax cell complex."""

    if labels.ndim != 3:
        raise WD3Error("cell labels must have shape [B,H,W]")
    band = torch.zeros_like(labels, dtype=torch.bool)
    vertical = labels[:, 1:] != labels[:, :-1]
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    band[:, 1:] |= vertical
    band[:, :-1] |= vertical
    band[:, :, 1:] |= horizontal
    band[:, :, :-1] |= horizontal
    return band


def derive_selective_cell_mask(
    student_argmax: torch.Tensor,
    teacher_argmax: torch.Tensor,
    original_argmax: torch.Tensor,
) -> torch.Tensor:
    """Select measured mismatches plus the original/teacher boundary annulus."""

    if not (student_argmax.shape == teacher_argmax.shape == original_argmax.shape and student_argmax.ndim == 3):
        raise WD3Error("student/teacher/original cell maps differ")
    mismatch = (student_argmax != teacher_argmax) | (student_argmax != original_argmax)
    return mismatch | boundary_band(teacher_argmax) | boundary_band(original_argmax)


CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")


def _edge_name(left: int, right: int) -> str:
    if not 0 <= left < len(CLASS_NAMES) or not 0 <= right < len(CLASS_NAMES):
        raise WD3Error("class id outside the five-class scorer vocabulary")
    a, b = sorted((CLASS_NAMES[left], CLASS_NAMES[right]))
    return f"{a}<->{b}"


def cell_edge_telemetry(
    student_argmax: torch.Tensor,
    original_argmax: torch.Tensor,
    *,
    selected: torch.Tensor | None = None,
) -> dict[str, Any]:
    if student_argmax.shape != original_argmax.shape or student_argmax.ndim != 3:
        raise WD3Error("edge telemetry maps differ")
    active = student_argmax != original_argmax
    if selected is not None:
        if selected.shape != active.shape:
            raise WD3Error("edge telemetry selection shape differs")
        active &= selected.bool()
    student = student_argmax.detach().cpu().numpy()
    target = original_argmax.detach().cpu().numpy()
    mask = active.detach().cpu().numpy()
    edge_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    pair_counts = []
    for pair in range(student.shape[0]):
        rows, columns = np.nonzero(mask[pair])
        pair_counts.append(int(rows.size))
        for row, column in zip(rows, columns, strict=True):
            expected = int(target[pair, row, column])
            observed = int(student[pair, row, column])
            edge_counts[_edge_name(expected, observed)] += 1
            target_counts[CLASS_NAMES[expected]] += 1
    return {
        "schema": "ddm_wd3_cell_edge_telemetry.v1",
        "hard_flip_count": int(mask.sum()),
        "sites": int(mask.size),
        "per_edge_flips": dict(sorted(edge_counts.items())),
        "road_lane_flips": int(edge_counts.get("Lane<->Road", 0)),
        "per_target_cell_flips": dict(sorted(target_counts.items())),
        "per_pair_flips": pair_counts,
    }


@dataclass
class DualState:
    margin: float = 0.0
    teacher_kl: float = 0.0
    decode: float = 0.0
    teacher_pose: float = 0.0

    def __post_init__(self) -> None:
        if min(self.margin, self.teacher_kl, self.decode, self.teacher_pose) < 0:
            raise WD3Error("WD3 duals must be nonnegative")

    def update(
        self,
        *,
        margin_violation: float,
        teacher_kl_violation: float,
        decode_violation: float,
        teacher_pose_violation: float,
        step_size: float,
    ) -> DualState:
        if step_size < 0:
            raise WD3Error("dual step size must be nonnegative")
        return DualState(
            margin=max(0.0, self.margin + step_size * max(0.0, margin_violation)),
            teacher_kl=max(
                0.0,
                self.teacher_kl + step_size * max(0.0, teacher_kl_violation),
            ),
            decode=max(0.0, self.decode + step_size * max(0.0, decode_violation)),
            teacher_pose=max(
                0.0,
                self.teacher_pose + step_size * max(0.0, teacher_pose_violation),
            ),
        )


@dataclass(frozen=True)
class StageThresholds:
    calibration_scale: float
    margin_ceiling: float
    teacher_kl_ceiling: float
    decode_ceiling: float = DECODE_MSE_CEILING

    def __post_init__(self) -> None:
        if self.calibration_scale < 0 or min(self.margin_ceiling, self.teacher_kl_ceiling, self.decode_ceiling) < 0:
            raise WD3Error("stage thresholds must be nonnegative")


def calibrate_soft_disagreement(student_logits: torch.Tensor, original_argmax: torch.Tensor) -> dict[str, float]:
    if student_logits.ndim != 4 or original_argmax.shape != (
        student_logits.shape[0],
        student_logits.shape[2],
        student_logits.shape[3],
    ):
        raise WD3Error("calibration logits/targets differ")
    probabilities = student_logits.softmax(dim=1)
    target_probability = probabilities.gather(1, original_argmax[:, None].long()).squeeze(1)
    soft = (1.0 - target_probability).mean()
    hard = (student_logits.argmax(dim=1) != original_argmax).float().mean()
    soft_value = float(soft.detach().cpu())
    hard_value = float(hard.detach().cpu())
    if soft_value <= 0:
        if hard_value > 0:
            raise WD3Error("positive hard d_seg with zero soft disagreement")
        scale = 0.0
    else:
        scale = hard_value / soft_value
    return {
        "hard_d_seg": hard_value,
        "mean_soft_disagreement": soft_value,
        "stage_frozen_calibration_scale": scale,
    }


def score_native_objective(
    *,
    student_logits: torch.Tensor,
    student_pose6: torch.Tensor,
    student_frame1: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_argmax: torch.Tensor,
    teacher_margin: torch.Tensor,
    teacher_pose6: torch.Tensor,
    original_argmax: torch.Tensor,
    original_pose6: torch.Tensor,
    teacher_frame1: torch.Tensor,
    selected_cells: torch.Tensor,
    thresholds: StageThresholds,
    duals: DualState,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact WD3 train objective; hard components remain selection authority."""

    if temperature != 2.0:
        raise WD3Error("WD3 teacher KL temperature is sealed at 2")
    expected = (student_logits.shape[0], student_logits.shape[2], student_logits.shape[3])
    for name, value in {
        "teacher_argmax": teacher_argmax,
        "teacher_margin": teacher_margin,
        "original_argmax": original_argmax,
        "selected_cells": selected_cells,
    }.items():
        if value.shape != expected:
            raise WD3Error(f"{name} shape differs from scorer logits")
    selected = selected_cells.bool()
    if not torch.any(selected):
        raise WD3Error("selective objective has zero measured cells")
    probabilities = student_logits.softmax(dim=1)
    target_probability = probabilities.gather(1, original_argmax[:, None].long()).squeeze(1)
    soft_disagreement = (1.0 - target_probability)[selected].mean()
    calibrated_seg = 100.0 * thresholds.calibration_scale * soft_disagreement

    winner = student_logits.gather(1, teacher_argmax[:, None].long()).squeeze(1)
    competitor_differences = winner[:, None] - student_logits
    required = teacher_margin[:, None]
    impostor_hinge = F.relu(required - competitor_differences)
    impostor_hinge = impostor_hinge.scatter(
        1, teacher_argmax[:, None].long(), torch.zeros_like(teacher_argmax[:, None], dtype=impostor_hinge.dtype)
    )
    margin_loss = impostor_hinge.permute(0, 2, 3, 1)[selected].mean()

    teacher_probability = (teacher_logits / temperature).softmax(dim=1)
    student_log_probability = (student_logits / temperature).log_softmax(dim=1)
    teacher_log_probability = (teacher_logits / temperature).log_softmax(dim=1)
    kl_map = (teacher_probability * (teacher_log_probability - student_log_probability)).sum(dim=1) * (temperature**2)
    teacher_kl = kl_map[selected].mean()

    pose_mse = (student_pose6 - original_pose6).square().mean()
    pose_score = torch.sqrt(torch.clamp(10.0 * pose_mse, min=1e-20))
    teacher_pose_mse = (student_pose6 - teacher_pose6).square().mean()
    decode_mse = (student_frame1 - teacher_frame1).square().mean()
    margin_violation = F.relu(margin_loss - thresholds.margin_ceiling)
    kl_violation = F.relu(teacher_kl - thresholds.teacher_kl_ceiling)
    decode_violation = F.relu(decode_mse - thresholds.decode_ceiling)
    total = (
        calibrated_seg
        + pose_score
        + duals.margin * margin_violation
        + duals.teacher_kl * kl_violation
        + duals.decode * decode_violation
        + duals.teacher_pose * teacher_pose_mse
    )
    components = {
        "seg_axis_train_loss_proxy": soft_disagreement,
        "seg_axis_stage_calibrated_score_proxy": calibrated_seg,
        "teacher_impostor_complete_margin_hinge_loss": margin_loss,
        "teacher_t2_kl_loss": teacher_kl,
        "pose_original_target_mse_train_quantity": pose_mse,
        "pose_exact_nonlinear_score_train_quantity": pose_score,
        "pose_teacher_first6_mse_preservation_telemetry": teacher_pose_mse,
        "pose_teacher_first6_adaptive_constraint_violation": teacher_pose_mse,
        "decode_mse_uint8_train_quantity": decode_mse,
        "margin_constraint_violation": margin_violation,
        "teacher_kl_constraint_violation": kl_violation,
        "decode_constraint_violation": decode_violation,
    }
    return total, components


def paired_receiver_tensor(
    *,
    model: receiver.StudentSemanticRenderer,
    allocation: receiver.AdaptiveQuantizationAllocation,
    tokens: torch.Tensor,
    pair_indices: torch.Tensor,
    fixed_frame0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real WD3 pair: fixed retained carrier frame 0 plus quantized student frame 1."""

    state = receiver.fake_quantize_state(model, allocation)
    frame1_eval = functional_call(model, state, (tokens.long(), pair_indices))
    frame1_camera = F.interpolate(
        frame1_eval,
        size=(receiver.CAMERA_H, receiver.CAMERA_W),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 255.0)
    rounded = frame1_camera.round().clamp(0.0, 255.0)
    frame1_camera = frame1_camera + (rounded - frame1_camera).detach()
    if fixed_frame0.shape != frame1_camera.shape:
        raise WD3Error("fixed frame-0 geometry differs from student frame 1")
    pair = torch.stack((fixed_frame0.to(frame1_camera), frame1_camera), dim=1)
    return pair, frame1_camera


def scorer_forward(pair: torch.Tensor, posenet: nn.Module, segnet: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Run both actual scorer graphs and return Pose6 plus five-class logits."""

    if pair.ndim != 5 or pair.shape[1:3] != (2, 3):
        raise WD3Error("scorer pair must have shape [B,2,3,H,W]")
    pose_output = posenet(posenet.preprocess_input(pair))
    if not isinstance(pose_output, Mapping) or "pose" not in pose_output:
        raise WD3Error("PoseNet output lacks the official pose head")
    pose6 = pose_output["pose"][..., :6]
    logits = segnet(segnet.preprocess_input(pair))
    if logits.ndim != 4 or logits.shape[1] != 5 or pose6.shape != (pair.shape[0], 6):
        raise WD3Error("frozen scorer output geometry differs")
    return pose6, logits


def quantization_sensitivity_table(
    model: receiver.StudentSemanticRenderer,
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, list[dict[str, float]]]:
    """First-order squared score effect for every parameter group and bit rung."""

    table: dict[str, list[dict[str, float]]] = {}
    for name, value in model.named_parameters():
        if value.ndim < 2:
            continue
        if name not in gradients or gradients[name].shape != value.shape:
            raise WD3Error(f"quantization sensitivity lacks gradient for {name}")
        axis = value.ndim - 1 if name.endswith("embed.weight") else 0
        source = value.detach().cpu().float().movedim(axis, 0).contiguous()
        gradient = gradients[name].detach().cpu().float().movedim(axis, 0).contiguous()
        rows = []
        for group in range(source.shape[0]):
            errors = {}
            byte_costs = {}
            for bit in range(2, 9):
                quantized = receiver.quantize_tensor_groups(source[group : group + 1], axis=0, bits=(bit,))[0]
                errors[str(bit)] = float(((source[group] - quantized) * gradient[group]).square().sum())
                byte_costs[str(bit)] = 2 + (source[group].numel() * bit + 7) // 8
            rows.append(
                {
                    "group": group,
                    "elements": source[group].numel(),
                    "errors": errors,
                    "bytes": byte_costs,
                }
            )
        table[name] = rows
    return table


def adaptive_allocation_from_sensitivity(
    model: receiver.StudentSemanticRenderer,
    sensitivity: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    maximum_predicted_error: float,
    selection_sha256: str,
) -> receiver.AdaptiveQuantizationAllocation:
    """Discrete waterfill: cheapest group depths satisfying a global error ceiling."""

    if maximum_predicted_error < 0:
        raise WD3Error("adaptive quantization error budget is negative")
    depths: dict[str, list[int]] = {}
    rows_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for name, value in model.state_dict().items():
        if value.ndim < 2:
            continue
        rows = sensitivity.get(name)
        group_count = value.shape[value.ndim - 1 if name.endswith("embed.weight") else 0]
        if rows is None or len(rows) != group_count:
            raise WD3Error(f"sensitivity group count differs for {name}")
        depths[name] = [2] * group_count
        for index, row in enumerate(rows):
            if int(row["group"]) != index or set(row["errors"]) != {str(bit) for bit in range(2, 9)}:
                raise WD3Error(f"sensitivity rung schema differs for {name}/{index}")
            rows_by_key[(name, index)] = row

    def total_error() -> float:
        return sum(
            float(rows_by_key[(name, index)]["errors"][str(bit)])
            for name, values in depths.items()
            for index, bit in enumerate(values)
        )

    while total_error() > maximum_predicted_error:
        choices = []
        took_free_upgrade = False
        for name, values in depths.items():
            for index, bit in enumerate(values):
                if bit == 8:
                    continue
                row = rows_by_key[(name, index)]
                saving = float(row["errors"][str(bit)]) - float(row["errors"][str(bit + 1)])
                extra_bytes = int(row["bytes"][str(bit + 1)]) - int(row["bytes"][str(bit)])
                if extra_bytes <= 0:
                    # Real coder measurements are not strictly monotone in bit depth: a
                    # higher-precision rung can compress to the same or fewer bytes. When
                    # it also reduces error it strictly dominates — take it for free;
                    # when it saves nothing it buys nothing — skip it as a candidate.
                    if saving > 0.0:
                        depths[name][index] += 1
                        took_free_upgrade = True
                    continue
                choices.append((saving / extra_bytes, saving, name, index))
        if took_free_upgrade:
            continue
        best = max(choices, key=lambda row: (row[0], row[1], row[2], -row[3])) if choices else None
        if best is None or best[1] <= 0.0:
            # Non-monotone real-coder error rows can strand the climb greedy above the
            # ceiling even though the uniform-4 reference state meets it by construction
            # (the ceiling IS the uniform-4 error sum).  While total > ceiling, some group
            # must sit above its own bit-4 reference error; snap the worst such group to
            # bit 4 — total error strictly falls toward the feasible all-4 state, so the
            # loop terminates (every accepted operation strictly reduces total error).
            snap: tuple[float, str, int] | None = None
            for name, values in depths.items():
                for index, bit in enumerate(values):
                    if bit == 4:
                        continue
                    row = rows_by_key[(name, index)]
                    excess = float(row["errors"][str(bit)]) - float(row["errors"]["4"])
                    if excess > 0.0 and (snap is None or excess > snap[0]):
                        snap = (excess, name, index)
            if snap is None:
                raise WD3Error("even int8 allocation cannot meet the predicted-error ceiling")
            depths[snap[1]][snap[2]] = 4
            continue
        _, saving, name, index = best
        depths[name][index] += 1
    allocation = receiver.AdaptiveQuantizationAllocation(
        bits={name: tuple(values) for name, values in depths.items()},
        selection_sha256=selection_sha256,
        policy="gradient_score_effect_discrete_waterfill_subint16",
    )
    allocation.validate(model)
    return allocation


def choose_cheapest_passing_quantization(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Naive re-score authority: select only a retained, parse-backed passing rung."""

    passing = []
    for row in rows:
        required = {
            "allocation_id",
            "packet_bytes",
            "hard_cell_gate_pass",
            "road_lane_gate_pass",
            "pose_gate_pass",
            "parse_back_exact",
            "retained_payload",
            "measured",
        }
        if not required.issubset(row):
            raise WD3Error("quantization re-score row is incomplete")
        if row["measured"] is not True:
            raise WD3Error("projected quantization row cannot select an allocation")
        if all(
            row[field] is True
            for field in (
                "hard_cell_gate_pass",
                "road_lane_gate_pass",
                "pose_gate_pass",
                "parse_back_exact",
                "retained_payload",
            )
        ):
            passing.append(row)
    if not passing:
        raise WD3Error("no naively re-scored quantization rung passes")
    return min(passing, key=lambda row: (int(row["packet_bytes"]), str(row["allocation_id"])))


class WD3ResumeController:
    """Additive ResumeRegistry state for the WD3 stage/controller seam."""

    def __init__(
        self,
        *,
        duals: DualState,
        thresholds: StageThresholds,
        epoch: int,
        batch_cursor: int,
        selection_sha256: str,
        allocation_sha256: str,
    ) -> None:
        self.duals = duals
        self.thresholds = thresholds
        self.epoch = int(epoch)
        self.batch_cursor = int(batch_cursor)
        self.selection_sha256 = selection_sha256
        self.allocation_sha256 = allocation_sha256

    def state_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_scalars": np.asarray(
                [
                    self.duals.margin,
                    self.duals.teacher_kl,
                    self.duals.decode,
                    self.duals.teacher_pose,
                    self.thresholds.calibration_scale,
                    self.thresholds.margin_ceiling,
                    self.thresholds.teacher_kl_ceiling,
                    self.thresholds.decode_ceiling,
                ],
                dtype=np.float64,
            ),
            f"{prefix}_cursor": np.asarray([self.epoch, self.batch_cursor], dtype=np.int64),
            f"{prefix}_selection_sha256": np.frombuffer(bytes.fromhex(self.selection_sha256), dtype=np.uint8).copy(),
            f"{prefix}_allocation_sha256": np.frombuffer(bytes.fromhex(self.allocation_sha256), dtype=np.uint8).copy(),
        }

    def restore_from_cfg(self, prefix: str, cfg: Mapping[str, Any]) -> bool:
        scalars = np.asarray(cfg[f"{prefix}_scalars"], dtype=np.float64)
        cursor = np.asarray(cfg[f"{prefix}_cursor"], dtype=np.int64)
        if scalars.shape != (8,) or cursor.shape != (2,):
            raise WD3Error("resume-controller state geometry differs")
        self.duals = DualState(*map(float, scalars[:4]))
        self.thresholds = StageThresholds(*map(float, scalars[4:]))
        self.epoch, self.batch_cursor = map(int, cursor)
        self.selection_sha256 = bytes(np.asarray(cfg[f"{prefix}_selection_sha256"], dtype=np.uint8)).hex()
        self.allocation_sha256 = bytes(np.asarray(cfg[f"{prefix}_allocation_sha256"], dtype=np.uint8)).hex()
        return True


def register_resume_controller(registry: ResumeRegistry, controller: WD3ResumeController) -> None:
    registry.register("wd3_scorer_controller", "__wd3_", controller)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "generator": generator.get_state(),
    }


def _restore_rng(payload: Mapping[str, Any], generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch_cpu"])
    generator.set_state(payload["generator"])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: wd2_build.DeploymentEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    controller: WD3ResumeController,
    allocation: receiver.AdaptiveQuantizationAllocation,
    selection_record: Mapping[str, Any],
    subset_ids: Mapping[str, Sequence[int]],
    config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    stage: str,
) -> dict[str, Any]:
    registry = ResumeRegistry()
    register_resume_controller(registry, controller)
    registry_state = registry.state_arrays()
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": stage,
        "live_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "ema": ema.state(),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "scaler": {"enabled": False, "state_dict": {}},
        "rng": _rng_state(generator),
        "resume_registry": registry_state,
        "allocation": allocation.as_dict(),
        "allocation_sha256": canonical_sha256(allocation.as_dict()),
        "selection": dict(selection_record),
        "subset_ids": {key: list(map(int, value)) for key, value in subset_ids.items()},
        "subset_sha256": canonical_sha256(subset_ids),
        "config": dict(config),
        "config_sha256": canonical_sha256(config),
        "history": list(history),
        "atomic": True,
        "deployment_weights": "ema_shadow",
    }
    return atomic_torch(path, payload)


def _resume_config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Config identity for crash-resume comparison, self-referential fields masked.

    Two fields cannot be known by a checkpoint ahead of time and so make
    full-dict equality unsatisfiable for a legitimate crash resume (surfaced
    by the 2026-08-25 s1a seed-2 external SIGKILL — the first real crash on
    this trainer): `resume_from` names the checkpoint itself, and
    `expected_builder_sha256` names the trainer code doing the resuming —
    which necessarily differs when the crash's cure edited this file. Source
    custody is NOT weakened: `_verify_launch_sources` still refuses unless
    the LIVE trainer matches the continuation config's pin, so a pin update
    is an explicit, committed act. Receiver/adapter pins and every other
    field stay strict.
    """
    masked = dict(config)
    masked.pop("resume_from", None)
    masked.pop("expected_builder_sha256", None)
    return masked


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: wd2_build.DeploymentEMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    expected_config: Mapping[str, Any],
) -> tuple[dict[str, Any], WD3ResumeController, receiver.AdaptiveQuantizationAllocation]:
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if not (magic.startswith(b"PK\x03\x04") or magic[:1] == b"\x80"):
        raise WD3Error(
            f"resume checkpoint {path} is not a PyTorch pickle/zip "
            f"(magic {magic!r}); refusing torch.load"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise WD3Error("resume checkpoint schema differs")
    stored_config = payload.get("config")
    if not isinstance(stored_config, Mapping) or _resume_config_identity(
        stored_config
    ) != _resume_config_identity(expected_config):
        raise WD3Error("resume checkpoint config/source identity differs")
    if payload.get("scaler") != {"enabled": False, "state_dict": {}}:
        raise WD3Error("resume scaler state differs")
    model.load_state_dict(payload["live_state_dict"], strict=True)
    ema.restore(payload["ema"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_rng(payload["rng"], generator)
    allocation = receiver.AdaptiveQuantizationAllocation.from_dict(payload["allocation"])
    allocation.validate(model)
    if canonical_sha256(allocation.as_dict()) != payload["allocation_sha256"]:
        raise WD3Error("resume allocation identity differs")
    placeholder = WD3ResumeController(
        duals=DualState(),
        thresholds=StageThresholds(0.0, 0.0, 0.0),
        epoch=0,
        batch_cursor=0,
        selection_sha256="0" * 64,
        allocation_sha256="0" * 64,
    )
    registry = ResumeRegistry()
    register_resume_controller(registry, placeholder)
    registry.restore(payload["resume_registry"])
    if placeholder.allocation_sha256 != payload["allocation_sha256"]:
        raise WD3Error("resume controller/allocation binding differs")
    return payload, placeholder, allocation


def _device(name: str) -> torch.device:
    if name not in {"cpu", "mps", "cuda"}:
        raise WD3Error("WD3 device must be cpu, mps, or cuda")
    device = torch.device(name)
    if name == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise WD3Error("MPS fallback must be disabled")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise WD3Error("MPS is unavailable; CPU substitution is forbidden")
    if name == "cuda" and not torch.cuda.is_available():
        raise WD3Error("CUDA is unavailable; device substitution is forbidden")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _fixed_frame0_memmap() -> np.memmap:
    return np.memmap(
        FIXED_PAIR_RAW,
        mode="r",
        dtype=np.uint8,
        shape=(receiver.N * 2, receiver.CAMERA_H, receiver.CAMERA_W, 3),
    )


def _teacher_master_memmap() -> np.memmap:
    return np.memmap(
        TEACHER_MASTER,
        mode="r",
        dtype=np.uint8,
        shape=(receiver.N, 3, receiver.CAMERA_H, receiver.CAMERA_W),
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_record(path)


CACHE_FIELDS: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "teacher_segnet_logits_f16": (np.dtype("<f2"), (receiver.N, 5, 384, 512)),
    "teacher_segnet_argmax_u8": (np.dtype("u1"), (receiver.N, 384, 512)),
    "teacher_top1_runnerup_margin_f16": (np.dtype("<f2"), (receiver.N, 384, 512)),
    "teacher_posenet_first6_f32": (np.dtype("<f4"), (receiver.N, 6)),
    "original_gt_segnet_argmax_u8": (np.dtype("u1"), (receiver.N, 384, 512)),
    "original_gt_posenet_first6_f32": (np.dtype("<f4"), (receiver.N, 6)),
}


def _validate_cache_chunk(
    path: Path,
    start: int,
    end: int,
    *,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(CACHE_FIELDS):
            raise WD3Error(f"teacher/scorer chunk fields differ: {path}")
        for name, (dtype, shape) in CACHE_FIELDS.items():
            expected = (end - start, *shape[1:])
            if payload[name].shape != expected or payload[name].dtype != dtype:
                raise WD3Error(f"teacher/scorer chunk geometry differs: {path}/{name}")
    record = file_record(path)
    if expected_binding is not None:
        receipt_path = path.with_suffix(".json")
        if not receipt_path.is_file():
            raise WD3Error(f"teacher/scorer chunk receipt is absent: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "ddm_wd3_teacher_scorer_cache_chunk.v1"
            or receipt.get("start_pair") != start
            or receipt.get("end_pair") != end
            or receipt.get("cache_binding") != dict(expected_binding)
            or receipt.get("payload") != record
            or receipt.get("complete") is not True
        ):
            raise WD3Error(f"teacher/scorer chunk binding differs: {path}")
    return record


def _aggregate_cache_repeat(
    repeat_root: Path,
    *,
    chunk_pairs: int,
    expected_binding: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    aggregate = repeat_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    maps: dict[str, np.memmap] = {}
    temporary_paths = {}
    for name, (dtype, shape) in CACHE_FIELDS.items():
        final = aggregate / f"{name}.npy"
        if final.exists():
            array = np.load(final, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise WD3Error(f"existing cache aggregate differs: {final}")
            continue
        temporary = aggregate / f".{name}.in_progress.npy"
        if temporary.is_file():
            existing = np.load(temporary, mmap_mode="r+", allow_pickle=False)
            if existing.shape != shape or existing.dtype != dtype:
                raise WD3Error(f"cache aggregate resume geometry differs: {temporary}")
            maps[name] = existing
        else:
            maps[name] = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        temporary_paths[name] = temporary
    if maps:
        for start in range(0, receiver.N, chunk_pairs):
            end = min(receiver.N, start + chunk_pairs)
            chunk = repeat_root / "chunks" / f"pairs_{start:04d}_{end:04d}.npz"
            _validate_cache_chunk(chunk, start, end, expected_binding=expected_binding)
            with np.load(chunk, allow_pickle=False) as payload:
                for name, mapped in maps.items():
                    mapped[start:end] = payload[name]
                    mapped.flush()
        for name, mapped in maps.items():
            del mapped
            os.replace(temporary_paths[name], aggregate / f"{name}.npy")
    return {name: file_record(aggregate / f"{name}.npy") for name in CACHE_FIELDS}


def prepare_teacher_scorer_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize both deterministic repeats of all charter-required scorer fields."""

    validation = validate_compiled_config(config)
    if config["action"] != "prepare_teacher_scorer_cache":
        raise WD3Error("cache builder received a non-cache config")
    assert_governed_admission("ddm_wd3_prepare_teacher_scorer_cache")
    stage_a = _stage_a_binding(config)
    storage = storage_preflight(
        Path(config["output"]),
        int(config["minimum_free_bytes"]),
        allowed_root=STAGE_A_OUTPUT_ROOT if stage_a is not None else OUTPUT_ROOT,
    )
    seed_everything(int(config["seed"]))
    device = _device(str(config["device"]))
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    fixed = _fixed_frame0_memmap()
    teacher = _teacher_master_memmap()
    cache_root = Path(config["resume_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    chunk_pairs = int(config["chunk_pairs"])
    cache_binding = {
        "schema": "ddm_wd3_teacher_scorer_cache_binding.v1",
        "config_sha256": validation["config_sha256"],
        "source_pins": config["source_pins"],
        "chunk_pairs": chunk_pairs,
        "seed": int(config["seed"]),
    }
    binding_path = cache_root / "CACHE_BINDING.json"
    if binding_path.is_file():
        if json.loads(binding_path.read_text(encoding="utf-8")) != cache_binding:
            raise WD3Error("teacher/scorer cache resume binding differs")
    else:
        atomic_json(binding_path, cache_binding)
    with np.load(GT_CACHE, allow_pickle=False) as gt:
        if gt["lstars"].shape != (receiver.N, 384, 512) or gt["gt_poses"].shape != (
            receiver.N,
            6,
        ):
            raise WD3Error("original scorer target cache geometry differs")
        for repeat in range(2):
            repeat_root = cache_root / f"repeat_{repeat}"
            for start in range(0, receiver.N, chunk_pairs):
                end = min(receiver.N, start + chunk_pairs)
                path = repeat_root / "chunks" / f"pairs_{start:04d}_{end:04d}.npz"
                if path.is_file():
                    _validate_cache_chunk(path, start, end, expected_binding=cache_binding)
                    continue
                frame0 = torch.from_numpy(np.asarray(fixed[2 * np.arange(start, end)]).copy()).permute(0, 3, 1, 2)
                frame1 = torch.from_numpy(np.asarray(teacher[start:end]).copy())
                pair = torch.stack((frame0, frame1), dim=1).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    pose6, logits = scorer_forward(pair, posenet, segnet)
                    top2 = logits.topk(k=2, dim=1).values
                    margin = top2[:, 0] - top2[:, 1]
                    arrays = {
                        "teacher_segnet_logits_f16": logits.cpu().numpy().astype("<f2"),
                        "teacher_segnet_argmax_u8": logits.argmax(dim=1).cpu().numpy().astype("u1"),
                        "teacher_top1_runnerup_margin_f16": margin.cpu().numpy().astype("<f2"),
                        "teacher_posenet_first6_f32": pose6.cpu().numpy().astype("<f4"),
                        "original_gt_segnet_argmax_u8": np.asarray(gt["lstars"][start:end]).astype("u1"),
                        "original_gt_posenet_first6_f32": np.asarray(gt["gt_poses"][start:end]).astype("<f4"),
                    }
                record = _atomic_npz(path, **arrays)
                atomic_json(
                    path.with_suffix(".json"),
                    {
                        "schema": "ddm_wd3_teacher_scorer_cache_chunk.v1",
                        "repeat": repeat,
                        "start_pair": start,
                        "end_pair": end,
                        "cache_binding": cache_binding,
                        "payload": record,
                        "complete": True,
                    },
                )
            _aggregate_cache_repeat(
                repeat_root,
                chunk_pairs=chunk_pairs,
                expected_binding=cache_binding,
            )
    del fixed, teacher
    repeats = []
    for repeat in range(2):
        payloads = _aggregate_cache_repeat(
            cache_root / f"repeat_{repeat}",
            chunk_pairs=chunk_pairs,
            expected_binding=cache_binding,
        )
        repeats.append(payloads)
    differing = [
        name
        for name in CACHE_FIELDS
        if repeats[0][name]["sha256"] != repeats[1][name]["sha256"]
        or repeats[0][name]["bytes"] != repeats[1][name]["bytes"]
    ]
    if differing:
        raise WD3Error(f"teacher/scorer cache repeat is not byte-identical: {differing}")
    original_targets = np.load(
        Path(repeats[0]["original_gt_segnet_argmax_u8"]["path"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    negative_strata = derive_negative_population_strata(original_targets)
    negative_n120 = stratified_random_indices(negative_strata)
    del original_targets
    receipt = {
        "schema": CACHE_SCHEMA,
        "complete": True,
        "axis": f"[{platform.system()}-{device.type} frozen-scorer training cache; no score claim]",
        "score_claim": False,
        "promotion_eligible": False,
        "config_sha256": validation["config_sha256"],
        "pins": validation["pins"],
        "upstream_snapshot_sha256": "fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799",
        "source_archive_sha256": wd2_build.EXPECTED["archive"][1],
        "scorer_weights": {
            "posenet": validation["pins"]["posenet"],
            "segnet": validation["pins"]["segnet"],
        },
        "command": [str(Path(__file__).resolve()), "prepare-teacher-scorer-cache", "--compiled-config"],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": str(device),
        },
        "chunk_pairs": chunk_pairs,
        "cache_binding": cache_binding,
        "repeats": repeats,
        "aggregate_sha256": canonical_sha256(repeats[0]),
        "determinism_repeat_byte_identical": True,
        "negative_strata": {
            "derivation": "temporal 60-pair block x dominant original SegNet target class",
            "population_sha256": hashlib.sha256(negative_strata.astype("<i8", copy=False).tobytes()).hexdigest(),
            "stratum_counts": dict(sorted(Counter(map(int, negative_strata)).items())),
        },
        "negative_n120": negative_n120.tolist(),
        "negative_n120_sha256": canonical_sha256(negative_n120.tolist()),
        "resumable_from_immutable_chunks": True,
        "all_payloads_retained": True,
        "storage": storage,
    }
    atomic_json(Path(config["output"]) / "TEACHER_SCORER_CACHE_RESULT.json", receipt)
    return receipt


def _load_cache_result(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != CACHE_SCHEMA
        or receipt.get("complete") is not True
        or receipt.get("determinism_repeat_byte_identical") is not True
        or receipt.get("all_payloads_retained") is not True
    ):
        raise WD3Error("teacher/scorer cache receipt is incomplete")
    arrays = {}
    for name, (dtype, shape) in CACHE_FIELDS.items():
        record = receipt["repeats"][0][name]
        payload = Path(record["path"])
        if file_record(payload) != record:
            raise WD3Error(f"teacher/scorer cache payload drifted: {name}")
        array = np.load(payload, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise WD3Error(f"teacher/scorer aggregate geometry differs: {name}")
        arrays[name] = array
    return receipt, arrays


def _load_fixed_frames(pair_ids: np.ndarray, device: torch.device) -> torch.Tensor:
    mapped = _fixed_frame0_memmap()
    array = np.asarray(mapped[2 * pair_ids]).copy()
    del mapped
    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)


def _load_teacher_frames(pair_ids: np.ndarray, device: torch.device) -> torch.Tensor:
    mapped = _teacher_master_memmap()
    array = np.asarray(mapped[pair_ids]).copy()
    del mapped
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def _retain_packet_archive(
    root: Path,
    model: receiver.StudentSemanticRenderer,
    allocation: receiver.AdaptiveQuantizationAllocation,
    stage_a: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packet = receiver.pack_student(model, allocation)
    parsed = receiver.unpack_student(packet)
    if receiver.pack_student(parsed, allocation) != packet:
        raise WD3Error("WD3 packet is not byte-idempotent")
    if stage_a is not None:
        from experiments import ddm_s1a_stage_a_adapter as s1a

        if sha256_file(Path(s1a.__file__).resolve()) != stage_a["adapter_sha256"]:
            raise WD3Error("Stage-A packet adapter source differs from the compiled binding")
        return s1a.retain_renderer_candidate(
            root=root,
            packet=packet,
            allocation=receiver.allocation_telemetry(model, allocation),
            binding=stage_a,
        )
    semantic_stream = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)
    hpac, _, carrier = wd2_build._source_streams()
    model_blob = wd2_build._pack_rx1_model(hpac, semantic_stream, carrier)
    member = model_blob + wd2_build.SOURCE_RESIDUAL.read_bytes() + wd2_build.SOURCE_TOKEN.read_bytes()
    archive = wd2_build.deterministic_zip(member)
    repeat = wd2_build.deterministic_zip(member)
    if archive != repeat:
        raise WD3Error("WD3 archive determinism repeat differs")
    payloads = {
        "student_packet": atomic_bytes(root / "semantic.wd3q", packet),
        "semantic_brotli_q11": atomic_bytes(root / "semantic.br", semantic_stream),
        "model": atomic_bytes(root / "models.rx1m", model_blob),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", repeat),
    }
    runtime = root / "submission"
    patch = receiver.patch_runtime_tree(wd2_build.SOURCE_RUNTIME, runtime)
    binding = receiver.bind_archive(runtime, root / "archive.zip")
    parsed_fields = wd2_build._load_residual_parts(runtime, runtime / "archive.zip")
    if parsed_fields["semantic_blob"] != packet:
        raise WD3Error("full-container WD3 parse-back changed the student packet")
    return {
        "schema": "ddm_wd3_retained_packet_archive.v1",
        "payloads": payloads,
        "runtime_patch": patch,
        "archive_binding": binding,
        "allocation": receiver.allocation_telemetry(model, allocation),
        "receiver_parse_back_exact": True,
        "archive_repeat_byte_identical": True,
        "archive_bytes": len(archive),
        "rate_contribution": 25.0 * len(archive) / RATE_DENOMINATOR,
    }


def evaluate_subset_and_retain(
    *,
    root: Path,
    model: receiver.StudentSemanticRenderer,
    allocation: receiver.AdaptiveQuantizationAllocation,
    pair_ids: Sequence[int],
    tokens: torch.Tensor,
    cache: Mapping[str, np.ndarray],
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    chunk_pairs: int,
    stage_a: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ids = np.asarray(pair_ids, dtype=np.int64)
    if ids.size < 1 or np.unique(ids).size != ids.size or chunk_pairs > MAX_CHUNK:
        raise WD3Error("evaluation subset/chunk geometry differs")
    root.mkdir(parents=True, exist_ok=True)
    evaluation_binding = {
        "pair_ids_sha256": canonical_sha256(ids.tolist()),
        "allocation_sha256": canonical_sha256(allocation.as_dict()),
        "student_packet_sha256": hashlib.sha256(receiver.pack_student(model, allocation)).hexdigest(),
        "cache_surface_sha256": canonical_sha256(_cache_surface_identity(cache)),
    }
    result_path = root / "EVALUATION_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            prior.get("schema") != "ddm_wd3_retained_subset_evaluation.v1"
            or prior.get("pair_ids") != ids.tolist()
            or prior.get("evaluation_binding") != evaluation_binding
            or prior.get("all_payloads_retained") is not True
            or file_record(Path(prior["receiver_pairs"]["path"])) != prior["receiver_pairs"]
            or file_record(Path(prior["scorer_bundle"]["path"])) != prior["scorer_bundle"]
            or file_record(Path(prior["packet_archive"]["payloads"]["student_packet"]["path"]))
            != prior["packet_archive"]["payloads"]["student_packet"]
            or file_record(Path(prior["packet_archive"]["payloads"]["archive"]["path"]))
            != prior["packet_archive"]["payloads"]["archive"]
        ):
            raise WD3Error(f"retained subset evaluation resume differs: {result_path}")
        return prior
    pair_path = root / "receiver_pairs.rgb.u8"
    temporary = root / "receiver_pairs.in_progress.u8"
    progress_path = root / "render_progress.json"
    expected_pair_bytes = ids.size * 2 * 3 * receiver.CAMERA_H * receiver.CAMERA_W
    if pair_path.is_file():
        if pair_path.stat().st_size != expected_pair_bytes:
            raise WD3Error("retained receiver-pair payload size differs")
        completed = ids.size
        rendered = None
    elif progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_subset_pairs", -1))
        if (
            progress.get("schema") != "ddm_wd3_subset_render_progress.v1"
            or progress.get("evaluation_binding") != evaluation_binding
            or not temporary.is_file()
            or temporary.stat().st_size != expected_pair_bytes
            or not 0 <= completed <= ids.size
        ):
            raise WD3Error("retained subset render resume state differs")
        rendered = np.memmap(
            temporary,
            mode="r+",
            dtype=np.uint8,
            shape=(ids.size, 2, 3, receiver.CAMERA_H, receiver.CAMERA_W),
        )
    else:
        with temporary.open("wb") as stream:
            stream.truncate(expected_pair_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        completed = 0
        rendered = np.memmap(
            temporary,
            mode="r+",
            dtype=np.uint8,
            shape=(ids.size, 2, 3, receiver.CAMERA_H, receiver.CAMERA_W),
        )
        atomic_json(
            progress_path,
            {
                "schema": "ddm_wd3_subset_render_progress.v1",
                "completed_subset_pairs": 0,
                "evaluation_binding": evaluation_binding,
            },
        )
    with torch.no_grad():
        for offset in range(completed, ids.size, chunk_pairs):
            chunk_ids = ids[offset : offset + chunk_pairs]
            index = torch.from_numpy(chunk_ids).to(device)
            pair, _ = paired_receiver_tensor(
                model=model,
                allocation=allocation,
                tokens=tokens[torch.from_numpy(chunk_ids)].to(device),
                pair_indices=index,
                fixed_frame0=_load_fixed_frames(chunk_ids, device),
            )
            pose6, logits = scorer_forward(pair, posenet, segnet)
            realized = pair.round().to(torch.uint8).cpu().numpy()
            assert rendered is not None
            rendered[offset : offset + chunk_ids.size] = realized
            rendered.flush()
            _atomic_npz(
                root / "scorer_chunks" / f"pairs_{offset:04d}_{offset + chunk_ids.size:04d}.npz",
                pair_ids=chunk_ids,
                segnet_logits_f32=logits.cpu().numpy().astype("<f4"),
                posenet_first6_f32=pose6.cpu().numpy().astype("<f4"),
            )
            atomic_json(
                progress_path,
                {
                    "schema": "ddm_wd3_subset_render_progress.v1",
                    "completed_subset_pairs": offset + chunk_ids.size,
                    "evaluation_binding": evaluation_binding,
                },
            )
    if rendered is not None:
        del rendered
        os.replace(temporary, pair_path)
    all_logits = []
    all_pose = []
    for offset in range(0, ids.size, chunk_pairs):
        end = min(ids.size, offset + chunk_pairs)
        chunk_path = root / "scorer_chunks" / f"pairs_{offset:04d}_{end:04d}.npz"
        if not chunk_path.is_file():
            raise WD3Error("retained scorer chunk is absent after render completion")
        with np.load(chunk_path, allow_pickle=False) as chunk:
            if not np.array_equal(chunk["pair_ids"], ids[offset:end]):
                raise WD3Error("retained scorer chunk pair order differs")
            all_logits.append(torch.from_numpy(chunk["segnet_logits_f32"].astype(np.float32)))
            all_pose.append(torch.from_numpy(chunk["posenet_first6_f32"].astype(np.float32)))
    logits = torch.cat(all_logits)
    pose6 = torch.cat(all_pose)
    gt_argmax = torch.from_numpy(np.asarray(cache["original_gt_segnet_argmax_u8"][ids]).copy()).long()
    gt_pose = torch.from_numpy(np.asarray(cache["original_gt_posenet_first6_f32"][ids]).copy()).float()
    student_argmax = logits.argmax(dim=1)
    hard_dseg = float((student_argmax != gt_argmax).float().mean())
    dpose = float((pose6 - gt_pose).square().mean())
    edge = cell_edge_telemetry(student_argmax, gt_argmax)
    scorer_bundle = _atomic_npz(
        root / "scorer_outputs.npz",
        pair_ids=ids,
        segnet_logits_f32=logits.numpy().astype("<f4"),
        segnet_logits_f16=logits.numpy().astype("<f2"),
        segnet_argmax_u8=student_argmax.numpy().astype("u1"),
        posenet_first6_f32=pose6.numpy().astype("<f4"),
    )
    packet_archive = _retain_packet_archive(root / "candidate", model, allocation, stage_a)
    result = {
        "schema": "ddm_wd3_retained_subset_evaluation.v1",
        "axis": f"[{platform.system()}-{device.type} frozen-scorer advisory]",
        "score_claim": False,
        "pair_ids": ids.tolist(),
        "subset_sha256": canonical_sha256(ids.tolist()),
        "evaluation_binding": evaluation_binding,
        "n_pairs": int(ids.size),
        "hard_d_seg": hard_dseg,
        "d_pose": dpose,
        "seg_contribution": 100.0 * hard_dseg,
        "pose_contribution": math.sqrt(10.0 * dpose),
        "cell_edges": edge,
        "receiver_pairs": file_record(pair_path),
        "scorer_bundle": scorer_bundle,
        "packet_archive": packet_archive,
        "all_payloads_retained": True,
    }
    atomic_json(result_path, result)
    return result


def _selection_memmap(path: Path, mode: str = "r") -> np.memmap:
    return np.memmap(
        path,
        mode=mode,
        dtype=np.uint8,
        shape=(receiver.N, 384, 512),
    )


def _cache_surface_identity(cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Bind resumable controller work to the already receipt-validated cache files."""

    identity: dict[str, Any] = {}
    for name in CACHE_FIELDS:
        array = cache[name]
        filename = getattr(array, "filename", None)
        identity[name] = {
            "path": str(Path(filename).resolve()) if filename is not None else None,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "bytes": int(array.nbytes),
        }
    return identity


def materialize_stage_controller(
    *,
    root: Path,
    model: receiver.StudentSemanticRenderer,
    tokens: torch.Tensor,
    cache: Mapping[str, np.ndarray],
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    chunk_pairs: int,
    stage_a: Mapping[str, Any] | None = None,
    cheap_to_shrink: ds1.CheapToShrinkConfig | None = None,
) -> tuple[dict[str, Any], receiver.AdaptiveQuantizationAllocation, StageThresholds]:
    """Build measured cells, frozen calibration, and adaptive packet allocation."""

    root.mkdir(parents=True, exist_ok=True)
    uniform4 = receiver.uniform_allocation(model, 4)
    controller_binding = {
        "schema": "ddm_wd3_stage_controller_binding.v1",
        "student_packet_sha256": hashlib.sha256(receiver.pack_student(model, uniform4)).hexdigest(),
        "cache_surface": _cache_surface_identity(cache),
        "chunk_pairs": int(chunk_pairs),
        "controller_ids_sha256": canonical_sha256(evenly_strided_indices().tolist()),
        "selection_rule": "teacher_student_gt_mismatch_union_plus_cell_boundary",
    }
    controller_binding_path = root / "CONTROLLER_BINDING.json"
    if controller_binding_path.is_file():
        if json.loads(controller_binding_path.read_text(encoding="utf-8")) != controller_binding:
            raise WD3Error("stage-controller root binding differs")
    else:
        atomic_json(controller_binding_path, controller_binding)
    result_path = root / "STAGE_CONTROLLER_RESULT.json"
    if result_path.is_file():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        cheap = cheap_to_shrink or ds1.DEFAULT_CONFIG
        if (
            prior.get("schema") != STAGE_SCHEMA
            or prior.get("complete") is not True
            or prior.get("controller_binding") != controller_binding
            or (
                (stage_a is not None or "cheap_to_shrink" in prior)
                and prior.get("cheap_to_shrink") != cheap.provenance()
            )
        ):
            raise WD3Error("stage-controller resume receipt is incomplete")
        if file_record(Path(prior["selection"]["path"])) != prior["selection"]:
            raise WD3Error("stage-controller selection payload drifted")
        allocation = receiver.AdaptiveQuantizationAllocation.from_dict(prior["chosen_allocation"])
        allocation.validate(model)
        if canonical_sha256(allocation.as_dict()) != prior.get("chosen_allocation_sha256"):
            raise WD3Error("stage-controller chosen allocation binding differs")
        _, ladder = _training_rung_allocations(model, allocation, cheap)
        if (stage_a is not None or "cheap_to_shrink_ladder" in prior) and prior.get("cheap_to_shrink_ladder") != ladder:
            raise WD3Error("stage-controller cheap-to-shrink ladder binding differs")
        thresholds = StageThresholds(**prior["thresholds"])
        return prior, allocation, thresholds
    selection_path = root / "selective_cell_mask.u8"
    in_progress = root / "selective_cell_mask.in_progress.u8"
    progress_path = root / "selection_progress.json"
    expected_selection_bytes = receiver.N * 384 * 512
    if selection_path.is_file():
        if selection_path.stat().st_size != expected_selection_bytes:
            raise WD3Error("completed selective-cell payload size differs")
        completed = receiver.N
        selected = None
    elif progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_pairs", -1))
        if (
            progress.get("schema") != "ddm_wd3_selection_progress.v1"
            or progress.get("controller_binding") != controller_binding
            or not in_progress.is_file()
            or in_progress.stat().st_size != expected_selection_bytes
            or not 0 <= completed <= receiver.N
        ):
            raise WD3Error("selective-cell resume state differs")
        selected = _selection_memmap(in_progress, mode="r+")
    else:
        with in_progress.open("wb") as stream:
            stream.truncate(expected_selection_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        completed = 0
        selected = _selection_memmap(in_progress, mode="r+")
        atomic_json(
            progress_path,
            {
                "schema": "ddm_wd3_selection_progress.v1",
                "completed_pairs": 0,
                "population_pairs": receiver.N,
                "chunk_pairs": chunk_pairs,
                "controller_binding": controller_binding,
            },
        )
    for start in range(completed, receiver.N, chunk_pairs):
        end = min(receiver.N, start + chunk_pairs)
        ids = np.arange(start, end, dtype=np.int64)
        with torch.no_grad():
            pair, _ = paired_receiver_tensor(
                model=model,
                allocation=uniform4,
                tokens=tokens[start:end].to(device),
                pair_indices=torch.arange(start, end, device=device),
                fixed_frame0=_load_fixed_frames(ids, device),
            )
            _, logits = scorer_forward(pair, posenet, segnet)
        mask = derive_selective_cell_mask(
            logits.argmax(dim=1).cpu(),
            torch.from_numpy(np.asarray(cache["teacher_segnet_argmax_u8"][start:end]).copy()).long(),
            torch.from_numpy(np.asarray(cache["original_gt_segnet_argmax_u8"][start:end]).copy()).long(),
        )
        assert selected is not None
        mask_array = mask.numpy().astype(np.uint8)
        selected[start:end] = mask_array
        selected.flush()
        chunk_record = atomic_bytes(
            root / "selection_chunks" / f"pairs_{start:04d}_{end:04d}.u8",
            mask_array.tobytes(),
        )
        atomic_json(
            root / "selection_chunks" / f"pairs_{start:04d}_{end:04d}.json",
            {
                "schema": "ddm_wd3_selection_chunk.v1",
                "start_pair": start,
                "end_pair": end,
                "controller_binding": controller_binding,
                "payload": chunk_record,
            },
        )
        atomic_json(
            progress_path,
            {
                "schema": "ddm_wd3_selection_progress.v1",
                "completed_pairs": end,
                "population_pairs": receiver.N,
                "chunk_pairs": chunk_pairs,
                "controller_binding": controller_binding,
            },
        )
    if selected is not None:
        del selected
        os.replace(in_progress, selection_path)
    selection_record = file_record(selection_path)
    uniform4 = receiver.uniform_allocation(model, 4, selection_sha256=selection_record["sha256"])

    controller_ids = evenly_strided_indices()
    baseline_evaluation = evaluate_subset_and_retain(
        root=root / "controller_baseline_uniform4",
        model=model,
        allocation=uniform4,
        pair_ids=controller_ids,
        tokens=tokens,
        cache=cache,
        posenet=posenet,
        segnet=segnet,
        device=device,
        chunk_pairs=chunk_pairs,
        stage_a=stage_a,
    )
    with np.load(Path(baseline_evaluation["scorer_bundle"]["path"]), allow_pickle=False) as baseline_scorer:
        logits_all = torch.from_numpy(baseline_scorer["segnet_logits_f32"].astype(np.float32))
        pose_all = torch.from_numpy(baseline_scorer["posenet_first6_f32"].astype(np.float32))
    baseline_pairs = np.memmap(
        Path(baseline_evaluation["receiver_pairs"]["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(controller_ids.size, 2, 3, receiver.CAMERA_H, receiver.CAMERA_W),
    )
    frame_all = torch.from_numpy(np.asarray(baseline_pairs[:, 1]).copy()).float()
    del baseline_pairs
    teacher_frame_all = _load_teacher_frames(controller_ids, torch.device("cpu"))
    gradients = {name: torch.zeros_like(value, device="cpu") for name, value in model.named_parameters()}
    gt_argmax = torch.from_numpy(np.asarray(cache["original_gt_segnet_argmax_u8"][controller_ids]).copy()).long()
    calibration = calibrate_soft_disagreement(logits_all, gt_argmax)
    teacher_logits = torch.from_numpy(np.asarray(cache["teacher_segnet_logits_f16"][controller_ids]).copy()).float()
    teacher_argmax = torch.from_numpy(np.asarray(cache["teacher_segnet_argmax_u8"][controller_ids]).copy()).long()
    teacher_margin = torch.from_numpy(
        np.asarray(cache["teacher_top1_runnerup_margin_f16"][controller_ids]).copy()
    ).float()
    selection_controller = torch.from_numpy(np.asarray(_selection_memmap(selection_path)[controller_ids]).copy()).bool()
    dummy_thresholds = StageThresholds(calibration["stage_frozen_calibration_scale"], float("inf"), float("inf"))
    _, baseline_components = score_native_objective(
        student_logits=logits_all,
        student_pose6=pose_all,
        student_frame1=frame_all,
        teacher_logits=teacher_logits,
        teacher_argmax=teacher_argmax,
        teacher_margin=teacher_margin,
        teacher_pose6=torch.from_numpy(np.asarray(cache["teacher_posenet_first6_f32"][controller_ids]).copy()).float(),
        original_argmax=gt_argmax,
        original_pose6=torch.from_numpy(
            np.asarray(cache["original_gt_posenet_first6_f32"][controller_ids]).copy()
        ).float(),
        teacher_frame1=teacher_frame_all,
        selected_cells=selection_controller,
        thresholds=dummy_thresholds,
        duals=DualState(),
    )
    thresholds = StageThresholds(
        calibration["stage_frozen_calibration_scale"],
        float(baseline_components["teacher_impostor_complete_margin_hinge_loss"]),
        float(baseline_components["teacher_t2_kl_loss"]),
    )
    # Second pass accumulates real score-native gradients for per-group bit waterfill.
    model.zero_grad(set_to_none=True)
    gradient_checkpoint = root / "gradient_accumulator.pt"
    gradient_start = 0
    if gradient_checkpoint.is_file():
        accumulated = torch.load(gradient_checkpoint, map_location="cpu", weights_only=False)
        if accumulated.get("schema") != "ddm_wd3_gradient_accumulator.v1":
            raise WD3Error("stage gradient resume checkpoint schema differs")
        if (
            accumulated.get("selection_sha256") != selection_record["sha256"]
            or accumulated.get("thresholds") != asdict(thresholds)
            or accumulated.get("controller_binding") != controller_binding
        ):
            raise WD3Error("stage gradient resume binding differs")
        gradient_start = int(accumulated["next_offset"])
        for name, value in model.named_parameters():
            value.grad = accumulated["gradients"][name].to(device)
    for offset in range(gradient_start, controller_ids.size, chunk_pairs):
        ids = controller_ids[offset : offset + chunk_pairs]
        pair, frame1 = paired_receiver_tensor(
            model=model,
            allocation=uniform4,
            tokens=tokens[torch.from_numpy(ids)].to(device),
            pair_indices=torch.from_numpy(ids).to(device),
            fixed_frame0=_load_fixed_frames(ids, device),
        )
        pose6, logits = scorer_forward(pair, posenet, segnet)
        sl = slice(offset, offset + ids.size)
        total, _ = score_native_objective(
            student_logits=logits,
            student_pose6=pose6,
            student_frame1=frame1,
            teacher_logits=teacher_logits[sl].to(device),
            teacher_argmax=teacher_argmax[sl].to(device),
            teacher_margin=teacher_margin[sl].to(device),
            teacher_pose6=torch.from_numpy(np.asarray(cache["teacher_posenet_first6_f32"][ids]).copy()).to(device),
            original_argmax=gt_argmax[sl].to(device),
            original_pose6=torch.from_numpy(np.asarray(cache["original_gt_posenet_first6_f32"][ids]).copy()).to(device),
            teacher_frame1=_load_teacher_frames(ids, device),
            selected_cells=selection_controller[sl].to(device),
            thresholds=thresholds,
            duals=DualState(),
        )
        (total / math.ceil(controller_ids.size / chunk_pairs)).backward()
        _atomic_npz(
            root / "gradient_chunks" / f"controller_{offset:04d}_{offset + ids.size:04d}.npz",
            pair_ids=ids,
            receiver_pairs_u8=pair.detach().round().to(torch.uint8).cpu().numpy(),
            segnet_logits_f16=logits.detach().cpu().numpy().astype("<f2"),
            posenet_first6_f32=pose6.detach().cpu().numpy().astype("<f4"),
        )
        atomic_torch(
            gradient_checkpoint,
            {
                "schema": "ddm_wd3_gradient_accumulator.v1",
                "next_offset": offset + ids.size,
                "gradients": {name: value.grad.detach().cpu().clone() for name, value in model.named_parameters()},
                "selection_sha256": selection_record["sha256"],
                "thresholds": asdict(thresholds),
                "controller_binding": controller_binding,
            },
        )
    for name, value in model.named_parameters():
        if value.grad is None:
            raise WD3Error(f"score-native gradient is absent for {name}")
        gradients[name].copy_(value.grad.detach().cpu())
    model.zero_grad(set_to_none=True)
    sensitivity = quantization_sensitivity_table(model, gradients)
    uniform4_error = sum(float(row["errors"]["4"]) for rows in sensitivity.values() for row in rows)
    adaptive = adaptive_allocation_from_sensitivity(
        model,
        sensitivity,
        maximum_predicted_error=uniform4_error,
        selection_sha256=selection_record["sha256"],
    )
    # Retain and naively re-score cheapest rungs; only measured rows may select.
    allocations = {
        "uniform2": receiver.uniform_allocation(model, 2, selection_sha256=selection_record["sha256"]),
        "uniform3": receiver.uniform_allocation(model, 3, selection_sha256=selection_record["sha256"]),
        "uniform4": receiver.uniform_allocation(model, 4, selection_sha256=selection_record["sha256"]),
        "adaptive": adaptive,
    }
    evaluations = {}
    for name, allocation in allocations.items():
        evaluations[name] = (
            baseline_evaluation
            if name == "uniform4"
            else evaluate_subset_and_retain(
                root=root / "quantization_race" / name,
                model=model,
                allocation=allocation,
                pair_ids=controller_ids,
                tokens=tokens,
                cache=cache,
                posenet=posenet,
                segnet=segnet,
                device=device,
                chunk_pairs=chunk_pairs,
                stage_a=stage_a,
            )
        )
    baseline = evaluations["uniform4"]
    rows = []
    for name, evaluation in evaluations.items():
        rows.append(
            {
                "allocation_id": name,
                "packet_bytes": evaluation["packet_archive"]["payloads"]["student_packet"]["bytes"],
                "hard_cell_gate_pass": evaluation["hard_d_seg"] <= baseline["hard_d_seg"],
                "road_lane_gate_pass": evaluation["cell_edges"]["road_lane_flips"]
                <= baseline["cell_edges"]["road_lane_flips"],
                "pose_gate_pass": evaluation["d_pose"] <= baseline["d_pose"],
                "parse_back_exact": evaluation["packet_archive"]["receiver_parse_back_exact"],
                "retained_payload": evaluation["all_payloads_retained"],
                "measured": True,
            }
        )
    winner = choose_cheapest_passing_quantization(rows)
    chosen = allocations[str(winner["allocation_id"])]
    cheap = cheap_to_shrink or ds1.DEFAULT_CONFIG
    rung_ladder = _training_rung_allocations(model, chosen, cheap)
    result = {
        "schema": STAGE_SCHEMA,
        "complete": True,
        "controller_binding": controller_binding,
        "selection": selection_record,
        "selection_derivation": "teacher/student/GT mismatch union plus one-cell cell-boundary band; no top-k",
        "controller_ids": controller_ids.tolist(),
        "calibration": calibration,
        "thresholds": asdict(thresholds),
        "quantization_sensitivity": sensitivity,
        "quantization_race": rows,
        "chosen_allocation": chosen.as_dict(),
        "chosen_allocation_sha256": canonical_sha256(chosen.as_dict()),
        "cheap_to_shrink": cheap.provenance(),
        "cheap_to_shrink_ladder": rung_ladder[1],
        "all_payloads_retained": True,
        "negative_claim": False,
    }
    atomic_json(result_path, result)
    return result, chosen, thresholds


def _training_rung_allocations(
    model: receiver.StudentSemanticRenderer,
    base: receiver.AdaptiveQuantizationAllocation,
    config: ds1.CheapToShrinkConfig,
) -> tuple[tuple[receiver.AdaptiveQuantizationAllocation, ...], dict[str, Any]]:
    """Materialize the exact ordered packet ladder used by the training objective."""

    if ds1.is_inert(config):
        return (), {
            "allocation_family": config.allocation_family,
            "active": False,
            "byte_cost_checked": True,
            "base_bytes": len(receiver.pack_student(model, base)),
            "rung_bytes": [],
        }
    if config.allocation_family != "uniform_bits":
        raise WD3Error("Stage-A cheap-to-shrink admits only the R0-corrected uniform ladder")
    ladder = ds1.derive_uniform_rung_ladder(
        base_allocation=base,
        allocation_for_bits=lambda bits: receiver.uniform_allocation(
            model,
            bits,
            selection_sha256=base.selection_sha256,
        ),
        config=config,
        byte_cost=lambda allocation: len(receiver.pack_student(model, allocation)),
    )
    return tuple(ladder.cheaper_allocations), {**ladder.diagnostics, "active": True}


def _verify_launch_sources(config: Mapping[str, Any]) -> None:
    if sha256_file(Path(__file__).resolve()) != config["expected_builder_sha256"]:
        raise WD3Error("sealed WD3 builder SHA changed")
    if sha256_file(Path(receiver.__file__).resolve()) != config["expected_receiver_sha256"]:
        raise WD3Error("sealed WD3 receiver SHA changed")
    stage_a = _stage_a_binding(config)
    if stage_a is not None and sha256_file(Path(stage_a["adapter_module"])) != stage_a["adapter_sha256"]:
        raise WD3Error("sealed Stage-A adapter SHA changed")


def _optimizer_config(config: Mapping[str, Any]) -> dict[str, float]:
    optimizer = config["optimizer"]
    _strict_fields(
        optimizer,
        {"lr", "weight_decay", "grad_clip", "dual_step", "reset_ramp_divisor"},
        "optimizer",
    )
    values = {name: float(value) for name, value in optimizer.items()}
    if min(values["lr"], values["grad_clip"], values["dual_step"]) <= 0:
        raise WD3Error("WD3 optimizer/dual magnitudes must be positive")
    if values["weight_decay"] < 0:
        raise WD3Error("WD3 weight decay is negative")
    if config["arm"] == "W0_reset" and not 3.16 <= values["reset_ramp_divisor"] <= 6.57:
        raise WD3Error("W0 reset ramp does not remove the measured 3.16x--6.57x excursion")
    return values


def _new_optimizer_scheduler(
    model: nn.Module, config: Mapping[str, Any]
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LRScheduler]:
    values = _optimizer_config(config)
    lr = values["lr"]
    if config["arm"] == "W0_reset":
        initial_lr = lr / values["reset_ramp_divisor"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=values["weight_decay"])
        ramp_epochs = min(5, int(config["epochs"]))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 + (values["reset_ramp_divisor"] - 1.0) * min(epoch, ramp_epochs) / ramp_epochs,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=values["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(config["epochs"])), eta_min=lr * 0.02
        )
    return optimizer, scheduler


def _birth_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    arm = str(config["arm"])
    return {
        "schema": "ddm_wd3_arm_birth_contract.v1",
        "arm": arm,
        "student_spec": asdict(ARM_SPECS[arm]),
        "seed": int(config["seed"]),
        "optimizer": dict(config["optimizer"]),
        "epochs": int(config["epochs"]),
        "batch_pairs": int(config["batch_pairs"]),
        "source_pins": dict(config["source_pins"]),
        "expected_builder_sha256": config["expected_builder_sha256"],
        "expected_receiver_sha256": config["expected_receiver_sha256"],
        # The ON/OFF pair at seed 20260815 deliberately shares this birth.  The
        # treatment is excluded so identical weights, optimizer moments and RNG
        # make the treatment the only changed mechanism.
        "stage_a": dict(config["stage_a"]) if config["stage_a"] is not None else None,
    }


def prepare_arm_birth(config: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a scorer-free, byte-close-loadable birth for non-W0 topologies."""

    validation = validate_compiled_config(config)
    if config["action"] != "prepare_arm_birth":
        raise WD3Error("arm-birth builder received a different action")
    if config["arm"] in {"W0_warm", "W0_reset"}:
        raise WD3Error("W0 arms must migrate the pinned matched WD2 checkpoint")
    _verify_launch_sources(config)
    seed_everything(int(config["seed"]))
    model = receiver.StudentSemanticRenderer(ARM_SPECS[str(config["arm"])])
    stage_a = _stage_a_binding(config)
    storage = storage_preflight(
        Path(config["output"]),
        int(config["minimum_free_bytes"]),
        allowed_root=STAGE_A_OUTPUT_ROOT if stage_a is not None else OUTPUT_ROOT,
    )
    initialization = None
    if stage_a is not None:
        initialization = torch.load(Path(stage_a["initializer"]), map_location="cpu", weights_only=False)
        if not isinstance(initialization, Mapping):
            raise WD3Error("Stage-A initializer is not a state dictionary")
        model.load_state_dict(initialization, strict=True)
    optimizer, scheduler = _new_optimizer_scheduler(model, config)
    updates = max(
        3,
        int(config["epochs"]) * math.ceil(receiver.N / int(config["batch_pairs"])),
    )
    ema = wd2_build.DeploymentEMA(model, 1.0 - 2.0 / updates)
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    payload = {
        "schema": ARM_BIRTH_SCHEMA,
        "birth_contract": _birth_contract(config),
        "live_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "ema": ema.state(),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "scaler": {"enabled": False, "state_dict": {}},
        "rng": _rng_state(generator),
        "epoch": 0,
        "history": [],
        "deployment_weights": "ema_shadow",
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "training_launched": False,
        "stage_a_initializer": file_record(Path(stage_a["initializer"])) if stage_a is not None else None,
        "initializer_loaded_strict": stage_a is not None,
    }
    checkpoint_path = Path(config["resume_root"]) / f"{config['arm']}_birth.pt"
    checkpoint = atomic_torch(checkpoint_path, payload)
    receipt = {
        "schema": "ddm_wd3_arm_birth_receipt.v1",
        "complete": True,
        "arm": config["arm"],
        "birth_contract": payload["birth_contract"],
        "checkpoint": checkpoint,
        "config_sha256": validation["config_sha256"],
        "resumable_from_disk": True,
        "all_payloads_retained": True,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "storage": storage,
    }
    atomic_json(Path(config["output"]) / f"{config['arm']}_BIRTH_RECEIPT.json", receipt)
    return receipt


def _initialize_training_state(
    config: Mapping[str, Any], device: torch.device
) -> tuple[
    receiver.StudentSemanticRenderer,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LRScheduler,
    wd2_build.DeploymentEMA,
    torch.Generator,
    int,
    list[dict[str, Any]],
    dict[str, Any] | None,
    WD3ResumeController | None,
    receiver.AdaptiveQuantizationAllocation | None,
]:
    arm = str(config["arm"])
    spec = ARM_SPECS[arm]
    model = receiver.StudentSemanticRenderer(spec).to(device)
    optimizer, scheduler = _new_optimizer_scheduler(model, config)
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    resume_path = Path(config["resume_from"])
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if resume.get("schema") == wd2_build.CHECKPOINT_SCHEMA:
        if arm not in {"W0_warm", "W0_reset"} or resume_path != WARM_CHECKPOINT:
            raise WD3Error("only matched W0 arms may migrate the pinned WD2 ep60 checkpoint")
        model.load_state_dict(resume["live_state_dict"], strict=True)
        ema = wd2_build.DeploymentEMA(model, float(resume["ema"]["decay"]))
        ema.restore(resume["ema"])
        if arm == "W0_warm":
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        # Reset arm intentionally starts fresh moments but preserves weights, EMA, RNG and cursor.
        wd2_build._restore_rng_state(resume["rng"], device, generator)
        history = list(resume.get("history", []))
        return (
            model,
            optimizer,
            scheduler,
            ema,
            generator,
            int(resume["epoch"]),
            history,
            resume,
            None,
            None,
        )
    if resume.get("schema") == ARM_BIRTH_SCHEMA:
        if resume.get("birth_contract") != _birth_contract(config):
            raise WD3Error("arm-birth checkpoint contract differs")
        model.load_state_dict(resume["live_state_dict"], strict=True)
        ema = wd2_build.DeploymentEMA(model, float(resume["ema"]["decay"]))
        ema.restore(resume["ema"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        _restore_rng(resume["rng"], generator)
        return (
            model,
            optimizer,
            scheduler,
            ema,
            generator,
            int(resume["epoch"]),
            list(resume.get("history", [])),
            resume,
            None,
            None,
        )
    if resume.get("schema") != CHECKPOINT_SCHEMA:
        raise WD3Error("resume checkpoint is neither pinned WD2 nor WD3")
    decay = float(resume["ema"]["decay"])
    ema = wd2_build.DeploymentEMA(model, decay)
    payload, controller, allocation = load_checkpoint(
        resume_path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        expected_config=config,
    )
    return (
        model,
        optimizer,
        scheduler,
        ema,
        generator,
        int(controller.epoch),
        list(payload["history"]),
        payload,
        controller,
        allocation,
    )


def _batch_objective(
    *,
    model: receiver.StudentSemanticRenderer,
    allocation: receiver.AdaptiveQuantizationAllocation,
    ids: np.ndarray,
    tokens: torch.Tensor,
    cache: Mapping[str, np.ndarray],
    selection: np.memmap,
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    thresholds: StageThresholds,
    duals: DualState,
    cheap_to_shrink: ds1.CheapToShrinkConfig = ds1.DEFAULT_CONFIG,
    rung_allocations: Sequence[receiver.AdaptiveQuantizationAllocation] = (),
    step: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = {
        "teacher_logits": torch.from_numpy(np.asarray(cache["teacher_segnet_logits_f16"][ids]).copy()).to(
            device=device, dtype=torch.float32
        ),
        "teacher_argmax": torch.from_numpy(np.asarray(cache["teacher_segnet_argmax_u8"][ids]).copy()).to(
            device=device, dtype=torch.long
        ),
        "teacher_margin": torch.from_numpy(np.asarray(cache["teacher_top1_runnerup_margin_f16"][ids]).copy()).to(
            device=device, dtype=torch.float32
        ),
        "teacher_pose6": torch.from_numpy(np.asarray(cache["teacher_posenet_first6_f32"][ids]).copy()).to(
            device=device, dtype=torch.float32
        ),
        "original_argmax": torch.from_numpy(np.asarray(cache["original_gt_segnet_argmax_u8"][ids]).copy()).to(
            device=device, dtype=torch.long
        ),
        "original_pose6": torch.from_numpy(np.asarray(cache["original_gt_posenet_first6_f32"][ids]).copy()).to(
            device=device, dtype=torch.float32
        ),
        "teacher_frame1": _load_teacher_frames(ids, device),
        "selected_cells": torch.from_numpy(np.asarray(selection[ids]).copy()).to(device=device, dtype=torch.bool),
        "thresholds": thresholds,
        "duals": duals,
    }
    batch_tokens = tokens[torch.from_numpy(ids)].to(device)
    pair_indices = torch.from_numpy(ids).to(device)
    fixed_frame0 = _load_fixed_frames(ids, device)

    def objective_at(
        candidate: receiver.AdaptiveQuantizationAllocation,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pair, frame1 = paired_receiver_tensor(
            model=model,
            allocation=candidate,
            tokens=batch_tokens,
            pair_indices=pair_indices,
            fixed_frame0=fixed_frame0,
        )
        pose6, logits = scorer_forward(pair, posenet, segnet)
        return score_native_objective(
            student_logits=logits,
            student_pose6=pose6,
            student_frame1=frame1,
            **target,
        )

    base_loss, components = objective_at(allocation)
    requested = ds1.rungs_for_step(cheap_to_shrink, step, len(rung_allocations))
    rung_losses = [(index, objective_at(rung_allocations[index])[0]) for index in requested]
    total, telemetry = ds1.apply(
        base_loss=base_loss,
        rung_losses=rung_losses,
        config=cheap_to_shrink,
    )
    components = dict(components)
    components["cheap_to_shrink_active"] = total.new_tensor(float(telemetry["ds1_active"]))
    components["cheap_to_shrink_rungs_evaluated"] = total.new_tensor(float(telemetry["ds1_rungs_evaluated"]))
    return total, components


def train(config: Mapping[str, Any]) -> dict[str, Any]:
    validation = validate_compiled_config(config)
    if config["action"] != "train":
        raise WD3Error("trainer received a non-training config")
    _verify_launch_sources(config)
    assert_governed_admission("ddm_wd3_scorer_aware_width_distillation_train")
    stage_a = _stage_a_binding(config)
    storage = storage_preflight(
        Path(config["output"]),
        int(config["minimum_free_bytes"]),
        allowed_root=STAGE_A_OUTPUT_ROOT if stage_a is not None else OUTPUT_ROOT,
    )
    seed_everything(int(config["seed"]))
    device = _device(str(config["device"]))
    cache_receipt, cache = _load_cache_result(Path(config["teacher_cache_result"]))
    tokens = wd2_build._load_tokens()
    posenet, segnet = load_differentiable_scorers(REPO / "upstream", device=device)
    posenet.eval()
    segnet.eval()
    (
        model,
        optimizer,
        scheduler,
        ema,
        generator,
        start_epoch,
        history,
        resume_payload,
        controller,
        allocation,
    ) = _initialize_training_state(config, device)
    cheap = cheap_to_shrink_config(config)
    output = Path(config["output"]) / str(config["arm"])
    if controller is None or allocation is None:
        stage_number = len(tuple(config["completed_arms"]))
        stage_root = output / "stage_controllers" / f"stage_{stage_number:02d}_from_epoch_{start_epoch:04d}"
        stage_result, allocation, thresholds = materialize_stage_controller(
            root=stage_root,
            model=model,
            tokens=tokens,
            cache=cache,
            posenet=posenet,
            segnet=segnet,
            device=device,
            chunk_pairs=int(config["chunk_pairs"]),
            stage_a=stage_a,
            cheap_to_shrink=cheap,
        )
        controller = WD3ResumeController(
            duals=DualState(),
            thresholds=thresholds,
            epoch=start_epoch,
            batch_cursor=0,
            selection_sha256=stage_result["selection"]["sha256"],
            allocation_sha256=stage_result["chosen_allocation_sha256"],
        )
        selection_record = stage_result["selection"]
    else:
        assert resume_payload is not None
        selection_record = dict(resume_payload["selection"])
        if file_record(Path(selection_record["path"])) != selection_record:
            raise WD3Error("resume selective-cell payload drifted")
    selection = _selection_memmap(Path(selection_record["path"]))
    rung_allocations, rung_ladder = _training_rung_allocations(model, allocation, cheap)
    optimizer_values = _optimizer_config(config)
    controller_ids = list(map(int, config["subsets"]["controller_n60"]))
    negative_ids = list(map(int, config["subsets"]["negative_n120"]))
    subset_ids = {"controller_n60": controller_ids, "negative_n120": negative_ids}
    end_epoch = start_epoch + int(config["epochs"])
    checkpoint_root = output / "checkpoints"
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        permutation = torch.randperm(receiver.N, generator=generator).numpy()
        epoch_components: defaultdict[str, float] = defaultdict(float)
        batches = 0
        for cursor in range(0, receiver.N, int(config["batch_pairs"])):
            ids = permutation[cursor : cursor + int(config["batch_pairs"])].astype(np.int64, copy=False)
            optimizer.zero_grad(set_to_none=True)
            total, components = _batch_objective(
                model=model,
                allocation=allocation,
                ids=ids,
                tokens=tokens,
                cache=cache,
                selection=selection,
                posenet=posenet,
                segnet=segnet,
                device=device,
                thresholds=controller.thresholds,
                duals=controller.duals,
                cheap_to_shrink=cheap,
                rung_allocations=rung_allocations,
                step=(epoch - 1) * math.ceil(receiver.N / int(config["batch_pairs"])) + batches,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_values["grad_clip"])
            optimizer.step()
            ema.update(model)
            for name, value in components.items():
                epoch_components[name] += float(value.detach().cpu())
            controller.duals = controller.duals.update(
                margin_violation=float(components["margin_constraint_violation"].detach().cpu()),
                teacher_kl_violation=float(components["teacher_kl_constraint_violation"].detach().cpu()),
                decode_violation=float(components["decode_constraint_violation"].detach().cpu()),
                teacher_pose_violation=float(
                    components["pose_teacher_first6_adaptive_constraint_violation"].detach().cpu()
                ),
                step_size=optimizer_values["dual_step"],
            )
            controller.batch_cursor = cursor + ids.size
            batches += 1
        scheduler.step()
        controller.epoch = epoch
        controller.batch_cursor = 0
        row: dict[str, Any] = {
            "epoch": epoch,
            "phase": "wd3_scorer_aware_qat",
            "learning_rate": scheduler.get_last_lr()[0],
            "loss_components": {name: value / batches for name, value in epoch_components.items()},
            "duals": asdict(controller.duals),
            "calibration_frozen_within_stage": asdict(controller.thresholds),
        }
        should_evaluate = (
            epoch == start_epoch + 1 or epoch % int(config["checkpoint_every_epochs"]) == 0 or epoch == end_epoch
        )
        if should_evaluate:
            with wd2_build.ema_scope(model, ema):
                evaluation = evaluate_subset_and_retain(
                    root=output / "retained/evaluations" / f"epoch_{epoch:04d}_n60",
                    model=model,
                    allocation=allocation,
                    pair_ids=controller_ids,
                    tokens=tokens,
                    cache=cache,
                    posenet=posenet,
                    segnet=segnet,
                    device=device,
                    chunk_pairs=int(config["chunk_pairs"]),
                    stage_a=stage_a,
                )
            row["controller_n60"] = {
                "hard_d_seg": evaluation["hard_d_seg"],
                "d_pose": evaluation["d_pose"],
                "road_lane_flips": evaluation["cell_edges"]["road_lane_flips"],
                "archive_bytes": evaluation["packet_archive"]["archive_bytes"],
                "receipt": atomic_json(output / "evaluations" / f"epoch_{epoch:04d}_n60.json", evaluation),
                "negative_authority": False,
            }
        history.append(row)
        if should_evaluate:
            save_checkpoint(
                checkpoint_root / f"wd3_epoch_{epoch:04d}.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                controller=controller,
                allocation=allocation,
                selection_record=selection_record,
                subset_ids=subset_ids,
                config=config,
                history=history,
                stage="wd3_scorer_aware_qat",
            )
    stage_checkpoint = save_checkpoint(
        checkpoint_root / f"wd3_stage_end_epoch_{end_epoch:04d}.pt",
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        controller=controller,
        allocation=allocation,
        selection_record=selection_record,
        subset_ids=subset_ids,
        config=config,
        history=history,
        stage="wd3_scorer_aware_qat_stage_end",
    )
    del selection
    result = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "instance_status": "TRAINED_PENDING_N120_IF_NEGATIVE_AND_N600_SAME_INSTRUMENT",
        "score_claim": False,
        "promotion_eligible": False,
        "axis": f"[{platform.system()}-{device.type} training/advisory; exact contest eval not run]",
        "config_sha256": validation["config_sha256"],
        "cache_receipt": file_record(Path(config["teacher_cache_result"])),
        "cache_binding_sha256": canonical_sha256(cache_receipt),
        "history": history,
        "stage_checkpoint": stage_checkpoint,
        "selection": selection_record,
        "allocation": allocation.as_dict(),
        "cheap_to_shrink": cheap.provenance(),
        "cheap_to_shrink_ladder": rung_ladder,
        "resumable_from_disk": True,
        "all_stage_checkpoints_preserved": True,
        "all_evaluated_payloads_retained": True,
        "n120_negative_confirmation_run": False,
        "n600_same_instrument_run": False,
        "contest_eval_run": False,
        "frontier_moved": False,
        "storage": storage,
    }
    atomic_json(output / "TRAIN_RESULT.json", result)
    return result


def contest_score_from_metrics(*, d_seg: float, d_pose: float, archive_bytes: int) -> float:
    if d_seg < 0 or d_pose < 0 or archive_bytes < 1:
        raise WD3Error("contest score inputs are outside their domains")
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / RATE_DENOMINATOR


def compile_same_instrument_admission(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Hard WD3 archive admission; advisory subsets can never pass this gate."""

    required = {
        "n_pairs",
        "hard_d_seg",
        "d_pose",
        "packet_archive",
        "all_payloads_retained",
        "same_instrument_base_sha256",
        "authority_axis",
    }
    if not required.issubset(candidate):
        raise WD3Error("same-instrument admission receipt is incomplete")
    archive = candidate["packet_archive"]
    archive_bytes = int(archive["archive_bytes"])
    score = contest_score_from_metrics(
        d_seg=float(candidate["hard_d_seg"]),
        d_pose=float(candidate["d_pose"]),
        archive_bytes=archive_bytes,
    )
    base_score = contest_score_from_metrics(d_seg=BASE_DSEG, d_pose=BASE_DPOSE, archive_bytes=BASE_BYTES)
    gates = {
        "n600": int(candidate["n_pairs"]) == receiver.N,
        "same_instrument": candidate["same_instrument_base_sha256"] == PINS["base_receipt"][2],
        "authority_axis": candidate["authority_axis"]
        in {
            "contest-CPU",
            "contest-CUDA",
        },
        "delta_score_strict": score - base_score < -3.5e-6,
        "delta_dseg_ceiling": float(candidate["hard_d_seg"]) - BASE_DSEG <= 1.07e-4,
        "receiver_parse_back_exact": archive.get("receiver_parse_back_exact") is True,
        "archive_repeat_byte_identical": archive.get("archive_repeat_byte_identical") is True,
        "all_payloads_retained": candidate["all_payloads_retained"] is True,
    }
    return {
        "schema": "ddm_wd3_same_instrument_admission.v1",
        "disposition": "ADMIT" if all(gates.values()) else "HOLD",
        "candidate_score": score,
        "base_score": base_score,
        "delta_score": score - base_score,
        "gates": gates,
    }


def compile_n120_negative_confirmation(
    *,
    arm: str,
    candidate: Mapping[str, Any],
    matched_baseline: Mapping[str, Any],
    expected_pair_ids: Sequence[int],
) -> dict[str, Any]:
    """Type an arm negative only from matched retained seeded-stratified n120 rows."""

    expected = list(map(int, expected_pair_ids))
    if len(expected) != 120 or np.unique(expected).size != 120 or expected == list(range(120)):
        raise WD3Error("negative authority IDs are not seeded nonprefix n120")
    for label, row in (("candidate", candidate), ("baseline", matched_baseline)):
        if (
            row.get("schema") != "ddm_wd3_retained_subset_evaluation.v1"
            or row.get("pair_ids") != expected
            or row.get("n_pairs") != 120
            or row.get("all_payloads_retained") is not True
            or row.get("packet_archive", {}).get("receiver_parse_back_exact") is not True
        ):
            raise WD3Error(f"{label} is not a retained receiver-closed n120 evaluation")
    candidate_binding = candidate["evaluation_binding"]
    baseline_binding = matched_baseline["evaluation_binding"]
    if candidate_binding.get("pair_ids_sha256") != baseline_binding.get("pair_ids_sha256") or candidate_binding.get(
        "cache_surface_sha256"
    ) != baseline_binding.get("cache_surface_sha256"):
        raise WD3Error("n120 candidate and baseline are not same-population/same-cache")
    candidate_score = contest_score_from_metrics(
        d_seg=float(candidate["hard_d_seg"]),
        d_pose=float(candidate["d_pose"]),
        archive_bytes=int(candidate["packet_archive"]["archive_bytes"]),
    )
    baseline_score = contest_score_from_metrics(
        d_seg=float(matched_baseline["hard_d_seg"]),
        d_pose=float(matched_baseline["d_pose"]),
        archive_bytes=int(matched_baseline["packet_archive"]["archive_bytes"]),
    )
    is_negative = candidate_score >= baseline_score
    return {
        "schema": "ddm_wd3_n120_instance_verdict.v1",
        "arm": arm,
        "verdict_scope": "instance",
        "population": "seeded_stratified_random_nonprefix_n120",
        "pair_ids_sha256": canonical_sha256(expected),
        "candidate_score": candidate_score,
        "matched_baseline_score": baseline_score,
        "delta_score": candidate_score - baseline_score,
        "disposition": "INSTANCE_NEGATIVE" if is_negative else "CONTINUE_TO_N600",
        "family_killed": False,
    }


def compile_surgical_finish_handoff(
    *,
    candidate_receipt: Mapping[str, Any],
    residual_edge_map: Mapping[str, Any],
    qs5_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers = []
    if candidate_receipt.get("schema") != "ddm_wd3_retained_subset_evaluation.v1":
        blockers.append("retained WD3 candidate receipt absent")
    if candidate_receipt.get("all_payloads_retained") is not True:
        blockers.append("candidate payload custody incomplete")
    if not residual_edge_map.get("per_edge_flips") or "Lane<->Road" not in residual_edge_map.get("per_edge_flips", {}):
        blockers.append("residual Road-hub edge map absent")
    if qs5_receipt is None or qs5_receipt.get("schema") not in {
        "ddm_qs5_receipt.v1",
        "ddm_qs5_resolve_compensation.v1",
    }:
        blockers.append("receiver-consumed QS5 Schur compensation receipt absent")
    elif not (
        qs5_receipt.get("repeat_identical") is True
        and qs5_receipt.get("pose_held_below_base") is True
        and qs5_receipt.get("receiver_consumed") is True
    ):
        blockers.append("QS5 Schur receipt lacks repeat/pose/receiver proof")
    return {
        "schema": "ddm_wd3_surgical_finish_handoff.v1",
        "disposition": "READY_FOR_QS2_QS5_COMPILE" if not blockers else "BLOCKED",
        "producer": [
            "experiments/ddm_qs2_compensation_rate_rung.py",
            "experiments/ddm_qs5_resolve_compensation.py",
        ],
        "targeting": "Road-hub class-pair edges; never seg per-pair tails",
        "blockers": blockers,
        "edits_materialized": False,
        "score_claim": False,
    }


def source_pin_contract() -> dict[str, dict[str, Any]]:
    return {name: {"bytes": size, "sha256": digest} for name, (_, size, digest) in PINS.items()}


def blocked_config_template() -> dict[str, Any]:
    """Canonical build receipt input; deliberately cannot launch."""

    strata = np.arange(receiver.N, dtype=np.int64) % 5
    return {
        "schema": SCHEMA,
        "action": "train",
        "output": str(OUTPUT_ROOT / "W0_warm"),
        "seed": SEED,
        "device": "mps",
        "chunk_pairs": 60,
        "retain_all_payloads": True,
        "checkpoint_every_epochs": 5,
        "minimum_free_bytes": 64 << 30,
        "base_receipt": str(BASE_RECEIPT),
        "teacher_cache_result": str(OUTPUT_ROOT / "teacher_scorer_cache/RESULT_OWED.json"),
        "resume_from": str(WARM_CHECKPOINT),
        "resume_root": str(OUTPUT_ROOT / "W0_warm/resume"),
        "arm": "W0_warm",
        "completed_arms": [],
        "negative_confirmed_arms": [],
        "capacity_pressure_confirmed": False,
        "real_coder_override_dense_w96": False,
        "subsets": {
            "controller_n60": evenly_strided_indices().tolist(),
            "negative_n120": stratified_random_indices(strata).tolist(),
            "controller_kind": "evenly_strided",
            "negative_kind": "seeded_stratified_random",
            "prefix": False,
        },
        "objective": {
            "scoreaware": True,
            "seg_score_coefficient": 100.0,
            "pose_exact_nonlinear": True,
            "temperature": 2.0,
            "adaptive_duals": True,
            "decode_mse_ceiling": DECODE_MSE_CEILING,
            "packet_quantizer_in_loop": True,
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "dual_step": 1e-3,
            "reset_ramp_divisor": 3.16,
        },
        "epochs": 5,
        "batch_pairs": 1,
        "scorer_lane": {
            "claimed": False,
            "claim_id": None,
            "agent": "MAIN",
            "platform": "macos-cpu",
        },
        "metal_lane": {
            "claimed": False,
            "claim_id": None,
            "agent": "MAIN",
            "platform": "macos-mps",
        },
        "launch_authorized": False,
        "r5_exit_verified": False,
        "source_pins": source_pin_contract(),
        "expected_builder_sha256": sha256_file(Path(__file__).resolve()),
        "expected_receiver_sha256": sha256_file(Path(receiver.__file__).resolve()),
        "stage_a": None,
        "cheap_to_shrink": {
            "mode": "off",
            "allocation_family": "waterfill_ceiling",
            "uniform_bits": [],
            "rung_weights": [],
            "base_weight": 1.0,
            "sampler_seed": SEED,
        },
    }


def verify_build(output: Path) -> dict[str, Any]:
    """Scorer-free behavioral build proof; retains every payload it materializes."""

    output.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    spec = receiver.StudentSpec("verify_flat_w8_d1", "flattened", 8, 1)
    model = receiver.StudentSemanticRenderer(spec)
    uniform = receiver.uniform_allocation(model, 4)
    packet = receiver.pack_student(model, uniform)
    parsed = receiver.unpack_student(packet)
    repacked = receiver.pack_student(parsed, uniform)
    if packet != repacked:
        raise WD3Error("build verification packet round-trip differs")
    adaptive_bits = {name: list(bits) for name, bits in uniform.bits.items()}
    first_name = sorted(adaptive_bits)[0]
    adaptive_bits[first_name][0] = 3
    adaptive = receiver.AdaptiveQuantizationAllocation(
        bits={name: tuple(bits) for name, bits in adaptive_bits.items()},
        selection_sha256="1" * 64,
        policy="verify_one_group_subint16",
    )
    adaptive_packet = receiver.pack_student(model, adaptive)
    if adaptive_packet == packet:
        raise WD3Error("adaptive allocation did not change counted packet bytes")
    payloads = {
        "uniform_packet": atomic_bytes(output / "retained/verify_uniform.wd3q", packet),
        "adaptive_packet": atomic_bytes(output / "retained/verify_adaptive.wd3q", adaptive_packet),
    }
    facts = {name: {"path": str(path), "bytes": size, "sha256": digest} for name, (path, size, digest) in PINS.items()}
    fire_order = compile_fire_order(blocked_config_template(), facts=facts, path_exists=lambda _path: False)
    if fire_order["disposition"] != "BLOCKED_NOT_LAUNCHABLE":
        raise WD3Error("sealed no-launch build unexpectedly compiled runnable")
    receipt = {
        "schema": "ddm_wd3_build_receipt.v1",
        "complete": True,
        "axis": "[scorer-free build and dry-run apparatus]",
        "score_claim": False,
        "promotion_eligible": False,
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
        "builder": file_record(Path(__file__).resolve()),
        "receiver": file_record(Path(receiver.__file__).resolve()),
        "design": file_record(DESIGN),
        "adaptive_packet_behavior": {
            "uniform_roundtrip_byte_identical": True,
            "adaptive_allocation_changes_counted_packet": True,
            "uniform_packet_bytes": len(packet),
            "adaptive_packet_bytes": len(adaptive_packet),
        },
        "payloads": payloads,
        "dry_run_fire_order": fire_order,
        "retention": "all materialized verification packets retained with bytes and SHA-256",
        "frontier_moved": False,
    }
    atomic_json(output / "BUILD_RECEIPT.json", receipt)
    return receipt


def inventory(output: Path) -> dict[str, Any]:
    retained = output / "retained"
    files = [file_record(path) for path in sorted(retained.rglob("*")) if path.is_file()]
    result = {
        "schema": "ddm_wd3_retention_inventory.v1",
        "root": str(retained.resolve()),
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "files": files,
        "cleanup_disposition": "KEEP; certify-or-block",
    }
    atomic_json(output / "RETENTION_INVENTORY.json", result)
    return result


def _load_compiled(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WD3Error("compiled config is not a JSON object")
    return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--request", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-build")
    verify.add_argument("--output", type=Path, required=True)
    birth = sub.add_parser("prepare-arm-birth")
    birth.add_argument("--compiled-config", type=Path, required=True)
    cache = sub.add_parser("prepare-teacher-scorer-cache")
    cache.add_argument("--compiled-config", type=Path, required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--compiled-config", type=Path, required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "compile":
        config = _load_compiled(args.request)
        fire_order = compile_fire_order(config)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "FIRE_ORDER.json", fire_order)
        if fire_order["disposition"] in {
            "READY_TO_FIRE",
            "READY_TO_MATERIALIZE_BUILD",
        }:
            atomic_json(args.output / "COMPILED_CONFIG.json", config)
        result = fire_order
    elif args.command == "verify-build":
        result = verify_build(args.output)
    elif args.command == "prepare-arm-birth":
        result = prepare_arm_birth(_load_compiled(args.compiled_config))
    elif args.command == "prepare-teacher-scorer-cache":
        result = prepare_teacher_scorer_cache(_load_compiled(args.compiled_config))
    elif args.command == "train":
        result = train(_load_compiled(args.compiled_config))
    else:
        result = inventory(args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"history", "files"}},
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
