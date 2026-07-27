#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize compact scorer costates in real actuator tangent coordinates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.optimization.taskspace_projected_population_costates_v1 import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    MAX_BATCH_PAIRS,
    PAIR_COUNT,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    PopulationScorePointV1,
    ProjectedPopulationCostateError,
    compute_batch_population_costates,
    exact_replay_projected_intervention,
    group_g72_batch_proposals,
    pareto_nondominated_projection_ids,
    realize_and_project_g72_group,
)
from tac.witness_control.taskspace_g87_g78_to_g72_exact_compile_adapter_v1 import (
    COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE,
    COMPLETE_MINIMUM_COMPONENT_SITES,
    G87ExactCompileAdapterError,
    open_g87_g78_to_g72_compile_input,
)
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
    G72BoundaryShearletProposalV1,
    G72StagePlanV1,
    derive_v9_boundary_shearlet_stage_proposals,
    reopen_stage_checkpoint,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    V15RoleAwareOverlayDecoderV1,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    CompactActuatorTypeV1,
    parse_compact_pvsa_member,
)

CONFIG_SCHEMA: Final = "tac.taskspace_projected_population_costates_config.v1"
BATCH_SCHEMA: Final = "tac.taskspace_projected_population_costate_batch.v1"
STAGE_SCHEMA: Final = "tac.taskspace_projected_population_costate_stage.v1"
AGGREGATE_SCHEMA: Final = "tac.taskspace_projected_population_costate_aggregate.v1"
BLOCKER_SCHEMA: Final = "tac.taskspace_projected_population_costate_blocker.v1"
STAGE_PAIRS: Final = 120
STAGE_COUNT: Final = 5
RAW_BYTES: Final = PAIR_COUNT * 2 * CAMERA_HEIGHT * CAMERA_WIDTH * 3


class ProjectedCostateMaterializerError(RuntimeError):
    """A typed input, runtime, resume, or authority invariant failed closed."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(value)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal(body: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise ProjectedCostateMaterializerError(f"{field} already exists")
    return {**body, field: _sha256_bytes(_canonical_json_bytes(body))}


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ProjectedCostateMaterializerError(f"immutable checkpoint differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectedCostateMaterializerError(f"{label} cannot be read") from exc
    if type(value) is not dict:
        raise ProjectedCostateMaterializerError(f"{label} is not one JSON object")
    return value


def _require_sha(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProjectedCostateMaterializerError(f"{label} is not canonical SHA-256")
    return value


def _require_file_identity(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"bytes", "path", "sha256"}:
        raise ProjectedCostateMaterializerError(f"{label} identity key set differs")
    path = value["path"]
    size = value["bytes"]
    if type(path) is not str or not path or type(size) is not int or size <= 0:
        raise ProjectedCostateMaterializerError(f"{label} identity values differ")
    _require_sha(value["sha256"], label=f"{label}.sha256")
    return dict(value)


@dataclass(frozen=True, slots=True)
class MaterializerConfigV1:
    path: Path
    output_root: Path
    seed: int
    num_threads: int
    safety_reserve_bytes: int
    source_video: dict[str, Any]
    g85_raw: dict[str, Any]
    g85_archive: dict[str, Any]
    semantic_archive: dict[str, Any]
    target_labels: dict[str, Any]
    segnet_weights: dict[str, Any]
    posenet_weights: dict[str, Any]
    upstream_members: tuple[dict[str, Any], ...]
    upstream_closure_sha256: str
    g78_aggregate: dict[str, Any]
    g78_aggregate_self_sha256: str
    g87_aggregate: dict[str, Any]
    g87_aggregate_self_sha256: str
    global_mean_pose_dist: float
    global_mean_seg_dist: float


@dataclass(frozen=True, slots=True)
class _G78StageDataV1:
    plan: G72StagePlanV1
    g78_stage_receipt_sha256: str
    target_cells_u8: np.ndarray
    target_margins_f32: np.ndarray
    described_cells_u8: np.ndarray


@dataclass(frozen=True, slots=True)
class _G78DataOnlyV1:
    stages: tuple[_G78StageDataV1, ...]


def load_config(path: Path) -> MaterializerConfigV1:
    """Load the one closed-key production configuration."""

    resolved = Path(path).expanduser().resolve()
    value = _load_mapping(resolved, label="G90 config")
    expected = {
        "g46_target_labels",
        "g78_aggregate",
        "g78_aggregate_self_sha256",
        "g85",
        "g87_aggregate",
        "g87_aggregate_self_sha256",
        "num_threads",
        "output_root",
        "posenet_weights",
        "safety_reserve_bytes",
        "schema",
        "seed",
        "segnet_weights",
        "semantic_archive",
        "source_video",
        "upstream_closure_sha256",
        "upstream_members",
    }
    if set(value) != expected or value.get("schema") != CONFIG_SCHEMA:
        raise ProjectedCostateMaterializerError("G90 config schema/key set differs")
    if type(value["output_root"]) is not str or not value["output_root"]:
        raise ProjectedCostateMaterializerError("output_root is not a path string")
    output_root = Path(value["output_root"]).expanduser().resolve()
    if not any(
        str(output_root).startswith(prefix)
        for prefix in (
            "/Volumes/VertigoDataTier/pact/",
            "/Volumes/APDataStore/pact/",
        )
    ):
        raise ProjectedCostateMaterializerError("production output must use the SSD waterfall")
    for field in ("seed", "num_threads", "safety_reserve_bytes"):
        if type(value[field]) is not int or value[field] < 1:
            raise ProjectedCostateMaterializerError(f"{field} must be one positive integer")
    g85 = value["g85"]
    if type(g85) is not dict or set(g85) != {
        "archive",
        "d_pose",
        "d_seg",
        "raw",
        "sample_count",
    }:
        raise ProjectedCostateMaterializerError("G85 score-point key set differs")
    if (
        g85["sample_count"] != PAIR_COUNT
        or not math.isfinite(float(g85["d_pose"]))
        or float(g85["d_pose"]) <= 0.0
        or not math.isfinite(float(g85["d_seg"]))
        or float(g85["d_seg"]) < 0.0
    ):
        raise ProjectedCostateMaterializerError("G85 exact score point differs")
    members = value["upstream_members"]
    if type(members) is not list or not members:
        raise ProjectedCostateMaterializerError("upstream closure members are empty")
    return MaterializerConfigV1(
        path=resolved,
        output_root=output_root,
        seed=value["seed"],
        num_threads=value["num_threads"],
        safety_reserve_bytes=value["safety_reserve_bytes"],
        source_video=_require_file_identity(value["source_video"], label="source video"),
        g85_raw=_require_file_identity(g85["raw"], label="G85 raw"),
        g85_archive=_require_file_identity(g85["archive"], label="G85 archive"),
        semantic_archive=_require_file_identity(
            value["semantic_archive"],
            label="semantic archive",
        ),
        target_labels=_require_file_identity(
            value["g46_target_labels"],
            label="G46 target labels",
        ),
        segnet_weights=_require_file_identity(
            value["segnet_weights"],
            label="SegNet weights",
        ),
        posenet_weights=_require_file_identity(
            value["posenet_weights"],
            label="PoseNet weights",
        ),
        upstream_members=tuple(_require_file_identity(row, label="upstream member") for row in members),
        upstream_closure_sha256=_require_sha(
            value["upstream_closure_sha256"],
            label="upstream closure",
        ),
        g78_aggregate=_require_file_identity(
            value["g78_aggregate"],
            label="G78 aggregate",
        ),
        g78_aggregate_self_sha256=_require_sha(
            value["g78_aggregate_self_sha256"],
            label="G78 aggregate self",
        ),
        g87_aggregate=_require_file_identity(
            value["g87_aggregate"],
            label="G87 aggregate",
        ),
        g87_aggregate_self_sha256=_require_sha(
            value["g87_aggregate_self_sha256"],
            label="G87 aggregate self",
        ),
        global_mean_pose_dist=float(g85["d_pose"]),
        global_mean_seg_dist=float(g85["d_seg"]),
    )


def _verify_identity(identity: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(identity["path"])).expanduser().resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != identity["bytes"]
        or _sha256_file(path) != identity["sha256"]
    ):
        raise ProjectedCostateMaterializerError(f"{label} exact bytes/SHA differ")
    return path


def _configure_determinism(config: MaterializerConfigV1) -> None:
    import torch

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False


def _open_g78_data_only(config: MaterializerConfigV1) -> _G78DataOnlyV1:
    """Reopen immutable scorer fields after a sealed runtime dependency drifts.

    This path is admitted only after this G90 output root already contains its
    immutable preflight proving the strict G78 loader closed.  It revalidates
    every aggregate, stage receipt, and field byte rather than treating current
    mutable receiver-source hashes as data custody.
    """

    aggregate_path = _verify_identity(config.g78_aggregate, label="G78 aggregate")
    aggregate = _load_mapping(aggregate_path, label="G78 aggregate")
    if (
        aggregate.get("aggregate_receipt_sha256") != config.g78_aggregate_self_sha256
        or len(aggregate.get("stages", [])) != STAGE_COUNT
    ):
        raise ProjectedCostateMaterializerError("G78 data-only aggregate custody differs")
    target_all = np.memmap(
        Path(config.target_labels["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, SCORER_HEIGHT, SCORER_WIDTH),
    )
    stages: list[_G78StageDataV1] = []
    for stage_index, binding in enumerate(aggregate["stages"]):
        start = stage_index * STAGE_PAIRS
        stop = start + STAGE_PAIRS
        if binding.get("stage_index") != stage_index or binding.get("pair_range") != [start, stop]:
            raise ProjectedCostateMaterializerError("G78 data-only stage binding differs")
        receipt_path = Path(str(binding["path"]))
        if receipt_path.stat().st_size != binding["bytes"] or _sha256_file(receipt_path) != binding["sha256"]:
            raise ProjectedCostateMaterializerError("G78 data-only stage receipt changed")
        receipt = _load_mapping(receipt_path, label="G78 stage receipt")
        if receipt.get("stage_receipt_sha256") != binding["stage_receipt_sha256"]:
            raise ProjectedCostateMaterializerError("G78 data-only stage self hash differs")
        files = receipt.get("files")
        if type(files) is not dict:
            raise ProjectedCostateMaterializerError("G78 data-only stage files differ")

        def field(
            name: str,
            dtype: np.dtype[Any],
            files_arg: dict[str, Any] = files,
        ) -> np.ndarray:
            identity = files_arg.get(name)
            if type(identity) is not dict:
                raise ProjectedCostateMaterializerError(f"G78 {name} identity missing")
            path = Path(str(identity["path"]))
            if (
                identity.get("shape") != [STAGE_PAIRS, SCORER_HEIGHT, SCORER_WIDTH]
                or path.stat().st_size != identity.get("bytes")
                or _sha256_file(path) != identity.get("sha256")
            ):
                raise ProjectedCostateMaterializerError(f"G78 {name} bytes differ")
            value = np.memmap(
                path,
                mode="r",
                dtype=dtype,
                shape=(STAGE_PAIRS, SCORER_HEIGHT, SCORER_WIDTH),
            )
            value.setflags(write=False)
            return value

        target_cells = target_all[start:stop]
        target_cells.setflags(write=False)
        stages.append(
            _G78StageDataV1(
                plan=G72StagePlanV1(
                    stage_index=stage_index,
                    pair_start=start,
                    pair_stop_exclusive=stop,
                ),
                g78_stage_receipt_sha256=binding["stage_receipt_sha256"],
                target_cells_u8=target_cells,
                target_margins_f32=field(
                    "target_margins_f32",
                    np.dtype("<f4"),
                ),
                described_cells_u8=field(
                    "described_cells_u8",
                    np.dtype(np.uint8),
                ),
            )
        )
    return _G78DataOnlyV1(stages=tuple(stages))


def _strict_preflight(config: MaterializerConfigV1) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(config.output_root)
    if usage.free <= config.safety_reserve_bytes:
        raise ProjectedCostateMaterializerError("SSD free bytes do not clear safety reserve")
    identities = {
        "source_video": _verify_identity(config.source_video, label="source video"),
        "g85_raw": _verify_identity(config.g85_raw, label="G85 raw"),
        "g85_archive": _verify_identity(config.g85_archive, label="G85 archive"),
        "semantic_archive": _verify_identity(
            config.semantic_archive,
            label="semantic archive",
        ),
        "target_labels": _verify_identity(
            config.target_labels,
            label="G46 target labels",
        ),
        "segnet_weights": _verify_identity(
            config.segnet_weights,
            label="SegNet weights",
        ),
        "posenet_weights": _verify_identity(
            config.posenet_weights,
            label="PoseNet weights",
        ),
    }
    if config.g85_raw["bytes"] != RAW_BYTES:
        raise ProjectedCostateMaterializerError("G85 raw size changed exact n600 camera ABI")
    for index, member in enumerate(config.upstream_members):
        _verify_identity(member, label=f"upstream member {index}")
    identities["g78_aggregate"] = _verify_identity(
        config.g78_aggregate,
        label="G78 aggregate",
    )
    identities["g87_aggregate"] = _verify_identity(
        config.g87_aggregate,
        label="G87 aggregate",
    )
    semantic_archive = identities["semantic_archive"].read_bytes()
    decoder = V15RoleAwareOverlayDecoderV1.open(
        semantic_archive,
        expected_archive_bytes=config.semantic_archive["bytes"],
        expected_archive_sha256=config.semantic_archive["sha256"],
        verify_member_effects=True,
    )
    try:
        g78 = open_g87_g78_to_g72_compile_input(
            identities["g78_aggregate"],
            expected_file_sha256=config.g78_aggregate["sha256"],
            expected_self_sha256=config.g78_aggregate_self_sha256,
        )
    except G87ExactCompileAdapterError:
        if not (config.output_root / "00_preflight_receipt.json").is_file():
            raise
        g78 = _open_g78_data_only(config)
    g87_receipt = _load_mapping(
        identities["g87_aggregate"],
        label="G87 aggregate",
    )
    g85_outer = parse_taskspace_outer_archive(
        identities["g85_archive"].read_bytes(),
        expected_archive_sha256=config.g85_archive["sha256"],
    )
    g85_pvsa = parse_compact_pvsa_member(
        g85_outer.member_bytes,
        maximum_member_bytes=len(g85_outer.member_bytes),
        maximum_section_bytes=len(g85_outer.member_bytes),
    )
    if (
        g85_pvsa.semantic_p_sha256 != config.semantic_archive["sha256"]
        or len(g85_pvsa.actuators) != 1
        or g85_pvsa.actuators[0].actuator_type is not CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT
        or g85_pvsa.actuators[0].operand is None
    ):
        raise ProjectedCostateMaterializerError(
            "G85 current base is not exact semantic P plus one incumbent G74 operand"
        )
    g85_receiver = g85_pvsa.open_receiver(verify_member_effects=True)
    if (
        g87_receipt.get("materialization_receipt_sha256") != config.g87_aggregate_self_sha256
        or g87_receipt.get("geometry")
        != {
            "class_count": 5,
            "pair_count": PAIR_COUNT,
            "scorer_batch_pairs": MAX_BATCH_PAIRS,
            "scorer_hw": [SCORER_HEIGHT, SCORER_WIDTH],
            "stage_count": STAGE_COUNT,
            "stage_pairs": STAGE_PAIRS,
        }
        or g87_receipt.get("five_g72_proposal_stages_materialized") is not True
        or len(g87_receipt.get("stages", [])) != STAGE_COUNT
    ):
        raise ProjectedCostateMaterializerError("G87 exact full-population custody differs")
    body = {
        "schema": "tac.taskspace_projected_population_costate_preflight.v1",
        "config": {
            "path": str(config.path),
            "bytes": config.path.stat().st_size,
            "sha256": _sha256_file(config.path),
        },
        "output_root": str(config.output_root),
        "seed": config.seed,
        "num_threads": config.num_threads,
        "geometry": {
            "pair_count": PAIR_COUNT,
            "batch_pairs_maximum": MAX_BATCH_PAIRS,
            "stage_pairs": STAGE_PAIRS,
            "stage_count": STAGE_COUNT,
            "camera_hw": [CAMERA_HEIGHT, CAMERA_WIDTH],
            "scorer_hw": [SCORER_HEIGHT, SCORER_WIDTH],
        },
        "score_point": {
            "d_pose": config.global_mean_pose_dist,
            "d_seg": config.global_mean_seg_dist,
            "archive_bytes": config.g85_archive["bytes"],
            "archive_sha256": config.g85_archive["sha256"],
            "sample_count": PAIR_COUNT,
        },
        "upstream_closure_sha256": config.upstream_closure_sha256,
        "source_identities": {
            "g85_raw": config.g85_raw,
            "g78_aggregate": config.g78_aggregate,
            "g78_aggregate_self_sha256": config.g78_aggregate_self_sha256,
            "g87_aggregate": config.g87_aggregate,
            "g87_aggregate_self_sha256": config.g87_aggregate_self_sha256,
            "g46_target_labels": config.target_labels,
            "semantic_archive": config.semantic_archive,
            "segnet_weights": config.segnet_weights,
            "posenet_weights": config.posenet_weights,
        },
        "storage": {
            "free_bytes_observed": usage.free,
            "safety_reserve_bytes": config.safety_reserve_bytes,
            "bulk_output_written": False,
            "dense_costates_persisted": False,
        },
        "resume": {
            "batch_checkpoint_policy": "immutable_atomic_every_batch",
            "stage_checkpoint_policy": "immutable_atomic_every_120_pairs",
            "prior_stage_overwrite_allowed": False,
        },
        "tangent_families": {
            "G72": "EXECUTABLE_EXACT_G74_DOUBLE_DECODE",
            "G88_XIP2_SE3": ("BLOCKED_FRESH_COUNTED_N600_XIP2_TRAJECTORY_AND_QUANTIZATION_SCALE_ABSENT"),
            "G89_CLASS_COMPLETE": "PLUGIN_SEAM_PRESENT_AWAITING_EXACT_OPERAND_CUSTODY",
        },
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    preflight_path = config.output_root / "00_preflight_receipt.json"
    if preflight_path.is_file():
        existing = _load_mapping(preflight_path, label="G90 preflight resume")
        existing_storage = existing.get("storage")
        if type(existing_storage) is not dict or type(existing_storage.get("free_bytes_observed")) is not int:
            raise ProjectedCostateMaterializerError("G90 preflight resume lost its storage observation")
        # Free space is a launch-time observation, not a reproducibility input.
        # Preserve the first immutable observation while rechecking the current
        # safety reserve above on every resume.
        body["storage"]["free_bytes_observed"] = existing_storage["free_bytes_observed"]
    preflight = _seal(body, field="preflight_sha256")
    _atomic_write_json(preflight_path, preflight)
    return {
        "preflight": preflight,
        "decoder": decoder,
        "g78": g78,
        "g87_receipt": g87_receipt,
        "g85_receiver": g85_receiver,
        "incumbent_atoms": g85_pvsa.actuators[0].operand.atoms,
        "incumbent_frame_selector": (g85_pvsa.actuators[0].operand.frame_selector),
    }


class _SourceCursor:
    def __init__(self, *, source: Path, batch_pairs: int, seed: int, num_threads: int) -> None:
        import torch

        upstream_root = source.parents[1]
        if str(upstream_root) not in sys.path:
            sys.path.insert(0, str(upstream_root))
        from frame_utils import AVVideoDataset

        dataset = AVVideoDataset(
            [source.name],
            data_dir=source.parent,
            batch_size=batch_pairs,
            device=torch.device("cpu"),
            num_threads=num_threads,
            seed=seed,
            prefetch_queue_depth=1,
        )
        dataset.prepare_data()
        self._iterator = iter(dataset)
        self._source = source.resolve()
        self._current: np.ndarray | None = None
        self._current_start = 0
        self._next_global_start = 0

    def _advance(self) -> None:
        try:
            path, _batch_index, batch = next(self._iterator)
        except StopIteration as exc:
            raise ProjectedCostateMaterializerError("AVVideoDataset ended before exact n600 custody") from exc
        if Path(path).resolve() != self._source:
            raise ProjectedCostateMaterializerError("AVVideoDataset source path drifted")
        value = np.ascontiguousarray(batch.cpu().numpy(), dtype=np.uint8)
        self._current = value
        self._current_start = self._next_global_start
        self._next_global_start += value.shape[0]

    def take(self, start: int, stop: int) -> np.ndarray:
        if not 0 <= start < stop <= PAIR_COUNT or stop - start > MAX_BATCH_PAIRS:
            raise ProjectedCostateMaterializerError("source request escaped scorer batch ABI")
        pieces: list[np.ndarray] = []
        cursor = start
        while cursor < stop:
            while self._current is None or cursor >= self._current_start + self._current.shape[0]:
                self._advance()
            assert self._current is not None
            if cursor < self._current_start:
                raise ProjectedCostateMaterializerError("source cursor requested nonmonotonic pairs")
            local_start = cursor - self._current_start
            local_stop = min(stop, self._current_start + self._current.shape[0]) - self._current_start
            pieces.append(self._current[local_start:local_stop])
            cursor += local_stop - local_start
        return np.ascontiguousarray(np.concatenate(pieces, axis=0), dtype=np.uint8)


def _load_models(config: MaterializerConfigV1):
    import torch
    from safetensors.torch import load_file

    upstream_root = Path(config.upstream_members[0]["path"]).resolve().parent
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))
    from modules import PoseNet, SegNet

    from tac.scorer import make_scorers_differentiable

    posenet = PoseNet().eval().to(torch.device("cpu"))
    segnet = SegNet().eval().to(torch.device("cpu"))
    posenet.load_state_dict(load_file(str(Path(config.posenet_weights["path"])), device="cpu"))
    segnet.load_state_dict(load_file(str(Path(config.segnet_weights["path"])), device="cpu"))
    make_scorers_differentiable(posenet, segnet)
    return posenet, segnet


def _stage_proposals(
    stage_row: Mapping[str, Any],
) -> tuple[G72BoundaryShearletProposalV1, ...]:
    checkpoint = reopen_stage_checkpoint(
        Path(str(stage_row["path"])),
        expected_checkpoint_sha256=str(stage_row["checkpoint_sha256"]),
    )
    if (
        _sha256_file(Path(str(stage_row["path"]))) != stage_row["sha256"]
        or checkpoint["pair_range"] != stage_row["pair_range"]
        or len(checkpoint["proposals"]) != stage_row["proposal_count"]
    ):
        raise ProjectedCostateMaterializerError("G87 stage file/proposal custody differs")
    rows: list[G72BoundaryShearletProposalV1] = []
    for source, expected_fingerprint in zip(
        checkpoint["proposals"],
        checkpoint["proposal_fingerprints"],
        strict=True,
    ):
        atom = BoundaryShearletAtomV1(**source["atom"])
        row = G72BoundaryShearletProposalV1(
            candidate_id=source["candidate_id"],
            fisher_priority=float(source["fisher_priority"]),
            atom=atom,
        )
        if row.fingerprint != expected_fingerprint:
            raise ProjectedCostateMaterializerError("G87 proposal fingerprint differs")
        rows.append(row)
    return tuple(rows)


def _seg_cells(segnet: Any, camera_pairs: np.ndarray) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(camera_pairs).copy()).permute(0, 1, 4, 2, 3).float().contiguous()
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(tensor))
    return np.ascontiguousarray(logits.argmax(dim=1).cpu().numpy(), dtype=np.uint8)


def _next_incomplete_stage(root: Path) -> int | None:
    for stage_index in range(STAGE_COUNT):
        path = (
            root
            / f"20_stage_{stage_index:02d}_{stage_index * STAGE_PAIRS:04d}_{(stage_index + 1) * STAGE_PAIRS:04d}"
            / "stage_receipt.json"
        )
        if not path.is_file():
            return stage_index
    return None


def _immutable_resume_frontier(root: Path) -> dict[str, Any]:
    """Return the verified contiguous batch frontier without trusting filenames."""

    sealed: list[dict[str, Any]] = []
    for stage_index in range(STAGE_COUNT):
        stage_start = stage_index * STAGE_PAIRS
        stage_stop = stage_start + STAGE_PAIRS
        stage_dir = root / f"20_stage_{stage_index:02d}_{stage_start:04d}_{stage_stop:04d}"
        for pair_start in range(stage_start, stage_stop, MAX_BATCH_PAIRS):
            pair_stop = min(pair_start + MAX_BATCH_PAIRS, stage_stop)
            path = stage_dir / "batches" / f"batch_{pair_start:04d}_{pair_stop:04d}.json"
            if not path.is_file():
                return {
                    "sealed_batch_count": len(sealed),
                    "sealed_batches": sealed,
                    "next_pair_range": [pair_start, pair_stop],
                }
            checkpoint = _load_mapping(path, label="G90 resume-frontier batch")
            expected_self = _sha256_bytes(
                _canonical_json_bytes(
                    {key: value for key, value in checkpoint.items() if key != "batch_checkpoint_sha256"}
                )
            )
            if (
                checkpoint.get("schema") != BATCH_SCHEMA
                or checkpoint.get("pair_range") != [pair_start, pair_stop]
                or checkpoint.get("batch_checkpoint_sha256") != expected_self
            ):
                raise ProjectedCostateMaterializerError(f"G90 resume-frontier checkpoint differs: {path}")
            sealed.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "batch_checkpoint_sha256": expected_self,
                    "pair_range": [pair_start, pair_stop],
                }
            )
    return {
        "sealed_batch_count": len(sealed),
        "sealed_batches": sealed,
        "next_pair_range": None,
    }


def _run_stage(config: MaterializerConfigV1, state: dict[str, Any], stage_index: int) -> Path:
    import torch

    stage_start = stage_index * STAGE_PAIRS
    stage_stop = stage_start + STAGE_PAIRS
    g78_stage = state["g78"].stages[stage_index]
    g87_stage_row = state["g87_receipt"]["stages"][stage_index]
    if (
        g78_stage.plan.pair_start != stage_start
        or g78_stage.plan.pair_stop_exclusive != stage_stop
        or g87_stage_row["pair_range"] != [stage_start, stage_stop]
    ):
        raise ProjectedCostateMaterializerError("G78/G87 stage ranges differ")
    donor_proposals = _stage_proposals(g87_stage_row)
    posenet, segnet = _load_models(config)
    source_cursor = _SourceCursor(
        source=Path(config.source_video["path"]),
        batch_pairs=MAX_BATCH_PAIRS,
        seed=config.seed,
        num_threads=config.num_threads,
    )
    if stage_start:
        source_cursor.take(stage_start - 1, stage_start)
    raw = np.memmap(
        Path(config.g85_raw["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, 2, CAMERA_HEIGHT, CAMERA_WIDTH, 3),
    )
    incumbent_atoms = tuple(state["incumbent_atoms"])
    incumbent_pair_ids = tuple(
        sorted({atom.pair_index for atom in incumbent_atoms if stage_start <= atom.pair_index < stage_stop})
    )
    current_described_cells = np.ascontiguousarray(g78_stage.described_cells_u8.copy())
    for offset in range(0, len(incumbent_pair_ids), MAX_BATCH_PAIRS):
        current_ids = incumbent_pair_ids[offset : offset + MAX_BATCH_PAIRS]
        current_camera = np.ascontiguousarray(raw[list(current_ids)])
        realized_current = state["g85_receiver"].render_camera_pair_batch(current_ids)
        if not np.array_equal(realized_current, current_camera):
            raise ProjectedCostateMaterializerError("exact G85 PVSA receiver differs from exact G85 raw current base")
        current_cells = _seg_cells(segnet, current_camera)
        for local_index, pair_id in enumerate(current_ids):
            current_described_cells[pair_id - stage_start] = current_cells[local_index]
    if incumbent_pair_ids:
        proposals = derive_v9_boundary_shearlet_stage_proposals(
            stage=g78_stage.plan,
            target_cells=g78_stage.target_cells_u8,
            target_margins=g78_stage.target_margins_f32,
            described_cells=current_described_cells,
            minimum_component_sites=COMPLETE_MINIMUM_COMPONENT_SITES,
            maximum_components_per_pair_role=(COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
        )
        incumbent_pair_set = set(incumbent_pair_ids)
        donor_unaffected = tuple(row for row in donor_proposals if row.atom.pair_index not in incumbent_pair_set)
        current_unaffected = tuple(row for row in proposals if row.atom.pair_index not in incumbent_pair_set)
        if donor_unaffected != current_unaffected:
            raise ProjectedCostateMaterializerError("current-base proposal regeneration changed a G87-unaffected pair")
    else:
        proposals = donor_proposals
    labels = np.memmap(
        Path(config.target_labels["path"]),
        mode="r",
        dtype=np.uint8,
        shape=(PAIR_COUNT, SCORER_HEIGHT, SCORER_WIDTH),
    )
    score_point = PopulationScorePointV1(
        global_mean_pose_dist=config.global_mean_pose_dist,
        sample_count=PAIR_COUNT,
        archive_bytes=config.g85_archive["bytes"],
        archive_sha256=config.g85_archive["sha256"],
    )
    stage_dir = config.output_root / f"20_stage_{stage_index:02d}_{stage_start:04d}_{stage_stop:04d}"
    batch_rows: list[dict[str, Any]] = []
    for pair_start in range(stage_start, stage_stop, MAX_BATCH_PAIRS):
        pair_stop = min(pair_start + MAX_BATCH_PAIRS, stage_stop)
        checkpoint_path = stage_dir / "batches" / f"batch_{pair_start:04d}_{pair_stop:04d}.json"
        if checkpoint_path.is_file():
            checkpoint = _load_mapping(checkpoint_path, label="G90 batch checkpoint")
            if (
                checkpoint.get("schema") != BATCH_SCHEMA
                or checkpoint.get("pair_range") != [pair_start, pair_stop]
                or checkpoint.get("batch_checkpoint_sha256")
                != _sha256_bytes(
                    _canonical_json_bytes(
                        {key: value for key, value in checkpoint.items() if key != "batch_checkpoint_sha256"}
                    )
                )
            ):
                raise ProjectedCostateMaterializerError("G90 resume checkpoint differs")
            source_cursor.take(pair_start, pair_stop)
            batch_rows.append(
                {
                    "path": str(checkpoint_path),
                    "bytes": checkpoint_path.stat().st_size,
                    "sha256": _sha256_file(checkpoint_path),
                    "batch_checkpoint_sha256": checkpoint["batch_checkpoint_sha256"],
                    "pair_range": [pair_start, pair_stop],
                }
            )
            continue
        pair_ids = tuple(range(pair_start, pair_stop))
        local_start = pair_start - stage_start
        local_stop = pair_stop - stage_start
        target = source_cursor.take(pair_start, pair_stop)
        base = np.ascontiguousarray(raw[pair_start:pair_stop])
        target_cells = np.ascontiguousarray(g78_stage.target_cells_u8[local_start:local_stop])
        described_cells = np.ascontiguousarray(current_described_cells[local_start:local_stop])
        if not np.array_equal(target_cells, labels[pair_start:pair_stop]):
            raise ProjectedCostateMaterializerError("G78 target cells differ from exact G46 labels")
        costates = compute_batch_population_costates(
            candidate_pairs_hwc=base,
            target_pairs_hwc=target,
            target_cells=target_cells,
            described_cells=described_cells,
            pair_ids=pair_ids,
            posenet=posenet,
            segnet=segnet,
            device="cpu",
            score_point=score_point,
        )
        groups = group_g72_batch_proposals(proposals, pair_ids=pair_ids)
        projected = []
        realized: dict[str, np.ndarray] = {}
        for group in groups:
            row, candidate = realize_and_project_g72_group(
                decoder=state["decoder"],
                group=group,
                base_camera_pairs=base,
                costates=costates,
                incumbent_atoms=incumbent_atoms,
                incumbent_frame_selector=state["incumbent_frame_selector"],
            )
            projected.append(row)
            realized[row.operand_id] = candidate
        pareto_ids = pareto_nondominated_projection_ids(tuple(projected))
        replayed = []
        for row in projected:
            if row.operand_id in pareto_ids:
                row = exact_replay_projected_intervention(
                    row,
                    candidate_pairs_hwc=realized[row.operand_id],
                    target_cells=target_cells,
                    costates=costates,
                    posenet=posenet,
                    segnet=segnet,
                    device="cpu",
                )
            replayed.append(row)
        replayed_by_id = {row.operand_id: row for row in replayed}
        basis_groups = []
        for group in groups:
            projected_row = replayed_by_id[group.group_id]
            basis_groups.append(
                {
                    "group_id": group.group_id,
                    "role": group.role,
                    "direction_rank": group.direction_rank,
                    "amplitude_scale": group.amplitude_scale,
                    "proposed_atoms_sha256": (projected_row.proposed_atoms_sha256),
                    "incumbent_atoms_sha256": (projected_row.incumbent_atoms_sha256),
                    "proposals": [proposal.to_dict() for proposal in group.proposals],
                    "proposal_fingerprints": [proposal.fingerprint for proposal in group.proposals],
                }
            )
        body = {
            "schema": BATCH_SCHEMA,
            "pair_range": [pair_start, pair_stop],
            "source_custody": {
                "candidate_camera_sha256": costates.candidate_sha256,
                "target_camera_sha256": costates.target_sha256,
                "target_cells_sha256": costates.target_cells_sha256,
                "described_cells_sha256": costates.described_cells_sha256,
                "g78_p_only_described_cells_changed_for_current_g85_base": (
                    int(np.count_nonzero(described_cells != g78_stage.described_cells_u8[local_start:local_stop]))
                ),
            },
            "base_components": {
                "pair_pose_mse_f32": [float(value) for value in costates.base_pair_pose_mse],
                "seg_mismatch_count": costates.base_mismatch_count,
                "target_minus_current_gap_sum": costates.base_gap_sum,
            },
            "population_pose_pair_mse_vjp_scale": (score_point.pair_pose_mse_vjp_scale),
            "projection_coordinate_count": len(replayed),
            "projection_rows": [row.to_dict() for row in replayed],
            "actuator_basis_groups": basis_groups,
            "actuator_basis_reconstructs_exact_measured_proposed_atoms": True,
            "incumbent_and_proposed_atom_custody_separate": True,
            "pareto_nondominated_operand_ids": list(pareto_ids),
            "dense_costates_persisted": False,
            "actual_zip_delta_measured": False,
            "member_bytes_used_as_rate": False,
            "local_admission_performed": False,
            "selection_consumer": "G83_WHOLE_STATE_ALLOCATOR_ONLY",
            "candidate_claim": False,
            "score_claim": False,
            "research_only": True,
            "encoder_only": True,
        }
        checkpoint = _seal(body, field="batch_checkpoint_sha256")
        _atomic_write_json(checkpoint_path, checkpoint)
        batch_rows.append(
            {
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256_file(checkpoint_path),
                "batch_checkpoint_sha256": checkpoint["batch_checkpoint_sha256"],
                "pair_range": [pair_start, pair_stop],
            }
        )
        del costates, realized, target, base
        gc.collect()
        print(
            json.dumps(
                {
                    "status": "batch_complete",
                    "pair_range": [pair_start, pair_stop],
                    "projection_count": len(replayed),
                    "pareto_count": len(pareto_ids),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    base_pose_sum = 0.0
    base_seg_errors = 0
    projection_count = 0
    for binding in batch_rows:
        checkpoint = _load_mapping(Path(binding["path"]), label="G90 batch checkpoint")
        base_pose_sum += float(
            np.asarray(
                checkpoint["base_components"]["pair_pose_mse_f32"],
                dtype=np.float32,
            ).sum(dtype=np.float32)
        )
        base_seg_errors += int(checkpoint["base_components"]["seg_mismatch_count"])
        projection_count += int(checkpoint["projection_coordinate_count"])
    body = {
        "schema": STAGE_SCHEMA,
        "stage_index": stage_index,
        "pair_range": [stage_start, stage_stop],
        "batch_count": len(batch_rows),
        "batches": batch_rows,
        "base_pose_squared_error_sum_f32": base_pose_sum,
        "base_segmentation_error_count": base_seg_errors,
        "projection_coordinate_count": projection_count,
        "g78_stage_receipt_sha256": g78_stage.g78_stage_receipt_sha256,
        "g87_stage_checkpoint_sha256": g87_stage_row["checkpoint_sha256"],
        "checkpoint_policy": "immutable_atomic_preserve_every_120_pair_stage",
        "dense_costates_persisted": False,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    receipt = _seal(body, field="stage_receipt_sha256")
    path = stage_dir / "stage_receipt.json"
    _atomic_write_json(path, receipt)
    del posenet, segnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def _write_aggregate_if_complete(config: MaterializerConfigV1) -> Path | None:
    stages: list[dict[str, Any]] = []
    pose_sum = 0.0
    seg_errors = 0
    projection_count = 0
    for stage_index in range(STAGE_COUNT):
        start = stage_index * STAGE_PAIRS
        path = (
            config.output_root
            / f"20_stage_{stage_index:02d}_{start:04d}_{start + STAGE_PAIRS:04d}"
            / "stage_receipt.json"
        )
        if not path.is_file():
            return None
        value = _load_mapping(path, label="G90 stage receipt")
        stages.append(
            {
                "stage_index": stage_index,
                "pair_range": value["pair_range"],
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "stage_receipt_sha256": value["stage_receipt_sha256"],
            }
        )
        pose_sum += float(value["base_pose_squared_error_sum_f32"])
        seg_errors += int(value["base_segmentation_error_count"])
        projection_count += int(value["projection_coordinate_count"])
    d_pose = pose_sum / PAIR_COUNT
    d_seg = seg_errors / (PAIR_COUNT * SCORER_HEIGHT * SCORER_WIDTH)
    exact_base_reproduced = round(d_pose, 8) == round(config.global_mean_pose_dist, 8) and round(d_seg, 8) == round(
        config.global_mean_seg_dist, 8
    )
    if not exact_base_reproduced:
        raise ProjectedCostateMaterializerError(
            "full-n600 base components do not reproduce the exact G85 row to reported precision"
        )
    body = {
        "schema": AGGREGATE_SCHEMA,
        "pair_range": [0, PAIR_COUNT],
        "stages": stages,
        "base_row": {
            "d_pose": d_pose,
            "d_seg": d_seg,
            "archive_bytes": config.g85_archive["bytes"],
            "archive_sha256": config.g85_archive["sha256"],
            "exact_g85_components_reproduced_to_reported_precision": True,
        },
        "projection_coordinate_count": projection_count,
        "dense_costates_persisted": False,
        "tangent_families": {
            "G72": "COMPLETE",
            "G88_XIP2_SE3": ("BLOCKED_FRESH_COUNTED_N600_XIP2_TRAJECTORY_AND_QUANTIZATION_SCALE_ABSENT"),
            "G89_CLASS_COMPLETE": "PLUGIN_SEAM_AWAITING_EXACT_OPERAND_CUSTODY",
        },
        "rate_axis": "UNMEASURED_UNTIL_G83_COMPOSES_ACTUAL_ZIP",
        "selection_consumer": "G83_WHOLE_STATE_ALLOCATOR_ONLY",
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    receipt = _seal(body, field="aggregate_receipt_sha256")
    path = config.output_root / "aggregate_receipt.json"
    _atomic_write_json(path, receipt)
    return path


def _write_blocker(config: MaterializerConfigV1, exc: BaseException) -> Path:
    exception_context = getattr(exc, "context", {})
    if type(exception_context) is not dict:
        exception_context = {}
    try:
        resume_frontier: dict[str, Any] = _immutable_resume_frontier(config.output_root)
        resume_frontier_error = None
    except Exception as frontier_exc:
        resume_frontier = {
            "sealed_batch_count": None,
            "sealed_batches": [],
            "next_pair_range": None,
        }
        resume_frontier_error = {
            "exception_type": type(frontier_exc).__name__,
            "exception_message": str(frontier_exc),
        }
    is_argmax_drift = (
        type(exc) is ProjectedPopulationCostateError
        and str(exc) == "current-base SegNet argmax differs from fresh G78 described cells"
    )
    body = {
        "schema": BLOCKER_SCHEMA,
        "config_path": str(config.path),
        "output_root": str(config.output_root),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_context": exception_context,
        "immutable_resume_frontier": resume_frontier,
        "resume_frontier_error": resume_frontier_error,
        "deterministic_retry_classification": {
            "class": (
                "CONDITIONAL_SINGLE_RETRY_AFTER_ISOLATED_EXACT_RECHECK"
                if is_argmax_drift
                else "FAIL_CLOSED_NO_AUTOMATIC_RETRY"
            ),
            "automatic_retry_allowed": False,
            "required_gate": (
                "same-config isolated scorer recheck must produce zero mismatches "
                "and exact expected cell SHA before one stage-only retry"
                if is_argmax_drift
                else None
            ),
        },
        "dense_costates_persisted": False,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
    }
    receipt = _seal(body, field="blocker_receipt_sha256")
    path = config.output_root / "blocker_receipt.json"
    payload = _canonical_json_bytes(receipt)
    if path.exists() and path.read_bytes() != payload:
        path = config.output_root / f"blocker_receipt_{receipt['blocker_receipt_sha256'][:12]}.json"
    _atomic_write_json(path, receipt)
    return path


def run_next_stage(config: MaterializerConfigV1) -> dict[str, Any]:
    """Materialize or resume exactly one immutable 120-pair production stage."""

    _configure_determinism(config)
    state = _strict_preflight(config)
    stage_index = _next_incomplete_stage(config.output_root)
    if stage_index is None:
        aggregate = _write_aggregate_if_complete(config)
        return {"status": "already_complete", "aggregate_receipt": str(aggregate)}
    stage_path = _run_stage(config, state, stage_index)
    aggregate = _write_aggregate_if_complete(config)
    return {
        "status": "stage_complete",
        "stage_index": stage_index,
        "stage_receipt": str(stage_path),
        "aggregate_receipt": None if aggregate is None else str(aggregate),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize-next-stage", action="store_true")
    mode.add_argument("--status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.status:
        next_stage = _next_incomplete_stage(config.output_root)
        print(
            json.dumps(
                {
                    "status": "complete" if next_stage is None else "incomplete",
                    "next_stage": next_stage,
                    "output_root": str(config.output_root),
                },
                sort_keys=True,
            )
        )
        return 0
    try:
        result = run_next_stage(config)
    except Exception as exc:
        blocker = _write_blocker(config, exc)
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blocker_receipt": str(blocker),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
