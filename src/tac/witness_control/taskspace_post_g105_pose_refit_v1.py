# SPDX-License-Identifier: MIT
"""Exact post-G105 generated-Y1 PoseNet inverse refit producer.

This module closes the producer side of
``taskspace_g110_generated_y1_pose_product_v1._verify_post_g105_refit``.
It does not train an RGB codec and it does not optimize an unparsed semantic
surrogate.  Its only source frame is the exact parsed G105 Y1 program, realized
through the public V10 factor-2 selected-preimage map.  Its only conditional
actuator is the counted XIP2 trajectory that the public receiver uses to warp
that camera-space Y1 into camera-space Y0.

The important state variable is the *shipped* integer XIP2 population, not an
unquantized per-pair twist.  XIP2 uses one maximum-derived scale per channel;
therefore a local edit can otherwise move every decoded pair.  The solver fixes
one deterministic extremum anchor per nonzero channel, optimizes the remaining
integer coordinates with exact finite-difference PoseNet Jacobians, and
arbitrates several q-level populations by the exact complete G110 archive
bytes plus the exact PoseNet score term.  This makes the local inverse problems
separable without pretending the global quantizer coupling does not exist.

Production is exact n600, frozen-upstream, batch-16 (with the canonical final
partial batch of eight), deterministic, SSD-only, governed, and resumable.
Every batch produces an immutable complete optimizer-state NPZ.  The final NPZ
and run receipt use exactly the schemas consumed by G110.  No score, candidate,
or pointer claim is made here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

from tac.boundary_math.warp_real_luma_frame0 import (
    GroundHomographyGeom,
    warp_frame0_uint8_numpy,
)
from tac.boundary_math.xi_pose_coder import (
    dequantize_xi,
    quantize_xi,
    serialize_xi_payload,
)
from tac.optimization.uint8_lattice_feasibility import (
    DisjointResizeOperator,
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_control.taskspace_v9_training_target_capsule_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_PAIR_COUNT,
    SSD_ROOTS,
    V9TrainingTargetCapsuleLoaderV1,
    file_identity,
    sha256_file,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    Y1WireCodecV1,
    encode_packet_y1_variants,
)
from tac.witness_dsl.taskspace_g105_exact_v9_semantic_root_adapter_v1 import (
    parse_packet as parse_v9_packet,
)
from tac.witness_dsl.taskspace_g110_generated_y1_pose_product_v1 import (
    _OUTER_ZIP_METHODS,
    CAMERA_H,
    CAMERA_W,
    CHANNELS,
    POST_G105_REFIT_CHECKPOINT_SCHEMA,
    POST_G105_REFIT_RUN_SCHEMA,
    RENDER_ORDER,
    SOURCE_DOMAIN,
    G110CompleteArchiveWireCandidateV1,
    G110G112CompileCustodyV1,
    _build_g110_archive_for_method,
    _final_y1_binding_from_population,
    _population_digest,
    _read_g110_archive_member,
    _xi_digest,
    parse_g110_generated_y1_pose_v1,
)
from tac.witness_dsl.taskspace_g110_generated_y1_pose_product_v1 import (
    _encode_packet as _encode_g110_packet,
)
from tac.witness_dsl.taskspace_g110_generic_two_layer_public_product_v1 import (
    OpenedFinalY1ProviderV1,
    final_y1_binding_sha256,
    open_final_y1_provider,
)
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    SCHEMA as V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
)
from tac.witness_dsl.v10_factor2_selected_preimage_v1 import (
    build_numpy_factor2_gather_plan,
)

CONFIG_SCHEMA: Final = "tac.post_g105_generated_y1_pose_refit_config.v1"
STATE_SCHEMA: Final = "tac.post_g105_generated_y1_pose_refit_optimizer_state.v1"
CANDIDATE_SCHEMA: Final = "tac.post_g105_generated_y1_pose_refit_candidate.v1"
AUDIT_SCHEMA: Final = "tac.post_g105_generated_y1_pose_refit_audit.v1"
BLOCKER_SCHEMA: Final = "tac.post_g105_generated_y1_pose_refit_blocker.v1"
GLOBAL_RANGE_BLOCKER_SCHEMA: Final = "tac.post_g105_generated_y1_pose_refit_global_range_reactivation_blocker.v1"
OPTIMIZER_VERDICT_SCOPE: Final = "fixed_extremum_gauge_integer_q_local_formulation"
PAIR_COUNT: Final = PRODUCTION_PAIR_COUNT
BATCH_PAIRS: Final = PRODUCTION_BATCH_PAIRS
POSE_DIM: Final = 6
ARCHIVE_DENOMINATOR_BYTES: Final = 37_545_489
MIN_FREE_BYTES: Final = 8 << 30
GOVERNED_MARKER_ENV: Final = "TAC_GOVERNED_ADMISSION"
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

_OPTIMIZER_STATE_MEMBERS: Final = frozenset(
    {
        "schema",
        "checkpoint_kind",
        "config_sha256",
        "run_id",
        "seed",
        "source_contract",
        "render_order",
        "y1_selected_preimage_schema",
        "source_g112_partition_receipt_sha256",
        "source_g112_semantic_child_sha256",
        "source_g112_pose_initializer_sha256",
        "semantic_packet_sha256",
        "final_y1_binding_sha256",
        "xi_initializer_sha256",
        "target_capsule_receipt_sha256",
        "pose_targets_sha256",
        "posenet_weights_sha256",
        "q_levels",
        "stage_index",
        "next_pair",
        "q",
        "scales",
        "xi_eff",
        "pose_outputs",
        "pose_losses",
        "anchor_rows",
        "controller_json",
        "numpy_rng_state_json",
        "torch_rng_state_u8",
        "torch_cuda_rng_states_u8",
    }
)
_CANDIDATE_STATE_MEMBERS: Final = frozenset(
    {
        "schema",
        "checkpoint_kind",
        "config_sha256",
        "q_levels",
        "q",
        "scales",
        "xi_eff",
        "pose_outputs",
        "pose_losses",
        "anchor_rows",
    }
)
_CANDIDATE_RECEIPT_MEMBERS: Final = frozenset(
    {
        "schema",
        "run_id",
        "config_sha256",
        "q_levels",
        "pair_count",
        "batch_pairs",
        "source_contract",
        "render_order",
        "semantic_packet_sha256",
        "final_y1_binding_sha256",
        "source_g112_partition_receipt_sha256",
        "target_capsule_receipt_sha256",
        "pose_targets_sha256",
        "posenet_weights_sha256",
        "xi_eff_sha256",
        "xip2_bytes",
        "xip2_sha256",
        "pose_mse",
        "pose_term",
        "complete_archive_bytes",
        "complete_archive_sha256",
        "product_packet_bytes",
        "product_packet_sha256",
        "selected_y1_wire_codec",
        "selected_outer_zip_method",
        "complete_archive_wire_candidates",
        "selection_objective_pose_plus_rate",
        "rate_term",
        "accepted_rows",
        "attempted_rows",
        "fixed_global_quantizer_anchor_rows",
        "optimizer_verdict_scope",
        "global_xip2_range_optimality_claim",
        "global_range_reactivation_required",
        "global_range_reactivation_blocker",
        "candidate_state",
        "exact_public_receiver_in_loop",
        "research_only",
        "candidate_claim",
        "score_claim",
        "pointer_moved",
        "candidate_receipt_sha256",
    }
)


class PostG105PoseRefitError(RuntimeError):
    """A production input, exact solver, or durable artifact failed closed."""


def _sha256(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PostG105PoseRefitError("value is not finite canonical ASCII JSON") from exc


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PostG105PoseRefitError(f"{name} must be canonical lowercase SHA-256")
    return value


def _seal(body: Mapping[str, object], *, field: str) -> dict[str, object]:
    if field in body:
        raise PostG105PoseRefitError(f"unsealed body already contains {field}")
    result = dict(body)
    result[field] = _sha256(_canonical_json(body))
    return result


def global_range_reactivation_blocker() -> dict[str, object]:
    """Describe the deliberately unclosed outer XIP2 range/reanchor family.

    Fixed extrema make the shipped integer-q local inverse problems honest and
    separable, but they do not prove that the initializer-derived per-channel
    ranges are globally optimal.  Every durable result carries this blocker so
    a strong local row cannot silently kill the outer scale/reanchor family.
    """

    return {
        "schema": GLOBAL_RANGE_BLOCKER_SCHEMA,
        "blocked_claim": "global_xip2_range_optimality",
        "optimizer_verdict_scope": OPTIMIZER_VERDICT_SCOPE,
        "reason": (
            "No deterministic outer per-channel XIP2 range/reanchor family "
            "was evaluated; initializer-derived scales and one extremum per "
            "channel remain fixed."
        ),
        "required_reactivation": (
            "Enumerate deterministic per-channel range/reanchor proposals, "
            "refit all shipped q coordinates through the same exact public "
            "PoseNet loop, and arbitrate exact complete archive bytes."
        ),
        "candidate_result_valid_within_scope": True,
        "global_optimality_claim": False,
    }


def _strict_file_binding(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise PostG105PoseRefitError(f"{name} binding key set differs")
    candidate = Path(str(value["path"])).expanduser()
    if candidate.is_symlink():
        raise PostG105PoseRefitError(f"{name} must not be a symlink")
    path = candidate.resolve()
    if not path.is_file():
        raise PostG105PoseRefitError(f"{name} is not a regular file")
    observed = file_identity(path)
    expected = {
        "path": str(path),
        "bytes": value["bytes"],
        "sha256": _require_sha256(value["sha256"], name=name),
    }
    if type(value["bytes"]) is not int or value["bytes"] < 0 or observed != expected or value["path"] != str(path):
        raise PostG105PoseRefitError(f"{name} physical identity differs")
    return expected


def _output_binding(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _npy_bytes(value: np.ndarray) -> bytes:
    handle = io.BytesIO()
    array = np.asarray(value)
    if array.ndim:
        array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise PostG105PoseRefitError("object arrays are forbidden in checkpoints")
    np.lib.format.write_array(
        handle,
        array,
        version=(1, 0),
        allow_pickle=False,
    )
    return handle.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    if not arrays:
        raise PostG105PoseRefitError("checkpoint cannot be empty")
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            if type(name) is not str or not name or not name.isascii() or "/" in name or "\\" in name:
                raise PostG105PoseRefitError("checkpoint member name is not flat bounded ASCII")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return output.getvalue()


def _write_once(path: Path, payload: bytes) -> None:
    """Atomically create an immutable file, accepting only byte-equal resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise PostG105PoseRefitError("immutable output path contains a symlink")
    scratch = path.parent / ".scratch"
    scratch.mkdir(mode=0o700, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".partial",
        dir=scratch,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise PostG105PoseRefitError(f"immutable output already contains different bytes: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_once(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    _write_once(path, payload)


@dataclass(frozen=True, slots=True)
class PostG105PoseRefitConfigV1:
    config_path: Path
    config_sha256: str
    run_id: str
    seed: int
    output_root: Path
    g112_partition_receipt: dict[str, object]
    target_capsule_receipt: dict[str, object]
    q_levels_candidates: tuple[int, ...]
    local_gauss_newton_stages: int
    finite_difference_q_steps: int
    damping: float
    trust_radius_q: int
    line_search_scales: tuple[float, ...]
    device: str
    torch_num_threads: int


def seal_config(body: Mapping[str, object]) -> dict[str, object]:
    """Return the strict self-hashed production config document."""

    return _seal(body, field="config_sha256")


def load_config(
    path: Path,
    *,
    allowed_output_roots: Sequence[Path] = SSD_ROOTS,
) -> PostG105PoseRefitConfigV1:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise PostG105PoseRefitError("config must not be a symlink")
    resolved = candidate.resolve()
    try:
        raw = resolved.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostG105PoseRefitError("config is not readable strict JSON") from exc
    keys = {
        "schema",
        "run_id",
        "seed",
        "output_root",
        "g112_partition_receipt",
        "target_capsule_receipt",
        "q_levels_candidates",
        "local_gauss_newton_stages",
        "finite_difference_q_steps",
        "damping",
        "trust_radius_q",
        "line_search_scales",
        "device",
        "torch_num_threads",
        "research_only",
        "candidate_claim",
        "score_claim",
        "pointer_moved",
        "config_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise PostG105PoseRefitError("config key set differs")
    expected_sha = _require_sha256(value["config_sha256"], name="config")
    if _sha256(_canonical_json({key: item for key, item in value.items() if key != "config_sha256"})) != expected_sha:
        raise PostG105PoseRefitError("config self-hash differs")
    if (
        value["schema"] != CONFIG_SCHEMA
        or type(value["run_id"]) is not str
        or _RUN_ID.fullmatch(value["run_id"]) is None
        or type(value["seed"]) is not int
        or value["research_only"] is not True
        or value["candidate_claim"] is not False
        or value["score_claim"] is not False
        or value["pointer_moved"] is not False
    ):
        raise PostG105PoseRefitError("config identity or false-authority fences differ")
    output = Path(str(value["output_root"])).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise PostG105PoseRefitError("output_root must be an absolute non-symlink")
    output = output.resolve()
    roots = tuple(root.expanduser().resolve() for root in allowed_output_roots)
    if not roots or not any(output == root or root in output.parents for root in roots):
        raise PostG105PoseRefitError("production output_root is outside the SSD storage waterfall")
    parent = next((item for item in (output, *output.parents) if item.exists()), None)
    if parent is None or parent.is_symlink() or not parent.is_dir():
        raise PostG105PoseRefitError("output_root has no existing real parent")
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE_BYTES:
        raise PostG105PoseRefitError(f"storage preflight needs {MIN_FREE_BYTES} bytes; observed {free}")
    q_levels_raw = value["q_levels_candidates"]
    if (
        type(q_levels_raw) is not list
        or not 1 <= len(q_levels_raw) <= 8
        or any(type(item) is not int or not 1 <= item <= 32_767 for item in q_levels_raw)
        or len(q_levels_raw) != len(set(q_levels_raw))
        or q_levels_raw != sorted(q_levels_raw)
    ):
        raise PostG105PoseRefitError("q_levels_candidates must be 1..8 unique ascending int16 levels")
    lines = value["line_search_scales"]
    if (
        type(lines) is not list
        or not 1 <= len(lines) <= 8
        or any(
            type(item) not in {int, float} or not math.isfinite(float(item)) or not 0.0 < float(item) <= 1.0
            for item in lines
        )
        or float(lines[0]) != 1.0
        or any(float(lines[index]) <= float(lines[index + 1]) for index in range(len(lines) - 1))
    ):
        raise PostG105PoseRefitError("line_search_scales must be finite, descending, and start at 1")
    if (
        type(value["local_gauss_newton_stages"]) is not int
        or not 1 <= value["local_gauss_newton_stages"] <= 8
        or type(value["finite_difference_q_steps"]) is not int
        or not 1 <= value["finite_difference_q_steps"] <= 1024
        or type(value["trust_radius_q"]) is not int
        or not 1 <= value["trust_radius_q"] <= 4096
        or type(value["damping"]) not in {int, float}
        or not math.isfinite(float(value["damping"]))
        or float(value["damping"]) <= 0.0
        or value["device"] not in {"cpu", "cuda"}
        or type(value["torch_num_threads"]) is not int
        or not 1 <= value["torch_num_threads"] <= 64
    ):
        raise PostG105PoseRefitError("solver/controller config is outside bounds")
    return PostG105PoseRefitConfigV1(
        config_path=resolved,
        config_sha256=expected_sha,
        run_id=value["run_id"],
        seed=value["seed"],
        output_root=output,
        g112_partition_receipt=_strict_file_binding(
            value["g112_partition_receipt"],
            name="G112 partition receipt",
        ),
        target_capsule_receipt=_strict_file_binding(
            value["target_capsule_receipt"],
            name="G109 target capsule receipt",
        ),
        q_levels_candidates=tuple(q_levels_raw),
        local_gauss_newton_stages=value["local_gauss_newton_stages"],
        finite_difference_q_steps=value["finite_difference_q_steps"],
        damping=float(value["damping"]),
        trust_radius_q=value["trust_radius_q"],
        line_search_scales=tuple(float(item) for item in lines),
        device=value["device"],
        torch_num_threads=value["torch_num_threads"],
    )


@dataclass(frozen=True, slots=True)
class PostG105RefitCustodyV1:
    base: G110G112CompileCustodyV1
    target_loader: V9TrainingTargetCapsuleLoaderV1
    semantic_packet: bytes
    provider: OpenedFinalY1ProviderV1
    final_y1_population_digest: bytes
    final_y1_binding_sha256: str
    pose_targets: np.ndarray
    upstream_root: Path
    posenet_binding: dict[str, object]


def open_custody(config: PostG105PoseRefitConfigV1) -> PostG105RefitCustodyV1:
    base = G110G112CompileCustodyV1.from_physical_partition_receipt(
        partition_receipt_path=Path(str(config.g112_partition_receipt["path"])),
        expected_partition_receipt_sha256=str(config.g112_partition_receipt["sha256"]),
        target_capsule_receipt=Path(str(config.target_capsule_receipt["path"])),
        expected_target_capsule_receipt_sha256=str(config.target_capsule_receipt["sha256"]),
    )
    loader = V9TrainingTargetCapsuleLoaderV1.open(
        config.target_capsule_receipt["path"],
        expected_sha256=str(config.target_capsule_receipt["sha256"]),
    )
    if (
        loader.pair_count != PAIR_COUNT
        or loader.batch_pairs != BATCH_PAIRS
        or loader.preflight.get("test_only_small_fixture") is not False
        or loader.receipt_path != Path(str(config.target_capsule_receipt["path"]))
    ):
        raise PostG105PoseRefitError("G109 target aggregate is not physical n600 batch-16 production custody")
    semantic_packet = base.semantic_child.semantic_packet
    provider = open_final_y1_provider(semantic_packet)
    population_digest = _population_digest(provider)
    binding = _final_y1_binding_from_population(
        semantic_packet,
        population_digest,
    )
    if binding != final_y1_binding_sha256(provider):
        raise PostG105PoseRefitError("G105 final-Y1 population binding helpers disagree")
    pose_targets = np.asarray(loader.targets.source_pose6_f32)
    if (
        pose_targets.dtype != np.float32
        or pose_targets.shape != (PAIR_COUNT, POSE_DIM)
        or not np.all(np.isfinite(pose_targets))
        or _sha256(np.ascontiguousarray(pose_targets, dtype="<f4").tobytes()) != base.pose_targets_sha256
    ):
        raise PostG105PoseRefitError("physical G109 Pose6 target bytes differ")
    scorer = loader.preflight.get("scorer_custody")
    if type(scorer) is not dict:
        raise PostG105PoseRefitError("G109 scorer custody is absent")
    posenet_binding = _strict_file_binding(
        scorer.get("posenet_weights"),
        name="frozen PoseNet weights",
    )
    closure = scorer.get("upstream_closure")
    if type(closure) is not dict or type(closure.get("root")) is not str:
        raise PostG105PoseRefitError("G109 upstream closure is absent")
    upstream_root = Path(closure["root"]).resolve()
    if (
        not upstream_root.is_dir()
        or Path(str(posenet_binding["path"])) != (upstream_root / "models" / "posenet.safetensors").resolve()
    ):
        raise PostG105PoseRefitError("PoseNet weights are outside the recursively bound upstream root")
    pose_copy = np.ascontiguousarray(pose_targets, dtype=np.float32)
    pose_copy.setflags(write=False)
    return PostG105RefitCustodyV1(
        base=base,
        target_loader=loader,
        semantic_packet=semantic_packet,
        provider=provider,
        final_y1_population_digest=population_digest,
        final_y1_binding_sha256=binding,
        pose_targets=pose_copy,
        upstream_root=upstream_root,
        posenet_binding=posenet_binding,
    )


@dataclass(frozen=True, slots=True)
class ExactCompleteArchiveRateOracleV1:
    """Prepared exact G110 wire/ZIP matrix with semantic rendering cached."""

    semantic_variants: tuple[tuple[Y1WireCodecV1, bytes, str], ...]
    pitch: float

    @classmethod
    def prepare(
        cls,
        custody: PostG105RefitCustodyV1,
    ) -> ExactCompleteArchiveRateOracleV1:
        semantic_program = parse_v9_packet(custody.semantic_packet)
        variants = encode_packet_y1_variants(semantic_program)
        if tuple(codec for codec, _packet in variants) != tuple(Y1WireCodecV1):
            raise PostG105PoseRefitError("G105 did not enumerate its complete semantic wire family")
        prepared: list[tuple[Y1WireCodecV1, bytes, str]] = []
        for codec, packet in variants:
            parsed = parse_v9_packet(packet)
            provider = open_final_y1_provider(packet)
            if parsed.y1_wire_codec is not codec or _population_digest(provider) != custody.final_y1_population_digest:
                raise PostG105PoseRefitError("prepared semantic wire alternative changes parsed Y1")
            prepared.append(
                (
                    codec,
                    packet,
                    _final_y1_binding_from_population(
                        packet,
                        custody.final_y1_population_digest,
                    ),
                )
            )
        return cls(
            semantic_variants=tuple(prepared),
            pitch=float(custody.base.pose_initializer.pitch),
        )

    def materialize(
        self,
        *,
        q: np.ndarray,
        scales: np.ndarray,
    ) -> tuple[
        bytes,
        bytes,
        bytes,
        G110CompleteArchiveWireCandidateV1,
        tuple[G110CompleteArchiveWireCandidateV1, ...],
    ]:
        """Return exact selected XIP2, packet, archive, and full wire matrix."""

        xip2 = serialize_xi_payload(q, scales, coder="delta_ar")
        artifacts: list[tuple[bytes, bytes]] = []
        records: list[G110CompleteArchiveWireCandidateV1] = []
        for codec, semantic_packet, binding in self.semantic_variants:
            packet = _encode_g110_packet(
                semantic_packet=semantic_packet,
                final_y1_binding=binding,
                xip2_payload=xip2,
                pitch=self.pitch,
            )
            parsed = parse_g110_generated_y1_pose_v1(packet)
            if (
                parsed.semantic_packet != semantic_packet
                or parsed.final_y1_binding_sha256 != binding
                or not np.array_equal(parsed.q, q)
                or not np.array_equal(parsed.scales, scales)
            ):
                raise PostG105PoseRefitError("prepared complete packet changes shipped XIP2/Y1 operands")
            for method in _OUTER_ZIP_METHODS:
                archive = _build_g110_archive_for_method(packet, method)
                reopened_packet, reopened_method = _read_g110_archive_member(archive)
                if reopened_packet != packet or reopened_method is not method:
                    raise PostG105PoseRefitError("prepared complete archive changes under parse-back")
                records.append(
                    G110CompleteArchiveWireCandidateV1(
                        y1_wire_codec=codec,
                        outer_zip_method=method,
                        semantic_packet_bytes=len(semantic_packet),
                        semantic_packet_sha256=_sha256(semantic_packet),
                        product_packet_bytes=len(packet),
                        product_packet_sha256=_sha256(packet),
                        archive_bytes=len(archive),
                        archive_sha256=_sha256(archive),
                    )
                )
                artifacts.append((packet, archive))
        expected = len(Y1WireCodecV1) * len(_OUTER_ZIP_METHODS)
        if len(records) != expected:
            raise AssertionError("prepared complete archive matrix is incomplete")
        selected_index = min(
            range(len(records)),
            key=lambda index: (
                records[index].archive_bytes,
                int(records[index].y1_wire_codec),
                int(records[index].outer_zip_method),
                records[index].archive_sha256,
            ),
        )
        selected_packet, selected_archive = artifacts[selected_index]
        return (
            xip2,
            selected_packet,
            selected_archive,
            records[selected_index],
            tuple(records),
        )


class ExactBatch16PoseOracleV1:
    """Frozen upstream PoseNet first-six oracle on exact public uint8 pairs."""

    def __init__(
        self,
        custody: PostG105RefitCustodyV1,
        *,
        device: str,
        seed: int,
        torch_num_threads: int,
    ) -> None:
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise PostG105PoseRefitError("frozen upstream PoseNet runtime dependencies are unavailable") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise PostG105PoseRefitError("configured CUDA PoseNet authority is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        torch.set_num_threads(torch_num_threads)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        modules_path = custody.upstream_root / "modules.py"
        if not modules_path.is_file():
            raise PostG105PoseRefitError("recursively bound upstream closure has no modules.py")
        module_name = "_tac_post_g105_upstream_" + _sha256(str(modules_path).encode("utf-8"))[:16]
        spec = importlib.util.spec_from_file_location(module_name, modules_path)
        if spec is None or spec.loader is None:
            raise PostG105PoseRefitError("cannot load frozen upstream modules.py")
        module = importlib.util.module_from_spec(spec)
        old_path = list(sys.path)
        previous_frame_utils = sys.modules.pop("frame_utils", None)
        try:
            sys.path.insert(0, str(custody.upstream_root))
            spec.loader.exec_module(module)
            loaded_frame_utils = sys.modules.get("frame_utils")
            loaded_frame_utils_path = getattr(loaded_frame_utils, "__file__", None)
            if (
                type(loaded_frame_utils_path) is not str
                or Path(loaded_frame_utils_path).resolve() != (custody.upstream_root / "frame_utils.py").resolve()
            ):
                raise PostG105PoseRefitError(
                    "frozen modules.py imported frame_utils outside its recursively bound upstream root"
                )
        finally:
            sys.path[:] = old_path
            sys.modules.pop("frame_utils", None)
            if previous_frame_utils is not None:
                sys.modules["frame_utils"] = previous_frame_utils
        model = module.PoseNet().eval()
        model.load_state_dict(
            load_file(str(custody.posenet_binding["path"]), device="cpu"),
            strict=True,
        )
        model = model.to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    def predict(
        self,
        *,
        pair_start: int,
        camera_y0: np.ndarray,
        camera_y1: np.ndarray,
    ) -> np.ndarray:
        stop = pair_start + int(camera_y0.shape[0])
        expected = min(BATCH_PAIRS, PAIR_COUNT - pair_start)
        if (
            type(pair_start) is not int
            or pair_start % BATCH_PAIRS != 0
            or not 0 <= pair_start < PAIR_COUNT
            or stop > PAIR_COUNT
            or camera_y0.dtype != np.uint8
            or camera_y1.dtype != np.uint8
            or camera_y0.shape != (expected, CAMERA_H, CAMERA_W, CHANNELS)
            or camera_y1.shape != camera_y0.shape
        ):
            raise PostG105PoseRefitError("PoseNet oracle call is not exact chronological batch-16 geometry")
        pair = np.ascontiguousarray(
            np.stack((camera_y0, camera_y1), axis=1),
            dtype=np.uint8,
        )
        tensor = (
            self.torch.from_numpy(pair)
            .to(device=self.device, dtype=self.torch.float32)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        with self.torch.inference_mode():
            output = self.model(self.model.preprocess_input(tensor))["pose"][..., :POSE_DIM]
        result = np.ascontiguousarray(
            output.detach().to(device="cpu", dtype=self.torch.float64).numpy(),
            dtype=np.float64,
        )
        if result.shape != (expected, POSE_DIM) or not np.all(np.isfinite(result)):
            raise PostG105PoseRefitError("frozen PoseNet returned invalid Pose6")
        return result

    def rng_arrays(self) -> dict[str, np.ndarray]:
        cpu = self.torch.get_rng_state().cpu().numpy().astype(np.uint8, copy=True)
        if self.torch.cuda.is_available():
            rows = [item.cpu().numpy().astype(np.uint8, copy=True) for item in self.torch.cuda.get_rng_state_all()]
            cuda = np.stack(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.uint8)
        else:
            cuda = np.zeros((0, 0), dtype=np.uint8)
        return {
            "torch_rng_state_u8": np.ascontiguousarray(cpu),
            "torch_cuda_rng_states_u8": np.ascontiguousarray(cuda),
        }

    def restore_rng(self, arrays: Mapping[str, np.ndarray]) -> None:
        cpu = np.asarray(arrays["torch_rng_state_u8"], dtype=np.uint8)
        self.torch.set_rng_state(self.torch.from_numpy(cpu.copy()))
        cuda = np.asarray(arrays["torch_cuda_rng_states_u8"], dtype=np.uint8)
        if cuda.size:
            if not self.torch.cuda.is_available():
                raise PostG105PoseRefitError("resume has CUDA RNG state but CUDA is unavailable")
            self.torch.cuda.set_rng_state_all([self.torch.from_numpy(row.copy()) for row in cuda])


@lru_cache(maxsize=1)
def _v10_factor2_operator() -> DisjointResizeOperator:
    """Build the certified generic V10 operator once per producer process."""

    operator, _indices, _valid = build_numpy_factor2_gather_plan()
    if (
        operator.camera_h != CAMERA_H
        or operator.camera_w != CAMERA_W
        or operator.scorer_h != 384
        or operator.scorer_w != 512
    ):
        raise PostG105PoseRefitError("cached public V10 factor-2 operator geometry differs")
    return operator


def _render_camera_y1_batch(
    provider: OpenedFinalY1ProviderV1,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    operator = _v10_factor2_operator()
    frames = [
        realize_factor2_uint8_scorer_plane(
            operator,
            provider.render_scorer_y1(pair_id),
        )
        for pair_id in range(start, stop)
    ]
    result = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
    if result.shape != (stop - start, CAMERA_H, CAMERA_W, CHANNELS):
        raise PostG105PoseRefitError("public V10 Y1 realization changed geometry")
    return result


def _render_camera_y0_batch(
    camera_y1: np.ndarray,
    *,
    xi_batch: np.ndarray,
    pitch: float,
) -> np.ndarray:
    if (
        camera_y1.dtype != np.uint8
        or camera_y1.ndim != 4
        or camera_y1.shape[1:] != (CAMERA_H, CAMERA_W, CHANNELS)
        or xi_batch.dtype != np.float64
        or xi_batch.shape != (camera_y1.shape[0], POSE_DIM)
        or not np.all(np.isfinite(xi_batch))
    ):
        raise PostG105PoseRefitError("camera warp batch operands differ")
    geometry = GroundHomographyGeom.eon(
        native_hw=(CAMERA_H, CAMERA_W),
        pitch=float(pitch),
    )
    frames = [
        warp_frame0_uint8_numpy(camera_y1[index], xi_batch[index], geometry) for index in range(camera_y1.shape[0])
    ]
    result = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
    if result.shape != camera_y1.shape:
        raise PostG105PoseRefitError("native public Y0 warp changed geometry")
    return result


def _predict_for_q_batch(
    oracle: ExactBatch16PoseOracleV1,
    camera_y1: np.ndarray,
    *,
    pair_start: int,
    q_batch: np.ndarray,
    scales: np.ndarray,
    pitch: float,
) -> np.ndarray:
    xi_batch = np.ascontiguousarray(
        q_batch.astype(np.float64) * scales.astype(np.float64),
        dtype=np.float64,
    )
    y0 = _render_camera_y0_batch(
        camera_y1,
        xi_batch=xi_batch,
        pitch=pitch,
    )
    return oracle.predict(
        pair_start=pair_start,
        camera_y0=y0,
        camera_y1=camera_y1,
    )


def deterministic_anchor_rows(
    q: np.ndarray,
    *,
    q_levels: int,
) -> np.ndarray:
    """Choose one frozen extremum per channel to fix XIP2's global scale gauge."""

    values = np.asarray(q)
    if values.dtype != np.int16 or values.shape != (PAIR_COUNT, POSE_DIM):
        raise PostG105PoseRefitError("anchor selection requires int16[600,6]")
    anchors = np.full((POSE_DIM,), -1, dtype=np.int64)
    for dimension in range(POSE_DIM):
        extrema = np.flatnonzero(np.abs(values[:, dimension].astype(np.int64)) == q_levels)
        if extrema.size:
            anchors[dimension] = int(extrema[0])
    return anchors


def _gauss_newton_delta(
    jacobian: np.ndarray,
    residual: np.ndarray,
    *,
    damping: float,
    trust_radius_q: int,
) -> np.ndarray:
    """Solve independent damped 6x6 inverse problems in integer-q coordinates."""

    jac = np.asarray(jacobian, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64)
    if (
        jac.ndim != 3
        or jac.shape[1:] != (POSE_DIM, POSE_DIM)
        or res.shape != (jac.shape[0], POSE_DIM)
        or not np.all(np.isfinite(jac))
        or not np.all(np.isfinite(res))
    ):
        raise PostG105PoseRefitError("Gauss-Newton operands differ")
    deltas = np.zeros((jac.shape[0], POSE_DIM), dtype=np.float64)
    identity = np.eye(POSE_DIM, dtype=np.float64)
    for index in range(jac.shape[0]):
        normal = jac[index].T @ jac[index] + damping * identity
        gradient = jac[index].T @ res[index]
        try:
            delta = -np.linalg.solve(normal, gradient)
        except np.linalg.LinAlgError:
            delta = -np.linalg.lstsq(normal, gradient, rcond=None)[0]
        deltas[index] = np.clip(
            delta,
            -float(trust_radius_q),
            float(trust_radius_q),
        )
    return deltas


@dataclass(slots=True)
class _SolverState:
    q_levels: int
    stage_index: int
    next_pair: int
    q: np.ndarray
    scales: np.ndarray
    pose_outputs: np.ndarray
    pose_losses: np.ndarray
    accepted_rows: int
    attempted_rows: int


def _state_arrays(
    *,
    state: _SolverState,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    anchor_rows: np.ndarray,
    numpy_rng_state_json: str,
    oracle: ExactBatch16PoseOracleV1,
    checkpoint_kind: str,
) -> dict[str, np.ndarray]:
    controller = {
        "solver": "exact_uint8_public_receiver_finite_difference_gauss_newton",
        "coordinate": "xip2_integer_q_with_fixed_global_extremum_gauge",
        "stage_index": state.stage_index,
        "next_pair": state.next_pair,
        "finite_difference_q_steps": config.finite_difference_q_steps,
        "damping": config.damping,
        "trust_radius_q": config.trust_radius_q,
        "line_search_scales": list(config.line_search_scales),
        "accepted_rows": state.accepted_rows,
        "attempted_rows": state.attempted_rows,
    }
    arrays = {
        "schema": np.asarray(STATE_SCHEMA),
        "checkpoint_kind": np.asarray(checkpoint_kind),
        "config_sha256": np.asarray(config.config_sha256),
        "run_id": np.asarray(config.run_id),
        "seed": np.asarray(config.seed, dtype=np.int64),
        "source_contract": np.asarray(SOURCE_DOMAIN),
        "render_order": np.asarray(RENDER_ORDER),
        "y1_selected_preimage_schema": np.asarray(V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA),
        "source_g112_partition_receipt_sha256": np.asarray(custody.base.partition_receipt_sha256),
        "source_g112_semantic_child_sha256": np.asarray(custody.base.semantic_child_sha256),
        "source_g112_pose_initializer_sha256": np.asarray(custody.base.pose_initializer_sha256),
        "semantic_packet_sha256": np.asarray(_sha256(custody.semantic_packet)),
        "final_y1_binding_sha256": np.asarray(custody.final_y1_binding_sha256),
        "xi_initializer_sha256": np.asarray(_xi_digest(custody.base.pose_initializer.xi_init)),
        "target_capsule_receipt_sha256": np.asarray(custody.base.target_capsule_receipt_sha256),
        "pose_targets_sha256": np.asarray(custody.base.pose_targets_sha256),
        "posenet_weights_sha256": np.asarray(custody.posenet_binding["sha256"]),
        "q_levels": np.asarray(state.q_levels, dtype=np.int64),
        "stage_index": np.asarray(state.stage_index, dtype=np.int64),
        "next_pair": np.asarray(state.next_pair, dtype=np.int64),
        "q": np.ascontiguousarray(state.q, dtype=np.int16),
        "scales": np.ascontiguousarray(state.scales, dtype=np.float32),
        "xi_eff": np.ascontiguousarray(
            dequantize_xi(state.q, state.scales),
            dtype=np.float64,
        ),
        "pose_outputs": np.ascontiguousarray(
            state.pose_outputs,
            dtype=np.float64,
        ),
        "pose_losses": np.ascontiguousarray(
            state.pose_losses,
            dtype=np.float64,
        ),
        "anchor_rows": np.ascontiguousarray(anchor_rows, dtype=np.int64),
        "controller_json": np.asarray(_canonical_json(controller).decode("ascii")),
        "numpy_rng_state_json": np.asarray(numpy_rng_state_json),
    }
    arrays.update(oracle.rng_arrays())
    return arrays


def _state_path(
    root: Path,
    *,
    q_levels: int,
    stage_index: int,
    next_pair: int,
    checkpoint_kind: str,
) -> Path:
    return (
        root
        / "20_optimizer_states"
        / (f"q{q_levels:05d}_stage{stage_index:02d}_pairs{next_pair:04d}_{checkpoint_kind}.npz")
    )


def _save_state(
    *,
    state: _SolverState,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    anchor_rows: np.ndarray,
    numpy_rng_state_json: str,
    oracle: ExactBatch16PoseOracleV1,
    checkpoint_kind: str,
) -> Path:
    path = _state_path(
        config.output_root,
        q_levels=state.q_levels,
        stage_index=state.stage_index,
        next_pair=state.next_pair,
        checkpoint_kind=checkpoint_kind,
    )
    _write_once(
        path,
        _deterministic_npz(
            _state_arrays(
                state=state,
                config=config,
                custody=custody,
                anchor_rows=anchor_rows,
                numpy_rng_state_json=numpy_rng_state_json,
                oracle=oracle,
                checkpoint_kind=checkpoint_kind,
            )
        ),
    )
    return path


def _load_latest_state(
    *,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    q_levels: int,
    oracle: ExactBatch16PoseOracleV1,
) -> tuple[_SolverState, np.ndarray, str] | None:
    root = config.output_root / "20_optimizer_states"
    paths = sorted(root.glob(f"q{q_levels:05d}_stage*_pairs*_*.npz"))
    if not paths:
        return None
    rows: list[tuple[int, int, int, Path]] = []
    pattern = re.compile(
        rf"^q{q_levels:05d}_stage([0-9]{{2}})_pairs([0-9]{{4}})"
        r"_(periodic|stage_complete)\.npz$"
    )
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None or path.is_symlink():
            raise PostG105PoseRefitError("optimizer-state filename differs")
        rows.append(
            (
                int(match.group(1)),
                int(match.group(2)),
                1 if match.group(3) == "stage_complete" else 0,
                path,
            )
        )
    path = max(rows)[3]
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise PostG105PoseRefitError("optimizer state is not strict NPZ") from exc
    if set(arrays) != _OPTIMIZER_STATE_MEMBERS:
        raise PostG105PoseRefitError("optimizer state member set differs")
    selected_match = pattern.fullmatch(path.name)
    if selected_match is None:
        raise AssertionError("selected optimizer-state path escaped the validated set")
    expected_stage = int(selected_match.group(1))
    expected_next_pair = int(selected_match.group(2))
    expected_checkpoint_kind = selected_match.group(3)
    scalar_expected = {
        "schema": STATE_SCHEMA,
        "checkpoint_kind": expected_checkpoint_kind,
        "config_sha256": config.config_sha256,
        "run_id": config.run_id,
        "seed": config.seed,
        "source_contract": SOURCE_DOMAIN,
        "render_order": RENDER_ORDER,
        "y1_selected_preimage_schema": V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA,
        "source_g112_partition_receipt_sha256": (custody.base.partition_receipt_sha256),
        "source_g112_semantic_child_sha256": custody.base.semantic_child_sha256,
        "source_g112_pose_initializer_sha256": (custody.base.pose_initializer_sha256),
        "semantic_packet_sha256": _sha256(custody.semantic_packet),
        "final_y1_binding_sha256": custody.final_y1_binding_sha256,
        "xi_initializer_sha256": _xi_digest(custody.base.pose_initializer.xi_init),
        "target_capsule_receipt_sha256": (custody.base.target_capsule_receipt_sha256),
        "pose_targets_sha256": custody.base.pose_targets_sha256,
        "posenet_weights_sha256": custody.posenet_binding["sha256"],
        "q_levels": q_levels,
        "stage_index": expected_stage,
        "next_pair": expected_next_pair,
    }
    for name, expected in scalar_expected.items():
        if name not in arrays or np.asarray(arrays[name]).reshape(()).item() != expected:
            raise PostG105PoseRefitError(f"resume state custody differs: {name}")
    q = np.asarray(arrays.get("q"))
    scales = np.asarray(arrays.get("scales"))
    xi = np.asarray(arrays.get("xi_eff"))
    outputs = np.asarray(arrays.get("pose_outputs"))
    losses = np.asarray(arrays.get("pose_losses"))
    anchors = np.asarray(arrays.get("anchor_rows"))
    stage = int(np.asarray(arrays.get("stage_index")).reshape(()).item())
    next_pair = int(np.asarray(arrays.get("next_pair")).reshape(()).item())
    if (
        q.dtype != np.int16
        or q.shape != (PAIR_COUNT, POSE_DIM)
        or scales.dtype != np.float32
        or scales.shape != (POSE_DIM,)
        or xi.dtype != np.float64
        or xi.shape != (PAIR_COUNT, POSE_DIM)
        or not np.array_equal(xi, dequantize_xi(q, scales))
        or outputs.dtype != np.float64
        or outputs.shape != (PAIR_COUNT, POSE_DIM)
        or losses.dtype != np.float64
        or losses.shape != (PAIR_COUNT,)
        or anchors.dtype != np.int64
        or anchors.shape != (POSE_DIM,)
        or not 0 <= stage <= config.local_gauss_newton_stages
        or not 0 <= next_pair <= PAIR_COUNT
        or (next_pair % BATCH_PAIRS != 0 and next_pair != PAIR_COUNT)
    ):
        raise PostG105PoseRefitError("resume optimizer arrays/cursor differ")
    if not np.array_equal(
        anchors,
        deterministic_anchor_rows(q, q_levels=q_levels),
    ):
        # Anchors bind the original fixed scale gauge, not a newly selected row.
        # Require only that each recorded anchor remains an exact extremum.
        for dimension, row in enumerate(anchors.tolist()):
            if row >= 0 and abs(int(q[row, dimension])) != q_levels:
                raise PostG105PoseRefitError("resume lost its fixed XIP2 extremum anchor")
    numpy_state = str(np.asarray(arrays.get("numpy_rng_state_json")).reshape(()).item())
    try:
        parsed_numpy_state = json.loads(numpy_state)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PostG105PoseRefitError("resume NumPy RNG state is invalid") from exc
    if _canonical_json(parsed_numpy_state).decode("ascii") != numpy_state:
        raise PostG105PoseRefitError("resume NumPy RNG state is not canonical")
    oracle.restore_rng(arrays)
    try:
        controller = json.loads(str(np.asarray(arrays["controller_json"]).reshape(()).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PostG105PoseRefitError("resume controller state is invalid") from exc
    expected_controller_keys = {
        "solver",
        "coordinate",
        "stage_index",
        "next_pair",
        "finite_difference_q_steps",
        "damping",
        "trust_radius_q",
        "line_search_scales",
        "accepted_rows",
        "attempted_rows",
    }
    if (
        type(controller) is not dict
        or set(controller) != expected_controller_keys
        or controller["solver"] != "exact_uint8_public_receiver_finite_difference_gauss_newton"
        or controller["coordinate"] != "xip2_integer_q_with_fixed_global_extremum_gauge"
        or controller["stage_index"] != stage
        or controller["next_pair"] != next_pair
        or controller["finite_difference_q_steps"] != config.finite_difference_q_steps
        or controller["damping"] != config.damping
        or controller["trust_radius_q"] != config.trust_radius_q
        or controller["line_search_scales"] != list(config.line_search_scales)
        or type(controller["accepted_rows"]) is not int
        or controller["accepted_rows"] < 0
        or type(controller["attempted_rows"]) is not int
        or controller["attempted_rows"] < controller["accepted_rows"]
    ):
        raise PostG105PoseRefitError("resume controller/cursor state differs")
    return (
        _SolverState(
            q_levels=q_levels,
            stage_index=stage,
            next_pair=next_pair,
            q=np.ascontiguousarray(q),
            scales=np.ascontiguousarray(scales),
            pose_outputs=np.ascontiguousarray(outputs),
            pose_losses=np.ascontiguousarray(losses),
            accepted_rows=int(controller["accepted_rows"]),
            attempted_rows=int(controller["attempted_rows"]),
        ),
        np.ascontiguousarray(anchors),
        numpy_state,
    )


def _initial_state(
    *,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    q_levels: int,
) -> tuple[_SolverState, np.ndarray, str]:
    q, scales = quantize_xi(
        custody.base.pose_initializer.xi_init,
        q_levels=q_levels,
    )
    xi = dequantize_xi(q, scales)
    q_check, scales_check = quantize_xi(xi, q_levels=q_levels)
    if not np.array_equal(q, q_check) or not np.array_equal(scales, scales_check):
        raise PostG105PoseRefitError("XIP2 population is not idempotent under compiler quantization")
    rng = np.random.default_rng(config.seed)
    numpy_state = _canonical_json(rng.bit_generator.state).decode("ascii")
    return (
        _SolverState(
            q_levels=q_levels,
            stage_index=0,
            next_pair=0,
            q=np.ascontiguousarray(q),
            scales=np.ascontiguousarray(scales),
            pose_outputs=np.full(
                (PAIR_COUNT, POSE_DIM),
                np.nan,
                dtype=np.float64,
            ),
            pose_losses=np.full((PAIR_COUNT,), np.nan, dtype=np.float64),
            accepted_rows=0,
            attempted_rows=0,
        ),
        deterministic_anchor_rows(q, q_levels=q_levels),
        numpy_state,
    )


def _validate_fixed_scale_gauge(
    q: np.ndarray,
    scales: np.ndarray,
    anchors: np.ndarray,
    *,
    q_levels: int,
) -> None:
    for dimension, row in enumerate(anchors.tolist()):
        if row >= 0 and abs(int(q[row, dimension])) != q_levels:
            raise PostG105PoseRefitError("local update attempted to move the fixed XIP2 scale anchor")
    xi = dequantize_xi(q, scales)
    q_check, scales_check = quantize_xi(xi, q_levels=q_levels)
    if not np.array_equal(q, q_check) or not np.array_equal(scales, scales_check):
        raise PostG105PoseRefitError("local update changed the global XIP2 scale gauge")


def _baseline_batch(
    state: _SolverState,
    *,
    start: int,
    stop: int,
    custody: PostG105RefitCustodyV1,
    oracle: ExactBatch16PoseOracleV1,
) -> None:
    y1 = _render_camera_y1_batch(custody.provider, start=start, stop=stop)
    predicted = _predict_for_q_batch(
        oracle,
        y1,
        pair_start=start,
        q_batch=state.q[start:stop],
        scales=state.scales,
        pitch=float(custody.base.pose_initializer.pitch),
    )
    residual = predicted - custody.pose_targets[start:stop].astype(np.float64)
    state.pose_outputs[start:stop] = predicted
    state.pose_losses[start:stop] = np.mean(residual * residual, axis=1)


@dataclass(frozen=True, slots=True)
class _BatchProposalV1:
    label: str
    q: np.ndarray
    pose_outputs: np.ndarray
    pose_losses: np.ndarray


def _pose_rate_key(
    *,
    pose_losses: np.ndarray,
    archive_bytes: int,
    proposal_order: int,
) -> tuple[float, int, float, int]:
    losses = np.asarray(pose_losses, dtype=np.float64)
    if (
        losses.shape != (PAIR_COUNT,)
        or not np.all(np.isfinite(losses))
        or np.any(losses < 0.0)
        or type(archive_bytes) is not int
        or archive_bytes <= 0
        or type(proposal_order) is not int
        or proposal_order < 0
    ):
        raise PostG105PoseRefitError("score-native pose/rate proposal operands differ")
    pose_mse = float(np.mean(losses, dtype=np.float64))
    objective = math.sqrt(10.0 * pose_mse) + 25.0 * archive_bytes / ARCHIVE_DENOMINATOR_BYTES
    return objective, archive_bytes, pose_mse, proposal_order


def _local_inverse_batch(
    state: _SolverState,
    *,
    start: int,
    stop: int,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    oracle: ExactBatch16PoseOracleV1,
    archive_oracle: ExactCompleteArchiveRateOracleV1,
    anchors: np.ndarray,
) -> None:
    y1 = _render_camera_y1_batch(custody.provider, start=start, stop=stop)
    base_q = np.ascontiguousarray(state.q[start:stop].copy())
    base_output = np.ascontiguousarray(state.pose_outputs[start:stop].copy())
    base_loss = np.ascontiguousarray(state.pose_losses[start:stop].copy())
    targets = custody.pose_targets[start:stop].astype(np.float64)
    if (
        not np.all(np.isfinite(base_output))
        or not np.all(np.isfinite(base_loss))
        or not np.all(np.isfinite(state.pose_losses))
    ):
        raise PostG105PoseRefitError("local inverse stage has no exact baseline PoseNet outputs")
    batch = stop - start
    jacobian = np.zeros((batch, POSE_DIM, POSE_DIM), dtype=np.float64)
    for dimension in range(POSE_DIM):
        plus = base_q.copy()
        minus = base_q.copy()
        plus_delta = np.minimum(
            config.finite_difference_q_steps,
            state.q_levels - base_q[:, dimension].astype(np.int64),
        )
        minus_delta = np.minimum(
            config.finite_difference_q_steps,
            base_q[:, dimension].astype(np.int64) + state.q_levels,
        )
        anchor = int(anchors[dimension])
        if start <= anchor < stop:
            local = anchor - start
            plus_delta[local] = 0
            minus_delta[local] = 0
        plus[:, dimension] = (plus[:, dimension].astype(np.int64) + plus_delta).astype(np.int16)
        minus[:, dimension] = (minus[:, dimension].astype(np.int64) - minus_delta).astype(np.int16)
        plus_output = _predict_for_q_batch(
            oracle,
            y1,
            pair_start=start,
            q_batch=plus,
            scales=state.scales,
            pitch=float(custody.base.pose_initializer.pitch),
        )
        minus_output = _predict_for_q_batch(
            oracle,
            y1,
            pair_start=start,
            q_batch=minus,
            scales=state.scales,
            pitch=float(custody.base.pose_initializer.pitch),
        )
        denominator = plus_delta + minus_delta
        active = denominator > 0
        jacobian[active, :, dimension] = (plus_output[active] - minus_output[active]) / denominator[
            active, None
        ].astype(np.float64)
    delta = _gauss_newton_delta(
        jacobian,
        base_output - targets,
        damping=config.damping,
        trust_radius_q=config.trust_radius_q,
    )
    best_q = base_q.copy()
    best_output = base_output.copy()
    best_loss = base_loss.copy()
    proposals = [
        _BatchProposalV1(
            label="base",
            q=base_q,
            pose_outputs=base_output,
            pose_losses=base_loss,
        )
    ]
    for line_index, scale in enumerate(config.line_search_scales):
        integer_delta = np.rint(delta * scale).astype(np.int64)
        candidate = np.clip(
            base_q.astype(np.int64) + integer_delta,
            -state.q_levels,
            state.q_levels,
        ).astype(np.int16)
        for dimension, anchor in enumerate(anchors.tolist()):
            if start <= anchor < stop:
                candidate[anchor - start, dimension] = base_q[
                    anchor - start,
                    dimension,
                ]
        output = _predict_for_q_batch(
            oracle,
            y1,
            pair_start=start,
            q_batch=candidate,
            scales=state.scales,
            pitch=float(custody.base.pose_initializer.pitch),
        )
        residual = output - targets
        losses = np.mean(residual * residual, axis=1)
        proposals.append(
            _BatchProposalV1(
                label=f"line_{line_index:02d}_all",
                q=np.ascontiguousarray(candidate),
                pose_outputs=np.ascontiguousarray(output),
                pose_losses=np.ascontiguousarray(losses),
            )
        )
        improving = losses < base_loss
        selective_q = base_q.copy()
        selective_output = base_output.copy()
        selective_loss = base_loss.copy()
        selective_q[improving] = candidate[improving]
        selective_output[improving] = output[improving]
        selective_loss[improving] = losses[improving]
        proposals.append(
            _BatchProposalV1(
                label=f"line_{line_index:02d}_pose_improving",
                q=selective_q,
                pose_outputs=selective_output,
                pose_losses=selective_loss,
            )
        )
        better = losses < best_loss
        best_q[better] = candidate[better]
        best_output[better] = output[better]
        best_loss[better] = losses[better]
    proposals.append(
        _BatchProposalV1(
            label="rowwise_best_pose",
            q=best_q,
            pose_outputs=best_output,
            pose_losses=best_loss,
        )
    )
    ranked: list[
        tuple[
            tuple[float, int, float, int],
            _BatchProposalV1,
        ]
    ] = []
    seen_q: dict[bytes, _BatchProposalV1] = {}
    for proposal_order, proposal in enumerate(proposals):
        q_identity = np.ascontiguousarray(
            proposal.q,
            dtype="<i2",
        ).tobytes()
        prior = seen_q.get(q_identity)
        if prior is not None:
            if not np.array_equal(
                prior.pose_outputs,
                proposal.pose_outputs,
            ) or not np.array_equal(
                prior.pose_losses,
                proposal.pose_losses,
            ):
                raise PostG105PoseRefitError("identical shipped q proposal produced different PoseNet outputs")
            continue
        seen_q[q_identity] = proposal
        full_q = state.q.copy()
        full_q[start:stop] = proposal.q
        _validate_fixed_scale_gauge(
            full_q,
            state.scales,
            anchors,
            q_levels=state.q_levels,
        )
        full_losses = state.pose_losses.copy()
        full_losses[start:stop] = proposal.pose_losses
        _xip2, _packet, archive, _selected, _alternatives = archive_oracle.materialize(
            q=full_q,
            scales=state.scales,
        )
        ranked.append(
            (
                _pose_rate_key(
                    pose_losses=full_losses,
                    archive_bytes=len(archive),
                    proposal_order=proposal_order,
                ),
                proposal,
            )
        )
    if not ranked:
        raise AssertionError("score-native batch proposal set is empty")
    _selected_key, selected_proposal = min(ranked, key=lambda item: item[0])
    changed_rows = np.any(selected_proposal.q != base_q, axis=1)
    state.q[start:stop] = selected_proposal.q
    state.pose_outputs[start:stop] = selected_proposal.pose_outputs
    state.pose_losses[start:stop] = selected_proposal.pose_losses
    state.accepted_rows += int(np.count_nonzero(changed_rows))
    state.attempted_rows += batch
    _validate_fixed_scale_gauge(
        state.q,
        state.scales,
        anchors,
        q_levels=state.q_levels,
    )


def _candidate_receipt_path(root: Path, q_levels: int) -> Path:
    return root / "30_candidate_rows" / f"q{q_levels:05d}_candidate.json"


def _candidate_state_path(root: Path, q_levels: int) -> Path:
    return root / "30_candidate_rows" / f"q{q_levels:05d}_final_state.npz"


def _candidate_receipt(
    *,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    archive_oracle: ExactCompleteArchiveRateOracleV1,
    state: _SolverState,
    anchors: np.ndarray,
) -> dict[str, object]:
    if (
        state.stage_index != config.local_gauss_newton_stages
        or state.next_pair != PAIR_COUNT
        or not np.all(np.isfinite(state.pose_outputs))
        or not np.all(np.isfinite(state.pose_losses))
    ):
        raise PostG105PoseRefitError("candidate is not a complete n600 solver state")
    xi = np.ascontiguousarray(
        dequantize_xi(state.q, state.scales),
        dtype=np.float64,
    )
    q_check, scales_check = quantize_xi(xi, q_levels=state.q_levels)
    if not np.array_equal(q_check, state.q) or not np.array_equal(
        scales_check,
        state.scales,
    ):
        raise PostG105PoseRefitError("candidate differs under compiler XIP2 round-trip")
    xip2, packet, archive, selected, alternatives = archive_oracle.materialize(
        q=state.q,
        scales=state.scales,
    )
    pose_mse = float(np.mean(state.pose_losses, dtype=np.float64))
    pose_term = math.sqrt(10.0 * pose_mse)
    rate_term = 25.0 * len(archive) / ARCHIVE_DENOMINATOR_BYTES
    state_path = _candidate_state_path(config.output_root, state.q_levels)
    state_arrays = {
        "schema": np.asarray(STATE_SCHEMA),
        "checkpoint_kind": np.asarray("candidate_final"),
        "config_sha256": np.asarray(config.config_sha256),
        "q_levels": np.asarray(state.q_levels, dtype=np.int64),
        "q": np.ascontiguousarray(state.q, dtype=np.int16),
        "scales": np.ascontiguousarray(state.scales, dtype=np.float32),
        "xi_eff": xi,
        "pose_outputs": np.ascontiguousarray(
            state.pose_outputs,
            dtype=np.float64,
        ),
        "pose_losses": np.ascontiguousarray(
            state.pose_losses,
            dtype=np.float64,
        ),
        "anchor_rows": np.ascontiguousarray(anchors, dtype=np.int64),
    }
    _write_once(state_path, _deterministic_npz(state_arrays))
    alternatives_rows = [
        {
            "y1_wire_codec": row.y1_wire_codec.name,
            "outer_zip_method": row.outer_zip_method.name,
            "semantic_packet_bytes": row.semantic_packet_bytes,
            "semantic_packet_sha256": row.semantic_packet_sha256,
            "product_packet_bytes": row.product_packet_bytes,
            "product_packet_sha256": row.product_packet_sha256,
            "archive_bytes": row.archive_bytes,
            "archive_sha256": row.archive_sha256,
        }
        for row in alternatives
    ]
    body: dict[str, object] = {
        "schema": CANDIDATE_SCHEMA,
        "run_id": config.run_id,
        "config_sha256": config.config_sha256,
        "q_levels": state.q_levels,
        "pair_count": PAIR_COUNT,
        "batch_pairs": BATCH_PAIRS,
        "source_contract": SOURCE_DOMAIN,
        "render_order": RENDER_ORDER,
        "semantic_packet_sha256": _sha256(custody.semantic_packet),
        "final_y1_binding_sha256": custody.final_y1_binding_sha256,
        "source_g112_partition_receipt_sha256": (custody.base.partition_receipt_sha256),
        "target_capsule_receipt_sha256": (custody.base.target_capsule_receipt_sha256),
        "pose_targets_sha256": custody.base.pose_targets_sha256,
        "posenet_weights_sha256": custody.posenet_binding["sha256"],
        "xi_eff_sha256": _xi_digest(xi),
        "xip2_bytes": len(xip2),
        "xip2_sha256": _sha256(xip2),
        "pose_mse": pose_mse,
        "pose_term": pose_term,
        "complete_archive_bytes": len(archive),
        "complete_archive_sha256": _sha256(archive),
        "product_packet_bytes": len(packet),
        "product_packet_sha256": _sha256(packet),
        "selected_y1_wire_codec": selected.y1_wire_codec.name,
        "selected_outer_zip_method": selected.outer_zip_method.name,
        "complete_archive_wire_candidates": alternatives_rows,
        "selection_objective_pose_plus_rate": pose_term + rate_term,
        "rate_term": rate_term,
        "accepted_rows": state.accepted_rows,
        "attempted_rows": state.attempted_rows,
        "fixed_global_quantizer_anchor_rows": anchors.tolist(),
        "optimizer_verdict_scope": OPTIMIZER_VERDICT_SCOPE,
        "global_xip2_range_optimality_claim": False,
        "global_range_reactivation_required": True,
        "global_range_reactivation_blocker": global_range_reactivation_blocker(),
        "candidate_state": _output_binding(state_path),
        "exact_public_receiver_in_loop": True,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    return _seal(body, field="candidate_receipt_sha256")


def _open_candidate_state(
    *,
    state_binding: Mapping[str, object],
    config: PostG105PoseRefitConfigV1,
    q_levels: int,
) -> dict[str, np.ndarray]:
    try:
        with np.load(str(state_binding["path"]), allow_pickle=False) as archive:
            if set(archive.files) != _CANDIDATE_STATE_MEMBERS:
                raise PostG105PoseRefitError("candidate optimizer-state member set differs")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError) as exc:
        raise PostG105PoseRefitError("candidate optimizer state is not strict NPZ") from exc
    scalar_expected = {
        "schema": STATE_SCHEMA,
        "checkpoint_kind": "candidate_final",
        "config_sha256": config.config_sha256,
        "q_levels": q_levels,
    }
    for name, expected in scalar_expected.items():
        if np.asarray(arrays[name]).reshape(()).item() != expected:
            raise PostG105PoseRefitError(f"candidate optimizer-state binding differs: {name}")
    q = arrays["q"]
    scales = arrays["scales"]
    xi = arrays["xi_eff"]
    outputs = arrays["pose_outputs"]
    losses = arrays["pose_losses"]
    anchors = arrays["anchor_rows"]
    if (
        q.dtype != np.int16
        or q.shape != (PAIR_COUNT, POSE_DIM)
        or scales.dtype != np.float32
        or scales.shape != (POSE_DIM,)
        or xi.dtype != np.float64
        or xi.shape != (PAIR_COUNT, POSE_DIM)
        or not np.array_equal(xi, dequantize_xi(q, scales))
        or outputs.dtype != np.float64
        or outputs.shape != (PAIR_COUNT, POSE_DIM)
        or not np.all(np.isfinite(outputs))
        or losses.dtype != np.float64
        or losses.shape != (PAIR_COUNT,)
        or not np.all(np.isfinite(losses))
        or np.any(losses < 0.0)
        or anchors.dtype != np.int64
        or anchors.shape != (POSE_DIM,)
    ):
        raise PostG105PoseRefitError("candidate optimizer arrays differ")
    _validate_fixed_scale_gauge(
        q,
        scales,
        anchors,
        q_levels=q_levels,
    )
    return arrays


def _load_candidate_receipt(
    *,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    q_levels: int,
) -> dict[str, object] | None:
    path = _candidate_receipt_path(config.output_root, q_levels)
    if not path.exists():
        return None
    if path.is_symlink():
        raise PostG105PoseRefitError("candidate receipt is a symlink")
    try:
        value = json.loads(path.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostG105PoseRefitError("candidate receipt is invalid JSON") from exc
    if (
        type(value) is not dict
        or set(value) != _CANDIDATE_RECEIPT_MEMBERS
        or value.get("schema") != CANDIDATE_SCHEMA
        or value.get("config_sha256") != config.config_sha256
        or value.get("q_levels") != q_levels
        or value.get("source_g112_partition_receipt_sha256") != custody.base.partition_receipt_sha256
        or value.get("target_capsule_receipt_sha256") != custody.base.target_capsule_receipt_sha256
        or value.get("pose_targets_sha256") != custody.base.pose_targets_sha256
        or value.get("optimizer_verdict_scope") != OPTIMIZER_VERDICT_SCOPE
        or value.get("global_xip2_range_optimality_claim") is not False
        or value.get("global_range_reactivation_required") is not True
        or value.get("global_range_reactivation_blocker") != global_range_reactivation_blocker()
        or _sha256(_canonical_json({key: item for key, item in value.items() if key != "candidate_receipt_sha256"}))
        != value.get("candidate_receipt_sha256")
    ):
        raise PostG105PoseRefitError("candidate receipt custody/self-hash differs")
    state_binding = _strict_file_binding(
        value.get("candidate_state"),
        name="candidate state",
    )
    _open_candidate_state(
        state_binding=state_binding,
        config=config,
        q_levels=q_levels,
    )
    return value


def _run_one_q_level(
    *,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    oracle: ExactBatch16PoseOracleV1,
    archive_oracle: ExactCompleteArchiveRateOracleV1,
    q_levels: int,
) -> dict[str, object]:
    existing = _load_candidate_receipt(
        config=config,
        custody=custody,
        q_levels=q_levels,
    )
    if existing is not None:
        return existing
    loaded = _load_latest_state(
        config=config,
        custody=custody,
        q_levels=q_levels,
        oracle=oracle,
    )
    if loaded is None:
        state, anchors, numpy_state = _initial_state(
            config=config,
            custody=custody,
            q_levels=q_levels,
        )
    else:
        state, anchors, numpy_state = loaded
    while state.stage_index <= config.local_gauss_newton_stages:
        while state.next_pair < PAIR_COUNT:
            start = state.next_pair
            stop = min(start + BATCH_PAIRS, PAIR_COUNT)
            if state.stage_index == 0:
                _baseline_batch(
                    state,
                    start=start,
                    stop=stop,
                    custody=custody,
                    oracle=oracle,
                )
            else:
                _local_inverse_batch(
                    state,
                    start=start,
                    stop=stop,
                    config=config,
                    custody=custody,
                    oracle=oracle,
                    archive_oracle=archive_oracle,
                    anchors=anchors,
                )
            state.next_pair = stop
            _save_state(
                state=state,
                config=config,
                custody=custody,
                anchor_rows=anchors,
                numpy_rng_state_json=numpy_state,
                oracle=oracle,
                checkpoint_kind="periodic",
            )
        _save_state(
            state=state,
            config=config,
            custody=custody,
            anchor_rows=anchors,
            numpy_rng_state_json=numpy_state,
            oracle=oracle,
            checkpoint_kind="stage_complete",
        )
        if state.stage_index == config.local_gauss_newton_stages:
            break
        state.stage_index += 1
        state.next_pair = 0
    receipt = _candidate_receipt(
        config=config,
        custody=custody,
        archive_oracle=archive_oracle,
        state=state,
        anchors=anchors,
    )
    _write_json_once(_candidate_receipt_path(config.output_root, q_levels), receipt)
    return receipt


def _git_head() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PostG105PoseRefitError("cannot bind source Git HEAD") from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise PostG105PoseRefitError("source Git HEAD is not canonical SHA-1")
    return value


def _final_checkpoint_arrays(
    *,
    config: PostG105PoseRefitConfigV1,
    custody: PostG105RefitCustodyV1,
    q_levels: int,
    xi_eff: np.ndarray,
) -> dict[str, np.ndarray]:
    base = custody.base
    return {
        "schema": np.asarray(POST_G105_REFIT_CHECKPOINT_SCHEMA),
        "run_id": np.asarray(config.run_id),
        "seed": np.asarray(config.seed, dtype=np.int64),
        "source_contract": np.asarray(SOURCE_DOMAIN),
        "render_order": np.asarray(RENDER_ORDER),
        "y1_selected_preimage_schema": np.asarray(V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA),
        "source_g112_partition_receipt_sha256": np.asarray(base.partition_receipt_sha256),
        "source_g112_semantic_child_sha256": np.asarray(base.semantic_child_sha256),
        "source_g112_pose_initializer_sha256": np.asarray(base.pose_initializer_sha256),
        "source_g111_deploy_checkpoint_sha256": np.asarray(base.source_deploy_checkpoint_sha256),
        "source_g111_resume_checkpoint_sha256": np.asarray(base.source_resume_checkpoint_sha256),
        "source_g111_lineage_receipt_sha256": np.asarray(base.source_lineage_receipt_sha256),
        "source_g111_checkpoint_id_sha256": np.asarray(base.source_checkpoint_id_sha256),
        "source_g111_root_sha256": np.asarray(base.source_root_sha256),
        "semantic_packet_sha256": np.asarray(_sha256(custody.semantic_packet)),
        "final_y1_binding_sha256": np.asarray(custody.final_y1_binding_sha256),
        "xi_initializer_sha256": np.asarray(_xi_digest(base.pose_initializer.xi_init)),
        "target_projection_sha256": np.asarray(base.target_projection_sha256),
        "target_capsule_receipt_sha256": np.asarray(base.target_capsule_receipt_sha256),
        "pose_targets_sha256": np.asarray(base.pose_targets_sha256),
        "exact_public_receiver_in_loop": np.asarray(1, dtype=np.int8),
        "pitch": np.asarray(base.pose_initializer.pitch, dtype=np.float64),
        "q_levels": np.asarray(q_levels, dtype=np.int64),
        "xi_eff": np.ascontiguousarray(xi_eff, dtype=np.float64),
    }


@dataclass(frozen=True, slots=True)
class PostG105PoseRefitResultV1:
    checkpoint_path: Path
    checkpoint_sha256: str
    run_receipt_path: Path
    run_receipt_sha256: str
    audit_receipt_path: Path
    selected_q_levels: int
    selected_pose_mse: float
    selected_archive_bytes: int
    selected_archive_sha256: str


def run_post_g105_pose_refit(
    *,
    config: PostG105PoseRefitConfigV1,
    resume_from: Path,
    command: Sequence[str],
) -> PostG105PoseRefitResultV1:
    """Run or resume the real exact-public n600 inverse refit."""

    resume = resume_from.expanduser().resolve()
    if resume != config.output_root:
        raise PostG105PoseRefitError("--resume-from must equal the typed config output_root")
    if os.environ.get(GOVERNED_MARKER_ENV) != "1":
        raise PostG105PoseRefitError("full n600 refit requires governed admission through tools/safe_run.py")
    tokens = [str(token) for token in command]
    if "--resume-from" not in tokens or not tokens:
        raise PostG105PoseRefitError("durable run command must contain --resume-from")
    config.output_root.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema": "tac.post_g105_generated_y1_pose_refit_preflight.v1",
        "run_id": config.run_id,
        "config": _output_binding(config.config_path),
        "output_root": str(config.output_root),
        "storage_waterfall": [str(root.resolve()) for root in SSD_ROOTS],
        "minimum_free_bytes": MIN_FREE_BYTES,
        "observed_free_bytes": shutil.disk_usage(config.output_root).free,
        "pair_count": PAIR_COUNT,
        "batch_pairs": BATCH_PAIRS,
        "governed_marker_env": GOVERNED_MARKER_ENV,
        "governed_marker_observed": True,
        "resumable_from_disk": True,
        "periodic_checkpoint_every_pairs": BATCH_PAIRS,
        "stage_checkpoints_preserved": True,
        "success_scratch_auto_cleaned": True,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    _write_json_once(config.output_root / "00_preflight.json", preflight)
    custody = open_custody(config)
    oracle = ExactBatch16PoseOracleV1(
        custody,
        device=config.device,
        seed=config.seed,
        torch_num_threads=config.torch_num_threads,
    )
    archive_oracle = ExactCompleteArchiveRateOracleV1.prepare(custody)
    candidate_rows = [
        _run_one_q_level(
            config=config,
            custody=custody,
            oracle=oracle,
            archive_oracle=archive_oracle,
            q_levels=q_levels,
        )
        for q_levels in config.q_levels_candidates
    ]
    selected = min(
        candidate_rows,
        key=lambda row: (
            float(row["selection_objective_pose_plus_rate"]),
            int(row["complete_archive_bytes"]),
            int(row["q_levels"]),
            str(row["complete_archive_sha256"]),
        ),
    )
    selected_state_binding = _strict_file_binding(
        selected["candidate_state"],
        name="selected candidate state",
    )
    q_levels = int(selected["q_levels"])
    selected_state = _open_candidate_state(
        state_binding=selected_state_binding,
        config=config,
        q_levels=q_levels,
    )
    xi_eff = np.ascontiguousarray(
        selected_state["xi_eff"],
        dtype=np.float64,
    )
    q_check, scales_check = quantize_xi(xi_eff, q_levels=q_levels)
    if (
        xi_eff.shape != (PAIR_COUNT, POSE_DIM)
        or not np.all(np.isfinite(xi_eff))
        or not np.array_equal(xi_eff, dequantize_xi(q_check, scales_check))
    ):
        raise PostG105PoseRefitError("selected candidate is not compiler-fixed XIP2 xi_eff")
    checkpoint_path = config.output_root / "90_final" / "post_g105_pose_refit_final.npz"
    _write_once(
        checkpoint_path,
        _deterministic_npz(
            _final_checkpoint_arrays(
                config=config,
                custody=custody,
                q_levels=q_levels,
                xi_eff=xi_eff,
            )
        ),
    )
    checkpoint_binding = _output_binding(checkpoint_path)
    stage_paths = sorted(
        (
            *(path for path in (config.output_root / "20_optimizer_states").glob("*.npz")),
            *(path for path in (config.output_root / "30_candidate_rows").glob("*_final_state.npz")),
        ),
        key=lambda path: str(path),
    )
    stage_bindings = [_output_binding(path) for path in stage_paths]
    stage_bindings.append(checkpoint_binding)
    base = custody.base
    run_body: dict[str, object] = {
        "schema": POST_G105_REFIT_RUN_SCHEMA,
        "run_id": config.run_id,
        "seed": config.seed,
        "source_git_sha": _git_head(),
        "command": tokens,
        "fresh_own_lineage": True,
        "source_contract": SOURCE_DOMAIN,
        "render_order": RENDER_ORDER,
        "y1_selected_preimage_schema": (V10_FACTOR2_SELECTED_PREIMAGE_SCHEMA),
        "source_g112_partition_receipt_sha256": (base.partition_receipt_sha256),
        "source_g112_semantic_child_sha256": base.semantic_child_sha256,
        "source_g112_pose_initializer_sha256": (base.pose_initializer_sha256),
        "source_g111_deploy_checkpoint_sha256": (base.source_deploy_checkpoint_sha256),
        "source_g111_resume_checkpoint_sha256": (base.source_resume_checkpoint_sha256),
        "source_g111_lineage_receipt_sha256": (base.source_lineage_receipt_sha256),
        "source_g111_checkpoint_id_sha256": (base.source_checkpoint_id_sha256),
        "source_g111_root_sha256": base.source_root_sha256,
        "semantic_packet_sha256": _sha256(custody.semantic_packet),
        "final_y1_binding_sha256": custody.final_y1_binding_sha256,
        "xi_initializer_sha256": _xi_digest(base.pose_initializer.xi_init),
        "target_projection_sha256": base.target_projection_sha256,
        "target_capsule_receipt_sha256": (base.target_capsule_receipt_sha256),
        "pose_targets_sha256": base.pose_targets_sha256,
        "exact_public_receiver_in_loop": True,
        "resumable_from_disk": True,
        "stage_checkpoints_preserved": True,
        "stage_checkpoints": stage_bindings,
        "final_checkpoint": checkpoint_binding,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    run_receipt = _seal(run_body, field="receipt_sha256")
    run_path = config.output_root / "91_post_g105_pose_refit_run_receipt.json"
    _write_json_once(run_path, run_receipt)
    audit_body: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "run_id": config.run_id,
        "config": _output_binding(config.config_path),
        "source_git_sha": run_body["source_git_sha"],
        "source_contract": SOURCE_DOMAIN,
        "render_order": RENDER_ORDER,
        "pair_count": PAIR_COUNT,
        "batch_pairs": BATCH_PAIRS,
        "solver": ("exact_uint8_public_receiver_finite_difference_gauss_newton"),
        "global_quantizer_coupling_control": ("fixed_extremum_gauge_per_channel_plus_exact_qlevel_archive_arbitration"),
        "optimizer_verdict_scope": OPTIMIZER_VERDICT_SCOPE,
        "global_xip2_range_optimality_claim": False,
        "global_range_reactivation_required": True,
        "global_range_reactivation_blocker": global_range_reactivation_blocker(),
        "candidate_rows": [
            _output_binding(
                _candidate_receipt_path(
                    config.output_root,
                    int(row["q_levels"]),
                )
            )
            for row in candidate_rows
        ],
        "selected_q_levels": q_levels,
        "selected_pose_mse": selected["pose_mse"],
        "selected_pose_term": selected["pose_term"],
        "selected_complete_archive_bytes": selected["complete_archive_bytes"],
        "selected_complete_archive_sha256": (selected["complete_archive_sha256"]),
        "selected_pose_plus_rate_objective": (selected["selection_objective_pose_plus_rate"]),
        "final_checkpoint": checkpoint_binding,
        "run_receipt": _output_binding(run_path),
        "exact_public_receiver_in_loop": True,
        "upstream_evaluate_py_not_run": True,
        "research_only": True,
        "candidate_claim": False,
        "score_claim": False,
        "pointer_moved": False,
    }
    audit = _seal(audit_body, field="audit_receipt_sha256")
    audit_path = config.output_root / "92_post_g105_pose_refit_audit.json"
    _write_json_once(audit_path, audit)
    scratch_paths = [path for path in config.output_root.rglob(".scratch") if path.is_dir()]
    for scratch in scratch_paths:
        if any(scratch.iterdir()):
            raise PostG105PoseRefitError(f"successful run left nonempty scratch: {scratch}")
        scratch.rmdir()
    return PostG105PoseRefitResultV1(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(checkpoint_binding["sha256"]),
        run_receipt_path=run_path,
        run_receipt_sha256=sha256_file(run_path),
        audit_receipt_path=audit_path,
        selected_q_levels=q_levels,
        selected_pose_mse=float(selected["pose_mse"]),
        selected_archive_bytes=int(selected["complete_archive_bytes"]),
        selected_archive_sha256=str(selected["complete_archive_sha256"]),
    )


def write_blocker_receipt(
    *,
    config_path: Path,
    resume_from: Path,
    command: Sequence[str],
    error: BaseException,
) -> Path | None:
    """Best-effort durable fail-closed blocker; never weakens the raised error."""

    root = resume_from.expanduser().resolve()
    roots = tuple(item.resolve() for item in SSD_ROOTS)
    if not any(root == item or item in root.parents for item in roots):
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {
            "schema": BLOCKER_SCHEMA,
            "config_path": str(config_path.expanduser().resolve()),
            "resume_from": str(root),
            "command": [str(token) for token in command],
            "error_type": type(error).__name__,
            "error": str(error),
            "prerequisite_closed": False,
            "exact_public_receiver_in_loop_completed": False,
            "upstream_evaluate_py_run": False,
            "research_only": True,
            "candidate_claim": False,
            "score_claim": False,
            "pointer_moved": False,
        }
        receipt = _seal(body, field="blocker_receipt_sha256")
        path = root / "00_fail_closed_blocker.json"
        _write_json_once(path, receipt)
        return path
    except (OSError, ValueError, PostG105PoseRefitError):
        return None


__all__ = [
    "ARCHIVE_DENOMINATOR_BYTES",
    "AUDIT_SCHEMA",
    "BATCH_PAIRS",
    "BLOCKER_SCHEMA",
    "CANDIDATE_SCHEMA",
    "CONFIG_SCHEMA",
    "GLOBAL_RANGE_BLOCKER_SCHEMA",
    "OPTIMIZER_VERDICT_SCOPE",
    "PAIR_COUNT",
    "POST_G105_REFIT_CHECKPOINT_SCHEMA",
    "POST_G105_REFIT_RUN_SCHEMA",
    "STATE_SCHEMA",
    "ExactBatch16PoseOracleV1",
    "PostG105PoseRefitConfigV1",
    "PostG105PoseRefitError",
    "PostG105PoseRefitResultV1",
    "PostG105RefitCustodyV1",
    "deterministic_anchor_rows",
    "global_range_reactivation_blocker",
    "load_config",
    "open_custody",
    "run_post_g105_pose_refit",
    "seal_config",
    "write_blocker_receipt",
]
