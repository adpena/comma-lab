#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the bounded J12 receiver-coordinate custody and decomposition producer.

The four J10 opening proposals are four sealed scalar actuator coordinates:
``W(alpha) = W0 + alpha * delta_p``.  Because RG1 can make the reverse
``alpha=-1`` reflection infeasible, J12 measures the boundary-forward realized
secant from the exact source at ``alpha=0`` to the sealed proposal at
``alpha=1`` through the exact receiver, uint8/R chain, and frozen scorers.
Each rank-4 Seg inner-Jacobian chunk and Pose6 Jacobian chunk is preserved on
the SSD with immutable pair and archive custody.

This tool is local advisory work only.  It cannot dispatch paid work, mutate
the acceptance rule, promote a score, or write the contest pointer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROJECT_MAIN = Path("/Users/adpena/Projects/pact")
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.local_acceleration.mlx_scorer_adapters import (  # noqa: E402
    load_mlx_distortion_scorer_adapter_from_upstream,
    temporary_mlx_device,
)
from tac.optimization.ddm_j11_opening_proposal_decomposition import (  # noqa: E402
    SEALED_OPENING_PROPOSALS,
    build_source_preserving_pc1_adapter_archive,
    null_projector_from_receiver_gram,
    parse_source_preserving_pc1_adapter_archive,
    receive_source_preserving_pc1_camera_pairs,
)
from tac.optimization.ddm_pc1_pose_stream import (  # noqa: E402
    DDMPC1TrainableParameterMapV1,
    PC1PosePacketV1,
    serialize_pc1_packet,
)
from tac.optimization.ddm_ws1_warm_start import receive_joint_descent_archive  # noqa: E402
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    AdamStateV1,
    DirectDescriptionJointDescentMLXModule,
    DirectDescriptionJointDescentTypedConfigV1,
    clipped_adam_step,
    compile_parameterized_archive,
    initial_adam_state,
    lift_v15_archive,
    linear_rewarmup_factor,
    opening_candidate_gradient,
    parameter_group_indices,
    project_adam_state_geometry,
    realize_parameter_theta,
    realized_training_state,
)
from tac.optimization.pure_priced_realized_objective import (  # noqa: E402
    RealizedObjectiveState,
    pure_priced_realized_delta,
)

DEFAULT_CONFIG = REPO / ".omx/research/configs/ddm_j12_366_receiver_coordinate_custody_producers_20260725.json"
CONFIG_SCHEMA: Final = "ddm_j12_receiver_coordinate_custody_config.v1"
RECEIPT_SCHEMA: Final = "ddm_j12_receiver_coordinate_custody_receipt.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SOURCE_BYTES: Final = 37_545_489
EXPECTED_TYPED_J10_HASH: Final = "478aaf0db82463104e0e848f95f69361a72950c70dd76cc2d2574ec8e3267a64"
EXPECTED_PROPOSAL_SHAS: Final = {
    "worldsheet_joint_active_x_+1": "679b096bc701e096aee0aae032aec23eab2e4a155e03c8b244ded8144ba45d46",
    "worldsheet_joint_active_x_-1": "e4103eecdbe19ea6a9aa7f55f6baeadc6158b954cac46a6ee27c122d995e9c5f",
    "worldsheet_joint_active_y_-1": "1e539c624271c33167fa82c13d71d080590be7265e020c4e0ea6817e7bc14ff2",
    "local_exact_gradient": "010ae7df873db9d8210de12d6d59e6293a6d0fed8b31e1b99aa6afbd86ee94fe",
}
PROPOSAL_BINDING_KEYS: Final = {
    "worldsheet_joint_active_x_+1": "j10_proposal_x_plus",
    "worldsheet_joint_active_x_-1": "j10_proposal_x_minus",
    "worldsheet_joint_active_y_-1": "j10_proposal_y_minus",
    "local_exact_gradient": "j10_proposal_local",
}


class J12Error(ValueError):
    """Fail-closed J12 custody or execution error."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise J12Error(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise J12Error(f"JSON artifact is not an object: {path}")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise J12Error(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical(dict(value)))


def _canonical_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Write a deterministic compressed NPZ with fixed ZIP metadata."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for name in sorted(arrays):
            array = np.ascontiguousarray(arrays[name])
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    required = {
        "schema",
        "run_id",
        "lane_id",
        "delegation_checkpoint_key",
        "output_root",
        "upstream_root",
        "pair_count",
        "verdict_batch",
        "torch_threads",
        "seed",
        "minimum_free_memory_gib",
        "minimum_free_storage_gib",
        "proposal_coordinate_domain",
        "secant_scheme",
        "secant_quantum",
        "proposal_ids",
        "pc1_accepted_steps",
        "source_baselines",
        "source_artifacts",
        "acceptance_rule",
        "pose_derivative_weight",
        "break_even_ratio",
        "conditional_smoke_steps",
        "paid_dispatch",
        "contest_eval",
        "research_only",
        "score_claim",
        "promotion_eligible",
        "pointer",
        "pointer_moved",
        "main_review_required",
    }
    if set(value) != required or value["schema"] != CONFIG_SCHEMA:
        raise J12Error("J12 typed config keys/schema differ")
    fixed = {
        "pair_count": 600,
        "verdict_batch": 32,
        "torch_threads": 4,
        "seed": 0,
        "minimum_free_memory_gib": 20,
        "minimum_free_storage_gib": 20,
        "secant_scheme": "forward_realized_source_alpha0_to_sealed_proposal_alpha1",
        "secant_quantum": 1,
        "proposal_ids": list(SEALED_OPENING_PROPOSALS),
        "pc1_accepted_steps": [8, 16],
        "acceptance_rule": "pure_priced_realized_delta.joint_delta_lt_zero",
        "pose_derivative_weight": "5/sqrt(10*d_pose)",
        "break_even_ratio": 1,
        "conditional_smoke_steps": 8,
        "paid_dispatch": False,
        "contest_eval": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "main_review_required": True,
    }
    drift = {key: value.get(key) for key, expected in fixed.items() if value.get(key) != expected}
    if drift:
        raise J12Error(f"J12 execution/authority contract differs: {drift}")
    if value["proposal_coordinate_domain"] != (
        "one boundary-feasible scalar amplitude alpha in [0,1] per exact sealed J10 proposal vector"
    ):
        raise J12Error("J12 proposal coordinate domain differs")
    output = Path(value["output_root"]).resolve()
    if not output.is_relative_to(Path("/Volumes/VertigoDataTier/pact").resolve()):
        raise J12Error("J12 output must use the primary SSD tier")
    return value, _sha256(_canonical(value))


def _validate_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise J12Error("bound artifact keys differ")
    path = _resolve(str(binding["path"]))
    if not path.is_file() or path.is_symlink():
        raise J12Error(f"bound regular file unavailable: {path}")
    byte_count, digest = _sha256_file(path)
    if byte_count != int(binding["bytes"]) or digest != binding["sha256"]:
        raise J12Error(f"bound artifact custody differs: {path}")
    return {"path": str(path), "bytes": byte_count, "sha256": digest}


def _free_memory_receipt(stage: str, minimum_gib: int) -> dict[str, Any]:
    import psutil

    available = int(psutil.virtual_memory().available)
    required = int(minimum_gib * 1024**3)
    if available < required:
        raise J12Error(f"REFUSE_{stage}_AVAILABLE_MEMORY_BELOW_{minimum_gib}_GIB")
    return {
        "stage": stage,
        "available_bytes": available,
        "required_bytes": required,
        "psutil_available": True,
        "admitted": True,
    }


def _preflight(config: Mapping[str, Any], typed_hash: str) -> dict[str, Any]:
    output = Path(config["output_root"])
    path = output / "00_preflight.json"
    if path.exists():
        value = _read_json(path)
        if value.get("typed_config_hash") != typed_hash or value.get("admitted") is not True:
            raise J12Error("preserved J12 preflight differs")
        value["fresh_resume_memory"] = _free_memory_receipt(
            "J12_RESUME",
            int(config["minimum_free_memory_gib"]),
        )
        return value
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    required_storage = int(config["minimum_free_storage_gib"]) * 1024**3
    if usage.free < required_storage:
        raise J12Error("REFUSE_J12_SSD_FREE_SPACE_BELOW_20_GIB")
    bindings = {name: _validate_binding(binding) for name, binding in sorted(config["source_artifacts"].items())}
    source = bindings["w_joint_step50"]
    baseline = config["source_baselines"]["W_joint_step50_live"]
    if source["sha256"] != baseline["archive_sha256"] or source["bytes"] != baseline["archive_bytes"]:
        raise J12Error("J12 exact source baseline/archive custody differs")
    raw_wseg = _read_json(Path(bindings["w_seg_raw_x_plus_verdict"]["path"]))
    raw_pricing = raw_wseg.get("pure_priced_delta", {})
    if raw_wseg.get("proposal_source") != "worldsheet_joint_active_x_+1" or not math.isclose(
        float(raw_pricing.get("joint_delta", math.nan)),
        -0.0004297730820253919,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise J12Error("historical W_seg raw x+ negative custody differs")
    memory = _free_memory_receipt("J12_PREFLIGHT", int(config["minimum_free_memory_gib"]))
    receipt = {
        "schema": "ddm_j12_receiver_coordinate_preflight.v1",
        "typed_config_hash": typed_hash,
        "bindings": bindings,
        "memory": memory,
        "storage": {
            "tier": "/Volumes/VertigoDataTier/pact",
            "output_root": str(output),
            "free_bytes": usage.free,
            "required_free_bytes": required_storage,
            "cleanup": (
                "pair cameras and scorer tensors are released after every chunk; "
                "only deterministic Jacobian NPZ chunks, exact archive bytes, "
                "and machine-readable receipts are preserved"
            ),
        },
        "historical_w_seg_raw_x_plus": {
            "archive_sha256": raw_wseg["archive_sha256"],
            "archive_bytes": raw_wseg["archive_bytes"],
            "d_seg": raw_wseg["d_seg"],
            "d_pose": raw_wseg["d_pose"],
            "pure_priced_delta": raw_pricing,
            "role": "PRE_DECOMPOSITION_REFERENCE_NOT_TRANSFERRED_AS_COMPONENT_VERDICT",
        },
        "governor": {
            "pair_count": 600,
            "verdict_batch": 32,
            "forward_secant_source_archives": 1,
            "forward_secant_proposal_archives": 4,
            "paid_dispatch": False,
            "contest_eval": False,
            "conditional_smoke_steps_max": 8,
        },
        "admitted": True,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
    }
    _atomic_json(path, receipt)
    return receipt


def _proposal_slug(proposal_id: str) -> str:
    return proposal_id.replace("+", "plus").replace("-", "minus")


def _load_proposal_receipts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for proposal_id, binding_key in PROPOSAL_BINDING_KEYS.items():
        binding = config["source_artifacts"][binding_key]
        path = _resolve(binding["path"])
        row = _read_json(path)
        if (
            row.get("proposal_source") != proposal_id
            or row.get("proposal_multiplier") != 32.0
            or row.get("archive_sha256") != EXPECTED_PROPOSAL_SHAS[proposal_id]
            or row.get("score_claim") is not False
        ):
            raise J12Error(f"sealed J10 proposal receipt differs: {proposal_id}")
        result[proposal_id] = row
    return result


def _proposal_index_path(output: Path) -> Path:
    return output / "01_proposals" / "index.json"


def _proposal_archive_path(output: Path, proposal_id: str, sign: str) -> Path:
    return output / "01_proposals" / _proposal_slug(proposal_id) / f"alpha_{sign}.zip.receipt-bytes"


def _proposal_state_path(output: Path, proposal_id: str) -> Path:
    return output / "01_proposals" / _proposal_slug(proposal_id) / "proposal_state.npz"


def _load_proposal_state(output: Path, proposal_id: str) -> np.ndarray:
    path = _proposal_state_path(output, proposal_id)
    with np.load(path, allow_pickle=False) as data:
        delta = np.asarray(data["realized_delta"], dtype=np.float32)
    if delta.shape != (368,):
        raise J12Error(f"proposal state geometry differs: {proposal_id}")
    return delta


def _rederive_proposals(config: Mapping[str, Any], typed_hash: str) -> dict[str, Any]:
    output = Path(config["output_root"])
    index_path = _proposal_index_path(output)
    receipts = _load_proposal_receipts(config)
    if index_path.exists():
        value = _read_json(index_path)
        if value.get("typed_config_hash") != typed_hash:
            raise J12Error("preserved proposal index typed identity differs")
        for proposal_id, row in value["proposals"].items():
            plus = _proposal_archive_path(output, proposal_id, "plus")
            base = _proposal_archive_path(output, proposal_id, "base")
            if _sha256(plus.read_bytes()) != row["plus_archive_sha256"]:
                raise J12Error(f"preserved plus proposal differs: {proposal_id}")
            if _sha256(base.read_bytes()) != row["base_archive_sha256"]:
                raise J12Error(f"preserved proposal base differs: {proposal_id}")
            _load_proposal_state(output, proposal_id)
        return value

    ticket_path = _resolve(config["source_artifacts"]["j10_ticket"]["path"])
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket_path)
    if typed.typed_config_hash() != EXPECTED_TYPED_J10_HASH:
        raise J12Error("J12 J10 typed-config identity differs")
    source = _resolve(config["source_artifacts"]["w_joint_step50"]["path"]).read_bytes()
    lift = lift_v15_archive(source)
    if len(lift.parameter_names) != 368 or lift.exact_reemit() != source:
        raise J12Error("J12 source parameter lift/reemit differs")
    target_path = _resolve(config["source_artifacts"]["target_cache"]["path"])
    labels = open_stored_npy_memmap(target_path, "lstars")
    poses = open_stored_npy_memmap(target_path, "gt_poses")
    schedule = typed.full_run_schedule
    if schedule is None or schedule.warm_start_reform is None:
        raise J12Error("J12 J10 schedule/reform unavailable")
    reform = schedule.warm_start_reform
    pair_ids = tuple(range(schedule.warm_start_pair, schedule.warm_start_pair + schedule.train_batch))
    active_groups = reform.opening_active_groups
    base_camera, template_masks, basis, basis_indices, local_theta, _ = realized_training_state(
        lift,
        np.zeros(len(lift.parameter_names), dtype=np.float32),
        pair_ids=pair_ids,
        active_groups=active_groups,
        include_lane_programs=False,
    )
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    with temporary_mlx_device("gpu"):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(typed.upstream_root, device="cpu")
        model = DirectDescriptionJointDescentMLXModule(
            lift=lift,
            scorer_adapter=adapter,
            seg_targets=labels,
            pose_targets=poses,
        )
        _loss, gradient = model.loss_and_grad(
            local_theta,
            pair_ids=pair_ids,
            base_camera=base_camera,
            template_masks=template_masks,
            realized_secant_basis=basis,
            realized_secant_indices=basis_indices,
            # The sealed J10 PoseFinish state was WAITING_FOR_STRICT_SEG_ADMISSION.
            pose_objective_weight=0.0,
        )
    groups = parameter_group_indices(lift)
    active = set().union(*(groups[name] for name in active_groups))
    gradient[[index for index in range(len(gradient)) if index not in active]] = 0.0
    gradient_path = output / "01_proposals" / "opening_gradient.npz"
    _atomic_bytes(
        gradient_path,
        _canonical_npz(
            {
                "gradient": np.asarray(gradient, dtype="<f4"),
                "pair_ids": np.asarray(pair_ids, dtype="<i4"),
            }
        ),
    )
    state = initial_adam_state(len(lift.parameter_names))
    base_realized = realize_parameter_theta(lift, state.theta)
    proposal_rows: dict[str, Any] = {}
    for proposal_id in SEALED_OPENING_PROPOSALS:
        proposal_gradient = opening_candidate_gradient(
            lift,
            proposal_id,
            gradient,
            active_pair_ids=reform.opening_candidate_pair_ids,
        )
        proposal_gradient[[index for index in range(len(proposal_gradient)) if index not in active]] = 0.0
        candidate = clipped_adam_step(
            state,
            proposal_gradient,
            learning_rate=(schedule.learning_rate_quantum_fraction * reform.lr_rewarmup_floor * 32.0),
            grad_clip=typed.grad_clip,
            ema_decay=typed.ema_decay,
            beta2=reform.adam_beta2,
            maximum_update=reform.maximum_continuous_update_quantum_fraction,
            theta_lattice_denominator=reform.proposal_q8_denominator,
        )
        candidate, geometry_events = project_adam_state_geometry(lift, candidate)
        plus_archive, plus_realized = compile_parameterized_archive(
            lift,
            candidate.theta,
            include_lane_programs=False,
        )
        delta = np.asarray(plus_realized - base_realized, dtype=np.float32)
        reverse = AdamStateV1(
            step=1,
            theta=np.ascontiguousarray(-delta),
            ema=np.ascontiguousarray(-delta),
            first_moment=np.zeros_like(delta),
            second_moment=np.zeros_like(delta),
        )
        reverse, reverse_events = project_adam_state_geometry(lift, reverse)
        reflected_archive, reflected_realized = compile_parameterized_archive(
            lift,
            reverse.theta,
            include_lane_programs=False,
        )
        central_reflection_feasible = np.array_equal(reflected_realized, -delta)
        plus_sha = _sha256(plus_archive)
        if plus_sha != EXPECTED_PROPOSAL_SHAS[proposal_id]:
            raise J12Error(
                f"rederived sealed proposal differs: {proposal_id}: {plus_sha} != {EXPECTED_PROPOSAL_SHAS[proposal_id]}"
            )
        if len(plus_archive) != int(receipts[proposal_id]["archive_bytes"]):
            raise J12Error(f"rederived sealed proposal bytes differ: {proposal_id}")
        plus_path = _proposal_archive_path(output, proposal_id, "plus")
        base_path = _proposal_archive_path(output, proposal_id, "base")
        reflected_path = _proposal_archive_path(output, proposal_id, "reflected_rg1")
        state_path = _proposal_state_path(output, proposal_id)
        _atomic_bytes(plus_path, plus_archive)
        _atomic_bytes(base_path, source)
        _atomic_bytes(reflected_path, reflected_archive)
        changed = np.flatnonzero(delta)
        _atomic_bytes(
            state_path,
            _canonical_npz(
                {
                    "changed_indices": np.asarray(changed, dtype="<i4"),
                    "realized_delta": np.asarray(delta, dtype="<f4"),
                }
            ),
        )
        row = {
            "proposal_id": proposal_id,
            "coordinate_domain": f"boundary_ray{{source->{proposal_id}}}",
            "alpha_base": 0,
            "alpha_plus": 1,
            "changed_parameter_count": len(changed),
            "changed_parameter_indices": changed.tolist(),
            "changed_parameter_names": [lift.parameter_names[index] for index in changed],
            "plus_archive_path": str(plus_path),
            "plus_archive_bytes": len(plus_archive),
            "plus_archive_sha256": plus_sha,
            "base_archive_path": str(base_path),
            "base_archive_bytes": len(source),
            "base_archive_sha256": _sha256(source),
            "forward_secant_denominator": 1,
            "secant_scheme": "forward_realized_source_alpha0_to_sealed_proposal_alpha1",
            "central_reflection_feasible": central_reflection_feasible,
            "reflected_rg1_archive_path": str(reflected_path),
            "reflected_rg1_archive_sha256": _sha256(reflected_archive),
            "reflected_rg1_realized_equals_negative_proposal": central_reflection_feasible,
            "plus_geometry_events": list(geometry_events),
            "reflected_rg1_geometry_events": list(reverse_events),
            "proposal_receipt_sha256": config["source_artifacts"][PROPOSAL_BINDING_KEYS[proposal_id]]["sha256"],
            "receiver_parseback_identity": True,
        }
        proposal_rows[proposal_id] = row
        _atomic_json(state_path.with_suffix(".json"), row)
        print(
            json.dumps(
                {
                    "stage": "proposal_rederived",
                    "proposal_id": proposal_id,
                    "plus_sha256": plus_sha,
                    "base_sha256": row["base_archive_sha256"],
                    "central_reflection_feasible": central_reflection_feasible,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "schema": "ddm_j12_sealed_proposal_coordinates.v1",
        "typed_config_hash": typed_hash,
        "j10_typed_config_hash": typed.typed_config_hash(),
        "source_archive_sha256": _sha256(source),
        "secant_scheme": "forward_realized_source_alpha0_to_sealed_proposal_alpha1",
        "opening_pair_ids": list(pair_ids),
        "opening_proposal_pair_ids": list(reform.opening_candidate_pair_ids),
        "pose_finish_state": "WAITING_FOR_STRICT_SEG_ADMISSION",
        "pose_objective_weight": 0.0,
        "proposal_multiplier": 32.0,
        "proposal_count": len(proposal_rows),
        "proposals": proposal_rows,
        "research_only": True,
        "score_claim": False,
    }
    _atomic_json(index_path, result)
    return result


def _load_models(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    from tools.measure_ddm_menu1_realized_flip_menu import Menu1Config, _load_models

    menu_path = _resolve(config["source_artifacts"]["menu1_config"]["path"])
    menu = Menu1Config.model_validate_json(menu_path.read_bytes())
    segnet, posenet, custody = _load_models(menu)
    import torch

    torch.set_num_threads(int(config["torch_threads"]))
    torch.manual_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    custody["batch_size"] = int(config["verdict_batch"])
    custody["seed"] = int(config["seed"])
    return segnet, posenet, custody


def _rank4_filters(segnet: Any) -> tuple[np.ndarray, dict[str, Any]]:
    head = segnet.segmentation_head[0]
    weight = head.weight.detach().cpu().numpy().astype(np.float64)
    if weight.shape[0] != 5 or weight.ndim != 4:
        raise J12Error("SegNet head geometry differs")
    winner_rival = (weight[1:] - weight[0:1]).reshape(4, -1)
    _u, singular, vh = np.linalg.svd(winner_rival, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(winner_rival.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != 4:
        raise J12Error("SegNet winner-rival head rank is not exactly four")
    basis = np.asarray(vh[:4], dtype=np.float64)
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    filters = np.ascontiguousarray(basis.reshape(4, *weight.shape[1:]), dtype=np.float32)
    return filters, {
        "head_weight_shape": list(weight.shape),
        "head_weight_sha256": _sha256(np.ascontiguousarray(weight, dtype="<f8").tobytes()),
        "winner_rival_shape": list(winner_rival.shape),
        "winner_rival_singular_values": singular.tolist(),
        "winner_rival_rank": rank,
        "rank_tolerance": tolerance,
        "rank4_filter_sha256": _sha256(filters.astype("<f4", copy=False).tobytes()),
        "construction": "SVD_row_basis_of_four_class1to4_minus_class0_3x3x16_head_filters",
    }


def _forward_inner_pose(
    segnet: Any,
    posenet: Any,
    camera_pairs: np.ndarray,
    rank4_filters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    camera = np.asarray(camera_pairs)
    if camera.dtype != np.uint8 or camera.shape[1:] != (2, 874, 1164, 3):
        raise J12Error("J12 scorer camera geometry differs")
    tensor = torch.from_numpy(np.ascontiguousarray(camera)).permute(0, 1, 4, 2, 3).contiguous().float()
    filters = torch.from_numpy(rank4_filters)
    with torch.inference_mode():
        seg_input = segnet.preprocess_input(tensor)
        features = segnet.encoder(seg_input)
        decoded = segnet.decoder(features)
        logits = segnet.segmentation_head(decoded)
        canonical_logits = segnet(seg_input)
        if not torch.equal(logits, canonical_logits):
            raise J12Error("manual SegNet rank4 tap differs from canonical forward")
        inner = functional.conv2d(decoded, filters, bias=None, stride=1, padding=1)
        pose_output = posenet(posenet.preprocess_input(tensor))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
        cells = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        inner_np = inner.cpu().numpy().astype(np.float32)
        pose6 = pose[:, :6].cpu().numpy().astype(np.float64)
    return (
        np.ascontiguousarray(cells),
        np.ascontiguousarray(inner_np),
        np.ascontiguousarray(pose6),
    )


def _jacobian_chunk_path(output: Path, proposal_id: str, start: int, stop: int) -> Path:
    return output / "02_jacobians" / _proposal_slug(proposal_id) / f"chunk_{start:04d}_{stop:04d}.npz"


def _jacobian_base_chunk_path(output: Path, start: int, stop: int) -> Path:
    return output / "02_jacobians" / "_source_base" / f"chunk_{start:04d}_{stop:04d}.npz"


def _endpoint_accumulator() -> dict[str, Any]:
    return {
        "errors": 0,
        "sites": 0,
        "pose_sse": 0.0,
        "pose_coordinates": 0,
        "class_errors": np.zeros(5, dtype=np.int64),
        "class_sites": np.zeros(5, dtype=np.int64),
    }


def _accumulate_endpoint(
    accumulator: dict[str, Any],
    *,
    cells: np.ndarray,
    pose6: np.ndarray,
    target: np.ndarray,
    target_pose: np.ndarray,
) -> None:
    mismatch = cells != target
    accumulator["errors"] += int(np.count_nonzero(mismatch))
    accumulator["sites"] += int(cells.size)
    accumulator["pose_sse"] += float(np.square(pose6 - target_pose).sum(dtype=np.float64))
    accumulator["pose_coordinates"] += int(pose6.size)
    for class_id in range(5):
        mask = target == class_id
        accumulator["class_errors"][class_id] += int(np.count_nonzero(mismatch & mask))
        accumulator["class_sites"][class_id] += int(np.count_nonzero(mask))


def _finalize_endpoint(
    accumulator: Mapping[str, Any],
    *,
    archive: bytes,
) -> dict[str, Any]:
    errors = int(accumulator["errors"])
    sites = int(accumulator["sites"])
    pose_sse = float(accumulator["pose_sse"])
    pose_coordinates = int(accumulator["pose_coordinates"])
    class_errors = np.asarray(accumulator["class_errors"], dtype=np.int64)
    class_sites = np.asarray(accumulator["class_sites"], dtype=np.int64)
    if errors != int(class_errors.sum()) or sites != int(class_sites.sum()):
        raise J12Error("J12 endpoint global/per-class totals differ")
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    return {
        "num_pairs": 600,
        "batch_size": 32,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "errors": errors,
        "sites": sites,
        "pose_squared_error_sum": pose_sse,
        "pose_coordinates": pose_coordinates,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "per_class": {
            name: {
                "class_id": class_id,
                "errors": int(class_errors[class_id]),
                "sites": int(class_sites[class_id]),
                "d_seg": float(class_errors[class_id] / class_sites[class_id]),
            }
            for class_id, name in enumerate(CLASS_NAMES)
        },
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }


def _measure_jacobians(
    config: Mapping[str, Any],
    typed_hash: str,
    proposal_index: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(config["output_root"])
    final_path = output / "02_jacobians" / "index.json"
    if final_path.exists():
        value = _read_json(final_path)
        if value.get("typed_config_hash") != typed_hash:
            raise J12Error("preserved Jacobian index typed identity differs")
        return value
    memory = _free_memory_receipt(
        "J12_N600_JACOBIAN",
        int(config["minimum_free_memory_gib"]),
    )
    target_path = _resolve(config["source_artifacts"]["target_cache"]["path"])
    labels = open_stored_npy_memmap(target_path, "lstars")
    poses = open_stored_npy_memmap(target_path, "gt_poses")
    if labels.shape != (600, 384, 512) or poses.shape != (600, 6):
        raise J12Error("J12 target-cache geometry differs")
    segnet, posenet, scorer_custody = _load_models(config)
    filters, rank4_custody = _rank4_filters(segnet)
    receipts = _load_proposal_receipts(config)
    canonical_base_archive = _resolve(config["source_artifacts"]["w_joint_step50"]["path"]).read_bytes()
    canonical_base_receiver = receive_joint_descent_archive(canonical_base_archive)
    proposals: dict[str, Any] = {}
    for proposal_id in SEALED_OPENING_PROPOSALS:
        proposal_row = proposal_index["proposals"][proposal_id]
        plus_archive = _proposal_archive_path(output, proposal_id, "plus").read_bytes()
        base_archive = _proposal_archive_path(output, proposal_id, "base").read_bytes()
        if base_archive != canonical_base_archive:
            raise J12Error(f"proposal base archive differs from canonical source: {proposal_id}")
        plus_receiver = receive_joint_descent_archive(plus_archive)
        plus_accumulator = _endpoint_accumulator()
        base_accumulator = _endpoint_accumulator()
        pose_gram = 0.0
        seg_gram = 0.0
        pose_digest = hashlib.sha256()
        seg_digest = hashlib.sha256()
        chunk_rows: list[dict[str, Any]] = []
        for start in range(0, 600, 32):
            stop = min(start + 32, 600)
            chunk_path = _jacobian_chunk_path(output, proposal_id, start, stop)
            metadata_path = chunk_path.with_suffix(".json")
            base_chunk_path = _jacobian_base_chunk_path(output, start, stop)
            base_metadata_path = base_chunk_path.with_suffix(".json")
            ids = tuple(range(start, stop))
            if base_chunk_path.exists() and base_metadata_path.exists():
                base_metadata = _read_json(base_metadata_path)
                if (
                    base_metadata.get("base_archive_sha256") != _sha256(base_archive)
                    or base_metadata.get("pair_range") != [start, stop]
                    or _sha256(base_chunk_path.read_bytes()) != base_metadata.get("npz_sha256")
                ):
                    raise J12Error(f"preserved source-base Jacobian chunk differs: {base_chunk_path}")
                with np.load(base_chunk_path, allow_pickle=False) as arrays:
                    base_cells = np.asarray(arrays["base_cells"], dtype=np.uint8)
                    base_inner = np.asarray(
                        arrays["base_rank4_inner"],
                        dtype=np.float32,
                    )
                    base_pose = np.asarray(arrays["base_pose6"], dtype=np.float64)
            else:
                base_camera = canonical_base_receiver.render_camera_pairs(ids)
                base_cells, base_inner, base_pose = _forward_inner_pose(
                    segnet,
                    posenet,
                    base_camera,
                    filters,
                )
                base_payload = _canonical_npz(
                    {
                        "base_cells": base_cells,
                        "base_pose6": base_pose.astype("<f8", copy=False),
                        "base_rank4_inner": base_inner.astype("<f4", copy=False),
                        "pair_ids": np.asarray(ids, dtype="<i4"),
                    }
                )
                _atomic_bytes(base_chunk_path, base_payload)
                base_metadata = {
                    "schema": "ddm_j12_receiver_coordinate_source_chunk.v1",
                    "pair_range": [start, stop],
                    "pair_ids": list(ids),
                    "base_archive_sha256": _sha256(base_archive),
                    "npz_path": str(base_chunk_path),
                    "npz_bytes": len(base_payload),
                    "npz_sha256": _sha256(base_payload),
                    "base_camera_sha256": _sha256(base_camera.tobytes(order="C")),
                    "rank4_inner_shape": list(base_inner.shape),
                    "pose6_shape": list(base_pose.shape),
                    "evidence_axis": EVIDENCE_AXIS,
                    "score_claim": False,
                }
                _atomic_json(base_metadata_path, base_metadata)
                del base_camera
                gc.collect()
            if chunk_path.exists() and metadata_path.exists():
                metadata = _read_json(metadata_path)
                if (
                    metadata.get("plus_archive_sha256") != _sha256(plus_archive)
                    or metadata.get("base_archive_sha256") != _sha256(base_archive)
                    or metadata.get("pair_range") != [start, stop]
                    or _sha256(chunk_path.read_bytes()) != metadata.get("npz_sha256")
                ):
                    raise J12Error(f"preserved Jacobian chunk custody differs: {chunk_path}")
                with np.load(chunk_path, allow_pickle=False) as arrays:
                    pair_ids = np.asarray(arrays["pair_ids"], dtype=np.int32)
                    pose_j = np.asarray(arrays["pose_j"], dtype=np.float64)
                    seg_j = np.asarray(arrays["seg_rank4_j"], dtype=np.float32)
                    plus_cells = np.asarray(arrays["plus_cells"], dtype=np.uint8)
                    plus_pose = np.asarray(arrays["plus_pose6"], dtype=np.float64)
            else:
                plus_camera = plus_receiver.render_camera_pairs(ids)
                plus_cells, plus_inner, plus_pose = _forward_inner_pose(
                    segnet,
                    posenet,
                    plus_camera,
                    filters,
                )
                pair_ids = np.asarray(ids, dtype=np.int32)
                pose_j = np.ascontiguousarray(plus_pose - base_pose, dtype=np.float64)
                seg_j = np.ascontiguousarray(plus_inner - base_inner, dtype=np.float32)
                payload = _canonical_npz(
                    {
                        "pair_ids": pair_ids.astype("<i4", copy=False),
                        "plus_cells": plus_cells,
                        "plus_pose6": plus_pose.astype("<f8", copy=False),
                        "pose_j": pose_j.astype("<f8", copy=False),
                        "seg_rank4_j": seg_j.astype("<f4", copy=False),
                    }
                )
                _atomic_bytes(chunk_path, payload)
                metadata = {
                    "schema": "ddm_j12_receiver_coordinate_jacobian_chunk.v1",
                    "proposal_id": proposal_id,
                    "coordinate_id": f"sealed_proposal_amplitude:{proposal_id}",
                    "pair_range": [start, stop],
                    "pair_ids": pair_ids.tolist(),
                    "plus_archive_sha256": _sha256(plus_archive),
                    "base_archive_sha256": _sha256(base_archive),
                    "forward_secant_denominator": 1,
                    "secant_scheme": "forward_realized_source_alpha0_to_sealed_proposal_alpha1",
                    "npz_path": str(chunk_path),
                    "npz_bytes": len(payload),
                    "npz_sha256": _sha256(payload),
                    "plus_camera_sha256": _sha256(plus_camera.tobytes(order="C")),
                    "source_base_chunk_sha256": base_metadata["npz_sha256"],
                    "pose_j_shape": list(pose_j.shape),
                    "seg_rank4_j_shape": list(seg_j.shape),
                    "evidence_axis": EVIDENCE_AXIS,
                    "score_claim": False,
                }
                _atomic_json(metadata_path, metadata)
                del plus_camera, plus_inner
                gc.collect()
            target = np.asarray(labels[start:stop], dtype=np.uint8)
            target_pose = np.asarray(poses[start:stop], dtype=np.float64)
            _accumulate_endpoint(
                plus_accumulator,
                cells=plus_cells,
                pose6=plus_pose,
                target=target,
                target_pose=target_pose,
            )
            _accumulate_endpoint(
                base_accumulator,
                cells=base_cells,
                pose6=base_pose,
                target=target,
                target_pose=target_pose,
            )
            pose_j_le = np.ascontiguousarray(pose_j, dtype="<f8")
            seg_j_le = np.ascontiguousarray(seg_j, dtype="<f4")
            pose_gram += float(np.square(pose_j_le).sum(dtype=np.float64))
            seg_gram += float(np.square(seg_j_le.astype(np.float64)).sum(dtype=np.float64))
            pose_digest.update(pose_j_le.tobytes(order="C"))
            seg_digest.update(seg_j_le.tobytes(order="C"))
            chunk_rows.append(metadata)
            print(
                json.dumps(
                    {
                        "stage": "jacobian_chunk",
                        "proposal_id": proposal_id,
                        "pair_range": [start, stop],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del (
                pair_ids,
                pose_j,
                seg_j,
                plus_cells,
                base_cells,
                plus_pose,
                base_pose,
                base_inner,
            )
            gc.collect()
        plus_endpoint = _finalize_endpoint(plus_accumulator, archive=plus_archive)
        base_endpoint = _finalize_endpoint(base_accumulator, archive=base_archive)
        settled = receipts[proposal_id]
        if (
            plus_endpoint["archive_sha256"] != settled["archive_sha256"]
            or plus_endpoint["archive_bytes"] != settled["archive_bytes"]
            or not math.isclose(
                plus_endpoint["d_seg"],
                float(settled["d_seg"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                plus_endpoint["d_pose"],
                float(settled["d_pose"]),
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
        ):
            raise J12Error(f"fresh J12 plus endpoint differs from settled J10: {proposal_id}")
        source_baseline = config["source_baselines"]["W_joint_step50_live"]
        if (
            base_endpoint["archive_sha256"] != source_baseline["archive_sha256"]
            or base_endpoint["archive_bytes"] != source_baseline["archive_bytes"]
            or not math.isclose(
                base_endpoint["d_seg"],
                float(source_baseline["d_seg"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                base_endpoint["d_pose"],
                float(source_baseline["d_pose"]),
                rel_tol=0.0,
                abs_tol=2.0e-12,
            )
        ):
            raise J12Error(f"fresh J12 base endpoint differs from settled source: {proposal_id}")
        pose_projector, pose_certificate = null_projector_from_receiver_gram(
            np.asarray([[pose_gram]], dtype=np.float64),
            coordinate_ids=(f"sealed_proposal_amplitude:{proposal_id}",),
            jacobian_id=f"J_pose::{proposal_id}",
        )
        seg_projector, seg_certificate = null_projector_from_receiver_gram(
            np.asarray([[seg_gram]], dtype=np.float64),
            coordinate_ids=(f"sealed_proposal_amplitude:{proposal_id}",),
            jacobian_id=f"J_seg_rank4_inner::{proposal_id}",
        )
        row = {
            "proposal_id": proposal_id,
            "coordinate_domain": proposal_row["coordinate_domain"],
            "plus_archive_sha256": _sha256(plus_archive),
            "base_archive_sha256": _sha256(base_archive),
            "secant_scheme": "forward_realized_source_alpha0_to_sealed_proposal_alpha1",
            "pair_count": 600,
            "batch_size": 32,
            "chunk_count": len(chunk_rows),
            "pose_jacobian": {
                "shape": [600, 6, 1],
                "dtype": "float64",
                "sha256": pose_digest.hexdigest(),
                "gram": [[pose_gram]],
                "certificate": pose_certificate,
                "null_projector": pose_projector.tolist(),
            },
            "seg_rank4_inner_jacobian": {
                "shape": [600, 4, 384, 512, 1],
                "dtype": "float32",
                "sha256": seg_digest.hexdigest(),
                "gram": [[seg_gram]],
                "certificate": seg_certificate,
                "null_projector": seg_projector.tolist(),
            },
            "plus_endpoint": plus_endpoint,
            "base_endpoint": base_endpoint,
            "chunk_receipts": chunk_rows,
            "receiver_chain": (
                "sealed proposal amplitude -> exact archive compile/parseback -> "
                "receiver camera874x1164 uint8 -> scorer bilinear R512x384 -> "
                "SegNet rank4 winner-rival inner / PoseNet output6"
            ),
            "proposal_pair_foreign_key": proposal_id,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        proposals[proposal_id] = row
        _atomic_json(
            output / "02_jacobians" / _proposal_slug(proposal_id) / "jacobian_receipt.json",
            row,
        )
    result = {
        "schema": "ddm_j12_receiver_coordinate_jacobian_bundle.v1",
        "typed_config_hash": typed_hash,
        "source_archive_sha256": config["source_baselines"]["W_joint_step50_live"]["archive_sha256"],
        "source_archive_bytes": config["source_baselines"]["W_joint_step50_live"]["archive_bytes"],
        "proposal_coordinate_domain": config["proposal_coordinate_domain"],
        "rank4_head_custody": rank4_custody,
        "scorer_custody": scorer_custody,
        "source_base_chunks": {
            "root": str(output / "02_jacobians" / "_source_base"),
            "chunk_count": 19,
            "reuse": "ONE_EXACT_BATCH32_SOURCE_FORWARD_SHARED_BY_ALL_FOUR_PROPOSAL_SECANTS",
        },
        "memory_preflight": memory,
        "proposals": proposals,
        "measured_jacobian_count": 8,
        "pair_count": 600,
        "batch_size": 32,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_json(final_path, result)
    return result


def _pc1_packet(config: Mapping[str, Any], accepted_step: int) -> PC1PosePacketV1:
    pc2_config = _read_json(_resolve(config["source_artifacts"]["pc2_config"]["path"]))
    admission_binding = pc2_config["source_artifacts"]["pc1_admission"]
    admission_path = _resolve(admission_binding["path"])
    admission = _read_json(admission_path)
    parameter = admission["parameter_map"]
    parameter_map = DDMPC1TrainableParameterMapV1(
        pair_count=600,
        knot_count=int(parameter["coordinate_schema"]["knot_count"]),
        xi_scales=tuple(float(value) for value in parameter["xi_scales"]),
        residual_scale=float(parameter["residual_scale"]),
    )
    checkpoint = _read_json(_resolve(config["source_artifacts"][f"pc2_accepted_{accepted_step:03d}"]["path"]))
    q_xi = np.asarray(checkpoint["q_xi"], dtype=np.int16)
    if q_xi.shape != (parameter_map.knot_count, 6):
        raise J12Error("PC1 accepted checkpoint q_xi geometry differs")
    packet = PC1PosePacketV1(
        active=True,
        pair_count=600,
        xi_scales=parameter_map.xi_scales,
        residual_scale=parameter_map.residual_scale,
        q_xi=q_xi,
        q_luma_phase=np.zeros((parameter_map.knot_count, 4), dtype=np.int8),
    )
    if _sha256(serialize_pc1_packet(packet)) != checkpoint["packet_sha256"]:
        raise J12Error("PC1 accepted checkpoint packet custody differs")
    return packet


def _movable_layer(receiver: Any) -> Any:
    try:
        return next(layer for layer in receiver.layers if layer.role == "Movable")
    except StopIteration as exc:
        raise J12Error("PC1 rehome parent lacks Movable layer") from exc


def _forward_cells_pose(segnet: Any, posenet: Any, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from tools.measure_ddm_menu1_realized_flip_menu import _forward

    return _forward(segnet, posenet, camera)


def _rehomed_endpoint(
    *,
    config: Mapping[str, Any],
    typed_hash: str,
    endpoint_id: str,
    parent_archive: bytes,
    packet: PC1PosePacketV1,
    labels: np.ndarray,
    poses: np.ndarray,
    scorers: tuple[Any, Any],
) -> tuple[dict[str, Any], bytes]:
    output = Path(config["output_root"])
    root = output / "03_pc1_rehome" / endpoint_id
    final_path = root / "n600.json"
    parent_sha = _sha256(parent_archive)
    adapter_archive = build_source_preserving_pc1_adapter_archive(
        parent_archive=parent_archive,
        parent_sha256=parent_sha,
        packet=packet,
    )
    parsed_parent, parsed_packet, _ = parse_source_preserving_pc1_adapter_archive(
        adapter_archive,
        expected_parent_archive=parent_archive,
        expected_parent_sha256=parent_sha,
        zero_home_packet=packet,
    )
    if parsed_parent != parent_archive or serialize_pc1_packet(parsed_packet) != serialize_pc1_packet(packet):
        raise J12Error("PC1 source-preserving adapter parse-back differs")
    if final_path.exists():
        value = _read_json(final_path)
        if value.get("typed_config_hash") != typed_hash or value.get("archive_sha256") != _sha256(adapter_archive):
            raise J12Error(f"preserved PC1 endpoint differs: {endpoint_id}")
        return value, adapter_archive
    parent_receiver = receive_joint_descent_archive(parent_archive)
    movable_layer = _movable_layer(parent_receiver)
    accumulator = _endpoint_accumulator()
    chunk_rows: list[dict[str, Any]] = []
    packet_sha = _sha256(serialize_pc1_packet(packet))
    for start in range(0, 600, 32):
        stop = min(start + 32, 600)
        chunk_path = root / f"chunk_{start:04d}_{stop:04d}.json"
        if chunk_path.exists():
            row = _read_json(chunk_path)
            if (
                row.get("pair_range") != [start, stop]
                or row.get("packet_sha256") != packet_sha
                or row.get("archive_sha256") != _sha256(adapter_archive)
            ):
                raise J12Error(f"preserved PC1 chunk differs: {chunk_path}")
            chunk_rows.append(row)
            continue
        ids = tuple(range(start, stop))
        parent_camera = parent_receiver.render_camera_pairs(ids)
        movable_masks = np.stack(
            [
                parent_receiver._mask_for_layer(
                    movable_layer,
                    pair_id,
                    replace_g1_movable=True,
                )
                for pair_id in ids
            ],
            axis=0,
        ).astype(np.bool_)
        camera = receive_source_preserving_pc1_camera_pairs(
            parent_camera=parent_camera,
            packet=packet,
            pair_ids=ids,
            movable_masks=movable_masks,
        )
        cells, pose6 = _forward_cells_pose(scorers[0], scorers[1], camera)
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        local = _endpoint_accumulator()
        _accumulate_endpoint(
            local,
            cells=cells,
            pose6=pose6,
            target=target,
            target_pose=target_pose,
        )
        row = {
            "schema": "ddm_j12_pc1_rehomed_n600_chunk.v1",
            "endpoint_id": endpoint_id,
            "pair_range": [start, stop],
            "packet_sha256": packet_sha,
            "parent_archive_sha256": parent_sha,
            "archive_sha256": _sha256(adapter_archive),
            "errors": local["errors"],
            "sites": local["sites"],
            "pose_squared_error_sum": local["pose_sse"],
            "pose_coordinates": local["pose_coordinates"],
            "class_errors": np.asarray(local["class_errors"]).tolist(),
            "class_sites": np.asarray(local["class_sites"]).tolist(),
            "parent_camera_sha256": _sha256(parent_camera.tobytes(order="C")),
            "camera_sha256": _sha256(camera.tobytes(order="C")),
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        _atomic_json(chunk_path, row)
        chunk_rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "pc1_rehomed_chunk",
                    "endpoint_id": endpoint_id,
                    "pair_range": [start, stop],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del parent_camera, movable_masks, camera, cells, pose6
        gc.collect()
    for row in chunk_rows:
        accumulator["errors"] += int(row["errors"])
        accumulator["sites"] += int(row["sites"])
        accumulator["pose_sse"] += float(row["pose_squared_error_sum"])
        accumulator["pose_coordinates"] += int(row["pose_coordinates"])
        accumulator["class_errors"] += np.asarray(row["class_errors"], dtype=np.int64)
        accumulator["class_sites"] += np.asarray(row["class_sites"], dtype=np.int64)
    endpoint = _finalize_endpoint(accumulator, archive=adapter_archive)
    endpoint.update(
        {
            "schema": "ddm_j12_pc1_rehomed_n600_verdict.v1",
            "typed_config_hash": typed_hash,
            "endpoint_id": endpoint_id,
            "packet_sha256": packet_sha,
            "parent_archive_sha256": parent_sha,
            "receiver_equation": "parent_plus_pc1_packet_minus_pc1_active_zero",
            "active_zero_archive_byte_identity": True,
            "receiver_parseback_identity": True,
            "chunk_count": len(chunk_rows),
            "research_only": True,
            "promotion_eligible": False,
        }
    )
    _atomic_json(final_path, endpoint)
    _atomic_bytes(root / "archive.zip.receipt-bytes", adapter_archive)
    return endpoint, adapter_archive


def _baseline_state(row: Mapping[str, Any]) -> RealizedObjectiveState:
    return RealizedObjectiveState(
        d_seg=float(row["d_seg"]),
        d_pose=float(row["d_pose"]),
        archive_bytes=int(row["archive_bytes"]),
    )


def _price(
    baseline: Mapping[str, Any],
    endpoint: Mapping[str, Any],
) -> dict[str, Any]:
    delta = pure_priced_realized_delta(
        _baseline_state(baseline),
        _baseline_state(endpoint),
    )
    return {
        "seg_term": delta.seg_term,
        "pose_term": delta.pose_term,
        "rate_term": delta.rate_term,
        "joint_delta": delta.joint_delta,
        "accepted": delta.accepted,
        "acceptance_authority": "strict_realized_joint_delta_s_lt_zero",
    }


def _standard_endpoint_from_settled(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "archive_sha256": row["archive_sha256"],
        "archive_bytes": int(row["archive_bytes"]),
        "d_seg": float(row["d_seg"]),
        "d_pose": float(row["d_pose"]),
        "per_class": row.get("per_class"),
        "evidence_axis": EVIDENCE_AXIS,
        "evidence_source": "SETTLED_EXACT_N600_BATCH32_HASH_BOUND_RECEIPT",
        "score_claim": False,
    }


def _materialize_and_price(
    config: Mapping[str, Any],
    typed_hash: str,
    proposal_index: Mapping[str, Any],
    jacobians: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(config["output_root"])
    final_path = output / "04_decomposition_pricing" / "receipt.json"
    if final_path.exists():
        value = _read_json(final_path)
        if value.get("typed_config_hash") != typed_hash:
            raise J12Error("preserved decomposition/pricing receipt differs")
        return value
    _free_memory_receipt(
        "J12_N600_PC1_REHOME",
        int(config["minimum_free_memory_gib"]),
    )
    labels = open_stored_npy_memmap(
        _resolve(config["source_artifacts"]["target_cache"]["path"]),
        "lstars",
    )
    poses = open_stored_npy_memmap(
        _resolve(config["source_artifacts"]["target_cache"]["path"]),
        "gt_poses",
    )
    segnet, posenet, _scorer_custody = _load_models(config)
    scorers = (segnet, posenet)
    packets = {step: _pc1_packet(config, step) for step in (8, 16)}
    source_archives = {
        "W_joint_step50_live": _resolve(config["source_artifacts"]["w_joint_step50"]["path"]).read_bytes(),
        "W_seg": _resolve(config["source_artifacts"]["w_seg"]["path"]).read_bytes(),
    }
    baselines = config["source_baselines"]
    rehome_endpoints: dict[str, dict[str, Any]] = {}
    rehome_archives: dict[str, bytes] = {}
    for base_id, step in (
        ("W_joint_step50_live", 8),
        ("W_joint_step50_live", 16),
        ("W_seg", 16),
    ):
        endpoint_id = f"{base_id}__pc1_accepted_{step:03d}"
        endpoint, archive = _rehomed_endpoint(
            config=config,
            typed_hash=typed_hash,
            endpoint_id=endpoint_id,
            parent_archive=source_archives[base_id],
            packet=packets[step],
            labels=labels,
            poses=poses,
            scorers=scorers,
        )
        rehome_endpoints[endpoint_id] = endpoint
        rehome_archives[endpoint_id] = archive
    local_descent = []
    for step in (0, 8, 16):
        if step == 0:
            endpoint = {
                **baselines["W_joint_step50_live"],
                "active_zero_archive_byte_identity": True,
            }
        else:
            endpoint = rehome_endpoints[f"W_joint_step50_live__pc1_accepted_{step:03d}"]
        local_descent.append(
            {
                "accepted_step": step,
                "archive_sha256": endpoint["archive_sha256"],
                "archive_bytes": endpoint["archive_bytes"],
                "d_seg": endpoint["d_seg"],
                "d_pose": endpoint["d_pose"],
                "pure_priced_from_rehomed_step0": _price(
                    baselines["W_joint_step50_live"],
                    endpoint,
                ),
            }
        )
    proposal_receipts = _load_proposal_receipts(config)
    tables: dict[str, Any] = {}
    accepted_rows: list[dict[str, Any]] = []
    for base_id in ("W_joint_step50_live", "W_seg"):
        baseline = baselines[base_id]
        base_archive = source_archives[base_id]
        singles: list[dict[str, Any]] = []
        composites: list[dict[str, Any]] = []
        for proposal_id in SEALED_OPENING_PROPOSALS:
            jacobian = jacobians["proposals"][proposal_id]
            for component_kind, projector_key in (
                ("pose-null_seg", "pose_jacobian"),
                ("seg-null_pose", "seg_rank4_inner_jacobian"),
            ):
                coefficient = float(jacobian[projector_key]["null_projector"][0][0])
                if math.isclose(coefficient, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
                    archive = base_archive
                    endpoint = {
                        **baseline,
                        "evidence_axis": EVIDENCE_AXIS,
                        "evidence_source": ("EXACT_BASE_ARCHIVE_IDENTITY_AFTER_INTEGER_ZERO_PROJECTION"),
                        "score_claim": False,
                    }
                    integer_coefficient = 0
                elif math.isclose(coefficient, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                    integer_coefficient = 1
                    if base_id == "W_joint_step50_live":
                        archive = _proposal_archive_path(
                            output,
                            proposal_id,
                            "plus",
                        ).read_bytes()
                        endpoint = _standard_endpoint_from_settled(proposal_receipts[proposal_id])
                    else:
                        lift = lift_v15_archive(base_archive)
                        delta = _load_proposal_state(output, proposal_id)
                        archive, _ = compile_parameterized_archive(
                            lift,
                            delta,
                            include_lane_programs=False,
                        )
                        raise J12Error(
                            "nonzero W_seg component requires a fresh exact n600 endpoint; "
                            "the bounded J12 implementation refuses unpriced transfer"
                        )
                else:
                    raise J12Error("one-dimensional null projector is neither zero nor identity")
                component_path = (
                    output
                    / "04_decomposition_pricing"
                    / base_id
                    / "singles"
                    / f"{_proposal_slug(proposal_id)}__{component_kind}.zip.receipt-bytes"
                )
                _atomic_bytes(component_path, archive)
                if _sha256(archive) != endpoint["archive_sha256"]:
                    raise J12Error("component archive/endpoint SHA custody differs")
                row = {
                    "component_id": f"{proposal_id}::{component_kind}",
                    "base_id": base_id,
                    "source_proposal_id": proposal_id,
                    "component_kind": component_kind,
                    "projector_source": projector_key,
                    "projector_coefficient": coefficient,
                    "integer_realized_coefficient": integer_coefficient,
                    "receiver_parseback_identity": True,
                    "archive_path": str(component_path),
                    "endpoint": endpoint,
                    "pure_priced_realized_delta": _price(baseline, endpoint),
                    "epistemic_status": "MEASURED_OR_HASH_IDENTICAL_TO_MEASURED_EXACT_N600_BATCH32",
                }
                singles.append(row)
                if row["pure_priced_realized_delta"]["accepted"]:
                    accepted_rows.append(row)
            pose_null_row = singles[-2]
            parent_archive = Path(pose_null_row["archive_path"]).read_bytes()
            composite_id = f"{proposal_id}::pose_null_seg+rehomed_pc1_pose"
            endpoint_id = (
                f"{base_id}__pc1_accepted_016"
                if parent_archive == base_archive
                else f"{base_id}__{_proposal_slug(proposal_id)}__pc1_accepted_016"
            )
            if endpoint_id not in rehome_endpoints:
                endpoint, composite_archive = _rehomed_endpoint(
                    config=config,
                    typed_hash=typed_hash,
                    endpoint_id=endpoint_id,
                    parent_archive=parent_archive,
                    packet=packets[16],
                    labels=labels,
                    poses=poses,
                    scorers=scorers,
                )
                rehome_endpoints[endpoint_id] = endpoint
                rehome_archives[endpoint_id] = composite_archive
            endpoint = rehome_endpoints[endpoint_id]
            composite_archive = rehome_archives[endpoint_id]
            composite_path = (
                output
                / "04_decomposition_pricing"
                / base_id
                / "composites"
                / f"{_proposal_slug(proposal_id)}__pose_null_seg_plus_pc1.zip.receipt-bytes"
            )
            _atomic_bytes(composite_path, composite_archive)
            row = {
                "composite_id": composite_id,
                "base_id": base_id,
                "source_proposal_id": proposal_id,
                "members": [
                    pose_null_row["component_id"],
                    "rehomed_pc1_accepted_016_pose_coordinate",
                ],
                "receiver_parseback_identity": True,
                "archive_path": str(composite_path),
                "endpoint": endpoint,
                "pure_priced_realized_delta": _price(baseline, endpoint),
                "epistemic_status": "MEASURED_EXACT_N600_BATCH32",
            }
            composites.append(row)
            if row["pure_priced_realized_delta"]["accepted"]:
                accepted_rows.append(row)
        derivative_weight = 5.0 / math.sqrt(10.0 * float(baseline["d_pose"]))
        tables[base_id] = {
            "baseline": baseline,
            "pose_derivative_weight": derivative_weight,
            "pose_derivative_weight_equation": "5/sqrt(10*d_pose)",
            "break_even_ratio": 1.0,
            "fixed_R_star_used": False,
            "single_component_rows": singles,
            "composite_rows": composites,
            "single_count": len(singles),
            "composite_count": len(composites),
        }
    raw_wseg = _read_json(_resolve(config["source_artifacts"]["w_seg_raw_x_plus_verdict"]["path"]))
    result = {
        "schema": "ddm_j12_decomposition_pricing.v1",
        "typed_config_hash": typed_hash,
        "pc1_adapter": {
            "equation": "parent_plus_pc1_packet_minus_pc1_active_zero",
            "active_zero_archive_bytes": len(source_archives["W_joint_step50_live"]),
            "active_zero_archive_sha256": _sha256(source_archives["W_joint_step50_live"]),
            "expected_source_archive_bytes": 138813,
            "expected_source_archive_sha256": ("2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"),
            "archive_byte_identity": True,
            "local_pose_descent_remeasurement": local_descent,
            "old_non_preserving_ratio_transferred": False,
        },
        "tables": tables,
        "counts": {
            "bases": 2,
            "singles_per_base": 8,
            "composites_per_base": 4,
            "singles_total": 16,
            "composites_total": 8,
        },
        "historical_raw_w_seg_x_plus": {
            "archive_sha256": raw_wseg["archive_sha256"],
            "archive_bytes": raw_wseg["archive_bytes"],
            "d_seg": raw_wseg["d_seg"],
            "d_pose": raw_wseg["d_pose"],
            "pure_priced_delta": raw_wseg["pure_priced_delta"],
            "decomposition_authority": False,
            "reason": "raw proposal precedes receiver-J null projection",
        },
        "accepted_rows": [
            {
                "id": row.get("component_id", row.get("composite_id")),
                "base_id": row["base_id"],
                "joint_delta": row["pure_priced_realized_delta"]["joint_delta"],
            }
            for row in accepted_rows
        ],
        "objective_gate_contradictions": [],
        "acceptance_rule": "strict_realized_joint_delta_s_lt_zero",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
    }
    _atomic_json(final_path, result)
    return result


def _conditional_checkpoint_path(output: Path, step: int) -> Path:
    return output / "05_conditional_smoke" / "checkpoints" / f"step_{step:03d}.npz"


def _load_conditional_state(
    output: Path,
    *,
    parameter_count: int,
    typed_hash: str,
) -> tuple[AdamStateV1, list[dict[str, Any]]]:
    latest_state = initial_adam_state(parameter_count)
    telemetry: list[dict[str, Any]] = []
    gap = False
    for step in range(1, 5):
        path = _conditional_checkpoint_path(output, step)
        metadata_path = path.with_suffix(".json")
        if not path.exists() or not metadata_path.exists():
            gap = True
            continue
        if gap:
            raise J12Error("conditional-smoke checkpoints contain a gap")
        metadata = _read_json(metadata_path)
        if (
            metadata.get("typed_config_hash") != typed_hash
            or metadata.get("step") != step
            or _sha256(path.read_bytes()) != metadata.get("npz_sha256")
        ):
            raise J12Error(f"conditional-smoke checkpoint custody differs: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            latest_state = AdamStateV1(
                step=int(arrays["step"][0]),
                theta=np.asarray(arrays["theta"], dtype=np.float32),
                ema=np.asarray(arrays["ema"], dtype=np.float32),
                first_moment=np.asarray(arrays["first_moment"], dtype=np.float32),
                second_moment=np.asarray(arrays["second_moment"], dtype=np.float32),
            )
        telemetry = [dict(row) for row in metadata["telemetry"]]
    return latest_state, telemetry


def _conditional_live_ema_smoke(
    config: Mapping[str, Any],
    typed_hash: str,
    pricing: Mapping[str, Any],
) -> dict[str, Any]:
    """Run four J10-engine steps, then exact-score fixed-PC1 live and EMA parents."""

    output = Path(config["output_root"])
    final_path = output / "05_conditional_smoke" / "receipt.json"
    if final_path.exists():
        value = _read_json(final_path)
        if value.get("typed_config_hash") != typed_hash or value.get("completed") is not True:
            raise J12Error("preserved conditional-smoke receipt differs")
        return value
    if not pricing["accepted_rows"]:
        return {
            "schema": "ddm_j12_conditional_live_ema_smoke.v1",
            "required": False,
            "completed": False,
            "reason": "NO_REALIZED_JOINT_NEGATIVE_ROW",
        }
    memory = _free_memory_receipt(
        "J12_CONDITIONAL_SMOKE",
        int(config["minimum_free_memory_gib"]),
    )
    ticket_path = _resolve(config["source_artifacts"]["j10_ticket"]["path"])
    typed = DirectDescriptionJointDescentTypedConfigV1.from_ticket(ticket_path)
    if typed.typed_config_hash() != EXPECTED_TYPED_J10_HASH:
        raise J12Error("conditional smoke J10 typed identity differs")
    source = _resolve(config["source_artifacts"]["w_joint_step50"]["path"]).read_bytes()
    lift = lift_v15_archive(source)
    target_path = _resolve(config["source_artifacts"]["target_cache"]["path"])
    labels = open_stored_npy_memmap(target_path, "lstars")
    poses = open_stored_npy_memmap(target_path, "gt_poses")
    schedule = typed.full_run_schedule
    if schedule is None or schedule.warm_start_reform is None:
        raise J12Error("conditional smoke lacks J10 schedule/reform")
    reform = schedule.warm_start_reform
    active_groups = reform.opening_active_groups
    groups = parameter_group_indices(lift)
    active = set().union(*(groups[name] for name in active_groups))
    state, telemetry = _load_conditional_state(
        output,
        parameter_count=len(lift.parameter_names),
        typed_hash=typed_hash,
    )
    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    with temporary_mlx_device("gpu"):
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(
            typed.upstream_root,
            device="cpu",
        )
        model = DirectDescriptionJointDescentMLXModule(
            lift=lift,
            scorer_adapter=adapter,
            seg_targets=labels,
            pose_targets=poses,
        )
        while state.step < 4:
            pair_start = (
                schedule.warm_start_pair
                if state.step < schedule.warm_start_steps
                else ((state.step - 1) * schedule.train_batch) % 600
            )
            pair_ids = tuple((pair_start + offset) % 600 for offset in range(schedule.train_batch))
            (
                base_camera,
                template_masks,
                basis,
                basis_indices,
                local_theta,
                _current_archive,
            ) = realized_training_state(
                lift,
                state.theta,
                pair_ids=pair_ids,
                active_groups=active_groups,
                include_lane_programs=False,
            )
            loss, gradient = model.loss_and_grad(
                local_theta,
                pair_ids=pair_ids,
                base_camera=base_camera,
                template_masks=template_masks,
                realized_secant_basis=basis,
                realized_secant_indices=basis_indices,
                # Keep the exact J10 PoseFinish gate closed; the PC1 row is not
                # misclassified as a component-safe residual admission.
                pose_objective_weight=0.0,
            )
            gradient[[index for index in range(len(gradient)) if index not in active]] = 0.0
            rewarmup = linear_rewarmup_factor(
                completed_steps=state.step,
                rewarmup_steps=reform.lr_rewarmup_steps,
                floor=reform.lr_rewarmup_floor,
            )
            candidate = clipped_adam_step(
                state,
                opening_candidate_gradient(
                    lift,
                    "local_exact_gradient",
                    gradient,
                    active_pair_ids=reform.opening_candidate_pair_ids,
                ),
                learning_rate=(schedule.learning_rate_quantum_fraction * rewarmup * 32.0),
                grad_clip=typed.grad_clip,
                ema_decay=typed.ema_decay,
                beta2=reform.adam_beta2,
                maximum_update=reform.maximum_continuous_update_quantum_fraction,
                theta_lattice_denominator=reform.proposal_q8_denominator,
            )
            candidate, geometry_events = project_adam_state_geometry(lift, candidate)
            live_parent, live_realized = compile_parameterized_archive(
                lift,
                candidate.theta,
                include_lane_programs=False,
            )
            ema_parent, ema_realized = compile_parameterized_archive(
                lift,
                candidate.ema,
                include_lane_programs=False,
            )
            state = candidate
            row = {
                "step": state.step,
                "pair_ids": list(pair_ids),
                "loss": float(loss),
                "gradient_norm": float(np.linalg.norm(gradient.astype(np.float64))),
                "pose_objective_weight": 0.0,
                "pose_finish_gate": "CLOSED_COMPONENT_SAFE_RESIDUAL_ADMISSION_NOT_PROVEN",
                "learning_rate_multiplier": 32.0,
                "learning_rate": (schedule.learning_rate_quantum_fraction * rewarmup * 32.0),
                "live_parent_archive_sha256": _sha256(live_parent),
                "live_parent_archive_bytes": len(live_parent),
                "live_realized_parameter_count": int(np.count_nonzero(live_realized)),
                "ema_parent_archive_sha256": _sha256(ema_parent),
                "ema_parent_archive_bytes": len(ema_parent),
                "ema_realized_parameter_count": int(np.count_nonzero(ema_realized)),
                "geometry_events": list(geometry_events),
                "score_claim": False,
            }
            telemetry.append(row)
            checkpoint_path = _conditional_checkpoint_path(output, state.step)
            checkpoint_payload = _canonical_npz(
                {
                    "ema": state.ema.astype("<f4", copy=False),
                    "first_moment": state.first_moment.astype("<f4", copy=False),
                    "second_moment": state.second_moment.astype("<f4", copy=False),
                    "step": np.asarray([state.step], dtype="<i8"),
                    "theta": state.theta.astype("<f4", copy=False),
                }
            )
            _atomic_bytes(checkpoint_path, checkpoint_payload)
            _atomic_json(
                checkpoint_path.with_suffix(".json"),
                {
                    "schema": "ddm_j12_conditional_smoke_checkpoint.v1",
                    "typed_config_hash": typed_hash,
                    "step": state.step,
                    "npz_path": str(checkpoint_path),
                    "npz_bytes": len(checkpoint_payload),
                    "npz_sha256": _sha256(checkpoint_payload),
                    "telemetry": telemetry,
                    "complete_state_preserved": True,
                    "score_claim": False,
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "conditional_j10_engine_step",
                        "step": state.step,
                        "live_parent_sha256": row["live_parent_archive_sha256"],
                        "ema_parent_sha256": row["ema_parent_archive_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    live_parent, live_realized = compile_parameterized_archive(
        lift,
        state.theta,
        include_lane_programs=False,
    )
    ema_parent, ema_realized = compile_parameterized_archive(
        lift,
        state.ema,
        include_lane_programs=False,
    )
    packet = _pc1_packet(config, 16)
    segnet, posenet, scorer_custody = _load_models(config)
    live_endpoint, live_archive = _rehomed_endpoint(
        config=config,
        typed_hash=typed_hash,
        endpoint_id="conditional_step004_live_pc1_accepted_016",
        parent_archive=live_parent,
        packet=packet,
        labels=labels,
        poses=poses,
        scorers=(segnet, posenet),
    )
    ema_endpoint, ema_archive = _rehomed_endpoint(
        config=config,
        typed_hash=typed_hash,
        endpoint_id="conditional_step004_ema_pc1_accepted_016",
        parent_archive=ema_parent,
        packet=packet,
        labels=labels,
        poses=poses,
        scorers=(segnet, posenet),
    )
    initial_endpoint = pricing["tables"]["W_joint_step50_live"]["composite_rows"][0]["endpoint"]
    live_delta = _price(initial_endpoint, live_endpoint)
    ema_delta = _price(initial_endpoint, ema_endpoint)
    worst_geometry = typed.worst_geometry_memory_contract
    receipt = {
        "schema": "ddm_j12_conditional_live_ema_smoke.v1",
        "typed_config_hash": typed_hash,
        "j10_typed_config_hash": typed.typed_config_hash(),
        "required": True,
        "completed": True,
        "bounded_steps": 4,
        "step_governor": {"minimum": 4, "maximum": 8, "actual": 4},
        "engine": "tac.optimization.direct_description_joint_descent",
        "engine_or_acceptance_gate_weakened": False,
        "pose_finish_gate_weakened": False,
        "fixed_pc1_packet_sha256": _sha256(serialize_pc1_packet(packet)),
        "initial_merged_archive": {
            "archive_sha256": initial_endpoint["archive_sha256"],
            "archive_bytes": initial_endpoint["archive_bytes"],
            "d_seg": initial_endpoint["d_seg"],
            "d_pose": initial_endpoint["d_pose"],
        },
        "telemetry": telemetry,
        "live": {
            "parent_realized_parameter_count": int(np.count_nonzero(live_realized)),
            "merged_archive_sha256": _sha256(live_archive),
            "merged_archive_bytes": len(live_archive),
            "endpoint": live_endpoint,
            "pure_priced_delta_vs_initial_merged": live_delta,
        },
        "ema": {
            "parent_realized_parameter_count": int(np.count_nonzero(ema_realized)),
            "merged_archive_sha256": _sha256(ema_archive),
            "merged_archive_bytes": len(ema_archive),
            "endpoint": ema_endpoint,
            "pure_priced_delta_vs_initial_merged": ema_delta,
        },
        "merged_main_worst_geometry_reseal": {
            "status": "PREPARED_REVIEW_REQUIRED",
            "merged_initial_archive_sha256": initial_endpoint["archive_sha256"],
            "j10_parent_archive_sha256": _sha256(source),
            "pc1_packet_sha256": _sha256(serialize_pc1_packet(packet)),
            "worst_geometry_contract": (None if worst_geometry is None else worst_geometry.to_payload()),
            "memory_preflight": memory,
            "scorer_custody": scorer_custody,
            "receiver_parseback_identity": True,
            "main_review_required": True,
        },
        "fire_authority": "MAIN_ONLY_AFTER_LANDING_REVIEW",
        "pointer_moved": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
    }
    _atomic_json(final_path, receipt)
    return receipt


def _final_receipt(
    config: Mapping[str, Any],
    typed_hash: str,
    preflight: Mapping[str, Any],
    proposals: Mapping[str, Any],
    jacobians: Mapping[str, Any],
    pricing: Mapping[str, Any],
    conditional: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(config["output_root"])
    accepted = pricing["accepted_rows"]
    if accepted and conditional.get("completed") is True:
        conditional_smoke = {
            "ran": True,
            "required": True,
            "receipt_path": str(output / "05_conditional_smoke" / "receipt.json"),
            "receipt_sha256": _sha256((output / "05_conditional_smoke" / "receipt.json").read_bytes()),
            "bounded_steps": conditional["bounded_steps"],
            "engine_or_acceptance_gate_weakened": False,
            "merged_main_worst_geometry_reseal": conditional["merged_main_worst_geometry_reseal"],
            "fire_authority": "MAIN_ONLY_AFTER_LANDING_REVIEW",
        }
        verdict = "MEASURED_J12_REHOMED_PC1_NEGATIVE_CONDITIONAL_SMOKE_COMPLETE"
        verdict_scope = (
            "MEASURED local advisory: rehomed PC1 accepted-016 is realized-joint "
            "negative on both named bases; four J10-engine Seg-gated steps plus "
            "exact live/EMA n600 endpoints are complete. FIRE and promotion remain MAIN-only."
        )
        obstruction = (
            "no execution obstruction remains inside J12; MAIN landing review "
            "and FIRE authority remain intentionally outstanding"
        )
    elif accepted:
        conditional_smoke = {
            "ran": False,
            "required": True,
            "reason": (
                "BLOCKED_CONDITIONAL_SMOKE_REQUIRES_MAIN_RESEAL_OF_NEW_COMPONENT "
                "BEFORE_J10_ENGINE_CAN_LAWFULLY_ACCEPT_A_DIFFERENT_SOURCE_ARCHIVE"
            ),
            "accepted_rows": accepted,
            "j10_engine_or_gate_weakened": False,
            "main_review_required": True,
        }
        verdict = "BLOCKED_J12_CONDITIONAL_SMOKE_RESEAL_PRECONDITION"
        verdict_scope = (
            "PRECONDITION: one or more exact J12 rows are realized-joint negative, "
            "but the sealed J10 engine cannot consume a different warm-start SHA "
            "without a reviewed typed reseal. This is not promotion authority."
        )
        obstruction = "MAIN reviewed typed warm-start/worst-geometry reseal is owed before live/EMA smoke"
    else:
        conditional_smoke = {
            "ran": False,
            "required": False,
            "reason": "NO_SINGLE_OR_COMPOSITE_ON_EITHER_BASE_HAS_REALIZED_JOINT_DELTA_S_LT_ZERO",
            "j10_engine_or_gate_weakened": False,
        }
        verdict = "FORMULATION_NEGATIVE_J12_PROPOSAL_SPAN_NULL_DECOMPOSITION"
        verdict_scope = (
            "FORMULATION negative only for the four sealed J10 scalar proposal "
            "spans, their exact receiver-J null projections, and the rehomed PC1 "
            "accepted-016 composites on W_joint_step50_live and W_seg. It is not "
            "a family negative for other scorer-recursive coordinates."
        )
        obstruction = (
            "the fully measured Pose6 and rank4-inner Seg Jacobians have full "
            "rank on every one-dimensional sealed proposal span, collapsing both "
            "null projections to active-zero; rehomed PC1 adds no strict negative row"
        )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "delegation_checkpoint_key": config["delegation_checkpoint_key"],
        "typed_config_hash": typed_hash,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "preflight_receipt": {
            "path": str(output / "00_preflight.json"),
            "sha256": _sha256((output / "00_preflight.json").read_bytes()),
        },
        "proposal_receipt": {
            "path": str(_proposal_index_path(output)),
            "sha256": _sha256(_proposal_index_path(output).read_bytes()),
            "proposal_count": proposals["proposal_count"],
        },
        "jacobian_receipt": {
            "path": str(output / "02_jacobians" / "index.json"),
            "sha256": _sha256((output / "02_jacobians" / "index.json").read_bytes()),
            "measured_jacobian_count": jacobians["measured_jacobian_count"],
        },
        "pricing_receipt": {
            "path": str(output / "04_decomposition_pricing" / "receipt.json"),
            "sha256": _sha256((output / "04_decomposition_pricing" / "receipt.json").read_bytes()),
            "counts": pricing["counts"],
        },
        "preflight_summary": {
            "source_archive_sha256": config["source_baselines"]["W_joint_step50_live"]["archive_sha256"],
            "source_archive_bytes": config["source_baselines"]["W_joint_step50_live"]["archive_bytes"],
            "pair_count": 600,
            "batch_size": 32,
            "available_memory_bytes": preflight["memory"]["available_bytes"],
            "paid_dispatch": False,
            "contest_eval": False,
        },
        "acceptance_rule": {
            "function": "tac.optimization.pure_priced_realized_objective.pure_priced_realized_delta",
            "admit_iff": "realized_joint_delta_s_lt_zero",
            "derivative_weight": "5/sqrt(10*d_pose)",
            "break_even_ratio": 1.0,
            "fixed_R_star_used": False,
            "weakened": False,
        },
        "conditional_smoke": conditional_smoke,
        "verdict": verdict,
        "verdict_scope": verdict_scope,
        "named_residual_obstruction": obstruction,
        "pointer": config["pointer"],
        "pointer_moved": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_review_required": True,
        "main_landing_review": {
            "required": True,
            "gate": "tools/codex_landing_review_gate.py",
            "status": "OWED_BEFORE_MAIN_LANDING",
        },
    }
    final_path = output / (
        "ddm_j12_receiver_coordinate_custody_receipt_v2.json"
        if conditional.get("completed") is True
        else "ddm_j12_receiver_coordinate_custody_receipt.json"
    )
    _atomic_json(final_path, receipt)
    return receipt


def _run(config_path: Path) -> dict[str, Any]:
    config, typed_hash = _load_config(config_path)
    preflight = _preflight(config, typed_hash)
    proposals = _rederive_proposals(config, typed_hash)
    jacobians = _measure_jacobians(config, typed_hash, proposals)
    pricing = _materialize_and_price(config, typed_hash, proposals, jacobians)
    conditional = _conditional_live_ema_smoke(config, typed_hash, pricing)
    return _final_receipt(
        config,
        typed_hash,
        preflight,
        proposals,
        jacobians,
        pricing,
        conditional,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        receipt = _run(args.config.resolve())
    except J12Error as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, sort_keys=True))
        return 4
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
