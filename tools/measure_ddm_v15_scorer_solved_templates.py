# SPDX-License-Identifier: MIT
"""Solve counted row-band RGB templates encode-side through exact R + SegNet.

This is a local, false-authority measurement lane.  The frozen scorer exists
only in this encoder process.  Archives contain the solved shared uint8 patch
bytes and grammar parameters, never scorer weights, logits, gradients, or a
ground-truth argmax table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    REALIZATION_PAINT_ORDER,
    REALIZATION_PROFILE_MEMBER,
    SCORER_SOLVED_TEMPLATE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    _decode_lane_knots,
    _decode_lane_programs,
    _decode_realization_profile,
    compile_carrier_compose_archive,
    decode_scorer_solved_template_bank,
    encode_scorer_solved_template_bank,
    parse_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
    recursive_carrier_byte_rows,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionError,
    _read_regular_file_once,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    EVIDENCE_AXIS,
    POINTER_SCORE_TEXT,
    _load_models,
    _measure_candidate,
    _publish_immutable,
    _storage_preflight,
)
from tools.run_ddm_v9_carrier_compose import open_stored_npy_memmap  # noqa: E402

RESULT_SCHEMA = "ddm_v15_scorer_solved_template_receipt.v1"
SOLVER_SCHEMA = "ddm_v15_projected_margin_secant_solver.v1"
REPRESENTATIVE_ISLANDS = (447, 53, 416, 296, 547, 278, 501, 346)
BREAK_EVEN_SCORE_PER_BYTE = 25.0 / 37_545_489.0


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bound_bytes(path: Path, digest: str, name: str) -> bytes:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != digest:
        raise DirectDescriptionError(f"{name} SHA-256 differs")
    return payload


def _bound_json(path: Path, digest: str, name: str) -> dict[str, Any]:
    return json.loads(_bound_bytes(path, digest, name))


class DDMV15ScorerSolvedTemplateConfigV1(BaseModel):
    """Typed, bounded local advisory solve/measurement contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV15ScorerSolvedTemplateConfigV1"] = Field(
        default="DDMV15ScorerSolvedTemplateConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: str = Field(min_length=8)
    seed: StrictInt = 1234
    pair_start: StrictInt = Field(ge=0, le=599)
    pair_count: StrictInt = Field(ge=1, le=600)
    v14_receipt_path: str = Field(min_length=1)
    v14_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v14_archive_path: str = Field(min_length=1)
    v14_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    solve_archive_path: str = Field(min_length=1)
    solve_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_cache_path: str = Field(min_length=1)
    target_cache_bytes: StrictInt = Field(gt=0)
    target_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_root: str = Field(min_length=1)
    representative_source_pair_ids: tuple[StrictInt, ...] = REPRESENTATIVE_ISLANDS
    row_band_edges: tuple[StrictInt, ...] = (0, 128, 256, 384)
    optimization_steps: StrictInt = Field(default=8, ge=1, le=64)
    initial_step_u8: StrictFloat = Field(default=32.0, gt=0.0, le=255.0)
    maximum_fisher_cells: StrictInt = Field(default=8192, ge=128, le=65536)
    scorer_threads: StrictInt = Field(default=4, ge=1, le=16)
    scorer_batch_size: Literal[16] = 16
    template_source_path: str | None = None
    template_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    archive_box_bytes: Literal[160000] = 160000
    movable_gate: StrictFloat = Field(default=0.05, ge=0.0, le=1.0)
    max_candidate_stages_per_invocation: Literal[1] = 1
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMV15ScorerSolvedTemplateConfigV1:
        if (self.pair_start, self.pair_count) not in {(448, 64), (0, 600)}:
            raise ValueError("v15 measurement windows must be n64 [448,512) or full n600")
        if self.representative_source_pair_ids != REPRESENTATIVE_ISLANDS:
            raise ValueError("v15 development set is the preregistered eight representative islands")
        if len(set(self.representative_source_pair_ids)) != 8:
            raise ValueError("v15 representative islands must be unique")
        if len(self.row_band_edges) < 2 or self.row_band_edges[0] != 0 or self.row_band_edges[-1] != 384:
            raise ValueError("v15 row-band edges must partition [0,384]")
        if any(b <= a for a, b in pairwise(self.row_band_edges)):
            raise ValueError("v15 row-band edges must be strictly increasing")
        if (self.template_source_path is None) != (self.template_source_sha256 is None):
            raise ValueError("v15 template source path and SHA-256 must be paired")
        if self.pair_count == 600 and self.template_source_path is None:
            raise ValueError("full n600 verdict must consume the SHA-bound n64 solved template bank")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _role_templates(
    role: str,
    application: str,
    edges: tuple[int, ...],
    colour: np.ndarray,
) -> tuple[RowBandScorerTemplateV1, ...]:
    rgb = bytes(np.asarray(colour, dtype=np.uint8).tolist())
    return tuple(RowBandScorerTemplateV1(role, application, start, stop, 1, 1, rgb) for start, stop in pairwise(edges))


def _bank_with(
    inherited: tuple[RowBandScorerTemplateV1, ...],
    role_rows: tuple[RowBandScorerTemplateV1, ...],
) -> ScorerSolvedTemplateBankV1:
    return ScorerSolvedTemplateBankV1(
        tuple(
            sorted(
                (*inherited, *role_rows),
                key=lambda row: (
                    REALIZATION_PAINT_ORDER.index(row.role),
                    0 if row.application == "fill" else 1,
                    row.scorer_row_start,
                    row.scorer_row_stop,
                    row.patch_height,
                    row.patch_width,
                ),
            )
        )
    )


def _compile_from_v14(base_archive: bytes, bank: ScorerSolvedTemplateBankV1) -> bytes:
    members, _homes = parse_carrier_compose_archive(base_archive)
    if SCORER_SOLVED_TEMPLATE_MEMBER in members:
        raise DirectDescriptionError("v15 compiler expected an untemplated V14 base archive")
    profile = _decode_realization_profile(members.get(REALIZATION_PROFILE_MEMBER, b""))
    if profile is None or WORLDSHEET_G1_MEMBER not in members:
        raise DirectDescriptionError("v15 compiler requires the receiver-closed V14 G1 base")
    archive, _ = compile_carrier_compose_archive(
        members["predictor.zip"],
        worldsheet_g1_payload=members[WORLDSHEET_G1_MEMBER],
        lane_programs=_decode_lane_programs(members.get("predict/lane_periodic_programs.ddlp", b"")),
        lane_knots=_decode_lane_knots(members.get("predict/lane_drift_knots.ddlk", b"")),
        realization_profile=profile,
        scorer_solved_templates=bank,
    )
    return archive


def _apply_parameters(
    torch: Any,
    base_camera: np.ndarray,
    masks: tuple[np.ndarray, ...],
    parameters: Any,
) -> Any:
    output = torch.from_numpy(np.asarray(base_camera)).float()
    for index, mask in enumerate(masks):
        active = torch.from_numpy(mask)[:, None, :, :, None]
        colour = parameters[index].view(1, 1, 1, 1, 3)
        output = torch.where(active, colour, output)
    return output


def _seg_logits(segnet: Any, camera: Any) -> Any:
    return segnet(segnet.preprocess_input(camera.permute(0, 1, 4, 2, 3).contiguous()))


def _rank_fisher_cells(
    torch: Any, logits: Any, labels: np.ndarray, role_id: int, limit: int
) -> tuple[Any, dict[str, Any]]:
    target = torch.from_numpy(np.ascontiguousarray(labels)).long()
    role_logits = logits[:, role_id]
    rivals = logits.clone()
    rivals[:, role_id] = -torch.inf
    margin = role_logits - rivals.max(dim=1).values
    eligible = target == role_id
    locations = eligible.nonzero(as_tuple=False)
    values = margin[eligible]
    fisher = 0.5 / torch.cosh(torch.clamp(values / 2.0, -20.0, 20.0)).square()
    take = min(limit, int(fisher.numel()))
    order = torch.argsort(fisher, descending=True, stable=True)[:take]
    selected = torch.zeros_like(eligible)
    chosen = locations[order]
    selected[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    return selected, {
        "eligible_target_cells": int(fisher.numel()),
        "ranked_target_cells": take,
        "fisher_trace_min": float(fisher[order].min().item()) if take else 0.0,
        "fisher_trace_max": float(fisher[order].max().item()) if take else 0.0,
        "ranking_law": "top1_top2_margin_then_half_sech2_fisher_trace",
    }


def _oracle(
    torch: Any,
    segnet: Any,
    camera: np.ndarray,
    labels: np.ndarray,
    base_cells: np.ndarray,
    role_id: int,
) -> dict[str, Any]:
    tensor = torch.from_numpy(np.ascontiguousarray(camera)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        cells = segnet(segnet.preprocess_input(tensor)).argmax(dim=1).cpu().numpy().astype(np.uint8)
    role = labels == role_id
    errors = cells != labels
    base_correct_off_target = (labels != role_id) & (base_cells == labels)
    harmful = base_correct_off_target & errors
    return {
        "cells": cells,
        "total_errors": int(np.count_nonzero(errors)),
        "total_sites": int(errors.size),
        "role_errors": int(np.count_nonzero(errors & role)),
        "role_sites": int(np.count_nonzero(role)),
        "harmful_off_target_flips": int(np.count_nonzero(harmful)),
        "changed_off_target_cells": int(np.count_nonzero((labels != role_id) & (cells != base_cells))),
    }


def _solve_role(
    *,
    config: DDMV15ScorerSolvedTemplateConfigV1,
    receiver: Any,
    base_camera: np.ndarray,
    labels: np.ndarray,
    segnet: Any,
    torch: Any,
    role: str,
    role_id: int,
    application: str,
    inherited: tuple[RowBandScorerTemplateV1, ...],
) -> tuple[tuple[RowBandScorerTemplateV1, ...], dict[str, Any]]:
    colour = receiver.realization_profile.colour_for(role)
    rows = _role_templates(role, application, config.row_band_edges, colour)
    masks = tuple(
        receiver.template_camera_masks(
            tuple(value - receiver.predictor.source_pair_start for value in config.representative_source_pair_ids), row
        )
        for row in rows
    )
    parameters = torch.tensor([list(row.rgb_u8) for row in rows], dtype=torch.float32, requires_grad=True)
    base_tensor = torch.from_numpy(np.asarray(base_camera)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        base_logits = segnet(segnet.preprocess_input(base_tensor))
        base_cells = base_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
    selected, fisher = _rank_fisher_cells(torch, base_logits, labels, role_id, config.maximum_fisher_cells)
    current_u8 = np.asarray([list(row.rgb_u8) for row in rows], dtype=np.uint8)
    current_camera = (
        _apply_parameters(torch, base_camera, masks, torch.from_numpy(current_u8).float())
        .round()
        .clamp(0, 255)
        .byte()
        .numpy()
    )
    current = _oracle(torch, segnet, current_camera, labels, base_cells, role_id)
    initial = {key: value for key, value in current.items() if key != "cells"}
    step_u8 = float(config.initial_step_u8)
    accepted_rows: list[dict[str, Any]] = []
    attempted_rows: list[dict[str, Any]] = []
    minimum_improving_collateral: int | None = None
    for step in range(config.optimization_steps):
        if parameters.grad is not None:
            parameters.grad.zero_()
        camera = _apply_parameters(torch, base_camera, masks, parameters)
        logits = _seg_logits(segnet, camera)
        role_logits = logits[:, role_id]
        rivals = logits.clone()
        rivals[:, role_id] = -torch.inf
        margin = role_logits - rivals.max(dim=1).values
        target_loss = torch.nn.functional.softplus(-margin[selected]).mean()
        target_loss.backward()
        gradient = parameters.grad.detach().cpu().numpy()
        scale = float(np.max(np.abs(gradient), initial=0.0))
        if not math.isfinite(scale) or scale == 0.0:
            attempted_rows.append({"step": step, "status": "ZERO_OR_NONFINITE_GRADIENT"})
            break
        proposal = current_u8.astype(np.float64) - step_u8 * gradient / scale
        admitted: tuple[np.ndarray, dict[str, Any], float] | None = None
        for alpha in (0.25, 0.5, 1.0):
            candidate_u8 = np.clip(np.rint(current_u8 + alpha * (proposal - current_u8)), 0, 255).astype(np.uint8)
            if np.array_equal(candidate_u8, current_u8):
                continue
            candidate_camera = (
                _apply_parameters(torch, base_camera, masks, torch.from_numpy(candidate_u8).float())
                .round()
                .clamp(0, 255)
                .byte()
                .numpy()
            )
            oracle = _oracle(torch, segnet, candidate_camera, labels, base_cells, role_id)
            improvement = current["role_errors"] - oracle["role_errors"]
            if improvement > 0:
                slack = int(oracle["harmful_off_target_flips"])
                minimum_improving_collateral = (
                    slack if minimum_improving_collateral is None else min(minimum_improving_collateral, slack)
                )
            attempted_rows.append(
                {
                    "step": step,
                    "alpha": alpha,
                    "step_u8": step_u8,
                    "role_error_improvement": improvement,
                    "harmful_off_target_flips": oracle["harmful_off_target_flips"],
                    "changed_off_target_cells": oracle["changed_off_target_cells"],
                    "rgb_u8": candidate_u8.tolist(),
                }
            )
            if improvement > 0 and oracle["harmful_off_target_flips"] == 0:
                admitted = (candidate_u8, oracle, alpha)
                break
        if admitted is None:
            step_u8 *= 0.5
            parameters = torch.tensor(current_u8.astype(np.float32), requires_grad=True)
            if step_u8 < 0.5:
                break
            continue
        current_u8, current, alpha = admitted
        parameters = torch.tensor(current_u8.astype(np.float32), requires_grad=True)
        accepted_rows.append(
            {
                "step": step,
                "alpha": alpha,
                "role_errors": current["role_errors"],
                "harmful_off_target_flips": 0,
                "rgb_u8": current_u8.tolist(),
            }
        )
    solved_rows = tuple(
        RowBandScorerTemplateV1(
            row.role,
            row.application,
            row.scorer_row_start,
            row.scorer_row_stop,
            1,
            1,
            bytes(current_u8[index].tolist()),
        )
        for index, row in enumerate(rows)
    )
    changed = not all(a.rgb_u8 == b.rgb_u8 for a, b in zip(rows, solved_rows, strict=True))
    final = {key: value for key, value in current.items() if key != "cells"}
    payload_delta = len(encode_scorer_solved_template_bank(_bank_with(inherited, solved_rows)))
    score_gain = 100.0 * (initial["role_errors"] - final["role_errors"]) / max(1, initial["role_sites"])
    return solved_rows, {
        "schema": SOLVER_SCHEMA,
        "role": role,
        "application": application,
        "formulation": "Fisher-margin-ranked projected gradient through exact torch bilinear R; uint8 projection; realized secant hard oracle",
        "fisher_ranking": fisher,
        "initial": initial,
        "final": final,
        "changed_from_v14_prototype": changed,
        "accepted_step_count": len(accepted_rows),
        "accepted_steps": accepted_rows,
        "attempted_steps": attempted_rows,
        "zero_collateral_constraint": "no baseline-correct off-target cell may become wrong",
        "minimum_harmful_off_target_flips_among_improving_proposals": minimum_improving_collateral,
        "constraint_disposition": (
            "FEASIBLE_ZERO_COLLATERAL_IMPROVEMENT"
            if changed
            else "NO_ADMISSIBLE_PROPOSAL_IN_BOUNDED_PROJECTED_SECANT_SEARCH"
        ),
        "verdict_scope": "FORMULATION x preregistered eight-island development set; family remains open",
        "estimated_payload_bytes": payload_delta,
        "measured_development_score_gain": score_gain,
        "score_gain_per_payload_byte": score_gain / max(1, payload_delta),
        "break_even_score_per_byte": BREAK_EVEN_SCORE_PER_BYTE,
        "reverse_waterfill_admitted": changed and score_gain / max(1, payload_delta) >= BREAK_EVEN_SCORE_PER_BYTE,
    }


def _control_row(receipt: dict[str, Any]) -> dict[str, Any]:
    row = next(value for value in receipt["fixed_ladder"] if value["candidate"] == "islands")
    return {**row, "candidate": "v14_islands_inherited_control", "remeasured": False}


def _derive_full_p_identity_measurement(
    *,
    config: DDMV15ScorerSolvedTemplateConfigV1,
    root: Path,
    base_archive: bytes,
    final_archive: bytes,
    control: dict[str, Any],
) -> dict[str, Any] | None:
    """Reuse the inherited score only after exact full-P camera-byte identity."""

    if config.pair_count != 600:
        return None
    base_receiver = receive_carrier_compose_archive(base_archive)
    final_receiver = receive_carrier_compose_archive(final_archive)
    stage = root / "stage_checkpoints" / "full_p_camera_identity"
    rows: list[dict[str, Any]] = []
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        path = stage / f"batch_{start:04d}_{stop:04d}.json"
        if path.exists():
            row = json.loads(_read_regular_file_once(path))
        else:
            indexes = tuple(range(start, stop))
            base_camera = base_receiver.render_camera_pairs(indexes)
            final_camera = final_receiver.render_camera_pairs(indexes)
            row = {
                "schema": "ddm_v15_full_p_camera_identity_batch.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "local_pair_range": [start, stop],
                "base_camera_sha256": _sha256(base_camera.tobytes()),
                "final_camera_sha256": _sha256(final_camera.tobytes()),
                "byte_identical": bool(np.array_equal(base_camera, final_camera)),
                "camera_bytes_released_after_compare": True,
                "score_claim": False,
            }
            _publish_immutable(path, rfc8785_canonicalize(row))
        if row.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("v15 full-P identity checkpoint typed config differs")
        rows.append(row)
    if len(rows) != 38 or not all(row["byte_identical"] for row in rows):
        return None
    return {
        **control,
        "candidate": "v15_solved_templates",
        "archive_bytes": len(final_archive),
        "archive_sha256": _sha256(final_archive),
        "receiver_custody": dict(final_receiver.custody),
        "byte_streams": recursive_carrier_byte_rows(final_archive),
        "measurement_authority": ("DERIVED_FROM_EXACT_FULL_P_CAMERA_BYTE_IDENTITY_TO_INHERITED_V14_FROZEN_SCORER_ROW"),
        "full_p_camera_identity": {
            "pair_count": 600,
            "batch_count": 38,
            "batch_size": 16,
            "all_camera_bytes_identical": True,
            "digest_chain_sha256": _sha256(
                "".join(row["base_camera_sha256"] + row["final_camera_sha256"] for row in rows).encode()
            ),
        },
        "remeasured": False,
        "score_claim": False,
    }


def _load_template_source(config: DDMV15ScorerSolvedTemplateConfigV1) -> ScorerSolvedTemplateBankV1:
    payload = _bound_bytes(Path(config.template_source_path), config.template_source_sha256, "v15 template source")
    bank = decode_scorer_solved_template_bank(payload)
    if bank is None:
        raise DirectDescriptionError("v15 template source decoded empty")
    return bank


def _deterministic_storage_receipt(preflight: dict[str, Any]) -> dict[str, Any]:
    """Keep the fail-closed gate while excluding invocation-volatile observations."""

    if preflight.get("status") != "PASS" or preflight.get("free_space_gate_satisfied") is not True:
        raise DirectDescriptionError("v15 storage preflight did not pass")
    return {
        "output_tier": "local_small_receipt",
        "required_free_bytes": int(preflight["required_free_bytes"]),
        "observed_free_bytes_recorded": False,
        "free_space_gate_satisfied": True,
        "bulk_target_tier": str(preflight["bulk_target_tier"]),
        "bulk_target_read_only": bool(preflight["bulk_target_read_only"]),
        "status": "PASS",
    }


def run(config: DDMV15ScorerSolvedTemplateConfigV1, root: Path, semantic_argv: list[str]) -> Path:
    storage = _deterministic_storage_receipt(_storage_preflight(root.resolve()))
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v15_scorer_solved_templates_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed v15 receipt typed config differs")
        print(json.dumps({"resumed": True, "complete": True, "receipt": str(receipt_path)}))
        return receipt_path
    v14 = _bound_json(Path(config.v14_receipt_path), config.v14_receipt_sha256, "v14 receipt")
    base_archive = _bound_bytes(Path(config.v14_archive_path), config.v14_archive_sha256, "v14 archive")
    solve_archive = _bound_bytes(Path(config.solve_archive_path), config.solve_archive_sha256, "v14 solve archive")
    cache = Path(config.target_cache_path)
    if not cache.is_file() or cache.stat().st_size != config.target_cache_bytes:
        raise DirectDescriptionError("frozen n600 target cache bytes are unavailable")
    labels = open_stored_npy_memmap(cache, "lstars")
    poses = open_stored_npy_memmap(cache, "gt_poses")
    segnet = posenet = None
    scorer_custody = {
        **v14["scorer_custody"],
        "reuse_authority": "exact full-P camera-byte identity required before inherited-row reuse",
    }
    solver_rows: list[dict[str, Any]] = []
    if config.template_source_path is None:
        segnet, posenet, scorer_custody = _load_models(config)
        solve_receiver = receive_carrier_compose_archive(solve_archive)
        local_ids = tuple(
            value - solve_receiver.predictor.source_pair_start for value in config.representative_source_pair_ids
        )
        base_camera = solve_receiver.render_camera_pairs(local_ids)
        target = np.ascontiguousarray(labels[np.asarray(config.representative_source_pair_ids)])
        import torch

        movable_rows, movable_solver = _solve_role(
            config=config,
            receiver=solve_receiver,
            base_camera=base_camera,
            labels=target,
            segnet=segnet,
            torch=torch,
            role="Movable",
            role_id=3,
            application="fill",
            inherited=(),
        )
        solver_rows.append(movable_solver)
        movable_bank = _bank_with((), movable_rows)
        movable_archive = _compile_from_v14(solve_archive, movable_bank)
        movable_receiver = receive_carrier_compose_archive(movable_archive)
        movable_camera = movable_receiver.render_camera_pairs(local_ids)
        lane_rows, lane_solver = _solve_role(
            config=config,
            receiver=movable_receiver,
            base_camera=movable_camera,
            labels=target,
            segnet=segnet,
            torch=torch,
            role="Lane",
            role_id=1,
            application="inner_boundary",
            inherited=movable_rows,
        )
        solver_rows.append(lane_solver)
        bank = _bank_with(movable_rows, lane_rows)
        _publish_immutable(
            root / "stage_checkpoints" / "01_encode_side_solver.json",
            rfc8785_canonicalize(
                {
                    "schema": "ddm_v15_encode_side_solver_checkpoint.v1",
                    "typed_config_sha256": config.typed_config_hash(),
                    "representative_source_pair_ids": list(config.representative_source_pair_ids),
                    "roles": solver_rows,
                    "score_claim": False,
                }
            ),
        )
    else:
        bank = _load_template_source(config)
        solver_rows = [
            {
                "schema": SOLVER_SCHEMA,
                "status": "SHA_BOUND_N64_SOLVER_REPLAY",
                "path": config.template_source_path,
                "sha256": config.template_source_sha256,
            }
        ]
    template_payload = encode_scorer_solved_template_bank(bank)
    template_path = root / "scorer_solved_templates.ddst"
    _publish_immutable(template_path, template_payload)
    final_archive = _compile_from_v14(base_archive, bank)
    archive_path = root / f"ddm_v15_solved_templates_n{config.pair_count}.not_a_candidate.zip.receipt-bytes"
    _publish_immutable(archive_path, final_archive)
    receiver = receive_carrier_compose_archive(final_archive)
    _publish_immutable(
        root / "stage_checkpoints" / "02_receiver_closed_archive.json",
        rfc8785_canonicalize(
            {
                "schema": "ddm_v15_receiver_closed_archive.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "archive": {"path": str(archive_path), "bytes": len(final_archive), "sha256": _sha256(final_archive)},
                "template": {
                    "path": str(template_path),
                    "bytes": len(template_payload),
                    "sha256": _sha256(template_payload),
                },
                "receiver_custody": dict(receiver.custody),
                "score_claim": False,
            }
        ),
    )
    control = _control_row(v14)
    measured_path = root / "stage_checkpoints" / "03_solved_template_measurement.json"
    if not measured_path.exists():
        measured = _derive_full_p_identity_measurement(
            config=config,
            root=root,
            base_archive=base_archive,
            final_archive=final_archive,
            control=control,
        )
        if measured is None:
            if segnet is None or posenet is None:
                segnet, posenet, scorer_custody = _load_models(config)
            measured = _measure_candidate(
                name="v15_solved_templates",
                archive=final_archive,
                receiver=receiver,
                config=config,
                root=root,
                labels=labels,
                poses=poses,
                segnet=segnet,
                posenet=posenet,
            )
        measured["scorer_custody"] = scorer_custody
        _publish_immutable(measured_path, rfc8785_canonicalize(measured))
    measured = json.loads(_read_regular_file_once(measured_path))
    movable = measured["per_stratum"]["Movable"]
    lane = measured["per_stratum"]["Lane"]
    gate_pass = (
        config.pair_count == 600
        and float(movable["d_seg"]) <= config.movable_gate
        and len(final_archive) <= config.archive_box_bytes
    )
    ar1_disposition = "BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY"
    producer_paths = (
        REPO_ROOT / "tools/measure_ddm_v15_scorer_solved_templates.py",
        REPO_ROOT / "src/tac/optimization/direct_description_carrier_compose.py",
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "lane_id": "ddm_v15_grammar_parametrized_scorer_solve",
        "tasks": [603, 613, 578],
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "producer_custody": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(_read_regular_file_once(path)),
            }
            for path in producer_paths
        ],
        "inherited_control": control,
        "solved_template_ladder": [measured],
        "selected_candidate": measured["candidate"],
        "template_bank": {
            "path": str(template_path),
            "bytes": len(template_payload),
            "sha256": _sha256(template_payload),
            "record_count": len(bank.templates),
            "rows": [
                {
                    "role": row.role,
                    "application": row.application,
                    "scorer_row_band": [row.scorer_row_start, row.scorer_row_stop],
                    "patch_hw": [row.patch_height, row.patch_width],
                    "rgb_u8": list(row.rgb_u8),
                }
                for row in bank.templates
            ],
            "counted_once_shared_by_grammar": True,
            "scorer_present_at_decode": False,
            "ground_truth_table_present_at_decode": False,
        },
        "solver": solver_rows,
        "conditionals": {"Movable": movable, "Lane": lane},
        "fork": {
            "condition": "full-n600 Movable conditional d_seg <=0.05 at <=160000 exact archive bytes",
            "passed": gate_pass,
            "disposition": (
                "FLAG_MAIN_FOR_R6_EXACT_EVAL_NO_MODAL_DISPATCH"
                if gate_pass
                else "FORMULATION_SCOPED_TEMPLATE_SOLVE_GATE_NOT_MET_JOINT_TRAINING_366_REMAINS_OPEN"
            ),
        },
        "lane_ar1_successor": {
            "measured": False,
            "disposition": ar1_disposition,
            "reason": "no independently observed physical BEV homography or decoder-free metric-pose custody",
        },
        "blocker_delta": (
            "V14 fixed RGB prototypes are replaced by counted row-band templates solved encode-side through exact R; "
            "any remaining projection debt is measured after uint8 projection and zero-collateral secant admission."
        ),
        "named_successor": "#366 joint predictor/template training under the same receiver and custody constraints",
        "fail_closed_mutation_proof": prove_carrier_archive_fail_closed(final_archive),
        "target_custody": {
            "path": str(cache),
            "bytes": config.target_cache_bytes,
            "sha256": config.target_cache_sha256,
            "mutated": False,
        },
        "storage_preflight": storage,
        "resume": {
            "solver_checkpoint_preserved": True,
            "receiver_checkpoint_preserved": True,
            "per_scorer_batch_checkpoints": True,
            "batch_size": config.scorer_batch_size,
            "all_preserved": True,
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md",
            ".omx/research/SPEC_v8_perclass_decomposition_20260708.md",
            config.v14_receipt_path,
            config.v14_archive_path,
            config.solve_archive_path,
            config.target_cache_path,
            ".omx/research/joint_seg_pose_inverse_solve_20260719_codex.md",
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/subagent_progress.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish_immutable(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "complete": True, "receipt": str(receipt_path)}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMV15ScorerSolvedTemplateConfigV1.model_validate_json(_read_regular_file_once(args.config))
    semantic_argv = [
        "tools/measure_ddm_v15_scorer_solved_templates.py",
        "--config",
        str(args.config),
        "--output-directory",
        str(args.output_directory),
    ]
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
