#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify Build #636 raw identity, decode timing, and advisory scorer row."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402, I001
from tac.canonical_equations.ddm_runtime_export_identity_20260723 import (  # noqa: E402
    export_identity,
    score_row,
)
from tac.optimization.ddm_runtime_exporter import (  # noqa: E402
    EVIDENCE_AXIS,
    REPO_ROOT as EXPORT_REPO_ROOT,
    _compile_seed_state,
    _publish_or_verify,
    _sha256,
    _sha256_file,
    load_config,
)
from tac.optimization import ddm_runtime_receiver as runtime  # noqa: E402
from tac.optimization.ddm_runtime_sensitivity import (  # noqa: E402
    DDMRuntimeDecodedStateV1,
    decode_runtime_state,
    stage_argmax_transition_counts,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    _load_models,
    _measure_candidate,
)
from tools.measure_ddm_v15_scorer_solved_templates import (  # noqa: E402
    DDMV15ScorerSolvedTemplateConfigV1,
)


class VerificationError(ValueError):
    """The runtime proof or scorer custody failed closed."""


@dataclass(frozen=True)
class _E2FrameOwnerReceiver:
    """Adapter that removes Seg-owned paint from frame 0 before scoring."""

    source: Any

    @property
    def custody(self) -> dict:
        return dict(self.source.custody)

    def render_camera_pairs(self, pair_ids: Sequence[int]) -> np.ndarray:
        from tac.through_r.resolution_chain import render_grid_to_camera_uint8

        indexes = tuple(int(value) for value in pair_ids)
        camera = self.source.render_camera_pairs(indexes)
        base = self.source.predictor.baseline.render_pairs(indexes)
        for local in range(len(indexes)):
            camera[local, 0] = render_grid_to_camera_uint8(base[local, 0])
        return np.ascontiguousarray(camera)


def _load_canonical_json(path: Path) -> dict:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VerificationError(f"malformed JSON: {path}") from exc
    canonical = rfc8785_canonicalize(value)
    if payload not in (canonical, canonical + b"\n"):
        raise VerificationError(f"JSON is not canonical: {path}")
    return value


def _find_fresh_runtime_receipt(proof_root: Path) -> tuple[Path, dict]:
    matches = sorted(
        (proof_root / "fresh" / ".ddm_runtime_checkpoints").glob(
            "*/*/inflate_receipt.json"
        )
    )
    if len(matches) != 1:
        raise VerificationError(
            f"expected one fresh runtime receipt, observed {len(matches)}"
        )
    path = matches[0]
    return path, _load_canonical_json(path)


def _chart_frame1_grid(
    state: DDMRuntimeDecodedStateV1,
    *,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Reconstruct the exact packet chart at its native scorer grid."""

    anchors = state.anchors[start:stop, 1].to(torch.int64)
    gradients = state.gradients[start:stop, 1].to(torch.int64)
    residuals = state.residuals[start:stop, 1].to(torch.int64)
    rows = torch.arange(12, dtype=torch.int64).reshape(1, 12, 1, 1)
    columns = torch.arange(16, dtype=torch.int64).reshape(1, 1, 16, 1)
    row_term = runtime._round_div_signed(
        gradients[:, 0].reshape(stop - start, 1, 1, 3)
        * (2 * rows - 11),
        22,
    )
    column_term = runtime._round_div_signed(
        gradients[:, 1].reshape(stop - start, 1, 1, 3)
        * (2 * columns - 15),
        30,
    )
    chart = (
        anchors.reshape(stop - start, 1, 1, 3)
        + row_term
        + column_term
        + residuals
    )
    if bool(torch.any((chart < 0) | (chart > 255))):
        raise VerificationError("stage-attribution chart escaped uint8")
    return (
        chart.to(torch.uint8)
        .repeat_interleave(32, dim=1)
        .repeat_interleave(32, dim=2)
        .contiguous()
    )


def _argmax_from_scorer_grid(segnet: Any, value: torch.Tensor) -> np.ndarray:
    if (
        value.dtype != torch.float32
        or tuple(value.shape[1:]) != (3, runtime.PAIR_H, runtime.PAIR_W)
    ):
        raise VerificationError("stage scorer-grid tensor has wrong geometry")
    with torch.inference_mode():
        cells = segnet(value).argmax(dim=1)
    return np.ascontiguousarray(cells.cpu().numpy().astype(np.uint8))


def _aggregate_stage_rows(batch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage_order = ["paint", "R_resample", "uint8", "scorer_consumption"]
    stream_order = ["base/chart.ddb", "semantic/composed.dds"]
    result: dict[str, Any] = {}
    for stream in stream_order:
        totals = {
            stage: dict.fromkeys(
                (
                    "owner_sites",
                    "argmax_diff_from_previous",
                    "errors_before",
                    "errors_after",
                    "errors_introduced",
                    "errors_corrected",
                    "errors_persisting",
                ),
                0,
            )
            for stage in stage_order
        }
        for batch in batch_rows:
            for stage in stage_order:
                row = batch["streams"][stream]["stages"][stage]
                for key in totals[stage]:
                    totals[stage][key] += int(row[key])
        if any(
            totals[stage]["owner_sites"] != totals["paint"]["owner_sites"]
            for stage in stage_order
        ):
            raise VerificationError("stage owner-site totals changed across transitions")
        if (
            totals["scorer_consumption"]["errors_after"]
            != sum(totals[stage]["errors_introduced"] for stage in stage_order)
            - sum(totals[stage]["errors_corrected"] for stage in stage_order)
        ):
            raise VerificationError("aggregate stage error-flow conservation failed")
        result[stream] = {
            "owner_definition": (
                "semantic_label_code==0 at scorer grid"
                if stream == "base/chart.ddb"
                else "semantic_label_code>0 at scorer grid"
            ),
            "owner_sites": totals["paint"]["owner_sites"],
            "stages": [
                {"stage": stage, **totals[stage]} for stage in stage_order
            ],
        }
    if (
        sum(int(row["owner_sites"]) for row in result.values())
        != 600 * runtime.PAIR_H * runtime.PAIR_W
    ):
        raise VerificationError("stream stage owners do not partition n600 scorer sites")
    return result


def _measure_e2_stage_attribution(
    *,
    archive_path: Path,
    archive_sha256: str,
    target_labels: np.ndarray,
    segnet: Any,
    root: Path,
    input_hashes: dict[str, str],
    batch_size: int,
) -> dict[str, Any]:
    """Emit resumable live-path paint/R/uint8/scorer argmax transitions."""

    with zipfile.ZipFile(archive_path) as archive:
        if tuple(archive.namelist()) != runtime.EXPECTED_MEMBERS:
            raise VerificationError("stage attribution packet members changed")
        members = {
            name: archive.read(name)
            for name in runtime.EXPECTED_MEMBERS
        }
    state = decode_runtime_state(members)
    if state.semantic_frame_policy != "frame1_only_seg_free_frame0":
        raise VerificationError("E2 stage attribution requires frame1-only paint")
    verifier_sha256 = _sha256_file(Path(__file__))[1]
    binding = {
        "archive_sha256": archive_sha256,
        "input_hashes": input_hashes,
        "schema": "ddm_e2_stream_stage_loss_attribution.v1",
        "stage_order": [
            "paint",
            "R_resample",
            "uint8",
            "scorer_consumption",
        ],
        "verifier_sha256": verifier_sha256,
    }
    binding_sha256 = _sha256(rfc8785_canonicalize(binding))
    stage_root = root / "e2_stream_stage_loss" / binding_sha256
    stage_root.mkdir(parents=True, exist_ok=True)
    expected_batches = (600 + batch_size - 1) // batch_size
    batch_rows: list[dict[str, Any]] = []
    for start in range(0, 600, batch_size):
        stop = min(start + batch_size, 600)
        checkpoint = stage_root / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = _load_canonical_json(checkpoint)
            if (
                row.get("binding_sha256") != binding_sha256
                or row.get("pair_range") != [start, stop]
            ):
                raise VerificationError("stage-attribution checkpoint custody changed")
            batch_rows.append(row)
            continue

        labels = state.labels[start:stop]
        target = np.ascontiguousarray(
            np.asarray(target_labels[start:stop], dtype=np.uint8)
        )
        owner_masks = {
            "base/chart.ddb": np.ascontiguousarray(labels.numpy() == 0),
            "semantic/composed.dds": np.ascontiguousarray(labels.numpy() > 0),
        }
        chart = _chart_frame1_grid(state, start=start, stop=stop)
        palette_grid = state.palette[labels.to(torch.int64)]
        paint = torch.where(
            (labels > 0).reshape(stop - start, runtime.PAIR_H, runtime.PAIR_W, 1),
            palette_grid,
            chart,
        )
        paint_cells = _argmax_from_scorer_grid(
            segnet,
            paint.permute(0, 3, 1, 2).contiguous().float(),
        )

        chart_nchw = chart.permute(0, 3, 1, 2).contiguous().float()
        with torch.inference_mode():
            camera_float = F.interpolate(
                chart_nchw,
                size=(runtime.CAMERA_H, runtime.CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
        overlay = (
            labels.index_select(1, state.camera_rows)
            .index_select(2, state.camera_columns)
            .contiguous()
        )
        camera_colours = (
            state.palette[overlay.to(torch.int64)]
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        camera_mask = (overlay > 0).reshape(
            stop - start,
            1,
            runtime.CAMERA_H,
            runtime.CAMERA_W,
        )
        camera_float = torch.where(
            camera_mask,
            camera_colours.float(),
            camera_float,
        )
        with torch.inference_mode():
            resampled = F.interpolate(
                camera_float,
                size=(runtime.PAIR_H, runtime.PAIR_W),
                mode="bilinear",
                align_corners=False,
            )
        resampled_cells = _argmax_from_scorer_grid(segnet, resampled)

        camera_uint8 = torch.where(
            camera_mask,
            camera_colours,
            torch.clamp(torch.round(camera_float), 0.0, 255.0).to(torch.uint8),
        )
        with torch.inference_mode():
            uint8_grid = F.interpolate(
                camera_uint8.float(),
                size=(runtime.PAIR_H, runtime.PAIR_W),
                mode="bilinear",
                align_corners=False,
            )
        uint8_cells = _argmax_from_scorer_grid(segnet, uint8_grid)

        actual_camera = runtime._render_batch(
            start=start,
            stop=stop,
            anchors=state.anchors,
            gradients=state.gradients,
            residuals=state.residuals,
            labels=state.labels,
            palette=state.palette,
            camera_rows=state.camera_rows,
            camera_columns=state.camera_columns,
            semantic_frame_policy=state.semantic_frame_policy,
        )
        actual_btchw = (
            actual_camera.permute(0, 1, 4, 2, 3).contiguous().float()
        )
        with torch.inference_mode():
            official_input = segnet.preprocess_input(actual_btchw)
            scorer_cells = (
                segnet(official_input)
                .argmax(dim=1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        if not torch.equal(official_input, uint8_grid):
            raise VerificationError(
                "manual uint8 R-down differs from official scorer consumption"
            )
        scorer_cells = np.ascontiguousarray(scorer_cells)
        streams: dict[str, Any] = {}
        for stream, owner_mask in owner_masks.items():
            stages = {
                "paint": stage_argmax_transition_counts(
                    before=target,
                    after=paint_cells,
                    target=target,
                    owner_mask=owner_mask,
                ),
                "R_resample": stage_argmax_transition_counts(
                    before=paint_cells,
                    after=resampled_cells,
                    target=target,
                    owner_mask=owner_mask,
                ),
                "uint8": stage_argmax_transition_counts(
                    before=resampled_cells,
                    after=uint8_cells,
                    target=target,
                    owner_mask=owner_mask,
                ),
                "scorer_consumption": stage_argmax_transition_counts(
                    before=uint8_cells,
                    after=scorer_cells,
                    target=target,
                    owner_mask=owner_mask,
                ),
            }
            streams[stream] = {"stages": stages}
        row = {
            "binding_sha256": binding_sha256,
            "first_rung": True,
            "pair_range": [start, stop],
            "research_only": True,
            "schema": "ddm_e2_stream_stage_loss_batch.v1",
            "score_claim": False,
            "scorer_cells_sha256": hashlib.sha256(
                scorer_cells.tobytes(order="C")
            ).hexdigest(),
            "streams": streams,
            "uint8_to_official_input_exact": True,
        }
        _publish_or_verify(
            checkpoint,
            rfc8785_canonicalize(row) + b"\n",
        )
        batch_rows.append(row)

    if len(batch_rows) != expected_batches:
        raise VerificationError("stage-attribution batch coverage is incomplete")
    aggregate = _aggregate_stage_rows(batch_rows)
    final_errors = sum(
        int(row["stages"][-1]["errors_after"])
        for row in aggregate.values()
    )
    final_sites = sum(int(row["owner_sites"]) for row in aggregate.values())
    return {
        "all_batches_checkpointed_and_preserved": True,
        "batch_count": len(batch_rows),
        "batch_digest_chain_sha256": hashlib.sha256(
            "".join(
                hashlib.sha256(
                    rfc8785_canonicalize(row)
                ).hexdigest()
                for row in batch_rows
            ).encode()
        ).hexdigest(),
        "batch_size": batch_size,
        "binding": binding,
        "binding_sha256": binding_sha256,
        "error_flow_conservation": (
            "errors_after=errors_before+introduced-corrected at every "
            "stream x stage transition"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "final_d_seg": format(final_errors / final_sites, ".12f"),
        "first_rung": True,
        "owner_partition": (
            "exact scorer-grid packet semantic code: chart==0, semantic>0"
        ),
        "research_only": True,
        "schema": "ddm_e2_stream_stage_loss_attribution.v1",
        "score_claim": False,
        "source_class": "(ii) live export realization-stage loss",
        "stage_checkpoint_root": str(stage_root),
        "stage_semantics": {
            "paint": "perfect target argmax -> native-grid painted description",
            "R_resample": "native-grid paint -> float bicubic-up/bilinear-down",
            "uint8": "float R -> camera round/clamp uint8 R",
            "scorer_consumption": (
                "manual uint8 R-down -> official SegNet.preprocess_input + argmax"
            ),
        },
        "streams": aggregate,
        "verdict_scope": (
            "E2 n600 frame1 SegNet argmax sites partitioned by packet fact owner; "
            "counts are live-path error attribution, not causal Shapley values "
            "and not PoseNet or contest-CPU/CUDA authority."
        ),
    }


def verify(
    *,
    export_config_path: Path,
    scorer_config_path: Path,
) -> tuple[dict, Path]:
    export_config = load_config(export_config_path)
    is_e2 = (
        export_config.run_id
        == "ddm_e2_pose_stream_and_doctrine_export_20260723"
    )
    scorer_value = json.loads(scorer_config_path.read_bytes())
    scorer_config = DDMV15ScorerSolvedTemplateConfigV1.model_validate(scorer_value)
    if scorer_config.pair_start != 0 or scorer_config.pair_count != 600:
        raise VerificationError("scorer config must bind the full n600 window")
    if EXPORT_REPO_ROOT != REPO_ROOT:
        raise VerificationError("exporter repository root drifted")

    output_root = (
        REPO_ROOT / export_config.output_directory
    ).resolve()
    export_receipt_path = (
        output_root.parent
        / (
            "ddm_e2_runtime_export_receipt.json"
            if is_e2
            else "ddm_e1_runtime_export_receipt.json"
        )
    )
    export_receipt = _load_canonical_json(export_receipt_path)
    upstream_receipt_path = (
        output_root.parent
        / (
            "ddm_e2_upstream_harness_receipt.json"
            if is_e2
            else "ddm_e1_upstream_harness_receipt.json"
        )
    )
    upstream_receipt = _load_canonical_json(upstream_receipt_path)
    archive_path = output_root / "archive.zip"
    archive_bytes, archive_sha256 = _sha256_file(archive_path)
    if (archive_bytes, archive_sha256) != (
        export_receipt["archive"]["bytes"],
        export_receipt["archive"]["sha256"],
    ):
        raise VerificationError("exported archive changed after compilation")
    if (
        upstream_receipt["status"] != "PASS"
        or upstream_receipt["score_claim"] is not False
        or upstream_receipt["failure_reasons"] != []
        or upstream_receipt["under_1800_seconds"] is not True
        or upstream_receipt["packet"]["archive.zip"]
        != {"bytes": archive_bytes, "sha256": archive_sha256}
    ):
        raise VerificationError("upstream harness receipt does not bind this archive")
    runtime_path = output_root / "inflate.py"
    runtime_bytes, runtime_sha256 = _sha256_file(runtime_path)
    if (runtime_bytes, runtime_sha256) != (
        export_receipt["runtime"]["inflate_py"]["bytes"],
        export_receipt["runtime"]["inflate_py"]["sha256"],
    ):
        raise VerificationError("runtime source changed after compilation")

    proof_root = Path(export_config.proof_root)
    runtime_receipt_path, runtime_receipt = _find_fresh_runtime_receipt(proof_root)
    final_path = Path(runtime_receipt["final"]["path"])
    final_identity = _sha256_file(final_path)
    if final_identity != (
        runtime_receipt["final"]["bytes"],
        runtime_receipt["final"]["sha256"],
    ):
        raise VerificationError("fresh final raw changed after inflate")
    identity = export_identity(
        pair_count=600,
        source_bytes=export_receipt["output_identity"]["bytes"],
        source_sha256=export_receipt["output_identity"]["sha256"],
        packaged_bytes=final_identity[0],
        packaged_sha256=final_identity[1],
    )
    if not identity.byte_identical:
        raise VerificationError("packaged raw is not byte-identical to the source receiver")
    if upstream_receipt["raw"] != {
        "bytes": final_identity[0],
        "sha256": final_identity[1],
    }:
        raise VerificationError("upstream harness raw differs from clean receiver raw")
    if float(runtime_receipt["total_seconds"]) >= 1800.0:
        raise VerificationError("fresh single-thread inflate exceeded 30 minutes")

    source_path = (REPO_ROOT / export_config.source_archive_path).resolve()
    source = source_path.read_bytes()
    if (len(source), _sha256(source)) != (
        export_config.source_archive_bytes,
        export_config.source_archive_sha256,
    ):
        raise VerificationError("sealed source archive changed")
    state_archive, state_receiver, dofs = _compile_seed_state(source)

    cache_path = Path(scorer_config.target_cache_path)
    cache_identity = _sha256_file(cache_path)
    if cache_identity != (
        scorer_config.target_cache_bytes,
        scorer_config.target_cache_sha256,
    ):
        raise VerificationError("target scorer cache custody mismatch")
    labels = open_stored_npy_memmap(cache_path, "lstars")
    poses = open_stored_npy_memmap(cache_path, "gt_poses")
    segnet, posenet, model_custody = _load_models(scorer_config)
    measurement_root = proof_root / "scorer_measurement"
    measurement_root.mkdir(parents=True, exist_ok=True)
    measured = _measure_candidate(
        name=(
            "e2_frame1_only_seeded_source_state"
            if is_e2
            else "seeded_source_state"
        ),
        archive=state_archive,
        receiver=(
            _E2FrameOwnerReceiver(state_receiver)
            if is_e2
            else state_receiver
        ),
        config=scorer_config,
        root=measurement_root,
        labels=labels,
        poses=poses,
        segnet=segnet,
        posenet=posenet,
    )
    stream_stage_loss_attribution = None
    if is_e2:
        stream_stage_loss_attribution = _measure_e2_stage_attribution(
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            target_labels=labels,
            segnet=segnet,
            root=measurement_root / "stage_attribution",
            input_hashes={
                "scorer_cache_sha256": cache_identity[1],
                "scorer_modules_sha256": model_custody["modules_sha256"],
                "segnet_weights_sha256": model_custody[
                    "segnet_weights_sha256"
                ],
            },
            batch_size=scorer_config.scorer_batch_size,
        )
        if stream_stage_loss_attribution["final_d_seg"] != measured["d_seg"]:
            raise VerificationError(
                "stream stage-attribution final d_seg differs from canonical meter"
            )
    terms = score_row(
        archive_bytes=archive_bytes,
        d_seg=float(measured["d_seg"]),
        d_pose=float(measured["d_pose"]),
    )
    payload_bytes = {
        row["member"]: int(row["member_payload_range"]["bytes"])
        for row in export_receipt["archive"]["member_homes"]
        if row["member"] is not None
    }
    block_bytes = [
        {
            "block": "L",
            "bytes": payload_bytes["base/chart.ddb"]
            + payload_bytes["semantic/composed.dds"],
            "contents": "composed semantic field plus base chart",
        },
        {
            "block": "D2",
            "bytes": 0,
            "contents": "per-element tolerance dual section inactive",
        },
        {
            "block": "D1",
            "bytes": 0,
            "contents": "amplitude field section inactive",
        },
        {
            "block": "D4",
            "bytes": archive_bytes - sum(payload_bytes.values()),
            "contents": "ZIP framing and container custody",
        },
        {
            "block": "D6",
            "bytes": payload_bytes["manifest.json"],
            "contents": "realization and archive metadata",
        },
        {
            "block": "D5",
            "bytes": 0,
            "contents": "texture-quotient residual-stat section inactive",
        },
    ]
    if sum(int(row["bytes"]) for row in block_bytes) != archive_bytes:
        raise VerificationError("joint-cycle block bytes do not close to archive bytes")
    total_camera_pixels = 600 * 2 * 874 * 1164
    described_camera_paint_fraction = (
        int(export_receipt["paint_jacobian"]["painted_camera_pixels_all_pairs_all_frames"])
        / total_camera_pixels
    )
    result = {
        "archive": {
            "bytes": archive_bytes,
            "member_homes": export_receipt["archive"]["member_homes"],
            "receiver_byte_home_bijection": export_receipt["archive"][
                "receiver_byte_home_bijection"
            ],
            "sha256": archive_sha256,
        },
        "cleanup": {
            "bulk_artifacts_preserved_on_ssd": True,
            "certify_or_block": "no proof, source, cache, or checkpoint bytes deleted",
            "proof_root": str(proof_root),
        },
        "dofs": dofs,
        "evidence_axis": EVIDENCE_AXIS,
        "identity": {
            "byte_identical": identity.byte_identical,
            "bytes": final_identity[0],
            "packaged_sha256": identity.packaged_sha256,
            "source_sha256": identity.source_sha256,
        },
        "joint_iteration_curve": {
            "block_order": ["L", "D2", "D1", "D4", "D6", "D5"],
            "cycles": [
                {
                    "archive_bytes": archive_bytes,
                    "apparatus_validity": {
                        "coder_refit_applied": True,
                        "input_hashes_verified_at_consumption": True,
                        "realized_residual_fresh": True,
                    },
                    "best_so_far": True,
                    "bytes_by_block": block_bytes,
                    "cycle_index": 0,
                    "described_camera_paint_fraction": format(
                        described_camera_paint_fraction, ".12f"
                    ),
                    "d_pose": measured["d_pose"],
                    "d_seg": measured["d_seg"],
                    "realized_argmax_agreement_proxy": format(
                        1.0 - float(measured["d_seg"]), ".12f"
                    ),
                    "score_total": terms["total"],
                    "staleness_input_hashes": {
                        "archive_sha256": archive_sha256,
                        "raw_sha256": final_identity[1],
                        "scorer_cache_sha256": cache_identity[1],
                        "state_archive_sha256": _sha256(state_archive),
                    },
                }
            ],
            "fixed_budget_bytes": archive_bytes,
            "fixed_point_status": "OPEN_ONE_MEASURED_EXPORT_CYCLE_ONLY",
            "global_reinvestment": True,
            "successor_cycle_measurement_policy": {
                "pose": "skip_only_while_pose_state_hash_is_unchanged",
                "seg": "consume_argmax_bit_identical_exact_forward_checkpoint_when_fresh",
                "stale_cache": "rederive_or_mark_stale_advisory",
            },
            "schema": "ddm_joint_fixed_budget_iteration_curve.v1",
            "stop_law": "full_joint_cycle_no_net_gain_at_constant_bytes",
            "successor_cycles_required": True,
        },
        "main_landing_review_required": True,
        "paint_jacobian": export_receipt["paint_jacobian"],
        "pointer_moved": False,
        "rederive_argv": [
            "/Users/adpena/Projects/pact/.venv/bin/python",
            "tools/verify_ddm_runtime_export.py",
            "--export-config",
            str(export_config_path.relative_to(REPO_ROOT)),
            "--scorer-config",
            str(scorer_config_path.relative_to(REPO_ROOT)),
        ],
        "research_only": True,
        "runtime": {
            "dependencies": ["torch", "brotli"],
            "fresh_receipt_path": str(runtime_receipt_path),
            "member_consumption": runtime_receipt["member_consumption"],
            "render_seconds": runtime_receipt["render_seconds"],
            "runtime_bytes": runtime_bytes,
            "runtime_sha256": runtime_sha256,
            "single_thread_cpu": runtime_receipt["single_thread_cpu"],
            "stage_count": runtime_receipt["resume"]["stage_count"],
            "total_seconds": runtime_receipt["total_seconds"],
            "under_1800_seconds": True,
        },
        "schema": (
            "ddm_e2_runtime_verification_receipt.v2"
            if is_e2
            else "ddm_e1_runtime_verification_receipt.v1"
        ),
        "score": {
            "archive_bytes": archive_bytes,
            "d_pose": measured["d_pose"],
            "d_seg": measured["d_seg"],
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            "terms": terms,
        },
        "score_claim": False,
        "scorer": {
            "batch_count": measured["batch_count"],
            "batch_size": measured["batch_size"],
            "batch_digest_chain_sha256": measured[
                "batch_digest_chain_sha256"
            ],
            "custody": model_custody,
            "measurement_root": str(measurement_root),
            "target_cache": {
                "bytes": cache_identity[0],
                "path": str(cache_path),
                "sha256": cache_identity[1],
            },
        },
        "state": {
            "archive_bytes": len(state_archive),
            "archive_sha256": _sha256(state_archive),
            "name": export_config.state_name,
        },
        "upstream_harness": upstream_receipt,
        "verdict": (
            "PASS_E2_RUNTIME_EXPORT_POSE_TUBE_BLOCKED_ADVISORY_ONLY"
            if is_e2
            else "PASS_EXACT_N600_RUNTIME_EXPORT_ADVISORY_ONLY"
        ),
        "verdict_scope": (
            "Exact n600 source-versus-packaged camera raw identity and local "
            "single-thread decode timing; frozen-scorer row is macOS-CPU advisory, "
            "not contest-CPU/CUDA authority or promotion evidence."
        ),
    }
    if is_e2:
        result.update(
            {
                "pose_contract": export_receipt["pose_contract"],
                "rate_doctrine": export_receipt["rate_doctrine"],
                "stream_stage_loss_attribution": (
                    stream_stage_loss_attribution
                ),
                "rate_decomposition": {
                    "archive_counted_bytes": archive_bytes,
                    "container_and_manifest_bytes": (
                        archive_bytes
                        - payload_bytes["base/chart.ddb"]
                        - payload_bytes["semantic/composed.dds"]
                    ),
                    "runtime_inflate_py_bytes_not_counted": runtime_bytes,
                    "state_archive_bytes_not_a_packet_member": len(state_archive),
                    "stream_payload_bytes": {
                        "base/chart.ddb": payload_bytes["base/chart.ddb"],
                        "semantic/composed.dds": payload_bytes[
                            "semantic/composed.dds"
                        ],
                    },
                    "false_177KB_remainder_explanation": (
                        "339094-134211-28108 mixes one counted archive with "
                        "two nonmember quantities and is not a byte partition"
                    ),
                },
            }
        )
    receipt_path = _publish_or_verify(
        output_root.parent
        / (
            "ddm_e2_runtime_verification_receipt_v2.json"
            if is_e2
            else "ddm_e1_runtime_verification_receipt.json"
        ),
        rfc8785_canonicalize(result) + b"\n",
    )
    return result, receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-config", required=True)
    parser.add_argument("--scorer-config", required=True)
    args = parser.parse_args(argv)
    export_config_path = Path(args.export_config).resolve()
    scorer_config_path = Path(args.scorer_config).resolve()
    for path in (export_config_path, scorer_config_path):
        try:
            path.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise VerificationError("config paths must be inside the repository") from exc
    result, receipt_path = verify(
        export_config_path=export_config_path,
        scorer_config_path=scorer_config_path,
    )
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path),
                "score": result["score"],
                "verdict": result["verdict"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
