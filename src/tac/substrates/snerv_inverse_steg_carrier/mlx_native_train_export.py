# SPDX-License-Identifier: MIT
"""Native MLX target-hydration to SNeRV SNAR1 archive export.

This is the first real adapter surface behind
``mlx_native_adapter_contract.py``.  It is deliberately honest about its
scope: MLX owns real-video target hydration and bridge custody; the current
decoder fit is the existing deterministic NumPy closed-form SNeRV fit, because
that is the receiver-portable training primitive already used by the advisory
lane.  Longer scorer-in-loop MLX optimization remains blocked until the runner
attaches real SegNet/PoseNet teachers and byte pressure inside the training
loop.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.snerv_step_map_coder import encode_step_maps_waterfill
from tac.contest_eval_contract import build_upstream_eval_contract
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import sha256_file, write_bytes_artifact, write_json
from tac.substrates._shared.mlx_score_aware.bridge_drift import (
    build_mlx_numpy_bridge_drift_bundle,
    mlx_numpy_bridge_drift_report,
)
from tac.substrates._shared.mlx_score_aware.targets import decode_mlx_targets
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SnervArchivePacket,
    decode_snerv_archive_frames,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.archive_candidate import (
    CAMERA_HW,
    SNERV_RECEIVER_PROOF_SCHEMA,
    expected_receiver_output_bytes_from_metadata,
    export_snerv_archive_bound_candidate_package,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    SnervModelSizeConfig,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)

SNERV_MLX_NATIVE_TRAIN_EXPORT_SCHEMA = "snerv_mlx_native_train_export.v1"
SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA = "snerv_mlx_native_prefilter_profile.v1"
SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA = "snerv_mlx_native_storage_preflight.v1"
SNERV_MLX_NATIVE_PACKET_FILENAME = "snerv_mlx_native_packet.snar"
SNERV_MLX_NATIVE_REPORT_FILENAME = "snerv_mlx_native_train_export.json"
SCORER_HW = (384, 512)

FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


class SnervMlxNativeExportError(ValueError):
    """Raised when the native MLX SNeRV adapter cannot build a valid export."""


@dataclass(frozen=True)
class SnervMlxNativeArtifact:
    """File-backed output of the native MLX SNeRV adapter."""

    schema: str
    output_dir: str
    packet_path: str
    packet_bytes: int
    packet_sha256: str
    num_pairs: int
    levels: int
    wavelet: str
    target_bits_per_coeff: float
    step_map_bits_per_coeff: float
    step_map_packet_schema: str
    step_map_coder_mode: str
    step_map_coder_groups: tuple[dict[str, Any], ...]
    decoder_payload_codec: str
    lf_payload_codec: str
    model_size: dict[str, Any]
    bridge_drift: dict[str, Any]
    scorer_custody: dict[str, Any]
    scorer_loop_qat: dict[str, Any]
    storage_preflight: dict[str, Any]
    archive_package: dict[str, Any] | None
    archive_path: str | None
    archive_bytes: int | None
    archive_sha256: str | None
    receiver_proof_path: str | None
    receiver_proof_passed: bool
    receiver_contract_satisfied: bool
    report_path: str | None
    blockers: tuple[str, ...]
    score_claim: bool = False
    frontier_score_claim: bool = False
    rank_or_kill_eligible: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def train_export_snerv_mlx_native(
    output_dir: str | Path,
    num_pairs: int,
    source_video_path: str | Path,
    modelsize_candidate: Mapping[str, Any] | None,
    scorer_upstream_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    output_height: int = SCORER_HW[0],
    output_width: int = SCORER_HW[1],
    run_archive_export: bool = True,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
    run_scorer_loop_qat: bool = False,
    scorer_loop_qat_max_trials: int = 0,
    scorer_loop_qat_search_mode: str = "random_signed",
    scorer_loop_qat_qat_bits: int = 8,
    scorer_loop_qat_decoder_payload_codec: str | None = None,
    scorer_loop_qat_component_guard_mode: str = "score_primary",
    scorer_loop_qat_device: str = "cpu",
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Hydrate real targets on MLX, export a NumPy-portable SNAR1 archive.

    The function name is intentionally the contract surface.  Its payload keeps
    the current training scope precise: ``mlx_target_hydration`` plus the
    existing deterministic closed-form SNeRV decoder fit.  It does not claim
    full score-aware long training.
    """

    started = time.monotonic()
    root = _repo_root(repo_root)
    scorer_custody = build_upstream_eval_contract(
        repo_root=root,
        upstream_dir=Path(scorer_upstream_dir).expanduser(),
    )
    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    candidate = dict(modelsize_candidate or {})
    levels = int(candidate.get("levels", candidate.get("snerv_levels", 3)))
    wavelet = str(candidate.get("wavelet", "haar"))
    target_bits_per_coeff = float(candidate.get("bits_per_coeff", 2.5))
    step_map_bits_per_coeff = float(
        candidate.get(
            "step_map_bits_per_coeff",
            candidate.get("snerv_step_map_bits_per_coeff", 4.0),
        )
    )
    decoder_payload_codec = str(
        candidate.get("decoder_payload_codec", "mixed_magnitude_symmetric")
    )
    active_decoder_payload_codec = (
        str(scorer_loop_qat_decoder_payload_codec)
        if scorer_loop_qat_decoder_payload_codec
        else decoder_payload_codec
    )
    lf_payload_codec = str(candidate.get("lf_payload_codec", "auto"))
    model_size = _model_size_from_candidate(candidate)

    target0_mlx, target1_mlx = decode_mlx_targets(
        source_video_path,
        num_pairs=int(num_pairs),
        output_height=int(output_height),
        output_width=int(output_width),
    )
    target0_np = np.asarray(target0_mlx, dtype=np.float32)
    target1_np = np.asarray(target1_mlx, dtype=np.float32)
    bridge = build_mlx_numpy_bridge_drift_bundle(
        (
            mlx_numpy_bridge_drift_report(
                label="target_frame0_nhwc01",
                mlx_array=target0_mlx,
                numpy_array=target0_np,
                atol=0.0,
                rtol=0.0,
            ),
            mlx_numpy_bridge_drift_report(
                label="target_frame1_nhwc01",
                mlx_array=target1_mlx,
                numpy_array=target1_np,
                atol=0.0,
                rtol=0.0,
            ),
        ),
        bundle_id="snerv_mlx_native_target_bridge",
    )
    if bridge["allclose"] is not True:
        raise SnervMlxNativeExportError(
            "MLX target bridge drift failed: " + ",".join(bridge["blockers"])
        )

    pairs_nchw255 = _target_pairs_to_nchw255(target0_np, target1_np)
    closed_form_archive = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs_nchw255,
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        step_map_bits_per_coeff=step_map_bits_per_coeff,
        decoder_payload_codec=active_decoder_payload_codec,
        lf_payload_codec=lf_payload_codec,
        model_size=model_size,
        metadata_extra={
            "source_video_path": Path(source_video_path).as_posix(),
            "training_backend": "mlx_target_hydration_numpy_closed_form_decoder_fit",
            "human_visual_fidelity_objective": False,
            "contest_scorer_distortion_objective": True,
            "score_aware_long_training_executed": False,
        },
    )
    scorer_loop_qat = _run_scorer_loop_qat_attachment(
        requested=bool(run_scorer_loop_qat),
        output_dir=out / "snerv_scorer_loop_qat",
        num_pairs=int(num_pairs),
        source_video_path=source_video_path,
        scorer_upstream_dir=scorer_upstream_dir,
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        model_size=model_size,
        max_trials=int(scorer_loop_qat_max_trials),
        search_mode=str(scorer_loop_qat_search_mode),
        qat_bits=int(scorer_loop_qat_qat_bits),
        decoder_payload_codec=active_decoder_payload_codec,
        component_guard_mode=str(scorer_loop_qat_component_guard_mode),
        device=str(scorer_loop_qat_device),
        allow_overwrite=allow_overwrite,
    )
    selected_packet = bytes(closed_form_archive.packet)
    selected_packet_source = "mlx_target_hydration_numpy_closed_form_decoder_fit"
    scorer_loop_qat_public = {
        key: value for key, value in scorer_loop_qat.items() if key != "_best_packet_bytes"
    }
    best_packet = scorer_loop_qat.get("_best_packet_bytes")
    if (
        isinstance(best_packet, bytes)
        and best_packet
        and scorer_loop_qat.get("accepted_improvement") is True
        and scorer_loop_qat.get("receiver_contract_satisfied") is True
    ):
        selected_packet = bytes(best_packet)
        selected_packet_source = "scorer_loop_qat_best_receiver_packet"
        scorer_loop_qat_public["emitted_packet_uses_scorer_loop_best_decoder"] = True
        scorer_loop_qat_public["emitted_packet_bytes"] = len(selected_packet)
        scorer_loop_qat_public["emitted_packet_sha256"] = _sha256_bytes(
            selected_packet
        )
        scorer_loop_qat_public["blockers"] = _ordered_unique(
            str(blocker)
            for blocker in scorer_loop_qat_public.get("blockers") or []
            if str(blocker)
            != "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export"
        )
        report = scorer_loop_qat_public.get("report_path")
        if report:
            _payload_for_disk = {
                key: value
                for key, value in scorer_loop_qat_public.items()
                if key != "report_path"
            }
            write_json(report, _payload_for_disk)

    packet_path = out / SNERV_MLX_NATIVE_PACKET_FILENAME
    write_bytes_artifact(
        packet_path,
        selected_packet,
        allow_overwrite=allow_overwrite,
        expected_existing_sha256=(
            sha256_file(packet_path)
            if allow_overwrite and packet_path.is_file()
            else None
        ),
    )

    storage_preflight = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=int(num_pairs),
        packet_bytes=len(selected_packet),
    )
    package: dict[str, Any] | None = None
    blockers = [
        "snerv_mlx_score_aware_long_training_not_executed",
        "contest_cpu_cuda_exact_eval_not_executed",
    ]
    if scorer_custody.get("contract_valid") is not True:
        blockers.append("snerv_mlx_native_scorer_custody_contract_invalid")
    blockers.extend(
        _scorer_loop_qat_blockers(
            scorer_loop_qat_public,
            num_pairs=int(num_pairs),
        )
    )
    if run_archive_export:
        if storage_preflight["preflight_passed"] is not True:
            blockers.append("snerv_mlx_native_receiver_proof_storage_preflight_failed")
        else:
            package = export_snerv_mlx_archive(
                model_or_artifact={
                    "packet_path": packet_path.as_posix(),
                    "packet_sha256": _sha256_bytes(selected_packet),
                },
                output_dir=out / "snerv_mlx_native_archive_bound_package",
                repo_root=root,
                retain_receiver_output=retain_receiver_output,
                receiver_proof_timeout_seconds=receiver_proof_timeout_seconds,
            )
    receiver_proof = dict(package.get("receiver_proof") or {}) if package else {}
    selected_archive_metadata = unpack_snerv_archive(selected_packet).metadata

    artifact = SnervMlxNativeArtifact(
        schema=SNERV_MLX_NATIVE_TRAIN_EXPORT_SCHEMA,
        output_dir=out.as_posix(),
        packet_path=packet_path.as_posix(),
        packet_bytes=len(selected_packet),
        packet_sha256=_sha256_bytes(selected_packet),
        num_pairs=int(num_pairs),
        levels=levels,
        wavelet=wavelet,
        target_bits_per_coeff=target_bits_per_coeff,
        step_map_bits_per_coeff=step_map_bits_per_coeff,
        step_map_packet_schema=str(
            selected_archive_metadata.get("step_map_packet_schema") or ""
        ),
        step_map_coder_mode=str(
            selected_archive_metadata.get("step_map_coder_mode") or ""
        ),
        step_map_coder_groups=tuple(
            dict(group)
            for group in selected_archive_metadata.get("step_map_coder_groups") or ()
        ),
        decoder_payload_codec=active_decoder_payload_codec,
        lf_payload_codec=lf_payload_codec,
        model_size=model_size.as_jsonable(),
        bridge_drift=bridge,
        scorer_custody=scorer_custody,
        scorer_loop_qat=scorer_loop_qat_public,
        storage_preflight=storage_preflight,
        archive_package=package,
        archive_path=(
            str(receiver_proof.get("archive_path"))
            if receiver_proof.get("archive_path")
            else None
        ),
        archive_bytes=(
            int(receiver_proof["archive_bytes"])
            if receiver_proof.get("archive_bytes") is not None
            else None
        ),
        archive_sha256=(
            str(receiver_proof.get("archive_sha256"))
            if receiver_proof.get("archive_sha256")
            else None
        ),
        receiver_proof_path=(
            str(receiver_proof.get("proof_path"))
            if receiver_proof.get("proof_path")
            else None
        ),
        receiver_proof_passed=(
            receiver_proof.get("runtime_consumption_proof_passed") is True
        ),
        receiver_contract_satisfied=(
            receiver_proof.get("receiver_contract_satisfied") is True
        ),
        report_path=None,
        blockers=tuple(dict.fromkeys(blockers)),
    )
    payload = artifact.as_jsonable()
    payload["packet_source"] = selected_packet_source
    payload["wall_seconds"] = round(time.monotonic() - started, 6)
    report_path = out / SNERV_MLX_NATIVE_REPORT_FILENAME
    if report_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(
            f"refusing to overwrite existing artifact: {report_path}"
        )
    payload["report_path"] = report_path.as_posix()
    write_json(report_path, payload)
    return payload


def export_snerv_mlx_archive(
    model_or_artifact: Any,
    output_dir: str | Path,
    repo_root: str | Path,
    *,
    retain_receiver_output: bool = False,
    receiver_proof_timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Export SNAR1 packet bytes through the canonical archive-bound package."""

    root = _repo_root(repo_root)
    out = Path(output_dir).expanduser().resolve(strict=False)
    packet = _packet_bytes_from_artifact(model_or_artifact)
    decoded = unpack_snerv_archive(packet)
    storage = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=int(decoded.metadata.get("n_pairs", 0)),
        packet_bytes=len(packet),
    )
    if storage["preflight_passed"] is not True:
        raise SnervMlxNativeExportError(
            "receiver proof storage preflight failed: "
            f"free={storage['free_bytes']} required={storage['required_bytes']}"
        )
    package = export_snerv_archive_bound_candidate_package(
        packet=packet,
        output_dir=out,
        repo_root=root,
        retain_receiver_output=retain_receiver_output,
        receiver_proof_timeout_seconds=receiver_proof_timeout_seconds,
    )
    package["snerv_mlx_native_storage_preflight"] = storage
    return package


def write_snerv_mlx_receiver_proof(
    archive_zip_path: str | Path,
    runtime_submission_dir: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_output: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run the generated SNeRV receiver proof for an existing archive package."""

    root = _repo_root(repo_root)
    archive_zip = Path(archive_zip_path).expanduser().resolve(strict=False)
    submission_dir = Path(runtime_submission_dir).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    packet_path = submission_dir / "0.bin"
    if not packet_path.is_file():
        raise SnervMlxNativeExportError(
            f"runtime submission dir is missing 0.bin: {submission_dir}"
        )
    decoded = unpack_snerv_archive(packet_path.read_bytes())
    expected_bytes = expected_receiver_output_bytes_from_metadata(decoded.metadata)
    storage = build_snerv_mlx_native_storage_preflight(
        output_dir=out,
        n_pairs=int(decoded.metadata.get("n_pairs", 0)),
        packet_bytes=packet_path.stat().st_size,
    )
    if storage["preflight_passed"] is not True:
        raise SnervMlxNativeExportError(
            "receiver proof storage preflight failed: "
            f"free={storage['free_bytes']} required={storage['required_bytes']}"
        )
    proof = run_generated_inflate_receiver_proof(
        archive_zip_path=archive_zip,
        archive_sha256=sha256_file(archive_zip),
        archive_bytes=archive_zip.stat().st_size,
        submission_dir=submission_dir,
        archive_dir_for_inflate=submission_dir,
        output_dir=out,
        repo_root=root,
        proof_schema=SNERV_RECEIVER_PROOF_SCHEMA,
        proof_filename="snerv_mlx_native_receiver_proof.json",
        candidate_label="snerv_mlx_native",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=expected_bytes,
        retain_receiver_output=retain_receiver_output,
        timeout_seconds=int(timeout_seconds),
    )
    proof["snerv_mlx_native_storage_preflight"] = storage
    return proof


def _run_scorer_loop_qat_attachment(
    *,
    requested: bool,
    output_dir: str | Path,
    num_pairs: int,
    source_video_path: str | Path,
    scorer_upstream_dir: str | Path,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    model_size: SnervModelSizeConfig,
    max_trials: int,
    search_mode: str,
    qat_bits: int,
    decoder_payload_codec: str,
    component_guard_mode: str,
    device: str,
    allow_overwrite: bool,
) -> dict[str, Any]:
    """Attach real SegNet/PoseNet scorer-loop evidence to a native export."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "snerv_scorer_loop_qat_attachment.json"
    if report_path.exists() and not allow_overwrite:
        raise SnervMlxNativeExportError(
            f"refusing to overwrite existing scorer-loop artifact: {report_path}"
        )
    if not requested:
        payload = {
            "schema": "snerv_mlx_native_scorer_loop_qat_attachment.v1",
            "requested": False,
            "executed": False,
            "receiver_contract_satisfied": False,
            "accepted_improvement": False,
            "full_video_coverage": False,
            "emitted_packet_uses_scorer_loop_best_decoder": False,
            "blockers": ["snerv_scorer_loop_qat_not_requested"],
            **FALSE_AUTHORITY,
        }
        write_json(report_path, payload)
        return {**payload, "report_path": report_path.as_posix()}

    try:
        from tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat import (
            run_snerv_scorer_loop_decoder_qat_smoke,
        )

        result = run_snerv_scorer_loop_decoder_qat_smoke(
            n_pairs=int(num_pairs),
            levels=int(levels),
            wavelet=str(wavelet),
            target_bits_per_coeff=float(target_bits_per_coeff),
            upstream_dir=str(scorer_upstream_dir),
            video_path=str(source_video_path),
            device=str(device),
            snerv_model_size_adapter=model_size.adapter,
            snerv_fc_dim=int(model_size.fc_dim),
            snerv_emb_size=int(model_size.emb_size),
            snerv_patch_radius=int(model_size.patch_radius),
            snerv_mfu_scales=tuple(int(v) for v in model_size.mfu_scales),
            snerv_hfr_gain=float(model_size.hfr_gain),
            snerv_temporal_context=int(model_size.temporal_context),
            decoder_payload_codec=str(decoder_payload_codec),
            qat_bits=int(qat_bits),
            max_trials=int(max_trials),
            search_mode=str(search_mode),
            component_guard_mode=str(component_guard_mode),
        )
        result_payload = result.as_jsonable()
        best_packet = getattr(result, "best_packet", b"")
        if best_packet:
            best_packet = bytes(best_packet)
        best_packet_path = out / "best_packet.snar"
        best_packet_materialized = False
        best_packet_path_str: str | None = None
        best_packet_path_sha256: str | None = None
        if best_packet:
            write_bytes_artifact(
                best_packet_path,
                best_packet,
                allow_overwrite=allow_overwrite,
                expected_existing_sha256=(
                    sha256_file(best_packet_path)
                    if allow_overwrite and best_packet_path.is_file()
                    else None
                ),
            )
            best_packet_materialized = True
            best_packet_path_str = best_packet_path.as_posix()
            best_packet_path_sha256 = sha256_file(best_packet_path)
        blockers = [
            str(blocker) for blocker in result_payload.get("blockers") or [] if blocker
        ]
        if bool(result_payload.get("receiver_contract_satisfied")) and not (
            best_packet_materialized
            and best_packet_path_sha256 == result_payload.get("best_packet_sha256")
        ):
            blockers.append(
                "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export"
            )
        payload = {
            "schema": "snerv_mlx_native_scorer_loop_qat_attachment.v1",
            "requested": True,
            "executed": True,
            "source_schema": result_payload.get("schema"),
            "axis_tag": result_payload.get("axis_tag"),
            "num_pairs": int(result_payload.get("n_pairs") or num_pairs),
            "full_video_coverage": int(result_payload.get("n_pairs") or 0) == 600,
            "scorer_loop_evaluations": int(
                result_payload.get("scorer_loop_evaluations") or 0
            ),
            "baseline_archive_bytes": _nested(result_payload, "baseline", "archive_bytes"),
            "best_archive_bytes": _nested(result_payload, "best", "archive_bytes"),
            "baseline_archive_sha256": _nested(
                result_payload, "baseline", "archive_sha256"
            ),
            "best_archive_sha256": _nested(result_payload, "best", "archive_sha256"),
            "baseline_score_linf": _nested(result_payload, "baseline", "score_linf"),
            "best_score_linf": _nested(result_payload, "best", "score_linf"),
            "best_packet_bytes": int(result_payload.get("best_packet_bytes") or 0),
            "best_packet_sha256": result_payload.get("best_packet_sha256"),
            "best_packet_path": best_packet_path_str,
            "best_packet_path_sha256": best_packet_path_sha256,
            "best_packet_materialized": best_packet_materialized,
            "decoder_payload_codec": str(
                result_payload.get("decoder_payload_codec") or decoder_payload_codec
            ),
            "accepted_improvement": bool(result_payload.get("accepted_improvement")),
            "receiver_contract_satisfied": bool(
                result_payload.get("receiver_contract_satisfied")
            ),
            "ready_for_pose_guard_gate": bool(
                result_payload.get("ready_for_pose_guard_gate")
            ),
            "emitted_packet_uses_scorer_loop_best_decoder": False,
            "result": result_payload,
            "blockers": _ordered_unique(blockers),
            **FALSE_AUTHORITY,
        }
        if best_packet_materialized and best_packet_path_sha256 == result_payload.get(
            "best_packet_sha256"
        ):
            payload["_best_packet_bytes"] = best_packet
    except Exception as exc:
        payload = {
            "schema": "snerv_mlx_native_scorer_loop_qat_attachment.v1",
            "requested": True,
            "executed": False,
            "failure": repr(exc),
            "receiver_contract_satisfied": False,
            "accepted_improvement": False,
            "full_video_coverage": False,
            "emitted_packet_uses_scorer_loop_best_decoder": False,
            "blockers": ["snerv_scorer_loop_qat_attachment_failed"],
            **FALSE_AUTHORITY,
        }
    payload_for_disk = {
        key: value for key, value in payload.items() if key != "_best_packet_bytes"
    }
    write_json(report_path, payload_for_disk)
    return {**payload, "report_path": report_path.as_posix()}


def _scorer_loop_qat_blockers(
    scorer_loop_qat: Mapping[str, Any],
    *,
    num_pairs: int,
) -> list[str]:
    blockers = [
        str(blocker)
        for blocker in scorer_loop_qat.get("blockers") or []
        if str(blocker)
    ]
    if scorer_loop_qat.get("executed") is not True:
        blockers.append("snerv_real_segnet_posenet_teacher_loop_not_attached")
        return _ordered_unique(blockers)
    if scorer_loop_qat.get("receiver_contract_satisfied") is not True:
        blockers.append("snerv_real_segnet_posenet_teacher_loop_receiver_unproven")
    if scorer_loop_qat.get("accepted_improvement") is not True:
        blockers.append("snerv_scorer_loop_qat_no_accepted_improvement")
    if int(num_pairs) != 600 or scorer_loop_qat.get("full_video_coverage") is not True:
        blockers.append("snerv_scorer_loop_qat_not_full_video")
    if scorer_loop_qat.get("emitted_packet_uses_scorer_loop_best_decoder") is not True:
        blockers.append(
            "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export"
        )
    return _ordered_unique(blockers)


def write_snerv_mlx_prefilter_profile(
    artifact: Any,
    archive_bytes: int,
    archive_sha256: str,
    output_path: str | Path,
    upstream_dir: str | Path,
    *,
    component_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a false-authority local MLX prefilter profile for queue gates."""

    art = _artifact_mapping(artifact)
    component = dict(component_profile or {})
    blockers = []
    artifact_blockers = [str(item) for item in art.get("blockers") or [] if item]
    if not component:
        blockers.append("snerv_mlx_prefilter_component_scorers_not_attached")
    if int(art.get("num_pairs") or 0) != 600:
        blockers.append("snerv_mlx_prefilter_not_full_video")
    if artifact_blockers:
        blockers.append("snerv_mlx_prefilter_artifact_has_blockers")
    bridge = art.get("bridge_drift") if isinstance(art.get("bridge_drift"), Mapping) else {}
    if bridge.get("allclose") is not True:
        blockers.append("snerv_mlx_prefilter_bridge_drift_unproven")
    if not art.get("archive_path") or not art.get("archive_sha256"):
        blockers.append("snerv_mlx_prefilter_archive_package_missing")
    elif str(art.get("archive_sha256")) != str(archive_sha256):
        blockers.append("snerv_mlx_prefilter_archive_sha_mismatch")
    if int(art.get("archive_bytes") or 0) and int(art.get("archive_bytes") or 0) != int(
        archive_bytes
    ):
        blockers.append("snerv_mlx_prefilter_archive_bytes_mismatch")
    if art.get("receiver_proof_passed") is not True:
        blockers.append("snerv_mlx_prefilter_receiver_proof_missing")
    if art.get("receiver_contract_satisfied") is not True:
        blockers.append("snerv_mlx_prefilter_receiver_contract_unsatisfied")
    ready_for_cpu_replay = bool(component) and not blockers
    payload = {
        "schema": SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA,
        "artifact_schema": art.get("schema"),
        "artifact_report_path": art.get("report_path"),
        "packet_path": art.get("packet_path"),
        "num_pairs": int(art.get("num_pairs") or 0),
        "archive_bytes": int(archive_bytes),
        "archive_sha256": str(archive_sha256),
        "upstream_dir": Path(upstream_dir).as_posix(),
        "component_profile": component or None,
        "prefilter_ready_for_cpu_replay": ready_for_cpu_replay,
        "artifact_blockers": artifact_blockers,
        "blockers": [b for b in dict.fromkeys(blockers) if b],
        **FALSE_AUTHORITY,
    }
    write_json(output_path, payload)
    return payload


def build_snerv_mlx_native_packet_from_numpy_pairs(
    pairs_nchw255: np.ndarray,
    *,
    levels: int,
    wavelet: str,
    target_bits_per_coeff: float,
    decoder_payload_codec: str,
    step_map_bits_per_coeff: float = 4.0,
    lf_payload_codec: str = "auto",
    model_size: SnervModelSizeConfig | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
) -> SnervArchivePacket:
    """Build an SNAR1 packet from NumPy pair frames using existing SNeRV codecs."""

    pairs = np.asarray(pairs_nchw255, dtype=np.float32)
    if pairs.ndim != 5 or pairs.shape[1] != 2 or pairs.shape[2] != 3:
        raise SnervMlxNativeExportError(
            "pairs_nchw255 must be shaped (pairs, 2, 3, H, W); "
            f"got {tuple(pairs.shape)}"
        )
    if not np.isfinite(pairs).all():
        raise SnervMlxNativeExportError("pairs_nchw255 contains nonfinite values")
    n_pairs, _frames, channels, h, w = (int(v) for v in pairs.shape)
    model_size = model_size or SnervModelSizeConfig()
    pyramids = []
    records: list[tuple[int, int, int, Any]] = []
    for pair_idx in range(n_pairs):
        for frame_idx in range(2):
            for channel_idx in range(channels):
                pyr = encode_frame_lf(
                    pairs[pair_idx, frame_idx, channel_idx],
                    levels=int(levels),
                    wavelet=str(wavelet),
                )
                pyramids.append(pyr)
                records.append((pair_idx, frame_idx, channel_idx, pyr))
    decoder = fit_hf_decoder_least_squares(
        pyramids,
        levels=int(levels),
        model_size=model_size,
    )

    lf_quant_planes: list[np.ndarray] = []
    lf_zero_points: list[float] = []
    step_maps: list[np.ndarray] = []
    n_levels = max(2, round(2.0 ** float(target_bits_per_coeff)))
    for _pair_idx, _frame_idx, _channel_idx, pyr in records:
        q, scale, zero = quantize_lf(pyr.lf, n_levels=n_levels)
        step = np.full(q.shape, float(scale), dtype=np.float32)
        lf_quant_planes.append(q)
        lf_zero_points.append(float(zero))
        step_maps.append(step)

    step_map_importance = np.ones(len(step_maps), dtype=np.float64)
    step_packet = encode_step_maps_waterfill(
        step_maps,
        map_importance=step_map_importance,
        target_bits_per_coeff=float(step_map_bits_per_coeff),
    )
    decoder_payload = encode_decoder_payload(decoder, codec=decoder_payload_codec)
    metadata = {
        "n_pairs": n_pairs,
        "frames_per_pair": 2,
        "channels": channels,
        "levels": int(levels),
        "wavelet": str(wavelet),
        "carrier_hw": [h, w],
        "orig_hw": [h, w],
        "lf_plane_count": len(lf_quant_planes),
        "lf_coeff_count_total": int(sum(int(p.lf.size) for p in pyramids)),
        "lf_zero_dtype": "float32_le",
        "lf_scale_mode": "implicit_per_element_steps_scale_1",
        "lf_payload_codec": str(lf_payload_codec),
        "step_map_packet_schema": step_packet.schema,
        "step_map_coder_mode": "waterfill_mlx_native_uniform_importance_bridge",
        "step_map_coder_bins": None,
        "step_map_waterfill_bits_per_coeff": float(step_map_bits_per_coeff),
        "step_map_coder_groups": [dict(group) for group in step_packet.groups],
        "target_bits_per_coeff": float(target_bits_per_coeff),
        "uniform_quantization_levels": int(n_levels),
        "allocation_mode": "uniform_mlx_native_closed_form_export",
        "hf_decoder_fit_mode": "least_squares_numpy_closed_form",
        "decoder_payload_codec": str(decoder_payload_codec),
        "snerv_fc_dim": int(model_size.fc_dim),
        "snerv_emb_size": int(model_size.emb_size),
        "snerv_patch_radius": int(model_size.patch_radius),
        "snerv_model_size_adapter": model_size.adapter,
        "snerv_spectra_preserving_adapter_enabled": (
            model_size.adapter == SNERV_SPECTRA_PRESERVING_ADAPTER
        ),
        "snerv_mfu_scales": [int(v) for v in model_size.mfu_scales],
        "snerv_hfr_gain": float(model_size.hfr_gain),
        "snerv_temporal_context": int(model_size.temporal_context),
        "decoder_feature_count": int(model_size.feature_count),
        **dict(metadata_extra or {}),
    }
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=lf_zero_points),
        lf_payload=encode_lf_quant_payload(lf_quant_planes, codec=lf_payload_codec),
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata=metadata,
    )
    _verify_receiver_frame_decode(archive, reference_shape=pairs.shape)
    return archive


def build_snerv_mlx_native_storage_preflight(
    *,
    output_dir: str | Path,
    n_pairs: int,
    packet_bytes: int,
    extra_margin_bytes: int = 512 * 1024 * 1024,
) -> dict[str, Any]:
    """Return free-space readiness for receiver proof and archive export."""

    out = Path(output_dir).expanduser().resolve(strict=False)
    out.mkdir(parents=True, exist_ok=True)
    raw_bytes = int(n_pairs) * 2 * 3 * int(CAMERA_HW[0]) * int(CAMERA_HW[1])
    required = raw_bytes + int(packet_bytes) + int(extra_margin_bytes)
    free = shutil.disk_usage(out).free
    return {
        "schema": SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA,
        "path": out.as_posix(),
        "n_pairs": int(n_pairs),
        "expected_receiver_raw_bytes": raw_bytes,
        "packet_bytes": int(packet_bytes),
        "extra_margin_bytes": int(extra_margin_bytes),
        "required_bytes": int(required),
        "free_bytes": int(free),
        "preflight_passed": bool(int(free) >= int(required)),
        **FALSE_AUTHORITY,
    }


def _target_pairs_to_nchw255(target0_np: np.ndarray, target1_np: np.ndarray) -> np.ndarray:
    if target0_np.shape != target1_np.shape:
        raise SnervMlxNativeExportError(
            f"target frame shapes differ: {target0_np.shape} vs {target1_np.shape}"
        )
    if target0_np.ndim != 4 or target0_np.shape[-1] != 3:
        raise SnervMlxNativeExportError(
            "targets must be NHWC RGB shaped (pairs, H, W, 3); "
            f"got {target0_np.shape}"
        )
    pair = np.stack([target0_np, target1_np], axis=1)
    pair = np.clip(pair.astype(np.float32), 0.0, 1.0) * 255.0
    return np.transpose(pair, (0, 1, 4, 2, 3)).astype(np.float32)


def _verify_receiver_frame_decode(
    archive: SnervArchivePacket,
    *,
    reference_shape: Sequence[int],
) -> None:
    decoded = decode_snerv_archive_frames(archive.packet)
    if tuple(int(v) for v in decoded.shape) != tuple(int(v) for v in reference_shape):
        raise SnervMlxNativeExportError(
            f"receiver decode shape {decoded.shape} != reference {tuple(reference_shape)}"
        )
    if not np.isfinite(decoded).all():
        raise SnervMlxNativeExportError("receiver decode produced nonfinite values")


def _model_size_from_candidate(candidate: Mapping[str, Any]) -> SnervModelSizeConfig:
    scales = candidate.get("mfu_scales", candidate.get("snerv_mfu_scales", (1, 2, 4)))
    if isinstance(scales, str):
        scales_tuple = tuple(int(v) for v in scales.split(",") if v.strip())
    else:
        scales_tuple = tuple(int(v) for v in scales)
    return SnervModelSizeConfig(
        fc_dim=int(candidate.get("fc_dim", candidate.get("snerv_fc_dim", 9))),
        emb_size=int(candidate.get("emb_size", candidate.get("snerv_emb_size", 0))),
        patch_radius=int(
            candidate.get("patch_radius", candidate.get("snerv_patch_radius", 1))
        ),
        mfu_scales=scales_tuple,
        hfr_gain=float(candidate.get("hfr_gain", candidate.get("snerv_hfr_gain", 0.0))),
        temporal_context=int(
            candidate.get(
                "temporal_context",
                candidate.get("snerv_temporal_context", 0),
            )
        ),
        adapter=str(
            candidate.get(
                "model_size_adapter",
                candidate.get("snerv_model_size_adapter", "snerv_fc_dim_emb_size_adapter_v1"),
            )
        ),
    )


def _packet_bytes_from_artifact(model_or_artifact: Any) -> bytes:
    if isinstance(model_or_artifact, bytes):
        return bytes(model_or_artifact)
    if isinstance(model_or_artifact, (str, Path)):
        return Path(model_or_artifact).read_bytes()
    mapping = _artifact_mapping(model_or_artifact)
    if isinstance(mapping.get("packet"), bytes):
        return bytes(mapping["packet"])
    for key in ("packet_path", "receiver_archive_packet_path"):
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return Path(value).expanduser().read_bytes()
    raise SnervMlxNativeExportError(
        "model_or_artifact must be packet bytes, a packet path, or a mapping "
        "with packet_path/receiver_archive_packet_path"
    )


def _artifact_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "as_jsonable") and callable(value.as_jsonable):
        return dict(value.as_jsonable())
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return [str(value) for value in dict.fromkeys(str(v) for v in values if str(v))]


def _repo_root(repo_root: str | Path | None) -> Path:
    return (
        Path(repo_root).expanduser().resolve(strict=False)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FALSE_AUTHORITY",
    "SCORER_HW",
    "SNERV_MLX_NATIVE_PACKET_FILENAME",
    "SNERV_MLX_NATIVE_PREFILTER_PROFILE_SCHEMA",
    "SNERV_MLX_NATIVE_REPORT_FILENAME",
    "SNERV_MLX_NATIVE_STORAGE_PREFLIGHT_SCHEMA",
    "SNERV_MLX_NATIVE_TRAIN_EXPORT_SCHEMA",
    "SnervMlxNativeArtifact",
    "SnervMlxNativeExportError",
    "build_snerv_mlx_native_packet_from_numpy_pairs",
    "build_snerv_mlx_native_storage_preflight",
    "export_snerv_mlx_archive",
    "train_export_snerv_mlx_native",
    "write_snerv_mlx_prefilter_profile",
    "write_snerv_mlx_receiver_proof",
]
