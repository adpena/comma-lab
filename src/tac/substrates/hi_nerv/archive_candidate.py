# SPDX-License-Identifier: MIT
"""Byte-closed HiNeRV archive export helpers for MLX/local training artifacts.

This module is the receiver/bundling half of the MLX HiNeRV adapter.  It
bridges the MLX renderer's PyTorch-layout ``export_state_dict()`` into the HIV1
archive grammar, writes a contest-shaped ``archive.zip``, projects the payload
into the HPRC representation spine for byte-value accounting, and emits the
shared archive-bound receiver proof/package.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.framework_agnostic.helpers import (
    npz_to_numpy_primitives,
    write_npz_bridge_artifact,
)
from tac.local_acceleration.mlx_numpy_portability_contract import (
    build_mlx_numpy_portability_contract,
)
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.hi_nerv.architecture import (
    HinervSubstrate,
    validate_decoder_state_dict,
)
from tac.substrates.hi_nerv.archive import pack_archive
from tac.substrates.hi_nerv.bitstream import (
    prepare_hi_nerv_decoder_bitstream_state,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.hprc.representation_spine import (
    build_hi_nerv_spine_from_archive_payload,
    write_representation_spine_projection,
)

if TYPE_CHECKING:
    from tac.substrates.hi_nerv.architecture import HinervConfig

HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "hi_nerv_mlx_archive_bound_adapter_package.v1"
)
HI_NERV_MLX_RECEIVER_PROOF_SCHEMA = "hi_nerv_mlx_generated_receiver_proof.v1"
HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID = "hi_nerv_mlx_archive_export"
HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY = "hi_nerv_mlx"
HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND = "hi_nerv_mlx_archive"
HI_NERV_DECODER_RENDERED_PIXEL_PROOF_SCHEMA = (
    "hi_nerv_decoder_preparation_rendered_pixel_proof.v1"
)

_LATENT_KEYS = ("latents_coarse", "latents_mid", "latents_fine")
_STATE_NPZ_NAME = "hi_nerv_mlx_exported_state.npz"
_STATE_NPZ_MANIFEST_NAME = "hi_nerv_mlx_exported_state_npz_manifest.json"
_BITSTREAM_PREPARATION_REPORT_NAME = "hi_nerv_bitstream_preparation.json"


def hi_nerv_mlx_numpy_portability_contract(
    *,
    canonical_npz_bridge_used: bool = True,
    training_backend: str = "mlx",
) -> dict[str, Any]:
    """Return the honest portability contract for the current HiNeRV receiver."""

    backend = str(training_backend)
    return build_mlx_numpy_portability_contract(
        substrate_id="hi_nerv",
        training_backend=backend,
        exported_state_kind=f"pytorch_layout_numpy_arrays_from_{backend}_model",
        archive_payload_kind="hiv1_monolithic_0_bin",
        receiver_runtime_kind="torch_decode_receiver",
        receiver_dependencies=("torch", "brotli", "python_stdlib"),
        numpy_array_export=True,
        canonical_npz_bridge_used=canonical_npz_bridge_used,
        pure_numpy_inflate=False,
        notes=(
            "HiNeRV MLX export is NumPy-array backed, but the contest receiver "
            "currently decodes with PyTorch. This is contest-compliant when "
            "dependency closure passes, but not pure NumPy inflate."
        ),
    )


def _expected_receiver_output_bytes(cfg: HinervConfig) -> int:
    return int(cfg.num_pairs) * 2 * int(CAMERA_HW[0]) * int(CAMERA_HW[1]) * 3


def hi_nerv_meta_from_config(cfg: HinervConfig) -> dict[str, object]:
    """Return the minimal receiver metadata needed to rebuild the decoder."""

    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "mid_injection_block_index": int(cfg.mid_injection_block_index),
        "fine_injection_block_index": int(cfg.fine_injection_block_index),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "use_hierarchical_feature_grid": bool(cfg.use_hierarchical_feature_grid),
        "use_convnext_blocks": bool(cfg.use_convnext_blocks),
        "local_grid_levels": int(cfg.local_grid_levels),
        "local_grid_channels": int(cfg.local_grid_channels),
        "convnext_mlp_ratio": int(cfg.convnext_mlp_ratio),
        "convnext_kernel_size": int(cfg.convnext_kernel_size),
    }


def _state_bridge_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / _STATE_NPZ_NAME, out_dir / _STATE_NPZ_MANIFEST_NAME


def _write_and_reload_exported_state_via_numpy_bridge(
    *,
    exported_state_dict: dict[str, np.ndarray],
    output_dir: Path,
    source_backend: str = "mlx",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Persist and reload the exact NumPy bridge consumed by the packer."""

    npz_path, manifest_path = _state_bridge_paths(output_dir)
    manifest = write_npz_bridge_artifact(
        exported_state_dict,
        npz_path,
        source_backend=str(source_backend),
        bridge_kind=f"hi_nerv_{source_backend}_export_state_dict_to_npz",
        manifest_path=manifest_path,
        require_finite=True,
    )
    if manifest.get("consumption_recommended") is not True:
        raise ValueError(
            "HiNeRV MLX export NPZ bridge is not consumption-recommended: "
            f"{manifest.get('blockers')}"
        )
    return npz_to_numpy_primitives(npz_path.read_bytes()), manifest


def _require_exported_tensor(
    exported_state_dict: dict[str, np.ndarray],
    key: str,
) -> torch.Tensor:
    if key not in exported_state_dict:
        raise ValueError(f"exported_state_dict missing {key!r}")
    return torch.from_numpy(np.asarray(exported_state_dict[key]).copy()).to(
        dtype=torch.float32
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().to("cpu").contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.asarray(arr.shape, dtype="<i8").tobytes())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _decoder_state_sha256(decoder_state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(decoder_state):
        h.update(str(name).encode("utf-8"))
        h.update(b"\0")
        h.update(_tensor_sha256(decoder_state[name]).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _changed_decoder_tensors(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> list[str]:
    changed: list[str] = []
    for name in sorted(set(before) | set(after)):
        before_tensor = before.get(name)
        after_tensor = after.get(name)
        if before_tensor is None or after_tensor is None:
            changed.append(name)
            continue
        if _tensor_sha256(before_tensor) != _tensor_sha256(after_tensor):
            changed.append(name)
    return changed


def _load_receiver_model_for_pixel_proof(
    *,
    cfg: HinervConfig,
    decoder_state: Mapping[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
) -> HinervSubstrate:
    model = HinervSubstrate(cfg).eval()
    state = {
        name: tensor.detach().clone().to(dtype=torch.float32, device="cpu")
        for name, tensor in decoder_state.items()
    }
    state.update(
        {
            "latents_coarse": latents_coarse.detach()
            .clone()
            .to(dtype=torch.float32, device="cpu"),
            "latents_mid": latents_mid.detach()
            .clone()
            .to(dtype=torch.float32, device="cpu"),
            "latents_fine": latents_fine.detach()
            .clone()
            .to(dtype=torch.float32, device="cpu"),
        }
    )
    model.load_state_dict(state, strict=True)
    return model


def _render_receiver_pixels(
    model: HinervSubstrate,
    pair_indices: torch.Tensor,
) -> torch.Tensor:
    rgb_0, rgb_1 = model(pair_indices)
    return (
        torch.stack((rgb_0, rgb_1), dim=1)
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
    )


def _sample_pair_indices_for_pixel_proof(
    *,
    num_pairs: int,
    max_pair_samples: int,
) -> torch.Tensor:
    """Return deterministic spread samples for rendered-pixel mutation proof."""

    total = int(num_pairs)
    requested = int(max_pair_samples)
    if total <= 0:
        raise ValueError("HiNeRV rendered-pixel proof requires num_pairs > 0")
    if requested <= 0:
        raise ValueError("HiNeRV rendered-pixel proof requires max_pair_samples > 0")
    pair_count = min(requested, total)
    if pair_count == total:
        values = list(range(total))
    elif pair_count == 1:
        values = [0]
    else:
        values = [
            round(index * (total - 1) / float(pair_count - 1))
            for index in range(pair_count)
        ]
    return torch.tensor(values, dtype=torch.long)


def _build_decoder_rendered_pixel_proof(
    *,
    decoder_state_before: Mapping[str, torch.Tensor],
    decoder_state_after: Mapping[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
    cfg: HinervConfig,
    max_pair_samples: int = 3,
) -> dict[str, Any]:
    changed_names = _changed_decoder_tensors(decoder_state_before, decoder_state_after)
    pair_indices = _sample_pair_indices_for_pixel_proof(
        num_pairs=int(cfg.num_pairs),
        max_pair_samples=int(max_pair_samples),
    )
    proof: dict[str, Any] = {
        "schema": HI_NERV_DECODER_RENDERED_PIXEL_PROOF_SCHEMA,
        "proof_kind": "sampled_receiver_rendered_pixel_delta",
        "pair_indices": [int(value) for value in pair_indices.tolist()],
        "sampled_pair_count": int(pair_indices.numel()),
        "decoder_tensor_count": len(decoder_state_after),
        "changed_decoder_tensor_count": len(changed_names),
        "changed_decoder_tensor_names": changed_names,
        "decoder_state_sha256_before": _decoder_state_sha256(decoder_state_before),
        "decoder_state_sha256_after": _decoder_state_sha256(decoder_state_after),
        "blockers": [
            "sampled_rendered_pixel_proof_not_full_video",
            "contest_cpu_cuda_exact_eval_not_executed",
            "scorer_replay_not_executed",
        ],
        **FALSE_AUTHORITY,
    }
    if not changed_names:
        proof.update(
            {
                "proof_status": "not_required_no_decoder_state_change",
                "decoder_state_changed": False,
                "rendered_pixels_changed": False,
                "changed_rendered_pixel_count": 0,
                "max_abs_rendered_pixel_delta": 0.0,
                "mean_abs_rendered_pixel_delta": 0.0,
            }
        )
        return proof

    before_model = _load_receiver_model_for_pixel_proof(
        cfg=cfg,
        decoder_state=decoder_state_before,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
    )
    after_model = _load_receiver_model_for_pixel_proof(
        cfg=cfg,
        decoder_state=decoder_state_after,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
    )
    with torch.no_grad():
        before_pixels = _render_receiver_pixels(before_model, pair_indices)
        after_pixels = _render_receiver_pixels(after_model, pair_indices)
    delta = torch.abs(after_pixels - before_pixels)
    max_abs_delta = float(delta.max().item()) if delta.numel() else 0.0
    changed_pixel_count = int(torch.count_nonzero(delta > 0.0).item())
    rendered_pixels_changed = bool(changed_pixel_count > 0 and max_abs_delta > 0.0)
    proof.update(
        {
            "proof_status": (
                "sampled_rendered_pixels_changed"
                if rendered_pixels_changed
                else "sampled_rendered_pixels_no_change"
            ),
            "decoder_state_changed": True,
            "rendered_pixels_changed": rendered_pixels_changed,
            "changed_rendered_pixel_count": changed_pixel_count,
            "max_abs_rendered_pixel_delta": max_abs_delta,
            "mean_abs_rendered_pixel_delta": (
                float(delta.mean().item()) if delta.numel() else 0.0
            ),
            "rendered_tensor_shape": [int(value) for value in after_pixels.shape],
            "rendered_tensor_sha256_before": _tensor_sha256(before_pixels),
            "rendered_tensor_sha256_after": _tensor_sha256(after_pixels),
        }
    )
    return proof


def _bitstream_report_with_rendered_pixel_proof(
    *,
    prepared_report: Mapping[str, Any],
    decoder_state_before: Mapping[str, torch.Tensor],
    decoder_state_after: Mapping[str, torch.Tensor],
    latents_coarse: torch.Tensor,
    latents_mid: torch.Tensor,
    latents_fine: torch.Tensor,
    cfg: HinervConfig,
) -> dict[str, Any]:
    report = copy.deepcopy(dict(prepared_report))
    proof = _build_decoder_rendered_pixel_proof(
        decoder_state_before=decoder_state_before,
        decoder_state_after=decoder_state_after,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
        cfg=cfg,
    )
    if proof["decoder_state_changed"] and not proof["rendered_pixels_changed"]:
        raise ValueError(
            "HiNeRV decoder bitstream preparation changed decoder tensors but "
            "sampled receiver rendered pixels did not change"
        )
    report["decoder_rendered_pixel_proof"] = proof
    waterfill = report.get("decoder_weight_waterfill")
    if isinstance(waterfill, dict):
        waterfill["rendered_pixel_proof"] = proof
        waterfill["rendered_pixel_proof_status"] = proof["proof_status"]
    return report


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: HinervConfig,
    decoder_codec: str = "int8_mixed",
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None = None,
    return_bitstream_report: bool = False,
) -> bytes | tuple[bytes, dict[str, Any]]:
    """Pack PyTorch-layout exported MLX tensors into HIV1 ``0.bin`` bytes."""

    latents_coarse = _require_exported_tensor(exported_state_dict, "latents_coarse")
    latents_mid = _require_exported_tensor(exported_state_dict, "latents_mid")
    latents_fine = _require_exported_tensor(exported_state_dict, "latents_fine")
    expected_shapes = {
        "latents_coarse": (int(cfg.num_pairs), int(cfg.latent_dim_coarse)),
        "latents_mid": (int(cfg.num_pairs), int(cfg.latent_dim_mid)),
        "latents_fine": (int(cfg.num_pairs), int(cfg.latent_dim_fine)),
    }
    for key, tensor in (
        ("latents_coarse", latents_coarse),
        ("latents_mid", latents_mid),
        ("latents_fine", latents_fine),
    ):
        if tuple(int(v) for v in tensor.shape) != expected_shapes[key]:
            raise ValueError(
                f"{key} shape {tuple(tensor.shape)} != {expected_shapes[key]}"
            )

    decoder_state: dict[str, torch.Tensor] = {}
    for name, arr in exported_state_dict.items():
        if name in _LATENT_KEYS:
            continue
        decoder_state[name] = torch.from_numpy(np.asarray(arr).copy()).to(
            dtype=torch.float32
        )
    validate_decoder_state_dict(
        decoder_state,
        cfg,
        context="hi_nerv_exported_decoder_state",
    )
    prepared = prepare_hi_nerv_decoder_bitstream_state(
        decoder_state,
        pruning_ratio=pruning_ratio,
        quant_noise_bits=quant_noise_bits,
        quant_noise_scale=quant_noise_scale,
        quant_noise_seed=quant_noise_seed,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
    )
    bitstream_report = _bitstream_report_with_rendered_pixel_proof(
        prepared_report=prepared.report,
        decoder_state_before=decoder_state,
        decoder_state_after=prepared.state_dict,
        latents_coarse=latents_coarse,
        latents_mid=latents_mid,
        latents_fine=latents_fine,
        cfg=cfg,
    )

    blob = pack_archive(
        prepared.state_dict,
        latents_coarse,
        latents_mid,
        latents_fine,
        {
            **hi_nerv_meta_from_config(cfg),
            "_hi_nerv_bitstream_preparation": bitstream_report,
        },
        decoder_codec=decoder_codec,
    )
    if return_bitstream_report:
        return blob, bitstream_report
    return blob


def export_hi_nerv_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    decoder_codec: str = "int8_mixed",
    source_backend: str = "mlx",
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX HiNeRV model as a contest-shaped ``archive.zip``."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = model.cfg
    exported_state_dict, npz_bridge_manifest = (
        _write_and_reload_exported_state_via_numpy_bridge(
            exported_state_dict=model.export_state_dict(),
            output_dir=out_dir,
            source_backend=source_backend,
        )
    )
    bin_bytes, bitstream_report = pack_archive_from_exported_state_dict(
        exported_state_dict=exported_state_dict,
        cfg=cfg,
        decoder_codec=decoder_codec,
        pruning_ratio=pruning_ratio,
        quant_noise_bits=quant_noise_bits,
        quant_noise_scale=quant_noise_scale,
        quant_noise_seed=quant_noise_seed,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
        return_bitstream_report=True,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)
    bitstream_report_path = out_dir / _BITSTREAM_PREPARATION_REPORT_NAME
    bitstream_report_path.write_text(
        json.dumps(bitstream_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="hi_nerv",
        repo_root=root,
        vendor_shared_inflate_runtime=True,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes = archive_zip_path.stat().st_size
    write_representation_spine_projection(
        output_dir=out_dir,
        spine=build_hi_nerv_spine_from_archive_payload(
            bin_bytes,
            source={
                "kind": "hi_nerv_export_payload",
                "archive_zip_path": archive_zip_path.as_posix(),
                "archive_zip_sha256": archive_sha256,
                "archive_zip_bytes": int(archive_bytes),
            },
            manifest_extra={
                "emitted_by": "export_hi_nerv_mlx_archive",
                "archive_bytes_are_authority_for_rate": True,
                "decoder_codec": decoder_codec,
                "hi_nerv_bitstream_preparation": bitstream_report,
                "hi_nerv_bitstream_preparation_path": (
                    bitstream_report_path.as_posix()
                ),
                "num_pairs": int(cfg.num_pairs),
                "state_npz_bridge": {
                    "artifact_path": npz_bridge_manifest["artifact_path"],
                    "artifact_sha256": npz_bridge_manifest["artifact_sha256"],
                    "manifest_path": npz_bridge_manifest["manifest_path"],
                    "tensor_count": npz_bridge_manifest["tensor_count"],
                },
                "export_source_backend": str(source_backend),
            },
        ),
        basename="hprc_representation_spine_hi_nerv",
    )
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="hi_nerv_mlx",
            transform_kind=HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind="hi_nerv_mlx_generated_inflate_sh_decode_only_receiver",
            proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="hi_nerv_mlx_receiver_proof.json",
            candidate_label="hi_nerv",
            expected_receiver_output_name="0.raw",
            expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg),
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "hi_nerv_mlx_runtime_adapter_manifest.v1",
                "latent_pyramid": ["coarse", "mid", "fine"],
                "decoder_codec": decoder_codec,
                "hi_nerv_bitstream_preparation": bitstream_report,
                "hi_nerv_bitstream_preparation_path": (
                    bitstream_report_path.as_posix()
                ),
                "num_pairs": int(cfg.num_pairs),
                "state_npz_bridge_manifest": npz_bridge_manifest,
                "mlx_numpy_portability_contract": (
                    hi_nerv_mlx_numpy_portability_contract(
                        training_backend=source_backend
                    )
                ),
            },
            candidate_row_schema="hi_nerv_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_hi_nerv_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    decoder_codec: str = "int8_mixed",
    source_backend: str = "mlx",
    pruning_ratio: float = 0.0,
    quant_noise_bits: int | None = None,
    quant_noise_scale: float = 0.0,
    quant_noise_seed: int = 0,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export HiNeRV MLX bytes and emit the shared candidate package."""

    archive_zip_path, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
        model,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
        decoder_codec=decoder_codec,
        source_backend=source_backend,
        pruning_ratio=pruning_ratio,
        quant_noise_bits=quant_noise_bits,
        quant_noise_scale=quant_noise_scale,
        quant_noise_seed=quant_noise_seed,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
    )
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    cfg = model.cfg
    _, npz_bridge_manifest_path = _state_bridge_paths(out_dir)
    npz_bridge_manifest = json.loads(
        npz_bridge_manifest_path.read_text(encoding="utf-8")
    )
    bitstream_report_path = out_dir / _BITSTREAM_PREPARATION_REPORT_NAME
    bitstream_report = json.loads(bitstream_report_path.read_text(encoding="utf-8"))
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="hi_nerv_mlx",
        transform_kind=HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind="hi_nerv_mlx_generated_inflate_sh_decode_only_receiver",
        proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="hi_nerv_mlx_receiver_proof.json",
        candidate_label="hi_nerv",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg),
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "hi_nerv_mlx_runtime_adapter_manifest.v1",
            "latent_pyramid": ["coarse", "mid", "fine"],
            "decoder_codec": decoder_codec,
            "hi_nerv_bitstream_preparation": bitstream_report,
            "hi_nerv_bitstream_preparation_path": bitstream_report_path.as_posix(),
            "num_pairs": int(cfg.num_pairs),
            "state_npz_bridge_manifest": npz_bridge_manifest,
            "mlx_numpy_portability_contract": (
                hi_nerv_mlx_numpy_portability_contract(
                    training_backend=source_backend
                )
            ),
        },
        candidate_row_schema="hi_nerv_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND",
    "export_hi_nerv_mlx_archive",
    "export_hi_nerv_mlx_archive_bound_candidate_package",
    "hi_nerv_meta_from_config",
    "hi_nerv_mlx_numpy_portability_contract",
    "pack_archive_from_exported_state_dict",
]
