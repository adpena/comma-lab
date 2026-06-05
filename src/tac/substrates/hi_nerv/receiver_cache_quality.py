# SPDX-License-Identifier: MIT
"""HiNeRV archive-backed receiver-cache quality probes.

This module renders scorer-input cache tensors directly from HIV1 archive
bytes through the same PyTorch receiver used by ``inflate.py``.  It is a local
MLX/CPU research-signal helper only: the cache and quality gate never create a
contest score claim, but they do prevent flat-renderer artifacts from entering
section-value, waterfill, or exact-dispatch queues unnoticed.
"""

from __future__ import annotations

import hashlib
import math
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.mlx_cache_quality_gate import write_mlx_cache_quality_gate
from tac.analysis.nerv_distortion_crux import (
    DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS,
    DEFAULT_DISTORTION_CRUX_TOP_K,
    NERV_DISTORTION_CRUX_SCHEMA,
    write_nerv_distortion_crux_report,
)
from tac.local_acceleration.mlx_preprocess import (
    write_scorer_input_cache_from_pair_batches,
)
from tac.repo_io import sha256_file, write_json
from tac.submission_archive import MINIMAL_SINGLE_MEMBER_NAME
from tac.substrates._shared.inflate_runtime import CAMERA_HW, rgb_pair_to_uint8_frames
from tac.substrates.hi_nerv.inflate import build_model_from_archive
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA = (
    "hi_nerv_receiver_cache_quality_report.v1"
)
HI_NERV_DIRECT_RECEIVER_CACHE_AUDIT_SCHEMA = (
    "hi_nerv_direct_receiver_render_cache_identity_audit.v1"
)
HI_NERV_DIRECT_RECEIVER_CACHE_REPORT_SCHEMA = (
    "hi_nerv_direct_receiver_cache_report.v1"
)
HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA = (
    "hi_nerv_receiver_cache_segnet_argmax_probe.v1"
)
HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA = NERV_DISTORTION_CRUX_SCHEMA
SEGNET_ARGMAX_OCCUPANCY_MIN_CLASS_FRACTION = 1.0e-3
SEGNET_ARGMAX_OCCUPANCY_MIN_CLASS_PIXELS = 2
SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE = 0.400001


def write_hi_nerv_receiver_cache_quality_report(
    *,
    archive_zip_path: str | Path,
    output_dir: str | Path,
    reference_cache_dir: str | Path | None = None,
    max_pairs: int = 1,
    batch_pairs: int = 1,
    sample_pairs: int | None = None,
    pair_indices: Sequence[int] | None = None,
    min_segnet_std: float = 1.0,
    min_segnet_dynamic_range: float = 16.0,
    max_segnet_mae_vs_reference_for_fit_gate: float = 64.0,
    min_posenet_yuv6_std: float = 1.0,
    min_posenet_yuv6_dynamic_range: float = 16.0,
    max_posenet_yuv6_mae_vs_reference_for_fit_gate: float = 64.0,
    segnet_argmax_probe_upstream_dir: str | Path | None = None,
    segnet_argmax_probe_device: str = "cpu",
    segnet_argmax_probe_batch_frames: int = 4,
    max_segnet_argmax_disagreement_for_fit_gate: float = 0.25,
    min_segnet_argmax_occupied_class_fraction_for_fit_gate: float = (
        SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE
    ),
    require_segnet_argmax_probe: bool = True,
    segnet_argmax_probe_logits_fn: Any | None = None,
    distortion_crux_probe: bool = True,
    distortion_crux_top_k: int = DEFAULT_DISTORTION_CRUX_TOP_K,
    distortion_crux_min_routable_pairs: int = DEFAULT_DISTORTION_CRUX_MIN_ROUTABLE_PAIRS,
    max_posenet_temporal_delta_mae_for_fit_gate: float = 64.0,
) -> dict[str, Any]:
    """Render a small HiNeRV receiver cache and optionally run a quality gate.

    ``archive_zip_path`` must contain the receiver-consumed HIV1 payload as
    either the current minimal member ``x`` or the legacy root ``0.bin``.  The
    receiver render path is the exact ``build_model_from_archive`` +
    ``rgb_pair_to_uint8_frames`` lowering used by the packaged inflate runtime,
    but the result is still false-authority local evidence.
    """

    archive_path = Path(archive_zip_path).expanduser().resolve(strict=False)
    out = Path(output_dir).expanduser().resolve(strict=False)
    if max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if batch_pairs < 1:
        raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
    if not archive_path.is_file():
        raise FileNotFoundError(f"HiNeRV archive.zip missing: {archive_path}")
    out.mkdir(parents=True, exist_ok=True)

    member_name, payload = _read_hiv1_payload_from_archive_zip(archive_path)
    archive_sha256 = sha256_file(archive_path)
    cache_dir = out / "candidate_cache"
    direct_report, manifest = write_hi_nerv_direct_receiver_cache_from_payload(
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        member_name=member_name,
        archive_payload=payload,
        output_cache_dir=cache_dir,
        max_pairs=int(max_pairs),
        batch_pairs=int(batch_pairs),
        pair_indices=pair_indices,
    )

    quality_gate: dict[str, Any] | None = None
    quality_gate_path: Path | None = None
    if reference_cache_dir is not None:
        quality_gate_path = out / "cache_quality_gate.json"
        quality_gate = write_mlx_cache_quality_gate(
            output_json=quality_gate_path,
            candidate_cache_dir=cache_dir,
            reference_cache_dir=reference_cache_dir,
            sample_pairs=int(sample_pairs or max_pairs),
            min_segnet_std=float(min_segnet_std),
            min_segnet_dynamic_range=float(min_segnet_dynamic_range),
            max_segnet_mae_vs_reference_for_fit_gate=float(
                max_segnet_mae_vs_reference_for_fit_gate
            ),
            min_posenet_yuv6_std=float(min_posenet_yuv6_std),
            min_posenet_yuv6_dynamic_range=float(min_posenet_yuv6_dynamic_range),
            max_posenet_yuv6_mae_vs_reference_for_fit_gate=float(
                max_posenet_yuv6_mae_vs_reference_for_fit_gate
            ),
        )

    segnet_argmax_probe: dict[str, Any] | None = None
    segnet_argmax_probe_path: Path | None = None
    if reference_cache_dir is not None and (
        bool(require_segnet_argmax_probe)
        or segnet_argmax_probe_upstream_dir is not None
        or segnet_argmax_probe_logits_fn is not None
    ):
        segnet_argmax_probe_path = out / "segnet_argmax_probe.json"
        if (
            segnet_argmax_probe_upstream_dir is None
            and segnet_argmax_probe_logits_fn is None
        ):
            segnet_argmax_probe = _build_segnet_argmax_probe_not_run_report(
                candidate_cache_dir=cache_dir,
                reference_cache_dir=reference_cache_dir,
                report_path=segnet_argmax_probe_path,
                reason="segnet_argmax_probe_upstream_dir_not_supplied",
            )
            write_json(segnet_argmax_probe_path, segnet_argmax_probe)
        else:
            try:
                if segnet_argmax_probe_logits_fn is not None:
                    segnet_argmax_probe = (
                        build_hi_nerv_receiver_cache_segnet_argmax_probe(
                            candidate_cache_dir=cache_dir,
                            reference_cache_dir=reference_cache_dir,
                            upstream_dir=segnet_argmax_probe_upstream_dir,
                            sample_pairs=int(sample_pairs or max_pairs),
                            batch_frames=int(segnet_argmax_probe_batch_frames),
                            device=str(segnet_argmax_probe_device),
                            max_segnet_argmax_disagreement_for_fit_gate=float(
                                max_segnet_argmax_disagreement_for_fit_gate
                            ),
                            min_segnet_argmax_occupied_class_fraction_for_fit_gate=float(
                                min_segnet_argmax_occupied_class_fraction_for_fit_gate
                            ),
                            segnet_logits_fn=segnet_argmax_probe_logits_fn,
                        )
                    )
                    segnet_argmax_probe["report_path"] = (
                        segnet_argmax_probe_path.as_posix()
                    )
                    write_json(segnet_argmax_probe_path, segnet_argmax_probe)
                else:
                    segnet_argmax_probe = (
                        write_hi_nerv_receiver_cache_segnet_argmax_probe(
                            output_json=segnet_argmax_probe_path,
                            candidate_cache_dir=cache_dir,
                            reference_cache_dir=reference_cache_dir,
                            upstream_dir=segnet_argmax_probe_upstream_dir,
                            sample_pairs=int(sample_pairs or max_pairs),
                            batch_frames=int(segnet_argmax_probe_batch_frames),
                            device=str(segnet_argmax_probe_device),
                            max_segnet_argmax_disagreement_for_fit_gate=float(
                                max_segnet_argmax_disagreement_for_fit_gate
                            ),
                            min_segnet_argmax_occupied_class_fraction_for_fit_gate=float(
                                min_segnet_argmax_occupied_class_fraction_for_fit_gate
                            ),
                        )
                    )
            except Exception as exc:  # pragma: no cover - exercised by runner refusal path
                upstream_value = (
                    Path(segnet_argmax_probe_upstream_dir)
                    .expanduser()
                    .resolve(strict=False)
                    .as_posix()
                    if segnet_argmax_probe_upstream_dir is not None
                    else None
                )
                segnet_argmax_probe = {
                    "schema": HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA,
                    "candidate_cache_dir": cache_dir.as_posix(),
                    "reference_cache_dir": Path(reference_cache_dir)
                    .expanduser()
                    .resolve(strict=False)
                    .as_posix(),
                    "upstream_dir": upstream_value,
                    "fit_gate_passed": False,
                    "failure": repr(exc),
                    "blockers": [
                        "hi_nerv_receiver_cache_segnet_argmax_probe_is_false_authority",
                        "hi_nerv_receiver_cache_segnet_argmax_probe_failed",
                    ],
                    **FALSE_AUTHORITY,
                }
                segnet_argmax_probe["report_path"] = segnet_argmax_probe_path.as_posix()
                write_json(segnet_argmax_probe_path, segnet_argmax_probe)

    distortion_crux: dict[str, Any] | None = None
    distortion_crux_path: Path | None = None
    if reference_cache_dir is not None and bool(distortion_crux_probe):
        distortion_crux_path = out / "distortion_crux_probe.json"
        try:
            distortion_crux = write_nerv_distortion_crux_report(
                output_json=distortion_crux_path,
                candidate_cache_dir=cache_dir,
                reference_cache_dir=reference_cache_dir,
                sample_pairs=int(sample_pairs or max_pairs),
                top_k=int(distortion_crux_top_k),
                min_routable_pairs=int(distortion_crux_min_routable_pairs),
                max_segnet_last_frame_mae_for_fit_gate=float(
                    max_segnet_mae_vs_reference_for_fit_gate
                ),
                max_posenet_yuv6_pair_mae_for_fit_gate=float(
                    max_posenet_yuv6_mae_vs_reference_for_fit_gate
                ),
                max_posenet_temporal_delta_mae_for_fit_gate=float(
                    max_posenet_temporal_delta_mae_for_fit_gate
                ),
            )
        except Exception as exc:  # pragma: no cover - failure path is runner-owned
            distortion_crux = {
                "schema": HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA,
                "candidate_cache_dir": cache_dir.as_posix(),
                "reference_cache_dir": Path(reference_cache_dir)
                .expanduser()
                .resolve(strict=False)
                .as_posix(),
                "fit_gate_passed": False,
                "failure": repr(exc),
                "blockers": [
                    "nerv_distortion_crux_is_false_authority",
                    "hi_nerv_receiver_cache_distortion_crux_probe_failed",
                ],
                **FALSE_AUTHORITY,
            }
            distortion_crux["report_path"] = distortion_crux_path.as_posix()
            write_json(distortion_crux_path, distortion_crux)

    blockers = ["hi_nerv_receiver_cache_quality_is_false_authority"]
    if quality_gate is None:
        blockers.append("hi_nerv_receiver_cache_quality_reference_gate_not_run")
    else:
        blockers.extend(str(v) for v in quality_gate.get("blockers") or [])
    if segnet_argmax_probe is not None:
        blockers.extend(str(v) for v in segnet_argmax_probe.get("blockers") or [])
    if distortion_crux is not None:
        blockers.extend(str(v) for v in distortion_crux.get("blockers") or [])

    quality_gate_passed = (
        bool(quality_gate.get("fit_gate_passed")) if quality_gate else False
    )
    if segnet_argmax_probe is not None:
        quality_gate_passed = quality_gate_passed and bool(
            segnet_argmax_probe.get("fit_gate_passed")
        )
    if distortion_crux is not None:
        quality_gate_passed = quality_gate_passed and bool(
            distortion_crux.get("fit_gate_passed")
        )

    report = {
        "schema": HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA,
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha256,
        "archive_bytes": int(archive_path.stat().st_size),
        "zip_member": member_name,
        "output_dir": out.as_posix(),
        "candidate_cache_dir": cache_dir.as_posix(),
        "candidate_cache_manifest_path": (cache_dir / "manifest.json").as_posix(),
        "candidate_cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "direct_receiver_cache_report": direct_report,
        "cache_manifest_summary": {
            "pair_count": int(manifest["pair_count"]),
            "raw_sha256": manifest.get("raw_sha256"),
            "array_sha256": dict(manifest.get("array_sha256") or {}),
            "source_kind": manifest.get("source_kind"),
        },
        "quality_gate_path": (
            quality_gate_path.as_posix() if quality_gate_path is not None else None
        ),
        "quality_gate": quality_gate,
        "segnet_argmax_probe_path": (
            segnet_argmax_probe_path.as_posix()
            if segnet_argmax_probe_path is not None
            else None
        ),
        "segnet_argmax_probe": segnet_argmax_probe,
        "distortion_crux_probe_path": (
            distortion_crux_path.as_posix()
            if distortion_crux_path is not None
            else None
        ),
        "distortion_crux_probe": distortion_crux,
        "hard_pair_coverage": (
            distortion_crux.get("hard_pair_coverage")
            if isinstance(distortion_crux, dict)
            else None
        ),
        "quality_gate_passed": quality_gate_passed,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }
    report_path = out / "hi_nerv_receiver_cache_quality_report.json"
    report["report_path"] = report_path.as_posix()
    write_json(report_path, report)
    return report


def write_hi_nerv_receiver_cache_segnet_argmax_probe(
    *,
    output_json: str | Path,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    upstream_dir: str | Path,
    sample_pairs: int = 16,
    batch_frames: int = 4,
    device: str = "cpu",
    max_segnet_argmax_disagreement_for_fit_gate: float = 0.25,
    min_segnet_argmax_occupied_class_fraction_for_fit_gate: float = (
        SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE
    ),
) -> dict[str, Any]:
    """Run real SegNet argmax disagreement on receiver-cache RGB tensors.

    The ordinary cache gate catches flat or badly scaled tensors.  This probe
    asks the scorer-shaped question directly: after the same SegNet forward the
    contest uses, how many last-frame pixels flip class relative to the source?
    It is still local false-authority evidence, but it is score-facing evidence.
    """

    report = build_hi_nerv_receiver_cache_segnet_argmax_probe(
        candidate_cache_dir=candidate_cache_dir,
        reference_cache_dir=reference_cache_dir,
        upstream_dir=upstream_dir,
        sample_pairs=sample_pairs,
        batch_frames=batch_frames,
        device=device,
        max_segnet_argmax_disagreement_for_fit_gate=(
            max_segnet_argmax_disagreement_for_fit_gate
        ),
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=(
            min_segnet_argmax_occupied_class_fraction_for_fit_gate
        ),
    )
    out = Path(output_json).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = out.as_posix()
    write_json(out, report)
    return report


def build_hi_nerv_receiver_cache_segnet_argmax_probe(
    *,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    upstream_dir: str | Path | None,
    sample_pairs: int = 16,
    batch_frames: int = 4,
    device: str = "cpu",
    max_segnet_argmax_disagreement_for_fit_gate: float = 0.25,
    min_segnet_argmax_occupied_class_fraction_for_fit_gate: float = (
        SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE
    ),
    segnet_logits_fn: Any | None = None,
) -> dict[str, Any]:
    """Build the SegNet argmax probe payload.

    ``segnet_logits_fn`` exists for tests and alternative scorer backends.  It
    receives NCHW RGB scorer tensors in the upstream 0..255 domain and returns
    logits in either NCHW or NHWC class layout.
    """

    if sample_pairs < 1:
        raise ValueError(f"sample_pairs must be >= 1, got {sample_pairs}")
    if batch_frames < 1:
        raise ValueError(f"batch_frames must be >= 1, got {batch_frames}")
    threshold = float(max_segnet_argmax_disagreement_for_fit_gate)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "max_segnet_argmax_disagreement_for_fit_gate must be in [0, 1], "
            f"got {threshold}"
        )
    min_candidate_occupied_class_fraction = float(
        min_segnet_argmax_occupied_class_fraction_for_fit_gate
    )
    if not 0.0 <= min_candidate_occupied_class_fraction <= 1.0:
        raise ValueError(
            "min_segnet_argmax_occupied_class_fraction_for_fit_gate must be "
            f"in [0, 1], got {min_candidate_occupied_class_fraction}"
        )

    candidate = Path(candidate_cache_dir).expanduser().resolve(strict=False)
    reference = Path(reference_cache_dir).expanduser().resolve(strict=False)
    upstream = (
        Path(upstream_dir).expanduser().resolve(strict=False)
        if upstream_dir is not None
        else None
    )
    cand_seg = _load_cache_array(candidate, "segnet_last_rgb.npy")
    ref_seg = _load_cache_array(reference, "segnet_last_rgb.npy")
    n = _sample_count(cand_seg, ref_seg, sample_pairs)
    cand_sample = np.asarray(cand_seg[:n], dtype=np.float32)
    ref_sample = np.asarray(ref_seg[:n], dtype=np.float32)
    _validate_segnet_cache_tensor("candidate_segnet_last_rgb", cand_sample)
    _validate_segnet_cache_tensor("reference_segnet_last_rgb", ref_sample)
    if cand_sample.shape[1:] != ref_sample.shape[1:]:
        raise ValueError(
            "candidate/reference SegNet cache shape mismatch: "
            f"{cand_sample.shape} vs {ref_sample.shape}"
        )

    if segnet_logits_fn is None:
        if upstream is None:
            raise ValueError(
                "upstream_dir is required when segnet_logits_fn is not injected"
            )
        logits_fn = _build_real_mlx_segnet_logits_fn(
            upstream_dir=upstream,
            device=device,
        )
    else:
        logits_fn = segnet_logits_fn
    cand_argmax, cand_margin = _run_segnet_argmax_batches(
        cand_sample,
        logits_fn=logits_fn,
        batch_frames=int(batch_frames),
    )
    ref_argmax, ref_margin = _run_segnet_argmax_batches(
        ref_sample,
        logits_fn=logits_fn,
        batch_frames=int(batch_frames),
    )
    if cand_argmax.shape != ref_argmax.shape:
        raise ValueError(
            "candidate/reference SegNet argmax shape mismatch: "
            f"{cand_argmax.shape} vs {ref_argmax.shape}"
        )

    mismatch = cand_argmax != ref_argmax
    total_pixels = int(mismatch.size)
    mismatch_pixels = int(np.count_nonzero(mismatch))
    disagreement = float(mismatch_pixels / total_pixels) if total_pixels else 1.0
    boundary_mask = _segnet_boundary_mask(ref_argmax)
    boundary_pixels = int(np.count_nonzero(boundary_mask))
    boundary_mismatch_pixels = int(np.count_nonzero(mismatch & boundary_mask))
    interior_pixels = int(total_pixels - boundary_pixels)
    interior_mismatch_pixels = int(mismatch_pixels - boundary_mismatch_pixels)
    max_class = int(max(np.max(cand_argmax), np.max(ref_argmax), 0))
    class_count = max(max_class + 1, 5)
    candidate_histogram = [
        int(v) for v in np.bincount(cand_argmax.reshape(-1), minlength=class_count)
    ]
    reference_histogram = [
        int(v) for v in np.bincount(ref_argmax.reshape(-1), minlength=class_count)
    ]
    candidate_occupancy = _argmax_histogram_occupancy(candidate_histogram)
    reference_occupancy = _argmax_histogram_occupancy(reference_histogram)
    candidate_occupied_class_fraction = candidate_occupancy[
        "occupied_class_fraction"
    ]
    reference_occupied_class_fraction = reference_occupancy[
        "occupied_class_fraction"
    ]
    blockers = ["hi_nerv_receiver_cache_segnet_argmax_probe_is_false_authority"]
    if disagreement > threshold:
        blockers.append("candidate_segnet_argmax_disagreement_too_high")
    class_collapse = (
        reference_occupied_class_fraction >= min_candidate_occupied_class_fraction
        and candidate_occupied_class_fraction < min_candidate_occupied_class_fraction
    )
    if class_collapse:
        blockers.append("hi_nerv_receiver_cache_segnet_argmax_class_collapse")

    return {
        "schema": HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA,
        "candidate_cache_dir": candidate.as_posix(),
        "reference_cache_dir": reference.as_posix(),
        "upstream_dir": upstream.as_posix() if upstream is not None else None,
        "scorer_backend": (
            "injected_segnet_logits_fn"
            if segnet_logits_fn is not None
            else "mlx_segnet_adapter"
        ),
        "device": str(device),
        "sample_pairs": int(n),
        "batch_frames": int(batch_frames),
        "argmax_shape": [int(dim) for dim in cand_argmax.shape],
        "total_pixels": total_pixels,
        "mismatch_pixels": mismatch_pixels,
        "segnet_argmax_disagreement_rate": disagreement,
        "boundary_pixels": boundary_pixels,
        "boundary_mismatch_pixels": boundary_mismatch_pixels,
        "boundary_argmax_disagreement_rate": (
            float(boundary_mismatch_pixels / boundary_pixels)
            if boundary_pixels
            else None
        ),
        "interior_pixels": interior_pixels,
        "interior_mismatch_pixels": interior_mismatch_pixels,
        "interior_argmax_disagreement_rate": (
            float(interior_mismatch_pixels / interior_pixels)
            if interior_pixels
            else None
        ),
        "candidate_argmax_histogram": candidate_histogram,
        "reference_argmax_histogram": reference_histogram,
        "candidate_occupied_class_fraction": candidate_occupied_class_fraction,
        "candidate_any_occupied_class_fraction": candidate_occupancy[
            "any_occupied_class_fraction"
        ],
        "reference_occupied_class_fraction": reference_occupied_class_fraction,
        "reference_any_occupied_class_fraction": reference_occupancy[
            "any_occupied_class_fraction"
        ],
        "candidate_top2_margin": _margin_stats(cand_margin),
        "reference_top2_margin": _margin_stats(ref_margin),
        "thresholds": {
            "max_segnet_argmax_disagreement_for_fit_gate": threshold,
            "min_candidate_occupied_class_fraction": (
                min_candidate_occupied_class_fraction
            ),
            "min_class_pixel_fraction_for_occupancy": (
                SEGNET_ARGMAX_OCCUPANCY_MIN_CLASS_FRACTION
            ),
            "min_class_pixel_count_for_occupancy": candidate_occupancy[
                "min_class_pixel_count"
            ],
        },
        "fit_gate_passed": disagreement <= threshold and not class_collapse,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _build_segnet_argmax_probe_not_run_report(
    *,
    candidate_cache_dir: str | Path,
    reference_cache_dir: str | Path,
    report_path: str | Path,
    reason: str,
) -> dict[str, Any]:
    candidate = Path(candidate_cache_dir).expanduser().resolve(strict=False)
    reference = Path(reference_cache_dir).expanduser().resolve(strict=False)
    out = Path(report_path).expanduser().resolve(strict=False)
    return {
        "schema": HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA,
        "candidate_cache_dir": candidate.as_posix(),
        "reference_cache_dir": reference.as_posix(),
        "upstream_dir": None,
        "scorer_backend": None,
        "fit_gate_passed": False,
        "reason": str(reason),
        "report_path": out.as_posix(),
        "blockers": [
            "hi_nerv_receiver_cache_segnet_argmax_probe_is_false_authority",
            "hi_nerv_receiver_cache_segnet_argmax_probe_not_run",
        ],
        **FALSE_AUTHORITY,
    }


def _argmax_histogram_occupied_fraction(values: list[int]) -> float:
    return _argmax_histogram_occupancy(values)["occupied_class_fraction"]


def _argmax_histogram_occupancy(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "occupied_class_fraction": 0.0,
            "any_occupied_class_fraction": 0.0,
            "min_class_pixel_count": SEGNET_ARGMAX_OCCUPANCY_MIN_CLASS_PIXELS,
            "min_class_pixel_fraction": 1.0,
        }
    total = sum(max(0, int(value)) for value in values)
    min_count = max(
        SEGNET_ARGMAX_OCCUPANCY_MIN_CLASS_PIXELS,
        math.ceil(total * SEGNET_ARGMAX_OCCUPANCY_MIN_CLASS_FRACTION),
    )
    if total <= 0:
        min_fraction = 1.0
        occupied = 0
        any_occupied = 0
    else:
        min_fraction = float(min_count / total)
        occupied = sum(1 for value in values if int(value) >= min_count)
        any_occupied = sum(1 for value in values if int(value) > 0)
    return {
        "occupied_class_fraction": float(occupied / len(values)),
        "any_occupied_class_fraction": float(any_occupied / len(values)),
        "min_class_pixel_count": int(min_count),
        "min_class_pixel_fraction": min_fraction,
    }


def write_hi_nerv_direct_receiver_cache_from_payload(
    *,
    archive_path: str | Path,
    archive_sha256: str,
    member_name: str,
    archive_payload: bytes,
    output_cache_dir: str | Path,
    max_pairs: int,
    batch_pairs: int = 1,
    pair_indices: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write a direct receiver scorer-input cache from raw HIV1 payload bytes."""

    import torch

    if max_pairs < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if batch_pairs < 1:
        raise ValueError(f"batch_pairs must be >= 1, got {batch_pairs}")
    source_archive = Path(archive_path).expanduser().resolve(strict=False)
    cache_dir = Path(output_cache_dir).expanduser().resolve(strict=False)
    arc, cfg, model = build_model_from_archive(archive_payload, device="cpu")
    raw_pair_count = int(cfg.num_pairs)
    selected_pair_indices = _select_receiver_cache_pair_indices(
        raw_pair_count=raw_pair_count,
        max_pairs=int(max_pairs),
        pair_indices=pair_indices,
    )
    pair_count = len(selected_pair_indices)
    h, w = CAMERA_HW
    scorer_pair_indices = np.array(
        [[2 * idx, 2 * idx + 1] for idx in selected_pair_indices],
        dtype=np.int64,
    )

    def pair_batches() -> Iterable[np.ndarray]:
        with torch.no_grad():
            for start in range(0, pair_count, int(batch_pairs)):
                chunk_indices = selected_pair_indices[start : start + int(batch_pairs)]
                rendered: list[np.ndarray] = []
                for pair_index in chunk_indices:
                    idx = torch.tensor([pair_index], device="cpu", dtype=torch.long)
                    rgb_0, rgb_1 = model(idx)
                    rendered.append(
                        rgb_pair_to_uint8_frames(
                            rgb_0,
                            rgb_1,
                            input_range="unit",
                        ).reshape(1, 2, h, w, 3)
                    )
                yield np.concatenate(rendered, axis=0)

    manifest = write_scorer_input_cache_from_pair_batches(
        pair_batches(),
        cache_dir,
        pair_count=pair_count,
        pair_indices=scorer_pair_indices,
        frame_shape_hwc=(h, w, 3),
        source=source_archive.as_posix(),
        source_kind="hi_nerv_direct_receiver_render",
        archive_sha256=str(archive_sha256),
        inflated_outputs_aggregate_sha256=None,
        batch_pairs=int(batch_pairs),
        compute_raw_sha256=True,
    )
    manifest["inflated_outputs_aggregate_sha256"] = manifest.get("raw_sha256")

    audit_path = cache_dir / "hi_nerv_direct_receiver_render_cache_identity_audit.json"
    audit = {
        "schema_version": HI_NERV_DIRECT_RECEIVER_CACHE_AUDIT_SCHEMA,
        "verdict": "PASS_HI_NERV_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY",
        "passed": True,
        "created_by": "tac.substrates.hi_nerv.receiver_cache_quality",
        "allowed_use": (
            "certify_hi_nerv_direct_mlx_cache_rebuildability_and_render_quality"
        ),
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch",
        "cache": {
            "archive_sha256": manifest.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": manifest.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "raw_sha256": manifest.get("raw_sha256"),
            "pair_count": manifest.get("pair_count"),
            "hash_domain": manifest.get("hash_domain"),
            "array_sha256": manifest.get("array_sha256"),
        },
        "source": {
            "archive_path": source_archive.as_posix(),
            "archive_sha256": str(archive_sha256),
            "zip_member": str(member_name),
            "archive_magic": "HIV1",
            "schema_version": int(arc.schema_version),
            "config": {
                "num_pairs": int(cfg.num_pairs),
                "latent_dim_coarse": int(cfg.latent_dim_coarse),
                "latent_dim_mid": int(cfg.latent_dim_mid),
                "latent_dim_fine": int(cfg.latent_dim_fine),
                "embed_dim": int(cfg.embed_dim),
                "initial_grid_h": int(cfg.initial_grid_h),
                "initial_grid_w": int(cfg.initial_grid_w),
                "decoder_channels": [int(c) for c in cfg.decoder_channels],
                "sin_frequency": float(cfg.sin_frequency),
                "num_upsample_blocks": int(cfg.num_upsample_blocks),
                "mid_injection_block_index": int(cfg.mid_injection_block_index),
                "fine_injection_block_index": int(cfg.fine_injection_block_index),
                "output_height": int(cfg.output_height),
                "output_width": int(cfg.output_width),
            },
        },
        "direct_render": {
            "raw_pair_count": raw_pair_count,
            "selected_pair_count": int(pair_count),
            "selected_pair_indices": [int(value) for value in selected_pair_indices],
            "selected_pair_ranges": _pair_index_ranges(selected_pair_indices),
            "pair_index_scope": (
                "explicit_source_pair_indices"
                if pair_indices is not None
                else "prefix_from_zero"
            ),
            "frame_shape_hwc": [h, w, 3],
            "batch_pairs": int(batch_pairs),
            "max_pairs": int(max_pairs),
            "raw_file_written": False,
            "rebuilds_from_archive_bytes": True,
            "lowering": "rgb_pair_to_uint8_frames_input_range_unit_bicubic",
        },
        "receiver_proof_required_for_promotion": True,
        **FALSE_AUTHORITY,
    }
    write_json(audit_path, audit)
    manifest["hi_nerv_direct_receiver_render_cache_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": audit_path.as_posix(),
        "sha256": sha256_file(audit_path),
        "verdict": audit["verdict"],
        "passed": True,
        "archive_path": source_archive.as_posix(),
        "archive_sha256": str(archive_sha256),
        **FALSE_AUTHORITY,
    }
    manifest["eligible_for_hi_nerv_direct_rebuild_cleanup"] = True
    write_json(cache_dir / "manifest.json", manifest)

    report = {
        "schema": HI_NERV_DIRECT_RECEIVER_CACHE_REPORT_SCHEMA,
        "source_family": "hi_nerv",
        "archive_path": source_archive.as_posix(),
        "archive_sha256": str(archive_sha256),
        "zip_member": str(member_name),
        "archive_magic": "HIV1",
        "schema_version": int(arc.schema_version),
        "raw_pair_count": raw_pair_count,
        "cached_pair_count": int(manifest["pair_count"]),
        "selected_pair_count": int(pair_count),
        "selected_pair_indices": [int(value) for value in selected_pair_indices],
        "selected_pair_ranges": _pair_index_ranges(selected_pair_indices),
        "pair_index_scope": (
            "explicit_source_pair_indices"
            if pair_indices is not None
            else "prefix_from_zero"
        ),
        "frame_shape_hwc": [h, w, 3],
        "direct_render_raw_bytes": int(manifest["pair_count"]) * 2 * h * w * 3,
        "direct_render_raw_pair_count": int(manifest["pair_count"]),
        "direct_render_raw_sha256": manifest.get("raw_sha256"),
        "direct_render_raw_sha256_scope": manifest.get("raw_sha256_scope"),
        "raw_file_written": False,
        "receiver_proof_required_for_promotion": True,
        "identity_audit_path": audit_path.as_posix(),
        "identity_audit_sha256": manifest[
            "hi_nerv_direct_receiver_render_cache_identity_audit"
        ]["sha256"],
        "candidate_cache_identity_mode": (
            "hi_nerv_direct_receiver_render_cache_identity_audited_false_authority"
        ),
        **FALSE_AUTHORITY,
    }
    return report, manifest


def _select_receiver_cache_pair_indices(
    *,
    raw_pair_count: int,
    max_pairs: int,
    pair_indices: Sequence[int] | None,
) -> list[int]:
    if int(raw_pair_count) < 1:
        raise ValueError("HiNeRV direct receiver cache has no complete pairs")
    if int(max_pairs) < 1:
        raise ValueError(f"max_pairs must be >= 1, got {max_pairs}")
    if pair_indices is None:
        return list(range(min(int(raw_pair_count), int(max_pairs))))
    selected: list[int] = []
    seen: set[int] = set()
    for raw in pair_indices:
        value = int(raw)
        if value < 0:
            raise ValueError(f"pair_indices must be non-negative, got {value}")
        if value >= int(raw_pair_count):
            raise ValueError(
                f"pair index {value} exceeds HiNeRV archive pair count {raw_pair_count}"
            )
        if value in seen:
            continue
        selected.append(value)
        seen.add(value)
        if len(selected) >= int(max_pairs):
            break
    if not selected:
        raise ValueError("pair_indices selected no receiver-cache pairs")
    return selected


def _pair_index_ranges(indices: Sequence[int]) -> list[list[int]]:
    values = [int(value) for value in indices]
    if not values:
        return []
    ranges: list[list[int]] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = prev = value
    ranges.append([start, prev])
    return ranges


def _read_hiv1_payload_from_archive_zip(archive_zip_path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive_zip_path, "r") as zf:
        names = [info.filename for info in zf.infolist() if not info.is_dir()]
        allowed_members = (MINIMAL_SINGLE_MEMBER_NAME, "0.bin")
        payload_members = [name for name in names if name in allowed_members]
        if len(payload_members) != 1:
            raise ValueError(
                "HiNeRV archive.zip must contain exactly one receiver payload "
                f"member named {allowed_members!r}; found payload members "
                f"{payload_members[:10]!r} among archive members {names[:10]!r}"
            )
        member_name = payload_members[0]
        payload = zf.read(member_name)
    if not payload.startswith(b"HIV1"):
        digest = hashlib.sha256(payload).hexdigest()
        raise ValueError(
            f"archive member {member_name} is not HIV1 payload; sha256={digest}"
        )
    return member_name, payload


def _build_real_mlx_segnet_logits_fn(*, upstream_dir: Path, device: str) -> Any:
    from tac.local_acceleration.mlx_scorer_adapters import (
        MLXSegNetAdapter,
        run_mlx_segnet_nchw,
    )
    from tac.scorer import load_default_segnet

    segnet = load_default_segnet(upstream_dir, device=device)
    segnet.eval()
    adapter = MLXSegNetAdapter(segnet)

    def run(x_nchw: np.ndarray) -> np.ndarray:
        return np.asarray(run_mlx_segnet_nchw(adapter, x_nchw), dtype=np.float32)

    return run


def _load_cache_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"cache array missing: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim < 2:
        raise ValueError(f"cache array has invalid rank: {path}")
    return arr


def _sample_count(a: np.ndarray, b: np.ndarray, sample_pairs: int) -> int:
    n = min(int(a.shape[0]), int(b.shape[0]), int(sample_pairs))
    if n < 1:
        raise ValueError("cache arrays have no sample rows")
    return n


def _validate_segnet_cache_tensor(name: str, value: np.ndarray) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must be NCHW rank-4, got shape={value.shape}")
    if int(value.shape[1]) != 3:
        raise ValueError(f"{name} must have 3 RGB channels, got shape={value.shape}")
    if int(value.shape[2]) < 1 or int(value.shape[3]) < 1:
        raise ValueError(f"{name} has invalid spatial shape={value.shape}")


def _run_segnet_argmax_batches(
    frames_nchw: np.ndarray,
    *,
    logits_fn: Any,
    batch_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    argmax_chunks: list[np.ndarray] = []
    margin_chunks: list[np.ndarray] = []
    for start in range(0, int(frames_nchw.shape[0]), int(batch_frames)):
        chunk = np.ascontiguousarray(frames_nchw[start : start + int(batch_frames)])
        logits = np.asarray(logits_fn(chunk), dtype=np.float32)
        class_axis = _infer_segnet_logits_class_axis(logits)
        logits_last = np.moveaxis(logits, class_axis, -1)
        argmax_chunks.append(np.argmax(logits_last, axis=-1).astype(np.int16))
        margin_chunks.append(_top2_margin(logits_last))
    return (
        np.concatenate(argmax_chunks, axis=0),
        np.concatenate(margin_chunks, axis=0),
    )


def _infer_segnet_logits_class_axis(logits: np.ndarray) -> int:
    if logits.ndim != 4:
        raise ValueError(f"SegNet logits must be rank-4, got shape={logits.shape}")
    if int(logits.shape[1]) == 5:
        return 1
    if int(logits.shape[-1]) == 5:
        return -1
    if 2 <= int(logits.shape[1]) <= 32 and int(logits.shape[2]) > 32:
        return 1
    if 2 <= int(logits.shape[-1]) <= 32 and int(logits.shape[1]) > 32:
        return -1
    raise ValueError(f"cannot infer SegNet class axis from shape={logits.shape}")


def _top2_margin(logits_nhwc: np.ndarray) -> np.ndarray:
    if logits_nhwc.shape[-1] < 2:
        raise ValueError("SegNet logits need at least two classes for top-2 margin")
    top2 = np.partition(logits_nhwc, kth=-2, axis=-1)[..., -2:]
    return (top2[..., 1] - top2[..., 0]).astype(np.float32)


def _segnet_boundary_mask(argmax_nhw: np.ndarray) -> np.ndarray:
    boundary = np.zeros(argmax_nhw.shape, dtype=bool)
    boundary[:, 1:, :] |= argmax_nhw[:, 1:, :] != argmax_nhw[:, :-1, :]
    boundary[:, :-1, :] |= argmax_nhw[:, 1:, :] != argmax_nhw[:, :-1, :]
    boundary[:, :, 1:] |= argmax_nhw[:, :, 1:] != argmax_nhw[:, :, :-1]
    boundary[:, :, :-1] |= argmax_nhw[:, :, 1:] != argmax_nhw[:, :, :-1]
    return boundary


def _margin_stats(margin: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(margin)),
        "p10": float(np.quantile(margin, 0.10)),
        "p50": float(np.quantile(margin, 0.50)),
        "p90": float(np.quantile(margin, 0.90)),
        "min": float(np.min(margin)),
        "max": float(np.max(margin)),
    }


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


__all__ = [
    "HI_NERV_DIRECT_RECEIVER_CACHE_AUDIT_SCHEMA",
    "HI_NERV_DIRECT_RECEIVER_CACHE_REPORT_SCHEMA",
    "HI_NERV_RECEIVER_CACHE_DISTORTION_CRUX_SCHEMA",
    "HI_NERV_RECEIVER_CACHE_QUALITY_REPORT_SCHEMA",
    "HI_NERV_RECEIVER_CACHE_SEGNET_ARGMAX_PROBE_SCHEMA",
    "SEGNET_ARGMAX_MIN_OCCUPIED_CLASS_FRACTION_FOR_FIT_GATE",
    "build_hi_nerv_receiver_cache_segnet_argmax_probe",
    "write_hi_nerv_direct_receiver_cache_from_payload",
    "write_hi_nerv_receiver_cache_quality_report",
    "write_hi_nerv_receiver_cache_segnet_argmax_probe",
]
