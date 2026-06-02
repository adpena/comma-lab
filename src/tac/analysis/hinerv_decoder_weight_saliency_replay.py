# SPDX-License-Identifier: MIT
"""HiNeRV decoder-weight saliency replay through the real scorer loss path.

This module consumes receiver/archive-owned HiNeRV rows, reconstructs the exact
local receiver model, runs rendered frame pairs through the canonical
score-aware Torch loss, and summarizes ``mean(grad^2)`` per decoder tensor.
Those values are the diagonal-Fisher group saliencies consumed by
``nerv_decoder_weight_waterfill``.

The output is deliberately false-authority. A sampled saliency replay is useful
for byte allocation, but it is not a contest score, not an exact eval, and not a
promotion signal.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.analysis.nerv_decoder_weight_waterfill import (
    DEFAULT_EXCLUDE_SUBSTRINGS,
    DEFAULT_INCLUDE_SUBSTRINGS,
    NervDecoderWeightWaterfillError,
    load_state_npz_from_manifest,
)
from tac.data import decode_video
from tac.repo_io import sha256_file
from tac.substrates.hi_nerv.architecture import (
    LATENT_STATE_KEYS,
    HinervConfig,
    HinervSubstrate,
    validate_decoder_state_dict,
)
from tac.substrates.hi_nerv.archive import parse_archive
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.score_aware_common import (
    CONTEST_POSE_SQRT_WEIGHT,
    CONTEST_SEG_WEIGHT,
    score_pair_components_dispatch,
)

SCHEMA = "hinerv_decoder_weight_saliency_replay.v1"
AUTHORITY = "false_authority_decoder_weight_saliency_replay_no_score_claim"
AXIS_TAG = "[macOS-CPU/GPU scorer-loss saliency replay]"


class HinervDecoderWeightSaliencyReplayError(ValueError):
    """Raised when a HiNeRV saliency replay input is incomplete or unsafe."""


ScorerLoader = Callable[[Path, torch.device], tuple[torch.nn.Module, torch.nn.Module]]
PairLoader = Callable[[Path, int, int, int, tuple[int, int]], list[torch.Tensor]]


def build_hinerv_decoder_weight_saliency_replay(
    *,
    archive_ladder_report: Mapping[str, Any],
    row_ids: Sequence[str] | None = None,
    video_path: str | Path,
    upstream_dir: str | Path,
    device: str | torch.device = "cpu",
    max_pairs: int = 1,
    start_pair: int = 0,
    pair_stride: int = 1,
    include_substrings: Sequence[str] = DEFAULT_INCLUDE_SUBSTRINGS,
    exclude_substrings: Sequence[str] = DEFAULT_EXCLUDE_SUBSTRINGS,
    segmentation_surrogate: str = "soft_cosine",
    segmentation_temperature: float = 1.0,
    scorer_loader: ScorerLoader | None = None,
    pair_loader: PairLoader | None = None,
    scorer_source: str = "real_upstream_differentiable_scorers",
) -> dict[str, Any]:
    """Build a false-authority decoder-weight saliency replay report.

    Args:
        archive_ladder_report: ``hinerv_archive_size_ladder.v1`` payload.
        row_ids: optional selected row ids. Defaults to every archive row.
        video_path: real contest video path for target pairs.
        upstream_dir: upstream scorer repo path for real scorer loading.
        device: Torch device used for replay.
        max_pairs / start_pair / pair_stride: sampled pair schedule. Only
            ``max_pairs == num_pairs`` with stride 1 from pair 0 is marked as
            full-video coverage.
        include_substrings / exclude_substrings: decoder tensor selector.
        segmentation_surrogate / segmentation_temperature: canonical SegNet
            differentiable boundary surrogate.
        scorer_loader / pair_loader: injectable only for tests. The CLI uses
            real upstream scorers and real decoded video.
        scorer_source: provenance label for the scorer source.
    """

    if archive_ladder_report.get("schema") != "hinerv_archive_size_ladder.v1":
        raise HinervDecoderWeightSaliencyReplayError(
            "archive_ladder_report must have schema 'hinerv_archive_size_ladder.v1'"
        )
    if int(max_pairs) <= 0:
        raise HinervDecoderWeightSaliencyReplayError("max_pairs must be positive")
    if int(start_pair) < 0:
        raise HinervDecoderWeightSaliencyReplayError("start_pair must be >= 0")
    if int(pair_stride) <= 0:
        raise HinervDecoderWeightSaliencyReplayError("pair_stride must be positive")

    rows = _selected_ladder_rows(archive_ladder_report, row_ids=row_ids)
    if not rows:
        raise HinervDecoderWeightSaliencyReplayError("no archive rows selected")

    dev = torch.device(device)
    video = Path(video_path).expanduser().resolve(strict=False)
    upstream = Path(upstream_dir).expanduser().resolve(strict=False)
    first_cfg = _cfg_from_row_archive(rows[0])
    target_hw = (int(first_cfg.output_height), int(first_cfg.output_width))
    pairs = (pair_loader or _load_real_pair_tensors)(
        video,
        int(max_pairs),
        int(start_pair),
        int(pair_stride),
        target_hw,
    )
    if len(pairs) != int(max_pairs):
        raise HinervDecoderWeightSaliencyReplayError(
            f"pair loader returned {len(pairs)} pairs, expected {max_pairs}"
        )

    loader = scorer_loader or _load_real_differentiable_scorers
    pose_scorer, seg_scorer = loader(upstream, dev)
    pose_scorer.eval()
    seg_scorer.eval()
    for scorer in (pose_scorer, seg_scorer):
        for param in scorer.parameters():
            param.requires_grad_(False)

    replay_rows = [
        _replay_row(
            row,
            pairs=pairs,
            pose_scorer=pose_scorer,
            seg_scorer=seg_scorer,
            device=dev,
            include_substrings=include_substrings,
            exclude_substrings=exclude_substrings,
            segmentation_surrogate=str(segmentation_surrogate),
            segmentation_temperature=float(segmentation_temperature),
            max_pairs=int(max_pairs),
            start_pair=int(start_pair),
            pair_stride=int(pair_stride),
        )
        for row in rows
    ]
    saliency_by_row = {
        str(row["row_id"]): dict(row["saliency_by_name"]) for row in replay_rows
    }
    combined_saliency = _combine_saliency_maps(
        row["saliency_by_name"] for row in replay_rows
    )
    full_video_coverage = all(bool(row["full_video_coverage"]) for row in replay_rows)
    blockers = _ordered_unique(
        [
            blocker
            for row in replay_rows
            for blocker in row.get("blockers", ())
        ]
        + ["contest_cpu_cuda_exact_eval_not_executed"]
    )
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "axis_tag": AXIS_TAG,
        "source_schema": archive_ladder_report.get("schema"),
        "archive_ladder_report_path": archive_ladder_report.get("report_path"),
        "family": "hi_nerv",
        "scorer_source": str(scorer_source),
        "video_path": video.as_posix(),
        "upstream_dir": upstream.as_posix(),
        "device": str(dev),
        "pair_schedule": {
            "max_pairs": int(max_pairs),
            "start_pair": int(start_pair),
            "pair_stride": int(pair_stride),
        },
        "segmentation_surrogate": str(segmentation_surrogate),
        "segmentation_temperature": float(segmentation_temperature),
        "full_video_coverage": bool(full_video_coverage),
        "row_count": len(replay_rows),
        "rows": replay_rows,
        "saliency_by_row": saliency_by_row,
        "saliency_by_name": combined_saliency,
        "saliency_rows": _saliency_rows(combined_saliency),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def write_hinerv_decoder_weight_saliency_replay(
    *,
    archive_ladder_report: Mapping[str, Any],
    output_json: str | Path,
    output_md: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write a HiNeRV decoder saliency replay report."""

    report = build_hinerv_decoder_weight_saliency_replay(
        archive_ladder_report=archive_ladder_report,
        **kwargs,
    )
    json_path = Path(output_json).expanduser().resolve(strict=False)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report["report_path"] = json_path.as_posix()
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if output_md is not None:
        md_path = Path(output_md).expanduser().resolve(strict=False)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_hinerv_decoder_weight_saliency_markdown(report))
    return report


def render_hinerv_decoder_weight_saliency_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing saliency replay summary."""

    lines = [
        "# HiNeRV decoder-weight saliency replay",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Scorer source: `{report.get('scorer_source')}`",
        "",
        "| row | groups | max saliency | full video | blockers |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("rows", ()):
        saliencies = [
            float(value)
            for value in (row.get("saliency_by_name") or {}).values()
            if _is_finite_number(value)
        ]
        lines.append(
            "| {row_id} | {groups} | {max_sal:.6g} | {full} | {blockers} |".format(
                row_id=row.get("row_id"),
                groups=len(saliencies),
                max_sal=max(saliencies) if saliencies else 0.0,
                full="yes" if row.get("full_video_coverage") else "no",
                blockers=len(row.get("blockers") or ()),
            )
        )
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers") or ()
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _selected_ladder_rows(
    archive_ladder_report: Mapping[str, Any],
    *,
    row_ids: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    rows = [
        row for row in archive_ladder_report.get("archive_rows", ())
        if isinstance(row, Mapping)
    ]
    if row_ids is None or not row_ids:
        return rows
    wanted = {str(row_id) for row_id in row_ids}
    selected = [row for row in rows if str(row.get("row_id")) in wanted]
    missing = sorted(wanted - {str(row.get("row_id")) for row in selected})
    if missing:
        raise HinervDecoderWeightSaliencyReplayError(
            f"requested row ids missing from archive ladder: {missing}"
        )
    return selected


def _load_real_differentiable_scorers(
    upstream_dir: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    from tac.scorer import load_differentiable_scorers

    pose_scorer, seg_scorer = load_differentiable_scorers(
        upstream_dir=upstream_dir,
        device=device,
    )
    return pose_scorer, seg_scorer


def _load_real_pair_tensors(
    video_path: Path,
    max_pairs: int,
    start_pair: int,
    pair_stride: int,
    target_hw: tuple[int, int],
) -> list[torch.Tensor]:
    last_pair = start_pair + (max_pairs - 1) * pair_stride
    frame_count = (last_pair + 1) * 2
    frames = decode_video(
        video_path,
        target_h=int(target_hw[0]),
        target_w=int(target_hw[1]),
        max_frames=frame_count,
    )
    if len(frames) < frame_count:
        raise HinervDecoderWeightSaliencyReplayError(
            f"decoded {len(frames)} frames from {video_path}; need {frame_count}"
        )
    pairs: list[torch.Tensor] = []
    for i in range(max_pairs):
        pair_index = start_pair + i * pair_stride
        pair = torch.stack([frames[2 * pair_index], frames[2 * pair_index + 1]])
        pairs.append(pair.permute(0, 3, 1, 2).float().unsqueeze(0).contiguous())
    return pairs


def _cfg_from_row_archive(row: Mapping[str, Any]) -> HinervConfig:
    archive_path = Path(str(row.get("archive_path") or "")).expanduser()
    if not archive_path.is_file():
        raise HinervDecoderWeightSaliencyReplayError(
            f"archive_path missing for row {row.get('row_id')!r}: {archive_path}"
        )
    expected_sha = str(row.get("archive_sha256") or "")
    if len(expected_sha) != 64:
        raise HinervDecoderWeightSaliencyReplayError(
            f"row {row.get('row_id')!r} missing 64-char archive_sha256"
        )
    actual_sha = sha256_file(archive_path)
    if actual_sha != expected_sha:
        raise HinervDecoderWeightSaliencyReplayError(
            f"archive sha mismatch for row {row.get('row_id')!r}: "
            f"expected={expected_sha} actual={actual_sha}"
        )
    blob = _read_zero_bin_from_archive_zip(archive_path)
    arc = parse_archive(blob)
    meta = arc.meta
    return HinervConfig(
        latent_dim_coarse=int(arc.latents_coarse.shape[1]),
        latent_dim_mid=int(arc.latents_mid.shape[1]),
        latent_dim_fine=int(arc.latents_fine.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        mid_injection_block_index=int(meta["mid_injection_block_index"]),
        fine_injection_block_index=int(meta["fine_injection_block_index"]),
        num_pairs=int(arc.latents_coarse.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )


def _read_zero_bin_from_archive_zip(archive_path: Path) -> bytes:
    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        if "0.bin" not in names:
            raise HinervDecoderWeightSaliencyReplayError(
                f"archive.zip missing 0.bin: {archive_path}"
            )
        return zf.read("0.bin")


def _model_from_row(row: Mapping[str, Any], *, device: torch.device) -> HinervSubstrate:
    cfg = _cfg_from_row_archive(row)
    manifest_path = row.get("state_npz_manifest_path")
    if not manifest_path:
        raise HinervDecoderWeightSaliencyReplayError(
            f"row {row.get('row_id')!r} missing state_npz_manifest_path"
        )
    try:
        arrays = load_state_npz_from_manifest(str(manifest_path))
    except NervDecoderWeightWaterfillError as exc:
        raise HinervDecoderWeightSaliencyReplayError(str(exc)) from exc
    state = {
        str(name): torch.from_numpy(np.asarray(array).copy()).to(dtype=torch.float32)
        for name, array in arrays.items()
    }
    decoder_state = {
        name: tensor for name, tensor in state.items() if name not in LATENT_STATE_KEYS
    }
    validate_decoder_state_dict(
        decoder_state,
        cfg,
        context="hi_nerv_saliency_replay_decoder_state",
    )
    model = HinervSubstrate(cfg).to(device)
    model.load_state_dict({name: tensor.to(device) for name, tensor in state.items()})
    model.train(False)
    return model


def _replay_row(
    row: Mapping[str, Any],
    *,
    pairs: Sequence[torch.Tensor],
    pose_scorer: torch.nn.Module,
    seg_scorer: torch.nn.Module,
    device: torch.device,
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
    segmentation_surrogate: str,
    segmentation_temperature: float,
    max_pairs: int,
    start_pair: int,
    pair_stride: int,
) -> dict[str, Any]:
    model = _model_from_row(row, device=device)
    selected_names = _selected_decoder_parameter_names(
        model,
        include_substrings=include_substrings,
        exclude_substrings=exclude_substrings,
    )
    grad_sum_sq = dict.fromkeys(selected_names, 0.0)
    grad_abs_sum = dict.fromkeys(selected_names, 0.0)
    grad_count = dict.fromkeys(selected_names, 0)
    loss_values: list[float] = []
    seg_values: list[float] = []
    pose_values: list[float] = []
    missing_grad_names: set[str] = set()

    for offset, pair in enumerate(pairs):
        pair_index = start_pair + offset * pair_stride
        if pair_index >= model.cfg.num_pairs:
            raise HinervDecoderWeightSaliencyReplayError(
                f"pair index {pair_index} out of range for row {row.get('row_id')!r}"
            )
        idx = torch.tensor([pair_index], device=device, dtype=torch.long)
        gt = pair.to(device=device, dtype=torch.float32)
        model.zero_grad(set_to_none=True)
        rgb_0_unit, rgb_1_unit = model(idx)
        rgb_0 = rgb_0_unit * 255.0
        rgb_1 = rgb_1_unit * 255.0
        seg_term, pose_term = score_pair_components_dispatch(
            seg_scorer=seg_scorer,
            pose_scorer=pose_scorer,
            rgb_0_rt=rgb_0,
            rgb_1_rt=rgb_1,
            gt_rgb_0=gt[:, 0],
            gt_rgb_1=gt[:, 1],
            segmentation_surrogate=segmentation_surrogate,
            segmentation_temperature=segmentation_temperature,
        )
        loss = CONTEST_SEG_WEIGHT * seg_term + CONTEST_POSE_SQRT_WEIGHT * torch.sqrt(
            pose_term.clamp(min=1e-12)
        )
        if not torch.isfinite(loss):
            raise HinervDecoderWeightSaliencyReplayError(
                f"non-finite scorer loss for row {row.get('row_id')!r}"
            )
        loss.backward()
        loss_values.append(float(loss.detach().cpu().item()))
        seg_values.append(float(seg_term.detach().cpu().item()))
        pose_values.append(float(pose_term.detach().cpu().item()))
        for name, param in model.named_parameters():
            if name not in selected_names:
                continue
            grad = param.grad
            if grad is None:
                missing_grad_names.add(name)
                continue
            g = grad.detach().to(device="cpu", dtype=torch.float64)
            grad_sum_sq[name] += float(torch.sum(g * g).item())
            grad_abs_sum[name] += float(torch.sum(torch.abs(g)).item())
            grad_count[name] += int(g.numel())

    saliency_by_name: dict[str, float] = {}
    tensor_rows: list[dict[str, Any]] = []
    for name in selected_names:
        count = int(grad_count[name])
        saliency = grad_sum_sq[name] / float(count) if count > 0 else 0.0
        saliency_by_name[name] = saliency
        tensor_rows.append(
            {
                "group_name": name,
                "decoder_weight_saliency": saliency,
                "grad_abs_mean": grad_abs_sum[name] / float(count) if count > 0 else 0.0,
                "grad_l2": math.sqrt(max(grad_sum_sq[name], 0.0)),
                "grad_numel": count,
                "gradient_available": count > 0,
            }
        )
    full_video = (
        int(max_pairs) == int(model.cfg.num_pairs)
        and int(start_pair) == 0
        and int(pair_stride) == 1
    )
    blockers = []
    if missing_grad_names:
        blockers.append("decoder_parameter_gradient_missing")
    if not full_video:
        blockers.append("full_video_coverage_missing")
    blockers.extend(
        [
            "receiver_replay_is_scorer_loss_saliency_not_archive_score",
            "contest_cpu_cuda_exact_eval_not_executed",
        ]
    )
    return {
        "row_id": row.get("row_id"),
        "archive_path": row.get("archive_path"),
        "archive_sha256": row.get("archive_sha256"),
        "archive_bytes": row.get("archive_bytes"),
        "state_npz_manifest_path": row.get("state_npz_manifest_path"),
        "num_pairs_in_model": int(model.cfg.num_pairs),
        "sampled_pairs": int(max_pairs),
        "pair_schedule": {
            "max_pairs": int(max_pairs),
            "start_pair": int(start_pair),
            "pair_stride": int(pair_stride),
        },
        "full_video_coverage": bool(full_video),
        "loss_summary": {
            "mean_score_loss_proxy": _mean(loss_values),
            "mean_seg_surrogate": _mean(seg_values),
            "mean_pose_dist": _mean(pose_values),
        },
        "saliency_by_name": saliency_by_name,
        "saliency_rows": tensor_rows,
        "missing_grad_names": sorted(missing_grad_names),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _selected_decoder_parameter_names(
    model: HinervSubstrate,
    *,
    include_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> tuple[str, ...]:
    includes = tuple(str(token) for token in include_substrings if str(token))
    excludes = tuple(str(token) for token in exclude_substrings if str(token))
    if not includes:
        raise HinervDecoderWeightSaliencyReplayError("include_substrings is empty")
    names = []
    for name, param in model.named_parameters():
        if param.numel() <= 0:
            continue
        if any(token in name for token in includes) and not any(
            token in name for token in excludes
        ):
            names.append(str(name))
    if not names:
        raise HinervDecoderWeightSaliencyReplayError(
            "decoder saliency selector matched zero parameters"
        )
    return tuple(sorted(names))


def _combine_saliency_maps(
    maps: Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for saliency_map in maps:
        for name, value in saliency_map.items():
            if not _is_finite_number(value):
                continue
            totals[str(name)] = totals.get(str(name), 0.0) + float(value)
            counts[str(name)] = counts.get(str(name), 0) + 1
    return {
        name: totals[name] / float(counts[name])
        for name in sorted(totals)
        if counts.get(name, 0) > 0
    }


def _saliency_rows(saliency_by_name: Mapping[str, float]) -> list[dict[str, Any]]:
    return [
        {
            "group_name": name,
            "decoder_weight_saliency": float(value),
            "score_saliency": float(value),
        }
        for name, value in sorted(saliency_by_name.items())
    ]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _is_finite_number(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value)
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


__all__ = [
    "AUTHORITY",
    "SCHEMA",
    "HinervDecoderWeightSaliencyReplayError",
    "build_hinerv_decoder_weight_saliency_replay",
    "render_hinerv_decoder_weight_saliency_markdown",
    "write_hinerv_decoder_weight_saliency_replay",
]
