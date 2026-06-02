# SPDX-License-Identifier: MIT
"""Source-grounded NeRV ``--modelsize`` to archive-byte planning curves.

This is not a trained archive measurement. It implements the public HNeRV/SNeRV
parameter-budget equations used to solve ``fc_dim`` from ``--modelsize`` and
turns them into an explicit byte-cap planning surface. The output is intentionally
fail-closed: it estimates ideal packed quant payload bytes, records the missing
entropy/receiver measurements, and grants no score, production, or promotion
authority.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

RAW_CONTEST_VIDEO_BYTES = 1164 * 874 * 3 * 1200
DEFAULT_BYTE_CAPS = (36_000, 72_000, 120_000, 150_000, 178_417)
DEFAULT_RESOLUTION_MODES = {
    "scorer_internal_384x512": 384 * 512,
    "contest_output_1164x874": 1164 * 874,
}
FALSE_AUTHORITY = {
    "score_claim": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

Family = Literal["hnerv", "snerv"]


class NervModelsizeCurveError(ValueError):
    """Raised when a source-grounded budget row cannot be computed."""


@dataclass(frozen=True)
class OfficialNervBudgetConfig:
    """Subset of official HNeRV/SNeRV CLI args used by the modelsize equation."""

    family: Family
    modelsize_mparams: float
    frame_pixels: int
    full_data_length: int = 600
    enc_strds: tuple[int, ...] = (5, 4, 4, 2, 2)
    dec_strds: tuple[int, ...] = (5, 4, 4, 2, 2)
    ks: tuple[int, int, int] = (0, 1, 5)
    reduce: float = 1.2
    lower_width: int = 12
    enc_dim: tuple[float, float] = (64.0, 16.0)
    fc_hw: tuple[int, int] = (9, 16)
    embed: str = ""
    emb_size: int = 0
    saturate_stages: int = -1
    quant_model_bit: int = 8
    quant_embed_bit: int = 6
    quant_embed2_bit: int = 6
    quant_axis: int = 0

    @staticmethod
    def hnerv_pr95_like(
        *,
        modelsize_mparams: float,
        frame_pixels: int,
    ) -> OfficialNervBudgetConfig:
        """PR95/HNeRV-like source defaults from official HNeRV README."""

        return OfficialNervBudgetConfig(
            family="hnerv",
            modelsize_mparams=float(modelsize_mparams),
            frame_pixels=int(frame_pixels),
            full_data_length=600,
            enc_strds=(5, 4, 4, 2, 2),
            dec_strds=(5, 4, 4, 2, 2),
            ks=(0, 1, 5),
            reduce=1.2,
            lower_width=12,
            enc_dim=(64.0, 16.0),
            quant_model_bit=8,
            quant_embed_bit=6,
        )

    @staticmethod
    def snerv_official_like(
        *,
        modelsize_mparams: float,
        frame_pixels: int,
        temporal: bool = False,
    ) -> OfficialNervBudgetConfig:
        """SNeRV/SNeRV_T-like source defaults from official SNeRV README."""

        return OfficialNervBudgetConfig(
            family="snerv",
            modelsize_mparams=float(modelsize_mparams),
            frame_pixels=int(frame_pixels),
            full_data_length=600,
            enc_strds=(5, 4, 2, 2, 2),
            dec_strds=(5, 4, 2, 2, 2),
            ks=(0, 1, 5),
            reduce=1.2,
            lower_width=12,
            enc_dim=(64.0, 16.0),
            emb_size=20 if temporal else 0,
            quant_model_bit=8,
            quant_embed_bit=6,
            quant_embed2_bit=6,
        )


def official_modelsize_budget_row(cfg: OfficialNervBudgetConfig) -> dict[str, Any]:
    """Compute one source-grounded parameter/ideal-byte budget row."""

    _validate_config(cfg)
    embed = _embedding_budget(cfg)
    decoder_size = float(cfg.modelsize_mparams) * 1_000_000.0 - embed["embed_param"]
    if decoder_size <= 0:
        raise NervModelsizeCurveError(
            f"modelsize too small for embedding overhead: {cfg.modelsize_mparams}"
        )

    fc_dim = _solve_fc_dim(cfg, embed_dim=float(embed["embed_dim"]), decoder_size=decoder_size)
    if fc_dim <= 0:
        raise NervModelsizeCurveError(f"non-positive fc_dim solved: {fc_dim}")

    ideal = _ideal_quant_payload_bytes(cfg, embed, decoder_size)
    return {
        "schema": "nerv_official_modelsize_budget_row.v1",
        "family": cfg.family,
        "modelsize_mparams": float(cfg.modelsize_mparams),
        "frame_pixels": int(cfg.frame_pixels),
        "full_data_length": int(cfg.full_data_length),
        "source_formula": _source_formula(cfg.family),
        "official_controls": {
            "--modelsize": float(cfg.modelsize_mparams),
            "--ks": "_".join(str(v) for v in cfg.ks),
            "--reduce": float(cfg.reduce),
            "--lower_width": int(cfg.lower_width),
            "--enc_dim": "_".join(_format_number(v) for v in cfg.enc_dim),
            "--enc_strds": list(cfg.enc_strds),
            "--dec_strds": list(cfg.dec_strds),
            "--quant_model_bit": int(cfg.quant_model_bit),
            "--quant_embed_bit": int(cfg.quant_embed_bit),
            "--quant_embed2_bit": int(cfg.quant_embed2_bit),
            "--quant_axis": int(cfg.quant_axis),
            "--emb_size": int(cfg.emb_size),
        },
        "derived": {
            "embed_dim": int(embed["embed_dim"]),
            "embed_hw": embed.get("embed_hw"),
            "fc_param": float(embed["fc_param"]),
            "fc_dim": int(fc_dim),
            "embed_param_estimate": float(embed["embed_param"]),
            "decoder_param_budget_estimate": float(decoder_size),
            "total_param_budget_estimate": float(embed["embed_param"] + decoder_size),
        },
        "ideal_quant_payload": ideal,
        "contest_rate_terms": {
            "ideal_payload_rate": 25.0
            * float(ideal["ideal_packed_payload_bytes"])
            / RAW_CONTEST_VIDEO_BYTES,
            "raw_video_bytes": RAW_CONTEST_VIDEO_BYTES,
        },
        "measured_archive_bytes": None,
        "measured_score_components": None,
        "lower_bound_only": True,
        "blockers": [
            "trained_state_dict_missing",
            "entropy_coded_payload_measurement_missing",
            "zip_brotli_archive_measurement_missing",
            "receiver_inflate_parity_missing",
            "scorer_component_deltas_missing",
        ],
        **FALSE_AUTHORITY,
    }


def build_modelsize_archive_curve(
    *,
    byte_caps: Sequence[int] = DEFAULT_BYTE_CAPS,
    resolution_modes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the no-authority curve artifact for official NeRV modelsize knobs."""

    modes = resolution_modes or dict(DEFAULT_RESOLUTION_MODES)
    curve_rows: list[dict[str, Any]] = []
    for resolution_label, frame_pixels in modes.items():
        for family in ("hnerv", "snerv"):
            for cap in byte_caps:
                template = _template_for_family(family, frame_pixels=frame_pixels)
                solved = solve_modelsize_for_ideal_payload_cap(template, int(cap))
                curve_rows.append(
                    {
                        "resolution_mode": resolution_label,
                        "target_archive_byte_cap": int(cap),
                        "target_cap_rate_term": 25.0
                        * float(cap)
                        / RAW_CONTEST_VIDEO_BYTES,
                        "solved_budget": solved,
                    }
                )

    return {
        "schema": "nerv_modelsize_archive_curve.v1",
        "axis_tag": "[planning/control]",
        "verdict": (
            "GO_MODEL_SIZE_PLANNING__NO_GO_ARCHIVE_OR_SCORE_AUTHORITY_UNTIL_"
            "TRAINED_RECEIVER_BYTES_ARE_MEASURED"
        ),
        "source": {
            "hnerv": _source_formula("hnerv"),
            "snerv": _source_formula("snerv"),
        },
        "byte_caps": [int(v) for v in byte_caps],
        "resolution_modes": modes,
        "raw_contest_video_bytes": RAW_CONTEST_VIDEO_BYTES,
        "curve_rows": curve_rows,
        "required_next_measurements": [
            "train_tiny_source_faithful_state_for_each_selected_modelsize",
            "export_quant_model_and_embedding_payloads",
            "measure_entropy_coded_payload_bytes",
            "measure_zip_or_brotli_archive_bytes",
            "prove_receiver_inflate_parity",
            "collect_macOS_or_MLX_component_delta_prefilter",
            "compare_same_axis_PR95_control_before_beat_claim",
        ],
        "blockers": [
            "modelsize_curve_is_ideal_packed_lower_bound_not_archive_measurement",
            "official_source_forward_parity_missing",
            "trained_quantized_receiver_payloads_missing",
            "component_score_curve_missing",
            "PR101_and_Z5_still_block_exact_full_video_cuda",
        ],
        **FALSE_AUTHORITY,
    }


def solve_modelsize_for_ideal_payload_cap(
    template: OfficialNervBudgetConfig,
    target_bytes: int,
    *,
    min_mparams: float = 0.001,
    max_mparams: float = 4.0,
    iterations: int = 48,
) -> dict[str, Any]:
    """Find the largest modelsize whose ideal packed payload fits ``target_bytes``."""

    if target_bytes <= 0:
        raise NervModelsizeCurveError("target_bytes must be positive")
    lo = float(min_mparams)
    hi = float(max_mparams)
    best: dict[str, Any] | None = None
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        cfg = replace(template, modelsize_mparams=mid)
        try:
            row = official_modelsize_budget_row(cfg)
        except NervModelsizeCurveError:
            lo = mid
            continue
        payload_bytes = int(row["ideal_quant_payload"]["ideal_packed_payload_bytes"])
        if payload_bytes <= target_bytes:
            best = row
            lo = mid
        else:
            hi = mid
    if best is None:
        best = _smallest_valid_budget_row(
            template,
            min_mparams=min_mparams,
            max_mparams=max_mparams,
            iterations=iterations,
        )
        best["blockers"] = [
            *best["blockers"],
            "target_byte_cap_below_minimum_solved_budget",
        ]
    best["target_fit"] = {
        "target_bytes": int(target_bytes),
        "ideal_packed_payload_bytes": int(
            best["ideal_quant_payload"]["ideal_packed_payload_bytes"]
        ),
        "slack_bytes": int(target_bytes)
        - int(best["ideal_quant_payload"]["ideal_packed_payload_bytes"]),
        "fit_is_lower_bound_only": True,
    }
    return best


def _smallest_valid_budget_row(
    template: OfficialNervBudgetConfig,
    *,
    min_mparams: float,
    max_mparams: float,
    iterations: int,
) -> dict[str, Any]:
    """Return the smallest modelsize row whose official equation is valid."""

    lo = float(min_mparams)
    hi = float(max_mparams)
    try:
        official_modelsize_budget_row(replace(template, modelsize_mparams=hi))
    except NervModelsizeCurveError as exc:
        raise NervModelsizeCurveError(
            f"max_mparams does not yield a valid budget: {max_mparams}"
        ) from exc
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        try:
            official_modelsize_budget_row(replace(template, modelsize_mparams=mid))
        except NervModelsizeCurveError:
            lo = mid
        else:
            hi = mid
    return official_modelsize_budget_row(replace(template, modelsize_mparams=hi))


def _template_for_family(
    family: str,
    *,
    frame_pixels: int,
) -> OfficialNervBudgetConfig:
    if family == "hnerv":
        return OfficialNervBudgetConfig.hnerv_pr95_like(
            modelsize_mparams=0.1,
            frame_pixels=frame_pixels,
        )
    if family == "snerv":
        return OfficialNervBudgetConfig.snerv_official_like(
            modelsize_mparams=0.1,
            frame_pixels=frame_pixels,
            temporal=False,
        )
    raise NervModelsizeCurveError(f"unknown family: {family}")


def _embedding_budget(cfg: OfficialNervBudgetConfig) -> dict[str, float | int | None]:
    if "pe" in cfg.embed or "le" in cfg.embed:
        embed_dim = int(cfg.embed.split("_")[-1]) * 2
        fc_param = math.prod(cfg.fc_hw)
        return {
            "embed_dim": embed_dim,
            "embed_hw": None,
            "fc_param": float(fc_param),
            "embed_param": 0.0,
            "embed_param_main": 0.0,
            "embed_param_norm": 0.0,
            "embed_param_temporal": 0.0,
        }

    total_enc_strds = math.prod(cfg.enc_strds)
    enc_dim1, embed_ratio = cfg.enc_dim
    if cfg.family == "hnerv":
        embed_hw = float(cfg.frame_pixels) / float(total_enc_strds**2)
        embed_dim = (
            int(
                embed_ratio
                * cfg.modelsize_mparams
                * 1_000_000.0
                / cfg.full_data_length
                / embed_hw
            )
            if embed_ratio < 1.0
            else int(embed_ratio)
        )
        embed_param = (
            float(embed_dim)
            / float(total_enc_strds**2)
            * cfg.frame_pixels
            * cfg.full_data_length
        )
    elif cfg.family == "snerv":
        embed_hw = math.floor(
            (float(cfg.frame_pixels) / float(total_enc_strds**2)) / 4.0
        )
        embed_dim = (
            int(
                embed_ratio
                * cfg.modelsize_mparams
                * 1_000_000.0
                / cfg.full_data_length
                / max(embed_hw, 1)
            )
            if embed_ratio < 1.0
            else int(embed_ratio)
        )
        embed_param_main = float(embed_dim) * float(embed_hw) * cfg.full_data_length
        embed_param_norm = 2.0 * cfg.full_data_length
        embed_param_temporal = (
            6.0 * float(cfg.emb_size) * float(cfg.emb_size) * 2.0 * cfg.full_data_length
        )
        embed_param = embed_param_main + embed_param_norm + embed_param_temporal
        return {
            "embed_dim": embed_dim,
            "embed_hw": int(embed_hw),
            "fc_param": float(
                (math.prod(cfg.enc_strds) // math.prod(cfg.dec_strds)) ** 2 * 9
            ),
            "embed_param": float(embed_param),
            "embed_param_main": float(embed_param_main),
            "embed_param_norm": float(embed_param_norm),
            "embed_param_temporal": float(embed_param_temporal),
        }
    else:
        raise NervModelsizeCurveError(f"unknown family: {cfg.family}")

    _ = enc_dim1
    return {
        "embed_dim": embed_dim,
        "embed_hw": float(embed_hw),
        "fc_param": float(
            (math.prod(cfg.enc_strds) // math.prod(cfg.dec_strds)) ** 2 * 9
        ),
        "embed_param": float(embed_param),
        "embed_param_main": float(embed_param),
        "embed_param_norm": 0.0,
        "embed_param_temporal": 0.0,
    }


def _solve_fc_dim(
    cfg: OfficialNervBudgetConfig,
    *,
    embed_dim: float,
    decoder_size: float,
) -> int:
    ch_reduce = 1.0 / float(cfg.reduce)
    dec_ks1, dec_ks2 = cfg.ks[1:]
    fix_ch_stages = len(cfg.dec_strds) if cfg.saturate_stages == -1 else cfg.saturate_stages
    active_strides = cfg.dec_strds[:fix_ch_stages]
    saturated_strides = cfg.dec_strds[fix_ch_stages:]
    a = ch_reduce * sum(
        ch_reduce ** (2 * i)
        * float(stride) ** 2
        * float(min((2 * i + dec_ks1), dec_ks2)) ** 2
        for i, stride in enumerate(active_strides)
    )
    b = embed_dim * ((math.prod(cfg.enc_strds) // math.prod(cfg.dec_strds)) ** 2 * 9)
    c = float(cfg.lower_width) ** 2 * sum(
        float(stride) ** 2 * float(min(2 * (fix_ch_stages + i) + dec_ks1, dec_ks2)) ** 2
        for i, stride in enumerate(saturated_strides)
    )
    root = _positive_quadratic_root(a, b, c - decoder_size)
    return int(root)


def _positive_quadratic_root(a: float, b: float, c: float) -> float:
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            raise NervModelsizeCurveError("degenerate modelsize equation")
        return -c / b
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        raise NervModelsizeCurveError(
            f"negative modelsize discriminant: {discriminant}"
        )
    roots = (
        (-b + math.sqrt(discriminant)) / (2.0 * a),
        (-b - math.sqrt(discriminant)) / (2.0 * a),
    )
    positive = [root for root in roots if root > 0.0]
    if not positive:
        raise NervModelsizeCurveError(f"no positive modelsize root: {roots}")
    return max(positive)


def _ideal_quant_payload_bytes(
    cfg: OfficialNervBudgetConfig,
    embed: dict[str, float | int | None],
    decoder_size: float,
) -> dict[str, Any]:
    model_bytes = _ceil_bits_to_bytes(decoder_size, cfg.quant_model_bit)
    embed_main = float(embed["embed_param_main"])
    embed_norm = float(embed["embed_param_norm"])
    embed_temporal = float(embed["embed_param_temporal"])
    embed_main_bytes = _ceil_bits_to_bytes(embed_main + embed_norm, cfg.quant_embed_bit)
    embed_temporal_bytes = _ceil_bits_to_bytes(embed_temporal, cfg.quant_embed2_bit)
    total = model_bytes + embed_main_bytes + embed_temporal_bytes
    return {
        "ideal_packed_payload_bytes": int(total),
        "model_payload_bytes": int(model_bytes),
        "embedding_payload_bytes": int(embed_main_bytes),
        "temporal_embedding_payload_bytes": int(embed_temporal_bytes),
        "assumption": (
            "ceil(param_count * quant_bits / 8) lower bound; excludes per-tensor "
            "min/scale overhead, entropy coding headers, zip/brotli container "
            "bytes, runtime glue, and scorer-fit effects"
        ),
        "measured": False,
    }


def _ceil_bits_to_bytes(param_count: float, bits: int) -> int:
    return math.ceil(max(float(param_count), 0.0) * int(bits) / 8.0)


def _source_formula(family: str) -> dict[str, Any]:
    if family == "hnerv":
        return {
            "repo_url": "https://github.com/haochen-rye/HNeRV.git",
            "head_sha_observed": "4872129c8d004a25477e0c1ffbbff4ba71943ad5",
            "source_file": "train_nerv_all.py",
            "formula": "decoder_size = modelsize*1e6 - embed_param; fc_dim=max_root([a,b,c-decoder_size])",
            "official_controls": [
                "--modelsize",
                "--ks",
                "--reduce",
                "--lower_width",
                "--enc_dim",
                "--enc_strds",
                "--dec_strds",
                "--quant_model_bit",
                "--quant_embed_bit",
            ],
        }
    if family == "snerv":
        return {
            "repo_url": "https://github.com/qwertja/SNeRV.git",
            "head_sha_observed": "0844a08f9591eea9625f8b961ed91d08030e06d1",
            "source_file": "train_snerv.py",
            "formula": "SNeRV extends HNeRV modelsize solve with LF/norm/temporal embedding overhead",
            "official_controls": [
                "--modelsize",
                "--fc_dim",
                "--ks",
                "--reduce",
                "--enc_dim",
                "--emb_size",
                "--quant_model_bit",
                "--quant_embed_bit",
                "--quant_embed2_bit",
            ],
        }
    raise NervModelsizeCurveError(f"unknown family: {family}")


def _validate_config(cfg: OfficialNervBudgetConfig) -> None:
    if cfg.family not in ("hnerv", "snerv"):
        raise NervModelsizeCurveError(f"unknown family: {cfg.family}")
    if cfg.modelsize_mparams <= 0:
        raise NervModelsizeCurveError("modelsize_mparams must be positive")
    if cfg.frame_pixels <= 0:
        raise NervModelsizeCurveError("frame_pixels must be positive")
    if cfg.full_data_length <= 0:
        raise NervModelsizeCurveError("full_data_length must be positive")
    if cfg.reduce <= 0:
        raise NervModelsizeCurveError("reduce must be positive")
    if len(cfg.ks) != 3:
        raise NervModelsizeCurveError("ks must contain three integers")
    if not cfg.enc_strds or not cfg.dec_strds:
        raise NervModelsizeCurveError("stride lists must be non-empty")
    if math.prod(cfg.dec_strds) == 0 or math.prod(cfg.enc_strds) == 0:
        raise NervModelsizeCurveError("strides must be non-zero")


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(float(value))


def parse_byte_caps(values: Iterable[str] | None) -> tuple[int, ...]:
    """Parse repeated/comma-separated byte caps for the CLI."""

    if not values:
        return DEFAULT_BYTE_CAPS
    caps: list[int] = []
    for raw in values:
        for part in str(raw).split(","):
            text = part.strip().replace("_", "")
            if text:
                caps.append(int(text))
    return tuple(caps)


__all__ = [
    "DEFAULT_BYTE_CAPS",
    "DEFAULT_RESOLUTION_MODES",
    "FALSE_AUTHORITY",
    "RAW_CONTEST_VIDEO_BYTES",
    "NervModelsizeCurveError",
    "OfficialNervBudgetConfig",
    "build_modelsize_archive_curve",
    "official_modelsize_budget_row",
    "parse_byte_caps",
    "solve_modelsize_for_ideal_payload_cap",
]
