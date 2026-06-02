# SPDX-License-Identifier: MIT
"""False-authority model-size ladders for HiNeRV and SNeRV controls.

The contest rate term prices every archive byte at a fixed score cost, so model
size is a discrete RD ladder rather than a free configuration knob. This module
does not score a carrier. It measures or projects receiver-visible payload
sections and emits the minimum non-rate-score improvement required for each
larger step to be worth spending.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from math import ceil
from typing import Any

from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates.hi_nerv.architecture import HinervConfig
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.carrier import SnervModelSizeConfig

NERV_MODELSIZE_LADDER_SCHEMA = "nerv_modelsize_ladder.v1"
SCORER_ONLY_OBJECTIVE_AUTHORITY: dict[str, Any] = {
    "objective": "contest_auth_eval_scorer_only",
    "allowed_selection_terms": [
        "SegNet_last_frame_distortion",
        "PoseNet_pair_distortion",
        "archive_zip_bytes_rate_term",
    ],
    "forbidden_selection_terms": [
        "human_visual_fidelity",
        "PSNR",
        "SSIM",
        "LPIPS",
        "perceptual_quality_unless_proven_scorer_causal",
    ],
    "rule": (
        "human visual fidelity is not an authority surface; optimize only the "
        "contest auth eval scorer and byte price"
    ),
}
_SCORER_H = 384
_SCORER_W = 512
_CHANNELS = 3
_FRAMES_PER_PAIR = 2
_PACK_GROUP_SIZE = 64
_SCALE_BYTES_PER_GROUP = 2


def build_nerv_modelsize_ladder(
    *,
    focus_families: Iterable[str] = ("hi_nerv", "snerv"),
    num_pairs: int = 600,
    scorer_height: int = _SCORER_H,
    scorer_width: int = _SCORER_W,
) -> dict[str, Any]:
    """Build a local model-size ladder for compact NeRV-family controls."""

    focus = tuple(
        dict.fromkeys(str(item).strip() for item in focus_families if str(item).strip())
    )
    family_rows = []
    if not focus or "hi_nerv" in focus:
        family_rows.append(_hi_nerv_family_row(num_pairs=int(num_pairs)))
    if not focus or "snerv" in focus:
        family_rows.append(
            _snerv_family_row(
                num_pairs=int(num_pairs),
                scorer_height=int(scorer_height),
                scorer_width=int(scorer_width),
            )
        )
    blockers = _ordered_unique(
        [
            blocker
            for row in family_rows
            for blocker in row.get("blockers", ())
        ]
        + [
            "modelsize_ladder_false_authority_no_nonrate_score",
            "archive_zip_runtime_overhead_not_in_payload_projection",
        ]
    )
    return {
        "schema": NERV_MODELSIZE_LADDER_SCHEMA,
        "authority": "false_authority_modelsize_ladder_no_score_claim",
        "focus_families": list(focus),
        "num_pairs": int(num_pairs),
        "frames": int(num_pairs) * _FRAMES_PER_PAIR,
        "scorer_input_hw": [int(scorer_height), int(scorer_width)],
        "objective_authority": SCORER_ONLY_OBJECTIVE_AUTHORITY,
        "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
        "section_payload_rule": (
            "payload byte estimates exclude ZIP/runtime overhead and require "
            "receiver-closed archive measurement before promotion"
        ),
        "family_rows": family_rows,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def render_nerv_modelsize_ladder_markdown(report: Mapping[str, Any]) -> str:
    """Render an operator-facing Markdown ladder."""

    lines = [
        "# NeRV model-size ladder",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Contest byte price: `{report.get('contest_byte_price_score_per_byte')}`",
        "",
    ]
    for family in report.get("family_rows", ()):
        lines.extend(
            [
                f"## {family['family']}",
                "",
                "| row | main control | fp32 bytes | int8 bytes | int4 bytes | int2 bytes |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in family.get("ladder_rows", ()):
            estimates = row["quantized_payload_estimates"]
            lines.append(
                "| {row_id} | {control} | {fp32} | {int8} | {int4} | {int2} |".format(
                    row_id=row["row_id"],
                    control=row["main_control"],
                    fp32=estimates["fp32"]["payload_bytes"],
                    int8=estimates["int8"]["payload_bytes"],
                    int4=estimates["int4"]["payload_bytes"],
                    int2=estimates["int2"]["payload_bytes"],
                )
            )
        lines.extend(["", "Marginal gates:", ""])
        for gate in family.get("marginal_gates", ()):
            lines.append(
                "- `{mode}` `{from_row_id}` -> `{to_row_id}` needs non-rate "
                "drop >= `{required}`".format(
                    mode=gate["quant_mode"],
                    from_row_id=gate["from_row_id"],
                    to_row_id=gate["to_row_id"],
                    required=gate["required_nonrate_score_improvement"],
                )
            )
        lines.append("")
    lines.extend(["## Blockers", ""])
    blockers = report.get("blockers", ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def hi_nerv_modelsize_config_rows(*, num_pairs: int = 600) -> list[dict[str, Any]]:
    """Return local HiNeRV config rows used by projected and measured ladders."""

    return [
        {
            "row_id": "hi_nerv_local_tiny",
            "modelsize_scale": 0.50,
            "config": replace(
                HinervConfig(),
                num_pairs=num_pairs,
                latent_dim_coarse=8,
                latent_dim_mid=10,
                latent_dim_fine=12,
                embed_dim=32,
                decoder_channels=(24, 20, 16, 12, 10, 8, 6),
            ),
        },
        {
            "row_id": "hi_nerv_local_small",
            "modelsize_scale": 0.75,
            "config": replace(
                HinervConfig(),
                num_pairs=num_pairs,
                latent_dim_coarse=12,
                latent_dim_mid=15,
                latent_dim_fine=18,
                embed_dim=48,
                decoder_channels=(36, 30, 24, 18, 15, 12, 9),
            ),
        },
        {
            "row_id": "hi_nerv_local_base",
            "modelsize_scale": 1.0,
            "config": replace(HinervConfig(), num_pairs=num_pairs),
        },
        {
            "row_id": "hi_nerv_local_wide",
            "modelsize_scale": 1.50,
            "config": replace(
                HinervConfig(),
                num_pairs=num_pairs,
                latent_dim_coarse=24,
                latent_dim_mid=30,
                latent_dim_fine=36,
                embed_dim=96,
                decoder_channels=(72, 60, 48, 36, 30, 24, 18),
            ),
        },
    ]


def _hi_nerv_family_row(*, num_pairs: int) -> dict[str, Any]:
    rows = []
    for spec in hi_nerv_modelsize_config_rows(num_pairs=num_pairs):
        row_id = str(spec["row_id"])
        modelsize_scale = float(spec["modelsize_scale"])
        cfg = spec["config"]
        section_counts = _hi_nerv_section_counts(cfg)
        total_params = sum(section_counts.values())
        rows.append(
            {
                "family": "hi_nerv",
                "row_id": row_id,
                "main_control": f"modelsize_scale={modelsize_scale:g}",
                "config": _hi_nerv_config_snapshot(cfg),
                "section_param_counts": section_counts,
                "total_parameter_count": int(total_params),
                "quantized_payload_estimates": _quant_estimates(section_counts),
                "nonrate_score_required": True,
                "blockers": [
                    "hi_nerv_nonrate_score_missing_for_modelsize_budget_plan",
                    "hi_nerv_payload_estimate_not_archive_zip_measurement",
                ],
            }
        )
    return {
        "family": "hi_nerv",
        "ladder_kind": "local_config_family_parameter_ladder",
        "source_fidelity_note": (
            "local configs expose a priced model-size ladder; official HiNeRV "
            "S/M/L and bitstream-q parity remain separate blockers"
        ),
        "ladder_rows": rows,
        "marginal_gates": _marginal_gates(rows),
        "recommended_next_actions": [
            "train_each_surviving_hi_nerv_size_with_same_score_aware_schedule",
            "replace_payload_projection_with_byte_closed_archive_section_measurement",
            "feed_measured_nonrate_score_rows_into_modelsize_budget_plan",
        ],
        "blockers": [
            "hi_nerv_measured_nonrate_modelsize_ladder_missing",
            "hi_nerv_byte_closed_modelsize_ladder_missing",
        ],
        **FALSE_AUTHORITY,
    }


def _snerv_family_row(
    *,
    num_pairs: int,
    scorer_height: int,
    scorer_width: int,
) -> dict[str, Any]:
    specs = (
        ("snerv_l4_lf2_fc4e0_decoder_int4", 4, 2, 4, 4, 0),
        ("snerv_l4_lf2_fc9e0_decoder_int4", 4, 2, 4, 9, 0),
        ("snerv_l4_lf2_fc12e4_decoder_int4", 4, 2, 4, 12, 4),
        ("snerv_l3_lf2_fc9e0_decoder_int4", 3, 2, 4, 9, 0),
        ("snerv_l3_lf4_fc9e0_decoder_int4", 3, 4, 4, 9, 0),
        ("snerv_l2_lf4_fc9e0_decoder_int4", 2, 4, 4, 9, 0),
        ("snerv_l2_lf8_fc9e0_decoder_int8", 2, 8, 8, 9, 0),
        ("snerv_l1_lf8_fc9e0_decoder_int8", 1, 8, 8, 9, 0),
    )
    rows = []
    for row_id, levels, lf_bits, decoder_bits, fc_dim, emb_size in specs:
        model_size = SnervModelSizeConfig(fc_dim=fc_dim, emb_size=emb_size)
        section_counts = _snerv_section_counts(
            levels=levels,
            lf_quant_bits=lf_bits,
            decoder_quant_bits=decoder_bits,
            model_size=model_size,
            num_pairs=num_pairs,
            height=scorer_height,
            width=scorer_width,
        )
        rows.append(
            {
                "family": "snerv",
                "row_id": row_id,
                "main_control": (
                    f"levels={levels}, lf_bits={lf_bits}, decoder_bits={decoder_bits}"
                ),
                "config": {
                    "levels": int(levels),
                    "lf_quant_bits": int(lf_bits),
                    "decoder_quant_bits": int(decoder_bits),
                    "snerv_fc_dim": int(model_size.fc_dim),
                    "snerv_emb_size": int(model_size.emb_size),
                    "snerv_patch_radius": int(model_size.patch_radius),
                    "decoder_feature_count": int(model_size.feature_count),
                    "snerv_model_size_adapter": model_size.adapter,
                    "n_pairs": int(num_pairs),
                    "scorer_input_hw": [int(scorer_height), int(scorer_width)],
                },
                "section_scalar_counts": {
                    "lf_coefficients": section_counts["lf_coefficients"],
                    "hf_decoder_parameters": section_counts["hf_decoder_parameters"],
                    "metadata_scalars": section_counts["metadata_scalars"],
                },
                "lf_shape": section_counts["lf_shape"],
                "quantized_payload_estimates": _snerv_payload_estimates(section_counts),
                "nonrate_score_required": True,
                "blockers": [
                    "snerv_nonrate_score_missing_for_modelsize_budget_plan",
                    "snerv_payload_estimate_not_archive_zip_measurement",
                ],
            }
        )
    rows.sort(key=lambda row: row["quantized_payload_estimates"]["configured"]["payload_bytes"])
    return {
        "family": "snerv",
        "ladder_kind": "lf_depth_quant_and_shared_decoder_payload_ladder",
        "source_fidelity_note": (
            "local inverse-steg SNeRV estimates LF/depth payloads; official "
            "DWT/MFU/HFR/SNeRV-T parity remains a separate blocker"
        ),
        "ladder_rows": rows,
        "marginal_gates": _marginal_gates(rows, quant_modes=("configured", "int4")),
        "recommended_next_actions": [
            "attach_full_video_advisory_nonrate_score_to_each_snerv_ladder_row",
            "replace_payload_projection_with_receiver_closed_archive_bytes",
            "sweep_receiver_visible_snerv_fc_dim_emb_size_against_full_video_score",
            "push_wavelet_group_saliency_into_lf_depth_and_quant_selection",
        ],
        "blockers": [
            "snerv_measured_nonrate_modelsize_ladder_missing",
            "snerv_byte_closed_modelsize_ladder_missing",
        ],
        **FALSE_AUTHORITY,
    }


def _hi_nerv_section_counts(cfg: HinervConfig) -> dict[str, int]:
    channels = [int(cfg.embed_dim), *[int(value) for value in cfg.decoder_channels]]
    latent_params = int(cfg.num_pairs) * (
        int(cfg.latent_dim_coarse)
        + int(cfg.latent_dim_mid)
        + int(cfg.latent_dim_fine)
    )
    latent_embed_out = (
        int(cfg.embed_dim) * int(cfg.initial_grid_h) * int(cfg.initial_grid_w)
    )
    latent_embed_params = int(cfg.latent_dim_coarse) * latent_embed_out + latent_embed_out
    up_block_params = 0
    for idx in range(int(cfg.num_upsample_blocks)):
        in_ch = channels[idx]
        out_ch = channels[idx + 1] * 4
        up_block_params += out_ch * in_ch * 3 * 3 + out_ch
    mid_ch = channels[int(cfg.mid_injection_block_index) + 1]
    fine_ch = channels[int(cfg.fine_injection_block_index) + 1]
    injector_params = (
        int(cfg.latent_dim_mid) * mid_ch
        + mid_ch
        + int(cfg.latent_dim_fine) * fine_ch
        + fine_ch
    )
    final_ch = channels[int(cfg.num_upsample_blocks)]
    head_params = 2 * (3 * final_ch * 3 * 3 + 3)
    return {
        "latents": int(latent_params),
        "latent_embed": int(latent_embed_params),
        "upsample_blocks": int(up_block_params),
        "latent_injectors": int(injector_params),
        "rgb_heads": int(head_params),
    }


def _hi_nerv_config_snapshot(cfg: HinervConfig) -> dict[str, Any]:
    return {
        "latent_dim_coarse": int(cfg.latent_dim_coarse),
        "latent_dim_mid": int(cfg.latent_dim_mid),
        "latent_dim_fine": int(cfg.latent_dim_fine),
        "embed_dim": int(cfg.embed_dim),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "num_pairs": int(cfg.num_pairs),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
    }


def _snerv_section_counts(
    *,
    levels: int,
    lf_quant_bits: int,
    decoder_quant_bits: int,
    model_size: SnervModelSizeConfig,
    num_pairs: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    lf_h, lf_w = int(height), int(width)
    for _ in range(int(levels)):
        lf_h = _ceil_div(lf_h, 2)
        lf_w = _ceil_div(lf_w, 2)
    frames = int(num_pairs) * _FRAMES_PER_PAIR
    lf_coefficients = frames * _CHANNELS * lf_h * lf_w
    # One HF generator is shared across all frame channels; channel planes are
    # extra training samples, not extra receiver decoder weights.
    hf_decoder_parameters = int(levels) * 3 * int(model_size.feature_count)
    metadata_scalars = frames * _CHANNELS * 2
    return {
        "levels": int(levels),
        "lf_quant_bits": int(lf_quant_bits),
        "decoder_quant_bits": int(decoder_quant_bits),
        "snerv_fc_dim": int(model_size.fc_dim),
        "snerv_emb_size": int(model_size.emb_size),
        "snerv_patch_radius": int(model_size.patch_radius),
        "decoder_feature_count": int(model_size.feature_count),
        "lf_shape": [lf_h, lf_w],
        "frames": frames,
        "lf_coefficients": int(lf_coefficients),
        "hf_decoder_parameters": int(hf_decoder_parameters),
        "metadata_scalars": int(metadata_scalars),
    }


def _quant_estimates(section_counts: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    return {
        "fp32": _payload_estimate(section_counts, bits=32, grouped=False),
        "fp16": _payload_estimate(section_counts, bits=16, grouped=False),
        "int8": _payload_estimate(section_counts, bits=8, grouped=True),
        "int4": _payload_estimate(section_counts, bits=4, grouped=True),
        "int2": _payload_estimate(section_counts, bits=2, grouped=True),
    }


def _payload_estimate(
    section_counts: Mapping[str, int],
    *,
    bits: int,
    grouped: bool,
) -> dict[str, Any]:
    section_bytes = {}
    for name, count in section_counts.items():
        section_bytes[name] = _packed_bytes(
            int(count),
            bits=int(bits),
            grouped=bool(grouped),
        )
    payload_bytes = int(sum(section_bytes.values()))
    return {
        "bits_per_scalar": int(bits),
        "grouped_scale_overhead": bool(grouped),
        "group_size": _PACK_GROUP_SIZE if grouped else None,
        "scale_bytes_per_group": _SCALE_BYTES_PER_GROUP if grouped else 0,
        "section_bytes": section_bytes,
        "payload_bytes": payload_bytes,
        "rate_score_at_contest_price": float(payload_bytes * CONTEST_BYTE_PRICE_SCORE),
    }


def _snerv_payload_estimates(section_counts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configured = {
        "lf_coefficients": _packed_bytes(
            int(section_counts["lf_coefficients"]),
            bits=int(section_counts["lf_quant_bits"]),
            grouped=True,
        ),
        "hf_decoder_parameters": _packed_bytes(
            int(section_counts["hf_decoder_parameters"]),
            bits=int(section_counts["decoder_quant_bits"]),
            grouped=True,
        ),
        "metadata_scalars": int(section_counts["metadata_scalars"]) * 2,
    }
    configured_bytes = int(sum(configured.values()))
    estimates = {
        "configured": {
            "lf_quant_bits": int(section_counts["lf_quant_bits"]),
            "decoder_quant_bits": int(section_counts["decoder_quant_bits"]),
            "section_bytes": configured,
            "payload_bytes": configured_bytes,
            "rate_score_at_contest_price": float(
                configured_bytes * CONTEST_BYTE_PRICE_SCORE
            ),
        }
    }
    scalar_counts = {
        "lf_coefficients": int(section_counts["lf_coefficients"]),
        "hf_decoder_parameters": int(section_counts["hf_decoder_parameters"]),
        "metadata_scalars": int(section_counts["metadata_scalars"]),
    }
    estimates.update(_quant_estimates(scalar_counts))
    return estimates


def _marginal_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    quant_modes: Sequence[str] = ("int8", "int4", "int2"),
) -> list[dict[str, Any]]:
    gates = []
    for mode in quant_modes:
        ordered = sorted(
            rows,
            key=lambda row: row["quantized_payload_estimates"][mode]["payload_bytes"],
        )
        for low, high in pairwise(ordered):
            low_bytes = int(low["quantized_payload_estimates"][mode]["payload_bytes"])
            high_bytes = int(high["quantized_payload_estimates"][mode]["payload_bytes"])
            bytes_added = high_bytes - low_bytes
            if bytes_added <= 0:
                continue
            gates.append(
                {
                    "quant_mode": str(mode),
                    "from_row_id": str(low["row_id"]),
                    "to_row_id": str(high["row_id"]),
                    "from_payload_bytes": low_bytes,
                    "to_payload_bytes": high_bytes,
                    "bytes_added": int(bytes_added),
                    "required_nonrate_score_improvement": float(
                        bytes_added * CONTEST_BYTE_PRICE_SCORE
                    ),
                    "contest_byte_price_score_per_byte": CONTEST_BYTE_PRICE_SCORE,
                    "spend_rule": (
                        "spend_only_if_measured_nonrate_drop_exceeds_required_improvement"
                    ),
                }
            )
    return gates


def _packed_bytes(count: int, *, bits: int, grouped: bool) -> int:
    payload = ceil(int(count) * int(bits) / 8)
    if not grouped or count <= 0:
        return int(payload)
    groups = ceil(int(count) / _PACK_GROUP_SIZE)
    return int(payload + groups * _SCALE_BYTES_PER_GROUP)


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _ordered_unique(items: Iterable[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "NERV_MODELSIZE_LADDER_SCHEMA",
    "SCORER_ONLY_OBJECTIVE_AUTHORITY",
    "build_nerv_modelsize_ladder",
    "hi_nerv_modelsize_config_rows",
    "render_nerv_modelsize_ladder_markdown",
]
