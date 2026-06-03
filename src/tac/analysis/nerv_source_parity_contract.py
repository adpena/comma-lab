# SPDX-License-Identifier: MIT
"""Source-faithful HiNeRV/SNeRV parity contract.

This module is the executable front door for Week-1 compact-carrier work.  It
does not decide whether HiNeRV or SNeRV is good; it decides whether the local
implementation is faithful enough to spend long training budget without
laundering config bugs into method negatives.

The contract is intentionally false-authority: online papers/repos and local
symbol checks can route implementation work, but only receiver-proven
archive-bound candidates plus exact auth eval can promote score.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.analysis.hinerv_official_source_parity_audit import (
    SCHEMA as HINERV_OFFICIAL_SOURCE_AUDIT_SCHEMA,
)
from tac.analysis.hinerv_official_source_parity_audit import (
    summarize_hinerv_official_source_audit,
)
from tac.analysis.snerv_official_source_parity_audit import (
    SCHEMA as SNERV_OFFICIAL_SOURCE_AUDIT_SCHEMA,
)
from tac.analysis.snerv_official_source_parity_audit import (
    summarize_snerv_official_source_audit,
)
from tac.analysis.source_marker_scan import read_python_source_for_marker_scan

SCHEMA = "nerv_source_parity_contract.v1"
AUTHORITY = "false_authority_source_parity_no_score_claim"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}

OFFICIAL_SOURCE_REFS: tuple[dict[str, str], ...] = (
    {
        "source_id": "hi_nerv_official_repo",
        "family": "hi_nerv",
        "url": "https://github.com/hmkx/HiNeRV",
        "why": "S/M/L configs, patch mode, bitstream-q, pruning and quantization pipeline",
    },
    {
        "source_id": "hi_nerv_paper",
        "family": "hi_nerv",
        "url": "https://arxiv.org/abs/2306.09818",
        "why": "hierarchical encodings and compressed bitstream procedure",
    },
    {
        "source_id": "snerv_spectra_preserving_official_repo",
        "family": "snerv",
        "url": "https://github.com/qwertja/SNeRV",
        "why": "DWT LF/HF stack, encoder/decoder strides, MFU, HFR, SNeRV_T and fc_dim controls",
    },
    {
        "source_id": "snerv_spectra_preserving_paper",
        "family": "snerv",
        "url": "https://arxiv.org/abs/2501.01681",
        "why": "spectral split, high-frequency restoration, multi-resolution fusion",
    },
    {
        "source_id": "snerv_scalable_layer_paper",
        "family": "snerv",
        "url": "https://openreview.net/forum?id=ZqN4bnXSSY",
        "why": "separate scalable base/enhancement-layer SNeRV; do not conflate with spectra-preserving DWT SNeRV",
    },
)


@dataclass(frozen=True)
class RequiredSymbol:
    """One local symbol that must exist before a feature can be considered bound."""

    family: str
    feature_id: str
    module: str
    symbol: str
    requirement: str


@dataclass(frozen=True)
class SourceFeature:
    """One official/source-faithfulness feature and its local binding contract."""

    family: str
    feature_id: str
    official_source_id: str
    implementation_target: str
    required_symbols: tuple[RequiredSymbol, ...] = ()
    required_source_markers: tuple[str, ...] = ()
    blocker_if_missing: str = ""
    required_for_long_training: bool = True


def build_nerv_source_parity_contract(
    *,
    repo_root: str | Path,
    families: Iterable[str] = ("hi_nerv", "snerv"),
    hinerv_official_source_audit: Mapping[str, Any] | None = None,
    snerv_official_source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the current source-parity contract for compact NeRV carriers."""

    root = Path(repo_root)
    selected_families = tuple(dict.fromkeys(str(f) for f in families))
    source_audits = _source_audit_rows(
        selected_families,
        hinerv_official_source_audit=hinerv_official_source_audit,
        snerv_official_source_audit=snerv_official_source_audit,
    )
    features = [feature for feature in _source_features() if feature.family in selected_families]
    feature_rows = [_feature_row(root, feature, source_audits=source_audits) for feature in features]
    control_rows = _control_rows(root, selected_families)
    blockers = _ordered_unique(
        [blocker for row in feature_rows for blocker in row["blockers"] if row["required_for_long_training"]]
        + [blocker for row in control_rows for blocker in row["blockers"] if row["required_for_long_training"]]
    )
    nonblocking_gaps = _ordered_unique(
        blocker for row in feature_rows for blocker in row["blockers"] if not row["required_for_long_training"]
    )
    family_rows = [
        _family_summary(family, feature_rows=feature_rows, control_rows=control_rows) for family in selected_families
    ]
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        **FALSE_AUTHORITY,
        "repo_root": root.as_posix(),
        "official_source_refs": list(OFFICIAL_SOURCE_REFS),
        "families": selected_families,
        "source_audits": source_audits,
        "family_rows": family_rows,
        "feature_rows": feature_rows,
        "control_rows": control_rows,
        "analogue_risk_rows": _analogue_risk_rows(selected_families),
        "required_for_long_training_ready": not blockers,
        "blockers": blockers,
        "nonblocking_gaps": nonblocking_gaps,
        "next_actions": _next_actions(blockers),
    }


def render_nerv_source_parity_markdown(report: Mapping[str, Any]) -> str:
    """Render a short operator-facing report."""

    lines = [
        "# NeRV Source-Parity Contract",
        "",
        f"Schema: `{report['schema']}`",
        f"Authority: `{report['authority']}`",
        "",
        "## Family Status",
        "",
        "| family | long-training ready | blockers |",
        "|---|---:|---:|",
    ]
    for row in report.get("family_rows", ()):
        lines.append(
            "| {family} | {ready} | {count} |".format(
                family=row["family"],
                ready="yes" if row["long_training_ready"] else "no",
                count=len(row["blockers"]),
            )
        )
    lines.extend(["", "## Blocking Gaps", ""])
    blockers = report.get("blockers", ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    lines.extend(["", "## Nonblocking Source Gaps", ""])
    nonblocking_gaps = report.get("nonblocking_gaps", ())
    if nonblocking_gaps:
        lines.extend(f"- `{gap}`" for gap in nonblocking_gaps)
    else:
        lines.append("- none")
    lines.extend(["", "## Analogue Risks", ""])
    for row in report.get("analogue_risk_rows", ()):
        lines.append(
            f"- `{row['surface_id']}`: `{row['insufficient_for']}` "
            f"({', '.join(row.get('remaining_blockers') or ())})"
        )
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {action}" for action in report.get("next_actions", ()))
    lines.append("")
    return "\n".join(lines)


def write_nerv_source_parity_contract(
    *,
    repo_root: str | Path,
    output_json: str | Path,
    output_md: str | Path | None = None,
    families: Iterable[str] = ("hi_nerv", "snerv"),
    hinerv_official_source_audit: Mapping[str, Any] | None = None,
    snerv_official_source_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write JSON and optional Markdown source-parity artifacts."""

    report = build_nerv_source_parity_contract(
        repo_root=repo_root,
        families=families,
        hinerv_official_source_audit=hinerv_official_source_audit,
        snerv_official_source_audit=snerv_official_source_audit,
    )
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_nerv_source_parity_markdown(report), encoding="utf-8")
    return report


def _source_features() -> tuple[SourceFeature, ...]:
    return (
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_symbol_map",
            official_source_id="hi_nerv_official_repo",
            implementation_target="architecture/archive/export symbols resolve",
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_symbol_map",
                    "tac.substrates.hi_nerv.architecture",
                    "HinervConfig",
                    "config surface",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_symbol_map",
                    "tac.substrates.hi_nerv.architecture",
                    "HinervSubstrate",
                    "torch forward surface",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_symbol_map",
                    "tac.substrates.hi_nerv.archive",
                    "pack_archive",
                    "receiver packet packer",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_symbol_map",
                    "tac.substrates.hi_nerv.archive",
                    "parse_archive",
                    "receiver packet parser",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_symbol_map",
                    "tac.substrates.hi_nerv.archive_candidate",
                    "export_hi_nerv_mlx_archive",
                    "archive-bound export",
                ),
            ),
            blocker_if_missing="hi_nerv_required_symbol_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_legacy_phase_a_false_authority_guard",
            official_source_id="hi_nerv_official_repo",
            implementation_target=(
                "legacy tac.hinerv_as_renderer path is explicitly research-only "
                "and points production work at tac.substrates.hi_nerv"
            ),
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_legacy_phase_a_false_authority_guard",
                    "tac.hinerv_as_renderer",
                    "LEGACY_HINERV_PHASE_A_BLOCKER",
                    "legacy Phase-A blocker constant",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_legacy_phase_a_false_authority_guard",
                    "tac.hinerv_as_renderer",
                    "legacy_hinerv_phase_a_false_authority",
                    "fail-closed false-authority payload builder",
                ),
            ),
            blocker_if_missing="hi_nerv_legacy_phase_a_false_authority_guard_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_mlx_tiny_forward_parity",
            official_source_id="hi_nerv_official_repo",
            implementation_target="MLX forward and PyTorch-layout export parity",
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_mlx_tiny_forward_parity",
                    "tac.substrates.hi_nerv.mlx_renderer",
                    "HinervSubstrateMLX",
                    "MLX renderer",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_mlx_tiny_forward_parity",
                    "tac.substrates.hi_nerv.mlx_renderer",
                    "_bilinear_resize_nhwc",
                    "resolution path",
                ),
            ),
            blocker_if_missing="hi_nerv_mlx_tiny_forward_parity_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_generic_resolution_path",
            official_source_id="hi_nerv_official_repo",
            implementation_target="arbitrary patch/resolution resize matches torch align_corners=False",
            required_source_markers=(
                "bilinear_resize_nhwc",
                "align_corners=False",
            ),
            blocker_if_missing="hi_nerv_generic_resolution_path_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_official_feature_grid_convnext_trilinear",
            official_source_id="hi_nerv_official_repo",
            implementation_target=(
                "official hierarchical feature grids, ConvNeXt blocks, and "
                "trilinear multi-scale upsampling are bound or explicitly forked"
            ),
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_feature_grid_convnext_trilinear",
                    "tac.substrates.hi_nerv.architecture",
                    "HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF",
                    "receiver-visible proof constant",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_feature_grid_convnext_trilinear",
                    "tac.substrates.hi_nerv.architecture",
                    "HierarchicalFeatureGrid",
                    "temporal-local feature grid",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_feature_grid_convnext_trilinear",
                    "tac.substrates.hi_nerv.architecture",
                    "ConvNeXtBlock",
                    "ConvNeXt refinement block",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_feature_grid_convnext_trilinear",
                    "tac.substrates.hi_nerv.official_grid",
                    "OfficialGridTrilinear3D",
                    "official temporal-only GridTrilinear3D",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_feature_grid_convnext_trilinear",
                    "tac.substrates.hi_nerv.official_grid",
                    "official_grid_trilinear3d_forward",
                    "official GridTrilinear3D functional replay",
                ),
            ),
            blocker_if_missing=("hi_nerv_official_feature_grid_convnext_trilinear_missing"),
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_official_forward_parity",
            official_source_id="hi_nerv_official_repo",
            implementation_target=(
                "official HiNeRV torch forward and local portable/MLX forward "
                "produce matching hashed outputs under a shared input and "
                "weight manifest"
            ),
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_forward_parity",
                    "tac.substrates.hi_nerv.architecture",
                    "HINERV_OFFICIAL_FULL_FORWARD_PARITY_PROOF",
                    "full official/local forward parity proof",
                ),
            ),
            blocker_if_missing="hinerv_official_forward_parity_missing",
            required_for_long_training=False,
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_official_patch_index_path",
            official_source_id="hi_nerv_official_repo",
            implementation_target=(
                "official patch dataset/index geometry is bound as portable "
                "NumPy receiver-side primitives before source-faithful training"
            ),
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_patch_index_path",
                    "tac.substrates.hi_nerv.official_patch",
                    "HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF",
                    "official patch/index proof constant",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_patch_index_path",
                    "tac.substrates.hi_nerv.official_patch",
                    "official_video_to_patch",
                    "official video-to-patch order",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_patch_index_path",
                    "tac.substrates.hi_nerv.official_patch",
                    "official_patch_to_video",
                    "official patch-to-video inverse",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_patch_index_path",
                    "tac.substrates.hi_nerv.official_patch",
                    "official_vidx_to_pidx",
                    "official child patch index expansion",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_patch_index_path",
                    "tac.substrates.hi_nerv.official_patch",
                    "official_compute_pixel_idx_3d",
                    "official padded 3D pixel index projection",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_official_patch_index_path",
                    "tac.substrates.hi_nerv.official_patch",
                    "official_flat_patch_index_to_thw",
                    "official VideoDataset flat index mapping",
                ),
            ),
            blocker_if_missing="hi_nerv_official_patch_index_path_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
            official_source_id="hi_nerv_official_repo",
            implementation_target=(
                "pruning, QuantNoise/QAT preparation, and decoder bitstream "
                "roundtrip are receiver-visible instead of prose-only"
            ),
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
                    "tac.substrates.hi_nerv.bitstream",
                    "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
                    "behavior-backed proof constant",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
                    "tac.substrates.hi_nerv.bitstream",
                    "apply_decoder_pruning",
                    "global magnitude pruning",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
                    "tac.substrates.hi_nerv.bitstream",
                    "apply_decoder_quant_noise",
                    "deterministic QuantNoise",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
                    "tac.substrates.hi_nerv.bitstream",
                    "measure_hi_nerv_decoder_bitstream_roundtrip",
                    "receiver codec measurement",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
                    "tac.substrates.hi_nerv.bitstream",
                    "select_hi_nerv_bitstream_codec_by_scorer_waterfill",
                    "score-priced codec waterfill selector",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_prune_quantnoise_receiver_bitstream_pipeline",
                    "tac.substrates.hi_nerv.archive",
                    "repack_archive_decoder_codec",
                    "latent-preserving decoder codec repack",
                ),
            ),
            blocker_if_missing="hi_nerv_prune_quantnoise_receiver_bitstream_pipeline_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_official_torchac_entropy_coder_parity",
            official_source_id="hi_nerv_official_repo",
            implementation_target=(
                "exact official torchac arithmetic entropy coder is vendored or "
                "explicitly superseded by measured same-axis receiver codec evidence"
            ),
            required_source_markers=("torchac",),
            blocker_if_missing="hi_nerv_official_torchac_entropy_coder_missing",
            required_for_long_training=False,
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_bitstream_quantization_roundtrip",
            official_source_id="hi_nerv_official_repo",
            implementation_target="receiver-visible compressed decoder and latent bitstream",
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_bitstream_quantization_roundtrip",
                    "tac.substrates.hi_nerv.archive",
                    "pack_archive",
                    "packet packer",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_bitstream_quantization_roundtrip",
                    "tac.substrates._shared.decoder_state_codec",
                    "serialize_decoder_state_dict",
                    "decoder codec",
                ),
            ),
            blocker_if_missing="hi_nerv_bitstream_roundtrip_missing",
        ),
        SourceFeature(
            family="hi_nerv",
            feature_id="hi_nerv_decoder_weight_saliency_binding",
            official_source_id="hi_nerv_paper",
            implementation_target="score saliency pushes into decoder/latent fit before export",
            required_symbols=(
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_decoder_weight_saliency_binding",
                    "tac.analysis.hinerv_latent_linf_allocation",
                    "decoder_jacobian_vjp",
                    "dense VJP adjoint",
                ),
                RequiredSymbol(
                    "hi_nerv",
                    "hi_nerv_decoder_weight_saliency_binding",
                    "tac.analysis.hinerv_latent_linf_allocation",
                    "allocate_linf_latent_steps",
                    "latent allocator",
                ),
            ),
            blocker_if_missing="hi_nerv_decoder_weight_saliency_binding_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_symbol_map",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target="DWT, LF/HF carrier, archive, and receiver symbols resolve",
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_symbol_map",
                    "tac.substrates.snerv_inverse_steg_carrier.dwt",
                    "dwt2_multilevel",
                    "DWT analysis",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_symbol_map",
                    "tac.substrates.snerv_inverse_steg_carrier.dwt",
                    "idwt2_multilevel",
                    "DWT synthesis",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_symbol_map",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "HfGenerationDecoder",
                    "HF generator",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_symbol_map",
                    "tac.substrates.snerv_inverse_steg_carrier.archive",
                    "pack_snerv_archive",
                    "archive packer",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_symbol_map",
                    "tac.substrates.snerv_inverse_steg_carrier.archive",
                    "decode_snerv_archive_frames",
                    "receiver decode",
                ),
            ),
            blocker_if_missing="snerv_required_symbol_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_receiver_bitstream_roundtrip",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target="SNAR1 packet consumes LF, decoder, step maps into frames",
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_bitstream_roundtrip",
                    "tac.substrates.snerv_inverse_steg_carrier.archive",
                    "unpack_snerv_archive",
                    "archive parser",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_bitstream_roundtrip",
                    "tac.substrates.snerv_inverse_steg_carrier.receiver_proof",
                    "build_snerv_receiver_archive_proof",
                    "receiver proof",
                ),
            ),
            blocker_if_missing="snerv_receiver_bitstream_roundtrip_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_receiver_safe_mfu_hfr_temporal_adapter",
            official_source_id="snerv_spectra_preserving_paper",
            implementation_target=(
                "local spectra-preserving MFU/HFR/SNeRV_T fork is executable, "
                "receiver-closed, and byte-priced before long training"
            ),
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_safe_mfu_hfr_temporal_adapter",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "MultiResolutionFusionUnit",
                    "local receiver-safe MFU adapter",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_safe_mfu_hfr_temporal_adapter",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "HighFrequencyRestorer",
                    "local receiver-safe HFR adapter",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_safe_mfu_hfr_temporal_adapter",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SnervTemporalExtension",
                    "local SNeRV_T analysis utility",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_safe_mfu_hfr_temporal_adapter",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
                    "local receiver-safe adapter proof constant",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_safe_mfu_hfr_temporal_adapter",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
                    "official SNeRV_T Haar/DWT1D temporal primitive proof",
                ),
            ),
            blocker_if_missing="snerv_receiver_safe_mfu_hfr_temporal_adapter_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_official_tub_haar_dwt1d_temporal_primitive",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target=(
                "official SNeRV_T Haar DWT1D lowpass/2 temporal primitive is "
                "receiver-visible and numerically tested; this is narrower than "
                "full official MFU/HFR/TUB source-forward parity"
            ),
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_official_tub_haar_dwt1d_temporal_primitive",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
                    "official SNeRV_T Haar/DWT1D lowpass proof constant",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_official_tub_haar_dwt1d_temporal_primitive",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SnervTemporalExtension.official_haar_dwt1d_lowpass_features",
                    "receiver-visible official SNeRV_T temporal primitive",
                ),
            ),
            required_source_markers=(
                "official_haar_dwt1d_lowpass_features",
                "1.0 / (2.0 * np.sqrt(2.0))",
            ),
            blocker_if_missing="snerv_official_tub_haar_dwt1d_temporal_primitive_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_official_mfu_hfr_tub_numeric_primitives",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target=(
                "portable NumPy/MLX official MFU/HFR/TUB numeric primitives are "
                "bound as source-graph components; this does not prove full "
                "official source-forward replay"
            ),
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_numeric_primitives",
                    "tac.substrates.snerv_inverse_steg_carrier.official_mfu",
                    "OfficialSnervMfu",
                    "official MFU executable graph primitive",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_numeric_primitives",
                    "tac.substrates.snerv_inverse_steg_carrier.official_mfu",
                    "conv_transpose2d_nchw",
                    "official MFU ConvTranspose2d NumPy reference",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_numeric_primitives",
                    "tac.substrates.snerv_inverse_steg_carrier.official_hfr",
                    "OfficialHfrHeads",
                    "official HFR learned-head layout primitive",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_numeric_primitives",
                    "tac.substrates.snerv_inverse_steg_carrier.official_hfr",
                    "conv2d_nchw_mlx",
                    "official HFR MLX conv reference",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_numeric_primitives",
                    "tac.substrates.snerv_inverse_steg_carrier.official_tub",
                    "prepare_official_tub_graph_inputs",
                    "official SNeRV_T TUB input preparation",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_numeric_primitives",
                    "tac.substrates.snerv_inverse_steg_carrier.official_tub",
                    "official_output2_fusion_shape",
                    "official SNeRV_T output_2 fusion shape primitive",
                ),
            ),
            blocker_if_missing="snerv_official_mfu_hfr_tub_numeric_primitives_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_official_mfu_hfr_tub_parity",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target=(
                "official spectra-preserving MFU/HFR/TUB source-forward parity "
                "is proven or explicitly superseded by same-axis receiver-closed "
                "evidence"
            ),
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_official_mfu_hfr_tub_parity",
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF",
                    "official MFU/HFR/TUB parity proof",
                ),
            ),
            blocker_if_missing="snerv_official_mfu_hfr_tub_parity_missing",
            required_for_long_training=False,
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_scorer_loop_decoder_qat",
            official_source_id="snerv_spectra_preserving_paper",
            implementation_target="score-aware decoder-weight QAT before packet export",
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_scorer_loop_decoder_qat",
                    "tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat",
                    "run_snerv_scorer_loop_decoder_qat",
                    "full trainer entrypoint",
                ),
            ),
            blocker_if_missing="snerv_scorer_loop_decoder_qat_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_qat_receiver_codec_pricing",
            official_source_id="snerv_spectra_preserving_paper",
            implementation_target="QAT objective prices the same decoder codec consumed by receiver",
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_qat_receiver_codec_pricing",
                    "tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat",
                    "SNERV_QAT_RECEIVER_CODEC_PRICING_PROOF",
                    "receiver-priced QAT proof constant",
                ),
            ),
            blocker_if_missing="snerv_qat_receiver_codec_pricing_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_official_haar_mode",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target="official Haar DWT/IDWT path or explicit forked-adapter label",
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_official_haar_mode",
                    "tac.substrates.snerv_inverse_steg_carrier.dwt",
                    "OFFICIAL_SNERV_HAAR_MODE_PROOF",
                    "official Haar mode proof constant",
                ),
            ),
            blocker_if_missing="snerv_official_haar_mode_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_receiver_dependency_custody",
            official_source_id="snerv_spectra_preserving_official_repo",
            implementation_target="receiver DWT runtime is pure NumPy or contest-proven dependency closure",
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_receiver_dependency_custody",
                    "tac.substrates.snerv_inverse_steg_carrier.dwt",
                    "SNERV_RECEIVER_DWT_RUNTIME_CUSTODY_PROOF",
                    "receiver dependency custody proof constant",
                ),
            ),
            blocker_if_missing="snerv_receiver_dependency_custody_missing",
        ),
        SourceFeature(
            family="snerv",
            feature_id="snerv_scalable_layer_admission_policy",
            official_source_id="snerv_scalable_layer_paper",
            implementation_target=(
                "base/enhancement layer bytes are priced by contest waterline "
                "before scalable-layer is promoted to a separate lane"
            ),
            required_symbols=(
                RequiredSymbol(
                    "snerv",
                    "snerv_scalable_layer_admission_policy",
                    "tac.analysis.snerv_scalable_layer_admission",
                    "SNERV_SCALABLE_LAYER_ADMISSION_PROOF",
                    "behavior-backed proof constant",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_scalable_layer_admission_policy",
                    "tac.analysis.snerv_scalable_layer_admission",
                    "build_snerv_scalable_layer_admission_report",
                    "real SNAR1 layer admission profiler",
                ),
                RequiredSymbol(
                    "snerv",
                    "snerv_scalable_layer_admission_policy",
                    "tac.analysis.snerv_scalable_layer_admission",
                    "write_snerv_scalable_layer_admission_report",
                    "durable report writer",
                ),
            ),
            blocker_if_missing="snerv_scalable_layer_admission_policy_missing",
            required_for_long_training=False,
        ),
    )


def _analogue_risk_rows(families: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if "snerv" in families:
        rows.extend(
            [
                _analogue_risk_row(
                    family="snerv",
                    surface_id="snerv_receiver_safe_mfu_hfr_temporal_adapter",
                    analogue_surface=(
                        "receiver-safe NumPy MFU/HFR/SNeRV_T adapter in carrier.py"
                    ),
                    insufficient_for="official_spectra_preserving_snerv_source_forward",
                    why=(
                        "the local adapter is executable and receiver-safe, but "
                        "does not consume official MFU/HFR/TUB weights through "
                        "the upstream neural graph"
                    ),
                    remaining_blockers=(
                        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
                        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
                    ),
                ),
                _analogue_risk_row(
                    family="snerv",
                    surface_id="snerv_official_mfu_hfr_tub_numeric_primitives",
                    analogue_surface=(
                        "portable official MFU/HFR/TUB numeric kernels and shape contracts"
                    ),
                    insufficient_for="byte_closed_official_snerv_export_runtime",
                    why=(
                        "numeric primitives prove local algebra, not a trained "
                        "official neural payload, receiver grammar, or source "
                        "forward replay"
                    ),
                    remaining_blockers=(
                        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
                        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
                    ),
                ),
                _analogue_risk_row(
                    family="snerv",
                    surface_id="snerv_local_modelsize_analogue",
                    analogue_surface=(
                        "fc_dim/emb_size/patch_radius/MFU/HFR receiver-visible "
                        "capacity controls"
                    ),
                    insufficient_for="official_snerv_modelsize_authority",
                    why=(
                        "local capacity changes bytes and decoded frames, but "
                        "official --modelsize authority requires the upstream "
                        "stride stack and neural graph to consume the solved fc_dim"
                    ),
                    remaining_blockers=(
                        "snerv_official_stride_stack_parity_missing",
                        "snerv_official_mfu_hfr_tub_full_stack_source_forward_parity_missing",
                        "snerv_measured_fc_dim_modelsize_ladder_missing",
                    ),
                ),
            ]
        )
    if "hi_nerv" in families:
        rows.extend(
            [
                _analogue_risk_row(
                    family="hi_nerv",
                    surface_id="hi_nerv_local_target_modelsize",
                    analogue_surface=(
                        "local target modelsize and capacity routing for archive ladders"
                    ),
                    insufficient_for="official_hinerv_config_family_authority",
                    why=(
                        "local target capacity can price bytes, but official "
                        "HiNeRV authority requires the upstream config family, "
                        "hierarchical feature grid, and same-runtime bitstream replay"
                    ),
                    remaining_blockers=(
                        "hi_nerv_official_symbol_parity_map_missing",
                        "hi_nerv_tiny_forward_parity_against_oss_missing",
                        "hi_nerv_measured_modelsize_budget_ladder_missing",
                    ),
                ),
                _analogue_risk_row(
                    family="hi_nerv",
                    surface_id="hi_nerv_mlx_backend_drift",
                    analogue_surface="MLX/Metal local archive backend drift rows",
                    insufficient_for="contest_cpu_cuda_auth_eval_authority",
                    why=(
                        "MLX is a high-value development accelerator, but drift "
                        "rows are still local false-authority until paired contest "
                        "CPU/CUDA replay closes"
                    ),
                    remaining_blockers=(
                        "contest_cpu_cuda_exact_eval_not_executed",
                        "receiver_closed_full600_archive_runtime_missing",
                    ),
                ),
            ]
        )
    rows.append(
        _analogue_risk_row(
            family="cross_stack",
            surface_id="pr95_hnerv_mlx_control_arm",
            analogue_surface="PR95-inspired HNeRV/MLX control-arm surfaces",
            insufficient_for="pr95_source_faithful_control_reproduction",
            why=(
                "PR95 is the same-axis control to beat; MLX or HNeRV-inspired "
                "surfaces remain analogues until PR95's source/runtime/export "
                "contract is reproduced or explicitly superseded by exact evidence"
            ),
            remaining_blockers=(
                "pr95_hnerv_mlx_archive_export_control_arm_not_pr95_faithful_reproduction",
                "paired_contest_cpu_cuda_replay_missing",
            ),
        )
    )
    return tuple(rows)


def _analogue_risk_row(
    *,
    family: str,
    surface_id: str,
    analogue_surface: str,
    insufficient_for: str,
    why: str,
    remaining_blockers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "family": family,
        "surface_id": surface_id,
        "analogue_surface": analogue_surface,
        "insufficient_for": insufficient_for,
        "why": why,
        "remaining_blockers": remaining_blockers,
        **FALSE_AUTHORITY,
    }


def _control_rows(root: Path, families: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "hi_nerv" in families:
        rows.extend(
            (
                _source_marker_control_row(
                    root,
                    family="hi_nerv",
                    control_id="hi_nerv_modelsize_config_ladder",
                    source_files=("src/tac/substrates/hi_nerv/architecture.py",),
                    markers=("latent_dim_coarse", "embed_dim", "decoder_channels"),
                    blocker="hi_nerv_measured_modelsize_ladder_missing",
                    required_for_long_training=True,
                ),
                _source_marker_control_row(
                    root,
                    family="hi_nerv",
                    control_id="hi_nerv_bitstream_q_control",
                    source_files=(
                        "src/tac/substrates/hi_nerv/archive.py",
                        "src/tac/substrates/hi_nerv/archive_candidate.py",
                    ),
                    markers=("decoder_codec", "int8_mixed"),
                    blocker="hi_nerv_bitstream_q_receiver_sweep_missing",
                    required_for_long_training=True,
                ),
            )
        )
    if "snerv" in families:
        rows.extend(
            (
                _source_marker_control_row(
                    root,
                    family="snerv",
                    control_id="snerv_fc_dim_modelsize_control",
                    source_files=("src/tac/substrates/snerv_inverse_steg_carrier",),
                    markers=("fc_dim", "emb_size"),
                    blocker="snerv_fc_dim_modelsize_control_missing",
                    required_for_long_training=True,
                ),
                _source_marker_control_row(
                    root,
                    family="snerv",
                    control_id="snerv_lf_stepmap_and_intN_control",
                    source_files=("src/tac/substrates/snerv_inverse_steg_carrier/archive.py",),
                    markers=("SNERV_LF_PAYLOAD_INTN_CODEC_PROOF",),
                    blocker="snerv_lf_quant_intn_codec_missing",
                    required_for_long_training=True,
                ),
            )
        )
    return rows


def _source_audit_rows(
    families: tuple[str, ...],
    *,
    hinerv_official_source_audit: Mapping[str, Any] | None,
    snerv_official_source_audit: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    if "hi_nerv" in families and isinstance(hinerv_official_source_audit, Mapping):
        schema = hinerv_official_source_audit.get("schema")
        if schema == HINERV_OFFICIAL_SOURCE_AUDIT_SCHEMA:
            summary = summarize_hinerv_official_source_audit(
                hinerv_official_source_audit
            )
            rows.append(
                {
                    "family": "hi_nerv",
                    "feature_id": "hi_nerv_official_forward_parity",
                    "audit_kind": "official_source_marker_and_numeric_forward_audit",
                    **summary,
                }
            )
        else:
            rows.append(
                {
                    "family": "hi_nerv",
                    "feature_id": "hi_nerv_official_forward_parity",
                    "audit_kind": "official_source_marker_and_numeric_forward_audit",
                    "schema": schema,
                    "authority": AUTHORITY,
                    "official_source_markers_present": False,
                    "local_receiver_bindings_present": False,
                    "official_forward_parity_proven": False,
                    "blockers": ["hinerv_official_source_audit_schema_invalid"],
                    **FALSE_AUTHORITY,
                }
            )
    if "snerv" in families and isinstance(snerv_official_source_audit, Mapping):
        schema = snerv_official_source_audit.get("schema")
        if schema == SNERV_OFFICIAL_SOURCE_AUDIT_SCHEMA:
            summary = summarize_snerv_official_source_audit(snerv_official_source_audit)
            rows.append(
                {
                    "family": "snerv",
                    "feature_id": "snerv_official_mfu_hfr_tub_parity",
                    "audit_kind": "official_source_marker_and_local_proof_audit",
                    **summary,
                }
            )
        else:
            rows.append(
                {
                    "family": "snerv",
                    "feature_id": "snerv_official_mfu_hfr_tub_parity",
                    "audit_kind": "official_source_marker_and_local_proof_audit",
                    "schema": schema,
                    "authority": AUTHORITY,
                    "official_source_markers_present": False,
                    "local_receiver_safe_adapter_present": False,
                    "official_mfu_hfr_tub_parity_proven": False,
                    "blockers": ["snerv_official_source_audit_schema_invalid"],
                    **FALSE_AUTHORITY,
                }
            )
    return tuple(rows)


def _feature_row(
    root: Path,
    feature: SourceFeature,
    *,
    source_audits: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    symbol_rows = [_symbol_row(symbol) for symbol in feature.required_symbols]
    marker_rows = [_source_marker_row(root, marker, feature.family) for marker in feature.required_source_markers]
    feature_source_audits = tuple(
        row
        for row in source_audits
        if row.get("family") == feature.family and row.get("feature_id") == feature.feature_id
    )
    blockers: list[str] = []
    if any(row["status"] != "present" for row in symbol_rows):
        blockers.append(feature.blocker_if_missing)
    if any(row["status"] != "present" for row in marker_rows):
        blockers.append(feature.blocker_if_missing)
    status = "implemented_or_bound" if not blockers else "missing_or_partial"
    return {
        "family": feature.family,
        "feature_id": feature.feature_id,
        "official_source_id": feature.official_source_id,
        "implementation_target": feature.implementation_target,
        "status": status,
        "required_for_long_training": feature.required_for_long_training,
        "symbol_rows": symbol_rows,
        "source_marker_rows": marker_rows,
        "source_audit_rows": feature_source_audits,
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _source_marker_control_row(
    root: Path,
    *,
    family: str,
    control_id: str,
    source_files: tuple[str, ...],
    markers: tuple[str, ...],
    blocker: str,
    required_for_long_training: bool,
) -> dict[str, Any]:
    text = "\n".join(_read_source_for_marker_scan(root / source_file) for source_file in source_files)
    missing = [marker for marker in markers if marker not in text]
    return {
        "family": family,
        "control_id": control_id,
        "markers": markers,
        "source_files": source_files,
        "status": "implemented_or_declared" if not missing else "missing_or_partial",
        "missing_markers": missing,
        "required_for_long_training": required_for_long_training,
        "blockers": () if not missing else (blocker,),
        **FALSE_AUTHORITY,
    }


def _source_marker_row(root: Path, marker: str, family: str) -> dict[str, Any]:
    source_dirs = {
        "hi_nerv": ("src/tac/substrates/hi_nerv", "src/tac/analysis"),
        "snerv": ("src/tac/substrates/snerv_inverse_steg_carrier", "src/tac/analysis"),
    }
    text = "\n".join(_read_source_for_marker_scan(root / source_dir) for source_dir in source_dirs[family])
    return {
        "marker": marker,
        "status": "present" if marker in text else "missing",
    }


def _symbol_row(symbol: RequiredSymbol) -> dict[str, Any]:
    status = "present"
    error = None
    try:
        module = importlib.import_module(symbol.module)
        target: Any = module
        for part in symbol.symbol.split("."):
            target = getattr(target, part)
    except Exception as exc:  # pragma: no cover - error text covered by callers.
        status = "missing"
        error = repr(exc)
    return {
        **asdict(symbol),
        "status": status,
        "error": error,
    }


def _family_summary(
    family: str,
    *,
    feature_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers = _ordered_unique(
        [
            blocker
            for row in (*feature_rows, *control_rows)
            if row["family"] == family and row["required_for_long_training"]
            for blocker in row["blockers"]
        ]
    )
    return {
        "family": family,
        "long_training_ready": not blockers,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _read_source(path: Path) -> str:
    if path.is_dir():
        return "\n".join(
            child.read_text(encoding="utf-8", errors="replace")
            for child in sorted(path.rglob("*.py"))
            if child.name != "nerv_source_parity_contract.py"
        )
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _read_source_for_marker_scan(path: Path) -> str:
    return read_python_source_for_marker_scan(
        path,
        exclude_names=("nerv_source_parity_contract.py",),
    )


def _next_actions(blockers: tuple[str, ...]) -> tuple[str, ...]:
    actions: list[str] = []
    if any("hi_nerv" in blocker for blocker in blockers):
        actions.append("close HiNeRV official config/modelsize ladder and receiver bitstream replay before long run")
    if any("snerv" in blocker for blocker in blockers):
        actions.append("close SNeRV fc_dim/MFU/HFR or explicitly block source-faithful SNeRV before long run")
    if not actions:
        actions.append("allow compact-carrier long-training smoke with cache gate and packet spine")
    return tuple(actions)


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)
