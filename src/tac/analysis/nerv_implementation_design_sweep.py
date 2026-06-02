# SPDX-License-Identifier: MIT
"""Source/design sweep for the top-priority SNeRV and HiNeRV stacks.

This is a fail-closed audit, not a promotion artifact. It checks official OSS
snapshots, local implementation surfaces, design/control hooks, and promotion
proof prerequisites so incomplete or arbitrary NeRV sketches cannot masquerade
as production-hardened stacks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import (
    FALSE_AUTHORITY as HPRC_FALSE_AUTHORITY,
)

SCHEMA = "nerv_implementation_design_sweep.v1"
AXIS_TAG = "[planning/control]"
FALSE_AUTHORITY = {
    **HPRC_FALSE_AUTHORITY,
    "frontier_score_claim": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
}
RELATED_OMX_MEMO_EXTENSIONS = {".json", ".jsonl", ".md"}
RELATED_OMX_MEMO_TERMS = (
    "snerv",
    "hinerv",
    "hi_nerv",
    "hnerv",
    "sr_nerv",
    "srnerv",
    "sr-neerv",
    "rnerv",
    "ffnerv",
    "boostnerv",
    "boost_nerv",
    "e_nerv",
    "enerv",
    "nerv",
    "pr95",
)

STACK_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "snerv": {
        "official_repo_name": "SNeRV",
        "official_required_files": [
            "train_snerv.py",
            "train_snerv_t.py",
            "model/snerv.py",
            "model/snerv_t.py",
            "model/layers.py",
        ],
        "official_feature_tokens": {
            "modelsize_fc_dim_solver": ["--modelsize", "fc_dim", "np.roots"],
            "haar_dwt": ["pytorch_wavelets", "DWT", "haar"],
            "mfu_hfr_modules": ["MFU", "HFR"],
            "temporal_extension": ["SNeRV_T", "emb_size"],
            "quant_payload": [
                "quant_model_bit",
                "quant_embed_bit",
                "quant_embed2_bit",
                "quant_vid.pth",
            ],
        },
        "local_surfaces": [
            "src/tac/substrates/snerv_inverse_steg_carrier/carrier.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/archive.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/allocation.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py",
            "src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py",
            "src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py",
            "src/tac/analysis/snerv_pose_guarded_decoder_gate.py",
            "src/tac/analysis/snerv_decoder_mode_assignment_probe.py",
            "src/tac/analysis/nerv_modelsize_archive_curve.py",
        ],
        "local_feature_tokens": {
            "source_faithful_adapter_gate": [
                "SNERV_SPECTRA_PRESERVING_ADAPTER",
                "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
                "MultiResolutionFusionUnit",
                "HighFrequencyRestorer",
                "SnervTemporalExtension",
            ],
            "receiver_proof": ["receiver", "archive", "sha256", "bytes"],
            "mixed_decoder_modes": ["int2", "int4", "int8", "fp16", "zero"],
            "pose_guarded_qat": ["PoseNet", "pose", "guard", "scorer_loop"],
            "waterfill_allocator": ["waterfill", "linf", "allocation"],
        },
        "promotion_proofs": [
            "official_forward_parity",
            "source_faithful_contest_adapter",
            "mixed_precision_receiver_byte_accounting",
            "pair_robust_scorer_loop_QAT",
            "full600_byte_closed_receiver_proof",
            "paired_contest_CPU_CUDA_pass",
        ],
    },
    "hinerv": {
        "official_repo_name": "HiNeRV",
        "official_required_files": [
            "hinerv_main.py",
            "hinerv_compress.py",
            "models/hinerv.py",
            "models/encoding.py",
            "models/patch_utils.py",
            "models/upsample.py",
            "compression/quant_utils.py",
            "compression/prune_utils.py",
            "compression/codec_utils.py",
        ],
        "official_feature_tokens": {
            "hierarchical_grid": ["GridEncoding", "grid_level", "grid_level_scale"],
            "temporal_local_grid": ["TemporalLocalGridEncoding", "temp_local_grid"],
            "patch_frame_mode": ["video_to_patch", "patch_to_video", "patch_mode"],
            "3d_upsampling": ["trilinear", "upsample", "out_patch_size"],
            "prune_quant_bitstream": [
                "--prune-ratio",
                "--quant-level",
                "--quant-noise",
                "--quant-ste",
                "compress_and_save_model",
            ],
        },
        "local_surfaces": [
            "src/tac/substrates/hi_nerv/architecture.py",
            "src/tac/substrates/hi_nerv/archive.py",
            "src/tac/substrates/hi_nerv/bitstream.py",
            "src/tac/substrates/_shared/compact_decoder_codec_sweep.py",
            "src/tac/substrates/hi_nerv/score_aware_loss.py",
            "src/tac/analysis/nerv_modelsize_archive_curve.py",
            "src/tac/substrates/_shared/mlx_score_aware_full_main.py",
            "src/tac/substrates/_shared/mlx_score_aware/coder_qat.py",
            "src/tac/master_gradient_mlx_pipeline.py",
        ],
        "local_feature_tokens": {
            "l0_sketch_gate": ["sketch", "l0", "not_source_faithful"],
            "score_aware_teacher": ["SegNet", "PoseNet", "score"],
            "coder_qat": [
                "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
                "apply_decoder_pruning",
                "apply_decoder_quant_noise",
                "measure_hi_nerv_decoder_bitstream_roundtrip",
                "select_hi_nerv_bitstream_codec_by_scorer_waterfill",
            ],
            "dense_vjp_allocator_inputs": ["master_gradient", "VJP", "linf"],
            "receiver_archive": ["archive", "receiver", "bytes"],
        },
        "promotion_proofs": [
            "official_forward_parity",
            "patch_mode_frame_mode_parity",
            "prune_quant_codec_roundtrip",
            "real_teacher_scorer_training",
            "full600_byte_closed_receiver_proof",
            "paired_contest_CPU_CUDA_pass",
        ],
    },
}


class NervImplementationDesignSweepError(ValueError):
    """Raised when the implementation/design sweep cannot run safely."""


def build_nerv_implementation_design_sweep(
    *,
    repo_root: str | Path,
    oss_audit_root: str | Path | None = None,
    generated_utc: str | None = None,
    proof_refs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed implementation/design sweep for SNeRV and HiNeRV."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise NervImplementationDesignSweepError(
            f"repo_root is not a directory: {root}"
        )

    generated = generated_utc or datetime.now(UTC).isoformat()
    oss_root = Path(oss_audit_root).resolve() if oss_audit_root else None
    refs = {key: list(value) for key, value in (proof_refs or {}).items()}
    stack_rows = [
        _sweep_stack(root, oss_root, stack_id, req, refs.get(stack_id, ()))
        for stack_id, req in STACK_REQUIREMENTS.items()
    ]
    memo_refs = _related_omx_design_memo_refs(root)
    blockers = _unique(
        blocker for row in stack_rows for blocker in row["production_blockers"]
    )
    blockers.extend(
        [
            "PR95_same_axis_control_replay_required_before_beat_claim",
            "PR101_and_Z5_nonterminal_block_new_exact_or_full_video_cuda",
        ]
    )

    return {
        "schema": SCHEMA,
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "verdict": (
            "GO_IMPLEMENTATION_TRIAGE__NO_GO_FULLY_OPTIMIZED_OR_PRODUCTION_"
            "HARDENED_UNTIL_ALL_STACK_GATES_PASS"
        ),
        "top_priority_carriers": ["snerv", "hinerv"],
        "stack_sweeps": stack_rows,
        "related_omx_design_memo_count": len(memo_refs),
        "related_omx_design_memo_refs": memo_refs,
        "memo_scan_policy": {
            "root": ".omx/research",
            "scope": "top_level_files_only",
            "extensions": sorted(RELATED_OMX_MEMO_EXTENSIONS),
            "filename_terms": list(RELATED_OMX_MEMO_TERMS),
            "reason": (
                "reference design/research ledgers deterministically without "
                "misclassifying nested queue outputs as design memos"
            ),
        },
        "anti_arbitrariness_policy": {
            "required": [
                "every architecture knob traces to official OSS, PR95 control, or a measured contest hook",
                "every rate knob has receiver-decoded byte accounting",
                "every scorer knob is tagged by evidence axis and component",
                "every helper extension reuses an existing hook unless math/contract differs",
                "bad SNeRV/HiNeRV advisory scores are implementation/config bug signals until source parity closes",
            ],
            "forbidden": [
                "minimal sketch called HiNeRV",
                "simplified DWT adapter called production SNeRV",
                "fake quant with fp32 receiver payload called rate optimization",
                "zero-parameter SR interpolation called SR-NeRV",
                "local MLX advisory score called PR95 beat evidence",
            ],
        },
        "global_next_checks": [
            "run_official_forward_shape_parity_smoke_for_SNeRV_and_HiNeRV",
            "measure_modelsize_to_archive_bytes_curve_on_trained_tiny_snapshots",
            "bind_receiver_byte_grammar_for_mixed_precision_decoder_payloads",
            "wire_real_teacher_SegNet_PoseNet_loss_with_eval_roundtrip_STE",
            "run_PR95_same_axis_control_replay_before_any_beat_claim",
        ],
        "blockers": _unique(blockers),
        **FALSE_AUTHORITY,
    }


def _sweep_stack(
    repo_root: Path,
    oss_audit_root: Path | None,
    stack_id: str,
    req: Mapping[str, Any],
    proof_refs: Sequence[str],
) -> dict[str, Any]:
    official_root = (
        oss_audit_root / "repos" / str(req["official_repo_name"])
        if oss_audit_root
        else None
    )
    official_files = _file_rows(
        official_root, req["official_required_files"], missing_root_ok=True
    )
    local_files = _file_rows(repo_root, req["local_surfaces"], missing_root_ok=False)
    official_text = _joined_existing_text(official_files)
    local_text = _joined_existing_text(local_files)
    official_features = _token_checks(official_text, req["official_feature_tokens"])
    local_features = _token_checks(local_text, req["local_feature_tokens"])
    missing_official_features = [
        key for key, value in official_features.items() if not value["present"]
    ]
    missing_local_features = [
        key for key, value in local_features.items() if not value["present"]
    ]
    proof_status = [
        {
            "proof": proof,
            "provided": proof in set(proof_refs),
        }
        for proof in req["promotion_proofs"]
    ]
    missing_proofs = [row["proof"] for row in proof_status if not row["provided"]]
    blockers = []
    if official_root is None or not official_root.is_dir():
        blockers.append(f"{stack_id}_official_oss_snapshot_missing")
    blockers.extend(
        f"{stack_id}_official_file_missing:{row['path']}"
        for row in official_files
        if not row["exists"]
    )
    blockers.extend(
        f"{stack_id}_local_surface_missing:{row['path']}"
        for row in local_files
        if not row["exists"]
    )
    blockers.extend(
        f"{stack_id}_official_feature_missing:{feature}"
        for feature in missing_official_features
    )
    blockers.extend(
        f"{stack_id}_local_feature_missing:{feature}"
        for feature in missing_local_features
    )
    blockers.extend(f"{stack_id}_proof_missing:{proof}" for proof in missing_proofs)

    return {
        "stack_id": stack_id,
        "verdict": (
            "NO_GO_PRODUCTION_HARDENED__SOURCE_AND_RECEIVER_GATES_INCOMPLETE"
            if blockers
            else "GO_STACK_GATES_COMPLETE"
        ),
        "official_repo_root": str(official_root) if official_root is not None else None,
        "official_files": official_files,
        "official_feature_checks": official_features,
        "local_files": local_files,
        "local_feature_checks": local_features,
        "promotion_proof_status": proof_status,
        "provided_proof_refs": list(proof_refs),
        "production_blockers": _unique(blockers),
        "fully_optimized_claim": False,
        "source_faithful_stack_claim": False,
        "promotion_eligible": False,
    }


def _related_omx_design_memo_refs(repo_root: Path) -> list[dict[str, Any]]:
    research_root = repo_root / ".omx" / "research"
    if not research_root.is_dir():
        return []
    refs: list[dict[str, Any]] = []
    for path in sorted(research_root.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if path.suffix.lower() not in RELATED_OMX_MEMO_EXTENSIONS:
            continue
        if not any(term in lower_name for term in RELATED_OMX_MEMO_TERMS):
            continue
        data = path.read_bytes()
        refs.append(
            {
                "path": str(path.relative_to(repo_root)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "topics": _memo_topics(lower_name),
                "artifact_kind": _memo_artifact_kind(lower_name),
            }
        )
    return refs


def _memo_topics(lower_name: str) -> list[str]:
    topics: list[str] = []
    if "snerv" in lower_name:
        topics.append("snerv")
    if "hinerv" in lower_name or "hi_nerv" in lower_name:
        topics.append("hinerv")
    if "hnerv" in lower_name or "pr95" in lower_name:
        topics.append("pr95_hnerv_control")
    if "sr_nerv" in lower_name or "srnerv" in lower_name or "sr-neerv" in lower_name:
        topics.append("sr_nerv_enhancer")
    if "rnerv" in lower_name:
        topics.append("rnerv_optimizer")
    if "ffnerv" in lower_name:
        topics.append("ffnerv_flow")
    if "boostnerv" in lower_name or "boost_nerv" in lower_name:
        topics.append("boostnerv")
    if "e_nerv" in lower_name or "enerv" in lower_name:
        topics.append("enerv")
    if "nerv" in lower_name and not topics:
        topics.append("nerv_family")
    return topics or ["nerv_family"]


def _memo_artifact_kind(lower_name: str) -> str:
    if lower_name.startswith("codex_findings_"):
        return "codex_findings_memo"
    if lower_name.startswith(("council_", "grand_council_")):
        return "council_or_design_memo"
    if "design" in lower_name:
        return "design_memo"
    if "adjudication" in lower_name or "advisory" in lower_name:
        return "empirical_adjudication"
    if lower_name.endswith(".json"):
        return "machine_readable_ledger"
    return "research_memo"


def _file_rows(
    root: Path | None,
    rel_paths: Sequence[str],
    *,
    missing_root_ok: bool,
) -> list[dict[str, Any]]:
    rows = []
    for rel_path in rel_paths:
        path = root / rel_path if root is not None else None
        exists = bool(path and path.is_file())
        if root is None and not missing_root_ok:
            exists = False
        rows.append(
            {
                "path": rel_path,
                "exists": exists,
                "bytes": path.stat().st_size if exists and path else None,
                "_abs_path": str(path) if exists and path else None,
            }
        )
    return rows


def _joined_existing_text(file_rows: Sequence[Mapping[str, Any]]) -> str:
    chunks: list[str] = []
    for row in file_rows:
        root_path = row.get("_abs_path")
        if root_path:
            chunks.append(Path(str(root_path)).read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _token_checks(
    text: str,
    checks: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, Any]]:
    return {
        check_id: {
            "required_tokens": list(tokens),
            "present_tokens": [token for token in tokens if token in text],
            "present": all(token in text for token in tokens),
        }
        for check_id, tokens in checks.items()
    }


def _unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


__all__ = [
    "AXIS_TAG",
    "SCHEMA",
    "STACK_REQUIREMENTS",
    "NervImplementationDesignSweepError",
    "build_nerv_implementation_design_sweep",
]
