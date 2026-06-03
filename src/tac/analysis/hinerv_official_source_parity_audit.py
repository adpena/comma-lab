# SPDX-License-Identifier: MIT
"""Source-backed HiNeRV official forward/bitstream parity audit.

The local HiNeRV carrier has useful receiver-visible controls, but those
controls are not enough to call the stack source-faithful to the official
HiNeRV repository.  This audit hashes the official source surfaces we depend on
and accepts only numeric replay evidence for official-forward parity.

The artifact is deliberately false-authority: it can route engineering budget
and block stale claims, but it cannot promote score or exact-eval readiness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tac.analysis.source_marker_scan import read_python_source_for_marker_scan
from tac.substrates.hi_nerv.official_grid import official_grid_trilinear3d_forward
from tac.substrates.hi_nerv.official_patch import (
    official_compute_pixel_idx_3d,
    official_flat_patch_index_to_thw,
    official_patch_to_video,
    official_video_to_patch,
    official_vidx_to_pidx,
)

SCHEMA = "hinerv_official_source_parity_audit.v1"
AUTHORITY = "false_authority_source_audit_no_score_claim"
FORWARD_PARITY_ARTIFACT_SCHEMA = "hinerv_official_forward_parity.v1"
OFFICIAL_REPO_URL = "https://github.com/hmkx/HiNeRV"
OFFICIAL_REPO_URL_GIT = "https://github.com/hmkx/HiNeRV.git"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

REQUIRED_OFFICIAL_FILES: tuple[str, ...] = (
    "models/hinerv.py",
    "models/layers.py",
    "datasets.py",
    "hinerv_compress.py",
    "compression/quant_utils.py",
    "compression/codec_utils.py",
    "compression/prune_utils.py",
)

REQUIRED_FORWARD_PARITY_COMPONENT_IDS: tuple[str, ...] = (
    "core_hierarchical_renderer",
    "patch_dataset_path",
    "prune_quant_codec",
)


@dataclass(frozen=True)
class MarkerGroup:
    """One official HiNeRV source surface and its required markers."""

    group_id: str
    description: str
    source_files: tuple[str, ...]
    markers: tuple[str, ...]


OFFICIAL_MARKER_GROUPS: tuple[MarkerGroup, ...] = (
    MarkerGroup(
        group_id="official_hierarchical_feature_grid",
        description="Official HiNeRV builds temporal-local FeatureGrid levels expanded by GridTrilinear3D.",
        source_files=("models/hinerv.py", "models/layers.py"),
        markers=(
            "class HiNeRV",
            "class HiNeRVEncoding",
            "FeatureGrid",
            "GridTrilinear3D",
            "self.grids.append",
            "self.grid_expands.append",
            "enc-grid-level",
        ),
    ),
    MarkerGroup(
        group_id="official_convnext_decoder",
        description="Official HiNeRV decoder uses ConvNeXt block families and staged upsampling.",
        source_files=("models/hinerv.py", "models/layers.py"),
        markers=(
            "class HiNeRVDecoder",
            "HiNeRVUpsampler",
            "ConvNeXtBlock",
            "ConvNeXtBlockLessNorm",
            "Upsample",
            "blocks",
        ),
    ),
    MarkerGroup(
        group_id="official_patch_dataset_path",
        description="Official training/eval consumes spatial patches through --patch-size.",
        source_files=("datasets.py",),
        markers=(
            "--patch-size",
            "load_all_patches",
            "'patch'",
            "patch_size",
        ),
    ),
    MarkerGroup(
        group_id="official_quant_prune_torchac_bitstream",
        description="Official compression path binds QuantNoise, PruningMask, and torchac arithmetic coding.",
        source_files=(
            "hinerv_compress.py",
            "compression/quant_utils.py",
            "compression/codec_utils.py",
            "compression/prune_utils.py",
        ),
        markers=(
            "QuantNoise",
            "PruningMask",
            "quant-level",
            "torchac.encode_float_cdf",
            "torchac.decode_float_cdf",
            "set_quantization",
            "set_pruning",
        ),
    ),
    MarkerGroup(
        group_id="official_config_family_controls",
        description="Official configs expose base/encoding grid size, levels, scales, and quant controls.",
        source_files=(
            "models/hinerv.py",
            "hinerv_compress.py",
            "cfgs/models/uvg-hinerv-s_1920x1080.txt",
        ),
        markers=(
            "--base-grid-size",
            "--base-grid-level",
            "--base-grid-level-scale",
            "--enc-grid-size",
            "--enc-grid-level",
            "--enc-grid-level-scale",
            "--quant-level",
        ),
    ),
)

LOCAL_BINDING_MARKERS: tuple[str, ...] = (
    "HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF",
    "OfficialGridTrilinear3D",
    "HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF",
    "official_video_to_patch",
    "official_compute_pixel_idx_3d",
    "HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF",
    "HierarchicalFeatureGrid",
    "ConvNeXtBlock",
    "trilinear_upsample",
    "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
    "apply_decoder_pruning",
    "apply_decoder_quant_noise",
    "measure_hi_nerv_decoder_bitstream_roundtrip",
)

_HINERV_FORWARD_PARITY_COMPONENT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "component_id": "core_hierarchical_renderer",
        "official_group_ids": (
            "official_hierarchical_feature_grid",
            "official_convnext_decoder",
            "official_config_family_controls",
        ),
        "local_receiver_markers": (
            (
                "src/tac/substrates/hi_nerv/architecture.py",
                "HINERV_OFFICIAL_FEATURE_GRID_CONVNEXT_PROOF",
            ),
            ("src/tac/substrates/hi_nerv/architecture.py", "class HierarchicalFeatureGrid"),
            ("src/tac/substrates/hi_nerv/architecture.py", "class ConvNeXtBlock"),
            ("src/tac/substrates/hi_nerv/official_grid.py", "class OfficialGridTrilinear3D"),
            (
                "src/tac/substrates/hi_nerv/official_grid.py",
                "HINERV_OFFICIAL_GRID_TRILINEAR3D_NUMPY_PROOF",
            ),
            ("src/tac/substrates/hi_nerv/mlx_renderer.py", "class ConvNeXtBlockMLX"),
        ),
        "local_source_forward_markers": (
            (
                "src/tac/substrates/hi_nerv/architecture.py",
                "HINERV_OFFICIAL_HIERARCHICAL_RENDERER_SOURCE_FORWARD_PROOF",
            ),
        ),
        "classification": "receiver_visible_official_like_renderer_without_source_forward_replay",
        "blocker": "hinerv_core_hierarchical_renderer_source_forward_replay_missing",
        "evidence_summary": (
            "Local HiNeRV has receiver-visible feature-grid/ConvNeXt/trilinear "
            "surfaces, but official hmkx HiNeRV torch source-forward replay is "
            "not proven until the numeric artifact carries matching hashes."
        ),
    },
    {
        "component_id": "patch_dataset_path",
        "official_group_ids": ("official_patch_dataset_path",),
        "local_receiver_markers": (
            ("src/tac/substrates/hi_nerv/architecture.py", "pair_indices"),
            ("src/tac/substrates/hi_nerv/architecture.py", "num_pairs"),
            ("src/tac/substrates/hi_nerv/inflate.py", "num_pairs"),
            (
                "src/tac/substrates/hi_nerv/official_patch.py",
                "HINERV_OFFICIAL_PATCH_INDEX_NUMPY_PROOF",
            ),
            ("src/tac/substrates/hi_nerv/official_patch.py", "def official_video_to_patch"),
            (
                "src/tac/substrates/hi_nerv/official_patch.py",
                "def official_compute_pixel_idx_3d",
            ),
        ),
        "local_source_forward_markers": (
            (
                "src/tac/substrates/hi_nerv/architecture.py",
                "HINERV_OFFICIAL_PATCH_DATASET_SOURCE_FORWARD_PROOF",
            ),
        ),
        "classification": "frame_index_receiver_path_without_official_patch_dataset_replay",
        "blocker": "hinerv_patch_dataset_source_forward_replay_missing",
        "evidence_summary": (
            "Official HiNeRV trains/evals spatial patches through --patch-size; "
            "the local receiver renders pair-indexed full frames and needs an "
            "explicit patch/frame equivalence replay before source-faithful use."
        ),
    },
    {
        "component_id": "prune_quant_codec",
        "official_group_ids": ("official_quant_prune_torchac_bitstream",),
        "local_receiver_markers": (
            (
                "src/tac/substrates/hi_nerv/bitstream.py",
                "HI_NERV_PRUNE_QUANTNOISE_BITSTREAM_PIPELINE_PROOF",
            ),
            ("src/tac/substrates/hi_nerv/bitstream.py", "def apply_decoder_pruning"),
            ("src/tac/substrates/hi_nerv/bitstream.py", "def apply_decoder_quant_noise"),
            (
                "src/tac/substrates/hi_nerv/bitstream.py",
                "def measure_hi_nerv_decoder_bitstream_roundtrip",
            ),
        ),
        "local_source_forward_markers": (
            (
                "src/tac/substrates/hi_nerv/bitstream.py",
                "HINERV_OFFICIAL_PRUNE_QUANT_TORCHAC_SOURCE_FORWARD_PROOF",
            ),
        ),
        "classification": "receiver_visible_prune_quantnoise_without_official_torchac_replay",
        "blocker": "hinerv_prune_quant_codec_source_forward_replay_missing",
        "evidence_summary": (
            "Local pruning/QuantNoise/bitstream probes are real receiver-visible "
            "controls, but they are not official torchac bitstream-q parity."
        ),
    },
)


def build_hinerv_official_source_parity_audit(
    *,
    official_repo_dir: str | Path,
    repo_root: str | Path,
    official_forward_parity_artifact_path: str | Path | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed official HiNeRV source-parity audit."""

    official_root = Path(official_repo_dir)
    local_root = Path(repo_root)
    if generated_utc is None:
        generated_utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_rows = [_file_row(official_root, rel_path) for rel_path in REQUIRED_OFFICIAL_FILES]
    missing_files = [row["rel_path"] for row in file_rows if row["status"] != "present"]
    official_group_rows = [_official_marker_group_row(official_root, group) for group in OFFICIAL_MARKER_GROUPS]
    local_binding_row = _local_marker_row(local_root, markers=LOCAL_BINDING_MARKERS)
    forward_row = _forward_parity_artifact_row(official_forward_parity_artifact_path)
    official_markers_present = not missing_files and all(row["all_markers_present"] for row in official_group_rows)
    local_bindings_present = bool(local_binding_row["all_markers_present"])
    official_forward_parity_proven = bool(
        official_markers_present and local_bindings_present and forward_row["parity_passed"]
    )
    component_state_rows = _component_state_rows(
        official_group_rows=official_group_rows,
        local_root=local_root,
        forward_parity_artifact_row=forward_row,
    )
    blockers = _ordered_unique(
        [
            "hinerv_official_source_files_missing" if missing_files else "",
            *[
                f"hinerv_official_source_marker_missing:{row['group_id']}"
                for row in official_group_rows
                if not row["all_markers_present"]
            ],
            "hinerv_local_receiver_bindings_missing" if not local_bindings_present else "",
            "hinerv_official_forward_parity_missing" if not official_forward_parity_proven else "",
        ]
    )
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "generated_utc": generated_utc,
        "family": "hi_nerv",
        "official_repo": {
            "repo_url": OFFICIAL_REPO_URL,
            "repo_url_git": OFFICIAL_REPO_URL_GIT,
            "root": official_root.as_posix(),
            "head_sha": _git_head_sha(official_root),
            "required_files": list(REQUIRED_OFFICIAL_FILES),
        },
        "local_repo_root": local_root.as_posix(),
        "official_file_rows": file_rows,
        "official_marker_group_rows": official_group_rows,
        "local_binding_marker_row": local_binding_row,
        "official_forward_parity_artifact_row": forward_row,
        "component_state_rows": component_state_rows,
        "official_source_markers_present": official_markers_present,
        "local_receiver_bindings_present": local_bindings_present,
        "official_forward_parity_proven": official_forward_parity_proven,
        "blockers": blockers,
        "next_actions": _next_actions(blockers),
        **FALSE_AUTHORITY,
    }


def build_hinerv_official_forward_parity_artifact(
    *,
    official_repo_dir: str | Path,
    repo_root: str | Path,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority HiNeRV source-forward proof/falsification artifact."""

    official_root = Path(official_repo_dir)
    local_root = Path(repo_root)
    if generated_utc is None:
        generated_utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_rows = [_file_row(official_root, rel_path) for rel_path in REQUIRED_OFFICIAL_FILES]
    official_group_rows = [_official_marker_group_row(official_root, group) for group in OFFICIAL_MARKER_GROUPS]
    official_forward_replay = _official_hinerv_forward_source_replay(official_root)
    component_rows = _component_state_rows(
        official_group_rows=official_group_rows,
        local_root=local_root,
        forward_parity_artifact_row={"parity_passed": False, "component_rows": []},
    )
    component_rows = _attach_official_forward_replay_to_component_rows(
        component_rows,
        official_forward_replay=official_forward_replay,
    )
    numeric_subcomponent_rows = [
        _official_grid_trilinear3d_numeric_replay_row(),
        _official_patch_index_numeric_replay_row(),
    ]
    parity_passed = all(bool(row["source_forward_parity_proven"]) for row in component_rows)
    parity_falsified = any(bool(row["source_forward_parity_falsified"]) for row in component_rows)
    return {
        "schema": FORWARD_PARITY_ARTIFACT_SCHEMA,
        "authority": AUTHORITY,
        "generated_utc": generated_utc,
        "family": "hi_nerv",
        "official_repo": {
            "repo_url": OFFICIAL_REPO_URL,
            "repo_url_git": OFFICIAL_REPO_URL_GIT,
            "root": official_root.as_posix(),
            "head_sha": _git_head_sha(official_root),
            "required_files": list(REQUIRED_OFFICIAL_FILES),
        },
        "local_repo_root": local_root.as_posix(),
        "official_file_rows": file_rows,
        "official_marker_group_rows": official_group_rows,
        "official_weight_manifest": official_forward_replay[
            "official_weight_manifest"
        ],
        "source_forward_replay": official_forward_replay,
        "component_rows": component_rows,
        "numeric_subcomponent_rows": numeric_subcomponent_rows,
        "official_forward_parity_passed": parity_passed,
        "official_forward_parity_falsified": parity_falsified,
        "blockers": _ordered_unique(
            [
                blocker
                for row in component_rows
                for blocker in row.get("blockers") or ()
            ]
        ),
        **FALSE_AUTHORITY,
    }


def summarize_hinerv_official_source_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact row-safe summary of a HiNeRV official-source audit."""

    official_repo = report.get("official_repo")
    if not isinstance(official_repo, Mapping):
        official_repo = {}
    forward_row = report.get("official_forward_parity_artifact_row")
    if not isinstance(forward_row, Mapping):
        forward_row = {}
    return {
        "schema": str(report.get("schema") or ""),
        "authority": str(report.get("authority") or ""),
        "family": str(report.get("family") or "hi_nerv"),
        "official_repo_root": str(official_repo.get("root") or ""),
        "official_repo_url": str(official_repo.get("repo_url") or ""),
        "official_head_sha": official_repo.get("head_sha"),
        "official_source_markers_present": bool(report.get("official_source_markers_present")),
        "local_receiver_bindings_present": bool(report.get("local_receiver_bindings_present")),
        "official_forward_parity_proven": bool(report.get("official_forward_parity_proven")),
        "forward_parity_falsified": bool(forward_row.get("parity_falsified")),
        "component_states": [
            {
                "component_id": row.get("component_id"),
                "classification": row.get("classification"),
                "source_forward_parity_proven": bool(row.get("source_forward_parity_proven")),
                "source_forward_parity_falsified": bool(row.get("source_forward_parity_falsified")),
            }
            for row in report.get("component_state_rows") or ()
            if isinstance(row, Mapping)
        ],
        "numeric_subcomponent_replays": [
            {
                "component_id": row.get("component_id"),
                "parent_component_id": row.get("parent_component_id"),
                "backend": row.get("backend"),
                "source_forward_parity_proven": bool(row.get("source_forward_parity_proven")),
                "full_hinerv_forward_parity_proven": bool(
                    row.get("full_hinerv_forward_parity_proven")
                ),
                "max_abs_error": row.get("max_abs_error"),
                "tolerance": row.get("tolerance"),
                "blockers": list(row.get("blockers") or ()),
            }
            for row in _numeric_subcomponent_rows_from_report(report)
        ],
        "blockers": list(report.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def render_hinerv_official_source_parity_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing HiNeRV audit."""

    official_repo = report.get("official_repo")
    if not isinstance(official_repo, Mapping):
        official_repo = {}
    forward_row = report.get("official_forward_parity_artifact_row")
    if not isinstance(forward_row, Mapping):
        forward_row = {}
    lines = [
        "# HiNeRV Official Source-Parity Audit",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Official repo: `{official_repo.get('root')}`",
        f"Official head SHA: `{official_repo.get('head_sha')}`",
        "",
        "## Verdict",
        "",
        f"- official source markers present: `{bool(report.get('official_source_markers_present'))}`",
        f"- local receiver bindings present: `{bool(report.get('local_receiver_bindings_present'))}`",
        f"- official forward parity proven: `{bool(report.get('official_forward_parity_proven'))}`",
        f"- official forward parity falsified: `{bool(forward_row.get('parity_falsified'))}`",
        f"- score claim: `{bool(report.get('score_claim'))}`",
        "",
        "## Marker Groups",
        "",
        "| group | present | missing |",
        "|---|---:|---|",
    ]
    for row in report.get("official_marker_group_rows") or ():
        if not isinstance(row, Mapping):
            continue
        missing = ", ".join(str(v) for v in row.get("missing_markers") or ())
        lines.append(f"| `{row.get('group_id')}` | `{bool(row.get('all_markers_present'))}` | `{missing}` |")
    lines.extend(["", "## Blockers", ""])
    blockers = list(report.get("blockers") or ())
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- none")
    component_rows = [
        row for row in report.get("component_state_rows") or () if isinstance(row, Mapping)
    ]
    if component_rows:
        lines.extend(["", "## Component States", ""])
        lines.append("| component | proven | falsified | classification |")
        lines.append("|---|---:|---:|---|")
        for row in component_rows:
            lines.append(
                "| "
                f"`{row.get('component_id')}` | "
                f"`{bool(row.get('source_forward_parity_proven'))}` | "
                f"`{bool(row.get('source_forward_parity_falsified'))}` | "
                f"`{row.get('classification')}` |"
            )
    lines.extend(["", "## Next Actions", ""])
    actions = list(report.get("next_actions") or ())
    if actions:
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def _file_row(root: Path, rel_path: str) -> dict[str, Any]:
    path = root / rel_path
    if not path.is_file():
        return {
            "rel_path": rel_path,
            "path": path.as_posix(),
            "status": "missing",
            "bytes": 0,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "rel_path": rel_path,
        "path": path.as_posix(),
        "status": "present",
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _official_marker_group_row(root: Path, group: MarkerGroup) -> dict[str, Any]:
    text = "\n".join(_read_source_for_marker_scan(root / rel_path) for rel_path in group.source_files)
    missing = [marker for marker in group.markers if marker not in text]
    return {
        "group_id": group.group_id,
        "description": group.description,
        "source_files": list(group.source_files),
        "markers": list(group.markers),
        "missing_markers": missing,
        "all_markers_present": not missing,
    }


def _read_source_for_marker_scan(path: Path) -> str:
    if path.suffix == ".py" or path.is_dir():
        return read_python_source_for_marker_scan(path)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _local_marker_row(root: Path, *, markers: Sequence[str]) -> dict[str, Any]:
    source = read_python_source_for_marker_scan(
        root / "src/tac/substrates/hi_nerv",
        exclude_names=(),
    )
    missing = [marker for marker in markers if marker not in source]
    return {
        "marker_group_id": "local_hinerv_receiver_visible_bindings",
        "source_root": (root / "src/tac/substrates/hi_nerv").as_posix(),
        "markers": list(markers),
        "present_markers": [marker for marker in markers if marker not in missing],
        "missing_markers": missing,
        "all_markers_present": not missing,
    }


def _forward_parity_artifact_row(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return _missing_forward_row(path=None)
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return _missing_forward_row(path=artifact_path.as_posix())
    data = artifact_path.read_bytes()
    digest = sha256(data).hexdigest()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "schema": "hinerv_official_forward_parity_artifact_row.v1",
            "path": artifact_path.as_posix(),
            "status": "unreadable",
            "artifact_schema": None,
            "bytes": len(data),
            "sha256": digest,
            "parity_passed": False,
            "parity_falsified": False,
            "component_rows": [],
            "error": str(exc),
            "blockers": ["hinerv_official_forward_parity_artifact_unreadable"],
            **FALSE_AUTHORITY,
        }
    component_rows = [
        dict(row)
        for row in payload.get("component_rows") or ()
        if isinstance(row, Mapping)
    ]
    artifact_schema = str(payload.get("schema") or "")
    parity_falsified = bool(
        artifact_schema == FORWARD_PARITY_ARTIFACT_SCHEMA
        and (
            payload.get("official_forward_parity_falsified") is True
            or any(row.get("source_forward_parity_falsified") is True for row in component_rows)
        )
    )
    evidence_blockers = _forward_parity_artifact_evidence_blockers(
        payload,
        component_rows=component_rows,
        accept_falsification=parity_falsified,
    )
    parity_passed = bool(
        artifact_schema == FORWARD_PARITY_ARTIFACT_SCHEMA
        and payload.get("official_forward_parity_passed") is True
        and payload.get("score_claim") is False
        and payload.get("ready_for_exact_eval_dispatch") is False
        and not evidence_blockers
    )
    falsification_accepted = bool(
        artifact_schema == FORWARD_PARITY_ARTIFACT_SCHEMA
        and parity_falsified
        and payload.get("score_claim") is False
        and payload.get("ready_for_exact_eval_dispatch") is False
        and not evidence_blockers
    )
    blockers = []
    if artifact_schema != FORWARD_PARITY_ARTIFACT_SCHEMA:
        blockers.append("hinerv_official_forward_parity_artifact_schema_invalid")
    if payload.get("official_forward_parity_passed") is not True and not parity_falsified:
        blockers.append("hinerv_official_forward_parity_artifact_not_passing")
    if payload.get("score_claim") is not False:
        blockers.append("hinerv_official_forward_parity_artifact_score_claim_not_false")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("hinerv_official_forward_parity_artifact_exact_flag_not_false")
    blockers.extend(evidence_blockers)
    return {
        "schema": "hinerv_official_forward_parity_artifact_row.v1",
        "path": artifact_path.as_posix(),
        "status": "present",
        "artifact_schema": artifact_schema,
        "bytes": len(data),
        "sha256": digest,
        "parity_passed": parity_passed,
        "parity_falsified": parity_falsified,
        "falsification_accepted": falsification_accepted,
        "component_rows": component_rows,
        "artifact_blockers": list(payload.get("blockers") or ()),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _missing_forward_row(*, path: str | None) -> dict[str, Any]:
    return {
        "schema": "hinerv_official_forward_parity_artifact_row.v1",
        "path": path,
        "status": "missing",
        "artifact_schema": None,
        "bytes": 0,
        "sha256": None,
        "parity_passed": False,
        "parity_falsified": False,
        "falsification_accepted": False,
        "component_rows": [],
        "blockers": ["hinerv_official_forward_parity_artifact_missing"],
        **FALSE_AUTHORITY,
    }


def _forward_parity_artifact_evidence_blockers(
    payload: Mapping[str, Any],
    *,
    component_rows: Sequence[Mapping[str, Any]],
    accept_falsification: bool = False,
) -> list[str]:
    blockers: list[str] = []
    weight_manifest = payload.get("official_weight_manifest")
    if not isinstance(weight_manifest, Mapping):
        blockers.append("hinerv_official_forward_parity_weight_manifest_missing")
    else:
        if not _is_sha256_hex(weight_manifest.get("state_dict_sha256")):
            blockers.append("hinerv_official_forward_parity_weight_manifest_sha256_missing")
        if _int_or_none(weight_manifest.get("state_dict_key_count")) is None:
            blockers.append("hinerv_official_forward_parity_weight_manifest_keys_missing")
    source_replay = payload.get("source_forward_replay")
    if not isinstance(source_replay, Mapping):
        blockers.append("hinerv_official_forward_parity_source_replay_missing")
    else:
        backend = str(source_replay.get("backend") or "")
        if backend not in {
            "torch_vs_numpy",
            "torch_vs_mlx",
            "official_torch_vs_portable",
            "official_torch_vs_mlx",
            "official_torch_cpu_full_hinerv_forward",
        }:
            blockers.append("hinerv_official_forward_parity_source_replay_backend_invalid")
        if not _is_sha256_hex(source_replay.get("input_bundle_sha256")):
            blockers.append("hinerv_official_forward_parity_source_replay_input_sha256_missing")
    by_component = {
        str(row.get("component_id") or ""): row
        for row in component_rows
        if isinstance(row, Mapping)
    }
    for component_id in REQUIRED_FORWARD_PARITY_COMPONENT_IDS:
        row = by_component.get(component_id)
        if row is None:
            blockers.append(f"hinerv_official_forward_parity_component_missing:{component_id}")
            continue
        if accept_falsification:
            if row.get("source_forward_parity_falsified") is not True:
                blockers.append(
                    f"hinerv_official_forward_parity_component_not_falsified:{component_id}"
                )
            continue
        blockers.extend(
            f"{blocker}:{component_id}"
            for blocker in _forward_parity_component_blockers(row)
        )
    return _ordered_unique(blockers)


def _forward_parity_component_blockers(row: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if row.get("source_forward_parity_proven") is not True:
        blockers.append("component_not_proven")
    tolerance = _float_or_none(row.get("tolerance") or row.get("max_abs_tolerance"))
    max_abs_error = _float_or_none(
        _first_not_none(row, ("max_abs_error", "max_error", "max_abs_delta"))
    )
    if tolerance is None or tolerance < 0.0:
        blockers.append("numeric_tolerance_missing")
    if max_abs_error is None:
        blockers.append("numeric_max_abs_error_missing")
    elif tolerance is not None and max_abs_error > tolerance:
        blockers.append("numeric_max_abs_error_exceeds_tolerance")
    for field in (
        "input_sha256",
        "official_output_sha256",
        "portable_output_sha256",
    ):
        if not _is_sha256_hex(row.get(field)):
            blockers.append(f"{field}_missing")
    if row.get("official_output_sha256") != row.get("portable_output_sha256"):
        blockers.append("official_portable_output_sha256_mismatch")
    if not _is_sha256_hex(row.get("official_weight_sha256")):
        blockers.append("official_weight_identity_missing")
    return blockers


def _component_state_rows(
    *,
    official_group_rows: Sequence[Mapping[str, Any]],
    local_root: Path,
    forward_parity_artifact_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    official_by_group = {str(row.get("group_id")): row for row in official_group_rows}
    artifact_passed = bool(forward_parity_artifact_row.get("parity_passed"))
    artifact_falsified = bool(forward_parity_artifact_row.get("parity_falsified"))
    artifact_component_rows = {
        str(row.get("component_id")): row
        for row in forward_parity_artifact_row.get("component_rows") or ()
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for spec in _HINERV_FORWARD_PARITY_COMPONENT_SPECS:
        component_id = str(spec["component_id"])
        official_group_ids = tuple(str(value) for value in spec["official_group_ids"])
        official_group_present = all(
            bool(official_by_group.get(group_id, {}).get("all_markers_present"))
            for group_id in official_group_ids
        )
        local_receiver_scan = _file_marker_rows(
            local_root,
            spec.get("local_receiver_markers") or (),
        )
        local_source_forward_scan = _file_marker_rows(
            local_root,
            spec.get("local_source_forward_markers") or (),
        )
        receiver_visible = bool(local_receiver_scan["all_markers_present"])
        local_source_forward_markers_present = bool(
            local_source_forward_scan["all_markers_present"]
        )
        artifact_component = artifact_component_rows.get(component_id, {})
        source_forward = bool(
            official_group_present
            and receiver_visible
            and artifact_passed
            and artifact_component.get("source_forward_parity_proven") is True
        )
        source_forward_falsified = bool(
            artifact_component.get("source_forward_parity_falsified")
            or (
                official_group_present
                and receiver_visible
                and not local_source_forward_markers_present
            )
        )
        if source_forward:
            classification = "official_source_forward_parity_proven"
        elif source_forward_falsified:
            classification = str(spec["classification"])
        elif receiver_visible:
            classification = "receiver_visible_analogue"
        else:
            classification = "missing_or_partial"
        artifact_blocker = ""
        if not artifact_passed:
            artifact_blocker = (
                "hinerv_official_forward_parity_artifact_falsifies_parity"
                if artifact_falsified
                else "hinerv_official_forward_parity_artifact_missing_or_failed"
            )
        blockers = _ordered_unique(
            [
                *[
                    (
                        f"hinerv_official_source_marker_missing:{group_id}"
                        if not bool(official_by_group.get(group_id, {}).get("all_markers_present"))
                        else ""
                    )
                    for group_id in official_group_ids
                ],
                (
                    f"hinerv_{component_id}_receiver_visible_adapter_missing"
                    if not receiver_visible
                    else ""
                ),
                (
                    f"hinerv_{component_id}_local_source_forward_markers_missing"
                    if not local_source_forward_markers_present
                    else ""
                ),
                artifact_blocker if not artifact_passed else "",
                "" if (source_forward or not source_forward_falsified) else str(spec["blocker"]),
            ]
        )
        rows.append(
            {
                "schema": "hinerv_official_component_state.v1",
                "component_id": component_id,
                "official_group_ids": list(official_group_ids),
                "official_markers_present": official_group_present,
                "receiver_visible_analogue_present": receiver_visible,
                "local_receiver_marker_rows": local_receiver_scan["marker_rows"],
                "local_source_forward_markers_present": local_source_forward_markers_present,
                "local_source_forward_marker_rows": local_source_forward_scan["marker_rows"],
                "forward_parity_artifact_passed": artifact_passed,
                "forward_parity_artifact_component": dict(artifact_component),
                "official_source_forward_replay": (
                    dict(artifact_component["official_source_forward_replay"])
                    if isinstance(
                        artifact_component.get("official_source_forward_replay"),
                        Mapping,
                    )
                    else None
                ),
                "source_forward_parity_proven": source_forward,
                "source_forward_parity_falsified": source_forward_falsified,
                "classification": classification,
                "evidence_summary": str(spec["evidence_summary"]),
                "blockers": blockers,
                **FALSE_AUTHORITY,
            }
        )
    return rows


def _file_marker_rows(
    root: Path,
    markers: Sequence[Sequence[str] | tuple[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cache: dict[str, str] = {}
    for marker_spec in markers:
        rel_path = str(marker_spec[0])
        marker = str(marker_spec[1])
        if rel_path not in cache:
            path = root / rel_path
            cache[rel_path] = (
                read_python_source_for_marker_scan(path) if path.is_file() else ""
            )
        rows.append(
            {
                "rel_path": rel_path,
                "marker": marker,
                "present": marker in cache[rel_path],
            }
        )
    return {
        "marker_rows": rows,
        "all_markers_present": all(row["present"] for row in rows),
        "missing_markers": [
            f"{row['rel_path']}::{row['marker']}"
            for row in rows
            if not row["present"]
        ],
    }


def _numeric_subcomponent_rows_from_report(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    forward_row = report.get("official_forward_parity_artifact_row")
    if not isinstance(forward_row, Mapping):
        return []
    path_value = forward_row.get("path")
    if not path_value:
        return []
    artifact_path = Path(str(path_value))
    if not artifact_path.is_file():
        return []
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    return [
        row
        for row in payload.get("numeric_subcomponent_rows") or ()
        if isinstance(row, Mapping)
    ]


class _SilentLogger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None


def _official_hinerv_forward_source_replay(official_root: Path) -> dict[str, Any]:
    """Run a tiny deterministic official HiNeRV torch forward.

    This is source-custody evidence only.  It proves the pinned upstream graph
    can be instantiated and executed on a hashed input; it does not compare
    against a local portable/MLX renderer, so it cannot clear forward parity.
    """

    input_record = _official_hinerv_tiny_input_record()
    backend = "official_torch_cpu_full_hinerv_forward"
    try:
        with _temporary_official_hinerv_import(official_root) as official_hinerv:
            import torch

            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                torch.manual_seed(20260603)
                model_input = _official_hinerv_torch_input(torch)
                args = _tiny_official_hinerv_args()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    model = official_hinerv.build_model(
                        args,
                        _SilentLogger(),
                        model_input,
                    )
                model.eval()
                with torch.no_grad():
                    output = model(model_input)
                output_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
                state_manifest = _torch_state_dict_manifest(model.state_dict())
            finally:
                torch.set_num_threads(previous_threads)

        output_sha = _array_sha256(output_np)
        return {
            "schema": "hinerv_official_source_forward_replay.v1",
            "authority": AUTHORITY,
            "backend": backend,
            "replay_ran": True,
            "official_repo_root": official_root.as_posix(),
            "official_repo_head_sha": _git_head_sha(official_root),
            "input_bundle": input_record,
            "input_bundle_sha256": _stable_json_sha256(input_record),
            "official_weight_manifest": state_manifest,
            "official_output_shape": [int(v) for v in output_np.shape],
            "official_output_dtype": str(output_np.dtype),
            "official_output_sha256": output_sha,
            "official_output_min": float(output_np.min()),
            "official_output_max": float(output_np.max()),
            "official_output_mean": float(output_np.mean()),
            "local_portable_output_sha256": None,
            "full_hinerv_forward_parity_proven": False,
            "blockers": [
                "hinerv_local_portable_full_forward_adapter_missing",
                "hinerv_official_forward_replay_is_source_only",
            ],
            **FALSE_AUTHORITY,
        }
    except Exception as exc:  # pragma: no cover - dependency/source dependent
        return {
            "schema": "hinerv_official_source_forward_replay.v1",
            "authority": AUTHORITY,
            "backend": backend,
            "replay_ran": False,
            "official_repo_root": official_root.as_posix(),
            "official_repo_head_sha": _git_head_sha(official_root),
            "input_bundle": input_record,
            "input_bundle_sha256": _stable_json_sha256(input_record),
            "official_weight_manifest": {
                "schema": "hinerv_official_weight_manifest.v1",
                "state_dict_sha256": _stable_json_sha256(
                    {"state_dict": "not_built"}
                ),
                "state_dict_key_count": 0,
                "state_dict_tensor_count": 0,
                "state_dict_numel": 0,
                "state_dict_keys": [],
                "replay_status": "official_model_build_or_forward_failed",
            },
            "official_output_shape": None,
            "official_output_dtype": None,
            "official_output_sha256": None,
            "local_portable_output_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
            "full_hinerv_forward_parity_proven": False,
            "blockers": ["hinerv_official_torch_forward_replay_failed"],
            **FALSE_AUTHORITY,
        }


def _attach_official_forward_replay_to_component_rows(
    component_rows: Sequence[Mapping[str, Any]],
    *,
    official_forward_replay: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_summary = {
        "schema": "hinerv_official_source_forward_replay_summary.v1",
        "backend": official_forward_replay.get("backend"),
        "replay_ran": bool(official_forward_replay.get("replay_ran")),
        "input_bundle_sha256": official_forward_replay.get("input_bundle_sha256"),
        "official_output_sha256": official_forward_replay.get(
            "official_output_sha256"
        ),
        "official_weight_sha256": (
            (
                official_forward_replay.get("official_weight_manifest")
                if isinstance(
                    official_forward_replay.get("official_weight_manifest"),
                    Mapping,
                )
                else {}
            ).get("state_dict_sha256")
        ),
        "blockers": list(official_forward_replay.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }
    for row in component_rows:
        mutable = dict(row)
        if mutable.get("component_id") == "core_hierarchical_renderer":
            mutable["official_source_forward_replay"] = replay_summary
        rows.append(mutable)
    return rows


class _temporary_official_hinerv_import:
    def __init__(self, official_root: Path) -> None:
        self.official_root = official_root
        self._saved_path: list[str] = []
        self._saved_modules: dict[str, Any] = {}
        self._model_module_names: list[str] = []

    def __enter__(self) -> Any:
        import importlib

        self._saved_path = list(sys.path)
        self._model_module_names = [
            name for name in sys.modules if name == "models" or name.startswith("models.")
        ]
        self._saved_modules = {name: sys.modules[name] for name in self._model_module_names}
        for name in self._model_module_names:
            del sys.modules[name]
        sys.path.insert(0, self.official_root.as_posix())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return importlib.import_module("models.hinerv")

    def __exit__(self, *_exc_info: object) -> None:
        for name in [name for name in sys.modules if name == "models" or name.startswith("models.")]:
            del sys.modules[name]
        sys.modules.update(self._saved_modules)
        sys.path[:] = self._saved_path


def _official_hinerv_tiny_input_record() -> dict[str, Any]:
    return {
        "schema": "hinerv_official_tiny_forward_input.v1",
        "idx": [[0, 0, 0]],
        "idx_dtype": "int64",
        "idx_max": [1, 1, 1],
        "video_size": [1, 2, 2],
        "patch_size": [1, 2, 2],
        "torch_manual_seed": 20260603,
        "torch_num_threads": 1,
    }


def _official_hinerv_torch_input(torch_module: Any) -> dict[str, Any]:
    record = _official_hinerv_tiny_input_record()
    return {
        "idx": torch_module.tensor(record["idx"], dtype=torch_module.long),
        "idx_max": tuple(record["idx_max"]),
        "video_size": tuple(record["video_size"]),
        "patch_size": tuple(record["patch_size"]),
    }


def _tiny_official_hinerv_args() -> SimpleNamespace:
    return SimpleNamespace(
        channels=4,
        channels_reduce=2.0,
        channels_reduce_base=1,
        channels_min=1,
        depths=[1],
        exps=[1.0],
        kernels=[1],
        scales_t=[1],
        scales_hw=[2],
        stem_kernels=1,
        stem_paddings=[0, 0, 0],
        paddings=[0, 0, 0],
        base_size=[-1, -1, -1],
        base_grid_size=[1, 1, 1, 4],
        base_grid_level=1,
        base_grid_level_scale=[1.0, 1.0, 1.0, 1.0],
        base_grid_init_scale=1.0e-3,
        block_type="mlp",
        block_norm="layernorm-no-affine",
        block_act="gelu",
        block_layerscale=0.0,
        block_dropout=0.0,
        block_droppath=0.0,
        block_bias=False,
        stem_type="conv_stem",
        stem_norm="none",
        stem_act="none",
        stem_layerscale=0.0,
        stem_dropout=0.0,
        stem_droppath=0.0,
        stem_bias=True,
        head_type="linear_head",
        head_norm="none",
        head_act="sigmoid",
        head_layerscale=0.0,
        head_dropout=0.0,
        head_droppath=0.0,
        head_bias=True,
        enc_type="normalized+temp_local_grid",
        enc_align_corners=False,
        enc_pe=[1.2, 60, 1.2, 60],
        enc_pe_no_t=False,
        enc_grid_size=[1, 1, 1, 1],
        enc_grid_level=1,
        enc_grid_level_scale=[1.0, 1.0, 1.0, 1.0],
        enc_grid_init_scale=1.0e-3,
        enc_grid_depth_scale=[1.0, 1.0, 1.0, 1.0],
        upsample_type="trilinear",
        upsample_config="matmul-th-w",
        upsample_norm="layernorm-no-affine",
        upsample_act="none",
        eval_patch_size=None,
        debug=False,
    )


def _torch_state_dict_manifest(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    tensor_rows: list[dict[str, Any]] = []
    for key in sorted(str(k) for k in state_dict):
        tensor = state_dict[key]
        array = tensor.detach().cpu().numpy()
        tensor_rows.append(
            {
                "key": key,
                "shape": [int(v) for v in array.shape],
                "dtype": str(array.dtype),
                "numel": int(array.size),
                "sha256": _array_sha256(array),
            }
        )
    state_dict_sha = _stable_json_sha256(tensor_rows)
    return {
        "schema": "hinerv_official_weight_manifest.v1",
        "state_dict_sha256": state_dict_sha,
        "state_dict_key_count": len(tensor_rows),
        "state_dict_tensor_count": len(tensor_rows),
        "state_dict_numel": int(sum(row["numel"] for row in tensor_rows)),
        "state_dict_keys": [str(row["key"]) for row in tensor_rows],
        "state_dict_tensor_rows": tensor_rows,
        "replay_status": "official_model_forward_ran",
    }


def _official_grid_trilinear3d_numeric_replay_row() -> dict[str, Any]:
    """Replay official temporal-only GridTrilinear3D against PyTorch.

    This proves only the standalone grid interpolation primitive.  It is not a
    full HiNeRV source-forward replay because no official checkpoint, patch
    dataset path, decoder, pruning, or torchac bitstream is involved.
    """

    rng = np.random.default_rng(20260603)
    x = rng.normal(size=(3, 2, 4, 5)).astype(np.float32)
    output_size = (7, 2, 4)
    try:
        import torch
        import torch.nn.functional as F

        torch_in = torch.from_numpy(x).reshape(1, 1, x.shape[0], int(np.prod(x.shape[1:])))
        official = (
            F.interpolate(
                torch_in,
                size=(output_size[0], int(np.prod(x.shape[1:]))),
                mode="bilinear",
                align_corners=False,
            )
            .reshape((*output_size, x.shape[-1]))
            .detach()
            .cpu()
            .numpy()
        )
        portable = official_grid_trilinear3d_forward(
            x,
            output_size=output_size,
            align_corners=False,
        ).astype(np.float32)
        max_abs_error = float(np.max(np.abs(official.astype(np.float64) - portable.astype(np.float64))))
        official_sha = _array_sha256(official)
        portable_sha = _array_sha256(portable)
        passed = bool(max_abs_error <= 1.0e-6)
        blockers = [] if passed else ["hinerv_official_grid_trilinear3d_numeric_replay_mismatch"]
        return {
            "schema": "hinerv_official_numeric_subcomponent_replay.v1",
            "component_id": "official_grid_trilinear3d",
            "parent_component_id": "core_hierarchical_renderer",
            "source_forward_parity_proven": passed,
            "full_hinerv_forward_parity_proven": False,
            "backend": "torch_vs_numpy",
            "input_shape": list(x.shape),
            "output_size": list(output_size),
            "align_corners": False,
            "tolerance": 1.0e-6,
            "max_abs_error": max_abs_error,
            "input_sha256": _array_sha256(x),
            "official_output_sha256": official_sha,
            "portable_output_sha256": portable_sha,
            "output_hashes_bit_identical": official_sha == portable_sha,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "schema": "hinerv_official_numeric_subcomponent_replay.v1",
            "component_id": "official_grid_trilinear3d",
            "parent_component_id": "core_hierarchical_renderer",
            "source_forward_parity_proven": False,
            "full_hinerv_forward_parity_proven": False,
            "backend": "torch_vs_numpy",
            "input_shape": list(x.shape),
            "output_size": list(output_size),
            "align_corners": False,
            "tolerance": 1.0e-6,
            "max_abs_error": None,
            "input_sha256": _array_sha256(x),
            "official_output_sha256": None,
            "portable_output_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
            "blockers": ["hinerv_official_grid_trilinear3d_numeric_replay_failed"],
            **FALSE_AUTHORITY,
        }


def _official_patch_index_numeric_replay_row() -> dict[str, Any]:
    """Replay official patch/index formulas against Torch tensor semantics.

    This proves only the standalone official patch/index primitives.  It does
    not prove full HiNeRV source-forward parity because the decoder, feature
    grids, pruning, quantization, and bitstream are not involved.
    """

    video = np.arange(2 * 4 * 6 * 8 * 3, dtype=np.float32).reshape(2, 4, 6, 8, 3)
    patch_size = (2, 3, 4)
    vidx = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    idx = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
    flat_patch_indices = np.array([0, 1, 3, 4, 11], dtype=np.int64)
    try:
        import torch

        torch_video = torch.from_numpy(video)
        n_patch_t, n_patch_h, n_patch_w = (
            torch_video.shape[d + 1] // patch_size[d] for d in range(3)
        )
        official_patch = (
            torch_video.view(
                -1,
                n_patch_t,
                patch_size[0],
                n_patch_h,
                patch_size[1],
                n_patch_w,
                patch_size[2],
                torch_video.shape[-1],
            )
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .view(-1, patch_size[0], patch_size[1], patch_size[2], torch_video.shape[-1])
            .numpy()
        )
        torch_patch = torch.from_numpy(official_patch)
        video_size = (4, 6, 8)
        n_patch_t, n_patch_h, n_patch_w = (
            video_size[d] // torch_patch.shape[d + 1] for d in range(3)
        )
        official_roundtrip = (
            torch_patch.view(
                -1,
                n_patch_t,
                n_patch_h,
                n_patch_w,
                torch_patch.shape[1],
                torch_patch.shape[2],
                torch_patch.shape[3],
                torch_patch.shape[-1],
            )
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .contiguous()
            .view(-1, video_size[0], video_size[1], video_size[2], torch_patch.shape[-1])
            .numpy()
        )

        torch_vidx = torch.from_numpy(vidx)
        scales = [2, 2, 3]
        pidx_t, pidx_h, pidx_w = [
            scales[d] * torch_vidx[:, d][:, None] + torch.arange(scales[d])[None, :]
            for d in range(3)
        ]
        official_pidx = (
            torch.stack(
                [
                    pidx_t[:, :, None, None].expand([torch_vidx.shape[0], *scales]),
                    pidx_h[:, None, :, None].expand([torch_vidx.shape[0], *scales]),
                    pidx_w[:, None, None, :].expand([torch_vidx.shape[0], *scales]),
                ],
                dim=-1,
            )
            .view(-1, 3)
            .numpy()
        )

        torch_idx = torch.from_numpy(idx)
        idx_max = (2, 3, 4)
        sizes = (4, 9, 8)
        padding = (1, 0, 2)
        patch_sizes = [sizes[d] // idx_max[d] for d in range(3)]
        padded = [patch_sizes[d] + padding[d] * 2 for d in range(3)]
        raw_pixel_idx = [
            torch_idx[:, d][:, None] * patch_sizes[d]
            - padding[d]
            + torch.arange(padded[d])[None, :]
            for d in range(3)
        ]
        official_pixel_idx = [
            torch.clip(raw_pixel_idx[d], 0, sizes[d] - 1).numpy()
            for d in range(3)
        ]
        official_pixel_masks = [
            ((raw_pixel_idx[d] >= 0) * (raw_pixel_idx[d] < sizes[d])).numpy()
            for d in range(3)
        ]

        flat = torch.from_numpy(flat_patch_indices)
        official_thw = torch.stack(
            [
                flat // (2 * 3),
                (flat % (2 * 3)) // 3,
                (flat % (2 * 3)) % 3,
            ],
            dim=-1,
        ).numpy()

        portable_patch = official_video_to_patch(video, patch_size=patch_size)
        portable_roundtrip = official_patch_to_video(portable_patch, video_size=video_size)
        portable_pidx = official_vidx_to_pidx(
            vidx,
            vidx_max=(2, 2, 2),
            pidx_max=(4, 4, 6),
        )
        portable_pixel = official_compute_pixel_idx_3d(
            idx,
            idx_max=idx_max,
            sizes=sizes,
            padding=padding,
            clipped=True,
            return_mask=True,
        )
        portable_thw = official_flat_patch_index_to_thw(
            flat_patch_indices,
            num_patches=(2, 2, 3),
        )

        max_abs_error = float(
            max(
                np.max(np.abs(official_patch.astype(np.float64) - portable_patch.astype(np.float64))),
                np.max(
                    np.abs(
                        official_roundtrip.astype(np.float64) - portable_roundtrip.astype(np.float64)
                    )
                ),
                np.max(np.abs(official_pidx.astype(np.float64) - portable_pidx.astype(np.float64))),
                np.max(
                    [
                        np.max(
                            np.abs(
                                official_pixel_idx[axis].astype(np.float64)
                                - portable_pixel.pixel_indices[axis].astype(np.float64)
                            )
                        )
                        for axis in range(3)
                    ]
                ),
                np.max(
                    [
                        np.max(
                            np.abs(
                                official_pixel_masks[axis].astype(np.int64)
                                - (portable_pixel.masks or ())[axis].astype(np.int64)
                            )
                        )
                        for axis in range(3)
                    ]
                ),
                np.max(np.abs(official_thw.astype(np.float64) - portable_thw.astype(np.float64))),
            )
        )
        official_hash_payload = np.concatenate(
            [
                official_patch.reshape(-1).astype(np.float32),
                official_roundtrip.reshape(-1).astype(np.float32),
                official_pidx.reshape(-1).astype(np.float32),
                *(arr.reshape(-1).astype(np.float32) for arr in official_pixel_idx),
                *(arr.reshape(-1).astype(np.float32) for arr in official_pixel_masks),
                official_thw.reshape(-1).astype(np.float32),
            ]
        )
        portable_hash_payload = np.concatenate(
            [
                portable_patch.reshape(-1).astype(np.float32),
                portable_roundtrip.reshape(-1).astype(np.float32),
                portable_pidx.reshape(-1).astype(np.float32),
                *(arr.reshape(-1).astype(np.float32) for arr in portable_pixel.pixel_indices),
                *(arr.reshape(-1).astype(np.float32) for arr in (portable_pixel.masks or ())),
                portable_thw.reshape(-1).astype(np.float32),
            ]
        )
        official_sha = _array_sha256(official_hash_payload)
        portable_sha = _array_sha256(portable_hash_payload)
        passed = bool(max_abs_error == 0.0 and official_sha == portable_sha)
        blockers = [] if passed else ["hinerv_official_patch_index_numeric_replay_mismatch"]
        return {
            "schema": "hinerv_official_numeric_subcomponent_replay.v1",
            "component_id": "official_patch_index_path",
            "parent_component_id": "patch_dataset_path",
            "source_forward_parity_proven": passed,
            "full_hinerv_forward_parity_proven": False,
            "backend": "torch_formula_vs_numpy",
            "video_shape": list(video.shape),
            "patch_size": list(patch_size),
            "tolerance": 0.0,
            "max_abs_error": max_abs_error,
            "input_sha256": _array_sha256(video),
            "official_output_sha256": official_sha,
            "portable_output_sha256": portable_sha,
            "output_hashes_bit_identical": official_sha == portable_sha,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "schema": "hinerv_official_numeric_subcomponent_replay.v1",
            "component_id": "official_patch_index_path",
            "parent_component_id": "patch_dataset_path",
            "source_forward_parity_proven": False,
            "full_hinerv_forward_parity_proven": False,
            "backend": "torch_formula_vs_numpy",
            "video_shape": list(video.shape),
            "patch_size": list(patch_size),
            "tolerance": 0.0,
            "max_abs_error": None,
            "input_sha256": _array_sha256(video),
            "official_output_sha256": None,
            "portable_output_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
            "blockers": ["hinerv_official_patch_index_numeric_replay_failed"],
            **FALSE_AUTHORITY,
        }


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _stable_json_sha256(payload: Any) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _git_head_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", root.as_posix(), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _next_actions(blockers: Sequence[str]) -> list[str]:
    actions: list[str] = []
    if any("source_files_missing" in blocker for blocker in blockers):
        actions.append("recover the official HiNeRV checkout before spending source-parity work")
    if any("source_marker_missing" in blocker for blocker in blockers):
        actions.append("refresh the official source marker map from the pinned HiNeRV checkout")
    if "hinerv_local_receiver_bindings_missing" in blockers:
        actions.append("bind local HiNeRV feature-grid, ConvNeXt, pruning, QuantNoise, and bitstream controls")
    if "hinerv_official_forward_parity_missing" in blockers:
        actions.append(
            "run tiny official HiNeRV torch forward versus local portable/MLX replay with weight/input/output SHA evidence"
        )
    return actions or ["official HiNeRV source audit has no source-parity blockers; keep false-authority flags"]


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_not_none(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


__all__ = [
    "AUTHORITY",
    "FORWARD_PARITY_ARTIFACT_SCHEMA",
    "SCHEMA",
    "build_hinerv_official_forward_parity_artifact",
    "build_hinerv_official_source_parity_audit",
    "render_hinerv_official_source_parity_markdown",
    "summarize_hinerv_official_source_audit",
]
