# SPDX-License-Identifier: MIT
"""Source-backed SNeRV official MFU/HFR/TUB parity audit.

The local SNeRV carrier is a receiver-safe NumPy/MLX contest adapter. That is
useful, but it is not the same claim as "source-faithful to official SNeRV".
This module records what the official OSS source actually contains, hashes the
files used for that conclusion, and then compares it to the local proof surface.

The audit is deliberately false-authority: it can close ambiguity and route
implementation work, but it cannot promote a score, launch exact auth, or prove
receiver parity by marker match alone.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from tac.analysis.source_marker_scan import read_python_source_for_marker_scan

SCHEMA = "snerv_official_source_parity_audit.v1"
AUTHORITY = "false_authority_source_audit_no_score_claim"
FORWARD_PARITY_ARTIFACT_SCHEMA = "snerv_official_mfu_hfr_tub_forward_parity.v1"
OFFICIAL_REPO_URL = "https://github.com/qwertja/SNeRV"
OFFICIAL_REPO_URL_GIT = "https://github.com/qwertja/SNeRV.git"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "production_hardened_claim": False,
    "source_faithful_stack_claim": False,
    "ready_for_exact_eval_dispatch": False,
}

REQUIRED_OFFICIAL_FILES: tuple[str, ...] = (
    "model/snerv.py",
    "model/snerv_t.py",
    "model/layers.py",
    "train_snerv.py",
)


@dataclass(frozen=True)
class MarkerGroup:
    """A source marker group that should be present in official OSS."""

    group_id: str
    description: str
    source_files: tuple[str, ...]
    markers: tuple[str, ...]


OFFICIAL_MARKER_GROUPS: tuple[MarkerGroup, ...] = (
    MarkerGroup(
        group_id="official_haar_dwt_lf_hf_split",
        description="Official SNeRV analyzes LF/HF with Haar DWT and reconstructs with IDWT.",
        source_files=("model/snerv.py",),
        markers=(
            "DWT",
            "IDWT",
            "haar",
            "periodization",
            "yl_norm",
            "yh_out",
        ),
    ),
    MarkerGroup(
        group_id="official_mfu_multi_resolution_fusion",
        description="Official SNeRV fuses multi-resolution decoder embeddings before the LF/HF heads.",
        source_files=("model/snerv.py",),
        markers=(
            "ConvTranspose2d",
            "RB",
            "torch.cat",
            "embed_list",
            "pyr_out",
            "decoder_len+3",
            "decoder_len+4",
            "decoder_len+5",
            "decoder_len+6",
        ),
    ),
    MarkerGroup(
        group_id="official_hfr_high_frequency_restoration",
        description="Official SNeRV predicts LH/HL/HH heads from the fused pyramid output.",
        source_files=("model/snerv.py",),
        markers=(
            "lh_out",
            "hl_out",
            "hh_out",
            "torch.stack",
            "HF_in",
            "idwt",
        ),
    ),
    MarkerGroup(
        group_id="official_tub_temporal_extension",
        description="Official SNeRV_T binds neighbor-frame temporal DWT features into the decoder.",
        source_files=("model/snerv_t.py",),
        markers=(
            "SNeRV_T",
            "DWT1D",
            "input_p",
            "input_n",
            "embed_hv_p",
            "embed_hv_n",
            "temp_emb_layer",
            "UpsampleBlock",
        ),
    ),
    MarkerGroup(
        group_id="official_modelsize_fc_dim_solver",
        description="Official training solves fc_dim from --modelsize and the parameter budget.",
        source_files=("train_snerv.py",),
        markers=(
            "--modelsize",
            "--fc_dim",
            "embed_param",
            "decoder_size",
            "np.roots",
            "args.fc_dim",
        ),
    ),
    MarkerGroup(
        group_id="official_quantized_payload_controls",
        description="Official training exposes separate model and embedding quantization controls.",
        source_files=("train_snerv.py",),
        markers=(
            "quant_model_bit",
            "quant_embed_bit",
            "quant_embed2_bit",
            "quant_axis",
        ),
    ),
)

LOCAL_RECEIVER_SAFE_MARKERS: tuple[str, ...] = (
    "MultiResolutionFusionUnit",
    "HighFrequencyRestorer",
    "SnervTemporalExtension",
    "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
    "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
)

LOCAL_OFFICIAL_PARITY_MARKERS: tuple[str, ...] = ("SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF",)

_SNERV_FORWARD_PARITY_COMPONENT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "component_id": "mfu",
        "official_group_id": "official_mfu_multi_resolution_fusion",
        "official_source_markers": (
            ("model/snerv.py", "upsample_5 = nn.ConvTranspose2d"),
            ("model/snerv.py", "decoder_layer5 = RB("),
            ("model/snerv.py", "upsample_6 = nn.ConvTranspose2d"),
            ("model/snerv.py", "decoder_layer6 = RB("),
            ("model/snerv.py", "up1 = self.decoder[self.decoder_len+3](embed_list[-3])"),
            ("model/snerv.py", "unet1 = self.decoder[self.decoder_len+4](torch.cat([up1, embed_list[-2]], dim=1))"),
            ("model/snerv.py", "pyr_out = self.decoder[self.decoder_len+6](torch.cat([unet1_up, embed_list[-1]], dim=1))"),
        ),
        "local_receiver_markers": (
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "class MultiResolutionFusionUnit"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "_box_pool_upsample"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "_central_gradients"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "_patch_features"),
        ),
        "local_source_forward_markers": (
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "nn.ConvTranspose2d"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "decoder_layer5 = RB("),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "torch.cat([up1, embed_list[-2]]"),
        ),
        "classification": "source_forward_parity_falsified_receiver_safe_analogue_only",
        "blocker": "snerv_mfu_source_forward_parity_falsified_receiver_safe_analogue_only",
        "evidence_summary": (
            "Official MFU is a learned ConvTranspose2d/RB pyramid over decoder embed_list; "
            "local MFU is a deterministic NumPy LF feature bank with pooled, gradient, and patch bases."
        ),
    },
    {
        "component_id": "hfr",
        "official_group_id": "official_hfr_high_frequency_restoration",
        "official_source_markers": (
            ("model/snerv.py", "decoder_layer2 = ConvBlock"),
            ("model/snerv.py", "decoder_layer3 = ConvBlock"),
            ("model/snerv.py", "decoder_layer4 = ConvBlock"),
            ("model/snerv.py", "HF_in = pyr_out"),
            ("model/snerv.py", "lh_out = self.decoder[self.decoder_len](HF_in)"),
            ("model/snerv.py", "hl_out = self.decoder[self.decoder_len+1](HF_in)"),
            ("model/snerv.py", "hh_out = self.decoder[self.decoder_len+2](HF_in)"),
            ("model/snerv.py", "yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)"),
        ),
        "local_receiver_markers": (
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "class HighFrequencyRestorer"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "context - _box_pool_upsample(context, 3)"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "_central_gradients(edge)"),
        ),
        "local_source_forward_markers": (
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "decoder_layer2 = ConvBlock"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "HF_in = pyr_out"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "lh_out = self.decoder[self.decoder_len](HF_in)"),
        ),
        "classification": "source_forward_parity_falsified_receiver_safe_analogue_only",
        "blocker": "snerv_hfr_source_forward_parity_falsified_receiver_safe_analogue_only",
        "evidence_summary": (
            "Official HFR is three learned ConvBlock detail heads fed by pyr_out; "
            "local HFR is deterministic LF-edge correction added to stored or generated detail subbands."
        ),
    },
    {
        "component_id": "tub",
        "official_group_id": "official_tub_temporal_extension",
        "official_source_markers": (
            ("model/snerv_t.py", "DWT1D(J=1, wave='haar', mode='periodization')"),
            ("model/snerv_t.py", "embed_lv_p, embed_hv_p"),
            ("model/snerv_t.py", "embed_lv_n, embed_hv_n"),
            ("model/snerv_t.py", "embed_hv_p = self.encoder[1]((embed_lv_p"),
            ("model/snerv_t.py", "embed_hv_n = self.encoder[2]((embed_lv_n"),
            ("model/snerv_t.py", "output_2 = self.decoder[self.decoder_len-1]"),
            ("model/snerv_t.py", "output = layer(output, output_2)"),
        ),
        "local_receiver_markers": (
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "class SnervTemporalExtension"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "official_haar_dwt1d_lowpass_features"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "(center + prev) * inv_two_sqrt2"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "(center + nxt) * inv_two_sqrt2"),
        ),
        "local_source_forward_markers": (
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "self.encoder[1]((embed_lv_p"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "self.decoder[self.decoder_len-1]"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "output = layer(output, output_2)"),
        ),
        "primitive_parity_markers": (
            ("model/snerv_t.py", "DWT1D(J=1, wave='haar', mode='periodization')"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "1.0 / (2.0 * np.sqrt(2.0))"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "(center + prev) * inv_two_sqrt2"),
            ("src/tac/substrates/snerv_inverse_steg_carrier/carrier.py", "(center + nxt) * inv_two_sqrt2"),
        ),
        "classification": "official_haar_temporal_lowpass_primitive_proven_full_tub_falsified",
        "blocker": "snerv_tub_full_source_forward_parity_falsified_primitive_only",
        "evidence_summary": (
            "Local TUB exposes the official Haar lowpass divided by two algebra, "
            "but it does not implement the official temporal encoders or output_2 decoder fusion graph."
        ),
    },
)


def build_snerv_official_source_parity_audit(
    *,
    official_repo_dir: str | Path,
    repo_root: str | Path,
    official_forward_parity_artifact_path: str | Path | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority source audit for official SNeRV controls."""

    official_root = Path(official_repo_dir)
    local_root = Path(repo_root)
    if generated_utc is None:
        generated_utc = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_rows = [_file_row(official_root, rel_path) for rel_path in REQUIRED_OFFICIAL_FILES]
    missing_files = [row["rel_path"] for row in file_rows if row["status"] != "present"]
    official_group_rows = [_official_marker_group_row(official_root, group) for group in OFFICIAL_MARKER_GROUPS]
    local_receiver_safe_row = _local_marker_row(
        local_root,
        marker_group_id="local_receiver_safe_mfu_hfr_temporal_adapter",
        markers=LOCAL_RECEIVER_SAFE_MARKERS,
    )
    local_official_parity_row = _local_marker_row(
        local_root,
        marker_group_id="local_official_mfu_hfr_tub_parity_proof",
        markers=LOCAL_OFFICIAL_PARITY_MARKERS,
    )
    forward_parity_artifact_row = _forward_parity_artifact_row(
        official_forward_parity_artifact_path
    )
    official_markers_present = not missing_files and all(row["all_markers_present"] for row in official_group_rows)
    local_receiver_safe_adapter_present = bool(local_receiver_safe_row["all_markers_present"])
    official_parity_proven = bool(
        official_markers_present
        and local_receiver_safe_adapter_present
        and local_official_parity_row["all_markers_present"]
        and forward_parity_artifact_row["parity_passed"]
    )
    component_state_rows = _component_state_rows(
        official_group_rows=official_group_rows,
        local_receiver_safe_row=local_receiver_safe_row,
        local_official_parity_row=local_official_parity_row,
        forward_parity_artifact_row=forward_parity_artifact_row,
    )
    blockers = _ordered_unique(
        [
            "snerv_official_source_files_missing" if missing_files else "",
            *[
                f"snerv_official_source_marker_missing:{row['group_id']}"
                for row in official_group_rows
                if not row["all_markers_present"]
            ],
            ("snerv_receiver_safe_mfu_hfr_temporal_adapter_missing" if not local_receiver_safe_adapter_present else ""),
            ("snerv_official_mfu_hfr_tub_parity_missing" if not official_parity_proven else ""),
        ]
    )
    head_sha = _git_head_sha(official_root)
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "generated_utc": generated_utc,
        "family": "snerv",
        "official_repo": {
            "repo_url": OFFICIAL_REPO_URL,
            "repo_url_git": OFFICIAL_REPO_URL_GIT,
            "root": official_root.as_posix(),
            "head_sha": head_sha,
            "required_files": list(REQUIRED_OFFICIAL_FILES),
        },
        "local_repo_root": local_root.as_posix(),
        "official_file_rows": file_rows,
        "official_marker_group_rows": official_group_rows,
        "local_receiver_safe_marker_row": local_receiver_safe_row,
        "local_official_parity_marker_row": local_official_parity_row,
        "official_forward_parity_artifact_row": forward_parity_artifact_row,
        "component_state_rows": component_state_rows,
        "official_source_markers_present": official_markers_present,
        "local_receiver_safe_adapter_present": local_receiver_safe_adapter_present,
        "official_mfu_hfr_tub_parity_proven": official_parity_proven,
        "blockers": blockers,
        "next_actions": _next_actions(blockers),
        **FALSE_AUTHORITY,
    }


def summarize_snerv_official_source_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact row-safe summary of a SNeRV official-source audit."""

    official_repo = report.get("official_repo")
    if not isinstance(official_repo, Mapping):
        official_repo = {}
    return {
        "schema": str(report.get("schema") or ""),
        "authority": str(report.get("authority") or ""),
        "family": str(report.get("family") or "snerv"),
        "official_repo_root": str(official_repo.get("root") or ""),
        "official_repo_url": str(official_repo.get("repo_url") or ""),
        "official_head_sha": official_repo.get("head_sha"),
        "official_source_markers_present": bool(report.get("official_source_markers_present")),
        "local_receiver_safe_adapter_present": bool(report.get("local_receiver_safe_adapter_present")),
        "official_mfu_hfr_tub_parity_proven": bool(report.get("official_mfu_hfr_tub_parity_proven")),
        "component_states": [
            {
                "component_id": row.get("component_id"),
                "classification": row.get("classification"),
                "source_forward_parity_proven": bool(
                    row.get("source_forward_parity_proven")
                ),
            }
            for row in report.get("component_state_rows") or ()
            if isinstance(row, Mapping)
        ],
        "blockers": list(report.get("blockers") or ()),
        **FALSE_AUTHORITY,
    }


def render_snerv_official_source_parity_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing Markdown audit."""

    official_repo = report.get("official_repo")
    if not isinstance(official_repo, Mapping):
        official_repo = {}
    lines = [
        "# SNeRV Official Source-Parity Audit",
        "",
        f"Schema: `{report.get('schema')}`",
        f"Authority: `{report.get('authority')}`",
        f"Official repo: `{official_repo.get('root')}`",
        f"Official head SHA: `{official_repo.get('head_sha')}`",
        "",
        "## Verdict",
        "",
        f"- official source markers present: `{bool(report.get('official_source_markers_present'))}`",
        f"- local receiver-safe adapter present: `{bool(report.get('local_receiver_safe_adapter_present'))}`",
        f"- official MFU/HFR/TUB parity proven: `{bool(report.get('official_mfu_hfr_tub_parity_proven'))}`",
        f"- score claim: `{bool(report.get('score_claim'))}`",
        "",
        "## Component States",
        "",
        "| component | classification | receiver analogue | official forward parity | blockers |",
        "|---|---|---:|---:|---|",
    ]
    for row in report.get("component_state_rows") or ():
        if not isinstance(row, Mapping):
            continue
        blockers = ", ".join(str(v) for v in row.get("blockers") or ())
        lines.append(
            f"| `{row.get('component_id')}` | `{row.get('classification')}` | "
            f"`{bool(row.get('receiver_safe_analogue_present'))}` | "
            f"`{bool(row.get('source_forward_parity_proven'))}` | `{blockers}` |"
        )
    lines.extend(
        [
        "## Marker Groups",
        "",
        "| group | present | missing |",
        "|---|---:|---|",
        ]
    )
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
    text = "\n".join(read_python_source_for_marker_scan(root / rel_path) for rel_path in group.source_files)
    missing = [marker for marker in group.markers if marker not in text]
    return {
        "group_id": group.group_id,
        "description": group.description,
        "source_files": list(group.source_files),
        "markers": list(group.markers),
        "missing_markers": missing,
        "all_markers_present": not missing,
    }


def _local_marker_row(
    root: Path,
    *,
    marker_group_id: str,
    markers: Sequence[str],
) -> dict[str, Any]:
    source = read_python_source_for_marker_scan(
        root / "src/tac/substrates/snerv_inverse_steg_carrier",
        exclude_names=(),
    )
    missing = [marker for marker in markers if marker not in source]
    return {
        "marker_group_id": marker_group_id,
        "source_root": (root / "src/tac/substrates/snerv_inverse_steg_carrier").as_posix(),
        "markers": list(markers),
        "present_markers": [marker for marker in markers if marker not in missing],
        "missing_markers": missing,
        "all_markers_present": not missing,
    }


def _forward_parity_artifact_row(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "schema": "snerv_official_forward_parity_artifact_row.v1",
            "path": None,
            "status": "missing",
            "artifact_schema": None,
            "bytes": 0,
            "sha256": None,
            "parity_passed": False,
            "blockers": ["snerv_official_forward_parity_artifact_missing"],
            **FALSE_AUTHORITY,
        }
    artifact_path = Path(path)
    if not artifact_path.is_file():
        return {
            "schema": "snerv_official_forward_parity_artifact_row.v1",
            "path": artifact_path.as_posix(),
            "status": "missing",
            "artifact_schema": None,
            "bytes": 0,
            "sha256": None,
            "parity_passed": False,
            "blockers": ["snerv_official_forward_parity_artifact_missing"],
            **FALSE_AUTHORITY,
        }
    data = artifact_path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "schema": "snerv_official_forward_parity_artifact_row.v1",
            "path": artifact_path.as_posix(),
            "status": "unreadable",
            "artifact_schema": None,
            "bytes": len(data),
            "sha256": sha256(data).hexdigest(),
            "parity_passed": False,
            "error": str(exc),
            "blockers": ["snerv_official_forward_parity_artifact_unreadable"],
            **FALSE_AUTHORITY,
        }
    artifact_schema = str(payload.get("schema") or "")
    parity_passed = bool(
        artifact_schema == FORWARD_PARITY_ARTIFACT_SCHEMA
        and payload.get("official_mfu_hfr_tub_forward_parity_passed") is True
        and payload.get("score_claim") is False
        and payload.get("ready_for_exact_eval_dispatch") is False
    )
    blockers = []
    if artifact_schema != FORWARD_PARITY_ARTIFACT_SCHEMA:
        blockers.append("snerv_official_forward_parity_artifact_schema_invalid")
    if payload.get("official_mfu_hfr_tub_forward_parity_passed") is not True:
        blockers.append("snerv_official_forward_parity_artifact_not_passing")
    if payload.get("score_claim") is not False:
        blockers.append("snerv_official_forward_parity_artifact_score_claim_not_false")
    if payload.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("snerv_official_forward_parity_artifact_exact_flag_not_false")
    return {
        "schema": "snerv_official_forward_parity_artifact_row.v1",
        "path": artifact_path.as_posix(),
        "status": "present",
        "artifact_schema": artifact_schema,
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
        "parity_passed": parity_passed,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _component_state_rows(
    *,
    official_group_rows: Sequence[Mapping[str, Any]],
    local_receiver_safe_row: Mapping[str, Any],
    local_official_parity_row: Mapping[str, Any],
    forward_parity_artifact_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    official_by_group = {str(row.get("group_id")): row for row in official_group_rows}
    local_present = {str(v) for v in local_receiver_safe_row.get("present_markers") or ()}
    parity_marker_present = bool(local_official_parity_row.get("all_markers_present"))
    artifact_passed = bool(forward_parity_artifact_row.get("parity_passed"))

    def build_row(
        *,
        component_id: str,
        official_group_id: str,
        local_markers: tuple[str, ...],
        analogue_classification: str,
    ) -> dict[str, Any]:
        official_row = official_by_group.get(official_group_id, {})
        official_markers_present = bool(official_row.get("all_markers_present"))
        receiver_safe = all(marker in local_present for marker in local_markers)
        source_forward = bool(
            official_markers_present
            and receiver_safe
            and parity_marker_present
            and artifact_passed
        )
        if source_forward:
            classification = "official_source_forward_parity_proven"
        elif receiver_safe:
            classification = analogue_classification
        else:
            classification = "missing_or_partial"
        blockers = _ordered_unique(
            [
                (
                    f"snerv_official_source_marker_missing:{official_group_id}"
                    if not official_markers_present
                    else ""
                ),
                (
                    f"snerv_{component_id}_receiver_safe_adapter_missing"
                    if not receiver_safe
                    else ""
                ),
                (
                    "snerv_official_mfu_hfr_tub_parity_marker_missing"
                    if not parity_marker_present
                    else ""
                ),
                (
                    "snerv_official_forward_parity_artifact_missing_or_failed"
                    if not artifact_passed
                    else ""
                ),
            ]
        )
        return {
            "schema": "snerv_official_component_state.v1",
            "component_id": component_id,
            "official_group_id": official_group_id,
            "official_markers_present": official_markers_present,
            "receiver_safe_analogue_present": receiver_safe,
            "local_receiver_markers": list(local_markers),
            "local_parity_marker_present": parity_marker_present,
            "forward_parity_artifact_passed": artifact_passed,
            "source_forward_parity_proven": source_forward,
            "classification": classification,
            "blockers": blockers,
            **FALSE_AUTHORITY,
        }

    return [
        build_row(
            component_id="mfu",
            official_group_id="official_mfu_multi_resolution_fusion",
            local_markers=(
                "MultiResolutionFusionUnit",
                "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
            ),
            analogue_classification="receiver_safe_analogue",
        ),
        build_row(
            component_id="hfr",
            official_group_id="official_hfr_high_frequency_restoration",
            local_markers=(
                "HighFrequencyRestorer",
                "SNERV_MFU_HFR_TEMPORAL_RECEIVER_PROOF",
            ),
            analogue_classification="receiver_safe_analogue",
        ),
        build_row(
            component_id="tub",
            official_group_id="official_tub_temporal_extension",
            local_markers=(
                "SnervTemporalExtension",
                "SNERV_OFFICIAL_TEMPORAL_HAAR_DWT1D_PROOF",
            ),
            analogue_classification="official_haar_primitive_only",
        ),
    ]


def _git_head_sha(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", path.as_posix(), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    head = result.stdout.strip()
    return head or None


def _next_actions(blockers: Sequence[str]) -> list[str]:
    blockers_set = set(blockers)
    actions: list[str] = []
    if "snerv_official_source_files_missing" in blockers_set or any(
        blocker.startswith("snerv_official_source_marker_missing:") for blocker in blockers_set
    ):
        actions.append("refresh the official SNeRV checkout and rerun the audit before source-parity work")
    if "snerv_receiver_safe_mfu_hfr_temporal_adapter_missing" in blockers_set:
        actions.append("restore the local receiver-safe MFU/HFR/SNeRV_T adapter before campaign launch")
    if "snerv_official_mfu_hfr_tub_parity_missing" in blockers_set:
        actions.append(
            "implement source-forward official MFU/HFR/TUB behavior proof or explicitly supersede it with same-axis receiver evidence"
        )
    if not actions:
        actions.append("official source markers and local parity proof are present")
    return actions


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "AUTHORITY",
    "FALSE_AUTHORITY",
    "FORWARD_PARITY_ARTIFACT_SCHEMA",
    "OFFICIAL_MARKER_GROUPS",
    "OFFICIAL_REPO_URL",
    "REQUIRED_OFFICIAL_FILES",
    "SCHEMA",
    "build_snerv_official_source_parity_audit",
    "render_snerv_official_source_parity_markdown",
    "summarize_snerv_official_source_audit",
]
