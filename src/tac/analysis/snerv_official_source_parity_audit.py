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
)

LOCAL_OFFICIAL_PARITY_MARKERS: tuple[str, ...] = ("SNERV_OFFICIAL_MFU_HFR_TUB_PARITY_PROOF",)


def build_snerv_official_source_parity_audit(
    *,
    official_repo_dir: str | Path,
    repo_root: str | Path,
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
    official_markers_present = not missing_files and all(row["all_markers_present"] for row in official_group_rows)
    local_receiver_safe_adapter_present = bool(local_receiver_safe_row["all_markers_present"])
    official_parity_proven = bool(
        official_markers_present
        and local_receiver_safe_adapter_present
        and local_official_parity_row["all_markers_present"]
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
        "missing_markers": missing,
        "all_markers_present": not missing,
    }


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
    "OFFICIAL_MARKER_GROUPS",
    "OFFICIAL_REPO_URL",
    "REQUIRED_OFFICIAL_FILES",
    "SCHEMA",
    "build_snerv_official_source_parity_audit",
    "render_snerv_official_source_parity_markdown",
    "summarize_snerv_official_source_audit",
]
