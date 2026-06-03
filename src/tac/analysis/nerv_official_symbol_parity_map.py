# SPDX-License-Identifier: MIT
"""Official-to-local symbol maps for SNeRV and HiNeRV.

The map is a narrow source-fidelity artifact: it proves that the local code has
named binding surfaces for the official OSS features we keep referring to.  It
does not prove source-forward parity, trained weight compatibility, receiver
payload closure, or contest score authority.
"""

from __future__ import annotations

import importlib
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.source_marker_scan import read_python_source_for_marker_scan
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

SCHEMA = "nerv_official_symbol_parity_map.v1"
AUTHORITY = "false_authority_symbol_map_no_source_forward_or_score_claim"
DEFAULT_OSS_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "oss_nerv_source_audit_20260602T113720Z/repos"
)
DEFAULT_SOURCE_ROOTS = {
    "snerv": DEFAULT_OSS_ROOT / "SNeRV",
    "hi_nerv": DEFAULT_OSS_ROOT / "HiNeRV",
}


@dataclass(frozen=True)
class SymbolBinding:
    """One official feature mapped to local executable symbols."""

    family: str
    feature_id: str
    official_repo: str
    official_head_sha: str
    official_paths: tuple[str, ...]
    official_markers: tuple[str, ...]
    local_bindings: tuple[tuple[str, str], ...]
    still_blocked_by: tuple[str, ...]


def build_nerv_official_symbol_parity_map(
    *,
    repo_root: str | Path | None = None,
    families: Iterable[str] = ("hi_nerv", "snerv"),
    source_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Return a false-authority official-to-local symbol map."""

    selected = tuple(
        dict.fromkeys(str(family).strip() for family in families if str(family).strip())
    )
    roots = {
        family: Path(source_roots[family]).expanduser()
        if source_roots and family in source_roots
        else DEFAULT_SOURCE_ROOTS.get(family)
        for family in selected
    }
    specs = [spec for spec in _binding_specs() if spec.family in selected]
    rows = [_binding_row(spec, source_root=roots.get(spec.family)) for spec in specs]
    family_rows = [
        _family_row(family, [row for row in rows if row["family"] == family])
        for family in selected
    ]
    blocking = _ordered_unique(
        blocker
        for row in rows
        for blocker in row["blockers"]
        if blocker.endswith("_local_symbol_missing")
    )
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repo_root": None if repo_root is None else Path(repo_root).as_posix(),
        "families": selected,
        "family_rows": family_rows,
        "symbol_rows": rows,
        "local_symbol_map_ready": bool(rows) and not blocking,
        "source_pins_verified": bool(rows)
        and all(row["source_marker_status"] == "verified" for row in rows),
        "blockers": blocking,
        **FALSE_AUTHORITY,
    }


def _binding_specs() -> tuple[SymbolBinding, ...]:
    return (
        SymbolBinding(
            family="snerv",
            feature_id="snerv_modelsize_fc_dim_solver",
            official_repo="https://github.com/qwertja/SNeRV.git",
            official_head_sha="0844a08f9591eea9625f8b961ed91d08030e06d1",
            official_paths=("train_snerv.py",),
            official_markers=("--modelsize", "--fc_dim", "np.roots"),
            local_bindings=(
                (
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "official_snerv_modelsize_to_fc_dim",
                ),
                (
                    "tac.substrates.snerv_inverse_steg_carrier.carrier",
                    "SNERV_OFFICIAL_MODELSIZE_TO_FC_DIM_PROOF",
                ),
            ),
            still_blocked_by=(
                "snerv_measured_fc_dim_modelsize_ladder_missing",
                "snerv_official_stride_stack_parity_missing",
            ),
        ),
        SymbolBinding(
            family="snerv",
            feature_id="snerv_mfu_hfr_tub_official_primitives",
            official_repo="https://github.com/qwertja/SNeRV.git",
            official_head_sha="0844a08f9591eea9625f8b961ed91d08030e06d1",
            official_paths=("model/snerv.py", "model/snerv_t.py", "model/layers.py"),
            official_markers=("HF_in = pyr_out", "DWT1D", "ConvBlock"),
            local_bindings=(
                (
                    "tac.substrates.snerv_inverse_steg_carrier.official_mfu",
                    "OfficialSnervMfu",
                ),
                (
                    "tac.substrates.snerv_inverse_steg_carrier.official_hfr",
                    "OfficialHfrHeads",
                ),
                (
                    "tac.substrates.snerv_inverse_steg_carrier.official_tub",
                    "prepare_official_tub_graph_inputs",
                ),
            ),
            still_blocked_by=(
                "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
            ),
        ),
        SymbolBinding(
            family="snerv",
            feature_id="snerv_quantized_checkpoint_payload",
            official_repo="https://github.com/qwertja/SNeRV.git",
            official_head_sha="0844a08f9591eea9625f8b961ed91d08030e06d1",
            official_paths=("train_snerv.py",),
            official_markers=("quant_model", "quant_vid.pth", "HuffmanCodec"),
            local_bindings=(
                (
                    "tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export",
                    "_official_primitives_export_binding",
                ),
            ),
            still_blocked_by=(
                "snerv_quantized_checkpoint_payload_replay_missing",
                "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
            ),
        ),
        SymbolBinding(
            family="hi_nerv",
            feature_id="hi_nerv_core_hierarchical_renderer",
            official_repo="https://github.com/hmkx/HiNeRV.git",
            official_head_sha="fdb92ec22492246f800621dfd454f6a5c62ab75b",
            official_paths=("models/hinerv.py", "models/layers.py"),
            official_markers=("class HiNeRV", "FeatureGrid", "GridTrilinear3D"),
            local_bindings=(
                ("tac.substrates.hi_nerv.architecture", "HinervSubstrate"),
                ("tac.substrates.hi_nerv.architecture", "HierarchicalFeatureGrid"),
                ("tac.substrates.hi_nerv.architecture", "trilinear_upsample"),
            ),
            still_blocked_by=(
                "hi_nerv_tiny_forward_parity_against_oss_missing",
                "hi_nerv_partial_local_three_scale_latent_pyramid_not_official_feature_grid",
            ),
        ),
        SymbolBinding(
            family="hi_nerv",
            feature_id="hi_nerv_convnext_patch_bitstream_pipeline",
            official_repo="https://github.com/hmkx/HiNeRV.git",
            official_head_sha="fdb92ec22492246f800621dfd454f6a5c62ab75b",
            official_paths=(
                "models/layers.py",
                "datasets.py",
                "compression/quant_utils.py",
                "compression/codec_utils.py",
                "compression/prune_utils.py",
            ),
            official_markers=(
                "ConvNeXtBlock",
                "--patch-size",
                "QuantNoise",
                "torchac",
                "PruningMask",
            ),
            local_bindings=(
                ("tac.substrates.hi_nerv.architecture", "ConvNeXtBlock"),
                ("tac.substrates.hi_nerv.bitstream", "apply_decoder_pruning"),
                ("tac.substrates.hi_nerv.bitstream", "apply_decoder_quant_noise"),
                (
                    "tac.substrates.hi_nerv.bitstream",
                    "measure_hi_nerv_decoder_bitstream_roundtrip",
                ),
            ),
            still_blocked_by=(
                "hi_nerv_missing_patch_frame_equivalence_proof",
                "hi_nerv_missing_official_3d_upsampling_parity",
                "hi_nerv_grouped_intN_zero_run_packet_layout_missing",
            ),
        ),
        SymbolBinding(
            family="hi_nerv",
            feature_id="hi_nerv_modelsize_config_family",
            official_repo="https://github.com/hmkx/HiNeRV.git",
            official_head_sha="fdb92ec22492246f800621dfd454f6a5c62ab75b",
            official_paths=("cfgs/models", "cfgs/train"),
            official_markers=(
                "--model HiNeRV",
                "--base-grid-size",
                "--enc-grid-level",
                "--patch-size",
                "--quant-level",
            ),
            local_bindings=(
                ("tac.analysis.nerv_modelsize_budget", "enumerate_hinerv_modelsize_candidates"),
                ("tac.analysis.hinerv_archive_size_ladder", "build_hinerv_archive_size_ladder"),
            ),
            still_blocked_by=(
                "hi_nerv_measured_modelsize_budget_ladder_missing",
                "hi_nerv_missing_measured_config_family_ladder",
            ),
        ),
    )


def _binding_row(
    spec: SymbolBinding,
    *,
    source_root: Path | None,
) -> dict[str, Any]:
    local_rows = [
        _local_symbol_row(module_name, symbol_name)
        for module_name, symbol_name in spec.local_bindings
    ]
    source_status = _source_marker_status(spec, source_root=source_root)
    local_missing = [
        row["symbol_ref"] for row in local_rows if row["status"] != "present"
    ]
    blockers = list(spec.still_blocked_by)
    if local_missing:
        blockers.append(f"{spec.family}_{spec.feature_id}_local_symbol_missing")
    return {
        "family": spec.family,
        "feature_id": spec.feature_id,
        "official_repo": spec.official_repo,
        "official_head_sha": spec.official_head_sha,
        "official_paths": list(spec.official_paths),
        "official_markers": list(spec.official_markers),
        "source_marker_status": source_status["status"],
        "source_marker_blockers": source_status["blockers"],
        "source_root": source_status["source_root"],
        "source_head_sha_observed": source_status["source_head_sha_observed"],
        "local_symbol_rows": local_rows,
        "local_symbols_present": not local_missing,
        "still_blocked_by": list(spec.still_blocked_by),
        "blockers": _ordered_unique(blockers),
        **FALSE_AUTHORITY,
    }


def _local_symbol_row(module_name: str, symbol_name: str) -> dict[str, Any]:
    status = "present"
    error = None
    try:
        module = importlib.import_module(module_name)
        target: Any = module
        for part in str(symbol_name).split("."):
            target = getattr(target, part)
    except Exception as exc:  # pragma: no cover - surfaced in row payload.
        status = "missing"
        error = repr(exc)
    return {
        "module": module_name,
        "symbol": symbol_name,
        "symbol_ref": f"{module_name}.{symbol_name}",
        "status": status,
        "error": error,
    }


def _source_marker_status(
    spec: SymbolBinding,
    *,
    source_root: Path | None,
) -> dict[str, Any]:
    if source_root is None or not source_root.exists():
        return {
            "status": "source_checkout_unavailable",
            "blockers": ["official_source_checkout_unavailable_for_symbol_scan"],
            "source_root": None if source_root is None else source_root.as_posix(),
            "source_head_sha_observed": None,
        }
    observed_sha = _git_head(source_root)
    blockers: list[str] = []
    if observed_sha != spec.official_head_sha:
        blockers.append("official_source_head_sha_mismatch")
    text = "\n".join(
        _read_source_for_marker_scan(source_root / rel_path)
        if (source_root / rel_path).exists()
        else ""
        for rel_path in spec.official_paths
    )
    missing = [marker for marker in spec.official_markers if marker not in text]
    blockers.extend(f"official_source_marker_missing:{marker}" for marker in missing)
    return {
        "status": "verified" if not blockers else "source_marker_mismatch",
        "blockers": blockers,
        "source_root": source_root.as_posix(),
        "source_head_sha_observed": observed_sha,
    }


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", path.as_posix(), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _read_source_for_marker_scan(path: Path) -> str:
    if path.is_dir():
        return "\n".join(
            child.read_text(encoding="utf-8", errors="replace")
            if child.suffix != ".py"
            else read_python_source_for_marker_scan(child)
            for child in sorted(path.rglob("*"))
            if child.is_file()
        )
    if path.suffix == ".py":
        return read_python_source_for_marker_scan(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _family_row(family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    local_blockers = _ordered_unique(
        blocker
        for row in rows
        for blocker in row["blockers"]
        if blocker.endswith("_local_symbol_missing")
    )
    return {
        "family": family,
        "row_count": len(rows),
        "local_symbol_map_ready": bool(rows) and not local_blockers,
        "source_pins_verified": bool(rows)
        and all(row["source_marker_status"] == "verified" for row in rows),
        "blockers": local_blockers,
        **FALSE_AUTHORITY,
    }


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


__all__ = [
    "AUTHORITY",
    "DEFAULT_SOURCE_ROOTS",
    "SCHEMA",
    "build_nerv_official_symbol_parity_map",
]
