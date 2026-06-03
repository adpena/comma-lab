# SPDX-License-Identifier: MIT
"""Shared SNeRV official primitive-replay binding.

The local SNeRV lane has executable MFU/HFR/TUB primitives, but primitive
numeric replay is not full official receiver/source-forward parity. This
module keeps that authority boundary in one place so inventory and stack audits
cannot drift on what has actually been proven.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tac.analysis.source_marker_scan import read_python_source_for_marker_scan

SCHEMA = "snerv_official_mfu_hfr_tub_primitive_replay_binding.v1"
ROW_SCHEMA = "snerv_official_primitive_source_replay_status.v1"

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


@dataclass(frozen=True)
class SnervPrimitiveReplaySpec:
    """Marker contract for one official SNeRV primitive family."""

    component_id: str
    feature_id: str
    source_path: str
    test_path: str
    source_markers: tuple[str, ...]
    test_markers: tuple[str, ...]


SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS: tuple[SnervPrimitiveReplaySpec, ...] = (
    SnervPrimitiveReplaySpec(
        component_id="mfu",
        feature_id="official_multi_resolution_fusion_blocks",
        source_path="src/tac/substrates/snerv_inverse_steg_carrier/official_mfu.py",
        test_path="src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_mfu.py",
        source_markers=(
            "OfficialSnervMfu",
            "OfficialConvTranspose2dNchw",
            "OfficialResidualBlocksWithInputConv",
            "conv_transpose2d_nchw",
        ),
        test_markers=(
            "test_official_mfu_full_numpy_forward_matches_torch_graph",
            "torch.nn.functional.conv_transpose2d",
            "np.testing.assert_allclose",
        ),
    ),
    SnervPrimitiveReplaySpec(
        component_id="hfr",
        feature_id="official_high_frequency_restoration_heads",
        source_path="src/tac/substrates/snerv_inverse_steg_carrier/official_hfr.py",
        test_path="src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_hfr.py",
        source_markers=(
            "OfficialHfrHeads",
            "OfficialHfrConvBlock",
            "conv2d_nchw_mlx",
            "SNERV_OFFICIAL_HFR_CONVBLOCK_NUMPY_PROOF",
        ),
        test_markers=(
            "test_official_hfr_numpy_matches_torch_convblock",
            "torch.nn.Conv2d",
            "test_official_hfr_mlx_default_is_fixed_reference_and_repeatable",
        ),
    ),
    SnervPrimitiveReplaySpec(
        component_id="tub",
        feature_id="official_temporal_extension_snerv_t",
        source_path="src/tac/substrates/snerv_inverse_steg_carrier/official_tub.py",
        test_path="src/tac/substrates/snerv_inverse_steg_carrier/tests/test_official_tub.py",
        source_markers=(
            "prepare_official_tub_graph_inputs",
            "official_output2_fusion_shape",
            "OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT",
        ),
        test_markers=(
            "test_official_snerv_t_tub_source_contract_is_pinned",
            "test_official_tub_graph_inputs_match_haar_lowpass_contract",
            "test_dwt1d_pair_reconstructs_current_and_previous_normalized_lf",
        ),
    ),
)


def build_snerv_official_primitive_replay_binding(*, repo_root: str | Path) -> dict[str, Any]:
    """Return the shared false-authority primitive replay binding."""

    root = Path(repo_root)
    rows = [_primitive_replay_row(root, spec) for spec in SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS]
    all_proven = all(bool(row["primitive_source_replay_proven"]) for row in rows)
    return {
        "schema": SCHEMA,
        "component_rows": rows,
        "all_primitive_source_replay_proven": all_proven,
        "full_stack_source_forward_replay_proven": False,
        "receiver_export_bound": False,
        "authority": "false_authority_primitive_replay_not_receiver_export",
        **FALSE_AUTHORITY,
    }


def snerv_primitive_source_replay_status(
    *,
    repo_root: str | Path,
    feature_id: str,
) -> dict[str, Any] | None:
    """Return one feature-level primitive replay row, if the feature is tracked."""

    root = Path(repo_root)
    for spec in SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS:
        if spec.feature_id == feature_id:
            return _primitive_replay_row(root, spec)
    return None


def _primitive_replay_row(root: Path, spec: SnervPrimitiveReplaySpec) -> dict[str, Any]:
    source_path = root / spec.source_path
    test_path = root / spec.test_path
    source_text = (
        read_python_source_for_marker_scan(source_path)
        if source_path.is_file()
        else ""
    )
    test_text = (
        test_path.read_text(encoding="utf-8", errors="replace")
        if test_path.is_file()
        else ""
    )
    missing_source = [marker for marker in spec.source_markers if marker not in source_text]
    missing_tests = [marker for marker in spec.test_markers if marker not in test_text]
    proven = source_path.is_file() and test_path.is_file() and not missing_source and not missing_tests
    return {
        "schema": ROW_SCHEMA,
        "component_id": spec.component_id,
        "feature_id": spec.feature_id,
        "source_path": spec.source_path,
        "test_path": spec.test_path,
        "source_rel_path": spec.source_path,
        "test_rel_path": spec.test_path,
        "source_present": source_path.is_file(),
        "test_present": test_path.is_file(),
        "source_markers": list(spec.source_markers),
        "test_markers": list(spec.test_markers),
        "missing_source_markers": missing_source,
        "missing_test_markers": missing_tests,
        "primitive_source_replay_proven": proven,
        "full_stack_source_forward_replay_proven": False,
        "status": (
            "primitive_source_replay_proven_full_stack_missing"
            if proven
            else "primitive_source_replay_missing"
        ),
        "authority": "false_authority_primitive_replay_not_full_receiver_export",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "FALSE_AUTHORITY",
    "SCHEMA",
    "SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS",
    "SnervPrimitiveReplaySpec",
    "build_snerv_official_primitive_replay_binding",
    "snerv_primitive_source_replay_status",
]
