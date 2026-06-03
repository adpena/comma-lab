# SPDX-License-Identifier: MIT
"""Shared SNeRV official primitive-replay binding.

The local SNeRV lane has executable MFU/HFR/TUB primitives, but primitive
numeric replay is not full official receiver/source-forward parity. This
module keeps that authority boundary in one place so inventory and stack audits
cannot drift on what has actually been proven.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from tac.analysis.source_marker_scan import read_python_source_for_marker_scan

SCHEMA = "snerv_official_mfu_hfr_tub_primitive_replay_binding.v1"
ROW_SCHEMA = "snerv_official_primitive_source_replay_status.v1"
RECEIVER_RUNTIME_DECODE_SCHEMA = (
    "snerv_official_mfu_hfr_tub_receiver_runtime_decode_contract.v1"
)
RECEIVER_RUNTIME_DECODE_ROW_SCHEMA = (
    "snerv_official_receiver_runtime_decode_component.v1"
)

FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "frontier_score_claim": False,
    "rank_or_kill_eligible": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}
FORBIDDEN_RECEIVER_IMPORT_MARKERS: tuple[str, ...] = (
    "import torch",
    "from torch",
    "load_score_exact_scorers",
    "tac.scorer",
    "segmentation_models_pytorch",
)


@dataclass(frozen=True)
class SnervPrimitiveReplaySpec:
    """Marker contract for one official SNeRV primitive family."""

    component_id: str
    feature_id: str
    source_path: str
    test_path: str
    source_markers: tuple[str, ...]
    test_markers: tuple[str, ...]
    runtime_entrypoint_markers: tuple[str, ...]
    receiver_decode_blockers: tuple[str, ...]


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
        runtime_entrypoint_markers=(
            "class OfficialSnervMfu",
            "low: np.ndarray",
            "class OfficialConvTranspose2dNchw",
        ),
        receiver_decode_blockers=(
            "snerv_mfu_official_receiver_archive_section_schema_missing",
            "snerv_mfu_official_weight_packet_decode_missing",
            "snerv_mfu_official_input_bundle_decode_missing",
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
        runtime_entrypoint_markers=(
            "class OfficialHfrHeads",
            "def forward(self, pyr_out",
            "class OfficialHfrConvBlock",
        ),
        receiver_decode_blockers=(
            "snerv_hfr_official_receiver_archive_section_schema_missing",
            "snerv_hfr_official_weight_packet_decode_missing",
            "snerv_hfr_official_pyr_out_packet_decode_missing",
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
        runtime_entrypoint_markers=(
            "def prepare_official_tub_graph_inputs",
            "def official_output2_fusion_shape",
            "class OfficialTubGraphInputs",
        ),
        receiver_decode_blockers=(
            "snerv_tub_official_receiver_archive_section_schema_missing",
            "snerv_tub_official_frame_triplet_packet_decode_missing",
            "snerv_tub_official_temporal_encoder_weight_decode_missing",
            "snerv_tub_official_output2_packet_decode_missing",
        ),
    ),
)


def build_snerv_official_primitive_replay_binding(*, repo_root: str | Path) -> dict[str, Any]:
    """Return the shared false-authority primitive replay binding."""

    root = Path(repo_root)
    rows = [_primitive_replay_row(root, spec) for spec in SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS]
    all_proven = all(bool(row["primitive_source_replay_proven"]) for row in rows)
    receiver_contract = build_snerv_official_receiver_runtime_decode_contract(
        repo_root=root,
    )
    return {
        "schema": SCHEMA,
        "component_rows": rows,
        "all_primitive_source_replay_proven": all_proven,
        "full_stack_source_forward_replay_proven": False,
        "official_receiver_runtime_decode_contract": receiver_contract,
        "receiver_export_bound": bool(
            receiver_contract["receiver_runtime_decode_proven"]
        ),
        "authority": "false_authority_primitive_replay_not_receiver_export",
        **FALSE_AUTHORITY,
    }


def build_snerv_official_receiver_runtime_decode_contract(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return the executable-source evidence and exact receiver decode blockers.

    These rows intentionally separate three claims:

    * the portable runtime modules are scorer-free/import-safe;
    * focused tests numerically replay official primitive math; and
    * a byte-closed receiver archive can decode official primitive packets.

    The first two can be true while the last remains false.
    """

    root = Path(repo_root)
    rows = [
        _receiver_runtime_decode_row(root, spec)
        for spec in SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS
    ]
    blockers = _ordered_unique(
        blocker for row in rows for blocker in row["blockers"]
    )
    runtime_safe = all(bool(row["runtime_module_import_safe"]) for row in rows)
    numeric_bound = all(
        bool(row["numeric_source_replay_test_present"]) for row in rows
    )
    decode_proven = all(bool(row["receiver_runtime_decode_proven"]) for row in rows)
    return {
        "schema": RECEIVER_RUNTIME_DECODE_SCHEMA,
        "component_rows": rows,
        "all_runtime_modules_import_safe": runtime_safe,
        "all_numeric_source_replay_tests_hashed": numeric_bound,
        "receiver_runtime_decode_proven": decode_proven,
        "receiver_export_bound": decode_proven,
        "blockers": blockers,
        "authority": "false_authority_runtime_decode_contract_not_byte_closed",
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
    source_file_row = _file_identity_row(source_path)
    test_file_row = _file_identity_row(test_path)
    runtime_decode = _receiver_runtime_decode_row(root, spec)
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
        "source_bytes": source_file_row["bytes"],
        "source_sha256": source_file_row["sha256"],
        "test_bytes": test_file_row["bytes"],
        "test_sha256": test_file_row["sha256"],
        "source_markers": list(spec.source_markers),
        "test_markers": list(spec.test_markers),
        "missing_source_markers": missing_source,
        "missing_test_markers": missing_tests,
        "primitive_source_replay_proven": proven,
        "numeric_source_replay_test_present": bool(
            runtime_decode["numeric_source_replay_test_present"]
        ),
        "receiver_runtime_decode_row": runtime_decode,
        "full_stack_source_forward_replay_proven": False,
        "status": (
            "primitive_source_replay_proven_full_stack_missing"
            if proven
            else "primitive_source_replay_missing"
        ),
        "authority": "false_authority_primitive_replay_not_full_receiver_export",
        **FALSE_AUTHORITY,
    }


def _receiver_runtime_decode_row(
    root: Path,
    spec: SnervPrimitiveReplaySpec,
) -> dict[str, Any]:
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
    source_identity = _file_identity_row(source_path)
    test_identity = _file_identity_row(test_path)
    missing_entrypoints = [
        marker for marker in spec.runtime_entrypoint_markers if marker not in source_text
    ]
    missing_numeric_markers = [
        marker for marker in spec.test_markers if marker not in test_text
    ]
    forbidden_import_markers = [
        marker for marker in FORBIDDEN_RECEIVER_IMPORT_MARKERS if marker in source_text
    ]
    runtime_import_safe = bool(source_path.is_file() and not forbidden_import_markers)
    numeric_test_present = bool(test_path.is_file() and not missing_numeric_markers)
    entrypoint_present = bool(source_path.is_file() and not missing_entrypoints)
    blockers = _ordered_unique(
        [
            f"official_{spec.component_id}_runtime_module_missing"
            if not source_path.is_file()
            else "",
            f"official_{spec.component_id}_numeric_replay_test_missing"
            if not test_path.is_file()
            else "",
            *[
                f"official_{spec.component_id}_runtime_entrypoint_missing:{marker}"
                for marker in missing_entrypoints
            ],
            *[
                f"official_{spec.component_id}_numeric_replay_marker_missing:{marker}"
                for marker in missing_numeric_markers
            ],
            *[
                f"official_{spec.component_id}_receiver_forbidden_import:{marker}"
                for marker in forbidden_import_markers
            ],
            *spec.receiver_decode_blockers,
        ]
    )
    return {
        "schema": RECEIVER_RUNTIME_DECODE_ROW_SCHEMA,
        "component_id": spec.component_id,
        "feature_id": spec.feature_id,
        "runtime_module_path": spec.source_path,
        "runtime_module_bytes": source_identity["bytes"],
        "runtime_module_sha256": source_identity["sha256"],
        "numeric_test_path": spec.test_path,
        "numeric_test_bytes": test_identity["bytes"],
        "numeric_test_sha256": test_identity["sha256"],
        "runtime_entrypoint_markers": list(spec.runtime_entrypoint_markers),
        "missing_runtime_entrypoint_markers": missing_entrypoints,
        "forbidden_receiver_import_markers": list(FORBIDDEN_RECEIVER_IMPORT_MARKERS),
        "present_forbidden_receiver_import_markers": forbidden_import_markers,
        "runtime_module_import_safe": runtime_import_safe,
        "runtime_entrypoints_present": entrypoint_present,
        "numeric_source_replay_test_present": numeric_test_present,
        "receiver_runtime_decode_proven": False,
        "receiver_export_bound": False,
        "blockers": blockers,
        "status": (
            "numeric_source_replay_bound_receiver_decode_missing"
            if runtime_import_safe and entrypoint_present and numeric_test_present
            else "receiver_runtime_decode_contract_incomplete"
        ),
        "authority": "false_authority_runtime_decode_contract_not_byte_closed",
        **FALSE_AUTHORITY,
    }


def _file_identity_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.as_posix(),
            "bytes": 0,
            "sha256": None,
        }
    data = path.read_bytes()
    return {
        "path": path.as_posix(),
        "bytes": len(data),
        "sha256": sha256(data).hexdigest(),
    }


def _ordered_unique(items: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


__all__ = [
    "FALSE_AUTHORITY",
    "RECEIVER_RUNTIME_DECODE_SCHEMA",
    "SCHEMA",
    "SNERV_OFFICIAL_PRIMITIVE_REPLAY_SPECS",
    "SnervPrimitiveReplaySpec",
    "build_snerv_official_primitive_replay_binding",
    "build_snerv_official_receiver_runtime_decode_contract",
    "snerv_primitive_source_replay_status",
]
