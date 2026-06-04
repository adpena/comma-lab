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
RECEIVER_ARCHIVE_PATH = "src/tac/substrates/snerv_inverse_steg_carrier/archive.py"
RECEIVER_ARCHIVE_TEST_PATH = (
    "src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive.py"
)
NATIVE_MLX_TRAIN_EXPORT_PATH = (
    "src/tac/substrates/snerv_inverse_steg_carrier/mlx_native_train_export.py"
)
NATIVE_MLX_RENDERER_PATH = (
    "src/tac/substrates/snerv_inverse_steg_carrier/mlx_renderer.py"
)
NATIVE_MLX_TRAIN_EXPORT_TEST_PATH = (
    "src/tac/substrates/snerv_inverse_steg_carrier/tests/test_mlx_native_train_export.py"
)
RECEIVER_EXPORT_BLOCKERS_AFTER_RUNTIME_DECODE: tuple[str, ...] = (
    "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
    "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
)
RECEIVER_EXPORT_BLOCKERS_AFTER_SOURCE_FORWARD_REPLAY: tuple[str, ...] = (
    "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
)
FORBIDDEN_RECEIVER_IMPORT_MARKERS: tuple[str, ...] = (
    "import torch",
    "from torch",
    "load_score_exact_scorers",
    "tac.scorer",
    "segmentation_models_pytorch",
)
NATIVE_MLX_EXPORT_MARKERS: tuple[str, ...] = (
    "_trained_official_packet",
    "_build_official_mfu_hfr_tub_packet_from_components",
    "trained_receiver_payload_exported",
    "snerv_mlx_official_mfu_hfr_tub_score_renderer",
)
NATIVE_MLX_RENDERER_MARKERS: tuple[str, ...] = (
    "class SnervMlxOfficialMfuHfrTubScoreRenderer",
    "def export_official_components",
    "SNERV_MLX_OFFICIAL_MFU_HFR_TUB_RENDERER_SCHEMA",
)
NATIVE_MLX_EXPORT_TEST_MARKERS: tuple[str, ...] = (
    "test_official_primitives_long_training_exports_trained_official_payload",
    "trained_receiver_payload_exported",
    "snerv_mlx_official_mfu_hfr_tub_score_renderer",
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
    receiver_archive_runtime_markers: tuple[str, ...]
    receiver_archive_test_markers: tuple[str, ...]
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
            "conv_transpose2d_nchw_mlx",
            "SNERV_OFFICIAL_MFU_TORCH_NUMPY_MLX_PARITY_PROOF",
        ),
        test_markers=(
            "test_official_mfu_full_numpy_forward_matches_torch_graph",
            "test_conv_transpose2d_mlx_modes_match_numpy_reference",
            "test_official_mfu_mlx_forward_modes_match_numpy_reference",
            "torch.nn.functional.conv_transpose2d",
            "np.testing.assert_allclose",
        ),
        runtime_entrypoint_markers=(
            "class OfficialSnervMfu",
            "def forward_mlx",
            "low: np.ndarray",
            "class OfficialConvTranspose2dNchw",
        ),
        receiver_archive_runtime_markers=(
            "DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA",
            "DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SELF_CONSISTENCY_SCHEMA",
            "encode_official_mfu_hfr_tub_decoder_payload",
            "decode_official_mfu_hfr_tub_decoder_payload",
            "execute_official_mfu_hfr_tub_decoder_payload",
            "class OfficialMfuHfrTubReceiverPayload",
            "def build_mfu",
            "receiver_self_consistency_reference",
            "source_forward_replay_bound_by_export",
            "_validate_official_receiver_self_consistency_reference",
            "OfficialSnervMfu",
            '"mfu.upsample_mid.weight"',
            '"inputs.mfu.low"',
        ),
        receiver_archive_test_markers=(
            "test_official_mfu_hfr_tub_decoder_payload_executes_receiver_primitives",
            "test_archive_can_carry_official_mfu_hfr_tub_receiver_payload",
            "test_official_mfu_hfr_tub_receiver_payload_decodes_batched_frames",
            "test_official_mfu_hfr_tub_self_consistency_reference_is_fail_closed",
            "test_official_mfu_hfr_tub_payload_bytes_change_receiver_output",
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
        receiver_archive_runtime_markers=(
            "DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA",
            "DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SELF_CONSISTENCY_SCHEMA",
            "encode_official_mfu_hfr_tub_decoder_payload",
            "decode_official_mfu_hfr_tub_decoder_payload",
            "execute_official_mfu_hfr_tub_decoder_payload",
            "class OfficialMfuHfrTubReceiverPayload",
            "def build_hfr_heads",
            "receiver_self_consistency_reference",
            "source_forward_replay_bound_by_export",
            "_validate_official_receiver_self_consistency_reference",
            "OfficialHfrHeads",
            '"hfr.lh.conv1.weight"',
            '"hfr.hh.conv2.bias"',
        ),
        receiver_archive_test_markers=(
            "test_official_mfu_hfr_tub_decoder_payload_executes_receiver_primitives",
            "test_archive_can_carry_official_mfu_hfr_tub_receiver_payload",
            "test_official_mfu_hfr_tub_self_consistency_reference_is_fail_closed",
            "test_official_mfu_hfr_tub_payload_bytes_change_receiver_output",
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
        receiver_archive_runtime_markers=(
            "DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA",
            "DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SELF_CONSISTENCY_SCHEMA",
            "encode_official_mfu_hfr_tub_decoder_payload",
            "decode_official_mfu_hfr_tub_decoder_payload",
            "execute_official_mfu_hfr_tub_decoder_payload",
            "class OfficialMfuHfrTubReceiverPayload",
            "prepare_official_tub_graph_inputs",
            "def tub_inputs",
            "receiver_self_consistency_reference",
            "source_forward_replay_bound_by_export",
            "_validate_official_receiver_self_consistency_reference",
            '"inputs.tub.current"',
            '"inputs.tub.next_frame"',
        ),
        receiver_archive_test_markers=(
            "test_official_mfu_hfr_tub_decoder_payload_executes_receiver_primitives",
            "test_archive_can_carry_official_mfu_hfr_tub_receiver_payload",
            "test_official_mfu_hfr_tub_self_consistency_reference_is_fail_closed",
            "test_official_mfu_hfr_tub_payload_bytes_change_receiver_output",
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
    all_numeric_proven = all(
        bool(row["primitive_numeric_graph_replay_proven"]) for row in rows
    )
    receiver_contract = build_snerv_official_receiver_runtime_decode_contract(
        repo_root=root,
    )
    native_export = _native_mlx_train_export_contract(root)
    receiver_export_bound = bool(receiver_contract["receiver_export_bound"])
    native_mlx_export_bound = bool(native_export["native_mlx_export_bound"])
    receiver_source_forward_bound = bool(
        receiver_contract["receiver_source_forward_replay_bound"]
    )
    return {
        "schema": SCHEMA,
        "component_rows": rows,
        "all_primitive_numeric_graph_replay_proven": all_numeric_proven,
        "all_receiver_primitive_replay_proven": all_numeric_proven,
        "all_primitive_numeric_source_fixture_replay_proven": all_numeric_proven,
        "all_primitive_source_replay_proven": False,
        "full_stack_source_forward_replay_proven": False,
        "receiver_export_self_consistency_verified": bool(
            receiver_contract["receiver_export_self_consistency_verified"]
        ),
        "receiver_source_forward_replay_bound": receiver_source_forward_bound,
        "official_receiver_runtime_decode_contract": receiver_contract,
        "receiver_archive_payload_bound": bool(
            receiver_contract["receiver_runtime_decode_proven"]
        ),
        "receiver_export_bound": receiver_export_bound,
        "native_mlx_export_bound": native_mlx_export_bound,
        "receiver_native_export_bound": bool(
            receiver_export_bound and native_mlx_export_bound
        ),
        "native_mlx_train_export_contract": native_export,
        "official_export_bound": _official_export_bound(
            receiver_export_bound=receiver_export_bound,
            native_mlx_export_bound=native_mlx_export_bound,
            receiver_source_forward_replay_bound=receiver_source_forward_bound,
        ),
        "official_export_bound_semantics": (
            "requires_receiver_export_native_mlx_export_and_source_forward_replay"
        ),
        "blockers": _ordered_unique(
            [*receiver_contract["blockers"], *native_export["blockers"]]
        ),
        "authority": "false_authority_source_forward_replay_not_native_mlx_or_scorer",
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
    self_consistency_verified = all(
        bool(row["receiver_export_self_consistency_verified"]) for row in rows
    )
    source_forward_bound = all(
        bool(row["receiver_source_forward_replay_bound"]) for row in rows
    )
    native_export = _native_mlx_train_export_contract(root)
    native_mlx_export_bound = bool(native_export["native_mlx_export_bound"])
    post_decode_blockers = (
        list(RECEIVER_EXPORT_BLOCKERS_AFTER_SOURCE_FORWARD_REPLAY)
        if source_forward_bound
        else (
            [
                "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
                *native_export["blockers"],
            ]
            if decode_proven
            else []
        )
    )
    return {
        "schema": RECEIVER_RUNTIME_DECODE_SCHEMA,
        "component_rows": rows,
        "all_runtime_modules_import_safe": runtime_safe,
        "all_numeric_graph_replay_tests_hashed": numeric_bound,
        "all_numeric_source_replay_tests_hashed": numeric_bound,
        "receiver_runtime_decode_proven": decode_proven,
        "receiver_export_self_consistency_verified": self_consistency_verified,
        "receiver_source_forward_replay_bound": source_forward_bound,
        "receiver_archive_payload_bound": decode_proven,
        "receiver_export_bound": self_consistency_verified,
        "native_mlx_export_bound": native_mlx_export_bound,
        "receiver_native_export_bound": bool(
            self_consistency_verified and native_mlx_export_bound
        ),
        "native_mlx_train_export_contract": native_export,
        "official_export_bound": _official_export_bound(
            receiver_export_bound=self_consistency_verified,
            native_mlx_export_bound=native_mlx_export_bound,
            receiver_source_forward_replay_bound=source_forward_bound,
        ),
        "official_export_bound_semantics": (
            "requires_receiver_export_native_mlx_export_and_source_forward_replay"
        ),
        "blockers": blockers + post_decode_blockers,
        "authority": "false_authority_source_forward_replay_not_native_mlx_or_scorer",
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
        "primitive_numeric_graph_replay_proven": proven,
        "primitive_numeric_source_fixture_replay_proven": proven,
        "receiver_primitive_replay_proven": proven,
        "primitive_source_replay_proven": False,
        "primitive_source_forward_replay_proven": False,
        "numeric_source_replay_test_present": bool(
            runtime_decode["numeric_source_replay_test_present"]
        ),
        "receiver_runtime_decode_row": runtime_decode,
        "full_stack_source_forward_replay_proven": False,
        "status": (
            "primitive_numeric_source_fixture_replay_proven_full_stack_missing"
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
    archive_path = root / RECEIVER_ARCHIVE_PATH
    archive_test_path = root / RECEIVER_ARCHIVE_TEST_PATH
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
    archive_text = (
        read_python_source_for_marker_scan(archive_path)
        if archive_path.is_file()
        else ""
    )
    archive_test_text = (
        archive_test_path.read_text(encoding="utf-8", errors="replace")
        if archive_test_path.is_file()
        else ""
    )
    source_identity = _file_identity_row(source_path)
    test_identity = _file_identity_row(test_path)
    archive_identity = _file_identity_row(archive_path)
    archive_test_identity = _file_identity_row(archive_test_path)
    missing_entrypoints = [
        marker for marker in spec.runtime_entrypoint_markers if marker not in source_text
    ]
    missing_numeric_markers = [
        marker for marker in spec.test_markers if marker not in test_text
    ]
    missing_archive_markers = [
        marker
        for marker in spec.receiver_archive_runtime_markers
        if marker not in archive_text
    ]
    missing_archive_test_markers = [
        marker
        for marker in spec.receiver_archive_test_markers
        if marker not in archive_test_text
    ]
    forbidden_import_markers = [
        marker for marker in FORBIDDEN_RECEIVER_IMPORT_MARKERS if marker in source_text
    ]
    archive_forbidden_import_markers = [
        marker for marker in FORBIDDEN_RECEIVER_IMPORT_MARKERS if marker in archive_text
    ]
    runtime_import_safe = bool(source_path.is_file() and not forbidden_import_markers)
    numeric_test_present = bool(test_path.is_file() and not missing_numeric_markers)
    entrypoint_present = bool(source_path.is_file() and not missing_entrypoints)
    archive_decode_present = bool(
        archive_path.is_file()
        and archive_test_path.is_file()
        and not missing_archive_markers
        and not missing_archive_test_markers
        and not archive_forbidden_import_markers
    )
    decode_proven = bool(
        runtime_import_safe
        and numeric_test_present
        and entrypoint_present
        and archive_decode_present
    )
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
            "snerv_official_receiver_archive_runtime_module_missing"
            if not archive_path.is_file()
            else "",
            "snerv_official_receiver_archive_runtime_test_missing"
            if not archive_test_path.is_file()
            else "",
            *[
                f"official_{spec.component_id}_receiver_archive_marker_missing:{marker}"
                for marker in missing_archive_markers
            ],
            *[
                f"official_{spec.component_id}_receiver_archive_test_marker_missing:{marker}"
                for marker in missing_archive_test_markers
            ],
            *[
                f"official_{spec.component_id}_receiver_archive_forbidden_import:{marker}"
                for marker in archive_forbidden_import_markers
            ],
            *(() if archive_decode_present else spec.receiver_decode_blockers),
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
        "receiver_archive_runtime_path": RECEIVER_ARCHIVE_PATH,
        "receiver_archive_runtime_bytes": archive_identity["bytes"],
        "receiver_archive_runtime_sha256": archive_identity["sha256"],
        "receiver_archive_test_path": RECEIVER_ARCHIVE_TEST_PATH,
        "receiver_archive_test_bytes": archive_test_identity["bytes"],
        "receiver_archive_test_sha256": archive_test_identity["sha256"],
        "runtime_entrypoint_markers": list(spec.runtime_entrypoint_markers),
        "missing_runtime_entrypoint_markers": missing_entrypoints,
        "receiver_archive_runtime_markers": list(spec.receiver_archive_runtime_markers),
        "missing_receiver_archive_runtime_markers": missing_archive_markers,
        "receiver_archive_test_markers": list(spec.receiver_archive_test_markers),
        "missing_receiver_archive_test_markers": missing_archive_test_markers,
        "forbidden_receiver_import_markers": list(FORBIDDEN_RECEIVER_IMPORT_MARKERS),
        "present_forbidden_receiver_import_markers": forbidden_import_markers,
        "present_receiver_archive_forbidden_import_markers": archive_forbidden_import_markers,
        "runtime_module_import_safe": runtime_import_safe,
        "runtime_entrypoints_present": entrypoint_present,
        "numeric_source_replay_test_present": numeric_test_present,
        "receiver_archive_payload_decode_present": archive_decode_present,
        "receiver_runtime_decode_proven": decode_proven,
        "receiver_export_self_consistency_verified": decode_proven,
        "receiver_source_forward_replay_bound": False,
        "receiver_archive_payload_bound": decode_proven,
        "receiver_export_bound": decode_proven,
        "native_mlx_export_bound": False,
        "blockers": blockers,
        "status": (
            "receiver_runtime_decode_proven_source_forward_missing"
            if decode_proven
            else (
                "numeric_source_replay_bound_receiver_decode_missing"
                if runtime_import_safe and entrypoint_present and numeric_test_present
                else "receiver_runtime_decode_contract_incomplete"
            )
        ),
        "authority": "false_authority_source_forward_replay_not_native_mlx_or_scorer",
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


def _native_mlx_train_export_contract(root: Path) -> dict[str, Any]:
    source_path = root / NATIVE_MLX_TRAIN_EXPORT_PATH
    renderer_path = root / NATIVE_MLX_RENDERER_PATH
    test_path = root / NATIVE_MLX_TRAIN_EXPORT_TEST_PATH
    source_text = (
        read_python_source_for_marker_scan(source_path)
        if source_path.is_file()
        else ""
    )
    renderer_text = (
        read_python_source_for_marker_scan(renderer_path)
        if renderer_path.is_file()
        else ""
    )
    test_text = (
        test_path.read_text(encoding="utf-8", errors="replace")
        if test_path.is_file()
        else ""
    )
    missing_source = [marker for marker in NATIVE_MLX_EXPORT_MARKERS if marker not in source_text]
    missing_renderer = [marker for marker in NATIVE_MLX_RENDERER_MARKERS if marker not in renderer_text]
    missing_tests = [marker for marker in NATIVE_MLX_EXPORT_TEST_MARKERS if marker not in test_text]
    bound = bool(
        source_path.is_file()
        and renderer_path.is_file()
        and test_path.is_file()
        and not missing_source
        and not missing_renderer
        and not missing_tests
    )
    blockers = _ordered_unique(
        [
            "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
            if not bound
            else "",
            *[
                f"snerv_official_native_mlx_train_export_marker_missing:{marker}"
                for marker in missing_source
            ],
            *[
                f"snerv_official_native_mlx_renderer_marker_missing:{marker}"
                for marker in missing_renderer
            ],
            *[
                f"snerv_official_native_mlx_train_export_test_marker_missing:{marker}"
                for marker in missing_tests
            ],
        ]
    )
    return {
        "schema": "snerv_official_mfu_hfr_tub_native_mlx_train_export_contract.v1",
        "native_mlx_export_bound": bound,
        "native_mlx_train_renderer_bound": bool(renderer_path.is_file() and not missing_renderer),
        "trained_receiver_payload_export_bound": bool(source_path.is_file() and not missing_source),
        "positive_train_export_test_bound": bool(test_path.is_file() and not missing_tests),
        "source_path": NATIVE_MLX_TRAIN_EXPORT_PATH,
        "renderer_path": NATIVE_MLX_RENDERER_PATH,
        "test_path": NATIVE_MLX_TRAIN_EXPORT_TEST_PATH,
        "source_file": _file_identity_row(source_path),
        "renderer_file": _file_identity_row(renderer_path),
        "test_file": _file_identity_row(test_path),
        "missing_source_markers": missing_source,
        "missing_renderer_markers": missing_renderer,
        "missing_test_markers": missing_tests,
        "blockers": blockers,
        "source_forward_replay_authority": False,
        **FALSE_AUTHORITY,
    }


def _official_export_bound(
    *,
    receiver_export_bound: bool,
    native_mlx_export_bound: bool,
    receiver_source_forward_replay_bound: bool,
) -> bool:
    """Return source-authority export status, not receiver-only payload status."""

    return bool(
        receiver_export_bound
        and native_mlx_export_bound
        and receiver_source_forward_replay_bound
    )


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
