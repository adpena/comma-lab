# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.optimization import repair_family_stack_search as repair_stack_search_module
from tac.optimization.archive_bound_candidate_adapter_spine import (
    ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA,
)
from tac.optimization.archive_bound_candidate_contract import (
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA,
    ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA,
    archive_bound_candidate_contract_fields_for_row,
    archive_substrate_tags_for_transform_kind,
)
from tac.optimization.repair_archive_entropy_substrate_coverage import (
    REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA,
)
from tac.optimization.repair_campaign_chain_contract import (
    RepairCampaignChainContractError,
)
from tac.optimization.repair_campaign_learning_signal import (
    REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA,
    build_repair_campaign_blocked_learning_signal_report,
    build_repair_campaign_materialization_learning_signal_report,
)
from tac.optimization.repair_campaign_posterior import (
    append_repair_campaign_blocked_learning_signal_report,
    load_repair_campaign_stackability_posterior_rows,
)
from tac.optimization.repair_campaign_scorer import (
    build_repair_campaign_posterior_prior_summary,
    score_repair_campaign,
)
from tac.optimization.repair_entropy_coder_runtime_adapters import (
    REPAIR_ENTROPY_CODER_RUNTIME_ADAPTER_MANIFEST_SCHEMA,
    ans_rans_prototype_encode,
    decode_entropy_coder_prototype_member,
    range_lzma_prototype_encode,
)
from tac.optimization.repair_entropy_stage_chain_executor import (
    REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA,
    build_repair_entropy_stage_chain_execution_bundle,
)
from tac.optimization.repair_family_byte_transform_executor import (
    FEC5_FIXED_K8_CODE_BITS,
    FEC6_FIXED_K16_CODE_BITS,
    REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_ROW_SCHEMA,
    REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA,
    REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA,
    REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
    REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA,
    build_repair_family_byte_transform_execution_report,
)
from tac.optimization.repair_family_exact_ready_bridge import (
    REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA,
    RepairFamilyExactReadyBridgeError,
    build_repair_family_exact_ready_bridge,
)
from tac.optimization.repair_family_materializers import (
    REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
    build_repair_campaign_family_materializer_manifest,
)
from tac.optimization.repair_family_stack_search import (
    REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA,
    REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
    REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA,
    REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
    build_repair_family_exact_handoff_plan,
    build_repair_family_stack_learning_signal_report,
    plan_repair_family_stack_search,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotable": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "field_selection_ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "gpu_launched": False,
        "exact_cuda_auth_eval": False,
        "contest_cuda_auth_eval": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
    }


def test_entropy_coder_runtime_adapters_roundtrip_receiver_decode() -> None:
    payloads = [
        b"",
        b"abc",
        bytes(range(256)) * 2,
        b"\x00" * 128,
        b"selector-payload" * 17,
    ]
    for payload in payloads:
        assert (
            decode_entropy_coder_prototype_member(
                coder_family="range",
                packet=range_lzma_prototype_encode(payload),
            )
            == payload
        )
        assert (
            decode_entropy_coder_prototype_member(
                coder_family="ans",
                packet=ans_rans_prototype_encode(payload),
            )
            == payload
        )


def test_archive_entropy_orphan_anti_pattern_penalizes_prototype_rows() -> None:
    penalty = repair_stack_search_module._archive_entropy_anti_pattern_penalty(
        {
            "anti_pattern_protections": [{"anti_pattern_id": ("probe_only_side_report_orphaned_from_optimizer_v1")}],
            "probed_substrates": [],
            "prototype_substrates": ["range_coding"],
            "blockers": [],
        }
    )

    assert penalty > 0.0


def test_archive_bound_candidate_contract_classifies_entropy_substrates() -> None:
    assert archive_substrate_tags_for_transform_kind("fec6_selector_payload_mutation") == [
        "fec",
        "selector_stream",
        "selector",
        "huffman",
    ]
    assert archive_substrate_tags_for_transform_kind("range_coder_lzma_prototype") == [
        "range_coder",
        "entropy_coder",
        "receiver_adapter",
    ]
    assert archive_substrate_tags_for_transform_kind("candidate_archive_native_zip_repack") == [
        "zip_ordering",
        "zip_container",
    ]
    assert archive_substrate_tags_for_transform_kind(
        "dreamer_v3_rssm_mlx_categorical_predictive_coding_archive"
    ) == [
        "neural_archive",
        "mlx_substrate",
        "predictive_coding",
        "dreamer_v3",
        "rssm",
    ]
    assert archive_substrate_tags_for_transform_kind(
        "z8_hierarchical_predictive_coding_archive"
    ) == ["neural_archive", "predictive_coding", "z8"]


def _repair_payload(tmp_path: Path) -> dict[str, object]:
    mlx = tmp_path / "segnet_mlx_response.json"
    ref = tmp_path / "segnet_reference_mlx_response.json"
    mlx.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    ref.write_text('{"schema":"mlx_scorer_response.v1"}\n', encoding="utf-8")
    return {
        "schema": "frontier_rate_attack_repair_budget_waterfill_work_order.v1",
        "chain_id": "unit_repair_chain",
        "receiver_closed_rate_credit": {
            "schema": "frontier_rate_attack_repair_waterfill_rate_credit.v1",
            "receiver_closed_saved_bytes_total": 40,
            **_false_authority(),
        },
        "typed_response_ledger": {
            "schema": "frontier_rate_attack_repair_budget_typed_response_ledger.v1",
            "available_receiver_closed_rate_credit_bytes": 40,
            "rows": [
                {
                    "schema": ("frontier_rate_attack_repair_budget_typed_response_row.v1"),
                    "typed_response_id": "segnet_region_ready",
                    "candidate_id": "segnet_class_region_waterfill",
                    "correction_family": "segnet_class_region_waterfill",
                    "targeted_dimensions": ["segnet", "region"],
                    "operation_levels": ["pixel", "boundary", "region", "frame"],
                    "entropy_position_label": ("before_entropy_coder_distribution_shaping"),
                    "requested_repair_bytes": 32,
                    "objective_delta_score_units": -0.0010,
                    "local_mlx_response_path": str(mlx),
                    "reference_local_mlx_response_path": str(ref),
                    "segnet_class_region_mask_ids": ["road_boundary"],
                    **_false_authority(),
                }
            ],
            **_false_authority(),
        },
        **_false_authority(),
    }


def _plan_from_score_report(score_report: dict[str, object]) -> dict[str, object]:
    child_id = "repair_budget_spent_child_unit_segnet"
    allocation = score_report["optimizer_decision"]["selected_allocation_rows"][0]  # type: ignore[index]
    return {
        "schema": "frontier_rate_attack_repair_budget_materialization_plan.v1",
        "chain_id": "unit_repair_chain",
        "parent_candidate_chain_id": "rate_parent",
        "candidate_chain_rows": [
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_plan_row.v1",
                "candidate_kind": "rate_only_floor_parent",
                "candidate_chain_id": "rate_parent",
                "materialization_order": 1,
                "candidate_archive_materialized": False,
                "receiver_consumed": False,
                **_false_authority(),
            },
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_plan_row.v1",
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": child_id,
                "parent_candidate_chain_id": "rate_parent",
                "materialization_order": 2,
                "typed_response_id": allocation["typed_response_id"],
                "allocation_candidate_id": allocation["candidate_id"],
                "correction_family": allocation["correction_family"],
                "operation_levels": ["pixel", "boundary", "region", "frame"],
                "entropy_position_label": allocation["entropy_position_label"],
                "candidate_archive_materialized": False,
                "receiver_consumed": False,
                **_false_authority(),
            },
        ],
        "budget_spend_allowed": False,
        "ready_for_exact_eval_dispatch": False,
        **_false_authority(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _fec6_selector_payload(codes: list[int]) -> bytes:
    bits = "".join(FEC6_FIXED_K16_CODE_BITS[code] for code in codes)
    bits += "0" * ((-len(bits)) % 8)
    encoded = bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))
    selector = b"FEC6" + struct.pack("<H", len(codes)) + encoded
    source_payload = b"hnerv-source-payload"
    return (
        b"FP11" + struct.pack("<I", len(source_payload)) + source_payload + struct.pack("<H", len(selector)) + selector
    )


def _pack_lsb_codes(codes: list[int], bits_per_symbol: int) -> bytes:
    out = bytearray((len(codes) * bits_per_symbol + 7) // 8)
    bit_pos = 0
    for code in codes:
        for shift in range(bits_per_symbol):
            if (int(code) >> shift) & 1:
                absolute = bit_pos + shift
                out[absolute // 8] |= 1 << (absolute % 8)
        bit_pos += bits_per_symbol
    return bytes(out)


def _fp11_selector_payload(selector: bytes) -> bytes:
    source_payload = b"hnerv-source-payload"
    return (
        b"FP11" + struct.pack("<I", len(source_payload)) + source_payload + struct.pack("<H", len(selector)) + selector
    )


def _fec3_selector_payload(codes: list[int]) -> bytes:
    bits_per_symbol = 2
    palette_table = b"".join(struct.pack("<BB", 0, index) for index in range(4))
    selector = (
        b"FEC3"
        + struct.pack("<HBB", len(codes), bits_per_symbol, 4)
        + palette_table
        + _pack_lsb_codes(codes, bits_per_symbol)
    )
    return _fp11_selector_payload(selector)


def _fec5_selector_payload(codes: list[int]) -> bytes:
    bits = "".join(FEC5_FIXED_K8_CODE_BITS[code] for code in codes)
    bits += "0" * ((-len(bits)) % 8)
    encoded = bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))
    return _fp11_selector_payload(b"FEC5" + struct.pack("<H", len(codes)) + encoded)


def _fes1_selector_payload(codes: list[int]) -> bytes:
    packed = _pack_lsb_codes(codes, 5)
    selector = b"FES1" + struct.pack("<HBBH", len(codes), 31, 5, len(packed)) + packed
    return _fp11_selector_payload(selector)


def _fec8_selector_payload(codes: list[int]) -> bytes:
    module_path = (
        REPO_ROOT / "submissions/hnerv_fec6_fixed_huffman_k16/encoder/"
        "build_pr101_frame_exploit_selector_packet_markov.py"
    )
    spec = importlib.util.spec_from_file_location("_test_fec8_markov_codec", module_path)
    if spec is None or spec.loader is None:
        pytest.skip("FEC8 Markov codec module unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selector = module.encode_fec8_markov_selector_static_second_order(
        codes,
        n_pairs=len(codes),
    )
    return _fp11_selector_payload(selector)


def _psv4_varint(value: int) -> bytes:
    out = bytearray()
    while value:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        out.append(byte)
    return bytes(out or b"\x00")


def _psv4_rle_selector(codes: list[int]) -> bytes:
    out = bytearray()
    index = 0
    while index < len(codes):
        code = int(codes[index])
        run = 1
        while index + run < len(codes) and int(codes[index + run]) == code:
            run += 1
        out.append(code)
        out.extend(_psv4_varint(run))
        index += run
    return bytes(out)


def _psv4_packet_payload(
    codes: list[int],
    *,
    palette_size: int = 16,
    latent_dim: int = 2,
) -> bytes:
    decoder_blob = b"unit-decoder-state-placeholder"
    latent_blob = b"\x00" * (len(codes) * latent_dim * 2)
    selector = _psv4_rle_selector(codes)
    meta = b'{"unit":"psv4"}'
    header = struct.pack(
        "<4sBHHBIIII",
        b"PSV4",
        1,
        latent_dim,
        len(codes),
        palette_size,
        len(decoder_blob),
        len(latent_blob),
        len(selector),
        len(meta),
    )
    return header + decoder_blob + latent_blob + selector + meta


def test_segnet_family_materializer_emits_ordered_fail_closed_manifest(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)

    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )

    assert manifest["schema"] == REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    assert manifest["target_kind"] == "segnet_class_region_waterfill"
    assert manifest["allocated_repair_bytes"] == 32
    assert manifest["objective_delta_score_units"] == -0.001
    assert manifest["candidate_chain_ids"] == ["repair_budget_spent_child_unit_segnet"]
    assert manifest["active_entropy_stage"]["order"] == 10
    assert manifest["fractal_optimization_scope"]["ordered_levels"] == [
        "bit",
        "byte",
        "pixel",
        "boundary",
        "region",
        "frame",
        "pair",
        "batch",
        "full_video",
    ]
    assert manifest["component_response_replayed"] is True
    assert manifest["byte_closed_candidate_emitted"] is False
    assert (
        "segnet_class_region_waterfill_byte_closed_candidate_archive_not_materialized" in manifest["readiness_blockers"]
    )
    assert "segnet_class_region_mask_ids_missing" not in manifest["readiness_blockers"]
    assert manifest["ready_for_exact_eval_dispatch"] is False


def test_family_materializer_preserves_candidate_family_and_budget_for_novel_families(
    tmp_path: Path,
) -> None:
    payload = _repair_payload(tmp_path)
    ledger = payload["typed_response_ledger"]  # type: ignore[index]
    row = ledger["rows"][0]  # type: ignore[index]
    row.update(
        {
            "typed_response_id": "frame0_k16_palette_ready",
            "candidate_id": "frame0_k16_palette_asymmetry",
            "correction_family": "frame0_k16_palette_asymmetry",
            "targeted_dimensions": ["palette", "frame0"],
            "operation_levels": ["pixel", "byte", "frame"],
            "requested_repair_bytes": 10,
            "objective_delta_score_units": -0.0012,
            "palette_dynamics_context": {
                "canonical_k": 16,
                "frame0_mode_count": 15,
                "frame1_mode_count": 0,
            },
        }
    )
    score_report = score_repair_campaign(payload=payload, repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)

    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="frame0_k16_palette_ready",
        candidate_id="frame0_k16_palette_asymmetry",
    )

    assert manifest["family_id"] == "frame0_k16_palette_asymmetry"
    assert manifest["target_kind"] == "frame0_k16_palette_asymmetry"
    assert manifest["allocated_repair_bytes"] == 10
    assert manifest["objective_delta_score_units"] == -0.0012
    assert "unsupported_repair_family_byte_transform" not in " ".join(manifest["readiness_blockers"])


def test_family_materializer_cli_writes_manifest(tmp_path: Path) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    score_report_path = _write_json(tmp_path / "score_report.json", score_report)
    plan_path = _write_json(tmp_path / "plan.json", plan)
    manifest_path = tmp_path / "family_manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "build_repair_campaign_family_materializer_manifest.py"),
            "--materialization-plan",
            str(plan_path),
            "--score-report",
            str(score_report_path),
            "--typed-response-id",
            "segnet_region_ready",
            "--candidate-id",
            "segnet_class_region_waterfill",
            "--materializer-manifest-out",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA
    assert manifest["component_response_replayed"] is True
    assert manifest["score_claim"] is False


def test_repair_family_byte_transform_executor_emits_replayable_delta(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)

    report, bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    assert report["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA
    assert report["family_id"] == "segnet_class_region_waterfill"
    assert report["byte_transform_delta_emitted"] is True
    assert report["byte_transform_delta"]["bytes"] > 0
    assert (tmp_path / report["byte_transform_delta"]["path"]).is_file()
    assert report["component_response_replayed"] is True
    assert report["exact_eval_handoff_gate"]["eligible_for_exact_eval_handoff"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert bundle["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA
    assert bundle["source_records_sha256"]


def test_byte_transform_executor_repacks_archive_native_candidate_when_custody_exists(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    archive_path = _write_zip(
        tmp_path / "source_archive.zip",
        {"0.bin": (b"segnet-region-waterfill" * 64)},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    plan["candidate_chain_rows"][1]["candidate_archive_path"] = str(archive_path)
    plan["candidate_chain_rows"][1]["candidate_archive_sha256"] = archive_sha
    plan["candidate_chain_rows"][1]["candidate_archive_bytes"] = archive_path.stat().st_size
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)

    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    candidate = report["candidate_archive"]
    assert report["byte_closed_candidate_emitted"] is True
    assert report["archive_native_transform_attempted"] is True
    assert report["archive_native_transform_kind"] in {
        "packet_member_entropy_boundary_recompress",
        "zip_repack_payload_identity",
    }
    assert {variant["archive_native_transform_kind"] for variant in report["candidate_archive_transform_variants"]} >= {
        "packet_member_entropy_boundary_recompress",
        "zip_repack_payload_identity",
    }
    assert candidate["runtime_consumption_proof_ready"] is True
    assert candidate["receiver_contract_satisfied"] is True
    assert (tmp_path / candidate["path"]).is_file()
    assert (tmp_path / candidate["runtime_consumption_proof_path"]).is_file()
    assert report["archive_bound_candidate_contract_schema"] == (ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA)
    assert report["archive_bound_candidate_contract_surface_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
    )
    contract = report["archive_bound_candidate_contract"]
    assert contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert contract["archive_bound_candidate_ready"] is True
    assert contract["receiver_contract_satisfied"] is True
    assert contract["archive_file_custody"]["custody_complete"] is True
    assert report["archive_bound_candidate_adapter_package_schema"] == (ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA)
    adapter_package = report["archive_bound_candidate_adapter_package"]
    assert adapter_package["schema"] == ARCHIVE_BOUND_CANDIDATE_ADAPTER_PACKAGE_SCHEMA
    assert adapter_package["candidate_row_count"] == 1
    assert adapter_package["deterministic_replay_bundles"][0]["replay_bundle_ready"] is True
    assert adapter_package["receiver_proof_gates"][0]["receiver_proof_gate_passed"] is True
    assert adapter_package["exact_axis_blockers"][0]["ready_for_exact_eval_dispatch"] is False
    assert adapter_package["posterior_update_hooks"][0]["negative_result_demotes_family_stage_scope"] is True
    assert "zip_container" in report["archive_contract_substrate_tags"]
    assert report["archive_bound_ready_contract_count"] >= 1
    assert report["exact_eval_handoff_gate"]["archive_bound_runtime_consumption_proof_ready"] is True
    assert report["exact_eval_handoff_gate"]["eligible_for_exact_eval_handoff"] is False
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        byte_credit_budget=10_000,
    )
    assert stack_plan["exact_eval_handoff_candidate_count"] == 1
    assert stack_plan["archive_bound_exact_handoff_candidate_count"] == 1
    assert stack_plan["exact_eval_handoff_gate"]["archive_bound_custody_complete"] is True
    assert (
        "byte_closed_archive_runtime_receiver_proof_required_per_stack"
        not in stack_plan["exact_eval_handoff_gate"]["blockers"]
    )
    handoff_row = stack_plan["exact_eval_handoff_candidates"][0]
    assert handoff_row["schema"] == REPAIR_FAMILY_EXACT_HANDOFF_CANDIDATE_ROW_SCHEMA
    assert handoff_row["archive_bound_custody_complete"] is True
    assert handoff_row["archive_bound_candidate_contract"]["schema"] == (ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA)
    assert handoff_row["candidate_archive"]["custody_complete"] is True
    assert handoff_row["runtime_consumption_proof"]["custody_complete"] is True
    assert handoff_row["ready_for_exact_eval_dispatch"] is False
    exact_handoff_plan = build_repair_family_exact_handoff_plan(
        stack_plan=stack_plan,
        stack_plan_path=report_path,
    )
    assert exact_handoff_plan["schema"] == REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA
    assert exact_handoff_plan["archive_bound_candidate_count"] == 1
    assert exact_handoff_plan["archive_bound_custody_complete"] is True
    assert exact_handoff_plan["ready_for_exact_eval_dispatch"] is False


def test_byte_transform_executor_mutates_fec6_selector_payload_when_detected(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(
        tmp_path / "fec6_source_archive.zip",
        {"x": _fec6_selector_payload([0, 2, 7, 13, 0, 2, 7, 13, 0, 2])},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    source_proof_path = _write_json(
        tmp_path / "source_archive_runtime_proof.json",
        {
            "schema": "source_archive_runtime_consumption_proof.v1",
            "runtime_consumption_proof_passed": True,
            "candidate_archive": {
                "path": str(archive_path),
                "sha256": archive_sha,
                "bytes": archive_path.stat().st_size,
            },
            "blockers": [],
            **_false_authority(),
        },
    )
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "materializer_id": "repair_family_materializer:frame0_k16_palette_asymmetry",
        "target_kind": "frame0_k16_palette_asymmetry",
        "family_id": "frame0_k16_palette_asymmetry",
        "typed_response_id": "fec6_palette_ready",
        "candidate_chain_id": "fec6_palette_chain",
        "candidate_chain_ids": ["fec6_palette_chain"],
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "active_entropy_stage": {
            "order": 10,
            "stage": "before_entropy_coder_distribution_shaping",
            "class": "pre_entropy_distribution_shaping",
        },
        "fractal_optimization_scope": {
            "active_levels": ["bit", "byte", "frame", "pair"],
            "declared_levels": ["bit", "byte", "frame", "pair"],
        },
        "component_response_replayed": True,
        "component_response_replay": {
            "axis_tag": "[macOS-MLX research-signal]",
            "component_response_terms": {
                "segnet_delta_score_units": -0.0001,
                "posenet_delta_score_units": -0.0002,
                **_false_authority(),
            },
            **_false_authority(),
        },
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
        },
        "runtime_consumption_proof_path": str(source_proof_path),
        "receiver_contract_satisfied": False,
        "byte_closed_candidate_emitted": True,
        "readiness_blockers": [],
        **_false_authority(),
    }
    manifest_path = _write_json(tmp_path / "fec6_manifest.json", manifest)

    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "fec6_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    candidate = report["candidate_archive"]
    detected_families = report["archive_family_probe"]["detected_archive_families"]
    assert "fp11_frame_selector_wrapper" in detected_families
    assert "fec6_fixed_huffman_k16_selector" in detected_families
    assert report["selected_archive_transform_kind"] == "fec6_selector_payload_mutation"
    assert report["archive_entropy_substrate_coverage_schema"] == REPAIR_ARCHIVE_ENTROPY_SUBSTRATE_COVERAGE_SCHEMA
    coverage = report["archive_entropy_substrate_coverage"]
    assert coverage["compiler_position_coverage"]["before_coder"] is True
    assert "fec_variants" in coverage["materialized_substrates"]
    assert "selector_streams" in coverage["materialized_substrates"]
    assert "huffman_coding" in coverage["materialized_substrates"]
    assert coverage["probed_substrates"] == []
    assert coverage["prototype_substrates"] == ["range_coding", "ans_coding"]
    assert coverage["probed_entropy_estimated_zero_order_savings_bytes"] >= 0
    anti_pattern_ids = {protection["anti_pattern_id"] for protection in coverage["anti_pattern_protections"]}
    assert "proxy_or_advisory_probe_masquerades_as_score_authority_v1" in anti_pattern_ids
    assert "probe_only_side_report_orphaned_from_optimizer_v1" in anti_pattern_ids
    assert "zero_order_entropy_estimate_promoted_as_materialized_savings_v1" in anti_pattern_ids
    assert coverage["denied_uses"] == [
        "score_claim",
        "promotion",
        "rank_or_kill",
        "budget_spend",
        "exact_eval_dispatch",
        "archive_submission",
    ]
    rows_by_substrate = {row["substrate"]: row for row in coverage["rows"]}
    assert rows_by_substrate["range_coding"]["coverage_status"] == ("prototype_materialized_runtime_adapter_proven")
    assert rows_by_substrate["ans_coding"]["coverage_status"] == ("prototype_materialized_runtime_adapter_proven")
    archive_variants = {
        variant["archive_native_transform_kind"]: variant for variant in report["candidate_archive_transform_variants"]
    }
    assert report["archive_bound_candidate_contract_surface_schema"] == (
        ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
    )
    contract_surface = report["archive_bound_candidate_contract_surface"]
    assert contract_surface["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SURFACE_SCHEMA
    assert contract_surface["candidate_contract_count"] == report["candidate_archive_transform_variant_count"]
    assert contract_surface["archive_bound_ready_contract_count"] >= 3
    assert {"fec", "selector", "huffman", "range_coder", "ans_coder"} <= set(contract_surface["archive_substrate_tags"])
    selected_contract = report["archive_bound_candidate_contract"]
    assert selected_contract["schema"] == ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA
    assert selected_contract["selected_archive_transform_variant"] is True
    assert selected_contract["archive_bound_candidate_ready"] is True
    assert {"fec", "selector", "huffman"} <= set(selected_contract["archive_substrate_tags"])
    for variant in archive_variants.values():
        assert variant["archive_bound_candidate_contract"]["schema"] == (ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA)
    assert report["archive_variant_signal_surface_schema"] == REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA
    variant_surface = report["archive_variant_signal_surface"]
    assert variant_surface["schema"] == REPAIR_ARCHIVE_VARIANT_SIGNAL_SURFACE_SCHEMA
    assert variant_surface["row_count"] == report["candidate_archive_transform_variant_count"]
    assert variant_surface["probe_count"] == 2
    assert variant_surface["prototype_count"] == 2
    assert variant_surface["runtime_proof_ready_count"] >= 3
    assert variant_surface["non_selected_signal_count"] >= 4
    assert "range_coder_entropy_probe" in variant_surface["signal_transform_kinds"]
    assert "ans_coder_entropy_probe" in variant_surface["signal_transform_kinds"]
    assert "range_coder_lzma_prototype" in variant_surface["signal_transform_kinds"]
    assert "ans_coder_rans_prototype" in variant_surface["signal_transform_kinds"]
    variant_rows = {row["archive_native_transform_kind"]: row for row in variant_surface["variant_signal_rows"]}
    for row in variant_rows.values():
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
    assert variant_rows["range_coder_entropy_probe"]["signal_class"] == ("probe_only_entropy_signal")
    assert variant_rows["ans_coder_entropy_probe"]["signal_class"] == ("probe_only_entropy_signal")
    assert variant_rows["range_coder_lzma_prototype"]["signal_class"] == ("prototype_runtime_proven")
    assert variant_rows["ans_coder_rans_prototype"]["signal_class"] == ("prototype_runtime_proven")
    assert report["archive_variant_materializer_backlog_schema"] == REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA
    backlog = report["archive_variant_materializer_backlog"]
    assert backlog["schema"] == REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_SCHEMA
    assert backlog["row_count"] == 2
    assert backlog["probe_only_signal_count"] == 2
    assert backlog["executable_task_count"] == 2
    assert backlog["byte_closed_materialized_task_count"] == 2
    assert backlog["runtime_adapter_ready_task_count"] == 2
    assert backlog["contest_runtime_adapter_ready_task_count"] == 0
    assert backlog["contest_runtime_adapter_integration_required_count"] == 2
    assert backlog["smallest_contest_runtime_adapter_task_count"] == 2
    assert backlog["opened_by_pipeline"] is True
    assert backlog["pipeline_consumer"] == ("repair_campaign_entropy_stage_materializer_work_order_bundle")
    backlog_rows = {row["target_archive_transform_kind"]: row for row in backlog["task_rows"]}
    assert set(backlog_rows) == {
        "range_coder_lzma_prototype",
        "ans_coder_rans_prototype",
    }
    for row in backlog_rows.values():
        assert row["schema"] == REPAIR_ARCHIVE_VARIANT_MATERIALIZER_BACKLOG_ROW_SCHEMA
        assert row["queue_owned"] is True
        assert row["opened_by_pipeline"] is True
        assert row["smallest_byte_closed_materializer_task"] is True
        assert row["byte_closed_candidate_materialized"] is True
        assert row["runtime_consumption_proof_ready"] is True
        assert row["runtime_adapter_ready"] is True
        assert row["runtime_adapter_scope"] == "member_decode_helper_only"
        assert row["member_decode_helper_ready"] is True
        assert row["contest_runtime_decoder_adapter_ready"] is False
        assert row["contest_runtime_adapter_integrated"] is False
        assert row["contest_runtime_adapter_integration_required"] is True
        assert row["smallest_contest_runtime_adapter_task"] is True
        assert row["next_materializer_action"] == "integrate_contest_runtime_decoder_adapter"
        assert "contest_runtime_decoder_adapter_integration_missing" in row[
            "contest_runtime_adapter_integration_blockers"
        ]
        assert row["receiver_contract_satisfied"] is True
        assert row["score_claim"] is False
        assert row["promotion_eligible"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
    for transform_kind in ("range_coder_entropy_probe", "ans_coder_entropy_probe"):
        probe = archive_variants[transform_kind]
        assert probe["materialized"] is False
        assert probe["archive_native_transform_attempted"] is True
        assert probe["runtime_consumption_proof_ready"] is False
        assert probe["estimated_zero_order_savings_bytes"] >= 0
        assert (tmp_path / probe["entropy_probe_path"]).is_file()
    for transform_kind in ("range_coder_lzma_prototype", "ans_coder_rans_prototype"):
        prototype = archive_variants[transform_kind]
        assert prototype["materialized"] is True
        assert prototype["prototype_only"] is True
        assert prototype["archive_native_transform_attempted"] is True
        assert prototype["runtime_consumption_proof_ready"] is True
        assert prototype["receiver_contract_satisfied"] is True
        assert prototype["decoded_matches_source_member"] is True
        assert prototype["score_affecting_payload_changed"] is False
        assert (tmp_path / prototype["path"]).is_file()
        assert (tmp_path / prototype["runtime_consumption_proof_path"]).is_file()
        assert prototype["runtime_adapter_scope"] == "member_decode_helper_only"
        assert prototype["contest_runtime_decoder_adapter_ready"] is False
        assert prototype["contest_runtime_adapter_integrated"] is False
        assert prototype["runtime_adapter_manifest"]["schema"] == (
            REPAIR_ENTROPY_CODER_RUNTIME_ADAPTER_MANIFEST_SCHEMA
        )
        assert prototype["runtime_adapter_manifest"]["runtime_adapter_scope"] == (
            "member_decode_helper_only"
        )
        assert prototype["runtime_adapter_manifest"][
            "contest_runtime_decoder_adapter_ready"
        ] is False
        assert "contest_runtime_decoder_adapter_integration_missing" in prototype[
            "runtime_adapter_manifest"
        ]["readiness_blockers"]
        proof = json.loads(
            (tmp_path / prototype["runtime_consumption_proof_path"]).read_text(
                encoding="utf-8"
            )
        )
        assert proof["runtime_adapter_manifest"]["schema"] == (
            REPAIR_ENTROPY_CODER_RUNTIME_ADAPTER_MANIFEST_SCHEMA
        )
        assert proof["runtime_consumption_probe"]["decoder_adapter_invoked"] is True
        assert proof["proof_scope"] == "encoded_member_decodes_to_source_member_payload"
        assert proof["runtime_adapter_scope"] == "member_decode_helper_only"
        assert proof["contest_runtime_decoder_adapter_integrated"] is False
        assert proof["contest_runtime_adapter_integrated"] is False
        assert "contest_runtime_decoder_adapter_integration_missing" in " ".join(prototype.get("blockers", []))
        assert "contest_runtime_adapter_missing" not in " ".join(prototype.get("blockers", []))
    assert report["semantic_payload_changed"] is True
    assert report["score_affecting_payload_changed"] is False
    assert candidate["semantic_payload_changed"] is True
    assert candidate["score_affecting_payload_changed"] is False
    assert candidate["runtime_consumption_proof_ready"] is True
    assert report["runtime_consumption_proof_path"] == candidate["runtime_consumption_proof_path"]
    assert report["runtime_consumption_proof_path"] != str(source_proof_path)
    assert candidate["mutation_details"]["changed_pair_count"] > 0
    assert (tmp_path / candidate["path"]).is_file()
    assert (tmp_path / candidate["runtime_consumption_proof_path"]).is_file()


def test_repair_stack_search_demotes_stale_archive_bound_contract(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(
        tmp_path / "candidate_archive.zip",
        {"payload.bin": b"candidate"},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    proof_path = _write_json(
        tmp_path / "runtime_consumption_proof.json",
        {
            "schema": "family_agnostic_runtime_consumption_proof_v1",
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            **_false_authority(),
        },
    )
    report = {
        "schema": REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
        "family_id": "segnet_class_region_waterfill",
        "typed_response_id": "segnet_stale_contract",
        "candidate_chain_id": "segnet_stale_contract_chain",
        "candidate_chain_ids": ["segnet_stale_contract_chain"],
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "active_entropy_stage": {
            "order": 10,
            "stage": "before_entropy_coder_distribution_shaping",
        },
        "fractal_optimization_scope": {
            "active_levels": ["pixel", "region", "frame"],
            "declared_levels": ["pixel", "region", "frame"],
        },
        "allocated_repair_bytes": 16,
        "byte_transform_delta": {
            "schema": "repair_family_byte_transform_delta.v1",
            "path": "delta.json",
            "bytes": 16,
            **_false_authority(),
        },
        "mlx_local_probe_delta": {
            "schema": "repair_family_byte_transform_mlx_probe_delta.v1",
            "combined_delta_score_units": -0.001,
            "segnet_delta_score_units": -0.001,
            "posenet_delta_score_units": 0.0,
            **_false_authority(),
        },
        "byte_closed_candidate_emitted": True,
        "candidate_archive_materialized": True,
        "candidate_archive": {
            "path": archive_path.relative_to(tmp_path).as_posix(),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
            "runtime_consumption_proof_path": proof_path.relative_to(tmp_path).as_posix(),
        },
        "runtime_consumption_proof_path": proof_path.relative_to(tmp_path).as_posix(),
        "runtime_consumption_proof_ready": True,
        "receiver_contract_satisfied": True,
        "exact_eval_handoff_gate": {
            "schema": "repair_family_exact_eval_handoff_gate.v1",
            "archive_bound_runtime_consumption_proof_ready": True,
            "eligible_for_exact_eval_handoff": False,
            "blockers": ["contest_exact_auth_eval_required"],
            **_false_authority(),
        },
        "blockers": [],
        **_false_authority(),
    }
    report.update(archive_bound_candidate_contract_fields_for_row(report, repo_root=tmp_path))
    report["receiver_contract_satisfied"] = False
    report_path = _write_json(tmp_path / "stale_contract_report.json", report)

    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        byte_credit_budget=1_000,
    )

    stack_row = stack_plan["stack_rows"][0]
    handoff_row = stack_plan["exact_eval_handoff_candidates"][0]
    assert stack_row["archive_bound_contract_ready"] is False
    assert handoff_row["archive_bound_custody_complete"] is False
    assert stack_plan["archive_bound_exact_handoff_candidate_count"] == 0
    assert any(
        "archive_bound_contract_stale_duplicate_field:receiver_contract_satisfied" in blocker
        for blocker in handoff_row["blockers"]
    )


@pytest.mark.parametrize(
    ("archive_family", "payload"),
    [
        (
            "fec3_compact_selector",
            _fec3_selector_payload([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
        ),
        (
            "fec5_fixed_huffman_k8_selector",
            _fec5_selector_payload([0, 1, 2, 3, 4, 5, 6, 7, 0, 1]),
        ),
        (
            "fes1_all_none_selector",
            _fes1_selector_payload([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ),
    ],
)
def test_byte_transform_executor_mutates_non_fec6_fp11_selector_payloads(
    tmp_path: Path,
    archive_family: str,
    payload: bytes,
) -> None:
    archive_path = _write_zip(tmp_path / f"{archive_family}.zip", {"x": payload})
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "family_id": "segnet_class_region_waterfill",
        "typed_response_id": f"{archive_family}_ready",
        "candidate_chain_id": f"{archive_family}_chain",
        "candidate_chain_ids": [f"{archive_family}_chain"],
        "component_response_replay": {
            "component_response_terms": {
                "segnet_delta_score_units": -0.001,
                "combined_delta_score_units": -0.001,
            },
            "local_mlx_custody_paths": [],
            **_false_authority(),
        },
        "receiver_verification": {
            "local_mlx_custody_paths": [],
            **_false_authority(),
        },
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
        },
        "receiver_contract_satisfied": False,
        "byte_closed_candidate_emitted": True,
        "readiness_blockers": [],
        **_false_authority(),
    }
    manifest_path = _write_json(tmp_path / f"{archive_family}_manifest.json", manifest)

    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / f"{archive_family}_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    candidate = report["candidate_archive"]
    detected_families = report["archive_family_probe"]["detected_archive_families"]
    assert archive_family in detected_families
    assert report["selected_archive_transform_kind"] == "fp11_selector_payload_mutation"
    assert report["archive_entropy_substrate_coverage"]["compiler_position_coverage"]["before_coder"] is True
    assert report["semantic_payload_changed"] is True
    assert report["score_affecting_payload_changed"] is False
    assert candidate["semantic_payload_changed"] is True
    assert candidate["score_affecting_payload_changed"] is False
    assert candidate["runtime_consumption_proof_ready"] is True
    assert candidate["mutation_details"]["changed_pair_count"] > 0
    assert (tmp_path / candidate["path"]).is_file()
    assert (tmp_path / candidate["runtime_consumption_proof_path"]).is_file()


def test_byte_transform_executor_mutates_fec8_fp11_selector_payload(
    tmp_path: Path,
) -> None:
    archive_family = "fec8_static_second_order_k16_selector"
    archive_path = _write_zip(
        tmp_path / "fec8.zip",
        {"x": _fec8_selector_payload([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "family_id": "segnet_class_region_waterfill",
        "typed_response_id": "fec8_ready",
        "candidate_chain_id": "fec8_chain",
        "candidate_chain_ids": ["fec8_chain"],
        "component_response_replay": {
            "component_response_terms": {
                "segnet_delta_score_units": -0.001,
                "combined_delta_score_units": -0.001,
            },
            "local_mlx_custody_paths": [],
            **_false_authority(),
        },
        "receiver_verification": {
            "local_mlx_custody_paths": [],
            **_false_authority(),
        },
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
        },
        "receiver_contract_satisfied": False,
        "byte_closed_candidate_emitted": True,
        "readiness_blockers": [],
        **_false_authority(),
    }
    manifest_path = _write_json(tmp_path / "fec8_manifest.json", manifest)

    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "fec8_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=REPO_ROOT,
        allow_overwrite=False,
    )

    candidate = report["candidate_archive"]
    assert archive_family in report["archive_family_probe"]["detected_archive_families"]
    assert report["selected_archive_transform_kind"] == "fp11_selector_payload_mutation"
    assert candidate["runtime_consumption_proof_ready"] is True
    assert candidate["semantic_payload_changed"] is True
    assert candidate["mutation_details"]["selector_magic"] == "FEC8"


def test_byte_transform_executor_mutates_psv4_selector_packet(
    tmp_path: Path,
) -> None:
    archive_path = _write_zip(
        tmp_path / "psv4.zip",
        {
            "0.bin": _psv4_packet_payload([0, 0, 2, 2, 7, 7, 13, 13, 0, 0, 2, 2]),
            "inflate.py": b"",
        },
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "family_id": "segnet_class_region_waterfill",
        "typed_response_id": "psv4_ready",
        "candidate_chain_id": "psv4_chain",
        "candidate_chain_ids": ["psv4_chain"],
        "component_response_replay": {
            "component_response_terms": {
                "segnet_delta_score_units": -0.001,
                "combined_delta_score_units": -0.001,
            },
            "local_mlx_custody_paths": [],
            **_false_authority(),
        },
        "receiver_verification": {
            "local_mlx_custody_paths": [],
            **_false_authority(),
        },
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
        },
        "receiver_contract_satisfied": False,
        "byte_closed_candidate_emitted": True,
        "readiness_blockers": [],
        **_false_authority(),
    }
    manifest_path = _write_json(tmp_path / "psv4_manifest.json", manifest)

    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "psv4_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    candidate = report["candidate_archive"]
    assert "pact_nerv_selector_v4_packet" in report["archive_family_probe"]["detected_archive_families"]
    assert report["selected_archive_transform_kind"] == "psv4_selector_payload_mutation"
    assert "header_elision_or_rewrite" in report["archive_entropy_substrate_coverage"]["materialized_substrates"]
    assert report["semantic_payload_changed"] is True
    assert report["score_affecting_payload_changed"] is False
    assert candidate["runtime_consumption_proof_ready"] is True
    assert candidate["semantic_payload_changed"] is True
    assert candidate["score_affecting_payload_changed"] is False
    assert candidate["mutation_details"]["selector_magic"] == "PSV4"
    assert candidate["mutation_details"]["changed_pair_count"] > 0
    assert (tmp_path / candidate["path"]).is_file()
    assert (tmp_path / candidate["runtime_consumption_proof_path"]).is_file()


def test_repair_family_byte_transform_cli_writes_report_and_bundle(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)
    report_path = tmp_path / "byte_transform_report.json"
    bundle_path = tmp_path / "byte_transform_bundle.json"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_repair_family_byte_transform_executor.py"),
            "--family-materializer-manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "byte_transform"),
            "--execution-report-out",
            str(report_path),
            "--replay-bundle-out",
            str(bundle_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert report["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA
    assert bundle["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA
    assert report["score_claim"] is False


def test_entropy_stage_chain_executor_composes_selected_archive_stages(
    tmp_path: Path,
) -> None:
    source_archive = _write_zip(
        tmp_path / "chain_source.zip",
        {"x": _fec6_selector_payload([0, 2, 7, 13, 0, 2, 7, 13, 0, 2, 7, 13])},
    )
    source_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()

    reports: list[dict[str, object]] = []
    report_paths: list[Path] = []
    for index, (family_id, typed_response_id, stage_order) in enumerate(
        (
            ("segnet_class_region_waterfill", "segnet_region_chain_ready", 10),
            ("per_region_selector_codec", "selector_codec_chain_ready", 30),
        ),
        start=1,
    ):
        manifest = {
            "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
            "materializer_id": f"repair_family_materializer:{family_id}",
            "target_kind": family_id,
            "family_id": family_id,
            "typed_response_id": typed_response_id,
            "candidate_chain_id": f"{family_id}_chain",
            "candidate_chain_ids": [f"{family_id}_chain"],
            "repair_budget_candidate_chain_id": f"{family_id}_chain",
            "repair_budget_candidate_chain_ids": [f"{family_id}_chain"],
            "entropy_position_label": (
                "selector_codec_entropy"
                if family_id == "per_region_selector_codec"
                else "before_entropy_coder_distribution_shaping"
            ),
            "active_entropy_stage": {
                "order": stage_order,
                "stage": (
                    "selector_codec_entropy"
                    if family_id == "per_region_selector_codec"
                    else "before_entropy_coder_distribution_shaping"
                ),
            },
            "fractal_optimization_scope": {
                "active_levels": ["bit", "byte", "region", "frame", "pair"],
                "declared_levels": ["bit", "byte", "region", "frame", "pair"],
            },
            "component_response_replayed": True,
            "component_response_replay": {
                "replayed": True,
                "axis_tag": "[macOS-MLX research-signal]",
                "component_response_terms": {
                    "combined_delta_score_units": -0.001 * index,
                    "segnet_delta_score_units": -0.001 * index,
                    "posenet_delta_score_units": 0.0,
                    **_false_authority(),
                },
                **_false_authority(),
            },
            "receiver_contract_satisfied": False,
            "candidate_archive": {
                "path": str(source_archive),
                "sha256": source_sha,
                "bytes": source_archive.stat().st_size,
            },
            "byte_closed_candidate_emitted": True,
            "readiness_blockers": [],
            **_false_authority(),
        }
        manifest_path = _write_json(tmp_path / f"{family_id}_manifest.json", manifest)
        report, _bundle = build_repair_family_byte_transform_execution_report(
            family_materializer_manifest=manifest,
            family_materializer_manifest_path=manifest_path,
            output_dir=tmp_path / f"{family_id}_leaf_transform",
            replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
            invocation_argv=["pytest"],
            repo_root=tmp_path,
            allow_overwrite=False,
        )
        report_path = _write_json(tmp_path / f"{family_id}_report.json", report)
        reports.append(report)
        report_paths.append(report_path)

    work_order_bundle = {
        "schema": "repair_campaign_entropy_stage_materializer_work_order_bundle.v1",
        "stage_order": ["before_coder", "coder_boundary", "after_coder"],
        "work_order_count": 2,
        "work_orders": [
            {
                "schema": "repair_campaign_entropy_stage_materializer_work_order.v1",
                "compiler_stage": "before_coder",
                "stage_materialization_order": 1,
                "family_order": ["segnet_class_region_waterfill"],
                "typed_response_order": ["segnet_region_chain_ready"],
                **_false_authority(),
            },
            {
                "schema": "repair_campaign_entropy_stage_materializer_work_order.v1",
                "compiler_stage": "coder_boundary",
                "stage_materialization_order": 2,
                "family_order": ["per_region_selector_codec"],
                "typed_response_order": ["selector_codec_chain_ready"],
                **_false_authority(),
            },
        ],
        **_false_authority(),
    }

    chain_bundle = build_repair_entropy_stage_chain_execution_bundle(
        execution_reports=reports,
        execution_report_paths=report_paths,
        work_order_bundle=work_order_bundle,
        output_dir=tmp_path / "entropy_stage_chain",
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    assert chain_bundle["schema"] == REPAIR_ENTROPY_STAGE_CHAIN_EXECUTION_BUNDLE_SCHEMA
    assert chain_bundle["chain_count"] == 1
    assert chain_bundle["materialized_chain_candidate_count"] == 1
    assert chain_bundle["archive_entropy_substrate_coverage_count"] == 2
    chain_report = chain_bundle["chain_reports"][0]
    assert chain_report["stage_count"] == 2
    assert chain_report["planned_stage_count"] == 2
    assert [stage["family_id"] for stage in chain_report["stages"]] == [
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
    ]
    assert chain_report["archive_bound_candidate_emitted"] is True
    assert chain_report["runtime_consumption_proof_ready"] is True
    final_archive = chain_report["candidate_archive"]
    assert final_archive["sha256"] != source_sha
    assert (tmp_path / final_archive["path"]).is_file()
    assert chain_bundle["ready_for_exact_eval_dispatch"] is False
    exact_handoff = build_repair_family_exact_handoff_plan(
        stack_plan={
            "schema": REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA,
            "execution_report_count": 0,
            "exact_eval_handoff_candidates": [],
            "exact_eval_handoff_gate": {
                "blockers": [],
                **_false_authority(),
            },
            **_false_authority(),
        },
        chain_execution_bundle=chain_bundle,
    )
    assert exact_handoff["entropy_stage_chain_candidate_count"] == 1
    assert exact_handoff["entropy_stage_chain_archive_bound_candidate_count"] == 1
    assert exact_handoff["archive_bound_candidate_count"] == 1
    assert exact_handoff["archive_bound_rows"][0]["family_id"] == "entropy_stage_chain"
    assert exact_handoff["archive_bound_rows"][0]["candidate_archive"]["sha256"] == final_archive["sha256"]


@pytest.mark.parametrize(
    "family_id",
    [
        "posenet_null_bottom_decile",
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
        "frame0_k16_palette_asymmetry",
        "entropy_boundary_probe",
    ],
)
def test_byte_transform_executor_supports_all_queue_owned_repair_families(
    tmp_path: Path,
    family_id: str,
) -> None:
    archive_path = _write_zip(
        tmp_path / f"{family_id}_source_archive.zip",
        {"0.bin": (family_id.encode("utf-8") * 64)},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest = {
        "schema": REPAIR_CAMPAIGN_FAMILY_MATERIALIZER_MANIFEST_SCHEMA,
        "materializer_id": f"repair_family_materializer:{family_id}",
        "target_kind": family_id,
        "family_id": family_id,
        "typed_response_id": f"{family_id}_typed",
        "candidate_chain_id": f"{family_id}_chain",
        "candidate_chain_ids": [f"{family_id}_chain"],
        "repair_budget_candidate_chain_id": f"{family_id}_chain",
        "repair_budget_candidate_chain_ids": [f"{family_id}_chain"],
        "entropy_position_label": "before_entropy_coder_distribution_shaping",
        "active_entropy_stage": {
            "order": 10,
            "stage": "before_entropy_coder_distribution_shaping",
            "class": "pre_entropy_distribution_shaping",
        },
        "fractal_optimization_scope": {
            "active_levels": ["bit", "byte", "pixel", "region", "frame"],
            "declared_levels": ["bit", "byte", "pixel", "region", "frame"],
        },
        "component_response_replayed": True,
        "component_response_replay": {
            "replayed": True,
            "axis_tag": "[macOS-MLX research-signal]",
            "component_response_terms": {
                "segnet_delta_score_units": -0.0001,
                "posenet_delta_score_units": -0.0002,
                **_false_authority(),
            },
            **_false_authority(),
        },
        "receiver_contract_satisfied": False,
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": archive_sha,
            "bytes": archive_path.stat().st_size,
        },
        "byte_closed_candidate_emitted": True,
        "readiness_blockers": [],
        **_false_authority(),
    }
    manifest_path = _write_json(tmp_path / f"{family_id}.json", manifest)

    report, bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / family_id,
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )

    assert report["family_id"] == family_id
    assert report["byte_transform_supported"] is True
    assert report["byte_transform_delta"]["transform_kind"]
    assert report["byte_closed_candidate_emitted"] is True
    assert report["candidate_archive"]["runtime_consumption_proof_ready"] is True
    assert report["exact_eval_handoff_gate"]["archive_bound_runtime_consumption_proof_ready"] is True
    assert report["ready_for_exact_eval_dispatch"] is False
    assert bundle["schema"] == REPAIR_FAMILY_BYTE_TRANSFORM_REPLAY_BUNDLE_SCHEMA


def test_repair_family_stack_search_demotes_negative_posterior(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)
    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    posterior_path = tmp_path / "posterior.jsonl"
    posterior_path.write_text(
        json.dumps(
            {
                "schema": "repair_campaign_stackability_posterior_row.v1",
                "typed_response_id": "segnet_negative",
                "candidate_id": "segnet_class_region_waterfill",
                "family_id": "segnet_class_region_waterfill",
                "acquisition_policy_delta": {
                    "recommended_acquisition_policy": ("decrease_family_priority_until_new_component_response_signal"),
                    "family_priority_direction": "decrease",
                    **_false_authority(),
                },
                "blockers": ["non_improving_local_objective_delta"],
                **_false_authority(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        posterior_path=posterior_path,
        byte_credit_budget=10_000,
    )

    assert stack_plan["schema"] == REPAIR_FAMILY_STACK_SEARCH_PLAN_SCHEMA
    row = stack_plan["stack_rows"][0]
    assert row["automatic_negative_result_demoted"] is True
    assert "automatic_negative_result_demotion_active" in row["blockers"]
    assert row["interaction_feature_vector"]["segnet_region_family"] is True
    assert stack_plan["interaction_tensor"]["cell_count"] == 1
    assert stack_plan["interaction_tensor"]["cells"][0]["negative_demoted_count"] == 1
    assert stack_plan["posterior_acquisition_surface"]["family_prior_count"] == 1
    assert row["posterior_acquisition_multiplier"] < 1.0
    assert stack_plan["budget_routing_decision"]["activation_action"] in {
        "demote_repair_family_until_new_component_signal",
        "run_next_byte_closed_materializer_or_mlx_probe",
    }
    assert stack_plan["ready_for_exact_eval_dispatch"] is False


def test_repair_family_stack_search_builds_pairwise_tensor_acquisition_path(
    tmp_path: Path,
) -> None:
    reports = []
    report_paths = []
    for family_id, stage, levels, delta, byte_count in (
        (
            "segnet_class_region_waterfill",
            10,
            ["pixel", "boundary", "region", "frame"],
            -0.0020,
            32,
        ),
        (
            "per_region_selector_codec",
            20,
            ["bit", "byte", "boundary", "region", "pair", "batch"],
            -0.0010,
            12,
        ),
    ):
        report = {
            "schema": REPAIR_FAMILY_BYTE_TRANSFORM_EXECUTION_REPORT_SCHEMA,
            "family_id": family_id,
            "typed_response_id": f"{family_id}_typed",
            "candidate_chain_id": f"{family_id}_chain",
            "candidate_chain_ids": [f"{family_id}_chain"],
            "entropy_position_label": "before_entropy_coder_distribution_shaping",
            "active_entropy_stage": {
                "order": stage,
                "stage": "before_entropy_coder_distribution_shaping",
            },
            "fractal_optimization_scope": {
                "active_levels": levels,
                "declared_levels": levels,
            },
            "allocated_repair_bytes": byte_count,
            "byte_transform_delta": {
                "schema": "repair_family_byte_transform_delta.v1",
                "path": f"{family_id}.json",
                "bytes": byte_count,
                **_false_authority(),
            },
            "mlx_local_probe_delta": {
                "schema": "repair_family_byte_transform_mlx_probe_delta.v1",
                "combined_delta_score_units": delta,
                "segnet_delta_score_units": delta,
                "posenet_delta_score_units": 0.0,
                **_false_authority(),
            },
            "byte_closed_candidate_emitted": False,
            "candidate_archive_materialized": False,
            "exact_eval_handoff_gate": {
                "schema": "repair_family_exact_eval_handoff_gate.v1",
                "archive_bound_runtime_consumption_proof_ready": False,
                "blockers": ["byte_closed_candidate_archive_missing"],
                **_false_authority(),
            },
            "blockers": [],
            **_false_authority(),
        }
        report_path = _write_json(tmp_path / f"{family_id}_report.json", report)
        reports.append(report)
        report_paths.append(report_path)

    stack_plan = plan_repair_family_stack_search(
        execution_reports=reports,
        execution_report_paths=report_paths,
        repo_root=tmp_path,
        byte_credit_budget=64,
    )

    assert stack_plan["pairwise_interaction_tensor_cell_count"] == 2
    assert stack_plan["n_way_hypergraph_acquisition_enabled"] is True
    assert stack_plan["hypergraph_interaction_tensor_cell_count"] == 1
    assert stack_plan["fractal_marginal_surface_cell_count"] >= 6
    assert stack_plan["fractal_marginal_surface"]["measured_mlx_marginal_update_count"] == 10
    assert stack_plan["measured_mlx_posterior_budget_routing_update_count"] == 10
    assert stack_plan["budget_routing_decision"]["measured_mlx_posterior_budget_routing_update_count"] == 10
    marginal_levels = {cell["level"] for cell in stack_plan["fractal_marginal_surface"]["cells"]}
    assert {"bit", "byte", "boundary", "region"} <= marginal_levels
    assert all(cell["measured_mlx_marginal_updates"] for cell in stack_plan["fractal_marginal_surface"]["cells"])
    frontier = stack_plan["stack_acquisition_frontier"]
    assert stack_plan["stack_acquisition_frontier_count"] == 1
    assert frontier[0]["source_tensor"] == "hypergraph_interaction_tensor"
    assert frontier[0]["family_order"] == [
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
    ]
    hyper_cell = stack_plan["hypergraph_interaction_tensor"]["cells"][0]
    assert hyper_cell["hyperedge_order"] == 2
    assert hyper_cell["transition_allowed_by_tensor"] is True
    assert hyper_cell["coupling_synergy"] > 0.0
    pair_cells = stack_plan["pairwise_interaction_tensor"]["cells"]
    forward_cell = next(
        cell
        for cell in pair_cells
        if cell["left_family_id"] == "segnet_class_region_waterfill"
        and cell["right_family_id"] == "per_region_selector_codec"
    )
    assert forward_cell["entropy_stage_gap"] == 10
    assert forward_cell["region_selector_coupling"] is True
    assert forward_cell["region_boundary_coupling"] is True
    assert forward_cell["coupling_synergy"] > 0.0
    assert forward_cell["transition_allowed_by_tensor"] is True
    primary_path = stack_plan["primary_stack_acquisition_path"]
    assert primary_path["path_kind"] == "n_way_hypergraph_interaction_tensor_acquisition"
    assert primary_path["family_order"] == [
        "segnet_class_region_waterfill",
        "per_region_selector_codec",
    ]
    assert primary_path["terminal_outcome_class"] == "precise_exact_axis_blocker"
    assert primary_path["total_delta_payload_bytes"] == 44
    assert all(row["selected_for_materialization_handoff"] for row in stack_plan["stack_rows"])
    assert stack_plan["bounded_autonomous_terminal_policy"]["terminal_outcome_class"] == "precise_exact_axis_blocker"
    assert stack_plan["ready_for_exact_eval_dispatch"] is False


def test_repair_stack_learning_signal_report_closes_posterior_loop(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    manifest_path = _write_json(tmp_path / "family_manifest.json", manifest)
    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=manifest_path,
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        byte_credit_budget=1,
    )
    learning_report = build_repair_family_stack_learning_signal_report(
        stack_plan=stack_plan,
    )

    assert learning_report["schema"] == REPAIR_FAMILY_STACK_LEARNING_SIGNAL_REPORT_SCHEMA
    assert learning_report["learning_signal_count"] == 1
    signal = learning_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]
    assert update["recommended_acquisition_policy"] == (
        "increase_receiver_closed_rate_credit_or_rebudget_earlier_entropy_stage"
    )
    assert update["planner_feature_vector"]["receiver_credit_exhausted"] is True


def test_repair_exact_ready_bridge_emits_blocked_source_queue(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    archive_path = _write_zip(
        tmp_path / "source_archive.zip",
        {"0.bin": (b"segnet-region-waterfill" * 64)},
    )
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    plan["candidate_chain_rows"][1]["candidate_archive_path"] = str(archive_path)
    plan["candidate_chain_rows"][1]["candidate_archive_sha256"] = archive_sha
    plan["candidate_chain_rows"][1]["candidate_archive_bytes"] = archive_path.stat().st_size
    manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    report, _bundle = build_repair_family_byte_transform_execution_report(
        family_materializer_manifest=manifest,
        family_materializer_manifest_path=_write_json(
            tmp_path / "family_manifest.json",
            manifest,
        ),
        output_dir=tmp_path / "byte_transform",
        replay_argv=["python", "tools/run_repair_family_byte_transform_executor.py"],
        invocation_argv=["pytest"],
        repo_root=tmp_path,
        allow_overwrite=False,
    )
    report_path = _write_json(tmp_path / "byte_transform_report.json", report)
    stack_plan = plan_repair_family_stack_search(
        execution_reports=[report],
        execution_report_paths=[report_path],
        repo_root=tmp_path,
        byte_credit_budget=10_000,
    )
    exact_handoff_plan = build_repair_family_exact_handoff_plan(
        stack_plan=stack_plan,
        stack_plan_path=tmp_path / "repair_family_exact_handoff_plan.json",
    )

    bridge = build_repair_family_exact_ready_bridge(
        exact_handoff_plan=exact_handoff_plan,
        exact_handoff_plan_path=tmp_path / "repair_family_exact_handoff_plan.json",
        repo_root=tmp_path,
    )

    bridge_report = bridge["bridge_report"]
    source_queue = bridge["source_optimizer_queue"]
    blocked_queue = bridge["blocked_exact_ready_queue"]
    assert bridge_report["schema"] == REPAIR_FAMILY_EXACT_READY_BRIDGE_REPORT_SCHEMA
    assert bridge_report["candidate_count"] == 1
    assert bridge_report["archive_custody_proven_count"] == 1
    assert bridge_report["runtime_proof_custody_proven_count"] == 1
    assert bridge_report["runtime_content_tree_custody_proven_count"] == 0
    assert source_queue["schema"] == "optimizer_candidate_queue_v1"
    assert source_queue["dispatch_ready"] == []
    assert blocked_queue["schema"] == "optimizer_candidate_exact_eval_ready_queue_v1"
    assert blocked_queue["dispatch_ready_count"] == 0
    source_row = source_queue["top_k"][0]
    assert source_row["target_modes"] == ["contest_exact_eval"]
    assert source_row["ready_for_exact_eval_dispatch"] is False
    assert source_row["archive_bound_candidate_contract"]["schema"] == (ARCHIVE_BOUND_CANDIDATE_CONTRACT_SCHEMA)
    assert "submission_dir_missing_for_runtime_content_tree_custody" in source_row["dispatch_blockers"]


def test_repair_exact_ready_bridge_rejects_stale_archive_contract_row(
    tmp_path: Path,
) -> None:
    source_archive = _write_zip(tmp_path / "source_archive.zip", {"0.bin": b"source"})
    source_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    candidate_archive = _write_zip(
        tmp_path / "candidate_archive.zip",
        {"0.bin": b"candidate"},
    )
    candidate_sha = hashlib.sha256(candidate_archive.read_bytes()).hexdigest()
    proof_path = _write_json(
        tmp_path / "receiver_proof.json",
        {
            "schema": "fixture_runtime_consumption_proof.v1",
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "source_archive": {
                "path": str(source_archive),
                "sha256": source_sha,
                "bytes": source_archive.stat().st_size,
            },
        },
    )
    proof_sha = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    handoff_row: dict[str, object] = {
        **_false_authority(),
        "family_id": "segnet_region",
        "typed_response_id": "segnet_region_ready",
        "candidate_chain_id": "segnet_region_chain",
        "candidate_archive": {
            "path": str(candidate_archive),
            "sha256": candidate_sha,
            "bytes": candidate_archive.stat().st_size,
        },
        "runtime_consumption_proof": {
            "path": str(proof_path),
            "sha256": proof_sha,
            "bytes": proof_path.stat().st_size,
        },
        "runtime_consumption_proof_path": str(proof_path),
        "runtime_consumption_proof_status": "present",
        "receiver_contract_satisfied": True,
        "runtime_adapter_ready": True,
        "byte_closed_candidate_emitted": True,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
    }
    handoff_row.update(
        archive_bound_candidate_contract_fields_for_row(
            handoff_row,
            repo_root=tmp_path,
            family_id="segnet_region",
            typed_response_id="segnet_region_ready",
            candidate_chain_id="segnet_region_chain",
        )
    )
    handoff_row["receiver_contract_satisfied"] = False

    with pytest.raises(
        RepairFamilyExactReadyBridgeError,
        match="archive_bound_contract_stale_duplicate_field:receiver_contract_satisfied",
    ):
        build_repair_family_exact_ready_bridge(
            exact_handoff_plan={
                "schema": REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
                "archive_bound_rows": [handoff_row],
                **_false_authority(),
            },
            repo_root=tmp_path,
        )


def test_repair_exact_ready_bridge_consumes_contract_only_custody(
    tmp_path: Path,
) -> None:
    source_archive = _write_zip(tmp_path / "source_archive.zip", {"0.bin": b"source"})
    source_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
    candidate_archive = _write_zip(
        tmp_path / "candidate_archive.zip",
        {"0.bin": b"candidate"},
    )
    candidate_sha = hashlib.sha256(candidate_archive.read_bytes()).hexdigest()
    proof_path = _write_json(
        tmp_path / "receiver_proof.json",
        {
            "schema": "fixture_runtime_consumption_proof.v1",
            "receiver_contract_satisfied": True,
            "runtime_consumption_proof_passed": True,
            "source_archive": {
                "path": str(source_archive),
                "sha256": source_sha,
                "bytes": source_archive.stat().st_size,
            },
            **_false_authority(),
        },
    )
    raw_handoff_row: dict[str, object] = {
        **_false_authority(),
        "family_id": "segnet_region",
        "typed_response_id": "segnet_region_ready",
        "candidate_chain_id": "segnet_region_chain",
        "candidate_archive": {
            "path": str(candidate_archive),
            "sha256": candidate_sha,
            "bytes": candidate_archive.stat().st_size,
        },
        "runtime_consumption_proof": {
            "path": str(proof_path),
            "bytes": proof_path.stat().st_size,
        },
        "runtime_consumption_proof_path": str(proof_path),
        "runtime_consumption_proof_status": "present",
        "receiver_contract_satisfied": True,
        "runtime_adapter_ready": True,
        "byte_closed_candidate_emitted": True,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
    }
    raw_handoff_row.update(
        archive_bound_candidate_contract_fields_for_row(
            raw_handoff_row,
            repo_root=tmp_path,
            family_id="segnet_region",
            typed_response_id="segnet_region_ready",
            candidate_chain_id="segnet_region_chain",
        )
    )
    contract_only_handoff_row = {
        **_false_authority(),
        "family_id": "segnet_region",
        "typed_response_id": "segnet_region_ready",
        "candidate_chain_id": "segnet_region_chain",
        "archive_bound_candidate_contract": raw_handoff_row["archive_bound_candidate_contract"],
        "archive_bound_candidate_contract_schema": raw_handoff_row["archive_bound_candidate_contract_schema"],
        "archive_bound_candidate_contract_surface": raw_handoff_row["archive_bound_candidate_contract_surface"],
        "archive_bound_candidate_contract_surface_schema": raw_handoff_row[
            "archive_bound_candidate_contract_surface_schema"
        ],
    }

    bridge = build_repair_family_exact_ready_bridge(
        exact_handoff_plan={
            "schema": REPAIR_FAMILY_EXACT_HANDOFF_PLAN_SCHEMA,
            "archive_bound_rows": [contract_only_handoff_row],
            **_false_authority(),
        },
        repo_root=tmp_path,
    )

    bridge_report = bridge["bridge_report"]
    assert bridge_report["candidate_count"] == 1
    assert bridge_report["archive_custody_proven_count"] == 1
    assert bridge_report["runtime_proof_custody_proven_count"] == 1
    bridge_row = bridge_report["rows"][0]
    assert bridge_row["archive_custody"]["sha256"] == candidate_sha
    assert bridge_row["runtime_consumption_proof_custody"]["present"] is True
    source_row = bridge["source_optimizer_queue"]["top_k"][0]
    assert source_row["candidate_archive_sha256"] == candidate_sha
    assert source_row["source_archive_sha256"] == source_sha
    assert "archive_bound_candidate_contract_missing" not in source_row["dispatch_blockers"]


def test_materialization_gate_learning_signal_updates_posterior(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    plan = _plan_from_score_report(score_report)
    family_manifest = build_repair_campaign_family_materializer_manifest(
        repo_root=tmp_path,
        materialization_plan=plan,
        score_report=score_report,
        materialization_plan_path=tmp_path / "plan.json",
        score_report_path=tmp_path / "score_report.json",
        typed_response_id="segnet_region_ready",
        candidate_id="segnet_class_region_waterfill",
    )
    family_manifest_path = _write_json(tmp_path / "family_manifest.json", family_manifest)
    execution_report = {
        "schema": "frontier_rate_attack_repair_budget_materialization_execution_report.v1",
        "chain_id": "unit_repair_chain",
        "candidate_archive_materialized": False,
        "runtime_consumption_proof_present": False,
        "receiver_consumed": False,
        "component_response_replayed": True,
        "execution_rows": [
            {
                "schema": "frontier_rate_attack_repair_budget_materialization_execution_row.v1",
                "candidate_kind": "spent_budget_repair_child",
                "candidate_chain_id": "repair_budget_spent_child_unit_segnet",
                "candidate_archive_materialized": False,
                "runtime_consumption_proof_present": False,
                "receiver_consumed": False,
                "component_response_replayed": True,
                "component_response_replay_axis_tag": "[macOS-MLX research-signal]",
                "blockers": ["candidate_archive_materialized_false"],
                **_false_authority(),
            }
        ],
        "blockers": ["candidate_archives_not_materialized"],
        **_false_authority(),
    }
    gate = {
        "schema": "repair_campaign_byte_closed_materialization_gate.v1",
        "typed_response_id": "segnet_region_ready",
        "candidate_id": "segnet_class_region_waterfill",
        "candidate_archive_materialized": False,
        "archive_bound_runtime_consumption_proof_ready": False,
        "component_response_replayed": True,
        "blockers": ["candidate_archive_materialized_false"],
        **_false_authority(),
    }
    execution_report_path = _write_json(tmp_path / "execution_report.json", execution_report)
    gate_path = _write_json(tmp_path / "gate.json", gate)

    signal_report = build_repair_campaign_materialization_learning_signal_report(
        materialization_execution_report_path=execution_report_path,
        materialization_execution_report=execution_report,
        materialization_gate_path=gate_path,
        materialization_gate=gate,
        family_materializer_manifest_path=family_manifest_path,
        family_materializer_manifest=family_manifest,
        repo_root=tmp_path,
    )

    assert signal_report["schema"] == REPAIR_CAMPAIGN_BLOCKED_LEARNING_SIGNAL_REPORT_SCHEMA
    signal = signal_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]
    assert update["recommended_acquisition_policy"] == ("prioritize_byte_closed_family_materializer_implementation")
    assert update["planner_feature_vector"]["entropy_stage_order"] == 10
    signal_report_path = _write_json(tmp_path / "signal_report.json", signal_report)
    posterior_path = tmp_path / "posterior.jsonl"
    append_report = append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=tmp_path / ".posterior.lock",
        repo_root=tmp_path,
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(posterior_path)
    summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )

    assert append_report["appended_count"] == 1
    assert posterior_rows[0]["typed_response_id"] == "segnet_region_ready"
    assert posterior_rows[0]["acquisition_policy_delta"]["family_priority_direction"] == "increase"
    assert (
        posterior_rows[0]["acquisition_policy_delta"]["posterior_budget_routing_hint"]
        == "route_budget_to_byte_closed_materializer_after_custody"
    )
    route = summary["acquisition_followup_routes"][0]
    assert route["activation_action"] == "implement_or_run_repair_family_byte_transform"


def test_blocked_credit_exhaustion_updates_posterior_budget_routing(
    tmp_path: Path,
) -> None:
    payload = _repair_payload(tmp_path)
    payload["receiver_closed_rate_credit"]["receiver_closed_saved_bytes_total"] = 0
    payload["typed_response_ledger"]["available_receiver_closed_rate_credit_bytes"] = 0
    score_report = score_repair_campaign(payload=payload, repo_root=tmp_path)
    score_report_path = _write_json(tmp_path / "score_report.json", score_report)

    signal_report = build_repair_campaign_blocked_learning_signal_report(
        score_report_path=score_report_path,
        score_report=score_report,
        repo_root=tmp_path,
    )
    signal = signal_report["learning_signal_rows"][0]
    update = signal["local_planning_update"]

    assert update["recommended_acquisition_policy"] == (
        "increase_receiver_closed_rate_credit_or_rebudget_earlier_entropy_stage"
    )
    assert update["planner_feature_vector"]["selection_blocker_class"] == ("receiver_credit_exhausted")
    assert update["planner_feature_vector"]["receiver_credit_exhausted"] is True

    signal_report_path = _write_json(tmp_path / "blocked_signals.json", signal_report)
    posterior_path = tmp_path / "posterior.jsonl"
    append_repair_campaign_blocked_learning_signal_report(
        blocked_learning_signal_report_path=signal_report_path,
        blocked_learning_signal_report=signal_report,
        posterior_path=posterior_path,
        lock_path=tmp_path / ".posterior.lock",
        repo_root=tmp_path,
    )
    posterior_rows = load_repair_campaign_stackability_posterior_rows(posterior_path)
    summary = build_repair_campaign_posterior_prior_summary(
        posterior_path=posterior_path,
    )
    route = summary["acquisition_followup_routes"][0]

    assert (
        posterior_rows[0]["acquisition_policy_delta"]["posterior_budget_routing_hint"]
        == "rebudget_receiver_closed_credit_before_exact_axis_spend"
    )
    assert route["activation_action"] == ("rebudget_receiver_credit_to_earliest_entropy_stage")
    assert route["queue_artifact_key"] == "repair_budget_waterfill_queue"


def test_family_materializer_rejects_stale_optimizer_solver_contract(
    tmp_path: Path,
) -> None:
    score_report = score_repair_campaign(payload=_repair_payload(tmp_path), repo_root=tmp_path)
    score_report["optimizer_decision"]["solver"] = "greedy_campaign_score_waterfill_v1"
    plan = _plan_from_score_report(score_report)

    with pytest.raises(RepairCampaignChainContractError, match="requires solver"):
        build_repair_campaign_family_materializer_manifest(
            repo_root=tmp_path,
            materialization_plan=plan,
            score_report=score_report,
            materialization_plan_path=tmp_path / "plan.json",
            score_report_path=tmp_path / "score_report.json",
            typed_response_id="segnet_region_ready",
            candidate_id="segnet_class_region_waterfill",
        )
