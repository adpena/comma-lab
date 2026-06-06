# SPDX-License-Identifier: MIT
"""Tests for the Z8 archive-bound runtime bridge."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

import tac.substrates.z8_hierarchical_predictive_coding.archive_candidate as z8_bridge
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    build_archive_bound_candidate_runtime_package,
)
from tac.substrates.z8_hierarchical_predictive_coding.archive import parse_archive
from tac.substrates.z8_hierarchical_predictive_coding.archive_candidate import (
    Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID,
    Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
    Z8_HPC_ARCHIVE_CANDIDATE_FAMILY,
    Z8_HPC_ARCHIVE_TRANSFORM_KIND,
    Z8_HPC_PIXEL_CONSUMED_ARCHIVE_SECTIONS,
    Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS,
    export_z8hpc1_archive_bytes,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    build_z8hpc1_archive_bytes_from_canonical_quadruple,
    parse_pair_blobs_from_wavelet_blob,
    reconstruct_pair_rgb_from_pyramid,
)
from tac.substrates.z8_hierarchical_predictive_coding.entropy_delta_schedule import (
    Z8_HPC1_DETAIL_ENTROPY_DELTA_RECEIVER_PROOF_SCHEMA,
    verify_z8_hpc1_detail_entropy_delta_receiver_contract,
)
from tac.substrates.z8_hierarchical_predictive_coding.inflate import CONTEST_RAW_BYTES
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
)
from tac.substrates.z8_hierarchical_predictive_coding.runtime_custody import (
    Z8_HPC_RUNTIME_CUSTODY_CONTRACT_SCHEMA,
    Z8_HPC_TRAINED_MLX_EXPORT_BLOCKER,
)
from tac.substrates.z8_hierarchical_predictive_coding.runtime_payload_bridge import (
    build_stack_context_payload_mutation_receiver_proofs,
    build_wyner_ziv_payload_mutation_receiver_proof,
    decode_wyner_ziv_top_states_from_archive,
    mutate_valid_stack_context_payload_in_archive,
    mutate_valid_wyner_ziv_payload_in_archive,
    project_decoded_top_states_into_pair_pyramids,
)


def _cfg() -> Z8HierarchicalConfig:
    return Z8HierarchicalConfig(
        num_levels=3,
        num_groups_per_level=(4, 3, 2),
        num_categories_per_level=(16, 8, 4),
        base_channels=8,
        decoder_latent_dim=12,
        num_pairs=1,
        deterministic_state_dim=16,
        gumbel_temperature=1.0,
        use_straight_through=True,
        eval_size=(16, 16),
    )


def _archive_bytes() -> bytes:
    rng = np.random.RandomState(7)
    f0 = rng.uniform(0, 1, size=(1, 16, 16, 3)).astype(np.float32)
    f1 = rng.uniform(0, 1, size=(1, 16, 16, 3)).astype(np.float32)
    binding = build_canonical_quadruple_binding_from_z8_config(_cfg())
    return build_z8hpc1_archive_bytes_from_canonical_quadruple(binding, f0, f1)


def test_z8_archive_meta_carries_fail_closed_runtime_custody_contract() -> None:
    archive = parse_archive(_archive_bytes())
    contract = archive.meta["runtime_custody_contract"]

    assert contract["schema"] == Z8_HPC_RUNTIME_CUSTODY_CONTRACT_SCHEMA
    assert contract["source"] == "z8_m9_canonical_quadruple_archive_meta"
    assert contract["section_name_style"] == "archive_parser"
    assert contract["pixel_consumed_archive_sections"] == [
        "decoder_blob",
        "indices_blob",
        "wavelet_blob",
        "wyner_ziv_blob",
        "dreamer_state_blob",
    ]
    assert contract["stack_semantic_pixel_consumed_sections"] == [
        "decoder_blob",
        "indices_blob",
        "dreamer_state_blob",
    ]
    assert contract["stack_custody_not_yet_pixel_consumed_sections"] == []
    assert contract["full_stack_pixel_consumption_claim"] is False
    assert contract["trained_mlx_renderer_archive_export_ready"] is False
    assert Z8_HPC_TRAINED_MLX_EXPORT_BLOCKER in contract["blockers"]
    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_z8_runtime_payload_projection_changes_frame1_pixels() -> None:
    archive_bytes = _archive_bytes()
    arc = parse_archive(archive_bytes)
    binding = build_canonical_quadruple_binding_from_z8_config(_cfg())
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    decoded = decode_wyner_ziv_top_states_from_archive(archive_bytes)

    projected, stats = project_decoded_top_states_into_pair_pyramids(
        pair_pyramids,
        decoded,
    )

    assert stats["projection_target"] == "frame_1_top_ll"
    assert stats["projected_pair_changed_count"] == 1
    base_f0, base_f1 = reconstruct_pair_rgb_from_pyramid(binding, pair_pyramids[0])
    projected_f0, projected_f1 = reconstruct_pair_rgb_from_pyramid(
        binding, projected[0]
    )
    np.testing.assert_allclose(projected_f0, base_f0)
    assert float(np.max(np.abs(projected_f1 - base_f1))) > 0.0


def test_z8_valid_wyner_ziv_payload_mutation_changes_frame1_pixels() -> None:
    archive_bytes = _archive_bytes()
    mutated_archive, mutation = mutate_valid_wyner_ziv_payload_in_archive(
        archive_bytes
    )
    assert mutated_archive != archive_bytes
    assert mutation["mutated_archive_sha256"] != mutation["original_archive_sha256"]

    proof = build_wyner_ziv_payload_mutation_receiver_proof(archive_bytes)

    assert proof["archive_member_byte_closed"] is True
    assert proof["valid_semantic_wyner_ziv_payload_mutation"] is True
    assert proof["wyner_ziv_top_state_pixel_consumption_proven"] is True
    assert proof["frame_0_max_abs_delta"] == 0.0
    assert proof["frame_1_max_abs_delta"] > 0.0
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_z8_valid_stack_context_payload_mutations_change_frame1_pixels() -> None:
    archive_bytes = _archive_bytes()
    for section in ("decoder_blob", "indices_blob", "dreamer_state_blob"):
        mutated_archive, mutation = mutate_valid_stack_context_payload_in_archive(
            archive_bytes,
            section,
        )
        assert mutated_archive != archive_bytes
        assert mutation["mutated_section"] == section
        assert mutation["mutated_archive_sha256"] != mutation["original_archive_sha256"]

    proof = build_stack_context_payload_mutation_receiver_proofs(archive_bytes)

    assert proof["archive_member_byte_closed"] is True
    assert proof["valid_semantic_stack_context_payload_mutations"] is True
    assert proof["stack_context_sections_pixel_consumed"] == [
        "decoder_blob",
        "indices_blob",
        "dreamer_state_blob",
    ]
    for section in proof["stack_context_sections_pixel_consumed"]:
        section_proof = proof["per_section"][section]
        assert section_proof["section_pixel_consumption_proven"] is True
        assert section_proof["frame_0_max_abs_delta"] == 0.0
        assert section_proof["frame_1_max_abs_delta"] > 0.0
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_z8_archive_export_emits_runtime_tree_without_mlx_import(tmp_path: Path) -> None:
    archive_zip, archive_sha, archive_bytes = export_z8hpc1_archive_bytes(
        _archive_bytes(),
        tmp_path,
        emit_archive_bound_candidate_package=False,
    )

    assert archive_zip.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_zip.stat().st_size
    byte_mutation_proof_path = tmp_path / "z8_hpc1_byte_mutation_proof.json"
    assert byte_mutation_proof_path.is_file()
    proof = json.loads(byte_mutation_proof_path.read_text(encoding="utf-8"))
    assert proof["distinguishing_feature_consumed"] is True
    assert "wavelet_blob" in proof["pixel_consumed_sections"]
    assert "wyner_ziv_blob" in proof["pixel_consumed_sections"]
    assert proof["custody_only_sections"] == []
    assert proof["mamba_dreamer_wyner_ziv_pixel_consumption_proven"] is True
    semantic_wz = proof["wyner_ziv_semantic_payload_mutation_receiver_proof"]
    assert semantic_wz["wyner_ziv_top_state_pixel_consumption_proven"] is True
    assert proof["wyner_ziv_payload_pixel_consumption_proven"] is True
    assert proof["stack_context_pixel_consumption_proven"] is True
    section_proofs = proof["section_pixel_consumption_proofs"]
    assert section_proofs["wavelet_blob"]["proof_kind"] == "raw_byte_mutation"
    assert section_proofs["wyner_ziv_blob"]["proof_kind"] == (
        "valid_semantic_payload_mutation"
    )
    stack = proof["predictive_stack_pixel_consumption"]
    assert stack["all_required_sections_pixel_consumed"] is True
    assert stack["full_stack_pixel_consumption_claim"] is False
    assert stack["status"] == (
        "section_pixel_consumption_proven_trained_export_pending"
    )
    for section in ("decoder_blob", "indices_blob", "dreamer_state_blob"):
        assert section_proofs[section]["proof_kind"] == (
            "valid_stack_context_payload_mutation"
        )
        assert section_proofs[section]["pixel_consumption_proven"] is True
        assert section in proof["pixel_consumed_sections"]
        assert section not in proof["custody_only_sections"]
    bridge_report_path = tmp_path / "z8_hpc1_runtime_payload_bridge_report.json"
    assert bridge_report_path.is_file()
    bridge_report = json.loads(bridge_report_path.read_text(encoding="utf-8"))
    assert bridge_report["wyner_ziv_top_state_decode_ready"] is True
    assert bridge_report["wyner_ziv_top_state_count"] == 1
    assert bridge_report["state_to_pixel_projection_ready"] is True
    assert bridge_report["state_to_pixel_projection"]["projected_pair_count"] == 1
    assert bridge_report["state_to_pixel_projection"]["projected_pair_changed_count"] == 1
    assert bridge_report["state_to_pixel_projection"]["stack_context_vector_length"] > 0
    assert bridge_report["stack_context_sections_pixel_consumed"] == [
        "decoder_blob",
        "indices_blob",
        "dreamer_state_blob",
    ]
    assert bridge_report["pixel_consumption_proven"] is True
    submission = tmp_path / "submission"
    assert (submission / "0.bin").is_file()
    assert (submission / "inflate.sh").is_file()
    inflate_source = (submission / "inflate.py").read_text(encoding="utf-8")
    assert "import inflate_one_video" in inflate_source
    assert "with_suffix('.raw')" in inflate_source
    assert "output_dir / rel" in inflate_source

    runtime_root = submission / "src" / "tac"
    assert (
        runtime_root
        / "substrates"
        / "z8_hierarchical_predictive_coding"
        / "runtime_custody.py"
    ).is_file()
    assert (
        runtime_root
        / "substrates"
        / "z8_hierarchical_predictive_coding"
        / "runtime_payload_bridge.py"
    ).is_file()
    assert (
        runtime_root
        / "substrates"
        / "z8_hierarchical_predictive_coding"
        / "canonical_quadruple_binding.py"
    ).is_file()
    assert (runtime_root / "optimization" / "mamba2_predictor.py").is_file()
    assert (
        runtime_root / "symposium_impls" / "daubechies_wavelet_codec.py"
    ).is_file()
    z8_inflate = (
        runtime_root / "substrates" / "z8_hierarchical_predictive_coding" / "inflate.py"
    ).read_text(encoding="utf-8")
    assert "mlx_renderer" not in z8_inflate

    with zipfile.ZipFile(archive_zip) as zf:
        names = set(zf.namelist())
    assert "0.bin" in names
    assert "inflate.sh" in names
    assert "src/tac/optimization/mamba2_predictor.py" in names
    assert "src/tac/symposium_impls/daubechies_wavelet_codec.py" in names


def test_z8_archive_export_can_measure_rate_without_runtime_bridge_report(
    tmp_path: Path,
) -> None:
    archive_zip, archive_sha, archive_bytes = export_z8hpc1_archive_bytes(
        _archive_bytes(),
        tmp_path,
        emit_archive_bound_candidate_package=False,
        emit_byte_mutation_proof=False,
        emit_runtime_payload_bridge_report=False,
    )

    assert archive_zip.is_file()
    assert len(archive_sha) == 64
    assert archive_bytes == archive_zip.stat().st_size
    assert not (tmp_path / "z8_hpc1_runtime_payload_bridge_report.json").exists()
    assert not (tmp_path / "z8_hpc1_byte_mutation_proof.json").exists()


def test_z8_entropy_delta_receiver_verifier_is_fail_closed(tmp_path: Path) -> None:
    proof = {
        "schema": Z8_HPC1_DETAIL_ENTROPY_DELTA_RECEIVER_PROOF_SCHEMA,
        "candidate_label": "z8_hpc1",
        "archive_path": "candidate/archive.zip",
        "archive_sha256": "a" * 64,
        "archive_bytes": 1234,
        "submission_dir": "candidate/submission",
        "runtime_tree_sha256": "b" * 64,
        "inflate_argv": [
            "candidate/submission/inflate.sh",
            "candidate/submission",
            "receiver_out",
            "file_list.txt",
        ],
        "file_list_path": "receiver_proof/file_list.txt",
        "receiver_output_path": "receiver_proof/runtime_out/0.raw",
        "receiver_output_kind": "file",
        "receiver_output_sha256": "c" * 64,
        "receiver_output_bytes": CONTEST_RAW_BYTES,
        "expected_receiver_output_bytes": CONTEST_RAW_BYTES,
        "returncode": 0,
        "timed_out": False,
        "runtime_consumption_proof_ready": True,
        "runtime_consumption_proof_passed": True,
        "receiver_contract_satisfied": True,
        "blockers": [],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    proof_path = tmp_path / "z8_hpc1_receiver_proof.json"
    proof_path.write_text(json.dumps(proof, sort_keys=True), encoding="utf-8")

    verified = verify_z8_hpc1_detail_entropy_delta_receiver_contract(
        runtime_consumption_proof=proof_path,
        required_candidate_archive_sha256="a" * 64,
        required_candidate_archive_bytes=1234,
    )

    assert verified["receiver_contract_satisfied"] is True
    assert verified["runtime_consumption_proof_ready"] is True
    assert verified["blockers"] == []

    stale = dict(proof)
    stale["schema"] = "legacy_z8_receiver_proof.v0"
    stale["receiver_output_bytes"] = 1
    stale["score_claim"] = True
    rejected = verify_z8_hpc1_detail_entropy_delta_receiver_contract(
        runtime_consumption_proof=stale,
        required_candidate_archive_sha256="a" * 64,
        required_candidate_archive_bytes=1234,
    )

    assert rejected["receiver_contract_satisfied"] is False
    assert "runtime_consumption_proof_schema_mismatch" in rejected["blockers"]
    assert "generated_inflate_receiver_output_bytes_mismatch" in rejected["blockers"]
    assert any(
        "forbidden truthy authority fields" in blocker
        for blocker in rejected["blockers"]
    )


def test_z8_archive_bound_package_stays_false_authority_when_receiver_blocked(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_emit_runtime_package(**kwargs: Any) -> dict[str, Any]:
        manifest_extra = kwargs["runtime_adapter_manifest_extra"]
        custody = manifest_extra["runtime_custody_contract"]
        assert custody["schema"] == Z8_HPC_RUNTIME_CUSTODY_CONTRACT_SCHEMA
        assert custody["source"] == "z8_hpc1_runtime_adapter_manifest"
        assert custody["section_name_style"] == "candidate_manifest"
        assert Z8_HPC_TRAINED_MLX_EXPORT_BLOCKER in custody["blockers"]
        assert manifest_extra["full_stack_pixel_consumption_claim"] is False
        assert manifest_extra["pixel_consumed_archive_sections"] == list(
            Z8_HPC_PIXEL_CONSUMED_ARCHIVE_SECTIONS
        )
        assert "decoder_blob" in manifest_extra["pixel_consumed_archive_sections"]
        assert "wyner_ziv_blob" in manifest_extra["pixel_consumed_archive_sections"]
        assert manifest_extra["stack_custody_not_yet_pixel_consumed_sections"] == list(
            Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS
        )
        assert manifest_extra["stack_custody_not_yet_pixel_consumed_sections"] == []
        assert "wyner_ziv_blob" not in (
            manifest_extra["stack_custody_not_yet_pixel_consumed_sections"]
        )
        mutation_proof = manifest_extra["byte_mutation_consumption_proof"]
        assert mutation_proof["distinguishing_feature_consumed"] is True
        for section in (
            "wavelet_blob",
            "wyner_ziv_blob",
            "decoder_blob",
            "indices_blob",
            "dreamer_state_blob",
        ):
            assert section in mutation_proof["pixel_consumed_sections"]
        assert mutation_proof["custody_only_sections"] == []
        assert mutation_proof["wyner_ziv_payload_pixel_consumption_proven"] is True
        assert mutation_proof["stack_context_pixel_consumption_proven"] is True
        section_proofs = mutation_proof["section_pixel_consumption_proofs"]
        assert section_proofs["wyner_ziv_blob"]["pixel_consumption_proven"] is True
        assert section_proofs["decoder_blob"]["pixel_consumption_proven"] is True
        stack = mutation_proof["predictive_stack_pixel_consumption"]
        assert stack["all_required_sections_pixel_consumed"] is True
        assert stack["status"] == (
            "section_pixel_consumption_proven_trained_export_pending"
        )
        assert (
            mutation_proof["mamba_dreamer_wyner_ziv_pixel_consumption_proven"]
            is True
        )
        assert manifest_extra["predictive_stack_runtime_consumption_status"] == (
            "section_pixel_consumption_proven_trained_export_pending"
        )
        status = manifest_extra["mamba_dreamer_wyner_ziv_runtime_consumption_status"]
        assert status == "pixel_consumed_proven"
        bridge_report = manifest_extra["runtime_payload_bridge_report"]
        assert bridge_report["wyner_ziv_top_state_decode_ready"] is True
        assert bridge_report["wyner_ziv_top_state_count"] == 1
        assert bridge_report["state_to_pixel_projection_ready"] is True
        assert bridge_report["pixel_consumption_proven"] is True
        assert bridge_report["next_required_task"] == (
            "serialize_trained_mlx_renderer_state_into_z8hpc1_archive"
        )
        assert "mamba_mallat_dreamer_wyner_ziv_stack" not in (
            manifest_extra["predictive_coding_family"]
        )
        proof = {
            "proof_path": "receiver_proof/fake_z8_receiver_proof.json",
            "inflate_argv": [
                "submission/inflate.sh",
                "submission",
                "out",
                "file_list",
            ],
            "runtime_consumption_proof_ready": False,
            "receiver_contract_satisfied": False,
            "blockers": ["z8_hpc1_generated_inflate_sh_not_executed_in_unit_test"],
        }
        return build_archive_bound_candidate_runtime_package(
            adapter_id=kwargs["adapter_id"],
            candidate_family=kwargs["candidate_family"],
            candidate_id_prefix=kwargs["candidate_id_prefix"],
            transform_kind=kwargs["transform_kind"],
            archive_zip_path=kwargs["archive_zip_path"],
            archive_sha256=kwargs["archive_sha256"],
            archive_bytes=kwargs["archive_bytes"],
            submission_dir=kwargs["submission_dir"],
            output_dir=kwargs["output_dir"],
            repo_root=kwargs["repo_root"],
            receiver_proof=proof,
            receiver_contract_kind=kwargs["receiver_contract_kind"],
            runtime_adapter_manifest_extra=kwargs["runtime_adapter_manifest_extra"],
            candidate_row_schema=kwargs["candidate_row_schema"],
            wrapper_schema=kwargs["wrapper_schema"],
            input_artifacts=kwargs["input_artifacts"],
            mlx_triage_argv=kwargs["mlx_triage_argv"],
        )

    monkeypatch.setattr(
        z8_bridge,
        "emit_archive_bound_candidate_runtime_package",
        fake_emit_runtime_package,
    )
    export_z8hpc1_archive_bytes(
        _archive_bytes(),
        tmp_path,
        emit_archive_bound_candidate_package=True,
    )
    package_path = tmp_path / "archive_bound_candidate_adapter_package.json"
    assert package_path.is_file()

    import json

    package = json.loads(package_path.read_text(encoding="utf-8"))
    assert package["schema"] == Z8_HPC_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA
    assert package["score_claim"] is False
    wrapped = package["archive_bound_candidate_adapter_package"]
    row = wrapped["candidate_rows"][0]
    assert wrapped["adapter_id"] == Z8_HPC_ARCHIVE_BOUND_ADAPTER_ID
    assert row["candidate_family"] == Z8_HPC_ARCHIVE_CANDIDATE_FAMILY
    assert row["target_kind"] == Z8_HPC_ARCHIVE_TRANSFORM_KIND
    assert "partial_predictive_stack" in row["target_kind"]
    assert "mamba_mallat_dreamer_wyner_ziv" not in row["target_kind"]
    assert "wyner_ziv_top_ll_pixel_driver" in row["target_kind"]
    assert row["byte_closed_candidate_materialized"] is True
    assert row["candidate_archive_sha256"]
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert "archive_bound_candidate_contract" in row
    assert any(
        str(path).endswith("z8_hpc1_byte_mutation_proof.json")
        for path in row["input_artifacts"]
    )
    assert any(
        str(path).endswith("z8_hpc1_runtime_payload_bridge_report.json")
        for path in row["input_artifacts"]
    )
