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
)
from tac.substrates.z8_hierarchical_predictive_coding.mlx_renderer import (
    Z8HierarchicalConfig,
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
    assert "dreamer_state_blob" in proof["custody_only_sections"]
    assert proof["mamba_dreamer_wyner_ziv_pixel_consumption_proven"] is False
    bridge_report_path = tmp_path / "z8_hpc1_runtime_payload_bridge_report.json"
    assert bridge_report_path.is_file()
    bridge_report = json.loads(bridge_report_path.read_text(encoding="utf-8"))
    assert bridge_report["wyner_ziv_top_state_decode_ready"] is True
    assert bridge_report["wyner_ziv_top_state_count"] == 1
    assert bridge_report["state_to_pixel_projection_ready"] is False
    assert bridge_report["pixel_consumption_proven"] is False
    submission = tmp_path / "submission"
    assert (submission / "0.bin").is_file()
    assert (submission / "inflate.sh").is_file()
    inflate_source = (submission / "inflate.py").read_text(encoding="utf-8")
    assert "import inflate_one_video" in inflate_source

    runtime_root = submission / "src" / "tac"
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


def test_z8_archive_bound_package_stays_false_authority_when_receiver_blocked(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    def fake_emit_runtime_package(**kwargs: Any) -> dict[str, Any]:
        manifest_extra = kwargs["runtime_adapter_manifest_extra"]
        assert manifest_extra["full_stack_pixel_consumption_claim"] is False
        assert manifest_extra["pixel_consumed_archive_sections"] == list(
            Z8_HPC_PIXEL_CONSUMED_ARCHIVE_SECTIONS
        )
        assert manifest_extra["stack_custody_not_yet_pixel_consumed_sections"] == list(
            Z8_HPC_STACK_CUSTODY_NOT_YET_PIXEL_CONSUMED_SECTIONS
        )
        mutation_proof = manifest_extra["byte_mutation_consumption_proof"]
        assert mutation_proof["distinguishing_feature_consumed"] is True
        assert mutation_proof["pixel_consumed_sections"] == ["wavelet_blob"]
        assert "dreamer_state_blob" in mutation_proof["custody_only_sections"]
        assert (
            mutation_proof["mamba_dreamer_wyner_ziv_pixel_consumption_proven"]
            is False
        )
        status = manifest_extra["mamba_dreamer_wyner_ziv_runtime_consumption_status"]
        expected_status = (
            "archive_bound_custody_only_distinguishing_feature_mutation"
            "_proved_not_pixel_consumed"
        )
        assert status == expected_status
        bridge_report = manifest_extra["runtime_payload_bridge_report"]
        assert bridge_report["wyner_ziv_top_state_decode_ready"] is True
        assert bridge_report["wyner_ziv_top_state_count"] == 1
        assert bridge_report["state_to_pixel_projection_ready"] is False
        assert bridge_report["pixel_consumption_proven"] is False
        assert bridge_report["next_required_task"] == (
            "fit_and_archive_state_to_top_ll_projection"
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
    assert "wavelet_pixel_consumed" in row["target_kind"]
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
