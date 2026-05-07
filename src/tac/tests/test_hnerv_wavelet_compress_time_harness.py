from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli
import pytest

from tac.hnerv_lowlevel_packer import read_strict_single_member_zip, write_stored_single_member_zip
from tac.hnerv_wavelet_apply_transform import build_wavelet_apply_transform_candidate
from tac.hnerv_wavelet_compress_time_harness import (
    ATOM_PLAN_FILENAME,
    ATOM_PLAN_SCHEMA,
    RUNTIME_DECODE_REVIEW_FILENAME,
    RUNTIME_DECODE_REVIEW_SCHEMA,
    SELECTED_ATOMS_FILENAME,
    SELECTED_ATOMS_SCHEMA,
    WR01_FIXED_ATOM_WIRE_BYTES,
    HnervWaveletCompressTimeHarnessError,
    build_wavelet_compress_time_harness,
)
from tac.hnerv_wavelet_sidechannel import (
    build_wavelet_sidechannel_archive_bytes,
    encode_wavelet_atom_sidechannel,
)

REPO = Path(__file__).resolve().parents[3]


def test_wavelet_compress_time_harness_manifest_is_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    out_dir = tmp_path / "harness"

    manifest = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=out_dir,
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=7,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
    )
    repeat = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=out_dir,
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=7,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
    )

    manifest_path = out_dir / "hnerv_wavelet_compress_time_harness.json"
    atom_plan_path = out_dir / ATOM_PLAN_FILENAME
    selected_atoms_path = out_dir / SELECTED_ATOMS_FILENAME
    runtime_decode_review_path = out_dir / RUNTIME_DECODE_REVIEW_FILENAME
    assert manifest == repeat
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert json.loads(atom_plan_path.read_text(encoding="utf-8")) == manifest["atom_plan_manifest"]
    assert json.loads(selected_atoms_path.read_text(encoding="utf-8")) == manifest[
        "selected_atoms_manifest"
    ]
    assert json.loads(runtime_decode_review_path.read_text(encoding="utf-8")) == manifest[
        "output_manifest"
    ]["runtime_decode_review"]
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_compress_time_training"] is False
    assert manifest["ready_for_atom_plan_review"] is True
    assert manifest["ready_for_selected_atom_review"] is True
    assert manifest["ready_for_runtime_apply_review"] is False
    assert manifest["ready_for_decode_validation_review"] is False
    assert manifest["ready_for_wavelet_sidechannel_candidate"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "missing_exact_decode_validation_manifest" in manifest["blockers"]
    assert "missing_runtime_apply_manifest" in manifest["blockers"]
    assert "compress_time_atom_training_not_implemented" in manifest["blockers"]
    assert "compress_time_harness_scaffold_only" in manifest["dispatch_blockers"]

    input_manifest = manifest["input_manifest"]
    output_manifest = manifest["output_manifest"]
    assert input_manifest["schema"] == "hnerv_wavelet_compress_time_input.v1"
    assert output_manifest["schema"] == "hnerv_wavelet_compress_time_output.v1"
    assert output_manifest["atom_plan_schema"] == ATOM_PLAN_SCHEMA
    assert output_manifest["atom_plan_manifest_path"] == str(atom_plan_path)
    assert output_manifest["atom_plan_manifest_sha256"] == manifest["atom_plan_manifest"][
        "manifest_sha256_excluding_self"
    ]
    assert output_manifest["selected_atoms_schema"] == SELECTED_ATOMS_SCHEMA
    assert output_manifest["selected_atoms_manifest_path"] == str(selected_atoms_path)
    assert output_manifest["selected_atoms_manifest_sha256"] == manifest["selected_atoms_manifest"][
        "manifest_sha256_excluding_self"
    ]
    assert output_manifest["runtime_decode_review_schema"] == RUNTIME_DECODE_REVIEW_SCHEMA
    assert output_manifest["runtime_decode_review_manifest_path"] == str(runtime_decode_review_path)
    assert output_manifest["runtime_decode_review_manifest_sha256"] == output_manifest[
        "runtime_decode_review"
    ]["manifest_sha256_excluding_self"]
    assert output_manifest["runtime_decode_review"]["status"] == "blocked"
    assert output_manifest["runtime_decode_review"]["ready_for_runtime_apply_review"] is False
    assert output_manifest["runtime_decode_review"]["ready_for_decode_validation_review"] is False
    assert "missing_runtime_apply_manifest" in output_manifest["runtime_decode_review"]["blockers"]
    assert input_manifest["source"]["source_archive_custody_mode"] == ("operator_expected_archive_identity_verified")
    assert input_manifest["config"]["seed"] == 123
    assert input_manifest["config"]["atom_budget"] == 7
    assert input_manifest["config"]["rng_state_mutated"] is False
    assert output_manifest["trained_atoms_manifest_path"] is None
    assert output_manifest["apply_readiness"]["ready"] is False
    assert output_manifest["apply_readiness"]["selected_atoms_manifest_path"] == str(selected_atoms_path)
    assert "no_candidate_archive_emitted" in output_manifest["apply_readiness"]["blockers"]
    assert output_manifest["wavelet_sidechannel_archive_path"] is None
    assert output_manifest["applied_candidate_archive_path"] is None
    assert output_manifest["decode_validation"]["fail_closed"] is True

    section = manifest["source_sections"][0]
    assert section["section_name"] == "latents_and_sidecar_brotli"
    assert section["decode_probe_status"] == "local_brotli_decode_only_not_exact_validation"
    assert section["raw_bytes"] > section["section_bytes"]
    assert section["score_claim"] is False
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["manifest_sha256_excluding_self"]) == 64

    atom_plan = manifest["atom_plan_manifest"]
    assert atom_plan["schema"] == ATOM_PLAN_SCHEMA
    assert atom_plan["source_archive_sha256"] == source.archive_sha256
    assert atom_plan["source_archive_bytes"] == source.archive_bytes
    assert atom_plan["budget"]["atom_count_budget_per_section"] == 7
    assert atom_plan["budget"]["atom_wire_bytes_budget_per_atom"] == WR01_FIXED_ATOM_WIRE_BYTES
    assert atom_plan["budget"]["atom_wire_bytes_budget_per_section"] == 7 * WR01_FIXED_ATOM_WIRE_BYTES
    assert atom_plan["ready_for_train_select_apply"] is False
    assert atom_plan["ready_for_exact_eval_dispatch"] is False
    assert "no_candidate_archive_emitted" in atom_plan["dispatch_blockers"]
    assert "requires_wr01_runtime_apply_path" in atom_plan["dispatch_blockers"]
    plan_section = atom_plan["sections"][0]
    assert plan_section["section_name"] == "latents_and_sidecar_brotli"
    assert plan_section["source_section_payload_span"]["start"] > 0
    assert plan_section["source_section_payload_span"]["end"] == source.member_bytes
    assert plan_section["emitted_atom_count"] == 7
    assert plan_section["candidate_atoms_before_budget"] >= plan_section["emitted_atom_count"]
    assert len(plan_section["candidate_atom_ids"]) == 7
    assert atom_plan["atom_ids"] == plan_section["candidate_atom_ids"]
    assert len(set(atom_plan["atom_ids"])) == 7
    first_atom = plan_section["candidate_atoms"][0]
    assert first_atom["atom_id"].startswith("wr01-latents-and-sidecar-brotli-")
    assert first_atom["budget_rank"] == 1
    assert first_atom["atom_wire_bytes_budget"] == WR01_FIXED_ATOM_WIRE_BYTES
    assert first_atom["selected_for_apply"] is False
    assert first_atom["score_claim"] is False
    assert len(atom_plan["plan_sha256"]) == 64

    selected = manifest["selected_atoms_manifest"]
    assert selected["schema"] == SELECTED_ATOMS_SCHEMA
    assert selected["source_archive_sha256"] == source.archive_sha256
    assert selected["source_archive_bytes"] == source.archive_bytes
    assert selected["atom_plan_manifest_path"] == str(atom_plan_path)
    assert selected["atom_plan_manifest_sha256"] == atom_plan["manifest_sha256_excluding_self"]
    assert selected["atom_plan_sha256"] == atom_plan["plan_sha256"]
    assert selected["selection_mode"] == "atom_plan_budget_rank_order"
    assert selected["selection_input"]["budget_is_dispatch_clearance"] is False
    assert selected["total_selected_atom_count"] == 7
    assert selected["selected_atom_ids"] == atom_plan["atom_ids"]
    assert selected["estimated_total_atom_wire_bytes"] == 7 * WR01_FIXED_ATOM_WIRE_BYTES
    assert selected["ready_for_selected_atom_review"] is True
    assert selected["ready_for_train_select_apply"] is False
    assert selected["ready_for_runtime_apply"] is False
    assert selected["ready_for_archive_preflight"] is False
    assert selected["ready_for_exact_eval_dispatch"] is False
    assert selected["candidate_archive_path"] is None
    assert selected["candidate_archive_sha256"] is None
    assert selected["candidate_archive_bytes"] is None
    assert "selected_atoms_manifest_is_planning_only" in selected["blockers"]
    assert "requires_exact_cuda_auth_eval" in selected["dispatch_blockers"]
    assert len(selected["selection_sha256"]) == 64
    assert len(selected["manifest_sha256_excluding_self"]) == 64
    selected_section = selected["sections"][0]
    assert selected_section["section_name"] == "latents_and_sidecar_brotli"
    assert selected_section["selected_atom_count"] == 7
    assert selected_section["selected_atom_ids"] == atom_plan["atom_ids"]
    assert selected_section["estimated_wire_bytes"] == 7 * WR01_FIXED_ATOM_WIRE_BYTES
    assert selected_section["atoms"] == selected_section["selected_atoms"]
    selected_first = selected_section["selected_atoms"][0]
    assert selected_first["atom_id"] == first_atom["atom_id"]
    assert selected_first["selection_rank"] == 1
    assert selected_first["estimated_wire_bytes"] == WR01_FIXED_ATOM_WIRE_BYTES
    assert selected_first["byte_delta"] == WR01_FIXED_ATOM_WIRE_BYTES
    assert selected_first["selected_for_apply_readiness"] is True
    assert selected_first["selected_for_runtime_apply"] is False
    assert selected_first["score_claim"] is False
    assert "no_candidate_archive_emitted" in selected_first["dispatch_blockers"]


def test_wavelet_compress_time_harness_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)

    with pytest.raises(HnervWaveletCompressTimeHarnessError, match="does not match"):
        build_wavelet_compress_time_harness(
            source_archive=archive,
            source_label="PR106x_WR01",
            output_dir=tmp_path / "harness",
            expected_source_archive_sha256="0" * 64,
            expected_source_archive_bytes=source.archive_bytes,
        )


def test_wavelet_compress_time_harness_exact_validation_remains_non_dispatchable(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    exact_decode_validation = {
        "source_archive_sha256": source.archive_sha256,
        "exact_decode_validation": True,
        "validated_sections": ["latents_and_sidecar_brotli"],
        "score_claim": False,
    }

    manifest = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=tmp_path / "harness",
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=7,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
        exact_decode_validation=exact_decode_validation,
    )

    decode_validation = manifest["output_manifest"]["decode_validation"]
    assert decode_validation["exact_validation_available"] is True
    assert decode_validation["fail_closed"] is False
    assert decode_validation["score_claim"] is False
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_wavelet_sidechannel_candidate"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["output_manifest"]["wavelet_sidechannel_archive_path"] is None
    assert manifest["output_manifest"]["applied_candidate_archive_path"] is None
    assert manifest["selected_atoms_manifest"]["ready_for_runtime_apply"] is False
    assert manifest["selected_atoms_manifest"]["candidate_archive_path"] is None
    assert "compress_time_harness_scaffold_only" in manifest["dispatch_blockers"]
    assert "requires_wr01_runtime_apply_path" in manifest["dispatch_blockers"]
    assert "requires_exact_cuda_auth_eval" in manifest["dispatch_blockers"]
    assert "compress_time_atom_training_not_implemented" in manifest["blockers"]
    assert manifest["atom_plan_manifest"]["ready_for_exact_eval_dispatch"] is False
    assert "no_candidate_archive_emitted" in manifest["atom_plan_manifest"]["dispatch_blockers"]


def test_wavelet_compress_time_harness_reviews_runtime_apply_decode_manifest(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    planning = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=tmp_path / "planning",
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=5,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
    )
    runtime_apply_manifest = _runtime_apply_manifest_for_selected_atoms(
        tmp_path=tmp_path,
        source_archive=archive,
        selected_atoms_manifest=planning["selected_atoms_manifest"],
    )
    runtime_decode_validation = runtime_apply_manifest["runtime_decode_validation"]

    manifest = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=tmp_path / "reviewed",
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=5,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
        runtime_apply_manifest=runtime_apply_manifest,
        exact_decode_validation=runtime_decode_validation,
    )

    review_path = tmp_path / "reviewed" / RUNTIME_DECODE_REVIEW_FILENAME
    review = manifest["output_manifest"]["runtime_decode_review"]
    decode_validation = manifest["output_manifest"]["decode_validation"]
    assert json.loads(review_path.read_text(encoding="utf-8")) == review
    assert review["schema"] == RUNTIME_DECODE_REVIEW_SCHEMA
    assert review["status"] == "ready"
    assert review["blockers"] == []
    assert review["ready_for_runtime_apply_review"] is True
    assert review["ready_for_decode_validation_review"] is True
    assert review["ready_for_archive_preflight"] is False
    assert review["ready_for_exact_eval_dispatch"] is False
    assert review["score_claim"] is False
    assert review["dispatch_attempted"] is False
    assert review["runtime_apply_manifest_sha256"]
    assert review["runtime_decode_validation_manifest_sha256"] == runtime_apply_manifest[
        "runtime_decode_validation"
    ]["manifest_sha256_excluding_self"]
    assert decode_validation["status"] == "ready"
    assert decode_validation["ready"] is True
    assert decode_validation["exact_validation_available"] is False
    assert decode_validation["runtime_decode_validation_available"] is True
    assert decode_validation["validation_manifest_sha256"] == runtime_decode_validation[
        "manifest_sha256_excluding_self"
    ]
    assert decode_validation["validated_sections"] == ["latents_and_sidecar_brotli"]
    assert decode_validation["blockers"] == []
    assert set(review["runtime_apply_atom_ids"]) == set(
        planning["selected_atoms_manifest"]["selected_atom_ids"]
    )
    assert manifest["ready_for_runtime_apply_review"] is True
    assert manifest["ready_for_decode_validation_review"] is True
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "missing_runtime_apply_manifest" not in manifest["blockers"]
    assert "missing_exact_decode_validation_manifest" not in manifest["blockers"]
    assert "exact_decode_validation_flag_missing" not in manifest["blockers"]
    assert "compress_time_atom_training_not_implemented" in manifest["blockers"]
    assert "requires_exact_cuda_auth_eval" in manifest["dispatch_blockers"]


def test_build_hnerv_wavelet_compress_time_harness_cli_writes_same_manifest(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    out_dir = tmp_path / "harness"
    json_out = tmp_path / "harness_manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_wavelet_compress_time_harness.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "PR106x_WR01",
            "--output-dir",
            str(out_dir),
            "--target-section",
            "latents_and_sidecar_brotli",
            "--seed",
            "123",
            "--atom-budget",
            "7",
            "--block-size",
            "16",
            "--quant-step",
            "2.0",
            "--expected-source-archive-sha256",
            source.archive_sha256,
            "--expected-source-archive-bytes",
            str(source.archive_bytes),
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    persisted = json.loads((out_dir / "hnerv_wavelet_compress_time_harness.json").read_text(encoding="utf-8"))
    atom_plan = json.loads((out_dir / ATOM_PLAN_FILENAME).read_text(encoding="utf-8"))
    selected_atoms = json.loads((out_dir / SELECTED_ATOMS_FILENAME).read_text(encoding="utf-8"))
    assert payload == persisted
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["output_manifest"]["atom_plan_manifest_path"] == str(out_dir / ATOM_PLAN_FILENAME)
    assert payload["output_manifest"]["selected_atoms_manifest_path"] == str(
        out_dir / SELECTED_ATOMS_FILENAME
    )
    assert payload["atom_plan_manifest"] == atom_plan
    assert payload["selected_atoms_manifest"] == selected_atoms
    assert atom_plan["total_emitted_atom_count"] == 7
    assert selected_atoms["total_selected_atom_count"] == 7
    assert selected_atoms["ready_for_exact_eval_dispatch"] is False
    assert selected_atoms["candidate_archive_path"] is None
    assert atom_plan["ready_for_train_select_apply"] is False
    assert payload["output_manifest"]["decode_validation"]["exact_validation_available"] is False
    assert payload["output_manifest"]["runtime_decode_review"]["status"] == "blocked"


def _source_archive(tmp_path: Path) -> Path:
    decoder = brotli.compress(bytes(range(251)) * 40, quality=1)
    latents = brotli.compress((b"alpha-wavelet-compress-time-signal-" * 80), quality=1)
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + latents
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    return archive


def _runtime_apply_manifest_for_selected_atoms(
    *,
    tmp_path: Path,
    source_archive: Path,
    selected_atoms_manifest: dict[str, object],
) -> dict[str, object]:
    source = read_strict_single_member_zip(source_archive)
    sidechannel_blob = encode_wavelet_atom_sidechannel(
        {
            "sections": selected_atoms_manifest["sections"],
        }
    )
    wavelet_payload = build_wavelet_sidechannel_archive_bytes(
        source_payload=source.payload,
        sidechannel_blob=sidechannel_blob,
    )
    wavelet_archive = tmp_path / "wavelet_for_selected_atoms.zip"
    write_stored_single_member_zip(
        wavelet_archive,
        member_name=source.member_name,
        payload=wavelet_payload,
    )
    return build_wavelet_apply_transform_candidate(
        wavelet_archive=wavelet_archive,
        output_dir=tmp_path / "runtime_apply",
        source_label="PR106x_WR01",
        strength_numerator=1,
        strength_denominator=2,
        source_archive=source_archive,
    )
