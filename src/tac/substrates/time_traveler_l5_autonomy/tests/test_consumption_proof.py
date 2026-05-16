# SPDX-License-Identifier: MIT
"""Tests for the TT5L temporal side-info consumption proof artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.optimization.l5_staircase_v2 import (
    TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH,
    TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256,
    L5V2GateEvidence,
    l5_v2_canonical_sideinfo_gate_evidence,
    l5_v2_dispatch_readiness,
)
from tac.substrates.time_traveler_l5_autonomy.architecture import (
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)
from tac.substrates.time_traveler_l5_autonomy.consumption_proof import (
    TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
    _build_toy_archive,
    build_tt5l_contest_full_frame_sideinfo_consumption_proof,
    build_tt5l_inflate_provenance_manifest,
    build_tt5l_sideinfo_consumption_proof,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tt5l_sideinfo_consumption_proof_builder_is_deterministic(
    tmp_path: Path,
) -> None:
    output_root = (
        Path("experiments/results/time_traveler_l5_v2/test_consumption_proof")
        / tmp_path.name
    )
    proof_path = output_root / "proof.json"
    manifest_path = output_root / "manifest.json"
    work_dir = output_root / "work"

    try:
        first = build_tt5l_sideinfo_consumption_proof(
            artifact_path=proof_path,
            manifest_path=manifest_path,
            work_dir=work_dir,
            repo_root=Path.cwd(),
        )
        first_sha = _sha256_file(first.proof_path)
        second = build_tt5l_sideinfo_consumption_proof(
            artifact_path=proof_path,
            manifest_path=manifest_path,
            work_dir=work_dir,
            repo_root=Path.cwd(),
        )

        assert _sha256_file(second.proof_path) == first_sha
        proof = second.proof
        assert proof["predicate_passed"] is True
        assert proof["score_claim"] is False
        assert proof["promotion_eligible"] is False
        assert proof["ready_for_exact_eval_dispatch"] is False
        assert proof["byte_mutation_proof"]["parser_consumed_bytes"] is True
        assert proof["byte_mutation_proof"]["output_changed"] is True
        assert proof["component_proofs"]["ac_state"]["parser_consumed_bytes"] is True
        assert proof["component_proofs"]["ac_state"]["output_changed"] is True
        assert "not_real_range_or_ans_entropy_decoder" in proof["ac_state_status"]
        for record in second.manifest["outputs"].values():
            assert not str(record["path"]).startswith("/")
    finally:
        shutil.rmtree(Path.cwd() / output_root.parent, ignore_errors=True)


def test_tt5l_sideinfo_consumption_proof_rejects_outside_repo_outputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="artifact_path must be inside repo root"):
        build_tt5l_sideinfo_consumption_proof(
            artifact_path=tmp_path / "proof.json",
            manifest_path="experiments/results/time_traveler_l5_v2/manifest.json",
            work_dir="experiments/results/time_traveler_l5_v2/work",
            repo_root=Path.cwd(),
        )


def test_committed_tt5l_sideinfo_consumption_proof_is_toy_scope_not_gate_final() -> None:
    proof_path = Path(TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH)
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert _sha256_file(proof_path) == TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256

    readiness = l5_v2_dispatch_readiness(
        gate_evidence=[
            L5V2GateEvidence(
                gate_id=TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
                artifact_path=TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH,
                artifact_sha256=TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_SHA256,
                predicate_id=proof["predicate_id"],
                predicate_passed=True,
                evidence_grade="local_no_gpu_parser_and_inflate_consumption_proof",
            )
        ]
    )
    sideinfo_gate = next(
        gate for gate in readiness["gates"]
        if gate["gate_id"] == TT5L_SIDEINFO_CONSUMPTION_GATE_ID
    )
    assert sideinfo_gate["evidence_valid"] is False
    assert any(
        "proof_scope_not_contest_full_frame" in blocker
        for blocker in readiness["blockers"]
    )
    assert readiness["ready_for_dispatch"] is False


def test_l5_v2_canonical_sideinfo_gate_evidence_rejects_toy_scope(
    tmp_path: Path,
) -> None:
    fixture_path = Path(TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH)
    proof_path = tmp_path / TT5L_SIDEINFO_CONSUMPTION_PROOF_ARTIFACT_PATH
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_bytes(fixture_path.read_bytes())

    evidence = l5_v2_canonical_sideinfo_gate_evidence(repo_root=tmp_path)

    assert evidence is None


def _contest_shape_tt5l_archive(
    side_info: np.ndarray,
    *,
    ac_state: bytes = b"contest-sideinfo-proof",
) -> bytes:
    torch.manual_seed(0)
    cfg = TimeTravelerConfig(
        hidden_dim=16,
        num_hidden_layers=2,
        output_height=64,
        output_width=96,
        num_pairs=600,
    )
    substrate = TimeTravelerSubstrate(cfg)
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in substrate.state_dict().items()
    }
    return _build_toy_archive(
        cfg=cfg,
        state_dict=state_dict,
        side_info=side_info,
        ac_state=ac_state,
    )


def _write_contest_file_list_and_outputs(
    root: Path,
    *,
    frame_nbytes: int = 1,
) -> tuple[Path, Path, Path]:
    baseline_dir = root / "baseline_outputs"
    mutated_dir = root / "mutated_outputs"
    baseline_dir.mkdir(parents=True)
    mutated_dir.mkdir(parents=True)
    entries = ["0.mkv"]
    file_list = root / "file_list.txt"
    file_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
    baseline_payload = bytearray(b"\x00" * (1200 * frame_nbytes))
    mutated_payload = bytearray(b"\x00" * (1200 * frame_nbytes))
    mutated_payload[17] = 1
    (baseline_dir / "0.raw").write_bytes(baseline_payload)
    (mutated_dir / "0.raw").write_bytes(mutated_payload)
    return file_list, baseline_dir, mutated_dir


def _write_inflate_provenance(
    root: Path,
    *,
    label: str,
    archive_path: Path,
    output_dir: Path,
    file_list: Path,
    frame_nbytes: int = 1,
) -> Path:
    log_path = output_dir.parent / f"{label}_inflate.log"
    log_path.write_text(f"{label} TT5L inflate exit_code=0\n", encoding="utf-8")
    result = build_tt5l_inflate_provenance_manifest(
        archive_path=archive_path,
        output_dir=output_dir,
        file_list_path=file_list,
        artifact_path=output_dir.parent / f"{label}_inflate_provenance.json",
        command=(
            ".venv/bin/python -m "
            "tac.substrates.time_traveler_l5_autonomy.inflate "
            "<archive_dir> <output_dir> <file_list>"
        ),
        log_path=log_path,
        frame_nbytes=frame_nbytes,
        repo_root=root,
    )
    return result.provenance_path


def test_tt5l_inflate_provenance_builder_binds_archive_runtime_outputs(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_inflate_provenance_{tmp_path.name}"
    )
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        archive = artifact_root / "0.bin"
        archive.write_bytes(_contest_shape_tt5l_archive(np.zeros((600, 45), dtype=np.int8)))
        file_list, output_dir, _ = _write_contest_file_list_and_outputs(artifact_root)
        log_path = artifact_root / "inflate.log"
        log_path.write_text("TT5L inflate completed with exit_code=0\n", encoding="utf-8")

        result = build_tt5l_inflate_provenance_manifest(
            archive_path=archive,
            output_dir=output_dir,
            file_list_path=file_list,
            artifact_path=artifact_root / "inflate_provenance.json",
            command=(
                "PACT_INFLATE_DEVICE=cpu .venv/bin/python -m "
                "tac.substrates.time_traveler_l5_autonomy.inflate "
                "archive_dir output_dir file_list.txt"
            ),
            log_path=log_path,
            repo_root=root,
            frame_nbytes=1,
        )

        payload = result.provenance
        assert result.provenance_path.is_file()
        assert payload["schema"] == "tt5l_inflate_provenance_v1"
        assert payload["archive_sha256"] == _sha256_file(archive)
        assert payload["file_list_sha256"] == _sha256_file(file_list)
        assert payload["runtime_tree_sha256"]
        assert payload["output_aggregate_sha256"]
        assert payload["log_path"] == str(log_path.relative_to(root))
        assert payload["log_sha256"] == _sha256_file(log_path)
        assert payload["log_bytes"] == log_path.stat().st_size
        assert payload["total_frames"] == 1200
        assert payload["score_claim"] is False
        assert payload["promotion_eligible"] is False
        assert payload["ready_for_exact_eval_dispatch"] is False
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_full_frame_sideinfo_proof_counts_video_raw_frames(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_contest_sideinfo_gate_{tmp_path.name}"
    )
    zero_side = np.zeros((600, 45), dtype=np.int8)
    mutated_side = zero_side.copy()
    mutated_side[17, 36:45] = 64
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_archive = artifact_root / "baseline_0.bin"
        mutated_archive = artifact_root / "mutated_0.bin"
        baseline_archive.write_bytes(_contest_shape_tt5l_archive(zero_side))
        mutated_archive.write_bytes(_contest_shape_tt5l_archive(mutated_side))
        file_list, baseline_dir, mutated_dir = _write_contest_file_list_and_outputs(
            artifact_root
        )
        baseline_provenance = _write_inflate_provenance(
            root,
            label="baseline",
            archive_path=baseline_archive,
            output_dir=baseline_dir,
            file_list=file_list,
        )
        mutated_provenance = _write_inflate_provenance(
            root,
            label="mutated",
            archive_path=mutated_archive,
            output_dir=mutated_dir,
            file_list=file_list,
        )
        proof_path = artifact_root / "tt5l_contest_sideinfo_proof.json"
        manifest_path = artifact_root / "tt5l_contest_outputs_manifest.json"

        result = build_tt5l_contest_full_frame_sideinfo_consumption_proof(
            baseline_archive_path=baseline_archive,
            mutated_archive_path=mutated_archive,
            baseline_output_dir=baseline_dir,
            mutated_output_dir=mutated_dir,
            file_list_path=file_list,
            artifact_path=proof_path,
            manifest_path=manifest_path,
            baseline_inflate_provenance_path=baseline_provenance,
            mutated_inflate_provenance_path=mutated_provenance,
            repo_root=root,
            frame_nbytes=1,
        )

        proof = result.proof
        assert proof["proof_scope"] == "contest_full_frame_sideinfo_consumption_proof"
        assert proof["predicate_passed"] is True
        assert proof["score_claim"] is False
        assert proof["byte_mutation_proof"]["parser_consumed_bytes"] is True
        assert proof["byte_mutation_proof"]["output_changed"] is True
        assert proof["byte_mutation_proof"]["non_target_payload_sections_identical"] is True
        assert proof["byte_mutation_proof"]["inflate_provenance_valid"] is True
        section_hashes = proof["byte_mutation_proof"]["section_hashes"]
        for section_name in ("world_model_blob", "ac_state_blob", "meta_blob"):
            assert section_hashes[section_name]["identical"] is True
        assert section_hashes["per_pair_side_info_blob"]["target_section"] is True
        assert proof["byte_mutation_proof"]["n_pairs_hashed"] == 600
        assert proof["byte_mutation_proof"]["total_frames"] == 1200
        assert proof["byte_mutation_proof"]["video_count"] == 1
        assert proof["byte_mutation_proof"]["raw_output_frame_nbytes"] == 1
        assert result.manifest["n_pairs_hashed"] == 600
        assert result.manifest["total_frames"] == 1200
        assert result.manifest["video_count"] == 1

        readiness = l5_v2_dispatch_readiness(
            gate_evidence=[
                {
                    "gate_id": TT5L_SIDEINFO_CONSUMPTION_GATE_ID,
                    "artifact_path": str(proof_path.relative_to(root)),
                    "artifact_sha256": _sha256_file(proof_path),
                    "predicate_id": proof["predicate_id"],
                    "predicate_passed": True,
                    "evidence_grade": (
                        "test_small_frame_nbytes_not_contest_gate_evidence"
                    ),
                }
            ],
            repo_root=root,
        )
        sideinfo_gate = next(
            gate
            for gate in readiness["gates"]
            if gate["gate_id"] == TT5L_SIDEINFO_CONSUMPTION_GATE_ID
        )
        assert sideinfo_gate["evidence_valid"] is False
        assert any(
            "raw_output_frame_nbytes" in blocker
            for blocker in sideinfo_gate["evidence_blockers"]
        )
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_sideinfo_proof_rejects_non_target_section_change(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_contest_sideinfo_non_target_{tmp_path.name}"
    )
    zero_side = np.zeros((600, 45), dtype=np.int8)
    mutated_side = zero_side.copy()
    mutated_side[17, 36:45] = 64
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_archive = artifact_root / "baseline_0.bin"
        mutated_archive = artifact_root / "mutated_0.bin"
        baseline_archive.write_bytes(_contest_shape_tt5l_archive(zero_side))
        mutated_archive.write_bytes(
            _contest_shape_tt5l_archive(
                mutated_side,
                ac_state=b"changed-non-target-ac-state",
            )
        )
        file_list, baseline_dir, mutated_dir = _write_contest_file_list_and_outputs(
            artifact_root
        )
        baseline_provenance = _write_inflate_provenance(
            root,
            label="baseline",
            archive_path=baseline_archive,
            output_dir=baseline_dir,
            file_list=file_list,
        )
        mutated_provenance = _write_inflate_provenance(
            root,
            label="mutated",
            archive_path=mutated_archive,
            output_dir=mutated_dir,
            file_list=file_list,
        )

        result = build_tt5l_contest_full_frame_sideinfo_consumption_proof(
            baseline_archive_path=baseline_archive,
            mutated_archive_path=mutated_archive,
            baseline_output_dir=baseline_dir,
            mutated_output_dir=mutated_dir,
            file_list_path=file_list,
            artifact_path=artifact_root / "proof.json",
            manifest_path=artifact_root / "manifest.json",
            baseline_inflate_provenance_path=baseline_provenance,
            mutated_inflate_provenance_path=mutated_provenance,
            repo_root=root,
            frame_nbytes=1,
        )

        proof = result.proof["byte_mutation_proof"]
        assert result.proof["predicate_passed"] is False
        assert proof["parser_consumed_bytes"] is True
        assert proof["output_changed"] is True
        assert proof["non_target_payload_sections_identical"] is False
        assert proof["section_hashes"]["ac_state_blob"]["identical"] is False
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_sideinfo_proof_requires_inflate_provenance(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_contest_sideinfo_provenance_{tmp_path.name}"
    )
    zero_side = np.zeros((600, 45), dtype=np.int8)
    mutated_side = zero_side.copy()
    mutated_side[17, 36:45] = 64
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_archive = artifact_root / "baseline_0.bin"
        mutated_archive = artifact_root / "mutated_0.bin"
        baseline_archive.write_bytes(_contest_shape_tt5l_archive(zero_side))
        mutated_archive.write_bytes(_contest_shape_tt5l_archive(mutated_side))
        file_list, baseline_dir, mutated_dir = _write_contest_file_list_and_outputs(
            artifact_root
        )

        result = build_tt5l_contest_full_frame_sideinfo_consumption_proof(
            baseline_archive_path=baseline_archive,
            mutated_archive_path=mutated_archive,
            baseline_output_dir=baseline_dir,
            mutated_output_dir=mutated_dir,
            file_list_path=file_list,
            artifact_path=artifact_root / "proof.json",
            manifest_path=artifact_root / "manifest.json",
            repo_root=root,
            frame_nbytes=1,
        )

        proof = result.proof["byte_mutation_proof"]
        assert result.proof["predicate_passed"] is False
        assert proof["inflate_provenance_valid"] is False
        assert "baseline_output_provenance_missing" in proof[
            "inflate_provenance_blockers"
        ]
        assert "mutated_output_provenance_missing" in proof[
            "inflate_provenance_blockers"
        ]
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_sideinfo_proof_rejects_logless_inflate_provenance(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_contest_sideinfo_logless_provenance_{tmp_path.name}"
    )
    zero_side = np.zeros((600, 45), dtype=np.int8)
    mutated_side = zero_side.copy()
    mutated_side[17, 36:45] = 64
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_archive = artifact_root / "baseline_0.bin"
        mutated_archive = artifact_root / "mutated_0.bin"
        baseline_archive.write_bytes(_contest_shape_tt5l_archive(zero_side))
        mutated_archive.write_bytes(_contest_shape_tt5l_archive(mutated_side))
        file_list, baseline_dir, mutated_dir = _write_contest_file_list_and_outputs(
            artifact_root
        )
        baseline_provenance = _write_inflate_provenance(
            root,
            label="baseline",
            archive_path=baseline_archive,
            output_dir=baseline_dir,
            file_list=file_list,
        )
        mutated_provenance = _write_inflate_provenance(
            root,
            label="mutated",
            archive_path=mutated_archive,
            output_dir=mutated_dir,
            file_list=file_list,
        )
        for provenance_path in (baseline_provenance, mutated_provenance):
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            payload.pop("log_path")
            payload.pop("log_sha256")
            payload.pop("log_bytes")
            provenance_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        result = build_tt5l_contest_full_frame_sideinfo_consumption_proof(
            baseline_archive_path=baseline_archive,
            mutated_archive_path=mutated_archive,
            baseline_output_dir=baseline_dir,
            mutated_output_dir=mutated_dir,
            file_list_path=file_list,
            artifact_path=artifact_root / "proof.json",
            manifest_path=artifact_root / "manifest.json",
            baseline_inflate_provenance_path=baseline_provenance,
            mutated_inflate_provenance_path=mutated_provenance,
            repo_root=root,
            frame_nbytes=1,
        )

        proof = result.proof["byte_mutation_proof"]
        assert result.proof["predicate_passed"] is False
        assert proof["inflate_provenance_valid"] is False
        assert "baseline_output_provenance_log_path_missing" in proof[
            "inflate_provenance_blockers"
        ]
        assert "mutated_output_provenance_log_path_missing" in proof[
            "inflate_provenance_blockers"
        ]
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_sideinfo_proof_rejects_stale_inflate_log_hash(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_contest_sideinfo_log_hash_{tmp_path.name}"
    )
    zero_side = np.zeros((600, 45), dtype=np.int8)
    mutated_side = zero_side.copy()
    mutated_side[17, 36:45] = 64
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_archive = artifact_root / "baseline_0.bin"
        mutated_archive = artifact_root / "mutated_0.bin"
        baseline_archive.write_bytes(_contest_shape_tt5l_archive(zero_side))
        mutated_archive.write_bytes(_contest_shape_tt5l_archive(mutated_side))
        file_list, baseline_dir, mutated_dir = _write_contest_file_list_and_outputs(
            artifact_root
        )
        baseline_provenance = _write_inflate_provenance(
            root,
            label="baseline",
            archive_path=baseline_archive,
            output_dir=baseline_dir,
            file_list=file_list,
        )
        mutated_provenance = _write_inflate_provenance(
            root,
            label="mutated",
            archive_path=mutated_archive,
            output_dir=mutated_dir,
            file_list=file_list,
        )
        (artifact_root / "baseline_inflate.log").write_text(
            "TT5L inflate log changed after provenance capture\n",
            encoding="utf-8",
        )

        result = build_tt5l_contest_full_frame_sideinfo_consumption_proof(
            baseline_archive_path=baseline_archive,
            mutated_archive_path=mutated_archive,
            baseline_output_dir=baseline_dir,
            mutated_output_dir=mutated_dir,
            file_list_path=file_list,
            artifact_path=artifact_root / "proof.json",
            manifest_path=artifact_root / "manifest.json",
            baseline_inflate_provenance_path=baseline_provenance,
            mutated_inflate_provenance_path=mutated_provenance,
            repo_root=root,
            frame_nbytes=1,
        )

        proof = result.proof["byte_mutation_proof"]
        assert result.proof["predicate_passed"] is False
        assert proof["inflate_provenance_valid"] is False
        assert "baseline_output_provenance_log_sha256_mismatch" in proof[
            "inflate_provenance_blockers"
        ]
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_sideinfo_proof_cli_returns_nonzero_on_failed_predicate(
    tmp_path: Path,
) -> None:
    root = Path.cwd()
    artifact_root = (
        root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / f"test_contest_sideinfo_cli_fail_{tmp_path.name}"
    )
    side_info = np.zeros((600, 45), dtype=np.int8)
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        baseline_archive = artifact_root / "baseline_0.bin"
        mutated_archive = artifact_root / "mutated_0.bin"
        baseline_archive.write_bytes(_contest_shape_tt5l_archive(side_info))
        mutated_archive.write_bytes(_contest_shape_tt5l_archive(side_info))
        file_list, baseline_dir, mutated_dir = _write_contest_file_list_and_outputs(
            artifact_root
        )
        baseline_provenance = _write_inflate_provenance(
            root,
            label="baseline",
            archive_path=baseline_archive,
            output_dir=baseline_dir,
            file_list=file_list,
        )
        mutated_provenance = _write_inflate_provenance(
            root,
            label="mutated",
            archive_path=mutated_archive,
            output_dir=mutated_dir,
            file_list=file_list,
        )
        proc = subprocess.run(
            [
                "tools/build_tt5l_contest_sideinfo_consumption_proof.py",
                "--baseline-archive",
                str(baseline_archive.relative_to(root)),
                "--mutated-archive",
                str(mutated_archive.relative_to(root)),
                "--baseline-output-dir",
                str(baseline_dir.relative_to(root)),
                "--mutated-output-dir",
                str(mutated_dir.relative_to(root)),
                "--file-list",
                str(file_list.relative_to(root)),
                "--baseline-inflate-provenance",
                str(baseline_provenance.relative_to(root)),
                "--mutated-inflate-provenance",
                str(mutated_provenance.relative_to(root)),
                "--artifact-out",
                str((artifact_root / "proof.json").relative_to(root)),
                "--manifest-out",
                str((artifact_root / "manifest.json").relative_to(root)),
                "--frame-nbytes",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert "predicate_passed=false" in proc.stdout
    finally:
        shutil.rmtree(artifact_root, ignore_errors=True)


def test_tt5l_contest_full_frame_sideinfo_proof_cli_help_uses_repo_venv() -> None:
    result = subprocess.run(
        ["tools/build_tt5l_contest_sideinfo_consumption_proof.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--baseline-archive" in result.stdout
    assert "--manifest-out" in result.stdout
    assert "--baseline-inflate-provenance" in result.stdout
    assert "--frame-nbytes" in result.stdout


def test_tt5l_inflate_provenance_cli_help_uses_repo_venv() -> None:
    result = subprocess.run(
        ["tools/build_tt5l_inflate_provenance.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--archive" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--command" in result.stdout
    assert "--frame-nbytes" in result.stdout
