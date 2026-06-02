# SPDX-License-Identifier: MIT
"""NO-FAKE tests for SNeRV receiver archive proof."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tac.substrates.snerv_inverse_steg_carrier.archive import unpack_snerv_archive
from tac.substrates.snerv_inverse_steg_carrier.receiver_proof import (
    AXIS_TAG,
    SCHEMA,
    build_snerv_receiver_archive_proof,
)


def test_receiver_archive_proof_reconstructs_from_archive_sections() -> None:
    proof, archive = build_snerv_receiver_archive_proof(bins=4, levels=2)
    decoded = unpack_snerv_archive(archive.packet)

    assert proof.schema == SCHEMA
    assert proof.axis_tag == AXIS_TAG
    assert proof.receiver_contract_satisfied is True
    assert proof.runtime_consumption_proof_ready is True
    assert proof.receiver_matches_direct is True
    assert proof.max_abs_diff == 0.0
    assert proof.archive_packet_sha256 == decoded.packet_sha256
    assert proof.archive_packet_bytes == archive.total_bytes
    assert proof.section_bytes == archive.section_bytes
    assert proof.score_claim is False
    assert proof.frontier_score_claim is False
    assert proof.promotion_eligible is False
    assert proof.ready_for_exact_eval_dispatch is False
    assert "paired_contest_cpu_cuda_auth_eval_missing" in proof.blockers


def test_receiver_proof_cli_writes_hash_matched_packet(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[5]
    cli_path = repo_root / "tools/prove_snerv_receiver_archive.py"
    spec = importlib.util.spec_from_file_location(
        "prove_snerv_receiver_archive_for_test",
        cli_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report_path = tmp_path / "proof.json"
    packet_path = tmp_path / "proof.snar"
    rc = module.main(
        [
            "--bins",
            "4",
            "--levels",
            "2",
            "--wavelet",
            "haar",
            "--out",
            str(report_path),
            "--packet-out",
            str(packet_path),
        ]
    )
    payload = json.loads(report_path.read_text())

    assert rc == 0
    assert packet_path.exists()
    assert payload["packet_artifact_bytes"] == packet_path.stat().st_size
    assert payload["packet_artifact_sha256"] == payload["archive_packet_sha256"]
    assert payload["packet_artifact_matches_proof"] is True
    assert payload["receiver_contract_satisfied"] is True
    assert payload["wavelet"] == "haar"


def test_receiver_proof_module_imports_no_torch_or_scorer() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.receiver_proof as proof_mod

    with open(proof_mod.__file__) as f:
        src = f.read()
    assert "import torch" not in src
    assert "load_score_exact_scorers" not in src
