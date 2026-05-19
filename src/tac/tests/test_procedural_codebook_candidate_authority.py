# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib
import zipfile
from pathlib import Path

import pytest

from tac.procedural_codebook_generator import (
    build_procedural_codebook_candidate_authority,
    build_procedural_seed_authority_packet,
)


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, payload)


def _consumer_module():
    return importlib.import_module(
        "tac.cathedral_consumers.procedural_codebook_generator_consumer"
    )


def _contest_ready_kwargs() -> dict[str, bool]:
    return {
        "runtime_consumption_proof": True,
        "self_contained_archive_proof": True,
        "scorer_free_inflate_proof": True,
        "no_external_state_proof": True,
        "packet_compiler_target_declared": True,
        "exact_eval_validated": False,
    }


def test_archive_seeded_candidate_carries_mutation_smoke_for_consumer(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(archive, [("seed.bin", b"seedbytes"), ("payload.bin", b"payload")])

    candidate = build_procedural_codebook_candidate_authority(
        "archive-seeded",
        archive_path=archive,
        seed_member="seed.bin",
        mutation_offsets=(0, 3),
        **_contest_ready_kwargs(),
    )
    packet = candidate["procedural_seed_authority_packet"]
    archive_mode = packet["modes"]["archive_seeded"]

    assert candidate["score_claim"] is False
    assert candidate["promotion_eligible"] is False
    assert candidate["ready_for_exact_eval_dispatch"] is False
    assert archive_mode["proofs"]["seed_mutation_smoke"]["passed"] is True
    assert archive_mode["proofs"]["seed_member"]["sha256"] == hashlib.sha256(
        b"seedbytes"
    ).hexdigest()

    row = _consumer_module().consume_candidate(candidate)
    authority = row["procedural_authority"]
    assert authority["ready_for_exact_eval_modes"] == ("archive_seeded",)
    assert authority["promotion_eligible_modes"] == ()
    assert not any(
        blocker.startswith("archive_seeded:seed_mutation")
        for blocker in authority["authority_blockers"]
    )


def test_consumer_blocks_archive_seeded_ready_claim_without_mutation_smoke() -> None:
    packet = build_procedural_seed_authority_packet(
        "proofless-archive-seed",
        modes=("archive_seeded",),
        **_contest_ready_kwargs(),
    )

    row = _consumer_module().consume_candidate({"procedural_seed_authority_packet": packet})

    authority = row["procedural_authority"]
    assert authority["ready_for_exact_eval_modes"] == ()
    assert "archive_seeded:seed_mutation_smoke_missing_or_failed" in authority[
        "authority_blockers"
    ]


def test_weight_derived_candidate_carries_frozen_source_and_no_new_bytes(
    tmp_path: Path,
) -> None:
    before = tmp_path / "before.zip"
    after = tmp_path / "after.zip"
    members = [("renderer.bin", b"charged-weights")]
    _write_zip(before, members)
    _write_zip(after, members)

    candidate = build_procedural_codebook_candidate_authority(
        "weight-derived",
        archive_path=after,
        source_member="renderer.bin",
        before_archive=before,
        after_archive=after,
        **_contest_ready_kwargs(),
    )
    packet = candidate["procedural_seed_authority_packet"]
    weight_mode = packet["modes"]["weight_derived"]

    assert weight_mode["source_member_sha256"] == hashlib.sha256(
        b"charged-weights"
    ).hexdigest()
    assert weight_mode["proofs"]["no_new_bytes_added"]["passed"] is True

    row = _consumer_module().consume_candidate(candidate)
    authority = row["procedural_authority"]
    assert authority["ready_for_exact_eval_modes"] == ("weight_derived",)
    assert "weight_derived:source_member_sha256_missing" not in authority[
        "authority_blockers"
    ]
    assert "weight_derived:no_new_bytes_proof_missing_or_failed" not in authority[
        "authority_blockers"
    ]


def test_weight_derived_candidate_without_archive_pair_is_not_ready(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(archive, [("renderer.bin", b"charged-weights")])

    candidate = build_procedural_codebook_candidate_authority(
        "weight-derived-missing-proof",
        archive_path=archive,
        source_member="renderer.bin",
        **_contest_ready_kwargs(),
    )

    row = _consumer_module().consume_candidate(candidate)

    authority = row["procedural_authority"]
    assert authority["ready_for_exact_eval_modes"] == ()
    assert "weight_derived:no_new_bytes_proof_missing_or_failed" in authority[
        "authority_blockers"
    ]


def test_candidate_authority_requires_a_requested_mode(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    _write_zip(archive, [("payload.bin", b"x")])

    with pytest.raises(ValueError, match="at least one"):
        build_procedural_codebook_candidate_authority(
            "empty",
            archive_path=archive,
        )
