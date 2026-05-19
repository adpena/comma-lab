# SPDX-License-Identifier: MIT
"""Tests for Catalog #338 procedural-codebook canonical-use guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_procedural_codebook_generator_canonical_use,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    (root / "submissions" / "candidate").mkdir(parents=True)
    (root / "src" / "tac").mkdir(parents=True)
    return root


def test_static_codebook_without_authority_packet_warns_and_strict_raises(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    offender = root / "experiments" / "train_substrate_seed_codegen.py"
    offender.write_text(
        "PREBAKED_CHROMA_CODEBOOK = [0, 1, 2, 3, 4, 5, 6, 7]\n",
        encoding="utf-8",
    )

    violations = check_procedural_codebook_generator_canonical_use(
        repo_root=root,
        strict=False,
    )

    assert len(violations) == 1
    assert "PREBAKED_CHROMA_CODEBOOK" in violations[0]
    assert "build_procedural_seed_authority_packet" in violations[0]
    with pytest.raises(PreflightError, match="Catalog #338"):
        check_procedural_codebook_generator_canonical_use(
            repo_root=root,
            strict=True,
        )


def test_authority_packet_reference_allows_static_probe_fixture(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate = root / "experiments" / "train_substrate_seed_codegen.py"
    candidate.write_text(
        "from tac.procedural_codebook_generator import build_procedural_seed_authority_packet\n"
        "STATIC_CODEBOOK = [0, 1, 2, 3, 4, 5, 6, 7]\n"
        "AUTH = build_procedural_seed_authority_packet('fixture')\n",
        encoding="utf-8",
    )

    assert (
        check_procedural_codebook_generator_canonical_use(
            repo_root=root,
            strict=False,
        )
        == []
    )


def test_reviewed_waiver_allows_non_score_bearing_constant(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate = root / "experiments" / "train_substrate_seed_codegen.py"
    candidate.write_text(
        "# PROCEDURAL_CODEBOOK_PREBAKED_OK:unit-test-non-score-bearing-fixture\n"
        "CHROMA_PALETTE = [0, 1, 2, 3, 4, 5, 6, 7]\n",
        encoding="utf-8",
    )

    assert (
        check_procedural_codebook_generator_canonical_use(
            repo_root=root,
            strict=True,
        )
        == []
    )


def test_small_non_payload_codebook_constant_is_ignored(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate = root / "experiments" / "train_substrate_seed_codegen.py"
    candidate.write_text("DEBUG_CODEBOOK = [0, 1, 2]\n", encoding="utf-8")

    assert (
        check_procedural_codebook_generator_canonical_use(
            repo_root=root,
            strict=True,
        )
        == []
    )


def test_generic_library_codebook_constant_is_not_candidate_surface(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    library = root / "src" / "tac" / "fp4_quantize.py"
    library.write_text(
        "import torch\n"
        "DEFAULT_CODEBOOK = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])\n"
        "SCHEMA = 'procedural_seed_authority_packet_v1'\n",
        encoding="utf-8",
    )

    assert (
        check_procedural_codebook_generator_canonical_use(
            repo_root=root,
            strict=True,
        )
        == []
    )


def test_comment_only_authority_reference_does_not_pass(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate = root / "tools" / "build_seed_archive_packet.py"
    candidate.write_text(
        "# build_procedural_seed_authority_packet should be used here later\n"
        "STATIC_CODEBOOK = [0, 1, 2, 3, 4, 5, 6, 7]\n",
        encoding="utf-8",
    )

    violations = check_procedural_codebook_generator_canonical_use(
        repo_root=root,
        strict=False,
    )

    assert len(violations) == 1
    assert "STATIC_CODEBOOK" in violations[0]


def test_placeholder_waiver_does_not_pass(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate = root / "tools" / "build_seed_archive_packet.py"
    candidate.write_text(
        "# PROCEDURAL_CODEBOOK_PREBAKED_OK:<rationale>\n"
        "STATIC_CODEBOOK = [0, 1, 2, 3, 4, 5, 6, 7]\n",
        encoding="utf-8",
    )

    violations = check_procedural_codebook_generator_canonical_use(
        repo_root=root,
        strict=False,
    )

    assert len(violations) == 1
    assert "STATIC_CODEBOOK" in violations[0]


def test_schema_enum_strings_containing_codebook_do_not_flag(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    candidate = root / "tools" / "build_seed_archive_packet.py"
    candidate.write_text(
        "CODEBOOK_KIND = 'archive_member_weight_derived'\n"
        "CODEBOOK_SCHEMA = 'procedural_seed_authority_packet_v1'\n",
        encoding="utf-8",
    )

    assert (
        check_procedural_codebook_generator_canonical_use(
            repo_root=root,
            strict=True,
        )
        == []
    )


def test_inflate_py_literal_seed_requires_authority_packet(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    inflate = root / "submissions" / "candidate" / "inflate.py"
    inflate.write_text(
        "SEED_CARRIER = 'inflate_py_literal_seed'\n",
        encoding="utf-8",
    )

    violations = check_procedural_codebook_generator_canonical_use(
        repo_root=root,
        strict=False,
    )

    assert len(violations) == 1
    assert "inflate_py_literal_seed" in violations[0]


def test_preflight_all_wires_catalog_338_warn_only() -> None:
    text = Path("src/tac/preflight.py").read_text(encoding="utf-8")
    call = "check_procedural_codebook_generator_canonical_use(\n            strict=False"
    assert call in text
