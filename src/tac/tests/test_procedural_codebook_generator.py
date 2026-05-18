# SPDX-License-Identifier: MIT
"""Tests for tac.procedural_codebook_generator planning-only helpers."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.procedural_codebook_generator import (
    derive_codebook_from_archive_bytes,
    emit_seed,
    expand_seed_to_codebook,
    freeze_source_member_sha256,
    verify_generator_seed_mutation_smoke,
    verify_no_new_bytes_added,
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, payload, compress_type=zipfile.ZIP_STORED)


def test_emit_seed_is_stable_and_shape_sensitive():
    seed_a = emit_seed((5, 256, 3))
    seed_b = emit_seed((5, 256, 3))
    seed_c = emit_seed((5, 256, 4))

    assert len(seed_a) == 8
    assert seed_a == seed_b
    assert seed_a != seed_c


def test_expand_seed_to_codebook_is_deterministic_int8():
    seed = emit_seed((4, 8))
    arr_a = expand_seed_to_codebook(seed, (4, 8))
    arr_b = expand_seed_to_codebook(seed, (4, 8))

    assert arr_a.dtype == np.int8
    assert arr_a.shape == (4, 8)
    assert np.array_equal(arr_a, arr_b)
    assert arr_a.min() >= np.iinfo(np.int8).min
    assert arr_a.max() <= np.iinfo(np.int8).max


def test_expand_seed_to_codebook_changes_with_seed_mutation():
    seed = bytearray(emit_seed((8, 8)))
    baseline = expand_seed_to_codebook(bytes(seed), (8, 8))
    seed[0] ^= 0xFF
    mutated = expand_seed_to_codebook(bytes(seed), (8, 8))

    assert not np.array_equal(baseline, mutated)


def test_expand_seed_to_codebook_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="target_shape"):
        expand_seed_to_codebook(b"seed", ())
    with pytest.raises(ValueError, match="unsupported"):
        expand_seed_to_codebook(b"seed", (2, 2), "normal_float32")
    with pytest.raises(ValueError, match="seed"):
        expand_seed_to_codebook(b"", (2, 2))


def test_expand_seed_to_codebook_is_subprocess_stable():
    seed_hex = emit_seed((3, 7)).hex()
    code = (
        "import hashlib\n"
        "from tac.procedural_codebook_generator import expand_seed_to_codebook\n"
        f"arr = expand_seed_to_codebook(bytes.fromhex({seed_hex!r}), (3, 7))\n"
        "print(hashlib.sha256(arr.tobytes()).hexdigest())\n"
    )
    env = os.environ.copy()
    src_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")

    digest_a = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    digest_b = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()

    assert len(digest_a) == 64
    assert digest_a == digest_b


def test_verify_generator_seed_mutation_smoke_reads_seed_member_from_zip(tmp_path):
    seed = emit_seed((5, 256, 3))
    archive = tmp_path / "archive.zip"
    _write_zip(archive, {"seed.bin": seed, "payload.bin": b"already charged"})

    assert verify_generator_seed_mutation_smoke(Path("seed.bin"), archive, [0, len(seed) - 1])


def test_verify_generator_seed_mutation_smoke_reads_seed_from_extracted_archive(tmp_path):
    seed = emit_seed((2, 2))
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "seed.bin").write_bytes(seed)

    assert verify_generator_seed_mutation_smoke(Path("seed.bin"), archive_dir, [0])


def test_verify_generator_seed_mutation_smoke_fails_missing_or_out_of_range(tmp_path):
    seed = emit_seed((2, 2))
    archive = tmp_path / "archive.zip"
    _write_zip(archive, {"seed.bin": seed})

    assert not verify_generator_seed_mutation_smoke(Path("missing.bin"), archive, [0])
    assert not verify_generator_seed_mutation_smoke(Path("seed.bin"), archive, [])
    assert not verify_generator_seed_mutation_smoke(Path("seed.bin"), archive, [len(seed)])


def test_freeze_source_member_sha256_reads_existing_archive_member(tmp_path):
    archive = tmp_path / "archive.zip"
    payload = b"renderer weights already inside archive"
    _write_zip(archive, {"renderer.bin": payload})

    assert freeze_source_member_sha256(archive, "renderer.bin") == hashlib.sha256(payload).hexdigest()


def test_derive_codebook_from_archive_bytes_uses_member_sha_as_seed(tmp_path):
    archive = tmp_path / "archive.zip"
    payload = b"stable renderer bytes"
    _write_zip(archive, {"renderer.bin": payload})
    member_sha = hashlib.sha256(payload).hexdigest()

    derived = derive_codebook_from_archive_bytes(archive, "renderer.bin", (3, 5))
    expected = expand_seed_to_codebook(bytes.fromhex(member_sha), (3, 5))

    assert derived.dtype == np.int8
    assert np.array_equal(derived, expected)


def test_weight_derived_helpers_reject_missing_or_unsafe_member(tmp_path):
    archive = tmp_path / "archive.zip"
    _write_zip(archive, {"renderer.bin": b"x"})

    with pytest.raises(KeyError):
        freeze_source_member_sha256(archive, "missing.bin")
    with pytest.raises(ValueError, match="unsafe"):
        freeze_source_member_sha256(archive, "../renderer.bin")


def test_verify_no_new_bytes_added_accepts_member_removal_without_growth(tmp_path):
    before = tmp_path / "before.zip"
    after = tmp_path / "after.zip"
    _write_zip(before, {"renderer.bin": b"A" * 128, "old_codebook.bin": b"B" * 128})
    _write_zip(after, {"renderer.bin": b"A" * 128})

    assert verify_no_new_bytes_added(before, after)


def test_verify_no_new_bytes_added_rejects_new_member_or_growth(tmp_path):
    before = tmp_path / "before.zip"
    after_new_member = tmp_path / "after_new_member.zip"
    after_growth = tmp_path / "after_growth.zip"
    _write_zip(before, {"renderer.bin": b"A" * 128})
    _write_zip(after_new_member, {"renderer.bin": b"A" * 128, "derived_codebook.bin": b"B"})
    _write_zip(after_growth, {"renderer.bin": b"A" * 1024})

    assert not verify_no_new_bytes_added(before, after_new_member)
    assert not verify_no_new_bytes_added(before, after_growth)


def test_verify_no_new_bytes_added_rejects_rewritten_retained_member(tmp_path):
    before = tmp_path / "before.zip"
    after_rewrite = tmp_path / "after_rewrite.zip"
    _write_zip(before, {"renderer.bin": b"A" * 128})
    _write_zip(after_rewrite, {"renderer.bin": b"B" * 128})

    assert not verify_no_new_bytes_added(before, after_rewrite)


def test_verify_no_new_bytes_added_rejects_duplicate_members(tmp_path):
    before = tmp_path / "before.zip"
    after = tmp_path / "after.zip"
    _write_zip(before, {"renderer.bin": b"A"})
    with zipfile.ZipFile(after, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("renderer.bin", b"A")
        with pytest.warns(UserWarning, match="Duplicate name"):
            zf.writestr("renderer.bin", b"A")

    assert not verify_no_new_bytes_added(before, after)
