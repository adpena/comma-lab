# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.g111_physical_native_opener_v1 import (
    CLAIM_SCOPE,
    G111PhysicalNativeOpenError,
    open_g111_native_v3_physical,
)
from tac.witness_control.trajectory_transaction_v2 import (
    ATOMIC_OWNERS,
    CANONICAL_DOMAIN_COVERAGE,
    LINEAGE_ENVELOPE,
    MANIFEST_KEY,
    SCHEMA,
    SEMANTIC_DOMAINS,
    DomainCoverage,
    EntryDescriptor,
    OwnerActivity,
    OwnerClaim,
    TransactionManifest,
    build_manifest,
    manifest_array,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_arrays() -> dict[str, np.ndarray]:
    return {
        f"owner_{index}.state": np.asarray([index, index + 1], dtype=np.int64)
        for index, _owner in enumerate(ATOMIC_OWNERS)
    }


def _valid_manifest(
    arrays: dict[str, np.ndarray],
    *,
    derived_lineage_keys: tuple[str, ...] = (),
) -> TransactionManifest:
    claims = {
        owner: (f"owner_{index}.state",)
        for index, owner in enumerate(ATOMIC_OWNERS)
    }
    return build_manifest(
        arrays,
        owner_claims=claims,
        activity=dict.fromkeys(ATOMIC_OWNERS, True),
        domain_coverage=dict(CANONICAL_DOMAIN_COVERAGE),
        derived_lineage_keys=derived_lineage_keys,
    )


def _write_npz(
    path: Path,
    arrays: dict[str, np.ndarray],
    manifest: TransactionManifest,
) -> None:
    np.savez(path, **arrays, **{MANIFEST_KEY: manifest_array(manifest)})


def _write_valid(path: Path) -> tuple[dict[str, np.ndarray], TransactionManifest]:
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    _write_npz(path, arrays, manifest)
    return arrays, manifest


def _rewrite_manifest(
    path: Path,
    arrays: dict[str, np.ndarray],
    manifest: TransactionManifest,
) -> None:
    _write_npz(path, arrays, manifest)


def test_opens_exact_physical_native_v3_and_returns_immutable_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native.npz"
    arrays, manifest = _write_valid(path)

    receipt = open_g111_native_v3_physical(
        path,
        expected_sha256=_sha256(path),
    )

    assert receipt.path == str(path)
    assert receipt.file_bytes == path.stat().st_size
    assert receipt.file_sha256 == _sha256(path)
    assert receipt.manifest_schema == SCHEMA
    assert receipt.entry_count == len(arrays)
    assert receipt.payload_nbytes == sum(array.nbytes for array in arrays.values())
    assert receipt.claim_scope == CLAIM_SCOPE
    assert tuple(receipt.owner_semantic_sha256) == ATOMIC_OWNERS
    assert tuple(owner.owner for owner in receipt.owners) == ATOMIC_OWNERS
    assert tuple(domain for domain, _owners in receipt.domain_coverage) == SEMANTIC_DOMAINS
    assert receipt.as_dict()["manifest_semantic_sha256"]
    assert receipt.manifest_array_sha256 == hashlib.sha256(
        manifest_array(manifest).tobytes()
    ).hexdigest()
    with pytest.raises(TypeError):
        receipt.owner_semantic_sha256[ATOMIC_OWNERS[0]] = "0" * 64  # type: ignore[index]


def test_requires_exact_lowercase_sha256(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    _write_valid(path)
    with pytest.raises(G111PhysicalNativeOpenError, match="64 lowercase"):
        open_g111_native_v3_physical(path, expected_sha256="A" * 64)
    with pytest.raises(G111PhysicalNativeOpenError, match="SHA-256 mismatch"):
        open_g111_native_v3_physical(path, expected_sha256="0" * 64)


def test_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "native.npz"
    _write_valid(target)
    link = tmp_path / "link.npz"
    link.symlink_to(target)
    with pytest.raises(G111PhysicalNativeOpenError, match="symlink"):
        open_g111_native_v3_physical(link, expected_sha256=_sha256(target))


def test_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "native.npz"
    _write_valid(target)
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(G111PhysicalNativeOpenError, match="path component"):
        open_g111_native_v3_physical(
            link / target.name,
            expected_sha256=_sha256(target),
        )


def test_rejects_object_array_without_pickle(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    arrays["owner_0.state"] = np.asarray([{"not": "canonical"}], dtype=object)
    _write_npz(path, arrays, manifest)
    with pytest.raises(G111PhysicalNativeOpenError, match="requires pickle"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


def test_explicit_derived_lineage_leaf_is_hashed_under_o6(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    arrays["lineage.derived_receipt"] = np.arange(5, dtype=np.uint8)
    manifest = _valid_manifest(
        arrays,
        derived_lineage_keys=("lineage.derived_receipt",),
    )
    _write_npz(path, arrays, manifest)

    receipt = open_g111_native_v3_physical(
        path,
        expected_sha256=_sha256(path),
    )

    o6 = receipt.owners[-1]
    assert "lineage.derived_receipt" not in o6.claimed_keys
    assert "lineage.derived_receipt" in o6.payload_keys
    assert receipt.derived_lineage_keys == ("lineage.derived_receipt",)


def test_rejects_descriptor_content_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    entries = list(manifest.entries)
    original = entries[0]
    entries[0] = EntryDescriptor(
        key=original.key,
        owner=original.owner,
        dtype=original.dtype,
        shape=original.shape,
        nbytes=original.nbytes,
        sha256="0" * 64,
    )
    poisoned = TransactionManifest(
        schema=manifest.schema,
        entries=tuple(entries),
        owner_claims=manifest.owner_claims,
        activity=manifest.activity,
        domain_coverage=manifest.domain_coverage,
        derived_lineage_keys=manifest.derived_lineage_keys,
    )
    _rewrite_manifest(path, arrays, poisoned)
    with pytest.raises(G111PhysicalNativeOpenError, match="descriptor differs"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


def test_rejects_inactive_or_empty_canonical_owner(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    inactive = TransactionManifest(
        schema=manifest.schema,
        entries=manifest.entries,
        owner_claims=manifest.owner_claims,
        activity=(
            OwnerActivity(ATOMIC_OWNERS[0], False),
            *manifest.activity[1:],
        ),
        domain_coverage=manifest.domain_coverage,
        derived_lineage_keys=manifest.derived_lineage_keys,
    )
    _rewrite_manifest(path, arrays, inactive)
    with pytest.raises(G111PhysicalNativeOpenError, match="must be active"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))

    claims = list(manifest.owner_claims)
    claims[0] = OwnerClaim(ATOMIC_OWNERS[0], ())
    empty = TransactionManifest(
        schema=manifest.schema,
        entries=manifest.entries,
        owner_claims=tuple(claims),
        activity=manifest.activity,
        domain_coverage=manifest.domain_coverage,
        derived_lineage_keys=manifest.derived_lineage_keys,
    )
    _rewrite_manifest(path, arrays, empty)
    with pytest.raises(G111PhysicalNativeOpenError, match="at least one"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


def test_rejects_noncanonical_fourteen_domain_coverage(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    coverage = list(manifest.domain_coverage)
    coverage[0] = DomainCoverage(
        coverage[0].domain,
        (LINEAGE_ENVELOPE,),
    )
    poisoned = TransactionManifest(
        schema=manifest.schema,
        entries=manifest.entries,
        owner_claims=manifest.owner_claims,
        activity=manifest.activity,
        domain_coverage=tuple(coverage),
        derived_lineage_keys=manifest.derived_lineage_keys,
    )
    _rewrite_manifest(path, arrays, poisoned)
    with pytest.raises(G111PhysicalNativeOpenError, match="fourteen-domain"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


def test_rejects_owner_reverse_coverage_gap(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    claims = list(manifest.owner_claims)
    claims[0] = OwnerClaim(ATOMIC_OWNERS[0], ("phantom.state",))
    poisoned = TransactionManifest(
        schema=manifest.schema,
        entries=manifest.entries,
        owner_claims=tuple(claims),
        activity=manifest.activity,
        domain_coverage=manifest.domain_coverage,
        derived_lineage_keys=manifest.derived_lineage_keys,
    )
    _rewrite_manifest(path, arrays, poisoned)
    with pytest.raises(G111PhysicalNativeOpenError, match="reverse coverage"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


def test_rejects_noncanonical_manifest_json(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    arrays = _base_arrays()
    manifest = _valid_manifest(arrays)
    noncanonical_json = json.dumps(manifest.as_dict(), indent=2).encode("utf-8")
    np.savez(
        path,
        **arrays,
        **{MANIFEST_KEY: np.frombuffer(noncanonical_json, dtype=np.uint8)},
    )
    with pytest.raises(G111PhysicalNativeOpenError, match="canonical JSON"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


def test_rejects_nonregular_file(tmp_path: Path) -> None:
    directory = tmp_path / "native.npz"
    directory.mkdir()
    with pytest.raises(G111PhysicalNativeOpenError):
        open_g111_native_v3_physical(directory, expected_sha256="0" * 64)


def test_rejects_malformed_npz_with_content_addressed_error(tmp_path: Path) -> None:
    path = tmp_path / "native.npz"
    path.write_bytes(b"")
    with pytest.raises(G111PhysicalNativeOpenError, match="NPZ is malformed"):
        open_g111_native_v3_physical(path, expected_sha256=_sha256(path))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "native.npz"
    os.mkfifo(fifo)
    # Opening a FIFO read-only would block, so the no-follow opener must inspect
    # and reject the non-regular path before attempting an unbounded read.
    with pytest.raises(G111PhysicalNativeOpenError, match="regular file"):
        open_g111_native_v3_physical(fifo, expected_sha256="0" * 64)
