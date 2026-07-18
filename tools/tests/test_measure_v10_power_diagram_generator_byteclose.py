# SPDX-License-Identifier: MIT
"""Containment tests for the retired v10 measurement tool tombstone."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import tools.measure_v10_power_diagram_generator_byteclose as tombstone


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize(
    "entrypoint",
    [
        tombstone.refuse,
        tombstone.main,
        tombstone.run_measurement,
        tombstone.prepare_extraction_scratch,
        tombstone.certify_feature_cache,
        tombstone.cleanup_certified_scratch,
    ],
)
def test_every_live_entrypoint_unconditionally_refuses_without_filesystem_mutation(
    tmp_path: Path, entrypoint: object
) -> None:
    evidence = tmp_path / "preserve.bin"
    evidence.write_bytes(b"immutable")
    before = _snapshot_tree(tmp_path)
    with pytest.raises(
        tombstone.RetiredV10MeasurementToolError,
        match=r"retired.*cleanup certificate",
    ):
        entrypoint(tmp_path, output=tmp_path / "would-be-output.json")
    assert _snapshot_tree(tmp_path) == before


def test_tombstone_exposes_no_historical_math_or_cleanup_implementation() -> None:
    assert tombstone.TOMBSTONE_STATUS == "RETIRED_UNSAFE_CLEANUP_CERTIFICATE_FAIL_CLOSED"
    for historical_name in (
        "StreamingRidgeSufficientStatistics",
        "ExtractionState",
        "compression_accounting",
        "validate_extraction_checkpoint",
        "atomic_write_json",
        "sha256_file",
    ):
        assert not hasattr(tombstone, historical_name)


def test_tombstone_source_does_not_import_historical_snapshot() -> None:
    source = Path(tombstone.__file__).read_text(encoding="utf-8")
    assert "importlib" not in source
    assert ".source.txt" not in source
    assert "exec(" not in source
    assert "unlink(" not in source
    assert "rmtree(" not in source


def test_historical_gzip_container_fails_direct_python_before_import_and_preserves_sentinel(
    tmp_path: Path,
) -> None:
    repo_root = Path(tombstone.__file__).resolve().parents[1]
    container = (
        repo_root / ".omx/research/evidence/measure_v10_power_diagram_generator_byteclose_"
        "be094a1540a94bf51aa98706b6d4515eec150bb569380f69b308ed66556cd7c9.source.gz"
    )
    sentinel = tmp_path / "mutation_sentinel.bin"
    sentinel.write_bytes(b"UNCHANGED")
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(container), "--output", str(sentinel)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0
    assert b"SyntaxError" in result.stderr
    assert b"argparse" not in result.stderr
    assert sentinel.read_bytes() == b"UNCHANGED"
    assert _snapshot_tree(tmp_path) == {"mutation_sentinel.bin": b"UNCHANGED"}
