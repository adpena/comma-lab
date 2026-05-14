# SPDX-License-Identifier: MIT
"""Tests for Catalog #158 — `check_deterministic_compiler_canonical_use`.

The gate refuses packet-compilation surfaces that bypass the canonical
`tac.packet_compiler.deterministic_compiler`. Pre-existing surfaces are
grandfathered in only through SHA-pinned legacy waivers; changed legacy files
must route through the canonical compiler or carry an explicit waiver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_deterministic_compiler_canonical_use,
)


def _make_tools_repo(tmp_path: Path) -> Path:
    """Create a fake repo with a tools/ directory."""

    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    return repo


_CANONICAL_BUILDER = '''\
#!/usr/bin/env python3
"""Canonical packet builder."""
import zipfile
from tac.packet_compiler.deterministic_compiler import compile_packet

def main():
    with zipfile.ZipFile("archive.zip", "w") as zf:
        zf.writestr("inflate.py", b"data")
    compile_packet("archive.zip", output_dir="out", mode="identity")

if __name__ == "__main__":
    main()
'''

_OFFENDING_BUILDER = '''\
#!/usr/bin/env python3
"""Builds a packet without routing through the canonical compiler."""
import zipfile

def build():
    with zipfile.ZipFile("archive.zip", "w") as zf:
        zf.writestr("renderer.bin", b"data")
    with open("inflate.sh", "w") as fh:
        fh.write("#!/bin/bash\\n")

if __name__ == "__main__":
    build()
'''

_WAIVED_BUILDER = '''\
#!/usr/bin/env python3
"""Builds a packet under explicit waiver."""
import zipfile  # DETERMINISTIC_COMPILER_OK:legacy-research-only-no-inflate-trio

def build():
    with zipfile.ZipFile("archive.zip", "w") as zf:
        zf.writestr("inflate.py", b"data")

if __name__ == "__main__":
    build()
'''

_NON_PACKET_BUILDER = '''\
#!/usr/bin/env python3
"""A non-packet build tool that happens to write a zip."""
import zipfile

def build():
    with zipfile.ZipFile("archive.zip", "w") as zf:
        zf.writestr("data.bin", b"x")
'''

_READ_ONLY_INSPECTOR = '''\
#!/usr/bin/env python3
"""Read-only packet inspector — no archive.zip write, no inflate emit."""
import zipfile

def inspect():
    with zipfile.ZipFile("archive.zip", "r") as zf:
        return zf.namelist()
'''

_COMMENT_ONLY_BUILDER = '''\
#!/usr/bin/env python3
"""Mentions tac.packet_compiler.deterministic_compiler without using it."""
import zipfile

def build():
    # tac.packet_compiler.deterministic_compiler is not a compiler call.
    with zipfile.ZipFile("archive.zip", "w") as zf:
        zf.writestr("renderer.bin", b"data")
    with open("inflate.sh", "w") as fh:
        fh.write("#!/bin/bash\\n")

if __name__ == "__main__":
    build()
'''

_ALIAS_CANONICAL_BUILDER = '''\
#!/usr/bin/env python3
"""Canonical packet builder with module alias."""
import zipfile
import tac.packet_compiler.deterministic_compiler as dc

def main():
    with zipfile.ZipFile("archive.zip", "w") as zf:
        zf.writestr("inflate.py", b"data")
    dc.compile_packet("archive.zip", output_dir="out", mode="identity")

if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# Canonical / waived / non-packet surfaces all pass
# ---------------------------------------------------------------------------


def test_canonical_builder_passes(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_packet.py").write_text(_CANONICAL_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert violations == []


def test_canonical_alias_builder_passes(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_packet.py").write_text(_ALIAS_CANONICAL_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert violations == []


def test_waiver_token_exempts(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_packet.py").write_text(_WAIVED_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert violations == []


def test_non_packet_filename_ignored(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    # No 'packet' / 'deterministic' / 'submission' in name.
    (repo / "tools" / "build_dataset.py").write_text(_OFFENDING_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert violations == []


def test_read_only_inspector_ignored(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_packet_inspector.py").write_text(_READ_ONLY_INSPECTOR)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    # Read-only inspector: no inflate emit, no violation.
    assert violations == []


# ---------------------------------------------------------------------------
# Offending surfaces are caught
# ---------------------------------------------------------------------------


def test_offending_builder_caught_warn(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_packet.py").write_text(_OFFENDING_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1
    assert "build_my_packet.py" in violations[0]
    assert "tac.packet_compiler.deterministic_compiler" in violations[0]


def test_comment_only_compiler_reference_is_not_compliance(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_packet.py").write_text(_COMMENT_ONLY_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1
    assert "AST-proof" in violations[0]


def test_offending_builder_strict_raises(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_packet.py").write_text(_OFFENDING_BUILDER)
    with pytest.raises(PreflightError) as excinfo:
        check_deterministic_compiler_canonical_use(
            repo_root=repo, strict=True,
        )
    assert "Catalog #158" in str(excinfo.value)


def test_deterministic_in_name_caught(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_my_deterministic_thing.py").write_text(
        _OFFENDING_BUILDER,
    )
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1


def test_submission_in_name_caught(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_submission_v2.py").write_text(_OFFENDING_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1


def test_materialize_archive_surface_caught(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "materialize_new_archive_candidate.py").write_text(
        _OFFENDING_BUILDER,
    )
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1
    assert "materialize_new_archive_candidate.py" in violations[0]


def test_materialize_packet_surface_caught(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "materialize_new_packet_candidate.py").write_text(
        _OFFENDING_BUILDER,
    )
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1
    assert "materialize_new_packet_candidate.py" in violations[0]


# ---------------------------------------------------------------------------
# Pre-existing allowlist surfaces are grandfathered
# ---------------------------------------------------------------------------


def test_allowlisted_legacy_surfaces_passes_real_repo() -> None:
    """The real repo should have 0 violations after the landing commit."""

    # Use the actual repo root so we exercise the live allowlist.
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo_root, strict=False,
    )
    # If this fails, a new surface was added to tools/ that needs to either
    # route through the canonical compiler, add itself to the allowlist, or
    # carry a same-line waiver.
    assert violations == [], "\n".join(violations)


def test_legacy_sha_pin_does_not_exempt_changed_offender(tmp_path: Path) -> None:
    """A legacy filename alone is not enough once the content hash changes."""

    repo = _make_tools_repo(tmp_path)
    # Use a known-allowlisted name.
    (repo / "tools" / "build_pr101_runtime_packet.py").write_text(
        _OFFENDING_BUILDER,
    )
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 1
    assert "Legacy SHA pin changed" in violations[0]


def test_missing_tools_dir_returns_empty(tmp_path: Path) -> None:
    # No tools/ dir; gate should no-op (e.g. running outside a repo).
    repo = tmp_path / "no_tools_repo"
    repo.mkdir()
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=True,  # Strict OK because no violations possible.
    )
    assert violations == []


def test_multiple_offending_surfaces_all_reported(tmp_path: Path) -> None:
    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_alpha_packet.py").write_text(_OFFENDING_BUILDER)
    (repo / "tools" / "build_beta_packet.py").write_text(_OFFENDING_BUILDER)
    (repo / "tools" / "build_gamma_packet.py").write_text(_OFFENDING_BUILDER)
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert len(violations) == 3
    names = "\n".join(violations)
    assert "build_alpha_packet.py" in names
    assert "build_beta_packet.py" in names
    assert "build_gamma_packet.py" in names


def test_zip_without_inflate_emission_ignored(tmp_path: Path) -> None:
    """A builder that writes archive.zip but does NOT mention inflate is exempt."""

    repo = _make_tools_repo(tmp_path)
    (repo / "tools" / "build_metric_packet.py").write_text('''\
import zipfile

def build():
    with zipfile.ZipFile("metrics.zip", "w") as zf:
        zf.writestr("m.json", b"{}")
''')
    violations = check_deterministic_compiler_canonical_use(
        repo_root=repo, strict=False,
    )
    assert violations == []
