# SPDX-License-Identifier: MIT
"""Fail-closed controls for the Vertigo certify-or-block mover."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "tools" / "vertigo_certify_move.py"
    spec = importlib.util.spec_from_file_location("_rvf1_vertigo_certify_move", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mover():
    return _load()


def test_census_refuses_unreadable_file_instead_of_omitting_it(mover, tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()
    victim = root / "payload.bin"
    victim.write_bytes(b"retained payload")
    real_stat = Path.stat

    def fail_victim(path, *args, **kwargs):
        if path == victim:
            raise PermissionError("injected unreadable file")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_victim)
    with pytest.raises(mover.CensusError, match=r"payload\.bin"):
        mover.take_census(root)


def test_census_refuses_unreadable_directory_from_os_walk(mover, tmp_path, monkeypatch):
    root = tmp_path / "tree"
    root.mkdir()

    def walk_with_error(_root, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError("injected scandir refusal"))
        yield from ()

    monkeypatch.setattr(mover.os, "walk", walk_with_error)
    with pytest.raises(mover.CensusError, match="unreadable directory"):
        mover.take_census(root)


def test_census_counts_directory_symlinks_without_following_them(mover, tmp_path):
    root = tmp_path / "tree"
    target = tmp_path / "target"
    root.mkdir()
    target.mkdir()
    (target / "outside.bin").write_bytes(b"outside")
    (root / "linkdir").symlink_to(target, target_is_directory=True)

    census = mover.take_census(root)

    assert census.n_symlinks == 1
    assert census.files == []
    assert census.symlinks == [("linkdir", str(target))]


def test_metadata_manifest_captures_file_modes_and_symlink_targets(mover, tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    executable = root / "run"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o751)
    (root / "alias").symlink_to("run")

    rows = mover.metadata_manifest_rows(mover.take_census(root))

    assert {
        "path": "run",
        "type": "regular_file",
        "mode": 0o751,
    } in rows
    assert {
        "path": "alias",
        "type": "symlink",
        "mode": None,
        "symlink_target": "run",
    } in rows


def test_metadata_mismatch_refuses_direct_move_and_selects_tar_fallback(mover, tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    executable = root / "run"
    executable.write_bytes(b"#!/bin/sh\n")
    executable.chmod(0o755)
    (root / "alias").symlink_to("run")
    census = mover.take_census(root)
    capabilities = mover.DestinationMetadataCapabilities(
        probe_root="synthetic-unrepresentable-destination",
        symlink_supported=True,
        file_mode_results={0o755: 0o700},
        dir_mode_results={stat.S_IMODE(root.stat().st_mode): 0o700},
    )

    blockers = mover.direct_move_metadata_blockers(census, capabilities)

    assert "regular_file_mode_not_representable:0o755->0o700" in blockers


def test_tar_wrap_roundtrip_restores_content_modes_and_symlinks(mover, tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    executable = root / "run"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o751)
    (root / "alias").symlink_to("run")
    census = mover.take_census(root)
    source_manifest = tmp_path / "source.sha256"
    source_digest = mover.build_manifest(
        root,
        [rel for rel, _size in census.files],
        source_manifest,
        workers=1,
    )
    archive = tmp_path / "payload.tar"

    receipt = mover.create_metadata_preserving_tar(root, archive)
    verification = mover.verify_tar_roundtrip(
        source=root,
        source_census=census,
        source_content_manifest_sha256=source_digest,
        archive_path=archive,
        workers=1,
    )

    assert receipt["sha256"] == mover.sha256_file(archive)
    assert verification["content_manifest_equal"] is True
    assert verification["metadata_manifest_equal"] is True


def test_clean_metadata_capable_tree_uses_direct_copy_with_exact_fidelity(
    mover, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "payload.bin"
    payload.write_bytes(b"clean direct payload")
    payload.chmod(0o640)
    census = mover.take_census(source)
    destination = tmp_path / "destination"
    capabilities = mover.probe_destination_metadata_capabilities(
        destination,
        file_modes=set(census.file_modes.values()),
        dir_modes=set(census.dir_modes.values()),
        require_symlink=False,
    )
    assert mover.direct_move_metadata_blockers(census, capabilities) == []
    source_manifest = tmp_path / "source.sha256"
    source_digest = mover.copy_and_hash(
        source,
        destination,
        [rel for rel, _size in census.files],
        source_manifest,
        workers=1,
        census=census,
    )
    destination_census = mover.take_census(destination)
    destination_manifest = tmp_path / "destination.sha256"
    destination_digest = mover.build_manifest(
        destination,
        [rel for rel, _size in destination_census.files],
        destination_manifest,
        workers=1,
    )

    assert source_digest == destination_digest
    assert mover.metadata_manifest_rows(census) == mover.metadata_manifest_rows(
        destination_census
    )
    assert stat.S_IMODE((destination / "payload.bin").stat().st_mode) == 0o640


def test_apply_refuses_unaccounted_reference_surface(mover, monkeypatch, capsys):
    """An empty referenced_by field cannot silently enter a destructive plan."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vertigo_certify_move.py",
            "--source", "/Volumes/VertigoDataTier/pact/example",
            "--dest-root", "/Volumes/APDataStore/pact/vertigo_coldstore",
            "--ledger", "/tmp/rvf1-ledger.jsonl",
            "--category", "rebuildable",
            "--reason", "deterministically rebuildable fixture",
            "--apply",
        ],
    )
    assert mover.main() == 2
    assert "requires --referenced-by" in capsys.readouterr().err


# --- headroom must measure the volume the caller named, not a literal mount ---
#
# The defect this pins: --dest-root (required) drove the destination PATHS while
# the headroom gate and the ledger's dest_df_before measured a hardcoded
# /Volumes/APDataStore. A caller naming any other tier was gated against a volume
# it never asked about, and the ledger row named the wrong filesystem while
# reporting a correct-looking number.


def test_existing_ancestor_walks_up_to_a_path_that_exists(mover, tmp_path):
    """df needs a real path; the destination subtree does not exist until the copy."""
    absent = tmp_path / "not" / "yet" / "created"
    assert mover.existing_ancestor(absent) == tmp_path.resolve()


def test_existing_ancestor_is_identity_for_a_path_that_exists(mover, tmp_path):
    assert mover.existing_ancestor(tmp_path) == tmp_path.resolve()


def test_df_for_path_measures_the_callers_filesystem_not_a_literal_mount(mover, tmp_path):
    """The regression in behavioural form: a local dest reports the LOCAL filesystem."""
    row = mover.df_kib_for_path(tmp_path / "dest" / "subtree")
    assert row["mounted_on"] != "/Volumes/APDataStore"
    assert row["avail_kib"] == mover.df_kib(str(tmp_path))["avail_kib"]
    assert row["device"] and row["mounted_on"]


def test_external_tier_classification_splits_volumes_from_local_disk(mover, tmp_path):
    assert mover.is_external_tier(Path("/Volumes/APDataStore/pact/coldstore")) is True
    assert mover.is_external_tier(Path("/Volumes/VertigoDataTier")) is True
    assert mover.is_external_tier(tmp_path / "coldstore") is False
    assert mover.is_external_tier(Path.home() / "coldstore") is False


def test_headroom_gate_derives_the_destination_volume_from_dest_root(mover):
    """Structural pin: no literal mount may stand in for --dest-root's filesystem.

    Kept as a source assertion because the defect was structural — a literal where
    a variable belonged — and the runtime path needs a real Vertigo source tree.
    The two Vertigo literals are correct BY INVARIANT (main refuses a source that
    is not under /Volumes/VertigoDataTier), so only the destination side is pinned.
    """
    source = (REPO / "tools" / "vertigo_certify_move.py").read_text(encoding="utf-8")
    assert 'df_kib("/Volumes/APDataStore")' not in source
    assert "df_kib_for_path(dest_root)" in source
    assert "df_kib_for_path(transfer_destination)" in source


def test_local_tier_destination_requires_an_explicit_opt_in_flag(mover):
    """CLAUDE.md storage waterfall: local disk is a destination only by opt-in."""
    source = (REPO / "tools" / "vertigo_certify_move.py").read_text(encoding="utf-8")
    assert "--allow-local-tier" in source
    assert "allow_local_tier" in source
