from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.training_source_provenance import capture_training_source_provenance
from tac.upstream_source_closure import (
    UPSTREAM_SOURCE_CLOSURE_MEMBERS,
    UPSTREAM_SOURCE_CLOSURE_SCHEMA,
    UpstreamSourceClosureError,
    compute_upstream_source_closure_identity,
)


def _write_members(root: Path) -> None:
    upstream = root / "upstream"
    upstream.mkdir(parents=True)
    for index, relative in enumerate(UPSTREAM_SOURCE_CLOSURE_MEMBERS):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}\n".encode())


def test_source_closure_is_portable_and_ignores_unrelated_workspace(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_members(first)
    _write_members(second)
    (first / "upstream" / ".venv").symlink_to(tmp_path / "host-venv")
    (second / "upstream" / "models").mkdir()
    (second / "upstream" / "models" / "segnet.safetensors").write_bytes(b"separate")

    identity_a = compute_upstream_source_closure_identity(first)
    identity_b = compute_upstream_source_closure_identity(second)

    assert identity_a["schema"] == UPSTREAM_SOURCE_CLOSURE_SCHEMA
    assert identity_a["root"] != identity_b["root"]
    assert identity_a["members"] == identity_b["members"]
    assert identity_a["closure_sha256"] == identity_b["closure_sha256"]
    assert all(set(row) == {"relative_path", "bytes", "sha256"} for row in identity_a["members"])


def test_source_closure_rejects_missing_or_symlinked_required_member(
    tmp_path: Path,
) -> None:
    _write_members(tmp_path)
    missing = tmp_path / "upstream" / UPSTREAM_SOURCE_CLOSURE_MEMBERS[0]
    missing.unlink()
    with pytest.raises(UpstreamSourceClosureError, match="absent"):
        compute_upstream_source_closure_identity(tmp_path)

    missing.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(UpstreamSourceClosureError, match="symlink"):
        compute_upstream_source_closure_identity(tmp_path)


def test_live_source_closure_matches_g46_member_custody() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    identity = compute_upstream_source_closure_identity(repo_root)
    receipt_path = Path(
        "/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/00_custody_storage_preflight.json"
    )
    if not receipt_path.is_file():
        pytest.skip("G46 external custody is not mounted")
    g46 = json.loads(receipt_path.read_text())
    external_rows = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in g46["upstream_closure"]["members"]
    ]
    assert identity["members"] == external_rows
    assert identity["closure_sha256"] != g46["upstream_closure"]["closure_sha256"]


def test_training_provenance_uses_portable_source_closure(tmp_path: Path) -> None:
    _write_members(tmp_path)
    (tmp_path / "upstream" / ".venv").symlink_to(tmp_path / "host-venv")
    provenance = capture_training_source_provenance(tmp_path)
    expected = compute_upstream_source_closure_identity(tmp_path)

    assert provenance["git_sha"] == "unknown"
    assert provenance["git_dirty"] is False
    assert provenance["upstream_snapshot_schema"] == UPSTREAM_SOURCE_CLOSURE_SCHEMA
    assert provenance["upstream_snapshot_sha256"] == expected["closure_sha256"]
