from __future__ import annotations

from tools.audit_orphan_recovery_canonicalization import (
    ORPHAN_ROOT,
    build_deletion_records,
    build_shadowed_modified_records,
    canonical_path_for_orphan,
    parse_git_status_records,
)


def test_canonical_path_for_orphan_strips_recovery_prefix() -> None:
    assert (
        canonical_path_for_orphan(f"{ORPHAN_ROOT}/src/tac/recovered.py")
        == "src/tac/recovered.py"
    )
    assert canonical_path_for_orphan("src/tac/recovered.py") is None


def test_build_deletion_records_flags_only_source_like_deletes() -> None:
    records = build_deletion_records(
        parse_git_status_records(
            "\n".join(
                [
                    f"D  {ORPHAN_ROOT}/src/tac/recovered.py",
                    f" D {ORPHAN_ROOT}/tools/unstaged.py",
                    f"D  {ORPHAN_ROOT}/src/tac/model.pt",
                    "D  src/tac/direct_delete.py",
                    "?? src/tac/new.py",
                ]
            )
        ),
        tracked_files={"src/tac/recovered.py"},
    )

    assert [record.path for record in records] == [
        f"{ORPHAN_ROOT}/src/tac/recovered.py",
        f"{ORPHAN_ROOT}/tools/unstaged.py",
        "src/tac/direct_delete.py",
    ]
    assert records[0].canonical_path == "src/tac/recovered.py"
    assert records[0].canonical_tracked is True
    assert records[0].staged_delete is True
    assert records[1].canonical_path == "tools/unstaged.py"
    assert records[1].staged_delete is False
    assert records[2].canonical_path is None


def test_build_shadowed_modified_records_maps_dirty_orphan_copies() -> None:
    records = build_shadowed_modified_records(
        parse_git_status_records(
            "\n".join(
                [
                    f" M {ORPHAN_ROOT}/src/tac/recovered.py",
                    f"M  {ORPHAN_ROOT}/experiments/missing.py",
                    f" D {ORPHAN_ROOT}/tools/delete.py",
                    " M src/tac/direct.py",
                    f" M {ORPHAN_ROOT}/src/tac/model.pt",
                ]
            )
        ),
        tracked_files={"src/tac/recovered.py"},
    )

    assert [record.path for record in records] == [
        f"{ORPHAN_ROOT}/src/tac/recovered.py",
        f"{ORPHAN_ROOT}/experiments/missing.py",
    ]
    assert records[0].canonical_path == "src/tac/recovered.py"
    assert records[0].canonical_tracked is True
    assert records[1].canonical_path == "experiments/missing.py"
    assert records[1].canonical_tracked is False
