from __future__ import annotations

from pathlib import Path

from tac.preflight import check_setup_first_seen_no_split_transactions


def _write_script(repo: Path, text: str) -> None:
    path = repo / "scripts" / "verify_vast_instances.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_check144_ignores_docstring_mentions_of_remove_helper(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        '''
def main():
    """Old flow called remove_setup_first_seen_locked after update."""
    # remove_setup_first_seen_locked is documented here, not called.
    update_setup_first_seen_locked(
        observed_setup_ids=observed,
        left_setup_ids=left,
        tracked_ids=tracked,
        now_ts=now,
    )
''',
    )

    assert (
        check_setup_first_seen_no_split_transactions(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )


def test_check144_flags_real_split_transaction_calls(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        '''
def main():
    update_setup_first_seen_locked(
        observed_setup_ids=observed,
        tracked_ids=tracked,
        now_ts=now,
    )
    remove_setup_first_seen_locked(left)
''',
    )

    violations = check_setup_first_seen_no_split_transactions(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "main" in violations[0]


def test_check144_accepts_canonical_helper_waiver(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        '''
def update_setup_first_seen_locked():  # SETUP_FIRST_SEEN_SINGLE_TXN_OK: canonical helper
    update_setup_first_seen_locked()
    remove_setup_first_seen_locked(set())
''',
    )

    assert (
        check_setup_first_seen_no_split_transactions(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
