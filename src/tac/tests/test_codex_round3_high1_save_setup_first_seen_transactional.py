"""Tests for codex round-3 HIGH 1 fix — `_save_setup_first_seen` transactional replace.

Bug class (codex round-3 HIGH 1, 2026-05-09): the previous
``_save_setup_first_seen`` reloaded the on-disk file inside the lock and
did ``existing.update(data)``, which silently merged-away the caller's
deliberate deletions of pruned-or-popped instances. Stale ``first_seen``
rows then persisted; ``--auto-destroy-stale`` could auto-destroy a fresh
instance that inherited an old age from the merge.

The fix: locked TRANSACTIONAL REPLACE. The caller owns the post-prune
map and we write it back directly. These tests pin the new contract.

Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_vast_instances.py"

_spec = importlib.util.spec_from_file_location(
    "_verify_vast_round3_high1", SCRIPT_PATH
)
verify_vast = importlib.util.module_from_spec(_spec)
sys.modules["_verify_vast_round3_high1"] = verify_vast
_spec.loader.exec_module(verify_vast)  # type: ignore[union-attr]


def test_save_setup_first_seen_preserves_deletions_under_lock(
    tmp_path, monkeypatch
):
    """Instance deleted from input map MUST stay deleted post-save.

    Pre-fix bug: `_save_setup_first_seen` re-loaded the on-disk file under
    lock and did `existing.update(data)`, so any key present on disk but
    NOT in `data` (because the caller pruned it for "no longer in
    tracker") was silently re-introduced. Repro: write {a, b, c} to disk,
    then call `_save_setup_first_seen({a, c})`. Post-fix: file is exactly
    {a, c}; `b` is gone.
    """
    p = tmp_path / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    # Seed the file with three instances.
    verify_vast._save_setup_first_seen({
        "111": 1714356000.0,
        "222": 1714356100.0,
        "333": 1714356200.0,
    })
    assert verify_vast._load_setup_first_seen() == {
        "111": 1714356000.0,
        "222": 1714356100.0,
        "333": 1714356200.0,
    }
    # Now caller has pruned "222" (e.g. instance no longer in tracker)
    # and deliberately calls save with the post-prune map.
    verify_vast._save_setup_first_seen({
        "111": 1714356000.0,
        "333": 1714356200.0,
    })
    on_disk = verify_vast._load_setup_first_seen()
    assert "222" not in on_disk, (
        "pruned instance silently resurrected — load→update→save merged "
        "stale state. Post-fix contract is transactional replace."
    )
    assert on_disk == {
        "111": 1714356000.0,
        "333": 1714356200.0,
    }


def test_save_setup_first_seen_does_not_resurrect_pruned_ids(
    tmp_path, monkeypatch
):
    """Pruned id MUST NOT reappear from old file.

    Sister of `test_save_setup_first_seen_preserves_deletions_under_lock`
    with end-state focus: the caller intends an EXACT REPLACE; nothing
    from the pre-existing file should leak through.
    """
    p = tmp_path / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    # Seed with old "haunted" rows that should be replaced.
    verify_vast._save_setup_first_seen({
        "ghost1": 1.0, "ghost2": 2.0, "ghost3": 3.0
    })
    # Caller writes a totally fresh map (no overlap with seeded keys).
    verify_vast._save_setup_first_seen({"fresh1": 100.0})
    on_disk = verify_vast._load_setup_first_seen()
    # Post-fix: the file is exactly the caller's input.
    assert on_disk == {"fresh1": 100.0}
    # Belt + suspenders: every ghost row is gone.
    for ghost_key in ("ghost1", "ghost2", "ghost3"):
        assert ghost_key not in on_disk, (
            f"ghost row {ghost_key!r} resurrected post-replace"
        )


def test_save_setup_first_seen_empty_map_clears_file(tmp_path, monkeypatch):
    """Saving the empty map clears the file (i.e. all instances pruned).

    Pre-fix bug: an empty `data` would `update({})` an existing dict,
    leaving the file unchanged. Post-fix: empty caller map = empty file.
    """
    p = tmp_path / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    verify_vast._save_setup_first_seen({"x": 1.0, "y": 2.0})
    assert verify_vast._load_setup_first_seen() == {"x": 1.0, "y": 2.0}
    verify_vast._save_setup_first_seen({})
    assert verify_vast._load_setup_first_seen() == {}


def test_save_setup_first_seen_writes_unique_tmp_path(tmp_path, monkeypatch):
    """Each save uses a unique pid-suffixed tmp path (no shared .tmp clobber)."""
    p = tmp_path / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    verify_vast._save_setup_first_seen({"x": 1.0})
    # After a successful save, no .tmp.* file should remain.
    leftover = list(tmp_path.glob("first_seen.json.tmp.*"))
    assert leftover == []


def test_save_setup_first_seen_overwrite_supersedes_concurrent_seed(
    tmp_path, monkeypatch
):
    """Even if a sister process seeded extra keys, the caller's map wins.

    The fix accepts that "merge concurrent writers" is NOT this writer's
    job — concurrent verify_vast_instances.py invocations are serialized
    on the same lockfile, so the LAST writer's transactional map wins.
    The previous code tried to be helpful by merging, but that helpfulness
    silently undid pruning. The new contract is honest: transactional
    replace, last writer wins.
    """
    p = tmp_path / "first_seen.json"
    monkeypatch.setattr(verify_vast, "SETUP_FIRST_SEEN_PATH", p)
    # Caller A's map (loaded the tracker, classified instances)
    map_a = {"alpha": 1.0, "beta": 2.0}
    # Sister process raced and wrote a different shape between A's load
    # and A's save. (We simulate by writing it directly to disk.)
    p.write_text(json.dumps({"gamma": 3.0}, indent=2, sort_keys=True))
    # Now caller A saves its post-prune map.
    verify_vast._save_setup_first_seen(map_a)
    # A's transactional replace wins; sister's "gamma" is gone.
    on_disk = verify_vast._load_setup_first_seen()
    assert on_disk == map_a
    assert "gamma" not in on_disk
