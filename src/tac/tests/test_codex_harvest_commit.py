from __future__ import annotations

from types import SimpleNamespace

from tools import codex_harvest_commit as harvest


def test_safe_only_harvest_refuses_without_named_consumer(monkeypatch) -> None:
    monkeypatch.setattr(harvest, "_manifest_files", lambda _label, _stamp: (["receipt.json"], {}))
    monkeypatch.setattr(harvest, "_changed_files", lambda: {"receipt.json"})

    called = False

    def _unexpected_serializer(*_args, **_kwargs) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(harvest, "_serializer_commit", _unexpected_serializer)

    assert harvest.harvest("lane", "stamp", None, False, None) == 2
    assert called is False


def test_harvest_forwards_named_consumer_to_terminal_disposition(monkeypatch) -> None:
    monkeypatch.setattr(harvest, "_manifest_files", lambda _label, _stamp: (["receipt.json"], {}))
    monkeypatch.setattr(harvest, "_changed_files", lambda: {"receipt.json"})
    monkeypatch.setattr(harvest, "_serializer_commit", lambda *_args, **_kwargs: 0)

    calls: list[list[str]] = []

    def _run(args, **_kwargs):
        calls.append([str(value) for value in args])
        if "rev-parse" in args:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harvest.subprocess, "run", _run)

    consumer = ".omx/research/example_DAG_FEED.md"
    assert harvest.harvest("lane", "stamp", None, False, consumer) == 0
    disposition = next(args for args in calls if "disposition" in args)
    assert disposition[disposition.index("--consumed-by") + 1] == consumer


def test_isolated_harvest_refuses_dirty_worktree_before_merge(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def _run(args, **_kwargs):
        calls.append([str(value) for value in args])
        return SimpleNamespace(returncode=0, stdout=" M uncommitted.py\n", stderr="")

    monkeypatch.setattr(harvest.subprocess, "run", _run)

    assert harvest.merge_worktree("lane", "stamp", "branch", str(tmp_path), True, "task#1") == 2
    assert not any("merge" in args for args in calls)


def test_harvester_never_bypasses_hooks_or_force_deletes_branches() -> None:
    source = harvest.Path(harvest.__file__).read_text(encoding="utf-8")
    assert '"--no-verify"' not in source
    assert '"-D", branch' not in source
