from __future__ import annotations

from pathlib import Path

import tac.preflight as preflight

from tac.preflight import (
    _preflight_all_clean_cache_hit,
    _preflight_developer_clean_cache_hit,
    _store_preflight_developer_clean_cache,
    _store_preflight_all_clean_cache,
)


def test_preflight_all_clean_cache_hits_unchanged_tree(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")
    (tmp_path / ".omx" / "research").mkdir(parents=True)
    (tmp_path / ".omx" / "research" / "ledger.md").write_text("signal\n")
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "state" / "lane_registry.json").write_text("{}\n")

    hit, token, paths = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    assert any(path.name == "example.py" for path in paths)
    assert any(path.name == "ledger.md" for path in paths)
    assert any(path.name == "lane_registry.json" for path in paths)

    _store_preflight_all_clean_cache(tmp_path, cache_token=token, paths=paths)

    hit, _, _ = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is True


def test_preflight_developer_clean_cache_hit_skips_source_index_setup(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")
    hit, token, paths = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    _store_preflight_developer_clean_cache(tmp_path, cache_token=token, paths=paths)

    import tac.source_index as source_index

    def fail_source_index_context(*args, **kwargs):
        raise AssertionError("clean developer cache hit should skip SourceIndex setup")

    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(source_index, "source_index_context", fail_source_index_context)

    preflight.preflight_developer(verbose=False)


def test_prewarm_preflight_source_index_populates_common_fact_groups(
    tmp_path,
    monkeypatch,
):
    for rel, text in {
        "experiments/train_example.py": "import torch\n",
        "experiments/run_example.sh": "#!/bin/sh\n",
        "src/tac/example.py": "VALUE = 'MPS'\n",
        "src/tac/tests/test_example.py": "from tac import example\n",
        "docs/example.md": "CPU advisory only\n",
    }.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    from tac.source_index import SourceIndex, source_index_context

    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    with source_index_context(tmp_path) as index:
        preflight._prewarm_preflight_source_index(tmp_path)
        stats = index.stats()

    assert stats["facts_group_cache_entries"] >= 4
    assert stats["text_facts_cache_entries"] >= 5


def test_prewarm_preflight_source_index_covers_developer_hot_scan_groups(
    tmp_path,
):
    for rel in (
        "scripts/remote_lane.sh",
        "experiments/train_example.py",
        "experiments/run_example.sh",
        "src/tac/example.py",
        "src/tac/contrib/hook.py",
        "src/tac/deploy/launcher.py",
        "src/tac/experiments/worker.py",
        "tools/claim.py",
        "submissions/robust_current/inflate.sh",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# preflight fixture\n")

    from tac.source_index import source_index_context

    hot_groups = (
        (
            (
                "scripts",
                "experiments",
                "src/tac/contrib",
                "src/tac/deploy",
                "src/tac/experiments",
            ),
            "*.py",
        ),
        (("experiments",), "*.sh"),
        (("src/tac", "tools", "experiments"), "*.py"),
        (("src/tac", "tools", "experiments", "scripts"), "*.py"),
        (("src/tac", "experiments", "scripts"), "*.py"),
        (("scripts", "tools", "submissions/robust_current"), "*.sh"),
    )
    with source_index_context(tmp_path) as index:
        preflight._prewarm_preflight_source_index(tmp_path)
        before = index.stats()["facts_group_hits"]
        for dirs, pattern in hot_groups:
            assert index.facts_for_files(dirs, pattern=pattern)
        after = index.stats()["facts_group_hits"]

    assert after - before == len(hot_groups)


def test_prewarm_preflight_source_index_reuses_multi_pattern_inventory(
    tmp_path,
):
    for rel in (
        "docs/ledger.md",
        "reports/latest.md",
        "scripts/run_lane.sh",
        "src/tac/example.py",
        "experiments/train_example.py",
        "experiments/run_example.sh",
        "submissions/robust_current/inflate.sh",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# preflight fixture\n")

    from tac.source_index import source_index_context

    broad_dirs = ("docs", "reports", "scripts", "src/tac", "experiments", "submissions")
    with source_index_context(tmp_path) as index:
        preflight._prewarm_preflight_source_index(tmp_path)
        stats = index.stats()
        assert stats["files_by_pattern_cache_entries"] >= 2

        before_hits = stats["files_by_pattern_hits"]
        grouped = index.files_by_pattern(
            broad_dirs,
            patterns=("*.py", "*.sh", "*.md"),
        )
        after_hits = index.stats()["files_by_pattern_hits"]

    assert after_hits == before_hits + 1
    assert {pattern for pattern, paths in grouped.items() if paths} == {
        "*.py",
        "*.sh",
        "*.md",
    }


def test_preflight_developer_warms_source_index_before_nested_scan(
    tmp_path,
    monkeypatch,
):
    calls: list[Path] = []

    def fake_prewarm(root):
        calls.append(Path(root))

    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(preflight, "_prewarm_preflight_source_index", fake_prewarm)

    preflight.preflight_developer(
        check_codebase=False,
        use_fs_cache=True,
        verbose=False,
    )

    assert calls == []

    try:
        preflight.preflight_developer(
            check_codebase=True,
            use_fs_cache=True,
            verbose=False,
        )
    except Exception:
        # The tiny fixture repo is not expected to satisfy the whole developer
        # preflight; this test only proves the wrapper invokes the prewarm
        # before nested checks run.
        pass

    assert calls == [tmp_path]


def test_preflight_all_clean_cache_ignores_rebuildable_research_artifacts(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")
    artifacts = tmp_path / ".omx" / "research" / "artifacts"
    artifacts.mkdir(parents=True)

    hit, token, paths = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    assert all(".omx/research/artifacts" not in path.as_posix() for path in paths)
    _store_preflight_all_clean_cache(tmp_path, cache_token=token, paths=paths)

    (artifacts / "profile.json").write_text('{"elapsed_s": 31.0}\n')

    hit, _, paths_after = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is True
    assert all(".omx/research/artifacts" not in path.as_posix() for path in paths_after)


def test_preflight_all_clean_cache_misses_after_source_change(tmp_path):
    source = tmp_path / "tool.py"
    source.write_text("VALUE = 1\n")
    hit, token, paths = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    _store_preflight_all_clean_cache(tmp_path, cache_token=token, paths=paths)

    source.write_text("VALUE = 22\n")

    hit, _, _ = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False


def test_preflight_developer_clean_cache_misses_after_source_change(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    source = tmp_path / "src" / "tac" / "example.py"
    source.write_text("VALUE = 1\n")

    hit, token, paths = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    _store_preflight_developer_clean_cache(tmp_path, cache_token=token, paths=paths)

    source.write_text("VALUE = 22\n")

    hit, _, _ = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False


def test_preflight_all_clean_cache_misses_after_public_pr_clone_change(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr90_intake_20260505_auto"
        / "source"
    )
    git_dir = clone / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "index").write_bytes(b"index")
    (clone / "submission.py").write_text("print('upstream')\n")

    hit, token, paths = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    assert any("public_pr90_intake_20260505_auto" in path.as_posix() for path in paths)
    _store_preflight_all_clean_cache(tmp_path, cache_token=token, paths=paths)

    hit, _, _ = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is True

    (clone / "local_waiver.py").write_text("# local waiver must invalidate cache\n")
    hit, _, _ = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False


def test_preflight_all_clean_cache_misses_after_result_status_change(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")
    status = tmp_path / "experiments" / "results" / "candidate" / "dispatch_status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"dirty_path_count": 0}\n')

    hit, token, paths = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    assert status in paths
    _store_preflight_all_clean_cache(tmp_path, cache_token=token, paths=paths)

    status.write_text('{"dirty_paths": ["stale.py"]}\n')
    hit, _, _ = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False


def test_preflight_developer_clean_cache_is_separate_and_hits_unchanged_tree(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")

    hit, token, paths = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    _store_preflight_developer_clean_cache(tmp_path, cache_token=token, paths=paths)
    all_hit, _, _ = _preflight_all_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert all_hit is False

    hit, _, _ = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is True


def test_preflight_developer_clean_cache_ignores_result_status_artifacts(tmp_path):
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "example.py").write_text("VALUE = 1\n")
    status = tmp_path / "experiments" / "results" / "candidate" / "dispatch_status.json"
    status.parent.mkdir(parents=True)
    status.write_text('{"dirty_path_count": 0}\n')

    hit, token, paths = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is False
    assert status not in paths
    _store_preflight_developer_clean_cache(tmp_path, cache_token=token, paths=paths)

    status.write_text('{"dirty_paths": ["developer-cache-ignores-results.py"]}\n')
    hit, _, _ = _preflight_developer_clean_cache_hit(
        tmp_path,
        profile_name=None,
        tto_frames_path=None,
        gt_poses_path=None,
        masks_path=None,
        renderer_path=None,
        archive_path=None,
    )
    assert hit is True
