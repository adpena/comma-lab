from __future__ import annotations

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
