# SPDX-License-Identifier: MIT
from __future__ import annotations

from tools.audit_untracked_source_artifacts import (
    _generated_source_filesystem_records,
    _generated_source_filesystem_records_via_rg,
    _generated_source_filesystem_records_via_walk,
    build_runtime_source_baseline,
    classify_untracked_path,
    find_disposition_for_path,
    find_invalid_disposition_paths,
    find_runtime_source_custody_blockers,
    load_disposition_manifest,
    parse_git_status_porcelain,
    parse_git_status_records,
)


def test_parse_git_status_porcelain_keeps_only_untracked() -> None:
    text = "\n".join(
        [
            " M src/tac/preflight.py",
            "?? src/tac/new_codec.py",
            "?? experiments/results/rebuildable/blob.json",
            "D  tools/old.py",
        ]
    )

    assert parse_git_status_porcelain(text) == [
        ("??", "src/tac/new_codec.py"),
        ("??", "experiments/results/rebuildable/blob.json"),
    ]


def test_parse_git_status_records_keeps_delete_statuses() -> None:
    text = "\n".join(
        [
            "D  tools/old.py",
            " D reverse_engineering/recovered.py",
            "?? tools/old.py",
        ]
    )

    assert parse_git_status_records(text) == [
        ("D ", "tools/old.py"),
        (" D", "reverse_engineering/recovered.py"),
        ("??", "tools/old.py"),
    ]


def test_classify_source_like_untracked_paths() -> None:
    assert classify_untracked_path("src/tac/new_codec.py").classification == "source_untracked"
    assert classify_untracked_path(".omx/research/new_findings.md").classification == "research_untracked"
    assert classify_untracked_path("reports/summary.json").classification == "source_untracked"
    assert (
        classify_untracked_path("reverse_engineering/pr95/replay.py").classification
        == "reverse_engineering_untracked"
    )


def test_generated_custody_source_paths_are_classified_for_disposition() -> None:
    assert (
        classify_untracked_path("experiments/results/run/manifest.json").classification
        == "generated_custody_source_untracked"
    )
    assert classify_untracked_path(".omx/state/provider.json").classification == "generated_custody_source_untracked"
    assert classify_untracked_path("reports/raw/log.txt").classification == "generated_custody_source_untracked"
    assert classify_untracked_path("outputs/local_run/config.yaml").classification == "generated_custody_source_untracked"
    assert classify_untracked_path("src/tac/model.pt") is None


def test_disposition_manifest_requires_valid_entries(tmp_path) -> None:
    path = tmp_path / "dispositions.json"
    path.write_text(
        """
        {
          "entries": [
            {
              "path": "src/tac/new_codec.py",
              "disposition": "track",
              "note": "canonical source"
            }
          ]
        }
        """
    )

    assert load_disposition_manifest(path)["src/tac/new_codec.py"]["disposition"] == "track"


def test_disposition_manifest_supports_generated_custody_prefixes(tmp_path) -> None:
    path = tmp_path / "dispositions.json"
    path.write_text(
        """
        {
          "entries": [
            {
              "path": "experiments/results/",
              "path_kind": "prefix",
              "disposition": "ignore_rebuildable",
              "note": "raw local custody; promote summaries only"
            }
          ]
        }
        """
    )

    dispositions = load_disposition_manifest(path)
    assert dispositions["experiments/results/"]["path_kind"] == "prefix"
    disposition = find_disposition_for_path(
        dispositions,
        "experiments/results/run/submission_dir/inflate.py",
    )
    assert disposition["disposition"] == "ignore_rebuildable"


def test_disposition_lookup_preserves_exact_and_longest_prefix_semantics() -> None:
    dispositions = {
        "experiments/results/": {
            "disposition": "ignore_rebuildable",
            "note": "broad generated custody",
            "path_kind": "prefix",
        },
        "experiments/results/important_run/": {
            "disposition": "ignore_private",
            "note": "more specific generated custody",
            "path_kind": "prefix",
        },
        "experiments/results/important_run/submission_dir/inflate.py": {
            "disposition": "track",
            "note": "exact runtime source reviewed for promotion",
            "path_kind": "exact",
        },
    }

    assert (
        find_disposition_for_path(
            dispositions,
            "experiments/results/other_run/submission_dir/inflate.py",
        )["disposition"]
        == "ignore_rebuildable"
    )
    assert (
        find_disposition_for_path(
            dispositions,
            "experiments/results/important_run/manifest.json",
        )["disposition"]
        == "ignore_private"
    )
    assert (
        find_disposition_for_path(
            dispositions,
            "experiments/results/important_run/submission_dir/inflate.py",
        )["disposition"]
        == "track"
    )


def test_disposition_manifest_rejects_prefix_outside_generated_custody(tmp_path) -> None:
    path = tmp_path / "dispositions.json"
    path.write_text(
        """
        {
          "entries": [
            {
              "path": "src/",
              "path_kind": "prefix",
              "disposition": "ignore_rebuildable",
              "note": "too broad; would hide source edits"
            }
          ]
        }
        """
    )

    import pytest

    with pytest.raises(ValueError, match="prefix is only valid under generated custody roots"):
        load_disposition_manifest(path)


def test_generated_source_filesystem_records_include_ignored_runtime_sources(tmp_path) -> None:
    rel = "experiments/results/ignored_run/submission_dir/inflate.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text("print('runtime')\n", encoding="utf-8")

    records = _generated_source_filesystem_records(tmp_path, tracked_paths=set())

    assert [record.path for record in records] == [rel]
    assert records[0].classification == "generated_custody_source_untracked"


def test_generated_source_filesystem_records_filters_suffixes_and_ignores_dirs(tmp_path) -> None:
    source_rel = "experiments/results/run/manifest.json"
    nested_source_rel = "experiments/results/run/source.py/notes.md"
    binary_rel = "experiments/results/run/blob.pt"
    (tmp_path / source_rel).parent.mkdir(parents=True)
    (tmp_path / source_rel).write_text("{}\n", encoding="utf-8")
    (tmp_path / nested_source_rel).parent.mkdir(parents=True)
    (tmp_path / nested_source_rel).write_text("notes\n", encoding="utf-8")
    (tmp_path / binary_rel).write_bytes(b"not source")

    records = _generated_source_filesystem_records(tmp_path, tracked_paths=set())

    assert sorted(record.path for record in records) == [source_rel, nested_source_rel]
    assert {record.classification for record in records} == {"generated_custody_source_untracked"}


def test_generated_source_filesystem_records_rg_fast_path_matches_python_walk(tmp_path) -> None:
    relpaths = [
        "experiments/results/a/manifest.json",
        "experiments/results/b/submission_dir/inflate.py",
        ".omx/state/provider.json",
        "outputs/local/config.yaml",
        "reports/raw/run/log.txt",
    ]
    for relpath in relpaths:
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "experiments/results/a/blob.pt").write_bytes(b"not source")

    walked = sorted(
        record.path
        for record in _generated_source_filesystem_records_via_walk(tmp_path, tracked_paths=set())
    )
    ripgrep = _generated_source_filesystem_records_via_rg(tmp_path, tracked_paths=set())
    if ripgrep is None:
        return
    indexed = sorted(
        record.path
        for record in ripgrep
    )
    public = sorted(
        record.path
        for record in _generated_source_filesystem_records(tmp_path, tracked_paths=set())
    )

    assert indexed == walked == public == sorted(relpaths)


def test_runtime_source_under_broad_prefix_requires_exact_entry_or_baseline(tmp_path) -> None:
    dispositions = {
        "experiments/results/": {
            "disposition": "ignore_rebuildable",
            "note": "raw local custody; promote summaries only",
            "path_kind": "prefix",
        }
    }

    blockers, baselines = find_runtime_source_custody_blockers(
        dispositions,
        ["experiments/results/new_run/submission_dir/inflate.py"],
        repo_root=tmp_path,
    )

    assert baselines == []
    assert blockers == [
        "experiments/results/new_run/submission_dir/inflate.py: runtime source-like artifact "
        "matched prefix experiments/results/ without exact disposition or runtime_source_baseline"
    ]


def test_runtime_source_under_broad_prefix_allows_exact_disposition(tmp_path) -> None:
    runtime_path = "experiments/results/new_run/submission_dir/inflate.py"
    dispositions = {
        "experiments/results/": {
            "disposition": "ignore_rebuildable",
            "note": "raw local custody; promote summaries only",
            "path_kind": "prefix",
        },
        runtime_path: {
            "disposition": "ignore_rebuildable",
            "note": "reviewed runtime source artifact",
            "path_kind": "exact",
        },
    }

    blockers, baselines = find_runtime_source_custody_blockers(
        dispositions,
        [runtime_path],
        repo_root=tmp_path,
    )

    assert blockers == []
    assert baselines == []


def test_runtime_source_baseline_detects_new_submission_inflate(tmp_path) -> None:
    baseline_path = "experiments/results/known_run/submission_dir/inflate.py"
    new_path = "experiments/results/new_run/submission_dir/inflate.py"
    (tmp_path / baseline_path).parent.mkdir(parents=True)
    (tmp_path / baseline_path).write_text("print('known')\n")
    (tmp_path / new_path).parent.mkdir(parents=True)
    (tmp_path / new_path).write_text("print('new')\n")
    baseline = build_runtime_source_baseline(tmp_path, [baseline_path])
    dispositions = {
        "experiments/results/": {
            "disposition": "ignore_rebuildable",
            "note": "raw local custody; promote summaries only",
            "path_kind": "prefix",
            "runtime_source_baseline": baseline,
        }
    }

    blockers, baselines = find_runtime_source_custody_blockers(
        dispositions,
        [baseline_path, new_path],
        repo_root=tmp_path,
    )

    assert len(baselines) == 1
    assert baselines[0]["prefix"] == "experiments/results/"
    assert baselines[0]["status"] == "mismatch"
    assert baselines[0]["expected_count"] == 1
    assert baselines[0]["actual_count"] == 2
    assert blockers == [
        "experiments/results/: runtime_source_baseline mismatch "
        f"(expected count=1 sha256={baseline['sha256']}; "
        f"actual count=2 sha256={baselines[0]['actual_sha256']})"
    ]


def test_runtime_source_exact_disposition_excludes_generated_packet_from_prefix_baseline(tmp_path) -> None:
    baseline_path = "experiments/results/known_run/submission_dir/inflate.py"
    generated_packet_path = "experiments/results/new_run/variants/candidate/packet/inflate.py"
    (tmp_path / baseline_path).parent.mkdir(parents=True)
    (tmp_path / baseline_path).write_text("print('known')\n")
    (tmp_path / generated_packet_path).parent.mkdir(parents=True)
    (tmp_path / generated_packet_path).write_text("print('generated packet')\n")
    baseline = build_runtime_source_baseline(tmp_path, [baseline_path])
    dispositions = {
        "experiments/results/": {
            "disposition": "ignore_rebuildable",
            "note": "raw local custody; promote summaries only",
            "path_kind": "prefix",
            "runtime_source_baseline": baseline,
        },
        generated_packet_path: {
            "disposition": "ignore_rebuildable",
            "note": "generated packet-local runtime is raw custody with separate ledger",
            "path_kind": "exact",
        },
    }

    blockers, baselines = find_runtime_source_custody_blockers(
        dispositions,
        [baseline_path, generated_packet_path],
        repo_root=tmp_path,
    )

    assert blockers == []
    assert len(baselines) == 1
    assert baselines[0]["prefix"] == "experiments/results/"
    assert baselines[0]["status"] == "matched"
    assert baselines[0]["actual_count"] == 1


def test_disposition_entries_remain_valid_after_tracking() -> None:
    dispositions = {
        "experiments/results/": {
            "disposition": "ignore_rebuildable",
            "note": "raw local custody; promote summaries only",
            "path_kind": "prefix",
        },
        "src/tac/new_codec.py": {"disposition": "track", "note": "reviewed"},
        "docs/runbooks/new.md": {"disposition": "track", "note": "reviewed"},
        "tools/missing.py": {"disposition": "track", "note": "reviewed"},
    }

    assert find_invalid_disposition_paths(
        dispositions,
        live_untracked_paths={"src/tac/new_codec.py"},
        tracked_source_like_paths={"docs/runbooks/new.md"},
    ) == ["tools/missing.py"]
