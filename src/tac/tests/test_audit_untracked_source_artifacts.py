from __future__ import annotations

from tools.audit_untracked_source_artifacts import (
    classify_untracked_path,
    find_disposition_for_path,
    find_invalid_disposition_paths,
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
