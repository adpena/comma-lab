# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tools.audit_staged_public_release_hygiene import (
    audit_public_staged_hygiene,
    is_public_release_path,
    select_staged_public_paths,
    staged_blob_text,
)


def test_select_staged_public_paths_excludes_private_custody() -> None:
    paths = [
        "AGENTS.md",
        "docs/release.md",
        ".omx/state/lightning_batch_jobs.json",
        ".omx/research/custody.md",
        "reports/raw/job.json",
        "reports/graphs/final_writeup_draft.md",
        "submissions/apogee/WRITEUP.md",
        "src/tac/codec.py",
    ]

    assert select_staged_public_paths(paths) == [
        "AGENTS.md",
        "docs/release.md",
        "reports/graphs/final_writeup_draft.md",
        "submissions/apogee/WRITEUP.md",
    ]


def test_is_public_release_path_covers_publish_surfaces() -> None:
    assert is_public_release_path("reverse_engineering/README.md")
    assert is_public_release_path("reports/silent_defaults.md")
    assert is_public_release_path(".github/workflows/ci.yml")
    assert is_public_release_path("submissions/apogee/PR_ONE_LINER.sh")
    assert not is_public_release_path("reverse_engineering/raw/intake.py")


def test_audit_public_staged_hygiene_detects_operator_path(tmp_path: Path) -> None:
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "release.md").write_text(
        "private notebook: /Users/adpena/Projects/pact/reports/private.ipynb\n",
        encoding="utf-8",
    )

    payload = audit_public_staged_hygiene(root, ["docs/release.md"])

    assert payload["violation_count"] == 1
    assert "local absolute operator path" in payload["violations"][0]


def test_audit_public_staged_hygiene_allows_placeholders(tmp_path: Path) -> None:
    root = tmp_path
    docs = root / "docs"
    docs.mkdir()
    (docs / "release.md").write_text(
        "Supplement: ${LIGHTNING_SUPPLEMENT_URL}\n",
        encoding="utf-8",
    )

    payload = audit_public_staged_hygiene(root, ["docs/release.md"])

    assert payload["violation_count"] == 0


def test_audit_public_staged_hygiene_reads_index_not_worktree(tmp_path: Path) -> None:
    import subprocess

    root = tmp_path
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    docs = root / "docs"
    docs.mkdir()
    release = docs / "release.md"
    release.write_text("private notebook: /Users/example/private.ipynb\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/release.md"], cwd=root, check=True)
    release.write_text("Supplement: ${LIGHTNING_SUPPLEMENT_URL}\n", encoding="utf-8")

    assert "Users/example" in (staged_blob_text(root, "docs/release.md") or "")
    payload = audit_public_staged_hygiene(root, ["docs/release.md"])

    assert payload["violation_count"] == 1
    assert "local absolute operator path" in payload["violations"][0]


def test_audit_public_staged_hygiene_reads_staged_secret_assignment(
    tmp_path: Path,
) -> None:
    import subprocess

    root = tmp_path
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    docs = root / "docs"
    docs.mkdir()
    release = docs / "release.md"
    release.write_text(
        "os.environ['LIGHTNING_API_KEY'] = 'redacted-test-token'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "docs/release.md"], cwd=root, check=True)
    release.write_text("Supplement: ${LIGHTNING_SUPPLEMENT_URL}\n", encoding="utf-8")

    payload = audit_public_staged_hygiene(root, ["docs/release.md"])

    assert payload["violation_count"] == 1
    assert "explicit secret environment assignment" in payload["violations"][0]
