from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "reports" / "graphs" / "build_report_history.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_report_history", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def commit_all(repo: Path, message: str, when: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": when,
        "GIT_COMMITTER_DATE": when,
    }
    git(repo, "add", ".", env=env)
    git(repo, "commit", "-m", message, env=env)
    return git(repo, "rev-parse", "HEAD")


class ReportHistoryBuildTests(unittest.TestCase):
    def test_collect_history_payload_includes_commit_context_and_content(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            git(repo, "init")
            git(repo, "config", "user.name", "Test User")
            git(repo, "config", "user.email", "test@example.com")

            notebook = repo / "reports" / "graphs" / "lab_notebook.md"
            focus = repo / ".omx" / "state" / "current_focus.md"
            notebook.parent.mkdir(parents=True, exist_ok=True)
            focus.parent.mkdir(parents=True, exist_ok=True)

            notebook.write_text("# Notebook\n\nfirst pass\n")
            first_commit = commit_all(repo, "add notebook", "2026-04-07T09:00:00+00:00")

            notebook.write_text("# Notebook\n\nsecond pass\nwith more detail\n")
            second_commit = commit_all(repo, "expand notebook", "2026-04-08T10:30:00+00:00")

            focus.write_text("# Focus\n\nstabilize history viewer\n")
            third_commit = commit_all(repo, "record focus", "2026-04-09T12:00:00+00:00")

            payload = mod.collect_history(
                repo,
                include_paths=[
                    "reports/graphs/lab_notebook.md",
                    ".omx/state/current_focus.md",
                ],
            )

            self.assertEqual(payload["file_count"], 2)
            self.assertEqual(payload["snapshot_count"], 3)
            self.assertEqual(payload["repo_head"], third_commit)

            by_path = {item["path"]: item for item in payload["files"]}
            notebook_history = by_path["reports/graphs/lab_notebook.md"]
            self.assertEqual(notebook_history["category"], "report_graph")
            self.assertEqual(len(notebook_history["snapshots"]), 2)
            newest = notebook_history["snapshots"][0]
            oldest = notebook_history["snapshots"][1]
            self.assertEqual(newest["commit"], second_commit)
            self.assertEqual(newest["short_commit"], second_commit[:7])
            self.assertEqual(newest["subject"], "expand notebook")
            self.assertIn("second pass", newest["content"])
            self.assertEqual(newest["line_count"], 4)
            self.assertEqual(oldest["commit"], first_commit)
            self.assertEqual(oldest["subject"], "add notebook")

            focus_history = by_path[".omx/state/current_focus.md"]
            self.assertEqual(focus_history["category"], "durable_state")
            self.assertEqual(focus_history["snapshots"][0]["commit"], third_commit)
            self.assertEqual(focus_history["snapshots"][0]["author_name"], "Test User")

    def test_resolve_default_surface_paths_excludes_site_copies_and_non_markdown(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "reports" / "graphs").mkdir(parents=True, exist_ok=True)
            (repo / "reports" / "graphs" / "site").mkdir(parents=True, exist_ok=True)
            (repo / ".omx" / "state").mkdir(parents=True, exist_ok=True)
            (repo / ".omx" / "research").mkdir(parents=True, exist_ok=True)
            (repo / ".ralph").mkdir(parents=True, exist_ok=True)
            (repo / "reports").mkdir(parents=True, exist_ok=True)

            (repo / "reports" / "graphs" / "lab_notebook.md").write_text("lab\n")
            (repo / "reports" / "graphs" / "dashboard_data.json").write_text("{}\n")
            (repo / "reports" / "graphs" / "site" / "lab_notebook.md").write_text("copied\n")
            (repo / ".omx" / "state" / "current_focus.md").write_text("focus\n")
            (repo / ".omx" / "research" / "findings.md").write_text("findings\n")
            (repo / ".ralph" / "run_log.md").write_text("log\n")
            (repo / "reports" / "latest.md").write_text("latest\n")
            (repo / "reports" / "lossless_latest.md").write_text("lossless latest\n")

            paths = mod.resolve_default_surface_paths(repo)

            self.assertEqual(
                paths,
                [
                    ".omx/research/findings.md",
                    ".omx/state/current_focus.md",
                    ".ralph/run_log.md",
                    "reports/graphs/lab_notebook.md",
                    "reports/latest.md",
                    "reports/lossless_latest.md",
                ],
            )

    def test_write_history_json_emits_expected_top_level_fields(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            git(repo, "init")
            git(repo, "config", "user.name", "Test User")
            git(repo, "config", "user.email", "test@example.com")

            notebook = repo / "reports" / "graphs" / "lab_notebook.md"
            notebook.parent.mkdir(parents=True, exist_ok=True)
            notebook.write_text("# Notebook\n")
            commit_all(repo, "seed notebook", "2026-04-09T13:00:00+00:00")

            output_path = repo / "reports" / "graphs" / "report_history.json"
            payload = mod.write_history_json(
                repo,
                output_path=output_path,
                include_paths=["reports/graphs/lab_notebook.md"],
            )

            on_disk = json.loads(output_path.read_text())
            self.assertEqual(on_disk["version"], 1)
            self.assertEqual(on_disk["file_count"], 1)
            self.assertEqual(on_disk["files"][0]["path"], "reports/graphs/lab_notebook.md")
            self.assertEqual(payload["generated_at_utc"], on_disk["generated_at_utc"])


if __name__ == "__main__":
    unittest.main()
