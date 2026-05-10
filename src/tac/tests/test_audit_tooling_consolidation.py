from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.source_index import SourceIndex
from tools.audit_tooling_consolidation import audit_tooling

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "audit_tooling_consolidation.py"


def test_audit_tooling_detects_duplicate_helper_patterns(tmp_path: Path) -> None:
    sample_dir = tmp_path / "tools"
    sample_dir.mkdir()
    sample = sample_dir / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "REPO = Path(__file__).resolve().parents[1]",
                "def _sha256_file(path):",
                "    return 'x'",
                "print(json.dumps({'score_claim': False}, indent=2, sort_keys=True))",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_tooling(tmp_path, ("tools",))
    payload = report.to_dict()

    assert payload["ready_for_incremental_consolidation"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    counts = payload["summary"]["pattern_counts"]
    assert counts["local_sha256_helper"] == 1
    assert counts["local_json_dump"] == 1
    assert counts["manual_repo_root_parents"] == 1
    assert counts["manual_audit_score_dispatch_metadata"] == 1
    assert payload["summary"]["affected_file_count"] == 1
    assert payload["per_file_counts"]["tools/sample.py"]["local_json_dump"] == 1


def test_audit_tooling_prunes_excluded_result_trees(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    live = experiments / "live.py"
    live.write_text("def _sha256_file(path):\n    return 'live'\n", encoding="utf-8")
    excluded_dir = experiments / "results" / "public_pr"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "archived.py").write_text(
        "def _sha256_file(path):\n    return 'archived'\n",
        encoding="utf-8",
    )

    report = audit_tooling(tmp_path, ("experiments",))
    payload = report.to_dict()

    assert payload["summary"]["file_count"] == 1
    assert payload["summary"]["pattern_counts"]["local_sha256_helper"] == 1
    assert "experiments/live.py" in payload["per_file_counts"]
    assert "experiments/results/public_pr/archived.py" not in payload["per_file_counts"]


def test_audit_tooling_source_index_matches_direct_scan(tmp_path: Path) -> None:
    sample_dir = tmp_path / "tools"
    sample_dir.mkdir()
    (sample_dir / "sample.py").write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "from pathlib import Path",
                "def _sha256_file(path):",
                "    return 'x'",
                "DATA = json.dumps({'score_claim': False}, indent=2, sort_keys=True)",
                "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))",
                "ROOT = Path(__file__).resolve().parents[2]",
                "dispatch_attempted = False",
            ]
        ),
        encoding="utf-8",
    )

    direct = audit_tooling(tmp_path, ("tools",))
    indexed = audit_tooling(tmp_path, ("tools",), source_index=SourceIndex(tmp_path))

    assert indexed == direct


def test_audit_tooling_source_index_uses_indexed_file_inventory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_dir = tmp_path / "tools"
    sample_dir.mkdir()
    sample = sample_dir / "sample.py"
    sample.write_text(
        "def _sha256_file(path):\n    return 'x'\n",
        encoding="utf-8",
    )

    class SpyIndex:
        def __init__(self) -> None:
            self.files_containing_substrings_calls = 0
            self.read_text_calls = 0

        def files_containing_substrings(
            self,
            dirs,
            *,
            pattern,
            substrings,
            require_all=True,
        ):
            self.files_containing_substrings_calls += 1
            assert tuple(dirs) == ("tools",)
            assert pattern == "*.py"
            assert "sha256" in tuple(substrings)
            assert "json.dumps" in tuple(substrings)
            assert "score_claim" in tuple(substrings)
            assert require_all is False
            return (sample,)

        def read_text(self, path, *, encoding=None, errors=None):
            self.read_text_calls += 1
            assert path == sample
            assert encoding == "utf-8"
            return sample.read_text(encoding=encoding or "utf-8", errors=errors)

    def fail_iter_files(*args, **kwargs):
        raise AssertionError("SourceIndex-backed audit must not run its own os.walk")

    monkeypatch.setattr(
        "tools.audit_tooling_consolidation.iter_files",
        fail_iter_files,
    )

    spy = SpyIndex()
    payload = audit_tooling(tmp_path, ("tools",), source_index=spy).to_dict()

    assert spy.files_containing_substrings_calls == 1
    assert spy.read_text_calls == 1
    assert payload["summary"]["file_count"] == 1
    assert payload["summary"]["pattern_counts"]["local_sha256_helper"] == 1


def test_audit_tooling_cli_json_contract() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--format",
            "json",
            "--scan-root",
            "tools/audit_hnerv_frontier_scorecard.py",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["audit"] == "tooling_consolidation_inventory"
    assert payload["ready_for_incremental_consolidation"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
