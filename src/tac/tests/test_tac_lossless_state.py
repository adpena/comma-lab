# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def jsonl_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TacLosslessStateTests(unittest.TestCase):
    def test_render_lossless_latest_is_deterministic(self) -> None:
        from tac.lossless.contracts import LosslessCompressionResult
        from tac.lossless.state import render_lossless_latest

        result = LosslessCompressionResult(
            profile="gpt_arithmetic_small",
            archive_path="artifacts/submission.zip",
            archive_bytes=123,
            original_bytes=456,
            compression_rate=0.26973684210526316,
            method="arithmetic_gpt",
            payload_bytes=111,
            record_count=3,
            checked_items=3,
            split=("0", "1"),
            evidence_root="reports/raw/example",
        )

        self.assertEqual(
            render_lossless_latest(result),
            (
                "# Lossless Latest\n\n"
                "Current promoted lossless baseline is **`0.2697`** via `gpt_arithmetic_small` using `arithmetic_gpt`.\n\n"
                "## promoted result\n\n"
                "- Status: exact round-trip confirmed over `3` items\n"
                "- Profile: `gpt_arithmetic_small`\n"
                "- Method: `arithmetic_gpt`\n"
                "- Compression rate: **`0.2697`**\n"
                "- Archive bytes: `123`\n"
                "- Original bytes: `456`\n"
                "- Archive path: `artifacts/submission.zip`\n"
                "- Ledgers: `reports/lossless_results.jsonl`, `reports/lossless_timeline.jsonl`\n"
            ),
        )

    def test_promote_lossless_result_updates_only_lossless_surfaces(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "lossless_result.json"
            archive_path = root / "submission" / "lossless.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(archive_path),
                        "archive_bytes": archive_path.stat().st_size,
                        "original_bytes": 800,
                        "compression_rate": 800 / archive_path.stat().st_size,
                        "method": "lzma",
                        "record_count": 3,
                        "checked_items": 3,
                    }
                )
            )

            lossy_latest = root / "reports" / "latest.md"
            lossy_results = root / "reports" / "results.jsonl"
            lossy_timeline = root / "reports" / "timeline.jsonl"
            lossy_focus = root / ".omx" / "state" / "current_focus.md"
            lossy_findings = root / ".omx" / "research" / "findings.md"
            for path, content in [
                (lossy_latest, "lossy latest\n"),
                (lossy_results, '{"score": 1.33}\n'),
                (lossy_timeline, '{"score": 1.33}\n'),
                (lossy_focus, "lossy focus\n"),
                (lossy_findings, "lossy findings\n"),
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

            outcome = promote_lossless_result(repo_root=root, result_path=result_path)

            self.assertEqual(outcome["command"], "lossless_promote")
            self.assertEqual(
                outcome["changed_paths"],
                [
                    ".omx/state/lossless_promoted_result.json",
                    "reports/lossless_results.jsonl",
                    "reports/lossless_timeline.jsonl",
                    "reports/lossless_latest.md",
                    ".omx/state/lossless_focus.md",
                    ".omx/state/lossless_next_experiments.md",
                    ".omx/research/lossless_findings.md",
                ],
            )
            self.assertEqual(lossy_latest.read_text(), "lossy latest\n")
            self.assertEqual(lossy_results.read_text(), '{"score": 1.33}\n')
            self.assertEqual(lossy_timeline.read_text(), '{"score": 1.33}\n')
            self.assertEqual(lossy_focus.read_text(), "lossy focus\n")
            self.assertEqual(lossy_findings.read_text(), "lossy findings\n")

            latest = (root / "reports" / "lossless_latest.md").read_text()
            focus = (root / ".omx" / "state" / "lossless_focus.md").read_text()
            next_experiments = (root / ".omx" / "state" / "lossless_next_experiments.md").read_text()
            findings = (root / ".omx" / "research" / "lossless_findings.md").read_text()
            self.assertIn("88.8889", latest)
            self.assertIn("Current promoted lossless baseline is", latest)
            self.assertIn("exact round-trip confirmed over `3` items", latest)
            self.assertIn("lzma_baseline", focus)
            self.assertIn("current promoted baseline", focus)
            self.assertIn("Preserve the first measured lossless baseline", next_experiments)
            self.assertIn("first measured lossless baseline", findings)
            self.assertIn("lossless promotion flow is separate", findings)
            canonical_record = json.loads((root / ".omx" / "state" / "lossless_promoted_result.json").read_text())
            self.assertEqual(canonical_record["profile"], "lzma_baseline")
            self.assertEqual(canonical_record["archive_path"], str(archive_path))

    def test_promote_lossless_result_requires_archive_to_exist(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "lossless_result.json"
            compression_rate = 800 / 200
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(root / "missing.zip"),
                        "archive_bytes": 200,
                        "original_bytes": 800,
                        "compression_rate": compression_rate,
                        "method": "lzma",
                    }
                )
            )

            with self.assertRaises(FileNotFoundError):
                promote_lossless_result(repo_root=root, result_path=result_path)

    def test_promote_lossless_result_requires_archive_bytes_to_match_file(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission" / "lossless.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_path = root / "lossless_result.json"
            compression_rate = 800 / (archive_path.stat().st_size + 1)
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(archive_path),
                        "archive_bytes": archive_path.stat().st_size + 1,
                        "original_bytes": 800,
                        "compression_rate": compression_rate,
                        "method": "lzma",
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "archive_bytes"):
                promote_lossless_result(repo_root=root, result_path=result_path)

    def test_promote_lossless_result_requires_compression_rate_to_match_archive_bytes(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission" / "lossless.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_path = root / "lossless_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(archive_path),
                        "archive_bytes": archive_path.stat().st_size,
                        "original_bytes": 800,
                        "compression_rate": 123.0,
                        "method": "lzma",
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "compression_rate"):
                promote_lossless_result(repo_root=root, result_path=result_path)

    def test_promote_lossless_result_requires_exact_verification_evidence(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission" / "lossless.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_path = root / "lossless_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(archive_path),
                        "archive_bytes": archive_path.stat().st_size,
                        "original_bytes": 800,
                        "compression_rate": 800 / archive_path.stat().st_size,
                        "method": "lzma",
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "checked_items"):
                promote_lossless_result(repo_root=root, result_path=result_path)

    def test_promote_lossless_result_is_idempotent_and_deduplicates_ledgers(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "lossless_result.json"
            archive_path = root / "submission" / "floor.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_payload = {
                "profile": "zpaq_baseline",
                "archive_path": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "original_bytes": 1200,
                "compression_rate": 1200 / archive_path.stat().st_size,
                "method": "zpaq",
                "record_count": 4,
                "checked_items": 4,
            }
            result_path.write_text(json.dumps(result_payload))

            reports = root / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            canonical_results_row = {
                "profile": "zpaq_baseline",
                "method": "zpaq",
                "archive_path": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "original_bytes": 1200,
                "compression_rate": 1200 / archive_path.stat().st_size,
                "event": "promotion",
                "record_count": 4,
                "checked_items": 4,
            }
            canonical_timeline_row = {
                "event": "promotion",
                "profile": "zpaq_baseline",
                "method": "zpaq",
                "compression_rate": 1200 / archive_path.stat().st_size,
                "archive_bytes": archive_path.stat().st_size,
                "original_bytes": 1200,
                "archive_path": str(archive_path),
                "record_count": 4,
                "checked_items": 4,
            }
            (reports / "lossless_results.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"profile": "older", "event": "measurement"}),
                        json.dumps(canonical_results_row),
                        json.dumps(canonical_results_row),
                    ]
                )
                + "\n"
            )
            (reports / "lossless_timeline.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"profile": "older", "event": "measurement"}),
                        json.dumps(canonical_timeline_row),
                        json.dumps(canonical_timeline_row),
                    ]
                )
                + "\n"
            )

            first = promote_lossless_result(repo_root=root, result_path=result_path)
            first_results_text = (reports / "lossless_results.jsonl").read_text()
            first_timeline_text = (reports / "lossless_timeline.jsonl").read_text()
            second = promote_lossless_result(repo_root=root, result_path=result_path)

            self.assertEqual(jsonl_rows(reports / "lossless_results.jsonl"), [
                {"profile": "older", "event": "measurement"},
                canonical_results_row,
            ])
            self.assertEqual(jsonl_rows(reports / "lossless_timeline.jsonl"), [
                {"profile": "older", "event": "measurement"},
                canonical_timeline_row,
            ])
            self.assertTrue(first["changed_paths"])
            self.assertEqual(second["changed_paths"], [])
            self.assertEqual((reports / "lossless_results.jsonl").read_text(), first_results_text)
            self.assertEqual((reports / "lossless_timeline.jsonl").read_text(), first_timeline_text)

    def test_promote_lossless_result_replaces_stale_archive_path_for_same_promoted_result(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "lossless_result.json"
            archive_path = root / "reports" / "raw" / "stable.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_payload = {
                "profile": "lzma_baseline",
                "archive_path": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "original_bytes": 800,
                "compression_rate": 800 / archive_path.stat().st_size,
                "method": "lzma",
                "record_count": 2,
                "checked_items": 2,
            }
            result_path.write_text(json.dumps(result_payload))

            reports = root / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            stale_row = {
                "profile": "lzma_baseline",
                "method": "lzma",
                "archive_path": "/tmp/old.zip",
                "archive_bytes": 200,
                "original_bytes": 800,
                "compression_rate": 800 / archive_path.stat().st_size,
                "event": "promotion",
                "record_count": 2,
                "checked_items": 2,
            }
            stale_timeline = {
                "event": "promotion",
                "profile": "lzma_baseline",
                "method": "lzma",
                "compression_rate": 800 / archive_path.stat().st_size,
                "archive_bytes": 200,
                "original_bytes": 800,
                "archive_path": "/tmp/old.zip",
                "record_count": 2,
                "checked_items": 2,
            }
            (reports / "lossless_results.jsonl").write_text(json.dumps(stale_row) + "\n")
            (reports / "lossless_timeline.jsonl").write_text(json.dumps(stale_timeline) + "\n")

            promote_lossless_result(repo_root=root, result_path=result_path)

            results_rows = jsonl_rows(reports / "lossless_results.jsonl")
            timeline_rows = jsonl_rows(reports / "lossless_timeline.jsonl")

            self.assertEqual(len(results_rows), 1)
            self.assertEqual(results_rows[0]["archive_path"], str(archive_path))
            self.assertEqual(len(timeline_rows), 1)
            self.assertEqual(timeline_rows[0]["archive_path"], str(archive_path))

    def test_promote_lossless_result_matches_comma_lab_lossless_doctor_expectations(self) -> None:
        from src.comma_lab import lossless_state_sync
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "submission" / "stable.zip"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(b"zip-bytes")
            result_path = root / "lossless_result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": str(archive_path),
                        "archive_bytes": archive_path.stat().st_size,
                        "original_bytes": 800,
                        "compression_rate": 800 / archive_path.stat().st_size,
                        "method": "lzma",
                        "record_count": 2,
                        "checked_items": 2,
                    }
                )
            )

            promote_lossless_result(repo_root=root, result_path=result_path)
            report = lossless_state_sync.doctor_repo(root)

        self.assertEqual(report.findings, ())


if __name__ == "__main__":
    unittest.main()
