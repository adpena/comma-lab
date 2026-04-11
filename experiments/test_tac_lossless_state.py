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
        )

        self.assertEqual(
            render_lossless_latest(result),
            (
                "# Lossless Latest\n\n"
                "## promoted result\n\n"
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
            result_path.write_text(
                json.dumps(
                    {
                        "profile": "lzma_baseline",
                        "archive_path": "submission/lossless.zip",
                        "archive_bytes": 200,
                        "original_bytes": 800,
                        "compression_rate": 0.25,
                        "method": "lzma",
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
                set(outcome["changed_paths"]),
                {
                    ".omx/research/lossless_findings.md",
                    ".omx/state/lossless_focus.md",
                    ".omx/state/lossless_next_experiments.md",
                    "reports/lossless_latest.md",
                    "reports/lossless_results.jsonl",
                    "reports/lossless_timeline.jsonl",
                },
            )
            self.assertEqual(lossy_latest.read_text(), "lossy latest\n")
            self.assertEqual(lossy_results.read_text(), '{"score": 1.33}\n')
            self.assertEqual(lossy_timeline.read_text(), '{"score": 1.33}\n')
            self.assertEqual(lossy_focus.read_text(), "lossy focus\n")
            self.assertEqual(lossy_findings.read_text(), "lossy findings\n")

            latest = (root / "reports" / "lossless_latest.md").read_text()
            focus = (root / ".omx" / "state" / "lossless_focus.md").read_text()
            findings = (root / ".omx" / "research" / "lossless_findings.md").read_text()
            self.assertIn("0.2500", latest)
            self.assertIn("lzma_baseline", focus)
            self.assertIn("lossless promotion flow is separate", findings)

    def test_promote_lossless_result_is_idempotent_and_deduplicates_ledgers(self) -> None:
        from tac.lossless.state import promote_lossless_result

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "lossless_result.json"
            result_payload = {
                "profile": "zpaq_baseline",
                "archive_path": "submission/floor.zip",
                "archive_bytes": 300,
                "original_bytes": 1200,
                "compression_rate": 0.25,
                "method": "zpaq",
            }
            result_path.write_text(json.dumps(result_payload))

            reports = root / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            canonical_results_row = {
                "profile": "zpaq_baseline",
                "method": "zpaq",
                "archive_path": "submission/floor.zip",
                "archive_bytes": 300,
                "original_bytes": 1200,
                "compression_rate": 0.25,
                "event": "promotion",
            }
            canonical_timeline_row = {
                "event": "promotion",
                "profile": "zpaq_baseline",
                "method": "zpaq",
                "compression_rate": 0.25,
                "archive_bytes": 300,
                "original_bytes": 1200,
                "archive_path": "submission/floor.zip",
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


if __name__ == "__main__":
    unittest.main()
