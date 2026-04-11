from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "fleet_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("live_fleet_snapshot", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiveFleetSnapshotTests(unittest.TestCase):
    def test_parse_best_meta_normalizes_slug_and_prefers_known_score_keys(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "postfilter_lane_alpha_best_meta.json"
            meta_path.write_text(json.dumps({"iteration": 12, "score": 3.456, "int8_size": 12345}))

            record = mod.parse_best_meta(meta_path)

        self.assertEqual(record["slug"], "lane_alpha")
        self.assertEqual(record["best"], {"epoch": 12, "scorer": 3.456, "int8_size": 12345})

    def test_parse_training_log_uses_latest_table_row(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "train_lane_alpha.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[lane] header",
                        "epoch      total     scorer         pose          seg  sal_recon         lr",
                        "    1     4.5000     4.4000     0.070000     0.040000     0.0300   0.000200",
                        "   10     4.2500     4.1250     0.060000     0.035000     0.0250   0.000100",
                    ]
                )
            )

            record = mod.parse_training_log(log_path)

        self.assertEqual(record["slug"], "lane_alpha")
        self.assertTrue(record["active"])
        self.assertEqual(
            record["latest"],
            {"epoch": 10, "scorer": 4.125, "pose": 0.06, "seg": 0.035},
        )

    def test_build_snapshot_combines_meta_and_log_only_lanes(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "postfilter_lane_alpha_best_meta.json").write_text(
                json.dumps({"epoch": 7, "scorer": 3.21, "int8_size": 111})
            )
            (root / "train_lane_alpha.log").write_text(
                "\n".join(
                    [
                        "[lane] start",
                        "    3     4.1000     4.0500     0.050000     0.030000     0.0200   0.000200",
                        "    8     3.9000     3.8500     0.040000     0.025000     0.0150   0.000100",
                    ]
                )
            )
            (root / "train_research_beta.log").write_text(
                "\n".join(
                    [
                        "[research] start",
                        "    2     5.1000     5.0000     0.080000     0.050000     0.0400   0.000300",
                    ]
                )
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = mod.main([str(root)])

        self.assertEqual(rc, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["directory"], str(root))
        self.assertEqual([lane["slug"] for lane in payload["lanes"]], ["lane_alpha", "research_beta"])

        alpha = payload["lanes"][0]
        self.assertEqual(alpha["best"], {"epoch": 7, "scorer": 3.21, "int8_size": 111})
        self.assertEqual(alpha["latest"], {"epoch": 8, "scorer": 3.85, "pose": 0.04, "seg": 0.025})
        self.assertTrue(alpha["active"])

        beta = payload["lanes"][1]
        self.assertNotIn("best", beta)
        self.assertEqual(beta["latest"], {"epoch": 2, "scorer": 5.0, "pose": 0.08, "seg": 0.05})
        self.assertTrue(beta["active"])

    def test_build_snapshot_merges_special_lane_log_aliases_into_best_meta_slug(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "postfilter_dilated_h64_long1000_best_meta.json").write_text(
                json.dumps({"epoch": 12, "scorer": 3.81, "int8_size": 45731})
            )
            (root / "train_dilated_h64.log").write_text(
                "\n".join(
                    [
                        "[lane] start",
                        "   20     4.1000     3.9900     0.060000     0.033000     0.0200   0.000200",
                    ]
                )
            )

            snapshot = mod.build_snapshot(root)

        self.assertEqual([lane["slug"] for lane in snapshot["lanes"]], ["dilated_h64_long1000"])
        lane = snapshot["lanes"][0]
        self.assertEqual(lane["best"], {"epoch": 12, "scorer": 3.81, "int8_size": 45731})
        self.assertEqual(lane["latest"], {"epoch": 20, "scorer": 3.99, "pose": 0.06, "seg": 0.033})
        self.assertTrue(lane["active"])


if __name__ == "__main__":
    unittest.main()
