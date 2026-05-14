# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "triage.py"


def load_module():
    spec = importlib.util.spec_from_file_location("proxy_gate_triage", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ProxyGateTriageTests(unittest.TestCase):
    def test_scan_and_rank_best_meta_files(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            promoted = root / "postfilter_promoted_best_meta.json"
            near = root / "postfilter_near_best_meta.json"
            far = root / "postfilter_far_best_meta.json"

            promoted.write_text(json.dumps({"epoch": 10, "scorer": 3.55, "int8_size": 45000}))
            near.write_text(json.dumps({"epoch": 11, "scorer": 3.66, "int8_size": 16000}))
            far.write_text(json.dumps({"epoch": 12, "scorer": 3.95, "int8_size": 16000}))

            records = mod.scan_best_meta(root)
            self.assertEqual([r["slug"] for r in records], ["postfilter_promoted", "postfilter_near", "postfilter_far"])

            report = mod.build_triage_report(records, promoted_slug="postfilter_promoted", proxy_gap_threshold=0.15)
            ranked = report["ranked"]
            self.assertEqual([item["slug"] for item in ranked], ["postfilter_promoted", "postfilter_near", "postfilter_far"])
            self.assertEqual(report["reference"]["slug"], "postfilter_promoted")
            self.assertTrue(ranked[1]["deploy_ready"])
            self.assertTrue(ranked[1]["proxy_ready"])
            self.assertTrue(ranked[2]["deploy_ready"])
            self.assertFalse(ranked[2]["proxy_ready"])

    def test_report_uses_best_scoring_record_as_default_reference(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "postfilter_b_best_meta.json").write_text(json.dumps({"epoch": 8, "scorer": 3.80, "int8_size": 100}))
            (root / "postfilter_a_best_meta.json").write_text(json.dumps({"epoch": 7, "scorer": 3.60, "int8_size": 100}))

            report = mod.build_triage_report(mod.scan_best_meta(root), proxy_gap_threshold=0.10)
            self.assertEqual(report["reference"]["slug"], "postfilter_a")
            self.assertEqual(report["ranked"][0]["slug"], "postfilter_a")
            self.assertAlmostEqual(report["ranked"][1]["delta_vs_reference"], 0.20)

    def test_scan_supports_legacy_score_key(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy = root / "postfilter_legacy_best_meta.json"
            legacy.write_text(json.dumps({"epoch": 5, "score": 4.5, "int8_size": 5125}))

            records = mod.scan_best_meta(root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["slug"], "postfilter_legacy")
            self.assertEqual(records[0]["scorer"], 4.5)

    def test_report_marks_already_proxied_candidates(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            promoted = root / "postfilter_promoted_best_meta.json"
            near = root / "postfilter_near_best_meta.json"
            proxy_log = root / "proxy_near.log"
            int8_path = root / "postfilter_near_best_int8.pt"

            promoted.write_text(json.dumps({"epoch": 10, "scorer": 3.55, "int8_size": 45000}))
            near.write_text(
                json.dumps(
                    {
                        "epoch": 11,
                        "scorer": 3.66,
                        "int8_size": 16000,
                        "int8_path": str(int8_path),
                    }
                )
            )
            proxy_log.write_text('{"weights": "' + str(int8_path) + '"}\n')

            report = mod.build_triage_report(
                mod.scan_best_meta(root),
                promoted_slug="postfilter_promoted",
                proxy_gap_threshold=0.20,
                resolved_proxy_logs=mod.scan_proxy_logs(root),
            )
            near_record = next(item for item in report["ranked"] if item["slug"] == "postfilter_near")
            self.assertTrue(near_record["proxy_already_run"])
            self.assertFalse(near_record["proxy_ready"])

    def test_report_marks_observation_only_variant_mismatch_as_not_deploy_ready(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            promoted = root / "postfilter_promoted_best_meta.json"
            special = root / "postfilter_dilated_h64_long1000_best_meta.json"

            promoted.write_text(json.dumps({"epoch": 10, "scorer": 3.55, "int8_size": 45000}))
            special.write_text(
                json.dumps(
                    {
                        "epoch": 11,
                        "scorer": 3.60,
                        "int8_size": 16000,
                        "meta": {"variant": "saliency_weighted"},
                    }
                )
            )

            report = mod.build_triage_report(
                mod.scan_best_meta(root),
                promoted_slug="postfilter_promoted",
                proxy_gap_threshold=0.20,
            )
            special_record = next(item for item in report["ranked"] if item["slug"] == "postfilter_dilated_h64_long1000")
            self.assertFalse(special_record["deploy_ready"])
            self.assertIn("variant mismatch for special lane slug", special_record["deploy_blockers"][0])
            self.assertFalse(special_record["proxy_ready"])

    def test_report_normalizes_resolved_proxy_paths(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_dir = root / "real"
            alias_dir = root / "alias"
            real_dir.mkdir()
            alias_dir.symlink_to(real_dir, target_is_directory=True)

            int8_real = real_dir / "postfilter_near_best_int8.pt"
            int8_real.write_bytes(b"x")
            promoted = root / "postfilter_promoted_best_meta.json"
            near = root / "postfilter_near_best_meta.json"
            proxy_log = root / "proxy_near.log"

            promoted.write_text(json.dumps({"epoch": 10, "scorer": 3.55, "int8_size": 45000}))
            near.write_text(
                json.dumps(
                    {
                        "epoch": 11,
                        "scorer": 3.66,
                        "int8_size": 16000,
                        "int8_path": str(int8_real),
                    }
                )
            )
            proxy_log.write_text('{"weights": "' + str(alias_dir / "postfilter_near_best_int8.pt") + '"}\n')

            report = mod.build_triage_report(
                mod.scan_best_meta(root),
                promoted_slug="postfilter_promoted",
                proxy_gap_threshold=0.20,
                resolved_proxy_logs=mod.scan_proxy_logs(root),
            )
            near_record = next(item for item in report["ranked"] if item["slug"] == "postfilter_near")
            self.assertTrue(near_record["proxy_already_run"])
            self.assertFalse(near_record["proxy_ready"])


if __name__ == "__main__":
    unittest.main()
