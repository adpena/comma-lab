from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "proxy_score_faithful.py"


def load_module():
    spec = importlib.util.spec_from_file_location("proxy_score_faithful", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProxyScoreFaithfulTests(unittest.TestCase):
    def test_resolve_archive_zip_prefers_live_submission_archive(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            live = root / "submissions" / "robust_current" / "archive.zip"
            legacy = root / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
            live.parent.mkdir(parents=True, exist_ok=True)
            legacy.parent.mkdir(parents=True, exist_ok=True)
            live.write_bytes(b"live")
            legacy.write_bytes(b"legacy")

            resolved = mod.resolve_archive_zip(None, project_root=root)
            self.assertEqual(resolved, live)

    def test_resolve_archive_zip_honors_explicit_path(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            explicit = root / "custom.zip"
            explicit.write_bytes(b"custom")

            resolved = mod.resolve_archive_zip(explicit, project_root=root)
            self.assertEqual(resolved, explicit)

    def test_prepare_submission_dir_copies_archive_and_raw(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "archive.zip"
            raw = root / "inflated.raw"
            archive.write_bytes(b"archive-bytes")
            raw.write_bytes(b"raw-bytes")

            submission_dir = mod.prepare_submission_dir(root / "work", archive, raw)

            self.assertEqual((submission_dir / "archive.zip").read_bytes(), b"archive-bytes")
            self.assertEqual((submission_dir / "inflated" / "0.raw").read_bytes(), b"raw-bytes")


if __name__ == "__main__":
    unittest.main()
