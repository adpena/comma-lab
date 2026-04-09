from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "reports" / "graphs" / "refresh_site.py"


class RefreshSiteOrderTests(unittest.TestCase):
    def test_refresh_runs_tests_before_final_site_copy(self) -> None:
        spec = importlib.util.spec_from_file_location("refresh_site", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        commands = module.COMMANDS
        self.assertIn(["python3", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"], commands)
        self.assertIn(["python3", "build_static_site.py"], commands)
        self.assertIn(["python3", "build_static_site.py", "--check"], commands)

        test_idx = commands.index(["python3", "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"])
        build_idx = commands.index(["python3", "build_static_site.py"])
        check_idx = commands.index(["python3", "build_static_site.py", "--check"])

        self.assertLess(test_idx, build_idx)
        self.assertLess(build_idx, check_idx)


if __name__ == "__main__":
    unittest.main()
