from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

MODULE_PATH = ROOT / "experiments" / "train_tac.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_tac", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainTacEntrypointTests(unittest.TestCase):
    def test_main_forwards_to_canonical_tac_cli_lossy_namespace(self) -> None:
        mod = load_module()

        with mock.patch.object(mod.tac_cli, "main", return_value=0) as mocked:
            result = mod.main(["--tag", "demo"])

        self.assertEqual(result, 0)
        mocked.assert_called_once_with(["lossy", "--tag", "demo"])

    def test_main_defaults_to_sys_argv(self) -> None:
        mod = load_module()

        with mock.patch.object(mod.tac_cli, "main", return_value=0) as mocked:
            with mock.patch.object(sys, "argv", ["train_tac.py", "--tag", "demo"]):
                result = mod.main()

        self.assertEqual(result, 0)
        mocked.assert_called_once_with(["lossy", "--tag", "demo"])


if __name__ == "__main__":
    unittest.main()
