from __future__ import annotations

import importlib.util
import inspect
from types import SimpleNamespace
from pathlib import Path
import subprocess
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_check.py"


def load_module():
    spec = importlib.util.spec_from_file_location("kaggle_check", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class KaggleCheckTests(unittest.TestCase):
    def test_kernelworkerstatus_error_is_error(self) -> None:
        mod = load_module()

        self.assertTrue(mod.is_error_status("KernelWorkerStatus.ERROR"))
        self.assertIn("ERROR", mod.format_status("KernelWorkerStatus.ERROR"))

    def test_kernelworkerstatus_complete_is_not_error(self) -> None:
        mod = load_module()

        self.assertFalse(mod.is_error_status("KernelWorkerStatus.COMPLETE"))
        self.assertIn("COMPLETE", mod.format_status("KernelWorkerStatus.COMPLETE"))

    def test_kernelworkerstatus_cancel_acknowledged_is_error(self) -> None:
        mod = load_module()

        self.assertTrue(mod.is_error_status("KernelWorkerStatus.CANCEL_ACKNOWLEDGED"))

    def test_get_kernel_log_skips_directories_and_reads_nested_files(self) -> None:
        mod = load_module()
        original_run = subprocess.run

        def fake_run(command, capture_output, text, timeout=None):
            out_dir = Path(command[command.index("-p") + 1])
            (out_dir / "upstream").mkdir()
            (out_dir / "nested").mkdir()
            (out_dir / "nested" / "log.txt").write_text("failure line\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        try:
            mod.subprocess.run = fake_run
            lines = mod.get_kernel_log(["kaggle"], "adpena/demo")
        finally:
            mod.subprocess.run = original_run

        self.assertIn("[file: nested/log.txt]", lines)
        self.assertIn("failure line", lines)

    def test_get_kernel_log_timeout_returns_diagnostic(self) -> None:
        mod = load_module()
        original_run = subprocess.run

        def fake_run(command, capture_output, text, timeout=None):
            raise subprocess.TimeoutExpired(command, timeout)

        try:
            mod.subprocess.run = fake_run
            lines = mod.get_kernel_log(["kaggle"], "adpena/demo", timeout_s=1)
        finally:
            mod.subprocess.run = original_run

        self.assertEqual(lines, ["(log fetch timed out after 1s for adpena/demo)"])

    def test_get_kernel_log_ignores_tempdir_cleanup_errors(self) -> None:
        mod = load_module()

        source = inspect.getsource(mod.get_kernel_log)

        self.assertIn("ignore_cleanup_errors=True", source)

    def test_get_kernel_status_timeout_returns_status_marker(self) -> None:
        mod = load_module()
        original_run = subprocess.run

        def fake_run(command, capture_output, text, timeout=None):
            raise subprocess.TimeoutExpired(command, timeout)

        try:
            mod.subprocess.run = fake_run
            status = mod.get_kernel_status(["kaggle"], "adpena/demo", timeout_s=2)
        finally:
            mod.subprocess.run = original_run

        self.assertEqual(status, "STATUS_TIMEOUT_AFTER_2S")
        self.assertIn("TIMEOUT", mod.format_status(status))
        self.assertTrue(mod.is_error_status(status))

    def test_pr101_proxy_sweep_is_in_default_status_watchlist(self) -> None:
        mod = load_module()

        self.assertIn("adpena/pr101-proxy-sweep", mod.KERNEL_SLUGS)
        self.assertIn("adpena/pr101-bias-refine", mod.KERNEL_SLUGS)


if __name__ == "__main__":
    unittest.main()
