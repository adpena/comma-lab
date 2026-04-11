from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "remote_job.py"


def load_module():
    spec = importlib.util.spec_from_file_location("remote_job", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RemoteJobTests(unittest.TestCase):
    def test_render_remote_launch_script_includes_log_and_command(self) -> None:
        mod = load_module()
        script = mod.render_remote_launch_script(
            remote_root="/home/adpena/pact-side",
            remote_log="experiments/postfilter_weights/train.log",
            remote_command="python train.py --epochs 1000",
        )
        self.assertIn("cd /home/adpena/pact-side", script)
        self.assertIn("train.log", script)
        self.assertIn("python train.py --epochs 1000", script)
        self.assertIn("nohup", script)

    def test_write_manifest_records_launch_metadata(self) -> None:
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "job.json"
            mod.write_manifest(
                manifest_path=manifest_path,
                data={
                    "slug": "long1000_qat_ema_alpha20_h16",
                    "host": "bat00",
                    "remote_pid": 406,
                },
            )
            payload = json.loads(manifest_path.read_text())

        self.assertEqual(payload["slug"], "long1000_qat_ema_alpha20_h16")
        self.assertEqual(payload["host"], "bat00")
        self.assertEqual(payload["remote_pid"], 406)
        self.assertIn("written_at", payload)


if __name__ == "__main__":
    unittest.main()
