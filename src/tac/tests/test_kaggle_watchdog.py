# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tac.deploy.kaggle.kaggle_watchdog as mod


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


class KaggleWatchdogTests(unittest.TestCase):
    def test_run_once_syncs_statuses_and_returns_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifests_dir = root / "manifests"
            status_dir = root / "status"
            log_dir = root / "logs"
            manifest_path = manifests_dir / "kaggle-demo.json"
            write_json(
                manifest_path,
                {
                    "slug": "kaggle-demo",
                    "run_id": "kaggle-demo-v1",
                    "kernel_ref": "alice/demo",
                    "status_path": str(status_dir / "kaggle-demo.json"),
                    "remote_command": "push-demo",
                },
            )

            payload = mod.run_once(
                manifests_dir=manifests_dir,
                log_dir=log_dir,
                max_active=2,
                status_sync=lambda manifest_path: {"slug": "kaggle-demo", "status": "running"},
                queue_tick=lambda manifests_dir, max_active: {"action": "noop"},
            )

            cycle_log = json.loads((log_dir / "latest.json").read_text())

        self.assertEqual(payload["synced"][0]["status"], "running")
        self.assertEqual(payload["queue"]["action"], "noop")
        self.assertEqual(cycle_log["synced"][0]["slug"], "kaggle-demo")


if __name__ == "__main__":
    unittest.main()
