from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tac.deploy.kaggle.kaggle_queue_tick as mod


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


class KaggleQueueTickTests(unittest.TestCase):
    def test_selects_quota_blocked_manifest_when_slot_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root / "a.json"
            b = root / "b.json"
            write_json(a, {"slug": "run-a", "status_path": str(root / "status-a.json"), "remote_command": "push-a"})
            write_json(b, {"slug": "run-b", "status_path": str(root / "status-b.json"), "remote_command": "push-b"})
            write_json(root / "status-a.json", {"status": "running"})
            write_json(root / "status-b.json", {"status": "paused", "phase": "quota_blocked"})

            selected = mod.select_manifest_for_repush([a, b], max_active=2)

        self.assertEqual(selected, b)

    def test_skips_repush_when_slots_are_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root / "a.json"
            b = root / "b.json"
            c = root / "c.json"
            write_json(a, {"slug": "run-a", "status_path": str(root / "status-a.json"), "remote_command": "push-a"})
            write_json(b, {"slug": "run-b", "status_path": str(root / "status-b.json"), "remote_command": "push-b"})
            write_json(c, {"slug": "run-c", "status_path": str(root / "status-c.json"), "remote_command": "push-c"})
            write_json(root / "status-a.json", {"status": "running"})
            write_json(root / "status-b.json", {"status": "running"})
            write_json(root / "status-c.json", {"status": "paused", "phase": "quota_blocked"})

            selected = mod.select_manifest_for_repush([a, b, c], max_active=2)

        self.assertIsNone(selected)

    def test_local_repush_submitted_does_not_count_as_remote_active_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root / "a.json"
            b = root / "b.json"
            write_json(a, {"slug": "run-a", "status_path": str(root / "status-a.json"), "remote_command": "push-a"})
            write_json(b, {"slug": "run-b", "status_path": str(root / "status-b.json"), "remote_command": "push-b"})
            write_json(root / "status-a.json", {"status": "queued", "phase": "repush_submitted"})
            write_json(root / "status-b.json", {"status": "paused", "phase": "kernel_cancel_acknowledged"})

            selected = mod.select_manifest_for_repush([a, b], max_active=1)

        self.assertEqual(selected, b)

    def test_run_repush_command_uses_shell_string(self) -> None:
        calls: list[str] = []

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def fake_runner(command: str) -> object:
            calls.append(command)
            return Result()

        rc = mod.run_repush_command("uv run --with kaggle kaggle kernels push -p /tmp/demo", command_runner=fake_runner)

        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["uv run --with kaggle kaggle kernels push -p /tmp/demo"])

    def test_run_repush_command_treats_kernel_push_error_as_failure(self) -> None:
        class Result:
            returncode = 0
            stdout = "Kernel push error: Notebook not found"
            stderr = ""

        rc = mod.run_repush_command(
            "uv run --with kaggle kaggle kernels push -p /tmp/demo",
            command_runner=lambda _command: Result(),
        )

        self.assertNotEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
