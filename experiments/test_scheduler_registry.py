from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.comma_lab.scheduler.registry import load_platform_registry


class SchedulerRegistryTests(unittest.TestCase):
    def test_load_platform_registry_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "platforms.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "platforms": [
                            {
                                "name": "cpu",
                                "kind": "local",
                                "result_devices": ["cpu"],
                                "budget": {
                                    "max_runs": 3,
                                    "max_archive_bytes": 2000000,
                                },
                            },
                            {
                                "name": "bat00",
                                "kind": "remote",
                                "manifest_globs": ["remote-runs/*/manifest.json"],
                                "status_globs": ["remote-runs/*/status.json"],
                                "budget": {
                                    "max_active_runs": 1,
                                    "max_failed_runs": 2,
                                },
                            },
                        ],
                    },
                    indent=2,
                )
            )

            registry = load_platform_registry(registry_path)

        self.assertEqual(registry.version, 1)
        self.assertEqual(sorted(registry.platforms), ["bat00", "cpu"])
        self.assertEqual(registry.platforms["cpu"].budget.max_runs, 3)
        self.assertEqual(registry.platforms["cpu"].budget.max_archive_bytes, 2000000)
        self.assertEqual(registry.platforms["bat00"].status_globs, ("remote-runs/*/status.json",))

    def test_load_platform_registry_supports_json_compatible_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "platforms.yaml"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "platforms": [
                            {
                                "name": "cpu",
                                "kind": "local",
                                "result_devices": ["cpu"],
                                "budget": {"max_runs": 2},
                            }
                        ],
                    }
                )
            )

            registry = load_platform_registry(registry_path)

        self.assertEqual(registry.platforms["cpu"].budget.max_runs, 2)

    def test_load_platform_registry_rejects_negative_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "platforms.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "platforms": [
                            {
                                "name": "cpu",
                                "kind": "local",
                                "budget": {"max_runs": -1},
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "max_runs"):
                load_platform_registry(registry_path)


if __name__ == "__main__":
    unittest.main()
