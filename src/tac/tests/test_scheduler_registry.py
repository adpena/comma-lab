from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.comma_lab.scheduler.registry import load_platform_registry


def sample_registry_payload() -> dict[str, object]:
    return {
        "version": 1,
        "platforms": [
            {
                "name": "local",
                "kind": "local",
                "result_devices": ["cpu", "mps"],
                "budget": {
                    "max_runs": 3,
                    "max_archive_bytes": 2000000,
                },
            },
            {
                "name": "remote-gpu",
                "kind": "remote",
                "manifest_globs": ["remote-runs/remote-gpu/*/manifest.json"],
                "status_globs": ["remote-runs/remote-gpu/*/status.json"],
                "ledger_paths": ["remote-runs/remote-gpu/_ledger.jsonl"],
                "budget": {
                    "max_active_runs": 1,
                    "max_failed_runs": 2,
                },
            },
            {
                "name": "kaggle",
                "kind": "remote",
                "result_devices": ["kaggle"],
                "manifest_globs": ["remote-runs/kaggle/*/manifest.json"],
                "status_globs": ["remote-runs/kaggle/*/status.json"],
                "ledger_paths": ["remote-runs/kaggle/_ledger.jsonl"],
                "budget": {"max_runs": 4},
            },
            {
                "name": "modal",
                "kind": "remote",
                "result_devices": ["modal"],
                "manifest_globs": ["remote-runs/modal/*/manifest.json"],
                "status_globs": ["remote-runs/modal/*/status.json"],
                "ledger_paths": ["remote-runs/modal/_ledger.jsonl"],
                "budget": {"max_runs": 4},
            },
            {
                "name": "coiled",
                "kind": "remote",
                "result_devices": ["coiled"],
                "manifest_globs": ["remote-runs/coiled/*/manifest.json"],
                "status_globs": ["remote-runs/coiled/*/status.json"],
                "ledger_paths": ["remote-runs/coiled/_ledger.jsonl"],
                "budget": {"max_runs": 8},
            },
        ],
    }


class SchedulerRegistryTests(unittest.TestCase):
    def test_load_platform_registry_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "platforms.json"
            registry_path.write_text(json.dumps(sample_registry_payload(), indent=2))

            registry = load_platform_registry(registry_path)

        self.assertEqual(registry.version, 1)
        self.assertEqual(sorted(registry.platforms), ["remote-gpu", "coiled", "kaggle", "local", "modal"])
        self.assertEqual(registry.platforms["local"].budget.max_runs, 3)
        self.assertEqual(registry.platforms["local"].budget.max_archive_bytes, 2000000)
        self.assertEqual(
            registry.platforms["remote-gpu"].status_globs,
            ("remote-runs/remote-gpu/*/status.json",),
        )
        self.assertEqual(
            registry.platforms["coiled"].ledger_paths,
            ("remote-runs/coiled/_ledger.jsonl",),
        )

    def test_load_platform_registry_supports_json_compatible_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "platforms.yaml"
            payload = sample_registry_payload()
            payload["platforms"] = [
                {
                    "name": "local",
                    "kind": "local",
                    "result_devices": ["cpu"],
                    "budget": {"max_runs": 2},
                }
            ]
            registry_path.write_text(json.dumps(payload))

            registry = load_platform_registry(registry_path)

        self.assertEqual(registry.platforms["local"].budget.max_runs, 2)

    def test_load_platform_registry_rejects_negative_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "platforms.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "platforms": [
                            {
                                "name": "local",
                                "kind": "local",
                                "budget": {"max_runs": -1},
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "max_runs"):
                load_platform_registry(registry_path)

    def test_repo_platform_registry_covers_free_tier_platforms(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        registry = load_platform_registry(repo_root / "configs" / "platforms.json")

        self.assertEqual(registry.version, 1)
        self.assertEqual(
            sorted(registry.platforms),
            ["remote-gpu", "coiled", "kaggle", "local", "modal"],
        )
        self.assertEqual(registry.platforms["local"].result_devices, ("cpu", "mps"))
        self.assertIn(".omx/logs/remote_jobs/remote-gpu-*.json", registry.platforms["remote-gpu"].manifest_globs)
        self.assertIn("remote-runs/remote-gpu/*/manifest.json", registry.platforms["remote-gpu"].manifest_globs)
        self.assertIn(".omx/logs/remote_jobs/kaggle-*.json", registry.platforms["kaggle"].manifest_globs)
        self.assertIn("remote-runs/kaggle/_ledger.jsonl", registry.platforms["kaggle"].ledger_paths)
        self.assertIn(".omx/status/modal-*.json", registry.platforms["modal"].status_globs)
        self.assertIn("remote-runs/modal/*/status.json", registry.platforms["modal"].status_globs)
        self.assertEqual(registry.platforms["coiled"].kind, "remote")


if __name__ == "__main__":
    unittest.main()
