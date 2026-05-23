# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "lane_maturity.py"
GATES = [
    "impl_complete",
    "real_archive_empirical",
    "contest_cuda",
    "contest_cpu",
    "strict_preflight",
    "three_clean_review",
    "memory_entry",
    "deploy_runbook",
]


def _write_registry(root: Path) -> None:
    state_dir = root / ".omx" / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "lane_registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at": "2026-05-23T00:00:00Z",
                "generated_at": "2026-05-23T00:00:00Z",
                "from_state_hash": "test",
                "description": "test registry",
                "gate_definitions": {},
                "level_rules": {},
                "lanes": [
                    {
                        "id": "lane_test",
                        "name": "Test lane",
                        "phase": 0.0,
                        "level": 0,
                        "gates": {
                            gate: {"status": False, "evidence": ""} for gate in GATES
                        },
                        "notes": "",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _mutation_script() -> str:
    return textwrap.dedent(
        f"""
        import importlib.util
        import json
        import sys
        import time
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("lane_maturity", {str(TOOL_PATH)!r})
        lm = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(lm)

        root = Path(sys.argv[1])
        gate = sys.argv[2]
        evidence = sys.argv[3]
        delay = float(sys.argv[4])
        with lm._mutation_lock(root):
            data = lm.load_registry(root)
            before = json.dumps(data["lanes"][0], sort_keys=True)
            lane = lm.mark_gate(data, "lane_test", gate, evidence, repo_root=root)
            time.sleep(delay)
            lm.save_registry(data, root)
            lm.append_audit_log(
                {{
                    "timestamp": lm._now_iso(),
                    "command": "mark",
                    "args": {{"lane_id": "lane_test", "gate": gate, "evidence": evidence}},
                    "before_state": before,
                    "after_state": json.dumps(lane, sort_keys=True),
                }},
                root,
            )
        """
    )


def test_lane_maturity_mutations_are_serialized_across_processes(tmp_path: Path) -> None:
    _write_registry(tmp_path)
    script = _mutation_script()

    first = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), "impl_complete", "impl evidence", "0.25"],
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.05)
    second = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            "strict_preflight",
            "preflight evidence",
            "0.0",
        ],
        stderr=subprocess.PIPE,
        text=True,
    )
    _, first_stderr = first.communicate(timeout=5)
    _, second_stderr = second.communicate(timeout=5)

    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second_stderr
    registry = json.loads((tmp_path / ".omx/state/lane_registry.json").read_text())
    gates = registry["lanes"][0]["gates"]
    assert gates["impl_complete"] == {"status": True, "evidence": "impl evidence"}
    assert gates["strict_preflight"] == {
        "status": True,
        "evidence": "preflight evidence",
    }
    assert registry["lanes"][0]["level"] == 1

    audit_rows = (tmp_path / ".omx/state/lane_maturity_audit.log").read_text().splitlines()
    assert len(audit_rows) == 2
