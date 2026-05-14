# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tac.hidden_gems import (
    CATEGORIES,
    SCHEMA_VERSION,
    STATUSES,
    all_hidden_gems,
    hidden_gem_to_dict,
    registry_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "tools" / "list_hidden_gems.py"

EXPECTED_KEYS = [
    "charged_mask_grammar_atoms",
    "component_sensitivity_byte_allocator",
    "coolchic_c3_renderer_trend_gate",
    "engineered_correction_atom_gate",
    "fridrich_inverse_steg_allocator",
    "hnerv_payload_scorecard_followups",
    "joint_stack_contract_manifest",
    "latent_sidecar_arithmetic_terminal",
    "nerv_mask_l2_readiness",
    "omega_w_v3_real_sensitivity_gate",
    "pr106_sidechannel_stack_gate",
    "pr91_hpm1_runtime_contract",
    "pr95_residual_atom_planner",
    "raft_radial_pose_readiness",
    "wavelet_residual_basis_gate",
]

EXPECTED_SCHEMA_FIELDS = {
    "category",
    "contest_compliance_notes",
    "evidence_paths",
    "integration_targets",
    "key",
    "next_patch",
    "status",
    "summary",
    "title",
}

FORBIDDEN_PATH_PREFIXES = (
    ".omx/state/",
    "experiments/results/",
    "reports/private/",
    "reports/raw/",
    "reverse_engineering/",
)

FORBIDDEN_PATH_PARTS = {
    ".env",
    "credential",
    "credentials",
    "id_rsa",
    "secret",
    "secrets",
    "ssh",
    "token",
    "tokens",
}

CONCRETE_SCORE_CLAIM_PATTERNS = (
    re.compile(r"\bscore_recomputed_from_components\b", re.IGNORECASE),
    re.compile(r"\bfinal_score\b", re.IGNORECASE),
    re.compile(r"\bavg_posenet_dist\b", re.IGNORECASE),
    re.compile(r"\bavg_segnet_dist\b", re.IGNORECASE),
    re.compile(r"\bleaderboard\s+rank\b", re.IGNORECASE),
    re.compile(r"\brank\s*#?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bscore\s*[=:]\s*[-+]?\d", re.IGNORECASE),
)


def test_hidden_gem_registry_schema_static_safety() -> None:
    entries = all_hidden_gems()
    assert entries

    for entry in entries:
        row = hidden_gem_to_dict(entry)
        assert set(row) == EXPECTED_SCHEMA_FIELDS
        assert row["category"] in CATEGORIES
        assert row["status"] in STATUSES
        assert row["evidence_paths"]
        assert row["integration_targets"]
        assert row["next_patch"]
        assert row["contest_compliance_notes"]

        for path in [*entry.evidence_paths, *entry.integration_targets]:
            assert not path.startswith(("/", "~"))
            assert "\\" not in path
            assert ".." not in Path(path).parts
            assert not any(path.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES)
            assert not ({part.lower() for part in Path(path).parts} & FORBIDDEN_PATH_PARTS)

        text = "\n".join(str(value) for value in row.values())
        for pattern in CONCRETE_SCORE_CLAIM_PATTERNS:
            assert not pattern.search(text), (entry.key, pattern.pattern)


def test_hidden_gem_registry_ordering_is_stable() -> None:
    assert [entry.key for entry in all_hidden_gems()] == EXPECTED_KEYS
    payload = registry_payload()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["entry_count"] == len(EXPECTED_KEYS)
    assert [entry["key"] for entry in payload["entries"]] == EXPECTED_KEYS


def test_list_hidden_gems_cli_json_output() -> None:
    proc = _run_cli("--format", "json")

    assert proc.stderr == ""
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["registry"] == "hidden_gems"
    assert [entry["key"] for entry in payload["entries"]] == EXPECTED_KEYS


def test_list_hidden_gems_cli_markdown_filter() -> None:
    proc = _run_cli("--format", "markdown", "--category", "geometry_pose")

    assert proc.stderr == ""
    assert proc.stdout.startswith("# Hidden-Gem Registry\n")
    assert "| `raft_radial_pose_readiness` | `geometry_pose` | `prototype` |" in proc.stdout
    assert "component_sensitivity_byte_allocator" not in proc.stdout


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
