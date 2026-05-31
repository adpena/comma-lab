# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from tools import build_z8_full_video_vjp_surface_bundle as cli


def _archive_stub(tmp_path: Path) -> Path:
    path = tmp_path / "0.bin"
    path.write_bytes(b"z8-stub")
    return path


def test_z8_vjp_cli_refuses_local_output_without_explicit_opt_in(tmp_path: Path) -> None:
    local_output = tmp_path / "repo-local-vjp-output"
    args = cli.parse_args(
        [
            "--archive-bin",
            str(_archive_stub(tmp_path)),
            "--output-dir",
            str(local_output),
        ]
    )

    with pytest.raises(StorageTierError, match="local_output_dir_requires_explicit_opt_in"):
        cli._resolve_output_dir(args)

    assert not local_output.exists()


def test_z8_vjp_cli_selects_storage_tier_when_output_dir_omitted(tmp_path: Path) -> None:
    tier = tmp_path / "external-tier"
    tier.mkdir()
    args = cli.parse_args(
        [
            "--archive-bin",
            str(_archive_stub(tmp_path)),
            "--storage-tier",
            f"fast={tier}",
            "--storage-reserve-free-gb",
            "0",
            "--storage-workload-subdir",
            "experiments/results/z8-vjp-test",
            "--allow-local-output-dir",
        ]
    )

    output_dir, storage_plan = cli._resolve_output_dir(args)
    payload = json.loads(storage_plan.read_text(encoding="utf-8"))

    assert output_dir == tier / "experiments/results/z8-vjp-test"
    assert output_dir.is_dir()
    assert payload["schema"] == cli.Z8_VJP_STORAGE_PLAN_SCHEMA
    assert payload["resolved_output_dir"] == output_dir.as_posix()
    assert payload["output_dir_was_explicit"] is False
    assert payload["local_output_explicitly_allowed"] is True
    assert payload["replay_provenance"]["schema"] == cli.Z8_VJP_REPLAY_PROVENANCE_SCHEMA
    assert payload["replay_provenance"]["archive"]["bytes"] == len(b"z8-stub")
    assert len(payload["replay_provenance"]["archive"]["sha256"]) == 64
    assert "--storage-tier" in payload["replay_provenance"]["argv"]
    assert payload["storage_plan"]["selected_tier"] == "fast"
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_z8_vjp_cli_allows_local_output_only_with_opt_in(tmp_path: Path) -> None:
    local_output = tmp_path / "explicit-local-vjp-output"
    storage_plan = tmp_path / "storage_plan.json"
    args = cli.parse_args(
        [
            "--archive-bin",
            str(_archive_stub(tmp_path)),
            "--output-dir",
            str(local_output),
            "--allow-local-output-dir",
            "--storage-plan-out",
            str(storage_plan),
        ]
    )

    output_dir, got_storage_plan = cli._resolve_output_dir(args)
    payload = json.loads(got_storage_plan.read_text(encoding="utf-8"))

    assert output_dir == local_output
    assert got_storage_plan == storage_plan
    assert local_output.is_dir()
    assert payload["output_dir_was_explicit"] is True
    assert payload["local_output_explicitly_allowed"] is True
    assert payload["storage_plan"]["selected_tier"] == "explicit_output_dir"
    assert payload["replay_provenance"]["env_allowlist_keys"] == list(cli.REPLAY_ENV_ALLOWLIST)
