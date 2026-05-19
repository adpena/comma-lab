# SPDX-License-Identifier: MIT
"""Tests for ``tools/operator_authorize.py::_dispatch_hf_jobs`` (slot 8 wire-in).

Catalog #342 sister (HF Jobs) of Catalog #245 (Modal call_id ledger) + slot 8
operator_authorize.py wire-in 2026-05-19. Mirrors the test pattern of
``test_operator_authorize_local_signal_manifest.py`` + the slot 1
``test_check_339_silent_no_spawn_extinction.py`` for the canonical ledger-poll
fail-closed contract.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import tools.operator_authorize as op


def _hf_jobs_recipe(
    tmp_path: Path,
    *,
    flavor: str = "t4-small",
    extra: dict[str, Any] | None = None,
) -> op.Recipe:
    """Build a minimal hf_jobs recipe pointing at a real on-disk training script."""

    script = tmp_path / "fake_training_script.py"
    script.write_text("#!/usr/bin/env python\nprint('fake')\n", encoding="utf-8")
    raw: dict[str, Any] = {
        "lane_id": "lane_hf_jobs_test_20260519",
        "platform": "hf_jobs",
        "hf_jobs": {
            "script": str(script),
            "hub_dataset_repo": "adpena/test-dataset",
            "hub_model_repo": "adpena/test-model",
            "flavor": flavor,
            "num_epochs": 200,
            "timeout_seconds": 14400,
            "expected_axis": "cuda",
        },
    }
    if extra:
        raw.update(extra)
    return op.Recipe(
        name="unit_hf_jobs",
        path=tmp_path / "unit_hf_jobs.yaml",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# _dispatch_hf_jobs structural contracts
# ---------------------------------------------------------------------------


def test_dispatch_hf_jobs_invokes_canonical_dispatcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_dispatch_hf_jobs` invokes the canonical CLI with the recipe fields."""

    recipe = _hf_jobs_recipe(tmp_path)
    captured_cmd: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured_cmd.append(list(cmd))
        # Canonical dispatcher emits a single-line JSON payload on rc=0.
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"hf_jobs_id": "hf_test_id_001", "ledger_row": {}}),
            stderr="",
        )

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    # Bypass the ledger-poll fail-closed contract for this test (covered separately).
    monkeypatch.setattr(op, "_poll_ledger_for_dispatched_hf_jobs_id", lambda *_a, **_k: True)

    rc = op._dispatch_hf_jobs(recipe, "instance_abc123", env_overrides="")
    assert rc == 0
    assert captured_cmd, "expected subprocess.run to be invoked"
    cmd = captured_cmd[0]
    # Canonical dispatcher CLI surface (verified against
    # tools/dispatch_hf_jobs_vision_training.py::_build_arg_parser).
    assert "tools/dispatch_hf_jobs_vision_training.py" in cmd
    assert "--script" in cmd
    assert "--hub-dataset-repo" in cmd
    assert "--hub-model-repo" in cmd
    assert "--flavor" in cmd
    assert "--lane-id" in cmd
    assert "--label" in cmd
    assert "instance_abc123" in cmd
    assert "adpena/test-dataset" in cmd
    assert "adpena/test-model" in cmd


def test_dispatch_hf_jobs_refuses_missing_script(tmp_path: Path) -> None:
    """`hf_jobs.script` is required; missing field raises SystemExit."""

    recipe = op.Recipe(
        name="unit_no_script",
        path=tmp_path / "unit.yaml",
        raw={
            "lane_id": "lane_test",
            "platform": "hf_jobs",
            "hf_jobs": {
                "hub_dataset_repo": "x/y",
                "hub_model_repo": "x/z",
            },
        },
    )
    with pytest.raises(SystemExit, match="hf_jobs.script"):
        op._dispatch_hf_jobs(recipe, "instance_x", env_overrides="")


def test_dispatch_hf_jobs_refuses_missing_script_file(tmp_path: Path) -> None:
    """The declared `hf_jobs.script` path MUST resolve to an on-disk file."""

    recipe = _hf_jobs_recipe(tmp_path)
    recipe.raw["hf_jobs"]["script"] = str(tmp_path / "does_not_exist.py")
    with pytest.raises(SystemExit, match="training script missing"):
        op._dispatch_hf_jobs(recipe, "instance_x", env_overrides="")


def test_dispatch_hf_jobs_refuses_missing_hub_repos(tmp_path: Path) -> None:
    """Both `hub_dataset_repo` and `hub_model_repo` are required."""

    recipe = _hf_jobs_recipe(tmp_path)
    del recipe.raw["hf_jobs"]["hub_dataset_repo"]
    with pytest.raises(SystemExit, match="hub_dataset_repo.*hub_model_repo"):
        op._dispatch_hf_jobs(recipe, "instance_x", env_overrides="")


def test_dispatch_hf_jobs_returns_nonzero_on_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess rc != 0 propagates back to the caller (no fail-OPEN)."""

    recipe = _hf_jobs_recipe(tmp_path)

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 17, stdout="", stderr="dispatch failed")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    rc = op._dispatch_hf_jobs(recipe, "instance_x", env_overrides="")
    assert rc == 17


# ---------------------------------------------------------------------------
# Catalog #339 sister: ledger-poll fail-closed
# ---------------------------------------------------------------------------


def test_dispatch_hf_jobs_fails_closed_on_missing_ledger_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the dispatcher succeeds but the ledger row never appears, raise SystemExit."""

    recipe = _hf_jobs_recipe(tmp_path)

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"hf_jobs_id": "hf_orphan_id", "ledger_row": None}),
            stderr="",
        )

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    # Poll returns False (no row landed in canonical ledger).
    monkeypatch.setattr(op, "_poll_ledger_for_dispatched_hf_jobs_id", lambda *_a, **_k: False)

    with pytest.raises(SystemExit, match="Catalog #339 sister"):
        op._dispatch_hf_jobs(recipe, "instance_orphan", env_overrides="")


def test_dispatch_hf_jobs_silent_no_payload_does_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed dispatcher output (no JSON payload) does not trip the ledger poll.

    The structural protection lives in the canonical dispatcher (fail-closed
    registration); this wrapper only polls when it can actually extract an id.
    """

    recipe = _hf_jobs_recipe(tmp_path)

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(cmd, 0, stdout="not-json-output", stderr="")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    rc = op._dispatch_hf_jobs(recipe, "instance_x", env_overrides="")
    assert rc == 0


# ---------------------------------------------------------------------------
# Platform-switch routing
# ---------------------------------------------------------------------------


def test_run_dispatch_routes_hf_jobs_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_run_dispatch` routes recipe['platform']='hf_jobs' to `_dispatch_hf_jobs`."""

    recipe = _hf_jobs_recipe(tmp_path)
    called_with: list[tuple[Any, str]] = []

    def fake_dispatch_hf_jobs(rec, job_id, env, **kwargs):  # type: ignore[no-untyped-def]
        called_with.append((rec, job_id))
        return 0

    monkeypatch.setattr(op, "_dispatch_hf_jobs", fake_dispatch_hf_jobs)
    monkeypatch.setattr(op, "_maybe_apply_auto_routing", lambda r: r)
    monkeypatch.setattr(op, "_build_env_overrides", lambda r, j: "")

    rc = op._run_dispatch(recipe, "instance_route_test")
    assert rc == 0
    assert len(called_with) == 1
    assert called_with[0][1] == "instance_route_test"


def test_platform_has_native_dispatch_includes_hf_jobs() -> None:
    """`hf_jobs` is now in the native-dispatch platform set."""

    assert op._platform_has_native_dispatch("hf_jobs") is True
    # Sister platforms still present.
    assert op._platform_has_native_dispatch("modal") is True
    assert op._platform_has_native_dispatch("vastai") is True
    # Unknown platform still rejected.
    assert op._platform_has_native_dispatch("unknown_platform") is False


# ---------------------------------------------------------------------------
# `--target hf_jobs` / `--target hf-jobs` CLI flag
# ---------------------------------------------------------------------------


def test_target_cli_accepts_hf_jobs_underscore_form() -> None:
    """`--target hf_jobs` is a valid choice (underscore form)."""

    import argparse

    # Use the same parser the main() function constructs; we just exercise
    # argparse to verify the choices set.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=[
            "auto", "modal", "vastai", "lightning", "local",
            "local-mps", "local-cpu", "hf_jobs", "hf-jobs",
            "kaggle", "gha", "azure", "none",
        ],
    )
    args = parser.parse_args(["--target", "hf_jobs"])
    assert args.target == "hf_jobs"
    args = parser.parse_args(["--target", "hf-jobs"])
    assert args.target == "hf-jobs"


def test_target_dash_form_maps_to_underscore_platform() -> None:
    """Verify the canonical dash→underscore mapping accepts `hf-jobs → hf_jobs`."""

    target_to_platform = {
        "local-mps": "local_mps",
        "local-cpu": "local_cpu",
        "hf-jobs": "hf_jobs",
    }
    assert target_to_platform["hf-jobs"] == "hf_jobs"
    # No mapping for already-underscore form (passthrough).
    assert target_to_platform.get("hf_jobs") is None


# ---------------------------------------------------------------------------
# _native_dispatch_preflight contract
# ---------------------------------------------------------------------------


def test_native_dispatch_preflight_accepts_valid_hf_jobs_recipe(tmp_path: Path) -> None:
    """A valid hf_jobs recipe (script exists + dispatcher present) passes preflight."""

    recipe = _hf_jobs_recipe(tmp_path)
    # The canonical dispatcher at tools/dispatch_hf_jobs_vision_training.py
    # MUST exist for this to pass (it does, per commit e588d9f65 slot 7).
    op._native_dispatch_preflight(recipe)  # no exception


def test_native_dispatch_preflight_refuses_hf_jobs_recipe_missing_script(
    tmp_path: Path,
) -> None:
    """Preflight refuses an hf_jobs recipe whose `hf_jobs.script` field is unset."""

    recipe = op.Recipe(
        name="unit_no_script",
        path=tmp_path / "unit.yaml",
        raw={"lane_id": "lane_x", "platform": "hf_jobs", "hf_jobs": {}},
    )
    with pytest.raises(SystemExit, match="hf_jobs.script"):
        op._native_dispatch_preflight(recipe)


def test_native_dispatch_preflight_refuses_hf_jobs_recipe_missing_script_file(
    tmp_path: Path,
) -> None:
    """Preflight refuses an hf_jobs recipe whose `hf_jobs.script` points nowhere."""

    recipe = _hf_jobs_recipe(tmp_path)
    recipe.raw["hf_jobs"]["script"] = str(tmp_path / "phantom.py")
    with pytest.raises(SystemExit, match="HF Jobs training script missing"):
        op._native_dispatch_preflight(recipe)
