from pathlib import Path

import pytest

from experiments.ddm_bs4_born_small_stage0_preflight import (
    BS4PreflightError,
    additive_checkpoint_path,
    atomic_json_once,
)


def test_additive_checkpoint_versions_preserve_append_only_race_guard(
    tmp_path: Path,
) -> None:
    base = tmp_path / "stage_00_source_preflight.json"
    original = atomic_json_once(base, {"run": 1})
    original_payload = base.read_bytes()

    second_value = {"run": 2}
    second = additive_checkpoint_path(base, second_value)
    assert second.name == "stage_00_source_preflight_r2.json"
    atomic_json_once(second, second_value)
    assert additive_checkpoint_path(base, second_value) == second
    assert base.read_bytes() == original_payload
    assert original["sha256"] != atomic_json_once(second, second_value)["sha256"]

    raced = additive_checkpoint_path(base, {"run": 3})
    assert raced.name == "stage_00_source_preflight_r3.json"
    atomic_json_once(raced, {"competing_run": True})
    with pytest.raises(BS4PreflightError, match="refusing to replace different checkpoint"):
        atomic_json_once(raced, {"run": 3})
