from __future__ import annotations

from pathlib import Path

from tac.optimization.ddm_ms4d_direct_completion import (
    _checkpoint_identity_matches,
    _raw_custody,
)

REPO = Path(__file__).resolve().parents[2]


def test_repository_artifact_custody_is_portable() -> None:
    result = _raw_custody(Path(__file__), repository_root=REPO)
    assert result["path"] == "tools/tests/test_ddm_ms4d_direct_completion.py"
    assert result["bytes"] > 0
    assert len(result["sha256"]) == 64


def test_path_bound_checkpoint_identity_normalizes_without_relaxing_hashes() -> None:
    current = {
        "run_id": "run",
        "config_path": ".omx/research/configs/direct.json",
        "config_sha256": "a" * 64,
        "source_config_sha256": "b" * 64,
    }
    legacy = {
        **current,
        "config_path": (
            "/transient/isolate/.omx/research/configs/direct.json"
        ),
    }
    assert _checkpoint_identity_matches(current, current)
    assert _checkpoint_identity_matches(legacy, current)
    assert not _checkpoint_identity_matches(
        {**legacy, "config_sha256": "c" * 64},
        current,
    )
    assert not _checkpoint_identity_matches(
        {**legacy, "config_path": "/different/config/direct.json"},
        current,
    )
