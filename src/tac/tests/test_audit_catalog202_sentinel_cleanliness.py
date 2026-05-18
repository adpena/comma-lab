# SPDX-License-Identifier: MIT
"""Tests for the Catalog #202 sentinel-cleanliness audit helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_catalog202_sentinel_cleanliness as audit  # noqa: E402


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_repo(tmp_path: Path) -> Path:
    for rel in (
        "experiments/modal_train_lane.py",
        "tools/operator_authorize.py",
        "tools/run_modal_smoke_before_full.py",
        "src/tac/deploy/modal/mount_manifest.py",
        "scripts/remote_lane_example.sh",
        "experiments/train_example.py",
        "src/tac/substrates/example/archive.py",
    ):
        _write(tmp_path / rel)
    recipe = tmp_path / ".omx/operator_authorize_recipes/example_recipe.yaml"
    _write(
        recipe,
        """
schema_version: 1
name: example_recipe
lane_id: lane_example
platform: modal
remote_driver: scripts/remote_lane_example.sh
modal:
  lane_script: scripts/remote_lane_example.sh
  cost_band_trainer: experiments/train_example.py
required_input_files_trainer: experiments/train_example.py
sentinel_files:
  - src/tac/substrates/example/archive.py
""".lstrip(),
    )
    return tmp_path


def test_build_audit_clean_sentinels_allows_catalog202_attestation(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _minimal_repo(tmp_path)
    monkeypatch.setattr(
        audit,
        "git_status_paths",
        lambda repo_root: {"docs/operator_note.md": " M"},
    )

    payload = audit.build_audit("example_recipe", repo_root=repo)

    assert payload["sentinel_set_clean_for_catalog202"] is True
    assert payload["ready_for_catalog202_paired_env_attestation"] is True
    assert payload["sentinel_set_snapshot_stable_for_catalog202"] is True
    assert (
        payload["ready_for_catalog202_audit_backed_dirty_sentinel_attestation"]
        is True
    )
    assert payload["audit_backed_attestation_blockers"] == []
    assert payload["attestation_blockers"] == []
    assert payload["dirty_worktree_path_count"] == 1
    assert payload["dirty_sentinel_path_count"] == 0
    assert payload["provider_dispatch_attempted"] is False
    assert payload["lane_claim_opened"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["catalog202_env_contract"]["intent_env_var"] == (
        "OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK"
    )
    assert payload["catalog202_env_contract"]["audit_json_env_var"] == (
        "OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_AUDIT_JSON"
    )
    assert payload["catalog202_env_contract"]["attestation_value_hint"].startswith(
        "catalog202_sentinel_audit:"
    )


def test_build_audit_dirty_sentinel_blocks_catalog202_attestation(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _minimal_repo(tmp_path)
    monkeypatch.setattr(
        audit,
        "git_status_paths",
        lambda repo_root: {
            "tools/run_modal_smoke_before_full.py": " M",
            "reports/latest.md": " M",
        },
    )

    payload = audit.build_audit("example_recipe", repo_root=repo)

    assert payload["sentinel_set_clean_for_catalog202"] is False
    assert payload["ready_for_catalog202_paired_env_attestation"] is False
    assert payload["sentinel_set_snapshot_stable_for_catalog202"] is True
    assert (
        payload["ready_for_catalog202_audit_backed_dirty_sentinel_attestation"]
        is True
    )
    assert payload["audit_backed_attestation_blockers"] == []
    assert payload["attestation_blockers"] == [
        "catalog202_sentinel_files_dirty_in_git"
    ]
    assert payload["dirty_sentinel_paths"] == [
        "tools/run_modal_smoke_before_full.py"
    ]
    assert payload["dirty_mounted_non_sentinel_paths"] == []
    assert payload["dirty_operator_side_paths"] == ["reports/latest.md"]


def test_effective_sentinels_flattens_list_values_and_preserves_dot_paths(
    tmp_path: Path,
) -> None:
    repo = _minimal_repo(tmp_path)
    _write(repo / ".omx/research/operator_side.md")
    recipe = {
        "required_input_files_trainer": [
            "experiments/train_example.py",
            "./src/tac/substrates/example/archive.py",
        ],
        "sentinel_files": [
            ".omx/research/operator_side.md",
        ],
    }

    effective, missing, outside_mount = audit.effective_sentinel_files(
        recipe, repo_root=repo
    )

    assert "experiments/train_example.py" in effective
    assert "src/tac/substrates/example/archive.py" in effective
    assert missing == []
    assert outside_mount == [".omx/research/operator_side.md"]


def test_operator_authorize_modal_sentinels_flattens_list_values(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _minimal_repo(tmp_path)
    _write(repo / ".omx/research/operator_side.md")
    import operator_authorize as oa  # noqa: E402

    monkeypatch.setattr(oa, "REPO_ROOT", repo)
    recipe = oa.Recipe(
        name="example_recipe",
        path=repo / ".omx/operator_authorize_recipes/example_recipe.yaml",
        raw={
            "required_input_files_trainer": [
                "experiments/train_example.py",
                "./src/tac/substrates/example/archive.py",
            ],
            "sentinel_files": [
                ".omx/research/operator_side.md",
            ],
        },
    )

    sentinels = oa._modal_sentinel_files(recipe).split(",")

    assert "experiments/train_example.py" in sentinels
    assert "src/tac/substrates/example/archive.py" in sentinels
    assert all("[" not in path and "]" not in path for path in sentinels)
    assert ".omx/research/operator_side.md" not in sentinels


def test_write_artifact_preserves_false_authority_flags(tmp_path: Path, monkeypatch):
    repo = _minimal_repo(tmp_path)
    monkeypatch.setattr(audit, "git_status_paths", lambda repo_root: {})
    payload = audit.build_audit("example_recipe", repo_root=repo)
    path = audit.write_artifact(payload, repo_root=repo)

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert path.name.startswith("example_recipe_")
    assert persisted["score_claim"] is False
    assert persisted["promotion_eligible"] is False
    assert persisted["provider_dispatch_attempted"] is False
    assert persisted["lane_claim_opened"] is False


def test_candidate4c_effective_sentinels_match_operator_authorize():
    """The audit helper must stay in sync with the actual dispatch wrapper."""

    from operator_authorize import _load_recipe, _modal_sentinel_files  # noqa: E402

    recipe_name = "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch"
    recipe = _load_recipe(recipe_name)
    expected = [p for p in _modal_sentinel_files(recipe).split(",") if p]
    actual, missing, outside_mount = audit.effective_sentinel_files(
        recipe.raw, repo_root=REPO_ROOT
    )

    assert missing == []
    assert outside_mount == []
    assert actual == expected
