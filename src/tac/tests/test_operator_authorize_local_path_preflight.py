from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools" / "operator_authorize.py"


def _run_dry_recipe(recipe: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), "--recipe", str(recipe), "--dry-run"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_dry_run_fails_before_claim_when_enabled_recipe_names_missing_local_paths(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "bad_stack_recipe.yaml"
    recipe.write_text(
        """
schema_version: 1
name: bad_stack_recipe
lane_id: lane_bad_stack_recipe
summary: enabled native dispatch with missing local dispatch artifacts
platform: modal
gpu: A100
cost_band:
  epochs: 10
  all_flags_on: true
  hand_calibrated_fallback_p50_usd: 1.00
remote_driver: scripts/remote_lane_substrate_missing_stack.sh
modal:
  lane_script: scripts/remote_lane_substrate_missing_stack.sh
  cost_band_trainer: experiments/missing_train_substrate_stack.py
dependencies:
  - tools/operator_authorize.py (Catalog #162)
  - scripts/remote_lane_substrate_missing_stack.sh
""".strip(),
        encoding="utf-8",
    )

    result = _run_dry_recipe(recipe)

    assert result.returncode != 0
    assert "declared local path preflight failed" in result.stderr
    assert "remote_driver, modal.lane_script, dependencies[1]" in result.stderr
    assert "modal.cost_band_trainer" in result.stderr
    assert "lane claim/provider setup" in result.stderr


def test_dry_run_skips_missing_future_paths_when_recipe_is_explicitly_deferred(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "deferred_recipe.yaml"
    recipe.write_text(
        """
schema_version: 1
name: deferred_recipe
lane_id: lane_deferred_recipe
summary: deferred native recipe with documented future paths
dispatch_enabled: false
dispatch_blockers:
  - remote_driver_missing
defer_reason: future substrate trainer is intentionally not ready
platform: modal
gpu: A100
cost_band:
  epochs: "deferred"
  all_flags_on: false
  hand_calibrated_fallback_p50_usd: 1.00
remote_driver: scripts/remote_lane_substrate_future.sh
modal:
  lane_script: scripts/remote_lane_substrate_future.sh
dependencies:
  - "[MISSING] scripts/remote_lane_substrate_future.sh"
""".strip(),
        encoding="utf-8",
    )

    result = _run_dry_recipe(recipe)

    assert result.returncode == 0
    assert "--dry-run; would refuse:" in result.stdout
    assert "remote_driver_missing" in result.stdout


def test_dry_run_accepts_annotated_existing_dependency_paths(tmp_path: Path) -> None:
    recipe = tmp_path / "good_recipe.yaml"
    recipe.write_text(
        """
schema_version: 1
name: good_recipe
lane_id: lane_good_recipe
summary: enabled native dispatch with annotated local paths
platform: modal
gpu: A100
cost_band:
  epochs: 10
  all_flags_on: true
  hand_calibrated_fallback_p50_usd: 1.00
remote_driver: scripts/remote_lane_substrate_siren.sh
modal:
  lane_script: scripts/remote_lane_substrate_siren.sh
  cost_band_trainer: experiments/train_substrate_siren.py
sentinel_files:
  - src/tac/substrates/siren/score_aware_loss.py
dependencies:
  - tools/validate_dispatch_required_inputs.py (Catalog #152)
  - tools/run_modal_smoke_before_full.py (Catalog #167)
""".strip(),
        encoding="utf-8",
    )

    result = _run_dry_recipe(recipe)

    assert result.returncode == 0
    assert "--dry-run; no confirmation prompt, no dispatch" in result.stdout
