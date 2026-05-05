from __future__ import annotations

import json
import subprocess
import sys

import comma_lab.preflight.strict_checks as strict_checks
from tac import preflight as tac_preflight


def test_strict_check_wrappers_point_at_live_tac_checks() -> None:
    assert strict_checks.check_no_mps_fallback_default is tac_preflight.check_no_mps_fallback_default
    assert strict_checks.check_42_train_inference_parity is tac_preflight.check_pose_projection_train_inference_parity
    assert strict_checks.check_public_release_hygiene is tac_preflight.check_public_release_hygiene


def test_emit_catalog_contains_ara_anchor_checks() -> None:
    catalog = strict_checks.emit_catalog()

    assert catalog["source_of_truth"] == "src/tac/preflight.py"
    assert catalog["adapter"] == "src/comma_lab/preflight/strict_checks.py"
    assert catalog["delegated_module"] == "tac.preflight"
    assert catalog["delegated_check_count"] > 0
    assert "check_no_mps_fallback_default" in catalog["anchor_checks"]
    assert "check_42_train_inference_parity" in catalog["anchor_checks"]


def test_module_emit_catalog_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "comma_lab.preflight.strict_checks", "--emit-catalog"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    payload = json.loads(result.stdout)
    assert payload["source_of_truth"] == "src/tac/preflight.py"
    assert payload["adapter"] == "src/comma_lab/preflight/strict_checks.py"
