# SPDX-License-Identifier: MIT
"""Regression guards for the Catalog #204 stack-of-stacks recovery route."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RECIPE_PATH = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/"
    "substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch.yaml"
)
DRIVER_PATH = REPO_ROOT / "scripts/remote_lane_substrate_stack_of_stacks.sh"


def _recipe() -> dict:
    return yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))


def _driver_text() -> str:
    return DRIVER_PATH.read_text(encoding="utf-8")


def test_recipe_does_not_mask_catalog204_modal_results_branch() -> None:
    """The recipe must not force Modal output back under /tmp/pact.

    modal_train_lane maps recipe values rooted at /workspace/pact to the
    writable /tmp/pact copy. If STACK_OF_STACKS_OUTPUT_DIR defaults there,
    the driver's /modal_results fallback is bypassed and contest_auth_eval
    refuses the evidence path again.
    """

    env = _recipe()["env_overrides"]
    value = env["STACK_OF_STACKS_OUTPUT_DIR"]
    assert value == "${STACK_OF_STACKS_OUTPUT_DIR:-}"


def test_recipe_requests_exact_cuda_for_archive_only_recovery_eval() -> None:
    env = _recipe()["env_overrides"]
    assert env["STACK_OF_STACKS_AUTH_EVAL_REQUIRE_CONTEST_CUDA"] == "1"


def test_driver_threads_langevin_cap_and_lane_id_to_trainer() -> None:
    body = _driver_text()
    assert '--langevin-t-init "$STACK_OF_STACKS_LANGEVIN_T_INIT_CAP"' in body
    assert '--langevin-polish-epochs "$STACK_OF_STACKS_LANGEVIN_POLISH_EPOCHS"' in body
    assert '--lane-id "$LANE_ID"' in body


def test_driver_unsets_modal_advisory_only_only_under_exact_cuda_opt_in() -> None:
    body = _driver_text()
    guard = 'if [ "${STACK_OF_STACKS_AUTH_EVAL_REQUIRE_CONTEST_CUDA:-0}" = "1" ]; then'
    assert guard in body
    assert "unset MODAL_AUTH_EVAL_ADVISORY_ONLY" in body
    assert body.index(guard) < body.index("unset MODAL_AUTH_EVAL_ADVISORY_ONLY")
