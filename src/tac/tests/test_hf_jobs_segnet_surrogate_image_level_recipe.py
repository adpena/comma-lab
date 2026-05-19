# SPDX-License-Identifier: MIT
"""Regression tests for the HF Jobs SegNet surrogate dispatch recipe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_PATH = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/"
    "substrate_hf_jobs_segnet_surrogate_distillation_t4_dispatch.yaml"
)


def _recipe() -> dict:
    data = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_hf_jobs_recipe_is_explicit_research_surrogate_non_promotional() -> None:
    recipe = _recipe()
    assert recipe["platform"] == "hf_jobs"
    assert recipe["dispatch_kind"] == "hf_jobs_research_surrogate"
    assert recipe["dispatch_enabled"] is True
    assert recipe["research_only"] is True
    assert recipe["score_claim"] is False
    assert recipe["promotion_eligible"] is False
    assert recipe["ready_for_exact_eval_dispatch"] is False
    assert recipe["authority_class"] == "research_advisory_surrogate_training"


def test_hf_jobs_recipe_pins_dataset_sha_into_dispatch_config() -> None:
    recipe = _recipe()
    hf_jobs = recipe["hf_jobs"]
    assert (
        hf_jobs["hub_dataset_sha"]
        == "52ef7313ed2cb6f84e9635cd99bd9b51bc1ecd9a"
    )


def test_check_dispatch_protocol_cli_treats_hf_jobs_as_native() -> None:
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        "tools/check_dispatch_protocol_complete.py",
        "--recipe",
        str(RECIPE_PATH),
        "--strict",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["native_dispatch"] is True
    assert payload["dispatch_protocol_complete"] is True
