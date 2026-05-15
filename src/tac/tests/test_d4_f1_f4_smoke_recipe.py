# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_d4_wrapper_defaults_to_f1_50_epoch_smoke() -> None:
    wrapper = (
        REPO_ROOT
        / "scripts"
        / "operator_authorize_substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.sh"
    ).read_text()

    assert 'D4_WYNER_ZIV_FRAME_0_SMOKE_EPOCHS:-50' in wrapper
    assert "--smoke-epochs" in wrapper


def test_d4_recipe_preserves_full_reference_and_f4_pair_cap() -> None:
    recipe = (
        REPO_ROOT
        / ".omx"
        / "operator_authorize_recipes"
        / "substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml"
    ).read_text()

    assert 'D4_WYNER_ZIV_FRAME_0_EPOCHS: "${D4_WYNER_ZIV_FRAME_0_EPOCHS:-2000}"' in recipe
    assert 'D4_WYNER_ZIV_FRAME_0_MAX_PAIRS: "${D4_WYNER_ZIV_FRAME_0_MAX_PAIRS:-200}"' in recipe
    assert "50 epochs" in recipe


def test_d4_remote_driver_defaults_to_t03_smoke_cap() -> None:
    driver = (
        REPO_ROOT / "scripts" / "remote_lane_substrate_d4_wyner_ziv_frame_0.sh"
    ).read_text()

    assert 'D4_WYNER_ZIV_FRAME_0_EPOCHS="${D4_WYNER_ZIV_FRAME_0_EPOCHS:-50}"' in driver
    assert 'D4_WYNER_ZIV_FRAME_0_MAX_PAIRS="${D4_WYNER_ZIV_FRAME_0_MAX_PAIRS:-200}"' in driver
    assert "max_pairs=$D4_WYNER_ZIV_FRAME_0_MAX_PAIRS" in driver
    assert '--max-pairs "$D4_WYNER_ZIV_FRAME_0_MAX_PAIRS"' in driver
    assert "stage_4_truncated_pair_smoke_skips_auth_eval" in driver
    assert "AUTH_EVAL_ARGS+=(--skip-auth-eval)" in driver
    assert '${AUTH_EVAL_ARGS[@]+"${AUTH_EVAL_ARGS[@]}"}' in driver


def test_d4_remote_driver_contest_cuda_marker_requires_valid_auth_eval_json() -> None:
    driver = (
        REPO_ROOT / "scripts" / "remote_lane_substrate_d4_wyner_ziv_frame_0.sh"
    ).read_text()

    assert "parse_auth_eval_score_claim" in driver
    assert "required_score_axis=\"contest_cuda\"" in driver
    assert "contest_auth_eval.json" in driver
    assert "auth_eval_missing" in driver
    assert "auth_eval_not_custody_valid" in driver
    assert "LANE_D4_WZF0_DONE [training-artifact]" in driver
    assert 'log "LANE_D4_WZF0_DONE [contest-CUDA] output_dir=' not in driver
    assert 'echo "LANE_D4_WZF0_DONE [contest-CUDA] $LANE_ID $(date' not in driver
