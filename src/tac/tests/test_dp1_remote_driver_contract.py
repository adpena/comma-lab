from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REMOTE_SCRIPT = REPO_ROOT / "scripts" / "remote_lane_substrate_pretrained_driving_prior.sh"
RECIPE = (
    REPO_ROOT
    / ".omx"
    / "operator_authorize_recipes"
    / "substrate_pretrained_driving_prior_modal_t4_dispatch.yaml"
)


def test_dp1_remote_driver_requires_explicit_comma2k19_source() -> None:
    script = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'DPP_DATASET_NAME" = "comma2k19' in script
    assert '[ -z "$DPP_COMMA2K19_CHUNKS_DIR" ]' in script
    assert '[ -z "$DPP_CACHE_DIR" ]' in script
    assert '[ "$DPP_USE_STREAMER" != "1" ]' in script
    assert "requires one explicit dataset source" in script
    assert "exit 26" in script


def test_dp1_remote_driver_forwards_full_run_controls() -> None:
    script = REMOTE_SCRIPT.read_text(encoding="utf-8")

    expected_flags = (
        "--max-disk-gb",
        "--log-incremental-base",
        "--log-incremental-max-chunks",
        "--log-incremental-quality-threshold",
        "--max-distillation-frames",
        "--max-distillation-chunks",
        "--max-pairs",
        "--val-pair-count",
        "--val-every-epochs",
        "--cache-dir",
        "--use-streamer",
        "--stream-log-dir",
        "--ram-buffer-gb",
        "--streamer-frames-per-chunk",
        "--enable-gt-scorer-cache",
        "--enable-torch-compile",
        "--skip-auth-eval",
        "--full-cpu",
        "--advisory-cpu-explicitly-waived",
    )
    for flag in expected_flags:
        assert flag in script


def test_dp1_recipe_no_longer_claims_full_main_not_implemented() -> None:
    recipe = RECIPE.read_text(encoding="utf-8")

    assert "raises NotImplementedError" not in recipe
    assert "DPP_RUN_FULL=1" in recipe
    assert "DPP_ENABLE_GT_SCORER_CACHE" in recipe
    assert "paired contest-CPU / contest-CUDA auth eval" in recipe

