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
        "--stream-chunking-mode",
        "--stream-frame-range-size",
        "--stream-byte-size-target",
        "--stream-temporal-window-sec",
        "--stream-motion-threshold",
        "--stream-entropy-threshold",
        "--stream-saliency-topk",
        "--enable-gt-scorer-cache",
        "--enable-torch-compile",
        "--skip-auth-eval",
        "--full-cpu",
        "--advisory-cpu-explicitly-waived",
        "--enable-procedural-codebook-replacement",
        "--procedural-codebook-seed-hex",
        "--procedural-codebook-generator-kind",
        "--procedural-codebook-null-exploit-control",
        "--no-procedural-codebook-validate-domain",
        "--procedural-variant-provenance-path",
        "--procedural-variant-distillation-skip",
    )
    for flag in expected_flags:
        assert flag in script


def test_dp1_remote_driver_reuses_procedural_flags_for_smoke_and_full() -> None:
    script = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert (
        'DPP_PROCEDURAL_CODEBOOK_REPLACEMENT="${DPP_PROCEDURAL_CODEBOOK_REPLACEMENT:-0}"'
        in script
    )
    assert "DPP_PROCEDURAL_ARGS+=(--enable-procedural-codebook-replacement)" in script
    assert '"${DPP_PROCEDURAL_ARGS[@]}"' in script
    assert script.count('"${DPP_PROCEDURAL_ARGS[@]}"') >= 2


def test_dp1_remote_driver_uses_recipe_lane_id_for_dispatch_claim() -> None:
    script = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'LANE_ID="${DPP_LANE_ID:-lane_pretrained_driving_prior_lane_scaffold_20260513}"' in script
    assert '"$CLAIM_SUMMARY_JSON" "$LANE_ID" "$DISPATCH_INSTANCE_JOB_ID"' in script


def test_dp1_recipe_no_longer_claims_full_main_not_implemented() -> None:
    recipe = RECIPE.read_text(encoding="utf-8")

    assert "raises NotImplementedError" not in recipe
    assert "DPP_RUN_FULL=1" in recipe
    assert "DPP_ENABLE_GT_SCORER_CACHE" in recipe
    assert "paired contest-CPU / contest-CUDA auth eval" in recipe


def test_dp1_recipe_watches_source_custody_files() -> None:
    recipe = RECIPE.read_text(encoding="utf-8")

    for path in (
        "src/tac/substrates/pretrained_driving_prior/dataset_source.py",
        "src/tac/substrates/pretrained_driving_prior/local_chunk_cache.py",
        "src/tac/substrates/pretrained_driving_prior/local_chunk_streamer.py",
        "src/tac/substrates/pretrained_driving_prior/log_incremental_feeder.py",
        "src/tac/substrates/pretrained_driving_prior/composition.py",
    ):
        assert path in recipe


def test_dp1_recipe_declares_full_hardware_feasibility_contract() -> None:
    recipe = RECIPE.read_text(encoding="utf-8")

    assert "min_vram_gb: 16" in recipe
    assert 'min_smoke_gpu: "T4"' in recipe
    assert "pyav_decode_strategy: cpu_thread_async_upload" in recipe
    assert "video_input_strategy: per_dispatch_local_copy" in recipe


def test_dp1_procedural_paired_smoke_recipes_exist_and_stay_operator_gated() -> None:
    recipe_dir = REPO_ROOT / ".omx" / "operator_authorize_recipes"
    recipe_names = (
        "substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch.yaml",
        "substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml",
        "substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch.yaml",
    )
    for name in recipe_names:
        text = (recipe_dir / name).read_text(encoding="utf-8")
        assert "dispatch_enabled: false" in text
        assert "smoke_only: false" in text
        assert "DPP_OUTPUT_DIR: /modal_results/${INSTANCE_JOB_ID}/output" in text
        assert "DPP_LANE_ID:" in text
        assert 'DPP_SKIP_AUTH_EVAL: "1"' in text
        assert "score_claim: false" in text
        assert "promotion_eligible: false" in text


def test_dp1_procedural_and_null_recipes_set_actual_trainer_env_names() -> None:
    recipe_dir = REPO_ROOT / ".omx" / "operator_authorize_recipes"
    procedural = (
        recipe_dir
        / "substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch.yaml"
    ).read_text(encoding="utf-8")
    null_control = (
        recipe_dir
        / "substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch.yaml"
    ).read_text(encoding="utf-8")

    assert 'DPP_PROCEDURAL_CODEBOOK_REPLACEMENT: "1"' in procedural
    assert 'DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND: "pcg64"' in procedural
    assert 'DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL: "0"' in procedural
    assert 'DPP_PROCEDURAL_CODEBOOK_REPLACEMENT: "1"' in null_control
    assert 'DPP_PROCEDURAL_CODEBOOK_GENERATOR_KIND: "pcg64"' in null_control
    assert 'DPP_PROCEDURAL_CODEBOOK_NULL_EXPLOIT_CONTROL: "1"' in null_control
