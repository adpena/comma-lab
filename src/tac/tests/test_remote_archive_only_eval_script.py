from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "remote_archive_only_eval.sh"
ENSURE_UV = REPO_ROOT / "scripts" / "ensure_remote_uv.sh"
WAVE3_DRIVER = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "lane_g_v3_owv3_wave3_refinement_20260501"
    / "wave3_chain_driver.sh"
)


def _script_text() -> str:
    return SCRIPT.read_text()


def test_archive_only_eval_requires_uv_before_cuda_eval() -> None:
    text = _script_text()
    assert "require_uv_and_ffmpeg_contract" in text
    assert "command -v uv" in text
    assert "robust_current/inflate.sh requires uv" in text
    assert "scripts/ensure_remote_uv.sh" in text
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" not in text
    assert "UV_PROJECT_ENVIRONMENT" in text
    assert "INFLATE_TORCH_SPEC" in text
    assert "torch==2.5.1+cu124" in text
    assert "download.pytorch.org/whl/cu124" in text
    assert "unsafe-best-match" in text
    assert "driver_major" in text
    assert "runtime_tooling.json" in text
    assert "inflate_torch_spec" in text
    assert "dali_bootstrap_dir" in text
    assert "KEEP_EVAL_WORK" in text
    assert "eval_work/inflated" in text
    assert "eval_work/extracted" in text
    assert "UV_ENV_REAL" in text
    assert "SKIP_UV_PROJECT_ENV_CLEANUP" in text
    assert '[[ "$UV_ENV_REAL" == "$LOG_DIR_REAL"/* ]]' in text
    assert "provenance.contest_auth_eval.json" in text
    assert "archive_custody.json" in text
    assert "CUSTODY_ARCHIVE" in text
    assert "archive custody copy drifted" in text
    assert "REQUIRED_SOURCE_SHA256S" in text
    assert "source_sha_ok" in text
    assert "required source SHA mismatch" in text
    assert 'INFLATE_SH="${INFLATE_SH:-submissions/robust_current/inflate.sh}"' in text
    assert "resolve_inflate_sh" in text
    assert "unsafe INFLATE_SH path" in text
    assert "inflate_sh_sha256" in text
    assert '--inflate-sh "$INFLATE_SH_ABS"' in text
    assert 'export INFLATE_REQUIRE_CUDA="${INFLATE_REQUIRE_CUDA:-1}"' in text


def test_archive_only_eval_bootstraps_scorer_runtime_dependencies() -> None:
    text = _script_text()
    assert "ensure_scorer_runtime_deps" in text
    assert "scorer_deps_probe.json" in text
    assert "scorer_deps_install.log" in text
    for dependency in (
        "timm>=0.9",
        "einops>=0.7",
        "segmentation-models-pytorch>=0.3",
        "safetensors>=0.4",
        "av>=10.0",
        "tqdm>=4.0",
    ):
        assert dependency in text
    for import_name in (
        "timm",
        "einops",
        "safetensors",
        "segmentation_models_pytorch",
        "av",
        "tqdm",
    ):
        assert import_name in text


def test_archive_only_eval_requires_ffmpeg_color_contract() -> None:
    text = _script_text()
    for option in (
        "in_range",
        "out_range",
        "in_color_matrix",
        "in_primaries",
        "in_transfer",
    ):
        assert option in text
    assert "installing BtbN master" in text
    assert "export FFMPEG_BIN" in text


def test_remote_uv_bootstrap_is_canonical_and_stdout_clean() -> None:
    text = ENSURE_UV.read_text()
    assert "set -euo pipefail" in text
    assert "candidate_uv" in text
    assert "UV_BOOTSTRAP_LOG" in text
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in text
    assert "printf '%s\\n' \"$UV_PATH\"" in text
    assert ">&2" in text, "logs must go to stderr so stdout remains the uv path"


def test_wave3_driver_uses_canonical_remote_uv_bootstrap() -> None:
    text = WAVE3_DRIVER.read_text()
    assert "scripts/ensure_remote_uv.sh" in text
    assert "UV_BIN=\"$(bash \"$WORKSPACE/scripts/ensure_remote_uv.sh\" --symlink-system)\"" in text
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" not in text
