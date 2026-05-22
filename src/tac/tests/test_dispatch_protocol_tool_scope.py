# SPDX-License-Identifier: MIT
"""Tests for Catalog #270 scope clarification: tool vs substrate dispatches.

Source lane: ``lane_catalog_270_scope_fix_tool_vs_substrate_dispatch_20260517``.

These tests pin the behavior of ``_is_tool_dispatch`` + the
``evaluate_dispatch_protocol_complete`` short-circuit for tool dispatches.
Substrate-trainer-only Tier 2/3 fields (Catalogs #172/#178/#179/#215-CPU/#226)
are skipped for tool dispatches; substrate dispatches retain full enforcement.
"""

from __future__ import annotations

from pathlib import Path

from tac.deploy.dispatch_protocol import (
    LEGAL_DISPATCH_KINDS,
    TOOL_DISPATCH_LEGAL_GPU_TOKENS,
    evaluate_dispatch_protocol_complete,
    is_tool_dispatch,
)


def _write_modal_driver_with_env(driver_path: Path) -> None:
    """Write a Modal driver script with the canonical NVML env block."""

    driver_path.parent.mkdir(parents=True, exist_ok=True)
    driver_path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
                'export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"',
                'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
                "python tools/extract_master_gradient.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_minimal_tool(tool_path: Path) -> None:
    """Write a minimal tool that uses no_grad (satisfies Catalog #180)."""

    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text(
        "\n".join(
            [
                "# SPDX-License-Identifier: MIT",
                '"""Tool dispatch fixture."""',
                "import torch",
                "def main():",
                "    with torch.no_grad():",
                "        return 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_substrate_trainer(trainer_path: Path) -> None:
    """Write a substrate trainer satisfying all Tier 3 fields."""

    trainer_path.parent.mkdir(parents=True, exist_ok=True)
    trainer_path.write_text(
        "\n".join(
            [
                "# SPDX-License-Identifier: MIT",
                "import argparse",
                "import torch",
                "from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call",
                "torch.backends.cuda.matmul.allow_tf32 = True",
                "def build_parser():",
                "    p = argparse.ArgumentParser()",
                "    p.add_argument('--enable-autocast-fp16', action='store_true')",
                "    p.add_argument('--enable-torch-compile', action='store_true')",
                "    return p",
                "def main():",
                "    with torch.no_grad():",
                "        gate_auth_eval_call([])",
                "    return 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_substrate_driver_with_mode_device(
    driver_path: Path,
    *,
    device_default: str = "cuda",
) -> None:
    """Write a Modal substrate driver with trainer mode and device env vars."""

    driver_path.parent.mkdir(parents=True, exist_ok=True)
    driver_path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
                'export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"',
                'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
                'SYNTH_TRAINER_MODE="${SYNTH_TRAINER_MODE:-smoke}"',
                f'SYNTH_DEVICE="${{SYNTH_DEVICE:-{device_default}}}"',
                'SMOKE_FLAG=""',
                'if [ "$SYNTH_TRAINER_MODE" = "smoke" ]; then',
                '    SMOKE_FLAG="--smoke"',
                "fi",
                'python experiments/train_substrate_alpha.py --device "$SYNTH_DEVICE" $SMOKE_FLAG',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _blockers(report) -> str:
    return "\n".join(report.blockers)


def _substrate_recipe(*, device: str = "cuda") -> dict:
    return {
        "name": "full_mode_device_test",
        "lane_id": "lane_full_mode_device_20260522",
        "dispatch_enabled": True,
        "dispatch_kind": "substrate",
        "platform": "modal",
        "gpu": "T4",
        "min_vram_gb": 16,
        "min_smoke_gpu": "T4",
        "video_input_strategy": "shared_volume_no_contention_expected",
        "pyav_decode_strategy": "cpu_thread_async_upload",
        "target_modes": ["research_substrate"],
        "canary_status": "independent_substrate",
        "cost_band": {"epochs": 1, "gpu_key": "T4"},
        "remote_driver": "scripts/remote_lane_substrate_alpha.sh",
        "required_input_files_trainer": "experiments/train_substrate_alpha.py",
        "env_overrides": {
            "SYNTH_TRAINER_MODE": "full",
            "SYNTH_DEVICE": device,
        },
    }


# ---------------------------------------------------------------------------
# _is_tool_dispatch / is_tool_dispatch helper
# ---------------------------------------------------------------------------


def test_is_tool_dispatch_returns_true_for_tools_path(tmp_path: Path) -> None:
    tool = tmp_path / "tools" / "my_extractor.py"
    _write_minimal_tool(tool)
    assert is_tool_dispatch({}, trainer_path=tool, repo_root=tmp_path) is True


def test_is_tool_dispatch_returns_false_for_substrate_trainer(tmp_path: Path) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    _write_substrate_trainer(trainer)
    assert is_tool_dispatch({}, trainer_path=trainer, repo_root=tmp_path) is False


def test_substrate_full_mode_refuses_cpu_device_before_dispatch(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    _write_substrate_trainer(trainer)
    driver = tmp_path / "scripts" / "remote_lane_substrate_alpha.sh"
    _write_substrate_driver_with_mode_device(driver)

    report = evaluate_dispatch_protocol_complete(
        _substrate_recipe(device="cpu"),
        repo_root=tmp_path,
    )

    assert report.dispatch_protocol_complete is False
    assert "full_mode_device_cpu_bug_class" in _blockers(report)


def test_substrate_full_mode_allows_cuda_device(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    _write_substrate_trainer(trainer)
    driver = tmp_path / "scripts" / "remote_lane_substrate_alpha.sh"
    _write_substrate_driver_with_mode_device(driver)

    report = evaluate_dispatch_protocol_complete(
        _substrate_recipe(device="cuda"),
        repo_root=tmp_path,
    )

    assert report.dispatch_protocol_complete is True, _blockers(report)


def test_substrate_full_mode_refuses_driver_default_cpu_device(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    _write_substrate_trainer(trainer)
    driver = tmp_path / "scripts" / "remote_lane_substrate_alpha.sh"
    _write_substrate_driver_with_mode_device(driver, device_default="cpu")
    recipe = _substrate_recipe(device="cuda")
    recipe["env_overrides"].pop("SYNTH_DEVICE")

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
    )

    assert report.dispatch_protocol_complete is False
    assert "full_mode_device_cpu_bug_class" in _blockers(report)


def test_is_tool_dispatch_returns_true_when_dispatch_kind_tool_explicit(
    tmp_path: Path,
) -> None:
    # Even with a substrate trainer path, explicit ``dispatch_kind: tool``
    # short-circuits to True (forward-compat for hybrid wrappers).
    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    _write_substrate_trainer(trainer)
    assert (
        is_tool_dispatch(
            {"dispatch_kind": "tool"}, trainer_path=trainer, repo_root=tmp_path
        )
        is True
    )


def test_is_tool_dispatch_returns_true_for_hf_jobs_research_surrogate_kind(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "hf_jobs_surrogate.py"
    _write_minimal_tool(trainer)
    assert (
        is_tool_dispatch(
            {"dispatch_kind": "hf_jobs_research_surrogate", "platform": "hf_jobs"},
            trainer_path=trainer,
            repo_root=tmp_path,
        )
        is True
    )


def test_is_tool_dispatch_rejects_hf_jobs_surrogate_kind_on_wrong_platform(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "hf_jobs_surrogate.py"
    _write_minimal_tool(trainer)
    assert (
        is_tool_dispatch(
            {"dispatch_kind": "hf_jobs_research_surrogate", "platform": "modal"},
            trainer_path=trainer,
            repo_root=tmp_path,
        )
        is False
    )


def test_hf_jobs_research_surrogate_requires_non_promotional_flags(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "experiments" / "hf_jobs_surrogate.py"
    _write_minimal_tool(trainer)
    driver = tmp_path / "tools" / "dispatch_hf_jobs_vision_training.py"
    _write_minimal_tool(driver)
    recipe = {
        "name": "hf_jobs_surrogate_test",
        "lane_id": "lane_hf_jobs_surrogate_20260519",
        "dispatch_enabled": True,
        "dispatch_kind": "hf_jobs_research_surrogate",
        "platform": "hf_jobs",
        "gpu": "t4-small",
        "min_vram_gb": 16,
        "min_smoke_gpu": "T4",
        "video_input_strategy": "shared_volume_no_contention_expected",
        "pyav_decode_strategy": "not_applicable",
        "target_modes": ["research_substrate"],
        "canary_status": "independent_substrate",
        "cost_band": {"epochs": 1, "gpu_key": "t4-small", "platform_key": "hf_jobs"},
        "remote_driver": "tools/dispatch_hf_jobs_vision_training.py",
        "required_input_files_trainer": "experiments/hf_jobs_surrogate.py",
        "hf_jobs": {"expected_axis": "cuda"},
    }

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    blockers = _blockers(report)
    assert report.dispatch_protocol_complete is False
    assert "hf_jobs_research_surrogate_requires_research_only_true" in blockers
    assert "hf_jobs_research_surrogate_requires_score_claim_false" in blockers
    assert "hf_jobs_research_surrogate_requires_promotion_eligible_false" in blockers
    assert (
        "hf_jobs_research_surrogate_requires_ready_for_exact_eval_dispatch_false"
        in blockers
    )
    assert "hf_jobs_research_surrogate_expected_axis_not_advisory" in blockers


def test_is_tool_dispatch_returns_false_when_dispatch_kind_substrate_explicit(
    tmp_path: Path,
) -> None:
    # Even with a tools/ path, explicit ``dispatch_kind: substrate`` keeps
    # substrate-only fields enforced (rare; useful for substrate research
    # scripts placed in tools/).
    tool = tmp_path / "tools" / "substrate_research_helper.py"
    _write_minimal_tool(tool)
    assert (
        is_tool_dispatch(
            {"dispatch_kind": "substrate"}, trainer_path=tool, repo_root=tmp_path
        )
        is False
    )


def test_is_tool_dispatch_returns_false_for_none_trainer(tmp_path: Path) -> None:
    # Defensive: no trainer path + no explicit kind = substrate (conservative
    # default; substrate-trainer Tier 3 enforcement kicks in if the trainer is
    # later resolved).
    assert is_tool_dispatch({}, trainer_path=None, repo_root=tmp_path) is False


def test_is_tool_dispatch_returns_false_for_path_outside_repo(
    tmp_path: Path,
) -> None:
    # If the trainer cannot be resolved under repo_root, do not classify as
    # tool (avoid silent escape).
    outside = tmp_path / "outside.py"
    outside.write_text("# stray\n", encoding="utf-8")
    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    assert is_tool_dispatch({}, trainer_path=outside, repo_root=other_root) is False


def test_legal_dispatch_kinds_pinned() -> None:
    assert (
        frozenset(
            {
                "substrate",
                "tool",
                "local_research_signal",
                "hf_jobs_research_surrogate",
            }
        )
        == LEGAL_DISPATCH_KINDS
    )


def test_tool_dispatch_legal_gpu_tokens_pinned() -> None:
    # Case-insensitive CPU tokens accepted for tool dispatches.
    assert "cpu" in TOOL_DISPATCH_LEGAL_GPU_TOKENS
    assert "CPU" in TOOL_DISPATCH_LEGAL_GPU_TOKENS


# ---------------------------------------------------------------------------
# Scoped acceptance: tool dispatches PASS without substrate Tier 3 fields
# ---------------------------------------------------------------------------


def _tool_recipe(driver_relpath: str, trainer_relpath: str) -> dict[str, object]:
    return {
        "name": "tool_dispatch_test",
        "lane_id": "lane_tool_20260517",
        "dispatch_enabled": True,
        "dispatch_kind": "tool",
        "platform": "modal",
        "gpu": "CPU",
        "min_vram_gb": 1,
        "min_smoke_gpu": "CPU",
        "video_input_strategy": "per_dispatch_local_copy",
        "pyav_decode_strategy": "cpu_thread_async_upload",
        "target_modes": ["research_substrate"],
        "canary_status": "independent_substrate",
        "cost_band": {"epochs": 1, "gpu_key": "CPU", "platform_key": "modal"},
        "remote_driver": driver_relpath,
        "required_input_files_trainer": trainer_relpath,
        "modal": {
            "lane_script": driver_relpath,
            "cost_band_trainer": trainer_relpath,
        },
    }


def test_tool_dispatch_passes_without_substrate_tier_3_fields(tmp_path: Path) -> None:
    """Tool dispatch (master gradient extractor pattern) passes the protocol.

    This is the canonical anchor for Catalog #270 scope fix: the master
    gradient extractor at ``tools/extract_master_gradient.py`` with the
    ``master_gradient_fec6_modal_cpu_dispatch.yaml`` recipe.
    """

    tool = tmp_path / "tools" / "extract_master_gradient.py"
    _write_minimal_tool(tool)
    driver = tmp_path / "scripts" / "operator_authorize_master_gradient_fec6.sh"
    _write_modal_driver_with_env(driver)
    recipe = _tool_recipe(
        "scripts/operator_authorize_master_gradient_fec6.sh",
        "tools/extract_master_gradient.py",
    )

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is True, _blockers(report)
    assert all(tier.passed for tier in report.tiers)


def test_tool_dispatch_passes_with_min_smoke_gpu_cpu_lowercase(
    tmp_path: Path,
) -> None:
    """Tool dispatch accepts lowercase ``cpu`` for ``min_smoke_gpu``."""

    tool = tmp_path / "tools" / "extract_master_gradient.py"
    _write_minimal_tool(tool)
    driver = tmp_path / "scripts" / "tool_driver.sh"
    _write_modal_driver_with_env(driver)
    recipe = _tool_recipe(
        "scripts/tool_driver.sh", "tools/extract_master_gradient.py"
    )
    recipe["min_smoke_gpu"] = "cpu"
    recipe["gpu"] = "cpu"
    recipe["cost_band"] = {"epochs": 1, "gpu_key": "cpu", "platform_key": "modal"}

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is True, _blockers(report)


def test_tool_dispatch_implicit_via_tools_path_no_explicit_dispatch_kind(
    tmp_path: Path,
) -> None:
    """Tool dispatch is detected by trainer path even without ``dispatch_kind``."""

    tool = tmp_path / "tools" / "implicit_tool.py"
    _write_minimal_tool(tool)
    driver = tmp_path / "scripts" / "tool_driver.sh"
    _write_modal_driver_with_env(driver)
    recipe = _tool_recipe(
        "scripts/tool_driver.sh", "tools/implicit_tool.py"
    )
    # Remove the explicit kind to test implicit detection.
    recipe.pop("dispatch_kind")

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is True, _blockers(report)


def test_tool_dispatch_still_enforces_no_grad(tmp_path: Path) -> None:
    """Tool dispatches still get the universal no_grad/inference_mode check.

    The no_grad/inference_mode discipline is universally applicable for any
    torch-using tool. A tool lacking BOTH no_grad AND the NO_GRAD_WAIVED token
    is still flagged so eval-time memory hygiene cannot silently regress.
    """

    tool = tmp_path / "tools" / "no_grad_missing.py"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(
        "# SPDX-License-Identifier: MIT\n"
        '"""Tool dispatch without no_grad."""\n'
        "def main():\n"
        "    return 0\n",
        encoding="utf-8",
    )
    driver = tmp_path / "scripts" / "tool_driver.sh"
    _write_modal_driver_with_env(driver)
    recipe = _tool_recipe(
        "scripts/tool_driver.sh", "tools/no_grad_missing.py"
    )

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    blockers = _blockers(report)
    assert "catalog_180_no_grad_eval_missing_or_unwaived" in blockers
    # Substrate-only Tier 3 fields must NOT appear for tool dispatch.
    assert "catalog_172_autocast_fp16_missing_or_unwaived" not in blockers
    assert "catalog_178_tf32_missing_or_unwaived" not in blockers
    assert "catalog_179_torch_compile_missing_or_unwaived" not in blockers
    assert "catalog_226_auth_eval_canonical_helper_missing" not in blockers


def test_tool_dispatch_still_enforces_modal_env_hygiene(tmp_path: Path) -> None:
    """Tool dispatches still get the universal Modal NVML env block check.

    Catalog #244 NVML env hygiene is universally applicable to any Modal
    dispatch — it prevents the NVML 999 driver crash bug class. Tool
    dispatches without the canonical env block are flagged.
    """

    tool = tmp_path / "tools" / "extract_master_gradient.py"
    _write_minimal_tool(tool)
    driver = tmp_path / "scripts" / "missing_env_driver.sh"
    driver.parent.mkdir(parents=True, exist_ok=True)
    # Driver missing the canonical NVML env block.
    driver.write_text("#!/bin/bash\nset -euo pipefail\necho missing env\n")
    recipe = _tool_recipe(
        "scripts/missing_env_driver.sh", "tools/extract_master_gradient.py"
    )

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is False
    assert "catalog_244_modal_env_hygiene_missing" in _blockers(report)


# ---------------------------------------------------------------------------
# Substrate dispatches RETAIN full enforcement (no regression)
# ---------------------------------------------------------------------------


def test_substrate_dispatch_still_enforces_autocast_fp16(tmp_path: Path) -> None:
    """Substrate trainer missing --enable-autocast-fp16 is STILL flagged."""

    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text(
        "import torch\n"
        "torch.backends.cuda.matmul.allow_tf32 = True\n"
        "def main():\n"
        "    with torch.no_grad():\n"
        "        return 0\n",
        encoding="utf-8",
    )
    driver = tmp_path / "scripts" / "remote_lane_substrate_alpha.sh"
    _write_modal_driver_with_env(driver)
    recipe = {
        "name": "substrate_alpha_modal_t4_dispatch",
        "lane_id": "lane_alpha_20260515",
        "dispatch_enabled": True,
        # No dispatch_kind → defaults to substrate (experiments/train_substrate_*.py path).
        "platform": "modal",
        "gpu": "T4",
        "min_vram_gb": 16,
        "min_smoke_gpu": "T4",
        "video_input_strategy": "per_dispatch_local_copy",
        "pyav_decode_strategy": "cpu_thread_async_upload",
        "target_modes": ["contest_one_video_replay"],
        "canary_status": "independent_substrate",
        "cost_band": {"epochs": 10, "gpu_key": "T4", "platform_key": "modal"},
        "remote_driver": "scripts/remote_lane_substrate_alpha.sh",
        "required_input_files_trainer": "experiments/train_substrate_alpha.py",
        "modal": {
            "lane_script": "scripts/remote_lane_substrate_alpha.sh",
            "cost_band_trainer": "experiments/train_substrate_alpha.py",
        },
    }

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    blockers = _blockers(report)
    assert report.dispatch_protocol_complete is False
    # All substrate-only Tier 3 fields STILL enforced.
    assert "catalog_172_autocast_fp16_missing_or_unwaived" in blockers
    assert "catalog_179_torch_compile_missing_or_unwaived" in blockers
    assert "catalog_226_auth_eval_canonical_helper_missing" in blockers


def test_substrate_dispatch_still_rejects_cpu_min_smoke_gpu(tmp_path: Path) -> None:
    """Substrate trainer claiming ``min_smoke_gpu: CPU`` is STILL flagged.

    A substrate trainer cannot pretend to be a tool dispatch by declaring CPU
    smoke; the GPU class enforcement remains active when ``dispatch_kind`` is
    not ``tool`` and the trainer path is not under ``tools/``.
    """

    trainer = tmp_path / "experiments" / "train_substrate_alpha.py"
    _write_substrate_trainer(trainer)
    driver = tmp_path / "scripts" / "remote_lane_substrate_alpha.sh"
    _write_modal_driver_with_env(driver)
    recipe = {
        "name": "substrate_alpha_modal_t4_dispatch",
        "lane_id": "lane_alpha_20260515",
        "dispatch_enabled": True,
        "platform": "modal",
        "gpu": "T4",
        "min_vram_gb": 16,
        "min_smoke_gpu": "CPU",  # Substrate claiming CPU smoke is forbidden.
        "video_input_strategy": "per_dispatch_local_copy",
        "pyav_decode_strategy": "cpu_thread_async_upload",
        "target_modes": ["contest_one_video_replay"],
        "canary_status": "independent_substrate",
        "cost_band": {"epochs": 10, "gpu_key": "T4", "platform_key": "modal"},
        "remote_driver": "scripts/remote_lane_substrate_alpha.sh",
        "required_input_files_trainer": "experiments/train_substrate_alpha.py",
        "modal": {
            "lane_script": "scripts/remote_lane_substrate_alpha.sh",
            "cost_band_trainer": "experiments/train_substrate_alpha.py",
        },
    }

    report = evaluate_dispatch_protocol_complete(
        recipe,
        repo_root=tmp_path,
        native_dispatch=True,
    )

    blockers = _blockers(report)
    assert report.dispatch_protocol_complete is False
    assert "catalog_215_min_smoke_gpu_missing_or_illegal" in blockers


# ---------------------------------------------------------------------------
# Live recipe regression: the actual master_gradient_fec6 recipe passes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sister gate: canonical_dispatch_optimization_protocol.py also respects scope
# ---------------------------------------------------------------------------


def _invoke_canonical_protocol_cli(
    repo_root: Path,
    trainer: str,
    recipe: str | None = None,
) -> dict:
    """Invoke ``tools/canonical_dispatch_optimization_protocol.py --json``."""
    import json
    import subprocess

    helper = repo_root / "tools" / "canonical_dispatch_optimization_protocol.py"
    venv_python = repo_root / ".venv" / "bin" / "python"
    cmd = [
        str(venv_python),
        str(helper),
        "--trainer",
        trainer,
        "--json",
    ]
    if recipe is not None:
        cmd.extend(["--recipe", recipe])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=30,
    )
    if not result.stdout.strip():
        raise RuntimeError(
            f"canonical protocol CLI emitted no JSON; stderr={result.stderr!r}"
        )
    return json.loads(result.stdout)


def test_canonical_dispatch_optimization_protocol_skips_tool_substrate_tier_3() -> None:
    """The sister gate ``tools/canonical_dispatch_optimization_protocol.py``
    also skips substrate-only Tier 1/3 signals for tool dispatches.

    This is the regression guard for the ``tools/local_pre_deploy_check.py``
    8th harness check which routes through this canonical protocol.
    """

    repo_root = Path(__file__).resolve().parents[3]
    trainer_path = repo_root / "tools" / "extract_master_gradient.py"
    if not trainer_path.is_file():
        import pytest

        pytest.skip(f"trainer missing at {trainer_path}")

    verdict = _invoke_canonical_protocol_cli(
        repo_root,
        "tools/extract_master_gradient.py",
        recipe="master_gradient_fec6_modal_cpu_dispatch",
    )

    assert verdict["overall_pass"] is True, (
        f"canonical protocol verdict for tool dispatch should pass; "
        f"got blockers: {verdict.get('blockers')}"
    )


def test_canonical_dispatch_optimization_protocol_still_enforces_substrate_tier_3() -> None:
    """Sister gate still enforces substrate-only Tier 1/3 for substrate trainers.

    No-regression guard: substrate trainers still get full Tier 1/3 checks.
    """

    repo_root = Path(__file__).resolve().parents[3]
    trainers_dir = repo_root / "experiments"
    substrate_trainers = sorted(trainers_dir.glob("train_substrate_*.py"))
    if not substrate_trainers:
        import pytest

        pytest.skip("no substrate trainers found for regression test")

    # Pick the first substrate trainer; we only assert the verdict map
    # includes substrate-only Tier 1 signal keys (not skipped).
    trainer_rel = substrate_trainers[0].relative_to(repo_root).as_posix()
    verdict = _invoke_canonical_protocol_cli(repo_root, trainer_rel)
    expected_tier1_signals = {
        "autocast_fp16",
        "tf32",
        "torch_compile",
        "canonical_scorer_loss",
        "no_grad_at_eval",
    }
    actual_signals = set(verdict["tier1"]["pass_signals"].keys())
    assert expected_tier1_signals.issubset(actual_signals), (
        f"substrate trainer must check all Tier 1 signals; got {actual_signals}"
    )


# ---------------------------------------------------------------------------
# Live master_gradient recipe regression
# ---------------------------------------------------------------------------


def test_live_master_gradient_recipe_passes_dispatch_protocol() -> None:
    """The ACTUAL ``.omx/operator_authorize_recipes/master_gradient_fec6_modal_cpu_dispatch.yaml``
    recipe passes the dispatch protocol after the Catalog #270 scope fix.

    This is the live regression guard: any future refactor that breaks the
    tool-vs-substrate scope clarification will fail this test.
    """

    import yaml

    repo_root = Path(__file__).resolve().parents[3]
    recipe_path = (
        repo_root
        / ".omx"
        / "operator_authorize_recipes"
        / "master_gradient_fec6_modal_cpu_dispatch.yaml"
    )
    trainer_path = repo_root / "tools" / "extract_master_gradient.py"
    driver_path = (
        repo_root
        / "scripts"
        / "operator_authorize_master_gradient_fec6_modal_cpu.sh"
    )

    if not recipe_path.is_file():
        # The recipe file may not exist in some test contexts; skip rather
        # than fail.
        import pytest

        pytest.skip(f"recipe not present at {recipe_path}")
    if not trainer_path.is_file():
        import pytest

        pytest.skip(f"trainer not present at {trainer_path}")
    if not driver_path.is_file():
        import pytest

        pytest.skip(f"driver not present at {driver_path}")

    raw = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    report = evaluate_dispatch_protocol_complete(
        raw,
        repo_root=repo_root,
        recipe_path=recipe_path,
        trainer_path=trainer_path,
        remote_driver_path=driver_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is True, _blockers(report)
    # Sanity: the recipe declares dispatch_kind=tool.
    assert raw.get("dispatch_kind") == "tool"
