"""Driver-generator output is canonical, deterministic, and bash-syntax-valid."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tac.substrate_registry.contract import (
    NOT_APPLICABLE_WITH_RATIONALE,
    SubstrateContract,
)
from tac.substrate_registry.driver_generator import (
    default_driver_relpath,
    generate_driver_shell,
)


def _baseline_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "dgen_test",
        "lane_id": "lane_dgen_test_20260515",
        "target_modes": ("research_substrate",),
        "deployment_target": "desktop_research",
        "council_verdict_provenance": None,
        "archive_grammar": "g",
        "parser_section_manifest": {"h": "magic"},
        "inflate_runtime_loc_budget": 80,
        "runtime_dep_closure": ("torch",),
        "export_format": "fp16_brotli",
        "score_aware_loss": "scorer_loss_terms_btchw",
        "bolt_on_loc_budget": 200,
        "no_op_detector_planned": True,
        "archive_bytes_added": None,
        "score_improvement_mechanism_status": "RESEARCH_ONLY",
        "runtime_overlay_consumed": False,
        "recipe_smoke_only": True,
        "recipe_research_only": True,
        "recipe_min_smoke_gpu": "T4",
        "recipe_min_vram_gb": 16,
        "recipe_pyav_decode_strategy": "cpu_thread_async_upload",
        "recipe_canary_status": "independent_substrate",
        "recipe_video_input_strategy": "per_dispatch_local_copy",
        "recipe_canary_dependency": None,
        "cost_band_epochs": 10,
        "cost_band_gpu_key": "T4",
        "cost_band_platform_key": "modal",
        "cost_band_p50_usd": 0.10,
        "hook_sensitivity_contribution": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_pareto_constraint": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_bit_allocator_class": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_autopilot_ranker_class_shift_token": None,
        "hook_continual_learning_anchor_kind": NOT_APPLICABLE_WITH_RATIONALE,
        "hook_probe_disambiguator": None,
        "catalog_compliance_declarations": ("catalog_205_select_inflate_device_used",),
        "hook_not_applicable_rationale": {
            "hook_sensitivity_contribution": "test",
            "hook_pareto_constraint": "test",
            "hook_bit_allocator_class": "test",
            "hook_continual_learning_anchor_kind": "test",
            "hook_probe_disambiguator": "test",
        },
    }
    base.update(overrides)
    return base


def test_default_driver_relpath_canonical() -> None:
    c = SubstrateContract(**_baseline_kwargs(id="alpha"))
    assert default_driver_relpath(c) == "scripts/remote_lane_substrate_alpha.sh"


def test_generate_driver_shell_is_deterministic() -> None:
    c = SubstrateContract(**_baseline_kwargs())
    a = generate_driver_shell(c)
    b = generate_driver_shell(c)
    assert a == b


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_generate_driver_shell_passes_bash_n() -> None:
    """Generated bash MUST pass syntax check (bash -n)."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert r.returncode == 0, f"bash -n failed:\nstdout:{r.stdout}\nstderr:{r.stderr}"
    finally:
        Path(path).unlink(missing_ok=True)


def test_generate_driver_shell_has_set_e_pipefail() -> None:
    """Per Catalog #2 (`check_shell_set_e_present`)."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert "set -euo pipefail" in src


def test_generate_driver_shell_has_canonical_bootstrap_sentinel() -> None:
    """Per Catalog #163 (`check_remote_lane_script_uses_sentinel_when_sourcing_bootstrap`)."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert "REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source" in src


def test_generate_driver_shell_has_canonical_modal_cuda_env_block() -> None:
    """Catalog #224/#244: generated drivers carry the shared Modal/CUDA env block."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert 'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in src
    assert 'export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"' in src
    assert (
        'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
        in src
    )


def test_generate_driver_shell_has_5min_heartbeat() -> None:
    """Per CLAUDE.md "Remote code parity"."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert "sleep 300" in src
    assert "heartbeat.log" in src


def test_generate_driver_shell_has_dispatch_claim_verification() -> None:
    """Per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION"."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert "DISPATCH_INSTANCE_JOB_ID" in src
    assert "claim_lane_dispatch.py" in src
    assert "completed_" in src
    assert "failed_" in src


def test_generate_driver_shell_has_modal_results_path_remap() -> None:
    """Per Catalog #204 (durable Modal output)."""
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert "/modal_results/" in src
    assert 'MODAL_RUNTIME:-0' in src


def test_generate_driver_shell_has_auth_eval_validation_gate() -> None:
    """Per CLAUDE.md "Auth eval EVERYWHERE" + Catalog #226."""
    c = SubstrateContract(**_baseline_kwargs(id="alpha"))
    src = generate_driver_shell(c)
    assert "auth_eval_score_claim_valid" in src
    assert "contest_cuda" in src
    assert "auth_eval_exact_cuda_complete" in src


def test_generate_driver_shell_emits_completion_marker() -> None:
    """Marker form: ``LANE_<ID>_DONE [contest-CUDA] ...``."""
    c = SubstrateContract(**_baseline_kwargs(id="myid"))
    src = generate_driver_shell(c)
    assert "LANE_MYID_DONE" in src
    assert "[contest-CUDA]" in src


def test_generate_driver_shell_has_provenance_emit() -> None:
    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    assert "PROVENANCE=" in src
    assert "started_at_utc" in src


def test_generate_driver_shell_id_drives_env_var_prefix() -> None:
    c = SubstrateContract(**_baseline_kwargs(id="my_lane_xyz"))
    src = generate_driver_shell(c)
    assert "MY_LANE_XYZ_VIDEO_PATH" in src
    assert "MY_LANE_XYZ_OUTPUT_DIR" in src
    assert "MY_LANE_XYZ_EPOCHS" in src


def test_generate_driver_shell_no_runtime_clock_drift() -> None:
    """Generated bash must not embed clock/host/random values."""
    import datetime
    import re

    c = SubstrateContract(**_baseline_kwargs())
    src = generate_driver_shell(c)
    iso_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    matches = iso_pattern.findall(src)
    # Allow none — runtime $(date -u +%FT%TZ) is a shell invocation, not a literal.
    assert not matches, f"generated bash embeds literal ISO timestamp: {matches}"
    today = datetime.datetime.utcnow().strftime("%Y%m%d")
    extras = [m for m in re.findall(r"\b\d{8}\b", src) if m != "20260515"]
    assert today not in extras


def test_generate_driver_shell_references_recipe_path() -> None:
    """The driver provenance section should reference the recipe path."""
    c = SubstrateContract(**_baseline_kwargs(id="paramref"))
    src = generate_driver_shell(c)
    assert "substrate_paramref_modal_t4_dispatch.yaml" in src
    assert "experiments/train_substrate_paramref.py" in src
