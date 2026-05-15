# SPDX-License-Identifier: MIT
"""Tests for the dispatch_protocol_complete umbrella gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.deploy.dispatch_protocol import (
    DispatchProtocolError,
    evaluate_dispatch_protocol_complete,
    require_dispatch_protocol_complete,
)


def _write_clean_surfaces(tmp_path: Path) -> dict[str, object]:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "scripts").mkdir()
    trainer = tmp_path / "experiments/train_substrate_alpha.py"
    trainer.write_text(
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
                "def train():",
                "    with torch.no_grad():",
                "        pass",
                "def main():",
                "    gate_auth_eval_call([])",
                "    return 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    driver = tmp_path / "scripts/remote_lane_substrate_alpha.sh"
    driver.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                'export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"',
                'export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"',
                'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"',
                "python experiments/train_substrate_alpha.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recipe: dict[str, object] = {
        "name": "substrate_alpha_modal_t4_dispatch",
        "lane_id": "lane_alpha_20260515",
        "dispatch_enabled": True,
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
    return {"recipe": recipe, "trainer": trainer, "driver": driver}


def _blockers(report) -> str:
    return "\n".join(report.blockers)


def test_dispatch_protocol_clean_recipe_passes(tmp_path: Path) -> None:
    surfaces = _write_clean_surfaces(tmp_path)

    report = evaluate_dispatch_protocol_complete(
        surfaces["recipe"],
        repo_root=tmp_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is True
    assert [tier.passed for tier in report.tiers] == [True, True, True]
    payload = report.to_dict()
    assert payload["schema"] == "pact.dispatch_protocol_complete.v1"
    assert payload["dispatch_protocol_complete"] is True


def test_dispatch_protocol_blocks_missing_modal_env_hygiene(tmp_path: Path) -> None:
    surfaces = _write_clean_surfaces(tmp_path)
    driver = surfaces["driver"]
    assert isinstance(driver, Path)
    driver.write_text("#!/bin/bash\nset -euo pipefail\necho missing env\n")

    report = evaluate_dispatch_protocol_complete(
        surfaces["recipe"],
        repo_root=tmp_path,
        native_dispatch=True,
    )

    assert report.dispatch_protocol_complete is False
    assert "catalog_244_modal_env_hygiene_missing" in _blockers(report)
    assert report.tiers[1].name == "tier2_hardware_correctness"
    assert report.tiers[1].passed is False


def test_dispatch_protocol_blocks_missing_trainer_tier_flags(tmp_path: Path) -> None:
    surfaces = _write_clean_surfaces(tmp_path)
    trainer = surfaces["trainer"]
    assert isinstance(trainer, Path)
    trainer.write_text(
        "import torch\n"
        "def main():\n"
        "    with torch.no_grad():\n"
        "        return 0\n",
        encoding="utf-8",
    )

    report = evaluate_dispatch_protocol_complete(
        surfaces["recipe"],
        repo_root=tmp_path,
        native_dispatch=True,
    )

    blockers = _blockers(report)
    assert report.dispatch_protocol_complete is False
    assert "catalog_172_autocast_fp16_missing_or_unwaived" in blockers
    assert "catalog_178_tf32_missing_or_unwaived" in blockers
    assert "catalog_179_torch_compile_missing_or_unwaived" in blockers
    assert "catalog_226_auth_eval_canonical_helper_missing" in blockers


def test_require_dispatch_protocol_raises_before_dispatch(tmp_path: Path) -> None:
    surfaces = _write_clean_surfaces(tmp_path)
    recipe = dict(surfaces["recipe"])
    recipe["dispatch_blockers"] = ["measured_config_regression"]

    with pytest.raises(DispatchProtocolError, match="dispatch_protocol_complete=false"):
        require_dispatch_protocol_complete(
            recipe,
            repo_root=tmp_path,
            native_dispatch=True,
        )


def test_non_native_dispatch_is_non_blocking_plan_surface(tmp_path: Path) -> None:
    report = evaluate_dispatch_protocol_complete(
        {"name": "release_notes_only", "platform": "none"},
        repo_root=tmp_path,
        native_dispatch=False,
    )

    assert report.dispatch_protocol_complete is True
    assert report.native_dispatch is False
    assert report.tiers[0].name == "non_native_dispatch"
