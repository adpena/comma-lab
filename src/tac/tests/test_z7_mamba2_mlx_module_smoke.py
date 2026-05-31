# SPDX-License-Identifier: MIT
"""Regression tests for the Z7-Mamba-2 trainable MLX module smoke surface."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="Z7-Mamba-2 MLX module smoke requires mlx on Apple Silicon",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAINER_PATH = _REPO_ROOT / "experiments" / (
    "train_substrate_time_traveler_l5_z7_mamba2_mlx_local.py"
)


def _load_z7_trainer_module():
    spec = importlib.util.spec_from_file_location("z7_mamba2_mlx_local_trainer", _TRAINER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_z7_mamba2_mlx_module_forward_shape_and_false_authority() -> None:
    """The trainable MLX module reconstructs pairs but carries no score authority."""
    import mlx.core as mx

    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_module import (
        MLX_EVIDENCE_GRADE,
        SCHEMA_VERSION,
        Z7Mamba2MLXModule,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXRenderConfig,
    )

    cfg = Z7Mamba2MLXRenderConfig(num_pairs=2)
    model = Z7Mamba2MLXModule(cfg, seed=0)
    rgb_0, rgb_1, latents = model(np.array([0, 1], dtype=np.int64))
    mx.eval(rgb_0, rgb_1, latents)

    assert SCHEMA_VERSION == "z7_mamba2_mlx_module_v1_20260528"
    assert MLX_EVIDENCE_GRADE == "[macOS-MLX research-signal]"
    assert tuple(int(x) for x in rgb_0.shape) == (2, 3, 384, 512)
    assert tuple(int(x) for x in rgb_1.shape) == (2, 3, 384, 512)
    assert tuple(int(x) for x in latents.shape) == (2, 24)
    assert model.num_parameters() > 0
    assert cfg.mamba2_mlx_backend_lineage == "reference_s6_mlx"
    assert cfg.canonical_ssd_mlx_backend_wired is False
    assert cfg.canonical_ssd_mlx_blocker == "canonical_ssd_mlx_backend_not_wired"


def test_z7_mamba2_mlx_canonical_ssd_backend_uses_helper_and_exports_bridge(
    tmp_path: Path,
) -> None:
    """SSD opt-in executes real recurrence and exports receiver-consumed weights."""
    import mlx.core as mx

    from tac.substrates._shared import mamba2_ssd
    from tac.substrates.time_traveler_l5_z7_mamba2.archive import (
        parse_archive,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.archive_candidate import (
        Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        Z7_MAMBA2_MLX_ARCHIVE_CANDIDATE_FAMILY,
        Z7_MAMBA2_MLX_ARCHIVE_TRANSFORM_KIND,
        export_z7_mamba2_mlx_archive,
        pack_archive_from_exported_state_dict,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.inflate import inflate_one_video
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_module import (
        Z7Mamba2MLXModule,
    )
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXNativeRenderer,
        Z7Mamba2MLXRenderConfig,
    )

    cfg = Z7Mamba2MLXRenderConfig(
        num_pairs=2,
        d_model=8,
        expand=2,
        d_state=4,
        use_canonical_ssd_mlx_backend=True,
        ssd_nheads=2,
    )
    assert cfg.canonical_ssd_mlx_backend_wired is True
    assert cfg.canonical_ssd_mlx_blocker == (
        "canonical_ssd_mlx_exact_cpu_cuda_replay_required"
    )

    with pytest.raises(NotImplementedError, match="Z7Mamba2MLXModule"):
        Z7Mamba2MLXNativeRenderer(cfg, seed=0)

    model = Z7Mamba2MLXModule(cfg, seed=0)
    with mock.patch(
        "tac.substrates._shared.mamba2_ssd.mamba2_ssd_step_mlx",
        wraps=mamba2_ssd.mamba2_ssd_step_mlx,
    ) as step_spy:
        rgb_0, rgb_1, latents = model(np.array([0, 1], dtype=np.int64))
        mx.eval(rgb_0, rgb_1, latents)

    assert step_spy.called
    assert tuple(int(x) for x in rgb_0.shape) == (2, 3, 384, 512)
    assert tuple(int(x) for x in rgb_1.shape) == (2, 3, 384, 512)
    assert tuple(int(x) for x in latents.shape) == (2, 24)
    assert model.num_parameters() > 0

    exported = model.export_state_dict()
    assert {
        "predictor.mamba_cell.A_log",
        "predictor.mamba_cell.B_proj.weight",
        "predictor.mamba_cell.C_proj.weight",
        "predictor.mamba_cell.dt_proj.weight",
        "predictor.mamba_cell.dt_proj.bias",
        "predictor.mamba_cell.D",
    } <= set(exported)
    blob = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        mlx_cfg=cfg,
    )
    archive = parse_archive(blob)
    assert archive.config.backend == "ssd_reference"
    assert archive.config.ssd_nheads == 2
    assert archive.config.ssd_headdim == 8
    authority = archive.meta["z7_mamba2_recurrent_predictive_coding_meta"]
    assert authority["ready_for_exact_eval_dispatch"] is False
    assert "canonical_ssd_mlx_exact_cpu_cuda_replay_required" in authority["blockers"]
    out_path = tmp_path / "z7_ssd_mlx_bridge_smoke.raw"
    frames = inflate_one_video(blob, out_path)
    assert frames == cfg.num_pairs * 2
    assert out_path.stat().st_size > 0

    archive_zip, archive_sha, archive_size = export_z7_mamba2_mlx_archive(
        model,
        tmp_path / "z7_ssd_mlx_bridge_export",
    )
    assert archive_zip.is_file()
    assert len(archive_sha) == 64
    assert archive_size == archive_zip.stat().st_size

    submission_dir = archive_zip.parent / "submission"
    adapter_package_path = archive_zip.parent / "archive_bound_candidate_adapter_package.json"
    assert adapter_package_path.is_file()
    adapter_package = json.loads(adapter_package_path.read_text(encoding="utf-8"))
    assert adapter_package["schema"] == Z7_MAMBA2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA
    receiver_proof = adapter_package["receiver_proof"]
    assert receiver_proof["runtime_consumption_proof_ready"] is True
    assert receiver_proof["receiver_contract_satisfied"] is True
    assert receiver_proof["receiver_output_retained"] is False
    assert receiver_proof["receiver_output_sha256"]
    assert not (archive_zip.parent / "receiver_proof" / "runtime_out" / "0").exists()
    shared_package = adapter_package["archive_bound_candidate_adapter_package"]
    assert shared_package["candidate_family"] == Z7_MAMBA2_MLX_ARCHIVE_CANDIDATE_FAMILY
    assert shared_package["ready_contract_count"] == 1
    assert shared_package["receiver_proof_gate_passed_count"] == 1
    assert shared_package["mlx_triage_ready_count"] == 0
    contract = shared_package["candidate_rows"][0]["archive_bound_candidate_contract"]
    assert contract["archive_native_transform_kind"] == Z7_MAMBA2_MLX_ARCHIVE_TRANSFORM_KIND
    assert contract["entropy_position_label"] == "before_entropy_coder"
    assert "z7_mamba2" in contract["archive_substrate_tags"]
    assert "mlx_substrate" in contract["archive_substrate_tags"]
    assert contract["runtime_adapter_ready"] is True
    assert contract["contest_runtime_decoder_adapter_ready"] is True
    assert contract["archive_bound_candidate_ready_for_exact_handoff"] is True
    assert contract["ready_for_exact_eval_dispatch"] is False
    exact_blockers = shared_package["exact_axis_blockers"][0]["blockers"]
    assert "canonical_ssd_mlx_exact_cpu_cuda_replay_required" in exact_blockers
    assert "contest_cpu_or_cuda_authority_required" in exact_blockers
    assert "lane_preclaim_required_before_exact_eval" in exact_blockers

    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")
    runtime_out = tmp_path / "runtime_out"
    result = subprocess.run(
        [
            str(submission_dir / "inflate.sh"),
            str(submission_dir),
            str(runtime_out),
            str(file_list),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHON": sys.executable},
        text=True,
    )
    assert result.stderr == ""
    runtime_raw = runtime_out / "0"
    assert runtime_raw.is_file()
    assert runtime_raw.stat().st_size == out_path.stat().st_size


def test_z7_mamba2_mlx_smoke_manifest_uses_trainable_module(tmp_path: Path) -> None:
    """The operator smoke exercises Z7Mamba2MLXModule, not the old native-only renderer."""
    output_dir = tmp_path / "z7_mlx_smoke"
    resolved = str(output_dir.resolve())
    if resolved.startswith(("/tmp/", "/private/tmp/")):
        pytest.skip("smoke CLI intentionally refuses persisted /tmp artifacts")

    trainer = _load_z7_trainer_module()

    parser = trainer._build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--num-pairs",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert trainer._smoke_main(args) == 0

    manifest = json.loads((output_dir / "smoke_manifest.json").read_text())
    assert manifest["renderer_module"].endswith(".mlx_module")
    assert manifest["renderer_num_parameters"] > 0
    assert manifest["score_claim"] is False
    assert manifest["promotable"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["mamba2_mlx_backend_lineage"] == "reference_s6_mlx"
    assert manifest["canonical_ssd_mlx_backend_wired"] is False
    assert manifest["canonical_ssd_mlx_blocker"] == (
        "canonical_ssd_mlx_backend_not_wired"
    )
    assert "canonical_ssd_mlx_backend_not_wired" in manifest["backend_claim_blockers"]
    assert "BLOCKED pending" not in manifest["canonical_provenance"]["rationale"]
    assert manifest["forward_smoke"]["latents_shape"] == [2, 24]


def test_z7_mamba2_mlx_smoke_manifest_can_use_canonical_ssd_backend(
    tmp_path: Path,
) -> None:
    """Smoke CLI records SSD recurrence provenance while staying non-promotable."""
    output_dir = tmp_path / "z7_mlx_ssd_smoke"
    resolved = str(output_dir.resolve())
    if resolved.startswith(("/tmp/", "/private/tmp/")):
        pytest.skip("smoke CLI intentionally refuses persisted /tmp artifacts")

    trainer = _load_z7_trainer_module()

    parser = trainer._build_parser()
    args = parser.parse_args(
        [
            "--smoke",
            "--num-pairs",
            "2",
            "--d-model",
            "8",
            "--expand",
            "2",
            "--d-state",
            "4",
            "--use-canonical-ssd-mlx-backend",
            "--ssd-nheads",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert trainer._smoke_main(args) == 0

    manifest = json.loads((output_dir / "smoke_manifest.json").read_text())
    assert manifest["use_canonical_ssd_mlx_backend"] is True
    assert manifest["ssd_nheads"] == 2
    assert manifest["ssd_headdim"] == 8
    assert manifest["mamba2_mlx_backend_lineage"] == (
        "canonical_mamba2_ssd_mlx_z7_gated_experimental"
    )
    assert manifest["canonical_ssd_mlx_backend_wired"] is True
    assert manifest["canonical_ssd_mlx_blocker"] == (
        "canonical_ssd_mlx_exact_cpu_cuda_replay_required"
    )
    assert manifest["ready_for_exact_eval_dispatch"] is False
