# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from experiments.train_substrate_hi_nerv_mlx_local import (
    TRAINER_SCHEMA,
    _build_parser,
    _coder_qat_config_from_args,
    _config_from_args,
    _metadata_safe,
    _resolve_output_dir,
)


def test_hinerv_mlx_trainer_binds_modelsize_row_and_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--modelsize-row",
            "hi_nerv_local_small",
            "--num-pairs",
            "7",
            "--decoder-channels",
            "9,8,7,6,5,4,3",
            "--latent-dim-coarse",
            "11",
            "--output-height",
            "96",
            "--output-width",
            "128",
        ]
    )

    cfg = _config_from_args(args)

    assert cfg.num_pairs == 7
    assert cfg.latent_dim_coarse == 11
    assert cfg.latent_dim_mid == 15
    assert cfg.latent_dim_fine == 18
    assert cfg.embed_dim == 48
    assert cfg.decoder_channels == (9, 8, 7, 6, 5, 4, 3)
    assert cfg.output_height == 96
    assert cfg.output_width == 128


def test_hinerv_mlx_trainer_coder_qat_config_is_real_and_validated() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--coder-qat",
            "--coder-qat-bits",
            "4",
            "--coder-qat-quant-residual-weight",
            "0.25",
            "--coder-qat-magnitude-weight",
            "0.125",
            "--coder-qat-delta-weight",
            "0.0625",
        ]
    )

    cfg = _coder_qat_config_from_args(args)

    assert cfg.enabled is True
    assert cfg.quant_bits == 4
    assert cfg.quant_residual_weight == pytest.approx(0.25)
    assert cfg.magnitude_weight == pytest.approx(0.125)
    assert cfg.delta_weight == pytest.approx(0.0625)


def test_hinerv_mlx_trainer_rejects_local_output_without_opt_in(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args(
        ["--smoke", "--output-dir", str(tmp_path / "local")]
    )

    with pytest.raises(StorageTierError, match="local_disk_tier_disabled"):
        _resolve_output_dir(args)


def test_hinerv_mlx_trainer_allows_explicit_local_smoke_output(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path / "local"),
            "--allow-local-output-dir",
        ]
    )

    output, storage = _resolve_output_dir(args)

    assert output == (tmp_path / "local").resolve(strict=False)
    assert output.is_dir()
    assert storage["schema"] == "hi_nerv_mlx_trainer_explicit_output_preflight.v1"
    assert storage["score_claim"] is False
    assert storage["ready_for_exact_eval_dispatch"] is False


def test_hinerv_mlx_trainer_parser_requires_mode() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])

    assert TRAINER_SCHEMA == "hi_nerv_mlx_score_aware_trainer.v1"


def test_hinerv_mlx_trainer_metadata_safe_drops_nested_authority_keys() -> None:
    payload = {
        "storage": {
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "selected_workload_root": "/Volumes/VertigoDataTier/pact/x",
            "children": [{"rank_or_kill_eligible": False, "keep": "yes"}],
        },
        "keep_top": True,
    }

    safe = _metadata_safe(payload)

    assert "score_claim" not in safe["storage"]
    assert "ready_for_exact_eval_dispatch" not in safe["storage"]
    assert safe["storage"]["selected_workload_root"].endswith("/x")
    assert safe["storage"]["children"] == [{"keep": "yes"}]
    assert safe["keep_top"] is True
