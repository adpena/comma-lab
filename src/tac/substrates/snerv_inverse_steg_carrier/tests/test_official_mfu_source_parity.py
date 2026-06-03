# SPDX-License-Identifier: MIT
"""Source-backed proof for official SNeRV MFU parity status.

These tests intentionally avoid importing the official SNeRV torch modules:
the proof needed here is source/shape semantics, not heavyweight training.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_SPECTRA_PRESERVING_ADAPTER,
    MultiResolutionFusionUnit,
    SnervModelSizeConfig,
)

OFFICIAL_SNERV_SHA = "0844a08f9591eea9625f8b961ed91d08030e06d1"
DEFAULT_OFFICIAL_SNERV_REPO = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "oss_nerv_source_audit_20260602T113720Z/repos/SNeRV"
)


def _official_repo() -> Path:
    repo = Path(os.environ.get("PACT_SNERV_OFFICIAL_REPO", DEFAULT_OFFICIAL_SNERV_REPO))
    if not repo.exists():
        pytest.skip(f"official SNeRV checkout is absent: {repo}")
    result = subprocess.run(
        ["git", "-C", repo.as_posix(), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.strip() == OFFICIAL_SNERV_SHA
    return repo


def _assert_source_line(repo: Path, rel_path: str, line_no: int, snippet: str) -> None:
    path = repo / rel_path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[line_no - 1].strip() == snippet


def test_official_snerv_mfu_source_contract_is_pinned() -> None:
    """Official MFU is ConvTranspose2d -> concat -> RB at two decoder scales."""

    repo = _official_repo()
    _assert_source_line(
        repo,
        "model/snerv.py",
        68,
        "upsample_5 = nn.ConvTranspose2d(ngf_list[-3], ngf_list[-3], args.dec_strds[-2], args.dec_strds[-2], 0)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        69,
        "decoder_layer5 = RB(in_channels=ngf_list[-3]+ngf_list[-2], out_channels=ngf_list[-2], num_blocks=args.num_blocks)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        70,
        "upsample_6 = nn.ConvTranspose2d(ngf_list[-2], ngf_list[-2], args.dec_strds[-1], args.dec_strds[-1], 0)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        71,
        "decoder_layer6 = RB(in_channels=ngf_list[-2]+new_ngf, out_channels=new_ngf, num_blocks=args.num_blocks)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        104,
        "up1 = self.decoder[self.decoder_len+3](embed_list[-3])",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        106,
        "unet1 = self.decoder[self.decoder_len+4](torch.cat([up1, embed_list[-2]], dim=1))",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        108,
        "unet1_up =self.decoder[self.decoder_len+5](unet1)",
    )
    _assert_source_line(
        repo,
        "model/snerv.py",
        109,
        "pyr_out = self.decoder[self.decoder_len+6](torch.cat([unet1_up, embed_list[-1]], dim=1))",
    )


def test_local_receiver_safe_mfu_falsifies_official_mfu_parity() -> None:
    """Local MFU is executable, but it is not the official neural MFU block."""

    _official_repo()
    source = inspect.getsource(MultiResolutionFusionUnit.features)
    for official_token in (
        "ConvTranspose2d",
        "torch.cat",
        "embed_list",
        "decoder_len+3",
        "decoder_len+4",
        "decoder_len+5",
        "decoder_len+6",
    ):
        assert official_token not in source
    for local_token in (
        "_box_pool_upsample",
        "_central_gradients",
        "_patch_features",
        "_select_feature_bank",
    ):
        assert local_token in source

    field = np.arange(4 * 6, dtype=np.float64).reshape(4, 6)
    features = MultiResolutionFusionUnit(scales=(1, 2, 4)).features(
        field,
        feature_count=12,
        patch_radius=1,
    )
    cfg = SnervModelSizeConfig(
        fc_dim=12,
        adapter=SNERV_SPECTRA_PRESERVING_ADAPTER,
        mfu_scales=(1, 2, 4),
    )

    assert features.shape == (4, 6, 12)
    assert cfg.adapter == SNERV_SPECTRA_PRESERVING_ADAPTER
    assert cfg.feature_count == 12
