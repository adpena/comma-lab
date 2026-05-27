# SPDX-License-Identifier: MIT
"""Regression tests for PR95 MLX -> PyTorch -> contest archive packaging."""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.local_acceleration.pr95_hnerv_mlx import (
    HNeRVSyntheticTrainingBundleMLX,
    pytorch_state_dict_from_mlx,
    write_pr95_public_archive_zip,
)
from tools.package_pr95_mlx_pytorch_state_dict_to_contest_archive import (
    PR95_MLX_PACKAGE_SCHEMA,
    Pr95MlxPackageError,
    package_pytorch_state_dict_to_contest_archive,
)


def test_package_pr95_mlx_pytorch_state_dict_byte_closed_runtime_smoke(
    tmp_path: Path,
) -> None:
    import torch

    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=8,
        seed=17,
    )
    source_archive_zip = tmp_path / "source_archive.zip"
    source_state_dict = pytorch_state_dict_from_mlx(bundle.decoder)
    write_pr95_public_archive_zip(
        source_state_dict,
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=source_archive_zip,
    )

    input_pt = tmp_path / "state_dict.pt"
    torch.save(pytorch_state_dict_from_mlx(bundle.decoder, as_torch=True), input_pt)
    submission_dir = tmp_path / "submission"
    report_path = tmp_path / "package_report.json"

    report = package_pytorch_state_dict_to_contest_archive(
        input_pt=input_pt,
        source_archive_zip=source_archive_zip,
        output_submission_dir=submission_dir,
        report_out=report_path,
    )

    assert report["schema_version"] == PR95_MLX_PACKAGE_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["exact_readiness_refusal"]["ready"] is False
    assert (submission_dir / "archive.zip").is_file()
    assert (submission_dir / "inflate.sh").is_file()
    assert (submission_dir / "inflate.py").is_file()
    assert (submission_dir / "src" / "model.py").is_file()
    assert (submission_dir / "src" / "codec.py").is_file()
    assert (submission_dir / "archive_manifest.json").is_file()
    assert json.loads(report_path.read_text())["archive_zip_sha256"] == report[
        "archive_zip_sha256"
    ]

    archive_dir = tmp_path / "archive_dir"
    archive_dir.mkdir()
    # The contest archive is a ZIP. For inflate.sh's archive_dir contract, extract
    # the single member into archive_dir/0.bin.
    with zipfile.ZipFile(submission_dir / "archive.zip") as zf:
        (archive_dir / "0.bin").write_bytes(zf.read("0.bin"))

    file_list = tmp_path / "file_list.txt"
    file_list.write_text("0.mp4\n")
    output_dir = tmp_path / "inflated"
    proc = subprocess.run(
        [
            str(submission_dir / "inflate.sh"),
            str(archive_dir),
            str(output_dir),
            str(file_list),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={
            **os.environ,
            "PYTHON": str(Path(__file__).resolve().parents[3] / ".venv/bin/python"),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    raw = output_dir / "0.raw"
    assert raw.is_file()
    assert raw.stat().st_size == 2 * 874 * 1164 * 3


def test_package_pr95_latents_from_pt_fails_before_output_on_count_mismatch(
    tmp_path: Path,
) -> None:
    import torch

    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=3,
        latent_dim=28,
        base_channels=8,
        seed=23,
    )
    source_archive_zip = tmp_path / "source_archive.zip"
    write_pr95_public_archive_zip(
        pytorch_state_dict_from_mlx(bundle.decoder),
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=source_archive_zip,
    )

    input_pt = tmp_path / "state_dict_with_short_latents.pt"
    state_dict = pytorch_state_dict_from_mlx(bundle.decoder, as_torch=True)
    state_dict["latents"] = torch.zeros((1, 28), dtype=torch.float32)
    torch.save(state_dict, input_pt)

    submission_dir = tmp_path / "submission"
    with pytest.raises(Pr95MlxPackageError, match="checkpoint_latent_count_mismatch"):
        package_pytorch_state_dict_to_contest_archive(
            input_pt=input_pt,
            source_archive_zip=source_archive_zip,
            output_submission_dir=submission_dir,
            latents_from_pt=True,
        )

    assert not submission_dir.exists()


def test_package_pr95_latents_from_pt_accepts_matching_trained_latents(
    tmp_path: Path,
) -> None:
    import torch

    bundle = HNeRVSyntheticTrainingBundleMLX(
        latent_count=2,
        latent_dim=28,
        base_channels=8,
        seed=31,
    )
    source_archive_zip = tmp_path / "source_archive.zip"
    canonical_state_dict = pytorch_state_dict_from_mlx(bundle.decoder)
    write_pr95_public_archive_zip(
        canonical_state_dict,
        bundle.latents,
        meta={"latent_dim": 28, "base_channels": 8, "eval_size": [384, 512]},
        output_zip_path=source_archive_zip,
    )

    input_pt = tmp_path / "state_dict_with_matching_latents.pt"
    state_dict = pytorch_state_dict_from_mlx(bundle.decoder, as_torch=True)
    state_dict["latents"] = torch.from_numpy(
        np.asarray(bundle.latents).astype("float32", copy=True)
    )
    torch.save(state_dict, input_pt)

    submission_dir = tmp_path / "submission"
    report = package_pytorch_state_dict_to_contest_archive(
        input_pt=input_pt,
        source_archive_zip=source_archive_zip,
        output_submission_dir=submission_dir,
        latents_from_pt=True,
    )

    assert report["archive_zip_bytes"] > 0
    assert report["latent_shape"] == [2, 28]
    assert report["parsed_latent_shape_after_roundtrip"] == [2, 28]
    assert report["ready_for_exact_eval_dispatch"] is False
    assert (submission_dir / "archive.zip").is_file()
