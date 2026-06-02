# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tac.local_acceleration.pr95_hnerv_mlx import (  # noqa: E402
    _expected_pr95_state_shapes,
    parse_pr95_public_archive_zip,
    write_pr95_public_archive_zip,
)
from tac.repo_io import write_json  # noqa: E402
from tools.profile_pr95_hnerv_mlx_section_value import (  # noqa: E402
    _load_external_baseline_reuse,
    _materialize_variants,
)


def test_pr95_section_value_materializes_decoder_and_latent_neutralizations(
    tmp_path: Path,
) -> None:
    archive = _write_pr95_archive(tmp_path / "archive.zip")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "inflate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    packet = parse_pr95_public_archive_zip(archive)

    variants, absent = _materialize_variants(
        archive=archive,
        packet=packet,
        submission_dir=runtime,
        output_dir=tmp_path / "profile",
        requested_sections=["decoder_qw", "latents_rc", "residual_rc"],
    )

    by_id = {variant.variant_id: variant for variant in variants}
    assert set(by_id) == {"baseline", "neutralize_decoder_qw", "neutralize_latents_rc"}
    assert absent[0]["section"] == "residual_rc"
    baseline = parse_pr95_public_archive_zip(by_id["baseline"].archive_zip_path)
    decoder_zero = parse_pr95_public_archive_zip(
        by_id["neutralize_decoder_qw"].archive_zip_path
    )
    latents_zero = parse_pr95_public_archive_zip(
        by_id["neutralize_latents_rc"].archive_zip_path
    )
    assert baseline.archive_zip_sha256 == packet.archive_zip_sha256
    assert all(float(np.max(np.abs(value))) == 0.0 for value in decoder_zero.state_dict.values())
    assert float(np.max(np.abs(decoder_zero.latents))) > 0.0
    assert float(np.max(np.abs(latents_zero.latents))) == 0.0
    assert any(float(np.max(np.abs(value))) > 0.0 for value in latents_zero.state_dict.values())
    assert by_id["neutralize_decoder_qw"].submission_dir == runtime
    assert by_id["neutralize_latents_rc"].submission_dir == runtime


def test_pr95_section_value_accepts_hash_matched_external_baseline_reuse(
    tmp_path: Path,
) -> None:
    archive = _write_pr95_archive(tmp_path / "archive.zip")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    packet = parse_pr95_public_archive_zip(archive)
    variants, _ = _materialize_variants(
        archive=archive,
        packet=packet,
        submission_dir=runtime,
        output_dir=tmp_path / "profile",
        requested_sections=[],
    )
    baseline = variants[0]
    cache_dir, cache_report, response = _write_external_baseline_reuse_files(
        tmp_path=tmp_path,
        baseline=baseline,
    )

    reuse = _load_external_baseline_reuse(
        baseline_variant=baseline,
        cache_dir=cache_dir,
        cache_report=cache_report,
        mlx_response=response,
        output_dir=tmp_path / "profile",
        max_pairs=2,
        scorer_batch_pairs=1,
    )

    assert reuse["metadata"]["status"] == "accepted_archive_hash_and_pair_shape_match"
    assert reuse["cache_row"]["reused_external_baseline_cache_report"] is True
    assert reuse["payload"]["archive_sha256"] == baseline.archive_sha256
    assert (tmp_path / "profile" / "mlx_responses" / "baseline.json").is_file()


def test_pr95_section_value_rejects_external_baseline_hash_mismatch(
    tmp_path: Path,
) -> None:
    archive = _write_pr95_archive(tmp_path / "archive.zip")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    packet = parse_pr95_public_archive_zip(archive)
    variants, _ = _materialize_variants(
        archive=archive,
        packet=packet,
        submission_dir=runtime,
        output_dir=tmp_path / "profile",
        requested_sections=[],
    )
    baseline = variants[0]
    cache_dir, cache_report, response = _write_external_baseline_reuse_files(
        tmp_path=tmp_path,
        baseline=baseline,
    )
    payload = {
        **_baseline_response_payload(baseline),
        "archive_sha256": "0" * 64,
    }
    write_json(response, payload)

    with pytest.raises(ValueError, match="archive sha256 mismatch"):
        _load_external_baseline_reuse(
            baseline_variant=baseline,
            cache_dir=cache_dir,
            cache_report=cache_report,
            mlx_response=response,
            output_dir=tmp_path / "profile",
            max_pairs=2,
            scorer_batch_pairs=1,
        )


def _write_pr95_archive(path: Path) -> Path:
    latent_dim = 3
    base_channels = 8
    shapes = _expected_pr95_state_shapes(
        latent_dim=latent_dim,
        base_channels=base_channels,
    )
    state = {
        name: np.full(shape, fill_value=(index + 1) / 100.0, dtype=np.float32)
        for index, (name, shape) in enumerate(shapes.items())
    }
    latents = np.asarray(
        [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]],
        dtype=np.float32,
    )
    write_pr95_public_archive_zip(
        state,
        latents,
        meta={
            "latent_dim": latent_dim,
            "base_channels": base_channels,
            "eval_size": [384, 512],
        },
        output_zip_path=path,
    )
    return path


def _write_external_baseline_reuse_files(
    *,
    tmp_path: Path,
    baseline: object,
) -> tuple[Path, Path, Path]:
    cache_dir = tmp_path / "external_cache"
    cache_dir.mkdir()
    manifest = cache_dir / "manifest.json"
    write_json(
        manifest,
        {
            "archive_sha256": baseline.archive_sha256,
            "pair_count": 2,
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    cache_report = tmp_path / "external_cache_report.json"
    write_json(
        cache_report,
        {
            "archive": {
                "bytes": baseline.archive_bytes,
                "sha256": baseline.archive_sha256,
            },
            "cache_manifest": manifest.as_posix(),
            "output_cache_dir": cache_dir.as_posix(),
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
        },
    )
    response = tmp_path / "external_mlx_response.json"
    write_json(response, _baseline_response_payload(baseline))
    return cache_dir, cache_report, response


def _baseline_response_payload(baseline: object) -> dict[str, object]:
    return {
        "archive_size_bytes": baseline.archive_bytes,
        "archive_sha256": baseline.archive_sha256,
        "n_samples": 2,
        "candidate_cache_pairs": 2,
        "reference_cache_pairs": 2,
        "batch_pairs": 1,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
