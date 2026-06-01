# SPDX-License-Identifier: MIT
"""Tests for compact renderer packet-spine adapters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.substrates.hprc.representation_spine import (  # noqa: E402
    HprcRepresentationFamily,
)
from tools.emit_compact_renderer_spine_adapter import (  # noqa: E402
    COMPACT_RENDERER_SPINE_ADAPTER_SCHEMA,
    CompactRendererSpineAdapterError,
    emit_compact_renderer_spine_adapter,
)


def _blob(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_emit_boostnerv_spine_requires_trained_provenance(tmp_path: Path) -> None:
    decoder = _blob(tmp_path / "decoder.bin", b"trained-weights")
    latents = _blob(tmp_path / "latents.bin", b"trained-latents")

    with pytest.raises(
        CompactRendererSpineAdapterError,
        match="trained_weights_provenance_required",
    ):
        emit_compact_renderer_spine_adapter(
            family="boostnerv",
            output_dir=tmp_path / "out",
            decoder_blob=decoder,
            latents_blob=latents,
            trained_weights_provenance="",
            trained_latents_provenance="stage8-smoke",
        )


def test_emit_rt_vq_nerv_spine_blocks_promotion_until_receiver_and_exact(
    tmp_path: Path,
) -> None:
    decoder = _blob(tmp_path / "decoder.bin", b"trained-rt-vq-decoder")
    latents = _blob(tmp_path / "tokens.bin", b"\x00\x01\x02\x03")
    codebooks = _blob(tmp_path / "codebooks.bin", b"codebook")
    selectors = _blob(tmp_path / "selectors.bin", b"selector")

    report = emit_compact_renderer_spine_adapter(
        family="rt_vq_nerv",
        output_dir=tmp_path / "out",
        decoder_blob=decoder,
        latents_blob=latents,
        codebooks_blob=codebooks,
        selectors_blob=selectors,
        trained_weights_provenance="unit-test-trained-weights",
        trained_latents_provenance="unit-test-trained-tokens",
    )

    assert report["schema"] == COMPACT_RENDERER_SPINE_ADAPTER_SCHEMA
    assert report["family"] == HprcRepresentationFamily.RT_VQ_NERV.value
    assert Path(report["projection"]["hprc_bin_path"]).is_file()
    assert Path(report["projection"]["manifest_path"]).is_file()
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "archive_zip_runtime_receiver_proof_not_yet_emitted" in report[
        "exact_gate"
    ]["blockers"]
    section_roles = {row["role"] for row in report["source_artifacts"]}
    assert "trained_decoder_weights_or_program" in section_roles
    assert "trained_latents_or_tokens" in section_roles
    assert "codebooks" in section_roles
    assert "selectors" in section_roles


def test_emit_pr95_hnerv_alias_projects_control_base(tmp_path: Path) -> None:
    decoder = _blob(tmp_path / "decoder.pt", b"trained-pr95-mlx-weights")
    latents = _blob(tmp_path / "latents.npy", b"trained-pr95-mlx-latents")

    report = emit_compact_renderer_spine_adapter(
        family="pr95_hnerv",
        output_dir=tmp_path / "out",
        decoder_blob=decoder,
        latents_blob=latents,
        trained_weights_provenance="mlx-smoke-checkpoint",
        trained_latents_provenance="mlx-smoke-latents",
    )

    assert report["family"] == HprcRepresentationFamily.PR95_HNERV.value
    assert report["score_claim"] is False
    assert Path(report["projection"]["manifest_path"]).is_file()
