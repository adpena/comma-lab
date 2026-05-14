# SPDX-License-Identifier: MIT
"""Roundtrip and custody tests for the DPW1 scaffold."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

from tac.substrates.driving_prior_world_model import (
    DPWM_MAGIC,
    DrivingPriorWorldModelConfig,
    DrivingPriorWorldModelError,
    apply_world_model_archive,
    build_readiness_manifest,
    deterministic_prior_weights,
    deterministic_residual_bytes,
    expected_prior_weight_bytes,
    expected_residual_bytes,
    pack_archive,
    parse_archive,
)


def _cfg() -> DrivingPriorWorldModelConfig:
    return DrivingPriorWorldModelConfig(
        output_height=6,
        output_width=8,
        num_pairs=3,
        codebook_entries=5,
        residual_grid_height=3,
        residual_grid_width=4,
    )


def test_deterministic_archive_roundtrip_and_apply() -> None:
    cfg = _cfg()
    prior = deterministic_prior_weights(cfg, seed=2032)
    residual = deterministic_residual_bytes(cfg, seed=2032, max_abs_delta=2)
    blob_a = pack_archive(cfg, prior, residual, metadata={"test_vector": "dpw1"})
    blob_b = pack_archive(cfg, prior, residual, metadata={"test_vector": "dpw1"})

    assert blob_a == blob_b
    assert blob_a[:4] == DPWM_MAGIC
    parsed = parse_archive(blob_a)
    assert parsed.config == cfg
    assert parsed.prior_weights == prior
    assert parsed.residual_bytes == residual
    assert parsed.metadata["score_claim"] is False
    assert parsed.metadata["ready_for_exact_eval_dispatch"] is False
    assert parsed.metadata["research_only"] is True
    assert parsed.metadata["scorer_at_inflate"] is False

    repacked = pack_archive(
        parsed.config,
        parsed.prior_weights,
        parsed.residual_bytes,
        metadata=parsed.metadata,
    )
    assert repacked == blob_a

    frames_a = apply_world_model_archive(parsed, pair_indices=[0, 2])
    frames_b = apply_world_model_archive(blob_a, pair_indices=[0, 2])
    assert frames_a.shape == (2, 2, cfg.output_height, cfg.output_width, 3)
    assert frames_a.dtype.name == "uint8"
    assert (frames_a == frames_b).all()


def test_charged_sections_are_consumed_and_trailing_bytes_rejected() -> None:
    cfg = _cfg()
    prior = bytearray(deterministic_prior_weights(cfg))
    residual = bytearray(bytes(expected_residual_bytes(cfg)))
    blob_a = pack_archive(cfg, bytes(prior), bytes(residual))
    frames_a = apply_world_model_archive(blob_a, pair_indices=[0])

    prior[0] = (prior[0] + 19) % 256
    blob_prior = pack_archive(cfg, bytes(prior), bytes(residual))
    frames_prior = apply_world_model_archive(blob_prior, pair_indices=[0])
    assert not (frames_a == frames_prior).all()

    residual[0] = 7
    blob_residual = pack_archive(cfg, bytes(prior), bytes(residual))
    frames_residual = apply_world_model_archive(blob_residual, pair_indices=[0])
    assert not (frames_prior == frames_residual).all()

    parsed = parse_archive(blob_residual)
    manifest = parsed.section_manifest()
    assert manifest[0]["name"] == "prior_weights"
    assert manifest[0]["length"] == expected_prior_weight_bytes(cfg)
    assert manifest[1]["name"] == "residual_bytes"
    assert manifest[1]["length"] == expected_residual_bytes(cfg)

    with pytest.raises(DrivingPriorWorldModelError, match="trailing bytes"):
        parse_archive(blob_residual + b"x")


def test_structural_noop_is_blocked_and_not_dispatch_ready() -> None:
    cfg = _cfg()
    blob = pack_archive(
        cfg,
        bytes(expected_prior_weight_bytes(cfg)),
        bytes(expected_residual_bytes(cfg)),
    )
    parsed = parse_archive(blob)
    assert parsed.structural_noop is True

    manifest = build_readiness_manifest(parsed)
    assert manifest["structural_noop"] is True
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "driving_prior_world_model_structural_noop" in manifest["dispatch_blockers"]

    with pytest.raises(DrivingPriorWorldModelError, match="structural no-op"):
        apply_world_model_archive(blob, pair_indices=[0])

    frames = apply_world_model_archive(blob, pair_indices=[0], require_nonzero=False)
    assert int(frames.max()) == 0


def test_false_authority_metadata_is_rejected() -> None:
    cfg = _cfg()
    with pytest.raises(DrivingPriorWorldModelError, match="score_claim"):
        pack_archive(cfg, metadata={"score_claim": True})
    with pytest.raises(DrivingPriorWorldModelError, match="ready_for_exact_eval_dispatch"):
        pack_archive(cfg, metadata={"ready_for_exact_eval_dispatch": True})
    with pytest.raises(DrivingPriorWorldModelError, match="research_only"):
        pack_archive(cfg, metadata={"research_only": False})
    with pytest.raises(DrivingPriorWorldModelError, match="proxy"):
        pack_archive(cfg, metadata={"proxy": False})


def test_invalid_lengths_and_pair_indices_fail_closed() -> None:
    cfg = _cfg()
    with pytest.raises(DrivingPriorWorldModelError, match="prior_weights"):
        pack_archive(cfg, b"\x01", bytes(expected_residual_bytes(cfg)))
    with pytest.raises(DrivingPriorWorldModelError, match="residual_bytes"):
        pack_archive(cfg, deterministic_prior_weights(cfg), b"\x00")

    blob = pack_archive(cfg)
    with pytest.raises(DrivingPriorWorldModelError, match="out of range"):
        apply_world_model_archive(blob, pair_indices=[cfg.num_pairs])


def test_module_import_and_apply_do_not_import_scorers(monkeypatch: pytest.MonkeyPatch) -> None:
    forbidden_imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        lowered = str(name).lower()
        if (
            lowered.startswith("tac.scorer")
            or lowered.startswith("upstream")
            or "segnet" in lowered
            or "posenet" in lowered
        ):
            forbidden_imports.append(str(name))
            raise AssertionError(f"forbidden scorer import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    for module_name in tuple(sys.modules):
        if module_name.startswith("tac.substrates.driving_prior_world_model"):
            sys.modules.pop(module_name)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = importlib.import_module("tac.substrates.driving_prior_world_model")
    cfg = module.DrivingPriorWorldModelConfig(
        output_height=4,
        output_width=4,
        num_pairs=1,
        codebook_entries=4,
        residual_grid_height=2,
        residual_grid_width=2,
    )
    archive = module.pack_archive(cfg)
    frames = module.apply_world_model_archive(archive, pair_indices=[0])

    assert frames.shape == (1, 2, 4, 4, 3)
    assert forbidden_imports == []
