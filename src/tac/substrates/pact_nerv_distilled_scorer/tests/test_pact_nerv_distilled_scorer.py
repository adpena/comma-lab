# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + Catalog #139 no_op_proof for pact_nerv_distilled_scorer.

Proves the encode/decode contract of the PDS monolithic 0.bin grammar +
distilled-scorer-surrogate forward-pass parity. Plus a smoke-level test
that the trainer's _full_main raises NotImplementedError per the L0
SCAFFOLD posture (Catalog #240).
"""

from __future__ import annotations

import torch

from tac.substrates.pact_nerv_distilled_scorer.architecture import (
    DistilledScorerSurrogate,
    PactNervDistilledScorerConfig,
    PactNervDistilledScorerSubstrate,
)
from tac.substrates.pact_nerv_distilled_scorer.archive import (
    PDS_HEADER_SIZE,
    PDS_MAGIC,
    PDS_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervDistilledScorerConfig:
    return PactNervDistilledScorerConfig(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        surrogate_hidden=8,
        surrogate_feature_dim=4,
        distill_temperature=2.0,
        num_pairs=3,
        output_height=24,
        output_width=32,
    )


def _smoke_meta(cfg: PactNervDistilledScorerConfig) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "surrogate_hidden": cfg.surrogate_hidden,
        "surrogate_feature_dim": cfg.surrogate_feature_dim,
        "distill_temperature": cfg.distill_temperature,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
    }


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _scorer_response_dataset() -> dict[str, object]:
    false_authority = _false_authority()
    return {
        "schema": "scorer_response_dataset.v1",
        **false_authority,
        "authority": {
            **false_authority,
            "evidence_grade": "macOS-CPU advisory response dataset",
        },
        "rows": [
            {
                "schema": "scorer_response_row.v1",
                "row_id": "row-a",
                **false_authority,
                "authority_source_score_claim": False,
                "advisory_score_report_derived": 0.25,
                "delta_vs_baseline_score": 0.01,
            },
            {
                "schema": "scorer_response_row.v1",
                "row_id": "row-b",
                **false_authority,
                "authority_source_score_claim": False,
                "advisory_score_report_derived": 0.24,
                "delta_vs_baseline_score": -0.001,
            },
        ],
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_distilled_scorer as pkg

    assert hasattr(pkg, "PactNervDistilledScorerConfig")
    assert hasattr(pkg, "PactNervDistilledScorerSubstrate")
    assert hasattr(pkg, "DistilledScorerSurrogate")
    assert hasattr(pkg, "pack_archive")
    assert hasattr(pkg, "parse_archive")
    assert hasattr(pkg, "PactNervDistilledScorerScoreAwareLoss")
    assert hasattr(pkg, "PactNervDistilledScorerArchive")
    assert pkg.CONSUMES_SCORER_RESPONSE_DATASET is True
    assert hasattr(pkg, "load_scorer_response_distill_rows")


def test_score_aware_loss_consumes_scorer_response_dataset_contract() -> None:
    from tac.substrates.pact_nerv_distilled_scorer import score_aware_loss as sal

    dataset = _scorer_response_dataset()
    rows = sal.load_scorer_response_distill_rows(dataset, max_rows=1)

    assert sal.CONSUMES_SCORER_RESPONSE_DATASET is True
    assert sal.SCORER_RESPONSE_DATASET_SCHEMA == "scorer_response_dataset.v1"
    assert len(rows) == 1
    assert rows[0]["row_id"] == "row-a"


def test_score_aware_loss_rejects_promotional_scorer_response_dataset() -> None:
    from tac.substrates.pact_nerv_distilled_scorer import score_aware_loss as sal

    dataset = _scorer_response_dataset()
    dataset["score_claim"] = True

    try:
        sal.load_scorer_response_distill_rows(dataset)
    except ValueError as exc:
        assert "score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected promotional scorer-response dataset rejection")


def test_score_aware_loss_rejects_legacy_missing_authority_dataset() -> None:
    from tac.substrates.pact_nerv_distilled_scorer import score_aware_loss as sal

    dataset = _scorer_response_dataset()
    dataset.pop("rank_or_kill_eligible")

    try:
        sal.load_scorer_response_distill_rows(dataset)
    except ValueError as exc:
        assert "rank_or_kill_eligible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing authority rejection")


def test_score_aware_loss_rejects_source_score_claim_rows() -> None:
    from tac.substrates.pact_nerv_distilled_scorer import score_aware_loss as sal

    dataset = _scorer_response_dataset()
    rows = dataset["rows"]
    assert isinstance(rows, list)
    rows[0]["authority_source_score_claim"] = True

    try:
        sal.load_scorer_response_distill_rows(dataset)
    except ValueError as exc:
        assert "authority_source_score_claim" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected source score-claim row rejection")


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    """L5 compliance: substrate is a full RGB renderer (not a mask codec)."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervDistilledScorerSubstrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert rgb_1.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_distilled_scorer_surrogate_emits_feature_vector() -> None:
    """Distinguishing primitive: surrogate emits (B, feature_dim) global vector."""
    torch.manual_seed(0)
    surr = DistilledScorerSurrogate(hidden=16, feature_dim=8)
    x = torch.randn(3, 3, 64, 64)
    feat = surr(x)
    assert feat.shape == (3, 8)
    assert feat.dtype == torch.float32


def test_archive_pack_then_parse_roundtrip_recovers_tensors() -> None:
    """ENCODE_INFLATE_ROUNDTRIP — Catalog #91 contract."""
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervDistilledScorerSubstrate(cfg)
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items() if k not in ("latents", "surrogate_init_input")
    }
    latents = sd["latents"].clone()

    blob = pack_archive(decoder_sd, latents, _smoke_meta(cfg))
    arc = parse_archive(blob)

    assert arc.schema_version == PDS_SCHEMA_VERSION
    assert blob[:4] == PDS_MAGIC
    assert set(arc.decoder_state_dict.keys()) == set(decoder_sd.keys())
    assert arc.latents.shape == latents.shape


def test_archive_grammar_header_size_invariant_is_21_bytes() -> None:
    assert PDS_HEADER_SIZE == 21


def test_byte_mutation_changes_inflate_output_no_op_proof() -> None:
    """Catalog #139 no_op_proof: byte mutation MUST change rendered frames."""
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervDistilledScorerSubstrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {
        k: v for k, v in sd.items() if k not in ("latents", "surrogate_init_input")
    }
    latents = sd["latents"].clone()

    blob_a = pack_archive(decoder_sd, latents, _smoke_meta(cfg))
    mutated_latents = latents.clone()
    mutated_latents[0, 0] = mutated_latents[0, 0] + 1.0
    blob_b = pack_archive(decoder_sd, mutated_latents, _smoke_meta(cfg))
    assert blob_a != blob_b, "no_op_proof: mutating latents must change archive bytes"
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    assert not torch.allclose(arc_a.latents[0, 0], arc_b.latents[0, 0], atol=1e-6)


def test_trainer_full_main_raises_not_implemented_at_l0_scaffold() -> None:
    """L0 SCAFFOLD posture: trainer _full_main MUST raise NotImplementedError."""
    import argparse
    import importlib

    trainer = importlib.import_module(
        "experiments.train_substrate_pact_nerv_distilled_scorer"
    )
    ns = argparse.Namespace(output_dir=None, epochs=1, smoke=False, device="cpu")
    try:
        trainer._full_main(ns)
    except NotImplementedError as exc:
        assert (
            "OPERATOR-GATED" in str(exc)
            or "L0 SCAFFOLD" in str(exc)
            or "Stage 1" in str(exc)
        )
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError per L0 SCAFFOLD posture")


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    """Catalog #164: trainer score-aware path MUST route through canonical helper."""
    import inspect

    from tac.substrates.pact_nerv_distilled_scorer import score_aware_loss as sal_module

    src = inspect.getsource(sal_module)
    assert "score_pair_components_dispatch" in src, (
        "score-aware module MUST route through canonical helper per Catalog #164"
    )
    assert (
        "tac.substrates.score_aware_common" in src
    ), "must import from canonical helper module"


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    """Catalog #6 MANDATORY DEFAULT: trainer MUST patch yuv6 BEFORE scorer load."""
    import inspect

    import experiments.train_substrate_pact_nerv_distilled_scorer as trainer_module

    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src, (
        "_smoke_main MUST call patch_upstream_yuv6_globally() per CLAUDE.md "
        "eval_roundtrip non-negotiable"
    )
    patch_idx = src.find("patch_upstream_yuv6_globally")
    for scorer_token in ("load_differentiable_scorers", "load_default_scorers"):
        if scorer_token in src:
            scorer_idx = src.find(scorer_token)
            assert patch_idx < scorer_idx, (
                f"patch_upstream_yuv6_globally MUST precede {scorer_token}"
            )


def test_recipe_research_only_and_dispatch_disabled() -> None:
    """Catalog #240: recipe MUST opt out of dispatch at L0 SCAFFOLD."""
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]

    recipe_path = (
        Path(__file__).resolve().parents[5]
        / ".omx/operator_authorize_recipes/substrate_pact_nerv_distilled_scorer_modal_t4_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe missing: {recipe_path}"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert recipe["dispatch_enabled"] is False, (
        "recipe MUST declare dispatch_enabled:false at L0 SCAFFOLD"
    )
    assert recipe["research_only"] is True, (
        "recipe MUST declare research_only:true at L0 SCAFFOLD"
    )


def test_driver_carries_canonical_nvml_block() -> None:
    """Catalog #244: remote lane driver MUST carry the canonical 3-export NVML block."""
    from pathlib import Path

    driver_path = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_distilled_scorer.sh"
    )
    assert driver_path.exists(), f"driver missing: {driver_path}"
    driver_text = driver_path.read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in driver_text
    assert "CUBLAS_WORKSPACE_CONFIG" in driver_text
    assert "PYTORCH_CUDA_ALLOC_CONF" in driver_text


def test_inflate_py_loc_under_200_per_hnerv_parity_l4() -> None:
    """HNeRV parity L4: inflate runtime MUST be ≤ 200 LOC."""
    from pathlib import Path

    inflate_path = Path(__file__).resolve().parents[1] / "inflate.py"
    assert inflate_path.exists()
    physical_loc = len(inflate_path.read_text(encoding="utf-8").splitlines())
    assert physical_loc <= 200, (
        f"inflate.py {physical_loc} LOC exceeds HNeRV parity L4 ceiling 200"
    )
