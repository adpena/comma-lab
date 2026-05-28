# SPDX-License-Identifier: MIT
"""Catalog #91 ENCODE_INFLATE_ROUNDTRIP + #139 no_op_proof + L0 SCAFFOLD contract for V3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from tac.substrates.pact_nerv_selector_v3.architecture import (
    PactNervSelectorV3Config,
    PactNervSelectorV3Substrate,
    RiceGolombSelectorCoder,
)
from tac.substrates.pact_nerv_selector_v3.archive import (
    DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11,
    PSV3_HEADER_SIZE,
    PSV3_MAGIC,
    PSV3_SCHEMA_VERSION,
    pack_archive,
    parse_archive,
)


def _smoke_cfg() -> PactNervSelectorV3Config:
    return PactNervSelectorV3Config(
        latent_dim=8,
        embed_dim=24,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(20, 16, 12),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=3,
        output_height=24,
        output_width=32,
        selector_palette_size=16,
        rice_golomb_k=2,
    )


def _smoke_meta(cfg: PactNervSelectorV3Config) -> dict[str, object]:
    return {
        "embed_dim": cfg.embed_dim,
        "initial_grid_h": cfg.initial_grid_h,
        "initial_grid_w": cfg.initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "sin_frequency": cfg.sin_frequency,
        "num_upsample_blocks": cfg.num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "rice_golomb_k": cfg.rice_golomb_k,
    }


def test_module_import_resolves_canonical_symbols() -> None:
    from tac.substrates import pact_nerv_selector_v3 as m
    for name in (
        "PactNervSelectorV3Config", "PactNervSelectorV3Substrate",
        "RiceGolombSelectorCoder", "pack_archive", "parse_archive",
        "PactNervSelectorV3ScoreAwareLoss", "PactNervSelectorV3Archive",
    ):
        assert hasattr(m, name), f"missing canonical symbol: {name}"


def test_substrate_forward_produces_unit_interval_rgb() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervSelectorV3Substrate(cfg).eval()
    idx = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    assert rgb_0.shape == (2, 3, cfg.output_height, cfg.output_width)
    assert float(rgb_0.min()) >= 0.0
    assert float(rgb_0.max()) <= 1.0


def test_rice_golomb_encode_geometric_decay_invariant() -> None:
    """Rice-Golomb: symbol n with parameter k costs (n>>k) + 1 + k bits."""
    coder = RiceGolombSelectorCoder(palette_size=16, k=2)
    # Symbol 0: q=0, 1 unary bit + 2 suffix bits = 3 bits per sym 0
    # Symbol 4: q=1, 2 unary bits + 2 suffix bits = 4 bits per sym
    bits = coder.encoded_bit_length([0, 4])
    assert bits == 3 + 4


def test_rice_golomb_rejects_invalid_symbols() -> None:
    coder = RiceGolombSelectorCoder(palette_size=16, k=2)
    try:
        coder.encode([16])
    except ValueError as exc:
        assert "out of palette" in str(exc)
    else:
        raise AssertionError("expected ValueError for symbol >= palette")


def test_rice_golomb_k_parameter_validation() -> None:
    try:
        RiceGolombSelectorCoder(palette_size=16, k=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for k=-1")
    try:
        RiceGolombSelectorCoder(palette_size=16, k=9)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for k=9")


def test_archive_pack_then_parse_roundtrip() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(0)
    model = PactNervSelectorV3Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    latents = sd["latents"].clone()
    selector_bytes = b"\x00\x01\x02"
    blob = pack_archive(
        decoder_sd, latents, selector_bytes, _smoke_meta(cfg), palette_size=16
    )
    arc = parse_archive(blob)
    assert arc.schema_version == PSV3_SCHEMA_VERSION
    assert blob[:4] == PSV3_MAGIC
    assert arc.palette_size == 16
    assert arc.selector_bytes == selector_bytes


def test_archive_int8_decoder_quantization_roundtrip_is_fail_closed_parseable() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(42)
    model = PactNervSelectorV3Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    blob = pack_archive(
        decoder_sd,
        sd["latents"].clone(),
        b"\x00\x01\x02",
        _smoke_meta(cfg),
        palette_size=16,
        decoder_quantization=DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11,
    )
    arc = parse_archive(blob)

    assert arc.meta["decoder_quantization"] == DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11
    for name, original in decoder_sd.items():
        recovered = arc.decoder_state_dict[name]
        assert recovered.shape == original.shape
        assert torch.is_floating_point(recovered)
        assert torch.max((recovered.float() - original.float()).abs()) < 0.02


def test_archive_rejects_unknown_decoder_quantization() -> None:
    cfg = _smoke_cfg()
    model = PactNervSelectorV3Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    try:
        pack_archive(
            decoder_sd,
            sd["latents"].clone(),
            b"\x00",
            _smoke_meta(cfg),
            palette_size=16,
            decoder_quantization="mystery_quant",
        )
    except ValueError as exc:
        assert "unsupported decoder_quantization" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown decoder_quantization")


def test_decoder_quant_repack_cli_writes_byte_closed_fail_closed_manifest(
    tmp_path: Path,
) -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(43)
    model = PactNervSelectorV3Substrate(cfg)
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    source = tmp_path / "source_0.bin"
    source.write_bytes(
        pack_archive(
            decoder_sd,
            sd["latents"].clone(),
            b"\x00\x01\x02",
            _smoke_meta(cfg),
            palette_size=16,
        )
    )
    out_dir = tmp_path / "candidate"

    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[5] / "tools"
                / "repack_pact_nerv_selector_v3_decoder_quant.py"),
            "--archive",
            str(source),
            "--output-dir",
            str(out_dir),
            "--n-proof-pairs",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stdout_payload = json.loads(proc.stdout)
    manifest = json.loads(
        (out_dir / "decoder_quant_repack_manifest.json").read_text(encoding="utf-8")
    )
    assert stdout_payload["schema_version"] == manifest["schema_version"]
    assert (out_dir / "archive.zip").is_file()
    assert (out_dir / "submission" / "inflate.py").is_file()
    assert (out_dir / "runtime_adapter" / "inflate.sh").is_file()
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["decoder_quantization"] == DECODER_QUANT_INT8_PER_CHANNEL_BROTLI_Q11
    assert manifest["local_decoder_drift"]["n_pairs_measured"] == 1
    assert manifest["runtime_adapter_tree_sha256"]
    assert manifest["runtime_consumption_proof_path"]
    assert manifest["optimizer_candidate_queue_path"]

    proof = json.loads(
        (out_dir / "runtime_consumption_proof.json").read_text(encoding="utf-8")
    )
    queue = json.loads(
        (out_dir / "optimizer_candidate_queue.json").read_text(encoding="utf-8")
    )
    row = queue["top_k"][0]
    assert proof["schema"] == "family_agnostic_runtime_consumption_proof_v1"
    assert proof["receiver_contract_satisfied"] is True
    assert proof["runtime_adapter_manifest"]["runtime_adapter_ready"] is True
    assert queue["schema"] == "optimizer_candidate_queue_v1"
    assert queue["dispatch_ready"] == []
    assert row["receiver_contract_satisfied"] is True
    assert row["runtime_adapter_ready"] is True
    assert row["runtime_consumption_proof_status"] == "present"
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    from tac.optimizer.materializer_submission_closure import (
        build_materializer_submission_runtime_closure,
    )

    closure_report = build_materializer_submission_runtime_closure(
        repo_root=Path(__file__).resolve().parents[5],
        source_queue_path=out_dir / "optimizer_candidate_queue.json",
        candidate_id=row["candidate_id"],
        submission_dir_out=tmp_path / "closed_submission",
        closed_source_queue_out=tmp_path / "closed_queue.json",
        closure_report_out=tmp_path / "closure_report.json",
        overwrite=True,
    )
    assert closure_report["archive_sha256"] == manifest["candidate_archive_zip_sha256"]
    closed_queue = json.loads(
        (tmp_path / "closed_queue.json").read_text(encoding="utf-8")
    )
    closed_row = closed_queue["top_k"][0]
    assert closed_row["runtime_adapter_ready"] is True
    assert closed_row["candidate_archive_path_unverified"] is False
    assert closed_row["score_claim"] is False
    assert closed_row["ready_for_exact_eval_dispatch"] is False


def test_archive_header_size_invariant_is_26_bytes() -> None:
    assert PSV3_HEADER_SIZE == 26


def test_byte_mutation_changes_archive_no_op_proof() -> None:
    cfg = _smoke_cfg()
    torch.manual_seed(13)
    model = PactNervSelectorV3Substrate(cfg).eval()
    sd = model.state_dict()
    decoder_sd = {k: v for k, v in sd.items() if k not in ("latents", "selectors")}
    latents = sd["latents"].clone()
    blob_a = pack_archive(decoder_sd, latents, b"\x00\x01", _smoke_meta(cfg), palette_size=16)
    blob_b = pack_archive(decoder_sd, latents, b"\xff\x01", _smoke_meta(cfg), palette_size=16)
    assert blob_a != blob_b


def test_trainer_full_main_implemented_and_cuda_gated(tmp_path) -> None:
    """PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: _full_main IMPLEMENTED + CUDA-gated."""
    import importlib
    import inspect

    import pytest

    trainer = importlib.import_module("experiments.train_substrate_pact_nerv_selector_v3")
    src = inspect.getsource(trainer._full_main)
    assert "raise NotImplementedError" not in src
    assert "run_pact_nerv_score_aware_training" in src
    args = trainer._build_parser().parse_args(
        ["--output-dir", str(tmp_path / "out"), "--device", "cpu"]
    )
    with pytest.raises(SystemExit):
        trainer._full_main(args)


def test_trainer_routes_through_canonical_scorer_loss_helper() -> None:
    import inspect

    from tac.substrates.pact_nerv_selector_v3 import score_aware_loss as sal
    src = inspect.getsource(sal)
    assert "score_pair_components_dispatch" in src
    assert "tac.substrates.score_aware_common" in src


def test_trainer_patches_differentiable_eval_roundtrip_before_scorer() -> None:
    import inspect

    import experiments.train_substrate_pact_nerv_selector_v3 as trainer_module
    src = inspect.getsource(trainer_module._smoke_main)
    assert "patch_upstream_yuv6_globally" in src


def test_recipe_research_only_and_dispatch_disabled() -> None:
    from pathlib import Path

    import yaml  # type: ignore[import-untyped]
    recipe = yaml.safe_load(
        (Path(__file__).resolve().parents[5]
         / ".omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_modal_t4_dispatch.yaml"
        ).read_text(encoding="utf-8")
    )
    assert recipe["dispatch_enabled"] is False
    assert recipe["research_only"] is True


def test_driver_carries_canonical_nvml_block() -> None:
    from pathlib import Path
    txt = (
        Path(__file__).resolve().parents[5]
        / "scripts/remote_lane_substrate_pact_nerv_selector_v3.sh"
    ).read_text(encoding="utf-8")
    assert "DALI_DISABLE_NVML" in txt
    assert "CUBLAS_WORKSPACE_CONFIG" in txt
    assert "PYTORCH_CUDA_ALLOC_CONF" in txt


def test_inflate_py_loc_under_200() -> None:
    from pathlib import Path
    loc = len(
        (Path(__file__).resolve().parents[1] / "inflate.py")
        .read_text(encoding="utf-8").splitlines()
    )
    assert loc <= 200
