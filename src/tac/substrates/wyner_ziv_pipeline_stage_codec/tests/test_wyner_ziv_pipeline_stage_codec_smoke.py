# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec L0 SCAFFOLD smoke tests.

Coverage per Catalog #325 6-step contract step (4) sextet pact deliberation
evidence (tests are the structural evidence the L0 scaffold ships):

1. SubstrateContract registers cleanly per Catalog #241/#242 (36-field schema).
2. Observability surface declares all 6 facets per Catalog #305.
3. Architecture wrapper builds canonical primitive config.
4. encode_pre_entropy_via_pipeline_stage_codec routes through canonical primitive.
5. Encode-decode roundtrip is byte-identical per Catalog #105/#139/#220/#272
   no-op detector.
6. Archive grammar round-trips per WZPSC01_MAGIC + HEADER_FORMAT contract.
7. select_inflate_device honors PACT_INFLATE_DEVICE env var per Catalog #205.
8. select_inflate_device refuses 'mps' per CLAUDE.md "MPS auth eval is NOISE".
9. _full_main raises NotImplementedError at L0 per Catalog #240 transparent
   non-dispatchable.
10. _smoke_main returns 0 on byte-identical roundtrip per Catalog #325.
11. End-to-end encode → archive → inflate roundtrip is byte-identical
    (PYTHONPATH self-containment per Catalog #295 verified).
12. Canonical Provenance defaults per Catalog #323 (evidence_grade='predicted'
    + score_claim=False + promotable=False + axis_tag='[predicted]').
"""

from __future__ import annotations

import os
import pytest

from tac.codec.wyner_ziv_layer import (
    InterceptLocation,
    WynerZivLayerConfig,
    WynerZivLayerResult,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec import (
    LANE_ID,
    OBSERVABILITY_SURFACE,
    SUBSTRATE_ID,
    WYNER_ZIV_PIPELINE_STAGE_CODEC_CONTRACT,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec.architecture import (
    WynerZivPipelineStageCodecArchitecture,
    encode_pre_entropy_via_pipeline_stage_codec,
    reconstruct_pre_entropy_via_pipeline_stage_codec,
    report_stage_byte_counts,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec.archive import (
    HEADER_SIZE,
    WZPSC01_MAGIC,
    WZPSC01_VERSION,
    ArchiveLayout,
    decode_archive_bytes_scaffold,
    encode_archive_bytes_scaffold,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec.inflate import (
    inflate_wyner_ziv_pipeline_stage_codec_scaffold,
    select_inflate_device,
)
from tac.substrates.wyner_ziv_pipeline_stage_codec.trainer import (
    L0_SCAFFOLD_NOT_IMPLEMENTED_MESSAGE,
    _full_main,
    _smoke_main,
    build_arg_parser,
    main,
)


# -----------------------------------------------------------------------------
# Catalog #241/#242 SubstrateContract registration
# -----------------------------------------------------------------------------


def test_substrate_contract_registers_cleanly():
    """Catalog #241/#242: contract validates the 36-field schema invariants."""
    contract = WYNER_ZIV_PIPELINE_STAGE_CODEC_CONTRACT
    assert contract.id == SUBSTRATE_ID == "wyner_ziv_pipeline_stage_codec"
    assert contract.lane_id == LANE_ID == "lane_wyner_ziv_pipeline_stage_codec_l0_scaffold_20260528"
    assert contract.target_modes == ("research_substrate",)
    assert contract.deployment_target == "desktop_research"
    assert contract.archive_grammar == "monolithic_single_file_wzpsc01"
    assert contract.inflate_runtime_loc_budget == 200
    assert contract.runtime_dep_closure == ("numpy", "brotli")
    assert contract.export_format == "custom"
    assert contract.score_aware_loss == "custom"
    assert contract.no_op_detector_planned is True
    # L0 invariant: RESEARCH_ONLY + runtime_overlay_consumed=False per __post_init__
    assert contract.score_improvement_mechanism_status == "RESEARCH_ONLY"
    assert contract.runtime_overlay_consumed is False
    assert contract.recipe_research_only is True
    assert contract.recipe_smoke_only is True


def test_substrate_contract_lane_id_satisfies_canonical_pattern():
    """Catalog #241: lane_id must match /^lane_[a-z0-9_]+_\\d{8}$/."""
    import re
    assert re.match(r"^lane_[a-z0-9_]+_\d{8}$", LANE_ID)


# -----------------------------------------------------------------------------
# Catalog #305 observability surface (6 facets)
# -----------------------------------------------------------------------------


def test_observability_surface_declares_all_6_facets():
    """Catalog #305: every substrate package declares the 6-facet observability surface."""
    expected_facets = {
        "inspectable_per_layer",
        "decomposable_per_signal",
        "diff_able_across_runs",
        "queryable_post_hoc",
        "cite_able",
        "counterfactual_able",
    }
    assert set(OBSERVABILITY_SURFACE.keys()) == expected_facets
    for facet_name, facet_text in OBSERVABILITY_SURFACE.items():
        assert isinstance(facet_text, str)
        assert len(facet_text) >= 20, (
            f"facet {facet_name!r} too short ({len(facet_text)} chars); "
            f"substrate-engineering observability surfaces must be substantive"
        )


# -----------------------------------------------------------------------------
# Architecture + canonical primitive routing
# -----------------------------------------------------------------------------


def test_architecture_builds_canonical_primitive_config():
    """The architecture's to_primitive_config() returns a valid WynerZivLayerConfig."""
    arch = WynerZivPipelineStageCodecArchitecture()
    config = arch.to_primitive_config()
    assert isinstance(config, WynerZivLayerConfig)
    assert config.intercept_location == InterceptLocation.STATE_DICT_SERIALIZATION
    assert config.side_info_source == "Comma2k19"
    assert config.main_codec == "lzma"
    assert config.compression_codec_for_side == "lzma"


def test_encode_pre_entropy_routes_through_canonical_primitive():
    """encode_pre_entropy_via_pipeline_stage_codec routes through insert_wyner_ziv_layer."""
    source = b"WZPSC_TEST_DATA_" * 64  # 1024 B
    side_info_y = b"Y_PREFIX_" + source[:512] + b"_Y_SUFFIX"  # contains 512 B overlap
    result = encode_pre_entropy_via_pipeline_stage_codec(
        pre_entropy_bytes=source,
        side_info_y=side_info_y,
    )
    assert isinstance(result, WynerZivLayerResult)
    # The overlap MUST be detected (Y contains a 512-byte prefix of source).
    assert result.main_bytes_raw == len(source) - 512
    # Canonical Provenance defaults per Catalog #323 + Catalog #341 Tier A markers.
    assert result.evidence_grade == "predicted"
    assert result.score_claim is False
    assert result.promotion_eligible is False


def test_encode_decode_roundtrip_byte_identical():
    """Catalog #105/#139/#220/#272: encode-decode roundtrip MUST be byte-identical."""
    source = b"WZPSC_ROUNDTRIP_TEST_" * 32 + bytes(range(256)) * 2
    overlap = source[:300]
    side_info_y = b"PRE_" + overlap + b"_MID_" + overlap + b"_POST"
    arch = WynerZivPipelineStageCodecArchitecture()

    from tac.codec.wyner_ziv_layer import _compress, _detect_y_derivable_prefix
    prefix_len = _detect_y_derivable_prefix(source, side_info_y)
    offset = side_info_y.find(source[:prefix_len]) if prefix_len else 0
    main_raw = source[prefix_len:]
    side_raw = offset.to_bytes(8, "big") + prefix_len.to_bytes(8, "big")
    main_compressed = _compress(arch.main_codec, main_raw)
    side_compressed_baked = _compress(arch.compression_codec_for_side, side_raw)

    reconstructed = reconstruct_pre_entropy_via_pipeline_stage_codec(
        main_compressed=main_compressed,
        side_compressed_baked=side_compressed_baked,
        side_info_y=side_info_y,
        architecture=arch,
    )
    assert reconstructed == source, (
        f"roundtrip not byte-identical: source={len(source)} B; "
        f"reconstructed={len(reconstructed)} B; prefix_len={prefix_len}; "
        f"offset_in_y={offset}"
    )


def test_report_stage_byte_counts_surfaces_canonical_provenance():
    """Catalog #305: report_stage_byte_counts exposes all observability fields."""
    source = b"WZPSC_REPORT_" * 64
    side_info_y = b"Y_" + source[:256] + b"_Y"
    result = encode_pre_entropy_via_pipeline_stage_codec(
        pre_entropy_bytes=source, side_info_y=side_info_y,
    )
    report = report_stage_byte_counts(result)
    expected_keys = {
        "intercept_location", "main_bytes_raw", "main_bytes_compressed",
        "side_bytes_raw", "side_bytes_compressed_baked",
        "score_savings_estimate", "inflate_py_loc_added",
        "decoder_complexity_estimate_seconds",
        "main_bytes_sha256", "side_info_sha256",
        "evidence_grade", "score_claim", "promotion_eligible",
    }
    assert set(report.keys()) == expected_keys
    assert report["evidence_grade"] == "predicted"
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False


# -----------------------------------------------------------------------------
# Archive grammar (WZPSC01)
# -----------------------------------------------------------------------------


def test_archive_grammar_roundtrip():
    """Archive bytes encode/decode preserves layout + payload."""
    main_compressed = b"\x01\x02\x03" * 100
    side_compressed_baked = b"\xff\xfe\xfd\xfc"
    archive_bytes = encode_archive_bytes_scaffold(
        main_compressed=main_compressed,
        side_compressed_baked=side_compressed_baked,
        intercept_location="state_dict_serialization",
        side_info_source="Comma2k19",
        main_codec="lzma",
        compression_codec_for_side="lzma",
    )
    # Magic + version header at the start
    assert archive_bytes[:4] == WZPSC01_MAGIC
    assert archive_bytes[4] == WZPSC01_VERSION
    # Decode roundtrip
    decoded = decode_archive_bytes_scaffold(archive_bytes)
    assert decoded["main_compressed"] == main_compressed
    assert decoded["side_compressed_baked"] == side_compressed_baked
    layout = decoded["layout"]
    assert isinstance(layout, ArchiveLayout)
    assert layout.main_len == len(main_compressed)
    assert layout.side_len == len(side_compressed_baked)
    assert layout.intercept_location == "state_dict_serialization"
    assert layout.side_info_source == "Comma2k19"


def test_archive_grammar_refuses_bad_magic():
    """Catalog #220: archive grammar header must validate magic + version."""
    archive_bytes = b"BADM\x01" + b"\x00" * (HEADER_SIZE - 5) + b"{}" + b""
    with pytest.raises(ValueError, match="magic mismatch"):
        decode_archive_bytes_scaffold(archive_bytes)


# -----------------------------------------------------------------------------
# Inflate runtime + Catalog #205 select_inflate_device
# -----------------------------------------------------------------------------


def test_select_inflate_device_default_cpu():
    """Catalog #205: default 'auto' resolves to 'cpu' for the numpy-portable inflate."""
    prior = os.environ.pop("PACT_INFLATE_DEVICE", None)
    try:
        assert select_inflate_device() == "cpu"
    finally:
        if prior is not None:
            os.environ["PACT_INFLATE_DEVICE"] = prior


def test_select_inflate_device_honors_env():
    """Catalog #205: PACT_INFLATE_DEVICE='cuda' is honored."""
    prior = os.environ.get("PACT_INFLATE_DEVICE")
    try:
        os.environ["PACT_INFLATE_DEVICE"] = "cuda"
        assert select_inflate_device() == "cuda"
    finally:
        if prior is None:
            os.environ.pop("PACT_INFLATE_DEVICE", None)
        else:
            os.environ["PACT_INFLATE_DEVICE"] = prior


def test_select_inflate_device_refuses_mps():
    """CLAUDE.md 'MPS auth eval is NOISE' + Catalog #1: 'mps' is REFUSED."""
    prior = os.environ.get("PACT_INFLATE_DEVICE")
    try:
        os.environ["PACT_INFLATE_DEVICE"] = "mps"
        with pytest.raises(ValueError, match="REFUSED"):
            select_inflate_device()
    finally:
        if prior is None:
            os.environ.pop("PACT_INFLATE_DEVICE", None)
        else:
            os.environ["PACT_INFLATE_DEVICE"] = prior


def test_e2e_encode_archive_inflate_roundtrip_byte_identical():
    """E2E: encode → archive grammar → inflate runtime → byte-identical reconstruction."""
    source = b"E2E_WZPSC_TEST_DATA_" * 50 + bytes(range(256))
    overlap = source[:400]
    side_info_y = b"_" * 100 + overlap + b"_TAIL"

    arch = WynerZivPipelineStageCodecArchitecture()
    from tac.codec.wyner_ziv_layer import _compress, _detect_y_derivable_prefix
    prefix_len = _detect_y_derivable_prefix(source, side_info_y)
    offset = side_info_y.find(source[:prefix_len]) if prefix_len else 0
    main_raw = source[prefix_len:]
    side_raw = offset.to_bytes(8, "big") + prefix_len.to_bytes(8, "big")
    main_compressed = _compress(arch.main_codec, main_raw)
    side_compressed_baked = _compress(arch.compression_codec_for_side, side_raw)

    archive_bytes = encode_archive_bytes_scaffold(
        main_compressed=main_compressed,
        side_compressed_baked=side_compressed_baked,
        intercept_location=arch.intercept_location.value,
        side_info_source=arch.side_info_source,
        main_codec=arch.main_codec,
        compression_codec_for_side=arch.compression_codec_for_side,
    )

    inflated = inflate_wyner_ziv_pipeline_stage_codec_scaffold(
        archive_bytes=archive_bytes,
        side_info_y=side_info_y,
    )
    assert inflated["reconstructed_pre_entropy_bytes"] == source


# -----------------------------------------------------------------------------
# L0->L1 LONG MLX promotion 2026-05-28: _full_main is now the canonical empirical-
# measurement harness per the sister L1 landing memo
# (.omx/research/wyner_ziv_pipeline_stage_codec_l1_long_mlx_600pair_landed_20260528.md).
# Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113: the L0 sentinel constant
# L0_SCAFFOLD_NOT_IMPLEMENTED_MESSAGE is preserved (for forensic citation of the
# L0 scaffold contract) but the runtime tests now exercise the L1 harness.
# -----------------------------------------------------------------------------


def test_l0_scaffold_message_constant_preserved_for_historical_provenance():
    """Per Catalog #110/#113: the L0 sentinel constant remains importable for
    forensic citation; the L1 harness does not raise NotImplementedError but
    the constant's existence documents the pre-L1 scaffold contract.
    """
    assert ".omx/research/wyner_ziv_pipeline_stage_codec_design_20260528.md" in L0_SCAFFOLD_NOT_IMPLEMENTED_MESSAGE
    assert "L0 SCAFFOLD" in L0_SCAFFOLD_NOT_IMPLEMENTED_MESSAGE


def test_l1_full_main_runs_to_zero_on_real_pr101_state_dict():
    """L1 LONG MLX harness: _full_main MUST exit 0 on real PR101 fp16 bytes
    and produce the canonical empirical anchor + WZPSC01 archive + byte-
    identical roundtrip per Catalog #105/#139/#220/#272.
    """
    import os as _os
    import tempfile
    from pathlib import Path as _Path

    parser = build_arg_parser()
    pr101_path = _Path(
        "experiments/results/pr101_codecop_sweep_20260507_codex/"
        "pr101_decoder_state_dict.pt"
    )
    if not pr101_path.exists():
        pytest.skip(f"PR101 reference state_dict not present at {pr101_path}")

    with tempfile.TemporaryDirectory(dir=".omx/tmp" if _os.path.isdir(".omx/tmp") else None) as td:
        out_dir = _Path(td) / "wzpsc_l1_test"
        args = parser.parse_args([
            "--full",
            "--output-dir", str(out_dir),
            "--base-substrate-bytes-path", str(pr101_path),
            "--base-substrate-bytes-form", "raw_fp16",
        ])
        rc = _full_main(args)
        assert rc == 0, f"L1 harness exited non-zero: {rc}"

        artifact_path = out_dir / "training_artifact.json"
        assert artifact_path.exists(), "training_artifact.json was not emitted"
        import json as _json
        artifact = _json.loads(artifact_path.read_text())

        # Canonical Provenance + Catalog #341 non-promotable markers
        prov = artifact["canonical_provenance"]
        assert prov["promotable"] is False
        assert prov["score_claim"] is False
        assert prov["evidence_grade"] == "macOS-MLX research-signal"
        assert prov["axis_tag"] == "[macOS-MLX research-signal]"

        # Empirical measurement structure
        assert "y_derivable_prefix_density_per_source" in artifact
        sources = set(artifact["y_derivable_prefix_density_per_source"].keys())
        assert sources == {"math_constants", "torch_defaults", "ImageNet", "Comma2k19"}

        # WZPSC01 archive emitted
        assert artifact["wzpsc01_archive"]["bytes_len"] > 0
        assert artifact["roundtrip_byte_identical"] is True

        # Verdict structure
        assert artifact["verdict"]["kind"].endswith(
            "FALSIFICATION_PER_CATALOG_307"
        ) or "PARTIAL" in artifact["verdict"]["kind"] or "SUB_FRONTIER" in artifact["verdict"]["kind"]


def test_l1_full_main_fails_closed_on_missing_base_substrate_bytes():
    """The L1 harness MUST fail closed (return 1) when base bytes don't exist.

    Per CLAUDE.md "Forbidden CLI flag inventions" + Catalog #152 required-
    input validation: missing files surface as rc=1 + stderr message.
    """
    from pathlib import Path as _Path
    parser = build_arg_parser()
    args = parser.parse_args([
        "--full",
        "--output-dir", "/nonexistent_out_dir_for_test",
        "--base-substrate-bytes-path",
        "/nonexistent/path/that/does/not/exist.pt",
    ])
    rc = _full_main(args)
    assert rc == 1


def test_smoke_main_returns_zero_on_byte_identical_roundtrip():
    """Catalog #325 6-step contract: smoke_main MUST exit 0 on byte-identical roundtrip."""
    parser = build_arg_parser()
    args = parser.parse_args(["--smoke"])
    rc = _smoke_main(args)
    assert rc == 0


def test_main_cli_smoke_flag():
    """The canonical main() entry-point dispatches --smoke to _smoke_main."""
    rc = main(["--smoke"])
    assert rc == 0


def test_main_cli_no_flag_prints_help_and_returns_one():
    """main() without --smoke or --full prints help + returns rc=1."""
    rc = main([])
    assert rc == 1
