# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.research_basis import (
    REQUIRED_SOURCE_FIELDS,
    RESEARCH_SOURCES,
    ResearchBasisError,
    canonical_research_basis_id,
    research_basis_ids_for_family,
    research_basis_manifest,
)


def test_research_basis_manifest_is_planning_only_and_hardened() -> None:
    manifest = research_basis_manifest(["lapose_2026", "foveated_telepresence_2025"])

    assert manifest["planning_only"] is True
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert "exact_cuda_auth_eval_required" in manifest["global_hardening_blockers"]
    assert [source["basis_id"] for source in manifest["sources"]] == [
        "foveated_telepresence_2025",
        "lapose_2026",
    ]
    for source in manifest["sources"]:
        assert source["url"].startswith("https://")
        assert source["charged_byte_contract"]
        assert source["hardening_blockers"]
        assert source["contest_terms"]


def test_family_lookup_prefers_latest_family_specific_sources() -> None:
    assert "lapose_2026" in research_basis_ids_for_family("lapose")
    assert "geometric_visual_servo_ot_2026" in research_basis_ids_for_family("lapose")
    assert "telescope_2026" in research_basis_ids_for_family("telescopic_foveation")
    assert "foveated_diffusion_2026" in research_basis_ids_for_family("telescopic_foveation")
    assert "foveated_telepresence_2025" in research_basis_ids_for_family("foveation")
    assert "geometric_visual_servo_ot_2026" in research_basis_ids_for_family(
        "optimal_transport"
    )
    assert "rdc_universal_2025" in research_basis_ids_for_family("categorical")
    assert "flavc_2025" in research_basis_ids_for_family("entropy")
    assert "dworetzky_fridrich_detector_batch_2025" in research_basis_ids_for_family(
        "meta_lagrangian"
    )
    assert "yousfi_onehot_jpeg_2020" in research_basis_ids_for_family("categorical")
    assert "constriction_ans" in research_basis_ids_for_family("aq_huffman")
    assert "ans_duda_2009" in research_basis_ids_for_family("aq_huffman")
    assert "rans_duda_2013" in research_basis_ids_for_family("aq_huffman")
    assert "minnen_joint_priors_2018" in research_basis_ids_for_family("entropy")
    assert "tensorflow_compression" in research_basis_ids_for_family("entropy")
    assert "cool_chic_5_2026" in research_basis_ids_for_family("cool_chic")
    assert "cool_chic_2023" in research_basis_ids_for_family("cool_chic")
    assert "c3_neural_compression_2024" in research_basis_ids_for_family("c3")
    assert "vqvae_2017" in research_basis_ids_for_family("vqvae")
    assert "siren_2020" in research_basis_ids_for_family("siren")
    assert "fourier_features_2020" in research_basis_ids_for_family("siren")
    assert "finer_2024" in research_basis_ids_for_family("inr")
    assert "bacon_2022" in research_basis_ids_for_family("inr")
    assert "wire_2023" in research_basis_ids_for_family("wavelet")
    assert "mallat_mra_1989" in research_basis_ids_for_family("wavelet")
    assert "raft_2020" in research_basis_ids_for_family("raft")
    l5_v2_ids = research_basis_ids_for_family("time_traveler_l5_v2")
    assert l5_v2_ids == [
        "rao_ballard_1999",
        "friston_free_energy_2010",
        "dreamerv3_2023",
        "ha_schmidhuber_world_models_2018",
        "pnvc_2025",
        "dcvc_rt_2025",
        "unified_intra_inter_nvc_2025",
        "glvc_2025",
        "gnvc_vd_2025",
        "snerv_spectra_2025",
        "metanerv_2025",
        "c3_neural_compression_2024",
        "atick_redlich_1990",
        "wyner_ziv_1976",
        "lu_dvc_2019",
        "rissanen_mdl_1978",
        "mackay_itila_2003",
        "tishby_information_bottleneck_1999",
        "tishby_zaslavsky_2015",
        "balle_hyperprior_2018",
        "hnerv_2023",
    ]
    assert research_basis_ids_for_family("predictive_receiver")[:4] == [
        "rao_ballard_1999",
        "friston_free_energy_2010",
        "dreamerv3_2023",
        "ha_schmidhuber_world_models_2018",
    ]
    assert "rissanen_mdl_1978" in research_basis_ids_for_family("mdl")
    assert "mackay_itila_2003" in research_basis_ids_for_family("mdl")


def test_legacy_research_basis_aliases_resolve_to_canonical_ids() -> None:
    assert canonical_research_basis_id("balle_2018") == "balle_hyperprior_2018"
    assert canonical_research_basis_id("Rao-Ballard1999") == "rao_ballard_1999"
    assert canonical_research_basis_id("dvc_2019") == "lu_dvc_2019"
    assert canonical_research_basis_id("wyner_ziv") == "wyner_ziv_1976"
    assert canonical_research_basis_id("pnvc") == "pnvc_2025"
    assert canonical_research_basis_id("snerv_spectra") == "snerv_spectra_2025"
    assert canonical_research_basis_id("metanerv") == "metanerv_2025"
    assert canonical_research_basis_id("dcvc_rt") == "dcvc_rt_2025"
    assert canonical_research_basis_id("unified_intra_inter") == "unified_intra_inter_nvc_2025"
    assert canonical_research_basis_id("glvc") == "glvc_2025"
    assert canonical_research_basis_id("gnvc_vd") == "gnvc_vd_2025"
    manifest = research_basis_manifest(["balle_2018", "balle_hyperprior_2018"])
    assert manifest["source_count"] == 1
    assert manifest["sources"][0]["basis_id"] == "balle_hyperprior_2018"


def test_all_registered_research_sources_satisfy_required_contract() -> None:
    manifest = research_basis_manifest(RESEARCH_SOURCES)

    assert manifest["source_count"] == len(RESEARCH_SOURCES)
    by_id = {source["basis_id"]: source for source in manifest["sources"]}
    assert "awq_2024" in by_id
    assert "lyra2_2026" in by_id
    assert "compression_as_adaptation_2026" in by_id
    assert "rdc_bernoulli_2026" in by_id
    assert "foveated_diffusion_2026" in by_id
    assert "geometric_visual_servo_ot_2026" in by_id
    assert "lu_dvc_2019" in by_id
    assert "pnvc_2025" in by_id
    assert "dcvc_rt_2025" in by_id
    assert "unified_intra_inter_nvc_2025" in by_id
    assert "glvc_2025" in by_id
    assert "gnvc_vd_2025" in by_id
    for basis_id, source in by_id.items():
        for field in REQUIRED_SOURCE_FIELDS:
            assert field in source, f"{basis_id} missing {field}"


def test_unknown_research_basis_fails_closed() -> None:
    with pytest.raises(ResearchBasisError, match="unknown research basis id"):
        research_basis_manifest(["made_up_paper"])
