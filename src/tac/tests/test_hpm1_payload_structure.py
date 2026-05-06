from __future__ import annotations

import numpy as np

from tac.hpm1_payload_structure import (
    HPM1_DECODE_REENCODE_BLOCKER_CONTRACT,
    HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT,
    build_hpm1_structural_decode_inventory,
)
from tac.pr91_hpm1_codec import build_hpm1_mask_segment
from tac.repo_io import sha256_bytes


def test_hpm1_structural_inventory_reemits_segment_but_refuses_parity() -> None:
    tokens = (np.arange(16, dtype=np.uint32) % 5).tobytes()
    segment = build_hpm1_mask_segment(
        tokens,
        b"synthetic-hpac-ppmd",
        N=2,
        H=4,
        W=4,
        P=2,
        delta=1,
        ch=4,
        use_spm=True,
        hpac_d_film=2,
    )

    inventory = build_hpm1_structural_decode_inventory(segment)

    assert (
        inventory["hpm1_structural_decode_inventory_contract"]
        == HPM1_STRUCTURAL_DECODE_INVENTORY_CONTRACT
    )
    assert inventory["score_claim"] is False
    assert inventory["dispatch_attempted"] is False
    assert inventory["ready_for_exact_eval_dispatch"] is False
    assert inventory["segment"]["bytes"] == len(segment)
    assert inventory["segment"]["sha256"] == sha256_bytes(segment)
    assert inventory["sections"][0]["name"] == "header"
    assert inventory["sections"][0]["offset"] == 0
    assert inventory["sections"][1]["name"] == "tokens"
    assert inventory["sections"][1]["offset"] == 48
    assert inventory["sections"][1]["bytes"] == len(tokens)
    assert inventory["sections"][2]["name"] == "hpac_ppmd_model"
    assert inventory["sections"][2]["offset"] == 48 + len(tokens)
    assert inventory["token_stream_inventory"]["uint32_word_count"] == 16
    assert inventory["decoded_geometry_contract"]["expected_decoded_symbol_count"] == 32
    assert inventory["structural_reencode"]["matches_source_segment"] is True
    assert inventory["structural_reencode"]["reencoded_segment_sha256"] == sha256_bytes(segment)
    assert inventory["structural_reencode"]["not_semantic_decode_reencode_parity"] is True
    assert inventory["full_decode"]["passed"] is False
    assert inventory["byte_exact_semantic_reencode"]["passed"] is False
    assert (
        inventory["blocker_manifest"]["blocker_manifest_contract"]
        == HPM1_DECODE_REENCODE_BLOCKER_CONTRACT
    )
    assert {
        row["name"] for row in inventory["unsupported_wire_constructs"]
    } == {
        "hpac_autoregressive_probability_rows",
        "constriction_range_decoder_uint32_queue_replay",
        "hpac_context_update_order",
        "range_encoder_uint32_reemit",
        "contest_runtime_sidecar_free_hpm1_loader",
    }
