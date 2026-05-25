# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.packet_compiler.cooperative_receiver_grammars import (
    COOPERATIVE_RECEIVER_PACKET_GRAMMARS,
    compiler_hook_rows,
    xray_magic_signatures,
    xray_substrate_classes,
)


def test_cooperative_receiver_packet_grammars_are_unique_four_byte_ascii() -> None:
    magics = [row.magic for row in COOPERATIVE_RECEIVER_PACKET_GRAMMARS]

    assert len(magics) == len(set(magics))
    registered = {magic.decode("ascii") for magic in magics}
    assert {
        "DFL1",
        "TT5L",
        "SBO1",
        "S2SB",
        "CMLR",
        "DPW1",
        # ── A1 + sidecar composition grammars (solver-stack wire-in 2026-05-13) ──
        "LPA1",  # A1 + LAPose foveal RGB residual sidecar
        "WAV1",  # A1 + DB4 IDWT detail-band wavelet residual sidecar
        "FGS1",  # HDM8/frame0 postdecode selector sidecar
        "FES1",  # PR106 frame-exploit selector sidecar
    }.issubset(registered)
    assert all(len(magic) == 4 for magic in magics)


def test_xray_and_compiler_views_share_the_same_registry() -> None:
    signatures = xray_magic_signatures()
    classes = xray_substrate_classes()
    compiler_rows = compiler_hook_rows()

    assert len(signatures) == len(COOPERATIVE_RECEIVER_PACKET_GRAMMARS)
    assert len(classes) == len(COOPERATIVE_RECEIVER_PACKET_GRAMMARS)
    assert len(compiler_rows) == len(COOPERATIVE_RECEIVER_PACKET_GRAMMARS)
    assert {row["magic_ascii"] for row in compiler_rows} == {
        magic.decode("ascii") for magic, _label in signatures
    }
    assert all(row["score_claim"] is False for row in compiler_rows)
    assert all(row["ready_for_exact_eval_dispatch"] is False for row in compiler_rows)

    rows_by_magic = {str(row["magic_ascii"]): row for row in compiler_rows}
    dfl1 = rows_by_magic["DFL1"]
    assert dfl1["xray_label"] == "renderer_payload_dfl1_native_v1"
    assert dfl1["substrate_class"] == "renderer_payload_dfl1_native_packet"
    assert dfl1["campaign_id"] == "codex_renderer_payload_dfl1_native_20260525"
    assert "parser smoke only" in str(dfl1["notes"])


def test_frame0_selector_is_visible_as_stackable_postdecode_compiler_atom() -> None:
    rows = {
        str(row["magic_ascii"]): row
        for row in compiler_hook_rows()
    }

    row = rows["FGS1"]
    assert row["substrate_class"] == "frame0_postdecode_selector_packet"
    assert row["compiler_stage"] == "postdecode_scorer_aware_selector_pack"
    assert row["source_module"] == "submissions.hdm8_film_grain_sidecar.inflate"
    assert row["score_claim"] is False
    assert row["ready_for_exact_eval_dispatch"] is False

    frame_exploit_row = rows["FES1"]
    assert frame_exploit_row["substrate_class"] == "frame_exploit_selector_sidecar_packet"
    assert frame_exploit_row["compiler_stage"] == "postdecode_pairwise_frame0_selector_pack"
    assert (
        frame_exploit_row["source_module"]
        == "submissions.frame_exploit_selector_sidecar.inflate"
    )
    assert frame_exploit_row["score_claim"] is False
    assert frame_exploit_row["ready_for_exact_eval_dispatch"] is False
