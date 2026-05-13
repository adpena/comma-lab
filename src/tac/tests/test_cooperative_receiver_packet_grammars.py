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
        "TT5L",
        "SBO1",
        "S2SB",
        "CMLR",
        "DPW1",
        # ── A1 + sidecar composition grammars (solver-stack wire-in 2026-05-13) ──
        "LPA1",  # A1 + LAPose foveal RGB residual sidecar
        "WAV1",  # A1 + DB4 IDWT detail-band wavelet residual sidecar
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
