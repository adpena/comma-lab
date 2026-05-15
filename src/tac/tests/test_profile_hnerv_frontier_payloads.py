# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_hnerv_frontier_payloads.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "profile_hnerv_frontier_payloads_under_test",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


profile = _load_module()


def test_pr106_sidecar_magic_wins_over_pr101_path_heuristic() -> None:
    decoder = b"decoder-brotli-bytes"
    latents = b"latent-brotli-bytes"
    inner = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
    sidecar = b"ranked-sidecar"
    framing = b"\x00\x00\x00\x00\x01\x01"
    payload = (
        bytes([0xFE, 0x02])
        + len(inner).to_bytes(4, "little")
        + inner
        + len(sidecar).to_bytes(2, "little")
        + sidecar
        + framing
    )

    kind = profile.infer_profile_kind(
        "auto",
        Path("contains_pr101_name/pr106_r2_pr101_grammar.zip"),
        "x",
        payload,
    )
    sections = profile.profile_payload(kind, payload)

    assert kind == "pr106_sidecar_wrapper"
    assert [section.name for section in sections] == [
        "pr106_sidecar_header_fe_fmt_len_u32",
        "inner_packed_header_ff_len24",
        "inner_decoder_packed_brotli",
        "inner_latents_and_sidecar_brotli",
        "sidecar_len_u16",
        "sidecar_payload_pr101_ranked_no_op",
        "sidecar_framing_meta_pr101",
    ]
    assert sections[2].bytes == len(decoder)
    assert sections[3].bytes == len(latents)
    assert sections[5].bytes == len(sidecar)


def test_pr106_fixed_meta_rank_elided_profile_has_no_length_or_framing_sections() -> None:
    decoder = b"decoder-brotli-bytes"
    latents = b"latent-brotli-bytes"
    inner = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
    sidecar = b"fixed-meta-rank-elided-sidecar"
    payload = bytes([0xFE, 0x05]) + len(inner).to_bytes(4, "little") + inner + sidecar

    sections = profile.profile_payload("pr106_sidecar_wrapper", payload)

    assert [section.name for section in sections] == [
        "pr106_sidecar_header_fe_fmt_len_u32",
        "inner_packed_header_ff_len24",
        "inner_decoder_packed_brotli",
        "inner_latents_and_sidecar_brotli",
        "sidecar_payload_pr101_fixed_meta_rank_elided",
    ]
    assert sections[2].bytes == len(decoder)
    assert sections[3].bytes == len(latents)
    assert sections[4].bytes == len(sidecar)


def test_pr106_rank_elided_profile_splits_terminal_framing_meta() -> None:
    decoder = b"decoder-brotli-bytes"
    latents = b"latent-brotli-bytes"
    inner = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
    sidecar = b"rank-elided-sidecar"
    framing = b"\x01\x02\x03\x04\x05"
    payload = bytes([0xFE, 0x04]) + len(inner).to_bytes(4, "little") + inner + sidecar + framing

    sections = profile.profile_payload("pr106_sidecar_wrapper", payload)

    assert [section.name for section in sections] == [
        "pr106_sidecar_header_fe_fmt_len_u32",
        "inner_packed_header_ff_len24",
        "inner_decoder_packed_brotli",
        "inner_latents_and_sidecar_brotli",
        "sidecar_payload_pr101_rank_elided",
        "sidecar_framing_meta_pr101_rank_elided",
    ]
    assert sections[4].bytes == len(sidecar)
    assert sections[5].bytes == len(framing)
