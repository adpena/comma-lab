# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.packet_compiler.fp11_source_brotli_recode import (
    Fp11SourceBrotliRecodeError,
    parse_decoder_blob_len,
    patch_decoder_blob_len_text,
)


def test_decoder_blob_len_patch_is_exactly_scoped() -> None:
    source = "A = 1\nDECODER_BLOB_LEN = 162_164\nB = 2\n"

    patched = patch_decoder_blob_len_text(source, new_len=162127)

    assert parse_decoder_blob_len(source) == 162164
    assert patched == "A = 1\nDECODER_BLOB_LEN = 162127\nB = 2\n"
    assert parse_decoder_blob_len(patched) == 162127


def test_decoder_blob_len_patch_fails_closed_on_ambiguous_runtime() -> None:
    source = "DECODER_BLOB_LEN = 10\nDECODER_BLOB_LEN = 11\n"

    with pytest.raises(Fp11SourceBrotliRecodeError, match="expected one"):
        patch_decoder_blob_len_text(source, new_len=9)
