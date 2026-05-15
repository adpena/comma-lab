# SPDX-License-Identifier: MIT
"""Round-trip parity test between encoder + vendored decoder.

The encoder lives in ``tac.codec.factorized_hnerv_codec``; the decoder is
vendored into ``submissions/factorized_hnerv_v1/src/codec.py`` so the
inflate-time path has zero ``tac`` dependency. This test asserts the
vendored decoder is a bit-faithful inverse of the encoder.

If this test fails, the vendored codec.py drifted from the canonical
implementation and must be re-synced.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.codec.factorized_hnerv_codec import (
    FIXED_STATE_SCHEMA,
    FactorizedSectionPlan,
    decode_factorized_section as canonical_decode,
    encode_factorized_section,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VENDORED_CODEC_PATH = REPO_ROOT / "submissions/factorized_hnerv_v1/src/codec.py"


def _load_vendored_codec():
    spec = importlib.util.spec_from_file_location(
        "_test_factorized_v1_codec", VENDORED_CODEC_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_factorized_v1_codec"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vendored_decoder_matches_canonical_decoder_on_synthetic():
    """Vendored decoder must produce the same state_dict as the canonical
    decoder for the same encoded bytes.
    """
    torch.manual_seed(0)
    sd = {n: torch.randn(*s) for n, s in FIXED_STATE_SCHEMA}
    plan = FactorizedSectionPlan(
        factorized_indices=(0, 2, 4),
        per_index_rank={0: 12, 2: 32, 4: 32},
    )
    section, _ = encode_factorized_section(sd, plan, brotli_quality=5)

    canonical_sd = canonical_decode(section)
    vendored = _load_vendored_codec()
    vendored_sd = vendored.decode_factorized_section(section)

    assert set(canonical_sd.keys()) == set(vendored_sd.keys())
    for name in canonical_sd:
        np.testing.assert_array_equal(
            canonical_sd[name].detach().cpu().numpy(),
            vendored_sd[name].detach().cpu().numpy(),
            err_msg=f"vendored decoder produced different bytes for {name!r}",
        )


def test_vendored_section_magic_constant_matches():
    vendored = _load_vendored_codec()
    from tac.codec.factorized_hnerv_codec import SECTION_MAGIC
    assert vendored.SECTION_MAGIC == SECTION_MAGIC


def test_vendored_archive_magic_byte_present():
    """Sanity: the archive-level magic byte the inflate.py uses is present."""
    vendored = _load_vendored_codec()
    assert vendored.ARCHIVE_MAGIC == 0xF1


def test_vendored_decoder_rejects_malformed_archive():
    vendored = _load_vendored_codec()
    bad = b"\xFF\x00\x00\x00\x00"  # wrong archive magic
    with pytest.raises(ValueError, match="bad archive magic"):
        vendored.parse_archive(bad)
