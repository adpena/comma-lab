from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from experiments.ddm_sd1_semantic_rd_curve import (
    pack_semantic_state,
    pack_signed_bits,
    load_or_initialize_progress,
    semantic_delta_s,
    stratified_random_pair_ids,
    unpack_semantic_state,
    unpack_signed_bits,
)


def _toy_state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        [
            ("embed.weight", torch.linspace(-1.0, 1.0, 15).reshape(3, 5)),
            ("conv.weight", torch.linspace(-0.7, 0.9, 24).reshape(4, 3, 2, 1)),
            ("conv.bias", torch.linspace(-0.2, 0.3, 4)),
        ]
    )


def test_signed_bit_pack_roundtrip_for_required_depths() -> None:
    for bits in (3, 4, 5):
        limit = (1 << (bits - 1)) - 1
        codes = torch.arange(-limit, limit + 1, dtype=torch.int8)
        blob = pack_signed_bits(codes, bits)
        decoded, remaining = unpack_signed_bits(memoryview(blob), codes.numel(), bits)
        assert not remaining
        assert torch.equal(decoded, codes)


def test_legacy_and_mixed_semantic_roundtrip() -> None:
    state = _toy_state()
    qnames = [name for name, value in state.items() if value.ndim >= 2]
    for allocation, legacy in (
        (OrderedDict((name, 4) for name in qnames), True),
        (OrderedDict(zip(qnames, (3, 5), strict=True)), False),
    ):
        blob, expected = pack_semantic_state(
            state, allocation, legacy_int4=legacy
        )
        decoded, decoded_allocation, format_name = unpack_semantic_state(blob, state)
        assert decoded_allocation == allocation
        assert format_name == ("legacy_int4" if legacy else "sd1_mixed_v1")
        assert list(decoded) == list(expected)
        for name in expected:
            assert torch.equal(decoded[name], expected[name])


def test_semantic_delta_s_uses_actual_archive_bytes() -> None:
    value, delta_dseg, delta_bytes = semantic_delta_s(
        101, 900, 100, 1000, 1000
    )
    assert delta_dseg == 0.001
    assert delta_bytes == -100
    assert value == 0.1 - 2500 / 37_545_489


def test_screen_is_seeded_stratified_random_not_prefix() -> None:
    selected = stratified_random_pair_ids(seed=20260809)
    assert len(selected) == 120
    assert len(set(selected)) == 120
    assert selected != list(range(120))
    for stratum in range(10):
        assert sum(stratum * 60 <= value < (stratum + 1) * 60 for value in selected) == 12
    assert selected == stratified_random_pair_ids(seed=20260809)


def test_resume_refuses_any_fingerprint_drift(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    load_or_initialize_progress(
        progress_path,
        {"measurement_source": {"sha256": "first"}},
        ["probe"],
        "[macOS-CPU advisory]",
    )
    with pytest.raises(ValueError, match="fingerprints do not match"):
        load_or_initialize_progress(
            progress_path,
            {"measurement_source": {"sha256": "changed"}},
            ["probe"],
            "[macOS-CPU advisory]",
        )
