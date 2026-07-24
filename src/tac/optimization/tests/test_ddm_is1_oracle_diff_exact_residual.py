from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from tac.optimization.ddm_is1_oracle_diff_exact_residual import (
    OracleDiffPriceConfigV1,
    OracleDiffPriceError,
    decode_exact_correction_stage,
    encode_exact_correction_stage,
)


def _delta(pairs: int = 2) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(
        -31,
        32,
        size=(pairs, 2, 384, 512, 3),
        dtype=np.int16,
    )


def test_exact_correction_record_is_deterministic_and_reversible() -> None:
    delta = _delta()
    first = encode_exact_correction_stage(delta, pair_start=9, pair_stop=11)
    second = encode_exact_correction_stage(delta, pair_start=9, pair_stop=11)
    assert first.record == second.record
    header, decoded = decode_exact_correction_stage(first.record)
    assert header["pair_start"] == 9
    assert header["pair_stop"] == 11
    assert header["coder"] in {"zlib9", "lzma0"}
    assert np.array_equal(decoded, delta)


def test_exact_correction_record_refuses_tamper() -> None:
    encoded = encode_exact_correction_stage(
        _delta(1),
        pair_start=0,
        pair_stop=1,
    )
    altered = bytearray(encoded.record)
    altered[-1] ^= 1
    with pytest.raises(OracleDiffPriceError, match="payload custody"):
        decode_exact_correction_stage(bytes(altered))


def test_exact_correction_geometry_is_strict() -> None:
    with pytest.raises(OracleDiffPriceError, match="shape"):
        encode_exact_correction_stage(
            np.zeros((1, 2, 8, 8, 3), dtype=np.int16),
            pair_start=0,
            pair_stop=1,
        )
    with pytest.raises(OracleDiffPriceError, match="bounded"):
        bad = _delta(1)
        bad[0, 0, 0, 0, 0] = 256
        encode_exact_correction_stage(bad, pair_start=0, pair_stop=1)


def test_config_is_fail_closed() -> None:
    value = {
        "schema": "ddm_is1_oracle_diff_exact_residual_config.v1",
        "run_id": "ddm_is1_oracle_diff_exact_residual_n600_20260724",
        "source_config_path": "source.json",
        "source_config_sha256": "0" * 64,
        "pair_count": 600,
        "chunk_size": 12,
        "coders": ["zlib9", "lzma0"],
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
    }
    config = OracleDiffPriceConfigV1.model_validate_json(json.dumps(value))
    assert json.loads(config.model_dump_json(by_alias=True))["score_claim"] is False
    value["score_claim"] = True
    with pytest.raises(ValidationError):
        OracleDiffPriceConfigV1.model_validate_json(json.dumps(value))
