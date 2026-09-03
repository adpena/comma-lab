from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_sp9_semantic_primary_corrected_race as sp9


def _fixture() -> np.ndarray:
    return np.array(
        [
            [[0, 0, 1, 1, 1], [2, 2, 2, 3, 4]],
            [[0, 0, 1, 1, 1], [2, 2, 2, 3, 3]],
        ],
        dtype=np.uint8,
    )


def test_exact_primary_entropy_and_pair_reset() -> None:
    result = sp9.analyze_field(_fixture(), (2, 2, 5))
    primary = result["primary"]

    # Seven run events occupy six joint cells: one cell has count two and
    # five have count one. Bucket residual widths sum to six exact bits.
    expected_joint_bits = 7 * math.log2(7) - 2
    assert primary["joint_mle_entropy_bits"] == pytest.approx(expected_joint_bits)
    assert primary["raw_residual_bits"] == 6
    assert primary["ideal_bits"] == pytest.approx(expected_joint_bits + 6)
    assert result["total_runs"] == 7
    assert sum(row["count"] for row in result["joint_transition_counts"]) == 5

    # Context resets between pairs: all three within-pair transitions are
    # deterministic, so both optimistic first-order conditional terms are zero.
    relaxations = result["relaxations"]
    assert (
        relaxations["joint_event_first_order_previous_event_free_at_pair_start"][
            "conditional_bits"
        ]
        == 0.0
    )
    assert (
        relaxations["exact_event_first_order_previous_event_free_at_pair_start"][
            "conditional_bits"
        ]
        == 0.0
    )


def test_wrong_shape_and_alphabet_refuse() -> None:
    fixture = _fixture()
    with pytest.raises(sp9.Sp9CeilingError, match="field shape"):
        sp9.analyze_field(fixture, (1, 2, 5))
    malformed = fixture.copy()
    malformed[1, 1, 3] = 5
    with pytest.raises(sp9.Sp9CeilingError, match="outside"):
        sp9.analyze_field(malformed, malformed.shape)


def test_incomplete_full_alphabet_refuses() -> None:
    fixture = _fixture()
    fixture[fixture == 4] = 3
    with pytest.raises(sp9.Sp9CeilingError, match="full field alphabet"):
        sp9.analyze_field(fixture, fixture.shape)


def test_pin_mismatch_refuses_before_output(tmp_path: Path) -> None:
    source = tmp_path / "field.u8"
    source.write_bytes(_fixture().tobytes())
    output = tmp_path / "CEILING_RESULT.json"
    with pytest.raises(sp9.Sp9CeilingError, match="SHA-256"):
        sp9.execute_ceiling(
            source,
            output,
            expected_shape=_fixture().shape,
            expected_bytes=source.stat().st_size,
            expected_sha256="0" * 64,
            minimum_free_bytes=0,
            free_space_probe=lambda _path: 10_000,
            storage_root=tmp_path,
        )
    assert not output.exists()


def test_shape_byte_contract_refuses_before_hash_or_output(tmp_path: Path) -> None:
    source = tmp_path / "field.u8"
    source.write_bytes(_fixture().tobytes())
    output = tmp_path / "CEILING_RESULT.json"
    with pytest.raises(sp9.Sp9CeilingError, match="shape implies"):
        sp9.execute_ceiling(
            source,
            output,
            expected_shape=_fixture().shape,
            expected_bytes=source.stat().st_size + 1,
            expected_sha256="not-reached",
            minimum_free_bytes=0,
            free_space_probe=lambda _path: 10_000,
            storage_root=tmp_path,
        )
    assert not output.exists()


def test_low_space_refuses_before_output(tmp_path: Path) -> None:
    source = tmp_path / "field.u8"
    source.write_bytes(_fixture().tobytes())
    output = tmp_path / "CEILING_RESULT.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(sp9.Sp9CeilingError, match="free space"):
        sp9.execute_ceiling(
            source,
            output,
            expected_shape=_fixture().shape,
            expected_bytes=source.stat().st_size,
            expected_sha256=digest,
            minimum_free_bytes=10_001,
            free_space_probe=lambda _path: 10_000,
            storage_root=tmp_path,
        )
    assert not output.exists()


def test_source_change_during_analysis_refuses_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "field.u8"
    source.write_bytes(_fixture().tobytes())
    output = tmp_path / "CEILING_RESULT.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    observed_hashes = iter((digest, "f" * 64))
    monkeypatch.setattr(sp9, "sha256_file", lambda _path: next(observed_hashes))

    with pytest.raises(sp9.Sp9CeilingError, match="changed during exact analysis"):
        sp9.execute_ceiling(
            source,
            output,
            expected_shape=_fixture().shape,
            expected_bytes=source.stat().st_size,
            expected_sha256=digest,
            minimum_free_bytes=0,
            free_space_probe=lambda _path: 10_000,
            storage_root=tmp_path,
        )
    assert not output.exists()


def test_atomic_deterministic_scalar_receipt_and_no_candidate_payload(tmp_path: Path) -> None:
    source = tmp_path / "field.u8"
    source.write_bytes(_fixture().tobytes())
    output = tmp_path / "retained" / "CEILING_RESULT.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    free_values = iter((10_000, 9_000))
    kwargs = {
        "expected_shape": _fixture().shape,
        "expected_bytes": source.stat().st_size,
        "expected_sha256": digest,
        "minimum_free_bytes": 0,
        "free_space_probe": lambda _path: next(free_values),
        "storage_root": tmp_path,
    }
    first = sp9.execute_ceiling(source, output, **kwargs)
    first_bytes = output.read_bytes()
    second = sp9.execute_ceiling(source, output, **kwargs)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes)["decision"]["coder_built"] is False
    assert sorted(path.name for path in output.parent.iterdir()) == ["CEILING_RESULT.json"]
