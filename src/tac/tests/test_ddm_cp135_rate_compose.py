from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_cp135_rate_compose as compose
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_packed_unsigned_round_trip_and_padding_guard() -> None:
    values = np.asarray([0, 1, 63, 64, 127, 5, 19, 72, 4, 3, 2, 1], dtype=np.int16)
    packed = compose._pack_unsigned(values, 7)
    assert np.array_equal(compose._unpack_unsigned(packed, len(values), 7), values)

    noncanonical = bytearray(compose._pack_unsigned(np.asarray([1], dtype=np.int16), 1))
    noncanonical[-1] |= 0x80
    with pytest.raises(RuntimeError, match="nonzero padding"):
        compose._unpack_unsigned(bytes(noncanonical), 1, 1)


def test_deterministic_zip_is_at_structural_floor() -> None:
    member = b"real-payload"
    first = compose.deterministic_zip(member)
    second = compose.deterministic_zip(member)
    assert first == second
    assert len(first) == len(member) + 100


def test_cp135_python_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_cp135_rate_compose.py",
            "src/tac/tests/test_ddm_cp135_rate_compose.py",
        ),
    )
    assert findings == []


def test_lotto_nibble_selectors_round_trip_and_refuse_padding() -> None:
    values = np.asarray([0, 14, 7, 3, 12], dtype=np.uint8)
    packed = compose._pack_nibbles_low_first(values)
    assert np.array_equal(compose._unpack_nibbles_low_first(packed, len(values)), values)

    noncanonical = bytearray(packed)
    noncanonical[-1] |= 0xF0
    with pytest.raises(RuntimeError, match="framing differs"):
        compose._unpack_nibbles_low_first(bytes(noncanonical), len(values))


@pytest.mark.skipif(not compose.DEFAULT_EXPERIMENT_BOOK.is_dir(), reason="PR135 ExperimentBook unavailable")
def test_checkpointed_rc64_resume_is_byte_identical() -> None:
    args = compose.parser().parse_args(["encode-rc64"])
    library = compose._compile_checkpointable_rc64(args)
    sys.path.insert(0, str(compose.DEFAULT_EXPERIMENT_BOOK / "src"))
    try:
        from cpr1_sub4.entropy.rc64 import NativeDecoder, NativeEncoder
    finally:
        sys.path.pop(0)
    probabilities = np.tile(np.asarray([[0.05, 0.10, 0.20, 0.25, 0.40]], dtype=np.float32), (8, 1))
    symbols = np.asarray([4, 3, 2, 1, 0, 4, 2, 3], dtype=np.int32)

    partial = NativeEncoder(library)
    partial.encode(symbols[:4], probabilities[:4])
    resumed = compose._rc64_resume(NativeEncoder, library, compose._rc64_snapshot(partial))
    resumed.encode(symbols[4:], probabilities[4:])
    resumed_payload = resumed.finish()

    direct = NativeEncoder(library)
    direct.encode(symbols, probabilities)
    assert resumed_payload == direct.finish()
    decoder = NativeDecoder(library, resumed_payload)
    assert np.array_equal(decoder.decode(probabilities), symbols)


@pytest.mark.skipif(not compose.DEFAULT_ARCHIVE.is_file(), reason="custodied PR135 archive unavailable")
def test_exact_pr135_representation_round_trips() -> None:
    runtime = compose.load_runtime(compose.DEFAULT_RUNTIME)
    parts = runtime.read_residual_archive(compose.DEFAULT_ARCHIVE)
    base = compose._base_physical_models(compose.DEFAULT_ARCHIVE)
    raw, hpac, semantic, carrier = compose._physical_model_parts(parts, parts.hpac_blob, base)
    assert raw == base

    step2, report = compose.step2_ihs2(parts.hpac_blob)
    assert len(step2) == len(parts.hpac_blob)
    assert report["changed_values"] == 2_371

    packed, cap1 = compose.pack_cap1_metadata(carrier)
    assert cap1["raw_delta"] == -40
    assert compose.unpack_cap1_metadata(packed) == carrier

    brotli = shutil.which("brotli")
    if brotli is None:
        pytest.skip("Brotli CLI unavailable")
    model, _ = compose.pack_split_models(
        hpac,
        semantic,
        packed,
        qualities=(11, 11, 10),
        brotli_binary=brotli,
    )
    restored = compose.unpack_split_models(model, brotli_binary=brotli)
    assert restored[:2] == (hpac, semantic)
    assert compose.unpack_cap1_metadata(restored[2]) == carrier
