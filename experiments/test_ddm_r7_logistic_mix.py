# SPDX-License-Identifier: MIT
from __future__ import annotations

import base64
import hashlib
import importlib.util
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("ddm_r7_logistic_mix.py")
SPEC = importlib.util.spec_from_file_location("ddm_r7_logistic_mix", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mix)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\x00",
        b"\x00" * 1024,
        b"\xff\x00" * 513,
        bytes(range(256)) * 4,
        b"static-mode-residual" * 97,
    ],
)
def test_exact_deterministic_roundtrip_and_accounting(raw: bytes) -> None:
    first = mix.encode_logistic_mix(raw)
    assert mix.encode_logistic_mix(raw) == first
    assert mix.decode_logistic_mix(first) == raw
    accounting = mix.frame_accounting(first)
    assert accounting["model_parameter_bytes"] == 0
    assert accounting["framed_bytes"] == (accounting["header_bytes"] + accounting["coded_payload_bytes"])


def test_bytes_like_inputs_and_non_bytes_refusal() -> None:
    raw = b"bytes-like-contract" * 4
    expected = mix.encode(raw)
    assert mix.encode(bytearray(raw)) == expected
    assert mix.encode(memoryview(raw)) == expected
    assert mix.decode(bytearray(expected)) == raw
    assert mix.decode(memoryview(expected)) == raw
    with pytest.raises(TypeError):
        mix.encode("not bytes")
    with pytest.raises(TypeError):
        mix.decode("not bytes")


def test_true_logit_domain_combination_is_not_probability_average() -> None:
    frequencies = (410, 3277, 1024, 2867)
    logits = tuple(mix._LOGIT_TABLE[frequency - 1] for frequency in frequencies)
    weights = [
        mix.WEIGHT_SCALE // 2,
        mix.WEIGHT_SCALE // 4,
        mix.WEIGHT_SCALE // 8,
        mix.WEIGHT_SCALE // 8,
    ]
    combined = mix._mix_logit(logits, weights, 0)
    expected = mix._round_div_signed(
        sum(weight * logit for weight, logit in zip(weights, logits, strict=True)),
        mix.WEIGHT_SCALE,
    )
    assert combined == expected
    logistic_frequency = mix._frequency_from_logit(combined)
    probability_average = mix._round_div_signed(
        sum(weight * frequency for weight, frequency in zip(weights, frequencies, strict=True)),
        mix.WEIGHT_SCALE,
    )
    assert logistic_frequency != probability_average


def test_online_error_gradient_updates_weights_in_expected_direction() -> None:
    mixer = mix._Mixer()
    mixer.global_counts[:] = [1, 15]
    mixer.position_counts[0][:] = [15, 1]
    probability_one, logits, order_key, run_key = mixer.predict(0, 0)
    before = tuple(mixer.weights)
    assert logits[0] > 0
    assert logits[1] < 0
    mixer.update(
        bit=1,
        probability_one=probability_one,
        expert_logits=logits,
        bit_position=0,
        order_key=order_key,
        run_key=run_key,
    )
    assert mixer.weights[0] > before[0]
    assert mixer.weights[1] < before[1]
    assert mixer.bias > 0


def test_strict_frame_rejects_truncation_trailer_headers_hash_and_stream() -> None:
    frame = mix.encode(b"strict-canonical-frame" * 31)
    corruptions = [
        frame[:-1],
        frame + b"\x00",
        bytes([frame[0] ^ 1]) + frame[1:],
        frame[:4] + bytes([frame[4] ^ 1]) + frame[5:],
        frame[:5] + bytes([frame[5] ^ 1]) + frame[6:],
        frame[:6] + bytes([frame[6] ^ 1]) + frame[7:],
        frame[:7] + bytes([frame[7] ^ 1]) + frame[8:],
        frame[:-1] + bytes([frame[-1] ^ 1]),
    ]
    digest_offset = mix._HEADER.size - 32
    digest_mutation = bytearray(frame)
    digest_mutation[digest_offset] ^= 1
    corruptions.append(bytes(digest_mutation))
    stream_mutation = bytearray(frame)
    stream_mutation[mix._HEADER.size + len(frame[mix._HEADER.size :]) // 2] ^= 1
    corruptions.append(bytes(stream_mutation))

    for corrupted in corruptions:
        with pytest.raises(mix.LogisticMixError):
            mix.decode(corrupted)


def test_declared_length_caps_and_zero_coded_length_fail_closed() -> None:
    frame = bytearray(mix.encode(b"length-contract"))
    fields = list(mix._HEADER.unpack_from(frame))
    fields[5] = mix.MAX_RAW_BYTES + 1
    too_large = mix._HEADER.pack(*fields) + bytes(frame[mix._HEADER.size :])
    with pytest.raises(mix.LogisticMixError, match="format cap"):
        mix.decode(too_large)

    fields = list(mix._HEADER.unpack_from(frame))
    fields[6] = 0
    zero_length = mix._HEADER.pack(*fields)
    with pytest.raises(mix.LogisticMixError, match=r"truncated|length"):
        mix.decode(zero_length)


def test_stdlib_only_decoder_works_in_isolated_python() -> None:
    raw = bytes(range(256)) * 2
    encoded = mix.encode(raw)
    script = f"""
import base64
import importlib.util
spec = importlib.util.spec_from_file_location("codec", {str(MODULE_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
frame = base64.b64decode({base64.b64encode(encoded).decode("ascii")!r})
raw = module.decode(frame)
assert raw == bytes(range(256)) * 2
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def test_fixed_point_table_and_wire_golden_vector() -> None:
    assert len(mix._LOGIT_TABLE) == mix.PROBABILITY_TOTAL - 1
    assert all(left < right for left, right in pairwise(mix._LOGIT_TABLE))
    frame = mix.encode(b"DDM-R7 logistic mixer golden vector\x00\xff" * 3)
    assert hashlib.sha256(frame).hexdigest() == ("04f0c161786484345f37d121096bc2c514bd0ebf227bcd0da3332b8ffbe1e530")
