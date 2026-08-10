"""Focused tests for the ai1 ANS receiver-closure builder."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments" / "ddm_ai1_ans_receiver_integration.py"
SPEC = importlib.util.spec_from_file_location("ddm_ai1", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not import ddm_ai1_ans_receiver_integration")
ai1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ai1
SPEC.loader.exec_module(ai1)


def test_zero_byte_ans_selector_preserves_legacy_models_and_tokens() -> None:
    models = b"legacy-model-section"
    tokens = struct.pack("<III", 4, 7, 11)
    legacy = ai1.receiver.pack_payload(
        models,
        tokens,
        token_codec="range",
        model_codec="legacy_lzma",
    )
    ans = ai1.receiver.pack_payload(
        models,
        tokens,
        token_codec="ans",
        model_codec="legacy_lzma",
    )

    assert len(ans) == len(legacy)
    assert ans[4:] == legacy[4:]
    assert [index for index, pair in enumerate(zip(legacy, ans, strict=True)) if pair[0] != pair[1]] == [3]
    assert ai1.receiver.split_payload(legacy).token_codec == "range"
    assert ai1.receiver.split_payload(ans).token_codec == "ans"
    assert ai1.receiver.split_payload(ans).models == models
    assert ai1.receiver.split_payload(ans).tokens == tokens


def test_deterministic_zip_is_byte_identical_and_stored() -> None:
    payload = struct.pack("<I", 5) + b"model" + struct.pack("<I", 17)
    first = ai1.deterministic_zip(payload)
    second = ai1.deterministic_zip(payload)

    assert first == second
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["p"]
        assert archive.getinfo("p").compress_type == zipfile.ZIP_STORED
        assert archive.read("p") == payload


def test_pinned_python_path_keeps_venv_identity_instead_of_resolving_symlink() -> None:
    assert str(ai1.PINNED_PYTHON).endswith("/venv/bin/python")
    assert ai1.PINNED_PYTHON.absolute() == ai1.PINNED_PYTHON


def test_decode_run_lock_refuses_a_concurrent_owner(tmp_path: Path) -> None:
    import pytest

    lock_path = tmp_path / ".run.lock"
    first = ai1.acquire_run_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="decode already active"):
            ai1.acquire_run_lock(lock_path)
    finally:
        first.close()

    second = ai1.acquire_run_lock(lock_path)
    second.close()


def test_ans_progress_checkpoint_round_trips_and_binds_payload(tmp_path: Path) -> None:
    import numpy as np
    import pytest
    import torch

    inflate = ai1.RUNTIME / "inflate.py"
    spec = importlib.util.spec_from_file_location("ddm_ai1_inflate", inflate)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    prior_path = list(sys.path)
    sys.path.insert(0, str(ai1.RUNTIME))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = prior_path

    tokens = torch.zeros((1, module.EVAL_H, module.EVAL_W), dtype=torch.uint8)
    tokens[:, 3, 7] = 4
    remaining = np.array([3, 5, 8], dtype="<u4")
    binding = {
        "archive_member_sha256": "a" * 64,
        "models_raw_sha256": "b" * 64,
        "token_payload_sha256": "c" * 64,
        "token_codec": "ans",
    }
    cache = tmp_path / "tokens.progress.npz"

    module.write_token_progress_checkpoint(
        tokens,
        remaining,
        binding=binding,
        cache_path=cache,
    )
    restored_tokens, restored_remaining = module.load_token_progress_checkpoint(
        binding=binding,
        cache_path=cache,
    )

    assert torch.equal(restored_tokens, tokens)
    assert np.array_equal(restored_remaining, remaining)
    with pytest.raises(ValueError, match="binding changed"):
        module.load_token_progress_checkpoint(
            binding={**binding, "token_payload_sha256": "d" * 64},
            cache_path=cache,
        )


def test_ans_decoder_continues_exactly_from_retained_stack_words() -> None:
    import constriction
    import numpy as np

    probabilities = np.array(
        [[0.10, 0.20, 0.70], [0.30, 0.30, 0.40], [0.80, 0.10, 0.10]],
        dtype=np.float32,
    )
    symbols = np.array([2, 0, 1], dtype=np.int32)
    family = constriction.stream.model.Categorical(perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(symbols, family, probabilities)

    decoder = constriction.stream.stack.AnsCoder(encoder.get_compressed())
    assert np.array_equal(decoder.decode(family, probabilities[:1]), symbols[:1])
    retained_words = decoder.get_compressed().astype("<u4", copy=False)
    resumed = constriction.stream.stack.AnsCoder(retained_words)

    assert np.array_equal(decoder.decode(family, probabilities[1:]), symbols[1:])
    assert decoder.is_empty()
    assert np.array_equal(resumed.decode(family, probabilities[1:]), symbols[1:])
    assert resumed.is_empty()


def test_real_pins_and_legacy_rebuild_when_ssd_is_mounted() -> None:
    required = (
        ai1.BASE_ARCHIVE,
        ai1.TM1_ANS_ARCHIVE,
        ai1.ANS_PAYLOAD,
        ai1.DT1_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        import pytest

        pytest.skip("ai1 source custody is not mounted")

    ai1.require_pin(
        ai1.BASE_ARCHIVE,
        size=ai1.EXPECTED_BASE_BYTES,
        digest=ai1.EXPECTED_BASE_SHA256,
        label="base",
    )
    ai1.require_pin(
        ai1.TM1_ANS_ARCHIVE,
        size=ai1.EXPECTED_TM1_ARCHIVE_BYTES,
        digest=ai1.EXPECTED_TM1_ARCHIVE_SHA256,
        label="TM1",
    )
    ai1.validate_dt1()
    base_member = ai1.read_stored_member(ai1.BASE_ARCHIVE)
    tm1_member = ai1.read_stored_member(ai1.TM1_ANS_ARCHIVE)
    base_parts = ai1.receiver.split_payload(base_member)
    tm1_parts = ai1.receiver.split_payload(tm1_member)
    tagged = ai1.receiver.pack_payload(
        base_parts.models,
        ai1.ANS_PAYLOAD.read_bytes(),
        token_codec="ans",
        model_codec="legacy_lzma",
    )

    assert ai1.deterministic_zip(base_member) == ai1.BASE_ARCHIVE.read_bytes()
    assert tm1_parts.models == base_parts.models
    assert tm1_parts.tokens == ai1.ANS_PAYLOAD.read_bytes()
    assert tm1_parts.token_codec == "range"
    assert ai1.receiver.split_payload(tagged).token_codec == "ans"
    assert len(ai1.deterministic_zip(tagged)) == ai1.EXPECTED_TM1_ARCHIVE_BYTES
