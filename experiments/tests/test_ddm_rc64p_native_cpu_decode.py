from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import constriction
import numpy as np
import torch
from tac.payload_retention_gate import check_no_measure_and_discard_payload


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "experiments" / "ddm_rc64p_native_cpu_decode.py"
IMPLEMENTATION = REPO / "experiments" / "ddm_rc64p_native_cpu_decode"


def load_harness():
    spec = importlib.util.spec_from_file_location("ddm_rc64p_harness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native_ans():
    path = IMPLEMENTATION / "native_ans.py"
    spec = importlib.util.spec_from_file_location("native_ans", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_route_b():
    path = IMPLEMENTATION / "route_b_rc64.py"
    spec = importlib.util.spec_from_file_location("ddm_rc64p_route_b_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_decoder_matches_constriction_and_midstream_state(tmp_path: Path):
    library = tmp_path / "liblc2_ans.dylib"
    subprocess.run(
        [
            "/usr/bin/cc",
            "-O3",
            "-std=c11",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-fast-math",
            str(IMPLEMENTATION / "ans_backend.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    native_ans = load_native_ans()
    rng = np.random.default_rng(6464)
    logits = rng.normal(size=(10_003, 5)).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    tables = np.exp(logits).astype(np.float32)
    tables /= tables.sum(axis=1, keepdims=True)
    symbols = rng.integers(0, 5, len(tables), dtype=np.int32)
    family = constriction.stream.model.Categorical(perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(symbols, family, tables)
    words = encoder.get_compressed().copy()
    oracle = constriction.stream.stack.AnsCoder(words.copy())
    native = native_ans.NativeAnsDecoder(
        library, words.astype("<u4", copy=False).tobytes(order="C")
    )
    split = 3_337
    assert np.array_equal(
        oracle.decode(family, tables[:split]),
        native.decode(family, tables[:split]),
    )
    assert np.array_equal(oracle.get_compressed(), native.get_compressed())
    expected = oracle.decode(family, tables[split:])
    actual = native.decode(family, tables[split:])
    assert np.array_equal(expected, actual)
    assert np.array_equal(np.concatenate([symbols[:split], actual]), symbols)
    assert oracle.is_empty()
    assert native.is_empty()


def test_receiver_patch_has_required_and_typed_fallback_modes():
    harness = load_harness()
    source = harness.patched_receiver(
        (harness.SOURCE_DIR / "receiver.py").read_text()
    )
    assert 'native_mode == "required"' in source
    assert "LC2_NATIVE_ANS_FALLBACK" in source
    assert "NativeAnsDecoder(Path(library_text), blob)" in source
    assert source.count("return constriction.stream.stack.AnsCoder(words)") == 1
    assert 'blob[:4] in (b"R6D1", b"R6C1")' in source
    assert "NativeRc64Decoder(Path(library_text), blob)" in source


def test_rc64_checkpointed_encode_decode_is_exact(tmp_path: Path):
    harness = load_harness()
    route_b = load_route_b()
    source = tmp_path / "rc64_backend.c"
    source.write_text(
        harness.GRANTED_RC64_SOURCE.read_text()
        + "\n"
        + route_b.RC64_CHECKPOINT_EXTENSION
    )
    library = tmp_path / "liblc2_rc64.dylib"
    subprocess.run(
        [
            "/usr/bin/cc",
            "-O3",
            "-std=c11",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            "-fno-fast-math",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    rng = np.random.default_rng(6465)
    logits = rng.normal(size=(10_003, 5)).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    tables = np.exp(logits).astype(np.float32)
    tables /= tables.sum(axis=1, keepdims=True)
    symbols = rng.integers(0, 5, len(tables), dtype=np.int32)
    split = 3_337

    encoder = route_b.NativeRc64Encoder(library)
    encoder.encode(symbols[:split], tables[:split])
    encoder_checkpoint = encoder.snapshot()
    encoder.close()
    resumed_encoder = route_b.NativeRc64Encoder(
        library, checkpoint=encoder_checkpoint
    )
    resumed_encoder.encode(symbols[split:], tables[split:])
    payload = resumed_encoder.finish()
    resumed_encoder.close()
    assert payload.startswith(b"R6D1")
    assert len(payload) % 4 == 0

    decoder = route_b.NativeRc64Decoder(library, payload)
    first = decoder.decode(None, tables[:split])
    decoder_checkpoint = decoder.get_compressed().tobytes(order="C")
    decoder.close()
    resumed_decoder = route_b.NativeRc64Decoder(library, decoder_checkpoint)
    rest = resumed_decoder.decode(None, tables[split:])
    resumed_decoder.close()
    assert np.array_equal(np.concatenate([first, rest]), symbols)


def test_cached_sparse_hpac_matches_settled_logits():
    runtime = load_harness().SOURCE_DIR
    sys.path.insert(0, str(runtime))
    try:
        for name in ("hpac_integer", "hpac_integer_sparse"):
            sys.modules.pop(name, None)
        from hpac_integer import IntegerHPAC
        from hpac_integer_sparse import SparseIntegerHPAC as SettledSparse

        path = IMPLEMENTATION / "hpac_integer_sparse_optimized.py"
        spec = importlib.util.spec_from_file_location(
            "ddm_rc64p_cached_sparse_test", path
        )
        assert spec and spec.loader
        optimized_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = optimized_module
        spec.loader.exec_module(optimized_module)

        torch.manual_seed(6466)
        model = IntegerHPAC(
            num_pairs=2,
            num_classes=5,
            patch=8,
            delta=2,
            channels=4,
            frame_dim=2,
            norm_mode="none",
            activation="relu",
            use_frame_scale=True,
            weight_bound=127,
            activation_bound=127,
            use_weight_scales=True,
            weight_exponent_min=-6,
            use_spm=False,
            use_norm_gates=False,
        ).eval()
        current = torch.zeros((1, 8, 8), dtype=torch.long)
        previous = torch.zeros_like(current)
        context = model.prepare_frame_context(torch.tensor([0]), previous)
        settled = SettledSparse(model, 8, 8)
        optimized = optimized_module.SparseIntegerHPAC(model, 8, 8)
        rows = torch.arange(8).view(8, 1)
        cols = torch.arange(8).view(1, 8)
        for group in range(22):
            expected = settled.selected_logits(current, context, group)
            actual = optimized.selected_logits(current, context, group)
            assert torch.equal(expected, actual)
            current[0, cols + 2 * rows == group] = group % 5
    finally:
        sys.path.remove(str(runtime))


def test_shipping_wrapper_compiles_before_literal_receiver_entrypoint():
    harness = load_harness()
    wrapper = harness.native_inflate_wrapper()
    assert "-ffp-contract=off -fno-fast-math" in wrapper
    assert "-Wl,-install_name,@rpath/liblc2_ans.dylib" in wrapper
    assert 'LC2_NATIVE_ANS_MODE:-auto' in wrapper
    assert 'LC2_NATIVE_BUILD_DIR:-"$DEPS_DIR.lc2-native-ans"' in wrapper
    assert "LC2_NATIVE_COMPILE_SMOKE_ONLY" in wrapper
    assert "exit 72" in wrapper
    assert 'exec "$SCRIPT_DIR/inflate_lc2.sh" "$@"' in wrapper


def test_payload_manifest_keeps_lc2_archive_unchanged():
    import json

    manifest = json.loads(
        (IMPLEMENTATION / "archive_payload_manifest.json").read_text()
    )
    assert manifest["archive_bytes_changed_by_route_a"] is False
    assert manifest["counted_payload"]["archive_bytes"] == 187_226
    assert manifest["video_derived_constants_embedded_in_native_source"] == []


def test_rc64p_python_files_pass_payload_retention_gate():
    findings = check_no_measure_and_discard_payload(
        repo_root=REPO,
        strict=False,
        roots=(
            "experiments/ddm_rc64p_native_cpu_decode.py",
            "experiments/ddm_rc64p_native_cpu_decode/native_ans.py",
            "experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py",
            "experiments/ddm_rc64p_native_cpu_decode/hpac_integer_sparse_optimized.py",
            "experiments/ddm_rc64p_native_cpu_decode/python_reference_equivalence_test.py",
        ),
    )
    assert findings == []
