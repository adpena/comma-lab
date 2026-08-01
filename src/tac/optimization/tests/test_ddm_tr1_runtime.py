# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.optimization import ddm_tr1_runtime as runtime


def _synthetic_config() -> dict[str, object]:
    # D=128 makes a tiny 3x4 token lattice while still exercising every
    # receiver upsample stage through the real 384x512 output geometry.
    return {
        "batch_pairs": 1,
        "class_weight_lane": 1.0,
        "code_width": 1,
        "ema_decay": 0.9,
        "ema_decay_provenance": "synthetic-test",
        "epochs": 1,
        "gate_every": 1,
        "grid_downsample": 128,
        "lotto_mask_density_init": 0.5,
        "lotto_seed": 19,
        "lr": 0.001,
        "margin_target": 1.0,
        "num_pairs": 2,
        "renderer_width": 2,
        "seed": 7,
        "seg_form_start": "ce",
        "token_quant_levels": 8,
        "token_ste": "round",
        "token_temporal_mode": "shared_base",
        "variant": "lotto",
        "w_seg": 100.0,
    }


def _synthetic_checkpoint(path: Path) -> dict[str, np.ndarray]:
    config = _synthetic_config()
    selector = runtime._selector_from_config(config)
    rng = np.random.default_rng(11)
    arrays: dict[str, np.ndarray] = {
        "tokens_base": rng.uniform(-0.4, 0.4, (3, 4, 1)).astype(np.float32),
        "tokens_delta": rng.uniform(-0.2, 0.2, (2, 3, 4, 1)).astype(np.float32),
    }
    for name, shape in runtime._conv_shapes(selector):
        score = rng.normal(0, 0.2, shape).astype(np.float32)
        # Make the first and last bits deterministic and ensure that padding
        # exists for the exact-consumption padding test geometry.
        score.reshape(-1)[0] = np.float32(-0.5)
        score.reshape(-1)[-1] = np.float32(0.5)
        arrays[f"s_{name}"] = score
        arrays[f"g_{name}"] = rng.uniform(
            0.8,
            1.2,
            (shape[0],),
        ).astype(np.float32)
        arrays[f"b_{name}"] = rng.uniform(
            -0.1,
            0.1,
            (shape[0],),
        ).astype(np.float32)
    config_bytes = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    meta = {
        "cfg": config,
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "stage": "synthetic",
    }
    payload: dict[str, np.ndarray] = {f"ema::{name}": value for name, value in arrays.items()}
    payload["meta::json"] = np.frombuffer(
        json.dumps(meta).encode(),
        dtype=np.uint8,
    )
    payload["meta::epoch"] = np.array([0], dtype=np.int64)
    np.savez(path, **payload)
    return arrays


def _compiled(tmp_path: Path) -> tuple[runtime.CompiledTR1Archive, Path]:
    checkpoint = tmp_path / "synthetic.npz"
    _synthetic_checkpoint(checkpoint)
    return runtime.compile_archive_from_checkpoint(checkpoint), checkpoint


def _directory_start(packet: bytes) -> int:
    _magic, _version, _count, metadata_length = runtime.PACKET_HEADER.unpack_from(packet)
    return runtime.PACKET_HEADER.size + metadata_length


def _replace_section_payload(
    parsed: runtime.ParsedTR1Packet,
    name: str,
    payload: bytes,
) -> bytes:
    rows = {
        section_name: (payload if section_name == name else parsed.section_payloads[index])
        for index, section_name in enumerate(runtime.SECTION_NAMES)
    }
    return runtime.build_packet(parsed.metadata, rows)


def test_combined_ema_tokens_and_effective_renderer_material_are_counted(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "synthetic.npz"
    source = _synthetic_checkpoint(checkpoint)
    compiled = runtime.compile_archive_from_checkpoint(checkpoint)
    parsed_archive = runtime.parse_archive(compiled.archive_bytes)
    parsed = parsed_archive.packet

    combined = np.clip(
        source["tokens_base"][None] + source["tokens_delta"],
        -1,
        1,
    )
    expected_codes = np.rint((combined + np.float32(1)) * np.float32(0.5) * np.float32(7)).astype(np.uint8)
    assert np.array_equal(parsed.token_codes, expected_codes)

    for index, (name, _shape) in enumerate(runtime._conv_shapes(parsed.selector)):
        assert np.array_equal(parsed.masks[index], source[f"s_{name}"] > 0)
        assert np.array_equal(
            parsed.gains[index],
            source[f"g_{name}"].astype(np.float16).astype(np.float32),
        )
        assert np.array_equal(
            parsed.biases[index],
            source[f"b_{name}"].astype(np.float16).astype(np.float32),
        )

    assert parsed.pose_stub_consumed is True
    assert parsed.metadata["pose_claim"] is False
    assert parsed_archive.manifest["pose_claim"] is False
    assert [row["name"] for row in runtime.section_ledger(parsed)] == list(runtime.SECTION_NAMES)


def test_packet_archive_are_deterministic_stored_and_exact_reemit(
    tmp_path: Path,
) -> None:
    first, checkpoint = _compiled(tmp_path)
    second = runtime.compile_archive_from_checkpoint(checkpoint)
    assert first.packet_bytes == second.packet_bytes
    assert first.archive_bytes == second.archive_bytes
    parsed = runtime.parse_archive(first.archive_bytes)
    assert runtime.reemit_packet(parsed.packet) == first.packet_bytes
    assert runtime.reemit_archive(parsed) == first.archive_bytes

    with zipfile.ZipFile(io.BytesIO(first.archive_bytes), mode="r") as archive:
        assert tuple(archive.namelist()) == runtime.ARCHIVE_MEMBERS
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())


def test_packet_refuses_corruption_swap_omission_and_trailing_bytes(
    tmp_path: Path,
) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    packet = compiled.packet_bytes

    corrupted = bytearray(packet)
    corrupted[-1] ^= 1
    with pytest.raises(runtime.TR1RuntimeError, match="SHA-256"):
        runtime.parse_packet(bytes(corrupted))

    directory_start = _directory_start(packet)
    entry_size = runtime.SECTION_ENTRY.size
    swapped = bytearray(packet)
    first = bytes(swapped[directory_start : directory_start + entry_size])
    second = bytes(swapped[directory_start + entry_size : directory_start + 2 * entry_size])
    swapped[directory_start : directory_start + entry_size] = second
    swapped[directory_start + entry_size : directory_start + 2 * entry_size] = first
    with pytest.raises(runtime.TR1RuntimeError, match="section order"):
        runtime.parse_packet(bytes(swapped))

    omitted = packet[:-1]
    with pytest.raises(runtime.TR1RuntimeError, match="escaped packet"):
        runtime.parse_packet(omitted)

    with pytest.raises(runtime.TR1RuntimeError, match="final cursor"):
        runtime.parse_packet(packet + b"\0")


def test_packet_refuses_offset_drift_and_inner_trailing_brotli(
    tmp_path: Path,
) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    packet = bytearray(compiled.packet_bytes)
    directory_start = _directory_start(packet)
    name, offset, length, digest = runtime.SECTION_ENTRY.unpack_from(
        packet,
        directory_start,
    )
    runtime.SECTION_ENTRY.pack_into(
        packet,
        directory_start,
        name,
        offset + 1,
        length,
        digest,
    )
    with pytest.raises(runtime.TR1RuntimeError, match="not contiguous"):
        runtime.parse_packet(bytes(packet))

    parsed = runtime.parse_packet(compiled.packet_bytes)
    tokens_with_trailing = parsed.section("tokens") + b"\0"
    rebuilt = _replace_section_payload(
        parsed,
        "tokens",
        tokens_with_trailing,
    )
    with pytest.raises(runtime.TR1RuntimeError, match="coded length"):
        runtime.parse_packet(rebuilt)


def test_selector_and_pose_stub_refuse_counted_but_inert_fields(
    tmp_path: Path,
) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    parsed = runtime.parse_packet(compiled.packet_bytes)

    selector = dict(parsed.selector)
    selector["training_only_mask_density"] = 0.5
    rebuilt = _replace_section_payload(
        parsed,
        "selector",
        runtime._canonical_json(selector),
    )
    with pytest.raises(runtime.TR1RuntimeError, match="selector keys differ"):
        runtime.parse_packet(rebuilt)

    pose = dict(runtime._POSE_STUB_VALUE)
    pose["d_pose"] = 0.0
    rebuilt = _replace_section_payload(
        parsed,
        "pose_stub",
        runtime._canonical_json(pose),
    )
    with pytest.raises(runtime.TR1RuntimeError, match="pose stub differs"):
        runtime.parse_packet(rebuilt)


def test_archive_refuses_packet_manifest_mismatch(tmp_path: Path) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    parsed = runtime.parse_archive(compiled.archive_bytes)
    manifest = dict(parsed.manifest)
    manifest["packet_sha256"] = "0" * 64
    tampered = runtime._deterministic_stored_zip(
        {
            "manifest.json": runtime._canonical_json(manifest),
            "state/tr1.ddt1": parsed.packet_bytes,
        }
    )
    with pytest.raises(runtime.TR1RuntimeError, match="packet SHA-256"):
        runtime.parse_archive(tampered)


def test_tiny_numpy_receiver_matches_mlx_cpu_forward(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_unflatten

    from experiments.train_tr1_partition_renderer_mlx import (
        TR1Config,
        build_module,
    )

    checkpoint = tmp_path / "synthetic.npz"
    _synthetic_checkpoint(checkpoint)
    compiled = runtime.compile_archive_from_checkpoint(checkpoint)
    parsed = runtime.parse_archive(compiled.archive_bytes).packet

    with np.load(checkpoint, allow_pickle=False) as stored:
        meta = json.loads(bytes(stored["meta::json"]).decode())
        model = build_module(TR1Config(**meta["cfg"]))
        ema = [(key[len("ema::") :], mx.array(stored[key])) for key in stored.files if key.startswith("ema::")]
    model.update(tree_unflatten(ema))
    mx.eval(model.parameters())
    with mx.stream(mx.cpu):
        token_reference = model.quantized_tokens(0)
        frame_reference = model.render_frame(0)
        mx.eval(token_reference, frame_reference)

    token_numpy = runtime.decode_token_grid(parsed, 0)
    frame_numpy = runtime.render_frame1_float(parsed, 0)
    assert np.array_equal(
        token_numpy,
        np.asarray(token_reference, dtype=np.float32),
    )
    deploy_delta = np.abs(frame_numpy - np.asarray(frame_reference, dtype=np.float32)[0])
    # The binding format deliberately deploys gamma+bias as fp16.  Compare to
    # the source EMA fp32 forward and bound the real quantization gap rather
    # than pretending the two parameterizations are identical.
    assert float(deploy_delta.max()) <= 0.01
    assert float(deploy_delta.mean()) <= 0.002


def test_legacy_token_codec_stays_absent_and_byte_identical(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.npz"
    _synthetic_checkpoint(checkpoint)
    legacy = runtime.compile_archive_from_checkpoint(checkpoint, token_codec=None)
    again = runtime.compile_archive_from_checkpoint(checkpoint, token_codec=None)
    assert legacy.archive_bytes == again.archive_bytes

    parsed = runtime.parse_archive(legacy.archive_bytes).packet
    # Absence is the ONE canonical spelling of the legacy framing: the selector
    # must not grow a key, or every pre-codec archive would stop parsing.
    assert "token_codec" not in dict(parsed.selector)
    assert parsed.section("tokens").startswith(runtime.TOKENS_FRAME_MAGIC)
    # The bare positional call (every pre-existing caller) and the
    # selector-driven legacy call must produce the same counted bytes.
    assert runtime._encode_tokens(parsed.token_codes) == parsed.section("tokens")
    assert runtime._encode_tokens(parsed.token_codes, parsed.selector) == parsed.section("tokens")
    # Re-derive the pre-codec framing independently, so this cannot pass merely
    # by agreeing with a drifted _encode_tokens.
    shape = tuple(int(value) for value in parsed.token_codes.shape)
    expected_raw = runtime.TOKENS_RAW_HEADER.pack(*shape) + parsed.token_codes.tobytes()
    assert parsed.section("tokens") == runtime._encode_brotli_frame(
        runtime.TOKENS_FRAME_MAGIC,
        expected_raw,
    )


def test_r7_smevr_token_section_round_trips_through_parse_and_reemit(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "synthetic.npz"
    _synthetic_checkpoint(checkpoint)
    legacy = runtime.compile_archive_from_checkpoint(checkpoint, token_codec=None)
    smevr = runtime.compile_archive_from_checkpoint(checkpoint)

    parsed = runtime.parse_archive(smevr.archive_bytes)
    assert dict(parsed.packet.selector)["token_codec"] == runtime.TOKEN_CODEC_R7_SMEVR
    assert parsed.packet.section("tokens").startswith(runtime.TOKENS_R7_FRAME_MAGIC)
    # Lossless: the coder changes bytes, never the counted token lattice.
    legacy_codes = runtime.parse_archive(legacy.archive_bytes).packet.token_codes
    assert np.array_equal(parsed.packet.token_codes, legacy_codes)
    assert parsed.packet.token_codes.dtype == np.uint8
    assert runtime.reemit_packet(parsed.packet) == smevr.packet_bytes
    assert runtime.reemit_archive(parsed) == smevr.archive_bytes
    assert runtime.compile_archive_from_checkpoint(checkpoint).archive_bytes == smevr.archive_bytes


def test_selector_refuses_unknown_or_noncanonical_token_codec(tmp_path: Path) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    parsed = runtime.parse_packet(compiled.packet_bytes)
    # An explicit legacy literal, a bare coder name, an empty string, and a
    # JSON null are all second spellings of an existing state: each refuses.
    for value in ("brotli11_raw_uint8_v1", "smevr", "", None):
        selector = dict(parsed.selector)
        selector["token_codec"] = value
        rebuilt = _replace_section_payload(
            parsed,
            "selector",
            runtime._canonical_json(selector),
        )
        with pytest.raises(runtime.TR1RuntimeError, match="token_codec"):
            runtime.parse_packet(rebuilt)


def test_r7_token_frame_cross_checks_shape_digest_and_stream(tmp_path: Path) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    parsed = runtime.parse_packet(compiled.packet_bytes)

    # A self-describing frame is not an excuse to skip the selector cross-check.
    selector = dict(parsed.selector)
    selector["num_pairs"] = int(selector["num_pairs"]) + 1
    rebuilt = _replace_section_payload(
        parsed,
        "selector",
        runtime._canonical_json(selector),
    )
    with pytest.raises(runtime.TR1RuntimeError, match="token shape differs"):
        runtime.parse_packet(rebuilt)

    # The outer frame keeps the independent raw-image digest the legacy framing
    # carries; corrupting only that field must still refuse.
    tokens = bytearray(parsed.section("tokens"))
    tokens[24] ^= 1
    rebuilt = _replace_section_payload(parsed, "tokens", bytes(tokens))
    with pytest.raises(runtime.TR1RuntimeError, match="raw SHA-256"):
        runtime.parse_packet(rebuilt)

    # Corruption inside the R7 stream is caught by the coder itself.
    tokens = bytearray(parsed.section("tokens"))
    tokens[-1] ^= 1
    rebuilt = _replace_section_payload(parsed, "tokens", bytes(tokens))
    with pytest.raises(runtime.TR1RuntimeError):
        runtime.parse_packet(rebuilt)


def test_r7_selector_refuses_a_legacy_framed_token_section(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.npz"
    _synthetic_checkpoint(checkpoint)
    legacy = runtime.compile_archive_from_checkpoint(checkpoint, token_codec=None)
    smevr = runtime.compile_archive_from_checkpoint(checkpoint)
    parsed = runtime.parse_packet(smevr.packet_bytes)
    legacy_tokens = runtime.parse_packet(legacy.packet_bytes).section("tokens")
    rebuilt = _replace_section_payload(parsed, "tokens", legacy_tokens)
    # A selector/payload codec disagreement fails LOUD, never silently.
    with pytest.raises(runtime.TR1RuntimeError, match="magic differs"):
        runtime.parse_packet(rebuilt)


def test_r7_token_codec_refuses_levels_above_the_coder_lattice() -> None:
    config = _synthetic_config()
    config["token_quant_levels"] = 32
    # The legacy uint8 framing still admits the wider lattice.
    assert runtime._selector_from_config(config)["token_quant_levels"] == 32
    with pytest.raises(runtime.TR1RuntimeError, match="at most 16 levels"):
        runtime._selector_from_config(
            config,
            token_codec=runtime.TOKEN_CODEC_R7_SMEVR,
        )


def test_pose_stub_pair_is_zero_frame0_without_pose_claim(
    tmp_path: Path,
) -> None:
    compiled, _checkpoint = _compiled(tmp_path)
    parsed = runtime.parse_archive(compiled.archive_bytes).packet
    pair = runtime.render_pair_camera_uint8(parsed, 0)
    assert pair.shape == (2, runtime.CAMERA_H, runtime.CAMERA_W, 3)
    assert pair.dtype == np.uint8
    assert np.count_nonzero(pair[0]) == 0
    assert parsed.metadata["pose_claim"] is False
