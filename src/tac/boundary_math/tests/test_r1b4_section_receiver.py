from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import tac.boundary_math.integer_plane_banded_trainer as trainer
from tac.boundary_math.integer_plane_banded_trainer import TrainerConfig, sha256_file
from tac.boundary_math.integer_plane_emitter_byte_close import build_counted_archive
from tac.boundary_math.power_diagram_witness import encode_pdw2, make_gauge_fixed_affine_target
from tac.boundary_math.r1b4_section_receiver import (
    MANIFEST_NAME,
    R1B4ReceiverError,
    ReplayWrite,
    build_r1b4_archive,
    canonical_json,
    decode_r1b4_archive,
    decode_replay_payload,
    default_receiver_policy,
    encode_replay_payload,
    parse_r1b4_archive,
    seal_output_assertion,
)
from tac.boundary_math.windowed_curvelet_frame import WindowedCurveletConfig
from tac.optimization.boundary_coordinate_joint_solve import (
    BoundaryCoordinatePacket,
    FrameFamily,
    encode_boundary_packet,
)
from tac.optimization.r1b3_producer_preflight import encode_xi0_payload
from tac.witness_dsl.integer_plane_emitter_policy import IntegerPlaneEmitterPolicy, PolicyMode

SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_base_archive(path: Path) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, b"counted-base-packet")


def _write_decoder(path: Path) -> None:
    path.write_text(
        "import os,sys\n"
        "n=int(os.environ['INFLATE_MAX_PAIRS'])\n"
        "fb=874*1164*3\n"
        "with open(sys.argv[2], 'wb') as f:\n"
        "  [f.write(bytes([96])*fb + bytes([160])*fb) for _ in range(n)]\n",
        encoding="ascii",
    )


def _pdw2_packet() -> bytes:
    target = make_gauge_fixed_affine_target(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([0.0, 0.25], dtype=np.float32),
    )
    return encode_pdw2(target)


def _c2_archive(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "one_packet.zip"
    decoder = tmp_path / "inflate.py"
    _write_base_archive(base)
    _write_decoder(decoder)
    config = TrainerConfig(
        policy=IntegerPlaneEmitterPolicy(mode=PolicyMode.BANDED_TRAINING),
        base_archive_sha256=sha256_file(base),
        base_decoder_sha256=sha256_file(decoder),
        source_sha256=SHA_A,
        band_sha256=SHA_B,
        band_mode="positive_anisotropic",
        output_dir=tmp_path,
        run_id="r1b4-test",
    )
    state = trainer._state_from_fresh(config)
    checkpoint = trainer._checkpoint(config, state, stage=config.stages[0], stage_complete=False)
    checkpoint_path = checkpoint.write_new(tmp_path, "r1b4_test")
    archive = tmp_path / "c2.zip"
    build_counted_archive(
        base_archive=base,
        checkpoint_path=checkpoint_path,
        output=archive,
        pdw2_packet=_pdw2_packet(),
    )
    return archive, decoder


def _boundary_payload(*, coefficient: int = 1) -> bytes:
    packet = BoundaryCoordinatePacket(
        family=FrameFamily.WINDOWED_CURVELET,
        frame_config=asdict(WindowedCurveletConfig(n_scales=1, n_orient0=2, n_trans=1)),
        scorer_height=384,
        scorer_width=512,
        atom_indices=np.asarray([0], dtype=np.uint32),
        coefficients=np.full((600, 1, 3), coefficient, dtype=np.int8),
        scales=np.ones(600, dtype=np.float16),
    )
    return encode_boundary_packet(packet)


def _xi0_payload(*, delta: float = 0.0) -> bytes:
    values = np.linspace(29.0, 33.0, 600, dtype=np.float32) + np.float32(delta)
    return encode_xi0_payload(values)


def _archive(
    tmp_path: Path,
    *,
    name: str = "r1b4.zip",
    coefficient: int = 1,
    xi_delta: float = 0.0,
    writes: tuple[ReplayWrite, ...] = (),
    policy: dict | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    base, decoder = _c2_archive(tmp_path)
    archive = tmp_path / name
    build_r1b4_archive(
        base_archive=base,
        boundary_payload=_boundary_payload(coefficient=coefficient),
        replay_payload=encode_replay_payload(writes),
        xi0_payload=_xi0_payload(delta=xi_delta),
        source_manifest_hashes={"fixture": hashlib.sha256(b"fixture").hexdigest()},
        output=archive,
        receiver_policy=policy,
        pair_cap=2,
    )
    return archive, decoder


def _rewrite(path: Path, output: Path, mutator) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    members = mutator(members)
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def test_replay_codec_is_canonical_exact_and_corruption_refuses() -> None:
    rows = (
        ReplayWrite(0, 0, 3, 4, 1, 17),
        ReplayWrite(1, 1, 5, 6, 2, 23),
    )
    payload = encode_replay_payload(rows)
    assert decode_replay_payload(payload) == rows
    with pytest.raises(R1B4ReceiverError, match="trailing"):
        decode_replay_payload(payload + b"x")
    corrupted = bytearray(payload)
    corrupted[-5] ^= 1
    with pytest.raises(R1B4ReceiverError, match=r"SHA|CRC|header"):
        decode_replay_payload(bytes(corrupted))
    with pytest.raises(R1B4ReceiverError, match="sorted"):
        encode_replay_payload(tuple(reversed(rows)))


def test_strict_archive_refuses_trailing_reordered_unknown_and_hash_drift(tmp_path: Path) -> None:
    archive, _decoder = _archive(tmp_path)
    parsed = parse_r1b4_archive(archive)
    assert parsed.manifest["receiver_search"] is False
    assert parsed.manifest["receiver_policy"]["receiver_search_invocations"] == 0

    trailing = tmp_path / "trailing.zip"
    shutil.copyfile(archive, trailing)
    with trailing.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(R1B4ReceiverError, match="trailing"):
        parse_r1b4_archive(trailing)

    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(archive.read_bytes()[:-8])
    with pytest.raises(R1B4ReceiverError, match=r"terminal ZIP EOCD|ZIP parse"):
        parse_r1b4_archive(truncated)

    duplicate = tmp_path / "duplicate.zip"
    shutil.copyfile(archive, duplicate)
    with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(duplicate, "a") as handle:
        handle.writestr("xi0.xi0", b"duplicate")
    with pytest.raises(R1B4ReceiverError, match=r"unsafe|duplicate|order"):
        parse_r1b4_archive(duplicate)

    reordered = tmp_path / "reordered.zip"
    _rewrite(archive, reordered, lambda rows: [*rows[:-2], rows[-1], rows[-2]])
    with pytest.raises(R1B4ReceiverError, match="order"):
        parse_r1b4_archive(reordered)

    unknown = tmp_path / "unknown.zip"
    _rewrite(archive, unknown, lambda rows: [*rows, ("unknown.bin", b"x")])
    with pytest.raises(R1B4ReceiverError, match="order"):
        parse_r1b4_archive(unknown)

    drift = tmp_path / "drift.zip"

    def mutate_manifest(rows):
        result = list(rows)
        index = next(i for i, (name, _payload) in enumerate(result) if name == MANIFEST_NAME)
        value = json.loads(result[index][1])
        value["sections"]["xi0.xi0"]["sha256"] = "0" * 64
        result[index] = (MANIFEST_NAME, canonical_json(value))
        return result

    _rewrite(archive, drift, mutate_manifest)
    with pytest.raises(R1B4ReceiverError, match="custody mismatch"):
        parse_r1b4_archive(drift)


def test_receiver_is_deterministic_section_consuming_and_frame_isolated(tmp_path: Path) -> None:
    writes = (ReplayWrite(0, 1, 0, 0, 0, 17),)
    archive, decoder = _archive(tmp_path, writes=writes)
    discovery = tmp_path / "discovery.raw"
    decode_r1b4_archive(
        archive=archive,
        base_decoder=decoder,
        scratch_root=tmp_path / "scratch-discovery",
        output_raw=discovery,
        receipt_path=tmp_path / "discovery.json",
        allow_unsealed_discovery=True,
    )
    sealed = tmp_path / "sealed.zip"
    seal_output_assertion(archive, decoded_path=discovery, output=sealed)
    first = tmp_path / "first.raw"
    second = tmp_path / "second.raw"
    receipt1 = decode_r1b4_archive(
        archive=sealed,
        base_decoder=decoder,
        scratch_root=tmp_path / "scratch-first",
        output_raw=first,
        receipt_path=tmp_path / "first.json",
    )
    receipt2 = decode_r1b4_archive(
        archive=sealed,
        base_decoder=decoder,
        scratch_root=tmp_path / "scratch-second",
        output_raw=second,
        receipt_path=tmp_path / "second.json",
    )
    assert first.read_bytes() == second.read_bytes()
    assert receipt1["decoded"]["sha256"] == receipt2["decoded"]["sha256"]
    assert receipt1["receiver_search_invocations"] == 0
    assert receipt1["section_consumption"]["boundary_coordinate.bgj"]["changed_pairs"] == 2
    assert receipt1["section_consumption"]["xi0.xi0"]["changed_pairs"] == 2
    assert receipt1["section_consumption"]["full_kernel_replay.r1k"]["effective_entries"] == 1

    shape = (2, 2, 874, 1164, 3)
    output = np.memmap(first, mode="r", dtype=np.uint8, shape=shape)
    zero_replay_archive, zero_replay_decoder = _archive(tmp_path / "zero", name="zero.zip")
    zero_output = tmp_path / "zero.raw"
    decode_r1b4_archive(
        archive=zero_replay_archive,
        base_decoder=zero_replay_decoder,
        scratch_root=tmp_path / "zero-scratch",
        output_raw=zero_output,
        receipt_path=tmp_path / "zero.json",
        allow_unsealed_discovery=True,
    )
    zero = np.memmap(zero_output, mode="r", dtype=np.uint8, shape=shape)
    assert np.array_equal(output[:, 0], zero[:, 0])
    assert not np.array_equal(output[:, 1], zero[:, 1])
    del output, zero


def test_valid_mutations_of_manifest_boundary_xi0_and_replay_change_output(tmp_path: Path) -> None:
    baseline, decoder = _archive(tmp_path / "baseline")

    policy = default_receiver_policy()
    policy["xi0_actuator"]["pixels_per_unit"] = -1.0
    manifest_mutation, manifest_decoder = _archive(tmp_path / "manifest", policy=policy)
    boundary_mutation, boundary_decoder = _archive(tmp_path / "boundary", coefficient=2)
    xi_mutation, xi_decoder = _archive(tmp_path / "xi", xi_delta=1.0)
    replay_mutation, replay_decoder = _archive(
        tmp_path / "replay",
        writes=(ReplayWrite(0, 1, 0, 0, 0, 17),),
    )
    variants = (
        (baseline, decoder, "baseline"),
        (manifest_mutation, manifest_decoder, "manifest"),
        (boundary_mutation, boundary_decoder, "boundary"),
        (xi_mutation, xi_decoder, "xi"),
        (replay_mutation, replay_decoder, "replay"),
    )
    hashes = {}
    outputs = {}
    for candidate, candidate_decoder, label in variants:
        raw = tmp_path / f"{label}.raw"
        row = decode_r1b4_archive(
            archive=candidate,
            base_decoder=candidate_decoder,
            scratch_root=tmp_path / f"scratch-{label}",
            output_raw=raw,
            receipt_path=tmp_path / f"{label}.json",
            allow_unsealed_discovery=True,
        )
        hashes[label] = row["decoded"]["sha256"]
        outputs[label] = raw
    assert all(hashes[label] != hashes["baseline"] for label in ("manifest", "boundary", "xi", "replay"))
    shape = (2, 2, 874, 1164, 3)
    frames = {
        label: np.memmap(path, mode="r", dtype=np.uint8, shape=shape) for label, path in outputs.items()
    }
    for label in ("manifest", "xi"):
        assert not np.array_equal(frames[label][:, 0], frames["baseline"][:, 0])
        assert np.array_equal(frames[label][:, 1], frames["baseline"][:, 1])
    assert np.array_equal(frames["boundary"][:, 0], frames["baseline"][:, 0])
    assert not np.array_equal(frames["boundary"][:, 1], frames["baseline"][:, 1])
    assert np.array_equal(frames["replay"][:, 0], frames["baseline"][:, 0])
    assert not np.array_equal(frames["replay"][:, 1], frames["baseline"][:, 1])
