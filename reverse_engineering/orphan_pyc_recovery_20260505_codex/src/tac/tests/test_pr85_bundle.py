# pyc-recovery pass2: rehydrated from git blob 406838df182432eaaf2cdbc0246c34962bb418fe via `git fsck --lost-found`
# original path: src/tac/tests/test_pr85_bundle.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pytest

from tac.pr85_bundle import (
    FIXED_V5_LENGTHS,
    PR85_HEADERLESS_RANDMULTI_SPECS,
    Pr85BundleError,
    SEGMENT_ORDER,
    build_pr85_qpost_bin,
    compare_pr85_randmulti_decoded_rows,
    decode_pr85_p1d1_pose_to_fp16,
    decode_pr85_randmulti_to_headerless_rows,
    decode_rmb1_randmulti_payload,
    expand_pr85_bundle_to_runtime_members,
    pack_pr85_bundle,
    parse_hpm1_mask_segment,
    parse_pr85_bundle,
    transcode_pr85_randmulti_to_qrm1,
    validate_pr85_member_name,
)


REPO = Path(__file__).resolve().parents[3]


def _segments() -> dict[str, bytes]:
    return {
        "mask": b"QMA9" + b"m" * 1200,
        "model": b"QH0" + b"r" * 1200,
        "pose": b"P1D1" + b"p" * 120,
        "post": b"post",
        "shift": b"shift",
        "frac": b"frac",
        "frac2": b"frac2",
        "frac3": b"frac3",
        "bias": b"B" * FIXED_V5_LENGTHS["bias"],
        "region": b"R" * FIXED_V5_LENGTHS["region"],
        "randmulti": b"randmulti",
    }


def _vlq(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _p1d1_pose_raw(*, dims: tuple[int, ...] = (0, 2)) -> bytes:
    streams = []
    for dim in dims:
        values = [0] * 600
        previous = 0
        stream = bytearray()
        for value in values:
            stream.extend(_vlq(_zigzag(value - previous)))
            previous = value
        streams.append(bytes(stream))
    header = b"P1D1" + bytes([len(dims)])
    for dim, stream in zip(dims, streams):
        header += bytes([dim]) + len(stream).to_bytes(2, "little")
    return header + b"".join(streams)


def _randmulti_zero_payload() -> bytes:
    return b"\x00" * sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS)


def _rmb1_from_headerless_rows(raw: bytes) -> bytes:
    brotli = pytest.importorskip("brotli")
    cursor = 0
    mask = bytearray()
    values = bytearray()
    while cursor < len(raw):
        count = raw[cursor]
        cursor += 1
        if count == 255:
            count = int.from_bytes(raw[cursor : cursor + 2], "little")
            cursor += 2
        row_mask = bytearray(75)
        idx = -1
        for _ in range(count):
            delta = 0
            shift = 0
            while True:
                byte = raw[cursor]
                cursor += 1
                delta |= (byte & 0x7F) << shift
                if byte < 128:
                    break
                shift += 7
            idx += delta + 1
            row_mask[idx // 8] |= 1 << (idx % 8)
        values.extend(raw[cursor : cursor + count])
        cursor += count
        mask.extend(row_mask)
    mask_br = brotli.compress(bytes(mask), quality=5)
    vals_br = brotli.compress(bytes(values), quality=5)
    return b"RMB1" + len(mask_br).to_bytes(2, "little") + mask_br + vals_br


def _archive_member(path: Path, member_name: str = "x") -> bytes:
    if not path.is_file():
        pytest.skip(f"local archive artifact is missing: {path}")
    with zipfile.ZipFile(path, "r") as zf:
        return zf.read(member_name)


def _hpm1_mask_segment() -> bytes:
    tokens = b"TOKN" * 3
    hpac = b"HPACMODEL"
    header = b"HPM1" + struct.pack(
        "<" + "I" * 11,
        600,
        384,
        512,
        32,
        2,
        64,
        1,
        8,
        len(tokens),
        len(hpac),
        4,
    )
    return header + tokens + hpac


def test_v5_pack_parse_roundtrip() -> None:
    segments = _segments()
    raw = pack_pr85_bundle(segments, header_mode="v5")
    parsed = parse_pr85_bundle(raw)

    assert parsed.format == "pr85_v5_micro_24bit_lengths_fixed_bias_region"
    assert parsed.header_bytes == 24
    assert parsed.fixed_length_segments == FIXED_V5_LENGTHS
    assert parsed.segment_lengths == {name: len(segments[name]) for name in SEGMENT_ORDER}
    assert dict(parsed.segments) == segments
    assert pack_pr85_bundle(parsed.segments, header_mode="v5") == raw


def test_explicit_30_pack_parse_roundtrip_with_changed_bias_region() -> None:
    segments = _segments()
    segments["bias"] = b"bias-short"
    segments["region"] = b"region-short"
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")
    parsed = parse_pr85_bundle(raw)

    assert parsed.format == "pr85_explicit_30byte_lengths"
    assert parsed.header_bytes == 30
    assert parsed.fixed_length_segments == {}
    assert dict(parsed.segments) == segments


def test_v5_rejects_changed_fixed_length_segments() -> None:
    segments = _segments()
    segments["bias"] = b"too-short"

    with pytest.raises(Pr85BundleError, match="fixed-length segment 'bias'"):
        pack_pr85_bundle(segments, header_mode="v5")


def test_rejects_unsafe_member_names() -> None:
    assert validate_pr85_member_name("x") == "x"
    for name in ("p", "../x", "/x", "dir/x"):
        with pytest.raises(Pr85BundleError):
            validate_pr85_member_name(name)


def test_hpm1_mask_segment_contract_is_typed_and_fail_closed() -> None:
    contract = parse_hpm1_mask_segment(_hpm1_mask_segment())

    assert contract.name == "mask"
    assert contract.codec == "HPM1"
    assert contract.metadata["runtime_contract"] == "HPM1_pr91_hpac_mask_stream"
    assert contract.metadata["N"] == 600
    assert contract.metadata["H"] == 384
    assert contract.metadata["W"] == 512
    assert contract.metadata["tokens_len"] == 12
    assert contract.metadata["tokens_uint32_aligned"] is True
    assert contract.metadata["hpac_len"] == 9

    with pytest.raises(Pr85BundleError, match="trailing bytes"):
        parse_hpm1_mask_segment(_hpm1_mask_segment() + b"x")


def test_runtime_expansion_materializes_qpost_qrm1_and_pose_fp16() -> None:
    brotli = pytest.importorskip("brotli")
    segments = {
        "mask": b"QMA9" + b"\x00" * 32,
        "model": brotli.compress(b"QH0" + b"renderer" * 32, quality=5),
        "pose": brotli.compress(_p1d1_pose_raw(), quality=5),
        "post": brotli.compress(bytes([0]) * 2400, quality=5),
        "shift": brotli.compress(b"SD4" + bytes([0]) * 600, quality=5),
        "frac": brotli.compress(b"FV1" + bytes([0]) * 8, quality=5),
        "frac2": brotli.compress(b"FH2" + bytes([4]) * 600, quality=5),
        "frac3": brotli.compress(b"FD3" + bytes([0]) * 600, quality=5),
        "bias": brotli.compress(b"BD1" + bytes([0]) * 600, quality=5),
        "region": brotli.compress(b"RH1" + bytes([0]) * 600, quality=5),
        "randmulti": brotli.compress(_randmulti_zero_payload(), quality=5),
    }
    raw = pack_pr85_bundle(segments, header_mode="explicit_30")

    expansion = expand_pr85_bundle_to_runtime_members(raw)
    members = expansion.members

    assert set(members) == {"masks.qma9", "renderer.bin", "optimized_poses.bin", "qpost.bin"}
    assert members["renderer.bin"].startswith(b"QH0")
    assert len(members["optimized_poses.bin"]) == 600 * 6 * 2
    assert members["qpost.bin"].startswith(b"QPS1")
    lengths = struct.unpack_from("<" + "I" * 8, members["qpost.bin"], 4)
    randmulti = members["qpost.bin"][4 + 8 * 4 + sum(lengths[:-1]) :]
    assert len(randmulti) == lengths[-1]
    assert brotli.decompress(randmulti).startswith(b"QRM1")
    assert expansion.manifest["qpost"]["randmulti"]["runtime_contract"] == "QRM1_sparse_group_id_stream"


def test_p1d1_pose_decode_and_randmulti_qrm1_transcode_are_closed() -> None:
    brotli = pytest.importorskip("brotli")
    pose_bytes, pose_meta = decode_pr85_p1d1_pose_to_fp16(
        brotli.compress(_p1d1_pose_raw(dims=(0,)), quality=5)
    )
    assert len(pose_bytes) == 600 * 6 * 2
    assert set(pose_meta["active_dimensions"]) == {0}
    assert pose_meta["raw_fp16_sha256"]

    qrm1, qrm1_meta = transcode_pr85_randmulti_to_qrm1(
        brotli.compress(_randmulti_zero_payload(), quality=5)
    )
    assert brotli.decompress(qrm1).startswith(b"QRM1")
    assert qrm1_meta["group_count"] == 72
    assert qrm1_meta["sparse_row_count"] == sum(
        spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS
    )


def test_rmb1_randmulti_decodes_to_pr85_headerless_rows() -> None:
    brotli = pytest.importorskip("brotli")
    raw = b"\x02\x00\x02\x05\x07" + b"\x00" * (
        sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS) - 1
    )
    encoded = _rmb1_from_headerless_rows(raw)

    decoded = decode_rmb1_randmulti_payload(encoded)
    normalized, meta = decode_pr85_randmulti_to_headerless_rows(encoded)
    qrm1, qrm1_meta = transcode_pr85_randmulti_to_qrm1(encoded)

    assert decoded == raw
    assert normalized == raw
    assert meta["codec"] == "RMB1_bitmask_value_randmulti"
    assert meta["group_count"] == 72
    assert meta["decoded_group_count"] == 72
    assert qrm1_meta["source_codec"] == "RMB1_bitmask_value_randmulti"
    assert brotli.decompress(qrm1).startswith(b"QRM1")


def test_current_pr92_rmb1_randmulti_is_decoded_row_parity_recode() -> None:
    pr85_archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    stbm_archive = (
        REPO
        / "experiments/results/pr85_stbm1br_mask_recode_20260504_worker/"
        "pr90_stbm1br_lossless_pr85_mask_recode/archive.zip"
    )
    pr92_archive = REPO / "experiments/results/public_pr92_intake_20260504_codex/archive.zip"

    pr85 = parse_pr85_bundle(_archive_member(pr85_archive))
    stbm = parse_pr85_bundle(_archive_member(stbm_archive))
    pr92 = parse_pr85_bundle(_archive_member(pr92_archive))

    assert bytes(stbm.segments["randmulti"]) == bytes(pr85.segments["randmulti"])
    assert bytes(pr92.segments["randmulti"]).startswith(b"RMB1")

    stbm_parity = compare_pr85_randmulti_decoded_rows(
        bytes(pr85.segments["randmulti"]),
        bytes(stbm.segments["randmulti"]),
    )
    pr92_parity = compare_pr85_randmulti_decoded_rows(
        bytes(pr85.segments["randmulti"]),
        bytes(pr92.segments["randmulti"]),
    )

    expected_decoded_sha = "87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9"
    assert stbm_parity["parity_status"] == "passed"
    assert pr92_parity["parity_status"] == "passed"
    assert pr92_parity["candidate"]["codec"] == "RMB1_bitmask_value_randmulti"
    assert pr92_parity["decoded_rows_sha256"] == expected_decoded_sha
    assert stbm_parity["decoded_rows_sha256"] == expected_decoded_sha


def test_qpost_builder_rejects_missing_sidechannel() -> None:
    segments = {name: b"x" for name in SEGMENT_ORDER}
    segments.pop("region")
    with pytest.raises(Exception, match="missing PR85 qpost stream"):
        build_pr85_qpost_bin(segments)


def test_real_pr85_archive_parse_if_available() -> None:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")

    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        assert [info.filename for info in infos] == ["x"]
        raw = zf.read("x")

    parsed = parse_pr85_bundle(raw)
    assert parsed.format == "pr85_v5_micro_24bit_lengths_fixed_bias_region"
    assert parsed.segment_lengths["mask"] == 159011
    assert parsed.segment_lengths["model"] == 57074
    assert parsed.segment_lengths["pose"] == 1487
    assert parsed.segment_lengths["randmulti"] == 16101
    assert pack_pr85_bundle(parsed.segments, header_mode="v5") == raw


def test_real_pr91_hpm1_archive_parse_if_available() -> None:
    archive = REPO / "experiments/results/public_pr91_intake_20260504_worker/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR91 intake archive is not present")

    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        assert [info.filename for info in infos] == ["x"]
        raw = zf.read("x")

    parsed = parse_pr85_bundle(raw)
    contracts = parsed.segment_contracts
    mask = contracts["mask"]

    assert parsed.format == "pr85_v5_micro_24bit_lengths_fixed_bias_region"
    assert parsed.segment_lengths["mask"] == 145087
    assert mask.codec == "HPM1"
    assert mask.metadata["header_bytes"] == 48
    assert mask.metadata["N"] == 600
    assert mask.metadata["H"] == 384
    assert mask.metadata["W"] == 512
    assert mask.metadata["tokens_len"] == 116796
    assert mask.metadata["tokens_sha256"] == "541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b"
    assert mask.metadata["hpac_ppmd_sha256"] == "de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd"
    assert pack_pr85_bundle(parsed.segments, header_mode="v5") == raw
