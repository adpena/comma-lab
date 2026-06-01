# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path

from tac.substrates.hprc.archive import HprcSectionKind, parse_hprc_packet
from tac.substrates.hprc.representation_spine import (
    HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA,
    PACT_NERV_LEN_PREFIXED_HEADER_FMT,
    HprcRepresentationFamily,
    build_generic_neural_spine_packet,
    build_packed_hnerv_spine_from_archive,
    build_pact_nerv_len_prefixed_spine_from_archive,
    build_pact_nerv_len_prefixed_spine_from_archive_payload,
    build_pact_nerv_vq_spine_from_archive,
    build_pr95_hnerv_spine_from_archive,
    write_representation_spine_projection,
)

REPO = Path(__file__).resolve().parents[5]
PVQ_MAGIC = b"PVQ\x00"
PVQ_HEADER_FMT = "<4sBHHHIIII"


def _single_member_zip(path: Path, payload: bytes, *, name: str = "0.bin") -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(name, payload)
    return path


def _pr95_payload(*, pairs: int = 600) -> bytes:
    chunks = []
    for payload in (f'{{"pairs":{pairs}}}'.encode(), b"decoder-brotli", b"latents-brotli"):
        chunks.append(struct.pack("<I", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def _pvq_payload() -> bytes:
    latent_dim = 3
    num_pairs = 4
    codebook_size = 2
    decoder = b"pvq-decoder"
    codebook = bytes(range(codebook_size * latent_dim * 2))
    indices = b"\x00\x00\x01\x00\x00\x00\x01\x00"
    meta = b'{"output_height":384,"output_width":512}'
    header = struct.pack(
        PVQ_HEADER_FMT,
        PVQ_MAGIC,
        1,
        latent_dim,
        num_pairs,
        codebook_size,
        len(decoder),
        len(codebook),
        len(indices),
        len(meta),
    )
    return header + decoder + codebook + indices + meta


def _pact_len_prefixed_payload(*, magic: bytes = b"PSV2") -> bytes:
    decoder = b"pact-decoder"
    latents = b"\x00\x01" * 6
    side = b"\x03\x04\x05"
    meta = b'{"output_height":24,"output_width":32}'
    header = struct.pack(
        PACT_NERV_LEN_PREFIXED_HEADER_FMT,
        magic,
        1,
        2,
        3,
        16,
        len(decoder),
        len(latents),
        len(side),
        len(meta),
    )
    return header + decoder + latents + side + meta


def test_pr95_hnerv_projects_to_common_spine(tmp_path: Path) -> None:
    archive = _single_member_zip(tmp_path / "pr95.zip", _pr95_payload())

    spine = build_pr95_hnerv_spine_from_archive(archive)
    packet = parse_hprc_packet(spine.hprc_bin)
    sections = packet.section_map()
    embedded = json.loads(sections[HprcSectionKind.MANIFEST_JSON])

    assert spine.family == HprcRepresentationFamily.PR95_HNERV
    assert packet.config.decoder_family_id == 95
    assert sections[HprcSectionKind.DECODER_QW] == b"decoder-brotli"
    assert sections[HprcSectionKind.LATENTS_RC] == b"latents-brotli"
    assert sections[HprcSectionKind.RECEIVER_STATE] == b'{"pairs":600}'
    assert embedded["schema"] == "hprc_representation_spine_manifest.v1"
    assert embedded["manifest_extra"]["num_pairs"] == 600
    assert embedded["score_claim"] is False


def test_pr95_hnerv_projection_preserves_short_pair_coverage(tmp_path: Path) -> None:
    archive = _single_member_zip(tmp_path / "pr95_short.zip", _pr95_payload(pairs=1))

    spine = build_pr95_hnerv_spine_from_archive(archive)
    projection = write_representation_spine_projection(
        output_dir=tmp_path / "projection",
        spine=spine,
        basename="pr95_short",
    )
    manifest = json.loads(Path(projection["manifest_path"]).read_text())

    assert manifest["manifest"]["representation_spine"]["manifest_extra"][
        "num_pairs"
    ] == 1


def test_packed_hnerv_projects_decoder_latents_and_header(tmp_path: Path) -> None:
    decoder = b"decoder-packed-brotli"
    latents = b"latent-sidecar-brotli"
    payload = b"\xff" + len(decoder).to_bytes(3, "little") + decoder + latents
    archive = _single_member_zip(tmp_path / "hnerv.zip", payload)

    spine = build_packed_hnerv_spine_from_archive(archive)
    sections = parse_hprc_packet(spine.hprc_bin).section_map()
    state = json.loads(sections[HprcSectionKind.RECEIVER_STATE])

    assert spine.family == HprcRepresentationFamily.HNERV_PACKED
    assert sections[HprcSectionKind.DECODER_QW] == decoder
    assert sections[HprcSectionKind.LATENTS_RC] == latents
    assert state["payload_kind"] == "raw_ff_hnerv"
    assert state["header_format"] == "ff_len24"


def test_pact_nerv_vq_projects_codebook_selectors_and_state(tmp_path: Path) -> None:
    archive = tmp_path / "pvq.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("inflate.sh", "#!/bin/sh\n")
        zf.writestr("0.bin", _pvq_payload())
        zf.writestr("src/runtime.py", "pass\n")

    spine = build_pact_nerv_vq_spine_from_archive(archive)
    packet = parse_hprc_packet(spine.hprc_bin)
    sections = packet.section_map()
    embedded = json.loads(sections[HprcSectionKind.MANIFEST_JSON])

    assert spine.family == HprcRepresentationFamily.PACT_NERV_VQ
    assert packet.config.decoder_family_id == 141
    assert sections[HprcSectionKind.DECODER_QW] == b"pvq-decoder"
    assert len(sections[HprcSectionKind.CODEBOOKS_Q]) == 12
    assert sections[HprcSectionKind.SELECTORS_RC] == b"\x00\x00\x01\x00\x00\x00\x01\x00"
    assert json.loads(sections[HprcSectionKind.RECEIVER_STATE])["pvq_header"]["codebook_size"] == 2
    assert embedded["manifest_extra"]["source_payload_kind"] == "pact_nerv_vq_pvq"


def test_pact_nerv_len_prefixed_payload_projects_side_channel() -> None:
    spine = build_pact_nerv_len_prefixed_spine_from_archive_payload(
        _pact_len_prefixed_payload(magic=b"PSV2"),
        payload_kind="pact_nerv_selector_v2_psv2",
        expected_magic=b"PSV2",
        side_channel_kind="arithmetic_selector_k16",
    )
    packet = parse_hprc_packet(spine.hprc_bin)
    sections = packet.section_map()
    embedded = json.loads(sections[HprcSectionKind.MANIFEST_JSON])

    assert spine.family == HprcRepresentationFamily.PACT_NERV
    assert packet.config.decoder_family_id == 140
    assert sections[HprcSectionKind.DECODER_QW] == b"pact-decoder"
    assert sections[HprcSectionKind.LATENTS_RC] == b"\x00\x01" * 6
    assert sections[HprcSectionKind.SELECTORS_RC] == b"\x03\x04\x05"
    assert embedded["manifest_extra"]["side_channel_kind"] == "arithmetic_selector_k16"
    assert embedded["manifest_extra"]["num_pairs"] == 3


def test_pact_nerv_len_prefixed_payload_accepts_ia3_conditioning() -> None:
    spine = build_pact_nerv_len_prefixed_spine_from_archive_payload(
        _pact_len_prefixed_payload(magic=b"PIA3"),
        payload_kind="pact_nerv_ia3_pia3",
        expected_magic=b"PIA3",
        side_channel_kind="ego_pose_conditioning",
    )
    sections = parse_hprc_packet(spine.hprc_bin).section_map()
    embedded = json.loads(sections[HprcSectionKind.MANIFEST_JSON])

    assert spine.family == HprcRepresentationFamily.PACT_NERV
    assert sections[HprcSectionKind.SELECTORS_RC] == b"\x03\x04\x05"
    assert embedded["manifest_extra"]["payload_magic"] == "PIA3"
    assert embedded["manifest_extra"]["side_channel_kind"] == "ego_pose_conditioning"


def test_pact_nerv_len_prefixed_archive_projects_by_magic(tmp_path: Path) -> None:
    archive = _single_member_zip(
        tmp_path / "pact.zip",
        _pact_len_prefixed_payload(magic=b"PSV4"),
    )

    spine = build_pact_nerv_len_prefixed_spine_from_archive(archive)
    sections = parse_hprc_packet(spine.hprc_bin).section_map()
    embedded = json.loads(sections[HprcSectionKind.MANIFEST_JSON])

    assert spine.family == HprcRepresentationFamily.PACT_NERV
    assert sections[HprcSectionKind.SELECTORS_RC] == b"\x03\x04\x05"
    assert embedded["manifest_extra"]["source_payload_kind"] == "pact_nerv_selector_v4_psv4"
    assert embedded["source"]["member_name"] == "0.bin"


def test_generic_rnerv_projection_writes_false_authority_manifest(tmp_path: Path) -> None:
    spine = build_generic_neural_spine_packet(
        family=HprcRepresentationFamily.RNERV,
        decoder_blob=b"rnerv-decoder",
        latents_blob=b"rnerv-recurrent-latents",
        receiver_state_blob=b'{"recurrence":"charged"}',
        manifest_extra={"mode": "rnerv_lite_latent_generator"},
    )

    projection = write_representation_spine_projection(output_dir=tmp_path, spine=spine)

    assert projection["schema"] == HPRC_REPRESENTATION_SPINE_PROJECTION_SCHEMA
    assert projection["score_claim"] is False
    assert Path(projection["hprc_bin_path"]).is_file()
    manifest = json.loads(Path(projection["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["family"] == "rnerv"
    assert manifest["manifest"]["representation_spine"]["family"] == "rnerv"


def test_projection_cli_accepts_generic_pact_neural_blobs(tmp_path: Path) -> None:
    tool = _load_projection_tool()
    decoder = tmp_path / "decoder.bin"
    latents = tmp_path / "latents.bin"
    decoder.write_bytes(b"pact-decoder")
    latents.write_bytes(b"pact-latents")
    out = tmp_path / "out"

    rc = tool.main(
        [
            "--family",
            "pact_nerv",
            "--decoder-blob",
            decoder.as_posix(),
            "--latents-blob",
            latents.as_posix(),
            "--output-dir",
            out.as_posix(),
            "--repo-root",
            REPO.as_posix(),
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "hprc_representation_spine_manifest.json").read_text())
    assert manifest["family"] == "pact_nerv"
    assert manifest["score_claim"] is False


def test_projection_cli_accepts_pact_nerv_archive(tmp_path: Path) -> None:
    tool = _load_projection_tool()
    archive = _single_member_zip(
        tmp_path / "pact.zip",
        _pact_len_prefixed_payload(magic=b"PSV3"),
    )
    out = tmp_path / "out"

    rc = tool.main(
        [
            "--family",
            "pact_nerv",
            "--archive",
            archive.as_posix(),
            "--output-dir",
            out.as_posix(),
            "--repo-root",
            REPO.as_posix(),
        ]
    )

    assert rc == 0
    manifest = json.loads((out / "hprc_representation_spine_manifest.json").read_text())
    assert manifest["family"] == "pact_nerv"
    assert manifest["manifest"]["representation_spine"]["manifest_extra"][
        "source_payload_kind"
    ] == "pact_nerv_selector_v3_psv3"


def _load_projection_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "build_hprc_representation_spine_projection_test",
        REPO / "tools/build_hprc_representation_spine_projection.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
