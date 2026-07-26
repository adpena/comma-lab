# SPDX-License-Identifier: MIT
"""Code-correctness tests for the non-authoritative C0B-ABI0 identity seam.

The tiny locally encoded HEVC fixture below is not contest evidence, does not
exercise n600, and cannot support a score, launch, rank/kill, or promotion
claim.  It exists only to verify byte custody, receiver semantics, and resume
failure modes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tac.witness_dsl.c0b_identity_receiver import (
    ARCHIVE_SCHEMA,
    CLASS_NAMES,
    INFLATE_SH_BYTES,
    MEMBER_ORDER,
    SCIENTIFIC_ROLE_IDS,
    SOURCE_MEMBER,
    STATE_MEMBER,
    IdentityReceiverError,
    build_identity_archive,
    canonical_json_bytes,
    decode_canonical_json,
    emit_standalone_runtime,
    inflate_archive,
    inspect_hevc_matroska,
    parse_identity_archive,
    sha256_bytes,
    sha256_file,
    storage_preflight,
    validate_state_header,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FRAME_UTILS = REPO_ROOT / "upstream" / "frame_utils.py"
VIDEO_NAMES = REPO_ROOT / "upstream" / "public_test_video_names.txt"
RUNTIME_SOURCE = REPO_ROOT / "src" / "tac" / "witness_dsl" / "c0b_identity_receiver.py"
BUILD_TOOL = REPO_ROOT / "tools" / "build_c0b_identity_archive.py"


def _require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg with libx265 is required for the tiny codec fixture")
    encoders = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "libx265" not in encoders:
        pytest.skip("ffmpeg lacks the libx265 encoder required for the tiny codec fixture")
    return executable


@pytest.fixture(scope="module")
def tiny_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four deterministic synthetic frames; explicitly never scientific evidence."""

    root = tmp_path_factory.mktemp("c0b-identity-source")
    source = root / "fixture.mkv"
    subprocess.run(
        [
            _require_ffmpeg(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=2",
            "-frames:v",
            "4",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "pools=none:frame-threads=1:log-level=error",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "matroska",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    return source


@pytest.fixture(scope="module")
def built_fixture(
    tmp_path_factory: pytest.TempPathFactory,
    tiny_source: Path,
):
    root = tmp_path_factory.mktemp("c0b-identity-build")
    archive = root / "submission" / "archive.zip"
    result = build_identity_archive(
        tiny_source,
        archive_path=archive,
        frame_utils_path=FRAME_UTILS,
        source_origin="test-fixture/fixture.mkv",
        stage_pairs=1,
        fixture_only=True,
        runtime_source_path=RUNTIME_SOURCE,
        allow_local_spill=True,
    )
    return result


def _extract(archive: Path, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(root)
    return root


def _deterministic_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _rewrite_archive(
    source_archive: Path,
    destination: Path,
    *,
    state_transform=None,
    source_transform=None,
) -> Path:
    with zipfile.ZipFile(source_archive, "r") as handle:
        state = handle.read(STATE_MEMBER)
        source = handle.read(SOURCE_MEMBER)
    if state_transform is not None:
        state = state_transform(state)
    if source_transform is not None:
        source = source_transform(source)
    with zipfile.ZipFile(destination, "w", allowZip64=False) as handle:
        handle.writestr(_deterministic_info(STATE_MEMBER), state)
        handle.writestr(_deterministic_info(SOURCE_MEMBER), source)
    return destination


def _manual_frozen_decode(source: Path) -> bytes:
    import av

    spec = importlib.util.spec_from_file_location("_test_frozen_frame_utils", FRAME_UTILS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payloads: list[bytes] = []
    with av.open(str(source), mode="r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            payloads.append(module.yuv420_to_rgb(frame).contiguous().numpy().tobytes(order="C"))
    return b"".join(payloads)


def _assert_mechanical_identity_control(value) -> None:
    assert value["role"] == "mechanical_identity_control"
    assert value["identity_control_only"] is True
    assert value["scientific_evidence"] is False
    assert value["scientific_state_composed"] is False
    assert value["c0b_gate_complete"] is False


def test_canonical_json_rejects_duplicate_and_noncanonical_spellings() -> None:
    with pytest.raises(IdentityReceiverError, match="duplicate JSON key"):
        decode_canonical_json(b'{"a":1,"a":2}')
    with pytest.raises(IdentityReceiverError, match="not canonical"):
        decode_canonical_json(b'{"a": 1}')


def test_source_inspection_is_real_hevc_matroska_pair_count(tiny_source: Path) -> None:
    info = inspect_hevc_matroska(tiny_source)
    assert info.codec_name == "hevc"
    assert info.container_name == "matroska"
    assert info.pixel_format == "yuv420p"
    assert (info.width, info.height, info.frame_count, info.pair_count) == (64, 48, 4, 2)
    assert info.sha256 == sha256_file(tiny_source)


def test_archive_contains_only_charged_header_and_exact_source(
    built_fixture,
    tiny_source: Path,
) -> None:
    with zipfile.ZipFile(built_fixture.archive_path, "r") as handle:
        infos = handle.infolist()
        assert [info.filename for info in infos] == list(MEMBER_ORDER)
        assert all(info.compress_type == zipfile.ZIP_STORED for info in infos)
        assert all(info.compress_size == info.file_size for info in infos)
        assert all(info.flag_bits == 0 and info.extra == b"" and info.comment == b"" for info in infos)
        assert handle.read(SOURCE_MEMBER) == tiny_source.read_bytes()
    assert not any("weight" in name or "model" in name or "public" in name for name in MEMBER_ORDER)


def test_five_classes_and_seven_roles_are_honest_same_source_aliases(built_fixture) -> None:
    parsed = parse_identity_archive(built_fixture.archive_path)
    aliases = parsed.header["identity_aliases"]
    assert aliases["independent_streams"] is False
    assert aliases["role"] == "mechanical_identity_control"
    assert aliases["scientific_state_composed"] is False
    assert aliases["populated_scientific_role_count"] == 0
    assert tuple(row["class_name"] for row in aliases["classes"]) == CLASS_NAMES
    assert tuple(row["role"] for row in aliases["scientific_roles"]) == SCIENTIFIC_ROLE_IDS
    for row in [*aliases["classes"], *aliases["scientific_roles"]]:
        assert row["source_member"] == SOURCE_MEMBER
        assert row["storage_mode"] == "abi0_identity_control_alias_same_counted_source"
        assert row["incremental_payload_bytes"] == 0
        assert row["scientific_stream_claim"] is False
    assert all(row["scientific_role_populated"] is False for row in aliases["scientific_roles"])


def test_header_has_explicit_frame0_frame1_obligations_and_false_authority(built_fixture) -> None:
    header = parse_identity_archive(built_fixture.archive_path).header
    assert header["schema"] == ARCHIVE_SCHEMA
    assert header["pair_policy"] == {
        "sequence_length": 2,
        "pair_order": "canonical_contiguous_source_order",
        "frame0": "decoded_source_frame[2*pair_index]",
        "frame1": "decoded_source_frame[2*pair_index+1]",
        "segnet_obligation": "frame1_only",
        "posenet_obligation": "ordered_frame0_and_frame1",
        "remainder_policy": "refuse_non_pair_tail",
    }
    assert header["authority"] == {
        "fixture_only": True,
        "role": "mechanical_identity_control",
        "identity_control_only": True,
        "research_only": True,
        "scientific_evidence": False,
        "scientific_state_composed": False,
        "c0b_gate_complete": False,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _assert_mechanical_identity_control(header["authority"])


def test_abi0_identity_control_cannot_be_mistaken_for_c0b_gate_completion(built_fixture) -> None:
    parsed = parse_identity_archive(built_fixture.archive_path)
    manifest = decode_canonical_json(built_fixture.manifest_path.read_bytes())
    assert parsed.header["representation"] == "c0b_abi0_mechanical_source_identity_control"
    _assert_mechanical_identity_control(parsed.header["authority"])
    _assert_mechanical_identity_control(parsed.header["identity_aliases"])
    _assert_mechanical_identity_control(manifest)
    assert parsed.header["identity_aliases"]["populated_scientific_role_count"] == 0
    assert all(
        row["scientific_role_populated"] is False
        for row in parsed.header["identity_aliases"]["scientific_roles"]
    )


def test_manifest_reconciles_all_counted_archive_bytes(built_fixture) -> None:
    manifest = decode_canonical_json(built_fixture.manifest_path.read_bytes())
    assert manifest["member_count"] == 2
    assert manifest["counted_bytes_reconcile"] == manifest["archive_bytes"]
    assert (
        manifest["source_bytes"]
        + manifest["charged_state_bytes"]
        + manifest["zip_framing_bytes"]
        == manifest["archive_bytes"]
    )
    assert manifest["nested_codec"] == "hevc-in-matroska-inside-zip-stored"
    assert manifest["scientific_evidence"] is False
    assert manifest["score_claim"] is False
    _assert_mechanical_identity_control(manifest)


def test_archive_build_is_byte_deterministic_across_directories(
    tmp_path: Path,
    tiny_source: Path,
    built_fixture,
) -> None:
    second = build_identity_archive(
        tiny_source,
        archive_path=tmp_path / "second" / "archive.zip",
        frame_utils_path=FRAME_UTILS,
        source_origin="test-fixture/fixture.mkv",
        stage_pairs=1,
        fixture_only=True,
        runtime_source_path=RUNTIME_SOURCE,
        allow_local_spill=True,
    )
    assert second.archive_sha256 == built_fixture.archive_sha256
    assert second.archive_path.read_bytes() == built_fixture.archive_path.read_bytes()


def test_nonfixture_build_refuses_non_upstream_origin(tmp_path: Path, tiny_source: Path) -> None:
    with pytest.raises(IdentityReceiverError, match=r"must bind upstream/videos/0\.mkv"):
        build_identity_archive(
            tiny_source,
            archive_path=tmp_path / "archive.zip",
            frame_utils_path=FRAME_UTILS,
            source_origin="test-fixture/fixture.mkv",
            stage_pairs=1,
            fixture_only=False,
            runtime_source_path=RUNTIME_SOURCE,
            allow_local_spill=True,
        )


def test_nonfixture_build_refuses_bytes_other_than_frozen_upstream(
    tmp_path: Path,
    tiny_source: Path,
) -> None:
    with pytest.raises(IdentityReceiverError, match="differs from the frozen upstream video"):
        build_identity_archive(
            tiny_source,
            archive_path=tmp_path / "archive.zip",
            frame_utils_path=FRAME_UTILS,
            source_origin="upstream/videos/0.mkv",
            stage_pairs=1,
            fixture_only=False,
            runtime_source_path=RUNTIME_SOURCE,
            allow_local_spill=True,
        )


def test_builder_requires_explicit_local_spill_opt_in(tmp_path: Path, tiny_source: Path) -> None:
    with pytest.raises(IdentityReceiverError, match="local artifact output refused"):
        build_identity_archive(
            tiny_source,
            archive_path=tmp_path / "archive.zip",
            frame_utils_path=FRAME_UTILS,
            source_origin="test-fixture/fixture.mkv",
            stage_pairs=1,
            fixture_only=True,
            runtime_source_path=RUNTIME_SOURCE,
        )


def test_write_once_archive_refuses_conflicting_existing_bytes(tmp_path: Path, tiny_source: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"conflict")
    with pytest.raises(IdentityReceiverError, match="write-once destination differs"):
        build_identity_archive(
            tiny_source,
            archive_path=archive,
            frame_utils_path=FRAME_UTILS,
            source_origin="test-fixture/fixture.mkv",
            stage_pairs=1,
            fixture_only=True,
            runtime_source_path=RUNTIME_SOURCE,
            allow_local_spill=True,
        )
    assert archive.read_bytes() == b"conflict"


def test_parseback_refuses_source_mutation(tmp_path: Path, built_fixture) -> None:
    mutated = _rewrite_archive(
        built_fixture.archive_path,
        tmp_path / "mutated-source.zip",
        source_transform=lambda payload: bytes([payload[0] ^ 1]) + payload[1:],
    )
    with pytest.raises(IdentityReceiverError, match="source bytes differ"):
        parse_identity_archive(mutated)


def test_parseback_refuses_alias_metadata_that_claims_independence(tmp_path: Path, built_fixture) -> None:
    def mutate(payload: bytes) -> bytes:
        value = decode_canonical_json(payload)
        value["identity_aliases"]["independent_streams"] = True
        return canonical_json_bytes(value)

    mutated = _rewrite_archive(
        built_fixture.archive_path,
        tmp_path / "mutated-state.zip",
        state_transform=mutate,
    )
    with pytest.raises(IdentityReceiverError, match="identity alias metadata differs"):
        parse_identity_archive(mutated)


def test_parseback_refuses_zip_trailing_payload(tmp_path: Path, built_fixture) -> None:
    mutated = tmp_path / "trailing.zip"
    shutil.copyfile(built_fixture.archive_path, mutated)
    with mutated.open("ab") as handle:
        handle.write(b"forbidden-trailing-payload")
    with pytest.raises(IdentityReceiverError, match="trailing bytes"):
        parse_identity_archive(mutated)


def test_runtime_bundle_is_exact_executable_and_write_once(tmp_path: Path) -> None:
    bundle = emit_standalone_runtime(tmp_path, runtime_source_path=RUNTIME_SOURCE)
    assert bundle.inflate_python_path.read_bytes() == RUNTIME_SOURCE.read_bytes()
    assert bundle.inflate_shell_path.read_bytes() == INFLATE_SH_BYTES
    assert bundle.inflate_python_sha256 == sha256_file(RUNTIME_SOURCE)
    assert bundle.inflate_shell_sha256 == sha256_bytes(INFLATE_SH_BYTES)
    assert os.stat(bundle.inflate_shell_path).st_mode & 0o777 == 0o755
    bundle.inflate_shell_path.write_bytes(b"conflict")
    with pytest.raises(IdentityReceiverError, match="write-once destination differs"):
        emit_standalone_runtime(tmp_path, runtime_source_path=RUNTIME_SOURCE)


def test_full_inflate_matches_frozen_frame_utils_exactly(
    tmp_path: Path,
    built_fixture,
    tiny_source: Path,
) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    result = inflate_archive(extracted, tmp_path / "output", VIDEO_NAMES)
    expected = _manual_frozen_decode(tiny_source)
    assert result.completed is True
    assert result.raw_path is not None
    assert result.raw_path.read_bytes() == expected
    assert result.raw_sha256 == hashlib.sha256(expected).hexdigest()
    assert result.raw_bytes == len(expected) == 4 * 48 * 64 * 3
    assert result.stages_preserved == result.stage_count == 2
    inflate_manifest = decode_canonical_json(
        (
            tmp_path
            / "output"
            / ".c0b-abi0-identity-receiver"
            / "0"
            / "inflate-manifest.json"
        ).read_bytes()
    )
    _assert_mechanical_identity_control(inflate_manifest)


def test_stop_after_stage_resumes_without_replacing_checkpoint(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    output = tmp_path / "output"
    partial = inflate_archive(extracted, output, VIDEO_NAMES, stop_after_stage=0)
    assert partial.completed is False
    assert partial.raw_path is None
    assert partial.stages_preserved == 1
    stage = output / ".c0b-abi0-identity-receiver" / "0" / "stage-000000.raw"
    state = stage.with_suffix(".json")
    before = (stage.stat().st_ino, stage.read_bytes(), state.read_bytes())
    _assert_mechanical_identity_control(decode_canonical_json(state.read_bytes()))
    complete = inflate_archive(extracted, output, VIDEO_NAMES)
    after = (stage.stat().st_ino, stage.read_bytes(), state.read_bytes())
    assert complete.completed is True
    assert before == after
    idempotent = inflate_archive(extracted, output, VIDEO_NAMES)
    assert idempotent.raw_sha256 == complete.raw_sha256


def test_resume_refuses_mutated_preserved_stage_before_progress(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    output = tmp_path / "output"
    inflate_archive(extracted, output, VIDEO_NAMES, stop_after_stage=0)
    stage = output / ".c0b-abi0-identity-receiver" / "0" / "stage-000000.raw"
    with stage.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(IdentityReceiverError, match="custody drifted"):
        inflate_archive(extracted, output, VIDEO_NAMES)
    assert not (output / "0.raw").exists()
    assert not (stage.parent / "stage-000001.raw").exists()


def test_resume_refuses_mutated_stage_state_before_progress(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    output = tmp_path / "output"
    inflate_archive(extracted, output, VIDEO_NAMES, stop_after_stage=0)
    state_path = output / ".c0b-abi0-identity-receiver" / "0" / "stage-000000.json"
    state = decode_canonical_json(state_path.read_bytes())
    state["score_claim"] = True
    state_path.write_bytes(canonical_json_bytes(state))
    with pytest.raises(IdentityReceiverError, match="custody drifted"):
        inflate_archive(extracted, output, VIDEO_NAMES)


def test_receiver_refuses_frozen_frame_utils_hash_mismatch(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "public_test_video_names.txt").write_text("0.mkv\n", encoding="utf-8")
    (frozen / "frame_utils.py").write_bytes(FRAME_UTILS.read_bytes() + b"\n# drift\n")
    with pytest.raises(IdentityReceiverError, match=r"frame_utils\.py identity differs"):
        inflate_archive(extracted, tmp_path / "output", frozen / "public_test_video_names.txt")


def test_receiver_refuses_extra_extracted_payload(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    (extracted / "forbidden.weights").write_bytes(b"no")
    with pytest.raises(IdentityReceiverError, match="exactly the two charged members"):
        inflate_archive(extracted, tmp_path / "output", VIDEO_NAMES)


def test_receiver_refuses_video_name_escape(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "public_test_video_names.txt").write_text("../escape.mkv\n", encoding="utf-8")
    (frozen / "frame_utils.py").write_bytes(FRAME_UTILS.read_bytes())
    with pytest.raises(IdentityReceiverError, match="safe relative"):
        inflate_archive(extracted, tmp_path / "output", frozen / "public_test_video_names.txt")
    assert not (tmp_path / "escape.raw").exists()


def test_receiver_refuses_symlinked_checkpoint_directory(tmp_path: Path, built_fixture) -> None:
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    output = tmp_path / "output"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / ".c0b-abi0-identity-receiver").symlink_to(outside, target_is_directory=True)
    with pytest.raises(IdentityReceiverError, match="directory chain contains a symlink"):
        inflate_archive(extracted, output, VIDEO_NAMES)
    assert list(outside.iterdir()) == []


def test_standalone_runtime_executes_without_tac_import(
    tmp_path: Path,
    built_fixture,
) -> None:
    submission = tmp_path / "submission"
    runtime = emit_standalone_runtime(submission, runtime_source_path=RUNTIME_SOURCE)
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    completed = subprocess.run(
        [
            "sh",
            str(runtime.inflate_shell_path),
            str(extracted),
            str(tmp_path / "output"),
            str(VIDEO_NAMES),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )
    receipt = json.loads(completed.stdout)
    assert receipt["completed"] is True
    assert receipt["launch_ready"] is False
    assert receipt["score_claim"] is False
    assert receipt["promotion_eligible"] is False
    _assert_mechanical_identity_control(receipt)


def test_standalone_runtime_refuses_its_own_source_mutation(tmp_path: Path, built_fixture) -> None:
    submission = tmp_path / "submission"
    runtime = emit_standalone_runtime(submission, runtime_source_path=RUNTIME_SOURCE)
    extracted = _extract(built_fixture.archive_path, tmp_path / "extracted")
    with runtime.inflate_python_path.open("ab") as handle:
        handle.write(b"\n# forbidden runtime drift\n")
    refused = subprocess.run(
        [
            "sh",
            str(runtime.inflate_shell_path),
            str(extracted),
            str(tmp_path / "output"),
            str(VIDEO_NAMES),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHON_BIN": sys.executable},
    )
    assert refused.returncode != 0
    assert "runtime source identity differs" in refused.stderr
    assert not (tmp_path / "output" / "0.raw").exists()


def test_builder_cli_emits_archive_runtime_and_false_authority(
    tmp_path: Path,
    tiny_source: Path,
) -> None:
    submission = tmp_path / "submission"
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILD_TOOL),
            "--submission-dir",
            str(submission),
            "--source-video",
            str(tiny_source),
            "--frame-utils",
            str(FRAME_UTILS),
            "--source-origin",
            "test-fixture/fixture.mkv",
            "--stage-pairs",
            "1",
            "--fixture-only",
            "--allow-local-spill",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    receipt = json.loads(completed.stdout)
    assert parse_identity_archive(submission / "archive.zip").archive_sha256 == receipt["archive_sha256"]
    assert (submission / "inflate.py").is_file()
    assert (submission / "inflate.sh").is_file()
    assert receipt["fixture_only"] is True
    assert receipt["scientific_evidence"] is False
    assert receipt["score_claim"] is False
    _assert_mechanical_identity_control(receipt)


def test_storage_preflight_reports_contest_output_without_local_authority(tmp_path: Path) -> None:
    receipt = storage_preflight(tmp_path, 0, contest_output=True)
    assert receipt["tier"] == "contest-output"
    assert receipt["passed"] is True
    assert receipt["required_bytes"] == 0


def test_validate_state_header_rejects_added_claim_field(built_fixture) -> None:
    value = dict(parse_identity_archive(built_fixture.archive_path).header)
    value["score"] = 0.0
    with pytest.raises(IdentityReceiverError, match="fields differ"):
        validate_state_header(value)
