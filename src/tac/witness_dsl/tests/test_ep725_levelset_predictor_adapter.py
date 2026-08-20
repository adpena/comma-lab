# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import struct
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryCoefficientDelta,
    ReceiverRealizationProfileV1,
)
from tac.witness_dsl import ep725_levelset_predictor_adapter as ep725
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    Frame1AnchoredY0FibreControlV1,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (
    EP725_ARCHIVE_SHA256,
    EP725_BOUNDED_RECEIPT_SCHEMA,
    EP725_CANONICAL_RENDERER_SHA256,
    EP725_CAUSAL_RECEIPT_SCHEMA,
    EP725_MEMBER_SHA256,
    EP725_RUNTIME_SHA256,
    EP725_SOURCE_DIRECTORY,
    Ep725BoundedDecodeReceiptV2,
    Ep725CountedMemberCausalDecodeReceiptV3,
    Ep725LevelsetPredictorAdapterError,
    Ep725PredictorDecodeBlocker,
    Ep725SourceExpectationsV2,
    Ep725SourceObjectV2,
    _load_canonical_decoder_module,
    _shipped_decode_once,
    decode_ep725_counted_member_ephemeral_surface,
    decode_ep725_prefix_ephemeral_surface,
    inspect_ep725_source,
    parse_ep725_bounded_decode_receipt,
    parse_ep725_counted_member_causal_decode_receipt,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_predictor_state_v2 import (
    NONE_TRANSPORT_CONTENT_SHA256,
    TransportKindV2,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    SIGNAL_LOSS_FULL_REPAINT,
    TaskspacePredictorV2SignalBlocker,
    compile_coupled_preimage_program_v2,
    compile_generative_taskspace_correction_v2,
    decode_coupled_preimage_program_v2,
    derive_g_correction_ownership_v2,
    require_signal_preserving_ep725_a_surface,
)


def _lvls1(*, trailing: bytes = b"") -> tuple[bytes, bytes]:
    manifest = json.dumps(
        {
            "camera_h": 874,
            "camera_w": 1164,
            "has_pose_sidecar": False,
            "n_classes": 5,
            "n_pairs": 2,
            "render_h": 384,
            "render_w": 512,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    blocks = (manifest, b"base", b"code", b"")
    member = b"LVLS1\x00" + b"".join(struct.pack("<I", len(block)) + block for block in blocks)
    return member + trailing, manifest


def _strict_zip(member: bytes, *, date_time: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo("0.bin", date_time=date_time)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output, mode="w") as handle:
        handle.writestr(info, member)
    return output.getvalue()


def _expectations(archive: bytes, member: bytes, runtime: bytes, manifest: bytes) -> Ep725SourceExpectationsV2:
    return Ep725SourceExpectationsV2(
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        archive_bytes=len(archive),
        member_sha256=hashlib.sha256(member).hexdigest(),
        member_bytes=len(member),
        runtime_sha256=hashlib.sha256(runtime).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        manifest_bytes=len(manifest),
        n_pairs=2,
        render_height=384,
        render_width=512,
        camera_height=874,
        camera_width=1164,
        n_classes=5,
    )


def _source_fixture(
    tmp_path: Path, *, member: bytes, manifest: bytes, archive: bytes | None = None
) -> tuple[Path, Ep725SourceExpectationsV2]:
    source = tmp_path / "source"
    source.mkdir()
    runtime = b"# exact synthetic runtime source\n"
    archive_bytes = _strict_zip(member) if archive is None else archive
    (source / "archive.zip").write_bytes(archive_bytes)
    (source / "inflate.py").write_bytes(runtime)
    return source, _expectations(archive_bytes, member, runtime, manifest)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _synthetic_receipt() -> Ep725BoundedDecodeReceiptV2:
    labels_sha256 = "9165ef7fc86fe997e01b26ecf751e877b8ae30352e7b3d3db659d920a908068e"
    renderer_sha256 = EP725_CANONICAL_RENDERER_SHA256
    semantic = hashlib.sha256(
        _canonical_json(
            {
                "schema": "tac.taskspace_predictor_semantic_binding.v2",
                "predictor_program_sha256": EP725_MEMBER_SHA256,
                "predictor_renderer_sha256": renderer_sha256,
                "source_archive_sha256": EP725_ARCHIVE_SHA256,
                "source_runtime_sha256": EP725_RUNTIME_SHA256,
                "source_member_name": "0.bin",
                "source_pair_ids": [0, 1],
                "labels_sha256": labels_sha256,
            }
        )
    ).hexdigest()
    binding = hashlib.sha256(
        _canonical_json(
            {
                "schema": "tac.taskspace_predictor_binding.v2",
                "semantic_binding_sha256": semantic,
                "transport": {
                    "kind": "NONE",
                    "counted_bytes": 0,
                    "counted_content_sha256": NONE_TRANSPORT_CONTENT_SHA256,
                    "decoded_values_sha256": NONE_TRANSPORT_CONTENT_SHA256,
                },
            }
        )
    ).hexdigest()
    return Ep725BoundedDecodeReceiptV2(
        schema=EP725_BOUNDED_RECEIPT_SCHEMA,
        predictor_state_binding_sha256=binding,
        predictor_semantic_binding_sha256=semantic,
        source_archive_sha256=EP725_ARCHIVE_SHA256,
        source_archive_bytes=83_838,
        source_member_sha256=EP725_MEMBER_SHA256,
        counted_predictor_bytes=84_536,
        source_runtime_sha256=EP725_RUNTIME_SHA256,
        canonical_renderer_sha256=renderer_sha256,
        source_pair_ids=(0, 1),
        chronological_camera_frame_sha256=(
            "fdd3d4d7a7d8c26f48c6dc89f248eda381971b08ea7cb4e9f0c71289a07e2e01",
            "38dfcaa5d3ef49467eca94d788f480feb6583f36ac58f6f7b47a06f2adddf758",
            "b4d9a6ec279780f10c7a05d76188c882d0b6004ef21cb544f0d757cc2290e1c5",
            "06cb6d8ef75fb230018fdbed20f55202e44dd2a0adeeacfc815e0d283814504e",
        ),
        chronological_raw_prefix_sha256="23f12caeb9f43149d94732289a7cbac34ef08c3171c6613750074e3c8e76b9e9",
        chronological_raw_prefix_bytes=4 * 874 * 1164 * 3,
        labels_sha256=labels_sha256,
        decoder_labels_origin="same_counted_LVLS1_member_decoder_phi_argmax",
        transport_kind="NONE",
        deterministic_double_decode=True,
        shipped_numpy_raw_bit_exact=True,
        source_zip_wrapper_is_counted_p=False,
        member_is_counted_p=True,
        dense_frames_persisted=False,
        scorer_invoked=False,
        score_claim=False,
        candidate_claim=False,
        promotion_eligible=False,
        research_only=True,
    )


def test_synthetic_strict_source_inspection_binds_exact_member_not_zip(tmp_path: Path) -> None:
    member, manifest = _lvls1()
    source, expectations = _source_fixture(tmp_path, member=member, manifest=manifest)

    inspected = inspect_ep725_source(source, expectations=expectations)

    assert inspected.member == member
    assert inspected.archive != member
    assert inspected.member_sha256 == expectations.member_sha256
    assert inspected.manifest_sha256 == expectations.manifest_sha256
    assert inspected.manifest["n_pairs"] == 2


def test_zip_trailing_bytes_and_noncanonical_metadata_fail_closed(tmp_path: Path) -> None:
    member, manifest = _lvls1()
    canonical = _strict_zip(member)
    source, expectations = _source_fixture(
        tmp_path,
        member=member,
        manifest=manifest,
        archive=canonical + b"trailing",
    )
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="trailing"):
        inspect_ep725_source(source, expectations=expectations)

    altered = _strict_zip(member, date_time=(1980, 1, 2, 0, 0, 0))
    (source / "archive.zip").write_bytes(altered)
    altered_expectations = replace(
        expectations,
        archive_sha256=hashlib.sha256(altered).hexdigest(),
        archive_bytes=len(altered),
    )
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="metadata"):
        inspect_ep725_source(source, expectations=altered_expectations)


def test_nested_zip_and_unknown_lvls1_trailing_block_fail_closed(tmp_path: Path) -> None:
    inner_member, inner_manifest = _lvls1()
    nested = _strict_zip(inner_member)
    source, expectations = _source_fixture(
        tmp_path,
        member=nested,
        manifest=inner_manifest,
    )
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="nested ZIP"):
        inspect_ep725_source(source, expectations=expectations)

    trailing_member, manifest = _lvls1(trailing=struct.pack("<I", 1) + b"x")
    trailing_archive = _strict_zip(trailing_member)
    (source / "archive.zip").write_bytes(trailing_archive)
    trailing_expectations = _expectations(
        trailing_archive,
        trailing_member,
        (source / "inflate.py").read_bytes(),
        manifest,
    )
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="trailing/unknown"):
        inspect_ep725_source(source, expectations=trailing_expectations)


def test_archive_and_runtime_symlinks_are_refused(tmp_path: Path) -> None:
    member, manifest = _lvls1()
    source, expectations = _source_fixture(tmp_path, member=member, manifest=manifest)
    real_archive = tmp_path / "real.zip"
    real_archive.write_bytes((source / "archive.zip").read_bytes())
    (source / "archive.zip").unlink()
    (source / "archive.zip").symlink_to(real_archive)
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="regular non-symlink"):
        inspect_ep725_source(source, expectations=expectations)

    (source / "archive.zip").unlink()
    (source / "archive.zip").write_bytes(real_archive.read_bytes())
    real_runtime = tmp_path / "runtime.py"
    real_runtime.write_bytes((source / "inflate.py").read_bytes())
    (source / "inflate.py").unlink()
    (source / "inflate.py").symlink_to(real_runtime)
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="regular non-symlink"):
        inspect_ep725_source(source, expectations=expectations)


def test_exact_hash_and_manifest_mutations_fail_closed(tmp_path: Path) -> None:
    member, manifest = _lvls1()
    source, expectations = _source_fixture(tmp_path, member=member, manifest=manifest)
    (source / "inflate.py").write_bytes(b"mutated runtime")
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="runtime_sha256"):
        inspect_ep725_source(source, expectations=expectations)


def test_canonical_decoder_executes_descriptor_bytes_not_stale_sys_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = ModuleType("tools.levelset_byte_close_and_eval")
    stale.numpy_oracle_reference_frames = lambda: "stale"
    monkeypatch.setitem(sys.modules, "tools.levelset_byte_close_and_eval", stale)

    module, source_sha256 = _load_canonical_decoder_module()

    assert module is not stale
    assert module.__name__ == "_tac_ep725_descriptor_bound_levelset_decoder"
    assert callable(module.numpy_oracle_reference_frames)
    assert len(source_sha256) == 64


def test_shipped_output_and_input_reopen_are_descriptor_bound(tmp_path: Path) -> None:
    member, manifest = _lvls1()
    expected_raw_bytes = 2 * 874 * 1164 * 3
    symlink_runtime = (
        "import os,sys\n"
        f"n={expected_raw_bytes}\n"
        "target=sys.argv[2]+'.target'\n"
        "with open(target,'wb') as f: f.truncate(n)\n"
        "os.symlink(target,sys.argv[2])\n"
    ).encode()
    source = Ep725SourceObjectV2(
        source_directory=tmp_path,
        archive=b"archive",
        member=member,
        runtime=symlink_runtime,
        manifest_bytes=manifest,
        manifest=json.loads(manifest),
    )
    with pytest.raises(Ep725PredictorDecodeBlocker, match="OUTPUT_MISSING"):
        _shipped_decode_once(source, 1, timeout_seconds=10.0)

    mutate_runtime = (
        "import sys\n"
        f"n={expected_raw_bytes}\n"
        "with open(sys.argv[1],'ab') as f: f.write(b'x')\n"
        "with open(sys.argv[2],'wb') as f: f.truncate(n)\n"
    ).encode()
    mutated_source = replace(source, runtime=mutate_runtime)
    with pytest.raises(Ep725PredictorDecodeBlocker, match="MEMBER_COPY_MUTATED"):
        _shipped_decode_once(mutated_source, 1, timeout_seconds=10.0)


def test_bounded_receipt_is_canonical_recomposable_and_truth_closed() -> None:
    receipt = _synthetic_receipt()
    payload = receipt.to_receipt_bytes()

    parsed = parse_ep725_bounded_decode_receipt(payload)

    assert parsed == receipt
    assert parsed.to_receipt_bytes() == payload
    assert parsed.receipt_sha256 == hashlib.sha256(payload).hexdigest()

    forged = receipt.as_dict()
    forged["score_claim"] = True
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="truth flag score_claim"):
        parse_ep725_bounded_decode_receipt(_canonical_json(forged))
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="byte-canonical"):
        parse_ep725_bounded_decode_receipt(payload + b"\n")
    duplicate = payload[:-1] + b',"schema":"tac.ep725_bounded_predictor_decode_receipt.v2"}'
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="repeats key"):
        parse_ep725_bounded_decode_receipt(duplicate)


@pytest.mark.skipif(not EP725_SOURCE_DIRECTORY.is_dir(), reason="frozen ep725 source SSD is unavailable")
def test_real_explicit_member_runtime_decode_is_source_independent_and_causally_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inspect_ep725_source()

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("source archive helper must not run during explicit-input decode")

    monkeypatch.setattr(ep725, "_read_source_directory", explode)
    monkeypatch.setattr(ep725, "inspect_ep725_source", explode)

    causal_surface = decode_ep725_counted_member_ephemeral_surface(
        source.member,
        shipped_runtime=source.runtime,
        pair_count=2,
        timeout_seconds=180.0,
    )
    receipt = causal_surface.causal_receipt
    assert type(receipt) is Ep725CountedMemberCausalDecodeReceiptV3
    assert receipt.schema == EP725_CAUSAL_RECEIPT_SCHEMA
    assert receipt.counted_member_sha256 == EP725_MEMBER_SHA256
    assert receipt.explicit_runtime_sha256 == EP725_RUNTIME_SHA256
    assert receipt.source_directory_or_archive_read is False
    assert receipt.source_archive_bytes_read == 0
    assert receipt.historical_archive_lineage_is_decode_input is False
    assert receipt.counted_member_consumed_by_canonical_numpy is True
    assert receipt.counted_member_consumed_by_shipped_runtime is True
    assert receipt.explicit_runtime_executed is True
    assert receipt.canonical_renderer_source_executed is True
    assert receipt.legacy_v2_bounded_receipt_sha256 == causal_surface.bounded_decode.receipt_sha256
    assert parse_ep725_counted_member_causal_decode_receipt(receipt.to_receipt_bytes()) == receipt

    substituted_member = bytearray(source.member)
    substituted_member[-1] ^= 0x01
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match=r"LVLS1|member_sha256"):
        decode_ep725_counted_member_ephemeral_surface(
            bytes(substituted_member),
            shipped_runtime=source.runtime,
            pair_count=2,
        )

    substituted_runtime = bytearray(source.runtime)
    substituted_runtime[-1] ^= 0x01
    with pytest.raises(Ep725LevelsetPredictorAdapterError, match="runtime differs"):
        decode_ep725_counted_member_ephemeral_surface(
            source.member,
            shipped_runtime=bytes(substituted_runtime),
            pair_count=2,
        )


@pytest.mark.skipif(not EP725_SOURCE_DIRECTORY.is_dir(), reason="frozen ep725 source SSD is unavailable")
def test_real_ep725_n2_double_decode_is_hash_bound_bit_exact_and_label_owned() -> None:
    surface = decode_ep725_prefix_ephemeral_surface(pair_count=2, timeout_seconds=120.0)
    result = surface.bounded_decode
    state = result.predictor_state

    assert result.source_archive_sha256 == EP725_ARCHIVE_SHA256
    assert result.source_member_sha256 == EP725_MEMBER_SHA256
    assert result.source_runtime_sha256 == EP725_RUNTIME_SHA256
    assert result.counted_predictor_bytes == 84_536
    assert result.source_archive_bytes == 83_838
    assert result.source_pair_ids == (0, 1)
    assert result.canonical_renderer_sha256 == "1cecaa3ee9873d1378eaeb66b04cee56b621bccabe3fcecc01f60d3a0e692ddc"
    assert result.labels_sha256 == "9165ef7fc86fe997e01b26ecf751e877b8ae30352e7b3d3db659d920a908068e"
    assert result.chronological_raw_prefix_sha256 == "23f12caeb9f43149d94732289a7cbac34ef08c3171c6613750074e3c8e76b9e9"
    assert result.chronological_camera_frame_sha256 == (
        "fdd3d4d7a7d8c26f48c6dc89f248eda381971b08ea7cb4e9f0c71289a07e2e01",
        "38dfcaa5d3ef49467eca94d788f480feb6583f36ac58f6f7b47a06f2adddf758",
        "b4d9a6ec279780f10c7a05d76188c882d0b6004ef21cb544f0d757cc2290e1c5",
        "06cb6d8ef75fb230018fdbed20f55202e44dd2a0adeeacfc815e0d283814504e",
    )
    assert result.chronological_raw_prefix_bytes == 4 * 874 * 1164 * 3
    assert result.deterministic_double_decode is True
    assert result.shipped_numpy_raw_bit_exact is True
    assert result.decoder_labels_origin == "same_counted_LVLS1_member_decoder_phi_argmax"
    assert result.source_zip_wrapper_is_counted_p is False
    assert result.member_is_counted_p is True
    assert result.dense_frames_persisted is False
    assert result.scorer_invoked is False
    assert result.score_claim is False
    assert result.candidate_claim is False
    assert result.promotion_eligible is False
    assert result.research_only is True
    assert state.predictor_program_sha256 == EP725_MEMBER_SHA256
    assert state.predictor_program_bytes == 84_536
    assert state.source_archive_sha256 == EP725_ARCHIVE_SHA256
    assert state.source_runtime_sha256 == EP725_RUNTIME_SHA256
    assert state.source_pair_ids == (0, 1)
    assert state.labels.shape == (2, 384, 512)
    assert state.labels.dtype.name == "uint8"
    assert state.labels.flags.writeable is False
    assert state.transport.kind is TransportKindV2.NONE
    assert not hasattr(state.transport, "pose6_codes")
    assert surface.chronological_camera_frames.shape == (2, 2, 874, 1164, 3)
    assert surface.chronological_camera_frames.dtype == np.uint8
    assert surface.chronological_camera_frames.flags.writeable is False
    assert surface.frame1_camera.shape == (2, 874, 1164, 3)
    assert surface.frame1_camera.flags.writeable is False
    assert surface.persisted is False
    assert surface.additional_counted_bytes == 0

    receipt_bytes = result.to_receipt_bytes()
    parsed_receipt = parse_ep725_bounded_decode_receipt(receipt_bytes)
    assert result.receipt_sha256 == "eb765a3e3252155857861dc11564a98d46b4147190ef634d710af69c1ebe7445"
    assert state.semantic_binding_sha256 == "4ec471a707a37d38f704586584f84fd5d0adb4ec2a70a8069481dc6e6113197d"
    assert state.binding_sha256 == "a25ee50104fc887ba5ee5e92110caac736ee431338584a3ad6ac8b54ab7cfae9"
    assert parsed_receipt.predictor_state_binding_sha256 == state.binding_sha256
    assert parsed_receipt.predictor_semantic_binding_sha256 == state.semantic_binding_sha256
    assert parsed_receipt.receipt_sha256 == result.receipt_sha256
    assert parsed_receipt.to_receipt_bytes() == receipt_bytes

    profile = ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )
    local_g = GenerativeCorrectionProgramV1(
        boundary_coefficients=(BoundaryCoefficientDelta(0, "Road", 0, 3.0),),
        realization_profile=profile,
    )
    target = state.labels.copy()
    target[0, 0, 0] = np.uint8((int(target[0, 0, 0]) + 1) % 5)
    teacher = EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=hashlib.sha256(memoryview(target).cast("B")).hexdigest(),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=1,
    )
    compiled_g = compile_generative_taskspace_correction_v2(
        state,
        local_g,
        teacher_evidence=teacher,
    )
    assert compiled_g.receipt.predictor_binding_sha256 == state.binding_sha256
    assert compiled_g.receipt.changed_cells > 0
    ownership = derive_g_correction_ownership_v2(state, compiled_g.decoded)
    assert ownership.changed_cells == compiled_g.receipt.changed_cells
    assert np.array_equal(ownership.owned_mask, compiled_g.decoded.labels != state.labels)
    overlay = overlay_g_on_predictor_camera_y1(
        surface.frame1_camera,
        state.labels,
        compiled_g.decoded,
    )
    assert np.array_equal(overlay.changed_label_mask, ownership.owned_mask)
    assert np.array_equal(
        overlay.camera_y1[~overlay.owned_camera_mask],
        surface.frame1_camera[~overlay.owned_camera_mask],
    )
    assert overlay.receipt.changed_scorer_cells == ownership.changed_cells
    assert (
        overlay.receipt.base_camera_y1_sha256
        == hashlib.sha256(memoryview(np.ascontiguousarray(surface.frame1_camera)).cast("B")).hexdigest()
    )
    assert overlay.receipt.predictor_rgb_preserved_outside_g is True
    assert overlay.receipt.full_frame_palette_repaint is False
    assert overlay.receipt.dense_predictor_rgb_serialized is False
    assert overlay.receipt.scorer_invoked is False
    assert overlay.receipt.score_claim is False
    assert overlay.receipt.candidate_claim is False
    with pytest.raises(TaskspacePredictorV2SignalBlocker) as blocked:
        require_signal_preserving_ep725_a_surface(state)
    assert blocked.value.code == SIGNAL_LOSS_FULL_REPAINT

    compiled_a = compile_coupled_preimage_program_v2(
        state,
        compiled_g.decoded,
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=(
            Frame1AnchoredY0FibreControlV1(0, 1, -2, (3, -4, 5)),
            Frame1AnchoredY0FibreControlV1(1, -1, 2, (-2, 4, 1)),
        ),
    )
    replay_a = decode_coupled_preimage_program_v2(
        compiled_a.packet,
        predictor_state=state,
        decoded_g=compiled_g.decoded,
    )
    assert replay_a.chronological_content_sha256 == compiled_a.receipt.chronological_content_sha256
    assert replay_a.chronological_frames.shape == (2, 2, 384, 512, 3)
