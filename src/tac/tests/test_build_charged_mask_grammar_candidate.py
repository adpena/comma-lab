from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
import zlib
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = REPO_ROOT / "experiments" / "build_charged_mask_grammar_candidate.py"
INFLATE_RENDERER_PATH = REPO_ROOT / "submissions" / "robust_current" / "inflate_renderer.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_charged_mask_grammar_candidate_test", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_inflate_renderer():
    spec = importlib.util.spec_from_file_location("inflate_renderer_cmg1_test", INFLATE_RENDERER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {info.filename: zf.read(info) for info in zf.infolist()}


def _write_runtime_members(path: Path, *, mask_bytes: bytes = b"source-mask-stream") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in (
            ("renderer.bin", b"renderer"),
            ("masks.mkv", mask_bytes),
            ("optimized_poses.bin", b"poses"),
        ):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            zf.writestr(info, data)


def test_raw_mask_stream_build_is_deterministic_and_charged(tmp_path: Path) -> None:
    builder = _load_builder()
    mask_stream = tmp_path / "masks.obu"
    raw = b"obu-mask-stream\x00\x01" * 17
    mask_stream.write_bytes(raw)
    same_bytes_different_name = tmp_path / "renamed-mask-stream.bin"
    same_bytes_different_name.write_bytes(raw)

    out_a = tmp_path / "a" / "archive.zip"
    out_b = tmp_path / "b" / "archive.zip"
    out_c = tmp_path / "c" / "archive.zip"
    meta_a = builder.build_candidate(
        input_mask_stream=mask_stream,
        output_archive=out_a,
        provenance_json=tmp_path / "a" / "build_provenance.json",
    )
    meta_b = builder.build_candidate(
        input_mask_stream=mask_stream,
        output_archive=out_b,
        provenance_json=tmp_path / "b" / "build_provenance.json",
    )
    meta_c = builder.build_candidate(
        input_mask_stream=same_bytes_different_name,
        output_archive=out_c,
        provenance_json=tmp_path / "c" / "build_provenance.json",
    )

    assert meta_a["output_archive_sha256"] == meta_b["output_archive_sha256"]
    assert meta_a["output_archive_sha256"] == meta_c["output_archive_sha256"]
    assert meta_a["payload_sha256"] == meta_b["payload_sha256"]
    assert meta_a["payload_sha256"] == meta_c["payload_sha256"]
    assert out_a.read_bytes() == out_b.read_bytes()
    assert out_a.read_bytes() == out_c.read_bytes()

    members = _read_members(out_a)
    assert list(members) == ["mask.cmg1", "cmg1_manifest.json"]
    payload = builder.decode_cmg1_payload(members["mask.cmg1"])
    assert payload["fixed_header"]["magic"] == "CMG1"
    assert payload["fixed_header"]["frames"] == 600
    assert payload["fixed_header"]["height"] == 384
    assert payload["fixed_header"]["width"] == 512
    assert payload["fixed_header"]["class_count"] == 5
    assert payload["raw_stream"] == raw

    manifest = json.loads(members["cmg1_manifest.json"])
    assert manifest["schema"] == builder.SCHEMA
    assert manifest["mode"] == builder.MODE_RAW_BIT_IDENTICAL
    assert manifest["score_claim"] is False
    assert manifest["source_mask_stream"]["bytes"] == len(raw)
    assert manifest["source_mask_stream"]["sha256"] == builder._sha256_bytes(raw)
    assert manifest["charged_member_accounting"]["mask.cmg1"]["sha256"] == builder._sha256_bytes(
        members["mask.cmg1"]
    )


def test_placeholder_manifest_is_strict_and_non_promotable(tmp_path: Path) -> None:
    builder = _load_builder()
    out = tmp_path / "placeholder" / "archive.zip"
    provenance = builder.build_candidate(input_mask_stream=None, output_archive=out)

    members = _read_members(out)
    manifest = json.loads(members["cmg1_manifest.json"])
    payload = builder.decode_cmg1_payload(members["mask.cmg1"])

    assert provenance["score_claim"] is False
    assert provenance["promotion_eligible"] is False
    assert provenance["exact_evaluable_archive"] is False
    assert manifest["mode"] == builder.MODE_PLACEHOLDER
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["exact_evaluable_archive"] is False
    assert manifest["runtime_integration"]["inflate_runtime_touched"] is False
    assert manifest["runtime_integration"]["requires_future_decoder"] is True
    assert manifest["source_mask_stream"] is None
    assert payload["raw_stream"] == b""


@pytest.mark.parametrize(
    "bad_name",
    [
        "../mask.cmg1",
        "/mask.cmg1",
        "nested/mask.cmg1",
        "._mask.cmg1",
        ".hidden.cmg1",
        ".DS_Store",
        "__MACOSX",
        "bad\\name.cmg1",
        "",
    ],
)
def test_rejects_zip_slip_and_hidden_member_names(tmp_path: Path, bad_name: str) -> None:
    builder = _load_builder()
    with pytest.raises(ValueError):
        builder.build_candidate(
            input_mask_stream=None,
            output_archive=tmp_path / "archive.zip",
            payload_member=bad_name,
        )


def test_emitted_zip_members_are_single_level_stored_and_deterministic(tmp_path: Path) -> None:
    builder = _load_builder()
    out = tmp_path / "archive.zip"
    builder.build_candidate(input_mask_stream=None, output_archive=out)

    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == ["mask.cmg1", "cmg1_manifest.json"]
        seen: set[str] = set()
        for info in zf.infolist():
            assert info.filename not in seen
            seen.add(info.filename)
            assert "/" not in info.filename
            assert "\\" not in info.filename
            assert not info.filename.startswith(".")
            assert info.date_time == builder.FIXED_ZIP_TIMESTAMP
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.external_attr >> 16 == 0o644


def test_non_promotable_metadata_lists_remaining_exact_eval_work(tmp_path: Path) -> None:
    builder = _load_builder()
    out = tmp_path / "archive.zip"
    builder.build_candidate(input_mask_stream=None, output_archive=out)

    manifest = json.loads(_read_members(out)["cmg1_manifest.json"])
    assert manifest["evidence_grade"] == "empirical_build_only_non_score"
    assert "no CUDA auth eval has been run" in manifest["non_promotable_reason"]
    assert manifest["runtime_integration"]["external_sidecars_allowed"] is False
    assert any(".cmg1" in step for step in manifest["required_next_steps_for_exact_evaluable_archive"])
    assert any("contest_auth_eval.py --device cuda" in step for step in manifest["required_next_steps_for_exact_evaluable_archive"])


def test_full_archive_mode_replaces_mask_member_with_runtime_cmg1_payload(tmp_path: Path) -> None:
    builder = _load_builder()
    base = tmp_path / "base.zip"
    renderer_bytes = b"renderer"
    pose_bytes = b"poses"
    old_mask_bytes = b"old-mask"
    with zipfile.ZipFile(base, "w") as zf:
        zf.writestr(builder._base_zip_info("renderer.bin"), renderer_bytes)
        zf.writestr(builder._base_zip_info("masks.mkv"), old_mask_bytes)
        zf.writestr(builder._base_zip_info("optimized_poses.pt"), pose_bytes)

    mask_stream = tmp_path / "masks.mkv"
    raw = b"\x1a\x45\xdf\xa3cmg1-test-mask-stream" * 3
    mask_stream.write_bytes(raw)
    out = tmp_path / "candidate.zip"
    provenance = builder.build_candidate(
        input_mask_stream=mask_stream,
        output_archive=out,
        base_archive=base,
        provenance_json=tmp_path / "build_provenance.json",
    )

    members = _read_members(out)
    assert list(members) == ["renderer.bin", "optimized_poses.pt", "masks.cmg1", "cmg1_manifest.json"]
    assert "masks.mkv" not in members
    assert members["renderer.bin"] == renderer_bytes
    assert members["optimized_poses.pt"] == pose_bytes
    assert provenance["base_archive"] == str(base.resolve())
    assert provenance["replaced_mask_member"] == "masks.mkv"

    manifest = json.loads(members["cmg1_manifest.json"])
    assert manifest["archive_assembly"]["full_archive_candidate"] is True
    assert manifest["runtime_integration"]["inflate_runtime_touched"] is True
    assert manifest["runtime_integration"]["requires_future_decoder"] is False
    assert manifest["runtime_integration"]["auth_eval_allowlist_update_required"] is True
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False

    payload = builder.decode_cmg1_payload(members["masks.cmg1"])
    assert payload["raw_stream"] == raw


def test_full_archive_mode_rejects_unsafe_base_members(tmp_path: Path) -> None:
    builder = _load_builder()
    base = tmp_path / "bad_base.zip"
    with zipfile.ZipFile(base, "w") as zf:
        zf.writestr("masks.mkv", b"old")
        zf.writestr("../renderer.bin", b"bad")
    mask_stream = tmp_path / "masks.mkv"
    mask_stream.write_bytes(b"mask")

    with pytest.raises(ValueError, match="unsafe base archive member"):
        builder.build_candidate(
            input_mask_stream=mask_stream,
            output_archive=tmp_path / "candidate.zip",
            base_archive=base,
        )


def test_amr1_repair_archive_is_deterministic_and_charges_runtime_supported_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    source = torch.tensor(
        [
            [[0, 1, 1, 2], [0, 2, 2, 2], [3, 3, 4, 4]],
            [[0, 1, 2, 2], [0, 0, 2, 4], [3, 3, 4, 4]],
        ],
        dtype=torch.int64,
    )
    candidate = source.clone()
    candidate[0, 0, 1] = 0
    candidate[0, 1, 2] = 1
    candidate[1, 2, 2] = 3
    candidate[1, 0, 0] = 1
    source_archive = tmp_path / "runtime_members.zip"
    _write_runtime_members(source_archive)
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy-grayscale-mask-stream")

    monkeypatch.setattr(
        builder,
        "_decode_source_masks_from_runtime_member",
        lambda data, member, *, max_frames: (source.clone(), {"decoder": "fake-source", "max_frames": max_frames}),
    )
    monkeypatch.setattr(
        builder,
        "_decode_lossy_grayscale_masks",
        lambda path, *, expected_shape: (candidate.clone(), {"decoder": "fake-grayscale", "shape": list(expected_shape)}),
    )

    policy = builder.ChargedRepairPolicy(
        label="unit_pose_sensitive",
        hard_pair_indices=(0,),
        boundary_dilation=0,
        foveal_boxes=(builder.RegionSpec(name="fovea", x0=2, y0=2, x1=4, y1=3, frames=(1,)),),
    )
    out_a = tmp_path / "a" / "archive.zip"
    manifest_a = tmp_path / "a" / "manifest.json"
    report_a = builder.build_amr1_repair_candidate(
        source_archive=source_archive,
        lossy_mask_stream=lossy,
        output_archive=out_a,
        manifest_json=manifest_a,
        policy=policy,
        repair_compressor="zlib",
    )
    out_b = tmp_path / "b" / "archive.zip"
    report_b = builder.build_amr1_repair_candidate(
        source_archive=source_archive,
        lossy_mask_stream=lossy,
        output_archive=out_b,
        manifest_json=tmp_path / "b" / "manifest.json",
        policy=policy,
        repair_compressor="zlib",
    )

    assert out_a.read_bytes() == out_b.read_bytes()
    assert report_a["archive"]["sha256"] == report_b["archive"]["sha256"]
    assert report_a["score_claim"] is False
    assert report_a["promotion_eligible"] is False
    assert report_a["exact_evaluable_archive"] is True
    assert report_a["cuda_jobs_launched"] is False
    assert report_a["runtime_contract"]["archive_members"] == [
        "renderer.bin",
        "grayscale.mkv",
        "alpha4_residual_repair.amr1.zlib",
        "optimized_poses.bin",
    ]
    assert report_a["runtime_contract"]["masks_mkv_omitted"] is True

    members = _read_members(out_a)
    assert list(members) == [
        "renderer.bin",
        "grayscale.mkv",
        "alpha4_residual_repair.amr1.zlib",
        "optimized_poses.bin",
    ]
    assert members["renderer.bin"] == b"renderer"
    assert members["grayscale.mkv"] == b"lossy-grayscale-mask-stream"
    assert members["optimized_poses.bin"] == b"poses"

    alpha = builder._load_alpha_builder()
    raw_payload = zlib.decompress(members["alpha4_residual_repair.amr1.zlib"])
    header, runs = alpha._decode_repair_payload(raw_payload)
    assert header["candidate_mask_u8_sha256"] == alpha._tensor_u8_sha256(candidate)
    assert header["source_mask_u8_sha256"] == alpha._tensor_u8_sha256(source)
    assert header["selection"]["policy"]["label"] == "unit_pose_sensitive"
    assert header["selection"]["selected_repair_pixels"] == 4
    assert header["selection"]["selected_repair_runs"] == len(runs)
    repaired = alpha._apply_repair_payload(candidate, raw_payload)
    assert int((source != repaired).sum().item()) == 0

    charged = report_a["charged_member_accounting"]["alpha4_residual_repair.amr1.zlib"]
    assert charged["role"] == "charged_amr1_residual_repair"
    assert charged["bytes"] == len(members["alpha4_residual_repair.amr1.zlib"])
    assert charged["sha256"] == builder._sha256_bytes(members["alpha4_residual_repair.amr1.zlib"])
    manifest_payload = json.loads(manifest_a.read_text())
    assert manifest_payload["repair_payload"]["compressed_size_bytes"] == charged["bytes"]
    assert manifest_payload["repair_selector"]["selected_repair_pixels_before_budget"] == 4


def test_amr1_repair_archive_rejects_unexpected_runtime_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    source_archive = tmp_path / "runtime_members.zip"
    with zipfile.ZipFile(source_archive, "w") as zf:
        zf.writestr("renderer.bin", b"renderer")
        zf.writestr("masks.mkv", b"mask")
        zf.writestr("optimized_poses.bin", b"poses")
        zf.writestr("debug.txt", b"sidecar")
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy")

    monkeypatch.setattr(
        builder,
        "_decode_source_masks_from_runtime_member",
        lambda data, member, *, max_frames: (_ for _ in ()).throw(AssertionError("decode should not run")),
    )

    with pytest.raises(ValueError, match="unexpected runtime archive member"):
        builder.build_amr1_repair_candidate(
            source_archive=source_archive,
            lossy_mask_stream=lossy,
            output_archive=tmp_path / "archive.zip",
            policy=builder.ChargedRepairPolicy(select_all_differences=True),
        )


def test_amr1_repair_payload_accounting_respects_repair_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    source = torch.tensor([[[2, 2, 2, 2], [1, 1, 1, 1]]], dtype=torch.int64)
    candidate = torch.zeros_like(source)
    source_archive = tmp_path / "runtime_members.zip"
    _write_runtime_members(source_archive)
    lossy = tmp_path / "lossy.mkv"
    lossy.write_bytes(b"lossy")
    monkeypatch.setattr(
        builder,
        "_decode_source_masks_from_runtime_member",
        lambda data, member, *, max_frames: (source.clone(), {"decoder": "fake-source"}),
    )
    monkeypatch.setattr(
        builder,
        "_decode_lossy_grayscale_masks",
        lambda path, *, expected_shape: (candidate.clone(), {"decoder": "fake-grayscale"}),
    )

    report = builder.build_amr1_repair_candidate(
        source_archive=source_archive,
        lossy_mask_stream=lossy,
        output_archive=tmp_path / "archive.zip",
        policy=builder.ChargedRepairPolicy(select_all_differences=True, boundary_dilation=0),
        repair_compressor="raw",
        max_repair_pixels=4,
        allow_partial_repair=True,
    )

    repair = report["repair_payload"]
    assert repair["archive_member"] == "alpha4_residual_repair.amr1"
    assert repair["selection"]["total_residual_pixels"] == 8
    assert repair["selection"]["policy_selected_repair_pixels_before_budget"] == 8
    assert repair["selection"]["selected_repair_pixels"] == 4
    assert repair["selection"]["partial_repair"] is True
    assert repair["selection"]["budget_limited"] is True
    assert repair["selection"]["partial_reason"].startswith("max_repair_pixels")
    members = _read_members(tmp_path / "archive.zip")
    assert len(members["alpha4_residual_repair.amr1"]) == repair["raw_amr1_size_bytes"]


def test_amr1_legacy_mask_mode_keeps_masks_member_and_records_runtime_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    source = torch.tensor([[[0, 1], [2, 4]]], dtype=torch.int64)
    candidate = torch.tensor([[[0, 0], [2, 3]]], dtype=torch.int64)
    source_archive = tmp_path / "runtime_members.zip"
    _write_runtime_members(source_archive, mask_bytes=b"source")
    lossy = tmp_path / "masks.crf52.mkv"
    lossy.write_bytes(b"lossy-legacy-mask")

    def fake_decode(data: bytes, member: str, *, max_frames: int | None):
        if data == b"source":
            return source.clone(), {"decoder": "fake-source"}
        if data == b"lossy-legacy-mask":
            return candidate.clone(), {"decoder": "fake-legacy-lossy"}
        raise AssertionError(data)

    monkeypatch.setattr(builder, "_decode_source_masks_from_runtime_member", fake_decode)

    report = builder.build_amr1_repair_candidate(
        source_archive=source_archive,
        lossy_mask_stream=lossy,
        output_archive=tmp_path / "archive.zip",
        policy=builder.ChargedRepairPolicy(select_all_differences=True, boundary_dilation=0),
        lossy_mask_member="masks.mkv",
        lossy_decode_mode="legacy",
        repair_compressor="raw",
    )

    members = _read_members(tmp_path / "archive.zip")
    assert list(members) == ["renderer.bin", "masks.mkv", "alpha4_residual_repair.amr1", "optimized_poses.bin"]
    assert members["masks.mkv"] == b"lossy-legacy-mask"
    assert report["candidate_family"] == "AMR1_charged_residual_over_lossy_legacy_masks"
    assert report["lossy_mask_base"]["decode_mode"] == "legacy"
    assert report["runtime_contract"]["masks_mkv_omitted"] is False
    assert report["runtime_contract"]["legacy_masks_member"] == "masks.mkv"
    assert "inflate_renderer.py applies charged AMR1 repair" in report["runtime_contract"]["inflate_auto_dispatch"]


def test_inflate_renderer_decodes_and_validates_cmg1_raw_stream() -> None:
    builder = _load_builder()
    inflate_renderer = _load_inflate_renderer()
    raw = b"\x1a\x45\xdf\xa3cmg1-test-mask-stream"
    payload = builder.encode_cmg1_payload(source_bytes=raw)

    decoded = inflate_renderer._decode_cmg1_payload(payload)
    assert decoded["frames"] == 600
    assert decoded["height"] == 384
    assert decoded["width"] == 512
    assert decoded["class_count"] == 5
    assert decoded["raw_stream"] == raw
    assert decoded["raw_stream_sha256"] == builder._sha256_bytes(raw)


def test_inflate_renderer_rejects_placeholder_and_tampered_cmg1_payload() -> None:
    builder = _load_builder()
    inflate_renderer = _load_inflate_renderer()

    placeholder = builder.encode_cmg1_payload(source_bytes=None)
    with pytest.raises(ValueError, match="raw bit-identical mode"):
        inflate_renderer._decode_cmg1_payload(placeholder)

    raw = b"\x1a\x45\xdf\xa3cmg1-test-mask-stream"
    payload = bytearray(builder.encode_cmg1_payload(source_bytes=raw))
    payload[-1] ^= 0x01
    with pytest.raises(ValueError, match="source SHA mismatch"):
        inflate_renderer._decode_cmg1_payload(bytes(payload))


def test_inflate_renderer_resolves_cmg1_when_legacy_masks_member_absent(tmp_path: Path) -> None:
    inflate_renderer = _load_inflate_renderer()
    cmg1 = tmp_path / "masks.cmg1"
    cmg1.write_bytes(b"CMG1")

    resolved = inflate_renderer._resolve_mask_path(tmp_path, "masks.mkv")

    assert resolved == cmg1


def test_inflate_renderer_applies_charged_amr1_repair_to_legacy_masks(tmp_path: Path) -> None:
    builder = _load_builder()
    inflate_renderer = _load_inflate_renderer()
    alpha = builder._load_alpha_builder()
    source = torch.tensor([[[0, 1, 1], [2, 2, 4]]], dtype=torch.int64)
    candidate = torch.tensor([[[0, 0, 1], [2, 3, 3]]], dtype=torch.int64)
    candidate._half_frame_only = True  # type: ignore[attr-defined]
    runs = [
        alpha.RepairRun(frame_index=0, y=0, x0=1, length=1, class_id=1),
        alpha.RepairRun(frame_index=0, y=1, x0=1, length=1, class_id=2),
        alpha.RepairRun(frame_index=0, y=1, x0=2, length=1, class_id=4),
    ]
    raw = alpha._encode_repair_payload(
        runs,
        shape=tuple(int(v) for v in source.shape),
        source_mask_sha256=alpha._tensor_u8_sha256(source),
        candidate_mask_sha256=alpha._tensor_u8_sha256(candidate),
        selection_meta={
            "total_residual_pixels": 3,
            "selected_repair_pixels": 3,
            "partial_repair": False,
        },
    )
    (tmp_path / "alpha4_residual_repair.amr1.zlib").write_bytes(zlib.compress(raw, level=9))

    repaired = inflate_renderer._maybe_apply_amr1_repair_from_archive_dir(tmp_path, candidate)

    assert torch.equal(repaired, source)

    with pytest.raises(RuntimeError, match="candidate SHA mismatch"):
        inflate_renderer._maybe_apply_amr1_repair_from_archive_dir(tmp_path, source)


def test_inflate_renderer_archive_mask_loader_does_not_bypass_amr1_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_builder()
    inflate_renderer = _load_inflate_renderer()
    alpha = builder._load_alpha_builder()
    source = torch.tensor([[[0, 1, 1], [2, 2, 4]]], dtype=torch.int64)
    candidate = torch.tensor([[[0, 0, 1], [2, 3, 3]]], dtype=torch.int64)
    raw = alpha._encode_repair_payload(
        [
            alpha.RepairRun(frame_index=0, y=0, x0=1, length=1, class_id=1),
            alpha.RepairRun(frame_index=0, y=1, x0=1, length=1, class_id=2),
            alpha.RepairRun(frame_index=0, y=1, x0=2, length=1, class_id=4),
        ],
        shape=tuple(int(v) for v in source.shape),
        source_mask_sha256=alpha._tensor_u8_sha256(source),
        candidate_mask_sha256=alpha._tensor_u8_sha256(candidate),
        selection_meta={
            "total_residual_pixels": 3,
            "selected_repair_pixels": 3,
            "partial_repair": False,
        },
    )
    (tmp_path / "masks.mkv").write_bytes(b"fake-av1")
    (tmp_path / "alpha4_residual_repair.amr1.zlib").write_bytes(zlib.compress(raw, level=9))
    def _fake_load_masks_from_archive(
        path: Path,
        expected_frames: int = inflate_renderer.NUM_FRAMES,
    ) -> torch.Tensor:
        del path, expected_frames
        loaded = candidate.clone()
        loaded._half_frame_only = True  # type: ignore[attr-defined]
        return loaded

    monkeypatch.setattr(
        inflate_renderer,
        "_load_masks_from_archive",
        _fake_load_masks_from_archive,
    )

    repaired = inflate_renderer._load_archive_masks_with_optional_amr1_repair(
        tmp_path,
        tmp_path / "masks.mkv",
    )

    assert torch.equal(repaired, source)
    assert getattr(repaired, "_half_frame_only", False) is True
