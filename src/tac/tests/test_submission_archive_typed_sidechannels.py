from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from tac.submission_archive import (
    RENDERER_COMPACT_MANIFEST,
    TYPED_SIDECHANNEL_CONTRACT_MEMBER,
    TypedSidechannelMember,
    build_submission_archive,
    validate_archive,
    write_deterministic_zip_member,
)


def _base_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"r" * 12000)
    masks = tmp_path / "masks.mkv"
    masks.write_bytes(b"m" * 12000)
    poses = tmp_path / "optimized_poses.bin"
    poses.write_bytes(b"p" * 7200)
    return renderer, masks, poses


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_archive_has_no_typed_sidechannel_member_and_is_stable(tmp_path: Path) -> None:
    renderer, masks, poses = _base_inputs(tmp_path)
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"

    result_a = build_submission_archive(
        archive_a,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
    )
    build_submission_archive(
        archive_b,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    assert result_a.valid
    assert result_a.dispatch_ready
    assert _sha256(archive_a) == _sha256(archive_b)
    with zipfile.ZipFile(archive_a) as zf:
        assert zf.namelist() == ["renderer.bin", "masks.mkv", "optimized_poses.bin"]


def test_typed_score_affecting_sidechannel_is_byte_closed_but_not_dispatch_ready(
    tmp_path: Path,
) -> None:
    renderer, masks, poses = _base_inputs(tmp_path)
    categorical = tmp_path / "categorical_payload.bin"
    categorical.write_bytes(b"QMA9 payload bytes")
    archive = tmp_path / "with_sidechannel.zip"

    result = build_submission_archive(
        archive,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        typed_sidechannels=[
            TypedSidechannelMember(
                member_name="categorical_payload.bin",
                source_path=categorical,
                kind="categorical_payload",
                score_affecting=True,
                consumed_by_runtime=False,
            )
        ],
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    assert result.valid
    assert not result.dispatch_ready
    assert (
        "categorical_payload_score_affecting_member_not_consumed_by_runtime"
        in result.dispatch_blockers
    )
    contract = result.typed_sidechannel_contract
    assert contract is not None
    assert contract["score_claim"] is False
    assert contract["dispatch_ready"] is False
    assert contract["members"][0]["member_name"] == "categorical_payload.bin"
    assert contract["members"][0]["sha256"] == _sha256(categorical)

    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == [
            "renderer.bin",
            "masks.mkv",
            "optimized_poses.bin",
            "categorical_payload.bin",
            TYPED_SIDECHANNEL_CONTRACT_MEMBER,
        ]


def test_uncontracted_sidechannel_member_fails_strict_validation(tmp_path: Path) -> None:
    renderer, masks, poses = _base_inputs(tmp_path)
    archive = tmp_path / "uncontracted.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        write_deterministic_zip_member(zf, "renderer.bin", renderer.read_bytes())
        write_deterministic_zip_member(zf, "masks.mkv", masks.read_bytes())
        write_deterministic_zip_member(zf, "optimized_poses.bin", poses.read_bytes())
        write_deterministic_zip_member(zf, "jcsp.bin", b"JCSP preview")

    result = validate_archive(archive, RENDERER_COMPACT_MANIFEST, strict=True)

    assert not result.valid
    assert "jcsp.bin" in result.files_unexpected
    assert any(error == "Unexpected file in archive: jcsp.bin" for error in result.errors)


def test_typed_consumption_claim_without_proof_fails_closed(tmp_path: Path) -> None:
    renderer, masks, poses = _base_inputs(tmp_path)
    sjkl = tmp_path / "sjkl.bin"
    sjkl.write_bytes(b"SJKL bytes")

    try:
        build_submission_archive(
            tmp_path / "sjkl.zip",
            renderer_bin=renderer,
            masks_mkv=masks,
            optimized_poses_bin=poses,
            typed_sidechannels=[
                {
                    "member_name": "sjkl.bin",
                    "source_path": sjkl,
                    "kind": "sjkl_residual",
                    "score_affecting": True,
                    "consumed_by_runtime": True,
                    "runtime_consumer": "submissions/robust_current/inflate_renderer.py",
                }
            ],
            manifest=RENDERER_COMPACT_MANIFEST,
        )
    except ValueError as exc:
        assert "runtime_consumption_proof_sha256" in str(exc)
    else:  # pragma: no cover - explicit failure message for clarity
        raise AssertionError("missing runtime consumption proof must fail validation")


def test_preflight_fails_non_dispatch_ready_typed_sidechannel(tmp_path: Path) -> None:
    from tac.preflight import PreflightError, preflight_check

    renderer, masks, poses = _base_inputs(tmp_path)
    lfv1 = tmp_path / "lapose_foveation_tuples.lfv1"
    lfv1.write_bytes(b"LFV1 bytes")
    archive = tmp_path / "lfv1.zip"
    build_submission_archive(
        archive,
        renderer_bin=renderer,
        masks_mkv=masks,
        optimized_poses_bin=poses,
        typed_sidechannels=[
            {
                "member_name": "lapose_foveation_tuples.lfv1",
                "source_path": lfv1,
                "kind": "lapose_lfv1",
                "score_affecting": True,
                "consumed_by_runtime": False,
            }
        ],
        manifest=RENDERER_COMPACT_MANIFEST,
    )

    try:
        preflight_check(archive_path=archive, verbose=False)
    except PreflightError as exc:
        assert "archive dispatch blocker" in str(exc)
        assert "lapose_lfv1_score_affecting_member_not_consumed_by_runtime" in str(exc)
    else:  # pragma: no cover - explicit failure message for clarity
        raise AssertionError("preflight must fail non-dispatch-ready typed sidechannels")
