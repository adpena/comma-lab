from __future__ import annotations

import zipfile
from pathlib import Path

import brotli
import pytest

import tac.pr79_segaction_payload as pr79_mod
from tac.pr79_segaction_payload import (
    parse_pr79_archive,
    parse_pr79_payload_bytes,
    write_pr79_single_member_archive,
)


def _br(raw: bytes) -> bytes:
    return brotli.compress(raw, quality=11)


def _parts() -> tuple[bytes, bytes, bytes, bytes]:
    mask = _br(b"\x12\x00mask-obu")
    model = _br(b"QZS3" + b"\x20\x00model")
    actions = _br(b"SG2\x00\x01\x00")
    pose = _br(b"QP1\x00\x00")
    return mask, model, actions, pose


def test_parse_p3_payload_validates_all_slices() -> None:
    mask, model, actions, pose = _parts()
    payload = (
        b"P3"
        + len(mask).to_bytes(4, "little")
        + len(model).to_bytes(2, "little")
        + len(actions).to_bytes(2, "little")
        + mask
        + model
        + actions
        + pose
    )

    parsed = parse_pr79_payload_bytes(payload)

    assert parsed.header == "P3"
    assert parsed.mask.compressed_bytes == len(mask)
    assert parsed.model.raw_magic == b"QZS3"
    assert parsed.actions.raw_magic == b"SG2"
    assert parsed.pose.raw_magic == b"QP1"
    assert parsed.summary()["slice_contract"]["actions_bytes"] == len(actions)


def test_parse_legacy_payload_is_not_length_only(monkeypatch: pytest.MonkeyPatch) -> None:
    mask, model, actions, pose = _parts()
    monkeypatch.setattr(pr79_mod, "PR79_MASK_BR_BYTES", len(mask))
    monkeypatch.setattr(pr79_mod, "PR79_MODEL_BR_CANDIDATE_BYTES", (len(model),))
    monkeypatch.setattr(pr79_mod, "PR79_POSE_BR_CANDIDATE_BYTES", (len(pose),))
    payload = mask + model + actions + pose

    parsed = parse_pr79_payload_bytes(payload)

    assert parsed.header == "legacy"
    assert parsed.model.compressed_bytes == len(model)
    assert parsed.actions.compressed == actions
    assert parsed.pose.compressed_bytes == len(pose)


def test_parse_rejects_same_lengths_with_bad_model_magic(monkeypatch: pytest.MonkeyPatch) -> None:
    mask, _model, actions, pose = _parts()
    model = _br(b"BAD!" + b"model")
    monkeypatch.setattr(pr79_mod, "PR79_MASK_BR_BYTES", len(mask))
    monkeypatch.setattr(pr79_mod, "PR79_MODEL_BR_CANDIDATE_BYTES", (len(model),))
    monkeypatch.setattr(pr79_mod, "PR79_POSE_BR_CANDIDATE_BYTES", (len(pose),))

    with pytest.raises(ValueError, match="no valid PR79 legacy split"):
        parse_pr79_payload_bytes(mask + model + actions + pose)


def test_replace_actions_preserves_p3_header_and_revalidates() -> None:
    mask, model, actions, pose = _parts()
    payload = (
        b"P3"
        + len(mask).to_bytes(4, "little")
        + len(model).to_bytes(2, "little")
        + len(actions).to_bytes(2, "little")
        + mask
        + model
        + actions
        + pose
    )
    parsed = parse_pr79_payload_bytes(payload)
    replacement = _br(b"TG1" + (32).to_bytes(2, "little") + b"\x00\x01\x02\x03")

    changed = parsed.replace_actions(replacement)
    reparsed = parse_pr79_payload_bytes(changed)

    assert reparsed.header == "P3"
    assert reparsed.actions.compressed == replacement
    assert changed.startswith(b"P3")


def test_write_single_member_archive_is_deterministic_and_parseable(tmp_path: Path) -> None:
    mask, model, actions, pose = _parts()
    payload = (
        b"P3"
        + len(mask).to_bytes(4, "little")
        + len(model).to_bytes(2, "little")
        + len(actions).to_bytes(2, "little")
        + mask
        + model
        + actions
        + pose
    )
    out = tmp_path / "archive.zip"

    meta = write_pr79_single_member_archive(out, payload)

    assert meta["archive_bytes"] == out.stat().st_size
    with zipfile.ZipFile(out) as zf:
        info = zf.getinfo("p")
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.read("p") == payload
    assert parse_pr79_archive(out).payload == payload
