from __future__ import annotations

import importlib.util
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np
import pytest
import torch

from tac.henosis_pr82_transfer import (
    Pr82RandmultiGroup,
    _encode_randmulti_rows,
    encode_randmulti_qrm1,
)


REPO = Path(__file__).resolve().parents[3]
APPLY_PATH = REPO / "submissions" / "robust_current" / "apply_qzs3_postprocess.py"
INFLATE_RENDERER_PATH = REPO / "submissions" / "robust_current" / "inflate_renderer.py"
PR81_PR82_STACK_DIR = REPO / "experiments/results/pr81_pr82_henosis_stack_20260503_codex"


def _load_apply():
    spec = importlib.util.spec_from_file_location("apply_qzs3_postprocess_qrm1_test", APPLY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_inflate_renderer():
    spec = importlib.util.spec_from_file_location(
        "inflate_renderer_qpost_no_router_test",
        INFLATE_RENDERER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qpost_with_randmulti(randmulti: bytes) -> bytes:
    lengths = [0] * 8
    lengths[-1] = len(randmulti)
    return b"QPS1" + struct.pack("<" + "I" * 8, *lengths) + randmulti


def test_joint_generator_without_router_actions_does_not_raise_nameerror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer_module = _load_inflate_renderer()

    class FakeJointGenerator(torch.nn.Module):
        q_faithful = True

        def forward(self, masks_t: torch.Tensor, masks_t1: torch.Tensor, **kwargs: object) -> torch.Tensor:
            del masks_t1, kwargs
            batch, height, width = masks_t.shape
            return torch.zeros((batch, 2, height, width, 3), dtype=torch.float32)

    observed: list[object] = []

    def fake_apply_router(pairs: torch.Tensor, actions: object, *, pair_start: int) -> torch.Tensor:
        del pair_start
        observed.append(actions)
        return pairs

    monkeypatch.setattr(renderer_module, "_apply_pr81_router_actions_to_pairs", fake_apply_router)
    output_path = tmp_path / "frames.raw"
    masks = torch.zeros((2, renderer_module.SEG_H, renderer_module.SEG_W), dtype=torch.long)

    n_written = renderer_module._generate_and_write(
        masks,
        FakeJointGenerator(),
        str(output_path),
        "cpu",
        batch_size=1,
        out_h=8,
        out_w=8,
    )

    assert n_written == 2
    assert observed == [None]
    assert output_path.stat().st_size == 2 * 8 * 8 * 3


def _group_for_spec(apply, spec: tuple[int, int, int, int], *, choice: int = 0) -> Pr82RandmultiGroup:
    group_id = apply.PR82_QRM1_RANDMULTI_SPECS.index(spec)
    rows = np.zeros((spec[3], 600), dtype=np.uint8)
    rows[0, 0] = choice
    return Pr82RandmultiGroup(
        group_id=group_id,
        n_frames=600,
        row_count=spec[3],
        nonzero_entries=int(np.count_nonzero(rows)),
        payload=_encode_randmulti_rows(rows.tolist()),
        extra={"height": spec[0], "width": spec[1], "amplitude": spec[2]},
    )


def test_qrm1_generic_randmulti_decodes_encoder_output(tmp_path: Path) -> None:
    apply = _load_apply()
    group = _group_for_spec(apply, (4, 4, 1, 1), choice=7)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([group])))

    state = apply.read_qpost(qpost_path, torch.device("cpu"))

    assert state.f1_randmulti is not None
    assert len(state.f1_randmulti) == 1
    choices, lh, lw, amp = state.f1_randmulti[0]
    assert (lh, lw, amp) == (4, 4, 1)
    assert tuple(choices.shape) == (1, 600)
    assert int(choices[0, 0].item()) == 7


def _encode_rmb1_from_headerless_rows(raw: bytes) -> bytes:
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
    mask_br = brotli.compress(bytes(mask), quality=11, lgwin=24)
    vals_br = brotli.compress(bytes(values), quality=11, lgwin=24)
    return b"RMB1" + len(mask_br).to_bytes(2, "little") + mask_br + vals_br


def test_rmb1_randmulti_decodes_to_headerless_sparse_rows() -> None:
    apply = _load_apply()
    # First current-runtime headerless group has scount=12; remaining groups
    # are empty one-row groups. This exercises RMB1's bitmask+value recode
    # without relying on public archive files.
    first_group = b"\x02\x00\x02\x05\x07" + b"\x00" * 11
    empty_groups = b"\x00" * 15
    encoded = _encode_rmb1_from_headerless_rows(first_group + empty_groups)

    decoded = apply._decode_randmulti(encoded, torch.device("cpu"))

    assert decoded is not None
    choices, lh, lw, amp = decoded[0]
    assert (lh, lw, amp) == (24, 32, 1)
    assert tuple(choices.shape) == (12, 600)
    assert int(choices[0, 0].item()) == 5
    assert int(choices[0, 3].item()) == 7


def test_qrm1_supported_global_special_branch_applies_to_second_frame(tmp_path: Path) -> None:
    apply = _load_apply()
    group = _group_for_spec(apply, (222, 222, 4, 1), choice=1)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([group])))
    state = apply.read_qpost(qpost_path, torch.device("cpu"))

    raw_path = tmp_path / "0.raw"
    frame0 = np.full((2, 2, 3), 10, dtype=np.uint8)
    frame1 = np.full((2, 2, 3), 10, dtype=np.uint8)
    raw_path.write_bytes(np.stack([frame0, frame1], axis=0).tobytes())

    apply.apply_qpost_to_raw(
        raw_path,
        state,
        height=2,
        width=2,
        batch_pairs=1,
        device=torch.device("cpu"),
    )

    decoded = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8).reshape(2, 2, 2, 3)
    assert np.array_equal(decoded[0], frame0)
    assert np.array_equal(decoded[1], np.full((2, 2, 3), 6, dtype=np.uint8))


def test_qrm1_duplicate_group_id_fails_closed(tmp_path: Path) -> None:
    apply = _load_apply()
    group_id = apply.PR82_QRM1_RANDMULTI_SPECS.index((4, 4, 1, 1))
    raw = b"QRM1" + (2).to_bytes(2, "little")
    raw += int(group_id).to_bytes(2, "little") + b"\x00"
    raw += int(group_id).to_bytes(2, "little") + b"\x00"
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(brotli.compress(raw, quality=11)))

    with pytest.raises(ValueError, match="duplicate randmulti group id"):
        apply.read_qpost(qpost_path, torch.device("cpu"))


def test_qrm1_mask_dependent_special_branch_requires_source_masks(tmp_path: Path) -> None:
    apply = _load_apply()
    group = _group_for_spec(apply, (223, 222, 2, 1), choice=1)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([group])))

    state = apply.read_qpost(qpost_path, torch.device("cpu"))
    assert apply.qpost_requires_source_masks(state) is True

    raw_path = tmp_path / "0.raw"
    frame0 = np.full((2, 2, 3), 10, dtype=np.uint8)
    frame1 = np.full((2, 2, 3), 10, dtype=np.uint8)
    raw_path.write_bytes(np.stack([frame0, frame1], axis=0).tobytes())

    with pytest.raises(ValueError, match="requires source masks"):
        apply.apply_qpost_to_raw(
            raw_path,
            state,
            height=2,
            width=2,
            batch_pairs=1,
            device=torch.device("cpu"),
        )


def test_qrm1_mask_dependent_class_branch_applies_with_source_masks(tmp_path: Path) -> None:
    apply = _load_apply()
    group = _group_for_spec(apply, (223, 222, 2, 1), choice=1)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([group])))
    state = apply.read_qpost(qpost_path, torch.device("cpu"))

    raw_path = tmp_path / "0.raw"
    frame0 = np.full((2, 2, 3), 10, dtype=np.uint8)
    frame1 = np.full((2, 2, 3), 10, dtype=np.uint8)
    raw_path.write_bytes(np.stack([frame0, frame1], axis=0).tobytes())
    source_masks = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)

    apply.apply_qpost_to_raw(
        raw_path,
        state,
        height=2,
        width=2,
        batch_pairs=1,
        device=torch.device("cpu"),
        source_masks=source_masks,
    )

    decoded = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8).reshape(2, 2, 2, 3)
    expected_f1 = frame1.copy()
    expected_f1[source_masks[0].numpy() == 0, 0] = 8
    assert np.array_equal(decoded[0], frame0)
    assert np.array_equal(decoded[1], expected_f1)


def test_qrm1_support_classifier_reports_active_unsupported_groups(tmp_path: Path) -> None:
    apply = _load_apply()
    supported_generic = _group_for_spec(apply, (4, 4, 1, 1), choice=7)
    supported_global = _group_for_spec(apply, (222, 222, 4, 1), choice=1)
    source_mask_boundary = _group_for_spec(apply, (223, 223, 4, 1), choice=2)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(
        _qpost_with_randmulti(
            encode_randmulti_qrm1([supported_generic, supported_global, source_mask_boundary])
        )
    )

    report = apply.classify_qpost_qrm1_support(qpost_path)

    assert report["contract"] == "QRM1_sparse_group_id_stream"
    assert report["dispatchable_qrm1"] is True
    assert supported_generic.group_id in report["supported_group_ids"]
    assert supported_global.group_id in report["supported_group_ids"]
    assert source_mask_boundary.group_id in report["supported_group_ids"]
    assert report["unsupported_group_ids"] == []
    assert report["active_unsupported_group_ids"] == []
    assert report["source_mask_required_group_ids"] == [source_mask_boundary.group_id]
    row = next(item for item in report["group_rows"] if item["group_id"] == source_mask_boundary.group_id)
    assert row["source_mask_required"] is True


def test_qrm1_support_classifier_marks_inactive_source_mask_group_dispatchable_without_requirement(
    tmp_path: Path,
) -> None:
    apply = _load_apply()
    inactive_source_mask = _group_for_spec(apply, (223, 222, 2, 1), choice=0)
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([inactive_source_mask])))

    report = apply.classify_qpost_qrm1_support(qpost_path)

    assert report["dispatchable_qrm1"] is True
    assert report["active_unsupported_group_ids"] == []
    assert report["source_mask_required_group_ids"] == []
    assert inactive_source_mask.group_id in report["supported_group_ids"]
    state = apply.read_qpost(qpost_path, torch.device("cpu"))
    assert state.f1_randmulti is not None
    assert apply.qpost_requires_source_masks(state) is False


def test_qrm1_archive_classifier_reads_candidate_qpost_member(tmp_path: Path) -> None:
    apply = _load_apply()
    group = _group_for_spec(apply, (223, 224, 4, 1), choice=3)
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("p", b"payload")
        zf.writestr("qpost.bin", _qpost_with_randmulti(encode_randmulti_qrm1([group])))

    report = apply.classify_archive_qrm1_support(archive)

    assert report["archive_members"] == ["p", "qpost.bin"]
    assert report["active_unsupported_group_ids"] == []
    assert report["source_mask_required_group_ids"] == [group.group_id]


def test_qrm1_group_id_outside_replay_specs_fails_closed(tmp_path: Path) -> None:
    apply = _load_apply()
    raw = b"QRM1" + (1).to_bytes(2, "little") + (999).to_bytes(2, "little") + b"\x00"
    qpost_path = tmp_path / "qpost.bin"
    qpost_path.write_bytes(_qpost_with_randmulti(brotli.compress(raw, quality=11)))

    with pytest.raises(ValueError, match="outside PR82 replay specs"):
        apply.read_qpost(qpost_path, torch.device("cpu"))


def test_qrm1_archive_classifier_fails_closed_on_duplicate_qpost_member(tmp_path: Path) -> None:
    apply = _load_apply()
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("qpost.bin", b"first")
        zf.writestr("qpost.bin", b"second")

    with pytest.raises(ValueError, match="duplicate qpost.bin"):
        apply.classify_archive_qrm1_support(archive)


@pytest.mark.skipif(
    not (
        PR81_PR82_STACK_DIR
        / "pr81_qma9_pr82_qps1_controls_qrm1_all072"
        / "archive.zip"
    ).exists(),
    reason="PR81+PR82 QRM1 stack candidates are not materialized",
)
def test_generated_pr81_pr82_qrm1_candidates_have_precise_unsupported_group_ids() -> None:
    apply = _load_apply()
    expected_source_mask_required = [62, 63, 64, 65, 66, 67, 68, 70]

    for candidate_id in (
        "pr81_qma9_pr82_qps1_qrm1_all072_randmulti",
        "pr81_qma9_pr82_qps1_controls_qrm1_all072",
    ):
        report = apply.classify_archive_qrm1_support(PR81_PR82_STACK_DIR / candidate_id / "archive.zip")
        assert report["active_unsupported_group_ids"] == []
        assert report["source_mask_required_group_ids"] == expected_source_mask_required
        assert report["dispatchable_qrm1"] is True

    for candidate_id in (
        "pr81_qma9_pr82_qps1_qrm1_supported_subset_randmulti",
        "pr81_qma9_pr82_qps1_controls_qrm1_supported_subset",
    ):
        report = apply.classify_archive_qrm1_support(PR81_PR82_STACK_DIR / candidate_id / "archive.zip")
        assert report["active_unsupported_group_ids"] == []
        assert report["source_mask_required_group_ids"] == expected_source_mask_required
        assert report["dispatchable_qrm1"] is True
        assert 61 in report["supported_group_ids"]
        assert 69 in report["supported_group_ids"]
        assert 71 in report["supported_group_ids"]

    nm2_report = apply.classify_archive_qrm1_support(
        PR81_PR82_STACK_DIR / "pr81_qma9_pr82_qps1_nm2_generic_randmulti" / "archive.zip"
    )
    assert nm2_report["contract"] == "not_qrm1"
    assert nm2_report["unsupported_group_ids"] == []
