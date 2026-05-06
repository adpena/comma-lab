#!/usr/bin/env python3
"""Apply counted qpose/QZS3 postprocess atoms to inflated raw frames.

This is a submission-runtime helper.  It does not load PoseNet or SegNet; it
only applies deterministic, archive-carried image transforms after the normal
renderer has written raw RGB frames.

``qpost.bin`` format:

    b"QPS1" + uint32[8] little-endian lengths + concatenated streams

The eight streams mirror the public PR65/henosis postprocess bundle:
post-code stages, integer frame-0 shifts, 0.5/0.25/0.125px frame-1 shifts,
frame-1 RGB bias, frame-1 regional bias, and multiscale random patterns.
Each stream remains Brotli-compressed inside ``qpost.bin`` and is charged as
archive bytes.

``randmulti`` additionally accepts ``QRM1`` streams emitted by
``tac.henosis_pr82_transfer.encode_randmulti_qrm1``.  QRM1 carries PR82 replay
group ids plus sparse per-pair rows.  This runtime supports generic random
patterns plus the PR82 tile/global and source-mask-conditioned frame-1 bias
branches.  Source-mask-conditioned branches fail closed at apply time unless
the charged archive mask stream can be loaded from the archive directory.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

import brotli
import numpy as np
import torch
import torch.nn.functional as F


MAGIC = b"QPS1"
STREAM_NAMES = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
HEADER = "<" + "I" * len(STREAM_NAMES)
PR82_QRM1_RANDMULTI_SPECS: tuple[tuple[int, int, int, int], ...] = (
    (24, 32, 1, 12), (12, 16, 1, 1), (6, 8, 1, 1), (3, 4, 1, 1),
    (2, 2, 1, 1), (8, 8, 1, 1), (4, 4, 1, 1), (4, 8, 1, 1),
    (2, 4, 1, 1), (2, 8, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1),
    (2, 1, 1, 1), (4, 1, 1, 1), (8, 1, 1, 1), (1, 8, 1, 1),
    (16, 1, 1, 1), (1, 16, 1, 1), (32, 1, 1, 1), (64, 1, 1, 1),
    (256, 1, 1, 1), (1024, 1, 1, 1), (2048, 1, 1, 1), (4096, 1, 1, 1),
    (8192, 1, 1, 1), (8192, 1, 1, 1), (16384, 1, 1, 1), (32768, 1, 1, 1),
    (65536, 1, 1, 1), (131072, 1, 1, 1), (262144, 1, 1, 1),
    (524288, 1, 1, 1), (1048576, 1, 1, 1), (874, 1, 1, 1),
    (874, 1, 1, 1), (2097152, 1, 1, 1), (875, 1, 1, 1),
    (876, 1, 1, 1), (877, 1, 1, 1), (1164, 1, 1, 1),
    (878, 1, 1, 1), (879, 1, 1, 1), (880, 1, 1, 1),
    (881, 1, 1, 1), (882, 1, 1, 1), (512, 2, 1, 1),
    (256, 2, 1, 1), (128, 2, 1, 1), (64, 2, 1, 1),
    (32, 2, 1, 1), (16, 2, 1, 1), (8, 2, 1, 1), (4, 2, 1, 1),
    (4, 4, 1, 1), (8, 4, 1, 1), (16, 4, 1, 1), (32, 4, 1, 1),
    (64, 4, 1, 1), (128, 4, 1, 1), (64, 8, 1, 1),
    (32, 8, 1, 1), (222, 222, 4, 1), (222, 223, 4, 1),
    (223, 222, 2, 1), (223, 223, 4, 1), (223, 221, 4, 1),
    (223, 224, 4, 1), (223, 221, 4, 1), (223, 219, 4, 1),
    (64, 16, 1, 1), (223, 218, 4, 1), (224, 222, 4, 1),
)
QRM1_SUPPORTED_SPECIAL_MAX_CHOICE = {
    (222, 222, 4): 32,
    (222, 223, 4): 40,
    (223, 222, 2): 60,
    (223, 223, 4): 48,
    (223, 221, 4): 120,
    (223, 224, 4): 144,
    (223, 219, 4): 144,
    (223, 218, 4): 144,
    (224, 222, 4): 96,
}
QRM1_SOURCE_MASK_PR82_SPECIAL_RANDMULTI = {
    (222, 223, 4): "class-conditioned all-channel bias",
    (223, 222, 2): "class-conditioned channel bias",
    (223, 223, 4): "boundary all-channel bias",
    (223, 221, 4): "class-conditioned channel bias",
    (223, 224, 4): "boundary/class-boundary channel bias",
    (223, 219, 4): "width-2 boundary channel/class bias",
    (223, 218, 4): "width-3 boundary channel/class bias",
}


class QPostState(NamedTuple):
    postprocess: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
    f0_shift: torch.Tensor | None
    f1_frac: torch.Tensor | None
    f1_frac2: torch.Tensor | None
    f1_frac3: torch.Tensor | None
    f1_bias: torch.Tensor | None
    f1_region: torch.Tensor | None
    f1_randmulti: list[tuple[torch.Tensor, int, int, int]] | None


def _qrm1_support_reason(lh: int, lw: int, amp: int) -> tuple[bool, str | None]:
    return True, None


def classify_qrm1_randmulti_stream(blob: bytes) -> dict[str, object]:
    """Classify a Brotli-compressed QRM1 randmulti stream for runtime support.

    This is a fail-closed preflight helper for PR82-derived candidates.  It
    does not relax runtime validation and does not apply any transform; it
    proves which carried PR82 group ids are locally supported by this raw-frame
    postprocess helper and which require the source mask tensor from the public
    PR82 replay path.
    """

    try:
        raw = brotli.decompress(blob)
    except brotli.error as exc:
        raise ValueError("randmulti stream is not Brotli-decodable") from exc
    if raw[:4] != b"QRM1":
        return {
            "contract": "not_qrm1",
            "dispatchable_qrm1": False,
            "reason": f"randmulti stream is not QRM1: {raw[:4]!r}",
            "supported_group_ids": [],
            "unsupported_group_ids": [],
        }
    if len(raw) < 6:
        raise ValueError("QRM1 randmulti stream is truncated")
    pos = 4
    gcount = int.from_bytes(raw[pos:pos + 2], "little")
    pos += 2
    rows_out: list[dict[str, object]] = []
    seen: set[int] = set()
    supported_group_ids: list[int] = []
    unsupported_group_ids: list[int] = []
    malformed_group_ids: list[int] = []
    source_mask_required_group_ids: list[int] = []
    for _ in range(gcount):
        if pos + 2 > len(raw):
            raise ValueError("QRM1 randmulti group id is truncated")
        group_id = int.from_bytes(raw[pos:pos + 2], "little")
        pos += 2
        if group_id in seen:
            raise ValueError(f"QRM1 duplicate randmulti group id: {group_id}")
        seen.add(group_id)
        if group_id >= len(PR82_QRM1_RANDMULTI_SPECS):
            raise ValueError(f"QRM1 randmulti group id outside PR82 replay specs: {group_id}")
        lh, lw, amp, scount = PR82_QRM1_RANDMULTI_SPECS[group_id]
        rows, pos = _decode_sparse_randmulti_rows(raw, pos, scount)
        nonzero = int(np.count_nonzero(rows))
        max_choice = int(rows.max(initial=0))
        supported, reason = _qrm1_support_reason(lh, lw, amp)
        max_supported = QRM1_SUPPORTED_SPECIAL_MAX_CHOICE.get((int(lh), int(lw), int(amp)))
        if max_supported is not None and max_choice > int(max_supported):
            supported = False
            reason = f"out-of-range choice {max_choice} for max supported choice {max_supported}"
            malformed_group_ids.append(group_id)
        if supported:
            supported_group_ids.append(group_id)
        else:
            unsupported_group_ids.append(group_id)
        if nonzero and (int(lh), int(lw), int(amp)) in QRM1_SOURCE_MASK_PR82_SPECIAL_RANDMULTI:
            source_mask_required_group_ids.append(group_id)
        rows_out.append(
            {
                "amplitude": int(amp),
                "group_id": group_id,
                "height": int(lh),
                "max_choice": max_choice,
                "nonzero_choice_total": nonzero,
                "reason": reason,
                "scount": int(scount),
                "source_mask_required": (
                    (int(lh), int(lw), int(amp)) in QRM1_SOURCE_MASK_PR82_SPECIAL_RANDMULTI
                    and nonzero > 0
                ),
                "supported": supported,
                "width": int(lw),
            }
        )
    if pos != len(raw):
        raise ValueError("QRM1 randmulti stream has trailing bytes")
    active_unsupported = [
        int(row["group_id"])
        for row in rows_out
        if not bool(row["supported"]) and int(row["nonzero_choice_total"]) > 0
    ]
    return {
        "active_unsupported_group_ids": active_unsupported,
        "contract": "QRM1_sparse_group_id_stream",
        "decoded_group_count": len(rows_out),
        "dispatchable_qrm1": not active_unsupported and not malformed_group_ids,
        "group_rows": rows_out,
        "malformed_group_ids": malformed_group_ids,
        "source_mask_required_group_ids": source_mask_required_group_ids,
        "supported_group_ids": supported_group_ids,
        "unsupported_group_ids": unsupported_group_ids,
    }


def classify_qpost_qrm1_support(qpost_path: Path) -> dict[str, object]:
    raw = qpost_path.read_bytes()
    header_size = len(MAGIC) + struct.calcsize(HEADER)
    if len(raw) < header_size or raw[:4] != MAGIC:
        raise ValueError(f"bad qpost magic in {qpost_path}")
    lengths = struct.unpack_from(HEADER, raw, len(MAGIC))
    pos = header_size
    streams: dict[str, bytes] = {}
    for name, n in zip(STREAM_NAMES, lengths):
        end = pos + int(n)
        if end > len(raw):
            raise ValueError(f"qpost stream {name} overruns payload")
        streams[name] = raw[pos:end]
        pos = end
    if pos != len(raw):
        raise ValueError(f"qpost has {len(raw) - pos} trailing bytes")
    if not streams["randmulti"]:
        return {
            "contract": "no_randmulti",
            "dispatchable_qrm1": True,
            "supported_group_ids": [],
            "unsupported_group_ids": [],
        }
    return classify_qrm1_randmulti_stream(streams["randmulti"])


def classify_archive_qrm1_support(archive_path: Path) -> dict[str, object]:
    """Return QRM1 support classification for a candidate archive's qpost.bin."""

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = sorted(info.filename for info in zf.infolist() if not info.is_dir())
        if names.count("qpost.bin") > 1:
            raise ValueError(f"{archive_path} contains duplicate qpost.bin members")
        if "qpost.bin" not in names:
            return {
                "archive_members": names,
                "contract": "no_qpost",
                "dispatchable_qrm1": True,
                "supported_group_ids": [],
                "unsupported_group_ids": [],
            }
        raw = zf.read("qpost.bin")
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".qpost.inspect.tmp")
    try:
        tmp_path.write_bytes(raw)
        report = classify_qpost_qrm1_support(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    report["archive_members"] = names
    return report


def _post_pair_tensor(value, default, device: torch.device) -> torch.Tensor:
    arr = torch.tensor(value if value is not None else default, dtype=torch.float32, device=device)
    if arr.ndim == 1:
        return arr.view(1, 1, 1, 3).expand(2, 1, 1, 3)
    if arr.ndim == 2 and tuple(arr.shape) == (2, 3):
        return arr.view(2, 1, 1, 3)
    raise ValueError(f"Bad postprocess tensor shape: {tuple(arr.shape)}")


def _B(r=0, g=0, b=0):
    return (float(r), float(g), float(b))


def _PB(f0=(0, 0, 0), f1=(0, 0, 0)):
    return (_B(*f0), _B(*f1))


def _post_defs(stage_id: int):
    if stage_id == 1:
        return [
            (None, None),
            (None, _B(2, 0, 0)),
            (None, _B(1, 1, 1)),
            (None, _B(0, 0, -2)),
            (None, _B(0, 0, 2)),
            (None, _B(-1, -1, -1)),
            (None, _B(-2, 0, 0)),
            (None, _B(2, 2, 2)),
            (None, _B(0, -2, 0)),
            (None, _B(0, 2, 0)),
            (_B(1.01, 1.01, 1.01), None),
            (_B(0.99, 0.99, 0.99), None),
        ]
    if stage_id == 2:
        defs = [(None, None)]
        for val in [-4, -3, -2, -1, 1, 2, 3, 4]:
            defs += [
                (None, _B(val, val, val)),
                (None, _B(val, 0, 0)),
                (None, _B(0, val, 0)),
                (None, _B(0, 0, val)),
            ]
        for frame in [0, 1]:
            for val in [-2, -1, 1, 2]:
                for _chan, ci in [("all", -1), ("r", 0), ("g", 1), ("b", 2)]:
                    f0, f1 = [0, 0, 0], [0, 0, 0]
                    target = f0 if frame == 0 else f1
                    if ci < 0:
                        target[0] = target[1] = target[2] = val
                    else:
                        target[ci] = val
                    defs.append((None, _PB(f0, f1)))
        return defs
    if stage_id == 3:
        defs = [(None, None)]
        for r in [-2, -1, 0, 1, 2]:
            for g in [-2, -1, 0, 1, 2]:
                for b in [-2, -1, 0, 1, 2]:
                    if (r, g, b) != (0, 0, 0):
                        defs.append((None, _PB((r, g, b), (0, 0, 0))))
        return defs
    if stage_id == 4:
        defs = [(None, None)]
        for r in [-1, 0, 1]:
            for g in [-1, 0, 1]:
                for b in [-1, 0, 1]:
                    if (r, g, b) != (0, 0, 0):
                        defs.append((None, _PB((r, g, b), (0, 0, 0))))
        return defs
    raise ValueError(f"unknown compact post stage id {stage_id}")


def _post_stage_from_defs(defs, choices_data: bytes, device: torch.device):
    gains = torch.stack([_post_pair_tensor(gain, [1.0, 1.0, 1.0], device) for gain, _ in defs], dim=0)
    biases = torch.stack([_post_pair_tensor(bias, [0.0, 0.0, 0.0], device) for _, bias in defs], dim=0)
    choices = torch.tensor([int(x) for x in choices_data], dtype=torch.long, device=device)
    return gains, biases, choices


def _load_post_codes(blob: bytes, device: torch.device):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    stages = []
    if raw[:4] == b"PCD1":
        pos = 4
        stage_count = raw[pos]
        pos += 1
        for _ in range(stage_count):
            stage_id = raw[pos]
            pos += 1
            n = struct.unpack_from("<H", raw, pos)[0]
            pos += 2
            choices = raw[pos:pos + n]
            pos += n
            stages.append(_post_stage_from_defs(_post_defs(stage_id), choices, device))
    else:
        pairs_per_file = 600
        if len(raw) % pairs_per_file != 0:
            raise ValueError("bad headerless post_codes length")
        stage_count = len(raw) // pairs_per_file
        if stage_count not in (3, 4):
            raise ValueError(f"bad headerless post_codes stage count {stage_count}")
        pos = 0
        for stage_id in range(1, stage_count + 1):
            choices = raw[pos:pos + pairs_per_file]
            pos += pairs_per_file
            stages.append(_post_stage_from_defs(_post_defs(stage_id), choices, device))
    return stages or None


def _decode_dense_or_delta(blob: bytes, *, magic_full: bytes, magic_delta: bytes, default: int, center: int | None, device: torch.device):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == magic_full:
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
    elif magic == magic_delta:
        d = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        arr = np.where(d == 0, default, d - 1).astype(np.int64)
    elif center is not None and magic == b"BV1":
        cnt = int.from_bytes(raw[3:5], "little")
        pos = 5
        arr = np.full(600, center, dtype=np.int64)
        idx = -1
        inds = []
        for _ in range(cnt):
            acc = 0
            sh = 0
            while True:
                by = raw[pos]
                pos += 1
                acc |= (by & 127) << sh
                if by & 128:
                    sh += 7
                else:
                    break
            idx += acc + 1
            inds.append(idx)
        vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos).astype(np.int64)
        for ii, vv in zip(inds, vals):
            arr[ii] = vv - 1
    else:
        raise ValueError(f"bad qpost stream magic {magic!r}")
    return torch.from_numpy(arr).to(device)


def _decode_frac(blob: bytes, device: torch.device):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == b"FH1":
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
    elif magic == b"FV1":
        cnt = int.from_bytes(raw[3:5], "little")
        pos = 5
        arr = np.full(600, 4, dtype=np.int64)
        idx = -1
        inds = []
        for _ in range(cnt):
            acc = 0
            sh = 0
            while True:
                by = raw[pos]
                pos += 1
                acc |= (by & 127) << sh
                if by & 128:
                    sh += 7
                else:
                    break
            idx += acc + 1
            inds.append(idx)
        vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos).astype(np.int64)
        for ii, vv in zip(inds, vals):
            arr[ii] = vv - 1
    else:
        raise ValueError("bad f1 fractional shift payload")
    return torch.from_numpy(arr).to(device)


def _decode_region(blob: bytes, device: torch.device):
    if not blob:
        return None
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == b"RH1":
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
    elif magic == b"RD1":
        d = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        arr = np.where(d == 0, 0, d - 1).astype(np.int64)
    elif magic == b"RV1":
        cnt = int.from_bytes(raw[3:5], "little")
        pos = 5
        arr = np.zeros(600, dtype=np.int64)
        idx = -1
        inds = []
        for _ in range(cnt):
            acc = 0
            sh = 0
            while True:
                by = raw[pos]
                pos += 1
                acc |= (by & 127) << sh
                if by & 128:
                    sh += 7
                else:
                    break
            idx += acc + 1
            inds.append(idx)
        vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos).astype(np.int64)
        for ii, vv in zip(inds, vals):
            arr[ii] = vv - 1
    else:
        raise ValueError("bad f1 region-bias payload")
    return torch.from_numpy(arr).to(device)


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while pos < len(data):
        by = data[pos]
        pos += 1
        value |= (by & 127) << shift
        if by < 128:
            return value, pos
        shift += 7
        if shift > 63:
            break
    raise ValueError("truncated or overlong randmulti VLQ")


def _decode_sparse_randmulti_rows(raw: bytes, pos: int, scount: int) -> tuple[np.ndarray, int]:
    rows = np.zeros((int(scount), 600), dtype=np.uint8)
    for si in range(int(scount)):
        if pos >= len(raw):
            raise ValueError("randmulti stream ended before count byte")
        cnt = int(raw[pos])
        pos += 1
        if cnt == 255:
            if pos + 2 > len(raw):
                raise ValueError("randmulti extended count is truncated")
            cnt = int.from_bytes(raw[pos:pos + 2], "little")
            pos += 2
        idx = -1
        inds = []
        for _ in range(cnt):
            acc, pos = _read_vlq(raw, pos)
            idx += acc + 1
            if idx < 0 or idx >= 600:
                raise ValueError(f"randmulti sparse index out of range: {idx}")
            inds.append(idx)
        end = pos + cnt
        if end > len(raw):
            raise ValueError("randmulti value stream is truncated")
        vals = np.frombuffer(raw, dtype=np.uint8, count=cnt, offset=pos)
        pos = end
        if cnt:
            rows[si, np.array(inds, dtype=np.int64)] = vals
    return rows, pos


def _validate_qrm1_group_choices(rows: np.ndarray, lh: int, lw: int, amp: int) -> None:
    spec_key = (int(lh), int(lw), int(amp))
    supported, reason = _qrm1_support_reason(lh, lw, amp)
    if not supported and int(np.count_nonzero(rows)) > 0:
        raise ValueError(
            "unsupported QRM1 PR82 randmulti branch "
            f"{spec_key}: {reason}"
        )
    max_choice = QRM1_SUPPORTED_SPECIAL_MAX_CHOICE.get(spec_key)
    if max_choice is not None and int(rows.max(initial=0)) > max_choice:
        raise ValueError(f"QRM1 PR82 randmulti branch {spec_key} has out-of-range choice")


def _randmulti_requires_source_masks(state: QPostState) -> bool:
    if state.f1_randmulti is None:
        return False
    for arr_choices, lh, lw, amp in state.f1_randmulti:
        if (int(lh), int(lw), int(amp)) in QRM1_SOURCE_MASK_PR82_SPECIAL_RANDMULTI:
            if bool((arr_choices != 0).any().item()):
                return True
    return False


def qpost_requires_source_masks(state: QPostState) -> bool:
    """Whether this parsed qpost needs source masks for non-noop randmulti."""

    return _randmulti_requires_source_masks(state)


def _decode_qrm1_randmulti(raw: bytes, device: torch.device):
    if len(raw) < 6:
        raise ValueError("QRM1 randmulti stream is truncated")
    pos = 4
    gcount = int.from_bytes(raw[pos:pos + 2], "little")
    pos += 2
    out = []
    seen: set[int] = set()
    for _ in range(gcount):
        if pos + 2 > len(raw):
            raise ValueError("QRM1 randmulti group id is truncated")
        group_id = int.from_bytes(raw[pos:pos + 2], "little")
        pos += 2
        if group_id in seen:
            raise ValueError(f"QRM1 duplicate randmulti group id: {group_id}")
        seen.add(group_id)
        if group_id >= len(PR82_QRM1_RANDMULTI_SPECS):
            raise ValueError(f"QRM1 randmulti group id outside PR82 replay specs: {group_id}")
        lh, lw, amp, scount = PR82_QRM1_RANDMULTI_SPECS[group_id]
        rows, pos = _decode_sparse_randmulti_rows(raw, pos, scount)
        _validate_qrm1_group_choices(rows, lh, lw, amp)
        out.append((torch.from_numpy(rows.astype(np.int64)).to(device), lh, lw, amp))
    if pos != len(raw):
        raise ValueError("QRM1 randmulti stream has trailing bytes")
    return out or None


def _decode_rmb1_randmulti_payload(encoded: bytes) -> bytes:
    """Decode PR92's charged RMB1 bitmask+value randmulti container."""

    if len(encoded) < 6 or encoded[:4] != b"RMB1":
        raise ValueError("bad RMB1 randmulti payload")
    mask_len = int.from_bytes(encoded[4:6], "little")
    mask_br = encoded[6 : 6 + mask_len]
    vals_br = encoded[6 + mask_len :]
    if not mask_br or not vals_br:
        raise ValueError("truncated RMB1 randmulti payload")
    try:
        mask = brotli.decompress(mask_br)
        vals = brotli.decompress(vals_br)
    except brotli.error as exc:
        raise ValueError("RMB1 randmulti substream is not Brotli-decodable") from exc
    if len(mask) % 75:
        raise ValueError("bad RMB1 mask length")

    out = bytearray()
    vals_pos = 0
    for row_start in range(0, len(mask), 75):
        row_mask = mask[row_start : row_start + 75]
        indices: list[int] = []
        row_values: list[int] = []
        for byte_i, byte in enumerate(row_mask):
            for bit in range(8):
                frame_i = byte_i * 8 + bit
                if frame_i >= 600:
                    break
                if byte & (1 << bit):
                    if vals_pos >= len(vals):
                        raise ValueError("truncated RMB1 values")
                    indices.append(frame_i)
                    row_values.append(vals[vals_pos])
                    vals_pos += 1
        count = len(indices)
        if count < 255:
            out.append(count)
        else:
            out.append(255)
            out.extend(count.to_bytes(2, "little"))
        last = -1
        for idx in indices:
            delta = idx - last - 1
            last = idx
            while True:
                byte = delta & 0x7F
                delta >>= 7
                if delta:
                    out.append(byte | 0x80)
                else:
                    out.append(byte)
                    break
        out.extend(row_values)
    if vals_pos != len(vals):
        raise ValueError("unused RMB1 values")
    return bytes(out)


def _decode_randmulti(blob: bytes, device: torch.device):
    if not blob:
        return None
    if blob[:4] == b"RMB1":
        raw = _decode_rmb1_randmulti_payload(blob)
    else:
        try:
            raw = brotli.decompress(blob)
        except brotli.error as exc:
            raise ValueError("randmulti stream is not Brotli-decodable") from exc
    out = []
    if raw[:4] == b"QRM1":
        return _decode_qrm1_randmulti(raw, device)
    if raw[:3] == b"NM1":
        if len(raw) < 4:
            raise ValueError("NM1 randmulti stream is truncated")
        scount = int(raw[3])
        if len(raw) != 4 + scount * 600:
            raise ValueError("NM1 randmulti payload length does not match scount")
        arr = np.frombuffer(raw, dtype=np.uint8, count=scount * 600, offset=4).reshape(scount, 600).astype(np.int64)
        out.append((torch.from_numpy(arr).to(device), 24, 32, 1))
    elif raw[:3] == b"NM2":
        if len(raw) < 4:
            raise ValueError("NM2 randmulti stream is truncated")
        pos = 4
        gcount = int(raw[3])
        for _ in range(gcount):
            if pos + 4 > len(raw):
                raise ValueError("NM2 randmulti group header is truncated")
            lh, lw, amp, scount = int(raw[pos]), int(raw[pos + 1]), int(raw[pos + 2]), int(raw[pos + 3])
            pos += 4
            if pos + scount * 600 > len(raw):
                raise ValueError("NM2 randmulti dense rows are truncated")
            arr = np.frombuffer(raw, dtype=np.uint8, count=scount * 600, offset=pos).reshape(scount, 600).astype(np.int64)
            pos += scount * 600
            out.append((torch.from_numpy(arr).to(device), lh, lw, amp))
        if pos != len(raw):
            raise ValueError("NM2 randmulti stream has trailing bytes")
    else:
        specs = [
            (24, 32, 1, 12), (12, 16, 1, 1), (6, 8, 1, 1), (3, 4, 1, 1),
            (2, 2, 1, 1), (8, 8, 1, 1), (4, 4, 1, 1), (4, 8, 1, 1),
            (2, 4, 1, 1), (2, 8, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1),
            (2, 1, 1, 1), (4, 1, 1, 1), (8, 1, 1, 1), (1, 8, 1, 1),
        ]
        pos = 0
        for lh, lw, amp, scount in specs:
            rows, pos = _decode_sparse_randmulti_rows(raw, pos, scount)
            out.append((torch.from_numpy(rows.astype(np.int64)).to(device), lh, lw, amp))
        if pos != len(raw):
            raise ValueError("bad headerless f1 randmulti payload")
    return out or None


def read_qpost(path: Path, device: torch.device) -> QPostState:
    raw = path.read_bytes()
    header_size = len(MAGIC) + struct.calcsize(HEADER)
    if len(raw) < header_size or raw[:4] != MAGIC:
        raise ValueError(f"bad qpost magic in {path}")
    lengths = struct.unpack_from(HEADER, raw, len(MAGIC))
    pos = header_size
    streams: dict[str, bytes] = {}
    for name, n in zip(STREAM_NAMES, lengths):
        if n < 0:
            raise ValueError(f"negative qpost stream length for {name}")
        end = pos + n
        if end > len(raw):
            raise ValueError(f"qpost stream {name} overruns payload")
        streams[name] = raw[pos:end]
        pos = end
    if pos != len(raw):
        raise ValueError(f"qpost has {len(raw) - pos} trailing bytes")

    return QPostState(
        postprocess=_load_post_codes(streams["post"], device),
        f0_shift=_decode_dense_or_delta(streams["shift"], magic_full=b"SH4", magic_delta=b"SD4", default=40, center=None, device=device),
        f1_frac=_decode_frac(streams["frac"], device),
        f1_frac2=_decode_dense_or_delta(streams["frac2"], magic_full=b"FH2", magic_delta=b"FD2", default=4, center=None, device=device),
        f1_frac3=_decode_dense_or_delta(streams["frac3"], magic_full=b"FH3", magic_delta=b"FD3", default=4, center=None, device=device),
        f1_bias=_decode_dense_or_delta(streams["bias"], magic_full=b"BH1", magic_delta=b"BD1", default=13, center=13, device=device),
        f1_region=_decode_region(streams["region"], device),
        f1_randmulti=_decode_randmulti(streams["randmulti"], device),
    )


def _shift_grid(cache: dict[tuple[float, int, int, int], torch.Tensor], ch: int, step: float, h: int, w: int, device: torch.device) -> torch.Tensor:
    key = (step, ch, h, w)
    if key not in cache:
        dy = (ch // 3 - 1) * step
        dx = (ch % 3 - 1) * step
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        gx = ((xx - dx) + 0.5) * 2.0 / w - 1.0
        gy = ((yy - dy) + 0.5) * 2.0 / h - 1.0
        cache[key] = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    return cache[key]


def apply_qpost_batch(
    batch_hwc: torch.Tensor,
    *,
    pair_start: int,
    state: QPostState,
    grid_cache: dict[tuple[float, int, int, int], torch.Tensor],
    randpat_cache: dict[tuple[int, int, int, int], torch.Tensor],
    source_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz, _two, out_h, out_w, _c = batch_hwc.shape
    device = batch_hwc.device
    if state.postprocess is not None:
        for gains, biases, choices in state.postprocess:
            idx = choices[pair_start:pair_start + bsz]
            if idx.numel() < bsz:
                idx = F.pad(idx, (0, bsz - idx.numel()))
            idx = idx.clamp(0, gains.shape[0] - 1)
            batch_hwc = (batch_hwc * gains[idx] + biases[idx]).clamp(0, 255).round()

    if state.f0_shift is not None:
        chs = state.f0_shift[pair_start:pair_start + bsz]
        for bi in range(bsz):
            ch = int(chs[bi].item()) if bi < chs.numel() else 40
            if ch != 40:
                dy = ch // 9 - 4
                dx = ch % 9 - 4
                img = batch_hwc[bi, 0].permute(2, 0, 1).unsqueeze(0)
                left, right = max(dx, 0), max(-dx, 0)
                top, bottom = max(dy, 0), max(-dy, 0)
                imgp = F.pad(img, (left, right, top, bottom), mode="replicate")
                y0, x0 = bottom, right
                batch_hwc[bi, 0] = imgp[0, :, y0:y0 + out_h, x0:x0 + out_w].permute(1, 2, 0)

    for choices, step, default in (
        (state.f1_frac, 0.5, 4),
        (state.f1_frac2, 0.25, 4),
        (state.f1_frac3, 0.125, 4),
    ):
        if choices is None:
            continue
        chs = choices[pair_start:pair_start + bsz]
        for bi in range(bsz):
            ch = int(chs[bi].item()) if bi < chs.numel() else default
            if ch != default:
                img = batch_hwc[bi, 0].permute(2, 0, 1).unsqueeze(0).float()
                img = F.grid_sample(img, _shift_grid(grid_cache, ch, step, out_h, out_w, device), mode="bilinear", padding_mode="border", align_corners=False)
                batch_hwc[bi, 0] = img[0].clamp(0, 255).round().permute(1, 2, 0)

    if state.f1_bias is not None:
        chs = state.f1_bias[pair_start:pair_start + bsz]
        for bi in range(bsz):
            ch = int(chs[bi].item()) if bi < chs.numel() else 13
            if ch != 13:
                br = ch // 9 - 1
                bg = (ch // 3) % 3 - 1
                bb = ch % 3 - 1
                bias = torch.tensor([br, bg, bb], device=device, dtype=batch_hwc.dtype)
                batch_hwc[bi, 0] = (batch_hwc[bi, 0] + bias).clamp(0, 255).round()

    if state.f1_region is not None:
        chs = state.f1_region[pair_start:pair_start + bsz]
        yy_idx = torch.arange(out_h, device=device).view(out_h, 1).expand(out_h, out_w)
        xx_idx = torch.arange(out_w, device=device).view(1, out_w).expand(out_h, out_w)
        for bi in range(bsz):
            ch = int(chs[bi].item()) if bi < chs.numel() else 0
            if ch != 0:
                j = ch - 1
                val_list = [-2, -1, 1, 2]
                val = float(val_list[j % 4])
                j //= 4
                ci = j % 4
                j //= 4
                mi = j
                if mi == 0:
                    mask = yy_idx < out_h // 2
                elif mi == 1:
                    mask = yy_idx >= out_h // 2
                elif mi == 2:
                    mask = xx_idx < out_w // 2
                elif mi == 3:
                    mask = xx_idx >= out_w // 2
                elif mi == 4:
                    mask = (yy_idx >= out_h // 3) & (yy_idx < 2 * out_h // 3)
                else:
                    mask = (xx_idx >= out_w // 3) & (xx_idx < 2 * out_w // 3)
                if ci == 0:
                    batch_hwc[bi, 0][mask, :] = (batch_hwc[bi, 0][mask, :] + val).clamp(0, 255).round()
                else:
                    cc = ci - 1
                    batch_hwc[bi, 0][mask, cc] = (batch_hwc[bi, 0][mask, cc] + val).clamp(0, 255).round()

    if state.f1_randmulti is not None:
        source_mask_low = None
        source_mask_up = None
        source_boundary_up = None

        def require_source_masks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            nonlocal source_mask_low, source_mask_up, source_boundary_up
            if source_masks is None:
                raise ValueError("QRM1 PR82 source-mask-conditioned randmulti requires source masks")
            if source_masks.ndim != 3:
                raise ValueError(f"source masks must have shape (pairs,H,W), got {tuple(source_masks.shape)}")
            if source_masks.shape[0] < bsz:
                raise ValueError(f"source masks batch has {source_masks.shape[0]} pairs, need {bsz}")
            if source_mask_low is None:
                source_mask_low = source_masks[:bsz].to(device=device, dtype=torch.long)
                if int(source_mask_low.min().item()) < 0 or int(source_mask_low.max().item()) > 4:
                    raise ValueError("source masks must contain class ids in [0,4]")
                mx = F.max_pool2d(source_mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                mn = -F.max_pool2d(-source_mask_low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                boundary_low = (mx != mn).float().unsqueeze(1)
                source_mask_up = F.interpolate(
                    source_mask_low.float().unsqueeze(1),
                    size=(out_h, out_w),
                    mode="nearest",
                )[:, 0].to(torch.uint8)
                source_boundary_up = F.interpolate(
                    boundary_low,
                    size=(out_h, out_w),
                    mode="nearest",
                )[:, 0].to(torch.bool)
            assert source_mask_up is not None
            assert source_boundary_up is not None
            return source_mask_low, source_mask_up, source_boundary_up

        def source_boundary_masks(width: int) -> tuple[torch.Tensor, torch.Tensor]:
            low, mask_up, boundary_up = require_source_masks()
            if width == 1:
                return mask_up, boundary_up
            boundary_low = (F.max_pool2d(
                (
                    F.max_pool2d(low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                    != -F.max_pool2d(-low.float().unsqueeze(1), 3, 1, 1)[:, 0]
                ).float().unsqueeze(1),
                3,
                1,
                1,
            ))
            if width == 3:
                boundary_low = F.max_pool2d(boundary_low, 3, 1, 1)
            if width not in (2, 3):
                raise ValueError(f"unsupported source boundary width: {width}")
            boundary = F.interpolate(boundary_low, size=(out_h, out_w), mode="nearest")[:, 0].to(torch.bool)
            return mask_up, boundary

        def channel_bias(val: int, typ: int) -> torch.Tensor:
            if typ == 0:
                return torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
            if typ == 1:
                return torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
            if typ == 2:
                return torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
            raise ValueError(f"unsupported channel bias type: {typ}")

        for arr_choices, lh, lw, amp in state.f1_randmulti:
            if (lh, lw, amp) == (224, 222, 4):
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                yy_idx = torch.arange(out_h, device=device).view(out_h, 1).expand(out_h, out_w)
                xx_idx = torch.arange(out_w, device=device).view(1, out_w).expand(out_h, out_w)
                tile_masks = [
                    ((yy_idx < out_h // 2) & (xx_idx < out_w // 2)),
                    ((yy_idx < out_h // 2) & (xx_idx >= out_w // 2)),
                    ((yy_idx >= out_h // 2) & (xx_idx < out_w // 2)),
                    ((yy_idx >= out_h // 2) & (xx_idx >= out_w // 2)),
                ]
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    tile = jj // 24
                    rem = jj % 24
                    val = vals[rem // 3]
                    typ = rem % 3
                    if typ == 0:
                        bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                    elif typ == 1:
                        bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                    else:
                        bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                    mm = tile_masks[tile].unsqueeze(-1)
                    batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (222, 222, 4):
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    val = vals[(ch - 1) // 4]
                    typ = (ch - 1) % 4
                    if typ == 0:
                        bias = torch.tensor([val, val, val], device=device, dtype=batch_hwc.dtype)
                    elif typ == 1:
                        bias = torch.tensor([val, 0, 0], device=device, dtype=batch_hwc.dtype)
                    elif typ == 2:
                        bias = torch.tensor([0, val, 0], device=device, dtype=batch_hwc.dtype)
                    else:
                        bias = torch.tensor([0, 0, val], device=device, dtype=batch_hwc.dtype)
                    batch_hwc[bi, 1] = (batch_hwc[bi, 1] + bias).clamp(0, 255).round()
                continue
            if (lh, lw, amp) == (223, 223, 4):
                _mask_low, mask_up, boundary = require_source_masks()
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    if jj < 8:
                        mm = boundary[bi]
                        val = vals[jj]
                    else:
                        jj -= 8
                        cls = jj // 8
                        val = vals[jj % 8]
                        mm = boundary[bi] & (mask_up[bi] == cls)
                    bias = torch.tensor([val, val, val], device=device, dtype=batch_hwc.dtype)
                    batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (223, 222, 2):
                _mask_low, mask_up, _boundary = require_source_masks()
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-2, -1, 1, 2]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    cls = jj // 12
                    rem = jj % 12
                    val = vals[rem // 3]
                    bias = channel_bias(val, rem % 3)
                    mm = (mask_up[bi] == cls).unsqueeze(-1)
                    batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (223, 224, 4):
                _mask_low, mask_up, boundary = require_source_masks()
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    if jj < 24:
                        mm = boundary[bi]
                        val = vals[jj // 3]
                        typ = jj % 3
                    else:
                        jj -= 24
                        cls = jj // 24
                        rem = jj % 24
                        val = vals[rem // 3]
                        typ = rem % 3
                        mm = boundary[bi] & (mask_up[bi] == cls)
                    bias = channel_bias(val, typ)
                    batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (223, 219, 4):
                mask_up, boundary = source_boundary_masks(2)
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    if jj < 24:
                        mm = boundary[bi]
                        val = vals[jj // 3]
                        typ = jj % 3
                    else:
                        jj -= 24
                        cls = jj // 24
                        rem = jj % 24
                        val = vals[rem // 3]
                        typ = rem % 3
                        mm = boundary[bi] & (mask_up[bi] == cls)
                    bias = channel_bias(val, typ)
                    batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (223, 218, 4):
                mask_up, boundary = source_boundary_masks(3)
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    if jj < 24:
                        mm = boundary[bi]
                        val = vals[jj // 3]
                        typ = jj % 3
                    else:
                        jj -= 24
                        cls = jj // 24
                        rem = jj % 24
                        val = vals[rem // 3]
                        typ = rem % 3
                        mm = boundary[bi] & (mask_up[bi] == cls)
                    bias = channel_bias(val, typ)
                    batch_hwc[bi, 1] = torch.where(mm.unsqueeze(-1), (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (223, 221, 4):
                _mask_low, mask_up, _boundary = require_source_masks()
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    jj = ch - 1
                    cls = jj // 24
                    rem = jj % 24
                    val = vals[rem // 3]
                    bias = channel_bias(val, rem % 3)
                    mm = (mask_up[bi] == cls).unsqueeze(-1)
                    batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            if (lh, lw, amp) == (222, 223, 4):
                _mask_low, mask_up, _boundary = require_source_masks()
                chs = arr_choices[0, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                vals = [-4, -3, -2, -1, 1, 2, 3, 4]
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        continue
                    cls = (ch - 1) // 8
                    val = vals[(ch - 1) % 8]
                    mm = (mask_up[bi] == cls).unsqueeze(-1)
                    bias = torch.tensor([val, val, val], device=device, dtype=batch_hwc.dtype)
                    batch_hwc[bi, 1] = torch.where(mm, (batch_hwc[bi, 1] + bias).clamp(0, 255).round(), batch_hwc[bi, 1])
                continue
            zero_low = None
            for st in range(arr_choices.shape[0]):
                chs = arr_choices[st, pair_start:pair_start + bsz]
                if chs.numel() < bsz:
                    chs = F.pad(chs, (0, bsz - chs.numel()))
                if bool((chs == 0).all().item()):
                    continue
                lows = []
                for bi in range(bsz):
                    ch = int(chs[bi].item())
                    if ch == 0:
                        if zero_low is None:
                            zero_low = torch.zeros((3, lh, lw), dtype=torch.float32, device=device)
                        lows.append(zero_low)
                    else:
                        key = (lh, lw, amp, ch)
                        if key not in randpat_cache:
                            rng = np.random.default_rng(1000 + ch)
                            arr = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(3, lh, lw)).astype(np.float32) * float(amp)
                            randpat_cache[key] = torch.from_numpy(arr).to(device)
                        lows.append(randpat_cache[key])
                pat = F.interpolate(torch.stack(lows, dim=0), size=(out_h, out_w), mode="nearest").permute(0, 2, 3, 1).contiguous()
                batch_hwc[:, 0] = (batch_hwc[:, 0] + pat).clamp(0, 255).round()

    return batch_hwc


def _load_inflate_renderer_module():
    module_path = Path(__file__).with_name("inflate_renderer.py")
    spec = importlib.util.spec_from_file_location("robust_current_inflate_renderer_for_qpost", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load inflate_renderer.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_source_pair_masks_from_archive(
    archive_dir: Path,
    *,
    expected_pairs: int,
) -> torch.Tensor:
    """Load the charged source mask tensor used by PR82 mask-aware randmulti."""

    runtime = _load_inflate_renderer_module()
    mask_path = runtime._resolve_mask_path(archive_dir, "masks.mkv")
    masks = runtime._load_archive_masks_with_optional_amr1_repair(archive_dir, mask_path)
    if not isinstance(masks, torch.Tensor) or masks.ndim != 3:
        raise ValueError(f"archive masks must decode to (frames,H,W), got {type(masks).__name__}")
    if int(masks.shape[0]) == int(expected_pairs):
        pair_masks = masks
    elif int(masks.shape[0]) == int(expected_pairs) * 2:
        pair_masks = masks[1::2]
    else:
        raise ValueError(
            f"archive source masks have {masks.shape[0]} frames; expected "
            f"{expected_pairs} pair masks or {expected_pairs * 2} full-frame masks"
        )
    if pair_masks.numel() and (int(pair_masks.min().item()) < 0 or int(pair_masks.max().item()) > 4):
        raise ValueError("archive source masks must contain class ids in [0,4]")
    return pair_masks.cpu().long().contiguous()


def apply_qpost_to_raw(
    raw_path: Path,
    state: QPostState,
    *,
    height: int,
    width: int,
    batch_pairs: int,
    device: torch.device,
    source_masks: torch.Tensor | None = None,
) -> dict[str, int | str]:
    frame_bytes = height * width * 3
    size = raw_path.stat().st_size
    if size % (2 * frame_bytes) != 0:
        raise ValueError(f"{raw_path} size {size} is not an even number of RGB frames for {width}x{height}")
    pair_count = size // (2 * frame_bytes)
    if source_masks is not None and source_masks.shape[0] < pair_count:
        raise ValueError(f"source masks have {source_masks.shape[0]} pairs, raw file has {pair_count}")
    tmp_path = raw_path.with_suffix(raw_path.suffix + ".qpost.tmp")
    grid_cache: dict[tuple[float, int, int, int], torch.Tensor] = {}
    randpat_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

    with raw_path.open("rb") as src, tmp_path.open("wb") as dst:
        for pair_start in range(0, pair_count, batch_pairs):
            n = min(batch_pairs, pair_count - pair_start)
            buf = src.read(n * 2 * frame_bytes)
            arr = np.frombuffer(buf, dtype=np.uint8).copy().reshape(n, 2, height, width, 3)
            batch = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
            batch = apply_qpost_batch(
                batch,
                pair_start=pair_start,
                state=state,
                grid_cache=grid_cache,
                randpat_cache=randpat_cache,
                source_masks=source_masks[pair_start:pair_start + n].to(device) if source_masks is not None else None,
            )
            out = batch.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            dst.write(out.tobytes())
    os.replace(tmp_path, raw_path)
    return {"raw_path": str(raw_path), "bytes": size, "pair_count": int(pair_count)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qpost", type=Path)
    parser.add_argument("inflated_dir", type=Path)
    parser.add_argument("video_names_file", type=Path)
    parser.add_argument("--height", type=int, default=874)
    parser.add_argument("--width", type=int, default=1164)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    state = read_qpost(args.qpost, device)
    source_pair_masks = None
    source_pair_cursor = 0
    if qpost_requires_source_masks(state):
        raw_pair_total = 0
        for line in args.video_names_file.read_text().splitlines():
            if not line.strip():
                continue
            raw_path = args.inflated_dir / (Path(line.strip()).stem + ".raw")
            if not raw_path.exists():
                raise FileNotFoundError(f"raw output missing before qpost apply: {raw_path}")
            frame_bytes = args.height * args.width * 3
            raw_size = raw_path.stat().st_size
            if raw_size % (2 * frame_bytes) != 0:
                raise ValueError(f"{raw_path} size {raw_size} is not an even number of RGB frames")
            raw_pair_total += raw_size // (2 * frame_bytes)
        source_pair_masks = load_source_pair_masks_from_archive(args.qpost.parent, expected_pairs=raw_pair_total)
    records = []
    for line in args.video_names_file.read_text().splitlines():
        if not line.strip():
            continue
        raw_path = args.inflated_dir / (Path(line.strip()).stem + ".raw")
        if not raw_path.exists():
            raise FileNotFoundError(f"raw output missing before qpost apply: {raw_path}")
        frame_bytes = args.height * args.width * 3
        pair_count = raw_path.stat().st_size // (2 * frame_bytes)
        file_source_masks = None
        if source_pair_masks is not None:
            file_source_masks = source_pair_masks[source_pair_cursor:source_pair_cursor + pair_count]
            source_pair_cursor += pair_count
        records.append(
            apply_qpost_to_raw(
                raw_path,
                state,
                height=args.height,
                width=args.width,
                batch_pairs=args.batch_pairs,
                device=device,
                source_masks=file_source_masks,
            )
        )
    print(json.dumps({"qpost": str(args.qpost), "device": str(device), "records": records}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
