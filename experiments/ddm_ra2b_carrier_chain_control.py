#!/usr/bin/env python3
"""RA2B: the CORRECTED carrier→frame_0 chain, validated by a byte-identity control.

AMENDMENT 3 of the ra2 charter established that ra2a read the archive at the wrong
layer (it called the CPR1 receiver's ``split_payload`` on an F26-WRAPPED member, which
correctly refused). This tool implements the RIGHT chain, taken verbatim from the
shipped ``runtime/f26_inflate.py::inflate_archive``:

    parts             = read_residual_archive(archive)
    carrier_blob, _   = split_frame0_selector_carrier(parts.carrier_blob)   # <- ra1 SKIPS
    canonical_carrier = materialize_cpr1(carrier_blob, renderer)
    semantic_pose     = pack("<II", 40252, len(cc)) + bytes(40252) + cc
    _, basis, coeff   = renderer.unpack_semantic_pose(semantic_pose)
    coeff            += compensation overlay (36 B)                         # <- ra1 SKIPS

and then reproduces ONLY the carrier half of ``render_video`` (inflate.py:659-676),
which is fully independent of the semantic model and the token stream:

    carrier = einsum("bk,kchw->bchw", coeff, basis) / sqrt(CARRIER_DIM)
    slave   = (127.5 + AMP*carrier).clamp(0,255).round()
    frame_0 = bicubic(slave -> CAMERA).clamp(0,255).round().uint8

WHY NO ARCHIVE WRITER IS NEEDED
-------------------------------
``residual_archive`` is read-only; there is no repacker on our side. There does not
need to be. d_pose is a function of the RENDERED FRAMES, not of the container, and the
carrier enters the render at exactly one site. The rate column is separately and
exactly MEASURED by ra1b through the shipped codec. See AMENDMENT 3 C8.

THE CONTROL (this tool's whole job)
-----------------------------------
Mirroring a pinned source is duplication, and duplication is only admissible with a
control that can fail. At alpha = 1 (carrier untouched) the mirrored chain must
reproduce the RETAINED BASE RENDER's even frames BYTE-IDENTICALLY. Any mismatch means
the mirror is wrong -- most likely a missing selector split or a missing compensation
overlay -- and NO ladder row is admissible until it matches. The tool reports the exact
first divergence rather than a pass/fail bit, so a failure names its own cure.

Axis: byte-identity against a retained CPU render [exact]. No scorer here; d_pose is
the successor once the chain is proven.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
GEN = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815"
    "/generations/hv1_base_control"
)
#: The retained CPU base render of THIS archive -- the control's reference.
BASE_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/0.raw"
)
ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
ARCHIVE_BYTES = 182_759

N = 600
CAMERA_H, CAMERA_W = 874, 1164
#: f26_inflate.py builds its synthetic semantic_pose packet with this exact width
#: marker; unpack_semantic_pose discards the semantic half and returns basis/coeff.
SEMANTIC_WIDTH_MARKER = 40_252


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_chain():
    """Import the SHIPPED f26 runtime + the fx1 CPR1 renderer. No re-implementation."""
    for extra in (str(REPO / "src/tac/pr130_runtime/fx1_runtime_tree"), str(GEN)):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location(
        "fx1_renderer", REPO / "src/tac/pr130_runtime/fx1_runtime_tree/inflate.py"
    )
    renderer = importlib.util.module_from_spec(spec)
    sys.modules["fx1_renderer"] = renderer
    spec.loader.exec_module(renderer)

    from runtime.residual_archive import read_residual_archive
    from runtime.carrier_repack import materialize_cpr1, split_frame0_selector_carrier
    from runtime.compensation_overlay import apply_compensation_overlay
    from runtime.frame0_selector import apply_pixel_mode, decode_selector

    return (
        renderer,
        read_residual_archive,
        split_frame0_selector_carrier,
        materialize_cpr1,
        apply_compensation_overlay,
        decode_selector,
        apply_pixel_mode,
    )


def decode_carrier(archive: Path, chain):
    """The exact f26 unwrap, up to (basis, coeff, selector_blob) + provenance."""
    (renderer, read_archive, split_selector, materialize, apply_overlay,
     _decode_selector, _apply_pixel_mode) = chain
    parts = read_archive(archive)
    if parts.schema != "fixed_boundary_int6" or parts.token_codec != "rc64":
        raise SystemExit(f"unexpected schema {parts.schema!r}/{parts.token_codec!r}")

    carrier_blob, selector_blob = split_selector(parts.carrier_blob)
    canonical = materialize(carrier_blob, renderer)
    semantic_pose = (
        struct.pack("<II", SEMANTIC_WIDTH_MARKER, len(canonical))
        + bytes(SEMANTIC_WIDTH_MARKER)
        + canonical
    )
    _, basis, coeff = renderer.unpack_semantic_pose(semantic_pose)

    provenance = {
        "archive_carrier_blob_bytes": len(parts.carrier_blob),
        "post_split_carrier_bytes": len(carrier_blob),
        "selector_blob_bytes": 0 if selector_blob is None else len(selector_blob),
        "canonical_cpr1_bytes": len(canonical),
        "canonical_cpr1_sha256": sha256_bytes(canonical),
        "compensation_applied": False,
        "compensation_bytes": 0,
        "compensation_changed_coordinates": 0,
    }

    # The 36 B compensation overlay. ra1's path SKIPS this; the receiver does not.
    if parts.compensation_blob is not None:
        basis_count = renderer.CARRIER_DIM * 3 * renderer.CARRIER_H * renderer.CARRIER_W
        _, _, coeff_scales, encoded = renderer.decode_compact_carrier(
            canonical, basis_count=basis_count, frames=renderer.N,
            dimensions=renderer.CARRIER_DIM,
        )
        delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
        base_codes = np.cumsum(delta, axis=0) & 0xFFF
        base_codes = np.where(base_codes >= 0x800, base_codes - 0x1000, base_codes)
        base_codes = base_codes.astype(np.int32)
        expected = (
            torch.from_numpy(base_codes).float()
            * torch.from_numpy(coeff_scales)[None]
        )
        if not torch.equal(coeff, expected):
            raise SystemExit(
                "compensation base-code reconstruction differs -- the mirror already "
                "diverges before the overlay; do not proceed"
            )
        candidate = apply_overlay(base_codes, parts.compensation_blob)
        coeff = torch.from_numpy(candidate).float() * torch.from_numpy(coeff_scales)[None]
        provenance.update(
            compensation_applied=True,
            compensation_bytes=len(parts.compensation_blob),
            compensation_changed_coordinates=int(
                np.count_nonzero(candidate != base_codes)
            ),
        )
    return basis, coeff, selector_blob, provenance


def render_frame0(renderer, basis, coeff, indices, alpha: float) -> np.ndarray:
    """The carrier half of render_video (inflate.py:659-676), verbatim, for `indices`.

    alpha scales the coefficients: 1.0 = untouched (the control), 0.0 = carrier deleted.
    """
    basis = renderer.normalized_basis(basis)
    out = np.empty((len(indices), CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
    for slot, idx in enumerate(indices):
        c = coeff[idx : idx + 1] * alpha
        carrier = torch.einsum("bk,kchw->bchw", c, basis)
        carrier = carrier / math.sqrt(renderer.CARRIER_DIM)
        slave_eval = (127.5 + renderer.CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round()
        slave = F.interpolate(
            slave_eval, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False,
        ).clamp(0.0, 255.0).round()
        out[slot] = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=GEN / "archive.zip")
    parser.add_argument("--base-raw", type=Path, default=BASE_RAW)
    parser.add_argument(
        "--pairs", type=str, default="0,299,599",
        help="comma-separated pair indices for the control; 'all' = 0..599",
    )
    parser.add_argument("--output", type=Path, default=Path(
        "/Volumes/APDataStore/pact/ddm_ra2b_carrier_chain_control_20260816/retained"
    ))
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    archive_bytes = args.archive.read_bytes()
    custody = {
        "archive_path": str(args.archive),
        "archive_sha256": sha256_bytes(archive_bytes),
        "archive_bytes": len(archive_bytes),
        "base_raw_path": str(args.base_raw),
        "base_raw_bytes": args.base_raw.stat().st_size,
    }
    if custody["archive_sha256"] != ARCHIVE_SHA or custody["archive_bytes"] != ARCHIVE_BYTES:
        raise SystemExit(
            f"CUSTODY REFUSED: archive sha/bytes differ from the pinned frontier\n"
            f"  got  {custody['archive_sha256']} / {custody['archive_bytes']}\n"
            f"  want {ARCHIVE_SHA} / {ARCHIVE_BYTES}"
        )
    expected_raw = N * 2 * CAMERA_H * CAMERA_W * 3
    if custody["base_raw_bytes"] != expected_raw:
        raise SystemExit(
            f"base render is {custody['base_raw_bytes']} B, expected {expected_raw} B"
        )

    chain = load_chain()
    renderer = chain[0]
    basis, coeff, selector_blob, provenance = decode_carrier(args.archive, chain)
    print(f"decoded carrier: basis {tuple(basis.shape)} coeff {tuple(coeff.shape)}")
    for key, value in provenance.items():
        print(f"  {key}: {value}")

    indices = (
        list(range(N)) if args.pairs.strip() == "all"
        else [int(x) for x in args.pairs.split(",")]
    )
    reference = np.memmap(
        args.base_raw, mode="r", dtype=np.uint8, shape=(N * 2, CAMERA_H, CAMERA_W, 3),
    )

    mine = render_frame0(renderer, basis, coeff, indices, alpha=1.0)

    # STEP 6 (f26_inflate:386 -> _apply_frame0_selector): the selector post-processes
    # output[2*frame_ids] = the frame_0 planes, AFTER the carrier render. ra1's path
    # skips this entirely; the shipped receiver does not. Mirrored here per-index.
    if selector_blob is not None:
        _decode_selector, _apply_pixel_mode = chain[5], chain[6]
        modes, sel_indices = _decode_selector(selector_blob)
        if sel_indices.size != N:
            raise SystemExit(
                f"selector covers {sel_indices.size} frames, expected {N}"
            )
        for slot, idx in enumerate(indices):
            mode = modes[int(sel_indices[idx])]
            mine[slot] = _apply_pixel_mode(mine[slot][None].copy(), mode)[0]
        provenance["selector_modes"] = len(modes)
        provenance["selector_applied"] = True
    else:
        provenance["selector_applied"] = False
    rows, mismatches = [], 0
    for slot, idx in enumerate(indices):
        ref = np.asarray(reference[2 * idx])
        got = mine[slot]
        identical = bool(np.array_equal(ref, got))
        diff = int(np.count_nonzero(ref != got))
        row = {
            "pair": idx,
            "byte_identical": identical,
            "differing_bytes": diff,
            "max_abs_delta": int(np.abs(ref.astype(np.int16) - got.astype(np.int16)).max()),
            "reference_sha256": sha256_bytes(ref.tobytes()),
            "mirror_sha256": sha256_bytes(got.tobytes()),
        }
        rows.append(row)
        mismatches += (not identical)
        flag = "IDENTICAL" if identical else f"DIVERGES ({diff:,} B, max {row['max_abs_delta']})"
        print(f"  pair {idx:3d} frame_0 -> {flag}")

    verdict = "CHAIN_PROVEN" if mismatches == 0 else "CHAIN_DIVERGES"
    receipt = {
        "schema": "ra2b_carrier_chain_control.v1",
        "verdict": verdict,
        "axis": "[exact byte-identity vs a retained CPU render]",
        "score_claim": False,
        "promotable": False,
        "custody": custody,
        "carrier_provenance": provenance,
        "control_rows": rows,
        "pairs_checked": len(indices),
        "mismatches": mismatches,
        "meaning": (
            "CHAIN_PROVEN: the mirrored f26 unwrap + carrier render reproduces the "
            "shipped receiver byte-for-byte, so an alpha-ladder built on it measures the "
            "REAL decode. CHAIN_DIVERGES: the mirror is wrong and no ladder row is "
            "admissible; the differing-byte counts localise the missing step."
        ),
        "elapsed_s": time.time() - started,
    }
    path = args.output / "RA2B_CHAIN_CONTROL.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(f"\nVERDICT: {verdict}  ({mismatches} of {len(indices)} pairs diverge)")
    print(f"receipt -> {path}")
    return 0 if mismatches == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
