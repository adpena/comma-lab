#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Regenerate the EP725 n2 bounded-target-G v2 encoder-control receipt.

WHY THIS TOOL EXISTS.  The July 2026 receipt
``.omx/research/original_taskspace_inverse_witness_codec_20260725/ep725_n2_bounded_target_g_v2_receipt.json``
had no producer: the only thing that could rebuild it was the test that
byte-compares against it (``ddm_ql1``).  A canonical artifact whose sole
regeneration path is its own assertion cannot be refreshed when a legitimate
upstream pin moves -- which is exactly what happened when ``ddm_ql1`` refreshed
the EP725 renderer source pin and three provenance digests in this receipt
moved with it.  This tool is the missing producer.

APPEND-ONLY, NEVER OVERWRITE.  The July receipt is a sealed custody artifact and
is retained untouched.  This tool writes a NEW, dated receipt beside it and
refuses to overwrite any existing file whose bytes differ, so a supersession can
never silently erase its predecessor.  ``--compare-to`` names the retained
receipt; the printed report states, field by field, exactly what moved.

NO SCORER.  ``compile_bounded_target_g_v2`` invokes no SegNet/PoseNet and emits
``scorer_invoked=false``; the target slice is read from the frozen GT label
cache.  This is a structural encoder control, never a score, a rate term, or an
n600 verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    ReceiverRealizationProfileV1,
)
from tac.witness_dsl.bounded_target_g_encoder import (  # noqa: E402
    FrozenTargetSliceCustodyV1,
    compile_bounded_target_g_v2,
    parse_bounded_target_g_encoder_receipt,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (  # noqa: E402
    decode_ep725_prefix_ephemeral_surface,
)

RESEARCH_ROOT = Path(".omx/research/original_taskspace_inverse_witness_codec_20260725")
RETAINED_RECEIPT = RESEARCH_ROOT / "ep725_n2_bounded_target_g_v2_receipt.json"
DEFAULT_OUTPUT = RESEARCH_ROOT / "ep725_n2_bounded_target_g_v2_receipt_20260903.json"
GT_CACHE = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
GT_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
PAIR_COUNT = 2
DECODE_TIMEOUT_SECONDS = 120.0

# The exact structural realization profile the July control used.  Pinned here,
# not re-derived, because changing it would change the packet and make the two
# receipts incomparable -- the whole point of the supersession is that ONLY the
# provenance digests move.
EP725_STRUCTURAL_PROFILE = ((20, 80, 20), (240, 220, 40), (30, 30, 30), (220, 40, 40), (40, 80, 220))


class BoundedTargetGReceiptBuildError(RuntimeError):
    """Raised when the receipt cannot be rebuilt or published honestly."""


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO / path)


def build_receipt_bytes(
    *,
    gt_cache: Path = GT_CACHE,
    gt_cache_sha256: str = GT_CACHE_SHA256,
    pair_count: int = PAIR_COUNT,
    timeout_seconds: float = DECODE_TIMEOUT_SECONDS,
) -> tuple[bytes, dict[str, object]]:
    """Recompile the control and return ``(canonical receipt bytes, receipt dict)``."""

    cache = _resolve(gt_cache)
    if not cache.is_file():
        raise BoundedTargetGReceiptBuildError(f"frozen GT label cache is missing: {cache}")
    surface = decode_ep725_prefix_ephemeral_surface(
        pair_count=pair_count, timeout_seconds=timeout_seconds
    )
    state = surface.bounded_decode.predictor_state
    lstars = open_stored_npy_memmap(cache, "lstars")
    target = np.ascontiguousarray(lstars[:pair_count], dtype=np.uint8)  # SUBSET_SELECTION_OK:the prefix IS this control's definition, not a sample of a population. The July 2026 n2 control is pinned to source_pair_ids (0, 1) inside the retained receipt, and this tool exists to reproduce that exact artifact byte-for-byte; a seeded random draw would compile a DIFFERENT control and make the two receipts incomparable. Nothing here estimates a population quantity: no d_seg, d_pose or rate is claimed, and the receipt carries research_only=true.
    result = compile_bounded_target_g_v2(
        state,
        target,
        target_custody=FrozenTargetSliceCustodyV1(
            cache_sha256=gt_cache_sha256,
            member_name="lstars",
            source_pair_ids=state.source_pair_ids,
            target_labels_sha256=_sha256(memoryview(target).cast("B")),
        ),
        realization_profile=ReceiverRealizationProfileV1(EP725_STRUCTURAL_PROFILE),
    )
    receipt = result.receipt
    payload = receipt.to_receipt_bytes()
    # Re-parse our own bytes before anyone can publish them: a receipt that does
    # not round-trip through the canonical parser is not a receipt.
    if parse_bounded_target_g_encoder_receipt(payload) != receipt:
        raise BoundedTargetGReceiptBuildError("rebuilt receipt does not round-trip its canonical bytes")
    return payload, receipt.as_dict()


def diff_against_retained(fresh: dict[str, object], retained_path: Path) -> dict[str, object]:
    """State field-by-field what moved between the retained receipt and this one."""

    resolved = _resolve(retained_path)
    if not resolved.is_file():
        return {"retained_present": False, "retained_path": retained_path.as_posix()}
    retained_file = resolved.read_bytes()
    if not retained_file.endswith(b"\n") or retained_file.endswith(b"\n\n"):
        raise BoundedTargetGReceiptBuildError("retained receipt lacks its single POSIX terminal newline")
    retained_bytes = retained_file[:-1]
    retained = parse_bounded_target_g_encoder_receipt(retained_bytes).as_dict()
    changed = {
        key: {"retained": retained.get(key), "fresh": fresh.get(key)}
        for key in sorted(set(retained) | set(fresh))
        if retained.get(key) != fresh.get(key)
    }
    return {
        "retained_present": True,
        "retained_path": retained_path.as_posix(),
        "retained_file_sha256": _sha256(retained_file),
        "retained_receipt_sha256": _sha256(retained_bytes),
        "changed_field_count": len(changed),
        "changed_fields": changed,
    }


def write_once(path: Path, payload: bytes) -> str:
    """Publish ``payload`` crash-atomically, never overwriting different bytes."""

    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = target.read_bytes()
        if existing != payload:
            raise BoundedTargetGReceiptBuildError(
                f"refusing to overwrite a different receipt: {target}"
            )
        return "already_present_byte_identical"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise BoundedTargetGReceiptBuildError(f"concurrent receipt differs: {target}") from None
    finally:
        temporary.unlink(missing_ok=True)
    if target.read_bytes() != payload:
        raise BoundedTargetGReceiptBuildError("published receipt bytes differ")
    return "written"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare-to", type=Path, default=RETAINED_RECEIPT)
    parser.add_argument("--gt-cache", type=Path, default=GT_CACHE)
    parser.add_argument("--gt-cache-sha256", default=GT_CACHE_SHA256)
    parser.add_argument("--pair-count", type=int, default=PAIR_COUNT)
    parser.add_argument("--timeout-seconds", type=float, default=DECODE_TIMEOUT_SECONDS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="rebuild and diff, but publish nothing",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload, fresh = build_receipt_bytes(
        gt_cache=args.gt_cache,
        gt_cache_sha256=args.gt_cache_sha256,
        pair_count=args.pair_count,
        timeout_seconds=args.timeout_seconds,
    )
    file_payload = payload + b"\n"
    delta = diff_against_retained(fresh, args.compare_to)
    # PAYLOAD WRITE ORDER: this tool's only product IS the receipt, so there is
    # no cheap record to strand behind it -- the report below is derived from
    # bytes already on disk.
    action = "dry_run" if args.dry_run else write_once(args.output, file_payload)
    print(
        json.dumps(
            {
                "action": action,
                "output": args.output.as_posix(),
                "receipt_sha256": _sha256(payload),
                "file_sha256": _sha256(file_payload),
                "packet_sha256": fresh["packet_sha256"],
                "scorer_invoked": fresh["scorer_invoked"],
                "score_claim": False,
                "promotion_eligible": False,
                "pointer_moved": False,
                "delta_vs_retained": delta,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
