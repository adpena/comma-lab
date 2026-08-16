#!/usr/bin/env python3
"""ddm_ps1u — the RECEIVER half of the byte-close: P1D1 as Q2C1's widened sibling.

WHAT THIS IS
------------
The hv1 receiver carries its frame-0 overlay in the **selector tail**:
``runtime/compensation_overlay.py::split_selector_compensation`` peels an exact ``F0E1``
selector prefix and treats everything after it as an optional ``Q2C1`` overlay, validating it
by decode; ``apply_compensation_overlay`` then adds the deltas to the real 600x12 signed-int12
carrier lattice.

``Q2C1`` cannot carry the ps1u pose payload (1-15 pairs, deltas in [-3, 4], versus 53+ pairs
with deltas in [-29, 48]). This module supplies the two dispatch functions the receiver needs
so ``P1D1`` rides the SAME slot as a widened sibling — magic-dispatched, both formats still
accepted, everything fail-closed.

WHAT THIS IS NOT
----------------
It does **not** mutate the pinned submission generation (runtime custody is MAIN-owned) and it
does **not** re-pack the container. The container writer is the remaining gap; this is the
receiver-side contract it will be grafted into, verified against the REAL runtime module and
the REAL archive bytes so the graft is not speculative.

VERIFICATION (``verify`` subcommand, all against live objects)
--------------------------------------------------------------
1. **Control — the shipped path is unchanged.** On the archive's ACTUAL selector tail, this
   module's split reproduces the runtime's own ``split_selector_compensation`` byte-for-byte,
   and the apply reproduces ``apply_compensation_overlay`` element-for-element.
2. **Byte-identity control (the section half of requirement (b)).** ``selector + overlay``
   re-joins to the original tail byte-for-byte, and an ABSENT overlay round-trips to the bare
   selector — so a build that omits P1D1 cannot perturb these bytes.
3. **P1D1 rides the slot.** A P1D1 overlay appended to the same selector splits, decodes, and
   applies correctly, and does not collide with the Q2C1 branch.
4. **Refusals carry through to inflate.** Truncation, bad magic, trailing bytes and int12
   overflow all raise at the receiver boundary rather than decoding to garbage.

AXIS. ``[local-CPU $0 receiver contract verification]``; no scorer, no score, not promotable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ps1u_carrier_delta_codec as p1d1  # noqa: E402

AXIS = "[local-CPU $0 receiver contract verification]"
HV1_GENERATION = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
    "hv1_ep0634_s1p25_c1p0_brotli_q10"
)
Q2C1_MAGIC = b"Q2C1"
P1D1_MAGIC = p1d1.MAGIC
PAIR_COUNT = p1d1.PAIR_COUNT
DIMENSIONS = p1d1.DIMENSIONS


class ReceiverDispatchError(ValueError):
    """The selector tail, overlay magic, or receiving lattice is invalid."""


def _purge_runtime_modules() -> None:
    """Drop any cached ``runtime.*`` package so hv1 and cp135 cannot collide.

    Both generations ship a package literally named ``runtime``; importing one and then
    the other without purging would silently serve the FIRST one's modules. That is a
    wrong-object confound, so it is purged explicitly rather than hoped away."""
    for name in [n for n in sys.modules if n == "runtime" or n.startswith("runtime.")]:
        del sys.modules[name]


def load_hv1_runtime(generation: Path) -> tuple[ModuleType, ModuleType]:
    """Import the REAL shipped ``runtime`` package (relative imports intact)."""
    if not (generation / "runtime" / "compensation_overlay.py").is_file():
        raise ReceiverDispatchError(f"shipped overlay module absent under {generation}")
    _purge_runtime_modules()
    sys.path.insert(0, str(generation))
    try:
        import runtime.compensation_overlay as overlay
        import runtime.residual_archive as residual

        return overlay, residual
    finally:
        sys.path.remove(str(generation))


def split_selector_overlay(
    payload: bytes, runtime: ModuleType
) -> tuple[bytes, bytes | None, str]:
    """Peel the F0E1 selector, then magic-dispatch the optional overlay.

    Returns (selector, overlay_or_None, format_tag). Fail-closed on every branch:
    an unknown magic is REFUSED rather than ignored, so a future format can never be
    silently dropped by an older receiver."""
    selector_bytes = runtime.selector_payload_bytes(payload)
    selector = payload[:selector_bytes]
    overlay = payload[selector_bytes:]
    if not overlay:
        return selector, None, "none"
    magic = overlay[:4]
    if magic == Q2C1_MAGIC:
        runtime.decode_compensation_overlay(overlay)
        return selector, overlay, "Q2C1"
    if magic == P1D1_MAGIC:
        p1d1.decode_carrier_deltas(overlay)
        return selector, overlay, "P1D1"
    raise ReceiverDispatchError(f"unknown frame-0 overlay magic: {magic!r}")


def apply_overlay(
    base_codes: np.ndarray, overlay: bytes | None, runtime: ModuleType
) -> np.ndarray:
    """Apply either overlay format to the real signed-int12 carrier lattice."""
    codes = np.asarray(base_codes, dtype=np.int32)
    if codes.shape != (PAIR_COUNT, DIMENSIONS):
        raise ReceiverDispatchError("base carrier code geometry differs")
    if overlay is None:
        return codes.copy()
    magic = overlay[:4]
    if magic == Q2C1_MAGIC:
        return runtime.apply_compensation_overlay(codes, overlay)
    if magic == P1D1_MAGIC:
        return p1d1.apply_carrier_deltas(codes, overlay)
    raise ReceiverDispatchError(f"unknown frame-0 overlay magic: {magic!r}")


def verify(generation: Path = HV1_GENERATION) -> dict[str, Any]:
    """Every check runs against the LIVE runtime module and the REAL archive bytes."""
    runtime, residual = load_hv1_runtime(generation)
    parts = residual.read_residual_archive(generation / "archive.zip")
    shipped_overlay_blob = parts.compensation_blob
    # The selector tail is built INSIDE _decode_rx1_models and never exposed on parts
    # (`selector_tail = SPARSE_SELECTOR_PREFIX + carrier_body[cap1_bytes:]`), so it is
    # reconstructed from the packed carrier + the compensation blob. If this cannot be
    # reconstructed the checks that depend on it MUST fail loudly -- a skipped check that
    # still reports green is the vacuity trap this arm refuses to ship.
    sys.path.insert(0, str(generation))
    try:
        from runtime.carrier_repack import split_frame0_selector_carrier
    finally:
        sys.path.remove(str(generation))
    _, selector_only = split_frame0_selector_carrier(parts.carrier_blob)
    if selector_only is None:
        raise ReceiverDispatchError(
            "could not recover the F0E1 selector from the packed carrier; the byte-identity "
            "control cannot be evaluated and MUST NOT be reported as passing"
        )
    tail = selector_only + (shipped_overlay_blob or b"")
    _purge_runtime_modules()
    from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1

    surface, _ = qs1.CP135Surface.load()
    base_codes = np.asarray(surface.codes, dtype=np.int32)
    checks: dict[str, Any] = {}

    # --- 1. control: shipped Q2C1 tail, our split == runtime split -----------------
    shipped_overlay = shipped_overlay_blob
    checks["shipped_overlay_present"] = shipped_overlay is not None
    checks["shipped_overlay_magic"] = (
        shipped_overlay[:4].decode(errors="replace") if shipped_overlay else None
    )
    checks["shipped_overlay_bytes"] = len(shipped_overlay) if shipped_overlay else 0
    if shipped_overlay is not None:
        runtime_applied = runtime.apply_compensation_overlay(base_codes, shipped_overlay)
        ours_applied = apply_overlay(base_codes, shipped_overlay, runtime)
        checks["control_apply_matches_runtime"] = bool(
            np.array_equal(runtime_applied, ours_applied)
        )
        checks["control_pairs_touched"] = int(
            np.count_nonzero((runtime_applied != base_codes).any(axis=1))
        )

    # --- 2. byte-identity control on the selector tail ------------------------------
    if True:
        sel_rt, ov_rt = runtime.split_selector_compensation(tail)
        sel_us, ov_us, tag = split_selector_overlay(tail, runtime)
        checks["split_selector_identical"] = sel_rt == sel_us
        checks["split_overlay_identical"] = ov_rt == ov_us
        checks["rejoin_is_byte_identical"] = (sel_us + (ov_us or b"")) == tail
        checks["shipped_tail_format_tag"] = tag
        sel_only, ov_only, tag_only = split_selector_overlay(sel_us, runtime)
        checks["absent_overlay_roundtrip"] = (
            sel_only == sel_us and ov_only is None and tag_only == "none"
        )

    # --- 3. P1D1 rides the same slot ------------------------------------------------
    pairs = list(range(0, 120, 2))
    deltas = np.zeros((len(pairs), DIMENSIONS), dtype=np.int64)
    deltas[:, 0] = 7
    deltas[:, 5] = -29
    deltas[:, 11] = 48
    section = p1d1.encode_carrier_deltas(pairs, deltas)
    if True:
        synthetic = sel_us + section
        sel_p, ov_p, tag_p = split_selector_overlay(synthetic, runtime)
        checks["p1d1_splits_from_same_slot"] = (
            sel_p == sel_us and ov_p == section and tag_p == "P1D1"
        )
    applied = apply_overlay(base_codes, section, runtime)
    expected = base_codes.copy()
    expected[pairs] += deltas.astype(np.int32)
    checks["p1d1_apply_correct"] = bool(np.array_equal(applied, expected))
    checks["p1d1_section_bytes"] = len(section)

    # --- 4. refusals carry through to the receiver boundary --------------------------
    refusals: dict[str, bool] = {}
    for name, payload in (
        ("truncated", section[:-1]),
        ("bad_magic", b"ZZZZ" + section[4:]),
        ("trailing_bytes", section + b"\x00"),
    ):
        try:
            split_selector_overlay(sel_us + payload, runtime)
        except (ReceiverDispatchError, p1d1.CarrierDeltaCodecError, ValueError):
            refusals[name] = True
        else:
            refusals[name] = False
    try:
        apply_overlay(np.full((PAIR_COUNT, DIMENSIONS), 2047, np.int32), section, runtime)
    except (ReceiverDispatchError, p1d1.CarrierDeltaCodecError):
        refusals["int12_overflow"] = True
    else:
        refusals["int12_overflow"] = False
    checks["refusals"] = refusals

    required = (
        "shipped_overlay_present", "control_apply_matches_runtime",
        "split_selector_identical", "split_overlay_identical",
        "rejoin_is_byte_identical", "absent_overlay_roundtrip",
        "p1d1_splits_from_same_slot", "p1d1_apply_correct",
    )
    missing = [k for k in required if k not in checks]
    checks["required_checks_total"] = len(required)
    checks["required_checks_evaluated"] = len(required) - len(missing)
    failed = [f"NOT_EVALUATED:{k}" for k in missing]
    failed += [k for k, v in checks.items() if isinstance(v, bool) and not v]
    failed += [f"refusal:{k}" for k, v in refusals.items() if not v]
    return {
        "schema": "ddm_ps1u_receiver_p1d1_contract.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "generation": str(generation),
        "generation_untouched": True,
        "checks": checks,
        "failed_checks": failed,
        "verdict": "RECEIVER_CONTRACT_VERIFIED" if not failed else "RECEIVER_CONTRACT_FAILED",
        "remaining_gap": (
            "the CONTAINER WRITER: re-packing the `p` member with this selector tail and "
            "re-zipping to a new archive. The runtime is decode-only by design "
            "(read_residual_archive has no writer), so the writer lives in the compile path "
            "and is the one piece this arm did not build."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default=str(HV1_GENERATION))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    report = verify(Path(args.generation))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if report["verdict"] == "RECEIVER_CONTRACT_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
