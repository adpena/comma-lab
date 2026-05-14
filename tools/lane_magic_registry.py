# SPDX-License-Identifier: MIT
"""Lane → expected renderer/state-dict magic-byte registry.

Bug class extincted (2026-05-01): Q-FAITHFUL v1/v2 dispatches CRASHED in
Stage 0 (argparse errors) but `tools/harvest_modal_calls.py` later harvested
SCAFFOLD FILES from the volume mount that LOOKED like training outputs.
The harvested renderer_fp4_ep3560.bin had FP4A magic + motion_hidden=32
(Lane A/V/V2/K family), NOT QFAI magic + JointFrameGenerator. Months of
confusion if undetected.

Permanent fix: every `lane_id` we dispatch declares the magic-bytes its
output renderer.bin / renderer_fp4.bin MUST start with. The harvester (and
PCC10 preflight check) refuses to mark a harvest "successful" unless the
file's first N bytes match the expected registry value.

Reference:
- feedback_pcc9_pcc12_permanent_fixes_20260501.md
- feedback_modal_spawn_result_cache_pattern_20260429.md (Modal harvest pattern)

Public API:
    expected_magic_for_lane(lane_id: str) -> bytes | None
    verify_renderer_magic(path, expected) -> tuple[bool, bytes]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

# Each value is the MAGIC BYTES the FIRST 4 (or 8) bytes of the output
# file MUST equal. None means the lane DOES NOT produce a magic-byte-tagged
# artifact (e.g. a pose-only or mask-only lane) — verification is skipped.
#
# Add a lane the moment it is registered in `tools/lane_maturity.py add-lane`.
# Wildcard '*' or substring lookup is supported via the lookup helper.
LANE_MAGIC_REGISTRY: Mapping[str, bytes | None] = {
    # FP4A family (Lane A 88K-param renderer + variants).
    "lane_a": b"FP4A",
    "lane_a_optimized": b"FP4A",
    "lane_a_sweep": b"FP4A",
    "lane_v_quantizr_replica_88k": b"FP4A",
    "lane_v_v2_annealed_halfframe": b"FP4A",
    "lane_g_v3": b"FP4A",
    "lane_g_v3_owv3": b"FP4A",
    "lane_g_v3_owv3_fisher": b"FP4A",
    "lane_g_v3_owv3_r7": b"FP4A",
    "lane_g_v3_owv3_0120": b"FP4A",
    "lane_pfp16": b"FP4A",
    "lane_pfp16_a_plus_plus": b"FP4A",
    "lane_owv3_0120_orthogonal_stack": b"FP4A",

    # QFAI: Quantizr-faithful 88K JointFrameGenerator (a different param layout
    # than FP4A — JointFrameGenerator vs RenderHead). The Q-FAITHFUL bug class
    # was explicitly that a QFAI dispatch silently emitted FP4A scaffold bytes.
    "lane_q_faithful": b"QFAI",
    "lane_q_faithful_jointgen": b"QFAI",
    "lane_q_faithful_h100": b"QFAI",
    "lane_q_faithful_h100_redeploy": b"QFAI",

    # Selfcomp / SegMap-paradigm lanes (block-FP weights + grayscale-LUT masks).
    "lane_sa": b"SCBP",  # Selfcomp clone, block-FP signature
    "lane_sc_plus_plus": b"SCBP",
    "lane_sc_plus_plus_v3": b"SCBP",
    "lane_so_hessian_block_fp": b"SCBP",
    "lane_mm": b"SCBP",
    "lane_mm_v2": b"SCBP",

    # NWC (neural weight codec) writes its own header.
    "lane_j_nwc": b"NWC1",
    "lane_nwc": b"NWC1",

    # NeRV mask codec writes a NRV1 header (per src/tac/nerv_mask_codec.py).
    "lane_12_nerv": b"NRV1",
    "lane_nerv": b"NRV1",

    # Ballé hyperprior-on-qint stream.
    "lane_20_balle": b"BHv1",
    "lane_balle": b"BHv1",

    # Lanes that produce ONLY pose / ONLY mask / ONLY codec metadata —
    # no renderer output that needs a magic-byte verify. Mark None
    # so the verifier skips them rather than emitting a false-positive.
    "lane_pd": None,            # pose-delta arithmetic codec
    "lane_pd_v2": None,
    "lane_19_logit_margin": None,  # SegNet logit margin tweak — same FP4A
    "lane_17_imp": b"FP4A",     # IMP-pruned variant of FP4A baseline
    "lane_j_imp": b"FP4A",
    "lane_8_multipass": b"FP4A",
}


# ────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ────────────────────────────────────────────────────────────────────────


def expected_magic_for_lane(lane_id: str) -> bytes | None:
    """Return expected magic bytes for `lane_id` (None if registry says skip).

    Lookup is exact first; falls back to a longest-prefix match so
    `lane_g_v3_owv3_r7_LANDED_1_013_20260501` resolves to `lane_g_v3_owv3_r7`.

    Returns None for unregistered lanes — caller (PCC10) decides whether to
    fail-closed on unknown lanes.
    """
    if lane_id in LANE_MAGIC_REGISTRY:
        return LANE_MAGIC_REGISTRY[lane_id]
    # Longest-prefix match (most specific wins).
    candidates = sorted(
        (k for k in LANE_MAGIC_REGISTRY if lane_id.startswith(k)),
        key=len,
        reverse=True,
    )
    if candidates:
        return LANE_MAGIC_REGISTRY[candidates[0]]
    return None


def verify_renderer_magic(
    path: Path | str,
    expected: bytes,
    *,
    bytes_to_read: int = 8,
) -> tuple[bool, bytes]:
    """Read the first `bytes_to_read` bytes of `path` and compare to `expected`.

    Returns (matched, actual_bytes). `matched` is True iff actual_bytes
    starts with expected. False on missing file / unreadable / non-match.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False, b""
    try:
        actual = p.read_bytes()[:bytes_to_read]
    except OSError:
        return False, b""
    return actual.startswith(expected), actual


def renderer_files_in_dir(directory: Path | str) -> Iterable[Path]:
    """Yield every renderer*.bin / renderer*.pt under directory recursively.

    Used by the harvester to walk a freshly-harvested artifact dir and
    verify each renderer payload it found.
    """
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return
    patterns = ("renderer*.bin", "renderer*.pt", "renderer*_fp4.bin")
    seen: set[Path] = set()
    for pat in patterns:
        for p in d.rglob(pat):
            if p in seen:
                continue
            seen.add(p)
            yield p


__all__ = [
    "LANE_MAGIC_REGISTRY",
    "expected_magic_for_lane",
    "verify_renderer_magic",
    "renderer_files_in_dir",
]
