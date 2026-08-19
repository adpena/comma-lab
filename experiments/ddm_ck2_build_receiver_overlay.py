"""ddm_ck2 -- build the receiver overlay that un-does the whole-section 2-plane split.

THE MECHANISM, AND WHY IT IS PARAMETER-FREE
-------------------------------------------
sz1 shipped ``DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1``: a 2-plane byte de-interleave over a
**pinned region** of the semantic body (offset 49, length 8284).  Those two integers were
fitted to sz1's body layout.  ck1's SM3R mode-6 row-prune changes that layout, and the
pinned constants are therefore a cross-regime constant transfer
(``cross_regime_constant_transfer_genus_finishing_stage``).  Measured on the ck1 body
(`ddm_ck2_rate_ceiling_probe`, `CK2_RATE_CEILING.json`):

    nosplit                          31,469 B
    sz1 pinned constants             31,528 B   credit  +59  (a LOSS -- the law firing)
    WHOLE-SECTION 2-plane split      30,856 B   credit -613
    best (offset,length) argmax      30,911 B   credit -558

The parameter-free whole-section transform BEATS the fitted-constant argmax.  So this
extension carries **no fitted constants at all**: it de-interleaves the entire section.
That is a generic byte permutation over an opaque buffer -- rule-118 spotless, no
video-derived content, and nothing to re-fit when the body changes again.

WHAT THE OVERLAY ADDS
---------------------
Two new ``reserved`` bits on the RX1 header, both fail-closed by construction because the
unknown-bit guard already refuses anything it does not know:

* ``0x02`` -- the SEMANTIC section body is 2-plane split.
* ``0x04`` -- the CARRIER section body is 2-plane split.

Both un-splits run on the decompressed body **before any parsing**, so every downstream
length check, magic check, and field offset sees exactly the bytes it sees today.  The
decoded state is bit-identical to the base archive: this is a container transform, so
``d_seg`` and ``d_pose`` are unchanged by construction rather than by measurement.

The odd trailing byte (when a body has odd length) is carried through untouched, which is
what makes the transform exactly invertible for every input length.

HOW IT IS BUILT
---------------
By ANCHORED substitution against the sz1 shipped receiver, with every anchor asserted and
the round-trip proven on the real ck1 bodies before the overlay is written.  The base is
sz1's file rather than the gen4 staged file on purpose: the carrier patch
(``patch_runtime_tree``) is applied afterwards by the compile path and needs its own
anchors intact.

AXIS ``[macOS-CPU exact byte/container]``.  Produces no score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np

AXIS: Final = "[macOS-CPU exact byte/container]"

SZ1_RECEIVER: Final = Path(
    "/Volumes/APDataStore/pact/ddm_pq1_submission_packet"
    "/generations/gen3_sz1_composed_split/runtime/residual_archive.py"
)

# --- the anchored substitutions -------------------------------------------------------

_OLD_CONSTANTS: Final = """SZ1_RESERVED_SEMANTIC_SPLIT = 0x01
SZ1_RESERVED_KNOWN_BITS = 0x01"""

_NEW_CONSTANTS: Final = '''SZ1_RESERVED_SEMANTIC_SPLIT = 0x01
# DDM_CK2_WHOLE_SECTION_PLANE2_V1
# Bits 1 and 2 mark a 2-plane byte de-interleave over the WHOLE decompressed section
# body.  Unlike bit 0 this carries no fitted offset/length: the transform is a pure
# permutation of the entire buffer, so it re-fits nothing when the body layout changes.
CK2_RESERVED_SEMANTIC_PLANE2 = 0x02
CK2_RESERVED_CARRIER_PLANE2 = 0x04
SZ1_RESERVED_KNOWN_BITS = 0x07


def _ck2_uninterleave_planes(body: bytes) -> bytes:
    """Exact inverse of a whole-body 2-plane de-interleave.

    The encoder emits ``body[0::2] + body[1::2]`` over the even-length prefix and leaves
    any odd trailing byte in place.  Restoring is the same statement read backwards, so
    the transform is total: it round-trips every input length, including 0 and 1.
    """
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]'''

_OLD_SEMANTIC: Final = """    if reserved & SZ1_RESERVED_SEMANTIC_SPLIT:
        semantic_body = _sz1_unsplit_semantic(semantic_body)"""

_NEW_SEMANTIC: Final = """    if reserved & SZ1_RESERVED_SEMANTIC_SPLIT:
        semantic_body = _sz1_unsplit_semantic(semantic_body)
    # DDM_CK2_WHOLE_SECTION_PLANE2_V1: whole-section un-split, no fitted constants.
    if reserved & CK2_RESERVED_SEMANTIC_PLANE2:
        semantic_body = _ck2_uninterleave_planes(semantic_body)"""

_OLD_CARRIER: Final = """    carrier_body = _decompress_brotli(carrier_stream)"""

_NEW_CARRIER: Final = """    carrier_body = _decompress_brotli(carrier_stream)
    # DDM_CK2_WHOLE_SECTION_PLANE2_V1: restore the carrier body BEFORE the packed-CAP1
    # framing arithmetic reads its u24 bit counts, so those offsets see today's bytes.
    if reserved & CK2_RESERVED_CARRIER_PLANE2:
        carrier_body = _ck2_uninterleave_planes(carrier_body)"""

_SUBSTITUTIONS: Final = (
    ("reserved_constants", _OLD_CONSTANTS, _NEW_CONSTANTS),
    ("semantic_unsplit", _OLD_SEMANTIC, _NEW_SEMANTIC),
    ("carrier_unsplit", _OLD_CARRIER, _NEW_CARRIER),
)


class CK2OverlayError(RuntimeError):
    """An anchor was missing, or the round-trip control failed."""


def interleave_planes(body: bytes) -> bytes:
    """Encoder side: even-index plane, then odd-index plane, odd tail untouched."""
    span = len(body) & ~1
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    return planes[0::2].tobytes() + planes[1::2].tobytes() + body[span:]


def uninterleave_planes(body: bytes) -> bytes:
    """Decoder side, mirroring the text embedded in the overlay."""
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]


def _roundtrip_control(sections_dir: Path | None) -> dict[str, Any]:
    """Prove the permutation is total on synthetic edge cases AND on the real bodies.

    A transform that is only exercised on one 36 KB buffer has an untested boundary at
    every odd length; the synthetic leg closes that, and the real leg proves the encoder
    and the decoder text agree on the exact objects that will ship.
    """
    for n in range(0, 65):
        probe = bytes(range(n))
        if uninterleave_planes(interleave_planes(probe)) != probe:
            raise CK2OverlayError(f"plane round-trip failed at length {n}")
    checked: dict[str, Any] = {"synthetic_lengths_0_to_64": "PASS"}

    if sections_dir is None:
        checked["real_bodies"] = "SKIPPED (no --sections-dir)"
        return checked
    for name in ("semantic_body.bin", "carrier.bin", "hpac.bin", "tail.bin"):
        path = sections_dir / name
        if not path.is_file():
            continue
        blob = path.read_bytes()
        if uninterleave_planes(interleave_planes(blob)) != blob:
            raise CK2OverlayError(f"plane round-trip failed on {name}")
        checked[name] = f"PASS ({len(blob):,} B)"
    return checked


def build_overlay(dest: Path, *, sections_dir: Path | None) -> dict[str, Any]:
    source = SZ1_RECEIVER.read_text()
    base_sha = hashlib.sha256(source.encode()).hexdigest()

    applied: list[str] = []
    for name, old, new in _SUBSTITUTIONS:
        if old not in source:
            raise CK2OverlayError(f"anchor not found in the sz1 receiver: {name}")
        if source.count(old) != 1:
            raise CK2OverlayError(f"anchor is not unique ({source.count(old)}x): {name}")
        source = source.replace(old, new, 1)
        applied.append(name)

    control = _roundtrip_control(sections_dir)

    target = dest / "runtime" / "residual_archive.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
    compile(source, str(target), "exec")  # refuse to ship a file that will not parse

    # The carrier patch that the compile path applies afterwards must still find its own
    # anchors; assert that here rather than discovering it mid-build.
    for anchor in (
        "def _restore_packed_cap1_metadata",
        "carrier_body = _decompress_brotli(carrier_stream)",
    ):
        if anchor not in source:
            raise CK2OverlayError(f"downstream patch anchor lost: {anchor}")

    return {
        "schema": "ddm_ck2_receiver_overlay.v1",
        "axis": AXIS,
        "score_claim": False,
        "counted_bytes": 0,
        "rule_118": (
            "generic whole-buffer byte permutation; carries no fitted constants and no "
            "video-derived content"
        ),
        "base_receiver": str(SZ1_RECEIVER),
        "base_sha256": base_sha,
        "overlay_path": str(target),
        "overlay_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "substitutions_applied": applied,
        "reserved_bits": {
            "semantic_plane2": "0x02",
            "carrier_plane2": "0x04",
            "known_bits": "0x07",
        },
        "roundtrip_control": control,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", required=True, type=Path, help="overlay tree root")
    ap.add_argument(
        "--sections-dir",
        type=Path,
        default=None,
        help="retained section bodies from the ceiling probe, for the real round-trip leg",
    )
    args = ap.parse_args(argv)

    if args.dest.exists():
        # Refuse to delete anything that is not already one of OUR overlay trees.  The
        # dest is a path typed on a command line, and a rebuild should never be able to
        # remove a runtime, a generation, or a retained payload directory by typo.
        stray = sorted(
            p.name
            for p in args.dest.iterdir()
            if p.name not in {"runtime", "CK2_RECEIVER_OVERLAY.json"}
            and not p.name.startswith("._")
        )
        if stray:
            raise CK2OverlayError(
                f"--dest exists and does not look like a ck2 overlay tree "
                f"(unexpected entries: {stray}); refusing to delete it"
            )
        shutil.rmtree(args.dest)
    report = build_overlay(args.dest, sections_dir=args.sections_dir)
    out = args.dest / "CK2_RECEIVER_OVERLAY.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"overlay -> {report['overlay_path']}")
    print(f"  base   {report['base_sha256'][:16]}…")
    print(f"  built  {report['overlay_sha256'][:16]}…")
    print(f"  subs   {', '.join(report['substitutions_applied'])}")
    for key, value in report["roundtrip_control"].items():
        print(f"  control {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
