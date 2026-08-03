# SPDX-License-Identifier: MIT
"""ddm_cx1 — compose a solved v4d pose line with the ddm_ix2 single-member container.

WHAT THIS IS
------------
``ddm_ix2`` measured that the bytes were BETWEEN the archive members, not inside
them: six independently framed ZIP members pay 686 B of headers, central-directory
entries and *filenames counted twice*, plus six independent coder warm-ups.  It
landed the container primitives and the receiver; it did not land an encoder that
takes a live 6-member archive to the container form.  This is that encoder.

It is deliberately a TRANSFORM and never a solver: every rung is LOSSLESS, so
``d_seg`` and ``d_pose`` are invariant **by construction** and BYTES are the whole
currency.  That is also the only reason this composition is safe on ``pj2``:
``ddm_cp1`` measured that a re-representation costs a TIGHTER solve relatively more
(``ms8`` −0.009 %, ``pj2`` +0.099 %, 11× worse) because the f16 storage lattice is
not invariant even where the algebra is.  ``pj2`` is the tightest solve we have, so
the moment any rung here re-quantizes rather than re-frames, that law lands hardest
on exactly this base.  Nothing here re-quantizes: the pose member is copied byte for
byte, the token lattice round-trips bit-identically, and the renderer frame is
re-emitted from the same mask bits and the same f16 words.

RULE-118 (the trap this encoder exists to not fall into)
--------------------------------------------------------
``inflate.py`` is free, unsized generic code, so a GENERIC algorithm or table may be
migrated there at zero counted cost — but a value FITTED to the clip is COUNTED
however it is spelled.  The verdict is a ``(field, ARCHIVE)`` pair property:

* ``st_grid`` is byte-equal to the vendored ``ST_GRID`` on ``dc1_fold`` (GENERIC
  there, migratable) and is ``ms8``'s FITTED ladder on ``ms8`` / ``pj2``
  (VIDEO_DERIVED there, counted).  Same key, same slot, opposite verdict.
* ``rs_beta_mags`` is a magnitude codebook the builder derives from this clip's
  solve on every v4d archive — counted on all of them.

So this tool never classifies by field name.  It calls
``classify_against_vendored`` per archive and ``pack_config_section`` refuses to
decide without the reference table.  It also refuses to run at all unless every
value the receiver carries as a MIGRATED CONSTANT is proved byte-equal to this
archive's own copy: a migrated constant that silently differs is a decode defect
wearing a rate saving.

The six ``manifest.sha256`` fields are DELETED rather than migrated.  ``ddm_ix2``
audited all six: 0 read+correct, 4 unread+correct, 2 unread+WRONG
(``tokens_sha256``, ``tr1_packet_sha256``).  A custody hash worth its bytes is one
the receiver reads and fails closed on; none of these were read by any decode step.
Build-time provenance lives in the emitted receipt at zero archive cost.

USAGE
-----
    python tools/cx1_build_ix2_container_archive.py \
        --legacy-archive /path/v4d_composed_pj2_archive.zip \
        --output /path/v4d_composed_cx1_pj2_ix2_archive.zip \
        --receipt /path/cx1_receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from tac.optimization.ddm_ix2_archive_container import (  # noqa: E402
    build_payload,
    build_single_member_zip,
    classify_against_vendored,
    decode_renderer_frame,
    decode_token_frame,
    encode_renderer_frame,
    encode_token_frame,
    pack_config_section,
    parse_payload,
    unpack_config_section,
    zip_framing_overhead,
)
from tac.optimization.ddm_tr1_runtime import (  # noqa: E402
    RENDERER_FRAME_MAGIC,
    RENDERER_RAW_HEADER,
    _decode_brotli_frame,
)

LEGACY_MEMBERS: Final = (
    "manifest.json",
    "state/tokens.dr7t",
    "state/renderer.sec",
    "state/selector.sec",
    "state/pose_stub.sec",
    "state/pose_warp.stp",
)
TOKEN_LEVELS: Final = 16


class CX1BuildError(ValueError):
    """Raised when the legacy archive cannot be carried to container form."""


@dataclass(frozen=True)
class LegacyArchive:
    """The six members of a v4d archive, read once and never re-read."""

    manifest: dict[str, Any]
    tokens: bytes
    renderer: bytes
    selector: bytes
    pose_stub: bytes
    pose_warp: bytes
    zip_bytes: int
    # The manifest's OWN member bytes.  Reporting ``len(json.dumps(manifest))``
    # instead would quietly re-serialise it and overstate the member by ~90 B —
    # a byte accounting that measures the reporter rather than the archive.
    manifest_bytes: int = 0


@dataclass
class BuildAccounting:
    """Byte accounting reported per section, plus the per-field rule-118 verdicts."""

    legacy_zip_bytes: int = 0
    container_zip_bytes: int = 0
    sections: dict[str, int] = field(default_factory=dict)
    legacy_members: dict[str, int] = field(default_factory=dict)
    custody: dict[str, str] = field(default_factory=dict)
    migrated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    counted_config_bytes: int = 0
    legacy_framing_bytes: int = 0
    container_framing_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "legacy_zip_bytes": self.legacy_zip_bytes,
            "container_zip_bytes": self.container_zip_bytes,
            "delta_bytes": self.container_zip_bytes - self.legacy_zip_bytes,
            "sections": dict(self.sections),
            "legacy_members": dict(self.legacy_members),
            "rule118_custody": dict(self.custody),
            "migrated_to_inflate_py": list(self.migrated),
            "deleted_fields": list(self.deleted),
            "counted_config_bytes": self.counted_config_bytes,
            "legacy_zip_framing_bytes": self.legacy_framing_bytes,
            "container_zip_framing_bytes": self.container_framing_bytes,
        }


def read_legacy_archive(path: Path) -> LegacyArchive:
    """Read the six members, refusing an archive that is not the v4d 6-member form."""

    with zipfile.ZipFile(path) as archive:
        names = tuple(info.filename for info in archive.infolist())
        if sorted(names) != sorted(LEGACY_MEMBERS):
            raise CX1BuildError(
                f"expected the v4d 6-member layout {LEGACY_MEMBERS}, found {names}"
            )
        blobs = {name: archive.read(name) for name in LEGACY_MEMBERS}
    return LegacyArchive(
        manifest=json.loads(blobs["manifest.json"]),
        tokens=blobs["state/tokens.dr7t"],
        renderer=blobs["state/renderer.sec"],
        selector=blobs["state/selector.sec"],
        pose_stub=blobs["state/pose_stub.sec"],
        pose_warp=blobs["state/pose_warp.stp"],
        zip_bytes=path.stat().st_size,
        manifest_bytes=len(blobs["manifest.json"]),
    )


def assert_migrated_constants_match(legacy: LegacyArchive, receiver: Any) -> list[str]:
    """Refuse unless every receiver-side MIGRATED constant equals this archive's copy.

    A migrated constant is generic-by-claim; it is only generic-in-fact when the
    archive it replaces carries the same value.  Checking this is what separates a
    legal rule-118 migration from a decode defect that happens to save bytes.
    """

    migrated: list[str] = []
    policy = legacy.manifest.get("frame0_policy")
    if policy != receiver.FRAME0_POLICY:
        raise CX1BuildError(
            f"frame0_policy {policy!r} differs from the receiver constant "
            f"{receiver.FRAME0_POLICY!r}: migration would change the decode"
        )
    migrated.append("frame0_policy")

    metadata = legacy.manifest.get("tr1_metadata")
    if metadata != receiver.IX2_TR1_METADATA:
        raise CX1BuildError(
            "tr1_metadata differs from the receiver's migrated constant"
        )
    migrated.append("tr1_metadata")

    if legacy.pose_stub != receiver.IX2_POSE_STUB:
        raise CX1BuildError(
            "state/pose_stub.sec differs from the receiver's migrated constant"
        )
    migrated.append("pose_stub")
    migrated.append("section_order")
    return migrated


def split_renderer_fields(renderer_sec: bytes) -> tuple[np.ndarray, bytes]:
    """Return ``(mask_bits, floats_f16_be)`` from a shipped TR1 renderer section."""

    raw = _decode_brotli_frame(
        renderer_sec, magic=RENDERER_FRAME_MAGIC, label="lotto_renderer"
    )
    if len(raw) < RENDERER_RAW_HEADER.size:
        raise CX1BuildError("renderer raw frame is truncated")
    mask_count, float_count = RENDERER_RAW_HEADER.unpack_from(raw)
    mask_bytes = (mask_count + 7) // 8
    expected = RENDERER_RAW_HEADER.size + mask_bytes + 2 * float_count
    if len(raw) != expected:
        raise CX1BuildError("renderer raw bytes do not close exactly")
    packed = np.frombuffer(
        raw, dtype=np.uint8, count=mask_bytes, offset=RENDERER_RAW_HEADER.size
    )
    bits = np.unpackbits(packed, bitorder="big")
    if np.any(bits[mask_count:] != 0):
        raise CX1BuildError("renderer mask padding carries counted inert bits")
    floats = raw[RENDERER_RAW_HEADER.size + mask_bytes :]
    return np.ascontiguousarray(bits[:mask_count]), bytes(floats)


def build_container(
    legacy: LegacyArchive,
    *,
    vendored_st_grid: object,
    token_codes: np.ndarray,
    receiver: Any,
) -> tuple[bytes, BuildAccounting]:
    """Build the single-member container payload and its byte accounting."""

    accounting = BuildAccounting(legacy_zip_bytes=legacy.zip_bytes)
    accounting.migrated = assert_migrated_constants_match(legacy, receiver)

    token_frame = encode_token_frame(token_codes, levels=TOKEN_LEVELS)
    if not np.array_equal(decode_token_frame(token_frame), token_codes):
        raise CX1BuildError("token frame does not round-trip bit-identically")

    mask_bits, floats_be = split_renderer_fields(legacy.renderer)
    renderer_frame = encode_renderer_frame(mask_bits, floats_be)
    back_bits, back_floats = decode_renderer_frame(renderer_frame)
    if not np.array_equal(back_bits, mask_bits) or back_floats != floats_be:
        raise CX1BuildError("renderer frame does not round-trip bit-identically")

    st_grid = legacy.manifest.get("st_grid")
    verdict = classify_against_vendored(st_grid, vendored_st_grid)
    accounting.custody["st_grid"] = verdict
    accounting.custody["rs_beta_mags"] = "VIDEO_DERIVED"
    accounting.custody["pose_dim0_offset"] = "VIDEO_DERIVED"
    if verdict == "GENERIC":
        accounting.migrated.append("st_grid")

    config = pack_config_section(
        float(legacy.manifest["pose_dim0_offset"]),
        legacy.manifest["rs_beta_mags"],
        st_grid,
        vendored_st_grid=vendored_st_grid,
    )
    accounting.counted_config_bytes = len(config)

    joint = {
        "config": config,
        "renderer": renderer_frame,
        "selector": legacy.selector,
        "pose_warp": legacy.pose_warp,
    }
    order = tuple(receiver.IX2_JOINT_ORDER)
    if sorted(order) != sorted(joint):
        raise CX1BuildError(
            f"receiver joint order {order} does not match the built sections "
            f"{tuple(sorted(joint))}"
        )
    payload = build_payload(token_frame, [joint[name] for name in order])

    accounting.sections = {
        "tokens(bulk, IX2TOK01)": len(token_frame),
        "config(counted)": len(config),
        "renderer(IX2REN01)": len(renderer_frame),
        "selector": len(legacy.selector),
        "pose_warp": len(legacy.pose_warp),
        "payload_total": len(payload),
    }
    accounting.legacy_members = {
        "manifest.json": legacy.manifest_bytes,
        "state/tokens.dr7t": len(legacy.tokens),
        "state/renderer.sec": len(legacy.renderer),
        "state/selector.sec": len(legacy.selector),
        "state/pose_stub.sec": len(legacy.pose_stub),
        "state/pose_warp.stp": len(legacy.pose_warp),
    }
    accounting.legacy_framing_bytes = zip_framing_overhead(
        list(LEGACY_MEMBERS), [0] * len(LEGACY_MEMBERS)
    )
    accounting.container_framing_bytes = zip_framing_overhead(["0.bin"], [0])
    # Only fields the archive ACTUALLY carried; listing a key that was never there
    # inflates the deletion story with work nobody did.
    unread = ("beta_idx_counts", "selector_num_two", "schema", "base", "pose_carrier")
    accounting.deleted = [
        key for key in sorted(legacy.manifest) if key.endswith("_sha256")
    ] + [key for key in unread if key in legacy.manifest]
    return payload, accounting


def verify_container_bytes(
    payload: bytes,
    legacy: LegacyArchive,
    token_codes: np.ndarray,
    receiver: Any,
) -> dict[str, bool]:
    """Reconstruct every counted field FROM THE CONTAINER BYTES ALONE.

    The decoder must never need the legacy archive it replaced.  Each check below
    is answered from ``payload`` and the receiver's generic constants only.
    """

    bulk, sections = parse_payload(payload)
    order = tuple(receiver.IX2_JOINT_ORDER)
    named = dict(zip(order, sections, strict=True))
    offset, beta_mags, st_grid = unpack_config_section(named["config"])
    mask_bits, floats_be = decode_renderer_frame(named["renderer"])
    legacy_bits, legacy_floats = split_renderer_fields(legacy.renderer)
    resolved_grid = (
        tuple(np.asarray(receiver.ST_GRID, dtype=np.float64).tolist())
        if st_grid is None
        else st_grid
    )
    return {
        "tokens_bit_identical": bool(
            np.array_equal(decode_token_frame(bulk), token_codes)
        ),
        "renderer_mask_bit_identical": bool(np.array_equal(mask_bits, legacy_bits)),
        "renderer_floats_bit_identical": floats_be == legacy_floats,
        "selector_bit_identical": named["selector"] == legacy.selector,
        "pose_warp_bit_identical": named["pose_warp"] == legacy.pose_warp,
        "dim0_offset_exact": float(offset)
        == float(legacy.manifest["pose_dim0_offset"]),
        "beta_mags_exact": list(beta_mags)
        == [float(v) for v in legacy.manifest["rs_beta_mags"]],
        "st_grid_exact": [float(v) for v in resolved_grid]
        == [float(v) for v in legacy.manifest["st_grid"]],
    }


DEFAULT_RUNTIME_TREE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
)


def _import_receiver(runtime_tree: Path) -> Any:
    """Import the receiver with the DECODE-TIME vendored runtime on the path.

    The gate copies ``ddm_tr1_runtime.py`` / ``pfs1_warp_receiver.py`` /
    ``ddm_r7_token_coder.py`` from the submission template, and those vendored
    copies are NOT textually identical to the repo sources.  Building against the
    repo copy and decoding against the vendored copy is exactly the stale-code
    class, so the encoder resolves the same modules the receiver will.  MEASURED
    (``test_vendored_runtime_matches_repo_on_every_path_this_build_touches``):
    they agree bit-for-bit on token decode, ``ST_GRID``, the renderer raw frame and
    ``_encode_tokens`` — a measurement, not an assumption.
    """

    if str(runtime_tree) not in sys.path:
        sys.path.insert(0, str(runtime_tree))
    import inflate_runner_v4d  # noqa: PLC0415

    return inflate_runner_v4d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--member-name",
        default="0.bin",
        help="single-member name; the receiver's compile-time constant",
    )
    parser.add_argument(
        "--runtime-tree",
        type=Path,
        default=DEFAULT_RUNTIME_TREE,
        help="decode-time vendored runtime tree (ddm_tr1_runtime, pfs1_warp_receiver)",
    )
    args = parser.parse_args(argv)

    receiver = _import_receiver(args.runtime_tree)
    from ddm_r7_token_coder import decode_token_codes  # noqa: PLC0415

    legacy = read_legacy_archive(args.legacy_archive)
    token_codes = np.ascontiguousarray(
        np.asarray(decode_token_codes(legacy.tokens), dtype=np.uint8)
    )
    payload, accounting = build_container(
        legacy,
        vendored_st_grid=receiver.ST_GRID,
        token_codes=token_codes,
        receiver=receiver,
    )
    checks = verify_container_bytes(payload, legacy, token_codes, receiver)
    if not all(checks.values()):
        raise CX1BuildError(f"container verification failed: {checks}")

    blob = build_single_member_zip(payload, name=args.member_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(blob)
    accounting.container_zip_bytes = len(blob)

    receipt = {
        "schema": "ddm_cx1_ix2_container_build.v1",
        "legacy_archive": str(args.legacy_archive),
        "legacy_sha256": hashlib.sha256(
            args.legacy_archive.read_bytes()
        ).hexdigest(),
        "container_archive": str(args.output),
        "container_sha256": hashlib.sha256(blob).hexdigest(),
        "verification": checks,
        **accounting.as_dict(),
    }
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
