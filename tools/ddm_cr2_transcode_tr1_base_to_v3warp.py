#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_cr2 — transcode an endpoint TR1 archive into the v3_warp base grammar.

WHY THIS EXISTS.  Two archive grammars are live and they carry the SAME token
grid in DIFFERENT codecs:

  ``ddm_tr1_runtime_archive.v1``   (the ep2 / endpoint line)
      manifest.json + state/tr1.ddt1          -- ONE monolithic 4-section
      packet whose ``tokens`` section uses the tr1-runtime native encoding
      (``_encode_tokens`` / ``_decode_tokens``).

  ``ddm_pfs1_composed_archive.v3_warp``  (the v4c/v4d pose line)
      manifest.json + state/{tokens.dr7t,renderer.sec,selector.sec,
      pose_stub.sec,pose_warp.stp}  -- the SAME 4 sections exploded into ZIP
      members, with ``tokens`` re-coded by the r7 ``smevr`` coder, plus the
      warp pose carrier.  ``inflate_runner_v4d.py:128-136`` decodes
      ``state/tokens.dr7t`` and re-encodes it to the packet form, so the render
      is BIT-IDENTICAL to the monolithic archive's -- d_seg is unchanged by the
      transcode and only the STORED bytes differ.

MEASURED on ``w03_ep854_representative``: the r7 coder stores the identical
code grid in 271,505 B where the packet-native encoding needs 355,182 B.

CANARY (P4 -- no meter without a canary).  Before transcoding anything this
tool re-encodes the reference v3_warp base's own token grid and REFUSES unless
it reproduces the shipped ``state/tokens.dr7t`` byte-for-byte.  That proves the
r7 coder in use is the coder those bytes were produced with.

AUTHORITY.  The runtime modules are imported from the SHIPPED submission dir,
not from the repo: ``src/tac/optimization/ddm_tr1_runtime.py`` and
``experiments/ddm_r7_token_coder.py`` both DIFFER from the shipped copies
(measured 2026-08-01), and the archive must decode under the shipped receiver.

Axis: no scorers are run here.  This tool moves bytes only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

SHIPPED_RUNTIME_DIR = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
)
DEFAULT_REFERENCE_BASE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_cell_drop50_archive.zip"
)
MEMBER_ORDER = ("manifest.json", "state/tokens.dr7t", "state/renderer.sec",
                "state/selector.sec", "state/pose_stub.sec", "state/pose_warp.stp")
DEFLATE_MEMBERS = {"manifest.json", "state/selector.sec"}
SECTION_TO_MEMBER = {"tokens": "state/tokens.dr7t",
                     "lotto_renderer": "state/renderer.sec",
                     "selector": "state/selector.sec",
                     "pose_stub": "state/pose_stub.sec"}


def _load_shipped_runtime(runtime_dir: Path):
    if not runtime_dir.is_dir():
        raise SystemExit(f"shipped runtime dir missing: {runtime_dir}")
    sys.path.insert(0, str(runtime_dir))
    import ddm_r7_token_coder as r7
    import ddm_tr1_runtime as rt
    for mod, want in ((rt, "ddm_tr1_runtime"), (r7, "ddm_r7_token_coder")):
        got = Path(mod.__file__).resolve().parent
        if got != runtime_dir.resolve():
            raise SystemExit(f"{want} resolved to {got}, not the shipped {runtime_dir}")
    return rt, r7


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {i.filename: zf.read(i.filename) for i in zf.infolist()}


def _zip(members: dict[str, bytes]) -> bytes:
    """Byte-for-byte the container convention of ddm_v4d_build_composed_archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            if name in DEFLATE_MEMBERS:
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, data, compresslevel=9)
            else:
                info.compress_type = zipfile.ZIP_STORED
                zf.writestr(info, data)
    return buf.getvalue()


def packet_sections(rt, packet: bytes) -> tuple[dict[str, bytes], dict]:
    """Split a TR1 packet into its raw section payloads + its metadata.

    ``rt.parse_packet`` validates every section SHA and refuses trailing bytes;
    we call it purely as the integrity gate and then read the offsets from the
    runtime's own header/entry structs so no section layout is re-invented here.
    """
    parsed = rt.parse_packet(packet)  # integrity gate: per-section SHA + contiguity
    metadata = dict(parsed.metadata)
    # ASSUMED: the last PACKET_HEADER field is the metadata length.  Rather than
    # trust that, every slice below is checked against the section SHA recorded
    # in the entry table, and the slices must tile the packet exactly -- so a
    # wrong header layout FAILS LOUDLY here instead of yielding plausible bytes.
    metadata_length = int(rt.PACKET_HEADER.unpack_from(packet, 0)[-1])
    table = rt.PACKET_HEADER.size + metadata_length
    sections: dict[str, bytes] = {}
    cursor = table + len(rt.SECTION_NAMES) * rt.SECTION_ENTRY.size
    for index, name in enumerate(rt.SECTION_NAMES):
        _, offset, length, digest = rt.SECTION_ENTRY.unpack_from(
            packet, table + index * rt.SECTION_ENTRY.size)
        if offset != cursor:
            raise SystemExit(f"section {name} is not contiguous at {cursor} "
                             f"(entry says {offset}); packet header layout differs")
        payload = packet[offset:offset + length]
        if hashlib.sha256(payload).digest() != digest:
            raise SystemExit(f"section {name} SHA differs from its entry-table "
                             "digest; the extracted slice is not the section")
        sections[name] = payload
        cursor = offset + length
    if cursor != len(packet):
        raise SystemExit(f"sections tile {cursor} of {len(packet)} packet bytes; "
                         "extraction is incomplete")
    return sections, metadata


def r7_canary(rt, r7, reference_base: Path) -> None:
    """REFUSE unless the r7 coder reproduces the reference base's shipped bytes."""
    members = _read_members(reference_base)
    shipped = members["state/tokens.dr7t"]
    codes = r7.decode_token_codes(shipped)
    again = r7.encode_token_codes(codes)
    if again != shipped:
        raise SystemExit(
            f"r7 CANARY FAILED on {reference_base.name}: re-encode is "
            f"{len(again)} B vs shipped {len(shipped)} B -- the r7 coder in use "
            "is not the coder these bytes were produced with; refusing to transcode")
    print(f"[canary] r7 round-trip on {reference_base.name}: "
          f"{len(shipped)} B BYTE-IDENTICAL")


def transcode(args: argparse.Namespace) -> None:
    rt, r7 = _load_shipped_runtime(Path(args.runtime_dir))
    reference_base = Path(args.reference_base)
    r7_canary(rt, r7, reference_base)

    src_members = _read_members(Path(args.tr1_archive))
    if args.packet_member not in src_members:
        raise SystemExit(
            f"{args.tr1_archive} has no {args.packet_member}; members="
            f"{sorted(src_members)}")
    sections, metadata = packet_sections(rt, src_members[args.packet_member])

    selector = json.loads(sections["selector"])
    codes = rt._decode_tokens(sections["tokens"], selector)
    dr7t = r7.encode_token_codes(codes)
    if not np.array_equal(r7.decode_token_codes(dr7t), codes):
        raise SystemExit("r7 round-trip on the source token grid is NOT lossless")
    # Second, independent leg: the packet-native re-encode must reproduce the
    # shipped section byte-for-byte, which proves the decode was correct.
    if rt._encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8)) != sections["tokens"]:
        raise SystemExit("packet-native re-encode differs from the shipped tokens "
                         "section -- the decode is not trustworthy")
    print(f"[tokens] packet-native {len(sections['tokens'])} B -> r7 {len(dr7t)} B "
          f"({len(dr7t) - len(sections['tokens']):+d} B), grid {codes.shape}, lossless")

    ref_members = _read_members(reference_base)
    manifest = json.loads(ref_members["manifest.json"])
    out: dict[str, bytes] = {}
    for name in MEMBER_ORDER:
        if name == "manifest.json":
            continue
        if name == "state/pose_warp.stp":
            # The warp pose carrier is POSE lineage, not seg lineage: carried
            # from the reference base so the emitted archive is a drop-in
            # v3_warp base (ddm_v4d_build_composed_archive reads st_coded here).
            out[name] = ref_members[name]
            continue
        section = next(k for k, v in SECTION_TO_MEMBER.items() if v == name)
        out[name] = dr7t if section == "tokens" else sections[section]

    manifest["tr1_metadata"] = metadata
    manifest["base"] = args.base_label
    manifest["tokens_sha256"] = _sha(out["state/tokens.dr7t"])
    manifest["renderer_sha256"] = _sha(out["state/renderer.sec"])
    manifest["selector_sha256"] = _sha(out["state/selector.sec"])
    manifest["pose_stub_sha256"] = _sha(out["state/pose_stub.sec"])
    manifest["pose_warp_sha256"] = _sha(out["state/pose_warp.stp"])
    manifest["tr1_packet_sha256"] = _sha(src_members[args.packet_member])
    manifest["cr2_transcode"] = {
        "source_archive_sha256": _sha(Path(args.tr1_archive).read_bytes()),
        "source_packet_member": args.packet_member,
        "pose_warp_from": reference_base.name,
        "token_codec": "r7_smevr",
        "packet_native_token_bytes": len(sections["tokens"]),
        "r7_token_bytes": len(dr7t),
        "sha_convention": "sha256_of_stored_zip_member",
    }
    manifest["score_claim"] = False
    manifest["pointer_moved"] = False
    ordered = {"manifest.json": json.dumps(manifest, sort_keys=True,
                                          separators=(",", ":")).encode()}
    for name in MEMBER_ORDER:
        if name != "manifest.json":
            ordered[name] = out[name]

    archive = _zip(ordered)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(archive)
    ref_bytes = reference_base.stat().st_size
    print(f"[out] {dest}  {len(archive)} B  sha256 {_sha(archive)[:16]}")
    print(f"[out] vs reference base {reference_base.name} {ref_bytes} B: "
          f"{len(archive) - ref_bytes:+d} B "
          f"({25.0 * (len(archive) - ref_bytes) / 37_545_489:+.6f} S rate)")

    # Parse-back: every emitted member must be readable and the packet the
    # receiver rebuilds must parse.
    rebuilt = rt.build_packet(manifest["tr1_metadata"], {
        "tokens": rt._encode_tokens(
            np.ascontiguousarray(r7.decode_token_codes(out["state/tokens.dr7t"]),
                                 dtype=np.uint8)),
        "lotto_renderer": out["state/renderer.sec"],
        "selector": out["state/selector.sec"],
        "pose_stub": out["state/pose_stub.sec"],
    })
    rt.parse_packet(rebuilt)
    print("[parse-back] receiver-path packet rebuild parses OK")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tr1-archive", required=True,
                    help="ddm_tr1_runtime_archive.v1 zip (monolithic packet member)")
    ap.add_argument("--out", required=True, help="destination v3_warp base zip")
    ap.add_argument("--base-label", required=True,
                    help="manifest['base'] label, e.g. w03_ep854")
    ap.add_argument("--reference-base", default=str(DEFAULT_REFERENCE_BASE),
                    help="v3_warp base supplying pose_warp.stp + the r7 canary")
    ap.add_argument("--packet-member", default="state/tr1.ddt1")
    ap.add_argument("--runtime-dir", default=str(SHIPPED_RUNTIME_DIR),
                    help="dir holding the SHIPPED ddm_tr1_runtime + ddm_r7_token_coder")
    transcode(ap.parse_args(argv))


if __name__ == "__main__":
    main()
