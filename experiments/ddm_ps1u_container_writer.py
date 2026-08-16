#!/usr/bin/env python3
"""ddm_ps1u — the CONTAINER WRITER: re-pack the hv1 archive with a P1D1 selector tail.

THE STRUCTURE (measured at source, not assumed)
-----------------------------------------------
``archive.zip`` holds ONE member ``p`` (182,659 B). Its RX1 model header is followed by the
hpac / semantic / carrier streams; the carrier stream is brotli. Decompressed, the carrier body
is ``PACKED_CAP1_SECTION_BYTES`` (22,183) of packed CAP1 metadata followed by the optional
frame-0 overlay — and on the shipped archive the bytes at exactly that offset are ``Q2C1``,
36 B, running to the end of the body (22,219). **So the overlay is the trailing slice of the
carrier body**, and swapping it is a splice plus one recompression.

THE RECOMPRESSION IS EXACT (the fact that makes byte-identity possible)
----------------------------------------------------------------------
``brotli.compress(body, quality=11, lgwin=24)`` reproduces the shipped 22,161 B carrier stream
**byte-for-byte**. That was measured, not guessed: quality 9 and 10 miss by length, and
quality 11 at lgwin 22 matches in length but not in bytes. The writer pins (11, 24) and
re-verifies the round-trip on every run rather than trusting the constant.

PROOFS THIS MODULE PRODUCES (the three the seal requires)
---------------------------------------------------------
(a) **byte-identity control** — writing with the ORIGINAL overlay (or with none, when the base
    carries none) must reproduce the pinned archive: 182,759 B, sha ``80d9c8c6…``. This runs
    through the WRITER path, so it proves the writer is a no-op when the payload is.
(b) **parse-back** — the written archive is read back through ``read_residual_archive`` and the
    recovered compensation blob must be byte-equal to the retained P1D1 section; the decoded
    deltas must equal the solved deltas exactly.
(c) **repeat-identical** — two independent writer runs must produce equal sha256.

AXIS ``[local-CPU $0 deterministic container write]``. No scorer, no score, not promotable.
The pinned generation is READ ONLY; every output goes to the arm's own store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ps1u_carrier_delta_codec as p1d1  # noqa: E402

AXIS = "[local-CPU $0 deterministic container write]"
HV1_GENERATION = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pq1_submission_packet/generations/"
    "hv1_ep0634_s1p25_c1p0_brotli_q10"
)
BASE_ARCHIVE_SHA256 = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
BASE_ARCHIVE_BYTES = 182_759
MEMBER_NAME = "p"
BROTLI_QUALITY = 11
BROTLI_LGWIN = 24


class ContainerWriterError(RuntimeError):
    """A container invariant, recompression identity, or parse-back proof differed."""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_runtime(generation: Path):
    for name in [n for n in sys.modules if n == "runtime" or n.startswith("runtime.")]:
        del sys.modules[name]
    sys.path.insert(0, str(generation))
    try:
        import runtime.residual_archive as residual

        return residual
    finally:
        sys.path.remove(str(generation))


def read_member(generation: Path) -> bytes:
    with zipfile.ZipFile(generation / "archive.zip") as handle:
        names = handle.namelist()
        if names != [MEMBER_NAME]:
            raise ContainerWriterError(f"unexpected archive members: {names}")
        return handle.read(MEMBER_NAME)


def zip_info_template(generation: Path) -> zipfile.ZipInfo:
    """Reuse the SHIPPED member's ZipInfo so container metadata cannot drift."""
    with zipfile.ZipFile(generation / "archive.zip") as handle:
        info = handle.getinfo(MEMBER_NAME)
    fresh = zipfile.ZipInfo(filename=info.filename, date_time=info.date_time)
    fresh.compress_type = info.compress_type
    fresh.external_attr = info.external_attr
    fresh.internal_attr = info.internal_attr
    fresh.create_system = info.create_system
    fresh.create_version = info.create_version
    fresh.extract_version = info.extract_version
    fresh.flag_bits = info.flag_bits
    return fresh


def splice_member(member: bytes, new_overlay: bytes | None, residual) -> dict[str, Any]:
    """Replace the frame-0 overlay in the carrier body and re-emit the member."""
    import brotli

    header = residual.RX1_MODEL_HEADER
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = (
        header.unpack_from(member)
    )
    if magic != residual.RX1_MAGIC or codec != residual.RX1_CODEC_BROTLI:
        raise ContainerWriterError("hv1 member is not an RX1/brotli container")
    offset = header.size + hpac_bytes + semantic_bytes
    carrier_stream = member[offset : offset + carrier_bytes]
    model_end = offset + carrier_bytes
    body = brotli.decompress(carrier_stream)

    # The recompression identity is RE-PROVEN every run, never trusted as a constant.
    if brotli.compress(body, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN) != carrier_stream:
        raise ContainerWriterError(
            "brotli(quality=11, lgwin=24) does not reproduce the shipped carrier stream; "
            "byte-identity is not provable under these parameters"
        )

    packed = residual.PACKED_CAP1_SECTION_BYTES
    old_overlay = body[packed:]
    if old_overlay and old_overlay[:4] not in (b"Q2C1", p1d1.MAGIC):
        raise ContainerWriterError(f"unexpected overlay magic in base: {old_overlay[:4]!r}")
    new_body = body[:packed] + (new_overlay or b"")
    new_stream = brotli.compress(new_body, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
    new_member = (
        header.pack(
            magic, version, codec, table_mode, reserved,
            hpac_bytes, semantic_bytes, len(new_stream),
        )
        + member[header.size : offset]
        + new_stream
        + member[model_end:]
    )
    return {
        "member": new_member,
        "old_overlay": old_overlay,
        "new_overlay": new_overlay or b"",
        "carrier_stream_bytes": len(new_stream),
        "carrier_stream_delta": len(new_stream) - carrier_bytes,
        "carrier_body_bytes": len(new_body),
    }


def write_archive(member: bytes, out_path: Path, template: zipfile.ZipInfo) -> dict[str, Any]:
    """Deterministic single-member zip: fixed ZipInfo, no timestamps of our own."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as handle:
        handle.writestr(template, member)
    payload = tmp.read_bytes()
    tmp.replace(out_path)
    return {"path": str(out_path), "bytes": len(payload), "sha256": _sha(payload)}


def build(
    *,
    generation: Path,
    section_path: Path | None,
    out_dir: Path,
    rows_glob: str | None = None,
) -> dict[str, Any]:
    residual = _load_runtime(generation)
    base_member = read_member(generation)
    template = zip_info_template(generation)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "ddm_ps1u_container_writer_proof.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "generation_read_only": True,
        "brotli": {"quality": BROTLI_QUALITY, "lgwin": BROTLI_LGWIN},
    }

    # --- (a) byte-identity control THROUGH THE WRITER PATH -------------------------
    spliced = splice_member(base_member, None, residual)
    control_overlay = spliced["old_overlay"]
    identity = splice_member(base_member, control_overlay, residual)
    control = write_archive(identity["member"], out_dir / "control_archive.zip", template)
    report["byte_identity_control"] = {
        "archive": control,
        "expected_bytes": BASE_ARCHIVE_BYTES,
        "expected_sha256": BASE_ARCHIVE_SHA256,
        "member_identical": identity["member"] == base_member,
        "bytes_match": control["bytes"] == BASE_ARCHIVE_BYTES,
        "sha_match": control["sha256"] == BASE_ARCHIVE_SHA256,
        "passed": (
            identity["member"] == base_member
            and control["bytes"] == BASE_ARCHIVE_BYTES
            and control["sha256"] == BASE_ARCHIVE_SHA256
        ),
    }
    if not report["byte_identity_control"]["passed"]:
        report["verdict"] = "BYTE_IDENTITY_CONTROL_FAILED"
        return report

    if section_path is None:
        report["verdict"] = "CONTROL_ONLY"
        return report

    # --- candidate build, twice, for (c) -------------------------------------------
    section = Path(section_path).read_bytes()
    p1d1.decode_carrier_deltas(section)
    builds = []
    for index in (1, 2):
        result = splice_member(base_member, section, residual)
        written = write_archive(
            result["member"], out_dir / f"candidate_build{index}.zip", template
        )
        builds.append({**written, "carrier_stream_bytes": result["carrier_stream_bytes"]})
    report["repeat_identical"] = {
        "build1": builds[0],
        "build2": builds[1],
        "passed": builds[0]["sha256"] == builds[1]["sha256"],
    }
    report["candidate_archive"] = builds[0]
    report["archive_delta_bytes"] = builds[0]["bytes"] - BASE_ARCHIVE_BYTES
    report["section_bytes"] = len(section)
    report["section_sha256"] = _sha(section)

    # --- (b) parse-back through the real receiver ----------------------------------
    parts = residual.read_residual_archive(Path(builds[0]["path"]))
    recovered = parts.compensation_blob
    ok_bytes = recovered == section
    pairs, deltas = p1d1.decode_carrier_deltas(recovered) if recovered else ([], None)
    parse_back: dict[str, Any] = {
        "recovered_overlay_bytes": len(recovered) if recovered else 0,
        "recovered_equals_retained_section": bool(ok_bytes),
        "recovered_pairs": len(pairs),
    }
    if rows_glob:
        import glob as _glob

        rows: list[dict[str, Any]] = []
        for path in _glob.glob(rows_glob):
            with open(path) as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        rows.sort(key=lambda r: int(r["pair"]))
        active = [r for r in rows if any(r["code_delta"])]
        expect_pairs = [int(r["pair"]) for r in active]
        expect_deltas = np.asarray([r["code_delta"] for r in active], dtype=np.int32)
        parse_back["deltas_equal_solved"] = bool(
            pairs == expect_pairs and np.array_equal(deltas, expect_deltas)
        )
    parse_back["passed"] = bool(ok_bytes and parse_back.get("deltas_equal_solved", True))
    report["parse_back"] = parse_back

    failed = [
        name
        for name in ("byte_identity_control", "repeat_identical", "parse_back")
        if not report[name]["passed"]
    ]
    report["failed_proofs"] = failed
    report["verdict"] = "CONTAINER_WRITER_PROVEN" if not failed else "CONTAINER_WRITER_FAILED"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default=str(HV1_GENERATION))
    ap.add_argument("--section", default=None)
    ap.add_argument("--rows-glob", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args(argv)
    report = build(
        generation=Path(args.generation),
        section_path=Path(args.section) if args.section else None,
        out_dir=Path(args.out_dir),
        rows_glob=args.rows_glob,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n")
    print(text)
    return 0 if report.get("verdict") in ("CONTAINER_WRITER_PROVEN", "CONTROL_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
