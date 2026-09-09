#!/usr/bin/env python3
"""ddm_rp1 build -- stage the re-encoded token stream into the move-37 member, then close.

THE ONE THING THAT HAD TO BE MEASURED RATHER THAN ASSUMED
----------------------------------------------------------
The RC64 encoder's ``finish()`` payload is NOT what the archive carries.  ``finish()``
returns ``TOKEN_MAGIC || body || flush``; the shipped TC1M rider carries the BODY alone.
MEASURED on cmp1's own artefacts: its ``mixed_0600.envelope`` is 119,784 B, the stream
inside cmp2's shipped rider is 119,779 B, and

    shipped_stream == envelope[4:-1]

holds byte for byte (4-byte ``R6D1`` magic off the front, one 0x00 flush byte off the end).
Splicing the envelope directly would have produced a 5-B-larger tail that no receiver
parses -- a mispricing AND a dead archive.  So the transform is applied, and it is proven
by a NULL BUILD on every run: this arm's own control envelope, sliced and re-packed, must
reproduce the pointer's archive to the byte.  If it does not, nothing downstream is a
measurement.

``[macOS-CPU advisory / scorer-free EXACT byte measurement]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg2_tail_reencode as jg2  # noqa: E402
import ddm_rp1_pose as rp1pose  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402
import ddm_sj1_joint_admission as sj1ja  # noqa: E402  (READ-ONLY: inflate-pin patcher)
import ddm_tc1_mixer_codec as tc1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

RESIDUAL_COMPACT_BYTES = jg2.RESIDUAL_COMPACT_BYTES  # 96
#: The counted TC1M weights the pointer ships, read from the pointer's own rider.
N_PAIRS = rp1.N_PAIRS


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def envelope_to_stream(envelope: bytes, route_magic: bytes) -> bytes:
    """``finish()`` payload -> the bytes the rider carries.  Checked, not assumed."""
    if not envelope.startswith(route_magic):
        raise rp1.Rp1Error("envelope does not start with the RC64 token magic")
    if envelope[-1] != 0:
        raise rp1.Rp1Error(
            f"envelope's final byte is {envelope[-1]:#x}, not the 0x00 flush byte the "
            "shipped rider drops; the transform is not the one that was measured"
        )
    return envelope[len(route_magic) : -1]


def pointer_parts(tree: Path) -> tuple[dict[str, bytes], bytes, bytes]:
    """Split the pointer member and pull its TC1M weights and current stream."""
    member = jg2.read_archive_member(tree / "archive.zip")
    parts = jg2.split_member(member)
    rider = parts["tail"][RESIDUAL_COMPACT_BYTES:]
    weights, stream = tc1.unpack_rider(rider)
    return parts, weights, stream


def build_member(parts: dict[str, bytes], weights: bytes, stream: bytes) -> bytes:
    rider = tc1.MAGIC + weights + stream
    parsed = tc1.unpack_rider(rider)
    if parsed != (weights, stream):
        raise rp1.Rp1Error("rider parse-back changed the stream")
    spliced = dict(parts)
    spliced["tail"] = parts["tail"][:RESIDUAL_COMPACT_BYTES] + rider
    return jg2.join_member(spliced)


def cmd_stage(args: argparse.Namespace) -> int:
    pointer = Path(args.pointer_runtime)
    live = rp1.verify_pointer(expect_sha=rp1pose.POINTER_ARCHIVE_SHA256)
    if not live["matches_expected"]:
        raise rp1.Rp1Error(f"pointer moved to {live['archive_sha256']}; re-base first")
    pointer_bytes = (pointer / "archive.zip").read_bytes()
    pointer_sha = sha256_bytes(pointer_bytes)
    if pointer_sha != rp1pose.POINTER_ARCHIVE_SHA256:
        raise rp1.Rp1Error(f"pointer tree sha {pointer_sha} is not the live row's")

    parts, weights, shipped_stream = pointer_parts(pointer)
    route_magic = jg2.load_route_b().TOKEN_MAGIC

    # NULL BUILD.  The control envelope, sliced and re-packed, must reproduce the pointer
    # archive byte for byte -- otherwise the candidate's byte delta is measured against a
    # baseline this code cannot even rebuild.
    control = Path(args.control_envelope).read_bytes()
    control_stream = envelope_to_stream(control, route_magic)
    if control_stream != shipped_stream:
        raise rp1.Rp1Error(
            f"control stream ({len(control_stream)} B) is not the stream the pointer "
            f"ships ({len(shipped_stream)} B); this arm's coder is not the live coder"
        )
    null_member = build_member(parts, weights, control_stream)
    null_sha = sha256_bytes(null_member)
    # Container framing is chosen by the packer, so compare the MEMBER, then the archive.
    if null_member != jg2.read_archive_member(pointer / "archive.zip"):
        raise rp1.Rp1Error("NULL BUILD FAILED: re-packed member differs from the pointer's")

    candidate = Path(args.candidate_envelope).read_bytes()
    stream = envelope_to_stream(candidate, route_magic)
    member = build_member(parts, weights, stream)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    staged = out / "staged_runtime"
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(pointer, staged)
    for cache in staged.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    archive = staged / "archive.zip"
    jg2.pack_archive(member, archive)
    facts = {"sha256": rp1.sha256_file(archive), "bytes": archive.stat().st_size}
    pins = sj1ja.patch_inflate_pins(staged, facts["sha256"], facts["bytes"])

    differing = sorted(
        str(f.relative_to(staged))
        for f in staged.rglob("*")
        if f.is_file()
        and (pointer / f.relative_to(staged)).is_file()
        and f.read_bytes() != (pointer / f.relative_to(staged)).read_bytes()
    )
    if differing != ["archive.zip", "inflate.py"]:
        raise rp1.Rp1Error(
            f"staged tree differs from the pointer in {differing}, expected exactly "
            "['archive.zip', 'inflate.py']"
        )

    report = {
        "schema": "ddm_rp1_stage_tail.v1",
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
        "pointer_runtime": str(pointer),
        "pointer_archive_sha256": pointer_sha,
        "pointer_archive_bytes": len(pointer_bytes),
        "null_build": {
            "member_sha256": null_sha,
            "reproduces_pointer_member": True,
            "control_envelope": str(args.control_envelope),
            "control_stream_bytes": len(control_stream),
        },
        "envelope_transform": "stream = envelope[len(TOKEN_MAGIC):-1]  (MEASURED)",
        "candidate_stream": {
            "envelope": str(args.candidate_envelope),
            "envelope_bytes": len(candidate),
            "stream_bytes": len(stream),
            "stream_sha256": sha256_bytes(stream),
            "delta_stream_bytes_vs_pointer": len(stream) - len(shipped_stream),
        },
        "staged_runtime": str(staged),
        "staged_archive": facts,
        "delta_bytes_vs_pointer": facts["bytes"] - len(pointer_bytes),
        "inflate_pins": pins,
        "sections": {k: len(v) for k, v in jg2.split_member(member).items()},
    }
    (out / "STAGE_TAIL.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    """Splice the re-solved carrier into the staged body and price it on measured legs."""
    import ddm_up3_carrier_splice as splice

    runtime = Path(args.body_runtime)
    base_bytes = (runtime / "archive.zip").read_bytes()
    observed = sha256_bytes(base_bytes)
    body = splice.parse_shipped_body(runtime, verify_sha=False)
    base_codes = np.asarray(body.codes, dtype=np.int32)

    identity = splice.build_archive(
        body, base_codes, runtime_dir=runtime, container_search=True
    )
    if identity["archive_sha256"] != observed:
        raise rp1.Rp1Error(
            "CARRIER IDENTITY CONTROL FAILED: rebuilding the staged body from its own "
            f"codes gives {identity['archive_sha256']} ({identity['archive_size']} B), "
            f"not {observed} ({len(base_bytes)} B)"
        )

    codes = np.load(args.codes).astype(np.int32)
    if codes.shape != (N_PAIRS, up2.CARRIER_DIM):
        raise rp1.Rp1Error(f"codes have shape {codes.shape}")
    admitted = sorted(int(p) for p in json.loads(Path(args.admitted_pairs).read_text()))
    candidate_codes = base_codes.copy()
    for pair in admitted:
        candidate_codes[pair] = codes[pair]

    built = splice.build_archive(
        body, candidate_codes, runtime_dir=runtime, container_search=True, verify=True
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "candidate_archive.zip"
    archive_path.write_bytes(built["archive_bytes"])
    np.save(out / "candidate_codes.npy", candidate_codes)

    import ddm_jg5_pose_resolve_on_edited_renders as jg5

    proof = jg5.section_identity(built["archive_bytes"], base_bytes, runtime=runtime)
    if not proof["frame1_sections_all_identical"]:
        raise rp1.Rp1Error(
            f"a frame-1 section moved ({proof}); refusing to launder a seg leg through a "
            "changed odd-frame section"
        )

    import math

    archive_bytes = int(built["archive_size"])
    d_seg = args.d_seg_t4
    d_pose = args.d_pose
    score = (
        100.0 * d_seg
        + math.sqrt(10.0 * d_pose)
        + 25.0 * archive_bytes / rp1.jg1.SCORE_RATE_DENOMINATOR
    )
    pointer_score = rp1pose.rp1.verify_pointer()["score"]
    report: dict[str, Any] = {
        "schema": "ddm_rp1_close.v1",
        "axis": (
            "seg VERIFIED equal to the base on whole decodes (0 of 117,964,800 cells); "
            "pose [macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]; bytes EXACT"
        ),
        "score_claim": False,
        "body_runtime": str(runtime),
        "body_archive_sha256": observed,
        "body_archive_bytes": len(base_bytes),
        "carrier_identity_control_passed": True,
        "carrier_identity_sha256": identity["archive_sha256"],
        "candidate_archive": {
            "path": str(archive_path),
            "sha256": built["archive_sha256"],
            "bytes": archive_bytes,
        },
        "carrier_pairs_spliced": len(admitted),
        "carrier_coordinates_changed": int((candidate_codes != base_codes).sum()),
        "frame1_section_identity": proof,
        "d_seg_t4": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "score_projected": score,
        "pointer_score_t4": pointer_score,
        "net_dS_vs_pointer": score - pointer_score,
        "admit_bar": -2e-05,
        "clears_admit_bar_vs_pointer": bool(score - pointer_score <= -2e-05),
    }
    (out / "CLOSE.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage-tail")
    stage.add_argument("--pointer-runtime", default=str(rp1pose.POINTER_TREE))
    stage.add_argument("--control-envelope", required=True)
    stage.add_argument("--candidate-envelope", required=True)
    stage.add_argument("--out-dir", required=True)
    stage.set_defaults(func=cmd_stage)

    close = sub.add_parser("close")
    close.add_argument("--body-runtime", required=True)
    close.add_argument("--codes", required=True)
    close.add_argument("--admitted-pairs", required=True)
    close.add_argument("--out-dir", required=True)
    close.add_argument("--d-seg-t4", type=float, required=True)
    close.add_argument("--d-pose", type=float, required=True)
    close.set_defaults(func=cmd_close)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
