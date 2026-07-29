#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pb1 P5 — compose the final archive, byte-close, and run the pn1 S1
Stage-A dress rehearsal ($0 local full-n600 advisory-exact via the locked-env
``evaluate.sh`` path).

``--build``: compose the final counted archive from the frozen seg endpoint
TR1 archive + the P3 pose coefficients:
  members (deterministic stored ZIP):
    manifest.json      canonical composed manifest (frame0 policy, member shas)
    state/tr1.ddt1     the TR1 packet (tokens+renderer+selector+pose-stub)
    state/pose.tpgn    TerminalPosePacketV1, Brotli-Q11-coded
It stages a full submission dir {archive.zip, inflate.sh, inflate_runner.py,
ddm_tr1_runtime.py, terminal_pose_gn.py} — receiver code is generic/free under
rule 118; every video-derived byte lives in archive.zip.  Parse-back +
exact-consumption asserts run at build time (round-trip decode of both
members; pose coefficient equality; TR1 section hashes via the committed
parser).

``--eval``: stage the locked eval root (pinned evaluate.py/sh + frame_utils +
modules + models from the eg1 rehearsal custody; GT video = the pinned
upstream ``videos/0.mkv``) and run ``bash evaluate.sh --device cpu`` with the
locked env-brotli python on PATH.  Output: report.txt + a receipt with S
recomputed from components (never the rounded field).

Axis: [macOS-CPU advisory — real evaluator, real bytes]; the local row is
ADVISORY until the Modal contest-CPU flight (P6, operator-GO).
score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pb1_p5_receipt.v1"
LOCKED_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_eg1_tr1_rehearsal_20260728")
LOCKED_PY = Path("/Volumes/VertigoDataTier/pact/evidence/"
                 "ddm_e4_brotli_declared_dep_20260724/env_brotli/bin/python")
UPSTREAM_VIDEO = REPO / "upstream/videos/0.mkv"
COMPOSED_MEMBERS = ("manifest.json", "state/tr1.ddt1", "state/pose.tpgn")

INFLATE_SH = """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/inflate_runner.py" "$1" "$2" "$3"
"""

INFLATE_RUNNER = '''from __future__ import annotations

import json
import sys
from pathlib import Path, PurePosixPath

import brotli
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ddm_tr1_runtime import parse_packet, render_frame1_camera_uint8
from terminal_pose_gn import (
    parse_terminal_pose_packet,
    realize_terminal_pose_pair,
)

RANK = 6


def render_basis(shape):
    height, width, channels = shape
    x = np.cos(2.0 * np.pi * (np.arange(width, dtype=np.float64) + 0.5) / width)
    y = np.cos(2.0 * np.pi * (np.arange(height, dtype=np.float64) + 0.5) / height)
    fields = np.zeros((RANK, height, width, channels), dtype=np.float32)
    for channel in range(3):
        fields[channel, :, :, channel] = x[None, :]
        fields[channel + 3, :, :, channel] = y[:, None]
    return fields


def main() -> None:
    archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
    names = [row.strip() for row in
             names_file.read_text().splitlines() if row.strip()]
    if names != ["0.mkv"]:
        raise SystemExit("this receiver serves exactly the custodied 0.mkv")
    manifest = json.loads((archive_dir / "manifest.json").read_text())
    packet = parse_packet((archive_dir / "state/tr1.ddt1").read_bytes())
    pose_raw = (archive_dir / "state/pose.tpgn").read_bytes()
    if manifest.get("pose_coding") == "brotli":
        pose_raw = brotli.decompress(pose_raw)
    pose = parse_terminal_pose_packet(pose_raw)
    policy = manifest["frame0_policy"]
    if policy not in ("zeros", "copy"):
        raise SystemExit("unknown frame0 policy")
    n_pairs = int(packet.selector["num_pairs"])
    if pose.coefficients.shape != (n_pairs, RANK):
        raise SystemExit("pose coefficient shape differs")

    name = PurePosixPath(names[0])
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit("unsafe video name")
    target = output_dir / name.with_suffix(".raw")
    target.parent.mkdir(parents=True, exist_ok=True)
    basis = None
    with open(target, "wb") as sink:
        for i in range(n_pairs):
            f1 = render_frame1_camera_uint8(packet, i)
            if basis is None:
                basis = render_basis(f1.shape)
            f0_base = np.zeros_like(f1) if policy == "zeros" else f1.copy()
            pair = realize_terminal_pose_pair(
                np.stack([f0_base, f1], axis=0),
                basis,
                pose.coefficients[i],
                amplitude_q8=pose.amplitude_q8,
            )
            sink.write(pair[0].tobytes())
            sink.write(pair[1].tobytes())


if __name__ == "__main__":
    main()
'''


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def deterministic_stored_zip(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    if tuple(members) != COMPOSED_MEMBERS:
        raise SystemExit("composed member order differs")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w",
                         compression=zipfile.ZIP_STORED,
                         allowZip64=False) as archive:
        for name_ in COMPOSED_MEMBERS:
            info = zipfile.ZipInfo(name_, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, members[name_])
    return stream.getvalue()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("build", "eval"), required=True)
    ap.add_argument("--seg-archive", type=Path,
                    help="frozen final seg endpoint TR1 archive (build)")
    ap.add_argument("--pose-progress", type=Path,
                    help="P3 progress json with coefficients (build)")
    ap.add_argument("--frame0-policy", choices=("zeros", "copy"))
    ap.add_argument("--work-dir", required=True, type=Path)
    return ap.parse_args()


def build(args: argparse.Namespace) -> None:
    import brotli

    sys.path.insert(0, str(REPO / "src"))
    from tac.optimization import ddm_tr1_runtime as rt
    from tac.optimization.terminal_pose_gn import (
        TerminalPosePacketV1,
        parse_terminal_pose_packet,
        serialize_terminal_pose_packet,
    )

    work = args.work_dir
    (work / "submission").mkdir(parents=True, exist_ok=True)
    tr1_bytes = args.seg_archive.read_bytes()
    parsed = rt.parse_archive(tr1_bytes)
    packet_bytes = rt.reemit_packet(parsed.packet)

    progress = json.loads(args.pose_progress.read_text())
    if progress["endpoint_archive_sha256"] != _sha(tr1_bytes):
        raise SystemExit("P3 progress endpoint sha differs from seg archive")
    if progress["frame0_policy"] != args.frame0_policy:
        raise SystemExit("frame0 policy differs from P3 progress")
    coeffs = np.asarray(progress["coefficients"], dtype=np.int16)
    pose_packet = TerminalPosePacketV1(
        seed=20260728,
        basis_selector="eg1_generic_low_frequency_six_v1",
        amplitude_q8=512,
        coefficients=coeffs,
    )
    pose_payload = serialize_terminal_pose_packet(pose_packet)
    pose_coded = brotli.compress(pose_payload, quality=11)

    manifest = {
        "schema": "ddm_pb1_composed_archive.v1",
        "frame0_policy": args.frame0_policy,
        "pose_coding": "brotli",
        "pose_raw_sha256": _sha(pose_payload),
        "tr1_packet_sha256": _sha(packet_bytes),
        "research_only": True,
        "score_claim": False,
        "pointer_moved": False,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True,
                                separators=(",", ":")).encode()
    archive = deterministic_stored_zip({
        "manifest.json": manifest_bytes,
        "state/tr1.ddt1": packet_bytes,
        "state/pose.tpgn": pose_coded,
    })

    # parse-back + exact-consumption asserts
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert tuple(i.filename for i in z.infolist()) == COMPOSED_MEMBERS
        rb_manifest = json.loads(z.read("manifest.json"))
        rb_packet = rt.parse_packet(z.read("state/tr1.ddt1"))
        rb_pose_raw = brotli.decompress(z.read("state/pose.tpgn"))
    if rb_manifest != manifest:
        raise SystemExit("manifest round-trip differs")
    rb_pose = parse_terminal_pose_packet(rb_pose_raw)
    if not np.array_equal(rb_pose.coefficients, coeffs):
        raise SystemExit("pose coefficients round-trip differs")
    if rt.reemit_packet(rb_packet) != packet_bytes:
        raise SystemExit("TR1 packet round-trip differs")
    nonzero_rows = int(np.count_nonzero(
        np.any(rb_pose.coefficients != 0, axis=1)))
    print(f"[consumption] TR1 sections + pose packet round-trip exact; "
          f"{nonzero_rows} nonzero pose rows consumed by the receiver",
          flush=True)

    sub = work / "submission"
    (sub / "archive.zip").write_bytes(archive)
    (sub / "inflate.sh").write_text(INFLATE_SH)
    (sub / "inflate_runner.py").write_text(INFLATE_RUNNER)
    shutil.copy2(REPO / "src/tac/optimization/ddm_tr1_runtime.py",
                 sub / "ddm_tr1_runtime.py")
    shutil.copy2(REPO / "src/tac/optimization/terminal_pose_gn.py",
                 sub / "terminal_pose_gn.py")
    receipt = {
        "schema": SCHEMA,
        "stage": "build",
        "archive_bytes": len(archive),
        "archive_sha256": _sha(archive),
        "members": {
            "manifest.json": len(manifest_bytes),
            "state/tr1.ddt1": len(packet_bytes),
            "state/pose.tpgn": len(pose_coded),
        },
        "pose_raw_bytes": len(pose_payload),
        "pose_coded_bytes": len(pose_coded),
        "seg_archive_sha256": _sha(tr1_bytes),
        "frame0_policy": args.frame0_policy,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
    }
    (work / "p5_build_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in
                      ("archive_bytes", "archive_sha256", "members",
                       "pose_coded_bytes")}, indent=1), flush=True)


def run_eval(args: argparse.Namespace) -> None:
    work = args.work_dir
    root = work / "eval_root"
    if not (root / "evaluate.sh").exists():
        root.mkdir(parents=True, exist_ok=True)
        for name_ in ("evaluate.py", "evaluate.sh", "frame_utils.py",
                      "modules.py", "public_test_video_names.txt"):
            shutil.copy2(LOCKED_ROOT / name_, root / name_)
        (root / "models").mkdir(exist_ok=True)
        for name_ in ("posenet.safetensors", "segnet.safetensors"):
            shutil.copy2(LOCKED_ROOT / "models" / name_,
                         root / "models" / name_)
        (root / "videos").mkdir(exist_ok=True)
        shutil.copy2(UPSTREAM_VIDEO, root / "videos" / "0.mkv")
    free = shutil.disk_usage(root).free
    if free < 8 * (1 << 30):
        raise SystemExit(f"storage preflight: only {free/(1<<30):.1f} GiB free")

    sub_src = work / "submission"
    sub = root / "submissions" / "pb1"
    if sub.exists():
        shutil.rmtree(sub)
    sub.parent.mkdir(exist_ok=True)
    shutil.copytree(sub_src, sub)

    env = os.environ.copy()
    env["PATH"] = f"{LOCKED_PY.parent}:{env['PATH']}"
    env["TQDM_DISABLE"] = "1"
    t0 = time.time()
    proc = subprocess.run(
        ["bash", str(root / "evaluate.sh"),
         "--submission-dir", str(sub),
         "--video-names-file", str(root / "public_test_video_names.txt"),
         "--device", "cpu"],
        cwd=root, env=env, capture_output=True, text=True, timeout=3600,
    )
    wall = time.time() - t0
    (work / "p5_eval_stdout.txt").write_text(proc.stdout)
    (work / "p5_eval_stderr.txt").write_text(proc.stderr)
    report_path = sub / "report.txt"
    report = report_path.read_text() if report_path.exists() else None
    receipt = {
        "schema": SCHEMA,
        "stage": "eval",
        "returncode": proc.returncode,
        "wall_seconds": wall,
        "report": report,
        "archive_sha256": _sha((sub_src / "archive.zip").read_bytes()),
        "evidence_axis": "[macOS-CPU advisory - real evaluator, real bytes]",
        "score_claim": False,
        "note": "ADVISORY until the Modal contest-CPU flight (P6 operator-GO)",
    }
    (work / "p5_eval_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"rc={proc.returncode} wall={wall:.0f}s", flush=True)
    if report:
        print(report, flush=True)
    else:
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "build":
        for req in ("seg_archive", "pose_progress", "frame0_policy"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req.replace('_', '-')} required for build")
        build(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
