#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_pfs1 D1 — recompose the pb1 composed archive with the WARP-BASE pose carrier.

Swap the pb1 archive's 6-cosine ``state/pose.tpgn`` (7,295 B raw / 1,876 B
Brotli) for a cheap decoder-reproducible WARP-BASE carrier:

  * frame_0 = ground-homography warp of the reconstructed frame_1 by the CARRIED
    6-value PoseNet target (t_p sidecar, float16) + a per-pair translation-scale
    index ``s_t`` (one of an 11-value grid).  Reproduces p3v2 ``warp_base_work``
    (``experiments/ddm_p3v2_optimal_form_pose_resolve.py``) EXACTLY.

CONSISTENCY FIX (pfs1 finding): p3v2's s3 warp (n600 mean d_pose 0.3931) was
solved on the STALE ct1 FRAME_ROOT frames (state 2a2c0367, 07-25), which are a
DIFFERENT vehicle than the pb1 burn-endpoint archive (p2c_aimed token render,
max_abs 255 apart, d_seg 0.389).  So 0.3931 does NOT transfer.  This tool
RE-SOLVES the warp on the archive's OWN shipped frame_1 (``--mode solve``) with
the SHIPPED float16 targets, so the receiver reconstruction is BYTE-IDENTICAL to
the solve by construction (same f1 render fn, same float16 t_p, same s_t, same
warp code).  n8 ballpark on the shipped f1: mean d_pose 0.2725 -> contribution
1.6507 (BETTER than the stale-frameroot 1.9827).

Grammar v3 (deterministic stored ZIP; only the pose member changes vs pb1 v2):
  manifest.json / state/tokens.dr7t / state/renderer.sec / state/selector.sec /
  state/pose_stub.sec  (verbatim from the pb1 seg endpoint) +
  state/pose_warp.stp  (NEW warp carrier = t_p float16 Brotli + s_t r7 SMEVR).

The receiver (free generic code, rule 118) decodes DR7T, rebuilds the TR1 packet
BYTE-IDENTICAL, renders frame_1 per pair, then reconstructs frame_0 by the SAME
deterministic numpy warp (vendored ``pfs1_warp_receiver.py``; NO scorer at
inflate).  The 6-value target + s_t are the ONLY video-derived payload.

Modes:  --mode solve  (n600 chunked, resumable: fit s_t on the shipped f1)
        --mode build  (compose archive v3; warp byte-identity + parse-back)
        --mode eval   (bash evaluate.sh --device cpu; locked env, full n600)

Axis: [macOS-CPU advisory - real evaluator, real bytes]; ADVISORY until the
Modal Stage-B flight (P6, operator-GO). score_claim=false. pointer_moved=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

from tac.process_group_kill import run_in_process_group

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_pfs1_d1_receipt.v3_warp"
LOCKED_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_eg1_tr1_rehearsal_20260728")
LOCKED_PY = Path("/Volumes/VertigoDataTier/pact/evidence/"
                 "ddm_e4_brotli_declared_dep_20260724/env_brotli/bin/python")
UPSTREAM_VIDEO = REPO / "upstream/videos/0.mkv"
SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729")
DEFAULT_SEG_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2c_aimed_archive.zip")
# Rebound from ``--seg-archive`` in main(); the pose warp is ALWAYS re-solved on
# whichever base is selected, because a warp fitted to one base's renders does
# not transfer to another's (this tool's own docstring, D1 provenance).
SEG_ARCHIVE = DEFAULT_SEG_ARCHIVE
TARGETS = Path(
    "/Volumes/VertigoDataTier/pact/"
    "ddm_ms4_metric_producers_and_measurement_20260724T042005Z/"
    "pose_metric_n600_batch32.json")
CAMERA_H, CAMERA_W = 874, 1164
ST_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
POSE_WARP_MAGIC = b"PFS1WPB1"
PB1_POSE_TPGN_CODED = 1876   # the pb1 pose member replaced by pose_warp.stp
PB1_POSE_TPGN_RAW = 7295

COMPOSED_MEMBERS = (
    "manifest.json",
    "state/tokens.dr7t",
    "state/renderer.sec",
    "state/selector.sec",
    "state/pose_stub.sec",
    "state/pose_warp.stp",
)

# Resolve the interpreter explicitly and fail CLOSED.  A bare ``python`` is
# not portable: on a host that ships only ``python3`` it dies with
# "python: command not found", and a backgrounding wrapper then reports the
# LAUNCHER's rc=0 instead of that 127 -- the failure and the success carry the
# same symbol.  Measured live by ddm_ob1 2026-08-03 (task #929).  This emitter
# is on the frontier arc's own authority path: it produced the canonical
# ``v4d_cx1_pj2ix2`` decode that ob1's independent inflate was compared
# against.  ``exec`` keeps the runner's own exit status as this script's exit
# status, so no rc can be swallowed here either.
INFLATE_SH = """#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PY="${candidate}"
      break
    fi
  done
fi
if [ -z "${PY}" ]; then
  echo "inflate.sh: FATAL: no Python interpreter (tried PYTHON env var, python3, python)" >&2
  exit 127
fi
exec "${PY}" "$HERE/inflate_runner.py" "$1" "$2" "$3"
"""

# --- vendored warp receiver (rule 118: generic deterministic numpy; constants are
#     the MEASURED-ANCHOR EON intrinsics that clip_profile reproduces bit-identically).
PFS1_WARP_RECEIVER = '''# SPDX-License-Identifier: MIT
"""pfs1 warp receiver (vendored, generic, deterministic; rule 118).

Byte-faithful copy of the warp engine in tools/measure_pose_warp_dseg.py +
tools/measure_screw_warp_through_R.py used by p3v2 warp_base_work, with the EON
road-camera intrinsics + camera height hardcoded to the documented literals
(comma2k19 utils/camera.py: focal 910, FULL_FRAME 1164x874, pp=(582,437);
openpilot HEIGHT_INIT 1.22).  clip_profile reproduces these bit-identically on
0.mkv, so this receiver needs NO tac dependency."""
from __future__ import annotations

import numpy as np

NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
CAMERA_HEIGHT_M = 1.22
CAMERA_H, CAMERA_W = 874, 1164
ST_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]


def intrinsics_native() -> np.ndarray:
    return np.array([[NATIVE_FX, 0.0, NATIVE_CX],
                     [0.0, NATIVE_FY, NATIVE_CY],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _target_grid(hh: int, ww: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(ww), np.arange(hh))
    return np.stack([us.ravel(), vs.ravel(), np.ones(hh * ww)], 0).astype(np.float64)


def _expmap_so3(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + K
    return (np.eye(3)
            + (np.sin(theta) / theta) * K
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (K @ K))


def pose_to_homography(pose6, K, Kinv, s_t, s_r, pitch) -> np.ndarray:
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    M = R - np.outer(t, n) / CAMERA_HEIGHT_M
    return K @ M @ Kinv


def regime_homography_ground(pose6, K, Kinv, s_t) -> np.ndarray:
    return pose_to_homography(pose6, K, Kinv, s_t, 0.0, 0.0)


def warp_rgb(src_hwc: np.ndarray, H: np.ndarray, tgt_grid: np.ndarray) -> np.ndarray:
    Hh, Ww, C = src_hwc.shape
    srcf = src_hwc.astype(np.float64)
    flat = srcf.reshape(-1, C)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Hinv = np.linalg.inv(H)
        src_h = Hinv @ tgt_grid
        z = src_h[2]
        su = src_h[0] / z
        sv = src_h[1] / z
    valid = (
        np.isfinite(su) & np.isfinite(sv) & (z > 0)
        & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1)
    )
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]
    Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]
    Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat)
    return out.reshape(Hh, Ww, C)


def _to_uint8(frame_f: np.ndarray) -> np.ndarray:
    return np.clip(np.round(frame_f), 0.0, 255.0).astype(np.uint8)


def warp_base_f0(f1_u8: np.ndarray, target6: np.ndarray, s_t: float) -> np.ndarray:
    """Deterministic camera-res uint8 frame_0 = ground-homography warp of f1 by
    the carried pose target at translation-scale s_t.  Byte-identical to p3v2
    warp_base_work's best_cam for the stored s_t index."""
    K = intrinsics_native()
    Kinv = np.linalg.inv(K)
    grid = _target_grid(CAMERA_H, CAMERA_W)
    H = regime_homography_ground(np.asarray(target6, np.float64), K, Kinv, float(s_t))
    return _to_uint8(warp_rgb(np.asarray(f1_u8, np.float64), H, grid))
'''

INFLATE_RUNNER = '''from __future__ import annotations

import json
import struct
import sys
from pathlib import Path, PurePosixPath

import brotli
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ddm_r7_token_coder import decode_token_codes
from ddm_tr1_runtime import (
    _encode_tokens,
    build_packet,
    parse_packet,
    render_frame1_camera_uint8,
)
from pfs1_warp_receiver import ST_GRID, warp_base_f0

MAGIC = b"PFS1WPB1"


def parse_pose_warp(payload: bytes):
    if payload[:8] != MAGIC:
        raise SystemExit("pose_warp magic differs")
    off = 8
    (n_pairs,) = struct.unpack_from("<I", payload, off); off += 4
    (l1,) = struct.unpack_from("<I", payload, off); off += 4
    tp_coded = payload[off:off + l1]; off += l1
    (l2,) = struct.unpack_from("<I", payload, off); off += 4
    st_coded = payload[off:off + l2]; off += l2
    tp = np.frombuffer(brotli.decompress(tp_coded),
                       dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    st_idx = np.asarray(decode_token_codes(st_coded), dtype=np.int64).reshape(-1)[:n_pairs]
    return n_pairs, tp, st_idx


def main() -> None:
    archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
    names = [row.strip() for row in
             names_file.read_text().splitlines() if row.strip()]
    if names != ["0.mkv"]:
        raise SystemExit("this receiver serves exactly the custodied 0.mkv")
    manifest = json.loads((archive_dir / "manifest.json").read_text())

    codes = decode_token_codes((archive_dir / "state/tokens.dr7t").read_bytes())
    token_payload = _encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8))
    packet_bytes = build_packet(manifest["tr1_metadata"], {
        "tokens": token_payload,
        "lotto_renderer": (archive_dir / "state/renderer.sec").read_bytes(),
        "selector": (archive_dir / "state/selector.sec").read_bytes(),
        "pose_stub": (archive_dir / "state/pose_stub.sec").read_bytes(),
    })
    packet = parse_packet(packet_bytes)

    if manifest["frame0_policy"] != "warp_base":
        raise SystemExit("unknown frame0 policy for pfs1 warp receiver")
    n_pairs, tp, st_idx = parse_pose_warp(
        (archive_dir / "state/pose_warp.stp").read_bytes())
    if int(packet.selector["num_pairs"]) != n_pairs:
        raise SystemExit("pose_warp n_pairs differs from selector")
    st_vals = np.asarray(ST_GRID, np.float64)

    name = PurePosixPath(names[0])
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit("unsafe video name")
    target = output_dir / name.with_suffix(".raw")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as sink:
        for i in range(n_pairs):
            f1 = render_frame1_camera_uint8(packet, i)
            f0 = warp_base_f0(f1, tp[i], float(st_vals[st_idx[i]]))
            sink.write(f0.tobytes())
            sink.write(f1.tobytes())


if __name__ == "__main__":
    main()
'''


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _refuse_tmp(path: Path) -> None:
    s = str(path)
    if any(s.startswith(p) for p in ("/tmp/", "/var/tmp/", "/private/tmp/",
                                     "/private/var/tmp/")):
        raise SystemExit(f"{path!r} is a /tmp-class path; use SSD/repo tier.")


def deterministic_stored_zip(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    if tuple(members) != COMPOSED_MEMBERS:
        raise SystemExit("composed member order differs")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED,
                         allowZip64=False) as archive:
        for name_ in COMPOSED_MEMBERS:
            info = zipfile.ZipInfo(name_, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, members[name_])
    return stream.getvalue()


def _vendor(src: Path, dst: Path, old: str, new: str) -> dict[str, str]:
    text = src.read_text()
    if old not in text:
        raise SystemExit(f"vendoring anchor not found in {src}")
    patched = text.replace(old, new, 1)
    dst.write_text(patched)
    return {"source": str(src), "source_sha256": _sha(src.read_bytes()),
            "vendored_sha256": _sha(dst.read_bytes()), "patch": f"{old!r} -> {new!r}"}


def load_targets(n: int) -> np.ndarray:
    bundle = json.loads(TARGETS.read_text())
    if bundle.get("output_dimension") != 6:
        raise SystemExit("pose target bundle is not 6-dim")
    rows = bundle["rows"]
    return np.stack([np.asarray(rows[i]["center"], np.float64) for i in range(n)], 0)


def _load_packet():
    """Parse the pb1 seg-endpoint archive and return its rebuilt TR1 packet."""
    sys.path.insert(0, str(REPO / "src"))
    from tac.optimization import ddm_tr1_runtime as rt
    parsed = rt.parse_archive(SEG_ARCHIVE.read_bytes())
    packet_bytes = rt.reemit_packet(parsed.packet)
    packet = rt.parse_packet(packet_bytes)
    return rt, parsed, packet_bytes, packet


def _solve_jsonl(work: Path) -> Path:
    return work / "d1_warp_solve.partial.jsonl"


def encode_pose_warp(tp_f16: np.ndarray, st_idx: np.ndarray) -> tuple[bytes, dict]:
    """pose_warp.stp = MAGIC + n + len(tp_coded) + tp_coded + len(st_coded) + st_coded."""
    import brotli

    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_r7_token_coder import decode_token_codes, encode_token_codes

    n = tp_f16.shape[0]
    tp_coded = brotli.compress(np.ascontiguousarray(tp_f16, dtype=np.float16).tobytes(),
                               quality=11)
    codes = np.ascontiguousarray(st_idx, dtype=np.uint8).reshape(n, 1, 1, 1)
    st_coded = encode_token_codes(codes, levels=len(ST_GRID), codec="auto")
    rb_tp = np.frombuffer(brotli.decompress(tp_coded), dtype=np.float16).reshape(n, 6)
    if not np.array_equal(rb_tp, tp_f16):
        raise SystemExit("tp float16 roundtrip differs")
    rb_st = np.asarray(decode_token_codes(st_coded), dtype=np.int64).reshape(-1)[:n]
    if not np.array_equal(rb_st, st_idx.astype(np.int64)):
        raise SystemExit("s_t index roundtrip differs")
    payload = (POSE_WARP_MAGIC + struct.pack("<I", n)
               + struct.pack("<I", len(tp_coded)) + tp_coded
               + struct.pack("<I", len(st_coded)) + st_coded)
    meta = {"pose_warp_bytes": len(payload), "tp_coded_bytes": len(tp_coded),
            "st_coded_bytes": len(st_coded), "tp_dtype": "float16",
            "tp_raw_bytes": int(np.ascontiguousarray(tp_f16, np.float16).nbytes)}
    return payload, meta


def contribution(d_pose_mean: float) -> float:
    return float(np.sqrt(10.0 * d_pose_mean))


# --------------------------------------------------------------------------- #
# solve: fit s_t on the archive's OWN shipped f1 (float16 targets); n600 chunked
# --------------------------------------------------------------------------- #
def run_solve(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    import torch
    torch.set_num_threads(4)

    import ddm_p3v2_optimal_form_pose_resolve as p3v2  # warp_base_work + PoseNet
    rt, _parsed, _pb, packet = _load_packet()

    posenet, _mods = p3v2.load_posenet()
    n = int(args.n_pairs)
    if int(packet.selector["num_pairs"]) != n:
        raise SystemExit(f"selector num_pairs != {n}")
    targets = load_targets(n)
    tp_f16 = targets.astype(np.float16).astype(np.float64)  # SHIPPED quantization

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    jl = _solve_jsonl(work)
    _refuse_tmp(jl)
    cache: dict[int, dict] = {}
    if jl.exists() and args.resume:
        for ln in jl.read_text().splitlines():
            if ln.strip():
                rr = json.loads(ln)
                cache[int(rr["pair"])] = rr
        print(f"[solve] resume: {len(cache)} cached", flush=True)
    fj = open(jl, "a")  # noqa: SIM115
    t0 = time.time()
    for i in range(n):
        if i in cache:
            continue
        if args.max_seconds and (time.time() - t0) > args.max_seconds:
            print(f"[solve] --max-seconds at pair {i}; {len(cache)} done; re-run --resume",
                  flush=True)
            fj.close()
            raise SystemExit(2)
        f1 = rt.render_frame1_camera_uint8(packet, i)   # SHIPPED f1
        _base, s_t, dpose = p3v2.warp_base_work(posenet, f1, tp_f16[i], 192, 256)
        rr = {"pair": i, "s_t_idx": int(ST_GRID.index(s_t)), "d_pose": float(dpose)}
        fj.write(json.dumps(rr) + "\n")
        fj.flush()
        os.fsync(fj.fileno())
        cache[i] = rr
        if i % 20 == 0 or i == n - 1:
            rm = float(np.mean([cache[k]["d_pose"] for k in cache]))
            print(f"  [solve {i:3d}] warp d_pose={dpose:.4f} s_t={s_t} "
                  f"(mean={rm:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    fj.close()
    dps = np.asarray([cache[i]["d_pose"] for i in range(n)], np.float64)
    receipt = {
        "schema": "ddm_pfs1_d1_solve.v1", "n_pairs": n,
        "axis": "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE",
        "score_claim": False, "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "carrier": "warp-base on the SHIPPED token-render f1 (float16 targets)",
        "d_pose_mean": float(dps.mean()), "d_pose_median": float(np.median(dps)),
        "d_pose_max": float(dps.max()), "d_pose_min": float(dps.min()),
        "pose_contribution": contribution(float(dps.mean())),
        "note": "re-solved on the pb1-archive OWN f1 (NOT the stale ct1 FRAME_ROOT). "
                "byte-identity to the receiver holds by construction.",
    }
    (work / "d1_solve_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1), flush=True)


# --------------------------------------------------------------------------- #
# build: compose archive v3 (byte-identity positive control on shipped f1)
# --------------------------------------------------------------------------- #
def build(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO))  # for tools.measure_pose_warp_dseg imports
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_r7_token_coder import decode_token_codes, encode_token_codes

    work = args.work_dir
    _refuse_tmp(work)
    sub = work / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    n = int(args.n_pairs)

    rt, parsed, packet_bytes, packet = _load_packet()
    tr1_bytes = SEG_ARCHIVE.read_bytes()
    codes = np.ascontiguousarray(parsed.packet.token_codes, dtype=np.uint8)

    dr7t = encode_token_codes(codes, levels=16, codec="smevr")
    rb = decode_token_codes(dr7t)
    if not np.array_equal(np.asarray(rb, dtype=np.uint8), codes):
        raise SystemExit("SMEVR roundtrip differs from endpoint codes")

    token_payload = rt._encode_tokens(codes)
    recon = rt.build_packet(parsed.packet.metadata, {
        "tokens": token_payload,
        "lotto_renderer": parsed.packet.section_payloads[1],
        "selector": parsed.packet.section_payloads[2],
        "pose_stub": parsed.packet.section_payloads[3],
    })
    if recon != packet_bytes:
        raise SystemExit("reconstructed TR1 packet is NOT byte-identical")
    if int(packet.selector["num_pairs"]) != n:
        raise SystemExit(f"selector num_pairs != {n}")

    # --- s_t indices from the solve (on the shipped f1) + SHIPPED float16 targets
    jl = _solve_jsonl(work)
    if not jl.exists():
        raise SystemExit(f"missing solve jsonl (run --mode solve first): {jl}")
    per: dict[int, int] = {}
    for ln in jl.read_text().splitlines():
        if ln.strip():
            rr = json.loads(ln)
            per[int(rr["pair"])] = int(rr["s_t_idx"])
    missing = [i for i in range(n) if i not in per]
    if missing:
        raise SystemExit(f"solve jsonl missing {len(missing)} pairs: {missing[:5]}")
    st_idx = np.asarray([per[i] for i in range(n)], dtype=np.uint8)
    tp_f16 = load_targets(n).astype(np.float16)
    pose_warp, pw_meta = encode_pose_warp(tp_f16, st_idx)

    # --- vendored-warp byte-identity positive control vs the p3v2 engine (shipped f1)
    recv_path = sub / "pfs1_warp_receiver.py"
    recv_path.write_text(PFS1_WARP_RECEIVER)
    sys.path.insert(0, str(sub))
    import importlib
    recv = importlib.import_module("pfs1_warp_receiver")
    from tools.measure_pose_warp_dseg import (
        _target_grid,
        intrinsics_at,
        regime_homography,
    )
    from tools.measure_screw_warp_through_R import _to_uint8, warp_rgb
    K = intrinsics_at(CAMERA_W, CAMERA_H)
    Kinv = np.linalg.inv(K)
    grid = _target_grid(CAMERA_H, CAMERA_W)
    tp_f16_f64 = tp_f16.astype(np.float64)
    sample = list(range(0, n, max(1, n // 24)))[:24]
    max_warp_mismatch = 0
    for i in sample:
        f1 = rt.render_frame1_camera_uint8(packet, i)
        s_t = ST_GRID[int(st_idx[i])]
        H = regime_homography(tp_f16_f64[i], K, Kinv, (float(s_t), 0.0, 0.0), "ground")
        f0_orig = _to_uint8(warp_rgb(np.asarray(f1, np.float64), H, grid))
        f0_recv = recv.warp_base_f0(f1, tp_f16_f64[i], float(s_t))
        max_warp_mismatch = max(max_warp_mismatch, int(
            np.abs(f0_orig.astype(np.int64) - f0_recv.astype(np.int64)).max()))
    if max_warp_mismatch != 0:
        raise SystemExit(f"vendored warp != original engine (max {max_warp_mismatch})")

    # --- compose archive v3
    manifest = {
        "schema": "ddm_pfs1_composed_archive.v3_warp",
        "frame0_policy": "warp_base",
        "pose_carrier": "ground_homography_warp_of_f1_by_target_plus_st_index",
        "st_grid": ST_GRID,
        "tr1_metadata": dict(parsed.packet.metadata),
        "tokens_sha256": _sha(dr7t),
        "renderer_sha256": _sha(parsed.packet.section_payloads[1]),
        "selector_sha256": _sha(parsed.packet.section_payloads[2]),
        "pose_stub_sha256": _sha(parsed.packet.section_payloads[3]),
        "pose_warp_sha256": _sha(pose_warp),
        "tr1_packet_sha256": _sha(packet_bytes),
        "research_only": True, "score_claim": False, "pointer_moved": False,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True,
                                separators=(",", ":")).encode()
    archive = deterministic_stored_zip({
        "manifest.json": manifest_bytes,
        "state/tokens.dr7t": dr7t,
        "state/renderer.sec": bytes(parsed.packet.section_payloads[1]),
        "state/selector.sec": bytes(parsed.packet.section_payloads[2]),
        "state/pose_stub.sec": bytes(parsed.packet.section_payloads[3]),
        "state/pose_warp.stp": pose_warp,
    })

    # parse-back of the whole composed archive
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert tuple(i.filename for i in z.infolist()) == COMPOSED_MEMBERS
        rb_manifest = json.loads(z.read("manifest.json"))
        rb_codes = decode_token_codes(z.read("state/tokens.dr7t"))
        rb_packet_bytes = rt.build_packet(rb_manifest["tr1_metadata"], {
            "tokens": rt._encode_tokens(np.ascontiguousarray(rb_codes, dtype=np.uint8)),
            "lotto_renderer": z.read("state/renderer.sec"),
            "selector": z.read("state/selector.sec"),
            "pose_stub": z.read("state/pose_stub.sec"),
        })
        rb_pw = z.read("state/pose_warp.stp")
    if rb_packet_bytes != packet_bytes:
        raise SystemExit("composed-archive receiver reconstruction differs")
    if rb_pw != pose_warp:
        raise SystemExit("pose_warp member roundtrip differs")
    rt.parse_packet(rb_packet_bytes)
    print("[consumption] DR7T roundtrip exact; TR1 packet BYTE-IDENTICAL; vendored "
          f"warp byte-identical to the engine on {len(sample)} shipped-f1 pairs "
          "(max_abs 0)", flush=True)

    (sub / "archive.zip").write_bytes(archive)
    (sub / "inflate.sh").write_text(INFLATE_SH)
    (sub / "inflate_runner.py").write_text(INFLATE_RUNNER)
    shutil.copy2(REPO / "src/tac/optimization/ddm_tr1_runtime.py",
                 sub / "ddm_tr1_runtime.py")
    vend = [
        _vendor(REPO / "src/tac/optimization/repair_entropy_coder_runtime_adapters.py",
                sub / "repair_entropy_coder_runtime_adapters.py",
                "from tac.repo_io import sha256_bytes",
                "from hashlib import sha256 as _sha256\n\n\n"
                "def sha256_bytes(data: bytes) -> str:\n"
                "    return _sha256(data).hexdigest()"),
        _vendor(REPO / "experiments/ddm_r7_token_coder.py",
                sub / "ddm_r7_token_coder.py",
                "from tac.optimization.repair_entropy_coder_runtime_adapters import (",
                "from repair_entropy_coder_runtime_adapters import ("),
    ]

    members = {
        "manifest.json": len(manifest_bytes),
        "state/tokens.dr7t": len(dr7t),
        "state/renderer.sec": len(parsed.packet.section_payloads[1]),
        "state/selector.sec": len(parsed.packet.section_payloads[2]),
        "state/pose_stub.sec": len(parsed.packet.section_payloads[3]),
        "state/pose_warp.stp": len(pose_warp),
    }
    # solve d_pose (advisory prediction; eval MEASURES)
    solve = json.loads((work / "d1_solve_receipt.json").read_text())
    rate = 25.0 * len(archive) / 37_545_489
    receipt = {
        "schema": SCHEMA, "stage": "build",
        "archive_bytes": len(archive), "archive_sha256": _sha(archive),
        "members": members, "pose_warp_meta": pw_meta,
        "pb1_pose_tpgn_replaced_coded_bytes": PB1_POSE_TPGN_CODED,
        "pb1_pose_tpgn_replaced_raw_bytes": PB1_POSE_TPGN_RAW,
        "seg_archive_sha256": _sha(tr1_bytes), "frame0_policy": "warp_base",
        "warp_vendored_vs_original_max_abs_pixel": max_warp_mismatch,
        "instrument_rate_term": rate,
        "seg_archive_path": str(SEG_ARCHIVE),
        "seg_archive_is_default_pb1_base": SEG_ARCHIVE == DEFAULT_SEG_ARCHIVE,
        "instrument_composed_prediction": {
            "note": "seg(pb1 endpoint) + pose(solve d_pose on shipped f1) + rate; "
                    "eval MEASURES seg+pose live.  The 0.38901 seg constant is "
                    "the pb1 p2c endpoint's measured seg term and is VALID ONLY "
                    "for the default base; on any other --seg-archive the seg "
                    "prediction is withheld (None) rather than carried over.",
            "seg_from_pb1": (0.38901 if SEG_ARCHIVE == DEFAULT_SEG_ARCHIVE else None),
            "pose_d_pose_solve": solve["d_pose_mean"],
            "pose_contribution": solve["pose_contribution"],
            "rate": rate,
            "S_pred": ((0.38901 + solve["pose_contribution"] + rate)
                       if SEG_ARCHIVE == DEFAULT_SEG_ARCHIVE else None),
        },
        "vendored": vend, "evidence_axis": "[macOS-CPU advisory]", "score_claim": False,
    }
    (work / "d1_build_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in
                      ("archive_bytes", "archive_sha256", "members",
                       "instrument_composed_prediction")}, indent=1), flush=True)


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
            shutil.copy2(LOCKED_ROOT / "models" / name_, root / "models" / name_)
        (root / "videos").mkdir(exist_ok=True)
        shutil.copy2(UPSTREAM_VIDEO, root / "videos" / "0.mkv")
    free = shutil.disk_usage(root).free
    if free < 8 * (1 << 30):
        raise SystemExit(f"storage preflight: only {free/(1<<30):.1f} GiB free")

    sub_src = work / "submission"
    sub = root / "submissions" / "pfs1"
    if sub.exists():
        shutil.rmtree(sub)
    sub.parent.mkdir(exist_ok=True)
    shutil.copytree(sub_src, sub)

    env = os.environ.copy()
    env["PATH"] = f"{LOCKED_PY.parent}:{env['PATH']}"
    env["TQDM_DISABLE"] = "1"
    t0 = time.time()
    proc = run_in_process_group(
        ["bash", str(root / "evaluate.sh"),
         "--submission-dir", str(sub),
         "--video-names-file", str(root / "public_test_video_names.txt"),
         "--device", "cpu"],
        cwd=root, env=env, capture_output=True, text=True, timeout=5400)
    wall = time.time() - t0
    (work / "d1_eval_stdout.txt").write_text(proc.stdout)
    (work / "d1_eval_stderr.txt").write_text(proc.stderr)
    report_path = sub / "report.txt"
    report = report_path.read_text() if report_path.exists() else None
    receipt = {
        "schema": SCHEMA, "stage": "eval", "returncode": proc.returncode,
        "wall_seconds": wall, "report": report,
        "archive_sha256": _sha((sub_src / "archive.zip").read_bytes()),
        "evidence_axis": "[macOS-CPU advisory - real evaluator, real bytes]",
        "score_claim": False,
        "note": "ADVISORY until the Modal contest-CPU flight (P6 operator-GO)",
    }
    (work / "d1_eval_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(f"rc={proc.returncode} wall={wall:.0f}s", flush=True)
    if report:
        print(report, flush=True)
    else:
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("solve", "build", "eval"), required=True)
    ap.add_argument("--work-dir", type=Path, default=SSD_OUT / "d1")
    ap.add_argument("--seg-archive", type=Path, default=DEFAULT_SEG_ARCHIVE,
                    help="TR1 seg-endpoint archive.zip to recompose onto. The "
                         "warp pose is re-solved on THIS archive's own shipped "
                         "frame_1, so a non-default base is a controlled "
                         "base-swap (same grammar, same pose mechanism).")
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


def main() -> int:
    global SEG_ARCHIVE
    args = parse_args()
    seg = Path(args.seg_archive)
    if not seg.is_file():
        raise SystemExit(f"--seg-archive not a file: {seg}")
    SEG_ARCHIVE = seg
    _refuse_tmp(args.work_dir)
    if args.mode == "solve":
        run_solve(args)
    elif args.mode == "build":
        build(args)
    else:
        run_eval(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
