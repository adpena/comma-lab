#!/usr/bin/env python3
"""Build a REAL byte-closed witness archive for a small-n smoke and prepare it
for the actual upstream inflate.sh -> evaluate.py pipeline.

$0 CPU-only. NO scorer weights ever enter the archive (README:118). The archive
is self-contained: inflate.py regenerates raw frames from archive bytes alone.

Modes
-----
store_raw      : identity witness. Stores both frames of each pair losslessly
                 (zlib). d_seg/d_pose -> ~0, rate huge. De-risks the pipeline +
                 proves parity + GT pair-ordering assumption against the real eval.
v2_det         : v2 DETERMINISTIC CORNER (simplest real version). Stores frame0
                 (canonical content, zlib) + the 6-dof pose sidecar (Quantizr
                 style) + a per-class-pose-warp of frame0 -> frame1 prediction +
                 the integer frame1 residual (zlib). inflate.py warps + adds
                 residual deterministically with integer ops. NO scorer at decode.

Output: a submission_dir with archive.zip + inflate.sh + inflate.py, ready for
`bash upstream/evaluate.sh --submission-dir <dir> --device cpu`.

The build also writes the EXACT render frames it intends (render_n{n}.npy is NOT
shipped; it is the parity oracle: inflate output must byte-equal it).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GT_CACHE = REPO / "experiments" / "results" / "mlx_fleet_gt_cache"
CAM_W, CAM_H, C = 1164, 874, 3
RATE_DENOM = 37_545_489  # bytes of upstream/videos/0.mkv (rate denominator)

# Vertical-band partition kept ONLY for the legacy/diagnostic stratified composite.
SKY_FRAC = 0.20
HOOD_FRAC = 0.78

# ---------------------------------------------------------------------------
# REAL plane-induced-homography warp (replaces the broken constant-roll warp).
#
# DAG FEED-la/lf root cause: the previous warp fed the velocity-scale forward
# pose (~33.6) straight in as a pixel shift -> clip(round(33.6*2),+-6)==6 -> a
# CONSTANT 6px vertical roll on EVERY pair -> PoseNet read ZERO coherent
# ego-motion -> d_pose ~= 190 (= the zero-motion null). MEASURED FIX
# (tools/measure_warp_dpose_through_R.py, n6+n24): the plane-induced homography
# H = K (R - t n^T / d) K^{-1} at the d_pose-OPTIMAL translation scale (s_t~0.16,
# positive geometric ego-motion scale, NOT the near-identity d_seg-optimal scale)
# carries pose: d_pose 190 -> ~12.6 (93% drop). The geometry is a FREE generic
# algorithm (rule-118, inflate.py-expandable); the per-pair 6-DOF pose is COUNTED-
# but-existing; the 3-scalar clip CALIBRATION (s_t,s_r,pitch) is video-derived ->
# COUNTED-but-tiny (stored in the archive payload, NOT hardcoded). EON intrinsics
# (fx=fy=910, cx=582, cy=437 native 1164x874), camera height 1.22 m.
NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
CAMERA_HEIGHT_M = 1.22
# Default calibration = the MEASURED d_pose-optimal (advisory; stored in payload
# so decode is self-contained and a future build can re-fit per clip).
DEFAULT_CALIB = (0.16, 0.0, -0.07)  # (s_t, s_r, pitch)  [advisory; n24 d_pose-fit]


def _native_K() -> np.ndarray:
    return np.array([[NATIVE_FX, 0.0, NATIVE_CX],
                     [0.0, NATIVE_FY, NATIVE_CY],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _expmap_so3(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    Kx = np.array([[0.0, -omega[2], omega[1]],
                   [omega[2], 0.0, -omega[0]],
                   [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + Kx
    return (np.eye(3) + (np.sin(theta) / theta) * Kx
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (Kx @ Kx))


def _pose_homography(pose6: np.ndarray, s_t: float, s_r: float, pitch: float) -> np.ndarray:
    K = _native_K()
    Kinv = np.linalg.inv(K)
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)  # (x,y,z=fwd)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    M = R - np.outer(t, n) / CAMERA_HEIGHT_M
    return K @ M @ Kinv


def _warp_rgb_bilinear(src_hwc: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Inverse-warp (H,W,3) uint8 by homography H, bilinear, persist-fallback."""
    Hh, Ww, Ch = src_hwc.shape
    srcf = src_hwc.astype(np.float64)
    flat = srcf.reshape(-1, Ch)
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    grid = np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        srch = np.linalg.inv(H) @ grid
        z = srch[2]
        su = srch[0] / z
        sv = srch[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1)
    sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64)
    y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1)
    y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]
    wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]; Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]; Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat)
    out = np.clip(np.round(out.reshape(Hh, Ww, Ch)), 0.0, 255.0).astype(np.uint8)
    return out


def _zlib(b: bytes) -> bytes:
    return zlib.compress(b, level=9)


def load_gt(n: int):
    p = GT_CACHE / f"gt_n{n}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing GT cache {p}")
    d = np.load(p)
    return (
        np.ascontiguousarray(d["gt_f0"]),  # (n,H,W,3) uint8
        np.ascontiguousarray(d["gt_f1"]),
        np.ascontiguousarray(d["gt_poses"]).astype(np.float64),  # (n,6)
    )


# ---------------------------------------------------------------------------
# Per-class pose warp (deterministic integer-friendly). Shared by builder and
# the generated inflate.py. Kept tiny + dependency-free (numpy only).
# ---------------------------------------------------------------------------
def warp_frame0_to_pred1(f0: np.ndarray, pose6: np.ndarray, calib=DEFAULT_CALIB) -> np.ndarray:
    """Predict frame1 from frame0 via the REAL plane-induced homography warp.

    f0: (H,W,3) uint8. pose6: (6,) PoseNet 6-vector. calib: (s_t, s_r, pitch)
    clip calibration (stored in the payload, COUNTED-tiny). Returns (H,W,3) uint8.

    Whole-frame ground-plane homography H = K (R - t n^T / d) K^{-1} at the
    d_pose-OPTIMAL translation scale. MEASURED to carry pose (d_pose 190 -> ~12.6,
    93% drop; tools/measure_warp_dpose_through_R.py). Replaces the broken
    constant-6px-roll (units bug: velocity-scale pose fed as a pixel shift ->
    constant roll -> zero ego-motion -> d_pose 190). Deterministic, no scorer.

    NOTE (measured): the d_pose-optimal whole-frame warp REDUCES d_pose but
    DEGRADES d_seg (the argmax-optimal warp is near-identity; the two terms want
    opposite scales). For an EXACT v2_det witness this is fine (the stored residual
    reconstructs gt_f1 exactly -> d_seg/d_pose = 0); the warp's only job here is to
    SHRINK that residual. For the lossy v2_warp arm, prefer the already-built
    stored pose sidecar for d_pose (the warp alone is dominated).
    """
    s_t, s_r, pitch = calib
    H = _pose_homography(np.asarray(pose6, dtype=np.float64), s_t, s_r, pitch)
    return _warp_rgb_bilinear(f0, H)


def build_render(mode: str, f0: np.ndarray, f1: np.ndarray, poses: np.ndarray, jpeg_q: int = 40,
                 calib=DEFAULT_CALIB):
    """Return (render_frames (2n,H,W,3) uint8 in pair order, archive_payload dict).

    render_frames is the EXACT bytes inflate.py must reproduce (parity oracle).
    archive_payload is what gets zipped (self-contained, no scorer).
    """
    n = f0.shape[0]
    H, W = f0.shape[1:3]
    render = np.empty((2 * n, H, W, C), dtype=np.uint8)
    calib_arr = np.asarray(calib, dtype=np.float64)  # (3,) s_t,s_r,pitch (COUNTED-tiny)

    if mode == "store_raw":
        # identity: ship both frames exactly.
        for i in range(n):
            render[2 * i] = f0[i]
            render[2 * i + 1] = f1[i]
        payload = {
            "mode": b"store_raw",
            "n": n,
            "f0": _zlib(f0.tobytes()),
            "f1": _zlib(f1.tobytes()),
        }
        return render, payload

    if mode == "v2_det":
        # v2 deterministic corner (simplest real): store f0 + pose, warp->pred1,
        # store integer residual r = f1 - pred1 (mod 256). Decode: pred1 + r.
        pred1 = np.empty_like(f1)
        for i in range(n):
            pred1[i] = warp_frame0_to_pred1(f0[i], poses[i], tuple(calib_arr))
        resid = (f1.astype(np.int16) - pred1.astype(np.int16)).astype(np.uint8)  # wraps mod 256
        for i in range(n):
            render[2 * i] = f0[i]
            render[2 * i + 1] = f1[i]  # exact f1 recovered = pred1 + resid (mod 256)
        # pose sidecar: store as float32 (24 bytes/pair raw; tiny). Quantizr style
        # would zlib + quantize further; float32 is the honest simple version.
        payload = {
            "mode": b"v2_det",
            "n": n,
            "f0": _zlib(f0.tobytes()),
            "poses": _zlib(poses.astype(np.float32).tobytes()),
            "calib": _zlib(calib_arr.tobytes()),  # 3 float64 (COUNTED-tiny)
            "resid": _zlib(resid.tobytes()),
        }
        return render, payload

    if mode == "v2_warp":
        # LOSSY deterministic corner probe: store f0 (canonical) + pose, set
        # f1 = pure per-class-pose-warp(f0) with NO residual. Near-zero marginal
        # bytes per pair (just the pose). Measures the distortion the
        # deterministic warp ALONE incurs (the real question: does warping carry
        # the scene well enough for d_seg/d_pose). f0 stored lossless here; the
        # n600 corner would replace f0 with a coded keyframe / shared canonical.
        for i in range(n):
            pred1 = warp_frame0_to_pred1(f0[i], poses[i], tuple(calib_arr))
            render[2 * i] = f0[i]
            render[2 * i + 1] = pred1
        payload = {
            "mode": b"v2_warp",
            "n": n,
            "f0": _zlib(f0.tobytes()),
            "poses": _zlib(poses.astype(np.float32).tobytes()),
            "calib": _zlib(calib_arr.tobytes()),  # 3 float64 (COUNTED-tiny)
        }
        return render, payload

    if mode == "store_jpeg":
        # LOSSY per-pair storage with a real codec (JPEG). Bounds the BEST simple
        # lossy per-pair storage: isolates whether the RATE half is solvable by
        # coding (independent of any warp). PIL is an external lib (README:118,
        # free). render oracle = decode(encode(frame)) so parity holds.
        from PIL import Image
        blobs_idx = io.BytesIO()
        for i in range(n):
            for k, fr in enumerate((f0[i], f1[i])):
                b = io.BytesIO()
                Image.fromarray(fr).save(b, format="JPEG", quality=jpeg_q)
                jb = b.getvalue()
                blobs_idx.write(struct.pack("<I", len(jb)))
                blobs_idx.write(jb)
                dec = np.asarray(Image.open(io.BytesIO(jb)).convert("RGB"), dtype=np.uint8)
                render[2 * i + k] = dec
        payload = {
            "mode": b"store_jpeg",
            "n": n,
            "jpegs": _zlib(blobs_idx.getvalue()),
        }
        return render, payload

    raise ValueError(f"unknown mode {mode}")


def pack_archive_bytes(payload: dict) -> bytes:
    """Serialize payload dict to a single self-describing blob (the archive
    member). Grammar: JSON header (lengths + scalars) + concatenated blobs."""
    blobs = []
    header = {"mode": payload["mode"].decode(), "n": int(payload["n"]), "sections": []}
    for key, val in payload.items():
        if key in ("mode", "n"):
            continue
        offset = sum(len(b) for b in blobs)
        blobs.append(val)
        header["sections"].append({"key": key, "offset": offset, "length": len(val)})
    hdr = json.dumps(header).encode()
    out = io.BytesIO()
    out.write(struct.pack("<I", len(hdr)))
    out.write(hdr)
    for b in blobs:
        out.write(b)
    return out.getvalue()


INFLATE_PY = r'''#!/usr/bin/env python3
"""Generated v2 witness inflate.py. Self-contained, integer-deterministic.
NO scorer weights. Reads archive_dir/witness.bin -> writes inflated_dir/0.raw.
"""
from __future__ import annotations
import argparse, io, json, struct, zlib
from pathlib import Path
import numpy as np

CAM_W, CAM_H, C = 1164, 874, 3
# EON intrinsics (native 1164x874) + camera height. The plane-induced homography
# warp is a FREE generic algorithm (rule-118); the 3-scalar calib is read from the
# payload (COUNTED-tiny), the per-pair 6-DOF pose is COUNTED-but-existing.
NATIVE_FX = NATIVE_FY = 910.0
NATIVE_CX, NATIVE_CY = 582.0, 437.0
CAMERA_HEIGHT_M = 1.22


def _native_K():
    return np.array([[NATIVE_FX, 0.0, NATIVE_CX],
                     [0.0, NATIVE_FY, NATIVE_CY],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _expmap_so3(omega):
    theta = float(np.linalg.norm(omega))
    Kx = np.array([[0.0, -omega[2], omega[1]],
                   [omega[2], 0.0, -omega[0]],
                   [-omega[1], omega[0], 0.0]], dtype=np.float64)
    if theta < 1e-12:
        return np.eye(3) + Kx
    return (np.eye(3) + (np.sin(theta) / theta) * Kx
            + ((1.0 - np.cos(theta)) / (theta * theta)) * (Kx @ Kx))


def warp_frame0_to_pred1(f0, pose6, calib):
    s_t, s_r, pitch = float(calib[0]), float(calib[1]), float(calib[2])
    K = _native_K(); Kinv = np.linalg.inv(K)
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    R = _expmap_so3(s_r * np.array([pose6[3], pose6[4], pose6[5]], dtype=np.float64))
    n = np.array([0.0, -np.cos(pitch), -np.sin(pitch)], dtype=np.float64)
    Hm = K @ (R - np.outer(t, n) / CAMERA_HEIGHT_M) @ Kinv
    Hh, Ww, Ch = f0.shape
    flat = f0.astype(np.float64).reshape(-1, Ch)
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh))
    grid = np.stack([us.ravel(), vs.ravel(), np.ones(Hh * Ww)], 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        srch = np.linalg.inv(Hm) @ grid
        z = srch[2]; su = srch[0] / z; sv = srch[1] / z
    valid = (np.isfinite(su) & np.isfinite(sv) & (z > 0)
             & (su >= 0) & (su <= Ww - 1) & (sv >= 0) & (sv <= Hh - 1))
    su_c = np.clip(su, 0.0, Ww - 1); sv_c = np.clip(sv, 0.0, Hh - 1)
    x0 = np.floor(su_c).astype(np.int64); y0 = np.floor(sv_c).astype(np.int64)
    x1 = np.minimum(x0 + 1, Ww - 1); y1 = np.minimum(y0 + 1, Hh - 1)
    wx = (su_c - x0)[:, None]; wy = (sv_c - y0)[:, None]
    Ia = flat[y0 * Ww + x0]; Ib = flat[y0 * Ww + x1]
    Ic = flat[y1 * Ww + x0]; Id = flat[y1 * Ww + x1]
    top = Ia * (1.0 - wx) + Ib * wx; bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid[:, None], sampled, flat)
    return np.clip(np.round(out.reshape(Hh, Ww, Ch)), 0.0, 255.0).astype(np.uint8)


def parse(blob):
    (hlen,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4:4 + hlen].decode())
    body = blob[4 + hlen:]
    secs = {}
    for s in header["sections"]:
        secs[s["key"]] = zlib.decompress(body[s["offset"]:s["offset"] + s["length"]])
    return header, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive_dir", type=Path)
    ap.add_argument("inflated_dir", type=Path)
    ap.add_argument("video_names_file", type=Path)
    ap.add_argument("--upstream-root", type=Path, default=None)
    a = ap.parse_args()
    blob = (a.archive_dir / "witness.bin").read_bytes()
    header, secs = parse(blob)
    n = int(header["n"]); mode = header["mode"]
    H, W = CAM_H, CAM_W
    if mode != "store_jpeg":
        f0 = np.frombuffer(secs["f0"], dtype=np.uint8).reshape(n, H, W, C)
    if mode == "store_raw":
        f1 = np.frombuffer(secs["f1"], dtype=np.uint8).reshape(n, H, W, C)
    elif mode == "v2_det":
        poses = np.frombuffer(secs["poses"], dtype=np.float32).reshape(n, 6)
        calib = np.frombuffer(secs["calib"], dtype=np.float64).reshape(3)
        resid = np.frombuffer(secs["resid"], dtype=np.uint8).reshape(n, H, W, C)
        f1 = np.empty((n, H, W, C), dtype=np.uint8)
        for i in range(n):
            pred = warp_frame0_to_pred1(f0[i], poses[i], calib)
            f1[i] = (pred.astype(np.int16) + resid[i].astype(np.int16)).astype(np.uint8)
    elif mode == "v2_warp":
        poses = np.frombuffer(secs["poses"], dtype=np.float32).reshape(n, 6)
        calib = np.frombuffer(secs["calib"], dtype=np.float64).reshape(3)
        f1 = np.empty((n, H, W, C), dtype=np.uint8)
        for i in range(n):
            f1[i] = warp_frame0_to_pred1(f0[i], poses[i], calib)
    elif mode == "store_jpeg":
        from PIL import Image
        import io as _io
        buf = secs["jpegs"]; off = 0
        frames = []
        for _ in range(2 * n):
            (ln,) = struct.unpack("<I", buf[off:off + 4]); off += 4
            jb = buf[off:off + ln]; off += ln
            frames.append(np.asarray(Image.open(_io.BytesIO(jb)).convert("RGB"), dtype=np.uint8))
        out = np.stack(frames).reshape(2 * n, H, W, C)
        f0 = out[0::2]; f1 = out[1::2]
    else:
        raise ValueError(mode)
    if mode != "store_jpeg":
        out = np.empty((2 * n, H, W, C), dtype=np.uint8)
        out[0::2] = f0
        out[1::2] = f1
    names = [ln.strip() for ln in a.video_names_file.read_text().splitlines() if ln.strip()]
    rel = names[0]
    raw_path = a.inflated_dir / (Path(rel).with_suffix(".raw"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(out.tobytes())
    print(f"Inflated {mode} n={n} -> {raw_path} ({out.shape})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

INFLATE_SH = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    'SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'python3 "$SELF_DIR/inflate.py" "$@"\n'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["store_raw", "v2_det", "v2_warp", "store_jpeg"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--jpeg-q", type=int, default=40, help="JPEG quality for store_jpeg mode")
    ap.add_argument("--calib-st", type=float, default=DEFAULT_CALIB[0],
                    help="warp translation scale s_t (d_pose-optimal default; see "
                         "tools/measure_warp_dpose_through_R.py)")
    ap.add_argument("--calib-sr", type=float, default=DEFAULT_CALIB[1], help="warp rotation scale s_r")
    ap.add_argument("--calib-pitch", type=float, default=DEFAULT_CALIB[2], help="road-plane pitch (rad)")
    ap.add_argument("--out", type=Path, required=True, help="submission_dir to create")
    a = ap.parse_args()

    f0, f1, poses = load_gt(a.n)
    calib = (a.calib_st, a.calib_sr, a.calib_pitch)
    render, payload = build_render(a.mode, f0, f1, poses, jpeg_q=a.jpeg_q, calib=calib)
    archive_bytes = pack_archive_bytes(payload)

    sub = a.out
    sub.mkdir(parents=True, exist_ok=True)
    # write archive members to a temp archive/ dir then zip
    archdir = sub / "_archive_src"
    archdir.mkdir(exist_ok=True)
    (archdir / "witness.bin").write_bytes(archive_bytes)
    # zip (store the single member). Use python zipfile for determinism.
    import zipfile
    zip_path = sub / "archive.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(archdir / "witness.bin", arcname="witness.bin")
    (sub / "inflate.py").write_text(INFLATE_PY)
    (sub / "inflate.sh").write_text(INFLATE_SH)
    (sub / "inflate.sh").chmod(0o755)

    # parity oracle (NOT shipped)
    render_path = sub / f"render_oracle_n{a.n}.npy"
    np.save(render_path, render)

    zsize = zip_path.stat().st_size
    rate_smalln = zsize / RATE_DENOM
    meta = {
        "mode": a.mode,
        "n": a.n,
        "archive_zip_bytes": zsize,
        "witness_bin_bytes": len(archive_bytes),
        "rate_smalln": rate_smalln,
        "rate_denom": RATE_DENOM,
        "render_sha256": hashlib.sha256(render.tobytes()).hexdigest(),
        "render_oracle_path": str(render_path),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # n600 byte projection: per-pair coded bytes (excl. fixed header) * 600
        "per_pair_archive_bytes_est": zsize / a.n,
        "n600_archive_bytes_proj": zsize / a.n * 600,
        "n600_rate_proj": (zsize / a.n * 600) / RATE_DENOM,
    }
    (sub / "build_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
