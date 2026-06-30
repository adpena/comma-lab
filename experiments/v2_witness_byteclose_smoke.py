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

# Canonical comma10k SegNet class spatial signatures are NOT needed at decode.
# The per-class warp uses 3 strata keyed by a fixed vertical band partition of
# the CAMERA frame (deterministic, no scorer): a sky band (top), a road band
# (middle/lower), and a hood band (bottom). This is a SIMPLE real stratification
# (it is intentionally not the optimal scorer-resolution argmax partition; the
# optimal-form version routes the SegNet argmax, which is a build-gated upgrade).
SKY_FRAC = 0.20   # top 20% rows -> rotation-only (approximated by identity here)
HOOD_FRAC = 0.78  # rows below 78% -> hood -> identity (static)
# middle band [SKY_FRAC, HOOD_FRAC) -> road -> ground-plane shift from pose.


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
def warp_frame0_to_pred1(f0: np.ndarray, pose6: np.ndarray) -> np.ndarray:
    """Predict frame1 from frame0 using a class-stratified pose warp.

    f0: (H,W,3) uint8. pose6: (6,) float (PoseNet 6 scalars; we use a small,
    fixed linear map pose->pixel-shift for the road band). Returns (H,W,3) uint8.

    This is a SIMPLE real warp: road band gets a vertical+horizontal integer
    pixel shift proportional to the forward/yaw pose components; sky+hood bands
    stay identity. Purely deterministic integer ops, no scorer.
    """
    H, W = f0.shape[:2]
    pred = f0.copy()
    sky_end = int(SKY_FRAC * H)
    hood_start = int(HOOD_FRAC * H)
    # forward translation (pose[0..2]) -> small vertical "zoom toward horizon"
    # yaw / lateral (pose[3..5]) -> small horizontal shift. Scale chosen tiny so
    # the residual stays small for nearby frames (20fps, adjacent frames).
    fwd = float(pose6[0]) if pose6.shape[0] > 0 else 0.0
    lat = float(pose6[4]) if pose6.shape[0] > 4 else 0.0
    dy = int(round(np.clip(fwd * 2.0, -6, 6)))   # vertical shift in road band
    dx = int(round(np.clip(lat * 2.0, -6, 6)))   # horizontal shift in road band
    road = f0[sky_end:hood_start]
    if dy != 0 or dx != 0:
        shifted = np.roll(np.roll(road, dy, axis=0), dx, axis=1)
        pred[sky_end:hood_start] = shifted
    return pred


def build_render(mode: str, f0: np.ndarray, f1: np.ndarray, poses: np.ndarray, jpeg_q: int = 40):
    """Return (render_frames (2n,H,W,3) uint8 in pair order, archive_payload dict).

    render_frames is the EXACT bytes inflate.py must reproduce (parity oracle).
    archive_payload is what gets zipped (self-contained, no scorer).
    """
    n = f0.shape[0]
    H, W = f0.shape[1:3]
    render = np.empty((2 * n, H, W, C), dtype=np.uint8)

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
            pred1[i] = warp_frame0_to_pred1(f0[i], poses[i])
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
            pred1 = warp_frame0_to_pred1(f0[i], poses[i])
            render[2 * i] = f0[i]
            render[2 * i + 1] = pred1
        payload = {
            "mode": b"v2_warp",
            "n": n,
            "f0": _zlib(f0.tobytes()),
            "poses": _zlib(poses.astype(np.float32).tobytes()),
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
SKY_FRAC, HOOD_FRAC = 0.20, 0.78


def warp_frame0_to_pred1(f0, pose6):
    H, W = f0.shape[:2]
    pred = f0.copy()
    sky_end = int(SKY_FRAC * H); hood_start = int(HOOD_FRAC * H)
    fwd = float(pose6[0]) if pose6.shape[0] > 0 else 0.0
    lat = float(pose6[4]) if pose6.shape[0] > 4 else 0.0
    dy = int(round(min(max(fwd * 2.0, -6), 6)))
    dx = int(round(min(max(lat * 2.0, -6), 6)))
    road = f0[sky_end:hood_start]
    if dy != 0 or dx != 0:
        pred[sky_end:hood_start] = np.roll(np.roll(road, dy, axis=0), dx, axis=1)
    return pred


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
        resid = np.frombuffer(secs["resid"], dtype=np.uint8).reshape(n, H, W, C)
        f1 = np.empty((n, H, W, C), dtype=np.uint8)
        for i in range(n):
            pred = warp_frame0_to_pred1(f0[i], poses[i])
            f1[i] = (pred.astype(np.int16) + resid[i].astype(np.int16)).astype(np.uint8)
    elif mode == "v2_warp":
        poses = np.frombuffer(secs["poses"], dtype=np.float32).reshape(n, 6)
        f1 = np.empty((n, H, W, C), dtype=np.uint8)
        for i in range(n):
            f1[i] = warp_frame0_to_pred1(f0[i], poses[i])
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
    ap.add_argument("--out", type=Path, required=True, help="submission_dir to create")
    a = ap.parse_args()

    f0, f1, poses = load_gt(a.n)
    render, payload = build_render(a.mode, f0, f1, poses, jpeg_q=a.jpeg_q)
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
