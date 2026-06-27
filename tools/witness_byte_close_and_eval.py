#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""WITNESS byte-close -> full-output inflate -> d_seg-vs-GT-on-inflated parity -> staged exact eval.

ONE command turns a trained MLX coord-INR WITNESS checkpoint (``witness_ema_mlx.npz`` +
``train_result.json``, produced by ``experiments/train_witness_realized_through_R_mlx.py``)
into a self-contained contest packet and proves it end-to-end ON CPU at $0:

  (a) archive.zip       -- int8+brotli witness blob (the MEASURED rate term st_size).
  (b) inflate.py        -- MLX-FREE (numpy fwd + torch R) FULL-OUTPUT decoder: emits the
                           complete (2*n_pairs, 874, 1164, 3) uint8 ``.raw`` (frame0 AND
                           frame1 for EVERY pair, camera-res, no header) -- the evaluator's
                           expected layout (upstream/evaluate.py + camera_size=(1164,874),
                           seq_len=2). Fixes BH3 blocker #1 (old path decoded only pair-0).
  (c) realized d_seg/d_pose on the INFLATED FRAMES (read back from the .raw) via the FROZEN
      CPU-torch SegNet/PoseNet vs GT -- NOT byte-repro-of-the-generator's-own-frames.
      Fixes BH3 blocker #2 (lossless_parity only proved byte-repro).
  (d) the staged contest-CPU exact-eval command (NOT run here).

WHY MLX-FREE INFLATE (the load-bearing constraint surfaced this build): the witness is
TRAINED in MLX (Apple-only), but the contest-CPU authority axis is Linux x86_64 where MLX
does NOT exist. So inflate.py MUST reconstruct the witness forward in numpy + use torch ONLY
for the R (bicubic up to camera -> round -> uint8), byte-identical to the trainer's verdict R
(``_torch_R_to_camera_uint8``). The numpy MLP forward mirrors ``RGBWitnessMLX.__call__``
exactly (MLX nn.Linear == x @ W.T + b); the deterministic Fourier B is regenerated from the
seed (FREE, rule 118, not stored).

BYTE ACCOUNTING (the rate term the contest actually scores): ``upstream/evaluate.py:63``
measures ``(submission_dir/'archive.zip').stat().st_size``. The blob is int8(params)+brotli,
matching the trainer's ``_quantize_blob_from_flat`` (base params concat one brotli stream +
codes a second), plus a small JSON manifest (dims/scales/shapes/seed) + zip container. ``_B``
is excluded (free deterministic table). The OPTIONAL stored-pose sidecar (scorer_targets,
~5KB fp16) is a counted 4th section for the capstone "d_seg-only witness + stored pose"
vehicle; the AS-TRAINED RGB witness carries pose in its per-(pair,frame) codes (train with
``--w-pose > 0``), so the sidecar defaults OFF (folding it onto a code-pose witness adds dead
bytes -- loud, fail-closed honesty).

AUTHORITY: ``[macOS-CPU advisory] NON-PROMOTABLE``. CPU only (no MPS, no CUDA, no paid eval).
The realized d_seg/d_pose here is the frozen CPU-torch mirror of evaluate.py over the measured
pair subset. The reported S is advisory until the SAME packet runs through
``upstream/evaluate.py`` on contest-compliant Linux x86_64 CPU (the staged command). NO
score/frontier/promotion claim is made; pointer UNMOVED unless a real byte-closed sub-frontier
exact-eval row lands.

Usage:
    .venv/bin/python tools/witness_byte_close_and_eval.py \\
        --ckpt-dir experiments/results/<run>/<arm> \\
        [--max-pairs 4] [--fold-pose-sidecar] [--keep-packet] [--out reports/...json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_witness_realized_through_R_mlx as twr  # noqa: E402  (light top-level imports)

CAMERA_H, CAMERA_W = 874, 1164
RATE_DENOM = 37_545_489.0
_MAGIC = b"WTNS1\x00"  # witness coord-INR carrier v1


def _advisory_axis_label() -> str:
    """Device-truthful advisory authority (DAG FEED-br sister — the axis mislabel). macOS
    CPU-torch is NOT 1:1 with the contest Linux x86_64 CPU runner -> [macOS-CPU advisory];
    only a real Linux x86_64 host earns [contest-CPU advisory]. Always NON-PROMOTABLE here."""
    import platform

    if platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"):
        return "[contest-CPU advisory] NON-PROMOTABLE"
    return "[macOS-CPU advisory] NON-PROMOTABLE"


_AUTHORITY = _advisory_axis_label()
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------
def _load_witness_ckpt(ckpt_dir: Path, use_live: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load (params, config) from a witness run dir. NO-FAKE: missing files raise.

    ``use_live`` (DAG FEED-br sister fix) byte-closes the LIVE-weights checkpoint
    (``witness_live_mlx.npz``) instead of the EMA shadow (``witness_ema_mlx.npz``). The
    EMA shadow can lag the live weights HARD (the 78x ema-lag trap); the trainer saves
    both. We also surface an EMA-LAG warning from the train_result best_verdict.
    """
    npz_name = "witness_live_mlx.npz" if use_live else "witness_ema_mlx.npz"
    npz = ckpt_dir / npz_name
    res = ckpt_dir / "train_result.json"
    if not npz.exists():
        if use_live:
            raise FileNotFoundError(
                f"--use-live requested but {npz} missing (rerun the trainer to save the "
                "LIVE-weights checkpoint, or drop --use-live to byte-close the EMA shadow)."
            )
        raise FileNotFoundError(f"witness checkpoint missing: {npz} (refusing to fabricate; NO-FAKE)")
    # EMA-LAG warning (DAG FEED-br sister): if the train_result records EMA d_seg >> live,
    # the EMA shadow is lagging -> --use-live byte-closes the better arm.
    if res.exists():
        try:
            bv = json.loads(res.read_text()).get("best_verdict", {})
            ema_ds, live_ds = bv.get("d_seg"), bv.get("d_seg_live")
            if isinstance(ema_ds, (int, float)) and isinstance(live_ds, (int, float)) and live_ds > 1e-9:
                ratio = ema_ds / live_ds
                if ratio >= 2.0 and not use_live:
                    print(f"[WARN ema-lag] best EMA d_seg={ema_ds:.6f} >> LIVE d_seg={live_ds:.6f} "
                          f"(lag {ratio:.1f}x) -- byte-closing the LAGGING EMA shadow. Consider "
                          f"--use-live (needs witness_live_mlx.npz).", flush=True)
        except Exception:
            pass
    z = np.load(npz, allow_pickle=False)
    params = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
    cfg: dict[str, Any] = {}
    if res.exists():
        cfg = json.loads(res.read_text()).get("config", {})
    # infer dims from tensors when config is silent
    if "code" not in params:
        raise ValueError("witness npz lacks a 'code' tensor (per-(pair,frame) modulation); NO-FAKE")
    n_codes = int(params["code"].shape[0])
    cfg.setdefault("n_pairs", n_codes // 2)
    cfg.setdefault("mod_dim", int(params["code"].shape[1]))
    cfg.setdefault("hidden_dim", int(params["out.weight"].shape[1]))
    cfg.setdefault("n_hidden", sum(1 for k in params if k.startswith("hidden.") and k.endswith(".weight")))
    in_feat = int(params["in_proj.weight"].shape[1])
    cfg.setdefault("n_fourier", in_feat // 2)
    cfg.setdefault("fourier_sigma", 8.0)
    cfg.setdefault("render_h", 256)
    cfg.setdefault("render_w", 384)
    cfg.setdefault("activation", "relu")
    cfg.setdefault("chroma", True)  # achromatic ablation control arm (chroma=False) MUST round-trip
    cfg.setdefault("hosc_beta", 4.0)
    cfg.setdefault("hosc_omega", 1.0)
    cfg.setdefault("siren_omega", 30.0)  # periodic freq for siren/finer/wire (FINER first-bias is baked into the saved weights)
    cfg.setdefault("wire_scale", 1.0)  # WIRE real-Gabor Gaussian window scale s
    cfg.setdefault("basis", "isotropic")
    cfg.setdefault("fourier_seed", twr._FOURIER_SEED)
    return params, cfg


# ---------------------------------------------------------------------------
# byte-close: int8 + brotli (matches trainer _quantize_blob_from_flat accounting)
# ---------------------------------------------------------------------------
def build_witness_blob(
    params: dict[str, np.ndarray], cfg: dict[str, Any], pose_sidecar: bytes | None
) -> tuple[bytes, dict[str, Any]]:
    """Build the 0.bin witness blob. ``_B`` is NOT stored (free; regen from seed in inflate).

    Layout: magic | u32 manifest_len | manifest_json | u32 base_brotli_len | base_brotli |
            u32 code_brotli_len | code_brotli | u32 pose_len | pose_sidecar(optional)
    The int8 quant uses the trainer's single-source ``_int8_symmetric`` so the byte count
    matches ``quantize_witness_blob`` exactly.
    """
    import brotli

    base_order = [k for k in params if k != "code" and not k.endswith("_B")]
    base_chunks: list[bytes] = []
    shapes: dict[str, list[int]] = {}
    base_scales: dict[str, float] = {}
    for name in base_order:
        a = np.asarray(params[name], dtype=np.float32)
        q, scale = twr._int8_symmetric(a)
        base_chunks.append(q.astype(np.int8).tobytes())
        shapes[name] = list(a.shape)
        base_scales[name] = float(scale)
    base_raw = b"".join(base_chunks)
    base_brotli = brotli.compress(base_raw, quality=11)

    code = np.asarray(params["code"], dtype=np.float32)
    qc, code_scale = twr._int8_symmetric(code)
    code_brotli = brotli.compress(qc.astype(np.int8).tobytes(), quality=11)

    manifest = {
        "format_version": 1,
        "n_pairs": int(cfg["n_pairs"]),
        "n_fourier": int(cfg["n_fourier"]),
        "hidden_dim": int(cfg["hidden_dim"]),
        "n_hidden": int(cfg["n_hidden"]),
        "mod_dim": int(cfg["mod_dim"]),
        "fourier_sigma": float(cfg["fourier_sigma"]),
        "fourier_seed": int(cfg.get("fourier_seed", twr._FOURIER_SEED)),
        "activation": str(cfg["activation"]),
        "chroma": bool(cfg.get("chroma", True)),
        "hosc_beta": float(cfg["hosc_beta"]),
        "hosc_omega": float(cfg["hosc_omega"]),
        "siren_omega": float(cfg.get("siren_omega", 30.0)),
        "wire_scale": float(cfg.get("wire_scale", 1.0)),
        "render_h": int(cfg["render_h"]),
        "render_w": int(cfg["render_w"]),
        "basis": str(cfg["basis"]),
        "camera_h": CAMERA_H,
        "camera_w": CAMERA_W,
        "base_param_order": base_order,
        "base_shapes": shapes,
        "base_scales": base_scales,
        "code_shape": list(code.shape),
        "code_scale": float(code_scale),
        "has_pose_sidecar": bool(pose_sidecar is not None),
        "pose_sidecar_dtype": "float16",
    }
    if str(cfg["basis"]) != "isotropic":
        raise ValueError(
            f"basis={cfg['basis']!r} is NOT byte-closeable (directional uses GT SegNet argmax, "
            "unavailable at decode). Only isotropic witnesses byte-close (NO-FAKE)."
        )
    mj = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    out = io_pack(mj, base_brotli, code_brotli, pose_sidecar)
    breakdown = {
        "n_params": int(sum(np.prod(s) for s in shapes.values())) + int(np.prod(code.shape)),
        "manifest_bytes": len(mj),
        "base_int8_brotli_bytes": len(base_brotli),
        "code_int8_brotli_bytes": len(code_brotli),
        "pose_sidecar_bytes": (len(pose_sidecar) if pose_sidecar else 0),
        "magic_and_prefixes_bytes": len(_MAGIC) + 16,
        "total_0bin_bytes": len(out),
    }
    return out, breakdown


def io_pack(manifest: bytes, base: bytes, code: bytes, pose: bytes | None) -> bytes:
    buf = bytearray()
    buf += _MAGIC
    for chunk in (manifest, base, code, (pose or b"")):
        buf += struct.pack("<I", len(chunk))
        buf += chunk
    return bytes(buf)


# ---------------------------------------------------------------------------
# the MLX-free inflate.py (numpy fwd + torch R). Self-contained string template.
# ---------------------------------------------------------------------------
_INFLATE_PY = r'''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# WITNESS coord-INR inflate -- MLX-FREE (numpy forward + torch R), self-contained.
# Emits the FULL (2*n_pairs, camera_h, camera_w, 3) uint8 .raw (frame0+frame1 per pair),
# byte-identical R to the trainer verdict (_torch_R_to_camera_uint8). No header.
import sys, json, struct
import numpy as np
import brotli
import torch

MAGIC = b"WTNS1\x00"


def _read_blob(path):
    raw = open(path, "rb").read()
    assert raw[:len(MAGIC)] == MAGIC, "bad witness magic"
    off = len(MAGIC)
    out = []
    for _ in range(4):
        (n,) = struct.unpack_from("<I", raw, off); off += 4
        out.append(raw[off:off + n]); off += n
    manifest = json.loads(out[0].decode("utf-8"))
    return manifest, out[1], out[2], out[3]


def _deterministic_fourier_B(n_fourier, sigma, seed):
    rng = np.random.default_rng(int(seed))
    return (rng.standard_normal((2, n_fourier)) * sigma).astype(np.float32)


def _build_render_coords(h, w):
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def _dequant(blob, order, shapes, scales):
    out, off = {}, 0
    flat = np.frombuffer(blob, dtype=np.int8)
    for name in order:
        shp = tuple(shapes[name]); n = int(np.prod(shp))
        out[name] = (flat[off:off + n].astype(np.float32) * float(scales[name])).reshape(shp)
        off += n
    return out


def _act(u, kind, beta, omega, wire_scale=1.0):
    if kind == "hosc":
        return np.tanh(beta * np.sin(omega * u))
    if kind == "siren":  # Sitzmann 2020
        return np.sin(omega * u)
    if kind == "finer":  # Liu CVPR2024 variable-periodic (first-layer bias baked into weights)
        return np.sin(omega * (np.abs(u) + 1.0) * u)
    if kind == "wire":  # Saragadam CVPR2024 real-Gabor sin(omega*u)*exp(-0.5*(s*u)^2)
        return np.sin(omega * u) * np.exp(-0.5 * np.square(wire_scale * u))
    return np.maximum(u, 0.0)


def _witness_forward(p, feats, code_row, n_hidden, hidden_dim, kind, beta, omega, wire_scale=1.0):
    # NUMPY fp32 forward = the DEPLOY ground truth. The MLX **GPU** train forward
    # accumulates matmuls in reduced precision (~0.19/255 RGB, ~500-1670 argmax px/pair vs
    # fp32); numpy/torch fp32 agree to ~1 ULP (0-3 px). The trainer VERDICT mirrors THIS
    # forward (train_witness_realized_through_R_mlx._witness_forward_numpy) so the scored
    # d_seg == the byte-closed d_seg (DAG FEED-br). Keep op-for-op identical.
    def lin(name, x):
        return x @ p[name + ".weight"].T + p[name + ".bias"]
    h = _act(lin("in_proj", feats), kind, beta, omega, wire_scale)
    film = (code_row @ p["film.weight"].T + p["film.bias"]).reshape(n_hidden, 2, hidden_dim)
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]; shift = film[li, 1]
        h = _act(lin(f"hidden.{li}", h) * scale + shift, kind, beta, omega, wire_scale)
    z = h @ p["out.weight"].T + p["out.bias"]
    return (1.0 / (1.0 + np.exp(-z))) * 255.0  # (P,3) float RGB in [0,255]


def _R_to_camera_uint8(rgb_render, render_h, render_w, cam_h, cam_w):
    # render-res float RGB -> bicubic up to camera -> round/clamp -> uint8 (cam_h,cam_w,3).
    x = torch.from_numpy(np.ascontiguousarray(rgb_render.reshape(render_h, render_w, 3)))
    x = x.permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(cam_h, cam_w), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    man, base_b, code_b, _pose = _read_blob(src)
    if man["basis"] != "isotropic":
        raise SystemExit("only isotropic witnesses inflate (directional is not byte-closeable)")
    params = _dequant(brotli.decompress(base_b), man["base_param_order"], man["base_shapes"], man["base_scales"])
    code = (np.frombuffer(brotli.decompress(code_b), dtype=np.int8).astype(np.float32)
            * float(man["code_scale"])).reshape(man["code_shape"])
    B = _deterministic_fourier_B(man["n_fourier"], man["fourier_sigma"], man["fourier_seed"])
    coords = _build_render_coords(man["render_h"], man["render_w"])
    proj = coords @ B
    feats = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)
    n_pairs = int(man["n_pairs"]); n_frames = 2 * n_pairs
    rh, rw = int(man["render_h"]), int(man["render_w"])
    ch, cw = int(man["camera_h"]), int(man["camera_w"])
    nh, hd = int(man["n_hidden"]), int(man["hidden_dim"])
    kind, beta = man["activation"], float(man["hosc_beta"])
    # siren/finer/wire use the periodic freq (siren_omega ~30); hosc uses its own omega. The verdict
    # forward must use the SAME omega the trained _act used or the inflated render diverges.
    omega = float(man.get("siren_omega", 30.0)) if kind in {"siren", "finer", "wire"} else float(man["hosc_omega"])
    wire_scale = float(man.get("wire_scale", 1.0))  # WIRE Gabor window scale (no-op for non-wire)
    chroma = bool(man.get("chroma", True))  # achromatic control arm: replicate BT.601 luma to R=G=B
    with open(dst, "wb") as f:  # stream frame-by-frame: peak RAM = one camera frame
        for fi in range(n_frames):
            rgb = _witness_forward(params, feats, code[fi], nh, hd, kind, beta, omega, wire_scale)
            if not chroma:  # mirror RGBWitnessMLX._apply_chroma (chroma=False) -- MUST match train-time forward
                luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
                rgb = np.concatenate([luma, luma, luma], axis=-1)
            frame = _R_to_camera_uint8(rgb, rh, rw, ch, cw)
            f.write(frame.tobytes())
    print(f"inflated {n_frames} frames ({n_pairs} pairs) -> {dst} "
          f"[{n_frames}x{ch}x{cw}x3 uint8]", flush=True)


if __name__ == "__main__":
    main()
'''

_INFLATE_SH = """#!/usr/bin/env bash
# Witness inflate launcher. Produces <OUTPUT_DIR>/<base>.raw = flat uint8 (N,874,1164,3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$1"; OUTPUT_DIR="$2"; FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"
  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1
  printf "Inflating %s ... " "$line"
  python "${HERE}/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


# ---------------------------------------------------------------------------
# packet assembly: archive.zip (0.bin) + inflate.py + inflate.sh
# ---------------------------------------------------------------------------
def assemble_packet(blob: bytes, packet_dir: Path) -> tuple[Path, int]:
    packet_dir.mkdir(parents=True, exist_ok=True)
    zip_path = packet_dir / "archive.zip"
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, blob)
    (packet_dir / "inflate.py").write_text(_INFLATE_PY)
    sh = packet_dir / "inflate.sh"
    sh.write_text(_INFLATE_SH)
    sh.chmod(0o755)
    return zip_path, int(zip_path.stat().st_size)


# ---------------------------------------------------------------------------
# run inflate (subprocess, exactly as the contest evaluate.sh does)
# ---------------------------------------------------------------------------
def run_inflate(packet_dir: Path, n_pairs_total: int, max_pairs: int | None) -> dict[str, Any]:
    """Unzip archive.zip -> 0.bin, run inflate.py -> 0.raw, validate the FULL output shape.

    ``max_pairs`` (test/speed): when set < n_pairs, a CAPPED 0.bin is written so inflate
    emits only the first ``max_pairs`` pairs (still frame0+frame1 each) -- the mechanism is
    identical; the full archive.zip always encodes ALL codes (the rate term is the full blob).
    """
    import io as _io

    archive_dir = packet_dir / "archive"
    inflated_dir = packet_dir / "inflated"
    archive_dir.mkdir(exist_ok=True)
    inflated_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(packet_dir / "archive.zip") as zf:
        zf.extractall(archive_dir)
    src_bin = archive_dir / "0.bin"

    eval_pairs = n_pairs_total if max_pairs is None else min(int(max_pairs), n_pairs_total)
    if eval_pairs < n_pairs_total:
        # write a capped 0.bin (truncate codes + n_pairs) for a fast end-to-end proof
        raw = src_bin.read_bytes()
        assert raw[: len(_MAGIC)] == _MAGIC
        off = len(_MAGIC)
        parts = []
        for _ in range(4):
            (n,) = struct.unpack_from("<I", raw, off)
            off += 4
            parts.append(raw[off : off + n])
            off += n
        man = json.loads(parts[0].decode())
        import brotli

        code = (
            np.frombuffer(brotli.decompress(parts[2]), dtype=np.int8).astype(np.float32)
            * man["code_scale"]
        ).reshape(man["code_shape"])
        code_cap = code[: 2 * eval_pairs]
        qc, sc = twr._int8_symmetric(code_cap)
        man["n_pairs"] = eval_pairs
        man["code_shape"] = list(code_cap.shape)
        man["code_scale"] = float(sc)
        mj = json.dumps(man, separators=(",", ":")).encode()
        capped = io_pack(mj, parts[1], brotli.compress(qc.astype(np.int8).tobytes(), quality=11), parts[3] or None)
        src_bin.write_bytes(capped)

    dst_raw = inflated_dir / "0.raw"
    # Local test runs the inflate via THIS interpreter (no bare `python` on macOS PATH); the
    # contest inflate.sh uses `python` which IS the venv interpreter on the Linux x86_64 runner.
    cmd = [sys.executable, str(packet_dir / "inflate.py"), str(src_bin), str(dst_raw)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(packet_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"inflate.py FAILED rc={proc.returncode}\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    n_frames_expected = 2 * eval_pairs
    expected_bytes = n_frames_expected * CAMERA_H * CAMERA_W * 3
    actual_bytes = dst_raw.stat().st_size
    full_output_ok = actual_bytes == expected_bytes
    return {
        "inflate_stdout": proc.stdout.strip(),
        "eval_pairs": eval_pairs,
        "n_frames_emitted": n_frames_expected,
        "raw_path": str(dst_raw),
        "raw_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "full_output_shape_ok": bool(full_output_ok),
        "frame_layout": f"({n_frames_expected}, {CAMERA_H}, {CAMERA_W}, 3) uint8 [f0,f1 per pair]",
    }


# ---------------------------------------------------------------------------
# d_seg / d_pose parity ON THE INFLATED FRAMES (frozen CPU-torch, vs GT)
# ---------------------------------------------------------------------------
def parity_on_inflated(raw_path: Path, eval_pairs: int, gt_cache: str | None) -> dict[str, Any]:
    """Read the .raw back, run the FROZEN CPU-torch SegNet/PoseNet on the INFLATED frames
    over all eval pairs, return realized d_seg/d_pose vs GT. This is the contest-faithful
    realized number (not a generator byte-repro)."""
    if gt_cache:
        gt, seg_cpu, posenet_cpu = twr.load_gt_from_cache(Path(gt_cache), eval_pairs)
    else:
        gt, seg_cpu, posenet_cpu = twr.precompute_gt(eval_pairs)
    P = min(eval_pairs, gt.n_pairs)
    frame_bytes = CAMERA_H * CAMERA_W * 3
    d_segs, d_poses = [], []
    with open(raw_path, "rb") as f:
        for pi in range(P):
            f0 = np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3)
            f1 = np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAMERA_H, CAMERA_W, 3)
            d_segs.append(twr.cpu_verdict_d_seg(seg_cpu, f1, gt.lstars[pi]))
            d_poses.append(twr.cpu_verdict_d_pose(posenet_cpu, f0, f1, gt.gt_poses[pi]))
    return {
        "pairs_scored": P,
        "d_seg_realized_on_inflated": float(np.mean(d_segs)),
        "d_pose_realized_on_inflated": float(np.mean(d_poses)),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(
    ckpt_dir: Path,
    *,
    max_pairs: int | None,
    fold_pose_sidecar: bool,
    pose_sidecar_path: Path | None,
    gt_cache: str | None,
    keep_packet: bool,
    packet_dir: Path | None,
    skip_parity: bool,
    use_live: bool = False,
) -> dict[str, Any]:
    params, cfg = _load_witness_ckpt(ckpt_dir, use_live=use_live)
    n_pairs = int(cfg["n_pairs"])

    pose_bytes: bytes | None = None
    pose_note = "off (RGB witness carries pose in per-(pair,frame) codes; train --w-pose>0)"
    if fold_pose_sidecar:
        if pose_sidecar_path and Path(pose_sidecar_path).exists():
            pose_bytes = Path(pose_sidecar_path).read_bytes()
            pose_note = f"folded {len(pose_bytes)} B from {pose_sidecar_path}"
        else:
            # synth from scorer_targets is heavy (decodes GT); require an explicit prebuilt file.
            raise FileNotFoundError(
                "--fold-pose-sidecar requires --pose-sidecar-path <posenet_targets.bin> "
                "(build via tac.scorer_targets.extract_and_save). NO-FAKE: refusing to fabricate."
            )

    blob, breakdown = build_witness_blob(params, cfg, pose_bytes)

    packet_dir = packet_dir or (
        _REPO / "experiments" / "results"
        / f"witness_packet_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    _refuse_tmp(packet_dir, "packet_dir")
    zip_path, zip_bytes = assemble_packet(blob, packet_dir)
    rate = zip_bytes / RATE_DENOM
    rate_term = 25.0 * rate

    print(f"[ckpt] {ckpt_dir}  n_pairs={n_pairs}  params={breakdown['n_params']}  {_AUTHORITY}", flush=True)
    print(f"[byte-close] 0.bin={breakdown['total_0bin_bytes']} B  archive.zip={zip_bytes} B  "
          f"rate={rate:.6f} rate_term={rate_term:.4f}  pose_sidecar={pose_note}", flush=True)

    inflate_info = run_inflate(packet_dir, n_pairs, max_pairs)
    print(f"[inflate] {inflate_info['frame_layout']}  full_output_ok={inflate_info['full_output_shape_ok']}  "
          f"raw_bytes={inflate_info['raw_bytes']}", flush=True)

    parity: dict[str, Any] = {"skipped": True}
    if not skip_parity:
        parity = parity_on_inflated(Path(inflate_info["raw_path"]), inflate_info["eval_pairs"], gt_cache)
        d_seg = parity["d_seg_realized_on_inflated"]
        d_pose = parity["d_pose_realized_on_inflated"]
        seg_term = 100.0 * d_seg
        pose_term = (10.0 * d_pose + 1e-12) ** 0.5
        score = seg_term + pose_term + rate_term
        parity.update({
            "seg_term": seg_term, "pose_term": pose_term, "rate_term": rate_term,
            "implied_S_advisory": score,
        })
        print(f"[parity] d_seg={d_seg:.6f} d_pose={d_pose:.6f} (realized on INFLATED frames, {parity['pairs_scored']} pairs) "
              f"| S_advisory={score:.4f}  {_AUTHORITY}", flush=True)

    # staged contest-CPU exact-eval command (NOT run here)
    contest_cmd = (
        f".venv/bin/python experiments/contest_auth_eval.py "
        f"--archive {zip_path} "
        f"--inflate-sh {packet_dir / 'inflate.sh'} "
        f"--device cpu  # [contest-CPU] authoritative ONLY on Linux x86_64 (Modal CPU); macOS-local = advisory"
    )

    report: dict[str, Any] = {
        "tool": "witness_byte_close_and_eval",
        "authority": _AUTHORITY,
        "promotion_claim": False,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ckpt_dir": str(ckpt_dir),
        "weights_arm": ("live" if use_live else "ema"),
        "n_pairs_total": n_pairs,
        "config": {k: cfg.get(k) for k in (
            "n_pairs", "n_fourier", "hidden_dim", "n_hidden", "mod_dim", "fourier_sigma",
            "render_h", "render_w", "activation", "basis", "w_pose")},
        "byte_close": {
            **breakdown,
            "archive_zip_bytes": zip_bytes,
            "zip_container_overhead_bytes": zip_bytes - breakdown["total_0bin_bytes"],
            "rate": rate, "rate_term": rate_term,
            "rate_denom_bytes": int(RATE_DENOM),
            "archive_zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            "pose_sidecar": pose_note,
        },
        "inflate": inflate_info,
        "parity_on_inflated_frames": parity,
        "contest_cpu_eval_cmd": contest_cmd,
        "packet_dir": str(packet_dir),
        "mlx_free_inflate": True,
        # The contest .raw MUST have 1200 frames (600 pairs x seq_len 2) to match GT. A
        # witness with n_pairs != 600 inflates 2*n_pairs frames -> the evaluate.py shape
        # assertion fails. Loud, fail-closed: only an n_pairs==600 witness is contest-ready.
        "contest_ready_full_600": bool(n_pairs == 600),
        "contest_ready_note": (
            "n_pairs==600 -> full 1200-frame .raw, contest-ready"
            if n_pairs == 600
            else f"n_pairs={n_pairs} != 600 -> inflate emits {2*n_pairs} frames; a 600-pair "
            "witness is required for the 1200-frame contest .raw (this checkpoint is test-only)"
        ),
    }
    if not keep_packet:
        import shutil
        shutil.rmtree(packet_dir, ignore_errors=True)
        report["packet_dir"] = "(deleted; pass --keep-packet to retain for the exact-eval row)"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path, required=True,
                    help="witness run dir with witness_ema_mlx.npz + train_result.json")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap inflate+parity pairs for SPEED (default: all). Archive always encodes all codes.")
    ap.add_argument("--fold-pose-sidecar", action="store_true",
                    help="append a counted stored-pose section (capstone d_seg-only vehicle); "
                         "requires --pose-sidecar-path. OFF by default (RGB witness carries pose in codes).")
    ap.add_argument("--pose-sidecar-path", type=Path, default=None,
                    help="prebuilt posenet_targets.bin (tac.scorer_targets.extract_and_save)")
    ap.add_argument("--gt-cache", type=str, default=None, help="optional shared GT npz for parity")
    ap.add_argument("--skip-parity", action="store_true", help="byte-close + inflate only (no GT decode)")
    ap.add_argument("--use-live", action="store_true",
                    help="(DAG FEED-br sister) byte-close the LIVE-weights checkpoint "
                         "(witness_live_mlx.npz) instead of the EMA shadow (witness_ema_mlx.npz); "
                         "use when the trainer flags an EMA-lag (the 78x trap).")
    ap.add_argument("--keep-packet", action="store_true", help="retain the packet dir for the exact-eval row")
    ap.add_argument("--out", type=Path, default=None, help="JSON report path (default reports/witness_byte_close_<ts>.json)")
    args = ap.parse_args(argv)

    report = run(
        args.ckpt_dir,
        max_pairs=args.max_pairs,
        fold_pose_sidecar=args.fold_pose_sidecar,
        pose_sidecar_path=args.pose_sidecar_path,
        gt_cache=args.gt_cache,
        keep_packet=args.keep_packet,
        packet_dir=None,
        skip_parity=args.skip_parity,
        use_live=args.use_live,
    )
    out = args.out or (_REPO / "reports" / f"witness_byte_close_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[report] wrote {out}  {_AUTHORITY}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
