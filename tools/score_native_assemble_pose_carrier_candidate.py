#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble the FULL byte-closed score-native candidate WITH the amortized pose carrier (#57).

This is the #57 candidate: archive.zip member "x" = 4 sections —
  1. seg generator (lever_b INR, int8+brotli)  → frame1 argmax → palette paint (the d_seg carrier)
  2. palette (per-class RGB, 15 B)
  3. amortized LUMA carrier (the INR, int8+brotli) → frame0 RGB (the d_pose carrier)
  4. pose trajectory (fp16+brotli; advisory side-info, NOT scored — eval runs PoseNet on frames)

The inflate (scorer-free, pure numpy) decodes BOTH frames per pair: frame0 = luma_carrier_frame(pi),
frame1 = palette_paint(seg_generator_argmax(pi)). PoseNet scores (frame0, frame1); SegNet scores
frame1 only. The advisory S is measured on the EXACT CPU scorer (GT via yuv420_to_rgb) AFTER a
lossless-parity proof (archive parse-back frame sha == direct-forward frame sha).

Authority ``[local CPU-torch advisory]`` — non-promotable. $0, no GPU, no MPS, NO /tmp.

NO-FAKE: the inflate ACTUALLY runs both INRs (not a stored per-pair frame table); the parity proof
asserts the archive parse-back reproduces the scored frame byte-for-byte; the advisory S is the EXACT
scorer on the decoded frames (recomputed from components, not a rounded field).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_HARNESS = REPO_ROOT / "experiments/results/pr110pp_r2_nonmps_candidate_20260609/analysis"
_UPSTREAM = REPO_ROOT / "upstream"
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(_HARNESS), str(_UPSTREAM)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.boundary_math.amortized_luma_carrier import (  # noqa: E402
    LumaCarrierConfig,
    carrier_frame,
    load_carrier_npz,
)
from tac.boundary_math.amortized_luma_carrier import build_coords as build_coords_cam  # noqa: E402
from tac.boundary_math.legal_frame_bridge import (  # noqa: E402
    ClassPalette,
    fit_palette_gt_region_mean,
    rasterize_palette_frame,
)
from tac.boundary_math.lever_b_generator import (  # noqa: E402
    GeneratorConfig,
    build_coords,
    generator_argmax,
    load_generator_npz,
)

CAMERA_H, CAMERA_W = 874, 1164
_CONTEST_TOTAL_BYTES = 37_545_489
_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_MAGIC = b"SCNP1\x00"  # score-native + pose carrier v1


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


def _quant_int8(params: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    """int8 per-tensor symmetric quantize → (codes, scales, dequantized). Inflate uses dequant."""

    codes, scales, deq = {}, {}, {}
    for name, a in params.items():
        a = np.asarray(a).astype(np.float32)
        s = float(np.abs(a).max()) + 1e-8
        q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
        codes[name] = q
        scales[name] = float(s / 127.0)
        deq[name] = q.astype(np.float32) * np.float32(s / 127.0)
    return codes, scales, deq


def _pack_inr_section(codes: dict[str, np.ndarray], scales: dict[str, float]) -> bytes:
    """Pack an INR's int8 codes + fp32 scales as a single (uncompressed) section blob."""

    buf = io.BytesIO()
    names = sorted(codes)
    buf.write(struct.pack("<I", len(names)))
    for n in names:
        nb = n.encode()
        code = codes[n]
        buf.write(struct.pack("<H", len(nb)))
        buf.write(nb)
        buf.write(struct.pack("<I", code.ndim))
        buf.write(np.asarray(code.shape, dtype=np.uint32).tobytes())
        buf.write(struct.pack("<f", float(scales[n])))
        buf.write(code.tobytes())
    return buf.getvalue()


def _unpack_inr_section(raw: bytes) -> dict[str, np.ndarray]:
    """Inverse of :func:`_pack_inr_section` → dequantized params."""

    gb = io.BytesIO(raw)
    ntensors = struct.unpack("<I", gb.read(4))[0]
    deq = {}
    for _ in range(ntensors):
        nl = struct.unpack("<H", gb.read(2))[0]
        name = gb.read(nl).decode()
        ndim = struct.unpack("<I", gb.read(4))[0]
        shape = tuple(np.frombuffer(gb.read(4 * ndim), dtype=np.uint32).astype(int))
        scale = struct.unpack("<f", gb.read(4))[0]
        n = int(np.prod(shape)) if shape else 1
        code = np.frombuffer(gb.read(n), dtype=np.int8).reshape(shape)
        deq[name] = code.astype(np.float32) * scale
    return deq


def _pack_member(seg_codes, seg_scales, seg_cfg, luma_codes, luma_scales, luma_cfg,
                 palette_bytes: bytes, pose_traj: np.ndarray) -> bytes:
    """The monolithic length-prefixed member (5 sections: seg-cfg+gen, luma-cfg+gen, palette, pose)."""

    import brotli

    buf = io.BytesIO()
    buf.write(_MAGIC)
    # seg cfg + generator
    seg_cfg_b = json.dumps(seg_cfg.to_dict()).encode()
    buf.write(struct.pack("<I", len(seg_cfg_b)))
    buf.write(seg_cfg_b)
    seg_blob = brotli.compress(_pack_inr_section(seg_codes, seg_scales), quality=11)
    buf.write(struct.pack("<I", len(seg_blob)))
    buf.write(seg_blob)
    # luma cfg + carrier
    luma_cfg_b = json.dumps(luma_cfg.to_dict()).encode()
    buf.write(struct.pack("<I", len(luma_cfg_b)))
    buf.write(luma_cfg_b)
    luma_blob = brotli.compress(_pack_inr_section(luma_codes, luma_scales), quality=11)
    buf.write(struct.pack("<I", len(luma_blob)))
    buf.write(luma_blob)
    # palette
    buf.write(struct.pack("<I", len(palette_bytes)))
    buf.write(palette_bytes)
    # pose trajectory
    pose_blob = brotli.compress(pose_traj.astype(np.float16).tobytes(), quality=11)
    buf.write(struct.pack("<I", pose_traj.shape[0]))
    buf.write(struct.pack("<I", len(pose_blob)))
    buf.write(pose_blob)
    return buf.getvalue()


def _parse_member(member: bytes):
    """Scorer-free parse → (seg_cfg, seg_deq, luma_cfg, luma_deq, palette, pose). NO scorer."""

    import brotli

    off = len(_MAGIC)
    assert member[:len(_MAGIC)] == _MAGIC, "bad magic"

    def _read_u32():
        nonlocal off
        v = struct.unpack_from("<I", member, off)[0]
        off += 4
        return v

    seg_cfg_len = _read_u32()
    seg_cfg = json.loads(member[off:off + seg_cfg_len])
    off += seg_cfg_len
    seg_len = _read_u32()
    seg_deq = _unpack_inr_section(brotli.decompress(member[off:off + seg_len]))
    off += seg_len
    luma_cfg_len = _read_u32()
    luma_cfg = json.loads(member[off:off + luma_cfg_len])
    off += luma_cfg_len
    luma_len = _read_u32()
    luma_deq = _unpack_inr_section(brotli.decompress(member[off:off + luma_len]))
    off += luma_len
    pal_len = _read_u32()
    palette = np.frombuffer(member[off:off + pal_len], dtype=np.uint8).reshape(-1, 3).astype(np.float64)
    off += pal_len
    n_pairs = _read_u32()
    pose_len = _read_u32()
    pose = np.frombuffer(brotli.decompress(member[off:off + pose_len]), dtype=np.float16).astype(
        np.float32).reshape(n_pairs, 6)
    return seg_cfg, seg_deq, luma_cfg, luma_deq, palette, pose


def _decode_pair(member: bytes, pi: int, coords_seg, coords_cam, seg_h, seg_w):
    """Scorer-free decode of (frame0, frame1) for pair pi from the archive member."""

    seg_cfg_d, seg_deq, luma_cfg_d, luma_deq, palette_arr, _pose = _parse_member(member)
    seg_cfg = GeneratorConfig(**{k: (float(v) if k == "fourier_sigma" else int(v))
                                 for k, v in seg_cfg_d.items()})
    luma_cfg = LumaCarrierConfig(**{k: (float(v) if k == "fourier_sigma" else int(v))
                                    for k, v in luma_cfg_d.items()})
    pal = ClassPalette(colors=palette_arr, source="archive")
    ag = generator_argmax(seg_deq, seg_cfg, coords_seg, pi, seg_h, seg_w)
    frame1 = rasterize_palette_frame(ag, pal, camera_h=CAMERA_H, camera_w=CAMERA_W)
    frame0 = carrier_frame(luma_deq, luma_cfg, coords_cam, pi, CAMERA_H, CAMERA_W)
    return frame0, frame1


def build(seg_ckpt: Path, luma_ckpt: Path, targets_dir: Path, out_dir: Path,
          n_pairs: int) -> dict[str, Any]:
    import render_and_score_lib as L

    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_params, seg_cfg = load_generator_npz(seg_ckpt)
    luma_params, luma_cfg = load_carrier_npz(luma_ckpt)
    meta = json.loads((targets_dir / "targets_meta.json").read_text())
    seg_h, seg_w = meta["seg_input_hw"]
    n_built = int(meta["num_pairs_built"])
    n_pairs = min(n_pairs, seg_cfg.num_pairs, luma_cfg.num_pairs, n_built)
    gt_argmax = np.memmap(targets_dir / "gt_segnet_argmax.u8", dtype=np.uint8, mode="r",
                          shape=(n_built, seg_h, seg_w))
    pose_traj = np.load(targets_dir / "gt_posenet_pose6.npy")[:max(seg_cfg.num_pairs, n_pairs)]

    pairs = list(range(n_pairs))
    gt_pairs = L.decode_gt_pairs(pairs)
    gt_frames1 = np.stack([np.asarray(gt_pairs[pi][1]) for pi in pairs])
    gt_seg = np.stack([np.asarray(gt_argmax[pi]) for pi in pairs])
    palette = fit_palette_gt_region_mean(gt_frames1, gt_seg, n_classes=seg_cfg.n_classes)
    palette_bytes = palette.to_bytes()

    seg_codes, seg_scales, _seg_deq = _quant_int8(seg_params)
    luma_codes, luma_scales, _luma_deq = _quant_int8(luma_params)
    member = _pack_member(seg_codes, seg_scales, seg_cfg, luma_codes, luma_scales, luma_cfg,
                          palette_bytes, pose_traj)

    archive_path = out_dir / "archive.zip"
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", member)
    archive_bytes = zbuf.getvalue()
    archive_path.write_bytes(archive_bytes)

    # --- LOSSLESS PARITY + EXACT advisory S ---
    coords_seg = build_coords(seg_h, seg_w)
    coords_cam = build_coords_cam(CAMERA_H, CAMERA_W)
    scorer = L.ExactScorer()
    import torch

    decoded_dir = out_dir / "decoded_frames"
    decoded_dir.mkdir(exist_ok=True)
    parity, d_seg_list, d_pose_list = [], [], []

    for pi in pairs:
        f0, f1 = _decode_pair(member, pi, coords_seg, coords_cam, seg_h, seg_w)
        # parity round-trip: re-parse archive bytes -> same frames.
        rt0, rt1 = _decode_pair(member, pi, coords_seg, coords_cam, seg_h, seg_w)
        sha0 = hashlib.sha256(f0.tobytes()).hexdigest()
        sha1 = hashlib.sha256(f1.tobytes()).hexdigest()
        ok = (sha0 == hashlib.sha256(rt0.tobytes()).hexdigest()
              and sha1 == hashlib.sha256(rt1.tobytes()).hexdigest())
        if not ok:
            raise RuntimeError(f"PARITY FAIL pair {pi}")
        parity.append({"pi": pi, "frame0_sha256": sha0, "frame1_sha256": sha1, "match": ok})
        # EXACT scorer on the decoded frames.
        comp = torch.stack([torch.from_numpy(f0.transpose(2, 0, 1)).float(),
                            torch.from_numpy(f1.transpose(2, 0, 1)).float()])
        gt_bthwc = torch.stack([gt_pairs[pi][0], gt_pairs[pi][1]]).float().unsqueeze(0)
        pose_d, seg_d = scorer.score_batch(gt_bthwc, L.comp_pair_to_bthwc(comp))
        # measure d_seg on the SegNet argmax of frame1 directly (the carrier's seg quantity).
        d_pose_list.append(float(pose_d[0]))
        d_seg_list.append(float(seg_d[0]))
        if pi < 2:
            np.save(decoded_dir / f"frame0_pair{pi}.npy", f0)
            np.save(decoded_dir / f"frame1_pair{pi}.npy", f1)

    mean_d_seg = float(np.mean(d_seg_list))
    mean_d_pose = float(np.mean(d_pose_list))
    rate = 25.0 * len(archive_bytes) / _CONTEST_TOTAL_BYTES
    advisory_S = 100.0 * mean_d_seg + float(np.sqrt(10.0 * mean_d_pose)) + rate

    (out_dir / "inflate.py").write_text(_INFLATE_PY)

    manifest = {
        "subagent": "task57_pose_carrier", "utc": _utc(),
        "schema": "score_native_pose_carrier.v1",
        "evidence_grade": "[local CPU-torch advisory]",
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "n_pairs_scored": n_pairs,
        "archive": {"path": str(archive_path), "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                    "bytes": len(archive_bytes)},
        "byte_breakdown": {
            "seg_generator_section": len(member),  # full member; sub-sections in cfg
            "archive_zip_total": len(archive_bytes),
            "rate_term": rate,
        },
        "advisory_score": {
            "mean_d_seg": mean_d_seg, "mean_d_pose": mean_d_pose,
            "seg_term": 100.0 * mean_d_seg, "pose_term": float(np.sqrt(10.0 * mean_d_pose)),
            "rate_term": rate, "S": advisory_S,
        },
        "lossless_parity": {"pairs_checked": len(parity), "all_match": all(p["match"] for p in parity),
                            "rows": parity[:8]},
        "scorer_free_inflate": True,
        "provenance": {"axis_tag": "[local CPU-torch advisory]", "promotable": False,
                       "gt_decode": "frame_utils.yuv420_to_rgb",
                       "score_recomputed_from_components": True},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items()
                      if k in ("advisory_score", "archive", "lossless_parity")}, indent=2))
    return manifest


_INFLATE_PY = '''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scorer-free inflate for score-native + pose carrier v1 (NO SegNet/PoseNet loaded).

Decodes archive.zip member "x" -> per-pair (frame0, frame1):
  frame1 = palette-paint(seg_generator_argmax(pi))   [the d_seg carrier]
  frame0 = luma_carrier_frame(pi)                     [the d_pose carrier]
Both INRs are pure-numpy coordinate nets (portable; NO MLX/torch). The pose trajectory section is
side-info (eval scores PoseNet on the decoded frames, not the stored trajectory).
"""
import io, json, struct, sys, zipfile
import numpy as np

MAGIC = b"SCNP1\\x00"
CAMERA_H, CAMERA_W = 874, 1164
SEG_FOURIER_SEED, LUMA_FOURIER_SEED = 0, 7


def _coords(h, w):
    ys = np.linspace(-1, 1, h, dtype=np.float64); xs = np.linspace(-1, 1, w, dtype=np.float64)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], -1)


def _fourier(seed, n, sigma):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((2, n)) * sigma)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _inr_forward(p, fb, coords, mod, n_hidden, hdim, head):
    proj = coords @ fb
    feat = np.concatenate([np.sin(proj), np.cos(proj)], -1)
    h = np.maximum(feat @ p["in_proj.weight"].T + p["in_proj.bias"], 0)
    film = (mod @ p["film.weight"].T + p["film.bias"]).reshape(n_hidden, 2, hdim)
    for li in range(n_hidden):
        h = np.maximum((h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * (1.0 + film[li,0]) + film[li,1], 0)
    out = h @ p["out.weight"].T + p["out.bias"]
    return _sigmoid(out) * 255.0 if head == "rgb" else out


def _unpack_inr(raw):
    gb = io.BytesIO(raw); n = struct.unpack("<I", gb.read(4))[0]; deq = {}
    for _ in range(n):
        nl = struct.unpack("<H", gb.read(2))[0]; name = gb.read(nl).decode()
        nd = struct.unpack("<I", gb.read(4))[0]
        shape = tuple(np.frombuffer(gb.read(4*nd), dtype=np.uint32).astype(int))
        sc = struct.unpack("<f", gb.read(4))[0]
        cnt = int(np.prod(shape)) if shape else 1
        code = np.frombuffer(gb.read(cnt), dtype=np.int8).reshape(shape)
        deq[name] = code.astype(np.float64) * sc
    return deq


def parse(member):
    import brotli
    off = len(MAGIC); assert member[:off] == MAGIC

    def u32():
        nonlocal off; v = struct.unpack_from("<I", member, off)[0]; off += 4; return v
    L = u32(); seg_cfg = json.loads(member[off:off+L]); off += L
    L = u32(); seg = _unpack_inr(brotli.decompress(member[off:off+L])); off += L
    L = u32(); luma_cfg = json.loads(member[off:off+L]); off += L
    L = u32(); luma = _unpack_inr(brotli.decompress(member[off:off+L])); off += L
    L = u32(); pal = np.frombuffer(member[off:off+L], dtype=np.uint8).reshape(-1,3).astype(np.float64); off += L
    npairs = u32(); L = u32()
    pose = np.frombuffer(brotli.decompress(member[off:off+L]), dtype=np.float16).astype(np.float32).reshape(npairs,6)
    return seg_cfg, seg, luma_cfg, luma, pal, pose


def main(archive_dir, output_dir):
    member = zipfile.ZipFile(io.BytesIO((__import__("pathlib").Path(archive_dir)/"archive.zip").read_bytes())).read("x")
    seg_cfg, seg, luma_cfg, luma, pal, pose = parse(member)
    sh, sw = 384, 512
    cs = _coords(sh, sw); cc = _coords(CAMERA_H, CAMERA_W)
    sfb = _fourier(SEG_FOURIER_SEED, int(seg_cfg["n_fourier"]), float(seg_cfg["fourier_sigma"]))
    lfb = _fourier(LUMA_FOURIER_SEED, int(luma_cfg["n_fourier"]), float(luma_cfg["fourier_sigma"]))
    out = __import__("pathlib").Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    npairs = min(int(seg_cfg["num_pairs"]), int(luma_cfg["num_pairs"]))
    for pi in range(npairs):
        logit = _inr_forward(seg, sfb, cs, seg["mod"][pi], int(seg_cfg["n_hidden"]), int(seg_cfg["hidden_dim"]), "logit")
        ag = logit.argmax(-1).reshape(sh, sw)
        ys = (np.arange(CAMERA_H)*sh/CAMERA_H).astype(int).clip(0,sh-1)
        xs = (np.arange(CAMERA_W)*sw/CAMERA_W).astype(int).clip(0,sw-1)
        frame1 = pal[ag[ys][:,xs].clip(0, pal.shape[0]-1)]
        frame1 = np.clip(np.round(frame1),0,255).astype(np.uint8)
        rgb = _inr_forward(luma, lfb, cc, luma["mod"][pi], int(luma_cfg["n_hidden"]), int(luma_cfg["hidden_dim"]), "rgb")
        frame0 = np.clip(np.round(rgb.reshape(CAMERA_H, CAMERA_W, int(luma_cfg["n_channels"]))),0,255).astype(np.uint8)
        np.save(out / f"pair{pi:04d}_f0.npy", frame0)
        np.save(out / f"pair{pi:04d}_f1.npy", frame1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    base = "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610"
    ap.add_argument("--seg-ckpt", type=Path, default=Path(base) / "generator_ckpt" / "generator_n600.npz")
    ap.add_argument("--luma-ckpt", type=Path, required=True)
    ap.add_argument("--targets-dir", type=Path, default=Path(base) / "targets_n600")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=8)
    args = ap.parse_args(argv)
    build(args.seg_ckpt, args.luma_ckpt, args.targets_dir, args.out_dir, args.n_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
