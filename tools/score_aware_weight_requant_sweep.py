#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Score-aware per-tensor weight re-quant SWEEP harness (Task #69).

THE HYPOTHESIS (audit #1 lurking exact-score mover): the frontier HNeRV decoder
weights are quantized for PIXEL-RECON fidelity, not for the minimal bytes that
hold the SCORED quantities (SegNet argmax d_seg + 6 PoseNet dims d_pose). The
frozen scorer tolerates far more weight error than recon. So per-tensor bit
RE-ALLOCATION by MEASURED scorer sensitivity cuts the dominant rate term while
holding d_seg / d_pose inside the same evaluator cell.

This is the EXACT-AUTHORITY harness (NO proxy): it decodes the frontier archive,
re-quantizes decoder tensors in the q-domain (tac.score_aware_weight_requant),
re-packs into the EXACT CTXR/FP11 grammar (byte-closed), DECODES the re-packed
archive through the FRONTIER inflate chain, and measures EXACT d_seg / d_pose on
the FROZEN upstream SegNet+PoseNet vs GT decoded via frame_utils.yuv420_to_rgb
(NEVER MPS). Score is recomputed from components.

Stages:
  1. baseline   — decode frontier archive, calibrate d_seg/d_pose vs pointer.
  2. rank       — per-tensor sensitivity finite-difference probe (subset pairs):
                  re-quant each tensor to a probe level, measure Δd_seg/Δd_pose.
  3. sweep      — allocate bits by sensitivity, build re-quant candidates at
                  increasing aggression, byte-close, measure EXACT score (full pairs).
  4. report     — operating-point curve + the exact pointer delta.

Outputs land under experiments/results/score_aware_weight_requant_<UTC>/ (durable,
NOT /tmp). GT decode + GT scorer outputs are cached once (the expensive half).

Usage:
  .venv/bin/python tools/score_aware_weight_requant_sweep.py \
      --archive experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip \
      --stage baseline
  ... --stage rank --rank-pairs 120 --probe-levels 4
  ... --stage sweep --sweep-pairs 600
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = REPO_ROOT / "upstream"
GT_VIDEO = UPSTREAM / "videos" / "0.mkv"
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/archive.zip"
)
CAMERA_H, CAMERA_W = 874, 1164
SEG_IN = (384, 512)  # segnet_model_input_size is (W=512,H=384); interpolate uses (H,W)
N_PAIRS = 600

# DQS1 q-substitution targets storage_index 26 (blocks.3.weight). Protect it from
# re-quant so the DQS1 patch base stays byte-exact (avoids a confounder).
DQS1_PROTECTED_STORAGE_INDEX = 26

sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.score_aware_weight_requant import (  # noqa: E402
    allocate_bits_by_sensitivity,
    contest_score_from_components,
    decode_byte_map_u8,
    encode_byte_map_u8,
    q_byte_entropy_bits,
    requant_signed_q,
    score_delta_components,
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _refuse_tmp(path: Path, field: str) -> None:
    if str(path).startswith("/tmp") or "/tmp/" in str(path):
        raise SystemExit(f"FAIL-CLOSED: {field} must not be a /tmp path (durable evidence): {path}")


# ---------------------------------------------------------------------------
# frontier inflate module (the EXACT render chain) + codec_ctx (entropy coder)
# ---------------------------------------------------------------------------


def _load_frontier_modules(submission_dir: Path):
    src = submission_dir / "src"
    for p in (str(src), str(submission_dir), str(UPSTREAM)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("frontier_inflate", submission_dir / "inflate.py")
    inflate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inflate_mod)
    import codec  # type: ignore[import-not-found]
    import codec_ctx  # type: ignore[import-not-found]
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    return inflate_mod, codec, codec_ctx, HNeRVDecoder


# ---------------------------------------------------------------------------
# Archive parsing: extract the FP11 member + its parts
# ---------------------------------------------------------------------------


def _read_member(archive: Path) -> bytes:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        member_name = "x" if "x" in names else names[0]
        return zf.read(member_name)


def _parse_fp11(member: bytes):
    assert member[:4] == b"FP11", f"bad FP11 magic {member[:4]!r}"
    pos = 4
    (source_len,) = struct.unpack_from("<I", member, pos)
    pos += 4
    source_payload = member[pos : pos + source_len]
    pos += source_len
    (sel_len,) = struct.unpack_from("<H", member, pos)
    pos += 2
    sel_payload = member[pos : pos + sel_len]
    pos += sel_len
    dqs1_tail = member[pos:]
    return source_payload, sel_payload, dqs1_tail


def _rebuild_member(source_payload: bytes, sel_payload: bytes, dqs1_tail: bytes) -> bytes:
    out = bytearray(b"FP11")
    out += struct.pack("<I", len(source_payload))
    out += source_payload
    out += struct.pack("<H", len(sel_payload))
    out += sel_payload
    out += dqs1_tail
    return bytes(out)


# ---------------------------------------------------------------------------
# Decoder raw-stream <-> per-tensor q-byte view
# ---------------------------------------------------------------------------


def _tensor_layout(HNeRVDecoder, codec):
    probe = HNeRVDecoder(latent_dim=codec.LATENT_DIM, base_channels=codec.BASE_CHANNELS,
                         eval_size=codec.EVAL_SIZE)
    items = list(probe.state_dict().items())
    layout = []  # (storage_index, name, numel, byte_map)
    for sidx in codec.DECODER_STORAGE_ORDER:
        name, tensor = items[sidx]
        layout.append((int(sidx), name, int(tensor.numel()), codec.DECODER_BYTE_MAPS.get(sidx, "zig")))
    return layout


def _requant_raw_joined(raw_joined: bytes, layout, levels_by_sidx: dict[int, int]) -> bytes:
    """Rebuild raw_joined with the given per-tensor re-quant levels (q-domain)."""
    out = bytearray()
    pos = 0
    for sidx, _name, numel, byte_map in layout:
        qb = np.frombuffer(raw_joined, dtype=np.uint8, count=numel, offset=pos).copy()
        pos += numel
        scale_bytes = raw_joined[pos : pos + 2]
        pos += 2
        levels = levels_by_sidx.get(sidx, 256)
        if levels < 256:
            q_signed = decode_byte_map_u8(qb, byte_map)
            q_rq = requant_signed_q(q_signed, levels)
            qb = encode_byte_map_u8(q_rq, byte_map)
        out += qb.tobytes() + bytes(scale_bytes)
    if pos != len(raw_joined):
        raise ValueError("raw_joined walk mismatch")
    return bytes(out)


def _split_to_streams(raw_joined: bytes, layout, codec) -> list[bytes]:
    """Split raw_joined into the 7 ctx streams at DECODER_STREAM_ENDS boundaries."""
    lens = [numel + 2 for (_sidx, _n, numel, _bm) in layout]
    streams = []
    ti = 0
    spos = 0
    for end in codec.DECODER_STREAM_ENDS:
        seg = bytearray()
        while ti < end:
            seg += raw_joined[spos : spos + lens[ti]]
            spos += lens[ti]
            ti += 1
        streams.append(bytes(seg))
    if spos != len(raw_joined):
        raise ValueError("stream split mismatch")
    return streams


# ---------------------------------------------------------------------------
# GT cache: decode 0.mkv via yuv420_to_rgb (the EXACT eval GT path; NEVER MPS)
# ---------------------------------------------------------------------------


def _decode_gt(n_frames: int) -> np.ndarray:
    import av
    from frame_utils import yuv420_to_rgb  # type: ignore[import-not-found]

    frames = []
    container = av.open(str(GT_VIDEO))
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if i >= n_frames:
            break
        frames.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return np.stack(frames, axis=0)  # (n,H,W,3) uint8


def _load_distortion_net():
    from modules import DistortionNet  # type: ignore[import-not-found]

    dn = DistortionNet().eval()
    dn.load_state_dicts(
        str(UPSTREAM / "models" / "posenet.safetensors"),
        str(UPSTREAM / "models" / "segnet.safetensors"),
        "cpu",
    )
    return dn


def _measure_distortion(dn, gt_pairs_t: torch.Tensor, comp_pairs_t: torch.Tensor, *, batch: int = 16):
    n = gt_pairs_t.shape[0]
    d_seg_all, d_pose_all = [], []
    for s in range(0, n, batch):
        e = min(s + batch, n)
        with torch.inference_mode():
            d_pose, d_seg = dn.compute_distortion(gt_pairs_t[s:e], comp_pairs_t[s:e])
        d_seg_all.extend([float(x) for x in d_seg.tolist()])
        d_pose_all.extend([float(x) for x in d_pose.tolist()])
    return np.asarray(d_seg_all, np.float64), np.asarray(d_pose_all, np.float64)


# ---------------------------------------------------------------------------
# Render: decode a (re-packed) member to camera-res comp frames via frontier inflate
# ---------------------------------------------------------------------------


def _render_member_to_frames(inflate_mod, member: bytes, n_pairs: int, work_dir: Path) -> np.ndarray:
    member_path = work_dir / "member.bin"
    raw_path = work_dir / "inflated.raw"
    member_path.write_bytes(member)
    n = inflate_mod.inflate(str(member_path), str(raw_path))
    comp = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(n, CAMERA_H, CAMERA_W, 3))
    comp = np.asarray(comp[: 2 * n_pairs]).reshape(n_pairs, 2, CAMERA_H, CAMERA_W, 3)
    try:
        member_path.unlink()
        raw_path.unlink()
    except OSError:
        pass
    return comp


def _build_recoded_member(
    codec_ctx, source_payload, sel_payload, dqs1_tail, raw_joined_new, codec, layout
):
    """Rebuild a byte-closed FP11 member with the re-quantized decoder section,
    keeping latent + sidecar + selector + DQS1 verbatim."""
    # Re-encode decoder section from the new raw streams.
    streams_new = _split_to_streams(raw_joined_new, layout, codec)
    dec_sec = codec_ctx.encode_decoder_section(streams_new)
    # Parse the original CTXR container to lift lat_sec + sidecar verbatim.
    assert source_payload[:4] == b"CTXR"
    p = 5
    dl = int.from_bytes(source_payload[p : p + 3], "little"); p += 3
    ll = int.from_bytes(source_payload[p : p + 3], "little"); p += 3
    sl = int.from_bytes(source_payload[p : p + 3], "little"); p += 3
    _orig_dec = source_payload[p : p + dl]; p += dl
    lat_sec = source_payload[p : p + ll]; p += ll
    sidecar = source_payload[p : p + sl]; p += sl
    # Rebuild CTXR container with new dec_sec.
    new_ctxr = bytearray(b"CTXR")
    new_ctxr.append(1)  # version
    new_ctxr += len(dec_sec).to_bytes(3, "little")
    new_ctxr += len(lat_sec).to_bytes(3, "little")
    new_ctxr += len(sidecar).to_bytes(3, "little")
    new_ctxr += dec_sec + lat_sec + sidecar
    return _rebuild_member(bytes(new_ctxr), sel_payload, dqs1_tail), len(dec_sec)


def _byte_close_archive(member: bytes, out_dir: Path) -> tuple[int, str]:
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x", member)
    data = archive_path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _frontier_pointer_cpu() -> float:
    ptr = REPO_ROOT / ".omx/state/canonical_frontier_pointer.json"
    data = json.loads(ptr.read_text())
    return float(data["our_local_frontier_contest_cpu"]["score"])


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    archive = Path(args.archive).resolve()
    submission_dir = archive.parent
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        REPO_ROOT / "experiments/results" / f"score_aware_weight_requant_{_utc()}"
    )
    _refuse_tmp(out_dir, "--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = REPO_ROOT / "experiments/results/score_aware_weight_requant_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    inflate_mod, codec, codec_ctx, HNeRVDecoder = _load_frontier_modules(submission_dir)
    layout = _tensor_layout(HNeRVDecoder, codec)
    member = _read_member(archive)
    source_payload, sel_payload, dqs1_tail = _parse_fp11(member)

    # Decoder raw streams (byte-exact) from the source CTXR.
    raw_streams, _latent_raw, _sidecar = inflate_mod.unpack_ctxr_container(source_payload)
    raw_joined = b"".join(raw_streams)

    # --- GT cache (decode + scorer GT side) ---
    n_pairs = int(args.sweep_pairs if args.stage == "sweep" else (args.rank_pairs if args.stage == "rank" else N_PAIRS))
    n_frames = 2 * n_pairs
    gt_cache = cache_dir / f"gt_camera_{n_frames}.npy"
    if gt_cache.exists():
        gt = np.load(gt_cache, mmap_mode="r")
        print(f"[gt] cached {gt.shape}", flush=True)
    else:
        t0 = time.time()
        gt = _decode_gt(n_frames)
        np.save(gt_cache, gt)
        print(f"[gt] decoded {gt.shape} in {time.time()-t0:.1f}s", flush=True)
    gt_pairs = np.asarray(gt[:n_frames]).reshape(n_pairs, 2, CAMERA_H, CAMERA_W, 3).astype(np.float32)
    gt_pairs_t = torch.from_numpy(gt_pairs)

    dn = _load_distortion_net()

    def score_member(m: bytes, np_pairs: int, label: str) -> dict:
        t0 = time.time()
        comp = _render_member_to_frames(inflate_mod, m, np_pairs, out_dir)
        comp_pairs_t = torch.from_numpy(comp.reshape(np_pairs, 2, CAMERA_H, CAMERA_W, 3).astype(np.float32))
        d_seg, d_pose = _measure_distortion(dn, gt_pairs_t[:np_pairs], comp_pairs_t, batch=args.batch)
        dt = time.time() - t0
        return {
            "label": label,
            "mean_d_seg": float(d_seg.mean()),
            "mean_d_pose": float(d_pose.mean()),
            "n_pairs": int(np_pairs),
            "seconds": round(dt, 1),
        }

    results: dict = {
        "subagent": "task69_score_aware_requant",
        "stage": args.stage,
        "utc": _utc(),
        "archive": str(archive),
        "frontier_cpu_pointer": _frontier_pointer_cpu(),
        "note": "EXACT scorer + frame_utils.yuv420_to_rgb GT (NEVER MPS); score recomputed from components",
    }

    if args.stage == "baseline":
        base = score_member(member, n_pairs, "baseline")
        size, sha = len(member), __import__("hashlib").sha256(member).hexdigest()
        sc = contest_score_from_components(
            d_seg=base["mean_d_seg"], d_pose=base["mean_d_pose"], archive_zip_size=archive.stat().st_size,
        )
        base.update({"archive_zip_size": archive.stat().st_size, "member_bytes": len(member),
                     "score": sc["score"], "score_components": sc})
        results["baseline"] = base
        print(json.dumps(base, indent=2), flush=True)

    elif args.stage == "rank":
        # Per-tensor sensitivity finite-difference probe at probe_levels.
        baseline = score_member(member, n_pairs, "baseline_subset")
        print(f"[rank] baseline subset d_seg={baseline['mean_d_seg']:.8f} "
              f"d_pose={baseline['mean_d_pose']:.8f} ({baseline['seconds']}s/candidate)", flush=True)
        probe_levels = int(args.probe_levels)
        rows = []
        # Precompute each tensor's byte offset in raw_joined (q-bytes start).
        offsets: dict[int, int] = {}
        _pos = 0
        for _sidx, _n, _numel, _bm in layout:
            offsets[_sidx] = _pos
            _pos += _numel + 2
        # Probe largest tensors first (where the bytes are).
        probe_order = sorted(layout, key=lambda r: -r[2])
        for sidx, name, numel, byte_map in probe_order:
            if sidx == DQS1_PROTECTED_STORAGE_INDEX:
                continue
            if numel < args.min_numel:
                continue
            rj2 = _requant_raw_joined(raw_joined, layout, {sidx: probe_levels})
            m2, _dl = _build_recoded_member(codec_ctx, source_payload, sel_payload, dqs1_tail, rj2, codec, layout)
            r = score_member(m2, n_pairs, f"probe_{name}")
            # Entropy delta of THIS tensor's stream (byte proxy).
            qb = np.frombuffer(raw_joined, dtype=np.uint8, count=numel, offset=offsets[sidx])
            H_full = q_byte_entropy_bits(qb)
            q_signed = decode_byte_map_u8(qb, byte_map)
            H_rq = q_byte_entropy_bits(encode_byte_map_u8(requant_signed_q(q_signed, probe_levels), byte_map))
            d_seg_sens = r["mean_d_seg"] - baseline["mean_d_seg"]
            d_pose_sens = r["mean_d_pose"] - baseline["mean_d_pose"]
            rows.append({
                "storage_index": sidx, "name": name, "numel": numel, "byte_map": byte_map,
                "probe_levels": probe_levels,
                "d_seg_sensitivity": d_seg_sens, "d_pose_sensitivity": d_pose_sens,
                "H_full_bits": H_full, "H_rq_bits": H_rq,
                "seconds": r["seconds"],
            })
            print(f"[rank] {name:28s} numel={numel:6d} Δd_seg={d_seg_sens:+.8f} "
                  f"Δd_pose={d_pose_sens:+.8e} H {H_full:.2f}->{H_rq:.2f}", flush=True)
        results["baseline_subset"] = baseline
        results["rank_rows"] = rows
        # Composite scorer sensitivity (marginal score units): use pointer operating
        # point coefficients: ∂S/∂d_seg=100, ∂S/∂d_pose=5/sqrt(10*d_pose).
        import math
        pose0 = max(baseline["mean_d_pose"], 1e-9)
        dpose_coef = 5.0 / math.sqrt(10.0 * pose0)
        for r in rows:
            r["score_sensitivity"] = 100.0 * abs(r["d_seg_sensitivity"]) + dpose_coef * abs(r["d_pose_sensitivity"])
        rows.sort(key=lambda r: -r["score_sensitivity"])
        results["ranked_by_score_sensitivity"] = [
            {"name": r["name"], "storage_index": r["storage_index"], "numel": r["numel"],
             "score_sensitivity": r["score_sensitivity"]} for r in rows
        ]
        print("\n[rank] TENSORS BY SCORE-SENSITIVITY (most->least):", flush=True)
        for r in rows:
            print(f"  {r['name']:28s} numel={r['numel']:6d} score_sens={r['score_sensitivity']:.6f}", flush=True)

    elif args.stage == "sweep":
        rank_file = Path(args.rank_file).resolve() if args.rank_file else None
        if not rank_file or not rank_file.exists():
            raise SystemExit("--rank-file (a prior --stage rank JSON) is required for sweep")
        rank_data = json.loads(rank_file.read_text())
        rank_rows = rank_data["rank_rows"]
        sensitivities = {int(r["storage_index"]): float(r["score_sensitivity"]) for r in rank_rows}
        numels = {int(r["storage_index"]): int(r["numel"]) for r in rank_rows}
        names = {int(r["storage_index"]): r["name"] for r in rank_rows}
        # baseline full-pairs
        base = score_member(member, n_pairs, "baseline_full")
        base_size, base_sha = _byte_close_archive(member, out_dir / "baseline")
        base_sc = contest_score_from_components(
            d_seg=base["mean_d_seg"], d_pose=base["mean_d_pose"], archive_zip_size=base_size,
        )
        base.update({"archive_zip_size": base_size, "score": base_sc["score"], "score_components": base_sc})
        print(f"[sweep] baseline_full d_seg={base['mean_d_seg']:.8f} d_pose={base['mean_d_pose']:.8e} "
              f"bytes={base_size} score={base_sc['score']:.8f}", flush=True)

        curve = []
        # Least-sensitive-first ordering of NON-protected, eligible tensors.
        eligible = [
            sidx for sidx in sorted(sensitivities, key=lambda k: sensitivities[k])
            if sidx != DQS1_PROTECTED_STORAGE_INDEX and numels[sidx] >= args.min_numel
        ]

        def _build_plan_specs():
            """Yield (label, levels_by_sidx). Two modes:
            --plans 'n_crush:levels,...'  -> crush the n LEAST-sensitive eligible
                                              tensors to `levels` (waterfill probe);
            else fall back to the threshold allocator over score_sensitivity.
            """
            if args.plans:
                for spec in args.plans.split(","):
                    n_crush_s, lv_s = spec.split(":")
                    n_crush, lv = int(n_crush_s), int(lv_s)
                    lvls = {s: 256 for s in sensitivities}
                    for sidx in eligible[:n_crush]:
                        lvls[sidx] = lv
                    lvls[DQS1_PROTECTED_STORAGE_INDEX] = 256
                    yield f"crush{n_crush}_int{int(round(np.log2(lv)))}", lvls
            else:
                for thr in [float(t) for t in args.thresholds.split(",")]:
                    plans = allocate_bits_by_sensitivity(
                        sensitivities=sensitivities, numels=numels, names=names,
                        sensitivity_threshold=thr, protect_top_k=int(args.protect_top_k),
                    )
                    lvls = {sidx: p.levels for sidx, p in plans.items()}
                    lvls[DQS1_PROTECTED_STORAGE_INDEX] = 256
                    yield f"thr_{thr:g}", lvls

        for label, levels_by_sidx in _build_plan_specs():
            rj2 = _requant_raw_joined(raw_joined, layout, levels_by_sidx)
            m2, dec_sec_len = _build_recoded_member(
                codec_ctx, source_payload, sel_payload, dqs1_tail, rj2, codec, layout
            )
            cand_dir = out_dir / label
            cand_dir.mkdir(parents=True, exist_ok=True)
            size, sha = _byte_close_archive(m2, cand_dir)
            r = score_member(m2, n_pairs, label)
            sc = contest_score_from_components(
                d_seg=r["mean_d_seg"], d_pose=r["mean_d_pose"], archive_zip_size=size,
            )
            delta = score_delta_components(base=base_sc, cand=sc)
            crushed_names = {names[s]: levels_by_sidx[s] for s in sorted(levels_by_sidx) if levels_by_sidx[s] < 256}
            point = {
                "label": label,
                "crushed_tensors": crushed_names,
                "n_crushed": len(crushed_names),
                "archive_zip_size": size, "archive_sha256": sha,
                "byte_delta_vs_baseline": size - base_size,
                "mean_d_seg": r["mean_d_seg"], "mean_d_pose": r["mean_d_pose"],
                "score": sc["score"], "score_delta_vs_baseline": delta["d_score"],
                "score_components": sc, "delta_components": delta,
                "seconds": r["seconds"],
            }
            curve.append(point)
            print(f"[sweep] {label:16s} crushed={point['n_crushed']:2d} bytes={size} "
                  f"(Δ{size-base_size:+d}) d_seg={r['mean_d_seg']:.8f} d_pose={r['mean_d_pose']:.8e} "
                  f"score={sc['score']:.8f} (Δ{delta['d_score']:+.8f})", flush=True)
        results["baseline_full"] = base
        results["operating_point_curve"] = curve
        # Best (lowest score) candidate.
        best = min(curve, key=lambda p: p["score"]) if curve else None
        results["best_candidate"] = best
        if best:
            print(f"\n[sweep] BEST: thr={best['threshold']:g} score={best['score']:.8f} "
                  f"(frontier {_frontier_pointer_cpu():.8f}, Δ{best['score']-_frontier_pointer_cpu():+.8f})", flush=True)

    out_json = out_dir / f"{args.stage}_result.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {out_json}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--stage", choices=["baseline", "rank", "sweep"], required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rank-file", default=None, help="prior --stage rank JSON (for sweep)")
    ap.add_argument("--rank-pairs", type=int, default=120)
    ap.add_argument("--sweep-pairs", type=int, default=600)
    ap.add_argument("--probe-levels", type=int, default=4)
    ap.add_argument("--min-numel", type=int, default=5000)
    ap.add_argument("--thresholds", default="0.02,0.05,0.1,0.2,0.5")
    ap.add_argument("--plans", default=None,
                    help="direct waterfill sweep 'n_crush:levels,...' "
                         "(crush the n LEAST-sensitive eligible tensors to `levels`)")
    ap.add_argument("--protect-top-k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
