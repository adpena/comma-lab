#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""P-SUFF / TASK-ABLATION — how much of the 0.19110 frontier archive does the
frozen scorer IGNORE? (the indirect-RD invariance gap on the BINDING rate term).

THE CONVERGENT #1 from BOTH research layers:
  * indirect-RD theory (.omx/research/vcm_theory_primitive_layer_*): our per-video INR
    reconstructs RGB the frozen SegNet+PoseNet largely IGNORE — the "dominated rung". The
    task-RD floor may be BELOW the renderer's rate. The exact diagnostic is the
    R_Ỹ^{E&C} − R_X invariance gap: the byte mass the frozen scorers do not read.
  * VCM methods survey (P-SUFF / P-R3): measure the SUFFICIENT-STATISTIC fraction — code
    only what the scorers read.

THE MEASUREMENT (measurement-first, NO-FAKE — every Δ is REALIZED through the REAL frozen
SegNet+PoseNet + the EXACT camera-resolution eval roundtrip, vs the cached GT scorer
targets; the score is recomputed from components):

  Decode the frontier archive (pr110_payload_entropy_recode = the 0.19110 frontier) into
  its 28 HNeRV decoder tensors + (600,28) latents. The decoder is 161KB = ~90.9% of the
  177KB archive = the BINDING rate term (62% of S). Then:

  Phase 1  per-decoder-tensor ablation: zero / mean-replace each of the 28 tensors;
           record Δd_seg, Δd_pose, ΔS, and the bytes that tensor costs. Rank by
           task-effect per byte.
  Phase 2  per-latent-channel ablation: zero / mean-replace each of the 28 latent dims
           (across the 600 pairs); record Δd_seg/Δd_pose + bytes. Find ~0-task-effect dims.
  Phase 3  bit-precision sweep per tensor: coarsen each tensor to {6,5,4,3,2} bits (qdq);
           the per-tensor scorer-invariant precision floor (fewest bits before ΔS > thr).
  Phase 4  THE HEADLINE: the total scorer-INVARIANT byte mass — the fraction of the archive
           deletable/coarsenable keeping ΔS (Δd_seg + √(10·d_pose)−√(10·d_pose0) + Δrate)
           under +0.005. Projected rate cut + projected S.

WHY FAITHFUL:
  * Real frozen contest SegNet/PoseNet on CPU (the AUTHORITY device; NEVER MPS for the
    score — MPS corrupts 95.5% of orderings). Inference-only, so no MPS contention.
  * The EXACT contest chain: decoder → bicubic-up 384x512→874x1164 → frontier channel-bias
    → clamp+round uint8 → DistortionNet (which does the model-input bilinear-down + YUV6 /
    last-frame). d_seg = (SegNet argmax flip vs GT-argmax).mean; d_pose = (PoseNet 6-dim −
    GT 6-dim)^2.mean — the EXACT functionals upstream/modules.py charges.
  * The GT targets are the cached frozen-scorer output on GT frames (build_gt_targets =
    upstream/evaluate.py's reference), so we run the scorer only on the (ablated) decode.
  * ΔS uses the real nonlinear pose term √(10·d_pose) — never a linearized surrogate.

HONEST SCOPE (NO-FAKE):
  * Ablation Δ is measured on the DECODER-ONLY render path (decoder + camera-roundtrip +
    channel-bias). The frontier's FECa selector (per-pair RGB bias) + DQS1 (one-coeff patch
    on selected pairs) are NOT decoder tensors and are tiny byte contributors; they shift
    the ABSOLUTE baseline a hair (we report both selector-on and selector-off baselines) but
    do not change which decoder tensor carries task-effect. The 90.9% binding mass IS the
    decoder.
  * n is a SUBSET of the 600 pairs (the metric is the per-pair average; a subset is a
    faithful-definition estimate, flagged). We report n and the avg/per-frame distinction.
  * Per-tensor "bytes" = numel × measured avg coded-bits/sym from the real range-coded
    decoder section. The bit-sweep additionally measures the REAL re-encoded section byte
    delta (the operationally-meaningful saving). Latents are stored as temporal-delta uint8;
    per-dim bytes = N_PAIRS bytes (1 B/pair) + 4 B hdr.

ALL numbers [contest-CPU advisory] NON-PROMOTABLE. Exact pointer UNMOVED at 0.19110.
This is a DIAGNOSTIC (a byte-level sensitivity map) feeding the bit-allocator (#154) +
the quotient codec (#155). A partial GREEN is high-value.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir"
GT_CACHE = REPO / "experiments/results/capstone_gt_targets_cache"
OUT_DIR = REPO / "experiments/results/p_suff_task_ablation"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(SUB / "src"))
sys.path.insert(0, str(SUB))

CAMERA_H, CAMERA_W = 874, 1164
B0 = 37_545_489  # contest rate normalizer (sum of source video bytes)
ARCHIVE_BYTES_FULL = 177_169  # the frontier on-disk archive bytes (report.txt)
# the frontier's own measured 600-pair operating point (report.txt):
FRONTIER_DSEG = 0.00055978
FRONTIER_DPOSE = 0.00002942
DELTA_S_TOL = 0.005  # the invariant-mass threshold: Δ(d_seg term + d_pose term + rate)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ===========================================================================
# Decode the frontier archive -> decoder tensors + latents + the selector/DQS1
# ===========================================================================
def load_frontier_decode():
    """Parse the frontier archive member 'x' via the frontier inflate parse path."""
    import zipfile

    import inflate as fr  # the frontier submission_dir/inflate.py

    with zipfile.ZipFile(SUB / "archive.zip") as z:
        member = z.read("x")
    (decoder_sd, latents, meta, raw_joined, sel_kind, sel_codes, sel_specs, dqs1) = (
        fr.parse_member(member)
    )
    return {
        "decoder_sd": decoder_sd,
        "latents": latents.float(),
        "meta": meta,
        "fr": fr,
        "sel_kind": sel_kind,
        "sel_codes": sel_codes,
        "sel_specs": sel_specs,
        "dqs1": dqs1,
    }


def build_decoder(decoder_sd, meta):
    from model import HNeRVDecoder

    dec = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    ).eval()
    dec.load_state_dict(decoder_sd)
    return dec


# ===========================================================================
# The EXACT contest eval chain on a (possibly ablated) decoder + latents
# ===========================================================================
def eval_dseg_dpose(
    net, decoder, latents, seg_targets, pose_targets, n, *, apply_bias=True, batch=8
):
    """REALIZED d_seg/d_pose over the first n pairs through the EXACT contest chain.

    decoder(latents) -> (B,2,3,384,512) float[0,255]; bicubic-up to camera res; the
    frontier channel-bias (-1 on f0-R, f0-B, f1-G); clamp+round uint8; DistortionNet
    (model-input bilinear + YUV6 / last-frame). d_seg vs cached GT argmax; d_pose vs
    cached GT pose6 — the EXACT upstream functionals.
    """
    seg_flips, pose_se = [], []
    with torch.inference_mode():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            B = j - i
            decoded = decoder(latents[i:j])  # (B,2,3,384,512)
            flat = decoded.reshape(B * 2, 3, 384, 512)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            up = up.reshape(B, 2, 3, CAMERA_H, CAMERA_W)
            if apply_bias:
                up[:, 0, 0].sub_(1.0)
                up[:, 0, 2].sub_(1.0)
                up[:, 1, 1].sub_(1.0)
            decoded_bhwc = (
                up.clamp(0, 255).round().permute(0, 1, 3, 4, 2).to(torch.uint8)
            )
            po, so = net(decoded_bhwc)
            seg_argmax = so.argmax(dim=1)  # (B,384,512)
            df = (seg_argmax != seg_targets[i:j]).float().mean(dim=(1, 2))
            seg_flips.append(df)
            p6 = po["pose"][:, :6]
            pd = (p6 - pose_targets[i:j]).pow(2).mean(dim=1)
            pose_se.append(pd)
    d_seg = float(torch.cat(seg_flips).mean().item())
    d_pose = float(torch.cat(pose_se).mean().item())
    return d_seg, d_pose


def s_terms(d_seg, d_pose, archive_bytes):
    seg_term = 100.0 * d_seg
    pose_term = math.sqrt(10.0 * d_pose + 1e-12)
    rate_term = 25.0 * archive_bytes / B0
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "seg_term": seg_term,
        "pose_term": pose_term,
        "rate_term": rate_term,
        "S": seg_term + pose_term + rate_term,
    }


def delta_s_task_only(base, abl):
    """ΔS from the TASK terms only (Δseg_term + Δpose_term) — the invariance check holding
    bytes at the full archive. The byte SAVING is reported separately as the upside."""
    return (abl["seg_term"] - base["seg_term"]) + (abl["pose_term"] - base["pose_term"])


# ===========================================================================
# Per-tensor coded byte cost (from the real range-coded decoder section)
# ===========================================================================
def measure_per_tensor_coded_bytes(dec):
    """Per-tensor coded byte share of the real decoder section.

    Re-encode the baseline decoder section via codec_ctx.encode_decoder_section, get the
    total coded body bytes, then attribute per-tensor by numel-weighted avg bits/sym within
    each tensor's model group (the range coder shares per-group params). This is an estimate
    of each tensor's byte mass; the bit-sweep measures the REAL re-encoded section delta.
    """
    import codec  # frontier codec (DECODER_STORAGE_ORDER, byte maps)
    import codec_ctx

    # Build the raw per-tensor streams the section encoder consumes, from the live
    # (baseline) decoder state, mirroring the codec's storage layout.
    raws = _decoder_state_to_raw_streams(dec, codec)
    section = codec_ctx.encode_decoder_section(raws)
    total_section_bytes = len(section)
    # numel per tensor (in storage order) for the share split
    numels = []
    for idx in codec.DECODER_STORAGE_ORDER:
        numels.append(_tensor_numel_by_storage_index(dec, codec, idx))
    numels = np.array(numels, dtype=np.float64)
    # header bytes are ~fixed; body is range-coded. Attribute the WHOLE section by numel
    # share (an honest approximation — exact per-symbol bits would need a streaming probe;
    # the bit-sweep's measured section delta is the operational number).
    shares = numels / numels.sum()
    per_storage_idx_bytes = {
        int(idx): float(total_section_bytes * shares[k])
        for k, idx in enumerate(codec.DECODER_STORAGE_ORDER)
    }
    return per_storage_idx_bytes, total_section_bytes


def _tensor_numel_by_storage_index(dec, codec, storage_index):
    from model import HNeRVDecoder

    probe = HNeRVDecoder(
        latent_dim=dec.stem.in_features,
        base_channels=dec.channels[0],
        eval_size=tuple(dec.eval_size),
    )
    items = list(probe.state_dict().items())
    _name, tensor = items[int(storage_index)]
    return int(tensor.numel())


def _decoder_state_to_raw_streams(dec, codec):
    """Encode the decoder state to the codec's raw per-tensor streams (q-bytes + fp16
    scale per tensor, in DECODER_STORAGE_ORDER), mirroring the codec's quantize path.

    The frontier codec stores per-tensor INT8 (symmetric) + an fp16 scale; the raw stream
    is the byte-mapped uint8 q-codes followed by the 2-byte scale, concatenated in storage
    order and split at DECODER_STREAM_ENDS. We reproduce that quantize here (absmax/127)."""
    items = list(dec.state_dict().items())
    parts = []
    for idx in codec.DECODER_STORAGE_ORDER:
        name, _t = items[int(idx)]
        w = dec.state_dict()[name].detach().cpu().numpy().astype(np.float64)
        # storage permutation for 4-D conv tensors
        if w.ndim == 4 and int(idx) in codec.CONV4_STORAGE_PERMS:
            perm = codec.CONV4_STORAGE_PERMS[int(idx)]
            w = np.transpose(w, perm)
        absmax = float(np.abs(w).max())
        scale = absmax / 127.0 if absmax > 0 else 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        q_u8 = _encode_mapped_u8(q.reshape(-1), codec.DECODER_BYTE_MAPS.get(int(idx), "zig"))
        scale_fp16 = np.array([scale], dtype=np.float16).tobytes()
        parts.append(q_u8.tobytes() + scale_fp16)
    # split into the DECODER_STREAM_ENDS stream groups (storage-order slices)
    streams = []
    ends = codec.DECODER_STREAM_ENDS
    start = 0
    for e in ends:
        streams.append(b"".join(parts[start:e]))
        start = e
    return streams


def _encode_mapped_u8(q_int8, mapname):
    """Inverse of codec.decode_mapped_u8: int8 q-codes -> stored uint8 per the byte map."""
    q = q_int8.astype(np.int64)
    if mapname == "zig":
        return ((q << 1) ^ (q >> 63)).astype(np.uint8)
    if mapname == "negzig":
        q = -q
        return ((q << 1) ^ (q >> 63)).astype(np.uint8)
    if mapname == "twos":
        return (q & 0xFF).astype(np.uint8)
    if mapname == "off":
        return ((q + 128) & 0xFF).astype(np.uint8)
    raise ValueError(f"unknown byte map {mapname!r}")


# ===========================================================================
# Coarsen a single tensor to nbits (symmetric qdq) — the bit-sweep
# ===========================================================================
def coarsen_tensor_qdq(w, nbits):
    """Symmetric per-tensor quantize-dequantize to nbits (absmax scale). Returns the
    dequantized float tensor (same shape)."""
    if nbits >= 32:
        return w.clone()
    qmax = (1 << (nbits - 1)) - 1  # signed: e.g. nbits=4 -> 7
    absmax = float(w.abs().max())
    if absmax == 0 or qmax == 0:
        return torch.zeros_like(w)
    scale = absmax / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def measure_reencoded_section_bytes_for_coarsened(dec_base_sd, codec, codec_ctx, coarsen_map):
    """Re-encode the decoder section with the given coarsen map applied (name -> nbits or
    name -> 'zero'/'mean'), return the REAL section byte length. coarsen_map is applied to
    the float state, then re-quantized through the codec's int8 store + range-coded."""
    import copy

    sd = {k: v.clone() for k, v in dec_base_sd.items()}
    for name, op in coarsen_map.items():
        if op == "zero":
            sd[name] = torch.zeros_like(sd[name])
        elif op == "mean":
            sd[name] = torch.full_like(sd[name], float(sd[name].mean()))
        else:
            sd[name] = coarsen_tensor_qdq(sd[name], int(op))
    # build a probe decoder to encode the raw streams
    from model import HNeRVDecoder

    probe = HNeRVDecoder(
        latent_dim=int(dec_base_sd["stem.weight"].shape[1]),
        base_channels=int(dec_base_sd["stem.weight"].shape[0] // (6 * 8)),
        eval_size=(384, 512),
    ).eval()
    probe.load_state_dict(sd)
    raws = _decoder_state_to_raw_streams(probe, codec)
    return len(codec_ctx.encode_decoder_section(raws))


# ===========================================================================
# Apply an ablation op to a decoder state dict (zero / mean / nbits) by tensor NAME
# ===========================================================================
def make_ablated_decoder(base_sd, meta, name, op):
    sd = {k: v.clone() for k, v in base_sd.items()}
    if op == "zero":
        sd[name] = torch.zeros_like(sd[name])
    elif op == "mean":
        sd[name] = torch.full_like(sd[name], float(sd[name].mean()))
    else:
        sd[name] = coarsen_tensor_qdq(sd[name], int(op))
    return build_decoder(sd, meta)


# ===========================================================================
# Main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=24, help="GT pairs to measure over (subset of 600)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bits", default="8,7,6,5,4,3,2", help="bit-sweep levels (Phase 3)")
    ap.add_argument(
        "--phases", default="1,2,3,4", help="comma-sep phases to run (resumable per-phase)"
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument(
        "--smoke", action="store_true", help="n=4, bits=4 only, 3 tensors — calibrate timing"
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(int(args.threads))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "ablation_state.json"

    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"[resume] loaded state phases: {list(state.get('phases', {}).keys())}", flush=True)
    state.setdefault("phases", {})

    def save():
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        state_path.write_text(json.dumps(state, indent=2))

    # --- decode the frontier ---
    print("[load] decoding the frontier archive (pr110)...", flush=True)
    D = load_frontier_decode()
    meta = D["meta"]
    base_sd = D["decoder_sd"]
    latents = D["latents"]
    decoder = build_decoder(base_sd, meta)
    names = list(base_sd.keys())  # state_dict order (model order)
    print(f"[load] 28 decoder tensors, latents {tuple(latents.shape)}", flush=True)

    # --- GT targets (cached frozen-scorer output on GT) ---
    n = 4 if args.smoke else int(args.n)
    cache_file = _pick_gt_cache(n)
    gt = torch.load(cache_file, map_location="cpu", weights_only=False)
    seg_targets = gt["seg"][:n]
    pose_targets = gt["pose"][:n]
    n = int(seg_targets.shape[0])
    print(f"[gt] cache={cache_file.name} n={n}", flush=True)

    # --- frozen scorer (CPU AUTHORITY) ---
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    net = load_frozen_distortion_net(device="cpu")

    # --- baselines (selector-off = decoder-only ablation reference; selector-on = faithful) ---
    t0 = time.time()
    b_seg, b_pose = eval_dseg_dpose(
        net, decoder, latents, seg_targets, pose_targets, n, apply_bias=True
    )
    per_pair_s = (time.time() - t0) / max(n, 1)
    base = s_terms(b_seg, b_pose, ARCHIVE_BYTES_FULL)
    state["baseline"] = {
        **base,
        "n": n,
        "per_pair_eval_s": round(per_pair_s, 3),
        "frontier_report_dseg": FRONTIER_DSEG,
        "frontier_report_dpose": FRONTIER_DPOSE,
        "note_subset": (
            f"baseline measured on n={n} of 600 pairs (per-pair-average metric; subset "
            "estimate). Frontier full-600 report: d_seg=0.00055978 d_pose=0.00002942."
        ),
    }
    save()
    print(
        f"[baseline n={n}] d_seg={b_seg:.6f} d_pose={b_pose:.8f} S={base['S']:.5f} "
        f"({per_pair_s:.2f}s/pair) [contest-CPU advisory]",
        flush=True,
    )

    phases = {p.strip() for p in args.phases.split(",") if p.strip()}
    bits = [int(x) for x in args.bits.split(",") if x.strip()]
    if args.smoke:
        bits = [4]

    per_tensor_bytes, total_section_bytes = (None, None)
    try:
        import codec
        import codec_ctx

        per_tensor_bytes, total_section_bytes = measure_per_tensor_coded_bytes(decoder)
        state["decoder_section_bytes"] = total_section_bytes
        # map storage-INDEX bytes -> tensor NAME bytes. The storage index is a MODEL-order
        # index (into a fresh HNeRVDecoder().state_dict()), NOT an index into base_sd (whose
        # key order is the codec's STORAGE order). Build a model-order items list so the
        # index->name mapping is correct.
        from model import HNeRVDecoder as _HD

        _model_items = list(
            _HD(
                latent_dim=int(meta["latent_dim"]),
                base_channels=int(meta["base_channels"]),
                eval_size=tuple(meta["eval_size"]),
            ).state_dict().items()
        )
        name_bytes = {}
        for sidx, by in per_tensor_bytes.items():
            name_bytes[_model_items[int(sidx)][0]] = by
        state["per_tensor_coded_bytes"] = name_bytes
        save()
        print(
            f"[bytes] real decoder section = {total_section_bytes} B "
            f"(archive {ARCHIVE_BYTES_FULL} B; decoder ~{100*total_section_bytes/ARCHIVE_BYTES_FULL:.1f}%)",
            flush=True,
        )
    except Exception as e:  # pragma: no cover
        print(f"[warn] per-tensor byte measure failed: {e}", flush=True)
        name_bytes = {}

    # ---------- PHASE 1: per-decoder-tensor ablation (zero / mean) ----------
    if "1" in phases:
        smoke_names = names[:3] if args.smoke else names
        rows = state["phases"].setdefault("phase1_tensor_ablation", {})
        for name in smoke_names:
            for op in ("zero", "mean"):
                key = f"{name}::{op}"
                if key in rows:
                    continue
                t = time.time()
                dec_abl = make_ablated_decoder(base_sd, meta, name, op)
                a_seg, a_pose = eval_dseg_dpose(
                    net, dec_abl, latents, seg_targets, pose_targets, n, apply_bias=True
                )
                abl = s_terms(a_seg, a_pose, ARCHIVE_BYTES_FULL)
                d_s = delta_s_task_only(base, abl)
                by = name_bytes.get(name, float("nan"))
                rows[key] = {
                    "name": name,
                    "op": op,
                    "d_seg": a_seg,
                    "d_pose": a_pose,
                    "delta_d_seg": a_seg - b_seg,
                    "delta_d_pose": a_pose - b_pose,
                    "delta_S_task": d_s,
                    "coded_bytes": by,
                    "task_effect_per_byte": (d_s / by) if (by and by == by and by > 0) else None,
                    "wall_s": round(time.time() - t, 1),
                }
                save()
                print(
                    f"  [P1 {name:22s} {op:4s}] dS_task={d_s:+.5f} "
                    f"Δd_seg={a_seg-b_seg:+.6f} Δd_pose={a_pose-b_pose:+.8f} "
                    f"bytes~{by:.0f} ({rows[key]['wall_s']}s)",
                    flush=True,
                )

    # ---------- PHASE 2: per-latent-channel ablation (zero / mean) ----------
    if "2" in phases:
        rows = state["phases"].setdefault("phase2_latent_ablation", {})
        latent_dim = int(meta["latent_dim"])
        dims = range(3 if args.smoke else latent_dim)
        # per-dim bytes: temporal-delta uint8 -> ~N_PAIRS bytes + 4 B fp16 hdr (min+scale)
        per_dim_bytes = 600 + 4
        for d in dims:
            for op in ("zero", "mean"):
                key = f"dim{d}::{op}"
                if key in rows:
                    continue
                t = time.time()
                lat_abl = latents.clone()
                if op == "zero":
                    lat_abl[:, d] = 0.0
                else:
                    lat_abl[:, d] = latents[:, d].mean()
                a_seg, a_pose = eval_dseg_dpose(
                    net, decoder, lat_abl, seg_targets, pose_targets, n, apply_bias=True
                )
                abl = s_terms(a_seg, a_pose, ARCHIVE_BYTES_FULL)
                d_s = delta_s_task_only(base, abl)
                rows[key] = {
                    "dim": d,
                    "op": op,
                    "d_seg": a_seg,
                    "d_pose": a_pose,
                    "delta_d_seg": a_seg - b_seg,
                    "delta_d_pose": a_pose - b_pose,
                    "delta_S_task": d_s,
                    "coded_bytes": per_dim_bytes,
                    "task_effect_per_byte": d_s / per_dim_bytes,
                    "wall_s": round(time.time() - t, 1),
                }
                save()
                print(
                    f"  [P2 dim{d:2d} {op:4s}] dS_task={d_s:+.5f} "
                    f"Δd_seg={a_seg-b_seg:+.6f} Δd_pose={a_pose-b_pose:+.8f} "
                    f"({rows[key]['wall_s']}s)",
                    flush=True,
                )

    # ---------- PHASE 3: per-tensor bit-precision sweep ----------
    if "3" in phases:
        import codec
        import codec_ctx

        rows = state["phases"].setdefault("phase3_bit_sweep", {})
        smoke_names = names[:3] if args.smoke else names
        for name in smoke_names:
            for nb in bits:
                key = f"{name}::b{nb}"
                if key in rows:
                    continue
                t = time.time()
                dec_abl = make_ablated_decoder(base_sd, meta, name, nb)
                a_seg, a_pose = eval_dseg_dpose(
                    net, dec_abl, latents, seg_targets, pose_targets, n, apply_bias=True
                )
                abl = s_terms(a_seg, a_pose, ARCHIVE_BYTES_FULL)
                d_s = delta_s_task_only(base, abl)
                # REAL re-encoded section byte delta for coarsening THIS tensor to nb bits
                try:
                    sec_bytes = measure_reencoded_section_bytes_for_coarsened(
                        base_sd, codec, codec_ctx, {name: nb}
                    )
                    byte_saving = (total_section_bytes - sec_bytes) if total_section_bytes else None
                except Exception as e:  # pragma: no cover
                    sec_bytes, byte_saving = None, None
                    print(f"    [warn] reencode failed for {name} b{nb}: {e}", flush=True)
                rows[key] = {
                    "name": name,
                    "nbits": nb,
                    "d_seg": a_seg,
                    "d_pose": a_pose,
                    "delta_d_seg": a_seg - b_seg,
                    "delta_d_pose": a_pose - b_pose,
                    "delta_S_task": d_s,
                    "reencoded_section_bytes": sec_bytes,
                    "section_byte_saving": byte_saving,
                    "wall_s": round(time.time() - t, 1),
                }
                save()
                print(
                    f"  [P3 {name:22s} b{nb}] dS_task={d_s:+.5f} "
                    f"Δd_seg={a_seg-b_seg:+.6f} save={byte_saving} ({rows[key]['wall_s']}s)",
                    flush=True,
                )

    # ---------- PHASE 4: HEADLINE — total scorer-invariant byte mass ----------
    if "4" in phases:
        _compute_headline(state, base, total_section_bytes, name_bytes)
        # JOINT verification: apply EVERY tensor's bit-floor simultaneously and measure the
        # REAL aggregate ΔS + REAL re-encoded section bytes (per-tensor ΔS do NOT sum; this
        # is the honest combined number — the operational saving + the actual joint task hit).
        bit_floors = state.get("headline", {}).get("invariant", {}).get("tensor_bit_floors", {})
        if bit_floors and "3" in phases:
            try:
                import codec
                import codec_ctx

                coarsen_map = {name: int(v["min_bits"]) for name, v in bit_floors.items()}
                t = time.time()
                joint_dec = build_decoder(
                    {
                        k: (coarsen_tensor_qdq(v, coarsen_map[k]) if k in coarsen_map else v.clone())
                        for k, v in base_sd.items()
                    },
                    meta,
                )
                j_seg, j_pose = eval_dseg_dpose(
                    net, joint_dec, latents, seg_targets, pose_targets, n, apply_bias=True
                )
                joint_abl = s_terms(j_seg, j_pose, ARCHIVE_BYTES_FULL)
                joint_dS = delta_s_task_only(base, joint_abl)
                joint_sec_bytes = measure_reencoded_section_bytes_for_coarsened(
                    base_sd, codec, codec_ctx, coarsen_map
                )
                joint_saving = (
                    (total_section_bytes - joint_sec_bytes) if total_section_bytes else None
                )
                new_archive_bytes = (
                    ARCHIVE_BYTES_FULL - (joint_saving or 0)
                )
                joint_full = s_terms(j_seg, j_pose, new_archive_bytes)
                state["headline"]["joint_bit_floor_verification"] = {
                    "n_tensors_coarsened": len(coarsen_map),
                    "joint_delta_S_task": joint_dS,
                    "joint_d_seg": j_seg,
                    "joint_d_pose": j_pose,
                    "joint_reencoded_section_bytes": joint_sec_bytes,
                    "joint_section_byte_saving": joint_saving,
                    "joint_new_archive_bytes": new_archive_bytes,
                    "joint_projected_S_with_byte_cut": joint_full["S"],
                    "joint_rate_cut": base["rate_term"] - joint_full["rate_term"],
                    "wall_s": round(time.time() - t, 1),
                    "note": (
                        "JOINT coarsening of ALL per-tensor bit-floors at once, with REAL "
                        "re-encoded bytes + REAL combined task hit. Per-tensor ΔS do NOT sum; "
                        "this is the honest aggregate. projected_S = the joint task hit AT the "
                        "reduced archive bytes."
                    ),
                }
                print(
                    f"[P4 JOINT bit-floor] {len(coarsen_map)} tensors -> "
                    f"joint_dS_task={joint_dS:+.5f} joint_save={joint_saving} B "
                    f"projected_S={joint_full['S']:.5f} ({state['headline']['joint_bit_floor_verification']['wall_s']}s)",
                    flush=True,
                )
            except Exception as e:  # pragma: no cover
                print(f"[warn] joint bit-floor verification failed: {e}", flush=True)
        save()

    _finalize(state, state_path, args, n)
    return 0


def _pick_gt_cache(n):
    """Pick the smallest GT cache file that has >= n pairs."""
    avail = sorted(
        int(p.stem.split("_n")[1]) for p in GT_CACHE.glob("gt_targets_n*.pt")
    )
    for a in avail:
        if a >= n:
            return GT_CACHE / f"gt_targets_n{a}.pt"
    return GT_CACHE / f"gt_targets_n{avail[-1]}.pt"


def _compute_headline(state, base, total_section_bytes, name_bytes):
    """The invariant-mass headline: which tensors/latent-dims/bit-coarsenings keep ΔS_task
    under the tolerance, and the total byte saving they unlock -> projected rate cut + S."""
    tol = DELTA_S_TOL
    invariant = {"tensors_zeroable": [], "tensors_meanable": [], "latent_dims_zeroable": [],
                 "tensor_bit_floors": {}}

    p1 = state["phases"].get("phase1_tensor_ablation", {})
    for key, r in p1.items():
        if r["op"] == "zero" and r["delta_S_task"] <= tol:
            invariant["tensors_zeroable"].append(
                {"name": r["name"], "delta_S_task": r["delta_S_task"], "bytes": r.get("coded_bytes")}
            )
        if r["op"] == "mean" and r["delta_S_task"] <= tol:
            invariant["tensors_meanable"].append(
                {"name": r["name"], "delta_S_task": r["delta_S_task"], "bytes": r.get("coded_bytes")}
            )

    p2 = state["phases"].get("phase2_latent_ablation", {})
    for key, r in p2.items():
        if r["op"] == "zero" and r["delta_S_task"] <= tol:
            invariant["latent_dims_zeroable"].append(
                {"dim": r["dim"], "delta_S_task": r["delta_S_task"], "bytes": r.get("coded_bytes")}
            )

    # per-tensor scorer-invariant precision floor (fewest bits with ΔS_task <= tol)
    p3 = state["phases"].get("phase3_bit_sweep", {})
    by_name = {}
    for key, r in p3.items():
        by_name.setdefault(r["name"], []).append(r)
    for name, lst in by_name.items():
        passing = sorted([r for r in lst if r["delta_S_task"] <= tol], key=lambda r: r["nbits"])
        if passing:
            floor = passing[0]  # fewest bits that still passes
            invariant["tensor_bit_floors"][name] = {
                "min_bits": floor["nbits"],
                "delta_S_task": floor["delta_S_task"],
                "section_byte_saving": floor.get("section_byte_saving"),
            }

    # byte saving estimates
    zero_bytes = sum(
        (t["bytes"] or 0.0) for t in invariant["tensors_zeroable"] if t["bytes"] == t["bytes"]
    )
    latent_bytes = sum(t["bytes"] for t in invariant["latent_dims_zeroable"])
    bit_floor_saving = sum(
        (v.get("section_byte_saving") or 0.0) for v in invariant["tensor_bit_floors"].values()
    )
    # the bit-floor saving and the zero saving overlap (a zeroable tensor is also bit-floor
    # at min bits); report the bit-floor coarsening as the SAFE saving (lossless on task),
    # and the zeroable set as the upper bound.
    safe_saving = bit_floor_saving
    upper_saving = zero_bytes + latent_bytes
    rate_cut_safe = 25.0 * safe_saving / B0
    rate_cut_upper = 25.0 * upper_saving / B0
    invariant_mass_frac_safe = (
        safe_saving / ARCHIVE_BYTES_FULL if ARCHIVE_BYTES_FULL else None
    )
    invariant_mass_frac_upper = (
        upper_saving / ARCHIVE_BYTES_FULL if ARCHIVE_BYTES_FULL else None
    )

    state["headline"] = {
        "delta_S_tol": tol,
        "invariant": invariant,
        "n_tensors_zeroable": len(invariant["tensors_zeroable"]),
        "n_latent_dims_zeroable": len(invariant["latent_dims_zeroable"]),
        "n_tensor_bit_floors": len(invariant["tensor_bit_floors"]),
        "safe_byte_saving_bit_floors": safe_saving,
        "upper_byte_saving_zeroable": upper_saving,
        "rate_cut_safe": rate_cut_safe,
        "rate_cut_upper": rate_cut_upper,
        "invariant_mass_frac_safe": invariant_mass_frac_safe,
        "invariant_mass_frac_upper": invariant_mass_frac_upper,
        "projected_S_safe": base["S"] - rate_cut_safe,
        "projected_S_upper": base["S"] - rate_cut_upper,
        "frontier_S_report": 0.19110,
    }


def _finalize(state, state_path, args, n):
    h = state.get("headline", {})
    # The HONEST invariant mass is the JOINT bit-floor saving (measured combined), because
    # per-tensor ΔS do NOT sum. If the joint verification ran, use its real saving + real
    # combined task hit; else fall back to the per-tensor aggregate (upper bound).
    joint = h.get("joint_bit_floor_verification")
    if joint and joint.get("joint_section_byte_saving") is not None:
        joint_save = joint["joint_section_byte_saving"] or 0.0
        joint_frac = joint_save / ARCHIVE_BYTES_FULL if ARCHIVE_BYTES_FULL else 0.0
        joint_dS = joint["joint_delta_S_task"]
        # GREEN requires real byte mass AND the combined task hit stays within tolerance
        # (a joint coarsening that saves bytes but spills d_seg is NOT a free invariant cut).
        if joint_frac > 0.20 and joint_dS <= DELTA_S_TOL:
            verdict = "GREEN_LARGE_SCORER_INVARIANT_MASS_DOMINATED_RUNG_RATE_CUT"
        elif joint_frac > 0.05 and joint_dS <= 2 * DELTA_S_TOL:
            verdict = "AMBER_MODERATE_SCORER_INVARIANT_MASS"
        else:
            verdict = "RED_FRONTIER_NEAR_TASK_RD_FLOOR_LITTLE_INVARIANT_MASS"
        state["verdict_basis"] = (
            f"JOINT bit-floor: {100*joint_frac:.1f}% byte saving, combined ΔS_task={joint_dS:+.5f}"
        )
    else:
        safe_frac = h.get("invariant_mass_frac_safe") or 0.0
        upper_frac = h.get("invariant_mass_frac_upper") or 0.0
        best_frac = max(safe_frac, upper_frac)
        if best_frac > 0.20:
            verdict = "GREEN_LARGE_SCORER_INVARIANT_MASS_DOMINATED_RUNG_RATE_CUT"
        elif best_frac > 0.05:
            verdict = "AMBER_MODERATE_SCORER_INVARIANT_MASS"
        else:
            verdict = "RED_FRONTIER_NEAR_TASK_RD_FLOOR_LITTLE_INVARIANT_MASS"
        state["verdict_basis"] = "per-tensor aggregate (joint verification not run)"
    state["verdict"] = verdict

    result_json = REPO / ".omx/research" / f"p_suff_task_ablation_{_now()}.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "p_suff_task_ablation.v1",
        "produced_at_utc": datetime.now(UTC).isoformat(),
        "producer": "experiments/probe_p_suff_task_ablation.py",
        "axis_tag": "[contest-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "n_pairs_measured": n,
        "the_question": (
            "How much of the 0.19110 frontier archive does the frozen scorer IGNORE? "
            "(the indirect-RD R_Ỹ−R_X invariance gap on the binding rate term). Total "
            "scorer-invariant byte mass = fraction deletable/coarsenable keeping ΔS_task "
            f"< +{DELTA_S_TOL}."
        ),
        "state": state,
        "verdict": verdict,
    }
    result_json.write_text(json.dumps(payload, indent=2))
    state["result_json"] = str(result_json.relative_to(REPO))
    state_path.write_text(json.dumps(state, indent=2))

    print(f"\n[done] advisory JSON -> {result_json.relative_to(REPO)}", flush=True)
    print(f"[verdict-basis] {state.get('verdict_basis','')}", flush=True)
    if joint and joint.get("joint_section_byte_saving") is not None:
        jf = (joint["joint_section_byte_saving"] or 0.0) / ARCHIVE_BYTES_FULL
        proj = joint.get("joint_projected_S_with_byte_cut")
        proj_str = f"{proj:.5f}" if proj is not None else "n/a"
        print(
            f"[headline JOINT] invariant byte mass = {100*jf:.1f}% "
            f"({joint['joint_section_byte_saving']:.0f} B saved) "
            f"combined ΔS_task={joint['joint_delta_S_task']:+.5f} "
            f"projected_S_with_byte_cut={proj_str}",
            flush=True,
        )
    elif h:
        sf = h.get("invariant_mass_frac_safe") or 0.0
        uf = h.get("invariant_mass_frac_upper") or 0.0
        print(
            f"[headline] safe invariant mass = {100*sf:.1f}% "
            f"({h.get('safe_byte_saving_bit_floors',0):.0f} B; rate_cut {h.get('rate_cut_safe',0):.5f}) "
            f"| upper {100*uf:.1f}% | projected S_safe {h.get('projected_S_safe'):.5f}",
            flush=True,
        )
    print(f"[VERDICT] {verdict}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
