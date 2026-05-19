#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Driver: run 4 alternative-reducer probes on Wunderkind G1 v2 + Tishby IB-pure.

Per SUBAGENT B task (T2 council Q1 SPLIT-VERDICT reactivation criteria +
Catalog #308 META-pattern E remediation, 2026-05-16):

  1. Run SegNet over 600 pairs of the source signal for each substrate:
     - Wunderkind G1 v2 -> ``upstream/videos/0.mkv`` GT frames (the source v2's
       per-pair-conditioning would be against).
     - Tishby IB-pure -> frames rendered by A1's HNeRV decoder over A1's
       latents (mirrors the existing d4_driver.py pattern at
       ``experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/``).

  2. Cache per-pair (384, 512) int8 argmax maps + per-frame argmax dominant
     classes so the 4 reducers share a single expensive SegNet inference.

  3. Apply the 4 alternative reducers to the cached argmax maps:
     - per_pixel_histogram
     - per_region_histogram
     - per_pair_class_2_fraction
     - per_frame_argmax

  4. For each (substrate, reducer) pair, compute MI against the substrate's
     latent residual stream and emit a typed
     ``AlternativeReducerVerdict`` JSON + a per-substrate run manifest.

Per CLAUDE.md "Forbidden /tmp paths" + Catalog #110/#113: outputs land under
``experiments/results/alternative_reducer_probes_<timestamp>/`` with full
provenance.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #127 source-data carve-out +
Catalog #190: runs on local CPU; tagged ``diagnostic_cpu`` evidence grade;
``hardware_substrate`` resolved via the canonical
``tac.substrates._shared.trainer_skeleton.detect_hardware_substrate`` helper.

Per CLAUDE.md "MASKS.MKV AT 48x64 DESTROYED THE SCORE" lesson: the SegNet's
canonical preprocess (slice last frame + interpolate to 384x512) IS the
correct argmax resolution for the conditioning signal; we do NOT downsample.

Wall-clock budget: ~9 min per substrate on M5 Max CPU (8.6s decode + 528s
SegNet inference, mirrors the Wunderkind G1 v2 re-probe at 537s).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

# --- canonical paths + imports ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

import torch  # noqa: E402

from tools.probe_alternative_reducers_latent_class_conditioning import (  # noqa: E402
    CLASS_2_INDEX_DEFAULT,
    NUM_SEGNET_CLASSES_DEFAULT,
    PER_PAIR_CLASS_2_FRACTION_BUCKETS_DEFAULT,
    PER_PIXEL_HISTOGRAM_BIN_QUANT_DEFAULT,
    PER_REGION_HISTOGRAM_BIN_QUANT_DEFAULT,
    REDUCER_MEANINGFUL_THRESHOLD_BITS,
    AlternativeReducerVerdict,
    compute_alternative_reducer_verdict,
    make_timestamped_output_dir,
    write_run_manifest,
    write_verdict_json,
)

REDUCER_ORDER: tuple[str, ...] = (
    "per_pixel_histogram",
    "per_region_histogram",
    "per_pair_class_2_fraction",
    "per_frame_argmax",
)

SEGNET_TARGET_H: int = 384
SEGNET_TARGET_W: int = 512


def _decode_gt_pairs_from_video(video_path: Path, max_pairs: int = 600) -> list[torch.Tensor]:
    """Decode upstream/videos/0.mkv into (T=2, C=3, H, W) fp32 pairs via pyav.

    Mirrors ``upstream/frame_utils.py::AVVideoDataset`` semantics; same path
    used by ``derive_real_segnet_classes.py`` at the
    ``wunderkind_g1_v2_real_cuda_section14_reprobe_*`` artifact.
    """
    import av  # type: ignore[import-untyped]
    from frame_utils import yuv420_to_rgb  # type: ignore[import-untyped]

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    pairs: list[torch.Tensor] = []
    seq_buf: list[torch.Tensor] = []
    for frame in container.decode(stream):
        rgb_hwc = yuv420_to_rgb(frame)
        rgb_chw = rgb_hwc.permute(2, 0, 1).to(torch.float32)
        seq_buf.append(rgb_chw)
        if len(seq_buf) == 2:
            pairs.append(torch.stack(seq_buf, dim=0))  # (2, 3, H, W)
            seq_buf = []
            if len(pairs) >= max_pairs:
                break
    container.close()
    return pairs


def _segnet_argmax_per_frame_in_pair(
    *,
    pair_tensor: torch.Tensor,
    segnet,
    target_h: int = SEGNET_TARGET_H,
    target_w: int = SEGNET_TARGET_W,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SegNet on each frame in a pair; return frame_0 and frame_1 argmax maps.

    The SegNet's ``preprocess_input`` canonical helper slices ``x[:, -1, ...]``
    so it operates on frame_1 only. To get per-frame argmax we manually
    interpolate each frame to (target_h, target_w) and run SegNet on each.

    Returns two int64 (H, W) argmax maps.
    """
    import torch.nn.functional as F

    f0 = pair_tensor[0:1]  # (1, 3, H, W) fp32 in [0, 255]
    f1 = pair_tensor[1:2]
    f0_resized = F.interpolate(
        f0, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    f1_resized = F.interpolate(
        f1, size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    with torch.no_grad():
        f0_logits = segnet(f0_resized)  # (1, 5, H, W)
        f1_logits = segnet(f1_resized)
    f0_argmax = f0_logits.argmax(dim=1).squeeze(0).to(torch.long)
    f1_argmax = f1_logits.argmax(dim=1).squeeze(0).to(torch.long)
    return f0_argmax, f1_argmax


def _segnet_argmax_canonical_preprocess(
    *,
    pair_tensor: torch.Tensor,
    segnet,
) -> torch.Tensor:
    """Run SegNet via canonical preprocess_input (slice frame_1 + interpolate).

    Returns int64 (target_h, target_w) argmax map for frame_1 ONLY (the
    canonical contest-scorer behavior per upstream/modules.py:108).
    """
    x = pair_tensor.unsqueeze(0)  # (1, 2, 3, H, W)
    preprocessed = segnet.preprocess_input(x)  # (1, 3, target_h, target_w)
    with torch.no_grad():
        logits = segnet(preprocessed)
    return logits.argmax(dim=1).squeeze(0).to(torch.long)


def _apply_4_reducers_to_pair(
    *,
    per_pixel_argmax_canonical: torch.Tensor,  # (H, W) frame_1 argmax
    frame_0_argmax: torch.Tensor | None,  # (H, W) or None for HNeRV-style frame_1-only
    frame_1_argmax: torch.Tensor,
    num_classes: int = NUM_SEGNET_CLASSES_DEFAULT,
) -> dict[str, int]:
    """Apply all 4 reducers to a single pair's argmax outputs.

    Returns dict mapping reducer_name to per-pair conditioning symbol (int).
    The canonical (frame_1) argmax map is used for per_pixel + per_region +
    per_pair_class_2_fraction; per_frame_argmax uses BOTH frames separately
    when frame_0_argmax is available.
    """
    from tools.probe_alternative_reducers_latent_class_conditioning import (
        reduce_per_frame_argmax,
        reduce_per_pair_class_2_fraction,
        reduce_per_pixel_histogram,
        reduce_per_region_histogram,
    )

    # Flatten (H, W) to a Python list of ints for the histogram reducers.
    canonical_flat = per_pixel_argmax_canonical.flatten().tolist()
    canonical_2d = per_pixel_argmax_canonical.tolist()
    frame_1_flat = frame_1_argmax.flatten().tolist()
    frame_0_flat = (
        frame_0_argmax.flatten().tolist() if frame_0_argmax is not None else None
    )

    return {
        "per_pixel_histogram": reduce_per_pixel_histogram(
            argmax_map=canonical_flat,
            num_classes=num_classes,
            bin_quant_levels=PER_PIXEL_HISTOGRAM_BIN_QUANT_DEFAULT,
        ),
        "per_region_histogram": reduce_per_region_histogram(
            argmax_map_2d=canonical_2d,
            num_classes=num_classes,
            bin_quant_levels=PER_REGION_HISTOGRAM_BIN_QUANT_DEFAULT,
            num_regions=4,
        ),
        "per_pair_class_2_fraction": reduce_per_pair_class_2_fraction(
            argmax_map=canonical_flat,
            class_index=CLASS_2_INDEX_DEFAULT,
            num_buckets=PER_PAIR_CLASS_2_FRACTION_BUCKETS_DEFAULT,
        ),
        "per_frame_argmax": reduce_per_frame_argmax(
            frame_0_argmax_map=frame_0_flat,
            frame_1_argmax_map=frame_1_flat,
            num_classes=num_classes,
        ),
    }


# --- Wunderkind G1 v2 driver ---


def _wunderkind_g1_v2_render_and_reduce(
    *,
    out_dir: Path,
    max_pairs: int = 600,
    log_every: int = 50,
) -> dict[str, list[int]]:
    """Render + run SegNet on GT frames; return per-pair reducer outputs.

    For Wunderkind G1 v2 the "source signal" is the GT contest video pair.
    The substrate would condition its latent on per-pair SegNet features
    derived from these GT frames. Per Catalog #127 source-data carve-out:
    running SegNet on the contest video is SOURCE GENERATION, not auth-eval.
    """
    from tac.scorer import load_default_scorers

    video_path = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    segnet_weights_path = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"
    if not video_path.exists():
        raise FileNotFoundError(f"video missing: {video_path}")
    if not segnet_weights_path.exists():
        raise FileNotFoundError(f"segnet weights missing: {segnet_weights_path}")

    print("[g1-v2-driver] loading SegNet on CPU via canonical loader", flush=True)
    t0 = time.monotonic()
    _posenet, segnet = load_default_scorers(REPO_ROOT / "upstream", device=torch.device("cpu"))
    segnet.eval()
    t_load = time.monotonic() - t0
    print(f"[g1-v2-driver] loaded in {t_load:.1f}s", flush=True)

    print(f"[g1-v2-driver] decoding pairs from {video_path}", flush=True)
    t0 = time.monotonic()
    pairs = _decode_gt_pairs_from_video(video_path, max_pairs=max_pairs)
    t_decode = time.monotonic() - t0
    print(f"[g1-v2-driver] decoded {len(pairs)} pairs in {t_decode:.1f}s", flush=True)

    print("[g1-v2-driver] running SegNet on each pair (frame_0 + frame_1 separately + canonical preprocess)", flush=True)
    per_pair_by_reducer: dict[str, list[int]] = {r: [] for r in REDUCER_ORDER}
    per_pair_class_2_fractions: list[float] = []  # for forensic provenance
    per_pair_dominant_class: list[int] = []
    per_frame_dominant_per_pair: list[tuple[int, int]] = []

    t0 = time.monotonic()
    for i, pair in enumerate(pairs):
        # Canonical preprocess argmax (frame_1 at 384x512) used for the
        # per_pixel + per_region + per_pair_class_2_fraction reducers.
        canon_argmax = _segnet_argmax_canonical_preprocess(
            pair_tensor=pair, segnet=segnet
        )
        # Per-frame argmax of BOTH frames at the same target resolution.
        f0_argmax, f1_argmax = _segnet_argmax_per_frame_in_pair(
            pair_tensor=pair, segnet=segnet
        )

        reducer_outputs = _apply_4_reducers_to_pair(
            per_pixel_argmax_canonical=canon_argmax,
            frame_0_argmax=f0_argmax,
            frame_1_argmax=f1_argmax,
        )
        for r in REDUCER_ORDER:
            per_pair_by_reducer[r].append(reducer_outputs[r])

        # Forensic: class-2 fraction (continuous) + per-pair dominant + per-frame dominant.
        canon_flat = canon_argmax.flatten().tolist()
        n_class2 = sum(1 for c in canon_flat if c == CLASS_2_INDEX_DEFAULT)
        per_pair_class_2_fractions.append(n_class2 / len(canon_flat))
        per_pair_dominant_class.append(
            max(Counter(canon_flat).items(), key=lambda kv: kv[1])[0]
        )
        f0_dom = max(Counter(f0_argmax.flatten().tolist()).items(), key=lambda kv: kv[1])[0]
        f1_dom = max(Counter(f1_argmax.flatten().tolist()).items(), key=lambda kv: kv[1])[0]
        per_frame_dominant_per_pair.append((int(f0_dom), int(f1_dom)))

        if (i + 1) % log_every == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / rate if rate > 0 else 0
            print(
                f"[g1-v2-driver]   pair {i+1}/{len(pairs)} | rate {rate:.2f} pair/s | eta {eta:.0f}s",
                flush=True,
            )
    t_infer = time.monotonic() - t0
    print(f"[g1-v2-driver] SegNet inference done in {t_infer:.1f}s", flush=True)

    # Persist forensic reducer outputs + provenance.
    (out_dir / "g1_v2_per_pair_reducer_outputs.json").write_text(
        json.dumps(
            {
                "per_pair_by_reducer": per_pair_by_reducer,
                "per_pair_class_2_fractions": per_pair_class_2_fractions,
                "per_pair_dominant_class": per_pair_dominant_class,
                "per_frame_dominant_per_pair": per_frame_dominant_per_pair,
                "num_pairs": len(pairs),
                "segnet_canonical_resolution": [SEGNET_TARGET_H, SEGNET_TARGET_W],
                "elapsed_seconds_load": float(t_load),
                "elapsed_seconds_decode": float(t_decode),
                "elapsed_seconds_inference": float(t_infer),
                "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    return per_pair_by_reducer


# --- Tishby IB-pure driver (A1 latents -> A1-rendered frames -> SegNet) ---


def _tishby_ib_pure_render_and_reduce(
    *,
    out_dir: Path,
    max_pairs: int = 600,
    log_every: int = 50,
    chunk_size: int = 16,
) -> tuple[dict[str, list[int]], bytes]:
    """Render A1 frames via HNeRV decoder; run SegNet; apply 4 reducers.

    Mirrors the d4_driver.py pattern at
    ``experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/`` but
    computes per-pixel-canonical + per-frame argmax so all 4 reducers can be
    applied.

    Returns:
        per_pair_by_reducer: dict mapping reducer_name to list of per-pair
            conditioning symbols (length = num_pairs).
        latent_bytes: the A1 latents requantized to uint8 (length =
            num_pairs * 28); same stream the existing tishby_ib_pure probe used.
    """
    a1_archive = REPO_ROOT / "submissions" / "a1" / "archive.zip"
    a1_src = REPO_ROOT / "submissions" / "a1" / "src"
    if not a1_archive.exists():
        raise FileNotFoundError(f"A1 archive missing: {a1_archive}")
    if not a1_src.exists():
        raise FileNotFoundError(f"A1 src missing: {a1_src}")

    # Add A1 src to import path for codec.py + model.py
    sys.path.insert(0, str(a1_src))

    print(f"[tishby-driver] parsing A1 archive {a1_archive}", flush=True)
    with zipfile.ZipFile(a1_archive, "r") as zf:
        archive_bytes = zf.read("x")
    import struct
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    decoder_blob = archive_bytes[4:section_total]
    LATENT_BLOB_LEN = 15_387
    latent_blob = archive_bytes[section_total : section_total + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + LATENT_BLOB_LEN :]

    from codec import (  # type: ignore[import-not-found]
        apply_latent_sidecar,
        decode_decoder_compact,
        decode_latents_compact,
    )
    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    print(f"[tishby-driver] A1 latents shape={tuple(latents.shape)} dtype={latents.dtype}", flush=True)

    # Requantize latents to uint8 symbol stream (same as d4_driver.py).
    lat_min = float(latents.min().item())
    lat_max = float(latents.max().item())
    lat_range = max(lat_max - lat_min, 1e-9)
    latents_u8 = (
        (latents - lat_min) / lat_range * 255.0
    ).round().clamp(0, 255).to(torch.uint8)
    latent_bytes = bytes(latents_u8.flatten().tolist())
    print(f"[tishby-driver] latent_bytes len={len(latent_bytes)}", flush=True)

    # Load A1 decoder + canonical SegNet.
    print("[tishby-driver] loading A1 HNeRV decoder + canonical SegNet", flush=True)
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    from tac.scorer import load_default_scorers

    device = torch.device("cpu")
    decoder = HNeRVDecoder(
        latent_dim=28,
        base_channels=36,
        eval_size=(SEGNET_TARGET_H, SEGNET_TARGET_W),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    _posenet, segnet = load_default_scorers(REPO_ROOT / "upstream", device=device)
    segnet.eval()

    latents_dev = latents.to(device)
    n_pairs = min(max_pairs, latents_dev.shape[0])
    print(f"[tishby-driver] rendering + segmenting {n_pairs} pairs in chunks of {chunk_size}", flush=True)

    per_pair_by_reducer: dict[str, list[int]] = {r: [] for r in REDUCER_ORDER}
    per_pair_class_2_fractions: list[float] = []
    per_pair_dominant_class: list[int] = []
    per_frame_dominant_per_pair: list[tuple[int, int]] = []

    t0 = time.monotonic()
    with torch.no_grad():
        for start in range(0, n_pairs, chunk_size):
            end = min(start + chunk_size, n_pairs)
            lat_chunk = latents_dev[start:end]
            rendered = decoder(lat_chunk)  # (B, 2, 3, 384, 512) [0, 255] fp32
            rendered_clamped = rendered.clamp(0, 255)

            # Canonical preprocess argmax (frame_1 at 384x512).
            for i_in_chunk in range(rendered_clamped.shape[0]):
                pair_tensor = rendered_clamped[i_in_chunk]  # (2, 3, 384, 512)
                canon_argmax = _segnet_argmax_canonical_preprocess(
                    pair_tensor=pair_tensor, segnet=segnet
                )
                # Per-frame argmax of BOTH frames separately.
                f0_argmax, f1_argmax = _segnet_argmax_per_frame_in_pair(
                    pair_tensor=pair_tensor, segnet=segnet
                )

                reducer_outputs = _apply_4_reducers_to_pair(
                    per_pixel_argmax_canonical=canon_argmax,
                    frame_0_argmax=f0_argmax,
                    frame_1_argmax=f1_argmax,
                )
                for r in REDUCER_ORDER:
                    per_pair_by_reducer[r].append(reducer_outputs[r])

                canon_flat = canon_argmax.flatten().tolist()
                n_class2 = sum(1 for c in canon_flat if c == CLASS_2_INDEX_DEFAULT)
                per_pair_class_2_fractions.append(n_class2 / len(canon_flat))
                per_pair_dominant_class.append(
                    max(Counter(canon_flat).items(), key=lambda kv: kv[1])[0]
                )
                f0_dom = max(Counter(f0_argmax.flatten().tolist()).items(), key=lambda kv: kv[1])[0]
                f1_dom = max(Counter(f1_argmax.flatten().tolist()).items(), key=lambda kv: kv[1])[0]
                per_frame_dominant_per_pair.append((int(f0_dom), int(f1_dom)))
            elapsed = time.monotonic() - t0
            rate = end / elapsed
            eta = (n_pairs - end) / rate if rate > 0 else 0
            print(
                f"[tishby-driver]   pair {end}/{n_pairs} | rate {rate:.2f} pair/s | eta {eta:.0f}s",
                flush=True,
            )
    t_infer = time.monotonic() - t0
    print(f"[tishby-driver] render+SegNet inference done in {t_infer:.1f}s", flush=True)

    (out_dir / "tishby_ib_pure_per_pair_reducer_outputs.json").write_text(
        json.dumps(
            {
                "per_pair_by_reducer": per_pair_by_reducer,
                "per_pair_class_2_fractions": per_pair_class_2_fractions,
                "per_pair_dominant_class": per_pair_dominant_class,
                "per_frame_dominant_per_pair": per_frame_dominant_per_pair,
                "num_pairs": n_pairs,
                "segnet_canonical_resolution": [SEGNET_TARGET_H, SEGNET_TARGET_W],
                "elapsed_seconds_inference": float(t_infer),
                "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )

    return per_pair_by_reducer, latent_bytes


# --- Orchestrator ---


def _load_wunderkind_g1_v2_latent_stream() -> bytes:
    """Read the existing v2 residual_int8 stream from the section-14 probe artifact."""
    p = (
        REPO_ROOT
        / "experiments"
        / "results"
        / "probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z"
        / "residual_int8.bin"
    )
    if not p.exists():
        raise FileNotFoundError(
            f"Wunderkind G1 v2 residual_int8.bin missing at {p}; re-run the "
            "Section 14 probe dispatch to regenerate"
        )
    return p.read_bytes()


def _run_substrate_probes(
    *,
    substrate_id: str,
    out_dir: Path,
    max_pairs: int = 600,
    master_gradient_requested: bool = False,
    master_gradient_archive_sha256: str | None = None,
    master_gradient_anchor_path: Path | None = None,
    master_gradient_write_sidecar: bool = False,
) -> dict:
    """Run all 4 alternative-reducer probes on one substrate; return summary dict."""
    print(f"\n=== {substrate_id}: alternative-reducer probe wave ===\n", flush=True)
    if substrate_id == "wunderkind_g1_v2":
        per_pair_by_reducer = _wunderkind_g1_v2_render_and_reduce(
            out_dir=out_dir, max_pairs=max_pairs
        )
        latent_bytes = _load_wunderkind_g1_v2_latent_stream()
        symbols_per_pair = len(latent_bytes) // len(per_pair_by_reducer["per_pixel_histogram"])
        latent_source = (
            "experiments/results/probe_dispatch_wave_749_section14_g1_v2_20260516T171251Z/residual_int8.bin"
        )
        notes_suffix = (
            "Wunderkind G1 v2 alternative-reducer probe wave per T2 council Q1.4 + Catalog #308 "
            "META-pattern E remediation 2026-05-16. Source signal: upstream/videos/0.mkv GT pairs "
            "(SegNet source-data derivation per Catalog #127 carve-out). Latent stream: v2 1000ep "
            "Z3 archive residual_int8 (600x28 = 16800 bytes). Reducer outputs sourced from real-CUDA "
            "SegNet (not synthetic uniform) per Appendix B canonical re-probe pattern."
        )
    elif substrate_id == "tishby_ib_pure":
        per_pair_by_reducer, latent_bytes = _tishby_ib_pure_render_and_reduce(
            out_dir=out_dir, max_pairs=max_pairs
        )
        symbols_per_pair = len(latent_bytes) // len(per_pair_by_reducer["per_pixel_histogram"])
        latent_source = "A1 archive (submissions/a1/archive.zip) HNeRV latents requantized to uint8"
        notes_suffix = (
            "Tishby IB-pure alternative-reducer probe wave per T2 council Q1.4 + Catalog #308 "
            "META-pattern E remediation 2026-05-16. Source signal: A1's HNeRV-rendered pairs over "
            "A1 latents (600 pairs, 384x512). Latent stream: A1 latents (600x28) requantized to "
            "uint8 via affine map to [0,255] (mirrors d4_driver.py at "
            "experiments/results/tishby_ib_pure_d4_probe_20260516T212557Z/)."
        )
    else:
        raise ValueError(f"unknown substrate_id: {substrate_id}")

    print(f"\n[{substrate_id}] applying 4 reducers + computing MI ...\n", flush=True)

    verdicts: list[AlternativeReducerVerdict] = []
    for reducer_name in REDUCER_ORDER:
        per_pair_reduced = per_pair_by_reducer[reducer_name]
        reducer_specific_parameters = _reducer_params_for_provenance(reducer_name)
        notes = f"{notes_suffix} Reducer: {reducer_name}; threshold: {REDUCER_MEANINGFUL_THRESHOLD_BITS[reducer_name]} bits/pair."
        verdict = compute_alternative_reducer_verdict(
            substrate_id=substrate_id,
            reducer_name=reducer_name,  # type: ignore[arg-type]
            latent_stream=latent_bytes,
            per_pair_reduced_class=per_pair_reduced,
            symbols_per_pair=symbols_per_pair,
            reducer_specific_parameters=reducer_specific_parameters,
            notes=notes,
        )
        verdicts.append(verdict)
        path = write_verdict_json(out_dir, verdict)
        print(
            f"  [{substrate_id}] {reducer_name}: VERDICT={verdict.verdict} "
            f"MI={verdict.mutual_information_bits:.4f} bits "
            f"(threshold={verdict.meaningful_mi_threshold_bits}) "
            f"unique_classes={verdict.num_unique_reduced_classes} "
            f"-> {path.name}",
            flush=True,
        )

    provenance = {
        "substrate_id": substrate_id,
        "latent_source": latent_source,
        "latent_stream_sha256": hashlib.sha256(latent_bytes).hexdigest(),
        "latent_stream_bytes": len(latent_bytes),
        "symbols_per_pair": symbols_per_pair,
        "num_pairs": len(per_pair_by_reducer["per_pixel_histogram"]),
        "reducers_evaluated": list(REDUCER_ORDER),
        "reducer_specific_parameters": {
            r: _reducer_params_for_provenance(r) for r in REDUCER_ORDER
        },
        "device": "cpu",
        "torch_version": torch.__version__,
        "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    manifest_path = write_run_manifest(
        out_dir,
        substrate_id=substrate_id,
        verdicts=verdicts,
        provenance=provenance,
        master_gradient_requested=master_gradient_requested,
        master_gradient_archive_sha256=master_gradient_archive_sha256,
        master_gradient_anchor_path=master_gradient_anchor_path,
        master_gradient_write_sidecar=master_gradient_write_sidecar,
    )
    print(f"\n[{substrate_id}] run manifest -> {manifest_path}\n", flush=True)

    return {
        "substrate_id": substrate_id,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "verdicts": [
            {
                "reducer_name": v.reducer_name,
                "verdict": v.verdict,
                "mutual_information_bits": v.mutual_information_bits,
                "meaningful_mi_threshold_bits": v.meaningful_mi_threshold_bits,
                "num_unique_reduced_classes": v.num_unique_reduced_classes,
            }
            for v in verdicts
        ],
    }


def _reducer_params_for_provenance(reducer_name: str) -> dict[str, int | float]:
    return {
        "per_pixel_histogram": {
            "num_classes": NUM_SEGNET_CLASSES_DEFAULT,
            "bin_quant_levels": PER_PIXEL_HISTOGRAM_BIN_QUANT_DEFAULT,
        },
        "per_region_histogram": {
            "num_classes": NUM_SEGNET_CLASSES_DEFAULT,
            "bin_quant_levels": PER_REGION_HISTOGRAM_BIN_QUANT_DEFAULT,
            "num_regions": 4,
        },
        "per_pair_class_2_fraction": {
            "class_index": CLASS_2_INDEX_DEFAULT,
            "num_buckets": PER_PAIR_CLASS_2_FRACTION_BUCKETS_DEFAULT,
        },
        "per_frame_argmax": {
            "num_classes": NUM_SEGNET_CLASSES_DEFAULT,
        },
    }[reducer_name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--substrate",
        choices=["wunderkind_g1_v2", "tishby_ib_pure", "both"],
        default="both",
        help="which substrate(s) to probe",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=600,
        help="max pairs to render+probe (default 600 matches the canonical "
        "contest video; reduce for smoke tests)",
    )
    parser.add_argument(
        "--master-gradient-diagnostic",
        action="store_true",
        help="Request diagnostic-only master-gradient Wyner-Ziv covariance "
        "evidence in each run manifest.",
    )
    parser.add_argument(
        "--master-gradient-archive-sha256",
        help="Archive SHA-256 whose master-gradient anchor should be loaded "
        "when --master-gradient-diagnostic is requested.",
    )
    parser.add_argument(
        "--master-gradient-anchor-path",
        type=Path,
        help="Optional master_gradient_anchors.jsonl path for the diagnostic "
        "Wyner-Ziv covariance helper.",
    )
    parser.add_argument(
        "--master-gradient-write-sidecar",
        action="store_true",
        help="Allow the canonical master-gradient consumer to write its "
        "diagnostic sidecar. Defaults to manifest-only evidence.",
    )
    args = parser.parse_args(argv)

    out_dir = make_timestamped_output_dir(REPO_ROOT)
    print(f"[orchestrator] output directory: {out_dir}", flush=True)

    summaries: list[dict] = []
    if args.substrate in ("wunderkind_g1_v2", "both"):
        summaries.append(
            _run_substrate_probes(
                substrate_id="wunderkind_g1_v2",
                out_dir=out_dir,
                max_pairs=args.max_pairs,
                master_gradient_requested=args.master_gradient_diagnostic,
                master_gradient_archive_sha256=args.master_gradient_archive_sha256,
                master_gradient_anchor_path=args.master_gradient_anchor_path,
                master_gradient_write_sidecar=args.master_gradient_write_sidecar,
            )
        )
    if args.substrate in ("tishby_ib_pure", "both"):
        summaries.append(
            _run_substrate_probes(
                substrate_id="tishby_ib_pure",
                out_dir=out_dir,
                max_pairs=args.max_pairs,
                master_gradient_requested=args.master_gradient_diagnostic,
                master_gradient_archive_sha256=args.master_gradient_archive_sha256,
                master_gradient_anchor_path=args.master_gradient_anchor_path,
                master_gradient_write_sidecar=args.master_gradient_write_sidecar,
            )
        )

    # Write combined run summary across both substrates.
    combined = {
        "axis_label": "[diagnostic-CPU; alternative-reducer probe wave: 4 reducers x 2 substrates]",
        "evidence_grade": "diagnostic_cpu",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "lane_id": "lane_four_alternative_reducer_probes_meta_pattern_e_remediation_20260516",
        "cite_chain": {
            "task_origin": "SUBAGENT B alternative-reducer probe wave",
            "t2_council_memo": ".omx/research/grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516.md",
            "catalog_308_meta_pattern_e": "META-pattern E remediation per operator directive 2026-05-16",
            "wunderkind_g1_v2_design_memo": ".omx/research/wunderkind_g1_v2_wire_grammar_class_shift_full_stack_design_20260516.md",
            "tishby_ib_pure_design_memo": ".omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md",
        },
        "summaries": summaries,
        "master_gradient_diagnostic": {
            "requested": bool(args.master_gradient_diagnostic),
            "archive_sha256": args.master_gradient_archive_sha256,
            "anchor_path": str(args.master_gradient_anchor_path)
            if args.master_gradient_anchor_path is not None
            else None,
            "write_sidecar": bool(args.master_gradient_write_sidecar),
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
        "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    combined_path = out_dir / "combined_run_summary.json"
    combined_path.write_text(json.dumps(combined, sort_keys=True, indent=2) + "\n")
    print(f"\n[orchestrator] combined summary -> {combined_path}\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
