#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""resize_null_preimage_postprocess — the universal preimage CLI (task #49).

The contest scores the PROJECTION ``y = R x`` (the shared bilinear resize, #47),
never the camera frame ``x``.  This CLI applies the resize-null preimage compiler
(``tac.optimization.resize_null_preimage``) to any vehicle's rendered frames:
it replaces the certified scorer-invisible degrees of freedom with the
entropy-optimal (measured-best) fill BEFORE any codec touches the bytes, proving
``R x̃ = R x`` exactly per frame and emitting a V3 row per application.

Frames in -> frames out (preimage, valid uint8, same shape) + a proof JSON +
optional V3-row JSONL.  Inputs:

  --frames-source video:<path>     decode N frames from a contest .mkv (PyAV;
                                   camera_size NHWC, the evaluate.py layout)
  --frames-source raw:<path>       a .raw inflate output (N, 874, 1164, 3) uint8
                                   (e.g. the SNeRV G1b frontier_inflate/0.raw)
  --frames-source npy:<path>       an (N, H, W, 3) uint8 .npy

Headline mode (``--headline``) runs tier-1 (and tier-2 if ``--tier 2``) over N
frames from BOTH a source-video set AND a vehicle-render set and reports the
coded-bytes reduction (brotli + lzma) at PROVEN zero scorer change — the
falsifiable measurement from the directive.

Authority: the exactness proof is ``mathematical-derivation`` (residual == 0.0).
The bytes-reduction numbers are ``[macOS-CPU advisory]`` (local CPU coder).  No
score claim; no promotion; ``promotable=false``.

CLAUDE.md compliance: no MPS; no /tmp persisted artifacts (durable SSD waterfall
for outputs); every application emits a V3 row (operator caveat (d)).

Examples
--------
    # Headline: tier-1 on 16 source frames + 16 SNeRV G1b render frames.
    PYTHONPATH=src:upstream .venv/bin/python tools/resize_null_preimage_postprocess.py \
        --headline --n-frames 16 \
        --source-video upstream/videos/0.mkv \
        --vehicle-raw /Volumes/VertigoDataTier/pact/snerv_branch_b_round2_*/frontier_inflate/0.raw \
        --out-dir /Volumes/VertigoDataTier/pact/resize_null_preimage_headline_<UTC>

    # Postprocess a single render to a preimage .raw + proof + V3 rows.
    PYTHONPATH=src:upstream .venv/bin/python tools/resize_null_preimage_postprocess.py \
        --frames-source raw:render.raw --n-frames 1200 --tier 1 \
        --out-frames preimage.raw --proof-json proof.json --v3-jsonl rows.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "upstream") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "upstream"))

from tac.optimization.evaluator_invisibility_basis import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    derive_tier1_resize_null_space,
)
from tac.optimization.resize_null_preimage import (  # noqa: E402
    FrameProof,
    ResizeProjector,
    apply_tier1_zero_weight_fill,
    apply_tier2_null_basis_descent,
    apply_tier3_blockwise_flat_preimage,
    preimage_rate_score_delta,
    zero_weight_pixel_mask,
)

# Durable SSD waterfall (CLAUDE.md disk hygiene — never /tmp for evidence).
SSD_WATERFALL = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

V3_FALSE_AUTHORITY = {
    "authority": "[macOS-CPU advisory]",
    "authority_tier": "exact_cpu_advisory",
    "metric_family": "resize_null_preimage_certified",
    "score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
}


# ---------------------------------------------------------------------------
# Frame loaders.
# ---------------------------------------------------------------------------
def _load_raw_frames(path: Path, n_frames: int, *, camera_h: int = CAMERA_H,
                     camera_w: int = CAMERA_W) -> np.ndarray:
    """Load up to ``n_frames`` from an inflate ``.raw`` (N, H, W, 3) uint8 via
    memmap (the evaluate.py TensorVideoDataset layout)."""
    frame_bytes = camera_h * camera_w * 3
    file_size = path.stat().st_size
    total = file_size // frame_bytes
    if total == 0:
        raise SystemExit(f"raw file {path} too small for one {camera_h}x{camera_w}x3 frame")
    take = min(n_frames, total)
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(total, camera_h, camera_w, 3))
    # copy out the slice (we mutate it); leave the memmap read-only.
    return np.array(mm[:take], dtype=np.uint8)


def _load_video_frames(path: Path, n_frames: int, *, camera_h: int = CAMERA_H,
                       camera_w: int = CAMERA_W) -> np.ndarray:
    """Decode up to ``n_frames`` RGB frames from a contest .mkv via PyAV, in the
    camera_size NHWC uint8 layout."""
    import av

    out: list[np.ndarray] = []
    container = av.open(str(path))
    try:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            if img.shape[:2] != (camera_h, camera_w):
                # contest frames are exactly camera_size; guard against surprises.
                raise SystemExit(
                    f"decoded frame {img.shape[:2]} != camera {(camera_h, camera_w)}"
                )
            out.append(np.ascontiguousarray(img, dtype=np.uint8))
            if len(out) >= n_frames:
                break
    finally:
        container.close()
    if not out:
        raise SystemExit(f"no frames decoded from {path}")
    return np.stack(out, axis=0)


def load_frames(spec: str, n_frames: int) -> np.ndarray:
    """Dispatch a ``kind:path`` frame spec (video|raw|npy)."""
    if ":" not in spec:
        raise SystemExit(f"--frames-source must be 'kind:path', got {spec!r}")
    kind, _, raw_path = spec.partition(":")
    path = Path(raw_path)
    if not path.exists():
        raise SystemExit(f"frames path does not exist: {path}")
    if kind == "video":
        return _load_video_frames(path, n_frames)
    if kind == "raw":
        return _load_raw_frames(path, n_frames)
    if kind == "npy":
        arr = np.load(path)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise SystemExit(f"npy must be (N,H,W,3), got {arr.shape}")
        return np.ascontiguousarray(arr[:n_frames], dtype=np.uint8)
    raise SystemExit(f"unknown frames kind {kind!r} (use video|raw|npy)")


# ---------------------------------------------------------------------------
# Tier dispatch.
# ---------------------------------------------------------------------------
def _apply_tier(tier: int, frame: np.ndarray, *, projector, basis, mask,
                frame_index: int) -> tuple[np.ndarray, FrameProof]:
    if tier == 1:
        return apply_tier1_zero_weight_fill(
            frame, projector=projector, basis=basis, mask=mask,
            frame_index=frame_index,
        )
    if tier == 2:
        return apply_tier2_null_basis_descent(
            frame, projector=projector, basis=basis, mask=mask,
            frame_index=frame_index,
        )
    if tier == 3:
        return apply_tier3_blockwise_flat_preimage(
            frame, projector=projector, basis=basis, mask=mask,
            frame_index=frame_index,
        )
    raise SystemExit(f"--tier must be 1|2|3, got {tier}")


def process_frames(frames: np.ndarray, *, tier: int) -> tuple[np.ndarray, list[FrameProof]]:
    """Apply the chosen tier to every frame; return preimage frames + per-frame
    proofs.  Fail-closed: a frame whose residual exceeds the exact tolerance is
    NEVER substituted (kept original) and its proof records the violation."""
    h, w = frames.shape[1], frames.shape[2]
    projector = ResizeProjector.build(camera_h=h, camera_w=w)
    basis = derive_tier1_resize_null_space(
        camera_h=h, camera_w=w,
        scorer_h=projector.scorer_h, scorer_w=projector.scorer_w,
    )
    mask = zero_weight_pixel_mask(camera_h=h, camera_w=w,
                                  scorer_h=projector.scorer_h,
                                  scorer_w=projector.scorer_w, basis=basis)
    out = np.empty_like(frames)
    proofs: list[FrameProof] = []
    for i in range(frames.shape[0]):
        pre, proof = _apply_tier(tier, frames[i], projector=projector, basis=basis,
                                 mask=mask, frame_index=i)
        if not proof.exact:
            # fail-closed: keep the original frame; the proof flags the breach.
            out[i] = frames[i]
        else:
            out[i] = pre
        proofs.append(proof)
    return out, proofs


# ---------------------------------------------------------------------------
# V3 row emission (operator caveat (d): every application emits a V3 row).
# ---------------------------------------------------------------------------
def v3_rows_from_proofs(
    proofs: list[FrameProof], *, base_archive_sha256: str, vehicle: str,
    coder: str = "lzma",
) -> list[dict]:
    """One V3 row per frame application (the methodology-memo schema, extended
    with the class-5 preimage_proof field ``max|Rx̃ - Rx|``)."""
    rows: list[dict] = []
    for p in proofs:
        bytes_freed = (p.bytes_reduction_lzma if coder == "lzma"
                       else p.bytes_reduction_brotli)
        # ΔS is exactly the rate term (distortion delta CERTIFIED 0.0).
        delta_score = preimage_rate_score_delta(bytes_freed)
        rows.append({
            "schema": "resize_null_preimage_application.v3",
            "base_archive_sha256": base_archive_sha256,
            "vehicle": vehicle,
            "pair_id": None,
            "frame_index": p.frame_index,
            "target_frame": "both",
            "payload_section": "rendered_frame_pixels",
            "mutation": f"resize_null_preimage_tier{p.tier}:{p.fill_strategy}",
            "tier": p.tier,
            "d_seg_delta": 0.0,          # CERTIFIED: R x̃ = R x => identical seg
            "d_pose_delta": 0.0,         # CERTIFIED: identical pose (RGB-before-YUV)
            "preimage_proof": p.max_abs_projection_residual,  # class-5 max|Rx̃-Rx|
            "preimage_exact": p.exact,
            "valid_uint8": p.valid_uint8,
            "n_pixels_changed": p.n_pixels_changed,
            "bytes_before": dict(p.bytes_before),
            "bytes_after": dict(p.bytes_after),
            "bytes_freed_coder": coder,
            "bytes_freed": int(bytes_freed),
            "score_delta": delta_score,        # exactly the rate term ΔS
            "delta_score_total": delta_score,
            "first_failed_surface": ("none" if p.exact else "preimage_proof"),
            "keep_or_reject": ("keep" if (p.exact and bytes_freed > 0) else "reject"),
            "proof_evidence_grade": "mathematical-derivation",
            "bytes_evidence_grade": "[macOS-CPU advisory]",
            **V3_FALSE_AUTHORITY,
        })
    return rows


def _aggregate(proofs: list[FrameProof]) -> dict:
    n = len(proofs)
    bb = sum(p.bytes_before["brotli"] for p in proofs)
    ba = sum(p.bytes_after["brotli"] for p in proofs)
    lb = sum(p.bytes_before["lzma"] for p in proofs)
    la = sum(p.bytes_after["lzma"] for p in proofs)
    max_resid = max((p.max_abs_projection_residual for p in proofs), default=0.0)
    all_exact = all(p.exact for p in proofs)
    all_valid = all(p.valid_uint8 for p in proofs)
    return {
        "n_frames": n,
        "all_exact": all_exact,
        "all_valid_uint8": all_valid,
        "max_abs_projection_residual": max_resid,
        "brotli_before": bb, "brotli_after": ba, "brotli_freed": bb - ba,
        "brotli_reduction_pct": (100.0 * (bb - ba) / bb) if bb else 0.0,
        "lzma_before": lb, "lzma_after": la, "lzma_freed": lb - la,
        "lzma_reduction_pct": (100.0 * (lb - la) / lb) if lb else 0.0,
        "rate_score_delta_brotli_total": preimage_rate_score_delta(bb - ba),
        "rate_score_delta_lzma_total": preimage_rate_score_delta(lb - la),
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_out_dir(out_dir: str | None) -> Path:
    if out_dir:
        p = Path(out_dir)
        if str(p).startswith("/tmp"):
            raise SystemExit("--out-dir must be durable (not /tmp)")
        p.mkdir(parents=True, exist_ok=True)
        return p
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for base in SSD_WATERFALL:
        if base.exists():
            p = base / f"resize_null_preimage_{stamp}"
            p.mkdir(parents=True, exist_ok=True)
            return p
    p = REPO_ROOT / "experiments" / "results" / f"resize_null_preimage_{stamp}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-source", help="kind:path (video|raw|npy)")
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--tier", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--out-frames", help="write preimage frames to this .raw")
    ap.add_argument("--proof-json", help="write the aggregate + per-frame proof JSON")
    ap.add_argument("--v3-jsonl", help="write one V3 row per frame application")
    ap.add_argument("--base-archive-sha256", default="unknown",
                    help="base archive sha for the V3 rows")
    ap.add_argument("--vehicle", default="unspecified",
                    help="vehicle label for the V3 rows (snerv_g1b / frontier / ...)")
    # headline mode
    ap.add_argument("--headline", action="store_true",
                    help="run source-vs-vehicle headline measurement")
    ap.add_argument("--source-video", help="contest .mkv for the headline source set")
    ap.add_argument("--vehicle-raw", help="vehicle .raw render for the headline set")
    ap.add_argument("--out-dir", help="durable output dir (default SSD waterfall)")
    args = ap.parse_args(argv)

    if args.headline:
        return _run_headline(args)

    if not args.frames_source:
        ap.error("--frames-source required (or use --headline)")
    frames = load_frames(args.frames_source, args.n_frames)
    t0 = time.time()
    pre, proofs = process_frames(frames, tier=args.tier)
    agg = _aggregate(proofs)
    agg["elapsed_seconds"] = time.time() - t0
    agg["tier"] = args.tier
    agg["frames_source"] = args.frames_source

    if args.out_frames:
        op = Path(args.out_frames)
        if str(op).startswith("/tmp"):
            raise SystemExit("--out-frames must be durable (not /tmp)")
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_bytes(np.ascontiguousarray(pre, dtype=np.uint8).tobytes())
        agg["out_frames_path"] = str(op)
        agg["out_frames_sha256"] = _sha256_file(op)

    if args.v3_jsonl:
        vp = Path(args.v3_jsonl)
        if str(vp).startswith("/tmp"):
            raise SystemExit("--v3-jsonl must be durable (not /tmp)")
        vp.parent.mkdir(parents=True, exist_ok=True)
        rows = v3_rows_from_proofs(proofs, base_archive_sha256=args.base_archive_sha256,
                                   vehicle=args.vehicle)
        vp.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
        agg["v3_jsonl_path"] = str(vp)
        agg["n_v3_rows"] = len(rows)

    out = {"aggregate": agg, "per_frame": [p.to_dict() for p in proofs]}
    if args.proof_json:
        pj = Path(args.proof_json)
        if str(pj).startswith("/tmp"):
            raise SystemExit("--proof-json must be durable (not /tmp)")
        pj.parent.mkdir(parents=True, exist_ok=True)
        pj.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(agg, indent=2, sort_keys=True))
    return 0


def _run_headline(args) -> int:
    if not (args.source_video and args.vehicle_raw):
        raise SystemExit("--headline needs --source-video AND --vehicle-raw")
    out_dir = _resolve_out_dir(args.out_dir)
    result = {
        "schema": "resize_null_preimage_headline.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_frames": args.n_frames,
        "tiers_run": [],
        **V3_FALSE_AUTHORITY,
    }
    sets = {
        "source_video": load_frames(f"video:{args.source_video}", args.n_frames),
        "vehicle_render": load_frames(f"raw:{args.vehicle_raw}", args.n_frames),
    }
    result["inputs"] = {
        "source_video": args.source_video,
        "vehicle_raw": args.vehicle_raw,
    }
    tiers = (1, 2) if args.tier >= 2 else (1,)
    for tier in tiers:
        result["tiers_run"].append(tier)
        for set_name, frames in sets.items():
            t0 = time.time()
            _, proofs = process_frames(frames, tier=tier)
            agg = _aggregate(proofs)
            agg["elapsed_seconds"] = time.time() - t0
            result[f"tier{tier}_{set_name}"] = agg

    out_json = out_dir / "headline.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    result["headline_json_path"] = str(out_json)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
