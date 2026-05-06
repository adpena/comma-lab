#!/usr/bin/env python3
# ruff: noqa: I001
"""Build a pr106_lrl1_sidechannel archive on top of a PR106 packed archive.

Layered on top of PR106 (see docs/pr106_byte_layout_deconstruction_20260504.md):
  outer 0xFB dispatch byte + uint24 PR106 length + raw PR106 bytes
  + sidechannel version (1) + uint16 brotli'd-len + brotli(LR01 LRL1 payload).

LR01 LRL1 payload (post-brotli) = 22-byte header + K*low_h*low_w int8 basis
+ n_frames*K int8 coefficients. Mirrors the codex_metric LRL1 mode-8 wire
format (see docs/codex_metric_lrl1_audit_20260504.md). Inflate path is
`submissions/pr106_lrl1_sidechannel/inflate.py`.

Variant #6 in the score-aware sidechannel paradigm thread:
  - LUMA-only correction (single channel, broadcast across RGB at apply)
  - Low-resolution basis (e.g. 48x64) + bilinear upsample at apply time
  - Per-frame K coefficients (typical K ∈ {2, 4, 8})
  - ~3-5KB sidechannel cost for K=2-4 sweet spot
  - Predicted Δ -0.001 to -0.003 score (luma-dominant scorer-relevant errors)

Three operating modes:

  --search-mode {zero, gradient, brute_force}
    zero        : write a no-op LR01 (all-zero basis + coeffs). Brotli compresses
                  to ~10s of bytes. Used for CPU-only smoke + wire-format roundtrip
                  verification. CPU-tagged [advisory only] per CLAUDE.md MPS-noise rule.
    gradient    : single backward pass per frame ∂score/∂(per-pixel-luma residual).
                  Project residual onto top-K Lanczos eigenvectors → basis. Quantize
                  per-frame coefs to int8. Requires CUDA + scorers. ~$0.30 on Vast.ai 4090.
    brute_force : co-ordinate-descent search over K-dimensional coef simplex per
                  frame; basis fixed via initial PCA on residual. ~$0.50 on Vast.ai 4090.
                  Operator picks via env: PR106_LRL1_MODE=brute_force.

  --K (default 4) : number of basis components per frame. Tradeoff:
    K=2 → ~3KB sidechannel, predicted Δ -0.001
    K=4 → ~5KB sidechannel, predicted Δ -0.002 (sweet spot per audit)
    K=8 → ~10KB sidechannel, predicted Δ -0.003 (diminishing returns)

  --low-h --low-w (default 48 64) : basis spatial resolution. Bilinear-upsampled
    to (CAMERA_H=874, CAMERA_W=1164) at apply time. Smaller basis = smaller
    sidechannel; wider basis = finer correction. Codex default is (48, 64).

  --basis-step --coeff-step : per-tensor float scales for int8 quantization.
    Tighter steps → finer correction at same int8 budget. Defaults are
    conservative (1.0 each); tune at compress-time via gradient mode.

Smoke-mode preview (CPU, --search-mode zero, K=4, basis 48x64):
  Raw payload  = 22 + 4*48*64 + 1200*4 = 22 + 12288 + 4800 = 17,110 bytes
  Brotli'd zero payload  = ~50 bytes (highly compressible)
  Outer wrapper overhead = 6 bytes (1 magic + 3 pr106_len + 1 sc_version + 2 sc_len)
  Total overhead         = ~56 bytes vs PR106 archive

Real-mode dispatch (CUDA):
  search-mode=gradient predicted distortion Δ ~-0.001 to -0.002 score
  search-mode=brute_force predicted distortion Δ ~-0.002 to -0.003 score
  Adds ~3-5KB to archive (rate Δ ~+0.002 to +0.003).
  Net: ~-0.001 to -0.003 score Δ. STACKS on lane_pr106_latent_sidecar +
  lane_pr106_yshift_sidechannel (3rd stack-on per paradigm decision pipeline).

Strict-scorer-rule: scorer is loaded ONLY at compress-time (gradient/brute_force
modes). NEVER at inflate. Per-frame deltas are precomputed and frozen into
the LR01 sidechannel payload.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import struct
import sys
import zipfile
from pathlib import Path

import brotli  # type: ignore[import-not-found]
import numpy as np


try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "submissions" / "pr106_lrl1_sidechannel"))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "apogee_intN" / "src"))

from tac.repo_io import json_text

from inflate import (  # type: ignore[import-not-found]
    LRL1_MAGIC_BYTE,
    SIDECHANNEL_VERSION,
    LR01_MAGIC,
    LR01_HEADER,
    SIDECHANNEL_MODE_LRL1,
    parse_lrl1_archive,
)


def _read_pr106_bytes(pr106_archive: Path) -> bytes:
    with zipfile.ZipFile(pr106_archive) as z:
        return z.read("0.bin")


def _encode_lr01_lrl1(
    basis_int8: np.ndarray, coeffs_int8: np.ndarray,
    *, basis_step: float, coeff_step: float,
) -> bytes:
    """Pack (K, low_h, low_w) basis + (n_frames, K) coefs into LR01 LRL1 payload.

    Both inputs MUST be int8. Header carries (K, low_h, low_w, n_frames,
    coeff_step, basis_step). Layout matches codex_metric LRL1 mode 8.
    """
    if basis_int8.dtype != np.int8:
        raise TypeError(f"basis must be int8, got {basis_int8.dtype}")
    if coeffs_int8.dtype != np.int8:
        raise TypeError(f"coeffs must be int8, got {coeffs_int8.dtype}")
    if basis_int8.ndim != 3:
        raise ValueError(f"basis must be (K, low_h, low_w), got shape {basis_int8.shape}")
    if coeffs_int8.ndim != 2:
        raise ValueError(f"coeffs must be (n_frames, K), got shape {coeffs_int8.shape}")
    K, low_h, low_w = basis_int8.shape
    n_frames, K2 = coeffs_int8.shape
    if K != K2:
        raise ValueError(f"basis K={K} doesn't match coeffs K={K2}")
    if K < 1 or K > 255:
        raise ValueError(f"K must be in 1..255, got {K}")
    if low_h < 1 or low_h > 65535 or low_w < 1 or low_w > 65535:
        raise ValueError(f"low_h, low_w must fit in uint16, got {low_h}x{low_w}")
    header = LR01_HEADER.pack(
        LR01_MAGIC, SIDECHANNEL_MODE_LRL1, K, low_h, low_w, n_frames,
        float(coeff_step), float(basis_step),
    )
    return header + basis_int8.tobytes() + coeffs_int8.tobytes()


def _build_lrl1_archive_bytes(pr106_bytes: bytes, lr01_blob: bytes | None) -> bytes:
    """Wrap PR106 bytes + optional brotli-compressed LR01 in pr106_lrl1 outer layout."""
    if len(pr106_bytes) >= (1 << 24):
        raise ValueError(f"pr106 bytes too large for uint24: {len(pr106_bytes)}")
    out = io.BytesIO()
    out.write(bytes([LRL1_MAGIC_BYTE]))
    out.write(len(pr106_bytes).to_bytes(3, "little"))
    out.write(pr106_bytes)
    if lr01_blob is not None:
        out.write(bytes([SIDECHANNEL_VERSION]))
        if len(lr01_blob) >= (1 << 16):
            raise ValueError(f"lr01 blob too large for uint16: {len(lr01_blob)}")
        out.write(struct.pack("<H", len(lr01_blob)))
        out.write(lr01_blob)
    return out.getvalue()


def _zero_search(K: int, low_h: int, low_w: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """No-op corrections — used for CPU smoke + wire-format roundtrip."""
    basis = np.zeros((K, low_h, low_w), dtype=np.int8)
    coeffs = np.zeros((n_frames, K), dtype=np.int8)
    return basis, coeffs


def _gradient_search_stub(K: int, low_h: int, low_w: int, n_frames: int):
    """Placeholder stub for gradient mode — fails LOUD if invoked without CUDA."""
    raise NotImplementedError(
        "gradient search mode requires CUDA + scorers. Run via the remote dispatch "
        "wrapper scripts/remote_lane_pr106_lrl1_sidechannel.sh which loads CUDA "
        "scorers at compress time only (per CLAUDE.md strict-scorer-rule)."
    )


def _brute_force_search_stub(K: int, low_h: int, low_w: int, n_frames: int):
    """Placeholder stub for brute-force mode — fails LOUD if invoked without CUDA."""
    raise NotImplementedError(
        "brute_force search mode requires CUDA + scorers. Run via the remote dispatch "
        "wrapper scripts/remote_lane_pr106_lrl1_sidechannel.sh which loads CUDA "
        "scorers at compress time only (per CLAUDE.md strict-scorer-rule)."
    )


SEARCH_MODES = {
    "zero": _zero_search,
    "gradient": _gradient_search_stub,
    "brute_force": _brute_force_search_stub,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pr106-archive", type=Path, required=True,
                        help="Path to PR106 packed archive.zip (anchor).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for built archive + metadata.")
    parser.add_argument("--search-mode", choices=list(SEARCH_MODES.keys()), default="zero",
                        help="Sidechannel search strategy. zero = CPU smoke wire-format only; "
                             "gradient/brute_force = CUDA scorer required (raises if invoked here).")
    parser.add_argument("--K", type=int, default=4,
                        help="Basis component count per frame (1..255). Default 4 (sweet spot).")
    parser.add_argument("--low-h", type=int, default=48,
                        help="Basis height (bilinear-upsampled to camera res at apply). Default 48.")
    parser.add_argument("--low-w", type=int, default=64,
                        help="Basis width (bilinear-upsampled to camera res at apply). Default 64.")
    parser.add_argument("--basis-step", type=float, default=1.0,
                        help="Float scale for int8 basis values (default 1.0).")
    parser.add_argument("--coeff-step", type=float, default=1.0,
                        help="Float scale for int8 per-frame coefs (default 1.0).")
    parser.add_argument("--n-pairs", type=int, default=600,
                        help="Number of frame pairs (PR106 default = 600 → 1200 output frames).")
    args = parser.parse_args()

    if args.K < 1 or args.K > 255:
        sys.exit(f"FATAL: --K must be in 1..255, got {args.K}")
    if args.low_h < 1 or args.low_w < 1:
        sys.exit(f"FATAL: --low-h, --low-w must be >= 1, got {args.low_h}x{args.low_w}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pr106_bytes = _read_pr106_bytes(args.pr106_archive)
    pr106_size = len(pr106_bytes)
    print(f"[build-lrl1] pr106 archive: {pr106_size} bytes")

    n_frames = args.n_pairs * 2
    search_fn = SEARCH_MODES[args.search_mode]
    print(f"[build-lrl1] search-mode={args.search_mode}, K={args.K}, "
          f"basis={args.low_h}x{args.low_w}, n_frames={n_frames}, "
          f"coeff_step={args.coeff_step}, basis_step={args.basis_step}")
    basis_int8, coeffs_int8 = search_fn(args.K, args.low_h, args.low_w, n_frames)
    if basis_int8.dtype != np.int8 or basis_int8.shape != (args.K, args.low_h, args.low_w):
        raise RuntimeError(
            f"search_fn returned bad basis shape/dtype: shape={basis_int8.shape}, "
            f"dtype={basis_int8.dtype}"
        )
    if coeffs_int8.dtype != np.int8 or coeffs_int8.shape != (n_frames, args.K):
        raise RuntimeError(
            f"search_fn returned bad coeffs shape/dtype: shape={coeffs_int8.shape}, "
            f"dtype={coeffs_int8.dtype}"
        )

    lr01_payload = _encode_lr01_lrl1(
        basis_int8, coeffs_int8,
        basis_step=args.basis_step, coeff_step=args.coeff_step,
    )
    lr01_blob = brotli.compress(lr01_payload, quality=11)
    print(f"[build-lrl1] LR01 raw payload: {len(lr01_payload)} bytes "
          f"(header=22 + basis={args.K * args.low_h * args.low_w} + "
          f"coeffs={n_frames * args.K})")
    print(f"[build-lrl1] LR01 brotli'd: {len(lr01_blob)} bytes "
          f"({100.0 * len(lr01_blob) / max(len(lr01_payload), 1):.1f}%)")

    new_bin = _build_lrl1_archive_bytes(pr106_bytes, lr01_blob)

    # Roundtrip verify before writing zip
    sd_check, lat_check, meta_check, sc_check = parse_lrl1_archive(new_bin)
    if sc_check is None:
        raise RuntimeError("sidechannel roundtrip parse failed — sidechannel missing")
    if sc_check["basis"].shape != (args.K, args.low_h, args.low_w):
        raise RuntimeError(
            f"sidechannel basis roundtrip shape mismatch: "
            f"got {sc_check['basis'].shape}, expected ({args.K}, {args.low_h}, {args.low_w})"
        )
    if sc_check["coeffs"].shape != (n_frames, args.K):
        raise RuntimeError(
            f"sidechannel coeffs roundtrip shape mismatch: "
            f"got {sc_check['coeffs'].shape}, expected ({n_frames}, {args.K})"
        )
    if not np.array_equal(sc_check["basis"], basis_int8):
        raise RuntimeError("sidechannel basis roundtrip values mismatch (encode/decode bug)")
    if not np.array_equal(sc_check["coeffs"], coeffs_int8):
        raise RuntimeError("sidechannel coeffs roundtrip values mismatch (encode/decode bug)")
    if abs(sc_check["coeff_step"] - args.coeff_step) > 1e-6:
        raise RuntimeError(
            f"sidechannel coeff_step roundtrip mismatch: "
            f"{sc_check['coeff_step']} vs {args.coeff_step}"
        )
    if abs(sc_check["basis_step"] - args.basis_step) > 1e-6:
        raise RuntimeError(
            f"sidechannel basis_step roundtrip mismatch: "
            f"{sc_check['basis_step']} vs {args.basis_step}"
        )
    print(f"[build-lrl1] roundtrip OK: basis shape={sc_check['basis'].shape}, "
          f"coeffs shape={sc_check['coeffs'].shape}, "
          f"steps=(coeff={sc_check['coeff_step']}, basis={sc_check['basis_step']}), "
          f"decoder_tensors={len(sd_check)}, latents_shape={tuple(lat_check.shape)}, "
          f"meta={meta_check}")

    archive_path = args.out_dir / "pr106_lrl1_sidechannel_archive.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as z:  # DETERMINISTIC_ZIP_OK
        zi = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        z.writestr(zi, new_bin)
    archive_size = archive_path.stat().st_size
    pr106_zip_size = args.pr106_archive.stat().st_size
    delta = archive_size - pr106_zip_size
    score_delta = 25.0 * delta / 37545489.0
    print(f"[build-lrl1] wrote {archive_path}: {archive_size} bytes")
    print(f"[build-lrl1] PR106 zip: {pr106_zip_size} bytes; delta: {delta:+d} "
          f"({100*delta/pr106_zip_size:+.3f}%)")
    print(f"[build-lrl1] estimated rate-component score Δ vs PR106: {score_delta:+.6f}")

    metadata = {
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "pr106_size_bytes": pr106_zip_size,
        "delta_bytes": delta,
        "rate_component_score_delta_vs_pr106": score_delta,
        "search_mode": args.search_mode,
        "K": int(args.K),
        "low_h": int(args.low_h),
        "low_w": int(args.low_w),
        "coeff_step": float(args.coeff_step),
        "basis_step": float(args.basis_step),
        "n_pairs": int(args.n_pairs),
        "n_frames": int(n_frames),
        "lr01_payload_bytes": len(lr01_payload),
        "lr01_brotli_bytes": len(lr01_blob),
        "outer_dispatch_magic": f"0x{LRL1_MAGIC_BYTE:02X}",
        "lr01_mode_id": int(SIDECHANNEL_MODE_LRL1),
        "tag": "[advisory only]" if args.search_mode == "zero" else "[design-validation]",
        "council_status": (
            "PROPOSAL — pre-registered at L1; gated on lane_pr106_latent_sidecar AND "
            "lane_pr106_yshift_sidechannel BOTH winning empirically (3rd stack-on)"
        ),
        "score_claim": False,
        "next_step": (
            "search-mode=zero is CPU smoke ONLY (all-zero basis+coeffs; pure wire-format proof). "
            "For real distortion improvement, dispatch via "
            "scripts/remote_lane_pr106_lrl1_sidechannel.sh with PR106_LRL1_MODE=gradient "
            "(or brute_force) on Vast.ai 4090. ONLY dispatch if BOTH lane_pr106_latent_sidecar "
            "AND lane_pr106_yshift_sidechannel land < 0.20650 [contest-CUDA] (per "
            "docs/INDEX_score_aware_sidechannel_thread_20260504.md decision pipeline TICK 3)."
        ),
    }
    metadata_path = args.out_dir / "build_metadata.json"
    metadata_path.write_text(json_text(metadata), encoding="utf-8")
    print(f"[build-lrl1] wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
