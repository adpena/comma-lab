#!/usr/bin/env python
"""Inflate factorized_hnerv_v1 archive: SVD-factorized HNeRV decoder + PR-style latents.

Wire format:
    archive[0]      = 0xF1 magic byte (factorized HNeRV v1)
    archive[1..5]   = uint32 decoder_section_len (LE)
    archive[5..]    = factorized section: "FHN1" magic + section header +
                      index table + brotli(factorized records) + brotli(non-
                      factorized records). See submissions.factorized_hnerv_v1.src.codec.
    [next]          = uint32 latent_section_len (LE)
    [next]          = brotli'd PR-style latent payload.

Strict-scorer-rule: this inflate path loads NO scorer weights. All work is
HNeRVDecoder forward + bicubic upsample + uint8 cast.

Per CLAUDE.md "FORBIDDEN PATTERNS / forbidden device-selection defaults":
this script REQUIRES CUDA when explicitly opted into score lanes (per
``--device cuda``). For local CPU-mode dual-eval (per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA"), the inflate honors the
``FACTORIZED_HNERV_DEVICE`` env var (set to "cpu" by the GHA CPU-eval
launcher; defaults to "cuda" for any score-lane invocation).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))

from codec import parse_archive  # type: ignore[import-not-found]  # noqa: E402
from model import HNeRVDecoder  # type: ignore[import-not-found]  # noqa: E402

CAMERA_H, CAMERA_W = 874, 1164


def _resolve_device() -> torch.device:
    """Resolve the inflate device. CUDA-required by default; CPU only when
    the dual-eval pipeline explicitly opts in via env var.

    Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
    CONTEST-COMPLIANT HARDWARE": the GHA-driven CPU eval sets
    FACTORIZED_HNERV_DEVICE=cpu; CUDA score-lane runs leave it unset.
    """
    requested = os.environ.get("FACTORIZED_HNERV_DEVICE", "cuda").lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            sys.exit(
                "factorized_hnerv_v1 inflate: CUDA requested but unavailable. "
                "Set FACTORIZED_HNERV_DEVICE=cpu only for the CPU-eval "
                "dual-track lane (NOT for score claims; tag those [contest-CPU])."
            )
        return torch.device("cuda")
    sys.exit(f"factorized_hnerv_v1 inflate: unknown FACTORIZED_HNERV_DEVICE={requested!r}")


def inflate(src_bin: str, dst_raw: str) -> int:
    archive_bytes = Path(src_bin).read_bytes()
    decoder_sd, latents, meta = parse_archive(archive_bytes)

    device = _resolve_device()
    decoder = HNeRVDecoder(
        latent_dim=meta["latent_dim"],
        base_channels=meta["base_channels"],
        eval_size=tuple(meta["eval_size"]),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n_pairs = meta["n_pairs"]
    eval_h, eval_w = meta["eval_size"]
    print(
        f"factorized_hnerv_v1: device={device}, decoder loaded, running forward...",
        file=sys.stderr,
    )

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            B = j - i
            decoded = decoder(latents[i:j])  # (B, 2, 3, eval_h, eval_w)
            flat = decoded.reshape(B * 2, 3, eval_h, eval_w)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
            )
            frames = (
                up.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()
            )
            fout.write(frames.tobytes())
            n += B * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m submissions.factorized_hnerv_v1.inflate <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
