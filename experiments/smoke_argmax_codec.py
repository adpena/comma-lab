"""End-to-end smoke test for the argmax_rle codec.

Decodes the highest-fidelity AV1 mask file we have (CRF30 — closest
proxy to lossless without exploding the file size), encodes it via the
new lossless AMRC codec, and prints the size ratios versus the AV1
reference points actually shipping in the project.

Yousfi council recommendation #8 (2026-04-26): replacing AV1 monochrome
mask encoding with a lossless RLE+Huffman+delta codec was projected to
cut 0.05–0.10 score points off the rate term. The numbers this script
prints are the empirical realization of that prediction.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.lossless.argmax_codec import (  # noqa: E402
    NUM_CLASSES,
    SAME_AS_PREV_SYMBOL,
    pack_archive,
    unpack_archive,
)


REAL_MASK_DIR = REPO_ROOT / "experiments" / "results" / "mask_sweep_20260425T142245"
AV1_CRF30 = REAL_MASK_DIR / "masks_av1mono_full_crf30.mkv"
AV1_CRF50 = REAL_MASK_DIR / "masks_av1mono_full_crf50.mkv"
AV1_CRF40 = REAL_MASK_DIR / "masks_av1mono_full_crf40.mkv"
AV1_CRF63 = REAL_MASK_DIR / "masks_av1mono_full_crf63.mkv"
ENTROPY_FULL_BIN = REAL_MASK_DIR / "masks_entropy_full.bin"


def _decode_av1_mask_video(path: Path) -> torch.Tensor:
    """Decode an AV1 monochrome mask video back to (N, H, W) class labels."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(path)],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "gray", "-v", "error", "pipe:1"],
        capture_output=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed: {proc.stderr.decode('utf-8', errors='replace')}"
        )
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = len(raw) // (h * w)
    pixels = raw.reshape(n, h, w)
    scale = 255 // (NUM_CLASSES - 1)
    masks = np.clip(np.round(pixels.astype(np.float32) / scale), 0,
                    NUM_CLASSES - 1).astype(np.int64)
    return torch.from_numpy(masks)


def _summarize_run_lengths(masks: torch.Tensor) -> None:
    """Print per-class average run length (intra-frame, scanline order)."""
    arr = masks.numpy()
    print("\n[run-length stats by class, intra-frame scanline order]")
    for cls in range(NUM_CLASSES):
        runs: list[int] = []
        for k in range(arr.shape[0]):
            flat = arr[k].reshape(-1)
            mask = (flat == cls).astype(np.int8)
            # Find run starts/ends via diff.
            d = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            if len(starts) > 0:
                runs.extend((ends - starts).tolist())
        if runs:
            avg = sum(runs) / len(runs)
            print(
                f"  class {cls}: n_runs={len(runs):>9,}  "
                f"avg_run_len={avg:>8.1f}  max={max(runs):>7,}  "
                f"total_pixels={sum(runs):>10,}"
            )
        else:
            print(f"  class {cls}: never appears")


def _summarize_delta_symbols(masks: torch.Tensor) -> None:
    """Print delta-symbol histogram across all frames after frame 0."""
    arr = masks.numpy()
    counts = Counter()
    n = arr.shape[0]
    if n < 2:
        return
    for k in range(1, n):
        diff = (arr[k] != arr[k - 1])
        counts[SAME_AS_PREV_SYMBOL] += int((~diff).sum())
        for cls in range(NUM_CLASSES):
            counts[cls] += int(((arr[k] == cls) & diff).sum())
    total = sum(counts.values())
    print("\n[delta-symbol distribution (frames 1..N-1)]")
    print(
        f"  symbol {SAME_AS_PREV_SYMBOL} (same-as-prev): "
        f"{counts[SAME_AS_PREV_SYMBOL]:>12,} "
        f"({100 * counts[SAME_AS_PREV_SYMBOL] / total:.2f}%)"
    )
    for cls in range(NUM_CLASSES):
        pct = 100 * counts[cls] / total if total else 0
        print(
            f"  symbol {cls} (new class {cls}):     "
            f"{counts[cls]:>12,} ({pct:.4f}%)"
        )


def _load_clean_masks() -> tuple[torch.Tensor, str]:
    """Prefer the lossless entropy-coded reference (pure SegNet argmax,
    no AV1 noise) when present — that is the apples-to-apples target the
    codec is meant to replace. Fall back to the AV1 CRF30 file for parity
    with the brief.
    """
    if ENTROPY_FULL_BIN.exists():
        from tac.mask_entropy_coder import decode_masks_entropy
        masks = decode_masks_entropy(str(ENTROPY_FULL_BIN))
        return masks, str(ENTROPY_FULL_BIN)
    return _decode_av1_mask_video(AV1_CRF30), str(AV1_CRF30)


def main() -> int:
    if not AV1_CRF30.exists():
        print(f"ERROR: real mask fixture not found at {AV1_CRF30}", file=sys.stderr)
        return 2

    print("Loading reference mask sequence ...")
    t0 = time.monotonic()
    masks, src = _load_clean_masks()
    t_decode_av1 = time.monotonic() - t0
    print(
        f"  Source: {src}\n"
        f"  Loaded {masks.shape[0]} frames @ {masks.shape[1]}x{masks.shape[2]} "
        f"in {t_decode_av1:.2f}s"
    )

    print("\nEncoding via AMRC codec ...")
    with tempfile.NamedTemporaryFile(suffix=".amrc", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        t0 = time.monotonic()
        amrc_bytes = pack_archive(masks, out_path)
        t_encode = time.monotonic() - t0
        print(f"  Encoded in {t_encode:.2f}s")

        t0 = time.monotonic()
        recovered = unpack_archive(out_path)
        t_decode = time.monotonic() - t0
        print(f"  Decoded in {t_decode:.2f}s")

        if not torch.equal(recovered, masks):
            n_diff = (recovered != masks).sum().item()
            print(
                f"FATAL: round-trip MISMATCH — {n_diff} pixels differ. "
                f"Codec is broken.",
                file=sys.stderr,
            )
            return 1
        print("  Round-trip: bit-identical to source")

        # ── Headline numbers ──
        av1_crf30 = AV1_CRF30.stat().st_size
        av1_crf50 = AV1_CRF50.stat().st_size if AV1_CRF50.exists() else 421054
        av1_crf40 = AV1_CRF40.stat().st_size if AV1_CRF40.exists() else 803_321
        av1_crf63 = AV1_CRF63.stat().st_size if AV1_CRF63.exists() else 108_724
        entropy_size = ENTROPY_FULL_BIN.stat().st_size if ENTROPY_FULL_BIN.exists() else None
        ratio_30 = amrc_bytes / av1_crf30
        ratio_50 = amrc_bytes / av1_crf50
        ratio_40 = amrc_bytes / av1_crf40
        ratio_63 = amrc_bytes / av1_crf63
        print(
            f"\n[HEADLINE] bytes={amrc_bytes:,}, "
            f"ratio_vs_av1_crf30={ratio_30:.3f}, "
            f"ratio_vs_av1_crf50={ratio_50:.3f}, "
            f"ratio_vs_av1_crf40={ratio_40:.3f}, "
            f"ratio_vs_av1_crf63={ratio_63:.3f}"
        )
        if entropy_size is not None:
            ratio_entropy = amrc_bytes / entropy_size
            print(
                f"           ratio_vs_entropy_lossless={ratio_entropy:.3f} "
                f"(both lossless — apples-to-apples)"
            )

        # Rate-term contribution (contest scoring formula).
        ORIGINAL_VIDEO_BYTES = 37_545_489
        rate_amrc = 25 * amrc_bytes / ORIGINAL_VIDEO_BYTES
        rate_av1_crf30 = 25 * av1_crf30 / ORIGINAL_VIDEO_BYTES
        rate_av1_crf50 = 25 * av1_crf50 / ORIGINAL_VIDEO_BYTES
        rate_av1_crf63 = 25 * av1_crf63 / ORIGINAL_VIDEO_BYTES
        print(
            f"\n[rate term contribution] "
            f"AMRC={rate_amrc:.4f}, AV1_CRF30={rate_av1_crf30:.4f}, "
            f"AV1_CRF50={rate_av1_crf50:.4f}, AV1_CRF63={rate_av1_crf63:.4f}"
        )
        print(
            f"  Score delta vs CRF50: {rate_amrc - rate_av1_crf50:+.4f} "
            f"(negative = AMRC wins)"
        )
        print(
            f"  Score delta vs CRF30: {rate_amrc - rate_av1_crf30:+.4f}"
        )
        print(
            f"  Score delta vs CRF63: {rate_amrc - rate_av1_crf63:+.4f} "
            f"(CRF63 is contest's smallest AV1)"
        )
        if entropy_size is not None:
            rate_entropy = 25 * entropy_size / ORIGINAL_VIDEO_BYTES
            print(
                f"  Score delta vs entropy_lossless ({entropy_size:,}B → "
                f"{rate_entropy:.4f}): {rate_amrc - rate_entropy:+.4f}"
            )

        _summarize_run_lengths(masks)
        _summarize_delta_symbols(masks)

    finally:
        out_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
