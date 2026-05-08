#!/usr/bin/env python3
"""Build a PR101 archive from a fine-tuned HNeRV decoder state_dict.

Phase A1 (track1_phase_a1_score_gradient) closure tool. Takes the EMA checkpoint
produced by ``experiments/train_score_gradient_pr101_finetune.py`` and rebuilds
the archive.zip using:

    1. The fine-tuned decoder state_dict re-encoded via PR101 split-Brotli
       (``tac.pr101_split_brotli_codec.encode_decoder_compact`` with default
       byte_maps for canonical wire-format compatibility).
    2. The ORIGINAL latent_blob + sidecar_blob from the source PR101 archive
       (NOT regenerated; the model's latent_blob is part of the substrate
       custody and must be preserved bit-for-bit).
    3. The "no-dead-K" wire format used by tools/pr101_unified_winners_stack_empirical:
       ``[uint32 LE: decoder_section_total_bytes] + [per_tensor_fp16_scales] +
        [brotli_payload]`` so the encoded decoder_blob can vary in size from
       the original 162,164 bytes.
    4. A forked submission_dir/{inflate.py, inflate.sh, src/codec.py, src/model.py}
       that decodes this format. The forked inflate.py is a verbatim copy of
       ``INFLATE_PY_NO_DEAD_K`` from the unified-winners-stack tool.

Usage:
    .venv/bin/python tools/build_pr101_finetuned_archive.py \\
        --state-dict experiments/results/track1_phase_a1_score_gradient/checkpoint_ema.pt \\
        --source-archive experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/archive.zip \\
        --pr101-source-dir experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/src \\
        --output-dir experiments/results/track1_a1_finetuned_archive

Outputs:
    <output-dir>/archive.zip                — submission archive
    <output-dir>/submission_dir/            — inflate.sh + inflate.py + src/
    <output-dir>/build_manifest.json        — provenance + bytes/sha
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import struct
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.pr101_split_brotli_codec import (
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    encode_decoder_compact,
)


# ---------------------------------------------------------------------------
# Source archive parsing
# ---------------------------------------------------------------------------

def _read_pr101_inner_blob(archive_path: Path) -> bytes:
    """Extract the single inner ZIP member 'x' from a PR101-shaped archive."""
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if "x" not in names:
            raise SystemExit(
                f"FATAL: archive {archive_path} has no member 'x'; got {names!r}"
            )
        return zf.read("x")


def _split_pr101_inner_blob(inner: bytes) -> tuple[bytes, bytes, bytes]:
    if len(inner) < DECODER_BLOB_LEN + LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: inner blob {len(inner)} < required {DECODER_BLOB_LEN + LATENT_BLOB_LEN}"
        )
    decoder_blob = inner[:DECODER_BLOB_LEN]
    latent_blob = inner[DECODER_BLOB_LEN : DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar_blob = inner[DECODER_BLOB_LEN + LATENT_BLOB_LEN :]
    return decoder_blob, latent_blob, sidecar_blob


# ---------------------------------------------------------------------------
# No-dead-K wire format
# ---------------------------------------------------------------------------

def _build_no_dead_k_decoder_section(
    state_dict: dict,
    *,
    brotli_quality: int,
) -> tuple[bytes, dict]:
    """Encode state_dict in the no-dead-K wire format.

    Wire format:
        uint32 LE: decoder_section_total_bytes (D, including header + scales + payload)
        byte * (n_tensors * 2): per_tensor_fp16_scale (LE half)
        byte * (D - 4 - n_tensors*2): brotli(int8s)

    NOTE: this uses PR101's canonical encoder. encode_decoder_compact returns
    the concatenation of brotli streams + interleaved fp16 scales as one blob.
    The no-dead-K format simply prefixes a uint32 length so the inflate side
    can find latent_blob without relying on a fixed 162,164-byte offset.
    """
    # Get the canonical encoded blob (same bytes PR101's own decoder consumes,
    # but we'll wrap with a length prefix for variable-size support).
    encoded_blob = encode_decoder_compact(
        state_dict,
        brotli_quality=brotli_quality,
    )
    # Prefix with uint32 LE length (4 bytes) of the section payload.
    section_payload_bytes = len(encoded_blob)
    section_total_bytes = 4 + section_payload_bytes  # header + payload
    header = struct.pack("<I", section_total_bytes)
    section_bytes = header + encoded_blob
    stats = {
        "section_total_bytes": section_total_bytes,
        "encoded_blob_bytes": section_payload_bytes,
        "n_tensors_in_schema": len(FIXED_STATE_SCHEMA),
        "brotli_quality": brotli_quality,
    }
    return section_bytes, stats


def _build_inner_blob(decoder_section: bytes, latent_blob: bytes, sidecar_blob: bytes) -> bytes:
    return decoder_section + latent_blob + sidecar_blob


def _write_archive_zip(archive_path: Path, inner_blob: bytes) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    # ZIP_STORED to match PR101's wire-format expectation.
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x")
        info.compress_type = zipfile.ZIP_STORED
        # Use deterministic timestamp (CLAUDE.md gate 19 / artifact lifecycle).
        info.date_time = (1980, 1, 1, 0, 0, 0)
        zf.writestr(info, inner_blob)


# ---------------------------------------------------------------------------
# Forked inflate.py source (no-dead-K wire format)
# ---------------------------------------------------------------------------

INFLATE_PY_NO_DEAD_K = '''#!/usr/bin/env python
"""Forked PR101 inflate for fine-tuned A1 archive (no-dead-K wire format).

Wire format (inner blob, single ZIP member \'x\'):
    uint32 LE: decoder_section_total_bytes (D)
    byte * (D - 4): encoded decoder blob (PR101 split-Brotli, canonical)
    byte * 15387: latent_blob (PR101 ORIGINAL — preserved from source archive)
    byte * remaining: sidecar_blob (PR101 ORIGINAL)
"""
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (
    LATENT_BLOB_LEN,
    decode_decoder_compact,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder

CAMERA_H, CAMERA_W = 874, 1164
EVAL_H, EVAL_W = 384, 512
LATENT_DIM = 28
BASE_CHANNELS = 36
N_PAIRS = 600


def parse_a1_finetuned_archive(archive_bytes: bytes):
    if len(archive_bytes) < 4:
        raise ValueError("archive too short to read decoder section header")
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    if section_total < 4 or section_total > len(archive_bytes):
        raise ValueError(f"bad decoder_section_total {section_total}")
    decoder_blob = archive_bytes[4:section_total]
    latent_blob = archive_bytes[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + LATENT_BLOB_LEN:]
    if not decoder_blob or len(latent_blob) != LATENT_BLOB_LEN:
        raise ValueError("bad finetuned-A1 archive layout")
    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents = parse_a1_finetuned_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=(EVAL_H, EVAL_W),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    latents = latents.to(device)
    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, N_PAIRS, 16):
            j = min(i + 16, N_PAIRS)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, EVAL_H, EVAL_W)
            up = F.interpolate(
                flat, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            fout.write(frames.tobytes())
            n += batch * 2

    print(f"saved {n} frames")
    return n


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python inflate.py <src.bin> <dst.raw>")
    inflate(sys.argv[1], sys.argv[2])
'''


INFLATE_SH_NO_DEAD_K = '''#!/usr/bin/env bash
# Fine-tuned-PR101 (Phase A1 score-gradient) inflate.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/x"
  if [ ! -f "$SRC" ]; then
    SRC="${DATA_DIR}/${BASE}.bin"
  fi
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  python "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
'''


# ---------------------------------------------------------------------------
# Submission_dir staging
# ---------------------------------------------------------------------------

_PR101_NESTED_SUBMISSION_PATH = Path("submissions") / "hnerv_ft_microcodec" / "src"


def _resolve_pr101_codec_dir(pr101_source_dir: Path) -> Path:
    """Locate the directory that actually holds codec.py + model.py.

    Two layouts are accepted:
      1. ``pr101_source_dir/codec.py`` (flat — legacy/test fixtures)
      2. ``pr101_source_dir/submissions/hnerv_ft_microcodec/src/codec.py``
         (PR101 intake clone — what `--pr101-source-dir` typically points at)
    """
    flat_codec = pr101_source_dir / "codec.py"
    flat_model = pr101_source_dir / "model.py"
    if flat_codec.is_file() and flat_model.is_file():
        return pr101_source_dir
    nested = pr101_source_dir / _PR101_NESTED_SUBMISSION_PATH
    if (nested / "codec.py").is_file() and (nested / "model.py").is_file():
        return nested
    raise SystemExit(
        "FATAL: PR101 codec.py/model.py not found at either layout under "
        f"{pr101_source_dir} (tried flat root and {_PR101_NESTED_SUBMISSION_PATH})"
    )


def _stage_submission_dir(submission_dir: Path, pr101_source_dir: Path) -> None:
    """Copy PR101 codec.py + model.py + write forked inflate.py + inflate.sh."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    codec_dir = _resolve_pr101_codec_dir(pr101_source_dir)
    for fname in ("codec.py", "model.py"):
        src_path = codec_dir / fname
        if not src_path.is_file():
            raise SystemExit(f"FATAL: PR101 source missing: {src_path}")
        shutil.copy2(src_path, src_dir / fname)

    inflate_py = submission_dir / "inflate.py"
    inflate_py.write_text(INFLATE_PY_NO_DEAD_K, encoding="utf-8")
    inflate_py.chmod(0o755)

    inflate_sh = submission_dir / "inflate.sh"
    inflate_sh.write_text(INFLATE_SH_NO_DEAD_K, encoding="utf-8")
    inflate_sh.chmod(0o755)


# ---------------------------------------------------------------------------
# Smoke roundtrip (CPU)
# ---------------------------------------------------------------------------

def _smoke_roundtrip(archive_path: Path, state_dict_in: dict) -> dict:
    """Decode the new archive bytes back to a state_dict and compare to input."""
    inner = _read_pr101_inner_blob(archive_path)
    if len(inner) < 4:
        return {"smoke_ok": False, "reason": "inner blob too short"}
    section_total = struct.unpack_from("<I", inner, 0)[0]
    if section_total < 4 or section_total > len(inner):
        return {"smoke_ok": False, "reason": f"bad section_total {section_total}"}
    decoder_blob = inner[4:section_total]
    from tac.pr101_split_brotli_codec import decode_decoder_compact
    sd_decoded = decode_decoder_compact(decoder_blob)
    # rel_err vs the input state_dict (which is already int8-quantized after first encode pass)
    rel_errs = []
    for name, _shape in FIXED_STATE_SCHEMA:
        a = state_dict_in[name].detach().cpu().float()
        b = sd_decoded[name].cpu().float()
        denom = (a.abs().mean() + 1e-12)
        rel = (a - b).abs().mean() / denom
        rel_errs.append(float(rel))
    return {
        "smoke_ok": True,
        "max_rel_err": max(rel_errs),
        "mean_rel_err": sum(rel_errs) / len(rel_errs),
        "n_tensors": len(rel_errs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--state-dict", type=Path, required=True,
                   help="Path to fine-tuned EMA checkpoint (.pt) produced by Phase A1 training")
    p.add_argument("--source-archive", type=Path, required=True,
                   help="Path to the source PR101 archive.zip (provides latent_blob + sidecar_blob)")
    p.add_argument("--pr101-source-dir", type=Path, required=True,
                   help="Path to PR101 source dir containing codec.py + model.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lane-id", default="track1_phase_a1_score_gradient")
    p.add_argument("--brotli-quality", type=int, default=11)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        sys.exit(f"FATAL: --state-dict not found: {args.state_dict}")
    if not args.source_archive.is_file():
        sys.exit(f"FATAL: --source-archive not found: {args.source_archive}")
    if not args.pr101_source_dir.is_dir():
        sys.exit(f"FATAL: --pr101-source-dir not found: {args.pr101_source_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load fine-tuned state_dict.
    sd_finetuned = torch.load(args.state_dict, map_location="cpu", weights_only=True)
    if not isinstance(sd_finetuned, dict):
        sys.exit(f"FATAL: --state-dict {args.state_dict} did not load as a dict")

    # Read source archive to extract latent_blob + sidecar_blob.
    pr101_inner = _read_pr101_inner_blob(args.source_archive)
    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)

    # Encode fine-tuned decoder.
    decoder_section, encode_stats = _build_no_dead_k_decoder_section(
        sd_finetuned,
        brotli_quality=args.brotli_quality,
    )

    # Build inner blob and write archive.
    inner_blob = _build_inner_blob(decoder_section, latent_blob, sidecar_blob)
    archive_path = output_dir / "archive.zip"
    _write_archive_zip(archive_path, inner_blob)

    # Stage submission_dir.
    submission_dir = output_dir / "submission_dir"
    _stage_submission_dir(submission_dir, args.pr101_source_dir)
    # Copy archive into submission_dir as well so contest_auth_eval can find it.
    shutil.copy2(archive_path, submission_dir / "archive.zip")

    # Smoke roundtrip.
    smoke = _smoke_roundtrip(archive_path, sd_finetuned)

    # Compute provenance.
    archive_bytes = archive_path.stat().st_size
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    state_dict_sha = hashlib.sha256(args.state_dict.read_bytes()).hexdigest()

    manifest = {
        "lane_id": args.lane_id,
        "schema_version": "pr101_finetuned_archive_v1",
        "build_timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wire_format": "no_dead_k_uint32_section_prefix",
        "source_state_dict_path": str(args.state_dict),
        "source_state_dict_sha256": state_dict_sha,
        "source_archive_path": str(args.source_archive),
        "pr101_source_dir": str(args.pr101_source_dir),
        "archive_path": str(archive_path),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "encode_stats": encode_stats,
        "latent_blob_bytes": len(latent_blob),
        "sidecar_blob_bytes": len(sidecar_blob),
        "smoke": smoke,
        "score_claim": False,
        "byte_proxy_only": False,
        # CLAUDE.md "Comment-only contracts — FORBIDDEN": this archive build
        # alone is NOT proof of dispatch-readiness. The remote script's
        # Stage 4 REPORT writes a separate manifest with the authoritative
        # CUDA eval result + flips its own flag. This builder MUST set False.
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[CPU-build proxy; no contest-CUDA eval yet]",
        "dispatch_blockers": ["awaiting_contest_cuda_exact_eval"],
        "tag_discipline": {
            "before_eval": "[advisory only]",
            "after_eval": "[contest-CUDA] iff Lightning T4 dispatch produces evaluate.py output",
        },
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[build_pr101_finetuned] archive_bytes={archive_bytes:,} sha256={archive_sha[:16]}...")
    print(f"[build_pr101_finetuned] smoke: {smoke}")
    print(f"[build_pr101_finetuned] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
