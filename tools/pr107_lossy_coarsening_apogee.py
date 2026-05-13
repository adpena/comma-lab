#!/usr/bin/env python3
"""PR107 (apogee) per-tensor analytical lossy coarsening — adapter of
``tools/pr101_lossy_coarsening_analytical.py`` for PR107's archive layout.

Substrate verification
----------------------
PR107 (apogee) ships a 28-tensor HNeRVDecoder with the **exact same**
``FIXED_STATE_SCHEMA`` as PR101 (228,958 INT8 parameters, identical
``(name, shape)`` tuples; latent_dim=28, base_channels=36, eval_size=(384,512)).
PR107's archive layout differs from PR101 only in framing:

    PR107 archive (member 0.bin) layout:
        [meta_brotli_len:u32][meta_brotli]              ~84 B
        [decoder_blob_len:u32][decoder_brotli(CD1 ...)]   162,347 B
        [latents_brotli_len:u32][latents_brotli]         15,853 B

    where CD1 = magic(3) + scale_bits(1) + n(4) + 28*fp16_scales(56) +
                228958 zigzag-bytes (= 229,022 raw, ~162,343 brotli'd)

This tool applies per-tensor K-coarsening to the decoder INT8 symbols only,
preserving the meta + latents sections byte-for-byte. The wire format stays
the existing CD1 codec (PR107's stock ``decode_decoder``), so inflate-time
parsing requires zero changes.

Operator authorization 2026-05-08: medal-path dispatch.
PR101 lossy_coarsening anchor: -21,800 B at 3.86% rel_err [CPU-prep].
Predicted PR107 score impact (rate term only): 25 * delta_B / 37545489.

CLAUDE.md compliance:
- No scorers loaded at compress or inflate time (strict-scorer-rule).
- Archive built by python ``zipfile.ZipFile`` (not shell ``zip``).
- All score claims tagged ``[contest-CPU]`` or ``[contest-CUDA]`` after
  dispatch; this tool itself emits ``[CPU-prep]`` byte anchor only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
APOGEE_SRC = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto"
    / "source/submissions/apogee/src"
)
sys.path.insert(0, str(APOGEE_SRC))

from tac.codec.rel_err import REL_ERR_FORM_KEY, RelErrForm, compute_rel_err  # noqa: E402

# Use the upstream PR107 codec verbatim for parse + the CD1 reference shape.
from codec import (  # type: ignore # noqa: E402
    parse_archive,
    quantize_state_dict,
    zigzag_encode_i8,
    N_QUANT,
)

TOOL_NAME = "tools/pr107_lossy_coarsening_apogee.py"
SCHEMA_VERSION = "pr107_lossy_coarsening_apogee.v1"
PR107_BASELINE_ARCHIVE_BYTES = 178_392  # archive.zip on disk
PR107_BASELINE_BIN_BYTES = 178_284  # 0.bin member content

# Architecture-ordered CD1 tensor names (matches HNeRVDecoder state_dict()).
ARCHITECTURE_ORDER = [
    "stem.weight",
    "stem.bias",
    "blocks.0.weight",
    "blocks.0.bias",
    "blocks.1.weight",
    "blocks.1.bias",
    "blocks.2.weight",
    "blocks.2.bias",
    "blocks.3.weight",
    "blocks.3.bias",
    "blocks.4.weight",
    "blocks.4.bias",
    "blocks.5.weight",
    "blocks.5.bias",
    "skips.2.weight",
    "skips.2.bias",
    "skips.3.weight",
    "skips.3.bias",
    "skips.4.weight",
    "skips.4.bias",
    "refine.0.weight",
    "refine.0.bias",
    "refine.1.weight",
    "refine.1.bias",
    "rgb_0.weight",
    "rgb_0.bias",
    "rgb_1.weight",
    "rgb_1.bias",
]


@dataclass
class TensorBlob:
    name: str
    raw_int8: np.ndarray  # int8 [n], range [-127, 127]
    scale: float
    shape: tuple[int, ...]


def parse_pr107_archive_to_tensors(archive_bin_bytes: bytes) -> tuple[list[TensorBlob], bytes, bytes]:
    """Parse a PR107 0.bin payload, return (per-tensor blobs, meta_section, latents_section).

    The meta + latents sections are returned as their on-disk framed bytes
    (length prefix + brotli payload) so we can re-emit them verbatim.
    """
    buf = io.BytesIO(archive_bin_bytes)
    meta_len = struct.unpack("<I", buf.read(4))[0]
    meta_brotli = buf.read(meta_len)
    meta_section = struct.pack("<I", meta_len) + meta_brotli

    dec_len = struct.unpack("<I", buf.read(4))[0]
    dec_compressed = buf.read(dec_len)

    lat_len = struct.unpack("<I", buf.read(4))[0]
    lat_compressed = buf.read(lat_len)
    latents_section = struct.pack("<I", lat_len) + lat_compressed

    if buf.read():
        raise ValueError("trailing bytes after PR107 archive sections")

    # Decompress decoder CD1 payload to extract per-tensor int8 symbols.
    raw_cd1 = brotli.decompress(dec_compressed)
    cd1 = io.BytesIO(raw_cd1)
    magic = cd1.read(3)
    if magic != b"CD1":
        raise ValueError(f"PR107 decoder is not CD1 (magic={magic!r}); refusing")
    scale_bits = struct.unpack("<B", cd1.read(1))[0]
    if scale_bits not in (16, 32):
        raise ValueError(f"unsupported CD1 scale_bits={scale_bits}")
    n_tensors = struct.unpack("<I", cd1.read(4))[0]
    if n_tensors != len(ARCHITECTURE_ORDER):
        raise ValueError(f"PR107 CD1 n_tensors={n_tensors} != {len(ARCHITECTURE_ORDER)}")

    # Mirror inflate-time loop: per-tensor scale + zigzag bytes (size = numel).
    # Need shapes; we get them from a HNeRVDecoder reference state_dict.
    from model import HNeRVDecoder  # type: ignore  # noqa: E402

    ref = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512)).state_dict()
    blobs: list[TensorBlob] = []
    for name in ARCHITECTURE_ORDER:
        ref_t = ref[name]
        size = ref_t.numel()
        if scale_bits == 16:
            scale = float(np.frombuffer(cd1.read(2), dtype=np.float16)[0])
        else:
            scale = struct.unpack("<f", cd1.read(4))[0]
        zz = np.frombuffer(cd1.read(size), dtype=np.uint8)
        # zigzag-decode back to signed int8
        arr = zz.astype(np.int32)
        q_i8 = np.where(arr % 2 == 0, arr // 2, -(arr // 2) - 1).astype(np.int8)
        blobs.append(
            TensorBlob(
                name=name,
                raw_int8=q_i8,
                scale=scale,
                shape=tuple(int(s) for s in ref_t.shape),
            )
        )
    if cd1.read():
        raise ValueError("trailing bytes in PR107 CD1 payload")
    return blobs, meta_section, latents_section


def find_best_K_for_tensor(symbols_int8: np.ndarray, budget: float) -> tuple[int, float]:
    """Largest K such that per-tensor L1-rel-err <= budget."""
    s = symbols_int8.astype(np.float64)
    abs_sum = float(np.abs(s).sum())
    if abs_sum < 1e-9:
        return 1, 0.0
    best_K = 1
    best_re = 0.0
    for K in range(1, 256):
        rounded = np.round(s / K) * K
        err = float(np.abs(rounded - s).sum())
        re = err / abs_sum
        if re <= budget:
            best_K = K
            best_re = re
        else:
            break
    return best_K, best_re


def build_cd1_payload(blobs: list[TensorBlob], rounded_int8: list[np.ndarray]) -> bytes:
    """Build a CD1 brotli-ready payload from per-tensor (rounded) int8 symbols."""
    out = io.BytesIO()
    out.write(b"CD1")
    out.write(struct.pack("<B", 16))  # fp16 scales
    out.write(struct.pack("<I", len(blobs)))
    for blob, rounded in zip(blobs, rounded_int8, strict=True):
        # Write fp16 scale
        out.write(np.array([blob.scale], dtype=np.float16).tobytes())
        # Zigzag-encode rounded int8 symbols (clamp to [-127,127] safety)
        clamped = rounded.clip(-127, 127).astype(np.int8)
        zz = zigzag_encode_i8(clamped)
        out.write(zz.tobytes())
    return out.getvalue()


def build_archive_bin(meta_section: bytes, decoder_brotli: bytes, latents_section: bytes) -> bytes:
    """Reassemble PR107 0.bin payload."""
    out = io.BytesIO()
    out.write(meta_section)  # already includes its 4-byte length prefix
    out.write(struct.pack("<I", len(decoder_brotli)))
    out.write(decoder_brotli)
    out.write(latents_section)  # already includes its 4-byte length prefix
    return out.getvalue()


def encode_at_budget(blobs: list[TensorBlob], budget: float, brotli_quality: int = 11) -> dict:
    """Apply per-tensor K search at given rel_err budget; report bytes + diagnostics."""
    Ks: list[int] = []
    rounded_chunks: list[np.ndarray] = []
    orig_chunks: list[np.ndarray] = []
    for tb in blobs:
        K, _ = find_best_K_for_tensor(tb.raw_int8, budget)
        Ks.append(K)
        s = tb.raw_int8.astype(np.float64)
        rounded = np.round(s / K) * K
        orig_chunks.append(s)
        rounded_chunks.append(rounded)

    cd1_payload = build_cd1_payload(blobs, rounded_chunks)
    decoder_brotli = brotli.compress(cd1_payload, quality=brotli_quality)
    rel_err = compute_rel_err(
        np.concatenate(rounded_chunks),
        np.concatenate(orig_chunks),
        mode=RelErrForm.L1_RATIO,
    )

    return {
        "budget": budget,
        "Ks": Ks,
        "rel_err": rel_err,
        REL_ERR_FORM_KEY: RelErrForm.L1_RATIO.value,
        "cd1_payload_bytes": len(cd1_payload),
        "decoder_brotli_bytes": len(decoder_brotli),
        "decoder_brotli": decoder_brotli,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/public_pr_intake_full/public_pr107_intake_20260505_auto/archive.zip",
        help="Path to PR107 archive.zip",
    )
    p.add_argument(
        "--budgets",
        type=str,
        default="0.005,0.01,0.02,0.03,0.04,0.05",
        help="Comma-separated per-tensor rel_err budgets to scan.",
    )
    p.add_argument(
        "--build-best",
        action="store_true",
        help="Build a final archive.zip + manifest at the best (rel_err < 0.05, smallest bytes) budget.",
    )
    p.add_argument(
        "--build-budget",
        type=float,
        default=None,
        help="Build a final archive.zip at this exact budget (overrides --build-best selection).",
    )
    p.add_argument(
        "--brotli-quality",
        type=int,
        default=11,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = p.parse_args(argv)

    if not args.archive.is_file():
        raise SystemExit(f"PR107 archive not found: {args.archive}")
    if args.output_dir is None:
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = REPO_ROOT / f"experiments/results/pr107_apogee_lossy_coarsening_{ts}"
    else:
        args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as zf:
        names = zf.namelist()
        if names != ["0.bin"]:
            raise SystemExit(f"unexpected archive members: {names}")
        bin_bytes = zf.read("0.bin")
    if len(bin_bytes) != PR107_BASELINE_BIN_BYTES:
        print(f"WARN: 0.bin bytes={len(bin_bytes)} != baseline {PR107_BASELINE_BIN_BYTES}")

    blobs, meta_section, latents_section = parse_pr107_archive_to_tensors(bin_bytes)
    print(f"[pr107-coarsen] parsed {len(blobs)} tensors")
    print(f"[pr107-coarsen] meta_section: {len(meta_section)} B")
    print(f"[pr107-coarsen] latents_section: {len(latents_section)} B")
    invariant_bytes = len(meta_section) + len(latents_section) + 4  # +4 for decoder length prefix
    print(f"[pr107-coarsen] invariant overhead (meta+latents+dec_len_prefix): {invariant_bytes} B")
    print(f"[pr107-coarsen] PR107 baseline 0.bin: {PR107_BASELINE_BIN_BYTES} B")
    print()

    print("--- per-tensor K coarsening sweep ---")
    print(f"{'budget':>9} {'rel_err':>9} {'CD1_payload':>12} {'dec_brotli':>11} {'0.bin':>8} {'archive_zip':>12} {'delta_bin':>10}")
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    sweep_results = []
    for budget in budgets:
        meas = encode_at_budget(blobs, budget, brotli_quality=args.brotli_quality)
        # Bytes of full reassembled 0.bin
        new_bin_bytes = build_archive_bin(meta_section, meas["decoder_brotli"], latents_section)
        # Compute archive.zip equivalent (zip framing overhead is ~108 B fixed for one 0.bin entry)
        # We will materialize the actual zip later if --build-best.
        delta = len(new_bin_bytes) - PR107_BASELINE_BIN_BYTES
        # Estimate archive.zip size (deterministic ZipInfo with fixed timestamp = same overhead)
        est_zip = len(new_bin_bytes) + (PR107_BASELINE_ARCHIVE_BYTES - PR107_BASELINE_BIN_BYTES)
        print(
            f"{budget:>9.4f} {meas['rel_err']:>9.4f} {meas['cd1_payload_bytes']:>12,} "
            f"{meas['decoder_brotli_bytes']:>11,} {len(new_bin_bytes):>8,} {est_zip:>12,} {delta:>+10,}"
        )
        sweep_results.append({
            "budget": budget,
            "rel_err": meas["rel_err"],
            "Ks": meas["Ks"],
            "cd1_payload_bytes": meas["cd1_payload_bytes"],
            "decoder_brotli_bytes": meas["decoder_brotli_bytes"],
            "bin_bytes": len(new_bin_bytes),
            "estimated_archive_zip_bytes": est_zip,
            "delta_bin_vs_baseline": delta,
        })

    # Decide build target
    target = None
    if args.build_budget is not None:
        target = next((r for r in sweep_results if abs(r["budget"] - args.build_budget) < 1e-12), None)
        if target is None:
            raise SystemExit(f"--build-budget {args.build_budget} not in --budgets")
    elif args.build_best:
        # smallest bytes among results with rel_err < 0.05
        cands = [r for r in sweep_results if r["rel_err"] < 0.05]
        if not cands:
            print("[pr107-coarsen] no candidate within rel_err < 0.05; skipping build")
        else:
            target = min(cands, key=lambda r: r["bin_bytes"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": "[CPU-prep]",
        "evidence_semantics": "cpu_byte_roundtrip_proxy_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": True,  # this tool produces a real archive bound for dispatch
        "dispatch_attempted": False,
        "input_archive_path": str(args.archive),
        "input_archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "input_archive_size_bytes": args.archive.stat().st_size,
        "pr107_baseline_archive_bytes": PR107_BASELINE_ARCHIVE_BYTES,
        "pr107_baseline_bin_bytes": PR107_BASELINE_BIN_BYTES,
        "sweep": sweep_results,
        "target_modes": ["contest_exact_eval"],
        "deployment_target": "linux_x86_64_cpu_and_t4_contest_runtime",
    }

    if target is not None:
        # Materialize the actual archive
        meas = encode_at_budget(blobs, target["budget"], brotli_quality=args.brotli_quality)
        new_bin_bytes = build_archive_bin(meta_section, meas["decoder_brotli"], latents_section)
        # Verify roundtrip: inflate-side parsing must succeed
        decoder_sd_rt, latents_rt, meta_rt = parse_archive(new_bin_bytes)
        assert len(decoder_sd_rt) == 28, f"roundtrip decoder n={len(decoder_sd_rt)}"
        assert latents_rt.shape == (600, 28), f"roundtrip latents shape={latents_rt.shape}"
        assert meta_rt == {"n_pairs": 600, "latent_dim": 28, "base_channels": 36, "eval_size": [384, 512]}

        # Write archive.zip with deterministic ZipInfo (zero timestamp, fixed perm)
        archive_zip = args.output_dir / "archive.zip"
        zi = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = (0o644 & 0xFFFF) << 16
        with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr(zi, new_bin_bytes)
        zip_bytes = archive_zip.stat().st_size

        member_sha = hashlib.sha256(new_bin_bytes).hexdigest()
        archive_sha = hashlib.sha256(archive_zip.read_bytes()).hexdigest()

        manifest.update({
            "build_target_budget": target["budget"],
            "build_rel_err": meas["rel_err"],
            "build_Ks": meas["Ks"],
            "build_archive_relpath": str(archive_zip.resolve().relative_to(REPO_ROOT)),
            "build_archive_sha256": archive_sha,
            "build_archive_size_bytes": zip_bytes,
            "build_member_name": "0.bin",
            "build_member_sha256": member_sha,
            "build_member_size_bytes": len(new_bin_bytes),
            "build_decoder_brotli_bytes": meas["decoder_brotli_bytes"],
            "build_meta_section_bytes": len(meta_section),
            "build_latents_section_bytes": len(latents_section),
            "build_delta_bin_vs_baseline": len(new_bin_bytes) - PR107_BASELINE_BIN_BYTES,
            "build_delta_zip_vs_baseline": zip_bytes - PR107_BASELINE_ARCHIVE_BYTES,
            "score_affecting_payload_changed": True,
            "charged_bits_changed": True,
        })
        print()
        print(f"[pr107-coarsen] BUILT archive: {archive_zip}")
        print(f"[pr107-coarsen]   archive_sha256: {archive_sha}")
        print(f"[pr107-coarsen]   archive_bytes : {zip_bytes:,}")
        print(f"[pr107-coarsen]   member_sha256 : {member_sha}")
        print(f"[pr107-coarsen]   member_bytes  : {len(new_bin_bytes):,}")
        print(f"[pr107-coarsen]   delta_zip     : {zip_bytes - PR107_BASELINE_ARCHIVE_BYTES:+,} B")
        print(f"[pr107-coarsen]   rel_err       : {meas['rel_err']:.4f}")
        print(f"[pr107-coarsen]   per-tensor Ks : {meas['Ks']}")

    (args.output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[pr107-coarsen] manifest: {args.output_dir / 'build_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
