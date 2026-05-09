#!/usr/bin/env python3
"""Track 4 from Fields-medal Grand Council (commit 9b44c2f6): UNIWARD + STC +
Hessian-aware bit allocation on the EXISTING A1 latent-aligned archive.

NO retraining. Operates purely on the byte-domain decoder state_dict extracted
from A1's archive, applies a per-tensor Hessian-weighted lossy coarsening, and
re-packs via the canonical PR101 split-Brotli codec.

Mathematical framework
----------------------
Fridrich-Yousfi UNIWARD insight: perturbations in high-entropy/textured regions
are imperceptible to convolutional detectors (here SegNet/PoseNet). Translated
to the parameter domain: low-Fisher-information parameters (small d²L/dθ²) are
"textured" — they accept distortion without scoring penalty.

We use the same Cramer-Rao shape as the council solver's Fisher allocator,
but with an important archive-level correction: each tensor's bit width is
charged by its parameter count. Tensors below the cutoff get re-quantized at
lower n_quant (sub-int8 codes), shrinking their Brotli-coded entropy.

The "STC" component here is still only a lane label inherited from the council
proposal. This v1 builder does not implement an explicit syndrome-trellis
codec; it emits the PR101 split-Brotli wire format after sensitivity-weighted
coarsening. Any future explicit STC/rANS/FSE stage must prove a changed payload
and byte win against this baseline before promotion.

Fisher proxy
------------
True Fisher = E[(∂L/∂θ)²]. Without retraining, we use the per-tensor diagonal
Hessian-trace proxy `mean(θ²) * scale²` (parameter-magnitude squared, an
empirical Bayes prior on saliency). This is the same proxy used by
`tools/theoretical_floor_solver_v2.py::a1_floor_gap_decomposition`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import struct
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np
import torch

from tac.pr101_split_brotli_codec import (
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    decode_decoder_compact,
    encode_decoder_compact,
)


# Canonical A1 latent-aligned anchor (Council 9b44c2f6 + commit 13e8e08c).
A1_ARCHIVE_PATH_DEFAULT = (
    "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
    "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/archive.zip"
)
A1_ARCHIVE_BYTES_EXPECTED = 178_262
A1_ARCHIVE_SHA256_EXPECTED = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_SUBMISSION_DIR_DEFAULT = (
    "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_"
    "20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir"
)


@dataclass(frozen=True)
class TensorBitAlloc:
    name: str
    n_params: int
    fisher_proxy: float
    bits_target: int
    n_quant_effective: int
    rel_err_post_coarsening: float
    bytes_brotli_post: int


@dataclass
class BuildResult:
    src_archive_bytes: int
    src_archive_sha256: str
    new_archive_bytes: int
    new_archive_sha256: str
    delta_bytes: int
    n_tensors: int
    src_inner_sha256: str
    new_inner_sha256: str
    src_decoder_blob_bytes: int
    new_decoder_blob_bytes: int
    src_decoder_blob_sha256: str
    new_decoder_blob_sha256: str
    score_affecting_payload_changed: bool
    fisher_proxy_total: float
    bit_alloc: list[TensorBitAlloc]
    codec_roundtrip_max_rel_err: float
    distortion_max_rel_err_vs_source: float
    distortion_global_l1_rel_err_vs_source: float
    runtime_packet_complete: bool
    advisory_score: float | None
    advisory_score_tag: str | None
    predicted_band: tuple[float, float]
    council_floor: float
    a1_baseline_cpu: float
    council_predicted_band: tuple[float, float]


def _read_a1_inner_blob(archive: Path) -> bytes:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if "x" not in names:
            raise SystemExit(f"FATAL: A1 archive {archive} missing inner 'x'; got {names!r}")
        return zf.read("x")


def _split_a1_inner_blob(inner: bytes) -> tuple[bytes, bytes, bytes]:
    """A1 latent-aligned no-dead-K wire format: [uint32 section_total][decoder_blob][latent_blob][sidecar]."""
    if len(inner) < 4:
        raise SystemExit("FATAL: inner blob too short for no-dead-K header")
    section_total = struct.unpack_from("<I", inner, 0)[0]
    if section_total < 4 or section_total > len(inner):
        raise SystemExit(f"FATAL: bad section_total={section_total} for inner_len={len(inner)}")
    decoder_blob = inner[4:section_total]
    latent_blob = inner[section_total : section_total + LATENT_BLOB_LEN]
    sidecar_blob = inner[section_total + LATENT_BLOB_LEN :]
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: A1 latent_blob len={len(latent_blob)} != expected {LATENT_BLOB_LEN}"
        )
    return decoder_blob, latent_blob, sidecar_blob


def compute_fisher_proxy(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Per-tensor Fisher diagonal proxy = mean(θ²).

    This is the empirical-Bayes saliency proxy that `theoretical_floor_solver_v2.
    a1_floor_gap_decomposition` uses. Without backprop access, mean(θ²) is the
    canonical no-data approximation to E[(∂L/∂θ)²] when score-gradient covariance
    is unknown (van Trees, "Detection, Estimation, and Modulation Theory" §2.4).
    """
    proxy: dict[str, float] = {}
    for name, t in state_dict.items():
        t_f = t.detach().float()
        proxy[name] = float((t_f * t_f).mean().item())
    return proxy


def coarsen_tensor_to_bits(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Lossy re-quantize a tensor to ``bits``-bit per-tensor symmetric INT.

    Returns a float32 tensor at the coarsened resolution. ``bits`` ∈ [2, 8].
    bits=8 → returns tensor unchanged (already at PR101's INT8 rep).
    bits<8 → effective n_quant = (1 << (bits-1)) - 1, i.e. signed magnitude
    quantization with the SAME scale-factor approach PR101 uses internally.
    """
    if bits >= 8:
        return tensor.detach().float().clone()
    if bits < 2:
        raise ValueError(f"bits={bits} below floor (2); refusing to zero-out tensor")
    n_quant_eff = (1 << (bits - 1)) - 1  # bits=4 → 7, bits=6 → 31, bits=2 → 1
    t = tensor.detach().float()
    abs_max = float(t.abs().max().item())
    scale = abs_max / n_quant_eff if abs_max > 0 else 1.0
    q = (t / scale).round().clamp(-n_quant_eff, n_quant_eff)
    return q * scale


def allocate_bits_per_tensor(
    state_dict: dict[str, torch.Tensor],
    fisher_proxy: dict[str, float],
    *,
    target_decoder_bytes: int,
    source_decoder_bytes: int,
    floor_bits: int = 4,
    ceiling_bits: int = 8,
) -> dict[str, int]:
    """Allocate integer per-tensor bit widths under a parameter-weighted budget.

    ``theoretical_floor_solver_v2.fisher_information_bit_allocation`` allocates
    per-parameter bit counts. This builder has only one saliency scalar per
    tensor, so the budget must be charged by ``numel(tensor)``. The previous
    draft accidentally allocated over 28 tensors, not 228K parameters, and then
    clamped everything back to 8 bits. This function makes the archive target
    load-bearing.

    ``target_decoder_bytes`` is a compressed Brotli target, not raw symbol bits.
    The source decoder's compressed byte count is therefore calibrated to the
    current 8-bit tensor grid; requested byte cuts scale that 8-bit grid down.
    """
    if floor_bits < 2:
        raise ValueError(f"floor_bits={floor_bits} below supported minimum 2")
    if ceiling_bits < floor_bits:
        raise ValueError(
            f"ceiling_bits={ceiling_bits} must be >= floor_bits={floor_bits}"
        )
    if target_decoder_bytes <= 0:
        raise ValueError(f"target_decoder_bytes must be positive, got {target_decoder_bytes}")
    if source_decoder_bytes <= 0:
        raise ValueError(f"source_decoder_bytes must be positive, got {source_decoder_bytes}")

    names = list(state_dict.keys())
    n_params = {name: int(state_dict[name].numel()) for name in names}
    total_params = sum(n_params.values())
    floor_total = total_params * int(floor_bits)
    ceiling_total = total_params * int(ceiling_bits)
    target_total = int(round(ceiling_total * (target_decoder_bytes / source_decoder_bytes)))
    target_total = max(floor_total, min(ceiling_total, target_total))

    bits = {name: int(floor_bits) for name in names}
    remaining = target_total - floor_total
    if remaining <= 0:
        return bits

    eps = 1e-12
    log_f = {name: 0.5 * math.log2(max(float(fisher_proxy[name]), eps)) for name in names}
    min_log = min(log_f.values())
    # Add a small base weight so low-Fisher tensors are not permanently starved.
    weights = {name: (log_f[name] - min_log) + 0.10 for name in names}
    denom = sum(weights[name] * n_params[name] for name in names)
    raw_bits: dict[str, float] = {}
    for name in names:
        extra = remaining * weights[name] / denom if denom > 0 else remaining / total_params
        raw_bits[name] = max(float(floor_bits), min(float(ceiling_bits), floor_bits + extra))
        bits[name] = int(math.floor(raw_bits[name]))

    used = sum(n_params[name] * bits[name] for name in names)

    def add_order() -> list[str]:
        return sorted(
            (name for name in names if bits[name] < ceiling_bits),
            key=lambda name: (
                raw_bits[name] - bits[name],
                weights[name],
                -n_params[name],
                name,
            ),
            reverse=True,
        )

    progressed = True
    while used < target_total and progressed:
        progressed = False
        for name in add_order():
            cost = n_params[name]
            if used + cost <= target_total and bits[name] < ceiling_bits:
                bits[name] += 1
                used += cost
                progressed = True

    return bits


def parse_manual_bit_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse ``NAME=BITS`` overrides for one-tensor trust-region probes."""
    overrides: dict[str, int] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"manual bit override must be NAME=BITS, got {raw!r}")
        name, bits_s = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"manual bit override has empty tensor name: {raw!r}")
        try:
            bits = int(bits_s)
        except ValueError as exc:
            raise ValueError(f"manual bit override has non-integer bits: {raw!r}") from exc
        overrides[name] = bits
    return overrides


def build(
    *,
    src_archive: Path,
    source_submission_dir: Path | None,
    out_dir: Path,
    target_bytes: int,
    floor_bits: int,
    ceiling_bits: int,
    manual_bit_overrides: dict[str, int] | None = None,
    brotli_quality: int = 11,
    require_sha: bool = True,
) -> BuildResult:
    """Construct a Track 4 refined archive from A1's latent-aligned A1 archive."""
    src_bytes = src_archive.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    if require_sha and src_sha != A1_ARCHIVE_SHA256_EXPECTED:
        raise SystemExit(
            f"FATAL: A1 archive SHA mismatch.\n"
            f"  expected {A1_ARCHIVE_SHA256_EXPECTED}\n"
            f"  got      {src_sha}\n"
            f"Pass --no-require-sha to override (NOT recommended for council-anchored runs)."
        )

    inner = _read_a1_inner_blob(src_archive)
    decoder_blob, latent_blob, sidecar_blob = _split_a1_inner_blob(inner)
    src_inner_sha = hashlib.sha256(inner).hexdigest()
    src_decoder_sha = hashlib.sha256(decoder_blob).hexdigest()

    # Decode A1 decoder state_dict (28 INT8-quantized tensors, dequantized to float32).
    sd_orig = decode_decoder_compact(decoder_blob)
    fisher_proxy = compute_fisher_proxy(sd_orig)
    manual_bit_overrides = manual_bit_overrides or {}
    unknown_overrides = sorted(set(manual_bit_overrides) - set(sd_orig))
    if unknown_overrides:
        raise SystemExit(
            "FATAL: manual bit override references unknown tensor(s): "
            + ", ".join(unknown_overrides)
        )
    for name, bits in manual_bit_overrides.items():
        if bits < floor_bits or bits > ceiling_bits:
            raise SystemExit(
                f"FATAL: manual bit override {name}={bits} outside "
                f"[{floor_bits}, {ceiling_bits}]"
            )
    if manual_bit_overrides:
        bits_per_tensor = {name: int(ceiling_bits) for name in sd_orig}
        bits_per_tensor.update(manual_bit_overrides)
    else:
        target_decoder_bytes = max(
            1,
            len(decoder_blob) + (int(target_bytes) - len(src_bytes)),
        )
        bits_per_tensor = allocate_bits_per_tensor(
            sd_orig,
            fisher_proxy,
            target_decoder_bytes=target_decoder_bytes,
            source_decoder_bytes=len(decoder_blob),
            floor_bits=floor_bits,
            ceiling_bits=ceiling_bits,
        )

    # Apply per-tensor lossy coarsening (UNIWARD-weighted distortion).
    sd_coarse: dict[str, torch.Tensor] = {}
    rel_errs: dict[str, float] = {}
    for name, t in sd_orig.items():
        bits = bits_per_tensor[name]
        coarse = coarsen_tensor_to_bits(t, bits)
        diff = (t - coarse).abs()
        denom = max(float(t.abs().max().item()), 1e-12)
        rel_errs[name] = float(diff.max().item()) / denom
        sd_coarse[name] = coarse

    # Re-encode via PR101 split-Brotli (canonical wire-format compatibility).
    new_decoder_blob = encode_decoder_compact(sd_coarse, brotli_quality=brotli_quality)
    new_decoder_sha = hashlib.sha256(new_decoder_blob).hexdigest()

    # Re-pack into A1's no-dead-K wire format: [uint32][decoder][latent][sidecar].
    section_total = 4 + len(new_decoder_blob)
    new_inner = (
        struct.pack("<I", section_total)
        + new_decoder_blob
        + latent_blob
        + sidecar_blob
    )
    new_inner_sha = hashlib.sha256(new_inner).hexdigest()

    out_dir.mkdir(parents=True, exist_ok=True)
    new_archive = out_dir / "archive.zip"
    with zipfile.ZipFile(new_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("x")
        info.compress_type = zipfile.ZIP_STORED
        info.date_time = (1980, 1, 1, 0, 0, 0)
        zf.writestr(info, new_inner)
    runtime_packet_complete = False
    if source_submission_dir is not None:
        if not source_submission_dir.is_dir():
            raise SystemExit(f"FATAL: source submission dir not found: {source_submission_dir}")
        packet_submission_dir = out_dir / "submission_dir"
        if packet_submission_dir.exists():
            shutil.rmtree(packet_submission_dir)
        shutil.copytree(source_submission_dir, packet_submission_dir)
        runtime_packet_complete = (packet_submission_dir / "inflate.sh").is_file()
    new_bytes_total = new_archive.stat().st_size
    new_sha = hashlib.sha256(new_archive.read_bytes()).hexdigest()

    # Smoke roundtrip via decode_decoder_compact.
    sd_smoke = decode_decoder_compact(new_decoder_blob)
    smoke_rel: dict[str, float] = {}
    for name, t_coarse in sd_coarse.items():
        t_smoke = sd_smoke[name]
        diff = (t_coarse - t_smoke).abs()
        denom = max(float(t_coarse.abs().max().item()), 1e-12)
        smoke_rel[name] = float(diff.max().item()) / denom
    smoke_max = max(smoke_rel.values()) if smoke_rel else 0.0

    abs_err_total = 0.0
    abs_orig_total = 0.0
    for name, t_orig in sd_orig.items():
        t_new = sd_smoke[name]
        diff = (t_orig - t_new).abs()
        abs_err_total += float(diff.sum().item())
        abs_orig_total += float(t_orig.abs().sum().item())
        denom = max(float(t_orig.abs().max().item()), 1e-12)
        rel_errs[name] = max(rel_errs[name], float(diff.max().item()) / denom)
    distortion_max = max(rel_errs.values()) if rel_errs else 0.0
    distortion_l1 = abs_err_total / abs_orig_total if abs_orig_total > 1e-12 else 0.0

    # Per-tensor brotli post sizes (approx).
    bit_alloc_records: list[TensorBitAlloc] = []
    for name in sd_orig:
        bits = bits_per_tensor[name]
        n_quant_eff = (1 << (bits - 1)) - 1 if bits < 8 else 127
        bit_alloc_records.append(
            TensorBitAlloc(
                name=name,
                n_params=int(sd_orig[name].numel()),
                fisher_proxy=fisher_proxy[name],
                bits_target=bits,
                n_quant_effective=n_quant_eff,
                rel_err_post_coarsening=rel_errs[name],
                bytes_brotli_post=0,  # not measured per-tensor in this v1
            )
        )

    # Council prediction + a1 baseline.
    council_floor = 0.140
    a1_cpu_baseline = 0.192847577437
    council_predicted_band = (0.173, 0.188)  # Track 4 (commit 9b44c2f6)

    return BuildResult(
        src_archive_bytes=len(src_bytes),
        src_archive_sha256=src_sha,
        new_archive_bytes=new_bytes_total,
        new_archive_sha256=new_sha,
        delta_bytes=new_bytes_total - len(src_bytes),
        n_tensors=len(sd_orig),
        src_inner_sha256=src_inner_sha,
        new_inner_sha256=new_inner_sha,
        src_decoder_blob_bytes=len(decoder_blob),
        new_decoder_blob_bytes=len(new_decoder_blob),
        src_decoder_blob_sha256=src_decoder_sha,
        new_decoder_blob_sha256=new_decoder_sha,
        score_affecting_payload_changed=new_inner_sha != src_inner_sha,
        fisher_proxy_total=float(sum(fisher_proxy.values())),
        bit_alloc=bit_alloc_records,
        codec_roundtrip_max_rel_err=smoke_max,
        distortion_max_rel_err_vs_source=distortion_max,
        distortion_global_l1_rel_err_vs_source=distortion_l1,
        runtime_packet_complete=runtime_packet_complete,
        advisory_score=None,
        advisory_score_tag=None,
        predicted_band=council_predicted_band,
        council_floor=council_floor,
        a1_baseline_cpu=a1_cpu_baseline,
        council_predicted_band=council_predicted_band,
    )


def write_manifest(out_dir: Path, result: BuildResult, *, params: dict) -> Path:
    """Emit build_manifest.json with full provenance + bit-alloc table."""
    manifest = {
        "lane_id": "track1_paradigm_delta_track4_uniward_stc_hessian_a1",
        "schema_version": "track4_uniward_stc_hessian_a1_v1",
        "completed_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "council_provenance": {
            "verdict_commit": "9b44c2f6",
            "track": "Track 4",
            "predicted_band": list(result.council_predicted_band),
            "theoretical_floor": result.council_floor,
            "a1_baseline_cpu": result.a1_baseline_cpu,
        },
        "src_archive": {
            "bytes": result.src_archive_bytes,
            "sha256": result.src_archive_sha256,
        },
        "new_archive": {
            "bytes": result.new_archive_bytes,
            "sha256": result.new_archive_sha256,
        },
        "delta_bytes": result.delta_bytes,
        "n_tensors": result.n_tensors,
        "src_inner_sha256": result.src_inner_sha256,
        "new_inner_sha256": result.new_inner_sha256,
        "src_decoder_blob_bytes": result.src_decoder_blob_bytes,
        "new_decoder_blob_bytes": result.new_decoder_blob_bytes,
        "src_decoder_blob_sha256": result.src_decoder_blob_sha256,
        "new_decoder_blob_sha256": result.new_decoder_blob_sha256,
        "score_affecting_payload_changed": result.score_affecting_payload_changed,
        "fisher_proxy_total": result.fisher_proxy_total,
        "codec_roundtrip_max_rel_err": result.codec_roundtrip_max_rel_err,
        "distortion_max_rel_err_vs_source": result.distortion_max_rel_err_vs_source,
        "distortion_global_l1_rel_err_vs_source": result.distortion_global_l1_rel_err_vs_source,
        "runtime_packet_complete": result.runtime_packet_complete,
        "advisory_score": result.advisory_score,
        "advisory_score_tag": result.advisory_score_tag,
        "predicted_band": list(result.predicted_band),
        "params": params,
        "allocator_mode": (
            "manual_bit_overrides" if params.get("manual_bit_overrides") else "target_decoder_bytes"
        ),
        "bit_alloc": [asdict(r) for r in result.bit_alloc],
        "evidence_grade": "[byte-anchor; Fisher-weighted lossy coarsening; smoke-roundtrip verified]",
        "score_claim": False,  # advisory only until contest-CPU/CUDA eval lands
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": (
            result.runtime_packet_complete
            and result.score_affecting_payload_changed
            and result.codec_roundtrip_max_rel_err < 0.01
        ),
        "dispatch_blockers": [
            blocker
            for blocker, present in (
                ("runtime_packet_missing_submission_dir", not result.runtime_packet_complete),
                ("no_score_affecting_payload_change", not result.score_affecting_payload_changed),
                ("codec_roundtrip_max_rel_err_above_0.01", result.codec_roundtrip_max_rel_err >= 0.01),
                ("contest_cpu_eval_pending", True),
                ("contest_cuda_eval_pending", True),
            )
            if present
        ],
    }
    path = out_dir / "build_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Track 4: UNIWARD + STC + Hessian-aware bit allocation on A1"
    )
    p.add_argument("--src-archive", type=Path, default=REPO_ROOT / A1_ARCHIVE_PATH_DEFAULT)
    p.add_argument(
        "--source-submission-dir",
        type=Path,
        default=REPO_ROOT / A1_SUBMISSION_DIR_DEFAULT,
        help="A1 runtime submission_dir to copy beside the candidate archive.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--target-bytes", type=int, default=160_000)
    p.add_argument("--floor-bits", type=int, default=4)
    p.add_argument("--ceiling-bits", type=int, default=8)
    p.add_argument("--brotli-quality", type=int, default=11)
    p.add_argument(
        "--set-bits",
        action="append",
        default=[],
        metavar="NAME=BITS",
        help=(
            "Manual one-tensor override. May be repeated. When supplied, all "
            "other tensors stay at --ceiling-bits and --target-bytes is "
            "metadata only."
        ),
    )
    p.add_argument(
        "--no-require-sha",
        dest="require_sha",
        action="store_false",
        help="Skip A1 archive SHA verification (NOT recommended)",
    )
    args = p.parse_args(argv)

    if not args.src_archive.is_file():
        raise SystemExit(f"FATAL: --src-archive not found: {args.src_archive}")
    try:
        manual_bit_overrides = parse_manual_bit_overrides(args.set_bits)
    except ValueError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc

    result = build(
        src_archive=args.src_archive,
        source_submission_dir=args.source_submission_dir,
        out_dir=args.out_dir,
        target_bytes=args.target_bytes,
        floor_bits=args.floor_bits,
        ceiling_bits=args.ceiling_bits,
        manual_bit_overrides=manual_bit_overrides,
        brotli_quality=args.brotli_quality,
        require_sha=args.require_sha,
    )
    params = {
        "target_bytes": args.target_bytes,
        "floor_bits": args.floor_bits,
        "ceiling_bits": args.ceiling_bits,
        "brotli_quality": args.brotli_quality,
        "require_sha": args.require_sha,
        "source_submission_dir": str(args.source_submission_dir),
        "manual_bit_overrides": manual_bit_overrides,
    }
    manifest_path = write_manifest(args.out_dir, result, params=params)
    print(f"[track4] wrote {manifest_path}")
    print(
        f"[track4] src={result.src_archive_bytes} B -> new={result.new_archive_bytes} B "
        f"(delta={result.delta_bytes:+d} B); codec_roundtrip_max_rel_err={result.codec_roundtrip_max_rel_err:.4e}; "
        f"distortion_l1_rel={result.distortion_global_l1_rel_err_vs_source:.4e}"
    )
    print(f"[track4] new_sha256={result.new_archive_sha256}")
    print(
        f"[track4] council prediction: {result.council_predicted_band} (floor={result.council_floor}, "
        f"A1 baseline CPU={result.a1_baseline_cpu})"
    )


if __name__ == "__main__":
    main()
