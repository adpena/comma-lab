#!/usr/bin/env python3
"""Build a Wave-Omega Omega-3 block-FP JointFrameGenerator archive.

Pipeline (per blueprint Phase 3):

    1. Load JFG state-dict from the source archive (QZS3, MQZ1, QBF1, BFJ1,
       QFAI, QH0/QM0, or raw torch state_dict renderer member).
    2. Validate every FiLM layer with
       :func:`tac.block_fp_jfg.validate_film_layer_block_fp`. Kill on first
       failure with a clear error message - Contrarian's Phase F gate.
    3. Quantize with :func:`quantize_jfg_block_fp` (FiLM layers protected
       by default - stored raw at FP16).
    4. Compress with :func:`compress_jfg_block_fp` (lzma + HWOI permute).
    5. Repack archive: source archive members but with the renderer
       payload replaced by the BFJ1 binary. Pose / mask members are
       preserved byte-for-byte from the source.
    6. Write ``manifest.json`` with full provenance: source SHA, output SHA,
       byte savings, FiLM validation MSE, and predicted-rate arithmetic tagged
       as non-score evidence.

CUDA policy: pure-CPU only. This codec is byte-deterministic and never
loads SegNet/PoseNet (strict-scorer-rule). No GPU spend.

Score-tagging policy: ALL numerical predictions in the manifest are
labeled ``[predicted]`` - never ``[contest-CUDA]``. The actual contest
score must come from a separate ``inflate.sh`` -> ``upstream/evaluate.py``
run on the EXACT archive bytes produced here.

Usage::

    .venv/bin/python experiments/build_omega3_block_fp_archive.py \\
        --source-archive experiments/results/.../jfg_frontier_archive.zip \\
        --output-dir experiments/results/lane_omega_3_block_fp_jfg_<ts> \\
        --block-size 64 \\
        --validate-film-mse-threshold 1e-3 \\
        --max-bytes 200000
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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.block_fp_jfg import (
    BFJ1_VERSION,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PROTECT_PATTERNS,
    MAGIC_BFJ1,
    BlockFPConfig,
    ValidationResult,
    compress_jfg_block_fp,
    is_film_protected,
    quantize_jfg_block_fp,
    validate_film_layer_block_fp,
)

# Source-archive loaders


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_SUPPORTED_JFG_CONTAINER_MAGICS: tuple[bytes, ...] = (
    b"QZS3",
    b"MQZ1",
    b"QBF1",
    b"BFJ1",
    b"QFAI",
    b"QH0",
    b"QM0",
)


def _looks_like_supported_jfg_container(blob: bytes) -> bool:
    return blob.startswith(_SUPPORTED_JFG_CONTAINER_MAGICS)


def _looks_like_torch_state_dict_container(blob: bytes) -> bool:
    return blob.startswith((b"PK\x03\x04", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"))


def _load_jfg_state_dict_from_blob(
    blob: bytes,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a JFG state dict from a ``.bin``-style blob.

    Supports the JFG formats present in the public frontier tooling:
    QZS3/MQZ1/QBF1/BFJ1/QFAI/QH0/QM0 plus raw torch ``state_dict``. Unknown
    non-pickle magics fail closed instead of falling through to ``torch.load``.

    Returns (state_dict, arch_meta) where ``arch_meta`` records the
    architectural fields encoded in the source container (or defaults).
    """
    if blob.startswith(b"QZS3"):
        from tac.quantizr_qzs3_codec import decode_qzs3_state_dict

        state = decode_qzs3_state_dict(blob, device=device)
        return state, {"container": "QZS3"}
    if blob.startswith(b"MQZ1"):
        from tac.quantizr_qzs3_codec import decode_mixed_qzs_block_state_dict

        state = decode_mixed_qzs_block_state_dict(blob, device=device)
        return state, {"container": "MQZ1"}
    if blob.startswith(b"QBF1"):
        from tac.qbf1_renderer_codec import decode_qbf1_renderer_state_dict

        state = decode_qbf1_renderer_state_dict(blob, device=device)
        return state, {"container": "QBF1"}
    if blob.startswith(b"BFJ1"):
        from tac.block_fp_jfg import decompress_jfg_block_fp

        state = decompress_jfg_block_fp(blob)
        return state, {"container": "BFJ1"}
    if blob.startswith((b"QH0", b"QM0")):
        from tac.qh0_renderer_codec import decode_qh0_state_dict

        state, report = decode_qh0_state_dict(blob, device=device)
        return state, {
            "container": blob[:3].decode("ascii", errors="replace"),
            "decode_report": getattr(report, "__dict__", {}),
        }
    if blob.startswith(b"QFAI"):
        offset = 4
        (header_len,) = struct.unpack("<I", blob[offset:offset + 4])
        offset += 4
        header = json.loads(blob[offset:offset + header_len].decode("utf-8"))
        offset += header_len
        state = torch.load(
            io.BytesIO(blob[offset:]),
            map_location=str(device),
            weights_only=True,
        )
        return state, {
            "container": "QFAI",
            "num_classes": int(header.get("num_classes", 5)),
            "pose_dim": int(header.get("pose_dim", 6)),
            "cond_dim": int(header.get("cond_dim", 48)),
            "depth_mult": int(header.get("depth_mult", 1)),
        }
    if blob[:4] and blob[:1] not in (b"\x80",):
        raise ValueError(
            "unsupported JFG renderer container magic for Omega-3 Block-FP "
            f"source: {blob[:8]!r}. Supported: QZS3, MQZ1, QBF1, BFJ1, "
            "QFAI, QH0, QM0, or raw torch state_dict."
        )
    # Raw torch state_dict
    state = torch.load(
        io.BytesIO(blob),
        map_location=str(device),
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise ValueError(
            f"_load_jfg_state_dict_from_blob: torch.load returned "
            f"{type(state).__name__}, expected dict (state_dict)"
        )
    return state, {"container": "raw_torch_save"}


def _identify_source_renderer_member(zf: zipfile.ZipFile) -> str:
    """Pick a reviewed JFG renderer member by content, not filename.

    PR106/HNeRV archives commonly use ``0.bin`` for a non-JFG packed payload.
    This builder is only for JointFrameGenerator payloads, so filename fallback
    would create a dangerous architecture-family mixup. Raw torch state-dicts
    are accepted only from explicit renderer-like filenames.
    """
    names = zf.namelist()
    preferred = ["renderer.bin", "p", *names]
    for candidate in dict.fromkeys(preferred):
        if candidate not in names:
            continue
        blob = zf.read(candidate)
        if _looks_like_supported_jfg_container(blob):
            return candidate
        if candidate == "renderer.bin" and _looks_like_torch_state_dict_container(blob):
            return candidate
    if not names:
        raise ValueError("source archive contains no members")
    raise ValueError(
        "source archive contains no supported JFG renderer member; expected "
        "QZS3/MQZ1/QBF1/BFJ1/QFAI/QH0/QM0 content, or a raw torch state_dict "
        "in renderer.bin. Refusing filename-only fallback."
    )


def _validate_supported_arch_meta(arch_meta: dict[str, Any]) -> None:
    """Fail closed on source architectures the BFJ1 runtime cannot rebuild."""
    depth_mult = int(arch_meta.get("depth_mult", 1))
    if depth_mult != 1:
        raise ValueError(
            "BFJ1 runtime currently rebuilds JointFrameGenerator with "
            f"depth_mult=1, but source arch_meta has depth_mult={depth_mult}. "
            "Refusing to build an archive that would fail strict load at inflate."
        )


# Manifest


@dataclass
class BuildManifest:
    """Provenance record written alongside the rebuilt archive."""

    timestamp_utc: str
    source_archive_path: str
    source_archive_sha256: str
    source_archive_bytes: int
    source_renderer_member: str
    source_renderer_bytes: int
    output_archive_path: str
    output_archive_sha256: str
    output_archive_bytes: int
    output_renderer_bytes: int
    byte_savings: int
    config: dict[str, Any]
    arch_meta: dict[str, Any]
    film_validation: list[dict[str, Any]]
    layer_count: int
    layer_protected_count: int
    layer_blockfp_count: int
    score_claim: bool = False  # NEVER True; contest-CUDA pipeline only
    score_evidence_grade: str = "predicted"
    predicted_rate_delta: dict[str, Any] | None = None
    notes: list[str] | None = None

    def to_json(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        return json.dumps(d, indent=2, sort_keys=True)


# Main pipeline


def _validate_all_film_layers(
    state_dict: dict[str, torch.Tensor],
    config: BlockFPConfig,
    threshold: float,
) -> list[ValidationResult]:
    """Run :func:`validate_film_layer_block_fp` on every FiLM-flagged layer.

    Raises ``RuntimeError`` on the first kill (Contrarian's gate).
    Returns the list of all ValidationResult records (only-on-success path).
    """
    results: list[ValidationResult] = []
    for name, tensor in state_dict.items():
        if not is_film_protected(name, config.protect_patterns):
            continue
        if not tensor.is_floating_point():
            continue  # int counters, position tables, etc.
        if tensor.numel() == 0:
            continue
        result = validate_film_layer_block_fp(
            tensor, config, layer_name=name, mse_threshold=threshold
        )
        results.append(result)
        if result.kill:
            raise RuntimeError(
                f"FiLM-layer validation FAILED: {name!r} "
                f"roundtrip_mse={result.roundtrip_mse:.6e} > "
                f"threshold={result.threshold:.6e} "
                f"(effective_bpw={result.effective_bpw:.3f}). "
                f"Per Contrarian's Phase F precondition (blueprint Component 3), "
                f"this lane is killed. The FiLM weights are too sensitive for "
                f"block-FP at this block_size."
            )
    return results


def _rebuild_archive(
    source_zip_path: Path,
    source_renderer_member: str,
    new_renderer_bytes: bytes,
    output_zip_path: Path,
) -> None:
    """Repack the source archive: every member byte-for-byte EXCEPT the
    renderer, which is replaced by ``new_renderer_bytes``.

    The output ZIP is written deterministically: members in the source
    archive's ORIGINAL namelist order, with timestamps fixed at the
    UNIX epoch (matches the contest archive convention).
    """
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip_path, "r") as src, zipfile.ZipFile(
        output_zip_path, "w", compression=zipfile.ZIP_STORED
    ) as dst:
        for name in src.namelist():
            payload = (
                new_renderer_bytes if name == source_renderer_member else src.read(name)
            )
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            dst.writestr(info, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Wave-Omega Omega-3 block-FP JointFrameGenerator archive.",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        required=True,
        help="Path to a source archive containing a JFG renderer member "
        "(QZS3, MQZ1, QBF1, BFJ1, QFAI, QH0/QM0, or raw torch state_dict).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the new archive.zip + manifest.json.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=f"Block-FP block_size for non-FiLM layers (default: {DEFAULT_BLOCK_SIZE}).",
    )
    parser.add_argument(
        "--film-block-size",
        type=int,
        default=32,
        help="Block size used inside the FiLM validation gate (default: 32).",
    )
    parser.add_argument(
        "--validate-film-mse-threshold",
        type=float,
        default=1e-3,
        help="Per-FiLM-layer roundtrip MSE kill threshold (default: 1e-3).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=200_000,
        help="Hard cap on output archive bytes; raises if exceeded "
             "(default: 200000).",
    )
    parser.add_argument(
        "--lzma-preset",
        type=int,
        default=9,
        help="lzma compression preset (default: 9, max).",
    )
    parser.add_argument(
        "--no-hwoi-permute",
        action="store_true",
        help="Disable HWOI permute on Conv2d weights (default: enabled).",
    )
    parser.add_argument(
        "--no-protect-film-layers",
        action="store_true",
        help="DANGEROUS: disable FiLM-layer protection. Only use for ablation. "
             "The validation gate still runs first.",
    )
    args = parser.parse_args(argv)

    source_path = args.source_archive.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"--source-archive not found: {source_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_archive_path = output_dir / "archive.zip"
    manifest_path = output_dir / "manifest.json"

    print(
        f"[omega3] reading source archive: {source_path}",
        file=sys.stderr,
    )
    src_sha = _sha256_path(source_path)
    src_bytes = source_path.stat().st_size

    with zipfile.ZipFile(source_path, "r") as zf:
        source_renderer_member = _identify_source_renderer_member(zf)
        renderer_blob = zf.read(source_renderer_member)
    print(
        f"[omega3] source renderer member: {source_renderer_member} "
        f"({len(renderer_blob):,} bytes)",
        file=sys.stderr,
    )

    state_dict, arch_meta = _load_jfg_state_dict_from_blob(
        renderer_blob, device=torch.device("cpu")
    )
    _validate_supported_arch_meta(arch_meta)
    n_params = sum(int(t.numel()) for t in state_dict.values())
    print(
        f"[omega3] JFG state_dict loaded: {len(state_dict)} tensors, "
        f"{n_params:,} params total",
        file=sys.stderr,
    )

    cfg = BlockFPConfig(
        block_size=int(args.block_size),
        protect_film_layers=not args.no_protect_film_layers,
        protect_patterns=DEFAULT_PROTECT_PATTERNS,
        film_block_size=int(args.film_block_size),
        hwoi_permute=not args.no_hwoi_permute,
        lzma_preset=int(args.lzma_preset),
    )

    print("[omega3] validating FiLM layers (Phase F gate)...", file=sys.stderr)
    film_results = _validate_all_film_layers(
        state_dict, cfg, threshold=float(args.validate_film_mse_threshold)
    )
    print(
        f"[omega3] FiLM validation: {len(film_results)} layers PASSED",
        file=sys.stderr,
    )

    print("[omega3] quantizing state_dict (block-FP int8 + per-block exp)...",
          file=sys.stderr)
    quantized = quantize_jfg_block_fp(state_dict, cfg)
    n_protected = sum(1 for q in quantized.values() if q.protected)
    n_blockfp = sum(1 for q in quantized.values() if not q.protected)
    print(
        f"[omega3] quantization done: {n_blockfp} block-FP layers, "
        f"{n_protected} FiLM-protected layers",
        file=sys.stderr,
    )

    print("[omega3] compressing (lzma + HWOI)...", file=sys.stderr)
    new_renderer_bytes = compress_jfg_block_fp(
        quantized,
        block_size=int(args.block_size),
        hwoi_permute=cfg.hwoi_permute,
        lzma_preset=int(args.lzma_preset),
    )
    assert new_renderer_bytes[:4] == MAGIC_BFJ1
    new_renderer_size = len(new_renderer_bytes)
    print(
        f"[omega3] new renderer payload: {new_renderer_size:,} bytes "
        f"(was {len(renderer_blob):,}; "
        f"savings {len(renderer_blob) - new_renderer_size:,})",
        file=sys.stderr,
    )

    print("[omega3] repacking archive...", file=sys.stderr)
    _rebuild_archive(
        source_path,
        source_renderer_member,
        new_renderer_bytes,
        output_archive_path,
    )

    out_sha = _sha256_path(output_archive_path)
    out_bytes = output_archive_path.stat().st_size
    if out_bytes > int(args.max_bytes):
        raise RuntimeError(
            f"output archive {out_bytes} bytes > --max-bytes {args.max_bytes}; "
            f"refusing to write a manifest claiming a saving when the cap is "
            f"violated. Adjust --max-bytes or compress harder."
        )

    # Predicted score arithmetic per blueprint:
    #   rate delta = 25 * byte_savings / 37545489 [predicted, CONDITIONAL]
    byte_savings = int(src_bytes - out_bytes)
    predicted_rate_delta = (
        25.0 * byte_savings / 37_545_489.0 if byte_savings > 0 else 0.0
    )

    manifest = BuildManifest(
        timestamp_utc=_dt.datetime.now(_dt.UTC).isoformat(),
        source_archive_path=str(source_path),
        source_archive_sha256=src_sha,
        source_archive_bytes=src_bytes,
        source_renderer_member=source_renderer_member,
        source_renderer_bytes=len(renderer_blob),
        output_archive_path=str(output_archive_path),
        output_archive_sha256=out_sha,
        output_archive_bytes=out_bytes,
        output_renderer_bytes=new_renderer_size,
        byte_savings=byte_savings,
        config={
            "block_size": cfg.block_size,
            "film_block_size": cfg.film_block_size,
            "hwoi_permute": cfg.hwoi_permute,
            "lzma_preset": cfg.lzma_preset,
            "protect_film_layers": cfg.protect_film_layers,
            "protect_patterns": list(cfg.protect_patterns),
            "validate_film_mse_threshold": float(args.validate_film_mse_threshold),
            "bfj1_magic": MAGIC_BFJ1.decode("ascii"),
            "bfj1_version": BFJ1_VERSION,
        },
        arch_meta=arch_meta,
        film_validation=[
            {
                "layer_name": r.layer_name,
                "roundtrip_mse": float(r.roundtrip_mse),
                "threshold": float(r.threshold),
                "passed": bool(r.passed),
                "effective_bpw": float(r.effective_bpw),
            }
            for r in film_results
        ],
        layer_count=len(quantized),
        layer_protected_count=n_protected,
        layer_blockfp_count=n_blockfp,
        score_claim=False,  # NEVER True
        score_evidence_grade="predicted",
        predicted_rate_delta={
            "magnitude": float(predicted_rate_delta),
            "tag": "[predicted]",
            "derivation": (
                "25 * byte_savings / 37545489 - conditional on FiLM "
                "compressibility (Contrarian's caveat)."
            ),
        },
        notes=[
            "Wave-Omega Omega-3 block-FP JFG transplant (Track F).",
            "Strict-scorer-rule: codec is pure CPU byte<->tensor math.",
            "Score must come from contest-CUDA inflate.sh + evaluate.py "
            "on the EXACT archive bytes; this manifest claims none.",
        ],
    )
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
    print(
        f"[omega3] manifest written: {manifest_path}",
        file=sys.stderr,
    )
    print(
        f"[omega3] DONE. archive={output_archive_path} "
        f"({out_bytes:,} bytes, sha256={out_sha[:12]}...)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
