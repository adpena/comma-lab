#!/usr/bin/env python3
# pyc-recovery pass2: rehydrated from git blob 7e386d0dd7d56ecd341f4067575eef52bd5a6148 via `git fsck --lost-found`
# original path: experiments/profile_pr95_hnerv_muon_intake.py
# This is OUR source, dropped during commit 66c59aae filter-repo cleanup; the .pyc was the only
# orphan left behind. Original blob SHA verified intact.
# Recovered: 2026-05-05 by Sherlock pass2
"""Static PR95 HNeRV/Muon intake profiler.

This tool performs local byte/source accounting only. It does not run the
contest scorer, does not load scorer models, and does not dispatch GPU work.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr95_intake_20260504_codex"
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_RELEASE_VIEW_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto"
)
DEFAULT_SOURCE_DIR = DEFAULT_RELEASE_VIEW_DIR / "source/submissions/hnerv_muon"
DEFAULT_STATIC_INTAKE = DEFAULT_INTAKE_DIR / "pr95_static_intake.json"
DEFAULT_JSON_OUT = DEFAULT_INTAKE_DIR / "profile_pr95_hnerv_muon_intake.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_INTAKE_DIR / "profile_pr95_hnerv_muon_intake.md"
CONTEST_ORIGINAL_BYTES = 37_545_489
SCHEMA = "pr95_hnerv_muon_static_intake_profile_v1"
TOOL = "experiments/profile_pr95_hnerv_muon_intake.py"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_u32(raw: bytes, cursor: int, label: str) -> tuple[int, int]:
    if cursor + 4 > len(raw):
        raise ValueError(f"truncated u32 while reading {label}")
    return struct.unpack_from("<I", raw, cursor)[0], cursor + 4


def _product(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def _require_brotli():
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("brotli is required for PR95 HNeRV/Muon blob intake") from exc
    return brotli


def read_pr95_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read and validate the public PR95 single-member archive."""
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate ZIP member names are forbidden: {names!r}")
        if names != ["0.bin"]:
            raise ValueError(f"PR95 HNeRV/Muon archive must contain exactly ['0.bin']; got {names!r}")
        info = infos[0]
        parts = Path(info.filename).parts
        if info.filename.startswith("/") or ".." in parts:
            raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
        payload = zf.read(info)
    archive_bytes = path.stat().st_size
    return (
        {
            "path": _rel(path),
            "archive_bytes": archive_bytes,
            "archive_sha256": _sha256_file(path),
            "member_count": len(infos),
            "members": [
                {
                    "name": info.filename,
                    "file_size": int(info.file_size),
                    "compress_size": int(info.compress_size),
                    "compress_type": int(info.compress_type),
                    "zip_stored": info.compress_type == zipfile.ZIP_STORED,
                    "crc32_hex": f"{info.CRC:08x}",
                    "date_time": list(info.date_time),
                    "sha256": _sha256_bytes(payload),
                }
            ],
            "zip_overhead_bytes": archive_bytes - int(info.compress_size),
            "rate_score_component": 25.0 * archive_bytes / CONTEST_ORIGINAL_BYTES,
        },
        payload,
    )


def parse_decoder_blob(blob: bytes) -> dict[str, Any]:
    """Parse ``codec.encode_decoder`` output after brotli decompression."""
    cursor = 0
    tensor_count, cursor = _read_u32(blob, cursor, "decoder tensor count")
    tensors: list[dict[str, Any]] = []
    total_params = 0
    muon_params = 0
    adamw_decoder_params = 0
    groups: dict[str, int] = {}

    for index in range(tensor_count):
        name_len, cursor = _read_u32(blob, cursor, f"tensor {index} name length")
        if cursor + name_len > len(blob):
            raise ValueError(f"truncated tensor name at index {index}")
        name = blob[cursor : cursor + name_len].decode("utf-8")
        cursor += name_len
        ndim, cursor = _read_u32(blob, cursor, f"{name} ndim")
        shape_values: list[int] = []
        for _ in range(ndim):
            value, cursor = _read_u32(blob, cursor, f"{name} shape")
            shape_values.append(value)
        if cursor + 4 > len(blob):
            raise ValueError(f"truncated scale for tensor {name}")
        scale = struct.unpack_from("<f", blob, cursor)[0]
        cursor += 4
        size, cursor = _read_u32(blob, cursor, f"{name} quantized size")
        if cursor + size > len(blob):
            raise ValueError(f"truncated quantized payload for tensor {name}")
        qbytes = blob[cursor : cursor + size]
        cursor += size

        shape = tuple(shape_values)
        params = _product(shape)
        if params != size:
            raise ValueError(f"tensor {name} shape product {params} != stored size {size}")
        lower = name.lower()
        is_muon = ndim >= 2 and "stem" not in lower and not lower.startswith("rgb") and ".rgb_" not in lower
        if is_muon:
            muon_params += params
            optimizer_partition = "muon_hidden_2d_plus_weight"
        else:
            adamw_decoder_params += params
            optimizer_partition = "adamw_decoder_or_bias"
        group = name.split(".", 1)[0]
        groups[group] = groups.get(group, 0) + params
        total_params += params
        tensors.append(
            {
                "name": name,
                "shape": list(shape),
                "params": params,
                "quantized_bytes": size,
                "scale": scale,
                "zigzag_byte_min": min(qbytes) if qbytes else None,
                "zigzag_byte_max": max(qbytes) if qbytes else None,
                "optimizer_partition": optimizer_partition,
            }
        )

    if cursor != len(blob):
        raise ValueError(f"decoder blob has trailing bytes: {len(blob) - cursor}")

    return {
        "tensor_count": tensor_count,
        "total_params": total_params,
        "muon_partition_params": muon_params,
        "adamw_decoder_partition_params": adamw_decoder_params,
        "parameter_groups": dict(sorted(groups.items())),
        "top_tensors_by_params": sorted(tensors, key=lambda item: item["params"], reverse=True)[:12],
        "tensors": tensors,
        "raw_decoder_table_bytes": len(blob),
    }


def parse_latents_payload(raw: bytes) -> dict[str, Any]:
    """Parse ``codec.encode_latents`` output after brotli decompression."""
    cursor = 0
    n_pairs, cursor = _read_u32(raw, cursor, "latent pair count")
    latent_dim, cursor = _read_u32(raw, cursor, "latent dim")
    fp16_table_bytes = latent_dim * 2
    mins_start = cursor
    cursor += fp16_table_bytes
    scales_start = cursor
    cursor += fp16_table_bytes
    if cursor > len(raw):
        raise ValueError("truncated latent min/scale tables")
    total = n_pairs * latent_dim
    cursor += total
    hi_start = cursor
    cursor += total
    if cursor != len(raw):
        raise ValueError(f"latent payload accounting mismatch: cursor={cursor}, len={len(raw)}")

    mins = struct.unpack("<" + "e" * latent_dim, raw[mins_start : mins_start + fp16_table_bytes])
    scales = struct.unpack("<" + "e" * latent_dim, raw[scales_start : scales_start + fp16_table_bytes])
    hi = raw[hi_start : hi_start + total]
    return {
        "n_frame_pairs": n_pairs,
        "latent_dim": latent_dim,
        "latent_values": total,
        "payload_bytes": len(raw),
        "header_bytes": 8,
        "mins_fp16_bytes": fp16_table_bytes,
        "scales_fp16_bytes": fp16_table_bytes,
        "lo_delta_bytes": total,
        "hi_delta_bytes": total,
        "hi_nonzero_count": sum(1 for value in hi if value),
        "hi_nonzero_fraction": (sum(1 for value in hi if value) / total) if total else 0.0,
        "mins_range": [float(min(mins)), float(max(mins))] if mins else [None, None],
        "scales_range": [float(min(scales)), float(max(scales))] if scales else [None, None],
    }


def parse_hnerv_muon_member(payload: bytes) -> dict[str, Any]:
    """Parse PR95 ``0.bin`` into meta, decoder, and latent sections."""
    brotli = _require_brotli()
    cursor = 0
    sections: list[dict[str, Any]] = []

    meta_len, cursor = _read_u32(payload, cursor, "meta brotli length")
    meta_brotli_start = cursor
    meta_brotli = payload[cursor : cursor + meta_len]
    cursor += meta_len
    if cursor > len(payload):
        raise ValueError("truncated meta section")
    meta = json.loads(brotli.decompress(meta_brotli))
    sections.append(
        {
            "name": "meta_json_brotli",
            "length_prefix_offset": 0,
            "compressed_bytes": meta_len,
            "uncompressed_bytes": len(json.dumps(meta, sort_keys=True).encode("utf-8")),
            "sha256": _sha256_bytes(meta_brotli),
        }
    )

    decoder_len_offset = cursor
    decoder_len, cursor = _read_u32(payload, cursor, "decoder blob length")
    decoder_blob = payload[cursor : cursor + decoder_len]
    cursor += decoder_len
    if cursor > len(payload):
        raise ValueError("truncated decoder section")
    decoder_raw = brotli.decompress(decoder_blob)
    decoder = parse_decoder_blob(decoder_raw)
    sections.append(
        {
            "name": "decoder_state_int8_brotli",
            "length_prefix_offset": decoder_len_offset,
            "compressed_bytes": decoder_len,
            "uncompressed_bytes": len(decoder_raw),
            "sha256": _sha256_bytes(decoder_blob),
        }
    )

    latents_len_offset = cursor
    latents_len, cursor = _read_u32(payload, cursor, "latents brotli length")
    latents_blob = payload[cursor : cursor + latents_len]
    cursor += latents_len
    if cursor != len(payload):
        raise ValueError(f"0.bin has trailing bytes or truncation: cursor={cursor}, len={len(payload)}")
    latents_raw = brotli.decompress(latents_blob)
    latents = parse_latents_payload(latents_raw)
    sections.append(
        {
            "name": "latents_delta_uint8_brotli",
            "length_prefix_offset": latents_len_offset,
            "compressed_bytes": latents_len,
            "uncompressed_bytes": len(latents_raw),
            "sha256": _sha256_bytes(latents_blob),
        }
    )

    return {
        "member_format": "hnerv_muon_codec_v1",
        "member_bytes": len(payload),
        "section_length_prefix_bytes": 12,
        "meta_brotli_offset": meta_brotli_start,
        "sections": sections,
        "meta": meta,
        "decoder": decoder,
        "latents": latents,
        "compressed_payload_bytes": sum(item["compressed_bytes"] for item in sections),
    }


def _docstring(path: Path) -> str:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return ""
    return ast.get_docstring(module) or ""


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _literal_number(pattern: str, text: str) -> float | int | None:
    value = _first_match(pattern, text)
    if value is None:
        return None
    if "." in value or "e" in value.lower():
        return float(value)
    return int(value)


def parse_source_summary(source_dir: Path) -> dict[str, Any]:
    """Extract deterministic PR95 HNeRV/Muon source and curriculum facts."""
    readme_path = source_dir / "README.md"
    model_path = source_dir / "src/model.py"
    optim_path = source_dir / "src/optim.py"
    codec_path = source_dir / "src/codec.py"
    score_path = source_dir / "src/score.py"
    train_path = source_dir / "src/train.py"
    stage_dir = source_dir / "src/stages"

    source_files = sorted(path for path in source_dir.rglob("*") if _is_source_file(path))
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    model_text = model_path.read_text(encoding="utf-8") if model_path.exists() else ""
    optim_text = optim_path.read_text(encoding="utf-8") if optim_path.exists() else ""

    stage_records: list[dict[str, Any]] = []
    for stage_path in sorted(stage_dir.glob("stage*.py")):
        text = stage_path.read_text(encoding="utf-8")
        stage_records.append(
            {
                "file": _rel(stage_path),
                "name": _first_match(r'name="([^"]+)"', text) or stage_path.stem,
                "docstring_head": _docstring(stage_path).splitlines()[0] if _docstring(stage_path) else "",
                "epochs_default": _literal_number(r"epochs:\s*int\s*=\s*(\d+)", text),
                "adamw_lr": _literal_number(r"adamw_lr=([0-9.eE+-]+)", text),
                "muon_lr": _literal_number(r"muon_lr=([0-9.eE+-]+)", text),
                "muon_weight_decay_default": _literal_number(r"muon_weight_decay:\s*float\s*=\s*([0-9.eE+-]+)", text),
                "cat_lambda": _literal_number(r"cat_lambda=([0-9.eE+-]+)", text),
                "cat_sigma": _literal_number(r"cat_sigma=([0-9.eE+-]+)", text),
                "uses_qat": "use_qat=True" in text,
                "uses_muon": "use_muon=True" in text,
                "loss_family": (
                    "ce"
                    if "ce_seg_loss" in text
                    else "tau_softplus"
                    if "tau_softplus_seg_loss" in text
                    else "smooth_disagreement"
                    if "smooth_disagreement_seg_loss" in text
                    else "l7_softplus"
                    if "l7_softplus_seg_loss" in text
                    else "unknown"
                ),
            }
        )

    score_formula = _docstring(score_path).splitlines()[2].strip() if score_path.exists() else ""
    default_latent_dim = _literal_number(r"latent_dim=(\d+)", model_text)
    default_base_channels = _literal_number(r"base_channels=(\d+)", model_text)
    eval_size = _first_match(r"eval_size=\((\d+,\s*\d+)\)", model_text)

    return {
        "source_dir": _rel(source_dir),
        "source_file_count": len(source_files),
        "source_tree_sha256": _source_tree_sha256(source_files),
        "key_files": {
            "README": _rel(readme_path),
            "model": _rel(model_path),
            "optimizer": _rel(optim_path),
            "codec": _rel(codec_path),
            "score": _rel(score_path),
            "train": _rel(train_path),
        },
        "readme_summary": {
            "title": readme.splitlines()[0] if readme else "",
            "archive_claim": _first_match(r"A\s+([^\n]+archive[^\n]+)", readme),
            "curriculum_claim": _first_match(r"The pipeline is an ([^\n]+)", readme),
            "reproduce_command": _first_match(r"```bash\n([^`]+)\n```", readme),
            "claimed_training_wallclock": _first_match(r"(~\d+ hours[^\n]+)", readme),
            "external_writeup": _first_match(r"Full writeup:\s*(\S+)", readme),
        },
        "model_defaults": {
            "latent_dim": default_latent_dim,
            "base_channels": default_base_channels,
            "eval_size": eval_size,
            "base_grid": [6, 8] if "self.base_h, self.base_w = 6, 8" in model_text else None,
            "architecture": "linear latent stem + six PixelShuffle upsample blocks + dilated refine + two RGB heads",
        },
        "optimizer_summary": {
            "muon_newton_schulz_steps_default": _literal_number(r"ns_steps=([0-9]+)", optim_text),
            "muon_default_lr": _literal_number(r"lr=([0-9.eE+-]+)", optim_text),
            "muon_default_momentum": _literal_number(r"momentum=([0-9.eE+-]+)", optim_text),
            "muon_default_weight_decay": _literal_number(r"weight_decay=([0-9.eE+-]+)", optim_text),
            "partition_rule": "Muon receives 2D+ weights outside stem/RGB heads; AdamW receives stem, RGB heads, biases, 1D params, and latents.",
        },
        "codec_summary": {
            "decoder_path": "per-tensor symmetric INT8 -> zigzag -> state table -> brotli q11",
            "latent_path": "per-dim uint8 min/max -> first-order temporal delta -> zigzag uint16 -> lo/hi streams -> brotli q11",
            "source_notes_hybrid_ac_delta_bytes": _literal_number(r"was ~(\d+) bytes smaller", codec_path.read_text(encoding="utf-8") if codec_path.exists() else ""),
        },
        "score_formula": score_formula or "score = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * archive_bytes / total_video_bytes",
        "training_stages": stage_records,
    }


def _source_tree_sha256(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(_rel(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_source_file(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.is_file()


def load_static_intake(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_score_terms(seg: float, pose: float, archive_bytes: int, denominator: int = CONTEST_ORIGINAL_BYTES) -> dict[str, float]:
    seg_component = 100.0 * seg
    pose_component = math.sqrt(10.0 * pose)
    rate_component = 25.0 * archive_bytes / denominator
    return {
        "seg_component": seg_component,
        "pose_component": pose_component,
        "rate_component": rate_component,
        "recomputed_score": seg_component + pose_component + rate_component,
    }


def build_profile(archive_path: Path, source_dir: Path, static_intake_path: Path | None = None) -> dict[str, Any]:
    archive, payload = read_pr95_archive(archive_path)
    member = parse_hnerv_muon_member(payload)
    source = parse_source_summary(source_dir)
    static_intake = load_static_intake(static_intake_path) if static_intake_path else None

    score_claims: dict[str, Any] = {
        "score_claim": False,
        "score_terms_from_static_intake": None,
        "formula": "score = 100 * seg + sqrt(10 * pose) + 25 * archive_bytes / 37,545,489",
        "denominator_bytes": CONTEST_ORIGINAL_BYTES,
    }
    if static_intake and static_intake.get("claimed_body_score_inputs"):
        inputs = static_intake["claimed_body_score_inputs"]
        recomputed = compute_score_terms(
            float(inputs["seg"]),
            float(inputs["pose"]),
            int(inputs["archive_bytes"]),
        )
        score_claims.update(
            {
                "score_claim": bool(static_intake.get("score_claim", False)),
                "score_terms_from_static_intake": {
                    "inputs": inputs,
                    "recomputed": recomputed,
                    "matches_recorded_recomputed_score": abs(
                        recomputed["recomputed_score"] - float(inputs.get("recomputed_score", recomputed["recomputed_score"]))
                    )
                    < 1e-12,
                },
            }
        )

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "inputs": {
            "archive": _rel(archive_path),
            "source_dir": _rel(source_dir),
            "static_intake": _rel(static_intake_path) if static_intake_path and static_intake_path.exists() else None,
        },
        "evidence_grade": "external_static_intake_only",
        "archive_anatomy": archive,
        "hnerv_muon_blob": member,
        "source_intake": source,
        "score_term_math": score_claims,
        "dispatch_readiness": dispatch_readiness(),
        "immediate_improvement_hypotheses": improvement_hypotheses(member),
    }


def dispatch_readiness() -> dict[str, Any]:
    return {
        "ready_for_dispatch": False,
        "fail_closed": True,
        "remote_dispatch_requested": False,
        "required_before_score_claims": [
            "Replay exact eval through archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA before any PR95 score claim.",
            "Record contest_auth_eval.json, archive SHA-256, archive bytes, runtime tree hash, hardware, sample count, and recomputed score.",
            "Owned retraining needs explicit manifest/checkpoint custody for every stage: source SHA, seed, stage config, checkpoint path, checkpoint SHA-256, optimizer state policy, and final archive builder provenance.",
        ],
        "blocked_claims": [
            "Static PR95 public intake is not promotable score evidence.",
            "README/body score inputs remain external until replayed under our exact CUDA auth eval custody.",
            "Any HNeRV retrain without checkpoint and manifest custody is non-promotable replay work.",
        ],
    }


def improvement_hypotheses(member_profile: dict[str, Any]) -> list[dict[str, Any]]:
    latents = member_profile["latents"]
    sections = {item["name"]: item for item in member_profile["sections"]}
    return [
        {
            "rank": 1,
            "hook": "RAFT/ego-motion/foveation latent bases",
            "rationale": (
                f"PR95 stores {latents['n_frame_pairs']}x{latents['latent_dim']} free per-frame-pair latents. "
                "Replace part of that table with charged coefficients over existing RAFT-like flow, ego-motion, and foveation bases so retraining searches a lower-dimensional, camera-aware latent manifold instead of replaying unconstrained latent memorization."
            ),
            "owned_next_artifact": "latent-basis manifest with basis SHA, coefficient bytes, reconstruction parity checks, and exact source archive SHA",
            "dispatch_constraint": "basis-guided retrains need exact replay eval first and checkpoint custody before any score claim",
        },
        {
            "rank": 2,
            "hook": "Cool-Chic/C3/wavelet residual bases",
            "rationale": (
                "HNeRV owns the coarse neural reconstruction, but the current archive has no explicit residual basis. "
                "Attach a tiny charged residual program over Cool-Chic, C3, or wavelet atoms for systematic SegNet/PoseNet residuals that are too local for global latent movement."
            ),
            "owned_next_artifact": "residual-basis intake probe with atom bytes, affected frames/classes, decode parity, and fail-closed archive manifest",
            "dispatch_constraint": "residual atoms must be charged in archive.zip and validated by exact CUDA auth eval",
        },
        {
            "rank": 3,
            "hook": "Fridrich/Lagrangian hard-pair weighting",
            "rationale": (
                "The eight-stage curriculum already changes seg losses, C1a strength, sigma, and Muon scheduling. "
                "Use exact component traces to drive hard-pair Lagrangian weights, selecting which frame pairs deserve latent movement, residual atoms, or entropy budget instead of uniform stage replay."
            ),
            "owned_next_artifact": "hard-pair weighting policy JSON with source eval SHA, break-even score math, selected pairs, and planning/promotable status",
            "dispatch_constraint": "weights derived from proxy or stale traces are planning-only until refreshed under exact CUDA custody",
        },
        {
            "rank": 4,
            "hook": "engineered corrections and pixel-diff sparse atoms",
            "rationale": (
                "HNeRV frame outputs can expose sparse pixel/class disagreements after inflate. "
                "Encode deterministic sparse corrections only where pixel-diff atoms clear byte-normalized component benefit, using the HNeRV output as the base representation."
            ),
            "owned_next_artifact": "sparse-atom planner with changed-pixel count, atom payload bytes, no-op detection, and local decode application proof",
            "dispatch_constraint": "pixel corrections are non-promotable until they are consumed by inflate and exact eval confirms component benefit",
        },
        {
            "rank": 5,
            "hook": "self-compression entropy objectives",
            "rationale": (
                f"Decoder section is {sections['decoder_state_int8_brotli']['compressed_bytes']:,} compressed bytes and "
                f"the latent hi-byte nonzero fraction is {latents['hi_nonzero_fraction']:.6f}. "
                "Extend C1a from a weight-shaping regularizer into a measured coder-aware objective over decoder tensors, latent deltas, and optional residual atoms."
            ),
            "owned_next_artifact": "entropy-objective report tying tensor/latent histograms to estimated charged bytes and stage checkpoint manifests",
            "dispatch_constraint": "entropy proxy wins require deterministic archive rebuild and exact replay before score claims",
        },
        {
            "rank": 6,
            "hook": "decoder entropy and section recode",
            "rationale": (
                f"Decoder section is {sections['decoder_state_int8_brotli']['compressed_bytes']:,} compressed bytes; "
                "source notes a prior hybrid categorical coder beat pure brotli by about 217 bytes. Generalize that into a deterministic per-tensor coder decision table and keep the inflate dependency surface auditable."
            ),
            "owned_next_artifact": "lossless decoder-section recode candidate with section SHA map and inflate parity test",
            "dispatch_constraint": "lossless recode still needs archive custody and exact replay before any score claim",
        },
    ]


def render_markdown(profile: dict[str, Any]) -> str:
    archive = profile["archive_anatomy"]
    member = profile["hnerv_muon_blob"]
    decoder = member["decoder"]
    latents = member["latents"]
    readiness = profile["dispatch_readiness"]
    score_terms = profile["score_term_math"]["score_terms_from_static_intake"]

    lines = [
        "# PR95 HNeRV/Muon Static Intake",
        "",
        "## Archive Anatomy",
        "",
        f"- archive: `{archive['path']}`",
        f"- bytes: `{archive['archive_bytes']}`",
        f"- sha256: `{archive['archive_sha256']}`",
        f"- member: `{archive['members'][0]['name']}` stored bytes `{archive['members'][0]['file_size']}`",
        f"- rate component at contest denominator: `{archive['rate_score_component']:.12f}`",
        "",
        "## Blob Sections",
        "",
    ]
    for section in member["sections"]:
        lines.append(
            f"- `{section['name']}`: compressed `{section['compressed_bytes']}`, "
            f"uncompressed `{section['uncompressed_bytes']}`, sha256 `{section['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Parameter And Latent Counts",
            "",
            f"- decoder tensors: `{decoder['tensor_count']}`",
            f"- decoder params: `{decoder['total_params']}`",
            f"- Muon partition params: `{decoder['muon_partition_params']}`",
            f"- AdamW decoder partition params: `{decoder['adamw_decoder_partition_params']}`",
            f"- latent matrix: `{latents['n_frame_pairs']} x {latents['latent_dim']}`",
            f"- latent hi-byte nonzero fraction: `{latents['hi_nonzero_fraction']:.12f}`",
            "",
            "## Score-Term Math",
            "",
        ]
    )
    if score_terms is None:
        lines.append("- no static score terms were provided in the intake bundle")
    else:
        inputs = score_terms["inputs"]
        recomputed = score_terms["recomputed"]
        lines.extend(
            [
                f"- provided seg: `{inputs['seg']}`",
                f"- provided pose: `{inputs['pose']}`",
                f"- recomputed seg component: `{recomputed['seg_component']:.12f}`",
                f"- recomputed pose component: `{recomputed['pose_component']:.12f}`",
                f"- recomputed rate component: `{recomputed['rate_component']:.12f}`",
                f"- recomputed total: `{recomputed['recomputed_score']:.12f}`",
                "- evidence status: external/static only, not a score claim",
            ]
        )
    lines.extend(
        [
            "",
            "## Training And Optimizer Stages",
            "",
        ]
    )
    for stage in profile["source_intake"]["training_stages"]:
        lines.append(
            f"- `{stage['name']}`: epochs `{stage['epochs_default']}`, loss `{stage['loss_family']}`, "
            f"AdamW lr `{stage['adamw_lr']}`, Muon `{stage['uses_muon']}`, QAT `{stage['uses_qat']}`, "
            f"C1a lambda `{stage['cat_lambda']}`, sigma `{stage['cat_sigma']}`"
        )
    lines.extend(
        [
            "",
            "## Dispatch Readiness",
            "",
            f"- ready_for_dispatch: `{readiness['ready_for_dispatch']}`",
            f"- fail_closed: `{readiness['fail_closed']}`",
        ]
    )
    for item in readiness["required_before_score_claims"]:
        lines.append(f"- required: {item}")
    for item in readiness["blocked_claims"]:
        lines.append(f"- blocked: {item}")
    lines.extend(
        [
            "",
            "## Immediate Improvement Hypotheses",
            "",
        ]
    )
    for hypothesis in profile["immediate_improvement_hypotheses"]:
        lines.append(f"{hypothesis['rank']}. **{hypothesis['hook']}**: {hypothesis['rationale']}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(profile: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(_json_text(profile), encoding="utf-8")
    markdown_out.write_text(render_markdown(profile), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--static-intake", type=Path, default=DEFAULT_STATIC_INTAKE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--no-write", action="store_true", help="Build the profile and print JSON without writing artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = build_profile(args.archive, args.source_dir, args.static_intake)
    if args.no_write:
        print(_json_text(profile), end="")
    else:
        write_outputs(profile, args.json_out, args.markdown_out)
        print(f"wrote {_rel(args.json_out)}")
        print(f"wrote {_rel(args.markdown_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
