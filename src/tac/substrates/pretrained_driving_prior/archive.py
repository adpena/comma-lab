"""Pre-trained driving prior archive grammar — DP1 monolithic 0.bin.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" lessons:

* L3 monolithic single-file ``0.bin`` (single-zip-member ``x`` slot in archive.zip)
* L4 inflate.py ≤ 200 LOC (substrate-engineering waiver justified: full
  codebook lookup + residual decode + render pipeline)
* L8 deterministic (sorted-keys JSON, fp16 renderer state_dict, fixed brotli)

Archive grammar (DP1)::

    MAGIC(4)             b"DP1\\x00"   (sister to TT5L, SIRN, BLLE)
    VERSION(1)           u8 (== 1)
    NUM_PAIRS(2)         u16
    OUTPUT_HEIGHT(2)     u16
    OUTPUT_WIDTH(2)      u16
    PER_PAIR_BYTES(1)    u8
    CODEBOOK_LEN(4)      u32    serialize_codebook(...) output bytes
    RENDERER_LEN(4)      u32    brotli(state_dict, fp16)
    RESIDUAL_LEN(4)      u32    brotli(int8 per-pair residual)
    META_LEN(4)          u32    utf-8 JSON
    CODEBOOK_BLOB        ...
    RENDERER_BLOB        ...
    RESIDUAL_BLOB        ...
    META_BLOB            ...

Section semantics:

* **CODEBOOK_BLOB** (5-10 KB): the frozen dashcam-statistical prior
  distilled offline from Comma2k19. Loaded once at inflate; never trained
  against the contest video.
* **RENDERER_BLOB**: brotli-compressed FP16 state_dict of the small
  contest-video-overfit renderer. The renderer learns the contest-specific
  delta from the prior.
* **RESIDUAL_BLOB**: brotli-compressed per-pair int8 residual
  (``num_pairs * per_pair_bytes`` bytes pre-brotli).
* **META_BLOB**: JSON with hparams + scales + dataset license attribution.

**Total target archive size: 60-90 KB** (5-10 KB codebook + 40-60 KB renderer
+ 10-20 KB residual). Larger than TT5L (95-110 KB) was intended to be... wait,
TT5L is 95-110 KB, this is SMALLER because the codebook offloads dashcam
statistics into a fixed lookup. **Predicted score-CPU: [0.175, 0.190] (MEDIUM-EV)**.

CLAUDE.md compliance:

* Deterministic (sorted-keys JSON, fp16 cpu state_dict, brotli quality fixed)
* No /tmp paths
* No scorer load
* No score claim (score_claim_valid=False everywhere)
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import torch

from tac.substrates.pretrained_driving_prior.codebook import (
    DashcamCodebook,
    parse_codebook,
    serialize_codebook,
)

DP1_MAGIC: bytes = b"DP1\x00"
"""Pre-trained driving prior archive magic; 4 bytes."""

DP1_SCHEMA_VERSION: int = 1
"""Schema version byte; bump on grammar changes."""

# Header layout: MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + OUT_H(2) + OUT_W(2)
#              + PER_PAIR_BYTES(1) + CODEBOOK_LEN(4) + RENDERER_LEN(4)
#              + RESIDUAL_LEN(4) + META_LEN(4)
# = 4+1+2+2+2+1+4+4+4+4 = 28 bytes
DP1_HEADER_FMT: str = "<4sBHHHBIIII"
DP1_HEADER_SIZE: int = struct.calcsize(DP1_HEADER_FMT)
assert DP1_HEADER_SIZE == 28, (
    f"DP1 header size invariant: expected 28, got {DP1_HEADER_SIZE}"
)

_BROTLI_QUALITY: int = 9


@dataclass(frozen=True)
class DrivingPriorArchive:
    """Parsed DP1 archive — the inflate-time data contract."""

    codebook: DashcamCodebook
    """The frozen dashcam codebook."""

    renderer_state_dict: dict[str, torch.Tensor]
    """The contest-overfit renderer state_dict (FP16)."""

    per_pair_residual: bytes
    """Raw int8 per-pair residual bytes (length = num_pairs * per_pair_bytes)."""

    meta: dict[str, object]
    """Sidecar JSON with hparams + dataset license tags."""

    schema_version: int
    num_pairs: int
    output_height: int
    output_width: int
    per_pair_bytes: int


def _serialize_state_dict(sd: dict[str, torch.Tensor]) -> bytes:
    """Pickle + brotli a state_dict deterministically (fp16 cpu, contiguous)."""
    buf = io.BytesIO()
    sd_cpu = {
        k: v.detach().to("cpu", dtype=torch.float16).contiguous()
        for k, v in sd.items()
    }
    pickle.dump(sd_cpu, buf, protocol=pickle.DEFAULT_PROTOCOL)
    return brotli.compress(buf.getvalue(), quality=_BROTLI_QUALITY)


def _deserialize_state_dict(blob: bytes) -> dict[str, torch.Tensor]:
    """Brotli + pickle.load a state_dict.

    Per CLAUDE.md WEIGHTS_ONLY_FALSE_OK: trusted-substrate-state-dict-local-artifact
    """
    raw = brotli.decompress(blob)
    # WEIGHTS_ONLY_FALSE_OK:trusted-pretrained-driving-prior-substrate-state-dict-local-artifact
    return pickle.loads(raw)


def pack_archive(
    codebook: DashcamCodebook,
    renderer_state_dict: dict[str, torch.Tensor],
    per_pair_residual: bytes,
    meta: dict[str, object],
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
    per_pair_bytes: int,
) -> bytes:
    """Pack the four DP1 sections into a deterministic byte string.

    Raises:
        ValueError: header field overflow or section bytes shape mismatch.
    """
    if not (0 <= num_pairs <= 0xFFFF):
        raise ValueError(f"num_pairs {num_pairs} overflows u16")
    if not (0 <= output_height <= 0xFFFF):
        raise ValueError(f"output_height {output_height} overflows u16")
    if not (0 <= output_width <= 0xFFFF):
        raise ValueError(f"output_width {output_width} overflows u16")
    if not (0 <= per_pair_bytes <= 0xFF):
        raise ValueError(f"per_pair_bytes {per_pair_bytes} overflows u8")
    if len(per_pair_residual) != num_pairs * per_pair_bytes:
        raise ValueError(
            f"per_pair_residual length {len(per_pair_residual)} != "
            f"num_pairs * per_pair_bytes = {num_pairs * per_pair_bytes}"
        )

    codebook_blob = serialize_codebook(codebook)
    renderer_blob = _serialize_state_dict(renderer_state_dict)
    residual_blob = brotli.compress(per_pair_residual, quality=_BROTLI_QUALITY)
    meta_blob = json.dumps(meta, sort_keys=True).encode("utf-8")

    header = struct.pack(
        DP1_HEADER_FMT,
        DP1_MAGIC,
        DP1_SCHEMA_VERSION,
        num_pairs,
        output_height,
        output_width,
        per_pair_bytes,
        len(codebook_blob),
        len(renderer_blob),
        len(residual_blob),
        len(meta_blob),
    )
    return header + codebook_blob + renderer_blob + residual_blob + meta_blob


def parse_archive(data: bytes) -> DrivingPriorArchive:
    """Parse DP1 archive bytes into a :class:`DrivingPriorArchive`.

    Raises:
        ValueError: short read, wrong magic, version mismatch, section
            byte-count mismatch.
    """
    if len(data) < DP1_HEADER_SIZE:
        raise ValueError(
            f"DP1 archive too short for header: {len(data)} < {DP1_HEADER_SIZE}"
        )
    (
        magic,
        version,
        num_pairs,
        out_h,
        out_w,
        per_pair_bytes,
        codebook_len,
        renderer_len,
        residual_len,
        meta_len,
    ) = struct.unpack(DP1_HEADER_FMT, data[:DP1_HEADER_SIZE])
    if magic != DP1_MAGIC:
        raise ValueError(f"DP1 archive magic mismatch: {magic!r} != {DP1_MAGIC!r}")
    if version != DP1_SCHEMA_VERSION:
        raise ValueError(
            f"DP1 schema version {version} != expected {DP1_SCHEMA_VERSION}"
        )

    cursor = DP1_HEADER_SIZE
    expected_total = (
        DP1_HEADER_SIZE + codebook_len + renderer_len + residual_len + meta_len
    )
    if len(data) < expected_total:
        raise ValueError(
            f"DP1 archive truncated: have {len(data)} bytes, expected {expected_total}"
        )

    codebook_blob = data[cursor : cursor + codebook_len]
    cursor += codebook_len
    renderer_blob = data[cursor : cursor + renderer_len]
    cursor += renderer_len
    residual_blob = data[cursor : cursor + residual_len]
    cursor += residual_len
    meta_blob = data[cursor : cursor + meta_len]

    codebook = parse_codebook(codebook_blob)
    renderer_state_dict = _deserialize_state_dict(renderer_blob)
    per_pair_residual = brotli.decompress(residual_blob)
    if len(per_pair_residual) != num_pairs * per_pair_bytes:
        raise ValueError(
            f"DP1 per-pair residual length {len(per_pair_residual)} != "
            f"num_pairs * per_pair_bytes = {num_pairs * per_pair_bytes}"
        )
    meta = json.loads(meta_blob.decode("utf-8"))

    return DrivingPriorArchive(
        codebook=codebook,
        renderer_state_dict=renderer_state_dict,
        per_pair_residual=per_pair_residual,
        meta=meta,
        schema_version=version,
        num_pairs=num_pairs,
        output_height=out_h,
        output_width=out_w,
        per_pair_bytes=per_pair_bytes,
    )


def build_readiness_manifest(
    *,
    archive_path: str,
    codebook_path: str,
    archive_bytes: int,
    codebook_bytes: int,
) -> dict[str, object]:
    """Build a proxy readiness manifest for L0/L1 scaffold reporting.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this manifest
    explicitly tags ``evidence_grade=[proxy]`` and ``score_claim=false``
    so future agents cannot mistake it for a contest-CUDA / contest-CPU
    measurement.
    """
    return {
        "schema": "dp1_readiness_manifest_v1",
        "archive_path": archive_path,
        "codebook_path": codebook_path,
        "archive_bytes": int(archive_bytes),
        "codebook_bytes": int(codebook_bytes),
        "evidence_grade": "[proxy]",
        "score_claim": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "dispatch_blockers": [
            "contest_cuda_eval_not_run",
            "contest_cpu_eval_not_run",
            "real_codebook_distillation_pending",
            "real_renderer_training_pending",
        ],
        "lane_id": "lane_pretrained_driving_prior_lane_scaffold_20260513",
        "target_modes": [
            "contest_one_video_replay",
            "contest_generalized",
            "production_generalized",
            "production_edge_adaptive",
            "research_substrate",
        ],
        "deployment_target": "t4_contest_runtime",
        "lane_class": "substrate_engineering",
        "research_only": True,
    }


__all__ = [
    "DP1_HEADER_FMT",
    "DP1_HEADER_SIZE",
    "DP1_MAGIC",
    "DP1_SCHEMA_VERSION",
    "DrivingPriorArchive",
    "build_readiness_manifest",
    "pack_archive",
    "parse_archive",
]
