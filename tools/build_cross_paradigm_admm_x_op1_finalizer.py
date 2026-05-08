#!/usr/bin/env python3
"""Build the cross-paradigm ADMM × continuous-K + Op1 finalizer byte-closed candidate archive.

Source-of-finding
-----------------
``reports/raw/pr101_cross_paradigm_hstack_vstack_<UTC>/manifest.json``
(commit 8d33d5c1 of ``tools/pr101_cross_paradigm_hstack_vstack_empirical.py``)
showed that running Op1 (PR101 split-Brotli with auto_select byte_maps) on
the *dequantized* post-ADMM substrate yields a CPL1-wrapped Op1 blob of
**137,531 B** [PHANTOM, pre-fix; corrected to 137,469 B at commit 98d2174b
per Codex CRITICAL #4.1] — vs ADMM-alone (continuous-K wire format) at 153,639 B
and Op1_alone on raw fp32 at 161,942 B. NOTE: this builder used a different
(intentional) dequant formula (``rounded * float(qt.scale)``, raw fp32 scale)
and produces ``op1_inner_blob_bytes = 137,348`` (see build_manifest.json line
``op1_inner_blob_bytes``); XPARADIGM + canonical orchestrator use the
fp16-cast formula (``rounded * float(np.float16(scale))``) matching the Path B
step 6 runtime decoder and produce 137,469 B. Both are internally consistent
(builder ships its own forked inflate.py); cross-tool divergence acknowledged.

CRITICAL CAVEAT (Subagent WIRE-DECODER, 2026-05-08): the ~137K B figure is
the *decoder section only*. A deployable archive STILL needs:

  - 8 byte CPLX wrapper (4 magic + 4 length)
  - 15,387 byte PR101 latent_blob (passthrough)
  - sidecar_blob (PR101 passthrough; ~707-1,500 B)

**Total deployable archive (this builder) = 153,513 B** (per build_manifest.json
of the 2026-05-08T062603Z run; see ``archive_bytes`` field).

This roughly TIES Path B step 6 (153,699 B) once latent + sidecar are added.
The original "winner" framing was an apples-to-oranges decoder-only comparison.
The cross-paradigm composition is still WORTH testing on contest-CUDA because
the *distortion structure* differs from Path B step 6: the substrate goes
through an Op1 re-quantization (fresh int8 with PR101's per-channel scales)
which may behave differently at the scorer-basin level than pure ADMM K-coarsening.

Mandate (Subagent WIRE-DECODER)
-------------------------------
1. Reuse the encoder logic from XPARADIGM's tool (ADMM bisect + dequantize +
   Op1 encode), but produce a deployable archive (decoder + latent + sidecar).
2. Output to ``experiments/results/cross_paradigm_admm_x_op1_finalizer_<ts>/``
   with ``archive.zip``, ``submission_dir/`` (forked inflate + src/), and
   ``build_manifest.json``.
3. Smoke-test the archive locally: parse the inner blob with the forked
   inflate, verify CPL1 round-trip + 600 latent pairs decoded.
4. Stamp the build_manifest.json with proper fail-closed custody fields.

Wire format (inner blob, single ZIP member 'x')
-----------------------------------------------
::

  +--------------------------------------------+
  | 4 bytes: magic = b"CPLX"                   |
  +--------------------------------------------+
  | 4 bytes: uint32 LE = decoder_total_bytes D |  (includes magic + length + CPL1 blob)
  +--------------------------------------------+
  | (D - 8) bytes: CPL1-wrapped Op1 blob       |
  +--------------------------------------------+
  | 15387 bytes: PR101 latent_blob (UNCHANGED) |
  +--------------------------------------------+
  | remaining: PR101 sidecar_blob (UNCHANGED)  |
  +--------------------------------------------+

The CPL1 blob is produced by ``CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])
.encode(dequantized_post_admm_state_dict, skip_validate=True)``.

The decoder reads magic, validates "CPLX", reads D, hands the inner CPL1 blob
to ``Op1_PR101SplitBrotli`` (with auto_select effective_byte_maps from op_state).

CLAUDE.md compliance
--------------------
- ``family_falsified=False``,
  ``falsification_scope="cross_paradigm_admm_x_continuous_k_plus_op1_finalizer_only"``.
- ``ready_for_exact_eval_dispatch=False`` (CPU build never promotes itself).
- ``cuda_eval_worth_testing=True`` (cross-paradigm score behavior unmeasured).
- Pure-CPU; never loads a scorer; tags evidence ``[CPU-build]``.

Out-of-scope
------------
- No dispatch (Lightning bootstrap is owned by other subagents).
- No core ``CodecPipeline`` modifications.

Usage
-----

.. code-block:: bash

    .venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    DECODER_BLOB_LEN,
    FIXED_STATE_SCHEMA,
    LATENT_BLOB_LEN,
    N_QUANT,
    _quantize_tensor,
)

# Re-use canonical archive helpers from the Lightning builder.
_LIGHTNING_BUILDER_PATH = (
    REPO_ROOT / "experiments" / "lossy_coarsening_lightning_cuda_test.py"
)
_spec = importlib.util.spec_from_file_location(
    "_lossy_coarsening_lightning_cuda_test", _LIGHTNING_BUILDER_PATH
)
if _spec is None or _spec.loader is None:  # pragma: no cover - sanity
    raise SystemExit(f"FATAL: could not load builder from {_LIGHTNING_BUILDER_PATH}")
_lightning_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lightning_builder)

_read_pr101_inner_blob = _lightning_builder._read_pr101_inner_blob
_split_pr101_inner_blob = _lightning_builder._split_pr101_inner_blob
_write_pr101_archive = _lightning_builder._write_pr101_archive

LANE_ID = "cross_paradigm_admm_continuous_k_plus_op1_finalizer"
SCHEMA_VERSION = "cross_paradigm_admm_x_op1_finalizer_build.v1"
TOOL_NAME = "tools/build_cross_paradigm_admm_x_op1_finalizer.py"

# CPLX wire-format magic (Cross-Paradigm Lossy + Op1).
CPLX_MAGIC = b"CPLX"

# ADMM Ks pulled from the same source as Path B step 6 (rms_target=0.0386).
ADMM_PATH_B_STEP6_KS: tuple[int, ...] = (
    2, 1, 5, 1, 5, 1, 5, 1, 2, 1, 2, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
)
ADMM_PROXY_ARCHIVE_BYTES = 153_639  # ADMM-alone archive (with PR101 wire format)
ADMM_PROXY_REL_ERR = 0.0415393353487541
ADMM_PROXY_LAMBDA = 1_276_154.6884887693
ADMM_RMS_TARGET = 0.0386
ADMM_SOURCE_MANIFEST = (
    "reports/raw/pr101_omega_opt_admm_x_lossy_coarsening_20260508T041303Z/manifest.json"
)
XPARADIGM_SOURCE_TOOL = (
    "tools/pr101_cross_paradigm_hstack_vstack_empirical.py"
)
XPARADIGM_DECODER_ONLY_BYTES = 137_531  # XPARADIGM's measurement (decoder only, no latent)

DEFAULT_PR101_STATE_DICT = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex"
    / "pr101_decoder_state_dict.pt"
)
DEFAULT_PR101_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "archive.zip"
)
DEFAULT_PR101_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr_intake_full"
    / "public_pr101_intake_20260505_auto"
    / "source/submissions/hnerv_ft_microcodec"
)
# Predicted band: roughly tied with Path B step 6 (0.18-0.22) since archive
# bytes are comparable and distortion structure is also K-coarsened-ish.
# Tagged [predicted] — actual band depends on Op1 re-quantization behavior.
DEFAULT_PREDICTED_BAND = (0.18, 0.22)

CPU_BUILD_SCORE_BLOCKERS = [
    "cpu_build_rel_err_proxy_not_score_evidence",
    "exact_cuda_auth_eval_not_yet_harvested",
    "requires_contest_auth_eval_json_before_score_promotion_rank_or_kill",
]


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cpu_build_proxy_guard_fields() -> dict[str, object]:
    """Fail-closed custody fields for the local CPU build artifact."""
    return {
        "evidence_semantics": "cpu_build_byte_closed_candidate_proxy_no_score",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "cuda_eval_worth_testing": True,
        "family_falsified": False,
        "falsification_scope": "cross_paradigm_admm_x_continuous_k_plus_op1_finalizer_only",
        "score_claim_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
        "dispatch_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
    }


def _resolve_output_root(output_root: Path) -> Path:
    """Resolve CLI output roots relative to the repo root.

    The builder records repo-relative artifact paths in its manifest. Accepting
    relative ``--output-root`` values is useful for worker-owned artifact dirs,
    but the build path must be absolute before ``relative_to(REPO_ROOT)`` is
    used for custody logging.
    """
    expanded = output_root.expanduser()
    if not expanded.is_absolute():
        expanded = REPO_ROOT / expanded
    return expanded.resolve()


def _build_dequantized_substrate_via_admm(
    state_dict_path: Path,
    Ks: list[int],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Apply per-tensor int8 quant + ADMM K-coarsening + dequantize to fp32.

    Mirrors the encoder side of XPARADIGM's ``_stack3_admm_continuous_k_then_op1``
    *exactly* (line for line) so the bytes_admm_then_op1 measurement is reproduced.

    Returns:
        (dequantized_state_dict, encode_stats)
    """
    if not state_dict_path.is_file():
        raise SystemExit(f"FATAL: PR101 state_dict not found: {state_dict_path}")
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)

    n_tensors = len(FIXED_STATE_SCHEMA)
    if len(Ks) != n_tensors:
        raise SystemExit(
            f"FATAL: Ks length {len(Ks)} != n_tensors {n_tensors}"
        )

    rebuilt: dict[str, torch.Tensor] = {}
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor_scales: list[float] = []
    n_symbols = 0
    for (name, shape), K in zip(FIXED_STATE_SCHEMA, Ks, strict=True):
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        rounded = (np.round(symbols_i32.astype(np.float64) / K) * K).astype(np.int32)
        rounded_clipped = np.clip(rounded, -127, 127).astype(np.int8)
        # Dequantize: weight_fp32 = q_i8 * scale (matches PR101's decode_decoder_compact
        # at line 645: ``sd[name] = q.astype(np.float32) * float(scale)``).
        # XPARADIGM's _stack3_admm_continuous_k_then_op1 used a (q_i8/N_QUANT)*scale
        # form which is 127x too small — that produced the right BYTE COUNT (Op1
        # re-quantizes from scratch so abs_max scaling absorbs the magnitude
        # difference) but corrupted the recovered weights. WIRE-DECODER fix:
        # use the canonical PR101 dequantize so the recovered substrate matches
        # what the scorer actually sees.
        deq = rounded_clipped.astype(np.float32) * float(qt.scale)
        rebuilt[name] = torch.from_numpy(deq.reshape(shape))
        abs_orig += float(np.abs(symbols_i32).astype(np.float64).sum())
        abs_err += float(np.abs(rounded_clipped.astype(np.int32) - symbols_i32).astype(np.float64).sum())
        per_tensor_scales.append(float(qt.scale))
        n_symbols += int(symbols_i32.size)

    rel_err = abs_err / abs_orig if abs_orig > 1e-9 else 0.0
    stats = {
        "rel_err_int8_after_admm": rel_err,
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "per_tensor_scale_fp32": per_tensor_scales,
        "Ks": list(Ks),
    }
    return rebuilt, stats


def _encode_op1_finalizer_blob(
    dequantized_state_dict: dict[str, torch.Tensor],
) -> tuple[bytes, dict[int, str], dict]:
    """Run PR101 split-Brotli (auto_select=True) on the dequantized substrate.

    NOTE on CodecPipeline bypass (WIRE-DECODER 2026-05-08):
    The canonical ``CodecPipeline([Op1_PR101SplitBrotli(auto_select=True)])``
    HAD a wire-format bug: ``effective_byte_maps`` integer keys are coerced
    to strings by ``json.dumps(op_state)`` in CPL1's wrapper, and on decode
    ``decode_decoder_compact`` does ``idx in effective_byte_maps`` for an
    integer ``idx`` — the string-keyed dict misses, falling back to
    ``DECODER_BYTE_MAPS`` which produces sign-flips on ``negzig`` tensors
    (e.g. ``blocks.5.bias`` was 100% sign-flipped).

    To sidestep that bug AND deliver a deployable archive in this subagent's
    scope, we bypass ``CodecPipeline`` entirely and call
    ``encode_decoder_compact`` directly. The returned blob is the raw PR101
    split-Brotli stream concatenation; the byte_maps are returned separately
    and serialized to our own CPLX wire format with int keys reconstructed
    on the decoder side.

    Bug-class status update (2026-05-08, ORCH-SYNC Bug 2): the underlying
    ``CodecPipeline`` int-key bug has now been fixed via CPL2 wire format
    (canonical default; ``CodecPipeline.encode`` emits CPL2 magic and
    preserves int keys in op_state via a sentinel envelope). New tools
    SHOULD use the canonical ``CodecPipeline`` directly. This tool keeps
    its own CPLX wrapper for forensic compat with the existing deployable
    archive at ``cross_paradigm_admm_x_op1_finalizer_20260508T062603Z/``.

    Returns:
        (op1_inner_blob, effective_byte_maps_int_keyed, op1_stats)
    """
    from tac.pr101_split_brotli_codec import (
        auto_select_byte_maps,
        encode_decoder_compact,
    )
    bm = auto_select_byte_maps(
        dequantized_state_dict, brotli_quality=11
    )
    blob = encode_decoder_compact(
        dequantized_state_dict,
        brotli_quality=11,
        effective_byte_maps=bm,
        auto_select=False,
    )
    return blob, dict(bm), {
        "op1_inner_blob_bytes": len(blob),
        "effective_byte_maps_count": len(bm),
        "non_default_byte_maps": {
            int(idx): m for idx, m in bm.items() if m != "zig"
        },
    }


def _build_decoder_section_cplx(
    op1_inner_blob: bytes,
    byte_maps: dict[int, str],
) -> bytes:
    """Wrap the Op1 inner blob in the CPLX container.

    Wire format (decoder section):
        4 bytes: magic = b"CPLX"
        4 bytes: uint32 LE = section_total_bytes D (incl magic + length + body)
        2 bytes: uint16 LE = byte_maps_json_len J
        J bytes: utf-8 JSON {str(idx): byte_map_str}
        (D - 10 - J) bytes: Op1 inner blob (raw PR101 split-Brotli streams)

    The decoder reads section_total -> walks past it to get latent_blob.
    The decoder converts JSON string keys back to int before passing to
    ``decode_decoder_compact`` (works around the CodecPipeline json key
    serialization bug).
    """
    bm_json = json.dumps(
        {str(int(idx)): str(m) for idx, m in byte_maps.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(bm_json) > 0xFFFF:
        raise SystemExit(
            f"FATAL: byte_maps JSON {len(bm_json)} > 65535; CPLX uint16 length "
            "overflow"
        )
    body = struct.pack("<H", len(bm_json)) + bm_json + op1_inner_blob
    section_total = 8 + len(body)  # 4 magic + 4 length + body
    return CPLX_MAGIC + struct.pack("<I", section_total) + body


def _stage_forked_submission_dir(
    submission_dir: Path,
    *,
    pr101_source_dir: Path,
) -> None:
    """Copy PR101's src/ (codec.py, model.py) and write forked inflate.py + .sh."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    pr101_src_dir = pr101_source_dir / "src"
    if not pr101_src_dir.is_dir():
        raise SystemExit(f"FATAL: PR101 source src/ not found: {pr101_src_dir}")
    for src_file_name in ("codec.py", "model.py"):
        src_path = pr101_src_dir / src_file_name
        if not src_path.is_file():
            raise SystemExit(f"FATAL: PR101 source missing: {src_path}")
        shutil.copy2(src_path, src_dir / src_file_name)

    # Vendor the tac modules the inflate path needs (PR101 split-Brotli
    # decoder + derivers). The inflate uv shell uses --no-project, so we
    # cannot rely on tac being importable. Vendor the minimum set.
    # NOTE: codec_pipeline.py is intentionally NOT vendored — we bypass
    # CPL1 entirely (see WIRE-DECODER NOTE in the inflate.py docstring).
    tac_src = REPO_ROOT / "src" / "tac"
    vendored_tac = src_dir / "tac"
    vendored_tac.mkdir(parents=True, exist_ok=True)
    (vendored_tac / "__init__.py").write_text("", encoding="utf-8")
    for fname in (
        "pr101_split_brotli_codec.py",
        "pr101_split_brotli_codec_derivers.py",
    ):
        src_path = tac_src / fname
        if not src_path.is_file():
            raise SystemExit(f"FATAL: tac source missing: {src_path}")
        shutil.copy2(src_path, vendored_tac / fname)

    inflate_py = submission_dir / "inflate.py"
    inflate_py.write_text(FORKED_INFLATE_PY, encoding="utf-8")
    inflate_sh = submission_dir / "inflate.sh"
    inflate_sh.write_text(FORKED_INFLATE_SH, encoding="utf-8")
    inflate_sh.chmod(0o755)
    inflate_py.chmod(0o755)


# ---------------------------------------------------------------------------
# Forked inflate sources
# ---------------------------------------------------------------------------

FORKED_INFLATE_PY = '''#!/usr/bin/env python
"""Forked inflate for cross-paradigm ADMM-x-Op1-finalizer archive.

Wire format (inner blob, single ZIP member 'x'):
    +-----------------------------------------------+
    | 4 bytes: magic = b"CPLX"                      |
    +-----------------------------------------------+
    | 4 bytes: uint32 LE = decoder_section_bytes D  |
    +-----------------------------------------------+
    | 2 bytes: uint16 LE = byte_maps_json_len J     |
    +-----------------------------------------------+
    | J bytes: utf-8 JSON {str(idx): byte_map_str}  |
    +-----------------------------------------------+
    | (D - 10 - J) bytes: Op1 inner blob            |
    |   = PR101 split-Brotli stream concatenation   |
    +-----------------------------------------------+
    | 15387 bytes: PR101 latent_blob (UNCHANGED)    |
    +-----------------------------------------------+
    | remaining: PR101 sidecar_blob (UNCHANGED)     |
    +-----------------------------------------------+

The Op1 inner blob is the raw PR101 split-Brotli stream concatenation produced
by ``encode_decoder_compact(state_dict, effective_byte_maps=byte_maps,
auto_select=False)`` on the dequantized post-ADMM substrate. The byte_maps
JSON is rehydrated to int keys here before passing to ``decode_decoder_compact``.

Latent + sidecar use PR101's authored decoder functions verbatim.

WIRE-DECODER NOTE 2026-05-08: this bypasses ``CodecPipeline`` because the CPL1
wrapper has a JSON-int-key bug (effective_byte_maps int keys → strings → miss
in ``decode_decoder_compact``'s ``idx in effective_byte_maps`` check).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from codec import (  # PR101 originals (vendored)
    LATENT_BLOB_LEN,
    N_PAIRS,
    LATENT_DIM,
    BASE_CHANNELS,
    EVAL_SIZE,
    decode_latents_compact,
    apply_latent_sidecar,
)
from model import HNeRVDecoder
# Vendored tac module: only PR101 split-Brotli decoder needed (we bypass
# CodecPipeline entirely; see WIRE-DECODER NOTE above).
from tac.pr101_split_brotli_codec import decode_decoder_compact

CPLX_MAGIC = b"CPLX"
CPLX_HEADER_LEN = 8  # 4 magic + 4 length

CAMERA_H, CAMERA_W = 874, 1164


def parse_cross_paradigm_archive(archive_bytes):
    if len(archive_bytes) < CPLX_HEADER_LEN + 2 + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for cross-paradigm format"
        )
    magic = archive_bytes[:4]
    if magic != CPLX_MAGIC:
        raise ValueError(
            f"bad magic {magic!r}, expected {CPLX_MAGIC!r} (cross-paradigm CPLX wire format)"
        )
    section_total = struct.unpack("<I", archive_bytes[4:8])[0]
    if section_total < CPLX_HEADER_LEN + 2:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum {CPLX_HEADER_LEN + 2}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    bm_json_len = struct.unpack("<H", archive_bytes[8:10])[0]
    bm_json_start = 10
    bm_json_end = bm_json_start + bm_json_len
    if bm_json_end > section_total:
        raise ValueError(
            f"byte_maps json length {bm_json_len} overflows section_total {section_total}"
        )
    bm_str_keyed = json.loads(
        archive_bytes[bm_json_start:bm_json_end].decode("utf-8")
    )
    # Rehydrate int keys (JSON serialization coerces dict[int, str] -> str-keyed).
    effective_byte_maps = {int(k): str(v) for k, v in bm_str_keyed.items()}

    op1_inner_blob = archive_bytes[bm_json_end:section_total]
    decoder_sd = decode_decoder_compact(
        op1_inner_blob, effective_byte_maps=effective_byte_maps
    )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in cross-paradigm archive")

    meta = {
        "n_pairs": N_PAIRS,
        "latent_dim": LATENT_DIM,
        "base_channels": BASE_CHANNELS,
        "eval_size": list(EVAL_SIZE),
    }
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    return decoder_sd, latents, meta


def inflate(src_bin: str, dst_raw: str):
    with open(src_bin, "rb") as f:
        archive_bytes = f.read()
    decoder_sd, latents, meta = parse_cross_paradigm_archive(archive_bytes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    n = 0
    with torch.inference_mode(), open(dst_raw, "wb") as fout:
        for i in range(0, n_pairs, 16):
            j = min(i + 16, n_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)
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

FORKED_INFLATE_SH = r'''#!/usr/bin/env bash
# Forked PR101 inflate.sh for cross-paradigm ADMM-x-Op1-finalizer lane.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

mkdir -p "$OUTPUT_DIR"

INFLATE_BROTLI_SPEC="${INFLATE_BROTLI_SPEC:-brotli==1.2.0}"
INFLATE_TORCH_SPEC="${INFLATE_TORCH_SPEC:-torch==2.5.1+cu124}"
INFLATE_NUMPY_SPEC="${INFLATE_NUMPY_SPEC:-numpy==2.4.4}"
UV_BIN="${UV_BIN:-$(command -v uv || echo /usr/local/bin/uv)}"
if [ ! -x "$UV_BIN" ]; then
  echo "FATAL: uv not on PATH (UV_BIN=$UV_BIN); the canonical inflate-time" >&2
  echo "       env requires uv. Bootstrap with scripts/ensure_remote_uv.sh." >&2
  exit 1
fi

UV_WITH_INFLATE_DEPS=(
  --with "$INFLATE_BROTLI_SPEC"
  --with "$INFLATE_TORCH_SPEC"
  --with "$INFLATE_NUMPY_SPEC"
)

echo "[cross-paradigm-inflate] uv specs: brotli=$INFLATE_BROTLI_SPEC torch=$INFLATE_TORCH_SPEC numpy=$INFLATE_NUMPY_SPEC" >&2

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
  "$UV_BIN" run --no-project "${UV_WITH_INFLATE_DEPS[@]}" python "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
'''


def _local_smoke_roundtrip(
    archive_path: Path, *, pr101_state_dict_path: Path, submission_dir: Path
) -> dict:
    """Smoke test: parse via the forked inflate, verify CPL1 round-trip + 600
    latent pairs decoded.

    Compares decoded fp32 weights against the encoder's lossy-coarsened
    dequantized substrate (this is what Op1 was fed; if the round-trip is
    correct, decoder_sd should equal that substrate to within Op1's lossy
    re-quantization noise).
    """
    spec_path = submission_dir / "inflate.py"
    if not spec_path.is_file():
        raise SystemExit(
            f"FATAL: forked inflate.py missing for smoke roundtrip: {spec_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "forked_inflate_cross_paradigm", spec_path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load forked inflate spec from {spec_path}")
    forked_inflate = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir / "src"))
    try:
        spec.loader.exec_module(forked_inflate)

        inner = _read_pr101_inner_blob(archive_path)
        decoder_sd, latents, meta = forked_inflate.parse_cross_paradigm_archive(inner)
    finally:
        sys.path.pop(0)

    # Compare to the lossy-coarsened ENCODER substrate (what Op1 was fed).
    sd_ref = torch.load(
        pr101_state_dict_path, map_location="cpu", weights_only=False
    )
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor: list[dict] = []
    for (name, _shape), K in zip(FIXED_STATE_SCHEMA, ADMM_PATH_B_STEP6_KS, strict=True):
        qt_ref = _quantize_tensor(name, sd_ref[name], n_quant=N_QUANT)
        symbols_i32 = qt_ref.q_i8.astype(np.int32).flatten()
        rounded = (np.round(symbols_i32.astype(np.float64) / K) * K).astype(np.int32)
        rounded_clipped = np.clip(rounded, -127, 127).astype(np.int8)
        # Match the canonical PR101 dequantize (q_i8 * scale, not q_i8/N_QUANT*scale).
        ref_lossy_fp32 = (
            rounded_clipped.astype(np.float32) * float(qt_ref.scale)
        ).reshape(decoder_sd[name].shape)
        t_dec = decoder_sd[name].cpu().numpy().astype(np.float32)
        denom = float(np.abs(ref_lossy_fp32).sum())
        err = float(np.abs(t_dec - ref_lossy_fp32).sum())
        per_tensor.append(
            {"name": name, "rel_err_vs_lossy_substrate": (err / denom) if denom > 1e-9 else 0.0}
        )
        abs_orig += denom
        abs_err += err
    rel_err_vs_lossy = abs_err / abs_orig if abs_orig > 1e-9 else 0.0

    # Also compute rel_err vs original fp32 weights (what scorer sees).
    abs_orig_full = 0.0
    abs_err_full = 0.0
    for name, _shape in FIXED_STATE_SCHEMA:
        t_ref = sd_ref[name].cpu().numpy().astype(np.float32)
        t_dec = decoder_sd[name].cpu().numpy().astype(np.float32)
        abs_orig_full += float(np.abs(t_ref).sum())
        abs_err_full += float(np.abs(t_dec - t_ref).sum())
    rel_err_vs_orig_fp32 = (
        abs_err_full / abs_orig_full if abs_orig_full > 1e-9 else 0.0
    )

    n_pairs = int(latents.shape[0]) if hasattr(latents, "shape") else None
    return {
        "rel_err_vs_lossy_substrate": rel_err_vs_lossy,
        "max_per_tensor_rel_err_vs_lossy": max(t["rel_err_vs_lossy_substrate"] for t in per_tensor),
        "rel_err_vs_orig_fp32": rel_err_vs_orig_fp32,
        "n_tensors_compared": len(per_tensor),
        "n_latent_pairs_decoded": n_pairs,
        "latent_dim_meta": meta.get("latent_dim"),
        "base_channels_meta": meta.get("base_channels"),
        "eval_size_meta": meta.get("eval_size"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dict", type=Path, default=DEFAULT_PR101_STATE_DICT)
    p.add_argument(
        "--frontier-archive", type=Path, default=DEFAULT_PR101_FRONTIER_ARCHIVE
    )
    p.add_argument(
        "--pr101-source-dir", type=Path, default=DEFAULT_PR101_SOURCE_DIR
    )
    p.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "experiments" / "results"
    )
    p.add_argument(
        "--predicted-low", type=float, default=DEFAULT_PREDICTED_BAND[0]
    )
    p.add_argument(
        "--predicted-high", type=float, default=DEFAULT_PREDICTED_BAND[1]
    )
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        sys.exit(f"FATAL: --state-dict not found: {args.state_dict}")
    if not args.frontier_archive.is_file():
        sys.exit(f"FATAL: --frontier-archive not found: {args.frontier_archive}")
    if not args.pr101_source_dir.is_dir():
        sys.exit(f"FATAL: --pr101-source-dir not found: {args.pr101_source_dir}")

    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_root = _resolve_output_root(args.output_root)
    build_dir = (
        output_root
        / f"cross_paradigm_admm_x_op1_finalizer_{timestamp}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_dir / "archive.zip"
    submission_dir = build_dir / "submission_dir"
    build_manifest_path = build_dir / "build_manifest.json"

    Ks = list(ADMM_PATH_B_STEP6_KS)
    print(
        f"[cross-paradigm-build] applying ADMM Ks (rms_target={ADMM_RMS_TARGET}, "
        f"lambda={ADMM_PROXY_LAMBDA:.0f})"
    )
    print(f"[cross-paradigm-build]   Ks = {Ks}")

    rebuilt, admm_stats = _build_dequantized_substrate_via_admm(args.state_dict, Ks)
    print(
        f"[cross-paradigm-build]   ADMM substrate built: rel_err_int8={admm_stats['rel_err_int8_after_admm']:.6f}"
    )

    op1_inner_blob, byte_maps, op1_stats = _encode_op1_finalizer_blob(rebuilt)
    print(
        f"[cross-paradigm-build]   Op1 inner blob: {op1_stats['op1_inner_blob_bytes']:,} B "
        f"(non-default byte_maps: {op1_stats['non_default_byte_maps']})"
    )

    decoder_section = _build_decoder_section_cplx(op1_inner_blob, byte_maps)
    print(
        f"[cross-paradigm-build]   CPLX decoder section: {len(decoder_section):,} B "
        f"(10 B CPLX header + byte_maps_json + {len(op1_inner_blob):,} B Op1 inner blob)"
    )

    pr101_inner = _read_pr101_inner_blob(args.frontier_archive)
    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)
    if len(_orig_decoder) != DECODER_BLOB_LEN:
        raise SystemExit(
            f"FATAL: PR101 frontier decoder length {len(_orig_decoder)} != expected "
            f"{DECODER_BLOB_LEN}"
        )
    if len(latent_blob) != LATENT_BLOB_LEN:
        raise SystemExit(
            f"FATAL: PR101 latent_blob length {len(latent_blob)} != expected "
            f"{LATENT_BLOB_LEN}"
        )
    print(
        f"[cross-paradigm-build]   PR101 latent_blob={len(latent_blob):,} B "
        f"sidecar_blob={len(sidecar_blob):,} B"
    )

    inner_blob = decoder_section + latent_blob + sidecar_blob
    _write_pr101_archive(inner_blob, archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[cross-paradigm-build] WROTE archive: {archive_path.relative_to(REPO_ROOT)} "
        f"size={archive_bytes:,} B sha256={archive_sha[:16]}..."
    )

    _stage_forked_submission_dir(
        submission_dir, pr101_source_dir=args.pr101_source_dir
    )
    print(
        f"[cross-paradigm-build] WROTE submission dir: {submission_dir.relative_to(REPO_ROOT)}"
    )

    print("[smoke] running CPU roundtrip ...")
    smoke = _local_smoke_roundtrip(
        archive_path,
        pr101_state_dict_path=args.state_dict,
        submission_dir=submission_dir,
    )
    print(
        f"[smoke] rel_err_vs_lossy_substrate={smoke['rel_err_vs_lossy_substrate']:.6f} "
        f"rel_err_vs_orig_fp32={smoke['rel_err_vs_orig_fp32']:.6f} "
        f"max_per_tensor_lossy={smoke['max_per_tensor_rel_err_vs_lossy']:.6f} "
        f"n_tensors={smoke['n_tensors_compared']} "
        f"n_latent_pairs={smoke['n_latent_pairs_decoded']}"
    )
    # Op1 re-quantizes fp32 → int8 with PR101's per-channel scales (which are
    # NOT the same scales we used to dequantize). The round-trip rel_err vs
    # the lossy substrate is therefore the Op1 re-quantization noise on top
    # of the lossy substrate. It should be small (<5%) but not zero.
    if smoke["rel_err_vs_lossy_substrate"] > 0.10:
        sys.exit(
            f"FATAL: roundtrip rel_err_vs_lossy_substrate "
            f"{smoke['rel_err_vs_lossy_substrate']:.4f} > 0.10; wire-format bug "
            f"or Op1 re-quantization grossly out of family"
        )
    if smoke["n_latent_pairs_decoded"] != 600:
        sys.exit(
            f"FATAL: smoke decoded n_pairs={smoke['n_latent_pairs_decoded']} != 600 "
            "(PR101 N_PAIRS); latent_blob passthrough broken"
        )

    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "lane_id": LANE_ID,
        "built_at_utc": _utc_now_iso(),
        "source_admm_manifest": ADMM_SOURCE_MANIFEST,
        "xparadigm_source_tool": XPARADIGM_SOURCE_TOOL,
        "xparadigm_decoder_only_bytes": XPARADIGM_DECODER_ONLY_BYTES,
        "admm_rms_target": ADMM_RMS_TARGET,
        "admm_lambda": ADMM_PROXY_LAMBDA,
        "admm_proxy_archive_bytes": ADMM_PROXY_ARCHIVE_BYTES,
        "admm_proxy_rel_err": ADMM_PROXY_REL_ERR,
        "rel_err_int8_after_admm": admm_stats["rel_err_int8_after_admm"],
        "rel_err_vs_lossy_substrate_smoke": smoke["rel_err_vs_lossy_substrate"],
        "rel_err_vs_orig_fp32_smoke": smoke["rel_err_vs_orig_fp32"],
        "max_per_tensor_rel_err_vs_lossy_smoke": smoke["max_per_tensor_rel_err_vs_lossy"],
        "op1_inner_blob_bytes": op1_stats["op1_inner_blob_bytes"],
        "non_default_byte_maps": op1_stats["non_default_byte_maps"],
        "effective_byte_maps_count": op1_stats["effective_byte_maps_count"],
        "cplx_decoder_section_bytes": len(decoder_section),
        "cplx_header_bytes": 10,
        "archive_relpath": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "submission_dir_relpath": str(submission_dir.relative_to(REPO_ROOT)),
        "input_state_dict": str(args.state_dict),
        "input_frontier_archive": str(args.frontier_archive),
        "input_pr101_source_dir": str(args.pr101_source_dir),
        "n_tensors": admm_stats["n_tensors"],
        "n_symbols": admm_stats["n_symbols"],
        "per_tensor_K": admm_stats["Ks"],
        "per_tensor_scale_fp32": admm_stats["per_tensor_scale_fp32"],
        "smoke_n_latent_pairs_decoded": smoke["n_latent_pairs_decoded"],
        "smoke_latent_dim_meta": smoke["latent_dim_meta"],
        "smoke_base_channels_meta": smoke["base_channels_meta"],
        "smoke_eval_size_meta": smoke["eval_size_meta"],
        "predicted_band": [args.predicted_low, args.predicted_high],
        "predicted_band_grade": "predicted",
        "evidence_grade": "[CPU-build]",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "wire_format": "CPLX1",
        "wire_format_doc": (
            "magic=CPLX[4] | section_len_le_u32[4] | bm_json_len_le_u16[2] | "
            "byte_maps_json[J] | op1_inner_blob[D-10-J] | "
            "latent_blob[15387] | sidecar_blob[remaining]"
        ),
        "wire_format_rationale": (
            "Bypasses CodecPipeline CPL1 wrapper because CPL1's json.dumps(op_state) "
            "coerces effective_byte_maps int keys to strings; decode_decoder_compact "
            "then misses on `idx in effective_byte_maps` check, falling back to "
            "DECODER_BYTE_MAPS and producing sign-flips on negzig tensors. CPLX "
            "stores byte_maps json with string keys but the inflate.py decoder "
            "rehydrates them to int keys before calling decode_decoder_compact."
        ),
        **cpu_build_proxy_guard_fields(),
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[cross-paradigm-build] manifest: {build_manifest_path.relative_to(REPO_ROOT)} "
        f"(archive={archive_bytes:,} B, sha256={archive_sha[:16]}...)"
    )
    print(
        "[cross-paradigm-build] DONE. CPU build complete. "
        "ready_for_exact_eval_dispatch=False; cuda_eval_worth_testing=True."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
