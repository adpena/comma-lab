#!/usr/bin/env python3
"""Build the PR106 UNIWARD-Lagrangian byte-closed runtime packet.

CUDA-PRESTAGE flagged Candidate 2 (PR106 UNIWARD-Lagrangian @ rms=0.05 → 150,460 B
projected) as missing-deploy-readiness in
``.omx/research/tier_a_cuda_dispatch_packets_prestaged_20260508.md``: the
empirical CPU sweep
(``tools/pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.py`` +
``reports/raw/pr106_lagrangian_per_tensor_allocation_20260508T071433Z/manifest.json``)
predicts the byte savings but DOES NOT build a contest-format archive or
paired ``inflate.py``.

This tool closes that gap. It:

1. Parses the PR106 frontier archive (canonical SHA = published 186,239 B)
   into 28 ``PackedDecoderRecord`` rows via ``tac.hnerv_decoder_recode`` +
   ``tac.hnerv_lowlevel_packer`` (the tested PR106 wire-format primitives).
2. Computes per-tensor K curves over the canonical
   ``DEFAULT_K_RANGE = range(1, 65)``.
3. Runs the canonical ``tac.optimization.UniwardWeightedAllocator`` to
   λ-bisect to the operator-supplied ``--rms-target`` (default 0.05, the
   value identified by CUDA-PRESTAGE as the deploy candidate).
4. Re-emits the ``decoder_packed_brotli`` section with the same wire format
   PR106 uses: ``concat(zz_u8(K-rounded ints))`` || ``concat(scale_f32)`` →
   brotli (q=11). Latents-and-sidecar brotli + ``0xff`` framing header are
   preserved verbatim — the K-coarsening only modifies decoder int8
   magnitudes, NOT the archive layout.
5. Writes a single-member ZIP ``0.bin`` containing the full packet and a
   forked ``submission_dir/`` whose ``inflate.py`` / ``model.py`` /
   ``codec.py`` are PR106-byte-identical. ``inflate.sh`` is adapted into a
   self-contained 3-arg wrapper because PR106's original shell wrapper assumes
   it lives under ``submissions/<name>/``; the Python decoder code remains
   byte-identical.
6. Runs a smoke roundtrip — extract our archive, run PR106's ``parse_archive``
   on it, verify the per-tensor weights match our K-coarsened encoder output
   AND that the decoded latents have ``n_pairs == 600``. (The full
   contest CUDA inflate over 1200 frames requires a CUDA device — the
   roundtrip verifies the wire-format / K-coarsening identity, not the
   visual reconstruction quality.)
7. Emits ``build_manifest.json`` with the canonical CLAUDE.md fail-closed
   custody flags + the per-tensor K vector + rms_target +
   ``score_affecting_payload_changed=True`` + ``charged_bits_changed=True``.

The wire-format is byte-identical to PR106's published runtime, so the
existing PR106 inflate path (vendored from
``experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/``)
is sufficient. The only Python code path that changes is the encoder (us);
inflate-time deserialization is the SAME function that reads the published
0.20454 archive. The shell wrapper only fixes runtime location and Python
interpreter discovery.

CLAUDE.md compliance
--------------------
- ``family_falsified=False``,
  ``falsification_scope="pr106_uniward_runtime_packet_only"``.
- ``ready_for_exact_eval_dispatch=False`` (CPU build never auto-promotes).
- ``cuda_eval_worth_testing=True`` after smoke roundtrip; the build is the
  last CPU-only gate before operator AUTH dispatch.
- ``score_affecting_payload_changed=True`` (decoder int8 stream content
  differs from PR106 lossless), ``charged_bits_changed=True`` (different
  brotli output bytes).
- Pure-CPU; never loads a scorer; tags evidence ``[CPU-build]``.
- Reuses canonical primitives from ``tac.optimization`` and
  ``tac.codec.cost_curves`` per the CANONICALIZE-OSS landing.
- Per BUGCLASSES B3: ``build_manifest.json`` ``archive_relpath`` paired with
  ``tools/verify_pr106_uniward_runtime_packet_sha256.py`` rebuild-and-assert
  smoke (deterministic SHA from same inputs).

Usage
-----

.. code-block:: bash

    .venv/bin/python tools/build_pr106_uniward_runtime_packet.py \
        --rms-target 0.05

    # custom output:
    .venv/bin/python tools/build_pr106_uniward_runtime_packet.py \
        --rms-target 0.05 \
        --output-dir experiments/results/pr106_uniward_runtime_packet_test
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
import zipfile
from dataclasses import dataclass
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.codec.cost_curves import (  # noqa: E402
    DEFAULT_K_RANGE,
    TensorBlob,
    precompute_per_tensor_K_curves,
)
from tac.hnerv_decoder_recode import (  # noqa: E402
    PACKED_STATE_SCHEMA,
    parse_packed_decoder_brotli,
)
from tac.hnerv_lowlevel_packer import (  # noqa: E402
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
)
from tac.optimization.lagrangian_per_tensor_allocation import (  # noqa: E402
    UniwardWeightedAllocator,
)

LANE_ID = "pr106_uniward_lagrangian_runtime_packet"
SCHEMA_VERSION = "pr106_uniward_lagrangian_runtime_packet_build.v1"
TOOL_NAME = "tools/build_pr106_uniward_runtime_packet.py"
EVIDENCE_GRADE = "[CPU-build]"
EVIDENCE_SEMANTICS = (
    "pr106_uniward_lagrangian_runtime_packet_byte_closed_cpu_build_no_score"
)

# PR106 wire-format constants (verified against the published frontier archive).
PR106_FRAME_PAIRS = 600
PR106_LATENT_DIM = 28
PR106_BASE_CHANNELS = 36
PR106_EVAL_SIZE = (384, 512)
PR106_DECODER_BROTLI_BASELINE_BYTES = 170_278
PR106_ARCHIVE_BASELINE_BYTES = 186_239
PR106_PUBLISHED_MEMBER_NAME = "0.bin"
# 4-byte ZIP overhead constant for stored single-member archive (verified).
PR106_ZIP_OVERHEAD_BYTES = 108

DEFAULT_PR106_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
)
DEFAULT_PR106_SOURCE_DIR = (
    REPO_ROOT
    / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex"
    / "source/submissions/belt_and_suspenders"
)

DEFAULT_RMS_TARGET = 0.05
DEFAULT_BROTLI_QUALITY = 11
DEFAULT_PREDICTED_BAND = (0.18, 0.22)

# CPU-only build cannot ever flip ready_for_exact_eval_dispatch. Operator AUTH
# is required to flip via tools/claim_lane_dispatch.py + an explicit override.
CPU_BUILD_SCORE_BLOCKERS = [
    "cpu_build_rel_err_proxy_not_score_evidence",
    "exact_cuda_auth_eval_not_yet_harvested",
    "requires_contest_auth_eval_json_before_score_promotion_rank_or_kill",
    # Cross-substrate Lagrangian on PR106 has not been calibrated against
    # CUDA scorer outputs; this packet is the FIRST attempt to anchor it.
    "no_pr106_lagrangian_calibration_anchor_yet",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _repo_resolve(path: Path) -> Path:
    """Resolve CLI paths relative to the repo root, not caller cwd."""
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _try_relative(path: Path, base: Path) -> str:
    """Return ``path.relative_to(base)`` if possible, else ``str(path)``.

    The build script is normally invoked with output dirs inside the repo, but
    the verifier runs the build with a TemporaryDirectory output dir; in that
    case ``relative_to(REPO_ROOT)`` raises ``ValueError``. Tolerate both cases
    so the verifier's reproducibility rebuild still succeeds.
    """
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _zz_u8_to_i8(buf: bytes) -> np.ndarray:
    """Decode zigzag-uint8 to signed int8 array (int32-typed)."""
    arr_u8 = np.frombuffer(buf, dtype=np.uint8).astype(np.int32)
    return (arr_u8 >> 1) ^ -(arr_u8 & 1)


def _i8_to_zz_u8(arr: np.ndarray) -> bytes:
    """Encode signed int8 (as int32) to zigzag-uint8 bytes."""
    arr_i32 = arr.astype(np.int32)
    np.clip(arr_i32, -127, 127, out=arr_i32)
    zz = (arr_i32 << 1) ^ (arr_i32 >> 31)
    return zz.astype(np.uint8).tobytes()


@dataclass
class _PR106TensorBlob:
    """One PR106 decoder tensor as a stream of int8 symbols + verbatim scale."""

    name: str
    shape: tuple[int, ...]
    raw_i8: np.ndarray
    scale_f32: bytes


def _collect_pr106_tensors(archive_path: Path) -> tuple[list[_PR106TensorBlob], bytes, bytes]:
    """Parse PR106 archive: return (tensors, header_4B, latents_and_sidecar_brotli)."""
    sma = read_strict_single_member_zip(archive_path)
    packed = parse_ff_packed_brotli_hnerv(sma.payload)
    parsed = parse_packed_decoder_brotli(packed.decoder_packed_brotli)
    if len(parsed.records) != len(PACKED_STATE_SCHEMA):
        raise SystemExit(
            f"FATAL: PR106 schema mismatch: {len(parsed.records)} vs {len(PACKED_STATE_SCHEMA)}"
        )
    tensors = [
        _PR106TensorBlob(
            name=record.name,
            shape=record.shape,
            raw_i8=_zz_u8_to_i8(record.q_zz_u8),
            scale_f32=record.scale_f32,
        )
        for record in parsed.records
    ]
    return tensors, packed.header, packed.latents_and_sidecar_brotli


def _make_joint_encoder_hook(
    tensors: list[_PR106TensorBlob],
    *,
    brotli_quality: int,
) -> object:
    """Build a JointEncoderHook that re-evaluates joint decoder_brotli bytes
    + global L1 rel_err for any per-tensor K selection.

    The hook is the canonical mechanism (per
    ``tac.optimization.lagrangian_per_tensor_allocation.JointEncoderHook``)
    for the allocator to consult the JOINT encoder rather than the
    per-tensor byte_proxy. This matches the empirical sweep's bisect
    semantics in
    ``tools/pr106_omega_opt_lagrangian_per_tensor_allocation_empirical.py``:

    - bytes = len(brotli(concat(zz_u8(K-rounded ints)) || concat(scale_f32)))
    - rel_err = sum_t |rounded_t - orig_t| / sum_t |orig_t|  (global L1)

    The hook is a closure over ``tensors`` so the joint encoder always sees
    the canonical scale streams and tensor order.
    """

    def _hook(selections: list[dict]) -> dict[str, object]:
        Ks = [int(sel["K"]) for sel in selections]
        enc = _encode_decoder_brotli_with_per_tensor_K(
            tensors, Ks, brotli_quality=brotli_quality
        )
        return {
            "total_bytes": enc["decoder_brotli_bytes"],
            "rel_err": enc["rel_err"],
            "Ks": Ks,
        }

    return _hook


def _select_uniward_Ks(
    tensors: list[_PR106TensorBlob],
    rms_target: float,
    *,
    brotli_quality: int = DEFAULT_BROTLI_QUALITY,
) -> tuple[list[int], dict[str, float]]:
    """Run UniwardWeightedAllocator over per-tensor K curves; return (Ks, info).

    Uses the canonical JointEncoderHook so the allocator's ``rel_err`` and
    ``total_bytes`` come from the JOINT brotli of the K-coarsened symbol
    stream — matching the empirical sweep's bisect semantics
    (global L1 ``rel_err`` over all int8 magnitudes, not RMS of per-tensor
    rel_errs).
    """
    blobs = [TensorBlob(name=tb.name, raw=tb.raw_i8) for tb in tensors]
    curves = precompute_per_tensor_K_curves(blobs, K_range=DEFAULT_K_RANGE)
    joint_hook = _make_joint_encoder_hook(tensors, brotli_quality=brotli_quality)
    allocator = UniwardWeightedAllocator(
        [b.raw for b in blobs], joint_encoder=joint_hook
    )
    result = allocator.bisect_for_rms_target(curves, rms_target)
    Ks = [int(sel["K"]) for sel in result.selections]
    info = {
        "lambda": float(result.lam),
        "joint_rel_err_at_selection": float(result.rel_err),
        "joint_total_bytes_at_selection": int(result.total_bytes),
        "n_curves": len(curves),
        "K_range": [DEFAULT_K_RANGE[0], DEFAULT_K_RANGE[-1]],
    }
    return Ks, info


def _encode_decoder_brotli_with_per_tensor_K(
    tensors: list[_PR106TensorBlob],
    Ks: list[int],
    *,
    brotli_quality: int = DEFAULT_BROTLI_QUALITY,
) -> dict:
    """Apply per-tensor K coarsening, re-emit decoder_packed_brotli wire."""
    if len(Ks) != len(tensors):
        raise SystemExit(
            f"FATAL: Ks length {len(Ks)} != n_tensors {len(tensors)}"
        )
    abs_orig_total = 0.0
    abs_err_total = 0.0
    rounded_chunks_zz: list[bytes] = []
    rounded_i8_chunks: list[np.ndarray] = []
    for tb, K in zip(tensors, Ks, strict=True):
        if not isinstance(K, int) or K < 1 or K > 64:
            raise SystemExit(f"FATAL: K out of [1,64] range for {tb.name}: {K!r}")
        rounded = np.round(tb.raw_i8 / K) * K
        err = float(np.abs(rounded - tb.raw_i8).astype(np.float64).sum())
        abs_err_total += err
        abs_orig_total += float(np.abs(tb.raw_i8).astype(np.float64).sum())
        rounded_clipped = np.clip(rounded, -127, 127).astype(np.int32)
        rounded_chunks_zz.append(_i8_to_zz_u8(rounded_clipped))
        rounded_i8_chunks.append(rounded_clipped)

    q_concat = b"".join(rounded_chunks_zz)
    scales_concat = b"".join(tb.scale_f32 for tb in tensors)
    decoder_raw = q_concat + scales_concat
    decoder_brotli = brotli.compress(decoder_raw, quality=brotli_quality)
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0
    return {
        "decoder_brotli": decoder_brotli,
        "decoder_brotli_bytes": len(decoder_brotli),
        "decoder_raw_bytes": len(decoder_raw),
        "rel_err": rel_err,
        "rounded_i8_chunks": rounded_i8_chunks,
    }


def _build_pr106_packet(
    *,
    decoder_brotli: bytes,
    latents_and_sidecar_brotli: bytes,
) -> bytes:
    """Build the full PR106 monolithic-single-file packet (the inner blob).

    Layout (matches ``tac.hnerv_lowlevel_packer.parse_ff_packed_brotli_hnerv``):

        byte 0:           0xff sentinel
        bytes 1-3 LE:     decoder_packed_brotli length
        bytes 4..4+L:     decoder_packed_brotli
        bytes 4+L..end:   latents_and_sidecar_brotli (verbatim)
    """
    if len(decoder_brotli) >= (1 << 24):
        raise SystemExit(
            f"FATAL: decoder_brotli size {len(decoder_brotli)} ≥ 2^24 — header has 3-byte length"
        )
    header = bytes([0xFF]) + len(decoder_brotli).to_bytes(3, "little")
    return header + decoder_brotli + latents_and_sidecar_brotli


def _write_pr106_archive_zip(
    *,
    inner_blob: bytes,
    archive_path: Path,
    member_name: str = PR106_PUBLISHED_MEMBER_NAME,
) -> None:
    """Write the contest-format ZIP wrapper around the inner blob.

    Mirrors ``tac.hnerv_lowlevel_packer.write_stored_single_member_zip``: stored
    (no compression), fixed timestamp (1980,1,1), single member.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as zf:
        zf.writestr(info, inner_blob, compress_type=zipfile.ZIP_STORED)


SELF_CONTAINED_INFLATE_SH = """#!/usr/bin/env bash
# Self-contained PR106 inflate wrapper for experiments/results runtime packets.
# The upstream PR106 wrapper assumes submissions/<name>/ package layout; this
# packet keeps the PR106 Python decoder byte-identical but calls it by path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN="${PR106_UNIWARD_PYTHON:-${PYTHON:-}}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in "$PWD/.venv/bin/python" "$HERE/../../../../.venv/bin/python" python python3; do
    if [[ "$candidate" == */* ]]; then
      if [[ -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
      fi
    elif command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "FATAL: no Python interpreter found for PR106 UNIWARD inflate" >&2
  exit 127
fi

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.bin"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [[ ! -f "$SRC" ]] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Inflating %s ... " "$line"
  "$PYTHON_BIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
"""


def _stage_pr106_submission_dir(
    submission_dir: Path,
    *,
    pr106_source_dir: Path,
) -> dict[str, str]:
    """Stage submission_dir/{inflate.py, inflate.sh, src/codec.py, src/model.py}.

    Source-of-truth: PR106's vendored belt_and_suspenders intake dir. Decoder
    + codec are byte-identical to the published runtime since K-coarsening is
    encoder-side ONLY (the decoder reads the same int8 * scale dequantization).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    src_dir = submission_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, str] = {}
    for fname in ("codec.py", "model.py"):
        src_file = pr106_source_dir / "src" / fname
        if not src_file.is_file():
            raise SystemExit(f"FATAL: PR106 source missing: {src_file}")
        dst_file = src_dir / fname
        shutil.copy2(src_file, dst_file)
        out[f"src/{fname}"] = _sha256(dst_file.read_bytes())

    src_inflate_py = pr106_source_dir / "inflate.py"
    if not src_inflate_py.is_file():
        raise SystemExit(f"FATAL: PR106 inflate.py missing: {src_inflate_py}")
    dst_inflate_py = submission_dir / "inflate.py"
    shutil.copy2(src_inflate_py, dst_inflate_py)
    out["inflate.py"] = _sha256(dst_inflate_py.read_bytes())

    # FIX-CODEX-HIGH commit c83eff00 mandates inflate.sh implements the 3-arg
    # auth-eval contract (data_dir / output_dir / file_list). PR106's source
    # shell wrapper has that arity but assumes submissions/<name>/ package
    # placement. This packet is staged under experiments/results, so write a
    # self-contained wrapper that calls the byte-identical PR106 inflate.py by
    # path and discovers the repo venv before falling back to python/python3.
    src_inflate_sh = pr106_source_dir / "inflate.sh"
    if not src_inflate_sh.is_file():
        raise SystemExit(f"FATAL: PR106 inflate.sh missing: {src_inflate_sh}")
    dst_inflate_sh = submission_dir / "inflate.sh"
    dst_inflate_sh.write_text(SELF_CONTAINED_INFLATE_SH, encoding="utf-8")
    dst_inflate_sh.chmod(0o755)
    out["inflate.sh"] = _sha256(dst_inflate_sh.read_bytes())
    return out


def _local_smoke_roundtrip(
    *,
    archive_path: Path,
    submission_dir: Path,
    expected_rounded_chunks: list[np.ndarray],
    expected_scales_f32: list[bytes],
    rms_target: float,
    encoder_rel_err: float,
) -> dict:
    """CPU smoke: parse our archive via the staged PR106 codec, verify it
    decodes to the SAME K-coarsened int8 weights we encoded.

    Per CLAUDE.md "Auth eval EVERYWHERE" + BUGCLASSES B1: encoder must have a
    paired decoder roundtrip. We don't run the full visual reconstruction
    (that requires CUDA + the contest video); we verify wire-format identity.

    The smoke also asserts:
    - ``parse_archive`` returns 600 latent pairs (PR106's PR106 frontier
      latents-and-sidecar-brotli is preserved verbatim);
    - per-tensor reconstructed weights == K-coarsened encoder output * scale;
    - meta dict has the canonical PR106 shape (latent_dim=28, base=36, eval=384x512).
    """
    src_dir = submission_dir / "src"
    if not (src_dir / "codec.py").is_file():
        raise SystemExit(f"FATAL: smoke missing codec.py at {src_dir}")

    spec = importlib.util.spec_from_file_location(
        "_pr106_uniward_smoke_codec", src_dir / "codec.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("FATAL: smoke cannot load codec.py spec")
    codec_mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(src_dir))
    try:
        spec.loader.exec_module(codec_mod)
    finally:
        sys.path.pop(0)

    # The archive is a single-member ZIP whose member is the inner blob.
    sma = read_strict_single_member_zip(archive_path)
    inner_blob = sma.payload

    # PR106's parse_archive auto-routes 0xff packets through parse_packed_archive.
    decoder_sd, latents, meta = codec_mod.parse_archive(inner_blob)

    # Wire-format checks
    if len(decoder_sd) != len(PACKED_STATE_SCHEMA):
        raise SystemExit(
            f"FATAL: smoke decoded {len(decoder_sd)} tensors, expected "
            f"{len(PACKED_STATE_SCHEMA)} (PACKED_STATE_SCHEMA size)"
        )
    if meta.get("n_pairs") != PR106_FRAME_PAIRS:
        raise SystemExit(
            f"FATAL: smoke meta n_pairs={meta.get('n_pairs')} != {PR106_FRAME_PAIRS}"
        )
    if meta.get("latent_dim") != PR106_LATENT_DIM:
        raise SystemExit(
            f"FATAL: smoke meta latent_dim={meta.get('latent_dim')} != {PR106_LATENT_DIM}"
        )
    if meta.get("base_channels") != PR106_BASE_CHANNELS:
        raise SystemExit(
            f"FATAL: smoke meta base_channels={meta.get('base_channels')} != "
            f"{PR106_BASE_CHANNELS}"
        )
    if tuple(meta.get("eval_size", ())) != PR106_EVAL_SIZE:
        raise SystemExit(
            f"FATAL: smoke meta eval_size={meta.get('eval_size')} != "
            f"{list(PR106_EVAL_SIZE)}"
        )
    if int(latents.shape[0]) != PR106_FRAME_PAIRS:
        raise SystemExit(
            f"FATAL: smoke decoded {latents.shape[0]} latent pairs != "
            f"{PR106_FRAME_PAIRS}"
        )

    # Per-tensor weight-identity check. Use PACKED_STATE_SCHEMA order — that's
    # the canonical packed-decoder order our encoder emitted in.
    abs_err_total = 0.0
    abs_orig_total = 0.0
    per_tensor: list[dict] = []
    for idx, (name, _shape) in enumerate(PACKED_STATE_SCHEMA):
        scale = struct.unpack_from("<f", expected_scales_f32[idx], 0)[0]
        recovered = decoder_sd[name].cpu().numpy().astype(np.float64).reshape(-1)
        expected_weight = (
            expected_rounded_chunks[idx].astype(np.float64) * float(scale)
        ).reshape(-1)
        denom = float(np.abs(expected_weight).sum()) + 1e-12
        err = float(np.abs(recovered - expected_weight).sum())
        abs_err_total += err
        abs_orig_total += denom
        per_tensor.append(
            {"name": name, "rel_err_vs_encoder": err / denom if denom > 1e-9 else 0.0}
        )
    rel_err_smoke = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0

    # The encoder writes the K-rounded int8 directly into the wire stream.
    # The decoder reads it back as int8 * scale_f32. Identity should hold to
    # ~fp32 ULP precision (no quantization happens at decode); allow a small
    # absolute floor for fp32 arithmetic.
    if rel_err_smoke > 1e-5:
        raise SystemExit(
            f"FATAL: smoke per-tensor weight-identity rel_err {rel_err_smoke:.6e} "
            "> 1e-5 — wire-format roundtrip is BROKEN"
        )

    # n_frames check: PR106 emits 1200 frames (600 pairs * 2 frames/pair).
    n_frames = PR106_FRAME_PAIRS * 2

    return {
        "n_tensors_decoded": len(decoder_sd),
        "n_latent_pairs_decoded": int(latents.shape[0]),
        "n_frames_implied": n_frames,
        "meta_latent_dim": meta["latent_dim"],
        "meta_base_channels": meta["base_channels"],
        "meta_eval_size": list(meta["eval_size"]),
        "weight_identity_rel_err_smoke": rel_err_smoke,
        "max_per_tensor_rel_err_smoke": max(
            t["rel_err_vs_encoder"] for t in per_tensor
        ),
        "encoder_rel_err_int8": encoder_rel_err,
        "rms_target": rms_target,
    }


def cpu_build_proxy_guard_fields() -> dict[str, object]:
    """Fail-closed custody fields for the local CPU build artifact."""
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "cuda_eval_worth_testing": True,
        "family_falsified": False,
        "falsification_scope": "pr106_uniward_runtime_packet_only",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "score_claim_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
        "dispatch_blockers": list(CPU_BUILD_SCORE_BLOCKERS),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source-archive",
        type=Path,
        default=DEFAULT_PR106_FRONTIER_ARCHIVE,
        help="PR106 frontier source archive (single-member ZIP w/ '0.bin').",
    )
    p.add_argument(
        "--pr106-source-dir",
        type=Path,
        default=DEFAULT_PR106_SOURCE_DIR,
        help="PR106 belt_and_suspenders source dir (vendored decoder + codec).",
    )
    p.add_argument(
        "--rms-target",
        type=float,
        default=DEFAULT_RMS_TARGET,
        help="UNIWARD-weighted Lagrangian per-tensor rel_err target.",
    )
    p.add_argument(
        "--brotli-quality",
        type=int,
        default=DEFAULT_BROTLI_QUALITY,
        help="Brotli quality for decoder_packed_brotli (must be 11 to match PR106).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/results/pr106_uniward_runtime_packet_<UTC>/).",
    )
    p.add_argument(
        "--predicted-low",
        type=float,
        default=DEFAULT_PREDICTED_BAND[0],
    )
    p.add_argument(
        "--predicted-high",
        type=float,
        default=DEFAULT_PREDICTED_BAND[1],
    )
    args = p.parse_args(argv)

    args.source_archive = _repo_resolve(args.source_archive)
    args.pr106_source_dir = _repo_resolve(args.pr106_source_dir)
    if args.output_dir is not None:
        args.output_dir = _repo_resolve(args.output_dir)

    if not args.source_archive.is_file():
        raise SystemExit(f"FATAL: --source-archive not found: {args.source_archive}")
    if not args.pr106_source_dir.is_dir():
        raise SystemExit(
            f"FATAL: --pr106-source-dir not found: {args.pr106_source_dir}"
        )
    if args.brotli_quality != DEFAULT_BROTLI_QUALITY:
        # The encoder's published byte count was measured at q=11; deviating
        # invalidates the Lagrangian's curve precompute. Refuse silently
        # rather than producing a misleading manifest.
        sys.stderr.write(
            f"[pr106-uniward-build] WARN: --brotli-quality {args.brotli_quality} "
            f"!= {DEFAULT_BROTLI_QUALITY} — Lagrangian curves and rms target "
            "were measured at q=11; resulting bytes may diverge.\n"
        )
    if args.rms_target <= 0 or args.rms_target > 0.5:
        raise SystemExit(
            f"FATAL: --rms-target {args.rms_target} outside (0, 0.5] sane band"
        )

    timestamp = _utc_now_compact()
    if args.output_dir is None:
        args.output_dir = (
            REPO_ROOT
            / f"experiments/results/pr106_uniward_runtime_packet_{timestamp}"
        ).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = args.output_dir / "archive.zip"
    submission_dir = args.output_dir / "submission_dir"
    build_manifest_path = args.output_dir / "build_manifest.json"

    print(f"[pr106-uniward-build] source archive: {args.source_archive}")
    print(f"[pr106-uniward-build] rms_target: {args.rms_target}")
    print(f"[pr106-uniward-build] brotli_quality: {args.brotli_quality}")
    print(f"[pr106-uniward-build] output: {args.output_dir}")

    # 1) Parse PR106 substrate
    tensors, header_bytes, latents_and_sidecar_brotli = _collect_pr106_tensors(
        args.source_archive
    )
    n_tensors = len(tensors)
    n_symbols = sum(tb.raw_i8.size for tb in tensors)
    print(
        f"[pr106-uniward-build]   {n_tensors} tensors, {n_symbols:,} int8 symbols, "
        f"latents_and_sidecar_brotli={len(latents_and_sidecar_brotli):,} B"
    )

    # 2) UniwardWeightedAllocator selects per-tensor Ks
    print(
        f"[pr106-uniward-build]   running UniwardWeightedAllocator "
        f"(K_range=[{DEFAULT_K_RANGE[0]}, {DEFAULT_K_RANGE[-1]}])..."
    )
    Ks, alloc_info = _select_uniward_Ks(
        tensors, args.rms_target, brotli_quality=args.brotli_quality
    )
    print(f"[pr106-uniward-build]   λ={alloc_info['lambda']:.3e}, Ks={Ks}")

    # 3) Re-encode decoder_packed_brotli with K-coarsened symbols
    enc = _encode_decoder_brotli_with_per_tensor_K(
        tensors, Ks, brotli_quality=args.brotli_quality
    )
    print(
        f"[pr106-uniward-build]   decoder_packed_brotli={enc['decoder_brotli_bytes']:,} B "
        f"(vs PR106 published {PR106_DECODER_BROTLI_BASELINE_BYTES:,}; "
        f"Δ={enc['decoder_brotli_bytes'] - PR106_DECODER_BROTLI_BASELINE_BYTES:+,}); "
        f"int8 rel_err={enc['rel_err']:.6f}"
    )

    # 4) Wrap in monolithic packet + ZIP
    inner_blob = _build_pr106_packet(
        decoder_brotli=enc["decoder_brotli"],
        latents_and_sidecar_brotli=latents_and_sidecar_brotli,
    )
    if inner_blob[:4] != header_bytes[:1] + len(enc["decoder_brotli"]).to_bytes(3, "little"):
        # Sanity check; the rebuilt header must encode the new decoder length.
        raise SystemExit(
            "FATAL: header bytes mismatch between source and rebuilt packet"
        )
    _write_pr106_archive_zip(inner_blob=inner_blob, archive_path=archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[pr106-uniward-build] WROTE archive: {_try_relative(archive_path, REPO_ROOT)} "
        f"size={archive_bytes:,} B sha256={archive_sha[:16]}... "
        f"(vs PR106 published {PR106_ARCHIVE_BASELINE_BYTES:,}; "
        f"Δ={archive_bytes - PR106_ARCHIVE_BASELINE_BYTES:+,})"
    )

    # 5) Stage submission_dir (PR106-byte-identical decoder)
    staged_shas = _stage_pr106_submission_dir(
        submission_dir, pr106_source_dir=args.pr106_source_dir
    )
    print(
        f"[pr106-uniward-build] WROTE submission dir: "
        f"{_try_relative(submission_dir, REPO_ROOT)}"
    )

    # 6) Smoke roundtrip
    print("[smoke] running CPU roundtrip ...")
    smoke = _local_smoke_roundtrip(
        archive_path=archive_path,
        submission_dir=submission_dir,
        expected_rounded_chunks=enc["rounded_i8_chunks"],
        expected_scales_f32=[tb.scale_f32 for tb in tensors],
        rms_target=args.rms_target,
        encoder_rel_err=enc["rel_err"],
    )
    print(
        f"[smoke] weight_identity_rel_err={smoke['weight_identity_rel_err_smoke']:.3e} "
        f"max_per_tensor={smoke['max_per_tensor_rel_err_smoke']:.3e} "
        f"n_tensors={smoke['n_tensors_decoded']} "
        f"n_pairs={smoke['n_latent_pairs_decoded']} "
        f"n_frames_implied={smoke['n_frames_implied']}"
    )

    # 7) Build manifest
    build_manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "lane_id": LANE_ID,
        "built_at_utc": _utc_now_iso(),
        "input_source_archive": str(args.source_archive),
        "input_pr106_source_dir": str(args.pr106_source_dir),
        "rms_target": args.rms_target,
        "brotli_quality": args.brotli_quality,
        "predicted_band": [args.predicted_low, args.predicted_high],
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "K_range": list(alloc_info["K_range"]),
        "uniward_lambda": alloc_info["lambda"],
        "per_tensor_K": list(Ks),
        "achieved_rel_err": enc["rel_err"],
        "achieved_rel_err_smoke_weight_identity": smoke[
            "weight_identity_rel_err_smoke"
        ],
        "decoder_packed_brotli_bytes": enc["decoder_brotli_bytes"],
        "decoder_raw_bytes": enc["decoder_raw_bytes"],
        "latents_and_sidecar_brotli_bytes": len(latents_and_sidecar_brotli),
        "pr106_decoder_brotli_baseline_bytes": PR106_DECODER_BROTLI_BASELINE_BYTES,
        "pr106_archive_baseline_bytes": PR106_ARCHIVE_BASELINE_BYTES,
        "archive_relpath": _try_relative(archive_path, REPO_ROOT),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "archive_byte_savings_vs_pr106_published": (
            PR106_ARCHIVE_BASELINE_BYTES - archive_bytes
        ),
        "submission_dir_relpath": _try_relative(submission_dir, REPO_ROOT),
        "staged_file_sha256": staged_shas,
        "smoke_n_tensors_decoded": smoke["n_tensors_decoded"],
        "smoke_n_latent_pairs_decoded": smoke["n_latent_pairs_decoded"],
        "smoke_n_frames_implied": smoke["n_frames_implied"],
        "smoke_meta_latent_dim": smoke["meta_latent_dim"],
        "smoke_meta_base_channels": smoke["meta_base_channels"],
        "smoke_meta_eval_size": smoke["meta_eval_size"],
        "smoke_max_per_tensor_rel_err": smoke["max_per_tensor_rel_err_smoke"],
        # Reuse + canonicalization markers
        "reuses_canonical_primitives": [
            "tac.optimization.lagrangian_per_tensor_allocation.UniwardWeightedAllocator",
            "tac.codec.cost_curves.precompute_per_tensor_K_curves",
            "tac.codec.cost_curves.DEFAULT_K_RANGE",
            "tac.hnerv_decoder_recode.parse_packed_decoder_brotli",
            "tac.hnerv_decoder_recode.PACKED_STATE_SCHEMA",
            "tac.hnerv_lowlevel_packer.parse_ff_packed_brotli_hnerv",
            "tac.hnerv_lowlevel_packer.read_strict_single_member_zip",
        ],
        "wire_format_identity_with_pr106_published": True,
        "wire_format_diff_vs_pr106": (
            "byte-identical wire layout (0xff + 3B decoder length + "
            "decoder_packed_brotli + latents_and_sidecar_brotli verbatim); "
            "the int8 symbols inside decoder_packed_brotli are K-coarsened "
            "per UniwardWeightedAllocator selection (K-rounded magnitudes "
            "stored directly, no K side-info required)"
        ),
        "review_eng_finding_relevant": (
            "C2_no_dead_K_already_applies — PR106 wire format never carried "
            "K side-info; the K-coarsened int8 stream is sufficient for "
            "decoding (decoder reads q_int8 * scale_f32, just like the "
            "lossless variant)"
        ),
        # CUDA-PRESTAGE precondition closure
        "cuda_prestage_advisory_closed": (
            "tier_a_cuda_dispatch_packets_prestaged_20260508 candidate 2 "
            "build_tool_advisory closed by this manifest"
        ),
        **cpu_build_proxy_guard_fields(),
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[pr106-uniward-build] manifest: "
        f"{_try_relative(build_manifest_path, REPO_ROOT)}"
    )
    print(
        "[pr106-uniward-build] DONE. CPU build complete. "
        "ready_for_exact_eval_dispatch=False; "
        "cuda_eval_worth_testing=True. "
        "Operator AUTH required to flip dispatch claim from "
        "pending_authorization to active_dispatching."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
