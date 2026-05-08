#!/usr/bin/env python3
"""lossy_coarsening_analytical — FULL Lightning T4 [contest-CUDA] auth eval.

Takes the PR101 frontier archive, applies the
``tools/pr101_lossy_coarsening_analytical.py`` codec to the decoder section,
builds a contest-CUDA-compliant single-file archive (per the
``feedback_pr101_archive_is_monolithic_single_file_20260508`` finding), then
dispatches to Lightning T4 (g4dn.2xlarge) to run ``inflate.sh`` +
``upstream/evaluate.py --device cuda`` against the new archive.

Source-of-finding
-----------------
``feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508`` —
the analytical per-tensor K-search produced 156,344 B archive at 3.86%
rel_err, beating the brotli baseline (178,144 B) AND every neural codec
attempt. Predicted score 0.189 [predicted] vs current frontier 0.20406 →
beats the 0.190 leaderboard band IF the rel_err doesn't blow distortion.

The 3.86% rel_err is in the CONDITIONAL band (~10-30% score regression
possible) — this dispatch is the ONLY way to know whether the score
actually drops or distortion-cost cancels byte-savings.

Architecture
------------
Unlike ``arch_shrink_x0.4_lightning_full.py`` which trains a renderer on
T4, this tool does **CPU-only build LOCALLY**, then dispatches a small
auth-eval-only Lightning job. The build consists of:

1. Read PR101 decoder state_dict from
   ``experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt``
   (the canonical PR101 weights).
2. Apply per-tensor K-search at the chosen rel_err budget (default 0.05).
3. Encode as: ``[uint32 LE len_decoder][28 B per-tensor K][brotli(flat int8s)]
   [latent_blob ORIGINAL][sidecar_blob ORIGINAL]`` — preserves PR101's
   latent + sidecar bytes verbatim, only swaps the decoder section with
   the lossy-coarsening-encoded form.
4. Pack as single-file ZIP with member ``x`` (PR101 monolithic format).
5. Stage a forked submission directory with custom ``inflate.py`` that
   parses the new format (uint32 length prefix → decoder section → offset
   to latent_blob).
6. Dispatch a Lightning Studio Job that runs:
   - ``contest_auth_eval.py --archive <archive> --inflate-sh <forked-inflate.sh>
     --device cuda``

CLAUDE.md compliance
--------------------
- INFLATE_TORCH_SPEC=cu124 (driver<580 cu13 wheel CPU-fallback trap)
- claim_lane_dispatch.py BEFORE submitting (cross-agent coordination)
- ``platform=lightning`` lowercase canonical
- Score tagged ``[contest-CUDA]`` ONLY by the harvester after the auth-eval
  JSON parses with a numeric score AND the archive bytes match the staged
  archive size.
- Cost cap $5 (single auth eval on T4, no training; predicted 10-30 min).

Usage
-----
.. code-block:: bash

    .venv/bin/python experiments/lossy_coarsening_lightning_cuda_test.py \\
        --rel-err-budget 0.05 \\
        --machine g4dn.2xlarge \\
        --predicted-low 0.18 --predicted-high 0.22

Outputs
-------
- Lightning Studio Job (status visible in
  https://lightning.ai/<user>/<teamspace>/studios/<studio>)
- Local-built archive at
  ``experiments/results/lossy_coarsening_<timestamp>/archive.zip``
- Forked submission at
  ``experiments/results/lossy_coarsening_<timestamp>/submission_dir/``
- ``.omx/state/lightning_active_jobs.json`` row with job_name + paths
- Dispatch claim row in ``.omx/state/active_lane_dispatch_claims.md``

Build details
-------------
The archive layout used here is BACKWARD-INCOMPATIBLE with vanilla PR101
inflate (which assumes fixed-offset 162,164-byte decoder slot). The forked
``inflate.py`` reads a 4-byte length prefix, then the per-tensor K side-info,
then the brotli'd flat int8 stream. It reconstructs the int32 quantized
symbols by multiplying-back per-tensor K, then dequantizes via PR101's
``_quantize_tensor`` inverse (per-tensor fp16 scale stored alongside).

Wire format (inner blob, before ZIP-stored):

    +--------------------------------------------+
    | uint32 LE: decoder_section_total_bytes (D) |
    +--------------------------------------------+
    | byte * 28: per_tensor_K (uint8 each)       |
    +--------------------------------------------+
    | byte * 56: per_tensor_fp16_scale (LE half) |
    +--------------------------------------------+
    | byte * (D - 28 - 56 - 4): brotli(int8s)    |
    +--------------------------------------------+
    | byte * 15387: latent_blob (PR101 ORIGINAL) |
    +--------------------------------------------+
    | byte * remaining: sidecar_blob (ORIGINAL)  |
    +--------------------------------------------+

The fp16 scales are needed to dequantize back to fp32 weights (PR101 stores
them implicitly per-tensor inside the brotli stream, but our format stores
them explicitly to keep the brotli payload pure int8).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))  # so `import tools.pr101_lossy_coarsening_analytical` resolves

from tac.deploy.lightning.defaults import (
    DEFAULT_LIGHTNING_REMOTE_PACT,
    default_remote_pact,
    default_ssh_target,
    default_studio,
    default_teamspace,
    default_user,
)

LANE_ID = "lossy_coarsening_analytical_cuda"
INFLATE_TORCH_SPEC = "torch==2.5.1+cu124"
UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
UV_INDEX_STRATEGY = "unsafe-best-match"
DEFAULT_MACHINE = "g4dn.2xlarge"  # AWS T4
DEFAULT_MAX_RUNTIME_SEC = 90 * 60  # 90 min cap (auth eval ~ 30 min)
DEFAULT_BUDGET_CAP_USD = 5.0  # operator-authorized cap for THIS dispatch
DEFAULT_REL_ERR_BUDGET = 0.05  # the empirically-validated CONDITIONAL-band threshold
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
LIGHTNING_ACTIVE_JOBS_PATH = REPO_ROOT / ".omx" / "state" / "lightning_active_jobs.json"


def _utc_now_iso() -> str:
    return dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_plus_minutes(minutes: int) -> str:
    return (dt.datetime.now(tz=dt.UTC) + dt.timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _job_name() -> str:
    ts = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"lossy-coarsening-cuda-{ts}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Stage A: CPU-side build of the contest-CUDA archive + forked inflate.
# ---------------------------------------------------------------------------


def _read_pr101_inner_blob(archive_path: Path) -> bytes:
    """Extract the single inner ``x`` member from a PR101 archive."""
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if names != ["x"]:
            raise ValueError(
                f"PR101 archive {archive_path} has members {names!r}; expected ['x']"
            )
        with zf.open("x") as fp:
            return fp.read()


def _split_pr101_inner_blob(blob: bytes) -> tuple[bytes, bytes, bytes]:
    """Split (decoder, latent, sidecar) per PR101 fixed offsets."""
    from tac.pr101_split_brotli_codec import DECODER_BLOB_LEN, LATENT_BLOB_LEN

    if len(blob) < DECODER_BLOB_LEN + LATENT_BLOB_LEN:
        raise ValueError(
            f"inner blob length {len(blob)} < required minimum "
            f"{DECODER_BLOB_LEN + LATENT_BLOB_LEN} for PR101 layout"
        )
    decoder = blob[:DECODER_BLOB_LEN]
    latent = blob[DECODER_BLOB_LEN:DECODER_BLOB_LEN + LATENT_BLOB_LEN]
    sidecar = blob[DECODER_BLOB_LEN + LATENT_BLOB_LEN:]
    return decoder, latent, sidecar


def _build_lossy_decoder_section(
    state_dict_path: Path, *, rel_err_budget: float, brotli_quality: int = 11
) -> dict:
    """Apply per-tensor K-search to PR101 weights; produce wire-format bytes.

    Wire format of the decoder section (returned in the ``decoder_bytes`` key):

        uint32 LE: total_section_bytes (D, including this prefix)
        bytes 28: per-tensor K (uint8)
        bytes 56: per-tensor fp16 scale (LE half)
        bytes (D - 4 - 28 - 56): brotli(concat(rounded_int8s))

    Returns a dict with keys: decoder_bytes, per_tensor_K, per_tensor_scale,
    rel_err, n_tensors, n_symbols, brotli_payload_bytes.
    """
    import brotli
    import numpy as np
    import torch

    from tac.pr101_split_brotli_codec import FIXED_STATE_SCHEMA, N_QUANT, _quantize_tensor
    from tools.pr101_lossy_coarsening_analytical import find_best_K_for_tensor

    if not state_dict_path.is_file():
        raise SystemExit(f"FATAL: PR101 state_dict not found: {state_dict_path}")

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    n_tensors = len(FIXED_STATE_SCHEMA)
    Ks: list[int] = []
    scales_fp16: list[float] = []
    rounded_chunks: list[np.ndarray] = []
    abs_orig_total = 0.0
    abs_err_total = 0.0
    n_symbols = 0
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        # PR101's _QuantizedTensor stores the scale as a float; PR101 itself
        # rounds it to fp16 when packing into the blob. Match that here so
        # the round-trip dequantize matches what PR101 itself does.
        scale_fp16 = float(np.float16(qt.scale))
        K, _ = find_best_K_for_tensor(symbols_i32, budget=rel_err_budget)
        rounded = np.round(symbols_i32 / K) * K
        rounded_clipped = rounded.clip(-127, 127)
        abs_orig_total += float(np.abs(symbols_i32).astype(np.float64).sum())
        abs_err_total += float(np.abs(rounded_clipped - symbols_i32).astype(np.float64).sum())
        rounded_chunks.append(rounded_clipped.astype(np.int8))
        Ks.append(K)
        scales_fp16.append(scale_fp16)
        n_symbols += int(symbols_i32.size)
    if len(Ks) != n_tensors:
        raise RuntimeError(
            f"K-list length {len(Ks)} != n_tensors {n_tensors}; FIXED_STATE_SCHEMA changed?"
        )
    flat = np.concatenate(rounded_chunks).tobytes()
    brotli_payload = brotli.compress(flat, quality=brotli_quality, lgwin=22, lgblock=24)
    rel_err = abs_err_total / abs_orig_total if abs_orig_total > 1e-9 else 0.0

    K_bytes = bytes(Ks)
    # Pack scales as LE half (fp16). numpy view→bytes in LE.
    scale_arr = np.array(scales_fp16, dtype=np.float16)
    if not scale_arr.dtype.isnative or sys.byteorder != "little":
        scale_bytes = scale_arr.astype("<f2").tobytes()
    else:
        scale_bytes = scale_arr.tobytes()

    section_no_prefix = K_bytes + scale_bytes + brotli_payload
    section_total = 4 + len(section_no_prefix)
    prefix = struct.pack("<I", section_total)
    decoder_bytes = prefix + section_no_prefix
    if len(decoder_bytes) != section_total:
        raise RuntimeError(
            f"decoder section length mismatch: declared {section_total}, actual {len(decoder_bytes)}"
        )
    return {
        "decoder_bytes": decoder_bytes,
        "per_tensor_K": Ks,
        "per_tensor_scale_fp16": scales_fp16,
        "rel_err": rel_err,
        "n_tensors": n_tensors,
        "n_symbols": n_symbols,
        "brotli_payload_bytes": len(brotli_payload),
        "K_bytes": len(K_bytes),
        "scale_bytes": len(scale_bytes),
        "section_total_bytes": section_total,
    }


def _build_inner_blob(
    decoder_bytes: bytes, latent_blob: bytes, sidecar_blob: bytes
) -> bytes:
    """Concatenate the new decoder section + original latent + sidecar."""
    return decoder_bytes + latent_blob + sidecar_blob


def _write_pr101_archive(inner_blob: bytes, archive_path: Path) -> None:
    """Pack inner blob as single-file ZIP with member ``x`` (PR101 layout).

    Uses ZIP_STORED (the inner blob is already entropy-coded; deflate adds
    overhead with no compression gain) and a fixed timestamp for determinism.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo(filename="x")
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(info, inner_blob)


# Forked inflate.py source — lives inside the staged submission directory.
# The decoder inverse derives FIXED_STATE_SCHEMA dynamically from
# ``model.HNeRVDecoder().state_dict()`` to keep schema in lockstep with the
# vendored model module (which is byte-faithful to PR101 src/model.py).
FORKED_INFLATE_PY = '''#!/usr/bin/env python
"""Forked PR101 inflate for lossy_coarsening_analytical archive.

Wire format (inner blob, single ZIP member 'x'):
    +--------------------------------------------+
    | uint32 LE: decoder_section_total_bytes (D) |
    +--------------------------------------------+
    | byte * 28: per_tensor_K (uint8 each)       |
    +--------------------------------------------+
    | byte * 56: per_tensor_fp16_scale (LE half) |
    +--------------------------------------------+
    | byte * (D - 4 - 28 - 56): brotli(int8s)    |
    +--------------------------------------------+
    | byte * 15387: latent_blob (PR101 ORIGINAL) |
    +--------------------------------------------+
    | byte * remaining: sidecar_blob (ORIGINAL)  |
    +--------------------------------------------+

The decoder section is decoded by:
1. Read uint32 prefix = D
2. Read 28 bytes K[i], 56 bytes scale_fp16[i]
3. brotli-decode the remaining (D - 88) bytes -> flat int8 array
4. Split flat into per-tensor int8 chunks per FIXED_STATE_SCHEMA shapes
5. Use each chunk directly as the recovered coarsened q_i8 (still bounded -127..127)
6. Apply per-tensor fp16 scale to recover float weights:
   recovered_fp32 = recovered_q_i8.astype(fp32) * scale_fp16

Latent + sidecar use the original PR101 codec functions.
"""
import struct
import sys
from pathlib import Path

import brotli
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

# Must match the ENCODER's FIXED_STATE_SCHEMA exactly. Derive it from the
# vendored PR101 model instead of freezing a second hand-copied schema here.
def _fixed_state_schema():
    probe = HNeRVDecoder(
        latent_dim=LATENT_DIM,
        base_channels=BASE_CHANNELS,
        eval_size=EVAL_SIZE,
    )
    return tuple((name, tuple(t.shape)) for name, t in probe.state_dict().items())


FIXED_STATE_SCHEMA = _fixed_state_schema()
N_TENSORS = len(FIXED_STATE_SCHEMA)
K_SECTION_BYTES = N_TENSORS  # 28 bytes
SCALE_SECTION_BYTES = N_TENSORS * 2  # fp16 = 2 bytes each = 56 bytes
PREFIX_BYTES = 4  # uint32 LE


CAMERA_H, CAMERA_W = 874, 1164


def parse_lossy_archive(archive_bytes):
    if len(archive_bytes) < PREFIX_BYTES + K_SECTION_BYTES + SCALE_SECTION_BYTES + LATENT_BLOB_LEN:
        raise ValueError(
            f"archive too short ({len(archive_bytes)} bytes) for lossy_coarsening format"
        )
    section_total = struct.unpack("<I", archive_bytes[:PREFIX_BYTES])[0]
    if section_total < PREFIX_BYTES + K_SECTION_BYTES + SCALE_SECTION_BYTES:
        raise ValueError(
            f"decoder_section_total ({section_total}) < minimum {PREFIX_BYTES + K_SECTION_BYTES + SCALE_SECTION_BYTES}"
        )
    if section_total > len(archive_bytes) - LATENT_BLOB_LEN:
        raise ValueError(
            f"decoder_section_total ({section_total}) leaves no room for "
            f"latent_blob {LATENT_BLOB_LEN}"
        )

    K_start = PREFIX_BYTES
    K_end = K_start + K_SECTION_BYTES
    Ks = list(archive_bytes[K_start:K_end])
    scale_start = K_end
    scale_end = scale_start + SCALE_SECTION_BYTES
    scales_fp16 = np.frombuffer(archive_bytes[scale_start:scale_end], dtype="<f2")
    if scales_fp16.size != N_TENSORS:
        raise ValueError(
            f"scale section size {scales_fp16.size} != N_TENSORS {N_TENSORS}"
        )

    brotli_start = scale_end
    brotli_end = section_total
    brotli_payload = archive_bytes[brotli_start:brotli_end]
    flat_int8 = np.frombuffer(brotli.decompress(brotli_payload), dtype=np.int8)

    decoder_sd = {}
    cursor = 0
    for idx, (name, shape) in enumerate(FIXED_STATE_SCHEMA):
        nelem = 1
        for d in shape:
            nelem *= d
        if cursor + nelem > flat_int8.size:
            raise ValueError(
                f"flat_int8 underflow at tensor {idx} ({name}): "
                f"need {nelem}, have {flat_int8.size - cursor}"
            )
        chunk = flat_int8[cursor:cursor + nelem].astype(np.int32)
        # The stream stores the K-coarsened q_i8 value directly. K remains
        # charged side-info for audit/reproducibility; multiplying here would
        # apply the coarsening twice.
        reconstructed_q = chunk
        # Dequantize: weight_fp32 = q_i8 * scale_fp16
        weight_fp32 = (reconstructed_q.astype(np.float32) * float(scales_fp16[idx]))
        decoder_sd[name] = torch.from_numpy(weight_fp32.reshape(shape).copy())
        cursor += nelem
    if cursor != flat_int8.size:
        raise ValueError(
            f"flat_int8 leftover {flat_int8.size - cursor} bytes after consuming all tensors"
        )

    latent_start = section_total
    latent_end = latent_start + LATENT_BLOB_LEN
    latent_blob = archive_bytes[latent_start:latent_end]
    sidecar_blob = archive_bytes[latent_end:]
    if not latent_blob:
        raise ValueError("missing latent_blob in lossy archive")

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
    decoder_sd, latents, meta = parse_lossy_archive(archive_bytes)

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


# Forked inflate.sh — same structure as PR101's, points to forked inflate.py.
FORKED_INFLATE_SH = '''#!/usr/bin/env bash
# Forked PR101 inflate.sh for lossy_coarsening_analytical lane.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${1:?data dir required}"
OUTPUT_DIR="${2:?output dir required}"
FILE_LIST="${3:?file list required}"

mkdir -p "$OUTPUT_DIR"

PYBIN="${PYTHON:-python}"

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
  "$PYBIN" "$HERE/inflate.py" "$SRC" "$DST"
done < "$FILE_LIST"
'''


def _stage_forked_submission_dir(
    submission_dir: Path,
    *,
    pr101_source_dir: Path,
) -> None:
    """Copy PR101's src/ (codec.py, model.py) and write forked inflate.py + .sh.

    The forked inflate imports from PR101's vendored ``codec`` and ``model``
    modules for latent + sidecar decoding (these are byte-faithful to PR101).
    Only the decoder section is replaced; the rest of the inflate path is
    PR101's authored code.
    """
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

    inflate_py = submission_dir / "inflate.py"
    inflate_py.write_text(FORKED_INFLATE_PY, encoding="utf-8")
    inflate_sh = submission_dir / "inflate.sh"
    inflate_sh.write_text(FORKED_INFLATE_SH, encoding="utf-8")
    inflate_sh.chmod(0o755)
    inflate_py.chmod(0o755)


def _local_roundtrip_smoke(
    archive_path: Path, *, pr101_state_dict_path: Path
) -> dict:
    """CPU roundtrip smoke: parse the lossy archive, dequantize, compare to
    original PR101 weights.

    This is a structural-correctness gate (tagged ``[CPU-smoke]``, NOT
    ``[contest-CUDA]``). Verifies the wire format round-trips before paying for
    GPU. Reports rel_err vs the original fp32 weights (NOT vs the
    int8-quantized symbols, which is the metric the build phase reports).
    """
    import importlib.util

    import numpy as np
    import torch

    from tac.pr101_split_brotli_codec import (
        FIXED_STATE_SCHEMA as ENCODER_SCHEMA,
    )
    from tac.pr101_split_brotli_codec import (
        N_QUANT,
        _quantize_tensor,
    )

    spec_path = archive_path.parent / "submission_dir" / "inflate.py"
    if not spec_path.is_file():
        raise SystemExit(
            f"FATAL: forked inflate.py missing for smoke roundtrip: {spec_path}"
        )
    spec = importlib.util.spec_from_file_location("forked_inflate", spec_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"FATAL: cannot load forked inflate spec from {spec_path}")
    forked_inflate = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(spec_path.parent / "src"))
    try:
        spec.loader.exec_module(forked_inflate)
    finally:
        sys.path.pop(0)

    inner = _read_pr101_inner_blob(archive_path)
    decoder_sd, _latents, _meta = forked_inflate.parse_lossy_archive(inner)

    sd_ref = torch.load(
        pr101_state_dict_path, map_location="cpu", weights_only=False
    )
    abs_orig = 0.0
    abs_err = 0.0
    per_tensor: list[dict] = []
    for name, _shape in ENCODER_SCHEMA:
        t_ref = sd_ref[name].cpu().numpy().astype(np.float32)
        t_dec = decoder_sd[name].cpu().numpy().astype(np.float32)
        if t_ref.shape != t_dec.shape:
            raise SystemExit(
                f"FATAL: shape mismatch on {name}: ref={t_ref.shape} decoded={t_dec.shape}"
            )
        # The encoder quantizes to int8 first (lossy by N_QUANT), then K-coarsens.
        # Compare to the ENCODER's quantized-then-dequantized form for fairness:
        # recovered_fp32 = q_i8 * scale_fp16 (matches what _quantize_tensor + inverse does).
        qt_ref = _quantize_tensor(name, sd_ref[name], n_quant=N_QUANT)
        ref_quantized = (
            qt_ref.q_i8.astype(np.float32) * float(np.float16(qt_ref.scale))
        ).reshape(t_ref.shape)
        denom_q = float(np.abs(ref_quantized).sum())
        err_q = float(np.abs(t_dec - ref_quantized).sum())
        per_tensor.append({
            "name": name,
            "rel_err_vs_quantized": (err_q / denom_q) if denom_q > 1e-9 else 0.0,
        })
        abs_orig += denom_q
        abs_err += err_q
    rel_err = abs_err / abs_orig if abs_orig > 1e-9 else 0.0
    return {
        "rel_err_vs_quantized_fp32": rel_err,
        "n_tensors_compared": len(per_tensor),
        "max_per_tensor_rel_err": max(t["rel_err_vs_quantized"] for t in per_tensor),
    }


# ---------------------------------------------------------------------------
# Stage B: Lightning T4 dispatch (auth-eval only).
# ---------------------------------------------------------------------------


def build_remote_command(
    *,
    job_name: str,
    remote_pact: str,
    archive_relpath: str,
    submission_dir_relpath: str,
) -> str:
    """Construct the bash command that runs auth-eval on Lightning T4.

    Unlike ``arch_shrink_x0.4_lightning_full.py`` this does NO training — the
    archive is pre-built locally and staged via ``lightning_repro_workspace.py``.
    """
    output_subdir = f"experiments/results/lightning_batch/{job_name}"
    auth_eval_dir = f"{output_subdir}/auth_eval_work"
    auth_eval_log = f"{output_subdir}/auth_eval.log"
    auth_eval_result_json = f"{output_subdir}/contest_auth_eval.json"

    return f"""set -euo pipefail
cd {remote_pact}

mkdir -p {output_subdir}
PYBIN=/teamspace/studios/this_studio/pact/.venv/bin/python
WORKSPACE={remote_pact}
export PYTHONHASHSEED=1234
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$WORKSPACE/src:$WORKSPACE/upstream:$WORKSPACE:${{PYTHONPATH:-}}"
export TAC_UPSTREAM_DIR="$WORKSPACE/upstream"
export INFLATE_TORCH_SPEC={INFLATE_TORCH_SPEC}
export UV_EXTRA_INDEX_URL={UV_EXTRA_INDEX_URL}
export UV_INDEX_STRATEGY={UV_INDEX_STRATEGY}

# AppleDouble cleanup before any rsync-staged paths are read.
find "$WORKSPACE" -name '._*' -type f -delete 2>/dev/null || true

LOG_DIR="$WORKSPACE/{output_subdir}"
mkdir -p "$LOG_DIR"

log() {{ echo "[lossy-coarsening] $(date -u +%FT%TZ) $*" | tee -a "$LOG_DIR/run.log"; }}

# Stage 0: GPU presence check.
log "=== Stage 0: GPU presence check ==="
"$PYBIN" -c "
import torch, sys
if not torch.cuda.is_available():
    print('FATAL: torch.cuda.is_available()=False on Lightning T4', file=sys.stderr)
    sys.exit(2)
name = torch.cuda.get_device_name(0)
mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f'OK: GPU={{name}} mem={{mem_gb}}GB cuda_version={{torch.version.cuda}}')
"

PROVENANCE="$LOG_DIR/provenance.json"
HEARTBEAT="$LOG_DIR/heartbeat.log"
GIT_HASH=$(cd "$WORKSPACE" && git rev-parse HEAD 2>/dev/null || echo "no-git")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
"$PYBIN" -c "
import json, time, torch
prov = {{
    'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'git_hash': '$GIT_HASH',
    'gpu_name': '$GPU_NAME',
    'driver_version': '$DRIVER_VER',
    'torch_version': torch.__version__,
    'cuda_version': getattr(torch.version, 'cuda', None),
    'cuda_available': torch.cuda.is_available(),
    'lane_id': '{LANE_ID}',
    'job_name': '{job_name}',
    'inflate_torch_spec': '{INFLATE_TORCH_SPEC}',
    'archive_relpath': '{archive_relpath}',
    'submission_dir_relpath': '{submission_dir_relpath}',
}}
with open('$PROVENANCE', 'w') as f:
    json.dump(prov, f, indent=2)
print('provenance:', json.dumps(prov))
"
( while true; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1 | tr '\\n' ' ')
    echo "[$(date -u +%FT%TZ)] lane={LANE_ID} gpu=$GPU" >> "$HEARTBEAT"
    sleep 60
  done ) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# Stage 1: lazily install nvidia-dali if absent (matches arch_shrink pattern).
log "=== Stage 1: nvidia-dali probe ==="
"$PYBIN" -c "
import importlib.util, subprocess, sys
if importlib.util.find_spec('nvidia.dali') is None:
    print('Installing nvidia-dali-cuda130 lazily...', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                    '--extra-index-url', 'https://pypi.nvidia.com',
                    'nvidia-dali-cuda130'], check=True)
import nvidia.dali as dali
print(f'DALI version: {{dali.__version__}}')
"

# Stage 2: contest_auth_eval [contest-CUDA] on the staged lossy archive.
ARCHIVE="$WORKSPACE/{archive_relpath}"
INFLATE_SH="$WORKSPACE/{submission_dir_relpath}/inflate.sh"
if [ ! -f "$ARCHIVE" ]; then
    log "FATAL: lossy archive missing on remote: $ARCHIVE"
    exit 4
fi
if [ ! -f "$INFLATE_SH" ]; then
    log "FATAL: forked inflate.sh missing on remote: $INFLATE_SH"
    exit 5
fi
ARCHIVE_BYTES=$(stat -c '%s' "$ARCHIVE" 2>/dev/null || stat -f '%z' "$ARCHIVE")
log "  archive: $ARCHIVE bytes=$ARCHIVE_BYTES"
log "  inflate: $INFLATE_SH"

log "=== Stage 2: contest_auth_eval [contest-CUDA] ==="
rm -rf "$WORKSPACE/{auth_eval_dir}"
"$PYBIN" -u experiments/contest_auth_eval.py \\
    --archive "$ARCHIVE" \\
    --inflate-sh "$INFLATE_SH" \\
    --upstream-dir upstream \\
    --device cuda \\
    --keep-work-dir \\
    --work-dir "$WORKSPACE/{auth_eval_dir}" 2>&1 | tee "{auth_eval_log}"
PIPE_RC=("${{PIPESTATUS[@]}}")
if [ "${{PIPE_RC[0]}}" -ne 0 ]; then
    log "FATAL: contest_auth_eval rc=${{PIPE_RC[0]}}"
    exit "${{PIPE_RC[0]}}"
fi

# Stage 3: capture RESULT_JSON.
"$PYBIN" -c "
import json, re
log_path = '{auth_eval_log}'
out_path = '{auth_eval_result_json}'
with open(log_path) as f:
    text = f.read()
m = re.search(r'RESULT_JSON\\s*(\\{{.*?\\}})', text, re.DOTALL)
if not m:
    raise SystemExit('FATAL: no RESULT_JSON in auth_eval log')
data = json.loads(m.group(1))
data['archive_path'] = '$ARCHIVE'
data['archive_bytes'] = int('$ARCHIVE_BYTES')
data['lane_id'] = '{LANE_ID}'
data['job_name'] = '{job_name}'
data['evidence_grade'] = '[contest-CUDA]'
data['archive_relpath'] = '{archive_relpath}'
data['submission_dir_relpath'] = '{submission_dir_relpath}'
with open(out_path, 'w') as f:
    json.dump(data, f, indent=2)
print('contest_auth_eval result:', json.dumps(data))
"

log "=== DONE lossy_coarsening_lightning_cuda_test ==="
""".strip()


def stage_workspace(
    *,
    job_name: str,
    archive_path: Path,
    submission_dir: Path,
    ssh_target: str,
    remote_pact: str,
) -> Path:
    """Stage src/, experiments/, submissions/, scripts/, upstream/, tools/ +
    the locally-built lossy archive and forked submission directory.
    """
    manifest_dir = REPO_ROOT / "experiments" / "results" / "lightning_batch" / job_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = manifest_dir / "source_manifest.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "lightning_repro_workspace.py"),
        "--remote",
        ssh_target,
        "--remote-pact",
        remote_pact,
        "--run-id",
        job_name,
        "--manifest-out",
        str(manifest_out),
        "--source",
        "src",
        "--source",
        "experiments",
        "--source",
        "submissions",
        "--source",
        "scripts",
        "--source",
        "upstream",
        "--source",
        "tools",
        "--source",
        "pyproject.toml",
        "--artifact",
        str(archive_path),
        "--artifact",
        str(submission_dir),
        "--requirements-mode",
        "no-install",
        "--no-install",
        "--ssh-connect-timeout",
        "30",
    ]
    print(f"[stage] {' '.join(cmd[:6])} ... ({len(cmd)} args total)")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: lightning_repro_workspace.py failed (rc={result.returncode})")
    return manifest_out


def claim_lane(*, job_name: str, force_claim: bool, force_claim_reason: str | None) -> None:
    notes = (
        f"FULL Lightning T4 [contest-CUDA] auth eval of "
        f"lossy_coarsening_analytical PR101 substrate via "
        f"experiments/lossy_coarsening_lightning_cuda_test.py {_utc_now_iso()}"
    )
    if force_claim:
        if not force_claim_reason:
            sys.exit("FATAL: --force-claim requires --force-claim-reason")
        notes = f"{notes}; force-claim: {force_claim_reason}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        LANE_ID,
        "--agent",
        "claude_lab",
        "--platform",
        "lightning",
        "--instance-job-id",
        job_name,
        "--predicted-eta-utc",
        _utc_plus_minutes(90),
        "--status",
        "active_dispatching",
        "--notes",
        notes,
        "--ttl-hours",
        "4",
    ]
    if force_claim:
        cmd += ["--force"]
    print(f"[claim] platform=lightning lane={LANE_ID} job={job_name}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        sys.exit(f"FATAL: claim_lane_dispatch.py failed (rc={result.returncode})")


def submit_lightning_job(
    *,
    job_name: str,
    machine: str,
    command: str,
    teamspace: str,
    studio: str,
    user: str,
    max_runtime_sec: int,
    dry_run: bool,
) -> dict[str, object]:
    """Submit a Lightning Studio Job via lightning_sdk.Job.run."""
    if dry_run:
        return {
            "dry_run": True,
            "command_preview": command[:400],
            "would_submit_machine": machine,
        }

    os.environ.setdefault("LIGHTNING_DISABLE_VERSION_CHECK", "1")
    try:
        from lightning_sdk import Job  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - env-dependent
        sys.exit(
            f"FATAL: lightning_sdk import failed; install with `uv pip install lightning-sdk` ({exc})"
        )

    print(f"[submit] Job.run name={job_name} machine={machine} studio={studio}")
    env = {
        "INFLATE_TORCH_SPEC": INFLATE_TORCH_SPEC,
        "UV_EXTRA_INDEX_URL": UV_EXTRA_INDEX_URL,
        "UV_INDEX_STRATEGY": UV_INDEX_STRATEGY,
    }
    job = Job.run(
        name=job_name,
        machine=machine,
        command=command,
        studio=studio,
        teamspace=teamspace,
        user=user,
        env=env,
        interruptible=False,
        max_runtime=max_runtime_sec,
    )
    return {
        "name": getattr(job, "name", job_name),
        "machine": machine,
        "studio": studio,
        "teamspace": teamspace,
        "user": user,
        "status_at_submit": str(getattr(job, "status", "unknown")),
    }


def persist_active_job(
    *,
    job_name: str,
    machine: str,
    submit_result: dict[str, object],
    manifest_path: Path,
    archive_path: Path,
    archive_bytes: int,
    rel_err_budget: float,
    rel_err_actual: float,
    predicted_band: tuple[float, float],
    smoke_roundtrip: dict[str, object] | None,
) -> None:
    """Append a row to .omx/state/lightning_active_jobs.json."""
    LIGHTNING_ACTIVE_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LIGHTNING_ACTIVE_JOBS_PATH.exists():
        existing = json.loads(LIGHTNING_ACTIVE_JOBS_PATH.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []
    record = {
        "schema_version": "lightning_active_jobs.v1",
        "lane_id": LANE_ID,
        "job_name": job_name,
        "submitted_at_utc": _utc_now_iso(),
        "machine": machine,
        "rel_err_budget": rel_err_budget,
        "rel_err_actual_int8": rel_err_actual,
        "rel_err_actual_fp32_smoke": (
            smoke_roundtrip.get("rel_err_vs_quantized_fp32")
            if smoke_roundtrip
            else None
        ),
        "archive_path_local": str(archive_path.relative_to(REPO_ROOT))
        if archive_path.is_relative_to(REPO_ROOT)
        else str(archive_path),
        "archive_bytes": archive_bytes,
        "predicted_band": list(predicted_band),
        "evidence_tag_pending": "[contest-CUDA]",
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT))
        if manifest_path.is_relative_to(REPO_ROOT)
        else str(manifest_path),
        "expected_artifact_dir": (
            f"experiments/results/lightning_batch/{job_name}"
        ),
        "expected_auth_eval_json": (
            f"experiments/results/lightning_batch/{job_name}/contest_auth_eval.json"
        ),
        "submit_result": submit_result,
    }
    existing.append(record)
    LIGHTNING_ACTIVE_JOBS_PATH.write_text(
        json.dumps(existing, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[persist] {LIGHTNING_ACTIVE_JOBS_PATH.relative_to(REPO_ROOT)} ({len(existing)} active jobs)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rel-err-budget",
        type=float,
        default=DEFAULT_REL_ERR_BUDGET,
        help=(
            f"Per-tensor rel_err budget for the K-search (default "
            f"{DEFAULT_REL_ERR_BUDGET}; the empirically-validated "
            f"CONDITIONAL-band threshold from the lossy_coarsening finding)."
        ),
    )
    p.add_argument(
        "--state-dict",
        type=Path,
        default=DEFAULT_PR101_STATE_DICT,
        help=f"PR101 decoder state_dict (.pt). Default: {DEFAULT_PR101_STATE_DICT}",
    )
    p.add_argument(
        "--frontier-archive",
        type=Path,
        default=DEFAULT_PR101_FRONTIER_ARCHIVE,
        help="PR101 frontier archive providing latent_blob + sidecar_blob.",
    )
    p.add_argument(
        "--pr101-source-dir",
        type=Path,
        default=DEFAULT_PR101_SOURCE_DIR,
        help="PR101 hnerv_ft_microcodec/ source dir for vendoring src/codec.py + src/model.py.",
    )
    p.add_argument(
        "--brotli-quality",
        type=int,
        default=11,
        help="Brotli quality for the per-tensor flat int8 stream (default 11).",
    )
    p.add_argument(
        "--machine",
        default=DEFAULT_MACHINE,
        help=f"Lightning machine class (default {DEFAULT_MACHINE} = AWS T4)",
    )
    p.add_argument(
        "--max-runtime-sec",
        type=int,
        default=DEFAULT_MAX_RUNTIME_SEC,
        help=f"Hard cap on Job runtime (default {DEFAULT_MAX_RUNTIME_SEC}s = 90 min)",
    )
    p.add_argument(
        "--predicted-low",
        type=float,
        default=0.18,
        help="Predicted score band low (advisory, not score-claimed).",
    )
    p.add_argument(
        "--predicted-high",
        type=float,
        default=0.22,
        help="Predicted score band high (advisory).",
    )
    p.add_argument(
        "--ssh-target",
        default=default_ssh_target(),
        help="Lightning Studio SSH target ($LIGHTNING_SSH_TARGET / $REMOTE).",
    )
    p.add_argument(
        "--remote-pact",
        default=default_remote_pact(),
        help=f"Remote pact dir (default {DEFAULT_LIGHTNING_REMOTE_PACT}).",
    )
    p.add_argument(
        "--teamspace",
        default=default_teamspace(),
        help="Lightning teamspace name (default $LIGHTNING_TEAMSPACE).",
    )
    p.add_argument(
        "--studio",
        default=default_studio(),
        help="Lightning Studio name (default $LIGHTNING_STUDIO).",
    )
    p.add_argument(
        "--user",
        default=default_user(),
        help="Lightning user (default $LIGHTNING_USER).",
    )
    p.add_argument(
        "--job-name",
        default=None,
        help="Override auto-generated job name.",
    )
    p.add_argument(
        "--budget-cap-usd",
        type=float,
        default=DEFAULT_BUDGET_CAP_USD,
        help=f"Budget cap (default ${DEFAULT_BUDGET_CAP_USD}); recorded in metadata.",
    )
    p.add_argument(
        "--skip-stage",
        action="store_true",
        help="Skip lightning_repro_workspace.py (workspace already staged).",
    )
    p.add_argument(
        "--force-claim",
        action="store_true",
        help="Force the dispatch claim only when replacing a known terminal/stale claim.",
    )
    p.add_argument(
        "--force-claim-reason",
        default=None,
        help="Required rationale when --force-claim is set.",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="Build the archive + forked submission dir locally and exit (no claim, no dispatch).",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved invocation + remote command without staging or submitting.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + stage + claim, but submit a Lightning dry-run (no GPU spend).",
    )
    p.add_argument(
        "--skip-roundtrip-smoke",
        action="store_true",
        help="Skip the local CPU roundtrip smoke check (NOT recommended).",
    )
    args = p.parse_args(argv)

    job_name = args.job_name or _job_name()

    # Local build dir under experiments/results/<lane>_<ts>/
    timestamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    build_dir = (
        REPO_ROOT
        / "experiments"
        / "results"
        / f"lossy_coarsening_{timestamp}"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    archive_path = build_dir / "archive.zip"
    submission_dir = build_dir / "submission_dir"
    build_manifest_path = build_dir / "build_manifest.json"

    # Pre-flight: required inputs.
    if not args.state_dict.is_file():
        sys.exit(f"FATAL: --state-dict not found: {args.state_dict}")
    if not args.frontier_archive.is_file():
        sys.exit(f"FATAL: --frontier-archive not found: {args.frontier_archive}")
    if not args.pr101_source_dir.is_dir():
        sys.exit(f"FATAL: --pr101-source-dir not found: {args.pr101_source_dir}")

    # ---- Stage A.1: build lossy decoder section.
    print(
        f"[build] lossy decoder section budget={args.rel_err_budget} "
        f"brotli_q={args.brotli_quality}"
    )
    section = _build_lossy_decoder_section(
        args.state_dict,
        rel_err_budget=args.rel_err_budget,
        brotli_quality=args.brotli_quality,
    )
    print(
        f"[build]   decoder section bytes={section['section_total_bytes']:,} "
        f"(brotli={section['brotli_payload_bytes']:,} K_bytes={section['K_bytes']} "
        f"scale_bytes={section['scale_bytes']})"
    )
    print(f"[build]   rel_err vs int8 quantized symbols: {section['rel_err']:.6f}")

    # ---- Stage A.2: extract original PR101 latent + sidecar.
    pr101_inner = _read_pr101_inner_blob(args.frontier_archive)
    _orig_decoder, latent_blob, sidecar_blob = _split_pr101_inner_blob(pr101_inner)
    print(
        f"[build] PR101 frontier archive: latent_blob={len(latent_blob):,} B "
        f"sidecar_blob={len(sidecar_blob):,} B"
    )

    # ---- Stage A.3: compose the new inner blob and pack as ZIP.
    inner_blob = _build_inner_blob(section["decoder_bytes"], latent_blob, sidecar_blob)
    _write_pr101_archive(inner_blob, archive_path)
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256(archive_path.read_bytes())
    print(
        f"[build] WROTE archive: {archive_path.relative_to(REPO_ROOT)} "
        f"size={archive_bytes:,} B sha256={archive_sha[:16]}..."
    )

    # ---- Stage A.4: stage forked submission directory.
    _stage_forked_submission_dir(
        submission_dir, pr101_source_dir=args.pr101_source_dir
    )
    print(
        f"[build] WROTE submission dir: {submission_dir.relative_to(REPO_ROOT)} "
        f"(inflate.sh + inflate.py + src/codec.py + src/model.py)"
    )

    # ---- Stage A.5: local CPU roundtrip smoke (structural-correctness gate).
    smoke_result: dict | None = None
    if args.skip_roundtrip_smoke:
        print("[smoke] SKIPPED per --skip-roundtrip-smoke (NOT recommended)")
    else:
        print("[smoke] running local CPU roundtrip ...")
        smoke_result = _local_roundtrip_smoke(
            archive_path, pr101_state_dict_path=args.state_dict
        )
        print(
            f"[smoke] rel_err_vs_quantized_fp32={smoke_result['rel_err_vs_quantized_fp32']:.6f} "
            f"max_per_tensor={smoke_result['max_per_tensor_rel_err']:.6f} "
            f"n_tensors={smoke_result['n_tensors_compared']}"
        )
        # The roundtrip rel_err should match the encoder's reported rel_err
        # (within fp16 scale rounding). If it diverges by >1.5x we have a bug.
        if smoke_result["rel_err_vs_quantized_fp32"] > section["rel_err"] * 2 + 1e-3:
            sys.exit(
                f"FATAL: roundtrip rel_err {smoke_result['rel_err_vs_quantized_fp32']:.4f} "
                f">> encoder rel_err {section['rel_err']:.4f}; wire-format bug"
            )

    # ---- Stage A.6: write build manifest (forensic durable state).
    build_manifest = {
        "schema_version": "lossy_coarsening_lightning_build.v1",
        "lane_id": LANE_ID,
        "job_name_planned": job_name,
        "built_at_utc": _utc_now_iso(),
        "rel_err_budget": args.rel_err_budget,
        "rel_err_actual_int8": section["rel_err"],
        "rel_err_actual_fp32_smoke": (
            smoke_result.get("rel_err_vs_quantized_fp32") if smoke_result else None
        ),
        "max_per_tensor_rel_err_fp32_smoke": (
            smoke_result.get("max_per_tensor_rel_err") if smoke_result else None
        ),
        "brotli_quality": args.brotli_quality,
        "archive_relpath": str(archive_path.relative_to(REPO_ROOT)),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "submission_dir_relpath": str(submission_dir.relative_to(REPO_ROOT)),
        "input_state_dict": str(args.state_dict),
        "input_frontier_archive": str(args.frontier_archive),
        "input_pr101_source_dir": str(args.pr101_source_dir),
        "section_total_bytes": section["section_total_bytes"],
        "section_brotli_payload_bytes": section["brotli_payload_bytes"],
        "n_tensors": section["n_tensors"],
        "n_symbols": section["n_symbols"],
        "per_tensor_K": section["per_tensor_K"],
        "predicted_band": [args.predicted_low, args.predicted_high],
        "evidence_grade": "[CPU-build]",
        "score_claim": False,
        "promotion_eligible": False,
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[build] manifest: {build_manifest_path.relative_to(REPO_ROOT)}")

    if args.build_only:
        print(
            f"[build-only] DONE. Archive ready at "
            f"{archive_path.relative_to(REPO_ROOT)} ({archive_bytes:,} B). "
            f"No claim filed, no Lightning Job submitted."
        )
        return 0

    if args.print_only:
        print(f"=== resolved Lightning Job submission for {job_name} ===")
        print(f"machine: {args.machine}")
        print(f"studio: {args.studio or '<unset>'}")
        print(f"teamspace: {args.teamspace or '<unset>'}")
        print(f"user: {args.user or '<unset>'}")
        print(f"ssh_target: {args.ssh_target or '<unset>'}")
        print(f"max_runtime_sec: {args.max_runtime_sec}")
        print(f"predicted_band: [{args.predicted_low}, {args.predicted_high}]")
        print(f"budget_cap_usd: {args.budget_cap_usd}")
        archive_relpath = str(archive_path.relative_to(REPO_ROOT))
        submission_relpath = str(submission_dir.relative_to(REPO_ROOT))
        cmd = build_remote_command(
            job_name=job_name,
            remote_pact=args.remote_pact,
            archive_relpath=archive_relpath,
            submission_dir_relpath=submission_relpath,
        )
        print("--- remote command preview (first 800 chars) ---")
        print(cmd[:800])
        print(f"--- (full length: {len(cmd)} chars) ---")
        return 0

    # Fail-loud env validation BEFORE any stage / claim / submit.
    missing = []
    if not args.ssh_target:
        missing.append(
            "--ssh-target / $LIGHTNING_SSH_TARGET (e.g. s_<token>@ssh.lightning.ai)"
        )
    if not args.studio:
        missing.append("--studio / $LIGHTNING_STUDIO (e.g. lossy-compression-challenge)")
    if not args.teamspace:
        missing.append("--teamspace / $LIGHTNING_TEAMSPACE (e.g. comma-lab)")
    if not args.user:
        missing.append("--user / $LIGHTNING_USER (e.g. adpena)")
    if missing:
        sys.exit(
            "FATAL: missing required Lightning environment values:\n  - "
            + "\n  - ".join(missing)
            + "\nReference recipe: "
            "~/.claude/projects/-Users-adpena-Projects-pact/memory/"
            "reference_lightning_studio_canonical_dispatch_recipe_20260505.md"
        )

    # ---- Stage B.1: workspace rsync + manifest.
    if args.skip_stage:
        manifest = (
            REPO_ROOT
            / "experiments"
            / "results"
            / "lightning_batch"
            / job_name
            / "source_manifest.json"
        )
        if not manifest.is_file():
            sys.exit(f"FATAL: --skip-stage but manifest not found: {manifest}")
    else:
        manifest = stage_workspace(
            job_name=job_name,
            archive_path=archive_path,
            submission_dir=submission_dir,
            ssh_target=args.ssh_target,
            remote_pact=args.remote_pact,
        )

    # ---- Stage B.2: dispatch claim.
    claim_lane(
        job_name=job_name,
        force_claim=args.force_claim,
        force_claim_reason=args.force_claim_reason,
    )

    # ---- Stage B.3: build remote command + submit.
    archive_relpath = str(archive_path.relative_to(REPO_ROOT))
    submission_relpath = str(submission_dir.relative_to(REPO_ROOT))
    command = build_remote_command(
        job_name=job_name,
        remote_pact=args.remote_pact,
        archive_relpath=archive_relpath,
        submission_dir_relpath=submission_relpath,
    )
    submit_result = submit_lightning_job(
        job_name=job_name,
        machine=args.machine,
        command=command,
        teamspace=args.teamspace,
        studio=args.studio,
        user=args.user,
        max_runtime_sec=args.max_runtime_sec,
        dry_run=args.dry_run,
    )

    # ---- Stage B.4: persist active-jobs row.
    persist_active_job(
        job_name=job_name,
        machine=args.machine,
        submit_result=submit_result,
        manifest_path=manifest,
        archive_path=archive_path,
        archive_bytes=archive_bytes,
        rel_err_budget=args.rel_err_budget,
        rel_err_actual=section["rel_err"],
        predicted_band=(args.predicted_low, args.predicted_high),
        smoke_roundtrip=smoke_result,
    )

    print(json.dumps(submit_result, indent=2, default=str))
    print(f"\n[submitted] job_name={job_name}")
    print(
        f"[harvest]   .venv/bin/python experiments/lossy_coarsening_lightning_harvest.py "
        f"--job-name {job_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
