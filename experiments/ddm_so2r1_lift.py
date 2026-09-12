"""ddm_so2r1 -- materialize SO2's block-local modulo-five lifting Z of the move-49 field.

WHAT THIS IS.  SO2 (`.omx/research/ddm_so2_successor_object_priced_with_the_learned_prior_20260913.md`)
specifies ONE construction: for each frame and each aligned 64x64 patch of the move-49 token
field F, split the patch into 2x2 cells ``(a,b;c,d)``, put the anchors ``a`` into the patch's
upper-left 32x32 quadrant and ``(b-a) mod 5`` / ``(c-a) mod 5`` / ``(d-a) mod 5`` into the
upper-right / lower-left / lower-right quadrants.  The result Z is a full-size five-symbol
field of exactly the same shape and alphabet, so the existing HPAC trainer, packer and RLC1
coder all accept it without a single change to their contracts.

WHY THE INVERSE IS WRITTEN TWICE.  The forward map and the inverse are implemented by two
INDEPENDENT routines with different index arithmetic: the forward one works in the patch
frame (reshape/transpose to (frames, 6, 8, 64, 64)), the inverse one works directly in image
coordinates with explicit strided slices.  A single routine run backwards would only prove it
is self-consistent; two independent routines that agree on all 117,964,800 bytes prove the map
is the bijection SO2 claims.  The inverse is the map the RECEIVER would have to learn, so it
is the one that must be checked, not the encoder's convenience.

NO SCORER, NO RENDERER, NO DISPATCH.  Z is a NEW artifact under this arm's own root; the live
field, the renderer, the carrier and every sealed tree are read-only inputs here.

Axis ``[macOS-CPU advisory; exact bytes, scorer-free]``; ``score_claim=false``.

Usage::

  python experiments/ddm_so2r1_lift.py --root /Volumes/APDataStore/pact/ddm_so2_first_rung
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np

N_PAIRS, EVAL_H, EVAL_W = 600, 384, 512
PATCH = 64
HALF = PATCH // 2
MODULUS = 5
#: SO2's pinned inputs.  Both are read-only to this arm and are checked by sha before use.
SOURCE_U8 = Path("/Volumes/APDataStore/pact/ddm_sj1_pass7/rlc1/fields/subset7.u8")
SOURCE_U8_SHA = "fdf2255f60364dcd1e67fb7de107c0640f5fe3a39a7b575c76c86636660efd1d"
SOURCE_NPZ = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/admission_pass7/field_admitted.npz")
SOURCE_NPZ_SHA = "6056c90585763078e2bea8dea32b14abf57462d39eb56c17d46fd21bd702ae0f"
INIT_DEPTHS = Path("/Volumes/VertigoDataTier/pact/ddm_dpi1/train_inputs/init_depths.pt")
INIT_DEPTHS_SHA = "f05bae5b2696b1e96817c214d9b1d94ecb64ca18f1c6c30b6e398cd5d4b22072"
#: SO2 asks for chunks of at most 120 frames so a crash leaves completed stage files behind.
CHUNK = 120
#: Free-space floor on the destination volume.  A refusal here is the guard working; MAIN
#: re-rooted this rung to APDataStore because Vertigo sits UNDER its own 40 GiB reserve, and
#: that reserve is never lowered to make a stage fit.
RESERVE_BYTES = 20 << 30
AXIS = "[macOS-CPU advisory; exact bytes, scorer-free]"


class LiftError(RuntimeError):
    """A lifting input or invariant is not what SO2's construction says it is."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def fact(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def preflight(path: Path, root: Path, need: int) -> None:
    """Refuse a write outside the owned root, or one the volume cannot hold."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise LiftError(f"write outside the owned root: {resolved}")
    free = shutil.disk_usage(root.parent).free
    if free < need + RESERVE_BYTES:
        raise LiftError(
            f"STORAGE_BLOCK on {root.parent}: need {need} B over a {RESERVE_BYTES} B reserve, "
            f"observed {free} B free"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)


def atomic_write(path: Path, payload: bytes, root: Path) -> dict:
    """Immutable publish: an existing payload must already be byte-identical."""
    path = Path(path)
    if path.exists():
        if sha256_file(path) != sha256_bytes(payload):
            raise LiftError(f"immutable payload changed: {path}")
        return fact(path)
    preflight(path, root, len(payload))
    temporary = path.with_name(path.name + ".new")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return fact(path)


def record(path: Path, value: object, root: Path) -> object:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    preflight(path, root, len(payload))
    temporary = Path(path).with_name(Path(path).name + ".new")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return value


# ----------------------------------------------------------------------------------
# the construction -- forward
# ----------------------------------------------------------------------------------

def lift_forward(frames: np.ndarray) -> np.ndarray:
    """SO2's modulo-five lifting, computed in the PATCH frame.

    ``frames`` is (n, 384, 512) uint8 in 0..4.  The patch grid is 6 x 8 because both image
    dimensions are divisible by the trainer's own patch size of 64.
    """
    if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (EVAL_H, EVAL_W):
        raise LiftError(f"forward input geometry is wrong: {frames.shape} {frames.dtype}")
    if int(frames.max(initial=0)) >= MODULUS:
        raise LiftError("forward input carries a symbol outside the five-label alphabet")
    n = frames.shape[0]
    rows, cols = EVAL_H // PATCH, EVAL_W // PATCH
    # (n, 384, 512) -> (n, rows, 64, cols, 64) -> (n, rows, cols, 64, 64)
    patches = frames.reshape(n, rows, PATCH, cols, PATCH).transpose(0, 1, 3, 2, 4)
    # (…, 64, 64) -> (…, 32, 2, 32, 2): cell (i, j) is [2i:2i+2, 2j:2j+2]
    cells = patches.reshape(n, rows, cols, HALF, 2, HALF, 2)
    a = cells[..., 0, :, 0]
    b = cells[..., 0, :, 1]
    c = cells[..., 1, :, 0]
    d = cells[..., 1, :, 1]
    out = np.empty((n, rows, cols, PATCH, PATCH), dtype=np.uint8)
    out[..., :HALF, :HALF] = a
    out[..., :HALF, HALF:] = (b.astype(np.int16) - a) % MODULUS
    out[..., HALF:, :HALF] = (c.astype(np.int16) - a) % MODULUS
    out[..., HALF:, HALF:] = (d.astype(np.int16) - a) % MODULUS
    return out.transpose(0, 1, 3, 2, 4).reshape(n, EVAL_H, EVAL_W)


# ----------------------------------------------------------------------------------
# the construction -- inverse, written independently in IMAGE coordinates
# ----------------------------------------------------------------------------------

def lift_inverse(lifted: np.ndarray) -> np.ndarray:
    """The receiver's map: read the four quadrants of each 64x64 patch and rebuild F.

    Deliberately NOT the forward routine run backwards.  It addresses the quadrants with
    explicit strided slices over image coordinates, so an indexing mistake in either routine
    shows up as a byte disagreement rather than cancelling out.
    """
    if lifted.dtype != np.uint8 or lifted.ndim != 3 or lifted.shape[1:] != (EVAL_H, EVAL_W):
        raise LiftError(f"inverse input geometry is wrong: {lifted.shape} {lifted.dtype}")
    if int(lifted.max(initial=0)) >= MODULUS:
        raise LiftError("inverse input carries a symbol outside the five-label alphabet")
    n = lifted.shape[0]
    out = np.empty((n, EVAL_H, EVAL_W), dtype=np.uint8)
    for top in range(0, EVAL_H, PATCH):
        for left in range(0, EVAL_W, PATCH):
            anchor = lifted[:, top:top + HALF, left:left + HALF].astype(np.int16)
            right = lifted[:, top:top + HALF, left + HALF:left + PATCH].astype(np.int16)
            lower = lifted[:, top + HALF:top + PATCH, left:left + HALF].astype(np.int16)
            corner = lifted[:, top + HALF:top + PATCH, left + HALF:left + PATCH].astype(np.int16)
            out[:, top:top + PATCH:2, left:left + PATCH:2] = anchor.astype(np.uint8)
            out[:, top:top + PATCH:2, left + 1:left + PATCH:2] = ((anchor + right) % MODULUS).astype(np.uint8)
            out[:, top + 1:top + PATCH:2, left:left + PATCH:2] = ((anchor + lower) % MODULUS).astype(np.uint8)
            out[:, top + 1:top + PATCH:2, left + 1:left + PATCH:2] = ((anchor + corner) % MODULUS).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------------
# stage driver
# ----------------------------------------------------------------------------------

def materialize(root: Path) -> dict:
    import torch

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = root / "lift/stages"
    if sha256_file(SOURCE_U8) != SOURCE_U8_SHA:
        raise LiftError("the pinned move-49 field u8 changed under this arm")
    if sha256_file(SOURCE_NPZ) != SOURCE_NPZ_SHA:
        raise LiftError("the pinned move-49 field npz changed under this arm")
    if sha256_file(INIT_DEPTHS) != INIT_DEPTHS_SHA:
        raise LiftError("the pinned warm initializer changed under this arm")

    source = np.memmap(SOURCE_U8, dtype=np.uint8, mode="r", shape=(N_PAIRS, EVAL_H, EVAL_W))
    # Independent re-read of the SAME field out of its npz: the two custody forms must agree
    # before either is lifted, so a silently-swapped file cannot enter the representation.
    with np.load(SOURCE_NPZ, allow_pickle=False) as data:
        if set(data.files) != {str(i) for i in range(N_PAIRS)}:
            raise LiftError("the pinned npz does not carry all 600 planes")
        from_npz = np.stack([data[str(i)] for i in range(N_PAIRS)])
    if from_npz.shape != (N_PAIRS, EVAL_H, EVAL_W) or from_npz.dtype != np.uint8:
        raise LiftError("the pinned npz geometry is wrong")
    if sha256_bytes(from_npz.tobytes(order="C")) != SOURCE_U8_SHA:
        raise LiftError("the npz and u8 custody forms of the move-49 field disagree")
    del from_npz

    stage_facts, mismatched, symbol_counts = {}, 0, np.zeros(MODULUS, dtype=np.int64)
    detail_zero = 0
    pieces = []
    for start in range(0, N_PAIRS, CHUNK):
        stop = min(start + CHUNK, N_PAIRS)
        block = np.ascontiguousarray(source[start:stop])
        lifted = lift_forward(block)
        rebuilt = lift_inverse(lifted)
        mismatched += int(np.count_nonzero(rebuilt != block))
        symbol_counts += np.bincount(lifted.reshape(-1), minlength=MODULUS).astype(np.int64)
        detail = np.concatenate(
            [
                lifted.reshape(-1, EVAL_H // PATCH, PATCH, EVAL_W // PATCH, PATCH)
                .transpose(0, 1, 3, 2, 4)[..., :HALF, HALF:].reshape(-1),
                lifted.reshape(-1, EVAL_H // PATCH, PATCH, EVAL_W // PATCH, PATCH)
                .transpose(0, 1, 3, 2, 4)[..., HALF:, :].reshape(-1),
            ]
        )
        detail_zero += int(np.count_nonzero(detail == 0))
        path = stages / f"Z_{start:04d}_{stop:04d}.u8"
        stage_facts[f"{start:04d}_{stop:04d}"] = atomic_write(path, lifted.tobytes(order="C"), root)
        pieces.append(lifted)
        print(json.dumps({"stage": "lift", "frames": [start, stop], "mismatched": mismatched}), flush=True)
    if mismatched:
        raise LiftError(f"INVERSE FAILED on {mismatched} bytes")

    lifted = np.concatenate(pieces, axis=0)
    del pieces
    if lifted.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise LiftError("assembled Z geometry is wrong")

    raw = lifted.tobytes(order="C")
    if len(raw) != N_PAIRS * EVAL_H * EVAL_W:
        raise LiftError("assembled Z byte count is wrong")
    z_sha = sha256_bytes(raw)
    z_fact = atomic_write(root / "Z.u8", raw, root)

    # The WHOLE-FIELD inverse, run once more on the assembled Z, compared against the source
    # memmap byte for byte.  This is the receipt SO2's falsifier list names: not a sample.
    whole = lift_inverse(lifted)
    if whole.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise LiftError("whole-field inverse geometry is wrong")
    inverse_raw = whole.tobytes(order="C")
    equal_bytes = int(np.count_nonzero(whole == np.asarray(source)))
    inverse_sha = sha256_bytes(inverse_raw)
    if inverse_sha != SOURCE_U8_SHA or equal_bytes != len(raw):
        raise LiftError(f"WHOLE-FIELD INVERSE FAILED: {equal_bytes}/{len(raw)} equal, sha {inverse_sha}")
    inverse_fact = atomic_write(root / "lift/F_from_Z.u8", inverse_raw, root)
    del whole, inverse_raw

    npz_path = root / "Z.npz"
    if not npz_path.exists():
        preflight(npz_path, root, len(raw) + (1 << 20))
        temporary = npz_path.with_name("Z.npz.new")
        with temporary.open("wb") as handle:
            np.savez(handle, **{str(i): lifted[i] for i in range(N_PAIRS)})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data.files) != {str(i) for i in range(N_PAIRS)}:
            raise LiftError("published Z.npz does not carry all 600 planes")
        if sha256_bytes(np.stack([data[str(i)] for i in range(N_PAIRS)]).tobytes(order="C")) != z_sha:
            raise LiftError("published Z.npz does not round-trip to Z")
    npz_fact = fact(npz_path)

    cache_path = root / "cache.pt"
    if not cache_path.exists():
        seg = torch.from_numpy(np.ascontiguousarray(lifted))
        if seg.device.type != "cpu" or seg.dtype != torch.uint8 or tuple(seg.shape) != (N_PAIRS, EVAL_H, EVAL_W):
            raise LiftError("cache seg is not the CPU uint8 (600,384,512) the trainer admits")
        preflight(cache_path, root, len(raw) + (1 << 20))
        temporary = cache_path.with_name("cache.pt.new")
        torch.save({"seg": seg, "spatial_token_sha256": z_sha}, temporary)
        os.replace(temporary, cache_path)
    reloaded = torch.load(cache_path, map_location="cpu", weights_only=True)
    seg = reloaded["seg"]
    if seg.device.type != "cpu" or seg.dtype != torch.uint8 or tuple(seg.shape) != (N_PAIRS, EVAL_H, EVAL_W):
        raise LiftError("published cache seg fails the trainer's own geometry contract")
    if int(seg.min()) < 0 or int(seg.max()) > 4:
        raise LiftError("published cache seg leaves the five-label alphabet")
    observed = sha256_bytes(seg.contiguous().numpy().tobytes(order="C"))
    if observed != z_sha or reloaded.get("spatial_token_sha256") != z_sha:
        raise LiftError("published cache content sha does not match Z")
    cache_fact = fact(cache_path)

    source_counts = np.bincount(np.asarray(source).reshape(-1), minlength=MODULUS).astype(np.int64)
    n_patches = N_PAIRS * (EVAL_H // PATCH) * (EVAL_W // PATCH)
    manifest = {
        "schema": "ddm_so2r1_train_inputs.v1",
        "axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "construction": {
            "name": "SO2 block-local modulo-five lifting",
            "patch": PATCH,
            "cell": 2,
            "modulus": MODULUS,
            "format_prefix": "SO2L",
            "version_tile_stride_modulus": [1, 64, 2, 5],
            "quadrants": "UL=a, UR=(b-a) mod 5, LL=(c-a) mod 5, LR=(d-a) mod 5",
            "patches": n_patches,
            "anchor_symbols": n_patches * HALF * HALF,
            "detail_symbols": 3 * n_patches * HALF * HALF,
        },
        "source_field": {"u8": fact(SOURCE_U8), "npz": fact(SOURCE_NPZ)},
        "init_depths": fact(INIT_DEPTHS),
        "Z": z_fact,
        "Z_npz": npz_fact,
        "cache": cache_fact,
        "expected_cache_content_sha256": z_sha,
        "cache_file_sha256": cache_fact["sha256"],
        "stages": stage_facts,
        "inverse_receipt": {
            "bytes_compared": len(raw),
            "bytes_equal": equal_bytes,
            "mismatched_bytes": 0,
            "inverse_field_sha256": inverse_sha,
            "source_field_sha256": SOURCE_U8_SHA,
            "inverse_artifact": inverse_fact,
            "independent_routines": True,
        },
        "telemetry": {
            "source_symbol_counts": source_counts.tolist(),
            "Z_symbol_counts": symbol_counts.tolist(),
            "detail_zero_symbols": detail_zero,
            "detail_symbols": 3 * n_patches * HALF * HALF,
            "detail_zero_fraction": detail_zero / (3 * n_patches * HALF * HALF),
        },
        "storage": {
            "root": str(root),
            "reserve_bytes": RESERVE_BYTES,
            "free_bytes_after": shutil.disk_usage(root.parent).free,
        },
        "producer": fact(Path(__file__)),
    }
    return record(root / "TRAIN_INPUTS.json", manifest, root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.root)
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
