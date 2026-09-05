"""ddm_pc1 -- pose-carrier efficiency: basis precision, coefficient precision,
generated basis, learned low rank -- each WITH the full n600 re-solve.

The object
----------
The frontier archive (cl2 repack, 179,982 B, sha ``08ec8533...``) stores its pose
carrier as an RX1 section of **22,031 B (12.24%)**, MEASURED by parsing the
container with the receiver's own ``runtime/residual_archive`` -- not recalled
from any earlier generation's table.  That stream is brotli(q9, lgwin16) over a
22,278 B body: 12,277 B of basis (27,648 five-bit zigzag symbols for a 12-atom,
3-plane, 24x32 grid, canonical-Huffman coded, then re-coded by the RR5 adaptive
arithmetic rider to 12,046 B), 9,830 B of AR1-predicted Rice-coded signed-int12
coefficients (600x12), 96 B of scales, 40 B of packed metadata, 29 B of frame-0
selector and 6 B of bit counts.  Brotli removes only 247 B (1.1%) of that body,
so payload savings pass through to the archive at very nearly 1:1.
``ddm_jg1`` MEASURED that
CUTTING the basis or the lattice WITHOUT re-solving damages d_pose 104.6-822.7x.
The re-solve (``ddm_jg5.refine_pair``) is the OBJECT CHANGE that reopens the cut
(the composition law [[m148]]): a closed leg survives only if another leg changes
its object first.

What this module does NOT do
---------------------------
It never invents a byte count.  Basis bytes come from the receiver's own coders
(``runtime/rr5_arith_basis``: canonical Huffman for the incumbent, adaptive
arithmetic for the rider); coefficient bytes come from the receiver's own
``carrier_repack._rice_encode`` through ``ddm_t1h_carrier_byte_pricer``.  Every
d_pose that is REPORTED is measured on ``cpu_torch`` fp32 through the exact,
non-STE receiver path over all 600 pairs.

Modes
-----
``probe``    -- byte anatomy of the shipped carrier + SVD spectrum of the 600
                realized carrier patterns (the prior for V5).  Scorer-free.
``variant``  -- build a variant basis/lattice, re-solve every assigned pair with
                ``jg5.refine_pair``, and retain the codes.  Resumable per pair.
``price``    -- exact bytes + n600 d_pose + Delta S for a solved variant.

Axis: bytes ``[exact local byte arithmetic]``; d_pose
``[macOS-CPU advisory, cpu_torch fp32 authority backend, n600]``.
``score_claim=false`` until a contest-CUDA T4 row exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

# --- the frontier body, pinned by sha256 (never inferred) ------------------- #
CL2_ARCHIVE_SHA256 = (
    "08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e"
)
CL2_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/rungs/"
    "lambda_1p0/retained/receiver_copy_runtime"
)
CL2_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder/parseback/"
    "lambda_1p0/0.raw"
)
CL2_ARCHIVE_BYTES = 179_982
# cl2 contest-CUDA T4 n600 row (the body being improved on).
CL2_SCORE_T4 = 0.14781744131049854
CL2_D_SEG_T4 = 0.00029229
CL2_D_POSE_T4 = 6.14e-06

N_PAIRS = 600
CARRIER_DIM = 12
BASIS_PLANES = 3
CARRIER_H, CARRIER_W = 24, 32
BASIS_SYMBOLS = CARRIER_DIM * BASIS_PLANES * CARRIER_H * CARRIER_W  # 27,648
BYTE_TO_SCORE = 25.0 / 37_545_489.0
#: Admit bar from the charter: 10x the pure-pose projection error (3.7e-6).
ADMIT_BAR = -2e-5
#: ft1's MEASURED same-object pose ceiling (gs3 addendum 4 item 3).  A candidate
#: whose n600 d_pose exceeds it is inadmissible REGARDLESS of its Delta S
#: arithmetic, because the object it would ship is outside the ceiling the same
#: object was measured to hold.
FT1_POSE_CEILING = 1.694e-5


def break_even_d_pose(d_pose_base: float, delta_rate: float) -> float:
    """The largest d_pose a rate change still pays for -- for BOTH signs.

    ``sqrt(10 d_new) <= sqrt(10 d_base) - delta_rate``.  A rate SAVING raises the
    bar above the base (the edit may cost pose); a byte-ADDING edit lowers it
    below the base (the edit must IMPROVE pose).  Returns ``0.0`` when the added
    bytes exceed the whole pose term, so no non-negative d_pose can pay.
    """
    if d_pose_base <= 0.0:
        raise Pc1Error("d_pose_base must be positive")
    leg = math.sqrt(10.0 * d_pose_base) - delta_rate
    if leg <= 0.0:
        return 0.0
    return leg * leg / 10.0


class Pc1Error(RuntimeError):
    """A ddm_pc1 precondition failed.  Fail closed; never approximate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return (
        100.0 * d_seg
        + math.sqrt(10.0 * d_pose)
        + 25.0 * archive_bytes / 37_545_489.0
    )


# --------------------------------------------------------------------------
# The receiver's own coders, loaded from the frontier runtime tree.
# --------------------------------------------------------------------------


def load_runtime_coders(runtime: Path):
    """Import ``rr5_arith_basis`` and ``carrier_repack`` from the SHIPPED tree.

    The rider file is the single source of truth for the basis coder; importing a
    repo copy instead would price against code the receiver does not run.
    """
    runtime = Path(runtime).resolve()
    inserted = [str(runtime), str(runtime / "cpr1")]
    for entry in inserted:
        sys.path.insert(0, entry)
    try:
        from runtime import (
            carrier_repack,  # type: ignore[import-not-found]
            rr5_arith_basis,  # type: ignore[import-not-found]
        )
    finally:
        for entry in inserted:
            if entry in sys.path:
                sys.path.remove(entry)
    return rr5_arith_basis, carrier_repack


def basis_symbols_from_codes(basis_codes: np.ndarray) -> np.ndarray:
    """Zigzag the signed basis codes into the 5-bit symbol alphabet."""
    values = np.asarray(basis_codes, dtype=np.int64).reshape(-1)
    return ((values << 1) ^ (values >> 63)).astype(np.int64)


def basis_payload_bits(basis_codes: np.ndarray, rr5) -> dict[str, int]:
    """EXACT basis payload bits under both shipped coders.

    ``huffman`` is the incumbent CPR1 inner code (32-symbol canonical Huffman,
    one table shared by all 12 atoms, table transmitted as 32 bytes).  ``arith``
    is the RR5 rider (adaptive, contexted on the atom index, no table).  Both are
    the receiver's own functions -- nothing is estimated.
    """
    symbols = basis_symbols_from_codes(basis_codes)
    alphabet = int(rr5.BASIS_ALPHABET)
    if symbols.min() < 0 or symbols.max() >= alphabet:
        raise Pc1Error(
            f"basis symbol outside the alphabet: [{symbols.min()}, {symbols.max()}] "
            f"not within [0, {alphabet})"
        )
    histogram = np.bincount(symbols, minlength=alphabet).astype(np.int64)
    lengths = rr5.huffman_lengths_from_histogram(histogram)
    _payload, huffman_bits = rr5.huffman_encode(symbols, lengths)
    _arith_payload, arith_bits = rr5.encode_basis_arith(symbols)
    return {
        "symbols": int(symbols.size),
        "distinct_symbols": int((histogram > 0).sum()),
        "huffman_bits": int(huffman_bits),
        "arith_bits": int(arith_bits),
        "huffman_payload_bytes": (int(huffman_bits) + 7) // 8,
        "arith_payload_bytes": (int(arith_bits) + 7) // 8,
    }


# --------------------------------------------------------------------------
# mode=probe -- byte anatomy + the SVD spectrum of the realized patterns.
# --------------------------------------------------------------------------


def load_state(runtime: Path, *, expect_sha: str = CL2_ARCHIVE_SHA256):
    import ddm_up2_shipping_pose_solve as up2

    runtime = Path(runtime).resolve()
    observed = sha256_file(runtime / "archive.zip")
    if observed != expect_sha:
        raise Pc1Error(
            f"runtime archive sha256 {observed} != expected {expect_sha}; refusing "
            "to measure against an unidentified body"
        )
    return up2, up2.load_carrier_state(runtime, verify_archive=False)


def run_probe(args) -> int:
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime)
    started = time.time()

    up2, state = load_state(runtime, expect_sha=args.expect_archive_sha256)
    rr5, carrier_repack = load_runtime_coders(runtime)

    basis_raw = state.basis_raw.double()  # (12, 3, 24, 32), already scaled
    codes = np.asarray(state.codes, dtype=np.int32)  # (600, 12) int12 lattice
    coefficient_scales = state.coefficient_scales.double().numpy()

    # --- the basis codes, recovered from the container (not from basis_raw,
    # --- which the receiver has already multiplied by basis_scales).
    from importlib import import_module

    inserted = [str(runtime.resolve()), str((runtime / "cpr1").resolve())]
    for entry in inserted:
        sys.path.insert(0, entry)
    try:
        renderer = import_module("inflate")
        from runtime.carrier_repack import (  # type: ignore[import-not-found]
            materialize_cpr1,
            split_frame0_selector_carrier,
        )
        from runtime.residual_archive import (  # type: ignore[import-not-found]
            read_residual_archive,
        )
    finally:
        for entry in inserted:
            if entry in sys.path:
                sys.path.remove(entry)

    parts = read_residual_archive(runtime / "archive.zip")
    carrier_blob, selector_blob = split_frame0_selector_carrier(parts.carrier_blob)
    canonical = materialize_cpr1(parts.carrier_blob, renderer)
    basis_count = CARRIER_DIM * BASIS_PLANES * renderer.CARRIER_H * renderer.CARRIER_W
    basis_scales, basis_codes, coeff_scales_arr, encoded = (
        renderer.decode_compact_carrier(
            canonical, basis_count=basis_count, frames=N_PAIRS,
            dimensions=CARRIER_DIM,
        )
    )

    bits = basis_payload_bits(basis_codes, rr5)

    # --- CONTROL: does basis_scales actually reach the render?  normalized_basis
    # --- centres and RMS-normalises PER ATOM, so a positive per-atom scale must
    # --- cancel exactly.  Measure it, never assume it.
    raw_codes_only = (
        torch.from_numpy(
            np.asarray(basis_codes, dtype=np.int64).reshape(
                CARRIER_DIM, BASIS_PLANES, renderer.CARRIER_H, renderer.CARRIER_W
            )
        ).float()
    )
    norm_shipped = renderer.normalized_basis(state.basis_raw.clone()).double()
    norm_codes_only = renderer.normalized_basis(raw_codes_only.clone()).double()
    scale_signs = np.sign(np.asarray(basis_scales, dtype=np.float64))
    scale_cancels_max_abs = float(
        (norm_shipped - norm_codes_only * torch.from_numpy(scale_signs)[
            :, None, None, None
        ]).abs().max()
    )

    # --- the 600 realized carrier patterns, in the space the render consumes.
    # --- render_frame0_float computes einsum(coeff, basis_norm)/sqrt(12); the
    # --- SVD prior for V5 is of exactly that field, so build it that way.
    coefficients = torch.from_numpy(codes.astype(np.float32)) * torch.from_numpy(
        coefficient_scales.astype(np.float32)
    )[None]
    patterns = torch.einsum(
        "bk,kchw->bchw", coefficients.double(), norm_shipped
    ) / math.sqrt(CARRIER_DIM)
    flat = patterns.reshape(N_PAIRS, -1).numpy()
    # Economy SVD of a (600, 589,824) matrix via the 600x600 Gram.
    gram = flat @ flat.T
    eigenvalues = np.linalg.eigvalsh(gram)[::-1].clip(min=0.0)
    singular = np.sqrt(eigenvalues)
    energy = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(energy)
    rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)
    rank_999 = int(np.searchsorted(cumulative, 0.999) + 1)

    # The coefficient matrix itself: rank of the 600x12 lattice in code space.
    coeff_singular = np.linalg.svd(
        coefficients.double().numpy(), compute_uv=False
    )
    coeff_energy = (coeff_singular**2) / (coeff_singular**2).sum()

    # --- per-plane structure: is the basis luma (3 identical planes)?
    plane_max_spread = float(
        (basis_raw - basis_raw.mean(dim=1, keepdim=True)).abs().max()
    )

    priced = {
        "carrier_section_bytes": len(parts.carrier_blob),
        "cpr1_canonical_bytes": len(canonical),
        "packed_carrier_bytes": len(carrier_blob),
        "selector_bytes": len(selector_blob) if selector_blob else 0,
        "basis_scales_bytes": int(CARRIER_DIM * 4),
        "coefficient_scales_bytes": int(CARRIER_DIM * 4),
    }
    ks, _payload, rice_bits = carrier_repack._rice_encode(
        _zigzag_delta_along_pairs(codes), 1
    )
    priced["rice_payload_bits_recomputed"] = int(rice_bits)
    priced["rice_payload_bytes_recomputed"] = (int(rice_bits) + 7) // 8
    priced["rice_ks"] = ks.reshape(-1).astype(int).tolist()

    summary: dict[str, Any] = {
        "schema": "ddm_pc1_probe.v1",
        "axis": "[exact local byte arithmetic] + [scorer-free]",
        "score_claim": False,
        "promotable": False,
        "runtime": str(runtime),
        "archive_sha256": sha256_file(runtime / "archive.zip"),
        "archive_bytes": int((runtime / "archive.zip").stat().st_size),
        "basis": {
            "shape": list(basis_codes.reshape(
                CARRIER_DIM, BASIS_PLANES, renderer.CARRIER_H, renderer.CARRIER_W
            ).shape),
            "code_min": int(np.min(basis_codes)),
            "code_max": int(np.max(basis_codes)),
            "scales": np.asarray(basis_scales, dtype=float).tolist(),
            "scale_signs_all_positive": bool((scale_signs > 0).all()),
            "scale_cancellation_max_abs_diff": scale_cancels_max_abs,
            "plane_max_spread_from_plane_mean": plane_max_spread,
            **bits,
        },
        "coefficients": {
            "shape": list(codes.shape),
            "code_min": int(codes.min()),
            "code_max": int(codes.max()),
            "scales": coefficient_scales.tolist(),
            "codes_sha256": sha256_array(codes),
        },
        "bytes": priced,
        "svd_realized_patterns": {
            "note": (
                "SVD of the 600 realized carrier fields "
                "einsum(coeff, basis_norm)/sqrt(12) at 3x384x512, via the 600x600 "
                "Gram; energy = sigma^2 share"
            ),
            "singular_values_top32": singular[:32].tolist(),
            "energy_share_top32": energy[:32].tolist(),
            "cumulative_energy_top16": cumulative[:16].tolist(),
            "effective_rank_99pct": rank_99,
            "effective_rank_999pct": rank_999,
        },
        "svd_coefficient_matrix": {
            "singular_values": coeff_singular.tolist(),
            "energy_share": coeff_energy.tolist(),
            "cumulative_energy": np.cumsum(coeff_energy).tolist(),
        },
        "wall_clock_seconds": time.time() - started,
    }
    (out_dir / "PROBE.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "carrier_state.npz",
        basis_codes=np.asarray(basis_codes, dtype=np.int32),
        basis_scales=np.asarray(basis_scales, dtype=np.float32),
        coefficient_scales=coefficient_scales.astype(np.float32),
        codes=codes,
    )
    print(json.dumps(summary, indent=2)[:6000])
    return 0


def _zigzag_delta_along_pairs(codes: np.ndarray) -> np.ndarray:
    """The receiver's coefficient pre-transform: zigzag deltas along the pair axis.

    Mirrors ``cpr1/inflate.py:236-244`` read backwards: the decoder cumsums the
    unzigzagged deltas modulo 2^12, so the encoder differences modulo 2^12 and
    zigzags.  Returned in the (frames, dimensions) shape ``_rice_encode`` wants.
    """
    values = np.asarray(codes, dtype=np.int64)
    if values.shape != (N_PAIRS, CARRIER_DIM):
        raise Pc1Error(f"coefficient lattice must be (600, 12), got {values.shape}")
    wrapped = values & 0xFFF
    delta = np.empty_like(wrapped)
    delta[0] = wrapped[0]
    delta[1:] = (wrapped[1:] - wrapped[:-1]) & 0xFFF
    signed = np.where(delta >= 0x800, delta - 0x1000, delta)
    return ((signed << 1) ^ (signed >> 63)).astype(np.int64) & 0xFFF


# --------------------------------------------------------------------------
# Variant construction -- the basis and the lattice under test.
# --------------------------------------------------------------------------


#: Quantiser step grid, searched per atom.  Not a tuned constant: the objective
#: below is evaluated at every point and the argmax is taken, so the grid only
#: has to be fine enough that the argmax is resolved (400 points over a range
#: derived from the atom's own dynamic range).
_STEP_GRID_POINTS = 400


def quantize_basis(
    basis_codes: np.ndarray, bits: int, *, grid_points: int = _STEP_GRID_POINTS
) -> np.ndarray:
    """Re-quantise the shipped basis onto a ``bits``-wide alphabet, AT OPTIMAL FORM.

    A narrower alphabet is a pure CONTENT change: ``carrier_codec`` reads whatever
    symbols the Huffman table carries, so nothing in the container moves.

    The quantiser step is searched PER ATOM, because it is free.  ``basis_scales``
    is per atom and ``normalized_basis`` centres and RMS-normalises each atom, so a
    positive per-atom scale cancels exactly (MEASURED in ``mode=probe``: 1.9e-06
    max abs difference).  The step therefore costs nothing and choosing one global
    step is leaving fidelity on the table -- a naive first pass, refused at the
    typing moment.  MEASURED on the shipped basis: a global ``codes/2`` gives
    per-atom cosine 0.899-0.983 at 4 bits, the searched step gives 0.991-1.000,
    and three of the twelve atoms (2, 5, 9) already span only [-7, 7] so they
    survive a 4-bit alphabet with NO loss at all.

    The objective is the cosine between the variant atom and the shipped atom in
    the space the render actually consumes -- after ``normalized_basis``'s bicubic
    upsample, centring and RMS -- not on the 24x32 code grid, because that is the
    field the carrier is built from.  That costs ``grid_points`` bicubic upsamples
    per atom and is deliberately not optimised: this runs a handful of times per
    campaign and exactness is worth more here than speed.  ``grid_points`` exists
    so tests can search a coarse grid; production callers must leave it alone,
    because the chosen step decides the stored basis bytes.
    """
    if not 2 <= bits <= 5:
        raise Pc1Error(f"basis bit depth {bits} outside the supported 2..5")
    import torch

    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    codes = np.asarray(basis_codes, dtype=np.float64).reshape(
        CARRIER_DIM, BASIS_PLANES, CARRIER_H, CARRIER_W
    )

    def rendered(atom: np.ndarray) -> np.ndarray:
        from torch.nn import functional

        field = functional.interpolate(
            torch.from_numpy(atom[None]).double(),
            size=(384, 512), mode="bicubic", align_corners=False,
        )[0].numpy().reshape(-1)
        field = field - field.mean()
        return field / max(float(np.sqrt((field**2).mean())), 1e-12)

    out = np.zeros(codes.shape, dtype=np.int32)
    for k in range(CARRIER_DIM):
        reference = rendered(codes[k])
        peak = float(np.abs(codes[k]).max())
        upper = peak / max(high, 1) * 3.0 + 1e-9
        best_cosine, best = -2.0, None
        for step in np.linspace(0.5, upper, grid_points):
            trial = np.clip(np.rint(codes[k] / step), low, high)
            if trial.std() == 0.0:
                continue
            cosine = float(np.dot(reference, rendered(trial)) / reference.size)
            if cosine > best_cosine:
                best_cosine, best = cosine, trial
        if best is None:
            raise Pc1Error(f"atom {k} collapses to a constant at {bits} bits")
        out[k] = best.astype(np.int32)
    return out


def dct_basis_2d(count: int, *, planes: int, height: int, width: int,
                 mode: str) -> np.ndarray:
    """A GENERATED basis -- zero video-derived bytes, so zero archive bytes.

    ``mode="luma"``   -- the lowest ``count`` separable 2-D DCT frequencies, the
                         same field on all three planes (an achromatic carrier).
    ``mode="planar"`` -- the lowest ``count // planes`` frequencies, each placed
                         on one plane alone (a carrier with chroma freedom).
    ``mode="opponent"``- the lowest ``count // planes`` frequencies crossed with
                         the three fixed generic colour directions
                         (luma, R-G, (R+G)/2-B).
    Ordered by ``u + v`` then ``max(u, v)`` then ``u`` so the set is the lowest
    frequencies under a total-degree order and depends on nothing but the shape.
    """
    order = sorted(
        ((u, v) for u in range(height) for v in range(width)),
        key=lambda uv: (uv[0] + uv[1], max(uv), uv[0]),
    )

    def cosines(size: int, index: int) -> np.ndarray:
        n = np.arange(size, dtype=np.float64)
        scale = math.sqrt((1.0 if index == 0 else 2.0) / size)
        return scale * np.cos(math.pi * (n + 0.5) * index / size)

    def field(u: int, v: int) -> np.ndarray:
        return np.outer(cosines(height, u), cosines(width, v))

    atoms = np.zeros((count, planes, height, width), dtype=np.float64)
    if mode == "luma":
        for k in range(count):
            u, v = order[k + 1]  # skip (0,0): a constant is killed by centring
            atoms[k, :, :, :] = field(u, v)[None]
    elif mode in ("planar", "opponent"):
        if count % planes:
            raise Pc1Error(f"{mode} basis needs count divisible by {planes}")
        per_plane = count // planes
        directions = (
            np.eye(planes)
            if mode == "planar"
            else np.array(
                [[1.0, 1.0, 1.0], [1.0, -1.0, 0.0], [0.5, 0.5, -1.0]]
            )
        )
        for k in range(per_plane):
            u, v = order[k + 1]
            base = field(u, v)
            for d in range(planes):
                atoms[k * planes + d] = directions[d][:, None, None] * base[None]
    else:
        raise Pc1Error(f"unknown generated-basis mode {mode!r}")
    return atoms


def quantize_generated(atoms: np.ndarray, bits: int = 5) -> np.ndarray:
    """Put a float generated basis on the signed code alphabet, per atom.

    Only used to render a generated basis through the SHIPPED decode path in the
    pricing control.  In the deployed V4 form the receiver generates the float
    atoms directly, so no quantisation happens at all and no bytes are stored.
    """
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    out = np.zeros(atoms.shape, dtype=np.int32)
    for k in range(atoms.shape[0]):
        peak = float(np.abs(atoms[k]).max())
        if peak <= 0.0:
            raise Pc1Error(f"generated atom {k} is identically zero")
        out[k] = np.clip(np.rint(atoms[k] / peak * high), low, high)
    return out


def svd_basis_from_state(basis_norm, codes: np.ndarray, coefficient_scales,
                         rank: int, basis_raw) -> np.ndarray:
    """Rank-``rank`` LEARNED basis: the top principal directions of the 600
    realized carrier fields, pulled back to the 24x32 code grid.

    The realized fields live in the span of the 12 shipped atoms, so the top
    ``rank`` right-singular directions are exactly ``W @ basis`` for a
    ``rank x 12`` mixing ``W`` recovered from the field-space SVD.  Applying the
    same ``W`` to ``basis_raw`` keeps the atoms on the 24x32 grid the container
    stores; the receiver's own per-atom centring and RMS then normalise them.
    """
    import torch

    coefficients = (
        torch.from_numpy(np.asarray(codes, dtype=np.float64))
        * coefficient_scales.double()[None]
    )
    # fields = coefficients @ Bflat, Bflat = (12, P).  SVD of the SMALL factor:
    # fields = U S V^T with V^T = (S^-1 U^T coefficients) Bflat, so the mixing is
    # W = S^-1 U^T coefficients, computed from the 600x600 / 12x12 side only.
    bflat = basis_norm.double().reshape(basis_norm.shape[0], -1)
    gram_b = bflat @ bflat.T
    inner = coefficients @ gram_b @ coefficients.T  # (600, 600) = fields fields^T
    eigenvalues, eigenvectors = torch.linalg.eigh(inner.double())
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    singular = eigenvalues[order].clamp_min(1e-300).sqrt()
    mixing = (eigenvectors[:, order].T @ coefficients) / singular[:, None]
    return (
        torch.einsum("rk,kchw->rchw", mixing, basis_raw.double()).numpy()
    )


@dataclass(frozen=True)
class VariantSpec:
    """One row of the prediction table, as an executable object."""

    name: str
    basis_bits: int | None  # None = basis unchanged
    generated_mode: str | None  # not None = V4 (no stored basis)
    svd_rank: int | None  # not None = V5
    lattice_factor: int  # 1 = shipped int12 lattice; 4 = the 10-bit lattice


VARIANTS: dict[str, VariantSpec] = {
    # The BASE row: the shipped basis and lattice, unchanged.  Every variant's
    # delta is measured against THIS row on THIS instrument, never against the
    # T4 number, which is a different axis
    # ([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]).
    "v0_base": VariantSpec("v0_base", None, None, None, 1),
    "v1_basis4": VariantSpec("v1_basis4", 4, None, None, 1),
    "v2_basis3": VariantSpec("v2_basis3", 3, None, None, 1),
    "v3_lattice10": VariantSpec("v3_lattice10", None, None, None, 4),
    # The lattice-factor LADDER.  The charter named x4; the factor is a free
    # continuous knob, so the neighbours are priced too rather than assumed
    # ([[m52]]: a bool is a UI over a continuum).
    "v3x2_lattice11": VariantSpec("v3x2_lattice11", None, None, None, 2),
    "v3x8_lattice9": VariantSpec("v3x8_lattice9", None, None, None, 8),
    "v3x16_lattice8": VariantSpec("v3x16_lattice8", None, None, None, 16),
    "v4_dct_luma": VariantSpec("v4_dct_luma", None, "luma", None, 1),
    "v4_dct_planar": VariantSpec("v4_dct_planar", None, "planar", None, 1),
    "v4_dct_opponent": VariantSpec("v4_dct_opponent", None, "opponent", None, 1),
    "v5_svd8": VariantSpec("v5_svd8", 5, None, 8, 1),
    # Compositions.  V1/V2 act on the basis payload and V3 on the coefficient
    # payload, so they touch DISJOINT bytes and compose; the composition is
    # PRICED exactly here rather than assumed additive ([[m164]]: union != sum).
    "v1v3_basis4_lattice10": VariantSpec("v1v3_basis4_lattice10", 4, None, None, 4),
    "v2v3_basis3_lattice10": VariantSpec("v2v3_basis3_lattice10", 3, None, None, 4),
    "v4v3_dct_luma_lattice10": VariantSpec(
        "v4v3_dct_luma_lattice10", None, "luma", None, 4
    ),
}


# --------------------------------------------------------------------------
# mode=rate -- EXACT archive bytes per variant, through the shipped container.
# --------------------------------------------------------------------------


def variant_basis_codes(spec: VariantSpec, shipped_codes: np.ndarray, *,
                        state, height: int, width: int) -> np.ndarray | None:
    """The variant's stored basis codes, or ``None`` when nothing is stored."""
    if spec.generated_mode is not None:
        return None
    atoms = np.asarray(shipped_codes, dtype=np.int32).reshape(
        CARRIER_DIM, BASIS_PLANES, height, width
    )
    if spec.svd_rank is not None:
        learned = svd_basis_from_state(
            state.basis_norm, state.codes, state.coefficient_scales,
            spec.svd_rank, state.basis_raw,
        )
        quantized = quantize_generated(learned, bits=spec.basis_bits or 5)
        padded = np.zeros_like(atoms)
        padded[: spec.svd_rank] = quantized
        return padded
    if spec.basis_bits is not None:
        return quantize_basis(atoms, spec.basis_bits)
    return atoms


def encode_basis_blob(basis_codes: np.ndarray, rr5) -> tuple[bytes, int, np.ndarray]:
    """Huffman payload, bit count and code-length vector for a basis."""
    symbols = basis_symbols_from_codes(basis_codes)
    histogram = np.bincount(symbols, minlength=int(rr5.BASIS_ALPHABET)).astype(np.int64)
    lengths = rr5.huffman_lengths_from_histogram(histogram)
    if int(lengths.max()) > 15:
        raise Pc1Error(
            f"variant Huffman code length {int(lengths.max())} exceeds the packed "
            "4-bit field; this basis cannot ride the shipped container"
        )
    payload, bits = rr5.huffman_encode(symbols, lengths)
    return payload, int(bits), np.asarray(lengths, dtype=np.uint8)


def assemble_archive_bytes(up3, body, *, carrier_body: bytes, runtime: Path,
                           residual_archive, container: tuple[bool, int, int]) -> bytes:
    """Compress a carrier body and rebuild archive.zip exactly as ``up3`` does.

    Used only for the container shapes ``up3.build_archive`` cannot express (the
    V4 body, which stores no basis at all).  Every other step -- rider order,
    brotli parameters, RX1 header, ZIP entry metadata -- is copied from the
    shipped body so the delta is attributable to the carrier alone.
    """
    import brotli

    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, _old = (
        body.rx1_header
    )
    encoded = up3._apply_entropy_riders(
        carrier_body, reserved=reserved, runtime_dir=runtime,
        residual_archive=residual_archive,
    )
    use_ck2, quality, lgwin = container
    raw = up3._ck2_interleave_planes(encoded) if use_ck2 else encoded
    stream = brotli.compress(raw, quality=quality, lgwin=lgwin)
    if brotli.decompress(stream) != raw:
        raise Pc1Error("brotli round-trip failed")
    reserved_out = (
        reserved | residual_archive.CK2_RESERVED_CARRIER_PLANE2
        if use_ck2
        else reserved & ~residual_archive.CK2_RESERVED_CARRIER_PLANE2
    )
    outer = b"".join((
        residual_archive.RX1_MODEL_HEADER.pack(
            magic, version, codec, table_mode, reserved_out,
            hpac_bytes, semantic_bytes, len(stream),
        ),
        body.hpac_stream, body.semantic_stream, stream, body.section_tail,
    ))
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        entry = zipfile.ZipInfo("p", date_time=tuple(body.zip_info["date_time"]))
        entry.compress_type = body.zip_info["compress_type"]
        entry.external_attr = body.zip_info["external_attr"]
        entry.create_system = body.zip_info["create_system"]
        archive.writestr(entry, outer)
    return buffer.getvalue()


def carrier_body_bytes(body, *, basis_bits: int, basis_blob: bytes,
                       packed_metadata: bytes, residual_bits: int,
                       rice_payload: bytes, scales: bytes) -> bytes:
    """The packed CAP1 carrier body, laid out exactly as the receiver reads it."""
    return b"".join((
        basis_bits.to_bytes(3, "little"),
        residual_bits.to_bytes(3, "little"),
        scales,
        packed_metadata,
        basis_blob,
        rice_payload,
        body.body_tail,
    ))


def run_rate(args) -> int:
    """EXACT archive bytes for every variant, at the SHIPPED coefficients.

    This is the rate CEILING, priced before any solver time is spent
    ([[m118]]: price the ceiling first, the denominator decides the answer).
    The pose side then has to be measured against it, not the other way round.
    """
    import dataclasses

    import ddm_up3_carrier_splice as up3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime)
    started = time.time()

    body = up3.parse_shipped_body(runtime, verify_sha=False)
    if body.archive_sha256 != args.expect_archive_sha256:
        raise Pc1Error(
            f"body sha {body.archive_sha256} != {args.expect_archive_sha256}"
        )
    ra, _cr, _ar1, _cp = up3._import_runtime(runtime)
    rr5, _carrier_repack = load_runtime_coders(runtime)
    up2, state = load_state(runtime, expect_sha=args.expect_archive_sha256)

    shipped_container = (body.ck2_carrier, 9, 16)
    identity = up3.build_archive(
        body, body.codes, runtime_dir=runtime,
        container_options=(shipped_container,),
    )
    if identity["archive_sha256"] != body.archive_sha256:
        raise Pc1Error(
            "CONTAINER IDENTITY CONTROL FAILED: rebuilding the shipped codes does "
            "not reproduce the shipped bytes, so no byte delta below would be "
            "attributable to the carrier"
        )

    # The shipped basis codes, as the RECEIVER decodes them (mode=probe wrote them
    # out of ``decode_compact_carrier``).  The re-encode control below proves this
    # array is the one the shipped payload carries, so it is checked, not trusted.
    probe = np.load(Path(args.carrier_state))
    shipped_basis_codes = probe["basis_codes"].astype(np.int32)

    control_blob, control_bits, control_lengths = encode_basis_blob(
        shipped_basis_codes, rr5
    )
    basis_reencode_identical = (
        control_blob == body.basis_blob
        and control_bits == body.basis_bits
        and np.array_equal(control_lengths, np.asarray(body.lengths, dtype=np.uint8))
    )
    if not basis_reencode_identical:
        raise Pc1Error(
            "CONTROL FAILED: re-encoding the shipped basis symbols does not "
            "reproduce the shipped basis payload; the basis pricer is untrusted"
        )

    rows: list[dict[str, Any]] = []
    for name in args.variants:
        spec = VARIANTS[name]
        codes = np.asarray(body.codes, dtype=np.int32)
        scales = body.scales
        note = ""
        if spec.lattice_factor != 1:
            factor = spec.lattice_factor
            codes = np.clip(
                np.rint(codes.astype(np.float64) / factor), -2048, 2047
            ).astype(np.int32)
            coefficient_scales = (
                np.frombuffer(body.scales[48:96], dtype="<f4").astype(np.float64)
                * factor
            )
            scales = body.scales[:48] + coefficient_scales.astype("<f4").tobytes()
            note = (
                f"coefficient lattice x{factor} coarser; the shipped codes are "
                "merely PROJECTED here -- the re-solve is what makes this a fair row"
            )

        variant_basis = variant_basis_codes(
            spec, shipped_basis_codes, state=state,
            height=CARRIER_H, width=CARRIER_W,
        )
        try:
            if variant_basis is None:
                # V4: no stored basis.  The receiver generates the atoms, so the
                # container carries a 1-BIT stub basis field.  It is a stub and
                # not a zero-length field because the shipped riders' shared
                # ``rr5.split_carrier_body`` refuses ``basis_bits == 0``; one
                # stub bit keeps the container shape byte-for-byte legal and the
                # DX2 coefficient rider running, at a cost of exactly one byte.
                ks, rice_payload, residual_bits = up3.encode_codes(
                    codes, factors=body.factors, biases=body.biases,
                    runtime_dir=runtime,
                )
                if int(ks.max()) - int(ks.min()) > 1:
                    raise Pc1Error(f"Rice ks {ks.tolist()} span the packed field")
                packed_metadata = up3.pack_cap1_metadata(
                    factors=body.factors, biases=body.biases,
                    lengths=body.lengths, ks=ks,
                )
                # The generated-basis body keeps the shipped Huffman table field
                # (it is inside the fixed-width packed metadata and costs nothing
                # extra) and stores basis_bits = 0 with an empty payload.
                cbody = carrier_body_bytes(
                    body, basis_bits=1, basis_blob=b"\x00",
                    packed_metadata=packed_metadata, residual_bits=residual_bits,
                    rice_payload=rice_payload, scales=scales,
                )
                # The RR5 rider re-codes the basis payload; with no basis to code
                # the rider must be OFF for this body.
                reserved_no_rr5 = body.rx1_header[4] & ~0x08
                stub = dataclasses.replace(
                    body,
                    rx1_header=(*body.rx1_header[:4], reserved_no_rr5,
                                *body.rx1_header[5:]),
                )
                archive_bytes = assemble_archive_bytes(
                    up3, stub, carrier_body=cbody, runtime=runtime,
                    residual_archive=ra, container=shipped_container,
                )
                note = (
                    "V4 stores NO basis: the receiver generates the atoms. The "
                    "RR5 arithmetic basis rider is switched OFF (it has nothing "
                    "to code), which is itself part of the measured delta."
                )
            else:
                blob, bits, lengths = encode_basis_blob(variant_basis, rr5)
                variant_body = dataclasses.replace(
                    body, basis_blob=blob, basis_bits=bits, lengths=lengths,
                    scales=scales,
                )
                built = up3.build_archive(
                    variant_body, codes, runtime_dir=runtime,
                    container_options=(shipped_container,), verify=False,
                )
                archive_bytes = built["archive_bytes"]
            size = len(archive_bytes)
            delta = size - CL2_ARCHIVE_BYTES
            rows.append({
                "variant": name,
                "ok": True,
                "archive_bytes": size,
                "delta_bytes": int(delta),
                "delta_score_rate": delta * BYTE_TO_SCORE,
                "basis_stored": variant_basis is not None,
                "basis_payload_bytes": (
                    0 if variant_basis is None
                    else (encode_basis_blob(variant_basis, rr5)[1] + 7) // 8
                ),
                "lattice_factor": spec.lattice_factor,
                "note": note,
            })
        except Exception as error:  # a variant the container refuses is a finding
            rows.append({
                "variant": name, "ok": False, "error": f"{type(error).__name__}: {error}",
                "note": note,
            })
        print(json.dumps(rows[-1]), flush=True)

    summary = {
        "schema": "ddm_pc1_rate.v1",
        "axis": "[exact local byte arithmetic]",
        "score_claim": False,
        "promotable": False,
        "body_archive_sha256": body.archive_sha256,
        "body_archive_bytes": CL2_ARCHIVE_BYTES,
        "container_identity_control": {
            "rebuilt_sha256": identity["archive_sha256"],
            "byte_identical": True,
            "container": identity["container"],
        },
        "basis_reencode_control": {
            "reproduces_shipped_basis_payload": bool(basis_reencode_identical),
            "shipped_basis_payload_bytes": len(body.basis_blob),
            "shipped_basis_bits": int(body.basis_bits),
        },
        "sections": {
            "hpac_stream_bytes": int(body.rx1_header[5]),
            "semantic_stream_bytes": int(body.rx1_header[6]),
            "carrier_stream_bytes": int(body.rx1_header[7]),
            "section_tail_bytes": len(body.section_tail),
            "rx1_header_bytes": int(ra.RX1_MODEL_HEADER.size),
            "zip_overhead_bytes": int(
                CL2_ARCHIVE_BYTES
                - (ra.RX1_MODEL_HEADER.size + body.rx1_header[5]
                   + body.rx1_header[6] + body.rx1_header[7]
                   + len(body.section_tail))
            ),
            "carrier_body_decoded_bytes": len(body.carrier_body),
            "basis_payload_bytes_huffman": len(body.basis_blob),
            "rice_payload_bytes": len(body.rice_payload),
            "scales_bytes": len(body.scales),
            "packed_metadata_bytes": len(body.packed_metadata),
            "selector_tail_bytes": len(body.body_tail),
        },
        "rows": rows,
        "wall_clock_seconds": time.time() - started,
    }
    (out_dir / "RATE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["sections"], indent=2))
    return 0


# --------------------------------------------------------------------------
# The variant INSTRUMENT -- jg5's instrument with the basis/lattice replaced.
# --------------------------------------------------------------------------


def variant_basis_raw(spec: VariantSpec, shipped_basis_codes: np.ndarray, state,
                      basis_scales: np.ndarray):
    """The variant's ``basis_raw``: what the receiver hands ``normalized_basis``.

    For a stored basis this is ``codes * basis_scales`` exactly as
    ``cpr1/inflate.py:245-246`` builds it.  For the generated V4 basis it is the
    float atoms themselves -- the receiver computes them, so no quantisation and
    no scales exist.
    """
    import torch

    if spec.generated_mode is not None:
        atoms = dct_basis_2d(
            CARRIER_DIM, planes=BASIS_PLANES, height=CARRIER_H, width=CARRIER_W,
            mode=spec.generated_mode,
        )
        return torch.from_numpy(atoms).float()
    codes = variant_basis_codes(
        spec, shipped_basis_codes, state=state, height=CARRIER_H, width=CARRIER_W,
    )
    grid = torch.from_numpy(
        np.asarray(codes, dtype=np.int64).reshape(
            CARRIER_DIM, BASIS_PLANES, CARRIER_H, CARRIER_W
        )
    ).float()
    return grid * torch.from_numpy(np.asarray(basis_scales, dtype=np.float32))[
        :, None, None, None
    ]


def build_variant_instrument(runtime: Path, spec: VariantSpec, *,
                             raw_path: Path, gt_cache: Path,
                             carrier_state_npz: Path,
                             expect_sha: str = CL2_ARCHIVE_SHA256):
    """jg5's instrument with the variant basis / lattice substituted.

    Everything else -- the frozen CPU PoseNet, the DALI GT targets, the shipped
    frame-1 renders, the selector -- is the SHIPPED instrument, so the only
    variable between the base row and the variant row is the carrier itself.
    """
    import dataclasses

    import ddm_br1_pose_basis_reorientation as br1
    import ddm_up2_shipping_pose_solve as up2
    import torch

    runtime = Path(runtime).resolve()
    observed = sha256_file(runtime / "archive.zip")
    if observed != expect_sha:
        raise Pc1Error(f"runtime archive sha {observed} != {expect_sha}")

    raw_path = Path(raw_path)
    expected_size = 2 * N_PAIRS * up2.CAMERA_H * up2.CAMERA_W * 3
    if raw_path.stat().st_size != expected_size:
        raise Pc1Error(
            f"raw decode is {raw_path.stat().st_size} B, expected {expected_size}"
        )
    raw = np.memmap(
        raw_path, dtype=np.uint8, mode="r",
        shape=(2 * N_PAIRS, up2.CAMERA_H, up2.CAMERA_W, 3),
    )

    state = up2.load_carrier_state(runtime, verify_archive=False)
    targets, lineage = up2.load_gt_poses(Path(gt_cache))
    if lineage != up2.LINEAGE_DALI:
        raise Pc1Error(
            f"GT lineage is {lineage}, not {up2.LINEAGE_DALI}: solving the "
            "contest-CPU objective would be a different object"
        )
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()

    stored = np.load(Path(carrier_state_npz))
    shipped_basis_codes = stored["basis_codes"].astype(np.int32)
    basis_scales = stored["basis_scales"].astype(np.float32)

    renderer = state.renderer
    base_norm = renderer.normalized_basis(state.basis_raw.clone())
    new_raw = variant_basis_raw(spec, shipped_basis_codes, state, basis_scales)
    new_norm = renderer.normalized_basis(new_raw.clone())

    coefficient_scales = state.coefficient_scales
    if spec.lattice_factor != 1:
        coefficient_scales = coefficient_scales * float(spec.lattice_factor)

    variant_state = dataclasses.replace(
        state, basis_raw=new_raw, basis_norm=new_norm,
        coefficient_scales=coefficient_scales,
    )

    # START CODES.  For a basis change the honest start is the least-squares
    # projection of the SHIPPED realized field onto the new basis (a warm start
    # that carries the incumbent solution across the object change, never a
    # cold zero); for a lattice change it is the shipped codes on the new lattice.
    shipped_codes = np.asarray(state.codes, dtype=np.int32)
    if spec.basis_bits is None and spec.generated_mode is None and spec.svd_rank is None:
        start = np.clip(
            np.rint(shipped_codes.astype(np.float64) / spec.lattice_factor),
            -2048, 2047,
        ).astype(np.int32)
        projection = {"kind": "lattice_rescale", "factor": spec.lattice_factor}
    else:
        old = base_norm.double().reshape(CARRIER_DIM, -1)
        new = new_norm.double().reshape(CARRIER_DIM, -1)
        gram_new = new @ new.T
        cross = new @ old.T
        old_coeff = (
            torch.from_numpy(shipped_codes.astype(np.float64))
            * state.coefficient_scales.double()[None]
        )
        solved = torch.linalg.solve(gram_new, cross @ old_coeff.T).T
        start = np.clip(
            np.rint((solved / coefficient_scales.double()[None]).numpy()),
            -2048, 2047,
        ).astype(np.int32)
        residual = old_coeff @ old - solved @ new
        signal = old_coeff @ old
        projection = {
            "kind": "least_squares_field_projection",
            "explained_energy": float(
                1.0 - (residual.square().sum() / signal.square().sum())
            ),
        }

    blow = br1.low_basis(variant_state)
    gram, bmat = br1.span_gram(blow)
    condition = float(torch.linalg.cond(gram.double()))
    if not math.isfinite(condition) or condition > 1e10:
        raise Pc1Error(
            f"variant basis Gram is numerically singular (cond {condition:.3e}); "
            "the min-image-norm step is undefined on this basis"
        )
    instrument = br1.Instrument(
        variant_state, raw, targets, posenet, blow, gram, bmat
    )
    meta = {
        "variant": spec.name,
        "runtime": str(runtime),
        "archive_sha256": observed,
        "raw_path": str(raw_path),
        "gt_lineage": lineage,
        "basis_gram_condition": condition,
        "start_codes_sha256": sha256_array(start),
        "start_projection": projection,
        "coefficient_scales": coefficient_scales.double().numpy().tolist(),
    }
    return instrument, start, meta


def load_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if rows_path.is_file():
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done[int(row["pair"])] = row
    return done


def run_solve(args) -> int:
    """Re-solve every assigned pair on the variant basis, ``jg5.refine_pair``.

    Resumable per pair: an interrupted shard resumes from ``rows.jsonl``.  The
    shard is STRIDED, never a contiguous block -- a contiguous prefix of this
    video is a different population and the bias is worst on the pose axis
    ([[m88]]/[[m96]]).
    """
    import ddm_jg5_pose_resolve_on_edited_renders as jg5

    spec = VARIANTS[args.variant]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    instrument, start_codes, meta = build_variant_instrument(
        Path(args.runtime), spec, raw_path=Path(args.raw),
        gt_cache=Path(args.gt_cache),
        carrier_state_npz=Path(args.carrier_state),
        expect_sha=args.expect_archive_sha256,
    )
    dd_threshold = jg5.materiality_dd_threshold(args.materiality_operating_point)

    if args.pairs:
        pairs = np.array(sorted(int(p) for p in args.pairs), dtype=np.int64)
    else:
        pairs = np.arange(args.shard, N_PAIRS, args.shards, dtype=np.int64)

    rows_path = out_dir / "rows.jsonl"
    np.save(out_dir / "start_codes.npy", start_codes)
    done = load_done(rows_path)
    started = time.time()
    print(json.dumps({
        "variant": args.variant, "pairs": int(pairs.size),
        "shard": args.shard, "shards": args.shards,
        "dd_threshold": dd_threshold, "already_done": len(done),
        "meta": meta,
    }, indent=2), flush=True)

    with rows_path.open("a", encoding="utf-8") as stream:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            row = jg5.refine_pair(
                instrument, int(pair), start_codes[int(pair)],
                dd_threshold=dd_threshold, outer_rounds=args.outer_rounds,
                max_gn_iterations=args.max_gn_iterations,
            )
            row["solver"] = "jg5.refine_pair"
            row["variant"] = args.variant
            row["body_archive_sha256"] = meta["archive_sha256"]
            done[int(pair)] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            elapsed = time.time() - started
            remaining = int(sum(1 for p in pairs if int(p) not in done))
            print(
                f"[{args.variant} {pairs.size - remaining}/{pairs.size}] "
                f"pair={int(pair)} start={row['start_d_pose']:.6e} "
                f"final={row['final_d_pose']:.6e} "
                f"stop={row['stop_reason']} "
                f"elapsed={elapsed / 60:.1f}m "
                f"eta={elapsed / max(1, position + 1) * remaining / 60:.1f}m",
                flush=True,
            )

    ordered = [done[int(p)] for p in pairs if int(p) in done]
    summary = {
        "schema": "ddm_pc1_solve.v1",
        "axis": "[macOS-CPU advisory, cpu_torch fp32, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "variant": args.variant,
        "solver_reference_form": (
            "ddm_jg5.refine_pair verbatim (br1 damped Gauss-Newton on the variant "
            "basis and lattice, +-2 polish, jg5 derived materiality stop)"
        ),
        "meta": meta,
        "dd_threshold": dd_threshold,
        "materiality_operating_point_d_pose": args.materiality_operating_point,
        "pairs": len(ordered),
        "shard": args.shard, "shards": args.shards,
        "sum_start_d_pose": float(sum(r["start_d_pose"] for r in ordered)),
        "sum_final_d_pose": float(sum(r["final_d_pose"] for r in ordered)),
        "stop_reasons": {
            reason: int(sum(1 for r in ordered if r.get("stop_reason") == reason))
            for reason in sorted({r.get("stop_reason", "?") for r in ordered})
        },
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("meta",)}, indent=2), flush=True)
    return 0


def run_measure(args) -> int:
    """EXACT n600 d_pose for a variant's codes -- the authority number.

    Non-STE, ``cpu_torch`` fp32, the frozen CPU PoseNet, the shipped frame-1
    renders, DALI GT: the same instrument the base row is measured on, so the
    delta is attributable to the carrier and nothing else.  There is no subset
    option: a pose number on a contiguous prefix of this video measures 2.54-4.21x
    HARDER than the population ([[m96]]) and a sampled subset still reports a
    different aggregate than the score sees ([[m88]]).
    """

    import ddm_up2_shipping_pose_solve as up2

    spec = VARIANTS[args.variant]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    instrument, start_codes, meta = build_variant_instrument(
        Path(args.runtime), spec, raw_path=Path(args.raw),
        gt_cache=Path(args.gt_cache),
        carrier_state_npz=Path(args.carrier_state),
        expect_sha=args.expect_archive_sha256,
    )
    if args.codes:
        codes = np.load(Path(args.codes)).astype(np.int32)
        source = str(args.codes)
    else:
        codes = start_codes
        source = "start_codes (least-squares projection; NOT re-solved)"
    if codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Pc1Error(f"codes must be (600, 12), got {codes.shape}")

    started = time.time()
    coefficients = up2.codes_to_coefficients(codes, instrument.state.coefficient_scales)
    pairs = np.arange(N_PAIRS, dtype=np.int64)
    per_pair, poses = up2.measure_pose(
        instrument.posenet, instrument.state, coefficients, instrument.raw,
        instrument.targets, pairs, batch_size=args.batch_size,
    )
    mean = float(per_pair.mean())
    summary = {
        "schema": "ddm_pc1_measure.v1",
        "axis": "[macOS-CPU advisory, cpu_torch fp32 authority backend, n600, DALI GT]",
        "score_claim": False,
        "promotable": False,
        "variant": args.variant,
        "codes_source": source,
        "codes_sha256": sha256_array(codes),
        "meta": meta,
        "pairs": int(N_PAIRS),
        "d_pose_n600_mean": mean,
        "d_pose_per_pair_max": float(per_pair.max()),
        "d_pose_per_pair_median": float(np.median(per_pair)),
        "pose_leg": pose_leg(mean),
        "elapsed_seconds": time.time() - started,
    }
    np.save(out_dir / "per_pair_d_pose.npy", per_pair)
    np.save(out_dir / "poses.npy", poses)
    np.save(out_dir / "measured_codes.npy", codes)
    (out_dir / "MEASURE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))
    return 0


def run_price(args) -> int:
    """Merge the solved shards, build the EXACT archive, close the arithmetic.

    Three numbers, three instruments, no mixing: the archive bytes come from the
    real container (identity-controlled), the d_pose comes from the frozen CPU
    PoseNet over all 600 pairs, and d_seg is carried UNCHANGED from the base row
    because a carrier edit only touches frame 0 and SegNet reads the LAST frame
    (``upstream/modules.py:109`` -- ``x[:, -1, ...]``), which is frame 1.
    """
    import dataclasses

    import ddm_up3_carrier_splice as up3

    spec = VARIANTS[args.variant]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime)

    merged: dict[int, dict[str, Any]] = {}
    for rows_path in args.rows:
        merged.update(load_done(Path(rows_path)))
    missing = [p for p in range(N_PAIRS) if p not in merged]
    if missing and not args.allow_partial:
        raise Pc1Error(
            f"{len(missing)} of {N_PAIRS} pairs are unsolved (first: {missing[:5]}); "
            "an n600 row cannot be closed on a subset -- rerun the missing shards "
            "or pass --allow-partial to record an explicitly PARTIAL receipt"
        )

    start_codes = np.load(Path(args.start_codes)).astype(np.int32)
    codes = start_codes.copy()
    for pair, row in merged.items():
        codes[int(pair)] = np.asarray(row["codes"], dtype=np.int32)
    np.save(out_dir / "solved_codes.npy", codes)

    body = up3.parse_shipped_body(runtime, verify_sha=False)
    if body.archive_sha256 != args.expect_archive_sha256:
        raise Pc1Error(f"body sha {body.archive_sha256} != expected")
    ra, _cr, _ar1, _cp = up3._import_runtime(runtime)
    rr5, _repack = load_runtime_coders(runtime)
    shipped_basis_codes = np.load(Path(args.carrier_state))["basis_codes"].astype(
        np.int32
    )
    _up2, state = load_state(runtime, expect_sha=args.expect_archive_sha256)

    shipped_container = (body.ck2_carrier, 9, 16)
    identity = up3.build_archive(
        body, body.codes, runtime_dir=runtime,
        container_options=(shipped_container,),
    )
    if identity["archive_sha256"] != body.archive_sha256:
        raise Pc1Error("container identity control FAILED")

    scales = body.scales
    if spec.lattice_factor != 1:
        coefficient_scales = (
            np.frombuffer(body.scales[48:96], dtype="<f4").astype(np.float64)
            * spec.lattice_factor
        )
        scales = body.scales[:48] + coefficient_scales.astype("<f4").tobytes()
    variant_basis = variant_basis_codes(
        spec, shipped_basis_codes, state=state, height=CARRIER_H, width=CARRIER_W
    )
    if variant_basis is None:
        raise Pc1Error(
            "the generated-basis body needs a receiver that generates its atoms; "
            "mode=rate prices it, mode=price does not build it"
        )
    blob, bits, lengths = encode_basis_blob(variant_basis, rr5)
    variant_body = dataclasses.replace(
        body, basis_blob=blob, basis_bits=bits, lengths=lengths, scales=scales
    )
    built = up3.build_archive(
        variant_body, codes, runtime_dir=runtime,
        container_options=(shipped_container,), verify=True,
    )
    archive_path = out_dir / "candidate_archive.zip"
    archive_path.write_bytes(built["archive_bytes"])

    base = json.loads(Path(args.base_measure).read_text())
    variant = json.loads(Path(args.variant_measure).read_text())
    d_pose_base = float(base["d_pose_n600_mean"])
    d_pose_variant = float(variant["d_pose_n600_mean"])
    delta_bytes = int(built["archive_size"] - CL2_ARCHIVE_BYTES)
    delta_rate = delta_bytes * BYTE_TO_SCORE
    delta_pose = pose_leg(d_pose_variant) - pose_leg(d_pose_base)
    net = delta_rate + delta_pose

    summary = {
        "schema": "ddm_pc1_price.v1",
        "axis": {
            "bytes": "[exact local byte arithmetic, receiver-verified parse-back]",
            "d_pose": "[macOS-CPU advisory, cpu_torch fp32, n600, DALI GT]",
            "d_seg": "carried unchanged from the base row (frame-0 only edit)",
        },
        "score_claim": False,
        "promotable": False,
        "variant": args.variant,
        "pairs_solved": len(merged),
        "pairs_missing": len(missing),
        "partial": bool(missing),
        "container_identity_control_passed": True,
        "parse_back_verified": True,
        "base": {
            "archive_bytes": CL2_ARCHIVE_BYTES,
            "archive_sha256": body.archive_sha256,
            "d_pose_n600": d_pose_base,
            "pose_leg": pose_leg(d_pose_base),
        },
        "candidate": {
            "archive_bytes": int(built["archive_size"]),
            "archive_sha256": built["archive_sha256"],
            "d_pose_n600": d_pose_variant,
            "pose_leg": pose_leg(d_pose_variant),
            "codes_sha256": sha256_array(codes),
            "rice_ks": built["rice_ks"],
            "basis_payload_bytes": (bits + 7) // 8,
        },
        "delta_bytes": delta_bytes,
        "delta_score_rate": delta_rate,
        "delta_score_pose": delta_pose,
        "net_delta_score": net,
        "admit_bar": ADMIT_BAR,
        "clears_admit_bar": bool(net <= ADMIT_BAR and not missing),
        "ft1_same_object_pose_ceiling": FT1_POSE_CEILING,
        "clears_ft1_pose_ceiling": bool(d_pose_variant <= FT1_POSE_CEILING),
        "break_even_d_pose": break_even_d_pose(d_pose_base, delta_rate),
    }
    (out_dir / "PRICE.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def run_coverage(args) -> int:
    """How much of the SHIPPED realized carrier field each generated basis spans.

    Selects V4's generated form at OPTIMAL FORM before any solver time is spent:
    a generated basis that cannot span the realized field cannot be rescued by
    the re-solve, and the covered-energy share is the cheapest available bound.
    """
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime)
    up2, state = load_state(runtime, expect_sha=args.expect_archive_sha256)
    renderer = state.renderer
    base_norm = renderer.normalized_basis(state.basis_raw.clone()).double()
    codes = np.asarray(state.codes, dtype=np.int32)
    coefficients = (
        torch.from_numpy(codes.astype(np.float64))
        * state.coefficient_scales.double()[None]
    )
    old = base_norm.reshape(CARRIER_DIM, -1)
    fields = coefficients @ old
    total = float(fields.square().sum())

    rows = []
    candidates: list[tuple[str, Any]] = [
        (mode, dct_basis_2d(CARRIER_DIM, planes=BASIS_PLANES, height=CARRIER_H,
                            width=CARRIER_W, mode=mode))
        for mode in ("luma", "planar", "opponent")
    ]
    for name, atoms in candidates:
        raw = torch.from_numpy(atoms).float()
        norm = renderer.normalized_basis(raw.clone()).double().reshape(
            CARRIER_DIM, -1
        )
        gram = norm @ norm.T
        projected = torch.linalg.solve(gram, norm @ fields.T).T @ norm
        residual = float((fields - projected).square().sum())
        rows.append({
            "generated_basis": name,
            "explained_energy_share": 1.0 - residual / total,
            "gram_condition": float(torch.linalg.cond(gram)),
        })
        print(json.dumps(rows[-1]), flush=True)

    summary = {
        "schema": "ddm_pc1_coverage.v1",
        "axis": "[scorer-free, exact linear algebra]",
        "score_claim": False,
        "note": (
            "explained energy of the 600 SHIPPED realized carrier fields inside "
            "each GENERATED basis span; an upper bound on what a generated basis "
            "can reproduce WITHOUT re-solving, not a d_pose claim"
        ),
        "rows": rows,
    }
    (out_dir / "COVERAGE.json").write_text(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    probe = sub.add_parser("probe", help="byte anatomy + SVD spectrum")
    probe.add_argument("--runtime", default=str(CL2_RUNTIME))
    probe.add_argument("--expect-archive-sha256", default=CL2_ARCHIVE_SHA256)
    probe.add_argument("--out", required=True)
    probe.set_defaults(func=run_probe)

    rate = sub.add_parser("rate", help="exact archive bytes per variant")
    rate.add_argument("--runtime", default=str(CL2_RUNTIME))
    rate.add_argument("--expect-archive-sha256", default=CL2_ARCHIVE_SHA256)
    rate.add_argument("--carrier-state", required=True,
                      help="carrier_state.npz written by mode=probe")
    rate.add_argument("--variants", nargs="+", default=sorted(VARIANTS))
    rate.add_argument("--out", required=True)
    rate.set_defaults(func=run_rate)

    coverage = sub.add_parser(
        "coverage", help="explained energy of the realized fields per generated basis"
    )
    coverage.add_argument("--runtime", default=str(CL2_RUNTIME))
    coverage.add_argument("--expect-archive-sha256", default=CL2_ARCHIVE_SHA256)
    coverage.add_argument("--out", required=True)
    coverage.set_defaults(func=run_coverage)

    solve = sub.add_parser("solve", help="re-solve pairs on a variant basis/lattice")
    solve.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    solve.add_argument("--runtime", default=str(CL2_RUNTIME))
    solve.add_argument("--expect-archive-sha256", default=CL2_ARCHIVE_SHA256)
    solve.add_argument("--raw", default=str(CL2_RAW))
    solve.add_argument(
        "--gt-cache",
        default="/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt",
    )
    solve.add_argument("--carrier-state", required=True)
    solve.add_argument("--pairs", nargs="*", default=None,
                       help="explicit pair list; default is the strided shard")
    solve.add_argument("--shard", type=int, default=0)
    solve.add_argument("--shards", type=int, default=1)
    solve.add_argument("--outer-rounds", type=int, default=40)
    solve.add_argument("--max-gn-iterations", type=int, default=400)
    solve.add_argument("--materiality-operating-point", type=float,
                       default=CL2_D_POSE_T4)
    solve.add_argument("--out", required=True)
    solve.set_defaults(func=run_solve)

    measure = sub.add_parser("measure", help="exact n600 d_pose for a code table")
    measure.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    measure.add_argument("--runtime", default=str(CL2_RUNTIME))
    measure.add_argument("--expect-archive-sha256", default=CL2_ARCHIVE_SHA256)
    measure.add_argument("--raw", default=str(CL2_RAW))
    measure.add_argument(
        "--gt-cache",
        default="/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt",
    )
    measure.add_argument("--carrier-state", required=True)
    measure.add_argument("--codes", default=None,
                         help="(600,12) .npy; default measures the start codes")
    measure.add_argument("--batch-size", type=int, default=8)
    measure.add_argument("--out", required=True)
    measure.set_defaults(func=run_measure)

    price = sub.add_parser("price", help="merge shards, build the archive, close DS")
    price.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    price.add_argument("--runtime", default=str(CL2_RUNTIME))
    price.add_argument("--expect-archive-sha256", default=CL2_ARCHIVE_SHA256)
    price.add_argument("--carrier-state", required=True)
    price.add_argument("--rows", nargs="+", required=True)
    price.add_argument("--start-codes", required=True)
    price.add_argument("--base-measure", required=True)
    price.add_argument("--variant-measure", required=True)
    price.add_argument("--allow-partial", action="store_true")
    price.add_argument("--out", required=True)
    price.set_defaults(func=run_price)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
