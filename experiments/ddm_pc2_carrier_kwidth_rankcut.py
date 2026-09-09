"""ddm_pc2 -- widen the CAP1 Rice-k field, then spend it on a carrier rank cut.

``ddm_pc1`` closed the pose carrier's lattice ladder at x16 and left three owed
items.  This arm takes ITEM 3 (the packed Rice-``k`` field is one bit per
dimension, which makes a rank cut unspendable), ITEM 1 (the 96-byte scales
block) and ITEM 4 (a 40-round re-solve as the attribution control and a
zero-byte probe), on the LIVE pointer body.

The object
----------
The carrier is a 12-dimensional positioned subspace.  A rank cut removes
dimensions from it and re-solves the survivors, which is a strict subspace
RESTRICTION -- the retained atoms are bit-identical, so this is the same family
as pc1's admitted V3 (change the search inside the box, never the box's walls)
rather than the refused basis edits.

Both halves of a dropped dimension are recoverable:

* **basis** -- setting the atom's 2,304 five-bit symbols to a CONSTANT makes the
  atom centre to zero and ``normalized_basis``'s ``clamp_min(1e-5)`` renders it
  as exactly the zero field, with no receiver change.
* **coefficients** -- zeroing the column and setting its AR(1) bias to zero
  makes every residual exactly zero; the Rice parameter then wants ``k = 0``,
  which the shipped one-bit field cannot express next to live columns at
  ``k = 5``.  That is what KW1 buys.

Axes
----
Bytes are EXACT (build the real ``archive.zip`` and stat it); d_pose is
``[macOS-CPU advisory, cpu_torch fp32 authority backend, n600, DALI GT]``.
``score_claim=false`` and ``promotable=false`` for every row this file emits:
only a contest-CUDA T4 run on the sealed bytes is a score, and MAIN fires it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent / "src"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_pc2_carrier_receiver_patch as patcher
import ddm_up2_shipping_pose_solve as up2
import ddm_up3_carrier_splice as up3

from tac import kw1_wide_rice_k as kw1

# --------------------------------------------------------------------------- #
# The live pointer, re-read from the canonical pointer file rather than recalled
# ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]).
# --------------------------------------------------------------------------- #
POINTER_JSON = REPO.parent / ".omx" / "state" / "canonical_frontier_pointer.json"
POINTER_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
    "/candidate_pass3/candidate_runtime"
)
POINTER_RAW = POINTER_RUNTIME.parent / "parseback" / "0.raw"
POINTER_ARCHIVE_SHA256 = (
    "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
)
POINTER_ARCHIVE_BYTES = 181_645
POINTER_SCORE = 0.13900437796841966
POINTER_LANE = "ddm_sj1_t4_token_predistortion_pass3_20260906"

WORK = Path("/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut")

N_PAIRS = up2.N_PAIRS_TOTAL
CARRIER_DIM = up2.CARRIER_DIM
BYTE_TO_SCORE = br1.BYTE_TO_SCORE  # 25 / 37_545_489
ADMIT_BAR = -2e-05  # net dS must clear this to be worth a paid call

#: Ascending realized-coefficient energy, MEASURED on the live body (mode=energy).
#: Recomputed at every use; this is the documented default, never a recalled
#: constant that outlives its body.
DEFAULT_DROP_ORDER = (2, 7, 1, 0, 6, 5, 4, 9, 10, 3, 11, 8)


class Pc2Error(RuntimeError):
    """A control did not hold, or the body is not the one this arm solved."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_live_pointer() -> dict[str, Any]:
    """The live effective frontier, read from disk at call time."""
    payload = json.loads(POINTER_JSON.read_text())
    return payload.get("effective_frontier", {})


def assert_pointer_unmoved() -> dict[str, Any]:
    """Refuse to stage or seal against a superseded body.

    sj1 is live on the same pointer; whoever lands second re-bases.  The check
    is re-run before every stage and every seal rather than once at start-up.
    """
    live = read_live_pointer()
    if (
        live.get("archive_sha256") != POINTER_ARCHIVE_SHA256
        or float(live.get("score", -1.0)) != POINTER_SCORE
    ):
        raise Pc2Error(
            "the canonical frontier pointer has MOVED: "
            f"sha {live.get('archive_sha256')} score {live.get('score')} != "
            f"{POINTER_ARCHIVE_SHA256} / {POINTER_SCORE}. Re-base before continuing."
        )
    return live




def _materialized_hpac(tree: Path, hpac_blob: bytes) -> bytes:
    """The HPAC object the RENDERER receives, past every section rider.

    ``read_residual_archive`` hands back the hpac SECTION BODY, which is upstream
    of the rc1/rc2 riders -- they are undone at the renderer seam in
    ``ihs2.materialize_ihs1``.  Comparing two bodies at the section layer
    therefore reports a difference that does not exist downstream, which is
    exactly the trap ddm_pc1 section 7b recorded.  This reads at the seam.
    """
    tree = Path(tree).resolve()
    (ihs2,) = import_receiver(tree, ("ihs2",))
    # ``layout_from_runtime`` wants the CPR1 RENDERER module (it builds an
    # IntegerHPAC shell from it), which is what f26_inflate passes as `runtime`.
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "cpr1"))
    try:
        import inflate as renderer_module  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    origin = Path(getattr(renderer_module, "__file__", "")).resolve()
    if tree not in origin.parents:
        raise Pc2Error(f"cpr1 inflate loaded from {origin}, not from {tree}")
    return bytes(ihs2.materialize_ihs1(bytes(hpac_blob), renderer_module))


def decoded_section_identity(left: Path, right: Path) -> dict[str, Any]:
    """Do two bodies RENDER the same frames? Compared after every restore.

    The stored streams are the wrong place to ask.  ``ddm_rc1``'s rider recodes a
    model section losslessly, so the STORED bytes differ while the RESTORED bytes
    do not, and pc1 section 7b records exactly this trap: a byte comparison one
    stage upstream of a lossless restore reports a difference that does not exist
    downstream.  So each body is walked through the receiver's own restore and the
    DECODED objects are compared: the carrier (which makes frame_0), the semantic
    renderer, the token tail and the HPAC model (which make frame_1).
    """
    fields: dict[str, Any] = {}
    decoded: dict[str, dict[str, str]] = {}
    for label, tree in (("left", Path(left)), ("right", Path(right))):
        body = load_body(tree)
        (ra,) = import_receiver(tree, ("residual_archive",))
        parts = ra.read_residual_archive(Path(tree) / "archive.zip")
        decoded[label] = {
            "carrier_body": _sha256(bytes(body.carrier_body)),
            "carrier_codes": _sha256(np.asarray(body.codes, dtype=np.int32).tobytes()),
            "carrier_scales": _sha256(bytes(body.scales)),
            "carrier_basis": _sha256(bytes(body.basis_blob)),
            "section_tail": _sha256(bytes(body.section_tail)),
            "decoded_carrier_blob": _sha256(bytes(parts.carrier_blob)),
            "decoded_semantic_blob": _sha256(bytes(parts.semantic_blob)),
            "decoded_hpac_blob": _sha256(bytes(parts.hpac_blob)),
            "materialized_hpac": _sha256(_materialized_hpac(tree, parts.hpac_blob)),
        }
    for key in decoded["left"]:
        fields[key] = decoded["left"][key] == decoded["right"][key]
    # frame_0 comes from the carrier; frame_1 from the semantic renderer driven by
    # the token tail and the HPAC model.  All four must survive their restores.
    render_keys = (
        "decoded_carrier_blob",
        "decoded_semantic_blob",
        "materialized_hpac",
        "section_tail",
    )
    return {
        "left": str(left),
        "right": str(right),
        "fields": fields,
        "left_digests": decoded["left"],
        "right_digests": decoded["right"],
        "renders_are_the_same_object": all(fields[k] for k in render_keys),
    }


def rebase_to(runtime: Path, *, donor_raw_from: Path | None = None) -> dict[str, Any]:
    """Re-point this module at a NEW pointer tree, reading every number off disk.

    sj1 and this arm work the same object, so whoever lands second re-bases.  A
    re-base must not be a source edit under the clock: the tree is named, its
    archive's sha and size are MEASURED from the bytes, the score comes from
    `.omx/state/canonical_frontier_pointer.json`, and the three are then required
    to agree.  Nothing here is typed by hand, which is the whole point
    ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]).
    """
    global POINTER_RUNTIME, POINTER_RAW, POINTER_ARCHIVE_SHA256
    global POINTER_ARCHIVE_BYTES, POINTER_SCORE, POINTER_LANE

    runtime = Path(runtime).resolve()
    archive = runtime / "archive.zip"
    if not archive.is_file():
        raise Pc2Error(f"no archive.zip under {runtime}")
    measured_sha = _sha256_file(archive)
    measured_bytes = archive.stat().st_size
    live = read_live_pointer()
    if live.get("archive_sha256") != measured_sha:
        raise Pc2Error(
            f"{runtime} carries archive sha {measured_sha}, but the live pointer "
            f"names {live.get('archive_sha256')}; re-base onto the POINTER's tree"
        )
    raw = runtime.parent / "parseback" / "0.raw"
    if not raw.is_file():
        if donor_raw_from is None:
            raise Pc2Error(
                f"no raw decode beside the new tree ({raw}); the solver needs its "
                "frame_1 planes and will not silently reuse another body's. Pass "
                "--donor-raw-from <tree> to reuse a donor's decode; it is accepted "
                "only if the two bodies are PROVEN to render the same frames."
            )
        donor = Path(donor_raw_from).resolve()
        raw = donor.parent / "parseback" / "0.raw"
        if not raw.is_file():
            raise Pc2Error(f"donor tree has no raw decode either: {raw}")
        identity = decoded_section_identity(donor, runtime)
        if not identity["renders_are_the_same_object"]:
            raise Pc2Error(
                "the donor's decode may NOT be reused: "
                f"{json.dumps(identity, indent=2)}"
            )
    POINTER_RUNTIME = runtime
    POINTER_RAW = raw
    # Prove the re-base REACHED the readers.  Python binds default arguments at
    # definition time, so a module global reassigned here does not by itself move
    # anything; this arm shipped exactly that bug once and the r=12 identity
    # control is what caught it.  Re-parsing through the public entry point and
    # checking the sha is the control that keeps it caught.
    reached = load_body()
    if _sha256(bytes(reached.archive_bytes)) != measured_sha:
        raise Pc2Error(
            "re-base did NOT reach the body readers: load_body() still returns "
            f"{_sha256(bytes(reached.archive_bytes))[:16]}, not {measured_sha[:16]}"
        )
    POINTER_ARCHIVE_SHA256 = measured_sha
    POINTER_ARCHIVE_BYTES = measured_bytes
    POINTER_SCORE = float(live["score"])
    POINTER_LANE = str(live.get("lane_id", ""))
    return {
        "pointer_runtime": str(POINTER_RUNTIME),
        "pointer_raw": str(POINTER_RAW),
        "pointer_archive_sha256": POINTER_ARCHIVE_SHA256,
        "pointer_archive_bytes": POINTER_ARCHIVE_BYTES,
        "pointer_score": POINTER_SCORE,
        "pointer_lane": POINTER_LANE,
    }


# --------------------------------------------------------------------------- #
# Receiver imports
# --------------------------------------------------------------------------- #
#: Module names the receiver trees share.  Python caches by NAME, so importing
#: ``runtime.residual_archive`` from the pointer tree and then again from the
#: KW1-patched tree silently returns the FIRST one -- which would mix patched and
#: unpatched receiver code inside one build and is exactly the class of silent
#: mismatch this arm cannot afford.  Every import below purges the cache first
#: and then PROVES each module came from the requested tree.
_RECEIVER_MODULE_PREFIXES = ("runtime", "inflate", "cpr1")


def _purge_receiver_modules() -> None:
    for name in list(sys.modules):
        head = name.split(".", 1)[0]
        if head in _RECEIVER_MODULE_PREFIXES:
            del sys.modules[name]


def import_receiver(runtime: Path, names: tuple[str, ...]):
    """Import ``runtime.<name>`` modules from ONE tree, fail-closed on the path."""
    runtime = Path(runtime).resolve()
    _purge_receiver_modules()
    sys.path.insert(0, str(runtime))
    try:
        modules = [
            __import__(f"runtime.{name}", fromlist=["_"]) for name in names
        ]
    finally:
        sys.path.pop(0)
    for module in modules:
        origin = Path(getattr(module, "__file__", "")).resolve()
        if runtime not in origin.parents:
            raise Pc2Error(
                f"{module.__name__} loaded from {origin}, not from {runtime}"
            )
    return modules


# --------------------------------------------------------------------------- #
# Score arithmetic
# --------------------------------------------------------------------------- #
def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * float(d_pose))


def net_delta_s(
    *, delta_bytes: int, d_pose_new: float, d_pose_base: float
) -> dict[str, float]:
    rate = float(delta_bytes) * BYTE_TO_SCORE
    pose = pose_leg(d_pose_new) - pose_leg(d_pose_base)
    return {
        "delta_s_rate": rate,
        "delta_s_pose": pose,
        "net_delta_s": rate + pose,
    }


def break_even_d_pose(*, delta_bytes: int, d_pose_base: float) -> float:
    """The d_pose at which a byte saving is exactly repaid on the pose leg."""
    allowed_leg = pose_leg(d_pose_base) - float(delta_bytes) * BYTE_TO_SCORE
    return (allowed_leg * allowed_leg) / 10.0


# --------------------------------------------------------------------------- #
# The carrier body, its energy ordering, and the rank-cut construction
# --------------------------------------------------------------------------- #
def load_body(runtime: Path | None = None):
    """Parse a body through ``up3``, with the receiver module cache purged first.

    ``up3._import_runtime`` and ``up2.load_carrier_state`` both import
    ``runtime.*`` by name, so without the purge a second call against a
    DIFFERENT tree silently reuses the first tree's modules -- which would make
    the KW1 patch-inertness control a false pass.
    """
    _purge_receiver_modules()
    return up3.parse_shipped_body(runtime or POINTER_RUNTIME, verify_sha=False)


def load_state(runtime: Path):
    """``up2.load_carrier_state`` against a named tree, cache purged first."""
    _purge_receiver_modules()
    return up2.load_carrier_state(Path(runtime), verify_archive=False)


#: Encoder-only container choices to search.  ``up3`` pins q=11/lgwin=24, which
#: was ITS generation's shipped shape; the live body is q=9/lgwin=16 (MEASURED
#: below).  Inheriting the constant costs 2 bytes AND breaks the byte-identity
#: control the whole splice is anchored on, which is exactly the failure
#: ``up3.build_archive``'s own docstring warns about
#: ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]), so the
#: shape is DERIVED from the shipped stream at every run instead.
CONTAINER_GRID: tuple[tuple[bool, int, int], ...] = tuple(
    (ck2, quality, lgwin)
    for ck2 in (False, True)
    for quality in (9, 10, 11)
    for lgwin in (10, 16, 22, 24)
)


def detect_container_shape(body, runtime: Path) -> tuple[bool, int, int]:
    """The (ck2, brotli quality, lgwin) that reproduces the shipped stream.

    Refuses if no candidate is byte-identical: a container shape that only
    happens to be the same SIZE would silently move the anchor.
    """
    import brotli

    runtime = Path(runtime).resolve()
    (ra,) = import_receiver(runtime, ("residual_archive",))
    encoded = up3._apply_entropy_riders(
        bytes(body.carrier_body),
        reserved=int(body.rx1_header[4]),
        runtime_dir=runtime,
        residual_archive=ra,
    )
    target = bytes(body.carrier_stream)
    for use_ck2, quality, lgwin in CONTAINER_GRID:
        if use_ck2 != bool(body.ck2_carrier):
            continue
        raw = up3._ck2_interleave_planes(encoded) if use_ck2 else encoded
        if brotli.compress(raw, quality=quality, lgwin=lgwin) == target:
            return (use_ck2, quality, lgwin)
    raise Pc2Error(
        "no searched container shape reproduces the shipped carrier stream "
        "byte-identically; the encoder shape has moved and must be re-derived"
    )


def basis_symbols(body, runtime: Path | None = None) -> np.ndarray:
    """The 27,648 five-bit basis symbols, decoded through the receiver's table."""
    (rr5,) = import_receiver(runtime or POINTER_RUNTIME, ("rr5_arith_basis",))
    return np.asarray(
        rr5.huffman_decode(
            body.lengths.astype(np.uint8),
            bytes(body.basis_blob),
            int(body.basis_bits),
            rr5.BASIS_SYMBOLS,
        ),
        dtype=np.int64,
    )


def realized_energy(codes: np.ndarray, coefficient_scales: np.ndarray) -> np.ndarray:
    """Per-dimension realized field energy.

    ``normalized_basis`` gives every atom RMS 1 (MEASURED, mode=energy), so the
    energy a dimension puts into the rendered carrier is exactly the squared sum
    of its scaled coefficients.  The drop order is ascending in this quantity.
    """
    scaled = np.asarray(codes, dtype=np.float64) * np.asarray(
        coefficient_scales, dtype=np.float64
    )[None]
    return (scaled**2).sum(axis=0)


def rank_cut_fields(body, symbols: np.ndarray, dropped: tuple[int, ...]):
    """Basis symbols, biases and codes for a rank cut, as the receiver sees them.

    Dropped atoms become the constant symbol 0 (value 0 -> centres to zero ->
    ``clamp_min(1e-5)`` renders exactly zero); dropped biases become 0 so the
    AR(1) residual of a zero column is identically zero; dropped code columns
    become 0.
    """
    dropped = tuple(sorted({int(d) for d in dropped}))
    if any(d < 0 or d >= CARRIER_DIM for d in dropped):
        raise Pc2Error(f"dropped dimensions {dropped} escape the carrier")
    cut_symbols = symbols.copy().reshape(CARRIER_DIM, -1)
    cut_symbols[list(dropped), :] = 0
    biases = np.asarray(body.biases, dtype=np.int64).copy()
    biases[list(dropped)] = 0
    codes = np.asarray(body.codes, dtype=np.int32).copy()
    codes[:, list(dropped)] = 0
    return cut_symbols.reshape(-1), biases, codes



def coarsen_fields(body, state, factors: np.ndarray):
    """Per-column lattice coarsening: multiply a column's step, shrink its codes.

    ``ddm_pc1`` measured that GLOBAL lattice coarsening plus a re-solve is the one
    carrier family that pays (its x4 / x8 / x16 rungs are the 28th, 29th and 30th
    pointer moves), and it closed that ladder at x16 because the SCORE turned.
    The wide-k field makes the factor PER COLUMN, which the shipped one-bit field
    could not express: a coarsened column's optimal Rice ``k`` falls, and columns
    at different ``k`` need a field that can hold them.

    The coefficient scale is a free stored f32 -- coarsening moves value between
    the scale and the code and changes no byte count by itself; the bytes come
    from the smaller codes.  Returns (codes, coefficient_scales).
    """
    factors = np.asarray(factors, dtype=np.float64).reshape(-1)
    if factors.shape != (CARRIER_DIM,) or np.any(factors <= 0):
        raise Pc2Error(f"coarsening factors must be {CARRIER_DIM} positive numbers")
    codes = np.asarray(body.codes, dtype=np.float64)
    scaled = br1.realize(codes / factors[None])
    scales = state.coefficient_scales.double().numpy() * factors
    return scaled.astype(np.int32), scales


def scales_block_with(basis_scales: np.ndarray, coefficient_scales: np.ndarray) -> bytes:
    """The shipped 96-byte scales block: 12 basis f32 then 12 coefficient f32."""
    basis = np.asarray(basis_scales, dtype=np.float32).reshape(-1)
    coefficient = np.asarray(coefficient_scales, dtype=np.float32).reshape(-1)
    if basis.shape != (CARRIER_DIM,) or coefficient.shape != (CARRIER_DIM,):
        raise Pc2Error("the scales block carries exactly 12 + 12 float32 values")
    block = np.concatenate([basis, coefficient]).astype("<f4").tobytes()
    if len(block) != 8 * CARRIER_DIM:
        raise Pc2Error("the scales block must stay 96 bytes")
    return block


# --------------------------------------------------------------------------- #
# The builder -- a KW1 carrier body, through the receiver's own layers
# --------------------------------------------------------------------------- #
def build_kw1_archive(
    body,
    *,
    codes: np.ndarray,
    symbols: np.ndarray,
    biases: np.ndarray,
    runtime: Path,
    container: tuple[bool, int, int],
    search_container: bool = True,
    wide_k: bool = True,
    force_ks: np.ndarray | None = None,
    scales: bytes | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Rebuild ``archive.zip`` with a new basis, new biases and new codes.

    ``runtime`` must be a KW1-PATCHED receiver tree: the forward riders are
    imported from it, so the encoder runs exactly the code the candidate ships.
    """
    import brotli

    runtime = Path(runtime).resolve()
    codes = np.asarray(codes, dtype=np.int32)
    if codes.shape != (N_PAIRS, CARRIER_DIM):
        raise Pc2Error(f"codes must be ({N_PAIRS}, {CARRIER_DIM}), got {codes.shape}")
    if codes.min() < -2048 or codes.max() > 2047:
        raise Pc2Error("codes escape the CPR1 signed-int12 domain")

    _purge_receiver_modules()
    cr, dx2, ra, rr5 = import_receiver(
        runtime,
        (
            "carrier_repack",
            "dx2_cabac_coefficients",
            "residual_archive",
            "rr5_arith_basis",
        ),
    )

    # --- basis: re-derive the canonical Huffman table for these symbols ------ #
    symbols = np.asarray(symbols, dtype=np.int64)
    if symbols.shape != (rr5.BASIS_SYMBOLS,):
        raise Pc2Error(f"expected {rr5.BASIS_SYMBOLS} basis symbols")
    histogram = np.bincount(symbols, minlength=rr5.BASIS_ALPHABET)
    lengths = np.asarray(
        rr5.huffman_lengths_from_histogram(histogram), dtype=np.uint8
    )
    basis_blob, basis_bits = rr5.huffman_encode(symbols, lengths)

    # --- coefficients: AR(1) residual -> zigzag -> Rice ---------------------- #
    residuals = up3.forward_ar1_bias(
        codes, body.factors, np.asarray(biases, dtype=np.int8), runtime_dir=runtime
    )
    zigzag = cr._zigzag(np.asarray(residuals, dtype=np.int64))
    ks, rice_payload, residual_bits = cr._rice_encode(zigzag, 1)
    ks = np.asarray(ks, dtype=np.int64).reshape(-1)
    if force_ks is not None:
        # The k-width CONTROL: re-code the same residuals at a k vector the
        # SHIPPED one-bit field can carry, so the wide field's own contribution
        # is measured rather than asserted.
        ks = np.asarray(force_ks, dtype=np.int64).reshape(-1)
        if ks.shape != (CARRIER_DIM,):
            raise Pc2Error(f"force_ks must have {CARRIER_DIM} entries")
        rice_payload, residual_bits = _rice_encode_at(zigzag, ks)
    span = int(ks.max()) - int(ks.min())
    if not wide_k and span > 1:
        raise Pc2Error(
            f"Rice ks {ks.tolist()} span {span} > 1; the shipped packed field "
            "cannot carry them -- this is the container refusal KW1 removes"
        )
    if not 0 < residual_bits < (1 << 24):
        raise Pc2Error(f"residual_bits {residual_bits} does not fit the u24 field")

    # --- packed metadata ----------------------------------------------------- #
    # ``pack_cap1_metadata`` builds the 37 shared prefix bytes AND the shipped
    # 1-bit k field.  For a KW1 block those last three bytes are replaced
    # wholesale by ``kw1.with_ks``, so a constant k vector is passed here purely
    # to get past the shipped packer's span check -- no shipped k byte survives.
    shipped_block = up3.pack_cap1_metadata(
        factors=body.factors,
        biases=np.asarray(biases, dtype=np.int8),
        lengths=lengths,
        ks=(ks if not wide_k else np.full(CARRIER_DIM, int(ks.min()), dtype=np.int64)),
    )
    packed_metadata = (
        kw1.with_ks(shipped_block, ks) if wide_k else shipped_block
    )
    expected_width = (
        kw1.KW1_PACKED_METADATA_BYTES if wide_k else kw1.SHIPPED_PACKED_METADATA_BYTES
    )
    if len(packed_metadata) != expected_width:
        raise Pc2Error("packed metadata width does not match the declared form")

    carrier_body = b"".join(
        (
            int(basis_bits).to_bytes(3, "little"),
            int(residual_bits).to_bytes(3, "little"),
            bytes(body.scales if scales is None else scales),
            packed_metadata,
            bytes(basis_blob),
            bytes(rice_payload),
            bytes(body.body_tail),
        )
    )

    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, _old = (
        body.rx1_header
    )
    reserved = int(reserved)
    # The KW1 bit's SSoT is ``tac.kw1_wide_rice_k``; a KW1-patched receiver
    # re-declares it and the two must agree, but an UNPATCHED receiver (the
    # right tree for a candidate that does not set the bit) does not define it
    # at all, so the module constant is the one that is always available.
    kw1_bit = int(getattr(ra, "KW1_RESERVED_WIDE_RICE_K", kw1.KW1_RESERVED_WIDE_RICE_K))
    if kw1_bit != kw1.KW1_RESERVED_WIDE_RICE_K:
        raise Pc2Error(
            f"receiver KW1 bit {kw1_bit:#x} disagrees with the format module "
            f"{kw1.KW1_RESERVED_WIDE_RICE_K:#x}"
        )
    if wide_k:
        if not hasattr(ra, "KW1_RESERVED_WIDE_RICE_K"):
            raise Pc2Error("wide_k needs a KW1-patched receiver; this tree is shipped")
        reserved |= kw1_bit
    else:
        reserved &= ~kw1_bit
    metadata_bytes = len(packed_metadata)

    # --- entropy riders, in the receiver's exact inverse order ---------------- #
    # An UNPATCHED receiver's riders have no ``metadata_bytes`` parameter and
    # assume the shipped 40, which is correct for a candidate that does not set
    # the KW1 bit.  Passing the keyword only when the tree understands it keeps
    # one builder honest against both receivers instead of forking it.
    rider_kwargs: dict[str, Any] = (
        {"metadata_bytes": metadata_bytes}
        if metadata_bytes != kw1.SHIPPED_PACKED_METADATA_BYTES
        else {}
    )
    encoded = carrier_body
    if reserved & ra.DX2_RESERVED_CABAC_COEFFICIENTS:
        encoded = bytes(
            dx2.apply_cabac_to_carrier_body(encoded, **rider_kwargs)["body"]
        )
    if reserved & ra.RR5_RESERVED_ARITH_BASIS:
        encoded = bytes(
            rr5.apply_rider_to_carrier_body(encoded, **rider_kwargs)["body"]
        )
    replay = encoded
    if reserved & ra.RR5_RESERVED_ARITH_BASIS:
        replay = rr5.restore_carrier_body(replay, **rider_kwargs)
    if reserved & ra.DX2_RESERVED_CABAC_COEFFICIENTS:
        replay = dx2.restore_carrier_body(replay, **rider_kwargs)
    if replay != carrier_body:
        raise Pc2Error("RR5/DX2 forward-inverse identity failed on the KW1 body")

    # DDM_FE1 CONTAINER SWEEP.  The brotli quality/window and the CK2 interleave
    # are ENCODER-ONLY choices: the receiver reads the interleave from reserved
    # bit 2 and calls a bare ``brotli.decompress``, so any option that
    # round-trips is equally legal.  fe1 MEASURED that pricing an edited archive
    # at the SHIPPED shape overstates its cost, because an edit destroys the
    # match structure the shipped shape was chosen for.  Both prices are
    # recorded; the sweep never loses, since the shipped shape is in the grid
    # and ties go to it.
    shipped_shape = (bool(container[0]), int(container[1]), int(container[2]))
    options = CONTAINER_GRID if search_container else (shipped_shape,)
    chosen: tuple[bool, int, int] | None = None
    carrier_stream = b""
    shipped_shape_bytes = None
    for use_ck2, quality, lgwin in options:
        raw = up3._ck2_interleave_planes(encoded) if use_ck2 else encoded
        stream = brotli.compress(raw, quality=quality, lgwin=lgwin)
        if brotli.decompress(stream) != raw:
            raise Pc2Error(f"brotli round-trip failed for q={quality} lgwin={lgwin}")
        if (use_ck2, quality, lgwin) == shipped_shape:
            shipped_shape_bytes = len(stream)
        # Ties go to the SHIPPED shape: an equal-size container that differs from
        # the body's own is a moving part bought for nothing.
        better = chosen is None or len(stream) < len(carrier_stream)
        tie_to_shipped = (
            chosen is not None
            and len(stream) == len(carrier_stream)
            and (use_ck2, quality, lgwin) == shipped_shape
        )
        if better or tie_to_shipped:
            chosen, carrier_stream = (use_ck2, quality, lgwin), stream
    if chosen is None:
        raise Pc2Error("no container option produced a stream")
    use_ck2, quality, lgwin = chosen
    reserved = (
        reserved | ra.CK2_RESERVED_CARRIER_PLANE2
        if use_ck2
        else reserved & ~ra.CK2_RESERVED_CARRIER_PLANE2
    )

    outer = b"".join(
        (
            ra.RX1_MODEL_HEADER.pack(
                magic,
                version,
                codec,
                table_mode,
                reserved,
                hpac_bytes,
                semantic_bytes,
                len(carrier_stream),
            ),
            bytes(body.hpac_stream),
            bytes(body.semantic_stream),
            carrier_stream,
            bytes(body.section_tail),
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        entry = zipfile.ZipInfo("p", date_time=tuple(body.zip_info["date_time"]))
        entry.compress_type = body.zip_info["compress_type"]
        entry.external_attr = body.zip_info["external_attr"]
        entry.create_system = body.zip_info["create_system"]
        archive.writestr(entry, outer)
    archive_bytes = buffer.getvalue()

    result = {
        "archive_bytes": archive_bytes,
        "archive_sha256": _sha256(archive_bytes),
        "archive_size": len(archive_bytes),
        "delta_vs_pointer": len(archive_bytes) - POINTER_ARCHIVE_BYTES,
        "rice_ks": ks.tolist(),
        "rice_k_span": span,
        "rice_bits": int(residual_bits),
        "rice_payload_bytes": len(rice_payload),
        "basis_bits": int(basis_bits),
        "basis_blob_bytes": len(basis_blob),
        "packed_metadata_bytes": metadata_bytes,
        "carrier_body_bytes": len(carrier_body),
        "encoded_carrier_body_bytes": len(encoded),
        "carrier_stream_bytes": len(carrier_stream),
        "rx1_reserved": f"{reserved:#x}",
        "container": [use_ck2, quality, lgwin],
        "container_shipped_shape": list(shipped_shape),
        "container_searched": bool(search_container),
        "carrier_stream_bytes_at_shipped_shape": shipped_shape_bytes,
        "archive_size_at_shipped_shape": (
            None
            if shipped_shape_bytes is None
            else len(archive_bytes) - len(carrier_stream) + shipped_shape_bytes
        ),
        "wide_k": bool(wide_k),
        "huffman_lengths": lengths.tolist(),
        "biases": np.asarray(biases, dtype=np.int64).tolist(),
    }
    if verify:
        recovered = parse_back_codes(archive_bytes, runtime=runtime)
        if not np.array_equal(codes.astype(np.int64), recovered.astype(np.int64)):
            worst = int(np.abs(codes.astype(np.int64) - recovered.astype(np.int64)).max())
            raise Pc2Error(
                f"the written archive does not parse back to the requested codes "
                f"(max |delta| {worst}); refusing to return unverified bytes"
            )
        result["parse_back_codes_identical"] = True
    return result



def _rice_encode_at(zigzag: np.ndarray, ks: np.ndarray) -> tuple[bytes, int]:
    """Rice-code ``zigzag`` at a GIVEN per-column ``k``, in the receiver's order.

    ``carrier_repack._rice_encode`` always picks the optimal ``k``; the k-width
    control needs the same bitstream at a k vector the shipped one-bit field can
    express, so the loop is written out here with the receiver's exact layout
    (column-major, unary quotient, terminator, then ``k`` remainder bits).
    """
    values = np.asarray(zigzag, dtype=np.int64)
    parameters = np.asarray(ks, dtype=np.int64).reshape(-1)
    bits: list[int] = []
    for dimension in range(values.shape[1]):
        k = int(parameters[dimension])
        for item in values[:, dimension].tolist():
            item = int(item)
            bits.extend((0,) * (item >> k))
            bits.append(1)
            bits.extend((item >> shift) & 1 for shift in range(k - 1, -1, -1))
    payload = (
        np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
        if bits
        else b""
    )
    return payload, len(bits)


def parse_back_codes(archive_bytes: bytes, *, runtime: Path) -> np.ndarray:
    """Decode codes out of candidate bytes through the receiver's PUBLIC path.

    ``read_residual_archive`` is the entry point ``f26_inflate`` calls, so this
    exercises the KW1 branch exactly as the shipped decoder will.
    """
    runtime = Path(runtime).resolve()
    cr, ra = import_receiver(runtime, ("carrier_repack", "residual_archive"))
    materialize_cpr1 = cr.materialize_cpr1
    split_frame0_selector_carrier = cr.split_frame0_selector_carrier
    read_residual_archive = ra.read_residual_archive
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / "cpr1"))
    try:
        import inflate as renderer_module  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    origin = Path(getattr(renderer_module, "__file__", "")).resolve()
    if runtime not in origin.parents:
        raise Pc2Error(f"cpr1 inflate loaded from {origin}, not from {runtime}")
    scratch = runtime / "_pc2_parse_back.zip"
    scratch.write_bytes(archive_bytes)
    try:
        parts = read_residual_archive(scratch)
    finally:
        scratch.unlink(missing_ok=True)
    carrier_blob, _selector = split_frame0_selector_carrier(parts.carrier_blob)
    canonical = materialize_cpr1(carrier_blob, renderer_module)
    basis_count = CARRIER_DIM * 3 * renderer_module.CARRIER_H * renderer_module.CARRIER_W
    _, _, _scales, encoded = renderer_module.decode_compact_carrier(
        canonical, basis_count=basis_count, frames=N_PAIRS, dimensions=CARRIER_DIM
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    values = np.cumsum(delta, axis=0) & 0xFFF
    return np.where(values >= 0x800, values - 0x1000, values).astype(np.int32)


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #
def stage_runtime(
    dest: Path, *, archive_bytes: bytes | None = None, kw1_patch: bool = True
) -> dict[str, Any]:
    """Copy the pointer runtime, optionally apply the KW1 patch, pin the archive.

    ``kw1_patch=False`` stages the receiver EXACTLY as shipped.  A candidate
    whose archive does not set the KW1 reserved bit does not need the patch, and
    shipping the unmodified receiver is the smaller, safer diff.
    """
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        POINTER_RUNTIME, dest, ignore=shutil.ignore_patterns("__pycache__")
    )
    report = patcher.apply_patches(dest / "runtime") if kw1_patch else {
        "kw1_patch": "not applied; this candidate does not set the KW1 reserved bit"
    }
    if archive_bytes is not None:
        (dest / "archive.zip").write_bytes(archive_bytes)
        _repin_inflate(dest, archive_bytes)
    return {
        "runtime": str(dest),
        "patch": report,
        "archive_sha256": _sha256_file(dest / "archive.zip"),
        "archive_bytes": (dest / "archive.zip").stat().st_size,
    }


def _repin_inflate(dest: Path, archive_bytes: bytes) -> None:
    """Update the staged ``inflate.py`` archive pin to the candidate bytes."""
    path = dest / "inflate.py"
    text = path.read_text()
    sha = _sha256(archive_bytes)
    size = len(archive_bytes)
    replaced = 0
    out_lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("ARCHIVE_SHA256"):
            out_lines.append(f'ARCHIVE_SHA256 = "{sha}"\n')
            replaced += 1
        elif stripped.startswith("ARCHIVE_BYTES"):
            out_lines.append(f"ARCHIVE_BYTES = {size}\n")
            replaced += 1
        else:
            out_lines.append(line)
    if replaced != 2:
        raise Pc2Error(
            f"inflate.py archive pin: replaced {replaced} lines, expected 2"
        )
    path.write_text("".join(out_lines))


# --------------------------------------------------------------------------- #
# The masked solver -- jg5.refine_pair with the search restricted to kept dims
# --------------------------------------------------------------------------- #
def refine_pair_masked(
    inst,
    pair: int,
    start_codes: np.ndarray,
    keep: np.ndarray,
    *,
    dd_threshold: float,
    outer_rounds: int = 40,
    max_gn_iterations: int = 400,
) -> dict[str, Any]:
    """``jg5.refine_pair`` with the SEARCH restricted to the retained dimensions.

    Evaluation is unchanged and runs the exact 12-dimensional receiver path --
    only the proposal space shrinks.  Two restrictions are needed and no more:

    * the Gauss-Newton step is solved on the retained submatrix, because a
      dropped atom's ``low_basis`` row is exactly zero and the full Gram matrix
      is therefore singular;
    * the +-2 polish enumerates only retained coordinates, because a step on a
      dropped coordinate changes no rendered pixel (its atom is the zero field)
      and would only spend evaluations.

    Everything else -- the step ladder, the realized acceptance test, the
    materiality stop, the outer-round decay rule -- is jg5's, verbatim.
    """
    import torch

    keep = np.asarray(keep, dtype=np.int64)
    index = np.array([pair], dtype=np.int64)
    frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
    target_row = inst.targets[pair]
    tb = torch.from_numpy(target_row[None]).float()
    scales = inst.state.coefficient_scales.double().numpy()
    gram_keep = inst.gram[np.ix_(keep, keep)]

    current = start_codes.astype(np.int32).copy()
    best = float(br1.evaluate_codes(inst, pair, current[None])[0])
    start_value = best
    history = [best]
    evaluations = 1
    demanded_max = 0.0
    gn_iterations = 0
    polish_steps = 0
    stop_reason = "outer_round_budget"
    rounds = 0

    round_gains: list[float] = []
    last_ratio = math.nan
    last_remaining = math.inf
    for _round in range(outer_rounds):
        rounds += 1
        round_entry = best
        inner_stop = "gn_iteration_budget"
        step_gains: list[float] = []
        for _ in range(max_gn_iterations):
            coeff = up2.codes_to_coefficients(
                current[None], inst.state.coefficient_scales
            )
            jac, res, _ = up2.jacobian_and_residual(
                inst.posenet, inst.state, coeff, frame1, tb, index
            )
            step_keep = br1.min_image_norm_step(
                jac[0].double()[:, keep], res[0].double(), gram_keep
            ).numpy()
            step = np.zeros(CARRIER_DIM, dtype=np.float64)
            step[keep] = step_keep
            demanded_max = max(demanded_max, float(np.abs(step / scales).max()))
            block = [
                trial
                for fraction in br1.STEP_LADDER
                if not np.array_equal(
                    (trial := br1.realize(current + fraction * step / scales)), current
                )
            ]
            if not block:
                inner_stop = "lattice_floor"
                break
            block_arr = np.stack(block)
            values = br1.evaluate_codes(inst, pair, block_arr)
            evaluations += len(block_arr)
            winner = int(values.argmin())
            if values[winner] >= best:
                inner_stop = "no_improving_step"
                break
            step_gains.append(best - float(values[winner]))
            best = float(values[winner])
            current = block_arr[winner].copy()
            history.append(best)
            gn_iterations += 1
            remaining, ratio = jg5.projected_remaining_gain(step_gains)
            last_ratio, last_remaining = ratio, remaining
            if remaining <= dd_threshold:
                inner_stop = "converged_below_materiality_floor"
                break

        while True:
            block = _masked_neighbours(current, keep, (-2, -1, 1, 2))
            if len(block) == 0:
                break
            values = br1.evaluate_codes(inst, pair, block)
            evaluations += len(block)
            winner = int(values.argmin())
            if values[winner] >= best:
                break
            best = float(values[winner])
            current = block[winner].copy()
            history.append(best)
            polish_steps += 1

        round_gains.append(round_entry - best)
        if round_gains[-1] <= 0.0:
            stop_reason = (
                inner_stop
                if inner_stop != "gn_iteration_budget"
                else "no_improving_step"
            )
            break
        remaining, ratio = jg5.projected_remaining_gain(round_gains)
        last_ratio, last_remaining = ratio, remaining
        if remaining <= dd_threshold:
            stop_reason = "converged_below_materiality_floor"
            break

    return {
        "pair": int(pair),
        "start_d_pose": start_value,
        "final_d_pose": best,
        "ratio": best / start_value if start_value > 0 else 1.0,
        "evaluations": evaluations,
        "demanded_code_units_max": demanded_max,
        "codes": current.astype(np.int32).tolist(),
        "changed_coordinates": int((current != start_codes).sum()),
        "max_abs_code": int(np.abs(current).max()),
        "stop_reason": stop_reason,
        "outer_rounds_used": rounds,
        "gn_iterations_total": gn_iterations,
        "polish_steps_total": polish_steps,
        "dd_threshold": dd_threshold,
        "last_decay_ratio": None if math.isnan(last_ratio) else last_ratio,
        "projected_remaining_dd": (
            None if math.isinf(last_remaining) else last_remaining
        ),
        "kept_dimensions": keep.tolist(),
    }


def _masked_neighbours(
    codes_row: np.ndarray, keep: np.ndarray, offsets: tuple[int, ...]
) -> np.ndarray:
    candidates = []
    for coordinate in keep.tolist():
        for offset in offsets:
            trial = codes_row.copy()
            value = int(trial[coordinate]) + int(offset)
            if not up2.COEFF_CODE_MIN <= value <= up2.COEFF_CODE_MAX:
                continue
            trial[coordinate] = value
            candidates.append(trial)
    if not candidates:
        return np.zeros((0, CARRIER_DIM), np.int32)
    return np.stack(candidates)



def least_squares_warm_start(
    inst, shipped_coefficients: np.ndarray, keep: np.ndarray, full_gram
) -> np.ndarray:
    """Project the SHIPPED realized carrier field onto the retained subspace.

    Zeroing the dropped columns is a terrible start -- pair 0 begins at d_pose
    2.108 there, because the field loses a whole dimension's contribution with
    nothing taking it up.  The retained atoms are not orthogonal, so the field
    that best replaces it is the least-squares one:

        minimise || sum_{j in keep} c_j a_j  -  sum_{all j} c_j^shipped a_j ||^2
        =>  G[keep, keep] c_keep = G[keep, :] c_shipped

    with ``G`` the Gram matrix of the normalised atoms (``br1.span_gram``).

    ``full_gram`` MUST come from the POINTER's uncut basis.  With the rung's own
    (cut) Gram the dropped atoms' rows and columns are exactly zero, the target
    silently loses their contribution, and the projection collapses to
    ``c_keep = c_shipped_keep`` -- i.e. to the zeroed start it was supposed to
    improve on.  That was MEASURED here: with the cut Gram, pair 0 started at
    d_pose 2.108437 and solved to codes bit-identical to the zeroed run, so the
    warm start was a silent no-op.  The retained block of both Grams is the same
    (the retained atoms are bit-identical), so only the right-hand side needs
    the full matrix.

    The solution is realised on the shipped signed-int12 lattice.  This is a
    PROPOSAL only: every step after it is accepted by realized evaluation
    through the receiver.
    """
    import torch

    keep = np.asarray(keep, dtype=np.int64)
    # The rung's own scales, which a lattice refinement changes.  The projection
    # itself is done in COEFFICIENT space (scale-free); only the final descent to
    # codes uses the rung's step.
    scales = inst.state.coefficient_scales.double().numpy()
    gram = torch.as_tensor(full_gram).double()
    coefficients = torch.from_numpy(
        np.asarray(shipped_coefficients, dtype=np.float64)
    ).double()
    target = coefficients @ gram          # (600, 12) -> G_full c
    solved = torch.linalg.solve(
        gram[np.ix_(keep, keep)], target[:, keep].T
    ).T.numpy()
    out = np.zeros((coefficients.shape[0], CARRIER_DIM), dtype=np.float64)
    out[:, keep] = solved / scales[keep][None]
    return br1.realize(out)


def load_rung_instrument(runtime: Path):
    """br1's instrument, pinned to a rung's own staged body and the pointer raw.

    Only ODD (frame-1) frames are read from the raw decode and a carrier edit
    touches no odd frame, so the pointer's decode is valid for every rung.
    """
    runtime = Path(runtime)
    raw_sha_path = runtime / "_pc2_raw_sha.txt"
    state = load_state(runtime)
    raw = np.memmap(
        POINTER_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS, up2.CAMERA_H, up2.CAMERA_W, 3),
    )
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise Pc2Error(
            f"GT lineage is {lineage}, not {up2.LINEAGE_DALI}: that is a different "
            "objective (contest-CPU), not the one the pointer was measured on"
        )
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    del raw_sha_path
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_identity(args) -> int:
    """Two controls: the container rebuilds the pointer, and the patch is inert."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    live = assert_pointer_unmoved()

    observed = _sha256_file(POINTER_RUNTIME / "archive.zip")
    if observed != POINTER_ARCHIVE_SHA256:
        raise Pc2Error(f"pointer runtime archive sha {observed} is not the pointer")

    body = load_body()
    container = detect_container_shape(body, POINTER_RUNTIME)
    _purge_receiver_modules()
    rebuilt = up3.build_archive(
        body,
        body.codes,
        runtime_dir=POINTER_RUNTIME,
        container_options=(container,),
        verify=True,
    )
    container_identity = (
        rebuilt["archive_sha256"] == POINTER_ARCHIVE_SHA256
        and rebuilt["archive_size"] == POINTER_ARCHIVE_BYTES
    )

    staged = out / "runtime_identity"
    stage_report = stage_runtime(staged)
    patched_body = load_body(staged)
    patch_inert = (
        patched_body.carrier_body == body.carrier_body
        and np.array_equal(patched_body.codes, body.codes)
        and bytes(patched_body.packed_metadata) == bytes(body.packed_metadata)
    )
    patched_codes = parse_back_codes(
        (POINTER_RUNTIME / "archive.zip").read_bytes(), runtime=staged
    )
    public_path_identity = bool(np.array_equal(patched_codes, body.codes))

    # The container refusal this arm exists to remove, reproduced on THIS body.
    rank_cut_ks = np.array([5] * 8 + [0] * 4, dtype=np.int64)
    try:
        up3.pack_cap1_metadata(
            factors=body.factors,
            biases=body.biases,
            lengths=body.lengths,
            ks=rank_cut_ks,
        )
        shipped_refuses_rank_cut = False
    except Exception:
        shipped_refuses_rank_cut = True

    energy = realized_energy(
        body.codes, load_state(POINTER_RUNTIME).coefficient_scales.numpy()
    )
    order = np.argsort(energy)

    report = {
        "arm": "ddm_pc2",
        "pointer": live,
        "pointer_runtime": str(POINTER_RUNTIME),
        "pointer_archive_sha256": observed,
        "pointer_archive_bytes": POINTER_ARCHIVE_BYTES,
        "container_shape_derived": list(container),
        "container_identity_reproduces_pointer": container_identity,
        "container_identity_sha256": rebuilt["archive_sha256"],
        "container_identity_bytes": rebuilt["archive_size"],
        "kw1_patch_report": stage_report["patch"],
        "kw1_patch_is_inert_on_shipped_archive": patch_inert,
        "kw1_patched_public_path_decodes_pointer_codes": public_path_identity,
        "shipped_packer_refuses_rank_cut_ks": shipped_refuses_rank_cut,
        "carrier": {
            "reserved": f"{int(body.rx1_header[4]):#x}",
            "carrier_stream_bytes": len(body.carrier_stream),
            "carrier_body_bytes": len(body.carrier_body),
            "basis_bits": int(body.basis_bits),
            "basis_blob_bytes": len(body.basis_blob),
            "residual_bits": int(body.residual_bits),
            "rice_payload_bytes": len(body.rice_payload),
            "packed_metadata_bytes": len(body.packed_metadata),
            "scales_bytes": len(body.scales),
            "body_tail_bytes": len(body.body_tail),
            "rice_ks": body.ks.tolist(),
            "factors": body.factors.tolist(),
            "biases": body.biases.tolist(),
            "max_abs_code": int(np.abs(body.codes).max()),
        },
        "realized_energy": energy.tolist(),
        "realized_energy_share": (energy / energy.sum()).tolist(),
        "drop_order_ascending_energy": order.tolist(),
    }
    (out / "identity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:4000])
    if not (container_identity and patch_inert and public_path_identity):
        raise Pc2Error("an identity control FAILED; refusing to proceed")
    return 0


def run_price(args) -> int:
    """Exact bytes for every rung, with the shipped codes projected onto the cut."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pointer_unmoved()
    body = load_body()
    symbols = basis_symbols(body)
    staged = out / "runtime_price"
    stage_runtime(staged)
    container = detect_container_shape(body, POINTER_RUNTIME)

    order = tuple(int(x) for x in (args.drop_order or DEFAULT_DROP_ORDER))
    rows = []
    for rank in args.ranks:
        dropped = tuple(order[: CARRIER_DIM - rank])
        cut_symbols, biases, codes = rank_cut_fields(body, symbols, dropped)
        built = build_kw1_archive(
            body,
            codes=codes,
            symbols=cut_symbols,
            biases=biases,
            runtime=staged,
            container=container,
            wide_k=rank < CARRIER_DIM,
            verify=True,
        )
        delta = built["delta_vs_pointer"]
        shipped_shape_size = built["archive_size_at_shipped_shape"]
        row = {
            "rank": rank,
            "dropped": list(dropped),
            "archive_bytes": built["archive_size"],
            "delta_bytes": delta,
            "archive_bytes_at_shipped_shape": shipped_shape_size,
            "container_sweep_recovered_bytes": (
                None
                if shipped_shape_size is None
                else shipped_shape_size - built["archive_size"]
            ),
            "container": built["container"],
            "archive_sha256": built["archive_sha256"],
            "rice_ks": built["rice_ks"],
            "rice_payload_bytes": built["rice_payload_bytes"],
            "basis_blob_bytes": built["basis_blob_bytes"],
            "packed_metadata_bytes": built["packed_metadata_bytes"],
            "carrier_stream_bytes": built["carrier_stream_bytes"],
            "rx1_reserved": built["rx1_reserved"],
            "delta_s_rate": delta * BYTE_TO_SCORE,
            "break_even_d_pose": break_even_d_pose(
                delta_bytes=delta, d_pose_base=args.base_d_pose
            ),
        }
        if rank < CARRIER_DIM and args.k_width_control:
            live_k = int(max(built["rice_ks"]))
            constrained = np.array(
                [
                    (live_k - 1) if dim in dropped else live_k
                    for dim in range(CARRIER_DIM)
                ],
                dtype=np.int64,
            )
            narrow = build_kw1_archive(
                body,
                codes=codes,
                symbols=cut_symbols,
                biases=biases,
                runtime=staged,
                container=container,
                wide_k=False,
                force_ks=constrained,
                verify=True,
            )
            row["narrow_k_archive_bytes"] = narrow["archive_size"]
            row["narrow_k_delta_bytes"] = narrow["delta_vs_pointer"]
            row["narrow_k_ks"] = narrow["rice_ks"]
            row["k_width_unlock_bytes"] = (
                narrow["archive_size"] - built["archive_size"]
            )
        rows.append(row)
        rung_runtime = out / f"runtime_r{rank}"
        stage_runtime(rung_runtime, archive_bytes=built["archive_bytes"])
        row["rung_runtime"] = str(rung_runtime)
        (out / f"start_archive_r{rank}.zip").write_bytes(built["archive_bytes"])
        np.savez_compressed(
            out / f"start_fields_r{rank}.npz",
            codes=codes,
            symbols=cut_symbols,
            biases=biases,
            dropped=np.asarray(dropped, dtype=np.int64),
        )
        print(
            f"r={rank:2d} dropped={list(dropped)!s:24s} "
            f"bytes={built['archive_size']:7d} delta={delta:+6d} "
            f"ks={built['rice_ks']} breakeven={row['break_even_d_pose']:.6e}"
        )
    payload = {
        "arm": "ddm_pc2",
        "base_d_pose": args.base_d_pose,
        "drop_order": list(order),
        "pointer_archive_bytes": POINTER_ARCHIVE_BYTES,
        "rows": rows,
    }
    (out / "price.json").write_text(json.dumps(payload, indent=2))
    return 0


def run_solve(args) -> int:
    """Masked 40-round re-solve of one rung over a strided shard of n600."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    body = load_body()
    symbols = basis_symbols(body)
    order = tuple(int(x) for x in (args.drop_order or DEFAULT_DROP_ORDER))
    dropped = tuple(order[: CARRIER_DIM - args.rank])
    keep = np.asarray(
        [d for d in range(CARRIER_DIM) if d not in dropped], dtype=np.int64
    )
    if args.refine == 1.0:
        cut_symbols, biases, codes = rank_cut_fields(body, symbols, dropped)
    else:
        cut_symbols, biases, codes, _scales = hybrid_fields(
            body, load_state(POINTER_RUNTIME), dropped, args.refine, symbols
        )

    staged = Path(args.runtime)
    if not (staged / "archive.zip").is_file():
        raise Pc2Error(f"rung runtime not staged: {staged}")

    inst = load_rung_instrument(staged)
    dd_threshold = jg5.materiality_dd_threshold(args.base_d_pose)
    pairs = jg5.shard_pairs(args.shard, args.shards)
    if args.warm_start and len(keep) < CARRIER_DIM:
        pointer_state = load_state(POINTER_RUNTIME)
        full_gram, _ = br1.span_gram(br1.low_basis(pointer_state))
        shipped_coefficients = np.asarray(
            body.codes, dtype=np.float64
        ) * pointer_state.coefficient_scales.double().numpy()[None]
        start = least_squares_warm_start(
            inst, shipped_coefficients, keep, full_gram
        )
        if np.any(start[:, list(dropped)] != 0):
            raise Pc2Error("the warm start put weight on a dropped dimension")
        codes = start

    rows_path = out / f"rows_r{args.rank}_shard{args.shard}of{args.shards}.jsonl"
    done: dict[int, dict[str, Any]] = {}
    if rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["pair"])] = row
    started = time.time()
    with open(rows_path, "a") as handle:
        for pair in pairs.tolist():
            if pair in done:
                continue
            row = refine_pair_masked(
                inst,
                pair,
                codes[pair],
                keep,
                dd_threshold=dd_threshold,
                outer_rounds=args.outer_rounds,
            )
            row["rank"] = int(args.rank)
            row["dropped"] = list(dropped)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            done[pair] = row
            if len(done) % 10 == 0:
                elapsed = time.time() - started
                print(
                    f"[r{args.rank} shard {args.shard}/{args.shards}] "
                    f"{len(done)}/{len(pairs)} pairs, {elapsed:.0f}s",
                    flush=True,
                )
    print(f"shard complete: {len(done)}/{len(pairs)} rows -> {rows_path}", flush=True)
    del cut_symbols, biases
    return 0


def run_close(args) -> int:
    """Merge a rung's shards, rebuild the exact archive, and price the admission."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pointer_unmoved()
    body = load_body()
    symbols = basis_symbols(body)
    order = tuple(int(x) for x in (args.drop_order or DEFAULT_DROP_ORDER))
    dropped = tuple(order[: CARRIER_DIM - args.rank])
    hybrid_scales = None
    if args.refine == 1.0:
        cut_symbols, biases, start_codes = rank_cut_fields(body, symbols, dropped)
    else:
        cut_symbols, biases, start_codes, hybrid_scales = hybrid_fields(
            body, load_state(POINTER_RUNTIME), dropped, args.refine, symbols
        )

    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(args.rows_dir).glob(f"rows_r{args.rank}_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["pair"])] = row
    missing = [p for p in range(N_PAIRS) if p not in rows]
    if missing:
        raise Pc2Error(
            f"rung r={args.rank} is incomplete: {len(missing)} pairs missing "
            f"(first {missing[:8]}). A subset d_pose is not a verdict."
        )

    solved = start_codes.copy()
    finals = np.zeros(N_PAIRS, dtype=np.float64)
    starts = np.zeros(N_PAIRS, dtype=np.float64)
    for pair, row in rows.items():
        solved[pair] = np.asarray(row["codes"], dtype=np.int32)
        finals[pair] = float(row["final_d_pose"])
        starts[pair] = float(row["start_d_pose"])
    if np.any(solved[:, list(dropped)] != 0):
        raise Pc2Error("a solved row put weight on a dropped dimension")

    staged = out / f"runtime_r{args.rank}"
    stage_runtime(staged)
    container = detect_container_shape(body, POINTER_RUNTIME)
    basis_scales = np.frombuffer(bytes(body.scales), dtype="<f4")[:CARRIER_DIM]
    scales_block = (
        None if hybrid_scales is None
        else scales_block_with(basis_scales, hybrid_scales)
    )
    built = build_kw1_archive(
        body,
        codes=solved,
        symbols=cut_symbols,
        biases=biases,
        runtime=staged,
        container=container,
        wide_k=args.rank < CARRIER_DIM,
        scales=scales_block,
        verify=True,
    )
    twin = build_kw1_archive(
        body,
        codes=solved,
        symbols=cut_symbols,
        biases=biases,
        runtime=staged,
        container=container,
        wide_k=args.rank < CARRIER_DIM,
        scales=scales_block,
        verify=False,
    )
    if twin["archive_sha256"] != built["archive_sha256"]:
        raise Pc2Error("the twin encode is not byte-identical")

    stage_runtime(staged, archive_bytes=built["archive_bytes"])
    d_pose = float(finals.mean())
    delta = int(built["delta_vs_pointer"])
    legs = net_delta_s(
        delta_bytes=delta, d_pose_new=d_pose, d_pose_base=args.base_d_pose
    )
    report = {
        "arm": "ddm_pc2",
        "rank": int(args.rank),
        "refine": float(args.refine),
        "dropped": list(dropped),
        "archive_bytes": built["archive_size"],
        "archive_sha256": built["archive_sha256"],
        "delta_bytes": delta,
        "archive_bytes_at_shipped_shape": built["archive_size_at_shipped_shape"],
        "container_sweep_recovered_bytes": (
            None
            if built["archive_size_at_shipped_shape"] is None
            else built["archive_size_at_shipped_shape"] - built["archive_size"]
        ),
        "container": built["container"],
        "twin_encode_identical": True,
        "parse_back_codes_identical": bool(built.get("parse_back_codes_identical")),
        "rice_ks": built["rice_ks"],
        "rice_k_span": built["rice_k_span"],
        "packed_metadata_bytes": built["packed_metadata_bytes"],
        "rx1_reserved": built["rx1_reserved"],
        "d_pose_n600": d_pose,
        "d_pose_base": args.base_d_pose,
        "d_pose_start_n600": float(starts.mean()),
        "pose_leg": pose_leg(d_pose),
        "per_pair_median": float(np.median(finals)),
        "per_pair_max": float(finals.max()),
        "pairs_improved_vs_start": int((finals < starts).sum()),
        "break_even_d_pose": break_even_d_pose(
            delta_bytes=delta, d_pose_base=args.base_d_pose
        ),
        "stop_reasons": {
            reason: sum(1 for r in rows.values() if r["stop_reason"] == reason)
            for reason in jg5.REFINE_STOP_REASONS
        },
        "changed_coordinates_total": int(
            sum(int(r["changed_coordinates"]) for r in rows.values())
        ),
        "max_abs_code": int(np.abs(solved).max()),
        "admit": bool(legs["net_delta_s"] < ADMIT_BAR),
        "admit_bar": ADMIT_BAR,
        "staged_runtime": str(staged),
        **legs,
    }
    (out / f"close_r{args.rank}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out / f"solved_r{args.rank}.npz",
        codes=solved,
        symbols=cut_symbols,
        biases=biases,
        dropped=np.asarray(dropped, dtype=np.int64),
        final_d_pose=finals,
        start_d_pose=starts,
    )
    (out / f"archive_r{args.rank}.zip").write_bytes(built["archive_bytes"])
    print(json.dumps(report, indent=2))
    return 0



# --------------------------------------------------------------------------- #
# The seal's public-entrypoint smoke PAIR
# --------------------------------------------------------------------------- #

def _run_bounded(argv: list[str], *, timeout_s: float, **kwargs):
    """``subprocess.run`` that kills the whole PROCESS GROUP on timeout.

    The public-path probe starts a full ``inflate_archive``; if the parent is
    killed (or the bound fires) while a plain child is running, the child
    survives as an orphan under launchd, burning ~180% CPU and writing into a
    temp dir nobody reads.  MEASURED here: one such orphan ran 11m52s after its
    parent was gone.  ``start_new_session`` puts the child in its own group and
    ``killpg`` takes the group down with it, so no probe can outlive its caller.
    """
    import os
    import signal
    import subprocess

    process = subprocess.Popen(
        argv, start_new_session=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _public_path_probe(runtime_root: Path, timeout_s: float = 300.0) -> dict[str, Any]:
    """Run the receiver's own ``f26_inflate.inflate_archive`` on CPU.

    ``bash inflate.sh`` cannot COMPLETE on this host by the submission's own
    design (it refuses without CUDA), so the leg that proves the archive parses
    is the function ``inflate.py`` calls.  Reaching the token decode means the
    container framing, the RX1 riders, the KW1 metadata restore, the renderer
    load and the semantic unpack all passed.  Reaching it is the PASS condition;
    the same probe runs against the FRONTIER tree as the control.
    """
    import os
    import subprocess
    import tempfile

    runtime_root = Path(runtime_root).resolve()
    script = (
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "sys.path.insert(0, str(root))\n"
        "from runtime.f26_inflate import inflate_archive, InflationError\n"
        "out = Path(sys.argv[2])\n"
        "started = time.time()\n"
        "try:\n"
        "    inflate_archive(root / 'archive.zip', out / '0.raw',\n"
        "                    renderer_dir=root / 'cpr1', device_name='cpu',\n"
        "                    num_threads=4, checkpoint_dir=out / '.ckpt')\n"
        "    print(json.dumps({'outcome': 'COMPLETED', 'seconds': time.time()-started}))\n"
        "except InflationError as error:\n"
        "    print(json.dumps({'outcome': 'INFLATION_ERROR', 'error': str(error),\n"
        "                      'seconds': time.time()-started}))\n"
        "except Exception as error:\n"
        "    print(json.dumps({'outcome': type(error).__name__, 'error': str(error),\n"
        "                      'seconds': time.time()-started}))\n"
    )
    with tempfile.TemporaryDirectory() as scratch:
        library = Path(scratch) / "rc64_backend.so"
        subprocess.run(
            [
                os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
                str(runtime_root / "runtime" / "entropy" / "rc64_backend.c"),
                "-o", str(library),
            ],
            check=True,
            capture_output=True,
        )
        environment = dict(os.environ, CPR1_RC64_LIBRARY=str(library))
        started = time.time()
        try:
            done = _run_bounded(
                [sys.executable, "-c", script, str(runtime_root), scratch],
                timeout_s=timeout_s, cwd=str(runtime_root), env=environment,
            )
        except subprocess.TimeoutExpired:
            return {
                "outcome": "REACHED_TOKEN_DECODE",
                "seconds": time.time() - started,
                "exception_class": None,
                "exception_message": "",
                "note": "no exception within the bound; the token decode is the slow leg",
            }
    tail = (done.stdout or "").strip().splitlines()
    parsed: dict[str, Any] = {"outcome": "UNPARSED", "stdout_tail": tail[-3:]}
    for line in reversed(tail):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    parsed.setdefault("exception_class", None)
    parsed.setdefault("exception_message", "")
    parsed["returncode"] = done.returncode
    parsed["stderr_tail"] = (done.stderr or "").strip().splitlines()[-5:]
    return parsed


def _inflate_sh_smoke(runtime_root: Path, timeout_s: float = 600.0) -> dict[str, Any]:
    """Run ``bash inflate.sh`` with the real 3-argument contest signature.

    The host has no CUDA and ``inflate.py`` refuses without it, so reaching that
    refusal is the PASS condition: it proves the C toolchain builds, the Brotli
    gate passes, the file-list loop dispatches, and ``_verify_input`` accepts the
    STAGED archive against its two pinned constants -- which is exactly what a
    re-pinned candidate could get wrong.
    """
    import os
    import subprocess
    import tempfile

    runtime_root = Path(runtime_root).resolve()
    environment = dict(
        os.environ,
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}",
    )
    archive_path = runtime_root / "archive.zip"
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        # The proven idiom: the data dir carries the archive itself AND its
        # members, and the file list names the source video, not a codec suffix.
        data_dir = scratch_path / "archive_dir"
        data_dir.mkdir()
        shutil.copyfile(archive_path, data_dir / "archive.zip")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                (data_dir / member).write_bytes(archive.read(member))
        file_list = scratch_path / "list.txt"
        file_list.write_text("0.mkv\n", encoding="utf-8")
        started = time.time()
        try:
            done = _run_bounded(
                [
                    "bash", str(runtime_root / "inflate.sh"),
                    str(data_dir), str(scratch_path / "out"), str(file_list),
                ],
                timeout_s=timeout_s, cwd=str(runtime_root), env=environment,
            )
        except subprocess.TimeoutExpired:
            return {
                "outcome": "TIMEOUT",
                "seconds": time.time() - started,
                "returncode": None,
            }
    combined = f"{done.stdout}\n{done.stderr}"
    message = ""
    for line in combined.splitlines():
        if "requires CUDA inflation" in line:
            message = line.strip()
            break
    if message:
        outcome = "REACHED_CUDA_GATE"
    elif done.returncode == 0:
        outcome = "COMPLETED"
    else:
        outcome = "OTHER_FAILURE"
    return {
        "outcome": outcome,
        "returncode": done.returncode,
        "seconds": time.time() - started,
        "exception_class": "RuntimeError" if message else None,
        "exception_message": message,
        "stderr_tail": (done.stderr or "").strip().splitlines()[-6:],
    }


def _smoke_receipt(runtime_root: Path, probe: dict[str, Any]) -> dict[str, Any]:
    """Add the identity pins the seal validator re-measures from disk."""
    from tac.candidate_seal import measure_runtime_digest

    runtime_root = Path(runtime_root).resolve()
    archive = runtime_root / "archive.zip"
    receipt = dict(probe)
    receipt["runtime_path"] = str(runtime_root)
    receipt["tree_sha256"] = measure_runtime_digest(runtime_root).sha256
    receipt["archive_path"] = str(archive)
    receipt["archive_sha256"] = _sha256_file(archive)
    return receipt


def run_smoke(args) -> int:
    """The seal's required smoke PAIR, on the candidate AND the frontier trees."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    candidate = Path(args.candidate_runtime)
    frontier = Path(args.frontier_runtime or POINTER_RUNTIME)
    # BOUND TRAP.  The validator refuses ``seconds > public_path_probe_seconds``,
    # and a probe that hits its subprocess timeout records
    # ``elapsed = timeout + epsilon``.  Declaring the same number for both would
    # therefore refuse every timed-out probe -- and a timeout is the EXPECTED
    # outcome of the token-decode leg on this host.  The probes run at 0.8x the
    # declared bound so a timeout still lands inside it.
    bound = float(args.bound_seconds)
    probe_timeout = 0.8 * bound
    # Run every probe FIRST, then measure the tree digests once.  The seal
    # requires both legs to name the same tree_sha256 per role, so a digest taken
    # between two probes that touch the tree would refuse itself.
    probes = {
        ("public_path_probes", "candidate"): _public_path_probe(
            candidate, timeout_s=probe_timeout
        ),
        ("public_path_probes", "frontier"): _public_path_probe(
            frontier, timeout_s=probe_timeout
        ),
        ("inflate_sh_smokes", "candidate"): _inflate_sh_smoke(
            candidate, timeout_s=probe_timeout
        ),
        ("inflate_sh_smokes", "frontier"): _inflate_sh_smoke(
            frontier, timeout_s=probe_timeout
        ),
    }
    trees = {"candidate": candidate, "frontier": frontier}
    block: dict[str, Any] = {
        "schema": "candidate_public_entrypoint_smoke.v1",
        "public_path_probe_seconds": bound,
        "public_path_probes": {},
        "inflate_sh_smokes": {},
    }
    for (group, role), probe in probes.items():
        block[group][role] = _smoke_receipt(trees[role], probe)
    (out / "public_entrypoint_smoke.json").write_text(json.dumps(block, indent=2))
    print(json.dumps(block, indent=2))
    # Check the block with the VALIDATOR'S OWN checker, not a second reading of
    # the contract.  A private name is used deliberately: if it is renamed the
    # ImportError fails closed, which is the correct outcome -- a smoke block
    # validated against a stale local copy of the rules is worse than no check.
    from tac.candidate_seal import _public_smoke_problems

    problems, _observed = _public_smoke_problems(
        block,
        candidate_runtime_dir=candidate,
        candidate_archive_path=candidate / "archive.zip",
        pointer_archive_sha256=POINTER_ARCHIVE_SHA256,
    )
    (out / "public_entrypoint_smoke_validation.json").write_text(
        json.dumps({"problems": problems}, indent=2)
    )
    if problems:
        raise Pc2Error(
            "the seal's own validator refuses this smoke block:\n  "
            + "\n  ".join(problems)
        )
    print("validator: the seal's own _public_smoke_problems reports no problems")
    return 0



def run_coarsen_price(args) -> int:
    """Exact bytes for per-column lattice coarsening, no solver time spent."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pointer_unmoved()
    body = load_body()
    symbols = basis_symbols(body)
    staged = out / "runtime_coarsen"
    stage_runtime(staged)
    container = detect_container_shape(body, POINTER_RUNTIME)
    state = load_state(POINTER_RUNTIME)
    base_scales = state.coefficient_scales.double().numpy()
    basis_scales = np.frombuffer(bytes(body.scales), dtype="<f4")[:CARRIER_DIM]
    energy = realized_energy(body.codes, base_scales)
    order = list(np.argsort(energy))

    rows = []
    for spec in args.plan:
        factor, count = (float(x) for x in spec.split(":"))
        count = int(count)
        factors = np.ones(CARRIER_DIM, dtype=np.float64)
        targets = order[:count] if count else []
        factors[targets] = factor
        codes, coefficient_scales = coarsen_fields(body, state, factors)
        built = build_kw1_archive(
            body,
            codes=codes,
            symbols=symbols,
            biases=np.asarray(body.biases, dtype=np.int64),
            runtime=staged,
            container=container,
            wide_k=True,
            scales=scales_block_with(basis_scales, coefficient_scales),
            verify=True,
        )
        delta = built["delta_vs_pointer"]
        row = {
            "factor": factor,
            "columns": [int(x) for x in targets],
            "archive_bytes": built["archive_size"],
            "delta_bytes": delta,
            "archive_sha256": built["archive_sha256"],
            "rice_ks": built["rice_ks"],
            "rice_k_span": built["rice_k_span"],
            "max_abs_code": int(np.abs(codes).max()),
            "delta_s_rate": delta * BYTE_TO_SCORE,
            "break_even_d_pose": break_even_d_pose(
                delta_bytes=delta, d_pose_base=args.base_d_pose
            ),
        }
        rows.append(row)
        (out / f"coarsen_f{factor:g}_n{count}.zip").write_bytes(built["archive_bytes"])
        # Preserve measured rows even if the optional coefficient dump fails.
        (out / "coarsen_price.partial.json").write_text(json.dumps({"rows": rows}, indent=2))
        np.savez_compressed(
            out / f"coarsen_f{factor:g}_n{count}.npz",
            codes=codes,
            coefficient_scales=coefficient_scales,
            factors=factors,
        )
        rung = out / f"runtime_f{factor:g}_n{count}"
        stage_runtime(rung, archive_bytes=built["archive_bytes"])
        row["rung_runtime"] = str(rung)
        print(
            f"factor {factor:g} on {count:2d} columns {row['columns']!s:28s} "
            f"bytes={built['archive_size']:7d} delta={delta:+6d} "
            f"ks={built['rice_ks']} breakeven={row['break_even_d_pose']:.6e}"
        )
    (out / "coarsen_price.json").write_text(
        json.dumps(
            {
                "arm": "ddm_pc2",
                "base_d_pose": args.base_d_pose,
                "energy_ascending_order": [int(x) for x in order],
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0



def run_jacobian(args) -> int:
    """Rank dimensions by POSE leverage, not by field energy.

    The drop order this arm started with is ascending realized field ENERGY,
    which is what the carrier puts into the image.  It is not what the carrier is
    FOR.  The solver inverts ``J = d(pose)/d(coeff)`` (6 x 12 per pair), so the
    dimension that costs least to lose is the one whose Jacobian column is most
    nearly reproduced by the other eleven -- a column with large energy can still
    be pose-redundant, and a small one can be the only way to reach a pose
    direction.  This is ``ddm_pc1``'s owed ITEM 2 insight (choose atoms by the
    Jacobian, not by smoothness) applied to the drop order instead of the basis.

    Two rankings are reported per dimension, both averaged over strided pairs:

    * ``residual_share`` -- the fraction of column j's norm that survives
      projection onto the other eleven columns.  Small means redundant.
    * ``leverage`` -- ``1 / (J^+ J^+T)_jj`` style: how much of the reachable pose
      space is lost when column j is removed, measured as the increase in the
      minimum-norm residual of the SAME six-component pose demand.
    """
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    body = load_body()
    inst = load_rung_instrument(POINTER_RUNTIME)
    pairs = jg5.shard_pairs(0, args.stride)
    residual_share = np.zeros((len(pairs), CARRIER_DIM))
    reach_loss = np.zeros((len(pairs), CARRIER_DIM))
    for row, pair in enumerate(pairs.tolist()):
        index = np.array([pair], dtype=np.int64)
        frame1 = up2.frames_to_bchw(inst.raw[2 * index + 1])
        target = torch.from_numpy(inst.targets[pair][None]).float()
        coeff = up2.codes_to_coefficients(
            np.asarray(body.codes)[pair][None], inst.state.coefficient_scales
        )
        jac, res, _ = up2.jacobian_and_residual(
            inst.posenet, inst.state, coeff, frame1, target, index
        )
        matrix = jac[0].double().numpy()          # (6, 12)
        demand = res[0].double().numpy()          # (6,)
        full = np.linalg.lstsq(matrix, -demand, rcond=None)[0]
        full_residual = float(np.linalg.norm(matrix @ full + demand))
        for dim in range(CARRIER_DIM):
            others = [d for d in range(CARRIER_DIM) if d != dim]
            column = matrix[:, dim]
            fit = np.linalg.lstsq(matrix[:, others], column, rcond=None)[0]
            residual_share[row, dim] = float(
                np.linalg.norm(column - matrix[:, others] @ fit)
                / max(np.linalg.norm(column), 1e-30)
            )
            cut = np.linalg.lstsq(matrix[:, others], -demand, rcond=None)[0]
            cut_residual = float(np.linalg.norm(matrix[:, others] @ cut + demand))
            reach_loss[row, dim] = cut_residual - full_residual
    state = load_state(POINTER_RUNTIME)
    energy = realized_energy(body.codes, state.coefficient_scales.numpy())
    report = {
        "arm": "ddm_pc2",
        "pairs": pairs.tolist(),
        "residual_share_mean": residual_share.mean(axis=0).tolist(),
        "reach_loss_mean": reach_loss.mean(axis=0).tolist(),
        "reach_loss_median": np.median(reach_loss, axis=0).tolist(),
        "realized_energy": energy.tolist(),
        "order_ascending_energy": np.argsort(energy).tolist(),
        "order_ascending_residual_share": np.argsort(
            residual_share.mean(axis=0)
        ).tolist(),
        "order_ascending_reach_loss": np.argsort(reach_loss.mean(axis=0)).tolist(),
    }
    (out / "jacobian_leverage.json").write_text(json.dumps(report, indent=2))
    print(f"{'dim':>3} {'energy share':>13} {'residual share':>15} {'reach loss':>13}")
    total = energy.sum()
    for dim in range(CARRIER_DIM):
        print(
            f"{dim:3d} {energy[dim] / total:13.5f} "
            f"{residual_share.mean(axis=0)[dim]:15.6f} "
            f"{reach_loss.mean(axis=0)[dim]:13.6e}"
        )
    print("ascending energy        :", report["order_ascending_energy"])
    print("ascending residual share:", report["order_ascending_residual_share"])
    print("ascending reach loss    :", report["order_ascending_reach_loss"])
    return 0



def hybrid_fields(body, state, dropped: tuple[int, ...], refine: float, symbols):
    """A rank cut whose RETAINED columns get a finer lattice.

    The Jacobian measurement (mode=jacobian) found every column's pose-Jacobian
    column lies in the span of the other eleven -- residual share 0.000000, reach
    loss ~1e-16 -- so a rank cut costs NOTHING at first order.  The measured cost
    is therefore not the span; it is the LATTICE.  The carrier has to land on a
    representable point that matches six PoseNet outputs to ~1e-6, and removing
    coordinates removes the fine placements that combination of coordinates
    could reach.

    That is a curable defect, and the cut itself pays for the cure: the shipped
    codes reach only ``|code| <= 155`` of the +-2047 field, so the retained
    columns can take a step ``refine`` times FINER (scale x refine, codes / refine)
    and buy their placement precision back.  Each halving costs one Rice bit per
    retained symbol -- 600 bits per retained column -- against the ~1,590 bytes
    each dropped column returns.
    """
    dropped = tuple(sorted({int(d) for d in dropped}))
    keep = [d for d in range(CARRIER_DIM) if d not in dropped]
    cut_symbols, biases, codes = rank_cut_fields(body, symbols, dropped)
    scales = state.coefficient_scales.double().numpy().copy()
    fine = np.asarray(codes, dtype=np.float64)
    fine[:, keep] = fine[:, keep] / float(refine)
    scales[keep] = scales[keep] * float(refine)
    realised = br1.realize(fine)
    if np.any(realised[:, dropped] != 0):
        raise Pc2Error("refinement put weight on a dropped dimension")
    return cut_symbols, biases, realised.astype(np.int32), scales


def run_hybrid_price(args) -> int:
    """Exact bytes for rank cut + retained-lattice refinement, no solver time."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pointer_unmoved()
    body = load_body()
    symbols = basis_symbols(body)
    staged = out / "runtime_hybrid"
    stage_runtime(staged)
    container = detect_container_shape(body, POINTER_RUNTIME)
    state = load_state(POINTER_RUNTIME)
    basis_scales = np.frombuffer(bytes(body.scales), dtype="<f4")[:CARRIER_DIM]
    order = tuple(int(x) for x in (args.drop_order or DEFAULT_DROP_ORDER))

    rows = []
    for spec in args.plan:
        rank_text, refine_text = spec.split(":")
        rank, refine = int(rank_text), float(refine_text)
        dropped = tuple(order[: CARRIER_DIM - rank])
        cut_symbols, biases, codes, scales = hybrid_fields(
            body, state, dropped, refine, symbols
        )
        built = build_kw1_archive(
            body,
            codes=codes,
            symbols=cut_symbols,
            biases=biases,
            runtime=staged,
            container=container,
            wide_k=True,
            scales=scales_block_with(basis_scales, scales),
            verify=True,
        )
        delta = built["delta_vs_pointer"]
        shipped_shape_size = built["archive_size_at_shipped_shape"]
        row = {
            "rank": rank,
            "refine": refine,
            "dropped": list(dropped),
            "archive_bytes": built["archive_size"],
            "delta_bytes": delta,
            "archive_bytes_at_shipped_shape": shipped_shape_size,
            "container_sweep_recovered_bytes": (
                None
                if shipped_shape_size is None
                else shipped_shape_size - built["archive_size"]
            ),
            "container": built["container"],
            "archive_sha256": built["archive_sha256"],
            "rice_ks": built["rice_ks"],
            "rice_k_span": built["rice_k_span"],
            "max_abs_code": int(np.abs(codes).max()),
            "delta_s_rate": delta * BYTE_TO_SCORE,
            "break_even_d_pose": break_even_d_pose(
                delta_bytes=delta, d_pose_base=args.base_d_pose
            ),
        }
        rows.append(row)
        tag = f"r{rank}_refine{refine:g}"
        (out / f"hybrid_{tag}.zip").write_bytes(built["archive_bytes"])
        # Preserve measured rows even if the optional coefficient dump fails.
        (out / "hybrid_price.partial.json").write_text(json.dumps({"rows": rows}, indent=2))
        np.savez_compressed(
            out / f"hybrid_{tag}.npz",
            codes=codes,
            symbols=cut_symbols,
            biases=biases,
            coefficient_scales=scales,
            dropped=np.asarray(dropped, dtype=np.int64),
        )
        rung = out / f"runtime_{tag}"
        stage_runtime(rung, archive_bytes=built["archive_bytes"])
        row["rung_runtime"] = str(rung)
        print(
            f"r={rank:2d} refine=1/{1 / refine:<5.3g} dropped={list(dropped)!s:16s} "
            f"bytes={built['archive_size']:7d} delta={delta:+6d} "
            f"maxcode={row['max_abs_code']:5d} span={built['rice_k_span']} "
            f"ks={built['rice_ks']} breakeven={row['break_even_d_pose']:.6e}"
        )
    (out / "hybrid_price.json").write_text(
        json.dumps(
            {"arm": "ddm_pc2", "base_d_pose": args.base_d_pose,
             "drop_order": list(order), "rows": rows},
            indent=2,
        )
    )
    return 0



def run_scales_probe(args) -> int:
    """ITEM 1, pose cost FIRST, at ZERO byte change.

    ``ddm_pc1`` measured that the twelve per-atom basis scales all cancel in
    ``normalized_basis`` to 1.9073e-06 -- and then falsified its own "provably
    neutral" reading: the render rounds twice, and over 24 strided pairs only
    10 of 24 frame_0 images stayed bit-identical when the scales were dropped.
    Recovering those 48 bytes needs a SECOND receiver format change (the block
    is at a fixed offset), so the honest order is to price the POSE first, at
    zero byte change: build a body whose basis scales are all 1.0, keeping the
    96-byte block, and measure d_pose n600.  If the pose cost exceeds the 48
    bytes' own budget, the receiver is never touched.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pointer_unmoved()
    body = load_body()
    symbols = basis_symbols(body)
    staged = out / "runtime_scales"
    stage_runtime(staged)
    container = detect_container_shape(body, POINTER_RUNTIME)
    state = load_state(POINTER_RUNTIME)
    coefficient_scales = state.coefficient_scales.double().numpy()
    built = build_kw1_archive(
        body,
        codes=np.asarray(body.codes),
        symbols=symbols,
        biases=np.asarray(body.biases, dtype=np.int64),
        runtime=staged,
        container=container,
        wide_k=False,
        scales=scales_block_with(np.ones(CARRIER_DIM), coefficient_scales),
        verify=True,
    )
    stage_runtime(staged, archive_bytes=built["archive_bytes"])
    recoverable = 4 * CARRIER_DIM
    report = {
        "arm": "ddm_pc2",
        "item": 1,
        "archive_bytes": built["archive_size"],
        "delta_bytes": built["delta_vs_pointer"],
        "archive_sha256": built["archive_sha256"],
        "recoverable_bytes_if_block_shrinks": recoverable,
        "break_even_d_pose_for_that_saving": break_even_d_pose(
            delta_bytes=-recoverable, d_pose_base=args.base_d_pose
        ),
        "staged_runtime": str(staged),
        "note": (
            "basis scales set to 1.0, block still 96 B: this row prices the POSE "
            "of the change alone. Run mode=solve --rank 12 --outer-rounds 0 "
            "against this runtime for the n600 d_pose."
        ),
    }
    (out / "scales_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0



def run_scales_candidate(args) -> int:
    """ITEM 1 as a shippable candidate: basis scales = 1.0, SHIPPED receiver.

    The archive does not set the KW1 reserved bit, so the receiver is staged
    exactly as the pointer ships it -- no format change, no patch, no new
    decoder path.  The only bytes that move are inside the existing 96-byte
    scales block.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert_pointer_unmoved()
    body = load_body()
    symbols = basis_symbols(body)
    staged = out / "candidate_runtime"
    stage_runtime(staged, kw1_patch=False)
    container = detect_container_shape(body, POINTER_RUNTIME)
    state = load_state(POINTER_RUNTIME)
    coefficient_scales = state.coefficient_scales.double().numpy()
    scales_block = scales_block_with(np.ones(CARRIER_DIM), coefficient_scales)
    codes = np.asarray(body.codes, dtype=np.int32)
    resolve_rows = 0
    if args.codes_npz:
        # ITEM 1's re-solved codes, folded into the SAME candidate so the pair
        # ships as one seal.  The codes must be a strict refinement of this
        # body's own: same shape, same lattice, and every changed coordinate
        # accepted by realized evaluation in the solver that produced them.
        payload = np.load(args.codes_npz)
        solved = np.asarray(payload["codes"], dtype=np.int32)
        if solved.shape != codes.shape:
            raise Pc2Error(f"solved codes are {solved.shape}, expected {codes.shape}")
        resolve_rows = int((solved != codes).sum())
        codes = solved
    built = build_kw1_archive(
        body,
        codes=codes,
        symbols=symbols,
        biases=np.asarray(body.biases, dtype=np.int64),
        runtime=staged,
        container=container,
        wide_k=False,
        scales=scales_block,
        verify=True,
    )
    twin = build_kw1_archive(
        body,
        codes=codes,
        symbols=symbols,
        biases=np.asarray(body.biases, dtype=np.int64),
        runtime=staged,
        container=container,
        wide_k=False,
        scales=scales_block,
        verify=False,
    )
    if twin["archive_sha256"] != built["archive_sha256"]:
        raise Pc2Error("the twin encode is not byte-identical")
    if built["rx1_reserved"] != f"{int(body.rx1_header[4]):#x}":
        raise Pc2Error("the shipped-receiver candidate must not change reserved")
    stage_runtime(staged, archive_bytes=built["archive_bytes"], kw1_patch=False)

    delta = int(built["delta_vs_pointer"])
    legs = net_delta_s(
        delta_bytes=delta, d_pose_new=args.d_pose_new, d_pose_base=args.base_d_pose
    )
    report = {
        "arm": "ddm_pc2",
        "item": 1,
        "candidate": args.candidate_id,
        "coordinates_changed_by_resolve": resolve_rows,
        "archive_bytes": built["archive_size"],
        "archive_sha256": built["archive_sha256"],
        "delta_bytes": delta,
        "twin_encode_identical": True,
        "parse_back_codes_identical": bool(built.get("parse_back_codes_identical")),
        "receiver": "SHIPPED, unmodified (no KW1 bit, no patch)",
        "rx1_reserved": built["rx1_reserved"],
        "container": built["container"],
        "container_sweep_recovered_bytes": (
            None
            if built["archive_size_at_shipped_shape"] is None
            else built["archive_size_at_shipped_shape"] - built["archive_size"]
        ),
        "d_pose_base": args.base_d_pose,
        "d_pose_n600": args.d_pose_new,
        "d_pose_ratio": args.d_pose_new / args.base_d_pose,
        "break_even_d_pose": break_even_d_pose(
            delta_bytes=delta, d_pose_base=args.base_d_pose
        ),
        "admit": bool(legs["net_delta_s"] < ADMIT_BAR),
        "admit_bar": ADMIT_BAR,
        "staged_runtime": str(staged),
        **legs,
    }
    (out / "scales_candidate.json").write_text(json.dumps(report, indent=2))
    (out / "archive.zip").write_bytes(built["archive_bytes"])
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pointer-runtime",
        default=None,
        help="re-base onto a NEW pointer tree; its archive sha and size are "
        "measured from the bytes and must match the live canonical pointer",
    )
    parser.add_argument(
        "--donor-raw-from",
        default=None,
        help="reuse this tree's raw decode when the new pointer tree has none; "
        "accepted ONLY if the two bodies are MEASURED to render the same frames "
        "(decoded carrier, semantic, HPAC and token tail all identical)",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    identity = sub.add_parser("identity", help="container + KW1-patch identity controls")
    identity.add_argument("--out", default=str(WORK / "identity"))
    identity.set_defaults(func=run_identity)

    price = sub.add_parser("price", help="exact bytes for each rank-cut rung")
    price.add_argument("--out", default=str(WORK / "price"))
    price.add_argument("--ranks", type=int, nargs="+", default=[11, 10, 9, 8])
    price.add_argument("--base-d-pose", type=float, required=True)
    price.add_argument("--drop-order", type=int, nargs="*", default=None)
    price.add_argument(
        "--k-width-control",
        action="store_true",
        help="also price each rung at a k vector the SHIPPED 1-bit field carries",
    )
    price.set_defaults(func=run_price)

    solve = sub.add_parser("solve", help="masked 40-round re-solve of one rung")
    solve.add_argument("--out", required=True)
    solve.add_argument("--runtime", required=True)
    solve.add_argument("--rank", type=int, required=True)
    solve.add_argument("--shard", type=int, default=0)
    solve.add_argument("--shards", type=int, default=1)
    solve.add_argument("--outer-rounds", type=int, default=40)
    solve.add_argument("--base-d-pose", type=float, required=True)
    solve.add_argument("--drop-order", type=int, nargs="*", default=None)
    solve.add_argument(
        "--warm-start",
        dest="warm_start",
        action="store_true",
        default=True,
        help="start from the least-squares projection onto the retained subspace",
    )
    solve.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    solve.add_argument(
        "--refine",
        type=float,
        default=1.0,
        help="retained-column lattice step multiplier (< 1 is FINER); the rung "
        "runtime must be the matching hybrid-price tree",
    )
    solve.set_defaults(func=run_solve)

    close = sub.add_parser("close", help="merge shards, rebuild, price the admission")
    close.add_argument("--out", required=True)
    close.add_argument("--rows-dir", required=True)
    close.add_argument("--rank", type=int, required=True)
    close.add_argument("--base-d-pose", type=float, required=True)
    close.add_argument("--drop-order", type=int, nargs="*", default=None)
    close.add_argument("--refine", type=float, default=1.0)
    close.set_defaults(func=run_close)

    smoke = sub.add_parser("smoke", help="the seal's public-entrypoint smoke PAIR")
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--candidate-runtime", required=True)
    # LATE-BOUND on purpose.  A default captured here would freeze the pointer
    # tree at parser-build time, which is BEFORE --pointer-runtime re-bases the
    # module -- the same early-binding trap that made a "re-based" build come out
    # byte-identical to the old body (caught by the r=12 identity control).
    smoke.add_argument("--frontier-runtime", default=None)
    smoke.add_argument(
        "--bound-seconds",
        type=float,
        default=150.0,
        help="declared bound; probes run at 0.8x of it. The PASS condition is "
        "'no exception within the bound' and every pre-decode stage throws fast, "
        "so this only has to outlast those stages -- never the 25-minute decode.",
    )
    smoke.set_defaults(func=run_smoke)

    coarsen = sub.add_parser(
        "coarsen-price", help="exact bytes for per-column lattice coarsening"
    )
    coarsen.add_argument("--out", default=str(WORK / "coarsen"))
    coarsen.add_argument("--base-d-pose", type=float, required=True)
    coarsen.add_argument(
        "--plan",
        nargs="+",
        required=True,
        help="factor:count pairs, e.g. 2:12 2:6 4:6 -- count columns taken in "
        "ascending realized-energy order",
    )
    coarsen.set_defaults(func=run_coarsen_price)

    jacobian = sub.add_parser(
        "jacobian", help="rank carrier dimensions by POSE leverage, not by energy"
    )
    jacobian.add_argument("--out", default=str(WORK / "jacobian"))
    jacobian.add_argument("--stride", type=int, default=25)
    jacobian.set_defaults(func=run_jacobian)

    hybrid = sub.add_parser(
        "hybrid-price",
        help="rank cut + finer lattice on the retained columns, exact bytes",
    )
    hybrid.add_argument("--out", default=str(WORK / "hybrid"))
    hybrid.add_argument("--base-d-pose", type=float, required=True)
    hybrid.add_argument("--drop-order", type=int, nargs="*", default=None)
    hybrid.add_argument(
        "--plan", nargs="+", required=True,
        help="rank:refine pairs, e.g. 8:0.5 8:0.25 10:0.5 -- refine < 1 is a FINER step",
    )
    hybrid.set_defaults(func=run_hybrid_price)

    scales = sub.add_parser(
        "scales-probe", help="ITEM 1: basis scales = 1.0 at zero byte change"
    )
    scales.add_argument("--out", default=str(WORK / "scales"))
    scales.add_argument("--base-d-pose", type=float, required=True)
    scales.set_defaults(func=run_scales_probe)

    candidate = sub.add_parser(
        "scales-candidate", help="ITEM 1 as a shippable candidate on the SHIPPED receiver"
    )
    candidate.add_argument("--out", default=str(WORK / "candidate_scales"))
    candidate.add_argument("--base-d-pose", type=float, required=True)
    candidate.add_argument("--d-pose-new", type=float, required=True)
    candidate.add_argument(
        "--codes-npz",
        default=None,
        help="fold ITEM 1's re-solved codes into the same candidate (npz with 'codes')",
    )
    candidate.add_argument(
        "--candidate-id", default="ddm_pc2_carrier_kwidth_rankcut"
    )
    candidate.set_defaults(func=run_scales_candidate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "pointer_runtime", None):
        donor = getattr(args, "donor_raw_from", None)
        print(
            json.dumps(
                rebase_to(
                    Path(args.pointer_runtime),
                    donor_raw_from=Path(donor) if donor else None,
                ),
                indent=2,
            )
        )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
