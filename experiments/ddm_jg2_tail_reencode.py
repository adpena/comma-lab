#!/usr/bin/env python3
"""ddm_jg2 -- REAL archive bytes for an edited token field, on the SHIPPED body.

WHY THIS EXISTS.  ``ddm_jg1`` priced a seg-repair token edit at **+4.718 bits per
changed token** and named that number's three weaknesses itself (its memo, S1d
caveats 3-5): the logits came from a DIFFERENT body (hm1, 182,759 B) whose model
is BLUNTER than the one we ship, the per-symbol delta ignores the decoder's
context cascade, and the table correction was omitted.  All three push the same
way -- the real price is probably HIGHER.  The whole -0.0104 S projection rests
on that one modelled constant.  This module deletes the model and measures.

WHY A LOCAL PRICE CANNOT BE CORRECT HERE.  Read the shipped decoder
(``runtime/residual_archive.py::decode_production_tokens``, :600-649).  Three
separate feedback paths make the cost of a token depend on OTHER tokens:

  * ``sparse.selected_logits(current, context, group)`` -- ``current`` holds the
    tokens decoded by EARLIER GROUPS OF THE SAME FRAME, so the model re-evaluates
    with them known.  Intra-frame cascade, across 190 groups.
  * ``context = model.prepare_frame_context(index, previous)`` -- the PREVIOUS
    frame's tokens condition the whole next frame.  Inter-frame cascade.
  * ``boundary = _boundary_buckets(previous_cpu)`` -- the previous frame's tokens
    also pick the fixed-table correction row, and seed ``FreeCorrector``, whose
    statistics then run forward for the rest of the clip.

So one changed token in frame 283 perturbs the probability model for every
symbol after it, to the end of frame 599.  There is no per-token price to look
up.  The only honest measurement is to re-encode the WHOLE stream and stat the
archive, which is what this module does.

THE MIRROR.  ``encode_tail`` is ``decode_production_tokens`` line for line, with
exactly one substitution::

    symbols = decoder.decode(corrector.coding_row(state))     # shipped decoder
    symbols = target[frame].reshape(-1)[flat_positions]       # here: KNOWN
    encoder.encode(symbols, corrector.coding_row(state))      # ...and coded

``current`` is then filled from those symbols, so the encoder walks the exact
trajectory the receiver will walk when it decodes the stream we emit.  Nothing
about the model, the group order, the table, the corrector, or the probability
quantization is reimplemented -- all of it is imported from the shipped runtime.

THE CONTROL IS THE PROOF.  ``--stage control`` encodes the UNEDITED shipped token
field and requires the output to be **byte-identical to the shipped 109,696 B RC64
token stream** (the tail's 96 B fixed-residual prefix is carried through untouched,
not re-encoded).  MEASURED 2026-08-19: it is, over all 109,696 bytes, sha
``15054e5da33640bcb2e9d4589615c3b89b1312ce27fd9aa8e2a0ec0284b506f2``.
If that holds, this encoder is the exact inverse of the shipping decoder
on this body, and every byte delta it reports afterwards is real rather than
modelled.  If it does not hold, the module refuses and reports no delta.

RC64.  The shipped ``runtime/entropy/rc64_backend.c`` (sha 05839d14...) is
DECODER-ONLY.  The encoder half comes from the ddm_rr2 lineage source pinned at
sha ``5c75e2c7...`` plus ``route_b_rc64.RC64_CHECKPOINT_EXTENSION`` (snapshot and
resume).  That pairing is not assumed to be the right inverse -- the control
stage is what proves it, here, on this body.

RESUMABILITY (P0).  Every ``--checkpoint-every`` frames the module atomically
writes the RC64 encoder interval snapshot, the FreeCorrector state, the previous
frame, and the per-frame bit ledger.  ``--resume`` continues bit-faithfully.

RESUME WAS THE ``ddm_jg4`` DEFECT, AND THE CURE IS STRUCTURAL.  The v1 checkpoint
called ``corrector.state_dict()``.  That method is defined ONLY on the ``rr4``
base class (``runtime/rr4_free_corrector.py:318``) and returns 7 keys.  The live
``FreeCorrector`` is three subclasses deeper -- ``Ma1WithinMissCorrector`` ->
``Fx2ModelAxisMixer`` -> ``FixedPointLogisticMixer`` -> ``rr4.FreeCorrector`` --
and neither subclass overrides it, so the checkpoint silently dropped **9 of the
16 arrays the corrector owns plus all 39 arrays owned by its 13 ``MixerFamily``
members**: the 4000x13 mixer ``weights``, ``sse_weight``, ``_miss_counts`` /
``_miss_expect`` / ``_miss_seen``, and every family's ``counts``/``hits``/
``phat_q``.  A resumed run therefore restarted the whole model-mixing half COLD
while reporting nothing wrong.  MEASURED: the two failing controls diverge from
the shipped stream at exactly their own resume frame and nowhere earlier.

So the capture is no longer a hand-written key list.  ``corrector_state`` walks
the object STRUCTURALLY -- every ``__slots__`` entry on every class in the MRO,
plus ``__dict__``, plus the same walk over ``families`` -- so a future subclass
cannot add state the checkpoint silently forgets.  And
``uncaptured_divergent_state`` is the detector that would have caught v1: it
diffs the live corrector against a COLD one and refuses if any attribute that has
moved away from cold is absent from the capture.  The checkpoint carries a schema
tag; a v1 checkpoint is REFUSED rather than resumed.

ALWAYS KEEP THE PAYLOAD.  Every stage that materializes a stream writes the
stream bytes and the per-frame bit ledger to the store and records sha256 +
length.  No stage reports only a length.

AXIS.  ``[macOS-CPU advisory / scorer-free EXACT byte measurement]``.  The rate
leg this produces is exact -- it is ``archive.zip`` stat'ed -- but this module
computes no distortion and makes no score claim.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# The body.  Both paths are DERIVED from the pointer, not chosen for convenience.
# --------------------------------------------------------------------------------------

#: The runtime tree whose decoder this module mirrors.  The sibling
#: ``ddm_to1/generations/to1_tail_override_r1`` carries byte-identical ``runtime/``
#: and ``cpr1/`` trees but pins an older archive sha in its own ``inflate.py``, so
#: it is not used here.
DEFAULT_RUNTIME_ROOT = Path("/Volumes/APDataStore/pact/ddm_up3/candidate_runtime")

#: The pointer archive is DERIVED from ``--runtime-root``, never hand-typed.
#:
#: v1 pinned ``POINTER_ARCHIVE_SHA = 7ce46fd7...`` (the ``ddm_up3`` body, 176,420 B).
#: That literal was doing two jobs and only one of them honestly: it DID stop a
#: delta being spliced into some unrelated archive, but it also welded the module to
#: one body, so the live ``ddm_br1`` pointer (sha 44e9e650..., 176,429 B) could not
#: be measured at all -- even though both bodies carry the SAME token stream
#: (sha 15054e5d..., 109,696 B) and the SAME 109,792 B tail, differing only in the
#: RX1 ``reserved`` field and 9 B of carrier.  Hand-typing br1's numbers next to
#: up3's would just have moved the weld.
#:
#: What the check actually needs to guarantee is "the archive I splice into is the
#: body whose decoder I mirrored".  That is now asserted DIRECTLY -- the pointer
#: member must be byte-identical to ``<runtime-root>/archive.zip``'s member -- which
#: is strictly stronger than a sha literal and holds on any body.  ``--expect-
#: pointer-sha256`` remains available when a caller wants to nail one specific body.
def default_pointer_archive(runtime_root: Path) -> Path:
    return Path(runtime_root) / "archive.zip"

#: The shipped decoded token field (the receiver's own HPAC decode of the tail).
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_to1/advisory/attempt_0002/work/inflated"
    "/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)

#: The RC64 encoder-bearing source, pinned by ddm_rr2 (`RC64_SOURCE_SHA`).
RC64_BASE_SHA = "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6"
ROUTE_B = REPO / "experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py"

N_PAIRS = 600
EVAL_H, EVAL_W = 384, 512
PLANE = EVAL_H * EVAL_W
NUM_CLASSES = 5

#: The RX1 tail is NOT all coder output.  ``read_residual_archive`` (:478-494) splits
#: it into a 96 B compact fixed residual table and the RC64 token stream:
#:     tail (109,792 B) = residual_compact (96 B) + token_stream (109,696 B)
#: The control stage must reproduce the TOKEN STREAM byte-identically; the 96 B
#: prefix is carried through untouched.
RESIDUAL_COMPACT_BYTES = 96
SHIPPED_TAIL_BYTES = 109_792
SHIPPED_TOKEN_STREAM_BYTES = 109_696

SCORE_RATE_DENOMINATOR = 37_545_489
S_PER_ARCHIVE_BYTE = 25.0 / SCORE_RATE_DENOMINATOR
S_PER_SEG_CELL = 100.0 / (N_PAIRS * PLANE)


class Jg2Error(RuntimeError):
    """A stage refused."""


# --------------------------------------------------------------------------------------
# Small helpers.
# --------------------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True).encode())


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically persist one numpy checkpoint without suffix ambiguity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def persist_immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    """Write once; an identical retry reuses bytes and a changed retry refuses."""
    if path.is_file():
        if path.read_bytes() != payload:
            raise Jg2Error(f"immutable {label} changed on retry: {path}")
        return
    atomic_write(path, payload)


def persist_immutable_npy(path: Path, payload: np.ndarray, *, label: str) -> None:
    """Numpy counterpart of :func:`persist_immutable_bytes`."""
    if path.is_file():
        prior = np.load(path, allow_pickle=False)
        if not np.array_equal(prior, payload):
            raise Jg2Error(f"immutable {label} changed on retry: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        np.save(handle, payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def progress(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


# --------------------------------------------------------------------------------------
# Archive layout.  The RX1 header is READ, never assumed.
# --------------------------------------------------------------------------------------

RX1_HEADER = struct.Struct("<4sBBBBHHH")


def split_member(member: bytes) -> dict[str, bytes]:
    """Slice an RX1 member into its four sections plus the header."""
    if len(member) < RX1_HEADER.size:
        raise Jg2Error("member too short for an RX1 header")
    magic, _v, _a, _b, _reserved, hpac, semantic, carrier = RX1_HEADER.unpack_from(member)
    if magic != b"RX1M":
        raise Jg2Error(f"unexpected member magic {magic!r}")
    offset = RX1_HEADER.size
    sections: dict[str, bytes] = {"header": member[:offset]}
    for name, length in (("hpac", hpac), ("semantic", semantic), ("carrier", carrier)):
        sections[name] = member[offset : offset + length]
        offset += length
    sections["tail"] = member[offset:]
    return sections


def join_member(sections: dict[str, bytes]) -> bytes:
    return (
        sections["header"]
        + sections["hpac"]
        + sections["semantic"]
        + sections["carrier"]
        + sections["tail"]
    )


#: The shipped container's central-directory metadata.  MEASURED off the pointer
#: archive, not chosen: ``create_system = 3`` (Unix) and ``external_attr =
#: 0x81a40000`` (mode 0o100644 << 16).  Getting these wrong costs ZERO bytes and
#: still breaks a byte-close seal -- the first identity control this module ran used
#: zipfile's defaults (create_system 0, external_attr 0) and produced an archive of
#: EXACTLY the right length whose sha differed in precisely 3 bytes.  Length-equal
#: and byte-equal are different tests; a rate measurement only needs the first, a
#: seal needs both.
SHIPPED_ZIP_CREATE_SYSTEM = 3
SHIPPED_ZIP_EXTERNAL_ATTR = 0x81A40000
SHIPPED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def pack_archive(member: bytes, destination: Path) -> None:
    """Repack a single STORED member `p`, reproducing the shipped container exactly."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("p", date_time=SHIPPED_ZIP_DATE_TIME)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = SHIPPED_ZIP_EXTERNAL_ATTR
        info.create_system = SHIPPED_ZIP_CREATE_SYSTEM
        archive.writestr(info, member)
    os.replace(temporary, destination)


def read_archive_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["p"]:
            raise Jg2Error(f"archive must contain exactly member p, found {names}")
        return archive.read("p")


# --------------------------------------------------------------------------------------
# RC64 encoder: the pinned source plus route_b's snapshot/resume extension.
# --------------------------------------------------------------------------------------


def load_route_b():
    spec = importlib.util.spec_from_file_location("ddm_jg2_route_b_rc64", ROUTE_B)
    if spec is None or spec.loader is None:
        raise Jg2Error(f"cannot import route_b from {ROUTE_B}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_rc64_base(route_b, work: Path) -> Path:
    """Return the pinned encoder-bearing RC64 base source, sha-checked.

    ddm_rr2 pinned this source by sha and compiled it as
    ``base + RC64_CHECKPOINT_EXTENSION``.  Its generated combined file is the
    durable custody copy, so the base is recovered from it by removing exactly
    the extension bytes and then verified against the pin.  ``TAC_JG2_RC64_SOURCE``
    overrides with a direct path; it is sha-checked identically.
    """
    override = os.environ.get("TAC_JG2_RC64_SOURCE")
    if override:
        candidate = Path(override)
        if sha256_file(candidate) != RC64_BASE_SHA:
            raise Jg2Error(f"TAC_JG2_RC64_SOURCE sha mismatch: {candidate}")
        return candidate

    generated = Path(
        os.environ.get(
            "TAC_JG2_RC64_GENERATED",
            "/Volumes/APDataStore/pact/ddm_rr2_encoder_build/work/rc64_backend_checkpoint.c",
        )
    )
    if not generated.is_file():
        raise Jg2Error(
            "cannot locate the RC64 encoder source; set TAC_JG2_RC64_SOURCE to the "
            f"file with sha {RC64_BASE_SHA}"
        )
    extension = ("\n" + route_b.RC64_CHECKPOINT_EXTENSION).encode()
    blob = generated.read_bytes()
    if not blob.endswith(extension):
        raise Jg2Error("generated RC64 source does not end with the pinned extension")
    base = blob[: len(blob) - len(extension)]
    if sha256_bytes(base) != RC64_BASE_SHA:
        raise Jg2Error("recovered RC64 base source fails its pinned sha")
    out = work / "rc64_base.c"
    atomic_write(out, base)
    return out


def compile_rc64(work: Path, route_b, tag: str) -> tuple[Path, dict[str, object]]:
    """Build the RC64 encoder into a PER-STAGE directory.

    The module's own docstring says the control and the edited encode may run
    concurrently.  They could not: both compiled ``rc64_backend_jg2.c`` and linked
    ``librc64_jg2.dylib`` at the same paths inside one ``--store``, so two live
    stages raced on the same ``cc -o`` output.  Giving the build its own directory
    per stage costs nothing and makes the concurrency the docstring promises real.
    """
    build = work / f"rc64_{tag}"
    build.mkdir(parents=True, exist_ok=True)
    base = resolve_rc64_base(route_b, build)
    generated = build / "rc64_backend_jg2.c"
    library = build / "librc64_jg2.dylib"
    source = base.read_bytes() + ("\n" + route_b.RC64_CHECKPOINT_EXTENSION).encode()
    atomic_write(generated, source)
    command = [
        "/usr/bin/cc",
        "-O3",
        "-std=c11",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(generated),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)
    return library, {
        "argv": command,
        "base_source": file_fact(base),
        "generated": file_fact(generated),
        "library": file_fact(library),
    }


# --------------------------------------------------------------------------------------
# The shipped runtime, imported rather than reimplemented.
# --------------------------------------------------------------------------------------


def load_runtime(root: Path):
    """Import the shipped runtime package and the cpr1 renderer module."""
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    renderer_dir = root / "cpr1"
    if str(renderer_dir) not in sys.path:
        sys.path.insert(0, str(renderer_dir))

    residual = importlib.import_module("runtime.residual_archive")
    spec = importlib.util.spec_from_file_location(
        "jg2_cpr1_inflate", renderer_dir / "inflate.py"
    )
    if spec is None or spec.loader is None:
        raise Jg2Error("cannot import cpr1/inflate.py")
    renderer = importlib.util.module_from_spec(spec)
    sys.modules["jg2_cpr1_inflate"] = renderer
    spec.loader.exec_module(renderer)
    return residual, renderer, renderer_dir


def load_tokens(path: Path) -> np.ndarray:
    """Memory-map the (600, 384, 512) uint8 token field."""
    expected = N_PAIRS * PLANE
    size = path.stat().st_size
    if size != expected:
        raise Jg2Error(f"token field must be {expected} B, found {size} B at {path}")
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(N_PAIRS, EVAL_H, EVAL_W))


def apply_edits(tokens: np.ndarray, edits_path: Path | None) -> tuple[np.ndarray, dict]:
    """Return a writable token field with the edited pairs spliced in.

    The edit file is an ``.npz`` whose keys are pair indices and whose values are
    ``(384, 512) uint8`` replacement planes -- the shape ``ddm_jg1`` retained.
    """
    field = np.array(tokens, dtype=np.uint8)
    if edits_path is None:
        return field, {"edited_pairs": [], "tokens_changed": 0}
    blob = np.load(edits_path)
    changed = 0
    pairs: list[int] = []
    for key in blob.files:
        pair = int(key)
        plane = np.asarray(blob[key], dtype=np.uint8)
        if plane.shape != (EVAL_H, EVAL_W):
            raise Jg2Error(f"edit plane {key} has shape {plane.shape}")
        if plane.max() >= NUM_CLASSES:
            raise Jg2Error(f"edit plane {key} carries a token outside 0..{NUM_CLASSES - 1}")
        changed += int((plane != field[pair]).sum())
        field[pair] = plane
        pairs.append(pair)
    return field, {
        "edited_pairs": sorted(pairs),
        "tokens_changed": changed,
        "edits_file": file_fact(edits_path),
    }


class EditOverlay:
    """Read-only token field that overlays retained replacement planes lazily.

    The historical ``apply_edits`` helper materializes all 117,964,800 candidate
    bytes.  Exact-coder searches only need one plane at a time, so this view keeps
    the retained ``.npz`` as the candidate payload and never creates a disposable
    full-field copy.
    """

    def __init__(self, base: np.ndarray, planes: dict[int, np.ndarray]) -> None:
        self.base = base
        self.planes = planes
        self.shape = base.shape
        self.dtype = base.dtype

    def __getitem__(self, key: object) -> np.ndarray | np.uint8:
        if isinstance(key, tuple):
            pair = key[0]
            if isinstance(pair, (int, np.integer)) and int(pair) in self.planes:
                return self.planes[int(pair)][key[1:]]
            return self.base[key]
        if isinstance(key, (int, np.integer)) and int(key) in self.planes:
            return self.planes[int(key)]
        return self.base[key]


def load_edit_overlay(
    tokens: np.ndarray, edits_path: Path | None
) -> tuple[np.ndarray | EditOverlay, dict[str, object]]:
    """Load retained pair-plane edits without materializing a full candidate field."""
    if edits_path is None:
        return tokens, {"edited_pairs": [], "tokens_changed": 0}
    planes: dict[int, np.ndarray] = {}
    changed = 0
    with np.load(edits_path, allow_pickle=False) as blob:
        for key in blob.files:
            pair = int(key)
            if pair < 0 or pair >= N_PAIRS:
                raise Jg2Error(f"edit pair {pair} is outside 0..{N_PAIRS - 1}")
            plane = np.asarray(blob[key], dtype=np.uint8)
            if plane.shape != (EVAL_H, EVAL_W):
                raise Jg2Error(f"edit plane {key} has shape {plane.shape}")
            if plane.max() >= NUM_CLASSES:
                raise Jg2Error(
                    f"edit plane {key} carries a token outside 0..{NUM_CLASSES - 1}"
                )
            if pair in planes:
                raise Jg2Error(f"duplicate edit pair {pair}")
            retained = plane.copy()
            retained.setflags(write=False)
            changed += int((retained != tokens[pair]).sum())
            planes[pair] = retained
    return EditOverlay(tokens, planes), {
        "edited_pairs": sorted(planes),
        "tokens_changed": changed,
        "edits_file": file_fact(edits_path),
        "materialized_full_field": False,
    }


# --------------------------------------------------------------------------------------
# CORRECTOR STATE.  Captured structurally, so a subclass cannot be forgotten.
# --------------------------------------------------------------------------------------

#: Bumped when the capture changes shape.  A v1 checkpoint (``corrector.state_dict()``,
#: 7 keys, model-mixing half missing) is REFUSED, never resumed: it is not a slower
#: path to the same answer, it is a different and wrong answer.
CHECKPOINT_SCHEMA = "ddm_jg4.corrector_state.v3"

#: Ledger keys the checkpoint owns.  Namespaced state keys all contain a ``.``, so
#: they cannot collide with these -- v1's "every key except these four" filter was
#: one careless name away from mis-restoring.
LEDGER_KEYS = frozenset(
    {"schema", "frame", "code_bits", "per_frame", "previous", "encoder_sha256"}
)


def state_names(obj: object) -> list[str]:
    """Every attribute name the object can hold, across its whole MRO.

    ``vars()`` alone is NOT enough and that is the whole point: the ``rr4`` base
    declares ``__slots__ = ('boundary','counts','have_prev','hits','phat_q','plane',
    'prev1','prev2','run')`` while the three subclasses above it use ``__dict__``.
    A ``vars()`` walk silently misses the base; a ``__slots__`` walk silently misses
    the subclasses.  Only the union sees the whole object.
    """
    names: list[str] = []
    for cls in type(obj).__mro__:
        for name in cls.__dict__.get("__slots__", ()) or ():
            if name not in names:
                names.append(name)
    for name in getattr(obj, "__dict__", {}):
        if name not in names:
            names.append(name)
    return sorted(names)


def _capture_value(value: object) -> np.ndarray | None:
    """Serialize one attribute, or ``None`` if it is not state."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bool, np.bool_)):
        return np.array([bool(value)], dtype=np.bool_)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return np.array([int(value)], dtype=np.int64)
    if isinstance(value, (float, np.floating)):
        return np.array([float(value)], dtype=np.float64)
    return None


def corrector_state(corrector: object) -> dict[str, np.ndarray]:
    """Every mutable value the corrector and its families own."""
    state: dict[str, np.ndarray] = {}
    for name in state_names(corrector):
        if name == "families":
            continue  # walked below, per family
        captured = _capture_value(getattr(corrector, name, None))
        if captured is not None:
            state[f"self.{name}"] = captured
    for index, family in enumerate(getattr(corrector, "families", ()) or ()):
        for name in state_names(family):
            captured = _capture_value(getattr(family, name, None))
            if captured is not None:
                state[f"fam.{index:02d}.{name}"] = captured
    return state


def _resolve_target(corrector: object, key: str) -> tuple[object, str]:
    if key.startswith("self."):
        return corrector, key[len("self.") :]
    if key.startswith("fam."):
        _, index, name = key.split(".", 2)
        return corrector.families[int(index)], name  # type: ignore[attr-defined]
    raise Jg2Error(f"unknown corrector state key {key!r}")


def load_corrector_state(corrector: object, state: dict[str, np.ndarray]) -> None:
    """Restore in place, then PROVE the restore landed.

    The round-trip assertion is not ceremony.  A silent partial restore is exactly
    the defect this function exists to end, so the function refuses to return
    having half-done its job.
    """
    for key, value in state.items():
        owner, name = _resolve_target(corrector, key)
        current = getattr(owner, name, None)
        array = np.asarray(value)
        if isinstance(current, np.ndarray):
            setattr(
                owner,
                name,
                np.asarray(array, dtype=current.dtype).reshape(current.shape).copy(),
            )
        elif isinstance(current, (bool, np.bool_)):
            setattr(owner, name, bool(array.reshape(-1)[0]))
        elif isinstance(current, (int, np.integer)) and not isinstance(current, bool):
            setattr(owner, name, int(array.reshape(-1)[0]))
        elif isinstance(current, (float, np.floating)):
            setattr(owner, name, float(array.reshape(-1)[0]))
        else:
            raise Jg2Error(f"cannot restore {key}: live attribute is {type(current)!r}")

    restored = corrector_state(corrector)
    if set(restored) != set(state):
        missing = sorted(set(state) - set(restored)) + sorted(set(restored) - set(state))
        raise Jg2Error(f"corrector restore changed the state key set: {missing[:8]}")
    for key, value in state.items():
        if not np.array_equal(restored[key], np.asarray(value)):
            raise Jg2Error(f"corrector restore did not land for {key}")


def uncaptured_divergent_state(
    corrector: object, cold: object, captured: set[str]
) -> list[str]:
    """THE DETECTOR: any attribute that has moved away from cold and is not saved.

    This is what makes the cure structural rather than a longer key list.  A cold
    corrector is the state a resumed run would silently start from, so anything that
    differs from cold and is not in the checkpoint is, by definition, state the
    resume would lose.  Run against v1's capture this returns the 48 arrays v1
    dropped; run against v2's it returns nothing.

    Callables (the cell/mixer/SSE rules) are construction-time and are skipped.
    """
    lost: list[str] = []

    def compare(live_owner: object, cold_owner: object, prefix: str) -> None:
        for name in state_names(live_owner):
            if name == "families":
                continue
            live = getattr(live_owner, name, None)
            frozen = getattr(cold_owner, name, None)
            if callable(live) or callable(frozen):
                continue
            if isinstance(live, np.ndarray) or isinstance(frozen, np.ndarray):
                same = (
                    isinstance(live, np.ndarray)
                    and isinstance(frozen, np.ndarray)
                    and live.shape == frozen.shape
                    and bool(np.array_equal(live, frozen))
                )
            else:
                try:
                    same = bool(live == frozen)
                except Exception:  # pragma: no cover - exotic attribute
                    same = False
            if not same and f"{prefix}{name}" not in captured:
                lost.append(f"{prefix}{name}")

    compare(corrector, cold, "self.")
    live_families = list(getattr(corrector, "families", ()) or ())
    cold_families = list(getattr(cold, "families", ()) or ())
    if len(live_families) != len(cold_families):
        lost.append("families:length")
    for index, (live, frozen) in enumerate(
        zip(live_families, cold_families, strict=True)
    ):
        compare(live, frozen, f"fam.{index:02d}.")
    return lost


def checkpoint_bundle_paths(root: Path, frame: int) -> tuple[Path, Path]:
    """Return the immutable state/RC64 paths for a pair-boundary checkpoint."""
    if frame < 0 or frame > N_PAIRS:
        raise Jg2Error(f"checkpoint frame must be in 0..{N_PAIRS}, found {frame}")
    return root / f"frame_{frame:04d}.npz", root / f"frame_{frame:04d}.encoder.bin"


def persist_checkpoint_bundle(
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    frame: int,
    code_bits: float,
    per_frame: np.ndarray,
    previous: np.ndarray,
    corrector: object,
    encoder: object,
    cold: object | None,
    replace: bool = False,
) -> dict[str, object]:
    """Persist the complete HPAC/corrector plus RC64 state at one boundary.

    The RC64 snapshot includes the already-emitted prefix bytes and interval.  The
    numpy side includes every mutable corrector/mixer table, the previous decoded
    plane, and the exact per-frame ledger.  Together they are the complete state
    needed to resume the existing physical encoder, not a rate proxy.
    """
    state = corrector_state(corrector)
    if cold is not None:
        lost = uncaptured_divergent_state(corrector, cold, set(state))
        if lost:
            raise Jg2Error(
                f"checkpoint at frame {frame} would LOSE corrector state that has "
                f"moved away from cold: {lost}. Extend the capture before writing "
                "a checkpoint that cannot be resumed faithfully."
            )
    encoder_payload = encoder.snapshot()  # type: ignore[attr-defined]
    checkpoint_payload = {
        "schema": np.array([CHECKPOINT_SCHEMA]),
        "frame": np.array([frame], dtype=np.int64),
        "code_bits": np.array([code_bits], dtype=np.float64),
        "per_frame": np.asarray(per_frame, dtype=np.float64),
        "previous": np.asarray(previous, dtype=np.uint8),
        # Cross-bind the two separately atomic files.  A crash between their
        # replacements may leave both paths present but from different frames;
        # load must refuse that mixed bundle instead of silently resuming it.
        "encoder_sha256": np.array([sha256_bytes(encoder_payload)]),
        **state,
    }
    existing = checkpoint_path.is_file(), encoder_path.is_file()
    if any(existing) and not all(existing):
        raise Jg2Error(
            f"refusing incomplete pre-existing checkpoint bundle: "
            f"{checkpoint_path} / {encoder_path}"
        )
    if all(existing) and not replace:
        if encoder_path.read_bytes() != encoder_payload:
            raise Jg2Error(f"immutable RC64 checkpoint changed at frame {frame}")
        with np.load(checkpoint_path, allow_pickle=False) as prior:
            if set(prior.files) != set(checkpoint_payload):
                raise Jg2Error(f"immutable checkpoint key set changed at frame {frame}")
            for key, value in checkpoint_payload.items():
                if not np.array_equal(prior[key], value):
                    raise Jg2Error(
                        f"immutable checkpoint value {key!r} changed at frame {frame}"
                    )
    else:
        atomic_write(encoder_path, encoder_payload)
        atomic_npz(checkpoint_path, checkpoint_payload)
    return {
        "frame": frame,
        "state_keys": len(state),
        "checkpoint": file_fact(checkpoint_path),
        "encoder_state": file_fact(encoder_path),
    }


def load_checkpoint_bundle(
    *,
    checkpoint_path: Path,
    encoder_path: Path,
    corrector: object,
    route_b: object,
    library: Path,
) -> tuple[int, float, np.ndarray, np.ndarray, object, dict[str, object]]:
    """Load and prove one complete boundary checkpoint before returning it."""
    if not checkpoint_path.is_file() or not encoder_path.is_file():
        raise Jg2Error(
            f"checkpoint bundle is incomplete: {checkpoint_path} / {encoder_path}"
        )
    encoder_payload = encoder_path.read_bytes()
    with np.load(checkpoint_path, allow_pickle=False) as blob:
        schema = (
            str(np.asarray(blob["schema"]).reshape(-1)[0])
            if "schema" in blob.files
            else "v1"
        )
        if schema != CHECKPOINT_SCHEMA:
            raise Jg2Error(
                f"REFUSING a {schema!r} checkpoint at {checkpoint_path}. Only "
                f"{CHECKPOINT_SCHEMA!r} carries the full corrector state."
            )
        expected_encoder_sha256 = str(np.asarray(blob["encoder_sha256"]).reshape(-1)[0])
        if sha256_bytes(encoder_payload) != expected_encoder_sha256:
            raise Jg2Error(
                f"checkpoint bundle is cross-file inconsistent at frame {int(blob['frame'][0])}: "
                "the RC64 state digest does not match the numpy checkpoint"
            )
        frame = int(blob["frame"][0])
        code_bits = float(blob["code_bits"][0])
        per_frame = np.asarray(blob["per_frame"], dtype=np.float64).copy()
        previous = np.asarray(blob["previous"], dtype=np.uint8).copy()
        state_keys_restored = len(blob.files) - len(LEDGER_KEYS)
        load_corrector_state(
            corrector, {key: blob[key] for key in blob.files if key not in LEDGER_KEYS}
        )
    encoder = route_b.NativeRc64Encoder(library, encoder_payload)
    fact = {
        "frame": frame,
        "schema": schema,
        "state_keys_restored": state_keys_restored,
        "checkpoint": file_fact(checkpoint_path),
        "encoder_state": file_fact(encoder_path),
    }
    return frame, code_bits, per_frame, previous, encoder, fact


# --------------------------------------------------------------------------------------
# THE MIRROR.
# --------------------------------------------------------------------------------------


def encode_tail(
    *,
    residual,
    renderer,
    renderer_dir: Path,
    parts,
    target: np.ndarray,
    library: Path,
    route_b,
    work: Path,
    tag: str,
    frames: int,
    checkpoint_every: int,
    resume: bool,
    resume_checkpoint: Path | None = None,
    resume_encoder_state: Path | None = None,
    checkpoint_history: Path | None = None,
    checkpoint_frames: set[int] | None = None,
    retain_terminal_checkpoint: bool = False,
) -> dict[str, object]:
    """Re-encode the token field along the receiver's own decode trajectory.

    This is ``runtime/residual_archive.py::decode_production_tokens`` with the
    decode call replaced by an encode of the KNOWN symbols.  Every other element
    -- model, group plan, boundary buckets, fixed table, FreeCorrector, and the
    probability quantization -- is the shipped object's own.
    """
    import torch
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    from runtime.hpac_inference import (  # type: ignore[import-not-found]
        optimize_sparse_evaluator,
    )

    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    corrector = FreeCorrector(renderer.EVAL_H * renderer.EVAL_W)

    group_plans = []
    for mask in masks:
        mask_array = mask.detach().cpu().numpy()
        flat_positions = np.flatnonzero(mask_array.reshape(-1))
        group_plans.append((torch.from_numpy(flat_positions).to(device), flat_positions))

    checkpoint_path = work / f"encode_{tag}.checkpoint.npz"
    encoder_state = work / f"encode_{tag}.encoder.bin"
    start_frame = 0
    code_bits = 0.0
    per_frame = np.zeros(N_PAIRS, dtype=np.float64)
    previous_seed: np.ndarray | None = None

    #: The cold reference the DETECTOR diffs against.  Built once, not per checkpoint.
    keep_state = bool(
        checkpoint_every
        or checkpoint_history is not None
        or retain_terminal_checkpoint
        or resume_checkpoint is not None
    )
    cold = FreeCorrector(renderer.EVAL_H * renderer.EVAL_W) if keep_state else None
    resumed_from: dict[str, object] | None = None

    if resume and checkpoint_path.is_file() and encoder_state.is_file():
        (
            start_frame,
            code_bits,
            per_frame,
            previous_seed,
            encoder,
            resumed_from,
        ) = load_checkpoint_bundle(
            checkpoint_path=checkpoint_path,
            encoder_path=encoder_state,
            corrector=corrector,
            route_b=route_b,
            library=library,
        )
        progress(
            {
                "stage": f"encode_{tag}",
                "event": "resumed",
                "frame": start_frame,
                "schema": resumed_from["schema"],
                "state_keys_restored": resumed_from["state_keys_restored"],
            }
        )
    elif resume_checkpoint is not None or resume_encoder_state is not None:
        if resume_checkpoint is None or resume_encoder_state is None:
            raise Jg2Error("explicit resume requires both checkpoint and encoder state")
        (
            start_frame,
            code_bits,
            per_frame,
            previous_seed,
            encoder,
            resumed_from,
        ) = load_checkpoint_bundle(
            checkpoint_path=resume_checkpoint,
            encoder_path=resume_encoder_state,
            corrector=corrector,
            route_b=route_b,
            library=library,
        )
        progress(
            {
                "stage": f"encode_{tag}",
                "event": "resumed_explicit",
                "frame": start_frame,
                "schema": resumed_from["schema"],
                "state_keys_restored": resumed_from["state_keys_restored"],
            }
        )
    else:
        encoder = route_b.NativeRc64Encoder(library)

    immutable_checkpoints: list[dict[str, object]] = []
    requested_frames = set(checkpoint_frames or set())
    if any(frame < 0 or frame > frames for frame in requested_frames):
        raise Jg2Error(f"checkpoint frames must be in 0..{frames}: {sorted(requested_frames)}")
    if checkpoint_history is not None and start_frame in requested_frames:
        history_checkpoint, history_encoder = checkpoint_bundle_paths(
            checkpoint_history, start_frame
        )
        immutable_checkpoints.append(
            persist_checkpoint_bundle(
                checkpoint_path=history_checkpoint,
                encoder_path=history_encoder,
                frame=start_frame,
                code_bits=code_bits,
                per_frame=per_frame,
                previous=(
                    previous_seed
                    if previous_seed is not None
                    else np.zeros((EVAL_H, EVAL_W), dtype=np.uint8)
                ),
                corrector=corrector,
                encoder=encoder,
                cold=cold,
            )
        )

    started = time.perf_counter()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        if previous_seed is not None:
            previous = torch.from_numpy(previous_seed.astype(np.int64)).reshape(
                1, EVAL_H, EVAL_W
            ).to(device)
        else:
            previous = torch.zeros((1, EVAL_H, EVAL_W), dtype=torch.long, device=device)

        for frame in range(start_frame, frames):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)

            plane_target = np.asarray(target[frame], dtype=np.uint8).reshape(-1)
            frame_bits = 0.0
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = (
                    boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                )
                corrected = base_logits + parts.table.values[feature]
                probability = residual._probability_table(
                    corrected, renderer.HPAC_LOGIT_PRECISION
                )
                state = corrector.group_state(probability, predicted, flat_positions)
                coding = corrector.coding_row(state)

                symbols = plane_target[flat_positions].astype(np.int64)
                frame_bits += _row_bits(coding, symbols)
                encoder.encode(symbols.astype(np.int32), coding)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)

            code_bits += frame_bits
            per_frame[frame] = frame_bits
            frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
            if not np.array_equal(frame_tokens.reshape(-1), plane_target):
                raise Jg2Error(f"frame {frame}: encoded field diverged from the target")
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current

            boundary = frame + 1
            if checkpoint_every and boundary % checkpoint_every == 0 and boundary < frames:
                saved = persist_checkpoint_bundle(
                    checkpoint_path=checkpoint_path,
                    encoder_path=encoder_state,
                    frame=boundary,
                    code_bits=code_bits,
                    per_frame=per_frame,
                    previous=frame_tokens,
                    corrector=corrector,
                    encoder=encoder,
                    cold=cold,
                    replace=True,
                )
                progress(
                    {
                        "stage": f"encode_{tag}",
                        "event": "checkpoint",
                        "frame": boundary,
                        "code_bytes_so_far": code_bits / 8.0,
                        "state_keys_saved": saved["state_keys"],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
            if checkpoint_history is not None and boundary in requested_frames:
                history_checkpoint, history_encoder = checkpoint_bundle_paths(
                    checkpoint_history, boundary
                )
                immutable_checkpoints.append(
                    persist_checkpoint_bundle(
                        checkpoint_path=history_checkpoint,
                        encoder_path=history_encoder,
                        frame=boundary,
                        code_bits=code_bits,
                        per_frame=per_frame,
                        previous=frame_tokens,
                        corrector=corrector,
                        encoder=encoder,
                        cold=cold,
                    )
                )

    terminal_checkpoint: dict[str, object] | None = None
    if retain_terminal_checkpoint:
        terminal_root = work / f"encode_{tag}.terminal"
        terminal_checkpoint_path, terminal_encoder_path = checkpoint_bundle_paths(
            terminal_root, frames
        )
        terminal_checkpoint = persist_checkpoint_bundle(
            checkpoint_path=terminal_checkpoint_path,
            encoder_path=terminal_encoder_path,
            frame=frames,
            code_bits=code_bits,
            per_frame=per_frame,
            previous=np.asarray(target[frames - 1], dtype=np.uint8),
            corrector=corrector,
            encoder=encoder,
            cold=cold,
        )

    payload = encoder.finish()
    if not payload.startswith(route_b.TOKEN_MAGIC):
        raise Jg2Error("RC64 payload lost its magic")
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    if not size or not pointer:
        raise Jg2Error("RC64 encoder produced no payload")
    body = ctypes.string_at(pointer, size)

    # ALWAYS KEEP THE PAYLOAD: the stream and the per-frame ledger are both
    # materialized here, so both are persisted rather than reduced to a length.
    stream_path = work / f"tail_{tag}.bin"
    persist_immutable_bytes(stream_path, body, label="RC64 stream")
    ledger_path = work / f"bits_per_frame_{tag}.npy"
    persist_immutable_npy(ledger_path, per_frame, label="per-frame bit ledger")

    return {
        "tag": tag,
        "frames": frames,
        "code_bits": code_bits,
        "code_bytes_ideal": code_bits / 8.0,
        "stream": file_fact(stream_path),
        "bits_per_frame_ledger": file_fact(ledger_path),
        "elapsed_seconds": time.perf_counter() - started,
        "start_frame": start_frame,
        "resumed_from": resumed_from,
        "immutable_checkpoints": immutable_checkpoints,
        "terminal_checkpoint": terminal_checkpoint,
    }


def _row_bits(rows: np.ndarray, symbols: np.ndarray) -> float:
    values = rows.astype(np.float64)
    picked = values[np.arange(values.shape[0]), symbols]
    return float(-np.log2(np.maximum(picked, 1e-300)).sum())


# --------------------------------------------------------------------------------------
# Stages.
# --------------------------------------------------------------------------------------


def _prepare(args, tag: str) -> dict[str, Any]:
    work = Path(args.store) / "work"
    work.mkdir(parents=True, exist_ok=True)
    route_b = load_route_b()
    library, build = compile_rc64(work, route_b, tag)
    residual, renderer, renderer_dir = load_runtime(Path(args.runtime_root))
    archive = Path(args.runtime_root) / "archive.zip"
    parts = residual.read_residual_archive(archive)
    sections = split_member(read_archive_member(archive))
    # The tail is `residual_compact || token_stream`; the parser owns that split, so
    # it is CHECKED here rather than assumed.
    tail = sections["tail"]
    if tail[RESIDUAL_COMPACT_BYTES:] != parts.token_stream:
        raise Jg2Error(
            "sliced tail suffix disagrees with the parsed token stream "
            f"({len(tail) - RESIDUAL_COMPACT_BYTES} B vs {len(parts.token_stream)} B)"
        )
    sections["residual_compact"] = tail[:RESIDUAL_COMPACT_BYTES]
    sections["token_stream"] = parts.token_stream
    return {
        "work": work,
        "route_b": route_b,
        "library": library,
        "build": build,
        "residual": residual,
        "renderer": renderer,
        "renderer_dir": renderer_dir,
        "parts": parts,
        "sections": sections,
    }


def stage_control(args) -> dict[str, object]:
    """Encode the UNEDITED field; refuse unless the tail is byte-identical."""
    env = _prepare(args, f"control_{args.frames}")
    tokens = load_tokens(Path(args.tokens))
    shipped_stream = env["sections"]["token_stream"]
    result = encode_tail(
        residual=env["residual"],
        renderer=env["renderer"],
        renderer_dir=env["renderer_dir"],
        parts=env["parts"],
        target=tokens,
        library=env["library"],
        route_b=env["route_b"],
        work=env["work"],
        tag=f"control_{args.frames}",
        frames=args.frames,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    emitted = Path(result["stream"]["path"]).read_bytes()  # type: ignore[index]
    full_run = args.frames == N_PAIRS
    # A partial run cannot match the length (the coder has not flushed the rest of
    # the clip), so the honest partial signal is how far the PREFIX agrees.
    common = min(len(emitted), len(shipped_stream))
    prefix_match = int(
        next((i for i in range(common) if emitted[i] != shipped_stream[i]), common)
    )
    identical = full_run and emitted == shipped_stream
    verdict = {
        **result,
        "shipped_token_stream_bytes": len(shipped_stream),
        "shipped_token_stream_sha256": sha256_bytes(shipped_stream),
        "emitted_bytes": len(emitted),
        "emitted_sha256": sha256_bytes(emitted),
        "byte_identical": identical,
        "prefix_bytes_matching": prefix_match,
        "full_run": full_run,
        "rc64_build": env["build"],
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
    }
    atomic_json(Path(args.store) / "retained" / f"S1_control_{args.frames}.json", verdict)
    if full_run and not identical:
        raise Jg2Error(
            "CONTROL FAILED: the re-encoded token stream is not byte-identical to the "
            f"shipped stream ({len(emitted)} B vs {len(shipped_stream)} B; prefix agrees "
            f"for {prefix_match} B). No byte delta from this encoder is trustworthy "
            "until this passes."
        )
    return verdict


def stage_encode(args) -> dict[str, object]:
    """Encode an EDITED field and report the REAL archive byte delta."""
    # The control is the PROOF that this encoder inverts the shipping decoder, but the
    # two encodes are independent compute, so the control is required at REPORTING time
    # rather than at start time.  That lets both run concurrently without ever letting
    # an unproven encoder emit a trusted delta.
    env = _prepare(args, args.tag)
    tokens = load_tokens(Path(args.tokens))
    field, edit_report = apply_edits(tokens, Path(args.edits) if args.edits else None)
    result = encode_tail(
        residual=env["residual"],
        renderer=env["renderer"],
        renderer_dir=env["renderer_dir"],
        parts=env["parts"],
        target=field,
        library=env["library"],
        route_b=env["route_b"],
        work=env["work"],
        tag=args.tag,
        frames=args.frames,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    emitted = Path(result["stream"]["path"]).read_bytes()  # type: ignore[index]
    shipped_stream = env["sections"]["token_stream"]
    shipped_tail = env["sections"]["tail"]

    # NO-FAKE: a byte delta is only the POINTER's delta if it was spliced into the
    # pointer's own bytes.  Verify that DIRECTLY -- the archive being spliced must be
    # the very body whose decoder this encoder mirrored -- rather than against a sha
    # literal that only recognises one historical body.
    pointer_path = Path(args.pointer_archive)
    pointer_sha = sha256_file(pointer_path)
    if args.expect_pointer_sha256 and pointer_sha != args.expect_pointer_sha256:
        raise Jg2Error(
            f"--pointer-archive {pointer_path} has sha {pointer_sha}, but "
            f"--expect-pointer-sha256 requires {args.expect_pointer_sha256}"
        )
    pointer_member = read_archive_member(pointer_path)
    runtime_member = read_archive_member(Path(args.runtime_root) / "archive.zip")
    if pointer_member != runtime_member:
        raise Jg2Error(
            f"--pointer-archive {pointer_path} is not the body this encoder mirrored: "
            f"its member ({len(pointer_member)} B) differs from "
            f"{args.runtime_root}/archive.zip ({len(runtime_member)} B). Splicing a "
            "stream into a body whose decoder produced a different one is not a delta."
        )
    pointer_sections = split_member(pointer_member)
    if pointer_sections["tail"] != shipped_tail:
        raise Jg2Error("pointer tail differs from the runtime tail; bodies are not siblings")
    # Splice the NEW coder output behind the untouched 96 B fixed residual table.
    pointer_sections["tail"] = shipped_tail[:RESIDUAL_COMPACT_BYTES] + emitted
    candidate = Path(args.store) / "retained" / f"candidate_{args.tag}.zip"
    pack_archive(join_member(pointer_sections), candidate)

    base_archive_bytes = pointer_path.stat().st_size
    candidate_bytes = candidate.stat().st_size
    delta_bytes = candidate_bytes - base_archive_bytes
    changed = int(edit_report["tokens_changed"])

    control_path = Path(args.store) / "retained" / f"S1_control_{N_PAIRS}.json"
    # The control and this encode are the same length of compute, so whichever
    # finishes first would otherwise report UNPROVEN purely on ordering and force a
    # hand reconciliation afterwards.  Waiting is honest -- it changes nothing about
    # what is proved, only when this process looks.
    if args.wait_for_control_seconds and not control_path.is_file():
        progress({"stage": "encode", "event": "waiting_for_control",
                  "path": str(control_path), "seconds": args.wait_for_control_seconds})
        deadline = time.monotonic() + args.wait_for_control_seconds
        while time.monotonic() < deadline and not control_path.is_file():
            time.sleep(5.0)

    control_ok = False
    control_fact: dict[str, object] | None = None
    if control_path.is_file():
        control = json.loads(control_path.read_text())
        control_ok = bool(control.get("byte_identical"))
        control_fact = {
            "path": str(control_path),
            "byte_identical": control_ok,
            "emitted_sha256": control.get("emitted_sha256"),
            "shipped_token_stream_sha256": control.get("shipped_token_stream_sha256"),
        }

    verdict = {
        **result,
        **edit_report,
        "token_stream_bytes_base": len(shipped_stream),
        "token_stream_bytes_candidate": len(emitted),
        "token_stream_delta_bytes": len(emitted) - len(shipped_stream),
        "archive_bytes_base": base_archive_bytes,
        "archive_bytes_candidate": candidate_bytes,
        "archive_delta_bytes": delta_bytes,
        "pointer_archive": {"path": str(pointer_path), "bytes": base_archive_bytes,
                            "sha256": pointer_sha},
        "candidate_archive": file_fact(candidate),
        "delta_S_rate": delta_bytes * S_PER_ARCHIVE_BYTE,
        "measured_bits_per_changed_token": (delta_bytes * 8.0 / changed) if changed else None,
        "modelled_bits_per_changed_token_jg1": 4.718,
        "realized_over_modelled": (
            (delta_bytes * 8.0 / changed) / 4.718 if changed else None
        ),
        "control": control_fact,
        "delta_trustworthy": control_ok,
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte measurement]",
        "score_claim": False,
    }
    atomic_json(Path(args.store) / "retained" / f"S1_encode_{args.tag}.json", verdict)
    if not control_ok:
        progress(
            {
                "stage": "encode",
                "event": "UNPROVEN",
                "note": (
                    "payload retained, but the 600-frame control has not proved this "
                    "encoder byte-identical on this body; the delta is NOT trustworthy"
                ),
            }
        )
    return verdict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("control", "encode"))
    parser.add_argument("--store", required=True, help="custody directory for this arm")
    parser.add_argument("--runtime-root", default=str(DEFAULT_RUNTIME_ROOT))
    parser.add_argument(
        "--pointer-archive",
        default=None,
        help="archive to splice into; defaults to <runtime-root>/archive.zip",
    )
    parser.add_argument(
        "--expect-pointer-sha256",
        default=None,
        help="optional pin: refuse unless the pointer archive has this sha256",
    )
    parser.add_argument("--tokens", default=str(DEFAULT_TOKENS))
    parser.add_argument("--edits", default=None, help="npz of {pair: (384,512) uint8}")
    parser.add_argument("--tag", default="edited")
    parser.add_argument("--frames", type=int, default=N_PAIRS)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wait-for-control-seconds",
        type=float,
        default=0.0,
        help="encode stage: poll this long for a concurrent control receipt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1 or args.frames > N_PAIRS:
        raise SystemExit(f"--frames must be in 1..{N_PAIRS}")
    if args.pointer_archive is None:
        args.pointer_archive = str(default_pointer_archive(Path(args.runtime_root)))
    stage = {"control": stage_control, "encode": stage_encode}[args.stage]
    verdict = stage(args)
    progress({"stage": args.stage, "event": "done", **{
        k: v for k, v in verdict.items() if not isinstance(v, dict)
    }})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
