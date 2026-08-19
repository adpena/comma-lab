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

#: The runtime tree whose decoder this module mirrors.  This is the POINTER body:
#: its ``archive.zip`` IS sha 7ce46fd7.  The sibling ``ddm_to1/generations/
#: to1_tail_override_r1`` carries byte-identical ``runtime/`` and ``cpr1/`` trees but
#: pins the older archive sha in its own ``inflate.py``, so it is not used here.
DEFAULT_RUNTIME_ROOT = Path("/Volumes/APDataStore/pact/ddm_up3/candidate_runtime")
#: The POINTER archive.  Byte-close splices into THIS.
DEFAULT_POINTER_ARCHIVE = DEFAULT_RUNTIME_ROOT / "archive.zip"
POINTER_ARCHIVE_SHA = "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"
POINTER_ARCHIVE_BYTES = 176_420
POINTER_MEMBER_BYTES = 176_320

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


def compile_rc64(work: Path, route_b) -> tuple[Path, dict[str, object]]:
    base = resolve_rc64_base(route_b, work)
    generated = work / "rc64_backend_jg2.c"
    library = work / "librc64_jg2.dylib"
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

    if resume and checkpoint_path.is_file() and encoder_state.is_file():
        blob = np.load(checkpoint_path, allow_pickle=False)
        start_frame = int(blob["frame"][0])
        code_bits = float(blob["code_bits"][0])
        per_frame = np.asarray(blob["per_frame"], dtype=np.float64).copy()
        previous_seed = np.asarray(blob["previous"], dtype=np.uint8).copy()
        corrector.load_state_dict(
            {k: blob[k] for k in blob.files if k not in {"frame", "code_bits", "per_frame", "previous"}}
        )
        encoder = route_b.NativeRc64Encoder(library, encoder_state.read_bytes())
        progress({"stage": f"encode_{tag}", "event": "resumed", "frame": start_frame})
    else:
        encoder = route_b.NativeRc64Encoder(library)

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

            if checkpoint_every and (frame + 1) % checkpoint_every == 0 and frame + 1 < frames:
                atomic_write(encoder_state, encoder.snapshot())
                state_dict = corrector.state_dict()
                np.savez(
                    work / f"encode_{tag}.checkpoint.npz.partial.npz",
                    frame=np.array([frame + 1], dtype=np.int64),
                    code_bits=np.array([code_bits], dtype=np.float64),
                    per_frame=per_frame,
                    previous=frame_tokens,
                    **{k: np.asarray(v) for k, v in state_dict.items()},
                )
                os.replace(
                    work / f"encode_{tag}.checkpoint.npz.partial.npz", checkpoint_path
                )
                progress(
                    {
                        "stage": f"encode_{tag}",
                        "event": "checkpoint",
                        "frame": frame + 1,
                        "code_bytes_so_far": code_bits / 8.0,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
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
    atomic_write(stream_path, body)
    ledger_path = work / f"bits_per_frame_{tag}.npy"
    np.save(ledger_path, per_frame)

    return {
        "tag": tag,
        "frames": frames,
        "code_bits": code_bits,
        "code_bytes_ideal": code_bits / 8.0,
        "stream": file_fact(stream_path),
        "bits_per_frame_ledger": file_fact(ledger_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _row_bits(rows: np.ndarray, symbols: np.ndarray) -> float:
    values = rows.astype(np.float64)
    picked = values[np.arange(values.shape[0]), symbols]
    return float(-np.log2(np.maximum(picked, 1e-300)).sum())


# --------------------------------------------------------------------------------------
# Stages.
# --------------------------------------------------------------------------------------


def _prepare(args) -> dict[str, Any]:
    work = Path(args.store) / "work"
    work.mkdir(parents=True, exist_ok=True)
    route_b = load_route_b()
    library, build = compile_rc64(work, route_b)
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
    env = _prepare(args)
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
    env = _prepare(args)
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
    # pointer's own bytes.  Verify that, rather than trusting the path.
    pointer_path = Path(args.pointer_archive)
    pointer_sha = sha256_file(pointer_path)
    if pointer_sha != POINTER_ARCHIVE_SHA:
        raise Jg2Error(
            f"--pointer-archive is not the frontier pointer: {pointer_path} has sha "
            f"{pointer_sha}, expected {POINTER_ARCHIVE_SHA}"
        )
    pointer_member = read_archive_member(pointer_path)
    if len(pointer_member) != POINTER_MEMBER_BYTES:
        raise Jg2Error(
            f"pointer member is {len(pointer_member)} B, expected {POINTER_MEMBER_BYTES}"
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
    parser.add_argument("--pointer-archive", default=str(DEFAULT_POINTER_ARCHIVE))
    parser.add_argument("--tokens", default=str(DEFAULT_TOKENS))
    parser.add_argument("--edits", default=None, help="npz of {pair: (384,512) uint8}")
    parser.add_argument("--tag", default="edited")
    parser.add_argument("--frames", type=int, default=N_PAIRS)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 1 or args.frames > N_PAIRS:
        raise SystemExit(f"--frames must be in 1..{N_PAIRS}")
    stage = {"control": stage_control, "encode": stage_encode}[args.stage]
    verdict = stage(args)
    progress({"stage": args.stage, "event": "done", **{
        k: v for k, v in verdict.items() if not isinstance(v, dict)
    }})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
