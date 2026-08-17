"""ddm_rr2 - byte-close the ddm_rr1 free decode-time corrector.

``ddm_rr1`` measured a 1,598 B code-length saving on the hv1 token stream from
a probability corrector that ships in ``inflate.py`` for free under rule 118,
and stated four times that the result was NOT byte-closed: no archive was
built, and the projected archive byte count went through a MEASURED constant
coder overhead rather than through a real stream.  This script builds the
stream, so the coder tax stops being assumed and starts being measured
(``ddm_rr1`` §5.5 says row 1 of the build order settles it by construction).

PRE-REGISTERED TARGET (``ddm_rr1`` §4)::

    archive.zip  == 181,161 B  (+/- 2 B)
    token stream == 110,512 B
    every other section byte-identical to 80d9c8c6...
    FALSIFIED IF the built archive exceeds 181,250 B, or the decoded token
    field is not bit-identical to sha 9ba2e52b...

WHAT THIS SCRIPT DOES.  Four stages, each fail-closed.

  ``control-repack``  Re-emit the frontier archive from its own parsed
      sections with no change at all.  It must reproduce sha 80d9c8c6...
      byte-identically.  This proves the repack layer is exact, so any later
      archive differs from the frontier by exactly the token-stream delta.

  ``control-encode``  Replay the shipped decode order, hand the RC64 encoder
      the receiver's OWN probability rows with no correction, and encode the
      already-decoded token field.  The output must be byte-identical to the
      shipped 112,110 B token stream.  This is the strongest available proof
      that the encoder is the exact inverse of the shipping decoder: same
      group order, same probability quantization, same coder, same flush.

  ``encode``  The same replay with the free corrector active.  Emits the new
      token stream and the exact measured code length.

  ``build``  Splice the new token stream into the member and repack.

WHERE THE PROBABILITIES COME FROM.  ``ddm_hm1`` retained the shipped decode's
pre-correction HPAC logits (``base_logits_int16_n600.i16``), and ``ddm_rr1``
proved they rebuild the receiver's ``corrected`` and CDF-input arrays to a
matching sha256.  Those logits are valid along the shipped decode trajectory,
and the trajectory is preserved here because the decoded field is unchanged by
construction - the corrector touches only the probability model.  The
``control-encode`` stage is what turns that argument into a measurement.

RESUMABILITY.  Every encode stage checkpoints at frame boundaries: the RC64
encoder's exact interval state via the snapshot ABI, plus the corrector's
statistics and per-pixel temporal memory.  ``--resume`` continues bit-faithfully.

AXIS.  ``[macOS-CPU advisory / scorer-free EXACT byte measurement]``.  This
script builds a real archive and stats it, so the rate leg is exact; there is
no distortion leg because the decoded field is bit-identical.  It is not an
exact-eval row and makes no score claim.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

# ddm_rr4: the corrector module is selectable so a platform-exact successor can
# be encoded WITHOUT mutating ddm_rr2's sealed proof chain.  The default is
# unchanged, so `--stage encode` with no environment override still reproduces
# ddm_rr2's byte targets exactly.  Set TAC_RR2_CORRECTOR_MODULE to a module name
# under experiments/ that exposes the same FreeCorrector surface.
CORRECTOR_MODULE = os.environ.get("TAC_RR2_CORRECTOR_MODULE", "ddm_rr2_free_corrector")
FreeCorrector = importlib.import_module(CORRECTOR_MODULE).FreeCorrector

PREPARED = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/prepared/hv1_base_control"
)
ARCHIVE = PREPARED / "archive.zip"
ARCHIVE_SHA = "80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e"
ARCHIVE_BYTES = 182759
MEMBER_SHA = "88828f3ded072886c8b4aa56c4215192e2d429fa1d80c173841fab6489052585"

HM1 = Path("/Volumes/APDataStore/pact/ddm_hm1_20260816/retained")
LOGITS = HM1 / "base_logits_int16_n600.i16"
BOUNDARY = HM1 / "boundary_bucket_n600.u8"
GROUP_INDEX = HM1 / "group_index.u8"
TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2/inflated/"
    ".f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
TOKEN_FIELD_SHA = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"

RC64_SOURCE = Path(
    "/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src/cpr1_sub4/entropy/rc64_backend.c"
)
RC64_SOURCE_SHA = "5c75e2c70b89f148bc9d117d4dbd39a24dfb2e72ec41b0a7e9b9cf490ca07ee6"
ROUTE_B = REPO / "experiments/ddm_rc64p_native_cpu_decode/route_b_rc64.py"

STORE = Path("/Volumes/APDataStore/pact/ddm_rr2_encoder_build")

N_FRAMES = 600
HEIGHT, WIDTH = 384, 512
PLANE = HEIGHT * WIDTH
NUM_CLASSES = 5
LOGIT_PRECISION = 8
SHIPPED_TOKEN_BYTES = 112110
RESIDUAL_COMPACT_BYTES = 96

SCORE_DENOMINATOR = 37_545_489
S_BASE = 0.15959729295498598
TARGET_ARCHIVE_BYTES = 181161
TARGET_TOKEN_BYTES = 110512
FALSIFY_ARCHIVE_BYTES = 181250

CHECKPOINT_EVERY = 25


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def progress(record: dict[str, object]) -> None:
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    with (STORE / "PROGRESS.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


# --- archive layout ---------------------------------------------------------


def split_member(member: bytes) -> tuple[bytes, bytes, bytes]:
    """Return (models prefix, 96 B compact RCF1 residual, RC64 token stream).

    The split point is exactly the one ``runtime.residual_archive`` parses:
    the tail is the fixed residual table's compact body followed by the token
    stream, and everything before it is the RX1M model container.
    """
    tail = RESIDUAL_COMPACT_BYTES + SHIPPED_TOKEN_BYTES
    if len(member) <= tail:
        raise SystemExit("member too short for the F24S tail")
    return member[: len(member) - tail], member[-tail : -SHIPPED_TOKEN_BYTES], member[-SHIPPED_TOKEN_BYTES:]


def pack_archive(member: bytes, destination: Path) -> None:
    """Rebuild the single-member STORED archive with the frontier's framing."""
    info = zipfile.ZipInfo(filename="p", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.partial")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr(info, member)
    os.replace(temporary, destination)


# --- RC64 encoder -----------------------------------------------------------


def load_route_b():
    spec = importlib.util.spec_from_file_location("ddm_rr2_route_b_rc64", ROUTE_B)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot import route_b_rc64")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_rc64(work: Path, route_b) -> tuple[Path, dict[str, object]]:
    """Compile the granted RC64 backend plus the snapshot/resume extension."""
    if sha256_file(RC64_SOURCE) != RC64_SOURCE_SHA:
        raise SystemExit("RC64 backend source sha mismatch - refusing")
    generated = work / "rc64_backend_checkpoint.c"
    library = work / "librc64_rr2.dylib"
    source = RC64_SOURCE.read_bytes() + ("\n" + route_b.RC64_CHECKPOINT_EXTENSION).encode()
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
    return library, {"argv": command, "source": file_fact(generated), "library": file_fact(library)}


# --- the replay -------------------------------------------------------------


def encode_stream(
    library: Path,
    route_b,
    table_values: np.ndarray,
    corrected: bool,
    work: Path,
    tag: str,
    resume: bool,
    frames: int,
) -> dict[str, object]:
    """Replay the shipped decode order and encode the already-decoded field.

    ``corrected=False`` hands the receiver's own probability rows straight to
    the coder, which must reproduce the shipped token stream byte-identically.
    ``corrected=True`` applies the free corrector.
    """
    logits_mm = np.memmap(LOGITS, dtype=np.int16, mode="r", shape=(N_FRAMES, PLANE, NUM_CLASSES))
    boundary_mm = np.memmap(BOUNDARY, dtype=np.uint8, mode="r", shape=(N_FRAMES, PLANE))
    tokens_mm = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N_FRAMES, PLANE))
    group_index = np.frombuffer(GROUP_INDEX.read_bytes(), dtype=np.uint8)
    n_groups = int(group_index.max()) + 1
    groups = [np.flatnonzero(group_index == g) for g in range(n_groups)]

    checkpoint_path = work / f"encode_{tag}.checkpoint.npz"
    encoder_state = work / f"encode_{tag}.encoder.bin"
    corrector = FreeCorrector(PLANE)
    start_frame = 0
    code_bits = 0.0
    per_frame = np.zeros(N_FRAMES, dtype=np.float64)

    if resume and checkpoint_path.is_file() and encoder_state.is_file():
        blob = np.load(checkpoint_path, allow_pickle=False)
        start_frame = int(blob["frame"][0])
        code_bits = float(blob["code_bits"][0])
        corrector.load_state_dict({key: blob[key] for key in blob.files})
        if "per_frame" in blob.files:
            per_frame = np.asarray(blob["per_frame"], dtype=np.float64).copy()
        encoder = route_b.NativeRc64Encoder(library, encoder_state.read_bytes())
        progress({"stage": f"encode_{tag}", "event": "resumed", "frame": start_frame})
    else:
        encoder = route_b.NativeRc64Encoder(library)

    started = time.perf_counter()
    for frame in range(start_frame, frames):
        base = np.asarray(logits_mm[frame], dtype=np.int16).astype(np.float32) / LOGIT_PRECISION
        boundary = np.asarray(boundary_mm[frame], dtype=np.int64)
        actual = np.asarray(tokens_mm[frame], dtype=np.int64)
        predicted_all = base.argmax(axis=1).astype(np.int64)
        table_row = table_values[boundary * NUM_CLASSES + predicted_all]
        corrected_all = base + table_row
        probability_all = _probability_table(corrected_all, LOGIT_PRECISION)
        corrector.begin_frame(boundary)

        frame_bits = 0.0
        for positions in groups:
            rows = probability_all[positions]
            symbols = actual[positions]
            state = None
            coding = rows
            if corrected:
                state = corrector.group_state(rows, predicted_all[positions], positions)
                coding = corrector.coding_row(state)
            _guard_rows(coding, frame)
            frame_bits += _row_bits(coding, symbols)
            encoder.encode(symbols.astype(np.int32), coding)
            if state is not None:
                corrector.observe(state, symbols)

        code_bits += frame_bits
        per_frame[frame] = frame_bits
        corrector.end_frame(actual.astype(np.uint8))

        if (frame + 1) % CHECKPOINT_EVERY == 0 and frame + 1 < frames:
            atomic_write(encoder_state, encoder.snapshot())
            state_dict = corrector.state_dict()
            np.savez(
                work / f"encode_{tag}.checkpoint.npz.partial.npz",
                frame=np.array([frame + 1], dtype=np.int64),
                code_bits=np.array([code_bits], dtype=np.float64),
                per_frame=per_frame,
                **{key: np.asarray(value) for key, value in state_dict.items()},
            )
            os.replace(work / f"encode_{tag}.checkpoint.npz.partial.npz", checkpoint_path)
            progress(
                {
                    "stage": f"encode_{tag}",
                    "event": "checkpoint",
                    "frame": frame + 1,
                    "code_bytes_so_far": code_bits / 8.0,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )

    # route_b frames its payload with a magic and a u32 pad for the lc2
    # container; F26 stores the raw coder output, so read the coder's own
    # buffer directly instead of un-framing (trailing zero bytes are legal
    # coder output and must not be stripped).
    payload = encoder.finish()
    if not payload.startswith(route_b.TOKEN_MAGIC):
        raise SystemExit("RC64 payload lost its magic")
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    if not size or not pointer:
        raise SystemExit("RC64 encoder produced no payload")
    body = ctypes.string_at(pointer, size)
    stream_path = work / f"token_stream_{tag}.bin"
    atomic_write(stream_path, body)

    # ALWAYS KEEP THE PAYLOAD: the learned statistics and the per-frame code
    # ledger are materialized here, so they are persisted here rather than
    # summarized into a scalar and discarded.
    retained = work.parent / "retained"
    retained.mkdir(parents=True, exist_ok=True)
    final_state = retained / f"corrector_final_state_{tag}.npz"
    np.savez(
        retained / f"corrector_final_state_{tag}.npz.partial.npz",
        frame=np.array([frames], dtype=np.int64),
        code_bits=np.array([code_bits], dtype=np.float64),
        per_frame=per_frame,
        **{key: np.asarray(value) for key, value in corrector.state_dict().items()},
    )
    os.replace(retained / f"corrector_final_state_{tag}.npz.partial.npz", final_state)
    ledger = retained / f"bits_per_frame_{tag}.npy"
    np.save(ledger, per_frame)
    return {
        "tag": tag,
        "corrected": corrected,
        "code_bytes": code_bits / 8.0,
        "stream": file_fact(stream_path),
        "final_corrector_state": file_fact(final_state),
        "bits_per_frame_ledger": file_fact(ledger),
        "warm_contexts": int((corrector.counts >= 32.0).sum()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _probability_table(logits: np.ndarray, precision: int) -> np.ndarray:
    """Byte-identical copy of runtime ``residual_archive._probability_table``."""
    quantized = np.clip(
        np.rint(np.asarray(logits, dtype=np.float32) * precision), -32768, 32767
    ).astype(np.int16)
    values = quantized.astype(np.float32) / precision
    values = values.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def _guard_rows(rows: np.ndarray, frame: int) -> None:
    """The RC64 backend refuses non-positive or non-normalized rows."""
    if not np.all(np.isfinite(rows)) or np.any(rows <= 0.0):
        raise SystemExit(f"frame {frame}: corrected row has a non-positive entry")
    sums = rows.astype(np.float64).sum(axis=1)
    if np.any(np.abs(sums - 1.0) > 2e-5):
        raise SystemExit(f"frame {frame}: corrected row does not sum to one")


def _row_bits(rows: np.ndarray, symbols: np.ndarray) -> float:
    values = rows.astype(np.float64)
    picked = values[np.arange(values.shape[0]), symbols]
    return float(-np.log2(np.maximum(picked, 1e-300)).sum())


# --- stages -----------------------------------------------------------------


def read_sections() -> tuple[bytes, bytes, bytes, bytes]:
    if sha256_file(ARCHIVE) != ARCHIVE_SHA:
        raise SystemExit("frontier archive sha mismatch - refusing")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if archive.namelist() != ["p"]:
            raise SystemExit("frontier archive is not a single member p")
        member = archive.read("p")
    if sha256_bytes(member) != MEMBER_SHA:
        raise SystemExit("frontier member sha mismatch - refusing")
    models, residual, token = split_member(member)
    return member, models, residual, token


def stage_control_repack(work: Path) -> dict[str, object]:
    member, models, residual, token = read_sections()
    rebuilt = models + residual + token
    if rebuilt != member:
        raise SystemExit("member split/join is not lossless")
    destination = work / "control_repack.zip"
    pack_archive(rebuilt, destination)
    fact = file_fact(destination)
    identical = fact["sha256"] == ARCHIVE_SHA and fact["bytes"] == ARCHIVE_BYTES
    result = {
        "stage": "control-repack",
        "sections": {
            "models_bytes": len(models),
            "residual_bytes": len(residual),
            "token_bytes": len(token),
            "token_sha256": sha256_bytes(token),
        },
        "rebuilt": fact,
        "frontier_sha256": ARCHIVE_SHA,
        "byte_identical_to_frontier": bool(identical),
    }
    if not identical:
        raise SystemExit("repack layer is not exact - refusing to continue")
    return result


def stage_encode(work: Path, corrected: bool, resume: bool, frames: int) -> dict[str, object]:
    _, _, _, token = read_sections()
    route_b = load_route_b()
    library, build = compile_rc64(work, route_b)
    sys.path.insert(0, str(PREPARED))
    sys.path.insert(0, str(PREPARED / "cpr1"))
    from runtime.residual_archive import read_residual_archive

    parts = read_residual_archive(ARCHIVE)
    if bytes(parts.token_stream) != token:
        raise SystemExit("parsed token stream disagrees with the member split")
    tag = ("corrected" if corrected else "bypass") + ("" if frames == N_FRAMES else f"_smoke{frames}")
    outcome = encode_stream(library, route_b, parts.table.values, corrected, work, tag, resume, frames)
    stream = Path(str(outcome["stream"]["path"])).read_bytes()
    result = {
        "stage": f"encode-{tag}",
        "rc64_build": build,
        "shipped_token_bytes": len(token),
        "shipped_token_sha256": sha256_bytes(token),
        "encoded": outcome,
        "byte_identical_to_shipped": bool(stream == token),
        "delta_bytes_vs_shipped": len(token) - len(stream),
        "measured_coder_overhead_bytes": len(stream) - outcome["code_bytes"],
    }
    result["frames"] = frames
    result["smoke_only_not_quotable"] = frames != N_FRAMES
    if frames != N_FRAMES:
        # A prefix cannot reproduce the full stream; it only proves the wiring.
        result["byte_identical_to_shipped"] = None
        result["prefix_of_shipped_stream"] = bool(token.startswith(stream[: max(len(stream) - 8, 0)]))
        return result
    if not corrected and not result["byte_identical_to_shipped"]:
        raise SystemExit(
            "bypass encode did not reproduce the shipped token stream - "
            "the encoder is not the inverse of the shipping decoder"
        )
    return result


def _parsed_section_shas(archive_path: Path) -> dict[str, str]:
    """Re-parse an archive with the receiver's own parser and hash each section.

    Comparing the built archive's PARSED sections against the frontier's is a
    real check; comparing the slices the builder just concatenated against
    themselves would be a tautology that passes no matter what was written.
    """
    sys.path.insert(0, str(PREPARED))
    sys.path.insert(0, str(PREPARED / "cpr1"))
    from runtime.residual_archive import read_residual_archive

    parts = read_residual_archive(archive_path)
    return {
        "semantic_blob": sha256_bytes(bytes(parts.semantic_blob)),
        "carrier_blob": sha256_bytes(bytes(parts.carrier_blob)),
        "hpac_blob": sha256_bytes(bytes(parts.hpac_blob)),
        "compensation_blob": sha256_bytes(bytes(parts.compensation_blob or b"")),
        "compressed_models": sha256_bytes(bytes(parts.compressed_models)),
        "residual_payload": sha256_bytes(bytes(parts.residual_payload)),
        "table_codes": sha256_bytes(np.ascontiguousarray(parts.table.codes).tobytes()),
        "token_stream": sha256_bytes(bytes(parts.token_stream)),
    }


def stage_build(work: Path, retained: Path) -> dict[str, object]:
    _member, models, residual, token = read_sections()
    stream_path = work / "token_stream_corrected.bin"
    if not stream_path.is_file():
        raise SystemExit("run the encode stage first")
    stream = stream_path.read_bytes()
    candidate_member = models + residual + stream
    destination = retained / "archive.zip"
    pack_archive(candidate_member, destination)
    repeat = work / "archive.repeat.zip"
    pack_archive(candidate_member, repeat)
    fact = file_fact(destination)
    saved = ARCHIVE_BYTES - int(fact["bytes"])
    delta_s = -saved * 25.0 / SCORE_DENOMINATOR
    atomic_write(retained / "token_stream.bin", stream)
    atomic_write(retained / "member.bin", candidate_member)
    frontier_sections = _parsed_section_shas(ARCHIVE)
    candidate_sections = _parsed_section_shas(destination)
    section_table = {
        name: {
            "frontier_sha256": frontier_sections[name],
            "candidate_sha256": candidate_sections[name],
            "byte_identical": frontier_sections[name] == candidate_sections[name],
        }
        for name in frontier_sections
    }
    unchanged = {name for name in section_table if name != "token_stream"}
    result = {
        "stage": "build",
        "archive": fact,
        "archive_repeat_sha256": sha256_file(repeat),
        "archive_repeat_byte_identical": sha256_file(repeat) == fact["sha256"],
        "member": {"bytes": len(candidate_member), "sha256": sha256_bytes(candidate_member)},
        "token_stream": {"bytes": len(stream), "sha256": sha256_bytes(stream)},
        "parsed_sections_vs_frontier": section_table,
        "every_other_section_byte_identical": all(
            section_table[name]["byte_identical"] for name in unchanged
        ),
        "token_stream_changed": not section_table["token_stream"]["byte_identical"],
        "bytes_saved_vs_frontier": saved,
        "delta_S": delta_s,
        "S_projected_rate_only": S_BASE + delta_s,
        "pre_registered_target": {
            "archive_bytes": TARGET_ARCHIVE_BYTES,
            "token_bytes": TARGET_TOKEN_BYTES,
            "archive_within_2B": abs(int(fact["bytes"]) - TARGET_ARCHIVE_BYTES) <= 2,
            "token_delta_vs_target": len(stream) - TARGET_TOKEN_BYTES,
        },
        "falsifier": {
            "archive_exceeds_181250": int(fact["bytes"]) > FALSIFY_ARCHIVE_BYTES,
            "decoded_field_target_sha256": TOKEN_FIELD_SHA,
            "decoded_field_verdict": "PENDING - run the receiver parse-back",
        },
    }
    return result


def main() -> int:
    # ddm_rr4 fix: this declaration used to sit AFTER `default=STORE` had already
    # read the name, which is a hard SyntaxError - the committed instrument did
    # not compile and so could not have produced ddm_rr2's artifacts.  Declaring
    # it first is semantically identical and makes the file executable again.
    global STORE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("control-repack", "control-encode", "encode", "build"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--frames",
        type=int,
        default=N_FRAMES,
        help="SMOKE ONLY below 600: a prefix cannot close any byte target",
    )
    parser.add_argument("--store", type=Path, default=STORE)
    args = parser.parse_args()

    STORE = args.store
    store = args.store
    work = store / "work"
    retained = store / "retained"
    work.mkdir(parents=True, exist_ok=True)
    retained.mkdir(parents=True, exist_ok=True)

    if args.stage == "control-repack":
        result = stage_control_repack(work)
    elif args.stage == "control-encode":
        result = stage_encode(work, corrected=False, resume=args.resume, frames=args.frames)
    elif args.stage == "encode":
        result = stage_encode(work, corrected=True, resume=args.resume, frames=args.frames)
    else:
        result = stage_build(work, retained)

    result["axis"] = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
    result["score_claim"] = False
    result["promotable"] = False
    result["numpy_version"] = np.__version__
    result["python_version"] = sys.version
    result["corrector_module"] = CORRECTOR_MODULE
    result["corrector_sha256"] = sha256_file(REPO / "experiments" / f"{CORRECTOR_MODULE}.py")
    suffix = "" if args.frames == N_FRAMES else f"_smoke{args.frames}"
    atomic_json(store / f"RESULT_{args.stage}{suffix}.json", result)
    progress({"stage": args.stage, "event": "complete", "result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
