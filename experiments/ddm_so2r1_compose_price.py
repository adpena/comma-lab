"""ddm_so2r1 -- the composition adapter SO2's rung specifies: a NEW prior AND a NEW field.

WHY AN ADAPTER EXISTS AT ALL.  Two rails already price this object and neither can price this
rung alone.  ``experiments/ddm_hpr1_shape_price.py`` changes the PRIOR on a FIXED field (its
``FIELD`` is a module constant).  ``experiments/ddm_sj1_rlc1_price.py`` changes the FIELD under
a FIXED prior (it feeds the live archive's own ``parts`` straight to the receiver loop).  SO2's
construction moves BOTH: the lifted field Z is coded under a prior retrained on Z.  So this
module composes hpr1's altered-parts input with sj1's known-symbol injection and owns nothing
else -- the packer, the RC3/CK2/Brotli container, the LaneMixer, the FreeCorrector, the
previous-plane state and the RC64 coder are all the shipped ones, loaded from the shipped tree.

WHY THE STOCK CLIs ARE NOT PATCHED.  ``ddm_sj1_rlc1_price`` binds its own source sha into every
encoder checkpoint receipt, so editing it would refuse every in-flight resume of a sister arm
([[binding_hash_whole_module_kills_checkpoints_20260909]]); its ``init`` also refuses a tree
whose archive differs from the live pointer, which is the guard that makes its control a real
falsifier.  Pointing it at a candidate tree would falsify that control rather than satisfy it.
This module therefore keeps its own INPUTS record with distinct source and candidate identities.

THE THREE CONTROLS THIS RUNG RESTS ON.
  1. ``pack`` re-runs the exact pack sequence on the ORIGINAL shipped IHS1 body and requires it
     to reproduce the live archive's own HPAC member byte-identically.  If it does not, no
     candidate member length is a price.
  2. ``ddm_sj1_rlc1_price.py encode --field control`` (run separately, in its own store) must
     reproduce move 49's archive byte-identically.  That is the loop control.
  3. ``ddm_sj1_rlc1_price.py encode --field lifted`` prices Z under the INCUMBENT prior, so the
     rung's own number can be split into "the representation" and "the refit".

Axis ``[macOS-CPU advisory; exact bytes, scorer-free]``; ``score_claim=false``; no scorer, no
renderer forward, no Modal, no pointer write.  A PASS here is a NEW OBJECT (the receiver would
have to learn the SO2L inverse), not a score.

Usage::

  python experiments/ddm_so2r1_compose_price.py pack --checkpoint <qat_stage_end_epoch_0060.pt>
  python experiments/ddm_so2r1_compose_price.py encode --tag primary
  python experiments/ddm_so2r1_compose_price.py decode --tag primary
  python experiments/ddm_so2r1_compose_price.py verdict
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "experiments")]
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_hpr1_shape_price import assert_layout_held
from experiments.ddm_rc1_model_section_adaptive_recode import ck2_interleave
from experiments.ddm_so2r1_lift import lift_inverse
from experiments.ddm_tc1_public_proof import build_libraries

#: Everything this rung writes lives under one owned root on APDataStore.  MAIN re-rooted the
#: rung here because Vertigo sits UNDER its own 40 GiB reserve today and that reserve is never
#: lowered to make a stage fit.
RIG = Path(os.environ.get("SO2R1_RIG", "/Volumes/APDataStore/pact/ddm_so2_first_rung"))
ROOT = RIG / "candidate"
LIVE = Path(os.environ.get(
    "SO2R1_LIVE", "/Volumes/VertigoDataTier/pact/ddm_sj1_pass7/candidate/candidate_runtime"))
POINTER = REPO / ".omx/state/canonical_frontier_pointer.json"
AXIS = "[macOS-CPU advisory; exact bytes, scorer-free]"
TAIL_PREFIX_BYTES = 96
RIDER_MAGIC = b"RLC1"
N_PAIRS, EVAL_H, EVAL_W = 600, 384, 512
#: SO2's declared research-container prefix: ``SO2L`` plus version/tile/stride/modulus.  It is
#: PHYSICALLY placed before the RX1M member, inside the archive's sole STORED member, so it is
#: charged in J.  Any further implementation side information would also have to be charged.
SO2_PREFIX = b"SO2L" + bytes([1, 64, 2, 5])
#: SO2's acceptance test: floor(0.95 x 130,525), the charter's 5% ceiling on the move-48
#: prior+stream subsystem.  SO2 corrected the ACCOUNTING to move 49's actual 130,567 B but
#: deliberately did NOT relax the gate, so the threshold stays where the charter put it.
PASS_CEILING_BYTES = 123_998
#: The conditional sub-0.12 subsystem ceiling at held distortion, for reporting only.
SUB012_CEILING_BYTES = 106_052
RESERVE_BYTES = 8 << 30
TRAIN_EPOCHS = 60


class ComposeError(RuntimeError):
    """A composition input or invariant is not what the shipped object says it is."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def preflight(path: Path, need: int = 1 << 20) -> None:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ComposeError(f"write outside the owned candidate root: {resolved}")
    mount = Path("/Volumes") / resolved.parts[2]
    if shutil.disk_usage(mount).free < need + RESERVE_BYTES:
        raise ComposeError(f"STORAGE_BLOCK on {mount}: need {need} B over an 8 GiB reserve")
    resolved.parent.mkdir(parents=True, exist_ok=True)


def write(path: Path, payload: bytes) -> dict:
    """Immutable publish: an existing payload must already be byte-identical."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ComposeError(f"immutable payload changed: {path}")
        return jg2.file_fact(path)
    preflight(path, len(payload))
    jg2.atomic_write(path, payload)
    return jg2.file_fact(path)


def record(path: Path, value: object) -> object:
    preflight(path)
    jg2.atomic_json(path, value)
    return value


def train_inputs() -> dict:
    manifest = json.loads((RIG / "TRAIN_INPUTS.json").read_text())
    for key in ("Z", "cache", "source_field"):
        if key not in manifest:
            raise ComposeError(f"TRAIN_INPUTS.json is missing {key}")
    z_path = Path(manifest["Z"]["path"])
    if jg2.file_fact(z_path) != manifest["Z"]:
        raise ComposeError("the lifted field Z changed under this arm")
    # The trainer pins its cache by FILE sha, so the chain cache.pt -> Z content must be held
    # here or a checkpoint could pass its own identity check while having been trained on a
    # different tensor inside an identically-named file.
    if jg2.file_fact(Path(manifest["cache"]["path"])) != manifest["cache"]:
        raise ComposeError("the trainer cache changed under this arm")
    if manifest["cache_file_sha256"] != manifest["cache"]["sha256"]:
        raise ComposeError("TRAIN_INPUTS.json disagrees with itself about the cache file sha")
    if manifest["inverse_receipt"]["mismatched_bytes"] != 0:
        raise ComposeError("TRAIN_INPUTS.json does not carry a passing inverse receipt")
    return manifest


def runtime_copy() -> Path:
    """Copy the live pointer tree into this store once, and prove the copy is that tree."""
    destination = ROOT / "runtime_copy"
    live_sha = jg2.sha256_file(LIVE / "archive.zip")
    pointer = json.loads(POINTER.read_text())["our_local_frontier_contest_cuda"]
    if pointer["archive_sha256"] != live_sha:
        raise ComposeError("POINTER_MOVED: the configured tree is not the live pointer")
    for src in sorted(LIVE.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts or src.suffix == ".pyc" \
                or src.name.startswith("._"):
            continue
        write(destination / src.relative_to(LIVE), src.read_bytes())
    if jg2.sha256_file(destination / "archive.zip") != live_sha:
        raise ComposeError("the copied archive is not the live archive")
    return destination


def build_geometry(runtime: Path, work: Path) -> dict:
    """Compile the RLC1 geometry library exactly as ``inflate.sh`` does."""
    target = work / "rlc1_geometry.so"
    source = runtime / "runtime/rlc1_geometry.c"
    command = [os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
               str(source), "-o", str(target)]
    work.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        (work / "rlc1_geometry.build.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
    os.environ["RLC1_GEOMETRY_LIBRARY"] = str(target)
    return {"source": jg2.file_fact(source), "library": jg2.file_fact(target), "argv": command}


def split_tail(tail: bytes, stream_bytes: int) -> tuple[bytes, bytes, bytes]:
    prefix = tail[:TAIL_PREFIX_BYTES]
    body = tail[TAIL_PREFIX_BYTES:]
    rider, stream = body[:len(body) - stream_bytes], body[len(body) - stream_bytes:]
    if rider[:len(RIDER_MAGIC)] != RIDER_MAGIC:
        raise ComposeError(f"the shipped rider magic is {rider[:4]!r}, not {RIDER_MAGIC!r}")
    return prefix, rider, stream


# ----------------------------------------------------------------------------------
# stage 1 -- pack the terminal EMA through the REAL current rail
# ----------------------------------------------------------------------------------

def pack_container(body: bytes, counts: list, header, weights, learning, label: str,
                   work: Path) -> tuple[bytes, bytes, dict]:
    """HPR1's `prepare` body-packing sequence: RC3 encode -> CK2 interleave -> Brotli q10/22.

    Returns ``(rider, compressed_member, receipt)``.  Both twins are produced and compared,
    and the shipping RC3 parser must return the packed body, so a member length is only
    reported for bytes the receiver can actually read back.
    """
    import brotli
    from runtime import rc3_shared_mixer as rc3

    riders, members, facts = [], [], {}
    for twin in range(2):
        rider, details = rc3.encode(body, counts, header[2], weights, learning)
        facts[f"{label}.twin{twin}.rider"] = write(work / f"{label}.twin{twin}.rider", rider)
        facts[f"{label}.twin{twin}.range"] = write(work / f"{label}.twin{twin}.range", details["payload"])
        facts[f"{label}.twin{twin}.params"] = write(work / f"{label}.twin{twin}.params", details["parameters"])
        outer = ck2_interleave(rider)
        facts[f"{label}.twin{twin}.ck2"] = write(work / f"{label}.twin{twin}.ck2", outer)
        encoded = brotli.compress(outer, quality=10, lgwin=22)
        facts[f"{label}.twin{twin}.br"] = write(work / f"{label}.twin{twin}.br", encoded)
        if rc3.restore_hpac(rider, counts) != body:
            raise ComposeError(f"{label}: the shipping RC3 parser disagrees with the packed body")
        riders.append(rider)
        members.append(encoded)
    if riders[0] != riders[1] or members[0] != members[1]:
        raise ComposeError(f"{label}: HPAC twin pack differs")
    return riders[0], members[0], facts


#: The preregistered law this rung's checkpoint must have been trained under, read off the
#: trainer's own ``run_identity.training_config`` keys rather than pattern-matched in prose.
EXPECTED_TRAINING_CONFIG = {
    "profile": "cl2_shipped_ladder",
    "epochs": TRAIN_EPOCHS,
    "rate_lambda": 1.0,
    "qat_fraction": 0.5,
    "seed": 20260716,
    "channels": 64,
    "patch": 64,
    "delta": 2,
    "frame_dim": 8,
    "past_dilation": 1,
    "conv_a_dilation": 1,
    "target_mode": "raw",
    "device": "mps",
}


def verify_checkpoint(checkpoint_path: Path, manifest: dict) -> dict:
    """The epoch-60 terminal EMA, and no other checkpoint, with this rung's own inputs.

    NOTE ON THE CACHE PIN.  The trainer records ``run_identity.cache_sha256`` as the FILE sha of
    ``--cache``, not the sha of the uint8 tensor inside it; the CONTENT sha is enforced
    separately by its own ``--expected-cache-content-sha256`` gate at launch.  So the chain that
    binds this checkpoint to Z is: checkpoint -> cache FILE sha -> (``train_inputs`` re-hashes
    that file) -> manifest content sha -> Z.u8.  Matching the content sha against the run
    identity directly would silently never fire, which is the shape of a gate that looks green
    because it is vacuous.
    """
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = payload.get("run_identity") or {}
    config = identity.get("training_config") or {}
    if payload.get("epoch") != TRAIN_EPOCHS:
        raise ComposeError(f"checkpoint is epoch {payload.get('epoch')}, not the preregistered {TRAIN_EPOCHS}")
    if payload.get("deployment_weights") != "ema_shadow":
        raise ComposeError("checkpoint's state_dict is not the EMA shadow")
    if payload.get("phase") != "discrete_qat":
        raise ComposeError(f"checkpoint phase is {payload.get('phase')}, not the terminal QAT stage")
    differences = {key: {"expected": value, "observed": config.get(key)}
                   for key, value in EXPECTED_TRAINING_CONFIG.items() if config.get(key) != value}
    if differences:
        raise ComposeError(f"checkpoint was not trained under the preregistered law: {differences}")
    cache_file_sha = manifest["cache_file_sha256"]
    init_sha = manifest["init_depths"]["sha256"]
    if identity.get("cache_sha256") != cache_file_sha:
        raise ComposeError(
            f"checkpoint's cache file sha is {identity.get('cache_sha256')}, not this rung's {cache_file_sha}")
    if Path(identity.get("cache_path", "")).resolve() != Path(manifest["cache"]["path"]).resolve():
        raise ComposeError(f"checkpoint's cache path is {identity.get('cache_path')}, not this rung's cache")
    if identity.get("init_sha256") != init_sha:
        raise ComposeError(
            f"checkpoint's initializer sha is {identity.get('init_sha256')}, not the pinned {init_sha}")
    return {
        "file": jg2.file_fact(checkpoint_path),
        "epoch": payload.get("epoch"),
        "phase": payload.get("phase"),
        "deployment_weights": payload.get("deployment_weights"),
        "causal_state_sha256": payload.get("causal_state_sha256"),
        "run_identity_sha256": payload.get("run_identity_sha256"),
        "cache_path": identity.get("cache_path"),
        "cache_file_sha256_bound": cache_file_sha,
        "cache_content_sha256_bound": manifest["expected_cache_content_sha256"],
        "init_sha256_bound": init_sha,
        "training_config": config,
        "terminal_history_row": history[-1] if (history := [
            row for row in (payload.get("history") or []) if isinstance(row, dict)]) else None,
    }


def pack(checkpoint_path: Path) -> dict:
    import torch

    from experiments import ddm_rx2_mc36_identity_race as rx2

    manifest = train_inputs()
    work = ROOT / "pack"
    preflight(work / "model")
    checkpoint = verify_checkpoint(checkpoint_path, manifest)

    runtime = runtime_copy()
    rx, renderer, _code_dir = jg2.load_runtime(runtime)
    # ``runtime`` only becomes importable once load_runtime has put the runtime COPY on the
    # path, so every import out of it belongs after this line, not in the function header.
    from runtime import ihs2

    parts = rx.read_residual_archive(runtime / "archive.zip")
    member = jg2.split_member(jg2.read_archive_member(runtime / "archive.zip"))
    layout = ihs2.layout_from_runtime(renderer)
    counts = list(layout.row_counts)
    shipped_body = rx.materialize_ihs1(parts.hpac_blob, renderer)

    packed = rx2._pack_terminal_ihs1(checkpoint_path, work / "model")
    body = Path(packed["raw"]["path"]).read_bytes()
    if body == shipped_body:
        raise ComposeError("the retrained prior packed to the shipped body; that is a no-op, not a rung")
    layout_held = assert_layout_held(body, shipped_body, counts)
    if renderer.load_hpac(body, torch.device("cpu")) is None:
        raise ComposeError("the real integer model failed to load from the candidate body")
    write(work / "hpac.candidate.ihs1", body)
    write(work / "hpac.shipped.ihs1", shipped_body)

    from runtime import rc3_shared_mixer as rc3

    header = rc3.HEADER.unpack_from(parts.hpac_blob)
    offset = rc3.HEADER.size + header[-3]
    learning = parts.hpac_blob[offset]
    weights = np.frombuffer(parts.hpac_blob[offset + 1: offset + 25], dtype=np.int8)

    rider, candidate_member, candidate_facts = pack_container(
        body, counts, header, weights, learning, "candidate", work)
    # MEMBER-IDENTITY CONTROL: the same sequence on the ORIGINAL body must reproduce the live
    # archive's own HPAC member exactly.  Without it, a candidate member length would only be
    # "what this code produces", not "what the shipped container charges".
    shipped_rider, shipped_member, shipped_facts = pack_container(
        shipped_body, counts, header, weights, learning, "shipped", work)
    if shipped_member != member["hpac"]:
        raise ComposeError(
            "HPAC_MEMBER_IDENTITY_CONTROL_FAILED: repacking the shipped body gives "
            f"{len(shipped_member)} B sha {sha256_bytes(shipped_member)}, not the archive's "
            f"{len(member['hpac'])} B sha {sha256_bytes(member['hpac'])}")
    if shipped_rider != bytes(parts.hpac_blob):
        raise ComposeError("the repacked shipped rider is not the archive's own hpac blob")

    result = dict(
        schema="ddm_so2r1_pack.v1", axis=AXIS, score_claim=False, promotion_eligible=False,
        checkpoint=checkpoint, packer=packed, layout_held=layout_held,
        candidate=dict(
            ihs1_bytes=len(body), ihs1_sha256=sha256_bytes(body),
            rider_bytes=len(rider), rider_sha256=sha256_bytes(rider),
            member_bytes=len(candidate_member), member_sha256=sha256_bytes(candidate_member),
            artifacts=candidate_facts),
        shipped=dict(
            ihs1_bytes=len(shipped_body), ihs1_sha256=sha256_bytes(shipped_body),
            member_bytes=len(shipped_member), member_sha256=sha256_bytes(shipped_member),
            artifacts=shipped_facts),
        member_identity_control_passed=True,
        archive_hpac_member_bytes=len(member["hpac"]),
        delta_member_bytes=len(candidate_member) - len(member["hpac"]),
        live_archive=jg2.file_fact(runtime / "archive.zip"),
        train_inputs_sha256=jg2.sha256_file(RIG / "TRAIN_INPUTS.json"),
        producer=jg2.file_fact(Path(__file__)),
        lift_producer=manifest["producer"],
    )
    return record(work / "PACK.json", result)


# ----------------------------------------------------------------------------------
# stage 2 -- the RLC1 known-symbol loop with BOTH the new prior and the new field
# ----------------------------------------------------------------------------------

def encode(tag: str) -> dict:
    from dataclasses import replace

    import torch

    manifest = train_inputs()
    pack_result = json.loads((ROOT / "pack/PACK.json").read_text())
    if not pack_result.get("member_identity_control_passed"):
        raise ComposeError("no candidate encode before the HPAC member-identity control passes")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260913)
    np.random.seed(20260913)
    torch.use_deterministic_algorithms(True)

    work = ROOT / "encode" / tag
    preflight(work / "build")
    runtime = runtime_copy()
    route = jg2.load_route_b()
    build_path = work / "ENCODER_BUILD.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        library = Path(build["library"]["path"])
    else:
        library, build = jg2.compile_rc64(work, route, f"so2r1_{tag}")
        record(build_path, build)
    build_libraries(runtime, work / "native")
    geometry = build_geometry(runtime, work / "native")

    rx, renderer, code_dir = jg2.load_runtime(runtime)
    parts = rx.read_residual_archive(runtime / "archive.zip")
    member = jg2.split_member(jg2.read_archive_member(runtime / "archive.zip"))
    _prefix, shipped_rider, shipped_stream = split_tail(member["tail"], len(bytes(parts.token_stream)))

    candidate_rider = Path(pack_result["candidate"]["artifacts"]["candidate.twin0.rider"]["path"]).read_bytes()
    candidate_member = Path(pack_result["candidate"]["artifacts"]["candidate.twin0.br"]["path"]).read_bytes()
    if sha256_bytes(candidate_rider) != pack_result["candidate"]["rider_sha256"]:
        raise ComposeError("the candidate rider on disk is not the packed one")
    # HPR1's altered-parts input: the receiver reads its prior out of ``hpac_blob``, so swapping
    # that field is exactly the prior swap, and everything else in ``parts`` stays move 49's.
    altered = replace(parts, hpac_blob=candidate_rider)

    checkpoints = work / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    os.environ["TC1_RECEIVER_CHECKPOINT_DIR"] = str(checkpoints)
    os.environ["TC1_RECEIVER_STOP_AFTER"] = str(N_PAIRS)
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.rlc1_mixer import LaneMixer
    from runtime.tc1_receiver_checkpoint import ReceiverCheckpoint

    binding = dict(
        train_inputs_sha=jg2.sha256_file(RIG / "TRAIN_INPUTS.json"),
        pack_sha=jg2.sha256_file(ROOT / "pack/PACK.json"),
        source_sha=jg2.sha256_file(Path(__file__)),
        lift_sha=manifest["producer"]["sha256"],
        jg2_sha=jg2.sha256_file(Path(jg2.__file__)),
        field=manifest["Z"]["sha256"],
        candidate_hpac_rider_sha=pack_result["candidate"]["rider_sha256"],
        candidate_hpac_member_sha=pack_result["candidate"]["member_sha256"],
        live_archive_sha=pack_result["live_archive"]["sha256"],
        build=build,
    )

    field = np.memmap(manifest["Z"]["path"], dtype=np.uint8, mode="r",
                      shape=(N_PAIRS, EVAL_H, EVAL_W))
    per_frame_bits = np.zeros(N_PAIRS, dtype=np.float64)
    state, start = None, 0
    latest = checkpoints / "LATEST.json"
    if latest.exists():
        start = int(json.loads(latest.read_text())["frame"])
        state_path = ROOT / "encoder_states" / tag / f"stage_{start:04d}.npz"
        receipt = json.loads(state_path.with_suffix(".json").read_text())
        if jg2.file_fact(state_path) != receipt["payload"] or receipt["binding"] != binding:
            raise ComposeError("encoder restart binding mismatch")
        with np.load(state_path, allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        per_frame_bits = state["per_frame_bits"].copy()

    twins = [route.NativeRc64Encoder(library, None if state is None else state[f"enc{i}"].tobytes())
             for i in range(2)]
    observed = {"frame": start, "positions": None}
    started = time.monotonic()
    original_coding, original_end = LaneMixer.coding, LaneMixer.end_frame
    original_save = ReceiverCheckpoint.save

    def coding(self, rows, positions, plane, previous):
        if observed["positions"] is not None or self.frame != observed["frame"]:
            raise ComposeError("known-symbol group order mismatch")
        observed["positions"] = positions.copy()
        return original_coding(self, rows, positions, plane, previous)

    def known_symbols(self, probabilities):
        positions = observed["positions"]
        if positions is None or len(positions) != len(probabilities):
            raise ComposeError("known-symbol decode call lacks its group")
        symbols = field[observed["frame"]].reshape(-1)[positions].astype(np.int32)
        rows = np.asarray(probabilities, dtype=np.float64)
        coded = np.clip(rows[np.arange(len(symbols)), symbols], 1e-12, 1.0)
        per_frame_bits[observed["frame"]] += float(-np.log2(coded).sum())
        for encoder in twins:
            encoder.encode(symbols, probabilities)
        observed["positions"] = None
        return symbols

    def end(self, plane, previous):
        # OUTPUT-LOSSLESS on Z: the plane the receiver reconstructs under the CANDIDATE prior is
        # the lifted field, frame by frame.  Equality on F is then the inverse's job, proved
        # separately by the `decode` stage rather than assumed here.
        np.testing.assert_array_equal(plane, field[observed["frame"]])
        original_end(self, plane, previous)
        observed["frame"] += 1

    def save(self, frame, tokens):
        if frame != observed["frame"] or observed["positions"] is not None:
            raise ComposeError("encoder checkpoint is not at a frame boundary")
        path = ROOT / "encoder_states" / tag / f"stage_{frame:04d}.npz"
        values = {f"enc{i}": np.frombuffer(e.snapshot(), dtype=np.uint8) for i, e in enumerate(twins)}
        values["per_frame_bits"] = per_frame_bits.copy()
        preflight(path, sum(v.nbytes for v in values.values()) + (1 << 20))
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(values) or any(
                        not np.array_equal(old[k], v) for k, v in values.items()):
                    raise ComposeError("immutable checkpoint differs")
        else:
            temporary = path.with_suffix(".npz.new")
            with temporary.open("wb") as handle:
                np.savez(handle, **values)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        record(path.with_suffix(".json"), {"payload": jg2.file_fact(path),
                                           "binding": binding, "frame": frame})
        (checkpoints / "LATEST.json.new").write_text(json.dumps({"frame": frame}))
        os.replace(checkpoints / "LATEST.json.new", latest)
        print(json.dumps(dict(tag=tag, frame=frame,
                              seconds=round(time.monotonic() - started, 1))), flush=True)
        original_save(self, frame, tokens)

    LaneMixer.coding, LaneMixer.end_frame = coding, end
    NativeDecoder.decode, ReceiverCheckpoint.save = known_symbols, save
    try:
        tokens, _ = rx.decode_production_tokens(altered, renderer, code_dir, torch.device("cpu"))
    finally:
        LaneMixer.coding, LaneMixer.end_frame = original_coding, original_end
        ReceiverCheckpoint.save = original_save
    decoded_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if observed["frame"] != N_PAIRS or decoded_sha != manifest["Z"]["sha256"]:
        raise ComposeError("the known-symbol loop did not reproduce Z over 600 frames")

    outputs, archives, research = {}, [], []
    for index, encoder in enumerate(twins):
        outputs[f"envelope{index}"] = write(work / f"envelope_{index}.bin", encoder.finish())
        raw = ctypes.string_at(encoder.library.rc64_encoder_data(encoder.context),
                               int(encoder.library.rc64_encoder_size(encoder.context)))
        outputs[f"stream{index}"] = write(work / f"stream_{index}.rc64", raw)
        changed = dict(member)
        changed["hpac"] = candidate_member
        # The header carries the three section LENGTHS, so a prior whose packed member changed
        # size must move field 5 or the parser would slice the sections apart in the wrong
        # places.  The struct is asserted against the shipped header and the shipped length
        # first: "index 5 is hpac" is a claim about the receiver's own struct, not a constant.
        if rx.RX1_MODEL_HEADER.size != len(member["header"]):
            raise ComposeError("the receiver's RX1M header struct is not the shipped header's size")
        fields = list(rx.RX1_MODEL_HEADER.unpack(changed["header"]))
        if fields[5] != len(member["hpac"]):
            raise ComposeError("RX1M header field 5 is not the shipped HPAC member length")
        fields[5] = len(candidate_member)
        changed["header"] = rx.RX1_MODEL_HEADER.pack(*fields)
        changed["tail"] = member["tail"][:TAIL_PREFIX_BYTES] + shipped_rider + raw
        rx1m = jg2.join_member(changed)
        write(work / f"member_{index}.bin", rx1m)
        inner = work / f"archive_rx1m_{index}.zip"
        if not inner.exists():
            preflight(inner, len(rx1m) + 4096)
            jg2.pack_archive(rx1m, inner)
        parsed = rx.read_residual_archive(inner)
        if bytes(parsed.token_stream) != raw:
            raise ComposeError("the packed archive does not parse back to the emitted stream")
        if bytes(parsed.hpac_blob) != candidate_rider:
            raise ComposeError("the packed archive does not parse back to the candidate prior")
        for name in ("semantic_blob", "carrier_blob", "tc1_weights", "residual_payload"):
            if getattr(parts, name) != getattr(parsed, name):
                raise ComposeError(f"unrelated archive component changed: {name}")
        archives.append(jg2.file_fact(inner))
        # SO2's research container: the declared 8 B format prefix physically precedes the RX1M
        # member inside the single STORED member, so the 8 B are charged in J, not assumed free.
        outer = ROOT / "encode" / tag / f"archive_research_{index}.zip"
        if not outer.exists():
            preflight(outer, len(rx1m) + 4096)
            jg2.pack_archive(SO2_PREFIX + rx1m, outer)
        research.append(jg2.file_fact(outer))
    if archives[0]["sha256"] != archives[1]["sha256"] or research[0]["sha256"] != research[1]["sha256"]:
        raise ComposeError("TWIN FAILED: the two in-process encoders disagree")

    stream_bytes = len((work / "stream_0.rc64").read_bytes())
    prior_bytes = len(candidate_member)
    j_z = prior_bytes + stream_bytes + len(SO2_PREFIX)
    result = dict(
        schema="ddm_so2r1_encode.v1", axis=AXIS, score_claim=False, promotion_eligible=False,
        tag=tag, frames=N_PAIRS, start_frame=start, geometry_build=geometry,
        outputs=outputs, rx1m_archives=archives, research_archives=research,
        decoded_field_sha256=decoded_sha, output_lossless_on_Z=True,
        packed_prior_bytes=prior_bytes, stream_bytes=stream_bytes,
        format_prefix_bytes=len(SO2_PREFIX), J_Z=j_z,
        shipped_stream_bytes=len(shipped_stream),
        shipped_prior_bytes=len(member["hpac"]),
        per_frame_bits=per_frame_bits.tolist(),
        ideal_bytes=float(per_frame_bits.sum() / 8.0),
        binding=binding,
    )
    record(work / "ENCODE.json", result)
    return {k: v for k, v in result.items() if k != "per_frame_bits"}


# ----------------------------------------------------------------------------------
# stage 3 -- independent decode: no injection, no encoder caches, then the SO2L inverse
# ----------------------------------------------------------------------------------

def decode(tag: str) -> dict:
    import torch

    manifest = train_inputs()
    encode_result = json.loads((ROOT / "encode" / tag / "ENCODE.json").read_text())
    work = ROOT / "decode" / tag
    preflight(work / "native")
    runtime = runtime_copy()
    build_libraries(runtime, work / "native")
    geometry = build_geometry(runtime, work / "native")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    research_path = Path(encode_result["research_archives"][0]["path"])
    research_member = jg2.read_archive_member(research_path)
    if research_member[:len(SO2_PREFIX)] != SO2_PREFIX:
        raise ComposeError("the research member does not carry the declared SO2L prefix")
    rx1m = research_member[len(SO2_PREFIX):]
    inner = work / "archive_rx1m_from_research.zip"
    if not inner.exists():
        preflight(inner, len(rx1m) + 4096)
        jg2.pack_archive(rx1m, inner)

    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    os.environ["TC1_RECEIVER_CHECKPOINT_DIR"] = str(work / "checkpoints")
    os.environ["TC1_RECEIVER_STOP_AFTER"] = str(N_PAIRS)
    rx, renderer, code_dir = jg2.load_runtime(runtime)
    # "No known-symbol injection" is a CLAIM unless it is checked: assert the arithmetic decoder
    # this process will run is the runtime's own function and not one of this module's closures.
    from runtime.entropy.rc64 import NativeDecoder

    if NativeDecoder.decode.__module__ != "runtime.entropy.rc64":
        raise ComposeError(
            f"the decoder is patched to {NativeDecoder.decode.__module__}; this is not an independent decode")
    parts = rx.read_residual_archive(inner)
    if bytes(parts.hpac_blob) != Path(
            json.loads((ROOT / "pack/PACK.json").read_text())["candidate"]["artifacts"][
                "candidate.twin0.rider"]["path"]).read_bytes():
        raise ComposeError("the research archive does not carry the candidate prior")
    tokens, _ = rx.decode_production_tokens(parts, renderer, code_dir, torch.device("cpu"))
    decoded = tokens.numpy()
    if decoded.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise ComposeError(f"decoded token geometry is wrong: {decoded.shape}")
    decoded_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
    if decoded_sha != manifest["Z"]["sha256"]:
        raise ComposeError(f"independent decode returned {decoded_sha}, not Z")
    write(work / "decoded_Z.u8", decoded.tobytes())

    rebuilt = lift_inverse(np.ascontiguousarray(decoded.astype(np.uint8)))
    rebuilt_sha = hashlib.sha256(rebuilt.tobytes()).hexdigest()
    source_sha = manifest["source_field"]["u8"]["sha256"]
    if rebuilt_sha != source_sha:
        raise ComposeError(f"the SO2L inverse of the decoded Z is {rebuilt_sha}, not move 49's field")
    write(work / "decoded_F.u8", rebuilt.tobytes())

    result = dict(
        schema="ddm_so2r1_decode.v1", axis=AXIS, score_claim=False, promotion_eligible=False,
        tag=tag, geometry_build=geometry, research_archive=encode_result["research_archives"][0],
        rx1m_archive=jg2.file_fact(inner),
        decoded_Z_sha256=decoded_sha, expected_Z_sha256=manifest["Z"]["sha256"],
        inverse_F_sha256=rebuilt_sha, expected_F_sha256=source_sha,
        independent_process=True, known_symbol_injection=False,
        bytes_reconstructed=int(rebuilt.size),
    )
    return record(work / "DECODE.json", result)


# ----------------------------------------------------------------------------------
# stage 4 -- the verdict, with every control on the same page
# ----------------------------------------------------------------------------------

def _sj1_encode(root: Path, field: str, tag: str) -> dict:
    path = root / "encode" / field / tag / "ENCODE.json"
    if not path.exists():
        raise ComposeError(f"missing sj1 encode receipt: {path}")
    return json.loads(path.read_text())


def verdict(tags: tuple[str, ...]) -> dict:
    manifest = train_inputs()
    pack_result = json.loads((ROOT / "pack/PACK.json").read_text())
    runs = [json.loads((ROOT / "encode" / tag / "ENCODE.json").read_text()) for tag in tags]
    shas = {run["rx1m_archives"][0]["sha256"] for run in runs}
    if len(shas) != 1:
        raise ComposeError("independent encode PROCESSES disagree; J would be variance, not a price")
    decodes = [json.loads((ROOT / "decode" / tag / "DECODE.json").read_text())
               for tag in tags if (ROOT / "decode" / tag / "DECODE.json").exists()]
    if not decodes:
        raise ComposeError("no independent decode receipt; a price without one is not lossless")

    control = [_sj1_encode(RIG / "control", "control", tag) for tag in tags]
    if not all(run["archive_matches_live_pointer"] for run in control):
        raise ComposeError("CONTROL IDENTITY FAILED: the loop does not reproduce move 49's archive")
    lifted = [_sj1_encode(RIG / "control_lifted", "lifted", tag) for tag in tags]
    lifted_shas = {run["archives"][0]["sha256"] for run in lifted}
    if len(lifted_shas) != 1:
        raise ComposeError("the incumbent-prior-on-Z control's processes disagree")

    live = json.loads(POINTER.read_text())["our_local_frontier_contest_cuda"]
    incumbent_prior = pack_result["archive_hpac_member_bytes"]
    incumbent_stream = runs[0]["shipped_stream_bytes"]
    incumbent_subsystem = incumbent_prior + incumbent_stream
    lifted_stream = lifted[0]["stream_bytes"]
    j_z = runs[0]["J_Z"]
    result = dict(
        schema="ddm_so2r1_verdict.v1", axis=AXIS, score_claim=False, promotion_eligible=False,
        pointer=dict(score=live.get("score"), archive_sha256=live["archive_sha256"],
                     archive_bytes=pack_result["live_archive"]["bytes"]),
        thresholds=dict(pass_ceiling_bytes=PASS_CEILING_BYTES,
                        sub012_subsystem_ceiling_bytes=SUB012_CEILING_BYTES),
        incumbent=dict(prior_bytes=incumbent_prior, stream_bytes=incumbent_stream,
                       subsystem_bytes=incumbent_subsystem),
        candidate=dict(prior_bytes=runs[0]["packed_prior_bytes"],
                       stream_bytes=runs[0]["stream_bytes"],
                       format_prefix_bytes=runs[0]["format_prefix_bytes"],
                       J_Z=j_z,
                       rx1m_archive_bytes=runs[0]["rx1m_archives"][0]["bytes"],
                       research_archive_bytes=runs[0]["research_archives"][0]["bytes"]),
        control_incumbent_prior_on_Z=dict(
            prior_bytes=incumbent_prior, stream_bytes=lifted_stream,
            J=incumbent_prior + lifted_stream + len(SO2_PREFIX),
            archive_bytes=lifted[0]["archives"][0]["bytes"],
            ideal_bytes=lifted[0]["ideal_bytes"]),
        control_loop_identity=dict(
            archive_sha256=control[0]["archives"][0]["sha256"],
            archive_bytes=control[0]["archives"][0]["bytes"],
            matches_live_pointer=True,
            stream_bytes=control[0]["stream_bytes"],
            ideal_bytes=control[0]["ideal_bytes"]),
        cross_entropy_telemetry=dict(
            candidate_ideal_bytes=runs[0]["ideal_bytes"],
            candidate_real_stream_bytes=runs[0]["stream_bytes"],
            real_minus_ideal_bytes=runs[0]["stream_bytes"] - runs[0]["ideal_bytes"],
            note=("the trained prior's own cross-entropy on Z, in bytes, against the byte count "
                  "the shipped integer-CDF coder actually emitted for the same symbols")),
        inverse=dict(
            receipts=[d["inverse_F_sha256"] for d in decodes],
            expected_F_sha256=manifest["source_field"]["u8"]["sha256"],
            passed=all(d["inverse_F_sha256"] == manifest["source_field"]["u8"]["sha256"] for d in decodes)),
        twins=dict(rx1m=sorted(shas) * len(runs), research=[r["research_archives"][0]["sha256"] for r in runs],
                   incumbent_on_Z=sorted(lifted_shas) * len(lifted)),
        margin_bytes=PASS_CEILING_BYTES - j_z,
        verdict="PASS" if j_z <= PASS_CEILING_BYTES else "FAIL",
        falsifier=("SO2's single scientific falsifier: J_Z > 123,998 B closes this specified "
                   "lifting + one-prior + 60-epoch law at INSTANCE/FORMULATION scope"),
        producer=jg2.file_fact(Path(__file__)),
    )
    return record(ROOT / "RESULT.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pack", "encode", "decode", "verdict"])
    parser.add_argument("--checkpoint", type=Path, help="pack: the epoch-60 terminal EMA checkpoint")
    parser.add_argument("--tag", default="primary")
    parser.add_argument("--tags", nargs="+", default=["primary", "repeat"])
    args = parser.parse_args()
    if args.stage == "pack":
        if args.checkpoint is None:
            raise SystemExit("pack needs --checkpoint")
        print(json.dumps(pack(args.checkpoint), sort_keys=True)[:3000])
    elif args.stage == "encode":
        print(json.dumps(encode(args.tag), sort_keys=True)[:3000])
    elif args.stage == "decode":
        print(json.dumps(decode(args.tag), sort_keys=True)[:3000])
    else:
        print(json.dumps(verdict(tuple(args.tags)), sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
