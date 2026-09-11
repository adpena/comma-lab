"""ddm_tmx1 — exact-byte price rail for a REFIT of the tail coder's counted mixer.

The 60 B RLC1 rider is ``variant(1) + 40 int8 weights + 19 B geometry``.  The 40
weights are tc1's 35 shared-mixer coefficients (5 winning classes x 7 features)
plus tc3's 5 lane coefficients.  They are COUNTED data read by shipped receiver
code, so replacing their values is not a receiver change.

This rail is a FORK of ``experiments/ddm_hpr1_shape_price.py`` (the landing that
priced move 47) with exactly one structural difference: the treatment replaces
``parts.tc1_weights`` instead of the HPAC body, so the ``hpac`` member is carried
through untouched and no RC3 re-encode runs.  Everything else -- the known-symbol
twin-encoder loop, the in-loop output-lossless assertion, the durable
encoder/receiver checkpoint pairing, the storage reserve -- is hpr1's, kept
line-for-line wherever the fork allows.

Axis ``[macOS-CPU advisory; exact bytes, scorer-free]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_rlc1_run as landed
from experiments import ddm_hpr1_shape_price as hpr1
from experiments.ddm_tc1_public_proof import build_libraries

#: The ONLY write root, with the 40 GiB fail-closed reserve this arm inherited from the
#: sister HPAC producers.  MEASURED 2026-09-11: the tier fell from 43 GiB to 35.3 GiB free
#: (verified twice: `df` and `diskutil apfs list`, no local snapshots) under a sister arm's
#: two live full-video ``inflate.sh`` diagnostics, whose 7.1 GiB of output persists, and
#: this arm's first control encode was REFUSED at the durable checkpoint.  The reserve is
#: NOT lowered and no second tier is used (MAIN, 2026-09-11: "never route to local or
#: APDataStore"); the refusal is reported as a TIER-level condition, because at free below
#: the reserve every payload refuses regardless of its size.  The footprint is cut instead:
#: no decoded-field raster is re-retained and the fit trace samples 1 in 32.
STORE_ROOTS = {"vertigo": (Path("/Volumes/VertigoDataTier/pact/ddm_tmx1/price"), 40 << 30)}
ROOT, RESERVE_BYTES = STORE_ROOTS["vertigo"]
#: The move-47 promoted tree (SEALED; copy only) and hpr1's in-flight move-48 composition.
PROMOTED47 = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_control/candidate_runtime")
COMPOSITION48 = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/price/retrain_frame_even/retained/archive.twin0.zip")
MOVE47_SHA = "d1fab05d69f31c90ac55173fa87072949e5ea1e069a0b7614337089b7a2a0ce9"
COMPOSITION48_SHA = "d830edd37164"  # prefix from hpr1's PRICE.json; the full sha is bound at prepare time
FIELD = Path("/Volumes/APDataStore/pact/ddm_hpr1/inputs/field.u8")
FIELD_SHA = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
SEED = 20260911
#: The sampling stride for the retained fit trace.  A deterministic hash of
#: (frame, flat position) keeps the sample decorrelated from the coding GROUP
#: lattice, which a plain modulus on the flat index would follow exactly.  32 keeps
#: 3.69 M of the 117,964,800 coded symbols -- ample for a PAIRED comparison of two
#: 40-parameter vectors -- at about 90 MB, after MAIN's 2026-09-11 footprint ruling.
SAMPLE_STRIDE = 32
CONTROLS = ("control47", "control48")
TREATMENTS = CONTROLS + ("refit47", "refit48", "refit47b", "refit48b")
ON_48 = ("control48", "refit48", "refit48b")


class PriceError(RuntimeError):
    """A pricing input or invariant is not what the shipped object says it is."""


def _mounted(path: Path) -> Path:
    """The nearest existing ancestor, so a fresh root still answers ``disk_usage``."""
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    raise PriceError("no existing ancestor for the store root")


def fact(path) -> dict:
    return landed.fact(Path(path))


def retain(path: Path, payload: bytes) -> dict:
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise PriceError("write outside this arm's price store")
    if shutil.disk_usage(_mounted(ROOT)).free < RESERVE_BYTES + len(payload):
        raise PriceError("STORAGE_BLOCK: keep all existing evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    landed.io.persist_immutable_bytes(path, payload, label="ddm_tmx1 retained payload")
    return fact(path)


def record(path: Path, value: dict) -> dict:
    retain(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    return value


def live_pointer() -> dict:
    """The CURRENT frontier pointer, read at call time, never a constant."""
    document = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    return document["our_local_frontier_contest_cuda"]


def sample_mask(frame: int, positions: np.ndarray) -> np.ndarray:
    """Deterministic 1-in-SAMPLE_STRIDE systematic sample, decorrelated from GROUP."""
    mixed = (positions.astype(np.uint64) * np.uint64(2654435761) + np.uint64(frame) * np.uint64(40503))
    mixed ^= mixed >> np.uint64(15)
    return (mixed % np.uint64(SAMPLE_STRIDE)) == 0


def base_tree(tag: str, work: Path) -> tuple[Path, str]:
    """Materialise the base tree this treatment prices against, and bind it by sha."""
    live = live_pointer()["archive_sha256"]
    pointer47 = fact(PROMOTED47 / "archive.zip")
    # The pointer moved to move 48 UNDER this arm, at 23:12 UTC, and this guard is what
    # caught it.  A CANDIDATE row must price against the live pointer -- that is the ON_48
    # branch below, which still refuses on a move.  A move-47 row is now a DIAGNOSTIC
    # against a named, superseded base, so it is bound to the RECORDED move-47 sha rather
    # than to "whatever is live", and it declares that it is not the live pointer.  Binding
    # a historical base to a moving pointer would be the opposite error to the one the
    # ON_48 guard prevents.
    if tag not in ON_48:
        if pointer47["sha256"] != MOVE47_SHA:
            raise PriceError(f"move-47 tree {pointer47['sha256']} is not the recorded move-47 archive")
        return PROMOTED47, pointer47["sha256"]
    # The move-48 composition exists only as a bare archive in hpr1's read-only price
    # store.  Its tree is move 47's tree with that archive swapped in: frame_even is a
    # weights rounding, so every receiver file is identical and the file-set assertion
    # below proves it rather than assuming it.
    composed = work / "base48"
    archive = fact(COMPOSITION48)
    if not archive["sha256"].startswith(COMPOSITION48_SHA):
        raise PriceError("move-48 composition archive is not the one hpr1 priced")
    # THE CANDIDATE GUARD: a row priced on this base is a row against the LIVE pointer, so
    # a move underneath it refuses rather than silently pricing a stale base.
    if archive["sha256"] != live:
        raise PriceError(f"POINTER_MOVED: composition base {archive['sha256']} is not the live pointer {live}")
    for src in sorted(PROMOTED47.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc" and not src.name.startswith("._"):
            destination = composed / src.relative_to(PROMOTED47)
            if destination.exists():
                continue
            retain(destination, COMPOSITION48.read_bytes() if src.name == "archive.zip" else src.read_bytes())
    return composed, archive["sha256"]


def prepare(tag: str, weights: Path | None):
    """Copy the base tree into this arm's store and bind every input by sha."""
    work = ROOT / tag
    tree, base_sha = base_tree(tag, work)
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise PriceError("field sha mismatch")
    runtime = work / "runtime_copy"
    sources = {}
    for src in sorted(tree.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc" and not src.name.startswith("._"):
            destination = runtime / src.relative_to(tree)
            relative = str(destination.relative_to(runtime))
            sources[relative] = retain(destination, src.read_bytes())
    if sources["archive.zip"]["sha256"] != base_sha:
        raise PriceError("copied archive is not the base archive")
    # NO receiver file is patched by this rail.  The assertion is structural: the
    # runtime copy must equal the base tree byte for byte, every file.
    for relative, value in sources.items():
        if value["sha256"] != fact(tree / relative)["sha256"]:
            raise PriceError(f"runtime copy diverged from the base tree: {relative}")
    rx, renderer, code_dir = landed.io.load_runtime(runtime)
    parts = rx.read_residual_archive(runtime / "archive.zip")
    shipped = bytes(parts.tc1_weights)
    if len(shipped) != 60 or shipped[0] != 1:
        raise PriceError("base rider is not a 60 B RLC1 variant-1 config")
    if tag in CONTROLS:
        config = shipped
    else:
        if weights is None:
            raise PriceError("a refit treatment needs --weights")
        fitted = weights.read_bytes()
        if len(fitted) != 40:
            raise PriceError("a refit supplies exactly 40 counted int8 weights")
        config = shipped[:1] + fitted + shipped[41:]
        if config == shipped:
            raise PriceError("refit treatment is a no-op against the shipped weights")
    if len(config) != len(shipped):
        raise PriceError("the 60 B counted state may not change size on this schema")
    retain(work / "retained/rider_config.bin", config)
    member = landed.io.split_member(landed.io.read_archive_member(runtime / "archive.zip"))
    binding = record(
        work / "INPUTS.json",
        {
            "tag": tag,
            "receiver_change": False,
            "base_archive": sources["archive.zip"],
            "base_archive_expected": {"sha256": base_sha, "bytes": fact(tree / "archive.zip")["bytes"]},
            "base_tree": str(tree),
            "live_pointer": {k: live_pointer().get(k) for k in ("archive_sha256", "score")},
            "prices_against_live_pointer": base_sha == live_pointer()["archive_sha256"],
            "shipped_rider_sha256": hashlib.sha256(shipped).hexdigest(),
            "candidate_rider_sha256": hashlib.sha256(config).hexdigest(),
            "shipped_weights_sha256": hashlib.sha256(shipped[1:41]).hexdigest(),
            "candidate_weights_sha256": hashlib.sha256(config[1:41]).hexdigest(),
            "weights_source": None if weights is None else fact(weights),
            "source_files": sources,
            "field": fact(FIELD),
            "producer": fact(Path(__file__)),
            "forked_from": fact(REPO / "experiments/ddm_hpr1_shape_price.py"),
            "hpac_section_bytes": len(member["hpac"]),
            "seed": SEED,
            "sample_stride": SAMPLE_STRIDE,
            "axis": "[macOS-CPU advisory; exact bytes, scorer-free]",
            "score_claim": False,
            "store_root": str(ROOT),
            "store_reserve_bytes": RESERVE_BYTES,
            "cleanup": "KEEP all payloads; immutable stage checkpoints; 40 GiB reserve",
        },
    )
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    record(
        work / f"STORAGE_ROUTING_{stamp}.json",
        {
            "store_root": str(ROOT),
            "store_reserve_bytes": RESERVE_BYTES,
            "store_free_bytes": shutil.disk_usage(_mounted(ROOT)).free,
            "vertigo_free_bytes": shutil.disk_usage(_mounted(STORE_ROOTS["vertigo"][0])).free,
            "note": "volatile observation, deliberately OUTSIDE INPUTS.json so the binding stays reproducible",
        },
    )
    from dataclasses import replace

    return work, runtime, rx, renderer, code_dir, replace(parts, tc1_weights=config), member, binding


def encode(tag: str, weights: Path | None = None, collect: bool = False) -> dict:
    """Run the shipping RLC1 causal loop with known-symbol twin arithmetic encoders."""
    import torch

    started = time.time()
    work, runtime, rx, renderer, code_dir, parts, member, binding = prepare(tag, weights)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    landed.ROOT = ROOT  # the shared io helper's storage guard, bound to THIS arm's store
    route = landed.io.load_route_b()
    build_path = work / "ENCODER_BUILD.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        for key in ("base_source", "generated", "library"):
            if fact(build[key]["path"]) != build[key]:
                raise PriceError("encoder build drift")
        library = Path(build["library"]["path"])
    else:
        library, build = landed.io.compile_rc64(work, route, "tmx1")
        record(build_path, build)
    build_libraries(runtime, work / "native")
    # hpr1's own geometry build, imported rather than re-derived: it writes only into the
    # directory it is handed, so this arm's store stays this arm's.
    geometry = hpr1.build_geometry(runtime, work / "native")
    checkpoints = work / "checkpoints"
    os.environ["TC1_RECEIVER_CHECKPOINT_DIR"] = str(checkpoints)
    os.environ["TC1_RECEIVER_STOP_AFTER"] = "600"
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.rlc1_mixer import LaneMixer
    from runtime.tc1_receiver_checkpoint import ReceiverCheckpoint

    field = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(600, 384, 512))
    state, start = None, 0
    latest = checkpoints / "LATEST.json"
    if latest.exists():
        start = int(json.loads(latest.read_text())["frame"])
        state_path = work / "encoder_states" / f"stage_{start:04d}.npz"
        state_receipt = json.loads(state_path.with_suffix(".json").read_text())
        if fact(state_path) != state_receipt["payload"] or state_receipt["binding"] != binding:
            raise PriceError("encoder restart binding mismatch")
        state = landed.arrays(state_path)
    if collect and start:
        raise PriceError("the fit trace must be collected by an uninterrupted encode")
    twins = [route.NativeRc64Encoder(library, None if state is None else state[f"enc{i}"].tobytes()) for i in range(2)]
    observed = {"frame": start, "positions": None}
    original_coding, original_save = LaneMixer.coding, ReceiverCheckpoint.save
    original_end, original_begin = LaneMixer.end_frame, LaneMixer.begin_frame

    # The EXACT ideal-code baseline, accumulated from the same rows the arithmetic
    # encoder is fed, split by frame parity so the held-out folds share one instrument.
    exact_bits = np.zeros(2, dtype=np.float64)
    exact_symbols = np.zeros(2, dtype=np.int64)
    trace = {"rows": [], "bins": [], "pos": [], "truth": [], "frame": [], "tables": [], "lane": []}

    def begin(self):
        original_begin(self)
        if collect:
            # LEVELS insertion order -- the SAME order ``SharedMixer.features`` walks to fill phi.
            trace["tables"].append(np.concatenate([t.reshape(-1) for t in self.old.tables.values()]))
            trace["lane"].append(self.table.reshape(-1).copy())

    def coding(self, rows, positions, plane, previous):
        if observed["positions"] is not None or self.frame != observed["frame"]:
            raise PriceError("known-symbol group order mismatch")
        observed["positions"] = positions.copy()
        result = original_coding(self, rows, positions, plane, previous)
        if collect:
            keep = sample_mask(observed["frame"], np.asarray(positions, dtype=np.int64))
            if keep.any():
                chosen = np.asarray(positions, dtype=np.int64)[keep]
                trace["rows"].append(np.asarray(rows, dtype=np.float32)[keep].copy())
                trace["bins"].append(self.bins[chosen].copy())
                trace["pos"].append(chosen.astype(np.int32))
                trace["truth"].append(field[observed["frame"]].reshape(-1)[chosen].copy())
                trace["frame"].append(np.full(len(chosen), observed["frame"], dtype=np.int16))
        return result

    def known_symbols(self, probabilities):
        positions = observed["positions"]
        if positions is None or len(positions) != len(probabilities):
            raise PriceError("known-symbol decode call lacks its group")
        symbols = field[observed["frame"]].reshape(-1)[positions].astype(np.int32)
        rows = np.asarray(probabilities, dtype=np.float64)
        coded = np.clip(rows[np.arange(len(symbols)), symbols], np.finfo(np.float64).tiny, 1.0)
        fold = observed["frame"] & 1
        exact_bits[fold] += float(-np.log2(coded).sum())
        exact_symbols[fold] += len(symbols)
        for encoder in twins:
            encoder.encode(symbols, probabilities)
        observed["positions"] = None
        return symbols

    def end(self, plane, previous):
        # The output-lossless proof: the plane the receiver reconstructs under the
        # candidate weights is the shipped field, frame by frame.
        np.testing.assert_array_equal(plane, field[observed["frame"]])
        original_end(self, plane, previous)
        observed["frame"] += 1

    def save(self, frame, tokens):
        if frame != observed["frame"] or observed["positions"] is not None:
            raise PriceError("encoder checkpoint is not at a frame boundary")
        if shutil.disk_usage(_mounted(ROOT)).free < RESERVE_BYTES + (256 << 20):
            raise PriceError("STORAGE_BLOCK at the durable prior checkpoint")
        path = work / "encoder_states" / f"stage_{frame:04d}.npz"
        landed.ROOT = ROOT
        payload = landed.save(
            path, {f"enc{i}": np.frombuffer(e.snapshot(), dtype=np.uint8) for i, e in enumerate(twins)}
        )
        record(path.with_suffix(".json"), {"payload": payload, "binding": binding, "frame": frame})
        original_save(self, frame, tokens)

    LaneMixer.coding, LaneMixer.end_frame, LaneMixer.begin_frame = coding, end, begin
    NativeDecoder.decode, ReceiverCheckpoint.save = known_symbols, save
    try:
        tokens, _ = rx.decode_production_tokens(parts, renderer, code_dir, torch.device("cpu"))
    finally:
        LaneMixer.coding, LaneMixer.end_frame, LaneMixer.begin_frame = original_coding, original_end, original_begin
        ReceiverCheckpoint.save = original_save
    field_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if observed["frame"] != 600 or field_sha != FIELD_SHA:
        raise PriceError("known-symbol loop did not process the full field")
    # The decoded field is NOT re-retained: it must EQUAL the bound input field by the
    # in-loop assertion above, so a second 112.5 MB copy of a file this arm already binds
    # by sha would be footprint without evidence.  The sha is the receipt (MAIN, 2026-09-11:
    # "your control needs no rasters at all -- it is a byte-identity of the coded stream").
    archives = []
    for index, encoder in enumerate(twins):
        envelope = encoder.finish()
        retain(work / f"retained/tail.twin{index}.envelope", envelope)
        import ctypes

        raw = ctypes.string_at(
            encoder.library.rc64_encoder_data(encoder.context),
            int(encoder.library.rc64_encoder_size(encoder.context)),
        )
        retain(work / f"retained/tail.twin{index}.rc64", raw)
        rider = b"RLC1" + bytes(parts.tc1_weights) + raw
        retain(work / f"retained/tail.twin{index}.rider", rider)
        changed = dict(member)
        changed["tail"] = member["tail"][:96] + rider
        body = landed.io.join_member(changed)
        retain(work / f"retained/member.twin{index}.bin", body)
        archive = work / f"retained/archive.twin{index}.zip"
        landed.pack(body, archive, "stored", None)
        parsed = rx.read_residual_archive(archive)
        if parsed.token_stream != raw or bytes(parsed.tc1_weights) != bytes(parts.tc1_weights):
            raise PriceError("archive parser mismatch")
        for name in ("hpac_blob", "semantic_blob", "carrier_blob", "residual_payload"):
            if getattr(parts, name) != getattr(parsed, name):
                raise PriceError(f"unrelated archive component changed: {name}")
        archives.append(fact(archive))
    if archives[0]["sha256"] != archives[1]["sha256"]:
        raise PriceError("tail/archive twins differ")
    if tag in CONTROLS and archives[0]["sha256"] != binding["base_archive_expected"]["sha256"]:
        raise PriceError("LIVE_LOOP_CONTROL_FAILED: no treatment prices are admissible")
    trace_receipt = _write_trace(work, trace) if collect else None
    return record(
        work / "PRICE.json",
        {
            "binding": binding,
            "geometry_build": geometry,
            "n": 600,
            "twins": archives,
            "hpac_section_bytes": len(member["hpac"]),
            "token_stream_bytes": len(raw),
            "shipped_token_stream_bytes": len(member["tail"]) - 96 - 4 - 60,
            "delta_bytes_vs_base": archives[0]["bytes"] - binding["base_archive_expected"]["bytes"],
            "decoded_field_sha256": field_sha,
            "output_lossless": field_sha == FIELD_SHA,
            "exact_ideal_bits": {"even_frames": exact_bits[0], "odd_frames": exact_bits[1],
                                 "total": float(exact_bits.sum()),
                                 "symbols": [int(exact_symbols[0]), int(exact_symbols[1])]},
            "fit_trace": trace_receipt,
            "wall_seconds": round(time.time() - started, 3),
            "public_decode_verified": False,
            "score_claim": False,
        },
    )


def _write_trace(work: Path, trace: dict) -> dict:
    """Persist the retained fit trace.  ALWAYS KEEP THE PAYLOAD."""
    landed.ROOT = ROOT
    payload = {
        "rows": np.concatenate(trace["rows"]).astype(np.float32),
        "bins": np.concatenate(trace["bins"]).astype(np.uint8),
        "pos": np.concatenate(trace["pos"]).astype(np.int32),
        "truth": np.concatenate(trace["truth"]).astype(np.uint8),
        "frame": np.concatenate(trace["frame"]).astype(np.int16),
        "tables": np.stack(trace["tables"]).astype(np.int16),
        "lane_tables": np.stack(trace["lane"]).astype(np.int16),
    }
    path = work / "retained/fit_trace.npz"
    receipt = landed.save(path, payload)
    return {
        "payload": receipt,
        "samples": int(len(payload["truth"])),
        "stride": SAMPLE_STRIDE,
        "rule": f"hash(frame, flat position) % {SAMPLE_STRIDE} == 0 -- systematic, decorrelated from the GROUP lattice",
        "tables": "per frame, the five tc1 context tables concatenated in LEVELS insertion order "
                  "(spatial2, spatial3, previous, run, rowband), each (K*levels, K) int16",
        "lane_tables": "per frame, the tc3/RLC1 lane table (K*BINS, K)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", choices=TREATMENTS, required=True)
    parser.add_argument("--weights", type=Path, help="40 counted int8 weights for a refit treatment")
    parser.add_argument("--collect", action="store_true", help="retain the sampled fit trace")
    parser.add_argument("--prepare-only", action="store_true", help="bind inputs and stop before the encode")
    parser.add_argument("--store", choices=tuple(STORE_ROOTS), default="vertigo",
                        help="write root; each carries its own fail-closed reserve, never lowered")
    args = parser.parse_args()
    global ROOT, RESERVE_BYTES
    ROOT, RESERVE_BYTES = STORE_ROOTS[args.store]
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        *_, binding = prepare(args.treatment, args.weights)
        print(json.dumps({k: binding[k] for k in ("tag", "receiver_change", "base_archive",
              "shipped_rider_sha256", "candidate_rider_sha256")}, indent=2, default=str))
        return 0
    result = encode(args.treatment, args.weights, args.collect)
    print(json.dumps({k: v for k, v in result.items() if k not in ("binding",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
