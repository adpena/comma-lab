"""ddm_hpr1 -- exact-byte pricing rail for HPAC receptive-field SHAPE rungs on move 45.

WHAT THIS IS.  The same pricing law ddm_ntb2 used for its HPAC value rungs, re-rooted
on the move-45 pointer and owned by this arm's store.  Nothing about the coder, the
mixer, the corrector or the group order is re-implemented: the shipping receiver loop
runs unchanged and only the arithmetic DECODE call is replaced by the landed native
ENCODER fed with the known source symbols.  A treatment is priced by exactly two
numbers, the ``hpac`` member bytes and the re-encoded ``tail`` bytes, because the HPAC
section is the coder's PRIOR: encoder and decoder build it from the same shipped bytes,
so changing it changes code LENGTHS and never decoded symbols.  The decoded field is
asserted byte-identical to the shipped field inside the loop, which is the
output-lossless proof (no scorer is needed or run).

THE CONTROL IS THE FALSIFIER.  ``--treatment control`` re-encodes the shipped prior and
must reproduce move 45's archive byte-identically.  No treatment price is admissible
until it does.

Axis ``[macOS-CPU advisory; exact bytes, scorer-free]``; ``score_claim=false``; this is
a producer, not a public decode -- a candidate identity proof needs a separate
unmodified public decode of the built archive.

Usage::

  python experiments/ddm_hpr1_shape_price.py --treatment control --resume-from <root>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_rlc1_run as landed
from experiments.ddm_rc1_model_section_adaptive_recode import ck2_interleave
from experiments.ddm_tc1_public_proof import build_libraries

#: Storage tiers this arm may write.  MAIN re-routed new payloads to Vertigo on
#: 2026-09-11 while APDataStore sat just above its fail-closed reserve; the legacy
#: root is kept read-only so the control priced there stays citable.
STORE_ROOTS = {
    "vertigo": Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/price"),
    "apdatastore": Path("/Volumes/APDataStore/pact/ddm_hpr1/price"),
}
LEGACY_ROOT = STORE_ROOTS["apdatastore"]
ROOT = STORE_ROOTS["vertigo"]
PROMOTED45 = Path("/Volumes/VertigoDataTier/pact/ddm_pc3_pose_carrier_curve/candidate/candidate_runtime")
POINTER45_SHA = "145e02e21f9a1cbc8276d1ecc34f0b9ae4762afa3fea7811e5836fee770ae60a"
POINTER45_BYTES = 180_246
#: The move-47 promoted tree: this arm's own retrain control, now sealed. A treatment that
#: composes ON TOP of the shipped prior reads its body out of THIS archive, so the base is
#: named by the live pointer rather than by a constant -- the pointer moved twice under this
#: arm already and each time a hardcoded sha went stale the moment it did.
PROMOTED47 = Path("/Volumes/VertigoDataTier/pact/ddm_hpr1/public/retrain_control/candidate_runtime")


def live_pointer() -> dict:
    """The CURRENT frontier pointer, read at call time, never a constant."""
    document = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    return document["our_local_frontier_contest_cuda"]
FIELD = Path("/Volumes/APDataStore/pact/ddm_hpr1/inputs/field.u8")
FIELD_SHA = "a92e7d902a4498961217f02c2b90d3fb9025901ba6d047201ff3bf297fa2f7a8"
#: Free-space floor, matched to the 40 GiB fail-closed reserve the sister HPAC
#: producers hold.  Never lowered: a refusal here is the guard working.
RESERVE_BYTES = 40 << 30
TREATMENTS = (
    "control", "retrain", "past_dil2", "past_dil3", "cone_dil2", "cone_dil3",
    # Composition rows: ntb2's even-rounding of the frame embedding, re-applied ON TOP of
    # the retrained prior that move 47 ships. Base = move 47, not move 45.
    "retrain_frame_even", "retrain_frame_quad",
    # The move-47 CONTROL: the shipped retrained prior, re-encoded unchanged. Its falsifier
    # is that it must reproduce move 47's archive byte-identically, which is what makes the
    # q values collected alongside it the SHIPPED mixer's own opinion and not an artefact.
    "control47",
)
#: Treatments whose base is the move-47 promoted tree rather than move 45's.
ON_MOVE47 = ("retrain_frame_even", "retrain_frame_quad", "control47")
#: The rounding step each composition row applies to `frame_embed.weight`.
FRAME_STEP = {"retrain_frame_even": 2, "retrain_frame_quad": 4}


class PriceError(RuntimeError):
    """A pricing input or invariant is not what the shipped object says it is."""


def fact(path) -> dict:
    return landed.fact(Path(path))


def retain(path: Path, payload: bytes) -> dict:
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise PriceError("write outside this arm's price store")
    if shutil.disk_usage(ROOT.parent).free < RESERVE_BYTES + len(payload):
        raise PriceError("STORAGE_BLOCK: keep all existing evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    landed.io.persist_immutable_bytes(path, payload, label="ddm_hpr1 retained payload")
    return fact(path)


def record(path: Path, value: dict) -> dict:
    retain(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    return value


def build_geometry(runtime: Path, work: Path) -> dict:
    """Compile the RLC1 geometry library exactly as ``inflate.sh`` does."""
    target = work / "rlc1_geometry.so"
    source = runtime / "runtime/rlc1_geometry.c"
    command = [
        os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(target)
    ]
    work.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (work / "rlc1_geometry.build.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
    os.environ["RLC1_GEOMETRY_LIBRARY"] = str(target)
    return {"source": fact(source), "library": fact(target), "argv": command}


#: A shape rung's receiver constant.  conv_past's dilation lives ENTIRELY in the
#: receiver: the packer's topology carries no dilation and the serialized rows are the
#: masked taps, so a dilated prior's IHS1 bytes have exactly the shipped structure.
#: That is what makes this a pure shape rung -- and what makes it a receiver change.
#: ``retrain`` is the SHAPE CONTROL: the shipped geometry, retrained under the same law
#: from the same warm start.  It is the only treatment here that is NOT a receiver
#: change, so if it wins it takes the normal seal path -- which is why it must be priced
#: before any shape rung is proposed as the candidate.
TREATMENT_PAST_DILATION = {
    "control": 1, "retrain": 1, "past_dil2": 2, "past_dil3": 3, "cone_dil2": 1, "cone_dil3": 1,
    "retrain_frame_even": 1, "retrain_frame_quad": 1, "control47": 1,
}
#: conv_a's spacing.  The receiver reaches conv_a's taps through geometry-general code in
#: BOTH the optimized torch path (``hpac_inference._conv_a_features`` builds its gather
#: from ``sparse.a_offsets`` with a bounds check) and the native export
#: (``f26_hpac_native.c`` reads ``a_offsets`` off the module), so this axis costs ONE
#: receiver constant and nothing in C.
TREATMENT_CONE_DILATION = {
    "control": 1, "retrain": 1, "past_dil2": 1, "past_dil3": 1, "cone_dil2": 2, "cone_dil3": 3,
    "retrain_frame_even": 1, "retrain_frame_quad": 1, "control47": 1,
}


def assert_layout_held(body: bytes, shipped_body: bytes, counts: list[int]) -> dict:
    """A shape rung moves taps; it must not add or drop any.

    The invariant is the LAYOUT -- the same rows with the same per-row value counts --
    NOT the byte length.  Packed length legitimately moves because QAT relearns the
    per-row bit depths, and that movement IS the rung's model leg.  Checking length
    here would refuse every real rung.
    """
    from runtime import rc2_hpac_semistatic_mixing as rc2

    rows, depths = rc2.unpack_rows(body, counts)
    shipped_rows, shipped_depths = rc2.unpack_rows(shipped_body, counts)
    if len(rows) != len(shipped_rows):
        raise PriceError("packed body changed the row count: this is not a shape rung")
    moved = [i for i, (a, b) in enumerate(zip(rows, shipped_rows, strict=True)) if a.size != b.size]
    if moved:
        raise PriceError(f"packed body changed per-row value counts at rows {moved[:8]}")
    return {
        "rows": len(rows),
        "values": int(sum(r.size for r in rows)),
        "mean_depth_bits": float(sum(int(d) for d in depths) / len(depths)),
        "shipped_mean_depth_bits": float(sum(int(d) for d in shipped_depths) / len(shipped_depths)),
        "body_bytes": len(body),
        "shipped_body_bytes": len(shipped_body),
        "delta_body_bytes": len(body) - len(shipped_body),
    }


def treatment_body(tag: str, shipped_body: bytes, work: Path, checkpoint: Path | None, renderer=None) -> bytes:
    """Return the IHS1 body this treatment ships.

    ``control`` returns the shipped bytes unchanged, which is what makes the archive
    reproduction a real falsifier.  A shape rung packs its own retrained terminal
    checkpoint through the landed IHS1 packer -- the same call cl2's ladder used -- so
    no serialization is re-implemented here either.
    """
    if tag in ("control", "control47"):
        return shipped_body
    if tag in ON_MOVE47:
        # ntb2's frame_even, re-applied on top of whatever prior the base archive ships.
        # `shipped_body` here IS the move-47 body, i.e. the retrained prior, so this
        # composes rather than re-does: the rounding is a value edit on one tail field and
        # touches nothing else.
        import numpy as _np
        from runtime import ihs2 as _ihs2
        from runtime import rc2_hpac_semistatic_mixing as _rc2

        step = FRAME_STEP[tag]
        layout = _ihs2.layout_from_runtime(renderer)
        counts = list(layout.row_counts)
        rows, depths = _rc2.unpack_rows(shipped_body, counts)
        if _np.any((depths < 0) | (depths > 15)):
            raise PriceError("IHS1 depth is outside the counted nibble domain")
        depths = depths.astype(_np.uint8)
        _, _, tail, _ = _rc2.split_ihs1(shipped_body, counts)
        frame = layout.frame_field
        if layout.tail_fields[0].name != frame.name:
            raise PriceError("frame embedding tail offset changed")
        values = _np.frombuffer(tail[: frame.byte_count], dtype=_np.int8).astype(_np.int16)
        limit = 128 - (128 % step)
        coarse = _np.clip(_np.rint(values / step) * step, -128, limit - step).astype(_np.int8)
        changed = int(_np.count_nonzero(values != coarse))
        tail = coarse.tobytes() + tail[frame.byte_count :]
        packed_depths = _np.zeros((len(depths) + 1) // 2, dtype=_np.uint8)
        packed_depths[:] = depths[::2]
        packed_depths[: len(depths) // 2] |= depths[1::2] << 4
        body = b"IHS1" + packed_depths.tobytes() + _rc2.pack_rows(rows, depths) + tail
        record(
            work / "COMPOSITION.json",
            {
                "tag": tag,
                "step": step,
                "field": frame.name,
                "values": int(values.size),
                "changed": changed,
                "quantization": f"nearest multiple of {step}; ties to even",
                "base": "the move-47 promoted archive's own IHS1 body (the retrained prior)",
                "score_claim": False,
            },
        )
        return body
    if checkpoint is None:
        raise PriceError(f"treatment {tag} requires --checkpoint")
    from experiments import ddm_rx2_mc36_identity_race as rx2

    packed = rx2._pack_terminal_ihs1(checkpoint, work / "model")
    return Path(packed["raw"]["path"]).read_bytes()


def patch_receiver_cone(runtime: Path, dilation: int) -> dict:
    """Set conv_a's dilation in this arm's runtime COPY -- one constant, no C change.

    ``residual_archive`` calls ``optimize_sparse_evaluator`` unconditionally, which
    rebinds ``selected_logits`` to the geometry-general implementation, so the padded
    dilation-1 routine in ``hpac_integer_sparse`` is dead on the shipped path.
    """
    target = runtime / "cpr1/inflate.py"
    before = fact(target)
    text = target.read_text()
    already = f"HPAC_CONE_DILATION = {dilation}"
    if already in text:
        return {"cone_dilation": dilation, "file": "cpr1/inflate.py", "before": before, "after": before, "resumed": True}
    anchor = "HPAC_LOGIT_PRECISION = 8"
    if anchor not in text or "HPAC_CONE_DILATION" in text:
        raise PriceError("cone patch anchor is absent, or a different dilation is applied")
    text = text.replace(anchor, anchor + f"\n# ddm_hpr1 shape rung: spacing of conv_a's causal cone.\nHPAC_CONE_DILATION = {dilation}", 1)
    call = "    deserialize_integer_model(model, raw)\n    return model.to(device)"
    if text.count(call) != 1:
        raise PriceError("receiver load_hpac anchor is not unique")
    text = text.replace(call, "    model.conv_a.dilation = HPAC_CONE_DILATION\n" + call, 1)
    target.chmod(0o644)
    target.write_text(text)
    return {"cone_dilation": dilation, "file": "cpr1/inflate.py", "before": before, "after": fact(target)}


def patch_receiver_dilation(runtime: Path, dilation: int) -> dict:
    """Set conv_past's dilation in this arm's runtime COPY -- a receiver change.

    The receptive field ships as receiver code, not as archive bytes, so a shape rung
    is a receiver change by construction and routes to the first-measurement chain.
    Only a generic integer constant moves; no video-derived value enters the code.
    """
    target = runtime / "cpr1/inflate.py"
    before = fact(target)
    text = target.read_text()
    anchor = "HPAC_LOGIT_PRECISION = 8"
    already = f"HPAC_PAST_DILATION = {dilation}"
    if already in text:
        # Idempotent: a resumed encode re-enters this path with the patch in place.
        return {
            "dilation": dilation,
            "receiver_constant": {"file": "cpr1/inflate.py", "before": before, "after": before},
            "native_stencil": {
                "file": "runtime/f26_hpac_native.c",
                "before": fact(runtime / "runtime/f26_hpac_native.c"),
                "after": fact(runtime / "runtime/f26_hpac_native.c"),
                "why": "already patched; resumed run",
            },
            "resumed": True,
        }
    if anchor not in text or "HPAC_PAST_DILATION" in text:
        raise PriceError("receiver patch anchor is absent, or a different dilation is applied")
    text = text.replace(
        anchor,
        anchor + "\n# ddm_hpr1 shape rung: dilation of conv_past, the prior's temporal tap set.\n"
        f"HPAC_PAST_DILATION = {dilation}",
        1,
    )
    call = "    deserialize_integer_model(model, raw)\n    return model.to(device)"
    if text.count(call) != 1:
        raise PriceError("receiver load_hpac anchor is not unique")
    text = text.replace(
        call,
        "    model.conv_past.dilation = HPAC_PAST_DILATION\n"
        "    model.conv_past.padding = HPAC_PAST_DILATION\n" + call,
        1,
    )
    target.chmod(0o644)
    target.write_text(text)
    after = fact(target)

    # The NATIVE token decoder (F26_TOKEN_DECODER=native-hpac, opt-in; the shipped
    # default is "python") re-implements conv_past in C with the dilation-1 stencil
    # baked into its source-coordinate arithmetic.  Leaving it unpatched would let one
    # decoder mode silently produce a DIFFERENT field from the other, so the same
    # constant moves in both places or the rung refuses.
    native = runtime / "runtime/f26_hpac_native.c"
    native_before = fact(native)
    source = native.read_text()
    replacements = [
        ("int32_t source_row = global_row + kernel_row - 1;",
         f"int32_t source_row = global_row + (kernel_row - 1) * {dilation};"),
        ("int32_t source_col = global_col + kernel_col - 1;",
         f"int32_t source_col = global_col + (kernel_col - 1) * {dilation};"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise PriceError(f"native conv_past stencil anchor is not unique: {old!r}")
        source = source.replace(old, new, 1)
    native.chmod(0o644)
    native.write_text(source)
    return {
        "dilation": dilation,
        "receiver_constant": {"file": "cpr1/inflate.py", "before": before, "after": after},
        "native_stencil": {
            "file": "runtime/f26_hpac_native.c",
            "before": native_before,
            "after": fact(native),
            "why": "native-hpac mode re-implements conv_past in C with the dilation baked in",
        },
    }


def prepare(tag: str, checkpoint: Path | None = None):
    """Copy the sealed move-45 tree into this arm's store and bind every input."""
    import brotli
    import torch

    live_row = live_pointer()
    live = live_row["archive_sha256"]
    on_47 = tag in ON_MOVE47
    base_tree = PROMOTED47 if on_47 else PROMOTED45
    base_sha = fact(base_tree / "archive.zip")["sha256"]
    if on_47:
        # A composition row prices against the LIVE pointer, so its base must BE the live
        # pointer; a move underneath refuses rather than silently pricing a stale base.
        if base_sha != live:
            raise PriceError(f"POINTER_MOVED: base {base_sha} is not the live pointer {live}")
    elif base_sha != POINTER45_SHA:
        raise PriceError(f"base tree {base_sha} is not the move-45 tree this rail priced")
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise PriceError("field sha mismatch")
    work = ROOT / tag
    runtime = work / "runtime_copy"
    # The two receiver files a shape rung patches.  On a RESUMED encode they already
    # hold the patched bytes, so re-copying the pristine source would either refuse
    # under immutable persistence or silently revert the rung.  Both are refused here
    # by naming them: a resumed copy keeps what is on disk and records its sha.
    patched_names = {"cpr1/inflate.py", "runtime/f26_hpac_native.c"}
    sources = {}
    for src in sorted(base_tree.rglob("*")):
        if src.is_file() and "__pycache__" not in src.parts and src.suffix != ".pyc" and not src.name.startswith("._"):
            destination = runtime / src.relative_to(base_tree)
            relative = str(destination.relative_to(runtime))
            if destination.exists() and relative in patched_names:
                sources[relative] = fact(destination)
                continue
            sources[relative] = retain(destination, src.read_bytes())
    if sources["archive.zip"]["sha256"] != base_sha:
        raise PriceError("copied archive is not the base archive")
    dilation = TREATMENT_PAST_DILATION[tag]
    cone = TREATMENT_CONE_DILATION[tag]
    receiver_patch = None if dilation == 1 else patch_receiver_dilation(runtime, dilation)
    cone_patch = None if cone == 1 else patch_receiver_cone(runtime, cone)
    rx, renderer, code_dir = landed.io.load_runtime(runtime)
    from runtime import ihs2
    from runtime import rc3_shared_mixer as rc3

    parts = rx.read_residual_archive(runtime / "archive.zip")
    layout = ihs2.layout_from_runtime(renderer)
    counts = list(layout.row_counts)
    shipped_body = rx.materialize_ihs1(parts.hpac_blob, renderer)
    body = treatment_body(tag, shipped_body, work, checkpoint, renderer)
    layout_held = assert_layout_held(body, shipped_body, counts)
    retain(work / "retained/hpac.ihs1", body)
    if (body == shipped_body) != (tag in ("control", "control47")):
        raise PriceError("treatment is a no-op, or the control body changed")
    if renderer.load_hpac(body, torch.device("cpu")) is None:
        raise PriceError("real integer model failed to load")
    header = rc3.HEADER.unpack_from(parts.hpac_blob)
    offset = rc3.HEADER.size + header[-3]
    learning = parts.hpac_blob[offset]
    weights = np.frombuffer(parts.hpac_blob[offset + 1 : offset + 25], dtype=np.int8)
    member = landed.io.split_member(landed.io.read_archive_member(runtime / "archive.zip"))
    containers = []
    for twin in range(2):
        rider, details = rc3.encode(body, counts, header[2], weights, learning)
        retain(work / f"retained/hpac.twin{twin}.rider", rider)
        retain(work / f"retained/hpac.twin{twin}.range", details["payload"])
        retain(work / f"retained/hpac.twin{twin}.params", details["parameters"])
        outer = ck2_interleave(rider)
        retain(work / f"retained/hpac.twin{twin}.ck2", outer)
        # The RC3 landing selected brotli q10 / lgwin22; ntb1's window24 variant
        # preserves the byte COUNT only, not the incumbent compressed bytes.
        encoded = brotli.compress(outer, quality=10, lgwin=22)
        retain(work / f"retained/hpac.twin{twin}.br", encoded)
        if rc3.restore_hpac(rider, counts) != body:
            raise PriceError("shipping RC3 parser disagrees with the packed body")
        containers.append(encoded)
    if containers[0] != containers[1]:
        raise PriceError("HPAC twin encode differs")
    if tag in ("control", "control47") and containers[0] != member["hpac"]:
        raise PriceError("HPAC_CONTAINER_CONTROL_FAILED")
    binding = record(
        work / "INPUTS.json",
        {
            "tag": tag,
            "past_dilation": dilation,
            "layout_held": layout_held,
            "cone_dilation": cone,
            "receiver_patch": receiver_patch,
            "cone_patch": cone_patch,
            "receiver_change": receiver_patch is not None or cone_patch is not None,
            "checkpoint": None if checkpoint is None else fact(checkpoint),
            "base_archive": sources["archive.zip"],
            "base_archive_expected": {"sha256": base_sha, "bytes": fact(base_tree / "archive.zip")["bytes"]},
            "base_tree": str(base_tree),
            "live_pointer": {"archive_sha256": live, "score": live_row.get("score")},
            "source_files": sources,
            "field": fact(FIELD),
            "producer": fact(Path(__file__)),
            "body": fact(work / "retained/hpac.ihs1"),
            "hpac_section_bytes": len(containers[0]),
            "shipped_hpac_section_bytes": len(member["hpac"]),
            "seed": 20260911,
            "axis": "[macOS-CPU advisory; exact bytes, scorer-free]",
            "score_claim": False,
            "store_root": str(ROOT),
            "cleanup": "KEEP all payloads; immutable stage checkpoints; 40 GiB reserve",
        },
    )
    from dataclasses import replace

    altered = replace(parts, hpac_blob=(work / "retained/hpac.twin0.rider").read_bytes())
    return work, runtime, rx, renderer, code_dir, altered, member, binding


def encode(tag: str, checkpoint: Path | None = None, collect_q: bool = False) -> dict:
    """Run the shipping RLC1 causal loop with known-symbol twin arithmetic encoders."""
    import torch

    work, runtime, rx, renderer, code_dir, parts, member, binding = prepare(tag, checkpoint)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260911)
    np.random.seed(20260911)
    torch.use_deterministic_algorithms(True)
    route = landed.io.load_route_b()
    build_path = work / "ENCODER_BUILD.json"
    if build_path.exists():
        build = json.loads(build_path.read_text())
        for key in ("base_source", "generated", "library"):
            if fact(build[key]["path"]) != build[key]:
                raise PriceError("encoder build drift")
        library = Path(build["library"]["path"])
    else:
        library, build = landed.io.compile_rc64(work, route, "hpr1")
        record(build_path, build)
    build_libraries(runtime, work / "native")
    geometry = build_geometry(runtime, work / "native")
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
    twins = [route.NativeRc64Encoder(library, None if state is None else state[f"enc{i}"].tobytes()) for i in range(2)]
    observed = {"frame": start, "positions": None}
    original_coding, original_save = LaneMixer.coding, ReceiverCheckpoint.save
    original_end = LaneMixer.end_frame

    def coding(self, rows, positions, plane, previous):
        if observed["positions"] is not None or self.frame != observed["frame"]:
            raise PriceError("known-symbol group order mismatch")
        observed["positions"] = positions.copy()
        return original_coding(self, rows, positions, plane, previous)

    # The coder's scalar q, as the mixer actually emitted it: the probability the coded
    # row assigned to the symbol that was coded. Collected from the SAME hook the encoder
    # is fed from, so it is the shipped mixer's own opinion on the shipped prior and not a
    # re-derivation of one.
    q_probability: list[np.ndarray] = []
    q_frame: list[np.ndarray] = []

    def known_symbols(self, probabilities):
        positions = observed["positions"]
        if positions is None or len(positions) != len(probabilities):
            raise PriceError("known-symbol decode call lacks its group")
        symbols = field[observed["frame"]].reshape(-1)[positions].astype(np.int32)
        if collect_q:
            rows = np.asarray(probabilities)
            q_probability.append(rows[np.arange(len(symbols)), symbols].astype(np.float32))
            q_frame.append(np.full(len(symbols), observed["frame"], dtype=np.int16))
        for encoder in twins:
            encoder.encode(symbols, probabilities)
        observed["positions"] = None
        return symbols

    def end(self, plane, previous):
        # The output-lossless proof: the plane the receiver reconstructs from this
        # prior is the shipped field, frame by frame, for every treatment.
        np.testing.assert_array_equal(plane, field[observed["frame"]])
        original_end(self, plane, previous)
        observed["frame"] += 1

    def save(self, frame, tokens):
        if frame != observed["frame"] or observed["positions"] is not None:
            raise PriceError("encoder checkpoint is not at a frame boundary")
        if shutil.disk_usage(ROOT.parent).free < RESERVE_BYTES + (256 << 20):
            raise PriceError("STORAGE_BLOCK at the durable prior checkpoint")
        # Save the encoder FIRST: a crash before the receiver LATEST commits leaves
        # the previous matched pair intact and completed stages stay immutable.
        path = work / "encoder_states" / f"stage_{frame:04d}.npz"
        landed.ROOT = ROOT
        payload = landed.save(
            path, {f"enc{i}": np.frombuffer(e.snapshot(), dtype=np.uint8) for i, e in enumerate(twins)}
        )
        record(path.with_suffix(".json"), {"payload": payload, "binding": binding, "frame": frame})
        original_save(self, frame, tokens)

    LaneMixer.coding, LaneMixer.end_frame = coding, end
    NativeDecoder.decode, ReceiverCheckpoint.save = known_symbols, save
    tokens, _ = rx.decode_production_tokens(parts, renderer, code_dir, torch.device("cpu"))
    field_sha = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if observed["frame"] != 600 or field_sha != FIELD_SHA:
        raise PriceError("known-symbol loop did not process the full field")
    retain(work / "retained/encoded_field.u8", tokens.numpy().tobytes())
    if collect_q:
        landed.ROOT = ROOT
        payload = landed.save(
            work / "retained/coded_row_q.npz",
            {"q": np.concatenate(q_probability), "frame": np.concatenate(q_frame)},
        )
        record(
            work / "Q_COLLECTION.json",
            {
                "payload": payload,
                "symbols": int(sum(len(a) for a in q_probability)),
                "definition": (
                    "per coded symbol, the probability the shipped mixer's coded row assigned to the "
                    "symbol that was actually coded, read off the same hook the arithmetic encoder is "
                    "fed from; and the frame it belongs to, for the two-fold held-out split"
                ),
                "prior": "the treatment's own HPAC prior",
                "score_claim": False,
            },
        )
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
        changed["hpac"] = (work / f"retained/hpac.twin{index}.br").read_bytes()
        fields = list(rx.RX1_MODEL_HEADER.unpack(changed["header"]))
        fields[5] = len(changed["hpac"])
        changed["header"] = rx.RX1_MODEL_HEADER.pack(*fields)
        changed["tail"] = member["tail"][:96] + rider
        body = landed.io.join_member(changed)
        retain(work / f"retained/member.twin{index}.bin", body)
        archive = work / f"retained/archive.twin{index}.zip"
        landed.pack(body, archive, "stored", None)
        parsed = rx.read_residual_archive(archive)
        if parsed.hpac_blob != parts.hpac_blob or parsed.token_stream != raw:
            raise PriceError("archive parser mismatch")
        for name in ("semantic_blob", "carrier_blob", "tc1_weights", "residual_payload"):
            if getattr(parts, name) != getattr(parsed, name):
                raise PriceError(f"unrelated archive component changed: {name}")
        archives.append(fact(archive))
    if archives[0]["sha256"] != archives[1]["sha256"]:
        raise PriceError("tail/archive twins differ")
    if tag in ("control", "control47") and archives[0]["sha256"] != binding["base_archive_expected"]["sha256"]:
        raise PriceError("LIVE_LOOP_CONTROL_FAILED: no treatment prices are admissible")
    return record(
        work / "PRICE.json",
        {
            "binding": binding,
            "geometry_build": geometry,
            "n": 600,
            "twins": archives,
            "hpac_section_bytes": len(changed["hpac"]),
            "shipped_hpac_section_bytes": len(member["hpac"]),
            "token_stream_bytes": len(raw),
            "shipped_token_stream_bytes": len(member["tail"]) - 96 - 4 - len(bytes(parts.tc1_weights)),
            "delta_bytes_vs_base": archives[0]["bytes"] - binding["base_archive_expected"]["bytes"],
            "decoded_field_sha256": field_sha,
            "output_lossless": field_sha == FIELD_SHA,
            "public_decode_verified": False,
            "score_claim": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", choices=TREATMENTS, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--store-root", choices=tuple(STORE_ROOTS), default="vertigo")
    parser.add_argument("--checkpoint", type=Path, help="terminal EMA QAT checkpoint for a shape rung")
    parser.add_argument("--collect-q", action="store_true",
                        help="also retain the coded-row probability of each coded symbol, for the q rung")
    args = parser.parse_args()
    global ROOT
    ROOT = STORE_ROOTS[args.store_root]
    if args.resume_from.resolve() != (ROOT / args.treatment).resolve():
        raise PriceError("wrong resume root")
    if args.treatment not in ("control", "control47"):
        control_price = ROOT / "control/PRICE.json"
        if not control_price.exists():
            control_price = LEGACY_ROOT / "control/PRICE.json"
        proof = json.loads(control_price.read_text())
        if proof["twins"][0]["sha256"] != POINTER45_SHA:
            raise PriceError("live-loop control required before any treatment price")
    print(json.dumps(encode(args.treatment, args.checkpoint, args.collect_q)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
