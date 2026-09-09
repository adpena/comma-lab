#!/usr/bin/env python3
"""ddm_rw1 -- the JOINT renderer fold-back on the shipped object.

``research_only``.  Arm-local research code for the ddm_rw1 charter
(``.omx/research/charters/ddm_rw1_boundary_local_renderer_weight_foldback_per_pair_admitted_20260909.md``).
Nothing here ships; the levers are not DSL ``Lever`` factories because the trainer
is not a production launch path (the triality rule's research-code branch).

THE ACTUATOR, and why it is the int4 CODES and not the float weights
-------------------------------------------------------------------
ft1 measured the realization gap directly: its trainer used a UNIFORM int4
fake-quant with no row prune, while the deployed SM3R section uses per-tensor
depths ``{3, 4}`` and keeps 2 of 192 rows in ``blocks.{1,2,3}.film.weight``.  The
trained object and the realized object differed by ``2.32e-3`` in max abs
(``retained/verdict_ft1_step600.json``: ``export.trained_vs_realized_max_abs_delta``).

This arm removes that gap by construction rather than by measuring it: the
trainable parameter IS the shipped signed-int4 code, the shipped fp16 per-row
scales are FROZEN, and only tensors that are (a) depth 4 and (b) NOT row-pruned
are opened.  On the live section that is exactly

    head.weight          (3, 96, 3, 3)   2,592 codes
    blocks.3.dw.weight   (96, 1, 3, 3)     864 codes
    blocks.3.pw.weight   (96, 96, 1, 1)  9,216 codes

-- the edge-drawing layers gs3 Addendum 15 points at.  A trained code vector is
therefore byte-expressible in the shipped packer with zero re-quantization, and
"trained == realized" is an identity, not a measurement.

THE RATE, closed-form, before any build
---------------------------------------
Those codes are ALREADY in the archive.  A weight delta does not ADD 6,336 B; it
perturbs 6,336 B that are already counted, and the rate delta is the change in
the brotli'd semantic section.  fe1's ``model_section_edit_container_break_fee_v1``
prices a container-searched edit at ``0.15*N + 8`` B for N changed codes, so one
changed code costs ``0.15 * 6.658589531221714e-07 = 9.988e-08`` S and one repaired
seg cell buys ``8.477105034722222e-07`` S.  The break-even is therefore

    cells_repaired >= 0.1178 * codes_changed

which is why the number of CHANGED CODES -- not the number of trained tensors --
is the rate object this arm reports.

THE POSE TERM, closed-form
--------------------------
``up2.jacobian_and_residual`` returns a 6x12 Jacobian: twelve free carrier
coefficients against six scored pose dimensions.  Generically that map is
SURJECTIVE, so at first order a per-pair re-solve can cancel ANY pose residual a
render change produces -- which is the mechanism behind fe1's measured 643-3,053x
per-pair recovery, and it is why the STALE d_pose is the wrong thing to penalise
in the loop.  What the re-solve cannot buy is reach: the correcting step must
survive the shipped signed-int12 coefficient lattice.  The in-loop pose cost is
therefore the DEMANDED CARRIER STEP, and the stale d_pose is carried only as a
diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1
import ddm_up2_shipping_pose_solve as up2

# ----------------------------------------------------------------------------------
# THE LIVE POINTER BODY -- re-read, never assumed (the charter's re-base rule).
# ----------------------------------------------------------------------------------
POINTER_JSON = REPO / ".omx/state/canonical_frontier_pointer.json"

SJ1_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion")
LIVE_TREE = SJ1_ROOT / "candidate_pass3/candidate_runtime"
LIVE_RUNTIME = LIVE_TREE / "runtime"
LIVE_ARCHIVE = LIVE_TREE / "archive.zip"
LIVE_RAW = SJ1_ROOT / "candidate_pass3/parseback/0.raw"
LIVE_FIELD = SJ1_ROOT / "admission_pass3/field_admitted.npz"
LIVE_ARGMAX = SJ1_ROOT / "seg_final_pass3/argmax_n600.npy"

LIVE_ARCHIVE_SHA256 = (
    "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
)
LIVE_ARCHIVE_BYTES = 181_645
LIVE_SCORE_T4 = 0.13900437796841966
LIVE_D_SEG_T4 = 1.0913879636e-04
LIVE_D_SEG_LOCAL = 0.0001090664333767361
LIVE_D_SEG_CELLS = 12_866
LIVE_D_POSE = 5.0928018072772644e-06
SEG_T4_RATIO = 1.0006634761602033
#: The shipped SM3R body, byte-identical to the one ft1's identity gate passed on.
LIVE_SM3R_SHA256 = (
    "17e0fd0b197ac147afe98397ef38f02f7915b69372d03c042e6be6fa0f992e50"
)

GT_CACHE_DALI = Path(
    "/Volumes/VertigoDataTier/pact/ddm_chroma_dali_av_20260809/gt_cache_dali.pt"
)
GT_CACHE_DALI_SHA256 = (
    "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994"
)

WORK = Path("/Volumes/VertigoDataTier/pact/ddm_rw1_renderer_edge_foldback")
BULK = Path("/Volumes/APDataStore/pact/ddm_rw1_renderer_edge_foldback")

N_PAIRS = 600
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
SEG_CELLS_TOTAL = N_PAIRS * EVAL_H * EVAL_W  # 117,964,800
CARRIER_DIM = 12

RATE_PER_BYTE = 25.0 / 37_545_489.0  # 6.658589531221714e-07 S / archive byte
S_PER_SEG_CELL = 100.0 / SEG_CELLS_TOTAL  # 8.477105034722222e-07 S / flipped cell
#: fe1's container-searched marginal, ``model_section_edit_container_break_fee_v1``.
BYTES_PER_CHANGED_CODE = 0.15
CONTAINER_BREAK_FIXED_BYTES = 8.0

#: Signed int4 domain of the shipped depth-4 code runs.
CODE_MIN, CODE_MAX = -8, 7

#: The tensors this arm opens.  Depth 4 AND not row-pruned, so a trained code is
#: byte-expressible with zero re-quantization.  ``blocks.2`` is the ONE widening the
#: charter's falsifier (a) allows; it is not opened by default.
EDGE_TENSORS = ("head.weight", "blocks.3.dw.weight", "blocks.3.pw.weight")
WIDENED_TENSORS = (
    *EDGE_TENSORS,
    "blocks.2.dw.weight",
    "blocks.2.pw.weight",
)

#: The seg residual census (sj1 s16, measured on the live row).  Kept here as data so
#: the loss derivation below can be checked against it, not as a tunable.
CENSUS_ROW_BAND = (128, 320)  # rows [128, 320) carry 100.00 % of the residual
CENSUS_CLASS_SHARES = {  # GT-side share of the 12,866 residual cells
    "road": 5289 / 12866,
    "lane": 3025 / 12866,
    "undrivable": 2388 / 12866,
    "movable": 1740 / 12866,
    "mycar": 424 / 12866,
}


class Rw1Error(RuntimeError):
    """A ddm_rw1 precondition failed.  Always fail closed."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_live_pointer() -> dict[str, Any]:
    """Refuse unless the pointer still names the body this module is pinned to.

    sj1 pass 4 and pc2 are live on the same object; a silent pointer move would make
    every number below a delta against a baseline that no longer exists
    (``a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803``).
    """
    pointer = json.loads(POINTER_JSON.read_text())
    frontier = pointer["effective_frontier"]
    observed = sha256_file(LIVE_ARCHIVE)
    if observed != LIVE_ARCHIVE_SHA256:
        raise Rw1Error(
            f"live tree archive sha {observed} != pinned {LIVE_ARCHIVE_SHA256}"
        )
    moved = frontier["archive_sha256"] != LIVE_ARCHIVE_SHA256
    return {
        "pointer_archive_sha256": frontier["archive_sha256"],
        "pointer_score": frontier["score"],
        "pointer_lane_id": frontier.get("lane_id"),
        "pinned_archive_sha256": LIVE_ARCHIVE_SHA256,
        "pointer_moved_since_pin": bool(moved),
        "tree_archive_sha256": observed,
    }


def import_live(runtime_dir: Path = LIVE_RUNTIME):
    """The receiver's own modules, imported the way ``inflate.sh`` imports them."""
    for entry in (str(runtime_dir.parent), str(runtime_dir.parent / "cpr1")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    ra = importlib.import_module("runtime.residual_archive")
    rc1 = importlib.import_module("rc1_adaptive_model_sections")
    renderer = importlib.import_module("inflate")
    return ra, rc1, renderer


# ----------------------------------------------------------------------------------
# The semantic section, generalised from fe1's single-tensor form to N tensors
# ----------------------------------------------------------------------------------


@dataclass
class TensorRun:
    """Where one quantized tensor's scales and codes live in the SM3R body."""

    name: str
    shape: tuple[int, ...]
    bits: int
    count: int
    scale_offset: int
    scale_length: int
    code_offset: int
    code_length: int
    row_pruned: bool


@dataclass
class MultiSemanticSection:
    """The live semantic section with EVERY quantized tensor's run located.

    fe1's ``SemanticSection`` locates ``frame_embed.weight`` only.  A renderer weight
    fold-back edits ``head`` and ``blocks.3``, so the run table is generalised here --
    by DRIVING the receiver's own ``walk_sm3r`` exactly as fe1 does, never by a
    hand-rolled offset table.
    """

    rc1_stream: bytes
    shift: int
    sm3r_body: bytes
    runs: dict[str, TensorRun]
    codes: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    template: Any
    rc1: Any

    def body_with_codes(self, edits: dict[str, np.ndarray]) -> bytes:
        """The SM3R body with the named tensors' code runs replaced.

        Length-preserving by construction: the depth and count are the shipped ones,
        so every later offset in the body is unchanged and the walk still parses.
        """
        body = bytearray(self.sm3r_body)
        for name, codes in edits.items():
            run = self.runs.get(name)
            if run is None:
                raise Rw1Error(f"{name} is not a quantized run in this section")
            if run.row_pruned:
                raise Rw1Error(
                    f"{name} is row-pruned (2 of 192 rows kept); this arm refuses to "
                    "edit a pruned run because the trained object would not be the "
                    "realized one -- that is the ft1 realization gap"
                )
            flat = np.asarray(codes, dtype=np.int64).ravel()
            if flat.size != run.count:
                raise Rw1Error(
                    f"{name}: {flat.size} codes offered, shipped run holds {run.count}"
                )
            if flat.min() < CODE_MIN or flat.max() > CODE_MAX:
                raise Rw1Error(
                    f"{name}: codes escape the shipped signed {run.bits}-bit domain "
                    f"[{CODE_MIN}, {CODE_MAX}]"
                )
            packed = self.rc1.pack_signed_codes(flat.astype(np.int32), run.bits)
            if len(packed) != run.code_length:
                raise Rw1Error(
                    f"{name}: repacked run is {len(packed)} B, shipped run is "
                    f"{run.code_length} B"
                )
            body[run.code_offset : run.code_offset + run.code_length] = packed
        return bytes(body)

    def stream_with_codes(self, edits: dict[str, np.ndarray]) -> bytes:
        """The RC1 semantic stream carrying ``edits``, through the SHIPPED coder."""
        return self.rc1.apply_semantic(
            self.body_with_codes(edits), self.template, self.shift
        )

    def changed_code_count(self, edits: dict[str, np.ndarray]) -> int:
        total = 0
        for name, codes in edits.items():
            base = self.codes[name].ravel().astype(np.int64)
            total += int((np.asarray(codes, dtype=np.int64).ravel() != base).sum())
        return total

    def dequantized(self, edits: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        """State-dict values for the edited tensors, exactly as the receiver builds them."""
        import torch

        edits = edits or {}
        out: dict[str, Any] = {}
        for name, run in self.runs.items():
            if run.row_pruned:
                # A pruned run stores the KEPT rows, not the dense tensor, so the
                # dense shape is not recoverable here without the prune mask.  This
                # arm never edits a pruned run (``body_with_codes`` refuses), so the
                # honest thing is to omit it rather than reshape the wrong array.
                continue
            codes = np.asarray(
                edits.get(name, self.codes[name]), dtype=np.float32
            ).reshape(run.shape)
            scales = self.scales[name].astype(np.float32)
            scale_shape = [1] * len(run.shape)
            scale_shape[-1 if name.endswith("embed.weight") else 0] = scales.size
            out[name] = torch.from_numpy(codes) * torch.from_numpy(
                scales.reshape(scale_shape)
            )
        return out


def load_semantic_section(
    archive_path: Path = LIVE_ARCHIVE, runtime_dir: Path = LIVE_RUNTIME
) -> MultiSemanticSection:
    """Open the live semantic section and locate EVERY quantized tensor's run.

    Refused unless ``apply_semantic(restore_semantic(stream)) == stream`` byte for
    byte: the coder this arm prices with must be the coder the archive ships.
    """
    ra, rc1, renderer = import_live(runtime_dir)
    parts = ra.read_residual_archive(Path(archive_path))
    stream = bytes(parts.semantic_blob)
    if not stream.startswith(rc1.SEMANTIC_MAGIC):
        raise Rw1Error("live semantic section does not carry the RC1 rider")
    _magic, _version, shift, _payload_len = rc1.RC1_HEADER.unpack_from(stream)
    template = renderer.SemanticTokenRenderer(96).state_dict()
    body = rc1.restore_semantic(stream, template)
    if rc1.apply_semantic(body, template, shift) != stream:
        raise Rw1Error("RC1 semantic round-trip is not byte-identical")
    observed = hashlib.sha256(body).hexdigest()
    if observed != LIVE_SM3R_SHA256:
        raise Rw1Error(f"SM3R body sha {observed} != pinned {LIVE_SM3R_SHA256}")

    version, mode, keep_percent, reserved = body[4:8]
    cursor = 10
    offsets: list[int] = []

    def read(_kind: str, length: int) -> bytes:
        nonlocal cursor
        offsets.append(cursor)
        chunk = body[cursor : cursor + length]
        cursor += length
        return chunk

    plan = rc1.walk_sm3r(read, template, (version, mode, keep_percent, reserved))
    if cursor != len(body):
        raise Rw1Error(f"SM3R walk ended at {cursor} of {len(body)}")

    runs: dict[str, TensorRun] = {}
    codes: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    index = 1  # plan[0] is the depth table
    for name, value in template.items():
        if value.ndim < 2:
            index += 1
            continue
        pruned = name in rc1.ROW_PRUNE_NAMES
        if pruned:
            index += 1  # the prune mask
        scale_item, scale_off = plan[index], offsets[index]
        index += 1
        code_item, code_off = plan[index], offsets[index]
        index += 1
        if code_item["kind"] != "codes":
            raise Rw1Error(f"{name}: expected a codes run, got {code_item['kind']}")
        run = TensorRun(
            name=name,
            shape=tuple(int(dim) for dim in value.shape),
            bits=int(code_item["bits"]),
            count=int(code_item["count"]),
            scale_offset=int(scale_off),
            scale_length=int(scale_item["length"]),
            code_offset=int(code_off),
            code_length=int(code_item["length"]),
            row_pruned=bool(pruned),
        )
        runs[name] = run
        scales[name] = np.frombuffer(
            body[run.scale_offset : run.scale_offset + run.scale_length], dtype="<f2"
        ).astype(np.float32)
        raw_codes = rc1.unpack_signed_codes(
            body[run.code_offset : run.code_offset + run.code_length],
            run.count,
            run.bits,
        )
        codes[name] = (
            np.asarray(raw_codes).reshape(run.shape)
            if not pruned
            else np.asarray(raw_codes)
        )

    return MultiSemanticSection(
        rc1_stream=stream,
        shift=int(shift),
        sm3r_body=body,
        runs=runs,
        codes=codes,
        scales=scales,
        template=template,
        rc1=rc1,
    )


def trainable_names(widened: bool) -> tuple[str, ...]:
    return WIDENED_TENSORS if widened else EDGE_TENSORS


def check_trainable(section: MultiSemanticSection, names: tuple[str, ...]) -> None:
    for name in names:
        run = section.runs.get(name)
        if run is None:
            raise Rw1Error(f"{name} has no quantized run in the live section")
        if run.row_pruned:
            raise Rw1Error(f"{name} is row-pruned; refusing (ft1 realization gap)")
        if run.bits != 4:
            raise Rw1Error(
                f"{name} ships at {run.bits} bits, not 4; the int4 code domain "
                f"[{CODE_MIN}, {CODE_MAX}] this arm trains in would be wrong"
            )


# ----------------------------------------------------------------------------------
# The live renderer + tokens
# ----------------------------------------------------------------------------------


def load_live_tokens(path: Path = LIVE_FIELD) -> np.ndarray:
    """The token field the LIVE receiver decodes (sj1's admitted pass-3 subset).

    ft1's ``cache_input_shipped_tokens.pt`` is the g8s-generation field and is a
    DIFFERENT object; conditioning on it would train against a render the pointer
    does not produce.  sj1 stores the admitted field as one npz plane per pair keyed
    by the pair index as a string, and a MISSING key silently reverts that pair to a
    different field (sj1 sec.17's fourth silent-revert class), so every pair is
    required rather than defaulted.
    """
    with np.load(path) as blob:
        planes = []
        for pair in range(N_PAIRS):
            key = str(pair)
            if key not in blob:
                raise Rw1Error(f"token field {path} has no plane for pair {pair}")
            plane = blob[key]
            if plane.shape != (EVAL_H, EVAL_W) or plane.dtype != np.uint8:
                raise Rw1Error(
                    f"pair {pair} plane is {plane.shape}/{plane.dtype}, expected "
                    f"({EVAL_H}, {EVAL_W})/uint8"
                )
            planes.append(plane)
    return np.ascontiguousarray(np.stack(planes))


def load_live_renderer(section: MultiSemanticSection | None = None):
    """The shipped ``SemanticTokenRenderer`` carrying the live weights."""
    _ra, _rc1, renderer = import_live()
    section = section or load_semantic_section()
    model = renderer.SemanticTokenRenderer(96)
    state = model.state_dict()
    tagged = renderer.unpack_variant_semantic_or_none(section.rc1_stream, state)
    if tagged is None:
        raise Rw1Error("live semantic section did not decode through the receiver")
    model.load_state_dict(tagged, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_gt_seg_dali() -> np.ndarray:
    """The DALI seg table, the lineage ``contest_cuda`` is actually scored against."""
    import torch

    observed = sha256_file(GT_CACHE_DALI)
    if observed != GT_CACHE_DALI_SHA256:
        raise Rw1Error(f"DALI GT sha {observed} != pinned {GT_CACHE_DALI_SHA256}")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    cache = torch.load(GT_CACHE_DALI, map_location="cpu", weights_only=False)
    seg = np.asarray(cache["seg"], dtype=np.uint8)
    if seg.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise Rw1Error(f"DALI seg has shape {seg.shape}, expected 600x384x512")
    return seg


# ----------------------------------------------------------------------------------
# mode=control -- the lifted forward must reproduce the SHIPPED frames byte for byte
# ----------------------------------------------------------------------------------

IDENTITY_PAIRS = (0, 1, 2, 137, 299, 300, 450, 599)


def cmd_control(args) -> int:
    """Identity control: re-render the live tokens and compare to the live decode.

    This is ft1's gate re-run against THIS body.  It is not inherited: the live tree
    ships sj1's pass-3 token field, so the frames differ from ft1's even though the
    semantic section is byte-identical.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    section = load_semantic_section()
    check_trainable(section, trainable_names(bool(args.widened)))
    tokens = load_live_tokens()
    model = load_live_renderer(section)

    indices = np.array(IDENTITY_PAIRS, dtype=np.int64)
    rendered = jg1.render_frame1(model, tokens[indices], indices)
    raw = up2.open_raw(LIVE_RAW, verify_sha=False)
    shipped = np.asarray(raw[2 * indices + 1])
    delta = np.abs(rendered.astype(np.int16) - shipped.astype(np.int16))

    runs = {
        name: {
            "shape": list(section.runs[name].shape),
            "codes": section.runs[name].count,
            "bits": section.runs[name].bits,
            "code_bytes": section.runs[name].code_length,
            "scale_bytes": section.runs[name].scale_length,
            "code_offset": section.runs[name].code_offset,
            "row_pruned": section.runs[name].row_pruned,
        }
        for name in trainable_names(bool(args.widened))
    }
    trainable_codes = sum(item["codes"] for item in runs.values())

    result = {
        "schema": "ddm_rw1_control.v1",
        "axis": "[macOS-CPU advisory; identity control, no score]",
        "score_claim": False,
        "pointer": pointer,
        "pairs": len(indices),
        "pair_ids": [int(p) for p in indices],
        "pixels_compared": int(shipped.size),
        "pixels_changed": int((delta > 0).sum()),
        "max_abs_delta": int(delta.max()),
        "byte_exact": bool(delta.max() == 0),
        "semantic_batch": 1,
        "sm3r_body_bytes": len(section.sm3r_body),
        "sm3r_body_sha256": hashlib.sha256(section.sm3r_body).hexdigest(),
        "rc1_stream_bytes": len(section.rc1_stream),
        "trainable_tensors": runs,
        "trainable_codes": trainable_codes,
        "rate_break_even_cells_per_code": BYTES_PER_CHANGED_CODE
        * RATE_PER_BYTE
        / S_PER_SEG_CELL,
        "token_field": str(LIVE_FIELD),
        "token_field_sha256": sha256_file(LIVE_FIELD),
        "raw": str(LIVE_RAW),
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    if not result["byte_exact"]:
        raise Rw1Error(
            "the lifted forward does not reproduce the shipped frames byte for byte; "
            "every later realized number would be realized against a different model"
        )
    return 0


# ----------------------------------------------------------------------------------
# mode=section -- the null build: repacking the SHIPPED codes must be byte-identical
# ----------------------------------------------------------------------------------


def cmd_section(args) -> int:
    """Null control on the packer seam: shipped codes in, shipped bytes out.

    Without this, a later "the archive grew N bytes" number cannot be attributed to
    the weight delta rather than to the repack.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)

    identity_edits = {name: section.codes[name] for name in names}
    body = section.body_with_codes(identity_edits)
    stream = section.stream_with_codes(identity_edits)
    result = {
        "schema": "ddm_rw1_section.v1",
        "axis": "[exact bytes; packer identity control]",
        "score_claim": False,
        "pointer": pointer,
        "body_byte_identical": body == section.sm3r_body,
        "stream_byte_identical": stream == section.rc1_stream,
        "body_bytes": len(body),
        "stream_bytes": len(stream),
        "changed_codes_on_identity": section.changed_code_count(identity_edits),
        "runs": {
            name: {
                "codes": section.runs[name].count,
                "code_bytes": section.runs[name].code_length,
                "bits": section.runs[name].bits,
                "scales": int(section.scales[name].size),
                "scale_min": float(section.scales[name].min()),
                "scale_max": float(section.scales[name].max()),
                "code_min": int(np.asarray(section.codes[name]).min()),
                "code_max": int(np.asarray(section.codes[name]).max()),
                "code_zero_fraction": float(
                    (np.asarray(section.codes[name]) == 0).mean()
                ),
            }
            for name in names
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    if not (result["body_byte_identical"] and result["stream_byte_identical"]):
        raise Rw1Error(
            "repacking the SHIPPED codes did not reproduce the shipped bytes; the "
            "packer seam is not an identity and no byte delta below is attributable"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(node: argparse.ArgumentParser) -> None:
        node.add_argument("--widened", action="store_true")
        node.add_argument("--threads", type=int, default=3)

    control = sub.add_parser("control")
    control.add_argument("--out", type=Path, default=WORK / "receipts/CONTROL.json")
    common(control)
    control.set_defaults(func=cmd_control)

    section = sub.add_parser("section")
    section.add_argument("--out", type=Path, default=WORK / "receipts/SECTION.json")
    common(section)
    section.set_defaults(func=cmd_section)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    threads = int(getattr(args, "threads", 3) or 3)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(threads))
    import torch

    torch.set_num_threads(threads)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
