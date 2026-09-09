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
import math
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
RC2_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing")

#: The 33rd pointer move (rc2, 2026-09-09).  ONLY the hpac model section and its
#: reader changed (-231 B).  MEASURED here, not assumed: against the pass-3 tree the
#: semantic (31,792 B), carrier (18,931 B), token stream (120,225 B) and residual
#: payload (100 B) sections are BYTE-IDENTICAL and only ``hpac_blob`` differs
#: (16,267 -> 16,061).  rc2's T4 row reproduced d_seg 0.00010913 and d_pose 5.1e-6
#: exactly, which is the authority receipt that the DECODED token field -- and hence
#: every render this arm measures -- is the same object.  So the pass-3 parse-back
#: decode is reused as the live decode rather than re-inflated, and the pinned
#: d_seg / d_pose / cell count carry across the move unchanged.
LIVE_TREE = RC2_ROOT / "candidate_runtime"
LIVE_RUNTIME = LIVE_TREE / "runtime"
LIVE_ARCHIVE = LIVE_TREE / "archive.zip"
LIVE_RAW = SJ1_ROOT / "candidate_pass3/parseback/0.raw"
LIVE_FIELD = SJ1_ROOT / "admission_pass3/field_admitted.npz"
LIVE_ARGMAX = SJ1_ROOT / "seg_final_pass3/argmax_n600.npy"
#: The sections the live decode is shared with, and the fingerprint that says so.
#:
#: NARROWED at the 34th move (pc2, carrier-only): the reuse claim this arm actually
#: needs is about the ODD frames.  Frame 2p+1 is the renderer's output and depends on
#: the SEMANTIC weights, the TOKEN field and the residual table; frame 2p is the pose
#: carrier and this arm never reads it -- the seg leg scores only the last frame of the
#: pair (``upstream/modules.py:105``) and the pose leg RENDERS frame 0 from the carrier
#: CODES rather than reading it from the raw (``ddm_br1.evaluate_codes`` ->
#: ``up2.render_frame0``).  So a carrier-only pointer move does not invalidate a
#: retained decode for either leg, and requiring carrier identity would refuse a body
#: that is in fact usable.  The carrier is still reported, as information.
SHARED_SECTION_SOURCE = SJ1_ROOT / "candidate_pass3/candidate_runtime/archive.zip"
SHARED_SECTIONS = ("semantic_blob", "token_stream", "residual_payload")
SHARED_SECTIONS_INFORMATIONAL = ("carrier_blob", "hpac_blob")

LIVE_ARCHIVE_SHA256 = (
    "c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e"
)
LIVE_ARCHIVE_BYTES = 181_414
LIVE_SCORE_T4 = 0.13885056455024844
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

#: fe1's bar: a candidate must buy at least this much S to be worth a paid row.
ADMIT_BAR = -2e-5

#: fe1's WORST measured per-pair carrier re-solve recovery (pair 382, stale/resolved).
#: Used as a conservative discount, never as the expected value.
FE1_WORST_RECOVERY = 643.0

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


#: The sub-0.12 target the whole campaign is aimed at.  Every verdict this module emits
#: carries BOTH the score delta AND its share of the remaining gap, because a magnitude
#: without its relative significance is not a verdict (MAIN, 2026-09-09).
TARGET_SCORE = 0.12


def _stake(cells: int) -> dict[str, float]:
    """Score stake of ``cells`` flipped cells, absolutely and as a share of the gap."""
    gap = LIVE_SCORE_T4 - TARGET_SCORE
    delta = cells * S_PER_SEG_CELL
    return {
        "dS": delta,
        "dS_share_of_gap_to_target": delta / gap if gap else float("nan"),
        "gap_to_target": gap,
        "operating_point_S": LIVE_SCORE_T4,
    }


class Rw1Error(RuntimeError):
    """A ddm_rw1 precondition failed.  Always fail closed."""


#: LATE-BOUND live-body pin.  The constants above are the body this module was written
#: against; when the pointer moves, ``rebase`` MEASURES the new tree and writes this
#: file, and every later import reads it.  Two reasons it is a file and not an edit:
#: (1) hand-typing shas and scores into a module is the hand-error genus the
#: pointer-move packet exists to remove; (2) a default captured at parser-build time is
#: bound BEFORE a re-base can act, which is the early-binding trap pc2's r=12 identity
#: control caught the hard way.
LIVE_PIN_PATH = WORK / "LIVE_PIN.json"
_PIN_PATH_KEYS = (
    "LIVE_TREE",
    "LIVE_RAW",
    "LIVE_FIELD",
    "LIVE_ARGMAX",
    "SHARED_SECTION_SOURCE",
)
_PIN_VALUE_KEYS = (
    "LIVE_ARCHIVE_SHA256",
    "LIVE_ARCHIVE_BYTES",
    "LIVE_SCORE_T4",
    "LIVE_D_SEG_T4",
    "LIVE_D_SEG_LOCAL",
    "LIVE_D_SEG_CELLS",
    "LIVE_D_POSE",
    "SEG_T4_RATIO",
    "LIVE_SM3R_SHA256",
)


def _apply_live_pin() -> dict[str, Any] | None:
    """Override the live-body constants from the pin file, if one has been written."""
    if not LIVE_PIN_PATH.is_file():
        return None
    pin = json.loads(LIVE_PIN_PATH.read_text())
    globals_ = globals()
    for key in _PIN_PATH_KEYS:
        if key in pin:
            globals_[key] = Path(pin[key])
    for key in _PIN_VALUE_KEYS:
        if key in pin:
            globals_[key] = pin[key]
    globals_["LIVE_RUNTIME"] = globals_["LIVE_TREE"] / "runtime"
    globals_["LIVE_ARCHIVE"] = globals_["LIVE_TREE"] / "archive.zip"
    return pin


LIVE_PIN = _apply_live_pin()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_shared_decode_sections() -> dict[str, Any]:
    """The live tree must share the decode-determining sections with LIVE_RAW's tree.

    rc2's move rewrote ONLY the hpac model section, so the pass-3 parse-back decode is
    reused instead of re-inflating.  That reuse is a claim about bytes, so it is
    CHECKED here rather than inherited from the move's memo: if any of the semantic,
    carrier, token-stream or residual sections ever diverges, the raw this arm
    measures against is a different object and every realized number would be
    unanchored.
    """
    ra, _rc1, _renderer = import_live()
    live = ra.read_residual_archive(LIVE_ARCHIVE)
    source = ra.read_residual_archive(SHARED_SECTION_SOURCE)
    report: dict[str, Any] = {}
    mismatched = []
    for field in SHARED_SECTIONS:
        same = bytes(getattr(live, field)) == bytes(getattr(source, field))
        report[field] = {
            "identical": bool(same),
            "bytes": len(getattr(live, field)),
            "required": True,
        }
        if not same:
            mismatched.append(field)
    if mismatched:
        raise Rw1Error(
            f"the live tree no longer shares {mismatched} with the tree that produced "
            f"{LIVE_RAW}; those sections determine the ODD frames this arm measures, so "
            "re-inflate before measuring anything against that decode"
        )
    for field in SHARED_SECTIONS_INFORMATIONAL:
        report[field] = {
            "identical": bytes(getattr(live, field)) == bytes(getattr(source, field)),
            "bytes_live": len(getattr(live, field)),
            "bytes_source": len(getattr(source, field)),
            "required": False,
        }
    report["raw_source_tree"] = str(SHARED_SECTION_SOURCE)
    return report


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
    shared = verify_shared_decode_sections()
    return {
        "shared_decode_sections": shared,
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

    def body_with_scales(
        self, code_edits: dict[str, np.ndarray], scale_edits: dict[str, np.ndarray]
    ) -> bytes:
        """The SM3R body with named code runs AND named fp16 SCALE runs replaced.

        A scale is 2 bytes and multiplies a whole row, so this is length-preserving for
        the same reason the code path is: the shipped run's geometry is reused, never
        recomputed.  The written value is round-tripped through float16 first, because
        the archive can only carry an fp16 and a value that is not one would silently
        become a different object at parse-back.
        """
        body = bytearray(self.body_with_codes(code_edits))
        for name, scales in scale_edits.items():
            run = self.runs.get(name)
            if run is None:
                raise Rw1Error(f"{name} is not a quantized run in this section")
            if run.row_pruned:
                raise Rw1Error(f"{name} is row-pruned; refusing (ft1 realization gap)")
            values = np.asarray(scales, dtype=np.float16)
            if values.size != self.scales[name].size:
                raise Rw1Error(
                    f"{name}: {values.size} scales offered, shipped run holds "
                    f"{self.scales[name].size}"
                )
            if not np.all(np.isfinite(values.astype(np.float32))):
                raise Rw1Error(f"{name}: non-finite scale")
            packed = values.astype("<f2").tobytes()
            if len(packed) != run.scale_length:
                raise Rw1Error(
                    f"{name}: repacked scales are {len(packed)} B, shipped run is "
                    f"{run.scale_length} B"
                )
            body[run.scale_offset : run.scale_offset + run.scale_length] = packed
        return bytes(body)

    def stream_with_scales(
        self, code_edits: dict[str, np.ndarray], scale_edits: dict[str, np.ndarray]
    ) -> bytes:
        return self.rc1.apply_semantic(
            self.body_with_scales(code_edits, scale_edits), self.template, self.shift
        )

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


# ----------------------------------------------------------------------------------
# mode=prep -- the per-pair pose geometry the in-loop term needs, measured ONCE
# ----------------------------------------------------------------------------------


def _carrier_reach_budget(codes: np.ndarray) -> np.ndarray:
    """How far each pair's twelve codes can move before the int12 lattice ends.

    ``COEFF_CODE_MIN/MAX`` are the shipped signed-int12 bounds (up2:68).  The
    re-solve's correcting step is only realisable inside them, so this is the
    physical reach the in-loop pose barrier is written against -- not a tuned number.
    """
    low = codes.astype(np.float64) - up2.COEFF_CODE_MIN
    high = up2.COEFF_CODE_MAX - codes.astype(np.float64)
    return np.minimum(low, high).min(axis=1)


def cmd_prep(args) -> int:
    """Per-pair pose geometry: base pose, target, Jacobian, re-solve operator, floor.

    ``up2.jacobian_and_residual`` returns the 6x12 map from the pair's twelve carrier
    coefficients to its six scored pose dimensions.  Twelve knobs against six
    constraints is generically SURJECTIVE, so the min-image-norm re-solve

        dc = G^-1 J^T (J G^-1 J^T)^-1 (-dr)

    cancels ANY first-order pose residual a render change produces.  What it cannot
    do is leave the lattice, so this stage stores (a) ``A = dc/dr`` in CODE units --
    the operator the in-loop reach barrier uses -- and (b) the closed-form
    LATTICE FLOOR, the d_pose that survives after a perfect re-solve because the
    realised codes are rounded.  The floor is a PREDICTION of the post-re-solve pose
    leg, written before any training step.
    """
    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    state = up2.load_carrier_state(LIVE_TREE, verify_archive=False)
    targets, lineage = up2.load_gt_poses(GT_CACHE_DALI)
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    raw = up2.open_raw(LIVE_RAW, verify_sha=False)

    import ddm_br1_pose_basis_reorientation as br1

    blow = br1.low_basis(state)
    gram, _bmat = br1.span_gram(blow)
    gram = gram.double()
    ginv = torch.linalg.inv(gram)
    scales = state.coefficient_scales.double().numpy()  # (12,)

    pairs = np.arange(N_PAIRS, dtype=np.int64)
    batch = int(args.batch)
    jac_all = np.zeros((N_PAIRS, up2.POSE_DIMS, CARRIER_DIM), dtype=np.float64)
    pose_all = np.zeros((N_PAIRS, up2.POSE_DIMS), dtype=np.float64)
    for start in range(0, N_PAIRS, batch):
        index = pairs[start : start + batch]
        frame1 = up2.frames_to_bchw(raw[2 * index + 1])
        coeff = state.coefficients[index]
        tb = torch.from_numpy(targets[index]).float()
        jac, _res, pose = up2.jacobian_and_residual(
            posenet, state, coeff, frame1, tb, index
        )
        jac_all[index] = jac.double().numpy()
        pose_all[index] = pose.double().numpy()
        if args.progress:
            print(
                f"prep jacobian {min(start + batch, N_PAIRS)}/{N_PAIRS} "
                f"in {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    # A maps a pose residual dr to the min-image-norm carrier step in CODE units.
    a_all = np.zeros((N_PAIRS, CARRIER_DIM, up2.POSE_DIMS), dtype=np.float64)
    rank_deficient = []
    floor = np.zeros(N_PAIRS, dtype=np.float64)
    ginv_np = ginv.numpy()
    for pair in range(N_PAIRS):
        jac = jac_all[pair]
        middle = jac @ ginv_np @ jac.T
        if np.linalg.matrix_rank(middle, tol=1e-12) < up2.POSE_DIMS:
            rank_deficient.append(int(pair))
            operator = np.zeros((CARRIER_DIM, up2.POSE_DIMS))
        else:
            operator = ginv_np @ jac.T @ np.linalg.inv(middle)
        a_all[pair] = operator / scales[:, None]
        # Lattice floor: after a perfect re-solve the realised codes are rounded, so a
        # uniform +-0.5-code error remains.  E||J d||^2 / 6 with d ~ U(-.5,.5)*scales
        # per coefficient and independent components gives (1/12)*scales^2 per dim.
        floor[pair] = float(
            ((jac * scales[None, :]) ** 2).sum() / 12.0 / up2.POSE_DIMS
        )

    reach = _carrier_reach_budget(np.asarray(state.codes))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "pose_geometry.npz",
        jacobian=jac_all.astype(np.float32),
        resolve_operator_codes=a_all.astype(np.float32),
        pose_base=pose_all.astype(np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        coefficient_scales=scales.astype(np.float32),
        codes=np.asarray(state.codes, dtype=np.int32),
        reach_budget=reach.astype(np.float32),
        lattice_floor=floor.astype(np.float32),
    )
    d_pose_base = float(((pose_all - targets) ** 2).mean(axis=1).mean())
    result = {
        "schema": "ddm_rw1_prep.v1",
        "axis": "[cpu_torch fp32 authority, n600; pose geometry]",
        "score_claim": False,
        "pointer": pointer,
        "pairs": N_PAIRS,
        "gt_lineage": lineage,
        "d_pose_base_recomputed": d_pose_base,
        "d_pose_base_live_receipt": LIVE_D_POSE,
        "d_pose_base_relative_error": abs(d_pose_base - LIVE_D_POSE)
        / max(LIVE_D_POSE, 1e-30),
        "rank_deficient_pairs": rank_deficient,
        "reach_budget_codes": {
            "min": float(reach.min()),
            "median": float(np.median(reach)),
            "max": float(reach.max()),
        },
        "naive_rounding_pose_bound": {
            "mean": float(floor.mean()),
            "median": float(np.median(floor)),
            "max": float(floor.max()),
            "over_base": float(floor.mean() / LIVE_D_POSE),
            "reading": (
                "UPPER bound for a re-solve that only ROUNDS to the lattice.  The "
                "INCUMBENT falsifies it as a prediction: the live row already sits at "
                "d_pose 5.0928e-06 on this same lattice, i.e. below this bound, so "
                "jg5's +-2 integer polish beats uniform rounding by the ratio in "
                "``over_base``.  The operative expectation for the post-re-solve leg "
                "is fe1's MEASURED per-pair recovery (0.24-1.36x base), not this bound"
            ),
        },
        "incumbent_falsifies_naive_bound": bool(floor.mean() > LIVE_D_POSE),
        "naive_bound_looseness_vs_incumbent": float(floor.mean() / LIVE_D_POSE),
        "live_pose_leg_S": float(math.sqrt(10.0 * LIVE_D_POSE)),
        "geometry_path": str(out_dir / "pose_geometry.npz"),
        "geometry_sha256": sha256_file(out_dir / "pose_geometry.npz"),
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt = Path(args.out)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# mode=train -- the joint fold-back, in the shipped int4 code domain
# ----------------------------------------------------------------------------------
#: The reference temperature every reported surrogate is read at.  sd1 measured that an
#: annealing tau DEFLATES the surrogate, so a loss curve read at the live tau is not a
#: curve of the same quantity twice
#: (``tau_anneal_deflates_the_surrogate_read_losses_at_fixed_tau_20260904``).
TAU_REFERENCE = 0.10
TAU_START, TAU_END = 0.15, 0.05


def _ste_round(value):
    import torch

    return value + (torch.round(value) - value).detach()


def _ste_uint8(value):
    """clamp(0,255) then round, with a straight-through gradient.

    This is the receiver's own ``clamp(0.0, 255.0).round()`` at
    ``cpr1/inflate.py:322-323``; the STE only supplies the gradient the round does
    not have.
    """
    import torch

    clamped = value.clamp(0.0, 255.0)
    return clamped + (torch.round(clamped) - clamped).detach()


class CodeFoldBack:
    """The trainable object: signed int4 codes with the SHIPPED scales frozen."""

    def __init__(self, section, names, device):
        import torch

        self.names = tuple(names)
        self.device = device
        self.base_codes = {
            name: torch.from_numpy(
                np.asarray(section.codes[name], dtype=np.float32)
            ).to(device)
            for name in self.names
        }
        self.scales = {}
        for name in self.names:
            run = section.runs[name]
            scale_shape = [1] * len(run.shape)
            scale_shape[0] = section.scales[name].size
            self.scales[name] = (
                torch.from_numpy(section.scales[name].astype(np.float32))
                .reshape(scale_shape)
                .to(device)
            )
        self.latent = {
            name: torch.nn.Parameter(self.base_codes[name].clone())
            for name in self.names
        }

    def parameters(self):
        return list(self.latent.values())

    def codes(self, latent=None):
        import torch

        latent = latent or self.latent
        return {
            name: torch.clamp(_ste_round(latent[name]), CODE_MIN, CODE_MAX)
            for name in self.names
        }

    def weights(self, latent=None):
        codes = self.codes(latent)
        return {name: codes[name] * self.scales[name] for name in self.names}

    def realized_codes(self, latent=None) -> dict[str, np.ndarray]:
        import torch

        latent = latent or self.latent
        with torch.no_grad():
            return {
                name: torch.clamp(torch.round(latent[name]), CODE_MIN, CODE_MAX)
                .to("cpu")
                .numpy()
                .astype(np.int64)
                for name in self.names
            }

    def changed_codes(self, latent=None) -> int:
        realized = self.realized_codes(latent)
        total = 0
        for name in self.names:
            base = self.base_codes[name].to("cpu").numpy().astype(np.int64)
            total += int((realized[name] != base).sum())
        return total


def _render_eval(model, fold, tokens_batch, index_batch, latent=None):
    """The renderer's (b, 3, 384, 512) output with the folded-back weights."""
    from torch.func import functional_call

    overrides = fold.weights(latent)
    return functional_call(model, overrides, (tokens_batch, index_batch))


def _exact_r_camera(frame_eval):
    """EVAL -> camera, exactly the receiver's bilinear + clamp/round, STE gradient."""
    import torch.nn.functional as functional

    up = functional.interpolate(
        frame_eval, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )
    return _ste_uint8(up)


def _seg_logits(segnet, camera_bchw):
    """SegNet logits through the evaluator's own preprocess (modules.py:105-107)."""
    return segnet(segnet.preprocess_input(camera_bchw.unsqueeze(1)))


def _expected_flip(logits, labels, tau: float):
    """``sigmoid(-margin / tau)`` -- the expected-flip surrogate for d_seg.

    ``margin = gt_logit - max(other logit)``.  It needs no band mask and no class
    weight: the sigmoid is already the at-risk selector (a confident cell contributes
    ~0), and one flipped cell costs the SAME 8.477e-07 S whatever its class, so a
    per-class multiplier would optimise a different objective than S.  The census's
    Lane 40.15x / Movable 10.92x enrichment is therefore a DIAGNOSTIC here, not a
    weight -- deriving the weight from the score, not from the census, is the
    closed-form-first answer.
    """
    import torch

    gt = torch.gather(logits, 1, labels.unsqueeze(1).long()).squeeze(1)
    masked = logits.scatter(
        1, labels.unsqueeze(1).long(), torch.full_like(gt.unsqueeze(1), -1e30)
    )
    other = masked.max(dim=1).values
    return torch.sigmoid(-(gt - other) / tau)


@dataclass
class PoseGeometry:
    resolve_operator: Any  # (600, 12, 6) float32 -- dr -> carrier step in CODE units
    pose_base: Any  # (600, 6)
    targets: Any  # (600, 6) -- the DALI GT poses the scorer measures against
    reach_budget: Any  # (600,)
    lattice_floor: Any  # (600,) naive-rounding bound; see cmd_prep's reading


def load_pose_geometry(path: Path, device):
    import torch

    with np.load(path) as blob:
        return PoseGeometry(
            resolve_operator=torch.from_numpy(
                np.asarray(blob["resolve_operator_codes"], dtype=np.float32)
            ).to(device),
            pose_base=torch.from_numpy(
                np.asarray(blob["pose_base"], dtype=np.float32)
            ).to(device),
            targets=torch.from_numpy(
                np.asarray(blob["targets"], dtype=np.float32)
            ).to(device),
            reach_budget=torch.from_numpy(
                np.asarray(blob["reach_budget"], dtype=np.float32)
            ).to(device),
            lattice_floor=torch.from_numpy(
                np.asarray(blob["lattice_floor"], dtype=np.float32)
            ).to(device),
        )


def _ema_decay_from_run_geometry(steps: int) -> float:
    """``ema_decay_run_geometry_v1``: the decay follows the RUN's geometry.

    The shadow's effective window is one fifth of the horizon, so
    ``decay = 1 - 5/steps``.  Deriving it here rather than importing ft1's
    0.9974448421062369 is deliberate: that value is this LawRef evaluated on a
    1,800-step run, and a decay carried across a different horizon is exactly the
    transferred-constant class ([[m21]] constants -> laws).
    """
    if steps < 25:
        raise Rw1Error(f"a {steps}-step run has no EMA geometry to derive from")
    return float(1.0 - 5.0 / steps)


def _evaluate_realized(
    model, fold, segnet, tokens, labels, device, *, latent=None, batch: int = 4
) -> dict[str, Any]:
    """Realized argmax flips over ALL 600 pairs through the trainer's exact R.

    ``[macOS-MPS research-signal]`` when ``device`` is mps: this is the checkpoint
    SELECTOR, never a verdict.  The verdict re-renders at batch 1 on cpu_torch and
    runs sj1's own n600 instrument on the decoded bytes.
    """
    import torch

    flips = 0
    per_pair = np.zeros(N_PAIRS, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, N_PAIRS, batch):
            index = np.arange(start, min(start + batch, N_PAIRS), dtype=np.int64)
            tokens_batch = torch.from_numpy(tokens[index].astype(np.int64)).to(device)
            index_batch = torch.from_numpy(index).to(device)
            frame = _render_eval(model, fold, tokens_batch, index_batch, latent)
            logits = _seg_logits(segnet, _exact_r_camera(frame))
            argmax = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
            wrong = (argmax != labels[index]).reshape(len(index), -1).sum(axis=1)
            per_pair[index] = wrong
            flips += int(wrong.sum())
    return {
        "flips": int(flips),
        "d_seg": float(flips / SEG_CELLS_TOTAL),
        "per_pair": per_pair,
    }


def cmd_train(args) -> int:
    """The joint fold-back.  Resumable, per-stage checkpoints, EMA shadow saved."""
    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    model = load_live_renderer(section).to(device)
    tokens = load_live_tokens()
    labels = load_gt_seg_dali()
    segnet = jg1.load_segnet().to(device).eval()
    for param in segnet.parameters():
        param.requires_grad_(False)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    posenet = posenet.to(device).eval()
    for param in posenet.parameters():
        param.requires_grad_(False)
    geom = load_pose_geometry(Path(args.geometry), device)
    raw = up2.open_raw(LIVE_RAW, verify_sha=False)

    fold = CodeFoldBack(section, names, device)
    optimizer = torch.optim.AdamW(fold.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    decay = _ema_decay_from_run_geometry(int(args.steps))
    shadow = {name: fold.latent[name].detach().clone() for name in names}

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    step0 = 0
    history: list[dict[str, Any]] = []
    if args.resume_from:
        blob = torch.load(args.resume_from, map_location=device, weights_only=False)
        for name in names:
            fold.latent[name].data.copy_(blob["latent"][name].to(device))
            shadow[name] = blob["shadow"][name].to(device)
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        step0 = int(blob["step"])
        history = list(blob.get("history", []))
        print(f"resumed from {args.resume_from} at step {step0}", flush=True)

    # TWO pose weights, both derived, because prep MEASURED that the barrier alone is
    # nearly vacuous on this object: every one of the 600 Jacobians is full rank 6 and
    # the reach budget is 1,892-2,020 code units, so the re-solve can cancel any
    # first-order residual without approaching the lattice edge.
    #
    # (1) the barrier: at full reach the correcting step leaves the shipped int12
    #     lattice and the pair becomes unpayable on pose, which is worth exactly the
    #     whole seg residual this arm is trying to buy.
    # (2) the stale term: what the re-solve CANNOT remove (lattice polish residual and
    #     second order) still grows with the stale excursion.  Its weight is the S
    #     linearisation 5/sqrt(10*d_pose) DISCOUNTED by fe1's WORST measured per-pair
    #     recovery (643x, pair 382) -- the conservative end of measured evidence, not
    #     the best case and not a guess.
    weight_pose = 100.0 * LIVE_D_SEG_LOCAL
    weight_stale = (5.0 / math.sqrt(10.0 * LIVE_D_POSE)) / FE1_WORST_RECOVERY

    def save(tag: str, step: int) -> Path:
        path = run_dir / f"ckpt.{tag}.step{step:06d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": "ddm_rw1_ckpt.v1",
                "step": step,
                "latent": {n: fold.latent[n].detach().cpu() for n in names},
                "shadow": {n: shadow[n].detach().cpu() for n in names},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "names": list(names),
                "config": {
                    str(k): str(v) for k, v in vars(args).items() if k != "func"
                }
                | {"ema_decay": decay},
                "deployment_weights": "ema_shadow",
            },
            tmp,
        )
        tmp.rename(path)
        return path

    rng = np.random.default_rng(args.seed + step0)
    best = {"step": -1, "flips": None, "path": None}

    def one_chunk(index: np.ndarray, tau: float):
        """Loss terms for one chunk of pairs.  Returns (loss, diagnostics)."""
        tokens_batch = torch.from_numpy(tokens[index].astype(np.int64)).to(device)
        index_batch = torch.from_numpy(index).to(device)
        labels_batch = torch.from_numpy(labels[index].astype(np.int64)).to(device)
        frame = _render_eval(model, fold, tokens_batch, index_batch)
        camera = _exact_r_camera(frame)
        logits = _seg_logits(segnet, camera)
        seg_surrogate = _expected_flip(logits, labels_batch, tau).mean()
        with torch.no_grad():
            seg_reference = _expected_flip(
                logits.detach(), labels_batch, TAU_REFERENCE
            ).mean()
        frame0 = up2.frames_to_bchw(np.asarray(raw[2 * index])).to(device)
        pose_new = up2.pose_from_frames(posenet, frame0, camera)
        delta_r = pose_new - geom.pose_base[index_batch]
        step_codes = torch.einsum(
            "bij,bj->bi", geom.resolve_operator[index_batch], delta_r
        )
        reach = step_codes.abs().amax(dim=1)
        budget = geom.reach_budget[index_batch].clamp_min(1.0)
        pose_term = (reach / budget).pow(2).mean()
        stale_d_pose = ((pose_new - geom.targets[index_batch]) ** 2).mean()
        loss = (
            100.0 * seg_surrogate
            + weight_pose * pose_term
            + weight_stale * stale_d_pose
        )
        return loss, {
            "seg_surrogate": float(seg_surrogate.detach()),
            "seg_reference": float(seg_reference),
            "pose_term": float(pose_term.detach()),
            "reach_max": float(reach.max()),
            "reach_mean": float(reach.mean()),
            "stale_d_pose": float(stale_d_pose.detach()),
            "pose_drift": float(
                ((pose_new - geom.pose_base[index_batch]) ** 2).mean().detach()
            ),
        }

    for step in range(step0, int(args.steps)):
        progress = step / max(int(args.steps) - 1, 1)
        tau = TAU_START + (TAU_END - TAU_START) * progress
        optimizer.zero_grad(set_to_none=True)
        if args.full_field:
            # FULL-FIELD gradient by accumulation over all 600 pairs.  Measured at
            # batch 4 the fixed-tau surrogate has NO visible trend over 1,300 steps
            # (first-10 mean 4.796e-04, last-10 4.976e-04) and the code drift saturates
            # at ~0.6 -- the per-step direction is not persistent.  With only 12,672
            # parameters and a deterministic n600 objective, the noise-free gradient is
            # affordable, so minibatch noise stops being the confound.
            chunks = [
                np.arange(s, min(s + int(args.batch), N_PAIRS), dtype=np.int64)
                for s in range(0, N_PAIRS, int(args.batch))
            ]
            acc: dict[str, float] = {}
            for chunk in chunks:
                loss_chunk, diag = one_chunk(chunk, tau)
                (loss_chunk * (len(chunk) / N_PAIRS)).backward()
                for key, value in diag.items():
                    weight = len(chunk) / N_PAIRS
                    if key in ("reach_max",):
                        acc[key] = max(acc.get(key, 0.0), value)
                    else:
                        acc[key] = acc.get(key, 0.0) + value * weight
            diagnostics = acc
            loss_value = (
                100.0 * acc["seg_surrogate"]
                + weight_pose * acc["pose_term"]
                + weight_stale * acc["stale_d_pose"]
            )
        else:
            index = np.sort(rng.choice(N_PAIRS, size=int(args.batch), replace=False))
            loss_tensor, diagnostics = one_chunk(index, tau)
            loss_tensor.backward()
            loss_value = float(loss_tensor.detach())
        torch.nn.utils.clip_grad_norm_(fold.parameters(), 2.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for name in names:
                shadow[name].mul_(decay).add_(fold.latent[name].detach(), alpha=1 - decay)

        if (step + 1) % int(args.log_every) == 0:
            drift = max(
                float((fold.latent[n].detach() - fold.base_codes[n]).abs().max())
                for n in names
            )
            row = {
                "step": step + 1,
                "tau": tau,
                "full_field": bool(args.full_field),
                "loss": loss_value,
                "seg_surrogate_at_tau": diagnostics["seg_surrogate"],
                "seg_surrogate_at_tau_reference": diagnostics["seg_reference"],
                "pose_barrier": diagnostics["pose_term"],
                "reach_codes_max": diagnostics["reach_max"],
                "reach_codes_mean": diagnostics["reach_mean"],
                "stale_d_pose_batch": diagnostics["stale_d_pose"],
                "pose_drift_from_base_batch": diagnostics["pose_drift"],
                "latent_drift_max_codes": drift,
                "changed_codes": fold.changed_codes(),
                "changed_codes_shadow": fold.changed_codes(shadow),
                "lr": float(scheduler.get_last_lr()[0]),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)

        # The forced final evaluation is right for a real run and wrong for a smoke: a
        # 30-step smoke that asks for no evaluations should not pay a full n600 pass.
        final_eval = (step + 1) == int(args.steps) and int(args.eval_every) <= int(
            args.steps
        )
        if (step + 1) % int(args.eval_every) == 0 or final_eval:
            evaluation = _evaluate_realized(
                model, fold, segnet, tokens, labels, device, latent=shadow
            )
            changed = fold.changed_codes(shadow)
            row = {
                "step": step + 1,
                "eval_weights": "ema_shadow",
                "eval_axis": f"[{args.device} research-signal; n600 realized argmax]",
                "flips": evaluation["flips"],
                "d_seg": evaluation["d_seg"],
                "flips_vs_live": evaluation["flips"] - LIVE_D_SEG_CELLS,
                "reach_fraction_of_residual": (
                    LIVE_D_SEG_CELLS - evaluation["flips"]
                )
                / LIVE_D_SEG_CELLS,
                "changed_codes_shadow": changed,
                "rate_bytes_predicted": BYTES_PER_CHANGED_CODE * changed
                + (CONTAINER_BREAK_FIXED_BYTES if changed else 0.0),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            np.save(run_dir / f"per_pair_flips.step{step + 1:06d}.npy", evaluation["per_pair"])
            if best["flips"] is None or evaluation["flips"] < best["flips"]:
                best = {
                    "step": step + 1,
                    "flips": evaluation["flips"],
                    "path": str(save("best", step + 1)),
                }

        if (step + 1) % int(args.checkpoint_every) == 0:
            save("periodic", step + 1)

    final = save("final", int(args.steps))
    result = {
        "schema": "ddm_rw1_train.v1",
        "axis": f"[{args.device} research-signal; checkpoint selector, no score]",
        "score_claim": False,
        "pointer": pointer,
        "config": {str(k): str(v) for k, v in vars(args).items() if k != "func"}
        | {
            "ema_decay": decay,
            "weight_pose_barrier": weight_pose,
            "weight_pose_stale": weight_stale,
        },
        "trainable_tensors": list(names),
        "trainable_codes": sum(section.runs[n].count for n in names),
        "ema_decay_derivation": "ema_decay_run_geometry_v1: 1 - 5/steps",
        "lr_derivation": (
            "AdamW's normalised update is ~lr per step, so a cosine run of T steps "
            "drifts ~0.5*lr*T code units; lr is set for an O(1)-code drift over the "
            "horizon, which is derived from the int4 grid this arm actuates, not "
            "transferred from another vehicle (hr1: 2e-7 is an ancestor anchor)"
        ),
        "best": best,
        "final_checkpoint": str(final),
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# mode=export -- the trained codes as archive BYTES, container-searched, parsed back
# ----------------------------------------------------------------------------------


def build_candidate_archive(
    section: MultiSemanticSection,
    edits: dict[str, np.ndarray],
    carrier_codes: np.ndarray,
    *,
    tree_dir: Path = LIVE_TREE,
    container_search: bool = True,
    verify: bool = True,
    scratch: Path | None = None,
) -> dict[str, Any]:
    """Archive bytes carrying the folded-back weight codes AND the re-solved carrier.

    Structurally fe1's ``build_candidate_archive`` with the semantic edit generalised
    from ``frame_embed`` to an arbitrary set of code runs.  The tie-break is fe1's and
    it is not cosmetic: two container shapes produce the SAME 30,246 B from different
    bytes, so a length-only tie-break ships a stream that differs from the pointer's
    for no reason and destroys the null-build identity every later byte number rests on.
    """
    import io
    import zipfile

    import brotli
    import ddm_fe1_pose_price as price
    import ddm_up3_carrier_splice as up3

    body = up3.parse_shipped_body(tree_dir, verify_sha=False)
    built = up3.build_archive(
        body,
        carrier_codes,
        runtime_dir=tree_dir,
        container_search=container_search,
        verify=verify,
    )
    with zipfile.ZipFile(io.BytesIO(built["archive_bytes"])) as archive:
        outer = archive.read("p")

    ra, _cr, _ar1, _cp = up3._import_runtime(tree_dir)
    header = ra.RX1_MODEL_HEADER.unpack_from(outer)
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = header
    offset = ra.RX1_MODEL_HEADER.size
    hpac_stream = outer[offset : offset + hpac_bytes]
    offset += hpac_bytes + semantic_bytes
    carrier_stream = outer[offset : offset + carrier_bytes]
    offset += carrier_bytes
    section_tail = outer[offset:]

    stream = section.stream_with_codes(edits)
    interleaved = up3._ck2_interleave_planes(stream)
    shapes: dict[tuple[str, int, int], bytes] = {}
    for quality in price.CONTAINER_QUALITIES:
        for lgwin in price.CONTAINER_LGWINS:
            shapes[("ck2", quality, lgwin)] = brotli.compress(
                interleaved, quality=quality, lgwin=lgwin
            )
            shapes[("plain", quality, lgwin)] = brotli.compress(
                stream, quality=quality, lgwin=lgwin
            )
    chosen = min(
        shapes, key=lambda key: (len(shapes[key]), key != price.SHIPPED_SHAPE, key)
    )
    semantic_stream = shapes[chosen]
    reserved = (
        reserved | ra.CK2_RESERVED_SEMANTIC_PLANE2
        if chosen[0] == "ck2"
        else reserved & ~ra.CK2_RESERVED_SEMANTIC_PLANE2
    )
    new_outer = b"".join(
        (
            ra.RX1_MODEL_HEADER.pack(
                magic,
                version,
                codec,
                table_mode,
                reserved,
                hpac_bytes,
                len(semantic_stream),
                len(carrier_stream),
            ),
            hpac_stream,
            semantic_stream,
            carrier_stream,
            section_tail,
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        entry = zipfile.ZipInfo("p", date_time=tuple(body.zip_info["date_time"]))
        entry.compress_type = body.zip_info["compress_type"]
        entry.external_attr = body.zip_info["external_attr"]
        entry.create_system = body.zip_info["create_system"]
        archive.writestr(entry, new_outer)
    archive_bytes = buffer.getvalue()

    if verify:
        scratch = Path(scratch or (WORK / "candidate/.verify_archive.zip"))
        scratch.parent.mkdir(parents=True, exist_ok=True)
        scratch.write_bytes(archive_bytes)
        recovered = load_semantic_section(archive_path=scratch, runtime_dir=tree_dir / "runtime")
        for name, codes in edits.items():
            if not np.array_equal(
                np.asarray(recovered.codes[name], dtype=np.int64),
                np.asarray(codes, dtype=np.int64).reshape(recovered.runs[name].shape),
            ):
                raise Rw1Error(
                    f"the written archive does not parse back to the requested {name} "
                    "codes; refusing to return unverified bytes"
                )
        for name, run in section.runs.items():
            if run.row_pruned or name in edits:
                continue
            if not np.array_equal(
                np.asarray(recovered.codes[name], dtype=np.int64),
                np.asarray(section.codes[name], dtype=np.int64),
            ):
                raise Rw1Error(f"{name} moved without being edited; refusing")
        recovered_carrier, _info = up3.parse_back_codes(archive_bytes, runtime_dir=tree_dir)
        if not np.array_equal(
            np.asarray(recovered_carrier, dtype=np.int64),
            np.asarray(carrier_codes, dtype=np.int64),
        ):
            raise Rw1Error("the written archive does not parse back to the carrier codes")
        if bytes(ra.read_residual_archive(scratch).token_stream) != bytes(
            ra.read_residual_archive(LIVE_ARCHIVE).token_stream
        ):
            raise Rw1Error(
                "the token stream is not byte-identical to the live row; this arm "
                "changes the semantic and carrier sections only"
            )

    return {
        "archive_bytes": archive_bytes,
        "archive_size": len(archive_bytes),
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "semantic_stream_bytes": len(semantic_stream),
        "semantic_container": list(chosen),
        "carrier_stream_bytes": len(carrier_stream),
        "carrier_container": built["container"],
        "rx1_reserved": f"{reserved:#x}",
        "bytes_vs_live": len(archive_bytes) - LIVE_ARCHIVE_BYTES,
    }


def _codes_from_checkpoint(section, names, path: Path, which: str) -> dict[str, np.ndarray]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if which not in {"shadow", "latent"}:
        raise Rw1Error(f"unknown weight set {which!r}")
    if which == "shadow" and blob.get("deployment_weights") != "ema_shadow":
        raise Rw1Error("checkpoint does not declare the EMA shadow as deployment")
    out = {}
    for name in names:
        latent = blob[which][name].float()
        out[name] = (
            torch.clamp(torch.round(latent), CODE_MIN, CODE_MAX).numpy().astype(np.int64)
        )
    return out


def cmd_export(args) -> int:
    """Trained codes -> archive bytes, priced by a REAL encode with the container search.

    The null build (``--checkpoint`` omitted) re-encodes the SHIPPED codes and MUST
    reproduce the live archive byte for byte; without that identity no later byte
    number is attributable to the weight delta.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)


    state = up2.load_carrier_state(LIVE_TREE, verify_archive=False)
    carrier_codes = np.asarray(state.codes, dtype=np.int64)
    if args.carrier_codes:
        carrier_codes = np.load(args.carrier_codes).astype(np.int64)
        if carrier_codes.shape != (N_PAIRS, CARRIER_DIM):
            raise Rw1Error(f"carrier codes are {carrier_codes.shape}, expected (600, 12)")

    if args.checkpoint:
        edits = _codes_from_checkpoint(section, names, Path(args.checkpoint), args.weights)
    else:
        edits = {name: np.asarray(section.codes[name], dtype=np.int64) for name in names}

    changed = section.changed_code_count(edits)
    built = build_candidate_archive(
        section,
        edits,
        carrier_codes,
        container_search=not args.no_container_search,
        verify=not args.no_verify,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "archive.zip").write_bytes(built.pop("archive_bytes"))
    np.savez(
        out_dir / "codes.npz",
        **{name.replace(".", "__"): edits[name].astype(np.int8) for name in names},
    )
    np.save(out_dir / "carrier_codes.npy", carrier_codes.astype(np.int32))

    null_build = args.checkpoint is None
    result = {
        "schema": "ddm_rw1_export.v1",
        "axis": "[exact bytes]",
        "score_claim": False,
        "pointer": pointer,
        "null_build": null_build,
        "weights": args.weights,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "changed_codes": changed,
        "changed_codes_by_tensor": {
            name: int(
                (
                    np.asarray(edits[name], dtype=np.int64)
                    != np.asarray(section.codes[name], dtype=np.int64)
                ).sum()
            )
            for name in names
        },
        "rate_predicted_bytes": BYTES_PER_CHANGED_CODE * changed
        + (CONTAINER_BREAK_FIXED_BYTES if changed else 0.0),
        "rate_predicted_S": (
            BYTES_PER_CHANGED_CODE * changed
            + (CONTAINER_BREAK_FIXED_BYTES if changed else 0.0)
        )
        * RATE_PER_BYTE,
        "archive_dir": str(out_dir),
        "elapsed_seconds": time.perf_counter() - started,
        **built,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    if null_build and result["archive_sha256"] != LIVE_ARCHIVE_SHA256:
        raise Rw1Error(
            "the NULL build did not reproduce the live archive byte for byte "
            f"({result['archive_sha256']} vs {LIVE_ARCHIVE_SHA256}); no later byte "
            "delta would be attributable to the weight change"
        )
    return 0


# ----------------------------------------------------------------------------------
# mode=render -- the candidate decode, at the receiver's OWN batch-1 CPU numerics
# ----------------------------------------------------------------------------------


def _candidate_raw_path(out_dir: Path) -> Path:
    return Path(out_dir) / "0.raw"


def cmd_render(args) -> int:
    """Write the candidate's own decode: even frames copied, odd frames re-rendered.

    ``semantic_batch`` is 1 and that is not a performance oversight -- up2 sec.6
    MEASURED batch 8 as byte-changing on this half (1,326 pixels by +-1), so a batch-8
    render would not reproduce the decode every later realized number is measured
    against (``jg1.render_frame1``'s docstring).

    The even frames are the pose carrier and this arm's weight delta does not touch
    them, so they are COPIED from the live decode rather than re-rendered: copying is
    exact, re-rendering would introduce a difference the candidate does not have.
    """
    import shutil

    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    tokens = load_live_tokens()

    if args.checkpoint:
        edits = _codes_from_checkpoint(section, names, Path(args.checkpoint), args.weights)
    else:
        edits = {name: np.asarray(section.codes[name], dtype=np.int64) for name in names}

    model = load_live_renderer(section)
    with torch.no_grad():
        state = model.state_dict()
        for name in names:
            run = section.runs[name]
            scale_shape = [1] * len(run.shape)
            scale_shape[0] = section.scales[name].size
            state[name].copy_(
                torch.from_numpy(
                    np.asarray(edits[name], dtype=np.float32).reshape(run.shape)
                )
                * torch.from_numpy(section.scales[name].reshape(scale_shape))
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = _candidate_raw_path(out_dir)
    expected = 2 * N_PAIRS * CAMERA_H * CAMERA_W * 3
    if args.seed_only:
        # Seeding is its OWN step, never something a shard does implicitly: four shards
        # racing on one 3.66 GB copy is a corrupt decode that would look like a render
        # difference.  The copy lands on a sibling and is renamed, so the destination
        # either does not exist or is complete.
        staging = destination.with_suffix(".raw.partial")
        shutil.copyfile(LIVE_RAW, staging)
        staging.rename(destination)
        report = {
            "schema": "ddm_rw1_render_seed.v1",
            "seeded": str(destination),
            "bytes": destination.stat().st_size,
            "source": str(LIVE_RAW),
            "elapsed_seconds": time.perf_counter() - started,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
        print(json.dumps(report, indent=1, sort_keys=True))
        return 0
    if not destination.is_file() or destination.stat().st_size != expected:
        raise Rw1Error(
            f"candidate raw is not the {expected} B copy of the live decode; run "
            "``render --seed-only`` once before launching the shards"
        )

    raw = np.memmap(
        destination, dtype=np.uint8, mode="r+", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3)
    )
    pairs = np.arange(args.shard_index, N_PAIRS, args.shard_count, dtype=np.int64)
    changed_pixels = 0
    for offset, pair in enumerate(pairs):
        index = np.array([int(pair)], dtype=np.int64)
        rendered = jg1.render_frame1(model, tokens[index], index)[0]
        before = np.asarray(raw[2 * int(pair) + 1])
        changed_pixels += int((before != rendered).sum())
        raw[2 * int(pair) + 1] = rendered
        if args.progress and (offset + 1) % 25 == 0:
            print(
                f"rendered {offset + 1}/{len(pairs)} in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    raw.flush()
    del raw

    result = {
        "schema": "ddm_rw1_render.v1",
        "axis": "[cpu_torch batch-1; the receiver's own numerics]",
        "score_claim": False,
        "pointer": pointer,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "pairs": len(pairs),
        "semantic_batch": 1,
        "changed_pixels_vs_live": changed_pixels,
        "changed_codes": section.changed_code_count(edits),
        "raw": str(destination),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "weights": args.weights,
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt = Path(args.out)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# mode=seg -- the n600 realized seg leg on a candidate decode, DALI lineage
# ----------------------------------------------------------------------------------


def cmd_seg(args) -> int:
    """Realized argmax flips over all 600 pairs, on cpu_torch against the DALI table.

    A sub-n600 seg verdict is a toy on this axis, so the shards are strided (never a
    contiguous prefix -- ``m88``) and the merge refuses partial coverage.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    labels = load_gt_seg_dali()
    net = jg1.load_segnet()
    raw_path = Path(args.raw)
    expected = 2 * N_PAIRS * CAMERA_H * CAMERA_W * 3
    if raw_path.stat().st_size != expected:
        raise Rw1Error(f"raw is {raw_path.stat().st_size} B, expected {expected}")
    raw = np.memmap(
        raw_path, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3)
    )
    pairs = np.arange(args.shard_index, N_PAIRS, args.shard_count, dtype=np.int64)
    argmax = np.zeros((len(pairs), EVAL_H, EVAL_W), dtype=np.uint8)
    for offset, pair in enumerate(pairs):
        frames = np.asarray(raw[2 * int(pair) + 1])[None]
        argmax[offset] = jg1.argmax_from_camera_frames(net, frames)[0]
        if args.progress and (offset + 1) % 25 == 0:
            print(
                f"argmax {offset + 1}/{len(pairs)} in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    wrong = (argmax != labels[pairs]).reshape(len(pairs), -1).sum(axis=1)
    out = Path(args.out_argmax)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, argmax)
    result = {
        "schema": "ddm_rw1_seg_shard.v1",
        "axis": "[macOS-CPU advisory; cpu_torch argmax, DALI lineage]",
        "score_claim": False,
        "pointer": pointer,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "pairs": [int(p) for p in pairs],
        "cells": int(len(pairs) * EVAL_H * EVAL_W),
        "cells_disagreeing": int(wrong.sum()),
        "per_pair_cells": [int(v) for v in wrong],
        "argmax_path": str(out),
        "raw": str(raw_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt = Path(args.out)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "per_pair_cells"}, indent=1))
    return 0


def cmd_seg_merge(args) -> int:
    """Merge strided seg shards into the n600 leg.  Refuses partial coverage."""
    started = time.perf_counter()
    covered: dict[int, int] = {}
    total_cells = 0
    for path in args.shards:
        row = json.loads(Path(path).read_text())
        for pair, cells in zip(row["pairs"], row["per_pair_cells"], strict=True):
            if pair in covered:
                raise Rw1Error(f"pair {pair} appears in two shards")
            covered[int(pair)] = int(cells)
        total_cells += int(row["cells"])
    missing = sorted(set(range(N_PAIRS)) - set(covered))
    if missing:
        raise Rw1Error(
            f"{len(missing)} pairs are missing ({missing[:8]}...); a sub-n600 seg "
            "verdict is a TOY on this axis and is refused"
        )
    per_pair = np.array([covered[p] for p in range(N_PAIRS)], dtype=np.int64)
    flips = int(per_pair.sum())
    d_seg = flips / SEG_CELLS_TOTAL
    repaired = LIVE_D_SEG_CELLS - flips
    # The per-pair comparison needs the LIVE per-pair leg, not a sign test on a count
    # that cannot be negative.  sj1's own n600 argmax of the live decode is the
    # baseline; deriving it here keeps the two legs on one GT table.
    labels = load_gt_seg_dali()
    live_argmax = np.load(LIVE_ARGMAX, mmap_mode="r")
    live_per_pair = np.zeros(N_PAIRS, dtype=np.int64)
    for pair in range(N_PAIRS):
        live_per_pair[pair] = int(
            (np.asarray(live_argmax[pair]) != labels[pair]).sum()
        )
    if int(live_per_pair.sum()) != LIVE_D_SEG_CELLS:
        raise Rw1Error(
            f"the live argmax reproduces {int(live_per_pair.sum())} cells, not the "
            f"pinned {LIVE_D_SEG_CELLS}; the baseline is not the one this delta claims"
        )
    result = {
        "schema": "ddm_rw1_seg.v1",
        "axis": "[macOS-CPU advisory; cpu_torch argmax, DALI lineage, n600]",
        "score_claim": False,
        "pairs": N_PAIRS,
        "cells": total_cells,
        "cells_disagreeing": flips,
        "d_seg_local": d_seg,
        "d_seg_t4_carried": d_seg * SEG_T4_RATIO,
        "live_cells": LIVE_D_SEG_CELLS,
        "cells_repaired": repaired,
        "reach_fraction_of_residual": repaired / LIVE_D_SEG_CELLS,
        "dS_seg": -repaired * S_PER_SEG_CELL,
        "per_pair_cells": [int(v) for v in per_pair],
        "live_per_pair_cells": [int(v) for v in live_per_pair],
        "pairs_improved": int((per_pair < live_per_pair).sum()),
        "pairs_worsened": int((per_pair > live_per_pair).sum()),
        "pairs_unchanged": int((per_pair == live_per_pair).sum()),
        "cells_repaired_on_improved_pairs": int(
            (live_per_pair - per_pair)[per_pair < live_per_pair].sum()
        ),
        "cells_broken_on_worsened_pairs": int(
            (per_pair - live_per_pair)[per_pair > live_per_pair].sum()
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    np.save(Path(args.out).with_suffix(".per_pair.npy"), per_pair)
    np.save(Path(args.out).with_suffix(".live_per_pair.npy"), live_per_pair)
    print(json.dumps({k: v for k, v in result.items() if k != "per_pair_cells"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# mode=pose -- the per-pair carrier re-solve on a GLOBAL render change
# ----------------------------------------------------------------------------------


def _pose_instrument(raw_path: Path):
    """br1's instrument on a named decode, pinned to the LIVE carrier and DALI GT."""
    import ddm_br1_pose_basis_reorientation as br1

    expected = 2 * N_PAIRS * CAMERA_H * CAMERA_W * 3
    raw_path = Path(raw_path)
    if raw_path.stat().st_size != expected:
        raise Rw1Error(f"raw is {raw_path.stat().st_size} B, expected {expected}")
    raw = np.memmap(
        raw_path, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3)
    )
    state = up2.load_carrier_state(LIVE_TREE, verify_archive=False)
    targets, lineage = up2.load_gt_poses(GT_CACHE_DALI)
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat), br1


def cmd_pose(args) -> int:
    """Per pair: base, stale, and re-solved d_pose on the candidate's own renders.

    fe1 measured the per-pair re-solve recovering 643-3,053x -- but on a change
    confined to ONE pair.  This arm's change moves ALL 600 renders, which the coupling
    law says is a different case, so the recovery is MEASURED here rather than
    inherited.  Each pair is still re-solved independently against its OWN new render
    from the LIVE coefficients, so the only thing the global case removes is the
    ability to leave the other 599 pairs alone.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    import ddm_jg5_pose_resolve_on_edited_renders as jg5

    moved, br1 = _pose_instrument(Path(args.raw))
    base, _br1 = _pose_instrument(LIVE_RAW)
    live_codes = np.asarray(base.state.codes, dtype=np.int32)
    threshold = jg5.materiality_dd_threshold(float(args.base_mean_d_pose))

    pairs = np.arange(args.shard_index, N_PAIRS, args.shard_count, dtype=np.int64)
    out_path = Path(args.out_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(int(json.loads(line)["pair"]))

    with out_path.open("a") as stream:
        for offset, pair in enumerate(pairs):
            pair = int(pair)
            if pair in done:
                continue
            d_base = float(br1.evaluate_codes(base, pair, live_codes[pair][None])[0])
            d_stale = float(br1.evaluate_codes(moved, pair, live_codes[pair][None])[0])
            refined = jg5.refine_pair(
                moved,
                pair,
                live_codes[pair],
                dd_threshold=threshold,
                outer_rounds=int(args.outer_rounds),
                max_gn_iterations=int(args.max_gn_iterations),
            )
            row = {
                "pair": pair,
                "d_pose_base": d_base,
                "d_pose_stale": d_stale,
                "d_pose_resolved": float(refined["final_d_pose"]),
                "resolved_codes": [int(v) for v in refined["codes"]],
                "live_codes": [int(v) for v in live_codes[pair]],
                "stale_over_base": d_stale / max(d_base, 1e-30),
                "recovery_stale_over_resolved": d_stale
                / max(float(refined["final_d_pose"]), 1e-30),
                "resolved_over_base": float(refined["final_d_pose"]) / max(d_base, 1e-30),
                "stop_reason": refined.get("stop_reason"),
                "rounds": refined.get("rounds"),
                "evaluations": refined.get("evaluations"),
                "shippable_d_pose": min(d_stale, float(refined["final_d_pose"])),
                "ships_resolved_codes": bool(float(refined["final_d_pose"]) <= d_stale),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if args.progress:
                print(
                    f"pose {offset + 1}/{len(pairs)} pair={pair} "
                    f"base={d_base:.3e} stale={d_stale:.3e} "
                    f"resolved={refined['final_d_pose']:.3e} "
                    f"in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    result = {
        "schema": "ddm_rw1_pose_shard.v1",
        "axis": "[cpu_torch fp32 authority, DALI GT]",
        "score_claim": False,
        "pointer": pointer,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "pairs": len(pairs),
        "rows": str(out_path),
        "materiality_dd_threshold": threshold,
        "raw": str(args.raw),
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# mode=admit -- one atom on seg, per-pair on the carrier, priced by a REAL encode
# ----------------------------------------------------------------------------------


def cmd_admit(args) -> int:
    """The admission.  The WEIGHT delta is one atom; the CARRIER choice is per pair.

    A weight change moves all 600 renders and the receiver has no per-pair selector
    for weights, so a pair that loses cannot opt out of the delta -- the sum over all
    600 pairs is the verdict.  What IS per-pair is the carrier: each pair ships either
    its re-solved twelve codes or the live ones, whichever measures lower, and both
    are representable, so that choice is free.  Everything is priced by a REAL encode.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    seg = json.loads(Path(args.seg).read_text())
    rows: dict[int, dict[str, Any]] = {}
    for path in args.pose_rows:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["pair"])] = row
    missing = sorted(set(range(N_PAIRS)) - set(rows))
    if missing:
        raise Rw1Error(f"{len(missing)} pose rows missing ({missing[:8]}...)")

    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    edits = _codes_from_checkpoint(section, names, Path(args.checkpoint), args.weights)
    changed_codes = section.changed_code_count(edits)

    state = up2.load_carrier_state(LIVE_TREE, verify_archive=False)
    carrier = np.asarray(state.codes, dtype=np.int64).copy()
    shipped_pose = np.zeros(N_PAIRS, dtype=np.float64)
    base_pose = np.zeros(N_PAIRS, dtype=np.float64)
    stale_pose = np.zeros(N_PAIRS, dtype=np.float64)
    resolved_pose = np.zeros(N_PAIRS, dtype=np.float64)
    resolved_shipped = 0
    for pair in range(N_PAIRS):
        row = rows[pair]
        base_pose[pair] = row["d_pose_base"]
        stale_pose[pair] = row["d_pose_stale"]
        resolved_pose[pair] = row["d_pose_resolved"]
        if row["ships_resolved_codes"]:
            carrier[pair] = np.asarray(row["resolved_codes"], dtype=np.int64)
            resolved_shipped += 1
        shipped_pose[pair] = row["shippable_d_pose"]

    built = build_candidate_archive(
        section, edits, carrier, container_search=True, verify=True
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "archive.zip").write_bytes(built.pop("archive_bytes"))
    np.save(out_dir / "carrier_codes.npy", carrier.astype(np.int32))
    np.savez(
        out_dir / "codes.npz",
        **{name.replace(".", "__"): edits[name].astype(np.int8) for name in names},
    )

    d_seg_new = float(seg["d_seg_local"])
    d_pose_new = float(shipped_pose.mean())
    bytes_new = int(built["archive_size"])
    dS_seg = 100.0 * (d_seg_new - LIVE_D_SEG_LOCAL)
    dS_pose = math.sqrt(10.0 * d_pose_new) - math.sqrt(10.0 * LIVE_D_POSE)
    dS_rate = (bytes_new - LIVE_ARCHIVE_BYTES) * RATE_PER_BYTE
    total = dS_seg + dS_pose + dS_rate

    result = {
        "schema": "ddm_rw1_admission.v1",
        "axis": "[macOS-CPU advisory seg + cpu_torch pose + exact bytes; projection]",
        "score_claim": False,
        "pointer": pointer,
        "admit_bar": ADMIT_BAR,
        "admits": bool(total < ADMIT_BAR),
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "changed_codes": changed_codes,
        "changed_codes_by_tensor": {
            name: int(
                (
                    np.asarray(edits[name], dtype=np.int64)
                    != np.asarray(section.codes[name], dtype=np.int64)
                ).sum()
            )
            for name in names
        },
        "seg": {
            "cells_live": LIVE_D_SEG_CELLS,
            "cells_candidate": int(seg["cells_disagreeing"]),
            "cells_repaired": int(seg["cells_repaired"]),
            "reach_fraction_of_residual": float(seg["reach_fraction_of_residual"]),
            "d_seg_live": LIVE_D_SEG_LOCAL,
            "d_seg_candidate": d_seg_new,
            "pairs_improved": seg.get("pairs_improved"),
            "pairs_worsened": seg.get("pairs_worsened"),
            "dS": dS_seg,
        },
        "pose": {
            "d_pose_live": LIVE_D_POSE,
            "d_pose_base_recomputed": float(base_pose.mean()),
            "d_pose_stale": float(stale_pose.mean()),
            "d_pose_resolved": float(resolved_pose.mean()),
            "d_pose_shipped": d_pose_new,
            "stale_over_base": float(stale_pose.mean() / max(base_pose.mean(), 1e-30)),
            "recovery_stale_over_shipped": float(
                stale_pose.mean() / max(d_pose_new, 1e-30)
            ),
            "shipped_over_base": float(d_pose_new / max(base_pose.mean(), 1e-30)),
            "pairs_shipping_resolved_codes": resolved_shipped,
            "dS": dS_pose,
        },
        "rate": {
            "bytes_live": LIVE_ARCHIVE_BYTES,
            "bytes_candidate": bytes_new,
            "bytes_delta": bytes_new - LIVE_ARCHIVE_BYTES,
            "predicted_bytes_from_code_law": BYTES_PER_CHANGED_CODE * changed_codes
            + (CONTAINER_BREAK_FIXED_BYTES if changed_codes else 0.0),
            "semantic_stream_bytes": built["semantic_stream_bytes"],
            "semantic_container": built["semantic_container"],
            "carrier_stream_bytes": built["carrier_stream_bytes"],
            "dS": dS_rate,
        },
        "dS_total": total,
        "S_projected": LIVE_SCORE_T4 + total,
        "S_live": LIVE_SCORE_T4,
        "archive_sha256": built["archive_sha256"],
        "archive_dir": str(out_dir),
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# mode=rate-law -- what a changed int4 CODE actually costs in this section
# ----------------------------------------------------------------------------------


def semantic_section_bytes(section: MultiSemanticSection, edits) -> dict[str, Any]:
    """Container-searched size of the semantic section carrying ``edits``.

    Every other section is byte-identical under this arm's change, so the archive
    delta IS the semantic-stream delta -- exactly, not approximately.  That is why the
    rate law can be measured without building a whole archive per point.
    """

    return _price_stream(section.stream_with_codes(edits))


def _price_stream(stream: bytes) -> dict[str, Any]:
    """Container-searched size of one already-built RC1 semantic stream."""
    import brotli
    import ddm_fe1_pose_price as price
    import ddm_up3_carrier_splice as up3

    interleaved = up3._ck2_interleave_planes(stream)
    shapes: dict[tuple[str, int, int], bytes] = {}
    for quality in price.CONTAINER_QUALITIES:
        for lgwin in price.CONTAINER_LGWINS:
            shapes[("ck2", quality, lgwin)] = brotli.compress(
                interleaved, quality=quality, lgwin=lgwin
            )
            shapes[("plain", quality, lgwin)] = brotli.compress(
                stream, quality=quality, lgwin=lgwin
            )
    chosen = min(
        shapes, key=lambda key: (len(shapes[key]), key != price.SHIPPED_SHAPE, key)
    )
    return {
        "searched_bytes": len(shapes[chosen]),
        "shipped_shape_bytes": len(shapes[price.SHIPPED_SHAPE]),
        "container": list(chosen),
    }


def cmd_rate_law(args) -> int:
    """Measure the archive cost of N changed codes on THIS section.

    fe1 fitted ``0.15*N + 8`` B on the frame_embed run (depth 3, N <= 200).  This arm
    edits head/blocks.3 (depth 4) at N up to 12,672, so the law is RE-DERIVED at this
    scope rather than extrapolated 63x past its measured range
    (``cross-regime constant transfer``, [[m143]]).  The point of the curve is the
    break-even: one repaired seg cell buys 8.477e-07 S, so the arm is viable only if
    the marginal cost per code stays well under 8.477e-07 / 6.6586e-07 = 1.273 B.
    """
    started = time.perf_counter()
    pointer = verify_live_pointer()
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    flat = np.concatenate(
        [np.asarray(section.codes[name], dtype=np.int64).ravel() for name in names]
    )
    sizes = [int(section.runs[name].count) for name in names]
    base_semantic = semantic_section_bytes(
        section, {n: section.codes[n] for n in names}
    )
    if base_semantic["searched_bytes"] != 30_246:
        raise Rw1Error(
            f"the unperturbed section prices at {base_semantic['searched_bytes']} B, "
            "not the shipped 30,246; the null control is broken"
        )

    rng = np.random.default_rng(int(args.seed))
    counts = [int(v) for v in str(args.counts).split(",") if v.strip()]
    rows = []
    for count in counts:
        for repeat in range(int(args.repeats)):
            perturbed = flat.copy()
            where = rng.choice(flat.size, size=min(count, flat.size), replace=False)
            # A code must MOVE, and it must stay in the shipped signed-int4 domain.
            step = rng.choice(np.array([-1, 1]), size=where.size)
            proposal = np.clip(perturbed[where] + step, CODE_MIN, CODE_MAX)
            stuck = proposal == perturbed[where]
            proposal[stuck] = np.clip(perturbed[where][stuck] - step[stuck], CODE_MIN, CODE_MAX)
            perturbed[where] = proposal
            changed = int((perturbed != flat).sum())
            edits = {}
            cursor = 0
            for name, size in zip(names, sizes, strict=True):
                edits[name] = perturbed[cursor : cursor + size].reshape(
                    section.runs[name].shape
                )
                cursor += size
            priced = semantic_section_bytes(section, edits)
            rows.append(
                {
                    "requested": count,
                    "repeat": repeat,
                    "changed_codes": changed,
                    "searched_bytes": priced["searched_bytes"],
                    "shipped_shape_bytes": priced["shipped_shape_bytes"],
                    "delta_searched": priced["searched_bytes"]
                    - base_semantic["searched_bytes"],
                    "delta_shipped_shape": priced["shipped_shape_bytes"]
                    - base_semantic["shipped_shape_bytes"],
                    "container": priced["container"],
                }
            )
            if args.progress:
                print(json.dumps(rows[-1]), flush=True)

    by_count: dict[int, list[int]] = {}
    for row in rows:
        by_count.setdefault(row["requested"], []).append(row["delta_searched"])
    summary = {
        str(count): {
            "n": len(values),
            "mean_delta_bytes": float(np.mean(values)),
            "min": int(min(values)),
            "max": int(max(values)),
            "bytes_per_code": float(np.mean(values)) / count,
            "cells_to_break_even": float(np.mean(values)) * RATE_PER_BYTE / S_PER_SEG_CELL,
        }
        for count, values in sorted(by_count.items())
    }
    ordered = sorted(by_count)
    if len(ordered) >= 2:
        xs = np.array(ordered, dtype=np.float64)
        ys = np.array([np.mean(by_count[c]) for c in ordered], dtype=np.float64)
        slope, intercept = np.polyfit(xs, ys, 1)
    else:
        slope, intercept = float("nan"), float("nan")
    result = {
        "schema": "ddm_rw1_rate_law.v1",
        "axis": "[exact bytes; container-searched real encode]",
        "score_claim": False,
        "pointer": pointer,
        "base_semantic_bytes": base_semantic,
        "trainable_codes": int(flat.size),
        "counts": counts,
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "summary": summary,
        "fit_bytes_per_code": float(slope),
        "fit_fixed_bytes": float(intercept),
        "fe1_predicted_bytes_per_code": BYTES_PER_CHANGED_CODE,
        "break_even_cells_per_code_measured": float(slope)
        * RATE_PER_BYTE
        / S_PER_SEG_CELL,
        "break_even_cells_per_code_fe1": BYTES_PER_CHANGED_CODE
        * RATE_PER_BYTE
        / S_PER_SEG_CELL,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# mode=rebase -- re-point at a moved frontier by MEASURING it, never by typing it
# ----------------------------------------------------------------------------------


def cmd_rebase(args) -> int:
    """Write the live-body pin from a moved pointer, measuring everything measurable.

    The archive sha, its size and the SM3R body sha are MEASURED from the tree; only the
    numbers that live in an authority receipt (the T4 score and its two legs) are
    accepted as arguments, and even those are read from the pointer file where it
    carries them.  The one hard refusal: if the new tree's SM3R body differs from the
    one this arm trained against, the frozen base codes are a different object and the
    trained delta does not transfer -- re-training, not re-pinning, is the answer, so
    the pin refuses unless the caller says so out loud.
    """
    started = time.perf_counter()
    tree = Path(args.tree)
    archive = tree / "archive.zip"
    if not archive.is_file():
        raise Rw1Error(f"no archive.zip under {tree}")
    ra, rc1, renderer = import_live(tree / "runtime")
    parts = ra.read_residual_archive(archive)
    stream = bytes(parts.semantic_blob)
    template = renderer.SemanticTokenRenderer(96).state_dict()
    body = (
        rc1.restore_semantic(stream, template)
        if stream.startswith(rc1.SEMANTIC_MAGIC)
        else stream
    )
    sm3r_sha = hashlib.sha256(body).hexdigest()
    trained_against = LIVE_SM3R_SHA256
    if sm3r_sha != trained_against and not args.allow_semantic_change:
        raise Rw1Error(
            f"the new tree's SM3R body is {sm3r_sha}, not the {trained_against} this "
            "arm's base codes were frozen against; a trained code DELTA is defined "
            "relative to those codes, so it does not transfer -- re-train, or pass "
            "--allow-semantic-change and say in the receipt why the delta still applies"
        )

    pointer = json.loads(POINTER_JSON.read_text())["effective_frontier"]
    observed = sha256_file(archive)
    if pointer["archive_sha256"] != observed:
        raise Rw1Error(
            f"the pointer names {pointer['archive_sha256']} but {archive} is {observed}; "
            "re-base onto the tree the pointer actually names"
        )
    pin = {
        "schema": "ddm_rw1_live_pin.v1",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "LIVE_TREE": str(tree),
        "LIVE_RAW": str(args.raw),
        "LIVE_FIELD": str(args.field),
        "LIVE_ARGMAX": str(args.argmax),
        "SHARED_SECTION_SOURCE": str(args.shared_source or archive),
        "LIVE_ARCHIVE_SHA256": observed,
        "LIVE_ARCHIVE_BYTES": archive.stat().st_size,
        "LIVE_SCORE_T4": float(pointer["score"]),
        "LIVE_D_SEG_CELLS": int(args.d_seg_cells),
        "LIVE_D_SEG_LOCAL": int(args.d_seg_cells) / SEG_CELLS_TOTAL,
        "LIVE_D_SEG_T4": float(args.d_seg_t4),
        "LIVE_D_POSE": float(args.d_pose),
        "SEG_T4_RATIO": SEG_T4_RATIO,
        "LIVE_SM3R_SHA256": sm3r_sha,
        "semantic_changed_vs_trained_base": sm3r_sha != trained_against,
        "pointer_lane_id": pointer.get("lane_id"),
        "measured_not_typed": [
            "LIVE_ARCHIVE_SHA256",
            "LIVE_ARCHIVE_BYTES",
            "LIVE_SM3R_SHA256",
            "LIVE_SCORE_T4",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    LIVE_PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_PIN_PATH.write_text(json.dumps(pin, indent=1, sort_keys=True))
    print(json.dumps(pin, indent=1, sort_keys=True))
    return 0


# ----------------------------------------------------------------------------------
# mode=perturb-control -- is the TRAINED direction better than a random one?
# ----------------------------------------------------------------------------------


def cmd_perturb_control(args) -> int:
    """n600 realized flips for RANDOM +-1 code changes of the same size as the trained one.

    Without this the trained result is uninterpretable.  If a random N-code change
    costs about the same as the trained N-code change, the surrogate is not steering at
    all and the finding is about the SEARCH, not about the actuator; if random costs far
    more, the actuator is steering and the finding is about its REACH.  The two readings
    prescribe different next moves, so the control is not optional.
    """
    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    model = load_live_renderer(section).to(device)
    tokens = load_live_tokens()
    labels = load_gt_seg_dali()
    segnet = jg1.load_segnet().to(device).eval()
    for param in segnet.parameters():
        param.requires_grad_(False)
    fold = CodeFoldBack(section, names, device)

    sizes = [int(v) for v in str(args.counts).split(",") if v.strip()]
    flat_sizes = [int(section.runs[n].count) for n in names]
    base_flat = np.concatenate(
        [np.asarray(section.codes[n], dtype=np.int64).ravel() for n in names]
    )
    rng = np.random.default_rng(int(args.seed))

    with torch.no_grad():
        null = _evaluate_realized(model, fold, segnet, tokens, labels, device)
    rows = []
    for count in sizes:
        for draw in range(int(args.draws)):
            perturbed = base_flat.copy()
            where = rng.choice(base_flat.size, size=count, replace=False)
            step = rng.choice(np.array([-1, 1]), size=count)
            proposal = np.clip(perturbed[where] + step, CODE_MIN, CODE_MAX)
            stuck = proposal == perturbed[where]
            proposal[stuck] = np.clip(
                perturbed[where][stuck] - step[stuck], CODE_MIN, CODE_MAX
            )
            perturbed[where] = proposal
            latent = {}
            cursor = 0
            for name, size in zip(names, flat_sizes, strict=True):
                latent[name] = torch.from_numpy(
                    perturbed[cursor : cursor + size]
                    .reshape(section.runs[name].shape)
                    .astype(np.float32)
                ).to(device)
                cursor += size
            with torch.no_grad():
                evaluation = _evaluate_realized(
                    model, fold, segnet, tokens, labels, device, latent=latent
                )
            rows.append(
                {
                    "changed_codes": int((perturbed != base_flat).sum()),
                    "draw": draw,
                    "flips": evaluation["flips"],
                    "flips_vs_null": evaluation["flips"] - null["flips"],
                    "cells_broken_per_code": (evaluation["flips"] - null["flips"])
                    / max(count, 1),
                }
            )
            print(json.dumps(rows[-1]), flush=True)

    result = {
        "schema": "ddm_rw1_perturb_control.v1",
        "axis": f"[{args.device} research-signal; n600 realized argmax]",
        "score_claim": False,
        "pointer": pointer,
        "null_flips_same_path": null["flips"],
        "live_cells": LIVE_D_SEG_CELLS,
        "device_gap_vs_cpu_instrument": null["flips"] - LIVE_D_SEG_CELLS,
        "counts": sizes,
        "draws": int(args.draws),
        "seed": int(args.seed),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# mode=gradient-topk -- the direct question, with no optimizer in the way
# ----------------------------------------------------------------------------------


def cmd_gradient_topk(args) -> int:
    """One exact n600 gradient, then realized flips for the top-k signed code moves.

    This asks the arm's question with nothing between the objective and the actuator:
    if the k codes with the largest |dL/dcode| are each moved ONE step down their own
    gradient, does the realized argmax improve?  No optimizer, no schedule, no EMA, no
    minibatch draw -- so a negative here is about the OBJECTIVE and the actuator, which
    is the only thing left to be about.

    It also sidesteps the fault the full-field probe exposed: AdamW normalises per
    parameter and therefore marches every code at the same rate, destroying the
    sparsity this actuator's rate law and collateral measurement both call for.  Top-k
    on the raw gradient preserves it by construction.
    """
    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    model = load_live_renderer(section).to(device)
    tokens = load_live_tokens()
    labels = load_gt_seg_dali()
    segnet = jg1.load_segnet().to(device).eval()
    for param in segnet.parameters():
        param.requires_grad_(False)
    fold = CodeFoldBack(section, names, device)

    # One exact field gradient of the SEG surrogate alone.  Pose is deliberately out:
    # the per-pair re-solve is what pays pose, and this probe is about seg reach.
    tau = float(args.tau)
    for param in fold.parameters():
        param.grad = None
    for start in range(0, N_PAIRS, int(args.batch)):
        index = np.arange(start, min(start + int(args.batch), N_PAIRS), dtype=np.int64)
        tokens_batch = torch.from_numpy(tokens[index].astype(np.int64)).to(device)
        index_batch = torch.from_numpy(index).to(device)
        labels_batch = torch.from_numpy(labels[index].astype(np.int64)).to(device)
        frame = _render_eval(model, fold, tokens_batch, index_batch)
        logits = _seg_logits(segnet, _exact_r_camera(frame))
        loss = _expected_flip(logits, labels_batch, tau).mean() * (len(index) / N_PAIRS)
        loss.backward()
        if args.progress and (start // int(args.batch)) % 25 == 0:
            print(
                f"grad {start}/{N_PAIRS} in {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    grads = {n: fold.latent[n].grad.detach().to("cpu").numpy().ravel() for n in names}
    sizes = [int(section.runs[n].count) for n in names]
    flat_grad = np.concatenate([grads[n] for n in names])
    base_flat = np.concatenate(
        [np.asarray(section.codes[n], dtype=np.int64).ravel() for n in names]
    )
    # RANKING is the thing under test, not a detail.  ``abs_desc`` is the textbook
    # first-order choice; this arm MEASURED it to be the worst possible one (the
    # highest-|gradient| codes are 20-35x more destructive than random to move by one
    # int4 step, because the linearisation is most invalid exactly where the gradient is
    # largest).  ``abs_asc`` and ``band`` test the ranking the AdamW run's own behaviour
    # implies: it beat random by selecting codes with SMALL, PERSISTENT gradients.
    magnitude = np.abs(flat_grad)
    ranking = str(args.rank)
    if ranking == "abs_desc":
        order = np.argsort(-magnitude)
    elif ranking == "abs_asc":
        nonzero = np.flatnonzero(magnitude > 0)
        order = nonzero[np.argsort(magnitude[nonzero])]
    elif ranking.startswith("band:"):
        lo_pct, hi_pct = (float(v) for v in ranking.split(":", 1)[1].split(","))
        lo, hi = np.percentile(magnitude, [lo_pct, hi_pct])
        inside = np.flatnonzero((magnitude >= lo) & (magnitude <= hi))
        order = inside[np.argsort(-magnitude[inside])]
    else:
        raise Rw1Error(
            f"unknown --rank {ranking!r}; use abs_desc, abs_asc or band:<lo_pct>,<hi_pct>"
        )
    grad_time = time.perf_counter() - started

    with torch.no_grad():
        null = _evaluate_realized(model, fold, segnet, tokens, labels, device)
    rows = []
    for k in [int(v) for v in str(args.k).split(",") if v.strip()]:
        if k > order.size:
            continue
        picked = order[:k]
        perturbed = base_flat.copy()
        # ONE step DOWN the gradient: a positive dL/dcode wants the code smaller.
        move = -np.sign(flat_grad[picked]).astype(np.int64)
        perturbed[picked] = np.clip(perturbed[picked] + move, CODE_MIN, CODE_MAX)
        changed = int((perturbed != base_flat).sum())
        latent = {}
        cursor = 0
        for name, size in zip(names, sizes, strict=True):
            latent[name] = torch.from_numpy(
                perturbed[cursor : cursor + size]
                .reshape(section.runs[name].shape)
                .astype(np.float32)
            ).to(device)
            cursor += size
        with torch.no_grad():
            evaluation = _evaluate_realized(
                model, fold, segnet, tokens, labels, device, latent=latent
            )
        rows.append(
            {
                "k": k,
                "changed_codes": changed,
                "flips": evaluation["flips"],
                "flips_vs_null": evaluation["flips"] - null["flips"],
                "cells_per_code": (evaluation["flips"] - null["flips"]) / max(changed, 1),
                "rate_bytes_predicted": BYTES_PER_CHANGED_CODE * changed
                + (CONTAINER_BREAK_FIXED_BYTES if changed else 0.0),
                **_stake(evaluation["flips"] - null["flips"]),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    result = {
        "schema": "ddm_rw1_gradient_topk.v1",
        "axis": f"[{args.device} research-signal; n600 exact gradient + realized argmax]",
        "score_claim": False,
        "pointer": pointer,
        "tau": tau,
        "rank": ranking,
        "ranked_pool": int(order.size),
        "null_flips_same_path": null["flips"],
        "device_gap_vs_cpu_instrument": null["flips"] - LIVE_D_SEG_CELLS,
        "gradient_seconds": grad_time,
        "gradient_nonzero_codes": int((flat_grad != 0).sum()),
        "gradient_abs_max": float(np.abs(flat_grad).max()),
        "gradient_abs_median": float(np.median(np.abs(flat_grad))),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# mode=code-search -- DISCRETE realized search over the int4 codes (the sj1/fe1 pattern)
# ----------------------------------------------------------------------------------


def _evaluate_subset(
    model, fold, segnet, tokens, labels, device, pairs, *, latent=None, batch: int = 4
) -> int:
    """Realized argmax flips over a NAMED subset of pairs.  Screening only."""
    import torch

    flips = 0
    with torch.no_grad():
        for start in range(0, len(pairs), batch):
            index = np.asarray(pairs[start : start + batch], dtype=np.int64)
            tokens_batch = torch.from_numpy(tokens[index].astype(np.int64)).to(device)
            index_batch = torch.from_numpy(index).to(device)
            frame = _render_eval(model, fold, tokens_batch, index_batch, latent)
            logits = _seg_logits(segnet, _exact_r_camera(frame))
            argmax = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
            flips += int((argmax != labels[index]).sum())
    return flips


def cmd_code_search(args) -> int:
    """Greedy realized acceptance over int4 code moves, ranked by the surrogate.

    The sj1/fe1 pattern moved onto this actuator: PROPOSE a group of +-1 code moves
    down the surrogate's own gradient sign, RENDER, score with the frozen SegNet, and
    accept the group only if realized flips strictly FALL.  A rejected group is halved
    and its halves retried, so a group that contains one good move is not thrown away
    with it.  Nothing is accepted on a predicted delta; the acceptance is realized.

    Screening runs on a SEEDED RANDOM subset of pairs (never a prefix -- a prefix of
    this video is a different population, worst on exactly these axes), and every
    accepted set is confirmed at n600 before it is reported.
    """
    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    model = load_live_renderer(section).to(device)
    tokens = load_live_tokens()
    labels = load_gt_seg_dali()
    segnet = jg1.load_segnet().to(device).eval()
    for param in segnet.parameters():
        param.requires_grad_(False)
    fold = CodeFoldBack(section, names, device)
    sizes = [int(section.runs[n].count) for n in names]
    base_flat = np.concatenate(
        [np.asarray(section.codes[n], dtype=np.int64).ravel() for n in names]
    )

    rng = np.random.default_rng(int(args.seed))
    screen_pairs = (
        np.sort(rng.choice(N_PAIRS, size=int(args.screen_pairs), replace=False))
        if int(args.screen_pairs)
        else np.arange(N_PAIRS, dtype=np.int64)
    )

    def as_latent(flat: np.ndarray):
        out = {}
        cursor = 0
        for name, size in zip(names, sizes, strict=True):
            out[name] = torch.from_numpy(
                flat[cursor : cursor + size]
                .reshape(section.runs[name].shape)
                .astype(np.float32)
            ).to(device)
            cursor += size
        return out

    # One exact gradient on the SCREEN subset gives the proposal ranking and its signs.
    for param in fold.parameters():
        param.grad = None
    for start in range(0, len(screen_pairs), int(args.batch)):
        index = screen_pairs[start : start + int(args.batch)]
        tokens_batch = torch.from_numpy(tokens[index].astype(np.int64)).to(device)
        index_batch = torch.from_numpy(index).to(device)
        labels_batch = torch.from_numpy(labels[index].astype(np.int64)).to(device)
        frame = _render_eval(model, fold, tokens_batch, index_batch)
        logits = _seg_logits(segnet, _exact_r_camera(frame))
        loss = _expected_flip(logits, labels_batch, float(args.tau)).mean() * (
            len(index) / len(screen_pairs)
        )
        loss.backward()
    flat_grad = np.concatenate(
        [fold.latent[n].grad.detach().to("cpu").numpy().ravel() for n in names]
    )
    magnitude = np.abs(flat_grad)
    ranking = str(args.rank)
    if ranking == "abs_desc":
        order = np.argsort(-magnitude)
    elif ranking == "abs_asc":
        nonzero = np.flatnonzero(magnitude > 0)
        order = nonzero[np.argsort(magnitude[nonzero])]
    elif ranking.startswith("band:"):
        lo_pct, hi_pct = (float(v) for v in ranking.split(":", 1)[1].split(","))
        lo, hi = np.percentile(magnitude, [lo_pct, hi_pct])
        inside = np.flatnonzero((magnitude >= lo) & (magnitude <= hi))
        order = inside[np.argsort(-magnitude[inside])]
    else:
        raise Rw1Error(f"unknown --rank {ranking!r}")

    current = base_flat.copy()
    screen_null = _evaluate_subset(
        model, fold, segnet, tokens, labels, device, screen_pairs, latent=as_latent(current)
    )
    best_screen = screen_null
    evaluations = 1
    accepted: list[int] = []
    rows = []
    pool = order[: int(args.pool)]
    group = int(args.group)
    queue = [pool[i : i + group] for i in range(0, len(pool), group)]
    while queue and evaluations < int(args.max_evaluations):
        candidate_idx = queue.pop(0)
        if candidate_idx.size == 0:
            continue
        trial = current.copy()
        move = -np.sign(flat_grad[candidate_idx]).astype(np.int64)
        trial[candidate_idx] = np.clip(
            trial[candidate_idx] + move, CODE_MIN, CODE_MAX
        )
        if np.array_equal(trial, current):
            continue
        flips = _evaluate_subset(
            model, fold, segnet, tokens, labels, device, screen_pairs, latent=as_latent(trial)
        )
        evaluations += 1
        improved = flips < best_screen
        row = {
            "group_size": int(candidate_idx.size),
            "screen_flips": int(flips),
            "screen_best": int(best_screen),
            "accepted": bool(improved),
            "evaluations": evaluations,
            "accepted_codes": len(accepted),
            "elapsed_seconds": time.perf_counter() - started,
        }
        if improved:
            current = trial
            best_screen = flips
            accepted.extend(int(v) for v in candidate_idx)
        elif candidate_idx.size > 1:
            half = candidate_idx.size // 2
            queue.insert(0, candidate_idx[half:])
            queue.insert(0, candidate_idx[:half])
            row["bisected"] = True
        rows.append(row)
        if args.progress:
            print(json.dumps(row), flush=True)

    # CONFIRM at n600 -- a screened result is not a verdict.
    with torch.no_grad():
        n600_null = _evaluate_realized(model, fold, segnet, tokens, labels, device)
        n600_final = _evaluate_realized(
            model, fold, segnet, tokens, labels, device, latent=as_latent(current)
        )
    changed = int((current != base_flat).sum())
    repaired = n600_null["flips"] - n600_final["flips"]
    rate_bytes = BYTES_PER_CHANGED_CODE * changed + (
        CONTAINER_BREAK_FIXED_BYTES if changed else 0.0
    )
    result = {
        "schema": "ddm_rw1_code_search.v1",
        "axis": f"[{args.device} research-signal; screened on a seeded RANDOM subset, "
        "confirmed at n600 realized argmax]",
        "score_claim": False,
        "pointer": pointer,
        "rank": ranking,
        "pool": int(args.pool),
        "group": group,
        "tau": float(args.tau),
        "screen_pairs": len(screen_pairs),
        "screen_null_flips": int(screen_null),
        "screen_best_flips": int(best_screen),
        "evaluations": evaluations,
        "accepted_codes": changed,
        "n600_null_flips": n600_null["flips"],
        "n600_final_flips": n600_final["flips"],
        "cells_repaired_n600": repaired,
        "rate_bytes_predicted": rate_bytes,
        "dS_rate": rate_bytes * RATE_PER_BYTE,
        "dS_seg": -repaired * S_PER_SEG_CELL,
        "dS_seg_plus_rate": -repaired * S_PER_SEG_CELL + rate_bytes * RATE_PER_BYTE,
        "prereg_falsifier_cells": 139,
        "falsifier_fired": bool(repaired < 139),
        **_stake(-repaired),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    np.save(Path(args.out).with_suffix(".codes.npy"), current.astype(np.int8))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


# ----------------------------------------------------------------------------------
# mode=scale-search -- the SUB-CODE actuator: per-row fp16 scales, realized acceptance
# ----------------------------------------------------------------------------------


def _fp16_step(values: np.ndarray, k: int) -> np.ndarray:
    """Move positive fp16 values by exactly ``k`` ULPs, on the fp16 grid itself.

    For a positive float16 the bit pattern read as uint16 is MONOTONE, so adding k walks
    exactly k representable values.  Doing it this way rather than by multiplying by
    (1+delta) matters: the archive can only carry an fp16, so a move that is not a whole
    number of ULPs is not a move the receiver can express.
    """
    bits = np.asarray(values, dtype=np.float16).view(np.uint16).astype(np.int64) + int(k)
    bits = np.clip(bits, 1, 0x7BFF)  # stay positive and finite
    return bits.astype(np.uint16).view(np.float16)


def cmd_scale_search(args) -> int:
    """Greedy realized acceptance over per-row fp16 SCALE moves.

    The actuator ddm_rw1's closing law pointed at: ``weight[i,j] = code[i,j]*scale[i]``,
    so moving one scale by k ULPs moves a whole row by a CODE-PROPORTIONAL fraction of a
    code step.  MEASURED before launch (`receipts/PREREG_SCALE_SEARCH.json`): one ULP is
    0.0242 of a code move on ``blocks.3.dw`` (41x finer), 0.141 on ``blocks.3.pw``
    (7.1x finer) and 1.342 on ``head`` (COARSER, so head is expected to behave like the
    code actuator and is excluded from the default pool).

    Acceptance is realized, never predicted, so the search cannot lose: worst case it
    returns the base object and the door closes on a clean negative.
    """
    import torch

    started = time.perf_counter()
    pointer = verify_live_pointer()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    section = load_semantic_section()
    names = trainable_names(bool(args.widened))
    check_trainable(section, names)
    pool_names = tuple(
        n for n in names if n in {v.strip() for v in str(args.tensors).split(",")}
    )
    if not pool_names:
        raise Rw1Error(f"--tensors {args.tensors!r} selected none of {names}")
    model = load_live_renderer(section).to(device)
    tokens = load_live_tokens()
    labels = load_gt_seg_dali()
    segnet = jg1.load_segnet().to(device).eval()
    for param in segnet.parameters():
        param.requires_grad_(False)
    fold = CodeFoldBack(section, names, device)
    base_scales = {n: section.scales[n].astype(np.float16).copy() for n in names}
    current = {n: base_scales[n].copy() for n in names}

    def apply(scales: dict[str, np.ndarray]) -> None:
        for name in names:
            run = section.runs[name]
            shape = [1] * len(run.shape)
            shape[0] = scales[name].size
            fold.scales[name] = (
                torch.from_numpy(scales[name].astype(np.float32).reshape(shape))
            ).to(device)

    rng = np.random.default_rng(int(args.seed))
    screen = (
        np.sort(rng.choice(N_PAIRS, size=int(args.screen_pairs), replace=False))
        if int(args.screen_pairs)
        else np.arange(N_PAIRS, dtype=np.int64)
    )
    apply(current)
    screen_null = _evaluate_subset(
        model, fold, segnet, tokens, labels, device, screen
    )
    best = screen_null
    steps = [int(v) for v in str(args.ulp_steps).split(",") if v.strip()]
    proposals = [(n, int(i)) for n in pool_names for i in range(base_scales[n].size)]
    rng.shuffle(proposals)

    evaluations = 1
    accepted: list[dict[str, Any]] = []
    rows = []
    for name, index in proposals:
        if evaluations >= int(args.max_evaluations):
            break
        for k in steps:
            if evaluations >= int(args.max_evaluations):
                break
            trial = {n: current[n].copy() for n in names}
            moved = _fp16_step(trial[name][index : index + 1], k)
            if moved[0] == trial[name][index]:
                continue
            trial[name][index] = moved[0]
            apply(trial)
            flips = _evaluate_subset(
                model, fold, segnet, tokens, labels, device, screen
            )
            evaluations += 1
            improved = flips < best
            rows.append(
                {
                    "tensor": name,
                    "row": index,
                    "ulp": k,
                    "screen_flips": int(flips),
                    "screen_best": int(best),
                    "delta": int(flips - best),
                    "accepted": bool(improved),
                    "evaluations": evaluations,
                    "accepted_rows": len(accepted),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if args.progress:
                print(json.dumps(rows[-1]), flush=True)
            if improved:
                current = trial
                best = flips
                accepted.append({"tensor": name, "row": index, "ulp": k})
                break
        apply(current)

    # CONFIRM at n600 and PRICE by a REAL container-searched encode.
    apply(base_scales)
    with torch.no_grad():
        n600_null = _evaluate_realized(model, fold, segnet, tokens, labels, device)
    apply(current)
    with torch.no_grad():
        n600_final = _evaluate_realized(model, fold, segnet, tokens, labels, device)
    changed = sum(int((current[n] != base_scales[n]).sum()) for n in names)
    code_edits = {n: section.codes[n] for n in names}
    priced_base = semantic_section_bytes(section, code_edits)
    stream = section.stream_with_scales(code_edits, {n: current[n] for n in names})
    priced = _price_stream(stream)
    rate_bytes = priced["searched_bytes"] - priced_base["searched_bytes"]
    repaired = n600_null["flips"] - n600_final["flips"]
    break_even_cells = rate_bytes * RATE_PER_BYTE / S_PER_SEG_CELL
    result = {
        "schema": "ddm_rw1_scale_search.v1",
        "axis": f"[{args.device} research-signal; screened on a seeded RANDOM subset, "
        "confirmed at n600 realized argmax]",
        "score_claim": False,
        "pointer": pointer,
        "tensors": list(pool_names),
        "ulp_steps": steps,
        "screen_pairs": len(screen),
        "screen_null_flips": int(screen_null),
        "screen_best_flips": int(best),
        "evaluations": evaluations,
        "proposals_available": len(proposals) * len(steps),
        "accepted_rows": len(accepted),
        "accepted": accepted,
        "changed_scales": changed,
        "n600_null_flips": n600_null["flips"],
        "n600_final_flips": n600_final["flips"],
        "cells_repaired_n600": repaired,
        "rate_bytes_measured": rate_bytes,
        "rate_break_even_cells": break_even_cells,
        "dS_seg": -repaired * S_PER_SEG_CELL,
        "dS_rate": rate_bytes * RATE_PER_BYTE,
        "dS_seg_plus_rate": -repaired * S_PER_SEG_CELL + rate_bytes * RATE_PER_BYTE,
        "falsifier_fired": bool(repaired < break_even_cells),
        **_stake(-repaired),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    np.savez(
        Path(args.out).with_suffix(".scales.npz"),
        **{n.replace(".", "__"): current[n] for n in names},
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
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

    prep = sub.add_parser("prep")
    prep.add_argument("--out", type=Path, default=WORK / "receipts/PREP.json")
    prep.add_argument("--out-dir", type=Path, default=WORK / "prep")
    prep.add_argument("--batch", type=int, default=25)
    prep.add_argument("--progress", action="store_true", default=True)
    common(prep)
    prep.set_defaults(func=cmd_prep)

    train = sub.add_parser("train")
    train.add_argument("--out", type=Path, default=WORK / "receipts/TRAIN.json")
    train.add_argument("--run-dir", type=Path, default=WORK / "runs/foldback")
    train.add_argument("--geometry", type=Path, default=WORK / "prep/pose_geometry.npz")
    train.add_argument("--device", default="mps")
    train.add_argument("--steps", type=int, default=3000)
    train.add_argument("--batch", type=int, default=4)
    train.add_argument("--lr", type=float, default=6.7e-4)
    train.add_argument("--seed", type=int, default=20260909)
    train.add_argument("--log-every", type=int, default=25)
    train.add_argument("--eval-every", type=int, default=300)
    train.add_argument("--checkpoint-every", type=int, default=300)
    train.add_argument("--resume-from", type=Path, default=None)
    train.add_argument("--full-field", action="store_true")
    common(train)
    train.set_defaults(func=cmd_train)

    export = sub.add_parser("export")
    export.add_argument("--out", type=Path, default=WORK / "receipts/EXPORT.json")
    export.add_argument("--out-dir", type=Path, default=WORK / "candidate")
    export.add_argument("--checkpoint", type=Path, default=None)
    export.add_argument("--weights", default="shadow", choices=("shadow", "latent"))
    export.add_argument("--carrier-codes", type=Path, default=None)
    export.add_argument("--no-container-search", action="store_true")
    export.add_argument("--no-verify", action="store_true")
    common(export)
    export.set_defaults(func=cmd_export)

    render = sub.add_parser("render")
    render.add_argument("--out", type=Path, default=WORK / "receipts/RENDER.json")
    render.add_argument("--out-dir", type=Path, default=BULK / "renders/candidate")
    render.add_argument("--checkpoint", type=Path, default=None)
    render.add_argument("--weights", default="shadow", choices=("shadow", "latent"))
    render.add_argument("--shard-index", type=int, default=0)
    render.add_argument("--shard-count", type=int, default=1)
    render.add_argument("--seed-only", action="store_true")
    render.add_argument("--progress", action="store_true", default=True)
    common(render)
    render.set_defaults(func=cmd_render)

    seg = sub.add_parser("seg")
    seg.add_argument("--out", type=Path, required=True)
    seg.add_argument("--out-argmax", type=Path, required=True)
    seg.add_argument("--raw", type=Path, required=True)
    seg.add_argument("--shard-index", type=int, default=0)
    seg.add_argument("--shard-count", type=int, default=1)
    seg.add_argument("--progress", action="store_true", default=True)
    common(seg)
    seg.set_defaults(func=cmd_seg)

    seg_merge = sub.add_parser("seg-merge")
    seg_merge.add_argument("--shards", nargs="+", required=True)
    seg_merge.add_argument("--out", type=Path, required=True)
    seg_merge.set_defaults(func=cmd_seg_merge)

    pose = sub.add_parser("pose")
    pose.add_argument("--out", type=Path, required=True)
    pose.add_argument("--out-rows", type=Path, required=True)
    pose.add_argument("--raw", type=Path, required=True)
    pose.add_argument("--shard-index", type=int, default=0)
    pose.add_argument("--shard-count", type=int, default=1)
    pose.add_argument("--outer-rounds", type=int, default=40)
    pose.add_argument("--max-gn-iterations", type=int, default=400)
    pose.add_argument("--base-mean-d-pose", type=float, default=LIVE_D_POSE)
    pose.add_argument("--resume", action="store_true", default=True)
    pose.add_argument("--progress", action="store_true", default=True)
    common(pose)
    pose.set_defaults(func=cmd_pose)

    admit = sub.add_parser("admit")
    admit.add_argument("--out", type=Path, default=WORK / "receipts/ADMISSION.json")
    admit.add_argument("--out-dir", type=Path, default=WORK / "candidate")
    admit.add_argument("--seg", type=Path, required=True)
    admit.add_argument("--pose-rows", nargs="+", required=True)
    admit.add_argument("--checkpoint", type=Path, required=True)
    admit.add_argument("--weights", default="shadow", choices=("shadow", "latent"))
    common(admit)
    admit.set_defaults(func=cmd_admit)

    rate = sub.add_parser("rate-law")
    rate.add_argument("--out", type=Path, default=WORK / "receipts/RATE_LAW.json")
    rate.add_argument("--counts", default="1,10,50,200,1000,3000,6000,12672")
    rate.add_argument("--repeats", type=int, default=3)
    rate.add_argument("--seed", type=int, default=20260909)
    rate.add_argument("--progress", action="store_true", default=True)
    common(rate)
    rate.set_defaults(func=cmd_rate_law)

    rebase = sub.add_parser("rebase")
    rebase.add_argument("--tree", type=Path, required=True)
    rebase.add_argument("--raw", type=Path, required=True)
    rebase.add_argument("--field", type=Path, required=True)
    rebase.add_argument("--argmax", type=Path, required=True)
    rebase.add_argument("--shared-source", type=Path, default=None)
    rebase.add_argument("--d-seg-cells", type=int, required=True)
    rebase.add_argument("--d-seg-t4", type=float, required=True)
    rebase.add_argument("--d-pose", type=float, required=True)
    rebase.add_argument("--allow-semantic-change", action="store_true")
    rebase.set_defaults(func=cmd_rebase)

    perturb = sub.add_parser("perturb-control")
    perturb.add_argument("--out", type=Path, default=WORK / "receipts/PERTURB.json")
    perturb.add_argument("--device", default="mps")
    perturb.add_argument("--counts", default="4,32,144")
    perturb.add_argument("--draws", type=int, default=2)
    perturb.add_argument("--seed", type=int, default=20260909)
    common(perturb)
    perturb.set_defaults(func=cmd_perturb_control)

    topk = sub.add_parser("gradient-topk")
    topk.add_argument("--out", type=Path, default=WORK / "receipts/GRADIENT_TOPK.json")
    topk.add_argument("--device", default="mps")
    topk.add_argument("--batch", type=int, default=4)
    topk.add_argument("--tau", type=float, default=TAU_REFERENCE)
    topk.add_argument("--k", default="1,4,16,64,256")
    topk.add_argument("--rank", default="abs_desc")
    topk.add_argument("--seed", type=int, default=20260909)
    topk.add_argument("--progress", action="store_true", default=True)
    common(topk)
    topk.set_defaults(func=cmd_gradient_topk)

    search = sub.add_parser("code-search")
    search.add_argument("--out", type=Path, default=WORK / "receipts/CODE_SEARCH.json")
    search.add_argument("--device", default="mps")
    search.add_argument("--batch", type=int, default=4)
    search.add_argument("--tau", type=float, default=TAU_REFERENCE)
    search.add_argument("--rank", default="abs_asc")
    search.add_argument("--pool", type=int, default=1024)
    search.add_argument("--group", type=int, default=64)
    search.add_argument("--screen-pairs", type=int, default=120)
    search.add_argument("--max-evaluations", type=int, default=120)
    search.add_argument("--seed", type=int, default=20260909)
    search.add_argument("--progress", action="store_true", default=True)
    common(search)
    search.set_defaults(func=cmd_code_search)

    scale = sub.add_parser("scale-search")
    scale.add_argument("--out", type=Path, default=WORK / "receipts/SCALE_SEARCH.json")
    scale.add_argument("--device", default="mps")
    scale.add_argument("--batch", type=int, default=4)
    scale.add_argument(
        "--tensors", default="blocks.3.dw.weight,blocks.3.pw.weight"
    )
    scale.add_argument("--ulp-steps", default="-1,1,-2,2")
    scale.add_argument("--screen-pairs", type=int, default=120)
    scale.add_argument("--max-evaluations", type=int, default=400)
    scale.add_argument("--seed", type=int, default=20260909)
    scale.add_argument("--progress", action="store_true", default=True)
    common(scale)
    scale.set_defaults(func=cmd_scale_search)

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
