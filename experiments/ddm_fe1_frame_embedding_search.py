#!/usr/bin/env python3
"""ddm_fe1 -- per-pair FRAME-EMBEDDING pre-distortion on the shipped object.

The renderer's ``frame_embed`` is a per-pair FiLM code vector: one row of 8 signed
integer codes per pair, times 8 fp16 COLUMN scales, feeding the ``film`` linear of all
four ``TokenBlock``s for that pair only (``candidate_pass3/candidate_runtime/cpr1/
inflate.py:96,127`` and ``:81``).  Changing one row changes ONE pair's render and
nothing else, so admission is per pair -- the lever the renderer-weight arms (rf1/ft1/
pr1) did not have.

VERIFIED AT SOURCE on the LIVE pointer tree (not assumed from the charter):

* ``frame_embed.weight`` is ``(600, 8)`` -- ``cpr1/inflate.py:96`` with ``N = 600``
  (:20) and ``SEMANTIC_FRAME_DIM = 8`` (:28).
* The shipped SM3R mode-6 depth table gives ``frame_embed.weight`` **3 bits**, NOT the
  int4 the charter assumed: signed domain ``[-4, 3]`` = 8 alternatives per code, of
  which the shipped field uses 7 (``[-3, 3]``).  Measured by walking the restored SM3R
  body (``cpr1/rc1_adaptive_model_sections.py:249`` ``walk_sm3r``); codes run at body
  offset 466, length 1800 B, count 4800, bits 3.  The charter's "15 alternatives each"
  is therefore corrected to 7 live alternatives per code.
* Column scales (8 fp16, body offset 450): 1.208984, 1.010742, 1.009766, 1.0, 1.204102,
  1.045898, 0.873047, 1.068359.  So ONE code step is a change of ~1.0 in that
  dimension of an 8-vector whose entries are at most 3.6 -- a coarse lattice, and that
  coarseness is the arm's central risk, recorded before any search.

Nothing here re-implements a receiver: the render is ``jg1.render_frame1`` (the
receiver's own forward model, batch 1 by construction) and the verdict is the frozen
CPU-torch SegNet argmax through the evaluator's own preprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1
import ddm_up2_shipping_pose_solve as up2

N_PAIRS = jg1.N_PAIRS
EVAL_H, EVAL_W = jg1.EVAL_H, jg1.EVAL_W
FRAME_DIM = 8
CODE_BITS = 3
CODE_MIN, CODE_MAX = -(1 << (CODE_BITS - 1)), (1 << (CODE_BITS - 1)) - 1

# ----------------------------------------------------------------------------------
# THE LIVE POINTER BODY (2026-09-08): sj1 pass-3 token pre-distortion + carrier re-solve.
# ----------------------------------------------------------------------------------
SJ1_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
)
LIVE_TREE = SJ1_ROOT / "candidate_pass3/candidate_runtime"
LIVE_RUNTIME = LIVE_TREE / "runtime"
LIVE_ARCHIVE = LIVE_TREE / "archive.zip"
LIVE_ARCHIVE_SHA256 = (
    "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
)
LIVE_ARCHIVE_BYTES = 181_645
#: The token field the live receiver decodes (sj1's admitted pass-3 subset).
LIVE_FIELD = SJ1_ROOT / "admission_pass3/field_admitted.npz"
#: The live receiver's OWN decode of that archive.
LIVE_RAW = SJ1_ROOT / "candidate_pass3/parseback/0.raw"
#: sj1's own n600 argmax of that decode (its seg leg receipt).
LIVE_ARGMAX = SJ1_ROOT / "seg_final_pass3/argmax_n600.npy"

LIVE_SCORE_T4 = 0.13900437796841966
LIVE_D_SEG_T4 = 1.0913879636e-04
LIVE_D_SEG_LOCAL = 0.0001090664333767361  # 12,866 cells, cpu_torch argmax on DALI
LIVE_D_SEG_CELLS = 12_866
LIVE_D_POSE = 5.0928018072772644e-06
#: The instrument ratio sj1 used to carry a local seg leg onto T4 (its SEAL falsifier 2).
SEG_T4_RATIO = 1.0006634761602033

WORK = Path(
    "/Volumes/VertigoDataTier/pact/ddm_fe1_frame_embedding_predistortion"
)

RATE_PER_BYTE = 25.0 / 37_545_489.0


class Fe1Error(RuntimeError):
    """A fe1 identity or contract failed; the caller must stop."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_threads(threads: int) -> None:
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
    torch.set_num_interop_threads(1)


# ----------------------------------------------------------------------------------
# The semantic section, opened the way the receiver opens it
# ----------------------------------------------------------------------------------


@dataclass
class SemanticSection:
    """The live semantic section, split into the pieces a code edit needs."""

    rc1_stream: bytes
    shift: int
    sm3r_body: bytes
    codes: np.ndarray  # (600, 8) int8, the shipped frame_embed codes
    scales: np.ndarray  # (8,) float32, the shipped fp16 column scales
    code_offset: int
    code_length: int
    code_bits: int
    template: Any
    rc1: Any

    def body_with_codes(self, codes: np.ndarray) -> bytes:
        """The SM3R body with ``frame_embed`` replaced -- same length by construction."""
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape != (N_PAIRS, FRAME_DIM):
            raise Fe1Error(f"codes must be (600, 8), got {codes.shape}")
        if codes.min() < CODE_MIN or codes.max() > CODE_MAX:
            raise Fe1Error(
                f"codes escape the shipped signed {self.code_bits}-bit domain "
                f"[{CODE_MIN}, {CODE_MAX}]"
            )
        packed = self.rc1.pack_signed_codes(
            codes.ravel().astype(np.int32), self.code_bits
        )
        if len(packed) != self.code_length:
            raise Fe1Error(
                f"repacked frame_embed is {len(packed)} B, shipped run is "
                f"{self.code_length} B"
            )
        return (
            self.sm3r_body[: self.code_offset]
            + packed
            + self.sm3r_body[self.code_offset + self.code_length :]
        )

    def stream_with_codes(self, codes: np.ndarray) -> bytes:
        """The RC1 semantic stream carrying ``codes``, through the SHIPPED coder."""
        return self.rc1.apply_semantic(
            self.body_with_codes(codes), self.template, self.shift
        )


def _import_live(runtime_dir: Path = LIVE_RUNTIME):
    added = [str(runtime_dir.parent), str(runtime_dir.parent / "cpr1")]
    for entry in added:
        if entry not in sys.path:
            sys.path.insert(0, entry)
    ra = importlib.import_module("runtime.residual_archive")
    rc1 = importlib.import_module("rc1_adaptive_model_sections")
    renderer = importlib.import_module("inflate")
    return ra, rc1, renderer


def load_semantic_section(
    archive_path: Path = LIVE_ARCHIVE, runtime_dir: Path = LIVE_RUNTIME
) -> SemanticSection:
    """Open the live semantic section and locate ``frame_embed``'s code run exactly.

    The run is found by DRIVING the receiver's own ``walk_sm3r`` over the restored body
    (not by a hand-rolled offset table), and the whole open is refused unless
    ``apply_semantic(restore_semantic(stream)) == stream`` byte for byte.
    """
    ra, rc1, renderer = _import_live(runtime_dir)
    parts = ra.read_residual_archive(archive_path)
    stream = bytes(parts.semantic_blob)
    if not stream.startswith(rc1.SEMANTIC_MAGIC):
        raise Fe1Error("live semantic section does not carry the RC1 rider")
    _magic, _version, shift, _payload_len = rc1.RC1_HEADER.unpack_from(stream)
    template = renderer.SemanticTokenRenderer(96).state_dict()
    body = rc1.restore_semantic(stream, template)
    if rc1.apply_semantic(body, template, shift) != stream:
        raise Fe1Error(
            "RC1 semantic round-trip is not byte-identical; the coder this arm would "
            "price with is not the coder the archive ships"
        )

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
        raise Fe1Error(f"SM3R walk ended at {cursor} of {len(body)}")

    # plan and offsets are in body order and the same length.
    index = 1  # plan[0] is the depth table
    located: dict[str, dict[str, tuple[int, int]]] = {}
    for name, value in template.items():
        if value.ndim < 2:
            index += 1
            continue
        if name in rc1.ROW_PRUNE_NAMES:
            index += 1  # prune mask
        scale_item, scale_off = plan[index], offsets[index]
        index += 1
        code_item, code_off = plan[index], offsets[index]
        index += 1
        if code_item["kind"] != "codes":
            raise Fe1Error(f"{name}: expected a codes run, got {code_item['kind']}")
        located[name] = {
            "scales": (scale_off, scale_item["length"]),
            "codes": (code_off, code_item["length"]),
            "bits": code_item["bits"],
            "count": code_item["count"],
        }

    fe = located["frame_embed.weight"]
    bits = int(fe["bits"])
    if bits != CODE_BITS:
        raise Fe1Error(
            f"frame_embed depth is {bits} bits on this tree, module pinned {CODE_BITS}"
        )
    scale_off, scale_len = fe["scales"]
    scales = np.frombuffer(body[scale_off : scale_off + scale_len], dtype="<f2")
    code_off, code_len = fe["codes"]
    codes = rc1.unpack_signed_codes(
        body[code_off : code_off + code_len], int(fe["count"]), bits
    ).reshape(N_PAIRS, FRAME_DIM)

    section = SemanticSection(
        rc1_stream=stream,
        shift=int(shift),
        sm3r_body=body,
        codes=codes.astype(np.int8),
        scales=np.asarray(scales, dtype=np.float32),
        code_offset=int(code_off),
        code_length=int(code_len),
        code_bits=bits,
        template=template,
        rc1=rc1,
    )
    if section.stream_with_codes(section.codes) != stream:
        raise Fe1Error("re-encoding the SHIPPED codes did not reproduce the stream")
    return section


# ----------------------------------------------------------------------------------
# The body
# ----------------------------------------------------------------------------------


@dataclass
class Body:
    """The live frontier body: renderer, scorer, token field, GT table, decode."""

    semantic: Any
    net: Any
    tokens: np.ndarray
    gt: np.ndarray
    section: SemanticSection
    raw: Any = None
    receipts: dict[str, Any] = field(default_factory=dict)


def load_token_field(path: Path = LIVE_FIELD) -> np.ndarray:
    """The live token field as ``(600, 384, 512) uint8``.

    sj1 stores its admitted field as one npz plane per pair keyed by the pair index as
    a string; a missing key would silently revert that pair to a DIFFERENT field, which
    is the fourth silent-revert class sj1 sec.17 records, so every pair is required.
    """
    with np.load(path) as blob:
        planes = []
        for pair in range(N_PAIRS):
            key = str(pair)
            if key not in blob:
                raise Fe1Error(f"token field {path} has no plane for pair {pair}")
            plane = blob[key]
            if plane.shape != (EVAL_H, EVAL_W) or plane.dtype != np.uint8:
                raise Fe1Error(
                    f"pair {pair} plane is {plane.shape}/{plane.dtype}, expected "
                    f"({EVAL_H}, {EVAL_W})/uint8"
                )
            planes.append(plane)
    return np.ascontiguousarray(np.stack(planes))


def load_body(
    *,
    with_raw: bool = False,
    verify_shas: bool = True,
    lineage: str = up2.LINEAGE_DALI,
    axis: str = "contest_cuda",
) -> Body:
    up2.verify_gt_lineage(axis=axis, declared_lineage=lineage)
    receipts: dict[str, Any] = {
        "axis": axis,
        "gt_lineage": lineage,
        "live_archive": str(LIVE_ARCHIVE),
        "live_field": str(LIVE_FIELD),
        "live_raw": str(LIVE_RAW),
    }
    if verify_shas:
        archive_sha = _sha256_file(LIVE_ARCHIVE)
        if archive_sha != LIVE_ARCHIVE_SHA256:
            raise Fe1Error(
                f"live archive sha {archive_sha} != pointer {LIVE_ARCHIVE_SHA256}"
            )
        if LIVE_ARCHIVE.stat().st_size != LIVE_ARCHIVE_BYTES:
            raise Fe1Error("live archive byte count is not the pointer's")
        receipts["live_archive_sha256"] = archive_sha
    tokens = load_token_field()
    gt = jg1.load_gt_seg_labels(lineage)
    semantic = jg1.load_semantic_renderer(
        archive_path=LIVE_ARCHIVE, runtime_dir=LIVE_RUNTIME
    )
    section = load_semantic_section()
    weight = semantic.frame_embed.weight.detach().numpy()
    recon = section.codes.astype(np.float32) * section.scales[None, :]
    if not np.array_equal(weight, recon):
        raise Fe1Error(
            "the codes located in the SM3R body do not reconstruct the renderer's "
            "loaded frame_embed weight"
        )
    net = jg1.load_segnet()
    raw = None
    if with_raw:
        raw = _open_raw(LIVE_RAW)
    return Body(
        semantic=semantic,
        net=net,
        tokens=tokens,
        gt=gt,
        section=section,
        raw=raw,
        receipts=receipts,
    )


def _open_raw(path: Path):
    expected = 2 * N_PAIRS * jg1.CAMERA_H * jg1.CAMERA_W * 3
    actual = path.stat().st_size
    if actual != expected:
        raise Fe1Error(f"decode {path} is {actual} B, expected {expected} B")
    return np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS, jg1.CAMERA_H, jg1.CAMERA_W, 3),
    )


# ----------------------------------------------------------------------------------
# Rendering one pair under an edited embedding
# ----------------------------------------------------------------------------------


def set_pair_codes(body: Body, pair: int, codes_row: Sequence[int]) -> None:
    """Write one pair's frame_embed row from integer codes, through the shipped scales."""
    import torch

    row = np.asarray(codes_row, dtype=np.float32)
    if row.shape != (FRAME_DIM,):
        raise Fe1Error(f"codes_row must be ({FRAME_DIM},), got {row.shape}")
    value = torch.from_numpy(row * body.section.scales)
    with torch.no_grad():
        body.semantic.frame_embed.weight[int(pair)] = value


def restore_pair_codes(body: Body, pair: int) -> None:
    set_pair_codes(body, pair, body.section.codes[int(pair)])


def render_pair(body: Body, pair: int) -> np.ndarray:
    """Frame ``2p+1`` for one pair, the receiver's own forward model (batch 1)."""
    return jg1.render_frame1(
        body.semantic, body.tokens[int(pair)][None], np.array([int(pair)])
    )


def argmax_pair(body: Body, pair: int) -> np.ndarray:
    return jg1.argmax_from_camera_frames(body.net, render_pair(body, pair))[0]


def flips_pair(argmax: np.ndarray, body: Body, pair: int) -> int:
    return int((argmax != body.gt[int(pair)]).sum())


# ----------------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------------


def _pairs_from_args(args) -> np.ndarray:
    if args.pairs:
        return np.array([int(p) for p in args.pairs.split(",")], dtype=np.int64)
    start, stop = int(args.start), int(args.stop)
    return np.arange(start, min(stop, N_PAIRS), dtype=np.int64)


def cmd_control(args) -> int:
    """Identity control: does re-rendering with the SHIPPED embedding reproduce 0.raw?"""
    _set_threads(args.threads)
    body = load_body(with_raw=True)
    pairs = _pairs_from_args(args)
    rows = []
    worst = 0
    for pair in pairs:
        rendered = render_pair(body, int(pair))[0]
        shipped = np.asarray(body.raw[2 * int(pair) + 1])
        delta = np.abs(rendered.astype(np.int16) - shipped.astype(np.int16))
        worst = max(worst, int(delta.max()))
        rows.append(
            {
                "pair": int(pair),
                "pixels": int(shipped.size),
                "pixels_changed": int((delta > 0).sum()),
                "max_abs_delta": int(delta.max()),
            }
        )
    result = {
        "schema": "ddm_fe1_control.v1",
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "pairs": [int(p) for p in pairs],
        "rows": rows,
        "byte_exact": worst == 0,
        "max_abs_delta": worst,
        "receipts": body.receipts,
        "semantic_batch": 1,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0 if worst == 0 else 3


def cmd_step0(args) -> int:
    """Reproduce the live seg leg from the receiver's own decode, one shard of pairs."""
    _set_threads(args.threads)
    body = load_body(with_raw=True, verify_shas=not args.no_sha)
    pairs = _pairs_from_args(args)
    argmax = np.empty((len(pairs), EVAL_H, EVAL_W), dtype=np.uint8)
    flips = np.zeros(len(pairs), dtype=np.int64)
    started = time.time()
    for row, pair in enumerate(pairs):
        frame = np.asarray(body.raw[2 * int(pair) + 1])[None]
        argmax[row] = jg1.argmax_from_camera_frames(body.net, frame)[0]
        flips[row] = int((argmax[row] != body.gt[int(pair)]).sum())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".argmax.npy"), argmax)
    result = {
        "schema": "ddm_fe1_step0.v1",
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "pairs": [int(p) for p in pairs],
        "flips_total": int(flips.sum()),
        "flips_per_pair": flips.tolist(),
        "cells": int(len(pairs) * EVAL_H * EVAL_W),
        "d_seg_shard": float(flips.sum() / (len(pairs) * EVAL_H * EVAL_W)),
        "elapsed_s": round(time.time() - started, 1),
        "receipts": body.receipts,
    }
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "flips_per_pair"}, indent=1))
    return 0


def _seeded_pairs(count: int, seed: int) -> np.ndarray:
    """A reproducible random pair sample -- SIZING only, never a verdict (n600 law)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N_PAIRS, size=count, replace=False)).astype(np.int64)


def single_code_moves(base_row: np.ndarray) -> list[tuple[int, int]]:
    """Every single-code alternative for one pair: ``(dim, new_code)``.

    The live depth is 3 bits, so the domain is ``[-4, 3]`` = 8 values and each dim
    offers 7 alternatives -- the charter's "15 alternatives each" was written for an
    int4 field and is corrected here against the shipped depth table.
    """
    moves: list[tuple[int, int]] = []
    for dim in range(FRAME_DIM):
        for code in range(CODE_MIN, CODE_MAX + 1):
            if code != int(base_row[dim]):
                moves.append((dim, code))
    return moves


def cmd_size(args) -> int:
    """SIZING: every single-code move on a seeded pair sample, realized flips each.

    This sizes the pass.  It produces NO verdict: the charter's stop rule reads the
    per-pair RANGE, and an admitted set is only ever measured at n600.
    """
    _set_threads(args.threads)
    body = load_body(with_raw=False, verify_shas=not args.no_sha)
    pairs = (
        _pairs_from_args(args)
        if args.pairs
        else _seeded_pairs(args.count, args.seed)
    )
    rows = []
    started = time.time()
    for pair in pairs:
        pair = int(pair)
        base_row = body.section.codes[pair].astype(np.int64)
        restore_pair_codes(body, pair)
        base_flips = flips_pair(argmax_pair(body, pair), body, pair)
        entries = []
        for dim, code in single_code_moves(base_row):
            row = base_row.copy()
            row[dim] = code
            set_pair_codes(body, pair, row)
            flips = flips_pair(argmax_pair(body, pair), body, pair)
            entries.append(
                {
                    "dim": int(dim),
                    "old": int(base_row[dim]),
                    "new": int(code),
                    "flips": int(flips),
                    "delta": int(flips - base_flips),
                }
            )
        restore_pair_codes(body, pair)
        deltas = np.array([e["delta"] for e in entries], dtype=np.int64)
        best = int(deltas.min())
        rows.append(
            {
                "pair": pair,
                "base_flips": int(base_flips),
                "base_row": base_row.tolist(),
                "moves": len(entries),
                "best_delta": best,
                "best_move": entries[int(deltas.argmin())],
                "worst_delta": int(deltas.max()),
                "reducing_moves": int((deltas < 0).sum()),
                "neutral_moves": int((deltas == 0).sum()),
                "best_relative": (
                    float(best / base_flips) if base_flips else None
                ),
                "range_relative": (
                    float((deltas.max() - deltas.min()) / base_flips)
                    if base_flips
                    else None
                ),
                "entries": entries,
            }
        )
        print(
            f"pair {pair}: base {base_flips} best_delta {best} "
            f"({100.0 * best / base_flips:.1f}%) reducing {int((deltas < 0).sum())}"
            f"/{len(entries)} worst +{int(deltas.max())}",
            flush=True,
        )
    relatives = [abs(r["best_relative"]) for r in rows if r["best_relative"] is not None]
    inert = bool(relatives) and max(relatives) < 0.02
    result = {
        "schema": "ddm_fe1_size.v1",
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "scope": "SIZING (12-pair seeded sample); sizes the pass, produces no verdict",
        "seed": int(args.seed),
        "pairs": [int(p) for p in pairs],
        "stop_rule": (
            "axis INERT if the best single-code move changes flips by < 2% on EVERY "
            "sampled pair"
        ),
        "axis_inert": inert,
        "best_relative_max": (max(relatives) if relatives else None),
        "best_relative_min": (min(relatives) if relatives else None),
        "elapsed_s": round(time.time() - started, 1),
        "rows": rows,
        "receipts": body.receipts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("rows", "receipts")}, indent=1
        )
    )
    return 0


#: SCOPE reduction, declared with its evidence: in the 12-pair sizing every one of the
#: 178 moves whose step size was 3 or more code units made the pair WORSE, minimum +4
#: flips, while both realized repairs came from steps of -1 and +2.  The n600 search
#: therefore enumerates steps of at most two code units.  This reduces SCOPE (how much
#: of the lattice is walked), never MECHANISM: acceptance stays the realized frozen
#: cpu_torch SegNet argmax on the receiver's own render.
MAX_STEP = 2


def stepped_moves(base_row: np.ndarray, max_step: int = MAX_STEP) -> list[tuple[int, int]]:
    moves: list[tuple[int, int]] = []
    for dim in range(FRAME_DIM):
        old = int(base_row[dim])
        for code in range(max(CODE_MIN, old - max_step), min(CODE_MAX, old + max_step) + 1):
            if code != old:
                moves.append((dim, code))
    return moves


def cmd_search(args) -> int:
    """n600 greedy per-pair search: best single code, then the best second given it.

    One JSONL row per pair, appended as the pair finishes, so a kill never orphans a
    pair without its move -- and a resume re-reads the rows rather than the search.
    """
    _set_threads(args.threads)
    body = load_body(with_raw=False, verify_shas=not args.no_sha)
    pairs = [
        int(p)
        for p in _pairs_from_args(args)
        if int(p) % args.shard_count == args.shard_index
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"search_rows_{args.shard_index}.jsonl"
    done: set[int] = set()
    if rows_path.is_file() and args.resume:
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(int(json.loads(line)["pair"]))
    started = time.time()
    completed = 0
    for pair in pairs:
        if pair in done:
            continue
        base_row = body.section.codes[pair].astype(np.int64)
        restore_pair_codes(body, pair)
        base_flips = flips_pair(argmax_pair(body, pair), body, pair)
        evaluations = 0
        singles = []
        for dim, code in stepped_moves(base_row):
            row = base_row.copy()
            row[dim] = code
            set_pair_codes(body, pair, row)
            flips = flips_pair(argmax_pair(body, pair), body, pair)
            evaluations += 1
            singles.append(
                {"dim": dim, "old": int(base_row[dim]), "new": code, "flips": int(flips)}
            )
        best_single = min(singles, key=lambda e: e["flips"])
        record = {
            "pair": pair,
            "base_flips": int(base_flips),
            "base_row": base_row.tolist(),
            "best_single": best_single,
            "best_single_delta": int(best_single["flips"] - base_flips),
            "reducing_singles": int(
                sum(1 for e in singles if e["flips"] < base_flips)
            ),
            "singles": singles,
            "second": None,
            "final_row": base_row.tolist(),
            "final_flips": int(base_flips),
            "final_delta": 0,
            "changed_codes": 0,
        }
        if best_single["flips"] < base_flips:
            first_row = base_row.copy()
            first_row[best_single["dim"]] = best_single["new"]
            record["final_row"] = first_row.tolist()
            record["final_flips"] = int(best_single["flips"])
            record["final_delta"] = int(best_single["flips"] - base_flips)
            record["changed_codes"] = 1
            seconds = []
            for dim, code in stepped_moves(first_row):
                if dim == best_single["dim"]:
                    continue
                row = first_row.copy()
                row[dim] = code
                set_pair_codes(body, pair, row)
                flips = flips_pair(argmax_pair(body, pair), body, pair)
                evaluations += 1
                seconds.append(
                    {
                        "dim": dim,
                        "old": int(first_row[dim]),
                        "new": code,
                        "flips": int(flips),
                    }
                )
            if seconds:
                best_second = min(seconds, key=lambda e: e["flips"])
                record["second"] = best_second
                if best_second["flips"] < best_single["flips"]:
                    second_row = first_row.copy()
                    second_row[best_second["dim"]] = best_second["new"]
                    record["final_row"] = second_row.tolist()
                    record["final_flips"] = int(best_second["flips"])
                    record["final_delta"] = int(best_second["flips"] - base_flips)
                    record["changed_codes"] = 2
        restore_pair_codes(body, pair)
        record["evaluations"] = evaluations
        with rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        completed += 1
        if args.progress:
            print(
                f"shard {args.shard_index} {completed}/{len(pairs) - len(done)} "
                f"pair {pair}: base {base_flips} final {record['final_flips']} "
                f"({record['final_delta']:+d}, {record['changed_codes']} codes) "
                f"{(time.time() - started) / completed:.1f} s/pair",
                flush=True,
            )
    # The search mutates the renderer in place several thousand times.  Prove it ended
    # holding exactly the shipped weights, or every row it wrote is suspect.
    final_weight = body.semantic.frame_embed.weight.detach().numpy()
    shipped = body.section.codes.astype(np.float32) * body.section.scales[None, :]
    if not np.array_equal(final_weight, shipped):
        raise Fe1Error(
            "the renderer does not hold the shipped frame_embed at the end of the "
            "shard; the rows written by this process cannot be trusted"
        )
    receipt = {
        "schema": "ddm_fe1_search_shard.v1",
        "restored_to_shipped_weights": True,
        "axis": "[macOS-CPU advisory, jg1/sj1 instrument, DALI GT lineage]",
        "score_claim": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_step": MAX_STEP,
        "pairs": pairs,
        "rows_path": str(rows_path),
        "elapsed_s": round(time.time() - started, 1),
        "receipts": body.receipts,
    }
    (out_dir / f"SEARCH_SHARD_{args.shard_index}.json").write_text(
        json.dumps(receipt, indent=1)
    )
    print(json.dumps({k: receipt[k] for k in ("shard_index", "elapsed_s")}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--pairs", default="", help="comma-separated pair indices")
        p.add_argument("--start", type=int, default=0)
        p.add_argument("--stop", type=int, default=N_PAIRS)
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--no-sha", action="store_true")

    control = sub.add_parser("control", help="identity control against the live 0.raw")
    common(control)
    control.add_argument("--out", default=str(WORK / "probe/CONTROL.json"))
    control.set_defaults(func=cmd_control)

    step0 = sub.add_parser("step0", help="reproduce the live seg leg from the decode")
    common(step0)
    step0.add_argument("--out", default=str(WORK / "probe/STEP0.json"))
    step0.set_defaults(func=cmd_step0)

    size = sub.add_parser("size", help="SIZING: single-code moves on a seeded sample")
    common(size)
    size.add_argument("--count", type=int, default=12)
    size.add_argument("--seed", type=int, default=20260908)
    size.add_argument("--out", default=str(WORK / "sizing/SIZE.json"))
    size.set_defaults(func=cmd_size)

    search = sub.add_parser("search", help="n600 greedy per-pair code search")
    common(search)
    search.add_argument("--shard-index", type=int, default=0)
    search.add_argument("--shard-count", type=int, default=1)
    search.add_argument("--resume", action="store_true", default=True)
    search.add_argument("--progress", action="store_true", default=True)
    search.add_argument("--out-dir", default=str(WORK / "search"))
    search.set_defaults(func=cmd_search)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
