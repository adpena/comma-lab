#!/usr/bin/env python3
"""ddm_pr1 -- run the terminal pose re-solve on a RENDERER-change candidate.

WHY THIS ARM EXISTS
-------------------
``ddm_ft1`` measured the shipped renderer's seg/pose coupling at
``|d(d_pose)|/|d(d_seg)| = 217.30`` and closed the seg-only renderer fine-tune by
arithmetic.  That closure was drawn BEFORE the terminal pose re-solve, which the
shipping chain runs after every seg change: ``FIRE_ORDER.sh:36-48`` gated FO-2 on
"realized d_seg DOWN" and it never ran.  The closing table therefore substituted
``ddm_jg5``'s **8.0x** carrier recovery, which jg5 measured for TOKEN edits
(``ddm_jg5_pose_resolve_on_edited_renders_20260819.md`` Sec 6.1) -- a TRANSFERRED
factor, not a measured one for this actuator.  A renderer weight change moves all
600 renders coherently; a token edit moves 573 of them locally.  Nothing says the
recovery transfers.  This module measures it.

WHAT IS REUSED, VERBATIM
------------------------
The solver is ``ddm_jg5.refine_pair`` -- br1's damped Gauss-Newton on the shipped
12-dim basis and the shipped signed-int12 lattice, alternated with the +-2 polish
under jg5's DERIVED materiality stopping rule.  That is the OPTIMAL FORM of this
family: jg5 Sec 4 records that ``ddm_up2``'s +-2-only search radius is a truncation
br1 was built to escape, so running up2's ``solve_pair_realized`` alone here would
measure the SOLVER's weakness and report it as the CARRIER's ceiling
([[caps_genus_trajectory_stopping_20260805]]).  ``--solver up2`` is available as a
labelled control, never as the headline.

THE ONE THING THAT CHANGES: THE ODD FRAMES
------------------------------------------
``br1.load_instrument`` reads odd frames from a decoded ``0.raw``.  A renderer
change rewrites every odd frame, and no ``0.raw`` of a candidate renderer exists,
so this module renders them on demand from the SHIPPED receiver's own
``SemanticTokenRenderer`` through ``ddm_ft1_verdict_bhw_pose.master_frames`` and
feeds them to the unchanged solver through a raw-shaped adapter.  Frame 0 stays
the shipped carrier path (``up2.render_frame0``: carrier render THEN the frame-0
selector), so the composition is the receiver's, including the 5 selector-active
pairs that ``ddm_ft1``'s verdict instrument does not apply.

AUTHORITY
---------
Frozen CPU-torch PoseNet on DALI-lineage GT -- the lineage ``upstream/evaluate.py``
scores on the contest-CUDA axis (``evaluate.py:31-42`` + the two asserts in
``frame_utils.py``).  ``[macOS-CPU advisory]``, ``score_claim=false``,
``promotable=false``.  Only ``upstream/evaluate.py`` on contest hardware is a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

#: The AFR1 frontier body this arm measures against.  Named and sha-gated, never
#: assumed: a delta against a superseded body double-counts a banked gain
#: ([[a_delta_without_its_baseline_is_unanchored_and_baselines_move_20260803]]).
FRONTIER_ARCHIVE_SHA256 = (
    "cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25"
)
FRONTIER_ARCHIVE_BYTES = 180_002

#: The AFR1 contest-CUDA T4 n600 receipt legs
#: (``experiments/results/modal_auth_eval_mirror/
#: contest_auth_eval_modal-ddm_afr1_tile48_groupbin8_cuda_n600_20260831.json``).
AFR1_D_SEG_T4 = 0.00020139
AFR1_D_POSE_T4 = 6.37e-06
AFR1_SCORE_T4 = 0.14797617125559104
AFR1_POSE_LEG_T4 = 0.00798123

N_PAIRS = 600
POSE_DIMS = 6
DEFAULT_BATCH = 8

#: The shipped carrier's signed-int12 lattice (``ddm_up2`` COEFF_CODE_MIN/MAX).
COEFF_CODE_MIN, COEFF_CODE_MAX = -2048, 2047


class Pr1Error(RuntimeError):
    """A ddm_pr1 precondition failed.  Fail closed, never approximate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pose_leg(d_pose: float) -> float:
    return math.sqrt(10.0 * d_pose)


def composed_score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + pose_leg(d_pose) + 25.0 * archive_bytes / 37_545_489.0


def payable_pose_ceiling(delta_d_seg: float) -> float:
    """Largest d_pose a seg cut of ``|delta_d_seg|`` can fund at ``dB = 0``.

    From the AFR1 receipt: the move is paid entirely on distortion, so it clears
    iff ``sqrt(10*d_pose_new) < pose_leg_T4 + 100*|delta_d_seg|``.  Solved for
    ``d_pose_new``.  A non-negative ``delta_d_seg`` (seg got worse or stood still)
    has no ceiling to speak of and returns the base pose leg's own d_pose.
    """
    budget = AFR1_POSE_LEG_T4 + 100.0 * max(-delta_d_seg, 0.0)
    return budget * budget / 10.0


# --------------------------------------------------------------------------
# The odd-frame source: the candidate renderer's own frames, raw-shaped.
# --------------------------------------------------------------------------


class RenderedOddFrames:
    """A ``0.raw``-shaped view whose ODD frames come from a named renderer state.

    ``br1.evaluate_codes`` / ``up2.measure_pose`` / ``jg5.refine_pair`` all read
    frame 1 as ``raw[2 * pair_index + 1]`` and never touch even frames, so a
    duck-typed object that answers exactly that query lets the solver run
    UNCHANGED against a renderer that has no decoded ``0.raw``.  Even indices are
    refused rather than silently served: frame 0 is the carrier's, and anything
    asking this object for it has a bug.
    """

    def __init__(self, renderer_state, tokens: np.ndarray, *, cache_pairs: int = 32):
        import torch

        from experiments.ddm_ft1_identity_gate_and_caches import (
            SEMANTIC_WIDTH,
            load_shipped_renderer_module,
        )

        shipped = load_shipped_renderer_module()
        model = shipped.SemanticTokenRenderer(SEMANTIC_WIDTH)
        model.load_state_dict(dict(renderer_state), strict=True)
        self._model = model.eval()
        self._tokens = tokens
        self._torch = torch
        self._cache: dict[int, np.ndarray] = {}
        self._cache_pairs = max(1, int(cache_pairs))
        self.renders = 0

    def _render(self, pair: int) -> np.ndarray:
        from experiments.ddm_ft1_verdict_bhw_pose import master_frames

        torch = self._torch
        tokens = torch.from_numpy(self._tokens[[pair]].astype(np.int64))
        with torch.no_grad():
            frame = master_frames(self._model, tokens, torch.tensor([pair]))
        self.renders += 1
        return frame[0].permute(1, 2, 0).to(torch.uint8).numpy()

    def _get(self, pair: int) -> np.ndarray:
        cached = self._cache.get(pair)
        if cached is None:
            if len(self._cache) >= self._cache_pairs:
                self._cache.clear()
            cached = self._render(pair)
            self._cache[pair] = cached
        return cached

    def __getitem__(self, key) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(key, dtype=np.int64))
        if np.any(indices % 2 != 1):
            raise Pr1Error(
                "RenderedOddFrames serves ODD (frame-1) indices only; frame 0 is "
                "the carrier's and must come from up2.render_frame0"
            )
        pairs = (indices - 1) // 2
        if np.any(pairs < 0) or np.any(pairs >= N_PAIRS):
            raise Pr1Error(f"pair index out of range in {indices.tolist()}")
        return np.stack([self._get(int(p)) for p in pairs])

    def field_sha256(self, pairs: np.ndarray) -> str:
        """sha256 of the rendered odd-frame field, so byte-identity is provable.

        The field itself is 1.83 GB and is deterministically rebuildable from the
        retained 36,130 B section plus ``tokens.u8``; the digest is what a
        successor needs to prove it rebuilt the same bytes.
        """
        digest = hashlib.sha256()
        for pair in pairs:
            digest.update(self._get(int(pair)).tobytes())
        return digest.hexdigest()


# --------------------------------------------------------------------------
# Instrument assembly.
# --------------------------------------------------------------------------


def load_renderer_state(source: Path):
    """Realized renderer state, read back through the SHIPPED receiver's parser.

    ``source`` is either an SM3R section ``.bin`` (the candidate) or the frontier
    archive itself (the incumbent).  Either way the state returned is what the
    receiver would LOAD, never a trainer checkpoint's own weights: ft1 Sec 6
    measured the export discarding 190 of 192 FiLM rows per pruned tensor, so a
    checkpoint's state is a model that never ships.
    """
    from experiments.ddm_ft1_identity_gate_and_caches import (
        SEMANTIC_WIDTH,
        load_shipped_renderer_module,
        read_semantic_section,
    )

    shipped = load_shipped_renderer_module()
    template = shipped.SemanticTokenRenderer(SEMANTIC_WIDTH).state_dict()
    source = Path(source)
    blob = (
        bytes(read_semantic_section(source))
        if source.suffix == ".zip"
        else source.read_bytes()
    )
    parsed = shipped.unpack_variant_semantic_or_none(blob, template)
    if parsed is None:
        raise Pr1Error(f"shipped receiver rejected the semantic section at {source}")
    return parsed, {
        "source": str(source),
        "section_bytes": len(blob),
        "section_sha256": hashlib.sha256(blob).hexdigest(),
    }


def load_tokens(path: Path) -> np.ndarray:
    tokens = np.fromfile(path, dtype=np.uint8)
    expected = N_PAIRS * 384 * 512
    if tokens.size != expected:
        raise Pr1Error(f"token field is {tokens.size} bytes, expected {expected}")
    return tokens.reshape(N_PAIRS, 384, 512)


def build_instrument(*, runtime: Path, gt_cache: Path, axis: str, renderer_source: Path,
                     tokens_path: Path, archive_sha256: str | None = FRONTIER_ARCHIVE_SHA256):
    """A ``br1.Instrument`` whose odd frames come from ``renderer_source``."""
    import ddm_br1_pose_basis_reorientation as br1
    import ddm_up2_shipping_pose_solve as up2

    runtime = Path(runtime)
    observed = sha256_file(runtime / "archive.zip")
    if archive_sha256 and observed != archive_sha256:
        raise Pr1Error(
            f"runtime archive sha256 {observed} != expected {archive_sha256}; "
            "refusing to measure a delta against an unidentified body"
        )
    state = up2.load_carrier_state(runtime, verify_archive=False)
    if state.has_compensation:
        raise Pr1Error(
            "this body carries a compensation overlay; the re-solve would have to "
            "compose with it and this arm has not measured that path"
        )
    targets, lineage = up2.load_gt_poses(Path(gt_cache))
    gate = up2.verify_gt_lineage(axis=axis, declared_lineage=lineage)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    renderer_state, renderer_meta = load_renderer_state(renderer_source)
    raw = RenderedOddFrames(renderer_state, load_tokens(Path(tokens_path)))
    instrument = br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)
    meta = {
        "runtime_dir": str(runtime),
        "archive_sha256": observed,
        "gt_cache": str(gt_cache),
        "gt_lineage_gate": gate,
        "renderer": renderer_meta,
        "selector_active_pairs": np.flatnonzero(state.selector_choices != 0).tolist(),
        "has_compensation": bool(state.has_compensation),
        "shipped_codes_sha256": sha256_array(state.codes),
    }
    return instrument, meta


def select_pairs(pairs: int, seed: int) -> np.ndarray:
    """Full field at ``pairs >= 600``, else a SEEDED RANDOM sample -- never a prefix.

    A contiguous prefix of this video measures the POSE axis 2.54-4.21x HARDER
    than the population ([[m96]]), which is exactly the axis this arm reports.
    """
    if pairs >= N_PAIRS:
        return np.arange(N_PAIRS, dtype=np.int64)
    if pairs < 1:
        raise Pr1Error(f"pairs must be >= 1, got {pairs}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N_PAIRS, size=pairs, replace=False)).astype(np.int64)


def shard_of(pairs: np.ndarray, shard_index: int, shard_count: int) -> np.ndarray:
    """STRIDED shard, never a contiguous block -- a dead shard leaves an unbiased
    partial rather than a scene-block one."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise Pr1Error(f"bad shard {shard_index}/{shard_count}")
    return pairs[shard_index::shard_count]


# --------------------------------------------------------------------------
# mode=measure -- per-pair d_pose at ONE declared batch shape.
# --------------------------------------------------------------------------


def run_measure(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    instrument, meta = build_instrument(
        runtime=args.runtime, gt_cache=args.gt_cache, axis=args.axis,
        renderer_source=args.renderer, tokens_path=args.tokens,
        archive_sha256=getattr(args, "expect_archive_sha256", None)
        or FRONTIER_ARCHIVE_SHA256,
    )
    pairs = select_pairs(args.pairs, args.seed)
    codes = instrument.state.codes.copy()
    if args.codes:
        override = np.load(args.codes).astype(np.int32)
        if override.shape != codes.shape:
            raise Pr1Error(f"codes {override.shape} != shipped {codes.shape}")
        codes = override
    coefficients = up2.codes_to_coefficients(codes, instrument.state.coefficient_scales)
    started = time.time()
    per_pair, poses = up2.measure_pose(
        instrument.posenet, instrument.state, coefficients, instrument.raw,
        instrument.targets, pairs, batch_size=args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_dir = out.parent / f"{out.stem}_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    per_pair_path = payload_dir / "per_pair_d_pose.npy"
    poses_path = payload_dir / "pose_vectors.npy"
    codes_path = payload_dir / "codes.npy"
    np.save(per_pair_path, per_pair)
    np.save(poses_path, poses)
    np.save(codes_path, codes)
    # The odd-frame field is 1.83 GB and is deterministically rebuildable from the
    # retained section plus tokens.u8, so it is not persisted -- but its digest is,
    # so a successor can PROVE it rebuilt the same bytes rather than assume it.
    render_digest = instrument.raw.field_sha256(pairs) if args.render_digest else None
    report = {
        "schema": "tac.ddm_pr1.measure.v1",
        "odd_frame_field_sha256": render_digest,
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "instrument": meta,
        "batch_size": args.batch_size,
        "pairs": len(pairs),
        "pair_selection": (
            "full n600" if len(pairs) >= N_PAIRS
            else f"seeded random draw of {len(pairs)}/600 (seed {args.seed}); NOT a prefix"
        ),
        "codes_source": str(args.codes) if args.codes else "shipped carrier codes",
        "codes_sha256": sha256_array(codes),
        "d_pose_mean": float(per_pair.mean()),
        "d_pose_median": float(np.median(per_pair)),
        "pose_leg": pose_leg(float(per_pair.mean())),
        "elapsed_seconds": time.time() - started,
        "payload": {
            "per_pair_d_pose": {
                "path": str(per_pair_path), "sha256": sha256_array(per_pair),
                "bytes": per_pair_path.stat().st_size,
            },
            "pose_vectors": {
                "path": str(poses_path), "sha256": sha256_array(poses),
                "bytes": poses_path.stat().st_size,
            },
            "codes": {
                "path": str(codes_path), "sha256": sha256_array(codes),
                "bytes": codes_path.stat().st_size,
            },
        },
        "pairs_index": pairs.tolist(),
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("label", "pairs", "d_pose_mean", "pose_leg", "elapsed_seconds")}, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=solve -- the terminal re-solve, resumable, one row per pair.
# --------------------------------------------------------------------------


def load_done(rows_path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if rows_path.is_file():
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[int(row["pair"])] = row
    return done


def run_solve(args) -> int:
    import ddm_jg5_pose_resolve_on_edited_renders as jg5
    import ddm_up2_shipping_pose_solve as up2

    instrument, meta = build_instrument(
        runtime=args.runtime, gt_cache=args.gt_cache, axis=args.axis,
        renderer_source=args.renderer, tokens_path=args.tokens,
        archive_sha256=getattr(args, "expect_archive_sha256", None)
        or FRONTIER_ARCHIVE_SHA256,
    )
    pairs = shard_of(select_pairs(args.pairs, args.seed), args.shard_index, args.shard_count)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    done = load_done(rows_path)
    # DERIVED, never hand-set: the materiality floor is the measured T4 band
    # equal-allocated over the population and converted through the exact
    # dS/dd_i at the operating point the solve is aiming AT (the AFR1 receipt's
    # own d_pose), not the inflated pre-solve mean.  Evaluating it at the start
    # mean would raise the floor by sqrt(start/target) and stop the solver early.
    dd_threshold = jg5.materiality_dd_threshold(args.materiality_operating_point)
    started = time.time()
    with rows_path.open("a", encoding="utf-8") as stream:
        for position, pair in enumerate(pairs):
            if int(pair) in done:
                continue
            start_codes = instrument.state.codes[int(pair)]
            if args.solver == "jg5":
                row = jg5.refine_pair(
                    instrument, int(pair), start_codes, dd_threshold=dd_threshold,
                    outer_rounds=args.outer_rounds, max_gn_iterations=args.max_gn_iterations,
                )
            else:
                row = up2.solve_pair_realized(
                    instrument.posenet, instrument.state, instrument.raw,
                    instrument.targets[int(pair)], int(pair), start_codes,
                )
                row["stop_reason"] = "up2_no_improving_neighbour"
            row["solver"] = args.solver
            done[int(pair)] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            elapsed = time.time() - started
            remaining = len(pairs) - sum(1 for p in pairs if int(p) in done)
            print(
                f"[{len(pairs) - remaining}/{len(pairs)}] pair={int(pair)} "
                f"start={row['start_d_pose']:.6e} final={row['final_d_pose']:.6e} "
                f"recov={row['start_d_pose'] / max(row['final_d_pose'], 1e-30):.3g}x "
                f"stop={row['stop_reason']} elapsed={elapsed / 60:.1f}m "
                f"eta={elapsed / max(1, position + 1) * remaining / 60:.1f}m",
                flush=True,
            )
    ordered = [done[int(p)] for p in pairs if int(p) in done]
    summary = {
        "schema": "tac.ddm_pr1.solve.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "solver": args.solver,
        "solver_reference_form": (
            "ddm_jg5.refine_pair (br1 damped Gauss-Newton on the shipped 12-dim "
            "basis and int12 lattice, +-2 polish, jg5 derived materiality stop)"
            if args.solver == "jg5" else
            "ddm_up2.solve_pair_realized (+-2 single-coordinate greedy; jg5 Sec 4 "
            "records this radius as a TRUNCATION -- control only, never the headline)"
        ),
        "instrument": meta,
        "materiality_operating_point_d_pose": args.materiality_operating_point,
        "dd_threshold": dd_threshold,
        "shard": [args.shard_index, args.shard_count],
        "pairs": len(ordered),
        "start_d_pose_mean": float(np.mean([r["start_d_pose"] for r in ordered])) if ordered else None,
        "final_d_pose_mean": float(np.mean([r["final_d_pose"] for r in ordered])) if ordered else None,
        "pairs_improved": int(sum(1 for r in ordered if r["final_d_pose"] < r["start_d_pose"])),
        "stop_reasons": {
            reason: int(sum(1 for r in ordered if r.get("stop_reason") == reason))
            for reason in sorted({r.get("stop_reason", "unknown") for r in ordered})
        },
        "total_changed_coordinates": int(sum(r["changed_coordinates"] for r in ordered)),
        "total_evaluations": int(sum(r["evaluations"] for r in ordered)),
        "rows_path": str(rows_path),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("label", "pairs", "start_d_pose_mean", "final_d_pose_mean",
                       "stop_reasons", "elapsed_seconds")}, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=codes -- merge solved rows into a full 600x12 code table.
# --------------------------------------------------------------------------


def run_codes(args) -> int:
    import ddm_up2_shipping_pose_solve as up2

    runtime = Path(args.runtime)
    observed = sha256_file(runtime / "archive.zip")
    if observed != FRONTIER_ARCHIVE_SHA256:
        raise Pr1Error(
            f"runtime archive sha256 {observed} != {FRONTIER_ARCHIVE_SHA256}; the "
            "unsolved rows would be another body's carrier codes"
        )
    state = up2.load_carrier_state(runtime, verify_archive=False)
    codes = state.codes.astype(np.int32).copy()
    merged: dict[int, list[int]] = {}
    for path in args.rows:
        for pair, row in load_done(Path(path)).items():
            merged[int(pair)] = row["codes"]
    for pair, row_codes in merged.items():
        codes[pair] = np.asarray(row_codes, dtype=np.int32)
    # FAIL CLOSED on a partial merge. A missing pair silently keeps the SHIPPED
    # code, so an incomplete shard would understate the recovery without any
    # symptom in the output -- the quiet-wrong-number shape this arm exists to
    # avoid. --allow-partial makes an interim read explicit and labelled.
    if len(merged) < N_PAIRS and not args.allow_partial:
        raise Pr1Error(
            f"merged {len(merged)} of {N_PAIRS} pairs; the missing pairs would keep "
            "the shipped codes and understate the recovery. Pass --allow-partial "
            "to read an interim table, and label it partial wherever it is quoted."
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, codes)
    # The re-solve is NOT free. ft1's FIRE_ORDER calls it "0 archive bytes"; jg5
    # measured a MIXED carrier splice at +45 B. A full 600-pair re-solve rewrites
    # every Rice residual, so the byte delta is priced here by building the
    # payload, with the shipped-payload reproduction as the anchor control.
    price, _ = up2.price_full_resolve_bytes(runtime, codes)
    record = {
        "schema": "tac.ddm_pr1.codes.v1",
        "path": str(out),
        "rows_merged_from": [str(p) for p in args.rows],
        "runtime_archive_sha256": observed,
        "carrier_rice_price": price,
        "delta_score_rate": price["delta_bytes"] * 25.0 / 37_545_489.0,
        "pairs_written": len(merged),
        "complete_n600": len(merged) == N_PAIRS,
        "shipped_codes_sha256": sha256_array(state.codes),
        "codes_sha256": sha256_array(codes),
        "changed_pairs": int((codes != state.codes).any(axis=1).sum()),
        "changed_coordinates": int((codes != state.codes).sum()),
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=report -- the coupling, the recovery distribution, the closing arithmetic.
# --------------------------------------------------------------------------


def _load_measure(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "tac.ddm_pr1.measure.v1":
        raise Pr1Error(f"{path} is not a ddm_pr1 measure report")
    return payload


def _per_pair(measure: dict[str, Any]) -> np.ndarray:
    return np.load(measure["payload"]["per_pair_d_pose"]["path"])


def _same_instrument(*measures: dict[str, Any]) -> None:
    """Refuse to difference rows that were not taken on ONE instrument.

    jg5 Sec 4b measured this forward moving with the batch shape, so a
    before/after pair taken at different shapes, GT lineages, batch sizes or
    pair sets is a CROSS-instrument comparison wearing a delta's clothes.
    """
    keys = {
        (
            m["batch_size"],
            tuple(m["pairs_index"]),
            m["instrument"]["gt_cache"],
            m["instrument"]["archive_sha256"],
        )
        for m in measures
    }
    if len(keys) != 1:
        raise Pr1Error(
            "measures differ in batch shape, pair set, GT lineage or body; "
            "differencing them would be a cross-instrument comparison"
        )


def solver_diagnostics(row_paths, after_pp: np.ndarray, pairs: np.ndarray) -> dict[str, Any]:
    """Why the surviving residue survives, read off the solver's own rows.

    ``demanded_code_units_max`` is the Gauss-Newton step the pose residual asks
    for, in signed-int12 code units.  The lattice runs [-2048, 2047], so a demand
    in the thousands says the correction the residual needs is NOT REPRESENTABLE
    in the shipped 12-dim basis at the shipped quantisation -- a REPRESENTATION
    limit, not a search limit.  Distinguishing the two is the whole difference
    between "solve harder" and "the carrier cannot reach it".
    """
    if not row_paths:
        return {"rows_available": False}
    rows: dict[int, dict[str, Any]] = {}
    for path in row_paths:
        rows.update(load_done(Path(path)))
    ordered = [rows[int(p)] for p in pairs if int(p) in rows]
    if not ordered:
        return {"rows_available": False}
    demanded = np.array(
        [float(r.get("demanded_code_units_max", float("nan"))) for r in ordered]
    )
    finite = demanded[np.isfinite(demanded)]
    worst = np.argsort(after_pp)[-10:][::-1] if len(after_pp) == len(pairs) else []
    return {
        "rows_available": True,
        "rows": len(ordered),
        "stop_reasons": {
            reason: int(sum(1 for r in ordered if r.get("stop_reason") == reason))
            for reason in sorted({r.get("stop_reason", "unknown") for r in ordered})
        },
        "lattice_range": [COEFF_CODE_MIN, COEFF_CODE_MAX],
        "lattice_span_code_units": COEFF_CODE_MAX - COEFF_CODE_MIN,
        "up2_search_radius_code_units": 2,
        "demanded_code_units_max": {
            "median": float(np.median(finite)) if finite.size else None,
            "p90": float(np.quantile(finite, 0.9)) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "fraction_exceeding_the_lattice_span": (
                float((finite > (COEFF_CODE_MAX - COEFF_CODE_MIN)).mean())
                if finite.size else None
            ),
            "fraction_exceeding_up2_search_radius": (
                float((finite > 2).mean()) if finite.size else None
            ),
        },
        "changed_coordinates_mean": float(
            np.mean([r.get("changed_coordinates", 0) for r in ordered])
        ),
        "worst_10_pairs_after_re_solve": [
            {
                "pair": int(pairs[i]),
                "d_pose_after": float(after_pp[i]),
                "demanded_code_units_max": (
                    float(rows[int(pairs[i])].get("demanded_code_units_max", float("nan")))
                    if int(pairs[i]) in rows else None
                ),
                "stop_reason": (
                    rows[int(pairs[i])].get("stop_reason") if int(pairs[i]) in rows else None
                ),
            }
            for i in worst
        ],
    }


def run_report(args) -> int:
    base = _load_measure(args.base_measure)
    before = _load_measure(args.before_measure)
    after = _load_measure(args.after_measure)
    _same_instrument(base, before, after)
    base_pp, before_pp, after_pp = _per_pair(base), _per_pair(before), _per_pair(after)
    pairs = np.asarray(base["pairs_index"], dtype=np.int64)

    ratio = before_pp / np.maximum(after_pp, 1e-30)
    improved = after_pp < before_pp
    mean_before, mean_after = float(before_pp.mean()), float(after_pp.mean())
    mean_base = float(base_pp.mean())
    delta_d_seg = float(args.delta_d_seg)
    k_pre = abs(mean_before - mean_base) / abs(delta_d_seg)
    k_post = (mean_after - mean_base) / abs(delta_d_seg)

    # The closing arithmetic, re-derived at a named seg cut rather than copied.
    cut = abs(float(args.seg_cut_fraction)) * AFR1_D_SEG_T4
    ceiling = payable_pose_ceiling(-cut)
    predicted_after_at_cut = AFR1_D_POSE_T4 + k_post * cut
    k_post_payable_bar = (ceiling - AFR1_D_POSE_T4) / cut

    report = {
        "schema": "tac.ddm_pr1.report.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "pairs": len(pairs),
        "pair_selection": base["pair_selection"],
        "batch_size": base["batch_size"],
        "instrument": base["instrument"],
        "inputs": {
            "base_measure": str(args.base_measure),
            "before_measure": str(args.before_measure),
            "after_measure": str(args.after_measure),
        },
        "d_pose": {
            "base_as_shipped": mean_base,
            "candidate_stale_carrier": mean_before,
            "candidate_re_solved": mean_after,
            "afr1_t4_receipt": AFR1_D_POSE_T4,
            "base_vs_t4_relative": mean_base / AFR1_D_POSE_T4 - 1.0,
        },
        "recovery": {
            "mean_based": mean_before / mean_after if mean_after else float("inf"),
            "median_per_pair": float(np.median(ratio)),
            "geometric_mean_per_pair": float(np.exp(np.log(np.maximum(ratio, 1e-30)).mean())),
            "pairs_improved": int(improved.sum()),
            "pairs_unchanged_or_worse": int((~improved).sum()),
            "quantiles_per_pair": {
                str(q): float(np.quantile(ratio, q))
                for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
            },
            "share_of_mean_after_from_worst_10_pairs": float(
                np.sort(after_pp)[-10:].sum() / after_pp.sum()
            ) if after_pp.sum() > 0 else None,
            "jg5_transferred_factor": 8.0,
            "fcd2_transferred_factor": 5.87,
        },
        "coupling": {
            "delta_d_seg_used": delta_d_seg,
            "delta_d_seg_source": args.delta_d_seg_source,
            "pre_re_solve": k_pre,
            "post_re_solve": k_post,
            "ft1_pre_re_solve_n200": 217.30366224024704,
            "rf1_pre_re_solve_n600": 166.80837961844966,
        },
        "why_the_residue_survives": solver_diagnostics(args.rows, after_pp, pairs),
        "closing_arithmetic": {
            "seg_cut_fraction": abs(float(args.seg_cut_fraction)),
            "seg_cut_d_seg": cut,
            "payable_pose_ceiling": ceiling,
            "predicted_d_pose_at_that_cut": predicted_after_at_cut,
            "overshoot_multiple": predicted_after_at_cut / ceiling,
            "k_post_payable_bar": k_post_payable_bar,
            "payable": bool(predicted_after_at_cut <= ceiling),
            "assumption": (
                "DIRECTION SYMMETRY: the measured candidate moved d_seg UP; applying "
                "k_post to a seg DECREASE assumes local linearity of the realized map "
                "around the shipped weights. Stated, not measured."
            ),
        },
        "charter_prediction": {
            "predicted_post_re_solve_coupling_below": 20.0,
            "prediction_holds": bool(k_post < 20.0),
            "falsifier_recovery_below": 3.0,
            "falsifier_fired": bool((mean_before / mean_after if mean_after else float("inf")) < 3.0),
            "note": (
                "the charter's success band (k_post < 20) and PAYABILITY "
                "(k_post <= the bar above) are different events and are reported apart"
            ),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Persist the READY record first, then the bulk payload, then enrich the record: a failed
    # bulk save must never strand the run's only readable product (write-order gate,
    # tac.confound_gates.check_no_bulk_write_strands_the_ready_record).
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    payload_dir = out.parent / f"{out.stem}_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    ratio_path = payload_dir / "per_pair_recovery_ratio.npy"
    try:
        np.save(ratio_path, ratio)
        report["payload"] = {
            "per_pair_recovery_ratio": {
                "path": str(ratio_path), "sha256": sha256_array(ratio),
                "bytes": ratio_path.stat().st_size,
            }
        }
    except OSError as exc:
        report["payload_error"] = f"{type(exc).__name__}: {exc}"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


# --------------------------------------------------------------------------
# mode=selector -- can the PER-PAIR frame-0 selector reach what the carrier cannot?
# --------------------------------------------------------------------------


def run_selector(args) -> int:
    """Sweep the shipped selector's 8 modes on the pairs that own the pose leg.

    The re-solve's surviving residue is concentrated on a handful of pairs whose
    Gauss-Newton step demands thousands of int12 code units -- a correction the
    12-dim carrier basis cannot represent (see ``solver_diagnostics``).  The
    frame-0 selector is a DIFFERENT actuator on the same frame: a per-pair
    integer pixel op (``runtime/frame0_selector.py`` SPARSE_PIXEL_MODES --
    identity, +-1 luma, a channel tilt, two rolls, two tile dithers), chosen
    independently per pair and shipped in a sparse combinatorial-rank blob.

    So the question this mode answers is narrow and cheap: on the pairs the
    carrier cannot reach, does ANY of the 7 non-identity modes lower d_pose?
    A "no" closes the selector as a rescue for this residue.  A "yes" is a
    per-pair actuator the renderer axis was told it did not have.

    Nothing here ships: the sweep changes ``selector_choices`` in memory only,
    and the byte cost of a changed selector is reported, never assumed free.
    """
    import ddm_up2_shipping_pose_solve as up2

    instrument, meta = build_instrument(
        runtime=args.runtime, gt_cache=args.gt_cache, axis=args.axis,
        renderer_source=args.renderer, tokens_path=args.tokens,
        archive_sha256=getattr(args, "expect_archive_sha256", None)
        or FRONTIER_ARCHIVE_SHA256,
    )
    state = instrument.state
    codes = state.codes.copy()
    if args.codes:
        override = np.load(args.codes).astype(np.int32)
        if override.shape != codes.shape:
            raise Pr1Error(f"codes {override.shape} != shipped {codes.shape}")
        codes = override
    pairs = np.asarray(args.pairs_list, dtype=np.int64)
    if pairs.size == 0:
        raise Pr1Error("selector sweep needs at least one pair")
    if np.any(pairs < 0) or np.any(pairs >= N_PAIRS):
        raise Pr1Error(f"pair index out of range in {pairs.tolist()}")
    modes = state.selector_modes
    if not modes:
        raise Pr1Error("this body ships no selector mode table to sweep")

    coefficients = up2.codes_to_coefficients(codes, state.coefficient_scales)
    original_choices = state.selector_choices.copy()
    rows = []
    started = time.time()
    for pair in pairs:
        per_mode = []
        for mode_index in range(len(modes)):
            state.selector_choices[int(pair)] = np.uint8(mode_index)
            value, _ = up2.measure_pose(
                instrument.posenet, state, coefficients, instrument.raw,
                instrument.targets, np.array([int(pair)], dtype=np.int64),
                batch_size=1,
            )
            per_mode.append(float(value[0]))
        state.selector_choices[int(pair)] = original_choices[int(pair)]
        shipped_mode = int(original_choices[int(pair)])
        best_mode = int(np.argmin(per_mode))
        rows.append({
            "pair": int(pair),
            "shipped_mode": shipped_mode,
            "d_pose_at_shipped_mode": per_mode[shipped_mode],
            "best_mode": best_mode,
            "d_pose_at_best_mode": per_mode[best_mode],
            "gain": per_mode[shipped_mode] - per_mode[best_mode],
            "ratio": (
                per_mode[shipped_mode] / per_mode[best_mode]
                if per_mode[best_mode] > 0 else float("inf")
            ),
            "d_pose_per_mode": per_mode,
        })
        print(json.dumps(rows[-1]), flush=True)
    gained = float(sum(r["gain"] for r in rows))
    newly_active = sum(
        1 for r in rows if r["best_mode"] != 0 and r["shipped_mode"] == 0
    )
    report = {
        "schema": "tac.ddm_pr1.selector_sweep.v1",
        "axis": "[macOS-CPU advisory, frozen CPU-torch PoseNet]",
        "score_claim": False,
        "promotable": False,
        "label": args.label,
        "instrument": meta,
        "codes_source": str(args.codes) if args.codes else "shipped carrier codes",
        "codes_sha256": sha256_array(codes),
        "batch_size": 1,
        "batch_shape_caveat": (
            "a single-pair evaluation is batch 1 BY CONSTRUCTION, and jg5 Sec 4b "
            "measured this forward moving with the batch shape. Within a pair the "
            "shipped-mode vs best-mode comparison is same-shape and the GAIN is "
            "valid; composing that gain against a batch-8 population mean is a "
            "cross-shape step. Any selector set adopted from this sweep MUST be "
            "re-measured at the declared batch shape before it is scored."
        ),
        "modes": [
            {"index": i, "kind": m.kind, "a": m.a, "b": m.b, "c": m.c}
            for i, m in enumerate(modes)
        ],
        "pairs_swept": pairs.tolist(),
        "rows": rows,
        "pairs_improved": int(sum(1 for r in rows if r["gain"] > 0)),
        "total_d_pose_gain_over_swept_pairs": gained,
        "n600_mean_gain_if_shipped": gained / N_PAIRS,
        "newly_active_pairs": newly_active,
        "byte_cost_note": (
            "the selector blob is a sparse combinatorial-rank encoding "
            "(runtime/frame0_selector.py): 7 header bytes + ceil(log2 C(600, k)) "
            "bits of rank + 3k bits of labels. Going from k=5 (14 B shipped) to "
            "k=15 costs about +12 B; the exact cost of any chosen set must be "
            "measured by encoding it, never assumed."
        ),
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "label", "pairs_improved", "total_d_pose_gain_over_swept_pairs",
        "n600_mean_gain_if_shipped", "newly_active_pairs", "elapsed_seconds")}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--runtime", type=Path, required=True)
        command.add_argument("--gt-cache", type=Path, required=True)
        command.add_argument("--axis", default="contest_cuda")
        command.add_argument("--renderer", type=Path, required=True)
        command.add_argument("--tokens", type=Path, required=True)
        command.add_argument("--pairs", type=int, default=N_PAIRS)
        command.add_argument("--seed", type=int, default=20260904)
        command.add_argument("--label", default="pr1")
        command.add_argument("--threads", type=int, default=4)
        command.add_argument(
            "--expect-archive-sha256", default=FRONTIER_ARCHIVE_SHA256,
            help=(
                "body this instrument is allowed to measure; defaults to the afr1 "
                "frontier this arm was written for. A SUCCESSOR body (ddm_fs1's "
                "candidate B and beyond) is measured by naming its sha here -- the "
                "gate stays closed, it just stops being frozen to one generation "
                "([[binding-instruction-numbers-expire-and-nobody-rederives-them]])."
            ),
        )

    measure = sub.add_parser("measure", help="per-pair d_pose at one declared batch shape")
    common(measure)
    measure.add_argument("--codes", type=Path, default=None)
    measure.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    measure.add_argument("--out", type=Path, required=True)
    measure.add_argument(
        "--render-digest", action="store_true",
        help="also record sha256 of the rendered odd-frame field (adds one render pass)",
    )

    solve = sub.add_parser("solve", help="terminal pose re-solve on the candidate's renders")
    common(solve)
    solve.add_argument("--out", type=Path, required=True)
    solve.add_argument("--solver", choices=("jg5", "up2"), default="jg5")
    solve.add_argument("--shard-index", type=int, default=0)
    solve.add_argument("--shard-count", type=int, default=1)
    solve.add_argument("--outer-rounds", type=int, default=40)
    solve.add_argument("--max-gn-iterations", type=int, default=400)
    solve.add_argument(
        "--materiality-operating-point", type=float, default=AFR1_D_POSE_T4,
        help="mean d_pose at which the DERIVED materiality floor is evaluated",
    )

    report = sub.add_parser("report", help="coupling + recovery + closing arithmetic")
    report.add_argument("--base-measure", type=Path, required=True)
    report.add_argument("--before-measure", type=Path, required=True)
    report.add_argument("--after-measure", type=Path, required=True)
    report.add_argument("--delta-d-seg", type=float, required=True)
    report.add_argument("--delta-d-seg-source", required=True)
    report.add_argument("--rows", nargs="*", default=[],
                        help="solver rows.jsonl files, for the representation-limit read")
    report.add_argument("--seg-cut-fraction", type=float, default=0.25)
    report.add_argument("--label", default="pr1_report")
    report.add_argument("--threads", type=int, default=4)
    report.add_argument("--out", type=Path, required=True)

    selector = sub.add_parser(
        "selector", help="sweep the per-pair frame-0 selector on named pairs")
    common(selector)
    selector.add_argument("--codes", type=Path, default=None)
    selector.add_argument("--pairs-list", type=int, nargs="+", required=True)
    selector.add_argument("--out", type=Path, required=True)

    codes = sub.add_parser("codes", help="merge solved rows into a 600x12 code table")
    codes.add_argument("--runtime", type=Path, required=True)
    codes.add_argument("--rows", nargs="+", required=True)
    codes.add_argument("--out", type=Path, required=True)
    codes.add_argument("--threads", type=int, default=4)
    codes.add_argument(
        "--allow-partial", action="store_true",
        help="merge fewer than 600 solved pairs (interim read; label it partial)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    threads = max(1, int(getattr(args, "threads", 4)))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    if args.command == "measure":
        return run_measure(args)
    if args.command == "solve":
        return run_solve(args)
    if args.command == "codes":
        return run_codes(args)
    if args.command == "report":
        return run_report(args)
    if args.command == "selector":
        return run_selector(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
