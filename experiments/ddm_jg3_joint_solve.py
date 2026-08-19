#!/usr/bin/env python3
"""ddm_jg3 -- the rate-aware joint solve that ddm_jg2 specified and handed off.

WHAT THIS IS
------------
``ddm_jg1`` measured that a single-cell token coordinate move repairs ~1.55 argmax
cells per changed token (realized, through the receiver's own forward model and the
frozen CPU SegNet), that a token-only seg solve DESTROYS pose by ~387x, and that
re-running the carrier's own coordinate descent against the EDITED frame recovers
``d_pose`` to ~1.073x at ~0 bytes.  ``ddm_jg2`` then replaced the one modelled leg:
a re-encoder that reproduces the shipped RC64 stream BYTE-IDENTICALLY prices the
edit set at **4.1379 bits per changed token**, and measured that token-edit rate
costs **superpose** (union/sum 1.0258, exact at the archive layer).

This module executes the S2 spec.  It is a **rate-aware greedy descent with a
Lagrangian stopping rule** -- not "descend until converged", which is what worked
for the pose carrier where bytes were nearly free.  Here bytes are not free:

    accept a move only while  cells_repaired * 10.185 bits > cost_bits

``ddm_jg1`` S1e measured the yield DECAYING from 1.46-1.50 cells/token on a first
pass to **0.390** when one pair was iterated to exhaustion -- and 0.390 is BELOW
the absolute break-even of 0.406 cells/token, where the rate term exactly cancels
the seg term.  So the stopping rule is not a refinement; it is the whole result.

THE THREE-WAY PROPOSAL CLASS, AND THE ONE CLASS THIS MODULE DOES NOT IMPLEMENT
-----------------------------------------------------------------------------
The jg2 spec names three proposal classes per cell: ``edit`` / ``drop`` / ``keep``.
This module implements ``edit`` and ``keep``.  It does NOT implement ``drop``, and
the reason is structural rather than scheduling -- recorded here because a silent
omission would be the orphan bug:

``drop`` is rc4's high-confidence prediction substitution.  jg2 measured that it is
**not a token-field edit -- it is a RECEIVER CHANGE**: the decoder must know which
positions were dropped so it can substitute its own prediction.  The pointer body's
``cpr1/inflate.py`` has no such path, so a drop cannot be byte-closed against
``7ce46fd7...`` without shipping a new receiver, which would invalidate the entire
byte-identity control chain this seal rests on.  ``edit`` alone projects to
-0.0102 S against a 0.006526 gap, so the honest ordering is: measure whether edits
alone clear, and treat the joint edit+drop waterfill as owed headroom if they do
not.  Per ``ddm_bu1``'s law the joint solve would be strictly better; per the
byte-close chain it costs a new receiver.  Both are true.

Block/dilation moves are NOT a proposal class: ``ddm_jg1`` S1c measured them worse
at every radius (-55% at r=1, -351% at r=2).

THE PACKING, AND ITS CONTROL
----------------------------
A realized evaluation costs ~0.68 s (render 0.228 s at batch 1 -- byte-identity
requires it, ``ddm_up2`` sec.6 measured batch 8 as byte-changing -- plus a SegNet
forward).  A pair has ~60 flip sites and each site has up to 25 candidates, so the
naive sweep is ~17 min/pair = 170 h at n600.  ``ddm_jg1`` S1e correction 3 measured
that a single token flip moves a MEDIAN of 1 argmax pixel with Chebyshev radius
0-11 px, so sites that are far apart can share one render.

This module packs sites into maximal Chebyshev-separated batches and screens all
candidates for a batch in one forward each.  **The packing is CONTROLLED, not
assumed**: the per-site local-window deltas must sum to the whole-frame delta, and
the residual is measured and recorded on every screen.  A nonzero residual means
sites interacted and the screen is contaminated.

Screening is only a RANKER.  The number this module banks is the **joint realized**
delta: every accepted edit for a pair is applied together, rendered once, and
re-segmented once.  If the joint result is worse than the base, the pair keeps
nothing.  Realized acceptance at the site level AND at the pair level.

AXIS
----
Every number is ``[macOS-CPU advisory]``.  ``score_claim=false`` ·
``promotable=false``.  The seg instrument reproduces the T4 seg leg at 0.99995x
(``ddm_jg1`` S1b) and pose is scored DIRECTLY on the DALI GT table -- never scaled
by a lineage factor, because the PyAV-vs-DALI pose gap is **additive**
(C = 1.4061e-04), which is why the per-pair ratio spans 0.887-1,627.

P0 CONTRACTS HONORED
--------------------
* **ALWAYS KEEP THE PAYLOAD.**  Every accepted edit field is persisted as an npz
  under the arm store, with sha256 recorded -- never only its measured length.
* **Resumable from disk.**  Per-pair checkpoints; ``--resume`` skips completed
  pairs and reloads their accepted edits.
* **Seeded random pairs, never a prefix** (``up2.select_pairs``): ``ddm_bp2`` /
  ``ddm_na2`` measured a prefix as a different population, 2.54-4.21x harder on
  pose and 0.95-0.97x easier on seg.
"""

from __future__ import annotations

import argparse
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

# ----------------------------------------------------------------------------
# The scored arithmetic.  Every constant here is derived from the contest scoring
# function, not tuned: S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489.
# ----------------------------------------------------------------------------

N_PAIRS = 600
GRID_H = 384
GRID_W = 512
PLANE = GRID_H * GRID_W
NUM_CLASSES = 5

SCORE_RATE_DENOMINATOR = 37_545_489
S_PER_ARCHIVE_BYTE = 25.0 / SCORE_RATE_DENOMINATOR          # 6.65857e-07
S_PER_SEG_CELL = 100.0 / (N_PAIRS * PLANE)                  # 8.47710e-07
BYTES_PER_SEG_CELL = S_PER_SEG_CELL / S_PER_ARCHIVE_BYTE    # 1.27310
BITS_PER_SEG_CELL = 8.0 * BYTES_PER_SEG_CELL                # 10.18477

#: ``ddm_jg2`` S1f, MEASURED on ``archive.zip`` through a byte-identical encoder.
#: This is a PLANNING PRIOR only -- it was measured at ONE edit density (58 sparse
#: tokens over 3 pairs) and n600 is ~200x denser, which is the
#: cross-regime-constant-transfer genus.  The re-encoder is the rate AUTHORITY.
RATE_PRIOR_BITS_PER_TOKEN = 4.1379

#: The pointer.  Every one of these is read from the T4 receipt the pointer names
#: (``experiments/results/modal_auth_eval_mirror/contest_auth_eval_up3_thirteenth_move_t4_r1_20260819.json``),
#: NOT from a charter or a predecessor memo -- and a test re-derives ``BASE_S`` from
#: the other three to prove they are the components the shipped score was computed
#: from.
#:
#: **CORRECTION this arm owes its predecessors.**  ``ddm_jg1`` and ``ddm_jg2`` both
#: quote ``d_pose = 7.649246787e-06``.  The receipt carries **7.65e-06**, and only
#: the receipt value reconstructs the pointer EXACTLY (bit-identically; the
#: predecessors' value misses by 4.3e-07 in S).  The two agree to 9.8e-05 relative,
#: which is inside the receipt's own 3-significant-figure quote, so this is not a
#: contradiction between measurements -- 7.649246787e-06 is ``ddm_up2``'s
#: higher-precision LOCAL pose instrument and is consistent with it.  But the score
#: that ships was computed from the receipt's number, so pose arithmetic quoted
#: against the pointer must use the receipt's number or the base does not close.
BASE_S = 0.15652626435208142
BASE_D_SEG = 0.00030309
BASE_D_POSE = 7.65e-06
BASE_ARCHIVE_BYTES = 176_420
POINTER_ARCHIVE_SHA = "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"

#: ``ddm_jg1``'s retained n600 SegNet argmax of the SHIPPED decode -- the object
#: every seg delta is measured against.  Reused rather than recomputed (recomputing
#: it costs 600 * 0.85 s and would be a rediscovery, not a measurement).
DEFAULT_BASE_ARGMAX = Path(
    "/Volumes/APDataStore/pact/ddm_jg1/retained/base_argmax_n600.npy"
)
BASE_ARGMAX_SHA = "0be911de99d5baefa469c2e55f7cdeafde67b353ed8a039962e41448cf1cd137"

#: ``ddm_hm1``'s per-cell class logits, int16 in units of 1/8 (the runtime's
#: ``HPAC_LOGIT_PRECISION``).  Used ONLY to RANK candidates by cells-per-bit --
#: never to price a result.  ``ddm_jg1`` S1d caveat 3: these are the hm1/182,759 B
#: generation and our body's model is SHARPER, so as an absolute price they are
#: cross-body.  As a RANKER within one pair they are the coder's own shape.
DEFAULT_LOGITS = Path(
    "/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/base_logits_int16_n600.i16"
)
LOGIT_SCALE = 8.0

#: Chebyshev separation for packed screening.  ``ddm_jg1`` S1e correction 3
#: measured a single flip's influence at Chebyshev radius 0-11 px, so 32 gives a
#: half-window of 16 -- above the measured reach with margin.  The packing CONTROL
#: (local deltas must sum to the whole-frame delta) is what actually proves it.
DEFAULT_SEPARATION = 32
DEFAULT_WINDOW = 15

#: Separation enforced on the ACCEPTED set, which is a different and stricter
#: constraint than the screening separation.  ``ddm_jg1`` S1e correction 3 measured
#: joint additivity holding at **>= 64 px** (ratios 1.000 / 0.818 / 0.750); below
#: that, co-accepted edits merge into a block move and S1c measured block moves as
#: strongly counterproductive at every radius.
DEFAULT_ACCEPT_SEPARATION = 64

#: The separations the solver SWEEPS per pair, each joint-rendered and re-segmented,
#: with the winner chosen on the score's own objective.  jg1's 64 is the top rung
#: rather than the answer: it was measured on an n=3 packing probe and this arm
#: measured it leaving repair on the table.
DEFAULT_ACCEPT_LADDER = (64, 48, 32, 24, 16, 8)

#: The fractions of the value-ranked accepted set the solver also tries at each
#: separation.  Reaching further down the ranking buys more repaired cells and
#: costs more tokens; which way that trades is a per-pair question, so it is
#: measured per pair rather than fixed.
DEFAULT_KEEP_FRACTIONS = (1.0, 0.75, 0.5, 0.25)

#: The candidate offsets: the failing cell itself and its four 4-neighbours.
#: ``ddm_jg1`` S1c measured the winning move is "almost always a single ADJACENT
#: cell", and S1e correction 2 measured that **0 of 12** accepted edits chose the
#: GT class -- the winning edits are ADVERSARIAL, so all five classes are tried.
CANDIDATE_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))


class Jg3Error(RuntimeError):
    """Fail closed.  A wrong seg or rate number here becomes a wrong seal."""


# ----------------------------------------------------------------------------
# Custody helpers
# ----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def progress(record: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(record, sort_keys=True) + "\n")
    sys.stdout.flush()


# ----------------------------------------------------------------------------
# The candidate model
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One single-cell token coordinate move: write ``value`` at ``(y, x)``."""

    y: int
    x: int
    value: int


def candidates_for_site(
    tokens_pair: np.ndarray, y: int, x: int
) -> list[Candidate]:
    """Every single-cell move available at one failing cell.

    The alphabet forces the mechanism (``ddm_jg1`` S1b): a token is one of five
    CLASS LABELS, there is no "more strongly Road" symbol, and at 95.9% of failing
    cells the stored token is ALREADY the right class.  So the only lever is
    SPATIAL -- re-label the cell or one of its neighbours to move the painted
    boundary.  A candidate whose value already equals the stored token is a no-op
    and is dropped.
    """
    out: list[Candidate] = []
    for dy, dx in CANDIDATE_OFFSETS:
        yy, xx = y + dy, x + dx
        if not (0 <= yy < GRID_H and 0 <= xx < GRID_W):
            continue
        current = int(tokens_pair[yy, xx])
        for value in range(NUM_CLASSES):
            if value == current:
                continue
            out.append(Candidate(yy, xx, value))
    return out


def independent_batches(
    sites: np.ndarray, separation: int
) -> list[np.ndarray]:
    """Partition sites into batches whose members are pairwise >= ``separation`` apart.

    Greedy, deterministic, and order-stable: sites arrive in row-major order so the
    partition is a pure function of the site set.  Chebyshev distance, because the
    influence region a flip perturbs is a square window.
    """
    remaining = list(range(len(sites)))
    batches: list[np.ndarray] = []
    while remaining:
        chosen: list[int] = []
        chosen_yx: list[tuple[int, int]] = []
        leftover: list[int] = []
        for index in remaining:
            y, x = int(sites[index, 0]), int(sites[index, 1])
            if all(
                max(abs(y - cy), abs(x - cx)) >= separation for cy, cx in chosen_yx
            ):
                chosen.append(index)
                chosen_yx.append((y, x))
            else:
                leftover.append(index)
        if not chosen:  # pragma: no cover - separation <= 0 would be a caller bug
            raise Jg3Error("independent set is empty; separation is unusable")
        batches.append(np.array(chosen, dtype=np.int64))
        remaining = leftover
    return batches


def select_separated(
    scored: dict[tuple[int, int], tuple[float, int, float, int]],
    accept_separation: int,
) -> list[tuple[tuple[int, int], tuple[float, int, float, int]]]:
    """Greedily keep the highest-value moves that stay ``accept_separation`` apart.

    **This constraint is the difference between a working solver and a broken one,
    and it was found by a control rather than by reasoning.**  ``ddm_jg1`` S1e
    correction 3 measured the joint-additivity law: improving edits reproduce the
    sum of their solo deltas only at **>= 64 px separation** (ratios 1.000 / 0.818 /
    0.750).  Screening guarantees separation WITHIN a batch, but the accepted set is
    pooled ACROSS batches, where two winners can land adjacent -- and a cluster of
    adjacent edits IS a block move, which S1c measured as strongly counterproductive
    (-55% at r=1, -351% at r=2).

    The first run of this solver omitted the constraint and pair 283 changed **66
    tokens to repair 3 cells** (yield 0.0455) where jg1 repaired 25 from 20 (yield
    1.25).  The screen was right about every individual move and the joint result
    was still nearly worthless.

    Moves are keyed by ``(y, x)`` and valued by ``payload[0]`` = the move's NET S
    GAIN, so the strongest move in each neighbourhood is the one that survives.
    Ties break on the coordinate key, which makes the selection a pure function of
    the input -- sister of the ``ddm_cw1`` container finding, where two configs tied
    at 176,420 B and only the lower-index one reproduced the shipped bytes.
    """
    ranked = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))
    selected: list[tuple[tuple[int, int], tuple[float, int, float, int]]] = []
    for key, payload in ranked:
        y, x = key
        if all(
            max(abs(y - cy), abs(x - cx)) >= accept_separation
            for (cy, cx), _ in selected
        ):
            selected.append((key, payload))
    return selected


def window_counts(
    mismatch: np.ndarray, sites: np.ndarray, window: int
) -> np.ndarray:
    """Flip count inside a +/-``window`` box around each site.

    ``mismatch`` is the boolean ``argmax != gt`` plane.  The windows are disjoint
    when the sites are ``>= 2*window + 2`` apart, which is what the packing
    guarantees, so these counts partition the frame's flips among the sites plus a
    remainder -- and that partition is exactly what the packing control checks.
    """
    counts = np.empty(len(sites), dtype=np.int64)
    for row in range(len(sites)):
        y, x = int(sites[row, 0]), int(sites[row, 1])
        y0, y1 = max(0, y - window), min(GRID_H, y + window + 1)
        x0, x1 = max(0, x - window), min(GRID_W, x + window + 1)
        counts[row] = int(mismatch[y0:y1, x0:x1].sum())
    return counts


# ----------------------------------------------------------------------------
# The rate ranker
# ----------------------------------------------------------------------------


class LogitPrice:
    """Per-move bit price under the coder's own class distribution.

    Used to RANK candidates by ``cells_repaired / bits`` rather than by cells
    alone.  ``ddm_jg1`` S1e point 1 measured why that matters: +4.718 bits was the
    MEAN over all four neighbour candidates and a solver pays the CHEAPEST -- the
    distribution is wide (p25 +0.902, p10 -2.344) and 20% of moves are free or
    better.

    **This is a RANKER, not a price.**  The logits are the hm1/182,759 B generation
    and our body's model is sharper (``ddm_jg1`` S1d caveat 3), and a token edit
    invalidates the downstream context anyway (``ddm_jg2`` S1b measured four
    feedback paths making the blast radius global).  The rate AUTHORITY is the
    re-encoder.
    """

    def __init__(self, path: Path | None, pairs: int = N_PAIRS):
        self.path = path
        self.pairs = pairs
        self.available = path is not None and path.exists()
        self._memmap = None
        if self.available:
            # FAIL CLOSED on size.  ``np.memmap`` raises when the file is too
            # SMALL, but silently truncates when it is too LARGE -- so a logits
            # file from a different generation would be read as if it were ours and
            # every candidate rank would be quietly wrong.  ``ddm_jg1`` already lost
            # a pass to exactly this class of error, reading these logits at scale
            # 256 instead of 8 and getting a price 200x off.
            expected = pairs * PLANE * NUM_CLASSES * 2
            actual = path.stat().st_size
            if actual != expected:
                raise Jg3Error(
                    f"logits file {path} is {actual} B, expected {expected} B for "
                    f"({pairs}, {PLANE}, {NUM_CLASSES}) int16; refusing to rank "
                    "candidates against a field of unknown provenance"
                )
            self._memmap = np.memmap(
                path, dtype=np.int16, mode="r", shape=(pairs, PLANE, NUM_CLASSES)
            )

    def bits_for(
        self, pair: int, moves: Sequence[Candidate], tokens_pair: np.ndarray
    ) -> np.ndarray:
        """``log2(p_old / p_new)`` for each move, or the flat prior if unavailable."""
        if not self.available or not moves:
            return np.full(len(moves), RATE_PRIOR_BITS_PER_TOKEN, dtype=np.float64)
        flat = np.array([m.y * GRID_W + m.x for m in moves], dtype=np.int64)
        rows = np.asarray(self._memmap[pair][flat], dtype=np.float64) / LOGIT_SCALE
        rows -= rows.max(axis=1, keepdims=True)
        expo = np.exp(rows)
        probability = expo / expo.sum(axis=1, keepdims=True)
        old = np.array(
            [tokens_pair[m.y, m.x] for m in moves], dtype=np.int64
        )
        new = np.array([m.value for m in moves], dtype=np.int64)
        index = np.arange(len(moves))
        floor = 1e-12
        p_old = np.maximum(probability[index, old], floor)
        p_new = np.maximum(probability[index, new], floor)
        return np.log2(p_old / p_new)


# ----------------------------------------------------------------------------
# The realized seg descent
# ----------------------------------------------------------------------------


@dataclass
class PairResult:
    """One pair's realized outcome.  Every field is measured, none predicted."""

    pair: int
    flips_before: int
    flips_after: int
    tokens_changed: int
    screened_candidates: int
    evaluations: int
    packing_residual_max: int
    rejected_for_separation: int
    seconds: float
    separation_sweep: list[dict[str, Any]] = field(default_factory=list)
    accept_separation_chosen: int = 0
    keep_fraction_chosen: float = 0.0
    accepted: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def repaired(self) -> int:
        return self.flips_before - self.flips_after

    @property
    def cells_per_changed_token(self) -> float:
        return self.repaired / self.tokens_changed if self.tokens_changed else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "pair": self.pair,
            "flips_before": self.flips_before,
            "flips_after": self.flips_after,
            "repaired": self.repaired,
            "tokens_changed": self.tokens_changed,
            "cells_per_changed_token": self.cells_per_changed_token,
            "screened_candidates": self.screened_candidates,
            "evaluations": self.evaluations,
            "packing_residual_max": self.packing_residual_max,
            "rejected_for_separation": self.rejected_for_separation,
            "separation_sweep": self.separation_sweep,
            "accept_separation_chosen": self.accept_separation_chosen,
            "keep_fraction_chosen": self.keep_fraction_chosen,
            "seconds": self.seconds,
            "accepted": self.accepted,
        }


def segnet_margin(net, frame_bhwc) -> np.ndarray:
    """Top1-minus-top2 SegNet logit margin for one camera frame, as ``(384, 512)``.

    **The margin field IS the Fisher surrogate**, and that is measured rather than
    assumed: the campaign measured Fisher curvature against ``(-margin)`` at Pearson
    **0.978**, so a low-margin cell is a cell whose argmax is closest to flipping.
    ``d_seg`` lives on the codim-1 boundary of the frozen argmax partition, where
    the Fisher geometry is anisotropic while the interior is flat -- so ordering
    work by margin is ordering it by where a move can actually pay.

    This costs NOTHING extra: it reads the same forward pass the argmax comes from.
    ``ddm_jg1`` owed item 6 notes that a DALI-lineage margin field cannot be built
    locally, but that is the margin of the GT decode; this is the margin of OUR
    rendered frame, which is a property of our own frames and is fully available.
    """
    import torch

    with torch.inference_mode():
        batch = up2.frames_to_bchw(frame_bhwc)
        logits = net(net.preprocess_input(batch.unsqueeze(1)))
        top2 = torch.topk(logits, k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).to(torch.float32).cpu().numpy()[0]


def rank_sites_by_margin_saliency(
    sites: np.ndarray, margin: np.ndarray, window: int
) -> np.ndarray:
    """Order flip sites by how much repairable margin mass sits around them.

    A ranker ORDERS; it never accepts.  Acceptance stays realized-only, so the worst
    a bad order can do is spend realizations in the wrong place -- it cannot put an
    unmeasured move into the state.  That asymmetry is what makes a geometric prior
    admissible here at all.

    The score is the NEGATIVE mean margin in the site's window: a neighbourhood full
    of cells the scorer is nearly undecided about is a neighbourhood where one token
    can move several argmax cells, which is exactly the ``1.55 cells per changed
    token`` mechanism ``ddm_jg1`` measured.  Ties break on site index so the order is
    a pure function of the inputs.
    """
    if len(sites) == 0:
        return np.zeros(0, dtype=np.int64)
    score = np.empty(len(sites), dtype=np.float64)
    for row in range(len(sites)):
        y, x = int(sites[row, 0]), int(sites[row, 1])
        y0, y1 = max(0, y - window), min(GRID_H, y + window + 1)
        x0, x1 = max(0, x - window), min(GRID_W, x + window + 1)
        score[row] = -float(margin[y0:y1, x0:x1].mean())
    return np.argsort(-score, kind="stable").astype(np.int64)


def _segnet_argmax_batched(net, frames, batch: int) -> np.ndarray:
    """SegNet argmax over a stack of camera frames, in chunks.

    BatchNorm is in eval mode so it reads RUNNING statistics -- batch size does not
    change the result, which is why ``ddm_jg1``'s n600 instrument validation ran at
    batch 4 and still reproduced the published leg at 0.99995x.
    """
    out = []
    for start in range(0, len(frames), batch):
        out.append(jg1.argmax_from_camera_frames(net, frames[start : start + batch]))
    return np.concatenate(out, axis=0)


def solve_pair(
    semantic,
    net,
    tokens_pair: np.ndarray,
    base_argmax_pair: np.ndarray,
    gt_pair: np.ndarray,
    pair: int,
    pricer: LogitPrice,
    *,
    separation: int = DEFAULT_SEPARATION,
    accept_separation: int = DEFAULT_ACCEPT_SEPARATION,
    accept_ladder: Sequence[int] = DEFAULT_ACCEPT_LADDER,
    keep_fractions: Sequence[float] = DEFAULT_KEEP_FRACTIONS,
    window: int = DEFAULT_WINDOW,
    segnet_batch: int = 8,
    max_sites: int = 0,
    max_candidates_per_site: int = 0,
    site_budget: int = 0,
    site_seed: int = 20260819,
) -> tuple[PairResult, np.ndarray]:
    """One pair: screen packed, select by cells-per-bit, then VERIFY jointly.

    Returns the measured result and the edited token plane.  The returned plane is
    the base plane when nothing survived joint verification -- realized acceptance
    applies to the pair as a whole, not only to its sites.
    """
    started = time.time()
    mismatch = base_argmax_pair != gt_pair
    sites = np.argwhere(mismatch)
    flips_before = int(mismatch.sum())
    if max_sites and len(sites) > max_sites:
        # A SEEDED RANDOM subsample, never ``sites[:max_sites]``.  Sites arrive in
        # row-major order, so a head slice is the top of the frame -- a spatial
        # prefix, and ``ddm_bp2``'s law is that a prefix of a skewed population is
        # a different population.  Frame rows are strongly class-skewed here
        # (Undrivable/sky occupies rows ~9-182, MyCar rows ~290-379), so a head
        # slice would sample a different edge mixture than the pair really has.
        picker = np.random.default_rng(site_seed + pair)
        keep = np.sort(picker.choice(len(sites), size=max_sites, replace=False))
        sites = sites[keep]

    # GEOMETRIC SITE ORDER, not row-major enumeration.  Sites arrive from
    # ``np.argwhere`` in row-major order, which is an arbitrary scan of the frame.
    # Ordering them by margin saliency spends realizations where the frozen
    # scorer is closest to flipping -- and with ``site_budget`` set, the sites that
    # get realized at all are the ones the geometry says can pay.
    margin = None
    evaluations_margin = 0
    if len(sites):
        base_frame = jg1.render_frame1(
            semantic, tokens_pair[None], np.array([pair])
        )
        margin = segnet_margin(net, base_frame)
        evaluations_margin = 1
        order = rank_sites_by_margin_saliency(sites, margin, window)
        sites = sites[order]
        if site_budget and len(sites) > site_budget:
            sites = sites[:site_budget]

    evaluations = evaluations_margin if len(sites) else 0
    screened = 0
    residual_max = 0
    # best[(y, x)] -> (net_S_gain, repaired, bits, value)
    best: dict[tuple[int, int], tuple[float, int, float, int]] = {}

    if len(sites):
        for batch_indices in independent_batches(sites, separation):
            batch_sites = sites[batch_indices]
            per_site = [
                candidates_for_site(tokens_pair, int(y), int(x)) for y, x in batch_sites
            ]
            if max_candidates_per_site:
                # Realize only the CHEAPEST candidates per site.  This is a
                # principled pre-filter, not a corner cut: the objective is
                # cells-per-BIT, and ``ddm_jg1`` S1d measured the per-move cost
                # distribution as very wide (p10 -2.344, p25 +0.902, p90 +12.443
                # bits), so the expensive tail can almost never clear the
                # Lagrangian test no matter how many cells it repairs.  Cutting it
                # trades a measured amount of yield for wall clock, and the trade
                # is measured at the n=3 rung rather than assumed.
                trimmed = []
                for options in per_site:
                    if len(options) <= max_candidates_per_site:
                        trimmed.append(options)
                        continue
                    cost = pricer.bits_for(pair, options, tokens_pair)
                    order = np.argsort(cost, kind="stable")[:max_candidates_per_site]
                    trimmed.append([options[int(i)] for i in sorted(order)])
                per_site = trimmed
            depth = max((len(c) for c in per_site), default=0)
            base_counts = window_counts(mismatch, batch_sites, window)
            # Render every candidate slot for this batch FIRST, then re-segment them
            # in ONE batched SegNet pass.  MEASURED here: SegNet costs 0.622 s/frame
            # at batch 1 and 0.206 s/frame at batch 8 -- a 3.0x saving that changes
            # nothing about the science, because BatchNorm is in eval mode and reads
            # RUNNING statistics, so the argmax is batch-size independent (which is
            # why ``ddm_jg1``'s n600 validation ran at batch 4 and still reproduced
            # the published seg leg at 0.99995x).  The RENDER stays at batch 1:
            # ``ddm_up2`` sec.6 measured semantic batch 8 as BYTE-CHANGING (1,326
            # pixels at +/-1 through the clamp/round), so batching that half would
            # break the forward-model identity this whole instrument rests on.
            slot_moves: list[list[Candidate]] = []
            slot_rows: list[list[int]] = []
            frames: list[np.ndarray] = []
            for slot in range(depth):
                moves: list[Candidate] = []
                rows: list[int] = []
                for row, options in enumerate(per_site):
                    if slot < len(options):
                        moves.append(options[slot])
                        rows.append(row)
                if not moves:
                    continue
                proposal = tokens_pair.copy()
                for move in moves:
                    proposal[move.y, move.x] = move.value
                frames.append(
                    jg1.render_frame1(semantic, proposal[None], np.array([pair]))[0]
                )
                slot_moves.append(moves)
                slot_rows.append(rows)
            if not frames:
                continue
            argmaxes = _segnet_argmax_batched(net, np.stack(frames), segnet_batch)
            evaluations += len(frames)
            for order in range(len(frames)):
                moves = slot_moves[order]
                rows = slot_rows[order]
                screened += len(moves)
                new_mismatch = argmaxes[order] != gt_pair
                new_counts = window_counts(new_mismatch, batch_sites, window)
                # PACKING CONTROL: with disjoint windows the per-site deltas must
                # account for the whole-frame delta.  A nonzero residual means the
                # packed sites interacted and this screen row is contaminated.
                whole = int(new_mismatch.sum()) - flips_before
                summed = int((new_counts - base_counts).sum())
                residual_max = max(residual_max, abs(whole - summed))
                bits = pricer.bits_for(pair, moves, tokens_pair)
                for slot_row, row in enumerate(rows):
                    repaired = int(base_counts[row] - new_counts[row])
                    if repaired <= 0:
                        continue
                    cost = float(bits[slot_row])
                    # The Lagrangian admission test, in the score's own units.
                    if repaired * BITS_PER_SEG_CELL <= cost:
                        continue
                    move = moves[slot_row]
                    key = (move.y, move.x)
                    # Rank by the SCORE'S OWN objective, not by a cells-per-bit
                    # ratio.  A ratio mis-ranks exactly the moves worth the most:
                    # ``ddm_jg1`` S1d measured **20% of moves are free or cheaper**
                    # (delta bits <= 0, p10 = -2.344), and ``repaired / max(cost, eps)``
                    # sends every one of those to a near-infinite score regardless of
                    # how few cells it repairs -- so a free 1-cell move would outrank
                    # a 10-cell move that costs 2 bits.  Net S gain is linear in both
                    # terms and signs the credit correctly.
                    gain = (
                        repaired * S_PER_SEG_CELL
                        - (cost / 8.0) * S_PER_ARCHIVE_BYTE
                    )
                    prior = best.get(key)
                    if prior is None or gain > prior[0]:
                        best[key] = (gain, repaired, cost, move.value)

    if not best:
        return (
            PairResult(
                pair=pair,
                flips_before=flips_before,
                flips_after=flips_before,
                tokens_changed=0,
                screened_candidates=screened,
                evaluations=evaluations,
                packing_residual_max=residual_max,
                rejected_for_separation=0,
                seconds=time.time() - started,
            ),
            tokens_pair,
        )

    # THE SEPARATION IS MEASURED, NOT BORROWED.  ``ddm_jg1``'s >= 64 px additivity
    # number came from an n=3 packing probe, and this arm measured that using it as
    # a fixed constant leaves repair on the table: pair 283 gave a healthy yield of
    # 1.50 but repaired only 9 of its 38 flips, where jg1's ad-hoc greedy repaired
    # 25.  Yield alone does not decide the score -- TOTAL repair is the other half.
    #
    # So the accepted set is built at several separations and every one of them is
    # JOINTLY RENDERED AND RE-SEGMENTED.  The winner is chosen on the score's own
    # objective, not on flip count and not on yield:
    #
    #     minimize   -repaired * S_PER_SEG_CELL  +  tokens * bits/8 * S_PER_BYTE
    #
    # A denser set that repairs more cells is admitted only if the extra cells pay
    # for the extra tokens.  This is the cross-regime-constant-transfer genus
    # answered by measurement: jg1's constant was measured on a different object.
    # The configuration space is TWO-dimensional, because there are two independent
    # ways to be too greedy.  SEPARATION controls how densely edits may cluster
    # (the block-move hazard).  KEEP-FRACTION controls how far down the value
    # ranking the solver reaches (the yield-decay hazard ``ddm_jg1`` S1e measured,
    # 1.50 -> 0.390 under iteration).  Sweeping only separation left the realized
    # yield at 0.92 on the jg1 control pairs -- below the ~1.06 the goal needs --
    # because a sparse set can still be padded with weak moves.
    #
    # Every point in the grid is JOINTLY RENDERED AND RE-SEGMENTED, so the winner is
    # measured, not predicted.  The grid is cheap: screening costs ~120 s/pair and
    # each extra configuration costs one render plus one batched SegNet row.
    ladder = [s for s in accept_ladder if s > 0] or [accept_separation]
    fractions = [f for f in keep_fractions if 0.0 < f <= 1.0] or [1.0]
    trials: list[tuple[int, float, list[Any]]] = []
    frames = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for rung in ladder:
        # ``select_separated`` returns in DESCENDING net-gain order, so a head slice
        # is the top-k by value -- not a positional prefix of anything spatial.
        chosen_full = select_separated(best, rung)
        for fraction in fractions:
            keep = max(1, round(len(chosen_full) * fraction))
            chosen = chosen_full[:keep]
            signature = tuple(sorted(key for key, _ in chosen))
            if signature in seen:
                continue  # the same edit set reached two ways costs one render
            seen.add(signature)
            edited_try = tokens_pair.copy()
            for (y, x), payload in chosen:
                edited_try[y, x] = payload[3]
            frames.append(
                jg1.render_frame1(semantic, edited_try[None], np.array([pair]))[0]
            )
            trials.append((rung, fraction, chosen))
    joint_argmaxes = _segnet_argmax_batched(net, np.stack(frames), segnet_batch)
    evaluations += len(frames)

    sweep = []
    best_net = 0.0  # keeping nothing is always available and scores exactly 0
    winner: tuple[int, float, list[Any]] | None = None
    winner_flips = flips_before
    for order, (rung, fraction, chosen) in enumerate(trials):
        after = int((joint_argmaxes[order] != gt_pair).sum())
        repaired_here = flips_before - after
        tokens_here = len(chosen)
        # PRICE THE CONFIGURATION WITH THE CONSTANT MEASURED ON OUR OWN BODY.
        #
        # The logit sum is the WRONG price here and this arm measured the size of
        # the error.  Back-solving the configurations this sweep first chose, the
        # hm1 logits charge **1.91 bits/token** where ``ddm_jg2`` MEASURED
        # **4.1379 bits/token** on ``archive.zip`` through a byte-identical
        # encoder -- a **2.2x under-price**.  An under-priced token biases every
        # rung comparison toward the DENSER configuration, because extra tokens
        # look nearly free; that is how the first sweep picked the densest rung on
        # all three control pairs.
        #
        # The logits stay useful for RANKING candidates within a site (a relative
        # question on one body).  The ABSOLUTE cost of a configuration is a
        # cross-body question, and only jg2's constant was measured on the body we
        # ship.  Both numbers are recorded so the gap stays visible.
        cost_bits_logit = sum(float(p[2]) for _, p in chosen)
        cost_bits = tokens_here * RATE_PRIOR_BITS_PER_TOKEN
        net = (
            -repaired_here * S_PER_SEG_CELL
            + (cost_bits / 8.0) * S_PER_ARCHIVE_BYTE
        )
        sweep.append(
            {
                "accept_separation": rung,
                "keep_fraction": fraction,
                "tokens": tokens_here,
                "flips_after": after,
                "repaired": repaired_here,
                "cost_bits_measured_prior": cost_bits,
                "cost_bits_logit_crossbody": cost_bits_logit,
                "net_delta_S": net,
            }
        )
        if net < best_net:
            best_net, winner, winner_flips = net, (rung, fraction, chosen), after

    if winner is None:
        return (
            PairResult(
                pair=pair,
                flips_before=flips_before,
                flips_after=flips_before,
                tokens_changed=0,
                screened_candidates=screened,
                evaluations=evaluations,
                packing_residual_max=residual_max,
                rejected_for_separation=0,
                seconds=time.time() - started,
                separation_sweep=sweep,
            ),
            tokens_pair,
        )

    accept_separation, keep_fraction_chosen, selected = winner
    rejected_for_separation = len(best) - len(selected)
    edited = tokens_pair.copy()
    for (y, x), payload in selected:
        edited[y, x] = payload[3]
    flips_after = winner_flips

    if flips_after >= flips_before:
        # Realized acceptance at the PAIR level: a screen that ranked well but does
        # not survive joint application keeps nothing.  Screening is a ranker.
        return (
            PairResult(
                pair=pair,
                flips_before=flips_before,
                flips_after=flips_before,
                tokens_changed=0,
                screened_candidates=screened,
                evaluations=evaluations,
                packing_residual_max=residual_max,
                rejected_for_separation=0,
                seconds=time.time() - started,
            ),
            tokens_pair,
        )

    accepted = sorted(
        (int(y), int(x), int(payload[3])) for (y, x), payload in selected
    )
    return (
        PairResult(
            pair=pair,
            flips_before=flips_before,
            flips_after=flips_after,
            tokens_changed=len(accepted),
            screened_candidates=screened,
            evaluations=evaluations,
            packing_residual_max=residual_max,
            rejected_for_separation=rejected_for_separation,
            seconds=time.time() - started,
            separation_sweep=sweep,
            accept_separation_chosen=accept_separation,
            keep_fraction_chosen=keep_fraction_chosen,
            accepted=accepted,
        ),
        edited,
    )


# ----------------------------------------------------------------------------
# Scored arithmetic on a measured run
# ----------------------------------------------------------------------------


def project(
    repaired: int,
    tokens: int,
    pairs_solved: int,
    *,
    bits_per_token: float = RATE_PRIOR_BITS_PER_TOKEN,
    pose_ratio: float = 1.073,
    measured_archive_delta_bytes: int | None = None,
) -> dict[str, Any]:
    """The score arithmetic, scaled from ``pairs_solved`` to the full 600.

    The rate leg uses the MEASURED archive delta when one is supplied and the jg2
    prior otherwise -- and says which, because a modelled rate leg and a measured
    one are different claims.  Rate may be scaled linearly because ``ddm_jg2`` S1h
    MEASURED that token-edit costs superpose (union/sum 1.0258, exact at the
    archive layer).  Seg may NOT be scaled across passes for the same reason
    reversed: S1e measured the yield decaying 1.50 -> 0.390 under iteration.  This
    is a FIRST PASS, so the per-pair mean is the honest scaling unit.
    """
    if pairs_solved <= 0:
        raise Jg3Error("cannot project from zero solved pairs")
    scale = N_PAIRS / pairs_solved
    full_repaired = repaired * scale
    full_tokens = tokens * scale

    if measured_archive_delta_bytes is not None:
        rate_bytes = measured_archive_delta_bytes * scale
        rate_source = "measured_reencoder_scaled_by_superposition"
    else:
        rate_bytes = full_tokens * bits_per_token / 8.0
        rate_source = "jg2_prior_4.1379_bits_per_token"

    d_seg_after = BASE_D_SEG - full_repaired / (N_PAIRS * PLANE)
    d_pose_after = BASE_D_POSE * pose_ratio
    archive_after = BASE_ARCHIVE_BYTES + rate_bytes

    seg_delta = -full_repaired * S_PER_SEG_CELL
    pose_delta = (10.0 * d_pose_after) ** 0.5 - (10.0 * BASE_D_POSE) ** 0.5
    rate_delta = rate_bytes * S_PER_ARCHIVE_BYTE
    net = seg_delta + pose_delta + rate_delta

    return {
        "pairs_solved": pairs_solved,
        "scale_to_n600": scale,
        "repaired_measured": repaired,
        "tokens_measured": tokens,
        "cells_per_changed_token": repaired / tokens if tokens else 0.0,
        "repaired_projected_n600": full_repaired,
        "tokens_projected_n600": full_tokens,
        "rate_bytes_projected": rate_bytes,
        "rate_source": rate_source,
        "seg_delta_S": seg_delta,
        "rate_delta_S": rate_delta,
        "pose_delta_S": pose_delta,
        "net_delta_S": net,
        "projected_S": BASE_S + net,
        "clears_sub_015": bool(BASE_S + net < 0.15),
        "d_seg_after": d_seg_after,
        "d_pose_after": d_pose_after,
        "archive_bytes_after": archive_after,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
    }


def break_even_yield() -> float:
    """The yield below which NO amount of repair can help, in cells per token.

    Setting the seg gain equal to the rate cost: ``S_PER_SEG_CELL * y =
    (bits/8) * S_PER_ARCHIVE_BYTE`` gives ``y = bits / BITS_PER_SEG_CELL``.  At
    jg2's measured 4.1379 bits/token this is **0.4063** -- and ``ddm_jg1``'s
    8-pass iterated yield of 0.390 is BELOW it.  That is the whole reason the
    stopping rule exists.
    """
    return RATE_PRIOR_BITS_PER_TOKEN / BITS_PER_SEG_CELL


# ----------------------------------------------------------------------------
# Resumable driver
# ----------------------------------------------------------------------------


def _load_base_argmax(path: Path, verify: bool) -> np.ndarray:
    if not path.exists():
        raise Jg3Error(
            f"base argmax not found at {path}; it is ddm_jg1 retained custody"
        )
    if verify:
        got = sha256_file(path)
        if got != BASE_ARGMAX_SHA:
            raise Jg3Error(
                f"base argmax sha {got} != retained {BASE_ARGMAX_SHA}; refusing"
            )
    return np.load(path, mmap_mode="r")


def cmd_solve(args) -> int:
    store = Path(args.store)
    (store / "retained").mkdir(parents=True, exist_ok=True)
    (store / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint = store / "checkpoints" / f"seg_solve_{args.tag}.jsonl"
    edits_path = store / "retained" / f"seg_edits_{args.tag}.npz"

    lineage = up2.verify_gt_lineage(
        axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI
    )
    progress({"stage": "lineage", **lineage})

    tokens = jg1.load_tokens(Path(args.tokens))
    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    base_argmax = _load_base_argmax(Path(args.base_argmax), not args.no_verify_sha)
    pricer = LogitPrice(Path(args.logits) if args.logits else None)
    progress(
        {
            "stage": "loaded",
            "tokens_sha_checked": True,
            "logits_available": pricer.available,
            "logits_note": (
                "RANKER ONLY -- cross-body (hm1/182,759 B); the re-encoder is the "
                "rate authority"
            ),
        }
    )

    if args.pair_list:
        # An EXPLICIT pair list exists for exactly one purpose: reproducing a prior
        # arm's pairs so the two solvers can be compared on the same object.  It is
        # NOT a sampling mode -- any pair list that is not a deliberate reproduction
        # re-introduces the selection bias ``up2.select_pairs`` exists to refuse.
        indices = np.array(
            [int(x) for x in args.pair_list.split(",") if x.strip()], dtype=np.int64
        )
        progress(
            {
                "stage": "pair_list",
                "pairs": indices.tolist(),
                "sampling": "EXPLICIT_reproduction_control_not_a_sample",
            }
        )
    else:
        indices = up2.select_pairs(args.pairs, args.seed)
        if args.shuffle_order:
            # ``up2.select_pairs(600)`` returns ``arange(600)`` -- SORTED.  Visiting
            # in that order would make every partial result a CONTIGUOUS PAIR
            # PREFIX, which is the exact shape ``ddm_bp2``/``ddm_na2`` measured as a
            # different population (pose prefixes 2.54-4.21x HARDER, seg prefixes
            # 0.95-0.97x easier).  That matters twice over here: the charter asks
            # for realized-vs-projected at n = 3/12/48/150/600, and a long run can
            # be interrupted.  With a seeded permutation, EVERY prefix of the run is
            # an unbiased random sample of the field -- so the rungs are legitimate
            # and an interrupted run still yields an honest estimate instead of a
            # biased one.
            indices = np.random.default_rng(args.seed).permutation(indices)
            progress(
                {
                    "stage": "visit_order",
                    "shuffled": True,
                    "seed": args.seed,
                    "why": (
                        "every prefix of the run must be a random sample, never a "
                        "contiguous pair prefix (ddm_bp2/ddm_na2)"
                    ),
                    "first_12": indices[:12].tolist(),
                }
            )
    done: dict[int, dict] = {}
    edits: dict[str, np.ndarray] = {}
    if args.resume and checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["pair"])] = row
        if edits_path.exists():
            with np.load(edits_path) as handle:
                edits = {key: handle[key] for key in handle.files}
        progress({"stage": "resume", "pairs_done": len(done), "edits": len(edits)})

    semantic = jg1.load_semantic_renderer()
    net = jg1.load_segnet()

    results: list[PairResult] = []
    computed = 0
    started = time.time()
    for position, pair in enumerate(indices):
        pair = int(pair)
        if pair in done:
            row = done[pair]
            results.append(
                PairResult(
                    pair=pair,
                    flips_before=row["flips_before"],
                    flips_after=row["flips_after"],
                    tokens_changed=row["tokens_changed"],
                    screened_candidates=row["screened_candidates"],
                    evaluations=row["evaluations"],
                    packing_residual_max=row["packing_residual_max"],
                    rejected_for_separation=row.get("rejected_for_separation", 0),
                    seconds=row["seconds"],
                    separation_sweep=row.get("separation_sweep", []),
                    accept_separation_chosen=row.get("accept_separation_chosen", 0),
                    keep_fraction_chosen=row.get("keep_fraction_chosen", 0.0),
                    accepted=[tuple(a) for a in row["accepted"]],
                )
            )
            continue
        result, edited = solve_pair(
            semantic,
            net,
            np.asarray(tokens[pair]),
            np.asarray(base_argmax[pair]),
            np.asarray(gt[pair]),
            pair,
            pricer,
            separation=args.separation,
            accept_separation=args.accept_separation,
            accept_ladder=[
                int(x) for x in str(args.accept_ladder).split(",") if x.strip()
            ],
            keep_fractions=[
                float(x) for x in str(args.keep_fractions).split(",") if x.strip()
            ],
            window=args.window,
            segnet_batch=args.segnet_batch,
            max_sites=args.max_sites,
            max_candidates_per_site=args.max_candidates_per_site,
            site_budget=args.site_budget,
            site_seed=args.seed,
        )
        results.append(result)
        computed += 1
        if result.tokens_changed:
            edits[str(pair)] = edited.astype(np.uint8)
        # ALWAYS KEEP THE PAYLOAD.  The JSONL line below carries the payload
        # LOSSLESSLY: ``accepted`` is the complete sparse edit list ``(y, x, value)``
        # and the base token field is fixed and sha-pinned, so the edited plane is
        # exactly reconstructible from it.  The npz is a convenience mirror, written
        # every ``--payload-every`` pairs because re-compressing the whole dict on
        # every pair is O(n^2) and would dominate an n600 run.
        with open(checkpoint, "a") as handle:
            handle.write(json.dumps(result.to_json(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if edits and computed % max(args.payload_every, 1) == 0:
            np.savez_compressed(edits_path, **edits)
        elapsed = time.time() - started
        solved = position + 1
        progress(
            {
                "stage": "pair",
                "position": solved,
                "of": len(indices),
                "pair": pair,
                "repaired": result.repaired,
                "tokens": result.tokens_changed,
                "yield": round(result.cells_per_changed_token, 4),
                "residual_max": result.packing_residual_max,
                "seconds": round(result.seconds, 2),
                "eta_hours": round(
                    (elapsed / max(computed, 1))
                    * max(len(indices) - solved, 0)
                    / 3600.0,
                    2,
                ),
            }
        )

    if edits:
        np.savez_compressed(edits_path, **edits)
    repaired = sum(r.repaired for r in results)
    tokens_changed = sum(r.tokens_changed for r in results)
    summary = {
        "arm": "ddm_jg3",
        "tag": args.tag,
        "pairs": len(results),
        "seed": args.seed,
        "sampling": (
            "explicit_reproduction_control"
            if args.pair_list
            else ("seeded_permutation_every_prefix_is_a_sample" if args.shuffle_order else "sorted_ORDER_PREFIX_BIASED")
        ),
        "lineage": lineage,
        "flips_before": sum(r.flips_before for r in results),
        "flips_after": sum(r.flips_after for r in results),
        "repaired": repaired,
        "tokens_changed": tokens_changed,
        "cells_per_changed_token": repaired / tokens_changed if tokens_changed else 0.0,
        "break_even_yield": break_even_yield(),
        "packing_residual_max": max((r.packing_residual_max for r in results), default=0),
        "evaluations": sum(r.evaluations for r in results),
        "wall_seconds": time.time() - started,
        "pairs_with_zero_accept": sum(1 for r in results if r.tokens_changed == 0),
        "per_pair": [r.to_json() for r in results],
        "projection": project(repaired, tokens_changed, len(results))
        if tokens_changed
        else None,
        "edits_payload": {
            "path": str(edits_path),
            "sha256": sha256_file(edits_path) if edits_path.exists() else None,
            "bytes": edits_path.stat().st_size if edits_path.exists() else 0,
            "pairs_edited": len(edits),
        },
    }
    out = store / "retained" / f"seg_solve_{args.tag}.json"
    atomic_json(out, summary)
    progress({"stage": "done", "out": str(out), **{
        k: summary[k] for k in ("repaired", "tokens_changed", "cells_per_changed_token")
    }})
    if summary["projection"]:
        progress({"stage": "projection", **summary["projection"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    solve = sub.add_parser("solve", help="rate-aware realized seg descent")
    solve.add_argument("--store", required=True)
    solve.add_argument("--tag", default="r1")
    solve.add_argument("--pairs", type=int, default=3)
    solve.add_argument("--seed", type=int, default=20260819)
    solve.add_argument("--tokens", default=str(jg1.DEFAULT_TOKENS))
    solve.add_argument("--base-argmax", default=str(DEFAULT_BASE_ARGMAX))
    solve.add_argument("--logits", default=str(DEFAULT_LOGITS))
    solve.add_argument("--separation", type=int, default=DEFAULT_SEPARATION)
    solve.add_argument(
        "--accept-separation", type=int, default=DEFAULT_ACCEPT_SEPARATION
    )
    solve.add_argument(
        "--accept-ladder",
        default=",".join(str(x) for x in DEFAULT_ACCEPT_LADDER),
    )
    solve.add_argument(
        "--keep-fractions",
        default=",".join(str(x) for x in DEFAULT_KEEP_FRACTIONS),
    )
    solve.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    solve.add_argument("--segnet-batch", type=int, default=8)
    solve.add_argument("--max-sites", type=int, default=0)
    solve.add_argument("--max-candidates-per-site", type=int, default=0)
    solve.add_argument("--site-budget", type=int, default=0)
    solve.add_argument("--payload-every", type=int, default=25)
    solve.add_argument("--pair-list", default=None)
    solve.add_argument(
        "--shuffle-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="visit pairs in a seeded permutation so every prefix is a random sample",
    )
    solve.add_argument("--resume", action="store_true")
    solve.add_argument("--no-verify-sha", action="store_true")
    solve.set_defaults(func=cmd_solve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
