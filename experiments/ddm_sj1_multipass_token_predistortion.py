#!/usr/bin/env python3
"""ddm_sj1 -- multi-pass token PRE-DISTORTION to convergence on the cl2 frontier body.

WHAT THIS IS
------------
``ddm_jg1`` measured the governing fact: the stored token field is 99.9985% identical
to the DALI GT argmax, so **95.9% of the seg debt is render -> re-segment loss**, not
stored-label error.  The actuator is therefore PRE-DISTORTION -- deliberately writing a
"wrong" label so that the receiver's own render, pushed back through the frozen SegNet,
lands on the RIGHT argmax.  ``ddm_jg3``/``ddm_jg5`` ran exactly ONE such pass and it
produced the sub-0.15 crossing.  This module runs passes 2..k, with a richer proposal
family, to convergence.

THE ONE THING THAT MAKES THIS AFFORDABLE
----------------------------------------
A naive realized loop costs one render + one SegNet forward PER PROPOSAL.  The frontier
body carries ~23.8k flipped cells over 600 pairs (~40/pair); the family in the charter
is 9 positions x 4 alternative classes = 36 moves per flipped cell, so a naive loop is
~855k renders -- hundreds of CPU-hours.

The escape is not a surrogate.  It is that a single-token change has BOUNDED SPATIAL
INFLUENCE, so many *well-separated* single-token moves can be realized in ONE render +
ONE SegNet forward and attributed independently.  The renderer's own receptive field is
DERIVED from the receiver source (``cpr1/inflate.py``):

    coord_mix 1x1 (r=0) -> TokenBlock depthwise 3x3 at dilations 1,1,2,4 (r=1+1+2+4=8)
    -> head 3x3 (r=1)                                        => r_render = 9 token cells

SegNet's own receptive field is large, so the *total* influence radius is MEASURED, not
assumed (``probe``), and -- decisively -- every accepted set is CLOSED by a composite
re-render + re-segment whose flip count is the realized truth.  Local attribution only
ORDERS the search; the composite verify is the acceptance.

AUTHORITY
---------
``cpu_torch`` SegNet argmax on the DALI GT lineage is the verdict authority
(``tac.ane_screening.AUTHORITY_BACKEND``).  The lineage gate is ``up2.verify_gt_lineage``
and it fails closed: ``ddm_up3`` measured the two GT lineages 1.43x apart on the seg
axis of this very body, and MAIN's pin records 20,671 argmax sites differing between
the DALI table and the PyAV ``gt_n600.npz`` -- 87% of the whole 23,757-flip budget.
Aiming at the wrong table aims at sites the contest does not score.

Everything here is ``[macOS-CPU advisory]``.  ``score_claim=false`` until a T4 row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
NUM_CLASSES = jg1.NUM_CLASSES

# --------------------------------------------------------------------------------------
# The cl2 frontier body.  Every path is a receipt from cl2's own SEAL / VERIFY / PARSEBACK.
# --------------------------------------------------------------------------------------

CL2_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_cl2_hpac_prior_capacity_ladder")
#: cl2 lambda_1p0 receiver copy -- the tree whose ``archive.zip`` carries the semantic
#: weights this arm renders with (SEAL runtime sha ce20617dc7db4ff6...).
BODY_TREE = CL2_ROOT / "rungs/lambda_1p0/retained/receiver_copy_runtime"
#: ``jg1.load_semantic_renderer`` takes the ``runtime/`` PACKAGE dir and derives the tree
#: root and ``cpr1/`` from its parent, exactly as the shipped ``inflate.py`` does.
BODY_RUNTIME = BODY_TREE / "runtime"
BODY_ARCHIVE = BODY_TREE / "archive.zip"
BODY_ARCHIVE_SHA256 = "08ec85333d13d71344b4482cf261e3b2d508725e49f3ca05971265a81498ad4e"
BODY_ARCHIVE_BYTES = 179_982
#: The receiver's OWN decode of that archive (PARSEBACK_RESULT.json, device=cpu).
BODY_RAW = CL2_ROOT / "parseback/lambda_1p0/0.raw"
BODY_RAW_SHA256 = "f86bfaf39f83bcccb1df14ed3cf982767dc94d3a91cd956f00be923612fec4e0"
#: The token field the receiver decoded.  Byte-identical across cl2's five copies
#: (inputs/, control/retained/, rungs/lambda_1p0/retained/, both parsebacks) -- verified
#: sha cc10a7b09353c0af..., so the field is held between the shipped body and cl2's repack.
BODY_TOKENS = CL2_ROOT / "rungs/lambda_1p0/retained/decoded_tokens.u8"

#: The base contest-CUDA T4 row this arm's candidate must beat (fs2 receipt, carried
#: forward by cl2's SEAL).  ``score_claim=false`` for anything measured here.
BASE_ARCHIVE_SHA256 = "a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6"
BASE_ARCHIVE_BYTES = 180_023
BASE_SCORE_T4 = 0.14784474152757654
BASE_D_SEG_T4 = 0.00020139
BASE_D_POSE_T4 = 6.14e-06
#: cl2's sealed rate-only candidate on the same field.
CL2_ARCHIVE_BYTES = 179_982
CL2_PROJECTED_SCORE = 0.14781744131049854

# --------------------------------------------------------------------------------------
# THE LIVE POINTER (MAIN, 2026-09-05): ddm_rc1's lossless recode of the two RX1 MODEL
# sections.  This arm BUILDS and SEALS on this tree and admits against this score.
#
# Why the token-field work composes onto it additively, VERIFIED not assumed:
#   * carrier section byte-identical to cl2 (22,031 B both)
#   * tail byte-identical to cl2 (113,515 B = 96 B residual table + 113,419 B token
#     stream), so the token CODER and the field it codes are untouched
#   * hpac 13,466 -> 12,343 B and semantic 30,856 -> 30,246 B are a LOSSLESS recode --
#     rc1's receiver restores each section body byte-for-byte BEFORE any parsing (rc1
#     SEAL falsifier 1, proven three ways), so the renderer weights this arm renders
#     with are the ones rc1 ships, and its distortion legs are zero BY CONSTRUCTION
#   * RX1 reserved byte 0x1A -> 0x7A (the 14 B header difference)
#   * MEASURED here: 100*0.00020139 + sqrt(10*6.14e-06) + 25*178249/37545489 reproduces
#     MAIN's quoted score to the last digit, and up3.parse_shipped_body + build_archive
#     rebuild this body from its OWN carrier codes byte-identically (sha 1438049e...,
#     178,249 B) -- so the carrier splice survives the recode.
#
# CONSEQUENCE for pricing: the edited field is re-encoded through CL2's tree (whose
# hpac section carries the model in the coding jg2's encoder mirrors), and the resulting
# token stream is spliced into RC1's member.  Encoding against rc1's tree would ask
# jg2's encoder to materialise a model out of a section coded by a codec it has never
# seen; the byte-identical tail is the proof that the two trees hold the SAME model, so
# the stream cl2's path emits is the stream rc1's receiver decodes.
@dataclass(frozen=True)
class PointerRow:
    """One banked contest-CUDA row, with the arithmetic that makes it checkable.

    Three pointer moves landed on this arm inside one session (rc1, then pc1 V3, then
    pc1 x8) with a fourth signalled.  Five scattered constants re-edited by hand each
    time is precisely the shape that goes half-applied and is never re-derived
    ([[binding-instruction-numbers-expire-and-nobody-rederives-them]]), so the pointer
    is a ROW with a self-check instead: a mistyped byte count or d_pose cannot survive
    ``verify_arithmetic``, which is run on every row at import.
    """

    label: str
    tree: Path
    archive_sha256: str
    archive_bytes: int
    d_seg_t4: float
    d_pose_t4: float
    score_t4: float

    def recomputed_score(self) -> float:
        """S from its three legs, exactly as ``upstream/evaluate.py`` composes them."""
        import math

        return (
            100.0 * self.d_seg_t4
            + math.sqrt(10.0 * self.d_pose_t4)
            + 25.0 * self.archive_bytes / jg1.SCORE_RATE_DENOMINATOR
        )

    def verify_arithmetic(self) -> None:
        if abs(self.recomputed_score() - self.score_t4) > 1e-15:
            raise Sj1Error(
                f"pointer row {self.label!r} does not recompute: legs give "
                f"{self.recomputed_score()!r}, row declares {self.score_t4!r}"
            )

    @property
    def archive(self) -> Path:
        return self.tree / "archive.zip"


_PC1_RETAINED = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency/retained"
)

#: The banked lineage, oldest first.  Every row after cl2 holds the token tail and both
#: MODEL sections byte-identical, which is why the seg measurements in this memo carry
#: across all of them and pass 2a never needed a restart.  The moves are: rc1 recodes the
#: two MODEL sections losslessly; pc1 V3 and pc1 x8 coarsen the CARRIER coefficient
#: lattice (x4 then x8) and RE-SOLVE all 600 pairs, which is why d_pose falls rather than
#: merely holding.
POINTER_LINEAGE: tuple[PointerRow, ...] = (
    PointerRow(
        label="fs2_base",
        tree=Path("/Volumes/APDataStore/pact/ddm_fs2"),
        archive_sha256=BASE_ARCHIVE_SHA256,
        archive_bytes=BASE_ARCHIVE_BYTES,
        d_seg_t4=BASE_D_SEG_T4,
        d_pose_t4=BASE_D_POSE_T4,
        score_t4=BASE_SCORE_T4,
    ),
    PointerRow(
        label="cl2_lambda1_repack",
        tree=BODY_TREE,
        archive_sha256=BODY_ARCHIVE_SHA256,
        archive_bytes=CL2_ARCHIVE_BYTES,
        d_seg_t4=BASE_D_SEG_T4,
        d_pose_t4=BASE_D_POSE_T4,
        score_t4=CL2_PROJECTED_SCORE,
    ),
    PointerRow(
        label="rc1_model_section_recode",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_rc1_model_section_adaptive_recode"
            "/staged_runtime"
        ),
        archive_sha256=(
            "1438049e3655fbcfa8eb289fa51ac58f834d72d8a09586353663cea68e57c122"
        ),
        archive_bytes=178_249,
        d_seg_t4=BASE_D_SEG_T4,
        d_pose_t4=BASE_D_POSE_T4,
        score_t4=0.14666350774473783,
    ),
    PointerRow(
        label="pc1_v3_lattice_x4",
        tree=_PC1_RETAINED / "v3_on_rc1_candidate_runtime",
        archive_sha256=(
            "891add546f5cf0943929b566f29dd4318f1d8b2ab76ae05183d8189098880f40"
        ),
        archive_bytes=176_448,
        d_seg_t4=BASE_D_SEG_T4,
        d_pose_t4=5.73e-06,
        score_t4=0.1451981569076111,
    ),
    #: MAIN object change #3.  MEASURED against the x4 rung before
    #: adoption: hpac/semantic/tail byte-identical, carrier 20,230 -> 19,358 B (-872),
    #: codes absmax 275 / absmean 64.9 against cl2's 2048 / 515.0, coefficient_scales
    #: exactly x8 cl2's on all 12 coordinates, every code still inside signed int12, and
    #: up3.build_archive rebuilds the body from its OWN codes byte-identically.
    PointerRow(
        label="pc1_v3x8_lattice_x8",
        tree=_PC1_RETAINED / "v3x8_on_rc1_candidate_runtime",
        archive_sha256=(
            "f7e0bb793645894b2f6885fca82b98cab3067837bd66181e222f3d4b1f43e1ff"
        ),
        archive_bytes=175_576,
        d_seg_t4=BASE_D_SEG_T4,
        d_pose_t4=5.58e-06,
        score_t4=0.1445177913121716,
    ),
    #: MAIN object change #4.  The first rung where the POSE LEG
    #: ROSE -- d_pose 5.58e-06 -> 5.77e-06 -- admitted on the exchange, not on the leg:
    #: pose +1.2611e-04 S against rate -5.2603e-04 S is net -3.9992e-04 S, twenty times
    #: the 2e-05 bar.  MEASURED before adoption: hpac/semantic/tail byte-identical to
    #: the x8 rung, carrier 19,358 -> 18,568 B (-790), codes absmax 142 / absmean 32.6
    #: (half the x8 rung's, as an x2 coarsening should give), coefficient_scales exactly
    #: x16 cl2's on all 12 coordinates, codes inside signed int12, and up3.build_archive
    #: rebuilds the body from its OWN codes to 1de6c5d7186a0b31... at exactly 174,786 B.
    PointerRow(
        label="pc1_v3x16_lattice_x16",
        tree=_PC1_RETAINED / "v3x16_on_rc1_candidate_runtime",
        archive_sha256=(
            "1de6c5d7186a0b31e5cc085bb6d2baab8275ee0d9de4d509f4d8add13695a629"
        ),
        archive_bytes=174_786,
        d_seg_t4=BASE_D_SEG_T4,
        d_pose_t4=5.77e-06,
        score_t4=0.14411787458634504,
    ),
    #: LIVE (MAIN 2026-09-06).  THIS ARM'S OWN candidate, fired on T4 and PROMOTED:
    #: call fc-01M1T6TCW2JS1JEW5CSZH3FVBY, lane ddm_sj1_t4_token_predistortion_joint_20260906.
    #: The first row in this lineage whose SEG leg moved -- every predecessor held d_seg
    #: at 0.00020139 and bought bytes.  Prediction vs measurement, carried openly:
    #:   projected 0.1398087424644421 -> MEASURED 0.1398140172839628, residual +5.2748e-06
    #:   (+0.0038%), decomposing as +3.955e-06 S of seg (parse-back read 14,157 flipped
    #:   cells, T4 read 14,166 -- NINE cells over 117,964,800) and +1.32e-06 S of pose
    #:   (5.398060e-06 measured here vs the row's 5.4e-06).  Both printed legs reproduce
    #:   the reported score to 1e-16, so this row needs no rounding allowance.
    PointerRow(
        label="sj1_token_predistortion_joint",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
            "/candidate/candidate_runtime"
        ),
        archive_sha256=(
            "42aa84b59f71d83b8f11a26c635a7af8f32dcfdf183e3fea4bb2007e74a5f2f8"
        ),
        archive_bytes=180_904,
        d_seg_t4=0.00012009,
        d_pose_t4=5.4e-06,
        score_t4=0.1398140172839628,
    ),
    #: LIVE (MAIN 2026-09-06).  Pass 3 of this arm -- a SECOND single-cell pre-distortion
    #: pass on the residual the pass-2a row left, restricted to the 370-pair admitted
    #: subset, plus a carrier re-solve.  Fired on T4 as call fc-01M1TFD35EPY2YZHNV3VKJP6MG,
    #: lane ddm_sj1_t4_token_predistortion_pass3_20260906, and PROMOTED as pointer move
    #: #32: -8.096393e-04 S against the row above.  Its three legs move together --
    #: seg 0.00012009 -> 0.00010913 (-1.096e-03 S), pose 5.4e-06 -> 5.1e-06 (-2.070e-04 S
    #: from the re-solve), rate 180,904 -> 181,645 B (+741 B, +4.934e-04 S).  The three
    #: sum to -8.096e-04 S, which is the pointer delta.
    #:
    #: Prediction vs measurement, carried openly:
    #:   projected 0.13900021607682325 -> MEASURED 0.13900437796841966, residual
    #:   +4.161892e-06 (+0.0030%), decomposing as -8.796e-07 S of seg (1.04 cells of
    #:   117,964,800 BETTER than the projected leg) and +5.042e-06 S of
    #:   pose (T4 printed 5.10e-06 against the 5.0928018e-06 this arm measured on CPU --
    #:   a genuine CPU->CUDA pose drift, not a print artefact).  Both projections in this
    #:   lineage have now come in OPTIMISTIC by +0.0030% and +0.0038% of S, which is the
    #:   margin any successor's CONTINUE/STOP arithmetic must clear.
    #:
    #: The seg leg landed EXACTLY where the admission put it: 12,866 flipped cells
    #: predicted on the shipped bytes, 12,866 measured, zero disagreement.
    PointerRow(
        label="sj1_token_predistortion_pass3",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
            "/candidate_pass3/candidate_runtime"
        ),
        archive_sha256=(
            "06c44dc464038649f1cc149f04ac03a518294ffcf49b87d8f66df30eb3c63cd3"
        ),
        archive_bytes=181_645,
        d_seg_t4=0.00010913,
        d_pose_t4=5.1e-06,
        score_t4=0.13900437796841966,
    ),
    #: LIVE (MAIN 2026-09-09).  NOT this arm's row: ddm_rc2 replaced the hpac MODEL
    #: section and its reader with a shared 8-byte logistic depth-mixing prior, lane
    #: ddm_rc2_t4_hpac_semistatic_mixing_20260909.  Pointer move #33, -1.5381e-04 S on
    #: -231 B alone: BOTH distortion legs reproduced EXACTLY on T4 (d_seg 0.00010913,
    #: d_pose 5.1e-06 — the same prints as the row above), so the whole delta is rate.
    #:
    #: VERIFIED HERE before this arm composed anything on it, by splitting both members
    #: with ``jg2.split_member`` rather than trusting the handoff:
    #:   semantic 30,246 B  BYTE-IDENTICAL to pass 3
    #:   carrier  18,621 B  BYTE-IDENTICAL to pass 3
    #:   tail    120,321 B  BYTE-IDENTICAL to pass 3
    #:   hpac     12,343 -> 12,112 B  (-231, exactly the declared delta)
    #:   header       14 B  differs (the codec-version rider, as at the rc1 rung)
    #: That identity is what makes pass 4 re-basable onto this tree without re-rendering:
    #: the carrier re-solve starts from the SAME coefficients, and the new tail splices
    #: into a member whose other sections this arm already measured.
    PointerRow(
        label="rc2_hpac_semistatic_mixing",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_rc2_hpac_semistatic_mixing"
            "/candidate_runtime"
        ),
        archive_sha256=(
            "c810c2c7f72e57670dc29bde27d584b18aa82feff68b063936a61dca89cf671e"
        ),
        archive_bytes=181_414,
        d_seg_t4=0.00010913,
        d_pose_t4=5.1e-06,
        score_t4=0.13885056455024844,
    ),
    #: LIVE (MAIN 2026-09-09).  Also not this arm's row: ddm_pc2 set all 12 basis scales
    #: to 1.0 and re-solved, lane ddm_pc2_t4_carrier_scales_resolve_20260909.  Pointer
    #: move #34, -2.7300e-05 S on -41 B.  Both distortion legs print the SAME as the two
    #: rows above (d_seg 0.00010913, d_pose 5.1e-06), so this is rate again.
    #:
    #: NOTE THE PATH: ``rebase_scales_resolve/``, not ``candidate_scales/``.  A carrier
    #: solve seeded from the wrong sibling directory is the exact failure
    #: ``assert_carrier_is_pointer`` exists to refuse, and the two directories differ by
    #: one word.
    #:
    #: VERIFIED HERE before the carrier chain was pointed at it, again by splitting the
    #: members rather than trusting the handoff:
    #:   semantic 30,246 B and tail 120,321 B  BYTE-IDENTICAL to pass 3
    #:   hpac     12,112 B                     BYTE-IDENTICAL to rc2 (rc2's section, kept)
    #:   carrier  18,621 -> 18,580 B           (-41, exactly the declared delta)
    #:   header       14 B                     differs (the codec-version rider)
    #: The byte-identical tail is why this arm's pricing control does not move: pass 4 is
    #: still measured against ``tail_sj1_pass3_subset.bin`` (120,225 B).  And nothing in
    #: this arm assumes the new scales: ``up2.load_carrier_state`` READS basis, coefficient
    #: scales and codes out of the pointer tree's own archive using that tree's own
    #: ``runtime/*`` reader, so ``refine_pair`` starts from pc2's coefficients by
    #: construction rather than by anyone remembering to change a constant.
    PointerRow(
        label="pc2_carrier_scales_resolve",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut"
            "/rebase_scales_resolve/candidate_runtime"
        ),
        archive_sha256=(
            "e138ee097905902ad6e1d49841b2ff2f043736298079f66c0a1628031bcd8372"
        ),
        archive_bytes=181_373,
        d_seg_t4=0.00010913,
        d_pose_t4=5.1e-06,
        score_t4=0.13882326433317044,
    ),
    #: LIVE (MAIN 2026-09-09).  THIS ARM'S pass 4, fired and PROMOTED as pointer move #35:
    #: lane ddm_sj1_t4_token_predistortion_pass4_20260909, -1.515461e-04 S on +148 B.
    #: VERIFIED AT SOURCE before this row was written -- the lane's own
    #: MODAL_REMOTE_RESULT.json reads sha b0ca809c…, 181,521 B, and
    #: score_recomputed_from_components 0.13867171823146562, and the tree on disk hashes
    #: to the same sha at the same size.
    #:
    #: THE CALIBRATION FLIPPED SIGN.  Projected 0.13867280587520867 -> MEASURED
    #: 0.13867171823146562, residual **-1.0876e-06**: the first PESSIMISTIC projection of
    #: the wave, against +5.2748e-06 at move 31 and +4.1619e-06 at move 32.  The band this
    #: arm carried (+0.0030% to +0.0038%, one-signed) is therefore NOT one-signed, and the
    #: honest reading is a residual of order 1e-06 to 5e-06 with EITHER sign -- the pose
    #: print moved 5.049766e-06 (measured here) -> 5.05e-06 (T4), and the seg leg came in
    #: at 0.00010698 against 0.00010699252 projected, i.e. about ONE cell better.
    PointerRow(
        label="sj1_token_predistortion_pass4",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion"
            "/candidate_pass4/candidate_runtime"
        ),
        archive_sha256=(
            "b0ca809ce2c657dfce97e73148a83b9b20c128461ced4c1f6ce1b386ddfd1d20"
        ),
        archive_bytes=181_521,
        d_seg_t4=0.00010698,
        d_pose_t4=5.05e-06,
        score_t4=0.13867171823146562,
    ),
    #: LIVE (MAIN 2026-09-09).  ddm_cmp1 composed rc3's 24-weight model-section mixer with
    #: tc1's 35-weight TOKEN-TAIL mixer over HPAC, lane
    #: ddm_cmp1_t4_rc3_tc1_composed_20260909.  Pointer move #36, **-4.987284e-04 S on
    #: -749 B**, the largest rate move of the wave.
    #:
    #: BOTH DISTORTION LEGS ARE THIS ARM'S OWN, UNCHANGED: d_seg 0.00010698 and d_pose
    #: 5.05e-06 are move 35's prints exactly, because cmp1 re-CODES the very field this arm
    #: shipped without altering a single token.  VERIFIED by splitting the members:
    #:   semantic 30,246 B and carrier 18,586 B  BYTE-IDENTICAL to move 35
    #:   hpac     12,112 -> 11,911 B  (-201, rc3's model-section mixer)
    #:   tail    120,463 -> 119,915 B  (-548, tc1's coder on an IDENTICAL token field)
    #:   header       14 B differs; -201 + -548 = -749, exactly the declared delta.
    #:
    #: CONSEQUENCE FOR THIS ARM, and it is not small: the shipped TAIL CODER is no longer
    #: the jg2/HPAC path every byte number in this memo was measured through.  A pass-5
    #: price carried from pass 4's 6.4000 bits/token would be measured against a coder the
    #: pointer no longer ships.  The break-even must be RE-DERIVED under tc1's
    #: probabilities and the control stream READ FROM THIS TREE, never carried.  The
    #: existing ``stage-tail --expect-pointer-stream`` guard already fail-closes on the
    #: mismatch: a jg2-coded stream cannot be spliced into a tc1-coded member.
    PointerRow(
        label="cmp1_rc3_tc1_composed",
        tree=Path("/Volumes/VertigoDataTier/pact/ddm_cmp1_compose/candidate_runtime"),
        archive_sha256=(
            "66b8d5bb8996f893f867a51e21d35ae8c8a705783fe1089dab1dffae0831b3c0"
        ),
        archive_bytes=180_772,
        d_seg_t4=0.00010698,
        d_pose_t4=5.05e-06,
        score_t4=0.13817298987557713,
    ),
    #: LIVE (MAIN 2026-09-09).  ddm_cmp2 recoded the SEMANTIC section with sm1's 24-weight
    #: mixer over rc1's coder, lane ddm_cmp2_t4_sm1_semantic_coder_20260909.  Pointer move
    #: #37, -2.556898e-04 S on -384 B.  MEASURED section deltas:
    #:   semantic 30,246 -> 29,862 B  (-384, the ONLY changed section besides the header)
    #:   hpac 11,911 B, carrier 18,586 B, tail 119,915 B  ALL BYTE-IDENTICAL to cmp1
    #: and both distortion legs are STILL this arm's move-35 prints.  Three consecutive
    #: pointer moves (36, 37 and cmp1's tail half) have now bought -1.13 mS of pure RATE on
    #: a token field and a carrier this arm has not touched since move 35.
    #:
    #: NAME-vs-CONTENT, recorded because this arm has met the genus three times today: the
    #: seal's ``candidate_id`` is ``ddm_cmp2_sm1_fe1_composed`` and names fe1, but the
    #: measured deltas show ONLY the semantic section's coding changed, with d_seg and
    #: d_pose unchanged -- so nothing in these bytes alters a render.  The label is wider
    #: than the object; the sections are the object.
    PointerRow(
        label="cmp2_sm1_semantic_coder",
        tree=Path("/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/candidate_runtime"),
        archive_sha256=(
            "670d38d05eb142fec9579337e21d7c6522592769ec00c0271aa971ee018ce6bc"
        ),
        archive_bytes=180_388,
        d_seg_t4=0.00010698,
        d_pose_t4=5.05e-06,
        score_t4=0.13791730003757818,
    ),
    #: MOVE 38 -- THIS ARM'S pass 5, PROMOTED.  Lane
    #: ddm_sj1_t4_token_predistortion_pass5_20260909, call fc-01M24EG4R59D365KWRT7538XA8,
    #: -2.700627e-05 S on +48 B: the 42-pair Lagrange subset of pass 5's 126-pair field
    #: (80 of 237 cells for 47 of 229 tail bytes) plus its carrier re-solve.  VERIFIED at
    #: source from the lane's own MODAL_REMOTE_RESULT.json.  The projection residual was
    #: +7.734902e-06 and it is EXACTLY the sum of the two 8-dp print gaps -- zero
    #: modelling error, because the seg leg was verified at zero-cell disagreement on the
    #: shipped bytes before the fire.
    PointerRow(
        label="sj1_pass5_token_predistortion",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_sj1_pass5_price/candidate_pass5/candidate_runtime"
        ),
        archive_sha256=(
            "eae99e0083129a911103bf691bdbf3763b9b7c0e3e365a6e3dc1cd2d2d4eec07"
        ),
        archive_bytes=180_436,
        d_seg_t4=0.00010632,
        d_pose_t4=5.06e-06,
        score_t4=0.1378902937630636,
    ),
    #: MOVE 39 -- LIVE.  rp1's rate-directed token pre-distortion, lane
    #: ddm_rp1_t4_rate_directed_predistortion_20260910, -2.21e-04 S on -250 B: 473
    #: argmax-NEUTRAL token changes over 253 pairs, so d_seg returns to move 37's
    #: 0.00010698 -- rp1's field repairs nothing and breaks nothing, it only makes the
    #: field cheaper to code.  It SUPERSEDES this arm's move 38 as the pointer; the
    #: pass-5 subset re-bases onto it by re-verification (282-pair union, 13 overlapping
    #: pairs, ZERO position collisions measured).
    #:
    #: WHY THIS ROW HAD TO BE APPENDED BEFORE THE NEXT POSE LEG: with the table stale at
    #: move 37, `pose --tag base` paired move 37's CARRIER with move 39's field renders
    #: and read 1.0429e-04 against move 39's 4.89e-06 print -- 21.3x.  pm2's
    #: pose_base_magnitude gate REFUSED it.  That is the stale-base genus
    #: ([[pose_base_must_be_measured_on_the_pointers_own_configuration_on_the_arms_instrument_20260909]])
    #: caught by a gate instead of by eyeballing a magnitude, which is the whole point of
    #: the gate: at 2x instead of 21x it would have sealed.
    PointerRow(
        label="rp1_rate_directed_predistortion",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_rp1_rate_directed_predistortion/candidate/candidate_runtime"
        ),
        archive_sha256=(
            "8877f75d87bf25b410264e08682959c7710cf677307bd5452039ce53835f6bf4"
        ),
        archive_bytes=180_186,
        d_seg_t4=0.00010698,
        d_pose_t4=4.89e-06,
        score_t4=0.1376693148220904,
    ),
    #: MOVE 40 -- LIVE.  THIS ARM'S move-38 subset RE-BASED onto move 39 by
    #: re-verification: 282-pair union, 13 overlapping pairs, zero position collisions,
    #: and the seg recount that mattered -- 80 repairs became 74, all six losses on the
    #: overlap.  Lane ddm_sj1_t4_compose39_rp1_union_20260910, call
    #: fc-01M24MA90ER7YTGRTCX9SNQNM4, -3.070463e-05 S on +47 B.  The projection residual
    #: was +2.839377e-06 and is again EXACTLY the sum of the two 8-dp print gaps.
    PointerRow(
        label="sj1_compose39_rp1_union",
        tree=Path(
            "/Volumes/VertigoDataTier/pact/ddm_sj1_compose39_price/candidate/candidate_runtime"
        ),
        archive_sha256=(
            "986d536b31ed1079c517dadea73ba33daf018c53692a2b2fbbf8d6244dfe9857"
        ),
        archive_bytes=180_233,
        d_seg_t4=0.00010636,
        d_pose_t4=4.89e-06,
        score_t4=0.13763861019288715,
    ),
)

for _row in POINTER_LINEAGE:
    _row.verify_arithmetic()

#: The live pointer is the LAST row, never a name repeated in five places.
LIVE_POINTER = POINTER_LINEAGE[-1]
POINTER_TREE = LIVE_POINTER.tree
POINTER_ARCHIVE = LIVE_POINTER.archive
POINTER_ARCHIVE_SHA256 = LIVE_POINTER.archive_sha256
POINTER_ARCHIVE_BYTES = LIVE_POINTER.archive_bytes
POINTER_SCORE_T4 = LIVE_POINTER.score_t4
POINTER_D_SEG_T4 = LIVE_POINTER.d_seg_t4
POINTER_D_POSE_T4 = LIVE_POINTER.d_pose_t4

#: The tree this arm RENDERS and MEASURES d_seg on.  Deliberately NOT the pointer tree:
#: jg1's semantic loader reads the section in cl2's coding, while rc1/pc1 ship it
#: adaptively recoded and restore it byte-for-byte inside the receiver.  Reading each
#: section from the tree whose codec the instrument understands is the same discipline
#: MAIN prescribed for the token stream -- it is not mixing two bodies, because the
#: sections involved are byte-identical objects.
RENDER_TREE = BODY_TREE


def assert_carrier_is_pointer(runtime_dir: Path) -> str:
    """Refuse a carrier solve that is not anchored on the LIVE pointer body.

    A carrier re-solve seeded from a superseded body's codes and scales produces codes
    that are structurally valid and numerically wrong: it reverts whichever carrier move
    the pointer last banked, and nothing downstream can see it, because every code still
    sits inside the signed-int12 container.  So the anchor is checked, not remembered.
    """
    archive = Path(runtime_dir) / "archive.zip"
    if not archive.is_file():
        raise Sj1Error(
            f"carrier runtime {runtime_dir} has no archive.zip to identify; a solve "
            "cannot be anchored on a body that is not on disk"
        )
    observed = _sha256_file(archive)
    if observed != POINTER_ARCHIVE_SHA256:
        raise Sj1Error(
            f"carrier runtime {runtime_dir} has archive sha {observed}, not the live "
            f"pointer's {POINTER_ARCHIVE_SHA256}. Solving from a superseded body's "
            "codes and scales silently reverts the carrier move the pointer banked "
            "(pc1 V3: scales x4, codes /4) while every code stays a valid int12."
        )
    return observed

#: MAIN's binding pin 2026-09-05: the T4-scored GT argmax table, and the instrument's
#: own step-0 reading of it on this body.  The 1.23% gap between them is the INSTRUMENT
#: RESIDUAL and is carried explicitly into every projection rather than being absorbed.
GT_DALI_SHA256 = "a91d98252fe377c51ff7f3380c2fc9d30d84093fc54ee89e5e5f5102e6354994"
INSTRUMENT_BASE_D_SEG_DALI = 2.0387e-4

#: Score arithmetic, from ``upstream/evaluate.py``.
S_PER_SEG_CELL = jg1.S_PER_SEG_CELL
S_PER_ARCHIVE_BYTE = jg1.S_PER_ARCHIVE_BYTE
BYTES_PER_SEG_CELL = jg1.BYTES_PER_SEG_CELL

#: The renderer's own receptive radius in token cells, DERIVED from cpr1/inflate.py
#: (coord_mix 1x1 + depthwise 3x3 at dilations 1,1,2,4 + head 3x3).
RENDER_RADIUS_TOKENS = 9


class Sj1Error(RuntimeError):
    """A ddm_sj1 precondition failed.  Fail closed, never approximate."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _set_threads(threads: int) -> None:
    """Pin torch's CPU thread count so N concurrent shards do not oversubscribe."""
    import torch

    torch.set_num_threads(max(1, threads))


# --------------------------------------------------------------------------------------
# Loading -- the body, held together so no caller can mix two bodies by accident.
# --------------------------------------------------------------------------------------


@dataclass
class Body:
    """The cl2 frontier body: renderer, scorer, token field, GT table -- all verified."""

    semantic: Any
    net: Any
    tokens: np.ndarray
    gt: np.ndarray
    raw: Any
    receipts: dict[str, Any] = field(default_factory=dict)


def load_body(
    *,
    with_raw: bool = True,
    verify_shas: bool = True,
    lineage: str = up2.LINEAGE_DALI,
    axis: str = "contest_cuda",
) -> Body:
    """Load the frontier body and refuse unless every identity holds.

    The GT lineage gate is ``up2.verify_gt_lineage`` and is not optional: the DALI and
    PyAV tables differ at 20,671 argmax sites on this body (MAIN pin), which is 87% of
    the entire flip budget this arm is trying to repair.
    """
    up2.verify_gt_lineage(axis=axis, declared_lineage=lineage)

    receipts: dict[str, Any] = {
        "axis": axis,
        "gt_lineage": lineage,
        "body_archive": str(BODY_ARCHIVE),
        "body_tokens": str(BODY_TOKENS),
        "body_raw": str(BODY_RAW),
    }
    if verify_shas:
        archive_sha = _sha256_file(BODY_ARCHIVE)
        if archive_sha != BODY_ARCHIVE_SHA256:
            raise Sj1Error(
                f"body archive sha {archive_sha} != sealed {BODY_ARCHIVE_SHA256}"
            )
        gt_path = jg1.DEFAULT_GT_DALI if lineage == up2.LINEAGE_DALI else jg1.DEFAULT_GT_AV
        gt_sha = _sha256_file(gt_path)
        if lineage == up2.LINEAGE_DALI and gt_sha != GT_DALI_SHA256:
            raise Sj1Error(
                f"GT DALI table sha {gt_sha} != MAIN's pinned {GT_DALI_SHA256}"
            )
        receipts["body_archive_sha256"] = archive_sha
        receipts["gt_table_sha256"] = gt_sha
        receipts["gt_table_path"] = str(gt_path)

    tokens = jg1.load_tokens(BODY_TOKENS)
    gt = jg1.load_gt_seg_labels(lineage)
    semantic = jg1.load_semantic_renderer(
        archive_path=BODY_ARCHIVE, runtime_dir=BODY_RUNTIME
    )
    net = jg1.load_segnet()
    raw = None
    if with_raw:
        # ``verify_sha=False`` because up2's known-sha set is the OLD pointer body; the
        # cl2 decode's identity is bound by BODY_RAW_SHA256 from its own PARSEBACK receipt.
        raw = up2.open_raw(BODY_RAW, verify_sha=False)
    return Body(
        semantic=semantic, net=net, tokens=tokens, gt=gt, raw=raw, receipts=receipts
    )


def argmax_for_tokens(body: Body, tokens_pair: np.ndarray, pair: int) -> np.ndarray:
    """Realize one pair: render frame 2p+1 from tokens, re-segment, return the argmax.

    This is the receiver's own forward model (``jg1.render_frame1``, batch 1 by
    construction) followed by the frozen CPU SegNet through the evaluator's own
    preprocess.  No surrogate anywhere on this path.
    """
    frame = jg1.render_frame1(body.semantic, tokens_pair[None], np.array([pair]))
    return jg1.argmax_from_camera_frames(body.net, frame)[0]


def argmax_from_raw(body: Body, pair: int) -> np.ndarray:
    """SegNet argmax of a DECODE for one pair, read from a receiver's own ``0.raw``."""
    frame = np.asarray(body.raw[2 * int(pair) + 1])[None]
    return jg1.argmax_from_camera_frames(body.net, frame)[0]


def open_raw_decode(path: Path):
    """Memmap any receiver decode at the shipped shape, refusing a wrong size.

    ``up2.open_raw`` pins a sha allow-list built for the OLD pointer body, so a candidate
    decode -- which by construction has a NEW sha -- could never pass it.  The identity
    that binds here is the size plus the caller's own recorded sha in the run receipt.
    """
    expected = 2 * N_PAIRS * jg1.CAMERA_H * jg1.CAMERA_W * 3
    actual = path.stat().st_size
    if actual != expected:
        raise Sj1Error(f"decode {path} is {actual} B, expected {expected} B")
    return np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        shape=(2 * N_PAIRS, jg1.CAMERA_H, jg1.CAMERA_W, 3),
    )


# --------------------------------------------------------------------------------------
# step0 -- reproduce the frontier body's seg leg before proposing anything
# --------------------------------------------------------------------------------------


def cmd_step0(args) -> int:
    """Measure d_seg on the frontier body, one shard of pairs, from the receiver's 0.raw.

    Reads the SHIPPED frames rather than re-rendering, so this leg is a property of the
    bytes cl2 sealed.  The re-render identity is a separate control (``control``).
    """
    _set_threads(args.threads)
    body = load_body(with_raw=args.raw is None, verify_shas=not args.no_verify_shas)
    if args.raw is not None:
        body.raw = open_raw_decode(args.raw)
        body.receipts["decode_path"] = str(args.raw)
        body.receipts["decode_sha256"] = _sha256_file(args.raw)
    indices = np.array_split(np.arange(N_PAIRS, dtype=np.int64), args.shard_count)[
        args.shard_index
    ]
    started = time.time()
    out = np.empty((len(indices), EVAL_H, EVAL_W), dtype=np.uint8)
    for row, pair in enumerate(indices):
        out[row] = argmax_from_raw(body, int(pair))
        if args.progress and (row + 1) % 10 == 0:
            rate = (row + 1) / (time.time() - started)
            print(
                f"  shard {args.shard_index}: {row + 1}/{len(indices)} "
                f"({rate:.2f} pair/s)",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    gt_chunk = body.gt[indices]
    disagree = int((out != gt_chunk).sum())
    receipt = {
        "schema": "ddm_sj1_step0_shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": [int(p) for p in indices],
        "cells": int(gt_chunk.size),
        "cells_disagreeing": disagree,
        "d_seg_shard": disagree / gt_chunk.size,
        "argmax_path": str(args.out),
        "argmax_sha256": _sha256_file(args.out),
        "elapsed_seconds": time.time() - started,
        "receipts": body.receipts,
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ("shard_index", "cells_disagreeing", "d_seg_shard", "elapsed_seconds")}))
    return 0


def cmd_step0_merge(args) -> int:
    """Merge step-0 shards into the n600 seg leg and compare against the published legs."""
    shards = sorted(args.shards, key=lambda p: json.loads(Path(p).read_text())["shard_index"])
    receipts = [json.loads(Path(p).read_text()) for p in shards]
    if len({r["shard_count"] for r in receipts}) != 1:
        raise Sj1Error("shard receipts disagree on shard_count")
    covered = [p for r in receipts for p in r["pairs"]]
    if sorted(covered) != list(range(N_PAIRS)):
        raise Sj1Error(
            f"shards cover {len(covered)} pairs, not the full n600 field; "
            "a sub-n600 seg verdict is a TOY on this axis"
        )
    argmax = np.empty((N_PAIRS, EVAL_H, EVAL_W), dtype=np.uint8)
    for r in receipts:
        chunk = np.load(r["argmax_path"])
        argmax[np.array(r["pairs"], dtype=np.int64)] = chunk
    args.out_argmax.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_argmax, argmax)

    tokens = jg1.load_tokens(BODY_TOKENS)
    legs: dict[str, Any] = {}
    for lineage in (up2.LINEAGE_DALI, up2.LINEAGE_AV_PYAV):
        gt = jg1.load_gt_seg_labels(lineage)
        per_pair = jg1.d_seg_per_pair(argmax, gt)
        entry = {
            "d_seg": float(per_pair.mean()),
            "cells_disagreeing": int((argmax != gt).sum()),
            "cells": int(gt.size),
            "flip_ledger": jg1.flip_ledger(argmax, gt, tokens),
            "token_vs_gt": jg1.token_vs_gt_agreement(
                tokens, gt, np.arange(N_PAIRS, dtype=np.int64)
            ),
        }
        legs[lineage] = entry
    dali = legs[up2.LINEAGE_DALI]["d_seg"]
    report = {
        "schema": "ddm_sj1_step0.v1",
        "pairs": N_PAIRS,
        "sampling": "full_field",
        "legs": legs,
        "published": {
            "t4_d_seg": BASE_D_SEG_T4,
            "instrument_expected_dali": INSTRUMENT_BASE_D_SEG_DALI,
        },
        "instrument_residual_vs_t4": dali / BASE_D_SEG_T4 - 1.0,
        "instrument_residual_vs_main_pin": dali / INSTRUMENT_BASE_D_SEG_DALI - 1.0,
        "argmax_path": str(args.out_argmax),
        "argmax_sha256": _sha256_file(args.out_argmax),
        "exchange_rate": {
            "s_per_seg_cell": S_PER_SEG_CELL,
            "s_per_archive_byte": S_PER_ARCHIVE_BYTE,
            "bytes_per_repaired_seg_cell": BYTES_PER_SEG_CELL,
        },
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"[dali] d_seg={dali:.8f} flips={legs[up2.LINEAGE_DALI]['cells_disagreeing']} "
        f"vs T4 {BASE_D_SEG_T4} (residual {report['instrument_residual_vs_t4']:+.4%}) "
        f"vs MAIN pin {INSTRUMENT_BASE_D_SEG_DALI} "
        f"(residual {report['instrument_residual_vs_main_pin']:+.4%})"
    )
    print(f"[av_pyav] d_seg={legs[up2.LINEAGE_AV_PYAV]['d_seg']:.8f}")
    return 0


# --------------------------------------------------------------------------------------
# control -- does re-rendering the shipped tokens reproduce the shipped frames exactly?
# --------------------------------------------------------------------------------------


def cmd_control(args) -> int:
    """The forward-model control.  Without it a 'realized' objective is realized against
    a model that may not be the receiver's."""
    _set_threads(args.threads)
    body = load_body(with_raw=True, verify_shas=not args.no_verify_shas)
    indices = up2.select_pairs(args.pairs, args.seed)
    started = time.time()
    rendered = jg1.render_frame1(body.semantic, body.tokens[indices], indices)
    shipped = np.asarray(body.raw[2 * indices + 1])
    delta = np.abs(rendered.astype(np.int16) - shipped.astype(np.int16))
    report = {
        "schema": "ddm_sj1_forward_control.v1",
        "pairs": [int(p) for p in indices],
        "pixels_compared": int(shipped.size),
        "pixels_changed": int((delta > 0).sum()),
        "max_abs_delta": int(delta.max()),
        "byte_exact": bool(delta.max() == 0),
        "semantic_batch": 1,
        "elapsed_seconds": time.time() - started,
        "receipts": body.receipts,
        "axis": "[macOS-CPU advisory]",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True)[:900])
    return 0 if report["byte_exact"] else 3


# --------------------------------------------------------------------------------------
# probe -- MEASURE the influence radius of a single token change
# --------------------------------------------------------------------------------------


#: The 3x3 token offsets of the charter's proposal family, centre first.
FAMILY_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def class_role(new_class: int, gt_class: int, our_class: int) -> str:
    """Name a proposed class by its ROLE at the site, not by its absolute index.

    ``gt`` (widen the class SegNet is failing to see), ``ours`` (widen the class it
    wrongly sees) and ``other`` generalise across sites; the absolute index does not,
    because Road is the hub of 80.5% of this body's flips and would swamp any
    index-keyed yield table.
    """
    if new_class == gt_class:
        return "gt"
    if new_class == our_class:
        return "ours"
    return "other"


def cmd_probe(args) -> int:
    """How far does ONE changed token move the SegNet argmax, and which moves PAY?

    Two questions, one sweep, because both need the same realized evaluation:

    * **radius** -- the renderer's own radius is DERIVED (9 token cells); SegNet's is
      large in theory, so the operative number is MEASURED as the Chebyshev distance of
      the furthest changed argmax cell.  This SIZES the batching.  It does not license
      it: every accepted set in a real pass is closed by a composite verify.
    * **yield by (offset, class role)** -- the charter's family is 9 offsets x 4
      alternative classes = 36 moves per flipped cell.  Measuring which of the 36 ever
      repairs anything is what makes a full n600 pass affordable.  It ORDERS the search;
      no combination is excluded from the pass on this evidence
      (a 10-pair yield table is a sizing instrument, never a family verdict).
    """
    _set_threads(args.threads)
    body = load_body(with_raw=False, verify_shas=not args.no_verify_shas)
    rng = np.random.default_rng(args.seed)
    indices = up2.select_pairs(args.pairs, args.seed)
    rows: list[dict[str, Any]] = []
    started = time.time()
    for pair in indices:
        pair = int(pair)
        base_argmax = argmax_for_tokens(body, body.tokens[pair], pair)
        gt_pair = body.gt[pair]
        wrong = base_argmax != gt_pair
        flips_before = int(wrong.sum())
        ys, xs = np.nonzero(wrong)
        if ys.size == 0:
            continue
        picks = rng.choice(ys.size, size=min(args.sites, ys.size), replace=False)
        for pick in picks:
            y, x = int(ys[pick]), int(xs[pick])
            gt_class = int(gt_pair[y, x])
            our_class = int(base_argmax[y, x])
            for dy, dx in FAMILY_OFFSETS:
                py = min(max(y + dy, 0), EVAL_H - 1)
                px = min(max(x + dx, 0), EVAL_W - 1)
                old_class = int(body.tokens[pair, py, px])
                for new_class in range(NUM_CLASSES):
                    if new_class == old_class:
                        continue
                    proposal = body.tokens[pair].copy()
                    proposal[py, px] = new_class
                    argmax = argmax_for_tokens(body, proposal, pair)
                    changed = argmax != base_argmax
                    cy, cx = np.nonzero(changed)
                    radius = (
                        int(max(np.abs(cy - py).max(), np.abs(cx - px).max()))
                        if cy.size
                        else -1
                    )
                    flips_after = int((argmax != gt_pair).sum())
                    rows.append(
                        {
                            "pair": pair,
                            "site_y": y,
                            "site_x": x,
                            "offset": [dy, dx],
                            "token_y": py,
                            "token_x": px,
                            "gt_class": gt_class,
                            "our_class": our_class,
                            "old_class": old_class,
                            "new_class": new_class,
                            "class_role": class_role(new_class, gt_class, our_class),
                            "argmax_cells_changed": int(changed.sum()),
                            "chebyshev_radius": radius,
                            "flips_before": flips_before,
                            "flips_after": flips_after,
                            "flips_repaired": flips_before - flips_after,
                        }
                    )
        if args.progress:
            print(
                f"  probe pair {pair}: {len(rows)} rows "
                f"({(time.time() - started) / max(1, len(rows)):.2f} s/move)",
                flush=True,
            )
    radii = [r["chebyshev_radius"] for r in rows if r["chebyshev_radius"] >= 0]
    yield_table: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['offset'][0]},{row['offset'][1]}|{row['class_role']}"
        cell = yield_table.setdefault(
            key, {"moves": 0, "positive": 0, "repaired_total": 0, "best": 0}
        )
        cell["moves"] += 1
        if row["flips_repaired"] > 0:
            cell["positive"] += 1
            cell["repaired_total"] += row["flips_repaired"]
            cell["best"] = max(cell["best"], row["flips_repaired"])
    for cell in yield_table.values():
        cell["hit_rate"] = cell["positive"] / cell["moves"]
        cell["mean_repaired_per_move"] = cell["repaired_total"] / cell["moves"]
    report = {
        "schema": "ddm_sj1_influence_probe.v2",
        "rows": rows,
        "moves": len(rows),
        "moves_with_any_argmax_change": len(radii),
        "moves_repairing": sum(1 for r in rows if r["flips_repaired"] > 0),
        "render_radius_tokens_derived": RENDER_RADIUS_TOKENS,
        "chebyshev_radius": {
            "max": max(radii) if radii else None,
            "p999": float(np.percentile(radii, 99.9)) if radii else None,
            "p99": float(np.percentile(radii, 99)) if radii else None,
            "p50": float(np.percentile(radii, 50)) if radii else None,
        },
        "yield_by_offset_and_class_role": yield_table,
        "elapsed_seconds": time.time() - started,
        "seconds_per_realized_move": (time.time() - started) / max(1, len(rows)),
        "receipts": body.receipts,
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "sizing_instrument_not_a_family_verdict": True,
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    summary = {
        k: report[k]
        for k in (
            "moves",
            "moves_with_any_argmax_change",
            "moves_repairing",
            "chebyshev_radius",
            "seconds_per_realized_move",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0


# --------------------------------------------------------------------------------------
# partition -- what fraction of the shipped vehicle's flips NO single-cell move repairs
# --------------------------------------------------------------------------------------


def cmd_partition(args) -> int:
    """Measure the PERSISTENT partition of the shipped vehicle's flipped cells.

    A site is PERSISTENT when no move in the whole 36-family (flipped cell + 8 neighbours
    x 4 alternative classes), applied ALONE, lowers the pair's realized flip count.  That
    is the ceiling of this formulation, so it is measured rather than assumed.

    Sites are drawn uniformly from the FULL flip set across all 600 pairs, not one per
    sampled pair: sampling pairs and then a site within each would weight every site by
    the inverse of its pair's flip count, under-sampling exactly the crowded pairs that
    carry most of the debt.  Shards take a disjoint slice of one seeded permutation, so
    the union of shards is one clean sample and any shard that dies leaves an unbiased
    partial.
    """
    _set_threads(args.threads)
    body = load_body(with_raw=False, verify_shas=not args.no_verify_shas)
    argmax_field = np.load(args.argmax)
    if argmax_field.shape != (N_PAIRS, EVAL_H, EVAL_W):
        raise Sj1Error(f"argmax field has shape {argmax_field.shape}")
    prior: dict[int, np.ndarray] = {}
    if args.field is not None:
        with np.load(args.field, allow_pickle=False) as blob:
            for key in blob.files:
                prior[int(key)] = np.asarray(blob[key], dtype=np.uint8)

    pairs_all, ys, xs = np.nonzero(argmax_field != body.gt)
    total_sites = int(pairs_all.size)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(total_sites)[: args.sites]
    order = np.sort(order[args.shard_index :: args.shard_count])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.time()
    cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
    for count, idx in enumerate(order.tolist()):
        pair = int(pairs_all[idx])
        site_y, site_x = int(ys[idx]), int(xs[idx])
        if pair not in cache:
            tokens = np.ascontiguousarray(prior.get(pair, body.tokens[pair]))
            argmax = (
                argmax_field[pair]
                if pair not in prior
                else argmax_for_tokens(body, tokens, pair)
            )
            cache[pair] = (tokens, argmax, int((argmax != body.gt[pair]).sum()))
        tokens, argmax, flips_before = cache[pair]
        gt_class = int(body.gt[pair, site_y, site_x])
        our_class = int(argmax[site_y, site_x])
        best = 0
        tried = 0
        winners: list[dict[str, Any]] = []
        for selector in CLASS_SELECTORS:
            new_class = resolve_selector(selector, gt_class, our_class)
            if new_class is None:
                continue
            for dy, dx in FAMILY_OFFSETS:
                ty = min(max(site_y + dy, 0), EVAL_H - 1)
                tx = min(max(site_x + dx, 0), EVAL_W - 1)
                if int(tokens[ty, tx]) == new_class:
                    continue
                proposal = tokens.copy()
                proposal[ty, tx] = new_class
                trial = argmax_for_tokens(body, proposal, pair)
                tried += 1
                repaired = flips_before - int((trial != body.gt[pair]).sum())
                if repaired > 0:
                    winners.append(
                        {
                            "selector": selector,
                            "offset": [dy, dx],
                            "new_class": new_class,
                            "repaired": repaired,
                        }
                    )
                    best = max(best, repaired)
        rows.append(
            {
                "pair": pair,
                "site_y": site_y,
                "site_x": site_x,
                "gt_class": gt_class,
                "our_class": our_class,
                "moves_tried": tried,
                "best_repaired": best,
                "persistent": best <= 0,
                "winners": winners,
            }
        )
        if args.progress and (count + 1) % 5 == 0:
            persistent = sum(1 for r in rows if r["persistent"])
            print(
                f"  shard {args.shard_index}: {count + 1}/{len(order)} sites, "
                f"persistent {persistent}/{len(rows)} "
                f"({(time.time() - started) / (count + 1):.1f} s/site)",
                flush=True,
            )
    persistent = sum(1 for r in rows if r["persistent"])
    report = {
        "schema": "ddm_sj1_persistent_partition_shard.v1",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "seed": args.seed,
        "sites_requested_total": args.sites,
        "sites_in_shard": len(rows),
        "population_total_flipped_cells": total_sites,
        "persistent_sites": persistent,
        "persistent_fraction_shard": persistent / len(rows) if rows else None,
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "receipts": body.receipts,
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps(
            {k: report[k] for k in ("shard_index", "sites_in_shard", "persistent_sites", "persistent_fraction_shard", "elapsed_seconds")}
        )
    )
    return 0


# --------------------------------------------------------------------------------------
# The pass -- realized coordinate descent on the token lattice
# --------------------------------------------------------------------------------------

#: Class SELECTORS, in the order the sweep tries them.  ``gt`` widens the class SegNet is
#: failing to see and is the mechanically motivated move; ``ours`` widens the class it
#: wrongly sees; ``other*`` are the remaining classes.  Naming them by ROLE rather than by
#: index is what lets one ordering serve every site on a body where Road is the hub of
#: 80.5% of flips.
CLASS_SELECTORS = ("gt", "ours", "other0", "other1", "other2")
STAGE_SELECTORS = {
    "gt": ("gt",),
    "rest": ("ours", "other0", "other1", "other2"),
    "full": CLASS_SELECTORS,
}


def resolve_selector(selector: str, gt_class: int, our_class: int) -> int | None:
    """Map a role name to a concrete class index at one site, or None if degenerate."""
    if selector == "gt":
        return gt_class
    if selector == "ours":
        return our_class
    others = [c for c in range(NUM_CLASSES) if c not in (gt_class, our_class)]
    rank = int(selector[-1])
    return others[rank] if rank < len(others) else None


@dataclass
class Move:
    """One single-token coordinate move, with the site it is meant to repair."""

    site_y: int
    site_x: int
    ty: int
    tx: int
    old_class: int
    new_class: int
    selector: str
    offset: tuple[int, int]
    priority: int


def enumerate_moves(
    tokens_pair: np.ndarray,
    argmax_pair: np.ndarray,
    gt_pair: np.ndarray,
    *,
    selectors: Sequence[str],
) -> list[Move]:
    """The charter's family -- flipped cell + 8 neighbours x 4 alternative classes.

    Ordered by (class selector, offset) so that a sweep tries the mechanically motivated
    moves first; DEDUPED on the concrete ``(token position, new class)`` because two
    adjacent flipped cells with the same GT class generate the same physical move and
    realizing it twice would double-count its repair.
    """
    ys, xs = np.nonzero(argmax_pair != gt_pair)
    moves: list[Move] = []
    seen: set[tuple[int, int, int]] = set()
    priority = 0
    for selector in selectors:
        for offset in FAMILY_OFFSETS:
            dy, dx = offset
            for site_y, site_x in zip(ys.tolist(), xs.tolist(), strict=True):
                gt_class = int(gt_pair[site_y, site_x])
                our_class = int(argmax_pair[site_y, site_x])
                new_class = resolve_selector(selector, gt_class, our_class)
                if new_class is None:
                    continue
                ty = min(max(site_y + dy, 0), EVAL_H - 1)
                tx = min(max(site_x + dx, 0), EVAL_W - 1)
                old_class = int(tokens_pair[ty, tx])
                if old_class == new_class:
                    continue
                key = (ty, tx, new_class)
                if key in seen:
                    continue
                seen.add(key)
                moves.append(
                    Move(
                        site_y=site_y,
                        site_x=site_x,
                        ty=ty,
                        tx=tx,
                        old_class=old_class,
                        new_class=new_class,
                        selector=selector,
                        offset=offset,
                        priority=priority,
                    )
                )
            priority += 1
    moves.sort(key=lambda m: m.priority)
    return moves


@dataclass
class PairState:
    """One pair's live greedy state inside the batched engine."""

    pair: int
    tokens: np.ndarray
    #: The plane this pass STARTED from.  Kept so ``tokens_changed`` is the measured
    #: diff rather than ``len(accepted)``: two accepted moves can land on one token
    #: position, and a move can restore a position to the class it already had, so
    #: counting accepted moves would overstate the rate cost and inflate
    #: cells-per-changed-token -- the exact quantity the break-even is computed from.
    tokens_initial: np.ndarray
    argmax: np.ndarray
    gt: np.ndarray
    flips: int
    flips_before: int
    moves: list[Move]
    cursor: int = 0
    evaluations: int = 0
    realized: int = 0
    accepted: list[dict[str, Any]] = field(default_factory=list)
    pending_move: Move | None = None
    pending_tokens: np.ndarray | None = None

    def next_move(self) -> Move | None:
        """Advance to the next move whose SITE is still flipped.

        A move exists to repair one site; once that site reads GT the move has no
        objective left, and the site may have been repaired by the FAR FIELD of an
        earlier accepted move rather than by a move of its own -- the probe measured a
        single token change moving argmax cells out to a Chebyshev radius of 416, so
        this skip is doing real work, not bookkeeping.
        """
        while self.cursor < len(self.moves):
            move = self.moves[self.cursor]
            self.cursor += 1
            if self.argmax[move.site_y, move.site_x] == self.gt[move.site_y, move.site_x]:
                continue
            if self.tokens[move.ty, move.tx] == move.new_class:
                continue
            return move
        return None


def run_shard_pass(
    body: Body,
    pairs: Sequence[int],
    prior: dict[int, np.ndarray],
    *,
    selectors: Sequence[str],
    batch: int,
    max_evaluations_per_pair: int,
    on_pair_done,
    progress: bool = False,
) -> None:
    """Greedy realized coordinate descent over a group of pairs, SegNet-batched.

    **Why greedy on the WHOLE-PAIR flip count and not on a local window.**  The
    influence probe MEASURED the argmax response to a single token change as SPARSE but
    LONG-RANGE: a move changes a median of 2 argmax cells (p90 = 6), but the furthest
    changed cell sits at a median Chebyshev distance of 19 and a p90 of 155 -- only 34%
    of moves keep every changed cell within 8 token cells.  A windowed attribution would
    therefore misprice the majority of moves by ignoring exactly the far-field cells that
    decide their sign.  So each move is realized on its own and judged on the pair's
    entire realized flip count.  Acceptance is monotone: a pair's flip count never rises.

    **Why the batch exists.**  The renderer must run at batch 1 (``ddm_up2`` sec.6
    measured batch 8 as byte-changing on this half, so batch 8 would not reproduce the
    receiver's decode), but SegNet has no such constraint and is 1.67x cheaper per frame
    at batch 8 (MEASURED 0.315 s/frame vs 0.526 s/frame at 3 threads).  Batching ACROSS
    pairs buys that without touching the render contract.
    """
    states: dict[int, PairState] = {}
    order = list(pairs)
    cursor = 0
    done = 0
    started = time.time()
    #: A pass over one shard is tens of thousands of realized moves and a pair only
    #: closes when its whole family is exhausted, so without this the run is SILENT for
    #: 20+ minutes.  Silence in a control plane is the failure that hides every other
    #: failure ([[m102]]), so the engine reports its own throughput on a fixed cadence.
    heartbeat_every = 200
    total_evaluations = 0
    last_beat = 0

    def admit_next() -> bool:
        nonlocal cursor
        while cursor < len(order):
            pair = int(order[cursor])
            cursor += 1
            tokens = np.ascontiguousarray(prior.get(pair, body.tokens[pair]))
            argmax = argmax_for_tokens(body, tokens, pair)
            gt_pair = body.gt[pair]
            flips = int((argmax != gt_pair).sum())
            moves = enumerate_moves(tokens, argmax, gt_pair, selectors=selectors)
            state = PairState(
                pair=pair,
                tokens=tokens,
                tokens_initial=tokens,
                argmax=argmax,
                gt=gt_pair,
                flips=flips,
                flips_before=flips,
                moves=moves,
                evaluations=1,
            )
            if flips == 0 or not moves:
                on_pair_done(_finish(state))
                continue
            states[pair] = state
            return True
        return False

    while len(states) < batch and admit_next():
        pass

    while states:
        slots: list[PairState] = []
        for state in list(states.values()):
            if len(slots) >= batch:
                break
            move = state.next_move()
            if move is None or state.evaluations >= max_evaluations_per_pair:
                on_pair_done(_finish(state))
                del states[state.pair]
                done += 1
                if progress:
                    elapsed = time.time() - started
                    print(
                        f"  pair {state.pair} done "
                        f"({done}/{len(order)}, {elapsed / max(1, done):.1f} s/pair)",
                        flush=True,
                    )
                continue
            proposal = state.tokens.copy()
            proposal[move.ty, move.tx] = move.new_class
            state.pending_move = move
            state.pending_tokens = proposal
            slots.append(state)
        while len(states) < batch and admit_next():
            pass
        if not slots:
            continue
        frames = np.concatenate(
            [
                jg1.render_frame1(
                    body.semantic,
                    state.pending_tokens[None],
                    np.array([state.pair]),
                )
                for state in slots
            ]
        )
        argmaxes = jg1.argmax_from_camera_frames(body.net, frames)
        for state, argmax in zip(slots, argmaxes, strict=True):
            move = state.pending_move
            state.evaluations += 1
            state.realized += 1
            flips = int((argmax != state.gt).sum())
            if flips < state.flips:
                state.accepted.append(
                    {
                        "site_y": move.site_y,
                        "site_x": move.site_x,
                        "token_y": move.ty,
                        "token_x": move.tx,
                        # The class the token ACTUALLY held when the move landed -- an
                        # earlier accepted move may already have written this position,
                        # and the enumerated ``move.old_class`` is the pass-start value.
                        "old_class": int(state.tokens[move.ty, move.tx]),
                        "new_class": move.new_class,
                        "selector": move.selector,
                        "offset": list(move.offset),
                        "repaired": state.flips - flips,
                    }
                )
                state.tokens = state.pending_tokens
                state.argmax = argmax
                state.flips = flips
            state.pending_move = None
            state.pending_tokens = None
        total_evaluations += len(slots)
        if progress and total_evaluations - last_beat >= heartbeat_every:
            last_beat = total_evaluations
            elapsed = time.time() - started
            live_repaired = sum(s.flips_before - s.flips for s in states.values())
            print(
                f"  [beat] evals={total_evaluations} "
                f"({elapsed / max(1, total_evaluations):.2f} s/eval) "
                f"pairs_done={done}/{len(order)} in_flight={len(states)} "
                f"repaired_in_flight={live_repaired}",
                flush=True,
            )


def _finish(state: PairState) -> dict[str, Any]:
    """Close one pair: the realized ledger row plus its plane when it changed."""
    tokens_changed = int((state.tokens != state.tokens_initial).sum())
    repaired = state.flips_before - state.flips
    return {
        "pair": state.pair,
        "moves_accepted": len(state.accepted),
        "flips_before": state.flips_before,
        "flips_after": state.flips,
        "flips_repaired": repaired,
        "accepted": state.accepted,
        "tokens_changed": tokens_changed,
        "cells_per_changed_token": repaired / tokens_changed if tokens_changed else 0.0,
        "proposals_enumerated": len(state.moves),
        "proposals_realized": state.realized,
        "evaluations": state.evaluations,
        "plane_changed": tokens_changed > 0,
        "plane": state.tokens if tokens_changed else None,
        "sites_persistent": state.flips,
    }


def cmd_pass(args) -> int:
    """Run one pass over a STRIDED shard of all 600 pairs.

    Strided, never contiguous: a contiguous block of this video is a different population
    ([[m88]]), so a shard that dies leaves an unbiased partial rather than a scene block.
    """
    _set_threads(args.threads)
    body = load_body(with_raw=False, verify_shas=not args.no_verify_shas)
    prior: dict[int, np.ndarray] = {}
    if args.field is not None:
        with np.load(args.field, allow_pickle=False) as blob:
            for key in blob.files:
                plane = np.asarray(blob[key], dtype=np.uint8)
                if plane.shape != (EVAL_H, EVAL_W) or plane.max() >= NUM_CLASSES:
                    raise Sj1Error(f"prior edit plane {key} is not a token plane")
                prior[int(key)] = plane

    indices = np.arange(args.shard_index, N_PAIRS, args.shard_count, dtype=np.int64)
    if args.targets is not None:
        allowed = set(np.load(args.targets).tolist())
        indices = np.array([p for p in indices if int(p) in allowed], dtype=np.int64)
    selectors = STAGE_SELECTORS[args.stage]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"rows_shard_{args.shard_index}.jsonl"
    planes_path = out_dir / f"planes_shard_{args.shard_index}.npz"

    done: set[int] = set()
    if rows_path.is_file() and args.resume:
        with rows_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    done.add(int(json.loads(line)["pair"]))
    planes: dict[str, np.ndarray] = {}
    if planes_path.is_file() and args.resume:
        with np.load(planes_path, allow_pickle=False) as blob:
            planes = {k: np.asarray(blob[k], dtype=np.uint8) for k in blob.files}

    # A row is appended for EVERY completed pair; the planes npz is only rewritten every
    # ``--checkpoint-every`` pairs.  A kill inside that window (the walltime cap, an OOM,
    # an operator cut) therefore leaves up to ``checkpoint_every - 1`` pairs per shard
    # with a ROW but no PLANE.  Resuming past them marks them done, never re-plans them,
    # and the merged field simply OMITS them -- and an omitted pair does not fall back to
    # the prior pass, it reverts all the way to the BASE field, because every downstream
    # consumer splices the npz onto ``BODY_TOKENS``.  That is the silent-revert genus this
    # arm has already been bitten by three times, here at the resume surface: the row
    # ledger would still read correct while the shipped field quietly lost every banked
    # edit for those pairs.  So the disagreement is REFUSED, not repaired by guesswork.
    if args.resume and done:
        orphaned = sorted(p for p in done if str(p) not in planes)
        if orphaned:
            raise Sj1Error(
                f"shard {args.shard_index} cannot resume: {len(orphaned)} pair(s) have a "
                f"row but no plane (first: {orphaned[:8]}). The planes npz is written "
                f"every {args.checkpoint_every} pairs while rows are written every pair, "
                "so a kill inside that window loses those planes; resuming would skip the "
                "pairs and the merged field would revert them to the BASE tokens, losing "
                "every previously banked edit for them. Delete BOTH "
                f"{rows_path} and {planes_path} and redo this shard."
            )

    started = time.time()
    todo = [int(p) for p in indices if int(p) not in done]
    completed = 0

    def on_pair_done(result: dict[str, Any]) -> None:
        nonlocal completed
        pair = int(result["pair"])
        plane = result.pop("plane", None)
        if plane is not None:
            planes[str(pair)] = plane
        elif pair in prior:
            # No move landed this pass; the prior pass's plane still ships.
            planes[str(pair)] = prior[pair]
        with rows_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        completed += 1
        if completed % args.checkpoint_every == 0 or completed == len(todo):
            np.savez_compressed(planes_path, **planes)
            elapsed = time.time() - started
            print(
                f"  shard {args.shard_index}: {completed}/{len(todo)} pairs "
                f"({elapsed / max(1, completed):.1f} s/pair)",
                flush=True,
            )

    run_shard_pass(
        body,
        todo,
        prior,
        selectors=selectors,
        batch=args.batch,
        max_evaluations_per_pair=args.max_evaluations_per_pair,
        on_pair_done=on_pair_done,
        progress=args.progress,
    )
    np.savez_compressed(planes_path, **planes)
    receipt = {
        "schema": "ddm_sj1_pass_shard.v2",
        "pass_index": args.pass_index,
        "stage": args.stage,
        "selectors": list(selectors),
        "batch": args.batch,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pairs": [int(p) for p in indices],
        "rows_path": str(rows_path),
        "planes_path": str(planes_path),
        "planes_sha256": _sha256_file(planes_path),
        "elapsed_seconds": time.time() - started,
        "receipts": body.receipts,
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
    }
    receipt_path = out_dir / f"PASS_SHARD_{args.shard_index}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(json.dumps({k: receipt[k] for k in ("shard_index", "elapsed_seconds")}))
    return 0


def cmd_pass_repair_shard(args) -> int:
    """Make a crashed shard resumable EXACTLY, by dropping only its orphaned rows.

    The refusal in ``cmd_pass`` says "delete both files and redo the shard", which is
    safe but throws away every pair the shard did finish.  It does not have to: a pair
    that HAS a plane has both artifacts on disk and they were written from the same
    ``result``, so the row/plane pair is consistent by construction.  The orphans are
    exactly the rows appended after the last plane checkpoint, and re-doing only those
    is an EXACT repair, not a guess.

    MEASURED cost of the difference, on the pass-4 crash this was written for: 182 rows
    against 170 planes, so 12 orphans.  Deleting and redoing costs 182 pairs; this costs
    12.

    Fail-closed on anything the truncation would not be exact for: a duplicate row for a
    pair (a prior resume already went wrong), or a plane with no row (a different
    inconsistency than the one this repairs).  The original rows file is never deleted --
    it is moved aside with a UTC suffix, per the keep-the-payload rule.
    """
    out_dir = Path(args.out_dir)
    rows_path = out_dir / f"rows_shard_{args.shard_index}.jsonl"
    planes_path = out_dir / f"planes_shard_{args.shard_index}.npz"
    for path in (rows_path, planes_path):
        if not path.is_file():
            raise Sj1Error(f"shard {args.shard_index} has no {path.name} to repair")

    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs = [int(r["pair"]) for r in rows]
    if len(set(pairs)) != len(pairs):
        dupes = sorted({p for p in pairs if pairs.count(p) > 1})
        raise Sj1Error(
            f"shard {args.shard_index} has duplicate rows for pairs {dupes[:8]}; the "
            "ledger would double-count flips_before at merge. This is not the orphan "
            "class this repairs -- inspect the shard by hand."
        )
    with np.load(planes_path, allow_pickle=False) as blob:
        planed = {int(k) for k in blob.files}
    if not planed <= set(pairs):
        stray = sorted(planed - set(pairs))
        raise Sj1Error(
            f"shard {args.shard_index} has planes for pairs with no row {stray[:8]}; "
            "that is the opposite inconsistency and this repair does not cover it."
        )

    keep = [r for r in rows if int(r["pair"]) in planed]
    dropped = [int(r["pair"]) for r in rows if int(r["pair"]) not in planed]
    verdict = {
        "schema": "ddm_sj1_pass_repair.v1",
        "shard_index": args.shard_index,
        "rows_before": len(rows),
        "planes": len(planed),
        "rows_after": len(keep),
        "orphaned_pairs_redone": dropped,
        "rows_sha256_before": _sha256_file(rows_path),
        "planes_sha256": _sha256_file(planes_path),
        "applied": bool(args.apply),
        "axis": "[bookkeeping only -- no measurement, no score]",
        "score_claim": False,
    }
    if args.apply and dropped:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = rows_path.with_suffix(f".jsonl.orphaned-{stamp}")
        rows_path.replace(backup)
        tmp = rows_path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in keep),
            encoding="utf-8",
        )
        tmp.replace(rows_path)
        verdict["rows_backup"] = str(backup)
        verdict["rows_sha256_after"] = _sha256_file(rows_path)
    (out_dir / f"REPAIR_SHARD_{args.shard_index}.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True)
    )
    print(json.dumps(verdict, sort_keys=True))
    return 0


def cmd_pass_merge(args) -> int:
    """Merge pass shards: the n600 ledger, the edited field, and the pass table row."""
    receipts = [json.loads(Path(p).read_text()) for p in args.receipts]
    covered = sorted(p for r in receipts for p in r["pairs"])
    if covered != list(range(N_PAIRS)):
        raise Sj1Error(
            f"pass shards cover {len(covered)} pairs, not the full n600 field; "
            "a sub-n600 pass yield is a TOY"
        )
    rows: list[dict[str, Any]] = []
    planes: dict[str, np.ndarray] = {}
    for r in receipts:
        with Path(r["rows_path"]).open("r", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
        with np.load(r["planes_path"], allow_pickle=False) as blob:
            for key in blob.files:
                planes[key] = np.asarray(blob[key], dtype=np.uint8)
    if len({row["pair"] for row in rows}) != N_PAIRS:
        raise Sj1Error("ledger rows do not cover the full n600 field")

    # Sister of the resume refusal above, at the surface where the damage would actually
    # ship.  The rows covering all 600 pairs is NOT proof the field does: a pair that is
    # merely ABSENT from the npz reverts to BODY_TOKENS downstream, so a complete-looking
    # ledger can sit on top of a field that silently dropped a banked pair.  Every pass
    # from pass 3 onward carries a prior field, so every pair must have a plane; the
    # escape exists only for a first pass with no prior, where an untouched pair
    # legitimately has none.
    missing = sorted(p for p in range(N_PAIRS) if str(p) not in planes)
    if missing and not args.allow_partial_planes:
        touched = {int(r["pair"]) for r in rows if int(r.get("tokens_changed", 0)) > 0}
        raise Sj1Error(
            f"{len(missing)} pair(s) have a ledger row but NO plane (first: "
            f"{missing[:8]}; {len(set(missing) & touched)} of them accepted moves). "
            "A pair absent from the merged npz reverts to the BASE tokens downstream, "
            "losing every banked edit for it while the ledger still reads complete. "
            "Pass --allow-partial-planes ONLY for a first pass run with no prior field."
        )

    tokens_changed_this_pass = sum(int(row.get("tokens_changed", 0)) for row in rows)
    base = jg1.load_tokens(BODY_TOKENS)
    tokens_changed = sum(
        int((plane != base[int(key)]).sum()) for key, plane in planes.items()
    )
    flips_before = sum(int(row["flips_before"]) for row in rows)
    flips_after = sum(int(row["flips_after"]) for row in rows)
    repaired = flips_before - flips_after
    accepted = sum(len(row.get("accepted", [])) for row in rows)
    realized = sum(int(row.get("proposals_realized", 0)) for row in rows)
    enumerated = sum(int(row.get("proposals_enumerated", 0)) for row in rows)
    evaluations = sum(int(row["evaluations"]) for row in rows)

    args.out_field.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_field, **planes)
    selector_table: dict[str, dict[str, int]] = {}
    for row in rows:
        for entry in row.get("accepted", []):
            key = f"{entry['offset'][0]},{entry['offset'][1]}|{entry['selector']}"
            # The engine records the REALIZED whole-pair repair as ``repaired``; the
            # old ``local_repaired`` name belonged to the windowed attribution the
            # influence probe falsified, and reading it here crashed the first merge
            # after every shard had finished (rc=1 on a complete pass).
            cell = selector_table.setdefault(key, {"accepted": 0, "repaired": 0})
            cell["accepted"] += 1
            cell["repaired"] += int(entry["repaired"])
    report = {
        "schema": "ddm_sj1_pass.v1",
        "pass_index": receipts[0]["pass_index"],
        "stage": receipts[0]["stage"],
        "selectors": receipts[0]["selectors"],
        "segnet_batch": receipts[0]["batch"],
        "pairs": N_PAIRS,
        "flips_before": flips_before,
        "flips_after": flips_after,
        "flips_repaired": repaired,
        "fraction_of_flips_repaired": repaired / flips_before if flips_before else 0.0,
        "d_seg_before": flips_before / (N_PAIRS * EVAL_H * EVAL_W),
        "d_seg_after": flips_after / (N_PAIRS * EVAL_H * EVAL_W),
        "proposals_enumerated": enumerated,
        "proposals_realized": realized,
        "moves_accepted": accepted,
        "tokens_changed_vs_base": tokens_changed,
        "tokens_changed_this_pass": tokens_changed_this_pass,
        "cells_per_changed_token_this_pass": (
            repaired / tokens_changed_this_pass if tokens_changed_this_pass else 0.0
        ),
        "realized_evaluations": evaluations,
        "pairs_edited": len(planes),
        "accepted_by_offset_and_selector": selector_table,
        "field_npz": str(args.out_field),
        "field_npz_sha256": _sha256_file(args.out_field),
        "seg_gain_bytes_equivalent": repaired * BYTES_PER_SEG_CELL,
        # THIS PASS's repairs over THIS PASS's tokens.  Dividing a per-pass numerator by
        # the CUMULATIVE token count (the bug this replaces) mixes two populations and
        # understated pass 3's break-even by 6.8x -- the same shape as the merge's earlier
        # `local_repaired` defect: a summary reading a field whose meaning had moved.
        "break_even_bits_per_changed_token_this_pass": (
            8.0 * BYTES_PER_SEG_CELL * repaired / tokens_changed_this_pass
            if tokens_changed_this_pass
            else 0.0
        ),
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"pass {report['pass_index']} stage={report['stage']}: "
        f"repaired {repaired}/{flips_before} ({report['fraction_of_flips_repaired']:.2%}) "
        f"with {tokens_changed_this_pass} changed tokens this pass "
        f"({report['cells_per_changed_token_this_pass']:.3f} cells/token, "
        f"break-even {report['break_even_bits_per_changed_token_this_pass']:.3f} bits/token); "
        f"d_seg {report['d_seg_before']:.8f} -> {report['d_seg_after']:.8f}"
    )
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--no-verify-shas", action="store_true")
    parser.add_argument("--progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    step0 = sub.add_parser("step0", help="one shard of the n600 baseline seg leg")
    _add_common(step0)
    step0.add_argument("--shard-index", type=int, required=True)
    step0.add_argument("--shard-count", type=int, required=True)
    step0.add_argument("--out", type=Path, required=True, help="argmax .npy for this shard")
    step0.add_argument("--receipt", type=Path, required=True)
    step0.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="decode to score (default: the cl2 body's own 0.raw); use a CANDIDATE decode "
        "to re-measure d_seg on the bytes that would actually ship",
    )
    step0.set_defaults(func=cmd_step0)

    merge = sub.add_parser("step0-merge", help="merge shards into the n600 seg leg")
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--out-argmax", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.set_defaults(func=cmd_step0_merge)

    control = sub.add_parser("control", help="forward-model byte identity control")
    _add_common(control)
    control.add_argument("--pairs", type=int, default=4)
    control.add_argument("--seed", type=int, default=20260905)
    control.add_argument("--out", type=Path, required=True)
    control.set_defaults(func=cmd_control)

    probe = sub.add_parser("probe", help="measure single-token influence radius + cost")
    _add_common(probe)
    probe.add_argument("--pairs", type=int, default=3)
    probe.add_argument("--sites", type=int, default=4)
    probe.add_argument("--seed", type=int, default=20260905)
    probe.add_argument("--out", type=Path, required=True)
    probe.set_defaults(func=cmd_probe)

    run = sub.add_parser("pass", help="one pass over a strided shard of all 600 pairs")
    _add_common(run)
    run.add_argument("--pass-index", type=int, required=True)
    run.add_argument("--stage", choices=sorted(STAGE_SELECTORS), default="full")
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, required=True)
    run.add_argument(
        "--batch",
        type=int,
        default=8,
        help="pairs held in flight so SegNet runs batched; the render stays batch 1",
    )
    run.add_argument(
        "--max-evaluations-per-pair",
        type=int,
        default=100_000,
        help="backstop only; the family exhausting is the intended stop",
    )
    run.add_argument(
        "--field", type=Path, default=None, help="npz of prior-pass token planes"
    )
    run.add_argument(
        "--targets", type=Path, default=None, help="npy of pair indices to restrict to"
    )
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--checkpoint-every", type=int, default=5)
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=cmd_pass)

    part = sub.add_parser(
        "partition", help="MEASURE the persistent partition (sites no single move repairs)"
    )
    _add_common(part)
    part.add_argument("--argmax", type=Path, required=True, help="realized argmax field .npy")
    part.add_argument("--field", type=Path, default=None, help="npz of edited planes")
    part.add_argument("--sites", type=int, default=300)
    part.add_argument("--seed", type=int, default=20260905)
    part.add_argument("--shard-index", type=int, default=0)
    part.add_argument("--shard-count", type=int, default=1)
    part.add_argument("--out", type=Path, required=True)
    part.set_defaults(func=cmd_partition)

    prep = sub.add_parser(
        "pass-repair-shard",
        help="drop a crashed shard's orphaned rows so --resume redoes exactly those",
    )
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--shard-index", type=int, required=True)
    prep.add_argument("--apply", action="store_true")
    prep.set_defaults(func=cmd_pass_repair_shard)

    pmerge = sub.add_parser("pass-merge", help="merge pass shards into the n600 ledger")
    pmerge.add_argument("--receipts", nargs="+", required=True)
    pmerge.add_argument("--out-field", type=Path, required=True)
    pmerge.add_argument("--out", type=Path, required=True)
    pmerge.add_argument(
        "--allow-partial-planes",
        action="store_true",
        help=(
            "permit a merged field that omits pairs. ONLY legal for a first pass with no "
            "prior field, where an untouched pair legitimately has no plane. With a prior "
            "field an omitted pair reverts to the BASE tokens and loses banked edits."
        ),
    )
    pmerge.set_defaults(func=cmd_pass_merge)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
