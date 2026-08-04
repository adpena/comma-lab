# ddm_gk2 — consumption-side governance-knob coverage sweep of the decode-read inventory

**Arm:** `ddm_gk2` (task #847, the unladdered-governance-knob / m51 finder) · **Date:** 2026-08-04
**Axis:** apparatus / RATE-finder. `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`.
**Own-vehicle frontier:** `S = 0.7910689 @ 353,805 B [macOS-CPU advisory]` — UNMOVED. This arm measures
no score; it enumerates rate/quality knobs that the NEXT scorer window can turn. MEANS, not the END.

## §0 — What this arm is (and what gk1/ca1 are not)

gk1 (`ddm_gk1_guarded_constant_20260803.md`) BUILT the `GuardedConstant` destination + the P6 gate,
**declaration-driven** — it fires only on constants already *declared* in the registry. gk1 §9.6 states
plainly that P6 **cannot** find #933 (the ±1.0 token range) because a value nobody declared is invisible
to it, and names the cure: *"a consumption-side coverage sweep — enumerate the values the receiver reads
at decode time and check each has a ladder rung."* That is this arm.

ca1 (`ddm_ca1_calibration_audit_20260803.md`) is the **producer-side** audit: "does each of the 463
scalar-returning derivations have a valid derivation / is it called in production?" ca1's verdict on the
live TR1 path is "good calibration shape." **This arm is orthogonal and finds a different class:** a value
can have a perfectly valid producer-side derivation *and still be an unladdered knob on the RATE axis* —
the worked example is `token_quant_levels=16` (§4), which ca1 correctly marks CLEAN ("pinned at the coder
ceiling") yet the #933 receipt measures **L=14 saves 23,655 B**. The producer asks *"is 16 a legal
value?"* (yes); the consumption/rate axis asks *"is maxing levels the rate optimum, or an unswept
tradeoff?"* (it is a tradeoff). Same constant, two axes, two verdicts — that is the value of this sweep.

## §1 — Scope: the actual decode-read tree (transitive from `inflate.sh`)

READ-ONLY. The live receiver is `experiments/inflate_runner_v4d.py` (staged as `inflate_runner.py`,
per `tac.submission_chain.audit_runtime_tree` repo_map). Its transitive import closure — the modules that
`inflate.sh` actually executes to reconstruct the archive — is exactly five files:

| module | role on the decode path |
|---|---|
| `experiments/inflate_runner_v4d.py` | top-level receiver; frame_0 warp + photometric + rolling-shutter |
| `src/tac/optimization/ddm_tr1_runtime.py` | token dequant + lotto renderer forward |
| `experiments/ddm_r7_token_coder.py` | R7 SMEVR token entropy coder (tr1-form token section) |
| `src/tac/optimization/ddm_ix2_archive_container.py` | ix2 single-member container + config-section decode (**live form**) |
| `src/tac/optimization/pfs1_warp_receiver.py` | intrinsics + `pose_to_homography` + `ST_GRID` |

`repair_entropy_coder_runtime_adapters.py` is transitively importable (r7's rANS branch) but its
`ANS_SCALE_BITS=12` (gk1 §6 #4) is **NOT read on the live path**: the shipped token codec is `smevr`
(pinned at `ddm_tr1_runtime.py:541`) in tr1 form and the ix2 bulk is coded by `code_block`'s
stored/deflate/brotli/lzma race — **no rANS ships**. So gk1's ANS_SCALE_BITS row is a dead decode-read
knob on this vehicle; recorded closed here.

## §2 — DENOMINATOR (the anti-vacuity number, m50)

**25 decode-read CONTROL values examined** across the 5 modules (a "control value" = a
constant/default/clamp/range/codebook/count/step the decode path READS to reconstruct the archive, and
that shapes RATE or QUALITY). Excluded as pure framing/structural and NOT counted in the 25: section
magics, `struct` widths, `VERSION` bytes, arithmetic-coder state constants (`_STATE_BITS=32`,
`_FULL_RANGE/_HALF/_QUARTER`), bounds/guards (`_MAX_VALUES`, `_MAX_DECIMAL_SCALE`, uint16 caps,
`levels∈[2,16]` validity), contest-fixed geometry (`SEG_H/W=384/512`, `CAMERA_H/W=874/1164`), context-model
structure (age-bucket cap 3), and the generic 0-byte section order/names.

**Of the 25: 8 are UNLADDERED governance knobs; 17 are laddered** (raced / derived-from-law /
counted-per-clip). Classification uses the CLAUDE.md value-provenance ladder rungs:
{derived-from-a-law · raced · measured-anchor · typed-waiver · counted-per-clip-payload}. A value with
NONE of those, that a program reads at decode and that controls rate/quality, is an unladdered knob.

## §3 — POSITIVE CONTROL: instrument WORKS

Both #933 controls surface in the 8:

- **±1.0 token range** — `ddm_tr1_runtime.py:480` (`np.clip(combined, -1, 1)`, encode) and the decode-side
  dequant `ddm_tr1_runtime.py:1226-1228` (`x01*2 - 1`). MEASURED (brief/#933): **33.3% of token mass is
  pinned against this clamp.** Surfaced as unladdered knob #3 below.
- **L (`token_quant_levels`), default 16** — `experiments/ddm_r7_token_coder.py:1424`,
  `ddm_tr1_runtime.py:482`/`:1214`. MEASURED (brief/#933): **L=14 saves 23,655 B** with pre-registered
  break-evens. Surfaced as unladdered knob #1 below.

Had the sweep missed either, the verdict would be INSTRUMENT-BROKEN. It did not; the instrument reads the
decode-read inventory, not a declaration list.

## §4 — The ranked table of the 8 unladdered knobs

Gap denominator (DERIVED): own-vehicle `S=0.7910689` − PR130 floor `0.172141` = **gap 0.6189279**;
`S/byte = 25/37_545_489 = 6.659e-7`, so **1% of gap ≈ 9,295 B** and **0.006189 S**. (This calibration
reproduces the brief's "L=14 = 2.54% of gap" exactly: 23,655 B × 6.659e-7 / 0.6189279 = 2.545%.)

Ranked by (estimated gap-fraction × cheapness-to-sweep), **UNBANKED residual only**:

| # | knob | site | rung today | lever | est. gap-fraction | evidence |
|---|---|---|---|---|---|---|
| 1 | **L = `token_quant_levels`, default 16** | r7:1424; tr1:482,1214; ca1 row 14 | UNLADDERED (pinned at coder ceiling, not rate-derived) | rate×quality | **2.54% (MEASURED, −23,655 B at L=14)** | #933 positive control; refines ca1 row 14 |
| 2 | **`window_solve` default OFF** | tr1:1382-1421 | UNLADDERED (tracked-queue default-off) | quality (d_seg) | **2.33% (MEASURED −0.01441 S, 0 bytes)** | tr1 docstring; gated by d_pose-under-v4d-warp |
| 3 | **±1.0 token range/clamp** | tr1:480,481,1226-1228 | UNLADDERED (hardcoded generic literal, no derivation) | quality / rate-shaping | INFERRED material (33.3% mass pinned) | #933 positive control; impact needs scorer |
| 4 | **`ST_GRID` s_t knot placement (11 hand-set knots)** | pfs1:18 | UNLADDERED support/spacing (subset IS raced — ca1 row 7) | rate (index entropy) + quality | INFERRED (m84 names "granularity re-race" an owed rate mover) | refines ca1 row 7 (race is over a FIXED support) |
| 5 | **`_LZMA_FILTERS` lc=3/lp=0/pb=0** | ix2:116 | UNLADDERED (borrowed PR101 lineage, m21/L24) | rate | INFERRED small (lzma competes in code_block race; may not win the bulk) | never re-raced within the lzma candidate |
| 6 | **`DEFAULT_BETA_MAGS` granularity `(0.0,0.5,1.0)`** | v4d:94 | UNLADDERED menu spacing (VALUES counted when fitted, ix2:498) | quality (d_seg) | INFERRED (−0.0135 S already BANKED as the 3-pt menu; residual = finer/optimal spacing) | v4d docstring; m52 (3-pt = coarse continuum) |
| 7 | **token codec pinned `"smevr"` (tr1 form)** | tr1:541 vs r7 9-codec menu + unused `auto` racer | UNLADDERED-on-tr1 (ix2 form DOES race via code_block) | rate | INFERRED small (ix2 is live; tr1-form only) | `auto` exists but the live encode hardcodes smevr |
| 8 | **`_RESCALE_AT = 1<<15`** | r7:94 | UNLADDERED coder constant | rate | INFERRED small (smevr/kt-scoped = tr1 form) | KT adaptive-rescale threshold, never swept |

Runner-up excluded from the 8 as **semi-laddered**: KT `counts+=2` (alpha=1/2, r7:253) is a named KT
estimator init, not a free tuning knob.

The 17 laddered decode-reads (recorded, not knobs): `code_block` coder race (raced, incl. stored so
"already at entropy" is representable), `encode_exact_table` f16<f32<f64<scaled-int ladder (raced-smallest-
exact), brotli `quality=11`/`lgwin=24` + zlib `9` (max = L32-derived), bicubic `A=-0.75` (derived, matches
the PyTorch scorer resize), geometric horizon `v=437` (derived from intrinsics), EON intrinsics
`910/582/437` + `CAMERA_HEIGHT_M=1.22` (hardware-derived, documented comma2k19/openpilot source), and the
per-clip counted selector payload (`mask_density`, `lotto_seed`, `renderer_width`, `code_width`,
`grid_downsample`, `token_ste`+`dither_seed`, `dim0_offset`, per-pair `sel`/`beta_idx`/`s_t`-index,
fitted `beta_mags` VALUES, fitted `st_grid` VALUES) — laddered-as-counted; their VALUES are legitimately
paid-for video-derived payload (§ix2 `classify_against_vendored` proves per-archive), while their
MENUS/GRANULARITY (rows 4, 6) are the unladdered residual surfaced above.

## §5 — Reconciliation with ca1 (honest, not a contradiction)

- **ca1 row 14 (`token_quant_levels=16` = CLEAN, "pinned at the coder ceiling").** Correct as a
  producer-side fact: 16 IS `_R7_SMEVR_MAX_LEVELS`, the coder's admissible max. My verdict is the RATE
  axis: "pin at the ceiling" silently assumes more levels is free; the #933 receipt (L=14 = −23,655 B)
  measures it is not. The two verdicts are compatible — ca1 asks legality, I ask rate-optimality — and
  together they say: 16 is legal, unswept, and rate-suboptimal. This is the exact m52 shape (a max-it-out
  default is a UI over a rate×quality continuum).
- **ca1 row 7 (`_EXHAUSTIVE_CAP=400000`, s_t codebook race certified over a 21-point support).** Confirms
  the s_t SUBSET selection is race-certified (laddered). My row 4 is the complement: the 21-point SUPPORT
  and the shipped 11-knot PLACEMENT `{0, 0.005, …, 0.24}` are hand-set and never re-raced — m84's owed
  "granularity re-race." ca1's own note ("`C(23,11)` breaches the cap") is evidence the support is a
  design choice, not a derivation.

## §6 — Fire / fold / queue (every follow-on with a fire-order)

All top-5 verifications need the scorer that sibling `ddm_bz1` currently owns (mutating the receiver /
submission_chain), so fire-order = **next scorer window**, and this arm does NOT edit the receiver.

1. **QUEUED — fire next scorer window:** L-sweep {14,15,16} through byte-close on the live pu2 vehicle;
   the L=14 −23,655 B is a producer number, the JOINT rate+d_seg through the real decode is the owed
   measurement (task ledger row below).
2. **QUEUED — fire next scorer window:** `window_solve=ON` d_pose-under-v4d-warp gate (the only blocker
   named in the tr1 docstring); −0.01441 S d_seg is already MEASURED.
3. **QUEUED:** ±1.0 range re-fit (widen vs narrow) — 0 counted bytes, needs scorer for d_seg.
4. **QUEUED:** ST_GRID support+knot re-race (m84 owed mover) — extends ca1's fixed-support race to the
   support itself; assert `comb(len(support), k) <= _EXHAUSTIVE_CAP` first (ca1 row 7's cheap owed guard).
5. **QUEUED:** re-race `_LZMA_FILTERS` lc/lp/pb within the lzma candidate (small, cheap).
6. **FOLDED (no wiring edit by this arm):** rows 1, 3, 4, 5 are the correct **destinations** for
   `GuardedConstant` once measured — `role=count/scale` (L), `role=scale`+token-units (±1.0),
   `role=scale` (ST_GRID), `role=scale`+bits (lzma). Per gk1 §9.6 "the class holds it; the gate keeps it
   held; neither finds it" — this sweep is the finder; wiring is a gk1-module landing, not a receiver edit.

## §7 — Boundaries / owed / not done

- **No score measured.** Every impact except L=14 (−23,655 B, producer-side) and window_solve
  (−0.01441 S d_seg) is INFERRED and needs the scorer window. The L=14 byte number is producer-side; its
  JOINT rate+d_seg through the live byte-close is UNMEASURED here.
- **Scope is the 5-module live decode tree only.** A different vehicle (tr1-manifest form vs ix2-container
  form) changes which of rows 7/8 are live; I flagged both. I did NOT sweep the encoder/trainer side
  (that is the producer axis ca1 owns).
- **Denominator is a judgement call at the framing/knob boundary.** I excluded arithmetic-coder algorithm
  constants and contest-fixed geometry as structural; a stricter reader could contest e.g. the age-bucket
  cap 3. The 8 unladdered knobs are robust to that boundary; the "25 examined" is the softer number.
- **I did not edit any receiver / `tac.submission_chain` / carrier-compose file** (bz1 owns those).
- Pointer UNMOVED. Own-vehicle frontier `S = 0.7910689 @ 353,805 B [macOS-CPU advisory]`.
