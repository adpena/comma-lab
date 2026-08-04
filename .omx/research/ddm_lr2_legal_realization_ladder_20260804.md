---
title: "the legal realization ladder for the phase-field seg gain: every offset-carrying rung LOSES against its own bar — the offset field itself is the dead leg; the live descendant is per-block SOLVED paint with no offset field at all"
unit: ddm_lr2
task: "operator binding 2026-08-04: 'There are legal and more optimal ways of realizing those gains' — measure the ladder of better legal realizers, each against ITS OWN recomputed bar"
date_utc: 2026-08-04
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED by this unit"
verdict_scope_default: "per-rung INSTANCE; family verdicts only where all named rungs are measured"
---

# ddm_lr2 — the legal realization ladder, measured rung by rung

## §0 ANSWER FIRST

bz1 measured ONE legal realizer (naive camera-RGB block translation, η ≈ 0.119) and correctly
scoped its negative to the instance. This unit measured the LADDER: five better legal realizers
of the same block16 offset field on the same pairs, plus the encode-side-solved rungs the m95
existence proof licenses. The finding is sharper than "the naive realizer was weak":

**Every rung that SHIPS the offset field loses against its own bar — not because realization is
weak, but because the offset field's 57,809 B buys transport that the frozen SegNet refuses to
honor (η ≤ 0.12 for every deterministic transport realizer measured), while the rungs whose η IS
high (solved paint, η 0.5–1.6) don't need the offsets at all.** The phase-field seg gain is real;
the offset CARRIER of it is the dead leg. The best measured cell of the whole ladder is the
no-offset, STATIC-addressed descendant (the sg3 composition, MEASURED not projected):
**C0-keys M32 addr=static — one 64 B block list shared by all 600 pairs + 96 B/pair solved
shifts + the pose stream: η 0.3999 vs its own bar 0.4246 → net +0.0045 S.** Still LOSES, but
**12× closer than the naive realizer's +0.0553**, on a 30-step cap-pinned solver FLOOR, with
collateral **0.62% med / 1.00% max — below sg3's 1.035% crossover on 8/8 pairs at M32** — and
three named, unmeasured levers each plausibly worth more than the remaining 0.0045 (§6
fire-order-1). cg1's encode/decode key-gap unknown is CLOSED with numbers (§5): the STATIC
GT-derived key retains 80% of the per-pair GT key's η; the 0-byte decoder-derivable proxy
retains only 41%.

| rung (all n=8, bz1's pairs, whole-frame η) | legal η (scorer path) | d_pose ratio (subset med) | carrier n600 | own bar η | net ΔS | verdict |
|---|---:|---:|---:|---:|---:|---|
| bz1 naive camera translate (inherited) | 0.1192 | — | 115,409 B | 0.4260 | +0.0553 | LOSES |
| **A1** token-resample + re-render | **−0.3802** (8/8 neg) | 69 | 115,409 B | 0.4260 | +0.1454 | LOSES |
| **A2** pre-R native translate | **+0.1200** | 379 | 115,409 B | 0.4260 | +0.0552 | LOSES |
| **A3-tok** response-solved token offsets | **−0.3725** | 183 | 119,430 B | 0.4408 | +0.1467 | LOSES |
| **A3-rgb** response-solved RGB offsets | **+0.1103** | 583 | 118,679 B | 0.4381 | +0.0591 | LOSES |
| **C** transport + solved block shifts M16 | +0.3332 | 453 | 167,684 B | 0.6190 | +0.0515 | LOSES |
| **C** transport + solved block shifts M32 | +0.5646 | 544 | 206,909 B | 0.7638 | +0.0359 | LOSES |
| **C** transport + solved shifts ALL(~78) | +0.7949 | ~470 | 302,534 B | 1.1167 | +0.0581 | LOSES |
| **B** transport + sparse residual K32–K256 | 0.209–0.527 | ~500 | 234–852 KB | 0.86–3.14 | +0.118–0.472 | LOSES (selector INSTRUMENT-limited) |
| **B** dense-band solved (unpayable ceiling) | **1.6050** | ~750 | — | — | — | ceiling only |
| **C0** NO-offset block paint M8 | 0.1303 | 4.8 | 88,050 B | 0.3250 | +0.0351 | LOSES |
| **C0** NO-offset block paint M16 | 0.2284 | 5.6 | 109,350 B | 0.4036 | +0.0316 | LOSES |
| **C0** NO-offset block paint **M32** | **0.4447** | 33.7 | **148,200 B** | **0.5470** | **+0.0185** | LOSES — best cell of the ladder |
| **C0** NO-offset block paint M64 | 0.6194 | 72.1 | 216,075 B | 0.7976 | +0.0321 | LOSES |
| **C0-null** rank-6-projected DC shifts (all M) | **0.0000 exactly** | 1.0 | 17–69 KB | 0.06–0.25 | +rate | **VACUOUS BY STRUCTURE (§4)** |
| **C0-keys** M32 addr=gt (per-pair indices) | 0.4991 | 23.7 | 154,275 B | 0.5695 | +0.0127 | LOSES |
| **C0-keys** M32 **addr=static** (64 B once) | **0.3999** | 17.4 | **115,039 B** | **0.4246** | **+0.0045** | LOSES — closest cell of the ladder |
| **C0-keys** M32 addr=proxy (0 B, decoder) | 0.2070 | 7.9 | 116,775 B | 0.4310 | +0.0404 | LOSES — the key gap, measured |

Denominators (re-verified at source): gross(n600) = **0.1803885565863715 S** (et1
`resolved_on_our_field.gross_S`) · offsets LZMA1 n600 = **57,809 B** (same receipt) · pose stream
= **57,600 B** (96 B/pair k=4, bz1 G1/G2) · S_per_flip 8.477105e-7 · rate_per_byte 6.658590e-7 ·
live best **0.7910689 @ 353,805 B** (pu2 `chain_receipt.json`, sha `c72ef357…`) · gap 0.6189279 ·
subset flips-representativeness **0.9745** (mean 826.1 vs population 847.7; the pairs are the
first 8 of sq1's stratified 32 — near-representative on seg, NEVER extrapolated on pose, m96).

## §1 What I refute — in my charter and in the inherited framing

1. **Charter hypothesis "the scorer-in-loop-trained renderer transports partition structure
   where raw-RGB translation cannot" is REFUTED, in the strongest direction.** RUNG A1 (bilinear
   token-field resample by (dy/16, dx/16) per cell — the token grid IS the block16 grid, 24×32,
   `grid_downsample=16` — then re-render through the shipped LOTTO renderer) measures pooled
   η = **−0.3802, negative on 8/8 pairs**, three times WORSE than doing nothing. The TR1
   renderer is not locally shift-equivariant: sub-cell token interpolation produces content the
   frozen head argmaxes differently everywhere. Control: the render path reproduces the shipped
   frame_1 **bit-exactly on 8/8 pairs** before any edit, so this is the realizer, not the
   harness. `verdict_scope: MECHANISM` (sub-cell token-space transport on this renderer).
2. **bz1's "double-resample" was never the naive realizer's problem.** A2 translates the
   renderer's NATIVE 384×512 float output pre-uint8/pre-bicubic (the #149 pre-R placement, no
   camera round-trip) and lands η = **0.1200** — per-pair within 0.041 of bz1's camera-lattice
   naive on every pair. Three transport realizers on three different lattices now agree at
   η ≈ 0.11–0.12: the blocker is SegNet's non-equivariance, not resampling loss. The instance
   negatives consolidate to `verdict_scope: MECHANISM` (deterministic per-block translation of
   ANY rendered representation of this content).
3. **The "re-solve offsets against the realized response" escape hatch (bz1 §6a) is now
   MEASURED, and it fails for an interesting reason.** RUNG A3 renders all 121 global shifts
   (rmax 5) on BOTH substrates (token-warp and native-RGB), selects PER BLOCK the offset whose
   REALIZED argmax best matches GT (zero-seeded ties), then re-renders the composite. The
   per-block selection bound is η ≈ **1.09 pooled on both substrates** — block-by-block, a
   shift exists that fixes the flips. The realized composite collapses to **−0.3725 (tok)** /
   **+0.1103 (rgb)**: **inter-block interference eats 90% of the bound.** Non-equivariance
   manifests as coupling: each block's fix breaks its neighbors through the head's receptive
   fields. Response-solved selection also buys nothing on rate (offsets price 5–7% HIGHER than
   label-solved: 1399/1382 vs 1308 LZMA1 on the subset). `verdict_scope: FORMULATION`
   (independent per-block offset selection, both substrates, this field).
4. **The inherited bar 0.426 was correct for bz1's rung but is NOT the family's bar.** Each rung
   here carries its own recomputed bar (table above); the C rungs' bars are 0.62–1.12 because
   their carriers grow, and the C0 bars fall to 0.11–0.41 because theirs shrink. Reusing 0.426
   anywhere else would have produced wrong verdicts in both directions.

## §2 The decisive structural finding — transport is dead, SOLVE is alive, and they don't compose

Two measured facts, side by side:

- Every DETERMINISTIC TRANSPORT of the offset field (camera-RGB, native-RGB, token-space,
  response-re-solved) realizes η ≤ 0.12 — far under every bar.
- Every ENCODE-SIDE SOLVE against the frozen head realizes η 0.5–1.6 on the same pairs — the
  m95 content-vs-solve law again, now on this field: **C ALL-blocks pooled η 0.7949** (per-pair
  0.567–1.292, 8/8 above the naive realizer's 0.119); **dense-band B reaches pooled 1.605**
  (fixing MORE than the described flips — whole-frame accounting credits neighborhood repair).

And the composition fact that kills the family as specified: **the solve does not need the
transport.** Exactly matched at M=32, same solver, same pairs: C-with-transport η 0.5646 vs
C0-direct (keys, gt key) η 0.4991 — the transport+offsets add **0.0655 of η for 57,809 B**,
marginal value 0.0118 S against a rate cost of 0.0385 S: **the offset field is net-negative
even as a solve PRECONDITIONER (−0.027 S marginal), not only as a stand-alone carrier.**
`verdict_scope: MEASURED` (matched A/B, n=8).

**RUNG B's knee is INSTRUMENT-limited, and B is dominated anyway.** Sparsifying the dense solve
by |delta| magnitude retains almost nothing (K64 pooled η 0.277 at ~360 B/pair — vs C's block
params at η 0.565 for 152 B/pair). At every measured byte level, block-granular params beat
pixel-granular residuals — the region-not-pixel law at the carrier level (pc2/m95: the
OBJECTIVE is regional, the ADDRESS should be supra-pixel; per-pixel is right only as the
actuator). A realized-influence-ranked sparsifier could move B's knee; `verdict_scope:
INSTRUMENT` on the |delta| selector, and B stays dominated by C at equal rate in every measured
cell, so the better sparsifier is not queued.

## §3 RUNG C0 — the no-offset descendant, the ladder's best cell

Base = the SHIPPED TR1 render (verified bit-exact against the archive decode). Encode-side:
select top-M blocks by per-block flips (block INDICES are counted payload, 2 B each), solve one
RGB shift per block (3 int8) against the frozen head — Adam/CE, realized-argmax best-iterate
(the m95 solve form) — receiver applies the shifts at the 4 private camera px of each scorer px
(m86; additive, texture-preserving). Carrier per pair ≈ 5 B/block, LZMA1-measured.

| arm | pooled η | per-pair range | B/pair | collateral med/max | d_pose med (abs, ratio) |
|---|---:|---|---:|---|---|
| M8 | 0.1303 | −0.06…0.59 | 51 | 0.962% / 1.727% | 0.0036, 4.8× |
| M16 | 0.2284 | 0.00…0.60 | 86 | 0.837% / 1.787% | 0.0049, 5.6× |
| **M32** | **0.4447** | 0.16…0.91 | **151** | **0.618% / 1.002%** | 0.0335, 33.7× |
| M64 | 0.6194 | 0.37…1.16 | 264 | 0.447% / 0.581% | 0.0520, 72.1× |

- The (η, bytes) knee the charter asked for is REAL and lands between M32 and M64: marginal η
  per byte falls from 3.0e-3/B (M16→M32) to 1.5e-3/B (M32→M64); every arm still loses because
  the **pose stream (57,600 B) is now the LARGEST single carrier line** — at M8 it is 65% of
  the carrier. The seg axis has stopped being the rate problem; the pose axis's carriage is.
- Instrument note: these solve0 M-levels were realized as nested subsets of one 64-block solve
  (ranked by |shift|·flips). The keys run solved M=32 DIRECTLY and reached 0.4991 vs the nested
  0.4447 — the nested realization UNDER-measures by ~11%; the keys numbers are the honest M32.
- Fixed-vs-introduced decomposition (pooled): M32 fixes 1,898 and introduces 697 — the
  introduced flips are the collateral sg3's crossover prices, and they sit BELOW the 1.035%
  line on 8/8 pairs at M32 (and on all pairs at M64).
- C0's pose damage at M8/M16 (med 4.8–5.6×, abs ~0.004) is INSIDE the range the k=4 frame_0
  repair has already repaired (js1/bz1, up to 123.85×); M32's max (284×) exceeds the proven
  range — the per-base repair gate is fire-order-2, cheap (the solve/repair harness exists).

## §4 Pose — the dual pair, reported per rung, never composed silently

Per the steer, every rung row carries (η_seg, d_pose ratio) as a DUAL PAIR (subset-scoped,
median; pose is 4.6× skewed vs seg on subsets, so no population ΔS is ever formed from these).
Measured facts:

- Every frame_1 rung damages pose heavily: subset medians 69× (A1) to 583× (A3-rgb) of pu2's
  shipped d_pose (absolute 0.02–0.86 on these pairs). The bars above already carry the
  57,600 B frame_0 repair stream — but **the k=4 repair's capacity was proven on damages
  ≤123.85× (bz1 G2); several rungs here sit above that measured range, so the repair's
  adequacy on a winning rung would be its own per-base gate (bz1 §9), fire-order-1 below.**
- **The pose-null subspace is AC-ONLY — a law this unit derived, verified, and then measured
  as a perfect vacuity.** Every per-block-constant RGB shift lies ENTIRELY in the rank-6
  projector's row space: `‖P·(c,c,c,c)‖ ≤ 1e-6` for arbitrary c (numeric check on sq1's exact
  projector), because any constant (dR,dG,dB) decomposes as α·K_Y + (β,0,γ), all three in the
  constraint row space. Consequence, confirmed on all 8 pairs × 4 M-levels: the projected C0
  actuator does literally NOTHING (η 0.000000, fixed 0, introduced 0, d_pose ratio 1.000000 —
  the m50 vacuity signature, here EXPECTED and diagnostic). Two corollaries: (a) **DC color
  shifts — the cheapest argmax movers — are 100% pose-visible; pose-neutral paint must carry
  within-block TEXTURE (AC) structure**, which explains et1 §7's measured η↔d_pose coupling
  mechanistically; (b) any future "pose-free cheap paint" proposal parameterized as per-region
  constants can be rejected on paper. `verdict_scope: DERIVED+MEASURED` (exact, this
  projector).

## §5 sg3 harvest — the crossover quantity, measured

sg3 (landed mid-unit) prices a STATIC risk-map address at **4,266 B total** (22.28% of pixels
ever flip; ceiling 27.90% of flips = 0.12028 S) and routes it to this rung as receiver context,
with the decisive crossover: **static addressing beats exact addressing iff the realizer's
collateral flip rate on already-correct pixels < 1.035%.**

**Measured (C0, `collateral_rate_in_region` on every arm): M32 = 0.618% median, 1.002% max —
BELOW the crossover on 8/8 pairs; M64 = 0.447%/0.581%.** The solved-paint realizer QUALIFIES
for static addressing — and the composition was then MEASURED, not projected (coordinator
fold-in, cg1's named unknown), as a three-way address-key A/B at matched M=32, same solver:

| address key | η pooled | capture of pair flips | address bytes | carrier n600 | bar | net ΔS |
|---|---:|---:|---:|---:|---:|---:|
| GT per-pair top-32 (encode key) | **0.4991** | 0.64 | 64 B/pair | 154,275 B | 0.5695 | +0.0127 |
| **STATIC top-32 by ever-flip mass** | **0.3999** | 0.39 | **64 B ONCE** | **115,039 B** | 0.4246 | **+0.0045** |
| decoder-derived edge-energy (0 B) | 0.2070 | 0.10 | 0 B | 116,775 B | 0.4310 | +0.0404 |

- **cg1's question is closed with numbers: there is no free decoder-derivable substitute for
  the GT risk key at this grain (41% of GT-key η, 0.10 capture), but the STATIC GT-derived key
  is nearly one (80% of GT-key η at ~1/600th the address rate).** The static block list is even
  cheaper than sg3's px-granular map for this use (64 B vs 4,266 B); the static set itself
  reproduces sg3's receipt exactly (43,798 px — cross-receipt control PASS).
- Static addressing loses only 20% of η while cutting capture 0.64→0.39: the solve fixes flips
  OUTSIDE its blocks (SegNet's regional response works FOR the static key).
- sg3's floors KILL, never confirm (mirage law); these realized η's are the confirming half.
  The static cell's remaining gap to zero is +0.0045 S on a 30-step solver FLOOR.

## §6 Follow-ons — FIRED / FOLDED / QUEUED-WITH-FIRE-ORDER

- **FIRED (this unit)** — the five charter rungs (A1 · A2 · A3-tok · A3-rgb · C M16/M32/ALL ·
  B K32–K256 + dense ceiling) · the no-offset descendant C0 at 4 M-levels · the pose-null
  vacuity derivation + 8-pair confirmation · the sg3 collateral crossover measurement · the
  3-way address-key A/B (coordinator fold-in; cg1's unknown closed) · the matched
  transport-as-preconditioner A/B · every rung's own recomputed bar.
- **QUEUED, fire-order-1 — close the +0.0045 S static-cell gap, then byte-close.** Three named
  levers, each measured elsewhere as live: (a) solver budget — every η here is a 30-step
  cap-pinned FLOOR (et1 §7 measured η still rising at 50 steps; sweep budget×M on the STATIC
  key, pick by net including pose); (b) M sweep on the static key (the static list stays 2 B/
  block ONCE — M=48 costs +32 B total address, +48 B/pair params); (c) params entropy — int8
  triplets LZMA'd at ~96 B/pair barely compress; a fitted small codebook (L29 pattern) or
  int6 depth could save 20–40% of the params line. Fire condition: any (η, bytes) cell with
  net < −0.005 S on n=8 → extend to n=32 stratified → byte-close ONLY through
  `tac.submission_chain` (canonical, never a probe script) with the frame_0 k=4 pose stream
  composed via `frame0_pose_repair_stream` (§5 of bz1).
- **QUEUED, fire-order-2 — the per-base pose-repair gate for the winning cell.** The static-key
  M32 damage (med 17.4×, abs ~0.01–0.03) sits at the edge of the k=4 repair's proven range
  (≤123.85×, repaired to ≤ shipped); run the bz1 G1/G2 harness on 3–4 of THESE edited pairs
  before any byte-close. Fire condition: fire-order-1 finds a banking cell.
- **QUEUED, fire-order-3 — pose-neutral paint via AC parameterization.** The §4 law says DC
  is 100% pose-visible; a per-block 3-param WITHIN-BLOCK pattern basis (fixed deterministic
  AC atoms through the rank-6 projector) could carry seg flips with near-zero pose damage and
  drop the 57,600 B pose stream — the carrier line that now dominates every C0 bar. This is a
  design+measure unit (the atoms must move argmax through P), cheap at n=4 first.
- **NOT QUEUED, with reasons** — B's influence-ranked sparsifier (B is dominated by C at every
  measured equal-rate cell); token-space actuation (A1/A3-tok: MECHANISM-dead on this
  renderer); offset-field re-solve variants beyond A3 (the field is net-negative even as a
  preconditioner, §2); truth-paint anywhere (sq1's −3.76, reconfirmed by nothing here needing
  it).

## §7 Self-caught defects (mine, not inherited)

1. My first transport smoke printed A1 pose ratios casually as "dpx 277" without absolute
   values; the memo reports both and never population-extrapolates (m96 discipline).
2. The B sparsifier's |delta| ranking was the cheap choice and it is the reason B's knee is
   unresolved — flagged as INSTRUMENT, not as a family fact, before any B verdict was typed.
3. The A3 candidate sweep initially priced response offsets per-pair summed (98,100 B ×75
   projection); corrected to ratio-pricing against the label field's joint-n600 LZMA1 (the
   honest comparable), which is what the table carries.

## §8 STATE-THE-BOUNDARIES

- **No archive was built, no byte was closed, no n600 scorer job was run** (fz1 owns the slot;
  all measurements are bounded n=8 stratified samples matched to bz1's pairs, labelled).
- The pooled η's are n=8 whole-frame; the pairs are flips-representative (0.9745) but the
  subset-to-n600 projection of PAYLOAD bytes (×75) and η×gross are projections, labelled.
- d_pose numbers are subset-scoped gates, never population ΔS.
- The k=4 pose repair was NOT re-measured (charter: settled); its capacity at the damage
  magnitudes these rungs produce (up to 1350× subset ratios) is beyond its measured range —
  a per-base gate for any winning rung, queued, not assumed.
- A1/A3-tok negatives are scoped to THIS renderer (tr1_lotto_combined_ema_v1) and sub-cell
  bilinear token resampling; a renderer trained WITH warp augmentation is a different vehicle.
- B's knee is unresolved (selector INSTRUMENT-limited); B is dominated by C at every measured
  equal-rate cell, which is the operative fact.
- Every solved η is a 30-step cap-pinned FLOOR (#874: response noted, cap not raised-and-
  quoted); η and pose damage are coupled (et1 §7), so budget is a joint lever, never free.
- The static key was built from the SAME vehicle's n600 caches the pairs are drawn from; its
  generalization is within-vehicle by construction (fine for this counted, this-clip payload).
- The keys arms' params pricing assumes the receiver orders params by the shipped/derived
  ranking (no per-pair indices for static/proxy); the static arm additionally assumes ONE
  64 B block list — both are grammar facts a byte-close must realize, not physics risks.

## §9 Receipts + STORES CONSULTED

Scripts (this unit): `experiments/ddm_lr2_realization_ladder.py` (transport / response / solve /
solve0 subcommands) · `experiments/ddm_lr2_aggregate.py`.
Receipts: `/Volumes/VertigoDataTier/pact/ddm_lr2_20260804/` — `lr2_transport_n8.json` ·
`lr2_response_n8.json` · `lr2_solve_n8.json` · `lr2_solve0_n8.json` · `lr2_keys_n8.json` ·
`lr2_ladder_table.json` · logs.
Controls (measured): TR1 render == shipped frame_1 **bit-exact 8/8** (A0) · A2 reproduces bz1's
naive per-pair within 0.041 (realizer-equivalence anti-drift) · the reconstructed static risk
set reproduces sg3's receipt EXACTLY (43,798 px) · the projector vacuity is confirmed both by
derivation (`‖P·const‖ ≤ 1e-6`) and by 32 independent zero measurements · zero-seeded
tie-breaks on every offset solve (et1 entropy discipline) · denominators re-derived from
et1/pu2 receipts, not quoted from memos.
Stores consulted: bz1/et1/js1/sq1/ph1/gp1/pu2/g4/sg3 memos + receipts · m95/m96/m88/m86/m87 ·
`ddm_pu2_20260803/submission_pu2/{ddm_tr1_runtime.py,inflate_runner.py}` (the shipped receiver,
imported not re-implemented) · `ddm_sq1_pose_null_constrained_paint.py` (rank-6 projector) ·
CLAUDE.md authority ladder + realization mirage law.

## §10 Pointer honesty

**The exact pointer did NOT move.** `0.1910828242 [contest-CPU]` UNMOVED. Own-vehicle frontier
**S = 0.7910689 @ 353,805 B [macOS-CPU advisory]** UNMOVED. A measured ladder, a mechanism-level
consolidation of the transport negatives, and a re-priced live descendant are MEANS. This unit
has not achieved the goal.

S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED
