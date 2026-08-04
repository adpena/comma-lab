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
shifts + the pose stream.** FO-1 (fired in-unit, §11) then uncapped the solver: the cell
**BANKED on n=8 (η 0.4406 vs bar 0.4260, net −0.0026)** and **REVERSED at the pre-registered
n=32 gate (η 0.3990 vs bar 0.4282, net +0.0053 — LOSES)** — the m88/m96 subset trap caught
before any byte-close was attempted. Final family state: **no measured cell banks at n=32**;
the closest legal cell sits 1.2% of gross from zero, per-pair η spans 0.24–1.09 (uniform-M
waterfill headroom named), collateral qualifies static addressing on 29/32 pairs even at
sg3's corrected 0.6285% crossover, and cg1's key-gap unknown is CLOSED with numbers (§5):
static GT-derived key = 80% of per-pair-GT η; 0-byte decoder proxy = 41%.

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
- **FIRED IN-UNIT via §11 (superseding the original fire-orders 1 and 3 below):** the uncapped
  budget×M sweep + n=32 gate (§11.1–11.2), the params-entropy trim (does not pay), the AC
  pose-null arm at 1 and 4 atoms (pose-neutral confirmed; seg-weak), the collateral spatial
  profile (§11.3), the triality registration (§11.4).
- **QUEUED, fire-order-1 (revised) — per-pair WATERFILL of block budget on the static key.**
  Per-pair η spans 0.24–1.09 at uniform M=32; allocate blocks per pair by measured response
  (ja1/QA73 atlas+waterfill machinery), per-pair M index ~1 B/pair. The n=32 receipts already
  hold per-pair (η, bytes, curves) — the allocator fits for $0 with no new scorer forward.
  Fire condition: fitted allocation projects net < −0.005 S at n=32 → byte-close ONLY through
  `tac.submission_chain` + `frame0_pose_repair_stream` (coordinate with fz1's chain wiring;
  the F0PR1/seg-base PROFILES entries remain owed there; n600 waits behind fz1's census).
- **QUEUED, fire-order-2 — the per-base pose-repair gate, now REQUIRED not precautionary.**
  n=32 damage max **1261.9×** exceeds the k=4 repair's proven ≤123.85× envelope; run the bz1
  G1/G2 harness on the worst 3–4 edited pairs of any banking variant before byte-close.
- **QUEUED, fire-order-3 (revised) — the band family's price-matched legal realizer.** Measure
  class-anchor paint (sq1 L1 form, zero value bytes) on the Road↔Lane r=1 band against sg3's
  81,365 B address price; the solved-value ceiling (η 1.14–1.34, §11.3) says the family is
  worth one honest measurement; the mirage law says the ceiling is not the row.
- **QUEUED, fire-order-4 — feature-matched AC atoms.** The pose-null basis fails through
  cosine/luma atoms (η ~0.1); atoms fit to the frozen head's features (curvelet/learned,
  through P) are the one unexhausted pose-null escape. Design+measure, n=4 first.
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
- **FO-1 phase (§11):** the deepest sample is n=32 (stratified, repr 0.9973); no byte-close
  was attempted — correctly, since no cell banks at n=32. The §11.3 four-arm race is PARTIAL
  (n=2 of 8 at landing; the run continues checkpointed into `lr2_tx_n8.json` pair by pair —
  the successor pools the full receipt). The TX negative is scoped to cosine/luma atoms; the
  BAND numbers are VALUE-ORACLE-scoped (the mirage law, applied to my own arm). The n=32
  pose-damage max (1261.9×) makes the per-base repair gate REQUIRED for any banking variant.

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

## §11 FO-1 ADDENDUM (coordinator-fired, same unit) — the uncapped sweep, the n=8→n=32 reversal, and the four-arm race

**Charter:** fire my own §6 fire-order-1 — uncap the solver (convergence-tested, curve shipped),
params-entropy trim, fold FO-3's AC lever in as sweep arms, best cell at n=32; plus the
operator's feature-bearing-paint steer and sg3's corrected crossover + band-address arm.

### §11.1 The uncapped sweep (n=8, static key, U vs AC, M ∈ {32, 64})

Convergence discipline: Adam with realized-flip patience early-stop (never a step bound as the
stop); per-cell curves in the receipts. Outcome: most cells CONVERGE well before the 150-step
safety bound (6/8 M32_U, 8/8 M64 and AC cells; the 2 cap-hits have near-flat tails, shipped).

| cell (n=8) | pooled η | dpx med | carrier | bar | net ΔS |
|---|---:|---:|---:|---:|---:|
| **M32_U int8** (+pose stream) | **0.4406** | 32.5 | 115,414 B | 0.4260 | **−0.00263 (banked on n=8)** |
| M32_U step2 / step4 | 0.4258 / 0.4139 | ~33 | 115.3/113.7 KB | 0.426/0.420 | −0.00005 / +0.00103 |
| M64_U int8 | 0.5487 | 53.4 | 159,053 B | 0.5871 | +0.00693 |
| M32_AC int8 (NO pose stream) | 0.1159 | **1.0** | 59,164 B | 0.2184 | +0.01849 |
| M64_AC int8 (NO pose stream) | 0.1862 | **1.0** | 110,153 B | 0.4066 | +0.03975 |

- Uncapping moved M32_U pooled η 0.3999 → 0.4406 (+0.041) — the cap-artifact was real and it
  crossed zero ON n=8. Params-entropy trim (step2/step4) does NOT pay: LZMA already absorbs
  the low-entropy tail; int8 is the best depth everywhere.
- The AC arm realizes the law's constructive half PERFECTLY — d_pose ratio 1.000–1.003 on
  every pair, stream-free carrier — but its seg power through the single content atom is ~4×
  too weak for its own (much lower) bar. `verdict_scope: FORMULATION` (this 1-atom AC basis).

### §11.2 The n=32 verdict — the banking cell REVERSES, and that is the finding

Best cell (M32_U int8) extended to the full stratified 32 (repr 0.9973, m88/m96-compliant):

| n | pooled η | carrier | bar | net ΔS | verdict |
|---:|---:|---:|---:|---:|---|
| 8 | 0.4406 | 115,414 B | 0.4260 | −0.00263 | banked |
| **32** | **0.3990** | 115,995 B | 0.4282 | **+0.00526** | **LOSES** |

**The n=8 window was optimistic for this realizer and the pre-registered n=32 gate caught it
before a single byte-close was attempted** — the exact failure m88/m96 names, working as
designed. No cell banks at n=32. Supporting facts at n=32: collateral med 0.420%, max 1.007%
(0/32 above the superseded 1.035% crossover; **3/32 above sg3's corrected 0.6285%**
(`2583e0f155`) — the realizer still qualifies for static addressing on 29/32 pairs);
d_pose ratio med 30.3× but **max 1261.9×** — one extension pair exceeds the k=4 repair's
proven ≤123.85× envelope, so the per-base repair gate (fire-order-2) is REQUIRED, not
precautionary, for any future banking variant. Per-pair η spans 0.24–1.09: the uniform-M
allocation is leaving large per-pair headroom unspent — the named successor lever is a
per-pair WATERFILL of block budget by measured response (ja1/QA73 machinery), not more
uniform budget.

### §11.3 The four-arm race (operator steer: feature-bearing paint; sg3: band address) — PARTIAL (n=2 of 8; run checkpointed per pair, receipts fill as it completes)

Arms at matched accounting, same pairs, each vs its OWN bar (`lr2_tx_*.json`):

| arm (pairs 0, 20 so far) | η | dpx | collateral | params/pair | reading |
|---|---|---|---|---|---|
| U_flat M32 (re-solve control) | 0.331 / 0.242 | 93.8 / 3.0 | 0.15–0.28% | ~96 B | matches §11.1 |
| **TX feature-bearing** (4 AC atoms through P, M16) | 0.116 / 0.096 | **1.00–1.01** | 0.05–0.13% | ~200 B (incompressible) | pose-neutral CONFIRMED again; seg-weak |
| TX M8 nested | 0.085 / 0.059 | 1.00 | 0.05–0.10% | ~107 B | under U at matched bytes |
| **BAND Road↔Lane r=1** (solved per-px paint) | **1.338 / 1.137** | 52.4 / 2.1 | 0.43–0.62% | — | **VALUE-ORACLE — see below** |

- **The feature-bearing steer's first measurement is NEGATIVE at this basis size (PROVISIONAL,
  n=2):** enriching the pose-null basis from 1 content atom to 4 (luma pattern + 3 low-order
  DCT, all through P) lifts η only 0.116 vs 0.44 for flat DC at HALF the params rate. The
  pose-null subspace's seg-power through per-block low-order atoms remains ~4× too weak for
  its bar. The race is honest at matched bytes: TX M8 (~107 B/pair, stream-free, bar ≈0.32)
  η 0.06–0.09; U M32 (~96 B/pair + stream, bar 0.428) η 0.24–0.33. **Flat-DC + pose stream
  still dominates AC + no stream at every measured cell.** The unexhausted escape: atoms
  matched to the head's features (learned/curvelet AC atoms), not cosine/luma ones.
- **The collateral spatial profile REFUTES the flat-paint-discontinuity mechanism at this
  granularity:** introduced flips within 1 px of the edit-region boundary are only 2–10% of
  collateral (U arm; 0–20% TX); 41–68% sit BEYOND 4 px. Collateral is far-field regional
  response (SegNet-sees-regions), not boundary-discontinuity dust. So texture will not cure
  collateral by smoothing patch seams — its value must come through feature alignment.
- **The BAND arm's big numbers (η 1.14–1.34, capture 0.40–0.46) are VALUE-ORACLE-scoped
  against sg3's 81,365 B price, by the mirage law applied to my own arm:** the solve is
  encode-side legal, but the 81,365 B carrier holds band POSITIONS + target labels — not the
  solved VALUES my realizer painted. The price-matched legal realizer (class-anchor paint from
  the stored labels, sq1's L1 form, zero extra value bytes) is the arm whose η actually prices
  that carrier, and it is UNMEASURED — the single named next measurement of the band family.
  Priced-value alternative: ~3 B × ~4,050 band px/pair ≈ 12 KB/pair — dead. Both crossover
  operating points reported: U/TX collateral is below even the corrected 0.6285%
  (`2583e0f155`) on these pairs; BAND sits at 0.43–0.62% — at the corrected line.

### §11.4 Triality debt — DISCHARGED

`pose_null_subspace_is_ac_only_v1` registered in `tac.canonical_equations`
(`src/tac/canonical_equations/pose_null_subspace_is_ac_only_20260804.py`, 7 tests pass,
ledger row appended via the canonical registry helper; commit of record in this unit's chain).
Evaluator `dc_projection_residual()` verifies ‖P·(c⊗1₄)‖∞ = 0 (measured 2.9e-13); helper
`ac_energy_fraction()` prices the pose-visibility of any candidate paint pattern; the
EmpiricalAnchor is the 32/32 vacuous-DC receipt (residual 0.0). Consumers declared: the FO-3/
AC arm, m85's integer-actuator caveat, the burn-spec pose-price arithmetic.

## §12 QUEUE-HEAD ADDENDUM (coordinator-fired) — the waterfill fit and the band's legal value-realizer

### §12.1 The band anchor-paint measurement — the legal value-realizer is DEAD at n=32

The price-matched legal realizer for sg3's 81,365 B Road↔Lane band (class-anchor paint: anchors
computed ENCODE-side from the decoder's own field, SHIPPED as a 15 B counted table; the receiver
paints each band px's 4 private camera px with its STORED target label's anchor — zero scorer
weights at decode, zero oracle values):

| variant (n=32 stratified, no solve, 4 s/pair) | pooled survival (bar 0.3956) | neg pairs | dpx med/max |
|---|---:|---:|---|
| **AP_pair** flat anchor paint (+pose stream) | **−1.0190** | **32/32** | 371× / 10,476× |
| **AP_null** projected, stream-free, pose-neutral | +0.0501 | 2/32 | **1.0× / 1.1×** |

**The band route is CLOSED at sg3's price point.** The legal CONTENT realizer roughly doubles
the flips it touches (the 5th independent no-go of content substitution in thin bands — sq1's
truth-paint −3.76, my §11.3 far-field collateral law, now anchors too); the pose-neutral
projection of it is near-inert (+0.05 vs the 0.3956 bar). The family's realization headroom
lives entirely in SOLVED values (the §11.3 oracle ceiling 1.14–1.34), and solved values at
band-px granularity cost ~12 KB/pair — which is precisely why the cheap solved-value carrier
is the per-block params family (C0), closing the circle: **the band route and the block-param
route are the same family at different value granularities, and the block-param end is the
live one.** `verdict_scope: FORMULATION` (content-anchor values on label-addressed thin bands,
this vehicle, n=32).

### §12.2 The waterfill-M fit ($0, from the n=32 receipts) — the first allocation that projects negative THROUGH a split-sample control

Per-pair ON/OFF allocation of the M32_U static cell (ON iff the pair's measured gain covers
its OWN marginal bytes: params_lzma1 + 96 B pose; mask 75 B + block list 64 B shipped once):

| fit | ON pairs | projected n600 net |
|---|---:|---:|
| in-sample (selection-biased, labelled) | 13/32 | **−0.0081 S** |
| **split-sample** (rule fit on half, evaluated on held-out half; break-even 152 flips) | 7/16 | **−0.0058 S** |

The uniform-M32 cell lost by +0.0053; per-pair selection flips the sign and SURVIVES the
split-sample control — the first legal, fully-priced allocation in this family to project
negative out-of-sample. **This is a PROJECTION, not a row**: the pre-registered confirming
measurement is the FIXED rule (ON iff gain > 152 flips at M32_U-static-int8) applied to FRESH
stratified pairs, then byte-close via `tac.submission_chain` + `frame0_pose_repair_stream`.
Pose gate input (per-pair damage on the 13 ON pairs): median ~23×, **one pair (485) at 1261.9×
(d_pose 0.765)** — outside the proven k=4 envelope; the pose-safe variant drops it for
−2.1e-5 S of margin, leaving 12 ON pairs all within the proven ≤123.85× repair range.

**Revised queue after this addendum:** fire-order-1 = the fresh-pair confirmation of the FIXED
waterfill rule (pose-safe variant) → byte-close path as §6; fire-order-2 = the per-base k=4
repair on the 12 ON pairs' edited frames (all within proven range); the band family and
feature-matched AC atoms drop BEHIND those (the band route closed at §12.1; AC atoms remain
the stream-free escape).

## §10 Pointer honesty

**The exact pointer did NOT move.** `0.1910828242 [contest-CPU]` UNMOVED. Own-vehicle frontier
**S = 0.7910689 @ 353,805 B [macOS-CPU advisory]** UNMOVED. A measured ladder, a mechanism-level
consolidation of the transport negatives, and a re-priced live descendant are MEANS. This unit
has not achieved the goal.

S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED
