---
title: "the phase-field row does NOT byte-close — but the blocker is the SEG CARRIAGE (a realization mirage), not the pose tail; the frame_0 pose repair passed both gates and is the proven, general, THIRD pose-carrying route"
unit: ddm_bz1
task: phase-field byte-close driver (convert js1's staging law into a real n600 evaluate.py row OR an honest verdict)
date_utc: 2026-08-04
axis: "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE"
score_claim: false
promotion_eligible: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
own_vehicle_frontier: "S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED by this unit"
verdict_scope_default: FORMULATION
---

# ddm_bz1 — the byte-close attempt that found the seg leg was never legally realizable

## §0 ANSWER FIRST

I was sent to byte-close js1's projected **−0.01769 S** phase-field row (seg offset field + k=4
frame_0 pose repair) into the first own-vehicle frontier move since pu2, or return an honest
tail-breaks-the-budget verdict. **Neither fork is the one that fired.** The pose gates passed; the
row died on a leg neither js1 nor et1 had measured.

| gate / question | measured | verdict |
|---|---|---|
| **G1** int16 quantiser re-score of the k=4 repair | max degradation **9.28e-6** (naive int16, no overflow, cmax ≤332) | **PASS** — naive suffices; adaptive/QAT unneeded |
| **G2** k=4 repair on the ABSOLUTE-MASS damage tail | pooled repaired/shipped **0.9464×** over 17 pairs incl. 471@84.78×→0.96×, 261@123.85×→0.997× | **PASS** — 96 B/pair holds, no pair needs k>4, all seg-exact |
| **the SEG carriage** (unmeasured by et1/js1) | legal deterministic RGB-translation η **≈0.11** (scorer) / **−0.05** (camera); needs **>0.426** to bank | **FAILS by 4×** |
| composed row, honest legal realization | net **+0.057 S** (LOSES); even with a FREE pose stream **+0.0187 S** | **does NOT byte-close** |

**The single sentence.** The frame_0 k=4 pose repair is real, clean, general, and within budget —
it undoes seg-edit pose damage from **1.06× to 123.85×** back to **≤ shipped** at **96 B/pair** with
int16 quantisation costing nothing — **but the phase-field seg base it was meant to rescue cannot be
carried at all**: et1's seg gain was measured with a **frozen-SegNet-guided paint** (η 0.5267) that
is **illegal at inflate** (no scorer) and whose pixel-shipping alternative is the gp1 band et1
already killed (331 KB); the only **legal deterministic** realization of the priced offset field
(translate the rendered frame_1 RGB by the offsets) recovers **η ≈ 0.11**, so the seg leg costs
**more rate than it buys** and the composed row is net-POSITIVE.

**Fork verdict: (b) — but NOT the routed successor.** This is not "tail breaks the budget → #366
joint descent." The pose axis is healthy; **pose is not the blocker**. The blocker is the seg
carriage, and the correct route is the one the coordinator elevated mid-unit: the frame_0 pose
repair is a **proven general mechanism** that must compose with a **REAL byte-closed seg base**
(#827 ep854, whose seg carriage IS realized because it byte-closed), not with the phase-field
mirage. Fire-order-1 below.

## §1 What I refute — in my charter, in js1's projection, and in the inherited seg pricing

1. **My charter's byte-ledger premise is WRONG (corrected, MEASURED).** The charter hypothesised
   "the phase field only REACHES ~251 pairs … the pose stream is ~96 B × 251 ≈ 24 KB." It is not:
   et1's **reach 41.84% is a FLIP-FIX fraction, not a pair count**. I measured all 600 pairs'
   block16 paint band: **600 of 600 have a nonzero band** (median 553 px), so **every pair is
   pose-damaged and needs a repair**. The pose stream is **57,600 B (all 600)**, exactly js1's
   number, NOT 24 KB. `verdict_scope: MEASURED` (band coverage n600).

2. **js1's −0.01769 projection is a seg-realization mirage, and it is mine to correct.** js1's
   composed table used `seg leg −0.086840 = n600 gross × η 0.4814`, taking et1's **SegNet-paint η**
   as the *realized* seg gain. That η is **not legally realizable at inflate** (§3). Under the legal
   deterministic realization (η ≈ 0.11) the seg leg is **−0.0198**, not −0.087. Honest composed net:
   **+0.057 S** (LZMA1 offset) — the row **loses**. m66/qd1 exactly: a ΔS whose *object* was never
   the shipped object is unanchored.

3. **The offset field's honest byte figure is the MEASURED LZMA1 57,809 B, not 46,247 B.** js1's
   table priced the offset field at **46,247 B**, which is et1's **SMEVR PROJECTION** (0.80× of the
   measured LZMA1 57,809, a coder ph1 measured elsewhere and never ran here). For a real byte-close
   the honest number is the coder we can actually run: **57,809 B LZMA1** → rate **0.038493 S**
   (brotli-q11 on the ph1 field was within ~1.5%, not decisive). The SMEVR 46,247 is admissible only
   if SMEVR is actually implemented and installs in the contest runtime.

4. **et1/ph1 measured a seg gain on the WRONG object.** ph1 §6 already flagged its gross as "an
   argmax-field result, not yet a realized one … an UPPER BOUND on the carrier." et1 then measured η
   with `solve_margin_optimal_paint` — a **frozen-SegNet-guided** solve. Both are upper bounds; the
   legal deterministic realization was **the next measurement neither did**, and it is §3.

## §2 The pose side — both gates pass, decisively (this is the good news)

Harness: `experiments/ddm_bz1_tail_and_quantiser.py`, a faithful copy of js1's k=4 solve that
additionally captures the coefficients at the best iterate and re-scores the int16-quantised
reconstruction from the real camera pair. **Faithfulness control (anti-drift):** on pair 48 the
FLOAT arm reproduces js1's `d_pose_verified_from_camera` **exactly** (0.3149×), so the copy did not
drift and its int16 arm is trustworthy.

**Targeting.** js1's k=4 ladder covered only MODERATE pairs. I targeted the **absolute-mass tail**
(et1 n=32): the ratio tail (123.8× pair 261) is NOT the mass tail — **pair 471 (84.78×) carries the
largest absolute damage +0.01742**, then 261 (+0.00938), 365, 433, 394. NONE had a k=4 repair.

### §2a G1 — int16 quantiser (naive form passes; steer #1/#2 forms unneeded)

Per operator steer #1 the ladder was naive → adaptive → quantization-aware; per "use the cheapest
that passes," **the naive int16 form passed**: max quantiser degradation **9.28e-6** (vs d_pose
~1e-3), **no overflow** (cmax 158–332 ≪ int16 32767), **seg-exact on every pair**. *(Deconflict: js1
resumed and is independently paying G1 the **quantization-aware** way — best-iterate on the quantised
synthesis, unbreakable-by-construction — on its ladder pairs. My naive-int16 arm is the complementary
measurement: it covers the **extreme absolute-mass tail (471, 261, 394)** js1's ladder did not, and
its negligible degradation is the belt to js1's QA suspenders. I did NOT duplicate js1's ladder pairs'
quantiser work.)* The DCT basis is
orthonormal, so int16 rounding error is ~0.008 RMS/pixel — an order below uint8 rounding. Adaptive
depth and quantization-aware solve remain **available and unneeded here** (they are a *rate*
optimization for the general mechanism, §5, not a G1 fix).

### §2b G2 — the k=4 repair holds pose ≤ shipped across the full damage spectrum

17 pairs measured (my 8 int16 + js1's 10 float, deduped), pooled ABSOLUTE d_pose (the mass that
moves population pose), stratified-HARD so this is an **UPPER BOUND** on population damage:

| slice | measured |
|---|---|
| extreme tail | 471 (84.78×)→**0.961×**, 261 (123.85×)→**0.997×**, 394 (21.11×)→1.042× |
| moderate | 48 (3.65×)→**0.315×**, 288/365/433 (3.8–5.1×)→1.01–1.07× |
| low | 20/179/196 (1.0–1.2×)→0.85–0.94× |
| **pooled repaired/shipped** | **0.9464×** (sum 0.01348 / 0.01424) — **≤ 1.0** |

**G2 PASS.** The frame_0 repair undoes the seg-edit pose damage back to **at or below shipped** even
on the 84×/124× pairs, at **96 B/pair** (< the 157.7 B/pair budget), no pair needing k>4. Since the
sample is hard-selected and the population is dominated by low-damage pairs, the population ratio is
**closer to 1.0** — the repair is pose-**neutral-to-slightly-improving** at scale. The per-pair
non-monotonicity (48→0.31× but 433→1.07×) is the cap-pinned-floor / under-conditioned-solver
signature js1 named; every ratio here is a **FLOOR**, improvable, not an optimum.

## §3 The SEG carriage — the linchpin, and the decisive negative

**The question.** What does the receiver DO at inflate to turn the 57,809 B offset field into a seg
gain? SegNet reads only the last frame, so the offset field must move frame_1. Two legal options:

| option | realizer | seg η | rate | verdict |
|---|---|---|---|---|
| (A) ship offset field | translate rendered frame_1 RGB blocks by offsets (deterministic, no scorer) | **≈0.11** (MEASURED) | 57,809 B | seg gain 0.0198 < rate 0.0385 — **loses** |
| (B) ship painted pixels | frozen-SegNet paint (et1's η) | 0.5267 | **331,824 B** (the gp1 band) | et1 already killed it — **DEAD** |
| (illegal) | SegNet AT inflate | 0.5267 | 57,809 B | **forbidden** (no scorer weights) |

et1's η=0.5267 is option (B)'s realizer priced at option (A)'s rate — the inconsistency that made
the row look bankable. **Measured (`experiments/ddm_bz1_deterministic_seg_realize.py`, n=8):** the
legal deterministic RGB-block translation of the label-solved offsets recovers **η mean 0.11**
(scorer-lattice) / **−0.051** (camera-resolution) — **NEGATIVE on 2–5 of 8 pairs** (translation
*increases* flips), because SegNet is **not translation-equivariant**: translating RGB blocks and
re-argmaxing does not reproduce the label-field translation the offsets were solved for.

**The arithmetic gap is 4×.** To bank, seg gain must exceed offset+pose rate: `η × 0.18039 >
0.0385 + 0.0384` → **η > 0.426**. Measured legal η ≈ 0.11. Even a FREE pose stream needs η > 0.213;
even SMEVR offset + free pose needs η > 0.171. All exceed the measured legal η by 2–4×.

**`verdict_scope: MECHANISM`** (deterministic RGB-block translation of label-solved offsets). Per
the operator's realization doctrine a disappointing realizer is MECHANISM-scoped, not a family kill;
untested legal realizers remain (§6 NOT-QUEUED-with-reason). But the CURRENT phase-field row **as
specified by et1/js1 does not byte-close**, by a margin no measured legal realizer closes.

## §4 The byte-ledger — ACTUAL section sizes (not js1's projection)

| leg | js1 projection | **honest (this unit)** | why |
|---|---|---|---|
| seg offset field | 46,247 B (SMEVR proj) | **57,809 B** (LZMA1 MEASURED) | SMEVR was never run; LZMA1 is the coder we have |
| pose k=4 stream | 57,600 B (all 600) | **57,600 B** (CONFIRMED all 600) | charter's 24 KB reach-set premise refuted (§1.1) |
| seg realized gain | η 0.4814 × gross | **η 0.11 × gross** | legal deterministic realization, not SegNet-paint |
| **composed net** | **−0.017692 S** | **+0.057 S (LOSES)** | seg leg is a mirage |

Denominators (re-confirmed from receipts): gap **0.6189279** = 0.7910689 − 0.172141 · block16 gross
**0.18039 S** (508,640→295,845 flips, n600 label ceiling) · S_per_flip 8.477e-7 · rate_per_byte
6.6586e-7 · break-even η (rate/gross) at LZMA1+pose = **0.426**.

## §5 The reusable, base-agnostic section — built + tested (coordinator ask #1)

`src/tac/optimization/frame0_pose_repair_stream.py` (+ 6 tests, all pass): the frame_0 pose repair
as a **swappable, self-describing COUNTED section**, independent of any seg base. It holds
`encode/decode` (int16 coefficients, LZMA1 or stored — chooses the smaller, never over-counts),
`apply_pose_repair_scorer` (byte-identical to `round(clamp(base0 + coef@A))` — asserted in tests),
`dct_atoms` (generic → FREE, rule 118), and `section_ledger` for
`tac.submission_chain.build_byte_ledger` to close against. The **seg base is a separate, swappable
section** the chain composes ahead of it — so ep854 / #827 / a future field drops in without
re-plumbing the pose stream. I deliberately did **not** build the phase-field seg section (it is the
mirage of §3) nor wire the ix2/TR1 grammar + inflate runtime (that needs a real base to test
end-to-end; §6 fire-order-1).

**Depth-adaptivity (operator steer #2), scoped honestly.** Bit-depth is an assumed constant;
per-coefficient depth (int16 DC + int8 mid + drop tail, reusing the L29 fp16-per-group-scale
pattern) plausibly puts k=4 at ~48–72 B/pair and could bring adaptive-depth k=8 under the 157.7
budget for pairs k=4 under-repairs. **But it does not rescue the phase-field row**: even a **free**
pose stream leaves the composed net at **+0.0187 S** (the seg leg is the killer). Depth-adaptivity
is therefore a **rate win for the general mechanism / #827 route**, not a phase-field fix — folded
to §6, not fired speculatively here.

## §6 Follow-ons — FIRED / FOLDED / QUEUED-WITH-FIRE-ORDER

- **FIRED** — G1 int16 re-score (§2a) · G2 absolute-mass tail (§2b) · the deterministic-seg-realize
  linchpin that killed the row (§3) · the byte-ledger corrections (§1.1, §1.3, §4) · the correction
  to js1's projection (§1.2) · the reusable base-agnostic section + tests (§5).
- **QUEUED, fire order 1 — apply the frame_0 k=4 pose repair to the #827 ep854 seg base.** This is
  the coordinator's "single highest-value next measurement" and, given §3, it is now the PRIMARY
  route: ep854 is byte-closed (its seg carriage is REAL) and pose-blind (`pose_stub` INERT, 83 B).
  Decode ep854's frame_0/frame_1 (`ddm_ep2_20260731/archives/w03_ep854_representative/archive.zip`,
  sha 37ba7a96…), measure its per-pair pose, and run the frame_0 k=4 repair; if it fits ≤157.7
  B/pair (likely — the mechanism held to 124× damage here) it composes a real net-negative row via
  the §5 section + `tac.submission_chain`. **Reconciliation respected:** #881's "DO NOT RE-FIRE the
  ep854 SEG re-solve" bans SEG re-solving, not a frame_0 POSE repair on the existing base.
- **QUEUED, fire order 2 — build depth-adaptive (per-coefficient int8/int16, per-pair k) carriage**
  for the §5 section, priced with the real ledger, in service of fire-order-1 (steer #2). Fire
  condition: fire-order-1 shows the pose repair fits but the stream wants trimming.
- **NOT QUEUED (seg-carriage family escape hatches, named, low-probability given the 4× gap)** —
  (a) re-solve offsets to maximise the **RGB-translation** flip reduction (not the label-field
  translation) — brute force is infeasible (121 offsets × 768 blocks × SegNet), a greedy version is
  possible but faces SegNet non-equivariance + the negative-η pairs; (b) token-level translation +
  re-render (ph1's intended receiver, unbuilt); (c) offset + small residual hybrid (the residual is
  the gp1 band, ~331 KB). All must clear η > 0.426 (4× the measured legal 0.11) to matter.
- **NOT ROUTED — #366 joint descent.** The charter's fork (b) routed a tail failure to joint pose
  descent. Pose did NOT fail (G1/G2 pass); routing pose work here would treat a healthy axis as the
  blocker. The seg carriage is the blocker, and its fix is a real seg base (#827), not joint pose.

## §7 Self-caught defects (mine, not inherited)

1. **My charter's reach-set byte premise (24 KB) was wrong** — reach is a flip-fraction, not a pair
   count; measured band coverage 600/600 (§1.1). Caught by measuring, not trusting the charter.
2. **A spurious macOS-Accelerate matmul FPE** ("divide by zero in matmul", finite inputs, output
   byte-identical to the float reference) tripped my warnings-are-blocking rule; silenced locally
   with `np.errstate` + a comment, after confirming the output is correct.
3. **I nearly reported the row as byte-closeable pending only the pose gates** — the seg carriage was
   the unmeasured leg the whole plan rested on. Caught by reading ph1 §6's own "upper bound" caveat
   and measuring the legal receiver instead of inheriting et1's paint η.

## §8 STATE-THE-BOUNDARIES — what I did NOT measure

- **No archive was built and no byte was closed. The pointer did not move.** No n600
  `upstream/evaluate.py` was run this unit (the seg leg failed the MVP gate, so the expensive n600
  was correctly not spent).
- **The legal deterministic seg-η is n=8, single realizer variant** (two lattices). I did not
  brute-force the RGB-objective offset re-solve, token translation, or a hybrid — named §6.
- **G2's 0.9464× is a subset (17 pairs), not the population d_pose** — it is an upper bound; the true
  population pose after repair needs the n600 realization, which is moot while the seg leg is dead.
- **The #827/ep854 frame_0-repair budget is UNMEASURED** — the mechanism held to 124× here, but
  ep854's pose-damage magnitude and its k requirement are its own measurement (fire-order-1).
- **The §5 section is tested in isolation, not wired into the ix2/TR1 grammar or inflate runtime** —
  that end-to-end wiring belongs with fire-order-1's real base.

## §9 Generality — the frame_0 pose canvas is the THIRD pose-carrying route (coordinator ask #2)

`upstream/modules.py`: `x = x[:, -1, ...]` — SegNet sees only frame_1, so **frame_0 is structurally
seg-invisible** and any additive correction there costs **zero seg flips by construction** (measured:
seg-exact on all 17 pairs under a ±40-LSB control + the repair). This makes frame_0 a **free pose
canvas**, a pose-carrying route **distinct from**:
- **bo1's dead frame_1 post-hoc family** (m09/#881) — which fights the seg field it lives on;
- **joint descent** (#366) — which pays a training tax.

**Scope, honest:** DERIVED (from the SegNet input structure) + **MEASURED n=17** across a damage
spectrum of **1.06×–123.85×**, k=4 / 96 B/pair / int16, on the pu2 vehicle. Each new base
(ep854/#827) needs its **own** measurement — its pose-damage magnitude may exceed the ≤157.7 B/pair
budget and force a higher k or depth-adaptive carriage (§5). The mechanism is proven; its *price on
a given base* is per-base.

## §10 Receipts + STORES CONSULTED

Scripts (this unit): `experiments/ddm_bz1_tail_and_quantiser.py` ·
`experiments/ddm_bz1_deterministic_seg_realize.py` · `src/tac/optimization/frame0_pose_repair_stream.py`
(+ `tests/test_frame0_pose_repair_stream.py`, 6 pass).
Receipts: `/Volumes/VertigoDataTier/pact/ddm_bz1_20260804/` — `bz1_tail_g1g2.json` (8 pairs, G1+G2,
int16) · `bz1_det_seg_realize.json` (n=8 deterministic seg-η).
Controls (measured): FLOAT-arm reproduces js1's 0.3149× on pair 48 (anti-drift) · yuv6 equivalence
max_abs_error 0.0 · gradient live · seg-exact per pair · int16 no overflow.
Stores consulted: `ddm_{js1,et1,ph1,pu2,cr1,gp1,sq1,ph5o}` memos · `upstream/modules.py`
(`x[:,-1]`; PoseNet two-frame yuv6) · `src/tac/submission_chain.py` (`build_byte_ledger`, PROFILES) ·
`src/tac/optimization/ddm_ix2_archive_container.py` (`build_payload`/`parse_payload`) ·
`ddm_ph1_20260803/offsets_n600_rmax5.npz` · CLAUDE.md authority ladder + m66/m86/m87/m96/m09.

## §11 Pointer honesty

**The exact pointer did NOT move.** `0.1910828242 [contest-CPU]` UNMOVED. Own-vehicle frontier
**S = 0.7910689 @ 353,805 B [macOS-CPU advisory]** UNMOVED. No archive was built; no byte was closed.
A confirmed pose mechanism, a killed seg carriage, and a corrected projection are **MEANS**. **This
unit has not achieved the goal** — but it correctly refused to spend an n600 on a row whose seg leg
was a realization mirage, and it hands the successor the proven pose mechanism aimed at a real base.

S = 0.7910689 @ 353,805 B [macOS-CPU advisory] — UNMOVED
