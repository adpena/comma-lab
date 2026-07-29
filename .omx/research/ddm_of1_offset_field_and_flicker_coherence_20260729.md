---
title: "ddm_of1 — the two gc6 $0 coherence probes: P2C-OF boundary-offset-field + W1-COH flicker-phase"
date_utc: "2026-07-29T18:40:00Z"
lane_id: "lane_ddm_of1_coherence_probes_20260729"
research_only: true
score_claim: false
promotion_eligible: false
authority_axis: "[macOS-CPU advisory]; pure array analysis of data already on disk (ru1 atlas_flat.npz + gt_n600 lstars); NO SegNet/PoseNet forward ran, NO training, NO launches (pb1 owns the scorer slot)"
verdict_scope: "FORMULATION (both probes test one representation of the endpoint residual; a falsified probe retires that representation, not the residual or the family)"
pointer_before: "0.1910828242 [contest-CPU]"
pointer_after: "0.1910828242 [contest-CPU] UNMOVED"
pointer_delta: 0
consumers: [pb1_P2c_round2, attack_search_arm, E2_tree_N3_N4, r7_flicker_phase_channel, c1_waterfill]
related_deliberation_ids: [ddm_gc6_from_endpoint_convocation_20260729, ddm_ru1_endpoint_typing_20260729, box_retired_min_s_target_20260728]
---

# ddm_of1 — offset-field coherence (P2C-OF) + flicker-phase coherence (W1-COH)

**POINTER HONESTY FIRST: `0.1910828242 [contest-CPU]` UNMOVED.** Competitive
effective frontier is the official **0.172141** (PR130). This arm moved NO
pointer and ran NO scorer — it is $0 pure-array analysis of ru1's flip atlas
and the gt_n600 GT-argmax cache, both already on disk. Every number is
`[macOS-CPU advisory]`, `score_claim=false`. A falsified probe is a **priced
door with a precondition tag**, not bad news.

## THE TWO DECISIVE NUMBERS (up front)

**① P2C-OF — offset-field DIMENSION COLLAPSE FAILS. `flips-per-DOF ≈ 1.1–3.3`
(all bands) / `4.4` (deep) vs ru1's `+24 flips/quantum`; autocorr length
`L = 1 px` (≤ the 3 px falsifier).** The endpoint residual is NOT a coherent
1-D normal-offset field δ(s) on the separatrix — it is short (median band
arclength 2 px, deep-band 3 px), razor-thin (τ̄ ≈ 1.14 px), speckly boundary
jitter with autocorrelation length 1 px and conditional entropy (0.85 b/node)
≈ marginal (1.09 b/node). Representing it in an offset basis is **5–21× LESS
efficient** than aimed single-quantum token edits. **FALSIFIER FIRED → the
offset-field solve is DEAD at FORMULATION scope.**

**② W1-COH — flicker regions ARE phase-coherent; the tail flicker mass
RE-PRICES from 1.2731 to `B/err ≈ 0.064` (all) / `0.29` (deep tail) — 4–20×
under water — but GATED on a NEW precondition (receiver-derivable region
support, since transmitting it costs ~1.42 MB).** Area-weighted per-component
phase agreement = **0.869 > 0.8** (falsifier does NOT fire). Incremental
phase-bit budget = **12,580 bytes** (1 bit / region-instance, 100,639
instances over 600 pairs) fixes ~197,544 coherent flicker flips
(d_seg reach ≤ 0.00167). **FALSIFIER NOT FIRED → door OPEN and priced**, but
the binding gate is support-derivability (a new $0 sub-probe), NOT coherence.

---

## PROBE 1 — P2C-OF: is the deep band a coherent 1-D offset field? (gc6 T2)

**The claim under test (gc6 T2):** 94% of flips sit ON a GT inter-class
contour → IF the flips project onto contours as a smooth low-entropy normal
offset δ(s), the 47.8% deep band is solvable as a low-dim OFFSET FIELD (through
the rank-4 head) instead of per-pixel — the path from the GN ceiling 0.002511
toward ~1e-3.

**Method (receipt `offset_field_coherence_receipt.json`, all n600, 2.7 s):**
reconstruct the flip mask per pair, connected-component it (8-conn — the flip
*bands*), measure per-band PCA arclength + thickness τ = area/arclength,
the within-band thickness autocorrelation along arclength (L = first lag <
1/e), the marginal vs conditional (next-node) entropy of the quantized
thickness field, and `flips-per-DOF = total_flips / total_DOF` with
`DOF = Σ_bands ⌈arclen / L⌉` for L ∈ {1, 3, per-band-constant}. Stratified by
band-mean m_def and by class-pair.

**MEASURED (all n600):**

| stratum | n bands | flips | band arclen (mean / p90) | τ̄ | autocorr L | flips/DOF (L=1 / L=3 / per-band) |
|---|---|---|---|---|---|---|
| all | 139,778 | 458,738 | 2.88 / 6 | 1.14 | **1 px** | **1.14 / 2.31 / 3.28** |
| deep (m_def>0.25) | 81,899 | 363,756 | 3.77 / 8 | 1.18 | **1 px** | 1.18 / 2.67 / **4.44** |
| shallow (≤0.25) | 57,879 | 94,982 | 1.62 / 3 | 1.01 | — | 1.01 / 1.52 / 1.64 |
| **very-deep tail (m_def≥1.0)** | 30,352 | 77,392 | **2.23 / 5** | — | **1 px (neg@lag1)** | **2.55** |

- **Geometry:** 93.9% of flips on-boundary, 99.9% within 3 px (ru1
  confirmed). Only **18.0%** of GT contour arclength carries ANY flip
  (2,551,382 boundary px, τ̄ = 0.180 flips/boundary-px) — the residual is a
  SPARSE thin decoration on the contour, not a coherent displaced band.
- **Autocorrelation length L = 1 px** for all AND the deep band (autocorr(1) ≈
  0.13–0.15, then ≈ 0 and negative). The very-deep expensive tail (m_def≥1.0,
  the 16.9% priced-above-water mass) is the MOST speckly: median area 1,
  autocorr(1) NEGATIVE (−0.12) — isolated deep-deficit pixels, not bands. The
  offset-field chart helps LEAST exactly where gc6 hoped it would help.
- **Entropy:** marginal 1.09 b/node, conditional (given previous arclength
  node) 0.85 b/node — mutual information only **0.24 b/node** (22% reduction).
  Conditional ≈ marginal ⇒ white field.
- **Class-pair check (Fridrich lane-corridor stored-stream exception):** the
  dominant Road↔Lane band (227,748 flips) has arclen 2.42, flips/band 2.7 —
  the LANE contours are the SHORTEST/speckliest, not coherent. The only
  moderately-elongated class-pairs (Road↔Movable arclen 5.29, Undriv↔Movable
  4.49) still give flips/band ≤ 7. **The lane-corridor exception does NOT
  reopen.**

**FALSIFIER — FIRED (both criteria).** autocorr length 1 px ≤ 3 px AND
conditional ≈ marginal entropy. `flips-per-DOF` under the MOST generous DOF
accounting (one constant offset per whole band) is 3.28 (all) / 4.44 (deep),
and at the measured L=1 it is ≈ 1.1 — every accounting is 5–21× BELOW ru1's
+24 flips/quantum aimed-edit currency. The offset field neither reduces the
independent-target count below the per-band level nor beats the aimed-edit
currency.

**VERDICT: offset-field dimension-collapse solve DEAD at FORMULATION scope.**
The 47.8% deep band is NOT coherently solvable by a δ(s) chart; it must be
attacked per-pixel / per-quantum (T3 attack-search) or by a substrate that
LENGTHENS the coherent segment. **Precondition tag (re-prices if):** a
smoothing / temporal-voting / morphological-opening substrate lands that
raises the within-band autocorrelation length above ~3 px (i.e. that turns the
1-px speckle into runnable segments). Fridrich constraint honored throughout —
δ(s) was only ever a solve-target chart here; sp1 already measured explicit
contour bytes DEAD (444 KB), and the lane-corridor exception is now closed by
measurement too.

## PROBE 2 — W1-COH: does GT flicker flip phase-coherently by region? (gc6 W1)

**The claim under test (gc6 W1, "the biggest wonder finding"):** flips are 40×
enriched on GT-flicker sites (49.5% at flips vs 1.246% base; 64% of the deep
tail). IF flicker components flip phase COHERENTLY within a region across
adjacent pairs, a per-pair per-REGION phase bit prices the tail at
bits-per-region instead of per-pixel B/err.

**Method (receipt `flicker_phase_coherence_receipt.json`, all n600, 1.1 s):**
per pair, flicker field = (lstars[p] ≠ lstars[nb]); connected-component it;
per component measure the majority-transition fraction (phase agreement);
the per-region phase-bit budget (1 bit/region-instance upper bound); the
pessimistic support-transmission cost (flicker-mask entropy); the tail re-price
B/err = phase_bytes / flicker_flips_fixed vs the 1.2731 water.

**MEASURED (all n600):**
- flicker = 1.246% of frame (ru1 confirmed); 167.7 components/pair;
  component area mean 14.6, median 3, p99 236, max 7221 (28.3% singletons).
- **Phase agreement (majority-transition fraction): unweighted 0.956,
  AREA-WEIGHTED 0.869, area≥4 = 0.861.** 70.9% of flicker mass lives in
  regions with agreement ≥ 0.8.
- **Phase-bit budget: 100,639 region-instances over 600 pairs → 12,580 bytes**
  (1 bit each, upper bound; temporal predictability would lower it).
- **Pessimistic support cost: 1,415,927 bytes** (flicker-mask entropy) — if the
  region supports must be TRANSMITTED, that alone is 0.94 S rate → the channel
  is DEAD. **Therefore support-derivability is the binding precondition, not
  coherence.**
- **Tail re-price:** B/err (all flicker, 197,544 fixed) = **0.0637**; B/err
  (deep tail flicker only, paying all region bytes) = **0.2913** — both 4–20×
  under the 1.2731 water. d_seg reach ≤ 0.00167 (all flicker) / 0.00037 (deep
  tail). Waterfill break-even check: benefit/cost ≈ **20×** (S-benefit/flip
  8.48e-7 vs S-cost/flip 4.24e-8) — clears break-even 25·ΔB/37.5M by 20×,
  CONDITIONAL on derivable support.

**FALSIFIER — NOT FIRED.** Area-weighted phase agreement 0.869 > 0.8. The
flicker mass is genuinely region-coherent (the majority phase describes ~87% of
each region's flicker), even for the large regions (area≥4 = 0.861).

**VERDICT: region-phase pricing is a PRICED, OPEN door (FORMULATION scope) — the
gc6 "biggest wonder finding" IS compressible.** The 64% flicker portion of the
16.9% priced-above-water tail re-prices from 1.2731 to 0.064–0.29 B/err,
collapsing the priced-above-water tail from ~16.9% toward ~6.1% (the 36%
non-flicker remainder). **BUT the realization gate is NOT coherence — it is
receiver support-derivability** (transmitting supports costs 1.42 MB;
prohibitive). **Precondition tag (the next $0 rung):** can the receiver DERIVE
the flicker region supports from its own reconstruction / token field, or from
a stored low-dim map cheaper than 12.6 KB? That is the SW-DERIVE sub-probe —
the true gate, and the honest successor question this probe hands forward
(**NOW MEASURED — §SW-DERIVE below; verdict: PARTIAL, channel ADMISSIBLE via a
stored static map at 12–26 KB all-in**). 15% of flicker mass (incoherent,
agreement < 0.8) stays per-pixel-priced.

## §SW-DERIVE — receiver support-derivability (coordinator round-2; the W1-COH admission gate)

**THE QUESTION:** can the RECEIVER derive the flicker-region supports at decode
time, so the ~12.6 KB phase channel pays without the 1.42 MB support
transmission? **THE ANSWER (one line): PARTIAL → REDUCED ARITHMETIC, CHANNEL
ADMISSIBLE — not derivable at 0 support bytes (the receiver-instability oracle
is structurally blind to the target mass), but the supports are strongly STATIC
in image coordinates, so a stored static map collapses support cost 1.42 MB →
9.7–13.5 KB and the channel nets B/err 0.075–0.141 vs the 1.2731 water (9–17×
under), at recall 0.79–0.92.** Receipt:
`swderive_support_derivability_receipt.json` (same SSD dir; n600; deterministic
re-run identical; NO scorer — realized argmax reconstructed EXACTLY from
gt lstars + the atlas flip records, which record every flip).

### Source (a) — receiver-realized argmax instability: ORACLE, and it FAILS structurally

R[p] = realized[p] ≠ realized[nb] (nb = p+1, ru1 convention). MEASURED:
- **As a FIELD, R aligns well with GT flicker:** pixel recall 0.708, precision
  0.768, IoU 0.583 — the receiver's own temporal instability lives in the same
  places as the scorer's.
- **But at the TARGET pixels (flicker-flagged flips) exact-pixel recall =
  0.054 — the coordinator's <50% falsifier FIRES.** Mechanism (the honest
  finding): a flip-on-flicker-site is a place where GT flickers and our carrier
  does NOT — the receiver's output is temporally STABLE precisely where it is
  stably wrong. The signal the receiver would look for is absent at the target
  mass BY CONSTRUCTION. Net at strict reading: 9,975 B / 10,645 fixed = B/err
  0.937 (under water only nominally; reach negligible).
- **Rim structure:** R dilated 1 px recalls **0.641** of flicker flips — the
  targets sit immediately ADJACENT to receiver-visible instability. With R⊕1 as
  support: ≤9,975 B (dilation only merges instances) / 126,600 fixed → B/err
  ≈ 0.079. BUT: **source (a) is an ORACLE BOUND regardless** — decode-time
  argmax needs SegNet, FORBIDDEN at inflate (strict scorer rule). Compliant
  realization would need a counted distilled proxy head (unpriced here).
  **Precondition tag:** re-prices if a tiny counted proxy (< ~10 KB) can
  reproduce R⊕1; not a live channel today.

### Source (b) — realized margin bands: UNMEASURABLE-WITHOUT-SCORER (skipped honestly)

Full-field realized margins are not on disk (atlas `gap12` covers flip sites
only). No scorer forwards permitted in this arm → skipped, stated honestly.

### Source (c) — static flicker-frequency map: THE COMPLIANT WINNER

f(y,x) = per-pixel GT-flicker frequency over n600 (g4 stationarity lineage,
recomputed fresh from GT lstars — vehicle-independent); static support =
{f ≥ θ}, shipped as COUNTED bytes (entropy-bound priced; rule-118 honest).
MEASURED sweep (θ → recall / total bytes / net B/err):

| θ | map px | static comps | map bytes | phase bytes (optimistic: comp×600) | total B | flicker-flip recall | flips fixed | net B/err | vs water |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 24,841 | 60 | 13,451 | 4,500 | **17,951** | **0.924** | 182,544 | **0.0983** | 13× under |
| **0.05** | 15,339 | 27 | 9,711 | 2,025 | **11,736** | **0.790** | 155,956 | **0.0753** | 17× under |
| 0.10 | 8,394 | 40 | 6,255 | 3,000 | 9,255 | 0.538 | 106,336 | 0.0870 | 15× under |
| 0.20 | 2,520 | 16 | 2,432 | 1,200 | 3,632 | 0.206 | 40,611 | 0.0894 | falsifier fired |

**Conservative pricing** (1 phase bit per PER-PAIR flicker component
intersecting the map, honest against out-of-phase subregions merged inside one
static component — the 0.869 coherence was measured at per-pair granularity):
θ=0.05 → 88,927 instances, total **20,827 B, net B/err 0.134**; θ=0.02 →
98,411 instances, total **25,753 B, net B/err 0.141**. **Both still 9–10×
under water — the verdict is robust to the pricing granularity.**

### The reduced arithmetic (the coordinator's decisive number)

- **Admissible operating point (reach-max, θ=0.02):** 17,951–25,753 B all-in
  (optimistic→conservative) fixes ~182,544 flicker flips → **d_seg reach
  0.00155**, net **B/err 0.098–0.141** vs water 1.2731. Deep-tail view: fixes
  ~39,889 of the 49,680 deep-tail flicker flips — even charging ALL bytes to
  the deep tail alone gives B/err 0.65, still under water.
- **Honest miss mass:** 17,206 flicker flips outside the θ=0.02 map (7.6%) +
  ~23,900 incoherent-region flips (1−0.869 of recalled) ≈ **44,724 flicker
  flips stay per-pixel-priced** (plus the 36% non-flicker tail remainder,
  unchanged). The priced-above-water tail collapses from ~16.9% of flip mass
  toward ~7–8% under this channel.
- **Labels:** recall/bytes/instances MEASURED; the ×0.869 coherence factor and
  the reach numbers DERIVED (information-layer); actuation (how the receiver
  realizes the phase-fix on frames — painting/token conditioning) is the
  remaining carrier-design question, NOT priced here (realization is
  quantization-gated; the lesson stands). Map bytes are entropy bounds; a real
  coder (brotli-packed bitmap) lands within ~1.2×, inside the water margin.
- **Verdict vocabulary (per coordinator): PARTIAL** — source (a) falsifier
  FIRED (0.054 < 0.5, structural), source (c) falsifier NOT fired (0.79–0.92
  recall), support cost is 9.7–13.5 KB stored (not 0). **Channel ADMISSIBLE
  under the reduced arithmetic; admission condition for E2/r7 = ship the
  static map + phase stream and solve the actuation leg.**

## Consumers (named, per gc6 §4 routing)

- **pb1 P2c round-2** ← Probe 1: **DROP the "boundary-walk along measured δ(s)"
  proposal** from the gc6-T3 proposal mix (P2C-OF measures δ(s) white — walking
  it is no dimension win). KEEP atlas-aimed channel-sign singles + joint 4-ch
  cell edits + square-attack patches; +24/quantum stays the target metric.
  ← Probe 2: **the coherent flicker regions (0.87 agreement, 70.9% of flicker
  mass ≥0.8) are HIGH-VALUE aimed-edit targets** — one aimed edit per coherent
  region fixes its bulk; deprioritize the incoherent (<0.8) 15% flicker mass.
- **attack-search arm** ← Probe 1: confirms per-pixel/per-quantum aimed editing
  is the CORRECT currency for the deep band; the offset-field chart does not
  beat it, so keep SparseRS/Square/boundary-walk-in-token-space (NOT in δ-space).
- **E2 tree N3–N4** ← Probe 1: **remove the offset-field lever from the
  beyond-GN seg menu** — the deep-band 0.00251→~1e-3 path via δ(s) is CLOSED.
  ← Probe 2: **add the flicker-phase channel as a candidate beyond-GN seg lever
  gated on SW-DERIVE** (reach ≤ 0.00167 d_seg if all-flicker; ≤ 0.00037 for the
  deep-tail portion; overlap with the GN band is unquantified here — a
  composition question for E2).
- **r7 flicker-phase channel** ← Probe 2 + §SW-DERIVE: the admission gate is
  now MEASURED — carrier design = static support map (θ∈[0.02,0.05], 9.7–13.5
  KB counted) + per-region phase stream (2–12.3 KB), net B/err 0.075–0.141;
  the remaining r7 leg is ACTUATION (how the decode realizes the phase-fix on
  frames), not support location.
- **c1 waterfill** ← Probe 2 + §SW-DERIVE: the flicker-phase action's
  conditional 0.064 row is re-priced UNCONDITIONAL at **0.098–0.141 B/err**
  (θ=0.02 reach-max, optimistic→conservative) — still clears break-even
  25·ΔB/37,545,489 by ~9–13×; enters the waterfill as a real action.
- **pb1 P2c round-2 (SW-DERIVE addendum):** the θ=0.02 static map (24,841 px,
  60 regions) is ALSO the natural aim-mask for flicker-targeted quantum edits —
  92.4% of flicker-flip mass lives inside it.

## Honesty labels, scope, hooks

- **MEASURED:** every number carries its receipt
  (`offset_field_coherence_receipt.json` / `flicker_phase_coherence_receipt.json`
  on SSD; atlas_flat.npz + gt_n600 lstars). **DERIVED:** flips-per-DOF formula,
  break-even 25·ΔB/37,545,489, water 1.2731, d_seg reach = flips/(384·512·600).
  **CONJECTURE (labeled inline):** the 87%-coherence → 87%-of-flicker-fixed
  reach is an INFORMATION upper bound (actuation via token edits is separate);
  the 12.6 KB budget assumes 1 bit/region (temporal predictability would lower
  it). **No scorer job ran; pb1's slot untouched; $0; no launches.**
- **verdict_scope FORMULATION** for both: Probe 1 retires the offset-field
  *representation*, not the residual or the attack family; Probe 2 opens the
  region-phase *representation* pending one named sub-probe. Neither is a
  FAMILY/PARADIGM verdict.
- **Sandbox honesty:** Probe 1 is a clean FALSIFIER-FIRED priced door (the deep
  band is per-quantum territory, not offset-field territory); Probe 2 is a
  FALSIFIER-NOT-FIRED priced door with exactly one binding precondition to test
  next. Neither moves the pointer; both sharpen the E2 seg-lever menu.
- **6-hook wire-in:** sensitivity-map = ACTIVE (band geometry + flicker
  decile); Pareto = ACTIVE (both B/err vs water); bit-allocator = ACTIVE
  (flicker-phase waterfill action); cathedral autopilot = N/A (no dispatchable
  archive — pb1 owns the candidate); continual-learning = ACTIVE (this memo +
  DAG FEED); probe-disambiguator = ACTIVE (both probes ARE $0 pre-registered
  falsifiers).

## DAG FEED — ddm_of1 (2026-07-29)

FEED-of1: the two gc6 $0 coherence probes MEASURED at n600 (pure array,
NO scorer, pb1 slot untouched). **P2C-OF (offset field): FALSIFIER FIRED.**
Endpoint residual is NOT a coherent 1-D δ(s) field — connected flip bands are
short (median arclen 2 px, deep 3 px, expensive m_def≥1.0 tail 2.23 px with
NEGATIVE lag-1 autocorr), razor-thin (τ̄ 1.14), autocorr length **1 px** (≤3
falsifier), conditional entropy 0.85 ≈ marginal 1.09 (MI 0.24 b/node). Only 18%
of GT contour arclength carries any flip. **flips-per-DOF ≈ 1.1 (L=1) to 3.3
(all)/4.4 (deep) under the most generous per-band-constant accounting — 5–21×
BELOW ru1's +24/quantum.** Offset-field dimension-collapse solve DEAD at
FORMULATION; the Fridrich lane-corridor stored-stream exception ALSO closed by
measurement (Road↔Lane bands arclen 2.42, the shortest). Precondition: reopens
only if a smoothing/temporal-voting substrate raises within-band autocorr >3 px.
**W1-COH (flicker phase): FALSIFIER NOT FIRED.** Area-weighted per-region phase
agreement **0.869 > 0.8** (area≥4: 0.861; 70.9% of flicker mass ≥0.8).
Incremental phase-bit budget **12,580 B** (100,639 region-instances, 1 bit each)
fixes ~197,544 coherent flicker flips → **B/err 0.064 (all) / 0.29 (deep tail),
4–20× under the 1.2731 water**, clearing break-even ~20×; d_seg reach ≤0.00167.
The gc6 "biggest wonder finding" (40× flicker enrichment) IS compressible — the
tail's flicker half re-prices below water. BINDING PRECONDITION is NOT coherence
but receiver SUPPORT-DERIVABILITY: transmitting region supports costs 1.42 MB
(0.94 S) → the door opens only if supports are receiver-derivable (SW-DERIVE
sub-probe = the next $0 rung, routed to r7). Consumers: pb1 P2c round-2 (drop
δ-space boundary-walk; use coherent flicker regions as aim targets) · attack
arm (per-quantum currency confirmed) · E2 N3-N4 (remove offset-field lever, add
SW-DERIVE-gated flicker-phase lever) · r7 (flicker channel design) · c1
(waterfill action B/err 0.064). Pointer 0.1910828242 [contest-CPU] UNMOVED; all
advisory. [no-triality] [p0-ledger-ok]

FEED-of1b (SW-DERIVE round-2, same arm): the W1-COH admission gate MEASURED at
n600, $0, NO scorer (realized argmax reconstructed EXACTLY from gt lstars +
atlas flips). **VERDICT: PARTIAL → channel ADMISSIBLE via stored static map.**
Source (a) receiver-instability ORACLE FAILS STRUCTURALLY at the target mass:
exact-pixel flicker-flip recall **0.054** (<0.5 falsifier FIRED) because the
receiver is temporally STABLE precisely where it is stably wrong (GT flickers,
carrier doesn't — that IS the flip); yet as a field R aligns (pixel recall
0.708/precision 0.768/IoU 0.583) and R⊕1px recalls **0.641** — flips sit on the
RIM of receiver-visible instability; (a) is compliance-blocked anyway (SegNet
at decode forbidden; counted-proxy precondition, unpriced). Source (b) realized
margins UNMEASURABLE-without-scorer (skipped honestly). **Source (c) static
flicker-frequency map = the compliant winner (g4 stationarity lineage, fresh
from GT lstars):** θ=0.02 → recall 0.924, 60 static regions, 17.9 KB all-in,
net B/err 0.098; θ=0.05 → recall 0.790, 11.7 KB, 0.075; CONSERVATIVE per-pair-
component pricing 20.8–25.8 KB → 0.134–0.141 — ALL 9–17× under the 1.2731
water. Reduced arithmetic: d_seg reach 0.00155 at θ=0.02; miss mass ≈44.7K
flicker flips stays per-pixel-priced; priced-above-water tail collapses ~16.9%
→ ~7–8%. Support cost 1.42 MB → 9.7–13.5 KB. Remaining leg = ACTUATION
(realization-is-quantization-gated), routed to r7 carrier design; c1 waterfill
row re-priced UNCONDITIONAL 0.098–0.141; pb1 gets the θ=0.02 map as flicker
aim-mask. Pointer 0.1910828242 [contest-CPU] UNMOVED; all advisory.
[no-triality] [p0-ledger-ok]

## STORES CONSULTED

CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md (cited per
contract); MEMORY.md CURRENT-STATE (box_retired_min_s_target band lemma /
water 1.2731 · frozen_scorer_exact_factorization · segnet_recursive_fractal ·
verdict_scope_ladder · findings_are_first_rungs · charter_composition_regrep);
ddm_gc6 convocation memo §T2 (P2C-OF spec) + §W1/W2 (W1-COH spec) + §4 rows 3-4
+ §5 E2 tree + STORES; ru1 tool tools/ru1_endpoint_residual_atlas.py (branch
ddm-ru1-20260729, loaders reused: parse-free atlas + open_stored_npy_memmap
lstars) + ru1 atlas_analysis_receipt.json (positive-control: n_flips 458738,
on-boundary 0.9386, flicker-at-flips 0.4954 — all reproduced here);
atlas_flat.npz custody; gt_n600.npz lstars (384×512×600 int64 GT argmax).
Receipts on SSD: /Volumes/VertigoDataTier/pact/ddm_of1_20260729/.
