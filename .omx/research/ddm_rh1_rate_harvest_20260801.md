# ddm_rh1 (#853) — the rate half is NOT separable from the pose wall; the real pose-independent lever is 27× smaller and sits inside the coder

**Date:** 2026-08-01 · **Arm:** ddm_rh1 · **Cost:** $0 (scorer-free, training-free, no dispatch,
no n600 slot taken) · **Axis:** byte columns are **EXACT** (the r7 encoder is deterministic and
lossless — not an advisory proxy); no seg or pose claim is made anywhere in this memo.
`score_claim=false · promotable=false · pointer_moved=false`.

**Review status:** pre-registered falsifier + own round-1 review (1 pass). NOT fresh-eyes reviewed.

**Pointer honesty first: own-vehicle frontier v4d = 0.9639878, bar 0.172141, gap 0.7918468.
UNMOVED. Nothing in this memo moves it**, and the largest item it finds would take v4d to
0.9621360 — still 5.6× the bar.

**STORES CONSULTED:** `tools/corpus_query.py "token coder race SMEVR brotli export path shipped
coder rate pricing"` (research 7352 / equations 864 / memory 2040 / dag 908 / council 292 / tasks
396 / docs 96); memory `[[tr1_architecture_mechanism_read_from_the_receiver_20260801]]`;
`.omx/research/ddm_cr1_composition_row_827_20260801.md`;
`.omx/research/ddm_rsf1_rate_surrogate_fidelity_20260801.md`;
`.omx/research/ddm_gd1_generic_default_census_20260731.md` (rows S5/T4/T5);
`.omx/research/ddm_r7_token_coder_race_20260729.md`; commits `743e70fb06`, `fe1175c4e5`;
`experiments/ddm_r7_token_coder.py`; `src/tac/optimization/ddm_tr1_runtime.py`.
**Deliberately NOT loaded:** the burn-4 window telemetry and the gc-series convocation memos —
neither bears on a lossless byte question.

---

## §0 ANSWER FIRST

The item as charted — *"−0.0499 S of rate, pose-independent, harvestable without a pose re-solve"* —
**does not exist at that size.** Three measured corrections, in descending importance:

1. **The −0.0499214 S arithmetic is right and reproduces exactly (−74,973 B). Its
   *separability* claim is wrong.** 99.0% of it is carried by the token **event field**, which *is*
   the renderer input. Harvesting it means shipping ep854's renders, which is exactly the object
   cr1 measured at **6.36× d_pose on 61 of 61 pairs**. Rate and the pose wall are the same object.
   **§4.**
2. **#843 was already settled before this arm existed, and its premise is refuted for our lineage.**
   v4d/gr1 ship `codec=smevr`, proven from the bytes. The shipped archive **is** getting the rate the
   trainer bought; no ledger rate figure needs restating. **§1.**
3. **A genuinely pose-independent rate lever does exist, and it is new:** the SMEVR **base rule** is
   an unraced generic default and is measurably suboptimal — **−2,781 B = ΔS −0.0018518** on the
   shipped v4d field, replicated on a second field. That is **27× smaller** than the charted figure.
   **§5.**

| item | ΔS | % of 0.7918468 gap | % of 0.097465 inventory | pose-independent? |
|---|---:|---:|---:|---|
| charted "rate half" | −0.0499214 | 6.304% | 51.22% | **NO — §4** |
| coder swap on v4d | −0.0000000 | 0.000% | 0.00% | yes, but **EXHAUSTED** |
| **base rule (NEW, measured)** | **−0.0018518** | **0.234%** | **1.90%** | **yes** |

*Calibration note:* the instructed **1.41e-4 S advisory-seg optimism** (cr1 §7) is **not applied**,
because it applies to advisory seg measurements and this memo makes none. Every number here is an
exact byte count from a deterministic lossless encoder. Stating where a calibration does *not*
belong is part of applying it.

**Reproduce:**
```bash
.venv/bin/python experiments/ddm_rh1_token_field_rate_decomposition.py \
  --field "v4d_shipped=/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_cell_drop50_archive.zip" \
  --field "ep854_burn=/Volumes/VertigoDataTier/pact/ddm_ep2_20260731/archives/w03_ep854_representative/archive.zip" \
  --json-out .omx/research/ddm_rh1_token_rate_decomposition_20260801.json
```

## §1 #843 — SETTLED, from the bytes (MAIN item 1)

Not inherited from the commit messages. `decode_token_codes(verify="canonical")` **re-encodes and
refuses on any difference**, so a successful canonical decode is itself proof of the shipped codec.

| MEASURED | |
|---|---|
| `gr1/state/tokens.dr7t` == `v4d/state/tokens.dr7t` | byte-identical, sha `305a2be96a29` |
| codec | **`smevr`** — canonical decode PASSED |
| re-encode with smevr | **byte-identical to the shipped member** |
| framing | 346,478 = header 56 + base 1,360 + delta 345,062 |

⇒ **The premise "the shipped archive is not getting the rate the trainer bought" is FALSE for the
own-vehicle lineage.** Every v4d rate column stands as recorded. This independently reproduces
`fe1175c4e5` rather than trusting it.

The Brotli path is real but is **not our shipping grammar**: ep854's TR1 packet carries a
`TR1TOK1!` Brotli token section at **355,182 B** vs **271,505 B** under SMEVR — an 83,677 B
over-count (ΔS 0.0557). That archive was built **2026-07-31T15:13:32**, eight hours *before* the
#843 receiver landed (23:34:37). It is a stale artifact.

### §1a A residual #843 gap I did not expect — for MAIN, warn-level

`ddm_tr1_runtime.py:319` reads `if "token_codec" in value and value["token_codec"] != ...` —
**absence is admitted by design** (the source comment says so). MEASURED: ep854's selector has
**no `token_codec` key** (19 keys, none of them it), and `parse_archive` **accepted it cleanly while
it shipped Brotli**. `compile_archive_from_checkpoint` now *defaults* to SMEVR (`:1102`) — that part
of the fix is real — but `token_codec=None` remains reachable and
`test_legacy_token_codec_stays_absent_and_byte_identical` pins that behaviour.

⇒ **The receiver cannot distinguish a legacy-Brotli archive from a compliant one; a mispriced
archive passes every check silently.** This is the
`[[vacuity_is_indistinguishable_from_pass_empty_scope_confound_20260801]]` class — absent scope
emits the same symbol as a clean scope. Not a blocker for score work, and I did not fix it: it is a
receiver-contract decision (does the vehicle still need the legacy path?) that belongs to MAIN.

## §2 No double-count (MAIN item 2)

The two figures are orthogonal, and the arithmetic shows why: `fe1175c4e5`'s "already spent" is the
**coder swap on v4d's own field** (Brotli→SMEVR = **0 B**, verified above). The −0.0499214 compares
**two different token FIELDS both already priced through SMEVR** (346,478 vs 271,505). Both sides
carry the coder cut, so it cannot be counted twice. **The headline does not change sign — it changes
category (§4).**

## §3 DECOMPOSING THE 21.6% — by substitution, not by narrative

Both fields are `(600,24,32,4)`, `levels=16`, priced through the same encoder, so grammar is
controlled by construction.

| leg | bytes | share |
|---|---:|---:|
| base stream | −754 | **1.0%** |
| **delta (event) stream** | **−74,219** | **99.0%** |

**Hypothesis (b), "grammar / cell-mask artifact" — KILLED by measurement.** Constant-cell fraction
is **0.50260 (gr1) vs 0.50521 (ep854)** — both carry the QA24 keep-50 mask. The mask is *not* the
differentiator. **Hypothesis (c), "missing pose"** — controlled: cr1 adds the identical 12,743 B
non-token overhead to both sides and this comparison is token-only.

**Hypothesis (a), learned structure — CONFIRMED and quantified:** event rate
**0.37415 → 0.27661** (×0.739); order-0 delta entropy **2.1888 → 1.7970 bits** (×0.821); the mode
base uses **16 → 9** of the available symbols. Seg-only training drives cells toward temporal
constancy.

**Controlled hybrids** (`reconstruct_mode_delta` lets one field's base be paired with the other's
events, then each is re-encoded):

| | framed |
|---|---:|
| gr1 (reference) | 346,478 |
| base = gr1, events = ep854 | 293,600 → **−52,878 B (70.5%)** |
| base = ep854, events = gr1 | 353,853 → **+7,375 B (the base ALONE is WORSE)** |
| ep854 (reference) | 271,505 |

The legs are **non-additive by −29,470 B (39.3%)**: SMEVR's residual is `(v − base) mod levels`, so
a base only pays with *its own* events. Reporting an additive split here would have been a fabricated
attribution; the interaction term is carried explicitly in `FieldGap.interaction_byte_delta`.

## §4 WHY THE RATE HALF IS NOT SEPARABLE — the correction that matters

The token codes **are** the renderer input:
`value = clip(tokens_base + tokens_delta[p], −1, 1)` → `conv0` → … → frames (receiver, memory
`[[tr1_architecture_mechanism_read_from_the_receiver_20260801]]`). §3 shows the −74,973 B is 99%
carried by the token *values*. Therefore:

> Changing the tokens changes the renders, hence **seg AND pose together**. The −0.0499214 S and
> cr1's 6.36×/61-of-61 pose wall are **two readings of one object**, not two separable items.

What *is* pose-independent is a **lossless recode of a FIXED field** — there the decoded lattice is
bit-identical, so seg and pose are unchanged by construction. That is the class §5 measures, and it
is the only class that needs no scorer slot. **"A rate-only harvest needs no pose re-solve" is
false as stated; it is true only of §5's class, which is 27× smaller.**

**The mechanism finding still transfers** — and it is the honest harvest: a **26% lower event rate
is what seg-only training produced**. Whether a burn window carrying a nonzero pose term keeps that
rate benefit while remaining pose-legible is cr1 reformulation #2, and it is a **training-config
question, not a recode**. It cannot be settled by re-coding anything we already hold.

## §5 THE SMEVR BASE RULE IS AN UNRACED GENERIC DEFAULT (new, measured, pose-independent)

`encode_token_codes` hardcodes `base, delta = factor_mode_delta(value, levels)` — a per-cell
**temporal mode**, chosen because it is deterministic, never raced for bytes. It maximises exact
zeros (which the occupancy stream likes) and is **blind to the value stream's rank cost**. This is a
gd1-class *GENERIC-CHOSEN-UNRACED, LIVE NOW* default sitting directly on the binding rate axis —
the same classification gd1 gave rows T4/T5, one layer deeper, inside the shipped coder.

Raced a derived family against **real encoder bytes** (losslessness asserted at every point; the
incumbent is the `α → ∞` corner):

```
base(cell) = argmin_b  Σ_s hist[cell,s] · C((s − b) mod L)
C(r)       = 0 if r == 0 else α + circdist(r)^p
```

| field | incumbent (mode) | argmin | bytes | ΔS |
|---|---:|---|---:|---:|
| **v4d/gr1 SHIPPED** | 346,478 | α=0, p=2.0 | **343,697** | **−0.0018518** |
| ep854 burn | 271,505 | α=0, p=1.5 | 270,333 | −0.0007804 |

- **The incumbent mode is the WORST corner of the family on both fields**, and the result
  **replicates on an independent field** — not one lucky artifact.
- The optimum is **interior**, not a boundary artifact: pushing `p` past 2 reverses
  (p=3 → −2,685; p=6 → −1,638; p=10 → −27; p=20 → **+3,062**; exact minimax limit → **+6,458**).
- **−2,781 B is a LOWER bound** on the lever: it is the argmin of *this* family, not of all bases.

**Blocker, stated plainly:** shipping this needs a **receiver/format change** —
`decode_token_codes(verify="canonical")` re-encodes with `factor_mode_delta` and refuses a non-mode
base. The base is already stored and already counted, so the change is rule-118 free, but it is a
format change with a two-landing obligation, not a free recode. **I measured the prize; I did not
modify the coder.**

**Coder choice on a fixed field is EXHAUSTED**, now confirmed on a *second* field: sweeping all 9
registered r7 codecs, `smevr` is the strict argmin on both (v4d next-best **+49,964 B**; ep854
next-best **+57,074 B**).

## §6 A SURROGATE-FIDELITY RESULT — my DERIVED rule lost to a generic control

I built a cost table *derived* from the incumbent encode's own statistics (measured occupancy rate →
`occ_bits(on)=1.418`, `occ_bits(off)=0.676`; measured rank histogram → `rank_bits ∈ [2.39, 12.70]`).
It proposed a base worth **−1,140 B**. A **generic circular-median control** scored **−2,117 B**, and
the raced family found **−2,781 B**. **The derived surrogate lost to the generic control by 1.9×.**

Mechanism: the surrogate is *context-blind*, while SMEVR's occupancy stream is conditioned on
(base value, previous-frame occupancy, left/upper spatial occupancy, age bucket) and adapts as it
codes. A marginal bit-cost cannot see that.

This is the same shape as **rsf1**, which measured the in-loop `entropy` rate model **anti-correlated**
(ρ = −0.7235) with shipped bytes on the live lineage. rsf1 found it for a *training* surrogate; this
finds it for a *coder-internal* one.

> **LAW: price with the real coder. A hand-derived surrogate misranks for the same reason an
> in-loop one does — both are marginal models of a context-adaptive stream.** "Derived" is not a
> synonym for "correct"; a derivation is a hypothesis until a real encode ranks it. Kept honest here
> only because every candidate was scored by `_encode_smevr`, never by the surrogate that proposed it.

## §7 VERDICT SCOPE

- **§4 separability — INSTANCE-level fact, not a negative.** It is a structural property of this
  vehicle (tokens are the renderer input), measured, not a failed formulation. It does not close any
  family; it re-routes the harvest to §5's class and to cr1's reformulation #2.
- **§5 base rule — POSITIVE, FORMULATION-scoped.** −2,781 B is the argmin of the
  `α + circdist^p` family on this field at `levels=16`, `shared_base`. **NOT tested:** other level
  counts, other geometries, a real-bytes coordinate descent (would bound the lever from above —
  1.85 s/encode makes a full 3,072-cell search 23.7 h, so it was not attempted), or joint
  base+context optimisation.
- **§1 codec adjudication — FAMILY-scoped** across v4b/v4c/v4d (they share the token member).
- **NOT tested, do not infer:** any seg or pose consequence of anything here (all §5 recodes are
  lossless, so there is none by construction); across-seed variance (single seed throughout).

---

## Observability surface

**Inspectable per layer:** `FieldGap` exposes each leg (base, events, interaction) separately;
`framed_bytes` returns header/base/delta independently. **Decomposable per signal:** every byte delta
is reported alongside its ΔS on the contest denominator. **Diff-able across runs:** `--json-out`
emits the full race and sweep tables. **Queryable post-hoc:**
`.omx/research/ddm_rh1_token_rate_decomposition_20260801.json`. **Cite-able:** each field is loaded
from a named archive path; the DR7T path decodes at `verify="canonical"`, which is itself the codec
proof. **Counterfactual-able:** `propose_base` makes the base a free variable, so any base rule can
be priced against the shipped ruler without touching the coder.

## Wire-in (6 hooks)

1. **sensitivity-map** — ACTIVE: the SMEVR base rule is a newly-measured rate lever (−2,781 B on the
   shipped field); token event-rate ↔ seg-only training is a measured coupling (×0.739).
2. **Pareto constraint** — ACTIVE: token-field changes are **not** separable into rate and pose legs;
   any candidate that alters the token field must carry all three terms jointly.
3. **bit-allocator** — ACTIVE: base rule `α=0, p=2.0` on the v4d field (343,697 vs 346,478 B).
4. **cathedral autopilot** — N/A: nothing promoted, nothing dispatchable (a receiver change gates §5).
5. **continual-learning posterior** — ACTIVE via this memo + the §6 surrogate law.
6. **probe-disambiguator** — N/A: the controlled hybrids in §3 **are** the disambiguator, and the
   family race in §5 is its own control (the incumbent is a point inside the swept family).
