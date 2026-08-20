# ddm_rr1 — DEEP COMPREHENSIVE RECURSIVE RECALL: v7–v18 (+adjacent) tools/levers vs the PR130 BASE

Operator 2026-08-09: *"We have a lot of related tools and levers and more from old vehicles
especially v7-18"* + *"Need to do deep comprehensive recursive recall."* Sister steers this turn:
*"used their token stream and learned prior approach in HPAC, but around all of that, we have some
room to experiment and explore"* · *"The big issue is beating their rate while still crushing
distortion"* · *"cosine is pretty much never optimal"* · *"pretty much every problem you run into
we have a solution for or there's a solution in PR130 or other PRs."*

RECALL, DO NOT RE-DERIVE. Cite receipts (path + commit/sha + the measured number). Where a thing
was never measured, say ABSENT — do not invent. Honest absence beats a plausible number.

## THE OBJECT YOU ARE RECALLING AGAINST (measured today, 2026-08-09)

BASE = PR130 CPR1, `S = 0.172141297491896447` `[contest-CUDA, DALI GT, n600]`, archive 191,052 B
sha `0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd`. Reproduced BYTE-IDENTICAL
here (`.omx/research/ddm_pr130_reproduce_20260809/`, commit 12031094d9).

Raw distortions (derived from the contributions): **d_seg 2.8609e-04 · d_pose 1.967006e-05.**
Contributions: seg 0.028609 (16.62%) · pose 0.014025 (8.15%) · rate 0.127214 (**73.90%**).

**sub-0.15 needs ΔS −0.0221413 = −33,252 B by rate alone (−17.4% of the archive).**
- d_pose alone is IMPOSSIBLE (whole contribution 0.014025 < 0.0221413 even at d_pose=0).
- d_seg alone needs 2.8609e-4 → 6.4677e-5 (4.4×).
- So RATE must carry most of it.

### THE THREE AXES, by measured distortion-coupling

| axis | bytes | share | coupling |
|---|---:|---:|---|
| **A1 tokens** (HPAC-coded; tokens ARE the exact GT SegNet argmax) | 116,980 | 61.26% | **ZERO** — lossless AR coder; better prior = fewer bytes, IDENTICAL tokens |
| **A2 hpac prior** (the AR model itself) | 20,179 raw | 10.56% | **ZERO** — pure model↔token exchange |
| **B semantic renderer** (quant_bits → the partition it emits) | 40,252 raw | 21.07% | **COUPLED** — this is the real RD curve, owns d_seg |
| **C pose carrier** | 23,054 raw | 12.07% | representation-only (MEASURED: every coder LOSES; 23,054 → 23,058) |

A1+A2 = 137,159 B = **71.8% of the archive with ZERO distortion coupling.** Reaching sub-0.15 from
that axis alone = 0.7576× → bpp 0.007933 → **0.006010**.

### Already measured on this base TODAY (do not re-measure; build on)
- OUR HPAC (`train_hpac_self_compress.py` on our GT labels) beats theirs: tokens **114,997 vs
  116,980 = −1,983 B**, joint 135,289 vs 137,159 = **−1,870 B**, ΔS −0.0012452
  (`.omx/research/ddm_hb3_20260808/`). bpp 0.0077987 vs 0.0079332.
- Lossless model recode: **−903 B** (split-stream + per-section brotli q11); brotli-free −234 B
  (`RATE_AXIS_LOSSLESS_RACE.md`).
- **Token stream is at HPAC's model entropy** (+5 B under brotli) → the lever is the MODEL or the
  TOKEN CONTENT, never a generic coder.
- **Our whole generic-coder lineage LOSES to HPAC by 2.2–3.6×** on the dense partition
  (SMEVR/CAE-INTER/KT-backoff/brotli/lzma/rANS) — `CODER_LINEAGE_VS_HPAC.md`. That cell is SHUT.
- Rate is provenance-INSENSITIVE (0.047% across DALI/AV/local label fields) — byte comparisons are
  safe; the DALI-vs-AV delta is a DISTORTION confound only (`ADDENDUM2_...md`).
- The **model↔token exchange rate d(tokens)/d(model) is UNMEASURED** and gap-sized.

## WHAT TO RECALL — recursive, three passes minimum

**Pass 1 — INVENTORY.** Every tool/lever/law/receipt from v7–v18 (and adjacent: g1–g4, ms1–ms7,
pf1–pf3, rd1, c1, e1–e5, dm1–dm4, v19/v19b/v19c, r7, wr1, sp1, tb1/TR1, hp1, xi1) that bears on
A1/A2/B/C above. Seed surface (found by MAIN, NOT exhaustive — extend it):
`src/tac/pr86_hpac_codec.py` · `experiments/ddm_hp1_learned_ar_prior_race.py` ·
`experiments/ddm_xi1_carried_xi_coder.py` + `ddm_xi1_carried_xi_race.py` ·
`src/tac/boundary_math/{context_partition_codec,witness_crosstensor_codec,weight_entropy_penalty_mlx,
keyframe_codec,curve_relative_offset_coder,defect_network_rate_code,movable_site_coder,xi_pose_coder}.py` ·
`src/tac/losses/rate_surrogate.py` · `src/tac/codecs/stc_dasher/` ·
`src/tac/analysis/hprc_saliency_rd_allocation.py` · `src/tac/optimization/ddm_lp1_layer_pricing.py` ·
`src/tac/anr_token_renderer.py` · `src/tac/categorical_substrate.py` ·
`src/tac/witness_dsl/{optimal_basis_20260714,taskspace_predictor_state_v2}.py` ·
`src/tac/stbm1br_mask_codec.py` · `src/tac/pr101_split_brotli_codec_derivers.py`.
Also the DSL lever registry (`tac.witness_dsl.lever_registry.completeness()`) and
`tac.canonical_equations` registry.

**Pass 2 — RECURSE.** For every hit in pass 1, follow its OWN cited receipts/consumers/successors
one more hop. The g2 "information LEDGER" (invisible / ξ-predictable / chart-expressible /
irreducible) and g4's "**free decoder-derived spatial context-model coding gain**" are named
high-value seeds — a context model the DECODER DERIVES costs ZERO counted bytes and is a direct
A2 improvement at zero model cost. Chase them to their receipts.

**Pass 3 — LOOP UNTIL DRY.** Repeat until a full pass yields zero new rows. Report the round count.

## DELIVERABLE — ONE ranked table, plus honest negatives

Per row: `tool/lever` · `where it lives (path)` · `what it actually does` · `MEASURED receipt
(path + number) or ABSENT` · `which axis (A1/A2/B/C)` · `applicability to PR130's ACTUAL objects
(their token field / their prior / their renderer / their pose carrier)` · `FIRE ORDER + the
falsifier that would kill it`.

Rank by **expected ΔS per unit of work on the DECOUPLED axis first** (A1/A2 — zero distortion
cost), then B, then C.

Call out explicitly:
1. Anything already BUILT that has NEVER been fired against a PR130-class object.
2. Anything whose NEGATIVE was scoped to a retired vehicle and whose precondition has now MOVED
   (the conditional-validity law #755/rv1) — PR130's dense-semantic token field is a DIFFERENT cell
   from our IX2TOK01 latent tokens; sv2/#859's SMEVR negative is explicitly cell-scoped.
3. Anything that would let the DECODER derive context for free (rule-118: generic algorithm in
   inflate.py = FREE; video-derived/learned content = COUNTED).

## OPTIMAL FORM
- Reference form: this is a RECALL/AUDIT arm, not a mechanism build. Its optimal form is
  EXHAUSTIVE COVERAGE + receipt-bound rows, and the reduction vs that is SCOPE only (v7–v18 +
  named adjacents), never MECHANISM.
- No measurement is required of you and none is authorized: this arm is scorer-free and
  launch-free. Do NOT train, do NOT dispatch, do NOT fire a scorer. If a row needs a measurement,
  NAME it with its falsifier and its cost.
- Provenance pins: cite path + commit sha for every code claim; path + number for every receipt.

## CONSTRAINTS
- READ-ONLY on `upstream/` and on every `*_intake_*` public-PR clone.
- `score_claim=false` on everything; MPS/MLX are never authority.
- Never invent CLI flags/APIs — grep `add_argument` / `def` before citing any interface.
- Land ONE durable memo under `.omx/research/ddm_rr1_20260809/` + commit via
  `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256`, tags
  `[no-triality] [p0-ledger-ok]`. NO Claude/AI attribution, no Co-Authored-By trailer.
