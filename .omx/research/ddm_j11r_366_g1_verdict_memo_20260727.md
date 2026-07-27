---
schema: claude_arm_findings_v1
lane_id: lane_ddm_j11r_366_g1_reopener_verdict_20260727
task: "#714 — j11r (1:1 replacement of quota-killed codex arm), the #366 G1 reopener"
charter: .omx/tmp/codex_prompts/ddm_j11r_366_opening_proposal_decomposition.md
research_only: true
score_claim: false
pointer_moved: false
main_review_required: true
verdict: G1_SPLIT_LEG_NOT_CLEARED_RANK1_NULL__PC1_JOINT_OPENING_RECEIPT_EXISTS__CONTINUATION_NOT_ESTABLISHED
verdict_scope: INSTANCE — sealed 4-ray J10 proposal set × W_joint-step-50 source; consumed measurements [macOS-CPU frozen-scorer advisory]
---

# DDM j11r — #366 G1 reopener verdict (verification-and-verdict arm; $0, no new scorer runs)

## Disposition — why this arm measured nothing new

The charter's ONE QUESTION ("does the pose-null/seg-null split realize EXACT joint ΔS<0 at
the opening window?") was **already answered by the measured record between the charter's
issuance and this arm's execution**: j11 (codex) landed the typed custody blocker
(`BLOCKED_J11_PROPOSAL_DECOMPOSITION_CUSTODY_PRECONDITION`, receipt SHA `25f092d3…`), and
**j12 executed j11's exact 4-step reopener in full** — SHA-bound receiver-coordinate Pose6 +
rank-4-inner Seg Jacobians per sealed proposal, byte-identical PC1 active-zero adapter,
project→integer-realize→parse-back→exact-price of all 16 singles + 8 composites from both
named bases, and the conditional 4-step live/EMA smoke. The j12 chain is **merged to main**
(mr1 harvest `aa83f79b21`, producer `tools/run_ddm_j12_receiver_coordinate_custody.py`,
independent-approver review memo included). Re-running that window would be settled
duplication (the ALREADY-SETTLED binding; rediscovery is the cardinal signal-loss sin).

j11r's honest remaining work, executed here: (1) verify the j12 custody against main
(receipt SHA `71b4ce59…` re-hashed and MATCHED; merge ancestry confirmed); (2) re-derive
every S-arithmetic row from the raw (d_seg, d_pose, bytes) triples per the per-landing
pantheon discipline — **all residuals ≤ 2.3e-15**; (3) re-anchor per card §9 (the 0.172 bar
/ #613 box, never 0.19108); (4) issue the typed G1 verdict the #710 card consumes.

## THE ONE QUESTION — answered (from receipts, re-derived)

**NO — the split itself cannot realize ΔS<0 on the sealed proposal set, and the reason is
structural, not a measurement failure.** On all four sealed 1-D proposal rays
`W(α)=W0+α·δ_p`, the receiver-coordinate Pose6 Jacobian and the rank-4-inner Seg Jacobian
are each **rank-1 with nullity 0** (exact positive Grams). Both null projectors are exactly
`[0]`: every pose-null seg component and every seg-null pose component of every sealed
proposal is identically zero. All 16 singles priced active-zero (ΔS = 0) under the strict
realized joint rule (no fixed R*; break-even ratio 1.0).

**Which projector leg failed: BOTH — degenerately.** A 1-D ray either lies inside a rank-1
map's null space (it does not; Grams positive) or projects to zero. INSTANCE scope: sealed
4-ray set × step-50 source. This is NOT a family negative for higher-dimensional proposal
families (multi-coordinate proposals can have nontrivial null components; that family is
open).

**The exact ΔS<0 opening-window receipt EXISTS — via the composed/joint leg.** The charter's
composite (`pose-null-seg + PC1-pose`) degenerates to the pure source-preserving PC1
pose-quotient carrier (the pose-null seg addend is zero). Its accepted-step window from the
ws3-arbitrated warm start (W_joint step-50, SHA `2a2c0367…`, 138,813 B) — exact n600,
[macOS-CPU frozen-scorer advisory]:

| step | bytes | d_seg | d_pose | S (advisory) | Δd_seg vs prev | Δd_pose vs prev | ΔS vs prev | ΔS vs source |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (source) | 138,813 | 0.069742771 | 35.499821 | 25.908103 | — | — | — | 0 |
| 8 | 139,693 | 0.064724189 | 31.761262 | 24.387124 | −0.005018582 | −3.738559 | **−1.520979** | −1.520979 |
| 16 | 139,701 | 0.062731959 | 28.159128 | 23.146899 | −0.001992230 | −3.602134 | **−1.240225** | **−2.761204** |

Component split 0→16: seg_term −0.701081 · pose_term −2.060714 · rate_term +0.000591.
From W_seg (base `264a09ab…`, S 40.762030) the unique composite (`b8013857…`) realizes
**ΔS −3.571143** (S 37.190887). Re-derivation residuals: 1.1e-15 / 2.2e-15 / 2.2e-15.

**Continuation is NOT established:** the required 4-step live J10 continuation from the
W_joint+PC1 endpoint **regressed +0.127593** (accepted=false); short-horizon EMA was
byte-identical (ΔS 0.0). Unsplit-opening baseline (the charter's "inert-or-ascending"
citation, j10/j11 receipts): all four sealed proposals ascend from step-50 under pure
pricing (+0.00994 / +0.04382 / +0.06532 / +0.19196).

## G1 verdict (typed, for the #710 card)

1. **SPLIT LEG: NOT-CLEARED** — typed reason `RANK1_RAY_NULL_PROJECTORS_ZERO`, both legs,
   INSTANCE scope (sealed 4-ray set × step-50 source).
2. **OPENING WINDOW: exact ΔS<0 receipt EXISTS** — PC1 joint carrier, −2.761204 (W_joint,
   16 accepted steps) / −3.571143 (W_seg), consumable as the G1 opening receipt.
3. **FIRE GATE: NOT satisfied by this alone** — D3 requires measured opening ΔS/hour vs the
   solve-line comparator; the continuation regression + `PREPARED_REVIEW_REQUIRED` reseal
   state (no J12 resealer profile) remain owed; #710 HOLD stands; FIRE is MAIN-only.
4. **Named reformulation (already routed by MAIN, confirmed here):** openings at this
   endpoint are JOINT/COMPOSED moves — the j12×pf3 twin convergence (routing card §3/§6):
   PC1 tube-finish on the descent line; the §8 composition wave (cb1 MyCar −0.0516 ·
   wf7 −1,776 B · c1 waterfill co-measure, non-additive pools) on the describe line.

## Re-anchor (card §9 binding; charter binding)

All arithmetic above is vs the **0.172 bar / #613 box**, never 0.19108. Absolute S at the
best window endpoint is **23.146899 [macOS-CPU frozen-scorer advisory] — ~22.97 S above the
0.172 bar**. The opening receipt clears G1 bookkeeping on its joint leg; it is not
competitive progress. HONEST STATE LINE: best submittable exact row 0.19108 [contest-CPU],
0.019 behind the 0.172 bar; nothing in this landing changes that.

## Receipt

`.omx/research/ddm_j11r_366_g1_verdict_window_receipt_20260727.json`
SHA-256 `de53a8888bcf1a4a8e17e6fba2dfbe818ac04115766a0fc9971d80ba99d239ec`
(binds: j12 compact `71b4ce59…` [re-hashed MATCH] · j12 full `0135dca6…` · pricing
`e73090c4…` · smoke `695c7943…` · Jacobian index `803b1101…` · j11 refusal `25f092d3…` ·
mr1 merge `aa83f79b21` · per-step re-derived rows + residuals).

## STORES CONSULTED

Charter `.omx/tmp/codex_prompts/ddm_j11r_366_opening_proposal_decomposition.md` · CLAUDE.md
(ALREADY-SETTLED, NO-FAKE, verdict-scope ladder, §9 re-anchor via routing card) · optimal-start
card §18–§23 (`optimal_start_card_366_refoundation_20260725.md`) · routing ledger
`council_coherent_optimal_path_routing_20260725.md` §1–§9 (D1–D4, j12 §3 review, §7 mr1
harvest, §9 operator re-anchor) · j11 findings + premise falsification + refusal receipt ·
j12 findings + compact receipt (re-hashed) · j10/ws2/ws3/ws4 chain (via card §19/§21/§22
re-derivations + ws3 KEEP_WJOINT row) · main git ancestry (`aa83f79b21` confirmed ancestor)
· pricing/marginal law dS/dd_pose = 5/√(10·d_pose) (re-computed: 0.265373 @35.50 ·
0.130693 @146.36, MATCH receipt). Quarantine: HARVEST-SIGNAL-ONLY respected — no banned
lineage bytes/scores consumed; PR130 0.172 used only as the external bar per §9.

## DAG FEED

FEED-603-j11r: G1 split-leg NOT-CLEARED (RANK1_RAY_NULL_PROJECTORS_ZERO, both legs,
INSTANCE) · PC1 joint opening receipt −2.761204/−3.571143 exact n600 advisory VERIFIED
against main custody (residuals ≤2.3e-15) · continuation regression +0.127593 re-confirmed
as the standing blocker · no new measurement (ALREADY-SETTLED honored) · #710 HOLD intact.
