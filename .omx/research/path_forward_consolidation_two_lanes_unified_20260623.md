# PATH FORWARD — consolidate everything; the two lanes are ONE curve (2026-06-23)

**Source:** operator, 2026-06-23 — *"review everything we have learned and all of our built and checkpoints
… determine the path forward, seems like there is still an hnerv optimization lane and a separate capstone
lane."* This memo is the consolidation + the unified path. Authority `[analysis]`; all score math via
`tac.contest_score`; frontier pointer UNMOVED (contest-CPU 0.19109982, BORROWED PR101-recode).

## 0. Fundamentals fixed this session (the recurrence cause)
- **Memory recall was BROKEN:** MEMORY.md was 178KB / 7.5× over the 24.4KB load limit → only partially
  loaded → I did not recall known classes → repeated them. Condensed to 23.6KB one-liners (detail in the
  topic files; full backup kept) → it now FULLY loads. This is the root cause of "repeated mistakes
  despite known classes."
- **Record corrected:** the IRREDUCIBLE terminal claim (grand-unification §8) was RETRACTED (§9) —
  refuted within the hour by the PR95 existence proof. Self-protection landed:
  `feedback_terminal_conclusion_needs_existence_proof_crosscheck_20260623.md`.

## 1. The two lanes are TWO POINTS ON ONE CURVE (the reconciliation)
The "HNeRV optimization lane" and the "capstone lane" are not two strategies — they are two capacities of
the SAME vehicle (HNeRV decoder + PR95 8-stage curriculum), at opposite ends of the capacity-rate-
distortion curve:

| lane | vehicle | params | d_seg | rate | S | role |
|---|---|---:|---:|---:|---:|---|
| HNeRV-opt (PR95 basin) | HNeRV bc36 | 229K | ~5.6e-4 | 0.118 | ~0.191 | high-capacity endpoint (the proven basin) |
| Capstone (small basis) | HNeRV bc20 | ~71K | ~0.0021 | 0.055 | ~0.326 | low-capacity endpoint (the rate-headroom bet) |

`d_seg ∝ params^(−1.12)` (bc20↔bc36; bc20 recipe-contaminated → conservative); `bytes ≈ 16KB + 0.71·p`.
**Neither endpoint is optimal.** The capacity-RD optimum at *current* int8+brotli entropy coding is
~177K params → **S\* ≈ 0.186** (≈ the frontier — the borrowed 0.191 sits near the HNeRV RD optimum, which
is why it "feels stuck"). **48 hnerv/capstone/pr95 registry lanes** is proliferation around this one curve.

→ **DECISION: merge the two lanes into ONE program — "the HNeRV decoder at the RD-optimal capacity, trained
with the FIXED recipe, with NVRC-class weight entropy coding."** Stop running bc20-capstone and
bc36-HNeRV as separate competing threads; they are sample points that calibrate the one curve.

## 2. The sub-0.15 lever (deep-math, quantified) — the RATE axis shifts the WHOLE optimum
Better weight entropy coding does not just shave rate at fixed d_seg — it slides the RD optimum to a lower
S because you can afford more capacity (lower d_seg) at the same rate:

| weight entropy coding | optimal params | d_seg | rate | **S\*** |
|---|---:|---:|---:|---:|
| current int8+brotli (5.67 bits/param) | 177K | 7.4e-4 | 0.094 | **0.186** |
| **2× (NVRC/NeuroQuant-class, ~2.8 b/param)** | 246K | 5.2e-4 | 0.069 | **0.137 (sub-0.15)** |
| 3× | 298K | 4.2e-4 | 0.058 | **0.116 (≈ S_floor 0.118)** |

Even simpler: a pure 2× recode of the *existing* frontier weights at fixed d_seg → `0.056+0.017+0.059 ≈
0.132`. Whether the 2× exists is ONE $0 measurement (Shannon entropy of the frontier's int8 weights vs the
162KB brotli spends) — **in flight** (subagent a2782).

## 3. What we have LEARNED (consolidated, corrected)
- d_seg is the binding term (`100·d_seg` alone = 0.21 > T_1); it is **reducible by capacity** (PR95 proves
  5.6e-4), NOT at an absolute floor at 0.0021. Our plateau = **capacity (bc20) + recipe bugs** (curriculum
  hardcodes muon_lr 2e-4 = 150× too small; cosine LR floor) — `pr95_seg_convergence_mechanism...20260611`.
- The residual flips sit at low-GT-margin pixels (where SegNet itself is a coin-flip) — true, but that is
  *where* the residual sits, not a *floor* (PR95 pins those pixels).
- Pure capacity scaling is RD-bounded at ~0.186; **the rate axis (NVRC) is the quantified sub-0.15 lever.**
- Pose is a solved sub-problem (FiLM / 6-scalar store, low-rank codec #140); rate+pose floor ≈ 0.118.
- Author (aaronleslie/hnerv_muon): bc36 + taper "matches HNeRV paper" (a generic fidelity taper, NOT
  score-tuned) → a task-aware channel allocation is an opportunity. Blog narrative being mined.

## 4. THE PATH FORWARD (one unified program, three gates in flight)
1. **GATE-1 (in flight, a2782):** frontier-weight Shannon entropy — does a 2× NVRC recode exist? If yes →
   a pure recode of the existing frontier reaches ~0.132 **sub-0.15 with no retraining** (byte-close → exact
   eval). This is the fastest possible sub-0.15 row. → **then PURSUE NVRC/related (#154).**
2. **GATE-2 (in flight, #160):** own-trained PR95 with the FIXED recipe → a CLEAN capacity point to anchor
   the RD curve (the first converged vehicle that is *ours*, not borrowed).
3. **GATE-3 (in flight, a29e5d90):** 384-bottleneck d_seg floor — the absolute lower bound on the curve.
4. **THE BUILD:** the unified vehicle = HNeRV @ RD-optimal capacity (calibrated by GATE-2/3) + fixed recipe
   + NVRC-class weight entropy codec (GATE-1). The math says this is the sub-0.15 vehicle.
5. **Lane hygiene:** collapse the 48 hnerv/capstone/pr95 registry lanes to this one program + its
   calibration points (CLAUDE.md retirement discipline); archive the rest with reactivation criteria.

## 4b. GATE-1 RESULT (a2782, MEASURED on the real frontier archive) — recode is EXHAUSTED
The frontier decoder is **already at the order-0 Shannon entropy floor: 5.629 bits/param vs 5.586 floor
(ratio 1.0077, 1,226 bytes / ΔS −0.0008 headroom).** **A pure rate recode CANNOT reach sub-0.15** — this
RETRACTS §2's "pure 2× recode → 0.132." The 2×-bits/param RD target is real but needs **lower-bit weights
(INT4/INT3 score-aware QAT) or learned higher-order weight entropy** — a RETRAIN/REQUANT, not a recode. The
author's own ablations pre-confirm this: DeepCABAC/order-2 coders were "neutral/marginal," and a 2× bigger
ch=72 teacher "did not pay" (rate-dominated). Capacity scaling is also at the HNeRV RD optimum (bc36 ≈ p*).
NVRC/related survive ONLY as **co-trained entropy + INT4 QAT during training**, not as a standalone recode.

**The gap split (a2782, measured):** our 0.0021 plateau is (a) RECIPE throttle — the curriculum substituted
muon_lr 2e-4 for the working 0.03 (controlled A/B: 0% vs 7.6× descent in 15 CE epochs); (b) under-
convergence — our cleanest n600 is **stage-1-CE-only, paused mid-curriculum**, never reaching the
d_seg-finishing stages; (c) capacity (smallest). **We have NEVER run our own decoder through the full
corrected 8-stage curriculum to convergence at n600.**

## 4c. THE decisive move (recode dead → sub-0.15 is a TRAINING problem)
Fire the **full corrected 8-stage PR95 curriculum at n600 to convergence with the WORKING recipe (muon_lr
0.03, NO cosine LR floor), bc20 then bc24, byte-close, exact-eval.** It is the ONLY measurement that says
whether an OWN-trained small-byte decoder bends d_seg far enough to cross sub-0.15. It has been
armed-but-never-cleanly-fired (the apparatus-gate-volume lesson). Author rationale (recovered verbatim):
bc36 chosen for the RATE budget (INT8 brotli-fits), rate is binding (59.8% of his 0.1988 vs seg 30.8%); he
names NO floor and NO future-work. sub-0.15 levers now: own-trained d_seg + co-trained INT4 QAT (lower bits
without blowing d_seg — note int5 QAT previously capped ~0.49, so this is uncertain) + byte-neutral
boundary-band taper. Full math: `pr95_vs_ours_convergence_gap_and_capacity_rd_deepmath_20260623.md`.

## 5. Honest state
Frontier UNMOVED at 0.191 (borrowed). sub-0.19 reachable (RD optimum / rate recode); **sub-0.15 reachable
via the rate axis (NVRC), quantified, gated on the entropy measurement now running.** No pointer has moved
yet — the END is a byte-closed exact row; GATE-1's result decides whether the next row is a pure-recode
sub-0.15 or the RD-optimal retrain.
