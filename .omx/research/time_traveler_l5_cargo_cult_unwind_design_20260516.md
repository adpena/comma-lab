# Time-Traveler L5 (Z6/Z7/Z8) — CARGO-CULT-UNWIND DESIGN

**Date:** 2026-05-16
**Substrate:** Time-Traveler L5 Autonomy (PRIORITY 4 / 5)
**Lane:** `lane_time_traveler_l5_autonomy_substrate_20260513` + sister L1 design lanes (`lane_time_traveler_l5_macos_cpu_smoke_execution_20260513` L2 macOS-CPU advisory)
**Recipe:** `.omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml`
**Trainer:** `experiments/train_substrate_time_traveler_l5_autonomy.py`
**Existing design memos:** `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md` + `.omx/research/grand_council_maximize_value_with_time_traveler_seat_20260514.md`
**Audit source:** `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.4 (commit `3768a4f3d`)
**Operator approval:** "fix all also" directive 2026-05-16

## Operating-within assumption-statement (per Catalog #292)

The assumption I am operating within: *"Time-Traveler L5's 5-design-move composition arithmetic (Rao-Ballard + Atick-Redlich + foveation + world-model + Tikhonov) assumes ADDITIVE ΔS via composition of per-move improvements. Per Boyd's Dykstra-feasibility lens + Z1 ablation 2026-05-14 empirical confirmation of within-class saturation, the composition is at best SUBADDITIVE in convex-intersection regime. The DESIGN-time UNWIND is to rewrite the composition arithmetic as a polytope intersection projection (NOT sum) and apply the Z1 within-class haircut."*

HARD-EARNED basis: Z1 ablation (Catalog #219) empirically confirmed within-class density saturation at A1; per Catalog #227 sister gate, class-shift claims require Tier C empirical evidence. The Assumption-Adversary seat would challenge: *"Is the 5-move composition itself a cargo-cult novelty claim, OR are the 5 moves genuinely independent constraint axes whose intersection produces a non-empty feasible polytope?"* — answer: the analytical Dykstra check IS the disambiguator; the composition could be subadditive but still non-empty.

---

## HARD-EARNED PRESERVED

1. **5 first-principles design moves are individually well-cited** — Atick-Redlich 1990 cooperative-receiver, Rao-Ballard 1999 predictive coding, Gibson 1950 foveation-FOE, world-model framework, Tikhonov regularization. Each move INDIVIDUALLY has empirical basis.
2. **macOS-CPU advisory + Linux x86_64 GHA paired sufficiency** — Catalog #192 + #197 enforce the advisory-vs-promotable contract correctly. macOS-CPU is NEVER 1:1 contest-CI hardware but IS a high-throughput dev-loop signal.
3. **Independent canary status** — Catalog #173: time-traveler doesn't share HNeRV-family failure surface (cooperative-receiver + predictive-coding + foveation; not a mask/renderer derivative). Correct architectural classification.
4. **Modal A100 min_smoke_gpu per Catalog #215** — A100 after SIREN T4 timeout finding 2026-05-13; coordinate-MLP / NeRV-family substrates need A100 minimum.
5. **TT5L recovered 25ep CUDA evidence already exists** — `time_traveler_recovered_tt5l_25ep_exact_cuda_evidence_row_20260515_codex.json` is empirical evidence at the lower-ep training regime.
6. **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"** — paired Linux x86_64 GHA + NVIDIA A100 required before frontier-claim language.

---

## CARGO-CULTED UNWOUND

### CC-1: "Single-archive TT5L packet at 95-110 KB target size is achievable" (MEDIUM-RISK; UNWIND)

- **Source:** existing design memo target (composition arithmetic)
- **Classification:** **CARGO-CULTED** — target derived from 5-design-move byte-budget composition; not empirically anchored.
- **UNWIND DISPOSITION:** ADD §predicted-byte-budget revision: each of the 5 design moves contributes an INDEPENDENT BUDGET ceiling (NOT floor). The composition byte budget is `max(per_move_budget)` (worst-case dominates) NOT `sum(per_move_savings)` (best-case additive). Revise target from 95-110 KB to `[95 KB ceiling pending paired Dykstra-feasibility]`.
- **Reactivation criterion:** Dykstra polytope emitted; revised target = polytope's distortion-axis projection lower bound.

### CC-2: "Five first-principles design moves compose additively for ΔS" (HIGH-RISK; HIGHEST-PRIORITY UNWIND)

- **Source:** existing design memo §composition arithmetic
- **Classification:** **CARGO-CULTED** — Dykstra-feasibility says composition is at best SUBADDITIVE in convex intersection regime. Z1 ablation 2026-05-14 EMPIRICALLY confirmed within-class saturation. The 5-move composition is mathematically convex-intersection (NOT additive) — the ΔS predicted by summing per-move improvements is structurally an OVER-ESTIMATE.
- **UNWIND DISPOSITION:** REWRITE design memo §composition-arithmetic as Dykstra-feasibility intersection:
  ```
  achievable_polytope = projection(rate_budget) ∩
                        projection(distortion_budget) ∩
                        ⋂_i projection(per_move_constraint_i)
  ```
  The 5-move composition is the INTERSECTION not the SUM. ΔS prediction must be the projection of the polytope's centroid onto the distortion axis (or its lower bound). The SUBADDITIVE penalty per move is a 5-move geometric factor; document this explicitly.
- **Reactivation criterion:** Dykstra-feasibility polytope emitted; predicted target revised to polytope intersection lower bound.

### CC-3: "macOS CPU advisory + Linux x86_64 GHA paired is sufficient compute envelope" (LOW-RISK; CONFIRMED HARD-EARNED)

- **Source:** existing design memo + recipe
- **Classification:** **HARD-EARNED** — Catalog #192 + #197 enforce the advisory-vs-promotable contract correctly.
- **UNWIND DISPOSITION:** NO CHANGE. Existing Catalog #192/#197 contracts hold; macOS-CPU advisory is a structurally legitimate first-pass signal.

---

## PROBE-DISAMBIGUATOR

- **Name:** `tools/check_substrate_dykstra_feasibility.py --substrate time_traveler_l5_5move` (SHARED with C6 + ATW + NSCS02)
- **Cost:** $0 analytical
- **Method:** alternating projections (Dykstra 1983) onto 5 convex constraint sets:
  - predictive-coding rate ≤ R_pc (Rao-Ballard 1999 hierarchy)
  - Atick-Redlich rate ≤ R_AR (cooperative-receiver theorem)
  - foveation rate ≤ R_fov (Gibson 1950 ego-motion-matched)
  - world-model rate ≤ R_wm (Markov-1 ego-pose prior)
  - Tikhonov regularization rate ≤ R_tik
  intersected with contest rate budget (`25 · archive_bytes / 37,545,489`); emit intersection polytope vertices
- **Disambiguates:** CC-2 (composition additivity vs subadditivity) + CC-1 (byte budget ceiling) simultaneously
- **Output:** `.omx/state/dykstra_feasibility_time_traveler_l5.json`; updates cathedral autopilot rate-prediction posterior per Catalog #128

**Sister probe (already executed):** `recovered_tt5l_25ep_exact_cuda` (codex 2026-05-15) is the existing empirical anchor at lower-ep regime; reconcile against Dykstra polytope projection.

---

## REACTIVATION CRITERIA (per CLAUDE.md "Forbidden premature KILL")

Sister L1 design lanes marked `research_only: true` until ALL of:

1. `tools/check_substrate_dykstra_feasibility.py --substrate time_traveler_l5_5move` emits polytope at `.omx/state/dykstra_feasibility_time_traveler_l5.json`.
2. Predicted target revised to polytope intersection lower bound (NOT 5-move sum) in BOTH this memo AND recipe.
3. Z1 within-class haircut applied per Catalog #219 (floor ΔS at -0.005 if Tier C classifies WITHIN-CLASS).
4. Council CONSENSUS per CLAUDE.md "Design decisions — non-negotiable" sextet pact on the revised composition arithmetic (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary; Rao-Ballard + Atick-Redlich grand council seats advisory).

L2 active dispatch lane (`lane_time_traveler_l5_macos_cpu_smoke_execution_20260513`) remains at macOS-CPU advisory; Linux x86_64 GHA paired smoke deferred to post-Dykstra revision.

NO KILL.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical | TF32 + CUDA discipline shared |
| Scorer loss helper (`score_pair_components`) | ADOPT canonical | Catalog #164 |
| eval_roundtrip | ADOPT canonical | CLAUDE.md non-negotiable |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE" |
| Archive grammar (`monolithic_tt5l_replaces_hnerv_a1_substrate`) | UNIQUE FORK | Substrate-specific: world model Stage 1 (~55-70 KB) + per-pair side info Stage 2 (~25-35 KB at 45 B/pair) + AC state Stage 3 (~10 KB) + header Stage 4 (~2 KB) |
| World-model architecture (Markov-1 ego-pose prior) | UNIQUE | First substrate-distinguishing primitive |
| Predictive coding hierarchy (Rao-Ballard 1999) | UNIQUE | Second substrate-distinguishing primitive |
| Foveation matched to ego-motion (Gibson 1950 + LAPose) | UNIQUE | Third substrate-distinguishing primitive |
| Cooperative-receiver primitive (Atick-Redlich 1990) | ADOPT canonical | `src/tac/codec/cooperative_receiver/atick_redlich.py` is shared with ATW + Z4 |
| Differentiable physics renderer | UNIQUE | Sub-60K params with SIREN-style init |
| Tikhonov regularization | ADOPT canonical | Standard regularization primitive |
| Inflate runtime (`select_inflate_device`) | ADOPT canonical | Catalog #205 |
| Auth eval helper (`gate_auth_eval_call`) | ADOPT canonical | Catalog #226 |
| Hardware substrate detection | ADOPT canonical | Catalog #190 |
| Modal A100 min_smoke_gpu | ADOPT canonical | Catalog #215 (NeRV-family floor) |

**Bolt-on vs substrate-engineering split per HNeRV parity L7:** This substrate is **substrate-engineering** (NEW architecture class composing 5 design moves). LOC budget exceeds bolt-on cap explicitly (63 KB trainer file). The 5 UNIQUE / UNIQUE FORK decisions ARE the substrate-optimal engineering surface.

**5-move composition note (CC-2 unwind):** the layers above are decomposed into INDEPENDENT design moves for the canonical-vs-unique table. The Dykstra-feasibility analysis treats each as a convex constraint axis; the achievable region is their INTERSECTION (projection geometry), NOT their SUM (additive arithmetic).

---

## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Status |
|---|---|---|
| 1 | UNIQUE-AND-COMPLETE-PER-METHOD bind | YES — 63 KB trainer file binds all 5 moves + archive + inflate; LOC exceeds 350 cap (substrate-engineering exception) |
| 2 | Canonical-vs-unique decision per layer | YES — landed (this memo §above) |
| 3 | HARD-EARNED-vs-CARGO-CULTED classification | YES — landed (this memo §above) |
| 4 | Probe-disambiguator per defensible interpretation | YES — Dykstra polytope + existing TT5L 25ep recovered anchor |
| 5 | Premise verification per Catalog #229 | YES — audit blueprint + recipe + existing design memos + Z1 ablation + recovered 25ep anchor |
| 6 | 6-hook wire-in or N/A rationale per Catalog #125 | DESIGN-ONLY at landing; runtime wire-in trainer-side per existing codex hardening commits |
| 7 | Predicted ΔS band with citation | YES — REVISED from [0.150, 0.170] additive-sum to NULL pending Dykstra polytope (this memo §Predicted band) |
| 8 | Reactivation criteria pinned | YES — 4 criteria landed (this memo §above) |
| 9 | Sister-subagent ownership map per Catalog #230 | YES — DESIGN-only; Dykstra tool implementation + trainer composition rewrite = operator-decision-required |

---

## Predicted ΔS band (per proposed Catalog #296)

**Predicted ΔS band:** `NULL pending Dykstra-feasibility 5-move polytope + Z1 within-class haircut application` [prediction]

**Existing recipe `predicted_band: [0.150, 0.170]`** — to be REVISED to `null` per CC-2 unwind. The literature-prediction synthesis IS an over-estimate per Dykstra-feasibility lens; the band needs polytope-projection-based revision.

**Composition arithmetic correction:**

OLD (CARGO-CULTED additive): `ΔS_TT5L = ΔS_pc + ΔS_AR + ΔS_fov + ΔS_wm + ΔS_tik`

NEW (HARD-EARNED Dykstra-intersection): `ΔS_TT5L = projection_distortion_axis(⋂_i polytope_i ∩ rate_budget_polytope)` where each per-move polytope is a convex constraint set, and the achievable region is their alternating-projections intersection (Dykstra 1983).

The SUBADDITIVE penalty per move is a 5-move geometric factor; the polytope projection's distortion-axis lower bound is the TRUE predicted ΔS lower bound.

**Z1 within-class haircut citation:** Per Catalog #219, if Tier C MDL ablation on TT5L archive classifies as WITHIN-CLASS (MDL density > 0.90), apply floor at ΔS = -0.005. Time-Traveler is hypothesized to be ACROSS-CLASS (substrate-class shift away from HNeRV-family) but this REQUIRES Tier C empirical evidence per Catalog #227.

**Empirical anchors available:**
- `recovered_tt5l_25ep_exact_cuda` (codex 2026-05-15): lower-ep regime empirical evidence
- macOS-CPU advisory (Catalog #192): non-promotable but high-throughput dev-loop signal
- Paired Linux x86_64 GHA: DEFERRED to post-Dykstra revision

**Council-grade tradeoff** per CLAUDE.md "Design decisions — non-negotiable": composition arithmetic rewrite (additive → subadditive) requires sextet-pact CONSENSUS (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary). Grand council seats Rao-Ballard + Atick-Redlich + Time-Traveler protégé advisory per CLAUDE.md "Grand Council" roster.

---

## Cross-references

- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.4
- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md` (existing design memo)
- `.omx/research/grand_council_maximize_value_with_time_traveler_seat_20260514.md` (council deliberation)
- `.omx/research/time_traveler_recovered_tt5l_25ep_exact_cuda_evidence_row_20260515_codex.json` (existing 25ep empirical anchor)
- `.omx/research/time_traveler_l5_*_hardening_20260516_codex.md` (4 codex hardening landings)
- CLAUDE.md Catalog #125 / #128 / #164 / #173 / #190 / #192 / #197 / #205 / #215 / #219 / #226 / #227 / #240 / #270 / #287 / #290 / #292 / #294 / #296

---

**Status:** UNWIND-LANDED 2026-05-16 (DESIGN-only; Dykstra tool + trainer composition rewrite = operator-decision-required; sextet-pact council CONSENSUS required).
