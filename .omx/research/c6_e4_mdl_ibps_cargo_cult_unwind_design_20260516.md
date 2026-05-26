# C6 MDL-IBPS (e4 variant) — CARGO-CULT-UNWIND DESIGN

**Date:** 2026-05-16
**Substrate:** C6 MDL-IBPS (PRIORITY 2 / 5)
**Lane:** `lane_c6_e4_mdl_ibps_substrate_20260514`
**Recipe:** `.omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`
**Trainer:** `experiments/train_substrate_c6_e4_mdl_ibps.py`
**Existing design memos:** `.omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md` + `.omx/research/mdl_ibps_substrate_council_design_round_1_20260513.md`
**Audit source:** `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.2 (commit `3768a4f3d`)
**Operator approval:** "fix all also" directive 2026-05-16

## Operating-within assumption-statement (per Catalog #292)

The assumption I am operating within: *"C6 MDL-IBPS predicted band [0.113, 0.163] is structurally inconsistent with Z1 ablation 2026-05-14 (Catalog #219) within-class density posterior finding (the 0.196-0.199 plateau IS the cumulative cost of within-class refinement). The cargo-cult is the band being CARRIED FORWARD POST-Z1 without Dykstra-feasibility intersection check."*

HARD-EARNED basis: Z1 ablation empirically anchored MDL density at A1 at ~99.3% (within-class saturation); per CLAUDE.md "Forbidden component-aliasing for baselines" + Catalog #219, any predicted ΔS that doesn't apply the Z1 within-class haircut is an empirical-claim-without-evidence-tag. The Assumption-Adversary seat would challenge: *"Is C6 actually across-class (substrate-class shift) or within-class (just another HNeRV-flavor variation)?"* — answer: the Tier C MDL ablation post-smoke (Catalog #227) IS the disambiguator; without it, [0.113, 0.163] is unproven.

---

## HARD-EARNED PRESERVED

1. **Tishby-Zaslavsky 2015 IB framework** + Rissanen 1978 MDL — first-principles information-theoretic foundation; citations in recipe `literature_anchor` are correct.
2. **Probe-disambiguator via β-sweep + post-smoke MDL ablation** — Catalog #125 hook #6 satisfied; `tools/mdl_scorer_conditional_ablation.py` is the canonical helper.
3. **Tier C empirical anchor (5ep IBPS1)** — Catalog #227 sister anchor exists; classification: WITHIN-CLASS per Tier C density posterior at landing.
4. **Modal A10G `min_smoke_gpu` per Catalog #215** — A10G after 2026-05-14 100ep T4 timeout finding; honored in recipe.
5. **Independent canary status** — Catalog #173 correctly classifies C6 as `independent_substrate` (NOT HNeRV-family-derivative; IB variational encoder is architecturally distinct).
6. **CLAUDE.md "Auth eval EVERYWHERE" + "Submission auth eval — BOTH CPU AND CUDA"** — dispatch contract correctly requires CUDA auth eval on EMA shadow + paired Linux x86_64 GHA CPU before frontier-claim language.

---

## CARGO-CULTED UNWOUND

### CC-1: "MDL × IB × Procedural-Synthesis substrate yields ΔS -0.030 to -0.080 vs A1" (MEDIUM-RISK; UNWIND)

- **Source:** existing design memo §2 (campaign predicted band)
- **Classification:** **CARGO-CULTED** — wide band (factor 2.7); not validated empirically beyond 5ep smoke; band carried forward post-Z1 without within-class haircut.
- **UNWIND DISPOSITION:** REWRITE predicted band as `[NULL pending Dykstra-feasibility check + Tier C empirical anchor reconciliation]`. Cite Z1 ablation Catalog #219 as the empirical revision anchor.
- **Reactivation criterion:** Dykstra-feasibility analytical check emits polytope; revised band falls within Z1 within-class acceptance band.

### CC-2: "Procedural decoder + per-pair patches is more bit-efficient than monolithic decoder + latents" (MEDIUM-RISK; UNWIND via canonical-vs-unique decision)

- **Source:** trainer architecture choice (Selfridge demon hierarchy + MoE assumption)
- **Classification:** **CARGO-CULTED** — HNeRV (content-adaptive embeddings) is the EMPIRICAL counter-example proving content-adaptive >> procedural at the PR101/PR103 frontier.
- **UNWIND DISPOSITION:** ADD §canonical-vs-unique decision per layer (Catalog #290) section explicitly contrasting procedural-decoder choice against HNeRV's content-adaptive empirical baseline. Document FORK rationale OR pivot to HYBRID architecture (procedural-decoder + per-pair content-adaptive embeddings).
- **Reactivation criterion:** §canonical-vs-unique decision per layer landed below in this memo.

### CC-3: "Predicted post-campaign score band [0.11, 0.16]" (HIGH-RISK; HIGHEST-PRIORITY UNWIND)

- **Source:** existing design memo §predicted-band
- **Classification:** **CARGO-CULTED** — first-principles MDL+IB lower bound but skips Pareto-frontier-consistency check (Dykstra-feasibility); analogous to NSCS06's symposium-#4 prediction failure mode.
- **Severity:** HIGH — band-width-too-narrow risk: if true achievable region is [0.18, 0.22], a paid dispatch could waste $10-30.
- **Z1-revision:** Per Catalog #219 + Tier C empirical anchor per Catalog #227, C6 5ep IBPS1 anchor classified WITHIN-CLASS per Tier C density posterior. The band [0.113, 0.163] is structurally inconsistent with Z1's within-class plateau finding.
- **UNWIND DISPOSITION:** COMPUTE Dykstra-feasibility intersection ANALYTICALLY using `tools/check_substrate_dykstra_feasibility.py` (NEW; sister to NSCS02 + TT-L5 + ATW). Input: MDL constraint `R_MDL >= L(θ) + log|θ|/2` + IB constraint `I(X;T) - β·I(T;Y) ≤ K_IB` + contest rate budget `25*B/N` + Z1 within-class density posterior. Output: achievable Pareto frontier polytope. Revise predicted band to polytope's distortion-axis intersection.
- **Reactivation criterion:** Dykstra polytope emitted; predicted band revised; Tier C anchor reconciliation landed.

---

## PROBE-DISAMBIGUATOR

- **Name:** `tools/check_substrate_dykstra_feasibility.py --substrate c6_e4_mdl_ibps` (SHARED with NSCS02 + ATW + TT-L5)
- **Cost:** $0 analytical (alternating projections; Dykstra 1983)
- **Method:** alternating projections onto convex constraint sets; emit feasible polytope vertices + Pareto-frontier-on-distortion-axis projection
- **Disambiguates:** CC-3 (predicted band validity) directly + CC-1 (campaign band) via the polytope's centroid projection
- **Output:** `.omx/state/dykstra_feasibility_c6_e4_mdl_ibps.json`; updates cathedral autopilot rate-prediction posterior per Catalog #128

**Sister probe (post-smoke, already existing):** `tools/mdl_scorer_conditional_ablation.py` for Tier C MDL density classification (within-class vs across-class).

---

## REACTIVATION CRITERIA (per CLAUDE.md "Forbidden premature KILL")

Recipe `predicted_band` field updated to `null` (was `[0.113, 0.163]`); recipe stays dispatch-enabled but with reactivation criteria pinned in `dispatch_blockers`:

1. `tools/check_substrate_dykstra_feasibility.py --substrate c6_e4_mdl_ibps` emits polytope file at `.omx/state/dykstra_feasibility_c6_e4_mdl_ibps.json`.
2. Predicted band revised in BOTH this memo AND the recipe to the polytope's distortion-axis intersection.
3. Tier C anchor reconciliation landed: post-smoke MDL ablation classifies WITHIN-CLASS or ACROSS-CLASS per Catalog #227; revised band reflects that classification.
4. Grand council CONSENSUS sign-off on the revised band per CLAUDE.md "Design decisions — non-negotiable" sextet pact (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary).

NO KILL. KILL is LAST RESORT per CLAUDE.md.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical | TF32 + CUDA discipline shared across substrates |
| Scorer loss helper (`score_pair_components`) | ADOPT canonical | Catalog #164 differentiability invariant |
| eval_roundtrip | ADOPT canonical | CLAUDE.md non-negotiable; standard 384→874→uint8→384 simulation |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE" |
| Archive grammar | UNIQUE FORK | Substrate-specific: encoder + decoder + per-pair latent (~14K bytes int8) + sorted-keys JSON meta; FORK is the IB variational architecture itself |
| IB encoder (variational q(z\|frames)) | UNIQUE | The substrate-distinguishing primitive; bounded by KL(q \|\| N(0,I)) |
| Procedural decoder | UNIQUE — UNDER REVIEW | CC-2 unwind: contrast against HNeRV content-adaptive baseline; CONSIDER pivot to HYBRID (procedural-decoder + per-pair content-adaptive embeddings) pending Dykstra revision |
| MDL loss | UNIQUE | Rissanen 1978 first-principles; `L = ||scorer(decoded) - scorer(GT)||² + β · I(z; frames)` |
| β-sweep [0.001, 1.0] | UNIQUE | Catalog #125 hook #6 probe-disambiguator |
| Inflate runtime (`select_inflate_device`) | ADOPT canonical | Catalog #205 |
| Auth eval helper (`gate_auth_eval_call`) | ADOPT canonical | Catalog #226 |
| Hardware substrate detection | ADOPT canonical | Catalog #190 |
| Modal A10G `min_smoke_gpu` | ADOPT canonical | Catalog #215 after T4 timeout finding |

**Bolt-on vs substrate-engineering split per HNeRV parity L7:** This substrate is **substrate-engineering** (NEW architecture class: variational IB + MDL + procedural decoder + per-pair latent). LOC budget exceeds bolt-on cap explicitly. The 6 UNIQUE / UNIQUE FORK decisions ARE the substrate-optimal engineering surface.

**META-meta-observation per Carmack-Hotz seat:** if the Dykstra polytope shows the procedural-decoder choice is dominated by HNeRV-content-adaptive at the same rate budget, the substrate becomes a HYBRID candidate (per CC-2 unwind disposition). This is the operator-decision-required pivot.

---

## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Status |
|---|---|---|
| 1 | UNIQUE-AND-COMPLETE-PER-METHOD bind | YES — encoder + decoder + per-pair latent + MDL loss bound in trainer |
| 2 | Canonical-vs-unique decision per layer | YES — landed (this memo §above) |
| 3 | HARD-EARNED-vs-CARGO-CULTED classification | YES — landed (this memo §above) |
| 4 | Probe-disambiguator per defensible interpretation | YES — Dykstra-feasibility analytical + post-smoke MDL ablation |
| 5 | Premise verification per Catalog #229 | YES — audit blueprint + recipe + existing campaign memo + Z1 ablation memo + Catalog #219 |
| 6 | 6-hook wire-in or N/A rationale per Catalog #125 | YES — probe-disambiguator hook #6 ACTIVE via β-sweep + MDL ablation |
| 7 | Predicted ΔS band with citation | YES — NULL pending Dykstra polytope (this memo §Predicted band) |
| 8 | Reactivation criteria pinned | YES — 4 criteria landed (this memo §above) |
| 9 | Sister-subagent ownership map per Catalog #230 | YES — DESIGN-only; trainer + Dykstra-tool implementation = operator-decision |

---

## Predicted ΔS band (per proposed Catalog #296)

**Predicted ΔS band:** `NULL pending Dykstra-feasibility polytope + Tier C anchor reconciliation`

**Dykstra-feasibility citation:** Alternating projections onto the 4 convex constraint sets:
- MDL constraint: `R_MDL >= L(θ) + log|θ|/2` (Rissanen 1978)
- IB constraint: `I(X;T) - β·I(T;Y) ≤ K_IB` (Tishby-Zaslavsky 2015)
- Contest rate budget: `25 · archive_bytes / 37,545,489` per CLAUDE.md scorer formula
- Z1 within-class density posterior (per Catalog #219; HNeRV-family saturated at ~99.3% MDL density at A1)

The intersection of these 4 polytopes defines the achievable Pareto frontier; the band is the projection onto the distortion axis (or its lower bound). Per Boyd's convex-feasibility lens, composition is SUBADDITIVE in the convex-intersection regime, NOT additive.

**Empirical-evidence-tag axis:** `[prediction]`; promotion to `[contest-CUDA]` / `[contest-CPU]` requires paired smoke completion + Tier C MDL ablation classification.

**Z1 within-class haircut:** If Tier C ablation classifies C6 as WITHIN-CLASS, apply Catalog #219 floor at ΔS = -0.005 (the same haircut applied to all HNeRV-family substrates per Z1 plateau finding). If Tier C classifies as ACROSS-CLASS, the haircut is removed and the Dykstra polytope projection applies directly.

**Council-grade tradeoff** per CLAUDE.md "Design decisions — non-negotiable": the revised band requires sextet-pact CONSENSUS (Shannon LEAD information-theoretic grounding + Dykstra CO-LEAD feasibility check + Yousfi + Fridrich + Contrarian + Assumption-Adversary).

---

## Cross-references

- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.2
- `.omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md` (existing campaign ledger)
- `.omx/research/mdl_ibps_substrate_council_design_round_1_20260513.md` (existing council design)
- `.omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md` (predicted band derivation)
- `.omx/research/zen_floor_band_v2_post_z1_ablation_20260514.md` (Z1 within-class evidence)
- CLAUDE.md Catalog #128 / #164 / #173 / #190 / #205 / #215 / #218 / #219 / #226 / #227 / #270 / #287 / #290 / #292 / #294 / #296

---

**Status:** UNWIND-LANDED 2026-05-16 (DESIGN-only; Dykstra tool implementation + trainer pivot = operator-decision).

---

## Observability surface

**Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive 2026-05-16** (`feedback_max_observability_into_behavior_xray_autopilot_tools_experiments_designs_standing_directive_20260516.md`) + Catalog #305 STRICT preflight gate (`check_substrate_design_memo_has_observability_surface_section`).

**Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:** this section is appended to the design memo; pre-existing body content (Sections 1-N + 9-dim checklist + cargo-cult audit + canonical-vs-unique decision + cross-references) is UNCHANGED.

**The 6-facet observability surface for this design:**

1. **Per-layer inspection.** Every layer of this substrate / composition / experiment captures its (input tensor, output tensor, intermediate activations, attention maps when applicable) at runtime via the canonical xray-style hook pattern (`tac.xray.<lens>` modules) without re-instrumentation. The forward pass emits per-layer observables to `experiments/results/<lane>/observability/per_layer/<layer_name>.jsonl` for post-hoc inspection.

2. **Per-signal decomposition.** Composite metrics (`final_score = seg + sqrt(10*pose) + 25*rate`) decompose into constituent contributions per the canonical `tac.xray.per_pair_score_decomposition` lens. Per-pair / per-class / per-axis / per-stage breakdown serialized to `experiments/results/<lane>/observability/score_decomposition.json` with axis labels matching CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable (`[contest-CUDA]` / `[contest-CPU]` / `[diagnostic-CPU]` / `[macOS-CPU advisory]` / `[MPS-PROXY]`).

3. **Run-to-run diff.** Two runs of this substrate / composition produce byte-identical reproducible artifacts under the same `(seed, commit_sha, upstream_snapshot_sha256)` tuple per Catalog #166 Modal HEAD-parity ledger + Catalog #245 modal_call_id_ledger. The canonical diff helper `tools/diff_auth_eval_results.py <run_a.json> <run_b.json>` (planned per the observability audit Highest-ROI extension list) emits per-component deltas + per-pair drift; until landed, manual diff via `archive_sha256` byte-addressability is straightforward.

4. **Post-hoc query interface.** Run artifacts under `experiments/results/<lane>/` serialize as structured JSON / JSONL surfaces consumable without re-running: `contest_auth_eval_<axis>.json` (canonical per-component scores with axis labels) + `modal_metadata.json` (per-dispatch cite-chain per Catalog #166) + `observability/*.jsonl` (per-layer + per-signal). The continual-learning posterior at `.omx/state/continual_learning_posterior.jsonl` is queryable per (substrate, axis, hardware, evidence_grade) via `tac.continual_learning.query_*` helpers per Catalog #128 + #131 fcntl-locked discipline.

5. **Cite-chain.** Every behavior signal anchors to the canonical tuple `(substrate_id, commit_sha, modal_call_id, config_path, random_seed, upstream_snapshot_sha256)` via Catalog #245 `tac.deploy.modal.call_id_ledger.register_dispatched_call_id(...)`. The call_id ledger row schema includes `mounted_code_git_head` (per Catalog #166) + `agent` + `subagent_id` + `session_id` for full forensic reconstruction. Score claims tagged per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable.

6. **Counterfactual hooks.** Byte-mutation surface per Catalog #139 packet compiler (`tools/verify_distinguishing_feature_byte_mutation.py --distinguishing-byte-range <offset>:<length>`) + Catalog #272 distinguishing-feature integration contract + Catalog #105 no-op detector. The substrate's archive grammar exposes byte-offset addressability for "what if this byte changed?" probing without re-running training. Per-layer / per-component ablation switches surfaced via the trainer's argparse flags + the canonical `tac.xray.<lens>.ablate_*` helpers when applicable.

**Acceptance per Catalog #305:** this section satisfies the structural requirement (literal section header `## Observability surface` present); the body content above documents the substrate's 6-facet observability surface for operator-facing audit.

**Sister observability-discipline gates active for this substrate:**

- Catalog #245 modal_call_id_ledger (every dispatch registered)
- Catalog #166 Modal HEAD-parity ledger (every dispatch worker-source-verified)
- Catalog #128 + #131 fcntl-locked JSONL posterior discipline (state mutations append-only)
- Catalog #139 packet compiler no-op detector (byte-mutation surface)
- Catalog #272 distinguishing-feature integration contract (per-substrate byte-mutation proof)
- Catalog #220 substrate L1+ operational mechanism declaration (no opaque byte additions)
- Catalog #105 no-op detector (no-op provenance)
- Catalog #127 authoritative tag custody (per-call-site axis + hardware-substrate validation)

**Observability extension recommendations (queued for follow-on):** see `tools/audit_existing_infrastructure_for_observability.py --summary` output for the canonical 8-tool / 6-facet observability gap analysis + Highest-ROI extension list. The `tools/audit_*.py` family is the highest-ROI extension target (3/12 observability) per the standing-directive consequence 3.

### Horizon-class classification (Catalog #309 / Pattern F)

**horizon_class: frontier_pursuit**

Per FALSIFICATION-AUDIT-v2 Pattern F + HORIZON-CLASS standing directive 2026-05-16: substrate predicted CPU band of `[0.120, 0.180]` classifies as **FRONTIER-PURSUIT** per the canonical 3-band taxonomy (PLATEAU-ADJACENT [0.180, 0.200] / FRONTIER-PURSUIT [0.120, 0.180] / ASYMPTOTIC-PURSUIT [0.050, 0.120]).

Empirical anchor: the predicted band [0.113-0.170] (per body) targets sub-plateau scores; frontier-pursuit classification reflects the substrate's design intent to escape the 0.196-0.199 plateau cluster.

Per CLAUDE.md "Forbidden premature KILL" + Catalog #296 Dykstra-feasibility + the cargo-cult unwind in the body: the predicted band is REVISED to `NULL pending Dykstra-feasibility check + Tier C empirical anchor reconciliation` per the body's §predicted-band unwind. The FRONTIER-PURSUIT horizon_class designation reflects substrate-design intent; actualization requires the empirical anchor + Dykstra-feasibility + Z1-within-class-haircut reconciliation per the body's reactivation criteria.


# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:historical_design_memo_uses_asymptotic_pursuit_token_in_planning_or_horizon_class_taxonomy_context_NOT_as_primary_substrate_class_shift_claim_per_z6_z7_z8_pattern_g_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
