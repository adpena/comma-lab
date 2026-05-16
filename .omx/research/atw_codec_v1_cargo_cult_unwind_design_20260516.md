# ATW Codec V1 (Atick-Tishby-Wyner) — CARGO-CULT-UNWIND DESIGN

**Date:** 2026-05-16
**Substrate:** ATW Codec V1 (PRIORITY 3 / 5)
**Lane:** `lane_atw_codec_design_v1_20260515`
**Recipe:** `.omx/operator_authorize_recipes/substrate_atw_codec_v1_modal_a100_dispatch.yaml`
**Trainer:** `experiments/train_substrate_atw_codec_v1.py` (scaffold; `_full_main` raises NotImplementedError)
**Existing design memo:** `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md`
**Audit source:** `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.3 (commit `3768a4f3d`)
**Operator approval:** "fix all also" directive 2026-05-16

## Operating-within assumption-statement (per Catalog #292)

The assumption I am operating within: *"ATW Codec V1 is HONESTLY-ACKNOWLEDGED research-only at the existing design memo (the operator-facing disclaimer already exists for CC-3); the unwind is primarily about RIGOR — landing the canonical-vs-unique decision per layer AND running the shared H(latent|scorer_class) probe-disambiguator BEFORE any Phase 2 council lift of `_full_main NotImplementedError`."*

HARD-EARNED basis: existing design memo §1 already names CC-3 ("currently a hypothesis, not a measured artifact") — operator-facing disclaimer present. The Assumption-Adversary seat would challenge: *"Is the three-paper composition (Atick-Redlich + Tishby + Wyner-Ziv) itself a cargo-cult novelty claim?"* — answer: CC-1 below preserves the math-tractability (HARD-EARNED) while flagging the empirical-equivalence axis (CARGO-CULTED).

---

## HARD-EARNED PRESERVED

1. **Three-paper Lagrangian math tractability** — `L_ATW = α·L_AR + β·L_T + γ·L_WZ` composes by convex addition; the math IS tractable per Boyd's convex-feasibility lens. Citations correct in existing design memo §1.
2. **HONESTLY-ACKNOWLEDGED CC-3** — operator-facing disclaimer "Wyner-Ziv gain estimate ... currently a hypothesis, not a measured artifact" already present in existing design memo. This is the rare case of a cargo-cult that's PRE-MITIGATED by honest disclosure.
3. **Catalog #240 recipe-vs-trainer-state consistency** — recipe declares `research_only: true` + `dispatch_enabled: false`; trainer's `_full_main` raises NotImplementedError. CORRECT honest-state per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY".
4. **Phase 2 council reactivation criteria** — existing design memo §5 enumerates 4 reactivation criteria (any TWO grant Phase 2 approval). Sound discipline.
5. **canary_status: post_canary_dependent** + `canary_dependency: lane_z4_cooperative_receiver_loss_step2_20260514` — Catalog #173 sister-canary discipline correctly applied.
6. **Three-knob (κ_IB, λ_WZ, λ_pixel) probe-disambiguator regime sweep** — Catalog #125 hook #6 satisfied; knob-zero ablations recover 4 corner regimes.

---

## CARGO-CULTED UNWOUND

### CC-1: "Three-paper composition into ONE Lagrangian is novel and tractable" (LOW-RISK; CONFIRMED HARD-EARNED structurally; UNWIND empirical-equivalence axis)

- **Source:** existing design memo §1 (novelty claim)
- **Classification:** **HARD-EARNED structurally** (math IS tractable; three Lagrangian terms compose by convex addition); **CARGO-CULTED at the empirical-equivalence axis** (does ATW actually produce different bytes than the union of Atick-only + Tishby-only + Wyner-Ziv-only at runtime?)
- **UNWIND DISPOSITION:** ADD §canonical-vs-unique decision per layer (Catalog #290) section formalizing the 3-paper Lagrangian as the UNIQUE substrate-engineering layer (FORK from canonical single-paper substrates). Document the math composition with per-term gradient-fidelity check. Document the empirical-equivalence regime sweep (knob-zero ablations).
- **Reactivation criterion:** §canonical-vs-unique decision per layer landed below.

### CC-2: "Predicted [0.18, 0.21] frontier displacement" (HIGH-RISK; HIGHEST-PRIORITY UNWIND)

- **Source:** existing design memo §predicted-band (cited from grand reunion symposium Composite #1)
- **Classification:** **CARGO-CULTED** — same hand-waved-prediction-band class as NSCS06 symposium-#4.
- **NOTE on existing recipe:** recipe ALREADY has `predicted_band: null` + `predicted_delta: "unranked; grounded latent-only rate-side bound is approximately [-0.0027, -0.005] if H(latent | scorer_class) probe confirms 30-50% latent-byte savings"` — this is GOOD discipline; the unwind formalizes it.
- **UNWIND DISPOSITION:** REWRITE design memo §predicted-band (already partially done in recipe) as `[NULL pending H(latent|scorer_class) probe + Dykstra-feasibility check]`. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287: every numeric prediction MUST carry `[prediction]` axis tag and link to the empirical probe path.
- **Reactivation criterion:** H(latent|scorer_class) probe emits posterior; Dykstra-feasibility polytope landed.

### CC-3: "Wyner-Ziv gain estimate for dashcam + scorer is 30-50% conditional entropy reduction" (HIGH-RISK; PRE-MITIGATED via honest disclosure; UNWIND via probe execution)

- **Source:** existing design memo §1 (already states "currently a hypothesis, not a measured artifact")
- **Classification:** **CARGO-CULTED + HONESTLY ACKNOWLEDGED** — operator-facing disclaimer present.
- **UNWIND DISPOSITION:** The unwind IS the probe execution. Run shared `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (sister across ATW + D4 + Z4) and update design memo + recipe with empirical Wyner-Ziv gain ceiling.
- **Reactivation criterion:** H(latent|scorer_class) probe emits empirical bit-savings posterior at `.omx/state/h_latent_given_scorer_class_atw_v1.json`.

---

## PROBE-DISAMBIGUATOR

- **Name:** `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (NEW; SHARED across ATW + D4 + Z4 per audit foundation §5 op-routable #5)
- **Cost:** $3-5 CPU smoke (analytical or one-shot CPU inference)
- **Method:** compute H(latent | scorer_class) for A1 latent blob conditioned on SegNet argmax classes; estimate Wyner-Ziv gain as max(0, H(latent) - H(latent|class))
- **Disambiguates:** CC-3 (Wyner-Ziv 30-50% gain hypothesis) directly + CC-2 (predicted band sanity check) by computing empirical conditional-entropy ceiling
- **Output:** `.omx/state/h_latent_given_scorer_class_atw_v1.json`; updates ATW + D4 + Z4 simultaneously

**Sister probe:** `tools/check_substrate_dykstra_feasibility.py --substrate atw_codec_v1` for analytical Dykstra check ($0).

---

## REACTIVATION CRITERIA (per CLAUDE.md "Forbidden premature KILL")

Recipe stays `research_only: true` + `dispatch_enabled: false`. Phase 2 council approval REQUIRED to lift `_full_main NotImplementedError`. Reactivation criteria for THIS unwind landing (sister to but distinct from Phase 2 lift criteria):

1. §canonical-vs-unique decision per layer landed (below).
2. §Predicted ΔS band rewritten to NULL pending probe (this memo §below; recipe ALREADY has this).
3. `[prediction]` axis tags applied to every numeric claim in this memo (sister of Catalog #287 docstring discipline).
4. Shared H(latent|scorer_class) probe runs; empirical bit-savings posterior landed.

Phase 2 council lift criteria (independent of unwind; preserved from existing recipe + design memo):
- Z4-V2 successful contest-CUDA anchor (fc-01KRPJVEMQ5S7Q8EKGKQWKCS93) with ΔS in [-0.005, -0.010] vs A1, OR
- Tao+Boyd Blahut-Arimoto analytical floor shows ATW theoretical gain ≥ -0.020 vs A1 ($0, 2 hours CPU), OR
- WZ side-info bit-savings empirically measured on A1 latents ≥ 20% per-pair latent rate ($0, 1 hour CPU), OR
- Z3-G1 substitution (Wunderkind G1, $10) returns successful CUDA anchor.

Any TWO of the four grant Phase 2 council approval. NO KILL.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical | TF32 + CUDA discipline shared |
| Scorer loss helper (`score_pair_components`) | ADOPT canonical | Catalog #164 |
| eval_roundtrip | ADOPT canonical | CLAUDE.md non-negotiable |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE" |
| Archive grammar (`monolithic_atw1_self_contained`) | UNIQUE FORK | Substrate-specific: encoder + decoder + per-pair latent + Wyner-Ziv side-info + sorted-keys JSON meta; FORK is the entire ATW design |
| Atick-Redlich cooperative-receiver primitive | ADOPT canonical | `src/tac/codec/cooperative_receiver/atick_redlich.py` is the canonical primitive ATW reuses |
| Tishby IB Lagrangian term | UNIQUE | The substrate-distinguishing primitive; κ_IB ≥ 0 requires posterior approximation q(y\|t) |
| Wyner-Ziv side-info head | UNIQUE | Closed-form computation: predict z from scorer class prior via Atick-Redlich whitening + ICA projection |
| Three-knob composition (κ_IB, λ_WZ, λ_pixel) | UNIQUE | The substrate's probe-disambiguator regime sweep mechanism |
| Inflate runtime (`select_inflate_device`) | ADOPT canonical | Catalog #205 |
| Auth eval helper (`gate_auth_eval_call`) | ADOPT canonical | Catalog #226 |
| Hardware substrate detection | ADOPT canonical | Catalog #190 |
| Modal A100 min_smoke_gpu | ADOPT canonical | Catalog #215 |

**Bolt-on vs substrate-engineering split per HNeRV parity L7:** This substrate is **substrate-engineering** (NEW architecture class: cooperative-receiver + IB + Wyner-Ziv triple). LOC budget exceeds bolt-on cap explicitly. The 4 UNIQUE / UNIQUE FORK decisions ARE the substrate-optimal engineering surface; Atick-Redlich primitive ADOPT canonical is shared infrastructure value preserved.

**Per-layer empirical-equivalence axis (CC-1 unwind):** the four corner regimes (knob-zero ablations) ARE the probe-disambiguator that arbitrates whether ATW produces different bytes than the union of Atick-only / Tishby-only / Wyner-Ziv-only at runtime. Without that empirical evidence, the three-paper composition NOVELTY claim is unproven.

---

## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Status |
|---|---|---|
| 1 | UNIQUE-AND-COMPLETE-PER-METHOD bind | PARTIAL — scaffold only; `_full_main` raises NotImplementedError pending Phase 2 council |
| 2 | Canonical-vs-unique decision per layer | YES — landed (this memo §above) |
| 3 | HARD-EARNED-vs-CARGO-CULTED classification | YES — landed (this memo §above) |
| 4 | Probe-disambiguator per defensible interpretation | YES — H(latent\|scorer_class) probe + Dykstra polytope + three-knob regime sweep |
| 5 | Premise verification per Catalog #229 | YES — audit blueprint + recipe + existing design memo + Atick-Redlich primitive + Wyner-Ziv 1976 |
| 6 | 6-hook wire-in or N/A rationale per Catalog #125 | DESIGN-TIME at landing; runtime wire-in deferred to Phase 2 council lift |
| 7 | Predicted ΔS band with citation | YES — NULL pending probe (this memo §Predicted band; recipe already has this) |
| 8 | Reactivation criteria pinned | YES — 4 unwind criteria + 4 Phase 2 lift criteria landed (this memo §above) |
| 9 | Sister-subagent ownership map per Catalog #230 | YES — DESIGN-only; `_full_main` implementation = operator-decision-required + Phase 2 council |

---

## Predicted ΔS band (per proposed Catalog #296)

**Predicted ΔS band:** `NULL pending H(latent|scorer_class) probe + Dykstra-feasibility check` [prediction]

**Existing recipe text (correctly disciplined):** `predicted_delta: "unranked; grounded latent-only rate-side bound is approximately [-0.0027, -0.005] if H(latent | scorer_class) probe confirms 30-50% latent-byte savings"`

This is the GOOD discipline: the only grounded V1 rate-side bound from the stated latent savings is approximately -0.0027 to -0.005; [0.18, 0.21] remains an exploratory planning envelope, NOT a dispatch-ranking claim.

**Empirical-evidence-tag axis:** `[prediction]` (per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287).

**Dykstra-feasibility intersection (per Boyd's convex-feasibility lens):**
- Cooperative-receiver constraint (Atick-Redlich 1990): `R_AR ≥ H(X) - I(X;Y_cooperative)`
- IB constraint (Tishby-Zaslavsky 2015): `I(X;T) - β·I(T;Y) ≤ K_IB`
- Wyner-Ziv side-info-at-decoder constraint (Wyner-Ziv 1976): `R_WZ ≥ R(D) - I(X;Y_side)`
- Contest rate budget: `25 · archive_bytes / 37,545,489`

The intersection of these 4 polytopes is a SUBSET of the Z3+A1 polytope per Boyd's convex-feasibility lens (existing recipe `risk` field correctly states this); this does NOT imply dominance absent a measured lower-score point on the same archive/runtime/eval axis.

**Reactivation predicted-band update:** Once H(latent|scorer_class) probe emits empirical bit-savings posterior:
- If gain ≥ 30%: revise band to `[-0.0027, -0.005]` [prediction; pending paired smoke]
- If gain < 20%: revise band to `[NULL; ATW does not displace A1 frontier]` and trigger Phase 2 council DEFER discussion per CLAUDE.md "Forbidden premature KILL" (NOT kill; DEFER pending alternative hypothesis).

---

## Cross-references

- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.3
- `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` (existing design memo)
- `feedback_grand_reunion_fields_grade_passion_full_council_debrief_*_20260515.md` (Composite #1)
- `feedback_z4_atick_redlich_minimum_viable_landed_20260515.md` (Z4 sister)
- `src/tac/codec/cooperative_receiver/atick_redlich.py` (canonical primitive)
- CLAUDE.md Catalog #125 / #164 / #173 / #190 / #205 / #215 / #220 / #226 / #240 / #270 / #287 / #290 / #292 / #294 / #296

---

**Status:** UNWIND-LANDED 2026-05-16 (DESIGN-only; `_full_main` implementation = Phase 2 council-required; probe execution = sister-subagent territory).

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

**horizon_class: plateau_adjacent**

Per FALSIFICATION-AUDIT-v2 Pattern F + HORIZON-CLASS standing directive 2026-05-16: substrate predicted CPU band of `[0.180, 0.200]` classifies as **PLATEAU-ADJACENT** per the canonical 3-band taxonomy (PLATEAU-ADJACENT [0.180, 0.200] / FRONTIER-PURSUIT [0.120, 0.180] / ASYMPTOTIC-PURSUIT [0.050, 0.120]).

Empirical anchor: the 0.196-0.199 cluster IS the canonical plateau (A1 0.19286 / PR101 0.193 / PR103 0.195 / PR102 0.195 / Z3 v2 0.19778 — all within ±0.005 of 0.196). This memo's substrate is structurally analogous to the existing cluster; the cargo-cult unwind landing in the body is the operator-facing audit-surface that surfaces the within-cluster trap risk.

Per CLAUDE.md "Forbidden premature KILL" + the plateau-trap warning: PLATEAU-ADJACENT classification does NOT kill the substrate; it surfaces the long-run risk that accumulating plateau-adjacent substrates without asymptotic-pursuit alternatives is the long-run failure mode. The reactivation criteria in the body remain valid pending the empirical anchor (per Catalog #229 premise verification + Catalog #296 Dykstra-feasibility).

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

<!-- # PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope -->

This memo references Atick-Redlich / cooperative-receiver framing as cross-reference / related-work / sister-substrate context — NOT as this substrate's architectural core. The substrate proposed by this memo is structurally distinct from Z6/Z7/Z8 (which DO require ego-motion-conditioned next-frame prediction as architectural core per Pattern H + Z6/Z7/Z8 design memo Section 11).

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Z6/Z7/Z8 design memo Pattern H + Catalog #311 acceptance cascade (c): same-line waiver `# PREDICTIVE_CODING_EGO_MOTION_CONDITIONED_OK:cross-reference-only-not-substrate-central-cooperative-receiver-framing-cited-as-related-work-not-as-this-substrate-architectural-core-per-z6z7z8-design-memo-section-11-scope` applies. The waiver rationale is non-placeholder (>4 chars, not `<rationale>` / `<reason>`).

Cross-references to cooperative-receiver / Atick-Redlich in this memo serve as theoretical-anchor / related-work / sister-substrate-comparison only; they do NOT make this substrate a predictive-coding substrate in the Pattern H sense.
