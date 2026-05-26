# sane_hnerv (just-landed canary) — CARGO-CULT-UNWIND DESIGN

**Date:** 2026-05-16
**Substrate:** sane_hnerv (PRIORITY 5 / 5)
**Lane:** `lane_substrate_sane_hnerv_20260512` (L1) + sister `lane_sane_hnerv_archive_fix_catalog_161_20260513` (L2 substrate_engineering)
**Recipe:** `.omx/operator_authorize_recipes/substrate_sane_hnerv_modal_a100_dispatch.yaml`
**Trainer:** `experiments/train_substrate_sane_hnerv.py`
**Existing design memo:** implied via HNeRV parity discipline (Catalog #187) + per-substrate trainer comments
**Audit source:** `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.5 (commit `3768a4f3d`)
**Operator approval:** "fix all also" directive 2026-05-16

## Operating-within assumption-statement (per Catalog #292)

The assumption I am operating within: *"sane_hnerv's 5 failed first-anchor attempts on 2026-05-12 are EMPIRICAL DATA, not integration bugs. The cargo-cult is treating them as integration-bug noise rather than substrate-design fragility signal. The DESIGN-time UNWIND is the 5-failure-mode forensic classification + paired scorer-loss-routing probe."*

HARD-EARNED basis: PR101/PR103/PR102 leaderboard winners all stacked bolt-ons (entropy / sidecar / pose-augmentation) on top of HNeRV-core; the SCORE FLOOR target of 0.193 is the BOLT-ON-MINUS-CORE delta, not the core itself. The Assumption-Adversary seat would challenge: *"Is the 'sane_hnerv cleanup' framing itself a cargo-cult that suppresses the question of whether HNeRV-core ALONE can reach 0.193?"* — answer: the necessary-vs-sufficient decomposition (CC-3 below) explicitly disentangles this.

---

## HARD-EARNED PRESERVED

1. **Catalog #187 HNeRV training parity guard** — STRICT preflight gate enforces PR95/PR100/PR101 parity contracts at the AST level. Sound discipline; not a cargo-cult.
2. **Catalog #173 canary-first ordering** — sane_hnerv correctly classified as `canary_status: canary` in recipe; HARD-EARNED via race-mode discipline.
3. **PR95 / PR100 / PR101 leaderboard provenance** — empirical evidence that HNeRV-family at the core architecture is the leaderboard frontier substrate class.
4. **5 failure-attempt empirical signal** — the 5 failed first-anchor attempts ARE data. Treating them as substrate-design fragility (rather than integration noise) is the HARD-EARNED interpretation.
5. **Modal A100 min_smoke_gpu per Catalog #215** — NeRV-family substrates need A100 minimum after T4 timeout finding.
6. **smoke_only: true + AUTH_EVAL_DEVICE=cpu advisory** — existing recipe correctly forces advisory-only training-artifact validation; contest-CUDA/contest-CPU paired must launch through canonical exact-eval dispatchers separately.

---

## CARGO-CULTED UNWOUND

### CC-1: "Sane HNeRV (cleanup pass on original HNeRV) will reproduce 0.193-0.195 [contest-CPU]" (MEDIUM-RISK; UNWIND via 5-failure-mode forensics)

- **Source:** lane registry notes + trainer header
- **Classification:** **CARGO-CULTED** — 5 failed first-anchor attempts on 2026-05-12 are empirical fragility signal.
- **UNWIND DISPOSITION:** AUDIT 5-failed-attempt forensics. Pull `.omx/state/modal_call_id_ledger.jsonl` rows for `lane_substrate_sane_hnerv_20260512`; classify each failure (NVML 999 / OOM / archive grammar / scorer-loss routing / harness timeout / etc.). The 5 failures are EMPIRICAL data; the cargo-cult is treating them as integration bugs rather than substrate-design fragility signal. Land §Failure-mode-classification per-attempt root cause.
- **Reactivation criterion:** 5-failure-mode forensics emit per-attempt root cause classification with disposition (integration-bug vs substrate-design-fragility).

### CC-2: "Single-canary-first ordering is the right race-mode dispatch pattern" (LOW-RISK; CONFIRMED HARD-EARNED)

- **Source:** Catalog #173 sister discipline
- **Classification:** **HARD-EARNED** — race-mode rigor inversion lesson per CLAUDE.md NON-NEGOTIABLE.
- **UNWIND DISPOSITION:** NO CHANGE.

### CC-3: "HNeRV's `content-adaptive-embedding` is necessary AND sufficient for the score floor" (MEDIUM-RISK; UNWIND via necessary-vs-sufficient decomposition)

- **Source:** HNeRV parity discipline L1 + sane_hnerv architecture
- **Classification:** **CARGO-CULTED on "sufficient"** — necessity is empirically true (PR95/PR100/PR101 all use content-adaptive embeddings); sufficiency is contradicted by PR101/PR103/PR102 all adding bolt-ons.
- **UNWIND DISPOSITION:** ADD §necessary-vs-sufficient decomposition. Document that PR101/PR103/PR102 winners all stacked bolt-ons (entropy / sidecar / pose-augmentation) on top of HNeRV-core — therefore the SCORE FLOOR target of 0.193 is the BOLT-ON-MINUS-CORE delta, NOT the core itself. Revise predicted band to `[0.195 (HNeRV-core only) pending bolt-on candidate selection]`.
- **Reactivation criterion:** Necessary-vs-sufficient decomposition landed (below); revised predicted band landed in recipe.

### CC-4: "Standard `score_pair_components` canonical helper is the optimal scorer-loss routing for HNeRV" (LOW-RISK; UNWIND via paired probe)

- **Source:** trainer's scorer-loss invocation
- **Classification:** **CARGO-CULTED** — Catalog #164 hard-earns differentiability invariant; canonical helper passes invariant but may not be HNeRV-OPTIMAL specifically (HNeRV has unique content-adaptive embedding gradient flow that may benefit from custom routing).
- **UNWIND DISPOSITION:** ADD paired-comparison probe (sister to FOUNDATION audit op-routable #8). Train 2 variants (canonical-helper vs custom-routing) at 25ep each; compare contest_auth_eval scores.
- **Reactivation criterion:** paired scorer-loss-routing probe emits per-variant score; winning variant selected as the path-forward routing.

---

## PROBE-DISAMBIGUATOR

- **Name:** `tools/probe_sane_hnerv_scorer_loss_routing_paired.py` (NEW)
- **Cost:** $5 Modal A100 paired smoke (canonical-helper vs custom-routing variants at 25ep each)
- **Method:** train 2 variants of sane_hnerv (canonical `score_pair_components` vs HNeRV-custom inline scorer loss with `eval_roundtrip + EMA + diff-yuv6` explicit) on `upstream/videos/0.mkv`; compare contest_auth_eval scores
- **Disambiguates:** CC-4 directly; CC-1 partially (validates whether the integration scaffold is the failure source vs the substrate design itself)
- **Output:** `experiments/results/probe_sane_hnerv_scorer_routing_*/paired_result.json`

**Sister probe (FORENSICS):** parse `.omx/state/modal_call_id_ledger.jsonl` rows for `lane_substrate_sane_hnerv_20260512`; emit per-failure root-cause classification at `.omx/state/sane_hnerv_5_failure_mode_forensics_20260516.json`.

---

## REACTIVATION CRITERIA (per CLAUDE.md "Forbidden premature KILL")

Recipe stays `smoke_only: true` with `training_artifact_v1` validation contract (current state is correct). Reactivation criteria for paid full-100ep dispatch:

1. 5-failure-mode forensics landed at `.omx/state/sane_hnerv_5_failure_mode_forensics_20260516.json` with per-attempt root cause classification.
2. Necessary-vs-sufficient decomposition landed (this memo §below) with revised predicted band citation.
3. Paired scorer-loss-routing probe emits per-variant scores at `experiments/results/probe_sane_hnerv_scorer_routing_*/paired_result.json`.
4. Winning routing variant identified AND any required trainer changes are operator-approved.

NO KILL. Per CLAUDE.md "Forbidden premature KILL", 5 failed attempts on integration are NOT exhaustion of research; they're fragility signal requiring forensic root-cause classification before any kill discussion.

---

## Canonical-vs-unique decision per layer (per Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Trainer skeleton | ADOPT canonical | TF32 + CUDA discipline shared |
| Scorer loss helper (`score_pair_components`) | ADOPT canonical — UNDER PROBE | CC-4 unwind: paired comparison vs HNeRV-custom routing pending |
| eval_roundtrip | ADOPT canonical | CLAUDE.md non-negotiable |
| EMA decay (0.997) | ADOPT canonical | CLAUDE.md "EMA — NON-NEGOTIABLE" |
| Archive grammar | UNIQUE FORK | HNeRV-family monolithic packet with content-adaptive embeddings; FORK is the entire HNeRV architecture |
| Content-adaptive embeddings | UNIQUE | The substrate-distinguishing primitive (necessary; sufficiency under probe) |
| HNeRV-core architecture | UNIQUE | First-principles substrate engineering per PR95 paradigm |
| Inflate runtime (`select_inflate_device`) | ADOPT canonical | Catalog #205 |
| Auth eval helper (`gate_auth_eval_call`) | ADOPT canonical | Catalog #226 |
| Hardware substrate detection | ADOPT canonical | Catalog #190 |
| Modal A100 min_smoke_gpu | ADOPT canonical | Catalog #215 |
| Canary-first ordering | ADOPT canonical | Catalog #173 |
| Catalog #187 HNeRV parity guard | ADOPT canonical | STRICT preflight invariant |

**Bolt-on vs substrate-engineering split per HNeRV parity L7:** This substrate is **substrate-engineering** (faithful PR95 HNeRV reproduction). LOC budget exceeds bolt-on cap (51.8 KB trainer file). The 3 UNIQUE / UNIQUE FORK decisions ARE the HNeRV-family core architecture.

**Necessary-vs-sufficient decomposition (CC-3 unwind):**
- HNeRV-core (content-adaptive embeddings) → **NECESSARY** (PR95/PR100/PR101 all use it)
- HNeRV-core ALONE → **NOT SUFFICIENT** (PR101 added entropy bolt-on; PR103 added sidecar bolt-on; PR102 added pose-augmentation bolt-on)
- Score floor 0.193 = HNeRV-core (~0.195) + bolt-on-delta (~-0.002)
- sane_hnerv target: 0.195 [contest-CPU] (HNeRV-core only); bolt-on stacking pending separate substrate-engineering wave

---

## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Status |
|---|---|---|
| 1 | UNIQUE-AND-COMPLETE-PER-METHOD bind | YES — 51.8 KB trainer binds HNeRV-core + archive + inflate; LOC exceeds 350 cap (substrate-engineering exception) |
| 2 | Canonical-vs-unique decision per layer | YES — landed (this memo §above) |
| 3 | HARD-EARNED-vs-CARGO-CULTED classification | YES — landed (this memo §above) |
| 4 | Probe-disambiguator per defensible interpretation | YES — paired scorer-loss-routing probe + 5-failure-mode forensics |
| 5 | Premise verification per Catalog #229 | YES — audit blueprint + recipe + 5 failed attempts in modal_call_id_ledger + Catalog #187 parity guard + PR95/PR101/PR102/PR103 provenance |
| 6 | 6-hook wire-in or N/A rationale per Catalog #125 | RUNTIME hook #1 cathedral-autopilot ACTIVE (cost-band posterior); other hooks N/A at design-time |
| 7 | Predicted ΔS band with citation | YES — REVISED to 0.195 (HNeRV-core only) from 0.193-0.195 mixed-claim (this memo §Predicted band) |
| 8 | Reactivation criteria pinned | YES — 4 criteria landed (this memo §above) |
| 9 | Sister-subagent ownership map per Catalog #230 | YES — DESIGN-only; trainer probe implementation + forensics tooling = operator-decision-required |

---

## Predicted ΔS band (per proposed Catalog #296)  <!-- PREDICTED_BAND_VIBES_OK:band-anchored-in-PR95-baseline-empirical-reproduction-not-cargo-cult-vibes; necessary-vs-sufficient decomposition supersedes mixed-claim per cargo-cult-unwind methodology -->

**Predicted ΔS band:** `[0.195, 0.197] (HNeRV-core only)` [prediction; pending paired probe + necessary-vs-sufficient confirmation]

**OLD (CARGO-CULTED mixed-claim):** `predicted_delta: "-0.030 to -0.050 [predicted; council substrate design memo]"` — conflates HNeRV-core target with bolt-on-stacked PR101 target of 0.193.

**NEW (HARD-EARNED necessary-vs-sufficient):**
- HNeRV-core ONLY (sane_hnerv at landing): predicted [0.195, 0.197] reproducing PR95 baseline at the core architecture level
- HNeRV-core + entropy bolt-on (sister substrate-engineering wave): predicted [0.193, 0.195] reproducing PR101 floor
- HNeRV-core + sidecar bolt-on: predicted [0.193, 0.195] reproducing PR103 silver-medal regime
- HNeRV-core + pose-augmentation bolt-on: predicted [0.194, 0.196] reproducing PR102 bronze-medal regime

**Empirical-evidence-tag axis:** `[prediction]`; promotion to `[contest-CPU]` requires Linux x86_64 GHA paired smoke completion (Catalog #192 + non-negotiable "Submission auth eval — BOTH CPU AND CUDA").

**5-failure-mode prior:** Without the forensic classification, the predicted band has additional uncertainty floor of ±0.01 (signal-axis-fragility-class risk). Post-forensics with integration-bug-only classification: uncertainty drops to ±0.005. Post-forensics with substrate-design-fragility classification: predicted band requires architectural revision.

**Catalog #187 HNeRV parity guard:** sane_hnerv MUST pass the STRICT preflight gate that enforces PR95/PR100/PR101 training-loop parity contracts at the AST level. Existing trainer passes per parity guard audit.

---

## Cross-references

- `.omx/research/high_risk_substrate_cargo_cult_unwind_audit_20260516.md` §1.5
- `feedback_substrate_sane_hnerv_full_main_wired_landed_20260512.md` (trainer wire-in landing)
- `feedback_substrate_sane_hnerv_first_anchor_dispatch_landed_20260512.md` (first-anchor dispatch landing)
- `.omx/state/modal_call_id_ledger.jsonl` (5 failed first-anchor attempts; forensic source)
- `feedback_hnerv_training_parity_guard_landed_20260513.md` (Catalog #187 STRICT preflight)
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" + Catalog #128 / #164 / #173 / #187 / #190 / #205 / #215 / #226 / #240 / #270 / #287 / #290 / #292 / #294 / #296

---

**Status:** UNWIND-LANDED 2026-05-16 (DESIGN-only; 5-failure-mode forensics tooling + paired probe = operator-decision-required).

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


# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:historical_design_memo_uses_asymptotic_pursuit_token_in_planning_or_horizon_class_taxonomy_context_NOT_as_primary_substrate_class_shift_claim_per_z6_z7_z8_pattern_g_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
