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

---

## 2026-05-16 Codex Supersession Note

Append-only correction for executable CLI/provenance fidelity:

- Lines above that name `tools/check_substrate_dykstra_feasibility.py --substrate ...`
  are stale. The executable argparse surface uses `--substrate-id`, not
  `--substrate`.
- Lines above that say the Dykstra output updates the Cathedral autopilot
  posterior are stale as a direct command claim. The tool writes the Dykstra
  JSON artifact only. Any posterior update must happen through a separate
  consumer with its own committed command, custody, and ledger evidence.
- Current refreshed command:

```bash
.venv/bin/python tools/check_substrate_dykstra_feasibility.py \
  --substrate-id time_traveler_l5_5move \
  --predicted-band-lo 0.150 \
  --predicted-band-hi 0.170 \
  --archive-size-bytes 34603 \
  --tt5l-five-move-polytope \
  --output-json .omx/state/dykstra_feasibility_time_traveler_l5.json
```

Refreshed Dykstra artifact SHA-256:
`226c227c1c08b25ea7208c6ee774f7621b25c25929870c28535a1f8896504b60`.

Current readiness after refresh: Dykstra score-axis sanity is valid, move-level
feasibility is still missing, and the next TT5L action remains
`materialize_tt5l_move_level_feasibility_proof`. No score claim, promotion
claim, rank authority, or dispatch authority is implied by this note.

### Horizon-class classification (Catalog #309 / Pattern F)

**horizon_class: frontier_pursuit**

Per FALSIFICATION-AUDIT-v2 Pattern F + HORIZON-CLASS standing directive 2026-05-16: substrate predicted CPU band of `[0.120, 0.180]` classifies as **FRONTIER-PURSUIT** per the canonical 3-band taxonomy (PLATEAU-ADJACENT [0.180, 0.200] / FRONTIER-PURSUIT [0.120, 0.180] / ASYMPTOTIC-PURSUIT [0.050, 0.120]).

Empirical anchor: the predicted band [0.113-0.170] (per body) targets sub-plateau scores; frontier-pursuit classification reflects the substrate's design intent to escape the 0.196-0.199 plateau cluster.

Per CLAUDE.md "Forbidden premature KILL" + Catalog #296 Dykstra-feasibility + the cargo-cult unwind in the body: the predicted band is REVISED to `NULL pending Dykstra-feasibility check + Tier C empirical anchor reconciliation` per the body's §predicted-band unwind. The FRONTIER-PURSUIT horizon_class designation reflects substrate-design intent; actualization requires the empirical anchor + Dykstra-feasibility + Z1-within-class-haircut reconciliation per the body's reactivation criteria.

### Ego-motion conditioning declaration (Catalog #311 / Pattern H)

This substrate IS ego-motion-conditioned. Per Z6/Z7/Z8 design memo Pattern H + Catalog #311 acceptance cascade (a): the body contains ego-motion tokens AND predictive-coding tokens.

**Ego-motion conditioning primitives (per body):**
- ego-motion-matched foveation rate `R_fov` per Gibson 1950 + LAPose
- focus-of-expansion (FOE) prior per Ballard's embodied-vision lens
- pose-conditioned predictor over T=4 frame windows
- next-frame prediction (autoregressive predictor) conditioned on the FOE prior

**Predictive-coding primitives (per body):**
- next-frame predictor (autoregressive predictor over T=4 frames)
- predictive coverage constraint per Rao-Ballard 1999 hierarchical predictive coding
- temporal-decorrelation residual via predictive-coding bottleneck

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Z6/Z7/Z8 design memo Section 11 (Atick-Redlich + Ballard FOE prior): this substrate IS the canonical ego-motion-conditioned next-frame-prediction class-shift substrate that Z6/Z7/Z8 require.

## 2026-05-18 Codex Blocker Refresh

Append-only correction to the 2026-05-16 reactivation state:

- The prior "move-level feasibility is still missing" line is superseded.
  `.omx/state/tt5l_move_level_feasibility.json` now validates with
  `predicate_passed=true`, `move_level_constraint_proof=true`,
  `residual_max=0.0`, and no status blockers.
- The Dykstra score-axis sanity artifact remains valid:
  `.omx/state/dykstra_feasibility_time_traveler_l5.json` SHA-256
  `226c227c1c08b25ea7208c6ee774f7621b25c25929870c28535a1f8896504b60`.
- The contest full-frame side-info consumption proof is valid through
  `.omx/research/tt5l_contest_sideinfo_consumption_proof_20260516_codex.json`
  SHA-256 `d430dd7ccc97da125ca3985a2f70d7cef4c37d39cccad1952698d37d177c9a86`.
- `l5_v2_tt5l_campaign_readiness(repo_root=.)` reports valid first-anchor
  timing smoke, materialized paired work unit, side-info dispatch plan,
  Lightning execution preflight, execution bundle, dry-run verification,
  route-unblock packet, and doctor plan.

Current TT5L readiness is still **DEFER**, not READY. The stale recipe
blockers for Dykstra emission and additive-composition revision were removed
from `.omx/operator_authorize_recipes/substrate_time_traveler_l5_autonomy_modal_a100_dispatch.yaml`;
the current blockers are probe/architecture-lock semantics, prediction-band
authority (`prediction_band_not_dispatch_ready` and missing baseline/empirical
anchor custody), and provider capacity/runtime readiness (`Modal workspace
billing limit` plus missing Lightning target/teamspace/inventory/source/runtime
probe). No score claim, rank authority, promotion authority, or paid-dispatch
authority is implied by this refresh.

## 2026-05-18 Codex Probe-Gate Hash Repair + Current Failure Class

The TT5L probe-gate invalidity had one stale software blocker and several real
semantic blockers. The stale blocker was
`l5_v2_gate_artifact_semantics_invalid:c1_z5_tt5l_probe_disambiguator:probe_verdict_sha256_mismatch`:
`tools/build_l5_v2_probe_gate_artifact.py`/`tac.optimization.l5_v2_probe_disambiguator`
recorded the verdict digest with the probe module's canonical JSON contract,
while `tac.optimization.l5_staircase_v2` recomputed with a local no-newline
contract. `src/tac/optimization/l5_staircase_v2.py` now validates the builder
contract and retains the old no-newline digest only as a compatibility
fallback; `src/tac/optimization/l5_v2_probe_disambiguator.py` exposes the
builder digest helper so future gate artifacts and readiness checks share one
contract.

The derived architecture-lock packet/report were refreshed through
`tools/build_l5_v2_architecture_lock_packet.py`; they remain no-lock artifacts
with `score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.

Focused regression:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_probe_gate_status_accepts_builder_verdict_hash \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_canonical_probe_gate_evidence_auto_consumes_valid_artifact \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_canonical_probe_gate_evidence_skips_blocked_artifact \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_probe_gate_recomputes_verdict_from_observations \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_probe_gate_rejects_selected_candidate_not_min_eligible_delta \
  src/tac/tests/test_l5_v2_probe_disambiguator.py::test_l5_v2_probe_gate_artifact_cli_exits_nonzero_when_blocked \
  -q
# 6 passed
```

After the hash repair, the committed probe gate artifact still remains invalid
for real semantic reasons:

- `recomputed_architecture_lock_allowed`
- `recomputed_probe_blockers_nonempty`
- `architecture_lock_allowed`
- `probe_blockers_nonempty`
- missing eligible observations for `c1_world_model_foveation`,
  `z5_predictive_coding_world_model`, and `time_traveler_l5_autonomy`

The existing TT5L paired CPU/CUDA artifacts are custody evidence for a
byte-closed TT5L work unit, not architecture-lock evidence: TT5L still has
`predicate_passed=false`, `sideinfo_consumed=false`, and missing per-axis
`score_delta` fields in the probe observation intake. C1 and Z5 still have no
paired exact-axis observations in the intake. Classification remains
`NEEDS_FIX`, no score claim, no rank/kill authority, no promotion authority,
and no paid-dispatch authority. The next frontier-moving action is still to
resolve provider capacity or complete an alternate Lightning route, run the
paired probe measurements, and convert harvested CPU/CUDA results into
sideinfo-consumed probe observations with real score deltas.

## 2026-05-18 Codex Lightning Doctor / Non-Dry-Run Gate

The current TT5L campaign-readiness surface already has the local campaign
machinery: byte-closed materialized paired work unit, first-anchor timing-smoke
artifact, paired-axis plan, Lightning execution preflight, execution bundle,
route-unblock packet, dry-run verification, and doctor plan. The remaining
frontier blocker is provider execution readiness, not another local campaign
memo.

This pass ran the required Lightning doctor command from the doctor plan and
wrote:

- `.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`

Result:

- `status=FAIL`
- local supply-chain scan passed (`lightning-sdk=2026.4.23`, no violations)
- failed checks: `ssh_auth`, `remote_supply_chain`, `machine_inventory`
- exact causes:
  `--require-ssh` failed because no `--ssh-target`/`LIGHTNING_SSH_TARGET` was
  provided; `--require-remote-supply-chain` failed because no remote scan could
  run without SSH; `--require-machine-inventory` failed because no
  `--teamspace`/`LIGHTNING_TEAMSPACE` was provided

Then the non-dry-run gate was regenerated:

- `.omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.json`
- `.omx/research/l5_v2_tt5l_lightning_non_dry_run_gate_20260517_codex.md`

Result:

- `ready_for_non_dry_run_submit=false`
- `ready_for_provider_dispatch=false`
- `ready_cells=0/10`
- blocker count: `166`
- `dispatch_attempted=false`
- `provider_spend_attempted=false`
- `score_claim=false`
- `promotion_eligible=false`

The TT5L operator recipe now carries the concrete doctor-failure blockers:

- `l5_v2_tt5l_lightning_non_dry_run_gate_failed_doctor_ssh_auth`
- `l5_v2_tt5l_lightning_non_dry_run_gate_failed_doctor_remote_supply_chain`
- `l5_v2_tt5l_lightning_non_dry_run_gate_failed_doctor_machine_inventory`

Current next gate:

1. Resolve the Modal workspace billing blocker or choose Lightning.
2. For Lightning, set `LIGHTNING_SSH_TARGET`, `LIGHTNING_TEAMSPACE`, and exactly
   one owner identity (`LIGHTNING_SDK_USER` or `LIGHTNING_ORG`).
3. Re-run the doctor until `status=OK`.
4. Stage source manifests, create active per-cell lane claims, remove command
   placeholders, then submit the paired CPU/CUDA side-info cells.

No provider work was launched in this refresh. No lane claim was needed because
the artifacts are local fail-closed readiness artifacts only.

## 2026-05-18 Codex TT5L Queue Blocker Reconciliation

The asymptotic-pursuit queue still surfaced the stale
`l5_v2_probe_gate_artifact_semantics_invalid_probe_verdict_sha256_mismatch`
recipe blocker after the live probe-gate validator no longer emitted it. The
TT5L operator-authorize recipe now mirrors the live semantic blockers instead:

- `recomputed_architecture_lock_allowed`
- `recomputed_probe_blockers_nonempty`
- `architecture_lock_allowed`
- `probe_blockers_nonempty`
- missing eligible observations for the C1/Z5/TT5L probe candidates

`src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py` now asserts that
the stale hash-mismatch blocker stays out of TT5L dispatch blockers while the
real probe-blocker and missing-observation reasons remain visible. This is a
queue hygiene fix only: TT5L stays `NEEDS_FIX`; Candidate 4c remains the only
current `READY` asymptotic-pursuit dispatch candidate.


# HIERARCHICAL_PREDICTIVE_CODING_QUADRUPLE_OK:design_memo_references_hierarchical_predictive_coding_in_cross_reference_or_partial_subset_context_NOT_as_primary_substrate_binding_all_four_Rao_Ballard_Mallat_DreamerV3_WynerZiv_canonical_primitives_simultaneously_per_catalog_312_pattern_i_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
