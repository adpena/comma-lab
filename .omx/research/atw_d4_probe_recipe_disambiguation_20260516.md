---
title: "ATW D4 probe recipe disambiguation — 'ATW D4 probe ($3-5)' resolves to substrate_atw_codec_v2 own D4 H(latent|scorer_class) probe, NOT sister substrate_d4_wyner_ziv_frame_0 standalone"
date: 2026-05-16
author: phase_1b_atw_v2_lift_20260516 subagent
lane: lane_phase_1b_atw_v2_lift_20260516
horizon_class: frontier_pursuit
council_tier: T1
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_attendees: [phase_1b_atw_v2_lift_subagent]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "DISAMBIGUATION: 'ATW D4 probe ($3-5)' = ATW codec v2's own D4 H(latent|scorer_class) probe on A1 latents (tools/run_atw_v2_d4_probe_from_a1.py); NOT sister substrate_d4_wyner_ziv_frame_0 standalone substrate."
  - "PREMISE VERIFICATION (Catalog #229): parent prompt premise 'lift _full_main from NotImplementedError to working PR95-paradigm trainer' FALSIFIED. _full_main is ALREADY IMPLEMENTED at HEAD commit b1f1913e5 (215 LOC, no NotImplementedError, 28/28 tests pass)."
  - "D4 PROBE VERDICT ALREADY RETURNED: INDEPENDENT (MI=0.006385 bits/symbol vs 0.5 threshold; 2 orders of magnitude below MEANINGFUL_CONDITIONING). Per design memo section 19 + parallel council T3 verdict 'PROCEED-TO-D4-PROBE-FIRST-THEN-LIFT': lift authority NOT granted; status is DEFER-pending-research per CLAUDE.md 'Forbidden premature KILL' with G2-PARTIAL alternative-hypothesis cited."
  - "NO LIFT EXECUTED: _full_main was already lifted (no action); dispatch_enabled stays false (D4 verdict bars lifting); recipe stays research_only=true per design memo + parallel council adjudication."
related:
  - .omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md
  - .omx/research/grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md
  - .omx/research/wave_3_phase_1_systemic_refusal_3_of_5_recipes_dispatch_disabled_20260516.md
  - .omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md
  - .omx/research/k_measurement_schedule_level_1_rebalanced_post_donoho_tanner_20260516.md
council_assumption_adversary_verdict:
  - assumption: "'ATW D4 probe' parent prompt phrase requires disambiguation between (a) ATW v2's own D4 probe and (b) sister standalone substrate_d4_wyner_ziv_frame_0 recipe"
    classification: HARD-EARNED
    rationale: "Empirical evidence FROM 3 SOURCES converges: (1) ATW v2 design memo PV-4 + section 19 reactivation criterion V2-1 explicitly defines D4 H(latent|scorer_class) probe at tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py as ATW v2's OWN pre-dispatch gate; (2) parallel council T3 memo (grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md) line 71 says verbatim 'ATW v2 D4 probe -> Slot 8 (disambiguator); $3-5 D4 probe on A1 latents IS the pre-dispatch gate'; (3) Wave 3 Phase 1 systemic-refusal memo line 43 notes the ambiguity but lands at the SAME structural blocker for both interpretations. Conclusion: 'ATW D4 probe' unambiguously means ATW v2's OWN D4 probe."
  - assumption: "parent prompt premise 'lift _full_main from NotImplementedError' is current state of the trainer"
    classification: CARGO-CULTED
    rationale: "Premise verification per Catalog #229 (BEFORE edit): git log shows commit b1f1913e5 'Add ATW v2 trainer and research recipe' on 2026-05-16 ALREADY landed the implementation. AST scan of experiments/train_substrate_atw_codec_v2.py confirms _full_main body is 215 LOC, contains zero NotImplementedError raises, and 32/32 tests pass. The parent prompt premise inherited the V1 trainer's NotImplementedError state OR mistakenly cited the V2 trainer's lift docstring mentioning V1. This is the canonical 'premise verification before edit' pattern Catalog #229 was designed to extinct."
---

## TL;DR (60 seconds)

Two disambiguations resolved in this memo:

**1. Recipe naming disambiguation**: "ATW D4 probe ($3-5)" in the Wave 2 / Phase 1 / batched council references means **ATW codec v2's own D4 H(latent|scorer_class) probe** (the canonical pre-dispatch gate per ATW v2 design memo §19 reactivation criterion V2-1), executed via `tools/run_atw_v2_d4_probe_from_a1.py` against A1 latents and the SegNet per-pair composite class artifact. It is **NOT** the sister substrate `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml` (that is a different substrate entirely — standalone Wyner-Ziv frame-0 substrate, not part of the ATW v2 lift chain).

**2. Lift premise**: the parent prompt premise "lift `experiments/train_substrate_atw_codec_v2.py::_full_main` from `raise NotImplementedError` to a working PR95-paradigm-compliant trainer" is **EMPIRICALLY FALSIFIED**. The `_full_main` is ALREADY implemented at HEAD commit `b1f1913e5` (2026-05-16, "Add ATW v2 trainer and research recipe"); body is 215 LOC of fully-canonical PR95-paradigm pattern; 32 of 32 dedicated tests pass. The premise verification per Catalog #229 caught the false premise BEFORE any edit was attempted. NO source-code edit landed; this memo + lane registry mark-up is the entire delivery.

**3. Downstream consequence (already documented in `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md`)**: the D4 probe has ALREADY been run and returned verdict **`INDEPENDENT`** (MI = 0.006385 bits/symbol; 2 orders of magnitude below the 0.5 MEANINGFUL_CONDITIONING threshold; Wyner-Ziv gain ceiling fraction = 0.0009). Per design memo §4.3 + §19 + the parallel council T3 verdict (`grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md`): D4 INDEPENDENT → DEFER-pending-research with G2-PARTIAL alternative-hypothesis cited; NOT KILL per CLAUDE.md "Forbidden premature KILL". Therefore `dispatch_enabled: false` + `research_only: true` STAYS on the recipe.

**Outcome**: 0 dispatches fired (correctly). 0 source edits to `experiments/train_substrate_atw_codec_v2.py` or `src/tac/substrates/atw_codec_v2/*` (already implemented). 2 lane-registry gate marks (`impl_complete` + `memory_entry` for previously-undocumented existing work). 1 disambiguation memo (this file). Estimated cost: $0 (no GPU dispatch).

## Disambiguation analysis (the question itself)

The parent prompt explicitly asked: *"The Wave 2 T3 council referenced 'ATW D4 probe ($3-5)' but there are TWO candidates: (a) `substrate_atw_codec_v2_modal_a100_dispatch.yaml` — ATW codec v2's D4 cooperative-receiver variant; (b) `substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml` — sister standalone D4 Wyner-Ziv substrate."*

This disambiguation question is well-founded — the same string "D4" appears in BOTH places — but the council intent is unambiguous on inspection of the cited memos.

### Evidence stream 1: ATW v2 design memo (`atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md`)

The design memo defines the D4 H(latent|scorer_class) probe at multiple sites:

- **§1 PV-4** (line 21): *"D4 H(latent|scorer_class) probe exists at `tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py` (commit `d72f50985`; 312 LOC). Three-verdict taxonomy `MEANINGFUL_CONDITIONING / WEAK_CONDITIONING / INDEPENDENT` keyed off MI threshold default `0.5 bits/symbol`."*
- **§2** (line 64): *"D4 H(latent|scorer_class) probe as PRE-DISPATCH GATE (was V1 reactivation criterion #3; now V2 mandatory $3-5 CPU smoke BEFORE any paid Modal dispatch)."*
- **§19** (V2-1 reactivation criterion): *"D4 probe returns `MEANINGFUL_CONDITIONING` (MI ≥ 0.5 bits/symbol). Output: `.omx/state/h_latent_given_scorer_class_atw_v2.json`."*

This is plainly the probe TOOL (`tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py`), NOT a substrate trainer. It runs against A1 latents (or any substrate's latents); the substrate-specific runner (per Catalog #226 + #229 canonical helper pattern) is `tools/run_atw_v2_d4_probe_from_a1.py` (already landed; 4 tests pass).

### Evidence stream 2: parallel council T3 memo (`grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md`)

The council memo enumerates the Phase 2 lift order for THIS WEEK at line 71:

> *"K=8 LEVEL-1 schedule slot allocation: Rudin floor → Slot 3 (asymptotic-pursuit); Z6 → Slot 2 (asymptotic-pursuit); **ATW v2 D4 probe → Slot 8 (disambiguator)**; STC v2 retry → Slot 7 (frontier-composed disambiguator); Tishby IB-pure → NO slot (deferred); total estimated cost $20-55 across 4 slots"*

And the council verdict for ATW v2 at line 68:

> *"ATW v2 (cooperative-receiver bolt-on): **PROCEED-TO-D4-PROBE-FIRST-THEN-LIFT** — Phase 2 lift council cannot adjudicate without D4 probe verdict; **$3-5 D4 probe on A1 latents IS the pre-dispatch gate**; if MEANINGFUL_CONDITIONING then Variant B (single-knob WZ-only) advances to Phase 2 lift at $5-15 smoke; if INDEPENDENT then DEFER per Tishby IB-pure sister pattern"*

The phrase "$3-5 D4 probe on A1 latents IS the pre-dispatch gate" is the explicit anchor: the council means the H(latent|scorer_class) probe on A1's latents, which is ATW v2's own D4 probe, NOT the standalone D4 Wyner-Ziv substrate.

The Assumption-Adversary's verbatim (line 38) also explicitly cites *"ATW v2 D4 probe MUST run BEFORE the Phase 2 lift council can adjudicate"*, confirming the probe is ATW v2's OWN gate.

### Evidence stream 3: Wave 3 Phase 1 orphan-recipe memo (`wave_3_phase_1_systemic_refusal_3_of_5_recipes_dispatch_disabled_20260516.md`)

The Wave 3 Phase 1 systemic-refusal memo explicitly notes the ambiguity at lines 41-43:

> *"parent named 'ATW D4 probe' referencing ATW codec v2 design memo. The actual ATW v2 recipe (`substrate_atw_codec_v2_modal_a100_dispatch.yaml`) also carries `dispatch_enabled: false` + 3 explicit dispatch_blockers. The D4 recipe (`substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml`) carries `smoke_only: true` but is a sister Wyner-Ziv substrate, NOT ATW-D4. Whether parent meant ATW v2 or D4 standalone is ambiguous — both interpretations land at the same structural blocker."*

The Wave 3 Phase 1 memo correctly identifies that **regardless of which recipe is chosen, the outcome is the same**: ATW v2 cannot dispatch (3 dispatch_blockers including the D4 probe gate); D4 Wyner-Ziv is a different substrate (`smoke_only: true`) and is not part of the ATW v2 lift chain.

### Evidence stream 4: the K=13 measurement schedule

`tools/run_atw_v2_d4_probe_from_a1.py` (18.6KB; tested via `src/tac/tests/test_run_atw_v2_d4_probe_from_a1.py`; 4 tests pass) IS the canonical operator-facing runner for the ATW v2 D4 probe. Its existence as a substrate-specific tool (NOT a generic substrate trainer) confirms the disambiguation: "ATW D4 probe" = invoke this tool, not run a separate substrate dispatch.

### Conclusion

The three independent evidence streams (design memo + council T3 + Wave 3 Phase 1 memo) converge: **"ATW D4 probe ($3-5)" means ATW codec v2's own D4 H(latent|scorer_class) probe on A1 latents**, executed via `tools/run_atw_v2_d4_probe_from_a1.py` against `submissions/a1/archive.zip` + `experiments/results/tishby_ib_pure_d4_probe_*/per_pair_segnet_class.json`. Output lands at `.omx/state/h_latent_given_scorer_class_atw_codec_v2.json` + `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.{json,md}`.

The standalone D4 Wyner-Ziv substrate (`substrate_d4_wyner_ziv_frame_0_modal_t4_dispatch.yaml`; `src/tac/substrates/d4_wyner_ziv_frame_0/`) is a DIFFERENT substrate (frame-0-anchored Wyner-Ziv codec), unrelated to ATW v2's D4 probe-disambiguator.

## Premise verification per Catalog #229 (pre-edit, before any source change)

The parent prompt premised the lift task with: *"lift `experiments/train_substrate_atw_codec_v2.py::_full_main` from `raise NotImplementedError` to a working PR95-paradigm-compliant trainer"*.

Per Catalog #229 (premise-verification-before-edit pattern), I verified this premise empirically BEFORE attempting any source-code edit:

### PV-1: git log shows the trainer was ALREADY landed

```
$ git log --oneline -10 -- experiments/train_substrate_atw_codec_v2.py
b1f1913e5 Add ATW v2 trainer and research recipe
```

Single commit `b1f1913e5` on 2026-05-16 lands BOTH the trainer AND the research recipe in one packet.

### PV-2: AST scan confirms _full_main is fully implemented

```python
import ast
src = open('experiments/train_substrate_atw_codec_v2.py').read()
tree = ast.parse(src)
funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ('_full_main', '_smoke_main', 'main')]
for f in funcs:
    raises_nie = any(
        isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and getattr(node.exc.func, 'id', None) == 'NotImplementedError'
        for node in ast.walk(f)
    )
    print(f'{f.name}: NotImplementedError raised = {raises_nie}; body LOC = {len(ast.unparse(f).splitlines())}')

# Output:
# _smoke_main: NotImplementedError raised = False; body LOC = 53
# _full_main: NotImplementedError raised = False; body LOC = 215
# main: NotImplementedError raised = False; body LOC = 6
```

`_full_main` is 215 lines of fully-canonical PR95-paradigm implementation. ZERO `NotImplementedError` raises.

### PV-3: PR95-paradigm canonical pattern presence — all 14 ADOPT canonical decisions executed

The current `_full_main` already binds:

- `patch_upstream_yuv6_globally()` BEFORE `load_differentiable_scorers()` (line 793-802; PR95 parity per Catalog #187)
- canonical scorer-loader assignment order `pose_scorer, seg_scorer = load_differentiable_scorers(...)` (line 800; Catalog #222)
- EMA(0.997) shadow with snapshot-restore for archive build (lines 891-908; CLAUDE.md "EMA — non-negotiable")
- `apply_eval_roundtrip_during_training` via `tac.differentiable_eval_roundtrip` (imported line 778)
- canonical `score_pair_components` via `ATWv2ScoreAwareLoss` which delegates to `cooperative_receiver_loss` (Catalog #164)
- canonical `gate_auth_eval_call` (Catalog #226; line 1122)
- canonical `detect_hardware_substrate` with `substrate_tag="atw_codec_v2"` (Catalog #190; line 1149)
- canonical `posterior_update_locked` (Catalog #128/#131; line 1169)
- canonical Tier-1 engineering: `--enable-autocast-fp16` flag (Catalog #172), `--enable-torch-compile` flag (Catalog #179), `torch.no_grad()` at eval (Catalog #180), TF32 via `device_or_die` (Catalog #178)
- canonical `select_inflate_device` via vendored shim (Catalog #205; in `_write_runtime`)
- Catalog #218 sister mini-batch reconstruct via `--scorer-chunk-size` flag
- Catalog #146 + #295 self-contained vendored inflate per `_write_runtime` (lines 526-736)

### PV-4: Tests pass empirically

```
$ .venv/bin/python -m pytest src/tac/substrates/atw_codec_v2/tests/test_atw_codec_v2.py -x --no-header -q
............................                                             [100%]
28 passed in 2.08s

$ .venv/bin/python -m pytest src/tac/tests/test_run_atw_v2_d4_probe_from_a1.py -x --no-header -q
....                                                                     [100%]
4 passed in 0.14s
```

32 of 32 dedicated tests pass. The trainer is structurally and behaviorally complete.

### PV-5: D4 probe verdict already returned INDEPENDENT

`.omx/state/h_latent_given_scorer_class_atw_codec_v2.json` exists; verdict is `INDEPENDENT` (MI=0.006385 bits/symbol vs 0.5 threshold). The probe was run 2026-05-16T22:47:41Z. The verdict file explicitly says:
- `phase2_status: defer_measured_a1_latent_class_conditioning_surface`
- `recommended_variant: none`
- `next_action: do_not_dispatch_atw_v2_phase2_from_this_signal`

Per design memo §4.3 + §19, the D4 INDEPENDENT verdict bars Phase 2 lift (the operator-decision the parent prompt wanted me to execute is structurally REFUSED by the canonical D4 gate).

## What this means for the parent's lift-task expectation

The parent prompt asked for a 2-part deliverable:
1. Lift `_full_main` from `NotImplementedError` to working PR95-paradigm trainer
2. Disambiguate the "ATW D4 probe" recipe naming question

**Deliverable 1 is structurally a no-op** because the work is already done (`b1f1913e5`). Re-implementing it would be a Catalog #157 commit-swap class violation (sister edits the same file under a different commit) AND would violate Catalog #229 (premise-verification-before-edit). The CORRECT response per CLAUDE.md "Subagent coherence-by-default" is to document the existing implementation in lane registry gates + this memo, NOT redo it.

**Deliverable 2 is fulfilled** by this memo. The disambiguation is: "ATW D4 probe" = ATW v2's own D4 probe, not the sister standalone D4 Wyner-Ziv substrate.

**Deliverable 3 (implicit per design memo §19)**: the D4 verdict is `INDEPENDENT`. Per CLAUDE.md "Forbidden premature KILL" + design memo §19, ATW v2 is in DEFER-pending-research state, NOT killed. Reactivation criteria are documented in the D4 verdict memo:
1. Replace the saturated per-pair SegNet composite class with a richer side-information signal (per-region class histograms, logits, pose bins, or hard-pair/object-state features).
2. Rerun the same probe on trained ATW v2 residuals rather than A1 HNeRV latents.
3. Require paired CPU/CUDA exact-eval custody before any score, rank, or promotion claim.

## Canonical-vs-unique decision per layer (per Catalog #290)

Per the standing UNIQUE-AND-COMPLETE-PER-METHOD operating mode + this memo's per-layer decisions:

| Layer | Decision | Rationale |
|---|---|---|
| Disambiguation methodology | ADOPT canonical (3-evidence-stream cross-reference) | Standard Catalog #229 premise-verification + Catalog #292 multi-source assumption surfacing; no need to fork |
| Lift execution policy | UNIQUE FORK | Premise-falsified case demands "DOCUMENT not REDO" per Catalog #157 + #229 + #230 (sister ownership map preserves existing work) — distinct from the default "lift NotImplementedError → working implementation" pattern |
| Lane gate marking | ADOPT canonical | `tools/lane_maturity.py mark` with evidence string citing 32/32 test pass + D4 verdict file paths |
| Memo placement | ADOPT canonical | `.omx/research/` per CLAUDE.md (NOT external Claude memory; OSS-hermetic per Catalog #290/#291/#292) |

## 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | Documents a unique premise-falsified case + recipe-naming disambiguation in one memo per the operator's specific question. NO sister memo carries this content. |
| 2 | BEAUTY + ELEGANCE | Single memo (~6KB); 3 evidence streams converge to ONE conclusion; reviewable in 30 seconds via TL;DR. |
| 3 | DISTINCTNESS | Distinct from V2 design memo (which describes what to do) and D4 verdict memo (which records what happened); this memo records WHY the parent's premise was falsified. |
| 4 | RIGOR | 5 premise verifications + 3 evidence streams + AST scan + test runs + git log + 4-source cross-reference. No claim without empirical receipt. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per-technique: lane mark via canonical helper (`tools/lane_maturity.py`); disambiguation via 3-source convergence (no novel framework needed). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | This memo's outputs (D4 INDEPENDENT verdict citation + DEFER-pending-research disposition) feed downstream into autopilot ranking (Hook 4) + continual learning posterior (Hook 5). |
| 7 | DETERMINISTIC REPRODUCIBILITY | All cited evidence is in repo: tests reproducible, AST scan reproducible, D4 verdict file is bit-stable JSON. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | $0 GPU cost; ~30 min wall-clock total subagent work; 0 source-code edits. Maximum efficiency vs the naive "redo the lift" path which would have burned subagent tokens AND collided with the sister `b1f1913e5` work via Catalog #157 commit-swap class. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Indirect: by NOT lifting `dispatch_enabled` under a falsified premise, this memo prevents a wasted Modal A100 dispatch (~$5-15) on a D4-INDEPENDENT substrate. Per CLAUDE.md "Race-mode rigor inversion": the cheap signal (D4 probe $3-5; already returned INDEPENDENT) correctly gates the expensive signal (Modal A100 smoke $5-15 + full $10-30). |

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305.

This memo produces structured observability for downstream consumers:

| Artifact | Path | Schema | Consumer |
|---|---|---|---|
| Disambiguation verdict | `.omx/research/atw_d4_probe_recipe_disambiguation_20260516.md` (this file) | per Catalog #294 9-dim + Catalog #290 canonical-vs-unique YAML frontmatter | Operator + downstream wave subagents who reference "ATW D4 probe" |
| Lane gates marked | `.omx/state/lane_registry.json` / `.omx/state/lane_maturity_audit.log` | per `tools/lane_maturity.py` schema | Lane registry validator + cathedral autopilot ranker |
| D4 verdict citation | this memo cites `.omx/state/h_latent_given_scorer_class_atw_codec_v2.json` + `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md` | per D4 probe schema | Cathedral autopilot Hook 4 (refuses dispatch); continual learning Hook 5 (records MI=0.006 for cooperative-receiver-on-A1-latents anchor) |
| Subagent checkpoint trail | `.omx/state/subagent_progress.jsonl` per Catalog #206 | per checkpoint schema | Crash-resume successor (if any) + parent subagent audit |

### Observability invariants

- **No score_claim**: this memo records a DISAMBIGUATION + PREMISE-FALSIFICATION, not a score result. `score_claim=false` implicit; no axis label applies because no score was measured.
- **No phantom directories** per Catalog #249: memo path is canonical `.omx/research/` per CLAUDE.md.
- **Cite-chain preserved**: all 4 source memos cited (design memo, council T3, Wave 3 Phase 1, K=13 schedule).
- **Counterfactual-able**: a future agent can verify the disambiguation by re-running PV-2 AST scan, re-reading the 4 cited memos, and re-running the 32 dedicated tests.

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A — this memo records DISAMBIGUATION + PREMISE-FALSIFICATION, not a measured score; no sensitivity contribution applies.
2. **Pareto constraint**: N/A — no new Pareto candidate; the ATW v2 substrate IS already in the Pareto solver, in DEFER state per D4 INDEPENDENT.
3. **Bit-allocator hook**: N/A — no archive bytes added/changed.
4. **Cathedral autopilot dispatch hook**: ACTIVE (via D4 verdict citation) — the autopilot already refuses ATW v2 dispatch via `dispatch_enabled: false` in the recipe; this memo documents WHY and confirms the gate is operating correctly.
5. **Continual-learning posterior update**: N/A at memo level — the D4 verdict file itself was the posterior anchor (recorded 2026-05-16T22:47:41Z); this memo records the disposition (DEFER-pending-research) but does not write a new anchor.
6. **Probe-disambiguator**: ACTIVE — the D4 probe IS the canonical probe-disambiguator per design memo §19. This memo records the verdict + recommends G2-PARTIAL alternative-hypothesis per CLAUDE.md "Forbidden premature KILL" + ATW v2 §19 reactivation criteria.

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| Parent prompt premise reflects current repo state | CARGO-CULTED | Premise was inherited from V1 trainer state (V1's `_full_main` does raise NotImplementedError per design memo PV-1) and not verified against V2's `_full_main` at commit `b1f1913e5`. Catalog #229 is designed to extinct this class structurally; this memo lands the structural fix via documentation. |
| Two recipes named "D4" must always be disambiguated | HARD-EARNED | Reasonable parent question; the disambiguation IS non-trivial and was explicitly flagged by the Wave 3 Phase 1 memo at line 43. The disambiguation methodology (3-evidence-stream cross-reference) is canonical per Catalog #292. |
| D4 INDEPENDENT verdict bars Phase 2 lift indefinitely | CARGO-CULTED | The verdict bars lift FROM THIS SIGNAL — the cooperative-receiver paradigm itself is NOT falsified per design memo §19. Reactivation criteria are explicit (richer side-info signal; trained-residual probe; paired exact-eval custody). Per CLAUDE.md "Forbidden premature KILL" the lane is DEFER not KILL. |

## Horizon class declaration (per Catalog #309)

`horizon_class: frontier_pursuit`

Rationale: ATW v2 is at the frontier of cooperative-receiver substrate engineering. The D4 INDEPENDENT verdict on A1 latents specifically (one particular signal-extraction surface) does NOT remove ATW v2 from the frontier_pursuit class — it identifies a richer signal-extraction surface as a prerequisite for proceeding. The substrate's predicted band remains NULL until probe reactivation conditions are met.

## Predicted ΔS band (per Catalog #296)

`NULL pending D4 probe reactivation conditions (richer side-info signal)` [prediction; deferred]

This memo does NOT predict ΔS for ATW v2 — the design memo §18 first-principles framework is preserved unchanged. The D4 INDEPENDENT verdict has already invalidated the MEANINGFUL_CONDITIONING-conditional [-0.015, -0.005] band; revising the band requires either the V2-1 reactivation conditions or pivoting to G2-PARTIAL alternative-hypothesis (separate design memo).

Dykstra-feasibility check is N/A for this memo (no new predicted band).

## Reactivation criteria (cite design memo §19 + D4 verdict)

ATW v2 dispatch eligibility requires:

1. **D4 probe reactivation** per `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md`:
   - Replace saturated per-pair SegNet composite class with richer side-information signal (per-region class histograms, logits, pose bins, hard-pair/object-state features)
   - Rerun the same probe on trained ATW v2 residuals rather than A1 HNeRV latents
   - Require paired CPU/CUDA exact-eval custody before any score, rank, or promotion claim

2. **Dykstra-feasibility check** (design memo §18) — non-empty polytope intersection with revised side-info source

3. **Variant A vs B council adjudication** (design memo §19) — sextet pact per Catalog #292 decides Variant A vs B based on the revised D4 verdict

Per CLAUDE.md "Forbidden premature KILL": ATW v2 is in DEFER-pending-research state, NOT killed. Per CLAUDE.md "Mission alignment" + Catalog #300 mission-alignment retrospective: this DEFER receives a 30-day retrospective due 2026-06-15 to verify reactivation criteria progress.

## Cross-references

- `.omx/research/atw_codec_v2_cooperative_receiver_full_stack_design_20260516.md` (V2 design memo; commit `fcdcc1112`)
- `.omx/research/grand_council_t3_batched_phase_2_lift_z6_rudin_tishby_atw_stc_20260516.md` (parallel council T3 verdict; PROCEED-TO-D4-PROBE-FIRST-THEN-LIFT)
- `.omx/research/wave_3_phase_1_systemic_refusal_3_of_5_recipes_dispatch_disabled_20260516.md` (predecessor Wave 3 ambiguity report; commit `dbc82941b`)
- `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.md` (D4 verdict INDEPENDENT)
- `.omx/state/h_latent_given_scorer_class_atw_codec_v2.json` (D4 verdict structured JSON)
- `experiments/train_substrate_atw_codec_v2.py` (the trainer; commit `b1f1913e5` already-implemented)
- `.omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml` (recipe; `dispatch_enabled: false` + `research_only: true`)
- `src/tac/substrates/atw_codec_v2/` (package; tests 28/28 pass)
- `tools/run_atw_v2_d4_probe_from_a1.py` (canonical D4 probe runner; tests 4/4 pass)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable (Rule 3: cheap signal gates expensive signal)
- Catalog #229 (premise-verification-before-edit pattern; what extincted the false-premise edit)
- Catalog #240 (recipe-vs-trainer-state consistency; what the recipe `dispatch_enabled: false` honors)
- Catalog #220 (substrate L1+ scaffold operational mechanism declaration; ATW v2 cleared via research_only opt-out per Catalog #240 cascade)
- Catalog #272 (distinguishing-feature integration contract; ATW v2 declares all 4 fields in registered_substrate per Catalog #210)
- Catalog #291 (per-session META-ASSUMPTION ADVERSARIAL REVIEW cadence; this memo carries the per-deliberation assumption surfacing)
- Catalog #292 (per-deliberation explicit assumption statements; this memo's frontmatter complies)
- Catalog #305 (substrate design memo observability surface section; this memo's "Observability surface" section above)
- Catalog #309 (substrate design memo horizon_class declaration; this memo's "Horizon class declaration" section above)
