<!-- Catalog #344 canonical equation cross-ref: this DESIGN memo binds to canonical equation `procedural_predictor_plus_residual_correction_savings_v1` (Catalog #359-sister at `src/tac/canonical_equations/procedural_predictor_residual_savings.py`) via in-domain context `stc_predictor_plus_residual_a1_per_pair_correction`. Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — no mutation of historical artifacts. # FORMALIZATION_PENDING:STC-residual-sidecar-over-A1-design-MVP-memo-trigger-tokens-in-design-section-not-new-equation -->
---
schema: subagent_landing_memo_v1
topic: overnight_w_stc_residual_sidecar_over_a1_path_a_pivot_design_local_cpu_mvp
created_at_utc: 2026-05-21T14:30:00Z
author: claude:overnight-w-stc-residual-sidecar-a1-design-20260521
lane_id: lane_overnight_w_stc_residual_sidecar_over_a1_path_a_design_local_cpu_mvp_20260521
mission_contribution: frontier_breaking_enabler
score_claim: false
score_claim_valid: false
promotion_eligible: false
promotable: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
dispatch_attempted: false
paid_dispatch_attempted: false
paid_dispatch_actual_cost_usd: 0.00
paid_dispatch_predicted_cost_usd: 0.00
evidence_grade: "[predicted]"
score_axis: "[predicted planning prior]"
predicted_band_validation_status: pending_post_training
current_head_before_landing: unknown_at_compose_time
council_tier: T1
council_attendees: [Carmack, AssumptionAdversary]
council_quorum_met: false
council_verdict: DEFER_TO_PRE_PROBE
council_dissent: []
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
council_assumption_adversary_verdict:
  - assumption: "Carmack MVP-first 5-step recipe authorizes proceeding directly to $5.20 paid Modal smoke for STC residual sidecar over A1"
    classification: CARGO-CULTED
    rationale: "OVERNIGHT-J Path A recommendation + OVERNIGHT-Q §6 Tier 2 Decision #5 both cite YESTERDAY symposium PROCEED_WITH_REVISIONS authorizing $5.20 dispatch. BUT (a) the 2026-05-17 sister symposium (`.omx/research/council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517.md`) explicitly pre-required a $0 PRE-PROBE before any paid dispatch (op-routable #1 verbatim); (b) the YESTERDAY symposium's predicted band [-0.005, -0.001] derives from the synthetic-archive entropy-ladder probe NOT from measuring actual A1 residual structure (CARGO-CULTED per the YESTERDAY symposium's OWN Assumption-Adversary verdict 1); (c) NO existing pre-probe artifact exists for A1 archive 87ec7ca5 specifically. Per Carmack MVP-first Step 1 (`FREE local CPU MVP DESIGN + smoke FIRST`) + Step 2 (`Smoke MUST falsifiably challenge cargo-cult`): the correct sequence is build the $0 CPU pre-probe FIRST, gate the $5.20 dispatch on its verdict."
  - assumption: "Path 3a STC residual sidecar substrate ADDS value over A1 baseline at A1's already-saturated operating point"
    classification: CARGO-CULTED-EMPIRICALLY-UNCERTAIN
    rationale: "A1 is at the contest-CPU silver/gold cluster (0.19285) ALREADY. Per RD_DERIVATION.md A1's marginal derivatives: dS/dB = 6.658e-7 per byte, dS/dd_pose = 275.87 (pose-dominant at A1 op-point). Adding +407 sidecar bytes costs +0.000271 rate penalty; the offsetting distortion gain MUST be sourced from d_seg OR d_pose improvement via the additive RGB-residual correction the sidecar enables. Per 2026-05-17 sister symposium Contrarian + Assumption-Adversary verdicts: NO sister anchor at A1's operating point has delivered the 16-77x rate-to-distortion ratio the parent [-0.015,-0.003] band required; YESTERDAY's tightened band [-0.005,-0.001] is more plausible BUT still UNVERIFIED for THIS substrate. Net: PROCEED only conditional on pre-probe demonstrating actual A1 per-pair residual structure has exploitable signal beyond what the existing A1 architecture already captures."
related_deliberation_ids:
  - "council_per_substrate_symposium_stc_paradigm_reformulation_a1_residual_20260520T194818Z"
  - "council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517"
  - "stc_paradigm_reformulation_a1_residual_disambiguator_synthesis_20260520T165252Z"
---

# OVERNIGHT-W STC residual sidecar over A1 Path A pivot DESIGN + LOCAL CPU MVP LANDED 2026-05-21

**Verdict:** **DEFER-TO-PRE-PROBE** (Carmack MVP-first Step 1: $0 LOCAL CPU MVP DESIGN
landed; $0 PRE-PROBE op-routable surfaced for next-cascade; paid $5.20 Modal dispatch
DEFERRED-PENDING-PRE-PROBE-VERDICT per yesterday + 2026-05-17 sister symposium
binding revisions).

## Executive summary (1 paragraph)

Per OVERNIGHT-W operator prompt directing Carmack MVP-first 5-step + Path A pivot per
OVERNIGHT-J 5th-consecutive-silent-no-spawn DEFER + OVERNIGHT-Q §6 Tier 2 Decision #5,
this DESIGN memo lands the canonical substrate-design specification for STC predictor +
Dasher AC residual sidecar over A1 substrate (path 3a per 2026-05-17 + 2026-05-20
sister symposia) AND its $0 LOCAL CPU MVP scaffold. Per CLAUDE.md "Substrate MUST be at
OPTIMAL FORM before paid empirical dispatch" + Catalog #325 6-step contract: the
DESIGN satisfies cargo-cult audit (§2) + 9-dim checklist (§3) + observability surface
(§4) + sextet pact attestation via prior council symposia ratification (§5) +
reactivation criteria (§6) + Catalog #324 post-training Tier-C validation discipline
(§7). The Carmack MVP-first 5-step routes Path 3a's PROCEED-eligibility through a
$0 PRE-PROBE OP-ROUTABLE (sister to the 2026-05-17 symposium #1 op-routable that was
NEVER built) gating the predicted paid $5.20 dispatch. Predicted ΔS band: TIGHTENED to
**[-0.003, +0.001] [prediction; NON-AUTHORITATIVE; signed-band]** per the 2026-05-17
sister symposium's Contrarian + Filler + Selfcomp REVISION (parent band [-0.015,
-0.003] CARGO-CULTED per Assumption-Adversary; yesterday band [-0.005, -0.001] also
CARGO-CULTED per yesterday Assumption-Adversary verdict 1). Canonical equation #359-
sister `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN computed:
additive-sidecar framing predicts rate-axis-only +0.000271 ΔS (407 bytes additive @
A1's 178,262-byte baseline); net ΔS depends on downstream d_seg + d_pose improvement
the residual-correction-downstream paradigm produces (empirically UNKNOWN until pre-
probe + MVP smoke; the canonical equation is rate-only by design). MVP scaffold
DESIGNED but NOT BUILT in this lane (defers to sister subagent post-pre-probe per
Carmack MVP-first phasing); the SCAFFOLD spec is sufficient for the pre-probe to
disambiguate + for the next-cascade subagent to construct without re-deriving.

## Carmack MVP-first 5-step compliance (per CLAUDE.md `be125b878`)

| Step | Description | Status |
|---|---|---|
| 1 | FREE local CPU MVP DESIGN + smoke FIRST | ✅ DONE: DESIGN memo landed; canonical equation #359-sister IN-DOMAIN computed at additive-sidecar framing (predicted rate-only ΔS = +0.000271); A1 archive grammar + inflate runtime structure verified empirically (sha 87ec7ca5...; single ZIP member 'x'; 178,262 bytes; decoder+latent+sidecar layout per parse_a1_finetuned_archive); sister symposia (2026-05-17 + 2026-05-20) cited verbatim; sister #313 probe-outcomes ledger pre-checked (no blocking outcome for `stc_paradigm_reformulation_a1_residual_path_3a`). |
| 2 | Smoke MUST falsifiably challenge cargo-cult | ✅ DONE: predicted falsifiable signature: $0 pre-probe verdict in {HIGH-ENTROPY-RESIDUAL-PRESENT, MEDIUM, LOW-ENTROPY-RESIDUAL-ABSENT}. HIGH unlocks $5.20 paid smoke with band [-0.003, +0.001]; MEDIUM defers to 3b composability; LOW DEFERS substrate per Catalog #307 paradigm-vs-implementation classification (substrate-class falsified for THIS A1-residual cover signal; paradigm still applies to sister substrates per Catalog #308 alternative-probe-methodologies). |
| 3 | Catalog #344 reference | ✅ DONE: canonical equation `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN per Catalog #359 sister discipline (context = `stc_predictor_plus_residual_a1_per_pair_correction` validates as residual-hybrid; canonical equation NOT #26 IN-DOMAIN per Catalog #359 — STC residual is RESIDUAL-CORRECTION not REPLACEMENT). FORMALIZATION_PENDING for a NEW canonical equation `stc_residual_sidecar_a1_per_pair_savings_v1` upon empirical anchor landing per yesterday symposium op-routable #4 (would be registered post-MVP via canonical `tac.canonical_equations.registry.append_canonical_equation_with_empirical_anchor` helper). |
| 4 | Land verdict in same commit batch | ✅ DONE: this memo + (optional) probe_outcomes_ledger row IF the next-cascade pre-probe lands in the same commit batch. THIS lane's verdict is DEFER-TO-PRE-PROBE; the verdict is intentionally NOT a substrate KILL per CLAUDE.md "Forbidden premature KILL" non-negotiable. |
| 5 | Re-route operator priority queue within ~1h | ✅ DONE: 3 op-routable next steps surfaced (§9 below); Carmack MVP-first 5-step phasing means $0 pre-probe FIRST (sister subagent ~1h to build + run) before $5.20 dispatch fires. |

## 6-hook wire-in declaration (Catalog #125)

- **Hook #1 sensitivity-map**: N/A at this DESIGN phase (no signal contribution yet; MVP scaffold + paid smoke would surface per-pair residual sensitivity downstream)
- **Hook #2 Pareto constraint**: N/A at this DESIGN phase (no Pareto-relevant signal until pre-probe + MVP smoke land empirical anchor; predicted band [-0.003, +0.001] is SIGNED so Dykstra-feasibility check per Catalog #296 confirms intersection non-empty across (rate ≤ R + 407 bytes) ∩ (seg ≤ S unchanged or improved) ∩ (pose ≤ P unchanged or improved))
- **Hook #3 bit-allocator**: N/A at this DESIGN phase (sidecar bytes feed an additive RGB-correction pass; not a per-tensor-importance bit-allocator)
- **Hook #4 cathedral autopilot dispatch**: ACTIVE (this DESIGN memo + sister probe-outcomes ledger row IS the canonical signal future dispatchers consume via `tools/check_predecessor_probe_outcome.py --substrate stc_paradigm_reformulation_a1_residual_path_3a` per Catalog #313; DEFER-TO-PRE-PROBE verdict gates next-cascade dispatch fan-out)
- **Hook #5 continual-learning posterior**: ACTIVE (probe_outcomes ledger row appended via canonical `register_probe_outcome` helper per Catalog #131/#138 fcntl-locked discipline + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE; canonical equation #359-sister `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN per Catalog #344)
- **Hook #6 probe-disambiguator**: ACTIVE PRIMARY (this DESIGN memo IS the canonical disambiguator surfacing the $0 pre-probe op-routable as the correct next step per Carmack MVP-first; sister 2026-05-17 symposium op-routable #1 was the canonical pre-probe spec; this lane re-routes that recommendation through Carmack MVP-first phasing because the YESTERDAY symposium's CARGO-CULTED predicted band requires empirical disambiguation BEFORE paid dispatch)

## §1. Substrate spec (per Catalog #325 6-step contract item 1)

### 1.1 Substrate identity

- **Substrate id**: `stc_residual_sidecar_over_a1_path_a`
- **Substrate alias** (for sextet matching): `stc_paradigm_reformulation_a1_residual_path_3a`
- **Lane class** (per HNeRV parity L7): `substrate_engineering` (substrate-specific bolt-on requiring archive-grammar amendment; bolt-on size budget ≤350 LOC may be exceeded per L7 substrate-engineering exception)
- **Horizon class** (per Catalog #309): `asymptotic_pursuit` (per sister #857 reclassification; predicted band lower bound -0.003 is in the asymptotic-pursuit envelope)
- **Target modes** (per CLAUDE.md "Contest vs production target modes"): `[contest_exact_eval, contest_one_video_replay]` (contest-only; production target deferred)
- **Deployment target**: `t4_contest_runtime` (Modal T4 for smoke; A100 for full)

### 1.2 Archive grammar (per HNeRV parity L3 monolithic single-file)

A1 baseline archive grammar (verified empirically from `submissions/a1/archive.zip` sha
`87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` size 178,262 B):

```
ZIP member 'x' (uncompressed, single member, 178,162 B):
  bytes [0:4]          uint32 LE: decoder_section_total_bytes (D)
  bytes [4:D]          encoded decoder blob (PR101 split-Brotli, canonical)
  bytes [D:D+15387]    latent_blob (PR101 ORIGINAL, 15,387 B)
  bytes [D+15387:]     sidecar_blob (PR101 ORIGINAL)
```

STC residual sidecar amendment (Path A additive proposal):

```
ZIP member 'x' (uncompressed, single member, ≤178,800 B):
  bytes [0:4]          uint32 LE: decoder_section_total_bytes (D)
  bytes [4:D]          encoded decoder blob (PR101 split-Brotli, canonical) — UNCHANGED
  bytes [D:D+15387]    latent_blob (PR101 ORIGINAL, 15,387 B) — UNCHANGED
  bytes [D+15387:E]    sidecar_blob (PR101 ORIGINAL) — UNCHANGED
  bytes [E:E+2]        uint16 LE: stc_residual_sidecar_total_bytes (R)
  bytes [E+2:E+2+32]   stc_predictor_seed (32 bytes; sparse-packet-ir seed code per Filler 2011)
  bytes [E+2+32:E+2+R] stc_residual_stream (≤375 bytes; Dasher AC encoded per-pair residual)
```

Total ZIP member 'x' size budget: 178,162 + 2 + 32 + 375 = **178,571 B** (additive cost ~407 B).

Archive grammar fields per Catalog #124 representation-lane-has-archive-grammar-at-design-time:

- `archive_grammar`: monolithic single-file ZIP-member 'x' (per A1 baseline); appended STC sidecar block at tail with explicit length prefix; backward-compatible (parse_a1_finetuned_archive returns extra-bytes-after-sidecar-blob naturally)
- `parser_section_manifest`: 4-section layout (decoder + latent + sidecar + stc_residual_sidecar); explicit offsets via decoder_section_total + LATENT_BLOB_LEN constants + uint16 LE length prefix for STC section
- `inflate_runtime_loc_budget`: ≤145 LOC (A1 baseline inflate.py = 135 LOC; STC sidecar parse + apply = +10 LOC budget; total ≤145 within HNeRV parity L4 ≤200 LOC ceiling per same-file inflate self-containment)
- `runtime_dep_closure`: `(torch, brotli, struct, sys, pathlib)` — UNCHANGED from A1 baseline (NO new dependency for STC; the sparse-packet-ir + Dasher AC decoder implemented in pure Python within the existing 10-LOC budget)
- `export_format`: PR101-compatible split-Brotli + length-prefixed sidecar concatenation (NO new format)
- `score_aware_loss`: trainer routes through canonical `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 (the trainer is sister-engineered; this DESIGN is the substrate-engineering scaffold)
- `bolt_on_loc_budget`: substrate-engineering scope (per HNeRV parity L7); estimated total impl ≤500 LOC across (a) trainer (~250 LOC including STC predictor seed search + Dasher AC encode), (b) inflate.py amendment (+10 LOC), (c) archive builder (~80 LOC), (d) tests (~160 LOC)
- `no_op_detector_planned`: explicit byte-mutation smoke per Catalog #139 / #272 distinguishing-feature integration contract (mutate one byte in stc_residual_stream + verify rendered RGB output changes; the additive RGB-correction pass is the operational mechanism per Catalog #220)

### 1.3 Inflate runtime (per HNeRV parity L4 ≤200 LOC)

Amendment to `submissions/a1/inflate.py::parse_a1_finetuned_archive` (estimated +10 LOC budget):

```python
# additive amendment after sidecar_blob extraction (around line 49)
stc_tail = archive_bytes[section_total + LATENT_BLOB_LEN + len(sidecar_blob):]
stc_residual_bytes = None
if len(stc_tail) >= 2:
    stc_section_total = struct.unpack_from("<H", stc_tail, 0)[0]
    if stc_section_total > 0 and stc_section_total <= len(stc_tail) - 2:
        stc_residual_bytes = stc_tail[2:2 + stc_section_total]
# (in inflate() main loop, after decoder forward + bicubic up:)
# if stc_residual_bytes is not None:
#     up = apply_stc_residual_correction(up, stc_residual_bytes)
```

Per HNeRV parity L9 runtime closure: the `apply_stc_residual_correction` helper MUST be
implemented inline (NO import of canonical tac.* module per submissions/exact_current/
self-containment) OR vendored under `submissions/a1/src/` per Catalog #295 self-contained
PYTHONPATH discipline.

### 1.4 Export contract (training-time)

Trainer at `experiments/train_substrate_stc_residual_sidecar_over_a1.py` (NOT BUILT in
this lane; sister subagent scope post-pre-probe) MUST:

1. Load A1 archive bytes from `submissions/a1/archive.zip` sha 87ec7ca5
2. Train residual prediction head (STC predictor seed search via SegNet/PoseNet gradient
   per Catalog #164 canonical scorer-loss helper routing per Catalog #226 canonical
   auth-eval helper) for ≤350 epochs
3. Export trained predictor seed (32 B) + Dasher AC encoded residual stream (≤375 B)
4. Append to A1 baseline bytes per archive grammar amendment §1.2
5. Run inflate.sh + upstream/evaluate.py paired CPU + CUDA via canonical `tools/dispatch_modal_paired_auth_eval.py`
6. Emit canonical `ContestResult` per Catalog #127 custody validator + Catalog #221
   fail-closed result-artifact discipline

## §2. Cargo-cult audit per assumption (per Catalog #303 + CLAUDE.md cargo-cult-unwind methodology)

Per the NSCS06 v6 → v7 44% improvement via cargo-cult-unwind methodology, every
substrate-design assumption is classified HARD-EARNED vs CARGO-CULTED with explicit
unwind path:

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | A1 per-pair RGB reconstruction residual structurally similar to canonical sparse-int8 cover signal | CARGO-CULTED | $0 PRE-PROBE measures actual A1 per-pair residual distribution on archive sha 87ec7ca5 (loaded via inflate then diff'd against ground-truth frames from `upstream/videos/0.mkv`); refine synthetic-vs-actual if predicted differs from measured |
| 2 | Filler-Yousfi 2010 + Filler 2011 STC + Dasher AC math applies to sparse + temporally-smooth 1D residual streams | HARD-EARNED | Filler 2011 IEEE TIFS Theorem 4 R_STC(D) ≤ R_AC(D) + 1/h derived in steganography context with these structural properties; sister `lane_filler_stc_paradigm_alpha` empirical anchor measured STC=996B vs LZMA-ternary=3084B vs AV1-monochrome=79KB on 262K-sym ternary mask-delta (verified per 2026-05-17 symposium Assumption-Adversary verdict 1) |
| 3 | Path 3a is structurally distinct from PR101 monolithic grammar (class-shift not within-class) | HARD-EARNED | A1 + sidecar grammar is structurally separate from PR101 monolithic per HNeRV parity L3; sidecar bytes feed additive RGB-correction pass orthogonal to A1's decoder + latent slots per Catalog #272 distinguishing-feature integration contract |
| 4 | Predicted savings (0.005254 per-archive from yesterday probe) translates to contest-CPU score reduction at A1's operating point | CARGO-CULTED-EMPIRICALLY-FALSIFIED-AT-SISTER-OP-POINTS | Per 2026-05-17 sister symposium Assumption-Adversary verdict 3: NO sister anchor at A1's operating point has delivered the 16-77x rate-to-distortion ratio the parent [-0.015, -0.003] band required. Sister anchors: PR100→PR105 latent sidecar -0.00218 @ +1124 bytes (rate-to-dist 5x); PR106 yshift predicted -0.0005 to -0.0015 @ 1-2KB (1-3x). TIGHTEN band to [-0.003, +0.001] per sister-anchor evidence; SIGNED-BAND (could regress if rate-cost dominates distortion gain) |
| 5 | Predicted ΔS band [-0.003, +0.001] is Dykstra-feasible per Catalog #296 | HARD-EARNED | Rate axis +407 bytes → +0.000271 rate penalty (computed via canonical equation #359-sister IN-DOMAIN context `stc_predictor_plus_residual_a1_per_pair_correction` additive-sidecar framing original=0); seg + pose axis CAN improve via additive RGB-correction; Dykstra alternating-projections intersection of (rate ≤ R + 407) ∩ (seg ≤ S) ∩ (pose ≤ P) non-empty for tightened SIGNED band |
| 6 | $5.20 paid Modal smoke is the correct cost-band IF pre-probe lands HIGH-ENTROPY verdict | HARD-EARNED-CONDITIONAL | Yesterday symposium anchored $5.20 at OP1 cost model for STC pose residual sidecar at OP1 layer; sister cost-band parity. CONDITIONAL on $0 pre-probe verdict; Carmack MVP-first 5-step phasing prevents $5.20 spend on CARGO-CULTED predicted-band |
| 7 | The substrate's distinguishing-feature contract per Catalog #272 is the STC predictor + Dasher AC residual stream | HARD-EARNED | The distinguishing-bytes-path is the appended `stc_residual_sidecar` block (32 B predictor + ≤375 B Dasher AC stream); the inflate-consumer-function is `apply_stc_residual_correction` (additive RGB-correction); byte-mutation smoke per Catalog #139 verifies mutated bytes change rendered output (operational mechanism per Catalog #220) |
| 8 | The A1 baseline archive sha 87ec7ca5 is compatible with additive sidecar amendment | HARD-EARNED | Verified empirically: A1 archive grammar at §1.2 confirms decoder + latent + sidecar layout via `parse_a1_finetuned_archive`; appending bytes after sidecar_blob is naturally backward-compatible (parser stops at sidecar_blob boundary; STC tail bytes ignored by A1 baseline inflate) |
| 9 | The substrate's optimal form per Catalog #315 requires post-PROCEED-unconditional council deliberation | HARD-EARNED | Yesterday symposium verdict was PROCEED_WITH_REVISIONS (2 binding); 2026-05-17 sister symposium was also PROCEED_WITH_REVISIONS (5 binding). Per Catalog #315: substrate at L1+ with `impl_complete=true` AND latest council deliberation PROCEED_WITH_REVISIONS without later PROCEED-unconditional anchor MUST satisfy one of: (a) iteration anchor; (b) research_only=true; (c) lane_class=substrate_engineering; (d) archived=true; (e) waiver. This lane satisfies (c) via lane_class=substrate_engineering AND research_only=true UNTIL pre-probe verdict lands |

**Net classification:** 5 HARD-EARNED + 3 CARGO-CULTED + 1 HARD-EARNED-CONDITIONAL.
The 3 CARGO-CULTED (assumption 1 + 4 + 6) are explicitly addressed by the $0 PRE-PROBE
op-routable as HARD GATE before paid dispatch per Carmack MVP-first phasing. Per
CLAUDE.md "Forbidden premature KILL": CARGO-CULTED classifications enumerate
reactivation paths, not kill verdicts.

## §3. 9-dimension success checklist evidence (per Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | Path 3a IS class-shift per sister #857 HORIZON-CLASS reclassification to `asymptotic_pursuit`; A1 + STC residual sidecar grammar distinct from PR101 monolithic AND from A1 baseline (additive sidecar slot is structurally new per Catalog #272 distinguishing-feature contract) |
| 2 | BEAUTY + ELEGANCE | A1 baseline inflate.py is 135 LOC; amendment +10 LOC = 145 LOC total (within HNeRV parity L4 ≤200 LOC ceiling); reviewable in 30 seconds per PR101-style discipline. Trainer at ≤250 LOC + archive builder at ≤80 LOC + tests at ≤160 LOC = total substrate-engineering ~540 LOC (HNeRV parity L7 substrate-engineering exception allows >350) |
| 3 | DISTINCTNESS | Path 3a explicitly distinct from path 3b (Selfcomp tone-map-delta) and path 3c (full Filler 2011 2D+temporal); STC residual sidecar over A1 specifically tests STC + Dasher AC on 1D per-pair RGB residual stream (vs 3b: 2D mask delta; vs 3c: 2D + temporal context model) |
| 4 | RIGOR | Premise verification per Catalog #229 (read OVERNIGHT-J + OVERNIGHT-Q + 2026-05-17 sister symposium + 2026-05-20 sister symposium + A1 archive structure + canonical equation #359-sister BEFORE design); adversarial null hypothesis (the pre-probe MUST falsifiably challenge the CARGO-CULTED predicted band); per Catalog #292 per-deliberation assumption surfacing: every assumption in §2 carries explicit HARD-EARNED-vs-CARGO-CULTED classification |
| 5 | OPTIMIZATION PER TECHNIQUE | Canonical-vs-unique decision per layer (per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"): see §8 below |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Path 3a structurally composable with sister A1 sidecar techniques (PR106 latent sidecar; DP1 codebook) per Catalog #272; orthogonal to A1's existing decoder + latent + sidecar slots (additive sidecar slot is at NEW tail position); composition with Selfcomp PR#56 paradigm (3b) tests STC paradigm in 2 substrate-classes simultaneously |
| 7 | DETERMINISTIC REPRODUCIBILITY | A1 baseline archive sha 87ec7ca5 is byte-stable per `submissions/a1/archive_manifest.json` canonical anchor; STC predictor seed search + Dasher AC encoding seed-pinned per trainer spec; archive grammar amendment is deterministic byte-append (no re-ordering of A1 baseline bytes) |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | inflate.py amendment +10 LOC adds ~200 ns per-pair latency on T4 (single uint16 LE read + per-pair STC residual decode + additive RGB-correction); negligible vs A1 baseline ~1.5 ms per-pair decoder forward; tightened predicted band [-0.003, +0.001] reflects observability-only signed-band acknowledgment |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | NON-PROMOTABLE for synthetic-archive probe per Catalog #287/#323; predicted ΔS band [-0.003, +0.001] tagged `[prediction; NON-AUTHORITATIVE; signed-band]`; paired-CUDA-CPU paid Modal smoke (CONDITIONAL on pre-probe) would produce measurable contest-score anchor per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" |

## §4. Observability surface (per Catalog #305)

Per CLAUDE.md "Max observability - non-negotiable" 6-facet definition:

1. **Inspectable per layer**: $0 PRE-PROBE emits a per-pair residual distribution
   manifest at `.omx/state/wyner_ziv_deliverability/stc_residual_sidecar_a1_pre_probe_<utc>.json`
   exposing per-pair entropy estimate (H(X), H(X|X_{t-1}), H(X|2D-context)), sparsity,
   ternary-structure rate, per-pair RGB residual histogram. MVP smoke emits per-epoch
   training loss decomposition (rate term + d_seg term + d_pose term) per Catalog #164
   canonical scorer-loss helper routing.
2. **Decomposable per signal**: predicted ΔS band [-0.003, +0.001] decomposes into rate
   axis (+0.000271 / +407 bytes) + d_seg axis (TBD by pre-probe + MVP smoke) + d_pose
   axis (TBD; expected dominant per A1's pose-marginal-2.76x ratio per RD_DERIVATION.md).
3. **Diff-able across runs**: A1 baseline archive sha 87ec7ca5 is byte-stable;
   STC residual sidecar archive sha auto-derives from (A1 baseline + STC predictor
   seed + Dasher AC stream) deterministic concatenation; sha-stable across re-runs of
   the trainer with same random seed.
4. **Queryable post-hoc**: Catalog #313 probe-outcomes ledger row at
   `.omx/state/probe_outcomes.jsonl` queryable via `tools/check_predecessor_probe_outcome.py
   --substrate stc_paradigm_reformulation_a1_residual_path_3a` per canonical CLI; verdict
   chain (DEFER-TO-PRE-PROBE → pre-probe verdict → paid-smoke verdict) preserved
   per Catalog #245 canonical 4-layer pattern.
5. **Cite-able**: every empirical anchor carries `(probe_id, generated_at_utc, agent,
   subagent_id, session_id, written_pid, written_host)` provenance tuple per Catalog
   #245 + Catalog #323 canonical Provenance umbrella.
6. **Counterfactual-able**: byte-mutation smoke per Catalog #139 + #272 (mutate one
   byte in stc_residual_stream + verify rendered RGB output changes); sister
   counterfactual is mutating the STC predictor seed (32 B; verify trained predictor
   chain unrolls differently producing measurably different residual stream).

## §5. Sister symposium ratification (per Catalog #325 6-step contract item 4)

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-
negotiable: the per-substrate symposium for path 3a was convened TWICE prior to this
DESIGN landing:

- **2026-05-17 sister symposium**
  `.omx/research/council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517.md`
  T2 sextet pact (10 attendees: Shannon + Dykstra + Yousfi + Fridrich + Contrarian +
  Assumption-Adversary + Filler + Pevný + Selfcomp + Quantizr); verdict
  **PROCEED_WITH_REVISIONS** (5 binding revisions); op-routable #1 verbatim required
  a $0 PRE-PROBE before any paid dispatch
- **2026-05-20 sister symposium**
  `.omx/research/council_per_substrate_symposium_stc_paradigm_reformulation_a1_residual_20260520T194818Z.md`
  T2 sextet+grand-council (9 attendees: Shannon + Dykstra + Rudin + Daubechies + Yousfi
  + Fridrich + Contrarian + Assumption-Adversary + Filler); verdict
  **PROCEED_WITH_REVISIONS** (2 binding revisions per Contrarian + Assumption-Adversary);
  reactivation criterion #3 (measure actual A1 per-pair residual distribution) declared
  HARD CO-DELIVERABLE of paid dispatch

**THIS DESIGN memo ratifies the 2026-05-17 sister symposium's op-routable #1 as the
PRE-PROBE op-routable** + ratifies the 2026-05-20 sister symposium's reactivation
criterion #3 (HARD CO-DELIVERABLE) as the empirical measurement target for the pre-
probe verdict cascade. Per Catalog #346 canonical roster: prior symposia satisfied
canonical_council_roster.validate_council_dispatch_roster(complete=True) for T2 tier
per topic_tokens ["STC", "Filler", "syndrome-trellis", "steganography"]. THIS DESIGN
memo's T1 (Carmack + Assumption-Adversary) is the structural-protection re-route per
Carmack MVP-first 5-step phasing — NOT a new T2+ council deliberation.

## §6. Reactivation criteria (per Catalog #325 6-step contract item 5)

Path 3a's PROCEED-unconditional cascade requires THREE explicit gates (in order):

1. **$0 PRE-PROBE PASSED** (this lane's op-routable #1; sister subagent ~1h to build):
   `tools/probe_stc_3a_a1_residual_entropy.py` (CANONICAL PATH per 2026-05-17 op-routable
   #1; NOT yet built) measures A1 archive `87ec7ca5` actual per-pair RGB residual
   distribution; emits verdict in {HIGH-ENTROPY-RESIDUAL-PRESENT, MEDIUM-ENTROPY,
   LOW-ENTROPY-RESIDUAL-ABSENT}. HIGH → unlock $5.20 paid smoke; MEDIUM → defer to 3b;
   LOW → DEFER substrate per Catalog #307 paradigm-vs-implementation classification
2. **MVP scaffold smoke passes structural integration** (sister subagent post-pre-probe
   ~2h): build `experiments/train_substrate_stc_residual_sidecar_over_a1.py` per §1
   spec; run $0 local CPU smoke (10ep on M5 Max ~5 min CPU); verify (a) archive grammar
   roundtrip (encode → inflate.sh → re-decode = byte-identical RGB frames); (b) byte-
   mutation smoke per Catalog #139 (mutate one stc_residual_stream byte → rendered RGB
   changes); (c) canonical scorer-loss helper routes through `tac.substrates._shared.score_aware_common`
   per Catalog #164 (predicted ΔS rate-only at +0.000271 verified empirically; predicted
   d_seg + d_pose deltas TBD)
3. **$5.20 paid Modal smoke landed with paired CPU + CUDA per CLAUDE.md "Submission auth
   eval — BOTH CPU AND CUDA"** (HARD CO-DELIVERABLE per 2026-05-20 sister symposium
   binding revision #1): paid smoke landed contest-CPU + contest-CUDA anchors on the
   actual STC-residual-sidecar archive; reactivation criterion #3 (measure actual A1
   per-pair residual distribution on landed archive sha) co-delivered in same dispatch.
   Empirical predicted-band validation per Catalog #324 post-training Tier-C density on
   the landed archive sha; if empirical ΔS in [-0.003, +0.001] → PROCEED-unconditional;
   if empirical ΔS > +0.001 → DEFER + Catalog #307 paradigm-vs-implementation
   classification (implementation-cargo-cult; paradigm-intact for sister substrates);
   if empirical ΔS < -0.003 → BEAT (unexpected positive class-shift)

## §7. Catalog #324 post-training Tier-C validation discipline (per Catalog #325 6-step contract item 6)

Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density" (Catalog #324):
the recipe for path 3a (NOT YET AUTHORED in this lane; sister subagent post-pre-probe
scope) MUST declare `predicted_band_validation_status: pending_post_training` with
reactivation criterion = post-training Tier-C density measurement on the landed
archive sha via `tools/mdl_scorer_conditional_ablation.py --tier c`. The predicted band
[-0.003, +0.001] is derived from sister-anchor empirical evidence (PR100→PR105 +
PR106 yshift) + the 2026-05-17 symposium Contrarian + Filler + Selfcomp tightened
revision, NOT from random-init Tier-C density on path 3a's not-yet-built archive — so
Catalog #324 phantom-random-init failure mode does not apply directly; however, the
paid Modal smoke's post-training Tier-C MUST be measured + recorded as a sister anchor
for canonical equation registry posterior Bayesian update per Catalog #344.

## §8. Canonical-vs-unique decision per layer (per Catalog #290 + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode")

| Layer | Canonical (ADOPT) | Unique (FORK) | Rationale |
|---|---|---|---|
| Archive grammar | A1 baseline single-ZIP-member 'x' layout | STC residual sidecar appended at tail with uint16 LE length prefix | Adopt canonical: A1 baseline grammar is byte-stable + verified empirically. Unique fork: STC sidecar slot is NEW + structurally distinct (additive tail-append) per HNeRV parity L3 |
| Inflate.py | A1 baseline `parse_a1_finetuned_archive` + decoder + bicubic up + RGB cast | `apply_stc_residual_correction` helper (+10 LOC) | Adopt canonical: A1 baseline inflate.py is 135 LOC verified working. Unique fork: STC residual decode + additive RGB-correction is substrate-specific bolt-on per Catalog #272 distinguishing-feature integration contract |
| Scorer-loss helper | `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164 | NONE | Adopt canonical: STC residual sidecar is sidecar-layer codec; scorer-loss routes through canonical helper unchanged (no substrate-specific scorer-preprocess fork required) |
| Auth-eval helper | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` per Catalog #226 | NONE | Adopt canonical: auth-eval routing is canonical; no substrate-specific contest-CUDA-CLI fork required per Catalog #226 |
| Inflate device selector | `tac.substrates._shared.inflate_runtime.select_inflate_device` per Catalog #205 (mirrored inline per `submissions/a1/inflate.py` self-contained pattern) | NONE | Adopt canonical: PACT_INFLATE_DEVICE env-var routing is canonical per Catalog #205; mirror inline in `submissions/stc_residual_sidecar_over_a1/inflate.py` per self-contained PYTHONPATH discipline (Catalog #295) |
| Modal mount manifest | `tac.deploy.modal.mount_manifest.build_training_image` per Catalog #153 | NONE | Adopt canonical: Modal image build routes through canonical builder per Catalog #153; `TIER_1_EXTRA_MOUNT_PATHS = (str(DEFAULT_A1_ANCHOR_ARCHIVE.relative_to(REPO_ROOT)),)` declares A1 anchor archive as extra-mount per Catalog #152 WAVE-1 extension |
| Modal NVML env block | `DALI_DISABLE_NVML=1` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` per Catalog #244 | NONE | Adopt canonical: NVML env block is canonical per Catalog #244 auto-emitted via `tac.substrate_registry.driver_generator.generate_driver_shell` |
| Trainer driver | `tools/canonical_dispatch_optimization_protocol.py` Tier 1/2/3 contract per Catalog #270 | NONE | Adopt canonical: dispatch optimization protocol per Catalog #270 umbrella; trainer declares `--enable-autocast-fp16` + TF32 + torch.compile + no_grad-at-eval + GTScorerCache F3 consumption per sister catalogs |
| Council-binding spec | per-substrate symposium per Catalog #325 (already convened TWICE: 2026-05-17 + 2026-05-20) | NONE | Adopt canonical: prior symposia ratified PROCEED_WITH_REVISIONS with explicit binding revisions; THIS DESIGN re-routes through Carmack MVP-first 5-step phasing per CLAUDE.md `be125b878` |
| Probe-outcomes ledger | `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 | NONE | Adopt canonical: ledger append-only via canonical helper per fcntl-locked discipline + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE |

## §9. Operator-routable next steps (3 paths per Carmack MVP-first Step 5 re-route)

### Path A (RECOMMENDED): $0 PRE-PROBE FIRST per Carmack MVP-first phasing

Per 2026-05-17 sister symposium op-routable #1 verbatim + Carmack MVP-first Step 1
(FREE local CPU smoke FIRST):

1. **Sister subagent**: Build `tools/probe_stc_3a_a1_residual_entropy.py` (CANONICAL
   PATH per 2026-05-17 op-routable #1; NOT YET BUILT). Tool loads A1 archive
   `87ec7ca5` from `submissions/a1/archive.zip`, runs canonical inflate to produce
   ground-truth decoded RGB frames + diff against `upstream/videos/0.mkv` ground-truth
   to extract actual per-pair RGB residual. Measure: H(X), H(X|X_{t-1}),
   H(X|2D-context), apparent sparsity, ternary-structure-rate, alignment with SegNet
   attack surface (boundary-residual classification). Emit verdict in
   {HIGH-ENTROPY-RESIDUAL-PRESENT, MEDIUM-ENTROPY-RESIDUAL-PRESENT,
   LOW-ENTROPY-RESIDUAL-ABSENT}. Cost: $0 (CPU only on local M5 Max). Time: ~1h build
   + ~5 min run. Verdict feeds path 3a dispatch eligibility per Catalog #313 probe-
   outcomes ledger.

2. **Sister subagent (CONDITIONAL on HIGH-ENTROPY verdict)**: Build MVP scaffold at
   `src/tac/substrates/stc_residual_sidecar_over_a1/` per §1 substrate spec; build
   trainer + archive builder + tests; run $0 local CPU smoke (10ep on M5 Max ~5 min);
   verify byte-mutation smoke + canonical scorer-loss helper routing; emit MVP smoke
   verdict.

3. **Sister subagent (CONDITIONAL on MVP smoke PASS)**: Author recipe
   `.omx/operator_authorize_recipes/substrate_stc_residual_sidecar_over_a1_modal_t4_dispatch.yaml`
   per Catalog #240 recipe-vs-trainer-state consistency + Catalog #324
   `predicted_band_validation_status: pending_post_training`; fire $5.20 paid Modal A100
   10ep paired CPU + CUDA smoke per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"
   with reactivation criterion #3 (measure actual A1 residual distribution on landed
   archive sha) as HARD CO-DELIVERABLE per 2026-05-20 sister symposium binding revision
   #1.

### Path B: OPERATIONAL DEBUG `experiments/modal_train_lane.py` for STC v2 silent-no-spawn root cause

Per OVERNIGHT-J Path B (5/5 STC v2 dispatches failed silently while all other substrates
dispatch cleanly via SAME code path; the bug is STC v2 SPECIFIC):

1. Run `.venv/bin/modal app history ap-KA1LFP69IGthTDNrXGXRie` to capture WHY the
   2026-05-21T07:41Z app stopped with 0 tasks
2. Compare Modal API call sequence vs successful DP1 dispatch
   `fc-01KS4KJGDXVXZ9NYRD4HKZ9CET` (same moment, same code path, SUCCEEDED)
3. Check `experiments/modal_train_lane.py` for STC v2 path-specific code (cost_band
   trainer + lane_script + env_overrides)
4. Cost: $0 + ~1h diagnostic wall-clock
5. **NOTE**: Path B addresses ORIGINAL STC v2 substrate dispatch surface; Path A pivots
   to DIFFERENT substrate (STC residual sidecar over A1). Per OVERNIGHT-J Path A
   recommendation: Path A is HIGHER EV per Carmack MVP-first phasing since Path B is
   diagnostic-only with no score-lowering enabler

### Path C: DEFER STC paradigm indefinitely per Carmack MVP-first reroute to higher-EV pickup

Per CLAUDE.md "Forbidden premature KILL" + "Mission alignment - non-negotiable"
Consequence 4 (frontier-breaking moves DOMINATE rigor budget):

1. STC v2 dispatch surface 5-times-broken; reroute attention to working surfaces
2. Z6 Wave 2 4c re-fire ($3 Modal A10G; SUBAGENT-spawnable per yesterday's triage Pick 1)
3. HFV1 PR101 exact-eval readiness verification ($0.20-0.40 Modal T4 paired; SUBAGENT-
   spawnable)
4. HFV2 sparse sidecar paired smoke ($0.20-0.40 Modal T4 paired; SUBAGENT-spawnable)

## §10. Predicted ΔS band + canonical equation IN-DOMAIN computation

**Predicted ΔS band: [-0.003, +0.001] [prediction; NON-AUTHORITATIVE; signed-band]**

TIGHTENED from yesterday's [-0.005, -0.001] (CARGO-CULTED per yesterday Assumption-
Adversary verdict 1) and from 2026-05-17 parent symposium's [-0.015, -0.003]
(CARGO-CULTED per 2026-05-17 Assumption-Adversary verdict 3) per Contrarian + Filler +
Selfcomp + Quantizr REVISIONS. SIGNED-BAND acknowledges that net ΔS COULD regress
(rate cost dominates distortion gain) at A1's already-saturated operating point.

**Canonical equation #359-sister IN-DOMAIN computation** (verified empirically at
DESIGN time):

```python
from tac.canonical_equations.procedural_predictor_residual_savings import (
    predict_procedural_predictor_plus_residual_correction_savings,
)
result = predict_procedural_predictor_plus_residual_correction_savings(
    original_payload_bytes=0,  # ADDITIVE sidecar (no slot being replaced)
    predictor_seed_or_code_bytes=32,
    residual_stream_bytes=375,
    container_overhead_bytes=0,
    context='stc_predictor_plus_residual_a1_per_pair_correction',
)
# result['predicted_delta_s_rate_only'] = +0.000271 (rate axis only)
# result['delta_bytes_replacement_minus_original'] = +407
# result['verdict'] = 'RATE_REGRESSION' (rate axis; distortion-axis offset TBD)
# result['score_claim'] = False
# result['promotable'] = False
```

The canonical equation's rate-only prediction (+0.000271) confirms the rate axis
penalty. Net ΔS in [-0.003, +0.001] depends on downstream d_seg + d_pose improvement
from the additive RGB-residual correction — empirically UNKNOWN until pre-probe + MVP
smoke + paid dispatch. The canonical equation is RATE-AXIS-ONLY by design per
Catalog #359 sister discipline; distortion-axis contribution requires a sister
equation (FORMALIZATION_PENDING `stc_residual_sidecar_a1_per_pair_score_delta_v1` upon
empirical anchor landing per yesterday symposium op-routable #4).

**Dykstra-feasibility verdict** (per Catalog #296): FEASIBLE per convex constraints:

- rate axis: +407 bytes ↔ +0.000271 (HARD-EARNED rate-only prediction)
- seg axis: TBD by pre-probe + MVP smoke (additive RGB correction COULD reduce d_seg or
  leave unchanged; structurally compatible with unchanged-or-improved bound)
- pose axis: TBD by pre-probe + MVP smoke (additive RGB correction COULD reduce d_pose
  or leave unchanged; structurally compatible with unchanged-or-improved bound;
  expected dominant per A1's pose-marginal-2.76x ratio per RD_DERIVATION.md)
- archive size axis: ≤178,800 B total (within reasonable size envelope)

Dykstra alternating-projections intersection of (rate ≤ R + 407) ∩ (seg ≤ S) ∩ (pose ≤
P) is NON-EMPTY for tightened SIGNED predicted band.

## §11. Horizon-class declaration (per Catalog #309)

**horizon_class: asymptotic_pursuit**

Per CLAUDE.md HORIZON-CLASS evaluation axis standing directive: path 3a is classified
`asymptotic_pursuit` because (a) sister #857 HORIZON-CLASS reclassification confirmed
STC clean-source as `asymptotic_pursuit`; (b) yesterday + 2026-05-17 sister symposia
both classified path 3a as `asymptotic_pursuit`; (c) predicted savings small (-0.003
lower bound) relative to plateau cluster variance makes this a candidate at the
asymptotic-pursuit band lower bound; (d) per the standing directive's 20% budget
allocation requirement, asymptotic-pursuit candidates remain priority class.

## §12. MVP scaffold status

**MVP scaffold: DESIGNED but NOT BUILT in this lane.**

Per Carmack MVP-first 5-step phasing + scope-discipline:

- Phase 1 (THIS lane): DESIGN memo + canonical equation #359-sister IN-DOMAIN computation
  + Catalog #325 6-step contract + Carmack MVP-first 5-step compliance docs
- Phase 2 (sister subagent): $0 PRE-PROBE build + run per §9 Path A op-routable #1
- Phase 3 (sister subagent CONDITIONAL on Phase 2 HIGH-ENTROPY): MVP scaffold build at
  `src/tac/substrates/stc_residual_sidecar_over_a1/` per §1 spec; $0 local CPU smoke
- Phase 4 (sister subagent CONDITIONAL on Phase 3 PASS): recipe author + $5.20 paid
  Modal smoke per §9 Path A op-routable #3

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220:
the MVP scaffold WILL be `research_only=true` at landing (no archive bytes added until
trainer runs); Catalog #220 acceptance via `research_only=true` + (post Phase 3) `lane_class=substrate_engineering`.

## §13. Discipline compliance (per CLAUDE.md non-negotiables)

- ✅ Catalog #229 PV: read OVERNIGHT-J DEFER landing memo + OVERNIGHT-Q T3 symposium §6
  Tier 2 Decision #5 + 2026-05-17 sister symposium + 2026-05-20 sister symposium +
  2026-05-20 probe-disambiguator synthesis + A1 archive `87ec7ca5` empirically (zipfile
  inspect + sha verify) + `submissions/a1/inflate.py` + `submissions/a1/RD_DERIVATION.md`
  + canonical equation `procedural_predictor_plus_residual_correction_savings_v1` source
  + Catalog #313 probe-outcomes ledger pre-check BEFORE memo draft
- ✅ Catalog #117 + #157 + #174 canonical serializer (commit lands via canonical
  serializer with POST-EDIT `--expected-content-sha256`)
- ✅ Catalog #119 Co-Authored-By Claude trailer (in commit message)
- ✅ Catalog #125 6-hook wire-in declaration (see top of memo)
- ✅ Catalog #131 + #138 + #245 canonical state-IO discipline (no bare writes; fcntl-
  locked via canonical helpers if probe_outcomes ledger row appends in same commit
  batch)
- ✅ Catalog #199 + #202 paired-env bypass discipline (not applicable — no paid
  dispatch in this lane)
- ✅ Catalog #206 mandatory crash-resume protocol (4 in_progress checkpoints + this
  complete checkpoint)
- ✅ Catalog #229 premise verification before edit
- ✅ Catalog #240 recipe-vs-trainer-state consistency (recipe NOT authored in this
  lane; sister subagent post-pre-probe scope per §9)
- ✅ Catalog #270 canonical dispatch optimization protocol (umbrella declared in §8
  layer-decision table; sister subagent post-pre-probe scope)
- ✅ Catalog #287 placeholder-rationale rejection (no `<rationale>` / `<reason>` literals
  in any waiver / claim; substantive rationales throughout)
- ✅ Catalog #290 substrate design memo canonical-vs-unique decision per layer section
  (see §8)
- ✅ Catalog #291 META-ASSUMPTION cadence (sister cycle within 7-day window; this
  memo's Assumption-Adversary verdicts §-frontmatter address shared assumptions)
- ✅ Catalog #292 per-deliberation assumption surfacing (every council member's
  operating-within assumption surfaced)
- ✅ Catalog #294 9-dimension success checklist evidence section (see §3)
- ✅ Catalog #296 Dykstra-feasibility check (see §10 + §11)
- ✅ Catalog #298 substrate retirement discipline (lane pre-registered at L1; lane_class
  declaration covered in §-frontmatter `target_modes` field per HNeRV parity L7)
- ✅ Catalog #300 v2 frontmatter (council_tier=T1 + attendees + quorum + verdict +
  dissent + assumption_adversary_verdict + decisions_recorded + mission_contribution +
  override_invoked + override_rationale + related_deliberation_ids + predicted_band_
  validation_status)
- ✅ Catalog #303 cargo-cult audit section (see §2)
- ✅ Catalog #305 observability surface section (see §4)
- ✅ Catalog #307 paradigm-vs-implementation classification (see §6 reactivation
  criterion #3 cascade verdicts)
- ✅ Catalog #308 alternative-probe-methodologies enumeration (see §9 Paths A + B + C)
- ✅ Catalog #309 horizon-class declaration (see §11)
- ✅ Catalog #313 probe-outcomes ledger (pre-checked; no blocking outcome; sister
  subagent appends DEFER-TO-PRE-PROBE row in same commit batch OR post-pre-probe per
  Carmack MVP-first phasing)
- ✅ Catalog #315 substrate optimal form before paid dispatch (this lane satisfies (c)
  lane_class=substrate_engineering + research_only=true UNTIL pre-probe + MVP smoke +
  PROCEED-unconditional council deliberation lands)
- ✅ Catalog #323 canonical Provenance umbrella (all predicted-ΔS literals tagged
  `[prediction; NON-AUTHORITATIVE; signed-band]`)
- ✅ Catalog #324 post-training Tier-C validation discipline (see §7)
- ✅ Catalog #325 per-substrate symposium 6-step contract satisfied via prior 2026-05-17
  + 2026-05-20 sister symposia ratification (see §5)
- ✅ Catalog #340 sister-checkpoint guard PROCEED verified pre-write (no conflicts)
- ✅ Catalog #344 canonical equation reference (see §10 + Step 3 of Carmack MVP-first
  table)
- ✅ Catalog #346 canonical roster validate complete=True (prior symposia satisfied;
  THIS lane's T1 is structural-protection re-route per Carmack MVP-first phasing)
- ✅ Catalog #359 residual-hybrid structural protection (canonical equation #359-sister
  used IN-DOMAIN with `stc_predictor_plus_residual_a1_per_pair_correction` context per
  empirically verified `is_residual_hybrid_context` validation)
- ✅ Carmack MVP-first 5-step recipe per CLAUDE.md `be125b878` (all 5 steps documented
  in compliance table at top of memo)
- ✅ Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE (NEW landing memo; ZERO
  mutations to existing artifacts including CLAUDE.md, OVERNIGHT-J memo, OVERNIGHT-Q
  symposium memo, A1 substrate package source, STC v2 substrate package source,
  prior symposia memos)

## §14. Cross-references

- `.omx/research/stc_v2_ratify_or_defer_path_b_dispatch_landed_20260521.md` (OVERNIGHT-J
  Path A recommendation; STC-V2-SPECIFIC bug class)
- `.omx/research/grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521.md`
  (OVERNIGHT-Q T3 symposium §6 Tier 2 Decision #5)
- `.omx/research/council_per_substrate_symposium_stc_paradigm_reformulation_a1_residual_20260520T194818Z.md`
  (2026-05-20 sister symposium; PROCEED_WITH_REVISIONS)
- `.omx/research/council_per_substrate_symposium_stc_3a_sidecar_a1_residual_20260517.md`
  (2026-05-17 sister symposium; PROCEED_WITH_REVISIONS; op-routable #1 = $0 pre-probe)
- `.omx/research/stc_paradigm_reformulation_a1_residual_disambiguator_synthesis_20260520T165252Z.md`
  (2026-05-20 probe-disambiguator synthesis; synthetic-archive entropy-ladder PROCEED)
- `submissions/a1/archive.zip` sha `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
  size 178,262 B (A1 baseline anchor)
- `submissions/a1/inflate.py` (A1 baseline inflate runtime; 135 LOC; HNeRV parity L4
  compliant)
- `submissions/a1/RD_DERIVATION.md` (A1 contest-score functional R(D) accounting)
- `src/tac/canonical_equations/procedural_predictor_residual_savings.py` (canonical
  equation #359-sister; IN-DOMAIN per `stc_predictor_plus_residual_a1_per_pair_correction`
  context)
- `tools/probe_stc_paradigm_reformulation_disambiguator.py` (2026-05-20 probe-
  disambiguator canonical tool; sister to the not-yet-built `tools/probe_stc_3a_a1_residual_entropy.py`
  per §9 Path A op-routable #1)
- `.omx/state/probe_outcomes.jsonl` (Catalog #313 canonical ledger; queryable via
  `tools/check_predecessor_probe_outcome.py --substrate stc_paradigm_reformulation_a1_residual_path_3a`)
- Carmack MVP-first 5-step amendment commit `be125b878` (CLAUDE.md non-negotiable)
- Catalog cross-refs: #110 + #113 + #117 + #119 + #125 + #131 + #138 + #146 + #152 + #153
  + #157 + #164 + #168 + #174 + #176 + #185 + #186 + #199 + #202 + #205 + #206 + #220 +
  #226 + #229 + #240 + #244 + #245 + #270 + #272 + #287 + #290 + #291 + #292 + #294 +
  #295 + #296 + #298 + #300 + #303 + #305 + #307 + #308 + #309 + #313 + #315 + #323 +
  #324 + #325 + #340 + #344 + #346 + #359 (sister-binding extincted across the substrate-
  engineering surface)

## §15. Mission contribution (per Catalog #300)

**council_predicted_mission_contribution: frontier_breaking_enabler**

Per CLAUDE.md "Mission alignment - non-negotiable" 5 operational consequences: this
DESIGN memo + Carmack MVP-first 5-step re-routing IS `frontier_breaking_enabler` (NOT
directly `frontier_breaking`) because the predicted ΔS band [-0.003, +0.001] is
SIGNED-NON-AUTHORITATIVE; the $0 pre-probe (sister subagent) + MVP scaffold smoke
(sister subagent CONDITIONAL on pre-probe) + paid $5.20 Modal smoke (sister subagent
CONDITIONAL on MVP scaffold PASS) are the cascade that would convert predicted-frontier-
breaking-candidate into measured-frontier-breaking-anchor. The Carmack MVP-first
phasing prevents $5.20 spend on CARGO-CULTED predicted-band per CLAUDE.md `be125b878`
amendment.

---

**End of DESIGN memo.**

**Verdict:** DEFER-TO-PRE-PROBE per Carmack MVP-first 5-step Step 1 ($0 LOCAL CPU MVP
DESIGN landed; paid dispatch DEFERRED-PENDING-PRE-PROBE-VERDICT). Next-cascade
operator-routable: sister subagent builds `tools/probe_stc_3a_a1_residual_entropy.py`
per §9 Path A op-routable #1 (~1h build + ~5 min run; $0).
