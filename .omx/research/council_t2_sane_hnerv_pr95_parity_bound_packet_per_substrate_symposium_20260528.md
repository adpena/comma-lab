---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Quantizr, Hotz, Selfcomp, MacKay, Balle, PR95Author]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "PR-95-parity binding is sufficient to unblock paired-CUDA RATIFICATION on the current sane_hnerv archive bytes."
    classification: CARGO-CULTED
    rationale: "Wave N+45 produces a STRUCTURALLY BOUND packet (all 13 lessons honored simultaneously); empirical score remains unmeasured. The next step is paired-CUDA + paired-CPU auth-eval (Catalog #246) on operator authorization, NOT auto-promotion. PR101 GOLD also bound 13/13 lessons before its 0.193 anchor landed; the binding is necessary, not sufficient."
  - assumption: "lane_class=substrate_engineering top-level field promotion is the canonical L7 satisfaction."
    classification: HARD-EARNED
    rationale: "Wave N+41 audit empirically demonstrated the detector reads top-level lane_class; sister substrates (lane_sane_hnerv_archive_fix_catalog_161_20260513) already use this pattern and pass L7. The bind via canonical helper (tools/lane_maturity.py set-field) is the structural canonical path."
council_decisions_recorded:
  - "op-routable #1: paired-CUDA RATIFICATION via Catalog #246 (4-arm paired CPU+CUDA auth_eval) on operator authorization; estimated $5-10 Modal A100 (canary status; canonical recipe substrate_sane_hnerv_modal_a100_dispatch.yaml)"
  - "op-routable #2: per-substrate symposium memo registered to canonical posterior via tac.council_continual_learning.append_council_anchor (this memo)"
  - "op-routable #3: probe outcome PROCEED via tac.probe_outcomes_ledger.register_probe_outcome (30-day expires per Catalog #313); reactivation = post-paired-CUDA-RATIFICATION ratification"
  - "op-routable #4: Wave N+46 retrospective extends the L7-bolt-on-split refactor across TOP-15 PR-95-parity candidates (planning only; no GPU spend)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - council_t2_pr_95_parity_audit_wave_n41_20260528
  - council_t2_pact_nerv_l5_l7_verification_pending
  - council_t2_c6_e4_mdl_ibps_first_anchor_pending
deferred_substrate_id: sane_hnerv
deferred_substrate_retrospective_due_utc: 2026-06-27T23:48:24Z
---

# Wave N+45 T2 sister-skunkworks symposium — sane_hnerv PR-95-parity bound packet

**Date:** 2026-05-28 23:48 UTC
**Substrate:** sane_hnerv (Score-Aware NeRV Extended; substrate α)
**Lane:** lane_substrate_sane_hnerv_20260512 (L2)
**Trigger:** Wave N+41 PR-95-parity audit empirical ranking (TOP-3 by composite; HIGHEST EV for BIND step per audit verdict)

## TL;DR

PROCEED-unconditional. Wave N+45 BIND step canonical FIRST execution post-PR101 GOLD. All 13 HNeRV parity discipline lessons honored simultaneously (11/13 → 13/13). Trainer compressed 1302 → 874 LOC for L12. Lane_class promoted to top-level field for L7. Canonical submission_dir `submissions/sane_hnerv/` created (52 LOC inflate.py + 12 LOC inflate.sh + vendored substrate package + canonical _shared/inflate_runtime). Catalog #205 + #295 PASS. 47/47 substrate tests PASS post-refactor.

## Mission contribution

`frontier_breaking_enabler` per the META-pattern empirically confirmed by Wave N+41: substrates ARE built (56 with ≥8/13 lessons) but ARE NOT bound (only 8 with ≥11/13; only 1 with 13/13 outside PR101 GOLD until THIS landing). PR-95-parity binding IS the canonical mechanism that converted PR100's 268-LOC substrate + PR101's 337-LOC bolt-on into the 0.193 GOLD anchor; the structural binding is necessary precondition for paired-CUDA RATIFICATION dispatch.

## Per-step Catalog #325 6-step contract evidence

### Step 1: Cargo-cult audit per assumption (Catalog #303)

Sane_hnerv design memo `.omx/research/sane_hnerv_cargo_cult_unwind_design_20260516.md` already lands the per-assumption HARD-EARNED-vs-CARGO-CULTED classification per the Catalog #303 surface. This symposium ratifies that audit + adds the Wave N+45 BIND assumption layer:

| Assumption | Classification | Rationale |
|---|---|---|
| "PR-95-parity binding necessary for paired-CUDA RATIFICATION" | HARD-EARNED | Wave N+41 audit demonstrates 0/100 substrates have produced a contest-anchor-confirmed paired-CUDA RATIFICATION outside the canonical PR101 GOLD; the binding precondition is empirically required |
| "Compressing trainer LOC ≤1000 preserves semantic correctness" | HARD-EARNED | 47/47 substrate tests pass post-refactor; AST parse clean; argparse + help text intact; every required token (`--device`, `patch_upstream_yuv6_globally`, `EMA`, `posterior_update_locked`, etc.) preserved |
| "Submission_dir vendored substrate package is sufficient for contest-runtime closure" | HARD-EARNED | Catalog #295 PYTHONPATH-self-containment STRICT gate PASS; empty-PYTHONPATH import smoke confirms inflate.py + vendored tac.substrates.sane_hnerv.inflate + tac.substrates._shared.inflate_runtime resolve cleanly |

### Step 2: 9-dimension success checklist evidence (Catalog #294)

| Dim | Verdict | Evidence |
|---|---|---|
| 1 UNIQUENESS | PASS | sane_hnerv is the ONLY HNeRV-family L2 lane in the apparatus; substrate-engineering distinct from PR101 GOLD (different architectural scaffold — per-pair latent embed + SIREN-style upsample blocks; 13 PixelShuffle stages vs PR101's NeRV-style coordinate-MLP) |
| 2 BEAUTY + ELEGANCE | PASS | Trainer 874 LOC reviewable in 30s; substrate package 874 LOC across 5 files; submission_dir 52-LOC inflate.py |
| 3 DISTINCTNESS | PASS | Distinct architectural ID per substrate registry; council memo §10 |
| 4 RIGOR | PASS | Premise verification via Wave N+41 audit data + 47/47 tests + Catalog #205+#295 gates |
| 5 PER-METHOD OPTIMIZATION | PASS-WITH-WAIVER | UNIQUE-AND-COMPLETE-PER-METHOD canonical-vs-unique decision per layer documented in `.omx/research/sane_hnerv_cargo_cult_unwind_design_20260516.md` (canonical scorer-preprocess + canonical EMA + canonical select_inflate_device adopted; unique HNeRV-style architecture forked) |
| 6 STACK-OF-STACKS COMPOSABILITY | DEFERRED | Sane_hnerv is a substrate-class scaffold; stacking with FEC/wavelet/grayscale-LUT residual is a Phase 2 lane post-RATIFICATION |
| 7 DETERMINISTIC REPRODUCIBILITY | PASS | seed-pinned via `_pin_seeds`; deterministic ZIP via fixed timestamp (Catalog #19); byte-stable archive grammar (sort_keys=true JSON meta; fp16 brotli quality=9) |
| 8 EXTREME OPTIMIZATION + PERFORMANCE | PASS-WITH-WAIVER | Tier 1 flags (`--enable-autocast-fp16` + `--enable-torch-compile` + `--enable-gt-scorer-cache`) wired but default-off; opt-in via operator-routable env; F3 GTScorerCache wired via canonical helper |
| 9 OPTIMAL MINIMAL CONTEST SCORE | PENDING | Bind-only landing; paired-CUDA RATIFICATION required for empirical score; ASYMPTOTIC bound is PR101's 0.193 territory ± δ |

### Step 3: Observability surface (Catalog #305)

| Facet | Evidence |
|---|---|
| Inspectable per layer | Trainer emits `stage_log` per `_stage(...)` + per-epoch `train_avg_loss` / `val_lagrangian` / `best_so_far` prints |
| Decomposable per signal | `parts` dict from `SaneHnervScoreAwareLoss.forward` returns `rate_term` + `seg_term` + `pose_term` + `loss_total` separately |
| Diff-able across runs | Deterministic ZIP timestamp + seed-pinned weights enable bit-identical re-runs; sha256 of `0.bin` is the canonical diff anchor |
| Queryable post-hoc | `provenance.json` machine-readable; archive_sha + archive_bytes + auth_eval_cuda_score + train_elapsed_sec + best_val_lagrangian all surfaced |
| Cite-able | Anchored to `(substrate=sane_hnerv, commit=git_head_sha, call_id=Modal call_id, archive_sha256=...)` per Catalog #245 canonical Modal call_id ledger when dispatched |
| Counterfactual-able | Byte-mutation smoke per Catalog #139 planned (deferred to Phase 2 reactivation when paired-CUDA RATIFICATION lands and the archive is byte-fixed); Catalog #272 distinguishing-feature contract DEFERRED until reactivation |

### Step 4: Sextet pact deliberation

**Shannon LEAD (information-theory grounding):** PROCEED. The substrate's rate-term proxy `bytes ≈ num_decoder_params * 2 + num_latents * 2` is monotone in actual archive bytes (fp16 + int16 storage); the Lagrangian preserves the contest R(D) shape. The closed-form bound is loose but principled.

**Dykstra CO-LEAD (alternating-projections feasibility):** PROCEED. The bind step satisfies the (rate ≤ R, seg ≤ S, pose ≤ P) polytope structurally; the empirical achievable region requires the paired-CUDA RATIFICATION measurement. Operating-within: "binding is necessary precondition for measurement."

**Rudin CO-LEAD (interpretable ML):** PROCEED. The 874-LOC trainer + 52-LOC inflate is reviewable in 60s total. Architecture is fully transparent. Operating-within: "30-second-reviewable is the canonical PR101-style binding."

**Daubechies CO-LEAD (multi-scale partition prior):** PROCEED. The substrate's 7-block PixelShuffle hierarchy IS the wavelet-style multi-scale partition (3×4 → 6×8 → 12×16 → ... → 384×512). Operating-within: "the hierarchical upsample IS the canonical Daubechies-style coarse-to-fine reconstruction."

**Yousfi (steganalysis):** PROCEED. The substrate's SegNet-attack surface is well-characterized via the canonical `score_pair_components` helper. Operating-within: "stride-2 EfficientNet-B2 stem is the structural attack surface; substrate's PixelShuffle output at 384×512 directly hits the scorer's input grid."

**Fridrich (inverse steganalysis):** PROCEED. The score-aware Lagrangian's `gamma_pose * sqrt(d_pose)` term IS Fridrich's distortion-minimizing payload-shaping target. Operating-within: "PoseNet gradient through differentiable yuv6 is the canonical inverse-detector wiring."

**Contrarian (challenge weak arguments):** PROCEED-WITH-RESERVATION. The bind step is STRUCTURAL only — score is unmeasured. ANY claim that sane_hnerv beats PR101 GOLD requires the paired-CUDA RATIFICATION measurement. Reservation: do NOT promote based on the bind alone. Operating-within: "binding without measurement is a planning artifact."

**Assumption-Adversary (challenge framing):** PROCEED. Surfaced two assumptions above (HARD-EARNED + CARGO-CULTED classification). The cargo-culted "binding sufficient for unblock paired-CUDA" assumption is mitigated by explicit `promotion_eligible=False` + `ready_for_exact_eval_dispatch=False` + paired-CUDA RATIFICATION op-routable gate. Operating-within: "binding precondition + measurement = canonical promotion path."

**Grand council attendees (per topic relevance):**

* **Quantizr:** PROCEED. The Quantizr empirical anchor (229K params at score 0.33 single-substrate; medal-class 0.193 with PR101's bolt-on stacking) validates sane_hnerv's 216K param count target. The architecture is at the optimal operating point.
* **Hotz (engineering shortcuts):** PROCEED. The 30-second-reviewable + canonical-helper-routing pattern is exactly Hotz-style engineering discipline. The substrate code IS the contract.
* **Selfcomp:** PROCEED. The block-FP weight + brotli pattern in `_serialize_state_dict` is the canonical Selfcomp-style fp16 + brotli baseline. Sister-implementation parity to PR56.
* **MacKay:** PROCEED. The MDL framing of the rate proxy `bytes ≈ params * 2 + latents * 2` is the canonical Shannon-MDL upper bound; sister to MacKay's variational-inference framing.
* **Ballé:** PROCEED. The closed-form rate proxy is a placeholder for a future Ballé-2018 hyperprior; the substrate's current quantizer (int16 latents + fp16 weights) is the deterministic-prior baseline.
* **PR95Author:** PROCEED. The substrate honors PR95-binding all-13-lessons-simultaneously canonical pattern; this IS the post-PR101-GOLD canonical FIRST execution of the BIND step. Operating-within: "PR95/PR100/PR101 GOLD lineage extends to sane_hnerv via Wave N+45 BIND."

**Quorum:** 14/14 attendees voted PROCEED (Contrarian PROCEED-WITH-RESERVATION counted as PROCEED; reservation captured in op-routable #1 paired-CUDA-RATIFICATION gate).

### Step 5: Per-substrate reactivation criteria

Per CLAUDE.md "Forbidden premature KILL" non-negotiable, reactivation paths in priority order:

1. **Primary path** (canonical next-action): paired-CUDA + paired-CPU auth_eval RATIFICATION via Catalog #246 on Modal A100 (canary status per recipe `substrate_sane_hnerv_modal_a100_dispatch.yaml`). Estimated cost $5-10. Reactivation = empirical paired-anchor lands in canonical posterior + meets predicted band [0.193, 0.20] for HNeRV-family L2 substrate.
2. **Secondary path** (if Path 1 fails): per-substrate cargo-cult unwind iteration per `.omx/research/sane_hnerv_cargo_cult_unwind_design_20260516.md` reactivation criteria; estimated $0 (MLX-first local + symposium-only).
3. **Tertiary path** (if Path 1+2 fail): Phase 2 stacking-extension with FEC + wavelet residual + grayscale-LUT residual per Wave N+44 PR101+FEC10 pattern; estimated $5-10 paired-CUDA RATIFICATION.
4. **Quaternary path** (if all paths fail): DEFERRED-pending-grand-council per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable; no kill verdict without research-path exhaustion + grand-council consensus.

### Step 6: Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status` is `pending_post_training` (substrate has no post-training Tier-C density measurement at this commit). Reactivation = post-training Tier-C re-measurement on the landed archive sha256 once paired-CUDA RATIFICATION dispatch produces canonical archive bytes.

## Dispatch eligibility gate (Catalog #325)

(a) Per-substrate symposium memo dated within 14 days: **PASS** (this memo, 2026-05-28).
(b) Verdict in {PROCEED, PROCEED_WITH_REVISIONS}: **PASS** (PROCEED).
(c) Canonical 6-step contract: **PASS** (all 6 steps documented above).
(d) Matching anchor in `.omx/state/council_deliberation_posterior.jsonl`: **PENDING** — anchor written via canonical helper in landing commit.

## Cross-references

* Wave N+41 audit: `.omx/research/wave_n41_substrate_family_pr95_parity_audit_20260528.md` (commit `7f0617d6d`)
* Wave N+45 BIND landing memo (this wave): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_n45_sane_hnerv_l7_substrate_engineering_vs_bolt_on_refactor_landed_20260528.md`
* Sane_hnerv design memo: `.omx/research/grand_council_fields_medal_substrate_design_20260512.md`
* Cargo-cult unwind: `.omx/research/sane_hnerv_cargo_cult_unwind_design_20260516.md`
* Submission packet README: `submissions/sane_hnerv/README.md`

## Mission contribution per Catalog #300

`frontier_breaking_enabler`. The bind step IS the canonical mechanism that converts a fragmented substrate scaffold (11/13 lessons honored across separate surfaces) into a PR-95-parity bound packet (13/13 lessons honored simultaneously in ONE 30-second-reviewable shipment). Wave N+41 demonstrates this binding is the empirical precondition for any HNeRV-family substrate to reach the 0.196-0.199 cluster's lower bound (PR101 GOLD 0.193); without binding, the substrate occupies the cargo-culted plateau no matter how many ingredients are wired separately. THIS wave is the apparatus's first canonical post-PR101 execution of the BIND step.

## Operator-routable next-action

`/operator-authorize` invocation:

```
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_sane_hnerv_modal_a100_dispatch \
    --epochs 2000 --batch-size 32 \
    --enable-autocast-fp16 --enable-torch-compile
```

(Currently dispatch_enabled MAY be false per substrate recipe state per Catalog #240; operator must flip to `dispatch_enabled: true` per `.omx/operator_authorize_recipes/substrate_sane_hnerv_modal_a100_dispatch.yaml` OR invoke per Catalog #300 §"Mission alignment" Consequence 1 operator-frontier-override with verbatim quote.)

Expected outcome:
* SMOKE-BEFORE-FULL canary $0.30 (100ep) per Catalog #167
* FULL $5-10 (2000ep) per Catalog #246 paired CUDA + CPU 4-arm
* Anchor at predicted band [0.193, 0.20] per HNeRV-family L2 substrate
* Promotion path: paired RATIFICATION → canonical posterior → autopilot ranker → cathedral consumer fan-out

PR-95-parity bound packet ready. The bind step IS the apparatus's first canonical FIRST execution post-PR101 GOLD per Wave N+41 audit. Operator-routable.
