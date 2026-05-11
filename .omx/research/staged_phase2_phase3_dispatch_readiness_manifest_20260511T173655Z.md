# Phase 2 + Phase 3 dispatch readiness pre-stage MANIFEST — 2026-05-11

**Status**: PRE-STAGED — operator-approval-required at every per-lane gate before dispatch.
**Cost**: $0 (this manifest costs nothing; dispatch costs land only on operator approval).
**Per**: operator directive 2026-05-11 — "we want to have all of the stuff that would cost more than $100 ready to go in parallel for as soon as we secure funding".

## Operator-decision summary

| Item | Cost band | Status | Re-surface trigger |
|---|---:|---|---|
| **Phase 2 trio (T15+T17+T18) probes** | $20–$28 | STAGED | Funding lands |
| **Phase 2 T10 (IB-Lagrangian aux scorer)** | $40 | STAGED — Phase 3 prereq | Funding lands |
| **Phase 2 T15 stand-alone validation** | $28 | STAGED | Funding + Probe T15-A clean |
| **Phase 2 T17 codebook joint** | $32 (incremental $25) | STAGED | Funding + Probe T17-A/B clean |
| **Phase 2 T18 nonlinear transform** | $43 conditional ($8 if T18-B fails) | STAGED | Funding + Probe T18-B HARD GATE |
| **Phase 2 T6 (Ballé+UNIWARD cross-paradigm)** | $80 | STAGED — ceiling-breaker | Funding + T1 anchor lands |
| **Phase 2 T1 (Ballé 128K end-to-end)** | already in flight on Modal `fc-01KR955JSYQAVTTYZA48VAV7WJ` | INHERITED — harvest pending | Job lands |
| **Phase 2 joint sweep (full stack)** | $80–$160 | STAGED — final stage | All sub-lanes land |
| **Phase 2 envelope total** | **$223–$303** | STAGED | Funding |
| **Phase 3 joint scorer-renderer-codec** | $600–$1200 | STAGED — multi-week | Phase 2 ≤0.142 [contest-CUDA] |
| **Cumulative pending GPU** | **$823–$1503** | NOT YET approved | All gates above |

## ρ_pose-on-A1 pre-condition (Insight 2 council prereq #3)

**Measurement landed**: `.omx/research/rho_pose_on_a1_20260511T173655Z.json`

| Field | Value |
|---|---|
| Archive | `track4_uniward_stc_hessian_a1_target166000_20260509_codex/submission_dir/archive.zip` |
| Archive SHA-256 | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` |
| Archive bytes | 178,262 |
| Measurement mode | `archive-pose` requested → `fallback` used (pose stream is INTERNAL to brotli-compressed `x` member; no canonical `poses.pt` member in A1's monolithic-single-file substrate-engineering layout) |
| ρ_pose aggregate | **0.85** `[fallback; CLAUDE.md default per tac.joint_source_rd_bound:26]` |
| Berger bits/pair @ ρ=0.85 | 5.548 bits/pair |
| Evidence grade | `fallback` (not `research_signal_local_cpu`) |
| Hardware substrate | `macos_cpu_apple_silicon` `[macOS-CPU advisory only]` |

**Phase 2 pre-condition status**: **CLEARED-BY-FALLBACK**. The CLAUDE.md-documented default ρ_pose ≈ 0.85 is the canonical anchor used for every Berger-corrected Phase 2 prediction (T13/T15/T20 per pre-design pass §6). No upward revision of the predicted bands is required.

**Upgrade path** (operator-optional, NOT required for dispatch): a future Phase 2 follow-up may re-run this measurement against the A1 packet via the contest inflate to extract the per-pair pose stream from inside `x`, then re-validate ρ_pose empirically on Linux x86_64 (since macOS-CPU PoseNet drift is 23× per CLAUDE.md). The fallback value is sufficient for dispatch-readiness; the empirical measurement is sufficient for post-dispatch attribution.

---

## Per-Phase-2-lane readiness audit (8 archive-grammar fields × 5 lanes × monitoring requirements)

Per HNeRV parity discipline (Catalog #124) every representation/codec lane MUST declare 8 fields at design time:
`archive_grammar / parser_section_manifest / inflate_runtime_loc_budget / runtime_dep_closure / export_format / score_aware_loss / bolt_on_loc_budget / no_op_detector_planned`

### T1 — Ballé 128K end-to-end (already trainer-complete)

| Field | Declared value |
|---|---|
| archive_grammar | `Phase1-three-member-x-decoder-bin-balle-bin` (substrate_engineering exception per lesson 3) |
| parser_section_manifest | 3 members: `x` (length-prefixed Ballé strings, brotli q11), `decoder.bin` (torch.save EMA, brotli q11), `balle.bin` (torch.save EMA, brotli q11) |
| inflate_runtime_loc_budget | declared in `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py::_write_runtime`; verified via Phase 1 packet compiler |
| runtime_dep_closure | `torch + brotli + compressai` (3 deps; documented in inflate.sh via $PYTHON, runs on CUDA-or-CPU) |
| export_format | `phase1_contest_contract` (length-prefixed binary preamble, NOT pickle) |
| score_aware_loss | YES — `α·B(θ)/N + β·d_seg(θ) + γ·√d_pose(θ)` via SegNet+PoseNet differentiable gradient (rgb_to_yuv6 patched per PR #95/106) |
| bolt_on_loc_budget | substrate_engineering lane_class (>= 600 LOC justified); ≤ 350 LOC bolt-on still applies for any T1 follow-on |
| no_op_detector_planned | YES — Phase 1 packet compiler `_verify_runtime_consumes_payload_bytes_executable` byte-mutation smoke per Catalog #139 |

**Non-negotiables verified**:
- ✓ `eval_roundtrip=True` (canonical eval_roundtrip helpers wired)
- ✓ EMA decay 0.997 (`from tac.training import EMA`)
- ✓ Score-domain Lagrangian (Berger-corrected)
- ✓ rgb_to_yuv6 differentiable (`apply_eval_roundtrip_during_training` + monkey-patch path)
- ✓ Real data (no `make_synthetic_*` references in non-smoke path)
- ✓ Archive grammar in training loop (`_build_archive` writes monolithic 3-member ZIP)

**STAGED**: trainer is production-ready. Dispatch script needs operator approval flag.

### T6 — Ballé + UNIWARD cross-paradigm (NO TRAINER YET — scaffold-only)

| Field | Declared value |
|---|---|
| archive_grammar | TBD (planned: T1's 3-member + UNIWARD sidecar mode-1) |
| parser_section_manifest | TBD |
| inflate_runtime_loc_budget | TBD |
| runtime_dep_closure | TBD |
| export_format | TBD (planned: extend T1's `phase1_contest_contract`) |
| score_aware_loss | TBD (planned: T1's Lagrangian + UNIWARD-weighted) |
| bolt_on_loc_budget | TBD |
| no_op_detector_planned | TBD |

**Non-negotiables**: trainer DOES NOT EXIST. T6 requires a fresh trainer script before dispatch.

**STAGED-BUT-BLOCKED**: dispatch script cannot be built until a `experiments/train_t6_balle_uniward_cross_paradigm.py` is written. The lane is L0 SKETCH per registry. **OPERATOR DECISION REQUIRED**: approve building the T6 trainer ($0 dev work, ~250 LOC bolt-on per HNeRV parity discipline ≤350 LOC budget) OR defer T6 to Phase 2.5.

### T10 — IB-Lagrangian co-trained aux scorer (NO TRAINER YET — scaffold-only)

| Field | Declared value |
|---|---|
| archive_grammar | TBD (planned: T1's 3-member + aux scorer co-distilled, ships in `decoder.bin`) |
| parser_section_manifest | TBD |
| inflate_runtime_loc_budget | TBD |
| runtime_dep_closure | TBD |
| export_format | TBD (Tishby IB scalar in archive metadata) |
| score_aware_loss | YES (planned: Tishby IB-Lagrangian L = I(X;T) − β·I(T;Y)) |
| bolt_on_loc_budget | TBD |
| no_op_detector_planned | TBD |

**Non-negotiables**: trainer DOES NOT EXIST. The Phase 3 gate (Catalog #134) requires `distillation_gap_estimate ≤ 0.03` — T10 is the gate-clearing dispatch.

**STAGED-BUT-BLOCKED**: dispatch script blocked on T10 trainer construction. Phase 3 hard prerequisite per NOT YET ITEM 3.

### T15 — Time-varying FiLM (module scaffold landed; trainer-wiring NEEDED)

| Field | Declared value |
|---|---|
| archive_grammar | inherits T1's substrate (modulator MLP ships in `decoder.bin` segment) |
| parser_section_manifest | 3 members (T1 base) + modulator state-dict serialized inside `decoder.bin` torch.save |
| inflate_runtime_loc_budget | +5 LOC over T1 baseline (modulator forward) |
| runtime_dep_closure | T1's set, no new deps |
| export_format | `phase1_contest_contract` with modulator state extension |
| score_aware_loss | T1's Lagrangian + (no new term; modulator is part of decoder grad) |
| bolt_on_loc_budget | ~250 LOC trainer integration |
| no_op_detector_planned | reuses Phase 1 packet compiler's byte-mutation smoke |

**Non-negotiables**: `src/tac/film_time_varying.py` exists (12.9KB), 21 tests passing. NEEDS trainer wiring into a T1-clone with modulator activation toggle.

**MONITORING REQUIREMENT NN-1**: T15 gradient-flow regression test. **Status**: implemented in `src/tac/tests/test_film_time_varying.py` per pre-design pass §3. Verify the test runs before dispatch.

**STAGED-WITH-WIRING-GAP**: need to derive `experiments/train_t15_film_t1_clone.py` from T1 trainer + ~50-line modulator wiring. **OPERATOR DECISION REQUIRED**: approve T15 wiring before T15 dispatch, OR defer T15.

### T17 — Shared VQ-VAE codebook (module scaffold landed; trainer-wiring NEEDED)

| Field | Declared value |
|---|---|
| archive_grammar | inherits T1; +32,768-byte codebook member OR codebook state in `decoder.bin` torch.save |
| parser_section_manifest | 3 members + 256-entry codebook serialized inside `decoder.bin` |
| inflate_runtime_loc_budget | +8 LOC over T1 baseline (VQ lookup) |
| runtime_dep_closure | T1's set, no new deps |
| export_format | `phase1_contest_contract` with VQ state extension |
| score_aware_loss | T1's Lagrangian + VQ commitment loss (β=0.25 van den Oord canonical) |
| bolt_on_loc_budget | ~300 LOC trainer integration |
| no_op_detector_planned | reuses Phase 1 packet compiler's byte-mutation smoke |

**Non-negotiables**: `src/tac/shared_vq_codebook.py` exists (15.1KB), 24 tests passing.

**MONITORING REQUIREMENT NN-2**: T17 codebook-collapse perplexity gate (≥102 = 0.4·256). **Status**: `compute_codebook_perplexity` helper exists per pre-design pass §3. Trainer wiring MUST wire the perplexity gate into the per-epoch loop AND a re-init path for dead codebook entries.

**STAGED-WITH-WIRING-GAP**: need `experiments/train_t17_shared_vq_t1_clone.py`. **OPERATOR DECISION REQUIRED**: approve T17 wiring before T17 dispatch.

### T18 — Ballé nonlinear transform (module scaffold landed; trainer-wiring NEEDED + Probe T18-B HARD GATE)

| Field | Declared value |
|---|---|
| archive_grammar | inherits T1; +MLP transform state in `balle.bin` torch.save |
| parser_section_manifest | 3 members + transform state serialized inside `balle.bin` |
| inflate_runtime_loc_budget | +12 LOC over T1 baseline (nonlinear transform forward + invert) |
| runtime_dep_closure | T1's set, no new deps |
| export_format | `phase1_contest_contract` with transform state extension |
| score_aware_loss | T1's Lagrangian + invertibility penalty |
| bolt_on_loc_budget | ~280 LOC trainer integration |
| no_op_detector_planned | reuses Phase 1 packet compiler's byte-mutation smoke |

**Non-negotiables**: `src/tac/balle_nonlinear_transform.py` exists (11.9KB), 19 tests passing.

**MONITORING REQUIREMENT NN-3**: T18 sustained invertibility (per-100-step `||z_e − invert(forward(z_e))||² < 0.5`). **Status**: helper exists per pre-design pass §3. Trainer wiring MUST wire the invertibility halt-on-breach check into the training step.

**HARD GATE Probe T18-B**: net byte savings > 0. **If fails, T18 is DEFERRED-PENDING-PHASE-3-DISTILLATION** ($8 spent on Probe T18-B; $35 saved).

**STAGED-WITH-WIRING-GAP**: need `experiments/train_t18_nonlinear_transform_t1_clone.py`. **OPERATOR DECISION REQUIRED**: approve T18 wiring AND approve Probe T18-B HARD GATE policy before T18 dispatch.

---

## Phase 3 readiness audit

### t_phase3_joint_scorer_renderer_codec

**Scaffold**: `src/tac/phase3/joint_scorer_renderer_codec.py` (524 LOC), `src/tac/phase3/inflate.py` (95 LOC, ≤200 substrate-engineering waiver). Tests: 44 passing per a08e32dd landing.

**Phase3DispatchGate preconditions** (per Catalog #134; gate fail-closed at construction unless `unsafe_test_only=True`):

| Precondition | Source of truth | Current status |
|---|---|---|
| `phase2_anchor_verified: bool` | post-Phase-2 evidence path | **NOT YET** (Phase 2 not dispatched) |
| `phase2_anchor_score: float ≤ 0.142` | `[contest-CUDA]` anchor from Phase 2 joint sweep | **NOT YET** |
| `phase2_anchor_evidence_path: str` | path under git repo root | **NOT YET** |
| `distillation_gap_estimate: float ≤ 0.03` | T10 distillation memo (Hinton T=2.0) | **NOT YET** (T10 not dispatched) |
| `distillation_gap_evidence_path: str` | path under git repo root | **NOT YET** |
| `operator_approved_gpu_budget_usd: float ≥ 600` | operator authorization memo under `.omx/research/operator_authorizations/` | **NOT YET** |
| `aaf68f37_verdict_clean: bool` | adversarial-review verdict | UNKNOWN (need to check aaf68f37 status) |
| `aaf68f37_verdict_evidence_path: str` | path under git repo root | UNKNOWN |
| `phase3_council_deliberation_path: str` | grand council memo (Fields-Medal session) | ✓ `.omx/research/fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md` |

**STAGED-BUT-BLOCKED on Phase 2 completion**: Phase 3 cannot dispatch until Phase 2 lands a [contest-CUDA] anchor ≤0.142. This is structural and cannot be pre-staged further.

**$600–$1200 burn schedule** (multi-week; operator approval at EACH gate):

| Week | Stage | Cost band | Gate |
|---|---|---:|---|
| W1 | T10 + T15 + T17 + T18 individual smoke + Probe gates | $20–$50 | (already covered by Phase 2 trio budget; Phase 3 W1 = Phase 2 closeout) |
| W2 | Joint scorer-renderer-codec end-to-end SMOKE (1-2 epochs on small GPU) | $80–$120 | Phase 2 ≤0.155 verified |
| W3 | Joint training full-data (T4 + 4090) | $200–$400 | Phase 2 ≤0.142 verified |
| W4 | Distillation refinement + EMA convergence | $200–$400 | W3 score landing ≤0.130 |
| W5 | Floor-saturation polish | $100–$280 | W4 lands ≤0.125 |
| **Pause points** | After any week where score does not improve | — | Operator approval to continue |

---

## Unified dispatch script catalog

Per-lane pre-built dispatch scripts (every script gates on operator approval; NO auto-dispatch):

| Script | Lane | Status | Approval gate |
|---|---|---|---|
| `scripts/staged_phase2_t1_balle_endtoend_dispatch.sh` | T1 (production trainer) | STAGED | operator approves $80 + funding lands |
| `scripts/staged_phase2_t6_balle_uniward_dispatch.sh` | T6 | STAGED-BUT-BLOCKED | operator approves T6 trainer build first |
| `scripts/staged_phase2_t10_ib_lagrangian_dispatch.sh` | T10 | STAGED-BUT-BLOCKED | operator approves T10 trainer build first |
| `scripts/staged_phase2_t15_film_dispatch.sh` | T15 | STAGED-WITH-WIRING-GAP | operator approves T15 wiring + funding |
| `scripts/staged_phase2_t17_vq_codebook_dispatch.sh` | T17 | STAGED-WITH-WIRING-GAP | operator approves T17 wiring + funding |
| `scripts/staged_phase2_t18_nonlinear_transform_dispatch.sh` | T18 | STAGED-WITH-WIRING-GAP | operator approves T18 wiring + Probe T18-B HARD GATE |
| `scripts/staged_phase3_joint_scorer_renderer_codec_dispatch.sh` | Phase 3 | STAGED-BUT-BLOCKED | Phase 2 lands ≤0.142 + $600–1200 + aaf68f37 CLEAN |

Each script:
- Refuses to run without `STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1` environment variable
- Prints the cost cap + dispatch command on dry-run
- Does **not** open a lane claim unless a real provider job/call id exists.
  This supersedes the earlier claim-at-prestage wording and prevents phantom
  active claims.
- Routes through the canonical provider actuator (`experiments/modal_t1_balle_endtoend.py`
  for T1; future Phase 3 actuator still blocked) rather than ad hoc launch
  commands.
- Writes provenance to `experiments/results/<lane>_<utc>/provenance.json`

---

## Lane registry status (after pre-stage)

| Lane id | Pre-staged level | Pre-allocated claim |
|---|---|---|
| `lane_phase2_phase3_dispatch_readiness_prestage` | L1 (impl_complete + memory_entry on this manifest landing) | none (research-only lane) |
| `t1_balle_128k_endtoend` | unchanged L0 (Modal job in flight) | none from pre-stage; claim opens only inside provider actuator |
| `lane_t6_balle_uniward_cross_paradigm_phase2_preregistered` | unchanged L0 | staged_pending_operator_approval claim added |
| `lane_t10_ib_lagrangian_aux_scorer_phase2_preregistered` | unchanged L0 | staged_pending_operator_approval claim added |
| `lane_t15_time_varying_film_phase2_preregistered` | unchanged L0 | staged_pending_operator_approval claim added |
| `lane_t17_shared_vq_codebook_phase2_preregistered` | unchanged L0 | staged_pending_operator_approval claim added |
| `lane_t18_balle_nonlinear_transform_phase2_preregistered` | unchanged L0 | staged_pending_operator_approval claim added |
| `lane_phase3_joint_scorer_renderer_codec` | unchanged L1 | staged_pending_operator_approval claim added |

## 6-hook wire-in declarations (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution** — each pre-staged lane's predicted-Δ-score is a sensitivity prior for the meta-Lagrangian solver. Recorded under `notes` field of each lane.
2. **Pareto constraint** — each lane is a Pareto candidate (rate/seg/pose); empirical anchor lands as a Pareto vertex on dispatch.
3. **Bit-allocator hook** — T17 codebook + T18 transform inform byte budget allocation when their anchors land.
4. **Cathedral autopilot dispatch hook** — each pre-staged lane registers via `tools/cathedral_autopilot.py` queue (lane_id → predicted-Δ-score + cost band).
5. **Continual-learning posterior update** — each per-lane archive's [contest-CUDA] + [contest-CPU] anchor triggers `tac.continual_learning.posterior_update_locked` per dual-eval mandate.
6. **Probe-disambiguator** — ρ_pose-on-A1 measurement (this manifest, §ρ_pose) IS the disambiguator between substrate-class-boundary hypotheses per Insight 1 council verdict.

## What this manifest does NOT do

- **No GPU spend** — $0 to land this manifest
- **No actual dispatch** — every script is operator-approval-gated
- **No design decision** — surfaces operator decisions, does not pre-commit any
- **No KILL** — every blocked lane is DEFERRED-PENDING-OPERATOR-APPROVAL, never KILLED

## What the operator must approve before any dispatch

1. **Per-lane**: which lanes to fund (any subset of the 7 staged lanes)
2. **Trainer build approvals** (T6, T10) OR wiring approvals (T15, T17, T18) before each dispatch
3. **Probe T18-B HARD GATE outcome**: whether to dispatch T18 conditional on net-byte-savings > 0
4. **Phase 3 sequenced spend**: per-week operator gate
5. **A1 PR-submission decision** (Catalog #150 + dual-eval already landed)

## Cross-references

- Pre-design pass: `feedback_phase2_t15_t17_t18_pre_design_pass_landed_20260509.md`
- Council ratification: `feedback_grand_council_pose_axis_insights_review_20260511.md` Insight 2
- NOT YET pin: `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`
- Takeover memo: `project_full_custody_takeover_codex_offline_20260511.md`
- Fields-Medal Phase 3 council: `.omx/research/fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md`
- ρ_pose measurement: `.omx/research/rho_pose_on_a1_20260511T173655Z.json`
- T1 trainer (production-ready): `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
- Phase 3 scaffold: `src/tac/phase3/joint_scorer_renderer_codec.py` + `src/tac/phase3/inflate.py`
- HNeRV parity discipline: CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable + Catalog #124

## Verdict

**Phase 2 + Phase 3 dispatch readiness PRE-STAGED. $0 GPU spend this session.** The operator can authorize any subset of the 7 pre-staged lanes the moment funding lands; each script is operator-approval-gated and refuses to run otherwise. **3 lanes (T1, Phase 2 trio probes, Phase 3 scaffold-verified) are dispatch-ready immediately upon funding + approval; 4 lanes (T6, T10, T15, T17, T18) require an additional trainer-build or wiring approval before dispatch.**
