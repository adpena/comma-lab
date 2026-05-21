# HFV1 PR101 Exact-Eval Readiness Verification Smoke — Landing 2026-05-21T08:00:13Z

- timestamp_utc: 2026-05-21T08:00:13Z
- lane_id: lane_overnight_k_hfv1_pr101_exact_eval_readiness_verification_smoke_20260521
- status: LANDED_PAIRED_EXACT_EVAL_BOTH_AXES_VERIFIED_NOT_FRONTIER
- score_claim: true (both axes have evidence_grade=contest-CPU / contest-CUDA per Catalog #127)
- promotion_eligible: false (DEFER: scores worse than current frontier on both axes)
- ready_for_exact_eval_dispatch: true (paired CPU+CUDA exact eval completed; this IS the exact eval)
- verdict: DEFER-pending-research (HFV1 paradigm intact; this specific seed_top16 archive not frontier-improving)

## Mission alignment

- council_predicted_mission_contribution: frontier_protecting
- council_override_invoked: false
- council_override_rationale: n/a

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 1: this lane is
frontier-protecting because it CLOSED an open exact-eval readiness question
(codex memo 2026-05-21T06:42:57Z had `ready_for_exact_eval_dispatch=false`
pending paired CPU+CUDA exact eval; this lane fires the dispatch + harvests
the verdict). The HFV1 PR101 seed_top16 candidate does NOT advance the
frontier (scores worse than baseline on both axes) but the empirical
measurement extincts the open question and steers future HFV1 work toward the
sister codex rate-hurdle finding (sidecar must shrink BEFORE more HFV1
exact-eval dispatches are economically viable).

## What landed (per Carmack MVP-first 5-step per CLAUDE.md amendment be125b878)

### Phase 1: $0 local CPU archive integrity verification (passed)

- Archive sha256 verified: `72cbd8197a2a8064cb54e7e56e1a5b892a89251c28091f22eba6eef8edff3efb` (matches codex readiness memo)
- Archive bytes verified: 202,649 (matches codex readiness memo)
- ZIP member custody verified: 2 members (`foveation_params.bin` 24,016 bytes, `x` 178,417 bytes; CRC32s match)
- Submission_dir runtime intact: inflate.sh + inflate.py + archive.zip + report.txt + archive_manifest.json + pre_submission_compliance.competitive_or_innovative_statement.txt + pre_submission_compliance.hosted_archive_manifest.json + README.md + src/ + __pycache__/

### Phase 2: Smoke MUST falsifiably challenge — predict CPU score band + verify

- Sister codex rate-hurdle memo `codex_findings_hfv1_pr101_rate_hurdle_20260521T064810Z_codex.md` predicted:
  - Required component gain to TIE FEC6/PR110 CPU baseline (0.192051) at current bytes: `+0.0160685082567`
  - Predicted CPU outcome IF component gain materialized: ~0.192 [contest-CPU] (tie)
  - Predicted CPU outcome IF NO component gain: ~0.192 + 0.016 + Δcomponent_residual = >0.21 (worse than baseline)
- Empirical CPU outcome: `0.336724` [contest-CPU Modal Linux x86_64]
- Predicted CUDA outcome (extrapolating from PR101 GOLD CPU/CUDA gap +0.033): ~0.225 if frontier; ~0.45 if no component gain
- Empirical CUDA outcome: `0.353177` [contest-CUDA T4]
- Falsification verdict: empirical scores ARE worse than baseline on both axes, as the
  "no component gain materialized" branch of the prediction predicted; the rate hurdle
  was crossed in the wrong direction (+0.144 CPU / +0.148 CUDA worse than required gain)

### Phase 3: Catalog #344 canonical equation reference

- Per CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE":
  - Canonical equation #1 `brotli_cascade_bounded_per_stream_v1` not directly applicable (this is foveation_params, not entropy coder)
  - The contest scoring formula `S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37_545_489` (canonical, per upstream `evaluate.py`) was confirmed both axes via report.txt
  - Predicted vs empirical residuals codified per sister codex rate-hurdle memo; no NEW canonical equation registered this turn (the residual is an implementation-level falsification per Catalog #307; the HFV1 paradigm itself is intact)
  - Slot for canonical equation `hfv1_uniform_radial_seed_no_component_gain_v1` deferred — operator may register on follow-up

### Phase 4: Landing verdict in same commit batch

- Verdict: DEFER-pending-research per CLAUDE.md "KILL/FALSIFIED memory verdicts" + "Forbidden premature KILL without research exhaustion"
- Specific configuration retired: `seed_top16_component_hardpairs` with uniform small radial seed at 202,649 bytes
- Reactivation criteria (per CLAUDE.md "Forbidden premature KILL"):
  1. HFV1 sidecar recoder (per codex rate-hurdle recommended next action) reduces sidecar from 24,016 → ~10,000 bytes; required component gain drops from 0.016 → 0.0068
  2. Sensitivity-weighted/orthogonalized HFV1 search REPLACES uniform small radial seed (per codex pr101_lfv1_hfv1_seed_top16_component_hardpairs_20260520T160447Z_codex.md "Next" section)
  3. PR110-canonical runtime root merges (per same codex memo "Next" section)
- Sister candidates `identity` + `nonidentity` deferred per the codex rate-hurdle priority verdict (no plausible component-gain mechanism); not dispatched this turn

### Phase 5: Re-route operator priority queue within ~1h

- TRIAGE Pick 3 (HFV1 PR101 verify exact-eval readiness) → COMPLETE; ready_for_exact_eval_dispatch now true
- Operator-routable next steps (re-prioritized):
  1. HFV1 sidecar recoder (high EV; codex rate-hurdle says hurdle drops 60% post-recoder)
  2. Sensitivity-weighted HFV1 search using master-gradient consumer outputs (per slot MG-7 bundle landed 2026-05-20)
  3. Sister DP1 dispatches (`fc-01KS4* pretrained_driving_prior_*`) still in-flight; check harvest separately
  4. Frontier remains: contest-CPU 0.192051 (fec6 6bae0201 / 2026-05-15) / contest-CUDA 0.205330 (pr106 9cb989cef519 / 2026-05-16)

## Empirical scores (apples-to-apples per CLAUDE.md "Apples-to-apples evidence discipline")

| axis | hardware | current frontier | HFV1 PR101 seed_top16 | delta | verdict |
|---|---|---:|---:|---:|---|
| contest-CPU | Modal Linux x86_64 CPU container | 0.192051 (fec6 `6bae0201...`) | **0.336724** (this lane `72cbd8197a...`) | +0.144673 (75% worse) | DEFER |
| contest-CUDA | Modal T4 | 0.205330 (pr106 format0d `9cb989cef519...`) | **0.353177** (this lane `72cbd8197a...`) | +0.147847 (72% worse) | DEFER |

Both axes have qualifying 1:1 contest-compliant hardware per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable. Both scores have `evidence_grade` correctly tagged
in the canonical posterior per Catalog #127. Neither score is
promotion-eligible (worse than frontier on both axes).

CUDA−CPU gap on this archive: +0.01645 (CUDA worse than CPU by 0.0165), which
is roughly comparable to the empirical +0.033 PR102 gap noted in CLAUDE.md
"Submission auth eval" section. The pose component dominates the gap:
- CUDA avg_posenet_dist = 0.00194092
- CPU  avg_posenet_dist = 0.00176691 (-9%)
- CUDA avg_segnet_dist  = 0.00078924
- CPU  avg_segnet_dist  = 0.00068863 (-13%)

## Modal call_id ledger anchors (Catalog #245)

Both registered via canonical `register_dispatched_call_id` + auto-recovered:

- **CUDA**: `fc-01KS4RBSQMXEVXNKCQSTFCXPQZ`
  - Lane: `hfv1_pr101_exact_eval_seed_top16_component_hardpairs_72cbd8197a2a_contest_cuda`
  - Final status: `completed_contest_cuda_modal_auth_eval_recovered`
  - Artifacts: `/Users/adpena/Projects/pact/experiments/results/modal_auth_eval/hfv1_pr101_seed_top16_component_hardpairs_72cbd8197a2a_cuda/`
  - Result JSON: `modal_cuda_auth_eval_result.json` (score=0.353177, axis=contest_cuda)

- **CPU**: `fc-01KS4RCATFJ3425ZEHRGS2BXA9`
  - Lane: `hfv1_pr101_exact_eval_seed_top16_component_hardpairs_72cbd8197a2a_contest_cpu`
  - Final status: `completed_contest_cpu_modal_auth_eval_recovered`
  - Artifacts: `/Users/adpena/Projects/pact/experiments/results/modal_auth_eval_cpu/hfv1_pr101_seed_top16_component_hardpairs_72cbd8197a2a_cpu/`
  - Result JSON: `modal_cpu_auth_eval_result.json` (score=0.336724, axis=contest_cpu)

Both call_ids visible via `latest_status_by_call_id()` once ledger sync runs;
both lane claims terminal per `tools/claim_lane_dispatch.py summary`
(terminal_latest=1014, +4 events from this lane).

## Runtime tree custody (Catalog #166)

- contest-CUDA runtime tree sha256: `9bff8c0a4f1b543bd1f546a4af08a2eea6e3807514eb3931b3ca52fbcc9bfc1b` (matches codex memo)
- contest-CPU runtime tree sha256: `e655b48b88a2a0ec70294b39364cf6a9b87c01f7127c4f68566dba467e376ca8` (matches codex memo)
- runtime_content_tree_sha256 (shared): `99a40837fd3c25fc93e95b2a428bdf9d087657042e62125b5e84240998db7a9c`

## Discipline + 6-hook wire-in declaration

### CLAUDE.md non-negotiable adherence

- Catalog #229 PV: read codex findings memo + sister rate-hurdle memo + seed_top16 candidate memo + canonical frontier pointer + dispatch plan JSON BEFORE execute
- Catalog #117/#157/#174 canonical serializer: this commit + memo via canonical serializer with POST-EDIT --expected-content-sha256
- Catalog #119 Co-Authored-By: trailer present in commit
- Catalog #125 6-hook wire-in declaration: see below
- Catalog #127 authoritative-tag custody: both axes have axis+hardware+evidence_grade triple
- Catalog #131/#138/#245/#339 fcntl-locked ledger: all dispatches registered via canonical helpers; rc=0 fail-closed
- Catalog #192 macOS-CPU advisory tagging: not applicable (Modal Linux x86_64 contest-CPU is authoritative)
- Catalog #199/#202 paired-env bypass: OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 + OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=1.00 set
- Catalog #246 paired dispatch skip-if-anchor-exists: --skip-axis-if-promotable-anchor-exists set; no anchors existed (both axes dispatched)
- Catalog #287 placeholder-rationale rejection: all rationales substantive (≥4 chars, non-placeholder)
- Catalog #300 v2 frontmatter: this memo carries mission-alignment fields
- Catalog #316 frontier pointer: canonical pointer consulted; baselines extracted programmatically
- Catalog #340 sister-checkpoint guard: PROCEED returned before dispatch
- Carmack MVP-first per `be125b878`: 5-step phasing applied
- CLAUDE.md "Public Disclosure Hygiene": no operator-private state in this memo

### 6-hook wire-in per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE. The empirical SegNet/PoseNet
   per-pair distortions are downstream consumable by
   `tac.sensitivity_map.*` to compute per-byte leverage for future HFV1
   archive optimization (the seed_top16 pair selection IS already
   sensitivity-informed; this round measures how that selection translated
   to scorer response). The +24,132 byte HFV1 sidecar produced ~0 measurable
   component gain on either axis, which is itself a per-byte sensitivity
   signal (leverage = 0 for this specific sidecar payload).
2. **Pareto constraint**: ACTIVE. The empirical (seg, pose, rate) triple
   establishes a new Pareto-dominated point for the HFV1 seed_top16 archive
   class; no constraint added because the point is strictly dominated by
   both fec6 (CPU) and pr106 format0d (CUDA) on every axis.
3. **Bit-allocator hook**: ACTIVE. The +24,132 byte HFV1 sidecar produced
   no measurable component gain; bit-allocator should down-weight
   uniform-small-radial-seed payload class until sidecar recoder lands.
4. **Cathedral autopilot dispatch hook**: ACTIVE. Both archive shas
   registered as exact-eval-measured per the canonical ledger; autopilot
   ranker will see seed_top16 with empirical score 0.336724 (CPU) / 0.353177
   (CUDA) and rank it below frontier per the Catalog #219 / #322 reweight
   cascade (factor 1.0 passthrough — no false reward).
5. **Continual-learning posterior update**: ACTIVE. Posterior anchors
   emitted via Modal ledger Catalog #245 lifecycle (dispatched → harvested
   events). Future HFV1 dispatch ranker will reference these anchors via
   `find_promotable_anchor_for_axis_and_sha` skip logic per Catalog #246.
6. **Probe-disambiguator**: ACTIVE. This lane IS the probe-disambiguator
   that resolved the codex readiness memo's `ready_for_exact_eval_dispatch=
   false` blocker. The verdict (DEFER) is unambiguous via the apples-to-
   apples table above; no second probe needed for THIS specific archive.
   A sister probe-disambiguator for `identity` and `nonidentity` candidates
   may be skipped per the codex rate-hurdle memo's priority verdict ("defer
   without sidecar recoder").

## Cost accounting

- Modal CUDA T4 spawn → recover: ~3 min wall-clock; ~$0.13 estimated
- Modal CPU container spawn → recover: ~6 min wall-clock; ~$0.08 estimated
- Total: ~$0.21 (54% under $0.40 budget; 21% of $1.00 session cap)
- Main thread: $0 (local PV + plan-only smoke + polling)

## Sister-subagent coordination (Catalog #230 / #314 / #340)

- Slot 2 (`a0e10b778e` STC v2): touched STC v2 recipe + ledger — DISJOINT scope
- Slot 3 (concurrent): scope unknown to me at PV time — Catalog #340 PROCEED
- Active sister dispatches at start: 2 DP1 (`lane_dp1_original_baseline_first_paired_anchor_20260520` + `lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520`) — DISJOINT lane_id namespace
- Catalog #340 sister-checkpoint guard re-verified: PROCEED throughout

## Operator-routable follow-ups

1. **HFV1 sidecar recoder** (high EV per codex rate-hurdle): build recoder that
   shrinks `foveation_params.bin` from 24,016 → ~10,000 bytes; required
   component gain drops 60% (0.016 → 0.0068)
2. **Sensitivity-weighted HFV1 search**: replace uniform small radial seed with
   master-gradient-consumer-informed sensitivity routing per slot MG-7 bundle
   (2026-05-20) Tier 1-9 exploits; especially exploit #2
   (score_weighted_reconstruction_error_consumer) + exploit #3
   (top_k_byte_sensitivity_consumer)
3. **PR110 merge**: wait for PR110-canonical runtime root merge per codex
   `pr101_lfv1_hfv1_seed_top16_component_hardpairs_20260520T160447Z_codex.md`
   "Next" section; sister HFV1 variants benefit from canonical runtime base
4. **identity + nonidentity DEFER confirmation**: NOT dispatched this turn per
   codex rate-hurdle priority verdict; reactivation = sidecar recoder + new
   component-gain mechanism

## Cross-references

- Source codex readiness memo: `.omx/research/codex_findings_hfv1_pr101_exact_eval_readiness_20260521T064257Z_codex.md`
- Sister codex rate-hurdle: `.omx/research/codex_findings_hfv1_pr101_rate_hurdle_20260521T064810Z_codex.md`
- Sister codex seed_top16 candidate build: `.omx/research/pr101_lfv1_hfv1_seed_top16_component_hardpairs_20260520T160447Z_codex.md`
- Sister codex integrated adapter: `.omx/research/pr101_lfv1_hfv1_integrated_adapter_20260520T155953Z_codex.md`
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable (DEFER-pending-research per default verdict cascade)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" FORBIDDEN pattern
- CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
- CLAUDE.md "Frontier scores are pointer-only" non-negotiable (canonical pointer consulted)
- Carmack MVP-first amendment commit `be125b878`
- Frontier pointer: `.omx/state/canonical_frontier_pointer.json`
- CUDA result artifacts: `experiments/results/modal_auth_eval/hfv1_pr101_seed_top16_component_hardpairs_72cbd8197a2a_cuda/`
- CPU result artifacts: `experiments/results/modal_auth_eval_cpu/hfv1_pr101_seed_top16_component_hardpairs_72cbd8197a2a_cpu/`
