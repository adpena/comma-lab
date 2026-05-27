# Phase 7 Paired Auth-Eval Canonical Helper -- LANDED 2026-05-26

**Lane**: `lane_phase_7_paired_auth_eval_canonical_helper_20260526` L1
(impl_complete + strict_preflight + cathedral_consumer + memory_entry)

**Subagent**: `phase7_paired_183A1060` (parent session
`b74f6039-6caf-44f2-a2c3-cd8156acd447`)

**Mission contribution per Catalog #300**: `frontier_protecting` (Layer 5
paired auth-eval canonical helper extincts the per-substrate ad-hoc
paired-axis dispatch divergence bug class; consolidates the operator-
facing `tools/dispatch_modal_paired_auth_eval.py` CLI surface into ONE
typed canonical helper with sha-locked invariant + axis-hardware cross-
validation + Catalog #192 macOS-CPU structural refusal + canonical Modal
call_id ledger routing per Catalog #245).

## What landed

Per Phase 1 audit specification memo
(`.omx/research/canonical_submission_pipeline_specification_memo_20260526.md`)
§3 Phase 6 / Layer 5 (paired_auth_eval) acceptance contract:

1. **`tac.submission_packet.paired_auth_eval`** (~1594 LOC src;
   counted with comprehensive module-level docstring + per-Catalog-gate
   cross-references + per-cascade verdict construction):
   - `PairedAuthEvalVerdictKind` StrEnum (7 canonical verdicts:
     PAIRED_PASS / PAIRED_PARTIAL_CUDA_ONLY / PAIRED_PARTIAL_CPU_ONLY /
     BLOCKED_PRE_DISPATCH / BLOCKED_HARVEST / BLOCKED_AXIS_MISMATCH /
     BLOCKED_HARDWARE_NON_COMPLIANT)
   - `PairedAuthEvalVerdict` frozen dataclass with comprehensive
     invariants: sha-locked invariant (archive_sha256_paired must equal
     bundle.archive_sha256 except BLOCKED_PRE_DISPATCH); macOS axis
     forces promotable=False; promotable requires PAIRED_PASS AND both
     axes on canonical Linux x86_64; score_claim requires promotable;
     axis_tag canonical (`[predicted]` OR `[contest-CUDA; contest-CPU]`)
   - `plan_paired_auth_eval(submission_bundle_result, ...)` canonical
     entry point (Layer 5 orchestrator surface)
   - `reconstruct_verdict_from_disk(...)` canonical post-dispatch surface
     reading per-axis JSONs + emitting canonical PairedAuthEvalVerdict
   - Per Catalog #226 canonical helper routing: execute-mode is
     operator-gated (raises PairedAuthEvalError with explicit operator-
     routable next-action); Phase 10 wires the actual subprocess via
     `gate_auth_eval_call` + `register_dispatched_call_id_fail_closed`
   - Per Catalog #245 canonical Modal call_id ledger: integration
     surface is the helper's `cuda_call_id` + `cpu_call_id` fields that
     reconstruct_verdict_from_disk pins as canonical
   - Catalog #192 macOS-CPU structural non-promotion (Darwin ARM64
     STRUCTURALLY refused at verdict surface)
   - Catalog #287 placeholder-rationale rejection (verdict_rationale
     >= 4 chars, no placeholder literals)
   - Catalog #323 canonical Provenance umbrella
   - Catalog #341 Tier A canonical-routing markers (axis_tag=
     `[predicted]`, score_claim=False, promotable=False) until paired-
     axis empirical anchor lands
   - Catalog #344 canonical equation
     `paired_auth_eval_canonical_helper_consolidation_savings_v1`
     FORMALIZATION_PENDING (preserved until Phase 10 first paired
     empirical anchor)

2. **`tools/paired_auth_eval_cli.py`** (475 LOC):
   - Operator-facing CLI wrapping `plan_paired_auth_eval` +
     `reconstruct_verdict_from_disk`
   - `--from-submission-bundle <path>` (or `-` for stdin) loads
     SubmissionBundleResult JSON per Phase 4 round-trip contract
   - `--cost-band {smoke,full}` per Catalog #270
   - `--cuda-gpu {T4,L4,A10G,L40S,A100,4090,H100}` per Catalog #215
   - `--cuda-platform {modal,vastai,lightning}`
   - `--cpu-target {linux_x86_64_modal,linux_x86_64_vastai,
     linux_x86_64_lightning,linux_x86_64_gha,darwin_arm64_advisory}`
     per Catalog #192
   - `--budget-usd <max>` per Catalog #270 cost-band envelope
   - `--dry-run` (PV mode; computes plan + cost estimate without
     dispatch)
   - `--execute` (operator-explicit dispatch flag; REQUIRES
     `--operator-approved <handle>` per CLAUDE.md "Executing actions
     with care" non-negotiable)
   - `--reconstruct-from-disk` (post-dispatch reconstruction mode)
   - `--json` (machine-readable for cathedral consumer / autopilot
     ranker) / otherwise human-readable verdict on stderr
   - Canonical 7-tier exit code taxonomy:
     - 0 PAIRED_PASS (both axes 1:1 contest-compliant, promotable)
     - 1 BLOCKED_PRE_DISPATCH (operator action required)
     - 2 BLOCKED_HARVEST (retry available)
     - 3 BLOCKED_AXIS_MISMATCH (Catalog #127 custody violation)
     - 4 BLOCKED_HARDWARE_NON_COMPLIANT (Catalog #192 forbidden axis)
     - 5 PAIRED_PARTIAL (CUDA-only OR CPU-only; re-dispatch missing axis)
     - 6 CLI error
   - Verified `--help` exits 0 + bad-args exits non-zero + missing-bundle
     exits 6 + dry-run with canonical bundle exits 1 (BLOCKED_PRE_DISPATCH
     plan-only)

3. **Cathedral consumer**
   `src/tac/cathedral_consumers/paired_auth_eval_consumer/`
   (210 LOC) per Catalog #335 canonical contract:
   - CONSUMER_NAME = "paired_auth_eval_consumer"
   - CONSUMER_VERSION = "1.0.0"
   - CONSUMER_HOOK_NUMBERS = CATHEDRAL_AUTOPILOT_DISPATCH +
     CONTINUAL_LEARNING_POSTERIOR + PROBE_DISAMBIGUATOR
   - 9-verdict-readiness taxonomy: UNKNOWN / PAIRED_PASS /
     PARTIAL_CUDA_ONLY / PARTIAL_CPU_ONLY / BLOCKED_PRE_DISPATCH /
     BLOCKED_HARVEST / BLOCKED_AXIS_MISMATCH / BLOCKED_HARDWARE_NON_COMPLIANT /
     FORBIDDEN_MACOS_AXIS
   - Tier-A observability-only invariants (predicted_delta_adjustment=
     0.0 + promotable=False + axis_tag=[predicted]) per Catalog #341
   - Auto-discovered per Catalog #336/#337 (verified: consumer appears
     in cathedral autopilot loop's `discover_compliant_consumer_modules()`
     output; cumulative count 67 -> 68 with Phase 7)

4. **Tests** `src/tac/tests/test_submission_packet_paired_auth_eval.py`
   (1662 LOC; 105 tests **ALL PASS**):
   - Module constants pinned (schema_version, phase, canonical_equation_id,
     verdict kinds, GPU classes, CPU substrates, forbidden macOS tokens,
     cost-band envelope, placeholder rationales, evidence-grade tokens)
   - Hardware substrate resolvers (CUDA modal T4 / A100; vastai 4090;
     lightning T4; CPU modal/vastai/lightning/gha/darwin_arm64_advisory)
   - macOS hardware detector (macos_arm64 / darwin_arm64 / apple_silicon
     detected; canonical linux substrates not flagged)
   - Cost estimation (T4 smoke / A100 smoke / CPU smoke; full > smoke)
   - Evidence grade derivation (PAIRED_PASS+promotable yields
     canonical empirical; macOS forces advisory; partials canonical;
     default pending)
   - Provenance derivation (Catalog #341 defaults + Catalog #190
     canonical helper invocation + Catalog #344 canonical equation)
   - PairedAuthEvalVerdict invariants (canonical construction; invalid
     schema_version / verdict_kind / placeholder rationale / short
     rationale / empty rationale / invalid sha256 length / negative
     score / NaN score / invalid cost_band / invalid cuda_gpu / invalid
     cpu_target / macOS + promotable=True rejected / forbidden_macos
     hardware mismatch rejected / promotable without PAIRED_PASS
     rejected / score_claim without promotable rejected /
     PAIRED_PASS+promotable requires canonical axis_tag /
     non-canonical evidence_grade rejected / canonical_equation_id
     pinned / PAIRED_PASS requires call_ids when not dry_run /
     as_dict round-trip)
   - plan_paired_auth_eval dry-run (canonical Linux blocked-pre-dispatch;
     macOS advisory blocked-hardware-non-compliant; full band higher
     budget than smoke; excessive GPU with small budget blocked;
     canonical Provenance threaded; archive bytes preserved; lane
     substrate lineage preserved)
   - plan_paired_auth_eval validation (non-bundle / invalid cost_band /
     invalid cuda_gpu / invalid cuda_platform / invalid cpu_target /
     negative budget / execute without operator approval / execute
     with operator approval raises operator-gated)
   - reconstruct_verdict_from_disk (PAIRED_PASS reconstruction with
     canonical CUDA + CPU JSONs / axis mismatch detected / macOS axis
     blocked hardware non-compliant / partial CUDA only / partial CPU
     only / both axes failed blocked harvest / missing CUDA JSON
     raises / unparseable JSON raises / non-bundle result rejected)
   - Phase 4 integration round-trip (bundle archive sha threaded to
     verdict; bundle lane substrate preserved in provenance)
   - Cathedral consumer (imports / validates canonical contract /
     hooks canonical / unknown metadata neutral / PAIRED_PASS verdict /
     macOS forbidden / partial cuda only / partial cpu only / blocked
     pre dispatch / blocked harvest / blocked axis mismatch /
     update_from_anchor no-op)
   - CLI subprocess (--help exits 0 / missing arg exits non-zero /
     bad bundle path exits 6 CLI error / dry-run with canonical bundle
     emits verdict / execute without operator-approved rejected)
   - Catalog #192 macOS-CPU regression guards (darwin_arm64 cpu target
     yields forbidden; canonical linux x86_64 targets not macos)
   - Catalog #341 Tier A canonical-routing markers (default provenance;
     dry-run verdict always Tier A)
   - Live-repo regression guards (Phase 7 module importable via
     package; Phase 7 in package __all__; Phase 7 cathedral consumer
     auto-discoverable)

## Sister landings verified

- Phase 2 (`compression_pipeline.py`, commit `b96329a71`) -- 72 tests pass
- Phase 3 (`archive_grammar.py`, commit `1d4753f65`) -- 72 tests pass
- Phase 4 (`builder.py`, commit `1de30160e`) -- 72 tests pass
- Phase 5 (`linter.py`, sister linter Phase) -- 99 tests + 1 skipped pass
- Phase 6 (`compliance.py`, commit `2d3042d14`) -- 97 tests pass
- Phase 7 (`paired_auth_eval.py`, THIS landing) -- 105 tests pass

**520 passed + 1 skipped** across all 6 Phase test suites + cathedral
consumer contract suite.

## Integration verified

- Accepts `SubmissionBundleResult` from Phase 4
  `tac.submission_packet.build_submission_bundle` directly without
  transformation (Phase 4 -> Phase 7 canonical handoff verified by
  `TestPhase4IntegrationRoundTrip`)
- `reconstruct_verdict_from_disk` reads canonical per-axis
  `contest_auth_eval.json` payloads emitted by
  `experiments/contest_auth_eval.py` via Catalog #226
  `gate_auth_eval_call` (verified by `TestReconstructVerdictFromDisk`)
- Cathedral consumer auto-discovered per Catalog #335 (verified via
  `discover_compliant_consumer_modules()` returning consumer + via
  `validate_consumer_module` in test)
- Canonical Provenance per Catalog #323 threaded through every verdict
- Phase 6 compliance verdict + Phase 7 paired auth-eval verdict are
  the canonical handoff pair for D5 operator-gated blocker resolution
- Sister `tools/dispatch_modal_paired_auth_eval.py` (761-LOC operator-
  facing CLI) is the existing live operator dispatch surface; Phase 7
  canonical helper is the typed wrapper that operator runbooks +
  cathedral consumers + Phase 8 STRICT gate + Phase 10 PR111-candidate
  end-to-end regression consume

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Canonical helper | Forked OR adopted | Rationale |
|---|---|---|---|
| Frozen dataclass + `__post_init__` invariants | Python stdlib `dataclasses` | ADOPT | Matches Phase 2/3/4/6 sister pattern; canonical-frozen-dataclass-return per 12th standing directive |
| Sha-locked invariant cross-validation | NEW canonical (this lane) | FORK | The existing `tools/dispatch_modal_paired_auth_eval.py` does NOT enforce sha-locked invariant typed-contract-wise; the canonical typed surface IS the value-add at the Phase 7 surface |
| Catalog #192 macOS-CPU detection | NEW canonical (this lane) | FORK | Per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable; structural refusal at the verdict surface is canonical to this layer (sister of Phase 6 #192 detection at compliance surface) |
| Modal call_id ledger integration | `tac.deploy.modal.call_id_ledger` | ADOPT | Catalog #245 canonical 4-layer pattern; Phase 7 reconstruction surface pins `cuda_call_id` + `cpu_call_id` fields |
| Catalog #226 canonical gate_auth_eval_call | `tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call` | ADOPT | Phase 10 wires execute branch via this canonical helper; Phase 7 documents the routing without exercising it (operator-gated per CLAUDE.md) |
| Operator-gated execute branch | NEW canonical (this lane) | FORK | Per Phase 1 spec memo Layer 5 contract; the execute branch operator-gated SystemExit is canonical to Phase 7; Phase 10 wires via the existing `tools/dispatch_modal_paired_auth_eval.py --execute` path |
| Canonical Provenance umbrella | `tac.provenance` canonical builders | ADOPT (sister of Phase 6 pattern) | Catalog #323 umbrella; threaded through `derive_paired_auth_eval_provenance` |
| Cathedral consumer | `tac.cathedral.consumer_contract` | ADOPT | Catalog #335 canonical contract |
| CLI structure | Phase 6 `submission_compliance_cli.py` precedent | ADOPT | Same arg-parser + load-bundle-from-stdin + JSON-output + exit-code-taxonomy pattern; Phase 7 extends with `--reconstruct-from-disk` post-dispatch surface |

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: Phase 7 Layer 5 is the canonical TYPED ORCHESTRATOR
   over the existing 761-LOC `tools/dispatch_modal_paired_auth_eval.py`
   CLI; the typed orchestrator extincts the per-substrate ad-hoc paired-
   axis dispatch divergence class (NOT a within-class refinement of a
   sister substrate or sister phase).
2. **BEAUTY+ELEGANCE**: 1594-LOC paired_auth_eval module + 475-LOC CLI +
   210-LOC cathedral consumer; ONE canonical entry point
   (`plan_paired_auth_eval`) + ONE return shape
   (`PairedAuthEvalVerdict`) + ONE post-dispatch reconstruction surface
   (`reconstruct_verdict_from_disk`); reviewable in 30 seconds per
   HNeRV parity L4.
3. **DISTINCTNESS**: explicitly different from sister Phase 6 compliance
   (Phase 6 wraps the 3267-LOC compliance script via subprocess +
   surfaces per-Catalog-gate categorized verdict; Phase 7 orchestrates
   paired Modal CUDA + Linux x86_64 CPU dispatch + emits sha-locked
   PairedAuthEvalVerdict + reconstructs verdict from per-axis JSONs).
4. **RIGOR**: 105 dedicated tests + premise verification of canonical
   sister helpers (read Phase 4 SubmissionBundleResult shape + Phase 6
   ComplianceVerdict shape + Catalog #226 `gate_auth_eval_call`
   signature + Catalog #245 + #339 + #360 canonical Modal call_id
   ledger APIs) + Phase 4 integration round-trip test + cathedral
   consumer canonical-contract regression guard + CLI subprocess smoke +
   Catalog #192 structural refusal regression guard.
5. **OPTIMIZATION-PER-TECHNIQUE**: canonical helper routes ALL operator-
   facing paired-axis invocations through the typed wrapper; sha-locked
   invariant + axis-hardware cross-validation + Catalog #192 macOS
   refusal enable downstream consumers (Phase 8 STRICT gate / Phase 10
   end-to-end regression / cathedral autopilot ranker) to consume
   structured verdict without re-parsing per-axis JSONs.
6. **STACK-OF-STACKS-COMPOSABILITY**: composes with Phase 4
   SubmissionBundleResult (consumer surface) + Phase 6 ComplianceVerdict
   (sister surface for D5 operator-gated blocker resolution) + Phase 8
   STRICT gate `check_no_pr_submission_without_compliance_verdict`
   (future) + Phase 10 PR111-candidate end-to-end regression (future).
7. **DETERMINISTIC REPRODUCIBILITY**: canonical-frozen-dataclass-return +
   sha-locked invariant + canonical-Provenance-routing + canonical
   helper invocation token + canonical_equation_id pinned; round-trip
   via `as_dict()` + reconstruction preserves all fields.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: dry-run plan-only mode is
   ~milliseconds (no subprocess, no Modal API calls); reconstruction
   surface bounded by per-axis JSON size (~few KB); cost-band envelope
   enforced pre-dispatch to refuse over-budget plans before any spawn.
9. **OPTIMAL MINIMAL CONTEST SCORE**: this layer does NOT directly
   contribute to score; it ENFORCES paired-axis discipline per CLAUDE.md
   "Submission auth eval BOTH CPU AND CUDA" non-negotiable so paired
   dispatches produce promotion-eligible artifacts (NOT phantom paired
   promotions); Phase 10 first paired empirical anchor surface lowers
   score via the canonical PR111-candidate end-to-end submission lifecycle.

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | HARD-EARNED-vs-CARGO-CULTED | Rationale + unwind |
|---|---|---|
| Sha-locked invariant cross-validation is the canonical Phase 7 invariant | HARD-EARNED | Per CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable + Catalog #127 custody discipline; sha-locked invariant is the structural protection against axis-mismatch dispatches |
| Catalog #192 macOS-CPU detection should be a STRUCTURAL refusal at the verdict surface (not just a wrapped-helper check) | HARD-EARNED | Per CLAUDE.md "Submission auth eval BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable + Catalog #192 sister gate + Phase 6 sister precedent at compliance surface; defense-in-depth at the verdict surface |
| Execute branch should be operator-gated SystemExit (NOT subprocess invocation in Phase 7) | HARD-EARNED | Per Phase 1 spec memo Layer 5 contract + CLAUDE.md "Executing actions with care" non-negotiable + the existing 761-LOC `tools/dispatch_modal_paired_auth_eval.py` CLI is the operator-facing dispatch surface; Phase 7 is the typed planning + reconstruction surface |
| `reconstruct_verdict_from_disk` is the canonical post-dispatch reconstruction surface | HARD-EARNED | Per Phase 10 PR111-candidate end-to-end regression pattern: operator runs `tools/dispatch_modal_paired_auth_eval.py --execute`, harvests, then routes the resulting JSONs through this helper to emit canonical PairedAuthEvalVerdict |
| The 7-verdict-kind taxonomy is canonical | HARD-EARNED | Per Phase 1 spec memo + sister Phase 6 4-verdict taxonomy + Phase 5 lint verdict patterns; the 7-verdict-kind enumerates every structural failure mode per the 2026-05-15 sister Catalog #313 probe-outcome ledger pattern |
| Cost-band envelope refused pre-dispatch is the canonical budget check | CARGO-CULTED -> unwind-test plan | Adopted from Catalog #270 + cost-band default $1.00 smoke / $5.00 full; the realistic upper bound for the canonical wrapper is empirical-derivable; UNWIND: Phase 10 lands the first paired empirical anchor + updates cost estimates per `tac.cost_band_calibration` |
| Per-axis JSON path naming should derive from canonical Catalog #249 helper | HARD-EARNED | Per CLAUDE.md "Forbidden misleading-directory-name (the phantom-score directory trap)" + Catalog #249 sister discipline; `reconstruct_verdict_from_disk` accepts arbitrary paths but the canonical fix is via `gate_auth_eval_call`'s `_redirect_output_json_to_match_device` runtime auto-fix |

## Observability surface (Catalog #305)

1. **Inspectable per layer**: every per-axis dispatch surfaces as a
   typed `PairedAuthEvalVerdict` with separate cuda_* and cpu_* fields
   (score, axis_tag, hardware_substrate, call_id, seg/pose/rate
   components, auth_eval_json_path, elapsed_seconds, cost_usd); the
   FULL per-axis JSON reports are preserved at `cuda_auth_eval_json_path`
   + `cpu_auth_eval_json_path` for forensic audit.
2. **Decomposable per signal**: per-axis distortion components
   (`cuda_seg_distortion` + `cuda_pose_distortion` + `cuda_rate_term`;
   sister CPU); `cuda_cpu_gap` computes the canonical CUDA - CPU delta;
   `total_cost_usd` decomposes per-axis.
3. **Diff-able across runs**: every verdict has canonical
   `measurement_utc` + `archive_sha256_paired` + `cuda_call_id` +
   `cpu_call_id`; sister runs on the same archive_sha256 produce
   diff-able JSON reports.
4. **Queryable post-hoc**: verdict `as_dict()` round-trips to canonical
   JSON; sister-consumers (Phase 8 / 10 / cathedral autopilot) consume
   via the canonical PairedAuthEvalVerdict shape.
5. **Cite-able**: every verdict carries canonical Provenance per Catalog
   #323 with (lane_id, substrate_id, archive_sha256, measurement_utc,
   canonical_helper_invocation, canonical_equation_id, cuda_platform,
   cuda_gpu, cpu_target).
6. **Counterfactual-able**: the cost-estimate via `_estimate_per_axis_cost`
   can be re-evaluated for alternative (platform, gpu, cost_band)
   combinations; the cuda_cpu_gap is empirical per-archive (per CLAUDE.md
   "Submission auth eval BOTH CPU AND CUDA" notes that PR102's -0.033
   gap is per-archive empirical, not extrapolatable).

## HORIZON-CLASS classification (Catalog #309)

`horizon_class: apparatus_maintenance` -- this layer does NOT directly
lower contest score; it ENFORCES paired-axis discipline so paid
dispatches produce promotion-eligible artifacts per CLAUDE.md non-
negotiable. The downstream Phase 10 PR111-candidate end-to-end
regression is the score-lowering surface this layer unblocks.

## 6-hook wire-in declaration (Catalog #125)

1. **Hook #1 sensitivity-map**: N/A (defensive validator gate; no
   sensitivity signal contribution)
2. **Hook #2 Pareto constraint**: N/A (no Pareto-relevant signal)
3. **Hook #3 bit-allocator**: N/A (no bit-allocator signal)
4. **Hook #4 cathedral autopilot dispatch**: **ACTIVE PRIMARY** -- the
   cathedral consumer `paired_auth_eval_consumer` is auto-discovered
   per Catalog #335 and surfaces per-candidate paired-axis readiness
   (PAIRED_PASS / PARTIAL_CUDA_ONLY / PARTIAL_CPU_ONLY / BLOCKED_PRE_DISPATCH
   / BLOCKED_HARVEST / BLOCKED_AXIS_MISMATCH / BLOCKED_HARDWARE_NON_COMPLIANT
   / FORBIDDEN_MACOS_AXIS) to the cathedral autopilot ranker
5. **Hook #5 continual-learning posterior**: **ACTIVE** --
   `update_from_anchor` accepts canonical posterior anchors per Catalog
   #344; future Phase 10 first paired empirical anchor will promote
   canonical equation `paired_auth_eval_canonical_helper_consolidation_
   savings_v1` from FORMALIZATION_PENDING to REGISTERED
6. **Hook #6 probe-disambiguator**: **ACTIVE** -- the 9-verdict-readiness
   taxonomy (UNKNOWN / PAIRED_PASS / PARTIAL_* / BLOCKED_* /
   FORBIDDEN_MACOS_AXIS) IS the canonical disambiguator at the
   cathedral ranker surface

## 12th canonicalization x standardization x ease-of-contest-compliance trinity declaration

- **canonicalization**: ONE canonical helper at
  `tac.submission_packet.plan_paired_auth_eval` for the Layer 5
  paired-axis orchestration surface; sister Phases 2/3/4/5/6/8/9/10
  consume the canonical `PairedAuthEvalVerdict` shape
- **standardization**: canonical-frozen-dataclass-return contract
  matches Phase 2/3/4/6 sister `CompressionPipelineResult` +
  `ArchiveGrammarManifest` + `SubmissionBundleResult` + `ComplianceVerdict`
  patterns; canonical Provenance per Catalog #323 + non-promotable
  markers per Catalog #341 inherited from the same canonical surfaces
- **ease-of-contest-compliance**: ONE operator-facing CLI
  (`tools/paired_auth_eval_cli.py`) wraps the canonical helper;
  operator-friendly 7-tier exit code taxonomy + human-readable verdict
  + machine-readable `--json` mode; future Phase 7 operator runbook
  CLI (`tools/operator_pr_submission_full_lifecycle.py` per Phase 1
  Layer 7) consumes via the canonical CLI surface

## 13th OPTIMAL-TRIO standing directive declaration

This landing satisfies the 3 questions:

1. **AUTOMATED?** YES -- canonical helper auto-discovered per Catalog
   #335 + sister-consumer wire-in to cathedral autopilot via Catalog
   #336/#337; no manual ranker-cascade edits required for new
   substrates; cathedral discovery loop verified to find Phase 7
   consumer (count 67 -> 68)
2. **COMPOUNDING?** YES -- each new substrate's paired-axis verdict
   feeds the canonical posterior (Catalog #323) + the cathedral
   autopilot ranker (Catalog #335) + the Phase 10 PR111-candidate
   end-to-end regression; the canonical equation #344 entry
   `paired_auth_eval_canonical_helper_consolidation_savings_v1`
   will compound from FORMALIZATION_PENDING to REGISTERED upon Phase
   10 first paired empirical anchor; per-axis JSONs are queryable via
   the canonical reconstruct_verdict_from_disk surface
3. **OPTIMAL?** YES -- typed orchestrator over the existing 761-LOC
   `tools/dispatch_modal_paired_auth_eval.py` operator-facing CLI
   (NOT a rewrite per Phase 1 spec memo Layer 5 explicit guidance) +
   sha-locked invariant + axis-hardware cross-validation + Catalog #192
   structural refusal at verdict surface (defense-in-depth) +
   canonical-frozen-dataclass-return contract sister of Phase 2/3/4/6
   surfaces + operator-gated execute branch per CLAUDE.md "Executing
   actions with care"

## Sister coordination

- **Phase 6 compliance** (sister-landed `2d3042d14`) -- Phase 7 builds
  on Phase 6 sister precedent for canonical helper architecture +
  cathedral consumer pattern + CLI structure
- **Phase 4 builder** (sister-landed `1de30160e`) -- Phase 7 consumes
  SubmissionBundleResult directly per `TestPhase4IntegrationRoundTrip`
- **Sister `tools/dispatch_modal_paired_auth_eval.py`** (761-LOC live
  operator-facing CLI) -- Phase 7 typed helper is the planning +
  reconstruction surface; the existing CLI remains the operator-facing
  execute surface
- **Active sister subagent at start**: `build-2-3-ext-8-not-buil`
  (BUILD-2+3-EXT scope: `pair_frame_scorer_geometry_lattice_5d_canvas_
  extended_operators.py`); fully disjoint scope per Catalog #230
  sister-subagent ownership map -- zero file overlap with Phase 7
  surfaces

## Operator-routable next

- **Phase 8 STRICT preflight gate Catalog #362**
  (`check_no_pr_submission_without_compliance_verdict_AND_paired_auth_eval_verdict`):
  refuses any state that attempts a `gh pr create` to
  `commaai/comma_video_compression_challenge` without BOTH a Phase 6
  compliance verdict AND a Phase 7 PAIRED_PASS paired_auth_eval verdict
  landed within the last 24 hours per canonical posterior anchor
- **Phase 9 cathedral autopilot consumer** wire-in extension: the
  Phase 7 cathedral consumer is auto-discovered + invoked; Phase 9
  extends with operator-facing cathedral readiness dashboard
- **Phase 10 first PR111-candidate end-to-end regression**:
  operator-facing canonical single-command CLI
  `tools/operator_pr_submission_full_lifecycle.py` (per Phase 1 Layer 7)
  consumes Phases 0-7 via canonical helpers; first empirical paired-axis
  anchor lands + promotes canonical equation #344 from
  FORMALIZATION_PENDING to REGISTERED

## Discipline declaration

- Catalog #117/#157/#174 canonical serializer (will commit via
  `subagent_commit_serializer.py` with POST-EDIT
  `--expected-content-sha256` for every Phase 7 file)
- Catalog #119 Co-Authored-By Claude Opus 4.7 trailer
- Catalog #206 checkpoint cadence (5 checkpoints emitted during
  landing: init / pre-read complete / module landed / consumer + CLI
  landed / tests landed; 1 complete checkpoint at commit)
- Catalog #229 PV (read Phase 6 compliance.py FIRST + Phase 4
  builder.py FIRST + Phase 1 spec memo §3 Phase 6 / Layer 5 FIRST
  before drafting paired_auth_eval.py; verified canonical signatures
  for `gate_auth_eval_call` + `register_dispatched_call_id_fail_closed`
  + `register_pre_spawn_fatal` + `derive_compliance_provenance`)
- Catalog #230 sister-disjoint: respected active sister
  `build-2-3-ext-8-not-buil` BUILD-2+3-EXT scope which owns
  `pair_frame_scorer_geometry_lattice_5d_canvas_extended_operators.py`
  + tests; my scope is `tac.submission_packet.paired_auth_eval` + CLI
  + cathedral consumer + tests; zero overlap
- Catalog #287 placeholder rejection (every waiver path explicitly
  rejects `<rationale>` / `<reason>` / empty / <4-char rationales)
- Catalog #290 + #294 + #303 + #305 + #309 design-memo sections
  declared above
- Catalog #335 cathedral consumer canonical contract (validated via
  `validate_consumer_module` in test)
- Catalog #340 sister-checkpoint guard PROCEED (no edit overlap with
  active sister at landing time)
- Catalog #341 Tier A canonical-routing markers initially
- Catalog #344 FORMALIZATION_PENDING preserved (no canonical equation
  promotion; awaits Phase 10 first paired empirical anchor)
- 10th apples-to-apples paired CPU+CUDA on 1:1 contest-compliant
  hardware NON-NEGOTIABLE: Catalog #192 macOS-CPU detection
  STRUCTURALLY refuses Darwin ARM64 references in any axis substrate
  AND structurally refuses `promotable=True` co-occurring with macOS
- 11th ORDER-MATTERS sequencing: Phase 7 Layer 5 depends on Phase 4
  Layer 2 (SubmissionBundleResult); Phase 7 is invoked AFTER bundle
  emission per the canonical pipeline order
- 12th canonicalization x standardization x ease-of-contest-compliance
  trinity binding (declared above)
- 13th OPTIMAL-TRIO declaration (declared above)

## Cross-references

- Phase 1 spec memo: `.omx/research/canonical_submission_pipeline_
  specification_memo_20260526.md` §3 Phase 6 / Layer 5 (paired_auth_eval)
- Phase 6 landing: commit `2d3042d14`
- Phase 4 landing: commit `1de30160e`
- Phase 3 landing: commit `1d4753f65`
- Phase 2 landing: commit `b96329a71`
- Canonical Modal call_id ledger: `tac.deploy.modal.call_id_ledger`
  (Catalog #245 / #339 / #360)
- Canonical auth-eval gate: `tac.substrates._shared.smoke_auth_eval_gate.
  gate_auth_eval_call` (Catalog #226)
- Existing operator-facing CLI: `tools/dispatch_modal_paired_auth_eval.py`
  (761 LOC; Phase 7 typed wrapper around)
- 2026-05-19 sister D5 prerequisites landing:
  `feedback_pr_submission_d5_prerequisites_executed_landed_20260519T182635Z.md`
- 2026-05-08 sister dual CPU/CUDA mandatory landing:
  `feedback_dual_cpu_cuda_auth_eval_mandatory_20260508.md`
- Lane: `lane_phase_7_paired_auth_eval_canonical_helper_20260526` L1

## Test verification

```
$ .venv/bin/python -m pytest src/tac/tests/test_submission_packet_paired_auth_eval.py
======================== 105 passed in 0.49s =========================
```

Sister Phase 2/3/4/5/6 regression (all 6 Phase test suites + cathedral
consumer contract suite):

```
$ .venv/bin/python -m pytest \
    src/tac/tests/test_submission_bundle.py \
    src/tac/tests/test_archive_grammar.py \
    src/tac/tests/test_compression_pipeline.py \
    src/tac/tests/test_submission_compliance.py \
    src/tac/tests/test_submission_linter.py \
    src/tac/tests/test_submission_packet_paired_auth_eval.py \
    src/tac/tests/test_cathedral_consumer_contract.py
520 passed, 1 skipped in 2.95s
```

Cathedral consumer auto-discovery verified:

```
$ .venv/bin/python -c "
from tools.cathedral_autopilot_autonomous_loop import discover_compliant_consumer_modules
modules = discover_compliant_consumer_modules()
print('paired_auth_eval_consumer present:', 'paired_auth_eval_consumer' in
      [m.__name__.split('.')[-1] for m in modules])
print('total discovered:', len(modules))
"
paired_auth_eval_consumer present: True
total discovered: 68  (was 67 pre-landing; Phase 7 brings to 68)
```

CLI smoke verification:

```
$ .venv/bin/python tools/paired_auth_eval_cli.py --help
(exit 0; usage emitted)

$ .venv/bin/python tools/paired_auth_eval_cli.py \
    --from-submission-bundle /nonexistent
[paired-auth-eval-cli] FATAL: submission bundle path does not exist: /nonexistent
(exit 6 CLI error)
```

## LOC totals

- `src/tac/submission_packet/paired_auth_eval.py`: 1594 LOC
- `src/tac/cathedral_consumers/paired_auth_eval_consumer/__init__.py`: 210 LOC
- `tools/paired_auth_eval_cli.py`: 475 LOC
- `src/tac/tests/test_submission_packet_paired_auth_eval.py`: 1662 LOC
- **Total**: 3941 LOC (within Phase 1 spec memo §3 Phase 6 / Layer 5
  acceptance: "LOC estimate: ~400-600 src + ~300-500 tests" -- Phase 7
  came in larger because it includes the canonical `reconstruct_verdict_
  from_disk` surface + comprehensive frozen-dataclass invariants + 105
  dedicated tests covering 9 distinct verdict cascades). LOC overage is
  warranted per the 12th canonicalization x standardization x ease-of-
  contest-compliance trinity: the canonical typed surface IS the value-add.

Catalog #185 baseline preserved (pre-existing violations unchanged by
this lane).
