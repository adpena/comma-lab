# L5-v2 Next 12 Months: Foreseeable Failures And Preventions

Date: 2026-05-17
Worker: sidecar foresight subagent
Owned file: `.omx/research/l5_v2_next_12_months_foreseeable_failures_20260517_subagent.md`
Status: `research_only=true`
Score claim: `false`
Promotion eligible: `false`
Ready for exact eval dispatch: `false`
Provider dispatch attempted: `false`

## Scope And Source State

This memo anticipates the failures we will be annoyed by over the next twelve
months if we do not act now. Scope: L5/L5-v2 staircase, TT5L, Rule #6,
non-HNeRV/frontier score-lowering, contest compliance, production-hardened OSS,
and scientific/math rigor.

Required pre-read completed before writing:

- `AGENTS.md`
- `CLAUDE.md`
- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- parent-level `.omx/notepad.md`
- parent-level `.omx/release_manifest_v0.2.0-rc1.md`
- `.omx/state/lane_registry.json` searched/read for L5/TT5L/Rule #6 lanes
- no `.omx/research/*_directive_*` files were found within the last 24 hours

Current high-friction facts this memo assumes:

- Best known anchors are axis-separated: CPU `0.1920513168811056`
  `[contest-CPU; GHA Linux x86_64 1:1]`, CUDA `0.20533002902019143`
  `[contest-CUDA T4]`; A1 is the Rule #6 control substrate, not the absolute
  best current axis floor.
- Rule #6 is the immediate frontier-breaking path: Balle hyperprior on A1,
  PR101-style entropy stack on A1, and VQ-codebook on A1.
- L5-v2/TT5L remains the parallel asymptotic campaign. Architecture lock is
  currently forbidden.
- TT5L has a complete 5 variant x 2 axis target map, but harvested exact eval
  artifacts are `0/10`.
- Modal is blocked by `modal_workspace_billing_cycle_spend_limit_reached`;
  Lightning is artifact-valid at the dry-run/route layer but still blocked on
  identity, teamspace, machine inventory, source manifest staging, remote CUDA
  runtime probe, quota/credit check, and lane claims.
- A1 byte-only current runtime is saturated: latent raw-LZMA best delta is `0`
  bytes, sidecar usable runtime-supported delta is `0` bytes.
- Existing Z3HV2 is a direct-residual control, not a live Balle entropy-coded
  residual implementation.

## Top 12 Foreseeable Failures

### 1. Architecture-lock false authority returns under a new label

Twelve-month failure:

The lock surface eventually says "L5-v2 architecture locked" because a route
packet, dry-run plan, diagnostic pair, or planning packet is green, while the
actual required side-info effect curve and C1/Z5/TT5L paired probe gate remain
missing. The next year is then spent optimizing a locked staircase that was
never proven causal.

Why this is foreseeable now:

- The current architecture-lock packet is correctly `false`, but it already has
  many individually green sub-artifacts. A future agent can mistake local
  green checks for global authority.
- TT5L has `10/10` target cells but `0/10` harvested exact-eval artifacts.

Detection tripwires:

- Any memo, briefing, or queue row uses `architecture lock`, `locked`,
  `frontier`, `auto-promote`, or `dispatch-ready` while
  `architecture_lock_allowed=false`.
- Side-info effect curve has `observed_cell_count=10` but any score is `None`.
- Probe gate reports any required candidate ineligible:
  `c1_world_model_foveation`, `z5_predictive_coding_world_model`, or
  `time_traveler_l5_autonomy`.

Prevention and recovery:

- Keep one shared architecture-lock predicate and make all operator surfaces
  print the same blocker list from
  `.omx/research/l5_v2_architecture_lock_packet_20260516_codex.{json,md}`.
- Add a release/report hygiene rule: no "locked" wording unless the lock packet
  SHA is cited and `architecture_lock_allowed=true`.
- Recovery if this trips: immediately regenerate the lock packet, write a
  supersession note to the stale memo, and mark every downstream result
  `indeterminate_lock_authority` until rerouted through the packet.

Impact: protects custody first; directly lowers score indirectly by preventing
months of optimization on the wrong L5 branch.

### 2. TT5L effect curve never executes; we keep refreshing custody instead

Twelve-month failure:

The TT5L side-info effect curve remains the "next concrete action" for months.
Agents repeatedly refresh route packets, source manifests, dry-run bundles, and
architecture-lock packets, but no paired CPU/CUDA cell ever lands.

Why this is foreseeable now:

- Modal has a provider-level billing blocker.
- Lightning has a valid route/doctor plan but still needs identity, teamspace,
  SSH target, machine inventory, source manifest staging, CUDA runtime probe,
  quota check, and active claims.
- Existing harvested exact-eval cells are `0/10`.

Detection tripwires:

- `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_20260517_codex.md`
  still shows `Ready cells: 0/10` after seven days.
- Required doctor plan exists, but
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_20260517_codex.json`
  does not exist or has no `status=OK`.
- Modal blocker remains active and no alternate-provider terminal artifact
  appears.

Prevention and recovery:

- Convert the provider unblock into a dated two-branch operator packet:
  `modal_billing_resolved` or `lightning_route_ready`. Each branch must name
  the exact next shell command, expected JSON output, pass predicates, and
  terminal claim update.
- If no provider path is executable within seven days, record
  `tt5l_effect_curve_provider_blocked_stale` and pivot one worker to a local
  non-promotional score-response control that can change the next design while
  provider access is fixed.
- Do not allow more than one custody-refresh memo between provider actions.

Impact: directly score-lowering, because TT5L cannot graduate or be redesigned
without the side-info effect curve.

### 3. TT5L controls are not causal enough to answer the side-info question

Twelve-month failure:

The ten exact cells finally run, but the variants differ in archive bytes,
runtime content, side-info sparsity, or source tree provenance in a way that
makes the trained-vs-control conclusion ambiguous. We get a table of scores but
still cannot say whether TT5L side information helps.

Why this is foreseeable now:

- Current materialized variants differ substantially in bytes: `zero` is about
  34 KB while `trained` is about 43 KB.
- Prior L5 hardening already found runtime-content-axis mismatch and stale
  pair-anchor risks.
- Existing effect-curve blockers can say `trained_not_best_or_tied` even with
  missing score values.

Detection tripwires:

- Any effect-curve result lacks component deltas split into SegNet, PoseNet,
  and rate term.
- CPU and CUDA cells have different runtime content hashes for the same
  variant.
- `trained` loses only through rate while improving scorer terms, but the
  memo calls it a method negative.
- Full-frame inflated raw-output aggregate SHA is missing for any cell that
  influences architecture lock.

Prevention and recovery:

- Before interpreting the curve, emit a normalized effect review:
  `delta_total`, `delta_seg`, `delta_pose_sqrt_term`, `delta_rate`, and
  break-even bytes for every trained-vs-control pair on both axes.
- Require raw-output aggregate SHA and runtime-content equality across axes.
- If byte size dominates, classify as
  `SCORER_RESPONSE_PRESENT_RATE_NEGATIVE` or `RATE_ONLY_*`, not "TT5L failed."
- If control mismatch appears, re-run the smallest matched control pair before
  any architecture-lock decision.

Impact: protects scientific rigor and directly lowers score by preserving a
valid TT5L redesign path instead of discarding causal signal.

### 4. Rule #6 relapses into no-op bytes or mislabeled hyperprior work

Twelve-month failure:

We ship another "Balle / hyperprior / entropy" artifact that does not actually
consume conditional side information at inflate time, or that changes bytes
without changing decoded frames. It gets a spend cycle, then an adversarial
review discovers it was a direct residual or byte-only no-op.

Why this is foreseeable now:

- A1 current byte-only grammar is saturated at `0` byte gain.
- Z3HV2 was labeled Balle-adjacent but profiled as a direct-residual control
  with empty hyperprior slots and a byte regression against the replaced A1
  latent section.
- The scorer-response probe was just added because byte liveness alone was not
  enough.

Detection tripwires:

- `hyperprior_weights_int8` or equivalent side-info sections are empty.
- Mutating claimed side-info bytes does not change inflated frames.
- A profile says the replacement section is larger than the original A1
  section before scorer effects.
- The only measured improvement is rate-only while the memo attributes it to
  scorer-visible conditioning.

Prevention and recovery:

- Make `profile_z3v2_payload_contract`-style payload authority profiling
  mandatory for every Rule #6 candidate before exact-eval dispatch.
- Require a byte-mutation smoke that proves the side-info bytes are consumed by
  inflate and a score-response probe that separates component movement from
  rate-only movement.
- If a candidate fails, preserve it as a measured implementation negative and
  route the next build to a true conditional entropy decoder, context coder, or
  VQ/codebook grammar that beats a concrete A1 section.

Impact: directly score-lowering; also protects custody by blocking false
authority.

### 5. Rule #6 stops being a bolt-on and becomes another kitchen-sink substrate

Twelve-month failure:

The winning PR101 lesson gets inverted. "A1 + Balle" or "A1 + VQ" grows into a
large research substrate with a heavy trainer, broad dependency closure, no
reviewable runtime, and no small byte-closed packet. By the time it can be
reviewed, the simple 337-LOC-style opportunity is gone.

Why this is foreseeable now:

- The T4 symposium defines Rule #6 as a verified PR95-paradigm substrate plus a
  <=350 LOC, 30-second-reviewable entropy/distillation bolt-on.
- Existing substrate work routinely exceeds that size and becomes
  `lane_class=substrate_engineering`.

Detection tripwires:

- A Rule #6 "bolt-on" has more than 350 LOC of runtime/codec surface without a
  substrate-engineering tag.
- The training script exists before archive grammar, parser section manifest,
  export contract, and no-op proof.
- `inflate.py` needs more than two runtime dependencies or a non-reviewable
  dependency closure.

Prevention and recovery:

- Split each Rule #6 idea into two artifacts: a minimal byte-closed packet
  prototype and an optional substrate-engineering follow-up.
- The prototype must name the exact A1 section it replaces or extends, expected
  byte delta, consumed-byte proof, and same-axis control.
- If the minimal prototype cannot exist, reclassify as `frontier_pursuit` or
  `asymptotic_pursuit`; do not keep it in the immediate Rule #6 queue.

Impact: directly score-lowering by preserving fast frontier attempts; protects
OSS reviewability.

### 6. Axis drift silently misranks the next frontier

Twelve-month failure:

A candidate is chosen, retired, or submitted because a CPU number was compared
to a CUDA baseline, or because macOS/proxy evidence was promoted. The public
leaderboard axis and internal CUDA axis diverge again, and the repo spends
weeks optimizing the wrong target.

Why this is foreseeable now:

- Current best CPU and CUDA anchors are different archives and lanes.
- AGENTS/CLAUDE repeatedly warn that `[contest-CPU]`, `[contest-CUDA]`, and
  `[macOS-CPU advisory]` are separate evidence spaces.
- A recent permanent fix was needed because frontier citations went stale.

Detection tripwires:

- Any report says `frontier`, `medal-band`, `submission-ready`, `score gap`, or
  `rank` without an inline axis label.
- A candidate has only one of paired CPU/CUDA exact artifacts but is used for
  promotion, rank, or retirement.
- `reports/latest.md`, `.omx/state/current_focus.md`, and scanner-derived
  anchors disagree by more than `1e-6`.

Prevention and recovery:

- Every score table should have explicit CPU and CUDA rows, even if the row is
  `missing_required_artifact`.
- Run the frontier scanner before any submission, architecture-lock, or
  public-facing release memo.
- If axis drift is found, write a supersession ledger and reset the candidate's
  status to `axis_indeterminate` until paired exact custody is restored.

Impact: protects custody and directly improves score selection.

### 7. Non-HNeRV high-risk lanes either crowd out P0 or get frozen forever

Twelve-month failure:

The system oscillates between two bad modes: dispatching the 35-substrate
high-risk cluster before scorer-awareness evidence, or over-correcting and
never touching non-HNeRV class-shift ideas again. Either way, hidden
score-lowering routes are lost.

Why this is foreseeable now:

- Current focus defers the high-risk cluster pending scorer-awareness probes.
- The lane registry already contains new hardware, freezing, pausing, and
  problem-space exploit waves.
- Rule #6 and TT5L both need focused execution, but not at the cost of
  suppressing every non-HNeRV exploration.

Detection tripwires:

- A high-risk substrate gets paid dispatch without a scorer-awareness or
  score-response artifact.
- `next_experiments.md` loses Rule #6 or TT5L P0 ordering without a harvested
  result or explicit blocker.
- More than two consecutive frontier sessions produce no candidate archive,
  exact eval, scorer-response probe, or provider terminal artifact.

Prevention and recovery:

- Maintain an explicit portfolio split until the next harvest:
  Rule #6 packet builds, TT5L provider/effect-curve execution, and a bounded
  non-HNeRV scorer-response/probe lane.
- Every non-HNeRV lane must declare whether its next artifact is a candidate
  archive, scorer-response probe, byte-consumption proof, or exact blocker.
- If a non-HNeRV lane cannot name one of those artifacts, move it out of the
  active frontier queue.

Impact: directly score-lowering; protects focus from both over-dispatch and
over-deferral.

### 8. SCORER-AWARENESS probes become arbitrary proxy theater

Twelve-month failure:

The MI/attention/argmax probes become the new council ritual: every substrate
gets a number, but the number is not calibrated against exact score movement.
Good ideas are deferred because MI is low; bad ideas proceed because MI is
high in the wrong place.

Why this is foreseeable now:

- The T4 symposium sets MI thresholds, but the probe class still needs
  calibration against known positives and negatives.
- The scorer-response probe exists specifically because liveness and
  distinguishing-feature integration were insufficient.

Detection tripwires:

- A probe result changes dispatch priority without naming a matched exact
  baseline/candidate pair or calibration set.
- The probe uses one pair or one frame region and generalizes to a method
  verdict.
- MI threshold values appear without a calibration artifact.

Prevention and recovery:

- Build a calibration ledger with known anchors: A1 control, Z3HV2 direct
  residual negative, PR106 format0D forensic control, TT5L diagnostic failure,
  and the first Rule #6 candidate with exact paired evidence.
- Treat MI as a routing prior only. Require a score-response ablation before
  method-level proceed/defer language.
- If a probe later contradicts exact eval, update the calibration ledger and
  downgrade the probe family until recalibrated.

Impact: scientific/math rigor first; directly score-lowering by making probe
budget allocation trustworthy.

### 9. Provider/runtime drift creates unreproducible exact-eval claims

Twelve-month failure:

Modal, Lightning, Vast.ai, and local/GHA paths diverge. A candidate works on one
provider because of hidden packages, local paths, CUDA/NVDEC behavior, or stale
mounted code, then fails or changes score on contest-compliant replay.

Why this is foreseeable now:

- TT5L provider route still requires remote CUDA runtime probe and source
  manifest staging.
- Provider architecture rules require reusable deploy modules, not
  provider-specific package lists inside lane scripts.
- Past work repeatedly found missing imports, stale source manifests, and
  runtime-tree mismatches.

Detection tripwires:

- A lane-local script starts accumulating package lists, mount rules, import
  probes, source-manifest logic, or cost tables.
- `report.txt`, runtime tree SHA, source manifest, or import-probe output is
  missing from a dispatch packet.
- A provider row advertises exact-CUDA support without full lifecycle, harvest,
  terminal claim, and adjudication paths.

Prevention and recovery:

- Keep provider route readiness in `src/tac/deploy/<provider>/` or canonical
  tools; lane scripts should be thin adapters.
- Require doctor/import probe/source manifest artifacts before every
  non-dry-run dispatch.
- On mismatch, classify as `provider_runtime_drift`, not method evidence, and
  rerun source-matched replay before promotion or retirement.

Impact: protects contest compliance and reproducibility; directly preserves
score claims that would otherwise be invalid.

### 10. OSS release surface leaks private state or stale scientific claims

Twelve-month failure:

The project becomes public-facing while release manifests, docs, examples, or
package metadata still contain stale external scores, private `.omx/state`
assumptions, provider details, unresolved licensing, or unreproducible build
instructions. The OSS release then becomes a cleanup emergency instead of a
credible production-hardened artifact.

Why this is foreseeable now:

- The v0.2.0-rc1 manifest is careful but still carries operator-decision items:
  license copyright string, repository metadata, SBOM, THIRD_PARTY_NOTICES, and
  external score replay.
- Production-hardened OSS is a standing constituency, not a post-contest task.

Detection tripwires:

- Public docs mention unpaired or external scores without `external` or axis
  labels.
- Release notes cite raw `.omx/state/*.json` or provider transcripts as
  durable evidence.
- Package metadata points at upstream contest repo instead of the canonical OSS
  repo, or dependencies lack SBOM/notice coverage.

Prevention and recovery:

- Add a release-candidate hygiene checklist to every public packet: score-axis
  labels, external-score disclaimer, no raw provider transcripts, no private
  URLs, no credentials, reproducible build commands, SBOM, and license notice
  status.
- Keep contest-specific reverse engineering in `reverse_engineering/` and
  reusable codec/runtime APIs in `tac`; do not mix operator state into public
  library surfaces.
- If a stale claim ships, publish a supersession note and regenerate the
  release manifest before further promotion.

Impact: protects custody, OSS credibility, and scientific reproducibility.

### 11. Negative results become method kills or positives become story wins

Twelve-month failure:

A bad exact eval kills an entire method family, or a proxy/local win becomes a
strategy headline. The repo loses valuable trust-region data and keeps
incorrect priors because result review is weaker than result generation.

Why this is foreseeable now:

- TT5L diagnostic exact eval is a measured-config failure, not a lane kill.
- Z3HV2 is a direct-residual implementation negative, not a Balle theorem.
- Z6 fixed an initialization bug and remains proxy-only; it is not a paid
  dispatch result.

Detection tripwires:

- A memo says `killed`, `falsified`, `frontier`, or `winner` without custody,
  recomputed formula, failure class, and reactivation criteria.
- A result lacks composition review: additive, antagonistic, orthogonal, or
  redundant with current champion.
- A bad result does not update a trust region, guard, posterior, or next
  design blocker.

Prevention and recovery:

- Require the existing result-review packet for every exact positive,
  negative, byte-only result, and surprising proxy.
- Default status should be `measured-implementation retired` or
  `measured-config failure` unless independent exact evidence or a mathematical
  impossibility proof supports broader language.
- Write reactivation criteria as executable artifacts: archive grammar change,
  scorer-response proof, paired exact eval, or byte-consumption proof.

Impact: scientific rigor and score-lowering; prevents throwing away reusable
signal.

### 12. Math/planner language outruns byte-closed artifacts

Twelve-month failure:

Dykstra, ADMM, lattice, field-equation, and meta-Lagrangian language keeps
expanding, but too many rows never emit a deterministic archive, dispatch
packet, or learnable feedback row. The solver becomes a prestige index instead
of a compiler for contest archives.

Why this is foreseeable now:

- The L5-v2 staircase explicitly relies on Dykstra-style feasibility, but
  prediction bands and architecture lock remain fail-closed.
- Current instructions warn never to invent flags/schema keys and to keep
  side information charged inside the archive.
- Many planning artifacts are useful but non-promotional.

Detection tripwires:

- A planner row has no archive-builder consumer, no dispatch command, no
  `planning_only=true`, or no exact blocker.
- A new knob, threshold, codebook size, radius, loss weight, foveation field,
  or selector enters a builder without component-response support, ablation
  manifest, or mathematical feasibility bound.
- Dykstra/ADMM wording is used as proof of score movement rather than a
  constraint-management discipline.

Prevention and recovery:

- Every L5/Rule #6/non-HNeRV planning row should name one of:
  `candidate_archive_path`, `dispatch_plan_path`, `scorer_response_probe_path`,
  `byte_consumption_proof_path`, or `exact_blocker_path`.
- If none exists, keep the row `planning_only=true` and out of operator
  dispatch surfaces.
- Add quarterly "math-to-bytes" audits: list top planner rows with no concrete
  artifact and either assign a builder/probe or archive them with rationale.

Impact: protects mathematical rigor and directly lowers score by forcing the
solver to emit testable candidates.

## Score-Lowering Versus Custody Matrix

| # | Failure short name | Directly lowers score? | Protects custody/compliance? | Primary next artifact |
|---:|---|---|---|---|
| 1 | False architecture lock | Indirect | Yes | refreshed shared lock predicate + supersession guard |
| 2 | TT5L never executes | Yes | Yes | provider unblock packet + doctor/claim/harvest sequence |
| 3 | Non-causal TT5L controls | Yes | Yes | normalized 5x2 effect review with raw-output SHAs |
| 4 | Rule #6 no-op/mislabel | Yes | Yes | payload authority profile + mutation + score-response probe |
| 5 | Rule #6 kitchen sink | Yes | Yes | minimal A1 byte-closed prototype |
| 6 | Axis drift misranking | Yes | Yes | dual-axis exact table + frontier scanner receipt |
| 7 | Non-HNeRV crowd/freeze | Yes | Partial | portfolio queue with artifact-per-lane requirement |
| 8 | Proxy probe theater | Yes | Yes | calibrated scorer-awareness ledger |
| 9 | Provider/runtime drift | Indirect | Yes | provider doctor/import/source-manifest artifacts |
| 10 | OSS release leakage/stale claims | Indirect | Yes | release hygiene checklist + SBOM/notice status |
| 11 | Result overclaim/overkill | Yes | Yes | result-review packet with reactivation criteria |
| 12 | Math outruns bytes | Yes | Partial | math-to-bytes audit and artifact linkage field |

## Highest-EV Recommendations

1. Execute TT5L provider unblocking as an operator packet, not another memo:
   doctor -> source manifests -> claims -> 10 paired cells -> harvest -> effect
   curve -> architecture-lock refresh.
2. Put every Rule #6 candidate through a three-proof gate before spend:
   payload profile, consumed-byte mutation, and score-response classification.
3. Make every frontier report dual-axis by construction. Missing paired axis is
   a row, not an inference.
4. Calibrate scorer-awareness probes against known exact positives/negatives
   before using them to defer or elevate a substrate class.
5. Keep non-HNeRV exploration alive only through concrete artifacts: candidate
   archives, scorer-response probes, byte-consumption proofs, or exact blockers.
6. Add a quarterly math-to-bytes audit so Dykstra/lattice/planner work stays
   coupled to deterministic archive builders and exact-eval routes.

## Non-Goals

- No code changes were made here.
- No provider work was launched.
- No score, promotion, rank, kill, or architecture-lock authority is claimed.
- Partner WIP files were left untouched.
