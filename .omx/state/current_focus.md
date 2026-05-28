# Current Focus - 2026-05-22 (MLX portable substrate + L5 v2 + Rule #6)

## Frontier

- Canonical scanner-derived best CPU anchor:
  `0.1920089730474962`
  `[contest-CPU; Linux x86_64 1:1]`, archive
  `18e3155fbbbe9ab23e1c21bc0d99ba8d18657a71c3129fc5ff9e0405b67d1669`,
  lane `feca_selector_reparam_scale64_alpha1_stable_cpu_exact`.
  Refresh from `reports/latest.md` and
  `.omx/state/canonical_frontier_pointer.json`; this file is a mirror, not a
  frontier source of truth.
- Canonical scanner-derived best CUDA anchor:
  `0.20533002902019143`
  `[contest-CUDA T4]`, archive
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`,
  lane `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`.
- A1 remains the Rule #6 control substrate, not the best current axis floor:
  `0.19284757743677347` `[contest-CPU; GHA Linux x86_64 1:1]` and
  `0.2263520234784395` `[contest-CUDA T4]`.
- A1 archive bytes/SHA-256:
  `178262` /
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- Public medal band remains the immediate score target:
  PR101/PR102/PR103 around `0.193`/`0.195`/`0.195`, axis-specific and
  external until exact replay custody proves otherwise.

## 2026-05-22 MLX Portable-Local-Substrate State

- MLX is now a calibrated local research/spend-triage substrate, not score
  authority. Current allowed axis tag is `[macOS-MLX research-signal]`; exact
  CPU/CUDA auth eval remains mandatory before any score, rank/kill, promotion,
  or submission claim.
- MLX score calibration is strict-pass for the current public-frontier set:
  `6/6` pairwise decisions certified, `0/6` uncertain, and minimum MLX gap for
  spend triage `8.801772121230789e-05`.
- MLX transfer calibration now also requires the auth-side payload to pass the
  strict contest-auth-axis contract in `tac.auth_eval_schema`: only full-sample
  `contest-CPU` / `contest_cpu` or `contest-CUDA` / `contest_cuda` payloads
  qualify. Advisory/proxy/local diagnostic payloads are historical or debug
  evidence only.
- The decoder-q surface-guided batch is blocked for exact-eval dispatch:
  candidates `a2f90a216aac4184`, `a9b04920db67ec71`, and
  `8f3a33e49b9b7906` all regressed on `[macOS-CPU advisory decoder-q]`.
- Current first MLX follow-up is sign calibration:
  `ll_decoder_q_surface_sign_calibration_repair`. Use the advisory-negative
  batch as labeled response data; do not send the three surface-guided
  candidates to exact CUDA.
- DQS1 pairset rank021/pair0371 exact Modal CPU recovery is now the CPU
  frontier: `0.19202828295713675 [contest-CPU]`. DQS1 top32 compact
  `sorted_gap_uleb` is superseded on CPU at
  `0.19202894881608987 [contest-CPU]`; its paired CUDA result
  `0.22619043540195719 [contest-CUDA T4]` is not a CUDA frontier.
- Next local-substrate portability work is full-weight PyTorch/NumPy/MLX trace
  parity plus CPU-stable response harvest expansion. MLX GPU/batch behavior
  remains research-only until batch invariance and transfer checks pass.

## Active Strategic Rebaseline

The May 17 T4 symposium supersedes the stale May 15 queue framing without
retiring the L5/L5-v2 staircase:

1. **Immediate frontier-breaking path**: Rule #6 bolt-ons on verified A1.
   Build small, byte-closed, PR101-style additions on the working A1 substrate
   before spending another wave on high-risk substrate-class guesses.
2. **L5 v2 / TT5L priority remains active**: TT5L side-info effect curve,
   L5-v2 probe gates, and architecture-lock custody remain the primary
   asymptotic campaign and must keep moving in parallel.
3. **High-risk substrate cluster**: 35-substrate per-pair-conditioning cluster
   is deferred pending SCORER-AWARENESS probes, not killed.
4. **Original Z6 FiLM path**: do not dispatch as-is; replace with
   per-frame-renderer-axis ego-motion conditioning.
5. **PR106/HNeRV local-basin work**: useful as forensic control and byte
   lessons only; do not let it crowd out Rule #6 or L5-v2 actuation. The
   2026-05-17 format0D WIP adversarial review classifies format0D as a local
   best `[contest-CUDA T4]` forensic/control anchor, not a public
   `[contest-CPU]` frontier or submission candidate.

## Current L5 v2 / TT5L State

- TT5L paired diagnostic exact eval is terminal and non-promotional:
  `3.8987840060549908` `[contest-CPU]` and
  `3.9007398365396795` `[contest-CUDA]` for archive
  `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`.
- Architecture lock remains forbidden. Current blocker class:
  missing complete L5-v2 gate evidence, missing C1/Z5/TT5L probe gate evidence,
  and missing paired CPU/CUDA side-info effect curve harvest.
- TT5L side-info Modal paired dispatch plan now consumes the shared exact
  dispatch authority gate and is unblocked at custody level:
  `ready_work_unit_count=5`, per-variant `dispatch_blockers=[]`, source runtime
  `report.txt` materialized, and all five adjacent `archive_manifest.json`
  files materialized. Ledger:
  `.omx/research/l5_v2_tt5l_dispatch_custody_materialization_20260517_codex.md`.
- TT5L side-info Lightning paired-axis plan has 10 cells, execution preflight
  has `ready_cell_count=10`, execution bundle has
  `ready_dry_run_cell_count=10`, dry-run verifier has `10/10` passing cells,
  route packet has `artifact_blocker_count=0`, and required doctor plan is
  `ready_for_operator_doctor=true`. The route and doctor packets were
  refreshed from a source-relevant tree with no paired-axis-plan source-path
  drift. Do not use the recorded generated-at commit as self-referential
  dispatch authority; rerun the route/doctor packet immediately before
  non-dry-run execution. Non-dry-run provider execution remains blocked on
  Lightning identity/quota, `LIGHTNING_TEAMSPACE`, `LIGHTNING_SSH_TARGET`,
  per-cell source manifests, active lane claims, remote CUDA/machine-inventory
  doctor checks, exact harvest, terminal claim rows, and architecture-lock
  packet refresh after harvest.
- Lightning required-doctor plan exists at
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`;
  it is planning-only and confers no dispatch or score authority. The route
  packet no longer hashes architecture-lock as an upstream source, so the
  route -> doctor -> architecture chain has no circular custody dependency.

## Active P0 Work

1. **Rule #6 A1 bolt-on #1**: Ballé-2018 hyperprior on A1 per-pair latent,
   with KL-on-logits `T=2.0` distillation from frozen A1 teacher.
   Existing Z3HV2 direct-residual export is not this implementation: it is now
   classified as a byte-negative direct-residual control with no active Ballé
   entropy residual decoder. See
   `.omx/research/rule6_z3v2_direct_residual_unwind_20260517_codex.md`.
2. **Rule #6 A1 bolt-on #2**: PR101-style per-tensor byte map plus
   Brotli/LZMA/Huffman sidecar on A1 weights/latents.
   Current A1 byte-escape profiler is saturated under the existing runtime:
   raw-LZMA latent sweep best equals source at `15387` bytes, current
   607-byte sidecar has only a 4-byte oracle entropy gap but no smaller
   runtime-supported representation for current semantics, and no candidate
   archive was emitted. Ledger:
   `.omx/research/a1_rule6_byte_escape_profile_20260517_codex.md`.
3. **Rule #6 A1 bolt-on #3**: VQ-codebook on A1 per-pair latent.
4. **TT5L side-info effect curve**: custody unblocker is complete. Next run
   the Lightning doctor, stage per-cell source manifests, claim each lane, and
   execute the 10 paired CPU/CUDA cells only if doctor and source-manifest
   custody are green.
5. **SCORER-AWARENESS probe wave**: measure whether substrate distinguishing
   features reach scorer attention/argmax maps before deferring high-risk
   per-pair-conditioning substrates.
6. **Z6 replacement design**: per-frame-renderer-axis ego-motion variant,
   not FiLM-bottlenecked Z6.

## Latest WIP Review

- `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md`
  is untracked partner WIP and was left unmodified.
- `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md`
  is also untracked partner WIP and was left unmodified. Its executive
  summary reinforces the paired CPU/CUDA hardware-axis split and scorer
  decomposition, but it is not a current score, dispatch, or promotion
  authority until landed and reviewed.
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
  is untracked partner WIP and was left unmodified. Its L5 row inherits the
  stale `-0.008` to `-0.015` rate-only expectation; current routing should use
  the L5 rate-only bound review instead.
- Master-gradient raw archive-byte finite differences are now blocked as an
  invalid probe grain for ZIP/entropy-coded packets. Current routing should
  use `.omx/research/master_gradient_raw_byte_finite_difference_adversarial_review_20260517_codex.md`
  and `tac.master_gradient_feasibility`: replace `(N_archive_bytes, 3)` bit/byte
  flip probes with `(N_valid_mutation_operators, 3)` packet-valid score-response
  rows that rebuild ZIP metadata/CRC and prove inflate success.
- Master-gradient operator-response planning surface is now materialized in
  `tac.master_gradient_operator_plan` and
  `tools/build_master_gradient_operator_plan.py`. Ledger:
  `.omx/research/master_gradient_operator_response_plan_landed_20260517_codex.md`.
  This supersedes raw-byte/autograd-per-archive-byte wording in untracked WIP
  campaign notes without editing partner WIP: the only valid next object is a
  grammar-aware operator-row manifest, still `score_claim=false` and
  dispatch-ineligible until packet closure proofs exist.
- Master-gradient dirty WIP adversarial review:
  `.omx/research/master_gradient_partner_wip_false_authority_review_20260517_codex.md`.
  The untracked `src/tac/master_gradient.py` and dirty cathedral-autopilot
  hook should not land as authority-bearing code while they expose an
  `(N_archive_bytes, 3)` tensor, `finite_difference_bit_flip` method naming,
  raw `{byte_idx: delta}` projection API, notes-parsed archive SHA lookup, and
  a "rerank" hook that currently leaves ordering unchanged. Valid routing
  remains `CandidateModificationSpec` / `grammar_aware_operator` response rows
  with packet rebuild, inflate proof, byte-consumption proof, and exact
  axis-labelled result review.
- First executable master-gradient operator row is now materialized for
  PR106-format Brotli sections:
  `tac.master_gradient_brotli_operator_candidate` and
  `tools/build_master_gradient_brotli_operator_candidate.py`. Ledger:
  `.omx/research/master_gradient_brotli_operator_candidate_landed_20260517_codex.md`.
  Local candidate signal: `decoder_packed_brotli` recompression on public PR106
  reduced archive bytes by `151` with CRC/header/parser/Brotli closure proven,
  but runtime inflate proof and byte-consumption proof are still missing, so
  the row remains non-dispatchable and non-promotional.
- Reviewed disposition:
  `.omx/research/pr106_format0d_wip_adversarial_review_20260517_codex.md`.
- FEC6 CPU frontier submission-surface adversarial review:
  `.omx/research/fec6_cpu_frontier_submission_surface_adversarial_review_20260517_codex.md`.
  Verdict at the 2026-05-17 review time: FEC6 remained the best
  `[contest-CPU]` anchor, but the WIP packet was not submission-ready. FEC6 is
  now superseded as best CPU anchor by the 2026-05-22 compact DQS1 gap-ULEB
  exact CPU result; the submission-surface blockers below remain relevant for
  any FEC6-derived packet. Strict compliance output is
  preserved at
  `experiments/results/fec6_cpu_submission_surface_review_20260517_codex/pre_submission_compliance_cpu.json`
  with `passed=false`; blockers include missing `archive.zip`/`report.txt` in
  `submission_dir`, incomplete member manifest, missing public packet README
  axis/source/repro labels, CPU score-claim schema mismatch, and terminal
  dispatch claim missing runtime-tree SHA binding.
- FEC6 writeup pose-marginal correction:
  `.omx/research/fec6_writeup_pose_marginal_correction_20260517_codex.md`.
  The live `docs/pr_writeups/cpu_frontier_fec6_20260517.md` WIP now uses
  `5/sqrt(10*d_pose) = 291.44`, not `922`, and its 1000-byte pose example is
  corrected to a net score regression unless `d_pose` drops by at least about
  `2.24e-6`. Future FEC6/Rule #6 byte-spending claims should use
  `tac.score_geometry.pose_byte_tradeoff`.
- FEC6 selector operator-space audit:
  `.omx/research/omx_parent_markdown_and_fec6_selector_operator_followup_20260517_codex.md`.
  Reusable surface:
  `tac.fec6_selector_operator_space` and
  `tools/audit_fec6_selector_operator_space.py`. Local artifact:
  `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space_manifest.json`.
  Result: `operator_row_count=40`, `raw_archive_byte_rows_emitted=0`,
  `proxy_and_nonpositive_bit_rows=[]`, selector payload entropy gap `8` bytes
  versus `78` required charged bytes to strictly cross `<0.192`. Same-runtime
  FEC6 selector byte-only polish from the current pair table is blocked; next
  FEC6 work must use new paired component rows or materialize a
  component-moving packet operator with byte-consumption proof.
- L5-v2 12-month foreseeable-failures sidecar:
  `.omx/research/l5_v2_next_12_months_foreseeable_failures_20260517_subagent.md`.
  Key tripwires: architecture-lock false authority, TT5L custody refresh
  loops without harvested cells, non-causal side-info controls, no-op Rule #6
  hyperprior/entropy work, axis drift, and optimizer/probe overfitting.
- L5 Wyner-Ziv rate-only bound review:
  `.omx/research/l5_wyner_ziv_rate_only_bound_adversarial_review_20260517_codex.md`.
  Corrected decision: L5 remains P0 because `4800 -> 2000` pose-stream shrink
  would lower FEC6 CPU to about `0.19019`, but a rate-only pose-stream shrink
  cannot justify `-0.008` to `-0.015` or `0.17-0.18` claims. Larger L5 claims
  now require component-moving proof or a larger charged-byte section.
- Key action change: harvest format0D's two-pass additive grammar as a donor
  primitive for Rule #6 A1/FEC6 byte-closed bolt-ons, but do not route the P0
  queue back into PR106-only local-basin polish. Any direct PR106 revisit must
  start with CPU/CUDA raw-output xray and extra-stream ablations.
- Parent-scope cargo-cult / Quantizr staircase follow-up:
  `.omx/research/omx_parent_markdown_cargo_cult_and_quantizr_staircase_review_20260517_codex.md`.
  Non-research `.omx` Markdown added no new score authority, but did preserve
  live no-retread signals: full-score curriculum difficulty, PoseNet
  preprocessing sensitivity, scorer-input anatomy, zero-order arithmetic
  failures versus Brotli, and stack-order discipline. The Quantizr staircase
  helper now has package discoverability, ruff hygiene, and softened
  score-authority wording; it remains a training scaffold, not a frontier
  claim, until byte-closed trainer adoption proves component movement.

## Dispatch Discipline

- No provider dispatch without `tools/claim_lane_dispatch.py claim`.
- No CPU/CUDA promotion without axis-labeled paired custody.
- No architecture lock until the shared authority predicate allows it.
- No score claim from planning, dry-run, macOS, proxy, or diagnostic anchors.
- Every result review must preserve failure class, custody, recomputed formula,
  and reactivation criteria.

## Parent-Scope OMX Markdown Scan

On 2026-05-17, the Markdown scan was widened from `.omx/research` to all
`.omx/**/*.md`, then repeated with `--hidden --no-ignore` so ignored
`.omx/auto_memory_snapshot_*` and `.omx/tmp` Markdown were not silently
excluded. Relevant non-research control surfaces checked:

- `.omx/state/current_focus.md` - refreshed by this file.
- `.omx/state/next_experiments.md` - refreshed alongside this file.
- `.omx/state/active_lane_dispatch_claims.md` - current source for dispatch
  conflict/terminal status.
- `.omx/auto_memory_snapshot_20260504T230223Z/*.md` - ignored historical
  Claude/OMX memory snapshot; no current L5/TT5L authority, but it preserves
  no-signal-loss, stack-order, entropy-coder, overfit-to-one-video,
  PoseNet-sensitivity, FiLM-pose-plumbing, derive/sweep/learn, and
  remote-tarball lessons.
- `.omx/tmp/*.md` - ignored temporary appendices and detached clone READMEs;
  useful as forensic inputs only, not current score authority.
- `.omx/notepad.md` - stale April AV1/Track-B notebook, not current L5
  authority.
- `.omx/release_manifest_v0.2.0-rc1.md` - release hygiene context, not current
  score authority.
- `.omx/state/dispatch_queue.md` - historical HTD queue; not the May 17
  Rule #6/L5-v2 priority list.

Detailed scan ledger:
`.omx/research/l5_v2_omx_parent_markdown_scope_refresh_20260517_codex.md`.
No-ignore follow-up ledger:
`.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`.
FEC6 selector operator follow-up:
`.omx/research/omx_parent_markdown_and_fec6_selector_operator_followup_20260517_codex.md`.
Parent-scope active-claims bugfix:
`.omx/research/omx_parent_markdown_modal_cpu_dispatch_bugfix_20260517_codex.md`
records the fresh no-ignore `.omx` Markdown scan that found
`master_gradient_fec6_modal_cpu_dispatch` failing with rc=2 after claim
creation because `experiments/modal_train_lane.py` did not support `--gpu CPU`.
The Modal dispatcher now has a CPU function target for tool dispatches; future
real master-gradient work still needs a fresh active claim and normal custody.
Latest TT5L route/doctor custody refresh is source-relevant-path clean but must
be regenerated immediately before provider execution; the generated packet's
recorded commit is a reproducibility input, not self-referential dispatch
authority.
Current-main full parent Markdown scan:
`.omx/research/omx_parent_markdown_current_main_full_scan_20260517_codex.md`
records `2410` total `.omx/**/*.md` files, `636` non-research Markdown files,
and `379` keyword-matching non-research files. It found no new authority beyond
`current_focus`, `next_experiments`, and `active_lane_dispatch_claims`, but it
did expose the proxy/advisory JSONL append-lock mismatch now fixed in the MPS
and macOS-CPU advisory helpers.

## Required Refresh Cadence

- Refresh this file after any Rule #6 dispatch result, TT5L side-info harvest,
  L5-v2 architecture-lock packet change, or public frontier intake that changes
  the score target.
- Refresh `.omx/state/next_experiments.md` whenever the active P0 work order
  changes.
- Catalog #316 now checks this file, `reports/latest.md`, and
  `.omx/state/next_experiments.md` against `tac.frontier_scan` so stale
  frontier citations fail preflight instead of becoming hidden control-plane
  signal loss.
