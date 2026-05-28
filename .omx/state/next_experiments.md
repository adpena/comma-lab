# Next Experiments - 2026-05-22

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- `rank_or_kill_eligible=false`
- `promotable=false`

This queue supersedes the stale 2026-05-09 A1-sidecar/Phase-1 queue. It is
derived from the May 17 T4 symposium, the parent-scope `.omx` Markdown scan
including ignored auto-memory/tmp Markdown, and the current L5-v2
architecture-lock/TT5L custody artifacts.

## Active Anchor

- Canonical scanner-derived best CPU anchor:
  `0.1920099730474962`
  `[contest-CPU; Linux x86_64 1:1]`, archive
  `18e3155fbbbe9ab23e1c21bc0d99ba8d18657a71c3129fc5ff9e0405b67d1669`,
  lane `feca_selector_reparam_scale64_alpha1_cpu_exact`.
  Refresh from `reports/latest.md` and
  `.omx/state/canonical_frontier_pointer.json`; this file is a mirror, not a
  frontier source of truth.
- Canonical scanner-derived best CUDA anchor:
  `0.20533002902019143`
  `[contest-CUDA T4]`, archive
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`,
  lane `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`.
- A1 remains the Stage-1 substrate for Rule #6 bolt-ons and the control arm
  for TT5L/L5-v2 score-lowering evidence, but not the best current axis floor:
  `0.19284757743677347` `[contest-CPU; GHA Linux x86_64 1:1]` and
  `0.2263520234784395` `[contest-CUDA T4]`.

## Queue

### 0. MLX Portable-Local-Substrate Sign Calibration

- Lane class: local research-signal and spend-triage hardening; no score
  claim.
- Required work: ingest the 2026-05-22 decoder-q advisory-negative batch as
  signed labels, fit a repair objective that distinguishes high visible-change
  atoms from score-lowering atoms, expand full-300-window structural features,
  and re-run OOF validation before generating a new bounded candidate batch.
- Dispatch blocker: exact-eval spend remains blocked unless a fixed-length,
  surface-guided candidate improves on `[macOS-CPU advisory decoder-q]` and the
  attached MLX score-calibration gate keeps the MLX margin above
  `8.801772121230789e-05` against a full-sample `contest-CPU` or
  `contest-CUDA` auth-axis payload accepted by
  `tac.auth_eval_schema.required_contest_auth_axis_payload_blockers`.
- Hard no-dispatch set: `a2f90a216aac4184`, `a9b04920db67ec71`, and
  `8f3a33e49b9b7906` are advisory regressions and must not enter exact CUDA
  queue from the current selector.
- Portability task: implement and verify full-weight PyTorch -> NumPy -> MLX
  state/intermediate trace parity for the canonical scorer path. MLX GPU/batch
  remains research-only until batch invariance passes against CPU-stable rows.
- DQS1 continuation: rank021/pair0371 is the exact `[contest-CPU]` frontier at
  `0.19202828295713675`; compact `sorted_gap_uleb` top32 is superseded on CPU
  at `0.19202894881608987` and its exact `[contest-CUDA T4]` recovery
  `0.22619043540195719` is not a CUDA frontier. Next work is subset/Pareto
  search and sign-calibrated response modeling, not another unbounded exact
  replay of the same top32 row.

### 1. Rule #6 A1 Ballé Hyperprior Bolt-On

- Lane class: frontier-breaking bolt-on on verified A1 substrate.
- Required work: design memo, archive grammar, export contract, scorer-aware
  training, KL-on-logits `T=2.0` A1-teacher distillation, byte-mutation no-op
  proof, paired CPU/CUDA plan.
- Quantizr staircase scaffold:
  `.omx/research/omx_parent_markdown_cargo_cult_and_quantizr_staircase_review_20260517_codex.md`
  reviewed the non-research `.omx` Quantizr/PoseNet/scorer/entropy notes and
  fixed the helper surface. Use `tac.training_curriculum.QuantizrFiveStageStaircase`
  only as a scaffold: it creates no score authority until a real trainer
  adoption proves full-score difficulty, transition firing, consumed bytes, and
  component movement.
- 2026-05-17 unwind constraint: do not reuse the existing Z3HV2
  direct-residual export as the Ballé implementation. Existing Z3HV2 full is
  classified in
  `.omx/research/rule6_z3v2_direct_residual_unwind_20260517_codex.md` as
  `direct_residual_control`, byte-negative by `860` inner bytes, with no active
  Ballé entropy residual decoder. A valid Ballé bolt-on must consume side-info
  in the residual entropy decoder or use a distinct byte-closed grammar.
- Dispatch policy: no spend until byte-closed packet exists and the lane is
  claimed. First paid step should be a timing smoke that measures seconds/epoch
  and produces a reviewable archive path.

### 2. Rule #6 A1 PR101-Style Entropy Stack Bolt-On

- Lane class: frontier-breaking byte-coding bolt-on.
- Required work: per-tensor byte map, Brotli/LZMA/Huffman tournament, manifest
  of consumed byte sections, monolithic archive grammar, no-op detector.
- 2026-05-17 PR106 donor constraint:
  `.omx/research/pr106_format0d_wip_adversarial_review_20260517_codex.md`
  validates format0D's two-pass additive grammar as a real mechanism but
  classifies the PR106 lane itself as local-basin forensic/control evidence.
  If this queue uses the format0D idea, transplant the primitive onto the
  verified A1/FEC6 CPU anchor with a CPU-aware objective and paired CPU/CUDA
  plan; do not retread PR106-only polish as P0.
- 2026-05-17 byte-escape profiler:
  `.omx/research/a1_rule6_byte_escape_profile_20260517_codex.md` classifies
  the current A1 runtime grammar as `saturated_byte_only_current_runtime`.
  Evidence: A1 latent raw-LZMA sweep has best `15387` bytes, equal to source;
  A1's 607-byte Huffman-enum sidecar has a 4-byte oracle entropy gap but the
  600-byte runtime format cannot represent the current choices (`max=445`),
  so the usable supported sidecar floor remains 607 bytes. This is not a
  method kill: it blocks retreading existing-grammar byte-only arithmetic and
  redirects this lane to a new consumed runtime grammar or component-changing
  byte map.
- No-ignore scan constraint: do not retread generic zero-order arithmetic
  coding on HNeRV-like latent bytes. The ignored auto-memory snapshot preserves
  a concrete warning that Brotli can beat zero-order entropy via LZ77/context;
  first profile section-conditioned entropy, then choose a terminal coder per
  stream with byte-consumption proof.
- FEC6 CPU-frontier packet constraint:
  `.omx/research/fec6_cpu_frontier_submission_surface_adversarial_review_20260517_codex.md`
  classifies FEC6 as a real best `[contest-CPU]` anchor but not a clean
  submission packet. Do not spend P0 on same-runtime byte polish unless a
  profiler finds at least `78` charged, consumed bytes. If the operator wants
  a CPU-mode submission, first materialize the contest-final packet surface
  and rerun strict compliance; otherwise keep this lane pointed at
  component-moving Rule #6 changes.
- FEC6 pose-vs-byte tradeoff constraint:
  `.omx/research/fec6_writeup_pose_marginal_correction_20260517_codex.md`
  corrects the writeup's pose marginal to `291.44`, not `922`, and proves that
  spending 1000 bytes for a `1e-6` `d_pose` reduction is a net regression. Any
  FEC6/Rule #6 packet that spends bytes to move PoseNet should include a
  `tac.score_geometry.pose_byte_tradeoff` row before paid dispatch.
- Dispatch policy: local byte/provenance proof first; paired CPU/CUDA exact
  only after a byte-different packet consumes changed bytes at inflate time.

### 2a. Master-Gradient Operator Response Rewrite

- Lane class: apparatus fix for CPU-frontier/Rule #6 search, not a score claim.
- 2026-05-17 adversarial review:
  `.omx/research/master_gradient_raw_byte_finite_difference_adversarial_review_20260517_codex.md`.
  Raw archive-byte or bit finite differences are blocked for ZIP plus
  entropy-coded packets because the mutation is usually a container/CRC/header
  or compressed-stream corruption, not a local score derivative.
- Required work: rewrite any `(N_archive_bytes, 3)` master-gradient campaign as
  an `(N_valid_mutation_operators, 3)` response matrix. Each row must declare a
  grammar section or mutation operator, rebuild ZIP metadata/CRC, prove inflate
  success, carry an axis label, and keep `score_claim=false` until exact result
  review.
- Landed planning surface:
  `tac.master_gradient_operator_plan` plus
  `tools/build_master_gradient_operator_plan.py`. The tool consumes a
  parser-proven `tac_frontier_archive_layout_v1` manifest or archive path and
  emits grammar-aware operator rows only; raw archive-byte rows emitted must
  remain `0`.
- First executable row class:
  `tac.master_gradient_brotli_operator_candidate` plus
  `tools/build_master_gradient_brotli_operator_candidate.py`. Current proven
  local candidate on public PR106 reduces `decoder_packed_brotli` by `151`
  archive bytes via lossless Brotli recompression, with packet closure proven
  and runtime/exact-eval gates still closed. This is a template, not the P0
  score target.
- Next score-relevant operator class: compact DQS1/FEC6-family selector-grammar
  mutation on the current `[contest-CPU]` frontier archive. Do not apply raw
  byte flips; parse the selector/DQS1 payload, mutate a grammar-level choice,
  rebuild the wrapper/ZIP, and prove runtime consumption before dispatch.
- Landed selector audit:
  `tac.fec6_selector_operator_space` plus
  `tools/audit_fec6_selector_operator_space.py`. The current FEC6 archive has
  a selector payload entropy gap of only `8` bytes against `78` charged bytes
  required to strictly cross `<0.192` with unchanged components. The available
  proxy pair table produced `40` grammar-aware operator rows and
  `0` proxy-improving/nonpositive-bit rows, with
  `raw_archive_byte_rows_emitted=0`. This blocks same-runtime selector-polish
  retreads from the current rows; proceed only with new paired component rows
  or a byte-different component-moving packet operator with consumption proof.
- Partner WIP blocker:
  `.omx/research/master_gradient_partner_wip_false_authority_review_20260517_codex.md`.
  Do not land the current untracked `src/tac/master_gradient.py` as a master
  authority while it models `(N_archive_bytes, 3)` gradients and raw
  `{byte_idx: delta}` projections. The next valid interface is a typed
  `CandidateModificationSpec` / operator-response row schema. The dirty
  cathedral-autopilot hook is anchor-presence diagnostics until candidates
  carry real packet-valid modification specs.
- Parent-scope dispatch actuator fix:
  `.omx/research/omx_parent_markdown_modal_cpu_dispatch_bugfix_20260517_codex.md`.
  The `.omx/state/active_lane_dispatch_claims.md` parent Markdown scan found
  `master_gradient_fec6_modal_cpu_dispatch` failed with rc=2 after claim
  creation because `experiments/modal_train_lane.py` did not support
  `--gpu CPU` even though the dispatch protocol and recipe allowed CPU tool
  dispatches. The dispatcher now has a no-GPU Modal CPU function target.
- Dispatch policy: no provider dispatch for raw byte/bit flip probes. A local
  operator-row manifest may proceed after
  `tools/audit_master_gradient_feasibility.py` reports
  `operator_response_probe_ready`.

### 3. Rule #6 A1 VQ-Codebook Bolt-On

- Lane class: frontier-breaking discrete latent bolt-on.
- Required work: A1 latent codebook, rate accounting, export grammar, scorer
  distillation, exact inflate smoke.
- Dispatch policy: run after either Ballé or PR101-style entropy stack has a
  clean byte-closed smoke, unless a subagent produces a stronger direct packet.

### 4. TT5L Side-Info Effect Curve

- Lane class: L5-v2 asymptotic evidence.
- 2026-05-17 rate-only bound:
  `.omx/research/l5_wyner_ziv_rate_only_bound_adversarial_review_20260517_codex.md`
  audits the L5 Wyner-Ziv pose-stream premise. A `4800 -> 2000` byte
  pose-stream shrink gives only `0.0018644050687420797` score savings,
  projecting FEC6 CPU to about `0.19019`; full removal of the `4800` byte
  section gives at most `0.0031961229749864224`, projecting about `0.18886`.
  This is still score-lowering and sub-`0.192`, but it is not the `-0.008` to
  `-0.015` band unless decoded-pose/frame component movement is proven.
- Current artifact state: Modal paired dispatch plan is custody-green at
  `ready_work_unit_count=5`; all per-variant `dispatch_blockers=[]`; the
  submission runtime has `report.txt`; all five variant archives have adjacent
  `archive_manifest.json` custody. The Lightning path is also refreshed:
  paired-axis plan `10/10`, execution preflight `ready_cell_count=10`,
  execution bundle `ready_dry_run_cell_count=10`, dry-run verification
  `10/10`, route packet `artifact_blocker_count=0`, and required doctor plan
  `ready_for_operator_doctor=true`; route/doctor packets are source-relevant
  clean but must be regenerated immediately before provider execution rather
  than treated as self-referential commit authority. Ledger:
  `.omx/research/l5_v2_tt5l_dispatch_custody_materialization_20260517_codex.md`.
- Next concrete action: run the required Lightning doctor from
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`,
  stage per-cell source manifests, claim each lane, and submit the 10 paired
  CPU/CUDA cells only after doctor/source-manifest custody is green. The
  refreshed route packet deliberately excludes architecture-lock as an upstream
  source artifact; architecture lock is regenerated downstream after route and
  doctor status to avoid circular SHA drift.
- Dispatch expectation: rate-only L5 packets should be judged against the
  `-0.0019` to `-0.0032` closed-form band and the `<0.192` crossing threshold,
  not against the stale `-0.008` to `-0.015` band. If a packet claims the
  larger band, it must include component deltas and raw-output custody proving
  PoseNet/SegNet movement.
- Dispatch policy: no non-dry-run cell without exact-dispatch authority green,
  doctor `status=OK`, active claim, source manifest, and terminal-claim plan.

### 5. SCORER-AWARENESS Probe Wave

- Lane class: apparatus fix that directly changes dispatch decisions.
- Required work: probe MI between each high-risk substrate distinguishing
  feature and scorer attention/argmax maps; classify conditioning versus
  regularization.
- Dispatch policy: forward-pass/probe-only is acceptable as a cheap gating
  wave; no class-wide substrate kill before these probes.

### 6. Z6 Per-Frame-Renderer-Axis Replacement

- Lane class: L5-v2 predictive-coding successor.
- Required work: replace original FiLM-bottlenecked Z6 with ego-motion
  conditioning on renderer coordinate sampling or another per-frame renderer
  axis; include a probe that compares it against identity and FiLM.
- Dispatch policy: design and dry-run proof first; original Z6 FiLM path is
  not dispatchable as a frontier lane.

## Explicitly Superseded Or Historical

- May 4 Omega-W/SJ-KL/legacy HTD queue rows are historical.
- A1 latent sidecar remains local/proxy until a full 600-pair byte-different
  runtime-consumed packet exists.
- PR106 packet variants remain forensic control and exact-eval lessons, not
  the active score-lowering priority. Latest reviewed disposition:
  `.omx/research/pr106_format0d_wip_adversarial_review_20260517_codex.md`.
- Existing Z3HV2 direct-residual exports are measured controls, not active
  frontier candidates: paired Z3 v2 full scored `0.1986956456779881`
  `[contest-CPU]` and `0.23170948072940661` `[contest-CUDA]` with archive
  `b6c4a6f1f1f4bb29695e8ee095ca3862690b2c4833fba31579406179aaf35a4b`.
- TT5L paired diagnostic anchor at score about `3.9` is a measured-config
  failure and custody anchor, not a lane kill and not a promotion result.

## Stop/Continue Thresholds

- Continue any Rule #6 bolt-on that produces a byte-different, consumed packet
  and improves A1 on either trusted same-axis smoke.
- Escalate to paired exact CPU/CUDA only after byte-closed local custody is
  green.
- Stop a measured configuration when exact paired evidence is worse and the
  result review classifies it as method-specific rather than harness/runtime.
- Do not retire a method family from one bad run; record the reactivation
  criteria and move to the next highest-EV unblocked lane.
