# Next Experiments - 2026-05-17

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

This queue supersedes the stale 2026-05-09 A1-sidecar/Phase-1 queue. It is
derived from the May 17 T4 symposium, the parent-scope `.omx` Markdown scan
including ignored auto-memory/tmp Markdown, and the current L5-v2
architecture-lock/TT5L custody artifacts.

## Active Anchor

- Canonical scanner-derived best CPU anchor:
  `0.1920513168811056`
  `[contest-CPU; GHA Linux x86_64 1:1]`, archive
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`,
  lane `lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515`.
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

### 1. Rule #6 A1 Ballé Hyperprior Bolt-On

- Lane class: frontier-breaking bolt-on on verified A1 substrate.
- Required work: design memo, archive grammar, export contract, scorer-aware
  training, KL-on-logits `T=2.0` A1-teacher distillation, byte-mutation no-op
  proof, paired CPU/CUDA plan.
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
- Dispatch policy: local byte/provenance proof first; paired CPU/CUDA exact
  only after a byte-different packet consumes changed bytes at inflate time.

### 3. Rule #6 A1 VQ-Codebook Bolt-On

- Lane class: frontier-breaking discrete latent bolt-on.
- Required work: A1 latent codebook, rate accounting, export grammar, scorer
  distillation, exact inflate smoke.
- Dispatch policy: run after either Ballé or PR101-style entropy stack has a
  clean byte-closed smoke, unless a subagent produces a stronger direct packet.

### 4. TT5L Side-Info Effect Curve

- Lane class: L5-v2 asymptotic evidence.
- Current artifact state: 10-cell Lightning paired-axis dry-run plan exists,
  and the Modal paired dispatch plan now routes each variant through shared
  exact-dispatch authority. The live Modal plan is blocked at
  `ready_work_unit_count=0` because the runtime is missing `report.txt` and the
  variants are missing exact archive manifests. Ledger:
  `.omx/research/l5_v2_tt5l_exact_dispatch_authority_hardening_20260517_codex.md`.
- Next concrete action: materialize the missing `report.txt` and per-variant
  `archive_manifest.json` custody, rebuild the dispatch plan to
  `ready_work_unit_count=5`, then run the required Lightning doctor from
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`,
  stage per-cell source manifests, and claim each lane before non-dry-run
  submission.
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
  the active score-lowering priority.
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
