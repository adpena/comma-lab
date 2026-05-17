# Next Experiments - 2026-05-17

Authority:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

This queue supersedes the stale 2026-05-09 A1-sidecar/Phase-1 queue. It is
derived from the May 17 T4 symposium, the parent-scope `.omx` Markdown scan,
and the current L5-v2 architecture-lock/TT5L custody artifacts.

## Active Anchor

- A1 is the verified local public-axis floor:
  `0.19284757743677347` `[contest-CPU; GHA Linux x86_64 1:1]`.
- A1 on Modal T4 CUDA is `0.2263520234784395` `[contest-CUDA]`.
- Treat A1 as the Stage-1 substrate for Rule #6 bolt-ons and as the control
  arm for TT5L/L5-v2 score-lowering evidence.

## Queue

### 1. Rule #6 A1 Ballé Hyperprior Bolt-On

- Lane class: frontier-breaking bolt-on on verified A1 substrate.
- Required work: design memo, archive grammar, export contract, scorer-aware
  training, KL-on-logits `T=2.0` A1-teacher distillation, byte-mutation no-op
  proof, paired CPU/CUDA plan.
- Dispatch policy: no spend until byte-closed packet exists and the lane is
  claimed. First paid step should be a timing smoke that measures seconds/epoch
  and produces a reviewable archive path.

### 2. Rule #6 A1 PR101-Style Entropy Stack Bolt-On

- Lane class: frontier-breaking byte-coding bolt-on.
- Required work: per-tensor byte map, Brotli/LZMA/Huffman tournament, manifest
  of consumed byte sections, monolithic archive grammar, no-op detector.
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
- Current artifact state: 10-cell Lightning paired-axis dry-run plan exists;
  execution still requires doctor and per-cell custody.
- Next concrete action: run the required Lightning doctor from
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`,
  then stage per-cell source manifests and claim each lane before non-dry-run
  submission.
- Dispatch policy: no non-dry-run cell without doctor `status=OK`, active
  claim, source manifest, and terminal-claim plan.

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
