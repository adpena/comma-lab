# Codex Findings - Over-Conservative Authority Bottlenecks

`research_only=true`  
`score_claim=false`  
`promotion_eligible=false`

## Scope

The operator flagged that Codex had been too conservative around A1/specialized deterministic receiver paths. Codex spawned an xhigh read-only adversarial-review agent to search for similar over-strong assumptions. No files were edited by the subagent.

This memo preserves the findings as routing signal for future Codex execution and Claude design amendment.

## Findings

### P1 - Exact-readiness accepts PR101 runtime-consumption proof but rejects deterministic packet compiler proof

`src/tac/optimizer/exact_readiness.py::validate_runtime_consumption_proof` currently accepts only `PR101_RUNTIME_CONSUMPTION_PROOF_SCHEMA`. `src/tac/packet_compiler/deterministic_compiler.py` already has a generic runtime-consumption proof path.

Why this is over-conservative:

- PR101 is one historical packet family, not the whole authority class.
- A1 specialized deterministic packets can be valid, self-contained, and exact-evaluable while still not being PR101-shaped.

Recommended fix:

- Promote a shared runtime-consumption proof validator that accepts both PR101 and deterministic packet compiler schemas when archive SHA, runtime tree SHA, consumed-section evidence, and false-authority fields are present.

### P1 - Strict-scorer rejection is token-based rather than model-load-based

`src/tac/packet_compiler/deterministic_compiler.py` forbids raw tokens such as `PoseNet`, `SegNet`, `rgb_to_yuv6`, and `upstream.modules` in runtime text.

Why this is over-conservative:

- The real contest rule is no scorer/model load at inflate time, not no scorer-named comments, schemas, labels, or pure deterministic preprocessing references.
- This can falsely block lawful deterministic packet compiler artifacts, especially the specialized A1 path sanctioned by `contest_one_video_replay`.

Recommended fix:

- Replace substring scanning with an AST/import/runtime dependency classifier.
- Hard-fail scorer model imports, weight loads, and scorer instantiation.
- Review or label pure preprocessing references separately.

### P1 - Exact-eval readiness uses byte floor as a dispatch veto

`src/tac/optimizer/exact_readiness.py::readiness_blockers` hard-blocks candidates above the active floor archive bytes unless an override flag is present.

Why this is over-conservative:

- The score formula is distortion plus rate. A larger archive can still beat the frontier when distortion improves enough.
- Byte-floor rejection is valid for submission/promotion hygiene only after score-budget accounting, not as a universal exact-eval dispatch veto.

Recommended fix:

- Replace the byte-only veto with a full score-delta budget check.
- Keep a stricter blocker for promotion/submission if no distortion improvement evidence exists.

### P2 - Cathedral autopilot has exact-eval authority but no sanctioned no-score timing-smoke/replay authority

`tools/cathedral_autopilot_autonomous_loop.py` currently expects exact-eval readiness metadata for autonomous dispatch.

Why this is over-conservative:

- Exact-eval gates are correct for score claims.
- They are too strong for non-promotional timing smokes, one-video replay probes, or deterministic receiver feasibility checks that must remain `score_claim=false`.

Recommended fix:

- Split `dispatch_action_kind`.
- Preserve strict exact-eval gates.
- Add a no-score timing-smoke/replay validator with cost cap, lane claim, artifact harvest, and forced `score_claim=false`.

### P2 - Master-gradient feasibility conflates distortion authority with rate-recode planning authority

`src/tac/master_gradient_feasibility.py` blocks raw/archive/compressed byte coordinate systems; downstream packet-diet consumers accept only logical/grammar/codec-symbol gradient domains.

Why this is over-conservative:

- That is right for distortion-response gradients.
- It is too strong for rate attacks where the operator is recoding the stream, not perturbing bytes and pretending to own a scorer derivative.

Recommended fix:

- Split `distortion_response_authority` from `rate_recode_planning_authority`.
- Add an explicit entropy/repack operator domain for archive recoding and byte-layout search.

### P2 - Semantic absence is inferred from names instead of decoded payload authority

Some route/argmax-style helpers infer semantic absence from section names and token lists.

Why this is over-conservative and also too weak:

- It can reject unnamed-but-real payload structure.
- It can also pass opaque binary payloads without proving absence.

Recommended fix:

- Distinguish `not_named_payload_detected` from `payload_absent_proven`.
- Require parser/decode evidence before retiring semantic payload routes.

### P3 - Canonical task extraction misses non-ITEM directives and degrades priority ordering

`tools/extract_canonical_tasks_from_directive.py` recognizes narrow ITEM-style forms. `src/tac/canonical_task_status/query.py` sorts null predicted bands as `0.0`, so many pending tasks become timestamp/task-id ordered.

Recommended fix:

- Add explicit directive task frontmatter/fenced blocks.
- Parse OP/phase/design-memo task forms.
- Add `priority`, `authority_surface`, and `expected_artifact_kind` fields for autopilot consumers.

## Routing

Highest-impact follow-up is the deterministic-packet authority fix:

1. Shared runtime-consumption proof validator.
2. AST/import scorer-load classifier replacing raw token bans.
3. Exact-readiness acceptance of deterministic packet compiler proofs.

That directly repairs the A1-specialized receiver path without weakening the contest rule against loading the full scorer at inflate time.
