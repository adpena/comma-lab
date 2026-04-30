# Training-Free GRPO Applicability To DX Hardening And Shannon-Floor Search

Date: 2026-04-30
Agent: Codex research/design/implementation worker
Scope: research and design ledger. No code changes. No score claims.

## Evidence Status

This document is external research, engineering inference, and implementation
proposal only. It does not promote, rank, kill, or score any contest lane.

The only source of promotion-grade score truth remains exact CUDA auth eval on
the exact archive bytes:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

Prefer `experiments/contest_auth_eval.py --device cuda` and its
`contest_auth_eval.json`. Any Training-Free GRPO controller, agent practice
memory, local model review, or experience library is advisory until a candidate
archive is independently evaluated through that path with custody, manifest,
hardware provenance, and component recomputation.

## Primary Sources Consulted

Fetched/consulted on 2026-04-30:

1. Training-Free Group Relative Policy Optimization, arXiv:2510.08191:
   https://arxiv.org/abs/2510.08191 and https://arxiv.org/pdf/2510.08191
2. Tencent/Youtu-Agent preview branch for the paper:
   https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO
3. Current Youtu-Agent repository and main-branch positioning:
   https://github.com/TencentCloudADP/youtu-agent
4. Current Youtu-Agent Agent Practice documentation:
   https://tencentcloudadp.github.io/youtu-agent/practice/
5. DeepSeekMath original GRPO source paper, arXiv:2402.03300:
   https://arxiv.org/abs/2402.03300

Repo control-plane sources read:

- `.omx/research/dx_self_protecting_harness_hardening_20260430_codex_progress.md`
- `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
- `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
- `.omx/research/pufferlib_rl_visual_primitives_shannon_floor_20260430_agent.md`
- `.omx/research/external_research_arxiv_2604_26919_shannon_floor_20260430_agent.md`
- `src/tac/preflight.py`
- `src/tac/tests/test_preflight_meta_bugs.py`
- `src/tac/tests/test_remote_auth_eval_hardening.py`

## Paper-Derived Facts

The arXiv paper proposes a training-free variant of GRPO for LLM agents. The
policy model stays frozen. Instead of applying gradient updates, the method
maintains external experiential knowledge `E`, injects it into later prompts,
and iteratively updates it from grouped rollouts.

The paper's core loop is:

1. For a query, generate a group of `G` rollouts from a policy conditioned on
   current experience `E`.
2. Score each rollout using a reward/verifier.
3. Only where the group has clear winners and losers, summarize trajectories
   and extract a natural-language group-relative semantic advantage.
4. Update the experience library by modifying, merging, or deleting short,
   general lessons.
5. Reuse the revised experience library as a token prior during later API
   calls.

The paper positions this as context-space policy optimization, not parameter
training. The authors evaluate math reasoning and web-searching tasks, and the
official Youtu-Agent documentation exposes this as an "Agent Practice" module
with custom verification functions, rollout concurrency, `grpo_n`, experience
records, and evaluation configs.

The original GRPO reference point is DeepSeekMath's parameter-space GRPO,
which estimates group-relative advantage from multiple completions while
reducing PPO memory cost by avoiding a separate critic model. Training-Free
GRPO keeps the group-relative comparison idea but replaces numeric advantage
updates with semantic experience updates.

## Repo-State Facts

The current contest system already has strong guardrails for failure modes that
a controller could otherwise amplify:

- `contest_auth_eval.json` is the score authority; human log score scraping is
  blocked by preflight and an adjudicator.
- CPU, MPS, proxy, Modal-CPU, stale-log, smoke, and byte-only evidence are
  non-promotable.
- Lightning exact-eval routing has supply-chain, CUDA, and writable-output
  preflights.
- Fake/random/debug sensitivity artifacts are fenced as non-promotable.
- MCP integrations are disabled unless explicitly re-enabled by the user.
- Broad kill/retirement wording is controlled by evidence grade and
  adversarial review discipline.
- The worktree is actively dirty and other agents are touching exact-eval and
  Lightning files, so any patch must avoid those surfaces.

## Executive Verdict

Training-Free GRPO is useful here as a disciplined advisory controller and
memory-distillation pattern. It should not be used as a learned codec, score
evidence generator, automatic launch authority, or source of compression
claims.

Highest-value adoption:

1. Lane forensics memory:
   distill exact bad-result reviews into short, scoped, reusable experiences
   that reduce repeat harness/config mistakes.

2. Experiment selection:
   compare groups of candidate lane cards under deterministic verifiers, then
   extract semantic lessons about which evidence gaps, byte gates, and failure
   classes matter before spending exact-eval budget.

3. Local model-assisted review:
   use local or API LLMs to generate structured proposals and adversarial
   critiques, but require deterministic validators and human/custody gates
   before dispatch or paper claims.

4. Controller workflow:
   use group-relative selection to order cheap build/smoke/diagnostic work,
   not to optimize against exact CUDA score directly.

5. Compression/search stack:
   only indirect value. The method can choose between OWV3 byte plans,
   Alpha-Geo diagnostics, NWCS build candidates, or residual-routing ideas.
   It does not itself reduce archive bytes or prove rate-distortion gains.

Do not import Youtu-Agent or add a new dependency for this repo now. The safe
transfer is the algorithmic protocol: grouped candidates, deterministic reward,
semantic advantage, versioned experience memory, and strict deployment gates.

## Contest Translation

### Query

A query is a bounded decision card, not a raw score request. Examples:

- Which exact-eval candidate deserves the next T4 slot?
- Which failure mode explains a bad CUDA result?
- Which smoke-only artifact should be upgraded to a deterministic build?
- Which sensitivity artifact is eligible for assembly into
  `component_sensitivity_v1`?
- Which archive-byte candidate is worth component-gated exact eval?

### Group Rollouts

A group rollout is a proposal from an agent, local model, heuristic scheduler,
or hand-written lane plan. Each rollout must be materialized as structured
JSON or markdown with:

- source ledgers read,
- candidate archive or build artifact if any,
- expected write set,
- evidence grade requested,
- required validators,
- exact commands,
- known blockers,
- non-overlap with active agents,
- no-score-claim declaration unless exact CUDA evidence already exists.

### Reward

The reward must be a deterministic tuple computed by repo validators, not an
LLM preference:

```text
reward = (
  evidence_grade_ceiling,
  archive_custody_closed,
  contest_auth_eval_json_present,
  cuda_device_verified,
  component_gates_configured,
  deterministic_manifest_ok,
  hidden_sidecar_absent,
  byte_gate_ok,
  stale_state_risk,
  expected_cost,
  novelty_vs_existing_lanes,
  failure_forensics_complete
)
```

For selection, this can be collapsed into an advisory scalar. That scalar is a
scheduler reward only. It is not a contest score and must never appear in
paper result tables.

### Semantic Advantage

Semantic advantage should be extracted only when a group has clear verified
winners and losers. A valid advantage is a short, scoped operational lesson:

- tied to artifact paths and validators,
- written as a reusable rule,
- no broader than the evidence supports,
- marked with source evidence grade,
- invalidated or superseded when newer exact evidence disagrees.

Invalid examples:

- "OWV3 is dead."
- "This proxy win means exact eval will improve."
- "The local model ranked this lane first, so promote it."

Valid examples:

- "When an archive-byte candidate beats the PFP16 byte frontier by byte count
  only, require exact CUDA eval with PoseNet and SegNet gates before ranking."
- "When a lane uses debug/fake sensitivity, stop before `contest_auth_eval.py`
  and record `promotion_eligible=false`."
- "When a failure combines good byte rate with PoseNet collapse, retire only
  the measured implementation/config and open geometry-preserving redesign."

## Proposed Artifact Schemas

These are future implementation proposals. They were not added in this turn.

### `tf_grpo_decision_card_v1`

```json
{
  "schema": "tf_grpo_decision_card_v1",
  "created_at_utc": "2026-04-30T00:00:00Z",
  "decision_id": "owv3_exact_eval_candidate_r1",
  "question": "Should this byte-feasible archive consume exact CUDA eval?",
  "candidate_paths": [],
  "source_ledgers": [],
  "proposed_action": "exact_eval | build_only | forensic_review | defer",
  "evidence_claimed": "none",
  "evidence_requested": "A | A++ | empirical | diagnostic",
  "validators_required": [],
  "known_blockers": [],
  "no_score_claim": true
}
```

### `tf_grpo_rollout_review_v1`

```json
{
  "schema": "tf_grpo_rollout_review_v1",
  "decision_id": "owv3_exact_eval_candidate_r1",
  "rollout_id": "candidate_a",
  "deterministic_checks": {
    "archive_sha256_present": false,
    "contest_auth_eval_json_present": false,
    "cuda_verified": false,
    "manifest_deterministic": false,
    "hidden_sidecars_absent": false
  },
  "advisory_reward": 0.0,
  "reward_basis": "selection-only, not score evidence",
  "reviewer_notes": []
}
```

### `tf_grpo_experience_v1`

```json
{
  "schema": "tf_grpo_experience_v1",
  "experience_id": "DX-20260430-001",
  "text": "When a remote eval emits both logs and contest_auth_eval.json, adjudicate only from JSON and recomputed components.",
  "scope": "remote exact-eval adjudication",
  "source_evidence": ["path/to/evidence.json"],
  "evidence_grade": "engineering_policy",
  "created_from_decision_ids": [],
  "status": "active",
  "supersedes": [],
  "invalidated_by": null
}
```

## Failure Modes

1. Score conflation:
   an advisory controller reward is mistaken for contest score. Mitigation:
   all controller artifacts must carry `no_score_claim=true`, and exact score
   rows must cite `contest_auth_eval.json`.

2. LLM reward hacking:
   rollouts optimize persuasive prose instead of valid artifacts. Mitigation:
   deterministic validators score custody, device, sample count, component
   gates, hidden sidecars, and byte budget before any semantic extraction.

3. Stale-memory lock-in:
   old experiences override newer exact evidence. Mitigation: every experience
   has source paths, date, scope, status, and supersession fields.

4. Overbroad retirement:
   a bad result becomes a family kill. Mitigation: semantic advantages must use
   scoped vocabulary and carry the measured implementation/config boundary.

5. Proxy leakage:
   CPU/MPS/local proxy results drive promotion. Mitigation: controller rewards
   may use proxy diagnostics only for triage; promotion reward requires CUDA
   exact eval fields.

6. Sidecar contamination:
   an experience library or local model output becomes a score-affecting
   runtime sidecar. Mitigation: no controller artifacts are read by
   `inflate.sh` or included in archive runtime unless explicitly charged and
   compliance-reviewed.

7. Group collapse:
   all candidates are equally invalid or equally incomplete. Mitigation: do
   not extract semantic advantage unless there are verified winners and losers.

8. Supply-chain and tool risk:
   importing Youtu-Agent or enabling MCP/remote tools expands attack surface.
   Mitigation: adopt the protocol only; do not add dependency/runtime
   integration without separate review.

9. Non-stationarity:
   lane state changes while experience is being distilled. Mitigation:
   decision cards must record source ledger timestamps, artifact SHAs, and live
   state probes where relevant.

10. Spend automation:
    controller dispatches exact eval too aggressively. Mitigation: phase gates
    require human approval until replay precision is measured on historical
    incidents.

## Reproducibility Requirements

Any future implementation must record:

- model name/provider and prompt hash for each rollout,
- input ledger paths and SHA-256 where practical,
- exact candidate artifact paths and SHA-256,
- deterministic verifier versions and command lines,
- raw rollouts, structured reviews, and selected semantic advantages,
- experience-library SHA-256 before and after update,
- no-score-claim flag,
- decision owner and deployment gate outcome.

For any score-affecting candidate selected by the controller, the deploy packet
still needs archive, archive SHA-256, byte count, manifest, CUDA
`contest_auth_eval.json`, logs, hardware provenance, exact command, upstream
hash, source/staged-tree manifest, and adversarial review status.

## Deployment Order

Phase 0 - Ledger only:
This document. No code and no controller artifacts. Use it to guide human
review and future implementation.

Phase 1 - Historical replay:
Create a small offline corpus of known incidents from existing ledgers: PFP16
parser bug, Lane 12 geometry collapse, OWV3 byte block, fake sensitivity,
Lightning runner preflight, and stale Vast state. Generate grouped decision
cards and measure whether deterministic rewards rank the known safe action
above the known bad action.

Phase 2 - Read-only experience library:
Materialize `tf_grpo_experience_v1` records under `.omx/research/` or
`.omx/state/`, but keep them outside any archive/runtime path. Add a validator
that requires source evidence, scope, status, and `no_score_claim`.

Phase 3 - Advisory proposal generator:
Use experiences in prompts for local or API model-assisted review. Outputs are
candidate cards only. They cannot launch, delete, promote, rank, or retire.

Phase 4 - Human-gated scheduler:
Allow the controller to recommend build-only or diagnostic jobs after replay
precision/recall is strong. Exact-eval jobs still require explicit human or
parent-agent approval and all current preflight gates.

Phase 5 - Controlled integration:
Only after phases 1-4 prove value, consider a small repo tool such as
`scripts/review_decision_cards.py` or `scripts/update_experience_library.py`.
Do not integrate with `inflate.sh`, scorer code, Lightning exact-eval files, or
archive construction.

## Candidate Uses By Stream

### Experiment Selection

Use group-relative comparison to choose between independent next actions:
OWV3 exact eval, Alpha-Geo redesign, NWCS build-only smoke, sensitivity
producer hardening, or live harvest. Reward selection by custody closure,
blocked/unblocked status, expected cost, and whether the action can produce
new exact evidence.

### Lane Forensics

After a bad result, generate multiple failure-mode hypotheses. Reward them by
artifact closure and whether they explain device, sample count, archive bytes,
component collapse, hidden sidecars, and config/provenance. Extract scoped
experiences only after the forensic path is verified.

### Local Model-Assisted Review

Local models can summarize logs, propose checklists, and draft structured
cards. They cannot score, rank, kill, or dispatch. Their JSON must pass schema
validation, path existence checks, and evidence-grade checks before it enters
the research ledger.

### RL/Controller Workflows

Use Training-Free GRPO before full RL. This repo's exact-eval environment is
too expensive for policy-gradient rollouts. The right controller is staged:
cheap deterministic validators first, build-only candidates second, exact CUDA
eval last.

### Compression/Search Stack

The method does not directly compress video. Its plausible compression value is
search efficiency: preventing exact-eval spend on byte-negative candidates,
forcing sensitivity artifacts to prove component validity, and concentrating
human effort on candidates with clean custody and byte gates.

## Patch Decision

No code patch was implemented in this turn.

Reason: the repo already has strong preflight coverage for the concrete
failure classes this design touches, and no `tf_grpo_*` artifact exists yet.
Adding a preflight rule now would either be too broad or too speculative. The
lowest-risk path is to land this design first, then add a schema validator only
if/when decision cards or an experience library are materialized.

Files intentionally not touched:

- `src/tac/deploy/lightning/batch_jobs.py`
- `src/tac/tests/test_lightning_batch_jobs.py`

## Work Landed

Changed files:

- `.omx/research/training_free_grpo_dx_shannon_floor_20260430_agent.md`

No code files were edited.

## Verification Commands

Run after writing this ledger:

```bash
git diff --check -- .omx/research/training_free_grpo_dx_shannon_floor_20260430_agent.md
test -s .omx/research/training_free_grpo_dx_shannon_floor_20260430_agent.md
perl -ne 'if (/[ \t]$/) { print "$ARGV:$.: trailing whitespace\n"; $bad=1 } END { exit($bad || 0) }' .omx/research/training_free_grpo_dx_shannon_floor_20260430_agent.md
```

Python compile, pytest, and shell syntax checks were not applicable because no
Python or shell files were touched.
