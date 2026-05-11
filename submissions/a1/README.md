# A1 PR submission entry packet — 5-turn greenup ready-to-trigger

This packet stages the A1 archive + dual-eval custody + N council Decision 5
expansions so the operator can trigger the **5-turn skunkworks council
greenup** the instant they choose. **It is NOT a PR-submission act.** Per
CLAUDE.md "Submission PR gate" non-negotiable and the N grand council
Decision 5 verdict (8/10 OPERATOR-TRIGGER-REQUIRED, 2026-05-11), both the
greenup process AND the PR submission act require explicit operator trigger.

## Identity

- **Candidate label**: `track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6`
- **Architecture class**: `hnerv_ft_microcodec` (HNeRV-cluster representation;
  PR101-derived score-gradient finetune + inflate-time bias-correction sidecar)
- **Lane**: `lane_a1_pr_submission_entry_packet`
- **Status**: PR-SUBMISSION-READY-PENDING-COUNCIL-GREENUP (operator-trigger-required)

## Bytes

| Field | Value |
|---|---|
| `archive_sha256` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` |
| `archive_size_bytes` | `178,262` |
| `archive_member_count` | 1 (member name: `x`) |
| `runtime_tree_sha256` | `89db4fe14ac2bbffc951f8e89ac2242fa1455e0880bb3fbe963aa48e4890b5eb` |
| `runtime_file_count` | 4 (`inflate.sh`, `inflate.py`, `src/codec.py`, `src/model.py`) |
| `upstream_evaluate_py_sha256` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| `upstream_commit` | `11ad728f563d8970929e8947a1cf6124ee6303e4` |

## Dual-eval custody (BOTH axes, EXACT same archive bytes)

| Axis | Score | Hardware | Tag |
|---|---:|---|---|
| **CPU** (ranking) | **0.19284757743677347** | github-actions-ubuntu-latest-x86_64 | `[contest-CPU GHA Linux x86_64]` |
| **CUDA** (paired-axis diagnostic) | **0.22635202347843951** | Modal Tesla T4 (driver 580.95.05, cu124) | `[contest-CUDA Tesla T4]` |

Δ (CUDA − CPU) = **+0.03350**. Both axes are 1:1 contest-compliant per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".

## Leaderboard position (CPU axis ranks)

A1 CPU `0.19284758` rounds to display `0.19` — **PR101 gold display tier**.

- vs PR101 (gold) CPU `0.19284`: identical to display precision (Δ < 0.00001)
- vs PR103 (silver) CPU `0.19487`: **+0.0020 better**
- vs PR102 (third) CPU `0.19538`: **+0.0025 better**
- vs PR #107 (our apogee) CPU `0.19664`: **+0.0038 better on the ranking axis**
- CUDA pair: A1 `0.22635` vs PR #107 `0.22933` — **+0.0030 better on CUDA too**

Both axes agree A1 displaces PR #107 if operator approves PR submission.

## Entry packet structure (D5 expansions)

Per N grand council Decision 5 verdict 2026-05-11, the entry packet carries
5 expansions to feed the 5-turn greenup:

| Expansion | File | Purpose |
|---|---|---|
| **#1 Device-axis-explanation** | `DEVICE_AXIS_EXPLANATION.md` | Explain CPU-axis ranking (per operator clarification 2026-05-11) + HNeRV-cluster substrate-class drift validation |
| **#2 Mechanism-attribution-table** | `MECHANISM_ATTRIBUTION.md` | Per-mechanism description + validation path table |
| **#3 R(D) derivation** | `RD_DERIVATION.md` | Shannon rate-distortion accounting for A1's score |
| **#4 Dual-eval refresh** | `contest_auth_eval.{cpu,cuda}.json` + `dual_eval_adjudicated.json` | Both axes in custody, SHA-verified on same archive bytes |
| **#5 Pre-submission compliance stub** | `pre_submission_compliance.contest_final.json` | Compliance-check command staged; operator-trigger executes it |

## Submission packet files

| File | Purpose |
|---|---|
| `archive.zip` | A1 contest archive (single member `x`; 178,262 bytes) |
| `inflate.sh` | A1 runtime entry point (≤100 LOC; 3 positional args; `set -euo pipefail`) |
| `inflate.py` | A1 HNeRV decoder + bias-correction sidecar runtime (≤200 LOC; no scorer load) |
| `src/codec.py` | PR101 split-Brotli decoder unpacker + latent sidecar application |
| `src/model.py` | HNeRV decoder class (latent_dim=28, base_channels=36) |
| `archive_manifest.json` | Per-member identity manifest (sha256 + sizes + CRC) |
| `contest_auth_eval.cuda.json` | Modal Tesla T4 retry3 auth-eval result (canonical CUDA anchor) |
| `contest_auth_eval.cpu.json` | GHA Linux x86_64 adjudicated CPU result (canonical CPU anchor) |
| `dual_eval_adjudicated.json` | Combined dual-axis adjudication record (this packet's source-of-truth for axes) |
| `report.txt` | Human-readable evaluation + custody summary |
| `pre_submission_compliance.contest_final.json` | **STUB** — compliance-check command staged; not yet executed |
| `DEVICE_AXIS_EXPLANATION.md` | D5 expansion #1 |
| `MECHANISM_ATTRIBUTION.md` | D5 expansion #2 |
| `RD_DERIVATION.md` | D5 expansion #3 |

## 5-turn greenup workflow (operator-trigger-required)

When the operator decides to fire the greenup, the workflow template at
`.omx/research/a1_pr_submission_5_turn_greenup_workflow_template_20260511.md`
defines the per-round rubric. 15 council members review; ANY issue resets the
counter to 0.

## Three operator sub-decisions (per NOT YET ITEM 1)

Pending operator approval, surfaced by `adffef18` 2026-05-09 + ratified by
N council Decision 5 + prior pose-axis council Insight 4:

1. **Initiate the 5-turn skunkworks council greenup on A1?**
   Dual-eval custody is complete; council greenup is the next gate per CLAUDE.md
   "Submission PR gate" non-negotiable. Cost: $0 (council subagents are $0).
2. **Freeze A1 as the submission candidate OR continue substrate engineering?**
   Open lateral leaps: per-pair latent sidecar resampling + finer bias-magnitude
   sweep around V7 — each could land another 0.001–0.002 score points.
3. **Should A1 displace PR #107 apogee as our best frontier candidate?**
   Both axes agree A1 wins (+0.0038 CPU, +0.0030 CUDA).

## Cost to operator

- **Greenup trigger cost**: $0 (15 council subagents at $0 LLM cost; no GPU
  spend; no PR submission act inside the greenup itself)
- **Compliance-check execution cost**: $0 (the
  `pre_submission_compliance_check.py` is a local Python script; runs in
  ~10s on the operator's workstation)

## Risk profile

Contest closed 2026-05-05 (per handoff: "no new contest PRs since 2026-05-05").
A1 is a **post-contest** submission candidate; this is **honesty/archive PR**,
not race PR. Per CLAUDE.md "Frontier target" non-negotiable: contest closed
means honesty/archive matters MORE not less. A submitted A1 PR with full dual-eval
custody serves as a public-record correction of PR #107's leaderboard position
on the CPU axis (the ranking axis).

## Loop pause status

Loop remains **PAUSED** per 2026-05-09 directive. This entry packet preparation
changes nothing about pause status. Operator-trigger is required for both the
5-turn greenup workflow AND any subsequent PR submission act.

## Cross-references

- `feedback_a1_dual_cuda_dispatch_landed_20260509.md` — A1 dual-anchor landing
- `feedback_grand_council_5_design_decisions_review_20260511.md` — D5 verdict
- `feedback_grand_council_pose_axis_insights_review_20260511.md` — prior pose-axis council (Insight 4: 10/10 READINESS)
- `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md` — NOT YET ITEM 1
- `.omx/research/device_axis_paired_anchor_matrix_20260511.md` — full device-axis matrix
- `.omx/research/a1_pr_submission_5_turn_greenup_workflow_template_20260511.md` — 5-turn workflow template (created by this packet)
- `.omx/research/a1_pr_submission_entry_packet_summary_20260511.md` — operator-decision-ready summary (created by this packet)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
- CLAUDE.md "Submission PR gate — non-negotiable"
- CLAUDE.md "Frontier target — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Public Disclosure Hygiene"
