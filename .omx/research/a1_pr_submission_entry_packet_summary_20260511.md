# A1 PR submission entry packet — operator-decision-ready summary

**Date**: 2026-05-11
**Lane**: `lane_a1_pr_submission_entry_packet`
**Status**: ENTRY PACKET STAGED — operator-trigger-required to fire the 5-turn greenup
**Cost incurred this lane**: $0
**Cost to trigger the next step**: $0

## One-paragraph operator briefing

A1 is our best frontier candidate on the contest's CPU ranking axis. Its CPU
score `0.19285` rounds to display `0.19` — identical to PR101's gold display
tier and **+0.0038 better than our current submitted PR #107 apogee**. The
CUDA paired-axis anchor `0.22635` is also +0.0030 better than PR #107 on
CUDA. Dual-eval custody is complete (both axes 1:1 contest-compliant on
EXACT same archive bytes). Per CLAUDE.md "Submission PR gate" non-negotiable
+ N grand council Decision 5 verdict 2026-05-11 (8/10 OPERATOR-TRIGGER-
REQUIRED), submitting an A1 PR requires:
(a) operator-triggered 5-turn skunkworks council greenup process; AND
(b) operator-triggered PR-submission decision after greenup completes.
The entry packet at `submissions/a1/` has all 5 D5 expansions staged
(device-axis-explanation, mechanism-attribution-table, R(D) derivation,
dual-eval refresh, pre-submission compliance stub) ready for the operator
to trigger the greenup workflow. **Contest closed 2026-05-05** — this is
honesty/archive PR, not race PR; per CLAUDE.md "Frontier target" non-
negotiable, contest-closed means honesty matters MORE not less.

## Three sub-decisions to surface

Per NOT YET ITEM 1 of `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`
+ N grand council Decision 5 ratification:

### Sub-decision 1: Initiate the 5-turn skunkworks council greenup on A1?

- **Cost**: $0 (15 council subagents at $0 LLM cost)
- **Time**: ~1 parent-agent session per round × 5 rounds = up to 5 sessions if any
  round fails and counter resets; ~1 session if all pass on first attempt
- **Outcome**: Council-ratified readiness verdict on A1 as a submission candidate
- **Risk if YES**: zero (no GPU spend, no PR submission)
- **Risk if NO**: A1 remains in NOT YET state indefinitely; the dual-eval custody work is unused
- **Cost-EV**: $0 GPU/$0 LLM; high information value (the council surfaces hidden assumptions)

### Sub-decision 2: Freeze A1 vs continue substrate engineering?

Per 2026-05-09 NOT YET pin:
- **Option A: Freeze A1** as the submission candidate; trigger sub-decision 1 immediately
- **Option B: Lateral leap** — per-pair latent sidecar resampling + finer bias-magnitude sweep around V7 (3 GHA dispatches at ~$0.10 each) — each could land another 0.001-0.002 score points
- **Cost if A**: $0
- **Cost if B**: ~$0.30 + delayed greenup
- **Operator stance** per takeover memo: "build everything ready for funding" suggests
  parallel preparation; both can proceed but greenup gates the SUBMISSION decision, not the
  ENGINEERING decision

### Sub-decision 3: Should A1 displace PR #107 apogee as our best frontier candidate?

A1 beats PR #107 on **both** axes (CPU +0.0038, CUDA +0.0030); both axes agree.
This is independent of sub-decisions 1+2.
- **Option A: Displace** — explicit acknowledgment in council memos + roadmap that A1 is
  the new local best; affects how future lanes phrase "frontier"
- **Option B: Do not displace** — keep PR #107 as the public-frontier reference for our
  submitted record; A1 sits as "frontier-claimed but unsubmitted" until sub-decision 1 fires
- **Default per prior council Insight 4 (10/10 READY)**: implicit displacement (A1 is the
  frontier-claimed candidate for internal planning; PR #107 remains the publicly-submitted
  reference until A1 is submitted)

## Entry packet inventory

`submissions/a1/`:

```
archive.zip                                 178,262 B  (single member 'x')
archive_manifest.json                       (member sha256 + sizes + CRC)
inflate.sh                                  599 B  (≤100 LOC budget honored)
inflate.py                                  3,333 B  (≤200 LOC budget honored)
src/codec.py                                16,734 B  (PR101 split-Brotli + latent sidecar)
src/model.py                                2,197 B  (HNeRV decoder class)
contest_auth_eval.cpu.json                  GHA Linux x86_64 adjudicated
contest_auth_eval.cuda.json                 Modal Tesla T4 retry3
dual_eval_adjudicated.json                  combined dual-axis record
report.txt                                  custody + dual-eval summary
pre_submission_compliance.contest_final.json  STUB (operator-trigger executes)
README.md                                   entry-packet README
DEVICE_AXIS_EXPLANATION.md                  D5 expansion #1
MECHANISM_ATTRIBUTION.md                    D5 expansion #2
RD_DERIVATION.md                            D5 expansion #3
```

5-turn workflow template:
- `.omx/research/a1_pr_submission_5_turn_greenup_workflow_template_20260511.md`

## Custody verification

All SHA-256 anchors match the dual-eval adjudicated source:

| Field | Value | Verified against |
|---|---|---|
| `archive_sha256` | `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` | ✓ `dual_eval_adjudicated.json` + `feedback_a1_dual_cuda_dispatch_landed_20260509.md` |
| `archive_size_bytes` | `178,262` | ✓ same |
| `runtime_tree_sha256` | `89db4fe14ac2bbffc951f8e89ac2242fa1455e0880bb3fbe963aa48e4890b5eb` | ✓ canonical from CUDA eval's `inflate_runtime_manifest` |
| `inflate.sh sha256` | `32c487cd1a48a2c80a964e252286971da8edf6f4946a832bc182029be2476af1` | ✓ direct shasum |
| `inflate.py sha256` | `2d2b97f2e59c50135b77e0335ed91807ce0fa7de223c55512d34b1be6f817a95` | ✓ direct shasum |
| `src/codec.py sha256` | `637fa5e4b47bfb2595358903dfaafbbc7a7ca48d5f7ccd81c06d1397e060bb72` | ✓ direct shasum |
| `src/model.py sha256` | `e63b04ad3df4942b9bc1e31afd8ec84177dfbe83827f67cf7c5a682b05c1b46b` | ✓ direct shasum |
| `upstream/evaluate.py sha256` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` | ✓ direct shasum of local upstream/evaluate.py |
| CPU canonical_score | `0.19284757743677347` | ✓ `contest_auth_eval.cpu.json` |
| CUDA canonical_score | `0.22635202347843951` | ✓ `contest_auth_eval.cuda.json` |

## Sanitization audit

Per CLAUDE.md "Public Disclosure Hygiene":
- ✓ No operator-local absolute paths in `inflate.sh`, `inflate.py`, `src/codec.py`, `src/model.py`
- ✓ No provider secrets (Modal/Lightning/Vast.ai credentials) in any committed file
- ✓ No `/tmp/` paths in README.md, report.txt, or D5 expansion markdown files
- ⚠ `contest_auth_eval.cuda.json` references `/tmp/modal_auth_eval/...` as Modal-side scratch
  (forensic-only; not propagated to the README or report.txt as durable evidence). Per CLAUDE.md
  FORBIDDEN_PATTERNS `forbidden_transient_tmp_paths_in_persisted_artifacts`, the README explicitly
  declines to cite those `/tmp` paths as reconstructable custody — they are eval-side scratch.
  Pre-submission compliance check `--public-scan-path` argument SCOPES the public scan to
  release-surface files only, matching the R2 reference packet pattern.

## Cost to operator (next step)

| Action | Cost | Time |
|---|---:|---|
| Trigger 5-turn greenup | **$0** | 1-5 parent-agent sessions |
| Execute pre-submission compliance check | **$0** | ~10s local |
| Submit PR (if greenup ratifies + operator approves) | **$0** | ~5 min manual gh + GitHub UI |

**Total cost from here to a submitted A1 PR**: $0 GPU + $0 LLM (the $0.80 dual-CUDA spend
was already paid 2026-05-09; the compliance check and greenup are zero-cost).

## What this packet does NOT do

- It does NOT initiate the 5-turn greenup. Operator-trigger required.
- It does NOT execute `pre_submission_compliance_check.py`. Operator-trigger required.
- It does NOT submit a PR. Operator-trigger required after greenup completes.
- It does NOT make any new score claim outside the dual-eval custody.
- It does NOT change loop pause status. Loop remains PAUSED.
- It does NOT consume any GPU credits or LLM cost above the parent-agent session that staged it.

## Loop pause status

**PAUSED** per 2026-05-09 directive. This entry packet preparation is $0 work and
changes nothing about pause status. Operator-trigger required for the next step.

## Cross-references

- Entry packet README: `submissions/a1/README.md`
- 5-turn workflow template: `.omx/research/a1_pr_submission_5_turn_greenup_workflow_template_20260511.md`
- A1 dual-eval landing: `feedback_a1_dual_cuda_dispatch_landed_20260509.md`
- N grand council D5 verdict: `feedback_grand_council_5_design_decisions_review_20260511.md`
- Prior pose-axis council Insight 4: `feedback_grand_council_pose_axis_insights_review_20260511.md`
- NOT YET ITEM 1: `project_top_priority_revisit_NOT_YET_operator_decisions_20260509.md`
- Device-axis matrix: `.omx/research/device_axis_paired_anchor_matrix_20260511.md`
- CLAUDE.md "Submission PR gate" + "Submission auth eval — BOTH CPU AND CUDA" + "Frontier target"
