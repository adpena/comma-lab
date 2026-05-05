# Lane 12 NeRV full-CUDA dispatch RETRY also REFUSED — same blockers, premise was wrong

**Status:** REFUSED dispatch (no GPU spend, no instance provisioned)
**Subagent:** claude:subagent-nerv-full-cuda-retry
**Date:** 2026-05-02 ~13:05 UTC
**Pinned commit:** `02fb6297c61820d7a69300302d651f76392248ea`
**Prior refusal:** `project_lane_12_nerv_full_cuda_dispatch_REFUSED_20260502.md` (2026-05-02 ~12:55 UTC, 156 lines)
**Anchor under stack consideration (unchanged):** C-067, sha `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, 276,214 bytes, score 0.31561703 [contest-CUDA T4 A++]
**Cross-refs:** dispatch claim row appended to `.omx/state/active_lane_dispatch_claims.md` 2026-05-02T13:05:00Z; CLAUDE.md non-negotiables (Score-target, Forbidden-score-claims, Comment-only-contracts FORBIDDEN, PCC4 council-review-required, Cross-agent dispatch coordination); reactivation criteria from prior refusal still binding.

## TL;DR

The retry mandate stated: *"Prior dispatch (subagent ae0392e705de35098, 2026-05-02 ~07:30 UTC) returned 'Monitor will notify when done' but produced ZERO visible artifacts (no Vast instance, no claim ledger row, no result dir, no commits). Likely failure: H100 SXM unavailable + no fallback attempted."*

That premise is **factually incorrect**. The prior dispatch:

1. Did produce a claim ledger row (`.omx/state/active_lane_dispatch_claims.md` row at 2026-05-02T12:55:00Z with status `refused_dispatch_handoff_to_user`)
2. Did produce a memory file (`project_lane_12_nerv_full_cuda_dispatch_REFUSED_20260502.md`, 156 lines)
3. Did NOT silently fail at chip provisioning — it explicitly REFUSED at pre-flight after analyzing 4 structural blockers
4. Did NOT cost any GPU spend ($0)

I verified at HEAD `02fb6297` that **NONE** of the prior-refusal blockers have been resolved between 2026-05-02 ~12:55 UTC and now (2026-05-02 ~13:05 UTC, ~10 min later). Re-dispatching with the same infrastructure into the same predicted-regression scenario violates CLAUDE.md "Score target — NON-NEGOTIABLE" + "Forbidden score claims" + the user's own "do NOT cycle hyperparameter tuning" mandate.

## Blocker re-verification (2026-05-02 ~13:05 UTC at HEAD 02fb6297)

| Blocker | Prior status (12:55 UTC) | Current status (13:05 UTC) | Source-of-truth check |
|---|---|---|---|
| `.omx/state/lane12_nerv_l2_clearance.json` exists | MISSING | MISSING | `ls /Users/adpena/projects/pact/.omx/state/lane12_nerv_l2_clearance.json` → "No such file or directory" |
| `_try_parse_public_pr67_NERV_qzs3_qp1_payload` parser exists in `unpack_renderer_payload.py` | NOT IN FILE | NOT IN FILE | `grep -n "_try_parse_public_pr67_NERV\|NERV_qzs3_qp1" submissions/robust_current/unpack_renderer_payload.py` → 0 matches; only existing reference is line 54 string literal `"masks.nrv"` |
| `ALPHA_PRIMITIVE_CONTRACT` for C-067 stack | MISSING | MISSING | No `alpha_geo_primitive_contract_v1` JSON exists for Lane 12 C-067 stack variant in `.omx/state/` |
| `ALPHA_GEO_PROVENANCE` with `pass_fail.overall_pass=true` | MISSING | MISSING | No `alpha_geo_0_nerv_geometry` JSON exists for the rebuilt-NeRV-C067 archive SHA |
| `POSE_REGEN_PROVENANCE` for the candidate mask stream | MISSING | MISSING | Candidate archive doesn't exist yet, so no pose-regen run is possible |
| Phase F empirical updated (≥1 NeRV checkpoint with measured disagreement < 0.05%) | 2.0% @ 1400 CPU steps | 2.0% @ 1400 CPU steps | `cat /Users/adpena/projects/pact/reports/lane_12_nerv_real_archive.json` → unchanged: `final_argmax_disagreement_vs_av1_source: 0.020032391018337675`, `training_steps_run: 1400`, `training_seconds: 228.7` |
| Council deliberation approving recipe redesign | NONE | NONE | No new memory file under `~/.claude/projects/-Users-adpena-Projects-pact/memory/` matching `*council*nerv*` or `*council*lane_12*` since the 2026-05-02T12:55 refusal |

## Convergence math (unchanged)

From `reports/lane_12_nerv_real_archive.json`:
- xent loss curve: 0.59 → 0.02 in 1400 CPU steps
- argmax disagreement: 2.0% at step 1400
- Linear extrapolation to 60K steps: ~0.5% final disagreement
- Distortion penalty `100 × 0.005 = +0.5` score swamps `-0.130` rate save

**Predicted stacked score (linear extrapolation, realistic):**
`0.31561703 - 0.130 + 0.5 = ~0.685` → REGRESSION vs C-067 0.316

**Predicted stacked score (optimistic 0.05% disagreement):**
`0.31561703 - 0.130 + 0.05 = ~0.236` → SUB-FRONTIER, but requires recipe change (not just more training steps)

The optimistic case is NOT achievable by running Phase F's recipe longer — Phase F's xent floor projects to ~0.005, which maps to ~0.5% disagreement, NOT 0.05%. Pushing past the 0.05% threshold requires architectural / loss / data-augmentation changes (Path A in prior refusal: larger NeRV, hard-pair mining, KL-distill against SegNet logits). All of those are design decisions requiring council deliberation per CLAUDE.md "Design decisions — non-negotiable".

## Vast.ai supply check (2026-05-02 ~13:05 UTC)

Retry mandate specified offer 27652327 (A100 SXM4 80GB at $1.30/hr).

`vastai search offers 'gpu_name=A100_SXM4 reliability>0.95 cuda_vers>=12.4 num_gpus=1 dph<2.0' -o 'dph'` returned 6 offers:

| # | offer ID | model | VRAM | $/hr | NV driver | reliability |
|---|---|---|---:|---:|---|---:|
| 1 | 31632914 | A100 SXM4 | 41 GB | $0.6676 | 570.133.20 | 99.9 |
| 2 | 36007740 | A100 SXM4 | 41 GB | $0.7089 | 565.57.01 | 99.7 |
| 3 | 36007762 | A100 SXM4 | 41 GB | $0.7369 | 565.57.01 | 98.6 |
| 4 | 31179882 | A100 SXM4 | 41 GB | $0.8481 | 580.95.05 | 97.1 |
| 5 | 29296806 | A100 SXM4 | 81.9 GB | $1.0543 | 580.95.05 | 99.8 |
| 6 | 26307909 | A100 SXM4 | 81.9 GB | $1.7801 | 570.195.03 | 99.3 |

Offer 27652327 NOT IN supply. Cheapest 80GB offer is 29296806 @ $1.05/hr ($8.40 for 8h, well under $20 cap). Cheapest A100 SXM4 is 31632914 @ $0.67/hr (40GB, suitable for NeRV's tiny memory budget).

**Chip supply is NOT the blocker.** Even with multiple suitable offers available, dispatching them solves zero of the structural blockers above.

## Why NOT to dispatch anyway (CLAUDE.md citations)

1. **"Score target — NON-NEGOTIABLE, HIGHEST EMPHASIS":** "Any auth score above 1.0 is UNACCEPTABLE. Do the math during training. If projected auth > 1.0, something is wrong — stop and fix it before burning more GPU hours." The 0.685 predicted-realistic case isn't above 1.0 but IS above the current frontier (C-067 0.316); the same fail-fast logic applies — if the math predicts a regression, don't burn the GPU.
2. **"Forbidden score claims":** subagent cannot ship a `[contest-CUDA]` score from a $5-20 GPU run that produces a regression and quietly burn the budget. The handoff must surface the prediction BEFORE spend, which is exactly what this refusal does.
3. **"Comment-only contracts — FORBIDDEN":** fabricating L2 clearance JSON / Alpha-Geo contract JSON / pose-regen provenance JSON to bypass `remote_lane_nerv.sh` gates would be the canonical example of this bug class.
4. **"KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE" + PCC4 (`check_kill_memory_files_have_council_review`):** any verdict requires explicit council deliberation. Fabricating "cleared_for_retraining_unblock=true" without a real council session is exactly the bug class PCC4 was landed to extinct.
5. **"Council conduct + Design decisions — non-negotiable":** changing the NeRV training recipe (architecture, loss, ground-truth target, hard-pair mining) is a design decision requiring inner-quintet sign-off. Subagent has no authority to redesign the recipe.
6. **User mandate "do NOT cycle through hyperparameter tuning":** dispatching at 60K steps and discovering 0.5% floor → trying 80K steps with hidden=80 → trying KL-distill weighting is exactly the pattern the user forbade.
7. **"Cross-agent dispatch coordination — NON-NEGOTIABLE":** the prior refusal row is the active claim. Re-claiming the same lane within 10 minutes without resolving the documented blockers violates the coordination ledger's purpose.

## What the retry mandate got right vs got wrong

**Right:**
- A100 SXM4 80GB is the correct chip class (per `feedback_q_faithful_4090_preemption_redeploy_h100_20260501.md`). Avoiding RTX 4090 is correct.
- Cost cap $20 is reasonable for an 8h run on $1.05/hr A100 SXM4.
- Determinism spec (pinned commit, profile, seed, deterministic CUDA, provenance JSON, EMA decay 0.997, CUDA-only score truth) is correct and matches prior refusal's analysis.
- "DO NOT silently exit" + "explicit failure report mandatory" is correct — and is exactly what the prior subagent did and what this retry subagent is doing.

**Wrong:**
- "Prior dispatch... produced ZERO visible artifacts" is false. Prior subagent produced a claim row + 156-line memory file + the refusal report.
- "Likely failure: H100 SXM unavailable + no fallback attempted" is false. Prior subagent did not fail at chip provisioning. It refused at pre-flight after analyzing 4 structural blockers, before reaching `vastai create`.
- "Use offer 27652327 explicitly" — that offer is no longer in Vast.ai supply (10 min later, supply has rotated; this is normal Vast behavior).

## What I did NOT do (intentionally, same as prior subagent)

- Did NOT call `vastai create instance` (no GPU spend, $0 charged)
- Did NOT modify `unpack_renderer_payload.py` to add a NeRV parser (out of mandate scope; would be Path B work, requires council)
- Did NOT fabricate L2 clearance / Alpha-Geo contract / pose-regen JSONs to bypass the gates (CLAUDE.md PCC4 + Comment-only-contracts forbidden)
- Did NOT modify `scripts/remote_lane_nerv.sh` to remove the gate checks (defeats the purpose of the gates)
- Did NOT run `experiments/train_nerv_mask.py` locally with `--device cpu` (banned per "MPS-falsification" + "advisory only" rules for any score-relevant decision)
- Did NOT spawn additional subagents
- Did NOT run any `kill` / `unmark` mutations on `tools/lane_maturity.py`
- Did NOT bypass review-tracker via `REVIEW_GATE_OVERRIDE=1`

## Reactivation criteria (UNCHANGED from prior refusal)

This refusal is REVERSED if any of the following lands BEFORE the next dispatch attempt:

- A council session approves Path A recipe redesign with a forecast band that includes <0.05% disagreement, supported by adjacent-art evidence (e.g., NeRV literature on similar boundary-segmentation tasks)
- Path B infrastructure work lands at HEAD with passing tests and an L2 clearance JSON signed off by a 3-pass council review (parser + Alpha-Geo contracts + pose-regen provenance for C-067 stack variant)
- An empirically-stronger NeRV checkpoint surfaces (e.g., from a parallel codex partner running Path A) with measured disagreement <0.05% on the C-067 PR67 mask source — then dispatch becomes "rebuild archive + eval", $2-3 of GPU
- User explicitly overrides with annotation in this file: "dispatch anyway, accept regression risk, $X GPU budget approved"

## Lane registry impact

Lane 12 stays at current Level (Level 2 INTEGRATION, per prior refusal + `tools/lane_maturity.py audit`). This refusal does NOT regress maturity. The blockers documented in the prior refusal are infrastructure / recipe gaps; nothing has changed.

## Cost honesty

- Prior refusal cost: $0 GPU, ~25 min subagent wall-clock
- This retry refusal cost: $0 GPU, ~10 min subagent wall-clock (faster because most of the analysis is in the prior memory file)
- Avoided cost (had I dispatched anyway): $5-20 GPU on a likely-regression run + 6-8h wall-clock + the cost of explaining a worse-than-baseline score
- Structural value: the next subagent or codex partner reading the dispatch claim ledger will see TWO refusal rows for the same lane on the same day, making it unambiguous that this lane needs human-loop unblocking before any further dispatch attempt

## Recommended next action for user

The user mandate "extreme rigor + deterministic reproducibility + push for sub 0.3 even at high variance + parallel both" applies, but the "push for sub 0.3" half cannot be exercised through Lane 12 NeRV without resolving the structural blockers first. Recommended sequencing:

1. **C-067 submission packet preparation** (the "parallel both" main-thread half) continues unaffected. C-067 0.316 is the safe submission.
2. **Lane 12 unblock requires a prior step** — either a council session for Path A (recipe redesign) OR a Path B code-first work session (parser + Alpha-Geo contracts) OR Path C re-aim at Lane G v3. None of these is a 1-shot subagent dispatch; they're either council deliberation or 2-6h coding work with tests + 3-pass review before dispatch.
3. **Alternative sub-0.3 paths** that don't require Lane 12 NeRV unblock: Block-FP transplant, SJ-KL residual, line-search refinement, or other lanes already on the C-067 stack roadmap. The user's offered alternatives in the dispatch mandate ("Block-FP transplant, SJ-KL residual, line-search refinement, OR ship C-067") are all valid and don't carry the Lane 12 structural blockers.

## Files referenced

- Prior refusal: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_lane_12_nerv_full_cuda_dispatch_REFUSED_20260502.md`
- Dispatch plan: `~/.claude/projects/-Users-adpena-Projects-pact/memory/project_lane_12_nerv_dispatch_plan_20260430.md`
- Phase F empirical: `/Users/adpena/projects/pact/reports/lane_12_nerv_real_archive.json`
- Target codec: `/Users/adpena/projects/pact/src/tac/nerv_mask_codec.py`
- Training entry: `/Users/adpena/projects/pact/experiments/train_nerv_mask.py`
- Deploy runbook (with gates): `/Users/adpena/projects/pact/scripts/remote_lane_nerv.sh`
- Inflate path (NeRV decoder): `/Users/adpena/projects/pact/submissions/robust_current/inflate_renderer.py`
- Inflate path (PR67 single-blob parser, no NeRV slice): `/Users/adpena/projects/pact/submissions/robust_current/unpack_renderer_payload.py`
- C-067 anchor archive: `/Users/adpena/projects/pact/experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Dispatch claim ledger (this row + prior refusal row): `/Users/adpena/projects/pact/.omx/state/active_lane_dispatch_claims.md`
