# ddm_fa1 FlowAdam Crosswalk Receipt

status: COMPLETE-NO-LAUNCH
arm: ddm_fa1
charter: .omx/tmp/codex_runs/fa1_prompt.md
common_contract: .omx/tmp/codex_runs/_common_contract.md
utc: 2026-08-07T11:17:43Z
repo_head_at_start: 14eaeb81814e
budget: USD 0

## Authority And Boundaries

- This packet is a paper-to-apparatus crosswalk only. It did not launch training, dispatch remote work, occupy a scorer slot, run n600 scoring, build an archive, or mutate code.
- Exact contest frontier is unchanged. The live hot-state advisory row read during this arm was `S=0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; the borrowed contest pointer remains separately unmoved. This packet is not goal progress by itself.
- All paper claims below are external-public-paper evidence, not Pact measurements. They are only admitted into follow-on design when paired with a named local consumer and falsifier.
- No protected files were edited.

## Primary Paper Read

Source: FlowAdam: Implicit Regularization via Geometry-Aware Soft Momentum Injection, arXiv:2604.06652 v1, submitted 2026-04-08.

Paper loci used:

- Eq. 2-4: EMA difficulty signal and mode selection.
- Eq. 6: soft momentum injection.
- Eq. 7: velocity clipping.
- Eq. 8 / Proposition 1: descent property for the clipped gradient-flow component.
- Eq. 9 / Lemma 2: bounded deviation under soft injection.
- Tables I, V, VI, VII and Fig. 3: reported matrix/tensor/inverse-kinematics/Jester gains and compute-matched comparison.
- Section VI: limitations, including mini-batch trigger sensitivity, no formal full-hybrid convergence guarantee, ODE overhead, manual mode choice, and small-to-medium benchmark scope.

## Rigor Triage

| Paper claim | Locus | Grade | Local use allowed here |
|---|---:|---|---|
| Clipped gradient-flow component is descent-directed under smoothness assumptions. | Eq. 8 / Prop. 1 | DERIVED-SOUND | May justify a local replay metric that penalizes transition updates pointing against recent gradients. It does not prove the full switched optimizer converges. |
| Soft momentum injection is bounded and less discontinuous than hard replacement. | Eq. 6, Eq. 7, Eq. 9 / Lemma 2 | DERIVED-SOUND for the bound; PLAUSIBLE-UNVERIFIED for the performance ablation | May motivate a default-off soft stage-boundary blend. No FlowAdam numeric constants are adopted. |
| EMA difficulty detector can choose injection mode. | Eq. 2-4, mode table in method | PLAUSIBLE-UNVERIFIED | Observer-only comparison against our typed event sensors; not a control-plane replacement. |
| FlowAdam improves reported matrix/tensor/inverse-kinematics/Jester metrics. | Tables I, V, VI, VII, Fig. 3 | PLAUSIBLE-UNVERIFIED | External prior only; requires local update/state custody before any Pact optimizer A/B. |
| FlowAdam exhibits implicit regularization in low-rank factorization. | Fig. 2, Table I, discussion | PLAUSIBLE-UNVERIFIED | Lesson-only analogy unless tied to a named consumer in our DE-derived acceptance surface. |
| FlowAdam is a drop-in optimizer for current Pact witness training. | Paper as a whole plus Section VI limitations | SUSPECT for Pact | Not adopted. The paper itself flags trigger noise, overhead, manual mode choice, and limited benchmark scale; we have no local gradient/curvature receipt for this optimizer. |

## Recall Evidence

Sources read before deciding:

- Governing docs: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Charter packet: `.omx/tmp/codex_runs/fa1_prompt.md`, `.omx/tmp/codex_runs/_common_contract.md`.
- Memory registry query scope: frontier routing, hot state, #899/#904, required-component guards, lane rules.
- Paper source: arXiv abstract, HTML, and PDF for `2604.06652`.

Repo recall queries included:

- `rg -n "#518|#269|warm-start|warm_start|FlowAdam|soft injection|momentum injection|reset operator|beta2|Muon|Adam" .omx docs src experiments`
- `rg -n "#315|#344|NCDE|event-driven|plateau|typed exit|EMA|difficulty detector|mode switch|trajectory" .omx docs src experiments`
- `rg -n "QA83|factorized output head|rank-1|rank 1|low-rank pose|pose codec|FiLM|coupled|curvature|precondition|diagonal" .omx docs src experiments`
- `rg -n "#318|calibrate-by-DE|DE-derivation|implicit regularization|viscosity|KKT|variational|canonical_equations" .omx docs src`
- `find .omx -iname "*j3*" -o -iname "*j4*" -o -iname "*warm*" -o -iname "*gc15*" -o -iname "*event*" -o -iname "*tp1*"`

Relevant local findings beyond the charter seeds:

- `p0_resume_warmup_geometry_build_20260717.md` built `adam_v_variance_warmup_length_v1` and resume LR rewarmup/trigger widening; efficacy was explicitly default-off/unmeasured.
- `codex_findings_ddm_j3_366_fullrun_mode_ticket_reseal_20260723_codex.md` scoped the J3 regression to an exact four-step replay and preserved component-regression blocking.
- `codex_findings_ddm_j4_366_warm_start_reform_20260723_codex.md` localized the J3 failure to opening optimizer/admission policy and built J4 guardrails, but J4 was byte-identical over four steps and did not bank descent.
- `ddm_gc15_fresh_vs_warm_20260731.md` priced the reset-operator issue directly: zeroed Adam moments can create a large boundary effective-LR spike; `v <- v_prev` or scorer-derived second-moment priors were the locally named high-value knobs.
- `event_wirings_build_20260708.md`, the canonical index, and TP1 receipts show an existing local event apparatus: backstop gates, lane-nucleus/annulus/muon events, trajectory-derived stopping, and typed exits.
- QA83/QA84 and the canonical index already name coupled low-rank output, FiLM/rank collapse, rowband, and pose-codec surfaces. These are local coupling consumers, but they do not by themselves justify importing FlowAdam.
- The canonical equations inventory already contains warm-start, plateau-tail, trajectory-stopping, closed-scorer variational/KKT, and archive-reachability entries; FlowAdam's implicit-regularization leg therefore maps to a lesson unless a concrete consumer is opened.

What changed because of recall:

- Direct FlowAdam adoption was rejected. The local warm-start/reset corpus is more specific than the paper for current Pact boundary failures.
- The one admissible generalization is not "use FlowAdam"; it is a default-off stage-transition soft velocity/state blend with Pact-local replay falsifiers.
- The EMA mode detector is not adopted because Pact already has typed event gates and scorer-component exits. It can only be compared as an observer.

## No-Launch Receipt

- No full-n600 scorer job was started.
- No GPU or remote job was started.
- No archive was produced.
- No candidate score was claimed.
- Follow-ons are dispositioned in `NEXT_IF_RESUMED.md`.

## Commit Attempt Receipt

- Serializer command attempted with `REVIEW_GATE_OVERRIDE=1`, `--no-co-author`, explicit `--files`, `--base-content-sha256 ...=new`, and post-edit `--expected-content-sha256` guards for all three FA1 files.
- Outcome: blocked at `git add` with rc=128 before commit. Stderr class: `error: unable to create temporary file: Operation not permitted`; `CROSSWALK.md` failed to insert into the Git object database.
- Retry: after recording this blocker and recomputing post-edit hashes, a second serializer attempt failed with the same rc=128 object-database error.
- Post-failure index check: `git diff --cached --name-status` returned empty, and `git status --porcelain=v1 -- .omx/research/ddm_fa1_20260807` showed the packet as untracked only.
