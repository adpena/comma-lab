# Recursive adversarial greenup review - 2026-04-30

Owner: Codex, senior-engineer pre-deploy review.

Scope: PFP16 A++, OWV3/Fisher, H-V3, Lane 19, SA, HM-S/KL, Lane 12 alpha redesign, Modal/Lightning routing, paper/writeup, and harness DX.

Constraint honored: review only; no jobs dispatched.

Changed path from this review:

- `.omx/research/codex_recursive_adversarial_greenup_review_20260430.md`

Last updated: 2026-04-30T16:35:32Z.

## Source set reviewed

- Program/source docs: `CLAUDE.md` council/design sections and AGENTS onboarding instructions.
- Required memory files:
  - `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_grand_council_paradigm_shift_to_shannon_floor_20260430.md`
  - `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_codec_stacking_composition_canonical_orders_20260429.md`
  - `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`
  - `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_codex_theoretical_floor_brutal_20260429.md`
  - `/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/project_codex_shannon_floor_orchestration_20260430.md`
- Current Codex progress docs:
  - `.omx/research/codex_source_doc_structure_and_compliance_map_20260430.md`
  - `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
  - `.omx/research/council_paradigm_shift_round1_20260430.md`
  - `.omx/research/council_paradigm_shift_round2_20260430.md`
  - `.omx/research/council_paradigm_shift_round3_20260430.md`
  - `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
  - `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
  - `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
  - `.omx/research/shannon_floor_execution_readiness_20260430.md`
  - `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
  - `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`
  - `.omx/research/dx_self_protecting_harness_hardening_20260430_codex_progress.md`
  - `.omx/research/active_dispatch_harvest_status_20260430.md`
  - `.omx/state/active_dispatches.md`
- Lane-specific docs/runbooks/artifacts:
  - `.omx/research/pfp16_a_plus_plus_exact_t4_eval_runbook_20260430.md`
  - `.omx/research/council_lane_pfp16_round1_20260430.md`
  - `.omx/research/council_lane_pfp16_round3_20260430.md`
  - `docs/owv3_fisher_runbook.md`
  - `.omx/research/council_lane_19_logit_margin_round6_20260430.md`
  - `.omx/research/paradigm_alpha_mask_overhaul_audit_20260430.md`
  - `.omx/research/recoverable_lanes_re_engineering_plans_20260430.md`
  - `.omx/research/council_lane_12_nerv_design_20260430.md`
  - `reports/lane_pfp16_real_archive.json`
  - `reports/silent_defaults.md`
  - `reports/lane_maturity.md`
  - `.omx/state/cost_projection.md`
- Scripts/code spot-checked for promotion risk:
  - `scripts/pfp16_a_plus_plus_exact_t4_eval.sh`
  - `scripts/remote_lane_g_v3_owv3_fisher_stack.sh`
  - `scripts/remote_lane_h_v3_jointly_trained_halfframe.sh`
  - `scripts/remote_lane_19_logit_margin.sh`
  - `scripts/remote_lane_sa_segmap_clone.sh`
  - `scripts/remote_lane_hm_s_segmap_homography.sh`
  - `scripts/adjudicate_contest_auth_eval.py`
  - `scripts/launch_lane_with_retry.py`
  - `scripts/reconcile_vast_dispatch_state.py`
  - `scripts/launch_lane_lightning.py`
  - `src/tac/deploy/lightning/lightning_dispatch.py`
  - `experiments/modal_train_lane.py`
  - `experiments/train_segmap.py`
  - `src/tac/profiles.py`
  - `src/tac/training.py`
  - `src/tac/losses_logit_margin.py`
  - `src/tac/experiments/train_renderer.py`

## Online research intake used

These sources guided design constraints only. They do not promote or kill a lane without local exact contest eval.

- Boyle and Dykstra's projection framing: Dykstra-style alternating projections target the projection of a point onto an intersection of closed convex sets, which supports using Dykstra as a feasibility/frontier discipline rather than as evidence of global optimum on the contest's discrete codec ladders. Source: https://www.scirp.org/reference/referencespapers?referenceid=1695111
- Boyd et al. ADMM survey: ADMM is appropriate for decomposed convex optimization and large-scale statistical-learning problems, but the contest stack still needs discrete projection, restarts, and exact archive evaluation because component R(D) curves are nonconvex and sampled sparsely. Source: https://www.nowpublishers.com/article/Details/MAL-016
- Ballé, Laparra, Simoncelli 2017: learned compression is trained against a rate-distortion objective `R + lambda D`, with rate represented through entropy/log-likelihood of quantized latents. This supports measuring every proposed neural codec in bits plus scorer distortion, not visual plausibility. Source: https://openreview.net/forum?id=rJxdQ3jeg and https://www.cns.nyu.edu/pub/Lcv/balle17a-final.pdf
- Ballé et al. 2018 scale hyperprior: hyperpriors are side information for capturing latent spatial dependencies; in this contest, that side information must be charged inside `archive.zip`. Source: https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/
- CompressAI docs: practical entropy-bottleneck and Gaussian-conditional models require an `update()` step before actual entropy coding; this reinforces that learned-prior experiments are not score evidence until they produce byte-exact coded streams. Source: https://interdigitalinc.github.io/CompressAI/models.html
- Constriction entropy-coding docs: ANS/range coders can be near entropy limits, but only after a real discrete symbol model exists; arithmetic/ANS must remain terminal, after quantization/symbolization. Source: https://bamler-lab.github.io/constriction/
- HAWQ-V3: mixed-precision quantization solves bit precision under model-perturbation and hardware constraints, supporting OWV3's score-sensitivity-weighted mixed precision as a plausible beta design, not yet evidence. Source: https://proceedings.mlr.press/v139/yao21a.html
- AWQ: protecting salient channels/weights can preserve quality under low-bit quantization, but the contest version must define salience through PoseNet/SegNet score sensitivity rather than generic activations. Source: https://mlsys.org/virtual/2024/poster/2653

## Evidence standard

A result is deploy-green only if all of these are true:

1. Exact archive SHA, archive bytes, and archive manifest are recorded.
2. All neural/runtime payloads needed by `inflate.sh` are inside `archive.zip`; no score-affecting sidecars.
3. Evaluation path is canonical: `inflate.sh` then upstream scorer/evaluator, with no scorer patching.
4. CUDA exact evaluation is recorded with `n_samples=600`; T4 is required for A++ claims.
5. `final_score` recomputes from component terms; use JSON adjudication, not log regex alone.
6. Inflation is within the 30-minute contest budget and the evidence is co-located with the archive.
7. Source provenance is sufficient to rebuild or explain the archive; staged non-git trees must be supplemented by a source manifest.

CPU, MPS, proxy, byte-only, or Modal CPU auth evals are useful diagnostics only. They are not promotion evidence.

## Exact contest-compliant eval gates

These are the exact gates for any result that wants to rank, promote, kill, or deploy:

1. Archive custody: preserve the exact evaluated `archive.zip`, byte count, SHA-256, member manifest, member hashes when available, and `zipinfo`/permissions metadata. Neural artifacts and postfilters must be archive members or fixed contest code; no local sidecars.
2. Manifest compliance: no `.DS_Store`, `__MACOSX`, debug checkpoints, hidden caches, stale payloads, zip-slip paths, or external score-relevant files. Required lane-specific payloads must be present and named in the manifest.
3. Eval path: `archive.zip -> inflate.sh -> upstream/evaluate.py`, or the canonical `experiments/contest_auth_eval.py` wrapper proving that path. No upstream scorer edits, renderer shortcuts, patched models, or alternative local scoring.
4. Device/sample gate: CUDA exact eval with `n_samples=600`. A++ requires T4 or documented contest-equivalent hardware, plus `gpu_t4_match=true` or an equivalent evidence field.
5. Inflate gate: raw output cardinality, frame count, geometry-derived byte size, and elapsed inflate time <= 1800 seconds on contest-equivalent hardware.
6. Score adjudication: recompute `100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489` from JSON components. `final_score` is rounded display only. `remote_provenance.json` never overrides `contest_auth_eval.json`.
7. Provenance gate: co-locate `contest_auth_eval.json`, adjudicator output, inflate logs, scorer/report logs, archive manifest, source commit or staged-tree manifest, upstream commit/hash, command line, and hardware metadata.
8. Review gate: zero unresolved findings after the recursive adversarial review specified below.

## Cross-document reconciliation

- The exact PFP16 T4 artifact is the controlling current frontier: score `1.043987524793892`, archive `686635` bytes, SHA `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Older narrative docs still describe the old `1.33` / Grade A baseline or say preserved lanes are not A++. Those docs are stale relative to the exact T4 artifact and must be synchronized before paper or deploy signoff.
- Lane 12 NeRV `jsonfix40` is retired for the current measured implementation/config by exact CUDA evidence: score `26.03719330455429`, PoseNet `49.7784996`, archive `296478` bytes. It should not consume more execution time without a redesigned alpha objective.
- Active remote ledgers are not authoritative. Reconciliation showed live jobs missing from the active table and stale active rows missing from live API state. Promotion must be based on lane-local artifacts, exact JSON, and archive hashes.

## Dykstra ceiling constraints

The Dykstra ceiling is a necessary feasibility constraint, not a promotion claim:

1. Rate-only sub-0.30 ceiling: `bytes <= floor(0.30 * 37545489 / 25) = 450545`. Any archive above `450545` bytes cannot produce score < `0.30`, even with zero SegNet/PoseNet distortion.
2. PFP16 is still far outside that ceiling: `686635` bytes gives rate contribution `25 * 686635 / 37545489 ~= 0.4572`, before distortion. It is deploy-green as the current frontier, but it is not a Shannon-floor architecture.
3. Current PFP16 non-rate score is already about `0.5868` (`100 * 0.00400656 + sqrt(10 * 0.00346442)`). Therefore bytes alone cannot reach sub-0.30; the path needs both alpha-style representation change and beta-style distortion preservation.
4. Dykstra/ADMM stack claims must be evaluated against the intersection of feasible sets: rate <= budget, SegNet <= target, PoseNet <= target, archive manifest <= contest rules, and inflate time <= 1800 seconds.
5. Additive deltas are conditional. Two independent-looking byte savings cannot be added until each component has a standalone exact eval and the stacked archive has its own exact eval. The convex-hull relaxation is only a guide; the discrete archive is the evidence.
6. ADMM is a coordinator after measured components exist. It is blocked before OWV3/Fisher or a redesigned alpha stream has exact evidence, because otherwise it only optimizes predictions.

## Bad-result suspicion policy

Every surprising result, especially a bad result, starts as suspicious until classified. The purpose is to avoid both false kills and false promotions.

1. Preserve first, interpret second: copy the exact archive, `contest_auth_eval.json`, logs, manifest, SHA, and provenance before any cleanup.
2. Recompute the score from components and verify the report used CUDA, 600 samples, and the canonical inflate/eval path.
3. Classify the failure as one of: legitimate regression, harness/eval bug, archive/manifest bug, no-op or encode-discard bug, config/dead-flag bug, CPU/MPS/proxy leakage, sidecar dependency, codec attribution confound, KL/PoseNet collapse, data geometry mismatch, timeout/NVDEC infrastructure, or indeterminate.
4. Scope retirements conservatively: a single exact bad result can retire the measured implementation and configuration. It kills a lane family only after independent exact reproductions or a mathematical impossibility argument plus clean Grand Council consensus. It never kills a broader paradigm from byte-only, CPU, smoke, or crashed evidence.
5. Treat bad good news the same way: a surprisingly low score is not trusted until archive custody, sidecar closure, scorer integrity, and reproduction/adjudication pass.
6. For disappointing or unexpected results, run a mitigation/stacking pass before retirement: identify whether the component can be rescued by side-info accounting, pose/seg guard losses, hybrid residuals, fallback routing, per-region gating, or stacking with PFP16/SA/H-V3/OWV3 alternatives.
7. Run a leaderboard-reverse-engineering pass before broad conclusions: compare archive member sizes, likely representation families, raw-output geometry, mask/pose/render stream allocation, and known Quantizr/Selfcomp-style full-stack patterns. A result may fail standalone while remaining useful as a subcomponent in a different full-stack allocation.
8. For current cases: Lane 12 `jsonfix40` retires the current NeRV mask replacement configuration after scope review, not all alpha mask compression; HM-S/KL bad evidence would not kill homography until KL, packaging, and codec confounds are separated; Lane 19 cannot prove logit-margin value while KL aux and packaging drift remain unresolved.

## Active lane blocker matrix

| Lane | Current evidence | Adversarial blockers | Exact greenup gate | Verdict |
| --- | --- | --- | --- | --- |
| PFP16 A++ | Exact T4 eval exists at `experiments/results/lane_g_v3_pfp16/pfp16_a_plus_plus_t4_20260430T1620Z_codex/contest_auth_eval.json`. Score recomputes to `1.043987524793892`; archive SHA `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`; archive size `686635`; `n_samples=600`; T4 provenance recorded; strict raw validation passed. Zip manifest contains `renderer.bin`, `masks.mkv`, `optimized_poses.bin`. | Source docs are not fully synchronized to A++. JSON has no explicit `inflate_duration_seconds`; the log proves budget (`elapsed=87.0s`, internal inflate `39.8s`) but the structured field is null. `eval_provenance.json` reports a `pact_commit` error because the staged Lightning tree was non-git. | Freeze this archive as current deploy baseline. Before final deploy/paper: attach `auth_eval.log`, `report.txt`, `eval_provenance.json`, manifest, SHA/bytes, upstream commit, and either source commit or staged-tree manifest. Update source audit docs to say PFP16 is current A++ frontier. | GREEN as baseline; not blocked for technical score, blocked only on documentation/provenance polish. |
| OWV3 / Fisher | Runbook and scaffold exist. Script has CPU-eval guard by default. No promotion-grade artifact found. Vast attempts failed NVDEC per progress docs. | No CUDA Fisher/Hessian artifact, no `hessian_per_weight.pt`, no sensitivity map, no train/holdout stability proof, no exact archive, no exact eval. Modal wrapper forces CPU auth eval, so Modal results cannot promote. Missing-layer handling cannot silently zero out scientific evidence. | Produce CUDA Fisher/Hessian per-channel artifact with calibration/holdout split, stability/CV threshold pre-registered, protected layers listed, and missing layers explicitly failed or justified. Build archive, run exact CUDA/T4 eval, record JSON/manifest/SHA. Must beat PFP16 `1.043987524793892` and keep PoseNet within a pre-registered regression limit, recommended <=20% relative to PFP16. | RED until Fisher evidence and exact eval exist. |
| H-V3 | Active/live per progress/reconcile: `lane_h_v3_joint_halfframe_2026-04-30_codex_a4` / instance `35907873`. Script uses JSON adjudicator and deterministic `ZipInfo`; regression threshold near `1.05`. | No lane-local exact eval yet. Script calls `nvidia-smi` before Stage 0 NVDEC, violating the current deploy-script hygiene rule. If `zoom_scalars.pt` is missing, script warns and ships identity zoom; that confounds attribution. Regression target must compare to PFP16 `1.043987524793892`, not the older `1.05` frontier. | Harvest only. Greenup requires lane-local `RESULT_JSON`, `contest_auth_eval.json`, CUDA/T4 evidence, n=600, manifest, SHA/bytes, and recorded presence of intended `zoom_scalars.pt` or explicit identity-zoom attribution. Promote only if exact score < PFP16 and PoseNet/SegNet terms do not hide a collapse. | YELLOW active-watch; no deploy without exact eval. |
| Lane 19 | Council Round 6 says 3/3 clean and cleared for deployment. Active/live likely `lane_19_logit_margin_2026-04-30_b`, but active ledger instance is stale. | Script still packages with `z.write(...)` and source mtimes, so archive is not deterministically normalized. It does not call `scripts/adjudicate_contest_auth_eval.py`; it greps `RESULT_JSON` and does not reliably copy lane-local `contest_auth_eval.json`. It calls `nvidia-smi` before Stage 0 NVDEC. Documentation says KL is disabled, but `LANE_19_LOGIT_MARGIN` inherits `kl_distill_weight=0.002` and `train_renderer.py` runs KL aux whenever the weight is >0. Script comments mention stale margin weight `0.1` while profile uses `10.0`. Scientific attribution is therefore confounded. | Before redeploy or promotion, reconcile intent: either explicitly keep KL aux and retag the lane, or disable KL and rerun under that claim. Fix deterministic archive packaging and JSON adjudication. Promotion requires exact CUDA/T4 eval, manifest/SHA, n=600, and score < PFP16 with no PoseNet collapse. | RED for promotion despite active run; YELLOW only as harvestable diagnostic. |
| SA | Active/live per reconcile: `lane_sa_segmap_clone_2026-04-30_codex_a2` / instance `35906669`; progress says stage 2 training. Plain variant uses standard loss/no KL. Script uses adjudicator and deterministic `ZipInfo`; neural payload intended inside archive. | No exact eval yet. Script calls `nvidia-smi` before Stage 0 NVDEC. Its adjudication thresholds are still older-baseline oriented (`1.15` / `1.30`); frontier promotion must compare to PFP16. | Harvest only. Greenup requires archive manifest containing `segmap_weights.tar.xz`, `grayscale.mkv`, and `optimized_poses.pt`, plus config proving `PYTHON_INFLATE=segmap`. Exact CUDA/T4 eval must beat PFP16 and have no sidecars. | YELLOW active-watch; potentially useful, not deployable yet. |
| HM-S / KL | Active/live per reconcile: `lane_hm_s_2026-04-30_b_a2` / instance `35885106`; progress says KL-distill homography variant heartbeat is fresh. `train_segmap.py` scopes this to SegNet aux for non-plain variants. | Any KL-like objective is high-risk under project lessons until exact CUDA proves no PoseNet collapse. Script lacks JSON adjudicator, lacks deterministic `ZipInfo`, and only leaves log evidence. It encodes grayscale through PyAV yuv420p with chroma=128 rather than the ffmpeg monochrome gray path used by SA, introducing a rate/codec confound. It calls `nvidia-smi` before Stage 0 NVDEC. | Treat as forensic unless it returns a strong exact score. Even then, promotion requires deterministic rebuild, JSON adjudication, manifest/SHA, n=600 CUDA/T4 eval, and explicit PoseNet non-collapse. If bad, do not conclude homography failed; separate KL, packaging, and codec confounds. | RED for deploy; harvest as diagnostic only. |
| Lane 12 alpha redesign | Exact CUDA regression exists for current `jsonfix40`: `experiments/results/lane_12_nerv_20260430_codex_jsonfix40/contest_auth_eval.json`, score `26.03719330455429`, PoseNet `49.7784996`, SegNet `0.03528685`, archive `296478`, n=600, CUDA RTX 4090. | The prior 2% argmax disagreement / byte-saving argument was not a sufficient objective. Failure is geometric/temporal/PoseNet preserving, not just mask-byte compression. More NeRV spend without diagnosis repeats the same error. | New alpha objective must pre-register a PoseNet-preserving mechanism: e.g. geometry-preserving grayscale LUT/SegMap, nearest-K or warped masks, wavelet residual/VQ with pose guard, or boundary-weighted NeRV only after failure diagnosis. Exact gate: CUDA n=600, archive < baseline mask bytes only if PoseNet stays within <=1.2x PFP16 and SegNet does not regress by >25%, unless explicitly labeled paper-only diagnostic. | RED for current NeRV implementation; redesign only. |
| Modal / Lightning routing | Lightning exact path produced PFP16 A++ when run from a hermetic staged tree. Modal and Lightning wrappers exist. Vast reconciliation tooling exists. | Generic Lightning dispatcher uses `set -uo pipefail` rather than `set -euo pipefail`, harvests nested JSON via regex, and looks for `score`/`total` instead of canonical adjudicated fields. Modal wrapper stubs NVDEC and forces `AUTH_EVAL_DEVICE=cpu`, so Modal auth eval is not promotion-grade. Vast state is drifted: live API and active ledger disagree. | Use Lightning for exact CUDA/T4 promotion evals with hermetic source manifest. Use Modal only for build/Fisher/smoke/ablation with `RUN_CONTEST_EVAL=0`, then rerun archive through Lightning exact eval. Before generic routing is green, add fail-fast shell behavior, JSON artifact copy, adjudicator use, GPU-tier recording, and no regex scoring. | YELLOW for exact PFP16-style hand-run path; RED for generic promotion router. |
| Paper / writeup | Blueprint exists and can frame rigor. PFP16 A++ and Lane 12 negative result have concrete evidence. | Source docs are stale in places. No score claims are allowed for OWV3, H-V3, Lane 19, SA, or HM-S/KL until exact eval artifacts exist. Any Shannon-floor claim must be phrased as hypothesis/engineering direction, not theorem or measured limit. Staged non-git Lightning provenance must be disclosed or supplemented. | Build a claim matrix: claim, formula, evidence path, archive SHA, review status, and limitation. Update audit docs first. Paper may claim PFP16 exact score and Lane 12 exact negative result; all active lanes remain hypotheses. | YELLOW; blocked on audit synchronization before publication. |
| Harness DX | `launch_lane_with_retry.py` has single-flight lock, duplicate live-prefix guard, and process-group cleanup. `reconcile_vast_dispatch_state.py` and adjudicator exist. | Reconcile found tracker drift (`tracker_count=204`, live=4), stale active rows, and live rows missing from active ledger. Active scripts still violate promotion hygiene: Lane 19/HM-S lack deterministic archive + JSON adjudicator, and several scripts call `nvidia-smi` before NVDEC. Regex log parsing remains in generic harvest paths. | Green harness requires either zero state drift or an explicit "live API overrides ledger" rule, plus all active/new remote scripts using deterministic zip, lane-local JSON adjudication, and no regex score parsing for promotion. Add heartbeat classifier and structured failure reasons. | YELLOW as tooling; RED as universal deploy gate until script drift is cleaned. |

## Exact gates by lane

### PFP16 A++

- Keep archive SHA exactly `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Keep archive bytes exactly `686635` unless the result is intentionally superseded.
- Co-locate `archive.zip`, `contest_auth_eval.json`, `auth_eval.log`, `report.txt`, `eval_provenance.json`, and manifest.
- Add missing structured inflation-duration field or clearly reference the log evidence in the final audit bundle.
- Patch paper/audit docs to reflect A++ status and staged-tree provenance limitation.

### OWV3 / Fisher

- Produce CUDA Fisher/Hessian artifact before archive training is treated as scientific.
- Store calibration and holdout split seeds, sample counts, and sensitivity map.
- Pre-register stability threshold and missing-layer behavior.
- Promotion only after exact CUDA/T4 archive eval beats `1.043987524793892`.

### H-V3

- Harvest current run only; no redeploy until script hygiene is fixed.
- Require lane-local `contest_auth_eval.json`, adjudicator output, SHA/bytes, manifest, CUDA/T4 provenance, and `n_samples=600`.
- Fail promotion if zoom scalars were intended but absent, unless the result is explicitly retagged as identity-zoom H-V3.

### Lane 19

- Reconcile KL claim before any promotion statement.
- Fix deterministic zip packaging and JSON adjudication before redeploy.
- Exact eval must beat PFP16 and include component scores; margin success is invalid if it only moves one term while PoseNet collapses.

### SA

- Harvest current plain/no-KL run.
- Gate on exact archive closure: SegMap weights, grayscale video, poses, and segmap inflate config all inside/evidenced.
- Promotion threshold is PFP16, not Lane A or old `1.15`.

### HM-S / KL

- Treat as a high-risk KL forensic run.
- If the result is poor, classify by confounds rather than killing homography globally.
- If the result is good, rebuild under deterministic/adjudicated harness before deploy.

### Lane 12 alpha redesign

- Do not rerun current NeRV alpha as a frontier candidate.
- Start with failure diagnosis around geometry/temporal consistency and PoseNet features.
- Any redesigned alpha must carry a PoseNet guard from the first smoke test, not only at final eval.

### Routing and harness

- Promotion-grade exact eval path: Lightning or equivalent CUDA/T4 runner, not Modal CPU.
- Modal is acceptable for artifact generation, Fisher computation, and ablations only when the resulting archive is later exact-evaluated elsewhere.
- Vast ledgers are advisory until reconciliation drift is resolved; live API plus lane-local artifact evidence controls.

## Shortest wall-clock parallel order

No jobs should be dispatched during this review. Once dispatching is allowed, the shortest safe ordering is:

1. Freeze PFP16 A++ as the deploy baseline immediately.
   - This requires no compute.
   - Synchronize audit/paper docs and attach provenance bundle.

2. Passively harvest active runs in parallel, without restarting:
   - H-V3 (`35907873`)
   - SA (`35906669`)
   - Lane 19 live prefix / current live instance
   - HM-S/KL (`35885106`)
   - Only accept lane-local exact JSON, archive SHA/bytes, and manifest.

3. In parallel with passive harvest, fix harness/script blockers that would invalidate redeploy:
   - Lane 19 deterministic zip + adjudicator + KL-claim reconciliation.
   - HM-S deterministic zip + adjudicator + codec attribution.
   - H-V3/SA/Lane19/HM-S Stage 0 NVDEC-before-GPU-marker cleanup.
   - Generic Lightning fail-fast and JSON harvest.

4. Triage harvested results against PFP16:
   - If exact score >= `1.043987524793892`, do not promote.
   - If exact score < `1.043987524793892`, rerun or rebuild only if script/provenance blockers remain.
   - Kill or retag scientifically confounded lanes rather than turning confounded failures into broad claims.

5. Start OWV3/Fisher only after the routing path is clean enough to preserve its evidence.
   - Run Fisher/build work on Modal or Lightning if useful.
   - Run promotion eval only on Lightning/exact CUDA/T4.

6. Redesign Lane 12 alpha offline while compute lanes run.
   - Diagnosis and objective design can happen without GPU jobs.
   - Do not put it ahead of harvesting active lanes or freezing PFP16.

7. Paper/writeup runs last but can be drafted in parallel after PFP16 freeze.
   - Only promote claims with exact evidence rows.
   - Keep active lanes in the hypothesis/future-work bucket until gates are met.

## Review gates before deploy

Use these gates before any lane is deployed, promoted, or used to supersede PFP16:

1. Evidence gate: exact contest-compliant eval gates above are complete, with no missing archive, no sidecar, no scorer patch, no CPU/MPS/proxy substitution, and JSON adjudication from components.
2. Shannon gate: recompute rate contribution, non-rate contribution, marginal byte slope, and any R(D) claim. Reject arbitrary thresholds without a bit/distortion argument.
3. Dykstra/Boyd gate: verify the candidate lies in the feasible-set intersection and does not rely on additive deltas without a stacked exact eval. For ADMM/coordinator work, require KKT residual logging, restart policy, and discrete projection evidence.
4. Yousfi/Fridrich gate: inspect boundary, texture, scorer-sensitivity, and PoseNet/SegNet failure modes. Any byte win that buys PoseNet/SegNet collapse fails.
5. Contrarian gate: search for no-op encoders, dead flags, default overrides, hidden fallbacks, local sidecars, stale source trees, regex scoring, and mismatched archive/eval SHA.
6. Hotz/Carmack gate: remove unnecessary moving parts from the candidate path and identify the fastest deterministic hardening patch before spending more GPU.
7. Clean-pass counter: three consecutive clean passes are required for internal deploy clearance after any issue is fixed. For public submission/PR language, follow the stricter CLAUDE rule: five consecutive clean passes by the full council, with any issue resetting the counter to zero.
8. Source-sync gate: update `contest_grade_all_lane_results_audit_20260430.md` or a stricter successor before paper/writeup claims. The frontier is the contest-grade ledger, not the presence of files under `experiments/results/`.

## Final pre-deploy blocker list

Hard blockers:

1. Source/audit docs are stale relative to PFP16 A++ evidence.
2. Active Lane 19 promotion claim is scientifically inconsistent until KL usage and margin config are reconciled.
3. HM-S/KL is not deployable without deterministic packaging, JSON adjudication, and PoseNet non-collapse proof.
4. OWV3/Fisher has no Fisher artifact or exact eval.
5. Lane 12 current NeRV alpha implementation is retired by exact CUDA regression and must not be re-promoted without redesign.
6. Generic routing/harness paths are not uniformly promotion-grade.

Soft blockers:

1. PFP16 `contest_auth_eval.json` lacks a structured `inflate_duration_seconds`; log evidence covers the budget but final bundle should make this machine-readable.
2. PFP16 staged Lightning source provenance needs a source manifest or commit-equivalent record.
3. Active dispatch ledgers are stale; live API and artifact evidence must override tracker tables.

Deploy answer as of this review: deploy PFP16 A++ only. Harvest active lanes for possible supersession, but none is green before exact adjudicated evidence exists.
