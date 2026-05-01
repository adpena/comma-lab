# Adversarial Grand Council Claim-State Review - Worker F - 2026-04-30

Scope: current claim state after PFP16 A++, OWV3 r4/R5 Lightning queue,
component sensitivity, Lane 12 Alpha/NeRV, and J-NWC/J-NWCS hardening.

This is an adversarial review ledger, not a score ledger. No MCP tools were
used. No CUDA eval, remote harvest, or paid compute was launched in this pass.

## Reviewed Control Plane

- `.omx/research/shannon_floor_claim_matrix_20260430_codex.md`
- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430.md`
- `.omx/research/grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430.md`
- `.omx/research/contest_grade_all_lane_results_audit_20260430_codex_progress.md`
- `.omx/research/shannon_floor_execution_readiness_20260430.md`
- `.omx/research/shannon_floor_execution_readiness_20260430_codex_progress.md`
- `.omx/research/shannon_floor_paper_rigor_writeup_blueprint_20260430.md`
- `.omx/research/shannon_floor_swarm_execution_delta_20260430_codex.md`
- `.omx/research/lightning_worker_a_exact_eval_harvest_status_20260430.md`
- `.omx/research/sensitivity_owv3_r5_readiness_20260430_worker.md`
- `.omx/research/component_sensitivity_owv3_nwcs_execution_plan_20260430_codex.md`
- `.omx/research/lane12_alpha_geo_dispatch_readiness_20260430_codex.md`
- `.omx/research/j_nwc_j_nwcs_manifest_fake_sensitivity_hardening_20260430_codex.md`
- `.omx/research/nwcs_build_only_smoke_readiness_20260430_worker.md`
- Lightning state records under `.omx/state/*owv3_r5*` and
  `.omx/state/*pfp16_paired*`.

## Current Claim Verdict

1. PFP16 remains the only current rankable A++ frontier.
   - T4 exact eval: recomputed `1.043987524793892`, bytes `686635`, archive
     SHA `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
     PoseNet `0.00346442`, SegNet `0.00400656`, `n_samples=600`,
     `gpu_t4_match=true`.
   - This is a deploy baseline, not Shannon-floor attainment.

2. OWV3 r4 is exact CUDA/T4 diagnostic-negative despite lower total score.
   - Exact r4: recomputed `1.0378905176070103`, bytes `686557`, archive SHA
     `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.
   - Adjudication rejected it because SegNet relative ratio `1.003654`
     exceeded the predeclared `1.002` cap.
   - It must not replace PFP16, rank as the frontier, or be used to relax the
     gate after the fact.

3. OWV3 R5 is queue-only, not evidence.
   - R5 rank-1 archive exists locally at `686468` bytes, SHA
     `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518`.
   - Local Lightning state shows submitted/running queue records for R5 and
     paired PFP16 calibration, but no local mirrored R5 `contest_auth_eval.json`
     or adjudication files exist.
   - R5 promotion requires paired same-run PFP16 calibration plus exact CUDA/T4
     eval and component-gated adjudication.

4. Component sensitivity is still diagnostic infrastructure.
   - The current producer is Fisher-proxy only, even on CUDA, and explicitly
     cannot emit a promotable `component_sensitivity_v1`.
   - No local `component_sensitivity_v1` exists. Do not use the 30-pair Fisher
     map as paper-grade sensitivity evidence.

5. Lane 12 Alpha/NeRV is scoped negative evidence only.
   - `jsonfix40` exact CUDA score `26.03719330455429` retires that measured
     implementation/config, not NeRV, INR, Alpha, or mask compression as a
     family.
   - Further dispatch remains blocked by decoded-baseline target custody,
     geometry diagnostics, pose-regeneration provenance, and L2 clearance.

6. J-NWC/J-NWCS is engineering-green in several guardrails but score-empty.
   - Corpus replay, fake/debug sensitivity stops, `NWCS1` loader path, exact
     adjudication custody fields, and negative/non-finite sensitivity rejection
     are useful hardening.
   - The only local smoke is CPU build-only with `score_claim=false`.
   - No J-NWC/J-NWCS ranking, promotion, or retirement claim is supported.

## Findings

1. Fixed in this pass: claim matrix C-021 was stale. It still described the
   original OWV3 byte-feasible candidate as pending exact eval, while C-039
   recorded the later exact r4 SegNet-gate failure. C-021 now points readers to
   C-039 for current exact status.

2. Fixed in this pass: the claim matrix lacked an explicit R5 queue-only row.
   Added C-040 so submitted/running Lightning jobs cannot be confused with
   harvested score evidence.

3. Remaining paper/writeup risk: public paper/report files are still listed as
   requiring regeneration or quarantine in the claim matrix. Do not ship them
   until old Lane G v3, Modal/local, CPU/MPS, and broad kill wording is removed.

4. Remaining reproducibility risk: PFP16 score custody is strong, but the
   final deploy/paper packet still needs the public docs regenerated from the
   claim matrix and should keep the missing remote staged-tree manifest caveat
   visible.

5. Remaining arbitrariness risk: OWV3 R5 candidate ranking uses a sensible
   conservative-neighbor heuristic after r4's SegNet miss, but it is still a
   heuristic until paired exact eval shows the component gate actually holds.

## Next-Turn Roadmap

1. Harvest or refresh the two Lightning jobs without using logs as score
   evidence: paired PFP16 calibration first, then OWV3 R5. Accept only local
   mirrored `contest_auth_eval.json`, archive SHA/bytes, runner preflight,
   supply-chain scan, and adjudication artifacts.

2. If R5 lands, adjudicate against paired PFP16 with `--max-segnet-relative
   1.002`, `--max-posenet-relative 1.002`, `--required-device cuda`, and
   `--required-samples 600`. If either component gate fails, record scoped
   A-negative diagnostic evidence and do not rank it.

3. Build the official component-response producer before any sensitivity paper
   claim: finite-difference component response, symmetric/directional curves,
   all-pair or reviewed sample plan, calibration/holdout stability, then
   `component_sensitivity_v1` assembly.

4. Regenerate public writeup docs from the claim matrix only after R5 harvest
   is resolved. Keep PFP16 as the headline until a component-gated exact
   archive beats it.

5. Keep Lane 12 on build/diagnostic gates until decoded-baseline custody,
   Alpha-Geo diagnostics, regenerated pose provenance, and L2 clearance are all
   present.

6. Keep J-NWC/J-NWCS exact eval blocked until real CUDA scorer-derived
   sensitivity artifacts exist for anchor and corpus and the scripts preserve
   adjudicated JSON plus custody hashes.

## 2026-04-30T22:57Z Recursive Follow-Up

Latest Codex progress ledgers were rechecked after the original Worker F pass:
the 22:48Z/22:53Z/22:55Z deltas classify the first Lightning PFP16/R5 attempts
as harness failures only, then record clean isolated reruns as `Running`.

Local read-only checks agree with that claim state:

- `experiments/results/lightning_batch/` contains local exact/adjudication JSON
  only for OWV3 r4, not for PFP16 paired calibration or OWV3 R5.
- `.omx/state/lightning_batch_jobs.json` records original
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r2` as `Failed` and
  original `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4` as `Stopped`.
  These are shared-venv/tqdm harness-contamination history, not lane evidence.
- The clean isolated jobs
  `pfp16_paired_calibration_20260430_codex_lightning_t4_r3_isolated_uv` and
  `owv3_r5_rank1_exact_cuda_20260430_codex_lightning_t4_r2_isolated_uv` are
  last locally recorded as `Running` at 22:57Z. Running jobs are not evidence.

Claim hygiene fix in this follow-up:

- Tightened C-040 in the claim matrix so it names the isolated reruns, labels
  abandoned first attempts as `failed_harness_history`, and preserves
  `no_score_claim` / `no_rank_frontier` until terminal harvest, archive
  identity validation, CUDA/T4 eval, adjudication artifacts, paired PFP16
  readjudication, and component-gate pass.

Residual risks after this pass:

- Public paper/report files remain outside this worker's write scope and still
  need regeneration/quarantine before any judge-facing packet.
- Lightning job state is time-sensitive. This pass did not refresh remote SDK
  status or harvest artifacts because doing so would mutate state outside the
  allowed write scope. Treat local `Running` status as the latest reviewed
  ledger/state, not as live proof that the jobs are still running.
- If the isolated jobs later finish, the next reviewer must classify terminal
  artifacts before any score claim: harness/custody failure, non-promotable
  component-gate failure, or promotable exact evidence after paired review.

## 2026-04-30T23:10Z Codex Integration Follow-Up

The running-state warning above is superseded by harvested exact CUDA/T4
forensic artifacts:

- PFP16 paired calibration `r3_isolated_uv`: score
  `1.037045485927815`, non-promotable because the predeclared SegNet component
  gate fired.
- OWV3 R5 rank-1 `r2_isolated_uv`: score `1.0373951773937642`,
  non-promotable because the same SegNet component gate fired.
- Paired result: R5 is `+0.00034969146594909795` worse than PFP16 while saving
  `167` archive bytes.

Claim hygiene verdict: C-040 remains non-ranking and non-frontier. This is a
scoped negative for the current R5 configuration, not a family-level KILL and
not a license to relax component gates.

## 2026-04-30T23:28Z Public-Doc Hygiene Check After R5 Harvest

Read-only verification scope: C-040, latest progress addenda, and public
paper/report/runbook surfaces. No MCP tools, no remote refresh, and no edits
outside this Worker F ledger.

Control-plane verdict:

- `shannon_floor_claim_matrix_20260430_codex.md` C-040 is correct. It labels
  the paired PFP16/R5 branch as `A-negative` scoped forensic evidence with
  `exact_cuda_t4_forensic`, `no_promotion`, and `no_rank_frontier`.
- C-040 records the clean isolated PFP16 paired calibration score
  `1.037045485927815` and clean isolated OWV3 R5 score
  `1.0373951773937642`; both fired the predeclared SegNet component gate, and
  R5 is `+0.00034969146594909795` worse than paired PFP16 despite `167` fewer
  bytes.
- `grand_council_paradigm_shift_to_shannon_floor_20260430_codex_progress.md`
  and `shannon_floor_execution_readiness_20260430_codex_progress.md` are
  correct at their latest 23:10Z sections: the harvest is exact CUDA/T4
  forensic evidence, not a new promotion packet and not a frontier result.
- `shannon_floor_swarm_execution_delta_20260430_codex.md` is also correct at
  its 23:10Z harvest section: valid exact CUDA/T4 forensic evidence with local
  JSON custody, not frontier evidence because the component gate fired and R5
  lost to paired PFP16.
- Earlier queued/running progress entries are append-only history and are
  superseded by the later harvest sections. They should not be quoted without
  the 23:10Z supersession.

Stale public paths and required next edits:

| Path | Stale state | Required next edit |
| --- | --- | --- |
| `docs/runbooks/owv3_r5_exact_eval_queue.md` | Still presents R5 as queue guidance: "Evaluate only after paired PFP16 calibration", "Queue only the rank-1 R5 archive", and "Rank 1 is selected for first exact eval." | Convert to a historical R5 forensic-result runbook. Add the paired PFP16/R5 exact CUDA/T4 scores, component-gate failure, R5-worse-than-paired-PFP16 delta, `no_promotion`, and `no_rank_frontier`; point next work to SegNet-conservative R6 or official finite-difference sensitivity. |
| `docs/owv3_fisher_runbook.md` | Still r2-oriented and says no exact score exists for the r2 OWV3 candidate; it omits r4 exact diagnostic failure and R5 exact forensic failure. | Add a current-status section covering r4 and R5: both exact CUDA/T4, both non-promotable due predeclared component gates, PFP16 A++ remains the only frontier. Keep r2 as historical byte/build context only. |
| `docs/paper/ara/evidence/results_index.json` | Public evidence index lists Modal/local diagnostic rows as plain result records with `[Modal-T4-CUDA]` score lanes and no `promotion_eligible`, `can_rank`, evidence grade, or invalid/diagnostic status. This can be misread as rankable evidence. | Regenerate from the claim matrix. Add explicit `evidence_grade`, `promotion_eligible`, `can_rank`, `component_gate_status`, and `custody_status`; only PFP16 A++ may be rankable. Move Modal/local/invalid rows to diagnostic history or quarantine. |
| `docs/paper/ara/evidence/era2/modal_repro/9b20bdfca246.json` | Placeholder says "pending" with a headline "Modal T4 reproduction of Lane G v3 within 0.01 noise floor" and expected `~1.04`, without current PFP16/R5 gate context. | Replace placeholder with harvested structured evidence or remove from public evidence. Label it historical predecessor corroboration only, not contest-grade frontier evidence. |
| `docs/paper/ara/logic/claims.md` C8 / `docs/paper/ara/logic/experiments.md` E8 / `docs/paper/ara/src/index.md` Modal repro command | The Lane G v3 Modal reproduction is public-safe but still too easy to read as a current rankable result because it says Modal T4 was within noise and has a direct repro command. | Add "historical predecessor / non-frontier / not A++" language and require `contest_auth_eval.py --device cuda` plus claim-matrix evidence grade before any ranking. |
| `reports/graphs/site/*` and generated graph/history JSON | Generated public graph packets still contain old "promoted floor/current frontier" CPU/local-era language. | Regenerate or quarantine before any public/judge-facing packet. Treat historical graph snapshots as dated history, not current frontier state. |

Checked but not stale for this specific R5 harvest:

- `reports/latest.md` and `reports/writeup_working.md` continue to headline
  PFP16 A++ as the current contest-CUDA authority and do not claim R5 as
  queued, running, promoted, or frontier.
- `docs/paper/ara/PAPER.md`, `docs/paper/ara/logic/problem.md`, and
  `docs/paper/ara/logic/related_work.md` headline PFP16 A++ only and do not
  promote R5.
