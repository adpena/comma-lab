# Codex Findings - OMX Markdown Directive And Orphan Sweep

**UTC:** 2026-05-19T19:09:27Z  
**Actor:** Codex  
**Scope:** all `.omx/**/*.md` files, with high-authority focus on `.omx/research`,
`.omx/state`, latest operator directives, Codex findings, and routing
directives.  
**Score claim:** none.

## Executive Verdict

The `.omx` Markdown control plane is not empty and not cleanly represented by
`canonical_task_status` alone.

The live, actionable work falls into four buckets:

1. Canonical task-status rows that are already mirrored and need normal
   execution, blocking, or dispatch reconciliation.
2. Codex routing directives with `OP-N` headings that were not extractable by
   the canonical task extractor before this pass.
3. Recent operator/PR-body directives that are real, but currently sit as
   partner WIP or Slot-K style integration work rather than canonical Codex
   task rows.
4. Older `.omx` memory snapshots and historical planning files that contain
   many words like "next" and "must" but are not current authority unless a
   current memo or state file reactivates them.

This pass fixed one real extractor gap: `tools/extract_canonical_tasks_from_directive.py`
now recognizes `OP-7 FIRST`, `OP-2`, and sibling `OP-N` routing headings. The
fix is covered by live-directive regression tests.

Follow-on artifact burndown is recorded in
`.omx/research/codex_findings_omx_markdown_orphan_burndown_20260519T194012Z_codex.md`.
Concrete closures landed for SIREN false-authority, PR106 PacketIR runtime
consumption, Catalog #309 horizon-class parsing, `pyppmd` import governance,
master-gradient alternative-reducer wiring, and trainer optimization-helper
audit surfacing.

## Corpus Counts

| Surface | Count |
|---|---:|
| All `.omx/**/*.md` files | 2858 |
| `.omx/**/*.md` modified in last 48h | 523 |
| Directive/finding/session/z7-special Markdown files | 200 |
| Directive/finding/session/z7-special files modified in last 48h | 185 |
| `.omx/research` Markdown files | 2218 |
| Old `.omx/auto_memory_snapshot_20260504T230223Z` Markdown files | 562 |

Generated/public-clone Markdown under `.omx/tmp`, recovered PR snapshots, and
old auto-memory snapshots are high-noise. They were scanned for signal, but
they should not outrank current `AGENTS.md`, `CLAUDE.md`, `.omx/state/*.jsonl`,
and dated `.omx/research` control memos.

## Extractor Bug Fixed

Before this pass, the extractor handled:

- `ITEM N`
- `Wire-in #N`
- `Build #N`
- `CLUSTER X`
- `Sub-cluster X.Y`

It did not handle `OP-N` headings. That made several live 2026-05-18 routing
directives invisible to the canonical extraction flow.

Patched:

- `tools/extract_canonical_tasks_from_directive.py`
- `src/tac/tests/test_extract_canonical_tasks_from_directive.py`

Verified:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_extract_canonical_tasks_from_directive.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check \
  tools/extract_canonical_tasks_from_directive.py \
  src/tac/tests/test_extract_canonical_tasks_from_directive.py
```

Result: `6 passed`; ruff clean.

## False-Authority Bug Fixed

The older `.omx` sweep surfaced one concrete P1 orphan that was still live in
code: `siren_renderer` remained a canonical pre-entropy candidate even though
`.omx/research/super_additive_lane_g_v3_siren_mechanism_20260517.md` records
that the live SIREN path was byte-identical to `lane_g_v3_renderer`. That made
the `lane_g_v3_renderer + siren_renderer` pair a false composition-alpha
authority row.

Patched:

- `tools/pre_entropy_substrate_pivot_prober.py`: removes `siren_renderer` from
  canonical default sweeps until a trained SIREN payload exists.
- `tools/q6_preprobe_pairwise_composition_alpha.py`: removes `siren_renderer`
  from the extended 10-candidate sweep and replaces it with the next ranked
  `distilled_segnet` substrate so the C(10,2) probe geometry stays intact.
- `src/tac/tests/test_pre_entropy_substrate_pivot_prober_phantom_score_fix.py`
  and `src/tac/tests/test_q6_preprobe_extended.py`: add regressions that keep
  the placeholder out of default authority surfaces.

Canonical task mirror:

- `codex_findings_omx_markdown_directive_orphan_sweep_20260519T190927Z_codex::SIREN_FALSE_AUTHORITY_GUARD`

## Concrete Orphan Burndown Recorded

Additional closures from this sweep:

- PR106 PacketIR candidate matrix regenerated with runtime-consumption
  evidence. The matrix now has 15 next exact-eval targets, 13 single-axis rows
  needing paired CPU, 2 runtime-consumed rows needing paired exact eval, and 1
  fail-closed runtime decoder blocker for `format_0x04_rank_elided`.
- Catalog #309 horizon-class gate now accepts Markdown-bold field syntax such
  as `**horizon_class:** plateau_adjacent`.
- Package-code `pyppmd` imports now run through
  `check_no_unwaived_pyppmd_imports(strict=True)` in `preflight_all()`. Legacy
  PPMd replay/HPAC compatibility paths carry explicit `PYPPMD_LGPL_OK:` waivers.
- `tools.probe_alternative_reducers_latent_class_conditioning` is no longer an
  unwired master-gradient surface. The manifest can request diagnostic
  Wyner-Ziv covariance via `tac.master_gradient_consumers.wyner_ziv_side_info_covariance`,
  and the runner exposes master-gradient diagnostic CLI flags.
- The trainer optimization-helper directive is now a reusable AST audit module,
  an operator CLI, and a warn-only preflight hook. Live count: 49 trainers
  scanned, 24 accepted, 25 missing, 0 waived.

## Current Canonical Task Queue

Latest `canonical_task_status` rows show these Codex-owned live rows:

| Status | Task | Sweep verdict |
|---|---|---|
| pending | `...BCEF...::CLUSTER_B` | Real pending: event-driven retroactive-sweep gate. Touches `preflight.py` and `CLAUDE.md`; coordinate around churn. |
| pending | `...BCEF...::CLUSTER_F1` | Real pending: sigma=15 per-substrate sweep design memo. Low collision, no GPU. |
| pending | `...paid_dispatch...::ITEM_1` | Needs reconciliation: CTS says pending, but probe-outcomes has active `lane_17_imp` DEFER blocker until 2026-06-17. Do not dispatch without #313/operator-frontier reconciliation. |
| pending | `...paid_dispatch...::ITEM_2` | Highest unblocked paid candidate by #313 scan, but no obvious operator-authorize recipe for PR106 #05+#06 paired smoke was found in the recipe list. Needs actuator resolution before spend. |
| pending | `...paid_dispatch...::ITEM_3` | Needs reconciliation: CTS says pending, but probe-outcomes has active `lane_mae_v_plus_saug_v2` DEFER blocker until 2026-06-17. Also stated cost can exceed $15 item cap unless smoke-only. |
| pending | `...paid_dispatch...::ITEM_4` | Cheap recovery candidate. No direct #313 blocker found, but exact Catalog #204 A1 passthrough recipe/actuator path is not obvious from `operator_authorize --list`. |
| blocked | `...wire_in...::BUILD_1` | HF Jobs Build #1 blocked on `402 Payment Required` before job id; retry only after credits or provider reroute. |
| blocked | `...wire_in...::BUILD_2` | Z6 4c re-fire blocked on post-fix fresh dispatch verification. |
| blocked | `...wire_in...::BUILD_3` | STC v2 ratify/defer blocked on post-fix successful STC smoke. |
| blocked | `...paid_dispatch...::ITEM_5` | Same Z6 post-fix verification blocker. |
| blocked | `...paid_dispatch...::ITEM_6` | Same STC smoke blocker. |
| blocked | `OP_SYN_1` | Master-gradient six-archive extension blocked on DP1 serializer/projector plus PR106/PR107 projector surfaces. |
| blocked | B1 codebook phase 1 | AV1 premise false; actual path is HEVC/YUV420p. Needs exact B1 lane, CPU-vs-DALI decode identity, rendered-frontier patch-density custody. |
| blocked | deterministic packet runtime authority | Correctly backed off partner-active deterministic compiler surface; procedural candidate authority gate landed separately. |

## Newly Extractable But Not Yet Registered OP Work

After the extractor fix, the following Codex routing directives expose `OP-N`
tasks that are not currently registered as task rows. Do not bulk-register them
blindly: some may have been absorbed under sister names, and some are larger
multi-day design packages. They need reconciliation before becoming the active
queue.

| Directive | Extracted tasks now visible | Notes |
|---|---:|---|
| `codex_routing_directive_cheap_probe_wave_pose_axis_op1_op2_op6_op7_op10_20260518.md` | 5 | OP-7 direct master-gradient pose-byte hoist, OP-2 classification extension, OP-10 autopilot cascade extension, OP-1 Wyner-Ziv pose hoist, OP-6 LFV1. |
| `codex_routing_directive_canonical_phase_1_fisher_precondition_package_20260518.md` | 7 | Canonical Fisher helper package: module, CLI, strict gate, 6-hook wire-in, audit, memory, task-status row. |
| `codex_routing_directive_canonical_riemannian_newton_meta_substrate_package_20260518.md` | 7 | Riemannian-Newton helper package. Depends on Fisher helper. |
| `codex_routing_directive_canonical_tropical_d_seg_solver_package_20260518.md` | 7 | Tropical d_seg helper package. |
| `codex_routing_directive_canonical_n_set_venn_classification_package_20260518.md` | 7 | N-set Venn helper package. |
| `codex_routing_directive_dp1_pr101_path_a_canonical_helper_package_20260518.md` | 8 | DP1 + PR101 Path A package; includes gated paid smoke. |
| `codex_routing_directive_dp1_pr101_op1_op2_zero_cost_probes_20260518.md` | 2 | DP1 OOD-similarity and architecture-compatibility probes. |

Total newly visible unregistered OP rows from these files: 43.

## Recent Operator/PR-Body Directives

The newest PR-body operator directive is live and high-authority but not a
normal score-lowering task. It is also currently partner WIP, so Codex should
avoid trampling in-flight body edits.

Key requirements from `operator_directive_pr_body_stealth_skunkworks_comprehensive_provenance_20260519T184500Z.md`:

- Add full PR provenance list: PR95, PR56, PR97, PR98, PR99, PR100, PR101,
  PR102, PR103, PR105, PR106, PR107, PR108 at minimum.
- Add a used-vs-innovated provenance section.
- Verify both `adpena/comma-lab` and `adpena/tac` are live on `origin/main`
  before PR submission.
- Strip promotion language rather than softening it.
- Re-run T3 council on final body if the previous verdict remains
  `PROCEED_WITH_REVISIONS`.

Follow-on recent memos:

- `pr_body_citations_expansion_audit_20260519T185329Z.md`: adds citation and
  URL guidance. It still needs Slot K integration into the actual PR body.
- `oss_audit_adpena_tac_for_pr_link_20260519T185843Z.md`: verdict
  `PASS_WITH_MINOR_GAPS`; `adpena/tac` is link-safe today, with minor
  README/CI follow-ups not blocking PR body linkage.
- `pr_95_quantizr_emulation_study_20260519T185329Z.md`: supports honest
  provenance language and recommends inline citations for HNeRV, FastViT,
  EfficientNet, segmentation-models-pytorch, and Brotli.

## Z7-Mamba Status

The Z7-Mamba implementation blockers are mostly burned down. The remaining
boundary is exact-eval evidence, not local implementation wiring.

Current evidence:

- `codex_findings_z7_mamba2_adversarial_hardening_20260519_codex.md` says no
  known implementation blocker remains in the reviewed control path.
- `codex_findings_z7_mamba2_chunked_600pair_handoff_ready_20260519T152544Z_codex.md`
  records a full 600-pair recurrent/static same-byte handoff with
  `ready_for_exact_eval_handoff=true` and no result-review blockers.

Required next action:

- Paired `[contest-CUDA]` and `[contest-CPU]` exact eval for recurrent and
  static-control archives under lane-claim custody.
- Keep MPS/local CPU evidence advisory only.
- If `mamba_ssm` is reintroduced, require byte-faithful state/export replay or
  keep it training-only.

## PR95 Local Training Status

The local PR95 path is useful and still advisory.

Current evidence:

- `codex_findings_pr95_local_auth_eval_integrated_bridge_20260519T165732Z_codex.md`
  records an opt-in local training -> codec -> archive -> auth-eval bridge.
- One real MPS training smoke replayed through macOS CPU auth eval with
  absolute score delta `4.277866568713762e-05` between training-side score and
  canonical replay for the same emitted archive.

Required next action:

- Run the next longer PR95 local Stage 1 trend with the integrated bridge.
- Keep results tagged `[macOS-CPU advisory]` or local advisory until paired
  contest-axis replay exists.

## Preflight Performance Status

The full preflight is slow but not silently dying.

Current evidence:

- `codex_findings_preflight_all_timing_profile_20260519T183000Z_codex.md`
  captured `35.562608s` all-scope preflight failure with timing JSON.
- First optimization tranche reduced wall time to `30.201587s`.
- The failure reaches the real pre-existing
  `check_substrate_at_optimal_form_before_paid_dispatch` 17-lane blocker.

Required next action:

- Add per-file result caches for `preflight_dead_resolvers` and
  `check_no_proxy_metric_drives_decision`.
- Avoid whole-check clean caching; invalidation is too coarse for this churny
  repo.

## Prioritized Queue After Sweep

1. Reconcile and register the 43 newly extractable `OP-N` rows, but only after
   checking for absorbed/completed sister work. Start with cheap pose-axis
   OP-7/OP-2/OP-10 and DP1 zero-cost probes, not the large multi-day helper
   packages.
2. Do not dispatch paid Item 1 or Item 3 until their active #313 DEFER blockers
   are reconciled with the operator-frontier override.
3. Resolve Item 2 actuator gap: find or create the PR106 #05+#06 reformulated
   paired-smoke recipe before any Modal spend.
4. Resolve Item 4 actuator gap for Catalog #204 A1 passthrough recovery; this
   is the cheapest clean recovery candidate if the recipe path exists.
5. Finish PR body Slot K integration only after partner WIP settles; apply full
   provenance, `adpena/tac` link, origin/main verification, and promotion
   language removal.
6. Send Z7-Mamba recurrent/static handoff packet to paired contest exact eval
   when dispatch claims and exact-eval custody are clean.
7. Run the next PR95 local MPS Stage 1 trend through the integrated auth bridge
   for training-velocity evidence.
8. Continue preflight performance hardening with per-file caches.
9. Repair B1 authority chain before any contest-video-codebook claim.
10. Revisit OP-SYN-1 only after the three missing projector/helper blockers are
    resolved or explicitly split into smaller tasks.

## Older Sweep Addendum

The read-only older-surface sweep found several non-current but still useful
orphan signals. Only the SIREN false-authority row was patched in this turn;
the rest should become explicit task-status rows only if reactivated.

| Priority | Orphan | Verdict |
|---|---|---|
| P1 | SIREN placeholder in canonical pre-entropy/Q6 candidate maps | Fixed in this turn; default sweeps no longer emit false pair authority. |
| P1 | PR106 PacketIR runtime-consumption blocker | Closed as an evidence queue: runtime-consumption manifests regenerated; 15 next exact-eval targets surfaced; `format_0x04_rank_elided` fails closed on current-runtime decode exception. |
| P1 | L5-v2 TT5L side-info execution bundle | Prepared but not active: 10/10 dry-run cells exist, but non-dry-run remains blocked on identity/source manifest/lane claims. |
| P2 | NSCS03 Phase 2 calibration | Council/operator gated and not mirrored into canonical task status. |
| P2 | PR101/Balle resurrection memo | Contains a stale/nonexistent `--latent-source atw_v2_chroma_residual` style assumption; must grep argparse before use. |
| P3 | G1 v3 per-pair adaptive sigma | Design-only and unregistered; no current execution authority. |

## Commands Run

```bash
find .omx -type f -name '*.md' | wc -l
find .omx -type f -name '*.md' -mtime -2 | wc -l
find .omx -type f -name '*.md' | sed 's#^./##' | awk -F/ '{print $2}' | sort | uniq -c | sort -nr
.venv/bin/python tools/canonical_task_status.py --list-by-owner codex
.venv/bin/python tools/codex_to_claude_inbox.py --compact-json summary
.venv/bin/python tools/extract_canonical_tasks_from_directive.py --json --directive '.omx/research/codex_routing_directive_*.md'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_extract_canonical_tasks_from_directive.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/extract_canonical_tasks_from_directive.py src/tac/tests/test_extract_canonical_tasks_from_directive.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_pre_entropy_substrate_pivot_prober_phantom_score_fix.py src/tac/tests/test_q6_preprobe_extended.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check tools/pre_entropy_substrate_pivot_prober.py tools/q6_preprobe_pairwise_composition_alpha.py src/tac/tests/test_pre_entropy_substrate_pivot_prober_phantom_score_fix.py src/tac/tests/test_q6_preprobe_extended.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_pr106_packetir_candidate_matrix.py src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py::test_pr106_runtime_consumption_fails_closed_for_format04_runtime_gap
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_check_309_horizon_class_declaration.py src/tac/tests/test_check_no_unwaived_pyppmd_imports.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_probe_alternative_reducers.py src/tac/tests/test_trainer_optimization_helper_audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_trainer_optimization_helpers.py
```

## Non-Claims

- This sweep does not promote any score.
- This sweep does not mark old snapshot plans as active by itself.
- This sweep does not dispatch paid jobs.
- This sweep does not mutate partner PR-body WIP.
