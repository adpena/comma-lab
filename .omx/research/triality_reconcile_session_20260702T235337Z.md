# #219 AS-BUILT triality reconcile — session 2026-07-02 (DAG ↔ DSL ↔ equations made consistent)

**Directive (operator 2026-07-02):** "update the triality as necessary." Reconcile the campaign
triality with this session's landings so the THREE LEGS AGREE (consistency = campaign-level
non-forgetting; drift between legs is the failure mode). **Pointer 0.19110 UNMOVED** — this is
apparatus / triality maintenance (a MEANS), NOT a score move. CONTAINMENT honored: build + register +
doc + commit ONLY; NO GPU, NO launch, NO trainer/R1 edits.

## What was reconciled (per landing)

| # | landing | DAG leg | DSL leg | equations leg | action taken |
|---|---|---|---|---|---|
| 1 | POWERPLAY campaign-meta / axis-9 | FEED-pp (new) | `tac.witness_dsl.powerplay` (existing) | **`powerplay_variant_ii_cost_isomorphism_v1`** (new) | registered equation + doc §4b |
| 2 | review axis-9 (Correctness Demonstration) | in FEED-pp | `powerplay.CorrectnessDemonstration` (existing) | in equation `domain_of_validity.mechanisms_named` | reflected in doc §4b + DAG |
| 3 | #205 OOM verdict-batch law | FEED-oom (new) | `launch_witness_run.py` gate + `witness_memory_preflight.py` (existing) | **`oom_verdict_batch_spike_peak_rss_v1`** (new) | registered equation |
| 4 | store-nothing pose carrier | FEED-snx (existing) | `PoseGauge.STORE_NOTHING_XI` (existing) | `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1` (built+tested, **NOT in JSONL**) | **DRIFT-CLOSE**: JSONL-registered |
| 5 | R1 gate + #205 GO-gate | FEED-219recon (new) | `--config sealed_205` SEALED, LAUNCH HELD | state node (no new equation) | reflected in DAG + doc §7 |
| 6 | orphan meta-fix #396 + drift-alarm | FEED-219recon | `check_measured_win_findings_are_wired_or_research_only` (WARN-ONLY) + #185 drift-alarm | (apparatus guards) | noted in DAG |
| 7 | compression-as-intelligence grounding | FEED-rdd (new) | `lever_b_generator`/`witness_autoconfig` (task-space design) | **`task_rd_dominates_reconstruction_rd_v1`** (new) | registered equation + doc §4c |

## New / reconciled canonical equations (registered into `.omx/state/canonical_equations_registry.jsonl`)

1. **`powerplay_variant_ii_cost_isomorphism_v1`** — S IS a POWERPLAY (arXiv:1112.5309) Variant-II cost
   `L(s)+α·Σ[t'−r]`: rate=L(s) (description bits), `100·d_seg+√(10·d_pose)`=task deficit, α=λ. Names
   axis-9 = the Correctness Demonstration; Variant-II accept = compose-without-regression; `K(T,q|hist)`
   = the #216 ordering. 1 anchor = the exact identity `powerplay_cost(x).S == compute_contest_score(x)`
   (VERIFIED_VIA_SOURCE_INSPECTION, residual 0.0; illustrative point, NOT a frontier row).
   python_callable = `tac.witness_dsl.powerplay:powerplay_cost`. **NOT a contest lever.**
2. **`oom_verdict_batch_spike_peak_rss_v1`** — the #205 n600 OOM = the advisory-verdict batched-scorer
   transient spike (`0.11 GiB/pair` unchunked → +66 GiB @ n600; `--verdict-batch 32` → +6 GiB floor).
   Score-neutral (verdict outside `value_and_grad`; eval-mode BN batch-independent → d_seg bit-identical).
   1 anchor = MEASURED micro-probe (VERIFIED_VIA_EMPIRICAL_ANCHOR); constants MIRROR + drift-guard-tested
   against `tools/witness_memory_preflight.py`. **Score-neutral launch-safety law, not a lever.**
3. **`task_rd_dominates_reconstruction_rd_v1`** — task-oriented `R_M(D) ≤ R_X(D)` (arXiv:2602.12866 /
   Dobrushin–Witsenhausen); the task-space witness dominating a full-RGB codec is a THEOREM. 1 anchor =
   citation (INFERRED_FROM_DOMAIN_LITERATURE, residual 0.0); bc20/bc36 corroboration honestly caveated
   (reskin, not equal-distortion). **Framing theorem (proves the direction), not a lever.**
4. **`store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`** — DRIFT-CLOSE. Built + tested in FEED-snx
   this session but never persisted to the JSONL (the equations leg was incomplete — module existed,
   registry row did not). Now registered → DSL `PoseGauge.STORE_NOTHING_XI` ↔ DAG FEED-snx ↔ registry
   all agree. (aa_sdf was already registered; store_nothing was the one drifted leg.)

## Files landed

- `src/tac/canonical_equations/powerplay_variant_ii_cost_isomorphism_20260702.py` (new module)
- `src/tac/canonical_equations/oom_verdict_batch_spike_peak_rss_20260702.py` (new module)
- `src/tac/canonical_equations/task_rd_dominates_reconstruction_rd_20260702.py` (new module)
- `src/tac/canonical_equations/__init__.py` (wire the 3 modules into the package API)
- `src/tac/canonical_equations/tests/test_powerplay_variant_ii_cost_isomorphism.py` (+4)
- `src/tac/canonical_equations/tests/test_oom_verdict_batch_spike_peak_rss.py` (+5, incl. mirror drift-guard)
- `src/tac/canonical_equations/tests/test_task_rd_dominates_reconstruction_rd.py` (+4)
- `tools/register_triality_reconcile_session_20260702_equations.py` (registers all 4 into the JSONL; idempotent)
- `.omx/state/canonical_equations_registry.jsonl` (+4 registered events; 212 unique / 261 events)
- `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` (FEED-pp/-oom/-rdd/-219recon)
- `docs/triality_dag_dsl_equations_deepmath.md` (§4b campaign-meta POWERPLAY layer, §4c compression grounding, §7 live state)

## 3-leg consistency verdict

Every session finding is now expressible in ALL THREE legs and they AGREE (see the FEED-219recon table).
The one drift found (store-nothing built+tested but absent from the JSONL) is closed. `query_equations`
returns all 4; ruff F-rules clean; every equation is non-orphan (producers + consumers declared);
13 new focused tests GREEN.

## Honest state

Pointer **0.19110 UNMOVED**. All four equations are MEANS (campaign-structure / apparatus / framing
laws), NOT contest levers — none carries a through-R ΔS. The END remains: R1 gate resolves → operator
GO on `--config sealed_205` → closed-loop-extended n600 run → #202 byte-close → the first
`upstream/evaluate.py` exact n600 row < 0.19110. Sisters: `powerplay_1112.5309_deep_crossref_20260702`,
`compression_as_intelligence_lineage_crossref_20260702`,
`n205_oom_is_verdict_batch_spike_not_accum_loop_chunk_verdict_20260702`, `project_witness_dsl_and_dag_dsl_duality_20260629`.
