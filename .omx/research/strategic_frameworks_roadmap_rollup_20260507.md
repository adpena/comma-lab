# Strategic frameworks roadmap rollup — 2026-05-07

**Frontier**: leaderboard 0.19, theoretical floor 0.155 and beyond
**Anchor evidence**: PR103-on-PR106 @ 0.20898 [contest-CUDA] (the ONLY current contest-CUDA result with all 3 axes populated)
**Scope**: comprehensive next-steps + outstanding-work + gap rollup for the cathedral's two strategic optimization frameworks

This memo answers the operator's question: *what is the roadmap and next steps and outstanding tasks and remaining gaps and work todos and everything for both strategic optimization frameworks*.

The cathedral has TWO strategic optimization frameworks running in parallel:

1. **Bilevel Optimization** (`tools/run_bilevel_optimization.py`) — outer δεζ joint training, middle meta-Lagrangian search, inner Joint-ADMM coordinator. 7-phase trajectory toward 0.155 floor.
2. **Meta-Lagrangian / Pareto** (`tools/meta_lagrangian_search_cli.py` + adapter + 3-axis Pareto) — atom catalog + selection-by-EIG + Pareto frontier filtering on the contest-objective surface.

These frameworks **compose** at the middle layer: bilevel's middle stage IS meta-Lagrangian search; Pareto filtering is meta-Lagrangian's pruning rule.

---

## Framework 1: Bilevel Optimization (the cathedral)

### Status: phases 1-3 wired, 4-7 are scaffolding

| Phase | Predicted score [predicted-band] | Status | Code path |
|---|---|---|---|
| 1: PR100+Op1+Op2+Op2.5 stack | 0.190 | **READY-TO-DISPATCH** (playbook landed `162e1a2d`) | `scripts/deferred_dispatch_playbook_phase1_pr101_canonical_winner_20260507.sh` |
| 2: + RAFT poses | 0.183 | **CodecOp landed** (`630fb28a`) | `src/tac/codec_pipeline_raft_pose.py` |
| 3: + δεζ training callback | 0.165 | **Driver landed** (`630fb28a`) | `tools/run_deltaepszeta_training.py` |
| 4: + KL-pose / Hilbert-manifold | 0.155 ⟵ **theoretical floor** | **CodecOp landed**, sweep adapter just landed | `src/tac/codec_pipeline_kl_pose.py`, `tools/codec_op_param_sweep_manifest.py` |
| 5: + tree-Wasserstein mask coding | 0.145 | research-only (Kalle p-adic addendum) | _scaffold pending_ |
| 6: + Ballé hyperprior / multi-level ultrametric | 0.135 | research-only | _scaffold pending_ |
| 7: + SJ-KL Wave-Ω basis | 0.125 | research-only (Wave-Ω-V3 launch-ready) | `experiments/build_sjkl_residual.py` (Track B landed) |

### Outstanding tasks for Framework 1

#### Phase 1 (READY): 1 blocker — operator GPU authorization
- ✅ Playbook script landed
- ✅ Pre-flight + lane claim + 4 provider invocations wired
- ⏸️ **BLOCKER**: operator must authorize ~$0.30-0.50 dispatch cost on Lightning 4090 / T4
- Deliverable when fired: PR101 canonical-winner contest-CUDA reproduction at ~0.193

#### Phase 2 (CodecOp ready): 2 gaps
- ✅ `Op_RAFTPoseStream` landed (398 LOC + 16 tests)
- ⏸️ **GAP 2A**: `Op_RAFTPoseStream` is not yet integrated with the **Joint-ADMM proximal codec** Protocol. The cathedral's Joint-ADMM coordinator consumes `StreamProximalCodec`, not `CodecOp`. Adapter wrap pending. Code: `src/tac/joint_admm_proximal_pose_delta.py` (sister file with the legacy contract).
- ⏸️ **GAP 2B**: archive-substitution surgery to actually ship a Phase-2 archive. Phase 2 means "PR100 + Op1+Op2+Op2.5 + RAFT poses substituted in" — substituting requires reading the existing pose blob's archive offset, replacing it, repacking. Not yet built.

#### Phase 3 (driver landed): 3 gaps
- ✅ `tools/run_deltaepszeta_training.py` landed (354 LOC + 12 tests, λ stable, rate budget enforced)
- ⏸️ **GAP 3A**: training driver runs CPU-only synthetic at the moment; real GPU dispatch to a substrate dataset is the next step. Estimated cost ~$5-10 for a meaningful training run on Lightning 4090.
- ⏸️ **GAP 3B**: the δεζ callback (`tac.codec_pipeline_deltaepszeta_callback`) currently uses **L²-distance** for mask distortion. Per Kalle p-adic addendum, the actionable subitem is swapping L² → tree-Wasserstein on the 5-class hierarchy (drivable / lane / object / background / unknown). 30-LOC change. Pending.
- ⏸️ **GAP 3C**: integration with bilevel atom ledger — the training driver needs to write atoms back to `experiments/results/bilevel_atom_ledger.jsonl` so subsequent meta-Lagrangian searches can reseed from empirical training residuals.

#### Phase 4 (KL-pose just landed): 1 blocker, 1 gap
- ✅ `Op_KLPoseStream` landed (this session) with empirical 2.80× compression on smooth driving trajectories
- ✅ `tools/codec_op_param_sweep_manifest.py` landed (this turn) — emits dispatch-shaped JSON manifest with predicted bands
- ⏸️ **BLOCKER**: archive-substitution surgery (see GAP 2B; same blocker for any non-baseline CodecOp shipping)
- ⏸️ **GAP 4A**: hyperbolic-foveation + lapose-foveation surface (10 files, 34 tests passing) is unwired into the bilevel trajectory. These are foveation paradigms that compose with KL-pose. Decision needed: are they a Phase 4.5 substep or a separate research lane?

#### Phases 5-7 (research-only): scaffolding gaps
- ⏸️ **GAP 5**: tree-Wasserstein mask payload allocation — operator's STUDY-priority addendum to the Kalle memo. 30-LOC change to `tac.codec_pipeline_deltaepszeta_callback`. Predicted impact unknown without empirical dispatch.
- ⏸️ **GAP 6**: Ballé hyperprior multi-level extension — Lane 20 Ballé hyperprior is at L3 (1.016 contest-CUDA from β Fisher) but the multi-level ultrametric generalization is research-only.
- ⏸️ **GAP 7**: Wave-Ω SJ-KL basis is launch-ready (Lane Ω-W-V3 has 5 commits + dispatch wrapper) but the Q-FAITHFUL substrate retrain that's the prerequisite is unauthorized GPU work.

### Bilevel framework cross-cutting blockers

1. **Operator GPU billing** — every "ship Phase N to contest-CUDA" step is gated on this. Last known state: Lightning 4090 + T4 available; Vast.ai credit exhausted.
2. **Archive-substitution surgery** — substrate-specific. PR101's archive layout is one specific case; PR100/PR106 have different ones. This is the main "actuator gap" between cathedral CodecOps and contest-eligible archives. ~4-6 hours of CPU work if done by topic (zip member offsets + manifest update); could also be done substrate-by-substrate as needed.
3. **`StreamProximalCodec` ↔ `CodecOp` adapter** — Joint-ADMM coordinator can't consume the new CodecOp ecosystem. ~150 LOC adapter. Blocks Joint-ADMM-driven dispatches.

---

## Framework 2: Meta-Lagrangian / Pareto

### Status: search engine landed, evidence corpus is sparse

| Component | Status | Code path |
|---|---|---|
| Atom emission protocol | ✅ landed (typed atoms with rate, score, evidence_grade, archive custody, blockers) | `tac.contest_rate_distortion_system` |
| Search CLI | ✅ landed (auto-sweep over bit widths) | `tools/meta_lagrangian_search_cli.py` |
| 3-axis Pareto frontier | ✅ landed THIS turn | `tools/contest_score_pareto_3axis.py` |
| Atom ledger adapter | ✅ landed (priority #5) | `tools/meta_lagrangian_atom_ledger_adapter.py` |
| Auto-promotion | ✅ landed (priority #4) | `tools/auto_promote_contest_cuda.py` |
| Predispatch sanity gate | ✅ landed (5-gate ladder) | `tools/predispatch_sanity_gate.py` (Track 1A) |
| **Empirical evidence corpus** | ⚠️ **1 candidate** (PR103-on-PR106) | `experiments/results/pr103_repack_pr106_standalone_20260507/` |

### Outstanding tasks for Framework 2

#### Evidence corpus (the dominant gap)
- ⚠️ **GAP M-1**: only **1 candidate** has full (d_seg, d_pose, B) contest-CUDA evidence in the cathedral schema. Older lanes used `auth_eval_renderer_fp4.json` schema variants the 3-axis loader doesn't recognize.
- ⏸️ **TODO M-1.1**: extend the evidence loader (`_extract_d_seg`, `_extract_d_pose`, `_extract_archive_bytes` in `tools/contest_score_pareto_3axis.py`) to recognize 3-5 legacy schema variants. ~50 LOC + tests.
- ⏸️ **TODO M-1.2**: write `tools/migrate_legacy_evidence.py` that scans older `auth_eval_renderer_fp4.json` files and rewrites them into `pre_submission_compliance.contest_final.json` shape. ~100 LOC + tests.
- ⏸️ **TODO M-1.3**: future contest-CUDA dispatches MUST write `pre_submission_compliance.contest_final.json` natively. Already enforced by `tools/auto_promote_contest_cuda.py`; ensure all new dispatch playbooks call it.

#### Search engine extensions
- ⏸️ **TODO M-2.1**: extend `tools/meta_lagrangian_search_cli.py` to consume the new `tools/codec_op_param_sweep_manifest.py` output. Currently the search CLI is hardcoded to the apogee_intN bit-width sweep; the new manifest format is more general.
- ⏸️ **TODO M-2.2**: add EIG (expected information gain) ranking. The cathedral docstring claims EIG-based atom selection but the current implementation ranks by predicted score only. Adding EIG requires a calibration model — could plug Bayesian-GP STUDY work in if/when its conditions flip to WIRE.
- ⏸️ **TODO M-2.3**: add tree-Wasserstein-cost-aware Pareto pruning per Kalle p-adic addendum. Closed-form on the 5-class hierarchy; ~80 LOC.

#### Pareto frontier extensions
- ✅ 3-axis (d_seg, d_pose, B) frontier landed
- ⏸️ **TODO M-3.1**: 4-axis frontier — add **wall-clock cost** as a 4th axis. The contest only counts the 3 axes for SCORE, but operator's GPU budget is implicit; a Pareto over (d_seg, d_pose, B, $) prunes dispatches that don't improve any contest-axis enough to justify their cost.
- ⏸️ **TODO M-3.2**: importance-flip-aware ranking — at d_pose < 2.5e-4 (PR106 regime), pose-marginal dominates seg-marginal by 2.71×. The 3-axis tool currently scores all candidates equally; an enhancement could weight ranking by the marginal-flip multiplier so pose-improving candidates rank higher in the pose-dominated regime.

### Meta-Lagrangian framework cross-cutting blockers

1. **Empirical sparsity** — search/Pareto are only as good as the candidate corpus. Without contest-CUDA dispatches, atoms remain `[predicted-band only]`. This is the same operator-GPU-authorization blocker as Framework 1's Phase 1.
2. **Calibration anchor count** — the predictor (`src/tac/predictor/score_band.py`) uses a 3-anchor empirical fit. Bayesian-GP paper synthesis flagged "anchor count ≥8 with structured residuals" as the flip condition for that direction. Currently <8.
3. **Adapter → Op chain** — the just-landed `tools/codec_op_param_sweep_manifest.py` emits manifests, but doesn't yet feed back into the meta-Lagrangian search CLI (TODO M-2.1).

---

## Framework intersections

The two frameworks meet at:

1. **Bilevel's middle stage IS meta-Lagrangian search**. `tools/run_bilevel_optimization.py` calls `meta_lagrangian_search_cli.py` between phases. Improvements to either propagate.
2. **Pareto pruning gates bilevel phase advancement**. Phase N → N+1 only happens if a candidate beats the prior phase's best on the 3-axis frontier. The 3-axis tool just landed → bilevel's gate is now machine-verified, not eyeballed.
3. **Atoms are common currency**. Both frameworks emit/consume `MetaLagrangianAtom` records (typed dataclass in `tools/meta_lagrangian_atom_ledger_adapter.py`). The ledger at `experiments/results/bilevel_atom_ledger.jsonl` is the shared blackboard.
4. **CodecOps are common building blocks**. Bilevel phases 2-7 each correspond to wrapping a new paradigm as a CodecOp; the meta-Lagrangian search consumes them via the new sweep manifest.

---

## Cross-cutting work items (independent of either framework)

### Engineering hygiene
- ⏸️ **TODO X-1**: foveation surface (10 files, 34 tests) is **unwired** into either framework. Decision needed whether to wrap as a Phase 4.5 CodecOp or treat as an alternate paradigm lane.
- ⏸️ **TODO X-2**: legacy evidence schema migration (TODO M-1.1 + M-1.2). High-value because it grows the empirical corpus without new dispatches.
- ⏸️ **TODO X-3**: `parallel_dispatch_top_k.py` currently has a strict gate that rejects predicted-band candidates. The operator-override pattern works but is undocumented. Add an `--accept-predicted-band` flag with a mandatory `--operator-override-reason` field for audit trail. ~40 LOC.

### Research workstreams (STUDY-tagged, no immediate WIRE)
- Hilbert manifolds direction (memo: `feedback_hilbert_manifolds_research_direction_20260507`):
  - HM #1 (KL-pose) — ✅ landed this session
  - HM #2 Fisher-Rao mask distortion — pending (overlaps with Framework 1 Phase 5 GAP 5)
  - HM #3 Wasserstein-2 scorer surrogate — pending
  - HM #4 RKHS renderer regularizer — pending
- Bayesian Decision Theory paper — ✅ STUDY verdict (no WIRE action)
- Kalle's ninth proof of folding — ✅ WIRE §4.1 (KL-pose k-sweep, blocked on archive surgery + GPU authorization)
- p-adic / ultrametric / tree-Wasserstein addendum — STUDY-priority subitem queued
- Alternative paradigms queue (`feedback_alternative_paradigms_research_queue_20260507`):
  - RAFT poses — ✅ wrapped
  - KL poses — ✅ wrapped
  - MAE — pending (Lane MAE-V deferred)
  - SIREN — pending (no implementation yet)
  - CLADE / SPADE — pending (low-priority research-only)
  - LA-pose — pending
  - MNeRV — pending
  - Wavelets — partial (`wavelet_mask_codec.py` exists; not in cathedral)
  - Telescopic foveation — landed but unwired (X-1)

### Subagent inventory (status as of this turn)
- ✅ Bug-hunter v3 (cathedral seams) — 6 commits landed
- ✅ RAFT-poses + δεζ scaffolding — 28 tests landed
- ✅ Bayesian GP paper analysis — STUDY memo landed
- ✅ Kalle's ninth proof of folding — memo + WIRE verdict landed
- ☠️ Telescopic foveation `a124fb91` (prior session, abandoned 3 days ago, work survives in repo)
- ⏳ HuggingFace dataset upload `task #368` — in_progress, status unknown

---

## Forward roadmap (priority-ordered, this session's recommendation)

### Now (CPU-only, no GPU dependency)
1. **TODO M-1.1 + M-1.2**: extend evidence-loader schema variants + migrate legacy evidence. Grows the 3-axis Pareto corpus from 1 candidate to potentially 20-50, unlocking real Pareto analysis on existing work without new dispatches. **2-3 hours**.
2. **TODO X-3**: `--accept-predicted-band` operator-override flag for `parallel_dispatch_top_k.py`. Unblocks dispatch of CodecOp sweep manifests when GPU returns. **1 hour**.
3. **GAP 2A** (`StreamProximalCodec` ↔ `CodecOp` adapter): unlocks Joint-ADMM-driven dispatches. **2-3 hours**.

### When operator authorizes GPU
1. **Phase 1 dispatch** (PR101 canonical-winner replay) via the landed playbook. ~$0.30-0.50 cost. Validates the cathedral end-to-end on a known anchor.
2. **KL-pose k-sweep** (Kalle WIRE §4.1) — predicted -0.001 to -0.0017 score per candidate. Requires Phase 1 to land first to validate the substitution path.
3. **Phase 3 GPU training** for δεζ driver — validate the rate-budgeted joint-training approach on real substrate. ~$5-10.

### Research workstreams (continue independently of GPU)
1. Tree-Wasserstein mask payload allocation (operator's p-adic actionable) — 30-LOC change.
2. HM #2 Fisher-Rao mask distortion in δεζ callback — overlaps with prior; pick one.
3. Wrap MAE / SIREN as CodecOps (Phase 5/6 scaffolding).

---

## Recommended ordering for next 4 hours of CPU work

```
1. (1.0h) M-1.1: extend 3-axis evidence loader schema variants
2. (1.5h) M-1.2: migrate legacy evidence to canonical schema
3. (0.5h) Run 3-axis Pareto on grown corpus, report findings
4. (1.0h) X-3: operator-override flag for parallel_dispatch_top_k.py
```

This sequence grows the empirical corpus + unblocks dispatch — the
two highest-leverage independent CPU workstreams. Total: ~4 hours,
deliverables include a richer empirical Pareto frontier and a
dispatch-ready manifest pipeline.

---

## Cross-references

- Frontier: `project_leaderboard_0_19_theoretical_floor_0_155_20260507`
- 7-phase trajectory: `.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md`
- This session's pose-axis forensic: `.omx/research/pr106_pose_axis_forensic_memo_20260507_claude.md`
- Hilbert manifolds: `feedback_hilbert_manifolds_research_direction_20260507`
- Kalle's p-adic addendum: `feedback_kalle_ninth_proof_of_folding_synthesis_20260507`
- Bayesian-GP STUDY verdict: `feedback_bayesian_gp_paper_synthesis_STUDY_verdict_20260507`
- Alternative paradigms queue: `feedback_alternative_paradigms_research_queue_20260507`
- Cathedral atom ledger: `experiments/results/bilevel_atom_ledger.jsonl`
- Lane registry: `.omx/state/lane_registry.json`
