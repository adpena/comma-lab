# Codex reinforcement packet — war-room dispatch (2026-06-09)

For Codex (gpt-5.5) agents returning to the repo. START HERE so four agents work disjoint lanes under
ONE proof law instead of colliding in a foggy repo. Read this + the two contracts, then take ONE lane.

## The two contracts every agent obeys (read first)
1. `.omx/research/pact_evidence_constitution_20260609.md` — the proof lattice: `authority_tier`
   (where) × `metric_family` (what) × eligibility (what it may change). Score roadmap moves ONLY on a
   contest-axis `exact_evaluate` row; promotion ONLY on paired CPU+CUDA on the same archive_sha256.
2. `.omx/research/dual_optimization_principle_intrinsic_and_contextual_20260609.md` — every element
   optimized intrinsically AND contextually (non-separable objective → coordinate-ascent w/ interaction
   terms). Plus CLAUDE.md + AGENTS.md non-negotiables (commit via `tools/subagent_commit_serializer.py`
   with post-edit `--expected-content-sha256`; review-gate `.py` via `tools/review_tracker.py`; never
   MPS as authority; never `/tmp` in evidence; NO FAKE implementations).

## State of truth (the only score axes)
- Frontier (pointer-only, `.omx/state/canonical_frontier_pointer.json`): contest_cpu **0.19199**
  (`fp11_source_brotli_recode`, sha b7106c9bdbb8) · contest_cuda 0.20533.
- The living vehicle index: run `.venv/bin/python tools/render_pact_compiler_dashboard.py` (generated,
  current). V3 = the compiler/judge; every lane must emit a typed `CandidateActionEvaluation`
  (`tac.optimization.harvest_evidence`) judged by exact ΔS.
- The contest objective is the ONLY law: `S = 100·d_seg + sqrt(10·d_pose) + 25·bytes/37,545,489` via
  `upstream/evaluate.py` on exact archive bytes. SegNet = frame1 argmax; PoseNet = both-frame YUV6.

## Live jobs (DO NOT touch their files / GPU slot)
- **HEAVY (Claude, the one Metal slot):** clean PR95-faithful baseline
  `b1_clean_pr95_baseline_20260609T171700Z` (ep1000 B2 exact eval running). Do NOT launch a competing
  MLX training; one heavy slot only.
- Claude owns: `tools/ingest_exact_eval_to_candidate.py`, `tools/run_hi_nerv_authority_trace.py`,
  `tools/run_hi_nerv_recon_fit_capacity.py`, `src/tac/optimization/composition_carrier_registry.py`,
  the clean-PR95 ep1000 ingest + route. **Avoid these.**

## The four disjoint lanes (take ONE; each ends with a CandidateActionEvaluation or an exact blocker)
**Codex A — V5 PR110++ selector/menu (frontier-direct, highest near-term ROI).**
- Reproduce the K=16 per-pair selector over a frozen HNeRV substrate → Huffman/range-coded selector
  stream → archive → B2 exact eval → `CandidateActionEvaluation` (vehicle=`atom` or a new pr110pp tag).
- Exploit the evaluator asymmetry (frame0 affects PoseNet, invisible to SegNet). Extend the mode
  catalog only after the K=16 reproduction emits an exact row. comp-Muon-*inspired* selector×menu later.
- Files: `src/tac/substrates/boost_nerv_pr110_residual/` + a new `tools/pr110pp_*`. Avoid HiNeRV trainer.

**Codex B — V4 PACT-NeRV-VQ first exact-eval path.**
- The maturity audit (`pact_nerv_vq_maturity_audit_for_codebook_investment_20260609.md`) found a real
  VQ codebook (K=512, score-aware) but NO contest eval + same ~4.5 dB failure. Get it to ONE exact B2
  row: export an archive → inflate → `evaluate.py` → `CandidateActionEvaluation`. Reuse its VQ/PVQ
  codec primitives. Files: `src/tac/substrates/pact_nerv_vq/`. Avoid HiNeRV trainer + PR110.

**Codex C — V2 SNeRV source-forward → exact row (must pay rent or be deprioritized).**
- Close the source-forward causal proof + TUB DROP_OR_REIFY + LF/HF/MFU/HFR binding → export → B2 exact
  eval → `CandidateActionEvaluation`. No SNeRV section is admitted because it exists; only causal +
  archive-real + score-real + rent-positive. Files: SNeRV lane. Avoid HiNeRV/PR110/PACT-VQ.

**Codex D — artifact index + dashboard mining (the `Vatlas` atom/atlas lane; the canonical `V6`
designation is RESERVED for the operator's incoming V6 design memo — do NOT claim `V6` until it lands).**
- Build `tools/index_pact_artifacts.py`: walk `experiments/results/`, `.omx/research/`,
  `/Volumes/VertigoDataTier/pact/` → emit `artifact_index.jsonl` (path, schema, vehicle, authority_tier,
  metric_family, archive_sha256, d_seg/d_pose/bytes, blockers, stale). Then
  `tools/backfill_candidate_action_rows.py` converts eligible historical artifacts into typed rows so
  V3 inherits scattered knowledge in one currency. Read-only mining; no training. Avoid all trainer files.

## Non-overlap rules
- One lane per agent. Cite your `lane_id` in your first commit. Check `.omx/state/lane_registry.json` +
  `active_lane_dispatch_claims.md` before any dispatch; claim via `tools/claim_lane_dispatch.py`.
- Every lane's output is a `CandidateActionEvaluation` (or an exact blocker artifact) — never prose-only.
- authority_tier + metric_family on every row. No score claim from advisory/proxy. No upstream edits.

## Tests to run before committing
`PYTHONPATH=src .venv/bin/python -m pytest src/tac/optimization/tests/ -q` (V3 surface) + your lane's
dedicated tests. Commit `.py` only after `review_tracker mark-file ... reviewed`.

## End-of-turn contract (every agent, every turn)
Committed artifact + exact blocker + next command + burning question. No circles. No leaf work unless
it unlocks a branch. Bold, fast, evidence-first.
