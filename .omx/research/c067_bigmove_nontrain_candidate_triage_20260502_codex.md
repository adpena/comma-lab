# C067 Big-Move Nontraining Candidate Triage - 2026-05-02 Codex

## Scope

Planner added: `experiments/plan_c067_bigmove_nontrain_candidates.py`

Generated artifact:
`experiments/results/c067_bigmove_nontrain_candidate_triage_20260502/c067_bigmove_nontrain_candidate_triage.json`

This is a planning/triage artifact only. It reads existing structural-mask,
multiresolution, multimask, PMG/topology, micro-mask, postdecode-repair, and
SJ-KL artifacts. It does not load scorers, touch `.omx/state`, build archives,
claim lanes, or dispatch remote work. All emitted records keep
`score_claim=false`.

## Inputs Read

- Multiresolution/multimask: `c067_multiresolution_stack_plan.json`,
  `c067_multimask_reconciliation_20260502_cmg3a_reconciler_threshold_fix1`,
  and the fix1 exact-negative context for extra065k/extra072k.
- Topology/repair atoms: PMG atomtop byte screens, PMG stride1 dynamic
  ego-foveal pair-protected manifest, hotspot geometry exact-negative plan,
  post-negative poseguard topology policies, micro-mask reencode plan, and
  postdecode pair/class waterfill plan.
- Existing exact diagnostics: multimask extra065k/extra072k, PMG hotspot,
  PMG atomtop4068, hotspot top0128, micro-mask save12k, postdecode top10 and
  budget8000, and SJ-KL C067 diagnostics.

## Result

The planner found 25 nontraining candidates and ranked them under the C067
frontier:

- C067 frontier score: `0.31561703078448233`
- Frontier archive bytes: `276214`
- Unchanged-distortion sub-0.300 byte gate: `252760`

Top exact-evaluable candidate:

- `c067_postdecode_repair_save12k_exact_trace_pair_waterfill_budget4000`
  - Archive bytes: `243422`
  - Archive SHA-256:
    `25f890f449796f79c0da246758d05684c40ccce8b29a3bc2322b8958fa7ae489`
  - Expected net delta vs C067 under pair/class component-trace prior:
    `-0.081658451692`
  - Dispatch status in planner:
    `dispatchable_after_lane_claim_and_active_eval_check`
  - Important boundary: no dispatch was performed here.

Highest byte-only no-dispatch screens:

- PMG atomtop64/128/512/2048 remain `no_dispatch`: the byte screens are
  attractive under unchanged distortion, but same-family PMG exact negatives
  showed PoseNet collapse, so unchanged-distortion projections are not
  component evidence.
- Multimask threshold_fix1 extra066k/067k/068k are byte-closed and below the
  gate, but remain `no_dispatch` because same-lineage extra065k/extra072k exact
  diagnostics collapsed PoseNet.

Existing active diagnostic recorded but not duplicated:

- SJ-KL C067 L40S diagnostic: `315515` bytes, exact score
  `0.500865441062`
- SJ-KL C067 v2 k4 a5 cap32k diagnostic: `296999` bytes, exact score
  `0.438612629329`

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_plan_c067_bigmove_nontrain_candidates.py -q
.venv/bin/python experiments/plan_c067_bigmove_nontrain_candidates.py
```

Observed focused pytest result: `3 passed`.

No remote jobs were launched and no `.omx/state` exact-eval JSON was modified.
