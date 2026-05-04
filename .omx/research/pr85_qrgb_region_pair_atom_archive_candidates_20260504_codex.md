# PR85 Pair-Atom Candidate Readiness

- tool: `experiments/build_pr85_pair_atom_candidates.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- dispatch_unlocked: true
- blocker_class: `none`

## Source Anchor

- archive bytes: 236328
- archive sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- known PR85 T4 anchor match: True

## Scorer-Gradient Intake

- plan: `/tmp/pr85_scorer_gradient_atoms_plan_baseline.json`
- plan status: `passed`
- stable digest: `0c47de9d4666f72793821776a447562db8eca1201ceb05b09cb22a222741d427`

## Top Pair Opportunities

- pair_0192: break_even_bytes=516.6922114573065 ranking_score=0.00034404413500734173
- pair_0060: break_even_bytes=484.2862579445188 ranking_score=0.00032246634072639115
- pair_0164: break_even_bytes=465.22756469211794 ranking_score=0.0003097759391894709
- pair_0197: break_even_bytes=452.8814701277624 ranking_score=0.0003015551815877018
- pair_0070: break_even_bytes=448.6622211777276 ranking_score=0.0002987457568988698
- pair_0496: break_even_bytes=447.820449774858 ranking_score=0.00029818525587378686
- pair_0106: break_even_bytes=437.8045133683296 ranking_score=0.00029151605494359765
- pair_0522: break_even_bytes=432.0508240749086 ranking_score=0.00028768490941409006

## Runtime Investigation

- Existing PR85 bundle code can slice and repack `x`; existing bridge code can materialize `qpost.bin`/`QRM1` and group-level sparse actions.
- Existing final-bias code stacks a coarse 300-byte `fb` atom, not pair-specific stream actions.
- No reviewed pair-action runtime contract was found in the existing PR85 runtime surfaces, and the scorer-gradient plan does not supply stream/value deltas.

## Readiness Decision

- Local byte-closed pair-atom candidates were built. Exact eval remains blocked until a lane claim is recorded.

## Exact Eval Update - 2026-05-04T05:46Z

- `pr85_qrgb_f1_region_pair_0060`
  - evidence grade: `A++ contest T4`
  - archive bytes: `236335`
  - archive sha256: `2dbf9e7f0d1951a4dac44fb365083b5587b0a52df3c1c534cada8a0d065506d2`
  - score: `0.2581684069751288`
  - delta vs PR85 T4 anchor: `+0.00010229668115091517`
  - SegNet: `0.00057185`
  - PoseNet: `0.00019025`
  - artifact dir: `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f1_region_pair_0060_t4_20260504T0528Z`
  - decision: measured singleton negative; do not redispatch or stack this atom without new interaction evidence.
- `pr85_qrgb_f1_region_pair_0164`
  - evidence grade: `A++ contest T4`
  - archive bytes: `236335`
  - archive sha256: `2dc89260bc2d7112f80163236ea194a60837793367a0a085b2c64e76cc631062`
  - score: `0.25807315802161573`
  - delta vs PR85 T4 anchor: `+0.000007047727637876644`
  - SegNet: `0.00057185`
  - PoseNet: `0.00018942`
  - artifact dir: `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f1_region_pair_0164_t4_20260504T0528Z`
  - decision: measured singleton negative; confirms the current PR90-derived bias/region singleton transfer is not score-moving absent interaction evidence.

## Minimal Implementation Needed

- A compression-time action source that maps each selected pair to explicit PR85 stream/value deltas or to a decoded-output-parity recode.
- A reviewed runtime contract proving those stream families are consumed without scorer loads or sidecars.
- Non-noop payload or decoded-output proof in the candidate manifest.
- `tools/claim_lane_dispatch.py claim ...` before any exact CUDA auth eval dispatch.
