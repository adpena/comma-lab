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

## Minimal Implementation Needed

- A compression-time action source that maps each selected pair to explicit PR85 stream/value deltas or to a decoded-output-parity recode.
- A reviewed runtime contract proving those stream families are consumed without scorer loads or sidecars.
- Non-noop payload or decoded-output proof in the candidate manifest.
- `tools/claim_lane_dispatch.py claim ...` before any exact CUDA auth eval dispatch.

## 2026-05-04T04:50Z Supersession: QRGB Pair Atoms Dispatched

The PR90 QRGB transfer path supplied explicit PR85-native stream/value action
evidence for three supported pair atoms.  The unsupported `randmulti` action was
excluded because `experiments/build_pr85_pair_atom_candidates.py` does not
decode that stream family.  The remaining `bias` and `region` actions produced
byte-closed, non-noop archive candidates against the PR85 public anchor.

Source custody:

- PR85 source archive bytes: `236328`
- PR85 source archive sha256:
  `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- QRGB transfer evidence:
  `experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json`
- Pair-action spec:
  `experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/action_spec_bias_region.json`
- Pair-action spec sha256:
  `b810fc7b0ded5c9fb7bd1e675c611d26b1698c00f222d97e639e5542d9b3975f`
- Runtime contract:
  `experiments/results/pr85_qrgb_pair_atom_archive_candidates_20260504_codex/runtime_contract_bias_region.json`
- Runtime contract sha256:
  `19240fb0457290158cc6ebcdc249de09cd526784e7b2c8f87cb21d4a8a9c3932`
- Scorer-gradient plan stable digest:
  `0c47de9d4666f72793821776a447562db8eca1201ceb05b09cb22a222741d427`

Built candidates:

- `pr85_qrgb_f1_bias_pair_0060`
  - bytes: `236336`
  - sha256:
    `81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81`
  - changed segment: `bias`
  - byte delta vs PR85: `+8`
  - fixed-runtime atom preflight: `ready`
- `pr85_qrgb_f1_bias_pair_0164`
  - bytes: `236335`
  - sha256:
    `d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e`
  - changed segment: `bias`
  - byte delta vs PR85: `+7`
  - fixed-runtime atom preflight: `ready`
- `pr85_qrgb_f1_region_pair_0197`
  - bytes: `236335`
  - sha256:
    `236751af46a9c98fa286ecfe613c23a2b96bffbe31784da052304701e02b71c6`
  - changed segment: `region`
  - byte delta vs PR85: `+7`
  - fixed-runtime atom preflight: `ready`

Remote dispatch:

- Staging manifest:
  `.omx/state/pr85_qrgb_pair_atoms_t4_20260504T0450Z_manifest.json`
- Lightning workspace verification:
  `REMOTE_MANIFEST_VERIFY OK`, `1116` files, `17594414` bytes.
- Exact eval jobs:
  - `exact_eval_pr85_qrgb_f1_bias_pair_0060_t4_20260504T0450Z`
  - `exact_eval_pr85_qrgb_f1_bias_pair_0164_t4_20260504T0450Z`
  - `exact_eval_pr85_qrgb_f1_region_pair_0197_t4_20260504T0450Z`
- Machine class: `g4dn.xlarge` / T4-equivalent.
- Dispatch claims:
  `.omx/state/active_lane_dispatch_claims.md`
- Current status at dispatch: `Pending`, zero cost.

Evidence status:

- These are exact-eval candidates, not score claims.
- Promotion/ranking remains blocked until a canonical CUDA
  `contest_auth_eval.adjudicated.json` is harvested for the exact archive bytes.

## 2026-05-04T05:14Z Exact Singleton Verdict

All three singleton T4 probes completed with A++ exact CUDA evidence and all
three were slightly worse than the PR85 anchor.

| candidate | hardware | score | delta_vs_pr85 | bytes | sha256 |
|---|---|---:|---:|---:|---|
| `pr85_qrgb_f1_bias_pair_0060` | T4 | `0.2580739080216157` | `+0.000007797727637870455` | `236336` | `81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81` |
| `pr85_qrgb_f1_bias_pair_0164` | T4 | `0.25808234771935784` | `+0.000016237425379983517` | `236335` | `d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e` |
| `pr85_qrgb_f1_region_pair_0197` | T4 | `0.2580777531130102` | `+0.00001164281903232034` | `236335` | `236751af46a9c98fa286ecfe613c23a2b96bffbe31784da052304701e02b71c6` |

Diagnostic L40S hedges also completed before stop landed and were mirrored:

| candidate | hardware | score | promotion eligible |
|---|---|---:|---|
| `0060` | L40S | `0.25931028308646` | false |
| `0164` | L40S | `0.25931066136375286` | false |
| `0197` | L40S | `0.2593129178321656` | false |

Decision:

- Treat this QRGB singleton wave as measured-implementation negative evidence.
- Do not dispatch the prepared PR85 QRGB combo archives from these atoms.
- Preserve the builder as a PR91/HPAC-basin transfer tool only, pending a
  PR91-specific byte-closed builder and exact PR91 replay confirmation.
