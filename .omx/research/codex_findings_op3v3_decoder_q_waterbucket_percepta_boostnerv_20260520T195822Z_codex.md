# Codex Findings: OP3-V3 Decoder-Q Waterbucket, Percepta, BoostNeRV

UTC: 2026-05-20T19:58:22Z
Owner: Codex
Scope: PR110/FEC6 frontier engineering follow-up; no live PR110 submission files edited.

## Summary

Three threads landed:

1. OP3-V3/FEC6 decoder-q waterbucket candidates were materialized, inflated through the stock runtime, and advisory-scored on local macOS CPU.
2. Percepta programs-into-transformer-weights research was adversarially rechecked by three xhigh subagents; the integration helper was corrected after review.
3. BoostNeRV was researched by an xhigh subagent; the local citation/mechanism appears to need correction before that WIP is promoted.

No score or promotion claim is made here. All advisory scores are `[macOS-CPU advisory decoder-q]`, not contest CUDA.

## OP3-V3/FEC6 Decoder-Q Result

Inputs:

- Source runtime: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1`
- Baseline raw: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/inflated/0.raw`
- Targeted feasibility: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_targeted_feasibility_20260520_codex.json`
- Waterbucket plan: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/decoder_q_signed_waterbucket_plan_post_smoke_20260520_codex.json`

Artifacts:

- Inflate controls: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/decoder_q_signed_waterbucket_inflate_controls_post_smoke_20260520_codex/summary.json`
- Advisory batch: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/decoder_q_signed_waterbucket_advisory_post_smoke_20260520_codex/summary.json`
- Decision packet: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/decoder_q_signed_waterbucket_decision_packet_post_advisory_20260520_codex.json`

Empirical result:

- Fixed-length ZIP_STORED candidates: 13
- Visible stock-inflate output changes: 13/13
- Advisory scorer successes: 13/13
- Advisory improvements vs baseline `0.19206142414659494`: 0/13
- Best advisory candidate: `ee3b1832f8549a40`
- Best advisory score: `0.19247233847162176`
- Best delta vs baseline: `+0.00041091432502682324`
- Decision packet recommendation: `do_not_dispatch_exact_eval__widen_search`

Best-to-worst completed advisory rows:

| candidate | score | delta vs baseline | PoseNet | SegNet |
| --- | ---: | ---: | ---: | ---: |
| `ee3b1832f8549a40` | 0.192472338471622 | +0.000410914325027 | 0.00002944 | 0.00056447 |
| `02f57998bfe58eb8` | 0.192491424146595 | +0.000430000000000 | 0.00002943 | 0.00056469 |
| `255155da7bb15b3a` | 0.192491424146595 | +0.000430000000000 | 0.00002943 | 0.00056469 |
| `e6a64488067af98b` | 0.192491424146595 | +0.000430000000000 | 0.00002943 | 0.00056469 |
| `58904052b32384de` | 0.192544424146595 | +0.000483000000000 | 0.00002943 | 0.00056522 |
| `7fc20a29e4fb1b77` | 0.192544424146595 | +0.000483000000000 | 0.00002943 | 0.00056522 |
| `b1bf64b136e2e387` | 0.192595338471622 | +0.000533914325027 | 0.00002944 | 0.00056570 |
| `ca1e19a91a304169` | 0.192595338471622 | +0.000533914325027 | 0.00002944 | 0.00056570 |
| `0a6d176fe5cead2e` | 0.192681252301730 | +0.000619828155136 | 0.00002945 | 0.00056653 |
| `29caff465ae8a2ee` | 0.192725252301730 | +0.000663828155136 | 0.00002945 | 0.00056697 |
| `78066f20f4af5d34` | 0.192933990825068 | +0.000872566678473 | 0.00002948 | 0.00056897 |
| `553b3d791f6353fe` | 0.193098078478202 | +0.001036654331607 | 0.00002947 | 0.00057064 |
| `e97f4613b5c16a74` | 0.193356425357723 | +0.001295001211128 | 0.00002965 | 0.00057270 |

Interpretation:

The OP3-V3 byte-gradient target and q-symbol mapping are real, consumed by stock inflate, and scorer-visible. However, this signed waterbucket branch is locally dominated: every measured branch increases score, primarily via SegNet regression. Do not spend exact-eval budget on these candidates. Widen to a different signal surface: runtime-side deterministic transforms, source-faithful BoostNeRV/TAT adaptation, or a learned/offline proposal engine that changes representation rather than only local final-head q-symbols.

## Percepta Research And Pact Translation

Subagent memos:

- `.omx/research/codex_findings_percepta_provenance_20260520T185739Z_codex.md`
- `.omx/research/codex_findings_percepta_wasm_mechanics_20260520T185820Z_codex.md`
- `.omx/research/codex_findings_percepta_pact_integration_20260520T185519Z_codex.md`
- Earlier local memo: `.omx/research/codex_findings_programs_into_weights_wasm_transformer_20260520T193000Z_codex.md`

Important correction:

Percepta now has a March 25, 2026 construction/code-release post and an Apache-2.0 GitHub repo. The supported claim is an analytically constructed code-to-weights transformer VM / WASM-subset execution system, not a trained LLM that learned arbitrary WebAssembly.

Primary sources:

- `https://www.percepta.ai/blog/can-llms-be-computers`
- `https://www.percepta.ai/blog/constructing-llm-computer`
- `https://github.com/Percepta-Core/transformer-vm`

Code landed by subagent and corrected by Codex review:

- `src/tac/optimization/percepta_microprogram_plan.py`
- `tools/plan_percepta_microprogram_candidate.py`
- `src/tac/tests/test_percepta_microprogram_plan.py`

Codex correction:

The initial helper emitted smoke commands that passed a ZIP path to `--archive-bin`. That is wrong for the decoder-q mutator, which expects the raw archive member bytes. The helper now points at the retained PR110/FEC6 source member:

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/data_dir/x`

and at the retained baseline raw:

`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/inflated/0.raw`

Pact verdict:

Use Percepta as a plausibility pattern for tiny deterministic circuits compiled into charged weights or archive bytes. Do not ship a general VM/interpreter in `inflate.py`, do not claim arbitrary C/WASM support, and do not cite it as score evidence.

## BoostNeRV Research

Subagent memo:

- `.omx/research/codex_findings_boostnerv_research_integration_20260520T191425Z_codex.md`

Primary sources identified:

- `https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Boosting_Neural_Representations_for_Videos_with_a_Conditional_Decoder_CVPR_2024_paper.html`
- `https://arxiv.org/abs/2402.18152`
- `https://github.com/Xinjie-Q/Boosting-NeRV`

Important correction:

Primary Boosting-NeRV is Zhang et al. CVPR 2024 conditional decoder work with TAT/SFT, not the locally referenced "Liu ECCV 2024 iterative residual refinement" mechanism. No primary source was found for the local citation/mechanism. The active WIP under `src/tac/substrates/boost_nerv/` was left untouched because a sibling worker owns it.

Recommended next action:

After the active BoostNeRV owner clears, correct scaffold metadata first. Then either rename the current residual-head mechanism as a Pact-only hypothesis or replace it with a source-faithful `Boosting-HNeRV-TAT` adapter. First gate should be a tiny local advisory smoke, not dispatch or PR110 edits.

## Inflate.py Surface

The operator is right that `inflate.py` may have underexplored score-moving capacity. Compliance-safe categories:

- Generic deterministic postprocessors: deblocking, color/temporal smoothing, sharpening, foveated refinement.
- Tiny source-generic circuits: runtime applies a generic operation; data-bearing bits live in charged archive bytes or weights.
- Deterministic coordinate/camera/frame-index transforms with no scorer calls or network access.
- Runtime optimization: mmap/streaming, copy reduction, stable thread controls, deterministic output order.

Unsafe categories:

- Hardcoded data payloads in source.
- Scorer-aware logic, label leakage, network fetches, or nondeterministic sampling.
- Large diffusion/model inference or a general VM in `inflate.py` unless distilled into a tiny deterministic operator and byte-accounted.

Next useful artifact: an `inflate_surface_audit` plus one identity-guarded postprocessor smoke. The smoke must prove stock `inflate.sh` output changes under a generic transform, records raw-output SHA, and then runs local advisory before any exact-eval spend.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_decoder_q_decision_packet.py \
  src/tac/tests/test_fec6_decoder_mutations.py \
  src/tac/tests/test_fec6_byte_targets.py \
  src/tac/tests/test_byte_score_impact.py \
  src/tac/tests/test_master_gradient_exploits_end_to_end.py \
  src/tac/tests/test_percepta_microprogram_plan.py
```

Result: `47 passed, 1 skipped`.

No live PR110 body, README, `archive_manifest.json`, or submission archive was edited.
