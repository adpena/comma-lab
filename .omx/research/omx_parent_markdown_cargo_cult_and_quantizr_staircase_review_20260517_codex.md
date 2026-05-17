# OMX Parent Markdown Cargo-Cult And Quantizr Staircase Review - 2026-05-17

## Why This Exists

The operator warned that the relevant cargo-cult / L5 / staircase documents may
live outside `.omx/research`. I therefore treated the non-research `.omx`
Markdown surface as part of the evidence base, not as stale scratch.

This review extends the earlier parent-scope and no-ignore scans by reading the
Quantizr, PoseNet, scorer-architecture, entropy-coder, and stacking notes in
the ignored Claude/OMX memory snapshot and applying them to the live Quantizr
5-stage staircase WIP.

## Scan Surface

Reused / refreshed commands:

```bash
find .omx -name '*.md' ! -path '.omx/research/*' -print
rg -l -i --hidden --no-ignore \
  'cargo[-_ ]?cult|local minima|local minimum|assumption|stale|authority|not current|no signal loss|stack|arithmetic|entropy|posenet|segnet|quantizr|time[-_ ]?trav|tt5l|l5|rule #?6|pr101|pr95' \
  .omx --glob '*.md' --glob '!.omx/research/**'
```

Prior full scan counts remain valid from
`.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`:

| Bucket | Markdown files |
|---|---:|
| `.omx/auto_memory_snapshot_20260504T230223Z` | 562 |
| `.omx/context` | 28 |
| `.omx/interviews` | 1 |
| `.omx/plans` | 4 |
| `.omx/root` | 2 |
| `.omx/specs` | 1 |
| `.omx/state` | 22 |
| `.omx/tmp` | 16 |

## Current Authority Result

No non-research `.omx` Markdown supersedes the active May 17 state:

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/state/active_lane_dispatch_claims.md`

`.omx/notepad.md` is explicitly historical April AV1 / Track-B memory.
`.omx/release_manifest_v0.2.0-rc1.md` is release hygiene, not current score
authority. The ignored Claude/OMX snapshot preserves signal, but it is not a
score, dispatch, promotion, or architecture-lock authority.

## Cargo-Cult Carry-Forward

| Source | Signal | Current classification | Routing consequence |
|---|---|---|---|
| `project_quantizr_definitive_binary_analysis.md` | PR55 used a 5-stage schedule, FiLM+DSConv, single odd-frame mask, dual heads, no optical flow/warp | HARD-EARNED for PR55 only; CARGO-CULTED if copied as universal proof | Quantizr staircase helper may land as a scaffold, but cannot claim current-frontier movement without trainer adoption + byte-closed eval. |
| `project_5stage_quantization_advantage.md` | The April memory says our later five-stage quantization stack was smarter than vanilla Quantizr QAT | CONFLICTING HISTORICAL SIGNAL | Do not cite Quantizr 5-stage as automatically superior; expose schedule choices and prove on current substrate. |
| `feedback_curriculum_must_use_full_score.md` | Difficulty must use `100*seg + sqrt(10*pose)`, not PoseNet-only | HARD-EARNED bug-class lesson | Any Quantizr/L5 staircase adoption must use full contest score difficulty and report per-component deltas. |
| `feedback_preprocessing_dead_end.md` + `feedback_posenet_sensitivity.md` | Blur / broad preprocessing destroyed PoseNet | HARD-EARNED for broad spatial degradation | L5/Rule #6 component-moving packets must not reintroduce spatial nullspace assumptions without scorer-awareness proof. |
| `project_scorer_architecture_confirmed.md` | SegNet uses last frame; PoseNet consumes YUV6 pair signal | HARD-EARNED scorer anatomy | Probes must measure whether side-info reaches actual scorer inputs, not only latent/parser consumption. |
| `feedback_arithmetic_qint_codec_pr106_latents_unviable_brotli_already_below_entropy_20260504.md` | Plain zero-order arithmetic lost to Brotli on PR106 latents | HARD-EARNED no-retread warning | Rule #6 entropy stack must be section/context conditioned; arithmetic is terminal after symbol formation, not a first step. |
| `project_codec_stacking_composition_canonical_orders_20260429.md` | Correct order is representation -> prediction -> quantization/VQ -> hyperprior -> arithmetic -> archive | HARD-EARNED stack-order prior | L5-v2 / Rule #6 work should build symbols and side-info first, then lower with entropy coding. |

## Quantizr Staircase WIP Review

Files reviewed:

- `src/tac/training_curriculum/quantizr_5_stage_staircase.py`
- `src/tac/tests/test_quantizr_5_stage_staircase.py`

Observed before fixes:

- Functional tests were green: `42 passed`.
- Ruff failed on import sorting, `dict()` literal style, blind
  `pytest.raises(Exception)`, tuple concatenation style, raw regex escaping,
  unused `field`, unsorted `__all__`, unused `noqa`, and nested `if`.
- The module docstring cited `tac.quantization.FakeQuantFP4`, but the actual
  FP4 primitive is under `tac.fp4_quantize` / `tac.quantization_wave`, not
  `tac.quantization`.
- The docstring over-stated the Quantizr anchor as if it were current
  score-moving authority. PR55's 0.33 is historical architecture/training
  evidence, not proof of movement at the 0.192 frontier.
- The helper was not exported from `tac.training_curriculum.__init__`, so it
  was not discoverable through the package surface.

Fixes applied:

- Cleaned ruff findings in the helper and test.
- Softened the authority wording: the schedule is a training-practice prior,
  not a score claim.
- Corrected FP4 citations to `tac.fp4_quantize.fake_quant_fp4` /
  `tac.fp4_quantize.FakeQuantFP4`.
- Exported the helper from `tac.training_curriculum` and added a package
  discoverability test.

## Current Decision

The Quantizr staircase is acceptable to land as a production-hardened training
scaffold with `score_claim=false`. It is not a frontier candidate by itself.

The next score-relevant act is not another council memo about the staircase.
It is a byte-closed adoption by a real L5-v2 / Rule #6 trainer that proves:

1. full contest-formula difficulty is used;
2. BN/EMA/LSQ/FP4 transition records fire on the actual model;
3. a consumed archive byte section changes;
4. PoseNet/SegNet components move on the correct CPU/CUDA axis;
5. exact result review classifies the movement before promotion.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_quantizr_5_stage_staircase.py -q
.venv/bin/python -m ruff check src/tac/training_curriculum/quantizr_5_stage_staircase.py src/tac/tests/test_quantizr_5_stage_staircase.py
```

The final command results are recorded in the commit message / terminal output
for this landing.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

