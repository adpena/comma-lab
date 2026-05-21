# Codex Findings: MLX Auth Scorer Comprehensive Adversarial Review

Date: 2026-05-21T22:39:31Z

- lane_id: lane_codex_adversarial_review_mlx_auth_scorer_20260521
- reviewer: codex
- evidence_grade: macOS-MLX-research-signal + code-review
- score_claim: false
- promotion_eligible: false
- rank_or_kill_eligible: false
- exact_eval_executed: false

## Scope

Reviewed recent code and Markdown landings from the last several hours, with
deep focus on the MLX port of the canonical upstream auth scorer:

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
- `src/tac/tests/test_mlx_scorer_adapters.py`
- recent MLX scorer `.omx/research/*.md` memos
- recent committed `.md` files via high-risk score/promotion/authority scan
- currently untracked council memos surfaced by `git status --short -- '*.md'`

## Findings

### 1. HIGH - T3 MLX council memo is not durable and contains stale/false commit-registration wording

Evidence:

- `git status --short -- '*.md'` shows
  `.omx/research/t3_grand_council_symposium_mlx_port_byte_fidelity_determinism_exploit_unlock_20260521.md`
  is untracked.
- The same memo says op-routable #4 has three Catalog #344 equations
  "REGISTERED in same commit batch" at line 59, but `rg` finds those equation
  names only in research memos, not in the canonical equation registry or
  `CLAUDE.md`.
- Lines 386-394 soften this to "queued for registration" with
  `FORMALIZATION_PENDING`, contradicting the frontmatter decision text.
- Line 418 claims commit via `tools/subagent_commit_serializer.py`, but the
  memo was still untracked at review time.

Impact:

Downstream agents can treat unregistered equations as canonical and can assume a
serializer landing that did not actually happen. This is exactly the kind of
Markdown authority drift that creates signal loss while looking compliant.

Required follow-up:

Preserve the memo as historical provenance, but add an erratum or successor memo
that changes the equation status to queued/pending until the actual registry
landing exists. Do not cite the three equations as registered canonical helpers
until a source/registry anchor exists.

### 2. HIGH - Full auth-scorer parity remains open despite strong PoseNet module parity

Evidence:

- `run_mlx_posenet_nchw(...)` accepts already-preprocessed NCHW tensors and
  returns PoseNet outputs only (`src/tac/local_acceleration/mlx_scorer_adapters.py`
  lines 868-874).
- The end-to-end PoseNet test uses a random `(1, 12, 64, 80)` tensor, not the
  raw auth-eval `(B,T,H,W,C)` pipeline or byte-closed scorer-input cache
  (`src/tac/tests/test_mlx_scorer_adapters.py` lines 305-315).
- The PoseNet memo states the boundary correctly: "SegNet and the full evaluator
  aggregation path remain open" (`codex_findings_mlx_posenet_end_to_end_parity`
  lines 60-61).
- SegNet coverage is still encoder-side. The latest committed SegNet memo says
  stages 2-6, the exact `TimmUniversalEncoder` feature-return contract, the U-Net
  decoder, and segmentation head remain open (`codex_findings_mlx_segnet_efficientnet_stage_prefix_parity`
  lines 57-61).

Impact:

The current MLX path is a useful component-parity surface, not an auth scorer.
It must not drive rank/kill, promotion, public score, or "contest-grade" claims.

Required follow-up:

Land a byte-closed evaluator wrapper that compares PoseNet + SegNet + aggregation
against the recovered Modal/Linux auth-eval anchor over the scorer-input cache.
Until then, keep `score_claim=false`, `promotion_eligible=false`, and
`rank_or_kill_eligible=false`.

### 3. MEDIUM - SegNet GPU drift is material and not pinned at the worst observed block

Evidence:

- The SegNet block memo records MLX GPU drift of `0.13507318496704102` max abs
  for `blocks[0][0]` and `0.03046560287475586` for `blocks[1][0]`
  (`codex_findings_mlx_segnet_efficientnet_block_parity` lines 39-42).
- The same memo labels GPU drift as non-authoritative (`lines 44-45`).
- The current test suite has a GPU drift test for `blocks[1][0]` only
  (`src/tac/tests/test_mlx_scorer_adapters.py` lines 363-376). There is no
  regression guard that preserves the larger `blocks[0][0]` drift as an explicit
  blocker or known non-authority marker.

Impact:

Future callers could accidentally route SegNet scorer/search work through MLX GPU
and inherit a block-level error far above the CPU parity band. The docs say this
is non-authoritative; the executable guard surface is still weaker than the
documented risk.

Required follow-up:

Add either a fail-closed scorer-authority guard for MLX GPU SegNet paths or a
dedicated GPU-drift regression that records `blocks[0][0]` as a known
non-authoritative blocker. CPU remains the only MLX parity gate.

### 4. MEDIUM - Live SegNet feature-list adapter is passing, but was dirty/uncommitted at review time

Evidence:

- Dirty diff adds `MLXEfficientNetFeaturesAdapter`,
  `torch_efficientnet_features_to_mlx`, `run_mlx_efficientnet_features_nchw`,
  and `test_segnet_efficientnet_features_match_torch_on_mlx_cpu`.
- The test suite passes locally: `29 passed in 4.38s`.
- All 24 EfficientNet-B2 encoder blocks convert through the current adapter
  surface in a local introspection smoke.
- Canonical-size encoder feature smoke on `(1, 3, 512, 384)` produced feature
  shapes `[(1,16,256,192), (1,24,128,96), (1,48,64,48), (1,120,32,24),
  (1,352,16,12)]` and max_abs `0.0009255409240722656`.

Impact:

The live code is a real improvement over the latest committed SegNet stage-prefix
memo, but without a durable memo/commit it is still signal-loss-prone partner
WIP.

Required follow-up:

Commit the feature-list adapter and test with this review memo or a dedicated
successor landing memo. Keep decoder/segmentation-head/evaluator boundaries
explicit.

### 5. LOW - Broad Markdown scan did not find score-authority true flags, but two untracked memos remain

Evidence:

- High-risk scan over recent `.md` files did not find `score_claim: true`,
  `promotion_eligible: true`, `rank_or_kill_eligible: true`, or
  `ready_for_exact_eval_dispatch: true`.
- The untracked T4 distortion-axis council memo carries `score_claim: false`,
  `promotion_eligible: false`, `rank_or_kill_eligible: false`,
  `dispatch_attempted: false`, and `paid_dispatch_attempted: false` in its
  frontmatter.

Impact:

The broad Markdown surface is mostly disciplined on score authority, but
untracked council artifacts are still at risk of being lost or bypassed by
canonical readers.

Required follow-up:

Preserve untracked council memos through scoped git commits after sister-overlap
checks. Do not use `git add .`; stage exact files only.

## Verification

Commands run:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
29 passed in 4.38s

.venv/bin/python - <<'PY'
# introspected all EfficientNet-B2 encoder blocks; all converted successfully
PY

.venv/bin/python - <<'PY'
# canonical-size SegNet encoder feature smoke
# max_abs = 0.0009255409240722656
PY
```

## Review Verdict

The MLX scorer port is moving in the right order: primitives, PoseNet module
parity, SegNet encoder parity, then feature-list parity. The main correctness
boundary is unchanged: this is not yet a full auth scorer, and MLX GPU SegNet is
not an authority path. The biggest process defect is Markdown durability and
authority drift in the untracked T3 memo.
