# Codex findings: xhigh sparse residual / scorer-gradient review

Date: 2026-05-20T21:15Z
Reviewer: Codex xhigh adversarial review worker
Scope: sparse residual oracle, scorer-gradient sparse residual, decode helper,
focused tests, result artifacts under
`experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/`.
Authority: review memo only; no code or artifact edits.

## Summary verdict

Do not promote or dispatch these sparse-residual results from the current
artifact state. The authority markings are mostly correct (`score_claim=false`,
`promotion_eligible=false`, advisory axis labels, exact-CUDA blockers), and the
latest upstream-decoded scorer-gradient smoke still regresses after byte charge.
However, earlier result JSONs were built against the wrong target decode
semantics and must be treated as tainted for gradient/local-pair conclusions.

Current code also has one real index-mapping guard gap: both gradient selectors
trust `RawVideoShape.height/width` without checking the actual gradient H/W. A
caller can silently map crop-space pixels into full-frame global indices.

## Findings

### High: Gradient selector can silently mis-map pixels when gradient H/W does not match `RawVideoShape`

Evidence:

- `src/tac/optimization/scorer_gradient_sparse_residual.py:116-125` validates
  gradient/residual equality, rank, frame count, and channels, but not
  `grad.shape[1:3] == (shape.height, shape.width)`.
- `src/tac/optimization/scorer_gradient_sparse_residual.py:152-156` computes
  `pixels_per_frame` from `shape.height * shape.width` and uses it to convert
  local flat offsets into global full-video indices.
- The new budgeted selector repeats the same pattern:
  `src/tac/optimization/scorer_gradient_sparse_residual.py:196-205` and
  `src/tac/optimization/scorer_gradient_sparse_residual.py:269-274`.
- Repro command run during review selected a full-frame index from a 1x1
  gradient instead of raising:
  `.venv/bin/python - <<'PY' ... select_gradient_aligned_residuals(...) ...`
  printed `{'selected_indices': [0], 'n_selected': 1}`.

Impact: future scorer-gradient, saliency-mask, crop, or downsampled-gradient
callers can write sparse correction bytes to the wrong frame pixels while all
existing tests still pass.

### High: Earlier target-decode bug taints the first sparse-residual and scorer-gradient conclusions

Evidence:

- The sparse-residual memo records target decode via ffmpeg:
  `.omx/research/codex_findings_sparse_residual_oracle_charged_smoke_20260520T204441Z_codex.md:37-43`
  and target SHA `4f1ca43f...`.
- The repaired decode helper now uses upstream evaluator semantics:
  `tools/decode_upstream_video_to_raw.py:60-83` imports
  `upstream.frame_utils.yuv420_to_rgb` and records the decode semantics in the
  manifest.
- The old scorer-gradient local-veto artifact points at `target/0.raw` with
  SHA `4f1ca43f...`:
  `experiments/results/.../scorer_gradient_pose_pair508_k512_d1_localveto_20260520_codex.json:198-199`.
- The corrected scorer-gradient artifact points at `target/upstream_av_0.raw`
  with SHA `bb9cb031...`:
  `experiments/results/.../scorer_gradient_pose_pair508_k512_d1_upstream_target_20260520_codex.json:198-199`.
- The target change materially changes the pair objective: old objective
  `0.194016173...` / pose dist `0.003764227...`
  at `...localveto_20260520_codex.json:230-235`; corrected objective
  `0.022882713...` / pose dist `5.2361858e-05`
  at `...upstream_target_20260520_codex.json:230-235`.

Impact: the old local pair improvement magnitude and old selected pixels are
not source-faithful. The corrected upstream-target run still regresses after
charge (`advisory_score=0.193390924...`, `delta_vs_baseline_score=+0.0013295`,
`score_claim=false` at
`...upstream_target_20260520_codex.json:276-287`), so the negative direction is
not reversed, but the older artifacts should be marked superseded rather than
used as evidence.

### Medium: Legacy result artifacts have mutable manifest/custody paths

Evidence:

- The old local-veto summary points at a generic manifest path:
  `...scorer_gradient_pose_pair508_k512_d1_localveto_20260520_codex.json:272`.
- That same manifest path now contains the corrected upstream-target run:
  `.../pose_pairs_508_k512_d1/scorer_gradient_sparse_residual_manifest.json:185-186`.
- Current partner code partially fixes future scorer-gradient runs by including
  a target-hash-derived `candidate_id` in the candidate directory and refusing
  accidental overwrites:
  `tools/run_scorer_gradient_sparse_residual_smoke.py:117-124`.
- The sparse oracle smoke still uses only `k{top_k}_d{delta}_{frame_selector}`
  for the candidate directory and writes a fixed manifest inside it:
  `tools/run_sparse_residual_oracle_smoke.py:110-119` and
  `tools/run_sparse_residual_oracle_smoke.py:188`.

Impact: old summaries embed their candidate rows, but their
`candidate_manifest` and advisory log paths can dereference a later run. This
is enough to confuse result custody in reviews and ledgers.

### Low: Sparse-oracle cleanup has a no-visible-change raw retention edge case

Evidence:

- If advisory is requested but the candidate has no visible raw change,
  `tools/run_sparse_residual_oracle_smoke.py:171-178` records a skipped advisory
  but the cleanup condition only deletes when advisory returned `returncode==0`
  or advisory was not requested.

Impact: a no-op sparse oracle run with `--cleanup-candidate-raw --run-advisory`
can leave a 3.4 GB `0.raw` behind. I found no remaining `>100M` files in the
two reviewed result directories at the end of this review.

## Residual risks

- The current smoke remains advisory-only: raw-output evaluator custody, not
  stock `inflate.sh archive_dir output_dir file_list` custody.
- Local pair improvements are not sufficient evidence because global advisory
  score still regresses after byte charge.
- MPS gradient selection and macOS CPU advisory scoring are separate axes; no
  CUDA or promotion inference is valid.
- Existing tests do not exercise the tool-level decode provenance, candidate
  directory overwrite guard, cleanup edge cases, or shape mismatch rejection.

## Recommended immediate fixes

1. Add explicit selector guards:
   `grad.shape[1] == shape.height`, `grad.shape[2] == shape.width`, and every
   `frame_index` within `[0, shape.frames)`. Mirror this in both aligned and
   budgeted selectors.
2. Add tests that fail today for gradient/shape mismatch and out-of-range
   frame indices.
3. Mark the `4f1ca43f...` target-decode artifacts superseded in the next result
   ledger; use only the upstream-decoded `bb9cb031...` run for follow-up
   gradient decisions.
4. Apply the scorer-gradient `candidate_id`/overwrite guard pattern to
   `tools/run_sparse_residual_oracle_smoke.py`.
5. For a rerun, emit a fresh SHA-suffixed candidate directory and summary whose
   `candidate_manifest` path cannot be overwritten by later smokes.

## Tests/commands run

- `git status --short --branch`
- `tail -n 40 .omx/state/subagent_progress.jsonl`
- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, top memory entries, latest
  target memos/artifacts, and current target source/test files.
- `.venv/bin/python -m pytest -q src/tac/tests/test_sparse_residual_oracle.py src/tac/tests/test_scorer_gradient_sparse_residual.py` -> `7 passed`
- `.venv/bin/python -m py_compile src/tac/optimization/sparse_residual_oracle.py tools/run_sparse_residual_oracle_smoke.py src/tac/optimization/scorer_gradient_sparse_residual.py tools/run_scorer_gradient_sparse_residual_smoke.py tools/decode_upstream_video_to_raw.py` -> pass
- Shape-mismatch repro snippet for `select_gradient_aligned_residuals` -> selected index `[0]` instead of raising
- `find ... -type f -size +100M -print` over the two reviewed result dirs -> no output at final check
