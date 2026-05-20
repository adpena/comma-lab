# Codex Findings: XHigh Authority / Math / Engineering Rigor Review

UTC: 2026-05-20T21:21:17Z
Owner: Codex xhigh adversarial review worker
Scope: memo-only review of 2026-05-20 Percepta / BoostNeRV / decoder-q / inflate-postprocess / sparse-residual claims.
Write scope honored: this memo only. No source code, PR110 body, README, archive, or manifest edited.

## Preflight Notes

- Read `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, top MEMORY entries, `reports/latest.md`, the 2026-05-20 routing directive, and `.omx/state/subagent_progress.jsonl` tail.
- Re-derived frontier scan: current best is `0.1920513168811056` `[contest-CPU]` and `0.20533002902019143` `[contest-CUDA]`. The reviewed OP3/postprocess/sparse results are advisory-only and must not be promoted into frontier evidence.
- Worktree is heavily dirty with active sibling lanes. I did not pre-register or mutate lane state because the operator explicitly constrained write scope to this one memo.

## Findings

### HIGH - Percepta mechanics memo is now superseded and factually stale

Evidence:

- `.omx/research/codex_findings_percepta_wasm_mechanics_20260520T185820Z_codex.md:11-16` says the public Percepta page is only a blog claim and that no official Percepta compiler, test suite, or C-to-Wasm artifact was found.
- Same memo at `:187-190` says no public C compiler path, Wasm parser, or full test corpus was found.
- Later local sources contradict that:
  - `.omx/research/codex_findings_percepta_provenance_20260520T185739Z_codex.md:29-50`
  - `.omx/research/codex_findings_op3v3_decoder_q_waterbucket_percepta_boostnerv_20260520T195822Z_codex.md:74-82`
- Live source check: `Percepta-Core/transformer-vm` exists at HEAD `6cfee30dd7a8f5bffd76d0b0fcf2932fdd41fc97`, Apache-2.0, and its README exposes `wasm-compile`, `wasm-run`, `wasm-eval`, `wasm-specialize`, tests, C/WASM compilation requirements, supported/lowered opcode lists, and C examples.

Recommended correction:

Prepend a supersession note to the mechanics memo. Replace the executive claim with:

> Superseded by the provenance pass and official repo inspection: Percepta has an official Apache-2.0 `transformer-vm` repository with a C/WASM-to-token pipeline, analytical weight construction, test files, and C++/Python inference paths. The remaining caution is not "no official artifact"; it is "official artifact is analytical/code-to-weights, finite-scope/lowered WASM, no formal paper/arXiv found, no Pact score evidence, and no learned LLM acquiring arbitrary computation."

### HIGH - Advisory scores/deltas carry false precision

Evidence:

- `upstream/evaluate.py:92-100` computes the score internally, but writes PoseNet, SegNet, and compression rate rounded to 8 decimals and final score rounded to 2 decimals.
- `tools/run_raw_advisory_eval.py:80-100` parses those rounded report fields and recomputes `canonical_score` from them.
- Therefore the high-precision values in these memos are report-rounded reconstructions, not exact evaluator-internal scores:
  - OP3 waterbucket: `.omx/research/codex_findings_op3v3_decoder_q_waterbucket_percepta_boostnerv_20260520T195822Z_codex.md:37-40`, `:47-59`
  - inflate postprocess: `.omx/research/codex_findings_inflate_postprocess_surface_smoke_20260520T202752Z_codex.md:32-36`
  - sparse residual: `.omx/research/codex_findings_sparse_residual_oracle_charged_smoke_20260520T204441Z_codex.md:49-67`

Impact:

- The decisions are directionally stable for the reviewed negatives: OP3 best is worse by about `+4.1e-4`, luma/temporal postprocesses are much worse, and sparse residual is rate-only worse.
- But the precision is overstated. Example: exact byte-rate cost for `687` bytes is `25 * 687 / 37_545_489 = 0.0004574451007949317`, while the memo/JSON advisory delta is `+0.00045725000000002014` because it uses the rounded report rate.

Recommended correction:

For all advisory result tables, change wording from exact-looking "score/delta" to:

> rounded-report-derived `[macOS-CPU advisory]` score/delta; adequate for triage sign and coarse ordering only, not exact contest score authority.

Also avoid calling these values `canonical_score` in narrative surfaces unless a full-precision scorer artifact exists.

### MEDIUM - Percepta integration memo still contains stale wrong CLI commands

Evidence:

- `.omx/research/codex_findings_percepta_pact_integration_20260520T185519Z_codex.md:116-124` passes `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip` to `--archive-bin`.
- `tools/probe_op3v3_decoder_mutation_feasibility.py:210-216` and `tools/materialize_decoder_q_candidates.py:139-146` read `--archive-bin` as raw member bytes and call FEC6 decoder extraction on those bytes.
- The corrected helper now points to the raw member path via `src/tac/optimization/percepta_microprogram_plan.py:25-34` and `:300-324`.
- The later OP3 memo correctly documents this fix at `.omx/research/codex_findings_op3v3_decoder_q_waterbucket_percepta_boostnerv_20260520T195822Z_codex.md:90-99`.

Recommended correction:

Append a correction to the Percepta integration memo replacing the stale `--archive-bin .../archive.zip` command arguments with:

`--archive-bin experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/op3v3_decoder_q_inflate_controls_20260520_codex/baseline/data_dir/x`

### MEDIUM - Inflate postprocess negative is slightly overgeneralized

Evidence:

- `.omx/research/codex_findings_inflate_postprocess_surface_smoke_20260520T202752Z_codex.md:16-23` correctly labels the run raw-level advisory-only.
- Lines `:42-51` then say this falsifies the easiest postfilter ideas and "Do not add a generic source-only postfilter to PR110."
- Tested surface was only three raw-level transforms: odd-frame `-1` luma, odd-frame `+1` luma, and odd-frame `1/8` temporal blend. None was a stock `inflate.py` runtime candidate with charged parameters.

Recommended correction:

Weaken line `:50` to:

> Do not add the tested constant odd-frame luma nudges or naive temporal blend to PR110. Any other source-only postfilter needs an identity guard, raw advisory improvement, and then stock-inflate runtime custody before exact eval.

This preserves the negative without killing untested deblocking, sharpening, foveated, geometry-aware, or parameterized postprocess families.

### MEDIUM - Sparse residual memo should foreground oracle/proxy authority

Evidence:

- `.omx/research/codex_findings_sparse_residual_oracle_charged_smoke_20260520T204441Z_codex.md:37-43` says the target raw was decoded from `upstream/videos/0.mkv`.
- The result JSON authority explicitly blocks promotion due to `target_video_used_at_compress_time_for_selection`, `not_stock_inflate_runtime_custody`, and `correction_bytes_not_yet_consumed_by_live_inflate_py`.
- Lines `:71-80` mostly handle this correctly, but "visible, packed, and charged" can read stronger than the JSON authority allows. The charge is via proxy archive size, not a live stock-inflate consumer.

Recommended correction:

Add immediately before the interpretation:

> This is an oracle raw-output experiment: selection used the ground-truth decoded target, the rate charge is a proxy archive-size charge, and correction bytes are not yet consumed by live `inflate.py`. The negative applies only to raw-error top-k `k=256`, `max_abs_delta=1`, this target decode, and this byte scale.

The existing "do not widen this exact raw-error top-k path" is acceptable and is not a method kill.

### LOW - BoostNeRV correction is source-faithful; keep the local scaffold distinction explicit

Evidence:

- `.omx/research/codex_findings_boostnerv_research_integration_20260520T191425Z_codex.md:13-24`, `:57-63`, `:153-158`, and `:272-286` correctly distinguish Zhang et al. CVPR 2024 Boosting-NeRV from the unsupported local `Liu ECCV 2024` residual-chain anchor.
- Live sources confirm the title/authors/venue and conditional-decoder/TAT/SFT mechanism:
  - CVPR OpenAccess / arXiv `2402.18152`
  - official repo `Xinjie-Q/Boosting-NeRV` at HEAD `d59ca91e7bae284a8970db007e5b2c7f804b0b46`

Recommended correction:

No authority rewrite needed. Keep using "current local residual-head scaffold is Pact-only unless renamed or replaced" in all follow-up prompts. Do not let later memos abbreviate that scaffold as source-faithful Boosting-NeRV.

## Immediate Correction Queue

1. Mark `codex_findings_percepta_wasm_mechanics_20260520T185820Z_codex.md` as superseded by the official-repo provenance pass.
2. Amend `codex_findings_percepta_pact_integration_20260520T185519Z_codex.md` command block so `--archive-bin` points at raw member `baseline/data_dir/x`, not `archive.zip`.
3. Add a standard advisory-score precision note to OP3, inflate-postprocess, and sparse-residual memos: report-rounded, macOS CPU, non-promotional, not exact scorer precision.
4. Weaken postprocess and sparse-residual negatives as above; preserve "do not dispatch exact eval" for the exact measured candidates.

## Source Checks Used

- Percepta official pages: `https://www.percepta.ai/blog/can-llms-be-computers`, `https://www.percepta.ai/blog/constructing-llm-computer`
- Percepta official repo: `https://github.com/Percepta-Core/transformer-vm`
- Boosting-NeRV CVPR/OpenAccess and arXiv: `https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Boosting_Neural_Representations_for_Videos_with_a_Conditional_Decoder_CVPR_2024_paper.html`, `https://arxiv.org/abs/2402.18152`
- Boosting-NeRV official repo: `https://github.com/Xinjie-Q/Boosting-NeRV`
