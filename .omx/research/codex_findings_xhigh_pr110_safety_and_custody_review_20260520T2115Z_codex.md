# Codex Findings: XHigh PR110 Safety And Custody Review

UTC: 2026-05-20T21:15Z
Reviewer: Codex xhigh adversarial review worker
Scope: PR110 safety/custody surface and contamination risk from current local PR110 experimentation.
Write scope honored: this memo only.

## Preflight

- Read `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md` enough to honor the relevant non-negotiables: axis separation, exact archive/runtime custody, no proxy promotion, no destructive ops, no live PR mutation, and partner-work preservation.
- Checked `git status --short --branch`: on `main`, ahead of origin, heavily dirty with many unrelated active partner files. The only reviewed dirty PR110-adjacent tracked files are:
  - `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`
  - `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json`
- Checked `.omx/state/subagent_progress.jsonl` tail. Multiple wave-3 agents are actively editing unrelated Pact-NeRV / council / wire-in files. I did not run lane registration because the operator gave an exact one-memo write scope.
- Read local PR110 memos from 2026-05-20 and rechecked live PR #110 and the release asset via read-only `gh` commands.

## Verdict

No live PR110 archive/runtime/body blocker found in this pass. The live PR remains open, mergeable-clean, and pinned at head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`; the release asset is still `archive.zip`, `178517` bytes, digest `sha256:6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.

The real risk is local contamination: the provisional PR110 experiment tree is huge, ignored, and contains many generic `archive.zip` files plus charge-proxy archives. The current tracked local README/manifest mirrors are also not fully synchronized with the live PR/release wording.

## Findings

### High - Provisional PR110 tree can be mistaken for submission custody

Paths:

- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/archive.zip`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_search/*/archive_charge_proxy.zip`
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/contest_oracle_batch/**/archive/archive.zip`

Evidence:

- The tree is ignored by `.gitignore:55` and `git status --ignored` reports it as `!!`.
- It is about `10G`, with `2308` files.
- It contains `182` `.zip` files, including `180` files literally named `archive.zip` and `2` files named `archive_charge_proxy.zip`.
- It still contains three `0.raw` files of `3662409600` bytes each.
- The charge-proxy manifests correctly say `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, and `is_valid_submission_archive_claim=false`, but the filenames are still dangerous during fast handoff.

Risk:

- A human or agent can accidentally grab an experimental `archive.zip` or `archive_charge_proxy.zip` from the PR110-provisional tree and treat it as the PR110 release archive.
- The risk is amplified because `runtime_hfv1/archive.zip` has the real PR110 SHA, while many sibling candidate archives have different SHAs and similar names.

Recommended fix:

- Keep this tree out of live PR/release paths.
- Before any handoff or cleanup sweep, reduce it to small JSON/MD/profiler artifacts plus hashes. Delete or externalize rebuildable `*.raw`, `source/`, and candidate archive directories after their summaries are preserved.
- If charge-proxy archives must remain, rename future outputs to include `not_submission` or add a local sentinel such as `DO_NOT_SUBMIT.md` at the tree root and in charge-proxy directories.

### Medium - Dirty local README still carries public-surface wording hazards

Path:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/README.md`

Evidence:

- Line 7 still says the CPU host `matches upstream ubuntu-latest GHA runner family`; line 88 repeats `matching the GHA runner family`. The live PR body correctly downgraded this to factual `num_threads: 2`.
- Line 74 installs `torch brotli` but omits `numpy`, even though the runtime import closure and release-body smoke require `torch + numpy + brotli`.
- Line 117 still says no other Python packages or shared libraries are loaded. The Python import closure is supportable, but native shared libraries may be loaded transitively by `torch`, `numpy`, and `brotli`.
- Lines 52 and 125 still use "lives alongside `archive.zip` in the submission directory" wording. That is locally true for this mirror, but misleading next to line 43's statement that live PR #110 stages runtime in the PR tree and hosts `archive.zip` as a release asset.

Risk:

- If this README is copied into public PR/release text or used as the next operator-facing source of truth, it reintroduces issues already fixed in the live PR body.

Recommended fix:

- Keep this README local-only until synchronized.
- Replace GHA-runner inference with factual Modal/Linux/evaluator wording.
- Add `numpy` to the smoke install command.
- Replace the shared-library overclaim with: Python import closure is stdlib plus `torch`, `numpy`, `brotli`; native shared libraries may be loaded transitively by those wheels.
- Clarify that live PR runtime and release archive are staged separately, while this local `submission_dir` is a mirror.

### Medium - Local archive manifest has stale/public-claim ambiguity

Path:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json`

Evidence:

- Lines 8-9 still say `"score_claim": false` and `"promotion_eligible": false`.
- Lines 150-154 simultaneously record public `[contest-CPU]` and `[contest-CUDA]` observations for the submitted PR110 packet.
- Lines 159-161 retain a v2 revision-log entry saying the manifest added "submission readiness false" and a "source-sync commit blocker", even though line 10 now says `"ready_for_submission": true` and line 11 says PR #110 is live.

Risk:

- The manifest now mixes two meanings of "claim": internal exact-dispatch queue authority versus public PR score observations. That is fine if explicitly scoped, but unsafe as written.

Recommended fix:

- Rename/scope the old booleans, for example `internal_queue_score_claim=false`, or add `public_pr_score_claim=true` plus `score_claim_axes=["contest-CPU","contest-CUDA T4"]`.
- Add a v3 revision-log entry stating the D-5/source-sync blocker is cleared by live PR #110 head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.

### Medium - Final evidence pack is a snapshot, not current local mirror custody

Path:

- `.omx/research/pr110_final_evidence_pack_20260520T141144Z_codex/`

Evidence:

- The pack's `archive_metadata.json`, `archive_sha256.txt`, and release comparison are internally consistent for archive SHA `6bae0201...` and size `178517`.
- The pack's `runtime_layout_verdict.json` correctly says no `archive.zip` is embedded in the PR head tree and runtime files live under `submissions/hnerv_fec6_fixed_huffman_k16/`.
- However, `local_readme_report_manifest_hashes.tsv` records old hashes for the local README and archive manifest. Current dirty-file hashes are now different:
  - README current SHA: `06e7c0e1ad267a36388e22f25cc702658ad717c5dd5c5cbe6a1b2abe5ca054b0`
  - archive manifest current SHA: `e195dcd26e0c8c7c7c01138f58dc812fe9eab381a0548211cd5e047389750986`
- The evidence pack also stores hashes for `report.txt` and manifest-like local files rather than copying all of those files into the pack.

Risk:

- If the final evidence pack is described as current after the dirty README/manifest edits, its local README/manifest custody claims are stale.

Recommended fix:

- Treat the pack as a valid `2026-05-20T14:11:44Z` snapshot only.
- After the local README/manifest mirror is finalized, either regenerate a new dated evidence pack or append a small dated delta memo with current hashes and explicit "live PR unchanged" status.

### Low - Local hosted-release metadata is stale relative to live release title

Path:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.hosted_archive_manifest.json:7`

Evidence:

- Local line 7 says `FEC6 frontier submission - 0.192051 [contest-CPU] / 0.226210 [contest-CUDA T4]`.
- Live release now says `FEC6 selector submission - 0.192051 [contest-CPU] / 0.226210 [contest-CUDA T4]`.

Risk:

- Low because this file is not tracked by `git ls-files` and the live release body/title are fixed. It is still stale if used as an audit input.

Recommended fix:

- Refresh or mark this local hosted-archive manifest as historical. If regenerated, keep the asset URL, digest, and byte count unchanged.

### Low - Competitive statement overbroadly says "current leaderboard frontier"

Path:

- `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/pre_submission_compliance.competitive_or_innovative_statement.txt:1`

Evidence:

- It opens with "current leaderboard frontier" while the supporting evidence is specifically the accepted/top-CPU PR #101 baseline and PR #108 late-submission gate.

Risk:

- Low because the tracked README and live PR body now use narrower CPU-axis wording. This file can still confuse future compliance reruns.

Recommended fix:

- Replace "current leaderboard frontier" with "accepted/top-CPU PR #101 baseline under the PR #108 late-submission gate."

## Live PR Body And Release Body Risks

No blocker found after read-only live recheck.

- Live PR #110 body has the corrected release-hosted archive wording.
- Live PR #110 body keeps CPU and CUDA axes separate.
- Live PR #110 body no longer contains the stale "cluster within", `ubuntu-latest`, `single-thread`, Claude/Codex/Anthropic/persona, or "companion PR" strings.
- Live release asset is unchanged: `archive.zip`, `178517` bytes, digest `sha256:6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`.
- Live release title/body use "selector submission", not the earlier generic "frontier submission" title.

The only live-body wording I would continue to watch is the broad "local experiments on this archive and related HNeRV variants" sentence. It is acceptable as written because it is qualitative and does not promote proxy artifacts, but it should not be expanded into a hard saturation claim without exact archive-byte evidence.

## Freeze / Allow List

Freeze:

- Live PR #110 branch/head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`.
- Live release asset `archive.zip` SHA/bytes.
- Live PR body and release body unless a factual blocker is found and routed through a new freeze-break memo.
- The exact PR110 public runtime files under `submissions/hnerv_fec6_fixed_huffman_k16/`.

Allowed:

- Local advisory/provisional experiments under ignored `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/`, as long as outputs remain labeled `score_claim=false`, `promotion_eligible=false`, and not exact-dispatch-ready.
- New dated `.omx/research/` memos.
- Local README/manifest mirror cleanup in the Pact repo, but not live PR branch edits, release replacement, or public PR text edits.
- Deleting or externalizing rebuildable raw/proxy outputs after preserving hashes and summaries.

## Bottom Line

Live PR110 custody is currently coherent. Local custody is not dangerous because the risky tree is ignored and advisory labels are mostly correct, but the naming and size of the provisional outputs make accidental contamination plausible during fast operations. The highest-value next safety action is not another live PR edit; it is a local cleanup/labeling pass that quarantines provisional archives and finishes the tracked README/manifest mirror synchronization.
