# PR submission expanded codebase/history audit

- Issued: 2026-05-19T23:56:33Z
- Lane: `lane_pr_submission_codebase_history_expanded_audit_20260519`
- Scope: broader follow-up to `.omx/research/pr_submission_adversarial_audit_20260519T230000Z_codex.md`
- Authority: `score_claim=false`, `promotion_eligible=false`, `ready_for_submission=false`
- Verdict: `NOT_SAFE_TO_PR`

## What I broadened

This pass expanded beyond the three public-facing draft files into:

- current worktree and branch/remote state;
- last-two-weeks git history, especially PR101/FEC6/PacketIR/PR-submission commits;
- current subagent ownership and task-state ledgers;
- PR draft/package surfaces under `.omx/research/pr_submission_check_in_package_20260519/`;
- current `submission_dir` runtime, manifest, archive, README, and report;
- current GitHub PR/release state via `gh`;
- pre-submission compliance gate behavior against the current local packet.

No PR was created, no release was published, no dispatch was launched, and no score was promoted.

## Current source-of-truth state

`main` is the only branch in use, but local state is not a PR-ready source of truth:

- `git branch --show-current`: `main`
- `git status -sb`: `main...origin/main [ahead 1]` plus dirty files
- local `HEAD`: `a1d36ea97d0f366f174cefdc4ff92946599970ae`
- `origin/main`: `e0e7d239b1c330449d9b799a67ad727a8737e789`

The single local-ahead commit is:

- `a1d36ea97 packet_compiler: fix codex F1+F2 - separate parser/runtime authority + validate compiler manifest schema`

Dirty/untracked PR-relevant state at audit time:

- modified PacketIR/compiler artifacts and tests:
  - `.omx/research/pr101_fec6_frontier_packetir_matrix_20260519_codex.{json,md}`
  - `src/tac/packet_compiler/__init__.py`
  - `src/tac/packet_compiler/deterministic_compiler.py`
  - `src/tac/tests/test_deterministic_compiler.py`
- untracked PR/FEC6 artifacts:
  - `.omx/research/pr_submission_adversarial_audit_20260519T230000Z_codex.md`
  - `.omx/research/pr101_fec6_deterministic_compiler_identity_20260519_codex.md`
  - `.omx/research/pr101_fec6_packetir_candidate_queue_20260519_codex.md`
  - `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/deterministic_packet_compiler_manifest.json`
  - `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/packetir_candidate_queue.json`
  - `tools/prove_pr101_fec6_deterministic_compiler_identity.py`

Implication: public PR text must not pin final source custody to this local tree until the relevant files are committed, pushed, and re-bound to a new immutable commit. Current public draft text still pins older commit `462f84cdd` in multiple places.

## Git history signal, 2026-05-05 through 2026-05-19

The last two weeks show a clear arc:

1. May 5-7: public frontier recovery, PR101/PR103/PR106 archive deconstruction, hidden-gem tooling, PR101 byte-floor and entropy experiments.
2. May 8-12: dual CPU/CUDA discipline, PR101 packet/runtime custody, packet compiler primitives, and PR101 clone/scorer-training scaffolds.
3. May 13-16: PacketIR exact-closure discipline, PR106/PR101 recode proofs, FEC6 K16 candidate creation, FEC6 CPU/CUDA result capture, and wrapper/parser profiling.
4. May 17-18: PR-body review work and broader frontier/state formalization; memory explicitly records FEC6 as a real CPU anchor but not submission-ready.
5. May 19: PR submission prep, upstream-template body, D5 prerequisite pass, PR95/Quantizr tone study, Slot Q/R recursive PR-body/README revisions, PacketIR authority matrix, deterministic compiler identity, and this Codex adversarial audit wave.

Relevant PR-draft/runtime commits in order:

- `595475a1c pr_submission_prep: D1+D2+D3 land + codex APPROVE_WITH_REVISIONS`
- `bfa4b59b0 pr-submission-d5-prereq: 6-of-8 prerequisites executed per T3 symposium PROCEED_WITH_REVISIONS`
- `8bc07a926 pr_95_quantizr_study_citations: revise PR body + study + audit`
- `ed25c2ecd slot_q: integrate Rounds 1-12 into final PR body + new submission_dir/README`
- `922aeeae6 oss: add INNOVATION inline grep convention to submission_dir source per operator directive`
- `085dc6bf5 oss: anchor PR body + submission README at commit 922aeeae6 with specific file permalinks + INNOVATION grep convention + 60-second easy reproduction smoke per operator directive`
- `462f84cdd slot_r round_1: fix fabricated CUDA 0.230320 -> canonical 0.226210 + provenance`
- `0f48f20a1 slot_r round_2: bump permalinks + fix line 19 inconsistency + replace gitignored path`
- `c1813ba4d slot_r round_3: rate term precision + README K=8 misattribution correction`
- `8f9d059a0 slot_r round_4: fix README full-eval CLI + inflate.sh arg-type per canonical contract`
- `70eaa8769 slot_r round_5: fix archive.zip member misdescription across 5 surfaces; SAFETY-CAP OPERATOR-ESCALATE`
- `1eb5e8196 slot_r LANDING: 5-round recursive council cycle COMPLETE; OPERATOR-ESCALATE per safety-cap`
- `b0e8f3f59 pr_body+readme: HONEST innovation attribution per operator-rigor question`
- `4a8191882 packetir: add PR101 frontier authority matrix`
- `a618142e7 packetir: prove PR101 FEC6 identity authority`
- `a1d36ea97 packet_compiler: fix codex F1+F2 - separate parser/runtime authority + validate compiler manifest schema`

The history is not "just stale review"; it contains real PR packet work. The problem is that the public-facing draft and final custody surfaces have not converged to a submission-safe state.

## Current external PR/release state

`gh pr list --repo commaai/comma_video_compression_challenge --author adpena --state all --limit 20` shows only:

- PR #107 `apogee submission (0.2293)`, state `CLOSED`, head `apogee-pr98-hnerv-adapter`

`gh pr list --repo adpena/comma-lab --state all --limit 20` returns no PRs.

`gh release list --repo adpena/comma_video_compression_challenge --limit 20` shows recent CPU auth-eval release assets, but no current FEC6 submission archive release matching the draft placeholder.

Implication: the current FEC6 PR is still a local draft/package, not an existing GitHub PR. The hosted archive URL remains unresolved.

## Current PR draft still fails the previous audit

The narrow audit remains current. Grep over the three public-facing files still finds:

- `<HOSTED_URL_PLACEHOLDER>` in the PR body and README;
- author-chain errors in PR body and README:
  - PR95 still attributed to `@SajayR` in public text;
  - PR98 still attributed to `@AaronLeslie138`;
  - PR100 still attributed to `@EthanYangTW`;
  - PR101 still attributed to `@BradyMeighan` in README;
- public raw Modal call id in the PR body;
- `canonical paired Modal A100 auth_eval` wording attached to a Modal T4 CUDA eval;
- `Vast.ai T4` wording in README despite the verified CUDA artifact being Modal T4;
- `0.000622` "net" delta wording, which double-counts the already-charged rate term;
- `brotli q=11 outer` / PR101 inherited `FP11` wording inconsistent with the PacketIR proof;
- `archive_manifest.json` still claims `innovation_4_arithmetic_coded_latent_residuals`, which the runtime does not implement;
- README still says selector stream in `0.bin`, but archive member is `x`;
- README full-score command still attempts to run `/tmp/archive_dir/inflate.sh` after unzipping an archive that contains only member `x`.

The exact author truth re-verified this pass:

- PR95: `AaronLeslie138`
- PR98: `EthanYangTW`
- PR100: `BradyMeighan`
- PR101: `SajayR`
- PR102: `EthanYangTW`
- PR103: `rem2`
- PR108: `andrei-minca`

## Runtime and auth-eval custody split

Current committed `submission_dir` contains the split runtime:

- `src/codec.py`, 6,107 bytes
- `src/codec_sidecar.py`, 12,158 bytes
- runtime tree includes `inflate.py`, `inflate.sh`, `src/codec.py`, `src/codec_sidecar.py`, `src/frame_selector.py`, `src/model.py`

The paired CUDA auth-eval artifact records a different runtime tree:

- `src/codec.py`, 17,108 bytes
- no `src/codec_sidecar.py`
- auth-eval runtime tree SHA-256: `ca7f4f323d57a346739532c74c95af4f0a82fedf400a9c6f9e201eb5124f1e61`

The current local deterministic compiler manifest records:

- runtime tree SHA-256: `fb4ba11f998ec8c0137dffaa7f567db416e5c8790d155aa9365a0aa0b3580dbb`

The current pre-submission gate computes a local submission runtime SHA-256:

- `fd4b36b0114789ffd25c6169f529bca70b20da8f70e4ee1336dad9fd64971a09`

These are not the same proof surface. The split refactor may be byte-identical at local inflate output level, but submission/public custody cannot claim the auth-eval was run against the exact current local runtime tree until one of these is true:

1. re-run paired CPU/CUDA auth eval on the final split runtime and bind those artifacts; or
2. revert public submission runtime to the exact monolithic runtime used by the existing auth-eval; or
3. land an explicit accepted same-runtime equivalence proof that the compliance gate consumes as authority.

At present, the compliance gate correctly refuses this as submission-ready.

## Current compliance gate result

I ran the current strict gate against the current local packet with the new deterministic runtime-tree SHA:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --contest-final \
  --strict \
  --submission-dir experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir \
  --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip \
  --auth-eval-json experiments/results/modal_auth_eval_paired_20260519/cuda/contest_auth_eval.json \
  --contest-cpu-auth-eval-json experiments/results/modal_auth_eval_paired_20260519/cpu/contest_auth_eval.json \
  --archive-manifest-json experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive_manifest.json \
  --submission-score-axis contest_cpu \
  --expected-archive-sha256 6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf \
  --expected-archive-size-bytes 178517 \
  --expected-runtime-tree-sha256 fb4ba11f998ec8c0137dffaa7f567db416e5c8790d155aa9365a0aa0b3580dbb \
  --expect-single-member x \
  --competitive-or-innovative-statement-file .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md \
  --public-scan-path .omx/research/pr_submission_check_in_package_20260519/PR_BODY_UPSTREAM_TEMPLATE_CONFORMANT.md
```

Result: exit code `1`, `passed=false`.

Passing structural checks include archive existence, archive SHA, archive size, single member `x`, ZIP local/central header parity, auth-eval JSON existence, archive identity in both CPU and CUDA auth-eval JSONs, and CPU frontier non-regression on the submitted axis.

Failing checks include:

- `auth_eval_runtime_tree_expected_match`: expected deterministic split-runtime SHA does not match CUDA auth-eval runtime tree.
- `submission_runtime_tree_matches_auth_eval`: current local runtime tree does not match auth-eval runtime tree.
- `contest_cpu_auth_eval_score_at_or_below_submission_threshold`: current hard threshold `0.192` is stricter than `0.1920513168811056`.
- archive manifest member table checks: member list/name/size/compress size/CRC/member SHA are absent from `archive_manifest.json`.
- `report_mentions_archive_sha256` and `report_mentions_archive_size_bytes`.
- `public_source_reproduce_command_or_sha_binding_present`.
- `public_evidence_contest_cuda_label_present`: draft uses `[contest-CUDA T4]`, not the literal gate token `[contest-CUDA]`.
- `contest_final_expected_lane_id_supplied`.
- `contest_final_expected_job_id_supplied`.
- `public_scan_has_no_private_surface`: raw Modal call id remains in public PR body line 44.

This is stronger than the earlier report because it uses the current split-runtime hash and still fails.

## PacketIR/compiler work helps future frontier work, not this PR draft

The current PacketIR/compiler artifacts are useful and non-promotional:

- deterministic compiler identity manifest:
  - archive SHA `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
  - archive bytes `178517`
  - no-op detector passed `True`
  - `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`
- PacketIR candidate queue:
  - 33 candidates, 29 operator candidates
  - materialized new archives `0`
  - all dispatch `False`
  - all score/promotion/dispatch flags false

This closes prior PacketIR matrix blockers, but it does not repair the current PR body, README, archive manifest, hosted URL, public hygiene, or runtime/auth-eval custody mismatch. Treat it as next-frontier infrastructure after the public PR packet is made honest, not as submission readiness evidence.

## Current highest-risk blockers

1. Public text still attributes PR authors incorrectly.
2. Public text still has unsupported feature attribution and codec-grammar claims.
3. Public text still leaks a raw Modal call id.
4. Hosted archive URL is still placeholder and no FEC6 release asset was found.
5. PR body/README pin old commit `462f84cdd`, while current source/runtime has moved past it.
6. Local source is ahead of origin and dirty; public permalinks cannot be final.
7. Current local runtime tree does not match existing CPU/CUDA auth-eval runtime-tree custody.
8. `archive_manifest.json` lacks the member table required by the strict submission gate and still includes a hallucinated arithmetic-coded residual innovation.
9. README reproduction command is still not runnable from `archive.zip` alone.
10. `pre_submission_compliance_check.py --contest-final --strict` still fails on the current packet.

## Required convergence path

Do not submit the current draft. To make it PR-ready:

1. Replace public PR body and README language with the corrected author chain and feature attribution from the narrow Codex audit Section F.
2. Remove raw Modal call ids and all unverified A100/Vast.ai wording from public text.
3. Replace `<HOSTED_URL_PLACEHOLDER>` with a real release asset URL and verify `curl -L` plus SHA-256.
4. Decide runtime custody strategy:
   - re-run paired CPU/CUDA auth eval on the split runtime, or
   - revert to the monolithic auth-eval runtime, or
   - wire an explicit accepted same-runtime equivalence proof into the compliance gate.
5. Fix `archive_manifest.json` with exact ZIP member table and delete the arithmetic-coded latent-residual claim.
6. Fix README commands so runtime tree and archive payload are staged separately.
7. Replace all `462f84cdd` permalinks with the final pushed commit containing the exact public source/runtime.
8. Re-run `pre_submission_compliance_check.py --contest-final --strict` with final expected lane/job id, final archive URL/text, final runtime tree SHA, and final public scan path.
9. Only after the gate passes, create the commaai PR.

## Bottom line

The codebase contains a legitimate FEC6 CPU-axis anchor and increasingly useful PacketIR/compiler authority, but the PR draft submission surface is not yet coherent with the codebase's own custody rules. The safe next step is not another council pass; it is a concrete public-packet convergence patch plus a fresh strict compliance run.
