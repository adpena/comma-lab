# ddm_scg1 — public-entrypoint smoke receipts are now a seal invariant

Outcome: the candidate-seal producer, validator, and fire path now fail closed unless the seal carries structured receipts for both receiver legs on both the candidate and frontier trees. The review tracker now re-parses a Python file before marking it, and the literal anchor-count assertion class was removed from the test tree. This was apparatus work only: `score_claim=false`; no scorer, Modal dispatch, or frontier mutation occurred.

# FORMALIZATION_PENDING:seal apparatus — no measured law; the gap is a contract, closed by refusal

## Contract closed

The new `candidate_public_entrypoint_smoke.v1` block contains `public_path_probes` and `inflate_sh_smokes`, each split into `candidate` and `frontier` receipts. Every receipt binds its outcome, wall time, exception class/message, runtime path and measured tree digest, archive path and measured archive SHA. The direct leg must reach token decode without exception. The shell leg must reach the exact nonzero CUDA-gate refusal. Both legs for a role must name the same tree and archive; the candidate identity must be the object being sealed; the frontier archive must match the seal's admit-bar pointer.

The fire tool checks this after the existing receiver/archive pin-consistency stage. A missing block receives a typed refusal. The only waiver is a substantive, explicit custody-replay reason for an archive SHA already present on a canonical scored pointer. A malformed or identity-drifted block is never waivable.

## Per-item receipt

| Item | File:line | Rule closed | Verification |
|---|---|---|---|
| 1 | `src/tac/candidate_seal.py:565`, `:817-965`, `:969-1032`, `:1115-1217`, `:1373-1392`; `tools/make_candidate_seal.py:90`, `:177-188` | `public_entrypoint_smoke` is required at construction and validation; receipt details and live identities are checked rather than trusting a boolean. | `test_a_freshly_sealed_candidate_validates`; `test_a_seal_without_public_entrypoint_smoke_refuses`; `test_rc1_magic_guard_shape_cannot_be_seal_valid`; `test_public_smoke_receipt_details_are_validated_not_just_the_outcome_boolean`; `test_the_producer_cli_seals_and_validates_its_own_output`. |
| 2 | `tools/fire_modal_auth_eval.py:571`, `:632`, `:714-718`, `:824-864`, `:869-899` | Public-entrypoint proof is additive beside pin consistency. Missing proof fails with the rule chain; only already-scored custody replay can use the named waiver. Existing lane maturity, nonpromotable-lane, no-seal, and pin gates remain intact. | `test_fire_dry_run_refuses_pre_fix_seal_with_public_rule_chain`; `test_missing_smoke_waiver_refuses_an_unscored_archive`; `test_missing_smoke_waiver_allows_only_scored_custody_replay`; the full existing candidate-seal consumer suite remained green. |
| 3 | `src/tac/subagent_contract.py:577-581`, `:683`, `:688-726`; `src/tac/preflight.py:93047`, `:93078-93079`, `:93169` | Every standard contract says exactly: “receiver identity means bash inflate.sh on the staged tree, not the library path”; the independent preflight census and exact-phrase gate prevent registry or composer self-waiver. | `test_standard_contract_default_includes_all_blocks`; all standard-contract variants; `test_contract_integrity_keeps_public_receiver_identity_composed`; direct strict integrity check returned no findings. |
| 4 | `tools/review_tracker.py:1062-1117`, `:1121-1150`; `src/tac/tests/test_review_tracker_scan_scope.py:128` | `mark-file` re-parses the current AST, atomically replaces that file's entity census, preserves surviving review metadata, and marks the current entities. Syntax/unreadable failures refuse. | `test_mark_file_rescans_and_marks_a_function_added_after_scan` proves a census change from 1 to 2 and both current functions reviewed. Two post-edit passes covered 2,295/2,295 current entities across this arm's 38 Python files on pass 1 and 2,295/2,295 on pass 2. |
| 5 | `src/tac/tests/test_resize_exploit_flip_fix_frontier.py:26-32` plus the 27 other anchor-bearing test files changed by the bounded sweep | Anchor tests assert identity/addressability or exact anchor sets where contractually meaningful, never a brittle literal cardinality. The named test requires a nonempty anchor-ID set and a residual-map entry for every anchor. | Denominator: 39 literal `len(...anchors...) == N` assertions found in `src/tac/tests/`; 39/39 converted; the same post-sweep query found 0. The 31 runnable changed test files passed 822/822 tests in 32.79 s. |

## Replay verdicts and prior-law adjudication

| Object | New-contract verdict | Evidence |
|---|---|---|
| rc1 magic-guard tree shape | `SEAL_PUBLIC_SMOKE_INVALID`; fire dry-run refuses before dispatch | The synthetic staged receiver reproduces the source magic-guard exception and cannot claim `REACHED_TOKEN_DECODE`. Two focused replay tests passed. |
| `SEAL_ddm_sj1_token_predistortion_joint_contest_cuda.json` | `SEAL_PUBLIC_SMOKE_MISSING` | The read-only seal has no `public_entrypoint_smoke` key. Its `falsifiers[6:8]` describe decode identity and public-entrypoint outcomes only as prose. |
| `SEAL_ddm_sj1_token_predistortion_pass3_contest_cuda.json` | `SEAL_PUBLIC_SMOKE_MISSING` | The read-only seal has no `public_entrypoint_smoke` key. Its `falsifiers[6:8]` likewise carry prose, not the required candidate/frontier receipt pair. |

The rc1 prior-law prediction held. Both sj1 revalidation predictions were falsified: 2 falsifiers out of the 3 predicted objects. The missing key in both is `public_entrypoint_smoke`. Their prose remains evidence of what those arms observed, but it cannot satisfy a machine contract that also requires the control identity, exception typing, time bound, and live digest remeasurement. Neither sealed file or tree was edited.

## VERIFIED-AT-SOURCE

- verified-at-source: `submissions/semantic_joint_ctxmix/runtime/f26_inflate.py:423-429` reads the archive and refuses non-`WANS1`/`SD1M`/`SM3R` semantic payloads at the F26 magic guard, before token decode.
- verified-at-source: `submissions/semantic_joint_ctxmix/inflate.py:42-60` is the public main path; it verifies input and then raises `RuntimeError` with the measured-CPU-budget CUDA-gate message when CUDA is unavailable.
- verified-at-source: `src/tac/candidate_seal.py:841-946` defines and checks the two receipt groups, two roles, outcomes, bound, exception details, tree digest, and archive SHA; `:948-965` closes candidate/frontier identity.
- verified-at-source: `tools/fire_modal_auth_eval.py:824-847` performs the existing staged receiver/archive pin-consistency check; the additive public-entrypoint stage begins at `:864`.

## RECALL EVIDENCE

I searched the full research/state corpus by content with `rg -n 'public_path_probes|inflate_sh_smokes|public_entrypoint_smoke|F26 requires WANS1|CUDA gate' .omx/research .omx/state /Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion`, inspected both sj1 seals with `jq`, queried the canonical equation catalog with `tools/list_canonical_equations.py --json`, and searched the canonical research index, `sub015_DAG_*` FEED material, specs, and task ledger for receiver/seal/public-entrypoint closure.

Beyond the charter seeds, the search found the rc1 `RESTAGE.json` machine vocabulary `public_path_probes` and `inflate_sh_smokes`, existing runtime-closure doctrine that treats receiver proof as prior to exact CPU/CUDA authority, and the decisive fact that both sj1 seals store their results only in prose falsifiers. This changed the plan from “revalidate the two old seals” to a typed `SEAL_PUBLIC_SMOKE_MISSING` refusal plus a narrowly queued successor-seal obligation. The catalog returned 476 registered equations and no measured law for this apparatus; this memo therefore carries the required formalization-pending marker rather than inventing an equation.

## Verification boundary

- 822/822 tests passed across every runnable changed test file. The focused core run passed 137/137, including all 50 candidate-seal cases.
- `ruff check` passed on the seal, fire, review-tracker, subagent-contract, preflight, and focused test surfaces. All changed Python files parsed as valid ASTs.
- `test_compute_dtype_seam.py` could not collect on this host because importing `mlx.nn` raises `RuntimeError: No Metal device available`; its only change is an anchor-assertion rewrite, and it passed static AST validation. This is an environment-scoped unrun, not a green test claim.
- The catalog JSON contained 476 equations; no seal apparatus law was claimed or added.
- No full decoder smoke was executed by this arm, no frames or large payloads were materialized, no scorer ran, no Modal job fired, no exact score was produced, and the frontier pointer did not move.
- No edits were made to `upstream/`, `submissions/semantic_joint_ctxmix/`, either sj1 seal, or any sealed `/Volumes/*/pact/ddm_*` tree. Unrelated shared-worktree changes were preserved and excluded from this arm's commit.

## NEXT_IF_RESUMED

### ITEM 7 — issue structured sj1 successor seals before replay

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN / the next sj1 custody-seal producer; consumer store: `.omx/state/canonical_task_status.jsonl`; fire trigger: before any future `tools/fire_modal_auth_eval.py` fire or replay of the sj1 joint/pass3 bytes, run both receiver legs against candidate and current-frontier staged trees, persist a complete `candidate_public_entrypoint_smoke.v1` block, and issue a successor seal. Do not edit or bless the historical seals.

## LIVE-HYPOTHESES

- A successor sj1 seal built from freshly captured candidate/frontier receipt pairs should validate without changing archive bytes, because the historical prose says both candidate receiver legs succeeded; this is plausible but untested because the control receipts and full structured identities were never recorded.
- The already-scored custody waiver should remain rarely needed: reproducible custody trees can usually capture the two bounded receipts and produce a successor seal before fire, while the waiver safely covers genuinely historical scored bytes whose old tree cannot be reconstructed.

## DEAD-ENDS

- Treating library-path decode identity as public receiver proof is closed: rc1 passed that route and still failed at the public entrypoint's F26 magic guard.
- Treating the sj1 falsifier prose as a structured receipt is closed: neither seal has `public_entrypoint_smoke`, neither names both candidate and frontier identities in both legs, and therefore both fail closed.
- Rendering a “first two pairs” public smoke is closed: contest `file_list` selects videos and only base `0` is accepted; the bounded smoke is the direct decode-reach leg plus the public preamble-to-CUDA-gate leg.
- Marking the review tracker's stale entity census is closed: `mark-file` now rescans the current AST before applying review status.
- Literal anchor-count assertions are closed in `src/tac/tests/`: the bounded sweep converted 39/39, and the post-sweep denominator is zero.

sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]
