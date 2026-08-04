# ddm_wk3 residue disposition, scorer-free

Date: 2026-08-04
Charter: `.omx/tmp/codex_runs/wk3_prompt.md`
Common contract: `.omx/tmp/codex_runs/_common_contract.md`

## Boundaries

- Scorer ownership: none. `fz4` owned the n600 scorer slot; this arm ran no scorer job.
- Index boundary: did not unstage, reset, stash, or commit any pre-existing staged entry.
- Cleanup boundary: copied residue to SSD cold store only. Originals remain in their source worktrees.
- Score boundary: no archive was evaluated and no frontier pointer moved.

Storage preflight before cold-store copy:

| tier | available |
|---|---:|
| `/Users/adpena/Projects/pact` | 493 GiB |
| `/Volumes/VertigoDataTier/pact` | 194 GiB |

## Cold-store Manifest

Machine-readable custody root:
`/Volumes/VertigoDataTier/pact/ddm_wk3_residue_rescue_20260804T163159Z`

| file | sha256 |
|---|---|
| `manifest.json` | `855ef43ecfc50415954bfc3d5548bf7f909fec504966e0f1508576a554a06d04` |
| `files.jsonl` | `ea3354b8b09b0266aeea462cd82183a3935afc5f1de4acfdf3f0b9a618966b80` |

`manifest.json` records source worktree, branch, HEAD, porcelain entries, file counts, source payload bytes, archive bytes, and archive SHA-256 per group. `files.jsonl` records per-file path, status source, byte count, kind, and SHA-256. The cold-store command was a Python tarfile copy over explicit `git status --porcelain=v1 -z` entries; no source file was deleted or moved.

Current verification rerun:

- `shasum -a 256` matched `manifest.json`, `files.jsonl`, and all seven cold-store tar archives above.
- `files.jsonl` has 2,079 per-file custody rows.
- Einstein/Kolmogorov targeted residue tests were rerun with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider src/tac/tests/test_check_154_manifestless_cleanup_identity_scope_extension.py src/tac/tests/test_check_344_anchor_roundtrip_scope_extension.py src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py -q`: 114 passed, 7 failed. The failures are in Catalog #351 producer-identity scope-extension coverage, so the uncommitted successor patch remains not landable as-is.
- Main and `codexwt/ddm_de1_20260803T112347Z` cached diffs were checked directly and were empty.

## Disposition Table

| residue group | source scope | disposition | evidence | follow-on fire order |
|---|---|---|---|---|
| Einstein/Kolmogorov canonical-equations residue | `codexwt/einstein_kolmogorov_crux_20260719T212159Z`, 17 status entries, 1,572 files | COLD-STORE-CERTIFIED; not landed | Archive `einstein_kolmogorov.tar`, 28,825,600 B, sha256 `9d0a6b24c9200c7c80865ca40bc20a75d0c33a1d24eac66b59d72a896852e586`. `py_compile` passed for the changed module/preflight/registry files, but the residue's targeted suites reported 114 passed and 7 failed. Failures are all in Catalog #351 producer-identity coverage; the docs' appended "107 dedicated tests passed" claim is stale for this working tree. Main already contains HEAD `6a78ee8209`; the uncommitted successor patch is the residue at risk, not an already clean landing. | QUEUED-WITH-A-FIRE-ORDER: repair or back out the uncommitted Catalog #351 scope-extension patch, rerun the three targeted suites, then start a fresh 3-clean-pass review before any main landing. Do not cite the uncommitted verification prose as measured. |
| v7 solved-plane tolerance waterfill | `codexwt/ddm_v7_solved_plane_tolerance_waterfill_20260722T083215Z`, 25 status entries, candidate receipts and rung checkpoints | COLD-STORE-CERTIFIED | Archive `v7_waterfill.tar`, 902,103,040 B, sha256 `20c172a870cd4065422a2b061cb37abad4e21bf4522a4c9b643d6959057830a6`. Payload contains 137 files / 901,854,112 source bytes. | QUEUED-WITH-A-FIRE-ORDER: restore from the archive only if a current owner needs the receipts; fold into the relevant waterfill ledger with explicit `not_a_candidate` and scorer-free labels before any launch. |
| v8 margin-gated correction | `codexwt/ddm_v8_margin_gated_correction_20260722T104341Z`, 20 status entries, tau checkpoints and launch manifests | COLD-STORE-CERTIFIED | Archive `v8_margin_gated.tar`, 110,520,320 B, sha256 `a820388bcf866d48fd2dd4c1643e9d4f8a966c2afca43586f9159ac1c3da478a`. Payload contains 88 files / 110,359,343 source bytes. | QUEUED-WITH-A-FIRE-ORDER: restore only for a scoped tau-checkpoint review; label any negative as artifact-instance scope until rerun on a current receiver. |
| v13 worldsheet REDERIVE dirs | `codexwt/ddm_v13_worldsheet_event_predictor_20260722T181444Z`, 5 artifact dirs | COLD-STORE-CERTIFIED | Archive `v13_worldsheet_rederive.tar`, 5,294,080 B, sha256 `b20e47507f97d518e0e5855c4f4a01fb0c9d647bd3151b24c1064f3f889ff110`. Payload contains 55 files / 5,184,858 source bytes. | QUEUED-WITH-A-FIRE-ORDER: fold only through a REDERIVE artifact ledger; no scorer or frontier claim without current byte-closed receiver proof. |
| v15 scorer-template REDERIVE dirs | `codexwt/ddm_v15_grammar_parametrized_scorer_solve_20260722T230509Z`, 6 artifact dirs | COLD-STORE-CERTIFIED | Archive `v15_scorer_templates_rederive.tar`, 1,607,680 B, sha256 `026e81201e24f86a8f239fee2f363ae3ba766fa051f1c13e765b7ec108945bf1`. Payload contains 126 files / 1,381,469 source bytes. | QUEUED-WITH-A-FIRE-ORDER: restore only under the v15 REDERIVE owner; keep as artifact evidence until a current receiver validates it. |
| v18b common-master REDERIVE dirs | `codexwt/ddm_v18b_common_master_rebaseline_and_pricing_20260723T033945Z`, 3 artifact dirs | COLD-STORE-CERTIFIED | Archive `v18b_common_master_rederive.tar`, 471,040 B, sha256 `81a62c4e28b018f33a5be4dfc7249a3eed9c051440a3dcf207f01949e911f19c`. Payload contains 98 files / 274,618 source bytes. | QUEUED-WITH-A-FIRE-ORDER: restore only for a current pricing/rebaseline audit; do not promote as live pricing without rerun. |
| ratecrush code files | `.claude/worktrees/agent-a1f080a4426f446f9`, three untracked code files | COLD-STORE-CERTIFIED; not landed | Archive `ratecrush.tar`, 40,960 B, sha256 `8feec3f68c6a5302134229a6bd693cba15e596e66755c0b4b9a9614377e7cbfd`. Files: `experiments/v10_ratecrush_rank_donor_coders.py`, `experiments/v10_ratecrush_rank_streams.py`, `src/tac/codec/v10_jxl_plane_codec.py`. | QUEUED-WITH-A-FIRE-ORDER: code-review and test the three files before landing; archive custody alone is not implementation truth. |

## Task #914 Staged-Index Hazard

Current diagnosis did not reproduce a live staged-index hazard in the checked scopes:

- Main worktree: `git diff --cached --name-status` returned empty.
- `codexwt/ddm_de1_20260803T112347Z`: `git status --porcelain=v1` and `git diff --cached --name-status` returned empty.

I did not unstage, reset, or commit any de1 index entry. If the hazard reappears, the clearing action is owner-only: either serializer-commit the staged file after provenance/review, or unstage/reset it only with explicit operator approval.

## Task #883 Serializer Boundary

This rescue did not edit the serializer repair path. Before landing this memo update, the main cached diff was empty. The landing target is this single memo file only, with an expected post-edit SHA; after commit, verify the cached diff is empty. That keeps the prior non-empty-index silent-commit class out of this landing.

## Final Frontier Line

Own-vehicle frontier unchanged by this scorer-free rescue: `S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`; contest pointer `0.1910828242` remains borrowed and unmoved.
