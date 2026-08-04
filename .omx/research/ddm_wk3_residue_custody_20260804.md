# ddm_wk3 residue custody receipt

run_id: ddm_wk3
timestamp_utc: 2026-08-04T23:28:37Z
charter: .omx/tmp/codex_runs/wk3_prompt.md
common_contract: .omx/tmp/codex_runs/_common_contract.md

## Governing Scope

This receipt executes the wk3 residue charter under the common contract. It is
custody/disposition work only: no contest archive, no n600 scorer run, no public
frontier claim, and no protected-file edit.

Protected files named by the common contract were not edited:

- .omx/research/ddm_cr1_composition_row_827_20260801.md
- .omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md
- src/tac/optimization/direct_description_carrier_compose.py

## Recall Evidence

Consulted sources:

- PROGRAM.md
- CLAUDE.md
- AGENTS.md
- docs/operating_manual_craft_handoff.md
- .omx/state/main_hot_state.md
- .omx/tmp/codex_runs/wk3_prompt.md
- .omx/tmp/codex_runs/_common_contract.md
- .omx/research/ddm_wk2r_worktree_custody_20260803.md
- /Users/adpena/.codex/memories/MEMORY.md targeted lines for #914 and wk3
  transcript recovery

Repository and artifact checks:

- `git worktree list --porcelain` to enumerate registered residue worktrees.
- Per-worktree `git status --porcelain=v1`.
- `git diff --cached --name-status` for the #914 staged-index hazard.
- Targeted `rg` and `git log` checks for einstein_kolmogorov, v7, v8, v13,
  v15, v18b, and ratecrush.
- Exact-path cold-store tar archives with SHA-256 and `tar -tf` verification.

Beyond-charter facts found:

- Main already contains a later Einstein-Kolmogorov hardening commit,
  `57e4c4e52b einstein-kolmogorov: harden canonical producer identity (#876)
  [no-triality] [p0-ledger-ok]`.
- Main already contains ratecrush integration commit
  `11b3b9c7ad v10 RATE-CRUSH recovery: JXL lossless plane codec...`.
- The current index had no staged file at the direct checks in this run.

## #914 Index Diagnosis

At 2026-08-04T23:22:55Z and again before creating this receipt,
`git diff --cached --name-status` was empty. There was no live staged file to
diagnose, unstage, reset, or commit.

`src/tac/canonical_equations/__init__.py` is modified in the working tree, but it
was not staged in this run. If #914 recurs, the clearing action is to identify
the staged path with `git diff --cached --name-status` and have the owning lane
intentionally commit or unstage it. This wk3 run did not alter the index to clear
#914.

## Einstein-Kolmogorov Disposition

Disposition: LAND-ALREADY / CERTIFY-REAL.

The charter-priority module
`src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py` in the
einstein_kolmogorov residue worktree is byte-identical to current main:

- worktree SHA-256:
  `d1b213feb8b80fb6128e6120befa274ebff3c21185c238624997b0bebd3298e6`
- main SHA-256:
  `d1b213feb8b80fb6128e6120befa274ebff3c21185c238624997b0bebd3298e6`

The module is not a fake solver. It implements exact contest action arithmetic,
strict/inclusive byte gates, fixed-byte palette delta accounting,
research-only decision wrappers, and hash-bound validation/registration for
research-only measurements and charts. It labels the output as research-only and
promotion-ineligible; it does not claim contest authority or pointer movement.

Focused verification on main:

```
PYTHONPATH=src .venv/bin/python -m pytest \
  src/tac/canonical_equations/tests/test_einstein_kolmogorov_crux_20260719.py
```

Result: 29 passed in 14.52s.

Canonical registry boundary: querying `tools/list_canonical_equations.py --json`
for `einstein_kolmogorov_crux_action_rate_contract_v1` returned no visible
registry row in this run. Direct builder invocation did construct the equation:

```
einstein_kolmogorov_crux_action_rate_contract_v1
2
True False
```

Meaning: 2 anchors, `research_only=True`, `promotion_eligible=False`. This run
did not edit the registry.

The xi bridge JSON receipts in the residue worktree are execution-surface
blockers, not score claims: backend was not started because evidence-path
filesystem custody failed with permission denied under
`/Volumes/VertigoDataTier/pact/evidence`; pointer was unmoved.

## Cold Store

Cold-store root:
`/Volumes/VertigoDataTier/pact/cold_store/ddm_wk3_residue_20260804T232255Z`

All listed archives passed `tar -tf`. Originals were left in place; nothing was
deleted.

| residue | archive | bytes | sha256 | tar entries | disposition |
| --- | --- | ---: | --- | ---: | --- |
| einstein_kolmogorov run artifacts and xi receipts | `einstein_kolmogorov_residue.tar` | 24,014,336 | `de0d98a272a5f87a2b48ec97a4bc8d59ae1aa0794bd18d6e0c6e413125a22a8d` | 1,592 | COLD-STORE-CERTIFIED |
| ddm_v7 solved-plane tolerance waterfill artifacts | `ddm_v7_waterfill_residue.tar` | 902,937,088 | `348f1505ca091b6736ec5b4896db89af939ab833879405439102b61112d847a6` | 159 | COLD-STORE-CERTIFIED |
| ddm_v8 margin-gated correction artifacts | `ddm_v8_margin_gated_residue.tar` | 111,323,136 | `ae4a347b1c10dffbf0c1f1be77e7b1d573d3f3f1b6a6a00ab5846e768aebf6ee` | 112 | COLD-STORE-CERTIFIED |
| ddm_v13 worldsheet event predictor artifacts | `ddm_v13_worldsheet_residue.tar` | 4,854,272 | `cc1bf51854f603bda6ce0b86f8cdb8c9ad7aad66c870d5f0c06dde4e22ebb29f` | 56 | COLD-STORE-CERTIFIED |
| ddm_v15 scorer-solved template artifacts | `ddm_v15_scorer_templates_residue.tar` | 1,773,568 | `d01abd002b942bf3e8a4232b3c1eed0af5d05378bab52fc342990f4502046b94` | 144 | COLD-STORE-CERTIFIED |
| ddm_v18b common-master pricing artifacts | `ddm_v18b_common_master_pricing_residue.tar` | 599,040 | `6704f5e977fdc16cf54d87b02ba21c069edac53820e192b8195eacb4dc926d8a` | 111 | COLD-STORE-CERTIFIED |

## Disposition Table

| item | decision | evidence | boundary |
| --- | --- | --- | --- |
| Einstein-Kolmogorov module | LAND-ALREADY / CERTIFY-REAL | Module and focused test are byte-identical to main; focused pytest passed. | Registry query did not expose a row; no registry edit here. |
| Einstein-Kolmogorov xi/run residue | COLD-STORE-CERTIFIED | Archive SHA and tar-list verified. | Filesystem custody blocker only; backend not started; no score authority. |
| ddm_v7 solved-plane tolerance waterfill | COLD-STORE-CERTIFIED | Receipt verdict: `FORMULATION_LEVEL_EXACT_RESIDUAL_KOLMOGOROV_RATE_WALL`; archive verified. | n256 formulation over v6 on `[macOS-CPU frozen-scorer advisory]`; pointer unmoved. |
| ddm_v8 margin-gated correction | COLD-STORE-CERTIFIED | Receipt verdict: `FORMULATION_LEVEL_MARGIN_GATED_CORRECTION_RATE_WALL`; archive verified. | Finite tau ladder n256 over bound v6/v7 reference on `[macOS-CPU frozen-scorer advisory]`; formulation only. |
| ddm_v13 worldsheet predictor | COLD-STORE-CERTIFIED | Receipt verdict: `ADVISORY_V13_INSTANCE_FALSIFIER_TRIGGERED_FORMULATION_ONLY`; archive verified. | Instance only; natural worldsheet and periodic Lane production families remain open. |
| ddm_v15 scorer-solved templates | COLD-STORE-CERTIFIED | Receipt schema `ddm_v15_scorer_solved_template_receipt.v1`; archive verified. | No contest score claim, no promotion eligibility, pointer unmoved. |
| ddm_v18b common-master pricing | COLD-STORE-CERTIFIED | Checkpoint schema `ddm_v18b_common_master_pricing_checkpoint.v1`; archive verified. | Stage checkpoint, not a final contest claim. |
| ratecrush donor coder | LAND-ALREADY | Worktree file is byte-identical to main. | No action needed. |
| ratecrush stream ranking and JXL codec residue | HONESTLY-DROP-WITH-REASON | Main has newer/superseding ratecrush work in commit `11b3b9c7ad`; worktree copies are older subsets. | Do not overwrite newer main with stale residue; residue remains in its worktree. |

## Task #883

This harvest did not invoke or edit the serializer repair path. The only planned
commit from this run is this markdown receipt, with an explicit file list and
expected post-edit SHA-256 through the serializer. No silent non-empty-index
commit was performed.

## Verification

- Focused Einstein-Kolmogorov pytest: 29 passed in 14.52s.
- All cold-store archives passed `tar -tf`.
- `git diff --cached --name-status` was empty before this receipt was created.
- Direct Einstein-Kolmogorov builder call returned the expected research-only,
  promotion-ineligible equation shape.

## Measured / Not Measured

Measured or verified in this run:

- File identity for the charter-priority Einstein-Kolmogorov module.
- Focused test pass for that module.
- Cold-store archive bytes, SHA-256, and tar-list readability.
- Live staged-index absence for #914.

Not measured in this run:

- No `upstream/evaluate.py` exact contest run.
- No n600 scorer run.
- No CUDA/contest-CPU authority row.
- No new archive.zip score.
- No deletion or space-reclaim action.

## Boundary And Frontier Honesty

This run did not move the public or own-vehicle frontier. It preserved residue
custody and classified what was already landed, what was certified to cold store,
and what should be honestly dropped because main already supersedes it.

Own-vehicle frontier remains:

`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`

