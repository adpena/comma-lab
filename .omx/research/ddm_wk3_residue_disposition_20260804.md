# ddm_wk3 residue disposition (#876 rescue)

Date: 2026-08-04
Charter: `.omx/tmp/codex_runs/wk3_prompt.md`
Common contract: `.omx/tmp/codex_runs/_common_contract.md`

This was a scorer-free/no-n600 rescue pass. I did not run a full evaluator, did
not claim a new score row, did not touch `upstream/`, and did not edit the
three forbidden files named in the common contract. Bulk residue was certified
to the SSD cold-store tier before any drop decision; the original residue paths
were left in place.

Current own-vehicle frontier at time of this pass, from the common contract and
live hot-state section: `S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`
(commit `4dcfd24870`, archive sha prefix `ad5dd0e4`). Contest pointer
`0.1910828242 [contest-CPU]` remains borrowed and unchanged.

## Disposition Table

| residue | disposition | evidence | cold-store certification | boundary / next action |
|---|---|---|---|---|
| `einstein_kolmogorov` canonical producer identity hardening | LAND | Committed narrow, real hardening in `57e4c4e52b`: path canonicalization now rejects absolute, empty, parent-traversal, symlink-out, non-regular, alias, and multi-hardlink producer references; focused tests pass. | Unlanded residue additionally archived as `einstein_kolmogorov_unlanded_residue.tar`; bytes `27,230,208`; sha256 `f3721d295d3d4ceeaafbc4e3b668edc3cd689c39749530492ce8d3ce4144a30f`; entries `1604`; first entry `.omx/research/einstein_kolmogorov_crux_20260719T214408Z_codex.md`. | Broad residue was not landed: its #351/preflight repair path failed its own focused test run (`7 failed, 30 passed`) against stale mutation text and live repo defects. |
| `ratecrush` JXL/rate code residue | COLD-STORE-CERTIFIED | Three untracked code files implement real local JXL roundtrip/ranking surfaces with fail-closed sha checks; local-only/non-promotable axis and external `cjxl`/`djxl` runtime closure were not sealed here. | `ratecrush_code_residue.tar`; bytes `31,232`; sha256 `f7ebf99ad46d8ade2c57b9019049bd866841006b880094a59d80f31d95236bc0`; entries `3`; first entry `experiments/v10_ratecrush_rank_donor_coders.py`. | Do not promote as a score lever without runtime closure, focused tests, and byte-closed candidate evidence. |
| `v7_waterfill` candidate/receipt residue | COLD-STORE-CERTIFIED | Receipt labels schema `direct_description_solved_plane_tolerance_waterfill.v1`, lane `ddm_v7_solved_plane_tolerance_waterfill`, verdict `FORMULATION_LEVEL_EXACT_RESIDUAL_KOLMOGOROV_RATE_WALL`, `score_claim=false`, `not_a_candidate=true`, `[macOS-CPU frozen-scorer advisory]`. | `v7_waterfill_residue.tar`; bytes `902,645,760`; sha256 `63838603a8f67732e64c940c15cb454fbc3034d69c8ecec0124f40941365bc33`; entries `166`; first entry `.omx/research/ddm_v7_solved_plane_tolerance_waterfill_603_613_20260722T102423Z.md`. | Formulation-scoped advisory residue only; not a contest row and not promoted. |
| `v8_margin_gated` tau/checkpoint residue | COLD-STORE-CERTIFIED | Receipt labels schema `direct_description_margin_gated_correction.v1`, lane `ddm_v8_margin_gated_correction`, verdict `FORMULATION_LEVEL_MARGIN_GATED_CORRECTION_RATE_WALL`, `score_claim=false`, `not_a_candidate=true`, `[macOS-CPU frozen-scorer advisory]`. | `v8_margin_gated_residue.tar`; bytes `110,622,720`; sha256 `82c8367b23c61c9dde5c8f9ce6b323f1b2258b1fca5f67d2ca1d82d93514c7ca`; entries `103`; first entry `.omx/research/ddm_v8_margin_gated_correction_603_613_20260722T115052Z.md`. | Formulation-scoped advisory residue only; not a contest row and not promoted. |
| `v13_worldsheet` rederive/artifact residue | COLD-STORE-CERTIFIED | DAG/feed says `research_only=true`, `execution_allowed=false`, `score_claim=false`; review verdict is `ADVISORY_V13_INSTANCE_FALSIFIER_TRIGGERED_FORMULATION_ONLY`; gate stops because total d_seg exceeds the gate. | `v13_worldsheet_residue.tar`; bytes `2,196,480`; sha256 `94eb66e152a2261c89796a36586d3a306f0bbd4498e794909ff7be9924ae2ac2`; entries `56`; first entry `.omx/research/ddm_v13_g1_worldsheet_predictor_n600_20260722T201500Z/`. | Instance/formulation-only advisory falsifier; no land/promotion from this pass. |
| `v15_scorer_templates` rederive/artifact residue | COLD-STORE-CERTIFIED | DAG/feed says `research_only=true`, `score_claim=false`; downstream blocker `FORMULATION_SCOPED_TEMPLATE_SOLVE_GATE_NOT_MET_JOINT_TRAINING_366_REMAINS_OPEN`; invalidation receipt identifies superseded/invalid older receipts. | `v15_scorer_templates_residue.tar`; bytes `603,648`; sha256 `85f08080fe7dfd0fcbd29f76a542bc55419b9430ea5f74885664ee24c7cc5aeb`; entries `63`; first entry `.omx/research/ddm_v15_grammar_parametrized_scorer_solve_DAG_FEED_20260723.md`. | Gate not met; no land/promotion from this pass. |
| `v18b_common_master` pricing/artifact residue | COLD-STORE-CERTIFIED | Logs include an immutable-output-diff failure for `00_common_master_source_closure.json`; checkpoint reports `negative_reduced_cost_count: 0`, `selected_global_mode_admitted: false`, `score_claim: false`, `[macOS-CPU frozen-scorer advisory]`. | `v18b_common_master_residue.tar`; bytes `3,549,184`; sha256 `7c618201345dc327af050b328317aecef25292445bcfa7dfd417a265a26fcb5b`; entries `644`; first entry `.omx/research/ddm_v18b_common_master_pricing_20260723T050800Z/`. | Advisory/pricing residue with no admitted global mode; no land/promotion from this pass. |
| Task #914 `ddm_de1` staged-index hazard | DIAGNOSED-NOT-LIVE-IN-SCOPED-INDEXES | `git diff --cached --name-status` in `main` and in `.omx/tmp/codex_worktrees/ddm_de1_20260803T112347Z` both returned empty; `ddm_de1` status was `## codexwt/ddm_de1_20260803T112347Z`. I did not unstage, commit, reset, or stash any staged content. | None needed; no staged file found in the scoped live indexes. | If the stale blocker recurs elsewhere, owner should first identify the staged path with `git diff --cached --name-status` in that exact worktree, then either serializer-commit owned content or have the owner unstage it. This pass did neither. |
| Task #883 serializer repair-path class | CHECKED-NOT-LIVE-FOR-WK3-LANDING | This pass did not edit `tools/subagent_commit_serializer.py` or a serializer repair path. The wk3 serializer commit for `57e4c4e52b` used a temp index, declared post-edit shas, and left the shared cached index empty. | None needed. | The historical class was not live for this landing; no serializer repair action taken. |

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_einstein_kolmogorov_canonical_producer_reference.py tests/test_einstein_kolmogorov_crux.py -q` -> `47 passed in 32.86s`.
- In the residue worktree, `.venv/bin/python -m pytest src/tac/tests/test_check_351_canonical_producer_identity_scope_extension.py -q` -> `7 failed, 30 passed`; therefore that broad repair was not landed.
- `tools/review_tracker.py mark-file` was run twice on the two changed Python files (`wk3-pass1`, `wk3-pass2`) before commit.
- `git diff --check -- src/tac/canonical_equations/einstein_kolmogorov_crux_20260719.py src/tac/tests/test_einstein_kolmogorov_canonical_producer_reference.py` was clean.
- `git diff --cached --name-status` was empty before and after the scoped index checks recorded above.
- Cold-store command family used `COPYFILE_DISABLE=1 tar --no-xattrs -cf` under `/Volumes/VertigoDataTier/pact/cold_store/ddm_wk3_residue_20260804`, followed by `ls -l`, `shasum -a 256`, and `tar -tf ... | wc -l`.

## Frontier

wk3 moved no score row. Own-vehicle frontier unchanged by this pass:
`S = 0.7541459 @ 358,084 B [macOS-CPU advisory] n600`.
