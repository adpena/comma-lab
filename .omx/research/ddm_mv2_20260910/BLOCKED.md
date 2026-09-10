# ddm_mv2 landing blocked before mutation

MEASURED [macOS-CPU scorer-free patch preflight]: HEAD `6515f94e3caa17da9b9b74492c648c05baef843f`. The pinned patch SHA is correct and its stat lists exactly 21 files. The rp1 GO file is absent. No patch was applied, catalog number claimed, commit attempted, or queue row marked landed. Tests, ruff, import, gate run, and review marking were not run because the charter stops this landing before application. These receipts are uncommitted.

The charter's Boundaries says: "If the patch's code no longer matches HEAD beyond those three hunks, STOP and report the exact hunks; do not improvise a rewrite of the arm's mechanism."

Additional code conflicts, from `git apply --check --verbose` (exit 1):

- `src/tac/candidate_seal.py:1413`: the retained-payload hunk expects the public-smoke block immediately before section 6. Current HEAD inserts decode-wall-clock validation there.
- `experiments/ddm_vr3_certified_raw_reclaim.py:19`: the import hunk expects `SCHEMA` immediately after typing imports. Current HEAD inserts the vr7 path setup and reproducer-descriptor import. Further vr3 changes also exist; Git reports the first failed hunk, so this is not an exhaustive adjudication of later hunks.

Ten added memo/evidence files already exist and are byte-identical to their patch payloads, including the charter's expected memo conflict. They were not overwritten. Detailed hashes and the exact failed context are retained in `BLOCKED.json` and `apply_check.json` beside this receipt. Mechanism deviations: none; implementation files untouched. Source payloads, protected trees, SSD artifacts and staged index were not modified.

The requested `final_serializer.stderr` was not found in the mv1 receipt directory's file inventory. `FINAL_HANDOFF.json` records the prior refusal as Git object creation denied, Operation not permitted; `serializer_status.json` corroborates return code 17. No new Git-write claim is made.

## RECALL EVIDENCE

Own recall searched `.omx/research/` content for `MOVED.json|moved.manifest|landing_an_arm_bundle|landing_a_gate_into`, then canonical research index, DAG, SPEC and task-ledger surfaces for `MOVED.json|moved.manifest|ddm_mv1|ddm_vr7`. Beyond the charter seeds, vr7's landed reproducer-descriptor work and the live hot board's landed decode timing work explain the two changed code contexts; Git's base-to-HEAD diff confirms both. This changes the disposition to the charter's mandatory STOP, rather than an improvised rebase. The equation registry command returned 483 entries; no matches for `MOVED.json|artifact_moved|ddm_mv1|ddm_vr7`. Query and result receipts are in `recall.json`. The memory registry's custody guidance reinforced retaining an honest unlanded receipt; no historical Git denial is treated as a current attempted failure.

## NEXT_IF_RESUMED

- QUEUED-WITH-A-FIRE-ORDER; owner MAIN; consumer store `.omx/research/ddm_mv2_20260910/BLOCKED.json`; action: issue a revised charter or refreshed patch covering the candidate-seal and vr3 conflicts while preserving intervening changes; fire trigger: MAIN harvests this blocker. Recheck rp1 GO absence before any authorized landing.

LIVE-HYPOTHESES: A refreshed landing may preserve mv1's mechanism while composing with the timing and vr7 changes; the first reported conflicts are insertion-context overlaps, but the combined behavior remains untested.

DEAD-ENDS: Applying the original patch under the three-conflict exception is closed for this HEAD: two additional code files fail. Treating existing evidence as content drift is closed for the ten compared files: their bytes match exactly.

composition S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600] (move 40) unchanged; no scorer run or score measurement in this task.
