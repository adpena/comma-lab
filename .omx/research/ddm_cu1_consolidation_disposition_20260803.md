# ddm_cu1 — consolidation pass: disposition of the 155-file working-tree pile

**Date:** 2026-08-03 · **Arm:** `ddm_cu1` · **Axis:** apparatus (custody). **Pointer UNMOVED.**
Companion to `ddm_cu1_frontier_custody_20260803.md`. Nothing here is a score claim.

---

## Answer first

**Most of this pile is one parallel session's WIP and is not mine to land.** 101 of the 113
untracked files carry the **same date, 2026-07-26**, and 31 of the 41 modified files carry the
**same mtime to the minute, 07-31 08:53** — these are two bulk events, not 132 independent
decisions. The 07-26 wave is **coherent and green: 589 of 590 of its own tests pass.** It is not
junk; it is a complete session's landable work that was never committed.

I dispositioned all 155 and found exactly one file that is mine to land — the one unambiguous
orphan in my own domain. Its commit is blocked by MLX contention with the live ds=32 chain, not by
anything about the file; it is owned with a fire-condition rather than forced through a guard. **An accurate boundary beats a large commit**, and absorbing ~1.5 MB of another session's
code behind 100 rubber-stamped review passes would BE the #911 absorption incident, not the cure.

**Denominator: 155 examined / 155 dispositioned / 155 present** (1 dispositioned LAND is verified
but not yet committed — see the status correction below; 154 required no commit from me). (The coordinator measured 153;
the tree grew by 2 while I worked — live arms are writing. Both counts are correct at their
instant, which is itself the point: this pile has no quiescent moment to snapshot.)

| disposition | files | note |
|---|---:|---|
| **LAND** — verified, blocked on live-chain MLX contention (owned) | 1 | `src/tac/canonical_equations/__init__.py` |
| **BELONGS-TO-A-LIVE-ARM** — left untouched, owner named | 13 | `qd2`, `pz1`, `cr1`, `bp2`, `tw1`, `pfs1`, MAIN |
| **NOT-MINE-TO-LAND** — one session's WIP, named residue | 135 | the 07-26 taskspace wave + its 07-31 modified counterparts |
| **BLOCKED — fails its own tests** | 3 | `direct_description_minimizer` + test + carrier_compose |
| **ARTIFACT — certified, kept (block, do not delete)** | 3 | no rebuild command known |
| **GENUINELY UNKNOWN** | 0 | — |

---

## LAND (1 file) — VERIFIED SAFE, **NOT LANDED: blocked by live-chain contention, not by itself**

> **Status correction, measured after the disposition above was written.** The commit was
> **REFUSED by the preflight hook**: `BLOCKED: CI-blind (MLX-gated) test failed`, with
> `Fatal Python error: Bus error`. I did **not** bypass with `PREFLIGHT_SKIP_CI_BLIND_TESTS=1`
> (the hook itself calls that "NOT recommended", and it is the only surface that runs these
> modules at all — GitHub Actions skips them because mlx has no Linux wheel).
>
> **Cause, MEASURED — environmental, and partly self-inflicted.** Staging this file selects
> **36 MLX-gated tests** by token match. Meanwhile pid `63155` is the live ds=32 chain's
> `experiments/train_tr1_partition_renderer_mlx.py --num-pairs 600` at **RSS ≈ 13.98 GiB**, and
> `vm_stat` reports **~213 MiB free** on a 128 GiB host. A second MLX process claiming Metal
> buffers against that crashes Bus-error. The 11th selected test **passes standalone in 0.47 s**,
> so no individual test is at fault, and 8 added export lines cannot cause a segfault-class crash.
>
> **I stopped rather than retried.** Retrying re-runs a 36-test MLX batch against a 14 GiB live
> MLX trainer — that is exactly the disturbance of the live chain I was told to avoid, so paying
> it twice to land 8 lines is the wrong trade.
>
> **OWNED, with a fire-condition:** `ddm_cu1` (or the next arm in `src/tac/canonical_equations/`)
> lands this **when the ds=32 chain is idle**, by re-running the same serializer call — no code
> change needed, the file is already reviewed twice and verified below. Nothing else in this
> disposition depends on it.



`src/tac/canonical_equations/__init__.py` — **+8 lines, 0 deletions.** It exports
`build_seg_only_base_pose_degradation_v1`, `populate_seg_only_base_pose_degradation_v1`,
`pose_degradation_ratio` from `ddm_cr1_seg_only_base_pose_degradation_20260801`.

This is a genuine orphan of exactly the class this arm exists to close: **the module landed and is
TRACKED; its export wire-in did not.** Measured before landing:

* **#883 hazard checked and CLEAR.** The coordinator flagged that the serializer repair path once
  silently committed a non-empty index and dropped a sister arm's canonical-equation rows. The
  working-tree diff is a **pure addition** — 8 insertions, **zero deletions**. Nothing was dropped.
* Target module exists and is **tracked**. `import tac.canonical_equations` → **255 exports**, all
  three names resolve.
* `pytest src/tac/canonical_equations/` → 1016 passed, 2 failed. **Both failures are PRE-EXISTING**
  — the same two test IDs fail identically in a clean clone at committed HEAD `18c4a1ba`
  (`test_live_registry_surfaces_task_504_application_equations_once`,
  `test_live_registry_surfaces_curvelet_shearlet_and_metric_equations_once`). Not introduced here;
  named so the next arm does not inherit them as mine.

---

## BELONGS-TO-A-LIVE-ARM (13 files — left untouched, owner named)

| file(s) | owner |
|---|---|
| `.omx/research/ddm_qd2_rebaseline_against_cx1_20260803.md` | **`ddm_qd2`** (LIVE — re-pricing banked ΔS) |
| `experiments/ddm_pz1_scorer_plane_pose_delta.py` | **`ddm_pz1`** (LIVE — pose) |
| `experiments/ddm_pfs1_ep_warp_pose_solve.py` (mtime 08-03 05:57) | **`ddm_pfs1`** / pose line |
| `.omx/research/ddm_cr1_20260801/cr1_intent.patch.txt` | **`ddm_cr1`** |
| `reports/ddm_bp2/{overlap,reach}_n600.jsonl` | **`ddm_bp2`** |
| `experiments/ddm_tw1_token_waterfill_state_dependence.py` + `.omx/research/ddm_tw1_…_20260801.md` | **`ddm_tw1`** |
| `tools/pfs1_recompose_warp_base_and_eval.py` (08-01) | **`pfs1`** line |
| `.omx/state/{current_focus.md, active_lane_dispatch_claims.md, lane_maturity_audit.log, operator_p0_ledger.jsonl}` | **MAIN** — all four are **append-only growth (+31 lines, 0 deletions)**; committing them mid-flight would race MAIN's own writes |

`ddm_de1` (codex) is in an isolated worktree and has no file in the shared tree — nothing to leave.

---

## NOT-MINE-TO-LAND — the named residue (135 files)

**One session, 2026-07-26, the v9/v10 `taskspace_*` / `witness_dsl` / `witness_control` lineage**
(plus ~14 modified counterparts in the 07-31 08:53 batch). Composition: 72 `src/tac`, 22 `tools/`,
14 `tools/tests/`, each module paired with its test.

**Measured, so the next owner does not have to re-measure:**

* **589 of 590 of its own tests pass.** Single failure:
  `test_taskspace_single_stage_score_attempt_v1.py::test_selected_row_is_frozen_from_exact_ledger_identity`
  → `SingleStageScoreAttemptError: selected row failed recursive G120/G112 reopen` (ledger-state
  dependent, not a code fault on its face).
* It contains **all 11 clone-breakers** from the companion memo — the modules whose absence is what
  makes a fresh clone fail. So landing this wave is also the fix for that.
* The 07-31 08:53 modified batch is **substantive, not mechanical** — e.g.
  `src/tac/score_target_filter.py` replaces a hardcoded `DEFAULT_SCORE_LOWERING_TARGET = 0.19` with
  a pointer-verified dynamic target (+79 lines). That is real, wanted work.

**Why I am not landing it:** ~50 `.py` files → **100 `review_tracker.py mark-file` passes** on code
I did not write and cannot review at depth in this arm. Two passes I cannot honestly back are worse
than an untracked file, because they convert "unlanded" into "reviewed" — a false custody claim.

**Note on the review gate, since #902 blames it:** it is not what blocks this. The gate's cost is
two recorded passes, which is the *right* price. What blocks it is that **nobody owns the wave**.
That is the finding: the gate makes landing cost real work, the working tree makes NOT landing cost
nothing visible, and an unowned wave therefore never lands. The cure is an owner, not a weaker gate.

**Owner needed. Fire-condition:** the next arm that touches `taskspace_*` / `witness_dsl` /
`witness_control` owns landing or explicitly retiring this wave, worst-first by the clone-breaking
ranking now emitted by `tools/audit_untracked_source_artifacts.py`.

---

## BLOCKED — fails its own tests in the working tree (3 files)

`src/tac/optimization/direct_description_minimizer.py`,
`src/tac/optimization/tests/test_direct_description_minimizer.py`, and their sibling
`src/tac/optimization/direct_description_carrier_compose.py`.

**MEASURED both ways:** in the working tree, 2 tests fail —
`test_failure_receipt_refuses_self_asserted_fixture_and_fabricated_axis` and
`test_unhealthy_optimizer_cannot_mint_failure_token`. In a clean clone at committed HEAD
`18c4a1ba`, **all 25 pass**. So the failures are introduced by the *uncommitted edits*, not by the
committed code and not by my changes.

**Do not land as-is.** These two tests guard failure-token minting — the honest-negative custody
surface — so a red here is exactly the kind of red that must not be waved through.

---

## ARTIFACTS — certify-or-block (3 files: CERTIFIED, KEPT, NOT DELETED)

CLAUDE.md's rule is "certify or block": never delete or move a large artifact unless a
machine-readable record preserves deterministic reproducibility, including the **rebuild command**.
**No rebuild command is known for any of these, so all three BLOCK — the bytes stay.** The record:

| path | bytes | sha256 (16) | mtime | note |
|---|---:|---|---|---|
| `ep854_tokens.dr7t` | 271,505 | `1bab7aef76b63416` | 08-01 20:56 | **stray at repo ROOT**; referenced by **no tracked code** (exhaustive grep over `tools/ experiments/ src/ .omx/research/`). Orphan blob — needs an owner to name its producer, then cold-store to the SSD tier. |
| `reports/ddm_bp2/overlap_n600.jsonl` | 232,003 | `a93293bfdbe18896` | 08-02 20:02 | `ddm_bp2` n600 measurement output |
| `reports/ddm_bp2/reach_n600.jsonl` | 678,373 | `448d0a582717306d` | 08-02 21:53 | `ddm_bp2` n600 measurement output |

(`.omx/research/ddm_cr1_20260801/cr1_intent.patch.txt`, 34,082 B, `cbebd84569a93a12` — small, and
`ddm_cr1`'s to keep or land.)

---

## What I did NOT verify

* I did not read the 135 residue files individually. I measured their **test outcome** and their
  **import health**; I did not audit their content, and I make no claim that they are correct — only
  that they are coherent and green, which is what decides "not junk", not what decides "land it".
* I did not diff the 07-31 08:53 batch file-by-file. I sampled 7 of 31 and confirmed the change is
  substantive rather than mechanical; that is enough to refuse the "it's just a formatter, land it"
  read, and not enough to endorse the batch.
* `pile_lines: 3731` (the consolidation hook's number) is **unreduced** by this pass — I removed
  ZERO files from the pile, since the one landable file is still blocked. Saying otherwise would be the means-as-ends failure.

---

## Bottom line for the back-pressure guard

The guard's refusal — *"153 uncommitted files … Launching more ACCRUES drift"* — is **correct and
still stands**. This pass did not clear it and could not honestly: **135 of the 155 are one
unowned session's coherent, green, unlanded work.** The blocker is an OWNER, not a decision.
Assign the 07-26 taskspace wave and the guard clears in one commit batch; leave it unowned and it
will still be here, and still green, and still unlanded, next week.
