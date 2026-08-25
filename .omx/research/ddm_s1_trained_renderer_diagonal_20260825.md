# DDM S1 trained-renderer diagonal — sealed, but not runnable on the current interfaces

**Date:** 2026-08-25  
**Harness task:** #1270  
**Disposition:** `BLOCKED_MISSING_COMPOSED_INTERFACES`  
**Authority:** scorer-free source/custody/build audit; no training, scorer, Metal, Modal, or evaluator invocation  
**Score claim:** false  
**Promotion eligible:** false

## Verdict first

The trained-renderer diagonal is **not entered and not refused**. I built and ran a typed S1 chain
compiler, verified the GB1 pointer archive, checked every record in RJ1's published custody inventory,
pre-registered the byte-versus-damage arithmetic, and sealed two seeds across three training windows.
Those six rows are registrations with null metrics, not training results. The compiler correctly
emitted no runnable command because the existing pieces do not compose on the object the charter names.

Five blockers are exact:

1. RJ1's signed inventory now verifies **189 retained file records / 192 inventory records**. The three
   absent records are 4,096-byte metadata sidecars named
   `runtime_sealed/.___pycache__`, one in each rung. The renderer payloads named below are present, but
   the custody manifest is not self-consistent, so S1 did not consume it.
2. WD3 rejects every seed except `20260815`; seed `20260816` cannot compile.
3. WD3 creates a fresh W96 model and packages it with the historical WD2 source sections. It cannot
   birth from the retained RJ1 initializer or byte-close on the GB1 body.
4. JG2 really re-encodes a supplied field, but no stage produces a moved-renderer token field. With no
   edits it only regenerates the unchanged field; calling that the missing object-changing half would
   be false.
5. The QS5 executable is hard-pinned to CP135/QS4 and its historical Vertigo store. It cannot consume a
   GB1/W96 archive, so an “in-compile” compensation claim would be stale-object theater.

Authoritative receipt:
`/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/seal_v2/S1_CHAIN_SEAL.json`, 14,860 B,
SHA-256 `f5cebf15430db8d8a7e172af05acac92cdb4519004d990954a40718e4c2c043c`. Its independent deterministic
repeat has the same bytes and SHA, and the full `--resume-from` invocation completed twice with the
same receipt. The exact machine blocker list is in `seal_v2/FIRE_ORDER.json`; source line evidence is
in `seal_v2/INTERFACE_AUDIT.json`. The v2 retention inventory covers 8 receipts / 43,222 B and has SHA
`e89f9f3cc59ee82287ff2b72c343e644b63a241694528a91f973368464f81e48`.

The exact pointer did not move. This unit therefore did **not** achieve the campaign goal.

Serializer landing is blocked by the managed filesystem. The required post-edit-SHA serializer was
invoked on only this memo, the S1 compiler, and its tests, with the required
`[no-triality] [p0-ledger-ok]` tags. Its canonical `git add` failed with `unable to create temporary
file: Operation not permitted`; the shared staged index remained empty. These three repo artifacts are
therefore untracked worktree files, not a commit. No alternate or fake commit was created.

## Source and custody verification

The GB1 pointer input is present and exact:

| numerator | denominator | result | receipt |
|---|---|---|---|
| verified GB1 archive bytes | required GB1 archive bytes | 180,215 / 180,215 | `SOURCE_PREFLIGHT.json` |
| observed GB1 SHA | required GB1 SHA | `ba1f3830…e3a4` / `ba1f3830…e3a4` | `SOURCE_PREFLIGHT.json` |

RJ1's manifest itself is intact: 73,543 B, SHA
`dd3b89b7f9d68f11f3d828457316b748b796aa55fea36304d566dd5cd2f8467c`, declaring 192 files,
5,375,503 payload bytes, and tree SHA
`576c16b2159cd3262dfa18e2df7bd53b7f8ac80c9c8dc546ccdc7dd5cd17d88a`. The result receipt is
20,334 B, SHA `405f89fb32a23fded3ded5b715989c2bf6efe7df6cedc79b98bbf89323fa26f0`.
The live filesystem check found:

| numerator | denominator | result | receipt |
|---|---|---|---|
| byte-and-SHA verified RJ1 records | RJ1 inventory records | **189 / 192** | `SOURCE_PREFLIGHT.json` |
| missing RJ1 records | RJ1 inventory records | **3 / 192** | `SOURCE_PREFLIGHT.json` |

All three failures are manifest-listed `.___pycache__` metadata sidecars, each expected at 4,096 B with
SHA `c3065f4a…b521`. This bounded fact does not prove semantic payload loss. It does prove the published
inventory cannot currently pass its own complete-custody assertion. No RJ1 input was copied or changed.

The three rungs are consequently typed as follows:

| RJ1 form | retained archive | S1 disposition |
|---|---:|---|
| `nested_group_dense_w72` | 169,489 B, SHA `a7310654…d2f6` | `REFERENCE_ONLY_NOT_W96_CLASS` |
| `pointwise_svd_w96_r32` | 175,177 B, SHA `ea930307…56d1` | `DEAD_AND_FORBIDDEN_SVD_MECHANISM` |
| `film_amortized_flat_w96` | 179,290 B, SHA `34855e3c…6d21` | `SOLE_ADMISSIBLE_RJ1_W96_FORM_FOR_S1_BUILD` after custody repair |

These are retained byte/container facts, not distortion measurements transferred into S1.

## The two missing halves, verified at source

NY1 names the unresolved W96 halves exactly as:

1. **`TRAINED-not-SVD W96 renderer`**
2. **`token re-encode on the moved object`**

The first is not satisfied by RJ1's arithmetic-mean film initializer, and the second is not satisfied
by copying or regenerating the old token stream. SVD-r32 stays dead unconditionally. Film-W96 stays
open only as an uncompensated, untrained instance.

## Pre-registered falsifier arithmetic

This table was written to `BREAK_EVEN_TABLE.json` before any S1 training or scoring run. No such run
occurred. The exchange rate is the exact contest rate coefficient
`25 / 37,545,489 = 6.658589531221714e-7 S/B`.

For every future exact row:

`damage_S = 100*(d_seg_candidate - 0.00020139) + sqrt(10*d_pose_candidate) - sqrt(10*6.37e-6)`

`composed_delta_S = damage_S - bytes_shed*(25/37,545,489)`

Admission requires `composed_delta_S < 0`. “Damage” is the numerator; the listed rate credit is its
maximum denominator-side allowance.

| window | bytes shed numerator | GB1 archive denominator | rate credit S = maximum Seg+Pose damage S |
|---|---:|---:|---:|
| zero-credit control | 0 | 180,215 B | 0 |
| RJ1 film-W96 observed renderer cut | 1,078 | 180,215 B | 0.0007177959515 |
| 5 KB | 5,000 | 180,215 B | 0.003329294766 |
| 10 KB | 10,000 | 180,215 B | 0.006658589531 |
| 15 KB | 15,000 | 180,215 B | 0.009987884297 |
| 20 KB | 20,000 | 180,215 B | 0.01331717906 |
| entire renderer block ceiling | 30,856 | 180,215 B | 0.02054574386 |
| GB1 fixed-distortion sub-0.12 demand | 42,229 | 180,215 B | 0.02811855773 |

Even deleting the whole 30,856-byte renderer leaves the score near 0.12757. A successful S1 row must
therefore collect real bytes from the re-encoded moved object as well as preserve distortion. The
1,078-byte film cut covers **1,078 B numerator / 42,229 B demand denominator = 2.553%** of the live
fixed-distortion demand. This is arithmetic only; no trained film row exists.

## Per-seed x per-window ledger

The floor is two seeds, `20260815` and `20260816`. The scope-reduced windows end at epochs 5, 15, and
30. Every metric remains null because the compiler refused before launch; null is not a negative.

| seed | window | bytes shed | d_seg | d_pose | composed delta S | status | planned receipt |
|---:|---:|---:|---:|---:|---:|---|---|
| 20260815 | ep0005 | null | null | null | null | `BLOCKED_NOT_RUN` | `stage_a/seed_20260815/epoch_0005/EVALUATION_RESULT.json` |
| 20260815 | ep0015 | null | null | null | null | `BLOCKED_NOT_RUN` | `stage_a/seed_20260815/epoch_0015/EVALUATION_RESULT.json` |
| 20260815 | ep0030 | null | null | null | null | `BLOCKED_NOT_RUN` | `stage_a/seed_20260815/epoch_0030/EVALUATION_RESULT.json` |
| 20260816 | ep0005 | null | null | null | null | `BLOCKED_NOT_RUN` | `stage_a/seed_20260816/epoch_0005/EVALUATION_RESULT.json` |
| 20260816 | ep0015 | null | null | null | null | `BLOCKED_NOT_RUN` | `stage_a/seed_20260816/epoch_0015/EVALUATION_RESULT.json` |
| 20260816 | ep0030 | null | null | null | null | `BLOCKED_NOT_RUN` | `stage_a/seed_20260816/epoch_0030/EVALUATION_RESULT.json` |

The machine copy is `SEED_WINDOW_LEDGER.json`. A future family refusal requires measured pose+Seg
damage above byte credit at every reachable W96 operating point for both seeds. S1 has measured
**0 rows / 6 sealed rows**, so neither an instance nor family refusal is licensed.

## Executable-chain audit

The worktree compiler is `experiments/ddm_s1_trained_renderer_diagonal.py`; its focused tests are
`experiments/tests/test_ddm_s1_trained_renderer_diagonal.py`. It performs real custody hashing,
pre-registers exact arithmetic, retains deterministic receipts, and refuses differing resume payloads.
It intentionally does not pretend the missing stage interfaces exist.

Current source evidence, captured in `INTERFACE_AUDIT.json`:

- WD3's fixed-seed guard is at `experiments/ddm_wd3_scorer_aware_width_distillation.py:402`.
- WD3's fresh model birth is at line 2117; its WD2 source-stream/residual/token packaging is at lines
  1486 and 1488.
- JG2's real edit application and typed `--edits` input are at
  `experiments/ddm_jg2_tail_reencode.py:404` and line 1024. No S1 moved-field producer was found in the
  searched composition surface.
- QS5's fixed output root is at `experiments/ddm_qs5_resolve_compensation.py:44`; CP135 bindings are at
  lines 173 and 175.

Verification completed:

```text
.venv/bin/python -m pytest -q experiments/tests/test_ddm_s1_trained_renderer_diagonal.py
11 passed
.venv/bin/ruff check experiments/ddm_s1_trained_renderer_diagonal.py experiments/tests/test_ddm_s1_trained_renderer_diagonal.py
All checks passed
```

No n600, scorer, Metal, Modal, or evaluator command ran. The S1 output root retains the source
preflight, interface audit, arithmetic table, seed/window ledger, blocked fire order, primary/repeat
seal, storage receipt, and retention inventory. The primary/repeat seal SHA remained identical across
two complete resume invocations. No GB1, DX2, RJ1, or shipped runtime tree was mutated.

## Sealed MAIN fire order

Every stage is `BLOCKED`; every `exact_command_argv` is null. That is deliberate: publishing argv for
an interface the invoked code rejects would be a fake fire order.

| stage | disposition | owner | consumer store | fire trigger |
|---|---|---|---|---|
| A: trained film-W96 | `BLOCKED-APPARATUS` | MAIN | `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_a/` | RJ1 custody is reissued coherent; WD3 admits both seeds, consumes the verified film initializer, and byte-closes on GB1 |
| B: jg2 re-encode | `BLOCKED-APPARATUS` | MAIN | `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/` | stage A retains a moved runtime and an explicit n600 token field or typed unchanged-field declaration |
| C: exact-object Schur | `BLOCKED-APPARATUS` | MAIN | `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_c/` | QS5 accepts and fingerprints the exact stage-B archive, runtime, frame-1 field, and base Pose6 |
| D: n600 authority | `BLOCKED-BEHIND-A-C` | MAIN sole scorer-lane router | `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/admission/` | A-C are receiver-closed, repeat-identical, fully retained, and sealed; score chunks are each <=120 pairs |

## RECALL EVIDENCE

**Stores consulted.** The pass read the charter and common contract; `PROGRAM.md`; the full governing
repo instructions; `main_hot_state.md`; `docs/vehicle_operating_system.md`; WD3's implementation spec,
handoff, fire order, trainer, receiver, and tests; RJ1's memo, runner, result, inventory, and retained
tree; NY1 memo plus JSONL correction rows; JG2's memo and real encoder; QS5's verdict, runner, and
exact-object binding functions; SY2, WA1, FB1, W72, DG2, RD2, JF1, GB1, JT21, DS1, WQ1, and AF1; the
canonical task-status, harness bridge, active-lane ledger, research indexes/DAG surfaces, and canonical
equation registry.

**Queries.** The bounded searches included `#1270`, `trained renderer`, `W96`, `TRAINED-not-SVD`,
`token re-encode`, `moved object`, `Schur compensation`, `seed`, `93.23`, `rate distortion`,
`cheap-to-shrink`, `resume`, `archive`, `runtime`, `edits`, `CP135`, and `qs5`. Canonical equations were
filtered across equation id, name, and summary for renderer, width, Schur, compensation,
rate-distortion, quantization, and distillation. No equation entry supplied a current GB1/W96
cross-object compile contract.

**Beyond-charter findings and plan changes.** WQ1 found the live training objective omitted a shipped
rate term; DS1 then built an exact multi-rung cheap-to-shrink objective, but a source search found it
only in its own module and tests, not wired into WD3. That is a live mechanism hypothesis, not a claimed
S1 blocker, because the charter did not authorize replacing WD3's objective. The task-store search found
#1270 in live `main_hot_state.md`, but did not find a matching current row in the bounded canonical
task-status/20260803 bridge surfaces. The active-lane ledger contains completed/stale RJ1 and QS4 rows,
not a live S1 claim. These findings changed the plan from “seal a runnable three-stage command” to
“seal an exact blocked state”; no lane was claimed and no compute was launched.

## What is and is not concluded

- **Concluded:** the current repo cannot execute the chartered S1 composition without source changes.
- **Concluded:** RJ1's published inventory has metadata-only record drift at 3 / 192 records and must be
  reissued before strict consumption.
- **Concluded:** no measured S1 seed/window row exists; the formulation is still unentered.
- **Not concluded:** film-W96 fails after training, moved-object re-encoding is useless, compensation
  cannot hold pose, or the dx2/gb1 lineage is globally unable to reach sub-0.12.
- **Not concluded:** DS1's objective improves S1. It is unconnected and unmeasured on this object.

Own-vehicle frontier unchanged: **S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**,
archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.

## NEXT_IF_RESUMED

- **Disposition:** `BLOCKED-CUSTODY-RESEAL`; **owner:** MAIN/RJ1 custody owner; **consumer store:** `/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1/`; **fire trigger:** re-inventory the current tree without transient metadata records, verify 100% of the new denominator, and publish the new manifest SHA without changing renderer payloads.
- **Disposition:** `BLOCKED-APPARATUS-STAGE-A`; **owner:** MAIN-designated WD3 implementer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_a/`; **fire trigger:** WD3 has reviewed tests proving both seeds compile, the film-W96 birth loads the verified RJ1 initialization, and packet/container construction preserves the exact GB1 non-renderer sections.
- **Disposition:** `BLOCKED-APPARATUS-STAGE-B`; **owner:** MAIN-designated moved-field producer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/`; **fire trigger:** stage A retains a moved runtime and a receiver-consumed n600 token-field payload; then JG2 control and encode both run resumably on that exact runtime and retain their streams and candidate archive.
- **Disposition:** `BLOCKED-APPARATUS-STAGE-C`; **owner:** MAIN-designated QS5 implementer; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_c/`; **fire trigger:** a reviewed QS5 exact-object entrypoint fingerprints the stage-B archive, runtime, realized frame-1 field, base Pose6, and solve outputs, then produces primary/repeat receiver-closed archives.
- **Disposition:** `QUEUED-BEHIND-A-C`; **owner:** MAIN sole n600 scorer-lane router; **consumer store:** `/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/admission/`; **fire trigger:** stages A-C are complete, every payload SHA verifies, each scorer chunk is <=120 pairs, and the exact candidate is sealed for one authority row.

## LIVE-HYPOTHESES

- Training the film-amortized W96 topology from its retained RJ1 initialization under WD3's real Seg/Pose objective may recover the initializer's pose failure while keeping its 1,078-byte renderer cut; it is plausible because the published film row was an arithmetic-mean initialization, not a trained optimum.
- Re-solving a token field against the moved renderer may expose conditional structure absent from the frozen-field RJ1 row; it is plausible under SY2's measured object-change law, but only a real retained field plus JG2 byte-close can price it.
- Wiring DS1's exact multi-rung cheap-to-shrink objective into the same WD3 trainer may improve the byte-damage slope across W96 allocations; it is plausible because fixed-allocation STE supplies no gradient about neighboring cheaper rungs, and the existing DS1 module evaluates the real quantizer rather than a weight proxy.
- Fresh QS5-style compensation may cancel most W96 pose leakage on the final object; it is plausible because QS5 proved the mechanism when solved in-compile, while stale compensation was specifically what failed.

## DEAD-ENDS

- SVD-r32 as the W96 mechanism is dead: its measured Seg component already requires an impossible negative archive to tie, and the charter forbids SVD truncation.
- Splicing any RJ1 renderer packet onto the old token stream is closed as a test of the diagonal: it leaves the priced object unchanged and only repeats the sharp frozen-field optimum.
- Regenerating the unchanged token stream with JG2 is not the missing moved-object half: it proves coder identity but creates no field/model interaction.
- Carrying QS4/QS5 compensation across objects is dead: the solve fingerprint is stale by construction and the prior stale row incurred the pose failure this charter is meant to avoid.
- A single-seed verdict is dead: WD3's seed has never been varied, so one seed cannot distinguish a treatment effect from training variance.

Own-vehicle frontier unchanged: **S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**,
archive SHA `ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4`.
