# ddm_pd1 — POSE-DIRECTED token pre-distortion, the re-solve credit as the objective

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd1_pose_directed_token_predistortion_all_pairs_20260913`. Base: move 49,
S 0.13632299781031237 @ 179,153 B, archive sha
`73e41a6620bd4ea3aaf236eff9de46391857907527358e8eb40ded0925a1c214`. Axis
`[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; bytes EXACT through the shipped RLC1 coder and the shipped container.
**No score is claimed. Only `upstream/evaluate.py` on shipped bytes is a score, and MAIN
fires.** No Modal, no fire, no packet.

Pose solver, stated once: every resolved pose number here comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` (= pass 8's refine = cr1's; the same
object at these budgets), driven at PoseNet batch 1 on the moved render; every n600 base
number comes from `up2.measure_pose` at batch 8 on move 49's own cold parse-back `0.raw`.

---

## 1. The headline

**A candidate is built and sealed: 179,195 B, archive sha
`1ea274f612a26183f31f6d439505d0289bd3751b4da9467343fc156203503cd7`, projected S
0.13628242427525275 — net −4.0574e−05 vs move 49, 2.03 bars, clearing the −2e−05 fire bar.
20 pairs, 20 changed tokens, 176 re-solved carrier coordinates.**

| leg | value | how |
|---|---:|---|
| seg | **+6.7862e−06** | 12,127 → 12,135 flipped cells; +8 net, MEASURED on the overlay pair by pair and re-measured on the candidate's own parse-back |
| pose | **−7.0557e−05** | RESOLVED 4.448947e−06 against the base 4.5435687e−06, re-verified n600 on the composed object |
| rate | **+2.6634e−05** | +42 B EXACT (tail +40 by the subset's own real encode, twins byte-identical; carrier +2 from the splice) |
| **net vs the pointer** | **−4.0574e−05** | **2.03 bars** |
| net like-for-like vs this instrument's null control | **−3.5805e−05** | 1.79 bars; the −4.769e−06 difference is the local-versus-T4 pose-print class |

**And the finding underneath it: pp1's floor law does not generalize off the 12 hard pairs.**
pp1 measured the best resolved-pose fall at **0.25 %–0.45 %** on the pairs that carry half the
pose mass and called the resolved pose a FLOOR. On ordinary pairs the same actuator — one
token, the shipped render, the frozen argmax, the same `refine_pair` — moves the resolved pose
by a **median −21.6 %** and by up to **−96.2 %**. The hard pairs are hard because their
residual lies outside the carrier's twelve-dimensional reach; ordinary pairs are not, and the
edit's job there is to move frame_1 somewhere the carrier CAN follow.

---

## 2. The base, reproduced rather than inherited

| quantity | measured | control | agreement |
|---|---:|---|---|
| n600 mean d_pose on move 49's own decode | **4.543568679770593e−06** | pp1's independently measured vector | **bit-identical**, max abs difference **0.0** |
| " | " | pass 8's F7 gate ratio | **0.998586523026504**, reproduced to all digits |
| " | " | the T4 print 4.55e−06 | ratio 0.998587 |
| median / max | 3.34503739671486e−07 / 2.1283759625946122e−04 | — | max/median **636×** |
| top-12 share | 52.49 % | pp1 | same 12 pairs |
| carrier codes | (600, 12) int32, sha `78cf4fe54b2bb6a9…` | read from move 49's own archive | — |

Receipts: `base/POSE_BASE_MOVE49.json`, `base/pose_base_move49.npy` (sha
`05bd30248a167a6b9dab00841dceb343f232265240c1590912875f2e625c11f1`), `base/codes_move49.npy`.

**F_base_reproduction did not fire.** Three independent measurements of the same object
agree bit for bit.

### The base band is ABSOLUTE, and this arm treats a population spanning 3,000×

pp1 §10's lesson is the denominator, not the epsilon. Re-measured here on **40 pairs drawn
across the whole d_pose range** (the 20 stratified smoke pairs plus 20 seeded random
non-floor pairs):

| quantity | measured |
|---|---:|
| batch-1 repeat, all 40 pairs | **bit-exact** |
| max **absolute** batch-1 vs batch-8 gap | **2.2643e−09** |
| max **relative** gap | 9.9181e−03 (on a pair at ~1e−07 — the same absolute noise over a tiny denominator) |
| gate used | **2.2643e−08** = 10× the observed absolute maximum |

A relative gate would have been ~400× too loose on the smallest pairs and ~4× too tight on
the largest. Receipt: `base/BASE_TOLERANCE.json`.

---

## 3. The economics, DERIVED before the search, and they EXPIRE

At this operating point (`base mean 4.5435687e−06`):

| term | value |
|---|---:|
| S per archive byte | 6.658589531221714e−07 |
| S per flipped SegNet cell (T4-carried) | 8.482724499051702e−07 |
| S per unit of **one pair's** d_pose | **1.2362895701574137** |

**A hard bound, not a heuristic.** A single-token edit changes one pair's pose, and
`|credit| ≤ base` because the resolved d_pose cannot go below zero. So a pair whose own
base d_pose is below `rate_S_per_token / 1.2363` can **never** admit with one token, no
matter what the edit does:

| price | bits/token | required per-pair credit | non-floor pairs able to pay |
|---|---:|---:|---:|
| pass 8's full field | 8.93 | 6.0121e−07 | **220** |
| pass 7's admitted subset | 5.0149 | 3.3762e−07 | **286** |
| an optimistic 3.0 | 3.00 | 2.0197e−07 | **345** |

The search population is the **345** pairs above the optimistic bound, excluding pp1's 12
floor pairs. The 243 pairs below it were not searched because no realizable credit there can
pay for its own token. Receipts: `PREREGISTRATION.json`, `search_population.json`.

---

## 4. The smoke — and it falsified the prior this arm was most afraid of

20 stratified pairs, pure pose-saliency ranking (pp1's `search_b` verbatim, K = 3 cells ×
{−1,+1}), every proposal fully realized.

**MEASURED: the median best per-pair resolved credit is −13.0 % of the pair's own d_pose,
best −86.1 %.** pp1 measured 0.25 %–0.45 % on the 12 hard pairs and found a FLOOR there.
**That floor does not generalize.** The hard pairs are hard because their residual lies
outside the carrier's twelve-dimensional reach; ordinary pairs are not, and on them a single
token edit moves the resolved pose by tens of per cent.

The binding leg is therefore **seg, not pose**: 51 of 53 smoke proposals cost ≥ 1 cell, and
one cell (8.48e−07 S) already exceeds the pose credit most pairs can give.

Stratified extrapolation of the smoke (MODELLED on rate at 8.93 bits/token):
**−1.62e−05 S = 0.81× the −2e−05 bar → stop-rule branch B (MARGINAL)**, which is the branch
the pre-registration said to expect. Receipts: `SMOKE_YIELD.json`, `SMOKE_VERDICT.json`.

---

## 5. K, declared — and why the full search screens before it refines

The smoke measured the binding leg, so K is not one number but two.

| | smoke | full search |
|---|---|---|
| proposals ranked per pair | 3 saliency cells × {−1,+1} | **16** cells × {−1,+1} |
| cell ranking | pose saliency | pose saliency **masked to cells whose shipped argmax is uniform over a (2r+1)² window, r = 3** |
| SCREENED per pair (render + frozen argmax) | 3.7 realized | **up to 32**, ~20 after alphabet clipping |
| REFINED per pair (carrier re-solve) | every screened proposal | the ≤ 2-cell survivors, cheapest-on-seg first, **K_refine = 3** |

**The ordering is pp1 §5's, not a new mechanism** ("an ORDERING change, not a mechanism
change: every number the admission uses is still realized"), and the reproduction control is
that my stage returns pp1's own `search_b` rows **bit-identically** — 3 of 3 comparable rows,
`d_cells` and `d_pose_resolved` equal to all digits, with `--interior-radius 0` so the cell
ranking is pp1's.

**What the ordering bought, MEASURED:**

| | pure saliency (smoke) | screened + interior mask |
|---|---:|---:|
| seg-neutral share of proposals | **3.8 %** (2 of 53) | **10.4 %** (76 of 730) |
| pairs with at least one seg-neutral candidate | — | **72.2 %** (26 of 36) |
| seg-neutral share of REFINED rows | 3.8 % | **52 %** |

**Timing, MEASURED on this host:** 130.4 s for one pair at 3 screens + 3 refines with
`--threads 3` under six-way contention (including ~70 s of body load); pp1 measured 475.0 s
for 16 fully realized proposals over 2 pairs. The full search runs 8 shards at
`--threads 2`, measured at **~2.2 min per pair per shard** → ~3.9 pairs/min over the fleet
→ the 345-pair population in ~90 min. The search walks each shard's pairs in **descending
d_pose**, so a prefix stop keeps the pairs that can pay rather than a random sample.

---

## 6. The search

### The controls that had to pass before a single row was read

| control | what it refuses | result |
|---|---|---|
| **reproduction vs pp1** | a look-alike actuator | with `--interior-radius 0` my stage returns pp1's own `search_b` rows **bit-identically** — 3 of 3, `d_cells` and `d_pose_resolved` to all digits |
| **F_flips_identity** | a seg instrument that is not the shipped one | the shipped decode's argmax against DALI GT gives **12,127** flips — pass 7's own receipt, exactly — and **every searched pair's `base_flips` matches its per-pair count, 0 disagreeing** |
| **F_base_gate** | a batch-shape offset masquerading as a credit | every scored row inside the MEASURED absolute band 2.2643e−08 |
| **unmoved-resolve attribution** (cr1's n600 vector) | a BASE-configuration credit sold as this edit's | cr1 measured the unmoved re-solve for all 600 pairs; its whole n600 gain is **−4.050e−07**, essentially all on pair 391 (not in this arm's admitted set). **Every admitting pair owes ≤ 0.08 % of its credit to slack that existed before the edit.** |
| **cr1's phantom-gain class** | `refine_pair` reporting a gain while returning its start codes | **every admitting pair moved 5 to 12 of its 12 carrier coordinates** |

### What the search found

The search was stopped at **101 pairs carrying a row** (308 realized rows, of a 345-pair
population) when the marginal return went flat — four consecutive pairs added no new
admitting row, and the remaining population is strictly lower d_pose, where the hard bound
`|credit| ≤ base` caps what any credit can be worth. The shards walk **descending d_pose**,
so the stop keeps the pairs that can pay rather than a random sample. The rows are on disk
and a successor arm resumes per pair.

| | measured |
|---|---:|
| pairs with at least one realized row | **101** |
| realized rows (render + frozen argmax + carrier re-solve, each) | **308** |
| rows inside the measured base band | **308 / 308**, 0 rejected |
| screened proposals (render + frozen argmax only) | 1,522 |
| pairs admitting on the modelled arithmetic | **31**, sum **−5.318e−05** = **2.66 bars** |
| carried into the candidate field (best per pair, kept if modelled ΔS < 0 at 3.0 bits/token) | **41 pairs, 41 tokens** |

**The per-pair credit histogram — the number that falsifies pp1's floor law off the hard
pairs.** For the 41 carried pairs, the best realized credit as a fraction of that pair's own
d_pose:

| band | pairs |
|---|---:|
| −100 % … −50 % | **7** |
| −50 % … −25 % | **11** |
| −25 % … −10 % | **13** |
| −10 % … −5 % | 5 |
| −5 % … −2 % | 4 |
| ≥ 0 % | 1 |

Median **−21.6 %**; best **−96.2 %** (pair 69, whose resolved pose is very nearly erased).
pp1 measured **0.25 %–0.45 %** on the 12 hard pairs. Two orders of magnitude, on the same
actuator, the same instrument, and the same object — the difference is WHICH pairs.

**And the seg cost, which the smoke said was binding, is largely paid off by the screen:**

| d_cells of the carried edit | pairs |
|---|---:|
| −1 (a seg REPAIR) | **4** |
| 0 | **22** |
| +1 | 9 |
| +2 | 4 |
| +5 | 1 |
| +12 | 1 |

**26 of 41 carried pairs cost zero or negative seg cells.** Total +30 cells across the 41.

Candidate field `assemble/field_candidate.npz` sha
`2e9d7089c52ab148ce9670d83a236fbf50a374612a2caf982b033fa30e4cf1e6`; credit sum
**−8.6146e−05** of d_pose (mean −1.4358e−07 over 600 pairs).

---

## 7. The admission, and the composition re-verified rather than summed

### The composition, MEASURED on one object

The per-pair search realizes each edit alone. The composed object is a different question, so
it was RE-MEASURED: a 600-pair overlay rendered from the candidate field, an n600 pose leg on
it, and a carrier re-solve that starts from the **SHIPPED** codes — never from the search's
own — so the re-verification is independent of the thing it checks.

| leg | per-pair sum | composed, re-verified | realized fraction |
|---|---:|---:|---:|
| pose (d_pose) | −8.614598e−05 | **−8.613602e−05** | **0.999884** |
| seg (cells) | +30 | **+30** | **1.000000** |

Per-pair, the composed re-solve returns the search's resolved value with ratio
**min = max = median = 1.0** over all 41 pairs; the 0.999884 in the aggregate is the batch-1
versus batch-8 band this arm measured at 2.2643e−09 absolute, not a composition loss.

**The two null controls that make those numbers mean something:**

* the **559 unedited pairs** are bit-identical to their base on BOTH legs — max
  |stale − base| = **0.000e+00** and max |resolved − base| = **0.000e+00** — so nothing leaks
  from an edited pair to an untouched one;
* the STALE pose (candidate renders, shipped carrier) is **48.06× the base**. The re-solve
  recovers all of that and then some. A credit measured without the re-solve would be a
  catastrophe measured as a win.

Composed d_pose mean **4.4000086487e−06** against base **4.5435686798e−06** — **−3.16 %** on
the population mean from 41 single-token edits.

### The rate leg, by REAL encode, twins — and it is the leg that decides the size of the win

| field | pairs / tokens | exact archive | Δ vs move 49 | bits/changed token | twins |
|---|---:|---:|---:|---:|---|
| control (move 49's own field) | 0 / 0 | **179,153 B** sha `73e41a66…` | 0 | — | **byte-identical, 2 processes** |
| full candidate field | 41 / 41 | **179,240 B** sha `b3506c69…` | **+87 B** | **16.98** | **byte-identical, 2 processes** |
| admitted subset | 20 / 20 | **179,193 B** sha `ad754903…` | **+40 B** | **16.00** | **byte-identical, 2 processes** |
| the built candidate (subset tail + re-solved carrier) | 20 / 20 | **179,195 B** sha `1ea274f6…` | **+42 B** | — | staged archive reproduces the pricer's repack byte for byte |

**A measured law this arm adds: an isolated one-token-per-pair edit costs about twice a
clustered one.** pass 8 measured **8.93** bits per changed token on a field whose 163 tokens
sat on 96 pairs; this arm measures **16.98** on a field whose 41 tokens sit on 41 different
pairs — one context break each, with no neighbour to amortise it against. The pre-registered
economics used 8.93 and were therefore **1.90× optimistic on rate**; the admission re-ran on
the measured ledger and the subset shrank from 31 pairs to 20 because of it. Said plainly:
the arm's own prior was wrong in the direction that matters, the real encode caught it, and
the number quoted is the measured one.

**And the container did NOT draw a lottery here.** `realized / first-order-ideal` = **0.99946**
on the full field (+87 B realized against +87.047 B of the coder's own ideal length), so the
container-break term that pp1 measured at ±33 B on the semantic member is absent on this
object at this edit shape. That is a measurement on one field, not a law.

### A by-product worth recording: repairs OUTSIDE the repair search space

pass 8's F14 — *"not one NEW repair on a pair whose render did not move"* — held three passes
running, and it is scoped to the REPAIR search space (candidate positions drawn from the
residual). This arm screens a different space: pose-salient cells masked to argmax-INTERIOR
positions. On pairs whose render has **not** moved since move 49 it finds seg **repairs**
there — MEASURED, and it is a lower bound because no attempt was made to enumerate that
space. The law is scoped to the search space, not to the pair. Receipt:
`REPAIRS_OUTSIDE_THE_REPAIR_SPACE.json`.

---

## 8. Four defects this arm caught with its own controls, and the two it inherited

1. **The token alphabet is 5, and the guard was 255.** pp1's `search_b` clamps a proposal
   with `0 <= new <= 255` — the uint8 STORAGE domain, not the SYMBOL domain. The shipped
   `token_embed` has exactly **5** rows, so a cell holding 4 with delta +1 indexes row 5 and
   torch raises `IndexError` inside the embedding. pp1 never met it because both its pairs'
   salient cells held 0 or 2. **All eight shards died on it**, not one — the defect was the
   class, and a shard-local fix would have been the point-fix. The cure reads
   `num_embeddings` off the renderer rather than hardcoding a class count, and adds a
   fail-closed check that the field's own maximum symbol fits the embedding.
2. **zsh arrays are 1-indexed — pp1 §10's own lesson, repeated here.** The first smoke launch
   built its shard pair lists from `${SH[$i]}` with `i` from 0, so shard 0 received an empty
   `--pairs` (argparse caught it in seconds) and the last pair set was never launched. No
   measurement was contaminated; every later launch passes pairs as literals.
3. **ExFAT AppleDouble stubs are not tree files.** `shutil.copytree`'s metadata copy writes a
   `._<name>` sidecar beside every file carrying an extended attribute on APDataStore, and
   the candidate-tree census reported **56 phantom extra files**. The build now copies bytes
   without metadata and every tree comparison skips `._` stubs and bytecode caches.
4. **The `patch_inflate_pins` / `MANIFEST.sha256` class — and a correction this arm owes its
   own commit message.** `ddm_pd1_candidate_tree.py` rewrites every listing row from the bytes
   on disk and re-hashes the whole listing from outside the tree, and its identity control
   passes: handed move 49's own archive it reproduces move 49's tree **byte for byte** (0 files
   differing, 49 of 49 rows unchanged, 0 validation problems). But the commit that landed it
   says it "closes the class pass 7 named," and that **over-claims**. MEASURED afterwards at
   source: `patch_inflate_pins` already calls `rebind_receiver_manifest` (commit `3e4645313`,
   "manifest pins: rebind receiver listing transactionally"), so the class was cured upstream
   and my builder's rewrite found **zero rows to change**. It is a VALIDATOR over a cure that
   already exists, not the cure. Said here rather than left standing in the commit message.

Inherited and honoured rather than rediscovered: cr1's warning that `refine_pair` can report
a gain while returning the codes it started with — **checked on every admitting pair, and
every one moved 5 to 12 of its 12 carrier coordinates**; and `sys.dont_write_bytecode` on
every module that imports from a custody tree.

---

## 9. What this does NOT claim

1. **No score of any kind.** Every S here is a PROJECTION on measured legs. Only
   `upstream/evaluate.py` on the shipped bytes is a score, and MAIN fires.
2. **The pose numbers are `[macOS-CPU advisory]`,** measured on a frozen CPU-torch PoseNet
   against DALI-lineage GT. The instrument's base is 4.5435687e−06 against the T4 print
   4.55e−06 (ratio 0.998587), and the last four packets each measured a local-versus-T4
   pose-print class of a few e−06 in the OPTIMISTIC direction. Both the like-for-like net
   (against this instrument's own null control) and the nominal net (against the T4 pointer
   value) are reported; the exact row decides between them.
3. **The container-break lottery is not removed, only measured.** The campaign's standing
   sd is 34.8 B = 2.32e−05 S ≈ 1.16 bars, and pp1 re-measured 33.11 B on a different
   container. The SUBSET's own real encode is the charge, so the lottery is drawn before the
   build rather than after — but a successor field draws again.
4. **verdict_scope: FORMULATION, on this object.** What is measured is pose-directed
   single-token pre-distortion on move 49's field, proposals ranked by pose saliency masked
   to argmax-interior cells, admitted on the resolved pose. Multi-token edits per pair, a
   different proposal ranking, and any other carrier parametrisation are untouched.
5. **The 243 pairs below the optimistic bound were not searched.** That exclusion is DERIVED
   (`|credit| ≤ base`), not measured — it says those pairs cannot pay at ≥ 3.0 bits/token,
   not that no edit there does anything.
6. **The per-pair search budget is not uniform across the whole run.** The 20 smoke pairs
   were searched at 3 saliency cells with every proposal refined; the rest at 16
   interior-masked cells with the ≤ 2-cell survivors refined. Each row is a realized
   measurement either way; "best per pair" means best found under that pair's budget.
7. **pp1's floor law is refuted OFF the hard pairs, not ON them.** Nothing here reopens the
   12 floor pairs, which remain excluded on pp1's measurement.
8. **The search is not exhausted.** 101 of the 345-pair population carry a row and the shards
   are resumable per pair; the stop was a judgement that the marginal return had gone flat,
   not a measurement that it had. A successor resumes the same command.
9. **16.98 bits/token is this field at this edit shape**, measured once. It is not a law, and
   the 8.93 it corrects is pass 8's on a different edit shape — both are real.

---

## 11. What this hands the next arm

1. **The actuator has supply left.** 31 of 101 searched pairs admit on the modelled
   arithmetic; 20 shipped. The 244 unsearched pairs above the optimistic bound were never
   touched, and the search resumes per pair.
2. **The binding leg is rate, not pose and not seg.** At 16.98 bits/token an isolated
   one-per-pair edit needs a credit of 6.01e−07 just to break even at pass 8's price and
   1.14e−06 at this one. Two levers follow directly: **cluster the edits** (pass 8's 8.93
   bits/token came from 163 tokens on 96 pairs, and a clustered pose search would amortise
   the context break), or **spend two tokens on one pair** where the second is nearly free.
3. **The floor is a property of the pair, not of the actuator.** The pairs where a single
   token moves the resolved pose by half are ordinary pairs whose carrier can follow frame_1;
   the 12 that cannot are outside every render-side actuator. A reach-side lever (more carrier
   DOF, priced by pc2/pc3 at a 349.9 B family slack) is still the only door to those.
4. **Repairs exist outside the repair search space** (§7) — a pose search found 17 of them on
   pairs pass 8's reach law says hold none. Enumerating that space is unowned.

---

## 10. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd1/`**. APDataStore rather than Vertigo because
Vertigo held **38 GiB** free at launch, below its 40 GiB reserve — the reserve was never
lowered and nothing was written there. Every payload is hashed in `RETENTION_MANIFEST.json`.

**MEASURED: 6,030,780,048 B over 893 files, every one hashed — under the 8 GiB cap.**
`RETENTION_MANIFEST.json` carries bytes and sha256 for each, including the losers: all 308
realized search rows and all 1,522 screened proposals, not only the 20 that shipped.

| path | what |
|---|---|
| `candidate/candidate_archive.zip` · `candidate/candidate_runtime/` | **179,195 B sha `1ea274f6…`** and the 51-file runtime, digest `aa10352c…` |
| `SEAL_ddm_pd1_pose_directed_token_predistortion_contest_cuda.json` | **SEAL_VALID**, file sha `62010074ba21b6cd…`, seal sha `430f3599781de996…` |
| `parseback/0.raw` · `parseback/PARSEBACK_RESULT.json` | the cold decode (3,662,409,600 B sha `135c9b3a…`) and its receipt |
| `seg_final/argmax_n600.npy` · `STEP0_RESULT.json` | F4 on that decode: 12,135 cells |
| `pose/overlay_pd1/` | the 600-pair odd-frame overlay the pose leg scored (1,831,204,800 B) |
| `pose/pose_stale.npy` · `pose/pose_resolved.npy` · `base/pose_base_move49.npy` | the three n600 pose vectors |
| `refine/refine_rows_*.jsonl` · `refine/codes_resolved.npy` | the composed re-solve, per pair |
| `search/search_b_*.jsonl` · `search/screen_*.jsonl` · `smoke/` | **every realized row and every screened proposal, winners and losers** |
| `assemble/` · `admission/` · `stage/` | the candidate field, the Lagrange sweep and its trace, the staged tail |
| `rlc1_price/` | the pricer's INPUTS, both control encodes, both full-field encodes, both subset encodes, the per-pair bit ledgers |
| `PREREGISTRATION.json` · `SMOKE_VERDICT.json` · `F6_CONTROL_ENCODE.json` · `F_COMPOSITION.json` · `F_COMPOSED_SEG.json` · `F_FLIPS_IDENTITY.json` · `F_UNMOVED_RESOLVE_ATTRIBUTION.json` · `REPAIRS_OUTSIDE_THE_REPAIR_SPACE.json` | the pre-registration and every control receipt |
| `PIPELINE_PLAN.md` | the chain as exact commands, written while the search ran |

Producers: `experiments/ddm_pd1_pose_directed.py` (`prereg | search | yield | assemble`) and
`experiments/ddm_pd1_candidate_tree.py` (`build`), both ruff-clean, plus pp1's
`search-b | base | base-tolerance`, sj1's `ddm_sj1_rlc1_price.py` and
`ddm_sj1_joint_admission.py` unchanged. Nothing under
`/Volumes/VertigoDataTier/pact/ddm_sj1_pass7`, `ddm_sj1_pass8`, `ddm_pp1`, `ddm_cb1`,
`ddm_cr1` or `ddm_so2` was written; every module that imports from a custody tree sets
`sys.dont_write_bytecode = True`. **Vertigo held 38 GiB at launch, below its 40 GiB reserve —
the reserve was not lowered and nothing was written there.**

<!-- # FORMALIZATION_PENDING: the equations leg is written by tools/pointer_move_packet.py --equations-leg at harvest, and only on an exact row. The score arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

### The seal

`/Volumes/APDataStore/pact/ddm_pd1/SEAL_ddm_pd1_pose_directed_token_predistortion_contest_cuda.json`
— file sha **`62010074ba21b6cdd596acb8cb908135890387d01c394886dbc886b84827b640`**, seal sha
`430f3599781de996186582212ca8ed17f99770b174cdea47057e091a7522e717`, **SEAL_VALID**,
axis `contest_cuda`, admit bar net ΔS < −2e−05 vs 0.13632300 at tolerance 0. Runtime 51 files,
1,008,554 B, digest `aa10352cb6aa4f7f…`. Decode wall clock INHERITED from move 49's own
`t4_direct` leg — 1,106.219 s against the 1,260 s limit — with the pr18 behaviour digest
`9f6e71680a13d859…` present and matching. **NO Modal, NO fire, NO packet: MAIN fires.**

Own-vehicle frontier (unchanged by this arm — MAIN fires):
**S 0.13632299781031237 @ 179,153 B [contest-CUDA T4 n600]** (move 49).
