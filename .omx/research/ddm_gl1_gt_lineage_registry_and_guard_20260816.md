---
arm: ddm_gl1
title: "GT decode lineage is now RECORDED and ASSERTED instead of lucky. Census of the whole reachable GT population (68 files / 48 distinct sha256, no cap): 16 DALI, 39 PyAV, 13 honest UNKNOWN. The headline is that pi2's cure cannot be keyed on a NAME: seven files are called gt_argmax_n600.npy, three distinct sha256, spanning BOTH lineages -- 37 instruments load the PyAV one, 12 load the DALI one. And gt_n600.npz, referenced by 320 distinct source files, is PyAV. Guard is fail-closed with a 6/6 positive control including the reconstructed pi2 defect."
utc: 2026-08-16
parent: ".omx/research/ddm_pi2_pose_axis_attribution_20260816.md"
sister: ".omx/research/ddm_pr130_reproduce_20260809/FX4_GT_LINEAGE.md"
axis: "[macOS-CPU advisory] scorer-free tensor differencing -- lineage classification is NEVER a score"
research_only: true
score_claim: false
promotion_eligible: false
promotable: false
pointer_moved: false
own_vehicle_frontier: "hv1 S 0.15959729295498598 @ archive 80d9c8c6 [contest-CUDA T4 n600] -- UNMOVED by this unit"
verdict_scope_default: "FORMULATION for the lineage-split mechanism and the content-addressing law (properties of upstream's two decode paths and of our own cache population); INSTANCE for every per-artifact sha256 classification"
tokens: "[no-triality] [p0-ledger-ok]"
---

# ddm_gl1 — GT lineage, made structural

STORES CONSULTED: parent `ddm_pi2_pose_axis_attribution_20260816.md` (§0 the rule, §4 the seg leg,
§6 item 8 this arm's charter, §9 item 2) · `ddm_pr130_reproduce_20260809/FX4_GT_LINEAGE.md` +
`FX4_GT_PROVENANCE_MANIFEST.json` + `fx4_gt_provenance_guard.sh` (the PRIOR ART, see §1) ·
`experiments/modal_dali_av_gt_cache_diff.py` (job #906, the rulers) · CLAUDE.md "Apples-to-apples
evidence discipline", "Gate consolidation discipline", "Bugs must be permanently fixed AND
self-protected against", `docs/operating_manual_craft_handoff.md` §4/§6.

## ANSWER FIRST

**Ground-truth decode lineage is now content-addressed, recorded, and asserted fail-closed.**
`src/tac/gt_lineage.py` + `src/tac/gt_lineage_registry.json` hold the lineage of every reachable
GT artifact keyed by **sha256**; `assert_gt_lineage` refuses a mismatch, refuses an unrecorded
artifact, and `assert_single_lineage` refuses an instrument whose GT sources span two lineages --
the ddm_pi2 defect written as a predicate. Positive control: **6/6, rc=0**, including the
reconstructed pi2 configuration.

**The finding that changes pi2's rule.** pi2 §0.3 says *"Keep using the cached
`gt_argmax_n600.npy` -- MEASURED DALI lineage."* That is true of the file pi2 measured. It is not
true of the name:

| filename | sha256 | lineage | instruments loading it |
|---|---|---|---:|
| `gt_argmax_n600.npy` | `91d3ff11a904…` | **DALI_NVDEC** (3 sites from ruler) | **12** |
| `gt_argmax_n600.npy` | `b74a14b226a5…` | **PYAV_YUV420_TO_RGB** (2 sites from AV ruler) | **37** |
| `gt_argmax_n600.npy` | `fee51ccfb9c1…` | **PYAV_YUV420_TO_RGB** | 0 |

Seven files carry that one name across three sha256 and **both** lineages. The DALI one lives at
`ddm_qs3_20260813/retained/inputs/`; the PyAV one at `ddm_pu2_20260803/argmax_cache/` -- and the
PyAV one is the hardcoded default in 37 instruments. A name-keyed rule is satisfiable by the wrong
bytes, which is why the registry is keyed by content. `gt_first6_n600.npy` collides the same way.

**And the most-used GT cache in the repo is PyAV.** `gt_n600.npz` (the MLX fleet cache),
referenced by **320 distinct source files**, measures 20,671 differing sites from the DALI ruler
and **2** from the AV ruler. Its subset siblings (`gt_n96`, `gt_strided_n200`,
`gt_heldout_n400`) are the same family.

**Pointer UNMOVED:** hv1 **S 0.15959729295498598** @ archive `80d9c8c6` `[contest-CUDA T4 n600]`.
This unit built apparatus. It did not lower the score.

## §1 PREMISE CORRECTION — prior art exists, and my charter did not know about it

My charter implied no lineage surface existed. **That is wrong, and I am recording it at source.**

`ddm_fx4` (2026-08-09) already landed `FX4_GT_PROVENANCE_MANIFEST.json`
(schema `ddm_fx4_gt_provenance_manifest_v1`) **and** a fail-closed guard
`fx4_gt_provenance_guard.sh` that returns typed `REFUSE` rc=42 on a cross-axis comparison. Both
are committed. I did not learn this from the charter; I found it because pi2 cited FX4 in its
STORES CONSULTED and I read it before building.

What FX4 covers, and why it does not close pi2's gap:

1. **Scope.** FX4's `target_axes` holds exactly three entries -- `retained_semantic_av_like_8248a60d`,
   `retained_official_dali_382d7dfe`, `strict_fresh_dali_dynamic`. All three are **PR130 replay**
   caches. Neither `gt_argmax_n600.npy` nor the `gt_cache_dali.pt` / `gt_cache_av.pt` rulers our
   live advisory instrument actually reads appear in it.
2. **Granularity.** FX4's guard compares two **PR130 replay legs** (`semantic` vs `carrier`) by
   name. A live instrument such as `ddm_lr2_realization_ladder.py` has no "leg", so the guard has
   nothing to look up for it.
3. **FX4 predicted this.** Its own §6 residual 4, `MEDIUM / FORMULATION`: *"The immutable intake
   does not call the sidecar guard … an unrelated ad hoc comparator can ignore it."* Its named
   falsifier is *"a maintained comparison harness that makes the manifest/guard mandatory."*

So this unit is **not a twin of FX4; it is FX4's named falsifier, generalized off the PR130 replay
and onto the live instrument population.** I deliberately reused FX4's vocabulary --
`decoder_classification`, `historical_producer_command_status`, `claim_boundary` -- so the two
surfaces read as one lineage. FX4's manifest remains authoritative for PR130 replay legs; nothing
here supersedes it.

**Charter correction #2.** The charter said `gt_argmax_n600.npy` is "authority-grade by luck".
Measured: **one** `gt_argmax_n600.npy` is authority-grade; **two** are not, and the non-authority
one has 3x the consumers. The luck was narrower than stated.

## §2 THE RULERS, and why the classification is trustworthy

Modal job #906 (`experiments/modal_dali_av_gt_cache_diff.py`, 2026-08-09) ran PR130's own
`build_gt_cache_official.py` **twice in one container on one Tesla T4** -- `--dataset av`, then
`--dataset dali`. Scorer, host, driver and clock fixed; only the decode path differs. Their
lineage is therefore **PRODUCER_DECLARED** (selected by argv), not inferred. Everything else in
the registry is classified by content against those two.

`gt_cache_dali.pt` sha `a91d9825…` (117,980,732 B) · `gt_cache_av.pt` sha `837b5852…`
(117,980,720 B), both on VertigoDataTier, **read-only** (893 MiB free; nothing was written there).

**Independent cross-validation against pi2.** My classifier reproduces pi2 §4 exactly, from a
separate implementation:

| artifact | vs DALI ruler | vs AV ruler | pi2 §4 |
|---|---:|---:|---|
| qs3 `gt_argmax_n600.npy` | **3** / 117,964,800 | 20,672 | "3 (2.54e-08)" and "20,672" — **exact match** |
| pu2 `gt_argmax_n600.npy` | 20,671 | **2** | consistent with pi2's T4-AV-vs-T4-DALI 20,671 |

**Classification is relative, never absolute.** "Equals the DALI ruler" would be the wrong test:
FX4 measured **1,644** seg sites of within-DALI drift between two DALI builds. So an artifact is
labelled by the **margin** between its two ruler distances (decisive factor 10x, chosen below the
observed 12.6x DALI-vs-AV separation so a legitimately drifted DALI cache is not refused), and the
margin travels with every label.

## §3 THE POPULATION — measured, with the denominator stated

Enumeration is code-driven plus a bounded disk sweep. **No cap was applied: 68 of 68 classified.**

| denominator | count |
|---|---:|
| GT path literals in our own `.py` instruments (test fixtures excluded) | 84 |
| files found by the bounded disk sweep (23.2 GiB) | 66 |
| **existing candidate files classified** | **68** |
| distinct sha256 among them | 48 |
| capped out | **0** |

The literal count reads 84 and not 99 because test fixtures are excluded: a test that writes
`gt_argmax_n600.npy` into `tmp_path` names a GT file no instrument ever loads. My first pass
counted 99, including 16 such fixtures — one of them from this arm's own test suite. Excluding
them does not change the classified population (68), because fixture paths do not exist on disk.

| lineage | files | distinct sha256 |
|---|---:|---:|
| `PYAV_YUV420_TO_RGB` | 39 | 30 |
| `DALI_NVDEC` | 16 | 6 |
| `UNKNOWN_AMBIGUOUS` | 7 | 6 |
| `UNKNOWN_UNCOMPARABLE` | 6 | 6 |
| **resolved** | **55 / 68 = 80.9%** | |

**Both lineages are live in the population simultaneously.** That is the state pi2's §0 rule
cannot express, because the rule names a file and the population contains two files with that name.

### The seg-axis instrument split

Re-derived precisely (a file counts only if it BOTH names `gt_argmax_n600` AND calls a loader --
my first pass counted directory mentions and over-reported by 5, including
`src/tac/witness_control/force_class_edge_ledger.py`, which only cites the path in an evidence
string):

| group | instruments |
|---|---:|
| load the **PyAV** argmax (`b74a14b2`, via `ddm_pu2_20260803/argmax_cache/`) | **37** |
| load the **DALI** argmax (`91d3ff11`, via qs3 / js1b / js1c) | **12** |
| load **both** (a pi2-shaped split inside one file) | **0** |

**Read this carefully, because the honest shape is not "37 broken instruments."** Each group is
internally coherent: an instrument that scores its own `cx1_argmax_n600.npy` render against the
PyAV GT is making a self-consistent **contest-CPU-axis** measurement. pi2 §5 established exactly
that -- both contest-CPU axes are reproducible from local files. The two real hazards are:

1. **Cross-group comparison.** Any comparison between a row from the 37 and a row from the 12 is a
   cross-axis comparison -- FX4's `REFUSE` condition -- and nothing currently detects it.
2. **Mislabelled authority tracking.** A d_seg from the 37 quoted as tracking the contest-CUDA
   authority carries pi2's measured **1.4425x** factor. pi2 §0.3 actively invites this, because it
   certifies "the cached `gt_argmax_n600.npy`" by name.

The split pi2 found was **within** one instrument (DALI seg cache + fresh PyAV pose decode). The
split measured here is **between** instrument groups on the seg axis. Different shape, same root.

## §4 THE CURE

**(A) The fix — lineage is recorded.** `src/tac/gt_lineage_registry.json`
(`ddm_gl1_gt_lineage_registry_v1`), 48 entries keyed by sha256, produced by
`experiments/ddm_gl1_gt_lineage_census.py` from measurement and never hand-typed. Every entry
carries lineage, evidence grade, the measurement that established it, all known paths, all known
basenames, and a `claim_boundary` stating how far the label may be pushed. Evidence ladder:
`PRODUCER_DECLARED` > `EMPIRICAL_EXACT_MATCH` > `EMPIRICAL_EXACT_FRAME_SUBSET` >
`EMPIRICAL_MAJORITY_FRAME_SUBSET` > `EMPIRICAL_NEAREST_RULER_*` > `NONE`. Only the two #906 rulers
hold `PRODUCER_DECLARED`; everything else is honestly labelled empirical.

**(B) The guard — instruments declare, and it fails closed.** `src/tac/gt_lineage.py`:

- `assert_gt_lineage(path, required=…)` -- refuses on mismatch, on an unregistered sha256, and on
  a registered-but-unresolved lineage. **Unknown never reads as pass**, because "unknown" is
  indistinguishable from the state pi2 found.
- `assert_single_lineage(sources, instrument=…)` -- the span predicate. Sources may be files **or
  runtime decoders**, because pi2's bug lived in the half that had no file at all: a fresh
  `frame_utils.yuv420_to_rgb` decode. A file-only registry would have been blind to it.
- `basename_lineage_collisions()` -- the standing measurement that any surviving name-keyed rule
  is unsafe.

**Why this is a runtime guard and not a new STRICT preflight gate.** `tools/claim_catalog_number.py
peek` returns **408**. CLAUDE.md "Gate consolidation discipline" forbids claiming a new STRICT gate
past #400 without retiring or replacing one. I retired nothing, so I claimed nothing. That is a
rule-derived scope decision, not a convenience.

### The positive control — 6/6, rc=0

`experiments/ddm_gl1_gt_lineage_positive_control.py`, run against REAL registered artifacts
selected from the registry at runtime (never hardcoded):

| # | control | expected | observed |
|---|---|---|---|
| 1 | DALI artifact, instrument requires DALI | PASS | PASS |
| 2 | **planted** DALI artifact, requires PyAV | refuse | `GtLineageMismatch` |
| 3 | **planted** PyAV artifact, requires AUTHORITY | refuse | `GtLineageMismatch` |
| 4 | unregistered file (fail-closed on unknown) | refuse | `GtLineageUnknown` |
| 5 | **the reconstructed pi2 instrument**: DALI cache + fresh PyAV decode | refuse | `GtLineageSplit` |
| 6 | coherent instrument: DALI cache + DALI decode | PASS | PASS |

Controls 1 and 6 exist because a guard that refuses everything is as useless as one that refuses
nothing. 25 hermetic unit tests in `src/tac/tests/test_gt_lineage.py` cover the same contract over
synthetic registries, so they keep working when the external volumes are unmounted.

### What the gauge reads if the cure is applied and nothing else changes

The sister law: a detector must not zero out on its own cure. I rejected the obvious gauge --
*"fraction of GT loads that call `assert_gt_lineage`"* -- because applying the cure drives it to
100% whether or not a single declared lineage is correct. It measures instrumentation, not reality.

`population_split_report()` instead counts **distinct lineages present in the reachable population**
and **instruments whose sources span more than one**. Adding declarations moves both by exactly
zero; only changing what an instrument READS moves them. Today it reads
`distinct_resolved_lineage_count = 2`, `population_is_single_lineage = False`, and it will keep
reading that after this landing -- correctly, because the split is now visible but **not repaired**.
`test_population_gauge_is_unchanged_by_adding_declarations` asserts the invariance directly.

## §5 MY OWN DEFECTS — three, all of one genus, all found by attacking my own output

A fix is unreviewed new code. Attacking my first pass moved 21 artifacts out of false-UNKNOWN:

| pass | resolved | defect found |
|---|---:|---|
| v1 | 34 / 68 (50.0%) | — |
| v2 | 50 / 68 (73.5%) | **dtype.** Rulers store seg as `uint8`; several caches store identical labels as `int64`. My full-field branch compared *values* (dtype-promoting) but my subset branch hashed raw *bytes*, so int64 never matched. `gt_targets_n100.pt` is in fact **100/100 frames bit-identical to the AV ruler**. |
| v3 | **55 / 68 (80.9%)** | **bare containers** (a `(600,6)` pose table stored as a bare `Tensor` was rejected as "not a dict"); **over-strict subsets** (four MLX caches scored 399/400, 199/200, 95/96, 23/24 against AV and **0** against DALI, and I was calling that "unknown"); **unimplemented pose-subset path** (now aligned through the pair-index map the seg leg already recovers). |

A fifth instance was the denominator itself: my first enumeration counted 99 GT path literals,
16 of which were **test fixtures** naming files that exist only inside `tmp_path`. The honest
count is 84. The classified population was never affected (fixture paths do not exist on disk),
but a denominator inflated by 19% would have travelled into every later ratio.

The genus is the same each time: **I trusted a summary where the full data was available.** The
fourth instance was in my own positive control -- it checked `entry.basename`, a single field, and
therefore reported "no basename collisions found" while the registry held the
`gt_argmax_n600.npy` collision that is this memo's headline. One content can carry several names
(the DALI argmax is stored as both `gt_argmax.npy` and `gt_argmax_n600.npy`), so the registry now
carries `known_basenames` and the check is a module function, not a script-local loop.

Had I shipped v1, the debt ledger would have named 21 artifacts as unknown-lineage debt that are
in fact decisively classified -- phantom debt for the next arm to chase.

## §6 WHAT THIS UNIT DID NOT ESTABLISH

- **No score.** Every row is `[macOS-CPU advisory]`, scorer-free tensor differencing. Pointer
  unmoved. No Modal dispatch (cap at $18.62/$20), no Metal launch, no scorer forward.
- **13 artifacts remain honestly UNKNOWN.** Enumerated in §7 with the specific reason for each.
- **The 37 instruments are NOT re-pointed.** Deciding which lineage each needs is the owner's call,
  not mine; re-pointing 37 files unilaterally is the migration marathon the charter forbids and
  would break live work I cannot adjudicate. Filed as debt with owner and fire-order.
- **No production consumer calls the guard yet.** The seam is proven end-to-end on real artifacts
  by the positive control, but until an instrument declares, this is capability, not enforcement --
  the "unwired-but-built" class, named honestly rather than hidden. Debt row 1.
- **RGB frame stacks cannot be classified locally at all.** The rulers are scorer OUTPUTS (argmax +
  pose); an RGB stack can be confirmed AV-lineage by re-decoding `0.mkv`, but confirming DALI needs
  nvdec, which needs a CUDA host. This is pi2 §6's open pixel-level question, unchanged.
- **`claim_boundary` is honest about what content identity cannot do.** Even
  `EMPIRICAL_EXACT_MATCH` does not recover the original argv, package versions, or host. Content
  identifies the object; it does not reconstruct the invocation. FX4 reached the same limit.
- **The "1 unmatched frame" signature is unexplained.** Three independent subset caches each have
  exactly one frame that matches neither ruler. Consistent, and not chased.

## §7 DEBT LEDGER — every unconverted site, with owner and fire-order

**Instrument debt**

| # | work | owner | fire-condition |
|---|---|---|---|
| 1 | Have ONE production instrument call `assert_gt_lineage` / `assert_single_lineage`, so the guard is enforcement and not just capability | next arm touching any GT-loading instrument | **first such touch** — do not let this sit; unwired-but-built is P0 |
| 2 | Adjudicate the **37** PyAV-argmax instruments: each declares whether it wants the contest-CPU (PyAV) or authority (DALI) axis, then asserts it. They are internally coherent today, so this is labelling, not repair | the seg line (rc4 · ra3 · hg1 · js1) | before any of their d_seg rows is quoted as tracking contest-CUDA |
| 3 | Re-key pi2 §0.3 from the NAME `gt_argmax_n600.npy` onto sha256 `91d3ff11a904…` | pi2 successor / MAIN | **NOW** — the rule as written certifies a file that 37 instruments do not load |
| 4 | Decide the axis for `gt_n600.npz` (PyAV, **320** referencing files). This is the widest-blast-radius row in the ledger | MLX fleet owner | before an MLX-fleet d_seg/d_pose row is compared against a contest-CUDA row |

**Artifact debt — the 13 UNKNOWNs, by specific reason**

| n | artifacts | why unknown | unblock |
|---:|---|---|---|
| 5 | `gt_poses.pt`, `gt_pose_targets.pt` ×2, `gt_posenet_pose6.npy` (n16), `gt_pose_n64.npy` ×2 | pose-only tables: no exact row match to either ruler, and **no seg companion** to recover the pair-index mapping, so MSE cannot be aligned | record the pair-index mapping beside pose subsets, or brute-force the mapping |
| 1 | `gt_n600_sR.npz` | differs from BOTH rulers at ~91.2M / 118M sites (77%) — **not a GT argmax field at all** | identify what `sR` holds; likely mis-swept, and correctly refused |
| 5 | `gt_frames.pt` ×3, `gt_pairs_btchw.pt`, `pr95_gt_pairs_600.npy` | RGB frame/pair stacks, not scorer-target tables; rulers are scorer outputs | needs the pixel-level nvdec comparison (pi2 §9 item 5, CUDA host) |
| 1 | `comma2k19_gt_pose_raw.npz` | different dataset (comma2k19), not `0.mkv` | out of scope; `0.mkv` rulers cannot classify it |

**Apparatus debt**

| # | work | owner | fire-condition |
|---|---|---|---|
| 8 | Fold this registry and FX4's manifest into ONE surface. Two lineage records is the split-bank disease | whoever next extends either | when a third lineage surface is proposed — refuse it, merge instead |
| 9 | A STRICT preflight gate refusing a GT load with no lineage assertion. **Blocked**: catalog is at 408, past the #400 quota, so it requires retiring or replacing an existing gate | apparatus owner | only with a retirement candidate identified |
| 10 | Re-run the census when new GT caches land; the registry is a measurement with a date, not a constant | any arm producing a GT cache | on producing a new GT cache |

## §8 RETAINED PAYLOADS (ALWAYS KEEP THE PAYLOAD)

Root `/Volumes/APDataStore/pact/ddm_gl1_20260816/` (238 GiB free). VertigoDataTier read-only
throughout (893 MiB free) — nothing written there, per the P0 disk rule.

| artifact | what it is |
|---|---|
| `GL1_GT_LINEAGE_CENSUS.json` | the full census: 68 rows, per-artifact sha256, both ruler distances, container member shapes, code-reference map |
| `GL1_POSITIVE_CONTROL.json` | the 6 controls with expected/observed, the exemplar sha256s, the collision table, the population gauge |
| `logs/disk_sweep.txt` | the bounded disk sweep (66 files, 23.2 GiB) |
| `logs/census.log`, `census_v2.log`, `census_v3.log` | all three passes, preserved so the defect progression in §5 is auditable rather than asserted |

Committed to the repo: `src/tac/gt_lineage.py`, `src/tac/gt_lineage_registry.json`,
`src/tac/tests/test_gt_lineage.py`, `experiments/ddm_gl1_gt_lineage_census.py`,
`experiments/ddm_gl1_gt_lineage_positive_control.py`.

Consumed unmodified and custody-verified: `gt_cache_dali.pt` (`a91d9825…`), `gt_cache_av.pt`
(`837b5852…`), and the 68 census artifacts (each sha256 recorded in the census JSON).

## §9 VERDICT SCOPE

- **FORMULATION** — the lineage-split mechanism, the content-addressing law (a name is not an
  identity), and the fail-closed contract. These are properties of upstream's two decode paths and
  of our cache population, not of any archive.
- **INSTANCE** — every per-artifact sha256 classification, the 37/12 instrument counts, the 320
  reference count, and the 55/68 resolution rate. All are properties of the repo as of
  2026-08-16 and must be re-measured when caches or call sites change.
- **Coefficients are `0.mkv`-specific**, inherited from pi2 §6: the 1.4425x seg factor and the
  1.4061e-04 pose floor are properties of one clip and two decoders.

**Own-vehicle frontier: hv1 S 0.15959729295498598 @ archive `80d9c8c6` `[contest-CUDA T4 n600]` —
UNMOVED.** This unit repaired an instrument's memory. It did not lower the score.
